#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kb_search.py — гибридный поиск по постам канала: ядро базы знаний (task-8).

Слой retrieval (движок):
  1. keyword — SQLite FTS5, две колонки raw + stemmed (стемминг вендоризован,
     tools/_vendor, pure Python); запрос стеммится тем же кодом; закавыченные токены,
     AND→OR; веса bm25(1.0, 2.0, 1.0) — id UNINDEXED первым (слоты bm25 идут по колонкам,
     проверено живьём task-9 §1): raw вдвое важнее stemmed. Покрытие целей то же (8/10),
     топ-ранги и точность лучше (stemming-results.md v2).
  2. dense — bge-m3 из дискового кэша (emb.npz); при его отсутствии/устаревании —
     keyword-only с warning (dense — опциональная зависимость sentence-transformers).
  3. hybrid — RRF (k=60) + бонус за реакции (P2: min-max нормализация + 0.05 × log1p(n)/(1+log1p(n))).

Контракт для агента (универсальный):
  - человекочитаемый вывод по умолчанию; --json — машинный режим ({query, hits:[{id, url, file, ...}]});
  - в хитах НЕТ полного текста (обрезка длинных строк — известный дефект агентского чтения):
    вместо этого path до поста-файла data/posts/<канал>/<id>.md — агент читает его ЦЕЛИКОМ;
  - exit codes: 0 = есть хиты, 1 = пусто, 2 = ошибка.

Зависимости: Python 3 stdlib. Стемминг вендоризован (tools/_vendor/snowballstemmer, BSD-3-Clause)
— работает с любым python3. Опционально sentence-transformers (dense); без него keyword-режим
деградирует честно (warning на stderr).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

# --------------------------------------------------------------------------- стемминг (P1, вендоризован)

# Вендор по решению task-9 (§4.3): pip install из песочницы агентов не исполним —
# стемминг работает с любым python3 без зависимостей (tools/_vendor, BSD-3-Clause).
sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))
try:
    import snowballstemmer
    _RU = snowballstemmer.stemmer("russian")
    _EN = snowballstemmer.stemmer("english")
    _HAS_STEMMER = True
except ImportError:
    _RU = _EN = None
    _HAS_STEMMER = False


def _stem(token: str) -> str:
    """Латиница → english, кириллица/прочее → russian (паттерн grepogram)."""
    if token.isascii():
        return _EN.stemWord(token) or token
    return _RU.stemWord(token) or token


def stem_text(text: str) -> str:
    """NFKC + lower + \\w+ (ничего не выбрасываем)."""
    norm = unicodedata.normalize("NFKC", text).lower()
    parts = [_stem(t) for t in re.findall(r"\w+", norm)]
    return " ".join(p for p in parts if p)


def stem_query_terms(text: str) -> list[str]:
    """Стемы запроса тем же кодом, дедуп, порядок сохраняется."""
    norm = unicodedata.normalize("NFKC", text).lower()
    seen: set[str] = set()
    terms: list[str] = []
    for t in re.findall(r"\w+", norm):
        s = _stem(t)
        if s and s not in seen:
            seen.add(s)
            terms.append(s)
    return terms


# --------------------------------------------------------------------------- данные

def resolve_data_dir(cli_value: str | None) -> Path:
    """Приоритет: --data-dir > env TG_KB_HOME > ./data (паттерн GREPOGRAM_HOME; проектный
    дефолт — следствие песочниц агентов: запись вне workspace недоступна, проверено живьём)."""
    if cli_value:
        return Path(cli_value)
    import os
    env = (os.environ.get("TG_KB_HOME") or "").strip()
    if env:
        return Path(env)
    return Path("data")


def posts_path(channel: str, dd: Path) -> Path:
    return dd / f"{channel}_posts.jsonl"


_CHANNEL_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")


def validate_channel(channel: str) -> None:
    """Имя канала — Telegram username ([A-Za-z0-9_], до 64 символов). Имя строится
    в пути файлов (dd / f"{channel}_posts.jsonl") — без валидации аргумент вида
    "../../x" читал и писал бы вне data-dir (аудит безопасности перед публикацией)."""
    if not _CHANNEL_RE.match(channel or ""):
        raise ValueError(f"некорректное имя канала: {channel!r} — ожидается [A-Za-z0-9_], до 64 символов")


def emb_cache_path(channel: str, dd: Path) -> Path:
    return dd / f"{channel}_posts.emb.npz"


def load_posts(channel: str, dd: Path) -> dict[int, dict]:
    p = posts_path(channel, dd)
    result: dict[int, dict] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # битая строка не должна крашить поиск (краш → exit 1 = «пусто»
                # маскирует ошибку под отсутствие хитов — фикс task-9 §3.1)
                print(f"warning: битая JSONL-строка пропущена: {line[:80]}…", file=sys.stderr)
                continue
            result[rec["id"]] = rec
    return result


def corpus_hash(posts: dict[int, dict]) -> str:
    h = hashlib.sha256()
    for k in sorted(posts):
        h.update(str(k).encode())
        h.update(b"\x1e")
        h.update((posts[k].get("text") or "").encode("utf-8"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- FTS-индекс (persist, grepogram-паттерн)

def fts_path(channel: str, dd: Path) -> Path:
    return dd / f"{channel}_fts.db"


def build_fts_index(channel: str, dd: Path, posts: dict[int, dict]) -> tuple[sqlite3.Connection, str]:
    """SQLite-файл с raw+stemmed колонками; перестраивается только при изменении корпуса
    (ключ = hash(тексты)). Без стеммера stemmed = текст без w-нормализации (деградация честно)."""
    p = fts_path(channel, dd)
    key = corpus_hash(posts)
    meta = p.with_suffix(".meta.json")
    if p.exists() and meta.exists():
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            # переиспользуем, если индекс не хуже того, что построили бы сейчас:
            # стеммированный индекс ценен и для python без стеммера (raw-колонка работает);
            # а вот стеммер готов, а индекс без стемминга — перестраиваем (апгрейд)
            if m.get("key") == key and (not _HAS_STEMMER or m.get("stemmer", False)):
                conn = sqlite3.connect(str(p))
                conn.execute("PRAGMA journal_mode=WAL")
                return conn, "cached"
        except (json.JSONDecodeError, sqlite3.DatabaseError):
            pass  # повреждён — перестроим

    if p.exists():
        p.unlink()
    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        'CREATE VIRTUAL TABLE posts USING fts5(id UNINDEXED, raw, stemmed, '
        'tokenize="unicode61 remove_diacritics 2")'
    )
    rows = []
    for k in sorted(posts):
        text = posts[k].get("text") or ""
        rows.append((k, text, stem_text(text) if _HAS_STEMMER else " ".join(re.findall(r"\w+", text.lower()))))
    conn.executemany("INSERT INTO posts (id, raw, stemmed) VALUES (?, ?, ?)", rows)
    conn.commit()
    meta.write_text(json.dumps({"key": key, "stemmer": _HAS_STEMMER}, ensure_ascii=False), encoding="utf-8")
    return conn, "built"


def keyword_search(conn: sqlite3.Connection, query: str, top_k: int = 5) -> list[tuple[int, float]]:
    """AND-запрос по стемам первым; пусто — OR-фолбэк (паттерн grepogram).
    Глубина выборки = max(top_k, 40) — кандидат-набор для фьюжа/фильтров,
    не зависит от k (фикс task-9 §3.6). Возвращает -bm25: выше = лучше."""
    terms = stem_query_terms(query) if _HAS_STEMMER else []
    if not terms:
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    if not terms:
        return []
    quoted = [f'"{t}"' for t in terms]
    depth = max(top_k, 40)

    def run(match: str, limit: int) -> list[tuple[int, float]]:
        try:
            # ВЕСА: слоты bm25 идут ПО КОЛОНКАМ таблицы, включая UNINDEXED id (проверено
            # живьём task-9 §1) → (id=1.0, raw=2.0, stemmed=1.0): raw вдвое важнее stemmed
            rows = conn.execute(
                "SELECT id, bm25(posts, 1.0, 2.0, 1.0) FROM posts WHERE posts MATCH ? ORDER BY 2 LIMIT ?",
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        # -bm25: выше = лучше (единая конвенция с dense/RRF и корректная min-max нормализация бонуса)
        return [(int(pid), -float(score)) for pid, score in rows]

    hits = run(" AND ".join(quoted), depth)
    if hits:
        return hits
    hits = run(" OR ".join(quoted), depth)
    return hits[:depth]


# --------------------------------------------------------------------------- dense (опционально)

def load_dense(channel: str, dd: Path, posts: dict[int, dict]):
    """Кэш emb.npz + модель. Возвращает (model, emb, ids_in_order) или (None, None, None) —
    keyword-only режим с warning. Кэш устарел при несовпадении id ИЛИ hash(текстов)
    (правка поста сохраняет id, но текст меняется — вектор молча старый; фикс task-9 §3.3)."""
    import os
    cache = emb_cache_path(channel, dd)
    meta_p = cache.with_suffix(".meta.json")
    if not cache.exists() or not meta_p.exists():
        print(f"warning: кэш эмбеддингов не найден ({cache}) — keyword-only режим", file=sys.stderr)
        return None, None, None
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("warning: meta эмбеддингов повреждена — keyword-only режим", file=sys.stderr)
        return None, None, None
    ids = meta.get("ids") or []
    want_hash = corpus_hash(posts)
    stale_ids = ids != sorted(posts)
    stale_texts = meta.get("hash") is not None and meta.get("hash") != want_hash
    if stale_ids or stale_texts:
        why = f"{len(ids)} vs {len(posts)} постов" if stale_ids else "правка постов (hash текстов)"
        print(f"warning: кэш эмбеддингов устарел ({why}) — keyword-only; "
              f"перекодируй: experiment_retrieval.py", file=sys.stderr)
        return None, None, None
    # существующий кэш моделей в корне workspace (~4.3 ГБ) — не плодим дубликат в data/ (task-9 §3.4)
    os.environ.setdefault("HF_HOME", str(dd.parent / ".hf-cache"))
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("warning: sentence-transformers не установлен — keyword-only режим", file=sys.stderr)
        return None, None, None
    emb = np.load(cache)["emb"]
    model = SentenceTransformer(meta.get("model") or "BAAI/bge-m3")
    return model, emb, ids


def dense_search(query: str, model, emb, ids: list[int], top_k: int = 5) -> list[tuple[int, float]]:
    q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    import numpy as np
    q = np.asarray(q, dtype=np.float32)[0]
    scores = emb @ q
    idx = np.argsort(-scores)[:top_k]
    return [(int(ids[int(i)]), float(scores[i])) for i in idx]


# --------------------------------------------------------------------------- RRF + бонус реакций (P2)

def rrf_fuse(dense: list[tuple[int, float]], keyword: list[tuple[int, float]],
             k: int = 60, top_k: int = 5) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, (pid, _) in enumerate(dense):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (pid, _) in enumerate(keyword):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]


def reaction_bonus(candidates: list[tuple[int, float]], posts: dict[int, dict],
                   weight: float = 0.05) -> list[tuple[int, float]]:
    """P2 из grepogram: min-max нормализация fused-оценок в [0,1], затем
    weight × log1p(n)/(1+log1p(n)), n = суммарное число реакций поста.
    Асимптота бонуса = weight (0.05); живьём 0.0401 при 56 реакциях (task-9 §3.5).
    ЧЕСТНО: бонус может переставлять соседние хиты (макс разница бонусов 0.0295
    против соседнего зазора RRF ~0.0155) — это дизайн «согласие чата поднимает пост»,
    не дефект; нормализация по кандидат-набору (глубина max(k, 40)) — ранжирование
    слабо зависит от k. <2 кандидатов — бонуса нет. Возвращает (pid, итоговая оценка)."""
    if len(candidates) < 2:
        return candidates
    values = [s for _, s in candidates]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1e-6

    def bonus(n: int) -> float:
        ln = math.log1p(max(0, n))
        return weight * ln / (1.0 + ln)

    out = []
    for pid, s in candidates:
        norm = (s - lo) / span
        n = sum(posts.get(pid, {}).get("reactions", {}).values())
        out.append((pid, norm + bonus(n)))
    out.sort(key=lambda kv: -kv[1])
    return out


# Дедуп (task-9 §3.7): для per-post корпуса «1 пост = 1 id» дубликат в фьюже невозможен
# (одинаковый id мерджится) — N/A. Станется ОБЯЗАТЕЛЬНЫМ при переходе на unit-индексацию
# (thread/window-юниты пересекаются — см. REPORT.md §12.4 P3): дедуп по пересечению
# msg_ids ≥ 0.5, как в grepogram search.py.


# --------------------------------------------------------------------------- вывод

def by_date(posts: dict[int, dict], pid: int) -> str:
    """YYYY-MM-DD поста для фильтров --since/--until; None если даты нет
    (явная обработка вместо мёртвых сантинелов — фикс task-9 §3.2/табл. 11)."""
    raw = (posts.get(pid, {}).get("date_utc") or "").strip()
    return raw[:10] if raw else None


def snippet(text: str, limit: int = 140) -> str:
    text = " ".join((text or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Гибридный поиск по постам канала (ядро базы знаний).")
    ap.add_argument("channel", help="username канала без @, например turboproject")
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5, help="сколько хитов (по умолчанию 5)")
    ap.add_argument("--mode", choices=["hybrid", "dense", "keyword"], default="hybrid")
    ap.add_argument("--json", action="store_true", help="машинный JSON-режим для агентов")
    ap.add_argument("--data-dir", default=None, help="папка данных; приоритет: --data-dir > env TG_KB_HOME > ./data")
    ap.add_argument("--since", help="YYYY-MM-DD: только посты не старше (фильтр по дате после поиска)")
    ap.add_argument("--until", help="YYYY-MM-DD: только посты не новее")
    ap.add_argument("--doctor", action="store_true", help="состояние базы: покрытие, индекс, кэши")
    ap.add_argument("--paths", action="store_true", help="показать resolved-пути и выйти")
    args = ap.parse_args(argv)

    validate_channel(args.channel)
    dd = resolve_data_dir(args.data_dir)
    if args.paths:
        import os
        src = "cli:--data-dir" if args.data_dir else ("env:TG_KB_HOME" if (os.environ.get("TG_KB_HOME") or "").strip() else "default:./data")
        print(json.dumps({
            "data_dir": str(dd), "source": src,
            "posts": str(posts_path(args.channel, dd)),
            "fts": str(fts_path(args.channel, dd)),
            "emb": str(emb_cache_path(args.channel, dd)),
            "post_files": str(dd / "posts" / args.channel),
            "hf_cache": str(dd.parent / ".hf-cache"),
        }, ensure_ascii=False, indent=2))
        return 0
    posts = load_posts(args.channel, dd)
    if not posts:
        print(f"Нет сохранённых постов — сначала: tg_channel_parser.py {args.channel} update", file=sys.stderr)
        return 2

    if args.doctor:
        emb_meta = {}
        meta_p = emb_cache_path(args.channel, dd).with_suffix(".meta.json")
        if meta_p.exists():
            try:
                emb_meta = json.loads(meta_p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                emb_meta = {"error": "повреждена"}
        post_files = dd / "posts" / args.channel
        n_files = len(list(post_files.glob("*.md"))) if post_files.exists() else 0
        corpus_h = corpus_hash(posts)
        emb_state = ("нет" if not emb_meta else
                     ("свежий" if emb_meta.get("ids") == sorted(posts) and (emb_meta.get("hash") is None or emb_meta.get("hash") == corpus_h)
                      else f"устарел ({len(emb_meta.get('ids') or [])} vs {len(posts)} или правка текстов)"))
        fts_meta = {}
        fts_meta_p = fts_path(args.channel, dd).with_suffix(".meta.json")
        if fts_meta_p.exists():
            try:
                fts_meta = json.loads(fts_meta_p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                fts_meta = {"error": "повреждена"}
        fts_state = ("нет" if not fts_meta else
                     ("устарел (пересоберётся при следующем поиске)" if fts_meta.get("key") != corpus_h else
                      (f"свежий, stemmer={fts_meta.get('stemmer', False)}" + (" (в процессе нет — запросы по raw)" if fts_meta.get("stemmer") and not _HAS_STEMMER else ""))))
        print(json.dumps({
            "channel": args.channel,
            "posts": len(posts),
            "min_max_id": [min(posts), max(posts)],
            "posts_without_text": sum(1 for p in posts.values() if not (p.get("text") or "").strip()),
            "fts": {"file": str(fts_path(args.channel, dd)), "index": fts_state,
                    "stemmer": ("вендоризован (tools/_vendor)" if _HAS_STEMMER else "НЕТ — стемминг недоступен"),
                    "hint": None if _HAS_STEMMER else "стеммер не найден: ожидается tools/_vendor/snowballstemmer/ или pip install snowballstemmer"},
            "emb": {"model": emb_meta.get("model"), "state": emb_state},
            "post_files": {"dir": str(post_files), "count": n_files,
                           "hint": f"rerender: python3 tools/tg_channel_parser.py {args.channel} rerender" if n_files < len(posts) else "ok"},
        }, ensure_ascii=False, indent=2))
        return 0

    t0 = time.time()
    fts, fts_status = build_fts_index(args.channel, dd, posts)
    fts_s = time.time() - t0

    model = emb = ids = None
    if args.mode in ("hybrid", "dense"):
        model, emb, ids = load_dense(args.channel, dd, posts)
        if model is None and args.mode == "dense":
            print("dense недоступен (см. warning) — выход", file=sys.stderr)
            return 2

    t0 = time.time()
    kw = keyword_search(fts, args.query, args.k) if args.mode in ("hybrid", "keyword") else []
    kw_s = time.time() - t0

    dn = []
    dn_s = 0.0
    if args.mode == "dense" or (args.mode == "hybrid" and model is not None):
        t0 = time.time()
        dn = dense_search(args.query, model, emb, ids, max(args.k, 40))
        dn_s = time.time() - t0

    if args.mode == "hybrid":
        if dn and kw:
            hits = rrf_fuse(dn, kw, top_k=max(args.k, 40))
        else:
            hits = dn or kw
        hits = reaction_bonus(hits, posts)
    elif args.mode == "dense":
        hits = dn
    else:
        hits = kw

    # over-fetch → фильтр по дате → trim (фикс task-9 §2: фильтр ПОСЛЕ усечения до k
    # терял свежие хиты — проверено живьём: 2 из 5 видимых при 5 существующих)
    if args.since:
        hits = [(pid, s) for pid, s in hits if (d := by_date(posts, pid)) is not None and d >= args.since]
    if args.until:
        hits = [(pid, s) for pid, s in hits if (d := by_date(posts, pid)) is not None and d <= args.until]
    hits = hits[:args.k]
    if not hits:
        if args.json:
            print(json.dumps({
                "query": args.query, "hits": [], "mode": args.mode, "error": "нет хитов",
                "hint": "попробуй 2–3 варианта запроса (русский + английский, термин + концепт, синонимы); "
                        "для точных токенов (id, названия, цены) — --mode keyword; "
                        "проверь фильтры --since/--until; обнови базу: python3 tools/tg_channel_parser.py "
                        f"{args.channel} update",
            }, ensure_ascii=False))
        else:
            print("Ничего не найдено. Подсказка: попробуй 2–3 варианта запроса (ru/en, синонимы); "
                  "для точных токенов — --mode keyword.")
        return 1

    by_id = posts
    if args.json:
        out = {
            "query": args.query,
            "mode": args.mode,
            "hits": [
                {
                    "id": pid,
                    "date": by_id[pid].get("date_utc", "")[:10],
                    "url": f"https://t.me/{args.channel}/{pid}",
                    "file": str(dd / "posts" / args.channel / f"{pid}.md"),
                    "score": round(score, 5),
                    "reactions": sum(by_id[pid].get("reactions", {}).values()),
                    "snippet": snippet(by_id[pid].get("text")),
                }
                for pid, score in hits
            ],
            "times_ms": {"fts": round(fts_s * 1000, 1), "keyword": round(kw_s * 1000, 1), "dense": round(dn_s * 1000, 1)},
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"Поиск «{args.query}» — {args.mode}, {len(hits)} хитов:\n")
        for pid, score in hits:
            r = by_id[pid]
            n_react = sum(r.get("reactions", {}).values())
            print(f"[{pid}] {r.get('date_utc', '')[:10]} score={score:.4f} reactions={n_react}")
            print(f"  https://t.me/{args.channel}/{pid} | файл: {dd / 'posts' / args.channel / f'{pid}.md'}")
            print(f"  {snippet(r.get('text'))}\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # неожиданный краш НЕ должен маскироваться под exit 1 («пусто»)
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
