#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment_stemming.py — измеряет ценность P1 (стемминг) на наших данных.

Точный перенос паттерна grepogram (research/experiments/grepogram-implementation-study.md):
  - FTS5 остаётся на стандартном токенайзере unicode61 remove_diacritics 2 (БЕЗ кастомного токенайзера);
  - две индексируемые колонки: raw + stemmed (стемминг Снежным стеммером, пакет snowballstemmer —
    чистый Python без C-расширений);
  - запрос стеммится ТЕМ ЖЕ кодом перед MATCH (консистентность сторон — гарантия);
  - каждое слово в кавычках (защита MATCH от `-`/`:`), дедуп стемов, AND первым, OR-фолбэк;
  - вес bm25(2.0, 1.0) — raw вдвое важнее stemmed;
  - бонус: Snowball сворачивает ё→е, unicode61 — нет (запрос и stemmed согласованы).

Сравнение с базой (retrieval-results.md, оценка лидом):
  keyword FTS5 без стемминга: 5/10 отлично (Q1 провален — частотные слова);
  hybrid RRF: 9/10.

Запуск:
  .venv/bin/python tools/experiment_stemming.py \
      --posts data/turboproject_posts.jsonl --questions tools/test_questions.json \
      --out research/experiments/stemming-results.md
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
import unicodedata
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

# Вендор (task-9 §4.3): стеммер работает с любым python3 без pip install
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))
import snowballstemmer

# --------------------------------------------------------------------------- стемминг (P1)

_RU = snowballstemmer.stemmer("russian")
_EN = snowballstemmer.stemmer("english")


@lru_cache(maxsize=65536)
def _stem(token: str) -> str:
    """Стемминг по скрипту: латиница → english, кириллица/прочее → russian (как в grepogram)."""
    if token.isascii():
        return _EN.stemWord(token) or token
    return _RU.stemWord(token) or token


def stem_text(text: str) -> str:
    """NFKC + lower + \\w+ (ничего не выбрасываем — цифры и короткие токены хранятся)."""
    norm = unicodedata.normalize("NFKC", text).lower()
    return " ".join(_stem(t) for t in re.findall(r"\w+", norm) if _stem(t))


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


# --------------------------------------------------------------------------- FTS5 raw+stemmed

def build_fts2(posts: list[dict]) -> sqlite3.Connection:
    """Две колонки: raw (текст как есть, токенайзер unicode61) + stemmed (стемы тем же кодом)."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        'CREATE VIRTUAL TABLE posts USING fts5(id UNINDEXED, raw, stemmed, '
        'tokenize="unicode61 remove_diacritics 2")'
    )
    rows = []
    for p in posts:
        text = p["text"]
        rows.append((p["id"], text, stem_text(text)))
    conn.executemany("INSERT INTO posts (id, raw, stemmed) VALUES (?, ?, ?)", rows)
    return conn


def keyword_search2(conn: sqlite3.Connection, query: str, top_k: int = 5) -> list[tuple[int, float]]:
    """AND-запрос по стемам первым; если пусто — OR-фолбэк заполняет остаток (паттерн grepogram)."""
    terms = stem_query_terms(query)
    if not terms:
        return []
    quoted = [f'"{t}"' for t in terms]

    def run(match: str, limit: int) -> list[tuple[int, float]]:
        try:
            rows = conn.execute(
                # ВЕСА: слоты bm25 идут по колонкам (id UNINDEXED первым — проверено
                # живьём task-9 §1) → (id=1.0, raw=2.0, stemmed=1.0)
                "SELECT id, bm25(posts, 1.0, 2.0, 1.0) FROM posts WHERE posts MATCH ? ORDER BY 2 LIMIT ?",
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        # -bm25: выше = лучше (конвенция одна с продом kb_search.py — фикс task-9 §3.8)
        return [(int(pid), -float(score)) for pid, score in rows]

    # 1) строго: все стемы запроса должны присутствовать
    hits = run(" AND ".join(quoted), top_k)
    if hits:
        return hits
    # 2) фолбэк: любой стем, добираем до top_k без дублей
    hits = run(" OR ".join(quoted), top_k * 2)
    if len(hits) <= top_k:
        return hits
    return hits[:top_k]


# --------------------------------------------------------------------------- dense (из кэша) и RRF

def load_posts(path: Path) -> list[dict]:
    posts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if (rec.get("text") or "").strip():
            posts.append(rec)
    posts.sort(key=lambda p: p["id"])
    return posts


def load_questions(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def dense_search(query: str, model, emb, posts, top_k=5):
    q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    import numpy as np
    q = np.asarray(q, dtype=np.float32)[0]
    scores = emb @ q
    idx = np.argsort(-scores)[:top_k]
    return [(int(posts[int(i)]["id"]), float(scores[i])) for i in idx]


def rrf_fuse(dense, keyword, k=60, top_k=5):
    scores: dict[int, float] = {}
    for rank, (pid, _) in enumerate(dense):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (pid, _) in enumerate(keyword):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]


def snippet(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


# --------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", default="data/turboproject_posts.jsonl")
    ap.add_argument("--questions", default="tools/test_questions.json")
    ap.add_argument("--out", default="research/experiments/stemming-results.md")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args(argv)

    posts = load_posts(Path(args.posts))
    questions = load_questions(args.questions)
    print(f"Постов: {len(posts)}; вопросов: {len(questions)}")

    # стемминг корпуса (время)
    t0 = time.time()
    conn = build_fts2(posts)
    stem_s = time.time() - t0
    n_stems = conn.execute("SELECT count(*) FROM posts WHERE stemmed != ''").fetchone()[0]
    print(f"FTS raw+stemmed построен за {stem_s:.1f} c (стемминг {n_stems} постов)")

    # dense из кэша
    import os
    import numpy as np
    if args.hf_home:
        os.environ.setdefault("HF_HOME", args.hf_home)
    cache = Path(args.posts).with_suffix(".emb.npz")
    meta_p = cache.with_suffix(".meta.json")
    if not cache.exists() or not meta_p.exists():
        print("Кэш эмбеддингов не найден — dense-часть будет пропущена (запусти experiment_retrieval.py)")
        emb = None
    else:
        emb = np.load(cache)["emb"]
    if emb is not None:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        model = SentenceTransformer(args.model)
        print(f"Модель загружена за {time.time() - t0:.0f} c (для кодирования запросов)")

    lines = [
        "# Измерение ценности P1 (стемминг raw/stemmed) на наших данных",
        "",
        f"- Дата: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Корпус: {len(posts)} постов; стемминг корпуса: {stem_s:.1f} c",
        f"- Паттерн: grepogram P1 — unicode61 remove_diacritics 2 + две колонки raw/stemmed "
        f"(snowballstemmer, чистый Python); запрос стеммится тем же кодом; AND→OR; bm25(2.0, 1.0)",
        f"- База для сравнения: keyword FTS5 без стемминга 5/10, hybrid RRF 9/10 (retrieval-results.md)",
        "",
    ]

    t_kw2 = t_dense = 0.0
    for qi, q in enumerate(questions, 1):
        query = q["q"] if isinstance(q, dict) else q
        t0 = time.time(); kw2 = keyword_search2(conn, query, args.top_k); t_kw2 += time.time() - t0
        if emb is not None:
            t0 = time.time(); dense = dense_search(query, model, emb, posts, args.top_k); t_dense += time.time() - t0
            hyb = rrf_fuse(dense, kw2, top_k=args.top_k)

        lines.append(f"## Q{qi}. {query}")
        lines.append("")
        by_id = {p["id"]: p for p in posts}
        lines.append(f"**keyword+стемминг** ({len(kw2)} хитов):")
        for pid, score in kw2:
            p = by_id[pid]
            lines.append(f"- [{pid}] {p.get('date_utc', '?')[:10]} score={score:.4f} :: {snippet(p['text'])}")
        if emb is not None:
            lines.append("**hybrid (dense + stemmed keyword)**:")
            for pid, score in hyb:
                p = by_id[pid]
                lines.append(f"- [{pid}] {p.get('date_utc', '?')[:10]} rrf={score:.5f} :: {snippet(p['text'])}")
        lines.append("")

    lines.append("## Времена")
    lines.append("")
    lines.append(f"- keyword+стемминг: {t_kw2 / len(questions) * 1000:.1f} мс/запрос")
    if emb is not None:
        lines.append(f"- dense: {t_dense / len(questions) * 1000:.1f} мс/запрос")
    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Результат: {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
