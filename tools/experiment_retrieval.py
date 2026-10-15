#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment_retrieval.py — эмпирическое сравнение ретривала на реальных постах канала.

Сравнивает три метода поиска по одним и тем же тестовым вопросам:
  1. dense   — bge-m3 (1024d), косинус, brute-force (numpy; качество как у любого ANN)
  2. keyword — SQLite FTS5 (дефолтный токенизатор unicode61, без русского стеммера)
  3. hybrid  — Reciprocal Rank Fusion (RRF) dense+keyword

Запуск:
  .venv/bin/python tools/experiment_retrieval.py --posts data/turboproject_posts.jsonl \
      --questions tools/test_questions.json --out research/experiments/retrieval-results.md

Результат: markdown-отчёт с топ-5 по каждому методу для каждого вопроса.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- данные

def load_posts(path: Path) -> list[dict]:
    """Семантика должна совпадать с kb_search.load_posts: ВСЕ посты (включая без
    текста), дедуп по id (последний в файле выигрывает), сортировка. Фильтр пустых
    текстов здесь давал корпус 4889 vs 4967 у kb_search — кэш эмбеддингов не
    принимался поиском никогда (расхождение найдено при перекодировке 2026-10-09)."""
    result: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        result[rec["id"]] = rec
    return [result[k] for k in sorted(result)]


def load_questions(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


# --------------------------------------------------------------------------- dense (bge-m3)

def build_dense_index(posts: list[dict], model_name: str, hf_home: str | None, batch_size: int = 16,
                      cache_path: Path | None = None, doc_prompt: str = "", text_only: bool = False):
    """Эмбеддинги корпуса с дисковым кэшем: повторные прогоны не перекодируют 30 минут."""
    import os
    import hashlib
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)

    # промпт входит в ключ: тот же корпус с другим промптом даёт другие векторы
    corpus_key = hashlib.sha256(
        (model_name + "|" + doc_prompt + "|" + "\x1e".join(p["text"] for p in posts)).encode("utf-8")
    ).hexdigest()[:16]
    if cache_path is not None:
        meta_path = cache_path.with_suffix(".meta.json")
        if cache_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("key") == corpus_key and meta.get("ids") == [p["id"] for p in posts]:
                emb = np.load(cache_path)["emb"]
                print(f"Эмбеддинги загружены из кэша: {cache_path} ({emb.shape[0]} x {emb.shape[1]})")
                return None, emb, 0.0, 0.0

    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = _load_model(model_name, text_only)
    load_s = time.time() - t0

    from kb_search import encode_text
    # тексты для эмбеддинга санитизируем (голая ссылка на картинку ломает
    # мультимодальный процессор — пост 100); hash остаётся по raw-текстам
    texts = [encode_text(p["text"]) for p in posts]
    t0 = time.time()
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=False, prompt_name=doc_prompt or None)
    encode_s = time.time() - t0
    emb = np.asarray(emb, dtype=np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, emb=emb)
        # hash обязателен: kb_search.load_dense проверяет ids И hash(текстов);
        # meta без hash считалась бы валидной при правке текста поста
        from kb_search import corpus_hash
        meta_path.write_text(json.dumps({"key": corpus_key, "model": model_name,
                                         "ids": [p["id"] for p in posts],
                                         "hash": corpus_hash({p["id"]: p for p in posts})},
                                        ensure_ascii=False), encoding="utf-8")
        print(f"Эмбеддинги сохранены в кэш: {cache_path}")
    return model, emb, load_s, encode_s


def _load_model(model_name: str, text_only: bool = False):
    """text_only: для модульных моделей (EmbeddingGemma 2) — не грузить vision/audio
    энкодеры (~3× меньше RAM); на обычные модели флага нет — грузим как есть."""
    from sentence_transformers import SentenceTransformer
    if text_only:
        return SentenceTransformer(model_name,
                                   config_kwargs={"vision_config": None, "audio_config": None})
    return SentenceTransformer(model_name)


def dense_search(query: str, model, emb: np.ndarray, posts: list[dict], top_k: int = 5,
                 query_prompt: str = "") -> list[tuple[int, float]]:
    """Возвращает (id поста, score) — позиции массива мапятся на реальные id."""
    q = np.asarray(model.encode([query], normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)[0]
    scores = emb @ q  # косинус (нормализованы)
    idx = np.argsort(-scores)[:top_k]
    return [(int(posts[int(i)]["id"]), float(scores[i])) for i in idx]


# --------------------------------------------------------------------------- keyword (FTS5)

def build_fts(posts: list[dict]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE posts USING fts5(id UNINDEXED, text)")
    conn.executemany("INSERT INTO posts (id, text) VALUES (?, ?)", [(p["id"], p["text"]) for p in posts])
    return conn


def keyword_search(conn: sqlite3.Connection, query: str, top_k: int = 5) -> list[tuple[int, float]]:
    # FTS5: префиксный режим по словам, чтобы ловить неполные совпадения
    terms = [t for t in "".join(c if c.isalnum() else " " for c in query.lower()).split() if len(t) > 2]
    if not terms:
        return []
    match = " OR ".join(f'"{t}"*' for t in terms)
    try:
        rows = conn.execute(
            "SELECT id, bm25(posts) FROM posts WHERE posts MATCH ? ORDER BY bm25(posts) LIMIT ?",
            (match, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(int(pid), float(score)) for pid, score in rows]


# --------------------------------------------------------------------------- hybrid (RRF)

def rrf_fuse(dense: list[tuple[int, float]], keyword: list[tuple[int, float]], k: int = 60, top_k: int = 5) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, (pid, _) in enumerate(dense):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (pid, _) in enumerate(keyword):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]


# --------------------------------------------------------------------------- отчёт

def snippet(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", default="data/turboproject_posts.jsonl")
    ap.add_argument("--questions", default="tools/test_questions.json")
    ap.add_argument("--out", default="research/experiments/retrieval-results.md")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--cache", default=None, help="путь кэша эмбеддингов (по умолчанию — рядом с JSONL); "
                                                 "разный для разных моделей при A/B")
    ap.add_argument("--query-prompt", default="", help="prompt_name для запросов (напр. SearchQuery у EmbeddingGemma)")
    ap.add_argument("--doc-prompt", default="", help="prompt_name для корпуса (напр. Document у EmbeddingGemma)")
    ap.add_argument("--text-only", action="store_true",
                    help="модульная модель: грузить только текстовый энкодер (EmbeddingGemma 2)")
    args = ap.parse_args(argv)

    posts = load_posts(Path(args.posts))
    questions = load_questions(args.questions)
    print(f"Постов: {len(posts)}; вопросов: {len(questions)}; модель: {args.model}")

    cache = Path(args.cache) if args.cache else Path(args.posts).with_suffix(".emb.npz")
    model, emb, load_s, encode_s = build_dense_index(posts, args.model, args.hf_home,
                                                     cache_path=cache, doc_prompt=args.doc_prompt,
                                                     text_only=args.text_only)
    if model is None:
        # кэш-попадание: модель всё равно нужна для кодирования запросов (загрузка ~8 c)
        import os
        if args.hf_home:
            os.environ.setdefault("HF_HOME", args.hf_home)
        load_s = time.time()
        model = _load_model(args.model, args.text_only)
        load_s = time.time() - load_s
    print(f"Модель загружена за {load_s:.0f} c; корпус закодирован за {encode_s:.0f} c; dims={emb.shape[1]}")

    conn = build_fts(posts)

    lines = [
        f"# Эмпирическое сравнение ретривала на постах канала",
        "",
        f"- Дата: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Корпус: {len(posts)} постов (полный текст), модель {args.model}, dims={emb.shape[1]}",
        f"- Промпты: запрос='{args.query_prompt or '—'}', документ='{args.doc_prompt or '—'}'; text_only={args.text_only}",
        f"- Загрузка модели: {load_s:.0f} c; кодирование корпуса: {encode_s:.0f} c",
        f"- Методы: dense (косинус, brute-force), keyword (FTS5 unicode61, без русского стеммера), hybrid (RRF, k=60)",
        "",
    ]

    t_dense = t_kw = 0.0
    for qi, q in enumerate(questions, 1):
        query = q["q"] if isinstance(q, dict) else q
        t0 = time.time(); dense = dense_search(query, model, emb, posts, args.top_k,
                                               query_prompt=args.query_prompt); t_dense += time.time() - t0
        t0 = time.time(); kw = keyword_search(conn, query, args.top_k); t_kw += time.time() - t0
        hyb = rrf_fuse(dense, kw, top_k=args.top_k)

        lines.append(f"## Q{qi}. {query}")
        lines.append("")
        by_id = {p["id"]: p for p in posts}
        for name, hits in (("dense", dense), ("keyword", kw), ("hybrid", hyb)):
            lines.append(f"**{name}**:")
            for pid, score in hits:
                p = by_id[pid]
                lines.append(f"- [{pid}] {p.get('date_utc', '?')[:10]} score={score:.4f} :: {snippet(p['text'])}")
            lines.append("")

    lines.append("## Времена поиска")
    lines.append("")
    lines.append(f"- dense: {t_dense / len(questions) * 1000:.1f} мс/запрос (без учёта загрузки модели)")
    lines.append(f"- keyword (FTS5): {t_kw / len(questions) * 1000:.1f} мс/запрос")
    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Результат: {out}")
    print(f"Времена: dense {t_dense / len(questions) * 1000:.1f} мс/запрос, keyword {t_kw / len(questions) * 1000:.1f} мс/запрос")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
