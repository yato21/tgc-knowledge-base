#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py — замеры скорости tg-kb по этапам выполнения.

Этапы:
  1. Парсинг (сеть): fetch+parse одной страницы t.me/s (~20 постов); backfill в temp-каталог.
  2. Поиск (в одном процессе): 10 вопросов × keyword/dense/hybrid — среднее время/запрос.
  3. Промежуточные этапы: load_posts, стемминг корпуса, FTS-индекс (cold/warm),
     load_dense, rerender (всё во временных каталогах, реальные данные не трогаются).
  4. Общее время: 10 вопросов end-to-end через CLI (hybrid, k=5) — полный цикл,
     включая спавн процесса, импорты и загрузку модели.

Запуск: .venv/bin/python tools/benchmark.py [--channel turboproject]
        [--out research/experiments/benchmarks.md]
Этапы с сетью помечены «сеть» — их время зависит от канала и соединения.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent


def questions() -> list[str]:
    data = json.loads((HERE / "test_questions.json").read_text(encoding="utf-8"))
    return [q["q"] for q in data["questions"]]


def bench_parsing(channel: str, out: list[str]) -> None:
    """Этап 1: парсинг (сеть)."""
    from tg_channel_parser import fetch_page
    t0 = time.perf_counter()
    posts = fetch_page(channel, None)
    dt = time.perf_counter() - t0
    out.append(f"- Первая страница t.me/s (fetch+parse): **{dt * 1000:.0f} мс**, {len(posts)} постов (сеть)")

    tmp = Path(tempfile.mkdtemp(prefix="tg-kb-bench"))
    try:
        t0 = time.perf_counter()
        subprocess.run(
            [sys.executable, str(HERE / "tg_channel_parser.py"), channel, "backfill",
             "--limit-pages", "3", "--data-dir", str(tmp)],
            capture_output=True, text=True, check=True)
        dt = time.perf_counter() - t0
        n = sum(1 for _ in (tmp / f"{channel}_posts.jsonl").read_text(encoding="utf-8").splitlines() if _.strip())
        out.append(f"- Backfill 3 страниц: **{dt:.1f} с**, {n} постов → **{n / dt:.0f} пост/с** (сеть)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def bench_search(channel: str, dd: Path, out: list[str]) -> None:
    """Этап 2: поиск в одном процессе (без загрузки модели на каждый запрос)."""
    from kb_search import (build_fts_index, dense_search, keyword_search,
                           load_dense, load_posts, reaction_bonus, rrf_fuse)
    posts = load_posts(channel, dd)
    conn, _ = build_fts_index(channel, dd, posts)
    model, emb, ids = load_dense(channel, dd, posts)
    qs = questions()
    for mode in ("keyword", "dense", "hybrid"):
        times = []
        for q in qs:
            t0 = time.perf_counter()
            kw = keyword_search(conn, q, 5) if mode in ("keyword", "hybrid") else []
            if mode == "dense" and model is not None:
                dense_search(q, model, emb, ids, max(5, 40))
            elif mode == "hybrid":
                if model is not None and kw:
                    dn = dense_search(q, model, emb, ids, max(5, 40))
                    hits = rrf_fuse(dn, kw, top_k=max(5, 40))
                    reaction_bonus(hits, posts)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        mean = sum(times) / len(times)
        med = times[len(times) // 2]
        note = "" if (mode != "dense" and model is not None) or mode == "keyword" else " (модель загружена один раз на процесс)"
        out.append(f"- Поиск {mode}: среднее **{mean:.1f} мс**/запрос, медиана **{med:.1f} мс** (10 вопросов){note}")


def bench_stages(channel: str, dd: Path, out: list[str]) -> None:
    """Этап 3: промежуточные этапы (temp-каталоги, реальные данные не трогаются)."""
    from kb_search import build_fts_index, load_dense, load_posts, stem_text
    t0 = time.perf_counter()
    posts = load_posts(channel, dd)
    out.append(f"- load_posts ({len(posts)} постов из JSONL): **{(time.perf_counter() - t0) * 1000:.0f} мс**")

    t0 = time.perf_counter()
    for p in posts.values():
        stem_text(p.get("text") or "")
    out.append(f"- Стемминг корпуса ({len(posts)} постов): **{time.perf_counter() - t0:.1f} с**")

    tmp = Path(tempfile.mkdtemp(prefix="tg-kb-bench"))
    try:
        t0 = time.perf_counter()
        build_fts_index(channel, tmp, posts)
        out.append(f"- FTS-индекс, cold-сборка ({len(posts)} постов, со стеммингом): **{time.perf_counter() - t0:.1f} с**")
        t0 = time.perf_counter()
        build_fts_index(channel, tmp, posts)
        out.append(f"- FTS-индекс, warm (переиспользование кэша): **{(time.perf_counter() - t0) * 1000:.0f} мс**")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    t0 = time.perf_counter()
    model, emb, ids = load_dense(channel, dd, posts)
    out.append(f"- load_dense (модель + emb-матрица {emb.shape[0]}×{emb.shape[1]}): **{time.perf_counter() - t0:.1f} с**")

    tmp2 = Path(tempfile.mkdtemp(prefix="tg-kb-bench-rr"))
    try:
        shutil.copy(dd / f"{channel}_posts.jsonl", tmp2 / f"{channel}_posts.jsonl")
        t0 = time.perf_counter()
        subprocess.run(
            [sys.executable, str(HERE / "tg_channel_parser.py"), channel, "rerender", "--data-dir", str(tmp2)],
            capture_output=True, text=True, check=True)
        out.append(f"- rerender {len(posts)} постов в .md: **{time.perf_counter() - t0:.1f} с**")
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)


def bench_total(channel: str, out: list[str]) -> None:
    """Этап 4: общее время — полный цикл агента (спавн процесса + импорты на каждый вопрос)."""
    qs = questions()
    t0 = time.perf_counter()
    for q in qs:
        subprocess.run([sys.executable, str(HERE / "kb_search.py"), channel, q,
                        "--mode", "hybrid", "--k", "5"], capture_output=True)
    dt = time.perf_counter() - t0
    out.append(f"- 10 вопросов end-to-end, hybrid через CLI: **{dt:.1f} с** всего, **{dt / len(qs):.1f} с**/вопрос "
               "(спавн процесса + импорты + модель на каждый вопрос)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Замеры скорости tg-kb по этапам выполнения.")
    ap.add_argument("--channel", default="turboproject", help="канал с собранными данными")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=str(HERE.parent / "research" / "experiments" / "benchmarks.md"))
    args = ap.parse_args(argv)

    from kb_search import resolve_data_dir
    dd = resolve_data_dir(args.data_dir)
    out: list[str] = []
    print("Этап 1: парсинг (сеть)...", flush=True)
    bench_parsing(args.channel, out)
    print("Этап 2: поиск...", flush=True)
    bench_search(args.channel, dd, out)
    print("Этап 3: промежуточные этапы...", flush=True)
    bench_stages(args.channel, dd, out)
    print("Этап 4: общее время (end-to-end)...", flush=True)
    bench_total(args.channel, out)

    env = "python3 + .venv (sentence-transformers, dense)" if "dense: среднее" in "\n".join(out) else "python3"
    report = (
        "# Замеры скорости tg-kb по этапам выполнения\n\n"
        f"- Дата: {time.strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"- Окружение: {env}; корпус и данные — без изменений (temp-каталоги для холодных сборок)\n"
        "- Этапы с сетью помечены «сеть» — время зависит от канала и соединения\n\n"
        + "\n".join(f"- {line.lstrip('- ')}" for line in out) + "\n"
    )
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
