#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
install.py — установка tg-kb в рабочую область проекта.

Запуск из склонированного репозитория:

    python3 install.py [целевая-папка]     # по умолчанию — текущая папка

Что делает:
  1. Копирует инструмент (tools/kb_search.py, tg_channel_parser.py, benchmark.py,
     experiment_retrieval.py, experiment_stemming.py, test_questions.json, _vendor/)
     в <цель>/tools/.
  2. Устанавливает скилл: <цель>/.claude/skills/tg-kb/SKILL.md и симлинк
     <цель>/.agents/skills/tg-kb/SKILL.md (если симлинк недоступен — копия).
  3. Дополняет <цель>/AGENTS.md секцией tg-kb (между маркерами <!-- tg-kb:start -->
     и <!-- tg-kb:end -->); создаёт файл, если его нет; повторная установка секцию
     не дублирует.

Идемпотентен: повторный запуск безопасен. data/ в цели не создаётся и не трогается —
база собирается отдельно (tools/tg_channel_parser.py <канал> backfill).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
TOOL_FILES = [
    "kb_search.py", "tg_channel_parser.py", "benchmark.py",
    "experiment_retrieval.py", "experiment_stemming.py", "test_questions.json",
]
START, END = "<!-- tg-kb:start -->", "<!-- tg-kb:end -->"


def install_tools(target: Path) -> list[str]:
    done = []
    tdir = target / "tools"
    tdir.mkdir(parents=True, exist_ok=True)
    for name in TOOL_FILES:
        shutil.copy2(SRC / "tools" / name, tdir / name)
        done.append(f"tools/{name}")
    vend = tdir / "_vendor"
    if (SRC / "tools" / "_vendor").exists():
        shutil.copytree(SRC / "tools" / "_vendor", vend, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        done.append("tools/_vendor/ (снежный стеммер, BSD-3-Clause)")
    return done


def install_skill(target: Path) -> list[str]:
    done = []
    src_skill = SRC / ".claude" / "skills" / "tg-kb" / "SKILL.md"
    if not src_skill.exists():
        return done
    dst = target / ".claude" / "skills" / "tg-kb" / "SKILL.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_skill, dst)
    done.append(".claude/skills/tg-kb/SKILL.md (Claude Code, OpenCode, Kilo)")

    link = target / ".agents" / "skills" / "tg-kb" / "SKILL.md"
    link.parent.mkdir(parents=True, exist_ok=True)
    rel = "../../../.claude/skills/tg-kb/SKILL.md"
    if link.is_symlink() or link.exists():
        link.unlink()
    try:
        link.symlink_to(rel)
        done.append(".agents/skills/tg-kb/SKILL.md (симлинк — DSH, OpenCode, Kilo, Pi)")
    except OSError:
        # Windows без core.symlinks: симлинк недоступен — копия (дрейф-риск помечаем)
        shutil.copy2(src_skill, link)
        done.append(".agents/skills/tg-kb/SKILL.md (копия — симлинк недоступен; правки SKILL.md дублируй)")
    return done


def install_agents_md(target: Path) -> list[str]:
    src = SRC / "AGENTS.md"
    text = src.read_text(encoding="utf-8") if src.exists() else ""
    if START not in text or END not in text:
        return ["AGENTS.md: секция tg-kb в исходнике не найдена (маркеры) — пропущено"]
    section = text.split(START, 1)[1].split(END, 1)[0].strip("\n")
    wrapped = f"{START}\n{section}\n{END}\n"

    dst = target / "AGENTS.md"
    if dst.exists():
        cur = dst.read_text(encoding="utf-8")
        if START in cur and END in cur:
            # переустановка: заменяем секцию целиком (обновление инструкций)
            pre = cur.split(START, 1)[0]
            post = cur.split(END, 1)[1]
            dst.write_text(pre + wrapped + post.lstrip("\n"), encoding="utf-8")
            return ["AGENTS.md: секция tg-kb обновлена"]
        dst.write_text(cur.rstrip("\n") + "\n\n" + wrapped, encoding="utf-8")
        return ["AGENTS.md: секция tg-kb дополнена"]
    dst.write_text(wrapped, encoding="utf-8")
    return ["AGENTS.md: создан с секцией tg-kb"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Установка tg-kb в рабочую область проекта.")
    ap.add_argument("target", nargs="?", default=".", help="целевая папка (по умолчанию — текущая)")
    args = ap.parse_args(argv)
    target = Path(args.target).resolve()
    if not target.exists():
        print(f"error: цель не существует: {target}", file=sys.stderr)
        return 2

    steps = install_tools(target) + install_skill(target) + install_agents_md(target)
    print(f"tg-kb установлен в {target}:")
    for s in steps:
        print(f"  - {s}")
    print("\nДалее — собери базу канала (сеть):")
    print(f"  cd {target} && python3 tools/tg_channel_parser.py <канал> backfill")
    print("Проверка: python3 tools/kb_search.py <канал> x --doctor")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
