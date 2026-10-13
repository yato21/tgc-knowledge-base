#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tg_channel_parser.py — парсер публичных Telegram-каналов через веб-превью t.me/s/<channel>.

Как это работает
================
Telegram отдаёт публичное веб-превью канала на https://t.me/s/<channel> (~20 постов на
страницу) с пагинацией назад через ?before=<message_id> (возвращает посты с id < before).
Это позволяет обойти канал до самого начала без API-ключей и без MTProto-логина.

Хранение
========
- data/<channel>_posts.jsonl  — все посты, одна JSON-строка на пост, отсортированы по id.
- data/<channel>_state.json   — состояние: max_id, диапазон покрытия, время последнего прогона.

Инкрементальные обновления
==========================
`update` берёт свежую страницу, идёт назад по ?before= пока не встретит уже сохранённые
посты, и мерджит по id (upsert) — так подхватываются и новые посты, и правки хвоста.
Опция --refresh-days N дополнительно перекачивает посты за последние N дней (ловит правки).

Пропуски id
===========
id постов в канале последовательные, но «дыры» в диапазоне нормальны и неустранимы через
превью: альбом из N фото занимает N соседних id, но показывается одним постом (первым id
группы), а удалённые посты в превью просто невидимы. `check` показывает покрытие и
раскладывает дыры на «вероятно альбом» и «неизвестно (вероятно удалён)».

Примеры
=======
  python3 tools/tg_channel_parser.py turboproject update                  # инкремент
  python3 tools/tg_channel_parser.py turboproject backfill --since 2025-10-08
  python3 tools/tg_channel_parser.py turboproject backfill --all          # вся история
  python3 tools/tg_channel_parser.py turboproject check                   # покрытие/пропуски
  python3 tools/tg_channel_parser.py turboproject export --format txt --out data/turbo.txt
  python3 tools/tg_channel_parser.py turboproject probe --before 1500     # одна страница (отладка)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)
VOID_TAGS = {"br", "img", "hr", "meta", "input", "link", "source", "area", "base", "col", "embed", "param", "track", "wbr"}

TEXT_DIV_CLASSES = ("tgme_widget_message_text",)
MEDIA_CLASS_MARKERS = {
    "photo": "tgme_widget_message_photo",
    "video": "tgme_widget_message_video",
    "file": "tgme_widget_message_file",
    "voice": "tgme_widget_message_voice",
    "sticker": "tgme_widget_message_sticker",
    "poll": "tgme_widget_message_poll",
    "round_video": "tgme_widget_message_round_video",
    "location": "tgme_widget_message_location",
}
NUM_SUFFIX = {"K": 1_000, "M": 1_000_000}


# --------------------------------------------------------------------------- HTTP

def fetch_html(url: str, retries: int = 5, timeout: float = 30.0) -> str:
    """Скачивает страницу с ретраями и экспоненциальной паузой на 429/5xx/сеть."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en,ru;q=0.9"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    raise urllib.error.HTTPError(url, resp.status, "non-200", resp.headers, None)
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError) as err:
            last_err = err
            status = getattr(err, "code", None)
            if status in (404, 403):  # канал закрыт/не существует — ретраи бессмысленны
                raise
            pause = min(2 ** attempt, 32) + random.uniform(0, 0.5)
            print(f"  ! попытка {attempt + 1}/{retries} не удалась ({err}); пауза {pause:.1f}с", file=sys.stderr)
            time.sleep(pause)
    raise RuntimeError(f"не удалось скачать {url}: {last_err}")


# --------------------------------------------------------------------------- парсинг

def _parse_count(raw: str) -> int | None:
    """'5.18K' -> 5180, '2.3M' -> 2300000, '39' -> 39."""
    raw = raw.strip().replace("\xa0", "").replace(" ", "")
    m = re.fullmatch(r"([\d.,]+)([KM]?)", raw)
    if not m:
        return None
    num = float(m.group(1).replace(",", "."))
    return int(num * NUM_SUFFIX.get(m.group(2), 1))


class TgmePageParser(HTMLParser):
    """Разбирает HTML страницы t.me/s/ в записи постов.

    Подход: стек открытых тегов. Начало поста — div с data-post; внутри поста
    собираем текст (класс tgme_widget_message_text), реакции (tgme_reaction),
    просмотры, автора/подпись, форвард, типы медиа, метку edited и дату (<time>).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, str, dict]] = []  # (tag, classes, attrs)
        self.messages: list[dict] = []
        self.cur: dict | None = None
        self.cur_depth = 0            # глубина стека на момент открытия поста
        self.text_parts: list | None = None
        self.cur_href: str | None = None
        self.reaction: dict | None = None
        self.seen_time = False

    # -- служебные ------------------------------------------------------------
    def _finish_current(self) -> None:
        if self.cur is not None:
            text = re.sub(r"\n{3,}", "\n\n", self.cur.get("text") or "").strip()
            self.cur["text"] = text
            self.cur["links"] = list(dict.fromkeys(self.cur.get("links", [])))
            self.messages.append(self.cur)
            self.cur = None
        self.text_parts = None
        self.cur_href = None
        self.reaction = None

    # -- события парсера ------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        ad = {k: (v or "") for k, v in attrs}
        classes = ad.get("class", "")
        if tag not in VOID_TAGS:
            self.stack.append((tag, classes, ad))

        if self.cur is None:
            if tag == "div" and "tgme_widget_message" in classes and "data-post" in ad \
                    and "tgme_widget_message_wrap" not in classes and "tgme_widget_message_user" not in classes:
                self.cur = {
                    "id": int(ad["data-post"].rsplit("/", 1)[-1]),
                    "text": "",
                    "views_raw": None,
                    "reactions": [],
                    "author": None,
                    "channel_name": None,
                    "forwarded_from": None,
                    "edited": False,
                    "media": [],
                    "links": [],
                    "date_utc": None,
                }
                self.cur_depth = len(self.stack)
                self.seen_time = False
            return

        # --- внутри поста ---
        # типы медиа собираем инкрементально по классам любых тегов внутри поста
        for media_name, marker in MEDIA_CLASS_MARKERS.items():
            if marker in classes and media_name not in self.cur["media"]:
                self.cur["media"].append(media_name)
        if self.text_parts is not None:
            if tag == "br":
                self.text_parts.append("\n")
            if tag == "a" and ad.get("href"):
                href = ad["href"]
                if href.startswith(("http://", "https://", "tg://")):
                    self.cur["links"].append(href)
                self.cur_href = href
        if "tgme_widget_message_text" in classes and self.text_parts is None:
            # reply-блок содержит цитату с тем же классом текста — не собираем её
            # (режим отказа из dev.to-обзора парсеров t.me/s; в этом канале не встречался)
            in_reply = any("tgme_widget_message_reply" in c for _, c, _ in self.stack)
            if not in_reply:
                self.text_parts = []
        if tag == "time" and not self.seen_time and ad.get("datetime"):
            self.cur["date_utc"] = ad["datetime"]
            self.seen_time = True
        if "tgme_widget_message_views" in classes:
            self.cur["views_raw"] = "PENDING_VIEWS"  # значение придёт в handle_data
        if "tgme_widget_message_from_author" in classes or (
                "tgme_widget_message_author" in classes and "tgme_widget_message_owner_name" not in classes):
            self.cur["author"] = "PENDING_AUTHOR"
        if "tgme_widget_message_owner_name" in classes:
            self.cur["channel_name"] = "PENDING_OWNER"
        if "tgme_widget_message_forwarded_from_name" in classes:
            self.cur["forwarded_from"] = "PENDING_FORWARD"
        if "tgme_reaction" in classes and self.reaction is None:
            self.reaction = {"emoji": "", "count": ""}
        if tag in ("p", "pre", "blockquote", "ul", "ol") and self.text_parts is not None:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        # граница строки на закрытии блочного элемента
        if self.cur is not None and self.text_parts is not None and self.stack:
            top_tag, top_classes, _ = self.stack[-1]
            if top_tag == tag and (tag in ("p", "pre", "blockquote", "ul", "ol")
                                   or "tgme_widget_message_text" in top_classes):
                self.text_parts.append("\n")
        if self.reaction is not None and tag == "span":
            if self.reaction["emoji"] and self.cur is not None:
                self.cur["reactions"].append(self.reaction)
            self.reaction = None
        if self.cur_href is not None and tag == "a":
            self.cur_href = None

        # снимаем со стека до совпадающего тега
        while self.stack:
            stag, sclasses, sad = self.stack.pop()
            if self.cur is not None and stag == "div" and "tgme_widget_message_text" in sclasses:
                if self.text_parts is not None:
                    self.cur["text"] = "".join(self.text_parts)  # фиксируем текст при закрытии текстового div
                self.text_parts = None
            if stag == tag:
                if self.cur is not None and "tgme_widget_message" in sclasses and "data-post" in sad \
                        and "tgme_widget_message_wrap" not in sclasses:
                    self._finish_current()
                break

    def handle_data(self, data: str) -> None:
        if self.cur is None:
            return
        if self.text_parts is not None:
            self.text_parts.append(data)
        if self.reaction is not None:
            if self.reaction["emoji"] == "" and data.strip():
                self.reaction["emoji"] = data.strip()
            else:
                self.reaction["count"] += data.strip()
        stack_classes = " ".join(c for _, c, _ in self.stack)
        if "tgme_widget_message_views" in stack_classes and self.cur.get("views_raw") in (None, "PENDING_VIEWS") \
                and data.strip() and not self.reaction:
            self.cur["views_raw"] = data.strip()
        if self.cur.get("author") == "PENDING_AUTHOR" and "tgme_widget_message_from_author" in stack_classes and data.strip():
            self.cur["author"] = data.strip()
        if self.cur.get("channel_name") == "PENDING_OWNER" and "tgme_widget_message_owner_name" in stack_classes and data.strip():
            self.cur["channel_name"] = data.strip()
        if self.cur.get("forwarded_from") == "PENDING_FORWARD" and "tgme_widget_message_forwarded_from_name" in stack_classes and data.strip():
            self.cur["forwarded_from"] = data.strip()
        if "tgme_widget_message_meta" in stack_classes and "edited" in data:
            self.cur["edited"] = True

    def close(self) -> None:  # noqa: D102
        super().close()
        self._finish_current()


def extract_messages(html: str) -> list[dict]:
    """HTML страницы t.me/s/ -> список постов (по возрастанию id)."""
    parser = TgmePageParser()
    parser.feed(html)
    parser.close()
    posts = parser.messages
    posts.sort(key=lambda p: p["id"])
    return posts


def normalize_post(post: dict, channel: str, scraped_at: str) -> dict:
    views_raw = post.get("views_raw")
    rec = {
        "id": post["id"],
        "channel": channel,
        "date_utc": post.get("date_utc"),
        "text": post.get("text", ""),
        "views_raw": views_raw,
        "views": _parse_count(views_raw) if views_raw else None,
        "reactions": {r["emoji"]: _parse_count(r["count"]) or 0 for r in post.get("reactions", [])},
        "author": post.get("author"),
        "channel_name": post.get("channel_name"),
        "forwarded_from": post.get("forwarded_from"),
        "edited": post.get("edited", False),
        "media": post.get("media", []),
        "links": post.get("links", []),
        "scraped_at": scraped_at,
    }
    return rec


# --------------------------------------------------------------------------- ход по страницам

def fetch_page(channel: str, before: int | None = None) -> list[dict]:
    url = f"https://t.me/s/{channel}" + (f"?before={before}" if before is not None else "")
    html = fetch_html(url)
    return extract_messages(html)


def walk_pages(channel: str, start_before: int | None, stop_check, delay: float, max_pages: int) -> tuple[list[dict], int]:
    """Идёт назад по ?before=, пока stop_check(страница) не скажет остановиться."""
    collected: dict[int, dict] = {}
    before = start_before
    pages = 0
    while pages < max_pages:
        posts = fetch_page(channel, before)
        pages += 1
        if not posts:
            break
        for p in posts:
            collected[p["id"]] = p
        page_min = min(p["id"] for p in posts)
        if stop_check(posts):
            break
        before = page_min
        time.sleep(delay)
    posts = [collected[k] for k in sorted(collected)]
    return posts, pages


# --------------------------------------------------------------------------- хранение

def resolve_data_dir(cli_value: str | None) -> Path:
    """Приоритет: --data-dir > env TG_KB_HOME > ./data (паттерн GREPOGRAM_HOME из grepogram;
    проектный дефолт — прямое следствие песочниц агентов: запись вне workspace недоступна)."""
    if cli_value:
        return Path(cli_value)
    env = (os.environ.get("TG_KB_HOME") or "").strip()
    if env:
        return Path(env)
    return Path("data")


def data_dir(data_dir_arg: str | None) -> Path:
    d = resolve_data_dir(data_dir_arg)
    d.mkdir(parents=True, exist_ok=True)
    return d


def posts_path(channel: str, dd: Path) -> Path:
    return dd / f"{channel}_posts.jsonl"


def state_path(channel: str, dd: Path) -> Path:
    return dd / f"{channel}_state.json"


def load_state(channel: str, dd: Path) -> dict:
    p = state_path(channel, dd)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"channel": channel, "max_id": None, "min_id": None, "post_count": 0,
            "last_update": None, "last_full_coverage": False}


def save_state(channel: str, dd: Path, state: dict) -> None:
    state_path(channel, dd).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


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
                # битая строка не должна крашить команду (фикс task-9 §3.1)
                print(f"warning: битая JSONL-строка пропущена: {line[:80]}…", file=sys.stderr)
                continue
            result[rec["id"]] = rec
    return result


def save_posts(channel: str, dd: Path, posts_by_id: dict[int, dict]) -> None:
    lines = [json.dumps(posts_by_id[k], ensure_ascii=False) for k in sorted(posts_by_id)]
    posts_path(channel, dd).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def to_utc_date(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


# --------------------------------------------------------------------------- команды

def cmd_update(args) -> int:
    dd = data_dir(args.data_dir)
    state = load_state(args.channel, dd)
    stored = load_posts(args.channel, dd)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    limit_pages = args.limit_pages or 10_000
    first_page = fetch_page(args.channel)
    if not first_page:
        print("Страница не содержит постов — канал пуст или превью закрыто.", file=sys.stderr)
        return 2
    fetched: dict[int, dict] = {p["id"]: p for p in first_page}
    new_max = max(fetched)
    pages_total = 1

    # 1) докачиваем назад, пока не встретим сохранённое
    page_min = min(fetched)
    known_max = state["max_id"]
    overlap_reached = known_max is not None and page_min <= known_max
    if not overlap_reached:
        # стоп: страница дошла до постов, уже известных корпусу (иначе —
        # прогулка по всей истории канала; найдено при update 2026-10-09:
        # 257 страниц, +2055 постов вместо инкремента)
        def stop_at_known(posts: list[dict]) -> bool:
            return any(p["id"] <= known_max for p in posts) if known_max is not None else False
        older, pages = walk_pages(args.channel, page_min, stop_at_known, args.delay, limit_pages - pages_total)
        pages_total += pages
        for p in older:
            fetched[p["id"]] = p
        overlap_reached = state["max_id"] is not None and min(fetched) <= state["max_id"]

    # 2) рефреш хвоста: докачиваем до постов старше N дней (ловим правки)
    if args.refresh_days > 0 and overlap_reached:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.refresh_days)

        def stop_by_date(posts: list[dict]) -> bool:
            return all((to_utc_date(p.get("date_utc")) or cutoff) < cutoff for p in posts)

        tail, pages = walk_pages(args.channel, min(fetched), stop_by_date, args.delay, limit_pages - pages_total)
        pages_total += pages
        for p in tail:
            fetched[p["id"]] = p

    # 3) мердж: обновляем существующие (правки) + добавляем новые.
    # Дрейф просмотров/реакций — нормальное явление: перезаписываем молча,
    # а «обновлением» считаем только содержательные изменения (текст/дата/edited/форвард).
    updated_existing = 0
    added_new = 0
    for pid, raw in fetched.items():
        rec = normalize_post(raw, args.channel, now_iso)
        old = stored.get(pid)
        if old is None:
            stored[pid] = rec
            added_new += 1
        else:
            content_changed = (old["text"] != rec["text"] or old.get("edited") != rec.get("edited")
                               or old.get("date_utc") != rec.get("date_utc")
                               or old.get("forwarded_from") != rec.get("forwarded_from"))
            stored[pid] = rec
            if content_changed:
                updated_existing += 1

    save_posts(args.channel, dd, stored)
    max_id = max(stored) if stored else None
    min_id = min(stored) if stored else None
    state.update({
        "max_id": max_id,
        "min_id": min_id,
        "post_count": len(stored),
        "last_update": now_iso,
    })
    save_state(args.channel, dd, state)

    print(f"Канал @{args.channel}: страниц {pages_total}, добавлено {added_new}, обновлено {updated_existing}.")
    print(f"Всего постов: {len(stored)}; диапазон id {min_id}..{max_id}.")
    if not overlap_reached and state["max_id"] is not None:
        print("Внимание: пересечение со старыми данными не достигнуто — в диапазоне возможны дыры, запусти ещё раз или check.")
    return 0


def cmd_backfill(args) -> int:
    dd = data_dir(args.data_dir)
    state = load_state(args.channel, dd)
    stored = load_posts(args.channel, dd)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    limit_pages = args.limit_pages or 10_000
    since_dt = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc) if args.since else None

    def stop_check(posts: list[dict]) -> bool:
        if since_dt is None:
            return False  # --all: идём до конца канала
        return all((to_utc_date(p.get("date_utc")) or since_dt) < since_dt for p in posts)

    fetched_list, pages_total = walk_pages(args.channel, args.before, stop_check, args.delay, limit_pages)
    fetched = {p["id"]: p for p in fetched_list}
    if not fetched:
        print("Посты не найдены (страница пуста или предел страниц).", file=sys.stderr)
        return 2

    added = 0
    updated = 0
    for pid, raw in fetched.items():
        rec = normalize_post(raw, args.channel, now_iso)
        old = stored.get(pid)
        if old is None:
            stored[pid] = rec
            added += 1
        elif old["text"] != rec["text"] or old.get("date_utc") != rec.get("date_utc"):
            stored[pid] = rec
            updated += 1

    save_posts(args.channel, dd, stored)
    state.update({
        "max_id": max(stored) if stored else None,
        "min_id": min(stored) if stored else None,
        "post_count": len(stored),
        "last_update": now_iso,
    })
    save_state(args.channel, dd, state)
    print(f"Страниц {pages_total}, добавлено {added}, обновлено {updated}. Всего постов: {len(stored)}.")
    return 0


def cmd_check(args) -> int:
    dd = data_dir(args.data_dir)
    stored = load_posts(args.channel, dd)
    if not stored:
        print("Нет сохранённых постов — сначала update/backfill.", file=sys.stderr)
        return 2
    ids = sorted(stored)
    lo, hi = ids[0], ids[-1]
    present = set(ids)
    missing = [i for i in range(lo, hi + 1) if i not in present]

    # раскладка дыр: если сразу после дыры пост с несколькими медиа — вероятно альбом
    likely_album, unknown = 0, []
    for i in missing:
        nxt = next((j for j in ids if j > i), None)
        if nxt and stored[nxt].get("media"):
            likely_album += 1
        else:
            unknown.append(i)

    empty_text = sum(1 for i in ids if not (stored[i].get("text") or "").strip())
    no_date = sum(1 for i in ids if not stored[i].get("date_utc"))
    print(f"Канал @{args.channel}: постов {len(ids)}, диапазон id {lo}..{hi}.")
    print(f"Дыр в диапазоне: {len(missing)} (вероятно альбомы: {likely_album}; прочие/удалённые: {len(unknown)})")
    if unknown:
        sample = ", ".join(map(str, unknown[:15]))
        print(f"  неизвестные дыры (первые 15): {sample}{'…' if len(unknown) > 15 else ''}")
    print(f"Постов без текста (только медиа): {empty_text}; без даты: {no_date}.")
    return 0


def cmd_export(args) -> int:
    dd = data_dir(args.data_dir)
    stored = load_posts(args.channel, dd)
    if not stored:
        print("Нет сохранённых постов — сначала update/backfill.", file=sys.stderr)
        return 2
    ids = sorted(stored)
    if args.since:
        since_dt = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        ids = [i for i in ids if (to_utc_date(stored[i].get("date_utc")) or since_dt) >= since_dt]

    out = Path(args.out) if args.out else dd / f"{args.channel}_export.{args.format}"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "jsonl":
        out.write_text("".join(json.dumps(stored[i], ensure_ascii=False) + "\n" for i in ids), encoding="utf-8")
    elif args.format == "txt":
        parts = []
        for i in ids:
            r = stored[i]
            reactions = " ".join(f"{e}{c}" for e, c in r.get("reactions", {}).items())
            flags = []
            if r.get("edited"):
                flags.append("edited")
            if r.get("forwarded_from"):
                flags.append(f"forward from {r['forwarded_from']}")
            if r.get("media"):
                flags.append("media: " + ",".join(r["media"]))
            head = f"=== POST {i} | {(r.get('date_utc') or '?')} UTC | views {r.get('views_raw') or '?'}"
            if reactions:
                head += f" | {reactions}"
            if flags:
                head += " | " + "; ".join(flags)
            parts.append(f"{head} ===\nhttps://t.me/{args.channel}/{i}\n\n{(r.get('text') or '').strip()}\n")
        out.write_text("\n".join(parts), encoding="utf-8")
    elif args.format == "md":
        parts = [f"# @{args.channel} — посты ({len(ids)})\n"]
        for i in ids:
            r = stored[i]
            date = (r.get("date_utc") or "?").replace("T", " ")[:16]
            parts.append(f"## POST {i} — {date} UTC\n\n{(r.get('text') or '').strip()}\n")
        out.write_text("\n".join(parts), encoding="utf-8")
    else:
        print(f"Неизвестный формат: {args.format}", file=sys.stderr)
        return 2
    print(f"Экспортировано {len(ids)} постов -> {out}")
    return 0


def cmd_rerender(args) -> int:
    """Каждый пост — отдельный файл <dir>/<id>.md: агент читает файл ЦЕЛИКОМ,
    без обрезки длинных JSONL-строк (дефект из file-based-эксперимента)."""
    dd = data_dir(args.data_dir)
    stored = load_posts(args.channel, dd)
    if not stored:
        print("Нет сохранённых постов — сначала update/backfill.", file=sys.stderr)
        return 2
    ids = sorted(stored)
    if args.since:
        since_dt = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        ids = [i for i in ids if (to_utc_date(stored[i].get("date_utc")) or since_dt) >= since_dt]
    out_dir = Path(args.out) if args.out else dd / "posts" / args.channel
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in ids:
        r = stored[i]
        reactions = " ".join(f"{e}{c}" for e, c in r.get("reactions", {}).items())
        flags = []
        if r.get("edited"):
            flags.append("edited")
        if r.get("forwarded_from"):
            flags.append(f"forward from {r['forwarded_from']}")
        if r.get("media"):
            flags.append("media: " + ",".join(r["media"]))
        head = f"POST {i} | {(r.get('date_utc') or '?')} UTC | views {r.get('views_raw') or '?'}"
        if reactions:
            head += f" | {reactions}"
        if flags:
            head += " | " + "; ".join(flags)
        body = f"# {head}\n\nhttps://t.me/{args.channel}/{i}\n\n{(r.get('text') or '').strip()}\n"
        (out_dir / f"{i}.md").write_text(body, encoding="utf-8")
        n += 1
    print(f"Перерендерено {n} постов -> {out_dir}/<id>.md")
    # сироты: .md-файлы без поста в JSONL вводят агента в заблуждение (фикс task-9 табл. 12)
    kept = {f"{i}.md" for i in ids}
    orphans = [f.name for f in out_dir.glob("*.md") if f.name not in kept]
    for name in orphans:
        (out_dir / name).unlink()
    if orphans:
        print(f"Удалено сирот: {len(orphans)} ({', '.join(orphans[:5])}{'…' if len(orphans) > 5 else ''})")
    return 0


def cmd_probe(args) -> int:
    """Одна страница ?before=ID — отладка разметки и проверка конкретных страниц."""
    posts = fetch_page(args.channel, args.before)
    print(f"Страница ?before={args.before}: постов {len(posts)}")
    for p in posts:
        print(f"  id={p['id']} date={p.get('date_utc')} views={p.get('views_raw')} "
              f"media={','.join(p.get('media', [])) or '-'} text_len={len(p.get('text') or '')} "
              f"edited={p.get('edited')} fwd={p.get('forwarded_from')}")
    if args.show_text and posts:
        for p in posts:
            print(f"\n----- id={p['id']} -----\n{p.get('text', '')[:1500]}")
    return 0


def cmd_stats(args) -> int:
    dd = data_dir(args.data_dir)
    state = load_state(args.channel, dd)
    stored = load_posts(args.channel, dd)
    src = "cli:--data-dir" if args.data_dir else ("env:TG_KB_HOME" if (os.environ.get("TG_KB_HOME") or "").strip() else "default:./data")
    print(json.dumps({
        "state": state,
        "posts_loaded": len(stored),
        "data_dir_resolved": {"path": str(dd), "source": src},
        "files": {"posts": str(posts_path(args.channel, dd)), "state": str(state_path(args.channel, dd))},
    }, ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Парсер публичных Telegram-каналов через t.me/s/ веб-превью.")
    ap.add_argument("channel", help="username канала без @, например turboproject")
    ap.add_argument("command", choices=["update", "backfill", "check", "export", "rerender", "probe", "stats"])
    ap.add_argument("--data-dir", default=None, help="папка для jsonl/state; приоритет: --data-dir > env TG_KB_HOME > ./data")
    ap.add_argument("--since", help="дата YYYY-MM-DD: нижняя граница (backfill/export)")
    ap.add_argument("--before", type=int, help="id для probe; для backfill — начать с этой страницы")
    ap.add_argument("--all", action="store_true", help="backfill: вся история канала")
    ap.add_argument("--limit-pages", type=int, help="максимум страниц за прогон")
    ap.add_argument("--refresh-days", type=int, default=2, help="update: перекачать хвост за N дней (ловит правки)")
    ap.add_argument("--delay", type=float, default=0.7, help="пауза между запросами, сек")
    ap.add_argument("--format", choices=["txt", "jsonl", "md"], default="txt", help="формат export")
    ap.add_argument("--out", help="файл экспорта")
    ap.add_argument("--show-text", action="store_true", help="probe: показать тексты")
    args = ap.parse_args(argv)

    # валидация имени канала (Telegram username) — имя строится в путях файлов;
    # без неё аргумент вида "../../x" писал бы вне data-dir (аудит безопасности)
    if not re.match(r"^[A-Za-z0-9_]{1,64}$", args.channel or ""):
        print(f"error: некорректное имя канала: {args.channel!r} — ожидается [A-Za-z0-9_], до 64 символов", file=sys.stderr)
        return 2

    if args.command == "update":
        return cmd_update(args)
    if args.command == "backfill":
        return cmd_backfill(args)
    if args.command == "check":
        return cmd_check(args)
    if args.command == "export":
        return cmd_export(args)
    if args.command == "rerender":
        return cmd_rerender(args)
    if args.command == "probe":
        return cmd_probe(args)
    if args.command == "stats":
        return cmd_stats(args)
    print(f"Неизвестная команда: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
