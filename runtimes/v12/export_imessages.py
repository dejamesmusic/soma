#!/usr/bin/env python3
"""export local Messages text into a privacy-conscious plain-text corpus.

requires macos full disk access because chat.db is protected by tcc. exported
turns preserve conversation, date, direction, and stable anonymized contacts;
attachments, reactions, and service messages are deliberately omitted.
"""

import argparse
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from Foundation import NSData, NSUnarchiver


APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)


def message_text(plain, attributed):
    if plain and plain.strip():
        text = plain
    elif attributed:
        try:
            data = NSData.dataWithBytes_length_(attributed, len(attributed))
            text = str(NSUnarchiver.unarchiveObjectWithData_(data).string())
        except Exception:
            return ""
    else:
        return ""
    return re.sub(r"\s+", " ", text.replace("\x1e", " ")).strip()


def apple_time(value):
    value = float(value or 0)
    seconds = value / 1_000_000_000 if value > 10_000_000_000 else value
    return (APPLE_EPOCH + timedelta(seconds=seconds)).astimezone().date().isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database", type=Path,
        default=Path.home() / "Library/Messages/chat.db")
    parser.add_argument(
        "--output", type=Path,
        default=Path.home() / "Library/Application Support/soma/data/imessage_history.txt")
    args = parser.parse_args()

    if not args.database.exists():
        raise SystemExit(f"messages database not found: {args.database}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    cursor = connection.execute("""
        select cmj.chat_id, m.date, m.is_from_me, h.id, m.text, m.attributedBody
        from chat_message_join as cmj
        join message as m on m.ROWID = cmj.message_id
        left join handle as h on h.ROWID = m.handle_id
        where coalesce(m.associated_message_type, 0) = 0
          and coalesce(m.is_empty, 0) = 0
          and coalesce(m.is_service_message, 0) = 0
          and (m.text is not null or m.attributedBody is not null)
        order by cmj.chat_id, m.date, m.ROWID
    """)

    contacts = {}
    conversations = defaultdict(list)
    skipped = 0
    for chat_id, date, is_from_me, handle, plain, attributed in cursor:
        text = message_text(plain, attributed)
        if not text:
            skipped += 1
            continue
        if is_from_me:
            speaker = "me"
        else:
            key = handle or f"unknown-{chat_id}"
            if key not in contacts:
                contacts[key] = f"other-{len(contacts) + 1:03d}"
            speaker = contacts[key]
        conversations[chat_id].append((apple_time(date), speaker, text))

    count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        out.write("imessage history\n")
        out.write("text messages only; attachments, reactions, and service events "
                  "omitted. participant labels are anonymized; message bodies are retained.\n\n")
        for ordinal, messages in enumerate(conversations.values(), 1):
            out.write(f"<conversation {ordinal:03d}>\n")
            current_date = None
            for date, speaker, text in messages:
                if date != current_date:
                    out.write(f"<date {date}>\n")
                    current_date = date
                out.write(f"{speaker}: {text}\n")
                count += 1
            out.write("</conversation>\n\n")

    print(f"wrote {args.output}")
    print(f"messages: {count:,}; conversations: {len(conversations):,}; "
          f"anonymized contacts: {len(contacts):,}; unreadable bodies skipped: {skipped:,}")


if __name__ == "__main__":
    main()
