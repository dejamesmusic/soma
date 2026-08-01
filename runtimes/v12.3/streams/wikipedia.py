"""
wikipedia stream — rolling corpus of recent Wikipedia edits.

Connects to the Wikimedia EventStream (free, no auth, public) and listens
for every edit on English Wikipedia. For each edit, fetches the current
page text via the Wikipedia API and prepends it to a corpus file.

The file is hard-capped at 1GB by default. Once full, oldest content
rolls off the tail as new content is prepended at the head.

This is a "stream" — a script in the streams/ directory that produces
text into a rolling corpus file. The soma launcher discovers any module
in streams/ that declares STREAM_NAME and STREAM_DESCRIPTION and offers
it as an option in the CLI and GUI.

Format:
    Each entry is prefixed by a separator line containing the title and
    timestamp, then the page text, then a trailing newline. Plain UTF-8.

Usage as a script:
    python streams/wikipedia.py corpus.txt [--max-bytes 1073741824]
                                           [--lang en]
                                           [--min-len 200]
                                           [--rewrite-every 50]
"""

# ── stream registration ──
# The launcher and GUI discover streams by parsing these constants out
# of each module in streams/. Keep them at the top of the file as plain
# string assignments so they can be read without executing the module.
STREAM_NAME = "wikipedia"
STREAM_DESCRIPTION = "rolling corpus from wikimedia recent-changes feed"

import sys
import os
import json
import time
import argparse
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from datetime import datetime


GB = 1024 * 1024 * 1024

WIKI_STREAM_URL = "https://stream.wikimedia.org/v2/stream/recentchange"
WIKI_API_URL_FMT = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "soma-corpus-builder/1.0 (research; https://github.com/)"


def open_stream(url, timeout=60):
    """Open the SSE stream. Returns a file-like response object.
    Reconnect handling lives in the outer loop."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/event-stream",
        },
    )
    return urllib.request.urlopen(req, timeout=timeout)


def parse_sse(stream):
    """Iterate SSE messages. Yields parsed JSON dicts of recentchange events.
    The Wikimedia stream emits one JSON object per data: line."""
    buf = b""
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            for line in event.split(b"\n"):
                if line.startswith(b"data: "):
                    try:
                        yield json.loads(line[6:].decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue


def fetch_page_text(title, lang="en", timeout=30):
    """Fetch the plain-text extract of one Wikipedia page.
    Uses the action=query&prop=extracts API which returns clean text
    (no wikicode, no HTML)."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "redirects": 1,
    }
    url = WIKI_API_URL_FMT.format(lang=lang) + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        text = page.get("extract", "")
        if text:
            return text
    return ""


def format_entry(title, timestamp, text):
    """One entry as it appears in the corpus. Plain text, lightly delimited."""
    header = f"\n=== {title} · {timestamp} ===\n\n"
    return header + text + "\n"


def prepend_to_file(path, new_text, max_bytes):
    """Prepend new_text to the head of path, then truncate to max_bytes
    from the head, dropping anything past that from the tail.

    Implementation: writes a fresh temp file, atomically replaces.
    Reads the existing file in chunks so memory use stays bounded
    even for 1GB+ files.
    """
    new_bytes = new_text.encode("utf-8", errors="replace")
    if len(new_bytes) >= max_bytes:
        # New entry alone exceeds budget — keep just its head
        with open(path, "wb") as f:
            f.write(new_bytes[:max_bytes])
        return

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as out:
        out.write(new_bytes)
        bytes_remaining = max_bytes - len(new_bytes)
        if path.exists() and bytes_remaining > 0:
            with open(path, "rb") as old:
                while bytes_remaining > 0:
                    chunk = old.read(min(1024 * 1024, bytes_remaining))
                    if not chunk:
                        break
                    out.write(chunk)
                    bytes_remaining -= len(chunk)
    tmp_path.replace(path)


def fmt_bytes(n):
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


def run(corpus_path, max_bytes, lang, min_len, rewrite_every):
    """Main loop. Reconnects on stream drops, fetches edited page bodies,
    prepends to the corpus, caps file size."""
    corpus_path = Path(corpus_path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    # Buffer of pending entries to write in batches — prepending one entry
    # at a time means rewriting the whole file per edit, which gets
    # painful at 1GB. Buffer up several entries' worth of text, then
    # flush as a single prepend.
    pending = []
    pending_bytes = 0
    flush_threshold_bytes = 256 * 1024  # 256KB

    stats = {
        "events": 0,
        "fetched": 0,
        "skipped": 0,
        "errors": 0,
        "bytes_added": 0,
        "started": time.time(),
    }

    # Track recently-fetched titles to avoid hammering the API for
    # rapid-fire edits to the same page (common during edit wars).
    seen_recently = {}
    SEEN_TTL = 60.0  # seconds

    print(f"  ░▒▓ wikistream ▓▒░")
    print(f"  corpus: {corpus_path}  max: {fmt_bytes(max_bytes)}")
    print(f"  lang: {lang}  min entry length: {min_len} chars")
    print(f"  press Ctrl-C to stop")
    print()

    while True:
        try:
            print(f"  • connecting to {WIKI_STREAM_URL}...")
            stream = open_stream(WIKI_STREAM_URL)
            print(f"  • connected")
            for event in parse_sse(stream):
                stats["events"] += 1

                # filter: only main-namespace edits on chosen wiki, no bots
                if event.get("type") != "edit":
                    continue
                if event.get("wiki") != f"{lang}wiki":
                    continue
                if event.get("namespace") != 0:  # main article namespace
                    continue
                if event.get("bot"):
                    continue

                title = event.get("title", "")
                if not title:
                    continue

                # dedupe rapid-fire edits to the same page
                now = time.time()
                last = seen_recently.get(title, 0)
                if now - last < SEEN_TTL:
                    continue
                seen_recently[title] = now
                # gc the seen-table occasionally
                if len(seen_recently) > 5000:
                    seen_recently = {
                        k: v for k, v in seen_recently.items()
                        if now - v < SEEN_TTL
                    }

                # fetch the page text
                try:
                    text = fetch_page_text(title, lang=lang)
                    stats["fetched"] += 1
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    stats["errors"] += 1
                    continue

                if len(text) < min_len:
                    stats["skipped"] += 1
                    continue

                # build the entry, buffer it
                ts = event.get("meta", {}).get("dt", "")
                entry = format_entry(title, ts, text)
                pending.append(entry)
                pending_bytes += len(entry.encode("utf-8", errors="replace"))

                # flush if buffer's big enough OR every N events as a heartbeat
                if (pending_bytes >= flush_threshold_bytes
                        or stats["fetched"] % rewrite_every == 0):
                    flush(pending, corpus_path, max_bytes, stats)
                    pending = []
                    pending_bytes = 0

                # periodic status line
                if stats["events"] % 100 == 0:
                    elapsed = time.time() - stats["started"]
                    rate = stats["fetched"] / elapsed if elapsed > 0 else 0
                    size = corpus_path.stat().st_size if corpus_path.exists() else 0
                    print(f"  ∿ events={stats['events']:>7} "
                          f"· fetched={stats['fetched']:>5} "
                          f"· {rate:.1f}/s "
                          f"· corpus={fmt_bytes(size)}/{fmt_bytes(max_bytes)} "
                          f"· {datetime.now().strftime('%H:%M:%S')}")

        except KeyboardInterrupt:
            print("\n  ▣ stopping — flushing pending entries")
            if pending:
                flush(pending, corpus_path, max_bytes, stats)
            elapsed = time.time() - stats["started"]
            print(f"  done. {stats['fetched']:,} entries · "
                  f"{fmt_bytes(stats['bytes_added'])} added · "
                  f"{elapsed/3600:.1f} h elapsed")
            return
        except (urllib.error.URLError, OSError, ConnectionError) as e:
            print(f"  ! stream error: {e}; reconnecting in 10s")
            time.sleep(10)
        except Exception as e:
            print(f"  ! unexpected error: {type(e).__name__}: {e}; "
                  f"reconnecting in 30s")
            time.sleep(30)


def flush(pending, corpus_path, max_bytes, stats):
    """Concatenate pending entries (newest first within batch) and prepend."""
    if not pending:
        return
    # Newest-first within the buffer too — pending was appended in order,
    # so reverse so the most recent edit ends up at the very top.
    text = "".join(reversed(pending))
    prepend_to_file(corpus_path, text, max_bytes)
    n = len(text.encode("utf-8", errors="replace"))
    stats["bytes_added"] += n


DATA_DIR = "data"
STREAMS_DATA_DIR = "data/streams"


def _bundle_root():
    """The bundle directory is the parent of streams/, where data/ lives."""
    soma_home = os.environ.get("SOMA_HOME")
    if soma_home:
        return Path(soma_home).expanduser()
    here = Path(__file__).resolve().parent
    if here.name == "streams":
        return here.parent
    return here


def _default_stream_corpus():
    """Default path a stream writes to: data/streams/<STREAM_NAME>.txt.

    Convention: any stream that doesn't explicitly take a corpus
    argument from the user writes to a file named after itself in
    the streams_data directory. This way, streams have predictable
    output locations the gui can list and offer for training.
    """
    d = _bundle_root() / STREAMS_DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f"{STREAM_NAME}.txt")


def _resolve_corpus_path(path):
    """Bare filenames go into <bundle>/data/; explicit paths used as-is.

    Empty / missing → the default stream corpus path. Matches the
    convention used by soma_v10.py and the launcher.
    """
    if not path:
        return _default_stream_corpus()
    if ('/' in path or '\\' in path or
            path.startswith('~') or path.startswith('.')):
        return os.path.expanduser(path)
    data_dir = _bundle_root() / DATA_DIR
    data_dir.mkdir(exist_ok=True)
    return str(data_dir / path)


def main():
    p = argparse.ArgumentParser(
        description="Maintain a rolling corpus of Wikipedia edits.")
    p.add_argument("corpus", type=str, nargs="?", default=None,
                   help="output file path (bare filename → data/, "
                        "explicit path used as-is, omitted → "
                        "data/streams/wikipedia.txt)")
    p.add_argument("--max-bytes", type=int, default=GB,
                   help=f"max corpus size in bytes (default {GB})")
    p.add_argument("--lang", type=str, default="en",
                   help="Wikipedia language code (default en)")
    p.add_argument("--min-len", type=int, default=200,
                   help="skip entries with extracted text shorter than this "
                        "many characters (default 200)")
    p.add_argument("--rewrite-every", type=int, default=50,
                   help="flush buffered entries to disk at least this "
                        "frequently in fetched-entry counts (default 50)")
    args = p.parse_args()

    corpus_path = _resolve_corpus_path(args.corpus)
    run(corpus_path, args.max_bytes, args.lang, args.min_len,
        args.rewrite_every)


if __name__ == "__main__":
    main()
