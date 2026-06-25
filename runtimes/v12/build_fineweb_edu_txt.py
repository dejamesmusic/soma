#!/usr/bin/env python3
"""
build_fineweb_edu_txt.py

Stream FineWeb-Edu from Hugging Face and write raw UTF-8 .txt files
for byte-level soma training.

Example:
  python build_fineweb_edu_txt.py --target-gb 2 --out-dir priant_fwe_2gb

Notes:
- This does NOT tokenize.
- It writes plain text.
- It uses streaming=True, so it does not download the whole dataset first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--split", default="train")
    p.add_argument("--out-dir", default="priant_fwe_txt")
    p.add_argument("--prefix", default="fineweb_edu")
    p.add_argument("--target-gb", type=float, default=2.0)
    p.add_argument("--chunk-mb", type=int, default=512)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--max-chars", type=int, default=200_000)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress-mb", type=int, default=100)
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


RE_WS = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    # Keep this intentionally conservative. We want raw text, not aggressive normalization.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(RE_WS.sub(" ", line).rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def looks_bad(text: str, min_chars: int, max_chars: int) -> bool:
    if not text:
        return True
    if len(text) < min_chars:
        return True
    if len(text) > max_chars:
        return True

    lower = text[:5000].lower()

    # Common junk / boilerplate filters. Keep light; FineWeb-Edu is already filtered.
    bad_markers = [
        "lorem ipsum",
        "enable javascript",
        "please enable cookies",
        "access denied",
        "cloudflare ray id",
    ]
    if any(m in lower for m in bad_markers):
        return True

    # Avoid extremely markup/code-heavy samples.
    sample = text[:10000]
    if sample.count("{") + sample.count("}") > 120:
        return True
    if sample.count("<") + sample.count(">") > 120:
        return True

    # Avoid tables/menus made of tiny fragments.
    lines = [ln for ln in sample.splitlines() if ln.strip()]
    if len(lines) > 30:
        short = sum(1 for ln in lines if len(ln.strip()) < 30)
        if short / max(len(lines), 1) > 0.75:
            return True

    return False


def open_chunk(out_dir: Path, prefix: str, index: int):
    path = out_dir / f"{prefix}_{index:04d}.txt"
    return path, open(path, "w", encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = int(args.target_gb * 1024**3)
    chunk_bytes_limit = int(args.chunk_mb * 1024**2)
    progress_bytes = int(args.progress_mb * 1024**2)

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "target_gb": args.target_gb,
        "chunk_mb": args.chunk_mb,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "shuffle_buffer": None if args.no_shuffle else args.shuffle_buffer,
        "seed": args.seed,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": [],
    }

    print(f"loading {args.dataset} split={args.split} streaming=True")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    if not args.no_shuffle:
        print(f"shuffling stream with buffer_size={args.shuffle_buffer}, seed={args.seed}")
        ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    if args.dry_run:
        print("dry run: first 3 cleaned samples")
        for i, row in enumerate(ds):
            text = clean_text(row.get("text", ""))
            if looks_bad(text, args.min_chars, args.max_chars):
                continue
            print("=" * 80)
            print(text[:2000])
            if i >= 2:
                break
        return 0

    total_written = 0
    current_written = 0
    accepted = 0
    skipped = 0
    chunk_index = 0
    next_progress = progress_bytes

    path, fh = open_chunk(out_dir, args.prefix, chunk_index)
    hasher = hashlib.sha256()

    try:
        for row in ds:
            text = clean_text(row.get("text", ""))
            if looks_bad(text, args.min_chars, args.max_chars):
                skipped += 1
                continue

            blob = (text + "\n\n").encode("utf-8")

            if current_written > 0 and current_written + len(blob) > chunk_bytes_limit:
                fh.close()
                manifest["files"].append({
                    "path": str(path),
                    "bytes": current_written,
                    "sha256": hasher.hexdigest(),
                })
                print(f"wrote chunk {path} ({current_written / 1024**2:.1f} MB)")

                chunk_index += 1
                path, fh = open_chunk(out_dir, args.prefix, chunk_index)
                hasher = hashlib.sha256()
                current_written = 0

            fh.write(blob.decode("utf-8"))
            hasher.update(blob)
            current_written += len(blob)
            total_written += len(blob)
            accepted += 1

            if total_written >= next_progress:
                print(
                    f"{total_written / 1024**3:.2f} GB written "
                    f"accepted={accepted:,} skipped={skipped:,}"
                )
                next_progress += progress_bytes

            if total_written >= target_bytes:
                break

    finally:
        fh.close()

    if current_written > 0:
        manifest["files"].append({
            "path": str(path),
            "bytes": current_written,
            "sha256": hasher.hexdigest(),
        })

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["total_bytes"] = total_written
    manifest["accepted_samples"] = accepted
    manifest["skipped_samples"] = skipped

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("done")
    print(f"output dir: {out_dir}")
    print(f"total: {total_written / 1024**3:.2f} GB")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
