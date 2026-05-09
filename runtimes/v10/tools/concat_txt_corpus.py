#!/usr/bin/env python3
"""
concat_txt_corpus.py

Concatenate .txt chunks into one large .txt file, preserving byte order.
Useful if your soma training path prefers a single file.

Example:
  python concat_txt_corpus.py --input-dir priant_fwe_2gb --output priant_fwe_2gb.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pattern", default="*.txt")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))

    if not files:
        raise SystemExit(f"no files matching {args.pattern} in {input_dir}")

    out = Path(args.output)
    total = 0
    with open(out, "wb") as w:
        for path in files:
            data = path.read_bytes()
            w.write(data)
            if not data.endswith(b"\n\n"):
                w.write(b"\n\n")
            total += len(data)
            print(f"added {path} ({len(data) / 1024**2:.1f} MB)")

    print(f"wrote {out} ({total / 1024**3:.2f} GB)")


if __name__ == "__main__":
    main()
