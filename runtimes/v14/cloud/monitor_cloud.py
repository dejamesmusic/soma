#!/usr/bin/env python3
"""small status and dream monitor for an unattended soma cloud run."""

import argparse
import json
import shutil
import time
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def count(value):
    value = float(value or 0)
    for suffix, scale in (("b", 1e9), ("m", 1e6), ("k", 1e3)):
        if value >= scale:
            return f"{value / scale:.1f}{suffix}"
    return str(int(value))


def status(root):
    curriculum = read_json(root / "curriculum_status.json")
    training = read_json(root / "logs" / "status.json")
    free = shutil.disk_usage(root).free / 1024 ** 3
    stage = curriculum.get("stage", training.get("stage", "starting"))
    event = training.get("event", curriculum.get("event", "starting"))
    fields = [stage, event]
    if "bpb" in training:
        fields.extend([
            f"{training['bpb']:.3f} bpb",
            f"{100 * training.get('token_accuracy', 0):.1f}% top-1",
            f"{count(training.get('source_bps'))} source b/s",
            f"stride {training.get('stride', 0)}",
            f"{count(training.get('bytes_seen'))} seen",
        ])
    fields.append(f"{free:.1f}gb free")
    print(" · ".join(fields), flush=True)


def dreams(path, lines):
    position = max(0, path.stat().st_size - lines) if path.exists() else 0
    while True:
        if path.exists():
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(position)
                text = handle.read()
                position = handle.tell()
            if text:
                print(text, end="", flush=True)
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("run"))
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--dreams", action="store_true")
    args = parser.parse_args()
    root = args.workdir.expanduser().resolve()
    if args.dreams:
        dreams(root / "logs" / "dreams.txt", 64 * 1024)
        return
    while True:
        status(root)
        if not args.follow:
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
