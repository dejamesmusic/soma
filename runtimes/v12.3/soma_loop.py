#!/usr/bin/env python3
"""perpetual v12.3.17 training over the moving head of a corpus."""

import argparse
import signal
import sys
import time
from pathlib import Path

import soma_v12_3 as soma


stop_requested = False


def on_signal(_signal, _frame):
    global stop_requested
    if stop_requested:
        raise KeyboardInterrupt
    stop_requested = True
    print("\n  stop requested; finishing the current pass")


def fmt(value):
    value = float(value)
    for suffix in ("b", "kb", "mb", "gb"):
        if value < 1000:
            return f"{value:.1f}{suffix}"
        value /= 1000
    return f"{value:.1f}tb"


def snapshot(source, destination, limit):
    written = 0
    with source.open("rb") as reader, destination.open("wb") as writer:
        remaining = min(int(limit), source.stat().st_size)
        while remaining > 0:
            chunk = reader.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            writer.write(chunk)
            written += len(chunk)
            remaining -= len(chunk)
    return written


def main():
    parser = argparse.ArgumentParser(
        description="train soma v12.3.17 forever on a rolling corpus")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--head-bytes", type=int, default=50_000_000)
    parser.add_argument("--cycle-pause", type=float, default=60.0)
    parser.add_argument("--min-corpus-bytes", type=int, default=1_000_000)
    parser.add_argument("--bands", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--base", type=float, default=1.6180)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", default="auto 0.002")
    parser.add_argument(
        "--decimation", type=float, default=1.0,
        help="gain applied to io2's decimation decision")
    parser.add_argument("--dream-every", type=int, default=64)
    parser.add_argument(
        "--dream-tokens", "--dream-length", dest="dream_length",
        default="128")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    args.corpus = args.corpus.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.corpus.exists():
        raise SystemExit(f"corpus not found: {args.corpus}")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lr, lr_auto, lr_base = soma._parse_auto_or_float(args.lr, 0.002)
    if args.checkpoint.exists():
        model = soma.SOMA(device=args.device).load(args.checkpoint)
        model.batch_size = args.batch
        model.lr = lr
        model.lr_base = lr_base
        model.lr_auto = lr_auto
        model.auto_mode = soma.DEFAULT_AUTO_MODE
        model.decimation_range = max(0.0, args.decimation)
        model._set_optimizer_lrs()
        model._update_decimation()
        print(f"  resumed {args.checkpoint.name} · {fmt(model.bytes_seen)} seen")
    else:
        model = soma.SOMA(
            n_bands=args.bands, depth=args.depth,
            base=args.base, batch_size=args.batch,
            lr=lr, lr_base=lr_base, lr_auto=lr_auto,
            auto_mode=soma.DEFAULT_AUTO_MODE, decimation_range=args.decimation,
            device=args.device)
        print(f"  fresh {model.params()/1e6:.1f}m parameter model")

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    temporary = args.checkpoint.parent / f".{args.checkpoint.stem}_head.tmp"
    cycle = 0
    try:
        while not stop_requested:
            size = args.corpus.stat().st_size
            if size < args.min_corpus_bytes:
                print(f"  corpus {fmt(size)}; waiting")
                time.sleep(args.cycle_pause)
                continue
            cycle += 1
            count = snapshot(args.corpus, temporary, args.head_bytes)
            print(f"\n  cycle {cycle} · {fmt(count)} current corpus head")
            model.train(
                str(temporary), save_path=str(args.checkpoint),
                report_every=1_000_000, save_every=0,
                dream_every_batches=args.dream_every,
                dream_length=args.dream_length,
                dream_temperature=args.temperature)
            if not stop_requested:
                time.sleep(max(0.0, args.cycle_pause))
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        model.save(args.checkpoint)


if __name__ == "__main__":
    main()
