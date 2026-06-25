#!/usr/bin/env python3
"""throughput benchmark for soma v12 train paths.

compares the legacy one-observation-at-a-time strided collector against
the block collector used by default. this is intentionally small enough
to run on a laptop, but the setting exercises the same auto-decimated
path used by long cloud runs.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soma_v12 import SOMA


def run_once(args, label, legacy):
    if legacy:
        os.environ["SOMA_LEGACY_STRIDED"] = "1"
    else:
        os.environ.pop("SOMA_LEGACY_STRIDED", None)

    model = SOMA(
        n_bands=args.bands,
        hidden_dim=args.hidden,
        n_layers=args.layers,
        batch_size=args.batch,
        device=args.device,
        lr=1.0,
        max_change=1.0,
        lr_auto=True,
        lr_base=1.0,
        max_change_auto=True,
        max_change_base=1.0,
        decimation_auto=True,
        decimation_range=1.0,
        auto_mode="full spectrum",
    )
    start = time.time()
    model.train(args.work_file, epochs=1)
    elapsed = time.time() - start
    bps = args.bytes / max(elapsed, 1e-9)
    print(
        f"result {label}: {bps:,.0f} b/s · {elapsed:.2f}s "
        f"· stride {model._stride} · target {model._spectral_target_band:.2f}",
        flush=True,
    )
    return bps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=(
        "/Users/jamesblight/Library/Application Support/soma/data/"
        "childrens_books.txt"))
    ap.add_argument("--bytes", type=int, default=80_000)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--work-file", default="/tmp/soma_throughput_benchmark.bin")
    ap.add_argument("--new-only", action="store_true")
    args = ap.parse_args()

    data = np.fromfile(args.data, dtype=np.uint8)[:args.bytes]
    data.tofile(args.work_file)
    args.bytes = int(len(data))

    print(
        f"benchmark: {args.bytes:,} bytes · bands={args.bands} "
        f"hidden={args.hidden} layers={args.layers} batch={args.batch}",
        flush=True,
    )
    if not args.new_only:
        legacy = run_once(args, "legacy", True)
    else:
        legacy = None
    block = run_once(args, "block", False)
    if legacy:
        print(f"speedup: {block / max(legacy, 1e-9):.2f}x")


if __name__ == "__main__":
    main()
