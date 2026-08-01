#!/usr/bin/env python3
"""resumable, observable cloud trainer for soma v12.3."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import soma_v12_3 as soma  # noqa: e402


def atomic_json(path, value):
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_json(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def environment():
    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        report["device"] = torch.cuda.get_device_name(0)
        report["vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--checkpoint", default="soma_v12_3.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bands", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--base", type=float, default=1.6180)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", default="auto 0.002")
    parser.add_argument(
        "--decimation", type=float, default=1.0,
        help="gain applied to io2's decimation decision")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--start-byte", type=int, default=0)
    parser.add_argument("--max-bytes", type=int, default=0)
    parser.add_argument("--report-bytes", type=int, default=5_000_000)
    parser.add_argument("--save-bytes", type=int, default=50_000_000)
    parser.add_argument("--dream-every", type=int, default=64)
    parser.add_argument(
        "--dream-tokens", "--dream-length", dest="dream_length",
        default="128")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    work = Path(args.workdir).expanduser().resolve()
    checkpoint = work / "checkpoints" / args.checkpoint
    logs = work / "logs"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    metrics = logs / "metrics.jsonl"
    dreams = logs / "dreams.txt"
    status = logs / "status.json"
    atomic_json(logs / "environment.json", environment())

    lr, lr_auto, lr_base = soma._parse_auto_or_float(args.lr, 0.002)
    if args.resume and checkpoint.exists():
        model = soma.SOMA(device=args.device).load(checkpoint)
        model.batch_size = args.batch
        model.lr = lr
        model.lr_base = lr_base
        model.lr_auto = lr_auto
        model.auto_mode = soma.DEFAULT_AUTO_MODE
        model.decimation_range = max(0.0, args.decimation)
        model.grad_clip = args.grad_clip
        model._set_optimizer_lrs()
        model._update_decimation()
        resumed = True
    else:
        model = soma.SOMA(
            n_bands=args.bands, depth=args.depth,
            base=args.base, batch_size=args.batch,
            lr=lr, lr_base=lr_base, lr_auto=lr_auto,
            auto_mode=soma.DEFAULT_AUTO_MODE, decimation_range=args.decimation,
            grad_clip=args.grad_clip, device=args.device)
        resumed = False

    started = time.time()
    session_start_bytes = model.bytes_seen
    original_report = model._report

    def report(epoch, epochs, reader, totals, began):
        original_report(epoch, epochs, reader, totals, began)
        nats = totals["nll"] / max(1, totals["bytes"])
        value = {
            "event": "report",
            "time": time.time(),
            "epoch": epoch + 1,
            "epochs": epochs,
            "position": reader.position,
            "bytes_seen": model.bytes_seen,
            "tokens_seen": model.tokens_seen,
            "nats_per_byte": nats,
            "bpb": nats / math.log(2.0),
            "token_loss": totals["nll"] / max(1, totals["tokens"]),
            "token_accuracy": totals["correct"] / max(1, totals["samples"]),
            "source_bps": ((model.bytes_seen - session_start_bytes)
                           / max(1e-9, time.time() - started)),
            "lr": model.lr,
            "stride": model._stride,
            "byte_stride": model._byte_stride,
            "bytes_per_token": model._bytes_per_token,
            "io2": model._io2_plasticity,
            "compression": model._row_compression_fraction,
            "rowclip": model._row_clip_fraction,
        }
        append_json(metrics, value)
        atomic_json(status, value)

    def dream(text, batch, seen):
        with dreams.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} · "
                f"batch {batch} · {seen} source bytes\n{text}\n")

    model._report = report
    start = {
        "event": "start", "time": started, "resumed": resumed,
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(checkpoint), "environment": environment(),
        "params": model.params(), "tokenizer": model.tokenizer.model_sha256,
    }
    append_json(metrics, start)
    atomic_json(status, start)
    try:
        model.train(
            args.data, epochs=args.epochs, save_path=str(checkpoint),
            start_byte=args.start_byte, max_bytes=args.max_bytes,
            report_every=args.report_bytes, save_every=args.save_bytes,
            dream_every_batches=args.dream_every,
            dream_length=args.dream_length,
            dream_temperature=args.temperature,
            dream_callback=dream)
    except KeyboardInterrupt:
        model.save(checkpoint)
        stopped = {"event": "stopped", "time": time.time(),
                   "bytes_seen": model.bytes_seen}
        append_json(metrics, stopped)
        atomic_json(status, stopped)


if __name__ == "__main__":
    main()
