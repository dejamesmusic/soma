#!/usr/bin/env python3
"""resumable, observable cloud trainer for soma v14."""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import soma_v14 as soma  # noqa: e402


def atomic_json(path, value):
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_json(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def rotate_checkpoint(path):
    path = Path(path)
    if not path.exists():
        return None
    previous = path.with_name(f"{path.stem}.previous{path.suffix}")
    temporary = Path(f"{previous}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(path, temporary)
    except OSError:
        shutil.copy2(path, temporary)
    os.replace(temporary, previous)
    return previous


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
    parser.add_argument("--checkpoint", default="soma_v14.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bands", type=int, default=48)
    parser.add_argument("--base", type=float, default=soma.PHI)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", default="auto 0.0003")
    parser.add_argument("--decimation", type=float, default=1.0)
    parser.add_argument("--update-interval", type=int, default=1)
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
    parser.add_argument("--stage", default="")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    args = parser.parse_args()

    work = Path(args.workdir).expanduser().resolve()
    checkpoint = work / "checkpoints" / args.checkpoint
    logs = work / "logs"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    metrics = logs / "metrics.jsonl"
    dreams = logs / "dreams.txt"
    status = logs / "status.json"
    recovery = logs / "recovery.json"
    atomic_json(logs / "environment.json", environment())

    lr, lr_auto, lr_base = soma._parse_auto_or_float(args.lr, 0.0003)
    if args.resume and checkpoint.exists():
        model = soma.SOMA.from_checkpoint(
            checkpoint, device=args.device)
        model.batch_size = args.batch
        model.lr = lr
        model.lr_base = lr_base
        model.lr_auto = lr_auto
        model.grad_clip = args.grad_clip
        model.trainer.grad_clip = args.grad_clip
        model.trainer.update_interval = max(1, args.update_interval)
        model.trainer.decimation_gain = max(0.0, args.decimation)
        model._pulse_duty = 1.0 / model.trainer.prediction_stride
        model._set_optimizer_lrs()
        resumed = True
    else:
        model = soma.SOMA(
            n_bands=args.bands,
            base=args.base, batch_size=args.batch,
            lr=lr, lr_base=lr_base, lr_auto=lr_auto,
            decimation_gain=args.decimation,
            update_interval=args.update_interval,
            grad_clip=args.grad_clip, device=args.device)
        resumed = False

    started = time.time()
    session_start_bytes = model.bytes_seen
    original_report = model._report
    original_save = model.save
    latest_position = int(args.start_byte)
    last_report_position = int(args.start_byte)
    last_save_position = int(args.start_byte)

    def save(path):
        free = shutil.disk_usage(checkpoint.parent).free
        if free < args.min_free_gb * 1024 ** 3:
            raise RuntimeError(
                f"only {free / 1024**3:.1f}gb free; refusing checkpoint save")
        rotate_checkpoint(checkpoint)
        original_save(path)
        value = {
            "event": "checkpoint",
            "time": time.time(),
            "stage": args.stage,
            "data": str(Path(args.data).resolve()),
            "checkpoint": str(checkpoint),
            "position": latest_position,
            "bytes_seen": model.bytes_seen,
            "tokens_seen": model.tokens_seen,
        }
        atomic_json(recovery, value)
        append_json(metrics, value)

    model.save = save

    def report(epoch, epochs, reader, totals, began):
        nonlocal latest_position, last_report_position, last_save_position
        latest_position = int(reader.position)
        report_due = (
            latest_position - last_report_position >= args.report_bytes
            or latest_position >= int(reader.limit)
        )
        save_due = latest_position - last_save_position >= args.save_bytes
        if save_due:
            model.save(checkpoint)
            last_save_position = latest_position
        if not report_due:
            return
        original_report(epoch, epochs, reader, totals, began)
        nats = totals["nll"] / max(1.0, totals["loss_bytes"])
        if not math.isfinite(nats):
            raise FloatingPointError(f"non-finite loss: {nats}")
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
            "token_loss": totals["nll"] / max(
                1.0, totals["loss_tokens"]),
            "token_accuracy": totals["correct"] / max(1, totals["samples"]),
            "source_bps": ((model.bytes_seen - session_start_bytes)
                           / max(1e-9, time.time() - started)),
            "lr": model.lr,
            "bytes_per_token": model._bytes_per_token,
            "pulse_duty": model._pulse_duty,
            "io2": model._io2_plasticity,
            "prediction_stride": model._stride,
            "byte_stride": model._byte_stride,
            "input_coherence": model._input_coherence,
            "output_coherence": model._output_coherence,
            "error_energy": model._error_energy,
            "error_band": model._spectral_target_band,
            "stage": args.stage,
        }
        append_json(metrics, value)
        atomic_json(status, value)
        last_report_position = latest_position

    def dream(text, batch, seen):
        with dreams.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} · "
                f"batch {batch} · {seen} source bytes\n{text}\n")

    model._report = report
    start = {
        "event": "start", "time": started, "resumed": resumed,
        "stage": args.stage,
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(checkpoint), "environment": environment(),
        "params": model.params(), "tokenizer": model.tokenizer.sha256,
        "controller": "io2", "decimation_gain": args.decimation,
    }
    append_json(metrics, start)
    atomic_json(status, start)
    try:
        model.train(
            args.data, epochs=args.epochs, save_path=str(checkpoint),
            start_byte=args.start_byte, max_bytes=args.max_bytes,
            report_every=1, save_every=0,
            dream_every_batches=args.dream_every,
            dream_length=args.dream_length,
            dream_temperature=args.temperature,
            dream_callback=dream)
        complete = {
            "event": "complete",
            "time": time.time(),
            "stage": args.stage,
            "data": str(Path(args.data).resolve()),
            "checkpoint": str(checkpoint),
            "position": latest_position,
            "bytes_seen": model.bytes_seen,
            "tokens_seen": model.tokens_seen,
        }
        append_json(metrics, complete)
        atomic_json(status, complete)
    except KeyboardInterrupt:
        model.save(checkpoint)
        stopped = {"event": "stopped", "time": time.time(),
                   "stage": args.stage, "bytes_seen": model.bytes_seen,
                   "position": latest_position}
        append_json(metrics, stopped)
        atomic_json(status, stopped)


if __name__ == "__main__":
    main()
