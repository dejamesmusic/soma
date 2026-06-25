#!/usr/bin/env python3
"""resumable soma v12 cloud trainer.

designed for vast/ssh runs:
- writes metrics.jsonl, dreams.txt, status.json
- saves rolling latest/backup checkpoints by time
- resumes from an existing checkpoint when present
- keeps all paths local to the bundle by default
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import soma_v12


def parse_auto(text, default_mode=None, default_value=1.0):
    text = str(text).strip().lower()
    if text in ("auto", "level"):
        return default_value, True, default_value, "level"
    for mode in ("io2", "full spectrum", "spectral", "progress"):
        if text.startswith(mode):
            rest = text[len(mode):].strip()
            val = float(rest) if rest else default_value
            return val, True, val, mode
    return float(text), False, float(text), default_mode


def fmt(n):
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}b"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}m"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def device_report():
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["total_vram_gb"] = round(props.total_memory / 1024**3, 2)
    info["soma_cuda_trace"] = os.environ.get("SOMA_CUDA_TRACE") == "1"
    return info


def build_model(args):
    lr, lr_auto, lr_base, lr_mode = parse_auto(args.lr)
    mc, mc_auto, mc_base, mc_mode = parse_auto(args.max_change)
    auto_mode = args.auto_mode
    for mode in ("io2", "full spectrum", "spectral", "progress"):
        if mode in (lr_mode, mc_mode):
            auto_mode = mode
            break
    model = soma_v12.SOMA(
        n_bands=args.bands,
        base=args.base,
        hidden_dim=args.hidden,
        n_layers=args.layers,
        direct_readout=False,
        scale_gate=False,
        clock=1,
        auto_mode=auto_mode,
        lr=lr,
        max_change=mc,
        lr_auto=lr_auto,
        lr_base=lr_base,
        max_change_auto=mc_auto,
        max_change_base=mc_base,
        weight_decay=0.0,
        batch_size=args.batch,
        decimation_auto=True,
        decimation_range=args.decimation,
        device=args.device,
    )
    return model


def load_or_create(args, ckpt):
    if ckpt.exists() and args.resume:
        cfg = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = soma_v12.SOMA(
            cfg.get("n_bands", args.bands),
            base=cfg.get("base", args.base),
            hidden_dim=cfg.get("hidden_dim", args.hidden),
            n_layers=cfg.get("n_layers", args.layers),
            direct_readout=bool(cfg.get("direct_readout", False)),
            scale_gate=bool(cfg.get("scale_gate", False)),
            clock=cfg.get("clock", 1),
            batch_size=cfg.get("batch_size", args.batch),
            device=args.device,
        )
        model.load(str(ckpt))
        lr, lr_auto, lr_base, lr_mode = parse_auto(args.lr)
        mc, mc_auto, mc_base, mc_mode = parse_auto(args.max_change)
        model.lr_auto = lr_auto
        model.lr_base = lr_base
        model.max_change_auto = mc_auto
        model.max_change_base = mc_base
        model.auto_mode = ("io2" if "io2" in (lr_mode, mc_mode)
                           else ("full spectrum" if "full spectrum" in (lr_mode, mc_mode)
                           else ("spectral" if "spectral" in (lr_mode, mc_mode)
                                 else ("progress" if "progress" in (lr_mode, mc_mode)
                                       else args.auto_mode))))
        if not lr_auto:
            model.lr = lr
        if not mc_auto:
            model.max_change = mc
        model.batch_size = args.batch
        model.decimation_auto = True
        model.decimation_range = args.decimation
        model.decimation_band = 0
        model._update_decimation()
        return model, True
    return build_model(args), False


def save_rolling(model, ckpt, backups):
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if ckpt.exists() and backups > 0:
        for i in range(backups - 1, 0, -1):
            src = ckpt.with_name(f"{ckpt.stem}.bak{i}{ckpt.suffix}")
            dst = ckpt.with_name(f"{ckpt.stem}.bak{i + 1}{ckpt.suffix}")
            if src.exists():
                src.replace(dst)
        shutil.copy2(ckpt, ckpt.with_name(f"{ckpt.stem}.bak1{ckpt.suffix}"))
    model.save(str(ckpt))


def train(args):
    work = Path(args.workdir).resolve()
    data_path = Path(args.data).resolve()
    ckpt = (work / "checkpoints" / args.checkpoint).resolve()
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    metrics_path = logs / "metrics.jsonl"
    dream_path = logs / "dreams.txt"
    status_path = logs / "status.json"
    env_path = logs / "environment.json"
    atomic_json(env_path, device_report())

    corpus = np.fromfile(data_path, dtype=np.uint8)
    if args.max_bytes > 0:
        corpus = corpus[:args.max_bytes]

    model, resumed = load_or_create(args, ckpt)
    model.print_config()

    state = {
        "event": "start",
        "time": time.time(),
        "data": str(data_path),
        "checkpoint": str(ckpt),
        "resumed": resumed,
        "corpus_bytes": int(len(corpus)),
        "args": vars(args),
        "device": device_report(),
    }
    append_jsonl(metrics_path, state)
    atomic_json(status_path, state)

    t0 = time.time()
    last_report = t0
    last_save = t0
    batch_id = 0
    total_loss = 0.0
    total_acc = 0
    total_samples = 0
    start_seen = int(model.bytes_seen)

    try:
        for epoch in range(args.epochs):
            pos = 0
            while pos < len(corpus):
                X0, yt, pos = model._collect_strided_batch(
                    corpus, pos, len(corpus), args.batch)
                if X0 is None:
                    break
                n = int(yt.shape[0])
                loss, acc = model._train_batch(X0, yt, n)
                model._update_auto_lr(loss, n)
                loss_v = float(loss.detach().cpu().item())
                acc_v = int(acc.detach().cpu().item())
                total_loss += loss_v
                total_acc += acc_v
                total_samples += n
                batch_id += 1

                now = time.time()
                if args.dream_every and batch_id % args.dream_every == 0:
                    length = args.dream_length
                    if args.dream_auto:
                        length = model._auto_dream_length(loss, n, args.dream_length)
                    if length > 0:
                        text = "".join(model.generate(
                            length=length, temperature=args.temperature))
                        with open(dream_path, "a", encoding="utf-8") as f:
                            f.write(
                                f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} · "
                                f"batch {batch_id} · {fmt(model.bytes_seen)} seen\n"
                            )
                            f.write(text + "\n")

                if now - last_report >= args.report_seconds or pos >= len(corpus):
                    avg = total_loss / max(1, total_samples)
                    report = {
                        "event": "report",
                        "time": now,
                        "epoch": epoch + 1,
                        "batch": batch_id,
                        "pos": int(pos),
                        "corpus_bytes": int(len(corpus)),
                        "bytes_seen": int(model.bytes_seen),
                        "new_bytes_seen": int(model.bytes_seen - start_seen),
                        "samples": int(total_samples),
                        "loss_nats": avg,
                        "bpb": avg / math.log(2),
                        "accuracy": total_acc / max(1, total_samples),
                        "bps_wall": (model.bytes_seen - start_seen) / max(1e-6, now - t0),
                        "sample_bps": total_samples / max(1e-6, now - t0),
                        "lr": float(model.lr),
                        "max_change": float(model.max_change),
                        "decimation_band": float(model.decimation_band),
                        "stride": int(model._stride),
                        "spectral_target_band": float(getattr(model, "_spectral_target_band", 0.0)),
                        "input_coherence": float(getattr(model, "_input_coherence", 0.0)),
                        "output_coherence": float(getattr(model, "_output_coherence", 0.0)),
                        "input_trust": float(getattr(model, "_input_trust", 0.0)),
                        "error_concentration": float(getattr(model, "_error_concentration", 0.0)),
                        "io2_plasticity": float(getattr(model, "_io2_plasticity", 0.0)),
                    }
                    append_jsonl(metrics_path, report)
                    atomic_json(status_path, report)
                    print(
                        f"{fmt(report['new_bytes_seen'])} new · "
                        f"{avg:.3f} nats · {100*report['accuracy']:.1f}% · "
                        f"{fmt(report['bps_wall'])} b/s · stride {model._stride} · "
                        f"band {model.decimation_band:.2f}",
                        flush=True,
                    )
                    last_report = now

                if args.save_minutes and now - last_save >= args.save_minutes * 60:
                    save_rolling(model, ckpt, args.backups)
                    last_save = now

            if args.save_each_epoch:
                save_rolling(model, ckpt, args.backups)

        save_rolling(model, ckpt, args.backups)
        append_jsonl(metrics_path, {"event": "done", "time": time.time(), "bytes_seen": model.bytes_seen})
    except KeyboardInterrupt:
        save_rolling(model, ckpt, args.backups)
        append_jsonl(metrics_path, {"event": "keyboard_interrupt", "time": time.time(), "bytes_seen": model.bytes_seen})
        raise
    except Exception as exc:
        try:
            emergency = ckpt.with_name(f"{ckpt.stem}.crash{ckpt.suffix}")
            model.save(str(emergency))
        finally:
            append_jsonl(metrics_path, {
                "event": "crash",
                "time": time.time(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default="/workspace/soma_v12")
    p.add_argument("--data", default="/workspace/soma_v12/data/enwik9")
    p.add_argument("--checkpoint", default="soma_cloud.pt")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-bytes", type=int, default=0)
    p.add_argument("--bands", type=int, default=50)
    p.add_argument("--base", type=float, default=1.6180)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", default="auto")
    p.add_argument("--max-change", default="auto")
    p.add_argument("--auto-mode", default="io2",
                   choices=("level", "progress", "spectral",
                            "full spectrum", "io2"))
    p.add_argument("--decimation", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-minutes", type=float, default=30.0)
    p.add_argument("--backups", type=int, default=2)
    p.add_argument("--save-each-epoch", action="store_true")
    p.add_argument("--report-seconds", type=float, default=60.0)
    p.add_argument("--dream-every", type=int, default=50)
    p.add_argument("--dream-length", type=int, default=200)
    p.add_argument("--dream-auto", action="store_true", default=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--cuda-trace", action="store_true",
                   help="experimental: keep trace banks on cuda in float64")
    args = p.parse_args()
    if args.cuda_trace:
        os.environ["SOMA_CUDA_TRACE"] = "1"
    train(args)


if __name__ == "__main__":
    main()
