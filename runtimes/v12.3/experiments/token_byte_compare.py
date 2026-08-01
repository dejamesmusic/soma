#!/usr/bin/env python3
"""memory-matched v12.2 byte vs v12.3.1 token comparison.

the score shared by both alphabets is negative log likelihood divided by the
original source bytes. deterministic tokenization makes this a valid code
length comparison even though token accuracy and byte accuracy are different
objects.
"""

import argparse
import gc
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from soma_v12_2 import SerialSOMA
from soma_v12_3 import TokenSOMA


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def release(model):
    del model
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def heldout_file(corpus, start, length):
    handle = tempfile.NamedTemporaryFile(
        prefix="soma-heldout-", suffix=".txt", delete=False)
    try:
        with Path(corpus).open("rb") as source:
            source.seek(start)
            remaining = int(length)
            while remaining > 0:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                handle.write(chunk)
                remaining -= len(chunk)
    finally:
        handle.close()
    return handle.name


def run_byte(args, heldout):
    seed_everything(args.seed)
    model = SerialSOMA(
        n_bands=args.bands,
        hidden_dim=args.byte_hidden,
        depth=args.depth,
        base=args.base,
        batch_size=args.batch,
        lr=args.lr,
        lr_base=args.lr,
        lr_auto=False,
        grad_clip=args.grad_clip,
        auto_mode="off",
        decimation_range=0.0,
        row_norm="off",
        device=args.device,
    )
    started = time.time()
    model.train(
        str(args.corpus),
        save_path="",
        max_bytes=args.train_bytes,
        report_every=args.report_every,
        save_every=0,
        dream_every_batches=0,
    )
    train_seconds = time.time() - started
    evaluation = model.evaluate(heldout, max_bytes=args.eval_bytes)
    model.bank.reset()
    model.ingest_prompt(args.prompt + "\n", online=False)
    dream = "".join(model.generate(
        length=args.dream_bytes,
        temperature=args.temperature,
        prelude_source="comparison"))
    result = {
        "kind": "v12.2-byte",
        "params": model.params(),
        "train_seconds": train_seconds,
        "train_source_bps": args.train_bytes / max(train_seconds, 1e-9),
        "eval_nats_per_byte": float(evaluation),
        "eval_bits_per_byte": None,
        "eval_accuracy": None,
        "dream": dream,
    }
    if result["eval_nats_per_byte"] is not None:
        result["eval_bits_per_byte"] = (
            result["eval_nats_per_byte"] / math.log(2.0))
    release(model)
    return result


def run_token(args):
    seed_everything(args.seed)
    model = TokenSOMA(
        n_bands=args.bands,
        hidden_dim=args.token_hidden,
        depth=args.depth,
        base=args.base,
        batch_size=args.batch,
        lr=args.lr,
        lr_base=args.lr,
        lr_auto=False,
        grad_clip=args.grad_clip,
        auto_mode="off",
        decimation_range=0.0,
        row_norm="off",
        device=args.device,
    )
    started = time.time()
    model.train(
        str(args.corpus),
        save_path="",
        max_bytes=args.train_bytes,
        report_every=args.report_every,
        save_every=0,
        dream_every_batches=0,
    )
    train_seconds = time.time() - started
    evaluation = model.evaluate(
        str(args.corpus),
        start_byte=args.train_bytes,
        max_bytes=args.eval_bytes,
    )
    model.bank.reset()
    model.ingest_prompt(args.prompt + "\n", online=False)
    dream = model.generate_text(
        length=args.dream_bytes,
        temperature=args.temperature,
        prelude=False,
    )
    result = {
        "kind": "v12.3.1-token",
        "params": model.params(),
        "train_seconds": train_seconds,
        "train_source_bps": args.train_bytes / max(train_seconds, 1e-9),
        "eval_nats_per_byte": evaluation["nats_per_byte"],
        "eval_bits_per_byte": evaluation["bits_per_byte"],
        "eval_accuracy": evaluation["accuracy"],
        "tokens_per_source_byte": (
            evaluation["tokens"] / max(1, evaluation["bytes"])),
        "tokenizer_sha256": model.tokenizer.model_sha256,
        "dream": dream,
    }
    release(model)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--train-bytes", type=int, default=5_000_000)
    parser.add_argument("--eval-bytes", type=int, default=1_000_000)
    parser.add_argument("--bands", type=int, default=12)
    parser.add_argument("--byte-hidden", type=int, default=858)
    parser.add_argument("--token-hidden", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--base", type=float, default=1.6180)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--report-every", type=int, default=1_000_000)
    parser.add_argument(
        "--dream-bytes", "--dream-tokens", dest="dream_bytes",
        type=int, default=360)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--prompt", default="once upon a time")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path,
                        default=Path("/private/tmp/soma-v123-compare.json"))
    args = parser.parse_args()
    args.corpus = args.corpus.expanduser().resolve()
    heldout = heldout_file(
        args.corpus, args.train_bytes, args.eval_bytes)
    try:
        results = [run_byte(args, heldout), run_token(args)]
    finally:
        os.unlink(heldout)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print("\ncomparison")
    for result in results:
        print(
            f"  {result['kind']:12s} · {result['params']/1e6:.3f}m params "
            f"· {result['eval_bits_per_byte']:.3f} bpb "
            f"· {result['train_source_bps']:,.0f} source b/s")
        print(f"    {result['dream'][:500]!r}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
