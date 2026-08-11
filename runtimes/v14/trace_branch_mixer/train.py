"""command-line training harness for the trace branch mixer."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .core import TraceBranchConfig
from .runtime import ReferenceTokenizer, TraceBranchTrainer


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = RUNTIME_ROOT / "tokenizer" / "soma-1k-v1.model"


def config_from_args(args: argparse.Namespace) -> TraceBranchConfig:
    if args.profile == "local":
        values = {
            "bands": 48,
            "symbol_dim": 256,
            "scale_rank": 64,
            "scale_depth": 3,
            "summaries": 8,
            "model_dim": 1024,
            "trunk_depth": 4,
            "expansion_dim": 2048,
        }
    else:
        values = {
            "bands": 32,
            "symbol_dim": 128,
            "scale_rank": 32,
            "scale_depth": 2,
            "summaries": 4,
            "model_dim": 512,
            "trunk_depth": 3,
            "expansion_dim": 1024,
        }
    for name in tuple(values):
        override = getattr(args, name)
        if override is not None:
            values[name] = override
    return TraceBranchConfig(
        vocab_size=args.vocab_size,
        beta0=args.beta0,
        **values,
    )


def encoded_chunks(
    path: Path,
    tokenizer: ReferenceTokenizer,
    *,
    read_bytes: int = 1024 * 1024,
) -> Iterator[tuple[np.ndarray, int]]:
    """tokenize whitespace-aligned chunks while preserving byte accounting."""

    with path.open("rb") as handle:
        pending = b""
        while True:
            chunk = handle.read(read_bytes)
            if not chunk:
                if pending:
                    ids = tokenizer.encode(
                        pending.decode("utf-8", errors="replace")
                    )
                    if len(ids):
                        yield ids, len(pending)
                return
            pending += chunk
            boundary = max(
                pending.rfind(b" "),
                pending.rfind(b"\n"),
                pending.rfind(b"\t"),
            )
            if boundary < 0:
                continue
            complete = pending[: boundary + 1]
            pending = pending[boundary + 1 :]
            ids = tokenizer.encode(
                complete.decode("utf-8", errors="replace")
            )
            if len(ids):
                yield ids, len(complete)


def token_batches(
    path: Path,
    tokenizer: ReferenceTokenizer,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, float]]:
    pending_ids = np.empty(0, dtype=np.int64)
    pending_byte_weights = np.empty(0, dtype=np.float64)
    for ids, source_bytes in encoded_chunks(path, tokenizer):
        weights = np.full(
            len(ids), float(source_bytes) / float(len(ids)), dtype=np.float64
        )
        pending_ids = np.concatenate((pending_ids, ids))
        pending_byte_weights = np.concatenate(
            (pending_byte_weights, weights)
        )
        while len(pending_ids) >= batch_size:
            yield (
                pending_ids[:batch_size],
                float(pending_byte_weights[:batch_size].sum()),
            )
            pending_ids = pending_ids[batch_size:]
            pending_byte_weights = pending_byte_weights[batch_size:]
    if len(pending_ids):
        yield pending_ids, float(pending_byte_weights.sum())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="train the soma trace branch mixer"
    )
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--profile", choices=("small", "local"), default="small")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--save", type=Path, default=Path("trace-mixer.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--dream-every", type=int, default=0)
    parser.add_argument("--dream-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--bands", type=int)
    parser.add_argument("--symbol-dim", type=int)
    parser.add_argument("--scale-rank", type=int)
    parser.add_argument("--scale-depth", type=int)
    parser.add_argument("--summaries", type=int)
    parser.add_argument("--model-dim", type=int)
    parser.add_argument("--trunk-depth", type=int)
    parser.add_argument("--expansion-dim", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = ReferenceTokenizer(args.tokenizer)
    if args.checkpoint:
        trainer = TraceBranchTrainer.load(
            args.checkpoint, tokenizer, device=args.device
        )
    else:
        config = config_from_args(args)
        trainer = TraceBranchTrainer(
            config,
            tokenizer,
            device=args.device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
        )

    print(
        f"{trainer.model.species} · {trainer.model.parameter_count / 1e6:.2f}m "
        f"params · {trainer.device}"
    )
    started = time.perf_counter()
    session_tokens = 0
    session_bytes = 0.0
    rolling_nll = 0.0
    rolling_bytes = 0.0

    stop = False
    for epoch in range(args.epochs):
        for ids, source_bytes in token_batches(
            args.corpus, tokenizer, args.batch_size
        ):
            metrics = trainer.train_batch(ids, source_bytes=source_bytes)
            session_tokens += metrics.tokens
            session_bytes += metrics.source_bytes
            rolling_nll += metrics.token_nll
            rolling_bytes += metrics.source_bytes

            if trainer.updates % args.report_every == 0:
                elapsed = max(1e-9, time.perf_counter() - started)
                recent_bpb = (
                    rolling_nll / (rolling_bytes * math.log(2.0))
                    if rolling_bytes
                    else float("nan")
                )
                print(
                    f"epoch {epoch + 1}/{args.epochs} · "
                    f"update {trainer.updates:,} · "
                    f"{recent_bpb:.3f} bpb · "
                    f"{metrics.top1 * 100:.1f}% top-1 · "
                    f"{session_bytes / elapsed:,.0f} source b/s · "
                    f"{trainer.trace.tokens_seen:,} tokens seen"
                )
                rolling_nll = 0.0
                rolling_bytes = 0.0

            if (
                args.dream_every
                and trainer.updates % args.dream_every == 0
            ):
                print(
                    "> "
                    + trainer.generate_text(
                        args.dream_length,
                        temperature=args.temperature,
                    ).replace("\n", "\n> ")
                )

            if args.save_every and trainer.updates % args.save_every == 0:
                trainer.save(args.save)

            if args.max_updates and trainer.updates >= args.max_updates:
                stop = True
                break
            if (
                args.max_minutes
                and time.perf_counter() - started
                >= args.max_minutes * 60.0
            ):
                stop = True
                break
        if stop:
            break

    saved = trainer.save(args.save)
    print(f"saved {saved}")


if __name__ == "__main__":
    main()
