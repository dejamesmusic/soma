"""
soma_loop.py — perpetual training loop on a rolling corpus.

Pairs with wikistream.py. While wikistream maintains a 1GB rolling file
of recent Wikipedia edits at HEAD, this script trains soma on it forever.

Each cycle:
    1. Read the head of the corpus (most recent N bytes)
    2. Train soma on it for one pass
    3. Save the checkpoint
    4. Brief pause
    5. Repeat — corpus has moved on, soma trains on the new head

The model never stops learning. The corpus never stops being current.
The trace bank carries multi-scale memory of recent bytes; the rolling
corpus carries multi-scale memory of recent days. Both timescales,
always live.

Soma's training never "completes" — it adapts. If wikipedia activity
shifts (new event, new topic, new vocabulary), soma drifts to match.

Usage:
    python soma_loop.py corpus.txt model.pt [--head-bytes 50000000]
                                            [--cycle-pause 60]
                                            [--soma-dir /path/to/soma_v12]

The script imports soma_v12 from --soma-dir (or the same directory by
default) and uses its SOMA class directly. No subprocess overhead.

Press Ctrl-C to stop cleanly between cycles. If you stop mid-cycle the
checkpoint from the previous cycle is preserved.
"""

import sys
import os
import time
import argparse
import signal
from pathlib import Path
from datetime import datetime


def fmt_bytes(n):
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


def fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    if seconds < 86400:
        return f"{seconds/3600:.1f}h"
    return f"{seconds/86400:.1f}d"


# graceful-stop flag — set by SIGINT handler so we can stop *between*
# cycles cleanly rather than mid-training (which would lose the cycle's
# work)
_stop_requested = False


def _on_sigint(signum, frame):
    global _stop_requested
    if _stop_requested:
        # second Ctrl-C: hard exit
        print("\n  ▣ hard stop")
        sys.exit(1)
    _stop_requested = True
    print("\n  ▣ stop requested — finishing current cycle, then exiting "
          "(Ctrl-C again for hard stop)")


def setup_soma_path(soma_dir):
    """Make soma_v12 importable from the given directory."""
    if soma_dir:
        sys.path.insert(0, str(soma_dir))
    else:
        sys.path.insert(0, str(Path(__file__).parent.absolute()))


def write_head_to_temp(corpus_path, n_bytes, tmp_path):
    """Read the first n_bytes of corpus_path and write them to tmp_path.

    Done as a stream (not full read) so memory stays bounded even if
    n_bytes is large. Returns actual bytes written (may be less than
    n_bytes if corpus is smaller).
    """
    written = 0
    with open(corpus_path, "rb") as src, open(tmp_path, "wb") as dst:
        remaining = n_bytes
        while remaining > 0:
            chunk = src.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
            written += len(chunk)
    return written


def load_or_build_soma(soma_v12, ckpt_path, head_temp_path, *,
                      n_bands, hidden_dim, n_layers, base, lr, max_change,
                      weight_decay, batch_size, auto_mode, decimation_range,
                      lr_auto, lr_base, max_change_auto, max_change_base):
    """Load checkpoint if it exists, else build a fresh soma."""
    if ckpt_path.exists():
        # We need n_bands etc. from the checkpoint, not from CLI flags,
        # because they must match the saved weights.
        import torch
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = soma_v12.SOMA(
            ckpt.get('n_bands', n_bands),
            base=ckpt.get('base', base),
            hidden_dim=ckpt.get('hidden_dim', hidden_dim),
            n_layers=ckpt.get('n_layers', n_layers),
            lr=ckpt.get('lr', lr),
            max_change=ckpt.get('max_change', max_change),
            weight_decay=ckpt.get('weight_decay', weight_decay),
            batch_size=batch_size,
            decimation_band=ckpt.get('decimation_band', 0),
            direct_readout=bool(ckpt.get('direct_readout', False)),
            scale_gate=bool(ckpt.get('scale_gate', False)),
            clock=ckpt.get('clock', 1),
            auto_mode=ckpt.get('auto_mode', auto_mode),
            lr_auto=ckpt.get('lr_auto', lr_auto),
            lr_base=ckpt.get('lr_base', lr_base),
            max_change_auto=ckpt.get('max_change_auto', max_change_auto),
            max_change_base=ckpt.get('max_change_base', max_change_base),
            decimation_auto=ckpt.get('decimation_auto', True),
            decimation_range=ckpt.get('decimation_range', decimation_range),
        )
        model.load(ckpt_path)
        print(f"  ⟐ resumed {ckpt_path} · "
              f"{fmt_bytes(model.bytes_seen)} seen so far")
        return model

    print(f"  ⟐ no checkpoint at {ckpt_path} — building fresh model")
    model = soma_v12.SOMA(
        n_bands=n_bands, base=base, hidden_dim=hidden_dim,
        n_layers=n_layers,
        lr=lr, max_change=max_change, weight_decay=weight_decay,
        batch_size=batch_size, decimation_band=0,
        direct_readout=False,
        scale_gate=False,
        clock=1,
        auto_mode=auto_mode,
        lr_auto=lr_auto, lr_base=lr_base,
        max_change_auto=max_change_auto, max_change_base=max_change_base,
        decimation_auto=True, decimation_range=decimation_range,
    )
    return model


def main():
    p = argparse.ArgumentParser(
        description="Train soma forever on a rolling corpus.")
    p.add_argument("corpus", type=str,
                   help="path to the rolling corpus file (e.g. from wikistream)")
    p.add_argument("checkpoint", type=str,
                   help="checkpoint path (created if missing, else resumed)")
    p.add_argument("--head-bytes", type=int, default=50_000_000,
                   help="bytes from the head of corpus to train on per cycle "
                        "(default 50M)")
    p.add_argument("--cycle-pause", type=float, default=60.0,
                   help="seconds to wait between cycles, letting the corpus "
                        "ingest more content (default 60s)")
    p.add_argument("--min-corpus-bytes", type=int, default=1_000_000,
                   help="don't start training until corpus has at least "
                        "this many bytes (default 1M)")
    p.add_argument("--soma-dir", type=str, default=None,
                   help="directory containing soma_v12.py (default: same "
                        "as this script)")

    # model build flags — only used when no checkpoint exists yet
    p.add_argument("--bands", type=int, default=32)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--base", type=float, default=1.6180)
    p.add_argument("--lr", type=str, default="spectral 1.0",
                   help="initial lr (literal or 'auto'/'auto N')")
    p.add_argument("--max-change", type=str, default="spectral 1.0")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--decimation", type=float, default=1.0,
                   help="adaptive decimation range fraction in [0, 1]")

    args = p.parse_args()

    # Resolve corpus and checkpoint paths
    corpus_path = Path(args.corpus).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    if not corpus_path.exists():
        print(f"  ! corpus file does not exist: {corpus_path}")
        print(f"    start wikistream.py first to create it")
        sys.exit(1)

    # parse lr / max_change for auto support — replicates soma's CLI parser
    setup_soma_path(args.soma_dir)
    import soma_v12  # noqa

    lr_val, lr_auto, lr_base = soma_v12._parse_auto_or_float(args.lr)
    mc_val, mc_auto, mc_base = soma_v12._parse_auto_or_float(args.max_change)
    auto_tokens = (args.lr + " " + args.max_change).lower()
    auto_mode = ('io2' if 'io2' in auto_tokens
                 else ('full spectrum' if 'full spectrum' in auto_tokens
                 else ('spectral' if 'spectral' in auto_tokens
                       else ('progress' if 'progress' in auto_tokens
                             else 'level'))))

    # Build or load model
    model = load_or_build_soma(
        soma_v12, ckpt_path, None,
        n_bands=args.bands, hidden_dim=args.hidden, n_layers=args.layers,
        base=args.base, lr=lr_val, max_change=mc_val,
        weight_decay=args.weight_decay, batch_size=args.batch,
        auto_mode=auto_mode,
        decimation_range=max(0.0, min(1.0, args.decimation)),
        lr_auto=lr_auto, lr_base=lr_base,
        max_change_auto=mc_auto, max_change_base=mc_base,
    )

    # Apply CLI overrides for things that *can* change between resumes
    model.batch_size = args.batch
    model.decimation_auto = True
    model.decimation_range = max(0.0, min(1.0, args.decimation))

    print()
    model.print_config()

    # set up the SIGINT handler for graceful stop
    signal.signal(signal.SIGINT, _on_sigint)

    print(f"  ░▒▓ soma_loop ▓▒░")
    print(f"  corpus: {corpus_path}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  head per cycle: {fmt_bytes(args.head_bytes)}")
    print(f"  cycle pause: {fmt_duration(args.cycle_pause)}")
    print(f"  Ctrl-C between cycles to stop cleanly")
    print()

    # working temp file for the head slice — kept next to the checkpoint
    # so it lives on the same filesystem (atomic rename behaviour)
    tmp_head = ckpt_path.parent / f".{ckpt_path.stem}_head.tmp"

    cycle = 0
    cycles_started = time.time()
    total_train_seconds = 0.0
    bytes_at_start = model.bytes_seen

    while not _stop_requested:
        cycle += 1
        cycle_start = time.time()

        # Wait for the corpus to be substantive enough to train on
        size = corpus_path.stat().st_size
        if size < args.min_corpus_bytes:
            print(f"  ⏵ corpus only {fmt_bytes(size)} — waiting for more")
            time.sleep(args.cycle_pause)
            continue

        # Snapshot the current head of the corpus to a temp file. This
        # decouples training from wikistream's writes — wikistream can
        # keep prepending new content while soma trains on the snapshot.
        n_to_read = min(args.head_bytes, size)
        n_written = write_head_to_temp(corpus_path, n_to_read, tmp_head)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n  ◯ cycle {cycle} · {ts} · "
              f"head={fmt_bytes(n_written)} · "
              f"corpus={fmt_bytes(size)}")

        # Train one pass over this snapshot
        train_start = time.time()
        try:
            model.train(
                str(tmp_head),
                epochs=1,
                save_every=0,                 # we handle saving below
                save_path=str(ckpt_path),
                start_byte=0,
            )
        except Exception as e:
            print(f"  ! training error: {type(e).__name__}: {e}")
            print(f"    will retry next cycle")
            time.sleep(args.cycle_pause)
            continue
        finally:
            try:
                tmp_head.unlink()
            except FileNotFoundError:
                pass

        train_seconds = time.time() - train_start
        total_train_seconds += train_seconds

        # Save checkpoint after each cycle so we can resume cleanly
        try:
            model.save(str(ckpt_path))
        except Exception as e:
            print(f"  ! save failed: {e}")

        # Cycle summary
        bytes_this_session = model.bytes_seen - bytes_at_start
        wall_time = time.time() - cycles_started
        print(f"  ◍ cycle {cycle} done · "
              f"trained {fmt_bytes(n_written)} in {fmt_duration(train_seconds)} "
              f"· total {fmt_bytes(bytes_this_session)} in "
              f"{fmt_duration(wall_time)}")

        if _stop_requested:
            break

        # Pause to let the corpus accumulate new content before the next cycle
        if args.cycle_pause > 0:
            print(f"  ⏵ pausing {fmt_duration(args.cycle_pause)} "
                  f"before next cycle")
            # Sleep in small increments so Ctrl-C is responsive
            slept = 0.0
            while slept < args.cycle_pause and not _stop_requested:
                time.sleep(min(1.0, args.cycle_pause - slept))
                slept += 1.0

    # final summary
    wall_time = time.time() - cycles_started
    bytes_this_session = model.bytes_seen - bytes_at_start
    print(f"\n  ▣ stopped after {cycle} cycle{'s' if cycle != 1 else ''} "
          f"· {fmt_duration(wall_time)} wall time")
    print(f"    {fmt_bytes(bytes_this_session)} trained this session "
          f"· {fmt_bytes(model.bytes_seen)} lifetime")
    print(f"    checkpoint saved at {ckpt_path}")


if __name__ == "__main__":
    main()
