"""somatic_core runtime: the application-facing learner, with the public surface of soma_gm_v1.runtime.SOMA (so the
soma harness's worker, chat bridge and model library work unchanged).

    train(corpus, ...)        ONE stream, read in order from a resumable, signed cursor (prequential: every tick is
                              scored before it is learned); report, dream and stop callbacks as in gm-v1
    evaluate(corpus, ...)     held-out bpb on a copy of the chat context; nothing learns
    ingest_prompt / generate  chat on its own context (history and match ring); the store and learner are shared;
                              online learning optional (generated tokens are read, never learned: they are samples)
    save / save_fork / from_checkpoint
                              gm-v1's publication protocol: lock, pending revision pointer, atomic replace, lineage,
                              sidecar; checkpoints load with weights_only=True
    reconfigure_dynamics      dynamics change on a continuation; identity settings are locked

The learner is core3 (core3.py): pair-keyed exact store on a phi ladder of timescales + learned reader + mix, exact
inside each tick, io2 learning rate, and nothing that grows (bounds.py). Bounded state: the checkpoint is fixed-size.
"""
from __future__ import annotations

import dataclasses
import fcntl
import gc
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

try:
    from .core import Matcher
    from .core3 import Core3
    from .data import FORBIDDEN, Stream, byte_lengths, encode_bytes, read_segment, tokenizer
except ImportError:                                                     # as scripts in this folder
    from core import Matcher
    from core3 import Core3
    from data import FORBIDDEN, Stream, byte_lengths, encode_bytes, read_segment, tokenizer

SPECIES = "somatic_core"
RUNTIME_ID = "soma.somatic_core"
RUNTIME_VERSION = "1.4.0"
SESSION_FILE = "soma_state"                                               # the control that opens a file in a session
CHECKPOINT_FORMAT = "somatic-core-pytorch"
CHECKPOINT_VERSION = 1
EPS = 1e-12
LN2 = math.log(2.0)
SOURCE_SIGNATURE_BYTES = 64 * 1024
DISK_RESERVE_BYTES = 512 * 1024**2
SEGMENT = 1024                                                          # tokens read per stream call


def _precision(device, config) -> None:
    """cuda numerics: TF32 tensor-core matmuls for fp32 (torch leaves them off by default)."""
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config.tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.tf32)


def resolve_device(value="auto") -> torch.device:
    if isinstance(value, torch.device):
        return value
    name = str(value).lower()
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def _hash_range(path: Path, start: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("corpus is shorter than its saved cursor")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _sign_corpus_state(state: dict) -> dict:
    result = dict(state)
    path = Path(result["path"]).expanduser().resolve()
    position = int(result["file_position"])
    if path.stat().st_size < position:
        raise ValueError("corpus is shorter than its saved cursor")
    prefix_length = min(SOURCE_SIGNATURE_BYTES, int(result["file_size"]))
    window_start = max(0, position - SOURCE_SIGNATURE_BYTES)
    result["signature"] = {
        "version": 1, "prefix_length": prefix_length, "prefix_sha256": _hash_range(path, 0, prefix_length),
        "window_start": window_start, "window_end": position,
        "window_sha256": _hash_range(path, window_start, position - window_start),
    }
    return result


def _corpus_matches(path, state: dict) -> bool:
    signature = state.get("signature", {})
    if not isinstance(signature, dict) or signature.get("version") != 1:
        return False
    path = Path(path).expanduser().resolve()
    try:
        return (path.stat().st_size >= int(state["file_position"])
                and _hash_range(path, 0, int(signature["prefix_length"])) == signature["prefix_sha256"]
                and _hash_range(path, int(signature["window_start"]),
                                int(signature["window_end"]) - int(signature["window_start"])) == signature["window_sha256"])
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _tensor_bytes(value, seen=None) -> int:
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item, seen) for item in value)
    return 0


def pack(x, copy=True):
    """numpy-free form of a state (numpy arrays become uint8 tensors with dtype and shape), for weights_only loads.
    copy=False shares the arrays' memory (1.2.1: a save that writes at once needs no second copy of the store)."""
    if isinstance(x, np.ndarray):
        flat = np.ascontiguousarray(x).view(np.uint8).reshape(-1)
        return {"__np__": x.dtype.str, "shape": list(x.shape), "data": torch.from_numpy(flat.copy() if copy else flat)}
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: pack(v, copy) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(pack(v, copy) for v in x)
    return x


def unpack(x):
    if isinstance(x, dict):
        if "__np__" in x:
            return x["data"].numpy().view(np.dtype(x["__np__"])).reshape(x["shape"]).copy()
        return {k: unpack(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(unpack(v) for v in x)
    return x


@dataclass
class EvaluationMetrics:
    bpb: float
    token_loss: float
    top1: float
    top5: float
    source_bytes: int
    tokens: int


LEGACY_DEFAULTS = dict(att_past=0, learner="mlp", att_heads=6, att_ctx=1024, att_inject="input", warmup=0, skips="", skip_size=3,
                       timescales="", k1=256, k2=32, k3=16, np2=4194304, np3=8388608, hidden=2048, depth=3, proj=16,
                       ladder="3,7,11,15,19,23", io2=1, lr=1e-3, wd=0.1, slot_page=0)

IDENTITY = ("kmax", "k1", "p2", "k2", "p3", "k3", "np1", "np2", "np3", "ladder", "window", "emb", "hidden", "depth",
            "final_norm", "bound_mode", "proj", "minl", "ring", "ctx_decay", "alltime_cap", "seed",
            "learner", "att_heads", "att_ctx", "att_inject", "att_past", "att_rope", "skips", "skip_size", "timescales", "slot_page")


@dataclass
class CoreConfig:
    # Nov 11 (James: "adopt the attention learner and commit to closing the gap"): the defaults are the adopted somatic
    # core, the best tested settings: attention learner 768x12 fed the store at every block, ctx 1024, lr 2e-4 fixed,
    # store at slots x8, pair pools x2, ladder phi^5..phi^25, proj 256, wd 0.002, paged slots. Locked as 1.2.0 (Nov 16).
    # Checkpoints from before carry their own config; fields they lack take LEGACY_DEFAULTS (the 1.1 learner).
    # identity (fixed at creation)
    kmax: int = 3
    k1: int = 2048
    p2: int = 1048576
    k2: int = 256
    p3: int = 2097152
    k3: int = 128
    np1: int = 1048576
    np2: int = 8388608
    np3: int = 16777216
    ladder: str = "5,9,13,17,21,25"
    timescales: str = ""            # dev: explicit rung timescales in tokens, overriding the phi ladder
    learner: str = "attn"           # dev: "attn" = a causal transformer over the last att_ctx tokens (hidden = width, depth = layers)
    att_heads: int = 12
    att_ctx: int = 1024
    skips: str = ""                 # dev: skip contexts, e.g. "3.2;3.1" (offsets back); extra orders fed to the learner
    skip_size: int = 3              # dev: skip orders' slots and pair pools are sized like this order (2 or 3)
    att_inject: str = "every"       # dev: where the store enters the attention learner ("input" | "every")
    att_past: int = 0               # dev: the window's earlier positions carry their own store reads (Learner3 past=1)
    att_rope: int = 0               # dev (Nov 18): rotary positions in the attention learner (needed by att_cache)
    att_cache: int = 0              # dev (Nov 18): each position computed once, at its tick; later ticks read its keys/values
                                    #   1 = the window is ctx per tick (as uncached); 2 = exactly ctx per position, any tick
    warmup: int = 200                 # dev: learner steps over which its rate ramps up (0: none)
    window: int = 16
    emb: int = 64
    hidden: int = 768
    depth: int = 12
    final_norm: int = 1            # a LayerNorm before the learner's output (1.0.1); older checkpoints load with 0
    proj: int = 256
    minl: int = 3
    ring: int = 65536
    ctx_decay: float = 0.99999
    alltime_cap: int = 33554432
    seed: int = 0
    # dynamics (may change on a continuation)
    chunk: int = 512
    io2: int = 0
    lr_base: float = 3e-3
    lr: float = 2e-4
    rate_ctl: str = "fixed"        # dev (Nov 13): "fixed" | "agree" (the rate follows gradient agreement; core3._agree)
    ctl_signal: str = "adam"       # dev: agree's signal: "adam" (v1, failed Nov 14) | "total" (v2, failed Nov 14) | "lag" (v3: against the displacement over ctl_lag..2*ctl_lag steps)
    ctl_lag: int = 64              # dev: agree "lag": steps between the snapshots its displacement is measured from
    ctl_drift: float = 0.0         # dev: agree "lag" v4: if > 0, lag = ctl_drift / lr steps (the learner's own timescale)
    ctl_lag_min: int = 8
    ctl_lag_max: int = 4096
    ctl_sub: int = 1               # dev: agree "lag": judge on every ctl_sub-th coordinate (snapshot memory / ctl_sub)
    feat_scatter: int = 1          # store read: scatter counts onto the vocabulary when cheaper than a per-slot gather (Nov 15)
    slot_page: int = 16            # store: 0 dense slot rows; R > 0 pages of R slots from a shared pool (Nov 15, same answers, ~6x smaller)
    att_ckpt: int = -1              # memory: recompute attention blocks in backward (same math; the 768 learner on an 8 GB Mac); -1 auto = on for mps
    att_kernel: int = -1            # 1.2.1: attention by the Metal flash kernel (flash_mps.py; same maths up to float order); -1 auto = on for mps
    hg_tier: int = 1               # dev: rate_ctl=hyper (Nov 15): 0 one-step hypergradient; 1 trace with diagonal Fisher curvature
    hg_mu: float = 0.01            # dev: hyper: e-folds of rate per tick at a unit normalised hypergradient
    hg_curv: float = 0.0           # dev: hyper tier 1: tokens N in curvature ~ N v_hat (0 = the tick)
    hg_rms: float = 0.99           # dev: hyper: decay of the hypergradient's running RMS
    hg_span: float = 30.0          # dev: hyper: rail on the rate, lr / span .. lr * span (hits are reported)
    ctl_k: float = 0.05             # dev: agree's gain: log-rate change per step per unit of agreement (cosine)
    ctl_min: float = 1e-6
    ctl_max: float = 1e-2
    mix_adam: float = 3e-3
    mix_shared: int = 0          # the exact mixer (wE/wM) is the learner's, not the store's: count() is counts only (Nov 18). 1: it steps with the learner (mean over its update); 2: every tick, as with 0
    gate_lr: float = 0.02
    wd: float = 0.002               # decoupled, on the learner's matrices (1.1; 0.002 from the Nov 12 1 GB runs); radial checkpoints keep their own
    learn_stride: int = 1          # learner optimiser steps on one tick in this many (it predicts every token)
    learn_mode: str = "skip"       # skip | accumulate (by tick) | select (every learn_stride-th prediction) | io2 (io2 picks it)
    tf32: int = 1                  # cuda only: fp32 matmuls on tensor cores (TF32); no effect on mps or cpu
    learner_bf16: int = 0          # cuda only: the learner's forward and backward in bf16 autocast (weights and Adam fp32)
    rehearse: int = 0              # dev (Nov 10): extra learner steps per tick on reads taken before their store update
    rehearse_span: int = 1         # dev: 1 = repeat the current tick; D > 1 = each extra step on one of the last D ticks
    # how the learner is bounded (1.1): wd = AdamW decoupled weight decay on matrices, vectors clamped to +-8 (default);
    # radial = 1.0.x (AdamW + the v14 radial compressor; checkpoints from before 1.1 open with it);
    # sphere = experimental (fixed-radius sphere SGD; lost 0.047 bpb in the Nov 9 A/B). Fixed when a model is created.
    bound_mode: str = "wd"
    sphere_rate: float = 0.7
    max_angle: float = 0.05

    def validate(self):
        assert 1 <= self.kmax <= 3, "orders 1-3"
        assert self.chunk >= 1 and self.window >= 1 and self.window <= 64
        assert self.hidden >= 16 and self.depth >= 1 and self.final_norm in (0, 1)
        assert self.learn_stride >= 1 and self.learn_mode in ("skip", "accumulate", "select", "io2")
        assert self.tf32 in (0, 1) and self.learner_bf16 in (0, 1)
        assert self.rehearse >= 0 and self.rehearse_span >= 1
        assert self.bound_mode in ("radial", "wd", "sphere")
        assert self.rate_ctl in ("fixed", "agree", "hyper") and not (self.rate_ctl == "agree" and self.io2), "one rate controller"
        assert self.learner != "attn" or self.att_ctx >= self.chunk or self.att_cache == 2, \
            "att_ctx must cover a tick (positions index past it), unless the window is per position (att_cache=2)"
        assert not self.att_cache or (self.att_rope and not self.att_past), "att_cache needs att_rope (and not att_past)"
        assert self.learner != "attn" or (self.att_heads >= 1 and self.hidden % self.att_heads == 0), \
            f"width {self.hidden} must divide into {self.att_heads} attention heads"
        return self

    def public_dict(self):
        return dataclasses.asdict(self)


class SOMA:
    """application-facing somatic_core learner with an exact, bounded state boundary."""

    species = SPECIES

    def __init__(self, config: CoreConfig | None = None, *, device="auto", tokenizer_path=None, seed=7, description=""):
        self.config = (config or CoreConfig()).validate()
        self.device = resolve_device(device)
        _precision(self.device, self.config)
        self.tokenizer = tokenizer()
        self.seed = int(seed)
        self.blen = byte_lengths(self.tokenizer)
        self.core = Core3(self.config.public_dict(), str(self.device), self.blen)
        self.stream_ctx = Matcher(self.config.ring, self.config.minl)
        self.chat_ctx = Matcher(self.config.ring, self.config.minl)
        self.pending = np.zeros(0, np.int64)                             # the stream's unfilled tick (1.3.0)
        self._synced = True                                              # chat_ctx == stream_ctx + pending
        self.description = str(description)
        self.checkpoint_id = uuid.uuid4().hex
        self.revision_id = uuid.uuid4().hex
        self.parent_revision_id = None
        self.root_lineage_id = self.checkpoint_id
        self.revision_history = []
        self.forked_from_checkpoint_id = None
        self.forked_from_revision_id = None
        self.created_at = time.time()
        self._revision_published = False
        self._loaded_checkpoint_path: Path | None = None
        self._source_cursor: dict | None = None
        self.life = dict(bytes=0, tokens=0, bits_exact=0.0, bits_learner=0.0, bits_final=0.0, scored_bytes=0.0)
        self.sampler = torch.Generator(device="cpu")
        self.sampler.manual_seed(self.seed)

    # ---- basic properties ----
    @property
    def net(self):
        return self.core.net

    @property
    def bytes_seen(self):
        return int(self.life["bytes"])

    @property
    def tokens_seen(self):
        return int(self.life["tokens"])

    @property
    def batch_size(self):
        return self.config.chunk

    def params(self) -> int:
        return int(self.core.params())

    def store_bytes(self) -> int:
        return int(self.core.store_floats() * 4)

    # ---- dynamics ----
    def reconfigure_dynamics(self, config: CoreConfig) -> None:
        config = config.validate()
        changed = {k: (getattr(self.config, k), getattr(config, k)) for k in IDENTITY
                   if getattr(self.config, k) != getattr(config, k)}
        if changed:
            raise ValueError(f"identity settings are locked for a saved model: {changed}")
        for k, v in config.public_dict().items():
            if k not in IDENTITY:
                setattr(self.config, k, v)
                self.core.c[k] = v
        _precision(self.device, self.config)
        if hasattr(self.core.net, "ckpt"):
            a = int(self.config.att_ckpt)
            self.core.net.ckpt = bool(a) if a >= 0 else self.device.type == "mps"
            fk = int(self.config.att_kernel)
            for b in self.core.net.blocks:
                b.flash = (bool(fk) if fk >= 0 else self.device.type == "mps") and self.device.type == "mps"
        if self.core.io2 is not None:
            self.core.io2.lr_base = float(config.lr_base)
        for g in self.core.mopt.param_groups:
            g["lr"] = float(config.mix_adam)
        for i, g in enumerate(self.core.opt.param_groups):
            if not (self.config.bound_mode == "wd" and i > 0):         # wd mode: group 0 is the matrices, 1 the vectors
                g["weight_decay"] = float(config.wd)
            if self.core.io2 is None:
                g["lr"] = float(config.lr)

    # ---- one tick ----
    def _tick(self, ids, ctx, learn=True):
        """ids: a 1-d token array of any length; stepped in ticks of `chunk`. -> (bits_final_sum, bytes)"""
        C = int(self.config.chunk)
        nb = 0.0
        for i in range(0, len(ids), C):
            y = ids[i:i + C]
            self.core.step(ctx, y, learn=learn)
            nb += float(self.blen[y].sum())
        return nb

    # ---- training ----
    def resume_state_for(self, corpus_path) -> dict | None:
        cur = self._source_cursor
        if not isinstance(cur, dict) or cur.get("kind") != "somatic_stream":
            return None
        requested = Path(corpus_path).expanduser().resolve()
        if requested != Path(cur.get("path", "")).expanduser().resolve():
            return None                                                 # a continual learner may meet a new source
        if not _corpus_matches(requested, cur.get("signed", {})):
            raise ValueError("corpus content differs from the checkpoint cursor")
        if int(np.max(cur.get("streams", {}).get("wraps", [0]))) >= 1:
            return None
        return cur

    def _cursor(self, stream, path, byte_range):
        pos = max(stream.state()["pos"])
        signed = _sign_corpus_state({"path": str(path), "file_position": pos, "file_size": Path(path).stat().st_size})
        return dict(kind="somatic_stream", path=str(path), byte_range=list(byte_range), streams=stream.state(),
                    signed=signed)

    def summary(self) -> dict:
        c = self.config
        return dict(species=self.species, params=self.params(), parameters=self.params(), width=c.hidden,
                    depth=c.depth, window=c.window, chunk=c.chunk, orders=c.kmax, ladder=c.ladder,
                    store_bytes=self.store_bytes(), pairs=c.np1 + c.np2 + c.np3, contexts=c.p2 + c.p3,
                    checkpoint_id=self.checkpoint_id, revision_id=self.revision_id, bytes_seen=self.bytes_seen,
                    tokens_seen=self.tokens_seen, tokenizer=self.tokenizer.vocab_size, device=str(self.device))

    def stream_position(self) -> int:
        cur = self._source_cursor
        if isinstance(cur, dict) and cur.get("kind") == "somatic_follow":
            return int(cur["position"])
        return 0

    def _drain_into(self, window, session):
        d = self.core.drain()
        for k_src, k in (("bits_exact", "bits_exact"), ("bits_learner", "bits_learner"), ("bits_final", "bits_final"),
                         ("correct", "correct"), ("scored", "scored")):
            window[k] = window.get(k, 0.0) + d[k_src]
            session[k] = session.get(k, 0.0) + d[k_src]
        for k in ("bits_exact", "bits_learner", "bits_final"):
            self.life[k] += d[k]

    def train(self, corpus_path, epochs=1, save_every=0, save_path="", start_byte=0, max_bytes=0, report_every=1,
              report_callback: Callable[[dict], None] | None = None, dream_every_batches=0, dream_length=128,
              dream_temperature=0.8, dream_callback: Callable[[str, int, int], None] | None = None,
              dream_stateful=False, resume_state=None, exact_start=False,
              stop_callback: Callable[[], bool] | None = None, learn=True, **_unused):
        """read the corpus in order, one stream, for `epochs` passes over the byte range (or follow an append-only
        file from an exact byte with exact_start). Every tick is scored before it is learned from.
        1.3.0: the corpus joins the model's one stream (see _feed): its tokens follow whatever the stream held
        last, are learned in exact ticks, and a final part-filled tick waits in `pending` for whatever comes next.
        learn=False scores and reads without learning."""
        path = Path(corpus_path).expanduser().resolve()
        size = path.stat().st_size
        lo = int(start_byte)
        hi = size if not max_bytes else min(size, lo + int(max_bytes))
        if resume_state and resume_state.get("byte_range"):
            lo, hi = resume_state["byte_range"]
        stream = Stream(path, 1, self.tokenizer, byte_range=(lo, hi), exact=bool(exact_start))
        if resume_state:
            stream.load(resume_state["streams"])
        wraps0 = stream.wraps.copy()
        session = dict(bytes=0.0, tokens=0, batches=0)
        window = dict(bytes=0.0)
        started = time.time()
        batches = 0
        while not (stop_callback and stop_callback()):
            if exact_start:
                if int(stream.bytes_read.sum()) >= hi - lo and not stream.buf[0].size:
                    break
            elif int((stream.wraps - wraps0).max()) >= max(1, int(epochs)):
                break
            ids = stream.next(SEGMENT)[0]
            if exact_start and stream.wraps[0]:
                break
            fed = {}
            self._feed(ids, learn=learn, totals=(fed,), read=False)
            nb = fed.get("bytes", 0.0)
            session["bytes"] += nb
            session["tokens"] += fed.get("tokens", 0)
            window["bytes"] += nb
            session["batches"] += 1
            batches += 1
            if report_every and batches % int(report_every) == 0:
                self._drain_into(window, session)
                self._source_cursor = (dict(kind="somatic_follow", path=str(path),
                                            position=lo + int(stream.bytes_read.sum()))
                                       if exact_start else self._cursor(stream, path, (lo, hi)))
                if report_callback:
                    report_callback(self._report(session, window, started, stream, lo, hi, epochs, wraps0))
                window = dict(bytes=0.0)
            if dream_every_batches and batches % int(dream_every_batches) == 0 and dream_callback:
                dream_callback(self.dream_text(int(dream_length), float(dream_temperature), source="stream"),
                               self.core.chunks, self.bytes_seen)
            if save_every and save_path and batches % int(save_every) == 0:
                self.save(save_path)
        self._drain_into(window, session)
        self._source_cursor = (dict(kind="somatic_follow", path=str(path), position=lo + int(stream.bytes_read.sum()))
                               if exact_start else self._cursor(stream, path, (lo, hi)))
        if save_path:
            self.save(save_path)
        return session

    def _report(self, session, window, started, stream, lo, hi, epochs, wraps0):
        now = time.time()
        prev_t, prev_tok = getattr(self, "_rep_clock", (started, 0))
        if prev_t < started:
            prev_t, prev_tok = started, 0
        dt = max(EPS, now - prev_t)
        self._rep_clock = (now, session["tokens"])
        wb = max(EPS, window["bytes"])
        st = self.core.stats()
        span = max(1, hi - lo)
        passes = int((stream.wraps - wraps0).max())
        frac = min(1.0, int(stream.bytes_read.sum()) / span)
        last = self.core.last
        surprise, top = [], []
        if last:
            toks, bits = last["tokens"][-48:], last["bits"][-48:].tolist()
            surprise = [[self.tokenizer.decode([int(t)]), float(b)] for t, b in zip(toks, bits)]
            vals, idx = last["next"]
            top = [{"token": self.tokenizer.decode([int(i)]), "probability": float(v)}
                   for v, i in zip(vals.tolist(), idx.tolist())]
        report = {
            "species": self.species,
            "bpb": window.get("bits_final", 0.0) / wb, "bpb_exact": window.get("bits_exact", 0.0) / wb,
            "bpb_learner": window.get("bits_learner", 0.0) / wb,
            "session_bpb": session.get("bits_final", 0.0) / max(EPS, session["bytes"]),
            "life_bpb": self.life["bits_final"] / max(EPS, self.life["scored_bytes"]),
            "acc": 100.0 * window.get("correct", 0.0) / max(EPS, window.get("scored", 0.0)),
            "token_loss": window.get("bits_final", 0.0) * LN2 / max(EPS, window.get("scored", 0.0)),
            "tokens_per_second": (session["tokens"] - prev_tok) / dt,
            "session_tokens_per_second": session["tokens"] / max(EPS, now - started),
            "bps": window["bytes"] / dt, "session_tokens": session["tokens"], "life_tokens": self.tokens_seen,
            "life_bytes": self.bytes_seen, "updates": int(st["chunks"]), "batch": self.batch_size,
            "parameters": self.params(), "store_bytes": self.store_bytes(),
            "pos": int(stream.bytes_read.sum()), "total": span * max(1, int(epochs)),
            "absolute_pos": int((passes + frac) * span), "absolute_total": span * max(1, int(epochs)),
            "epoch": min(passes + 1, max(1, int(epochs))), "epochs": max(1, int(epochs)),
            "lr": float(st.get("lr", 0.0)), "gate_exact": float(st.get("gate_exact", 0.0)),
            "surprise": surprise, "top_predictions": top,
        }
        io2 = getattr(self.core, "io2", None)
        if io2 is not None and any(io2.spectrum):
            # the harness's "residual by band": the learner's translated residual, share per io2 band (fast to slow),
            # against its own slow average; the centre is io2's target band (where the residual sits)
            spec = np.asarray(io2.spectrum, np.float64)
            base = getattr(self, "_spec_base", None)
            self._spec_base = spec.copy() if base is None or len(base) != len(spec) else base + (spec - base) / 64.0
            report["residual_spectrum"] = spec.tolist()
            report["residual_baseline"] = self._spec_base.tolist()
            report["residual_excess"] = np.clip(spec - self._spec_base, 0.0, None).tolist()
            report["traversal_band"] = float(io2.target_band)
        # "listening": how the exact mix weighs the phi ladder, rung by rung (tau = phi^3 ... phi^23, then all-time)
        with torch.no_grad():
            w = self.core.wE.detach().float().clamp_min(0.0).mean(0)
            report["listening"] = (w / w.sum().clamp_min(EPS)).cpu().tolist()
        for k in ("io2_ratio", "io2_energy", "io2_coherence", "io2_band", "radial_pressure", "max_row_over_ceiling",
                  "halvings"):
            if k in st:
                report[k] = float(st[k])
        for k in self.core.orders:
            for f in ("pair_fill", "pair_evictions", "fill", "recycles", "overflows"):
                key = f"o{k}_{f}"
                if key in st:
                    report[key] = float(st[key])
        return report

    # ---- reading, evaluation, chat ----
    def _sample(self, p: torch.Tensor, temperature: float, forbidden) -> int:
        p = torch.nan_to_num(p.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
        if forbidden:
            p[list(forbidden)] = 0.0
        if temperature != 1.0:
            p = p.clamp_min(1e-30) ** (1.0 / max(1e-3, float(temperature)))
        if not float(p.sum()) > 0.0:                                     # nothing left: uniform over the allowed tokens
            p = torch.ones_like(p)
            if forbidden:
                p[list(forbidden)] = 0.0
        p = p / p.sum()
        return int(torch.multinomial(p, 1, generator=self.sampler))

    # ---- the stream (1.3.0) ----
    # A model lives one stream. Every token it meets goes into it in order, whatever the source: words typed to it,
    # files it reads, corpora it trains on, and its own replies, with the markers that open each span:
    #   <|user|>        words someone typed
    #   <|soma_state|>  a file (the alphabet's reserved control, never used before 1.3.0; SESSION_FILE names it)
    #   <|soma|>        its own reply, closed by <|turn|>
    # It learns the stream in exact ticks of `chunk` tokens, the tick training has always run: every token is scored
    # once, before it is learned from, and learned once, whatever its source. Tokens wait in `pending` until a tick
    # fills, and the unfilled tick is saved with the checkpoint. stream_ctx is the stream as learned; chat_ctx, the
    # context the model reads and replies from, is the same stream with the pending tail. So nothing in the learning
    # changes with a prompt's length, and its own words stay in its context (and its store's counts) like any other.
    def _chat_window(self):
        """bring the reading context up to the stream: the learned stream, then its unlearned tail. (Before 1.3.0 the
        chat had a context of its own; a checkpoint from then reads from its stream from here on.)"""
        if self._synced:
            return
        self.chat_ctx = self.stream_ctx.clone()
        getattr(self.core, "read", self.core.observe)(self.chat_ctx, self.pending)   # read: through the learner's window
        self._synced = True

    def _feed(self, ids, learn=True, totals=None, read=True):
        """tokens join the stream: read into the context at once (read=False leaves the reading context to be rebuilt,
        for long reads), learned in whole ticks as they fill. totals, if given, collects the ticks' bytes and tokens."""
        ids = np.asarray(ids, np.int64).reshape(-1)
        if not len(ids):
            return
        C = int(self.config.chunk)
        # a tick learned here moves the weights, and the reading context's attention window with them: it is then
        # rebuilt from the stream (its training window plus the unlearned tail) at next use, not read token by token
        # into a window the update has outdated (Nov 18)
        if read and self._synced and len(self.pending) + len(ids) < C:
            getattr(self.core, "read", self.core.observe)(self.chat_ctx, ids)
        else:
            self._synced = False
        self.pending = np.concatenate([self.pending, ids])
        while len(self.pending) >= C:
            tick, self.pending = self.pending[:C], self.pending[C:]
            nb = self._tick(tick, self.stream_ctx, learn=learn)
            self.life["bytes"] += nb
            self.life["tokens"] += C
            self.life["scored_bytes"] += nb
            if totals is not None:
                for d in totals:
                    d["bytes"] = d.get("bytes", 0.0) + nb
                    d["tokens"] = d.get("tokens", 0) + C
                    d["batches"] = d.get("batches", 0) + 1

    def _learned_into(self, *dicts):
        """the scored ticks' bits since the last call, into each dict (and the life)"""
        d = self.core.drain()
        for k in ("bits_exact", "bits_learner", "bits_final"):
            self.life[k] += d[k]
        for out in dicts:
            for k_src in ("bits_exact", "bits_learner", "bits_final", "correct", "scored"):
                out[k_src] = out.get(k_src, 0.0) + d[k_src]

    def _mark(self, name, online):
        self._chat_window()
        self._feed([self.tokenizer.control_ids[name]], learn=bool(online))
        self._learned_into()

    def ingest_text(self, text, online=False):
        """words straight into the stream, with no marker"""
        self._chat_window()
        self._feed(encode_bytes(self.tokenizer, str(text).encode("utf-8")), learn=bool(online))
        self._learned_into()

    def ingest_prompt(self, text, online=False, mark_turn=True):
        """a typed turn and the reply's opening: <|user|> text <|soma|> (the pre-1.3.0 chat call, now on the stream)"""
        if mark_turn:
            self.session_text(text, online=online)
        else:
            self.ingest_text(text, online=online)
        self.session_reply(online=online)

    def session_text(self, text, online=False):
        """typed words: <|user|> text"""
        self._chat_window()
        body = np.asarray(encode_bytes(self.tokenizer, str(text).encode("utf-8")), np.int64)
        self._feed(np.concatenate([np.asarray([self.tokenizer.control_ids["user"]], np.int64), body]), learn=bool(online))
        self._learned_into()

    def session_file(self, path, online=True, report_every=1, report_callback=None, stop_callback=None):
        """a file: <|soma_state|> then its bytes once, start to end, in the stream's ticks (nothing after its end: a
        training pass wraps to fill its last tick, which here would put the file's opening after its end).
        -> the read's totals (bytes, tokens and bits of the ticks it completed)"""
        path = Path(path).expanduser().resolve()
        assert not any(w in path.name.lower() for w in FORBIDDEN), "never read iMessage-derived data"
        size = path.stat().st_size
        self._mark(SESSION_FILE, online)
        shim = type("Read", (), {})()
        shim.bytes_read, shim.wraps = np.zeros(1, np.int64), np.zeros(1, np.int64)
        session, window = dict(bytes=0.0, tokens=0, batches=0), dict(bytes=0.0)
        started, reported = time.time(), 0
        with path.open("rb") as f:
            while not (stop_callback and stop_callback()):
                raw = read_segment(f, size, 4096)
                if not raw:
                    break
                shim.bytes_read[0] += len(raw)
                self._feed(encode_bytes(self.tokenizer, raw), learn=bool(online), totals=(session, window), read=False)
                if report_every and session["batches"] - reported >= int(report_every):
                    reported = session["batches"]
                    self._learned_into(window, session)
                    if report_callback:
                        report_callback(self._report(session, window, started, shim, 0, size, 1, np.zeros(1, np.int64)))
                    window = dict(bytes=0.0)
        self._learned_into(window, session)
        self._chat_window()
        session["file_bytes"] = int(shim.bytes_read[0])
        return session

    def session_reply(self, online=False):
        """the reply begins: <|soma|> (generate() then samples until <|turn|> or its length)"""
        self._mark("soma", online)

    @torch.no_grad()
    def next_logits(self) -> torch.Tensor:
        self._chat_window()
        return torch.log(self.core.next_distribution(self.chat_ctx).clamp_min(1e-30))

    def _generation_tokens(self, length, temperature, ctx=None, learn=True):
        """samples; in the reading context each one joins the stream (and is learned when its tick fills, like any
        other token). A sample from a copy of a context (ctx given) stays there."""
        stream = ctx is None
        if stream:
            self._chat_window()
        ctx = self.chat_ctx if ctx is None else ctx
        forbidden = list(self.tokenizer.generation_forbidden_ids)
        turn = self.tokenizer.control_ids["turn"]
        ended = False

        def take(t):
            if stream:
                self._feed([t], learn=learn)
                self._learned_into()
                self._chat_window()                  # after a learned tick the reading context is rebuilt
            else:
                self.core.observe(ctx, [t])
        try:
            for _ in range(int(length)):
                c = self.chat_ctx if stream else ctx
                t = self._sample(self.core.next_distribution(c), float(temperature), forbidden)
                take(t)
                if t == turn:
                    ended = True
                    break
                yield t
        finally:
            if not ended:
                take(turn)

    def generate_ids(self, length=200, temperature=1.0, learn=True, **_unused):
        return list(self._generation_tokens(length, temperature, learn=learn))

    def generate_text(self, length=200, temperature=1.0, **kwargs):
        return self.tokenizer.decode(self.generate_ids(length, temperature))

    def generate(self, length=200, temperature=1.0, learn=True, **_unused):
        decoder = self.tokenizer.incremental_decoder()
        for t in self._generation_tokens(length, temperature, learn=learn):
            yield from decoder.push(t)
        yield from decoder.finish()

    def dream_text(self, length=128, temperature=0.8, *, stateful=False, source="chat") -> str:
        """a dream from a copy of a context (changes nothing unless stateful)."""
        if source != "stream":
            self._chat_window()
        base = self.stream_ctx if source == "stream" else self.chat_ctx
        ctx = base if stateful else base.clone()
        if stateful:
            self._synced = False                     # a stateful dream writes into a context: rebuild the reading one
        state = None if stateful else self.sampler.get_state()
        try:
            out = []
            # a dream runs the stream freely, speakers included: in a dialogue corpus every <|turn|> is followed by a role
            # token, and forbidding the roles forced each new turn to start on a leftover token ("?", "'m", "know").
            # chat replies (generate) still forbid them and stop at <|turn|>.
            ctl = self.tokenizer.control_ids
            forbidden = [t for t in self.tokenizer.generation_forbidden_ids if t not in (ctl["user"], ctl["soma"])]
            for _ in range(int(length)):
                t = self._sample(self.core.next_distribution(ctx), float(temperature), forbidden)
                self.core.observe(ctx, [t])
                out.append(t)
        finally:
            if state is not None:
                self.sampler.set_state(state)
        return self.tokenizer.decode(out)

    @torch.no_grad()
    def evaluate(self, corpus_path, max_bytes=0, start_byte=0, carry_state=True) -> EvaluationMetrics:
        """held-out metrics on a slice, read with nothing learning, on a copy of the chat context (or a fresh one)."""
        if carry_state:
            self._chat_window()
        ctx = self.chat_ctx.clone() if carry_state else Matcher(self.config.ring, self.config.minl)
        path = Path(corpus_path).expanduser().resolve()
        size = path.stat().st_size
        hi = size if not max_bytes else min(size, int(start_byte) + int(max_bytes))
        st = Stream(path, 1, self.tokenizer, byte_range=(int(start_byte), hi))
        saved = self.core.drain()                                           # keep training's accumulators apart
        nb = 0.0
        tokens = 0
        while int(st.bytes_read.sum()) < hi - int(start_byte):
            ids = st.next(SEGMENT)[0]
            nb += self._tick(ids, ctx, learn=False)
            tokens += len(ids)
        d = self.core.drain()
        self.core.acc += torch.tensor([saved["bits_exact"], saved["bits_learner"], saved["bits_final"], 0.0,
                                       saved["correct"], saved["scored"]], device=self.core.dev)
        return EvaluationMetrics(bpb=d["bits_final"] / max(EPS, nb), token_loss=d["bits_final"] * LN2 / max(EPS, d["scored"]),
                                 top1=d["correct"] / max(EPS, d["scored"]), top5=float("nan"),
                                 source_bytes=int(st.bytes_read.sum()), tokens=tokens)

    def diagnostics(self) -> dict:
        st = self.core.stats()
        return dict(species=self.species, runtime_id=RUNTIME_ID, runtime_version=RUNTIME_VERSION,
                    parameters=self.params(), store_bytes=self.store_bytes(), tokens_seen=self.tokens_seen,
                    bytes_seen=self.bytes_seen, config=self.config.public_dict(),
                    store={k: float(v) for k, v in st.items() if isinstance(v, (int, float))}, life=dict(self.life))

    # ---- checkpoints (gm-v1's publication protocol) ----
    def _checkpoint_dict(self, revision_id) -> dict:
        parent = self.revision_id if self._revision_published else self.parent_revision_id
        history = list(self.revision_history)
        if self._revision_published:
            rec = {"checkpoint_id": self.checkpoint_id, "revision_id": self.revision_id,
                   "parent_revision_id": self.parent_revision_id}
            if not history or history[-1] != rec:
                history.append(rec)
        return {
            "checkpoint_format": CHECKPOINT_FORMAT, "checkpoint_version": CHECKPOINT_VERSION,
            "runtime_id": RUNTIME_ID, "runtime_version": RUNTIME_VERSION, "species": self.species,
            "config": self.config.public_dict(), "seed": self.seed, "description": self.description,
            "checkpoint_id": self.checkpoint_id, "revision_id": revision_id, "parent_revision_id": parent,
            "root_lineage_id": self.root_lineage_id, "created_at": self.created_at, "revision_history": history[-256:],
            "forked_from_checkpoint_id": self.forked_from_checkpoint_id,
            "forked_from_revision_id": self.forked_from_revision_id, "tokenizer_sha256": self.tokenizer.sha256,
            "optimizer": "absent" if getattr(self, "_slim", False) else "present",
            "core": pack(self._core_state(), copy=False), "stream_ctx": pack(self.stream_ctx.state()),
            "chat_ctx": pack(self.chat_ctx.state()), "pending": [int(t) for t in self.pending], "sampler": self.sampler.get_state(),
            "source_cursor": pack(self._source_cursor), "life": dict(self.life),
            # plain metadata for readers that never unpack the core (logOS's checkpoint parser)
            "bytes_seen": int(self.bytes_seen), "tokens_seen": int(self.tokens_seen), "parameters": int(self.params()),
            "hidden_dim": int(self.config.hidden), "depth": int(self.config.depth),
        }

    def save(self, path):
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path, pointer_path = Path(f"{path}.lock"), Path(f"{path}.revision.json")
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            previous = _read_json(pointer_path)
            if previous.get("state") == "pending":
                raise RuntimeError("checkpoint has an unfinished publication")
            same = self._loaded_checkpoint_path == path
            if path.exists() and not same:
                raise FileExistsError("checkpoint exists; load it or save a fork")
            disk_revision = previous.get("revision_id")
            if same and disk_revision and disk_revision != self.revision_id:
                raise RuntimeError("checkpoint changed on disk; save a fork")
            revision = uuid.uuid4().hex
            _write_json_atomic(pointer_path, {"version": 1, "state": "pending", "checkpoint_id": self.checkpoint_id,
                                              "previous_revision_id": self.revision_id, "revision_id": revision})
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                checkpoint = self._checkpoint_dict(revision)
                tensor_bytes = _tensor_bytes(checkpoint)
                required = math.ceil(tensor_bytes * 1.1) + max(DISK_RESERVE_BYTES, tensor_bytes // 2)
                if shutil.disk_usage(path.parent).free < required:
                    raise OSError("not enough free disk space for an atomic checkpoint")
                torch.save(checkpoint, temporary)
                os.replace(temporary, path)
                self.parent_revision_id = checkpoint["parent_revision_id"]
                self.revision_history = checkpoint["revision_history"]
                self.revision_id = revision
                self._revision_published = True
                self._loaded_checkpoint_path = path
                _write_json_atomic(pointer_path, {"version": 1, "state": "complete",
                                                  "checkpoint_id": self.checkpoint_id, "revision_id": revision})
                self._write_sidecar(path)
            except Exception:
                temporary.unlink(missing_ok=True)
                if previous:
                    _write_json_atomic(pointer_path, previous)
                else:
                    pointer_path.unlink(missing_ok=True)
                raise
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        gc.collect()
        return path

    def _write_sidecar(self, path: Path) -> None:
        existing = _read_json(Path(f"{path}.meta.json"))
        c = self.config
        sidecar = {
            "sidecar_version": 4, "optimizer": "absent" if getattr(self, "_slim", False) else "present",
            "species": self.species, "runtime_id": RUNTIME_ID, "runtime_version": RUNTIME_VERSION,
            "checkpoint_format": CHECKPOINT_FORMAT, "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_id": self.checkpoint_id, "revision_id": self.revision_id,
            "parent_revision_id": self.parent_revision_id, "root_lineage_id": self.root_lineage_id,
            "revision_history": self.revision_history, "forked_from_checkpoint_id": self.forked_from_checkpoint_id,
            "forked_from_revision_id": self.forked_from_revision_id, "description": self.description,
            "created_at": self.created_at, "updated_at": time.time(), "parameters": self.params(),
            "params": self.params(), "width": c.hidden, "depth": c.depth, "window": c.window, "chunk": c.chunk,
            "orders": c.kmax, "store_bytes": self.store_bytes(), "tokens_seen": self.tokens_seen,
            "bytes_seen": self.bytes_seen, "life_bpb": self.life["bits_final"] / max(EPS, self.life["scored_bytes"]),
        }
        sidecar["config"] = dataclasses.asdict(c)
        cur = self._source_cursor
        if isinstance(cur, dict) and cur.get("path"):
            # the harness reads source_cursor.reader to prefill the corpus and show / apply "resume at"
            src = Path(cur["path"])
            pos = int(cur["position"]) if cur.get("kind") == "somatic_follow" else int(max(cur["streams"]["pos"]))
            lo = int((cur.get("byte_range") or [0])[0] or 0)
            sidecar["source_cursor"] = {"kind": cur.get("kind"), "reader": {
                "path": str(src), "file_position": pos, "file_size": src.stat().st_size if src.exists() else None,
                "byte_range": cur.get("byte_range"),
                # the same position as logOS's cloud agent reads it (start_byte + bytes_consumed)
                "start_byte": lo, "bytes_consumed": max(0, pos - lo)}}
            sidecar["source_position"] = pos
        if "profile" in existing:
            sidecar["profile"] = existing["profile"]
        _write_json_atomic(Path(f"{path}.meta.json"), sidecar)

    def _core_state(self) -> dict:
        st = self.core.state_dict(lean=True)
        if getattr(self, "_slim", False):                 # slim: the optimisers' moments stay home (about 2/3 of a
            st = dict(st, opt=None, mopt=None)            # learner's bytes); loading re-warms the rate
        return st

    def export_slim(self, path):
        """a copy to share (1.4.0): a fork of this model without optimiser state. This model stays as it was. Opened,
        it reads and talks exactly like the original; when it learns, its rate re-warms over Core3.REWARM ticks."""
        keep = (self.checkpoint_id, self.parent_revision_id, self.root_lineage_id, self.revision_id,
                self._loaded_checkpoint_path, self.created_at, self._revision_published, self.revision_history,
                self.forked_from_checkpoint_id, self.forked_from_revision_id)
        self._slim = True
        try:
            return self.save_fork(path)
        finally:
            self._slim = False
            (self.checkpoint_id, self.parent_revision_id, self.root_lineage_id, self.revision_id,
             self._loaded_checkpoint_path, self.created_at, self._revision_published, self.revision_history,
             self.forked_from_checkpoint_id, self.forked_from_revision_id) = keep

    def save_fork(self, path):
        old = (self.checkpoint_id, self.parent_revision_id, self.root_lineage_id, self.revision_id,
               self._loaded_checkpoint_path, self.created_at, self._revision_published, self.revision_history,
               self.forked_from_checkpoint_id, self.forked_from_revision_id)
        record = {"checkpoint_id": self.checkpoint_id, "revision_id": self.revision_id,
                  "parent_revision_id": self.parent_revision_id}
        history = list(self.revision_history)
        if not history or history[-1] != record:
            history.append(record)
        self.forked_from_checkpoint_id, self.forked_from_revision_id = self.checkpoint_id, self.revision_id
        self.parent_revision_id = self.revision_id
        self.checkpoint_id, self.revision_id = uuid.uuid4().hex, uuid.uuid4().hex
        self.revision_history = history[-256:]
        self._loaded_checkpoint_path, self.created_at, self._revision_published = None, time.time(), False
        try:
            return self.save(path)
        except Exception:
            (self.checkpoint_id, self.parent_revision_id, self.root_lineage_id, self.revision_id,
             self._loaded_checkpoint_path, self.created_at, self._revision_published, self.revision_history,
             self.forked_from_checkpoint_id, self.forked_from_revision_id) = old
            raise

    @classmethod
    def create(cls, values: dict | None = None, description="", device="auto", seed=7):
        """a new model from settings by name (strings as a form sends them; blanks keep the defaults)"""
        base = CoreConfig().public_dict()
        for key, value in (values or {}).items():
            if key in base and str(value).strip() != "":
                kind = type(base[key])
                base[key] = (str(value).lower() in ("1", "true", "yes")) if kind is bool else kind(value)
        return cls(CoreConfig(**base), device=device, seed=int(seed), description=str(description))

    @classmethod
    def from_checkpoint(cls, path, device="auto", tokenizer_path=None):
        path = Path(path).expanduser().resolve()
        pointer = _read_json(Path(f"{path}.revision.json"))
        if pointer.get("state") == "pending":
            raise ValueError("checkpoint publication is unfinished")
        ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)   # mmap: one copy on an 8 GB Mac
        if ck.get("checkpoint_format") != CHECKPOINT_FORMAT or int(ck.get("checkpoint_version", 0)) != CHECKPOINT_VERSION:
            raise ValueError("unsupported somatic_core checkpoint")
        if pointer.get("state") == "complete" and pointer.get("revision_id") != ck.get("revision_id"):
            raise ValueError("checkpoint revision pointer does not match")
        model = cls(CoreConfig(**{"final_norm": 0, "bound_mode": "radial", **LEGACY_DEFAULTS, **ck["config"]}), device=device, seed=ck.get("seed", 7),
                    description=ck.get("description", ""))
        if model.tokenizer.sha256 != ck.get("tokenizer_sha256"):
            raise ValueError("checkpoint tokenizer differs from runtime tokenizer")
        core_state = unpack(ck.pop("core"))
        model.core.load_state_dict(core_state)
        del core_state
        gc.collect()
        model.stream_ctx = Matcher.from_state(unpack(ck["stream_ctx"]))
        model.chat_ctx = Matcher.from_state(unpack(ck["chat_ctx"]))
        # 1.3.0: the stream's unfilled tick; the reading context is rebuilt from the stream at first use
        model.pending = np.asarray(ck.get("pending", np.zeros(0, np.int64)), np.int64).reshape(-1)
        model._synced = False
        model.sampler.set_state(ck["sampler"])
        model._source_cursor = unpack(ck.get("source_cursor"))
        for k in ("checkpoint_id", "revision_id", "parent_revision_id", "root_lineage_id", "created_at",
                  "revision_history", "forked_from_checkpoint_id", "forked_from_revision_id"):
            setattr(model, k, ck.get(k))
        model.life = dict(ck.get("life", model.life))
        model._revision_published = True
        model._loaded_checkpoint_path = path
        return model
