"""Many people, one model, one clock: the lane engine (Nov 24; design in LANES.md, James's decisions there).

A lane is one person's chat: its own store (on its own clock: it ages with its own tokens), its learned stream and
reading context, its unlearned tail. All lanes share the learner, its optimiser, the gate and the mixer. round() takes
one whole tick from every lane that has one and learns them as one batched pass (lanes.batched_ticks) and one shared
update (the mean over the lanes that learned). Idle lanes are left out (masked): nothing of theirs moves. Generation
is per lane on the same clock (stage 1); a lane whose tail has filled waits for the next round, so what it reads next
always comes after its tick is learned, as in the single-stream runtime.

  eng = LaneEngine(soma, lanes=30)
  i = eng.attach(state)              # a person's chat (the chat bridge's context format), or None for a new one
  eng.feed(i, ids, learn=True)       # tokens join that person's stream
  eng.round()                        # one tick of the clock: every lane with a whole tick learns it
  p = eng.next_distribution(i)       # what that person's model says next (its reading context)
  state = eng.detach(i)              # their chat, to keep

One lane is the single-stream runtime exactly (tests/lane_engine.py).
"""
import copy
import dataclasses
import math

import numpy as np
import torch

from .core import Matcher
from .core2 import Learner3
from .core3 import Core3
from .lanes import _lite, batched_ticks


@dataclasses.dataclass
class Lane:
    core: Core3
    stream_ctx: Matcher
    chat_ctx: Matcher
    pending: np.ndarray
    synced: bool = True
    learn: bool = True
    chat: object = None                                    # whose: the caller's id for the person's chat; None = free
    totals: dict = dataclasses.field(default_factory=lambda: dict(bytes=0.0, tokens=0, ticks=0))


class LaneEngine:
    def __init__(self, soma, lanes: int):
        cfg, m = soma.config, soma.core
        assert int(cfg.mix_shared) >= 1, "lanes share the mixer: this model's store is not counts alone (mix_shared 0)"
        assert cfg.rate_ctl == "fixed" and not cfg.io2 and not cfg.rehearse, "lanes: a fixed rate"
        assert isinstance(m.net, Learner3) and m.net.cache_on, "lanes: the cached attention learner (att_cache)"
        self.soma, self.m, self.C, self.n = soma, m, int(cfg.chunk), int(lanes)
        # serving through lanes batches the store's read and update too (the model's own config says how it trains;
        # these change no maths a model keeps, only how lanes are computed)
        for k in ("lanes_read", "lanes_store"):
            m.c.setdefault(k, 1)
            if not int(m.c.get(k, 0)):
                m.c[k] = 1
        self.lanes = []
        for i in range(self.n):
            if i == 0:
                core = m                                   # the model's own core: lane 0 (its store, until replaced)
                ln = Lane(core, soma.stream_ctx, soma.chat_ctx, np.asarray(soma.pending, np.int64), soma._synced)
            else:
                core = Core3(_lite(cfg).public_dict(), str(soma.device), soma.blen)
                core.net, core.opt, core.theta = m.net, m.opt, m.theta
                core.wE, core.wM, core.mopt = m.wE, m.wM, m.mopt
                core.c = dict(m.c)
                if hasattr(m, "_vecs"):
                    core._vecs = m._vecs
                core._kv, core._kv_end = None, -1
                ln = Lane(core, Matcher(cfg.ring, cfg.minl), Matcher(cfg.ring, cfg.minl), np.zeros(0, np.int64))
            core._ext = True                               # gradients only: the shared update is the engine's
            self.lanes.append(ln)
        self.ticks = int(m.chunks)                         # every lane's learned ticks so far (the rate's warmup counts them)
        self._empty = None
        self._kv_views = None

    # ---- people in and out ----
    def empty_state(self) -> dict:
        if self._empty is None:
            fresh = Core3(_lite(self.soma.config).public_dict(), "cpu", self.soma.blen)
            mt = Matcher(self.soma.config.ring, self.soma.config.minl)
            self._empty = {"store": fresh.store_state(), "stream_ctx": mt.state(), "chat_ctx": mt.state(), "pending": [],
                           "synced": True}
        return self._empty

    def free(self) -> list:
        return [i for i, ln in enumerate(self.lanes) if ln.chat is None]

    def attach(self, chat, state: dict | None = None, lane: int | None = None) -> int:
        """a person's chat into a free lane -> the lane"""
        if lane is None:
            free = self.free()
            if not free:
                raise RuntimeError("every lane is in use")
            lane = free[0]
        ln = self.lanes[lane]
        assert ln.chat is None, f"lane {lane} is in use"
        s = state or self.empty_state()
        ln.core.load_store(s["store"])
        ln.stream_ctx = Matcher.from_state(copy.deepcopy(s["stream_ctx"]))
        ln.chat_ctx = Matcher.from_state(copy.deepcopy(s["chat_ctx"]))
        ln.pending = np.asarray(s["pending"], np.int64)
        ln.synced = bool(s["synced"])
        ln.chat, ln.totals = chat, dict(bytes=0.0, tokens=0, ticks=0)
        self._kv_views = None
        return lane

    def state(self, lane: int) -> dict:
        ln = self.lanes[lane]
        return {"store": ln.core.store_state(), "stream_ctx": ln.stream_ctx.state(), "chat_ctx": ln.chat_ctx.state(),
                "pending": [int(x) for x in ln.pending], "synced": bool(ln.synced)}

    def detach(self, lane: int) -> dict:
        s = self.state(lane)
        self.lanes[lane].chat = None
        return s

    # ---- the stream ----
    def feed(self, lane: int, ids, learn: bool = True, read: bool = True) -> None:
        """tokens join a person's stream: read into their reading context at once while their tick isn't full (as
        SOMA._feed), learned in whole ticks by round()"""
        ln = self.lanes[lane]
        ids = np.asarray(ids, np.int64).reshape(-1)
        if not len(ids):
            return
        if read and ln.synced and len(ln.pending) + len(ids) < self.C:
            getattr(ln.core, "read", ln.core.observe)(ln.chat_ctx, ids)
        else:
            ln.synced = False
        ln.pending = np.concatenate([ln.pending, ids])
        ln.learn = bool(learn)

    def ready(self) -> list:
        return [i for i, ln in enumerate(self.lanes) if ln.chat is not None and len(ln.pending) >= self.C]

    def round(self) -> dict:
        """one tick of the clock: every lane with a whole tick takes it; the learning lanes as one batched pass and one
        shared update; a lane reading without learning (learn off) reads its tick on its own. -> {lane: bits (C,)}"""
        ready = self.ready()
        out = {}
        if not ready:
            return out
        C = self.C
        learners = [i for i in ready if self.lanes[i].learn]
        for i in ready:
            if i not in learners:                          # learn off: scored, nothing moves
                ln = self.lanes[i]
                tick, ln.pending = ln.pending[:C], ln.pending[C:]
                out[i] = ln.core.step(ln.stream_ctx, tick, learn=False)
                ln.last_tick = tick
                ln.synced = False
        if learners:
            lanes = [self.lanes[i] for i in learners]
            ticks = []
            for ln in lanes:
                tick, ln.pending = ln.pending[:C], ln.pending[C:]
                ticks.append(tick)
            bits = batched_ticks(self, self.m, [ln.core for ln in lanes], [ln.stream_ctx for ln in lanes], ticks,
                                 stride=len(lanes))
            n = len(lanes)
            self.ticks += n
            self.m.apply_update(max(0, math.ceil(self.ticks / n) - 1), n)
            for i, ln, tick, b in zip(learners, lanes, ticks, bits):
                ln.last_tick = tick
                ln.synced = False                          # the weights moved: its reading context is rebuilt at next use
                nb = float(self.soma.blen[tick].sum())
                ln.totals["bytes"] += nb; ln.totals["tokens"] += C; ln.totals["ticks"] += 1
                self.soma.life["bytes"] += nb; self.soma.life["tokens"] += C; self.soma.life["scored_bytes"] += nb
                out[i] = b
        return out

    # ---- what a person's model says next ----
    def _window(self, ln: Lane) -> None:
        """its reading context brought up to its stream (SOMA._chat_window)"""
        if ln.synced:
            return
        ln.chat_ctx = ln.stream_ctx.clone()
        getattr(ln.core, "read", ln.core.observe)(ln.chat_ctx, ln.pending)
        ln.synced = True

    def next_distribution(self, lane: int) -> torch.Tensor:
        ln = self.lanes[lane]
        if len(ln.pending) >= self.C:                      # its tick is full: learned first, as the single stream would
            self.round()
        self._window(ln)
        return ln.core.next_distribution(ln.chat_ctx)
