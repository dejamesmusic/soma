"""somatic_core: the engine. A bounded-state continual learner in token units (soma-8k-v1 BPE), ported from the
research script investigations/2026-11-05-the-bet/hybrid_t.py (Nov 8: 1.198 bpb at 5.5M bytes of simplestories, one
stream in order, prequential; the BPE transformer probe on the same stream: 1.248).

THE STORE (fixed size; nothing grows)
  order 0      one dense row of counts over the vocabulary.
  order 1      a slot table per previous token (V nodes x K1 successor slots).
  order k >= 2 a pool of P_k nodes, each owning one exact k-token context. A new context takes a free node, else
               evicts the least-used of 8 sampled nodes (use decays at ctx_decay per token). Pool.
  slots        each node keeps K_k successor slots: (token, counts). A new successor takes an empty slot, else
               replaces the slot with the least all-time count not touched this chunk, else the trash slot. The
               replaced mass stays in the node total, so the back-off inherits it. Slots.
  timescales   every count is kept on a phi ladder of decays (tau = phi^e tokens) plus a slowest rung that halves
               when its order-0 total passes alltime_cap tokens (so float32 stays exact on long runs).
READERS
  exact        Witten-Bell back-off along the chain per rung, rungs mixed in log space with weights chosen by the
               deepest order present, plus a token match model (the last earlier occurrence of the last >= minl
               tokens); weights trained online on the exact reader's own loss.
  learned      an MLP over the last W token embeddings and the chain's log-distributions (through a shared
               projection V -> proj), trained with Adam on its OWN loss.
  final        p = s p_exact + (1 - s) p_learned, s = sigmoid(theta[deepest order, match length, entropy bin]),
               theta trained online on the mixture's loss.
ORDER OF A CHUNK (strictly prequential)
  contexts and match predictions are formed from tokens before each position; the chunk is scored with the store as
  it stood at the chunk start; only then are successor slots assigned, replaced mass removed and counts added.
  (hybrid_t.py assigned slots before scoring, which leaks a little of the answer on overflow; parity=True keeps
  that order, for checking this port against the research script only.)
"""
from __future__ import annotations

import math

import numpy as np
import torch

PHI = 1.618034
LB = 6                                    # match-length buckets (0: no match)
ENT_EDGES = (1.0, 2.0, 4.0, 6.0)          # entropy bins (bits) of the exact reader, for the final mix gate
NE = len(ENT_EDGES) + 1


class Matcher:
    """the history of one context (a training stream, a chat): a ring of the last `ring` tokens, a short tail of
    python ints for keys, and the token match model over the ring. Bounded: the key map is pruned past 4 rings."""

    def __init__(self, ring: int, minl: int):
        self.ring, self.minl = int(ring), int(minl)
        self.R = np.zeros(self.ring, np.int64)
        self.t = 0
        self.mp, self.ml = -1, 0
        self.last: dict = {}
        self.tail: list = []
        self.ready, self.cur = False, (-1, 0)

    def window(self, w: int) -> list:
        tail = self.tail[-w:]
        return [0] * (w - len(tail)) + tail

    def peek(self):
        """(predicted token or -1, match-length bucket) for the next position; computed once per position."""
        if self.ready:
            return self.cur
        t, r, mn = self.t, self.ring, self.minl
        if self.mp >= 0 and t - self.mp < r:
            self.ml += 1
        else:
            self.mp, self.ml = -1, 0
        if self.mp < 0 and t >= mn:
            q = self.last.get(tuple(self.tail[-mn:]))
            if q is not None and t - q < r:
                self.mp, self.ml = q, mn
                R = self.R
                while (self.ml < 32 and q - self.ml - 1 >= 0 and t - (q - self.ml - 1) < r
                       and R[(q - self.ml - 1) % r] == R[(t - self.ml - 1) % r]):
                    self.ml += 1
        if self.mp >= 0:
            self.cur = (int(self.R[self.mp % r]), min(LB - 1, 1 + int(math.log2(max(1, self.ml - mn + 1)))))
        else:
            self.cur = (-1, 0)
        self.ready = True
        return self.cur

    def push(self, y: int):
        self.peek()
        t, r, y = self.t, self.ring, int(y)
        self.R[t % r] = y
        if self.mp >= 0:
            self.mp = self.mp + 1 if self.R[self.mp % r] == y else -1
        if t >= self.minl:
            self.last[tuple(self.tail[-self.minl:])] = t
            if len(self.last) > 4 * r:
                self.last = {k: v for k, v in self.last.items() if t - v < r}
        self.tail.append(y)
        if len(self.tail) > 128:
            del self.tail[:64]
        self.t += 1
        self.ready = False
        if self.t >= self.rebase_at:
            self.rebase()

    rebase_at = 1 << 30

    def rebase(self):
        """shift every position back by whole rings (nothing grows; ring slots and all differences are unchanged)."""
        off = (self.t // self.ring - 1) * self.ring
        if off <= 0:
            return
        self.t -= off
        if self.mp >= 0:
            self.mp -= off
        self.last = {k: v - off for k, v in self.last.items() if v - off >= 0}

    def state(self) -> dict:
        keys = np.array(list(self.last.keys()), np.int64).reshape(-1, self.minl)
        return dict(ring=self.ring, minl=self.minl, R=self.R.copy(), t=self.t, mp=self.mp, ml=self.ml,
                    keys=keys, vals=np.array(list(self.last.values()), np.int64),
                    tail=np.array(self.tail, np.int64), ready=self.ready, cur=list(self.cur))

    @classmethod
    def from_state(cls, s: dict) -> "Matcher":
        m = cls(int(s["ring"]), int(s["minl"]))
        m.R = np.asarray(s["R"], np.int64).copy()
        m.t, m.mp, m.ml = int(s["t"]), int(s["mp"]), int(s["ml"])
        m.last = {tuple(k): int(v) for k, v in zip(np.asarray(s["keys"]).tolist(), np.asarray(s["vals"]).tolist())}
        m.tail = [int(v) for v in np.asarray(s["tail"]).tolist()]
        m.ready, m.cur = bool(s["ready"]), (int(s["cur"][0]), int(s["cur"][1]))
        return m

    def clone(self) -> "Matcher":
        return Matcher.from_state(self.state())


class IntIndex:
    """key -> node for non-negative int keys, in two numpy arrays (1.2.1). Open addressing, linear probing, load <= 0.5,
    deletion by backward shift (no tombstones). Replaces a dict of tuples that held ~300 bytes a node as Python objects
    (909 MiB at the 768 model's 3.1M pool nodes; this is ~70 MB). Membership is all that matters: same answers."""

    MUL = 0x9E3779B97F4A7C15

    def __init__(self, P: int):
        self.bits = max(4, int(2 * max(1, P) - 1).bit_length())
        self.mask = (1 << self.bits) - 1
        self.keys = np.full(1 << self.bits, -1, np.int64)
        self.vals = np.zeros(1 << self.bits, np.int32)

    def _home(self, k: int) -> int:
        return ((k * self.MUL) & 0xFFFFFFFFFFFFFFFF) >> (64 - self.bits)

    def get(self, k: int, default=-1):
        K, i = self.keys, self._home(k)
        while True:
            x = K[i]
            if x == k:
                return int(self.vals[i])
            if x < 0:
                return default
            i = (i + 1) & self.mask

    def set(self, k: int, v: int):
        K, i = self.keys, self._home(k)
        while K[i] >= 0 and K[i] != k:
            i = (i + 1) & self.mask
        K[i] = k
        self.vals[i] = v

    def delete(self, k: int):
        K, V, m, i = self.keys, self.vals, self.mask, self._home(k)
        while K[i] != k:
            if K[i] < 0:
                return
            i = (i + 1) & m
        j = i
        while True:                                   # backward shift: pull later entries of the run into the hole
            j = (j + 1) & m
            x = K[j]
            if x < 0:
                break
            h = self._home(int(x))
            if (j > i and (h <= i or h > j)) or (j < i and (h <= i and h > j)):
                K[i], V[i] = x, V[j]
                i = j
        K[i] = -1

    def build(self, keys: np.ndarray, vals: np.ndarray):
        """bulk insert (a load), vectorised in probe rounds: an entry takes home + r when that slot is still empty."""
        self.keys.fill(-1)
        keys = keys.astype(np.int64); vals = vals.astype(np.int32)
        home = (((keys.astype(np.uint64) * np.uint64(self.MUL)) >> np.uint64(64 - self.bits))).astype(np.int64)
        todo = np.arange(len(keys)); r = 0
        while len(todo):
            pos = (home[todo] + r) & self.mask
            free = self.keys[pos] < 0
            cand, cpos = todo[free], pos[free]
            _, first = np.unique(cpos, return_index=True)
            put, ppos = cand[first], cpos[first]
            self.keys[ppos] = keys[put]; self.vals[ppos] = vals[put]
            done = np.zeros(len(keys), bool); done[put] = True
            todo = todo[~done[todo]]
            r += 1


def _key(t) -> int:
    """a context tuple of tokens (< 2^16 each) as one int"""
    k = 0
    for i, x in enumerate(t):
        k |= int(x) << (16 * i)
    return k


class Pool:
    """P nodes for the exact k-token contexts of one order; LFU-of-8-sampled eviction (ctxbank.allocate, online).
    1.2.1: the index, scores and times are numpy (IntIndex, float64, int64), not Python objects; same arithmetic."""

    def __init__(self, k: int, P: int, gamma: float, seed: int):
        self.k, self.P = int(k), int(P)
        self.lg = math.log(gamma)
        self.map = IntIndex(self.P)
        self.keys = np.full((self.P, self.k), -1, np.int32)
        self.score = np.zeros(self.P, np.float64)
        self.last = np.zeros(self.P, np.int64)
        self.nxt = 0
        self.rng = np.random.default_rng(seed)
        self.recycles = 0

    def lookup(self, key) -> int:
        return self.map.get(_key(key), -1)

    def touch(self, key, t: int):
        """the node for `key` at token time t, allocating (and possibly evicting) on a first sight. -> (node, evicted)"""
        kk = _key(key)
        n = self.map.get(kk, None)
        evicted = False
        sc, la = self.score, self.last
        if n is None:
            if self.nxt < self.P:
                n = self.nxt
                self.nxt += 1
            else:
                lg = self.lg
                cand = self.rng.integers(0, self.P, size=8)
                n = int(min(cand, key=lambda c: float(sc[c]) * math.exp(lg * (t - int(la[c])))))
                self.map.delete(_key(self.keys[n].tolist()))
                evicted = True
                self.recycles += 1
            self.map.set(kk, n)
            self.keys[n] = key
            sc[n] = 0.0
            la[n] = t
        sc[n] = float(sc[n]) * math.exp(self.lg * (t - int(la[n]))) + 1.0
        la[n] = t
        return n, evicted

    def state(self) -> dict:
        return dict(k=self.k, P=self.P, lg=self.lg, keys=self.keys.copy(), score=self.score.copy(),
                    last=self.last.copy(), nxt=self.nxt, rng=self.rng.bit_generator.state, recycles=self.recycles)

    def load(self, s: dict):
        self.keys = np.asarray(s["keys"], np.int32).copy()
        self.score = np.asarray(s["score"], np.float64).copy()
        self.last = np.asarray(s["last"], np.int64).copy()
        self.nxt, self.recycles = int(s["nxt"]), int(s["recycles"])
        self.rng.bit_generator.state = s["rng"]
        used = np.nonzero(self.keys[:, 0] >= 0)[0]
        kk = np.zeros(len(used), np.int64)
        for i in range(self.k):
            kk |= self.keys[used, i].astype(np.int64) << (16 * i)
        self.map.build(kk, used)


class Slots:
    """the successor-slot map of one order, on the CPU: token, all-time count and last chunk per (node, slot)."""

    def __init__(self, P: int, K: int):
        self.P, self.K = int(P), int(K)
        self.tok = np.full((self.P, self.K), -1, np.int32)
        self.cnt = np.zeros((self.P, self.K), np.int32)
        self.ch = np.full((self.P, self.K), -1, np.int64)
        self.overflows = 0
        self.trash = 0
        self.assignments = 0

    def reset(self, n: int):
        self.tok[n] = -1
        self.cnt[n] = 0
        self.ch[n] = -1

    def assign(self, n: int, y: int, ch: int):
        """the slot of successor y at node n in chunk ch. -> (slot, new): new means its old counts must be zeroed."""
        row = self.tok[n]
        hit = np.nonzero(row == y)[0]
        if len(hit):
            j, new = int(hit[0]), False
        else:
            emp = np.nonzero(row < 0)[0]
            if len(emp):
                j = int(emp[0])
            else:
                self.overflows += 1
                ok = np.nonzero(self.ch[n] != ch)[0]
                if not len(ok):
                    self.trash += 1
                    return self.K, False
                j = int(ok[np.argmin(self.cnt[n, ok])])
            self.tok[n, j] = y
            self.cnt[n, j] = 0
            new = True
            self.assignments += 1
        if self.cnt[n, j] < 2**31 - 1:
            self.cnt[n, j] += 1
        self.ch[n, j] = ch
        return j, new

    def state(self) -> dict:
        return dict(tok=self.tok.copy(), cnt=self.cnt.copy(), ch=self.ch.copy(), overflows=self.overflows,
                    trash=self.trash, assignments=self.assignments)

    def load(self, s: dict):
        self.tok = np.asarray(s["tok"], np.int32).copy()
        self.cnt = np.asarray(s["cnt"], np.int32).copy()
        self.ch = np.asarray(s["ch"], np.int64).copy()
        self.overflows, self.trash, self.assignments = int(s["overflows"]), int(s["trash"]), int(s["assignments"])


class Learner(torch.nn.Module):
    def __init__(self, V, W, D, H, DP, O, NB):
        super().__init__()
        self.emb = torch.nn.Embedding(V, D)
        self.win = torch.nn.Linear(W * D, H)
        self.pr = torch.nn.Linear(V, DP, bias=False)
        self.cell = torch.nn.Linear(O * NB * DP, H, bias=False)
        self.ln = torch.nn.LayerNorm(O * NB * DP)
        self.l2 = torch.nn.Linear(H, H)
        self.out = torch.nn.Linear(H, V)

    def forward(self, win, lch):
        f = self.ln(self.pr((torch.clamp(lch, min=-14.0) + 7.0) / 3.0).reshape(lch.shape[0], -1))
        h = torch.nn.functional.gelu(self.win(self.emb(win).reshape(win.shape[0], -1)) + self.cell(f))
        return self.out(torch.nn.functional.gelu(self.l2(h)))


DEFAULTS = dict(
    # identity (fixed at creation)
    kmax=3, k1=256, p2=262144, k2=32, p3=524288, k3=16, p4=524288, k4=16,
    ladder="3,7,11,15,19,23", timescales="", skips="", skip_size=3, window=16, emb=64, hidden=1024, proj=16, minl=3, ring=65536, ctx_decay=0.99999,
    alltime_cap=8388608, seed=0,
    # dynamics (may change on a continuation)
    chunk=16, lr=1e-3, mix_lr=0.002, gate_lr=0.02,
)
IDENTITY = ("kmax", "k1", "p2", "k2", "p3", "k3", "p4", "k4", "ladder", "window", "emb", "hidden", "proj", "minl",
            "ring", "ctx_decay", "alltime_cap", "seed")


class Core:
    """the store, the readers and the mix; contexts (Matcher) are passed in, so a training stream and a chat share
    the store and the learner but keep their own histories."""

    def __init__(self, cfg: dict, dev, blen: np.ndarray, parity: bool = False):
        c = dict(DEFAULTS)
        c.update(cfg)
        self.c = c
        self.dev = torch.device(dev)
        self.parity = bool(parity)
        self.blen = np.asarray(blen, np.int64)
        self.V = V = len(self.blen)
        torch.manual_seed(int(c["seed"]))
        self.KMAX = int(c["kmax"])
        assert 1 <= self.KMAX <= 4
        # dev (Nov 11; James: "a less strict match may actually help the present with less exact context"): skip
        # contexts, exact keys over non-adjacent tokens ("3.2" = the tokens 3 and 2 back, skipping the last). They are
        # extra orders after the chain: counted and read like any order, fed to the learner, outside the back-off chain
        # and the exact mix. Empty: unchanged.
        sk = str(c.get("skips", "") or "")
        self.skips = [tuple(int(o) for o in g.split(".")) for g in sk.split(";")] if sk else []
        self.KCH = self.KMAX
        self.O = self.KMAX + 1 + len(self.skips)
        self.lad = tuple(int(v) for v in str(c["ladder"]).split(","))
        # dev (Nov 10; James: "phi is an inherited belief. worth putting it up to the light"): explicit timescales in
        # tokens override the phi ladder; rung b then decays by 1/tau_b per token. Empty: the phi ladder, unchanged.
        ts = str(c.get("timescales", "") or "")
        self.taus = tuple(float(v) for v in ts.split(",")) if ts else tuple(PHI ** e for e in self.lad)
        self.NB = len(self.taus) + 1
        self.PK = {1: V, 2: int(c["p2"]), 3: int(c["p3"]), 4: int(c["p4"])}
        self.KK = {1: int(c["k1"]), 2: int(c["k2"]), 3: int(c["k3"]), 4: int(c["k4"])}
        self.orders = list(range(1, self.KMAX + 1 + len(self.skips)))
        self.skip_offs = {self.KMAX + 1 + j: offs for j, offs in enumerate(self.skips)}
        self.skip_like = int(c.get("skip_size", 3) or 3)        # skip orders are sized like this order (3; 2 is cheaper)
        for k in self.skip_offs:
            self.PK[k], self.KK[k] = int(c[f"p{self.skip_like}"]), int(c[f"k{self.skip_like}"])
        self.pools = {k: Pool(len(self.skip_offs.get(k, range(k))), self.PK[k], float(c["ctx_decay"]), k)   # key width
                      for k in self.orders if k >= 2}
        dense = getattr(self, "_dense_slots", True)            # core3 keeps slot counts in a pair pool instead
        self.slots = {k: Slots(self.PK[k], self.KK[k]) for k in self.orders} if dense else {}
        d, NB = self.dev, self.NB
        self.ab = torch.tensor([PHI ** -e for e in self.lad] + [0.0], device=d) if not ts else \
            torch.tensor([1.0 / t for t in self.taus] + [0.0], device=d)
        self.ar = torch.arange(NB, device=d)
        self.r0 = torch.zeros(NB, V, device=d)
        self.n0 = torch.zeros(NB, device=d)
        self.sc = {k: torch.zeros(self.PK[k] + 1, NB, self.KK[k] + 1, device=d) for k in self.orders} if dense else {}
        self.st = {k: torch.zeros(self.PK[k] + 1, NB, device=d) for k in self.orders}
        self.sl = {k: torch.zeros(self.PK[k] + 1, dtype=torch.long, device=d) for k in self.orders}
        self.net = Learner(V, int(c["window"]), int(c["emb"]), int(c["hidden"]), int(c["proj"]), self.O, NB).to(d)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(c["lr"]))
        self.wE = torch.zeros(self.O, NB, device=d)
        self.wE[:, -1] = 1.0
        self.wE.requires_grad_(True)
        self.wM = torch.zeros(self.O, LB, device=d, requires_grad=True)
        self.theta = torch.zeros(self.O * LB * NE, device=d)
        self.edges = torch.tensor(ENT_EDGES, device=d)
        self.clock = 0                    # tokens learned (the store's time)
        self.r0_t = 0                     # the clock when the order-0 row was last decayed
        self.n0_all = 0.0                 # the slowest rung's order-0 total, tracked on the CPU
        self.halvings = 0
        self.chunks = 0
        self.acc = torch.zeros(6, device=d)   # bits exact, learner, final, (unused), correct, scored
        self.last = {}

    # ---- sizes ----
    def params(self) -> int:
        return sum(p.numel() for p in self.net.parameters()) + self.wE.numel() + self.wM.numel() + self.theta.numel()

    def store_floats(self) -> int:
        return sum(self.sc[k].numel() + self.st[k].numel() for k in self.orders) + self.r0.numel()

    def set_dynamics(self, chunk=None, lr=None, mix_lr=None, gate_lr=None):
        for k, v in (("chunk", chunk), ("lr", lr), ("mix_lr", mix_lr), ("gate_lr", gate_lr)):
            if v is not None:
                self.c[k] = type(DEFAULTS[k])(v)
        for g in self.opt.param_groups:
            g["lr"] = float(self.c["lr"])

    # ---- the CPU pass of one chunk ----
    def _contexts(self, m, ys, learn: bool):
        """node ids per order and position, match predictions and windows, advancing the context's history.
        ys may be None (a single read of the next position, nothing pushed).
        m may be a list of B Matchers (lanes, a dev feature, Nov 10): the chunk is then B equal blocks, block j from
        lane j's stream with lane j's own history. The store and learner see the blocks in order, as one stream."""
        C = 1 if ys is None else len(ys)
        lanes = list(m) if isinstance(m, (list, tuple)) else [m]
        assert C % len(lanes) == 0, "a laned chunk is B equal blocks"
        Lc = C // len(lanes)
        W = int(self.c["window"])
        ids = np.full((len(self.orders), C), -1, np.int64)
        rec = {k: [] for k in self.orders}
        mp = np.full(C, -1, np.int64)
        mb = np.zeros(C, np.int64)
        win = np.zeros((C, W), np.int64)
        for i in range(C):
            m = lanes[i // Lc]
            tail = m.tail
            n_hist = m.t
            win[i] = m.window(W)
            if n_hist >= 1:
                ids[0, i] = tail[-1]
            for k in self.orders[1:]:
                offs = self.skip_offs.get(k)
                if offs is not None and n_hist >= max(offs):
                    key = tuple(tail[-o] for o in offs)
                    if learn:
                        n, ev = self.pools[k].touch(key, self.clock + i)
                        if ev:
                            rec[k].append((i, n))
                    else:
                        n = self.pools[k].lookup(key)
                    ids[k - 1, i] = n
                elif offs is None and n_hist >= k:
                    key = tuple(tail[-k:])
                    if learn:
                        n, ev = self.pools[k].touch(key, self.clock + i)
                        if ev:
                            rec[k].append((i, n))
                    else:
                        n = self.pools[k].lookup(key)
                    ids[k - 1, i] = n
            mp[i], mb[i] = m.peek()
            if ys is not None:
                m.push(int(ys[i]))
        return ids, rec, mp, mb, win

    def _assign(self, ids, rec, ys):
        """successor slots for the chunk (with each recycle at its position). -> slot index, new (node, slot) per order"""
        C = len(ys)
        slot, new = {}, {}
        for k in self.orders:
            S = self.slots[k]
            sk = np.full(C, S.K, np.int64)
            nk = []
            ev = rec[k]
            e = 0
            for i in range(C):
                while e < len(ev) and ev[e][0] == i:
                    S.reset(ev[e][1])
                    e += 1
                n = ids[k - 1, i]
                if n < 0:
                    continue
                j, fresh = S.assign(int(n), int(ys[i]), self.chunks)
                sk[i] = j
                if fresh:
                    nk.append((int(n), j))
            slot[k], new[k] = sk, nk
        return slot, new

    def _maps(self, ids):
        out = {}
        for k in self.orders:
            S = self.slots[k]
            n = ids[k - 1]
            M = np.full((len(n), S.K + 1), -1, np.int64)
            ok = n >= 0
            M[ok, :S.K] = S.tok[n[ok]]
            out[k] = torch.from_numpy(np.maximum(M, 0)).to(self.dev)
        return out

    # ---- the GPU read ----
    def _read(self, idt, maps, mp, mb, win, decay_now: int, grad: bool):
        """-> lch (C,O,NB,V), lpE, lpL (C,V), gate key (C,), deep (C,)"""
        C, V, NB, d = win.shape[0], self.V, self.NB, self.dev
        keep = 1.0 - self.ab
        with torch.no_grad():
            f0 = keep ** float(decay_now - self.r0_t)
            cs = [(self.r0 * f0[:, None]).expand(C, NB, V)]
            ns = [(self.n0 * f0).expand(C, NB)]
            for k in self.orders:
                idc = idt[k]
                lag = (decay_now - self.sl[k][idc]).float()
                f = keep[None] ** lag[:, None]
                rows = self.sc[k][idc] * f[..., None]
                cs.append(torch.zeros(C, NB, V, device=d).scatter_add_(
                    -1, maps[k][:, None, :].expand(C, NB, self.KK[k] + 1), rows))
                ns.append(self.st[k][idc] * f)
            p = torch.full((C, NB, V), 1.0 / V, device=d)
            chain = []
            last_band = self.ar[None, :, None] == NB - 1
            for k in range(self.O):
                ck, nk = cs[k], ns[k][..., None]
                dist = (ck > 0.05).sum(-1, keepdim=True).float()
                if k > 0:
                    lost = torch.clamp(nk - ck.sum(-1, keepdim=True), min=0.0)
                    bt = torch.where(last_band, dist.clamp(min=1.0), dist + 1.0)          # at least one type (core2)
                else:
                    lost, bt = 0.0, torch.ones_like(nk)
                p = (ck + (bt + lost) * p) / (nk + bt)
                chain.append(p)
            lch = torch.log(torch.stack(chain, 1))
            deep = torch.stack([(ns[k][:, -1] > 0) * k for k in range(self.O)], 1).max(1).values
            mpt = torch.from_numpy(mp).to(d)
            mbt = torch.from_numpy(mb).to(d)
            hm = mpt >= 0
            Lm = torch.full((C, V), math.log(0.02 / V), device=d)
            Lm[hm, mpt[hm]] = math.log(0.98 + 0.02 / V)
        zE = (self.wE[deep][:, :, None] * lch[:, -1]).sum(1) + (self.wM[deep, mbt] * hm.float())[:, None] * Lm
        lpE = torch.log_softmax(zE, -1)
        wt = torch.from_numpy(win).to(d)
        if grad:
            lpL = torch.log_softmax(self.net(wt, lch), -1)
        else:
            with torch.no_grad():
                lpL = torch.log_softmax(self.net(wt, lch), -1)
        with torch.no_grad():
            ent = -(lpE.exp() * lpE).sum(-1) / math.log(2)
            key = (deep * LB + mbt) * NE + torch.bucketize(ent, self.edges)
        return lch, lpE, lpL, key, deep

    transfers = "queued"                                  # cuda only: "queued", "pinned" or "blocking" (diagnostics)

    def _dv(self, a):
        """a host array on the device. On CUDA the copy is queued behind the GPU's work instead of waiting for it: a
        blocking copy (`.to(dev)`, `torch.tensor(..., device=dev)`, or a Python scalar assigned through an index) drains
        the whole queue first, and the tick had 48 of them, so the CPU never ran ahead of the GPU (Nov 10)."""
        t = torch.from_numpy(a) if isinstance(a, np.ndarray) else a
        if self.dev.type != "cuda" or self.transfers == "blocking":
            return t.to(self.dev)
        if self.transfers == "pinned" and t.numel():
            t = t.pin_memory()
        return t.to(self.dev, non_blocking=True)

    def _idt(self, ids):
        return {k: self._dv(np.where(ids[k - 1] < 0, self.PK[k], ids[k - 1])) for k in self.orders}

    # ---- one chunk ----
    def step(self, m: Matcher, ys, learn: bool = True):
        """score the chunk ys (before learning from it), then learn from it if `learn`. Accumulates bits into self.acc
        and returns the chunk's per-token bits of the final mix (a GPU tensor)."""
        ys = np.asarray(ys, np.int64)
        C = len(ys)
        d = self.dev
        t0 = self.clock
        ids, rec, mp, mb, win = self._contexts(m, ys, learn)
        idt = self._idt(ids)
        if learn:
            with torch.no_grad():                                     # recycled nodes start from zero for their new context
                for k in self.orders:
                    if rec[k]:
                        nd = torch.tensor([n for _, n in rec[k]], device=d)
                        self.sc[k][nd] = 0.0
                        self.st[k][nd] = 0.0
                        self.sl[k][nd] = t0
        if learn and self.parity:
            slot, new = self._assign(ids, rec, ys)
            self._zero_new(new)
            maps = self._maps(ids)
        else:
            maps = self._maps(ids)
        y = torch.from_numpy(ys).to(d)
        lch, lpE, lpL, key, deep = self._read(idt, maps, mp, mb, win, t0, grad=learn)
        lyE = lpE.gather(1, y[:, None])[:, 0]
        lyL = lpL.gather(1, y[:, None])[:, 0]
        with torch.no_grad():
            sg = torch.sigmoid(self.theta[key])
            a_, b_ = lyE.detach().exp(), lyL.detach().exp()
            pf = sg * a_ + (1 - sg) * b_
            pfull = sg[:, None] * lpE.detach().exp() + (1 - sg)[:, None] * lpL.detach().exp()
            correct = (pfull.argmax(-1) == y).float().sum()
            bits = torch.stack([-lyE.detach().sum(), -lyL.detach().sum(), -torch.log(pf).sum()]) / math.log(2)
            self.acc[:3] += bits
            self.acc[4] += correct
            self.acc[5] += C
        self.last = dict(tokens=ys, bits=-torch.log2(pf), next=pfull[-1].topk(5))
        if not learn:
            return -torch.log2(pf)
        # ---- learn ----
        loss = -lyL.mean()
        lossE = -lyE.mean()
        self.opt.zero_grad()
        loss.backward()
        gE = torch.autograd.grad(lossE, [self.wE, self.wM])
        with torch.no_grad():
            self.theta.index_put_((key,), float(self.c["gate_lr"]) * sg * (1 - sg) * (a_ - b_) / pf, accumulate=True)
        self.opt.step()
        if not self.parity:
            slot, new = self._assign(ids, rec, ys)
        with torch.no_grad():
            lr_e = float(self.c["mix_lr"]) * C
            self.wE -= lr_e * gE[0]
            self.wM -= lr_e * gE[1]
            self._zero_new(new) if not self.parity else None
            keep = 1.0 - self.ab
            self.r0 *= (keep ** float(t0 - self.r0_t))[:, None]
            self.n0 *= keep ** float(t0 - self.r0_t)
            self.r0_t = t0
            NB = self.NB
            self.r0.index_put_((self.ar[None].expand(C, NB), y[:, None].expand(C, NB)), torch.ones(C, NB, device=d),
                               accumulate=True)
            self.n0 += C
            h1 = torch.ones(C, NB, device=d)
            ib = self.ar[None].expand(C, NB)
            for k in self.orders:
                idc = idt[k]
                U = torch.unique(idc)
                U = U[U != self.PK[k]]
                f = keep[None] ** (t0 - self.sl[k][U]).float()[:, None]
                self.sc[k][U] *= f[..., None]
                self.st[k][U] *= f
                self.sl[k][U] = t0
                ix = idc[:, None].expand(C, NB)
                sj = torch.from_numpy(slot[k]).to(d)[:, None].expand(C, NB)
                self.sc[k].index_put_((ix, ib, sj), h1, accumulate=True)
                self.st[k].index_put_((ix, ib), h1, accumulate=True)
                self.sc[k][self.PK[k]] = 0.0
                self.st[k][self.PK[k]] = 0.0
                self.sc[k][idc, :, self.KK[k]] = 0.0          # trash slots of the touched nodes only
        self.clock += C
        self.chunks += 1
        self.n0_all += C
        if self.n0_all > float(self.c["alltime_cap"]):
            self._halve()
        return -torch.log2(pf)

    def _zero_new(self, new):
        for k in self.orders:
            if new[k]:
                nn = torch.tensor([n for n, _ in new[k]], device=self.dev)
                jj = torch.tensor([j for _, j in new[k]], device=self.dev)
                self.sc[k][nn, :, jj] = 0.0

    @torch.no_grad()
    def _halve(self):
        """the slowest rung halves: its counts stay exact in float32 however long the run (bounded numerics)."""
        self.r0[-1] *= 0.5
        self.n0[-1] *= 0.5
        for k in self.orders:
            self.sc[k][:, -1] *= 0.5
            self.st[k][:, -1] *= 0.5
        self.n0_all *= 0.5
        self.halvings += 1

    def learn(self, m: Matcher, ys, learn: bool = True):
        """any number of tokens, in chunks of `chunk`."""
        ys = np.asarray(ys, np.int64).reshape(-1)
        C = int(self.c["chunk"])
        for i in range(0, len(ys), C):
            self.step(m, ys[i:i + C], learn)

    def observe(self, m: Matcher, ys):
        """read tokens into a context's history only (no scoring, no learning)."""
        for y in np.asarray(ys, np.int64).reshape(-1):
            m.peek()
            m.push(int(y))

    @torch.no_grad()
    def next_distribution(self, m: Matcher) -> torch.Tensor:
        """the final mix's distribution over the next token of context m (changes nothing but m's match cursor,
        which is the same computation the next push would make)."""
        ids, rec, mp, mb, win = self._contexts(m, None, False)
        lch, lpE, lpL, key, deep = self._read(self._idt(ids), self._maps(ids), mp, mb, win, self.clock, grad=False)
        sg = torch.sigmoid(self.theta[key])
        return (sg[:, None] * lpE.exp() + (1 - sg)[:, None] * lpL.exp())[0]

    def drain(self):
        v = self.acc.tolist()
        self.acc.zero_()
        return dict(bits_exact=v[0], bits_learner=v[1], bits_final=v[2], correct=v[4], scored=v[5])

    def stats(self) -> dict:
        out = dict(chunks=self.chunks, clock=self.clock, halvings=self.halvings,
                   gate_exact=float(torch.sigmoid(self.theta).mean()))
        for k in self.orders:
            S = self.slots[k]
            used = (self.pools[k].nxt if k >= 2 else (int((S.tok[:, 0] >= 0).sum()) if hasattr(S, "tok") else int((S.ptok[np.where(S.ptab[:, 0] >= 0, S.ptab[:, 0], S.NPAGE), 0] >= 0).sum())))
            out[f"o{k}_nodes"] = used
            out[f"o{k}_fill"] = used / self.PK[k]
            out[f"o{k}_overflows"] = S.overflows
            out[f"o{k}_trash"] = S.trash
            if k >= 2:
                out[f"o{k}_recycles"] = self.pools[k].recycles
        return out

    # ---- state ----
    def state_dict(self) -> dict:
        if self.dev.type == "mps":
            torch.mps.synchronize()
        cpu = lambda t: t.detach().to("cpu").clone()
        return dict(
            config=dict(self.c), clock=self.clock, r0_t=self.r0_t, n0_all=self.n0_all, halvings=self.halvings,
            chunks=self.chunks, r0=cpu(self.r0), n0=cpu(self.n0),
            sc={str(k): cpu(self.sc[k]) for k in self.orders}, st={str(k): cpu(self.st[k]) for k in self.orders},
            sl={str(k): cpu(self.sl[k]) for k in self.orders},
            slots={str(k): self.slots[k].state() for k in self.orders},
            pools={str(k): self.pools[k].state() for k in self.pools},
            net=self.net.state_dict(), opt=self.opt.state_dict(), wE=cpu(self.wE), wM=cpu(self.wM),
            theta=cpu(self.theta),
        )

    def load_state_dict(self, s: dict):
        d = self.dev
        self.clock, self.r0_t, self.n0_all = int(s["clock"]), int(s["r0_t"]), float(s["n0_all"])
        self.halvings, self.chunks = int(s["halvings"]), int(s["chunks"])
        self.r0.copy_(s["r0"].to(d))
        self.n0.copy_(s["n0"].to(d))
        for k in self.orders:
            self.sc[k].copy_(s["sc"][str(k)].to(d))
            self.st[k].copy_(s["st"][str(k)].to(d))
            self.sl[k].copy_(s["sl"][str(k)].to(d))
            self.slots[k].load(s["slots"][str(k)])
        for k in self.pools:
            self.pools[k].load(s["pools"][str(k)])
        self.net.load_state_dict(s["net"])
        self.opt.load_state_dict(s["opt"])
        with torch.no_grad():
            self.wE.copy_(s["wE"].to(d))
            self.wM.copy_(s["wM"].to(d))
            self.theta.copy_(s["theta"].to(d))
