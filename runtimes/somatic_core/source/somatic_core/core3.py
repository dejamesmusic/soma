"""somatic_core engine, third form: core2 with the counts keyed by (context, successor) PAIRS (Nov 8).

Why: at 6M bytes the slot store's GPU tensor (nodes x rungs x K+1 slots) was 5-10% occupied: median successors per
context is 1 at orders 2-3, p99 is 34 / 12. A pair pool holds counts only for pairs that exist:
  GPU   pc[k]  (NP_k + 1, rungs)   counts per pair (row NP_k is a zero dummy)
        st[k]  (P_k + 1, rungs)    node totals, sl[k] node clocks (lazy decay is per node, shared by its pairs)
  CPU   tok, pid (P_k, K_k) int32  each node's successor slots: token and pair id (-1 empty)
        pcnt, plast, pnode, pslot   per pair: all-time count, last chunk, owner (node, slot); a free stack
A new successor takes a free pair; when none is free it evicts the least-used of 8 sampled pairs (count decayed by
chunk age; pairs touched this chunk are never evicted), and the owner node's slot empties: that mass is lost to the
owner's back-off, exactly as a replaced slot's mass already was. With pools large enough that no pair is evicted, the
answers equal core2's (tests/equal3.py).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .core2 import Core2, Learner3, _rot
from .core import Matcher
from . import bounds, jit


class PairSlots:
    def __init__(self, P: int, K: int, NP: int, lg_chunk: float, seed: int):
        self.P, self.K, self.NP = int(P), int(K), int(NP)
        self.tok = np.full((self.P, self.K), -1, np.int32)
        self.pid = np.full((self.P, self.K), -1, np.int32)
        self.pcnt = np.zeros(self.NP, np.int32)
        self.plast = np.full(self.NP, -1, np.int64)
        self.pnode = np.full(self.NP, -1, np.int32)
        self.pslot = np.zeros(self.NP, np.int16)
        self.free = np.arange(self.NP - 1, -1, -1, dtype=np.int32)
        self.nfree = self.NP
        self.lg = float(lg_chunk)
        self.rng = np.random.default_rng(seed)
        self.overflows = self.trash = self.assignments = self.pair_evictions = 0

    def node_pids(self, n: int) -> np.ndarray:
        p = self.pid[n]
        return p[p >= 0]

    def rows(self, nodes: np.ndarray):
        """-> (tok, pid), each (len(nodes), K): the nodes' successor slots (-1 empty)."""
        return self.tok[nodes], self.pid[nodes]

    def reset(self, n: int):
        p = self.node_pids(n)
        if len(p):
            self.free[self.nfree:self.nfree + len(p)] = p
            self.nfree += len(p)
            self.pnode[p] = -1
        self.tok[n] = -1
        self.pid[n] = -1

    def _alloc(self, ch: int) -> int:
        if self.nfree:
            self.nfree -= 1
            return int(self.free[self.nfree])
        for _ in range(4):
            cand = self.rng.integers(0, self.NP, size=8)
            cand = cand[self.plast[cand] != ch]
            if len(cand):
                sc = self.pcnt[cand] * np.exp(self.lg * (ch - self.plast[cand]))
                p = int(cand[np.argmin(sc)])
                n2, s2 = int(self.pnode[p]), int(self.pslot[p])
                if n2 >= 0:
                    self.tok[n2, s2] = -1
                    self.pid[n2, s2] = -1
                self.pair_evictions += 1
                return p
        return -1

    def assign(self, n: int, y: int, ch: int):
        """-> (slot, pair, new): the successor y's slot and pair at node n; new pairs must have their counts zeroed."""
        row = self.tok[n]
        hit = np.nonzero(row == y)[0]
        if len(hit):
            j = int(hit[0])
            p = int(self.pid[n, j])
            self.pcnt[p] = min(self.pcnt[p] + 1, 2**31 - 1)
            self.plast[p] = ch
            return j, p, False
        emp = np.nonzero(row < 0)[0]
        if len(emp):
            j = int(emp[0])
            p = self._alloc(ch)
            if p < 0:
                self.trash += 1
                return self.K, -1, False
        else:
            self.overflows += 1
            ps = self.pid[n]
            ok = np.nonzero(self.plast[ps] != ch)[0]
            if not len(ok):
                self.trash += 1
                return self.K, -1, False
            j = int(ok[np.argmin(self.pcnt[ps[ok]])])
            p = int(ps[j])
        self.tok[n, j] = y
        self.pid[n, j] = p
        self.pnode[p] = n
        self.pslot[p] = j
        self.pcnt[p] = 1
        self.plast[p] = ch
        self.assignments += 1
        return j, p, True

    def state(self, copy: bool = True) -> dict:
        cp = (lambda a: a.copy()) if copy else (lambda a: a)               # copy=False: views, for a save that writes now
        return dict(tok=cp(self.tok), pid=cp(self.pid), pcnt=cp(self.pcnt), plast=cp(self.plast),
                    pnode=cp(self.pnode), pslot=cp(self.pslot), free=cp(self.free[:self.nfree]),
                    rng=self.rng.bit_generator.state, overflows=self.overflows, trash=self.trash,
                    assignments=self.assignments, pair_evictions=self.pair_evictions)

    def load(self, s: dict):
        for k in ("tok", "pid", "pcnt", "plast", "pnode", "pslot"):
            setattr(self, k, np.asarray(s[k]).astype(getattr(self, k).dtype).copy())
        fr = np.asarray(s["free"], np.int32)
        self.free[:len(fr)] = fr
        self.nfree = len(fr)
        self.rng.bit_generator.state = s["rng"]
        for k in ("overflows", "trash", "assignments", "pair_evictions"):
            setattr(self, k, int(s[k]))


class PagedPairSlots(PairSlots):
    """PairSlots with the same slots, stored in pages (Nov 15). At 1 GB the dense (P, K) tok/pid rows were ~3% used
    (median successors per context 1, a few hot contexts near K) and held 4 of the 7.5 GiB checkpoint. Here a node's
    logical slot j lives in page ptab[n, j // R] at offset j % R; pages come from a shared pool and return to it when the
    node resets. A node's pages are always a prefix (slots fill first-empty-first, pages are freed only on reset), so the
    pool needs at most NP / R + P pages if rows had no holes; holes from pair evictions add a little (measured on the
    1 GB checkpoint: below that bound at R = 16). Same slot indices, same answers as PairSlots while the pool has pages;
    if it runs dry a node that needs a new page takes the overflow path (counted in page_fallbacks)."""

    def __init__(self, P: int, K: int, NP: int, lg_chunk: float, seed: int, R: int = 16, pages: int = 0):
        self.P, self.K, self.NP = int(P), int(K), int(NP)
        self.R = int(min(R, K))
        assert self.K % self.R == 0, (K, R)
        self.NG = self.K // self.R
        self.NPAGE = int(pages) if pages else (self.NP + self.NP // 4) // self.R + self.P   # +25%: room for holes
        self.ptab = np.full((self.P, self.NG), -1, np.int32)
        self.ptok = np.full((self.NPAGE + 1, self.R), -1, np.int16)        # row NPAGE: the empty dummy page
        self.ppid = np.full((self.NPAGE + 1, self.R), -1, np.int32)
        self.pgfree = np.arange(self.NPAGE - 1, -1, -1, dtype=np.int32)
        self.npgfree = self.NPAGE
        self.pcnt = np.zeros(self.NP, np.int32)
        self.plast = np.full(self.NP, -1, np.int64)
        self.pnode = np.full(self.NP, -1, np.int32)
        self.pslot = np.zeros(self.NP, np.int16)
        self.free = np.arange(self.NP - 1, -1, -1, dtype=np.int32)
        self.nfree = self.NP
        self.lg = float(lg_chunk)
        self.rng = np.random.default_rng(seed)
        self.overflows = self.trash = self.assignments = self.pair_evictions = self.page_fallbacks = 0

    def _pg(self, nodes):
        g = self.ptab[nodes]
        return np.where(g >= 0, g, self.NPAGE)

    def rows(self, nodes: np.ndarray):
        g = self._pg(nodes)
        sh = g.shape[:-1] + (self.K,)
        return self.ptok[g].reshape(sh).astype(np.int32), self.ppid[g].reshape(sh)

    def node_pids(self, n: int) -> np.ndarray:
        g = self.ptab[n]
        p = self.ppid[g[g >= 0]].ravel()
        return p[p >= 0]

    def reset(self, n: int):
        p = self.node_pids(n)
        if len(p):
            self.free[self.nfree:self.nfree + len(p)] = p
            self.nfree += len(p)
            self.pnode[p] = -1
        g = self.ptab[n]
        g = g[g >= 0]
        if len(g):
            self.ptok[g] = -1
            self.ppid[g] = -1
            self.pgfree[self.npgfree:self.npgfree + len(g)] = g[::-1]
            self.npgfree += len(g)
        self.ptab[n] = -1

    def _put(self, n: int, j: int, y: int, p: int):
        g = self.ptab[n, j // self.R]
        self.ptok[g, j % self.R] = y
        self.ppid[g, j % self.R] = p

    def _alloc(self, ch: int) -> int:
        if self.nfree:
            self.nfree -= 1
            return int(self.free[self.nfree])
        for _ in range(4):
            cand = self.rng.integers(0, self.NP, size=8)
            cand = cand[self.plast[cand] != ch]
            if len(cand):
                sc = self.pcnt[cand] * np.exp(self.lg * (ch - self.plast[cand]))
                p = int(cand[np.argmin(sc)])
                n2, s2 = int(self.pnode[p]), int(self.pslot[p])
                if n2 >= 0:
                    self._put(n2, s2, -1, -1)
                self.pair_evictions += 1
                return p
        return -1

    def assign(self, n: int, y: int, ch: int):
        # only the node's own pages are searched (a prefix of its page table); the median node has one
        g = self.ptab[n]
        na = int(np.count_nonzero(g >= 0))
        ga = g[:na]
        tr = self.ptok[ga].ravel()
        hit = np.nonzero(tr == y)[0]
        if len(hit):
            j = int(hit[0])
            p = int(self.ppid[ga].ravel()[j])
            self.pcnt[p] = min(self.pcnt[p] + 1, 2**31 - 1)
            self.plast[p] = ch
            return j, p, False
        emp = np.nonzero(tr < 0)[0]
        if len(emp):
            j = int(emp[0])                                # a hole in an allocated page (first empty, as dense)
        elif na < self.NG and self.npgfree:
            j = na * self.R                                # the first slot of the next page
        else:
            j = -1
            if na < self.NG:
                self.page_fallbacks += 1
        if j >= 0:
            p = self._alloc(ch)
            if p < 0:
                self.trash += 1
                return self.K, -1, False
            q = j // self.R
            if self.ptab[n, q] < 0:
                self.npgfree -= 1
                self.ptab[n, q] = self.pgfree[self.npgfree]
        else:
            self.overflows += 1
            pr = self.ppid[ga].ravel()
            ok = np.nonzero((pr >= 0) & (self.plast[np.maximum(pr, 0)] != ch))[0]
            if not len(ok):
                self.trash += 1
                return self.K, -1, False
            j = int(ok[np.argmin(self.pcnt[pr[ok]])])
            p = int(pr[j])
        assert 0 <= y < 32768
        self._put(n, j, y, p)
        self.pnode[p] = n
        self.pslot[p] = j
        self.pcnt[p] = 1
        self.plast[p] = ch
        self.assignments += 1
        return j, p, True

    def state(self, copy: bool = True) -> dict:
        cp = (lambda a: a.copy()) if copy else (lambda a: a)
        return dict(paged=self.R, ptab=cp(self.ptab), ptok=cp(self.ptok), ppid=cp(self.ppid),
                    pgfree=cp(self.pgfree[:self.npgfree]), pcnt=cp(self.pcnt), plast=cp(self.plast),
                    pnode=cp(self.pnode), pslot=cp(self.pslot), free=cp(self.free[:self.nfree]),
                    rng=self.rng.bit_generator.state, overflows=self.overflows, trash=self.trash,
                    assignments=self.assignments, pair_evictions=self.pair_evictions, page_fallbacks=self.page_fallbacks)

    def load(self, s: dict):
        for k in ("pcnt", "plast", "pnode", "pslot"):
            setattr(self, k, np.asarray(s[k]).astype(getattr(self, k).dtype).copy())
        fr = np.asarray(s["free"], np.int32)
        self.free[:len(fr)] = fr
        self.nfree = len(fr)
        self.rng.bit_generator.state = s["rng"]
        for k in ("overflows", "trash", "assignments", "pair_evictions"):
            setattr(self, k, int(s[k]))
        self.page_fallbacks = int(s.get("page_fallbacks", 0))
        if s.get("paged"):
            assert int(s["paged"]) == self.R and np.asarray(s["ptok"]).shape[0] == self.NPAGE + 1
            self.ptab = np.asarray(s["ptab"], np.int32).copy()
            self.ptok = np.asarray(s["ptok"], np.int16).copy()
            self.ppid = np.asarray(s["ppid"], np.int32).copy()
            gf = np.asarray(s["pgfree"], np.int32)
            self.pgfree[:len(gf)] = gf
            self.npgfree = len(gf)
            return
        # a dense PairSlots state: lay its rows into pages (each node's pages a prefix up to its last used slot)
        tok, pid = s["tok"], s["pid"]
        self.ptab[:] = -1; self.ptok[:] = -1; self.ppid[:] = -1
        nxt, B = 0, 1 << 14
        for a in range(0, self.P, B):
            t = np.asarray(tok[a:a + B]).reshape(-1, self.NG, self.R)
            u = np.asarray(pid[a:a + B]).reshape(-1, self.NG, self.R)
            used = (t >= 0).any(2)
            need = np.where(used.any(1), self.NG - np.argmax(used[:, ::-1], 1), 0)
            m = np.arange(self.NG)[None, :] < need[:, None]
            cnt = int(m.sum())
            assert nxt + cnt <= self.NPAGE, "page pool too small for this dense state"
            ids = np.arange(nxt, nxt + cnt, dtype=np.int32)
            blk = self.ptab[a:a + B]
            blk[m] = ids
            self.ptok[ids] = t[m].astype(np.int16)
            self.ppid[ids] = u[m]
            nxt += cnt
        self.pgfree = np.arange(self.NPAGE - 1, nxt - 1, -1, dtype=np.int32)
        self.pgfree = np.concatenate([self.pgfree, np.zeros(nxt, np.int32)])
        self.npgfree = self.NPAGE - nxt


class _PairRows:
    """lets core2's read gather a node's slot counts (self.sc[k][idc]) from the pair pool, via the tick's pid map."""

    def __init__(self, owner, k):
        self.o, self.k = owner, k

    def __getitem__(self, idc):
        return self.o.pc[self.k][self.o._pidmap[self.k]].permute(0, 2, 1)       # (C, NB, K+1)


class Core3(Core2):
    _dense_slots = False                                  # never allocate core.py's dense slot tensors (startup spike)

    def __init__(self, cfg: dict, dev, blen: np.ndarray):
        c = dict(cfg)
        super().__init__(c, dev, blen)
        c = self.c
        NPd = {1: int(c.get("np1", 524288)), 2: int(c.get("np2", 2097152)), 3: int(c.get("np3", 4194304)),
               4: int(c.get("np4", 4194304))}
        lgc = math.log(float(c["ctx_decay"])) * int(c["chunk"])
        for k in self.skip_offs:                                # skip orders: pair pools sized like skip_size's order
            NPd[k] = NPd[self.skip_like]
        self.NPk = {k: NPd[k] for k in self.orders}
        R = int(c.get("slot_page", 0))                         # Nov 15: > 0 stores slots in pages of R (PagedPairSlots)
        self.slots = {k: (PagedPairSlots(self.PK[k], self.KK[k], self.NPk[k], lgc, 100 + k, R) if R > 0
                          else PairSlots(self.PK[k], self.KK[k], self.NPk[k], lgc, 100 + k)) for k in self.orders}
        self.__dict__.pop("sc", None)
        if self.dev.type == "mps":
            torch.mps.empty_cache()
        self.pc = {k: torch.zeros(self.NPk[k] + 1, self.NB, device=self.dev) for k in self.orders}
        self.sc = {k: _PairRows(self, k) for k in self.orders}
        self._pidmap = {}
        # nothing grows (bounds.py): AdamW + clipping + radial compression on the learner, clamped mix and gate,
        # halved pair counts, rebased clocks
        self.bounded = bool(int(c.get("bounded", 1)))
        self.rebase_at = bounds.REBASE
        self.rebase_keep = (1 << 20, 1 << 16)                                       # stamps kept positive after a rebase
        if self.bounded:
            with torch.no_grad():                         # start every row inside its radial ceiling (torch's default
                self.net.emb.weight.normal_(0.0, 0.1)    # embedding rows are ~8x the ceiling and would start frozen)
                self.net.P.normal_(0.0, 0.1)
            # bound_mode (Nov 9 A/B): 'radial' (1.0.x: AdamW wd + clip + radial compression), 'wd' (AdamW with decoupled
            # weight decay on matrices + clip, no radial), 'sphere' (GM's sphere SGD: fixed radius per matrix, no moments)
            self.bound_mode = str(c.get("bound_mode", "radial"))
            if self.bound_mode == "sphere":
                with torch.no_grad():                     # a zero matrix has no direction: start the block outputs at
                    for b in self.net.blocks:             # torch's default init (final_norm keeps the scale harmless)
                        b.down.reset_parameters()
                lr_base = float(c.get("lr_base", 3e-3))
                self.opt = bounds.SphereSGD(self.net.parameters(), self.io2.lr() if self.io2 is not None else lr_base,
                                            float(c.get("sphere_rate", 0.7)) / lr_base, int(c["hidden"]),
                                            max_angle=float(c.get("max_angle", 0.05)))
            elif self.bound_mode == "wd":
                ps = list(self.net.parameters())
                self._vecs = [p for p in ps if p.ndim < 2]             # no decay on vectors: a hard bound instead
                self.opt = torch.optim.AdamW([dict(params=[p for p in ps if p.ndim >= 2], weight_decay=float(c.get("wd", 1e-4))),
                                              dict(params=[p for p in ps if p.ndim < 2], weight_decay=0.0)],
                                             lr=float(c["lr"]), fused=self.dev.type in ("mps", "cuda"))
            else:
                self.opt = torch.optim.AdamW(self.net.parameters(), lr=float(c["lr"]), weight_decay=float(c.get("wd", 1e-4)),
                                             fused=self.dev.type in ("mps", "cuda"))
            if self.io2 is not None:
                for g in self.opt.param_groups:
                    g["lr"] = self.io2.lr()

    def store_floats(self) -> int:
        return sum(self.pc[k].numel() + self.st[k].numel() for k in self.orders) + self.r0.numel()

    def _maps2(self, ids):
        out = {}
        if jit.HAVE and all(type(self.slots[k]) is PagedPairSlots for k in self.orders):
            for k in self.orders:                           # compiled, int32 across, widened on the device (Nov 18)
                S = self.slots[k]
                C = ids.shape[1]
                M = np.empty((C, S.K + 1), np.int32)
                Pm = np.empty((C, S.K + 1), np.int32)
                jit.maps_tick(S.ptab, S.ptok, S.ppid, S.R, S.K, self.NPk[k], np.ascontiguousarray(ids[k - 1], np.int64), M, Pm)
                Mt = self._dv(M).long()
                out[f"v{k}"] = Mt >= 0
                out[k] = Mt.clamp(min=0)
                self._pidmap[k] = self._dv(Pm).long()
            return out
        for k in self.orders:
            S = self.slots[k]
            n = ids[k - 1]
            M = np.full((len(n), S.K + 1), -1, np.int64)
            Pm = np.full((len(n), S.K + 1), self.NPk[k], np.int64)
            ok = n >= 0
            tr, pr = S.rows(n[ok])
            M[ok, :S.K] = tr
            pp = pr.astype(np.int64)
            Pm[ok, :S.K] = np.where(pp >= 0, pp, self.NPk[k])
            out[f"v{k}"] = M >= 0
            out[k] = self._dv(np.maximum(M, 0))
            self._pidmap[k] = self._dv(Pm)
        return out

    def _node_pairs(self, k, nodes: np.ndarray):
        """flat pair ids of the given nodes and, for each, the index of its node in `nodes`."""
        S = self.slots[k]
        if jit.HAVE and type(S) is PagedPairSlots:
            return jit.node_pairs(S.ptab, S.ppid, S.R, np.ascontiguousarray(nodes, np.int64))
        P = S.rows(nodes)[1]
        r, cidx = np.nonzero(P >= 0)
        return P[r, cidx].astype(np.int64), r

    def _assign(self, ids, rec, ys):
        if jit.HAVE and all(type(self.slots[k]) is PagedPairSlots for k in self.orders):
            return self._assign_jit(ids, rec, ys)
        C = len(ys)
        slot, pair, new = {}, {}, {}
        for k in self.orders:
            S = self.slots[k]
            sk = np.full(C, S.K, np.int64)
            pk = np.full(C, self.NPk[k], np.int64)
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
                j, p, fresh = S.assign(int(n), int(ys[i]), self.chunks)
                sk[i] = j
                if p >= 0:
                    pk[i] = p
                if fresh:
                    nk.append(p)
            slot[k], pair[k], new[k] = sk, pk, nk
        return slot, pair, new

    def _assign_jit(self, ids, rec, ys):
        """_assign on the compiled path (jit.assign_tick): the same slots, pairs and evictions, order by order"""
        C = len(ys)
        ys = np.ascontiguousarray(ys, np.int64)
        ids = np.ascontiguousarray(ids, np.int64)
        slot, pair, new = {}, {}, {}
        nw = np.empty(C, np.int64)
        for k in self.orders:
            S = self.slots[k]
            sk = np.full(C, S.K, np.int64)
            pk = np.full(C, self.NPk[k], np.int64)
            mt = np.array([S.nfree, S.overflows, S.trash, S.assignments, S.pair_evictions, S.page_fallbacks, S.npgfree],
                          np.int64)
            rng = jit.pcg_pack(S.rng.bit_generator.state)
            rng0 = rng.copy()
            ev = rec[k]
            rp = np.array([e[0] for e in ev], np.int64)
            rn = np.array([e[1] for e in ev], np.int64)
            nn = jit.assign_tick(S.ptab, S.ptok, S.ppid, S.pgfree, S.pcnt, S.plast, S.pnode, S.pslot, S.free, mt, S.NG, S.R,
                                 S.K, S.NP, S.lg, rng, ids[k - 1], ys, rp, rn, len(ev), self.chunks, sk, pk, nw)
            S.nfree, S.overflows, S.trash, S.assignments, S.pair_evictions, S.page_fallbacks, S.npgfree = \
                (int(v) for v in mt)
            if not np.array_equal(rng, rng0):
                S.rng.bit_generator.state = jit.pcg_unpack(rng)
            slot[k], pair[k], new[k] = sk, pk, nw[:nn].tolist()
        return slot, pair, new

    @torch.no_grad()
    def next_distribution(self, m: Matcher) -> torch.Tensor:
        """the next token's distribution in context m (chat, dreams). The attention learner reads m's own last tokens,
        exactly as a tick would for its first position (1.2.0: before this it read whatever window the last tick left)."""
        if not isinstance(self.net, Learner3):
            return super().next_distribution(m)
        # 1.2.1: a training tick's geometry, with a KV cache. A window is anchored on the last ctx - chunk + 1 tokens (as a
        # tick's first prediction sees them) and grows by one position per new token until it is full, then re-anchors.
        # The cache lives on the context (a clone starts without one) and is dropped when the weights move.
        ts, ctx = m.t, self.net.ctx
        ver = (id(self.net), self.chunks)
        g = getattr(m, "_gen", None)
        # att_cache=2 (slide): a tick's every position reads exactly ctx positions through the cached past, so a
        # generation window is the last ctx tokens and then slides (forward_kv trims), as training does. (Nov 18 fix:
        # before this it anchored ctx - chunk + 1 positions, which is ONE token when chunk == ctx, and sampled from
        # the last token alone.) Other modes keep the 1.2.1 geometry.
        slide = int(getattr(self.net, "slide", 0) or 0)
        if g is not None and g["ver"] == ver and g.get("p_ts") == ts and ts == g["start"] + g["n"]:
            return g["p"]                            # this position is already read (a sample, then read() of its token)
        room = (g["n"] + 1 < self.net.rcos.shape[0]) if (slide and g is not None and self.net.rope) else (g is not None and g["n"] < ctx)
        if not (g is not None and g["ver"] == ver and ts == g["start"] + g["n"] + 1 and room):
            kv0 = getattr(self, "_kv", None)
            if slide and self.net.rope and kv0 is not None and self._kv_end == ts:
                # the training window itself: its positions were computed with their store features at their own
                # ticks (a window rebuilt from bare tokens is not what this learner was trained to read). Keys are
                # cached un-rotated; generation's cache holds them rotated, at positions 0 .. Lk-1.
                Lk = int(kv0[0][0].shape[2])
                cs, sn = self.net.rcos[:Lk], self.net.rsin[:Lk]
                g = dict(ver=ver, start=ts - 1 - Lk, n=Lk, kv=[[_rot(k, cs.to(k.dtype), sn.to(k.dtype)), v] for k, v in kv0])
                seq = m.R[[(ts - 1) % m.ring]]
            else:
                L = max(1, min(ts, ctx)) if slide else max(1, min(ts, ctx - int(self.c["chunk"]) + 1))
                seq = m.R[np.arange(ts - L, ts) % m.ring] if ts > 0 else np.zeros(1, np.int64)
                g = dict(ver=ver, start=ts - L if ts > 0 else 0, n=0, kv=[])
                self.gen_rebuilds = getattr(self, "gen_rebuilds", 0) + 1
            m._gen = g
        else:
            seq = m.R[[(ts - 1) % m.ring]]
        self.net.seq = self._dv(np.asarray(seq, np.int64))
        self.net.past_use = None
        self.net.gen = g
        try:
            p = super().next_distribution(m)
            g["p"], g["p_ts"] = p, ts
            return p
        finally:
            self.net.gen = None

    @torch.no_grad()
    def read(self, m: Matcher, ys):
        """tokens into a context the way generation will read them (Nov 18): with the attention learner's sliding
        cache, each one passes through the learner with its store features, so the window a reply reads holds the
        same kind of positions training made. Other learners: observe."""
        if not (isinstance(self.net, Learner3) and int(getattr(self.net, "slide", 0) or 0)):
            return self.observe(m, ys)
        for y in np.asarray(ys, np.int64).reshape(-1):
            self.next_distribution(m)
            self.observe(m, [int(y)])

    def count(self, m: Matcher, ys):
        """the counting pass: the store (and its exact mix) advances exactly as step() would advance it, with no
        learner read, gradient, gate or cache. The store is a function of the stream alone (checked Nov 18), so this
        is how a far-apart lane is seeded with the stream's store at its offset."""
        self._count_only = True
        try:
            return self.step(m, ys)
        finally:
            self._count_only = False

    def _count_pure(self, ids, rec, ys, t0, C):
        idt = self._idt(ids)
        with torch.no_grad():                             # recycled nodes (and their pairs) read as empty this tick
            for k in self.orders:
                if rec[k]:
                    nodes = np.array([n for _, n in rec[k]], np.int64)
                    nd = self._dv(nodes)
                    self.st[k].index_fill_(0, nd, 0.0)
                    self.sl[k].index_fill_(0, nd, t0)
                    pp, _ = self._node_pairs(k, nodes)
                    if len(pp):
                        self.pc[k].index_fill_(0, self._dv(pp), 0.0)
        slot, pair, new = self._assign(ids, rec, ys)
        self._t('cpu contexts')
        self._store_update(ids, idt, pair, new, self._dv(ys), t0, C)
        self._t('store update')
        self.clock += C
        self.chunks += 1
        self.n0_all += C
        if self.n0_all > float(self.c["alltime_cap"]):
            self._halve()
        if self.clock >= self.rebase_at:
            self._rebase()
        if self.chunks % 1024 == 0:
            bounds.cap_steps_(self.mopt)
        return None

    def _store_update(self, ids, idt, pair, new, y, t0, C):
        """the counts take this tick's tokens: decay to its end, add each position's successor (data only)"""
        d = self.dev
        with torch.no_grad():
            keep = 1.0 - self.ab
            te = t0 + C
            wj = keep[None, :] ** (C - torch.arange(C, device=d).float())[:, None]
            self.r0 *= (keep ** float(te - self.r0_t))[:, None]
            self.n0 *= keep ** float(te - self.r0_t)
            self.r0_t = te
            NB = self.NB
            ib = self.ar[None].expand(C, NB)
            self.r0.index_put_((ib, y[:, None].expand(C, NB)), wj, accumulate=True)
            self.n0 += wj.sum(0)
            for k in self.orders:
                if new[k]:
                    self.pc[k].index_fill_(0, self._dv(np.asarray(new[k], np.int64)), 0.0)
                idc = idt[k]
                Un = np.unique(ids[k - 1])
                Un = Un[Un >= 0]
                U = self._dv(Un)
                f = keep[None] ** (te - self.sl[k][U]).float()[:, None]                  # (|U|, NB)
                pp, owner = self._node_pairs(k, Un)
                if len(pp):
                    self.pc[k][self._dv(pp)] *= f[self._dv(owner)]
                self.st[k][U] *= f
                self.sl[k].index_fill_(0, U, te)
                pj = self._dv(pair[k])[:, None].expand(C, NB)
                self.pc[k].index_put_((pj, ib), wj, accumulate=True)
                ix = idc[:, None].expand(C, NB)
                self.st[k].index_put_((ix, ib), wj, accumulate=True)
                self.pc[k][self.NPk[k]].zero_()
                self.st[k][self.PK[k]].zero_()

    def step(self, m: Matcher, ys, learn: bool = True):
        cnt_ = getattr(self, "_count_only", False)
        ms_ = int(self.c.get("mix_shared", 0))           # 1: mixer on the learner's update cadence; 2: per tick, as before
        ms = ms_ == 1                                      # either way the mixer is not the store: count() leaves it alone
        ys = np.asarray(ys, np.int64)
        C = len(ys)
        d = self.dev
        t0 = self.clock
        self._t('_start')
        ids, rec, mp, mb, win = self._contexts(m, ys, learn)
        if cnt_ and ms_:                                     # the store is counts only: no read at all
            return self._count_pure(ids, rec, ys, t0, C)
        if isinstance(self.net, Learner3) and not cnt_:      # the attention learner reads the stream's recent tokens
            assert not isinstance(m, (list, tuple)), "the attention learner takes one stream"
            ts = m.t - C                                   # the clock of ys[0] in this context
            L = max(1, min(ts, self.net.ctx - C + 1))
            prev = m.R[np.arange(ts - L, ts) % m.ring] if ts > 0 else np.zeros(1, np.int64)
            if self.net.cache_on:                          # the cached past: only this tick's inputs are computed
                assert str(self.c.get("learn_mode", "skip")) in ("skip", "accumulate") and not int(self.c.get("rehearse", 0))
                self.net.seq = self._dv(np.concatenate([prev[-1:], ys[:-1]]).astype(np.int64))
                ok = getattr(self, "_kv", None) is not None and self._kv_end == ts
                if not ok:                                 # first tick, a resume without one, or a gap: rebuild
                    Lc = max(0, min(ts - 1, self.net.ctx - (1 if self.net.slide else C)))
                    self._kv = self.net.prefill(self._dv(m.R[np.arange(ts - 1 - Lc, ts - 1) % m.ring].astype(np.int64))) \
                        if Lc > 0 else None
                    self.kv_rebuilds = getattr(self, "kv_rebuilds", 0) + 1
                self.net.cache_in = self._kv
            else:
                self.net.seq = self._dv(np.concatenate([prev, ys[:-1]]).astype(np.int64))
            if self.net.past_on:                           # this stream's own earlier reads, if they end where we start
                ok = learn and getattr(self, "_past", None) is not None and self._past_end == ts
                self.net.past_use = self._past if ok else None
        idt = self._idt(ids)
        gid = self._gids(ids, rec, C)
        self._t('cpu contexts')
        if learn:
            with torch.no_grad():                         # recycled nodes (and their pairs) read as empty this tick
                for k in self.orders:
                    if rec[k]:
                        nodes = np.array([n for _, n in rec[k]], np.int64)
                        nd = self._dv(nodes)
                        self.st[k].index_fill_(0, nd, 0.0)    # index_fill_, not x[i] = 0.0: see _dv
                        self.sl[k].index_fill_(0, nd, t0)
                        pp, _ = self._node_pairs(k, nodes)
                        if len(pp):
                            self.pc[k].index_fill_(0, self._dv(pp), 0.0)
        maps = self._maps2(ids)
        y = self._dv(ys)
        self._t('maps + recycles')
        # learner decimation. The learner predicts every token either way; what changes is what it learns from.
        #   skip        its optimiser steps on one tick in learn_stride; the other ticks give it no gradient
        #   accumulate  every tick's gradient is kept, averaged over learn_stride ticks, one step
        #   select      by prediction (rma / v14): every s-th token (s = learn_stride) gives a gradient, one step per io2
        #               quantum of those predictions; io2 hears the sampled residuals with their gaps
        #   io2         select, with s chosen by io2's decimation band (phi^band, dithered so the density is exact)
        mode = str(self.c.get("learn_mode", "skip"))
        sel_ = mode in ("select", "io2")
        if learn and not cnt_:
            self._io2_flush()                             # last tick's io2 host work: waits only on that tick's tail
        if sel_:
            s_, acc_ = (self._sel_stride(mode) if learn else 1), False
            Q = float(self.io2.quantum) if self.io2 is not None else float(C)
            rows_np = np.arange((-t0) % s_, C, s_) if s_ > 1 else None
            nsel = C if rows_np is None else len(rows_np)
            lgrad = learn and nsel > 0
            self._sel_mass = getattr(self, "_sel_mass", 0.0) + (nsel if learn else 0)
            upd = learn and self._sel_mass + 1e-9 >= Q
            rows = self._dv(rows_np) if rows_np is not None and nsel > 0 else None
        else:
            s_ = max(1, int(self.c.get("learn_stride", 1)))
            acc_ = mode == "accumulate"
            upd = learn and self.chunks % s_ == s_ - 1
            lgrad = learn and (s_ == 1 or upd or acc_)
            rows, rows_np, nsel = None, None, C
        assert not (ms and sel_), "mix_shared steps the mixer on the learner's tick cadence (skip/accumulate)"
        if cnt_:
            upd = lgrad = sel_ = acc_ = False
        if getattr(self, "_ext", False):                  # a lane (lanes.py): gradients only; the shared update is
            upd = False                                    # applied once per round by apply_update
        self.net.keep_h = lgrad and self.io2 is not None             # io2's residual comes from the backward pass
        self._loo = self._loo_delta(t0) if (lgrad and getattr(self, 'loo_ages', None)) else None
        lpE, lpL, key, deep = self._read2(idt, gid, maps, ys, mp, mb, win, t0, grad=lgrad, rows=rows)
        self.net.keep_h = False
        lyE = lpE.gather(1, y[:, None])[:, 0]
        lyL = lpL.gather(1, y[:, None])[:, 0]
        with torch.no_grad():
            sg = torch.sigmoid(self.theta[key])
            a_, b_ = lyE.detach().exp(), lyL.detach().exp()
            pf = sg * a_ + (1 - sg) * b_
            pfull = sg[:, None] * lpE.detach().exp() + (1 - sg)[:, None] * lpL.detach().exp()
            self.acc[:3] += torch.stack([-lyE.detach().sum(), -lyL.detach().sum(), -torch.log(pf).sum()]) / math.log(2)
            self.acc[4] += (pfull.argmax(-1) == y).float().sum()
            self.acc[5] += C
        self.last = dict(tokens=ys, bits=-torch.log2(pf), exact=-lyE.detach() / math.log(2), next=pfull[-1].topk(5),
                         argmax=pfull.argmax(-1),                     # 1.5.0: each position's most likely token (reading turns)
                         learner=-lyL.detach() / math.log(2), deep=deep)          # deep: the store's deepest known order, per token
        if getattr(self, 'probe', None) is not None and not cnt_:   # before any learning from this tick
            self.probe(self, y, sg, lpE.detach(), lpL.detach())
        self._t('scoring')
        if not learn:
            return -torch.log2(pf)
        if isinstance(self.net, Learner3) and self.net.cache_on and self.net.cache_out is not None and not cnt_:
            keep = self.net.ctx - (1 if self.net.slide else C)   # bounded: ctx positions in all (slide: per query)
            ci, co = self.net.cache_in, self.net.cache_out
            kv = co if not ci else [(torch.cat([a[0], b[0]], 2), torch.cat([a[1], b[1]], 2)) for a, b in zip(ci, co)]
            self._kv = [(k[:, :, max(0, k.shape[2] - keep):].contiguous(), v[:, :, max(0, v.shape[2] - keep):].contiguous())
                        for k, v in kv] \
                if keep > 0 else None
            self._kv_end = m.t
            self.net.cache_in = self.net.cache_out = None
        if isinstance(self.net, Learner3) and self.net.past_on and self.net.feat_now is not None and not cnt_:
            keep = self.net.ctx - C                        # bounded: the most earlier positions a window holds
            pp = self.net.feat_now if self.net.past_use is None else torch.cat([self.net.past_use, self.net.feat_now], 0)
            self._past, self._past_end = pp[pp.shape[0] - keep:].clone() if keep > 0 else None, m.t
        lossE = -lyE.mean()
        if lgrad:
            if sel_:
                ysel = y if rows is None else y[rows]
                lsel = lyL if rows is None else self._sel_lp.gather(1, ysel[:, None])[:, 0]
                (-lsel.sum() / Q).backward()                  # mean over the quantum of selected predictions
                self._sel_lp = None
            else:
                if not acc_:
                    self.opt.zero_grad()
                lyT = lyL if self._lpL_train is None else self._lpL_train.gather(1, y[:, None])[:, 0]
                (-lyT.mean() / (s_ if acc_ else 1)).backward()
        gE = torch.autograd.grad(lossE, [self.wE, self.wM])
        slot, pair, new = self._assign(ids, rec, ys)          # CPU, while the GPU runs the read and backward
        self._t('backward')
        with torch.no_grad():
            if not cnt_:
                self.theta.index_put_((key,), float(self.c["gate_lr"]) * sg * (1 - sg) * (a_ - b_) / pf, accumulate=True)
        wu = int(self.c.get("warmup", 0))                  # dev: the learner's rate ramps over its first `warmup` steps
        if upd and self.c.get("rate_ctl", "fixed") == "agree":
            if not (wu and self.chunks < wu):
                self._agree()
            base = float(getattr(self, "ctl_lr", self.c["lr"]))
            f = (self.chunks + 1) / wu if (wu and self.chunks < wu) else 1.0
            for g in self.opt.param_groups:
                g["lr"] = base * f
        elif upd and self.c.get("rate_ctl", "fixed") == "hyper":
            if not (wu and self.chunks < wu):
                self._hyper_pre()
            base = float(getattr(self, "ctl_lr", self.c["lr"]))
            f = (self.chunks + 1) / wu if (wu and self.chunks < wu) else 1.0
            for g in self.opt.param_groups:
                g["lr"] = base * f
            if not (wu and self.chunks < wu):
                self._hg_snap()
        elif upd and wu and self.chunks < wu:
            base = self.io2.lr() if self.io2 is not None else float(self.c["lr"])
            for g in self.opt.param_groups:
                g["lr"] = base * (self.chunks + 1) / wu
        elif upd and getattr(self, "_rewarm", None) is not None:
            r0, rw = self._rewarm
            base = self.io2.lr() if self.io2 is not None else float(self.c["lr"])
            f = min(1.0, (self.chunks - r0 + 1) / rw)
            for g in self.opt.param_groups:
                g["lr"] = base * f
            if f >= 1.0:
                self._rewarm = None
        if not upd:
            pass
        elif self.bounded:
            params = [p for p in self.net.parameters()]
            if self.bound_mode == "sphere":
                self.opt.step()                                         # the angle cap is its clip; the sphere its bound
            elif self.bound_mode == "wd":
                bounds.clip_(params, 1.0)
                self.opt.step()
                with torch.no_grad():
                    torch._foreach_clamp_min_(self._vecs, -8.0); torch._foreach_clamp_max_(self._vecs, 8.0)
            else:
                before = bounds.norms_before(params)
                bounds.clip_(params, 1.0)                               # torch's clip: 57 ms/tick on MPS, this ~2
                self.opt.step()
                bounds.compress_(params, before)
        else:
            self.opt.step()
        if upd and getattr(self, "_hg_prev", None) is not None and self.c.get("rate_ctl", "fixed") == "hyper" \
                and not (wu and self.chunks < wu):
            self._hyper_post()
        if upd and int(self.c.get("rehearse", 0)) and not (acc_ or sel_):
            self._rehearse()
        if upd and (acc_ or sel_):
            self.opt.zero_grad()
        elif upd:
            self.opt.zero_grad(set_to_none=True)          # 1.2.1: free the grads now, not at the next backward (0.4 GB
                                                          # at 768 held through the next tick's forward); same maths
        if upd and sel_:
            self._sel_mass = 0.0
        if self.chunks % 1024 == 0:
            if not isinstance(self.opt, bounds.SphereSGD) and not cnt_:
                bounds.cap_steps_(self.opt)
            bounds.cap_steps_(self.mopt)
        self._t('optimiser + bounds')
        if self.io2 is not None and not cnt_:
            with torch.no_grad():
                Wo = self.net.out.weight
                h = self.net.h if lgrad else None
                scale = Q if sel_ else float(C) * (s_ if acc_ else 1)
                if h is not None and h.grad is not None:       # backward already formed (p - onehot) W_out / scale
                    res = h.grad * (-scale) / Wo.norm(dim=1).mean()
                else:
                    res = (Wo[y] - lpL.detach().exp() @ Wo) / Wo.norm(dim=1).mean()
                self.net.h = None
            # its host work (a float64 bank on the CPU) runs at the next tick's start, before that tick's optimiser
            # step: every learning rate is applied exactly where it was, without stalling this tick on the GPU
            if sel_ and rows is not None and h is not None:
                self._io2_pending = (res, -lyL[rows].detach().sum(), nsel, float(self.blen[ys[rows_np]].sum()), s_)
            else:
                self._io2_pending = (res, -lyL.detach().sum(), C, float(self.blen[ys].sum()), 1)
        self._t('io2')
        with torch.no_grad():
            gE = [torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) for g in gE]   # one bad tick must not reach Adam
            if ms:                                         # mix_shared: the mixer is the learner's, on its cadence;
                for w, g in zip((self.wE, self.wM), gE):   # the mean over the update's ticks, one Adam step
                    w.grad = g / s_ if w.grad is None else w.grad + g / s_
                if upd:
                    self.mopt.step()
                    self.wE.grad = self.wM.grad = None
            elif self.mix_opt == "adam":
                self.wE.grad, self.wM.grad = gE[0], gE[1]
                self.mopt.step()
            else:
                lr_e = float(self.c["mix_lr"]) * C
                self.wE -= lr_e * gE[0]
                self.wM -= lr_e * gE[1]
            if self.bounded:
                for v in (self.wE, self.wM, self.theta):
                    v.clamp_(-bounds.SCALAR_BOUND, bounds.SCALAR_BOUND)
            self._store_update(ids, idt, pair, new, y, t0, C)
        self._t('store update')
        self.clock += C
        self.chunks += 1
        self.n0_all += C
        if self.n0_all > float(self.c["alltime_cap"]):
            self._halve()
        if self.clock >= self.rebase_at:
            self._rebase()
        return -torch.log2(pf)

    # ---- lanes (lanes.py, Nov 18): one learner, many far-apart stores ----
    STORE_SCALARS = ("clock", "r0_t", "n0_all", "halvings", "chunks", "rebases")

    def store_state(self) -> dict:
        """the store alone (a function of the stream: checked Nov 18), as an independent CPU copy"""
        cpu = lambda t: t.detach().to("cpu").clone()
        s = {k: getattr(self, k, 0) for k in self.STORE_SCALARS}
        s.update(r0=cpu(self.r0), n0=cpu(self.n0),
                 pc={str(k): cpu(self.pc[k]) for k in self.orders}, st={str(k): cpu(self.st[k]) for k in self.orders},
                 sl={str(k): cpu(self.sl[k]) for k in self.orders},
                 slots={str(k): self.slots[k].state(copy=True) for k in self.orders},
                 pools={str(k): self.pools[k].state() for k in self.pools})
        return s

    def load_store(self, s: dict):
        d = self.dev
        self.clock, self.r0_t, self.n0_all = int(s["clock"]), int(s["r0_t"]), float(s["n0_all"])
        self.halvings, self.chunks, self.rebases = int(s["halvings"]), int(s["chunks"]), int(s.get("rebases", 0))
        self.r0.copy_(s["r0"].to(d))
        self.n0.copy_(s["n0"].to(d))
        for k in self.orders:
            self.pc[k].copy_(s["pc"][str(k)].to(d))
            self.st[k].copy_(s["st"][str(k)].to(d))
            self.sl[k].copy_(s["sl"][str(k)].to(d))
            self.slots[k].load(s["slots"][str(k)])
        for k in self.pools:
            self.pools[k].load(s["pools"][str(k)])
        self._kv, self._kv_end = None, -1                  # the window is rebuilt from the matcher's ring

    # ---- lanes in one learner pass (Nov 24, throughput; lanes_batch). step() in three parts for a lane (_ext,
    # learn_mode accumulate, mix_shared 1, fixed rate, no io2, att_cache): lane_pre reads the store and builds the
    # learner's inputs; Lanes runs every lane's learner as one (B, C) pass; lane_post_a scores and moves the gate (in
    # lane order, as step does); one backward for all lanes; lane_post_b the mixer's gradient and the store's update.
    # The store updates wait for the backward: the features' graph reads the lane's store (r0). Same arithmetic as step
    # for every lane; only the learner's backward sums in another order (tests/lanes_batch.py).
    def lane_pre(self, m: Matcher, ys, read: bool = True) -> dict:
        ys = np.asarray(ys, np.int64)
        C = len(ys)
        t0 = self.clock
        ids, rec, mp, mb, win = self._contexts(m, ys, True)
        net = self.net
        assert isinstance(net, Learner3) and net.cache_on and not net.past_on, "lanes_batch: the cached attention learner"
        ts = m.t - C
        L = max(1, min(ts, net.ctx - C + 1))
        prev = m.R[np.arange(ts - L, ts) % m.ring] if ts > 0 else np.zeros(1, np.int64)
        seq = self._dv(np.concatenate([prev[-1:], ys[:-1]]).astype(np.int64))
        ok = getattr(self, "_kv", None) is not None and self._kv_end == ts
        if not ok:
            Lc = max(0, min(ts - 1, net.ctx - (1 if net.slide else C)))
            self._kv = net.prefill(self._dv(m.R[np.arange(ts - 1 - Lc, ts - 1) % m.ring].astype(np.int64))) if Lc > 0 else None
            self.kv_rebuilds = getattr(self, "kv_rebuilds", 0) + 1
        cache = self._kv
        idt = self._idt(ids)
        gid = self._gids(ids, rec, C)
        with torch.no_grad():
            for k in self.orders:
                if rec[k]:
                    nodes = np.array([n for _, n in rec[k]], np.int64)
                    nd = self._dv(nodes)
                    self.st[k].index_fill_(0, nd, 0.0)
                    self.sl[k].index_fill_(0, nd, t0)
                    pp, _ = self._node_pairs(k, nodes)
                    if len(pp):
                        self.pc[k].index_fill_(0, self._dv(pp), 0.0)
        maps = self._maps2(ids)
        y = self._dv(ys)
        P = dict(m=m, ys=ys, C=C, t0=t0, ids=ids, rec=rec, idt=idt, y=y, seq=seq, cache=cache,
                 s_=max(1, int(self.c.get("learn_stride", 1))))
        if not read:                                       # lanes_read: every lane's read in one pass (lanes_read.py)
            P.update(gid=gid, maps=maps, mp=mp, mb=mb, win=win)
            return P
        lpE, _, key, deep, feat = self._read2(idt, gid, maps, ys, mp, mb, win, t0, grad=True, rows=None, defer=True)
        P.update(lpE=lpE, key=key, deep=deep, feat=feat)
        return P

    def lane_post_a(self, P: dict, lpL, cache_out, cache_done: bool = False):
        """scoring, the gate's step and the cached window -> this lane's learner loss (already / learn_stride)"""
        y, C, m, lpE, key = P["y"], P["C"], P["m"], P["lpE"], P["key"]
        lyE = lpE.gather(1, y[:, None])[:, 0]
        lyL = lpL.gather(1, y[:, None])[:, 0]
        with torch.no_grad():
            sg = torch.sigmoid(self.theta[key])
            a_, b_ = lyE.detach().exp(), lyL.detach().exp()
            pf = sg * a_ + (1 - sg) * b_
            pfull = sg[:, None] * lpE.detach().exp() + (1 - sg)[:, None] * lpL.detach().exp()
            self.acc[:3] += torch.stack([-lyE.detach().sum(), -lyL.detach().sum(), -torch.log(pf).sum()]) / math.log(2)
            self.acc[4] += (pfull.argmax(-1) == y).float().sum()
            self.acc[5] += C
            self.theta.index_put_((key,), float(self.c["gate_lr"]) * sg * (1 - sg) * (a_ - b_) / pf, accumulate=True)
        self.last = dict(tokens=P["ys"], bits=-torch.log2(pf), exact=-lyE.detach() / math.log(2), next=pfull[-1].topk(5),
                         learner=-lyL.detach() / math.log(2), deep=P["deep"])
        if not cache_done:                                 # (Lanes keeps the batched window itself when it can)
            keep = self.net.ctx - (1 if self.net.slide else C)
            ci, co = P["cache"], cache_out
            kv = co if not ci else [(torch.cat([a[0], b[0]], 2), torch.cat([a[1], b[1]], 2)) for a, b in zip(ci, co)]
            self._kv = [(k[:, :, max(0, k.shape[2] - keep):].contiguous(), v[:, :, max(0, v.shape[2] - keep):].contiguous())
                        for k, v in kv] if keep > 0 else None
            self._kv_end = m.t
        P["lossE"], P["pf"] = -lyE.mean(), pf
        P["feat"] = P["lpE"] = None
        return -lyL.mean() / P["s_"]

    def lane_post_b(self, P: dict, mixer: bool = True, store: bool = True):
        """after the shared backward: the mixer's gradient, the store's update, the clocks"""
        ids, rec, ys, idt, y, t0, C, s_ = (P[k] for k in ("ids", "rec", "ys", "idt", "y", "t0", "C", "s_"))
        gE = torch.autograd.grad(P["lossE"], [self.wE, self.wM]) if mixer else None   # (lanes_read: once a round)
        slot, pair, new = self._assign(ids, rec, ys)
        if self.chunks % 1024 == 0:
            if not isinstance(self.opt, bounds.SphereSGD):
                bounds.cap_steps_(self.opt)
            bounds.cap_steps_(self.mopt)
        with torch.no_grad():
            if gE is not None:
                gE = [torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) for g in gE]
                for w, g in zip((self.wE, self.wM), gE):
                    w.grad = g / s_ if w.grad is None else w.grad + g / s_
            if self.bounded:
                for v in (self.wE, self.wM, self.theta):
                    v.clamp_(-bounds.SCALAR_BOUND, bounds.SCALAR_BOUND)
            if store:
                self._store_update(ids, idt, pair, new, y, t0, C)
            else:                                          # lanes_store: every lane's update at once, then lane_clock
                P["pair"], P["new"] = pair, new
                return -torch.log2(P["pf"])
        self.lane_clock(P)
        return -torch.log2(P["pf"])

    def lane_clock(self, P: dict):
        """after the store's update: the store's clocks (its own: it ages with its own tokens)"""
        C = P["C"]
        self.clock += C
        self.chunks += 1
        self.n0_all += C
        if self.n0_all > float(self.c["alltime_cap"]):
            self._halve()
        if self.clock >= self.rebase_at:
            self._rebase()

    REWARM = 256                                          # ticks over which a slim checkpoint's rate re-warms

    def apply_update(self, k: int, lanes: int):
        """the shared learner's k-th update (0-based), after `lanes` ticks whose gradients were accumulated with weight
        1/lanes each (learn_mode accumulate, learn_stride = lanes, _ext set). The same arithmetic as step()'s own
        update at stride `lanes`: one lane reproduces a single stream exactly (tests/lanes.py)."""
        assert self.io2 is None and self.c.get("rate_ctl", "fixed") == "fixed" and not int(self.c.get("rehearse", 0))
        wu = int(self.c.get("warmup", 0))
        t = (k + 1) * lanes                                # the stream's ticks so far, as step() counts them
        if wu and t - 1 < wu:
            for g in self.opt.param_groups:
                g["lr"] = float(self.c["lr"]) * t / wu
        params = list(self.net.parameters())
        if not self.bounded or self.bound_mode == "sphere":
            self.opt.step()
        elif self.bound_mode == "wd":
            bounds.clip_(params, 1.0)
            self.opt.step()
            with torch.no_grad():
                torch._foreach_clamp_min_(self._vecs, -8.0); torch._foreach_clamp_max_(self._vecs, 8.0)
        else:
            before = bounds.norms_before(params)
            bounds.clip_(params, 1.0)
            self.opt.step()
            bounds.compress_(params, before)
        self.opt.zero_grad()
        if int(self.c.get("mix_shared", 0)) == 1:
            with torch.no_grad():
                self.mopt.step()
                self.wE.grad = self.wM.grad = None
                if self.bounded:
                    for v in (self.wE, self.wM, self.theta):
                        v.clamp_(-bounds.SCALAR_BOUND, bounds.SCALAR_BOUND)

    def _bounded_step(self):
        """one optimiser step under the model's bounding, exactly as the tick's own step."""
        params = [p for p in self.net.parameters()]
        if not self.bounded:
            self.opt.step()
        elif self.bound_mode == "sphere":
            self.opt.step()
        elif self.bound_mode == "wd":
            bounds.clip_(params, 1.0)
            self.opt.step()
            with torch.no_grad():
                torch._foreach_clamp_min_(self._vecs, -8.0); torch._foreach_clamp_max_(self._vecs, 8.0)
        else:
            before = bounds.norms_before(params)
            bounds.clip_(params, 1.0)
            self.opt.step()
            bounds.compress_(params, before)

    def _rehearse(self):
        """rehearsal (dev, Nov 10; James: "rehearsal sounds like the right way to level the playing field"). Extra
        learner steps per tick on reads taken BEFORE their tick's store update, so the store never hands the learner
        the answer it has since counted. The token projection P is applied fresh, so it learns too. span 1 repeats
        the current tick; span D picks each step's tick from the last D (a transient cache: it refills in D ticks
        after a resume, and is not part of the checkpoint). io2 hears none of it."""
        R, D = int(self.c["rehearse"]), int(self.c.get("rehearse_span", 1))
        q = self.__dict__.setdefault("_rq", [])
        q.append(self._pack); self._pack = None
        del q[:-D]
        rng = self.__dict__.setdefault("_rrng", np.random.default_rng(int(self.c.get("seed", 0)) + 7))
        keep, self.net.keep_h = self.net.keep_h, False
        for _ in range(R):
            pk = q[-1] if D == 1 else q[int(rng.integers(len(q)))]
            self.opt.zero_grad(set_to_none=True)
            lp = torch.log_softmax(self._learner(pk["wt"], self._feat(pk, None)), -1)
            (-lp.gather(1, pk["y"][:, None]).mean()).backward()
            self._bounded_step()
        self.opt.zero_grad(set_to_none=True)
        self.net.keep_h = keep

    def _loo_delta(self, t0):
        """(NB,) what this position's earlier instances weigh now: keep^age on each decaying rung, 0.5^halvings on the
        all-time rung, summed over the earlier passes (ages from run.py: the same text, whole passes back)."""
        keep = 1.0 - self.ab.detach().cpu().double().numpy()
        hc = np.asarray(getattr(self, "_halv_clocks", []), np.float64)
        out = np.zeros(self.NB)
        for D in self.loo_ages:
            dl = keep ** float(D)
            dl[-1] = 0.5 ** int((hc > t0 - D).sum())
            out += dl
        return self._dv(out.astype(np.float32))

    def _sel_stride(self, mode) -> int:
        """the prediction stride: learn_stride, or io2's (rma's rule: integers dithered so 1/stride averages 1/target)."""
        if mode != "io2" or self.io2 is None:
            return max(1, int(self.c.get("learn_stride", 1)))
        cap = int(self.io2.stride_cap)
        target = max(1.0, min(float(cap), ((1 + 5 ** 0.5) / 2) ** float(self.io2.band)))
        lower = max(1, int(math.floor(target)))
        upper = min(cap, lower + 1)
        if lower == upper:
            return lower
        self._sel_phase = getattr(self, "_sel_phase", 0.0) + (1.0 / lower - 1.0 / target) / (1.0 / lower - 1.0 / upper)
        if self._sel_phase + 1e-12 >= 1.0:
            self._sel_phase -= 1.0
            return upper
        return lower

    @torch.no_grad()
    def _agree(self):
        """rate_ctl=agree (Nov 13; James: plasticity should depend on what an error means given the model's state).
        c = cosine between where this tick's error would move the weights and where the last step moved them.
        c > 0: the last step fell short, the error is systematic (learnable): raise the rate. c < 0: it overshot, the
        error is noise at the model's present resolution: lower it. log lr += ctl_k * c. Nothing here counts age.
        ctl_signal 'adam' (Nov 13, v1): last step = Adam's m/(sqrt v + eps), error = the raw gradient. Failed (Nov 14):
          c stayed > 0 at every rate up to 1e-2, because weight decay's steady pull is balanced by a steady gradient,
          which reads as permanent agreement.
        ctl_signal 'total' (v2): both in the step's own coordinates and including the decay, the way AdamW moves:
          last step D = m_hat/(sqrt v_hat + eps) + wd*theta; this error G = g/(sqrt v_hat + eps) + wd*theta. At a
          settled point E[D] = 0 (decay and gradient cancel), so only the fluctuation is left to agree or not."""
        sig = str(self.c.get("ctl_signal", "adam"))
        if sig == "lag":
            return self._agree_lag()
        num = den_g = den_d = None
        for grp in self.opt.param_groups:
            ps = [p for p in grp["params"] if p.grad is not None and self.opt.state.get(p) and "exp_avg" in self.opt.state[p]]
            if not ps:
                continue
            st = [self.opt.state[p] for p in ps]; eps = grp["eps"]
            ms = [x["exp_avg"] for x in st]; vs = [x["exp_avg_sq"] for x in st]; gs = [p.grad for p in ps]
            if sig == "total":
                b1, b2 = grp["betas"]; n = float(st[0]["step"])
                bc1, bc2 = 1 - b1 ** n, 1 - b2 ** n
                den = torch._foreach_div(vs, bc2); den = torch._foreach_sqrt(den); torch._foreach_add_(den, eps)
                ds = torch._foreach_div(ms, den); torch._foreach_div_(ds, bc1)
                es = torch._foreach_div(gs, den)
                wd = float(grp.get("weight_decay", 0.0))
                if wd:
                    torch._foreach_add_(ds, ps, alpha=wd); torch._foreach_add_(es, ps, alpha=wd)
            else:
                den = torch._foreach_sqrt(vs); torch._foreach_add_(den, eps); ds = torch._foreach_div(ms, den); es = gs
            n_ = torch.stack([(e * d).sum() for e, d in zip(es, ds)]).sum()
            ge = torch.stack(torch._foreach_norm(es)).pow(2).sum(); gd = torch.stack(torch._foreach_norm(ds)).pow(2).sum()
            num = n_ if num is None else num + n_
            den_g = ge if den_g is None else den_g + ge
            den_d = gd if den_d is None else den_d + gd
        if num is None:
            return
        c = float(num / (den_g * den_d).sqrt().clamp_min(1e-30))
        lr = float(getattr(self, "ctl_lr", self.c["lr"])) * math.exp(float(self.c.get("ctl_k", 0.05)) * c)
        self.ctl_lr = min(float(self.c.get("ctl_max", 1e-2)), max(float(self.c.get("ctl_min", 1e-6)), lr))
        self.ctl_cos = 0.99 * float(getattr(self, "ctl_cos", 0.0)) + 0.01 * c

    @torch.no_grad()
    def _agree_lag(self):
        """ctl_signal 'lag' (Nov 14, v3). One-step agreement can't judge the learner: a document spans several ticks,
        so successive errors agree and a faster step does help the next tick (v1/v2 ran to the cap). Judge it on its
        own timescale instead: c = cos(-S, theta - theta_old), S = g/(sqrt v_hat + eps) + wd*theta (the AdamW step this
        error asks for, decay included), theta_old a snapshot L..2L steps back (L = ctl_lag, well past a document).
        Settled: theta - theta_old is mostly past noise that the gradient pulls back, c < 0, the rate falls. Drifting:
        the error keeps pointing where the weights have been heading, c > 0, it rises. Two snapshots (2x the learner)."""
        L = int(self.c.get("ctl_lag", 64))
        D = float(self.c.get("ctl_drift", 0.0))          # v4 (Nov 15): lag = D / lr, the learner's own timescale
        sub = max(1, int(self.c.get("ctl_sub", 1)))      # v4: judge on every sub-th coordinate (sketch; snapshots /sub)
        lr_now = float(getattr(self, "ctl_lr", self.c["lr"]))
        if D > 0:
            L = int(min(int(self.c.get("ctl_lag_max", 4096)), max(int(self.c.get("ctl_lag_min", 8)), round(D / lr_now))))
        ps_all, es_all = [], []
        for grp in self.opt.param_groups:
            ps = [p for p in grp["params"] if p.grad is not None and self.opt.state.get(p) and "exp_avg" in self.opt.state[p]]
            if not ps:
                continue
            st = [self.opt.state[p] for p in ps]; b1, b2 = grp["betas"]; n = float(st[0]["step"])
            vs = [x["exp_avg_sq"] for x in st]; gs = [p.grad for p in ps]; pv = ps
            if sub > 1:
                vs = [v.reshape(-1)[::sub] for v in vs]; gs = [g.reshape(-1)[::sub] for g in gs]; pv = [p.reshape(-1)[::sub] for p in ps]
            den = torch._foreach_div(vs, 1 - b2 ** n); den = torch._foreach_sqrt(den)
            torch._foreach_add_(den, grp["eps"]); es = torch._foreach_div(gs, den)
            wd = float(grp.get("weight_decay", 0.0))
            if wd:
                torch._foreach_add_(es, pv, alpha=wd)
            ps_all += pv; es_all += es
        if not ps_all:
            return
        k = getattr(self, "_lag_k", 0); self._lag_k = k + 1
        old = getattr(self, "_snap_old", None)
        if old is not None and len(old) == len(ps_all) and all(o.shape == q.shape for o, q in zip(old, ps_all)):
            disp = torch._foreach_sub(ps_all, old)
            num = -torch.stack([(e * d).sum() for e, d in zip(es_all, disp)]).sum()
            ne = torch.stack(torch._foreach_norm(es_all)).pow(2).sum(); nd = torch.stack(torch._foreach_norm(disp)).pow(2).sum()
            c = float(num / (ne * nd).sqrt().clamp_min(1e-30))
            lr = lr_now * math.exp(float(self.c.get("ctl_k", 0.05)) * c)
            self.ctl_lr = min(float(self.c.get("ctl_max", 1e-2)), max(float(self.c.get("ctl_min", 1e-6)), lr))
            self.ctl_cos = 0.99 * float(getattr(self, "ctl_cos", 0.0)) + 0.01 * c
        if D > 0:                                          # rotate when the newer snapshot is L steps old
            if k - getattr(self, "_snap_k", -10**9) >= L:
                self._snap_old = getattr(self, "_snap_new", None)
                self._snap_new = [p.detach().clone() for p in ps_all]; self._snap_k = k
        elif k % L == 0:                                   # rotate: the snapshot in use is always L..2L steps old
            self._snap_old = getattr(self, "_snap_new", None)
            self._snap_new = [p.detach().clone() for p in ps_all]
        self.ctl_L = L

    # ---- rate_ctl=hyper (Nov 15; spec lab/specs/hypergrad_controller.md, James's yes the same day) ----------------
    # The log-rate beta descends the prequential loss: d L_{t+1} / d beta = g_{t+1} . h_t, h_t = d theta_t / d beta, the
    # rate held constant over the past. g_{t+1} is this tick's learning gradient (taken at theta_t, before the step).
    # h's recursion through an AdamW step theta' = theta - a (u + wd theta), a = e^beta:
    #   h' = h (1 - a wd) + d_theta - a (du/dtheta) h          (d_theta = the step actually taken, = d(-a u)/d beta)
    # hg_tier 0: h' = d_theta (one step; Baydin et al. 2018). Expected to behave like agree v1 (judged on the next tick).
    # hg_tier 1: du/dtheta ~ diag curvature / (sqrt v_hat + eps), curvature ~ N v_hat (Fisher of the per-token mean loss;
    #            v_hat is the second moment of an N-token mean gradient), N = hg_curv (0 -> the tick, chunk). The trace
    #            forgets at the rate the surface pulls weights back; nothing here is a window we chose (test 3 varies N).
    # Meta-step: beta -= hg_mu * x / rms(x), x = g . h, rms over ~1/(1 - hg_rms) ticks; beta clamped to
    # [lr0 / hg_span, lr0 * hg_span] as a rail (clamp hits are counted and reported: a hit is a failure, not a feature).
    @torch.no_grad()
    def _hyper_pre(self):
        h = getattr(self, "_hg_h", None)
        ps = [p for p in self.net.parameters()]
        lr0 = float(self.c["lr"])
        if not hasattr(self, "ctl_lr"):
            self.ctl_lr = lr0
        if h is None or any(p.grad is None for p in ps):
            return
        x = float(torch.stack([(p.grad * q).sum() for p, q in zip(ps, h)]).sum())
        if not math.isfinite(x):
            return
        rd = float(self.c.get("hg_rms", 0.99))
        self._hg_n = getattr(self, "_hg_n", 0) + 1
        self._hg_r = rd * getattr(self, "_hg_r", 0.0) + (1 - rd) * x * x
        r = math.sqrt(self._hg_r / (1 - rd ** self._hg_n))
        z = x / r if r > 0 else 0.0
        span = float(self.c.get("hg_span", 30.0))
        b = math.log(self.ctl_lr) - float(self.c.get("hg_mu", 0.01)) * z
        lo, hi = math.log(lr0 / span), math.log(lr0 * span)
        if b <= lo or b >= hi:
            self.hg_clamps = getattr(self, "hg_clamps", 0) + 1
        self.ctl_lr = math.exp(min(hi, max(lo, b)))
        self.ctl_cos = 0.99 * float(getattr(self, "ctl_cos", 0.0)) + 0.01 * z     # logged: the normalised hypergradient

    @torch.no_grad()
    def _hg_snap(self):
        ps = [p.detach() for p in self.net.parameters()]
        if getattr(self, "_hg_prev", None) is None:
            self._hg_prev = [p.clone() for p in ps]
        else:
            torch._foreach_copy_(self._hg_prev, ps)

    @torch.no_grad()
    def _hyper_post(self):
        tier = int(self.c.get("hg_tier", 1))
        if not hasattr(self, "_hg_index"):
            self._hg_index = {id(p): i for i, p in enumerate(self.net.parameters())}
        allp = [p.detach() for p in self.net.parameters()]
        d = torch._foreach_sub(allp, self._hg_prev)                           # the step taken, clip and clamps included
        h = getattr(self, "_hg_h", None)
        if tier == 0 or h is None:
            self._hg_h = d
            return
        N = float(self.c.get("hg_curv", 0)) or float(self.c["chunk"])
        for grp in self.opt.param_groups:
            a, wd, eps = float(grp["lr"]), float(grp.get("weight_decay", 0.0)), float(grp["eps"])
            b2 = grp["betas"][1]
            for p in grp["params"]:
                i = self._hg_index[id(p)]; st = self.opt.state.get(p)
                keep = None
                if st and "exp_avg_sq" in st:
                    n = float(st["step"]); vh = st["exp_avg_sq"] / (1 - b2 ** n)
                    keep = 1.0 - a * wd - a * N * vh / (vh.sqrt() + eps)
                    keep.clamp_(0.0, 1.0)
                if keep is None:
                    h[i].mul_(1.0 - a * wd)
                else:
                    h[i].mul_(keep)
                h[i].add_(d[i])

    def _io2_flush(self):
        pend, self._io2_pending = getattr(self, "_io2_pending", None), None
        if pend is None or self.io2 is None:
            return
        res, nll, C, nb, gap = pend
        self.io2.advance(res, np.full(len(res), float(gap)))
        if self.io2.accumulate(float(nll), C, nb):
            for g in self.opt.param_groups:
                g["lr"] = self.io2.lr()

    @torch.no_grad()
    def _halve(self):
        if getattr(self, "loo_ages", None):                                   # leave-one-out reads (a lab option)
            hc = self.__dict__.setdefault("_halv_clocks", [])                 # bounded: nothing grows (1.4.0)
            hc.append(self.clock)
            del hc[:-1024]
        self.r0[-1] *= 0.5
        self.n0[-1] *= 0.5
        for k in self.orders:
            self.pc[k][:, -1] *= 0.5
            self.st[k][:, -1] *= 0.5
            self.slots[k].pcnt >>= 1                                                    # the eviction score's count
        self.n0_all *= 0.5
        self.halvings += 1

    @torch.no_grad()
    def _rebase(self):
        """shift every store time stamp back (only differences of times are used): nothing grows."""
        off = self.clock - self.rebase_keep[0]
        offc = self.chunks - self.rebase_keep[1]
        if off <= 0 or offc <= 0:
            return
        self.clock -= off
        self.r0_t -= off
        for k in self.orders:
            self.sl[k] -= off
            self.slots[k].plast -= offc
        for k, pool in self.pools.items():
            pool.last = pool.last - off
        self.chunks -= offc
        if getattr(self, "_halv_clocks", None):
            self._halv_clocks = [c - off for c in self._halv_clocks]
        self.rebases = getattr(self, "rebases", 0) + 1

    def stats(self) -> dict:
        out = super().stats()
        if hasattr(self, "ctl_cos"):
            out["ctl_cos"] = float(self.ctl_cos)
        if hasattr(self, "ctl_L"):
            out["ctl_L"] = int(self.ctl_L)
        if hasattr(self, "ctl_lr") and self.c.get("rate_ctl", "fixed") == "hyper":
            out["lr"] = float(self.ctl_lr); out["hg_clamps"] = int(getattr(self, "hg_clamps", 0))
        if self.bounded:
            out.update(bounds.pressure(list(self.net.parameters())))
            out["max_abs_theta"] = float(self.theta.abs().max())
        for k in self.orders:
            S = self.slots[k]
            out[f"o{k}_pairs"] = S.NP - S.nfree
            out[f"o{k}_pair_fill"] = (S.NP - S.nfree) / S.NP
            out[f"o{k}_pair_evictions"] = S.pair_evictions
            if isinstance(S, PagedPairSlots):
                out[f"o{k}_page_fill"] = (S.NPAGE - S.npgfree) / S.NPAGE
                out[f"o{k}_page_fallbacks"] = S.page_fallbacks
        return out

    # ---- state (bounded and self-contained: everything the learner carries) ----
    def state_dict(self, lean: bool = False) -> dict:
        """lean=True (1.2.1, for a save that writes at once): no copies. Device tensors stay where they are (torch.save
        moves one storage at a time) and the store's arrays are views. The default makes an independent CPU copy."""
        self._io2_flush()
        if self.dev.type == "mps":
            torch.mps.synchronize()
        cpu = (lambda t: t.detach()) if lean else (lambda t: t.detach().to("cpu").clone())
        return dict(
            engine="core3", config=dict(self.c), clock=self.clock, r0_t=self.r0_t, n0_all=self.n0_all,
            halvings=self.halvings, chunks=self.chunks, rebases=getattr(self, "rebases", 0),
            r0=cpu(self.r0), n0=cpu(self.n0),
            pc={str(k): cpu(self.pc[k]) for k in self.orders}, st={str(k): cpu(self.st[k]) for k in self.orders},
            sl={str(k): cpu(self.sl[k]) for k in self.orders},
            slots={str(k): self.slots[k].state(copy=not lean) for k in self.orders},
            pools={str(k): self.pools[k].state() for k in self.pools},
            net=self.net.state_dict(), opt=self.opt.state_dict(), mopt=self.mopt.state_dict(),
            wE=cpu(self.wE), wM=cpu(self.wM), theta=cpu(self.theta),
            io2=self.io2.state() if self.io2 is not None else None,
            sel=dict(mass=float(getattr(self, "_sel_mass", 0.0)), phase=float(getattr(self, "_sel_phase", 0.0))),
            past=None if getattr(self, "_past", None) is None else dict(f=cpu(self._past), end=int(self._past_end)),
            rewarm=None if getattr(self, "_rewarm", None) is None else list(self._rewarm),
            kv=None if getattr(self, "_kv", None) is None else dict(k=[cpu(k) for k, _ in self._kv],
                                                                  v=[cpu(v) for _, v in self._kv], end=int(self._kv_end)),
            ctl=dict(lr=float(self.ctl_lr), cos=float(getattr(self, "ctl_cos", 0.0)), k=int(getattr(self, "_lag_k", 0)),
                     snap_k=int(getattr(self, "_snap_k", -10**9)),
                     old=[t.cpu() for t in self._snap_old] if getattr(self, "_snap_old", None) else None,
                     new=[t.cpu() for t in self._snap_new] if getattr(self, "_snap_new", None) else None,
                     hg_h=[t.cpu() for t in self._hg_h] if getattr(self, "_hg_h", None) else None,
                     hg_r=float(getattr(self, "_hg_r", 0.0)), hg_n=int(getattr(self, "_hg_n", 0)),
                     hg_clamps=int(getattr(self, "hg_clamps", 0)))
                if hasattr(self, "ctl_lr") else None,
        )

    def load_state_dict(self, s: dict):
        assert s.get("engine") == "core3", "not a core3 state"
        d = self.dev
        self.clock, self.r0_t, self.n0_all = int(s["clock"]), int(s["r0_t"]), float(s["n0_all"])
        self.halvings, self.chunks, self.rebases = int(s["halvings"]), int(s["chunks"]), int(s.get("rebases", 0))
        self.r0.copy_(s["r0"].to(d))
        self.n0.copy_(s["n0"].to(d))
        for k in self.orders:
            self.pc[k].copy_(s["pc"][str(k)].to(d))
            self.st[k].copy_(s["st"][str(k)].to(d))
            self.sl[k].copy_(s["sl"][str(k)].to(d))
            self.slots[k].load(s["slots"][str(k)])
        for k in self.pools:
            self.pools[k].load(s["pools"][str(k)])
        self.net.load_state_dict(s["net"])
        # 1.4.0: a slim checkpoint (shared, read-only use) carries no optimiser state. Learning from one starts fresh
        # moments, so its rate re-warms over REWARM ticks instead of taking Adam's full-size first steps
        if s.get("opt") is not None:
            self.opt.load_state_dict(s["opt"])
        else:
            self._rewarm = (int(self.chunks), self.REWARM)
        if s.get("mopt") is not None:
            self.mopt.load_state_dict(s["mopt"])
        if s.get("rewarm") is not None:
            self._rewarm = tuple(int(v) for v in s["rewarm"])
        with torch.no_grad():
            self.wE.copy_(s["wE"].to(d))
            self.wM.copy_(s["wM"].to(d))
            self.theta.copy_(s["theta"].to(d))
        if self.io2 is not None and s.get("io2") is not None:
            self.io2.load(s["io2"])
        p_ = s.get("past")
        self._past, self._past_end = (p_["f"].to(d), int(p_["end"])) if p_ else (None, -1)
        kv_ = s.get("kv")
        self._kv, self._kv_end = ([(k.to(d), v.to(d)) for k, v in zip(kv_["k"], kv_["v"])], int(kv_["end"])) if kv_ \
            else (None, -1)
        if s.get("ctl"):
            self.ctl_lr, self.ctl_cos = float(s["ctl"]["lr"]), float(s["ctl"]["cos"])
            cs = s["ctl"]; self._lag_k = int(cs.get("k", 0)); self._snap_k = int(cs.get("snap_k", -10**9))
            for nm in ("old", "new"):
                if cs.get(nm) is not None:
                    setattr(self, "_snap_" + nm, [t.to(d) for t in cs[nm]])
            if cs.get("hg_h") is not None:
                self._hg_h = [t.to(d) for t in cs["hg_h"]]
            self._hg_r, self._hg_n, self.hg_clamps = float(cs.get("hg_r", 0.0)), int(cs.get("hg_n", 0)), int(cs.get("hg_clamps", 0))
        sel = s.get("sel") or {}
        self._sel_mass, self._sel_phase = float(sel.get("mass", 0.0)), float(sel.get("phase", 0.0))
