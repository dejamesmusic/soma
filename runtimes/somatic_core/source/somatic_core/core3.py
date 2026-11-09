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

from .core2 import Core2
from .core import Matcher
from . import bounds


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

    def state(self) -> dict:
        return dict(tok=self.tok.copy(), pid=self.pid.copy(), pcnt=self.pcnt.copy(), plast=self.plast.copy(),
                    pnode=self.pnode.copy(), pslot=self.pslot.copy(), free=self.free[:self.nfree].copy(),
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
        self.NPk = {k: NPd[k] for k in self.orders}
        self.slots = {k: PairSlots(self.PK[k], self.KK[k], self.NPk[k], lgc, 100 + k) for k in self.orders}
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
            self.opt = torch.optim.AdamW(self.net.parameters(), lr=float(c["lr"]), weight_decay=float(c.get("wd", 1e-4)),
                                         fused=self.dev.type in ("mps", "cuda"))
            if self.io2 is not None:
                for g in self.opt.param_groups:
                    g["lr"] = self.io2.lr()

    def store_floats(self) -> int:
        return sum(self.pc[k].numel() + self.st[k].numel() for k in self.orders) + self.r0.numel()

    def _maps2(self, ids):
        out = {}
        for k in self.orders:
            S = self.slots[k]
            n = ids[k - 1]
            M = np.full((len(n), S.K + 1), -1, np.int64)
            Pm = np.full((len(n), S.K + 1), self.NPk[k], np.int64)
            ok = n >= 0
            M[ok, :S.K] = S.tok[n[ok]]
            pp = S.pid[n[ok]].astype(np.int64)
            Pm[ok, :S.K] = np.where(pp >= 0, pp, self.NPk[k])
            out[f"v{k}"] = M >= 0
            out[k] = torch.from_numpy(np.maximum(M, 0)).to(self.dev)
            self._pidmap[k] = torch.from_numpy(Pm).to(self.dev)
        return out

    def _node_pairs(self, k, nodes: np.ndarray):
        """flat pair ids of the given nodes and, for each, the index of its node in `nodes`."""
        S = self.slots[k]
        P = S.pid[nodes]
        r, cidx = np.nonzero(P >= 0)
        return P[r, cidx].astype(np.int64), r

    def _assign(self, ids, rec, ys):
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

    def step(self, m: Matcher, ys, learn: bool = True):
        ys = np.asarray(ys, np.int64)
        C = len(ys)
        d = self.dev
        t0 = self.clock
        self._t('_start')
        ids, rec, mp, mb, win = self._contexts(m, ys, learn)
        idt = self._idt(ids)
        gid = self._gids(ids, rec, C)
        self._t('cpu contexts')
        if learn:
            with torch.no_grad():                         # recycled nodes (and their pairs) read as empty this tick
                for k in self.orders:
                    if rec[k]:
                        nodes = np.array([n for _, n in rec[k]], np.int64)
                        nd = torch.from_numpy(nodes).to(d)
                        self.st[k][nd] = 0.0
                        self.sl[k][nd] = t0
                        pp, _ = self._node_pairs(k, nodes)
                        if len(pp):
                            self.pc[k][torch.from_numpy(pp).to(d)] = 0.0
        maps = self._maps2(ids)
        y = torch.from_numpy(ys).to(d)
        self._t('maps + recycles')
        # learner decimation. The learner predicts every token either way; what changes is what it learns from.
        #   skip        its optimiser steps on one tick in learn_stride; the other ticks give it no gradient
        #   accumulate  every tick's gradient is kept, averaged over learn_stride ticks, one step
        #   select      by prediction (rma / v14): every s-th token (s = learn_stride) gives a gradient, one step per io2
        #               quantum of those predictions; io2 hears the sampled residuals with their gaps
        #   io2         select, with s chosen by io2's decimation band (phi^band, dithered so the density is exact)
        mode = str(self.c.get("learn_mode", "skip"))
        sel_ = mode in ("select", "io2")
        if learn:
            self._io2_flush()                             # last tick's io2 host work: waits only on that tick's tail
        if sel_:
            s_, acc_ = (self._sel_stride(mode) if learn else 1), False
            Q = float(self.io2.quantum) if self.io2 is not None else float(C)
            rows_np = np.arange((-t0) % s_, C, s_) if s_ > 1 else None
            nsel = C if rows_np is None else len(rows_np)
            lgrad = learn and nsel > 0
            self._sel_mass = getattr(self, "_sel_mass", 0.0) + (nsel if learn else 0)
            upd = learn and self._sel_mass + 1e-9 >= Q
            rows = torch.from_numpy(rows_np).to(d) if rows_np is not None and nsel > 0 else None
        else:
            s_ = max(1, int(self.c.get("learn_stride", 1)))
            acc_ = mode == "accumulate"
            upd = learn and self.chunks % s_ == s_ - 1
            lgrad = learn and (s_ == 1 or upd or acc_)
            rows, rows_np, nsel = None, None, C
        self.net.keep_h = lgrad and self.io2 is not None             # io2's residual comes from the backward pass
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
        self.last = dict(tokens=ys, bits=-torch.log2(pf), exact=-lyE.detach() / math.log(2), next=pfull[-1].topk(5))
        if getattr(self, 'probe', None) is not None:          # before any learning from this tick
            self.probe(self, y, sg, lpE.detach(), lpL.detach())
        self._t('scoring')
        if not learn:
            return -torch.log2(pf)
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
                (-lyL.mean() / (s_ if acc_ else 1)).backward()
        gE = torch.autograd.grad(lossE, [self.wE, self.wM])
        slot, pair, new = self._assign(ids, rec, ys)          # CPU, while the GPU runs the read and backward
        self._t('backward')
        with torch.no_grad():
            self.theta.index_put_((key,), float(self.c["gate_lr"]) * sg * (1 - sg) * (a_ - b_) / pf, accumulate=True)
        if not upd:
            pass
        elif self.bounded:
            params = [p for p in self.net.parameters()]
            before = bounds.norms_before(params)
            bounds.clip_(params, 1.0)                                   # torch's clip: 57 ms/tick on MPS, this ~2
            self.opt.step()
            bounds.compress_(params, before)
        else:
            self.opt.step()
        if upd and (acc_ or sel_):
            self.opt.zero_grad()
        if upd and sel_:
            self._sel_mass = 0.0
        if self.chunks % 1024 == 0:
            bounds.cap_steps_(self.opt)
            bounds.cap_steps_(self.mopt)
        self._t('optimiser + bounds')
        if self.io2 is not None:
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
            if self.mix_opt == "adam":
                self.wE.grad, self.wM.grad = gE[0], gE[1]
                self.mopt.step()
            else:
                lr_e = float(self.c["mix_lr"]) * C
                self.wE -= lr_e * gE[0]
                self.wM -= lr_e * gE[1]
            if self.bounded:
                for v in (self.wE, self.wM, self.theta):
                    v.clamp_(-bounds.SCALAR_BOUND, bounds.SCALAR_BOUND)
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
                    self.pc[k][torch.tensor(new[k], device=d)] = 0.0
                idc = idt[k]
                Un = np.unique(ids[k - 1])
                Un = Un[Un >= 0]
                U = torch.from_numpy(Un).to(d)
                f = keep[None] ** (te - self.sl[k][U]).float()[:, None]                  # (|U|, NB)
                pp, owner = self._node_pairs(k, Un)
                if len(pp):
                    self.pc[k][torch.from_numpy(pp).to(d)] *= f[torch.from_numpy(owner).to(d)]
                self.st[k][U] *= f
                self.sl[k][U] = te
                pj = torch.from_numpy(pair[k]).to(d)[:, None].expand(C, NB)
                self.pc[k].index_put_((pj, ib), wj, accumulate=True)
                ix = idc[:, None].expand(C, NB)
                self.st[k].index_put_((ix, ib), wj, accumulate=True)
                self.pc[k][self.NPk[k]] = 0.0
                self.st[k][self.PK[k]] = 0.0
        self._t('store update')
        self.clock += C
        self.chunks += 1
        self.n0_all += C
        if self.n0_all > float(self.c["alltime_cap"]):
            self._halve()
        if self.clock >= self.rebase_at:
            self._rebase()
        return -torch.log2(pf)

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
            pool.last = [v - off for v in pool.last]
        self.chunks -= offc
        self.rebases = getattr(self, "rebases", 0) + 1

    def stats(self) -> dict:
        out = super().stats()
        if self.bounded:
            out.update(bounds.pressure(list(self.net.parameters())))
            out["max_abs_theta"] = float(self.theta.abs().max())
        for k in self.orders:
            S = self.slots[k]
            out[f"o{k}_pairs"] = S.NP - S.nfree
            out[f"o{k}_pair_fill"] = (S.NP - S.nfree) / S.NP
            out[f"o{k}_pair_evictions"] = S.pair_evictions
        return out

    # ---- state (bounded and self-contained: everything the learner carries) ----
    def state_dict(self) -> dict:
        self._io2_flush()
        if self.dev.type == "mps":
            torch.mps.synchronize()
        cpu = lambda t: t.detach().to("cpu").clone()
        return dict(
            engine="core3", config=dict(self.c), clock=self.clock, r0_t=self.r0_t, n0_all=self.n0_all,
            halvings=self.halvings, chunks=self.chunks, rebases=getattr(self, "rebases", 0),
            r0=cpu(self.r0), n0=cpu(self.n0),
            pc={str(k): cpu(self.pc[k]) for k in self.orders}, st={str(k): cpu(self.st[k]) for k in self.orders},
            sl={str(k): cpu(self.sl[k]) for k in self.orders},
            slots={str(k): self.slots[k].state() for k in self.orders},
            pools={str(k): self.pools[k].state() for k in self.pools},
            net=self.net.state_dict(), opt=self.opt.state_dict(), mopt=self.mopt.state_dict(),
            wE=cpu(self.wE), wM=cpu(self.wM), theta=cpu(self.theta),
            io2=self.io2.state() if self.io2 is not None else None,
            sel=dict(mass=float(getattr(self, "_sel_mass", 0.0)), phase=float(getattr(self, "_sel_phase", 0.0))),
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
        self.opt.load_state_dict(s["opt"])
        self.mopt.load_state_dict(s["mopt"])
        with torch.no_grad():
            self.wE.copy_(s["wE"].to(d))
            self.wM.copy_(s["wM"].to(d))
            self.theta.copy_(s["theta"].to(d))
        if self.io2 is not None and s.get("io2") is not None:
            self.io2.load(s["io2"])
        sel = s.get("sel") or {}
        self._sel_mass, self._sel_phase = float(sel.get("mass", 0.0)), float(sel.get("phase", 0.0))
