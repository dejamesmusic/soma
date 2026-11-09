"""somatic_core engine, second form: the same store and readers, with the read factorised (Nov 8).

What changes against core.py, and why (per-tick cost, James Nov 8: "it's per tick cost ... that matters"):

1. EXACT INSIDE THE TICK. Every position's counts include the earlier positions of its own tick (a C x C mask per
   order, decayed by keep^(i - j)), so the exact reader equals the one-token-per-tick reader at any chunk size. The
   chunk becomes a compute and plasticity clock only (one optimiser step per tick, like v12's 512-token batch); it no
   longer changes what the exact reader sees.
2. ONE DENSE TENSOR. The back-off chain is affine in its lower level, so the final distribution per rung is
       p_K = Gamma_0 p_0 + sum_k coef_k c_k
   with c_k sparse (the node's successor slots plus in-tick successors) and p_0 the order-0 row. Only p_K is ever
   materialised over the vocabulary: (C, rungs, V) once, instead of every order's chain and its log.
3. LINEAR FEATURES FOR THE LEARNER. The learner reads each chain level's expected projection, P^T p_L (P: V x proj,
   learned), computed exactly from the same sparse decomposition, plus log(1 + node total). core.py projected the
   clamped log-distributions instead; this is a change of the learner's perspective and is tested as one.

Lazy decay is stamped at the tick end: a count from position j is stored as keep^(t_end - t_j) and read at time t
as stored * keep^(t - t_end) = keep^(t - t_j), exactly as in core.py at C = 1.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .core import LB, NE, ENT_EDGES, PHI, DEFAULTS, IDENTITY, Matcher, Pool, Slots, Core as _Core
from .io2 import IO2


class Block(torch.nn.Module):
    """a pre-LayerNorm residual MLP block (depth > 1)."""
    def __init__(self, H, X):
        super().__init__()
        self.ln = torch.nn.LayerNorm(H)
        self.up = torch.nn.Linear(H, X)
        self.down = torch.nn.Linear(X, H)
        torch.nn.init.zeros_(self.down.weight)
        torch.nn.init.zeros_(self.down.bias)

    def forward(self, h):
        return h + self.down(torch.nn.functional.gelu(self.up(self.ln(h))))


class Learner2(torch.nn.Module):
    def __init__(self, V, W, D, H, DP, O, NB, depth=1, final_norm=0):
        super().__init__()
        self.emb = torch.nn.Embedding(V, D)
        self.win = torch.nn.Linear(W * D, H)
        self.P = torch.nn.Parameter(torch.randn(V, DP) * 0.5)
        self.ln = torch.nn.LayerNorm(O * NB * (DP + 1))
        self.cell = torch.nn.Linear(O * NB * (DP + 1), H, bias=False)
        self.l2 = torch.nn.Linear(H, H)
        self.blocks = torch.nn.ModuleList([Block(H, H) for _ in range(max(0, int(depth) - 1))])
        # final_norm (1.0.1): a LayerNorm on the residual stream before the output. without it the stream is unnormalised
        # and Adam's first steps on the zero-initialised block outputs kick it to norms in the thousands (2048x3 survived
        # the transient; 4096x4 diverged). checkpoints made before it load with final_norm 0 and are unchanged.
        self.ln_f = torch.nn.LayerNorm(H) if final_norm else None
        self.out = torch.nn.Linear(H, V)
        self.keep_h, self.h = False, None

    def forward(self, win, feat):
        h = torch.nn.functional.gelu(self.win(self.emb(win).reshape(win.shape[0], -1))
                                     + self.cell(self.ln(feat.reshape(feat.shape[0], -1))))
        h = torch.nn.functional.gelu(self.l2(h))
        for b in self.blocks:
            h = b(h)
        if self.ln_f is not None:
            h = self.ln_f(h)
        if self.keep_h and h.requires_grad:
            h.retain_grad()
            self.h = h
        return self.out(h)


class Core2(_Core):
    def __init__(self, cfg: dict, dev, blen: np.ndarray):
        super().__init__(cfg, dev, blen, parity=False)
        c = self.c
        self.net = Learner2(self.V, int(c["window"]), int(c["emb"]), int(c["hidden"]), int(c["proj"]), self.O,
                            self.NB, depth=int(c.get("depth", 1)), final_norm=int(c.get("final_norm", 0))).to(self.dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(c["lr"]), fused=self.dev.type in ("mps", "cuda"))
        # the rung and match weights: SGD at mix_lr * C (core.py) takes one large step per tick and overshoots at
        # large ticks; Adam's step does not scale with the tick (mix_opt 'adam', lr mix_adam)
        self.mix_opt = str(c.get("mix_opt", "adam"))
        self.mopt = torch.optim.Adam([self.wE, self.wM], lr=float(c.get("mix_adam", 3e-3)))
        # io2 (v14 form) sets the learner's learning rate from its translated residual, once per quantum
        self.io2 = None
        if int(c.get("io2", 0)):
            self.io2 = IO2(int(c["hidden"]), self.V, lr_base=float(c.get("lr_base", 2e-3)),
                           quantum=float(c.get("io2_quantum", 512)))
            for g in self.opt.param_groups:
                g["lr"] = self.io2.lr()

    _prof = None                                                          # dict -> per-section seconds (synchronised)

    def _t(self, name):
        if self._prof is None:
            return
        import time as _time
        torch.mps.synchronize() if self.dev.type == "mps" else None
        now = _time.perf_counter()
        last = self._prof.get("_last")
        if last is not None:
            self._prof[name] = self._prof.get(name, 0.0) + now - last
        self._prof["_last"] = now

    # ---- CPU: group ids that make in-tick sharing exact across a recycle ----
    @staticmethod
    def _gids(ids, rec, C):
        """per order, an id per position that is equal only for positions sharing a node AND a context
        (a node recycled inside the tick serves a different context after its recycle position)."""
        out = {}
        for k, ev in rec.items():
            g = ids[k - 1].copy()
            if ev:
                bump = {}
                e = 0
                for i in range(C):
                    while e < len(ev) and ev[e][0] == i:
                        bump[ev[e][1]] = bump.get(ev[e][1], 0) + 1
                        e += 1
                    n = ids[k - 1, i]
                    if n >= 0 and n in bump:
                        g[i] = n + (1 << 40) * bump[n]
            out[k] = g
        return out

    # ---- GPU: the factorised read ----
    def _read2(self, idt, gid, maps, ys, mp, mb, win, t0, grad, rows=None):
        """-> zE inputs: lpE, lpL (C, V), gate key, deep; plus nothing else (no lch)."""
        C, V, NB, d = win.shape[0], self.V, self.NB, self.dev
        keep = 1.0 - self.ab                                             # (NB,)
        pos = torch.arange(C, device=d)
        y = torch.from_numpy(ys).to(d)
        with torch.no_grad():
            lag = (pos[:, None] - pos[None, :]).float()                  # i - j
            lower = (lag > 0).float()
            W0 = keep[None, None, :] ** lag.clamp(min=0)[..., None] * lower[..., None]   # (C, C, NB) [i, j, b]
            W0 = W0.permute(0, 2, 1).contiguous()                       # (C, NB, C) [i, b, j]
            self._t('read:in-tick masks')
            # order 0
            f0 = keep[None, :] ** (t0 - self.r0_t + pos.float())[:, None]                 # (C, NB)
            n0i = f0 * self.n0[None] + W0.sum(-1)                                        # (C, NB)
            # orders >= 1: merged counts per position
            toks, vals, Ns = [], [], []
            for k in self.orders:
                idc = idt[k]
                M = maps[k]                                                              # (C, K+1) tokens
                fk = keep[None, :] ** (t0 + pos.float() - self.sl[k][idc].float())[:, None]
                R = self.sc[k][idc] * fk[..., None]                                      # (C, NB, K+1)
                g = torch.from_numpy(gid[k]).to(d)
                same = ((g[:, None] == g[None, :]) & (g[:, None] >= 0)).float()          # (C, C) [i, j]
                Wk = W0 * same[:, None, :]                                               # (C, NB, C)
                N = self.st[k][idc] * fk + Wk.sum(-1)
                # in-tick successors already in the node's slot row go into that slot
                hit = (M[:, :self.KK[k]] == y[:, None]) & (torch.from_numpy(maps[f"v{k}"]).to(d)[:, :self.KK[k]])
                sidx = torch.where(hit.any(1), hit.float().argmax(1), torch.full_like(y, -1))   # (C,) per j
                ins = (sidx >= 0).float()
                R = R.scatter_add(2, sidx.clamp(min=0)[None, None, :].expand(C, NB, C), Wk * ins[None, None, :])
                # the rest, aggregated at the first in-tick occurrence of (context, token)
                eq = (y[:, None] == y[None, :]) & (g[:, None] == g[None, :])            # (C, C) [j, j']
                first = torch.where(eq & (pos[None, :] <= pos[:, None]), pos[None, :], C).min(1).values
                F = torch.nn.functional.one_hot(first, C).float()                        # (C, C) [j, first]
                A = (Wk * (1 - ins)[None, None, :]) @ F                                  # (C, NB, C)
                toks.append(torch.cat([M, y[None, :].expand(C, C)], 1))                  # (C, K+1+C)
                vals.append(torch.cat([R, A], 2))                                        # (C, NB, K+1+C)
                Ns.append(N)
            self._t('read:merge orders')
            # the chain's coefficients
            hs, gs = [], []
            last = (torch.arange(NB, device=d) == NB - 1)[None, :]
            for k in range(len(self.orders)):
                v, N = vals[k], Ns[k]
                dist = (v > 0.05).sum(-1).float()
                lost = torch.clamp(N - v.sum(-1), min=0.0)
                # all-time rung: at least one type. Halving scales its counts, so after five halvings a pair seen once
                # (1/32) falls under the 0.05 type threshold; with no type the escape was exactly 0 and every other
                # token read log 0 = -inf (Nov 8, James's run at 102M bytes). Equal to dist + (N == 0) everywhere else.
                bt = torch.where(last, dist.clamp(min=1.0), dist + 1.0)
                h = 1.0 / (N + bt)
                hs.append(h)
                gs.append((bt + lost) * h)
            K = len(self.orders)
            # final level: p_K = G0 * p0 + sum_k coef_k c_k
            G0 = torch.ones(C, NB, device=d)
            coef = [None] * K
            run = torch.ones(C, NB, device=d)
            for k in range(K - 1, -1, -1):
                coef[k] = hs[k] * run
                run = run * gs[k]
            G0 = run                                                                     # prod of all g
            self._t('read:chain coefs')
            a0 = G0 / (n0i + 1.0)
            P = torch.addcmul((a0 / V)[..., None], (a0 * f0)[..., None], self.r0[None])  # (C, NB, V), one kernel
            P.scatter_add_(2, y[None, None, :].expand(C, NB, C), W0 * a0[..., None])
            for k in range(K):
                T = toks[k].shape[1]
                P.scatter_add_(2, toks[k][:, None, :].expand(C, NB, T), vals[k] * coef[k][..., None])
            self._t('read:dense P build')
            LP = P.clamp_min_(1e-30).log_()                                             # never log 0
            deep = torch.stack([(n0i[:, -1] > 0).long() * 0] + [(Ns[k][:, -1] > 0).long() * (k + 1) for k in range(K)],
                               1).max(1).values
            mpt = torch.from_numpy(mp).to(d)
            mbt = torch.from_numpy(mb).to(d)
            hm = mpt >= 0
            bump = math.log(0.98 + 0.02 / V) - math.log(0.02 / V)
        # rungs mixed in log space: a batched mat-vec, not a broadcast; the match model's log-distribution is a
        # constant per row (cancelled by the softmax) plus a bump on the predicted token, so only the bump is added
        zE = torch.bmm(self.wE[deep][:, None, :], LP).squeeze(1)
        zE = zE.index_put((torch.arange(C, device=d)[hm], mpt[hm]), (self.wM[deep, mbt] * bump)[hm], accumulate=True)
        lpE = torch.log_softmax(zE, -1)
        self._t('read:log + exact mix')
        # ---- the learner's features: P^T p_L for every level L, exactly, from the same decomposition ----
        def features(ix):
            """the learner's features for rows ix (None: every row), exactly, from the same decomposition."""
            sel = (lambda t: t) if ix is None else (lambda t: t[ix])
            n = C if ix is None else len(ix)
            Pm = self.net.P                                                              # (V, DP)
            X0 = (sel(f0)[..., None] * (self.r0 @ Pm)[None] + Pm.sum(0) / V
                  + torch.einsum("ibj,jd->ibd", sel(W0), Pm[y])) / (sel(n0i) + 1.0)[..., None]   # (n, NB, DP)
            Xs = [torch.einsum("ibt,itd->ibd", sel(vals[k]), Pm[sel(toks[k])]) for k in range(K)]
            fs = [torch.cat([X0, torch.log1p(sel(n0i))[..., None]], -1)]
            for L in range(1, K + 1):
                run = torch.ones(n, NB, device=d)
                acc = 0.0
                for k in range(L - 1, -1, -1):
                    acc = acc + (sel(hs[k]) * run)[..., None] * Xs[k]
                    run = run * sel(gs[k])
                acc = acc + run[..., None] * X0
                fs.append(torch.cat([acc, torch.log1p(sel(Ns[L - 1]))[..., None]], -1))
            return torch.stack(fs, 1)                                                    # (n, O, NB, DP+1)

        wt = torch.from_numpy(win).to(d)
        self._sel_lp = None
        if rows is None:
            with torch.set_grad_enabled(grad):
                feat = features(None)
                self._t('read:features')
                lpL = torch.log_softmax(self._learner(wt, feat), -1)
                self._t('learner fwd')
        else:
            # learner decimation by prediction: every row is predicted without a graph; only the selected rows are
            # recomputed with one (the learner reads each row on its own, so the two passes agree exactly)
            with torch.no_grad():
                keep, self.net.keep_h = self.net.keep_h, False
                feat = features(None)
                lpL = torch.log_softmax(self._learner(wt, feat), -1)
                self.net.keep_h = keep
            self._t('learner fwd')
            with torch.set_grad_enabled(grad):
                self._sel_lp = torch.log_softmax(self._learner(wt[rows], features(rows)), -1)
            self._t('read:features')
        with torch.no_grad():
            ent = -(lpE.exp() * lpE).sum(-1) / math.log(2)
            key = (deep * LB + mbt) * NE + torch.bucketize(ent, self.edges)
        if getattr(self, 'probe', None) is not None:          # instruments only (tests/receipts.py); read, never written
            self._pr = dict(LP=LP, feat=feat.detach(), wt=wt, deep=deep, mpt=mpt, mbt=mbt, hm=hm, bump=bump)
        return lpE, lpL, key, deep

    def _learner(self, wt, feat):
        """the learner's logits (fp32); on cuda with learner_bf16 its matmuls run in bf16 autocast."""
        if self.dev.type == "cuda" and int(self.c.get("learner_bf16", 0)):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = self.net(wt, feat)
            return z.float()
        return self.net(wt, feat)

    def _maps2(self, ids):
        out = {}
        for k in self.orders:
            S = self.slots[k]
            n = ids[k - 1]
            M = np.full((len(n), S.K + 1), -1, np.int64)
            ok = n >= 0
            M[ok, :S.K] = S.tok[n[ok]]
            out[f"v{k}"] = M >= 0
            out[k] = torch.from_numpy(np.maximum(M, 0)).to(self.dev)
        return out

    # ---- one tick ----
    def step(self, m: Matcher, ys, learn: bool = True):
        ys = np.asarray(ys, np.int64)
        C = len(ys)
        d = self.dev
        t0 = self.clock
        self._t('_start')
        ids, rec, mp, mb, win = self._contexts(m, ys, learn)
        self._t('cpu contexts')
        idt = self._idt(ids)
        gid = self._gids(ids, rec, C)
        if learn:
            with torch.no_grad():
                for k in self.orders:
                    if rec[k]:
                        nd = torch.tensor([n for _, n in rec[k]], device=d)
                        self.sc[k][nd] = 0.0
                        self.st[k][nd] = 0.0
                        self.sl[k][nd] = t0
        maps = self._maps2(ids)
        y = torch.from_numpy(ys).to(d)
        self._t('maps + recycles')
        lpE, lpL, key, deep = self._read2(idt, gid, maps, ys, mp, mb, win, t0, grad=learn)
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
        if not learn:
            return -torch.log2(pf)
        self._t('scoring')
        loss = -lyL.mean()
        lossE = -lyE.mean()
        self.opt.zero_grad()
        loss.backward()
        gE = torch.autograd.grad(lossE, [self.wE, self.wM])
        self._t('backward')
        with torch.no_grad():
            self.theta.index_put_((key,), float(self.c["gate_lr"]) * sg * (1 - sg) * (a_ - b_) / pf, accumulate=True)
        self.opt.step()
        if self.io2 is not None:
            with torch.no_grad():
                Wo = self.net.out.weight                                                   # (V, H)
                res = (Wo[y] - lpL.detach().exp() @ Wo) / Wo.norm(dim=1).mean()          # (C, H)
            self.io2.advance(res, np.ones(C))
            if self.io2.accumulate(float(-lyL.detach().sum()), C, float(self.blen[ys].sum())):
                for g in self.opt.param_groups:
                    g["lr"] = self.io2.lr()
        self._t('adam')
        slot, new = self._assign(ids, rec, ys)
        self._t('cpu assign')
        with torch.no_grad():
            if self.mix_opt == "adam":
                self.wE.grad, self.wM.grad = gE[0], gE[1]
                self.mopt.step()
            else:
                lr_e = float(self.c["mix_lr"]) * C
                self.wE -= lr_e * gE[0]
                self.wM -= lr_e * gE[1]
            self._zero_new(new)
            keep = 1.0 - self.ab
            te = t0 + C
            wj = keep[None, :] ** (C - torch.arange(C, device=d).float())[:, None]       # (C, NB) stamped at te
            self.r0 *= (keep ** float(te - self.r0_t))[:, None]
            self.n0 *= keep ** float(te - self.r0_t)
            self.r0_t = te
            NB = self.NB
            ib = self.ar[None].expand(C, NB)
            self.r0.index_put_((ib, y[:, None].expand(C, NB)), wj, accumulate=True)
            self.n0 += wj.sum(0)
            for k in self.orders:
                idc = idt[k]
                Un = np.unique(ids[k - 1])
                U = torch.from_numpy(Un[Un >= 0]).to(d)                                  # unique on the CPU: no GPU sort/sync
                f = keep[None] ** (te - self.sl[k][U]).float()[:, None]
                self.sc[k][U] *= f[..., None]
                self.st[k][U] *= f
                self.sl[k][U] = te
                ix = idc[:, None].expand(C, NB)
                sj = torch.from_numpy(slot[k]).to(d)[:, None].expand(C, NB)
                self.sc[k].index_put_((ix, ib, sj), wj, accumulate=True)
                self.st[k].index_put_((ix, ib), wj, accumulate=True)
                self.sc[k][self.PK[k]] = 0.0
                self.st[k][self.PK[k]] = 0.0
                self.sc[k][idc, :, self.KK[k]] = 0.0
        self._t('store update')
        self.clock += C
        self.chunks += 1
        self.n0_all += C
        if self.n0_all > float(self.c["alltime_cap"]):
            self._halve()
        return -torch.log2(pf)

    def stats(self) -> dict:
        out = super().stats()
        if self.io2 is not None:
            out.update(self.io2.report())
        else:
            out["lr"] = float(self.opt.param_groups[0]["lr"])
        return out

    @torch.no_grad()
    def next_distribution(self, m: Matcher) -> torch.Tensor:
        ids, rec, mp, mb, win = self._contexts(m, None, False)
        ys = np.zeros(1, np.int64)
        gid = self._gids(ids, rec, 1)
        lpE, lpL, key, deep = self._read2(self._idt(ids), gid, self._maps2(ids), ys, mp, mb, win, self.clock, False)
        sg = torch.sigmoid(self.theta[key])
        return (sg[:, None] * lpE.exp() + (1 - sg)[:, None] * lpL.exp())[0]
