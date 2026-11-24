"""Every lane's store read in one pass (Nov 24, throughput; lanes_read). core2._read2 + _chain + _feat with a leading lane
dimension: the same arithmetic per lane, about one eighth of the kernel launches (after batching the learner the trainer
was bound by the CPU launching ~7.7k small kernels a round, most of them the eight lanes' reads).

Each lane keeps its own store; its rows are gathered per lane (cheap), then everything after runs once over (B, C, ...).
The in-tick decay W0 depends only on the tick length, so it is computed once. Lanes configuration only: no skip orders, no
leave-one-out, no decimation by prediction, no rehearsal. tests/lanes_read.py checks it against _read2 lane by lane.

  out = read_lanes(cores, items)     # items: lane_pre(..., read=False) per lane
  out: lpE (B, C, V) with grad to the mixer, key (B, C), deep (B, C), feat (B, C, O, NB, DP+1) with grad to the learner's P
"""
import math

import torch

from .core import LB, NE


def read_lanes(cores, items, group: int = 0):
    """group > 0: the lanes read `group` at a time (the batched read's working memory grows with the lanes), joined"""
    group = group or int(cores[0].c.get("lanes_read_group", 0))
    if group and len(cores) > group:
        parts = [_read(cores[i:i + group], items[i:i + group]) for i in range(0, len(cores), group)]
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    return _read(cores, items)


def _read(cores, items):
    m = cores[0]                                            # shared: ladder, orders, mixer, gate edges, the learner's P
    B, C = len(items), items[0]["C"]
    V, NB, d, K = m.V, m.NB, m.dev, m.KCH
    assert len(m.orders) == K, "lanes_read: no skip orders"
    keep = 1.0 - m.ab                                        # (NB,)
    pos = torch.arange(C, device=d)
    y = torch.stack([it["y"] for it in items])                                   # (B, C)
    t0 = torch.tensor([it["t0"] for it in items], device=d)                      # (B,) the lanes' clocks
    with torch.no_grad():
        lag = (pos[:, None] - pos[None, :]).float()
        lower = (lag > 0).float()
        W0 = keep[None, None, :] ** lag.clamp(min=0)[..., None] * lower[..., None]
        W0 = W0.permute(0, 2, 1).contiguous()                                     # (C, NB, C) [i, b, j], every lane's
        r0t = torch.tensor([c.r0_t for c in cores], device=d)
        f0 = keep[None, None, :] ** ((t0 - r0t)[:, None].float() + pos.float()[None, :])[..., None]  # (B, C, NB)
        n0 = torch.stack([c.n0 for c in cores])                                   # (B, NB)
        n0i = f0 * n0[:, None, :] + W0.sum(-1)[None]                              # (B, C, NB)
        toks, vals, Ns = [], [], []
        for k in m.orders:
            idc = [it["idt"][k] for it in items]
            M = torch.stack([it["maps"][k] for it in items])                     # (B, C, K+1)
            Mv = torch.stack([it["maps"][f"v{k}"] for it in items])
            SL = torch.stack([c.sl[k][i] for c, i in zip(cores, idc)])           # (B, C)
            fk = keep[None, None, :] ** ((t0[:, None] - SL).float() + pos.float()[None, :])[..., None]   # (B, C, NB)
            R = torch.stack([c.sc[k][i] for c, i in zip(cores, idc)]) * fk[..., None]          # (B, C, NB, K+1)
            g = torch.stack([c._dv(it["gid"][k]) for c, it in zip(cores, items)])              # (B, C)
            same = ((g[:, :, None] == g[:, None, :]) & (g[:, :, None] >= 0)).float()            # (B, C, C)
            Wk = W0[None] * same[:, :, None, :]                                                # (B, C, NB, C)
            N = torch.stack([c.st[k][i] for c, i in zip(cores, idc)]) * fk + Wk.sum(-1)
            KK = m.KK[k]
            hit = (M[..., :KK] == y[..., None]) & Mv[..., :KK]
            sidx = torch.where(hit.any(-1), hit.float().argmax(-1), torch.full_like(y, -1))     # (B, C)
            ins = (sidx >= 0).float()
            R = R.scatter_add(3, sidx.clamp(min=0)[:, None, None, :].expand(B, C, NB, C), Wk * ins[:, None, None, :])
            eq = (y[:, :, None] == y[:, None, :]) & (g[:, :, None] == g[:, None, :])
            first = torch.where(eq & (pos[None, None, :] <= pos[None, :, None]), pos[None, None, :], C).min(-1).values
            F = torch.nn.functional.one_hot(first, C).float()                                  # (B, C, C)
            # (B, C*NB, C) @ (B, C, C): one batched product per tick (F[:, None] broadcast it into B*C tiny ones)
            A = torch.bmm((Wk * (1 - ins)[:, None, None, :]).reshape(B, C * NB, C), F).reshape(B, C, NB, C)
            toks.append(torch.cat([M, y[:, None, :].expand(B, C, C)], 2))                       # (B, C, K+1+C)
            vals.append(torch.cat([R, A], 3))                                                   # (B, C, NB, K+1+C)
            Ns.append(N)
        hs, gs = _chain(vals, Ns, NB, d)
        coef = [None] * K
        run = torch.ones(B, C, NB, device=d)
        for k in range(K - 1, -1, -1):
            coef[k] = hs[k] * run
            run = run * gs[k]
        G0 = run
        a0 = G0 / (n0i + 1.0)
        R0 = torch.stack([c.r0 for c in cores])                                   # (B, NB, V)
        P = torch.addcmul((a0 / V)[..., None], (a0 * f0)[..., None], R0[:, None])  # (B, C, NB, V)
        P.scatter_add_(3, y[:, None, None, :].expand(B, C, NB, C), W0[None] * a0[..., None])
        for k in range(K):
            T = toks[k].shape[2]
            P.scatter_add_(3, toks[k][:, :, None, :].expand(B, C, NB, T), vals[k] * coef[k][..., None])
        LP = P.clamp_min_(1e-30).log_()
        del P
        deep = torch.stack([(n0i[..., -1] > 0).long() * 0] + [(Ns[k][..., -1] > 0).long() * (k + 1) for k in range(K)],
                           2).max(2).values                                                    # (B, C)
        mpt = torch.stack([c._dv(it["mp"]) for c, it in zip(cores, items)])
        mbt = torch.stack([c._dv(it["mb"]) for c, it in zip(cores, items)])
        hm = mpt >= 0
        bump = math.log(0.98 + 0.02 / V) - math.log(0.02 / V)
    dz, mz, hz = deep.reshape(-1), mbt.reshape(-1), hm.reshape(-1)
    zE = torch.bmm(m.wE[dz][:, None, :], LP.reshape(B * C, NB, V)).squeeze(1)                  # (B*C, V)
    del LP
    if d.type == "cuda":
        zE = zE.scatter_add(1, mpt.reshape(-1).clamp(min=0)[:, None], (m.wM[dz, mz] * (bump * hz.float()))[:, None])
    else:
        zE = zE.index_put((torch.arange(B * C, device=d)[hz], mpt.reshape(-1)[hz]), (m.wM[dz, mz] * bump)[hz], accumulate=True)
    lpE = torch.log_softmax(zE, -1).reshape(B, C, V)
    with torch.no_grad():
        ent = -(lpE.exp() * lpE).sum(-1) / math.log(2)
        key = (deep * LB + mbt) * NE + torch.bucketize(ent, m.edges)
    feat = _feat(m, dict(f0=f0, W0=W0, n0i=n0i, y=y, vals=vals, toks=toks, hs=hs, gs=gs, Ns=Ns, K=K, B=B, C=C, R0=R0))
    return dict(lpE=lpE, key=key, deep=deep, feat=feat)


def _chain(vals, Ns, NB, d):
    hs, gs = [], []
    last = (torch.arange(NB, device=d) == NB - 1)
    for v, N in zip(vals, Ns):
        dist = (v > 0.05).sum(-1).float()
        lost = torch.clamp(N - v.sum(-1), min=0.0)
        bt = torch.where(last, dist.clamp(min=1.0), dist + 1.0)
        h = 1.0 / (N + bt)
        hs.append(h)
        gs.append((bt + lost) * h)
    return hs, gs


def _feat(m, pk):
    f0, W0, n0i, y, vals, toks, hs, gs, Ns, K, B, C, R0 = (pk[k] for k in ("f0", "W0", "n0i", "y", "vals", "toks", "hs", "gs",
                                                                           "Ns", "K", "B", "C", "R0"))
    NB, d, V = m.NB, m.dev, m.V
    Pm = m.net.P                                                                  # (V, DP)
    X0 = (f0[..., None] * (R0 @ Pm)[:, None] + Pm.sum(0) / V
          + torch.matmul(W0.reshape(C * NB, C)[None], Pm.index_select(0, y.reshape(-1)).reshape(B, C, -1)).reshape(B, C, NB, -1)
          ) / (n0i + 1.0)[..., None]
    Xs = [_succ(m, vals[k], toks[k], Pm) for k in range(K)]                        # (B, C, NB, DP)
    fs = [torch.cat([X0, torch.log1p(n0i)[..., None]], -1)]
    for L in range(1, K + 1):
        run = torch.ones(B, C, NB, device=d)
        acc = 0.0
        for k in range(L - 1, -1, -1):
            acc = acc + (hs[k] * run)[..., None] * Xs[k]
            run = run * gs[k]
        acc = acc + run[..., None] * X0
        fs.append(torch.cat([acc, torch.log1p(Ns[L - 1])[..., None]], -1))
    return torch.stack(fs, 2)                                                     # (B, C, O, NB, DP+1)


def _succ(m, v, t, Pm):
    B, C, NB, T = v.shape
    V, DP = Pm.shape
    if T * DP <= NB * V or not int(m.c.get("feat_scatter", 1)):
        return torch.einsum("zibt,zitd->zibd", v, Pm.index_select(0, t.reshape(-1)).reshape(B, C, T, DP))
    A = torch.zeros(B, C, NB, V, device=v.device, dtype=v.dtype)
    A.scatter_add_(3, t[:, :, None, :].expand(B, C, NB, T), v)
    if v.is_cuda and int(m.c.get("feat_bf16", 0)):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return (A @ Pm).float()
    return A @ Pm
