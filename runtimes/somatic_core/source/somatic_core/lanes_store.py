"""Every lane's store update in one pass (Nov 24, throughput; lanes_store). With many lanes the trainer is CPU-bound by
per-lane bookkeeping (~21 ms per lane-tick, Nov 24 profile), most of it the store update's ~20 small GPU ops per lane.

StoreBank stacks the lanes' stores: each store tensor (r0, n0, st[k], sl[k], pc[k]) becomes one tensor with a leading
lane dimension, and each lane's core keeps a view of its row (every store operation is in place, so the views stay
bound). store_update_lanes is core3._store_update for all lanes at once: the same arithmetic per lane, with each lane's
indices offset into the flattened bank. The CPU parts (unique nodes, their pairs) stay per lane; they are numpy/numba.
"""
import numpy as np
import torch

from .lanes_read import _host


class StoreBank:
    def __init__(self, cores):
        self.cores = cores
        c0 = cores[0]
        self.B, self.orders = len(cores), list(c0.orders)
        with torch.no_grad():
            self.r0 = torch.stack([c.r0 for c in cores])                 # (B, NB, V)
            self.n0 = torch.stack([c.n0 for c in cores])                 # (B, NB)
            self.st = {k: torch.stack([c.st[k] for c in cores]) for k in self.orders}    # (B, PK+1, NB)
            self.sl = {k: torch.stack([c.sl[k] for c in cores]) for k in self.orders}    # (B, PK+1)
            self.pc = {k: torch.stack([c.pc[k] for c in cores]) for k in self.orders}    # (B, NP+1, NB)
        for b, c in enumerate(cores):
            c.r0, c.n0 = self.r0[b], self.n0[b]
            for k in self.orders:
                c.st[k], c.sl[k], c.pc[k] = self.st[k][b], self.sl[k][b], self.pc[k][b]

    def bound(self) -> bool:
        """every lane's store still the bank's (a core that replaced a tensor instead of writing into it would not be)"""
        return all(c.r0.data_ptr() == self.r0[b].data_ptr() and all(c.pc[k].data_ptr() == self.pc[k][b].data_ptr()
                                                                     for k in self.orders)
                   for b, c in enumerate(self.cores))


def store_update_lanes(bank: StoreBank, items, idx=None):
    """core3._store_update for the lanes at bank positions idx (default: all); items[i]: ids, idt, pair, new, y, t0, C of
    that lane's tick"""
    idx = list(range(bank.B)) if idx is None else list(idx)
    cores = [bank.cores[i] for i in idx]
    m = bank.cores[0]
    d, NB, B = m.dev, m.NB, len(idx)
    full = idx == list(range(bank.B))
    pos = _host(idx, d)
    C = items[0]["C"]
    with torch.no_grad():
        keep = 1.0 - m.ab
        wj = keep[None, :] ** (C - torch.arange(C, device=d).float())[:, None]          # (C, NB): every lane's
        te = [it["t0"] + C for it in items]
        dt = _host([float(t - c.r0_t) for t, c in zip(te, cores)], d)
        g0 = keep[None, :] ** dt[:, None]                                                 # (B, NB)
        if full:
            bank.r0 *= g0[:, :, None]
            bank.n0 *= g0
        else:
            bank.r0[pos] = bank.r0[pos] * g0[:, :, None]
            bank.n0[pos] = bank.n0[pos] * g0
        for c, t in zip(cores, te):
            c.r0_t = t
        lane = pos[:, None, None].expand(B, C, NB)
        ib = m.ar[None, None, :].expand(B, C, NB)
        y = torch.stack([it["y"] for it in items])                                        # (B, C)
        wjB = wj[None].expand(B, C, NB)
        bank.r0.index_put_((lane, ib, y[:, :, None].expand(B, C, NB)), wjB, accumulate=True)
        if full:
            bank.n0 += wj.sum(0)[None]
        else:                                              # only the lanes that took a tick (idle lanes don't move)
            bank.n0.index_add_(0, pos, wj.sum(0)[None].expand(B, NB).contiguous())
        for k in bank.orders:
            NP1, PK1 = bank.pc[k].shape[1], bank.st[k].shape[1]
            NB_ = bank.B
            PCf, STf, SLf = bank.pc[k].view(NB_ * NP1, NB), bank.st[k].view(NB_ * PK1, NB), bank.sl[k].view(NB_ * PK1)
            new = [np.asarray(it["new"][k], np.int64) + b * NP1 for b, it in zip(idx, items) if it["new"][k]]
            if new:
                PCf.index_fill_(0, m._dv(np.concatenate(new)), 0.0)
            Us, ts, pps, owners, off = [], [], [], [], 0
            for i, (b, c, it) in enumerate(zip(idx, cores, items)):
                Un = np.unique(it["ids"][k - 1])
                Un = Un[Un >= 0]
                pp, owner = c._node_pairs(k, Un)
                Us.append(Un + b * PK1)
                ts.append(np.full(len(Un), te[i], np.int64))
                if len(pp):
                    pps.append(np.asarray(pp, np.int64) + b * NP1)
                    owners.append(np.asarray(owner, np.int64) + off)
                off += len(Un)
            U = m._dv(np.concatenate(Us))
            T = m._dv(np.concatenate(ts))
            f = keep[None] ** (T - SLf[U]).float()[:, None]                              # (sum |U|, NB)
            if pps:
                PCf[m._dv(np.concatenate(pps))] *= f[m._dv(np.concatenate(owners))]
            STf[U] *= f
            SLf.index_put_((U,), T)
            pj = torch.stack([m._dv(np.asarray(it["pair"][k], np.int64)) for it in items]) + \
                pos[:, None] * NP1                                                        # (B, C)
            PCf.index_put_((pj[:, :, None].expand(B, C, NB).reshape(-1, NB), ib.reshape(-1, NB)), wjB.reshape(-1, NB),
                           accumulate=True)
            ix = torch.stack([it["idt"][k] for it in items]) + pos[:, None] * PK1
            STf.index_put_((ix[:, :, None].expand(B, C, NB).reshape(-1, NB), ib.reshape(-1, NB)), wjB.reshape(-1, NB),
                           accumulate=True)
            bank.pc[k][:, NP1 - 1].zero_()                # every lane's dummy row (idle lanes' are zero already)
            bank.st[k][:, PK1 - 1].zero_()
