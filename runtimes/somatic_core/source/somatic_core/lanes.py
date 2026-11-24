"""Far-apart lanes (Nov 18; James: "batching with lanes is just about throughput"; spec and verdict in lab/lanes_probe.py).

One causal stream of N tokens, cut into B consecutive stretches of P = N / B. Lane b reads stretch b, from token bP, with
the store exactly as the single stream has it at bP. All lanes share one learner, its optimiser, the gate and the exact
mixer (mix_shared >= 1: the mixer is the learner's, so the store is counts alone); each lane owns its store, its match
history and its cached window. A round is one tick of T tokens from every lane, then one update of the shared learner on
the mean of their gradients: B * T tokens per update, B far-apart pieces of the one stream.

Seeding: the store is a function of the stream alone (checked Nov 18), so lane b's store at bP comes from a counting pass
(Core3.count: counts only, no learner) over [0, bP). One counter walks the stream and hands each lane its store as it
passes; it ends as the last lane, so B stores exist in all.

The model this leaves is the learner with the last lane's store and history (export): the stream's store at N, exactly
the store a single pass would hold, and a learner that read every token of it once.

  lanes = Lanes(config, B, device, blen)
  lanes.seed(ids)                  # the counting pass; P = len(ids) // B, rounded down to whole ticks
  while lanes.round(ids): ...      # one tick per lane and one shared update; False when the stretches are read
  state = lanes.export()           # a Core3 state_dict: learner + the stream's store at the end
"""
import contextlib
import math
import copy
import dataclasses

import numpy as np
import torch

from .core import Matcher
from .core3 import Core3


def _lite(cfg):
    """a lane's own core: the same store, a token learner it never uses (it reads the shared one)"""
    c = dataclasses.replace(cfg, hidden=16, depth=1, att_heads=2, proj=min(cfg.proj, 16))
    return c


class Lanes:
    def __init__(self, cfg, B: int, device, blen: np.ndarray):
        assert int(cfg.mix_shared) >= 1, "lanes share the mixer: mix_shared 1 or 2 (the store must be counts alone)"
        assert cfg.learn_mode == "accumulate" and cfg.learn_stride == B, "lanes: learn_mode accumulate, learn_stride = B"
        assert not cfg.io2 and cfg.rate_ctl == "fixed" and not cfg.rehearse
        self.cfg, self.B, self.dev, self.blen = cfg, int(B), device, blen
        self.T = int(cfg.chunk)
        self.master = Core3(cfg.public_dict(), device, blen)          # lane 0: the learner, optimiser, gate, mixer
        self.lanes = [self.master]
        for _ in range(1, self.B):
            ln = Core3(_lite(cfg).public_dict(), device, blen)
            self._share(ln)
            self.lanes.append(ln)
        for ln in self.lanes:
            ln._ext = True
        self.ms = [Matcher(cfg.ring, cfg.minl) for _ in range(self.B)]
        self.P = 0
        self.r = 0                                                     # rounds done = shared updates done

    def _share(self, ln):
        m = self.master
        ln.net, ln.opt, ln.theta = m.net, m.opt, m.theta
        ln.wE, ln.wM, ln.mopt = m.wE, m.wM, m.mopt
        ln.c = dict(m.c)                                               # the full learner's config (ctx, cache, ...)
        if hasattr(m, "_vecs"):
            ln._vecs = m._vecs
        ln._kv, ln._kv_end = None, -1

    # ---- seeding: one counting pass over [0, (B - 1) P) ----
    def seed(self, ids, report=None, device=None, threads=8):
        """device: where the counter runs. The store is counts alone, so a CPU core can count for GPU lanes (on the H100
        pod the CPU counted ~1.7x faster than the launch-bound GPU, Nov 18); its stores are handed over by value."""
        T, B = self.T, self.B
        N = len(ids) - len(ids) % (B * T)
        self.P = P = N // B
        if device is not None and torch.device(device) != self.master.dev and B > 1:
            torch.set_num_threads(int(threads))
            cnt = Core3(_lite(self.cfg).public_dict(), device, self.blen)
            mc = Matcher(self.cfg.ring, self.cfg.minl)
        else:
            cnt = self.lanes[-1]                                       # the counter ends as the last lane
            mc = self.ms[-1]
        c0 = 0                                                         # tokens counted (the core's clock rebases)
        for b in range(1, B):
            while c0 < b * P:
                cnt.count(mc, ids[c0:c0 + T])
                c0 += T
                if report and (c0 // T) % 4096 == 0:
                    report(c0, b)
            if b < B - 1 or cnt is not self.lanes[-1]:
                self.lanes[b].load_store(cnt.store_state())
                self.ms[b] = Matcher.from_state(copy.deepcopy(mc.state()))
        return P

    # ---- training ----
    cast_once = True                                       # cuda + learner_bf16: cast the weights once per round

    def round(self, ids) -> bool:
        T, P = self.T, self.P
        if (self.r + 1) * T > P:
            return False
        # autocast keeps its bf16 copies of the weights until the outermost autocast region exits; a disabled region
        # around the round keeps them across the lanes' forwards (the weights only change in apply_update), while the
        # store's own maths stays outside autocast. The same casts, made once: identical values (tests on the pod).
        dev = self.master.dev
        keep = self.cast_once and dev.type == "cuda" and int(self.master.c.get("learner_bf16", 0))
        with torch.autocast(device_type=dev.type, enabled=False) if keep else contextlib.nullcontext():
            if int(self.master.c.get("lanes_batch", 0)):
                self._round_batched(ids)
            else:
                for b, ln in enumerate(self.lanes):
                    c0 = b * P + self.r * T
                    ln.step(self.ms[b], ids[c0:c0 + T])
        self.master.apply_update(self.r, self.B)
        self.r += 1
        return True

    def _round_batched(self, ids):
        """lanes_batch (Nov 24): every lane's tick through batched_ticks"""
        T, P = self.T, self.P
        batched_ticks(self, self.master, self.lanes, self.ms,
                      [ids[b * P + self.r * T: b * P + self.r * T + T] for b in range(self.B)])

    def bits(self):
        """the round's per-token bits, lane by lane (B, T)"""
        return torch.stack([ln.last["bits"].detach() for ln in self.lanes])

    # ---- the model it leaves ----
    def export(self) -> dict:
        """a Core3 state_dict: the shared learner, gate and mixer with the last lane's store, history and window"""
        s = self.master.state_dict()
        last = self.lanes[-1]
        s.update(last.store_state())
        kv = getattr(last, "_kv", None)                                # its window, as training built it (with store
        s["kv"] = None if kv is None else dict(k=[k.cpu() for k, _ in kv], v=[v.cpu() for _, v in kv],   # features;
                                               end=int(last._kv_end))  # generation reads it, Nov 18)
        s["config"] = dict(self.master.c)
        return s

    def export_matcher(self) -> Matcher:
        return Matcher.from_state(copy.deepcopy(self.ms[-1].state()))

    # ---- resume ----
    def state_dict(self) -> dict:
        return dict(engine="lanes", B=self.B, P=self.P, r=self.r, master=self.master.state_dict(),
                    stores=[ln.store_state() for ln in self.lanes[1:]],
                    kv=[None if getattr(ln, "_kv", None) is None else
                        dict(k=[k.cpu() for k, _ in ln._kv], v=[v.cpu() for _, v in ln._kv], end=int(ln._kv_end))
                        for ln in self.lanes[1:]],
                    matchers=[m.state() for m in self.ms])

    def load_state_dict(self, s: dict):
        assert s["engine"] == "lanes" and s["B"] == self.B
        self.P, self.r = int(s["P"]), int(s["r"])
        self.master.load_state_dict(s["master"])
        for ln, st, kv in zip(self.lanes[1:], s["stores"], s["kv"]):
            ln.load_store(st)
            if kv is not None:
                ln._kv = [(k.to(self.master.dev), v.to(self.master.dev)) for k, v in zip(kv["k"], kv["v"])]
                ln._kv_end = int(kv["end"])
        self.ms = [Matcher.from_state(m) for m in s["matchers"]]


def batched_ticks(holder, m, lanes, ms, ticks, stride=None):
    """lanes_batch (Nov 24): every lane's store read, then all lanes' learner as one (B, C) pass and one backward, then
    each lane's scoring and store update, in lane order (core3.lane_pre / lane_post_a / lane_post_b). The ticks must be
    the same length. holder keeps the batched attention window between calls (Lanes, or the lane engine). stride: the
    lanes' gradients are summed / stride (default: the config's learn_stride; the engine passes the lanes that learned)."""
    T = len(ticks[0])
    B = len(lanes)
    joint = bool(int(m.c.get("lanes_read", 0)))
    pres = [ln.lane_pre(mt, tk, read=not joint) for ln, mt, tk in zip(lanes, ms, ticks)]
    if stride is not None:
        for p in pres:
            p["s_"] = stride
    feat_all = None
    lpE_all = None
    if joint:                                              # every lane's store read in one pass
        from .lanes_read import read_lanes
        rd = read_lanes(lanes, pres)
        for b, p in enumerate(pres):
            p.update(lpE=rd["lpE"][b], key=rd["key"][b], deep=rd["deep"][b], feat=None)
        feat_all, lpE_all, key_all = rd["feat"], rd["lpE"], rd["key"]
        del rd
    caches = [p["cache"] for p in pres]
    done = False
    if all(c is not None for c in caches) and len({c[0][0].shape[2] for c in caches}) == 1:
        views = getattr(holder, "_kv_views", None)
        if views is not None and len(views) == B and all(c is v for c, v in zip(caches, views)):
            cache = holder._kvb                            # last round's batched window, as the lanes left it
        else:
            nb = len(caches[0])
            cache = [(torch.cat([c[i][0] for c in caches]), torch.cat([c[i][1] for c in caches])) for i in range(nb)]
        z, out = m._learner_lanes(torch.stack([p["seq"] for p in pres]),
                                  feat_all if joint else torch.stack([p["feat"] for p in pres]), cache)
        outs = [(z[b], None) for b in range(B)]
        keep = m.net.ctx - (1 if m.net.slide else T)       # every lane's window, kept batched: one cat per block
        holder._kvb = [(torch.cat([a[0], b[0]], 2)[:, :, -keep:], torch.cat([a[1], b[1]], 2)[:, :, -keep:])
                       for a, b in zip(cache, out)] if keep > 0 else None
        holder._kv_views = []
        for b, (ln, p) in enumerate(zip(lanes, pres)):
            ln._kv = [(k[b:b + 1], v[b:b + 1]) for k, v in holder._kvb] if keep > 0 else None
            ln._kv_end = p["m"].t
            holder._kv_views.append(ln._kv)
        done = True
        del cache, out
    else:                                                  # windows of different lengths (new chats): batched by length
        outs = [None] * B
        groups = {}
        for b, c in enumerate(caches):
            groups.setdefault(None if c is None else c[0][0].shape[2], []).append(b)
        for length, idx in groups.items():
            seq = torch.stack([pres[b]["seq"] for b in idx])
            feat = feat_all[idx] if joint else torch.stack([pres[b]["feat"] for b in idx])
            cache = None if length is None else [(torch.cat([caches[b][i][0] for b in idx]), torch.cat([caches[b][i][1] for b in idx]))
                                                 for i in range(len(caches[idx[0]]))]
            zg, out = m._learner_lanes(seq, feat, cache)
            for j, b in enumerate(idx):
                outs[b] = (zg[j], [(k[j:j + 1], v[j:j + 1]) for k, v in out])
            del zg, out, cache
        z = None
        holder._kv_views = None
    if joint and done:                                     # every lane scored at once; the gate read as the round began
        loss = _score_lanes(m, lanes, pres, z, lpE_all, key_all)
    else:
        loss = torch.stack([ln.lane_post_a(p, torch.log_softmax(zb, -1), out, cache_done=done)
                            for ln, p, (zb, out) in zip(lanes, pres, outs)]).sum()
    del outs, z
    loss.backward()
    if joint:                                              # the mixer's gradient, all lanes at once (one graph)
        gE = torch.autograd.grad(torch.stack([p["lossE"] for p in pres]).sum(), [m.wE, m.wM])
        with torch.no_grad():
            for w, g in zip((m.wE, m.wM), gE):
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) / pres[0]["s_"]
                w.grad = g if w.grad is None else w.grad + g
    if not int(m.c.get("lanes_store", 0)):
        return [ln.lane_post_b(p, mixer=not joint) for ln, p in zip(lanes, pres)]
    from .lanes_store import StoreBank, store_update_lanes   # every lane's store update in one pass
    bank = getattr(holder, "_bank", None)
    if bank is None or not all(any(ln is c for c in bank.cores) for ln in lanes) or not bank.bound():
        bank = holder._bank = StoreBank(getattr(holder, "bank_cores", None) or lanes)
    bits = [ln.lane_post_b(p, mixer=not joint, store=False) for ln, p in zip(lanes, pres)]
    where = [next(i for i, c in enumerate(bank.cores) if c is ln) for ln in lanes]
    store_update_lanes(bank, pres, where)
    for ln, p in zip(lanes, pres):
        ln.lane_clock(p)
    return bits


def _score_lanes(m, lanes, pres, z, lpE, key):
    """core3.lane_post_a for every lane at once (lanes_read): the lanes are simultaneous, so each reads the gate as the
    round began and the gate's step is one accumulated write (one lane: identical to lane_post_a). -> the learner's loss"""
    lpL = torch.log_softmax(z, -1)                                               # (B, C, V)
    y = torch.stack([p["y"] for p in pres])
    lyE = lpE.gather(2, y[..., None])[..., 0]
    lyL = lpL.gather(2, y[..., None])[..., 0]
    ln2 = math.log(2)
    with torch.no_grad():
        sg = torch.sigmoid(m.theta[key])
        a_, b_ = lyE.detach().exp(), lyL.detach().exp()
        pf = sg * a_ + (1 - sg) * b_
        last = sg[:, -1, None] * lpE[:, -1].detach().exp() + (1 - sg[:, -1, None]) * lpL[:, -1].detach().exp()
        full_arg = (sg[..., None] * lpE.detach().exp() + (1 - sg)[..., None] * lpL.detach().exp()).argmax(-1)
        sums = torch.stack([-lyE.detach().sum(1), -lyL.detach().sum(1), -torch.log(pf).sum(1)], 1) / ln2   # (B, 3)
        hits = (full_arg == y).float().sum(1)
        m.theta.index_put_((key.reshape(-1),), (float(m.c["gate_lr"]) * sg * (1 - sg) * (a_ - b_) / pf).reshape(-1),
                           accumulate=True)
        top = last.topk(5)
        bits, exact, learner = -torch.log2(pf), -lyE.detach() / ln2, -lyL.detach() / ln2
    for b, (ln, p) in enumerate(zip(lanes, pres)):
        ln.acc[:3] += sums[b]
        ln.acc[4] += hits[b]
        ln.acc[5] += p["C"]
        ln.last = dict(tokens=p["ys"], bits=bits[b], exact=exact[b], next=(top.values[b], top.indices[b]),
                       learner=learner[b], deep=p["deep"])
        p["lossE"], p["pf"] = -lyE[b].mean(), pf[b]
        p["feat"] = p["lpE"] = None
    return -(lyL.mean(1) / torch.tensor([p["s_"] for p in pres], device=lyL.device)).sum()
