"""Every generating lane's next token in one pass (Nov 24; LANES.md stage 2). core3.next_distribution for many lanes at
once: each lane's one-position store read (lanes_read.read_lanes with a tick of one), the learner's cached step for all
lanes as one batch (their caches have different lengths: padded, with a mask), the gate's mix.

Each lane keeps its own generation cache exactly as next_distribution keeps it (m._gen: start, n, kv per block), so a
lane can step alone or in a batch interchangeably. A lane whose cache must be rebuilt (a new window, weights that moved,
a gap) steps alone through next_distribution, which rebuilds it; from then on it batches.

  ps = decode_lanes(cores, contexts)       # [p (V,)] per lane, the same as [c.next_distribution(m) for c, m in ...]
"""
import numpy as np
import torch

from .core2 import Learner3, _rot
from .lanes_read import read_lanes


def _fast(core, m):
    """this lane can take a batched step: its cache is current and one position behind -> (g, ts) or None"""
    net = core.net
    if not (isinstance(net, Learner3) and int(getattr(net, "slide", 0) or 0) and net.rope):
        return None
    g = getattr(m, "_gen", None)
    ts = m.t
    if g is None or g["ver"] != (id(net), core.chunks):
        return None
    if g.get("p_ts") == ts and ts == g["start"] + g["n"]:
        return None                                        # already read: next_distribution returns it at once
    if ts != g["start"] + g["n"] + 1 or not g["kv"] or g["n"] + 1 >= net.rcos.shape[0]:
        return None
    return g, ts


@torch.no_grad()
def decode_lanes(cores, contexts):
    out = [None] * len(cores)
    batch = []
    for i, (c, m) in enumerate(zip(cores, contexts)):
        f = _fast(c, m)
        if f is None:
            out[i] = c.next_distribution(m)                # one lane alone (rebuilds its cache when it must)
        else:
            batch.append((i, c, m) + f)
    if not batch:
        return out
    m0 = batch[0][1]
    net, d = m0.net, m0.dev
    B = len(batch)
    # the store's read of each lane's next position (a tick of one, nothing pushed)
    items = []
    for _, c, m, g, ts in batch:
        ids, rec, mp, mb, win = c._contexts(m, None, False)
        items.append(dict(C=1, t0=c.clock, y=c._dv(np.zeros(1, np.int64)), idt=c._idt(ids), gid=c._gids(ids, rec, 1),
                          maps=c._maps2(ids), mp=mp, mb=mb))
    rd = read_lanes([b[1] for b in batch], items)
    # the learner's cached step, every lane at once
    seq = torch.stack([c._dv(np.asarray(m.R[[(ts - 1) % m.ring]], np.int64)) for _, c, m, g, ts in batch])   # (B, 1)
    pos = torch.tensor([g["n"] for *_, g, ts in batch], device=d)                                          # (B,)
    cuda_bf16 = d.type == "cuda" and int(m0.c.get("learner_bf16", 0))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(cuda_bf16)):
        feat = rd["feat"]
        inj = net.cell(net.ln(feat.reshape(B, 1, -1)))
        if net.fmlp is not None:
            inj = inj + net.fmlp(inj)
        x = net.emb(seq) + inj
        cos, sin = net.rcos[pos][:, None, None, :], net.rsin[pos][:, None, None, :]          # (B, 1, 1, d/2)
        win = int(net.slide)
        for li, blk in enumerate(net.blocks):
            if li and net.inject.startswith("every"):
                x = x + inj
            H = x.shape[-1]
            q, k, v = blk.qkv(blk.l1(x)).view(B, 1, 3, blk.heads, H // blk.heads).permute(2, 0, 3, 1, 4)   # (B, h, 1, dh)
            q, k = _rot(q, cos, sin), _rot(k, cos, sin)
            if blk.half:                                                      # as forward_kv (att_bf16)
                q, k = q.to(v.dtype), k.to(v.dtype)
            lens, Ks, Vs = [], [], []
            for b, (_, c, m, g, ts) in enumerate(batch):                     # each lane's cache grows by its new key
                kv = g["kv"][li]
                kv[0] = torch.cat([kv[0].to(k.dtype), k[b:b + 1]], 2)
                kv[1] = torch.cat([kv[1].to(v.dtype), v[b:b + 1]], 2)
                if win and kv[0].shape[2] > win:
                    kv[0], kv[1] = kv[0][:, :, -win:], kv[1][:, :, -win:]
                Ks.append(kv[0]); Vs.append(kv[1]); lens.append(kv[0].shape[2])
            L = max(lens)
            if min(lens) == L:
                Kb, Vb, mask = torch.cat(Ks), torch.cat(Vs), None
            else:
                Kb = torch.zeros(B, blk.heads, L, k.shape[-1], device=d, dtype=Ks[0].dtype)
                Vb = torch.zeros_like(Kb)
                for b, (kk, vv) in enumerate(zip(Ks, Vs)):
                    Kb[b, :, :kk.shape[2]], Vb[b, :, :vv.shape[2]] = kk[0], vv[0]
                mask = (torch.arange(L, device=d)[None, :] < torch.tensor(lens, device=d)[:, None])[:, None, None, :]
            a = torch.nn.functional.scaled_dot_product_attention(q, Kb, Vb, attn_mask=mask)
            x = x + blk.o(a.transpose(1, 2).reshape(B, 1, H))
            x = x + blk.f2(torch.nn.functional.gelu(blk.f1(blk.l2(x))))
        z = net.out(net.ln_f(x[:, 0]))
    lpL = torch.log_softmax(z.float(), -1)                                   # (B, V)
    lpE = rd["lpE"][:, 0]
    sg = torch.sigmoid(m0.theta[rd["key"][:, 0]])
    p = sg[:, None] * lpE.exp() + (1 - sg)[:, None] * lpL.exp()
    for b, (i, c, m, g, ts) in enumerate(batch):
        g["n"] += 1
        g["p"], g["p_ts"] = p[b], ts
        out[i] = p[b]
    return out
