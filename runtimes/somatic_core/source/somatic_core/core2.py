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
import torch.utils.checkpoint
try:
    from . import flash_mps                                                 # 1.2.1: Metal flash attention (MPS)
except ImportError:
    import flash_mps

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


def _rot(x, cos, sin):
    """rotary positions (att_rope): x (B, h, T, d) turned by its position's angles; cos/sin (T, d/2)."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def _rope_table(n, d, device):
    inv = 1.0 / (10000.0 ** (torch.arange(0, d // 2, device=device).float() * 2 / d))
    a = torch.arange(n, device=device).float()[:, None] * inv[None]
    return a.cos(), a.sin()


_FLEX = {}


def _flex_attn(q, k, v, Lc, win):
    """the sliding band of forward_cache (query i at Lc + i reads keys j with i + Lc - win < j <= i + Lc) through
    FlexAttention (Nov 24, throughput): the same mask, a fused kernel that skips the blocks outside the band instead of
    an explicit (C, T) mask in the memory-efficient kernel. Compiled once per shape."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    C, T = q.shape[2], k.shape[2]
    key = (C, T, Lc, win, q.device)
    if key not in _FLEX:
        def band(b, h, qi, kj):
            qa = qi + Lc
            return (kj <= qa) & (kj > qa - win) if win else kj <= qa
        _FLEX[key] = create_block_mask(band, None, None, C, T, device=q.device)
        if "fn" not in _FLEX:
            # dynamic=None: a shape that changes (lanes come and go, a window grows) is compiled once as dynamic, not
            # once per value until torch's recompile limit drops it to the slow eager kernel (Nov 24, lanes on soma-1b)
            _FLEX["fn"] = torch.compile(flex_attention, dynamic=None)
            torch._dynamo.config.cache_size_limit = max(int(torch._dynamo.config.cache_size_limit), 64)
    return _FLEX["fn"](q, k, v, block_mask=_FLEX[key])


class AttnBlock(torch.nn.Module):
    """a pre-LayerNorm causal self-attention block with a GELU MLP (GPT-2 form)."""
    def __init__(self, H, heads, L):
        super().__init__()
        self.heads = heads
        self.flash = False                                                   # set by the core (att_kernel)
        self.flex = False                                                    # cuda: the band through FlexAttention
        # att_bf16 (Nov 24, throughput): rotary is computed in fp32 (its tables are fp32), which made attention itself
        # run in fp32 under autocast; with this set, q/k/v go back to the projections' dtype after rotation
        self.half = False
        self.l1 = torch.nn.LayerNorm(H); self.qkv = torch.nn.Linear(H, 3 * H); self.o = torch.nn.Linear(H, H)
        self.l2 = torch.nn.LayerNorm(H); self.f1 = torch.nn.Linear(H, 4 * H); self.f2 = torch.nn.Linear(4 * H, H)
        for m, sd in ((self.qkv, 0.02), (self.f1, 0.02), (self.o, 0.02 / (2 * L) ** 0.5), (self.f2, 0.02 / (2 * L) ** 0.5)):
            torch.nn.init.normal_(m.weight, 0.0, sd); torch.nn.init.zeros_(m.bias)

    def forward(self, x, cos=None, sin=None):                                # (1, T, H)
        B, T, H = x.shape
        q, k, v = self.qkv(self.l1(x)).view(B, T, 3, self.heads, H // self.heads).permute(2, 0, 3, 1, 4)
        if cos is not None:
            q, k = _rot(q, cos[:T], sin[:T]), _rot(k, cos[:T], sin[:T])
            if self.half:
                q, k = q.to(v.dtype), k.to(v.dtype)
        if self.flash and flash_mps.supported(q):                          # 1.2.1: no T x T is ever stored
            a = flash_mps.FlashAttn.apply(q, k, v).transpose(1, 2).reshape(B, T, H)
        else:
            a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, H)
        x = x + self.o(a)
        return x + self.f2(torch.nn.functional.gelu(self.f1(self.l2(x))))

    def forward_cache(self, x, ck, cv, cos, sin, win=0, whole=False):
        """att_cache (Nov 18; James: "the store/count/window all observe the same now"): x holds only this tick's
        positions; ck/cv are the earlier positions' keys (un-rotated) and values, computed once, at their own tick, and
        read here without recomputation or gradient. Returns x and this tick's own keys/values for the cache."""
        B, C, H = x.shape
        q, k, v = self.qkv(self.l1(x)).view(B, C, 3, self.heads, H // self.heads).permute(2, 0, 3, 1, 4)
        Lc = 0 if ck is None else ck.shape[2]
        T = Lc + C
        ka = k if ck is None else torch.cat([ck.to(k.dtype), k], 2)                 # a prefilled cache may be fp32 under bf16
        va = v if cv is None else torch.cat([cv.to(v.dtype), v], 2)
        qr, kr = _rot(q, cos[Lc:T], sin[Lc:T]), _rot(ka, cos[:T], sin[:T])        # relative: only distances matter
        if self.half:
            qr, kr = qr.to(v.dtype), kr.to(v.dtype)
        if Lc == 0 and not (win and C > win):
            a = torch.nn.functional.scaled_dot_product_attention(qr, kr, va, is_causal=True)
        elif self.flex and x.is_cuda:
            a = _flex_attn(qr, kr, va.to(qr.dtype), Lc, win)
        else:                                                                 # query i sees the past and new j <= i
            qi = (Lc + torch.arange(C, device=x.device))[:, None]
            kj = torch.arange(T, device=x.device)[None, :]
            m = (kj <= qi) & (kj > qi - win) if win else kj <= qi              # win: exactly win positions per query
            a = torch.nn.functional.scaled_dot_product_attention(qr, kr, va, attn_mask=m)
        x = x + self.o(a.transpose(1, 2).reshape(B, C, H))
        if whole:                                          # Nov 28: the joined window, so lanes needn't join it again
            return x + self.f2(torch.nn.functional.gelu(self.f1(self.l2(x)))), ka, va
        return x + self.f2(torch.nn.functional.gelu(self.f1(self.l2(x)))), k, v

    def forward_kv(self, x, kv, cos=None, sin=None, n=0, win=0):
        """generation (1.2.1): x holds new positions only; kv = [K, V] of the earlier ones (grown in place). win (slide):
        the cache keeps the last win positions, so each new query reads exactly win, itself included."""
        B, T, H = x.shape
        q, k, v = self.qkv(self.l1(x)).view(B, T, 3, self.heads, H // self.heads).permute(2, 0, 3, 1, 4)
        if cos is not None:                                                   # rotary: cached keys keep their angles
            q, k = _rot(q, cos[n:n + T], sin[n:n + T]), _rot(k, cos[n:n + T], sin[n:n + T])
            if self.half:                                                     # att_bf16: the cache in bf16 too (Nov 24)
                q, k = q.to(v.dtype), k.to(v.dtype)
        if kv[0] is None:
            kv[0], kv[1] = k, v
            a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            kv[0], kv[1] = torch.cat([kv[0].to(k.dtype), k], 2), torch.cat([kv[1].to(v.dtype), v], 2)
            assert T == 1, "one new position at a time"
            if win and kv[0].shape[2] > win:                                  # rotated keys: trimming keeps distances
                kv[0], kv[1] = kv[0][:, :, -win:], kv[1][:, :, -win:]
            a = torch.nn.functional.scaled_dot_product_attention(q, kv[0], kv[1])
        x = x + self.o(a.transpose(1, 2).reshape(B, T, H))
        return x + self.f2(torch.nn.functional.gelu(self.f1(self.l2(x))))


class Learner3(torch.nn.Module):
    """the attention learner (dev, Nov 10; James: "what if the learner just had more exact tokens … make the present
    a richer picture"). A causal transformer over the last `ctx` exact tokens of the stream (read from the matcher's
    ring: nothing new is stored), with the store's features for each new position added to its input. The core sets
    `seq` before each call: the earlier tokens, then the inputs of the C positions being predicted."""
    def __init__(self, V, DP, O, NB, H, L, heads, ctx, inject="input", past=0, rope=0):
        super().__init__()
        # rope (Nov 18): rotary positions instead of a learned absolute table, so a key keeps its meaning as the window
        # slides; required by the cached past (att_cache), where earlier keys are not recomputed
        self.rope = bool(rope)
        # past=1 (dev, Nov 11; James: does attention run over every layer?): the window's earlier positions carry the
        # store reads made when they were predicted (core3 keeps them, bounded to ctx - C rows), so every layer sees
        # the store at every position, not only at the tick's own. Honest: each read is from its own time.
        self.past_on, self.past_use, self.feat_now = bool(past), None, None
        self.inject = inject                                                   # "input" | "every" (each block's input)
        self.emb = torch.nn.Embedding(V, H); torch.nn.init.normal_(self.emb.weight, 0.0, 0.02)
        if self.rope:
            self.pos = None
            cos, sin = _rope_table(2 * ctx, H // heads, "cpu")
            self.register_buffer("rcos", cos, persistent=False); self.register_buffer("rsin", sin, persistent=False)
        else:
            self.pos = torch.nn.Embedding(ctx, H); torch.nn.init.normal_(self.pos.weight, 0.0, 0.02)
        # the cached past (att_cache): the core sets cache_in = [(K, V)] per block (or None) before a tick; the forward
        # leaves this tick's own (K, V) in cache_out
        self.cache_on, self.cache_in, self.cache_out = False, None, None
        self.slide = 0             # att_cache=2: every position reads exactly ctx positions (itself included), any tick length
        self.P = torch.nn.Parameter(torch.randn(V, DP) * 0.5)
        self.ln = torch.nn.LayerNorm(O * NB * (DP + 1))
        self.cell = torch.nn.Linear(O * NB * (DP + 1), H, bias=False)
        # "every+mlp": the store's features through a small dense network first, as the current learner treats them
        self.fmlp = torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(H, H), torch.nn.GELU(), torch.nn.Linear(H, H)) \
            if inject.endswith("+mlp") else None
        self.blocks = torch.nn.ModuleList([AttnBlock(H, heads, L) for _ in range(L)])
        self.ln_f = torch.nn.LayerNorm(H)
        self.out = torch.nn.Linear(H, V)
        self.keep_h, self.h, self.seq, self.ctx = False, None, None, ctx
        # gen (1.2.1): a generation cache, set by the core around a no-grad read. dict(n=positions cached, kv=[[K, V]] per
        # block). n == 0: seq is a whole window (prefill); n > 0: seq is the one new token, at position n.
        self.gen = None
        # ckpt (Nov 15, memory): MPS materialises each block's T x T attention (~117 MB a block at 768 x 12, T 1024), so
        # the 768 learner can't learn live on an 8 GB Mac. Recomputing blocks in backward holds one block's at a time.
        # Same math; costs one more forward of the blocks.
        self.ckpt = False

    def forward(self, win, feat):
        if self.gen is not None and not torch.is_grad_enabled():
            return self._forward_gen(feat)
        if self.cache_on:
            return self._forward_cached(feat)
        C, T = feat.shape[0], self.seq.shape[0]
        x = self.emb(self.seq)[None] if self.rope else \
            (self.emb(self.seq) + self.pos(torch.arange(T, device=self.seq.device)))[None]     # (1, T, H)
        rc = (self.rcos, self.rsin) if self.rope else ()
        f = feat.reshape(C, -1)
        if self.past_on:                                                      # the earlier positions' own reads
            self.feat_now = f.detach()
            pu = self.past_use
            pu = f.new_zeros(T - C, f.shape[1]) if pu is None else pu
            if pu.shape[0] < T - C:
                pu = torch.cat([f.new_zeros(T - C - pu.shape[0], f.shape[1]), pu], 0)
            f, C = torch.cat([pu[pu.shape[0] - (T - C):], f], 0), T
        inj = self.cell(self.ln(f))                                                      # (C, H): the store
        if self.fmlp is not None:
            inj = inj + self.fmlp(inj)
        x = torch.cat([x[:, :T - C], x[:, T - C:] + inj[None]], 1)
        ck = self.ckpt and torch.is_grad_enabled()                            # att_ckpt: recompute each block in backward
        for i, b in enumerate(self.blocks):
            if i and self.inject.startswith("every"):                         # the store again at every block
                x = torch.cat([x[:, :T - C], x[:, T - C:] + inj[None]], 1)
            x = torch.utils.checkpoint.checkpoint(b, x, *rc, use_reentrant=False) if ck else b(x, *rc)
        C = feat.shape[0]
        h = self.ln_f(x[0, T - C:])
        if self.keep_h and h.requires_grad:
            h.retain_grad()
            self.h = h
        return self.out(h)

    def _forward_cached(self, feat):
        """seq holds this tick's C inputs only; the earlier positions come from cache_in."""
        C = feat.shape[0]
        need = C + (self.cache_in[0][0].shape[2] if self.cache_in else 0)
        if need > self.rcos.shape[0]:
            self.rcos, self.rsin = _rope_table(2 * need, self.rcos.shape[1] * 2, self.rcos.device)
        assert self.seq.shape[0] == C and self.rope and not self.past_on
        inj = self.cell(self.ln(feat.reshape(C, -1)))
        if self.fmlp is not None:
            inj = inj + self.fmlp(inj)
        x = self.emb(self.seq)[None] + inj[None]
        ck = self.ckpt and torch.is_grad_enabled()
        out = []
        for i, b in enumerate(self.blocks):
            if i and self.inject.startswith("every"):
                x = x + inj[None]
            kv = self.cache_in[i] if self.cache_in else (None, None)
            if ck:
                x, k, v = torch.utils.checkpoint.checkpoint(b.forward_cache, x, kv[0], kv[1], self.rcos, self.rsin,
                                                            self.slide, use_reentrant=False)
            else:
                x, k, v = b.forward_cache(x, kv[0], kv[1], self.rcos, self.rsin, self.slide)
            out.append((k.detach(), v.detach()))
        self.cache_out = out
        h = self.ln_f(x[0])
        if self.keep_h and h.requires_grad:
            h.retain_grad()
            self.h = h
        return self.out(h)

    def forward_lanes(self, seq, feat, cache, whole=False):
        """B lanes' ticks in one pass (Nov 24, throughput; lanes.py): seq (B, C) inputs, feat (B, C, ...) store features,
        cache = [(K, V)] per block with K (B, heads, Lk, d), or None. The same maths as _forward_cached for each lane
        (one lane: identical); the weights don't change within a round, so the lanes' graphs are one graph.
        -> logits (B, C, V) and this tick's [(K, V)] per block."""
        B, C = seq.shape
        need = C + (cache[0][0].shape[2] if cache else 0)
        if need > self.rcos.shape[0]:
            self.rcos, self.rsin = _rope_table(2 * need, self.rcos.shape[1] * 2, self.rcos.device)
        assert self.rope and not self.past_on
        inj = self.cell(self.ln(feat.reshape(B, C, -1)))
        if self.fmlp is not None:
            inj = inj + self.fmlp(inj)
        x = self.emb(seq) + inj
        ck = self.ckpt and torch.is_grad_enabled()
        out = []
        for i, b in enumerate(self.blocks):
            if i and self.inject.startswith("every"):
                x = x + inj
            kv = cache[i] if cache else (None, None)
            if ck:
                x, k, v = torch.utils.checkpoint.checkpoint(b.forward_cache, x, kv[0], kv[1], self.rcos, self.rsin,
                                                            self.slide, whole, use_reentrant=False)
            else:
                x, k, v = b.forward_cache(x, kv[0], kv[1], self.rcos, self.rsin, self.slide, whole)
            out.append((k.detach(), v.detach()))
        return self.out(self.ln_f(x)), out

    @torch.no_grad()
    def prefill(self, seq):
        """the cached past rebuilt from tokens alone (a resumed or broken stream): one causal pass with today's weights
        and no store reads, as the uncached window treats its earlier positions."""
        x = self.emb(seq)[None]
        out = []
        for b in self.blocks:
            x, k, v = b.forward_cache(x, None, None, self.rcos, self.rsin)
            out.append((k, v))
        return out


def _gen_forward(self, feat):
    """Learner3 with a KV cache: the same window as a full forward, positions computed once (exact up to float order)."""
    g = self.gen
    assert not self.past_on, "generation cache: att_past is not supported"
    C, n, Tn = feat.shape[0], g["n"], self.seq.shape[0]
    x = self.emb(self.seq)[None] if self.rope else \
        (self.emb(self.seq) + self.pos(torch.arange(n, n + Tn, device=self.seq.device)))[None]
    rc = dict(cos=self.rcos, sin=self.rsin, n=n) if self.rope else {}
    inj = self.cell(self.ln(feat.reshape(C, -1)))
    if self.fmlp is not None:
        inj = inj + self.fmlp(inj)
    x = torch.cat([x[:, :Tn - C], x[:, Tn - C:] + inj[None]], 1)
    if not g["kv"]:
        g["kv"] = [[None, None] for _ in self.blocks]
    for i, b in enumerate(self.blocks):
        if i and self.inject.startswith("every"):
            x = torch.cat([x[:, :Tn - C], x[:, Tn - C:] + inj[None]], 1)
        x = b.forward_kv(x, g["kv"][i], win=self.slide, **rc)
    g["n"] = n + Tn
    return self.out(self.ln_f(x[0, Tn - C:]))


Learner3._forward_gen = _gen_forward


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
        if self.dev.type == "cuda":                       # Nov 24: the trainer never set this (only the app did), so the
            torch.backends.cuda.matmul.allow_tf32 = bool(int(c.get("tf32", 1)))   # 1B run's fp32 matmuls ran without TF32
            torch.backends.cudnn.allow_tf32 = bool(int(c.get("tf32", 1)))
        if str(c.get("learner", "mlp")) == "attn":                         # dev: hidden is its width, depth its layers
            self.net = Learner3(self.V, int(c["proj"]), self.O, self.NB, int(c["hidden"]), int(c.get("depth", 1)),
                                int(c.get("att_heads", 6)), int(c.get("att_ctx", 1024)),
                                inject=str(c.get("att_inject", "input")), past=int(c.get("att_past", 0)),
                                rope=int(c.get("att_rope", 0))).to(self.dev)
            self.net.cache_on = bool(int(c.get("att_cache", 0)))
            self.net.slide = int(c.get("att_ctx", 1024)) if int(c.get("att_cache", 0)) == 2 else 0
            assert not self.net.cache_on or (self.net.rope and not self.net.past_on), "att_cache needs att_rope, not att_past"
            ck = int(c.get("att_ckpt", 0))                                 # -1 auto: on for mps (8 GB Macs), off elsewhere
            self.net.ckpt = bool(ck) if ck >= 0 else self.dev.type == "mps"
            fk = int(c.get("att_kernel", 0))                               # -1 auto: the Metal flash kernel on mps
            for b in self.net.blocks:
                b.flash = (bool(fk) if fk >= 0 else self.dev.type == "mps") and self.dev.type == "mps"
                b.flex = bool(int(c.get("att_flex", 0))) and self.dev.type == "cuda"
                b.half = bool(int(c.get("att_bf16", 0)))
            if int(c.get("att_compile", 0)) and self.dev.type == "cuda":
                for b in self.net.blocks:
                    b.forward_cache = torch.compile(b.forward_cache, dynamic=False)
        else:
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
        torch.mps.synchronize() if self.dev.type == "mps" else (torch.cuda.synchronize() if self.dev.type == "cuda" else None)
        now = _time.perf_counter()
        if self.dev.type == "cuda" and getattr(self, "_prof_mem", None) is not None:   # peak memory per section (GB)
            self._prof_mem[name] = max(self._prof_mem.get(name, 0.0), torch.cuda.max_memory_allocated() / 2**30)
            torch.cuda.reset_peak_memory_stats()
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
    def _read2(self, idt, gid, maps, ys, mp, mb, win, t0, grad, rows=None, defer=False):
        """-> zE inputs: lpE, lpL (C, V), gate key, deep; plus nothing else (no lch)."""
        C, V, NB, d = win.shape[0], self.V, self.NB, self.dev
        keep = 1.0 - self.ab                                             # (NB,)
        pos = torch.arange(C, device=d)
        y = self._dv(ys)
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
            toks, vals, Ns, sidxs = [], [], [], []
            for k in self.orders:
                idc = idt[k]
                M = maps[k]                                                              # (C, K+1) tokens
                # ages from integer clocks first: float32 stamps near 2^30 are 64 tokens apart (the fast rungs' whole
                # range), so subtracting after the cast rounded ages away (assessment, Nov 19; fixed in 1.4.0)
                fk = keep[None, :] ** ((t0 - self.sl[k][idc]).float() + pos.float())[:, None]
                R = self.sc[k][idc] * fk[..., None]                                      # (C, NB, K+1)
                g = self._dv(gid[k])
                same = ((g[:, None] == g[None, :]) & (g[:, None] >= 0)).float()          # (C, C) [i, j]
                Wk = W0 * same[:, None, :]                                               # (C, NB, C)
                N = self.st[k][idc] * fk + Wk.sum(-1)
                # in-tick successors already in the node's slot row go into that slot
                hit = (M[:, :self.KK[k]] == y[:, None]) & (self._dv(maps[f"v{k}"])[:, :self.KK[k]])
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
                sidxs.append(sidx)
            self._t('read:merge orders')
            # the chain's coefficients
            hs, gs = self._chain(vals, Ns)
            K = self.KCH                                         # the back-off chain; skip orders sit after it
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
            mpt = self._dv(mp)
            mbt = self._dv(mb)
            hm = mpt >= 0
            bump = math.log(0.98 + 0.02 / V) - math.log(0.02 / V)
        # rungs mixed in log space: a batched mat-vec, not a broadcast; the match model's log-distribution is a
        # constant per row (cancelled by the softmax) plus a bump on the predicted token, so only the bump is added
        zE = torch.bmm(self.wE[deep][:, None, :], LP).squeeze(1)
        if d.type == "cuda":
            # one add per row, at the predicted token; rows with no match add 0 at token 0. A boolean-mask index makes
            # the CPU wait for the GPU to count the mask (Nov 10). Elsewhere the mask stays: on MPS that wait is what
            # keeps two runs of the learner within 1e-7 of each other (without it they drift to 1e-5; cause unknown)
            zE = zE.scatter_add(1, mpt.clamp(min=0)[:, None], (self.wM[deep, mbt] * (bump * hm.float()))[:, None])
        else:
            zE = zE.index_put((torch.arange(C, device=d)[hm], mpt[hm]), (self.wM[deep, mbt] * bump)[hm], accumulate=True)
        lpE = torch.log_softmax(zE, -1)
        self._t('read:log + exact mix')
        if getattr(self, "_count_only", False):            # the counting pass: the store's own read, no learner
            with torch.no_grad():
                ent = -(lpE.exp() * lpE).sum(-1) / math.log(2)
                key = (deep * LB + mbt) * NE + torch.bucketize(ent, self.edges)
            return lpE, lpE.detach(), key, deep
        # ---- the learner's features: P^T p_L for every level L, exactly, from the same decomposition ----
        # everything the features need, as read before this tick's store update: rehearsal replays it (core3)
        pk = dict(f0=f0, W0=W0, n0i=n0i, y=y, vals=vals, toks=toks, hs=hs, gs=gs, Ns=Ns, K=K, C=C, S=len(self.orders) - K,
                  r0=self.r0.clone() if int(self.c.get("rehearse", 0)) else self.r0)
        features = lambda ix: self._feat(pk, ix)
        wt = self._dv(win)
        self._sel_lp = None
        self._lpL_train = None
        loo = getattr(self, "_loo", None) if grad else None
        if loo is not None:                                # a repeat pass: the learner trains on the leave-one-out read
            assert rows is None, "leave-one-out reads are not combined with decimation by prediction"
            with torch.no_grad():
                pk_loo = self._loo_pack(pk, sidxs, loo)
            with torch.no_grad():                          # the honest read scores the token and drives the mix
                keep_h, self.net.keep_h = self.net.keep_h, False
                lpL = torch.log_softmax(self._learner(wt, features(None)), -1)
                self.net.keep_h = keep_h
            self._lpL_train = torch.log_softmax(self._learner(wt, self._feat(pk_loo, None)), -1)
            self._t('learner fwd')
        elif rows is None:
            with torch.set_grad_enabled(grad):
                feat = features(None)
                self._t('read:features')
                if defer:                                  # lanes_batch: the learner runs later, every lane in one pass
                    with torch.no_grad():
                        ent = -(lpE.exp() * lpE).sum(-1) / math.log(2)
                        key = (deep * LB + mbt) * NE + torch.bucketize(ent, self.edges)
                    return lpE, None, key, deep, feat
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
        if grad and int(self.c.get("rehearse", 0)):
            pk["wt"] = wt
            self._pack = pk
        if getattr(self, 'probe', None) is not None:          # instruments only (tests/receipts.py); read, never written
            self._pr = dict(LP=LP, feat=feat.detach(), wt=wt, deep=deep, mpt=mpt, mbt=mbt, hm=hm, bump=bump)
        return lpE, lpL, key, deep

    def _chain(self, vals, Ns):
        """the chain's escape (g) and level (h) coefficients per order, from merged counts."""
        NB, d = self.NB, self.dev
        hs, gs = [], []
        last = (torch.arange(NB, device=d) == NB - 1)[None, :]
        for k in range(len(vals)):
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
        return hs, gs

    def _loo_pack(self, pk, sidxs, loo):
        """the leave-one-out read (LOO_SPEC.md; James's yes, Nov 10): the same read with this position's own earlier
        instances removed. loo (NB,) is what those instances weigh now on each rung; a removal never takes a count
        below zero. Pair slot and node total per order, and the order-0 count; the chain follows from the reduced counts."""
        C, NB, d = pk["C"], self.NB, self.dev
        ix = torch.arange(C, device=d)
        vals, Ns = [], []
        for k in range(pk["K"]):
            v = pk["vals"][k].clone(); s_ = sidxs[k]; found = (s_ >= 0).float()[:, None]
            col = s_.clamp(min=0)
            cur = v[ix, :, col]                                                       # (C, NB): the target's slot
            sub = torch.minimum(cur, loo[None, :]) * found
            v[ix, :, col] = cur - sub
            vals.append(v); Ns.append((pk["Ns"][k] - sub).clamp(min=0.0))
        r0y = (pk["f0"] * pk["r0"].index_select(1, pk["y"]).T)                         # (C, NB): the target's order-0 count
        d0 = torch.minimum(r0y, loo[None, :])
        hs, gs = self._chain(vals, Ns)
        return dict(pk, vals=vals, Ns=Ns, hs=hs, gs=gs, n0i=(pk["n0i"] - d0).clamp(min=0.0), d0=d0)

    def _succ(self, v, t, Pm):
        """sum_t v[i,b,t] * Pm[tok[i,t]] -> (n, NB, DP). Gathering a projection row per successor slot costs n*T*DP floats
        (1.07 GiB a tick at order 1 with K=2048, DP=256: OOM on an 8 GB Mac, Nov 15); scattering the counts onto the vocabulary
        and doing one matmul costs n*NB*V. Same sum, different float order; take whichever is smaller."""
        n, NB, T = v.shape
        V, DP = Pm.shape
        if T * DP <= NB * V or not int(self.c.get("feat_scatter", 1)):
            return torch.einsum("ibt,itd->ibd", v, Pm.index_select(0, t.reshape(-1)).reshape(n, T, DP))
        A = torch.zeros(n, NB, V, device=v.device, dtype=v.dtype)
        A.scatter_add_(2, t[:, None, :].expand(n, NB, T), v)
        if v.is_cuda and int(self.c.get("feat_bf16", 0)):     # Nov 24: bf16 operands, fp32 accumulation
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return (A @ Pm).float()
        return A @ Pm

    def _feat(self, pk, ix):
        """the learner's features for rows ix (None: every row), exactly, from the read's decomposition (a pack)."""
        f0, W0, n0i, y, vals, toks, hs, gs, Ns, K, C = (pk[k] for k in ("f0", "W0", "n0i", "y", "vals", "toks", "hs", "gs", "Ns", "K", "C"))
        NB, d, V = self.NB, self.dev, self.V
        sel = (lambda t: t) if ix is None else (lambda t: t[ix])
        n = C if ix is None else len(ix)
        Pm = self.net.P                                                              # (V, DP)
        X0 = (sel(f0)[..., None] * (pk["r0"] @ Pm)[None] + Pm.sum(0) / V
              + torch.einsum("ibj,jd->ibd", sel(W0), Pm.index_select(0, y))) / (sel(n0i) + 1.0)[..., None]   # (n, NB, DP)
        if "d0" in pk:                                     # leave-one-out: this instance's order-0 count removed
            X0 = X0 - (sel(pk["d0"])[..., None] * sel(Pm.index_select(0, y))[:, None, :]) / (sel(n0i) + 1.0)[..., None]
        # index_select, not Pm[...]: same maths, but its backward is an index_add, not a sort (half of CUDA time, Nov 9)
        Xs = [self._succ(sel(vals[k]), sel(toks[k]), Pm)
              for k in range(K)]
        fs = [torch.cat([X0, torch.log1p(sel(n0i))[..., None]], -1)]
        for L in range(1, K + 1):
            run = torch.ones(n, NB, device=d)
            acc = 0.0
            for k in range(L - 1, -1, -1):
                acc = acc + (sel(hs[k]) * run)[..., None] * Xs[k]
                run = run * sel(gs[k])
            acc = acc + run[..., None] * X0
            fs.append(torch.cat([acc, torch.log1p(sel(Ns[L - 1]))[..., None]], -1))
        for k in range(K, K + pk.get("S", 0)):            # skip orders: their own level, backed off to order 0
            Xk = self._succ(sel(vals[k]), sel(toks[k]), Pm)
            acc = sel(hs[k])[..., None] * Xk + sel(gs[k])[..., None] * X0
            fs.append(torch.cat([acc, torch.log1p(sel(Ns[k]))[..., None]], -1))
        return torch.stack(fs, 1)                                                    # (n, O, NB, DP+1)

    def _learner(self, wt, feat):
        """the learner's logits (fp32); on cuda with learner_bf16 its matmuls run in bf16 autocast."""
        if self.dev.type == "cuda" and int(self.c.get("learner_bf16", 0)):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = self.net(wt, feat)
            return z.float()
        return self.net(wt, feat)

    def _learner_lanes(self, seq, feat, cache, whole=False):
        """forward_lanes under the same autocast as _learner -> (logits fp32 (B, C, V), this tick's [(K, V)])"""
        if self.dev.type == "cuda" and int(self.c.get("learner_bf16", 0)):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z, out = self.net.forward_lanes(seq, feat, cache, whole)
            return z.float(), out
        return self.net.forward_lanes(seq, feat, cache, whole)

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
