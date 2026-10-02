"""the body: a small pretrained causal transformer that soma consults.

A lean, frozen Llama forward (SmolLM2's architecture), read straight from its
safetensors file.  Its opinion before a token is a deterministic function of the tokens
just before it, so it keeps no state of its own for learning: `hidden_at` reads fresh
windows, each starting at least `context` tokens before the first position it serves
and serving up to `span` positions, batched together.  Only positions soma actually
trains on are asked for, so stretches io2 decimates past are never read.  Generation
extends one token at a time through a bounded KV cache (`read`, `step`): when it fills,
the last `keep` tokens are re-read and the rest dropped.

On MPS in half precision, `step` runs a fused decode: one-token matvecs at memory
bandwidth, the SwiGLU in the gate/up pass, rotary embedding fused with the cache
write, and residual-add fused with RMSNorm.  The eager forward is the specification.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BODY_NAMES = ("smollm2-135m",)
BODIES = Path(__file__).resolve().parent / "bodies"


def body_directory(name) -> Path:
    """where a named body lives: $SOMA_BODY_DIR/<name>, else the package's bodies/."""
    roots = [Path(os.environ["SOMA_BODY_DIR"])] if os.environ.get("SOMA_BODY_DIR") else []
    roots.append(BODIES)
    for root in roots:
        candidate = root / str(name)
        if (candidate / "model.safetensors").exists():
            return candidate
    raise FileNotFoundError(f"body {name!r} not found under {', '.join(map(str, roots))}")


def read_safetensors(path):
    """name -> float32 tensor, for BF16/F16/F32 safetensors files."""
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
        data = memoryview(handle.read())
    tensors = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        start, end = spec["data_offsets"]
        kind = spec["dtype"]
        if kind == "BF16":
            raw = np.frombuffer(data[start:end], dtype=np.uint16)
            array = (raw.astype(np.uint32) << 16).view(np.float32)
        elif kind == "F16":
            array = np.frombuffer(data[start:end], dtype=np.float16).astype(np.float32)
        elif kind == "F32":
            array = np.frombuffer(data[start:end], dtype=np.float32).copy()
        else:
            raise ValueError(f"unsupported tensor type {kind} for {name}")
        tensors[name] = torch.from_numpy(array.reshape(spec["shape"]))
    return tensors


_METAL = r"""
#include <metal_stdlib>
using namespace metal;

// residual += delta (optional); out = rms_norm(residual) * weight.  One threadgroup.
kernel void add_rms_norm(
    device half* residual [[buffer(0)]], device const half* delta [[buffer(1)]],
    device const half* weight [[buffer(2)]], device half* out [[buffer(3)]],
    constant uint& width [[buffer(4)]], constant float& eps [[buffer(5)]],
    constant uint& add [[buffer(6)]],
    uint i [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[32];
    float x = 0.0f;
    if (i < width) {
        x = float(residual[i]);
        if (add) { x += float(delta[i]); residual[i] = half(x); }
    }
    float s = simd_sum(x * x);
    if (lane == 0) partial[group] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0) {
        float t = simd_sum(lane < (width + 31) / 32 ? partial[lane] : 0.0f);
        if (lane == 0) partial[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = rsqrt(partial[0] / float(width) + eps);
    if (i < width) out[i] = half(float(half(x)) * scale * float(weight[i]));
}

// q heads, then k heads, then v heads; one thread per (head, dim).  q and k are
// rotated (rotate-half convention); k and v are written into the cache at `position`.
kernel void rope_cache(
    device const half* qkv [[buffer(0)]], device const half* cos_t [[buffer(1)]],
    device const half* sin_t [[buffer(2)]], device half* cache [[buffer(3)]],
    device half* q_out [[buffer(4)]],
    constant uint& heads [[buffer(5)]], constant uint& kv_heads [[buffer(6)]],
    constant uint& D [[buffer(7)]], constant uint& context [[buffer(8)]],
    constant uint& layer [[buffer(9)]], constant uint& position [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    uint head = gid / D, d = gid % D, half_d = D / 2;
    if (head >= heads + 2 * kv_heads) return;
    float x = float(qkv[gid]), value = x;
    if (head < heads + kv_heads) {
        float partner = d < half_d ? -float(qkv[gid + half_d]) : float(qkv[gid - half_d]);
        value = x * float(cos_t[position * D + d]) + partner * float(sin_t[position * D + d]);
    }
    if (head < heads) { q_out[gid] = half(value); return; }
    uint kind = head < heads + kv_heads ? 0 : 1;
    uint h = kind == 0 ? head - heads : head - heads - kv_heads;
    cache[(((ulong(layer) * 2 + kind) * kv_heads + h) * context + position) * D + d] = half(value);
}

// y = W x for one token: a simdgroup per row, four rows per threadgroup.
kernel void gemv(
    device const half* W [[buffer(0)]], device const half* x [[buffer(1)]],
    device half* y [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& cols [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tg * 4 + group;
    if (row >= rows) return;
    device const half4* w4 = (device const half4*)(W + ulong(row) * cols);
    device const half4* x4 = (device const half4*)x;
    float acc = 0.0f;
    for (uint i = lane; i < cols / 4; i += 32) acc += dot(float4(w4[i]), float4(x4[i]));
    acc = simd_sum(acc);
    if (lane == 0) y[row] = half(acc);
}

// y_i = silu(gate_i . x) * (up_i . x); gate rows [0, n), up rows [n, 2n)
kernel void gemv_swiglu(
    device const half* W [[buffer(0)]], device const half* x [[buffer(1)]],
    device half* y [[buffer(2)]], constant uint& n [[buffer(3)]], constant uint& cols [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tg * 4 + group;
    if (row >= n) return;
    device const half4* g4 = (device const half4*)(W + ulong(row) * cols);
    device const half4* u4 = (device const half4*)(W + ulong(row + n) * cols);
    device const half4* x4 = (device const half4*)x;
    float g = 0.0f, u = 0.0f;
    for (uint i = lane; i < cols / 4; i += 32) {
        float4 xv = float4(x4[i]);
        g += dot(float4(g4[i]), xv);
        u += dot(float4(u4[i]), xv);
    }
    g = simd_sum(g); u = simd_sum(u);
    if (lane == 0) y[row] = half(half(g) / (1.0f + exp(-float(half(g)))) * float(half(u)));
}
"""

_library = None


def _metal_library():
    global _library
    if _library is None:
        _library = torch.mps.compile_shader(_METAL)
    return _library


class Body:
    def __init__(self, name="smollm2-135m", device="cpu", dtype=None, context=2048, keep=1024):
        self.name = str(name)
        self.directory = body_directory(name)
        config = json.loads((self.directory / "config.json").read_text())
        if config.get("model_type") != "llama" or not config.get("tie_word_embeddings", False):
            raise ValueError("a body must be a Llama with tied embeddings")
        self.device = torch.device(device)
        self.dtype = dtype or (torch.float16 if self.device.type == "mps" else torch.float32)
        self.width = int(config["hidden_size"])
        self.heads = int(config["num_attention_heads"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.head_dim = self.width // self.heads
        self.layers = int(config["num_hidden_layers"])
        self.eps = float(config["rms_norm_eps"])
        self.vocab = int(config["vocab_size"])
        self.context = min(int(context), int(config["max_position_embeddings"]))
        self.keep = min(int(keep), self.context // 2)
        weights = read_safetensors(self.directory / "model.safetensors")
        to = lambda t: t.to(self.device, self.dtype).contiguous()
        self.embedding = to(weights["model.embed_tokens.weight"])
        self.final_norm = to(weights["model.norm.weight"])
        self.blocks = []
        for i in range(self.layers):
            p = f"model.layers.{i}."
            self.blocks.append({
                "attention_norm": to(weights[p + "input_layernorm.weight"]),
                "qkv": to(torch.cat([weights[p + f"self_attn.{n}_proj.weight"] for n in "qkv"])),
                "out": to(weights[p + "self_attn.o_proj.weight"]),
                "mlp_norm": to(weights[p + "post_attention_layernorm.weight"]),
                "gate_up": to(torch.cat([weights[p + "mlp.gate_proj.weight"], weights[p + "mlp.up_proj.weight"]])),
                "down": to(weights[p + "mlp.down_proj.weight"]),
            })
        del weights
        half = self.head_dim // 2
        frequency = 1.0 / (float(config["rope_theta"]) ** (
            torch.arange(half, dtype=torch.float64) * 2 / self.head_dim))
        angle = torch.arange(self.context, dtype=torch.float64)[:, None] * frequency[None]
        self.cos = to(torch.cat([angle.cos()] * 2, 1))
        self.sin = to(torch.cat([angle.sin()] * 2, 1))
        self.cache = torch.zeros(
            (self.layers, 2, 1, self.kv_heads, self.context, self.head_dim),
            device=self.device, dtype=self.dtype,
        )
        self.length = 0
        self.tokens = []                  # the ids the cache holds, for re-reading on overflow
        self.fused = self.device.type == "mps" and self.dtype == torch.float16
        if self.fused:
            self._lib = _metal_library()
            buffer = lambda n: torch.zeros(n, device=self.device, dtype=self.dtype)
            self._residual, self._normed = buffer(self.width), buffer(self.width)
            self._delta, self._zero = buffer(self.width), buffer(self.width)
            self._qkv = buffer(self.blocks[0]["qkv"].shape[0])
            self._q = buffer(self.heads * self.head_dim)
            self._act = buffer(self.blocks[0]["down"].shape[1])

    @property
    def parameter_count(self):
        return self.embedding.numel() + self.final_norm.numel() + sum(
            t.numel() for block in self.blocks for t in block.values())

    # ---------------------------------------------------------------- state

    def reset(self):
        self.length = 0
        self.tokens = []

    def restore(self, tokens):
        """rebuild the cache from a saved token list (the body's whole state)."""
        self.reset()
        tokens = [int(t) for t in tokens][-self.context:]
        if tokens:
            return self.read(tokens)[-1:]
        return None

    # -------------------------------------------------------------- forward

    def _rope(self, x, start):
        cos = self.cos[start:start + x.shape[2]]
        sin = self.sin[start:start + x.shape[2]]
        half = self.head_dim // 2
        return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin

    def _forward(self, ids):
        """ids [T] from the current cache position; final hidden [T, width]."""
        T, start = len(ids), self.length
        x = self.embedding[ids][None]
        q_width, kv_width = self.width, self.kv_heads * self.head_dim
        for index, block in enumerate(self.blocks):
            h = F.rms_norm(x, (self.width,), block["attention_norm"], self.eps)
            q, k, v = (h @ block["qkv"].T).split((q_width, kv_width, kv_width), -1)
            q = self._rope(q.view(1, T, self.heads, self.head_dim).transpose(1, 2), start)
            k = self._rope(k.view(1, T, self.kv_heads, self.head_dim).transpose(1, 2), start)
            v = v.view(1, T, self.kv_heads, self.head_dim).transpose(1, 2)
            self.cache[index, 0, :, :, start:start + T] = k
            self.cache[index, 1, :, :, start:start + T] = v
            keys = self.cache[index, 0, :, :, :start + T]
            values = self.cache[index, 1, :, :, :start + T]
            if T == 1:
                attended = F.scaled_dot_product_attention(q, keys, values, enable_gqa=True)
            else:
                mask = torch.ones(T, start + T, dtype=torch.bool, device=self.device).tril(start)
                attended = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, enable_gqa=True)
            x = x + attended.transpose(1, 2).reshape(1, T, self.width) @ block["out"].T
            h = F.rms_norm(x, (self.width,), block["mlp_norm"], self.eps)
            gate, up = (h @ block["gate_up"].T).chunk(2, -1)
            x = x + (F.silu(gate) * up) @ block["down"].T
        self.length += T
        return F.rms_norm(x[0], (self.width,), self.final_norm, self.eps)

    def _norm(self, delta, weight, add=True):
        self._lib.add_rms_norm(self._residual, delta, weight, self._normed, self.width,
                               float(self.eps), int(add), threads=self.width, group_size=self.width)
        return self._normed

    def _gemv(self, W, x, out):
        rows = W.shape[0]
        self._lib.gemv(W, x, out, rows, W.shape[1], threads=((rows + 3) // 4) * 128, group_size=128)
        return out

    def _fused_step(self, token_id):
        position, D, H, KV = self.length, self.head_dim, self.heads, self.kv_heads
        self._residual.copy_(self.embedding[token_id])
        delta, first = self._zero, True
        for index, block in enumerate(self.blocks):
            h = self._norm(delta, block["attention_norm"], add=not first)
            first = False
            qkv = self._gemv(block["qkv"], h, self._qkv)
            self._lib.rope_cache(qkv, self.cos, self.sin, self.cache, self._q, H, KV, D,
                                 self.context, index, position, threads=(H + 2 * KV) * D, group_size=D)
            attended = F.scaled_dot_product_attention(
                self._q.view(1, H, 1, D), self.cache[index, 0, :, :, :position + 1],
                self.cache[index, 1, :, :, :position + 1], enable_gqa=True)
            delta = self._gemv(block["out"], attended.view(-1), self._delta)
            h = self._norm(delta, block["mlp_norm"])
            n = self._act.shape[0]
            self._lib.gemv_swiglu(block["gate_up"], h, self._act, n, self.width,
                                  threads=((n + 3) // 4) * 128, group_size=128)
            delta = self._gemv(block["down"], self._act, self._delta)
        self.length += 1
        return self._norm(delta, self.final_norm).clone()[None]

    def _make_room(self, incoming):
        if self.length + incoming <= self.context:
            return
        kept = self.tokens[-self.keep:] if incoming < self.context - self.keep else []
        self.reset()
        if kept:
            self._read(kept)

    def _read(self, ids, chunk=512):
        hidden = []
        for lo in range(0, len(ids), chunk):
            part = [int(t) for t in ids[lo:lo + chunk]]
            self._make_room(len(part))
            hidden.append(self._forward(torch.as_tensor(part, device=self.device)))
            self.tokens.extend(part)
        if not hidden:
            return torch.empty(0, self.width, device=self.device, dtype=self.dtype)
        return torch.cat(hidden)

    @torch.no_grad()
    def read(self, ids):
        """read ids; the final hidden state [T, width] at each of them."""
        return self._read(list(ids))

    @torch.no_grad()
    def step(self, token_id):
        """append one token; its final hidden state [1, width]."""
        self._make_room(1)
        token_id = int(token_id)
        if self.fused:
            hidden = self._fused_step(token_id)
        else:
            hidden = self._forward(torch.as_tensor([token_id], device=self.device))
        self.tokens.append(token_id)
        return hidden

    # ------------------------------------------------------ fresh windows

    def _forward_batch(self, x):
        """stateless causal forward of fresh sequences x [N, T] (right-padded
        is fine); final hidden [N, T, width]."""
        N, T = x.shape
        h = self.embedding[x]
        q_width, kv_width = self.width, self.kv_heads * self.head_dim
        cos, sin, half = self.cos[:T], self.sin[:T], self.head_dim // 2
        rope = lambda t: t * cos + torch.cat((-t[..., half:], t[..., :half]), -1) * sin
        for block in self.blocks:
            a = F.rms_norm(h, (self.width,), block["attention_norm"], self.eps)
            q, k, v = (a @ block["qkv"].T).split((q_width, kv_width, kv_width), -1)
            q = rope(q.view(N, T, self.heads, self.head_dim).transpose(1, 2))
            k = rope(k.view(N, T, self.kv_heads, self.head_dim).transpose(1, 2))
            v = v.view(N, T, self.kv_heads, self.head_dim).transpose(1, 2)
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            h = h + attended.transpose(1, 2).reshape(N, T, self.width) @ block["out"].T
            a = F.rms_norm(h, (self.width,), block["mlp_norm"], self.eps)
            gate, up = (a @ block["gate_up"].T).chunk(2, -1)
            h = h + (F.silu(gate) * up) @ block["down"].T
        return F.rms_norm(h, (self.width,), self.final_norm, self.eps)

    @staticmethod
    def windows(positions, context, span):
        """(start, end, first, last) for fresh windows over a stream that serve
        the increasing `positions`: each window starts `context` tokens before
        the first position it serves (or at the stream's start) and serves the
        positions within `span` of it.  A gap wider than `context` starts a new
        window: reading its own context is cheaper than reading the gap, so a
        position costs min(gap, context + 1) tokens."""
        out, i = [], 0
        while i < len(positions):
            first = positions[i]
            j = i + 1
            while (
                j < len(positions) and positions[j] < first + span
                and positions[j] - positions[j - 1] <= context
            ):
                j += 1
            out.append((max(0, first - context), positions[j - 1] + 1, i, j))
            i = j
        return out

    @torch.no_grad()
    def hidden_at(self, stream, positions, context=256, span=1024, budget=12288):
        """final hidden state after stream[p] for each increasing p in
        positions, each read in a fresh window holding at least `context`
        tokens before it; windows of similar work are batched together."""
        positions = [int(p) for p in positions]
        out = torch.empty(len(positions), self.width, device=self.device, dtype=self.dtype)
        plan = self.windows(positions, int(context), int(span))
        index = 0
        while index < len(plan):
            group, longest = [plan[index]], plan[index][1] - plan[index][0]
            while index + len(group) < len(plan):
                nxt = plan[index + len(group)]
                wider = max(longest, nxt[1] - nxt[0])
                if wider * (len(group) + 1) > budget:
                    break
                group.append(nxt)
                longest = wider
            # lengths round up to a multiple of 32 so a few kernel shapes are
            # reused; padding after the served positions cannot reach them
            x = torch.zeros(len(group), -(-longest // 32) * 32, dtype=torch.long)
            rows, columns, targets = [], [], []
            for row, (start, end, first, last) in enumerate(group):
                x[row, :end - start] = torch.as_tensor(stream[start:end])
                for k in range(first, last):
                    rows.append(row)
                    columns.append(positions[k] - start)
                    targets.append(k)
            hidden = self._forward_batch(x.to(self.device))
            out[torch.as_tensor(targets, device=self.device)] = hidden[
                torch.as_tensor(rows, device=self.device),
                torch.as_tensor(columns, device=self.device),
            ]
            index += len(group)
        return out

    # ------------------------------------------------------------- opinions

    def logits(self, hidden):
        return (hidden @ self.embedding.T).float()

    @torch.no_grad()
    def opinion(self, hidden, targets, exclude=None, chunk=512, full=False):
        """(log p of each target, entropy, top log-prob) from the hidden rows
        before them; rows are projected a chunk at a time.  With `exclude`,
        target log-probabilities are conditional on not being those ids.  With
        `full`, also every row's whole log-distribution."""
        target_log, entropy, top, whole = [], [], [], []
        for lo in range(0, len(hidden), chunk):
            logits = self.logits(hidden[lo:lo + chunk])
            normaliser = torch.logsumexp(logits, -1)
            text_normaliser = (
                torch.logsumexp(logits.index_fill(1, exclude, float("-inf")), -1)
                if exclude is not None else normaliser
            )
            entropy.append(normaliser - (torch.softmax(logits, -1) * logits).sum(-1))
            top.append(logits.max(-1).values - normaliser)
            target_log.append(
                logits.gather(1, targets[lo:lo + chunk, None])[:, 0] - text_normaliser
            )
            if full:
                whole.append(logits - normaliser[:, None])
        empty = torch.empty(0, device=self.device)
        cat = lambda x: torch.cat(x) if x else empty
        if full:
            return cat(target_log), cat(entropy), cat(top), (
                torch.cat(whole) if whole else torch.empty(0, self.vocab, device=self.device)
            )
        return cat(target_log), cat(entropy), cat(top)
