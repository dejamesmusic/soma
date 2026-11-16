"""causal flash attention for MPS (Metal simdgroup matrices), forward + backward, fp32, head dim 64 (1.2.1, Nov 16).

FlashAttention-2 tiling: 32 queries x 32 keys per tile, 4 simdgroups of 8 rows per threadgroup. Forward keeps only the
per-row log-sum-exp; backward recomputes P from it (one kernel over key blocks for dK, dV; one over query blocks for dQ).
Nothing T x T is ever stored, so the attention learner needs no block recompute for memory.
Shapes: q, k, v (B, H, T, 64), T a multiple of 32. Same maths as softmax(q k^T / 8, causal) v, up to float order."""
import torch

D, BQ = 64, 32

SRC = r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#define D 64
#define BQ 32
#define BK 32

// ---- forward: one threadgroup = 32 queries of one (batch, head); simdgroup sg owns rows sg*8 .. sg*8+7 ----
kernel void flash_fwd(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
                      device const float* V [[buffer(2)]], device float* O [[buffer(3)]], device float* LSE [[buffer(4)]],
                      constant uint& T [[buffer(5)]], constant float& scale [[buffer(6)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float Ks[BK * D];
    threadgroup float Vs[BK * D];
    threadgroup float Sc[4][8 * BK];
    threadgroup float Dg[4][64];
    const uint bh = tg.y, qb = T / BQ - 1 - tg.x;          // heaviest (latest) query blocks first
    const uint q0 = qb * BQ + sg * 8;
    device const float* Qp = Q + ((ulong)bh * T + q0) * D;
    simdgroup_float8x8 Qm[8], Om[8];
    for (int i = 0; i < 8; i++) { simdgroup_load(Qm[i], Qp + i * 8, D); Om[i] = simdgroup_float8x8(0); }
    const uint r = lane / 4, sub = lane % 4, qi = q0 + r;
    float m = -INFINITY, l = 0.0f;
    for (uint kb = 0; kb <= qb; kb++) {
        device const float* Kp = K + ((ulong)bh * T + kb * BK) * D;
        device const float* Vp = V + ((ulong)bh * T + kb * BK) * D;
        for (uint e = tid; e < BK * D / 4; e += 128) {
            ((threadgroup float4*)Ks)[e] = ((device const float4*)Kp)[e]; ((threadgroup float4*)Vs)[e] = ((device const float4*)Vp)[e]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < 4; j++) {                       // S = Q K^T, 8 x 32
            simdgroup_float8x8 S = simdgroup_float8x8(0), Kt;
            for (int i = 0; i < 8; i++) {
                simdgroup_load(Kt, Ks + (j * 8) * D + i * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S, Qm[i], Kt, S);
            }
            simdgroup_store(S, Sc[sg] + j * 8, BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float s[8], mx = -INFINITY;
        for (int c = 0; c < 8; c++) {
            uint kj = kb * BK + sub * 8 + c;
            float x = (kj <= qi) ? Sc[sg][r * BK + sub * 8 + c] * scale : -INFINITY;
            s[c] = x; mx = max(mx, x);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1)); mx = max(mx, simd_shuffle_xor(mx, 2));
        const float mn = max(m, mx), alpha = exp(m - mn);
        float sum = 0.0f;
        for (int c = 0; c < 8; c++) { float p = exp(s[c] - mn); Sc[sg][r * BK + sub * 8 + c] = p; sum += p; }
        sum += simd_shuffle_xor(sum, 1); sum += simd_shuffle_xor(sum, 2);
        l = l * alpha + sum; m = mn;
        Dg[sg][lane] = 0.0f; Dg[sg][lane + 32] = 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (sub == 0) Dg[sg][r * 9] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 A, tmp; simdgroup_load(A, Dg[sg], 8);
        for (int i = 0; i < 8; i++) { simdgroup_multiply(tmp, A, Om[i]); Om[i] = tmp; }
        for (int k2 = 0; k2 < 4; k2++) {                    // O += P V
            simdgroup_float8x8 P, Vm; simdgroup_load(P, Sc[sg] + k2 * 8, BK);
            for (int i = 0; i < 8; i++) {
                simdgroup_load(Vm, Vs + (k2 * 8) * D + i * 8, D);
                simdgroup_multiply_accumulate(Om[i], P, Vm, Om[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    Dg[sg][lane] = 0.0f; Dg[sg][lane + 32] = 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (sub == 0) Dg[sg][r * 9] = 1.0f / l;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 A, tmp; simdgroup_load(A, Dg[sg], 8);
    device float* Op = O + ((ulong)bh * T + q0) * D;
    for (int i = 0; i < 8; i++) { simdgroup_multiply(tmp, A, Om[i]); simdgroup_store(tmp, Op + i * 8, D); }
    if (sub == 0) LSE[(ulong)bh * T + qi] = m + log(l);
}

// ---- backward, keys: one threadgroup = 32 keys, over the query blocks at or after them. Two kernels so each holds
// fewer matrices (one kernel holding K, V, dK, dV spilled registers: 6.3 ms vs 2.2 for dQ): dV needs P only; dK needs P, dP.
template <int WHICH>
void bwd_keys(device const float* Q, device const float* K, device const float* V, device const float* dO,
              device const float* LSE, device const float* Dv, device float* dX, uint T, float scale,
              uint2 tg, uint tid, uint sg, uint lane,
              threadgroup float* Qs, threadgroup float* dOs, threadgroup float (*Sc)[8 * BQ], threadgroup float (*Sd)[8 * BQ],
              threadgroup float* Ls, threadgroup float* Ds) {
    const uint bh = tg.y, kb = tg.x, nqb = T / BQ;
    const uint k0 = kb * BK + sg * 8;
    simdgroup_float8x8 Km[8], Vm[8], Xm[8];
    device const float* Kp = K + ((ulong)bh * T + k0) * D;
    device const float* Vp = V + ((ulong)bh * T + k0) * D;
    for (int i = 0; i < 8; i++) {
        simdgroup_load(Km[i], Kp + i * 8, D);
        if (WHICH == 1) simdgroup_load(Vm[i], Vp + i * 8, D);
        Xm[i] = simdgroup_float8x8(0);
    }
    const uint r = lane / 4, sub = lane % 4, kr = k0 + r;
    for (uint qb = kb; qb < nqb; qb++) {
        device const float* Qp = Q + ((ulong)bh * T + qb * BQ) * D;
        device const float* dOp = dO + ((ulong)bh * T + qb * BQ) * D;
        for (uint e = tid; e < BQ * D / 4; e += 128) {
            ((threadgroup float4*)Qs)[e] = ((device const float4*)Qp)[e]; ((threadgroup float4*)dOs)[e] = ((device const float4*)dOp)[e]; }
        if (tid < BQ) { Ls[tid] = LSE[(ulong)bh * T + qb * BQ + tid]; Ds[tid] = Dv[(ulong)bh * T + qb * BQ + tid]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < 4; j++) {                       // S^T = K Q^T (and dP^T = V dO^T), 8 keys x 32 queries
            simdgroup_float8x8 S = simdgroup_float8x8(0), P = simdgroup_float8x8(0), Qt;
            for (int i = 0; i < 8; i++) {
                simdgroup_load(Qt, Qs + (j * 8) * D + i * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S, Km[i], Qt, S);
                if (WHICH == 1) {
                    simdgroup_load(Qt, dOs + (j * 8) * D + i * 8, D, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(P, Vm[i], Qt, P);
                }
            }
            simdgroup_store(S, Sc[sg] + j * 8, BQ);
            if (WHICH == 1) simdgroup_store(P, Sd[sg] + j * 8, BQ);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = 0; c < 8; c++) {
            uint col = sub * 8 + c, qj = qb * BQ + col, e = r * BQ + col;
            float p = (qj >= kr) ? exp(Sc[sg][e] * scale - Ls[col]) : 0.0f;
            if (WHICH == 0) Sc[sg][e] = p;
            else Sd[sg][e] = p * (Sd[sg][e] - Ds[col]) * scale;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int k2 = 0; k2 < 4; k2++) {                    // dV += P^T dO   |   dK += dS^T Q
            simdgroup_float8x8 G, X;
            simdgroup_load(G, (WHICH == 0 ? Sc[sg] : Sd[sg]) + k2 * 8, BQ);
            for (int i = 0; i < 8; i++) {
                simdgroup_load(X, (WHICH == 0 ? dOs : Qs) + (k2 * 8) * D + i * 8, D);
                simdgroup_multiply_accumulate(Xm[i], G, X, Xm[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* dXp = dX + ((ulong)bh * T + k0) * D;
    for (int i = 0; i < 8; i++) simdgroup_store(Xm[i], dXp + i * 8, D);
}

#define BWD_KEYS_KERNEL(NAME, W) \
kernel void NAME(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]], \
                 device const float* V [[buffer(2)]], device const float* dO [[buffer(3)]], \
                 device const float* LSE [[buffer(4)]], device const float* Dv [[buffer(5)]], \
                 device float* dX [[buffer(6)]], constant uint& T [[buffer(7)]], constant float& scale [[buffer(8)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float Qs[BQ * D]; threadgroup float dOs[BQ * D]; \
    threadgroup float Sc[4][8 * BQ]; threadgroup float Sd[4][8 * BQ]; threadgroup float Ls[BQ], Ds[BQ]; \
    bwd_keys<W>(Q, K, V, dO, LSE, Dv, dX, T, scale, tg, tid, sg, lane, Qs, dOs, Sc, Sd, Ls, Ds); }
BWD_KEYS_KERNEL(flash_bwd_v, 0)
BWD_KEYS_KERNEL(flash_bwd_k, 1)

// ---- backward, queries: one threadgroup = 32 queries; dQ accumulates over the key blocks at or before them ----
kernel void flash_bwd_q(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
                        device const float* V [[buffer(2)]], device const float* dO [[buffer(3)]],
                        device const float* LSE [[buffer(4)]], device const float* Dv [[buffer(5)]],
                        device float* dQ [[buffer(6)]], constant uint& T [[buffer(7)]], constant float& scale [[buffer(8)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float Ks[BK * D];
    threadgroup float Vs[BK * D];
    threadgroup float Sc[4][8 * BK];
    threadgroup float Sd[4][8 * BK];
    const uint bh = tg.y, qb = T / BQ - 1 - tg.x;
    const uint q0 = qb * BQ + sg * 8;
    simdgroup_float8x8 Qm[8], dOm[8], dQm[8];
    device const float* Qp = Q + ((ulong)bh * T + q0) * D;
    device const float* dOp = dO + ((ulong)bh * T + q0) * D;
    for (int i = 0; i < 8; i++) {
        simdgroup_load(Qm[i], Qp + i * 8, D); simdgroup_load(dOm[i], dOp + i * 8, D); dQm[i] = simdgroup_float8x8(0);
    }
    const uint r = lane / 4, sub = lane % 4, qi = q0 + r;
    const float Lr = LSE[(ulong)bh * T + qi], Dr = Dv[(ulong)bh * T + qi];
    for (uint kb = 0; kb <= qb; kb++) {
        device const float* Kp = K + ((ulong)bh * T + kb * BK) * D;
        device const float* Vp = V + ((ulong)bh * T + kb * BK) * D;
        for (uint e = tid; e < BK * D / 4; e += 128) {
            ((threadgroup float4*)Ks)[e] = ((device const float4*)Kp)[e]; ((threadgroup float4*)Vs)[e] = ((device const float4*)Vp)[e]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < 4; j++) {                       // S = Q K^T, dP = dO V^T, 8 x 32
            simdgroup_float8x8 S = simdgroup_float8x8(0), P = simdgroup_float8x8(0), Kt;
            for (int i = 0; i < 8; i++) {
                simdgroup_load(Kt, Ks + (j * 8) * D + i * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S, Qm[i], Kt, S);
                simdgroup_load(Kt, Vs + (j * 8) * D + i * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(P, dOm[i], Kt, P);
            }
            simdgroup_store(S, Sc[sg] + j * 8, BK);
            simdgroup_store(P, Sd[sg] + j * 8, BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = 0; c < 8; c++) {
            uint col = sub * 8 + c, kj = kb * BK + col, e = r * BK + col;
            float p = (kj <= qi) ? exp(Sc[sg][e] * scale - Lr) : 0.0f;
            Sd[sg][e] = p * (Sd[sg][e] - Dr) * scale;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int k2 = 0; k2 < 4; k2++) {                    // dQ += dS K
            simdgroup_float8x8 G, X;
            simdgroup_load(G, Sd[sg] + k2 * 8, BK);
            for (int i = 0; i < 8; i++) {
                simdgroup_load(X, Ks + (k2 * 8) * D + i * 8, D);
                simdgroup_multiply_accumulate(dQm[i], G, X, dQm[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* dQp = dQ + ((ulong)bh * T + q0) * D;
    for (int i = 0; i < 8; i++) simdgroup_store(dQm[i], dQp + i * 8, D);
}
'''

_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(SRC)
    return _lib


def supported(q) -> bool:
    return q.device.type == "mps" and q.dtype == torch.float32 and q.shape[-1] == D and q.shape[-2] % BQ == 0


class FlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        B, H, T, _ = q.shape
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        o = torch.empty_like(q)
        lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
        scale = 1.0 / D ** 0.5
        lib().flash_fwd(q, k, v, o, lse, T, scale, threads=(128 * (T // BQ), B * H), group_size=(128, 1))
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, H, T, _ = q.shape
        do = do.contiguous()
        dv_ = (do * o).sum(-1).contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        scale = 1.0 / D ** 0.5
        L = lib()
        g = dict(threads=(128 * (T // BQ), B * H), group_size=(128, 1))
        L.flash_bwd_v(q, k, v, do, lse, dv_, dv, T, scale, **g)
        L.flash_bwd_k(q, k, v, do, lse, dv_, dk, T, scale, **g)
        L.flash_bwd_q(q, k, v, do, lse, dv_, dq, T, scale, threads=(128 * (T // BQ), B * H), group_size=(128, 1))
        return dq, dk, dv


def attention(q, k, v):
    """causal attention, (B, H, T, 64): the Metal kernel when it applies, else torch's."""
    if supported(q):
        return FlashAttn.apply(q, k, v)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
