"""
Metal compute kernel for the soma trace scan.

The single hot operation, run for every byte:
    states[t, k, :] = traces[k, :]
    traces[k, :] *= decay[k]
    traces[k, bytes[t]] += alphas[k]

One thread per band. Each thread walks all N bytes sequentially, holding
its band's decay and alpha in registers, updating its own row of traces.
Bands are independent — no synchronization between threads.

Layout:
    traces:  (K, V)  k-major, each thread's row contiguous
    states:  (N, K, V) snapshot before each absorb
    bytes:   (N,) uint8

Falls back to pure-PyTorch sequential scan when MPS or compile_shader
isn't available, or if the kernel dispatch fails for any reason. Same
math, same numerics, just slower.
"""

import torch


# Metal kernel source. Compiled once on first use.
#
# Two kernels:
#   trace_scan    - one thread per band, walks all V channels serially.
#                   simple, ~24K b/s on M-series. used as a fallback.
#   trace_scan_v2 - K*V threads, each owns one (band, channel) pair.
#                   each thread keeps its single value in a register and
#                   walks N timesteps. ~100x more parallelism than v1.
#                   no threadgroup barrier needed: threads only ever read
#                   their own register and broadcast-read bytes[t]; no
#                   cross-thread data flow within the loop.
_METAL_SCAN_SRC = """
#include <metal_stdlib>
using namespace metal;

// v1 — one thread per band. Each walks all 256 channels per timestep.
// Output layout: states[(t * V + v) * K + k]  (N, V, K) — matches v2.
kernel void trace_scan(
    device float*       traces  [[buffer(0)]],
    device float*       states  [[buffer(1)]],
    device const uchar* bytes   [[buffer(2)]],
    device const float* decay   [[buffer(3)]],
    device const float* alphas  [[buffer(4)]],
    constant uint&      N       [[buffer(5)]],
    constant uint&      K       [[buffer(6)]],
    constant uint&      V       [[buffer(7)]],
    uint k [[thread_position_in_grid]]
) {
    if (k >= K) return;

    const float d = decay[k];
    const float a = alphas[k];
    device float* row = traces + k * V;

    for (uint t = 0; t < N; t++) {
        // snapshot pre-absorb into (N, V, K)
        for (uint v = 0; v < V; v++) {
            states[(t * V + v) * K + k] = row[v];
        }
        // decay every channel
        for (uint v = 0; v < V; v++) {
            row[v] = row[v] * d;
        }
        // absorb at observed byte
        uint b = (uint)bytes[t];
        row[b] = row[b] + a;
    }
}

// v2 — K threadgroups of V threads. Each thread owns one (band, channel).
// The thread keeps its trace value in a register across all N timesteps,
// only writing to global memory once per timestep (the snapshot).
//
// No barrier needed: each thread reads only its own register state +
// the broadcast-read bytes[t]; threads never read each other's writes
// inside the loop.
//
// Output layout: states[(t * V + v) * K + k]  (N, V, K) — matches the
// layout the rest of the pipeline expects, so no transpose needed.
kernel void trace_scan_v2(
    device float*       traces  [[buffer(0)]],
    device float*       states  [[buffer(1)]],
    device const uchar* bytes   [[buffer(2)]],
    device const float* decay   [[buffer(3)]],
    device const float* alphas  [[buffer(4)]],
    constant uint&      N       [[buffer(5)]],
    constant uint&      K       [[buffer(6)]],
    constant uint&      V       [[buffer(7)]],
    uint k [[threadgroup_position_in_grid]],
    uint v [[thread_position_in_threadgroup]]
) {
    if (k >= K || v >= V) return;

    const float d = decay[k];
    const float a = alphas[k];

    // load this thread's trace value into a register, keep it there
    float my_val = traces[k * V + v];

    for (uint t = 0; t < N; t++) {
        // snapshot pre-absorb: write into (N, V, K) layout
        states[(t * V + v) * K + k] = my_val;

        // decay
        my_val = my_val * d;

        // absorb: only the thread for the observed byte adds alpha
        uint b = (uint)bytes[t];
        if (v == b) {
            my_val = my_val + a;
        }
    }

    // write back final trace state
    traces[k * V + v] = my_val;
}

// fast_forward — DSP-reformulated forward pass.
//
// Computes logit[t, c] = sum_k alpha[k] * u[t, c, k]
// where u[t, c, k] = (1 - alpha[k]) * u[t-1, c, k] + Wp[c, byte_t, k]
//
// Launch: C threadgroups of K threads each. K=32 fits in one SIMD-group
// on Apple GPUs (M1+), so we use simd_sum() for the K-axis reduction —
// no threadgroup memory required.
//
// Threadgroup c handles one output channel; thread k maintains its u
// state in a register.
kernel void fast_forward(
    device const float* Wp       [[buffer(0)]],   // (C, V, K)
    device const uchar* bytes    [[buffer(1)]],   // (N,)
    device const float* alphas   [[buffer(2)]],   // (K,)
    device const float* decay    [[buffer(3)]],   // (K,)
    device const float* u_init   [[buffer(4)]],   // (C, K)
    device float*       logits   [[buffer(5)]],   // (N, C)
    device float*       u_final  [[buffer(6)]],   // (C, K)
    constant uint&      N        [[buffer(7)]],
    constant uint&      C        [[buffer(8)]],
    constant uint&      V        [[buffer(9)]],
    constant uint&      K        [[buffer(10)]],
    uint c [[threadgroup_position_in_grid]],
    uint k [[thread_position_in_threadgroup]]
) {
    if (c >= C || k >= K) return;

    const float a = alphas[k];
    const float d = decay[k];

    // load initial u for this (c, k) into a register
    float my_u = u_init[c * K + k];

    for (uint t = 0; t < N; t++) {
        // SIMD-group reduction across K threads — sums alpha*u across k.
        // K=32 fits in one simdgroup; for K<32 the unused lanes contribute
        // zero. simd_sum returns the same value to every lane in the group.
        float my_contrib = a * my_u;
        float logit_val = simd_sum(my_contrib);

        // thread 0 writes the final logit
        if (k == 0) {
            logits[t * C + c] = logit_val;
        }

        // all threads read Wp slice and update u
        uint b = (uint)bytes[t];
        // Wp[c, b, k] — layout (C, V, K), index c*V*K + b*K + k
        float wp_val = Wp[c * V * K + b * K + k];
        my_u = d * my_u + wp_val;
    }

    // write back final u
    u_final[c * K + k] = my_u;
}


// fast_backward — DSP-reformulated backward pass.
//
// For each i from N-1 down to 0:
//   grad_Wp[c, bytes[i], k] += alphas[k] * g[c, k]    where g represents
//                                                      sum over t > i
//   g[c, k] = errors[i, c] + decay[k] * g[c, k]       update g for next step
//
// Launch: C threadgroups of K threads each. Threadgroup c handles one
// output channel. Thread k maintains my_g for (c, k) in a register.
// Each thread accumulates into grad_Wp[c, bytes[i], k] — within a thread,
// the write is sequential across i (no race within threadgroup); across
// threadgroups, each c writes to its own row (no race across).
kernel void fast_backward(
    device const float* errors    [[buffer(0)]],  // (N, C)
    device const uchar* bytes     [[buffer(1)]],  // (N,)
    device const float* alphas    [[buffer(2)]],  // (K,)
    device const float* decay     [[buffer(3)]],  // (K,)
    device float*       grad_Wp   [[buffer(4)]],  // (C, V, K), accumulate
    constant uint&      N         [[buffer(5)]],
    constant uint&      C         [[buffer(6)]],
    constant uint&      V         [[buffer(7)]],
    constant uint&      K         [[buffer(8)]],
    uint c [[threadgroup_position_in_grid]],
    uint k [[thread_position_in_threadgroup]]
) {
    if (c >= C || k >= K) return;

    const float a = alphas[k];
    const float d = decay[k];

    // g_{N-1} = 0 (no t > N-1)
    float my_g = 0.0f;

    // walk i from N-1 down to 0
    for (int i = (int)N - 1; i >= 0; i--) {
        uint b = (uint)bytes[i];

        // accumulate into grad_Wp[c, b, k]
        // index: c*V*K + b*K + k
        grad_Wp[c * V * K + b * K + k] += a * my_g;

        // update my_g: g_{i-1} = errors[i, c] + decay * g_i
        float e = errors[i * C + c];
        my_g = e + d * my_g;
    }
}
// states is (N, V, K) layout. One thread per (n, v, k).
// Each thread reads its value and (if k < K-1) the next band's value,
// computes the difference, writes to features.
//
// Done as a separate kernel: in trace_scan_v2 adjacent bands live in
// different threadgroups, so we can't do this fusion in-kernel without
// cross-threadgroup synchronization (which Metal doesn't provide).
//
// Writes to a separate `features` buffer rather than in-place: this is
// the only safe way given the read-after-write dependency between
// adjacent k indices.
kernel void bandpass(
    device const float* states_in  [[buffer(0)]],
    device float*       features   [[buffer(1)]],
    constant uint&      N          [[buffer(2)]],
    constant uint&      K          [[buffer(3)]],
    constant uint&      V          [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = N * V * K;
    if (gid >= total) return;

    uint k = gid % K;
    uint v = (gid / K) % V;
    uint n = gid / (V * K);

    float val = states_in[(n * V + v) * K + k];
    if (k < K - 1) {
        float next_val = states_in[(n * V + v) * K + (k + 1)];
        features[(n * V + v) * K + k] = val - next_val;
    } else {
        features[(n * V + v) * K + k] = val;
    }
}
"""

_compiled_lib = None
_compile_failed = False
_dispatch_failed = False
_dispatch_strategy_v1 = None  # learned on first successful v1 call
_dispatch_strategy_v2 = None  # learned on first successful v2 call
_use_v2 = True  # default to the parallel-channel kernel


def _try_compile():
    """Compile the metal kernel. Returns the lib, or None if unavailable."""
    global _compiled_lib, _compile_failed
    if _compiled_lib is not None:
        return _compiled_lib
    if _compile_failed:
        return None

    if not (hasattr(torch, 'mps') and torch.backends.mps.is_available()):
        _compile_failed = True
        return None
    if not hasattr(torch.mps, 'compile_shader'):
        print("  (torch.mps.compile_shader not available — need PyTorch 2.5+)")
        _compile_failed = True
        return None

    try:
        _compiled_lib = torch.mps.compile_shader(_METAL_SCAN_SRC)
        return _compiled_lib
    except Exception as e:
        print(f"  (metal kernel compile failed: {e})")
        _compile_failed = True
        return None


def _try_dispatch_v1(lib, traces, states, bytes_t, decay, alphas, N, K, V):
    """Dispatch the v1 kernel: one thread per band."""
    global _dispatch_strategy_v1

    if _dispatch_strategy_v1 == 'auto':
        lib.trace_scan(traces, states, bytes_t, decay, alphas, N, K, V)
        return
    if _dispatch_strategy_v1 == 'threads':
        lib.trace_scan(traces, states, bytes_t, decay, alphas, N, K, V,
                       threads=K)
        return

    last_err = None
    for strat, kwargs in [
        ('auto', {}),
        ('threads', {'threads': K}),
    ]:
        try:
            lib.trace_scan(traces, states, bytes_t, decay, alphas, N, K, V,
                           **kwargs)
            _dispatch_strategy_v1 = strat
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"could not dispatch v1: {last_err}")


def _try_dispatch_v2(lib, traces, states, bytes_t, decay, alphas, N, K, V):
    """Dispatch the v2 kernel: K threadgroups of V threads each.

    The kernel uses [[threadgroup_position_in_grid]] for k and
    [[thread_position_in_threadgroup]] for v, so we need K total
    threadgroups and V threads per threadgroup.
    """
    global _dispatch_strategy_v2

    if _dispatch_strategy_v2 == 'tg_kwargs':
        lib.trace_scan_v2(
            traces, states, bytes_t, decay, alphas, N, K, V,
            threads=K * V,
            group_size=V,
        )
        return
    if _dispatch_strategy_v2 == 'group_size_threads':
        lib.trace_scan_v2(
            traces, states, bytes_t, decay, alphas, N, K, V,
            threads=K * V,
            threads_per_threadgroup=V,
        )
        return

    last_err = None
    for strat, kwargs in [
        # several plausible spellings of "K threadgroups of V threads each"
        ('group_size_threads',
            {'threads': K * V, 'threads_per_threadgroup': V}),
        ('tg_kwargs',
            {'threads': K * V, 'group_size': V}),
        ('thread_groups',
            {'thread_groups': K, 'threads_per_threadgroup': V}),
        ('grid_threadgroup',
            {'grid_size': (K, 1, 1), 'threadgroup_size': (V, 1, 1)}),
        ('grid_size_only',
            {'grid_size': (K * V, 1, 1)}),
    ]:
        try:
            lib.trace_scan_v2(
                traces, states, bytes_t, decay, alphas, N, K, V, **kwargs)
            _dispatch_strategy_v2 = strat
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"could not dispatch v2: {last_err}")


def trace_scan_metal(traces, bytes_tensor, decay, alphas):
    """Run the metal trace scan kernel.

    All inputs must be on the mps device with float32 dtype (uint8 for bytes).

        traces:       (K, V)  float32, modified in-place
        bytes_tensor: (N,)    uint8
        decay:        (K,)    float32
        alphas:       (K,)    float32

    Returns:
        states: (N, V, K) float32 — pre-absorb snapshot at each t,
        in (N, V, K) layout to match the feature layout downstream.
        OR None if the kernel is unavailable. Caller falls back.

    Tries the v2 (parallel-channel) kernel first; if its dispatch shape
    can't be expressed in this PyTorch version, falls back to v1.
    """
    global _dispatch_failed, _use_v2

    lib = _try_compile()
    if lib is None:
        return None
    if _dispatch_failed:
        return None

    K, V = traces.shape
    N = bytes_tensor.shape[0]

    states = torch.empty(N, V, K, dtype=torch.float32, device=traces.device)

    if _use_v2:
        try:
            _try_dispatch_v2(
                lib, traces, states, bytes_tensor, decay, alphas, N, K, V)
            return states
        except Exception as e:
            print(f"  (v2 dispatch failed, falling back to v1: {e})")
            _use_v2 = False

    try:
        _try_dispatch_v1(
            lib, traces, states, bytes_tensor, decay, alphas, N, K, V)
    except Exception as e:
        print(f"  (v1 dispatch also failed, falling back to torch: {e})")
        _dispatch_failed = True
        return None

    return states


_dispatch_strategy_bp = None


def _try_dispatch_bandpass(lib, states, features, N, K, V):
    """Dispatch the bandpass kernel."""
    global _dispatch_strategy_bp
    total = N * V * K

    if _dispatch_strategy_bp == 'auto':
        lib.bandpass(states, features, N, K, V)
        return
    if _dispatch_strategy_bp == 'threads':
        lib.bandpass(states, features, N, K, V, threads=total)
        return

    last_err = None
    for strat, kwargs in [
        ('auto', {}),
        ('threads', {'threads': total}),
    ]:
        try:
            lib.bandpass(states, features, N, K, V, **kwargs)
            _dispatch_strategy_bp = strat
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"could not dispatch bandpass: {last_err}")


def bandpass_metal(states):
    """Apply bandpass differencing to (N, V, K) states tensor.

    Returns features of the same shape, with feature[..., k] =
    states[..., k] - states[..., k+1] for k<K-1, and
    feature[..., K-1] = states[..., K-1].

    Returns None if the kernel is unavailable.
    """
    lib = _try_compile()
    if lib is None:
        return None
    if not hasattr(lib, 'bandpass'):
        return None

    N, V, K = states.shape
    features = torch.empty_like(states)

    try:
        _try_dispatch_bandpass(lib, states, features, N, K, V)
    except Exception as e:
        print(f"  (bandpass kernel dispatch failed: {e})")
        return None

    return features


def bandpass_torch(states):
    """Pytorch fallback for bandpass."""
    features = torch.empty_like(states)
    features[..., :-1] = states[..., :-1] - states[..., 1:]
    features[..., -1] = states[..., -1]
    return features


def bandpass_W_torch(W):
    """K-axis bandpass on weight tensor for the DSP-reformulated path.

    W' is the bandpass of W. Used to rewrite the forward as IIR-on-W'.
    Operates on the K (last) axis.

        W'[..., 0] = W[..., 0]
        W'[..., k] = W[..., k] - W[..., k-1]   for k>=1
    """
    Wp = torch.empty_like(W)
    Wp[..., 0] = W[..., 0]
    Wp[..., 1:] = W[..., 1:] - W[..., :-1]
    return Wp


def bandpass_T_W_torch(Wp):
    """Transpose-bandpass: maps gradient w.r.t. W' back to gradient w.r.t. W.

    For Wp = bandpass(W), grad_W = bandpass^T(grad_Wp). The transpose of the
    bandpass on the K axis is K-axis differencing in the *opposite* direction.

        [B^T g'][..., k]   = g'[..., k] - g'[..., k+1]   for k < K-1
        [B^T g'][..., K-1] = g'[..., K-1]
    """
    g = torch.empty_like(Wp)
    g[..., :-1] = Wp[..., :-1] - Wp[..., 1:]
    g[..., -1] = Wp[..., -1]
    return g


# ───────────────── DSP-reformulated kernels ─────────────────

_dispatch_strategy_ff = None
_dispatch_strategy_bb = None


def _try_dispatch_fast_forward(lib, Wp, bytes_t, alphas, decay, u_init,
                                logits, u_final, N, C, V, K):
    """Dispatch the fast_forward kernel: C threadgroups of K threads each."""
    global _dispatch_strategy_ff
    args = (Wp, bytes_t, alphas, decay, u_init, logits, u_final,
            N, C, V, K)

    if _dispatch_strategy_ff == 'group_size':
        lib.fast_forward(*args, threads=C * K, group_size=K)
        return

    last_err = None
    for strat, kwargs in [
        ('group_size', {'threads': C * K, 'group_size': K}),
        ('tg', {'threads': C * K, 'threads_per_threadgroup': K}),
    ]:
        try:
            lib.fast_forward(*args, **kwargs)
            _dispatch_strategy_ff = strat
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"could not dispatch fast_forward: {last_err}")


def _try_dispatch_fast_backward(lib, errors, bytes_t, alphas, decay,
                                 grad_Wp, N, C, V, K):
    """Dispatch fast_backward: C threadgroups of K threads each."""
    global _dispatch_strategy_bb
    args = (errors, bytes_t, alphas, decay, grad_Wp, N, C, V, K)

    if _dispatch_strategy_bb == 'group_size':
        lib.fast_backward(*args, threads=C * K, group_size=K)
        return

    last_err = None
    for strat, kwargs in [
        ('group_size', {'threads': C * K, 'group_size': K}),
        ('tg', {'threads': C * K, 'threads_per_threadgroup': K}),
    ]:
        try:
            lib.fast_backward(*args, **kwargs)
            _dispatch_strategy_bb = strat
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"could not dispatch fast_backward: {last_err}")


def fast_forward_metal(Wp, bytes_tensor, alphas, decay, u_init):
    """One-dispatch forward pass via the DSP reformulation.

    Args:
        Wp:           (C, V, K) bandpassed weights, fp32, on mps
        bytes_tensor: (N,) uint8, on mps
        alphas:       (K,) fp32, on mps
        decay:        (K,) fp32, on mps  (= 1 - alphas)
        u_init:       (C, K) fp32, on mps  — initial u state

    Returns:
        (logits, u_final): logits (N, C), u_final (C, K)
        OR None if kernel unavailable.
    """
    lib = _try_compile()
    if lib is None:
        return None
    if not hasattr(lib, 'fast_forward'):
        return None

    C, V, K = Wp.shape
    N = bytes_tensor.shape[0]
    device = Wp.device

    logits = torch.empty(N, C, dtype=torch.float32, device=device)
    u_final = torch.empty(C, K, dtype=torch.float32, device=device)

    try:
        _try_dispatch_fast_forward(
            lib, Wp, bytes_tensor, alphas, decay, u_init,
            logits, u_final, N, C, V, K)
    except Exception as e:
        print(f"  (fast_forward dispatch failed: {e})")
        return None

    return logits, u_final


def fast_backward_metal(errors, bytes_tensor, alphas, decay, V):
    """One-dispatch backward pass.

    Args:
        errors:       (N, C) fp32, on mps
        bytes_tensor: (N,) uint8, on mps
        alphas:       (K,) fp32, on mps
        decay:        (K,) fp32, on mps
        V:            int (vocab size)

    Returns:
        grad_Wp: (C, V, K) fp32 — gradient accumulated into W'-space.
        OR None if kernel unavailable.
    """
    lib = _try_compile()
    if lib is None:
        return None
    if not hasattr(lib, 'fast_backward'):
        return None

    N, C = errors.shape
    K = alphas.shape[0]
    device = errors.device

    grad_Wp = torch.zeros(C, V, K, dtype=torch.float32, device=device)

    try:
        _try_dispatch_fast_backward(
            lib, errors, bytes_tensor, alphas, decay,
            grad_Wp, N, C, V, K)
    except Exception as e:
        print(f"  (fast_backward dispatch failed: {e})")
        return None

    return grad_Wp


def trace_scan_torch(traces, bytes_tensor, decay, alphas):
    """Pure-PyTorch fallback. Same math as the kernel.

        traces:       (K, V)  modified in-place
        bytes_tensor: (N,)    int/uint
        decay:        (K,)
        alphas:       (K,)

    Returns:
        states: (N, V, K) — pre-absorb snapshots, in (N, V, K) layout
        matching the metal kernels.
    """
    K, V = traces.shape
    N = bytes_tensor.shape[0]
    device = traces.device
    dtype = traces.dtype

    # store in (K, V) per-timestep then transpose at write time. We do this
    # to keep the inner-loop logic straightforward and identical in form to
    # the kernels: tracking traces in (K, V), only differing in output layout.
    states = torch.empty(N, V, K, dtype=dtype, device=device)
    decay_col = decay.unsqueeze(1)  # (K, 1)
    bytes_long = bytes_tensor.long()

    for t in range(N):
        # snapshot in (V, K) layout for output
        states[t] = traces.T
        traces.mul_(decay_col)
        traces[:, bytes_long[t]] += alphas

    return states


# ─────────────── uniform fast-path API ───────────────
# These aliases match the names used by _impl_torch and _impl_triton so
# the dispatcher can use any backend interchangeably.

def bandpass_W(W):
    """K-axis bandpass on weight tensor (uniform-API alias)."""
    return bandpass_W_torch(W)


def bandpass_T_W(Wp):
    """Transpose-bandpass for gradient mapping (uniform-API alias)."""
    return bandpass_T_W_torch(Wp)


def fast_forward(Wp, bytes_tensor, alphas, decay, u_init):
    """Metal fast forward, with PyTorch fallback if kernel unavailable."""
    result = fast_forward_metal(Wp, bytes_tensor, alphas, decay, u_init)
    if result is None:
        # kernel unavailable — fall back to portable impl
        import _impl_torch
        return _impl_torch.fast_forward(
            Wp, bytes_tensor, alphas, decay, u_init)
    return result


def fast_backward(errors, bytes_tensor, alphas, decay, V):
    """Metal fast backward, with PyTorch fallback if kernel unavailable."""
    result = fast_backward_metal(errors, bytes_tensor, alphas, decay, V)
    if result is None:
        import _impl_torch
        return _impl_torch.fast_backward(
            errors, bytes_tensor, alphas, decay, V)
    return result


def is_available(device):
    """Metal kernels require MPS device and the kernel must compile."""
    if device.type != 'mps':
        return False
    return _try_compile() is not None
