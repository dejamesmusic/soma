"""
Triton implementation of the soma fast-path operations.

Targets CUDA GPUs via Triton. Mirrors the Metal kernels in _impl_metal.py
algorithmically — same DSP reformulation, same parallelization shape.

⚠️ UNTESTED ⚠️
This module was written without access to CUDA hardware. The algorithm
is verified equivalent (matches the Metal version which is bit-tested),
but the Triton-specific dispatch syntax, kernel launch parameters, and
edge cases have not been validated on a real GPU.

If you have CUDA hardware and find issues:
  1. Run test_soma_fast.py — it validates against the bit-exact PyTorch
     reference, so kernel bugs will show up as numeric divergence.
  2. The forward kernel parallelizes one program-per-output-channel with
     K lanes per program. K=32 fits one warp on NVIDIA GPUs.
  3. The backward kernel writes into grad_Wp[c, b, k] sequentially
     within each program (per output channel). No atomics needed since
     each program owns its own row of grad_Wp.

If Triton is unavailable, falls back to _impl_torch.

API matches _impl_torch.py and _impl_metal.py.
"""

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# Same as torch implementation — small, no CUDA-specific optimization needed.
# These run as pytorch ops which on cuda will be dispatched to cuDNN/cuBLAS
# anyway.
def bandpass_W(W):
    Wp = torch.empty_like(W)
    Wp[..., 0] = W[..., 0]
    Wp[..., 1:] = W[..., 1:] - W[..., :-1]
    return Wp


def bandpass_T_W(Wp):
    g = torch.empty_like(Wp)
    g[..., :-1] = Wp[..., :-1] - Wp[..., 1:]
    g[..., -1] = Wp[..., -1]
    return g


if _TRITON_AVAILABLE:

    @triton.jit
    def _fast_forward_kernel(
        Wp_ptr,       # (C, V, K) fp32
        bytes_ptr,    # (N,) uint8
        alphas_ptr,   # (K,) fp32
        decay_ptr,    # (K,) fp32
        u_init_ptr,   # (C, K) fp32
        logits_ptr,   # (N, C) fp32 — output
        u_final_ptr,  # (C, K) fp32 — output
        N: tl.constexpr,
        C: tl.constexpr,
        V: tl.constexpr,
        K: tl.constexpr,
    ):
        """One program per output channel c. Within a program, K lanes
        each hold their own u[c, k] in a register.
        """
        c = tl.program_id(0)

        # K-lane vectors
        k_arange = tl.arange(0, K)

        # Load alphas, decay (broadcast across all programs)
        a = tl.load(alphas_ptr + k_arange)
        d = tl.load(decay_ptr + k_arange)

        # Load this program's initial u state from u_init[c, :]
        u = tl.load(u_init_ptr + c * K + k_arange)

        # Walk N timesteps
        for t in range(N):
            # logit_t_c = sum_k alphas[k] * u[c, k]
            contrib = a * u
            logit_val = tl.sum(contrib, axis=0)
            # write logit[t, c]
            tl.store(logits_ptr + t * C + c, logit_val)

            # read byte_t (uint8 → cast)
            b = tl.load(bytes_ptr + t).to(tl.int32)
            # read Wp[c, b, :K]
            wp_slice = tl.load(Wp_ptr + c * V * K + b * K + k_arange)
            # update u: u = decay * u + wp_slice
            u = d * u + wp_slice

        # write final u
        tl.store(u_final_ptr + c * K + k_arange, u)

    @triton.jit
    def _fast_backward_kernel(
        errors_ptr,   # (N, C) fp32
        bytes_ptr,    # (N,) uint8
        alphas_ptr,   # (K,) fp32
        decay_ptr,    # (K,) fp32
        grad_Wp_ptr,  # (C, V, K) fp32 — output, must be pre-zeroed
        N: tl.constexpr,
        C: tl.constexpr,
        V: tl.constexpr,
        K: tl.constexpr,
    ):
        """One program per output channel c. K lanes per program.
        Each program walks errors backward through time, accumulating
        into its own row of grad_Wp.
        """
        c = tl.program_id(0)

        k_arange = tl.arange(0, K)

        a = tl.load(alphas_ptr + k_arange)
        d = tl.load(decay_ptr + k_arange)

        # g_{N-1} = 0
        g = tl.zeros((K,), dtype=tl.float32)

        # walk i from N-1 down to 0
        for i in range(N - 1, -1, -1):
            b = tl.load(bytes_ptr + i).to(tl.int32)
            # grad_Wp[c, b, :] += alphas * g  (no race: this program owns row c)
            addr = grad_Wp_ptr + c * V * K + b * K + k_arange
            current = tl.load(addr)
            tl.store(addr, current + a * g)

            # update g: g_{i-1} = errors[i, c] + decay * g
            e = tl.load(errors_ptr + i * C + c)
            g = e + d * g


def fast_forward(Wp, bytes_tensor, alphas, decay, u_init):
    """Triton implementation.

    Falls back to _impl_torch if Triton is unavailable.
    """
    if not _TRITON_AVAILABLE:
        from . import _impl_torch
        return _impl_torch.fast_forward(Wp, bytes_tensor, alphas, decay, u_init)

    C, V, K = Wp.shape
    N = bytes_tensor.shape[0]
    device = Wp.device
    dtype = Wp.dtype

    logits = torch.empty(N, C, dtype=dtype, device=device)
    u_final = torch.empty(C, K, dtype=dtype, device=device)

    # ensure bytes are uint8 contiguous
    bytes_t = bytes_tensor.to(torch.uint8).contiguous()

    grid = (C,)
    _fast_forward_kernel[grid](
        Wp, bytes_t, alphas, decay, u_init,
        logits, u_final,
        N, C, V, K,
    )

    return logits, u_final


def fast_backward(errors, bytes_tensor, alphas, decay, V):
    """Triton implementation."""
    if not _TRITON_AVAILABLE:
        from . import _impl_torch
        return _impl_torch.fast_backward(errors, bytes_tensor, alphas, decay, V)

    N, C = errors.shape
    K = alphas.shape[0]
    device = errors.device
    dtype = errors.dtype

    grad_Wp = torch.zeros(C, V, K, dtype=dtype, device=device)
    bytes_t = bytes_tensor.to(torch.uint8).contiguous()

    grid = (C,)
    _fast_backward_kernel[grid](
        errors, bytes_t, alphas, decay, grad_Wp,
        N, C, V, K,
    )

    return grad_Wp


def is_available(device):
    """Triton path requires CUDA + Triton installed."""
    return _TRITON_AVAILABLE and device.type == 'cuda'
