"""
Pure-PyTorch implementation of the soma fast-path operations.

This module provides the four functions required by the fast-path training
loop, implemented in plain PyTorch. Works on any device (cpu, mps, cuda).
Bit-equivalent (modulo float ordering noise) to the device-specific kernel
implementations.

This is the universal fallback. It's slower than the Metal kernels because
it materializes intermediate tensors and walks the time loop in Python,
but it's correct everywhere and has no compilation step.

API (used by SOMA._fast_train_batch_*):

    bandpass_W(W)             → Wp          # K-axis bandpass on weights
    bandpass_T_W(Wp)          → grad_W      # transpose-bandpass for gradient map
    fast_forward(Wp, bytes_t, alphas, decay, u_init)
                              → (logits, u_final)
    fast_backward(errors, bytes_t, alphas, decay, V)
                              → grad_Wp
"""

import torch


def bandpass_W(W):
    """K-axis bandpass on a weight tensor.

    W'[..., 0]  = W[..., 0]
    W'[..., k]  = W[..., k] - W[..., k-1]   for k >= 1

    Operates on the last (K) axis, broadcasts over leading dims.
    """
    Wp = torch.empty_like(W)
    Wp[..., 0] = W[..., 0]
    Wp[..., 1:] = W[..., 1:] - W[..., :-1]
    return Wp


def bandpass_T_W(Wp):
    """Transpose of the K-axis bandpass: maps gradient w.r.t. W'
    back to gradient w.r.t. W.

    [B^T g'][..., k]   = g'[..., k] - g'[..., k+1]   for k < K-1
    [B^T g'][..., K-1] = g'[..., K-1]
    """
    g = torch.empty_like(Wp)
    g[..., :-1] = Wp[..., :-1] - Wp[..., 1:]
    g[..., -1] = Wp[..., -1]
    return g


def fast_forward(Wp, bytes_tensor, alphas, decay, u_init):
    """DSP-reformulated forward pass.

    Computes:
        u[t][c, k] = decay[k] * u[t-1][c, k] + Wp[c, byte_t, k]
        logit[t, c] = sum_k alphas[k] * u[t][c, k]

    Args:
        Wp:           (C, V, K) bandpassed weights, fp32
        bytes_tensor: (N,) integer byte values
        alphas:       (K,) fp32
        decay:        (K,) fp32, = 1 - alphas
        u_init:       (C, K) fp32, initial state

    Returns:
        (logits, u_final): logits (N, C), u_final (C, K)

    Implementation: Python loop over time. Each step is a vectorized
    PyTorch op over the (C, K) state. Slower than a kernel because of
    per-step dispatch overhead, but works on any device.
    """
    C, V, K = Wp.shape
    N = bytes_tensor.shape[0]
    device = Wp.device
    dtype = Wp.dtype

    bytes_long = bytes_tensor.long()
    u = u_init.clone()
    logits = torch.empty(N, C, dtype=dtype, device=device)

    for t in range(N):
        # logit at time t uses u BEFORE absorbing byte_t
        logits[t] = (alphas * u).sum(dim=1)
        # absorb byte_t: u[c, k] += Wp[c, b_t, k], with decay first
        b = int(bytes_long[t])
        u = decay * u + Wp[:, b, :]

    return logits, u


def fast_backward(errors, bytes_tensor, alphas, decay, V):
    """DSP-reformulated backward pass.

    Computes:
        grad_Wp[c, v, k] = alphas[k] * sum_i 1[bytes_i = v] * g_i[c, k]
        where g_i[c, k] = decay[k] * g_{i+1}[c, k] + errors[i+1, c]
        with g_{N-1} = 0 (initial value when stepping back from N-1)

    Args:
        errors:       (N, C) fp32 — gradient signal at each time step
        bytes_tensor: (N,) integer byte values
        alphas:       (K,) fp32
        decay:        (K,) fp32 = 1 - alphas
        V:            int, vocabulary size (V dim of output)

    Returns:
        grad_Wp: (C, V, K) — accumulated gradient w.r.t. bandpassed weights

    Implementation: Python loop walking backwards through time. At each
    step accumulates alpha · g into one column of grad_Wp (the column
    indexed by bytes_tensor[i]).
    """
    N, C = errors.shape
    K = alphas.shape[0]
    device = errors.device
    dtype = errors.dtype

    bytes_long = bytes_tensor.long()
    grad_Wp = torch.zeros(C, V, K, dtype=dtype, device=device)
    g = torch.zeros(C, K, dtype=dtype, device=device)

    for i in range(N - 1, -1, -1):
        b = int(bytes_long[i])
        # grad_Wp[:, b, :] += alphas * g  (vectorized over C, K)
        grad_Wp[:, b, :] += alphas * g
        # update g for the next-earlier index: g_{i-1} = errors[i] + decay * g_i
        g = errors[i].unsqueeze(1) + decay * g

    return grad_Wp


def is_available(device):
    """Pure PyTorch always available — used by the dispatcher."""
    return True
