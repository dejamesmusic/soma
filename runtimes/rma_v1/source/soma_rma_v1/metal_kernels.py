"""fused bounded-state kernels for soma rma v1.

the dense torch implementations are the executable specification. metal is an
optional acceleration path and must remain numerically equivalent at the
controller boundary.
"""

from __future__ import annotations

import math

import torch


PHI = (1.0 + math.sqrt(5.0)) / 2.0
MAX_BANDS = 64


_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// one threadgroup per categorical coordinate, one thread per temporal band.
// each thread retains its trace scalar in a register throughout the batch.
// only q x (k-1) squared band energies leave the kernel.
kernel void residual_trace_spectrum(
    device const float* probabilities [[buffer(0)]], // (n, v)
    device const int*   targets       [[buffer(1)]], // (n,)
    device const float* weights       [[buffer(2)]], // (n,)
    device float*       traces        [[buffer(3)]], // (v, k), persistent
    device atomic_float* energy       [[buffer(4)]], // (ceil(n/q), k-1)
    device const float* decay         [[buffer(5)]], // (k,)
    device const float* alpha         [[buffer(6)]], // (k,)
    constant uint& n                  [[buffer(7)]],
    constant uint& v_count            [[buffer(8)]],
    constant uint& k_count            [[buffer(9)]],
    constant uint& quantum            [[buffer(10)]],
    uint v [[threadgroup_position_in_grid]],
    uint k [[thread_position_in_threadgroup]]
) {
    if (v >= v_count || k >= k_count) return;

    threadgroup float shared_trace[64];
    float state = traces[v * k_count + k];
    uint observation = 0;

    for (uint t = 0; t < n; ++t) {
        float error = -probabilities[t * v_count + v];
        if ((uint)targets[t] == v) error += 1.0f;
        error *= weights[t];
        state = decay[k] * state + alpha[k] * error;

        bool boundary = (((t + 1) % quantum) == 0) || (t + 1 == n);
        if (boundary) {
            shared_trace[k] = state;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (k + 1 < k_count) {
                float difference = state - shared_trace[k + 1];
                atomic_fetch_add_explicit(
                    &energy[observation * (k_count - 1) + k],
                    difference * difference,
                    memory_order_relaxed
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            observation += 1;
        }
    }
    traces[v * k_count + k] = state;
}

// Reduce <gradient, weight> and ||gradient||^2 for one matrix.  One atomic
// addition per threadgroup avoids contention across individual elements.
kernel void matrix_sphere_stats(
    device const float* weight [[buffer(0)]],
    device const float* gradient [[buffer(1)]],
    device atomic_float* stats [[buffer(2)]], // [dot, gradient_square]
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]]
) {
    threadgroup float local_dot[256];
    threadgroup float local_square[256];
    float w = gid < count ? weight[gid] : 0.0f;
    float g = gid < count ? gradient[gid] : 0.0f;
    local_dot[lane] = w * g;
    local_square[lane] = g * g;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_width / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            local_dot[lane] += local_dot[lane + stride];
            local_square[lane] += local_square[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        atomic_fetch_add_explicit(&stats[0], local_dot[0], memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[1], local_square[0], memory_order_relaxed);
    }
}

// Deterministic one-pass statistics for the sphere update: <w,g>, ||g||^2
// and ||w||^2 from a single read of each matrix.  Every thread accumulates a
// fixed set of elements in a fixed order, each threadgroup reduces them by a
// fixed tree and writes its partials; the host sums partials in float64.
// No atomics, so the result is identical from run to run.
kernel void matrix_sphere_partials(
    device const float* weight [[buffer(0)]],
    device const float* gradient [[buffer(1)]],
    device float* partials [[buffer(2)]],        // (all groups, 3)
    constant uint& count [[buffer(3)]],
    constant uint& groups [[buffer(4)]],
    constant uint& base [[buffer(5)]],           // this matrix's first group
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    threadgroup float local_dot[256];
    threadgroup float local_gradient[256];
    threadgroup float local_weight[256];
    float dot = 0.0f, gradient_square = 0.0f, weight_square = 0.0f;
    uint stride = groups * 256u;
    for (uint i = gid; i < count; i += stride) {
        float w = weight[i];
        float g = gradient[i];
        dot += w * g;
        gradient_square += g * g;
        weight_square += w * w;
    }
    local_dot[lane] = dot;
    local_gradient[lane] = gradient_square;
    local_weight[lane] = weight_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half_width = 128u; half_width > 0u; half_width >>= 1) {
        if (lane < half_width) {
            local_dot[lane] += local_dot[lane + half_width];
            local_gradient[lane] += local_gradient[lane + half_width];
            local_weight[lane] += local_weight[lane + half_width];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        uint row = (base + group) * 3u;
        partials[row] = local_dot[0];
        partials[row + 1u] = local_gradient[0];
        partials[row + 2u] = local_weight[0];
    }
}

// Exact retraction of a tangent SGD step onto a fixed Frobenius sphere.
// Tangency gives ||w-rate*g_t||^2 = radius^2 + rate^2*||g_t||^2,
// so the normalization is known before the update and needs no second pass.
kernel void matrix_sphere_update(
    device float* weight [[buffer(0)]],
    device const float* gradient [[buffer(1)]],
    constant float& dot_over_radius_square [[buffer(2)]],
    constant float& rate [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float w = weight[gid];
    float tangent = gradient[gid] - dot_over_radius_square * w;
    weight[gid] = scale * (w - rate * tangent);
}
"""


_library = None
_compile_error = None
_dispatch_style = None
_stats_dispatch_style = None
_update_dispatch_style = None


def coefficients(bands: int, *, base: float = PHI, device="cpu"):
    alpha = torch.tensor(
        [base ** -index for index in range(int(bands))],
        dtype=torch.float32,
        device=device,
    )
    return alpha, 1.0 - alpha


def _compile():
    global _library, _compile_error
    if _library is not None:
        return _library
    if _compile_error is not None:
        return None
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        _compile_error = "torch mps shader compilation is unavailable"
        return None
    try:
        _library = torch.mps.compile_shader(_SOURCE)
    except Exception as error:  # pragma: no cover - hardware/compiler specific
        _compile_error = str(error)
        return None
    return _library


def metal_available():
    return _compile() is not None


def compile_error():
    _compile()
    return _compile_error


def _dispatch(kernel, arguments, threads, group_size):
    global _dispatch_style
    if _dispatch_style == "group_size":
        kernel(*arguments, threads=threads, group_size=group_size)
        return
    if _dispatch_style == "threads_per_threadgroup":
        kernel(
            *arguments, threads=threads,
            threads_per_threadgroup=group_size,
        )
        return
    failures = []
    for name, options in (
        ("group_size", {"threads": threads, "group_size": group_size}),
        ("threads_per_threadgroup", {
            "threads": threads, "threads_per_threadgroup": group_size,
        }),
    ):
        try:
            kernel(*arguments, **options)
            _dispatch_style = name
            return
        except Exception as error:  # pragma: no cover - version specific
            failures.append(f"{name}: {error}")
    raise RuntimeError("metal dispatch failed: " + " | ".join(failures))


def residual_spectrum_torch(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    traces: torch.Tensor,
    decay: torch.Tensor,
    alpha: torch.Tensor,
    quantum: int = 16,
):
    """dense specification; updates `(v, k)` traces in place."""
    if traces.ndim != 2:
        raise ValueError("traces must have shape (vocabulary, bands)")
    n, vocab = probabilities.shape
    if traces.shape[0] != vocab or traces.shape[1] != len(alpha):
        raise ValueError("probability and trace shapes disagree")
    observations = []
    for start in range(0, n, quantum):
        end = min(n, start + quantum)
        for position in range(start, end):
            error = -probabilities[position]
            error = error.clone()
            error[targets[position]] += 1.0
            error.mul_(weights[position])
            traces.mul_(decay.unsqueeze(0)).add_(
                error.unsqueeze(1) * alpha.unsqueeze(0)
            )
        difference = traces[:, :-1] - traces[:, 1:]
        observations.append(difference.square().sum(dim=0))
    if not observations:
        return traces.new_empty((0, traces.shape[1] - 1))
    return torch.stack(observations)


def residual_spectrum_metal(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    traces: torch.Tensor,
    decay: torch.Tensor,
    alpha: torch.Tensor,
    quantum: int = 16,
):
    """fused exact-categorical residual update and band-energy reduction."""
    library = _compile()
    if library is None:
        return None
    if probabilities.device.type != "mps":
        raise ValueError("metal inputs must be on mps")
    probabilities = probabilities.contiguous().float()
    targets = targets.contiguous().to(dtype=torch.int32)
    weights = weights.contiguous().float()
    traces = traces.contiguous()
    decay = decay.contiguous().float()
    alpha = alpha.contiguous().float()
    n, vocab = probabilities.shape
    bands = traces.shape[1]
    if bands > MAX_BANDS:
        raise ValueError(f"metal kernel supports at most {MAX_BANDS} bands")
    if tuple(traces.shape) != (vocab, len(alpha)):
        raise ValueError("probability and trace shapes disagree")
    count = (n + int(quantum) - 1) // int(quantum)
    energy = torch.zeros(
        count, bands - 1, dtype=torch.float32, device=probabilities.device
    )
    arguments = (
        probabilities, targets, weights, traces, energy, decay, alpha,
        n, vocab, bands, int(quantum),
    )
    _dispatch(
        library.residual_trace_spectrum,
        arguments,
        threads=vocab * bands,
        group_size=bands,
    )
    return energy


class ResidualSpectrumState:
    """bounded persistent residual state with transparent acceleration."""

    def __init__(self, vocab, bands=32, *, base=PHI, device="mps", quantum=16):
        self.vocab = int(vocab)
        self.bands = int(bands)
        self.quantum = int(quantum)
        self.device = torch.device(device)
        self.alpha, self.decay = coefficients(
            self.bands, base=base, device=self.device
        )
        self.traces = torch.zeros(
            self.vocab, self.bands, dtype=torch.float32, device=self.device
        )

    def reset(self):
        self.traces.zero_()

    def advance(self, probabilities, targets, weights):
        if self.device.type == "mps":
            result = residual_spectrum_metal(
                probabilities, targets, weights, self.traces,
                self.decay, self.alpha, self.quantum,
            )
            if result is not None:
                return result
        return residual_spectrum_torch(
            probabilities, targets, weights, self.traces,
            self.decay, self.alpha, self.quantum,
        )


def _dispatch_stats(kernel, arguments, count):
    global _stats_dispatch_style
    group = 256
    threads = ((int(count) + group - 1) // group) * group
    if _stats_dispatch_style == "group_size":
        kernel(*arguments, threads=threads, group_size=group)
        return
    if _stats_dispatch_style == "threads_per_threadgroup":
        kernel(*arguments, threads=threads, threads_per_threadgroup=group)
        return
    failures = []
    for name, options in (
        ("group_size", {"threads": threads, "group_size": group}),
        ("threads_per_threadgroup", {
            "threads": threads, "threads_per_threadgroup": group,
        }),
    ):
        try:
            kernel(*arguments, **options)
            _stats_dispatch_style = name
            return
        except Exception as error:  # pragma: no cover
            failures.append(f"{name}: {error}")
    raise RuntimeError("matrix stats dispatch failed: " + " | ".join(failures))


def matrix_sphere_statistics(parameter, gradient):
    """return fp32 `[<g,w>, ||g||^2]` using one fused reduction."""
    library = _compile()
    if library is None or parameter.device.type != "mps":
        flat_parameter = parameter.detach().float().reshape(-1)
        flat_gradient = gradient.detach().float().reshape(-1)
        return torch.stack((
            (flat_parameter * flat_gradient).sum(),
            flat_gradient.square().sum(),
        ))
    if not parameter.is_contiguous() or not gradient.is_contiguous():
        raise ValueError("sphere kernel requires contiguous tensors")
    stats = torch.zeros(2, dtype=torch.float32, device=parameter.device)
    count = parameter.numel()
    _dispatch_stats(
        library.matrix_sphere_stats,
        (parameter, gradient, stats, count),
        count,
    )
    return stats


SPHERE_GROUPS = 1024


def sphere_statistics(parameters, extra=None):
    """[<w,g>, ||g||^2, ||w||^2] per matrix, as host float64 lists.

    On MPS one deterministic kernel pass per matrix fills a shared buffer of
    per-group partials, summed on the host in float64.  `extra` is an optional
    1-d device tensor sent back in the same transfer; its host values are
    returned second.
    """
    parameters = list(parameters)
    library = _compile()
    on_metal = (
        library is not None and parameters
        and all(p.device.type == "mps" and p.is_contiguous()
                and p.grad is not None and p.grad.is_contiguous()
                and p.dtype == torch.float32 for p in parameters)
    )
    if not on_metal:
        stacked = torch.stack([
            torch.stack((
                (p.detach() * p.grad.detach()).sum(),
                p.grad.detach().square().sum(),
                p.detach().square().sum(),
            ))
            for p in parameters
        ]) if parameters else torch.zeros(0, 3)
        host = torch.cat((stacked.reshape(-1).to(extra.device), extra)) if extra is not None and len(parameters) \
            else (stacked.reshape(-1) if extra is None else extra)
        host = host.detach().cpu().double().tolist()
        statistics = [host[3 * i:3 * i + 3] for i in range(len(parameters))]
        return statistics, host[3 * len(parameters):]
    counts = [p.numel() for p in parameters]
    groups = [max(1, min(SPHERE_GROUPS, (n + 255) // 256)) for n in counts]
    partials = torch.empty(
        sum(groups) * 3 + (0 if extra is None else extra.numel()),
        dtype=torch.float32, device=parameters[0].device,
    )
    base = 0
    for parameter, count, group_count in zip(parameters, counts, groups):
        _dispatch(
            library.matrix_sphere_partials,
            (parameter.detach(), parameter.grad.detach(), partials,
             count, group_count, base),
            group_count * 256, 256,
        )
        base += group_count
    if extra is not None:
        partials[sum(groups) * 3:] = extra
    host = partials.cpu().double().numpy()
    statistics = []
    base = 0
    for group_count in groups:
        rows = host[base * 3:(base + group_count) * 3].reshape(group_count, 3)
        statistics.append(rows.sum(axis=0).tolist())
        base += group_count
    return statistics, host[sum(groups) * 3:].tolist()


def matrix_sphere_update_(
    parameter, gradient, *, rate, radius, dot, tangent_square,
    norm_square=None,
):
    """apply the exact compact tangent step in place.

    With `norm_square`, the parameter's measured squared norm, the tangent
    projection and the retraction start from where the parameter actually
    is, so every step lands back on `radius` and rounding cannot accumulate
    into drift.  Without it the parameter is assumed to sit on the radius.
    """
    norm_square = (
        float(radius) ** 2 if norm_square is None else float(norm_square)
    )
    dot_over_radius_square = float(dot) / max(norm_square, 1e-20)
    scale = float(radius) / math.sqrt(
        max(norm_square + float(rate) ** 2 * float(tangent_square), 1e-20)
    )
    library = _compile()
    if library is None or parameter.device.type != "mps":
        with torch.no_grad():
            tangent = gradient - dot_over_radius_square * parameter
            parameter.copy_(scale * (parameter - float(rate) * tangent))
        return
    global _update_dispatch_style
    arguments = (
        parameter, gradient, dot_over_radius_square, float(rate), scale,
        parameter.numel(),
    )
    if _update_dispatch_style == "auto":
        library.matrix_sphere_update(*arguments)
        return
    if _update_dispatch_style == "threads":
        library.matrix_sphere_update(*arguments, threads=parameter.numel())
        return
    failures = []
    for name, options in (
        ("auto", {}),
        ("threads", {"threads": parameter.numel()}),
    ):
        try:
            library.matrix_sphere_update(*arguments, **options)
            _update_dispatch_style = name
            return
        except Exception as error:  # pragma: no cover
            failures.append(f"{name}: {error}")
    raise RuntimeError("matrix update dispatch failed: " + " | ".join(failures))
