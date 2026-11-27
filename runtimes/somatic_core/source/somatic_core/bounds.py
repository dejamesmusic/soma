"""Nothing grows (James, Nov 8: "no number can grow unbounded in a continual learner. weights can't grow indefinitely").

Every number the learner carries, and what bounds it:
  learned matrices      soft radial compression after each step (soma v14 trace_branch_mixer, unchanged): per row,
                        ceiling = max(1, 0.1 sqrt(fan_in)); outward norm growth is scaled by max(0.01, 1/(1 + r^4)),
                        r = previous norm / ceiling; a row at or below the ceiling cannot cross it; direction is free.
  1-d parameters        (biases, LayerNorm gains) the same rule on the whole vector with a soft RMS ceiling of 2
                        (v14 bounds only scalars and relies on weight decay for vectors).
  optimiser             AdamW (weight decay 1e-4, v14) and global gradient clipping at 1.0; Adam's step count is
                        capped (its bias correction is 1 long before), so no counter in the optimiser grows.
  gate / mix weights    theta, rung weights, match weights clamped to [-8, 8] (v14's scalar-gate bound).
  counts                decayed rungs are bounded by 1/alpha; the slowest rung halves at alltime_cap; the CPU per-pair
                        all-time counts halve with it.
  clocks                every time stamp is rebased when the token clock passes REBASE: store clock, node stamps,
                        order-0 stamp, chunk index, pool and pair stamps, and each context's match history (rebased by
                        whole rings, so ring positions are unchanged). Only differences of times are ever used.
  memories              pools, slots, pairs, rings and key maps are fixed size (bounded state).
Reporting counters (bytes seen, steps) are metadata, not state the computation reads.
"""
from __future__ import annotations

import math

import torch

CEILING_SCALE = 0.1
FLOOR = 0.01
SCALAR_BOUND = 8.0
STEP_CAP = 1 << 20
REBASE = 1 << 30


def row_view(p: torch.Tensor) -> torch.Tensor:
    return p.view(1, -1) if p.ndim == 1 else p.flatten(start_dim=1)


def ceiling(p: torch.Tensor) -> float:
    """matrices: v14's row ceiling max(1, 0.1 sqrt(fan_in)). vectors (biases, LayerNorm gains): a soft RMS ceiling of 2,
    i.e. 2 sqrt(n) on the whole vector (a LayerNorm gain of ones sits at half its ceiling)."""
    if p.ndim == 1:
        return max(1.0, 2.0 * math.sqrt(p.numel()))
    return max(1.0, math.sqrt(row_view(p).shape[1]) * CEILING_SCALE)


@torch.no_grad()
def clip_(params, max_norm: float) -> None:
    """torch.nn.utils.clip_grad_norm_ (same coefficient, max_norm / (norm + 1e-6) capped at 1) without its whole-tensor
    vector_norm, which MPS runs ~30x slower than a dot product (27 ms for one 8192 x 2048 gradient); no host sync."""
    g = [p.grad for p in params if p.grad is not None]
    if not g:
        return
    if g[0].is_cuda:                                       # Nov 28: two multi-tensor kernels, not ~500 small ones (the
        sq = torch.stack(torch._foreach_norm(g)).square().sum()   # 1B trainer launches kernels faster than it can)
        coef = torch.clamp(max_norm / (sq.sqrt() + 1e-6), max=1.0)
        torch._foreach_mul_(g, coef)
        return
    sq = torch.stack([torch.dot(x.reshape(-1), x.reshape(-1)) for x in g]).sum()
    coef = torch.clamp(max_norm / (sq.sqrt() + 1e-6), max=1.0)
    for x in g:
        x.mul_(coef)


def norms_before(params) -> dict:
    return {id(p): torch.linalg.vector_norm(row_view(p.detach()), dim=1, keepdim=True) for p in params if p.ndim >= 1}


@torch.no_grad()
def compress_(params, before: dict):
    """soft radial compression (v14): only outward radial growth is attenuated."""
    for p in params:
        prev = before.get(id(p))
        if prev is None:
            continue
        cur = row_view(p.data)
        now = torch.linalg.vector_norm(cur, dim=1, keepdim=True)
        out = (now - prev).clamp_min(0.0)
        c = ceiling(p)
        gain = (1.0 / (1.0 + (prev / c) ** 4)).clamp_min(FLOOR)
        want = now - out * (1.0 - gain)
        want = torch.minimum(want, torch.where(prev <= c, torch.full_like(prev, c), prev))
        cur.mul_(want / now.clamp_min(1e-12))


@torch.no_grad()
def pressure(params) -> dict:
    """mean radial pressure 1 - 1/(1 + (norm/ceiling)^4) and the largest norm/ceiling over all rows."""
    pr, mx = [], 0.0
    for p in params:
        n = torch.linalg.vector_norm(row_view(p.detach()).float(), dim=1)
        r = n / ceiling(p)
        pr.append((1.0 - 1.0 / (1.0 + r ** 4)).mean().item())
        mx = max(mx, float(r.max()))
    return dict(radial_pressure=sum(pr) / max(1, len(pr)), max_row_over_ceiling=mx)


def cap_steps_(opt):
    for st in opt.state.values():
        s = st.get("step")
        if s is not None:
            if torch.is_tensor(s):
                s.clamp_(max=STEP_CAP)
            elif s > STEP_CAP:
                st["step"] = STEP_CAP


class SphereSGD:
    """GM's sphere SGD (soma_gm_v1/optim.py), for bound_mode 'sphere': every learned matrix keeps the Frobenius radius it
    was initialised with; a step moves only its direction (the gradient's tangent part, the rotation capped at max_angle
    radians), then retracts to the sphere. No moments are kept. Vectors take a plain step and are bounded in norm.
    The rate is param_groups[0]['lr'] * scale (io2 sets lr = lr_base * ratio; scale = sphere_rate / lr_base), times
    min(1, 256 / width) ** 0.5 as in GM. Nothing grows: radii are fixed, vectors are capped."""

    def __init__(self, params, lr, scale, width, max_angle=0.05, vector_bound=16.0):
        self.params = [p for p in params if p.requires_grad]
        self.param_groups = [{"lr": float(lr), "weight_decay": 0.0}]
        self.scale = float(scale) * min(1.0, 256.0 / float(width)) ** 0.5
        self.max_angle, self.vector_bound = float(max_angle), float(vector_bound)
        self.radii = [float(p.detach().square().sum().sqrt()) if p.ndim >= 2 else 0.0 for p in self.params]
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        rate = self.param_groups[0]["lr"] * self.scale
        tan = math.tan(self.max_angle)
        for p, r in zip(self.params, self.radii):
            g = p.grad
            if g is None:
                continue
            if p.ndim >= 2:
                u = p / r
                gt = g - (g * u).sum() * u
                step = rate * gt.square().sum().sqrt()
                shrink = (tan * r / step.clamp_min(1e-30)).clamp(max=1.0)
                p.sub_(gt * (shrink * rate))
                p.mul_(r / p.square().sum().sqrt().clamp_min(1e-30))
            else:
                p.sub_(g, alpha=rate)
                p.mul_((self.vector_bound / p.square().sum().sqrt().clamp_min(1e-30)).clamp(max=1.0))

    def state_dict(self):
        return {"kind": "sphere", "lr": self.param_groups[0]["lr"], "radii": list(self.radii)}

    def load_state_dict(self, s):
        assert s.get("kind") == "sphere", "not a sphere optimiser state"
        self.param_groups[0]["lr"] = float(s["lr"])
        self.radii = [float(x) for x in s["radii"]]
