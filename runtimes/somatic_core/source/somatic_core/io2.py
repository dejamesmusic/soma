"""io2, James's production controller, ported from its newest form: soma v14's trace-branch mixer
(soma_harness/runtimes/v14/.../trace_branch_mixer/runtime.py, spec soma_v14_spec.md §7-8). v12.4's ratio formula is
unchanged there; v14 adds the parts that make it independent of how the stream is cut into ticks.

Evidence. For each selected prediction the categorical residual is translated into the learner's own coordinates
through its readout, c = W_out^T (onehot(y) - p) / mean row norm (v14 uses the tied token dictionary; here, the
readout rows: c is the negative loss gradient at the last hidden layer). c advances a float64 phi bank of k+1 bands
on the source-token clock; adjacent differences give k visible bands (the hidden extra-slow band keeps the last
visible band from being a forced low-pass remainder). A sampled residual is a causal point observation: elapsed
time decays the bank first, then it is injected with the stationary-variance gain for the current stride, and
adjacent-band energy is calibrated to the dense white-residual baseline.
    a_j = |bandpass_j|,  pi = a / sum a
    energy     = sum a / (sum a + sqrt k)
    coherence  = 1 - H(pi) / log k
    band       = sum_j j pi_j
    ratio      = clamp(2.5 sqrt(energy) (0.25 + 0.75 sqrt(coherence)) * clamp(phi^((loss - ln V)/ln V), 1/phi, phi),
                       0.02, 1)
Actuation happens once per canonical prediction quantum (accumulated selected predictions), not per tick:
    lr = lr_base * amplitude      (amplitude = ratio; with pulses, ratio / frequency)
    decimation band -> (1 - ratio) * ((1 - relax) * band + relax * native),  relax = clamp(1 - bpb, 0, 1),
        rising <= 1 and falling <= 2 bands per quantum (scaled by the mass actually accumulated); stride = phi^band
    pulse frequency (spectral) f = max(ratio, 1 - cost * coherence * (1 - phi^-band))
"""
from __future__ import annotations

import math

import torch

PHI = (1 + 5 ** 0.5) / 2
EPS = 1e-12


class IO2:
    def __init__(self, dim: int, vocab: int, lr_base: float = 3e-4, bands: int = 16, base: float = PHI,
                 quantum: float = 512.0, stride_cap: int = 256, decimation_gain: float = 1.0, pulse: bool = False):
        self.dim, self.vocab, self.lr_base = int(dim), int(vocab), float(lr_base)
        self.k, self.base = int(bands), float(base)
        self.quantum, self.stride_cap, self.gain, self.pulse = float(quantum), int(stride_cap), float(decimation_gain), bool(pulse)
        self.retention = 1.0 - torch.tensor([self.base ** -i for i in range(self.k + 1)], dtype=torch.float64)
        self.trace = torch.zeros(self.k + 1, self.dim, dtype=torch.float64)
        self.ratio, self.energy, self.coherence, self.target_band = 1.0, 0.0, 0.0, 0.0
        self.band = 0.0
        self.spectrum = [0.0] * self.k
        self.update_cost = 0.0
        self.mass = self.nll = self.bytes_ = 0.0
        self.actuations = 0
        self.stride_ = 1
        self.lr_ = self.lr_base

    # ---- the bank ----
    def _sampling_gain(self, s):
        num = -torch.expm1(2.0 * s * torch.log(self.retention))
        den = -torch.expm1(2.0 * torch.log(self.retention))
        return torch.sqrt(num / den.clamp_min(1e-300))

    def _calibration(self, s):
        r, inj = self.retention, 1.0 - self.retention
        rs, injs = r ** s, inj * self._sampling_gain(s)

        def cov(lr_, rr, li, ri):
            return li * ri / (-torch.expm1(torch.log(lr_) + torch.log(rr))).clamp_min(1e-300)

        dense = (cov(r, r, inj, inj)[:-1] + cov(r, r, inj, inj)[1:] - 2 * cov(r[:-1], r[1:], inj[:-1], inj[1:])).clamp_min(1e-300)
        samp = (cov(rs, rs, injs, injs)[:-1] + cov(rs, rs, injs, injs)[1:] - 2 * cov(rs[:-1], rs[1:], injs[:-1], injs[1:])).clamp_min(1e-300)
        return torch.sqrt(dense / samp)

    def advance(self, values: torch.Tensor, gaps):
        """values (n, dim) in time order; gaps (n,) tokens elapsed before each (decay first, then inject)."""
        v = values.detach().float().cpu().double()
        if not len(v):
            return
        g = torch.as_tensor(gaps, dtype=torch.float64).clamp_min(1.0)
        cum = g.cumsum(0)
        rem = cum[-1] - cum
        s = max(1, int(round(float(g.mean()))))
        inj = (1.0 - self.retention) * self._sampling_gain(s)
        coef = inj[None] * self.retention[None] ** rem[:, None]                      # (n, k+1)
        self.trace.mul_((self.retention ** cum[-1])[:, None]).add_(coef.T @ v)
        a = torch.linalg.vector_norm(self.trace[:-1] - self.trace[1:], dim=1) * self._calibration(s)
        tot = float(a.sum())
        if tot <= EPS:
            self.energy = self.coherence = self.target_band = 0.0
            self.spectrum = [0.0] * self.k
            return
        pi = a / tot
        self.spectrum = pi.tolist()                                       # residual share per band, fast to slow (reporting)
        self.coherence = max(0.0, min(1.0, 1.0 - float(-(pi * pi.clamp_min(EPS).log()).sum()) / math.log(max(2, self.k))))
        self.target_band = float((torch.arange(self.k, dtype=torch.float64) * pi).sum())
        self.energy = tot / (tot + math.sqrt(self.k))

    def ratio_for(self, token_loss: float) -> float:
        plast = 2.5 * math.sqrt(self.energy) * (0.25 + 0.75 * math.sqrt(self.coherence))
        chance = math.log(self.vocab)
        plast *= max(1.0 / self.base, min(self.base, self.base ** ((float(token_loss) - chance) / chance)))
        return max(0.02, min(1.0, plast))

    # ---- actuation, once per canonical quantum of predictions ----
    def accumulate(self, token_nll_sum: float, predictions: float, source_bytes: float) -> bool:
        """add one tick's selected predictions; returns True when io2 actuated."""
        self.nll += float(token_nll_sum)
        self.mass += float(predictions)
        self.bytes_ += float(source_bytes)
        if self.mass + 1e-9 < self.quantum:
            return False
        loss = self.nll / self.mass
        bpb = self.nll / max(EPS, self.bytes_) / math.log(2)
        elapsed = self.mass / self.quantum
        r = self.ratio_for(loss)
        self.ratio = r
        self.lr_ = self.lr_base * self.amplitude()
        max_band = math.log(max(1, self.stride_cap), self.base)
        native = min(float(self.k - 1), max_band)
        relax = max(0.0, min(1.0, 1.0 - bpb))
        raw = (1.0 - r) * ((1.0 - relax) * max(0.0, min(native, self.target_band)) + relax * native)
        tb = max(0.0, min(max_band, self.gain * raw))
        tb = min(tb, self.band + elapsed) if tb > self.band else max(tb, self.band - 2.0 * elapsed)
        self.band = tb
        self.stride_ = min(self.stride_cap, max(1, int(round(self.base ** self.band))))
        self.mass = self.nll = self.bytes_ = 0.0
        self.actuations += 1
        return True

    def frequency(self) -> float:
        if not self.pulse:
            return 1.0
        f = 1.0 - max(0.0, min(1.0, self.update_cost)) * self.coherence * (1.0 - self.base ** -self.target_band)
        return max(self.ratio, min(1.0, f))

    def amplitude(self) -> float:
        return self.ratio / max(self.frequency(), EPS) if self.pulse else self.ratio

    def lr(self) -> float:
        return self.lr_

    def stride(self) -> int:
        return self.stride_

    def report(self) -> dict:
        return dict(io2_ratio=self.ratio, io2_energy=self.energy, io2_coherence=self.coherence,
                    io2_band=self.target_band, io2_decimation_band=self.band, io2_stride=self.stride_, lr=self.lr_)

    def state(self) -> dict:
        return dict(trace=self.trace.clone(), ratio=self.ratio, energy=self.energy, coherence=self.coherence,
                    target_band=self.target_band, band=self.band, update_cost=self.update_cost, mass=self.mass,
                    nll=self.nll, bytes_=self.bytes_, actuations=self.actuations, stride_=self.stride_, lr_=self.lr_,
                    lr_base=self.lr_base, gain=self.gain, quantum=self.quantum)

    def load(self, s: dict):
        self.trace = s["trace"].clone().double()
        for k in ("ratio", "energy", "coherence", "target_band", "band", "update_cost", "mass", "nll", "bytes_",
                  "lr_", "lr_base", "gain", "quantum"):
            setattr(self, k, float(s[k]))
        self.actuations, self.stride_ = int(s["actuations"]), int(s["stride_"])
