#!/usr/bin/env python3
"""soma v12.2 experimental runtime.

one trace bank handles time. a serial mlp handles composition inside the now.
there are no trace banks over hidden states and no backprop through time.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import soma_v12 as core

PHI = core.PHI
EPS = 1e-10
VOCAB = 256
AUTO_DREAM_HARD_CAP = 2000
TURN_DELIMITER = 0x1E
MOTOR_DEC_CONTROL = 0x11
MOTOR_DEC_LEGACY = 0x12
MOTOR_IDS = (MOTOR_DEC_CONTROL, MOTOR_DEC_LEGACY)
DEFAULT_DECIMATION_STRIDE_CAP = 512
DEFAULT_AUTO_MODE = "io2"


def default_decimation_range(auto_mode):
    """Return the production-safe range for a controller mode.

    io2's historical ``1`` means the full normalized spectral range; the
    experimental controllers use an explicit twelve-band exploration range.
    ``off`` keeps dense, un-decimated training.
    """
    mode = str(auto_mode or DEFAULT_AUTO_MODE).strip().lower()
    if mode == "io2":
        return 1.0
    if mode == "off":
        return 0.0
    return 12.0

_RUNTIME_DIR = os.environ.get("SOMA_HOME", "")
DATA_DIR = os.path.join(_RUNTIME_DIR, "data") if _RUNTIME_DIR else "data"
CHECKPOINT_DIR = (os.path.join(_RUNTIME_DIR, "checkpoints")
                  if _RUNTIME_DIR else "checkpoints")


def _fmt_bytes(n):
    n = float(n)
    for unit in ("", "k", "m", "g"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n:.0f}"
        n /= 1000
    return f"{n:.1f}t"


def _resolve_path(path, kind):
    if not path:
        return path
    if ('/' in path or '\\' in path or
            path.startswith('~') or path.startswith('.')):
        return os.path.expanduser(path)
    folder = DATA_DIR if kind == 'corpus' else CHECKPOINT_DIR
    if kind == 'corpus' and not _RUNTIME_DIR:
        app_path = (Path.home() / "Library/Application Support/soma/data"
                    / path)
        if app_path.exists():
            return str(app_path)
    if kind == 'checkpoint' and not _RUNTIME_DIR:
        app_path = (Path.home()
                    / "Library/Application Support/soma/checkpoints"
                    / path)
        if app_path.exists():
            return str(app_path)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, path)


def _parse_auto_or_float(text, default_base=0.001):
    s = str(text).strip().lower()
    if s.startswith("auto") or s.startswith("io2"):
        parts = s.split()
        base = default_base
        for part in reversed(parts[1:]):
            try:
                base = float(part)
                break
            except ValueError:
                pass
        return base, True, base
    try:
        v = float(s)
    except ValueError:
        v = default_base
    return v, False, v


def _open_corpus(path):
    return np.memmap(_resolve_path(path, "corpus"), dtype=np.uint8, mode="r")


def _bar(frac, width=30):
    fill = int(frac * width)
    mid = 1 if 0 < frac < 1 and fill < width else 0
    return "▓" * fill + "▒" * mid + "░" * (width - fill - mid)


class BudgetRelu(nn.Module):
    def __init__(self, budget):
        super().__init__()
        self.budget = float(budget)

    def forward(self, x):
        h = F.relu(x)
        return h * (self.budget / (h.sum(dim=1, keepdim=True) + EPS))


class SerialNet(nn.Module):
    def __init__(self, n_features, hidden_dim, depth, budget_scale=0.1):
        super().__init__()
        self.n_features = int(n_features)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.budget = self.hidden_dim * float(budget_scale)
        self.layers = nn.ModuleList()
        dims = [self.n_features] + [self.hidden_dim] * self.depth
        for i in range(self.depth):
            self.layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        self.out = nn.Linear(self.hidden_dim, VOCAB, bias=False)
        self.act = BudgetRelu(self.budget)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.normal_()
                target = math.sqrt(layer.weight.shape[1]) * 0.1
                layer.weight.mul_(
                    target / (layer.weight.norm(dim=1, keepdim=True) + EPS))
            self.out.weight.normal_()
            target = math.sqrt(self.out.weight.shape[1]) * 0.1
            self.out.weight.mul_(
                target / (self.out.weight.norm(dim=1, keepdim=True) + EPS))

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))
        return self.out(x)


class SerialSOMA:
    species = "soma_v12_2_serial"

    def __init__(self, n_bands=16, hidden_dim=512, depth=3,
                 base=PHI, batch_size=512, lr=0.001,
                 grad_clip=1.0, decimation_range=None,
                 auto_mode=DEFAULT_AUTO_MODE, lr_auto=True, lr_base=None,
                 row_norm="auto", row_norm_mult=4.0,
                 row_norm_every=100,
                 decimation_stride_cap=DEFAULT_DECIMATION_STRIDE_CAP,
                 description="",
                 device="auto"):
        self.n_bands = int(n_bands)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.base = float(base)
        self.batch_size = int(batch_size)
        self.description = str(description or "")
        self.lr = float(lr)
        self.lr_base = float(lr if lr_base is None else lr_base)
        self.lr_auto = bool(lr_auto)
        self.grad_clip = float(grad_clip)
        self.row_norm = str(row_norm if row_norm is not None else "auto")
        self.row_norm_mult = float(row_norm_mult)
        self.row_norm_every = int(row_norm_every)
        self._train_steps = 0
        self._row_clip_fraction = 0.0
        self._row_clip_max_ratio = 0.0
        self._row_norm_max = 0.0
        if decimation_range is None:
            decimation_range = default_decimation_range(auto_mode)
        self.decimation_range = float(decimation_range)
        self.decimation_stride_cap = int(max(
            1, decimation_stride_cap or DEFAULT_DECIMATION_STRIDE_CAP))
        self.decimation_auto = True
        self.decimation_band = 0.0
        self.auto_mode = str(auto_mode or DEFAULT_AUTO_MODE).strip().lower()
        if self.auto_mode not in ("wallclock", "model", "io2", "off"):
            self.auto_mode = DEFAULT_AUTO_MODE
        self.wallclock_stride = 8.0
        self.device = core.SOMA._select_device(device)
        self.bank = core.TraceBank(VOCAB, self.n_bands, self.base, self.device)
        self.error_bank = core.TraceBank(
            VOCAB, self.n_bands, self.base, self.device)
        self.n_features = self.bank.n_features
        self.net = SerialNet(
            self.n_features, self.hidden_dim, self.depth).to(self.device)
        self.opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=0.0)
        self.bytes_seen = 0
        self.checkpoint_history = []
        self._spectral_drive = torch.ones(self.n_bands, device=self.device)
        self._spectral_target_band = 0.0
        self._input_coherence = 0.0
        self._output_coherence = 0.0
        self._input_trust = 0.0
        self._error_energy = 0.0
        self._error_coherence = 0.0
        self._error_concentration = 0.0
        self._io2_plasticity = 0.0
        self._motor_value = 0.0
        self._motor_target = 0.0
        self._motor_energy_push = 0.0
        self._motor_delta = 0.0
        self._motor_prev_value = None
        self._motor_volatility = 0.0
        self._motor_habituation = 0.0
        self._motor_habit_rate = 0.05
        self._motor_habit_scale = 40.0
        self._motor_stability_loss = 0.0
        self._motor_opportunity_loss = 0.0
        self._motor_feedback_value = 0.0
        self._motor_feedback_raw = 0.0
        self._motor_feedback_bias = 0.0
        self._motor_feedback_pulse = 0.0
        self._motor_feedback_pulse_prob = 0.0
        self._motor_feedback_pulse_max = 0.0
        self._motor_opportunity_ratio = 0.0
        self._motor_temp = 1.0
        self._motor_salience_center = 1.0
        self._motor_prob_scale = 32.0
        self._motor_smoothing = 0.08
        self._motor_surprise_smoothing = 0.30
        self._motor_surprise = 0.0
        self._motor_surprise_gain = 6.0
        self._motor_surprise_floor = 0.02
        self._motor_prediction_loss = 0.0
        self._motor_prediction_ratio = 1.0
        self._motor_loss_fast = None
        self._motor_loss_slow = None
        self._motor_loss_dev = 0.0
        self._attention_budget = 1.0
        self._attention_spend_rate = 0.004
        self._attention_recharge_rate = 0.010
        self._attention_reward_rate = 0.006
        self._attention_threshold_band = 0.0
        self._attention_cost = 0.0
        self._attention_recharge = 0.0
        self._attention_reward = 0.0
        self._attention_loss_gain = 0.0
        self._motor_stability_ratio = 0.0
        self._stride_jitter_band = 0.75
        self._sampled_stride = 1
        self._sampled_stride_float = 1.0
        # wallclock extremum-seeking state. the dither identifies whether a
        # slightly faster or slower gradient clock improves descent per second.
        self._wallclock_center_band = math.log(8.0, self.base)
        self._wallclock_probe_band = 0.18
        self._wallclock_probe_period = 512
        self._wallclock_probe_step = 0
        self._wallclock_dither = 0.0
        self._wallclock_last_loss = None
        self._wallclock_last_t = None
        self._wallclock_reward_ema = 0.0
        self._wallclock_reward_var = 1e-8
        self._wallclock_gain = 0.003
        self._last_loss_per_byte = None
        self._ema_fast = None
        self._ema_slow = None
        self._update_decimation()
        self._configure_wallclock_stride()

    def params(self):
        return sum(p.numel() for p in self.net.parameters())

    def _row_norm_ceiling(self, weight):
        mode = str(self.row_norm).strip().lower()
        if mode in ("", "off", "none", "false", "0"):
            return None
        if mode == "auto":
            return (math.sqrt(weight.shape[1]) * 0.1 *
                    max(0.0, self.row_norm_mult))
        try:
            return max(0.0, float(mode))
        except ValueError:
            return (math.sqrt(weight.shape[1]) * 0.1 *
                    max(0.0, self.row_norm_mult))

    def _apply_weight_ceiling(self, diagnose=False):
        total_rows = 0
        clipped_rows = 0
        max_ratio = 0.0
        max_norm_seen = 0.0
        with torch.no_grad():
            for module in list(self.net.layers) + [self.net.out]:
                weight = module.weight
                ceiling = self._row_norm_ceiling(weight)
                if ceiling is None or ceiling <= 0:
                    continue
                norms = weight.norm(dim=1, keepdim=True)
                scale = torch.clamp((ceiling / (norms + EPS)), max=1.0)
                weight.mul_(scale)
                if diagnose:
                    ratio = norms / (ceiling + EPS)
                    clipped = ratio > 1.0
                    clipped_rows += int(clipped.sum().item())
                    total_rows += weight.shape[0]
                    max_ratio = max(max_ratio, float(ratio.max().item()))
                    max_norm_seen = max(
                        max_norm_seen, float(norms.max().item()))
        if diagnose:
            self._row_clip_fraction = (
                clipped_rows / max(1, total_rows) if total_rows else 0.0)
            self._row_clip_max_ratio = max_ratio
            self._row_norm_max = max_norm_seen

    def _update_decimation(self):
        self.decimation_band = min(
            float(self.decimation_band), float(self._max_decimation_band()))
        self._stride, confidence = core.compute_band_confidence(
            self.n_bands, self.base, self.decimation_band)
        self._stride = min(self._stride, self.decimation_stride_cap)
        self._sampled_stride = self._stride
        self._band_confidence = torch.from_numpy(
            confidence).float().to(self.device)

    def _configure_wallclock_stride(self):
        """install the measured fixed-rate policy without touching the trace clock."""
        if self.auto_mode != "wallclock":
            return
        self.wallclock_stride = min(float(self.decimation_stride_cap), 8.0)
        self._wallclock_center_band = min(
            float(self._max_decimation_band()),
            math.log(max(1.0, self.wallclock_stride), self.base))
        self.decimation_band = self._wallclock_center_band
        self._update_decimation()

    def _batch_stride(self):
        if self.auto_mode == "wallclock":
            phase = (2.0 * math.pi * self._wallclock_probe_step
                     / max(2, self._wallclock_probe_period))
            self._wallclock_probe_step += 1
            self._wallclock_dither = self._wallclock_probe_band * math.sin(phase)
            band = max(0.0, min(
                float(self._max_decimation_band()),
                self._wallclock_center_band + self._wallclock_dither))
            self.decimation_band = band
            self._sampled_stride_float = min(
                float(self.decimation_stride_cap), self.base ** band)
            self._sampled_stride = max(1, int(round(self._sampled_stride_float)))
            return self._sampled_stride
        stride = max(1, int(self._stride))
        max_band = float(self._max_decimation_band())
        if (self.auto_mode != "model"
                or self._stride_jitter_band <= 0
                or max_band <= 0):
            self._sampled_stride = stride
            self._sampled_stride_float = float(stride)
            return stride
        center = max(0.0, min(max_band, float(self.decimation_band)))
        diffusion = center / max(max_band, EPS)
        if diffusion < 0.02:
            self._sampled_stride = stride
            self._sampled_stride_float = float(stride)
            return stride
        upper = min(
            float(self.decimation_stride_cap),
            max(1.0, self.base ** center))
        lower = 1.0 + (upper - 1.0) * ((1.0 - diffusion) ** 2)
        lower = max(1.0, min(lower, upper))
        if upper - lower <= EPS:
            self._sampled_stride = stride
            self._sampled_stride_float = float(stride)
            return stride
        sampled = random.uniform(lower, upper)
        lo = int(math.floor(sampled))
        frac = sampled - lo
        sampled_int = lo + (1 if random.random() < frac else 0)
        self._sampled_stride_float = float(sampled)
        self._sampled_stride = min(
            max(1, sampled_int), self.decimation_stride_cap)
        return self._sampled_stride

    def _max_decimation_band(self):
        if self.decimation_range <= 1.0:
            range_band = self.decimation_range * (self.n_bands - 1)
        else:
            range_band = self.decimation_range
        cap_band = math.log(max(1, self.decimation_stride_cap), self.base)
        return max(0, min(int(round(range_band)), self.n_bands - 1,
                          int(math.floor(cap_band))))

    def _attention_threshold(self):
        return 0.5 * float(self._max_decimation_band())

    def _sampled_band(self, stride):
        stride = max(1, int(stride))
        if stride <= 1:
            return 0.0
        return math.log(stride, self.base)

    def _sampled_stride_feedback(self, stride):
        max_band = float(self._max_decimation_band())
        if max_band <= EPS:
            return 0.0
        return max(0.0, min(1.0, self._sampled_band(stride) / max_band))

    def _biased_motor_feedback(self, value):
        pulse = 0.0
        if random.random() < self._motor_feedback_pulse_prob:
            pulse = random.random() * self._motor_feedback_pulse_max
        self._motor_feedback_pulse = float(pulse)
        return max(0.0, min(1.0, float(value) + pulse))

    def _update_attention_budget(self, max_auto_band, sampled_stride=None,
                                 loss_per_byte=None):
        if max_auto_band <= 0:
            self._attention_threshold_band = 0.0
            self._attention_cost = 0.0
            self._attention_recharge = 0.0
            self._attention_reward = 0.0
            self._attention_loss_gain = 0.0
            return
        threshold = max(0.0, min(float(max_auto_band),
                                 self._attention_threshold()))
        self._attention_threshold_band = threshold
        if sampled_stride is None:
            return
        sampled_band = max(
            0.0, min(float(max_auto_band), self._sampled_band(sampled_stride)))
        if threshold <= EPS:
            attention = 0.0
        else:
            attention = max(0.0, (threshold - sampled_band)
                            / max(threshold, EPS))
        if threshold >= max_auto_band - EPS:
            relaxation = 0.0
        else:
            relaxation = max(0.0, (sampled_band - threshold)
                             / max(max_auto_band - threshold, EPS))
        cost = self._attention_spend_rate * (attention ** 2)
        recharge = self._attention_recharge_rate * math.sqrt(
            max(0.0, relaxation))
        reward = 0.0
        gain_score = 0.0
        if loss_per_byte is not None and self._last_loss_per_byte is not None:
            prev = float(self._last_loss_per_byte)
            gain = max(0.0, prev - float(loss_per_byte))
            denom = max(float(self._motor_loss_dev),
                        0.002 * max(abs(prev), EPS),
                        EPS)
            gain_score = max(0.0, min(1.0, gain / denom))
            reward = self._attention_reward_rate * gain_score
        self._attention_budget = max(
            0.0, min(1.0,
                     self._attention_budget + recharge + reward - cost))
        self._attention_cost = float(cost)
        self._attention_recharge = float(recharge)
        self._attention_reward = float(reward)
        self._attention_loss_gain = float(gain_score)

    def _coherence_tensor(self, x):
        x = x.abs().flatten()
        total = x.sum()
        p = x / (total + EPS)
        entropy = -(p * torch.log(p + EPS)).sum()
        max_entropy = math.log(max(2, int(p.numel())))
        coh = 1.0 - torch.clamp(entropy / max_entropy, 0.0, 1.0)
        valid = torch.isfinite(total) & (total > EPS)
        return torch.where(valid, coh, torch.zeros_like(coh))

    def _update_io2_state(self, errors, probs):
        entropy = -(probs * torch.log(probs + EPS)).sum(dim=1)
        out_coh = torch.clamp(
            1.0 - (entropy.mean() / math.log(VOCAB)), 0.0, 1.0)

        self.error_bank.advance(errors.detach())
        bp = core.SOMA._bank_bandpass(self.error_bank).view(
            VOCAB, self.n_bands)
        energy = torch.linalg.vector_norm(bp, dim=0)
        total = energy.sum()
        energy_level = total / (
            total + math.sqrt(max(1, self.n_bands)) + EPS)
        drive = self.n_bands * energy / (total + EPS)
        self._spectral_drive = torch.clamp(
            drive, 0.1, 3.0).to(device=self.device, dtype=torch.float32)
        bands = torch.arange(self.n_bands, device=energy.device,
                             dtype=energy.dtype)
        center = (bands * energy).sum() / (total + EPS)
        p = energy / (total + EPS)
        entropy_b = -(p * torch.log(p + EPS)).sum()
        concentration = 1.0 - torch.clamp(
            entropy_b / math.log(max(2, self.n_bands)), 0.0, 1.0)
        valid_energy = torch.isfinite(total) & (total > EPS)
        center = torch.where(valid_energy, center, torch.zeros_like(center))
        concentration = torch.where(
            valid_energy, concentration, torch.zeros_like(concentration))

        input_bp = core.SOMA._bank_bandpass(self.bank).view(
            VOCAB, self.n_bands)
        channel_coh = self._coherence_tensor(input_bp)
        band_energy = torch.linalg.vector_norm(input_bp, dim=0)
        band_coh = self._coherence_tensor(band_energy)
        input_coh = torch.sqrt(torch.clamp(channel_coh * band_coh, min=0.0))
        vals = torch.stack((
            out_coh.detach().cpu().to(dtype=torch.float32),
            center.detach().cpu().to(dtype=torch.float32),
            concentration.detach().cpu().to(dtype=torch.float32),
            input_coh.detach().cpu().to(dtype=torch.float32),
            energy_level.detach().cpu().to(dtype=torch.float32),
        )).tolist()
        self._output_coherence = float(vals[0])
        self._spectral_target_band = float(vals[1])
        self._error_concentration = float(vals[2])
        self._error_coherence = float(vals[2])
        self._input_coherence = float(vals[3])
        self._error_energy = float(vals[4])

    def _progress_ratio_value(self, loss_per_byte):
        ref = 0.03
        if self._ema_fast is None:
            self._ema_fast = loss_per_byte
            self._ema_slow = loss_per_byte
            return 1.0
        self._ema_fast += 0.30 * (loss_per_byte - self._ema_fast)
        self._ema_slow += 0.08 * (loss_per_byte - self._ema_slow)
        drive = (abs(self._ema_fast - self._ema_slow)
                 / (ref * max(self._ema_slow, 1e-6)))
        return max(0.02, min(1.0, drive))

    def _io2_ratio(self, loss_per_byte):
        energy = max(0.0, min(1.0, self._error_energy))
        coherence = max(0.0, min(1.0, self._error_coherence))
        plasticity = 2.5 * math.sqrt(energy) * (
            0.25 + 0.75 * math.sqrt(coherence))
        chance = math.log(VOCAB)
        loss_scale = self.base ** (
            (float(loss_per_byte) - chance) / max(chance, EPS))
        loss_scale = max(1.0 / self.base, min(self.base, loss_scale))
        plasticity *= loss_scale
        self._input_trust = max(0.0, min(1.0, 3.0 * self._input_coherence))
        self._io2_plasticity = float(plasticity)
        return max(0.02, min(1.0, plasticity))

    def _io2_decimation_target(self, loss_per_byte, plasticity,
                               max_auto_band):
        loss_below = max(0.0, 1.0 - loss_per_byte)
        input_skip = 1.0 - max(0.0, min(1.0, self._input_trust))
        spectral_center = max(0.0, min(
            float(self._spectral_target_band), float(max_auto_band)))
        relax = max(min(1.0, loss_below), 0.5 * input_skip)
        throughput_target = (
            (1.0 - relax) * spectral_center + relax * float(max_auto_band))
        return (1.0 - plasticity) * throughput_target

    def _set_lr_ratio(self, ratio):
        if not self.lr_auto:
            return
        self.lr = self.lr_base * max(0.0, float(ratio))
        for group in self.opt.param_groups:
            group["lr"] = self.lr

    def _motor_control(self, logits):
        mask = torch.ones(VOCAB, dtype=torch.bool, device=logits.device)
        mask[list(MOTOR_IDS)] = False
        field = logits[..., mask]
        mu = field.mean(dim=-1)
        sigma = field.std(dim=-1, unbiased=False).clamp_min(1e-6)
        z = (logits[..., MOTOR_DEC_CONTROL] - mu) / sigma
        z = z - float(self._motor_salience_center)
        return torch.sigmoid(z / max(float(self._motor_temp), 1e-6))

    def _motor_from_logits(self, logits):
        value = self._motor_control(logits)
        return float(value.clamp(0.0, 1.0).detach().cpu())

    def _tick_motor_state(self, value):
        v = torch.zeros(VOCAB, dtype=torch.float32, device=self.device)
        v[MOTOR_DEC_CONTROL] = float(max(0.0, min(1.0, value)))
        self.bank.tick(v)

    def _tick_byte_with_motor(self, byte, value):
        v = torch.zeros(VOCAB, dtype=torch.float32, device=self.device)
        v[int(byte)] = 1.0
        v[MOTOR_DEC_CONTROL] += float(max(0.0, min(1.0, value)))
        self.bank.tick(v)

    def _apply_model_decimation(self):
        if self.auto_mode != "model":
            return
        max_auto_band = float(self._max_decimation_band())
        if max_auto_band <= 0:
            self._motor_value = 0.0
            self._motor_energy_push = 0.0
            self._motor_delta = 0.0
            self._motor_target = 0.0
            self._attention_threshold_band = 0.0
            return
        self._attention_threshold_band = 0.5 * max_auto_band
        self._attention_budget = 1.0
        self._attention_cost = 0.0
        self._attention_recharge = 0.0
        self._attention_reward = 0.0
        self._attention_loss_gain = 0.0
        was_training = self.net.training
        self.net.eval()
        try:
            with torch.no_grad():
                logits = self.net(self.bank.tap().unsqueeze(0)).squeeze(0)
                motor_value = self._motor_from_logits(logits)
        finally:
            if was_training:
                self.net.train()
        control = max(0.0, min(1.0, motor_value))
        target = control * max_auto_band
        target = max(0.0, min(target, max_auto_band))
        smoothing = max(0.0, min(1.0, self._motor_smoothing))
        next_band = (
            (1.0 - smoothing) * self.decimation_band
            + smoothing * target)
        if self._motor_prev_value is None:
            control_delta = 0.0
        else:
            control_delta = abs(float(motor_value)
                                - float(self._motor_prev_value))
        self._motor_prev_value = float(motor_value)
        self._motor_volatility += 0.05 * (
            control_delta - self._motor_volatility)
        habit_stability = 1.0 / (
            1.0 + self._motor_habit_scale * self._motor_volatility)
        habit_push = self._motor_habit_rate * habit_stability
        next_band = min(max_auto_band, next_band + habit_push)
        self._motor_value = float(motor_value)
        self._motor_target = float(target)
        self._motor_energy_push = float(habit_push)
        self._motor_habituation = float(habit_push)
        self._motor_delta = float(next_band - self.decimation_band)
        if abs(next_band - self.decimation_band) > 1e-6:
            self.decimation_band = next_band
            self._update_decimation()

    def _apply_motor_byte(self, byte):
        if byte != MOTOR_DEC_CONTROL:
            return
        max_auto_band = float(self._max_decimation_band())
        if max_auto_band <= 0:
            return
        target = max_auto_band
        self.decimation_band = (
            (1.0 - self._motor_smoothing) * self.decimation_band
            + self._motor_smoothing * target)
        self._update_decimation()

    def _apply_controller(self, batch_loss, n_bytes):
        loss_per_byte = float(batch_loss) / max(1, int(n_bytes))
        if self.auto_mode == "wallclock":
            ratio = self._io2_ratio(loss_per_byte)
            self._set_lr_ratio(ratio)
            self._update_wallclock_center(loss_per_byte)
            self._last_loss_per_byte = float(loss_per_byte)
            return
        if self.auto_mode == "model":
            ratio = self._io2_ratio(loss_per_byte)
            self._set_lr_ratio(ratio)
            self._io2_plasticity = float(ratio)
            self._last_loss_per_byte = float(loss_per_byte)
            if self._motor_loss_fast is None or self._motor_loss_slow is None:
                self._motor_loss_fast = float(loss_per_byte)
                self._motor_loss_slow = float(loss_per_byte)
                self._motor_loss_dev = 0.0
            else:
                self._motor_loss_fast += (
                    0.30 * (float(loss_per_byte) - self._motor_loss_fast))
                self._motor_loss_slow += (
                    0.02 * (float(loss_per_byte) - self._motor_loss_slow))
                gap = abs(self._motor_loss_fast - self._motor_loss_slow)
                self._motor_loss_dev += 0.05 * (gap - self._motor_loss_dev)
            return
        if self.auto_mode != "io2":
            return
        ratio = self._io2_ratio(loss_per_byte)
        self._set_lr_ratio(ratio)

        max_auto_band = self._max_decimation_band()
        target = self._io2_decimation_target(
            loss_per_byte, ratio, max_auto_band)
        target = max(0.0, min(target, float(max_auto_band)))
        if target > self.decimation_band:
            target = min(target, self.decimation_band + 1.0)
        else:
            target = max(target, self.decimation_band - 2.0)
        if abs(target - self.decimation_band) > 1e-6:
            self.decimation_band = target
            self._update_decimation()

    def _parse_dream_length(self, text):
        s = str(text).strip().lower()
        if s.startswith("auto"):
            parts = s.split()
            cap = 200
            if len(parts) > 1:
                try:
                    cap = min(int(float(parts[1])), AUTO_DREAM_HARD_CAP)
                except ValueError:
                    pass
            return True, 0, cap
        try:
            return False, int(float(s)), AUTO_DREAM_HARD_CAP
        except ValueError:
            return False, 200, AUTO_DREAM_HARD_CAP

    def _auto_dream_length(self, loss_value, cap):
        ratio = max(0.0, min(1.0, float(loss_value) / math.log(VOCAB)))
        return int(round((1.0 - ratio) * cap))

    def _text_array(self, text, mark_turn=False):
        data = [ord(c) % VOCAB for c in str(text)]
        if mark_turn:
            data.append(TURN_DELIMITER)
        return np.array(data, dtype=np.uint8)

    def _state_prelude_text(self, source="generation"):
        desc = self.description.replace("\n", " ").strip()[:160]
        lines = [
            "",
            "<soma_state>",
            f"source: {source}",
            f"species: {self.species}",
            "architecture: serial trace mlp",
            f"bytes_seen: {self.bytes_seen}",
            f"bands: {self.n_bands}",
            f"hidden: {self.hidden_dim}",
            f"depth: {self.depth}",
            f"base: {self.base:.6f}",
            f"batch: {self.batch_size}",
            f"auto_mode: {self.auto_mode}",
            f"lr: {self.lr:.8f}",
            f"lr_base: {self.lr_base:.8f}",
            f"lr_auto: {int(self.lr_auto)}",
            f"gradclip: {self.grad_clip:.6f}",
            f"rowclip: {self._row_clip_fraction:.6f}",
            f"decimation: {self.decimation_band:.6f}",
            f"stride: {self._stride}",
            f"sampled_stride: {self._sampled_stride}",
            f"sampled_stride_float: {self._sampled_stride_float:.6f}",
            f"io2: {self._io2_plasticity:.6f}",
            f"motor: {self._motor_value:.6f}",
            f"motor_target: {self._motor_target:.6f}",
            f"motor_feedback: {self._motor_feedback_value:.6f}",
            f"motor_feedback_raw: {self._motor_feedback_raw:.6f}",
            f"motor_feedback_pulse: {self._motor_feedback_pulse:.6f}",
            f"motor_feedback_pulse_prob: {self._motor_feedback_pulse_prob:.6f}",
            f"motor_feedback_pulse_max: {self._motor_feedback_pulse_max:.6f}",
            f"motor_delta: {self._motor_delta:.6f}",
            f"motor_habituation: {self._motor_habituation:.6f}",
            f"motor_volatility: {self._motor_volatility:.6f}",
            f"motor_prediction_loss: {self._motor_prediction_loss:.6f}",
            f"motor_prediction_ratio: {self._motor_prediction_ratio:.6f}",
            f"motor_surprise: {self._motor_surprise:.6f}",
            f"motor_stability_loss: {self._motor_stability_loss:.6f}",
            f"motor_opportunity_loss: {self._motor_opportunity_loss:.6f}",
            f"attention_budget: {self._attention_budget:.6f}",
            f"attention_threshold: {self._attention_threshold_band:.6f}",
            f"attention_cost: {self._attention_cost:.6f}",
            f"attention_recharge: {self._attention_recharge:.6f}",
            f"attention_reward: {self._attention_reward:.6f}",
            f"attention_loss_gain: {self._attention_loss_gain:.6f}",
            f"last_loss_per_byte: {self._last_loss_per_byte or 0.0:.6f}",
            f"motor_loss_fast: {self._motor_loss_fast or 0.0:.6f}",
            f"motor_loss_slow: {self._motor_loss_slow or 0.0:.6f}",
            f"motor_loss_dev: {self._motor_loss_dev:.6f}",
            f"input_coherence: {self._input_coherence:.6f}",
            f"output_coherence: {self._output_coherence:.6f}",
            f"input_trust: {self._input_trust:.6f}",
            f"error_energy: {self._error_energy:.6f}",
            f"error_coherence: {self._error_coherence:.6f}",
            f"error_concentration: {self._error_concentration:.6f}",
            f"spectral_target_band: {self._spectral_target_band:.6f}",
        ]
        if desc:
            lines.append(f"description: {desc}")
        lines.extend(("</soma_state>", ""))
        return "\n".join(lines)

    def ingest_self_state(self, source="generation", online=True):
        self.ingest_prompt(
            self._state_prelude_text(source), online=online, mark_turn=True)

    def _collect_strided_batch(self, corpus, pos, total, max_rows,
                               stop_at_newline=False):
        stride = self._batch_stride()
        motor_target = None
        if self.auto_mode == "model":
            raw_feedback = self._sampled_stride_feedback(stride)
            feedback = self._biased_motor_feedback(raw_feedback)
            self._tick_motor_state(feedback)
            self.bytes_seen += 1
            self._motor_feedback_raw = float(raw_feedback)
            self._motor_feedback_value = float(feedback)
            motor_target = float(feedback)
        mean_stride = (self._sampled_stride_float
                       if self.auto_mode == "wallclock" else float(stride))
        end = min(pos + int(round(mean_stride * max_rows)), total)
        chunk = corpus[pos:end]
        if stop_at_newline and len(chunk) > 0:
            hits = np.flatnonzero(chunk == ord("\n"))
            if hits.size:
                end = pos + int(hits[0]) + 1
                chunk = corpus[pos:end]
        if len(chunk) == 0:
            return None, None, None, pos, 0
        if self.auto_mode == "wallclock" and mean_stride > 1:
            rows = np.flatnonzero(
                np.random.random(len(chunk)) < 1.0 / mean_stride
            ).astype(np.int64)
            if not len(rows):
                rows = np.array([random.randrange(len(chunk))], dtype=np.int64)
        else:
            rows = np.arange(0, len(chunk), stride, dtype=np.int64)
        x = self.bank.process_block_select(chunk, rows)
        y = torch.from_numpy(
            chunk[rows].astype(np.int64)).to(self.device)
        motor_y = None
        if motor_target is not None:
            motor_y = torch.full(
                (len(rows),), motor_target, dtype=torch.float32,
                device=self.device)
        self.bytes_seen += len(chunk)
        return x, y, motor_y, end, len(chunk)

    def _update_wallclock_center(self, loss_per_byte):
        now = time.monotonic()
        if self._wallclock_last_loss is None:
            self._wallclock_last_loss = float(loss_per_byte)
            self._wallclock_last_t = now
            return
        elapsed = max(now - float(self._wallclock_last_t), 1e-4)
        reward = (float(self._wallclock_last_loss) - float(loss_per_byte)) / elapsed
        self._wallclock_last_loss = float(loss_per_byte)
        self._wallclock_last_t = now
        deviation = reward - self._wallclock_reward_ema
        self._wallclock_reward_ema += 0.02 * deviation
        self._wallclock_reward_var += 0.02 * (
            deviation * deviation - self._wallclock_reward_var)
        normalized = deviation / math.sqrt(self._wallclock_reward_var + EPS)
        step = self._wallclock_gain * normalized * self._wallclock_dither
        step = max(-0.004, min(0.004, step))
        self._wallclock_center_band = max(0.0, min(
            float(self._max_decimation_band()),
            self._wallclock_center_band + step))

    def _train_xy(self, x, y, motor_y=None):
        logits = self.net(x)
        motor_rows = torch.zeros_like(y, dtype=torch.bool)
        for motor_id in MOTOR_IDS:
            motor_rows |= (y == motor_id)
        language_rows = ~motor_rows
        losses = []
        language_n = int(language_rows.sum().item())
        language_loss = None
        if language_n:
            language_logits = logits[language_rows]
            language_loss = F.cross_entropy(language_logits, y[language_rows])
            losses.append(language_loss)
            if self.auto_mode == "model" and self._motor_opportunity_ratio > 0:
                control = self._motor_control(logits[language_rows]).clamp(
                    0.0, 1.0)
                threshold = 0.5
                attention = F.relu(threshold - control) / threshold
                opportunity = attention.pow(2).mean()
                loss_now = float(language_loss.detach().cpu())
                surprise = 0.0
                if self._motor_loss_slow is not None:
                    slow = float(self._motor_loss_slow)
                    upward = max(0.0, loss_now - slow)
                    denom = max(
                        float(self._motor_loss_dev),
                        self._motor_surprise_floor * max(abs(slow), EPS),
                        EPS)
                    surprise = upward / denom
                surprise_discount = 1.0 / (
                    1.0 + self._motor_surprise_gain * surprise)
                self._motor_surprise = float(surprise)
                self._motor_opportunity_loss = float(
                    opportunity.detach().cpu())
                losses.append(
                    self._motor_opportunity_ratio
                    * language_loss.detach()
                    * surprise_discount
                    * opportunity)
            else:
                self._motor_opportunity_loss = 0.0
                self._motor_surprise = 0.0
            if (self.auto_mode == "model"
                    and motor_y is not None
                    and self._motor_prediction_ratio > 0):
                motor_pred = self._motor_control(logits).clamp(0.0, 1.0)
                motor_target = motor_y.to(
                    device=logits.device, dtype=motor_pred.dtype)
                motor_loss = F.mse_loss(motor_pred, motor_target)
                self._motor_prediction_loss = float(
                    motor_loss.detach().cpu())
                losses.append(self._motor_prediction_ratio * motor_loss)
            else:
                self._motor_prediction_loss = 0.0
        if (self.auto_mode == "model"
                and self._motor_stability_ratio > 0
                and language_loss is not None):
            control = self._motor_control(logits).clamp(0.0, 1.0)
            max_auto_band = float(self._max_decimation_band())
            current = 0.0
            if max_auto_band > 0:
                current = max(0.0, min(1.0,
                                       self.decimation_band / max_auto_band))
            stability = (control - current).pow(2).mean()
            self._motor_stability_loss = float(stability.detach().cpu())
            scale = (self._motor_stability_ratio * language_loss.detach()
                     / (stability.detach() + EPS))
            losses.append(scale * stability)
        else:
            self._motor_stability_loss = 0.0
        if not losses:
            return 0.0, 0, 0
        loss = torch.stack(losses).sum()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), self.grad_clip)
        self.opt.step()
        self._train_steps += 1
        if (self.row_norm_every > 0
                and self._train_steps % self.row_norm_every == 0):
            self._apply_weight_ceiling(
                diagnose=(self._train_steps % (self.row_norm_every * 10) == 0))
        with torch.no_grad():
            if language_n:
                language_logits = logits[language_rows]
                y_lang = y[language_rows]
                acc = (language_logits.argmax(1) == y_lang).sum().item()
                probs = F.softmax(language_logits.detach(), dim=1)
                errors = probs.clone()
                idx = torch.arange(y_lang.shape[0], device=self.device)
                errors[idx, y_lang] -= 1.0
                self._update_io2_state(errors, probs)
            else:
                acc = 0
        return float(loss.item()), int(acc), language_n

    def _train_batch(self, chunk):
        x = self.bank.process_block(chunk)
        y = torch.from_numpy(chunk.astype(np.int64)).to(self.device)
        loss, acc, n = self._train_xy(x, y)
        self.bytes_seen += len(chunk)
        self._apply_controller(loss * n, n)
        return loss, acc, n

    def _train_data(self, data):
        total = len(data)
        pos = 0
        total_loss = 0.0
        total_acc = 0
        total_samples = 0
        while pos < total:
            self._apply_model_decimation()
            x, y, motor_y, pos, _consumed_n = self._collect_strided_batch(
                data, pos, total, self.batch_size)
            if x is None:
                break
            loss, acc, n = self._train_xy(x, y, motor_y)
            total_loss += loss * n
            total_acc += acc
            total_samples += n
            self._apply_controller(loss * n, n)
        return total_loss, total_acc, total_samples

    def train(self, corpus_path, epochs=1, save_path="v12_2.pt",
              start_byte=0, max_bytes=0, report_every=100_000,
              save_every=10_000_000, dream_every_batches=50,
              dream_length="auto 200", dream_temperature=1.0,
              dream_callback=None):
        corpus = _open_corpus(corpus_path)
        if start_byte > 0:
            corpus = corpus[start_byte:]
        if max_bytes and max_bytes > 0:
            corpus = corpus[:int(max_bytes)]
        total = len(corpus)
        dream_auto, dream_fixed, dream_cap = self._parse_dream_length(
            dream_length)
        dream_every_batches = max(0, int(dream_every_batches))

        print()
        print(f"  • soma v12.2 serial · {self.device} · "
              f"{_fmt_bytes(self.bytes_seen)} seen")
        print(
            f"    {self.n_bands} bands · hidden={self.hidden_dim:,} "
            f"· depth={self.depth} · {self.params()/1e6:.2f}m params")
        print(
            f"    lr={self.lr:g} · grad_clip={self.grad_clip:g} "
            f"· batch={self.batch_size:,} · auto={self.auto_mode} "
            f"· decimation range={self.decimation_range:g} "
            f"· row norm={self.row_norm} x{self.row_norm_mult:g}/"
            f"{self.row_norm_every}")
        print()
        print(f"  ∿ training {corpus_path} ({_fmt_bytes(total)} bytes)")
        print()

        batch_count = 0
        last_save = 0
        for epoch in range(epochs):
            t0 = time.time()
            total_loss = 0.0
            correct = 0
            samples = 0
            consumed = 0
            last_report = 0
            pos = 0
            while pos < total:
                self._apply_model_decimation()
                align_dream = (
                    dream_every_batches
                    and (batch_count + 1) % dream_every_batches == 0)
                x, y, motor_y, pos, consumed_n = self._collect_strided_batch(
                    corpus, pos, total, self.batch_size,
                    stop_at_newline=align_dream)
                if x is None:
                    break
                loss, acc, n = self._train_xy(x, y, motor_y)
                total_loss += loss * n
                correct += acc
                samples += n
                consumed += consumed_n
                batch_count += 1
                self._apply_controller(loss * n, n)

                if dream_every_batches and batch_count % dream_every_batches == 0:
                    length = dream_fixed
                    if dream_auto:
                        length = self._auto_dream_length(loss, dream_cap)
                    if length > 0:
                        text = "".join(self.generate(
                            length=length, temperature=dream_temperature,
                            prelude_source="dream"))
                        if dream_callback:
                            dream_callback(text, batch_count, self.bytes_seen)
                        else:
                            print(f"\n    dream {batch_count} · "
                                  f"{_fmt_bytes(self.bytes_seen)} seen")
                            print(f"    {text}\n", flush=True)

                if report_every and consumed - last_report >= report_every:
                    self._report(epoch, epochs, consumed, total, total_loss,
                                 correct, samples, t0)
                    last_report = consumed

                if save_path and save_every and consumed - last_save >= save_every:
                    self.save(save_path)
                    last_save = consumed

            self._report(epoch, epochs, consumed, total, total_loss,
                         correct, samples, t0)
        if save_path:
            self.save(save_path)

    def _report(self, epoch, epochs, pos, total, loss, correct, samples, t0):
        elapsed = time.time() - t0
        avg = loss / max(1, samples)
        acc = 100 * correct / max(1, samples)
        frac = pos / max(1, total)
        bps = pos / max(elapsed, 1e-9)
        stride_text = str(self._stride)
        if self._sampled_stride != self._stride:
            stride_text = f"{self._sampled_stride}/{self._stride}"
        print(
            f"    [{epoch + 1}/{epochs}] {_bar(frac)} {frac*100:4.1f}% "
            f"· {avg:.3f} nats · {acc:.1f}% · {bps:,.0f} b/s "
            f"· stride {stride_text} · band {self.decimation_band:.2f} "
            f"· lr {self.lr:.5f} "
            f"· ctrl {self._motor_value:.2f} "
            f"· motorloss {self._motor_prediction_loss:.4f} "
            f"· habit {self._motor_habituation:.4f} "
            f"· motor {self._motor_delta:+.3f} "
            f"· rowclip {100*self._row_clip_fraction:.1f}%",
            flush=True,
        )

    def evaluate(self, corpus_path, max_bytes=0):
        data = _open_corpus(corpus_path)
        if max_bytes and max_bytes > 0:
            data = data[:int(max_bytes)]
        self.bank.reset()
        total_loss = 0.0
        correct = 0
        samples = 0
        t0 = time.time()
        with torch.no_grad():
            for pos in range(0, len(data), self.batch_size):
                chunk = data[pos:pos + self.batch_size]
                x = self.bank.process_block(chunk)
                y = torch.from_numpy(chunk.astype(np.int64)).to(self.device)
                logits = self.net(x)
                loss = F.cross_entropy(logits, y)
                total_loss += float(loss.item()) * len(chunk)
                correct += int((logits.argmax(1) == y).sum().item())
                samples += len(chunk)
        elapsed = time.time() - t0
        avg = total_loss / max(1, samples)
        print(
            f"  eval · {avg:.4f} nats · "
            f"{100*correct/max(1,samples):.1f}% · "
            f"{samples/max(elapsed,1e-9):,.0f} b/s")
        return avg

    def generate(self, length=200, temperature=1.0,
                 prelude=True, prelude_source="generation"):
        if prelude:
            self.ingest_self_state(prelude_source, online=True)
        was_training = self.net.training
        self.net.eval()
        try:
            for _ in range(int(length)):
                x = self.bank.tap().unsqueeze(0)
                with torch.no_grad():
                    logits = self.net(x).squeeze(0)
                    motor_value = self._motor_from_logits(logits)
                    sample_logits = logits / max(float(temperature), 1e-6)
                    sample_logits = sample_logits.clone()
                    sample_logits[list(MOTOR_IDS)] = -1e9
                    probs = F.softmax(sample_logits, dim=0)
                    byte = torch.multinomial(probs, 1).item()
                if byte in (ord("\n"), TURN_DELIMITER):
                    self._tick_byte_with_motor(byte, motor_value)
                    break
                yield chr(byte) if 32 <= byte < 127 else "."
                self._tick_byte_with_motor(byte, motor_value)
        finally:
            self.bank.tick(TURN_DELIMITER)
            if was_training:
                self.net.train()

    def ingest_prompt(self, text, online=False, mark_turn=True):
        data = self._text_array(text, mark_turn=mark_turn)
        if online:
            self._train_data(data)
        else:
            self.bank.process_block(data)
            self.bytes_seen += len(data)

    def save(self, path):
        if not path:
            return
        path = _resolve_path(path, "checkpoint")
        ckpt = {
            "species": self.species,
            "n_bands": self.n_bands,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "base": self.base,
            "batch_size": self.batch_size,
            "description": self.description,
            "architecture": "soma v12.2 serial trace mlp",
            "runtime_version": "v12.2",
            "lr": self.lr,
            "lr_base": self.lr_base,
            "lr_auto": self.lr_auto,
            "grad_clip": self.grad_clip,
            "row_norm": self.row_norm,
            "row_norm_mult": self.row_norm_mult,
            "row_norm_every": self.row_norm_every,
            "train_steps": self._train_steps,
            "auto_mode": self.auto_mode,
            "decimation_range": self.decimation_range,
            "decimation_stride_cap": self.decimation_stride_cap,
            "decimation_band": self.decimation_band,
            "stride_jitter_band": self._stride_jitter_band,
            "sampled_stride_float": self._sampled_stride_float,
            "wallclock_center_band": self._wallclock_center_band,
            "wallclock_probe_band": self._wallclock_probe_band,
            "wallclock_probe_period": self._wallclock_probe_period,
            "wallclock_probe_step": self._wallclock_probe_step,
            "wallclock_reward_ema": self._wallclock_reward_ema,
            "wallclock_reward_var": self._wallclock_reward_var,
            "motor_value": self._motor_value,
            "motor_target": self._motor_target,
            "motor_feedback_value": self._motor_feedback_value,
            "motor_feedback_raw": self._motor_feedback_raw,
            "motor_feedback_bias": self._motor_feedback_bias,
            "motor_feedback_pulse": self._motor_feedback_pulse,
            "motor_feedback_pulse_prob": self._motor_feedback_pulse_prob,
            "motor_feedback_pulse_max": self._motor_feedback_pulse_max,
            "motor_delta": self._motor_delta,
            "motor_energy_push": self._motor_energy_push,
            "motor_prev_value": self._motor_prev_value,
            "motor_volatility": self._motor_volatility,
            "motor_habituation": self._motor_habituation,
            "motor_habit_rate": self._motor_habit_rate,
            "motor_habit_scale": self._motor_habit_scale,
            "motor_prediction_loss": self._motor_prediction_loss,
            "motor_prediction_ratio": self._motor_prediction_ratio,
            "motor_surprise": self._motor_surprise,
            "motor_surprise_gain": self._motor_surprise_gain,
            "motor_surprise_floor": self._motor_surprise_floor,
            "motor_salience_center": self._motor_salience_center,
            "motor_stability_loss": self._motor_stability_loss,
            "motor_opportunity_loss": self._motor_opportunity_loss,
            "motor_opportunity_ratio": self._motor_opportunity_ratio,
            "attention_budget": self._attention_budget,
            "attention_threshold_band": self._attention_threshold_band,
            "attention_cost": self._attention_cost,
            "attention_recharge": self._attention_recharge,
            "attention_reward": self._attention_reward,
            "attention_loss_gain": self._attention_loss_gain,
            "attention_spend_rate": self._attention_spend_rate,
            "attention_recharge_rate": self._attention_recharge_rate,
            "attention_reward_rate": self._attention_reward_rate,
            "last_loss_per_byte": self._last_loss_per_byte,
            "motor_loss_fast": self._motor_loss_fast,
            "motor_loss_slow": self._motor_loss_slow,
            "motor_loss_dev": self._motor_loss_dev,
            "ema_fast": self._ema_fast,
            "ema_slow": self._ema_slow,
            "spectral_drive": self._spectral_drive.detach().cpu(),
            "spectral_target_band": self._spectral_target_band,
            "input_coherence": self._input_coherence,
            "output_coherence": self._output_coherence,
            "input_trust": self._input_trust,
            "error_energy": self._error_energy,
            "error_coherence": self._error_coherence,
            "error_concentration": self._error_concentration,
            "io2_plasticity": self._io2_plasticity,
            "bytes_seen": self.bytes_seen,
            "traces": self.bank.traces.detach().cpu(),
            "error_traces": self.error_bank.state_numpy(),
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "checkpoint_history": self.checkpoint_history[-256:],
        }
        # A completed checkpoint is always either the old file or the new
        # file. An interrupted autosave must never leave a half-written model.
        tmp_path = f"{path}.tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)
        print(f"    ⟐ saved {path}")

    def load(self, path):
        ckpt = torch.load(
            _resolve_path(path, "checkpoint"), map_location="cpu",
            weights_only=False)
        if ckpt.get("species") != self.species:
            raise ValueError("this is not a v12.2 serial checkpoint")
        self.n_bands = int(ckpt["n_bands"])
        self.hidden_dim = int(ckpt["hidden_dim"])
        self.depth = int(ckpt["depth"])
        self.base = float(ckpt["base"])
        self.batch_size = int(ckpt.get("batch_size", self.batch_size))
        self.description = str(ckpt.get("description", ""))
        self.lr = float(ckpt.get("lr", self.lr))
        self.lr_base = float(ckpt.get("lr_base", self.lr))
        self.lr_auto = bool(ckpt.get("lr_auto", True))
        self.grad_clip = float(ckpt.get("grad_clip", self.grad_clip))
        self.row_norm = str(ckpt.get("row_norm", self.row_norm))
        self.row_norm_mult = float(
            ckpt.get("row_norm_mult", self.row_norm_mult))
        self.row_norm_every = int(
            ckpt.get("row_norm_every", self.row_norm_every))
        self.auto_mode = str(ckpt.get(
            "auto_mode", DEFAULT_AUTO_MODE)).strip().lower()
        if self.auto_mode not in ("wallclock", "model", "io2", "off"):
            self.auto_mode = DEFAULT_AUTO_MODE
        self.decimation_range = float(
            ckpt.get("decimation_range", self.decimation_range))
        self.decimation_stride_cap = int(max(
            1, ckpt.get("decimation_stride_cap",
                        self.decimation_stride_cap)))
        self.decimation_band = float(
            ckpt.get("decimation_band", 0.0))
        self.bank = core.TraceBank(
            VOCAB, self.n_bands, self.base, self.device)
        self.bank.traces = ckpt["traces"].to(
            device=self.bank.trace_device, dtype=self.bank.trace_dtype)
        self.error_bank = core.TraceBank(
            VOCAB, self.n_bands, self.base, self.device)
        if "error_traces" in ckpt:
            self.error_bank.load_state(ckpt["error_traces"])
        self.n_features = self.bank.n_features
        self.net = SerialNet(
            self.n_features, self.hidden_dim, self.depth).to(self.device)
        self.net.load_state_dict(ckpt["net"])
        self.opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=0.0)
        if "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])
            for state in self.opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        self.bytes_seen = int(ckpt.get("bytes_seen", 0))
        self._train_steps = int(ckpt.get("train_steps", 0))
        self.checkpoint_history = list(ckpt.get("checkpoint_history", []))
        self._ema_fast = ckpt.get("ema_fast", None)
        self._ema_slow = ckpt.get("ema_slow", None)
        self._spectral_drive = ckpt.get(
            "spectral_drive", torch.ones(self.n_bands)).float().to(
                self.device)
        self._spectral_target_band = ckpt.get("spectral_target_band", 0.0)
        self._input_coherence = ckpt.get("input_coherence", 0.0)
        self._output_coherence = ckpt.get("output_coherence", 0.0)
        self._input_trust = ckpt.get("input_trust", 0.0)
        self._error_energy = ckpt.get("error_energy", 0.0)
        self._error_coherence = ckpt.get("error_coherence", 0.0)
        self._error_concentration = ckpt.get("error_concentration", 0.0)
        self._io2_plasticity = ckpt.get("io2_plasticity", 0.0)
        self._motor_value = ckpt.get("motor_value", 0.0)
        self._motor_target = ckpt.get("motor_target", 0.0)
        self._motor_feedback_value = ckpt.get("motor_feedback_value", 0.0)
        self._motor_feedback_raw = ckpt.get(
            "motor_feedback_raw", self._motor_feedback_value)
        self._motor_feedback_bias = 0.0
        self._motor_feedback_pulse = ckpt.get("motor_feedback_pulse", 0.0)
        self._motor_feedback_pulse_prob = 0.0
        self._motor_feedback_pulse_max = 0.0
        self._motor_delta = ckpt.get("motor_delta", 0.0)
        self._motor_energy_push = ckpt.get("motor_energy_push", 0.0)
        self._motor_prev_value = ckpt.get("motor_prev_value", None)
        self._motor_volatility = ckpt.get("motor_volatility", 0.0)
        self._motor_habituation = ckpt.get("motor_habituation", 0.0)
        self._motor_habit_rate = ckpt.get("motor_habit_rate", 0.05)
        self._motor_habit_scale = ckpt.get("motor_habit_scale", 40.0)
        self._motor_prediction_loss = ckpt.get("motor_prediction_loss", 0.0)
        self._motor_prediction_ratio = ckpt.get(
            "motor_prediction_ratio", 1.0)
        self._motor_surprise = ckpt.get("motor_surprise", 0.0)
        self._motor_surprise_gain = max(
            float(ckpt.get("motor_surprise_gain", 6.0)), 6.0)
        self._motor_surprise_floor = min(
            float(ckpt.get("motor_surprise_floor", 0.02)), 0.02)
        self._motor_salience_center = float(
            ckpt.get("motor_salience_center", 1.0))
        self._motor_stability_loss = ckpt.get(
            "motor_stability_loss", ckpt.get("motor_balance_loss", 0.0))
        self._motor_opportunity_loss = ckpt.get(
            "motor_opportunity_loss", 0.0)
        self._motor_opportunity_ratio = 0.0
        self._attention_budget = 1.0
        self._attention_threshold_band = ckpt.get(
            "attention_threshold_band", 0.0)
        self._attention_cost = ckpt.get("attention_cost", 0.0)
        self._attention_recharge = ckpt.get("attention_recharge", 0.0)
        self._attention_reward = ckpt.get("attention_reward", 0.0)
        self._attention_loss_gain = ckpt.get("attention_loss_gain", 0.0)
        self._attention_spend_rate = min(
            float(ckpt.get("attention_spend_rate", 0.004)), 0.004)
        self._attention_recharge_rate = ckpt.get(
            "attention_recharge_rate", 0.010)
        self._attention_reward_rate = ckpt.get(
            "attention_reward_rate", 0.006)
        self._last_loss_per_byte = ckpt.get("last_loss_per_byte", None)
        self._motor_loss_fast = ckpt.get("motor_loss_fast", None)
        self._motor_loss_slow = ckpt.get("motor_loss_slow", None)
        self._motor_loss_dev = ckpt.get("motor_loss_dev", 0.0)
        self._stride_jitter_band = ckpt.get("stride_jitter_band", 0.75)
        self._sampled_stride_float = ckpt.get("sampled_stride_float", 1.0)
        self._wallclock_center_band = float(ckpt.get(
            "wallclock_center_band", math.log(8.0, self.base)))
        self._wallclock_probe_band = float(ckpt.get(
            "wallclock_probe_band", 0.18))
        self._wallclock_probe_period = int(ckpt.get(
            "wallclock_probe_period", 512))
        self._wallclock_probe_step = int(ckpt.get("wallclock_probe_step", 0))
        self._wallclock_dither = 0.0
        self._wallclock_last_loss = None
        self._wallclock_last_t = None
        self._wallclock_reward_ema = float(ckpt.get("wallclock_reward_ema", 0.0))
        self._wallclock_reward_var = float(ckpt.get("wallclock_reward_var", 1e-8))
        self._update_decimation()
        if self.auto_mode == "wallclock":
            if "wallclock_center_band" in ckpt:
                self.decimation_band = max(0.0, min(
                    float(self._max_decimation_band()),
                    self._wallclock_center_band))
                self._update_decimation()
            else:
                self._configure_wallclock_stride()
        print(f"    ⟐ loaded {_resolve_path(path, 'checkpoint')}")


SOMA = SerialSOMA


def cmd_train(args):
    lr, lr_auto, lr_base = _parse_auto_or_float(args.lr, 0.001)
    if args.resume:
        model = SerialSOMA(device=args.device)
        model.load(args.resume)
        loaded_auto_mode = model.auto_mode
        auto_mode = args.auto_mode or loaded_auto_mode
        decimation_range = (args.decimation if args.decimation is not None
                            else model.decimation_range)
        model.lr = lr
        model.lr_base = lr_base
        model.lr_auto = lr_auto
        model.auto_mode = auto_mode.strip().lower()
        model.decimation_range = float(decimation_range)
        model.decimation_stride_cap = int(max(1, args.max_stride))
        model.grad_clip = float(args.grad_clip)
        model.description = str(args.description or model.description)
        model.row_norm = str(args.row_norm)
        model.row_norm_mult = float(args.row_norm_mult)
        model.row_norm_every = int(args.row_norm_every)
        model.batch_size = int(args.batch)
        for group in model.opt.param_groups:
            group["lr"] = model.lr
        model._update_decimation()
        if (model.auto_mode == "wallclock"
                and loaded_auto_mode != "wallclock"):
            model._configure_wallclock_stride()
    else:
        auto_mode = args.auto_mode or DEFAULT_AUTO_MODE
        decimation_range = (args.decimation if args.decimation is not None
                            else default_decimation_range(auto_mode))
        model = SerialSOMA(
            n_bands=args.bands,
            hidden_dim=args.hidden,
            depth=args.depth,
            base=args.base,
            batch_size=args.batch,
            lr=lr,
            grad_clip=args.grad_clip,
            decimation_range=decimation_range,
            decimation_stride_cap=args.max_stride,
            auto_mode=auto_mode,
            lr_auto=lr_auto,
            lr_base=lr_base,
            row_norm=args.row_norm,
            row_norm_mult=args.row_norm_mult,
            row_norm_every=args.row_norm_every,
            description=args.description,
            device=args.device,
        )
    model.train(
        _resolve_path(args.corpus, "corpus"),
        epochs=args.epochs,
        save_path=_resolve_path(args.save, "checkpoint"),
        start_byte=args.start,
        max_bytes=args.bytes,
        report_every=args.report_every,
        save_every=args.save_every,
        dream_every_batches=args.dream_every,
        dream_length=args.dream_length,
        dream_temperature=args.temperature,
    )


def cmd_chat(args):
    model = SerialSOMA(device=args.device)
    model.load(args.checkpoint)
    print("soma v12.2 serial chat. ctrl-c to quit.")
    try:
        while True:
            try:
                prompt = input("\nyou › ")
            except EOFError:
                break
            model.ingest_prompt(prompt + "\n", online=args.online)
            print("soma › " + "".join(
                model.generate(
                    args.length, args.temperature,
                    prelude_source="chat")))
    except KeyboardInterrupt:
        print()
    if args.save:
        model.save(args.checkpoint)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("corpus")
    t.add_argument("--save", default="v12_2.pt")
    t.add_argument("--resume", default="")
    t.add_argument("--bands", type=int, default=20)
    t.add_argument("--hidden", type=int, default=1536)
    t.add_argument("--depth", type=int, default=3)
    t.add_argument("--base", type=float, default=1.6180)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--description", default="")
    t.add_argument("--lr", default="auto 0.001")
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--row-norm", default="auto",
                   help="row norm ceiling: auto, off, or numeric")
    t.add_argument("--row-norm-mult", type=float, default=4.0)
    t.add_argument("--row-norm-every", type=int, default=100)
    t.add_argument("--auto-mode", default=None,
                   choices=("wallclock", "model", "io2", "off"))
    t.add_argument("--decimation", type=float, default=None)
    t.add_argument("--max-stride", type=int,
                   default=DEFAULT_DECIMATION_STRIDE_CAP,
                   help="hard cap for auto-decimation stride in bytes")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--start", type=int, default=0)
    t.add_argument("--bytes", type=int, default=0)
    t.add_argument("--save-every", type=int, default=10_000_000)
    t.add_argument("--report-every", type=int, default=100_000)
    t.add_argument("--dream-every", type=int, default=100)
    t.add_argument("--dream-length", default="auto 300")
    t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--device", default="auto")
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("chat")
    c.add_argument("checkpoint")
    c.add_argument("--length", type=int, default=300)
    c.add_argument("--temperature", type=float, default=1.0)
    c.add_argument("--online", action="store_true")
    c.add_argument("--save", action="store_true")
    c.add_argument("--device", default="auto")
    c.set_defaults(func=cmd_chat)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
