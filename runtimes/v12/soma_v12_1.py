#!/usr/bin/env python3
"""soma v12.1 — belief-stack runtime.

v12.1 keeps soma's trace-bank learner but replaces hidden-trace depth
with belief-stack depth:

    byte trace -> belief -> trace belief -> belief correction

fresh checkpoints are a new species. v12 checkpoints are not loaded
into v12.1, because the deep state has a different meaning.
"""

import argparse
import gc
import hashlib
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import soma_v12 as core

PHI = core.PHI
EPS = core.EPS
VOCAB = core.VOCAB
AUTO_DREAM_HARD_CAP = core.AUTO_DREAM_HARD_CAP

_RUNTIME_DIR = os.environ.get("SOMA_HOME", "")
DATA_DIR = os.path.join(_RUNTIME_DIR, "data") if _RUNTIME_DIR else "data"
CHECKPOINT_DIR = (os.path.join(_RUNTIME_DIR, "checkpoints")
                  if _RUNTIME_DIR else "checkpoints")


def _fmt_bytes(n):
    n = float(n)
    for unit in ("", "K", "M", "G"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n:.0f}"
        n /= 1000
    return f"{n:.1f}T"


def _bar(frac, width=30):
    fill = int(frac * width)
    mid = 1 if 0 < frac < 1 and fill < width else 0
    return "▓" * fill + "▒" * mid + "░" * (width - fill - mid)


def _resolve_path(path, kind):
    if not path:
        return path
    if ('/' in path or '\\' in path or
            path.startswith('~') or path.startswith('.')):
        return os.path.expanduser(path)
    folder = DATA_DIR if kind == 'corpus' else CHECKPOINT_DIR
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, path)


def _parse_auto_or_float(text, default_base=1.0):
    s = str(text).strip().lower()
    if s.startswith("auto"):
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


def _parse_decimation_range(text, default_value=1.0):
    try:
        return float(str(text).strip())
    except ValueError:
        return default_value


class BeliefSOMA:
    def __init__(self, n_bands=16, hidden_dim=1536, n_stacks=None,
                 base=PHI, lr=1.0, max_change=1.0, batch_size=512,
                 decimation_range=1.0, device="auto", n_layers=None,
                 **_retired):
        self.n_bands = int(n_bands)
        self.hidden_dim = int(hidden_dim)
        if n_stacks is None:
            n_stacks = 2 if n_layers is None else n_layers
        self.n_stacks = max(1, int(n_stacks))
        self.base = float(base)
        self.lr = float(lr)
        self.max_change = float(max_change)
        self.lr_base = float(lr)
        self.max_change_base = float(max_change)
        self.lr_auto = True
        self.max_change_auto = True
        self.decimation_auto = True
        self.decimation_range = float(decimation_range)
        self.batch_size = int(batch_size)
        self.device = core.SOMA._select_device(device)

        self.banks = [
            core.TraceBank(VOCAB, self.n_bands, self.base, self.device)
            for _ in range(self.n_stacks)
        ]
        self.bank = self.banks[0]
        self.error_bank = core.TraceBank(
            VOCAB, self.n_bands, self.base, self.device)

        self.n_features = self.banks[0].n_features
        self.hidden_budget = self.hidden_dim * 0.1
        self.u_norm = math.sqrt(self.n_features) * 0.1
        self.w_norm = math.sqrt(self.hidden_dim) * 0.1

        self.U_list = []
        self.W_list = []
        for _ in range(self.n_stacks):
            self.U_list.append(torch.randn(
                self.hidden_dim, self.n_features, device=self.device))
            self.W_list.append(torch.randn(
                VOCAB, self.hidden_dim, device=self.device))

        self.decimation_band = 0.0
        self._update_decimation()
        self._spectral_drive = torch.ones(self.n_bands, device=self.device)
        self._spectral_target_band = 0.0
        self._input_coherence = 0.0
        self._output_coherence = 0.0
        self._input_trust = 0.0
        self._error_concentration = 0.0
        self._io2_plasticity = 0.0
        self._ema_fast = None
        self._ema_slow = None

        self.bytes_seen = 0
        self.checkpoint_history = []
        self._dream_batch_counter = 0
        self.auto_mode = "io2"
        self.weight_decay = 0.0
        self.Wd = None
        self.G_list = []
        self.gb_list = []
        self.scale_gate = False
        self.clock = 1
        self._normalize_all()

    @property
    def n_layers(self):
        return self.n_stacks

    def params(self):
        return int(sum(x.numel() for x in self.U_list) +
                   sum(x.numel() for x in self.W_list))

    def print_config(self):
        print()
        print(f"  • soma v12.1 · {self.device} · "
              f"{_fmt_bytes(self.bytes_seen)} seen")
        print(
            f"    {self.n_bands} bands · base={self.base:.4f} "
            f"· hidden={self.hidden_dim:,} · stacks={self.n_stacks} "
            f"· {self.params()/1e6:.1f}m params")
        print(
            f"    lr=auto {self.lr_base} · max_change=auto "
            f"{self.max_change_base} · decimation range={self.decimation_range}")
        print()

    def _normalize_all(self):
        with torch.no_grad():
            for U in self.U_list:
                U.mul_(self.u_norm / (U.norm(dim=1, keepdim=True) + EPS))
            for W in self.W_list:
                W.mul_(self.w_norm / (W.norm(dim=1, keepdim=True) + EPS))

    def _update_decimation(self):
        self._stride, confidence = core.compute_band_confidence(
            self.n_bands, self.base, self.decimation_band)
        self._band_confidence = torch.from_numpy(
            confidence).float().to(self.device)

    def _features_to_compute(self, x):
        if x.device != self.device or x.dtype != torch.float32:
            return x.to(device=self.device, dtype=torch.float32)
        return x

    def _hidden(self, U, X):
        pre = X @ U.T
        relu = F.relu(pre)
        total = relu.sum(dim=1, keepdim=True) + EPS
        h = relu * (self.hidden_budget / total)
        return pre, total, h

    def _column_scale(self):
        return (self._band_confidence * self._spectral_drive).repeat(VOCAB)

    def _apply_clipped(self, param, grad):
        raw = self.lr * grad
        max_delta = self.max_change * param.abs()
        param -= torch.clamp(raw, -max_delta, max_delta)

    def _apply_feature_update(self, param, grad):
        scale = self._column_scale().view(1, -1)
        raw = self.lr * grad * scale
        max_delta = self.max_change * param.abs()
        param -= torch.clamp(raw, -max_delta, max_delta)

    def _coherence_from_distribution(self, x):
        x = x.abs().flatten()
        total = x.sum()
        if not torch.isfinite(total) or total <= EPS:
            return 0.0
        p = x / (total + EPS)
        entropy = -(p * torch.log(p + EPS)).sum()
        max_entropy = math.log(max(2, int(p.numel())))
        coh = 1.0 - torch.clamp(entropy / max_entropy, 0.0, 1.0)
        return float(coh.detach().cpu().item())

    def _update_spectral_drive(self, errors, probs_for_output):
        entropy = -(probs_for_output * torch.log(
            probs_for_output + EPS)).sum(dim=1)
        out_coh = 1.0 - (entropy.mean() / math.log(VOCAB))
        self._output_coherence = float(torch.clamp(
            out_coh, 0.0, 1.0).detach().cpu().item())

        self.error_bank.advance(errors)
        bp = core.SOMA._bank_bandpass(self.error_bank).view(
            VOCAB, self.n_bands)
        energy = torch.linalg.vector_norm(bp, dim=0)
        total = energy.sum()
        if not torch.isfinite(total) or total <= EPS:
            self._spectral_drive = torch.ones(
                self.n_bands, device=self.device)
            return
        drive = self.n_bands * energy / (total + EPS)
        self._spectral_drive = torch.clamp(
            drive, 0.1, 3.0).to(device=self.device, dtype=torch.float32)
        bands = torch.arange(self.n_bands, device=energy.device,
                             dtype=energy.dtype)
        center = (bands * energy).sum() / (total + EPS)
        if torch.isfinite(center):
            self._spectral_target_band = float(center.detach().cpu().item())

        p = energy / (total + EPS)
        entropy_b = -(p * torch.log(p + EPS)).sum()
        concentration = 1.0 - torch.clamp(
            entropy_b / math.log(max(2, self.n_bands)), 0.0, 1.0)
        self._error_concentration = float(
            concentration.detach().cpu().item())

        input_bp = core.SOMA._bank_bandpass(self.banks[0]).view(
            VOCAB, self.n_bands)
        channel_coh = self._coherence_from_distribution(input_bp)
        band_energy = torch.linalg.vector_norm(input_bp, dim=0)
        band_coh = self._coherence_from_distribution(band_energy)
        self._input_coherence = float((channel_coh * band_coh) ** 0.5)

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
        progress_gate = self._progress_ratio_value(loss_per_byte)
        input_trust = max(0.0, min(1.0, 3.0 * self._input_coherence))
        output_coh = max(0.0, min(1.0, self._output_coherence))
        error_coh = max(0.0, min(1.0, self._error_concentration))
        loss_above = max(0.0, loss_per_byte - 1.0)
        underfit = max(0.0, input_trust - output_coh)
        confident_wrong = output_coh * loss_above
        learnability = (
            loss_above * input_trust * (0.5 + 2.0 * error_coh)
            * (0.5 + underfit + confident_wrong) * progress_gate
        )
        plasticity = learnability / (1.0 + learnability)
        self._input_trust = float(input_trust)
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

    def _update_auto(self, batch_loss, n_bytes):
        if hasattr(batch_loss, "item"):
            batch_loss = batch_loss.item()
        loss_per_byte = float(batch_loss) / max(1, int(n_bytes))
        ratio = self._io2_ratio(loss_per_byte)
        self.lr = self.lr_base * ratio
        self.max_change = self.max_change_base * ratio

        max_auto_band = int(round(
            self.decimation_range * (self.n_bands - 1)))
        max_auto_band = max(0, min(max_auto_band, self.n_bands - 1))
        target = self._io2_decimation_target(
            loss_per_byte, ratio, max_auto_band)
        target = max(0.0, min(target, float(max_auto_band)))
        if target > self.decimation_band:
            target = min(target, self.decimation_band + 1.0)
        else:
            target = max(target, self.decimation_band - 2.0)
        if target != self.decimation_band:
            self.decimation_band = target
            self._update_decimation()

    def _forward_from_x0(self, X0):
        Xs, hs, pres, sums, stack_logits = [], [], [], [], []
        logits = torch.zeros(X0.shape[0], VOCAB, device=self.device)

        X = X0
        for s in range(self.n_stacks):
            Xs.append(X)
            pre, total, h = self._hidden(self.U_list[s], X)
            pres.append(pre)
            sums.append(total)
            hs.append(h)
            z = h @ self.W_list[s].T
            stack_logits.append(z)
            logits = logits + z
            if s + 1 < self.n_stacks:
                belief = F.softmax(z, dim=1).detach()
                X = self._features_to_compute(
                    self.banks[s + 1].process_block(belief))
        return logits, Xs, hs, pres, sums, stack_logits

    def _train_batch(self, X0, yt, n):
        logits, Xs, hs, pres, sums, _stack_logits = self._forward_from_x0(X0)
        probs = F.softmax(logits, dim=1)
        idx = torch.arange(n, device=self.device)
        loss = -torch.log(probs[idx, yt] + EPS).sum()
        acc = (logits.argmax(1) == yt).sum()
        errors = probs.clone()
        errors[idx, yt] -= 1.0
        self._update_spectral_drive(errors, probs)

        with torch.no_grad():
            for s in range(self.n_stacks):
                W = self.W_list[s]
                U = self.U_list[s]
                X = Xs[s]
                h = hs[s]
                pre = pres[s]
                total = sums[s]

                grad_W = (errors.T @ h) / n
                grad_h = errors @ W
                self._apply_clipped(W, grad_W)

                scale = self.hidden_budget / total
                grad_relu = (
                    grad_h * scale
                    - (grad_h * h).sum(dim=1, keepdim=True)
                    * scale / self.hidden_budget
                )
                grad_pre = grad_relu * (pre > 0).float()
                grad_U = (grad_pre.T @ X) / n
                self._apply_feature_update(U, grad_U)

            self._normalize_all()
        return loss, acc

    def _collect_strided_batch(self, corpus, pos, total, max_rows):
        stride = max(1, int(self._stride))
        end = min(pos + stride * max_rows, total)
        chunk = corpus[pos:end]
        if len(chunk) == 0:
            return None, None, pos
        rows = np.arange(0, len(chunk), stride, dtype=np.int64)
        X0 = self._features_to_compute(
            self.banks[0].process_block_select(chunk, rows))
        yt = torch.from_numpy(
            chunk[rows].astype(np.int64)).to(self.device)
        self.bytes_seen += len(chunk)
        return X0, yt, end

    def _parse_dream_length(self, dream_length):
        s = str(dream_length).strip().lower()
        if s.startswith("auto"):
            parts = s.split()
            cap = AUTO_DREAM_HARD_CAP
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

    def _auto_dream_length(self, batch_loss, n, cap):
        if hasattr(batch_loss, "item"):
            batch_loss = batch_loss.item()
        ratio = max(0.0, min(1.0, (batch_loss / max(1, n))
                             / float(np.log(VOCAB))))
        return int(round((1.0 - ratio) * cap))

    def train(self, corpus_path, epochs=1, save_every=0,
              save_path="model.pt", start_byte=0, report_every=250_000,
              dream_every_batches=0, dream_length="auto 200",
              dream_temperature=1.0, max_bytes=0, dream_callback=None):
        corpus = np.fromfile(corpus_path, dtype=np.uint8)
        if start_byte > 0:
            corpus = corpus[start_byte:]
        if max_bytes and max_bytes > 0:
            corpus = corpus[:int(max_bytes)]
        n_total = len(corpus)
        dream_auto, dream_fixed, dream_cap = self._parse_dream_length(
            dream_length)
        dream_every_batches = max(0, int(dream_every_batches))

        print()
        print(f"  • soma v12.1 · {self.device} · {_fmt_bytes(self.bytes_seen)} seen")
        print(
            f"    {self.n_bands} bands · hidden={self.hidden_dim:,} "
            f"· stacks={self.n_stacks} · {self.params()/1e6:.1f}m params")
        print(
            f"    io2 · lr auto {self.lr_base} · max_change auto "
            f"{self.max_change_base} · decimation range={self.decimation_range}")
        print()
        print(f"  ∿ training {corpus_path} ({_fmt_bytes(n_total)} bytes)")
        print(f"    batch={self.batch_size:,} · {epochs} epoch")
        print()

        last_save_pos = 0
        batch_count = 0
        for epoch in range(epochs):
            total_loss = torch.zeros((), device=self.device)
            correct = torch.zeros((), dtype=torch.int64, device=self.device)
            samples = 0
            pos = 0
            t0 = time.time()
            next_report = min(report_every, n_total)
            while pos < n_total:
                X0, yt, pos = self._collect_strided_batch(
                    corpus, pos, n_total, self.batch_size)
                if X0 is None:
                    break
                n = int(yt.shape[0])
                loss, acc = self._train_batch(X0, yt, n)
                total_loss += loss
                correct += acc
                samples += n
                batch_count += 1
                self._update_auto(loss, n)

                if dream_every_batches and batch_count % dream_every_batches == 0:
                    length = dream_fixed
                    if dream_auto:
                        length = self._auto_dream_length(loss, n, dream_cap)
                    if length > 0:
                        text = "".join(self.generate(
                            length=length, temperature=dream_temperature))
                        if dream_callback is not None:
                            dream_callback(text, batch_count,
                                           self.bytes_seen)
                        else:
                            print(f"\n    dream {batch_count} · "
                                  f"{_fmt_bytes(self.bytes_seen)} seen")
                            print(f"    {text}\n", flush=True)

                if pos >= next_report or pos >= n_total:
                    self._report(epoch, epochs, pos, n_total,
                                 total_loss.item(), correct.item(),
                                 samples, t0)
                    next_report += report_every

                if save_every and pos - last_save_pos >= save_every:
                    self.save(save_path)
                    last_save_pos = pos

            elapsed = time.time() - t0
            avg = total_loss.item() / max(1, samples)
            acc_pct = 100 * correct.item() / max(1, samples)
            print(f"    epoch {epoch + 1} done · {avg:.4f} nats "
                  f"· {acc_pct:.1f}% · {elapsed:.1f}s "
                  f"· {n_total / max(elapsed, 1e-9):,.0f} b/s")
        if save_path:
            self.save(save_path)

    def _report(self, epoch, epochs, pos, total, loss, correct, samples, t0):
        elapsed = time.time() - t0
        avg = loss / max(1, samples)
        acc = 100 * correct / max(1, samples)
        bps = pos / max(elapsed, 1e-9)
        frac = pos / max(1, total)
        print(
            f"    [{epoch + 1}/{epochs}] {_bar(frac)} {frac*100:4.1f}% "
            f"· {avg:.3f} nats · {acc:.1f}% · {bps:,.0f} b/s "
            f"· stride {self._stride} · band {self.decimation_band:.2f} "
            f"· lr {self.lr:.3f}",
            flush=True,
        )

    def _single_forward(self):
        logits = torch.zeros(VOCAB, device=self.device)
        hs, stack_logits = [], []
        for s in range(self.n_stacks):
            x = self.banks[s].tap().unsqueeze(0)
            _pre, _total, h = self._hidden(self.U_list[s], x)
            h = h.squeeze(0)
            z = self.W_list[s] @ h
            hs.append(h)
            stack_logits.append(z)
            logits = logits + z
        return logits, stack_logits

    def _tick_all(self, byte_val, stack_logits):
        self.banks[0].tick(int(byte_val))
        for s in range(1, self.n_stacks):
            belief = F.softmax(stack_logits[s - 1], dim=0).detach()
            self.banks[s].tick(belief)

    def generate(self, length=200, temperature=1.0):
        for _ in range(int(length)):
            logits, stack_logits = self._single_forward()
            logits = logits / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=0)
            byte_val = torch.multinomial(probs, 1).item()
            if byte_val == ord("\n"):
                break
            yield chr(byte_val) if 32 <= byte_val < 127 else "."
            self._tick_all(byte_val, stack_logits)

    def ingest_prompt(self, text, online=False):
        data = np.array([ord(c) % VOCAB for c in text], dtype=np.uint8)
        if len(data) == 0:
            return
        if online:
            for pos in range(0, len(data), self.batch_size):
                chunk = data[pos:pos + self.batch_size]
                X0 = self._features_to_compute(
                    self.banks[0].process_block(chunk))
                yt = torch.from_numpy(
                    chunk.astype(np.int64)).to(self.device)
                loss, _acc = self._train_batch(X0, yt, len(chunk))
                self.bytes_seen += len(chunk)
                self._update_auto(loss, len(chunk))
        else:
            X0 = self._features_to_compute(self.banks[0].process_block(data))
            with torch.no_grad():
                self._forward_from_x0(X0)
            self.bytes_seen += len(data)

    def _hash_tensor_full(self, h, tensor, chunk_elems=1_000_000):
        if tensor is None:
            h.update(b"none")
            return
        t = tensor.detach().reshape(-1)
        h.update(str(tuple(tensor.shape)).encode())
        for i in range(0, t.numel(), chunk_elems):
            h.update(t[i:i + chunk_elems].cpu().numpy().tobytes())

    def checkpoint_id(self):
        h = hashlib.sha256()
        for U in self.U_list:
            self._hash_tensor_full(h, U)
        for W in self.W_list:
            self._hash_tensor_full(h, W)
        for bank in self.banks:
            self._hash_tensor_full(h, bank.traces)
        self._hash_tensor_full(h, self.error_bank.traces)
        for val in [self.n_bands, self.hidden_dim, self.n_stacks,
                    self.base, self.bytes_seen, self.decimation_band]:
            h.update(str(val).encode())
        return h.hexdigest()

    def _pre_save_memory_hygiene(self):
        try:
            if self.device.type == "mps" and hasattr(torch, "mps"):
                if hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
        except Exception:
            pass
        gc.collect()

    def save(self, path):
        self._pre_save_memory_hygiene()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ckpt_id = self.checkpoint_id()
        history = self.checkpoint_history + [ckpt_id]
        data = {
            "soma_version": "v12.1",
            "architecture": "belief_stack",
            "checkpoint_id": ckpt_id,
            "checkpoint_history": history,
            "n_bands": self.n_bands,
            "hidden_dim": self.hidden_dim,
            "n_stacks": self.n_stacks,
            "base": self.base,
            "lr": self.lr,
            "max_change": self.max_change,
            "lr_base": self.lr_base,
            "max_change_base": self.max_change_base,
            "decimation_range": self.decimation_range,
            "decimation_band": self.decimation_band,
            "batch_size": self.batch_size,
            "bytes_seen": self.bytes_seen,
            "ema_fast": self._ema_fast,
            "ema_slow": self._ema_slow,
            "spectral_drive": self._spectral_drive.detach().cpu(),
            "spectral_target_band": self._spectral_target_band,
            "input_coherence": self._input_coherence,
            "output_coherence": self._output_coherence,
            "input_trust": self._input_trust,
            "error_concentration": self._error_concentration,
            "io2_plasticity": self._io2_plasticity,
            "u_norm": self.u_norm,
            "w_norm": self.w_norm,
            "hidden_budget": self.hidden_budget,
            "error_traces": self.error_bank.state_numpy(),
        }
        for s in range(self.n_stacks):
            data[f"U_{s}"] = self.U_list[s].cpu()
            data[f"W_{s}"] = self.W_list[s].cpu()
            data[f"traces_{s}"] = self.banks[s].state_numpy()
        torch.save(data, path)
        self.checkpoint_history = history
        del data
        gc.collect()
        print(f"    ⟐ saved {path} · {ckpt_id[:12]}", flush=True)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("soma_version") != "v12.1":
            raise ValueError("this is not a v12.1 belief-stack checkpoint")
        if ckpt.get("architecture") != "belief_stack":
            raise ValueError("checkpoint architecture is not belief_stack")
        self.n_bands = int(ckpt["n_bands"])
        self.hidden_dim = int(ckpt["hidden_dim"])
        self.n_stacks = int(ckpt["n_stacks"])
        self.base = float(ckpt["base"])
        self.lr = float(ckpt.get("lr", self.lr))
        self.max_change = float(ckpt.get("max_change", self.max_change))
        self.lr_base = float(ckpt.get("lr_base", self.lr_base))
        self.max_change_base = float(
            ckpt.get("max_change_base", self.max_change_base))
        self.decimation_range = float(
            ckpt.get("decimation_range", self.decimation_range))
        self.decimation_band = float(
            ckpt.get("decimation_band", self.decimation_band))
        self.batch_size = int(ckpt.get("batch_size", self.batch_size))
        self.bytes_seen = int(ckpt.get("bytes_seen", 0))
        self._ema_fast = ckpt.get("ema_fast", None)
        self._ema_slow = ckpt.get("ema_slow", None)
        self._spectral_drive = ckpt.get(
            "spectral_drive", torch.ones(self.n_bands)).float().to(
                self.device)
        self._spectral_target_band = ckpt.get("spectral_target_band", 0.0)
        self._input_coherence = ckpt.get("input_coherence", 0.0)
        self._output_coherence = ckpt.get("output_coherence", 0.0)
        self._input_trust = ckpt.get("input_trust", 0.0)
        self._error_concentration = ckpt.get("error_concentration", 0.0)
        self._io2_plasticity = ckpt.get("io2_plasticity", 0.0)
        self.u_norm = ckpt.get("u_norm", self.u_norm)
        self.w_norm = ckpt.get("w_norm", self.w_norm)
        self.hidden_budget = ckpt.get("hidden_budget", self.hidden_budget)
        self.checkpoint_history = ckpt.get("checkpoint_history", [])

        self.banks = [
            core.TraceBank(VOCAB, self.n_bands, self.base, self.device)
            for _ in range(self.n_stacks)
        ]
        self.bank = self.banks[0]
        self.error_bank = core.TraceBank(
            VOCAB, self.n_bands, self.base, self.device)
        if "error_traces" in ckpt:
            self.error_bank.load_state(ckpt["error_traces"])
        self.U_list, self.W_list = [], []
        for s in range(self.n_stacks):
            self.U_list.append(ckpt[f"U_{s}"].float().to(self.device))
            self.W_list.append(ckpt[f"W_{s}"].float().to(self.device))
            self.banks[s].load_state(ckpt[f"traces_{s}"])
        self.n_features = self.banks[0].n_features
        self._update_decimation()
        self._normalize_all()
        print(f"    ⟐ loaded {path}")


def new_model(args):
    lr, _lr_auto, lr_base = _parse_auto_or_float(args.lr, 1.0)
    mc, _mc_auto, mc_base = _parse_auto_or_float(args.max_change, 1.0)
    model = BeliefSOMA(
        n_bands=args.bands,
        hidden_dim=args.hidden,
        n_stacks=args.stacks,
        base=args.base,
        lr=lr,
        max_change=mc,
        batch_size=args.batch,
        decimation_range=args.decimation,
        device=args.device,
    )
    model.lr_base = lr_base
    model.max_change_base = mc_base
    return model


SOMA = BeliefSOMA


def cmd_train(args):
    if args.resume:
        model = BeliefSOMA(device=args.device)
        model.load(_resolve_path(args.resume, "checkpoint"))
        model.device = core.SOMA._select_device(args.device)
    else:
        model = new_model(args)
    corpus = _resolve_path(args.corpus, "corpus")
    save_path = _resolve_path(args.save, "checkpoint")
    model.train(
        corpus,
        epochs=args.epochs,
        save_every=args.save_every,
        save_path=save_path,
        start_byte=args.start,
        report_every=args.report_every,
        dream_every_batches=args.dream_every,
        dream_length=args.dream_length,
        dream_temperature=args.temperature,
        max_bytes=args.bytes,
    )


def cmd_chat(args):
    model = BeliefSOMA(device=args.device)
    model.load(_resolve_path(args.checkpoint, "checkpoint"))
    print("soma v12.1 chat. ctrl-c to quit.")
    try:
        while True:
            try:
                prompt = input("\nyou › ")
            except EOFError:
                break
            model.ingest_prompt(prompt + "\n", online=args.online)
            text = "".join(model.generate(args.length, args.temperature))
            print(f"soma › {text}")
    except KeyboardInterrupt:
        print()
    if args.save:
        model.save(_resolve_path(args.checkpoint, "checkpoint"))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("corpus")
    t.add_argument("--save", default="v12_1.pt")
    t.add_argument("--resume", default="")
    t.add_argument("--bands", type=int, default=16)
    t.add_argument("--hidden", type=int, default=1536)
    t.add_argument("--stacks", type=int, default=2)
    t.add_argument("--base", type=float, default=1.6180)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--lr", default="auto 1.0")
    t.add_argument("--max-change", default="auto 1.0")
    t.add_argument("--decimation", type=float, default=1.0)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--start", type=int, default=0)
    t.add_argument("--bytes", type=int, default=0)
    t.add_argument("--save-every", type=int, default=10_000_000)
    t.add_argument("--report-every", type=int, default=500_000)
    t.add_argument("--dream-every", type=int, default=50)
    t.add_argument("--dream-length", default="auto 200")
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
