"""
soma v8 — spectral online machine architecture.

    ░▒▓ soma ▓▒░

trace bank:
    256 channels × K bands of parallel exponential traces.
    fixed dynamics, no learned parameters. float64 on cpu.
    maps a byte stream to a bounded temporal spectrum.

forward path:
    pure function: bandpass features → logits.
    U projects features to hidden, W projects hidden to output.
    optional Wd direct residual bypassing the hidden layer.

weight update:
    analytical gradients, per-band confidence scaling, clipped steps.
    every band updates every step, scaled by observation resolution.
    no accumulators, no staggered firing, no gradient storage.

signal flow:
    training:   bytes[] → trace bank process_block → forward_batch → update
    generation: tap → forward → sample → tick → repeat (no learning)
    prompting:  tap → forward → learn from user byte → tick → repeat

decimation:
    decimation_band selects observation resolution. stride = base^band.
    band k confidence = min(1, base^(k - decimation_band)).
    bands at or above decimation_band: full confidence.
    bands below: geometrically reduced.
"""

import hashlib
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PHI = (1 + np.sqrt(5)) / 2
EPS = 1e-10
VOCAB = 256


# ─────────────────────────────────────────────────────────────────────
# terminal ui
# ─────────────────────────────────────────────────────────────────────

GLYPH = {
    'logo':     "    ░▒▓ soma ▓▒░",
    'bar_fill':  '▓',
    'bar_mid':   '▒',
    'bar_empty': '░',
    'sep':       '─',
    'bullet':    '·',
    'arrow':     '›',
    'spark':     '⚡',
    'wave':      '∿',
    'dot':       '•',
    'save':      '⟐',
    'load':      '⟐',
    'train':     '∿',
    'eval':      '⊘',
    'chat':      '⟡',
    'gen':       '◌',
}


def _sep(width=52):
    return GLYPH['sep'] * width


def _banner():
    print()
    print(_sep())
    print(GLYPH['logo'])
    print(_sep())


def _bar(frac, width=30):
    """compact progress bar."""
    filled = int(frac * width)
    mid = 1 if filled < width else 0
    empty = width - filled - mid
    return (GLYPH['bar_fill'] * filled +
            GLYPH['bar_mid'] * mid +
            GLYPH['bar_empty'] * empty)


def _fmt_bytes(n):
    if n >= 1e9: return f"{n / 1e9:.1f}B"
    if n >= 1e6: return f"{n / 1e6:.1f}M"
    if n >= 1e3: return f"{n / 1e3:.1f}K"
    return str(n)


def _fmt_params(n):
    if n >= 1e6: return f"{n / 1e6:.1f}M"
    if n >= 1e3: return f"{n / 1e3:.1f}K"
    return str(n)


def _prompt(text, default=""):
    result = input(f"  {GLYPH['arrow']} {text}").strip()
    return result if result else default


# ─────────────────────────────────────────────────────────────────────
# trace bank
# ─────────────────────────────────────────────────────────────────────

class TraceBank:
    """256 × K exponential traces. float64 on cpu.

    each of 256 byte channels has K traces at geometrically spaced
    decay rates. traces are the system's temporal memory. bandpass
    features (adjacent trace differences) are the system's input
    representation.

    float64 is required because slow-band decay rates approach 1.0
    and lose precision in float32. output features are cast to
    float32 for the forward path.
    """

    def __init__(self, n_bands, base, device):
        self.n_bands = n_bands
        self.base = base
        self.device = device
        self.n_features = VOCAB * n_bands

        # decay rates: alpha_k = 1 / base^k
        alphas_np = np.array(
            [1.0 / (base ** k) for k in range(n_bands)], dtype=np.float64)
        decay_np = 1.0 - alphas_np

        self.alphas = torch.from_numpy(alphas_np)
        self.decay = torch.from_numpy(decay_np)
        self.log_decay = torch.log(torch.clamp(self.decay, min=1e-300))

        # trace state: (256, K) float64 on cpu
        self.traces = torch.zeros(VOCAB, n_bands, dtype=torch.float64)

    def reset(self):
        self.traces.zero_()

    # ── single-sample operations ──

    def tick(self, byte_val):
        """update traces for one observed byte."""
        self.traces *= self.decay
        self.traces[byte_val] += self.alphas

    def tap(self):
        """read bandpass features. returns (256*K,) float32 on device.

        bandpass = difference between adjacent traces, isolating the
        frequency content between each pair of timescales. the slowest
        band has no slower neighbor and returns its trace directly.
        """
        bp = torch.empty_like(self.traces)
        bp[:, :-1] = self.traces[:, :-1] - self.traces[:, 1:]
        bp[:, -1] = self.traces[:, -1]
        return bp.reshape(-1).float().to(self.device)

    # ── block operations ──

    def advance(self, bytes_np):
        """advance traces through a byte sequence without computing features.

        uses closed-form IIR solution: the contribution of each byte to
        the final trace state is computed via weighted sums with exponential
        decay weights, avoiding sequential iteration.
        """
        N = len(bytes_np)
        if N == 0:
            return

        indices = torch.from_numpy(bytes_np.astype(np.int64))
        one_hot = torch.zeros(N, VOCAB, dtype=torch.float64)
        one_hot.scatter_(1, indices.unsqueeze(1), 1.0)

        pos = torch.arange(N, dtype=torch.float64)
        exponents = (N - 1) - pos
        weights = torch.exp(
            exponents.unsqueeze(1) * self.log_decay.unsqueeze(0))
        weighted_counts = one_hot.T @ weights

        decay_N = self.decay ** N
        self.traces *= decay_N.unsqueeze(0)
        self.traces += self.alphas.unsqueeze(0) * weighted_counts

    def process_block(self, indices_np):
        """compute trace snapshots for every position in a byte block.

        returns (N, 256*K) float32 on device. advances traces to the
        state after the full block.

        the sequential scan runs in float64. on cuda it runs on gpu
        (cuda supports float64). on mps/cpu it stays on cpu. bands
        are processed in memory-limited chunks.
        """
        N = len(indices_np)
        K = self.n_bands

        use_gpu = (self.device.type == 'cuda')
        compute_device = self.device if use_gpu else torch.device('cpu')

        indices = torch.from_numpy(
            indices_np.astype(np.int64)).to(compute_device)
        one_hot = torch.zeros(
            N, VOCAB, device=compute_device, dtype=torch.float64)
        one_hot.scatter_(1, indices.unsqueeze(1), 1.0)

        decay_cd = self.decay.to(compute_device)
        alphas_cd = self.alphas.to(compute_device)
        log_decay_cd = self.log_decay.to(compute_device)
        traces_cd = self.traces.to(compute_device)

        # limit working tensor to ~4GB: (N, 256, chunk) in float64
        BAND_CHUNK = max(1, min(K, int(4e9 / (N * VOCAB * 8))))
        lowpass = torch.empty(
            N, VOCAB, K, device=compute_device, dtype=torch.float64)

        for k_start in range(0, K, BAND_CHUNK):
            k_end = min(k_start + BAND_CHUNK, K)
            Kc = k_end - k_start
            dk = decay_cd[k_start:k_end]
            ak = alphas_cd[k_start:k_end]

            inp = one_hot.unsqueeze(2) * ak
            S = traces_cd[:, k_start:k_end].clone()
            states = torch.empty(
                N, VOCAB, Kc, device=compute_device, dtype=torch.float64)

            for t in range(N):
                states[t] = S
                S = dk * S + inp[t]

            lowpass[:, :, k_start:k_end] = states

        # bandpass: adjacent differences
        bandpass = torch.empty_like(lowpass)
        bandpass[:, :, :-1] = lowpass[:, :, :-1] - lowpass[:, :, 1:]
        bandpass[:, :, -1] = lowpass[:, :, -1]
        features = bandpass.reshape(N, -1).float().to(self.device)

        # advance traces to final state via closed-form
        pos = torch.arange(N, device=compute_device, dtype=torch.float64)
        exponents = (N - 1) - pos
        weights = torch.exp(
            exponents.unsqueeze(1) * log_decay_cd.unsqueeze(0))
        weighted_counts = one_hot.T @ weights
        decay_N = decay_cd ** N
        self.traces = (traces_cd * decay_N.unsqueeze(0) +
                       alphas_cd.unsqueeze(0) * weighted_counts).cpu()

        return features

    # ── state ──

    def state_numpy(self):
        return self.traces.numpy()

    def load_state(self, traces_np):
        self.traces = torch.from_numpy(
            traces_np.astype(np.float64)).clone()


# ─────────────────────────────────────────────────────────────────────
# decimation
# ─────────────────────────────────────────────────────────────────────

def compute_band_confidence(n_bands, base, decimation_band):
    """per-band confidence weights for a given decimation level.

    band k confidence = min(1.0, base^(k - decimation_band)).
    bands at or above decimation_band: confidence 1.0.
    bands below: geometrically reduced.

    returns (stride, confidence_array).
    """
    decimation_band = max(0, min(decimation_band, n_bands - 1))
    stride = max(1, int(round(base ** decimation_band)))
    confidence = np.array(
        [min(1.0, base ** (k - decimation_band)) for k in range(n_bands)],
        dtype=np.float64)
    return stride, confidence


# ─────────────────────────────────────────────────────────────────────
# soma
# ─────────────────────────────────────────────────────────────────────

class SOMA:
    """spectral online machine architecture.

    components:
        bank:   TraceBank — temporal memory, fixed dynamics
        U, W:   weight matrices — forward path (features → logits)
        Wd:     optional direct residual (features → logits, bypassing hidden)

    training: weight updates use analytical gradients with per-band
    confidence scaling and clipped steps. no automatic differentiation.
    """

    def __init__(self, n_bands=46, base=None, max_window=None,
                 hidden_dim=256, lr=0.1, max_change=0.1, weight_decay=1e-4,
                 batch_size=50000, decimation_band=0, device='auto',
                 direct_readout=False):
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.direct_readout = bool(direct_readout) if hidden_dim > 0 else False

        # base: from max_window, explicit, or default to phi
        if max_window is not None:
            self.base = max_window ** (1.0 / (n_bands - 1))
        elif base is not None:
            self.base = base
        else:
            self.base = PHI

        self.lr = lr
        self.max_change = max_change
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.decimation_band = decimation_band
        self.device = self._select_device(device)

        # trace bank
        self.bank = TraceBank(n_bands, base=self.base, device=self.device)
        self.n_features = self.bank.n_features
        self.max_window = self.base ** (n_bands - 1)

        # decimation confidence
        self._update_decimation()

        # per-band column indices for gradient scatter
        K = n_bands
        self._band_slices = []
        for k in range(K):
            cols = torch.arange(k, VOCAB * K, K, device=self.device)
            self._band_slices.append(cols)

        # weight matrices
        if hidden_dim > 0:
            self.hidden_budget = hidden_dim * 0.1
            self.u_norm = np.sqrt(self.n_features) * 0.1
            self.w_norm = np.sqrt(hidden_dim) * 0.1

            self.U = torch.randn(
                hidden_dim, self.n_features, device=self.device)
            self.W = torch.randn(VOCAB, hidden_dim, device=self.device)
            self._normalize_U()
            self._normalize_W()

            if self.direct_readout:
                self.wd_norm = np.sqrt(self.n_features) * 0.1
                self.Wd = torch.randn(
                    VOCAB, self.n_features, device=self.device)
                self._normalize_Wd()
            else:
                self.wd_norm = None
                self.Wd = None
        else:
            self.U = None
            self.hidden_budget = None
            self.u_norm = None
            self.wd_norm = None
            self.Wd = None

            self.w_norm = np.sqrt(self.n_features) * 0.1
            self.W = torch.randn(
                VOCAB, self.n_features, device=self.device)
            self._normalize_W()

        self.bytes_seen = 0
        self.checkpoint_history = []

    # ── forward path ──

    def _forward(self, features):
        """single-sample forward: features (n_features,) → logits (256,)."""
        if self.hidden_dim > 0:
            hidden = F.relu(self.U @ features)
            h_sum = hidden.sum() + EPS
            hidden_norm = hidden * (self.hidden_budget / h_sum)
            logits = self.W @ hidden_norm
            if self.Wd is not None:
                logits = logits + self.Wd @ features
            return logits
        else:
            return self.W @ features

    def _forward_batch(self, features_batch):
        """batched forward: (N, n_features) → (logits, cache).

        cache contains intermediate values for gradient computation.
        """
        if self.hidden_dim > 0:
            hidden = features_batch @ self.U.T
            hidden_relu = F.relu(hidden)
            hidden_sum = hidden_relu.sum(dim=1, keepdim=True) + EPS
            hidden_norm = hidden_relu * (self.hidden_budget / hidden_sum)
            logits = hidden_norm @ self.W.T
            if self.Wd is not None:
                logits = logits + features_batch @ self.Wd.T
            return logits, {
                'hidden': hidden,
                'hidden_relu': hidden_relu,
                'hidden_sum': hidden_sum,
                'hidden_norm': hidden_norm,
                'X': features_batch,
            }
        else:
            logits = features_batch @ self.W.T
            return logits, {'X': features_batch}

    # ── weight updates ──

    def _update_weights(self, errors, cache, n):
        """compute analytical gradients and apply confidence-scaled updates.

        every band updates every step. confidence determines the scale
        of each band's gradient contribution. W is updated with full
        confidence (it maps hidden → output, not band-indexed). U and
        Wd are updated per-band with confidence scaling.
        """
        K = self.n_bands

        with torch.no_grad():
            if self.hidden_dim > 0:
                hidden_norm = cache['hidden_norm']
                hidden = cache['hidden']
                hidden_sum = cache['hidden_sum']
                X = cache['X']

                # W gradient and update (not band-indexed)
                grad_W = (errors.T @ hidden_norm) / n
                self._apply_clipped_update(self.W, grad_W)
                self.W *= (1.0 - self.weight_decay)
                self._normalize_W()

                # gradient through budget normalization and relu
                grad_hidden_norm = errors @ self.W
                scale = self.hidden_budget / hidden_sum
                grad_hidden_relu = (
                    grad_hidden_norm * scale
                    - (grad_hidden_norm * hidden_norm).sum(
                        dim=1, keepdim=True)
                    * scale / self.hidden_budget
                )
                grad_hidden = grad_hidden_relu * (hidden > 0).float()

                # U gradient, applied per-band with confidence
                grad_U_full = (grad_hidden.T @ X) / n
                grad_Wd_full = None
                if self.Wd is not None:
                    grad_Wd_full = (errors.T @ X) / n

                for k in range(K):
                    cols = self._band_slices[k]
                    c = self._band_confidence[k]
                    self._apply_band_update(
                        self.U, cols, grad_U_full[:, cols] * c)
                    if grad_Wd_full is not None:
                        self._apply_band_update(
                            self.Wd, cols, grad_Wd_full[:, cols] * c)

                self.U *= (1.0 - self.weight_decay)
                self._normalize_U()
                if self.Wd is not None:
                    self.Wd *= (1.0 - self.weight_decay)
                    self._normalize_Wd()

            else:
                X = cache['X']
                grad_W_full = (errors.T @ X) / n

                for k in range(K):
                    cols = self._band_slices[k]
                    c = self._band_confidence[k]
                    self._apply_band_update(
                        self.W, cols, grad_W_full[:, cols] * c)
                self.W *= (1.0 - self.weight_decay)
                self._normalize_W()

    def _apply_clipped_update(self, param, grad):
        """clipped gradient step: change per element ≤ max_change × |element|."""
        raw_delta = self.lr * grad
        max_delta = self.max_change * param.abs()
        delta = torch.clamp(raw_delta, -max_delta, max_delta)
        param -= delta

    def _apply_band_update(self, param, cols, grad_band):
        """clipped gradient step on a subset of columns (one band)."""
        band_vals = param[:, cols]
        raw_delta = self.lr * grad_band
        max_delta = self.max_change * band_vals.abs()
        delta = torch.clamp(raw_delta, -max_delta, max_delta)
        param[:, cols] = band_vals - delta

    # ── weight normalization ──

    def _normalize_U(self):
        if self.U is not None:
            with torch.no_grad():
                norms = self.U.norm(dim=1, keepdim=True)
                self.U.mul_(self.u_norm / (norms + EPS))

    def _normalize_W(self):
        with torch.no_grad():
            norms = self.W.norm(dim=1, keepdim=True)
            self.W.mul_(self.w_norm / (norms + EPS))

    def _normalize_Wd(self):
        if self.Wd is not None:
            with torch.no_grad():
                norms = self.Wd.norm(dim=1, keepdim=True)
                self.Wd.mul_(self.wd_norm / (norms + EPS))

    # ── decimation ──

    def _update_decimation(self):
        self._stride, confidence = compute_band_confidence(
            self.n_bands, self.base, self.decimation_band)
        self._band_confidence = torch.from_numpy(
            confidence).float().to(self.device)

    @staticmethod
    def _select_device(device):
        if device != 'auto':
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    # ── training ──

    def train(self, corpus_path, epochs=1,
              report_every=1_000_000, save_every=0, save_path="model.pt",
              start_byte=0):
        corpus = np.fromfile(corpus_path, dtype=np.uint8)
        if start_byte > 0:
            corpus = corpus[start_byte:]
        N = len(corpus)
        batch_size = self.batch_size
        stride = self._stride

        start_str = f", start={_fmt_bytes(start_byte)}" if start_byte > 0 else ""
        print(f"\n  {GLYPH['train']} training {corpus_path} "
              f"({_fmt_bytes(N)} bytes{start_str})")
        print(f"    batch={batch_size:,} "
              f"{GLYPH['bullet']} decimation_band={self.decimation_band} "
              f"(stride={stride}) "
              f"{GLYPH['bullet']} {epochs} epoch{'s' if epochs != 1 else ''}")
        print()

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            samples = 0
            t0 = time.time()
            last_report = 0
            last_save = 0

            if stride == 1:
                for batch_start in range(0, N, batch_size):
                    batch_end = min(batch_start + batch_size, N)
                    n = batch_end - batch_start
                    chunk = corpus[batch_start:batch_end]

                    Xt = self.bank.process_block(chunk)
                    yt = torch.from_numpy(
                        chunk.astype(np.int64)).to(self.device)

                    loss, acc = self._train_batch(Xt, yt, n)
                    total_loss += loss
                    correct += acc
                    samples += n
                    self.bytes_seen += n

                    if report_every and batch_end - last_report >= report_every:
                        self._report(
                            epoch, epochs, batch_end, N,
                            total_loss, correct, samples, t0)
                        last_report = batch_end
                    if save_every and batch_end - last_save >= save_every:
                        self.save(save_path)
                        last_save = batch_end
            else:
                features_buf = torch.empty(
                    batch_size, self.n_features, device=self.device)
                targets_buf = torch.empty(
                    batch_size, dtype=torch.int64, device=self.device)
                pos = 0
                n_collected = 0

                while pos < N:
                    features_buf[n_collected] = self.bank.tap()
                    targets_buf[n_collected] = int(corpus[pos])
                    n_collected += 1

                    advance_end = min(pos + stride, N)
                    chunk = corpus[pos:advance_end]
                    self.bank.advance(chunk)
                    self.bytes_seen += len(chunk)
                    pos = advance_end

                    if n_collected >= batch_size:
                        loss, acc = self._train_batch(
                            features_buf[:n_collected],
                            targets_buf[:n_collected],
                            n_collected)
                        total_loss += loss
                        correct += acc
                        samples += n_collected
                        n_collected = 0

                    if report_every and pos - last_report >= report_every:
                        if n_collected > 0:
                            loss, acc = self._train_batch(
                                features_buf[:n_collected],
                                targets_buf[:n_collected],
                                n_collected)
                            total_loss += loss
                            correct += acc
                            samples += n_collected
                            n_collected = 0
                        self._report(
                            epoch, epochs, pos, N,
                            total_loss, correct, samples, t0)
                        last_report = pos
                    if save_every and pos - last_save >= save_every:
                        if n_collected > 0:
                            loss, acc = self._train_batch(
                                features_buf[:n_collected],
                                targets_buf[:n_collected],
                                n_collected)
                            total_loss += loss
                            correct += acc
                            samples += n_collected
                            n_collected = 0
                        self.save(save_path)
                        last_save = pos

                if n_collected > 0:
                    loss, acc = self._train_batch(
                        features_buf[:n_collected],
                        targets_buf[:n_collected],
                        n_collected)
                    total_loss += loss
                    correct += acc
                    samples += n_collected

            elapsed = time.time() - t0
            avg = total_loss / samples if samples > 0 else 0
            bpb = avg / np.log(2)
            acc = 100 * correct / samples if samples > 0 else 0
            print(f"    epoch {epoch + 1} done "
                  f"{GLYPH['bullet']} {avg:.4f} nats ({bpb:.2f} bpb) "
                  f"{GLYPH['bullet']} {acc:.1f}% "
                  f"{GLYPH['bullet']} {elapsed:.1f}s "
                  f"{GLYPH['bullet']} {N / elapsed:,.0f} b/s")

    def _train_batch(self, Xt, yt, n):
        """forward, compute cross-entropy error, update weights."""
        with torch.no_grad():
            logits, cache = self._forward_batch(Xt)
            probs = F.softmax(logits, dim=1)
            idx = torch.arange(n, device=self.device)
            loss = -torch.log(probs[idx, yt] + EPS).sum().item()
            acc = (logits.argmax(1) == yt).sum().item()
            probs[idx, yt] -= 1.0

        self._update_weights(probs, cache, n)
        return loss, acc

    def _report(self, epoch, epochs, pos, total, loss, correct, samples, t0):
        elapsed = time.time() - t0
        avg = loss / samples if samples > 0 else 0
        bpb = avg / np.log(2)
        acc = 100 * correct / samples if samples > 0 else 0
        bps = pos / elapsed if elapsed > 0 else 0
        frac = pos / total if total > 0 else 0
        print(f"    [{epoch + 1}/{epochs}] "
              f"{_bar(frac)} {frac * 100:4.1f}% "
              f"{GLYPH['bullet']} {avg:.3f} nats ({bpb:.2f} bpb) "
              f"{acc:.1f}% "
              f"{GLYPH['bullet']} {bps:,.0f} b/s")

    # ── evaluation ──

    def evaluate(self, corpus_path):
        corpus = np.fromfile(corpus_path, dtype=np.uint8)
        N = len(corpus)
        batch_size = self.batch_size
        print(f"\n  {GLYPH['eval']} evaluating {corpus_path} "
              f"({_fmt_bytes(N)} bytes)")

        self.bank.reset()
        total_loss = 0.0
        total_correct = 0
        t0 = time.time()

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            n = batch_end - batch_start
            chunk = corpus[batch_start:batch_end]

            Xt = self.bank.process_block(chunk)
            yt = torch.from_numpy(chunk.astype(np.int64)).to(self.device)

            with torch.no_grad():
                logits, _ = self._forward_batch(Xt)
                probs = F.softmax(logits, dim=1)
                idx = torch.arange(n, device=self.device)
                total_loss -= torch.log(probs[idx, yt] + EPS).sum().item()
                total_correct += (logits.argmax(1) == yt).sum().item()

        elapsed = time.time() - t0
        avg = total_loss / N
        bpb = avg / np.log(2)
        acc = 100 * total_correct / N
        print(f"    {avg:.4f} nats ({bpb:.2f} bpb) "
              f"{GLYPH['bullet']} {acc:.1f}% "
              f"{GLYPH['bullet']} {elapsed:.1f}s "
              f"{GLYPH['bullet']} {N / elapsed:,.0f} b/s")
        return avg

    # ── generation ──

    def generate(self, length=200, temperature=0.8):
        """autoregressive generation. no weight updates.

        tap → forward → sample → tick → repeat.
        generated bytes are fed back into the trace bank as memory
        but produce no learning signal.
        """
        output = []

        for _ in range(length):
            features = self.bank.tap()
            logits = self._forward(features)

            logits = logits / temperature
            probs = F.softmax(logits, dim=0)
            byte_val = torch.multinomial(probs, 1).item()

            if byte_val == ord('\n'):
                break
            output.append(chr(byte_val) if 32 <= byte_val < 127 else '.')

            self.bank.tick(byte_val)

        return ''.join(output)

    # ── online learning (prompt ingestion only) ──

    def _learn_single(self, features, logits, target_byte):
        """single-sample weight update from an external target byte.

        used during online prompt ingestion where the true next byte
        is known. not used during generation — the model's own samples
        carry no external error signal.
        """
        with torch.no_grad():
            probs = F.softmax(logits, dim=0)
            error = probs.clone()
            error[target_byte] -= 1.0

            error_batch = error.unsqueeze(0)
            features_batch = features.unsqueeze(0)

            if self.hidden_dim > 0:
                hidden_pre = self.U @ features
                hidden_relu = F.relu(hidden_pre)
                h_sum = hidden_relu.sum() + EPS
                hidden_norm = hidden_relu * (self.hidden_budget / h_sum)
                cache = {
                    'hidden': hidden_pre.unsqueeze(0),
                    'hidden_relu': hidden_relu.unsqueeze(0),
                    'hidden_sum': h_sum.unsqueeze(0).unsqueeze(0),
                    'hidden_norm': hidden_norm.unsqueeze(0),
                    'X': features_batch,
                }
            else:
                cache = {'X': features_batch}

            self._update_weights(error_batch, cache, 1)

    def ingest_prompt(self, text, online=False):
        """feed text through the trace bank.

        if online=True, also updates weights against each user byte.
        each byte is an external target — the error signal is informative.

        if online=False, advances the trace bank without computing
        features or updating weights.
        """
        prompt_bytes = np.array([ord(c) for c in text], dtype=np.uint8)

        if online:
            for b in prompt_bytes:
                features = self.bank.tap()
                logits = self._forward(features)
                self._learn_single(features, logits, int(b))
                self.bank.tick(int(b))
                self.bytes_seen += 1
        else:
            self.bank.advance(prompt_bytes)

    # ── save / load ──

    def _checkpoint_id(self):
        """sha256 hash of weights + traces + fixed config."""
        h = hashlib.sha256()
        h.update(self.W.cpu().numpy().tobytes())
        if self.U is not None:
            h.update(self.U.cpu().numpy().tobytes())
        if self.Wd is not None:
            h.update(self.Wd.cpu().numpy().tobytes())
        h.update(self.bank.state_numpy().tobytes())
        for val in [self.n_bands, self.hidden_dim, self.base,
                    bool(self.direct_readout)]:
            h.update(str(val).encode())
        return h.hexdigest()

    def save(self, path):
        current_id = self._checkpoint_id()
        history = self.checkpoint_history + [current_id]
        data = {
            'W': self.W.cpu(),
            'traces': self.bank.state_numpy(),
            'n_bands': self.n_bands,
            'base': self.base,
            'hidden_dim': self.hidden_dim,
            'lr': self.lr,
            'max_change': self.max_change,
            'weight_decay': self.weight_decay,
            'w_norm': self.w_norm,
            'bytes_seen': self.bytes_seen,
            'batch_size': self.batch_size,
            'decimation_band': self.decimation_band,
            'direct_readout': bool(self.direct_readout),
            'soma_version': 'v8',
            'checkpoint_id': current_id,
            'checkpoint_history': history,
        }
        if self.U is not None:
            data['U'] = self.U.cpu()
            data['u_norm'] = self.u_norm
            data['hidden_budget'] = self.hidden_budget
        if self.Wd is not None:
            data['Wd'] = self.Wd.cpu()
            data['wd_norm'] = self.wd_norm

        torch.save(data, path)
        print(f"    {GLYPH['save']} saved {path} · {current_id[:12]}")

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)

        self.W = ckpt['W'].float().to(self.device)
        self.bank.load_state(ckpt['traces'])

        self.lr = ckpt.get('lr', self.lr)
        self.max_change = ckpt.get('max_change', self.max_change)
        self.weight_decay = ckpt.get('weight_decay',
                                     ckpt.get('shrinkage', self.weight_decay))
        self.w_norm = ckpt.get('w_norm', self.w_norm)
        self.bytes_seen = ckpt.get('bytes_seen', 0)
        self.batch_size = ckpt.get('batch_size', self.batch_size)
        self.checkpoint_history = ckpt.get('checkpoint_history', [])

        if 'decimation_band' in ckpt:
            self.decimation_band = ckpt['decimation_band']
        elif 'downsample' in ckpt:
            ds = ckpt['downsample']
            if ds <= 1:
                self.decimation_band = 0
            else:
                self.decimation_band = max(0, int(round(
                    np.log(ds) / np.log(self.base))))

        if self.hidden_dim > 0:
            if 'U' in ckpt:
                self.U = ckpt['U'].float().to(self.device)
            self.u_norm = ckpt.get('u_norm', self.u_norm)
            self.hidden_budget = ckpt.get(
                'hidden_budget', self.hidden_budget)
            self._normalize_U()
            self._normalize_W()
            if self.Wd is not None and 'Wd' in ckpt:
                self.Wd = ckpt['Wd'].float().to(self.device)
                self.wd_norm = ckpt.get('wd_norm', self.wd_norm)
                self._normalize_Wd()
        else:
            self._normalize_W()

        self._update_decimation()
        print(f"    {GLYPH['load']} loaded {path}")

    # ── display ──

    def print_config(self):
        if self.hidden_dim > 0:
            params = self.U.numel() + self.W.numel()
            if self.Wd is not None:
                params += self.Wd.numel()
        else:
            params = self.W.numel()

        hidden_str = (f"hidden={self.hidden_dim:,}"
                      if self.hidden_dim > 0 else "linear")
        if self.hidden_dim > 0 and self.Wd is not None:
            hidden_str += " + direct"

        n_full = sum(1 for k in range(self.n_bands)
                     if self._band_confidence[k] >= 1.0)
        band_str = f"{self.n_bands} bands"
        if self.decimation_band > 0:
            band_str = (f"{self.n_bands} bands, {n_full} full confidence "
                        f"(decimation_band={self.decimation_band}, "
                        f"stride={self._stride})")

        print(f"\n  {GLYPH['dot']} soma v8 {GLYPH['bullet']} {self.device} "
              f"{GLYPH['bullet']} {_fmt_bytes(self.bytes_seen)} seen")
        print(f"    {band_str}")
        print(f"    base={self.base:.4f} "
              f"{GLYPH['bullet']} range={self.max_window:,.0f} "
              f"{GLYPH['bullet']} {hidden_str} "
              f"{GLYPH['bullet']} {_fmt_params(params)} params")
        print(f"    lr={self.lr} "
              f"{GLYPH['bullet']} max_change={self.max_change} "
              f"{GLYPH['bullet']} weight_decay={self.weight_decay}")
        print()


# ─────────────────────────────────────────────────────────────────────
# cli
# ─────────────────────────────────────────────────────────────────────

def main():
    _banner()

    mode = _prompt("mode (train/eval/chat): ")

    if mode == "train":
        corpus = _prompt("corpus: ")
        if not Path(corpus).exists():
            return print(f"  not found: {corpus}")

        ckpt = _prompt("checkpoint (enter for new): ")
        if ckpt and Path(ckpt).exists():
            cfg = torch.load(ckpt, map_location='cpu', weights_only=False)
            saved_lr = cfg.get('lr', 0.1)
            saved_mc = cfg.get('max_change', 0.1)
            saved_wd = cfg.get('weight_decay',
                               cfg.get('shrinkage', 1e-4))
            saved_bs = cfg.get('batch_size', 50000)
            saved_db = cfg.get('decimation_band',
                               cfg.get('downsample', 0))
            if 'decimation_band' not in cfg and 'downsample' in cfg:
                ds_val = cfg['downsample']
                base_val = cfg.get('base', PHI)
                saved_db = 0 if ds_val <= 1 else max(0, int(round(
                    np.log(ds_val) / np.log(base_val))))
            lr = float(_prompt(f"lr [{saved_lr}]: ", str(saved_lr)))
            mc = float(_prompt(
                f"max_change [{saved_mc}]: ", str(saved_mc)))
            wd = float(_prompt(
                f"weight_decay [{saved_wd}]: ", str(saved_wd)))
            bs = int(_prompt(f"batch [{saved_bs}]: ", str(saved_bs)))
            db = int(_prompt(
                f"decimation_band [{saved_db}]: ", str(saved_db)))

            model = SOMA(
                cfg.get('n_bands', cfg.get('num_timescales', 46)),
                base=cfg.get('base', PHI),
                hidden_dim=cfg.get('hidden_dim', 256),
                lr=lr, max_change=mc, weight_decay=wd,
                batch_size=bs, decimation_band=db,
                direct_readout=bool(cfg.get('direct_readout', False)))
            model.load(ckpt)
            model.lr = lr
            model.max_change = mc
            model.weight_decay = wd
            model.batch_size = bs
            model.decimation_band = db
            model._update_decimation()
        else:
            bands = int(_prompt("bands [46]: ", "46"))
            range_str = _prompt(
                "range (base or window) [2500000000]: ", "2500000000")
            val = float(range_str)
            if val < 100:
                base, max_window = val, None
            else:
                base, max_window = None, val
            hd = int(_prompt("hidden (0=linear) [256]: ", "256"))
            lr = float(_prompt("lr [0.1]: ", "0.1"))
            mc = float(_prompt("max_change [0.1]: ", "0.1"))
            wd = float(_prompt("weight_decay [0.0001]: ", "0.0001"))
            bs = int(_prompt("batch [50000]: ", "50000"))
            ds = int(_prompt("decimation_band [0]: ", "0"))
            dr = int(_prompt("direct readout (0/1) [0]: ", "0"))

            model = SOMA(bands, base=base, max_window=max_window,
                         hidden_dim=hd, lr=lr, max_change=mc,
                         weight_decay=wd, batch_size=bs,
                         decimation_band=ds, direct_readout=bool(dr))

        model.print_config()

        epochs = int(_prompt("epochs [1]: ", "1"))
        start_byte = int(_prompt("start byte [0]: ", "0"))
        report = int(_prompt("report every [1000000]: ", "1000000"))
        save_every = int(_prompt("save every (0=end) [0]: ", "0"))
        save_path = _prompt("save path [model.pt]: ", "model.pt")

        model.train(corpus, epochs=epochs, start_byte=start_byte,
                    report_every=report, save_every=save_every,
                    save_path=save_path)
        model.save(save_path)

    elif mode == "eval":
        ckpt = _prompt("checkpoint: ")
        corpus = _prompt("corpus: ")
        if not Path(ckpt).exists() or not Path(corpus).exists():
            return print("  file not found")
        cfg = torch.load(ckpt, map_location='cpu', weights_only=False)
        model = SOMA(
            cfg.get('n_bands', cfg.get('num_timescales', 46)),
            base=cfg.get('base', PHI),
            hidden_dim=cfg.get('hidden_dim', 256),
            batch_size=cfg.get('batch_size', 50000),
            direct_readout=bool(cfg.get('direct_readout', False)))
        model.load(ckpt)
        model.print_config()
        model.evaluate(corpus)

    elif mode == "chat":
        ckpt = _prompt("checkpoint: ")
        if not Path(ckpt).exists():
            return print(f"  not found: {ckpt}")
        cfg = torch.load(ckpt, map_location='cpu', weights_only=False)
        model = SOMA(
            cfg.get('n_bands', cfg.get('num_timescales', 46)),
            base=cfg.get('base', PHI),
            hidden_dim=cfg.get('hidden_dim', 256),
            direct_readout=bool(cfg.get('direct_readout', False)))
        model.load(ckpt)
        model.print_config()

        temp = float(_prompt("temperature [0.8]: ", "0.8"))
        maxlen = int(_prompt("max length [200]: ", "200"))
        online = _prompt(
            "online learning (y/n) [n]: ", "n").lower() in ('y', 'yes')

        if online:
            lr_str = _prompt(f"lr [{model.lr}]: ", str(model.lr))
            mc_str = _prompt(
                f"max_change [{model.max_change}]: ", str(model.max_change))
            model.lr = float(lr_str)
            model.max_change = float(mc_str)
            model.batch_size = 1
            model.decimation_band = 0
            model._update_decimation()
            print(f"    online learning enabled "
                  f"(learns from your input, not its own output)")

        print(f"\n  {GLYPH['chat']} chat "
              f"{GLYPH['bullet']} temp={temp} "
              f"{GLYPH['bullet']} online={online} "
              f"{GLYPH['bullet']} 'quit' to exit")
        print()

        while True:
            try:
                user = input(f"  you {GLYPH['arrow']} ")
            except EOFError:
                break
            if user.lower() in ('quit', 'q', 'exit'):
                break
            if user.lower() == 'save':
                save_path = _prompt("save path: ", ckpt)
                model.save(save_path)
                continue
            if user:
                model.ingest_prompt(user + ' ', online=online)
                response = model.generate(
                    length=maxlen, temperature=temp)
                print(f"  {GLYPH['gen']} {GLYPH['arrow']} {response}\n")

        if _prompt(
                "save state? (y/n) [y]: ", "y").lower() in ('y', 'yes'):
            save_path = _prompt("save path: ", ckpt)
            model.save(save_path)


if __name__ == '__main__':
    main()
