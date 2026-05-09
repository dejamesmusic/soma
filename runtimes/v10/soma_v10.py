"""
soma v10 — spectral online machine architecture · device-aware variant.

    ░▒▓ soma ▓▒░

implements the soma v8 algorithm spec with device-specific fast paths:

    • trace bank lives on the active device (cpu/cuda/mps) in float32
      where K ≤ 35, falling back to cpu+float64 for K > 35 since slow
      band decay rates round to 1.0 in float32.

    • mps cannot do fp64. for K > 35 on a mac, the bank runs on cpu in
      fp64; weights stay on mps in fp32; features are transferred at
      the matmul boundary. this preserves checkpoint compatibility for
      large-K models.

    • forward and backward kernel implementations are dispatched at
      construction time based on device. metal kernels for apple
      silicon, triton (untested) for cuda, pure pytorch fallback
      everywhere. all are mathematically equivalent.

    • checkpoint format is byte-compatible with soma.py: traces are
      stored as float64 numpy regardless of runtime dtype, so
      checkpoints can be moved between devices freely.

intent:
    test the size-vs-time tradeoff. tiny model + many fast updates
    versus large model + few slow updates, at fixed wall-clock.
    everything-on-device is the prerequisite for the inner loop
    being fast enough to evaluate the thesis.
"""

import hashlib
import os
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Backend dispatch — try to import each implementation. Each module exposes
# the same fast-path API (bandpass_W, bandpass_T_W, fast_forward,
# fast_backward, is_available). The SOMA class picks the best one for
# the active device at construction time.
try:
    import _impl_metal
    _METAL_AVAILABLE = True
    soma_metal = _impl_metal  # legacy name kept for slow-path code below
except ImportError:
    _METAL_AVAILABLE = False
    soma_metal = None

try:
    import _impl_triton
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    _impl_triton = None

try:
    import _impl_torch
    _TORCH_IMPL_AVAILABLE = True
except ImportError:
    _TORCH_IMPL_AVAILABLE = False
    _impl_torch = None

PHI = (1 + np.sqrt(5)) / 2
EPS = 1e-10
VOCAB = 256

# float32 caps usable bands. φ^-k < 2^-23 (float32 mantissa) at
# k ≈ 33. allow a small buffer; past 35 bands behaviour degrades.
FP32_MAX_BANDS = 35


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


def _parse_auto_or_float(s, default_base=1.0):
    """Parse either a literal float, "auto", or "auto N".

    Returns (value, is_auto, base):
      - "0.1"      → (0.1, False, default_base)
      - "auto"     → (default_base, True, default_base)
      - "auto 0.5" → (0.5, True, 0.5)

    The returned `value` is the immediate value to use (non-auto: literal,
    auto: the base which is what auto-mode starts at). `is_auto` flags
    auto mode. `base` is the auto base scalar.
    """
    s = s.strip().lower()
    if s.startswith('auto'):
        rest = s[4:].strip()
        if rest:
            base = float(rest)
        else:
            base = default_base
        return base, True, base
    return float(s), False, default_base


def _device_supports_fp64(device):
    """metal does not support float64. cpu and cuda do."""
    return device.type in ('cpu', 'cuda')


# Convention: bare filenames at the cli (no "/" or "~", no leading ".")
# are auto-routed into the conventional folder for that file type:
#   corpus files     →   data/
#   checkpoint files →   checkpoints/
# Users typing a path with directory separators bypass this convention
# and the path is used literally.

_RUNTIME_DIR = os.environ.get("SOMA_HOME", "")
DATA_DIR = os.path.join(_RUNTIME_DIR, "data") if _RUNTIME_DIR else "data"
CHECKPOINT_DIR = (os.path.join(_RUNTIME_DIR, "checkpoints")
                  if _RUNTIME_DIR else "checkpoints")


def _resolve_path(path, kind):
    """Map a user-typed path to its on-disk location."""
    if not path:
        return path
    if ('/' in path or '\\' in path or
            path.startswith('~') or path.startswith('.')):
        return os.path.expanduser(path)
    folder = DATA_DIR if kind == 'corpus' else CHECKPOINT_DIR
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, path)


# ─────────────────────────────────────────────────────────────────────
# trace bank — on-device
# ─────────────────────────────────────────────────────────────────────

class TraceBank:
    """256 × K exponential traces, on a chosen device.

    dtype is float64 when the device supports it (cpu, cuda) and float32
    on mps. for K > FP32_MAX_BANDS (~35), float64 is required to avoid
    slow-band decay rates rounding to 1.0; in that case the bank runs on
    cpu regardless of the compute device, and features are transferred
    to the compute device at the matmul boundary.

    on mps with K ≤ FP32_MAX_BANDS, the bank runs in float32 directly on
    mps to keep the entire pipeline on-device.
    """

    def __init__(self, n_bands, base, device):
        """device is the *requested* device. for K > FP32_MAX_BANDS we
        may transparently move the bank to cpu (since mps lacks fp64).
        """
        self.n_bands = n_bands
        self.base = base

        # decide where the bank actually runs based on K and device
        # capability. callers can read self.device to find out.
        if _device_supports_fp64(device):
            self.device = device
            self.dtype = torch.float64
        elif n_bands <= FP32_MAX_BANDS:
            self.device = device
            self.dtype = torch.float32
        else:
            # K is too large for fp32; this device can't do fp64. fall back
            # to cpu+fp64. compute device stays separate; the SOMA class
            # transfers features at the boundary.
            self.device = torch.device('cpu')
            self.dtype = torch.float64

        self.n_features = VOCAB * n_bands

        # decay rates: alpha_k = 1 / base^k. compute in float64 for the
        # constants then cast — gives best precision regardless of dtype.
        alphas_f64 = np.array(
            [1.0 / (base ** k) for k in range(n_bands)], dtype=np.float64)
        decay_f64 = 1.0 - alphas_f64

        self.alphas = torch.from_numpy(alphas_f64).to(
            device=self.device, dtype=self.dtype)
        self.decay = torch.from_numpy(decay_f64).to(
            device=self.device, dtype=self.dtype)
        # log_decay is used in closed-form scans (advance, process_block) as
        # exp(N · log_decay). For band 0, decay=0, so we'd get log(0) = -inf,
        # then 0 · -inf = NaN. We clamp decay before log to avoid this.
        # On fp32 (mps), denormals flush to zero on Apple GPUs, so we must
        # clamp well above the denormal threshold (~1e-38). 1e-20 is safe.
        clamp_min = 1e-20 if self.dtype == torch.float32 else 1e-300
        self.log_decay = torch.log(torch.clamp(self.decay, min=clamp_min))

        # trace state: (256, K) on the bank's resolved device
        self.traces = torch.zeros(
            VOCAB, n_bands, dtype=self.dtype, device=self.device)

    def reset(self):
        self.traces.zero_()

    # ── single-sample operations ──

    def tick(self, byte_val):
        """update traces for one observed byte."""
        self.traces *= self.decay
        self.traces[byte_val] += self.alphas

    def tap(self):
        """read bandpass features. returns (256*K,) float32 on device.

        bandpass = difference between adjacent traces. slowest band
        passes through directly.
        """
        bp = torch.empty_like(self.traces)
        bp[:, :-1] = self.traces[:, :-1] - self.traces[:, 1:]
        bp[:, -1] = self.traces[:, -1]
        return bp.reshape(-1).float()

    # ── block operations ──

    def advance(self, bytes_np):
        """advance traces through a byte sequence without computing
        features. closed-form IIR — O(N) in N but vectorised, no
        sequential scan.
        """
        N = len(bytes_np)
        if N == 0:
            return

        indices = torch.from_numpy(bytes_np.astype(np.int64)).to(self.device)
        one_hot = torch.zeros(
            N, VOCAB, dtype=self.dtype, device=self.device)
        one_hot.scatter_(1, indices.unsqueeze(1), 1.0)

        pos = torch.arange(N, dtype=self.dtype, device=self.device)
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
        post-block state.

        on mps with metal kernel available: dispatches one fused metal
        compute kernel that does the entire sequential scan on-gpu in
        one launch, no per-timestep python or pytorch dispatch.

        on cpu/cuda or when metal is unavailable: runs the original
        sequential scan in pytorch.
        """
        N = len(indices_np)
        K = self.n_bands

        # try metal fast path first if applicable
        if (_METAL_AVAILABLE and self.device.type == 'mps'
                and self.dtype == torch.float32):
            return self._process_block_metal(indices_np)

        return self._process_block_torch(indices_np)

    def _process_block_metal(self, indices_np):
        """metal kernel scan path. one dispatch for the entire scan,
        one for the bandpass.

        kernel writes states in (N, V, K) layout — same as the feature
        layout we need at output. bandpass kernel turns states into
        features without any python-side tensor allocation pressure.
        """
        N = len(indices_np)
        K = self.n_bands

        bytes_tensor = torch.from_numpy(indices_np).to(
            device=self.device, dtype=torch.uint8)

        # kernel uses (K, V) layout; we store (V, K). transpose at boundary.
        traces_KV = self.traces.T.contiguous()

        # one kernel dispatch — sequential scan happens entirely on-gpu.
        # states_NVK has shape (N, V, K) — already the right layout.
        states_NVK = soma_metal.trace_scan_metal(
            traces_KV, bytes_tensor, self.decay, self.alphas)

        if states_NVK is None:
            # kernel compile failed at runtime — fall back
            return self._process_block_torch(indices_np)

        # write back the post-block traces in (V, K) layout
        self.traces = traces_KV.T.contiguous()

        # bandpass via kernel — single dispatch covers the whole tensor
        # instead of pytorch's two slice writes (which on mps cost ~4 ops
        # plus a 320MB allocation at batch=10000).
        features_NVK = soma_metal.bandpass_metal(states_NVK)
        if features_NVK is None:
            features_NVK = soma_metal.bandpass_torch(states_NVK)

        # already (N, V, K), just flatten to features
        return features_NVK.reshape(N, -1)

    def _process_block_torch(self, indices_np):
        """pytorch sequential scan — original implementation."""
        N = len(indices_np)
        K = self.n_bands

        indices = torch.from_numpy(
            indices_np.astype(np.int64)).to(self.device)
        one_hot = torch.zeros(
            N, VOCAB, device=self.device, dtype=self.dtype)
        one_hot.scatter_(1, indices.unsqueeze(1), 1.0)

        # working-memory budget: 1GB worth of trace state per chunk.
        # sized in self.dtype bytes. mps has limited shared memory so
        # smaller chunks are safer there too.
        bytes_per_elem = 8 if self.dtype == torch.float64 else 4
        BAND_CHUNK = max(1, min(K, int(1e9 / (N * VOCAB * bytes_per_elem))))

        lowpass = torch.empty(
            N, VOCAB, K, device=self.device, dtype=self.dtype)

        for k_start in range(0, K, BAND_CHUNK):
            k_end = min(k_start + BAND_CHUNK, K)
            Kc = k_end - k_start
            dk = self.decay[k_start:k_end]
            ak = self.alphas[k_start:k_end]

            inp = one_hot.unsqueeze(2) * ak
            S = self.traces[:, k_start:k_end].clone()
            states = torch.empty(
                N, VOCAB, Kc, device=self.device, dtype=self.dtype)

            for t in range(N):
                states[t] = S
                S = dk * S + inp[t]

            lowpass[:, :, k_start:k_end] = states

        # bandpass = adjacent differences, slowest band passes through
        bandpass = torch.empty_like(lowpass)
        bandpass[:, :, :-1] = lowpass[:, :, :-1] - lowpass[:, :, 1:]
        bandpass[:, :, -1] = lowpass[:, :, -1]
        features = bandpass.reshape(N, -1).float()

        # advance traces to final state via closed-form (vectorised)
        pos = torch.arange(N, device=self.device, dtype=self.dtype)
        exponents = (N - 1) - pos
        weights = torch.exp(
            exponents.unsqueeze(1) * self.log_decay.unsqueeze(0))
        weighted_counts = one_hot.T @ weights
        decay_N = self.decay ** N
        self.traces = (self.traces * decay_N.unsqueeze(0) +
                       self.alphas.unsqueeze(0) * weighted_counts)

        return features

    # ── state ──

    def state_numpy(self):
        """return traces as float64 numpy regardless of internal dtype.
        keeps checkpoints portable across devices.
        """
        return self.traces.detach().cpu().double().numpy()

    def load_state(self, traces_np):
        """load from float64 numpy. casts to self.dtype on device."""
        t = torch.from_numpy(traces_np.astype(np.float64))
        self.traces = t.to(device=self.device, dtype=self.dtype).clone()


# ─────────────────────────────────────────────────────────────────────
# decimation
# ─────────────────────────────────────────────────────────────────────

def compute_band_confidence(n_bands, base, decimation_band):
    """per-band confidence weights. unchanged from soma.py."""
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

    same components, gradients, normalization, decimation as soma.py.
    only the trace bank is reimplemented to live on-device.
    """

    def __init__(self, n_bands=46, base=None, max_window=None,
                 hidden_dim=256, lr=0.1, max_change=0.1, weight_decay=1e-4,
                 batch_size=50000, decimation_band=0, device='auto',
                 direct_readout=False,
                 lr_auto=False, lr_base=1.0,
                 max_change_auto=False, max_change_base=1.0):
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.direct_readout = bool(direct_readout) if hidden_dim > 0 else False

        if max_window is not None:
            self.base = max_window ** (1.0 / (n_bands - 1))
        elif base is not None:
            self.base = base
        else:
            self.base = PHI

        # ── auto-lr machinery ──
        # When lr_auto / max_change_auto is True, the *effective* values
        # (self.lr, self.max_change) are recomputed each batch as
        #   lr        = lr_base        * (last_batch_loss / ln(VOCAB))
        #   max_change = max_change_base * (last_batch_loss / ln(VOCAB))
        # so they linearly interpolate between 0 (loss=0) and lr_base
        # (loss=ln(VOCAB) ≈ 5.545, the uniform-prediction baseline).
        # No smoothing — the per-batch loss IS the signal: hard batches
        # warrant a stronger update, easy batches a weaker one.
        # Initial value assumes loss = ln(VOCAB) so first batch uses lr_base.
        self.lr_auto = bool(lr_auto)
        self.lr_base = float(lr_base)
        self.max_change_auto = bool(max_change_auto)
        self.max_change_base = float(max_change_base)

        # If non-auto, lr/max_change are static. If auto, these get
        # overwritten each batch but we still need an initial value
        # so the first batch's update uses something reasonable.
        if self.lr_auto:
            self.lr = self.lr_base
        else:
            self.lr = lr
        if self.max_change_auto:
            self.max_change = self.max_change_base
        else:
            self.max_change = max_change

        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.decimation_band = decimation_band
        # Training-time checkpoint defaults are kept separate from runtime
        # chat overrides. Chat/online prompting can temporarily force
        # batch=1 and decimation=0 without rewriting the saved train config.
        self.checkpoint_batch_size = batch_size
        self.checkpoint_decimation_band = decimation_band
        # save defaults — persisted in checkpoint so resume can prefill them
        self.save_path = "model.pt"
        self.save_every = 0
        self.device = self._select_device(device)
        self._impl = self._select_impl()

        # trace bank — checks band count vs device fp64 support
        self.bank = TraceBank(n_bands, base=self.base, device=self.device)
        self.n_features = self.bank.n_features
        self.max_window = self.base ** (n_bands - 1)

        self._update_decimation()

        # per-band column indices for gradient scatter
        K = n_bands
        self._band_slices = []
        for k in range(K):
            cols = torch.arange(k, VOCAB * K, K, device=self.device)
            self._band_slices.append(cols)

        # weight matrices — identical to soma.py
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
        K = self.n_bands

        with torch.no_grad():
            if self.hidden_dim > 0:
                hidden_norm = cache['hidden_norm']
                hidden = cache['hidden']
                hidden_sum = cache['hidden_sum']
                X = cache['X']

                grad_W = (errors.T @ hidden_norm) / n
                W_pre = self.W.clone()
                self._apply_clipped_update(self.W, grad_W)
                self.W *= (1.0 - self.weight_decay)
                self._normalize_W()

                grad_hidden_norm = errors @ W_pre
                scale = self.hidden_budget / hidden_sum
                grad_hidden_relu = (
                    grad_hidden_norm * scale
                    - (grad_hidden_norm * hidden_norm).sum(
                        dim=1, keepdim=True)
                    * scale / self.hidden_budget
                )
                grad_hidden = grad_hidden_relu * (hidden > 0).float()

                # U / Wd gradients computed once as a (H, V*K) tensor, then
                # consumed band-by-band so per-update intermediates stay
                # bounded at one band's worth of columns. for K=52, H=20192
                # the full grad is ~1GB and a single fused clipped update
                # would instantiate 3 more 1GB intermediates (raw_delta,
                # max_delta, delta), which OOMs the unified memory pool on
                # M3-class macs. band-by-band keeps peak per-update memory
                # at ~21MB (H*V*4) instead of ~4GB.
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
                # band-by-band for the same memory reason as above
                for k in range(K):
                    cols = self._band_slices[k]
                    c = self._band_confidence[k]
                    self._apply_band_update(
                        self.W, cols, grad_W_full[:, cols] * c)
                self.W *= (1.0 - self.weight_decay)
                self._normalize_W()

    def _apply_clipped_update(self, param, grad):
        raw_delta = self.lr * grad
        max_delta = self.max_change * param.abs()
        delta = torch.clamp(raw_delta, -max_delta, max_delta)
        param -= delta

    def _apply_band_update(self, param, cols, grad_band):
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

        # Per-feature confidence vector: feature column c belongs to band
        # (c % K), so feature_confidence[c] = band_confidence[c % K].
        # Cached so the per-step weight update can apply it as one broadcast
        # multiply instead of K gather/scatter iterations.
        K = self.n_bands
        n_features = VOCAB * K
        feature_band = torch.arange(n_features, device=self.device) % K
        self._feature_confidence = self._band_confidence[feature_band]

    def _features_to_compute(self, Xt):
        """Move bank-produced features to the compute device in float32.

        When the trace bank lives on a different device than the compute
        device (which happens when K > FP32_MAX_BANDS forces fp64-on-cpu
        but the user wants compute on mps), we need to bridge across.
        When bank.device == self.device, this is essentially a no-op
        (same-device .to() returns self if already matching).
        """
        if Xt.device != self.device or Xt.dtype != torch.float32:
            return Xt.to(device=self.device, dtype=torch.float32)
        return Xt

    def _update_auto_lr(self, batch_loss, n_bytes):
        """Set lr/max_change from the most recent batch's loss (auto mode).

        batch_loss is the total cross-entropy in nats over the batch.
        n_bytes is the number of bytes in the batch. Per-batch loss is
        the signal — high-loss batches warrant a stronger update,
        low-loss batches a weaker one.

        When neither lr nor max_change is auto, this is a fast no-op.
        """
        if not (self.lr_auto or self.max_change_auto):
            return

        if hasattr(batch_loss, 'item'):
            batch_loss = batch_loss.item()
        loss_per_byte = batch_loss / max(1, n_bytes)

        # ratio in [0, 1] — clamped because rare bad batches can push
        # transient loss above the uniform baseline; we don't want lr
        # to exceed lr_base.
        ratio = max(0.0, min(1.0, loss_per_byte / float(np.log(VOCAB))))

        if self.lr_auto:
            self.lr = self.lr_base * ratio
        if self.max_change_auto:
            self.max_change = self.max_change_base * ratio

    @staticmethod
    def _select_device(device):
        if device != 'auto':
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _select_impl(self):
        """Pick the fastest available backend for the active device.

        Priority: native kernel for the device → portable PyTorch → None.
        Returns a module exposing the fast-path API:
            bandpass_W, bandpass_T_W, fast_forward, fast_backward.
        Returns None if no fast-path backend is available (caller falls
        back to the original slow path).
        """
        d = self.device
        # CUDA → Triton (preferred), else PyTorch fallback
        if d.type == 'cuda':
            if _TRITON_AVAILABLE and _impl_triton.is_available(d):
                return _impl_triton
            if _TORCH_IMPL_AVAILABLE:
                return _impl_torch
            return None
        # MPS → Metal (preferred), else PyTorch fallback
        if d.type == 'mps':
            if _METAL_AVAILABLE and _impl_metal.is_available(d):
                return _impl_metal
            if _TORCH_IMPL_AVAILABLE:
                return _impl_torch
            return None
        # CPU → PyTorch
        if _TORCH_IMPL_AVAILABLE:
            return _impl_torch
        return None

    # ── training ──

    def train(self, corpus_path, epochs=1,
              save_every=0, save_path="model.pt",
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
            # Accumulate on device — every += stays on-GPU. Only sync to CPU
            # when reporting. At batch=1000 this saves ~2 syncs/batch which
            # on mps is many ms each (queue drain + dispatch round-trip).
            total_loss = torch.zeros((), device=self.device)
            correct = torch.zeros((), dtype=torch.int64, device=self.device)
            samples = 0
            t0 = time.time()
            last_save = 0

            # Hot inner-loop dispatch decision: pick the fast linear path
            # (one kernel for fwd, one for bwd, no states tensor) when its
            # preconditions are met. Falls through to the standard path
            # otherwise.
            use_fast = self._can_use_fast_path() and stride == 1
            if use_fast:
                kind = "linear" if self.hidden_dim == 0 else f"hidden={self.hidden_dim}"
                backend = self._impl.__name__.replace('_impl_', '')
                print(f"    {GLYPH['bullet']} fast {kind} path enabled "
                      f"({backend} backend, K={self.n_bands})")

            if stride == 1:
                for batch_start in range(0, N, batch_size):
                    batch_end = min(batch_start + batch_size, N)
                    n = batch_end - batch_start
                    chunk = corpus[batch_start:batch_end]

                    if use_fast:
                        if self.hidden_dim == 0:
                            loss, acc = self._fast_train_batch_linear(chunk)
                        else:
                            loss, acc = self._fast_train_batch_hidden(chunk)
                    else:
                        Xt = self.bank.process_block(chunk)
                        Xt = self._features_to_compute(Xt)
                        yt = torch.from_numpy(
                            chunk.astype(np.int64)).to(self.device)
                        loss, acc = self._train_batch(Xt, yt, n)
                    total_loss += loss
                    correct += acc
                    samples += n
                    self.bytes_seen += n
                    # update auto-lr if enabled; cheap no-op otherwise.
                    # in auto mode this incurs one per-batch sync via
                    # .item() inside _update_auto_lr; an acceptable
                    # cost since the user opted in.
                    self._update_auto_lr(loss, n)

                    # report every batch — clean sync point, GPU queue
                    # already drained by the batch method
                    self._report(
                        epoch, epochs, batch_end, N,
                        total_loss.item(), correct.item(), samples, t0)
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
                    features_buf[n_collected] = self._features_to_compute(
                        self.bank.tap())
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
                        self._update_auto_lr(loss, n_collected)
                        n_collected = 0

                        # report every batch
                        self._report(
                            epoch, epochs, pos, N,
                            total_loss.item(), correct.item(), samples, t0)
                    if save_every and pos - last_save >= save_every:
                        if n_collected > 0:
                            loss, acc = self._train_batch(
                                features_buf[:n_collected],
                                targets_buf[:n_collected],
                                n_collected)
                            total_loss += loss
                            correct += acc
                            samples += n_collected
                            self._update_auto_lr(loss, n_collected)
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
                    self._update_auto_lr(loss, n_collected)
                    # report the final partial batch
                    self._report(
                        epoch, epochs, pos, N,
                        total_loss.item(), correct.item(), samples, t0)

            elapsed = time.time() - t0
            # final sync at end of epoch
            tl = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
            cc = correct.item() if isinstance(correct, torch.Tensor) else correct
            avg = tl / samples if samples > 0 else 0
            bpb = avg / np.log(2)
            acc = 100 * cc / samples if samples > 0 else 0
            print(f"    epoch {epoch + 1} done "
                  f"{GLYPH['bullet']} {avg:.4f} nats ({bpb:.2f} bpb) "
                  f"{GLYPH['bullet']} {acc:.1f}% "
                  f"{GLYPH['bullet']} {elapsed:.1f}s "
                  f"{GLYPH['bullet']} {N / elapsed:,.0f} b/s")

    def _can_use_fast_path(self):
        """The DSP-reformulated path applies when:
          - no direct readout (no Wd)
          - decimation_band == 0 (so all band confidences are 1.0)
          - K <= 32 (simd_sum reduction limit on Metal; matches Triton too)
          - using float32 traces
          - a fast-path backend is available

        Both linear (hidden_dim==0) and hidden cases are supported; the
        train loop branches on hidden_dim to pick the right method.
        """
        return (
            self.Wd is None
            and self.decimation_band == 0
            and self.n_bands <= 32
            and self.bank.dtype == torch.float32
            and self._impl is not None
        )

    def _ensure_fast_buffers(self, n):
        """Lazily allocate persistent buffers for the fast linear path.
        Reuses across batches as long as n stays constant. Call site below
        updates these in-place rather than reallocating.
        """
        K = self.n_bands
        V = VOCAB
        C = VOCAB
        device = self.device
        # cache key is (n, K) — if n changes (e.g. last small batch),
        # allocate fresh.
        cache = getattr(self, '_fast_cache', None)
        if cache is not None and cache['n'] == n and cache['K'] == K:
            return cache
        cache = {
            'n': n,
            'K': K,
            'Wp':       torch.empty(C, V, K, dtype=torch.float32, device=device),
            'u_init':   torch.empty(C, K, dtype=torch.float32, device=device),
            'idx':      torch.arange(n, device=device),
            'grad_W_3d':torch.empty(C, V, K, dtype=torch.float32, device=device),
            'delta':    torch.empty(C, V * K, dtype=torch.float32, device=device),
        }
        self._fast_cache = cache
        return cache

    def _fast_train_batch_linear(self, chunk_np):
        """Train one batch using the DSP reformulation.

        No process_block, no states tensor, no feature matmul. Forward and
        backward each run as a single Metal kernel dispatch. Trace bank is
        advanced with the closed-form formula afterwards (no snapshots).

        Persistent buffers: most intermediates are pre-allocated once and
        reused across batches. This avoids allocator churn that otherwise
        causes throughput to drift downward over long training runs.

        Args:
            chunk_np: (n,) np.uint8 byte array

        Returns:
            (loss, acc) as device tensors.
        """
        n = len(chunk_np)
        K = self.n_bands
        buf = self._ensure_fast_buffers(n)

        # bytes on device once
        bytes_t = torch.from_numpy(chunk_np).to(
            device=self.device, dtype=torch.uint8)
        target_idx = bytes_t.long()  # for indexing into logits/probs

        with torch.no_grad():
            # 1. Wp = bandpass(W) on the K axis — write into persistent buffer
            W_3d = self.W.view(VOCAB, VOCAB, K)
            Wp = buf['Wp']
            Wp[..., 0] = W_3d[..., 0]
            Wp[..., 1:] = W_3d[..., 1:] - W_3d[..., :-1]

            # 2. u_init = einsum('cvk,vk->ck', Wp, T)
            #    we still let einsum allocate its output — small (C,K)=32KB,
            #    not worth the gymnastics to in-place it
            u_init = torch.einsum('cvk,vk->ck', Wp, self.bank.traces)

            # 3. fast forward — one dispatch through the active backend
            logits, _u_final = self._impl.fast_forward(
                Wp, bytes_t, self.bank.alphas, self.bank.decay, u_init)

            # 4. error signal
            probs = F.softmax(logits, dim=1)
            idx = buf['idx']
            loss = -torch.log(probs[idx, target_idx] + EPS).sum()
            acc = (logits.argmax(1) == target_idx).sum()
            # mutate probs in-place into errors
            probs[idx, target_idx] -= 1.0
            errors = probs

            # 5. fast backward — accumulates into a fresh grad_Wp tensor
            grad_Wp = self._impl.fast_backward(
                errors, bytes_t, self.bank.alphas, self.bank.decay, VOCAB)
            grad_Wp = grad_Wp / n

            # 6. map gradient back to W-space via B^T — write into persistent buffer
            grad_W_3d = buf['grad_W_3d']
            grad_W_3d[..., :-1] = grad_Wp[..., :-1] - grad_Wp[..., 1:]
            grad_W_3d[..., -1] = grad_Wp[..., -1]

            # 7. clipped update on W (in place)
            grad_W_flat = grad_W_3d.view(VOCAB, -1)
            delta = buf['delta']
            torch.mul(grad_W_flat, self.lr, out=delta)
            # clamp delta to ± max_change * |W|, all in-place where possible
            max_delta = self.max_change * self.W.abs()
            delta.clamp_(min=-max_delta, max=max_delta)
            self.W -= delta

            # 8. weight decay (in place)
            self.W *= (1.0 - self.weight_decay)

            # 9. row normalize (in place)
            self._normalize_W()

            # 10. advance trace bank
            self.bank.advance(chunk_np)

        # explicit batch-boundary sync prevents queue runaway
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        return loss, acc

    def _fallback_train_linear(self, chunk_np):
        """Used when fast kernel is unavailable for some reason."""
        Xt = self.bank.process_block(chunk_np)
        Xt = self._features_to_compute(Xt)
        yt = torch.from_numpy(chunk_np.astype(np.int64)).to(self.device)
        return self._train_batch(Xt, yt, len(chunk_np))

    def _ensure_fast_buffers_hidden(self, n):
        """Persistent buffers for the hidden fast path.
        H = hidden_dim. Buffers sized to H for the kernel inputs/outputs;
        W gradients are small (V x H) so we let those allocate naturally.
        """
        K = self.n_bands
        V = VOCAB
        H = self.hidden_dim
        device = self.device
        cache = getattr(self, '_fast_cache_h', None)
        if cache is not None and cache['n'] == n and cache['K'] == K and cache['H'] == H:
            return cache
        cache = {
            'n': n,
            'K': K,
            'H': H,
            'Up':         torch.empty(H, V, K, dtype=torch.float32, device=device),
            'u_init':     torch.empty(H, K, dtype=torch.float32, device=device),
            'idx':        torch.arange(n, device=device),
            'grad_U_3d':  torch.empty(H, V, K, dtype=torch.float32, device=device),
            'delta_U':    torch.empty(H, V * K, dtype=torch.float32, device=device),
        }
        self._fast_cache_h = cache
        return cache

    def _fast_train_batch_hidden(self, chunk_np):
        """Train one batch with hidden layer using DSP kernels for the
        U matmul (the only V-dim contraction in the network).

        The W matmul is small (vocab × hidden) and stays as a plain
        PyTorch matmul. Same with ReLU, budget normalization, softmax.
        """
        n = len(chunk_np)
        K = self.n_bands
        V = VOCAB
        H = self.hidden_dim
        buf = self._ensure_fast_buffers_hidden(n)

        bytes_t = torch.from_numpy(chunk_np).to(
            device=self.device, dtype=torch.uint8)
        target_idx = bytes_t.long()

        with torch.no_grad():
            # ── 1. compute U' = bandpass(U) along K axis ──
            U_3d = self.U.view(H, V, K)
            Up = buf['Up']
            Up[..., 0] = U_3d[..., 0]
            Up[..., 1:] = U_3d[..., 1:] - U_3d[..., :-1]

            # ── 2. u_init = einsum('hvk,vk->hk', Up, traces) ──
            u_init = torch.einsum('hvk,vk->hk', Up, self.bank.traces)

            # ── 3. fast forward — one dispatch through the active backend
            # output: hidden_pre of shape (n, H) — same kernel as linear
            # but C = H instead of VOCAB
            hidden_pre, _u_final = self._impl.fast_forward(
                Up, bytes_t, self.bank.alphas, self.bank.decay, u_init)

            # ── 4. ReLU + budget normalization ──
            hidden_relu = F.relu(hidden_pre)
            hidden_sum = hidden_relu.sum(dim=1, keepdim=True) + EPS
            hidden_norm = hidden_relu * (self.hidden_budget / hidden_sum)

            # ── 5. logits = hidden_norm @ W.T (small matmul) ──
            logits = hidden_norm @ self.W.T

            # ── 6. softmax + error signal ──
            probs = F.softmax(logits, dim=1)
            idx = buf['idx']
            loss = -torch.log(probs[idx, target_idx] + EPS).sum()
            acc = (logits.argmax(1) == target_idx).sum()
            probs[idx, target_idx] -= 1.0
            errors = probs   # (n, V) — gradient of loss wrt logits

            # ── 7. backward through W ──
            # grad_W = errors^T @ hidden_norm / n  (V, H)
            grad_W = (errors.T @ hidden_norm) / n
            # need W BEFORE update for gradient backprop
            W_pre = self.W.clone()
            # apply clipped update on W (in place using existing helper)
            self._apply_clipped_update(self.W, grad_W)
            self.W *= (1.0 - self.weight_decay)
            self._normalize_W()

            # grad_hidden_norm = errors @ W_pre  (n, H)
            grad_hidden_norm = errors @ W_pre

            # ── 8. backward through budget normalization ──
            # hidden_norm = hidden_relu * scale, where scale = budget / sum
            # grad_hidden_relu = grad_hidden_norm * scale
            #                   - <grad_hidden_norm, hidden_norm> * scale / budget
            scale = self.hidden_budget / hidden_sum
            grad_hidden_relu = (
                grad_hidden_norm * scale
                - (grad_hidden_norm * hidden_norm).sum(dim=1, keepdim=True)
                * scale / self.hidden_budget
            )

            # ── 9. backward through ReLU ──
            # using hidden_pre > 0 as the mask (ReLU derivative)
            grad_hidden_pre = grad_hidden_relu * (hidden_pre > 0).float()
            # grad_hidden_pre is now (n, H) — this is the "errors" feeding
            # into the U^T @ X gradient

            # ── 10. fast backward kernel: compute grad_U' ──
            # the impl reads (errors, bytes, alphas, decay) and writes
            # gradients accumulated into a (C, V, K) buffer. Here C = H.
            grad_Up = self._impl.fast_backward(
                grad_hidden_pre, bytes_t,
                self.bank.alphas, self.bank.decay, V)
            grad_Up = grad_Up / n

            # ── 11. map grad_U' back to grad_U via B^T (transpose-bandpass) ──
            grad_U_3d = buf['grad_U_3d']
            grad_U_3d[..., :-1] = grad_Up[..., :-1] - grad_Up[..., 1:]
            grad_U_3d[..., -1] = grad_Up[..., -1]

            # ── 12. clipped update on U ──
            grad_U_flat = grad_U_3d.view(H, -1)
            delta = buf['delta_U']
            torch.mul(grad_U_flat, self.lr, out=delta)
            max_delta = self.max_change * self.U.abs()
            delta.clamp_(min=-max_delta, max=max_delta)
            self.U -= delta
            self.U *= (1.0 - self.weight_decay)
            self._normalize_U()

            # ── 13. advance trace bank ──
            self.bank.advance(chunk_np)

        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        return loss, acc

    def _fallback_train_hidden(self, chunk_np):
        """Used when fast kernel is unavailable for hidden case."""
        Xt = self.bank.process_block(chunk_np)
        Xt = self._features_to_compute(Xt)
        yt = torch.from_numpy(chunk_np.astype(np.int64)).to(self.device)
        return self._train_batch(Xt, yt, len(chunk_np))

    def _train_batch(self, Xt, yt, n):
        """Forward, compute error signal, update weights.

        Returns loss and acc as MPS tensors (not Python scalars). The caller
        accumulates them on-device. We only .item() at report boundaries to
        avoid forcing a CPU↔GPU sync per batch — those syncs are the bulk
        of the overhead at small batch sizes on MPS.
        """
        with torch.no_grad():
            logits, cache = self._forward_batch(Xt)
            probs = F.softmax(logits, dim=1)
            idx = torch.arange(n, device=self.device)
            # keep these as device tensors — no .item() in hot path
            loss = -torch.log(probs[idx, yt] + EPS).sum()
            acc = (logits.argmax(1) == yt).sum()
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
        # show lr — window-average when auto (more representative of
        # the report period than instantaneous), static value otherwise.
        if self.lr_auto:
            avg_ratio = max(0.0, min(1.0, avg / np.log(VOCAB)))
            avg_lr = self.lr_base * avg_ratio
            lr_str = f" {GLYPH['bullet']} lr={avg_lr:.3f}"
        elif self.max_change_auto:
            avg_ratio = max(0.0, min(1.0, avg / np.log(VOCAB)))
            avg_mc = self.max_change_base * avg_ratio
            lr_str = f" {GLYPH['bullet']} mc={avg_mc:.3f}"
        else:
            lr_str = ""
        print(f"    [{epoch + 1}/{epochs}] "
              f"{_bar(frac)} {frac * 100:4.1f}% "
              f"{GLYPH['bullet']} {avg:.3f} nats ({bpb:.2f} bpb) "
              f"{acc:.1f}% "
              f"{GLYPH['bullet']} {bps:,.0f} b/s{lr_str}")

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
            Xt = self._features_to_compute(Xt)
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
        for _ in range(length):
            features = self._features_to_compute(self.bank.tap())
            logits = self._forward(features)

            logits = logits / temperature
            probs = F.softmax(logits, dim=0)
            byte_val = torch.multinomial(probs, 1).item()

            if byte_val == ord('\n'):
                break
            ch = chr(byte_val) if 32 <= byte_val < 127 else '.'
            yield ch

            self.bank.tick(byte_val)

    # ── online learning ──

    def _learn_single(self, features, logits, target_byte):
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
        prompt_bytes = np.array([ord(c) for c in text], dtype=np.uint8)

        if online:
            n = len(prompt_bytes)
            if n == 0:
                return

            features_list = []
            for b in prompt_bytes:
                features_list.append(self._features_to_compute(self.bank.tap()))
                self.bank.tick(int(b))

            Xt = torch.stack(features_list)
            yt = torch.from_numpy(prompt_bytes.astype(np.int64)).to(self.device)

            loss, _acc = self._train_batch(Xt, yt, n)
            self.bytes_seen += n
            # auto-lr responds to the user's input as one batch.
            # cheap no-op when neither lr nor max_change is auto.
            self._update_auto_lr(loss, n)
        else:
            self.bank.advance(prompt_bytes)

    # ── save / load ──

    def _checkpoint_id(self):
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
            'lr_auto': self.lr_auto,
            'lr_base': self.lr_base,
            'max_change_auto': self.max_change_auto,
            'max_change_base': self.max_change_base,
            'save_path': self.save_path,
            'save_every': self.save_every,
            'weight_decay': self.weight_decay,
            'w_norm': self.w_norm,
            'bytes_seen': self.bytes_seen,
            'batch_size': getattr(self, 'checkpoint_batch_size',
                                  self.batch_size),
            'decimation_band': getattr(self, 'checkpoint_decimation_band',
                                       self.decimation_band),
            'direct_readout': bool(self.direct_readout),
            'soma_version': 'v10',
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
        # auto-lr fields are optional — older checkpoints don't have them
        self.lr_auto = ckpt.get('lr_auto', self.lr_auto)
        self.lr_base = ckpt.get('lr_base', self.lr_base)
        self.max_change_auto = ckpt.get('max_change_auto', self.max_change_auto)
        self.max_change_base = ckpt.get('max_change_base', self.max_change_base)
        self.save_path = ckpt.get('save_path', self.save_path)
        self.save_every = ckpt.get('save_every', self.save_every)
        self.weight_decay = ckpt.get('weight_decay',
                                     ckpt.get('shrinkage', self.weight_decay))
        self.w_norm = ckpt.get('w_norm', self.w_norm)
        self.bytes_seen = ckpt.get('bytes_seen', 0)
        self.batch_size = ckpt.get('batch_size', self.batch_size)
        self.checkpoint_batch_size = self.batch_size
        self.checkpoint_history = ckpt.get('checkpoint_history', [])

        if 'decimation_band' in ckpt:
            self.decimation_band = ckpt['decimation_band']
            self.checkpoint_decimation_band = self.decimation_band
        elif 'downsample' in ckpt:
            ds = ckpt['downsample']
            if ds <= 1:
                self.decimation_band = 0
                self.checkpoint_decimation_band = self.decimation_band
            else:
                self.decimation_band = max(0, int(round(
                    np.log(ds) / np.log(self.base))))
                self.checkpoint_decimation_band = self.decimation_band

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

        dtype_str = "fp64" if self.bank.dtype == torch.float64 else "fp32"
        if self.bank.device != self.device:
            device_str = (f"{self.device} "
                          f"{GLYPH['bullet']} traces={dtype_str}@{self.bank.device}")
        else:
            device_str = f"{self.device} {GLYPH['bullet']} traces={dtype_str}"
        print(f"\n  {GLYPH['dot']} soma v10 {GLYPH['bullet']} {device_str} "
              f"{GLYPH['bullet']} {_fmt_bytes(self.bytes_seen)} seen")
        print(f"    {band_str}")
        print(f"    base={self.base:.4f} "
              f"{GLYPH['bullet']} range={self.max_window:,.0f} "
              f"{GLYPH['bullet']} {hidden_str} "
              f"{GLYPH['bullet']} {_fmt_params(params)} params")
        lr_str = (f"auto({self.lr_base})→{self.lr:.4f}"
                  if self.lr_auto else f"{self.lr}")
        mc_str = (f"auto({self.max_change_base})→{self.max_change:.4f}"
                  if self.max_change_auto else f"{self.max_change}")
        print(f"    lr={lr_str} "
              f"{GLYPH['bullet']} max_change={mc_str} "
              f"{GLYPH['bullet']} weight_decay={self.weight_decay}")
        print()


# ─────────────────────────────────────────────────────────────────────
# cli
# ─────────────────────────────────────────────────────────────────────

def main():
    _banner()

    mode = _prompt("mode (train/eval/chat): ")

    if mode == "train":
        corpus = _resolve_path(_prompt("corpus: "), 'corpus')
        if not Path(corpus).exists():
            return print(f"  not found: {corpus}")

        ckpt = _resolve_path(
            _prompt("checkpoint (enter for new): "), 'checkpoint')
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
            # Render saved values nicely if they were auto. We display
            # the auto+base format so the user can edit easily.
            saved_lr_disp = (f"auto {cfg['lr_base']}"
                             if cfg.get('lr_auto') else str(saved_lr))
            saved_mc_disp = (f"auto {cfg['max_change_base']}"
                             if cfg.get('max_change_auto') else str(saved_mc))
            lr_str = _prompt(f"lr [{saved_lr_disp}]: ", saved_lr_disp)
            mc_str = _prompt(
                f"max_change [{saved_mc_disp}]: ", saved_mc_disp)
            lr_val, lr_auto, lr_base = _parse_auto_or_float(lr_str)
            mc_val, mc_auto, mc_base = _parse_auto_or_float(mc_str)
            wd = float(_prompt(
                f"weight_decay [{saved_wd}]: ", str(saved_wd)))
            bs = int(_prompt(f"batch [{saved_bs}]: ", str(saved_bs)))
            db = int(_prompt(
                f"decimation_band [{saved_db}]: ", str(saved_db)))

            model = SOMA(
                cfg.get('n_bands', cfg.get('num_timescales', 46)),
                base=cfg.get('base', PHI),
                hidden_dim=cfg.get('hidden_dim', 256),
                lr=lr_val, max_change=mc_val, weight_decay=wd,
                batch_size=bs, decimation_band=db,
                direct_readout=bool(cfg.get('direct_readout', False)),
                lr_auto=lr_auto, lr_base=lr_base,
                max_change_auto=mc_auto, max_change_base=mc_base)
            model.load(ckpt)
            # The user's CLI choices override anything that was in the
            # checkpoint (load may have overwritten with old values).
            model.lr_auto = lr_auto
            model.lr_base = lr_base
            model.max_change_auto = mc_auto
            model.max_change_base = mc_base
            if not lr_auto:
                model.lr = lr_val
            if not mc_auto:
                model.max_change = mc_val
            model.weight_decay = wd
            model.batch_size = bs
            model.decimation_band = db
            model.checkpoint_batch_size = bs
            model.checkpoint_decimation_band = db
            model._update_decimation()
        else:
            bands = int(_prompt("bands [32]: ", "32"))
            range_str = _prompt(
                "range (base or window) [1.6180]: ", "1.6180")
            val = float(range_str)
            if val < 100:
                base, max_window = val, None
            else:
                base, max_window = None, val
            hd = int(_prompt("hidden (0=linear) [256]: ", "256"))
            lr_str = _prompt("lr [0.1]: ", "0.1")
            mc_str = _prompt("max_change [0.1]: ", "0.1")
            lr_val, lr_auto, lr_base = _parse_auto_or_float(lr_str)
            mc_val, mc_auto, mc_base = _parse_auto_or_float(mc_str)
            wd = float(_prompt("weight_decay [0.0001]: ", "0.0001"))
            bs = int(_prompt("batch [1000]: ", "1000"))
            ds = int(_prompt("decimation_band [0]: ", "0"))
            dr = int(_prompt("direct readout (0/1) [0]: ", "0"))

            model = SOMA(bands, base=base, max_window=max_window,
                         hidden_dim=hd, lr=lr_val, max_change=mc_val,
                         weight_decay=wd, batch_size=bs,
                         decimation_band=ds, direct_readout=bool(dr),
                         lr_auto=lr_auto, lr_base=lr_base,
                         max_change_auto=mc_auto, max_change_base=mc_base)

        model.print_config()

        epochs = int(_prompt("epochs [1]: ", "1"))
        start_byte = int(_prompt("start byte [0]: ", "0"))
        save_every = int(_prompt(
            f"save every (0=end) [{model.save_every}]: ",
            str(model.save_every)))
        save_path = _resolve_path(
            _prompt(f"save path [{model.save_path}]: ", model.save_path),
            'checkpoint')
        # persist these on the model so the next run gets them as defaults
        model.save_every = save_every
        model.save_path = save_path

        model.train(corpus, epochs=epochs, start_byte=start_byte,
                    save_every=save_every,
                    save_path=save_path)
        model.save(save_path)

    elif mode == "eval":
        ckpt = _resolve_path(_prompt("checkpoint: "), 'checkpoint')
        corpus = _resolve_path(_prompt("corpus: "), 'corpus')
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
        ckpt = _resolve_path(_prompt("checkpoint: "), 'checkpoint')
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
            # show auto state in default if the loaded model is auto
            lr_disp = (f"auto {model.lr_base}"
                       if model.lr_auto else str(model.lr))
            mc_disp = (f"auto {model.max_change_base}"
                       if model.max_change_auto else str(model.max_change))
            lr_str = _prompt(f"lr [{lr_disp}]: ", lr_disp)
            mc_str = _prompt(f"max_change [{mc_disp}]: ", mc_disp)
            lr_val, lr_auto, lr_base = _parse_auto_or_float(lr_str)
            mc_val, mc_auto, mc_base = _parse_auto_or_float(mc_str)
            model.lr_auto = lr_auto
            model.lr_base = lr_base
            model.max_change_auto = mc_auto
            model.max_change_base = mc_base
            model.lr = lr_val if not lr_auto else lr_base
            model.max_change = mc_val if not mc_auto else mc_base
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
                save_path = _resolve_path(
                    _prompt("save path: ", ckpt), 'checkpoint')
                model.save(save_path)
                continue
            if user:
                model.ingest_prompt(user + ' ', online=online)
                print(f"  {GLYPH['gen']} {GLYPH['arrow']} ", end='', flush=True)
                for ch in model.generate(length=maxlen, temperature=temp):
                    print(ch, end='', flush=True)
                print('\n')

        if _prompt(
                "save state? (y/n) [y]: ", "y").lower() in ('y', 'yes'):
            save_path = _resolve_path(
                _prompt("save path: ", ckpt), 'checkpoint')
            model.save(save_path)


if __name__ == '__main__':
    main()
