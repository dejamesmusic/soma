"""
░░▒▒▓▓                         soma                         ▓▓▒▒░░
░░▒▒▓▓              v12.2 · spectral trace learner           ▓▓▒▒░░

v12.2 is the current shippable v12 line.

fresh models default to the io2 controller path:

    bands 50 · base 1.6180 · hidden 1024 · layers 3
    auto mode io2 · lr auto · max_change auto
    batch 256 · decimation range 1.0
    dreams every 50 batches · dream length auto 200 · temp 1.0

retired knobs remain load-compatible for older checkpoints, but the
fresh cli/gui path no longer asks for direct readout, scale gate,
clock, or weight decay.

spectral mode keeps the ordinary byte-prediction objective and adds a
residual trace bank. the residual bank decomposes prediction error by
timescale, then uses that spectrum to choose band-specific plasticity
and the current decimation target. this allocates compute toward the
scale where coherent error currently lives while preserving the normal
hidden-layer learning rule.

bare corpus names resolve to data/. bare checkpoint names resolve to
checkpoints/. absolute paths and paths containing "/" are used as-is.
"""

import os
import sys
import gc
import time
import math
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# numpy pickle compat (checkpoints saved under numpy 2.x load on 1.x)
# ─────────────────────────────────────────────────────────────────────

def _install_numpy_pickle_compat():
    try:
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric
    except ImportError:
        return
    sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy_numeric)


_install_numpy_pickle_compat()


PHI = (1 + np.sqrt(5)) / 2
EPS = 1e-10
VOCAB = 256
AUTO_DREAM_HARD_CAP = 10_000


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
    print(GLYPH['sep'] * width)


def _banner():
    print()
    _sep()
    print(GLYPH['logo'] + " v12")
    _sep()


def _bar(frac, width=30):
    fill = int(frac * width)
    mid = 1 if 0 < frac < 1 and fill < width else 0
    return (GLYPH['bar_fill'] * fill + GLYPH['bar_mid'] * mid
            + GLYPH['bar_empty'] * (width - fill - mid))


def _fmt_bytes(n):
    for unit in ("", "K", "M", "G"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n:.0f}"
        n /= 1000
    return f"{n:.1f}T"


def _fmt_params(n):
    return _fmt_bytes(n)


def _prompt(text, default=""):
    try:
        v = input(f"  {GLYPH['arrow']} {text}").strip()
    except EOFError:
        v = ""
    return v if v else default


def _parse_auto_or_float(s, default_base=1.0):
    """parse a literal float or an auto mode string.
    returns (value, is_auto, base)."""
    s = str(s).strip().lower()
    if (s.startswith('auto') or s.startswith('progress')
            or s.startswith('spectral')
            or s.startswith('full spectrum')
            or s.startswith('io2')):
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


def _parse_auto_or_int(s, default_value=0):
    s = str(s).strip().lower()
    if s == "auto":
        return default_value, True
    try:
        return int(float(s)), False
    except ValueError:
        return default_value, False


def _parse_decimation_range(s, default_value=0.25):
    s = str(s).strip().lower()
    if s == "auto":
        return default_value
    try:
        return float(s)
    except ValueError:
        return default_value


# ─────────────────────────────────────────────────────────────────────
# path conventions — bare filenames route to data/ and checkpoints/
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# demisa — the canonical config. one keystroke at the cli; the gui
# can `from soma_v12 import DEMISA, TRAIN_DEFAULTS`.
# ~468M params · ~1.9GB checkpoint · sized for apple silicon.
# ─────────────────────────────────────────────────────────────────────

DEMISA = dict(
    n_bands=50, base=1.6180,
    hidden_dim=1024, n_layers=3,
    scale_gate=False,
    clock=1,
    direct_readout=False,
    auto_mode='io2',
    lr=1.0, lr_auto=True, lr_base=1.0,
    max_change=1.0, max_change_auto=True, max_change_base=1.0,
    weight_decay=0.0,
    batch_size=256,
    decimation_auto=True, decimation_range=1.0,
)

TRAIN_DEFAULTS = dict(
    epochs=1,
    dream_every_batches=50, dream_length='auto 200', dream_temperature=1.0,
    save_every=1_000_000,
)

DEFAULT_CKPT = "demisa.pt"

_RUNTIME_DIR = os.environ.get("SOMA_HOME", "")
DATA_DIR = os.path.join(_RUNTIME_DIR, "data") if _RUNTIME_DIR else "data"
CHECKPOINT_DIR = (os.path.join(_RUNTIME_DIR, "checkpoints")
                  if _RUNTIME_DIR else "checkpoints")


def _resolve_path(path, kind):
    if not path:
        return path
    if ('/' in path or '\\' in path or
            path.startswith('~') or path.startswith('.')):
        return os.path.expanduser(path)
    folder = DATA_DIR if kind == 'corpus' else CHECKPOINT_DIR
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, path)


def _parse_cycle_spec(text):
    parts = str(text).strip().split()
    if not parts or parts[0].lower() != 'cycle':
        return None
    if len(parts) == 1:
        return 1
    try:
        return max(1, int(float(parts[1])))
    except ValueError:
        return 1


def _data_cycle_files(data_dir=DATA_DIR):
    root = Path(data_dir)
    if not root.exists():
        return []
    skip_dirs = {'streams'}
    files = []
    for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if path.name.startswith('.'):
            continue
        if path.is_dir():
            if path.name in skip_dirs:
                continue
            continue
        if path.is_file():
            files.append(str(path))
    return files


# ─────────────────────────────────────────────────────────────────────
# trace bank — float64 on cpu, exact closed-form block scan
# ─────────────────────────────────────────────────────────────────────

class TraceBank:
    """multi-timescale exponential traces over C channels.

    channels are bytes (C=256, one-hot input) at layer 0, or hidden
    patterns (C=H, nonnegative activation vector summing to ~1) at
    deeper layers. the dynamics are identical:

        E[:, k] ← (1 - α_k) · E[:, k]
        E[:, k] ← E[:, k] + α_k · x_t          (x_t one-hot or dense)

    process_block computes the per-position pre-update states in
    closed form: within a chunk of length L the recurrence
    S_t = d·S_{t-1} + α·x_t unrolls to a lower-triangular kernel
    matmul per band; across chunks the carry uses the same
    closed-form advance the reference implementation uses. float64
    throughout — numerically identical to the sequential loop.
    """

    CHUNK = 128

    def __init__(self, n_channels, n_bands, base, device):
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.base = base
        self.device = device
        self.n_features = n_channels * n_bands
        self.trace_device = (
            device if (os.environ.get('SOMA_CUDA_TRACE') == '1'
                       and getattr(device, 'type', None) == 'cuda')
            else torch.device('cpu'))
        self.trace_dtype = torch.float64

        alphas_np = np.array(
            [1.0 / (base ** k) for k in range(n_bands)], dtype=np.float64)
        self.alphas = torch.from_numpy(alphas_np).to(self.trace_device)
        self.decay = torch.from_numpy(1.0 - alphas_np).to(self.trace_device)
        self.log_decay = torch.log(torch.clamp(self.decay, min=1e-300))

        self.traces = torch.zeros(
            n_channels, n_bands, dtype=self.trace_dtype,
            device=self.trace_device)

        # closed-form chunk kernel: ker[k, t, j] = α_k · d_k^(t-1-j), j < t
        L = self.CHUNK
        t_idx = torch.arange(
            L, dtype=self.trace_dtype, device=self.trace_device).view(L, 1)
        j_idx = torch.arange(
            L, dtype=self.trace_dtype, device=self.trace_device).view(1, L)
        expo = t_idx - 1.0 - j_idx
        mask = (j_idx < t_idx)
        logd = self.log_decay.view(n_bands, 1, 1)
        raw = torch.where(mask.unsqueeze(0), expo.unsqueeze(0) * logd,
                          torch.full((1, 1, 1), float('-inf'),
                                     dtype=self.trace_dtype,
                                     device=self.trace_device))
        self._ker = torch.exp(raw) * self.alphas.view(n_bands, 1, 1)
        self._dpow = torch.exp(
            torch.arange(
                L, dtype=self.trace_dtype,
                device=self.trace_device).view(1, L)
            * self.log_decay.view(n_bands, 1))            # (K, L): d^t

    def reset(self):
        self.traces.zero_()

    def _to_trace_tensor(self, x):
        x = x.detach()
        if self.trace_device.type == 'cpu':
            return x.to('cpu').to(self.trace_dtype)
        return x.to(device=self.trace_device, dtype=self.trace_dtype)

    # ── single-step ──

    def tick(self, x):
        """absorb one step. x: int byte (layer 0) or 1-d tensor (deep)."""
        self.traces *= self.decay
        if isinstance(x, (int, np.integer)):
            self.traces[x] += self.alphas
        else:
            xv = self._to_trace_tensor(x).reshape(-1)
            self.traces += self.alphas.unsqueeze(0) * xv.unsqueeze(1)

    def tap(self):
        """bandpass features of current state → (C·K,) float32 device."""
        bp = torch.empty_like(self.traces)
        bp[:, :-1] = self.traces[:, :-1] - self.traces[:, 1:]
        bp[:, -1] = self.traces[:, -1]
        return bp.reshape(-1).float().to(self.device)

    # ── block ──

    def _to_dense(self, x):
        """(N,) uint8 bytes → (N, C) one-hot f64, or pass dense through."""
        if isinstance(x, np.ndarray) and x.dtype == np.uint8:
            N = len(x)
            idx = torch.from_numpy(x.astype(np.int64)).to(self.trace_device)
            dense = torch.zeros(
                N, self.n_channels, dtype=self.trace_dtype,
                device=self.trace_device)
            dense.scatter_(1, idx.unsqueeze(1), 1.0)
            return dense
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(
                device=self.trace_device, dtype=self.trace_dtype)
        return self._to_trace_tensor(x)

    def advance(self, x):
        """advance state through a block without producing features."""
        dense = self._to_dense(x)
        N = dense.shape[0]
        if N == 0:
            return
        pos = torch.arange(N, dtype=self.trace_dtype, device=self.trace_device)
        w = torch.exp(((N - 1) - pos).unsqueeze(1)
                      * self.log_decay.unsqueeze(0))      # (N, K)
        counts = dense.T @ w                              # (C, K)
        self.traces = (self.traces * (self.decay ** N).unsqueeze(0)
                       + self.alphas.unsqueeze(0) * counts)

    def process_block(self, x):
        """features for every position (state BEFORE absorbing that
        position), then advance. returns (N, C·K) float32 on device."""
        dense = self._to_dense(x)                         # (N, C)
        N = dense.shape[0]
        K = self.n_bands
        C = self.n_channels
        L = self.CHUNK

        feats = torch.empty(
            N, C * K, dtype=torch.float32, device=self.trace_device)
        S = self.traces.clone()                           # (C, K)

        for c0 in range(0, N, L):
            c1 = min(c0 + L, N)
            n = c1 - c0
            xc = dense[c0:c1]                             # (n, C)
            conv = torch.matmul(self._ker[:, :n, :n],
                                xc.unsqueeze(0))          # (K, n, C)
            carry = (self._dpow[:, :n].unsqueeze(2)
                     * S.T.unsqueeze(1))                  # (K, n, C)
            states = conv + carry
            bp = torch.empty_like(states)
            bp[:-1] = states[:-1] - states[1:]
            bp[-1] = states[-1]
            feats[c0:c1] = bp.permute(1, 2, 0).reshape(n, -1).float()
            pos = torch.arange(
                n, dtype=self.trace_dtype, device=self.trace_device)
            w = torch.exp(((n - 1) - pos).unsqueeze(1)
                          * self.log_decay.unsqueeze(0))
            S = (S * (self.decay ** n).unsqueeze(0)
                 + self.alphas.unsqueeze(0) * (xc.T @ w))

        self.traces = S
        return feats.to(self.device)

    def process_block_select(self, x, rows):
        """features for selected pre-update positions, then advance.

        rows are offsets into x. this is equivalent to process_block(x)[rows]
        but avoids materializing/transferring unsampled feature rows in the
        auto-decimated training path.
        """
        dense = self._to_dense(x)                         # (N, C)
        rows = np.asarray(rows, dtype=np.int64)
        N = dense.shape[0]
        K = self.n_bands
        C = self.n_channels
        L = self.CHUNK

        feats = torch.empty(
            len(rows), C * K, dtype=torch.float32, device=self.trace_device)
        S = self.traces.clone()                           # (C, K)
        out0 = 0

        for c0 in range(0, N, L):
            c1 = min(c0 + L, N)
            n = c1 - c0
            xc = dense[c0:c1]                             # (n, C)
            conv = torch.matmul(self._ker[:, :n, :n],
                                xc.unsqueeze(0))          # (K, n, C)
            carry = (self._dpow[:, :n].unsqueeze(2)
                     * S.T.unsqueeze(1))                  # (K, n, C)
            states = conv + carry

            lo = np.searchsorted(rows, c0, side='left')
            hi = np.searchsorted(rows, c1, side='left')
            if hi > lo:
                local = torch.from_numpy(
                    rows[lo:hi] - c0).long().to(self.trace_device)
                selected = states.index_select(1, local)
                bp = torch.empty_like(selected)
                bp[:-1] = selected[:-1] - selected[1:]
                bp[-1] = selected[-1]
                take = hi - lo
                feats[out0:out0 + take] = (
                    bp.permute(1, 2, 0).reshape(take, -1).float())
                out0 += take

            pos = torch.arange(
                n, dtype=self.trace_dtype, device=self.trace_device)
            w = torch.exp(((n - 1) - pos).unsqueeze(1)
                          * self.log_decay.unsqueeze(0))
            S = (S * (self.decay ** n).unsqueeze(0)
                 + self.alphas.unsqueeze(0) * (xc.T @ w))

        self.traces = S
        return feats.to(self.device)

    # ── state ──

    def state_numpy(self):
        return self.traces.detach().cpu().numpy()

    def load_state(self, traces_np):
        self.traces = torch.from_numpy(
            np.asarray(traces_np, dtype=np.float64)).to(
                device=self.trace_device, dtype=self.trace_dtype).clone()


# ─────────────────────────────────────────────────────────────────────
# decimation
# ─────────────────────────────────────────────────────────────────────

def compute_band_confidence(n_bands, base, decimation_band):
    decimation_band = max(0.0, min(float(decimation_band), n_bands - 1))
    stride = max(1, int(round(base ** decimation_band)))
    confidence = np.array(
        [min(1.0, base ** (k - decimation_band)) for k in range(n_bands)],
        dtype=np.float64)
    return stride, confidence


# ─────────────────────────────────────────────────────────────────────
# soma — deep spectral online machine
# ─────────────────────────────────────────────────────────────────────

class SOMA:
    """a stack of soma blocks with local credit and one shared target.

    layer 0:   bank over bytes   → features X0 → U0 → relu/budget → h0
    layer ℓ:   bank over h_{ℓ-1} → features Xℓ → Uℓ → relu/budget → hℓ
    heads:     logits = Σ_ℓ Wℓ hℓ  (+ Wd X0 if direct_readout)

    each Uℓ learns only through its own head Wℓ — gradients never
    cross a trace bank, so no bptt and no shifting targets. deep
    banks trace hℓ/budget (a distribution over patterns), so deep
    trace magnitudes match byte-trace magnitudes by construction.
    """

    def __init__(self, n_bands=46, base=None, max_window=None,
                 hidden_dim=256, lr=0.1, max_change=0.1, weight_decay=1e-4,
                 batch_size=50000, decimation_band=0, device='auto',
                 direct_readout=False, n_layers=1,
                 scale_gate=False, clock=1, auto_mode='level',
                 lr_auto=False, lr_base=1.0,
                 max_change_auto=False, max_change_base=1.0,
                 decimation_auto=False, decimation_range=0.25):
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.n_layers = max(1, int(n_layers)) if hidden_dim > 0 else 1
        if hidden_dim == 0 and n_layers > 1:
            print("    (hidden=0 is linear — layers forced to 1)")
        self.direct_readout = bool(direct_readout) if hidden_dim > 0 else False

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

        # auto machinery (v10 semantics)
        self.lr_auto = bool(lr_auto)
        self.lr_base = float(lr_base)
        self.max_change_auto = bool(max_change_auto)
        self.max_change_base = float(max_change_base)
        self.decimation_auto = bool(decimation_auto)
        self.decimation_range = float(decimation_range)
        if self.lr_auto:
            self.lr = self.lr_base
        if self.max_change_auto:
            self.max_change = self.max_change_base
        # effort controller:
        # 'level'    tracks how far loss sits above zero.
        # 'progress' tracks short-term loss movement.
        # 'spectral' uses the normal level controller globally, then
        # redistributes band plasticity by the trace-bank decomposition
        # of the residual stream.
        # 'full spectrum' removes global loss scaling and lets the
        # residual spectrum control band plasticity + decimation.
        # 'io2' separates input/output coherence: learn densely when the
        # input is structured and the model is underfit/confident-wrong;
        # traverse more quickly when input looks incoherent or loss is below
        # the homeostatic target.
        self.auto_mode = (auto_mode if auto_mode in ('level', 'progress',
                                                     'spectral',
                                                     'full spectrum',
                                                     'io2')
                          else 'level')
        self._ema_fast = None
        self._ema_slow = None
        self.save_path = "model.pt"
        self.save_every = 0

        # banks: layer 0 over bytes; deeper over hidden patterns
        L = self.n_layers
        H = hidden_dim
        self.banks = [TraceBank(VOCAB, n_bands, self.base, self.device)]
        for _ in range(1, L):
            self.banks.append(
                TraceBank(H, n_bands, self.base, self.device))
        self.error_bank = TraceBank(VOCAB, n_bands, self.base, self.device)
        self._spectral_drive = torch.ones(n_bands, device=self.device)
        self._spectral_target_band = 0.0
        self._input_coherence = 0.0
        self._output_coherence = 0.0
        self._input_trust = 0.0
        self._error_concentration = 0.0
        self._io2_plasticity = 0.0
        self.bank = self.banks[0]            # v8/v10 compatibility alias
        self.n_features = self.banks[0].n_features
        self.max_window = self.base ** (n_bands - 1)

        self._update_decimation()

        # per-band column index slices per layer (feature col = c·K + k)
        K = n_bands
        self._band_slices = []
        for ell in range(L):
            C = self.banks[ell].n_channels
            self._band_slices.append(
                [torch.arange(k, C * K, K, device=self.device)
                 for k in range(K)])

        # ── v12: scale gate ──
        # per-layer (K, K) matrix + (K,) bias over band-energy stats.
        # zero init = uniform attention = v11 behaviour exactly.
        self.scale_gate = bool(scale_gate) if hidden_dim > 0 else False
        if self.scale_gate:
            self.G_list = [torch.zeros(K, 2 * K, device=self.device)
                           for _ in range(L)]
            self.gb_list = [torch.zeros(K, device=self.device)
                            for _ in range(L)]
        else:
            self.G_list, self.gb_list = [], []

        # ── v12: multirate clock ──
        # layer ℓ ticks once per clock^ℓ stream items; between ticks
        # its state (and therefore its features) hold. phase + carry
        # buffers give exact continuity across batch boundaries and
        # between batch and chat modes.
        self.clock = max(1, int(clock)) if self.n_layers > 1 else 1
        self._phase = [0 for _ in range(L)]          # items in carry
        self._carry = [None for _ in range(L)]       # (H,) running sum

        # weights
        if H > 0:
            self.hidden_budget = H * 0.1
            self.U_list, self.W_list = [], []
            self.u_norm_list = []
            self.w_norm = np.sqrt(H) * 0.1
            for ell in range(L):
                nf = self.banks[ell].n_features
                u_norm = np.sqrt(nf) * 0.1
                U = torch.randn(H, nf, device=self.device)
                W = torch.randn(VOCAB, H, device=self.device)
                self.U_list.append(U)
                self.W_list.append(W)
                self.u_norm_list.append(u_norm)
            self._normalize_all()
            if self.direct_readout:
                self.wd_norm = np.sqrt(self.n_features) * 0.1
                self.Wd = torch.randn(
                    VOCAB, self.n_features, device=self.device)
                self._normalize_Wd()
            else:
                self.wd_norm = None
                self.Wd = None
        else:
            self.U_list, self.u_norm_list = [], []
            self.hidden_budget = None
            self.Wd, self.wd_norm = None, None
            self.w_norm = np.sqrt(self.n_features) * 0.1
            self.W_list = [torch.randn(
                VOCAB, self.n_features, device=self.device)]
            self._normalize_all()

        self.bytes_seen = 0
        self.checkpoint_history = []

    # v8/v10 compatibility properties (single-layer views)
    @property
    def U(self):
        return self.U_list[0] if self.U_list else None

    @property
    def W(self):
        return self.W_list[0]

    @property
    def u_norm(self):
        return self.u_norm_list[0] if self.u_norm_list else None

    # ── normalization ──

    def _normalize_all(self):
        with torch.no_grad():
            for ell, W in enumerate(self.W_list):
                norms = W.norm(dim=1, keepdim=True)
                W.mul_(self.w_norm / (norms + EPS))
            for ell, U in enumerate(self.U_list):
                norms = U.norm(dim=1, keepdim=True)
                U.mul_(self.u_norm_list[ell] / (norms + EPS))

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
        self._ones_confidence = torch.ones_like(self._band_confidence)

    def _features_to_compute(self, Xt):
        if Xt.device != self.device or Xt.dtype != torch.float32:
            return Xt.to(device=self.device, dtype=torch.float32)
        return Xt

    def _update_auto_lr(self, batch_loss, n_bytes):
        if not (self.lr_auto or self.max_change_auto or self.decimation_auto):
            return
        if hasattr(batch_loss, 'item'):
            batch_loss = batch_loss.item()
        loss_per_byte = batch_loss / max(1, n_bytes)
        if self.auto_mode == 'io2':
            ratio = self._io2_ratio(loss_per_byte)
        elif self.auto_mode == 'progress':
            # effort follows the DERIVATIVE of loss, not its level.
            # |fast − slow| is large while learning or during a regime
            # shift, and decays to zero on anything flat — whether
            # mastered text or unlearnable noise. REF sets how big a
            # relative movement counts as full effort.
            REF = 0.03
            if self._ema_fast is None:
                self._ema_fast = loss_per_byte
                self._ema_slow = loss_per_byte
                ratio = 1.0
            else:
                self._ema_fast += 0.30 * (loss_per_byte - self._ema_fast)
                self._ema_slow += 0.08 * (loss_per_byte - self._ema_slow)
                drive = (abs(self._ema_fast - self._ema_slow)
                         / (REF * max(self._ema_slow, 1e-6)))
                ratio = max(0.02, min(1.0, drive))   # plasticity floor
        elif self.auto_mode == 'full spectrum':
            ratio = 1.0
        else:
            ratio = max(0.0, min(1.0,
                                 loss_per_byte / float(np.log(VOCAB))))
        if self.lr_auto:
            self.lr = self.lr_base * ratio
        if self.max_change_auto:
            self.max_change = self.max_change_base * ratio
        if self.decimation_auto:
            max_auto_band = int(round(
                self.decimation_range * (self.n_bands - 1)))
            max_auto_band = max(0, min(max_auto_band, self.n_bands - 1))
            if self.auto_mode == 'io2':
                target = self._io2_decimation_target(
                    loss_per_byte, ratio, max_auto_band)
            elif self.auto_mode in ('spectral', 'full spectrum'):
                target = min(float(max_auto_band),
                             float(self._spectral_target_band))
            else:
                target = (1.0 - ratio) * max_auto_band
            target = max(0.0, min(target, float(max_auto_band)))
            if target > self.decimation_band:
                target = min(target, self.decimation_band + 1.0)
            elif self.auto_mode == 'io2':
                target = max(target, self.decimation_band - 2.0)
            if target != self.decimation_band:
                self.decimation_band = target
                self._update_decimation()

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
        loss_above = max(0.0, (loss_per_byte / 1.0) - 1.0)
        underfit = max(0.0, input_trust - output_coh)
        confident_wrong = output_coh * loss_above
        learnability = (
            loss_above
            * input_trust
            * (0.5 + 2.0 * error_coh)
            * (0.5 + underfit + confident_wrong)
            * progress_gate
        )
        plasticity = learnability / (1.0 + learnability)
        self._input_trust = float(input_trust)
        self._io2_plasticity = float(plasticity)
        return max(0.02, min(1.0, plasticity))

    def _io2_decimation_target(self, loss_per_byte, plasticity, max_auto_band):
        loss_below = max(0.0, 1.0 - (loss_per_byte / 1.0))
        input_skip = 1.0 - max(0.0, min(1.0, self._input_trust))
        spectral_center = max(0.0, min(
            float(self._spectral_target_band), float(max_auto_band)))
        relax = max(min(1.0, loss_below), 0.5 * input_skip)
        throughput_target = (
            (1.0 - relax) * spectral_center + relax * float(max_auto_band))
        return (1.0 - plasticity) * throughput_target

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

    @staticmethod
    def _bank_bandpass(bank):
        traces = bank.traces
        bp = torch.empty_like(traces)
        bp[:, :-1] = traces[:, :-1] - traces[:, 1:]
        bp[:, -1] = traces[:, -1]
        return bp

    def _update_spectral_drive(self, errors):
        """trace the residual stream and allocate band plasticity by
        where coherent error currently lives in soma's own spectrum."""
        if self.auto_mode not in ('spectral', 'full spectrum', 'io2'):
            self._spectral_drive = torch.ones(self.n_bands, device=self.device)
            return
        self.error_bank.advance(errors)
        bp = self._bank_bandpass(self.error_bank).view(VOCAB, self.n_bands)
        energy = torch.linalg.vector_norm(bp, dim=0)
        total = energy.sum()
        if not torch.isfinite(total) or total <= EPS:
            self._spectral_drive = torch.ones(
                self.n_bands, device=self.device)
            return
        drive = self.n_bands * energy / (total + EPS)
        self._spectral_drive = torch.clamp(drive, 0.1, 3.0).to(
            device=self.device, dtype=torch.float32)
        bands = torch.arange(self.n_bands, device=energy.device,
                             dtype=energy.dtype)
        center = (bands * energy).sum() / (total + EPS)
        if torch.isfinite(center):
            self._spectral_target_band = float(center.detach().cpu().item())
        if self.auto_mode == 'io2':
            p = energy / (total + EPS)
            entropy = -(p * torch.log(p + EPS)).sum()
            max_entropy = math.log(max(2, self.n_bands))
            concentration = 1.0 - torch.clamp(
                entropy / max_entropy, 0.0, 1.0)
            self._error_concentration = float(
                concentration.detach().cpu().item())
            input_bp = self._bank_bandpass(self.banks[0]).view(
                VOCAB, self.n_bands)
            channel_coh = self._coherence_from_distribution(input_bp)
            band_energy = torch.linalg.vector_norm(input_bp, dim=0)
            band_coh = self._coherence_from_distribution(band_energy)
            self._input_coherence = float((channel_coh * band_coh) ** 0.5)

    @staticmethod
    def _select_device(device):
        if device != 'auto':
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    # ── forward (batched) ──

    # ── v12 helpers ──

    def _gate(self, ell, X_rows):
        """per-row softmax attention over the K bands of this layer.
        returns (gated features, gate, stats). when the gate is off,
        returns the input untouched (exact v11 path)."""
        if not self.scale_gate:
            return X_rows, None, None
        R = X_rows.shape[0]
        K = self.n_bands
        C = self.banks[ell].n_channels
        Xr = X_rows.view(R, C, K)
        s1 = torch.log1p(Xr.abs().sum(dim=1))           # (R, K) energy
        s2 = torch.log1p(Xr.abs().amax(dim=1))          # (R, K) peak
        s = torch.cat([s1 - s1.mean(dim=1, keepdim=True),
                       s2 - s2.mean(dim=1, keepdim=True)], dim=1)
        g = F.softmax(s @ self.G_list[ell].T + self.gb_list[ell], dim=1)
        Xg = (Xr * g.unsqueeze(1)).reshape(R, -1)
        return Xg, g, s

    def _pool_stream(self, ell, stream):
        """absorb a stream of layer-(ell-1) tick values into layer
        ell's clock. returns (pooled (M, H), idx_shift) and updates
        the layer's phase/carry. idx for a stream item q is
        (phase_before + q) // clock — computed by the caller."""
        c = self.clock
        S = stream.shape[0]
        if c <= 1:
            self._phase[ell] = 0
            self._carry[ell] = None
            return stream.detach().cpu().to(torch.float64), S

        phase = self._phase[ell]
        carry = self._carry[ell]
        if carry is None:
            carry = torch.zeros(stream.shape[1], dtype=torch.float64)
            self._carry[ell] = carry
        stream64 = stream.detach().to('cpu').to(torch.float64)
        M = (phase + S) // c
        pooled = torch.empty(M, stream.shape[1], dtype=torch.float64)
        pos = 0
        for j in range(M):
            take = c - phase
            window = stream64[pos:pos + take]
            pooled[j] = (carry + window.sum(dim=0)) / c
            carry.zero_()
            phase = 0
            pos += take
        if pos < S:
            carry += stream64[pos:].sum(dim=0)
            phase += S - pos
        self._phase[ell] = phase
        return pooled, M

    def _stack_forward(self, X0, advance_deep=True):
        """run the layer stack on a block of layer-0 features.

        X0: (N, 256·K) float32 on device, rows are pre-update states.
        deep layers run at ROW level (one row per tick of their own
        clock, plus one post-state row); positions map onto rows by
        hold (idx). gradients pool exactly over hold windows.

        returns (logits, caches)."""
        N = X0.shape[0]
        caches = []
        logits = torch.zeros(N, VOCAB, device=self.device)

        if self.hidden_dim == 0:
            logits = X0 @ self.W_list[0].T
            caches.append({'X': X0})
            return logits, caches

        X_rows = X0
        idx = None                       # None = identity (layer 0)
        stream_count = N                 # ticks of the layer below
        for ell in range(self.n_layers):
            Xg, g, s = self._gate(ell, X_rows)
            U = self.U_list[ell]
            hidden = Xg @ U.T
            hidden_relu = F.relu(hidden)
            hidden_sum = hidden_relu.sum(dim=1, keepdim=True) + EPS
            hidden_norm = hidden_relu * (self.hidden_budget / hidden_sum)
            h_pos = hidden_norm if idx is None else hidden_norm[idx]
            logits = logits + h_pos @ self.W_list[ell].T
            caches.append({
                'hidden': hidden, 'hidden_norm': hidden_norm,
                'hidden_sum': hidden_sum, 'X': X_rows, 'Xg': Xg,
                'g': g, 's': s, 'idx': idx, 'n_rows': X_rows.shape[0],
            })
            if ell + 1 < self.n_layers:
                if not advance_deep:
                    saved = (self.banks[ell + 1].traces.clone(),
                             self._phase[ell + 1],
                             None if self._carry[ell + 1] is None
                             else self._carry[ell + 1].clone())
                # stream items = h at each tick of THIS layer
                stream = (hidden_norm[:stream_count]
                          / self.hidden_budget)
                phase_before = self._phase[ell + 1]
                pooled, M = self._pool_stream(ell + 1, stream)
                # row index per item q of the stream below
                q = (torch.arange(stream_count)
                     if idx is None else idx.cpu())
                next_idx = ((phase_before + q) // self.clock).to(
                    self.device)
                if M > 0:
                    Xr = self.banks[ell + 1].process_block(pooled)
                    Xr = self._features_to_compute(Xr)
                    post = self.banks[ell + 1].tap().unsqueeze(0)
                    X_rows = torch.cat([Xr, post], dim=0)
                else:
                    X_rows = self.banks[ell + 1].tap().unsqueeze(0)
                X_rows = self._features_to_compute(X_rows)
                idx = next_idx.clamp(max=X_rows.shape[0] - 1)
                stream_count = M
                if not advance_deep:
                    self.banks[ell + 1].traces = saved[0]
                    self._phase[ell + 1] = saved[1]
                    self._carry[ell + 1] = saved[2]

        if self.Wd is not None:
            logits = logits + X0 @ self.Wd.T
        return logits, caches

    # ── updates ──

    def _apply_band_update(self, param, cols, grad_band):
        band_vals = param[:, cols]
        raw_delta = self.lr * grad_band
        max_delta = self.max_change * band_vals.abs()
        delta = torch.clamp(raw_delta, -max_delta, max_delta)
        param[:, cols] = band_vals - delta

    def _apply_clipped_update(self, param, grad):
        raw_delta = self.lr * grad
        max_delta = self.max_change * param.abs()
        delta = torch.clamp(raw_delta, -max_delta, max_delta)
        param -= delta

    def _band_column_scale(self, ell, conf):
        C = self.banks[ell].n_channels
        return conf.repeat(C)

    def _apply_feature_update(self, param, grad, ell, conf):
        scale = self._band_column_scale(ell, conf).view(1, -1)
        raw_delta = self.lr * grad * scale
        max_delta = self.max_change * param.abs()
        delta = torch.clamp(raw_delta, -max_delta, max_delta)
        param -= delta

    def _update_weights(self, errors, caches, n):
        """local credit: each layer's U updates through its own head
        only. position-level gradients pool exactly onto the layer's
        own rows (hold windows). gate gradients flow through the
        softmax; the gate uses an additive clamp (zero-init params
        cannot move under a multiplicative clip)."""
        K = self.n_bands
        with torch.no_grad():
            if self.hidden_dim == 0:
                X = caches[0]['X']
                W = self.W_list[0]
                grad_W_full = (errors.T @ X) / n
                conf = self._band_confidence * self._spectral_drive
                self._apply_feature_update(W, grad_W_full, 0, conf)
                W *= (1.0 - self.weight_decay)
                self._normalize_all()
                return

            for ell in range(self.n_layers):
                cache = caches[ell]
                W = self.W_list[ell]
                U = self.U_list[ell]
                hidden_norm = cache['hidden_norm']
                hidden = cache['hidden']
                hidden_sum = cache['hidden_sum']
                Xg = cache['Xg']
                idx = cache['idx']
                R = cache['n_rows']

                h_pos = hidden_norm if idx is None else hidden_norm[idx]
                grad_W = (errors.T @ h_pos) / n
                grad_h_pos = errors @ W            # before W mutates
                self._apply_clipped_update(W, grad_W)
                W *= (1.0 - self.weight_decay)

                # pool position-level gradient onto this layer's rows
                if idx is None:
                    grad_hidden_norm = grad_h_pos
                else:
                    grad_hidden_norm = torch.zeros(
                        R, self.hidden_dim, device=self.device)
                    grad_hidden_norm.index_add_(0, idx, grad_h_pos)

                scale = self.hidden_budget / hidden_sum
                grad_hidden_relu = (
                    grad_hidden_norm * scale
                    - (grad_hidden_norm * hidden_norm).sum(
                        dim=1, keepdim=True) * scale / self.hidden_budget)
                grad_hidden = grad_hidden_relu * (hidden > 0).float()

                conf = (self._band_confidence if ell == 0
                        else self._ones_confidence)
                conf = conf * self._spectral_drive
                grad_U = (grad_hidden.T @ Xg) / n
                self._apply_feature_update(U, grad_U, ell, conf)
                U *= (1.0 - self.weight_decay)

                # gate gradient: through x̃ = x ⊙ g, then softmax jac
                if self.scale_gate:
                    C = self.banks[ell].n_channels
                    Xr = cache['X'].view(R, C, K)
                    grad_xg = (grad_hidden @ U).view(R, C, K)
                    grad_g = (grad_xg * Xr).sum(dim=1)            # (R, K)
                    g = cache['g']
                    grad_pre = g * (grad_g
                                    - (grad_g * g).sum(dim=1, keepdim=True))
                    grad_G = (grad_pre.T @ cache['s']) / n
                    grad_gb = grad_pre.sum(dim=0) / n
                    self.G_list[ell] -= torch.clamp(
                        self.lr * grad_G, -0.05, 0.05)
                    self.gb_list[ell] -= torch.clamp(
                        self.lr * grad_gb, -0.05, 0.05)

                if ell == 0 and self.Wd is not None:
                    X0 = cache['X']
                    grad_Wd = (errors.T @ X0) / n
                    conf0 = self._band_confidence * self._spectral_drive
                    self._apply_feature_update(self.Wd, grad_Wd, 0, conf0)
                    self.Wd *= (1.0 - self.weight_decay)
                    self._normalize_Wd()

            self._normalize_all()

    def _train_batch(self, X0, yt, n):
        with torch.no_grad():
            logits, caches = self._stack_forward(X0)
            probs = F.softmax(logits, dim=1)
            idx = torch.arange(n, device=self.device)
            loss = -torch.log(probs[idx, yt] + EPS).sum()
            acc = (logits.argmax(1) == yt).sum()
            if self.auto_mode == 'io2':
                entropy = -(probs * torch.log(probs + EPS)).sum(dim=1)
                out_coh = 1.0 - (entropy.mean() / math.log(VOCAB))
                self._output_coherence = float(torch.clamp(
                    out_coh, 0.0, 1.0).detach().cpu().item())
            probs[idx, yt] -= 1.0
            self._update_spectral_drive(probs)
        self._update_weights(probs, caches, n)
        return loss, acc

    def _collect_strided_batch(self, corpus, pos, total, max_rows):
        """collect one auto-decimated batch with a single layer-0 scan.

        The previous path tapped and advanced the byte trace bank once per
        sampled position. On GPU runs that creates hundreds of small
        CPU-to-device transfers per batch. This keeps the same sampling
        semantics while vectorizing layer-0 feature collection:

            observe pos, pos+stride, ...
            advance through every byte covered by those observations

        Auto controllers still update only after the returned batch trains,
        so stride is intentionally fixed for this collection window.
        """
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

    # ── training ──

    def train(self, corpus_path, epochs=1,
              save_every=0, save_path="model.pt",
              start_byte=0, dream_every_batches=0,
              dream_length=200, dream_temperature=0.8,
              dream_callback=None):
        corpus = np.fromfile(corpus_path, dtype=np.uint8)
        if start_byte > 0:
            corpus = corpus[start_byte:]
        N = len(corpus)
        batch_size = self.batch_size
        dream_every_batches = max(0, int(dream_every_batches))
        dream_auto, dream_fixed, dream_cap = self._parse_dream_length(
            dream_length)
        dream_temperature = float(dream_temperature)

        start_str = f", start={_fmt_bytes(start_byte)}" if start_byte else ""
        print(f"\n  {GLYPH['train']} training {corpus_path} "
              f"({_fmt_bytes(N)} bytes{start_str})")
        print(f"    batch={batch_size:,} "
              f"{GLYPH['bullet']} decimation_band={self.decimation_band} "
              f"(stride={self._stride}) "
              f"{GLYPH['bullet']} layers={self.n_layers} "
              f"{GLYPH['bullet']} {epochs} epoch{'s' if epochs != 1 else ''}")
        print()

        def maybe_dream(batch_loss, batch_samples):
            if dream_every_batches <= 0:
                return
            self._dream_batch_counter = getattr(
                self, '_dream_batch_counter', 0) + 1
            if self._dream_batch_counter % dream_every_batches != 0:
                return
            length = dream_fixed
            if dream_auto:
                length = self._auto_dream_length(
                    batch_loss, batch_samples, dream_cap)
            if length <= 0:
                return
            text = ''.join(self.generate(
                length=length, temperature=dream_temperature))
            if dream_callback is not None:
                dream_callback(text, self._dream_batch_counter,
                               self.bytes_seen)
            else:
                print(f"\n    dream {self._dream_batch_counter} "
                      f"{GLYPH['bullet']} {_fmt_bytes(self.bytes_seen)} seen")
                print(f"    {text}\n")

        for epoch in range(epochs):
            total_loss = torch.zeros((), device=self.device)
            correct = torch.zeros((), dtype=torch.int64, device=self.device)
            samples = 0
            t0 = time.time()
            last_save = 0

            if self._stride == 1 and not self.decimation_auto:
                for b0 in range(0, N, batch_size):
                    b1 = min(b0 + batch_size, N)
                    n = b1 - b0
                    chunk = corpus[b0:b1]
                    X0 = self._features_to_compute(
                        self.banks[0].process_block(chunk))
                    yt = torch.from_numpy(
                        chunk.astype(np.int64)).to(self.device)
                    loss, acc = self._train_batch(X0, yt, n)
                    total_loss += loss
                    correct += acc
                    samples += n
                    self.bytes_seen += n
                    self._update_auto_lr(loss, n)
                    maybe_dream(loss, n)
                    self._report(epoch, epochs, b1, N,
                                 total_loss.item(), correct.item(),
                                 samples, t0)
                    if save_every and b1 - last_save >= save_every:
                        self.save(save_path)
                        last_save = b1
            else:
                # strided / auto-decimated observation. layer-0 features
                # are collected with one block scan per training batch;
                # the deep stack still runs at flush, so deep banks tick
                # once per observation.
                if os.environ.get('SOMA_LEGACY_STRIDED') == '1':
                    feat_rows, targets = [], []
                    pos = 0
                    while pos < N:
                        feat_rows.append(self._features_to_compute(
                            self.banks[0].tap()))
                        targets.append(int(corpus[pos]))
                        advance_end = min(pos + self._stride, N)
                        self.banks[0].advance(corpus[pos:advance_end])
                        self.bytes_seen += advance_end - pos
                        pos = advance_end

                        flush = (len(feat_rows) >= batch_size or pos >= N)
                        if flush and feat_rows:
                            X0 = torch.stack(feat_rows)
                            yt = torch.tensor(
                                targets, dtype=torch.int64,
                                device=self.device)
                            n = len(targets)
                            loss, acc = self._train_batch(X0, yt, n)
                            total_loss += loss
                            correct += acc
                            samples += n
                            self._update_auto_lr(loss, n)
                            maybe_dream(loss, n)
                            feat_rows, targets = [], []
                            self._report(epoch, epochs, pos, N,
                                         total_loss.item(), correct.item(),
                                         samples, t0)
                        if save_every and pos - last_save >= save_every:
                            self.save(save_path)
                            last_save = pos
                    continue

                pos = 0
                while pos < N:
                    X0, yt, pos = self._collect_strided_batch(
                        corpus, pos, N, batch_size)
                    if X0 is None:
                        break
                    n = int(yt.shape[0])
                    loss, acc = self._train_batch(X0, yt, n)
                    total_loss += loss
                    correct += acc
                    samples += n
                    self._update_auto_lr(loss, n)
                    maybe_dream(loss, n)
                    self._report(epoch, epochs, pos, N,
                                 total_loss.item(), correct.item(),
                                 samples, t0)
                    if save_every and pos - last_save >= save_every:
                        self.save(save_path)
                        last_save = pos

            elapsed = time.time() - t0
            avg = total_loss.item() / samples if samples > 0 else 0
            bpb = avg / np.log(2)
            acc_pct = 100 * correct.item() / samples if samples > 0 else 0
            print(f"    epoch {epoch + 1} done "
                  f"{GLYPH['bullet']} {avg:.4f} nats ({bpb:.2f} bpb) "
                  f"{GLYPH['bullet']} {acc_pct:.1f}% "
                  f"{GLYPH['bullet']} {elapsed:.1f}s "
                  f"{GLYPH['bullet']} {N / max(elapsed, 1e-9):,.0f} b/s")

    def _parse_dream_length(self, dream_length):
        s = str(dream_length).strip().lower()
        if s.startswith('auto'):
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
        if hasattr(batch_loss, 'item'):
            batch_loss = batch_loss.item()
        ratio = max(0.0, min(1.0, (batch_loss / max(1, n))
                             / float(np.log(VOCAB))))
        return int(round((1.0 - ratio) * cap))

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
              f"{GLYPH['bullet']} {bps:,.0f} b/s", flush=True)

    # ── evaluation ──

    def evaluate(self, corpus_path):
        corpus = np.fromfile(corpus_path, dtype=np.uint8)
        N = len(corpus)
        batch_size = self.batch_size
        print(f"\n  {GLYPH['eval']} evaluating {corpus_path} "
              f"({_fmt_bytes(N)} bytes)")

        for bank in self.banks:
            bank.reset()
        total_loss = 0.0
        total_correct = 0
        t0 = time.time()

        for b0 in range(0, N, batch_size):
            b1 = min(b0 + batch_size, N)
            n = b1 - b0
            chunk = corpus[b0:b1]
            X0 = self._features_to_compute(
                self.banks[0].process_block(chunk))
            yt = torch.from_numpy(chunk.astype(np.int64)).to(self.device)
            with torch.no_grad():
                logits, _ = self._stack_forward(X0)
                probs = F.softmax(logits, dim=1)
                idx = torch.arange(n, device=self.device)
                total_loss -= torch.log(
                    probs[idx, yt] + EPS).sum().item()
                total_correct += (logits.argmax(1) == yt).sum().item()

        elapsed = time.time() - t0
        avg = total_loss / max(N, 1)
        bpb = avg / np.log(2)
        acc = 100 * total_correct / max(N, 1)
        print(f"    {avg:.4f} nats ({bpb:.2f} bpb) "
              f"{GLYPH['bullet']} {acc:.1f}% "
              f"{GLYPH['bullet']} {elapsed:.1f}s "
              f"{GLYPH['bullet']} {N / max(elapsed, 1e-9):,.0f} b/s")
        return avg

    # ── single-step path (chat / generation) ──

    def _single_forward(self):
        """forward from current bank states. deep layers' taps hold
        between their ticks automatically. returns (logits, hs, xs)."""
        if self.hidden_dim == 0:
            x0 = self.banks[0].tap()
            return self.W_list[0] @ x0, [], [x0]
        hs, xs = [], []
        logits = torch.zeros(VOCAB, device=self.device)
        for ell in range(self.n_layers):
            x = self.banks[ell].tap()
            xs.append(x)
            xg, _g, _s = self._gate(ell, x.unsqueeze(0))
            hidden = F.relu(self.U_list[ell] @ xg.squeeze(0))
            h_sum = hidden.sum() + EPS
            h_norm = hidden * (self.hidden_budget / h_sum)
            hs.append(h_norm)
            logits = logits + self.W_list[ell] @ h_norm
        if self.Wd is not None:
            logits = logits + self.Wd @ xs[0]
        return logits, hs, xs

    def _tick_all(self, byte_val, hs):
        """absorb one step. layer 0 takes the byte; each deeper layer
        accumulates the layer below's pattern distribution and ticks
        once per `clock` accumulations — the cascade only propagates
        upward when a layer actually ticks."""
        self.banks[0].tick(int(byte_val))
        if self.hidden_dim == 0:
            return
        item = (hs[0] / self.hidden_budget).detach().to(
            'cpu').to(torch.float64)
        for ell in range(1, self.n_layers):
            if self._carry[ell] is None:
                self._carry[ell] = torch.zeros(
                    self.hidden_dim, dtype=torch.float64)
            self._carry[ell] += item
            self._phase[ell] += 1
            if self._phase[ell] < self.clock:
                break
            pooled = self._carry[ell] / self.clock
            self.banks[ell].tick(pooled)
            self._carry[ell].zero_()
            self._phase[ell] = 0
            item = (hs[ell] / self.hidden_budget).detach().to(
                'cpu').to(torch.float64)

    def generate(self, length=200, temperature=0.8):
        for _ in range(length):
            logits, hs, _xs = self._single_forward()
            logits = logits / temperature
            probs = F.softmax(logits, dim=0)
            byte_val = torch.multinomial(probs, 1).item()
            if byte_val == ord('\n'):
                break
            ch = chr(byte_val) if 32 <= byte_val < 127 else '.'
            yield ch
            self._tick_all(byte_val, hs)

    def ingest_prompt(self, text, online=False):
        prompt_bytes = np.array([ord(c) % VOCAB for c in text],
                                dtype=np.uint8)
        total_n = len(prompt_bytes)
        if total_n == 0:
            return
        if online:
            max_batch = max(1, int(self.batch_size))
            for start in range(0, total_n, max_batch):
                chunk = prompt_bytes[start:start + max_batch]
                n = len(chunk)
                X0 = self._features_to_compute(
                    self.banks[0].process_block(chunk))
                yt = torch.from_numpy(
                    chunk.astype(np.int64)).to(self.device)
                loss, _acc = self._train_batch(X0, yt, n)
                self.bytes_seen += n
                self._update_auto_lr(loss, n)
        else:
            if self.n_layers == 1:
                self.banks[0].advance(prompt_bytes)
            else:
                # deep banks need the hidden stream for context
                X0 = self._features_to_compute(
                    self.banks[0].process_block(prompt_bytes))
                with torch.no_grad():
                    self._stack_forward(X0)

    # ── save / load ──

    def _hash_tensor_full(self, h, tensor, chunk_elems=1_000_000):
        """full-content hash, streamed in bounded chunks: ids stay
        unfakeable without doubling peak memory on large checkpoints."""
        if tensor is None:
            h.update(b"none")
            return
        t = tensor.detach().reshape(-1)
        h.update(str(tuple(tensor.shape)).encode())
        for i in range(0, t.numel(), chunk_elems):
            h.update(t[i:i + chunk_elems].cpu().numpy().tobytes())

    def _checkpoint_id(self):
        h = hashlib.sha256()
        for W in self.W_list:
            self._hash_tensor_full(h, W)
        for U in self.U_list:
            self._hash_tensor_full(h, U)
        self._hash_tensor_full(h, self.Wd)
        for bank in self.banks:
            self._hash_tensor_full(h, bank.traces)
        self._hash_tensor_full(h, self.error_bank.traces)
        for G in self.G_list:
            self._hash_tensor_full(h, G)
        for val in [self.n_bands, self.hidden_dim, self.base,
                    bool(self.direct_readout), self.n_layers,
                    bool(self.scale_gate), self.clock,
                    self.bytes_seen, self.decimation_band]:
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
        current_id = self._checkpoint_id()
        history = self.checkpoint_history + [current_id]
        data = {
            'W': self.W_list[0].cpu(),
            'traces': self.banks[0].state_numpy(),
            'error_traces': self.error_bank.state_numpy(),
            'spectral_drive': self._spectral_drive.detach().cpu(),
            'spectral_target_band': self._spectral_target_band,
            'input_coherence': self._input_coherence,
            'output_coherence': self._output_coherence,
            'input_trust': self._input_trust,
            'error_concentration': self._error_concentration,
            'io2_plasticity': self._io2_plasticity,
            'n_bands': self.n_bands,
            'base': self.base,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'lr': self.lr,
            'max_change': self.max_change,
            'lr_auto': self.lr_auto,
            'lr_base': self.lr_base,
            'max_change_auto': self.max_change_auto,
            'max_change_base': self.max_change_base,
            'decimation_auto': self.decimation_auto,
            'decimation_range': self.decimation_range,
            'auto_mode': self.auto_mode,
            'ema_fast': self._ema_fast,
            'ema_slow': self._ema_slow,
            'save_path': self.save_path,
            'save_every': self.save_every,
            'weight_decay': self.weight_decay,
            'w_norm': self.w_norm,
            'bytes_seen': self.bytes_seen,
            'batch_size': getattr(self, 'checkpoint_batch_size',
                                  self.batch_size),
            'decimation_band': (
                self.decimation_band if self.decimation_auto
                else getattr(self, 'checkpoint_decimation_band',
                             self.decimation_band)),
            'direct_readout': bool(self.direct_readout),
            'scale_gate': bool(self.scale_gate),
            'clock': self.clock,
            'soma_version': 'v12',
            'checkpoint_id': current_id,
            'checkpoint_history': history,
        }
        if self.U_list:
            data['U'] = self.U_list[0].cpu()
            data['u_norm'] = self.u_norm_list[0]
            data['hidden_budget'] = self.hidden_budget
        for ell in range(1, self.n_layers):
            data[f'U_{ell}'] = self.U_list[ell].cpu()
            data[f'W_{ell}'] = self.W_list[ell].cpu()
            data[f'u_norm_{ell}'] = self.u_norm_list[ell]
            data[f'traces_{ell}'] = self.banks[ell].state_numpy()
            data[f'phase_{ell}'] = self._phase[ell]
            if self._carry[ell] is not None:
                data[f'carry_{ell}'] = self._carry[ell].numpy()
        if self.scale_gate:
            for ell in range(self.n_layers):
                data[f'G_{ell}'] = self.G_list[ell].cpu()
                data[f'gb_{ell}'] = self.gb_list[ell].cpu()
        if self.Wd is not None:
            data['Wd'] = self.Wd.cpu()
            data['wd_norm'] = self.wd_norm

        torch.save(data, path)
        del data
        gc.collect()
        print(f"    {GLYPH['save']} saved {path} "
              f"{GLYPH['bullet']} {current_id[:12]}")

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)

        self.W_list[0] = ckpt['W'].float().to(self.device)
        self.banks[0].load_state(ckpt['traces'])
        if 'error_traces' in ckpt:
            self.error_bank.load_state(ckpt['error_traces'])
        if 'spectral_drive' in ckpt:
            self._spectral_drive = ckpt['spectral_drive'].float().to(
                self.device)
        self._spectral_target_band = ckpt.get(
            'spectral_target_band', self._spectral_target_band)

        self.lr = ckpt.get('lr', self.lr)
        self.max_change = ckpt.get('max_change', self.max_change)
        self.lr_auto = ckpt.get('lr_auto', self.lr_auto)
        self.lr_base = ckpt.get('lr_base', self.lr_base)
        self.max_change_auto = ckpt.get(
            'max_change_auto', self.max_change_auto)
        self.max_change_base = ckpt.get(
            'max_change_base', self.max_change_base)
        self.decimation_auto = ckpt.get(
            'decimation_auto', self.decimation_auto)
        self.decimation_range = ckpt.get(
            'decimation_range',
            0.25 if self.decimation_auto else self.decimation_range)
        self.auto_mode = ckpt.get('auto_mode', self.auto_mode)
        self._input_coherence = ckpt.get(
            'input_coherence', self._input_coherence)
        self._output_coherence = ckpt.get(
            'output_coherence', self._output_coherence)
        self._input_trust = ckpt.get('input_trust', self._input_trust)
        self._error_concentration = ckpt.get(
            'error_concentration', self._error_concentration)
        self._io2_plasticity = ckpt.get(
            'io2_plasticity', self._io2_plasticity)
        self._ema_fast = ckpt.get('ema_fast', None)
        self._ema_slow = ckpt.get('ema_slow', None)
        self.save_path = ckpt.get('save_path', self.save_path)
        self.save_every = ckpt.get('save_every', self.save_every)
        self.weight_decay = ckpt.get(
            'weight_decay', ckpt.get('shrinkage', self.weight_decay))
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
            self.decimation_band = (0 if ds <= 1 else max(0, int(round(
                np.log(ds) / np.log(self.base)))))
            self.checkpoint_decimation_band = self.decimation_band

        if self.hidden_dim > 0:
            if 'U' in ckpt:
                self.U_list[0] = ckpt['U'].float().to(self.device)
            self.u_norm_list[0] = ckpt.get('u_norm', self.u_norm_list[0])
            self.hidden_budget = ckpt.get(
                'hidden_budget', self.hidden_budget)
            for ell in range(1, self.n_layers):
                if f'U_{ell}' in ckpt:
                    self.U_list[ell] = ckpt[f'U_{ell}'].float().to(
                        self.device)
                    self.W_list[ell] = ckpt[f'W_{ell}'].float().to(
                        self.device)
                    self.u_norm_list[ell] = ckpt.get(
                        f'u_norm_{ell}', self.u_norm_list[ell])
                if f'traces_{ell}' in ckpt:
                    self.banks[ell].load_state(ckpt[f'traces_{ell}'])
                self._phase[ell] = ckpt.get(f'phase_{ell}', 0)
                if f'carry_{ell}' in ckpt:
                    self._carry[ell] = torch.from_numpy(
                        np.asarray(ckpt[f'carry_{ell}'],
                                   dtype=np.float64)).clone()
            if self.scale_gate:
                for ell in range(self.n_layers):
                    if f'G_{ell}' in ckpt:
                        self.G_list[ell] = ckpt[f'G_{ell}'].float().to(
                            self.device)
                        self.gb_list[ell] = ckpt[f'gb_{ell}'].float().to(
                            self.device)
            if self.Wd is not None and 'Wd' in ckpt:
                self.Wd = ckpt['Wd'].float().to(self.device)
                self.wd_norm = ckpt.get('wd_norm', self.wd_norm)
                self._normalize_Wd()
        self._normalize_all()
        self._update_decimation()
        print(f"    {GLYPH['load']} loaded {path}")

    # ── display ──

    def print_config(self):
        params = sum(W.numel() for W in self.W_list)
        params += sum(U.numel() for U in self.U_list)
        params += sum(G.numel() + gb.numel()
                      for G, gb in zip(self.G_list, self.gb_list))
        if self.Wd is not None:
            params += self.Wd.numel()

        hidden_str = (f"hidden={self.hidden_dim:,}"
                      if self.hidden_dim > 0 else "linear")
        if self.Wd is not None:
            hidden_str += " + direct"
        if self.n_layers > 1:
            hidden_str += f" {GLYPH['bullet']} layers={self.n_layers}"
            if self.clock > 1:
                hidden_str += f" {GLYPH['bullet']} clock={self.clock}"
        if self.scale_gate:
            hidden_str += f" {GLYPH['bullet']} gated"

        tok = (self.auto_mode if self.auto_mode in (
               'progress', 'spectral', 'full spectrum', 'io2') else 'auto')
        lr_str = (f"{tok} {self.lr_base}" if self.lr_auto else f"{self.lr}")
        mc_str = (f"{tok} {self.max_change_base}" if self.max_change_auto
                  else f"{self.max_change}")
        band_str = f"{self.n_bands} bands"
        if self.decimation_auto:
            band_str += (f" {GLYPH['bullet']} decimation auto "
                         f"(range={self.decimation_range})")
        elif self.decimation_band > 0:
            band_str += (f" {GLYPH['bullet']} decimation_band="
                         f"{self.decimation_band} (stride={self._stride})")

        print(f"\n  {GLYPH['dot']} soma v12 {GLYPH['bullet']} {self.device} "
              f"{GLYPH['bullet']} {_fmt_bytes(self.bytes_seen)} seen")
        print(f"    {band_str}")
        print(f"    base={self.base:.4f} "
              f"{GLYPH['bullet']} range={self.max_window:,.0f} "
              f"{GLYPH['bullet']} {hidden_str} "
              f"{GLYPH['bullet']} {_fmt_params(params)} params")
        print(f"    lr={lr_str} "
              f"{GLYPH['bullet']} max_change={mc_str}")
        print()


# ─────────────────────────────────────────────────────────────────────
# cli
# ─────────────────────────────────────────────────────────────────────

def main():
    _banner()

    mode = _prompt("mode (train/eval/chat): ")

    if mode == "train":
        corpus_input = _prompt("corpus: ")
        cycle_count = _parse_cycle_spec(corpus_input)
        if cycle_count is None:
            corpus = _resolve_path(corpus_input, 'corpus')
            cycle_files = None
            if not Path(corpus).exists():
                return print(f"  not found: {corpus}")
        else:
            corpus = corpus_input
            cycle_files = _data_cycle_files()
            if not cycle_files:
                return print(f"  no corpus files found in {DATA_DIR}")

        ckpt = _resolve_path(
            _prompt(f"checkpoint [{DEFAULT_CKPT}]: ", DEFAULT_CKPT),
            'checkpoint')
        if ckpt and Path(ckpt).exists():
            cfg = torch.load(ckpt, map_location='cpu', weights_only=False)
            saved_lr = cfg.get('lr', 0.1)
            saved_mc = cfg.get('max_change', 0.1)
            saved_wd = cfg.get('weight_decay', cfg.get('shrinkage', 1e-4))
            saved_bs = cfg.get('batch_size', 50000)
            saved_db = cfg.get('decimation_band', cfg.get('downsample', 0))
            saved_mode = cfg.get('auto_mode', 'level')
            saved_tok = (saved_mode if saved_mode in (
                         'progress', 'spectral', 'full spectrum', 'io2')
                         else 'auto')
            saved_lr_disp = (f"{saved_tok} {cfg['lr_base']}"
                             if cfg.get('lr_auto') else str(saved_lr))
            saved_mc_disp = (f"{saved_tok} {cfg['max_change_base']}"
                             if cfg.get('max_change_auto') else str(saved_mc))
            saved_dr = cfg.get(
                'decimation_range',
                1.0 if cfg.get('decimation_auto') else (
                    saved_db / max(1, cfg.get('n_bands', 46) - 1)))
            lr_str = _prompt(f"lr [{saved_lr_disp}]: ", saved_lr_disp)
            mc_str = _prompt(f"max_change [{saved_mc_disp}]: ", saved_mc_disp)
            lr_val, lr_auto, lr_base = _parse_auto_or_float(lr_str)
            mc_val, mc_auto, mc_base = _parse_auto_or_float(mc_str)
            auto_tokens = (lr_str + " " + mc_str).lower()
            auto_mode = ('io2' if 'io2' in auto_tokens
                         else ('full spectrum' if 'full spectrum' in auto_tokens
                         else ('spectral' if 'spectral' in auto_tokens
                         else ('progress' if 'progress' in auto_tokens
                               else cfg.get('auto_mode', 'level')))))
            wd = 0.0  # retired: decay-then-renormalize cancels exactly
            bs = int(_prompt(f"batch [{saved_bs}]: ", str(saved_bs)))
            db_range = _parse_decimation_range(_prompt(
                f"decimation range [{saved_dr}]: ", str(saved_dr)))

            model = SOMA(
                cfg.get('n_bands', cfg.get('num_timescales', 46)),
                base=cfg.get('base', PHI),
                hidden_dim=cfg.get('hidden_dim', 256),
                n_layers=cfg.get('n_layers', 1),
                scale_gate=bool(cfg.get('scale_gate', False)),
                clock=cfg.get('clock', 1), auto_mode=auto_mode,
                lr=lr_val, max_change=mc_val, weight_decay=wd,
                batch_size=bs, decimation_band=0,
                direct_readout=bool(cfg.get('direct_readout', False)),
                lr_auto=lr_auto, lr_base=lr_base,
                max_change_auto=mc_auto, max_change_base=mc_base,
                decimation_auto=True, decimation_range=db_range)
            model.load(ckpt)
            model.lr_auto = lr_auto
            model.lr_base = lr_base
            model.max_change_auto = mc_auto
            model.max_change_base = mc_base
            model.decimation_auto = True
            model.decimation_range = max(0.0, min(1.0, db_range))
            if not lr_auto:
                model.lr = lr_val
            if not mc_auto:
                model.max_change = mc_val
            model.weight_decay = wd
            model.batch_size = bs
            model.decimation_band = 0
            model.checkpoint_batch_size = bs
            model.checkpoint_decimation_band = 0
            model._update_decimation()
        else:
            shape = _prompt("config (enter=demisa / custom): ", "demisa")
            if shape != "custom":
                model = SOMA(**DEMISA)
                model.print_config()
                model.save_every = TRAIN_DEFAULTS['save_every']
                model.save_path = ckpt
                if cycle_files is None:
                    model.train(corpus, start_byte=0,
                                save_path=ckpt, **TRAIN_DEFAULTS)
                else:
                    print(f"  cycling {len(cycle_files)} data files "
                          f"{cycle_count} time"
                          f"{'s' if cycle_count != 1 else ''}")
                    for ci in range(cycle_count):
                        print(f"\n  cycle {ci + 1}/{cycle_count}")
                        for cp in cycle_files:
                            model.train(cp, start_byte=0,
                                        save_path=ckpt, **TRAIN_DEFAULTS)
                model.save(ckpt)
                return
            bands = int(_prompt("bands [50]: ", "50"))
            range_str = _prompt("range (base or window) [1.6180]: ", "1.6180")
            val = float(range_str)
            if val < 100:
                base, max_window = val, None
            else:
                base, max_window = None, val
            hd = int(_prompt("hidden (0=linear) [1024]: ", "1024"))
            layers = int(_prompt("layers [3]: ", "3"))
            sg = 0
            clk = 1
            lr_str = _prompt(
                "lr (n / auto / progress / spectral / full spectrum / io2) [auto]: ",
                "auto")
            mc_str = _prompt("max_change [auto]: ", "auto")
            lr_val, lr_auto, lr_base = _parse_auto_or_float(lr_str)
            mc_val, mc_auto, mc_base = _parse_auto_or_float(mc_str)
            auto_tokens = (lr_str + " " + mc_str).lower()
            auto_mode = ('io2' if 'io2' in auto_tokens
                         else ('full spectrum' if 'full spectrum' in auto_tokens
                         else ('spectral' if 'spectral' in auto_tokens
                         else ('progress' if 'progress' in auto_tokens
                         else 'io2'))))
            wd = 0.0  # retired: decay-then-renormalize cancels exactly
            bs = int(_prompt("batch [256]: ", "256"))
            ds_range = _parse_decimation_range(
                _prompt("decimation range [1.0]: ", "1.0"))
            dr = 0

            model = SOMA(bands, base=base, max_window=max_window,
                         hidden_dim=hd, n_layers=layers,
                         scale_gate=bool(sg), clock=clk,
                         auto_mode=auto_mode,
                         lr=lr_val, max_change=mc_val,
                         weight_decay=wd, batch_size=bs,
                         decimation_band=0, direct_readout=bool(dr),
                         lr_auto=lr_auto, lr_base=lr_base,
                         max_change_auto=mc_auto, max_change_base=mc_base,
                         decimation_auto=True, decimation_range=ds_range)

        model.print_config()

        epochs = int(_prompt("epochs [1]: ", "1"))
        start_byte = int(_prompt("start byte [0]: ", "0"))
        dream_every = int(_prompt("dream every batches (0=off) [50]: ", "50"))
        dream_length = _prompt("dream length [auto 200]: ", "auto 200")
        dream_temp = float(_prompt("dream temperature [1.0]: ", "1.0"))
        save_every = int(_prompt(
            f"save every (0=end) [{model.save_every}]: ",
            str(model.save_every)))
        save_path = _resolve_path(
            _prompt(f"save path [{model.save_path}]: ", model.save_path),
            'checkpoint')
        model.save_every = save_every
        model.save_path = save_path

        if cycle_files is None:
            model.train(corpus, epochs=epochs, start_byte=start_byte,
                        save_every=save_every, save_path=save_path,
                        dream_every_batches=dream_every,
                        dream_length=dream_length,
                        dream_temperature=dream_temp)
        else:
            print(f"  cycling {len(cycle_files)} data files "
                  f"{cycle_count} time{'s' if cycle_count != 1 else ''}")
            for cycle_idx in range(cycle_count):
                print(f"\n  cycle {cycle_idx + 1}/{cycle_count}")
                for corpus_path in cycle_files:
                    model.train(corpus_path, epochs=1, start_byte=0,
                                save_every=save_every, save_path=save_path,
                                dream_every_batches=dream_every,
                                dream_length=dream_length,
                                dream_temperature=dream_temp)
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
            n_layers=cfg.get('n_layers', 1),
            scale_gate=bool(cfg.get('scale_gate', False)),
            clock=cfg.get('clock', 1),
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
            n_layers=cfg.get('n_layers', 1),
            scale_gate=bool(cfg.get('scale_gate', False)),
            clock=cfg.get('clock', 1),
            direct_readout=bool(cfg.get('direct_readout', False)))
        model.load(ckpt)
        model.print_config()

        temp = float(_prompt("temperature [0.8]: ", "0.8"))
        maxlen = int(_prompt("max length [200]: ", "200"))
        online = _prompt(
            "online learning (y/n) [n]: ", "n").lower() in ('y', 'yes')

        if online:
            auto_tok = (model.auto_mode if model.auto_mode in (
                        'progress', 'spectral', 'full spectrum', 'io2')
                        else 'auto')
            lr_disp = (f"{auto_tok} {model.lr_base}"
                       if model.lr_auto else str(model.lr))
            mc_disp = (f"{auto_tok} {model.max_change_base}"
                       if model.max_change_auto else str(model.max_change))
            lr_str = _prompt(f"lr [{lr_disp}]: ", lr_disp)
            mc_str = _prompt(f"max_change [{mc_disp}]: ", mc_disp)
            lr_val, lr_auto, lr_base = _parse_auto_or_float(lr_str)
            mc_val, mc_auto, mc_base = _parse_auto_or_float(mc_str)
            auto_tokens = (lr_str + " " + mc_str).lower()
            model.auto_mode = ('io2' if 'io2' in auto_tokens
                               else ('full spectrum' if 'full spectrum' in auto_tokens
                               else ('spectral' if 'spectral' in auto_tokens
                               else ('progress' if 'progress' in auto_tokens
                                     else 'level'))))
            model.lr_auto = lr_auto
            model.lr_base = lr_base
            model.max_change_auto = mc_auto
            model.max_change_base = mc_base
            model.lr = lr_val if not lr_auto else lr_base
            model.max_change = mc_val if not mc_auto else mc_base
            model.decimation_band = 0
            model._update_decimation()
            print(f"    online learning enabled "
                  f"(learns from your input, not its own output)")

        print(f"\n  {GLYPH['chat']} chat "
              f"{GLYPH['bullet']} temp={temp} "
              f"{GLYPH['bullet']} online={online} "
              f"{GLYPH['bullet']} layers={model.n_layers} "
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
                print(f"  {GLYPH['gen']} {GLYPH['arrow']} ",
                      end='', flush=True)
                for ch in model.generate(length=maxlen, temperature=temp):
                    print(ch, end='', flush=True)
                print('\n')

        if _prompt("save state? (y/n) [y]: ", "y").lower() in ('y', 'yes'):
            save_path = _resolve_path(
                _prompt("save path: ", ckpt), 'checkpoint')
            model.save(save_path)


if __name__ == '__main__':
    main()
