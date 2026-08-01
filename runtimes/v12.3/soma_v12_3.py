"""soma v12.3.17: token-native continual spectral learning."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

import soma_v12 as core


PHI = 1.6180
EPS = 1e-8
TOKENIZER_NAME = "soma-8k-v1"
TOKEN_DIM = 256
ONLINE_TRAIN_BLOCK = 64
DEFAULT_AUTO_MODE = "io2"
DEFAULT_DECIMATION_STRIDE_CAP = 256
TURN_LOSS_WEIGHT = 0.1


def default_decimation_range(auto_mode):
    return 1.0 if str(auto_mode).strip().lower() == "io2" else 0.0


def _parse_auto_or_float(text, default_base=0.002):
    value = str(text).strip().lower()
    if value.startswith(("auto", "io2")):
        base = float(default_base)
        for part in reversed(value.split()[1:]):
            try:
                base = float(part)
                break
            except ValueError:
                pass
        return base, True, base
    try:
        number = float(value)
    except ValueError:
        number = float(default_base)
    return number, False, number


def _runtime_root():
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def _tokenizer_path():
    override = os.environ.get("SOMA_TOKENIZER", "")
    if override:
        return Path(override).expanduser()
    return _runtime_root() / "tokenizer" / f"{TOKENIZER_NAME}.model"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt_count(value):
    value = float(value)
    for suffix in ("", "k", "m", "g", "t"):
        if abs(value) < 1000:
            return f"{value:.1f}{suffix}" if suffix else f"{value:.0f}"
        value /= 1000.0
    return f"{value:.1f}p"


def _resolve_path(path, kind):
    if not path:
        return path
    path = os.path.expanduser(str(path))
    if os.path.isabs(path) or "/" in path or "\\" in path:
        return path
    home = os.environ.get("SOMA_HOME", "")
    if home:
        folder = "data" if kind == "corpus" else "checkpoints"
        root = Path(home) / folder
    else:
        root = Path("data" if kind == "corpus" else "checkpoints")
    root.mkdir(parents=True, exist_ok=True)
    return str(root / path)


class SomaTokenizer:
    def __init__(self, model_path=None):
        self.path = Path(model_path or _tokenizer_path())
        if not self.path.exists():
            raise FileNotFoundError(f"missing soma tokenizer: {self.path}")
        self.model_sha256 = _sha256(self.path)
        self.processor = spm.SentencePieceProcessor(model_file=str(self.path))
        self.vocab_size = int(self.processor.vocab_size())
        self.control_ids = {
            name: int(self.processor.piece_to_id(f"<|{name}|>"))
            for name in ("soma_state", "turn", "user", "soma", "dream")
        }
        self.generation_forbidden_ids = (
            int(self.processor.unk_id()),
            self.control_ids["soma_state"],
            self.control_ids["user"],
            self.control_ids["soma"],
            self.control_ids["dream"],
        )

    def encode(self, text):
        return np.asarray(
            self.processor.encode(str(text), out_type=int), dtype=np.int64)

    def decode(self, ids):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return self.processor.decode([int(value) for value in ids])

    def piece(self, token_id):
        return self.processor.id_to_piece(int(token_id))

    def incremental_decoder(self):
        return IncrementalTokenDecoder(self)


class IncrementalTokenDecoder:
    """sentencepiece-compatible streaming decode, including byte fallback."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pending_bytes = bytearray()

    def _flush_bytes(self, final=False):
        if not self.pending_bytes:
            return ""
        try:
            text = bytes(self.pending_bytes).decode("utf-8")
        except UnicodeDecodeError as error:
            if not final and error.reason == "unexpected end of data":
                return ""
            text = bytes(self.pending_bytes).decode("utf-8", errors="replace")
        self.pending_bytes.clear()
        return text

    def push(self, token_id):
        piece = self.tokenizer.piece(token_id)
        if (len(piece) == 6 and piece.startswith("<0x")
                and piece.endswith(">")):
            try:
                self.pending_bytes.append(int(piece[3:5], 16))
            except ValueError:
                return self._flush_bytes(final=True) + self.tokenizer.decode(
                    [token_id])
            return self._flush_bytes(final=False)
        return self._flush_bytes(final=True) + self.tokenizer.decode([token_id])

    def finish(self):
        return self._flush_bytes(final=True)


class TokenCorpus:
    """lossless-enough streaming tokenizer with exact source-byte cursors."""

    def __init__(self, path, tokenizer, start_byte=0, max_bytes=0):
        self.path = Path(_resolve_path(path, "corpus"))
        self.tokenizer = tokenizer
        self.file = self.path.open("rb")
        self.file_size = self.path.stat().st_size
        self.start_byte = max(0, min(int(start_byte), self.file_size))
        self.file.seek(self.start_byte)
        if self.start_byte:
            self._align_forward()
        self.start_byte = self.file.tell()
        self.limit = self.file_size
        if max_bytes and int(max_bytes) > 0:
            self.limit = min(self.file_size, self.start_byte + int(max_bytes))
        self.bytes_consumed = 0
        self.tokens_consumed = 0
        self._pending = []
        self._pending_tokens = 0
        self._protocol_mode = "plain"
        self.last_loss_weights = np.empty(0, dtype=np.float32)
        self.last_loss_bytes = 0.0
        if self.start_byte:
            self._prime_protocol_mode()

    def _prime_protocol_mode(self):
        """Recover dialogue masking state when resuming inside a record."""
        position = self.file.tell()
        lookback = min(position, 64 * 1024)
        self.file.seek(position - lookback)
        prefix = self.file.read(lookback).decode("utf-8", errors="replace")
        self.file.seek(position)
        ids = self.tokenizer.encode(prefix)
        if len(ids):
            self._loss_weights(ids)

    def _loss_weights(self, ids):
        """Mask synthetic protocol/state while retaining recorded language."""
        controls = self.tokenizer.control_ids
        control_ids = set(controls.values())
        weights = np.ones(len(ids), dtype=np.float32)
        mode = self._protocol_mode
        for index, token_id in enumerate(ids):
            token_id = int(token_id)
            if token_id == controls["soma_state"]:
                weights[index] = 0.0
                mode = "state"
                continue
            if mode == "state":
                weights[index] = 0.0
                if token_id == controls["turn"]:
                    mode = "assistant"
                continue
            if token_id in control_ids:
                weights[index] = 0.0
                if token_id == controls["user"]:
                    mode = "user"
                elif token_id in (controls["soma"], controls["dream"]):
                    mode = "assistant"
                elif token_id == controls["turn"]:
                    if mode == "assistant":
                        weights[index] = TURN_LOSS_WEIGHT
                    mode = "between"
        self._protocol_mode = mode
        return weights

    def _align_forward(self):
        while self.file.tell() < self.file_size:
            byte = self.file.read(1)
            if not byte or byte in b" \t\r\n":
                return

    @property
    def position(self):
        return self.start_byte + self.bytes_consumed

    @property
    def total_bytes(self):
        return max(0, self.limit - self.start_byte)

    def close(self):
        self.file.close()

    def _read_segment(self, target_bytes):
        remaining = self.limit - self.file.tell()
        if remaining <= 0:
            return b""
        take = min(max(256, int(target_bytes)), remaining)
        raw = self.file.read(take)
        if self.file.tell() < self.limit:
            extension = bytearray()
            while len(extension) < 1024 and self.file.tell() < self.limit:
                byte = self.file.read(1)
                if not byte:
                    break
                extension.extend(byte)
                if byte in b" \t\r\n":
                    break
            raw += bytes(extension)
        return raw

    def next_block(self, minimum_tokens):
        minimum_tokens = max(1, int(minimum_tokens))
        target_bytes = max(1024, int(minimum_tokens) * 4)
        while (self._pending_tokens < minimum_tokens
               and self.file.tell() < self.limit):
            raw = self._read_segment(target_bytes)
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace")
            encoded = self.tokenizer.encode(text)
            if len(encoded):
                loss_weights = self._loss_weights(encoded)
                self._pending.append({
                    "ids": encoded,
                    "loss_weights": loss_weights,
                    "offset": 0,
                    "source_bytes": len(raw),
                    "committed_bytes": 0,
                    "committed_loss_bytes": 0.0,
                })
                self._pending_tokens += len(encoded)
        if not self._pending_tokens:
            return None
        requested = min(minimum_tokens, self._pending_tokens)
        pieces = []
        loss_pieces = []
        source_bytes = 0
        loss_bytes = 0.0
        remaining = requested
        while remaining > 0:
            segment = self._pending[0]
            ids = segment["ids"]
            loss_weights = segment["loss_weights"]
            old_offset = segment["offset"]
            take = min(remaining, len(ids) - old_offset)
            new_offset = old_offset + take
            pieces.append(ids[old_offset:new_offset])
            loss_pieces.append(loss_weights[old_offset:new_offset])
            target_committed = round(
                segment["source_bytes"] * new_offset / len(ids))
            delta_bytes = target_committed - segment["committed_bytes"]
            source_bytes += delta_bytes
            target_loss_bytes = (
                segment["source_bytes"]
                * float(loss_weights[:new_offset].sum()) / len(ids))
            loss_bytes += (
                target_loss_bytes - segment["committed_loss_bytes"])
            segment["committed_bytes"] = target_committed
            segment["committed_loss_bytes"] = target_loss_bytes
            segment["offset"] = new_offset
            remaining -= take
            self._pending_tokens -= take
            if new_offset == len(ids):
                # Rounding cannot leave the final byte behind.
                source_bytes += (
                    segment["source_bytes"] - segment["committed_bytes"])
                self._pending.pop(0)
        ids = np.concatenate(pieces)
        self.last_loss_weights = np.concatenate(loss_pieces)
        self.last_loss_bytes = float(loss_bytes)
        self.bytes_consumed += source_bytes
        self.tokens_consumed += requested
        return ids, source_bytes, self.position


class ProjectedTokenBank:
    """exact token traces, projected through the current learned dictionary."""

    CHUNK = 128

    def __init__(self, vocab_size, token_dim, n_bands, base, device):
        self.vocab_size = int(vocab_size)
        self.token_dim = int(token_dim)
        self.n_bands = int(n_bands)
        self.base = float(base)
        self.device = device
        alpha = np.asarray(
            [1.0 / (self.base ** index) for index in range(self.n_bands)],
            dtype=np.float64)
        self.alphas_cpu = torch.from_numpy(alpha)
        self.decay_cpu = torch.from_numpy(1.0 - alpha)
        self.traces = torch.zeros(
            self.vocab_size, self.n_bands, dtype=torch.float64)
        self.alphas = self.alphas_cpu.float().to(self.device)
        self.decay = self.decay_cpu.float().to(self.device)
        self._build_kernels()

    @property
    def n_features(self):
        return self.token_dim * self.n_bands

    def _build_kernels(self):
        length = self.CHUNK
        kernels = np.zeros(
            (self.n_bands, length, length), dtype=np.float32)
        powers = np.zeros((self.n_bands, length), dtype=np.float32)
        steps = np.zeros((self.n_bands, length + 1), dtype=np.float32)
        for band in range(self.n_bands):
            alpha = float(self.alphas_cpu[band])
            decay = float(self.decay_cpu[band])
            for time_index in range(length):
                powers[band, time_index] = decay ** time_index
                for source_index in range(time_index):
                    kernels[band, time_index, source_index] = (
                        alpha * decay ** (time_index - 1 - source_index))
            for step in range(length + 1):
                steps[band, step] = decay ** step
        self.kernel = torch.from_numpy(kernels).to(self.device)
        self.powers = torch.from_numpy(powers).to(self.device)
        self.steps = torch.from_numpy(steps).to(self.device)

    def reset(self):
        self.traces.zero_()

    def _projected_state(self, embedding):
        identity = self.traces.T.float().to(self.device)
        return identity @ embedding.weight

    def projected_state(self, embedding):
        return self._projected_state(embedding)

    def _bandpass(self, states):
        output = torch.empty_like(states)
        output[:-1] = states[:-1] - states[1:]
        output[-1] = states[-1]
        return output

    def tap(self, embedding):
        state = self._projected_state(embedding)
        return self.tap_projected(state)

    def tap_projected(self, state):
        return self._bandpass(state).T.reshape(-1)

    def advance_projected(self, state, token_id, embedding):
        value = embedding.weight[int(token_id)].detach()
        return (
            state * self.decay.unsqueeze(1)
            + value.unsqueeze(0) * self.alphas.unsqueeze(1))

    def tick(self, token_id):
        self.traces.mul_(self.decay_cpu.unsqueeze(0))
        self.traces[int(token_id)].add_(self.alphas_cpu)

    def _advance_exact(self, ids):
        ids_cpu = torch.from_numpy(np.asarray(ids, dtype=np.int64))
        for start in range(0, len(ids_cpu), self.CHUNK):
            chunk = ids_cpu[start:start + self.CHUNK]
            count = len(chunk)
            self.traces.mul_(
                torch.pow(self.decay_cpu, count).unsqueeze(0))
            exponent = torch.arange(
                count - 1, -1, -1, dtype=torch.float64).unsqueeze(1)
            weights = self.alphas_cpu.unsqueeze(0) * torch.pow(
                self.decay_cpu.unsqueeze(0), exponent)
            self.traces.index_add_(0, chunk, weights)

    def process_block_select(self, ids, embedding, rows=None):
        ids = np.asarray(ids, dtype=np.int64)
        if rows is None:
            rows = np.arange(len(ids), dtype=np.int64)
        else:
            rows = np.asarray(rows, dtype=np.int64)
        ids_device = torch.from_numpy(ids).to(self.device)
        state = self._projected_state(embedding)
        outputs = []
        for start in range(0, len(ids), self.CHUNK):
            end = min(start + self.CHUNK, len(ids))
            count = end - start
            values = embedding(ids_device[start:end])
            conv = torch.matmul(
                self.kernel[:, :count, :count], values.unsqueeze(0))
            carry = (
                self.powers[:, :count].unsqueeze(2) * state.unsqueeze(1))
            states = conv + carry
            lo = int(np.searchsorted(rows, start, side="left"))
            hi = int(np.searchsorted(rows, end, side="left"))
            if hi > lo:
                local = torch.from_numpy(rows[lo:hi] - start).to(self.device)
                selected = states.index_select(1, local)
                bandpassed = self._bandpass(selected)
                outputs.append(
                    bandpassed.permute(1, 2, 0).reshape(hi - lo, -1))
            weights = self.powers[:, :count].T.flip(0)
            state = (
                state * self.steps[:, count].unsqueeze(1)
                + self.alphas.unsqueeze(1) * (values.T @ weights).T)
        self._advance_exact(ids)
        if not outputs:
            return torch.empty(
                0, self.n_features, dtype=embedding.weight.dtype,
                device=self.device)
        return torch.cat(outputs, dim=0)


class BudgetRelu(nn.Module):
    def __init__(self, budget):
        super().__init__()
        self.budget = float(budget)

    def forward(self, value):
        positive = F.relu(value)
        return positive * (
            self.budget / (positive.sum(dim=1, keepdim=True) + EPS))


class TokenResidualNet(nn.Module):
    def __init__(self, vocab_size, token_dim, n_bands, hidden_dim, depth,
                 budget_scale=0.1, residual_init=0.1):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.token_dim = int(token_dim)
        self.n_bands = int(n_bands)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.n_features = self.token_dim * self.n_bands
        self.budget = self.hidden_dim * float(budget_scale)
        self.embedding = nn.Embedding(self.vocab_size, self.token_dim)
        self.input = nn.Linear(self.n_features, self.hidden_dim, bias=False)
        self.layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            for _ in range(max(0, self.depth - 1))
        ])
        gate = math.log(residual_init / max(EPS, 1.0 - residual_init))
        self.residual_logits = nn.ParameterList([
            nn.Parameter(torch.tensor(gate)) for _ in self.layers
        ])
        self.language = nn.Linear(self.hidden_dim, self.token_dim, bias=False)
        self.token_bias = nn.Parameter(torch.zeros(self.vocab_size))
        self.activation = BudgetRelu(self.budget)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.embedding.weight.normal_(std=0.1)
            for layer in [self.input, *self.layers, self.language]:
                layer.weight.normal_()
                target = math.sqrt(layer.weight.shape[1]) * 0.1
                layer.weight.mul_(
                    target / (layer.weight.norm(dim=1, keepdim=True) + EPS))
            self.token_bias.zero_()

    def forward(self, features, return_language=False):
        hidden = self.activation(self.input(features))
        for index, layer in enumerate(self.layers):
            gate = torch.sigmoid(self.residual_logits[index])
            hidden = self.activation(hidden + gate * layer(hidden))
        language = self.language(hidden)
        logits = F.linear(language, self.embedding.weight, self.token_bias)
        if return_language:
            return logits, language, hidden
        return logits


class TranslatedErrorBank:
    """geometric traces of signed categorical error in token space."""

    def __init__(self, n_bands, dimension, base):
        self.n_bands = int(n_bands)
        self.dimension = int(dimension)
        alpha = torch.tensor(
            [1.0 / (base ** index) for index in range(self.n_bands)],
            dtype=torch.float64)
        self.alpha = alpha
        self.decay = 1.0 - alpha
        self.trace = torch.zeros(
            self.n_bands, self.dimension, dtype=torch.float64)

    def reset(self):
        self.trace.zero_()

    def advance(self, values, gaps=1):
        values = values.detach().float().cpu().double()
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if np.isscalar(gaps):
            gaps = [max(1, int(gaps))] * len(values)
        for value, gap in zip(values, gaps):
            retention = torch.pow(self.decay, max(1, int(gap)))
            self.trace.mul_(retention.unsqueeze(1)).add_(
                (1.0 - retention).unsqueeze(1) * value.unsqueeze(0))

    def elapse(self, steps):
        steps = max(0, int(steps))
        if steps:
            self.trace.mul_(torch.pow(self.decay, steps).unsqueeze(1))

    def bandpass(self):
        output = torch.empty_like(self.trace)
        output[:-1] = self.trace[:-1] - self.trace[1:]
        output[-1] = self.trace[-1]
        return output


class TokenSOMA:
    species = "soma_v12_3_token_residual"

    def __init__(self, n_bands=16, hidden_dim=None, depth=4,
                 base=PHI, batch_size=512, lr=0.002, lr_auto=True,
                 lr_base=None, grad_clip=1.0, auto_mode=DEFAULT_AUTO_MODE,
                 decimation_range=1.0, decimation_stride_cap=256,
                 row_norm="auto", row_norm_mult=4.0, row_norm_every=100,
                 description="", device="auto", tokenizer_path=None):
        self.tokenizer = SomaTokenizer(tokenizer_path)
        self.vocab_size = self.tokenizer.vocab_size
        self.token_dim = TOKEN_DIM
        self.n_bands = int(n_bands)
        self.hidden_dim = (
            self.n_bands * self.token_dim
            if hidden_dim is None else int(hidden_dim))
        self.depth = int(depth)
        self.base = float(base)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.lr_base = float(lr if lr_base is None else lr_base)
        self.lr_auto = bool(lr_auto)
        self.grad_clip = float(grad_clip)
        self.auto_mode = DEFAULT_AUTO_MODE
        # Kept under its historical checkpoint key for compatibility. It is
        # now a gain on io2's selected band, not a maximum-band selector.
        self.decimation_range = max(0.0, float(decimation_range))
        self.decimation_stride_cap = max(1, int(decimation_stride_cap))
        self.decimation_band = 0.0
        self.row_norm = str(row_norm)
        self.row_norm_mult = float(row_norm_mult)
        self.row_norm_every = max(0, int(row_norm_every))
        self.description = str(description or "")
        self.device = core.SOMA._select_device(device)
        self.net = TokenResidualNet(
            self.vocab_size, self.token_dim, self.n_bands,
            self.hidden_dim, self.depth).to(self.device)
        self.bank = ProjectedTokenBank(
            self.vocab_size, self.token_dim, self.n_bands,
            self.base, self.device)
        self.error_bank = TranslatedErrorBank(
            self.n_bands, self.token_dim, self.base)
        self.opt = self._build_optimizer()
        self.bytes_seen = 0
        self.tokens_seen = 0
        self._train_steps = 0
        self._stride = 1
        self._sampled_stride = 1
        self._bytes_per_token = 3.236
        self._byte_stride = self._bytes_per_token
        self._row_clip_fraction = 0.0
        self._row_clip_max_ratio = 0.0
        self._row_compression_fraction = 0.0
        self._row_norm_reference = []
        self._row_shared_fraction = 0.0
        self._io2_plasticity = 1.0
        self._spectral_target_band = 0.0
        self._spectral_drive = torch.ones(self.n_bands)
        self._input_spectral_drive = torch.ones(self.n_bands)
        self._weight_spectral_drive = torch.ones(self.n_bands)
        self._input_coherence = 0.0
        self._output_coherence = 0.0
        self._error_energy = 0.0
        self._error_coherence = 0.0
        self._error_concentration = 0.0
        self._layer_maturity = 0.0
        self._layer_maturity_target = 0.0
        self._layer_competence = 0.0
        self._layer_plasticity_floor = 0.03
        self._layer_lr_multipliers = [1.0] * (self.depth + 1)
        self._last_token_loss = None
        self._top5_accuracy = 0.0
        self._top_predictions = []
        self._active_memory = []
        self._dream_input_tokens = 0
        self._dream_output_tokens = 0
        self._dream_rehearsal_ratio = 0.0
        self._diagnostic_every = 32
        self.checkpoint_history = []
        self.checkpoint_id = uuid.uuid4().hex
        self._update_decimation()
        self._reset_row_norm_reference()

    def params(self):
        return sum(parameter.numel() for parameter in self.net.parameters())

    def _optimizer_group_specs(self):
        groups = [{
            "params": [
                self.net.embedding.weight, self.net.input.weight],
            "layer_name": "dictionary_input",
        }]
        for index, layer in enumerate(self.net.layers):
            groups.append({
                "params": [layer.weight, self.net.residual_logits[index]],
                "layer_name": f"residual_{index}",
            })
        groups.append({
            "params": [
                self.net.language.weight, self.net.token_bias],
            "layer_name": "language_readout",
        })
        return groups

    def _build_optimizer(self):
        return torch.optim.AdamW(
            self._optimizer_group_specs(), lr=self.lr, weight_decay=0.0)

    def _max_decimation_band(self):
        cap = math.log(max(1, self.decimation_stride_cap), self.base)
        return max(0.0, float(cap))

    def _native_decimation_band(self):
        return max(0.0, min(
            float(self.n_bands - 1), self._max_decimation_band()))

    def _update_decimation(self):
        self.decimation_band = max(
            0.0, min(float(self._max_decimation_band()), self.decimation_band))
        self._stride = min(
            self.decimation_stride_cap,
            max(1, int(round(self.base ** self.decimation_band))))
        self._sampled_stride = self._stride
        self._byte_stride = self._stride * self._bytes_per_token

    def _layer_gates(self):
        if self.auto_mode != "io2":
            return [1.0] * len(self.opt.param_groups)
        maturity = max(0.0, min(1.0, self._layer_maturity))
        hidden = [
            self._layer_plasticity_floor
            + (1.0 - self._layer_plasticity_floor)
            * (1.0 - maturity ** (index + 1))
            for index in range(max(1, len(self.opt.param_groups) - 1))
        ]
        return hidden + [1.0]

    def _set_optimizer_lrs(self):
        self._layer_lr_multipliers = self._layer_gates()
        for group, multiplier in zip(
                self.opt.param_groups, self._layer_lr_multipliers):
            group["lr"] = self.lr * float(multiplier)

    def _row_modules(self):
        return [
            self.net.embedding, self.net.input, *self.net.layers,
            self.net.language]

    def _row_ceiling(self, weight):
        if self.row_norm.strip().lower() in ("off", "none", "0", "false"):
            return None
        return math.sqrt(weight.shape[1]) * 0.1 * self.row_norm_mult

    def _reset_row_norm_reference(self):
        with torch.no_grad():
            self._row_norm_reference = [
                module.weight.norm(dim=1, keepdim=True).detach().clone()
                for module in self._row_modules()
            ]

    def _apply_weight_compressor(self, diagnose=False):
        """compress outward row growth while preserving learned direction."""
        modules = self._row_modules()
        if len(self._row_norm_reference) != len(modules):
            self._reset_row_norm_reference()
        proposed_growth = 0.0
        rejected_growth = 0.0
        pressured = 0
        total = 0
        max_ratio = 0.0
        with torch.no_grad():
            for index, module in enumerate(modules):
                weight = module.weight
                ceiling = self._row_ceiling(weight)
                if not ceiling:
                    continue
                proposed_norm = weight.norm(dim=1, keepdim=True)
                reference = self._row_norm_reference[index].to(
                    device=weight.device, dtype=proposed_norm.dtype)
                outward = torch.clamp(proposed_norm - reference, min=0.0)
                relative = torch.clamp(reference / ceiling, 0.0, 1.0)
                # fourth-power knee leaves small rows almost untouched, then
                # smoothly removes radial plasticity near the capacity wall.
                radial_gain = torch.clamp(1.0 - relative.pow(4), 0.0, 1.0)
                target_norm = torch.where(
                    proposed_norm > reference,
                    reference + radial_gain * outward,
                    proposed_norm)
                target_norm = torch.clamp(target_norm, max=float(ceiling))
                weight.mul_(target_norm / (proposed_norm + EPS))
                actual_norm = weight.norm(dim=1, keepdim=True)
                self._row_norm_reference[index] = actual_norm.detach().clone()
                if diagnose:
                    rejected = torch.clamp(
                        proposed_norm - actual_norm, min=0.0)
                    proposed_growth += float(outward.sum().item())
                    rejected_growth += float(rejected.sum().item())
                    pressured += int((rejected > EPS).sum().item())
                    total += proposed_norm.numel()
                    max_ratio = max(
                        max_ratio,
                        float((proposed_norm / ceiling).max().item()))
        if diagnose:
            pressure = max(0.0, min(
                1.0, rejected_growth / max(proposed_growth, EPS)))
            self._row_compression_fraction = pressure
            # Retain the old field for checkpoint and web parser compatibility.
            self._row_clip_fraction = pressure
            self._row_clip_max_ratio = max_ratio

    def _input_spectrum(self):
        with torch.no_grad():
            projected = self.bank._projected_state(self.net.embedding)
            bandpassed = self.bank._bandpass(projected)
            energy = torch.linalg.vector_norm(bandpassed, dim=1)
            total = energy.sum()
            drive = self.n_bands * energy / (total + EPS)
            probability = energy / (total + EPS)
            entropy = -(probability * torch.log(probability + EPS)).sum()
            coherence = 1.0 - entropy / math.log(max(2, self.n_bands))
        self._input_spectral_drive = torch.clamp(
            drive, 0.1, 3.0).detach().cpu()
        self._input_coherence = float(torch.clamp(coherence, 0.0, 1.0).cpu())

    def _representation_diagnostics(self, logits=None):
        """bounded-cost view of first-layer allocation and token memory."""
        with torch.no_grad():
            weight = self.net.input.weight.detach()
            sample_count = min(256, weight.shape[0])
            rows = torch.linspace(
                0, weight.shape[0] - 1, sample_count,
                device=weight.device).round().long()
            sampled = weight.index_select(0, rows).reshape(
                sample_count, self.token_dim, self.n_bands)
            energy = sampled.float().square().mean(dim=(0, 1))
            self._weight_spectral_drive = (
                self.n_bands * energy / (energy.sum() + EPS)
            ).detach().cpu()

            traces = self.bank.traces
            bandpassed = torch.empty_like(traces)
            bandpassed[:, :-1] = traces[:, :-1] - traces[:, 1:]
            bandpassed[:, -1] = traces[:, -1]
            band_gain = self._weight_spectral_drive.double()
            salience = (bandpassed.abs() * band_gain.unsqueeze(0)).sum(1)
            salience *= self.net.embedding.weight.detach().float().norm(
                dim=1).cpu().double()
            count = min(64, len(salience))
            values, token_ids = torch.topk(salience, count)
            peak = float(values.max().item()) if count else 0.0
            active = []
            for value, token_id in zip(values.tolist(), token_ids.tolist()):
                token = self.tokenizer.decode([int(token_id)])
                if float(value) <= EPS:
                    continue
                if not any(character.isalnum() for character in token):
                    continue
                active.append({
                    "token": token or self.tokenizer.piece(int(token_id)),
                    "piece": self.tokenizer.piece(int(token_id)),
                    "score": float(value) / max(peak, EPS),
                })
                if len(active) >= 6:
                    break
            self._active_memory = active

            if logits is not None and len(logits):
                probabilities = F.softmax(logits[-1].detach().float(), dim=0)
                values, token_ids = torch.topk(probabilities, min(5, len(probabilities)))
                self._top_predictions = [
                    {
                        "token": (
                            self.tokenizer.decode([int(token_id)])
                            or self.tokenizer.piece(int(token_id))),
                        "piece": self.tokenizer.piece(int(token_id)),
                        "probability": float(value),
                    }
                    for value, token_id in zip(values.tolist(), token_ids.tolist())
                ]

    def _update_error_state(self, translated, probabilities, stride):
        dictionary = self.net.embedding.weight.detach()
        dictionary_scale = dictionary.norm(dim=1).mean()
        translated = translated / (dictionary_scale + EPS)
        self.error_bank.advance(translated, gaps=stride)
        bandpassed = torch.linalg.vector_norm(
            self.error_bank.bandpass(), dim=1)
        total = bandpassed.sum()
        probability = bandpassed / (total + EPS)
        entropy = -(probability * torch.log(probability + EPS)).sum()
        concentration = 1.0 - entropy / math.log(max(2, self.n_bands))
        bands = torch.arange(self.n_bands, dtype=torch.float64)
        center = (bands * bandpassed).sum() / (total + EPS)
        drive = self.n_bands * bandpassed / (total + EPS)
        self._spectral_drive = torch.clamp(drive, 0.1, 3.0).float()
        self._spectral_target_band = float(center)
        self._error_energy = float(total / (total + math.sqrt(self.n_bands)))
        self._error_concentration = float(max(0.0, min(1.0, concentration)))
        self._error_coherence = self._error_concentration
        entropy_out = -(probabilities * torch.log(
            probabilities + EPS)).sum(dim=1).mean()
        self._output_coherence = float(torch.clamp(
            1.0 - entropy_out / math.log(self.vocab_size), 0.0, 1.0).cpu())
        self._input_spectrum()

    def _io2_ratio(self, token_loss):
        energy = max(0.0, min(1.0, self._error_energy))
        coherence = max(0.0, min(1.0, self._error_coherence))
        plasticity = 2.5 * math.sqrt(energy) * (
            0.25 + 0.75 * math.sqrt(coherence))
        chance = math.log(self.vocab_size)
        loss_scale = self.base ** (
            (float(token_loss) - chance) / max(chance, EPS))
        plasticity *= max(1.0 / self.base, min(self.base, loss_scale))
        return max(0.02, min(1.0, plasticity))

    @staticmethod
    def _throughput_relaxation(nats_per_byte):
        """Mastery bonus begins below one bit per source byte."""
        bits_per_byte = float(nats_per_byte) / math.log(2.0)
        return max(0.0, min(1.0, 1.0 - bits_per_byte))

    def _apply_controller(self, token_loss, nats_per_byte):
        if self.auto_mode != "io2":
            return
        ratio = self._io2_ratio(token_loss)
        self._io2_plasticity = ratio
        if self.lr_auto:
            self.lr = self.lr_base * ratio
        chance = math.log(self.vocab_size)
        competence = 1.0 - token_loss / max(chance, EPS)
        target = max(0.0, min(1.0, competence * (1.0 - ratio)))
        rate = 0.01 if target > self._layer_maturity else 0.002
        self._layer_maturity += rate * (target - self._layer_maturity)
        self._layer_maturity_target = target
        self._layer_competence = max(0.0, min(1.0, competence))
        self._set_optimizer_lrs()
        max_band = self._max_decimation_band()
        native_band = self._native_decimation_band()
        spectral = max(
            0.0, min(float(native_band), self._spectral_target_band))
        relaxation = self._throughput_relaxation(nats_per_byte)
        throughput_target = (
            (1.0 - relaxation) * spectral + relaxation * native_band)
        raw_target = (1.0 - ratio) * throughput_target
        target_band = max(0.0, min(
            float(max_band), self.decimation_range * raw_target))
        if target_band > self.decimation_band:
            target_band = min(target_band, self.decimation_band + 1.0)
        else:
            target_band = max(target_band, self.decimation_band - 2.0)
        self.decimation_band = target_band
        self._update_decimation()
        self._last_token_loss = float(token_loss)

    def _train_tokens(self, ids, source_bytes, stride=1, pad_to=0,
                      loss_weights=None, loss_source_bytes=None):
        rows = np.arange(0, len(ids), max(1, int(stride)), dtype=np.int64)
        if loss_weights is None:
            loss_weights = np.ones(len(ids), dtype=np.float32)
        else:
            loss_weights = np.asarray(loss_weights, dtype=np.float32)
            if len(loss_weights) != len(ids):
                raise ValueError("loss weights must match token ids")
        selected_weights = loss_weights[rows]
        weight_total = float(selected_weights.sum())
        full_weight_total = float(loss_weights.sum())
        if loss_source_bytes is None:
            loss_source_bytes = float(source_bytes)
        if weight_total <= EPS:
            self.bank.process_block_select(
                ids, self.net.embedding, rows=np.empty(0, dtype=np.int64))
            self.error_bank.elapse(len(ids))
            self.tokens_seen += len(ids)
            self.bytes_seen += int(source_bytes)
            return {
                "token_loss": self._last_token_loss or 0.0,
                "nats_per_byte": 0.0,
                "nll": 0.0,
                "loss_tokens": 0.0,
                "loss_bytes": 0.0,
                "correct": 0,
                "top5": 0,
                "samples": 0,
                "tokens": len(ids),
                "bytes": int(source_bytes),
            }
        features = self.bank.process_block_select(
            ids, self.net.embedding, rows)
        targets = torch.from_numpy(ids[rows]).to(self.device)
        weights = torch.from_numpy(selected_weights).to(self.device)
        padded_count = max(len(features), int(pad_to))
        if padded_count > len(features):
            padding = torch.zeros(
                padded_count - len(features), features.shape[1],
                dtype=features.dtype, device=features.device)
            network_features = torch.cat((features, padding), dim=0)
        else:
            network_features = features
        logits_full, language, _ = self.net(
            network_features, return_language=True)
        logits = logits_full[:len(targets)]
        language.retain_grad()
        token_losses = F.cross_entropy(logits, targets, reduction="none")
        loss = (token_losses * weights).sum() / weights.sum()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        translated_error = (
            -language.grad[:len(targets)].detach() * weight_total)
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), self.grad_clip)
        self.opt.step()
        self._train_steps += 1
        if (self.row_norm_every > 0
                and self._train_steps % self.row_norm_every == 0):
            self._apply_weight_compressor(
                diagnose=(self._train_steps % (10 * self.row_norm_every) == 0))
        with torch.no_grad():
            probabilities = F.softmax(logits.detach(), dim=1)
            metric_mask = weights >= 0.5
            correct = int(((logits.argmax(1) == targets)
                           & metric_mask).sum().item())
            top5 = int(((logits.topk(
                min(5, self.vocab_size), dim=1).indices
                == targets.unsqueeze(1)).any(dim=1) & metric_mask).sum().item())
            samples = int(metric_mask.sum().item())
            self._update_error_state(
                translated_error, probabilities, stride)
            if self._train_steps == 1 or (
                    self._train_steps % self._diagnostic_every == 0):
                self._representation_diagnostics(logits)
        token_loss = float(loss.item())
        # Stride samples the objective; report/controller byte loss estimates
        # the full supervised stream, as the pre-mask runtime did.
        estimated_nll = token_loss * full_weight_total
        nats_per_byte = estimated_nll / max(EPS, float(loss_source_bytes))
        if source_bytes > 0 and len(ids):
            observed = source_bytes / len(ids)
            self._bytes_per_token += 0.02 * (
                observed - self._bytes_per_token)
            self._byte_stride = self._stride * self._bytes_per_token
        self._apply_controller(token_loss, nats_per_byte)
        self.tokens_seen += len(ids)
        self.bytes_seen += int(source_bytes)
        return {
            "token_loss": token_loss,
            "nats_per_byte": nats_per_byte,
            "nll": estimated_nll,
            "loss_tokens": full_weight_total,
            "loss_bytes": float(loss_source_bytes),
            "correct": correct,
            "top5": top5,
            "samples": samples,
            "tokens": len(ids),
            "bytes": int(source_bytes),
        }

    def train(self, corpus_path, epochs=1, save_path="v12_3.pt",
              start_byte=0, max_bytes=0, report_every=1_000_000,
              save_every=10_000_000, dream_every_batches=64,
              dream_length="128", dream_temperature=1.0,
              dream_callback=None):
        print()
        print(f"  soma v12.3.17 token residual · {self.device}")
        print(
            f"  {self.n_bands} bands · token dim {self.token_dim} · "
            f"hidden {self.hidden_dim:,} · depth {self.depth} · "
            f"{self.params()/1e6:.2f}m params")
        print(
            f"  {self.tokenizer.vocab_size:,} token {TOKENIZER_NAME} · "
            f"lr {self.lr:g} · controller io2 · batch {self.batch_size}")
        print()
        for epoch in range(int(epochs)):
            reader = TokenCorpus(
                corpus_path, self.tokenizer,
                start_byte=(start_byte if epoch == 0 else 0),
                max_bytes=max_bytes)
            started = time.time()
            report_bytes = 0
            save_bytes = 0
            batch_count = 0
            input_tokens_since_dream = 0
            totals = {
                "nll": 0.0, "bytes": 0, "correct": 0, "top5": 0,
                "samples": 0, "tokens": 0, "loss_tokens": 0.0,
                "loss_bytes": 0.0,
            }
            try:
                while True:
                    stride = max(1, self._stride)
                    block = reader.next_block(self.batch_size * stride)
                    if block is None:
                        break
                    ids, source_bytes, _ = block
                    result = self._train_tokens(
                        ids, source_bytes, stride,
                        loss_weights=reader.last_loss_weights,
                        loss_source_bytes=reader.last_loss_bytes)
                    totals["nll"] += result["nll"]
                    totals["bytes"] += result["bytes"]
                    totals["correct"] += result["correct"]
                    totals["top5"] += result["top5"]
                    totals["samples"] += result["samples"]
                    totals["tokens"] += result["tokens"]
                    totals["loss_tokens"] += result["loss_tokens"]
                    totals["loss_bytes"] += result["loss_bytes"]
                    report_bytes += result["bytes"]
                    save_bytes += result["bytes"]
                    batch_count += 1
                    input_tokens_since_dream += result["tokens"]

                    if (dream_every_batches and
                            batch_count % int(dream_every_batches) == 0):
                        length = self._dream_length(
                            dream_length, result["token_loss"])
                        # A native marker preserves the self/world boundary
                        # without filling the fast bands with telemetry text.
                        self.ingest_text("<|dream|>", online=False)
                        dream_ids = self.generate_ids(
                            length=0, temperature=dream_temperature,
                            prelude=False, source="dream",
                            max_tokens=length, close_turn=True)
                        text = self.tokenizer.decode(dream_ids)
                        self._dream_input_tokens = input_tokens_since_dream
                        self._dream_output_tokens = len(dream_ids)
                        self._dream_rehearsal_ratio = (
                            len(dream_ids)
                            / max(1, input_tokens_since_dream))
                        input_tokens_since_dream = 0
                        if dream_callback:
                            dream_callback(text, batch_count, self.bytes_seen)
                        else:
                            print(
                                f"\n  dream {batch_count} · "
                                f"{_fmt_count(self.bytes_seen)} bytes seen")
                            print(f"  {text}\n", flush=True)

                    if report_every and report_bytes >= int(report_every):
                        self._report(epoch, epochs, reader, totals, started)
                        report_bytes = 0
                    if save_path and save_every and save_bytes >= int(save_every):
                        self.save(save_path)
                        save_bytes = 0
                self._report(epoch, epochs, reader, totals, started)
            finally:
                reader.close()
        if save_path:
            self.save(save_path)
        return reader.position

    def _dream_length(self, value, token_loss):
        text = str(value).strip().lower()
        if text.startswith("auto"):
            parts = text.split()
            cap = int(parts[-1]) if len(parts) > 1 else 128
            confidence = max(
                0.0, min(1.0, 1.0 - token_loss / math.log(self.vocab_size)))
            return max(1, int(round(cap * confidence)))
        try:
            return max(0, int(float(text)))
        except ValueError:
            return 128

    def _report(self, epoch, epochs, reader, totals, started):
        elapsed = time.time() - started
        token_loss = totals["nll"] / max(EPS, totals["loss_tokens"])
        nats_byte = totals["nll"] / max(EPS, totals["loss_bytes"])
        accuracy = totals["correct"] / max(1, totals["samples"])
        top5_accuracy = totals["top5"] / max(1, totals["samples"])
        self._top5_accuracy = top5_accuracy
        bps = totals["bytes"] / max(elapsed, EPS)
        fraction = reader.bytes_consumed / max(1, reader.total_bytes)
        print(
            f"  [{epoch+1}/{epochs}] {100*fraction:5.1f}% · "
            f"{token_loss:.3f} n/tok · {nats_byte:.3f} n/byte · "
            f"{100*accuracy:.1f}% top1 · {100*top5_accuracy:.1f}% top5 · "
            f"{bps:,.0f} b/s · "
            f"stride {self._stride}t/{self._byte_stride:.1f}b · "
            f"lr {self.lr:.6f} · "
            f"io2 {self._io2_plasticity:.3f} · "
            f"compression {100*self._row_compression_fraction:.1f}%",
            flush=True)

    def evaluate(self, corpus_path, max_bytes=0, start_byte=0):
        reader = TokenCorpus(
            corpus_path, self.tokenizer, start_byte=start_byte,
            max_bytes=max_bytes)
        self.bank.reset()
        was_training = self.net.training
        self.net.eval()
        total_nll = 0.0
        total_tokens = 0
        total_loss_tokens = 0.0
        total_bytes = 0
        total_loss_bytes = 0.0
        correct = 0
        total_samples = 0
        started = time.time()
        try:
            with torch.no_grad():
                while True:
                    block = reader.next_block(self.batch_size)
                    if block is None:
                        break
                    ids, source_bytes, _ = block
                    features = self.bank.process_block_select(
                        ids, self.net.embedding)
                    targets = torch.from_numpy(ids).to(self.device)
                    weights = torch.from_numpy(
                        reader.last_loss_weights).to(self.device)
                    logits = self.net(features)
                    losses = F.cross_entropy(
                        logits, targets, reduction="none")
                    total_nll += float((losses * weights).sum().item())
                    total_tokens += len(ids)
                    total_loss_tokens += float(weights.sum().item())
                    total_bytes += source_bytes
                    total_loss_bytes += reader.last_loss_bytes
                    metric_mask = weights >= 0.5
                    correct += int(((logits.argmax(1) == targets)
                                    & metric_mask).sum().item())
                    total_samples += int(metric_mask.sum().item())
        finally:
            reader.close()
            if was_training:
                self.net.train()
        result = {
            "token_loss": total_nll / max(EPS, total_loss_tokens),
            "nats_per_byte": total_nll / max(EPS, total_loss_bytes),
            "bits_per_byte": (
                total_nll / max(EPS, total_loss_bytes) / math.log(2.0)),
            "accuracy": correct / max(1, total_samples),
            "bytes_per_second": total_bytes / max(time.time() - started, EPS),
            "tokens": total_tokens,
            "bytes": total_bytes,
        }
        print(
            f"  eval · {result['token_loss']:.3f} n/tok · "
            f"{result['nats_per_byte']:.3f} n/byte · "
            f"{result['bits_per_byte']:.3f} bpb · "
            f"{100*result['accuracy']:.1f}%")
        return result

    def _state_prelude(self, source):
        return (
            f"<|soma_state|>\nsource: {source}\n"
            f"bytes_seen: {self.bytes_seen}\n"
            f"tokens_seen: {self.tokens_seen}\n"
            f"lr: {self.lr:.8f}\ntoken_stride: {self._stride}\n"
            f"byte_stride: {self._byte_stride:.4f}\n"
            f"io2: {self._io2_plasticity:.6f}\n"
            f"maturity: {self._layer_maturity:.6f}\n<|turn|>")

    def ingest_text(self, text, online=False):
        ids = self.tokenizer.encode(text)
        if not len(ids):
            return
        source_bytes = len(str(text).encode("utf-8"))
        if online:
            # Fixed-shape online blocks avoid both one backward pass per token
            # and unbounded MPS graph-shape caching across arbitrary prompts.
            committed_bytes = 0
            for start in range(0, len(ids), ONLINE_TRAIN_BLOCK):
                end = min(len(ids), start + ONLINE_TRAIN_BLOCK)
                target_bytes = round(source_bytes * end / len(ids))
                token_bytes = target_bytes - committed_bytes
                self._train_tokens(
                    ids[start:end], token_bytes, stride=1,
                    pad_to=ONLINE_TRAIN_BLOCK)
                committed_bytes = target_bytes
        else:
            self.bank.process_block_select(
                ids, self.net.embedding, rows=np.empty(0, dtype=np.int64))
            self.error_bank.elapse(len(ids))
            self.tokens_seen += len(ids)
            self.bytes_seen += source_bytes

    def ingest_self_state(self, source="generation", online=False):
        """Expose telemetry to the trace bank without teaching its text."""
        self.ingest_text(self._state_prelude(source), online=False)

    def generate_ids(self, length=200, temperature=1.0, prelude=False,
                     source="generation", max_tokens=None, close_turn=True):
        if prelude:
            self.ingest_self_state(source, online=False)
        was_training = self.net.training
        self.net.eval()
        generated = []
        consumed = []
        turn_id = self.tokenizer.control_ids["turn"]
        projected = self.bank.projected_state(self.net.embedding)
        token_limit = (
            max(0, int(max_tokens)) if max_tokens is not None
            else max(0, int(length)))
        try:
            for _ in range(token_limit):
                with torch.inference_mode():
                    token_id = self._sample_token(temperature, projected)
                    projected = self.bank.advance_projected(
                        projected, token_id, self.net.embedding)
                consumed.append(token_id)
                if token_id == turn_id:
                    break
                generated.append(token_id)
        finally:
            if close_turn and (not consumed or consumed[-1] != turn_id):
                consumed.append(turn_id)
            if consumed:
                self.bank._advance_exact(np.asarray(consumed, dtype=np.int64))
                self.error_bank.elapse(len(consumed))
            if was_training:
                self.net.train()
        return generated

    def _sample_token(self, temperature, projected=None):
        if projected is None:
            features = self.bank.tap(self.net.embedding)
        else:
            features = self.bank.tap_projected(projected)
        features = features.unsqueeze(0)
        logits = self.net(features).squeeze(0)
        logits[list(self.tokenizer.generation_forbidden_ids)] = -1e9
        if float(temperature) <= 0:
            return int(logits.argmax().item())
        probabilities = F.softmax(logits / float(temperature), dim=0)
        return int(torch.multinomial(probabilities, 1).item())

    def generate_text(self, length=200, temperature=1.0, prelude=False,
                      source="generation"):
        return self.tokenizer.decode(self.generate_ids(
            length, temperature, prelude=prelude, source=source))

    def generate(self, length=200, temperature=1.0, prelude=False,
                 prelude_source="generation", close_turn=True):
        if prelude:
            self.ingest_self_state(prelude_source, online=False)
        was_training = self.net.training
        self.net.eval()
        consumed = []
        token_limit = max(0, int(length))
        decoder = self.tokenizer.incremental_decoder()
        turn_id = self.tokenizer.control_ids["turn"]
        projected = self.bank.projected_state(self.net.embedding)
        try:
            for _ in range(token_limit):
                with torch.inference_mode():
                    token_id = self._sample_token(temperature, projected)
                    projected = self.bank.advance_projected(
                        projected, token_id, self.net.embedding)
                consumed.append(token_id)
                if token_id == turn_id:
                    break
                delta = decoder.push(token_id)
                if delta:
                    yield delta
            tail = decoder.finish()
            if tail:
                yield tail
        finally:
            if close_turn and (not consumed or consumed[-1] != turn_id):
                consumed.append(turn_id)
            if consumed:
                self.bank._advance_exact(np.asarray(consumed, dtype=np.int64))
                self.error_bank.elapse(len(consumed))
            if was_training:
                self.net.train()

    def ingest_prompt(self, text, online=False, mark_turn=True):
        # Protocol establishes role; only the user's external language is an
        # online target. Soma's subsequent sample remains context, not truth.
        self.ingest_text("<|user|>", online=False)
        self.ingest_text(text, online=online)
        if mark_turn:
            self.ingest_text("<|turn|><|soma|>", online=False)

    def save(self, path):
        if not path:
            return
        path = _resolve_path(path, "checkpoint")
        revision_id = uuid.uuid4().hex
        next_history = (self.checkpoint_history + [revision_id])[-256:]
        checkpoint = {
            "species": self.species,
            "runtime_version": "v12.3.17",
            "architecture": "token spectral residual mlp",
            "checkpoint_id": self.checkpoint_id,
            "revision_id": revision_id,
            "error_bank_kind": "translated_token_residual_v1",
            "tokenizer_name": TOKENIZER_NAME,
            "tokenizer_sha256": self.tokenizer.model_sha256,
            "vocab_size": self.vocab_size,
            "token_dim": self.token_dim,
            "n_bands": self.n_bands,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "base": self.base,
            "batch_size": self.batch_size,
            "description": self.description,
            "lr": self.lr,
            "lr_base": self.lr_base,
            "lr_auto": self.lr_auto,
            "grad_clip": self.grad_clip,
            "auto_mode": self.auto_mode,
            "decimation_range": self.decimation_range,
            "decimation_gain": self.decimation_range,
            "decimation_stride_cap": self.decimation_stride_cap,
            "decimation_band": self.decimation_band,
            "bytes_per_token": self._bytes_per_token,
            "byte_stride": self._byte_stride,
            "row_norm": self.row_norm,
            "row_norm_mult": self.row_norm_mult,
            "row_norm_every": self.row_norm_every,
            "row_compression_kind": "radial_headroom_v1",
            "bytes_seen": self.bytes_seen,
            "tokens_seen": self.tokens_seen,
            "train_steps": self._train_steps,
            "io2_plasticity": self._io2_plasticity,
            "spectral_target_band": self._spectral_target_band,
            "spectral_drive": self._spectral_drive,
            "input_spectral_drive": self._input_spectral_drive,
            "input_coherence": self._input_coherence,
            "output_coherence": self._output_coherence,
            "error_energy": self._error_energy,
            "error_coherence": self._error_coherence,
            "error_concentration": self._error_concentration,
            "layer_maturity": self._layer_maturity,
            "layer_maturity_target": self._layer_maturity_target,
            "layer_competence": self._layer_competence,
            "layer_lr_multipliers": self._layer_lr_multipliers,
            "token_traces": self.bank.traces,
            "error_trace": self.error_bank.trace,
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "checkpoint_history": next_history,
        }
        temporary = f"{path}.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
        self.checkpoint_history = next_history
        meta_path = Path(f"{path}.meta.json")
        try:
            existing_metadata = json.loads(
                meta_path.read_text(encoding="utf-8"))
            if not isinstance(existing_metadata, dict):
                existing_metadata = {}
        except (OSError, json.JSONDecodeError, TypeError):
            existing_metadata = {}
        profile = existing_metadata.get("profile")
        if not isinstance(profile, dict):
            profile = {
                "name": Path(path).stem,
                "tagline": "local continual language model",
                "art": (
                    "███  ███  █▄█  ███\n"
                    "█▄▄  █ █  █ █  █▄█\n"
                    "▄▄█  ███  █ █  █ █"),
                "about": "",
            }
        metadata = {
            key: checkpoint[key] for key in (
                "species", "runtime_version", "architecture",
                "checkpoint_id", "revision_id",
                "error_bank_kind",
                "tokenizer_name", "tokenizer_sha256", "vocab_size",
                "token_dim", "n_bands", "hidden_dim", "depth", "base",
                "batch_size", "description", "bytes_seen", "tokens_seen",
                "auto_mode", "lr", "lr_base", "grad_clip",
                "decimation_range", "decimation_gain", "decimation_band",
                "bytes_per_token", "byte_stride")
        }
        metadata.update({
            "sidecar_version": 2,
            "profile": profile,
            "params": self.params(),
            "row_clip_fraction": self._row_clip_fraction,
            "row_compression_fraction": self._row_compression_fraction,
            "layer_maturity": self._layer_maturity,
            "saved_at": time.time(),
        })
        meta_temp = Path(f"{meta_path}.tmp")
        meta_temp.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        os.replace(meta_temp, meta_path)
        print(f"  saved {path}")

    def load(self, path):
        checkpoint = torch.load(
            _resolve_path(path, "checkpoint"), map_location="cpu",
            weights_only=False)
        if checkpoint.get("species") != self.species:
            raise ValueError("this is not a soma v12.3 token checkpoint")
        if checkpoint.get("tokenizer_sha256") != self.tokenizer.model_sha256:
            raise ValueError("checkpoint tokenizer does not match soma-8k-v1")
        self.n_bands = int(checkpoint["n_bands"])
        self.hidden_dim = int(checkpoint["hidden_dim"])
        self.depth = int(checkpoint["depth"])
        self.checkpoint_id = str(
            checkpoint.get("checkpoint_id") or uuid.uuid4().hex)
        self.base = float(checkpoint["base"])
        self.batch_size = int(checkpoint.get("batch_size", self.batch_size))
        self.description = str(checkpoint.get("description", ""))
        self.lr = float(checkpoint.get("lr", self.lr))
        self.lr_base = float(checkpoint.get("lr_base", self.lr_base))
        self.lr_auto = bool(checkpoint.get("lr_auto", self.lr_auto))
        self.grad_clip = float(checkpoint.get("grad_clip", self.grad_clip))
        self.auto_mode = DEFAULT_AUTO_MODE
        self.decimation_range = max(0.0, float(checkpoint.get(
            "decimation_gain",
            checkpoint.get("decimation_range", self.decimation_range))))
        self.decimation_stride_cap = int(checkpoint.get(
            "decimation_stride_cap", self.decimation_stride_cap))
        self.decimation_band = float(checkpoint.get("decimation_band", 0.0))
        self.row_norm = str(checkpoint.get("row_norm", self.row_norm))
        self.row_norm_mult = float(checkpoint.get(
            "row_norm_mult", self.row_norm_mult))
        self.row_norm_every = int(checkpoint.get(
            "row_norm_every", self.row_norm_every))
        self.net = TokenResidualNet(
            self.vocab_size, self.token_dim, self.n_bands,
            self.hidden_dim, self.depth).to(self.device)
        self.net.load_state_dict(checkpoint["net"])
        self._reset_row_norm_reference()
        self.bank = ProjectedTokenBank(
            self.vocab_size, self.token_dim, self.n_bands,
            self.base, self.device)
        self.bank.traces.copy_(checkpoint["token_traces"].double())
        self.error_bank = TranslatedErrorBank(
            self.n_bands, self.token_dim, self.base)
        saved_error_trace = checkpoint.get("error_trace")
        if (torch.is_tensor(saved_error_trace)
                and tuple(saved_error_trace.shape)
                == tuple(self.error_bank.trace.shape)):
            self.error_bank.trace.copy_(saved_error_trace.double())
        self.opt = self._build_optimizer()
        self.opt.load_state_dict(checkpoint["opt"])
        for state in self.opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)
        self.bytes_seen = int(checkpoint.get("bytes_seen", 0))
        self.tokens_seen = int(checkpoint.get("tokens_seen", 0))
        self._bytes_per_token = float(checkpoint.get(
            "bytes_per_token", self._bytes_per_token))
        self._train_steps = int(checkpoint.get("train_steps", 0))
        self._io2_plasticity = float(checkpoint.get("io2_plasticity", 1.0))
        self._spectral_target_band = float(checkpoint.get(
            "spectral_target_band", 0.0))
        self._spectral_drive = torch.as_tensor(checkpoint.get(
            "spectral_drive", torch.ones(self.n_bands))).float()
        self._input_spectral_drive = torch.as_tensor(checkpoint.get(
            "input_spectral_drive", torch.ones(self.n_bands))).float()
        self._input_coherence = float(checkpoint.get("input_coherence", 0.0))
        self._output_coherence = float(checkpoint.get("output_coherence", 0.0))
        self._error_energy = float(checkpoint.get("error_energy", 0.0))
        self._error_coherence = float(checkpoint.get("error_coherence", 0.0))
        self._error_concentration = float(checkpoint.get(
            "error_concentration", 0.0))
        self._layer_maturity = float(checkpoint.get("layer_maturity", 0.0))
        self._layer_maturity_target = float(checkpoint.get(
            "layer_maturity_target", 0.0))
        self._layer_competence = float(checkpoint.get("layer_competence", 0.0))
        self._layer_lr_multipliers = list(checkpoint.get(
            "layer_lr_multipliers", [1.0] * len(self.opt.param_groups)))
        self.checkpoint_history = list(checkpoint.get("checkpoint_history", []))
        self._update_decimation()
        self._set_optimizer_lrs()
        return self


SOMA = TokenSOMA


def build_parser():
    parser = argparse.ArgumentParser(description="soma v12.3.17 token runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("corpus")
    train.add_argument("--load", default="")
    train.add_argument("--save", default="v12_3.pt")
    train.add_argument("--bands", type=int, default=16)
    train.add_argument("--depth", type=int, default=4)
    train.add_argument("--base", type=float, default=PHI)
    train.add_argument("--batch", type=int, default=512)
    train.add_argument("--lr", type=float, default=0.002)
    train.add_argument(
        "--decimation", type=float, default=1.0,
        help="gain applied to io2's decimation decision (default: 1)")
    train.add_argument("--bytes", type=int, default=0)
    train.add_argument("--start-byte", type=int, default=0)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--report-every", type=int, default=1_000_000)
    train.add_argument("--save-every", type=int, default=10_000_000)
    train.add_argument("--dream-every", type=int, default=64)
    train.add_argument(
        "--dream-tokens", "--dream-length", dest="dream_length",
        default="128",
        help="maximum generated tokens per dream (default: 128)")
    train.add_argument("--temperature", type=float, default=1.0)
    train.add_argument("--device", default="auto")
    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("corpus")
    evaluate.add_argument("--bytes", type=int, default=200_000)
    evaluate.add_argument("--device", default="auto")
    generate = subparsers.add_parser("generate")
    generate.add_argument("checkpoint")
    generate.add_argument(
        "--length", type=int, default=200,
        help="maximum generated tokens")
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--prompt", default="")
    generate.add_argument("--device", default="auto")
    chat = subparsers.add_parser("chat")
    chat.add_argument("checkpoint")
    chat.add_argument(
        "--length", type=int, default=200,
        help="maximum generated tokens per reply")
    chat.add_argument("--temperature", type=float, default=0.8)
    chat.add_argument("--online", action="store_true")
    chat.add_argument("--save", action="store_true")
    chat.add_argument("--device", default="auto")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "train":
        model = TokenSOMA(
            n_bands=args.bands, depth=args.depth,
            base=args.base, batch_size=args.batch, lr=args.lr,
            lr_base=args.lr, auto_mode=DEFAULT_AUTO_MODE,
            decimation_range=args.decimation, device=args.device)
        if args.load:
            model.load(args.load)
            model.batch_size = args.batch
            model.decimation_range = max(0.0, float(args.decimation))
            model._update_decimation()
        model.train(
            args.corpus, epochs=args.epochs, save_path=args.save,
            start_byte=args.start_byte,
            max_bytes=args.bytes, report_every=args.report_every,
            save_every=args.save_every,
            dream_every_batches=args.dream_every,
            dream_length=args.dream_length,
            dream_temperature=args.temperature)
    elif args.command == "eval":
        model = TokenSOMA(device=args.device).load(args.checkpoint)
        model.evaluate(args.corpus, max_bytes=args.bytes)
    elif args.command == "generate":
        model = TokenSOMA(device=args.device).load(args.checkpoint)
        if args.prompt:
            model.ingest_prompt(args.prompt, online=False)
        print(model.generate_text(args.length, args.temperature))
    else:
        model = TokenSOMA(device=args.device).load(args.checkpoint)
        print("soma v12.3.17 chat. ctrl-c to quit.")
        try:
            while True:
                try:
                    prompt = input("\nyou › ")
                except EOFError:
                    break
                model.ingest_prompt(prompt, online=args.online)
                print("soma › ", end="", flush=True)
                for text in model.generate(
                        args.length, args.temperature,
                        prelude_source="chat"):
                    print(text, end="", flush=True)
                print()
        except KeyboardInterrupt:
            print()
        if args.save:
            model.save(args.checkpoint)


if __name__ == "__main__":
    main()
