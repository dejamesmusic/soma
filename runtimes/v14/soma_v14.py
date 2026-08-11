"""soma v14: adaptive temporal-lattice runtime."""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import math
import os
import shutil
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from trace_branch_mixer import (
    CHECKPOINT_VERSION,
    PHI,
    ReferenceTokenizer,
    TokenCorpus,
    TraceBranchConfig,
    TraceBranchTrainer,
)
from soma_profiles import (
    DEFAULT_CAPACITY,
    capacity_values,
    default_batch_size,
    infer_capacity,
)


RUNTIME_VERSION = "14.0.0-dev.2"
RUNTIME_ID = "soma.v14"
CHECKPOINT_FORMAT = "soma-v14-pytorch"
LOGOS_ECOSYSTEM_SCHEMA = "logos.checkpoint.v1"
TEXT_PROTOCOL_ID = "soma-text-turn-v1"
TOKENIZER_NAME = "soma-1k-v1"
DEFAULT_BATCH_SIZE = 1024
LEARNING_RATE_REFERENCE_BATCH = 1024
MAX_SOURCE_BLOCK_TOKENS = 8192
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_GRAD_CLIP = 1.0
SOURCE_SIGNATURE_BYTES = 64 * 1024
CHECKPOINT_DISK_RESERVE_BYTES = 2 * 1024**3


def _checkpoint_tensor_bytes(value, seen=None):
    """estimate serialized tensor payload without counting shared objects twice."""

    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, dict):
        return sum(_checkpoint_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_checkpoint_tensor_bytes(item, seen) for item in value)
    return 0


def _format_disk_bytes(value):
    return f"{float(value) / 1024**3:.1f} gb"


def _require_atomic_save_space(path, checkpoint):
    path = Path(path)
    tensor_bytes = _checkpoint_tensor_bytes(checkpoint)
    existing_bytes = path.stat().st_size if path.exists() else 0
    checkpoint_bytes = max(existing_bytes, math.ceil(tensor_bytes * 1.1))
    reserve = max(CHECKPOINT_DISK_RESERVE_BYTES, checkpoint_bytes // 2)
    required = checkpoint_bytes + reserve
    free = shutil.disk_usage(path.parent).free
    if free < required:
        raise OSError(
            "not enough free disk space for an atomic checkpoint save: "
            f"about {_format_disk_bytes(required)} required, "
            f"{_format_disk_bytes(free)} available"
        )


def _hash_file_range(path: Path, start: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(int(start))
        remaining = max(0, int(length))
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    if remaining:
        raise ValueError("corpus is shorter than its checkpoint boundary")
    return digest.hexdigest()


def _sign_corpus_state(state: dict) -> dict:
    state = dict(state)
    path = Path(state["path"]).expanduser().resolve()
    file_position = int(state["file_position"])
    file_size = path.stat().st_size
    if file_size < file_position:
        raise ValueError("corpus is shorter than its checkpoint boundary")
    prefix_length = min(SOURCE_SIGNATURE_BYTES, int(state["file_size"]))
    window_start = max(0, file_position - SOURCE_SIGNATURE_BYTES)
    state["signature"] = {
        "version": 1,
        "prefix_length": prefix_length,
        "prefix_sha256": _hash_file_range(path, 0, prefix_length),
        "window_start": window_start,
        "window_end": file_position,
        "window_sha256": _hash_file_range(
            path, window_start, file_position - window_start
        ),
    }
    return state


def _corpus_state_matches(path: str | Path, state: dict) -> bool:
    signature = state.get("signature", {})
    if not isinstance(signature, dict) or signature.get("version") != 1:
        return False
    path = Path(path).expanduser().resolve()
    try:
        if path.stat().st_size < int(state["file_position"]):
            return False
        prefix_length = int(signature["prefix_length"])
        window_start = int(signature["window_start"])
        window_end = int(signature["window_end"])
        return (
            _hash_file_range(path, 0, prefix_length)
            == signature["prefix_sha256"]
            and _hash_file_range(
                path, window_start, window_end - window_start
            ) == signature["window_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _revision_path(path: Path) -> Path:
    return Path(f"{path}.revision.json")


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = Path(f"{path}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json_dict(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _runtime_root() -> Path:
    import sys

    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def _tokenizer_path() -> Path:
    override = os.environ.get("SOMA_TOKENIZER", "")
    if override:
        return Path(override).expanduser()
    return _runtime_root() / "tokenizer" / f"{TOKENIZER_NAME}.model"


def _resolve_path(path, kind: str):
    if not path:
        return path
    value = os.path.expanduser(str(path))
    if os.path.isabs(value) or "/" in value or "\\" in value:
        return value
    home = os.environ.get("SOMA_HOME", "")
    root = Path(home) / ("data" if kind == "corpus" else "checkpoints")
    root.mkdir(parents=True, exist_ok=True)
    return str(root / value)


def _parse_auto_or_float(text, default_base=DEFAULT_LEARNING_RATE):
    value = str(text).strip().lower()
    if value.startswith("auto"):
        for part in reversed(value.split()[1:]):
            try:
                return float(part), True, float(part)
            except ValueError:
                continue
        return float(default_base), True, float(default_base)
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default_base)
    return number, False, number


def _legacy_profile_values(profile: str) -> dict:
    if profile == "small":
        return {
            "symbol_dim": 128,
            "scale_rank": 32,
            "scale_depth": 2,
            "summaries": 4,
            "model_dim": 512,
            "trunk_depth": 3,
            "expansion_dim": 1024,
        }
    return {
        "symbol_dim": 256,
        "scale_rank": 64,
        "scale_depth": 3,
        "summaries": 8,
        "model_dim": 1024,
        "trunk_depth": 4,
        "expansion_dim": 2048,
    }


class SOMA:
    """continual application runtime around one v14 learner."""

    species = "soma_v14_adaptive_trace_branch_mixer"

    def __init__(
        self,
        n_bands: int = 48,
        base: float = PHI,
        lr: float = DEFAULT_LEARNING_RATE,
        lr_base: float | None = None,
        lr_auto: bool = True,
        grad_clip: float = DEFAULT_GRAD_CLIP,
        batch_size: int | None = None,
        description: str = "",
        device: str = "auto",
        tokenizer_path=None,
        profile: str | None = None,
        capacity: str | None = None,
        update_interval: int = 1,
        decimation_gain: float = 1.0,
        decimation_stride_cap: int = 256,
        lattice_rate: float = 0.0,
        **_legacy,
    ) -> None:
        self.tokenizer = ReferenceTokenizer(tokenizer_path or _tokenizer_path())
        canonical_lr = float(lr_base if lr_base is not None else lr)
        if not math.isfinite(canonical_lr) or canonical_lr < 0.0:
            raise ValueError("learning rate must be finite and nonnegative")
        if (
            isinstance(n_bands, bool)
            or not math.isfinite(float(n_bands))
            or int(n_bands) != float(n_bands)
            or int(n_bands) < 1
        ):
            raise ValueError("bands must be a positive integer")
        n_bands = int(n_bands)
        if capacity is None and profile == "small":
            values = _legacy_profile_values("small")
            self.capacity = "custom"
            if batch_size is None:
                batch_size = DEFAULT_BATCH_SIZE
        else:
            self.capacity = str(capacity or DEFAULT_CAPACITY).lower()
            values = capacity_values(self.capacity, n_bands)
            if batch_size is None:
                batch_size = default_batch_size(self.capacity)
        if (
            isinstance(batch_size, bool)
            or not math.isfinite(float(batch_size))
            or int(batch_size) != float(batch_size)
            or int(batch_size) < 1
        ):
            raise ValueError("batch size must be a positive integer")
        batch_size = int(batch_size)
        config = TraceBranchConfig(
            vocab_size=self.tokenizer.vocab_size,
            bands=int(n_bands),
            phi=float(base),
            **values,
        )
        self.trainer = TraceBranchTrainer(
            config,
            self.tokenizer,
            device=device,
            learning_rate=(
                canonical_lr
                * batch_size
                / LEARNING_RATE_REFERENCE_BATCH
            ),
            grad_clip=float(grad_clip),
            update_interval=update_interval,
            lr_auto=lr_auto,
            decimation_gain=decimation_gain,
            decimation_stride_cap=decimation_stride_cap,
            pulse_enabled=True,
            pulse_quantum=max(1, int(batch_size)),
            pulse_schedule="spectral",
            optimizer_quantum=max(1, int(batch_size)),
            lattice_rate=float(lattice_rate),
        )
        self.description = str(description or "")
        self.trainer.description = self.description
        self.lr_base = canonical_lr
        self.lr_auto = bool(lr_auto)
        self.lr = self.trainer.learning_rate
        self.batch_size = batch_size
        self.grad_clip = float(grad_clip)
        self._bytes_per_token = 1.0
        self._pulse_duty = 1.0
        self._dream_input_tokens = 0
        self._dream_output_tokens = 0
        self._dream_rehearsal_ratio = 0.0
        self._source_cursor: dict | None = None
        self._loaded_checkpoint_path: Path | None = None
        self._last_logits = None
        self._last_diagnostics_at = 0.0
        self._reset_diagnostics()

    @classmethod
    def _from_trainer(cls, trainer: TraceBranchTrainer, batch_size: int):
        self = cls.__new__(cls)
        self.tokenizer = trainer.tokenizer
        self.trainer = trainer
        self.capacity = infer_capacity(trainer.config)
        self.description = trainer.description
        self._batch_size = max(1, int(batch_size))
        self.trainer.pulse_quantum = self._batch_size
        if self.trainer.optimizer_quantum <= 0.0:
            self.trainer.optimizer_quantum = float(self._batch_size)
        self.lr = trainer.learning_rate
        self.lr_base = trainer.learning_rate_base
        self.lr_auto = trainer.lr_auto
        self.grad_clip = trainer.grad_clip
        self._bytes_per_token = 1.0
        self._pulse_duty = trainer.updates / max(1, trainer.batches)
        self._dream_input_tokens = 0
        self._dream_output_tokens = 0
        self._dream_rehearsal_ratio = 0.0
        self._source_cursor = None
        self._loaded_checkpoint_path = None
        self._last_logits = None
        self._last_diagnostics_at = 0.0
        self._reset_diagnostics()
        return self

    @property
    def config(self) -> TraceBranchConfig:
        return self.trainer.config

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if (
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or int(value) != float(value)
            or int(value) < 1
        ):
            raise ValueError("batch size must be a positive integer")
        new_size = int(value)
        self._batch_size = new_size
        if hasattr(self, "trainer"):
            self.trainer.request_training_quantum(self._batch_size)
        if all(hasattr(self, name) for name in ("trainer", "lr_base", "lr_auto")):
            self._set_optimizer_lrs()

    @property
    def learning_rate_batch_scale(self) -> float:
        return self.batch_size / LEARNING_RATE_REFERENCE_BATCH

    @property
    def net(self):
        return self.trainer.model

    @property
    def device(self):
        return self.trainer.device

    @property
    def n_bands(self) -> int:
        return self.config.bands

    @property
    def base(self) -> float:
        return self.config.phi

    @property
    def bytes_seen(self) -> int:
        return int(round(self.trainer.trace.source_bytes_seen))

    @property
    def tokens_seen(self) -> int:
        return self.trainer.trace.tokens_seen

    @property
    def checkpoint_id(self) -> str:
        return self.trainer.checkpoint_id

    @property
    def model_id(self) -> str:
        return self.trainer.checkpoint_id

    @property
    def revision_id(self) -> str:
        return self.trainer.revision_id

    @property
    def parent_revision_id(self) -> str | None:
        return self.trainer.parent_revision_id

    @property
    def root_lineage_id(self) -> str:
        return self.trainer.root_lineage_id

    @property
    def token_dim(self) -> int:
        return self.config.symbol_dim

    @property
    def aperture_width(self) -> int:
        return self.config.scale_rank

    @property
    def language_depth(self) -> int:
        return self.config.trunk_depth

    def params(self) -> int:
        return self.trainer.model.parameter_count

    def active_params(self) -> int:
        return self.params()

    def _set_optimizer_lrs(self) -> None:
        scaled_base = float(self.lr_base) * self.learning_rate_batch_scale
        self.trainer.learning_rate_base = scaled_base
        self.trainer.lr_auto = bool(self.lr_auto)
        if self.lr_auto:
            self.trainer.learning_rate = (
                scaled_base * self.trainer.plasticity_amplitude()
            )
        else:
            self.trainer.learning_rate = scaled_base
        self.lr = self.trainer.learning_rate
        self.trainer._set_optimizer_learning_rate()

    def _reset_diagnostics(self) -> None:
        zeros = torch.zeros(self.config.bands, dtype=torch.float32)
        self._input_spectral_drive = zeros.clone()
        self._weight_spectral_drive = zeros.clone()
        self._spectral_drive = zeros.clone()
        self._spectral_target_band = 0.0
        self._input_coherence = 0.0
        self._output_coherence = 0.0
        self._error_energy = 0.0
        self._radial_pressure = 0.0
        self._row_spread = 1.0
        self._update_duty = 1.0
        self._pulse_frequency = self.trainer._pulse_frequency()
        self._update_cost = self.trainer.update_cost_fraction
        self._optimizer_updates = self.trainer.updates
        self._io2_plasticity = self.trainer.io2_plasticity
        self._stride = self.trainer.prediction_stride
        self._byte_stride = float(self._stride)
        self._active_memory = []
        self._top_predictions = []

    @torch.no_grad()
    def _update_diagnostics(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_diagnostics_at < 5.0:
            return
        self._last_diagnostics_at = now
        trace = self.trainer.trace.trace
        prior = self.trainer.trace.prior()
        centered = trace - prior.unsqueeze(1)
        input_field = torch.empty_like(centered)
        input_field[:, :-1] = centered[:, :-1] - centered[:, 1:]
        input_field[:, -1] = centered[:, -1]
        input_energy = input_field.square().mean(dim=0).sqrt().float()
        context_weight = self.net.context.weight.detach()
        temporal_width = self.config.summaries * self.config.symbol_dim
        temporal = context_weight[:, :temporal_width].reshape(
            self.config.model_dim,
            self.config.summaries,
            self.config.symbol_dim,
        )
        summary_allocation = temporal.float().square().mean(
            dim=(0, 2)
        ).sqrt()
        pool = self.net.scale_encoder.pool_weights.to(
            device=summary_allocation.device,
            dtype=summary_allocation.dtype,
        )
        allocation = summary_allocation @ pool
        error = self.trainer.error_spectrum.float()

        def relative_energy(value):
            total = float(value.sum()) if value.numel() else 0.0
            if total <= 0.0:
                return value
            return value * self.config.bands / total

        self._input_spectral_drive = relative_energy(input_energy).cpu()
        self._weight_spectral_drive = relative_energy(allocation).cpu()
        self._spectral_drive = error.cpu()
        self._spectral_target_band = self.trainer.error_target_band
        self._error_energy = self.trainer.error_energy
        self._io2_plasticity = self.trainer.io2_plasticity
        self._stride = self.trainer.prediction_stride
        self._pulse_duty = (
            self.trainer.updates / max(1, self.trainer.batches)
        )
        self._pulse_frequency = self.trainer._pulse_frequency()
        self._update_cost = self.trainer.update_cost_fraction
        self._update_duty = self._pulse_duty
        self._optimizer_updates = self.trainer.updates
        self._byte_stride = self._stride * self._bytes_per_token
        weight_state = self.trainer.weight_diagnostics()
        self._radial_pressure = weight_state["radial_pressure"]
        self._row_spread = weight_state["row_spread"]

        input_total = float(input_energy.sum())
        if input_total > 1e-12:
            input_probability = (input_energy / input_total).clamp_min(1e-12)
            input_entropy = float(
                -(input_probability * input_probability.log()).sum()
            )
            self._input_coherence = max(
                0.0,
                1.0 - input_entropy / math.log(max(2, self.config.bands)),
            )
        else:
            self._input_coherence = 0.0
        logits = self.trainer.next_logits()
        probabilities = F.softmax(logits, dim=-1).detach().cpu()
        output_entropy = float(
            -(probabilities.clamp_min(1e-12).log() * probabilities).sum()
        )
        self._output_coherence = max(
            0.0, 1.0 - output_entropy / math.log(self.config.vocab_size)
        )
        top = probabilities.topk(min(6, self.config.vocab_size))
        self._top_predictions = [
            {
                "token": self.tokenizer.diagnostic_piece(int(token_id)),
                "probability": float(probability),
            }
            for probability, token_id in zip(top.values, top.indices)
        ]
        memory_score = centered[:, -1].abs()
        memory = memory_score.topk(min(6, self.config.vocab_size))
        self._active_memory = [
            {
                "token": self.tokenizer.diagnostic_piece(int(token_id)),
                "energy": float(energy),
            }
            for energy, token_id in zip(memory.values, memory.indices)
        ]

    def _report(self, epoch, epochs, reader, totals, started) -> None:
        if totals["loss_bytes"] <= 0.0:
            return
        bpb = totals["nll"] / (totals["loss_bytes"] * math.log(2.0))
        elapsed = max(1e-9, time.time() - started)
        epoch_bytes = reader.bytes_consumed - getattr(
            self, "_epoch_bytes_consumed_start", 0
        )
        print(
            f"epoch {epoch + 1}/{epochs} · {bpb:.3f} bpb · "
            f"{100 * totals['correct'] / max(1, totals['samples']):.1f}% "
            f"top-1 · {epoch_bytes / elapsed:,.0f} source b/s"
        )

    @staticmethod
    def _dream_length(value, _token_loss=0.0) -> int:
        parts = str(value).strip().split()
        for part in reversed(parts):
            try:
                return max(1, int(float(part)))
            except ValueError:
                continue
        return 128

    def _record_source_cursor(
        self,
        reader: TokenCorpus,
        *,
        epoch: int,
        epochs: int,
    ) -> None:
        self._source_cursor = {
            "kind": "corpus",
            "epoch": int(epoch),
            "epochs": int(epochs),
            "reader": reader.state_dict(),
        }

    def resume_state_for(self, corpus_path) -> dict | None:
        """return a verified unfinished reader state for this checkpoint."""

        cursor = self._source_cursor
        if not isinstance(cursor, dict) or cursor.get("kind") != "corpus":
            return None
        reader_state = cursor.get("reader")
        if not isinstance(reader_state, dict):
            return None
        position = int(reader_state.get("start_byte", 0)) + int(
            reader_state.get("bytes_consumed", 0)
        )
        limit = int(reader_state.get("limit", 0))
        if position <= 0 or position >= limit:
            return None
        if _corpus_state_matches(corpus_path, reader_state):
            resumed = dict(reader_state)
            resumed["path"] = str(Path(corpus_path).expanduser().resolve())
            return resumed
        saved_path = Path(str(reader_state.get("path", ""))).expanduser()
        if saved_path and saved_path.resolve() == Path(corpus_path).expanduser().resolve():
            raise ValueError("corpus content differs from the checkpoint cursor")
        return None

    def _checkpoint_source_cursor(self) -> dict | None:
        cursor = self._source_cursor
        if not isinstance(cursor, dict):
            return None
        result = dict(cursor)
        reader_state = cursor.get("reader")
        if not isinstance(reader_state, dict):
            return result
        try:
            result["reader"] = _sign_corpus_state(reader_state)
        except (OSError, TypeError, ValueError):
            result["reader"] = dict(reader_state)
        return result

    def _source_block_tokens(self) -> int:
        """bound source delivery while allowing decimation to reduce work."""

        stride = max(1, int(self.trainer.prediction_stride))
        remaining_candidates = [
            value
            for value in (
                self.trainer.optimizer_prediction_mass_remaining(),
                self.trainer.pulse_prediction_mass_remaining(),
            )
            if value > 0.0
        ]
        remaining = min(remaining_candidates) if remaining_candidates else 0.0
        prediction_tokens = (
            max(1, int(math.ceil(remaining)))
            if remaining > 0.0
            else self.batch_size
        )
        prediction_tokens = min(prediction_tokens, self.batch_size)
        first_row = (-int(self.trainer.training_tokens_seen)) % stride
        required = first_row + (prediction_tokens - 1) * stride + 1
        return min(required, MAX_SOURCE_BLOCK_TOKENS)

    def train(
        self,
        corpus_path,
        epochs=1,
        save_every=0,
        save_path="model.pt",
        start_byte=0,
        max_bytes=0,
        report_every=1,
        dream_every_batches=0,
        dream_length=128,
        dream_temperature=1.0,
        dream_callback=None,
        resume_state=None,
    ):
        corpus_path = _resolve_path(corpus_path, "corpus")
        totals = None
        dream_pending = False
        dream_interval = max(0, int(dream_every_batches or 0))
        next_dream_update = (
            ((int(self.trainer.updates) // dream_interval) + 1)
            * dream_interval
            if dream_interval
            else None
        )

        def emit_dream() -> None:
            nonlocal dream_pending
            length = self._dream_length(dream_length)
            before = self.tokens_seen
            text = self.generate_text(
                length=length,
                temperature=float(dream_temperature),
                prelude_source="dream",
            )
            self._dream_input_tokens = int(
                self.trainer.training_tokens_seen
            )
            self._dream_output_tokens += self.tokens_seen - before
            self._dream_rehearsal_ratio = (
                self._dream_output_tokens
                / max(1, self._dream_input_tokens)
            )
            dream_pending = False
            if dream_callback:
                dream_callback(text, self.trainer.updates, self.bytes_seen)

        for epoch in range(max(1, int(epochs))):
            epoch_start = int(start_byte) if epoch == 0 else 0
            started = time.time()
            totals = {
                "nll": 0.0,
                "loss_bytes": 0.0,
                "loss_tokens": 0.0,
                "correct": 0.0,
                "top5": 0.0,
                "samples": 0.0,
                "selected_prediction_mass": 0.0,
            }
            with TokenCorpus(
                corpus_path,
                self.tokenizer,
                start_byte=epoch_start,
                max_bytes=max_bytes,
                resume_state=(resume_state if epoch == 0 else None),
            ) as reader:
                self._epoch_bytes_consumed_start = reader.bytes_consumed
                self._record_source_cursor(
                    reader, epoch=epoch, epochs=max(1, int(epochs))
                )
                while True:
                    dream_boundary_due = dream_pending
                    block = reader.next_block(
                        self._source_block_tokens(),
                        stop_at_newline=dream_pending,
                    )
                    if block is None:
                        break
                    ids, source_bytes, _position = block
                    weights = reader.last_loss_weights
                    weight_sum = float(weights.sum())
                    if weight_sum <= 0.0:
                        self.trainer.observe(
                            ids,
                            source_bytes=source_bytes,
                            update_prior=False,
                        )
                        self._record_source_cursor(
                            reader,
                            epoch=epoch,
                            epochs=max(1, int(epochs)),
                        )
                        if report_every:
                            self._report(
                                epoch, int(epochs), reader, totals, started
                            )
                        if dream_boundary_due:
                            emit_dream()
                        continue
                    metrics = self.trainer.train_batch(
                        ids,
                        source_bytes=source_bytes,
                        loss_bytes=reader.last_loss_bytes,
                        loss_weights=weights,
                    )
                    self._record_source_cursor(
                        reader,
                        epoch=epoch,
                        epochs=max(1, int(epochs)),
                    )
                    self._bytes_per_token = (
                        source_bytes / max(1, len(ids))
                    )
                    self.lr = self.trainer.learning_rate
                    self._io2_plasticity = self.trainer.io2_plasticity
                    self._stride = self.trainer.prediction_stride
                    self._pulse_duty = (
                        self.trainer.updates
                        / max(1, self.trainer.batches)
                    )
                    self._byte_stride = self._stride * self._bytes_per_token
                    totals["nll"] += metrics.token_nll
                    totals["loss_bytes"] += metrics.loss_bytes
                    totals["loss_tokens"] += weight_sum
                    totals["correct"] += metrics.top1 * weight_sum
                    totals["top5"] += metrics.top5 * weight_sum
                    totals["samples"] += weight_sum
                    totals["selected_prediction_mass"] += (
                        metrics.prediction_mass
                    )
                    self._last_logits = None

                    if report_every and self.trainer.batches % int(report_every) == 0:
                        self._report(
                            epoch, int(epochs), reader, totals, started
                        )
                    if (
                        next_dream_update is not None
                        and metrics.updated
                        and self.trainer.updates >= next_dream_update
                    ):
                        dream_pending = True
                        while next_dream_update <= self.trainer.updates:
                            next_dream_update += dream_interval
                    if (
                        dream_pending
                        and (
                            reader.last_block_ended_at_newline
                            or dream_boundary_due
                        )
                    ):
                        emit_dream()
                    if save_every and self.trainer.batches % int(save_every) == 0:
                        self.save(save_path)
        if dream_pending:
            emit_dream()
        if save_path:
            self.save(save_path)
        return totals

    def evaluate(self, corpus_path, max_bytes=0, start_byte=0):
        corpus_path = _resolve_path(corpus_path, "corpus")
        ids = []
        loss_weights = []
        source_bytes = 0
        loss_bytes = 0.0
        with TokenCorpus(
            corpus_path,
            self.tokenizer,
            start_byte=start_byte,
            max_bytes=max_bytes,
        ) as reader:
            while True:
                block = reader.next_block(self.batch_size)
                if block is None:
                    break
                chunk, chunk_bytes, _ = block
                ids.append(chunk)
                loss_weights.append(reader.last_loss_weights.copy())
                source_bytes += chunk_bytes
                loss_bytes += reader.last_loss_bytes
        if not ids:
            raise ValueError("evaluation corpus is empty")
        return self.trainer.evaluate_ids(
            np.concatenate(ids),
            source_bytes=source_bytes,
            batch_size=self.batch_size,
            loss_weights=np.concatenate(loss_weights),
            loss_bytes=loss_bytes,
        )

    def ingest_text(self, text, online=False):
        value = str(text)
        ids = self.tokenizer.encode(value)
        if not len(ids):
            return
        source_bytes = len(value.encode("utf-8"))
        if online:
            byte_per_token = source_bytes / max(1, len(ids))
            start = 0
            while start < len(ids):
                block_tokens = self._source_block_tokens()
                chunk = ids[start : start + block_tokens]
                chunk_bytes = byte_per_token * len(chunk)
                self.trainer.train_batch(
                    chunk,
                    source_bytes=chunk_bytes,
                    loss_bytes=chunk_bytes,
                )
                start += len(chunk)
        else:
            self.trainer.observe(
                ids, source_bytes=source_bytes, update_prior=True
            )

    def _observe_control(self, name: str) -> None:
        self.trainer.observe(
            [self.tokenizer.control_ids[name]], update_prior=False
        )

    def ingest_prompt(self, text, online=False, mark_turn=True):
        if mark_turn:
            self._observe_control("user")
        self.ingest_text(str(text), online=online)
        self._observe_control("soma")

    def _generation_tokens(self, length, temperature, greedy=False):
        if int(length) < 1:
            return
        if temperature <= 0.0 and not greedy:
            raise ValueError("temperature must be positive")
        turn_id = self.tokenizer.control_ids["turn"]
        forbidden = set(self.tokenizer.generation_forbidden_ids)
        prior = self.trainer.trace.prior()
        projected, direct, _ = self.net.project_state(
            self.trainer.trace.trace,
            prior,
            self.trainer.trace.last_token,
        )
        ended = False
        try:
            with torch.no_grad():
                for _ in range(int(length)):
                    logits = self.net.forward_features(
                        projected.unsqueeze(0),
                        direct.unsqueeze(0),
                        prior,
                    )[0]
                    logits[list(forbidden)] = float("-inf")
                    if greedy:
                        token_id = int(logits.argmax())
                    else:
                        probabilities = F.softmax(
                            logits / float(temperature), dim=-1
                        )
                        token_id = self.trainer._sample_token(probabilities)
                    self.trainer.observe([token_id], update_prior=False)
                    projected, direct = self.net.advance_projected(
                        projected, token_id, prior
                    )
                    if token_id == turn_id:
                        ended = True
                        break
                    yield token_id
        finally:
            if not ended:
                self.trainer.observe([turn_id], update_prior=False)

    def generate_ids(
        self,
        length=200,
        temperature=1.0,
        prelude=False,
        prelude_source="generation",
        greedy=False,
        **_unused,
    ):
        del prelude, prelude_source
        return list(self._generation_tokens(length, temperature, greedy))

    def generate_text(
        self,
        length=200,
        temperature=1.0,
        prelude=False,
        prelude_source="generation",
        greedy=False,
        **_unused,
    ):
        return self.tokenizer.decode(
            self.generate_ids(
                length=length,
                temperature=temperature,
                prelude=prelude,
                prelude_source=prelude_source,
                greedy=greedy,
            )
        )

    def generate(
        self,
        length=200,
        temperature=1.0,
        prelude=False,
        prelude_source="generation",
        **_unused,
    ):
        del prelude, prelude_source
        decoder = self.tokenizer.incremental_decoder()
        for token_id in self._generation_tokens(length, temperature):
            text = decoder.push(token_id)
            for character in text:
                yield character
        for character in decoder.finish():
            yield character

    def _checkpoint_dict(self, *, revision_id=None):
        self.trainer.description = self.description
        self._update_diagnostics(force=True)
        source_cursor = self._checkpoint_source_cursor()
        checkpoint = self.trainer.checkpoint_dict(
            metadata={
                "runtime": RUNTIME_VERSION,
                "capacity": self.capacity,
            },
            revision_id=revision_id,
        )
        checkpoint.update(
            {
                "n_bands": self.n_bands,
                "base": self.base,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "lr_base": self.lr_base,
                "lr_reference_batch": LEARNING_RATE_REFERENCE_BATCH,
                "lr_batch_scale": self.learning_rate_batch_scale,
                "lr_auto": self.lr_auto,
                "decimation_gain": self.trainer.decimation_gain,
                "lattice_rate": self.trainer.lattice_rate,
                "lattice_drift": self.trainer.lattice_drift(),
                "decimation_band": self.trainer.decimation_band,
                "prediction_stride": self.trainer.prediction_stride,
                "optimizer_quantum": self.trainer.optimizer_quantum,
                "accumulated_prediction_mass": (
                    self.trainer.accumulated_prediction_mass
                ),
                "io2_plasticity": self.trainer.io2_plasticity,
                "grad_clip": self.grad_clip,
                "description": self.description,
                "bytes_seen": self.bytes_seen,
                "tokens_seen": self.tokens_seen,
                "tokenizer_name": TOKENIZER_NAME,
                "tokenizer_vocab": self.tokenizer.vocab_size,
                "parameter_count": self.params(),
                "capacity": self.capacity,
                "weight_spectrum": self._weight_spectral_drive.tolist(),
                "source_cursor": source_cursor,
            }
        )
        return checkpoint

    def save(self, path):
        path = Path(_resolve_path(path, "checkpoint")).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(f"{path}.lock")
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            pointer_path = _revision_path(path)
            previous_pointer = _read_json_dict(pointer_path)
            if pointer_path.exists() and not previous_pointer:
                raise RuntimeError("checkpoint revision pointer is invalid")
            if previous_pointer.get("state") == "pending":
                raise RuntimeError(
                    "checkpoint has an unfinished publication transaction"
                )
            loaded_path = self._loaded_checkpoint_path
            same_artifact = loaded_path is not None and loaded_path == path
            if path.exists() and not same_artifact:
                raise FileExistsError(
                    "checkpoint already exists; load it or choose a new path"
                )
            disk_revision = previous_pointer.get("revision_id")
            if not disk_revision:
                disk_revision = _read_json_dict(
                    Path(f"{path}.meta.json")
                ).get("revision_id")
            if (
                same_artifact
                and disk_revision
                and str(disk_revision) != self.revision_id
            ):
                raise RuntimeError(
                    "checkpoint changed on disk; save as a fork instead"
                )
            revision_id = os.urandom(16).hex()
            _write_json_atomic(pointer_path, {
                "version": 1,
                "state": "pending",
                "checkpoint_id": self.checkpoint_id,
                "previous_revision_id": self.revision_id,
                "revision_id": revision_id,
            })
            published = False
            try:
                result = self._save_unlocked(path, revision_id)
                published = True
                self._loaded_checkpoint_path = path
                _write_json_atomic(pointer_path, {
                    "version": 1,
                    "state": "complete",
                    "checkpoint_id": self.checkpoint_id,
                    "revision_id": revision_id,
                })
                return result
            except Exception:
                if not published:
                    if previous_pointer:
                        _write_json_atomic(pointer_path, previous_pointer)
                    else:
                        try:
                            pointer_path.unlink()
                        except FileNotFoundError:
                            pass
                raise
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def save_fork(self, path):
        """publish the current state as a new model with an explicit parent."""

        path = Path(_resolve_path(path, "checkpoint")).expanduser().resolve()
        if path.exists():
            raise FileExistsError("fork target already exists")
        trainer = self.trainer
        previous = (
            trainer.checkpoint_id,
            trainer.parent_revision_id,
            trainer._revision_published,
            trainer.created_at,
            self._loaded_checkpoint_path,
        )
        trainer.parent_revision_id = trainer.revision_id
        trainer.checkpoint_id = os.urandom(16).hex()
        trainer._revision_published = False
        trainer.created_at = time.time()
        self._loaded_checkpoint_path = None
        try:
            return self.save(path)
        except Exception:
            (
                trainer.checkpoint_id,
                trainer.parent_revision_id,
                trainer._revision_published,
                trainer.created_at,
                self._loaded_checkpoint_path,
            ) = previous
            raise

    def _save_unlocked(self, path, revision_id):
        path = Path(_resolve_path(path, "checkpoint")).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        checkpoint = self._checkpoint_dict(revision_id=revision_id)
        _require_atomic_save_space(path, checkpoint)
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
        except Exception:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        self.trainer.parent_revision_id = checkpoint[
            "parent_revision_id"
        ]
        self.trainer.revision_id = revision_id
        self.trainer._revision_published = True
        meta_path = Path(f"{path}.meta.json")
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except (OSError, json.JSONDecodeError, TypeError):
            existing = {}
        cursor = checkpoint.get("source_cursor") or {}
        cursor_reader = (
            cursor.get("reader", {}) if isinstance(cursor, dict) else {}
        )
        ecosystem = {
            "schema": LOGOS_ECOSYSTEM_SCHEMA,
            "identity": {
                "checkpoint_id": self.checkpoint_id,
                "revision_id": self.revision_id,
                "parent_revision_id": self.parent_revision_id,
                "root_lineage_id": self.root_lineage_id,
            },
            "artifact": {
                "filename": path.name,
                "media_type": "application/x-pytorch-checkpoint",
                "bytes": path.stat().st_size,
            },
            "runtime": {
                "id": RUNTIME_ID,
                "version": RUNTIME_VERSION,
                "checkpoint_format": CHECKPOINT_FORMAT,
                "checkpoint_version": CHECKPOINT_VERSION,
            },
            "model": {
                "species": self.species,
                "parameters": self.params(),
                "modalities": ["text"],
            },
            "interface": {
                "input": "text",
                "output": "text",
                "protocol": TEXT_PROTOCOL_ID,
                "tokenizer": {
                    "id": TOKENIZER_NAME,
                    "sha256": self.tokenizer.sha256,
                    "vocabulary": self.tokenizer.vocab_size,
                },
            },
            "capabilities": [
                "inference.text",
                "training.continual",
                "training.resume",
                "learning.online",
                "state.temporal_trace",
            ],
            "experience": {
                "source_bytes": self.bytes_seen,
                "tokens": self.tokens_seen,
            },
        }
        sidecar = {
            **existing,
            "sidecar_version": 4,
            "ecosystem": ecosystem,
            "species": self.species,
            "architecture": self.species,
            "checkpoint_version": CHECKPOINT_VERSION,
            "runtime_id": RUNTIME_ID,
            "runtime_version": RUNTIME_VERSION,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "capacity": self.capacity,
            "n_bands": self.n_bands,
            "base": self.base,
            "token_dim": self.config.symbol_dim,
            "scale_rank": self.config.scale_rank,
            "scale_depth": self.config.scale_depth,
            "summaries": self.config.summaries,
            "model_dim": self.config.model_dim,
            "language_depth": self.config.trunk_depth,
            "language_expansion": self.config.expansion_dim,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "lr_base": self.lr_base,
            "lr_reference_batch": LEARNING_RATE_REFERENCE_BATCH,
            "lr_batch_scale": self.learning_rate_batch_scale,
            "lr_auto": self.lr_auto,
            "decimation_gain": self.trainer.decimation_gain,
            "lattice_rate": self.trainer.lattice_rate,
            "lattice_drift": self.trainer.lattice_drift(),
            "lattice_coordinates": self.trainer.lattice.coordinates.tolist(),
            "decimation_band": self.trainer.decimation_band,
            "prediction_stride": self.trainer.prediction_stride,
            "optimizer_quantum": self.trainer.optimizer_quantum,
            "accumulated_prediction_mass": (
                self.trainer.accumulated_prediction_mass
            ),
            "io2_plasticity": self.trainer.io2_plasticity,
            "grad_clip": self.grad_clip,
            "update_interval": self.trainer.update_interval,
            "radial_compression": self.trainer.radial_compression,
            "radial_ceiling_scale": self.trainer.radial_ceiling_scale,
            "radial_floor": self.trainer.radial_floor,
            "pulse_enabled": self.trainer.pulse_enabled,
            "pulse_quantum": self.trainer.pulse_quantum,
            "pulse_schedule": self.trainer.pulse_schedule,
            "pulse_frequency": self.trainer._pulse_frequency(),
            "update_cost_fraction": self.trainer.update_cost_fraction,
            "description": self.description,
            "bytes_seen": self.bytes_seen,
            "tokens_seen": self.tokens_seen,
            "params": self.params(),
            "checkpoint_id": self.checkpoint_id,
            "model_id": self.model_id,
            "revision_id": self.revision_id,
            "parent_revision_id": self.parent_revision_id,
            "root_lineage_id": self.root_lineage_id,
            "tokenizer": TOKENIZER_NAME,
            "tokenizer_sha256": self.tokenizer.sha256,
            "tokenizer_vocab": self.tokenizer.vocab_size,
            "weight_spectrum": self._weight_spectral_drive.tolist(),
            "radial_pressure": self._radial_pressure,
            "row_spread": self._row_spread,
            "source_name": Path(str(cursor_reader.get("path", ""))).name,
            "source_position": (
                int(cursor_reader.get("start_byte", 0))
                + int(cursor_reader.get("bytes_consumed", 0))
            ),
            "source_limit": int(cursor_reader.get("limit", 0)),
        }
        try:
            _write_json_atomic(meta_path, sidecar)
        except OSError:
            pass
        del checkpoint
        gc.collect()
        if self.device.type == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        return path

    @classmethod
    def from_checkpoint(cls, path, device="auto", tokenizer_path=None):
        path = Path(_resolve_path(path, "checkpoint")).expanduser().resolve()
        pointer_path = _revision_path(path)
        pointer = _read_json_dict(pointer_path)
        if pointer_path.exists() and not pointer:
            raise ValueError("checkpoint revision pointer is invalid")
        if pointer.get("state") == "pending":
            raise ValueError("checkpoint publication is unfinished")
        tokenizer = ReferenceTokenizer(tokenizer_path or _tokenizer_path())
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint root must be a dictionary")
        if (
            pointer.get("state") == "complete"
            and str(pointer.get("revision_id"))
            != str(checkpoint.get("revision_id"))
        ):
            raise ValueError("checkpoint revision pointer does not match")
        trainer = TraceBranchTrainer.from_checkpoint_dict(
            checkpoint, tokenizer, device=device
        )
        model = cls._from_trainer(
            trainer,
            batch_size=int(checkpoint.get("batch_size", DEFAULT_BATCH_SIZE)),
        )
        model.description = str(checkpoint.get("description", ""))
        model.trainer.description = model.description
        model.lr = float(checkpoint.get("lr", trainer.learning_rate))
        model.lr_base = float(checkpoint.get(
            "lr_base", trainer.learning_rate_base
        ))
        pre_controller = "translated_error_bank" not in checkpoint
        model.lr_auto = (
            True if pre_controller
            else bool(checkpoint.get("lr_auto", trainer.lr_auto))
        )
        model.grad_clip = float(
            checkpoint.get("grad_clip", trainer.grad_clip)
        )
        source_cursor = checkpoint.get("source_cursor")
        model._source_cursor = (
            dict(source_cursor) if isinstance(source_cursor, dict) else None
        )
        model._loaded_checkpoint_path = path.expanduser().resolve()
        model._set_optimizer_lrs()
        del checkpoint
        gc.collect()
        if model.device.type == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        return model

    def load(self, path):
        replacement = self.from_checkpoint(path, device=self.device)
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        return self


def _build_parser():
    parser = argparse.ArgumentParser(
        description="soma v14 continual language runtime"
    )
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser("train", help="train on a text corpus")
    train.add_argument("corpus")
    train.add_argument("--checkpoint")
    train.add_argument("--save", default="model.pt")
    train.add_argument("--bands", type=int)
    train.add_argument(
        "--capacity", choices=("small", "medium", "large"),
        default=None,
    )
    train.add_argument("--base", type=float)
    train.add_argument("--batch", type=int)
    train.add_argument("--lr")
    train.add_argument("--decimation", type=float)
    train.add_argument("--lattice-rate", type=float)
    train.add_argument("--grad-clip", type=float)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--start-byte", type=int)
    train.add_argument("--max-bytes", type=int, default=0)
    train.add_argument("--dream-every", type=int)
    train.add_argument("--dream-length", type=int, default=128)
    train.add_argument("--temperature", type=float, default=1.0)
    train.add_argument("--device", default="auto")

    chat = subparsers.add_parser("chat", help="talk to a checkpoint")
    chat.add_argument("checkpoint")
    chat.add_argument("--temperature", type=float, default=0.8)
    chat.add_argument("--length", type=int, default=200)
    chat.add_argument("--online", action="store_true")
    chat.add_argument("--device", default="auto")

    generate = subparsers.add_parser(
        "generate", help="sample from a checkpoint"
    )
    generate.add_argument("checkpoint")
    generate.add_argument("--length", type=int, default=200)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--device", default="auto")

    info = subparsers.add_parser("info", help="inspect a checkpoint")
    info.add_argument("checkpoint")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if args.command == "train":
        if args.lattice_rate is not None and (
            not math.isfinite(args.lattice_rate)
            or not 0.0 <= args.lattice_rate <= 1.0
        ):
            parser.error("--lattice-rate must be between 0 and 1")
        resume_state = None
        if args.checkpoint:
            model = SOMA.from_checkpoint(
                args.checkpoint, device=args.device
            )
            if args.batch is not None:
                model.batch_size = args.batch
            if args.lr is not None:
                lr_value, lr_auto, lr_base = _parse_auto_or_float(args.lr)
                model.lr = lr_value
                model.lr_base = lr_base
                model.lr_auto = lr_auto
                model._set_optimizer_lrs()
            if args.grad_clip is not None:
                model.grad_clip = args.grad_clip
                model.trainer.grad_clip = args.grad_clip
            if args.decimation is not None:
                model.trainer.decimation_gain = max(
                    0.0, float(args.decimation)
                )
            model.trainer.lattice_rate = max(
                0.0,
                float(args.lattice_rate)
                if args.lattice_rate is not None else 0.0,
            )
            if args.start_byte is None:
                resume_state = model.resume_state_for(args.corpus)
        else:
            lr_value, lr_auto, lr_base = _parse_auto_or_float(
                args.lr or "auto 0.0003"
            )
            model = SOMA(
                n_bands=args.bands if args.bands is not None else 48,
                capacity=args.capacity or DEFAULT_CAPACITY,
                base=args.base if args.base is not None else PHI,
                batch_size=args.batch,
                lr=lr_value,
                lr_base=lr_base,
                lr_auto=lr_auto,
                grad_clip=(
                    args.grad_clip
                    if args.grad_clip is not None
                    else DEFAULT_GRAD_CLIP
                ),
                decimation_gain=(
                    args.decimation if args.decimation is not None else 1.0
                ),
                lattice_rate=(
                    args.lattice_rate
                    if args.lattice_rate is not None else 0.002
                ),
                device=args.device,
            )
        print(
            f"{model.species} · {model.params() / 1e6:.2f}m params · "
            f"{model.device}"
        )
        model.train(
            args.corpus,
            epochs=args.epochs,
            save_path=args.save,
            start_byte=args.start_byte or 0,
            max_bytes=args.max_bytes,
            report_every=1,
            dream_every_batches=args.dream_every or 0,
            dream_length=args.dream_length,
            dream_temperature=args.temperature,
            dream_callback=lambda text, update, seen: print(
                f"\ndream at update {update} · {seen:,} source bytes\n> "
                + text.replace("\n", "\n> ")
            ),
            resume_state=resume_state,
        )
        return
    if args.command == "info":
        checkpoint = torch.load(
            Path(args.checkpoint).expanduser(),
            map_location="cpu",
            weights_only=True,
        )
        config = checkpoint.get("config", {})
        print(f"species       {checkpoint.get('species', 'unknown')}")
        print(f"checkpoint id {checkpoint.get('checkpoint_id', 'unknown')}")
        print(f"bands        {config.get('bands', '?')}")
        print(f"parameters   {checkpoint.get('parameter_count', '?')}")
        print(f"tokens seen  {checkpoint.get('tokens_seen', 0):,}")
        print(f"bytes seen   {checkpoint.get('bytes_seen', 0):,}")
        return

    model = SOMA.from_checkpoint(args.checkpoint, device=args.device)
    if args.command == "generate":
        print(
            model.generate_text(
                args.length, temperature=args.temperature
            )
        )
        return

    print("type /quit to leave; /save writes the lived state")
    while True:
        try:
            prompt = input("you  › ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip() == "/quit":
            break
        if prompt.strip() == "/save":
            model.save(args.checkpoint)
            print("saved")
            continue
        model.ingest_prompt(prompt, online=args.online)
        print("soma › ", end="", flush=True)
        for character in model.generate(
            args.length, temperature=args.temperature
        ):
            print(character, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
