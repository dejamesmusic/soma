"""training, generation, and checkpoint runtime for the trace branch mixer."""

from __future__ import annotations

import hashlib
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

from .core import (
    MixerOutput,
    TemporalLattice,
    TraceBranchConfig,
    TraceBranchMixer,
    TraceState,
)
from .protocol import CONTROL_NAMES, IncrementalTokenDecoder


CHECKPOINT_VERSION = 14
COMPATIBLE_CHECKPOINT_VERSIONS = (13, 14)
RUNTIME_VERSION = "14.0.0-dev.2"
MAX_PREDICTION_STRIDE = 256
COMPATIBLE_SPECIES = (
    "soma_v13_trace_branch_mixer",
    TraceBranchMixer.species,
)


def _finite_float(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and (
        number < minimum
        or (not minimum_inclusive and number == minimum)
    ):
        relation = "greater than" if not minimum_inclusive else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return number


def _positive_int(name: str, value: Any, *, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = float(value)
        integer = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a positive integer") from None
    if not math.isfinite(number) or integer != number or integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and integer > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return integer


def resolve_device(value: str | torch.device = "auto") -> torch.device:
    if isinstance(value, torch.device):
        return value
    name = str(value).strip().lower()
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite_tensors(value: Any, label: str) -> None:
    """reject non-finite numeric state before it reaches the live model."""

    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"checkpoint {label} contains non-finite tensors")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_tensors(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_tensors(item, f"{label}[{index}]")


class ReferenceTokenizer:
    """the immutable sentencepiece alphabet for one model lineage."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"missing tokenizer: {self.path}")
        self.sha256 = _sha256(self.path)
        self.processor = spm.SentencePieceProcessor(
            model_file=str(self.path)
        )
        self.vocab_size = int(self.processor.vocab_size())
        self.control_ids = {
            name: int(self.processor.piece_to_id(f"<|{name}|>"))
            for name in CONTROL_NAMES
        }
        missing = [
            name for name, token_id in self.control_ids.items()
            if (
                token_id < 0
                or self.processor.id_to_piece(token_id) != f"<|{name}|>"
            )
        ]
        if missing:
            raise ValueError(
                "tokenizer is missing control pieces: " + ", ".join(missing)
            )
        if len(set(self.control_ids.values())) != len(self.control_ids):
            raise ValueError("tokenizer control pieces must have unique ids")
        self.generation_forbidden_ids = (
            int(self.processor.unk_id()),
            self.control_ids["soma_state"],
            self.control_ids["user"],
            self.control_ids["soma"],
            self.control_ids["dream"],
        )

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(
            self.processor.encode(str(text), out_type=int),
            dtype=np.int64,
        )

    def decode(self, ids: list[int] | np.ndarray | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return self.processor.decode([int(value) for value in ids])

    def piece(self, token_id: int) -> str:
        return self.processor.id_to_piece(int(token_id))

    def diagnostic_piece(self, token_id: int) -> str:
        """render one vocabulary coordinate as a readable instrument label."""

        token_id = int(token_id)
        for name, control_id in self.control_ids.items():
            if token_id == control_id:
                return f"[{name}]"
        piece = self.piece(token_id)
        if piece == "<unk>":
            return "[unknown]"
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            try:
                value = int(piece[3:5], 16)
            except ValueError:
                return piece
            if value == 0x0A:
                return "\\n"
            if value == 0x09:
                return "\\t"
            if value == 0x0D:
                return "\\r"
            if 0x20 <= value <= 0x7E:
                return chr(value)
            return f"[byte {value:02x}]"
        if piece.startswith("▁"):
            return "_" + piece[1:]
        return piece

    def incremental_decoder(self) -> IncrementalTokenDecoder:
        return IncrementalTokenDecoder(self)


@dataclass
class BatchMetrics:
    loss: float
    token_nll: float
    bpb: float
    top1: float
    grad_norm: float
    top5: float
    tokens: int
    source_bytes: float
    loss_bytes: float
    updated: bool
    batch: int
    update: int
    predictions: int
    prediction_mass: float
    accumulated_prediction_mass: float
    optimizer_quantum: float


@dataclass
class EvaluationMetrics:
    token_nll: float
    bpb: float
    top1: float
    tokens: int
    source_bytes: float


class TranslatedErrorBank:
    """signed prediction corrections on the trace's temporal lattice."""

    def __init__(
        self,
        config: TraceBranchConfig,
        trace: torch.Tensor | None = None,
        *,
        lattice: TemporalLattice | None = None,
    ) -> None:
        self.config = config
        self.lattice = (
            lattice if lattice is not None else TemporalLattice(config)
        )
        self.refresh_lattice()
        self.dimension = int(config.symbol_dim)
        expected = (config.bands + 1, self.dimension)
        if trace is None:
            trace = torch.zeros(expected, dtype=torch.float64)
        trace = torch.as_tensor(trace, dtype=torch.float64, device="cpu")
        if trace.shape != expected:
            raise ValueError(f"translated error bank must have shape {expected}")
        if not torch.isfinite(trace).all():
            raise ValueError("translated error bank must be finite")
        self.trace = trace.clone()

    def refresh_lattice(self) -> None:
        """follow the shared lattice, extended by one slower band."""

        _, retention, _, _ = self.lattice.derive(extended=True)
        self.retention = retention

    def advance(
        self,
        values: torch.Tensor,
        gaps: torch.Tensor | np.ndarray | list[int],
    ) -> None:
        values = torch.as_tensor(
            values, dtype=torch.float64, device="cpu"
        )
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("translated errors have the wrong shape")
        gaps = torch.as_tensor(gaps, dtype=torch.float64).flatten()
        if gaps.shape != (values.shape[0],):
            raise ValueError("error gaps must match translated errors")
        if values.shape[0] == 0:
            return
        gaps = gaps.clamp_min(1.0)
        cumulative = gaps.cumsum(0)
        total = cumulative[-1]
        remaining = total - cumulative
        event_retention = torch.pow(
            self.retention.unsqueeze(0), gaps.unsqueeze(1)
        )
        coefficients = (
            (1.0 - event_retention)
            * torch.pow(
                self.retention.unsqueeze(0), remaining.unsqueeze(1)
            )
        )
        self.trace.mul_(
            torch.pow(self.retention, total).unsqueeze(1)
        ).add_(
            coefficients.transpose(0, 1) @ values
        )

    def advance_impulses(
        self,
        values: torch.Tensor,
        gaps: torch.Tensor | np.ndarray | list[int],
        importance: torch.Tensor | np.ndarray | list[float] | float = 1.0,
        band_gain: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        """advance causal sampled residuals as weighted point observations."""

        values = torch.as_tensor(
            values, dtype=torch.float64, device="cpu"
        )
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("translated errors have the wrong shape")
        gaps = torch.as_tensor(gaps, dtype=torch.float64).flatten()
        if gaps.shape != (values.shape[0],):
            raise ValueError("error gaps must match translated errors")
        if values.shape[0] == 0:
            return
        importance = torch.as_tensor(
            importance, dtype=torch.float64, device="cpu"
        ).flatten()
        if importance.numel() == 1:
            importance = importance.expand(values.shape[0])
        if importance.shape != (values.shape[0],):
            raise ValueError("importance must match translated errors")
        if not torch.isfinite(importance).all() or float(importance.min()) < 0:
            raise ValueError("importance must be finite and nonnegative")
        if band_gain is None:
            band_gain = torch.ones_like(self.retention)
        else:
            band_gain = torch.as_tensor(
                band_gain, dtype=torch.float64, device="cpu"
            ).flatten()
            if band_gain.shape != self.retention.shape:
                raise ValueError("band gain must match residual bands")
            if (
                not torch.isfinite(band_gain).all()
                or float(band_gain.min()) < 0
            ):
                raise ValueError("band gain must be finite and nonnegative")

        gaps = gaps.clamp_min(1.0)
        cumulative = gaps.cumsum(0)
        total = cumulative[-1]
        remaining = total - cumulative
        injection = 1.0 - self.retention
        coefficients = (
            injection.unsqueeze(0)
            * importance.unsqueeze(1)
            * band_gain.unsqueeze(0)
            * torch.pow(
                self.retention.unsqueeze(0), remaining.unsqueeze(1)
            )
        )
        self.trace.mul_(
            torch.pow(self.retention, total).unsqueeze(1)
        ).add_(coefficients.transpose(0, 1) @ values)

    def sampling_gain(self, stride: int) -> torch.Tensor:
        """match each sampled ema's dense stationary white-noise variance."""

        stride = max(1, int(stride))
        numerator = -torch.expm1(
            2.0 * stride * torch.log(self.retention)
        )
        denominator = -torch.expm1(2.0 * torch.log(self.retention))
        return torch.sqrt(numerator / denominator.clamp_min(1e-300))

    def bandpass_sampling_calibration(self, stride: int) -> torch.Tensor:
        """calibrate adjacent residual energy to its dense white baseline."""

        stride = max(1, int(stride))
        retention = self.retention
        injection = 1.0 - retention
        sampled_retention = torch.pow(retention, stride)
        sampled_injection = injection * self.sampling_gain(stride)

        def covariance(
            left_retention: torch.Tensor,
            right_retention: torch.Tensor,
            left_injection: torch.Tensor,
            right_injection: torch.Tensor,
        ) -> torch.Tensor:
            denominator = -torch.expm1(
                torch.log(left_retention) + torch.log(right_retention)
            )
            return (
                left_injection * right_injection
                / denominator.clamp_min(1e-300)
            )

        dense_variance = covariance(
            retention, retention, injection, injection
        )
        dense_cross = covariance(
            retention[:-1],
            retention[1:],
            injection[:-1],
            injection[1:],
        )
        dense_bandpass = (
            dense_variance[:-1]
            + dense_variance[1:]
            - 2.0 * dense_cross
        ).clamp_min(1e-300)

        sampled_variance = covariance(
            sampled_retention,
            sampled_retention,
            sampled_injection,
            sampled_injection,
        )
        sampled_cross = covariance(
            sampled_retention[:-1],
            sampled_retention[1:],
            sampled_injection[:-1],
            sampled_injection[1:],
        )
        sampled_bandpass = (
            sampled_variance[:-1]
            + sampled_variance[1:]
            - 2.0 * sampled_cross
        ).clamp_min(1e-300)
        return torch.sqrt(dense_bandpass / sampled_bandpass)

    def bandpass(self) -> torch.Tensor:
        return self.trace[:-1] - self.trace[1:]

    def energy(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.bandpass(), dim=1)


class TraceBranchTrainer:
    """one continual learner with atomic temporal-state commits."""

    def __init__(
        self,
        config: TraceBranchConfig,
        tokenizer: ReferenceTokenizer,
        *,
        device: str | torch.device = "auto",
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        update_interval: int = 1,
        lr_auto: bool = True,
        decimation_gain: float = 1.0,
        decimation_stride_cap: int = 256,
        radial_compression: bool = True,
        radial_ceiling_scale: float = 0.1,
        radial_floor: float = 0.01,
        pulse_enabled: bool = False,
        pulse_quantum: int = 1024,
        pulse_frequency_exponent: float = 0.5,
        pulse_schedule: str = "spectral",
        optimizer_quantum: float = 0.0,
        lattice_rate: float = 0.0,
        lattice_elastic: float = 0.1,
        seed: int = 7,
    ) -> None:
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError(
                "tokenizer vocabulary does not match model configuration"
            )
        learning_rate = _finite_float(
            "learning_rate", learning_rate, minimum=0.0
        )
        weight_decay = _finite_float(
            "weight_decay", weight_decay, minimum=0.0
        )
        grad_clip = _finite_float("grad_clip", grad_clip, minimum=0.0)
        decimation_gain = _finite_float(
            "decimation_gain", decimation_gain, minimum=0.0
        )
        decimation_stride_cap = _positive_int(
            "decimation_stride_cap",
            decimation_stride_cap,
            maximum=MAX_PREDICTION_STRIDE,
        )
        radial_ceiling_scale = _finite_float(
            "radial_ceiling_scale",
            radial_ceiling_scale,
            minimum=0.0,
            minimum_inclusive=False,
        )
        radial_floor = _finite_float(
            "radial_floor", radial_floor, minimum=0.0, maximum=1.0
        )
        pulse_quantum = _positive_int("pulse_quantum", pulse_quantum)
        pulse_frequency_exponent = _finite_float(
            "pulse_frequency_exponent",
            pulse_frequency_exponent,
            minimum=0.0,
            maximum=1.0,
        )
        optimizer_quantum = _finite_float(
            "optimizer_quantum", optimizer_quantum, minimum=0.0
        )
        lattice_rate = _finite_float(
            "lattice_rate", lattice_rate, minimum=0.0, maximum=1.0
        )
        lattice_elastic = _finite_float(
            "lattice_elastic", lattice_elastic, minimum=0.0, maximum=1.0
        )
        if isinstance(seed, bool) or int(seed) != seed:
            raise ValueError("seed must be an integer")
        seed = int(seed)
        if isinstance(update_interval, bool) or int(update_interval) != update_interval:
            raise ValueError("update_interval must be one")
        if not isinstance(lr_auto, bool):
            raise ValueError("lr_auto must be boolean")
        if not isinstance(radial_compression, bool):
            raise ValueError("radial_compression must be boolean")
        if not isinstance(pulse_enabled, bool):
            raise ValueError("pulse_enabled must be boolean")
        self.config = config
        self.tokenizer = tokenizer
        self.device = resolve_device(device)
        self.sampling_generator = torch.Generator(device="cpu")
        self.sampling_generator.manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = TraceBranchMixer(config).to(self.device)
        self.learning_rate = learning_rate
        self.learning_rate_base = learning_rate
        self.lr_auto = lr_auto
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        if int(update_interval) != 1:
            raise ValueError(
                "update_interval is no longer supported; use io2 pulse scheduling"
            )
        self.update_interval = 1
        self.decimation_gain = decimation_gain
        self.decimation_stride_cap = decimation_stride_cap
        self.radial_compression = radial_compression
        self.radial_ceiling_scale = radial_ceiling_scale
        self.radial_floor = radial_floor
        self.pulse_enabled = pulse_enabled
        self.pulse_quantum = pulse_quantum
        self.pulse_frequency_exponent = pulse_frequency_exponent
        self.pulse_schedule = str(pulse_schedule).strip().lower()
        if self.pulse_schedule not in {"spectral", "factor"}:
            raise ValueError("pulse schedule must be spectral or factor")
        self.pulse_phase = 0.0
        self.pulse_quantum_fill = 0.0
        self.pulse_quantum_updates: bool | None = None
        self.optimizer_quantum = optimizer_quantum
        self.pending_optimizer_quantum: float | None = None
        self.accumulated_prediction_mass = 0.0
        self.accumulated_loss_mass = 0.0
        self.accumulation_learning_rate: float | None = None
        self.controller_prediction_mass = 0.0
        self.controller_token_nll = 0.0
        self.controller_loss_token_mass = 0.0
        self.controller_loss_bytes = 0.0
        self.update_time_ema = 0.0
        self.observe_time_ema = 0.0
        self.update_cost_fraction = 0.75
        self.decimation_band = 0.0
        self.prediction_stride = 1
        self.io2_plasticity = 1.0
        self.error_energy = 0.0
        self.error_coherence = 0.0
        self.error_target_band = 0.0
        self.error_spectrum = torch.zeros(
            config.bands, dtype=torch.float64
        )
        self.last_error_token = 0
        self.error_sampling_stride = 1
        self.training_tokens_seen = 0
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # the lattice is one shared object: trace, error bank, and mixer
        # all read from it. it is controller-owned — never optimized.
        self.lattice = TemporalLattice(config)
        self.lattice_rate = lattice_rate
        self.lattice_elastic = lattice_elastic
        self._last_lattice_token = 0
        self.trace = TraceState(config, lattice=self.lattice)
        self.error_bank = TranslatedErrorBank(config, lattice=self.lattice)
        self.error_trace = torch.zeros(config.bands, dtype=torch.float64)
        self.batches = 0
        self.updates = 0
        self.predictions = 0
        self.total_nll = 0.0
        self.loss_bytes = 0.0
        self.checkpoint_id = uuid.uuid4().hex
        self.revision_id = uuid.uuid4().hex
        self.root_lineage_id = self.checkpoint_id
        self.parent_revision_id: str | None = None
        self._revision_published = False
        self.created_at = time.time()
        self.description = ""
        self.metadata: dict[str, Any] = {}

    def optimizer_prediction_mass_remaining(self) -> float:
        if self.optimizer_quantum <= 0.0:
            return 0.0
        return max(
            0.0,
            self.optimizer_quantum - self.accumulated_prediction_mass,
        )

    def pulse_prediction_mass_remaining(self) -> float:
        quantum = self._controller_quantum() or float(self.pulse_quantum)
        return max(0.0, quantum - self.pulse_quantum_fill)

    def request_training_quantum(self, quantum: int) -> None:
        """change the learning clock without reinterpreting partial evidence."""

        quantum = _positive_int("training quantum", quantum)
        current = self._controller_quantum() or float(self.pulse_quantum)
        unfinished = (
            self.accumulated_prediction_mass > 0.0
            or self.controller_prediction_mass > 0.0
            or self.pulse_quantum_fill > 0.0
            or self.pulse_quantum_updates is not None
        )
        if unfinished and abs(current - float(quantum)) > 1e-9:
            self.pending_optimizer_quantum = float(quantum)
            return
        self.optimizer_quantum = float(quantum)
        self.pulse_quantum = quantum
        self.pending_optimizer_quantum = None

    def _adopt_pending_training_quantum(self) -> None:
        """commit a deferred clock change only on a complete old boundary."""

        pending = self.pending_optimizer_quantum
        if pending is None:
            return
        if (
            self.accumulated_prediction_mass > 1e-9
            or self.controller_prediction_mass > 1e-9
            or self.pulse_quantum_fill > 1e-9
            or self.pulse_quantum_updates is not None
        ):
            return
        self.optimizer_quantum = float(pending)
        self.pulse_quantum = int(round(pending))
        self.pending_optimizer_quantum = None

    def _clear_gradient_accumulation(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulated_prediction_mass = 0.0
        self.accumulated_loss_mass = 0.0
        self.accumulation_learning_rate = None

    def _optimizer_step_from_accumulation(self) -> float:
        compute_mass = float(self.accumulated_prediction_mass)
        loss_mass = float(self.accumulated_loss_mass)
        if compute_mass <= 0.0 or loss_mass <= 0.0:
            return 0.0
        if self.accumulation_learning_rate is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = float(self.accumulation_learning_rate)
        inverse_mass = 1.0 / loss_mass
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(inverse_mass)
        previous_norms = {
            id(parameter): torch.linalg.vector_norm(
                parameter.detach().flatten(start_dim=1),
                dim=1,
                keepdim=True,
            )
            for parameter in self.model.parameters()
            if parameter.ndim >= 2 and parameter.requires_grad
        }
        grad_norm_value = 0.0
        if self.grad_clip > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            if not torch.isfinite(grad_norm):
                self._clear_gradient_accumulation()
                raise FloatingPointError("non-finite gradient norm")
            grad_norm_value = float(grad_norm.detach().cpu())
        self.optimizer.step()
        self._bound_scalar_gates_()
        self._soft_radial_compress_(previous_norms)
        self.updates += 1
        self._clear_gradient_accumulation()
        self._set_optimizer_learning_rate()
        return grad_norm_value

    def _row_ceiling(self, parameter: torch.Tensor) -> float:
        fan_in = int(parameter[0].numel())
        return max(1.0, math.sqrt(fan_in) * self.radial_ceiling_scale)

    @torch.no_grad()
    def _bound_scalar_gates_(self) -> None:
        """keep raw tanh gates finite inside their saturated regime."""

        for parameter in self.model.parameters():
            if parameter.ndim != 0 or not parameter.requires_grad:
                continue
            if not torch.isfinite(parameter):
                raise FloatingPointError("non-finite scalar gate")
            parameter.clamp_(-8.0, 8.0)

    @torch.no_grad()
    def _soft_radial_compress_(
        self,
        previous_norms: dict[int, torch.Tensor],
    ) -> None:
        """compress outward growth while preserving direction and legacy rows.

        below the ceiling, outward motion is progressively attenuated. at the
        ceiling only the radial component is removed, so tangential learning
        can continue. rows loaded above the ceiling are never snapped down,
        but may not grow farther; weight decay and inward gradients can return
        them gradually.
        """

        if not self.radial_compression:
            return
        for parameter in self.model.parameters():
            if parameter.ndim < 2 or not parameter.requires_grad:
                continue
            previous_norm = previous_norms.get(id(parameter))
            if previous_norm is None:
                continue
            current = parameter.flatten(start_dim=1)
            previous_norm = previous_norm.to(
                device=current.device, dtype=current.dtype
            )
            current_norm = torch.linalg.vector_norm(
                current, dim=1, keepdim=True
            )
            if not torch.isfinite(current_norm).all():
                raise FloatingPointError("non-finite parameter row norm")
            outward = (current_norm - previous_norm).clamp_min(0.0)
            ceiling = self._row_ceiling(parameter)
            ratio = previous_norm / ceiling
            gain = 1.0 / (1.0 + ratio.pow(4))
            gain = gain.clamp_min(self.radial_floor)
            desired_norm = current_norm - outward * (1.0 - gain)
            # new rows cannot cross the capacity boundary. checkpoint rows
            # already beyond it retain their learned magnitude, but no
            # additional outward growth, avoiding a destructive migration.
            maximum_norm = torch.where(
                previous_norm <= ceiling,
                torch.full_like(previous_norm, ceiling),
                previous_norm,
            )
            desired_norm = torch.minimum(desired_norm, maximum_norm)
            scale = desired_norm / current_norm.clamp_min(1e-12)
            current.mul_(scale)

    @torch.no_grad()
    def weight_diagnostics(self) -> dict[str, float]:
        """summarize row geometry in the coordinates used by compression."""

        pressures: list[torch.Tensor] = []
        row_norms: list[torch.Tensor] = []
        for parameter in self.model.parameters():
            if parameter.ndim < 2 or not parameter.requires_grad:
                continue
            rows = parameter.detach().flatten(start_dim=1).float()
            norms = torch.linalg.vector_norm(rows, dim=1).cpu()
            if norms.numel() == 0:
                continue
            ceiling = self._row_ceiling(parameter)
            pressure = 1.0 - 1.0 / (1.0 + (norms / ceiling).pow(4))
            pressures.append(pressure)
            row_norms.append(norms)
        if not row_norms:
            return {"radial_pressure": 0.0, "row_spread": 1.0}
        pressure = torch.cat(pressures)
        norms = torch.cat(row_norms)
        median = float(norms.median())
        p99 = float(torch.quantile(norms, 0.99))
        return {
            "radial_pressure": float(pressure.mean()),
            "row_spread": p99 / max(1e-12, median),
        }

    def _max_decimation_band(self) -> float:
        return max(
            0.0,
            math.log(self.decimation_stride_cap, self.config.phi),
        )

    def _native_decimation_band(self) -> float:
        return min(float(self.config.bands - 1), self._max_decimation_band())

    def _set_prediction_stride(self) -> None:
        maximum = self._max_decimation_band()
        self.decimation_band = max(
            0.0, min(maximum, float(self.decimation_band))
        )
        self.prediction_stride = min(
            self.decimation_stride_cap,
            max(1, int(round(self.config.phi ** self.decimation_band))),
        )

    def _set_optimizer_learning_rate(self) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(self.learning_rate)

    def _sample_token(self, probabilities: torch.Tensor) -> int:
        """sample portably from a model-owned, checkpointed cpu rng."""

        uniform = torch.rand((), generator=self.sampling_generator)
        cumulative = probabilities.cumsum(dim=-1)
        cumulative[-1] = 1.0
        return int(torch.searchsorted(
            cumulative, uniform.to(device=cumulative.device)
        ))

    def _pulse_frequency(self) -> float:
        if not self.pulse_enabled or not self.lr_auto:
            return 1.0
        ratio = max(0.0, min(1.0, self.io2_plasticity))
        if self.pulse_schedule == "factor":
            return ratio ** self.pulse_frequency_exponent
        characteristic = self.config.phi ** (-self.error_target_band)
        confidence = max(0.0, min(1.0, self.error_coherence))
        cost = max(0.0, min(1.0, self.update_cost_fraction))
        frequency = 1.0 - cost * confidence * (1.0 - characteristic)
        return max(ratio, min(1.0, frequency))

    def plasticity_amplitude(self, ratio: float | None = None) -> float:
        ratio = self.io2_plasticity if ratio is None else float(ratio)
        ratio = max(0.0, min(1.0, ratio))
        if not self.pulse_enabled:
            return ratio
        if self.pulse_schedule == "spectral":
            return ratio / max(self._pulse_frequency(), 1e-12)
        return ratio ** (1.0 - self.pulse_frequency_exponent)

    def effective_lattice_rate(self) -> float:
        """current mesh mobility in the controller's plasticity register."""

        if self.lattice_rate <= 0.0:
            return 0.0
        amplitude = self.plasticity_amplitude() if self.lr_auto else 1.0
        return self.lattice_rate * amplitude

    def _record_batch_time(self, elapsed: float, updated: bool) -> None:
        rate = 0.05
        elapsed = max(0.0, float(elapsed))
        if updated:
            if self.update_time_ema <= 0.0:
                self.update_time_ema = elapsed
            else:
                self.update_time_ema += rate * (
                    elapsed - self.update_time_ema
                )
        else:
            if self.observe_time_ema <= 0.0:
                self.observe_time_ema = elapsed
            else:
                self.observe_time_ema += rate * (
                    elapsed - self.observe_time_ema
                )
        if self.update_time_ema > 0.0 and self.observe_time_ema > 0.0:
            extra = max(0.0, self.update_time_ema - self.observe_time_ema)
            measured = extra / max(self.update_time_ema, 1e-12)
            self.update_cost_fraction += 0.05 * (
                measured - self.update_cost_fraction
            )

    def _schedule_update(self, observed_predictions: float) -> bool:
        del observed_predictions
        if self.pulse_quantum_updates is not None:
            return self.pulse_quantum_updates
        if not self.pulse_enabled or not self.lr_auto:
            decision = True
        elif self.updates == 0 and self.pulse_phase <= 1e-12:
            decision = True
        else:
            self.pulse_phase += self._pulse_frequency()
            decision = self.pulse_phase + 1e-12 >= 1.0
            if decision:
                self.pulse_phase = max(0.0, self.pulse_phase - 1.0)
        self.pulse_quantum_updates = decision
        return decision

    def _commit_pulse_mass(self, prediction_mass: float) -> None:
        quantum = self._controller_quantum() or float(self.pulse_quantum)
        self.pulse_quantum_fill += max(0.0, float(prediction_mass))
        if self.pulse_quantum_fill + 1e-9 >= quantum:
            self.pulse_quantum_fill = max(
                0.0, self.pulse_quantum_fill - quantum
            )
            self.pulse_quantum_updates = None

    def _translated_correction(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = F.softmax(logits.detach().float(), dim=-1)
        dictionary = self.model.embedding.weight.detach().float()
        expected = probabilities @ dictionary
        observed = dictionary.index_select(0, targets)
        # the readout's sqrt(d) is a numerical logit scale, not part of the
        # categorical correction. removing it keeps io2's signal invariant
        # to the chosen symbol-coordinate width.
        correction = observed - expected
        dictionary_scale = dictionary.norm(dim=1).mean().clamp_min(1e-12)
        return correction * weights.detach().float().unsqueeze(1) / dictionary_scale

    def _advance_error_bank(
        self,
        correction: torch.Tensor,
        event_rows: torch.Tensor,
        block_start: int,
        sampling_stride: int | None,
    ) -> None:
        event_times = event_rows.detach().cpu().long() + int(block_start) + 1
        if event_times.numel() == 0:
            return
        previous = int(getattr(self, "last_error_token", 0))
        gaps = torch.empty_like(event_times)
        gaps[0] = max(1, int(event_times[0]) - previous)
        if event_times.numel() > 1:
            gaps[1:] = event_times[1:] - event_times[:-1]
        stride = 1 if sampling_stride is None else max(1, int(sampling_stride))
        self.error_sampling_stride = stride
        self.error_bank.advance_impulses(
            correction.detach().cpu(),
            gaps,
            band_gain=self.error_bank.sampling_gain(stride),
        )
        self.last_error_token = int(event_times[-1])
        self._refresh_error_state()

    def _advance_error_time_to(self, token_time: int) -> None:
        """age controller state through trace time with no new residual."""

        token_time = int(token_time)
        elapsed = token_time - int(self.last_error_token)
        if elapsed < 0:
            raise ValueError("controller time cannot move backwards")
        if elapsed == 0:
            return
        self.error_bank.trace.mul_(
            torch.pow(self.error_bank.retention, elapsed).unsqueeze(1)
        )
        self.last_error_token = token_time
        self._refresh_error_state()

    def _refresh_error_state(self) -> None:
        energy = self.error_bank.energy()
        energy = energy * self.error_bank.bandpass_sampling_calibration(
            self.error_sampling_stride
        )
        total = float(energy.sum())
        self.error_trace = energy.clone()
        if total <= 1e-12:
            self.error_spectrum.zero_()
            self.error_energy = 0.0
            self.error_coherence = 0.0
            self.error_target_band = 0.0
            return
        probability = energy / total
        entropy = float(
            -(probability * probability.clamp_min(1e-12).log()).sum()
        )
        self.error_coherence = max(
            0.0,
            min(1.0, 1.0 - entropy / math.log(max(2, self.config.bands))),
        )
        coordinates = self.lattice.interval_coordinates(extended=True)
        self.error_target_band = float((coordinates * probability).sum())
        self.error_energy = total / (
            total + math.sqrt(self.config.bands)
        )
        self.error_spectrum = self.config.bands * probability

    def _io2_ratio(self, token_loss: float) -> float:
        plasticity = 2.5 * math.sqrt(self.error_energy) * (
            0.25 + 0.75 * math.sqrt(self.error_coherence)
        )
        chance = math.log(self.config.vocab_size)
        loss_scale = self.config.phi ** (
            (float(token_loss) - chance) / max(chance, 1e-12)
        )
        plasticity *= max(
            1.0 / self.config.phi,
            min(self.config.phi, loss_scale),
        )
        return max(0.02, min(1.0, plasticity))

    @staticmethod
    def _throughput_relaxation(nats_per_byte: float) -> float:
        bits_per_byte = float(nats_per_byte) / math.log(2.0)
        return max(0.0, min(1.0, 1.0 - bits_per_byte))

    def _controller_quantum(self) -> float:
        if self.optimizer_quantum > 0.0:
            return self.optimizer_quantum
        return 0.0

    def _accumulate_io2(
        self,
        *,
        token_nll: float,
        prediction_mass: float,
        loss_token_mass: float,
        loss_bytes: float,
    ) -> None:
        """apply control only at canonical supervised-mass boundaries."""

        self.controller_token_nll += float(token_nll)
        self.controller_prediction_mass += float(prediction_mass)
        self.controller_loss_token_mass += float(loss_token_mass)
        self.controller_loss_bytes += float(loss_bytes)
        quantum = self._controller_quantum()
        if (
            quantum > 0.0
            and self.controller_prediction_mass + 1e-9 < quantum
        ):
            return
        prediction_mass = self.controller_prediction_mass
        if prediction_mass <= 0.0:
            return
        token_loss = self.controller_token_nll / prediction_mass
        estimated_nll = token_loss * self.controller_loss_token_mass
        nats_per_byte = (
            estimated_nll / self.controller_loss_bytes
            if self.controller_loss_bytes > 0.0
            else token_loss
        )
        self._apply_io2(
            token_loss,
            nats_per_byte,
            elapsed_predictions=prediction_mass,
        )
        self.controller_prediction_mass = 0.0
        self.controller_token_nll = 0.0
        self.controller_loss_token_mass = 0.0
        self.controller_loss_bytes = 0.0

    def _apply_io2(
        self,
        token_loss: float,
        nats_per_byte: float,
        *,
        elapsed_predictions: float | None = None,
    ) -> None:
        ratio = self._io2_ratio(token_loss)
        self.io2_plasticity = ratio
        if self.lr_auto:
            amplitude = self.plasticity_amplitude(ratio)
            self.learning_rate = self.learning_rate_base * amplitude
            self._set_optimizer_learning_rate()
        native_band = self._native_decimation_band()
        spectral_band = max(
            0.0, min(native_band, self.error_target_band)
        )
        relaxation = self._throughput_relaxation(nats_per_byte)
        throughput_target = (
            (1.0 - relaxation) * spectral_band
            + relaxation * native_band
        )
        raw_target = (1.0 - ratio) * throughput_target
        target = max(
            0.0,
            min(
                self._max_decimation_band(),
                self.decimation_gain * raw_target,
            ),
        )
        quantum = self._controller_quantum()
        if elapsed_predictions is None or quantum <= 0.0:
            control_elapsed = 1.0
        else:
            control_elapsed = max(0.0, float(elapsed_predictions)) / quantum
        if target > self.decimation_band:
            target = min(target, self.decimation_band + control_elapsed)
        else:
            target = max(target, self.decimation_band - 2.0 * control_elapsed)
        self.decimation_band = target
        self._set_prediction_stride()
        self._adapt_lattice()

    def _adapt_lattice(self) -> None:
        """let the error spectrum slowly deform the temporal lattice.

        mesh refinement on time: neighbouring bandpass residual densities
        move their shared boundary toward the denser interval. the horizon
        is conserved, each coordinate moves on its own clock, and elasticity
        leans the interior mesh back toward the phi prior. the trace follows
        automatically through the shared lattice; the error bank and mixer
        are re-anchored explicitly."""

        effective_rate = self.effective_lattice_rate()
        if effective_rate <= 0.0 or self.config.bands < 2:
            return
        elapsed = self.training_tokens_seen - self._last_lattice_token
        if elapsed <= 0:
            return
        self._last_lattice_token = self.training_tokens_seen
        before = self.lattice.coordinates.clone()
        self.lattice.deform_(
            self.error_trace[: self.config.bands - 1],
            elapsed,
            effective_rate,
            self.lattice_elastic,
        )
        if torch.equal(before, self.lattice.coordinates):
            return
        self.error_bank.refresh_lattice()
        self.model.refresh_lattice(self.lattice)
        self._refresh_error_state()

    def lattice_drift(self) -> dict[str, float]:
        """how far the receiver has tuned away from the phi prior."""

        drift = self.lattice.drift()
        return {
            "max_abs": float(drift.abs().max()),
            "mean_abs": float(drift.abs().mean()),
        }

    def train_batch(
        self,
        token_ids: torch.Tensor | np.ndarray | list[int],
        *,
        source_bytes: float = 0.0,
        loss_bytes: float | None = None,
        loss_weights: torch.Tensor | np.ndarray | None = None,
    ) -> BatchMetrics:
        ids_cpu = torch.as_tensor(
            token_ids, dtype=torch.long, device="cpu"
        ).flatten()
        if ids_cpu.numel() == 0:
            raise ValueError("training batch cannot be empty")
        batch_started = time.perf_counter()

        block_start = self.training_tokens_seen
        prior = self.trace.prior()
        ids = ids_cpu.to(self.device)

        if loss_weights is None:
            full_weights = torch.ones(
                ids.shape, device=self.device, dtype=torch.float32
            )
        else:
            full_weights = torch.as_tensor(
                loss_weights,
                device=self.device,
                dtype=torch.float32,
            ).flatten()
            if full_weights.shape != ids.shape:
                raise ValueError("loss weights must match the token batch")
        prior_weights = (
            full_weights.detach().to(device="cpu").to(dtype=torch.float64)
        )
        for token_id in self.tokenizer.control_ids.values():
            prior_weights[ids_cpu == int(token_id)] = 0.0
        candidate_trace = self.trace.advanced(
            ids_cpu,
            source_bytes=source_bytes,
            update_prior=True,
            prior_weights=prior_weights,
        )
        stride = max(1, int(self.prediction_stride))
        first_row = (-int(block_start)) % stride
        if first_row >= int(ids.numel()):
            rows = torch.empty(0, device=self.device, dtype=torch.long)
        else:
            rows = torch.arange(
                first_row,
                ids.numel(),
                stride,
                device=self.device,
                dtype=torch.long,
            )
        scan_stride: int | None = stride
        selected_weights = full_weights.index_select(0, rows)
        if (
            rows.numel() == 0
            or float(selected_weights.sum().detach().cpu()) <= 0.0
        ):
            supervised = torch.nonzero(full_weights > 0.0).flatten()
            if supervised.numel() == 0:
                raise ValueError("training batch has no weighted targets")
            rows = supervised[:1]
            scan_stride = None
            selected_weights = full_weights.index_select(0, rows)
        selected_ids = ids.index_select(0, rows)
        prediction_priors = self.trace.priors_before(
            ids_cpu,
            rows.detach().cpu(),
            prior_weights=prior_weights,
        )

        self.batches += 1
        update_due = self._schedule_update(int(rows.numel()))
        self.model.train(update_due)
        if update_due and self.accumulated_prediction_mass <= 0.0:
            self.optimizer.zero_grad(set_to_none=True)
        grad_context = torch.enable_grad() if update_due else torch.no_grad()
        with grad_context:
            output = self.model(
                self.trace.trace,
                prior,
                self.trace.last_token,
                ids,
                prediction_rows=rows,
                prediction_stride=scan_stride,
                prediction_priors=prediction_priors,
                return_details=True,
            )
        assert isinstance(output, MixerOutput)
        per_token = F.cross_entropy(
            output.logits, selected_ids, reduction="none"
        )
        weights = selected_weights.to(per_token.dtype)
        denominator = weights.sum()
        if float(denominator.detach().cpu()) <= 0.0:
            raise ValueError("training batch has no weighted targets")
        prediction_mass = float(denominator.detach().cpu())
        # pulse time measures forward/backward compute, not language-loss
        # weight. A masked control prediction still costs one prediction;
        # conflating the two leaves fractional pulse boundaries in protocol
        # corpora even though plain text happens to remain integral.
        pulse_mass = float(rows.numel())
        pulse_remaining = self.pulse_prediction_mass_remaining()
        if pulse_mass > pulse_remaining + 1e-6:
            raise ValueError("training batch crosses a pulse quantum boundary")
        token_nll = (per_token * weights).sum()
        loss = token_nll / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite language loss")
        correction = self._translated_correction(
            output.logits, selected_ids, weights
        )

        grad_norm_value = 0.0
        optimizer_stepped = False
        if update_due:
            if self.accumulated_prediction_mass <= 0.0:
                self.accumulation_learning_rate = float(self.learning_rate)
            token_nll.backward()
            # Update cadence belongs to compute time, while gradient scale
            # belongs to supervised language mass. Keeping these clocks
            # separate prevents masked protocol tokens from fragmenting a
            # quantum into repeated one-prediction MPS calls.
            self.accumulated_prediction_mass += pulse_mass
            self.accumulated_loss_mass += prediction_mass
            quantum_reached = (
                self.optimizer_quantum <= 0.0
                or self.accumulated_prediction_mass + 1e-9
                >= self.optimizer_quantum
            )
            if quantum_reached:
                grad_norm_value = self._optimizer_step_from_accumulation()
                optimizer_stepped = True

        self.trace = candidate_trace
        self.training_tokens_seen += int(ids_cpu.numel())
        self._advance_error_bank(
            correction, rows, block_start, sampling_stride=scan_stride
        )
        self._advance_error_time_to(self.training_tokens_seen)
        full_denominator = float(full_weights.sum().detach().cpu())
        selected = int(rows.numel())
        nll_value = float(loss.detach().cpu()) * full_denominator
        if loss_bytes is None:
            loss_bytes = float(source_bytes)
        loss_bytes = float(loss_bytes)
        self.predictions += selected
        self.total_nll += nll_value
        self.loss_bytes += loss_bytes
        top1 = float(
            (output.logits.argmax(dim=-1) == selected_ids)
            .to(per_token.dtype)
            .mul(weights)
            .sum()
            .div(denominator)
            .detach()
            .cpu()
        )
        top5 = float(
            output.logits.topk(min(5, self.config.vocab_size), dim=-1)
            .indices.eq(selected_ids.unsqueeze(-1)).any(dim=-1)
            .to(per_token.dtype).mul(weights).sum().div(denominator)
            .detach().cpu()
        )
        bpb = (
            nll_value / (loss_bytes * math.log(2.0))
            if loss_bytes > 0.0
            else float("nan")
        )
        self._accumulate_io2(
            token_nll=float(token_nll.detach().cpu()),
            prediction_mass=float(denominator.detach().cpu()),
            loss_token_mass=full_denominator,
            loss_bytes=loss_bytes,
        )
        self._commit_pulse_mass(pulse_mass)
        self._adopt_pending_training_quantum()
        self._record_batch_time(time.perf_counter() - batch_started, update_due)
        return BatchMetrics(
            loss=float(loss.detach().cpu()),
            token_nll=nll_value,
            bpb=bpb,
            top1=top1,
            grad_norm=grad_norm_value,
            top5=top5,
            tokens=int(ids.numel()),
            source_bytes=float(source_bytes),
            loss_bytes=loss_bytes,
            updated=optimizer_stepped,
            batch=self.batches,
            update=self.updates,
            predictions=selected,
            prediction_mass=float(denominator.detach().cpu()),
            accumulated_prediction_mass=float(
                self.accumulated_prediction_mass
            ),
            optimizer_quantum=float(self.optimizer_quantum),
        )

    def observe(
        self,
        token_ids: torch.Tensor | np.ndarray | list[int],
        *,
        source_bytes: float = 0.0,
        update_prior: bool = False,
    ) -> None:
        ids = torch.as_tensor(
            token_ids, dtype=torch.long, device="cpu"
        ).flatten()
        self.trace.advance_(
            ids,
            source_bytes=source_bytes,
            update_prior=update_prior,
        )
        self.training_tokens_seen += int(ids.numel())
        self._advance_error_time_to(self.training_tokens_seen)

    @torch.no_grad()
    def evaluate_ids(
        self,
        token_ids: torch.Tensor | np.ndarray | list[int],
        *,
        source_bytes: float,
        batch_size: int = 512,
        loss_weights: torch.Tensor | np.ndarray | list[float] | None = None,
        loss_bytes: float | None = None,
    ) -> EvaluationMetrics:
        """score a contiguous continuation without changing lived state."""

        ids_cpu = torch.as_tensor(
            token_ids, dtype=torch.long, device="cpu"
        ).flatten()
        if ids_cpu.numel() == 0:
            raise ValueError("evaluation sequence cannot be empty")
        if source_bytes <= 0.0:
            raise ValueError("evaluation source bytes must be positive")
        if loss_weights is None:
            weights_cpu = torch.ones(ids_cpu.shape, dtype=torch.float32)
        else:
            weights_cpu = torch.as_tensor(
                loss_weights, dtype=torch.float32, device="cpu"
            ).flatten()
            if weights_cpu.shape != ids_cpu.shape:
                raise ValueError("evaluation weights must match token ids")
            if (
                not bool(torch.isfinite(weights_cpu).all())
                or float(weights_cpu.min()) < 0.0
            ):
                raise ValueError(
                    "evaluation weights must be finite and nonnegative"
                )
        supervised_mass = float(weights_cpu.sum())
        if supervised_mass <= 0.0:
            raise ValueError("evaluation sequence has no weighted targets")
        loss_bytes = float(source_bytes if loss_bytes is None else loss_bytes)
        if loss_bytes <= 0.0:
            raise ValueError("evaluation loss bytes must be positive")

        self.model.eval()
        state = self.trace.clone()
        total_nll = 0.0
        total_correct = 0.0
        for start in range(0, int(ids_cpu.numel()), int(batch_size)):
            chunk_cpu = ids_cpu[start : start + int(batch_size)]
            chunk = chunk_cpu.to(self.device)
            chunk_weights_cpu = weights_cpu[
                start : start + int(batch_size)
            ]
            chunk_weights = chunk_weights_cpu.to(self.device)
            prior = state.prior()
            prior_weights = chunk_weights_cpu.to(dtype=torch.float64)
            for token_id in self.tokenizer.control_ids.values():
                prior_weights[chunk_cpu == int(token_id)] = 0.0
            prediction_priors = state.priors_before(
                chunk_cpu, prior_weights=prior_weights
            )
            logits = self.model(
                state.trace,
                prior,
                state.last_token,
                chunk,
                prediction_priors=prediction_priors,
            )
            per_token = F.cross_entropy(logits, chunk, reduction="none")
            total_nll += float(
                (per_token * chunk_weights).sum().detach().cpu()
            )
            total_correct += float(
                (logits.argmax(dim=-1) == chunk)
                .to(chunk_weights.dtype)
                .mul(chunk_weights)
                .sum()
                .detach()
                .cpu()
            )
            state = state.advanced(
                chunk_cpu,
                update_prior=True,
                prior_weights=prior_weights,
            )

        return EvaluationMetrics(
            token_nll=total_nll,
            bpb=total_nll / (loss_bytes * math.log(2.0)),
            top1=total_correct / supervised_mass,
            tokens=int(ids_cpu.numel()),
            source_bytes=float(source_bytes),
        )

    @torch.no_grad()
    def next_logits(self) -> torch.Tensor:
        self.model.eval()
        prior = self.trace.prior()
        projected, direct, _ = self.model.project_state(
            self.trace.trace, prior, self.trace.last_token
        )
        return self.model.forward_features(
            projected.unsqueeze(0),
            direct.unsqueeze(0),
            prior,
        )[0]

    @torch.no_grad()
    def generate_ids(
        self,
        length: int,
        *,
        temperature: float = 1.0,
        greedy: bool = False,
        stop_ids: set[int] | None = None,
        forbidden_ids: set[int] | tuple[int, ...] | None = None,
    ) -> list[int]:
        if length < 1:
            return []
        if temperature <= 0.0 and not greedy:
            raise ValueError("temperature must be positive")
        self.model.eval()
        prior = self.trace.prior()
        projected, direct, _ = self.model.project_state(
            self.trace.trace, prior, self.trace.last_token
        )
        generated: list[int] = []
        for _ in range(int(length)):
            logits = self.model.forward_features(
                projected.unsqueeze(0),
                direct.unsqueeze(0),
                prior,
            )[0]
            if forbidden_ids:
                logits[list(forbidden_ids)] = float("-inf")
            if greedy:
                token_id = int(logits.argmax())
            else:
                probabilities = F.softmax(logits / temperature, dim=-1)
                token_id = self._sample_token(probabilities)
            generated.append(token_id)
            projected, direct = self.model.advance_projected(
                projected, token_id, prior
            )
            if stop_ids and token_id in stop_ids:
                break
        self.observe(
            generated, source_bytes=0.0, update_prior=False
        )
        return generated

    def generate_text(
        self,
        length: int,
        *,
        temperature: float = 1.0,
        greedy: bool = False,
        stop_ids: set[int] | None = None,
        forbidden_ids: set[int] | tuple[int, ...] | None = None,
    ) -> str:
        return self.tokenizer.decode(
            self.generate_ids(
                length,
                temperature=temperature,
                greedy=greedy,
                stop_ids=stop_ids,
                forbidden_ids=forbidden_ids,
            )
        )

    def checkpoint_dict(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        revision_id: str | None = None,
    ) -> dict[str, Any]:
        _, retention, injection, _ = self.lattice.derive()
        combined_metadata = dict(self.metadata)
        combined_metadata.update(metadata or {})
        candidate_revision = str(revision_id or uuid.uuid4().hex)
        parent_revision = (
            self.revision_id
            if self._revision_published
            else self.parent_revision_id
        )
        gradients = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.grad is not None
        }
        return {
            "species": self.model.species,
            "checkpoint_version": CHECKPOINT_VERSION,
            "runtime_version": RUNTIME_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "model_id": self.checkpoint_id,
            "revision_id": candidate_revision,
            "parent_revision_id": parent_revision,
            "root_lineage_id": self.root_lineage_id,
            "created_at": self.created_at,
            "saved_at": time.time(),
            "description": self.description,
            "config": self.config.to_dict(),
            "lattice": {
                "coordinates": self.lattice.coordinates.clone(),
                "monitor_density": self.lattice.monitor_density.clone(),
                "retention": retention,
                "injection": injection,
                "direct_channel": "previous_token",
            },
            "tokenizer": {
                "sha256": self.tokenizer.sha256,
                "filename": self.tokenizer.path.name,
            },
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "trace": self.trace.state_dict(),
            "error_trace": self.error_trace.clone(),
            "translated_error_bank": self.error_bank.trace.clone(),
            "gradients": gradients,
            "rng": {
                "sampling": self.sampling_generator.get_state(),
            },
            "training": {
                "updates": self.updates,
                "batches": self.batches,
                "predictions": self.predictions,
                "total_nll": self.total_nll,
                "loss_bytes": self.loss_bytes,
                "learning_rate": self.learning_rate,
                "learning_rate_base": self.learning_rate_base,
                "lr_auto": self.lr_auto,
                "weight_decay": self.weight_decay,
                "grad_clip": self.grad_clip,
                "update_interval": self.update_interval,
                "decimation_gain": self.decimation_gain,
                "decimation_stride_cap": self.decimation_stride_cap,
                "radial_compression": self.radial_compression,
                "radial_ceiling_scale": self.radial_ceiling_scale,
                "radial_floor": self.radial_floor,
                "pulse_enabled": self.pulse_enabled,
                "pulse_quantum": self.pulse_quantum,
                "pulse_frequency_exponent": self.pulse_frequency_exponent,
                "pulse_schedule": self.pulse_schedule,
                "pulse_phase": self.pulse_phase,
                "pulse_quantum_fill": self.pulse_quantum_fill,
                "pulse_quantum_updates": self.pulse_quantum_updates,
                "optimizer_quantum": self.optimizer_quantum,
                "pending_optimizer_quantum": self.pending_optimizer_quantum,
                "accumulated_prediction_mass": (
                    self.accumulated_prediction_mass
                ),
                "accumulated_loss_mass": self.accumulated_loss_mass,
                "accumulation_learning_rate": (
                    self.accumulation_learning_rate
                ),
                "controller_prediction_mass": (
                    self.controller_prediction_mass
                ),
                "controller_token_nll": self.controller_token_nll,
                "controller_loss_token_mass": (
                    self.controller_loss_token_mass
                ),
                "controller_loss_bytes": self.controller_loss_bytes,
                "update_time_ema": self.update_time_ema,
                "observe_time_ema": self.observe_time_ema,
                "update_cost_fraction": self.update_cost_fraction,
                "decimation_band": self.decimation_band,
                "prediction_stride": self.prediction_stride,
                "io2_plasticity": self.io2_plasticity,
                "last_error_token": self.last_error_token,
                "error_sampling_stride": self.error_sampling_stride,
                "training_tokens_seen": self.training_tokens_seen,
                "lattice_rate": self.lattice_rate,
                "lattice_elastic": self.lattice_elastic,
                "last_lattice_token": self._last_lattice_token,
            },
            "metadata": combined_metadata,
        }

    def save(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        revision_id = uuid.uuid4().hex
        checkpoint = self.checkpoint_dict(
            metadata=metadata,
            revision_id=revision_id,
        )
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
        self.parent_revision_id = checkpoint["parent_revision_id"]
        self.revision_id = revision_id
        self._revision_published = True
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        tokenizer: ReferenceTokenizer,
        *,
        device: str | torch.device = "auto",
    ) -> "TraceBranchTrainer":
        checkpoint = torch.load(
            Path(path).expanduser(),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        return cls.from_checkpoint_dict(
            checkpoint, tokenizer, device=device
        )

    @classmethod
    def from_checkpoint_dict(
        cls,
        checkpoint: dict[str, Any],
        tokenizer: ReferenceTokenizer,
        *,
        device: str | torch.device = "auto",
    ) -> "TraceBranchTrainer":
        """restore an already-read checkpoint without a second disk load."""

        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint root must be a dictionary")
        if checkpoint.get("species") not in COMPATIBLE_SPECIES:
            raise ValueError("checkpoint is not a trace branch mixer")
        if (
            checkpoint.get("checkpoint_version")
            not in COMPATIBLE_CHECKPOINT_VERSIONS
        ):
            raise ValueError("unsupported checkpoint version")
        if checkpoint["tokenizer"]["sha256"] != tokenizer.sha256:
            raise ValueError("tokenizer hash does not match checkpoint")

        for field in (
            "model",
            "optimizer",
            "trace",
            "error_trace",
            "translated_error_bank",
            "gradients",
        ):
            if field in checkpoint:
                _require_finite_tensors(checkpoint[field], field)

        config = TraceBranchConfig.from_dict(checkpoint["config"])
        lattice_block = checkpoint.get("lattice", {})
        if lattice_block.get("direct_channel") != "previous_token":
            raise ValueError("checkpoint direct-channel convention differs")
        coordinates = lattice_block.get("coordinates")
        if coordinates is None:
            # a v13 checkpoint is a valid v14 individual whose lattice
            # sits exactly at the phi prior with no accumulated drift.
            lattice = TemporalLattice(config)
        else:
            lattice = TemporalLattice(
                config,
                coordinates,
                monitor_density=lattice_block.get("monitor_density"),
            )
        _, retention, injection, _ = lattice.derive()
        if not torch.equal(lattice_block.get("retention"), retention):
            raise ValueError("checkpoint retention lattice differs")
        if not torch.equal(lattice_block.get("injection"), injection):
            raise ValueError("checkpoint injection lattice differs")
        training = checkpoint.get("training", {})
        trainer = cls(
            config,
            tokenizer,
            device=device,
            learning_rate=training.get(
                "learning_rate_base",
                training.get("learning_rate", 3e-4),
            ),
            weight_decay=training.get("weight_decay", 1e-4),
            grad_clip=training.get("grad_clip", 1.0),
            update_interval=training.get("update_interval", 1),
            lr_auto=training.get("lr_auto", True),
            decimation_gain=training.get("decimation_gain", 1.0),
            decimation_stride_cap=training.get(
                "decimation_stride_cap", 256
            ),
            radial_compression=training.get("radial_compression", True),
            radial_ceiling_scale=training.get(
                "radial_ceiling_scale", 0.1
            ),
            radial_floor=training.get("radial_floor", 0.01),
            pulse_enabled=training.get("pulse_enabled", False),
            pulse_quantum=training.get("pulse_quantum", 1024),
            pulse_frequency_exponent=training.get(
                "pulse_frequency_exponent", 0.5
            ),
            pulse_schedule=training.get("pulse_schedule", "spectral"),
            optimizer_quantum=training.get("optimizer_quantum", 0.0),
            lattice_rate=training.get("lattice_rate", 0.0),
            lattice_elastic=training.get("lattice_elastic", 0.1),
        )
        trainer.lattice = lattice
        trainer.model.load_state_dict(checkpoint["model"], strict=True)
        trainer.model.refresh_lattice(lattice)
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in trainer.optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(trainer.device)
        trainer.trace = TraceState.from_state_dict(
            config, checkpoint["trace"], lattice=lattice
        )
        translated_error = checkpoint.get("translated_error_bank")
        if translated_error is not None:
            trainer.error_bank = TranslatedErrorBank(
                config, translated_error, lattice=lattice
            )
            trainer._refresh_error_state()
        else:
            trainer.error_bank.lattice = lattice
            trainer.error_bank.refresh_lattice()
        error_trace = checkpoint.get("error_trace")
        if translated_error is None and error_trace is not None:
            error_trace = torch.as_tensor(
                error_trace, dtype=torch.float64, device="cpu"
            )
            if error_trace.shape != (config.bands,):
                raise ValueError("checkpoint error trace shape differs")
            trainer.error_trace = error_trace.clone()
        trainer.learning_rate = float(training.get(
            "learning_rate", trainer.learning_rate_base
        ))
        trainer.decimation_band = float(training.get("decimation_band", 0.0))
        trainer._set_prediction_stride()
        trainer.io2_plasticity = float(
            training.get("io2_plasticity", 1.0)
        )
        trainer.pulse_phase = float(training.get("pulse_phase", 0.0))
        trainer.pulse_quantum_fill = float(
            training.get("pulse_quantum_fill", 0.0)
        )
        pulse_decision = training.get("pulse_quantum_updates")
        if pulse_decision is not None and not isinstance(pulse_decision, bool):
            raise ValueError("checkpoint pulse decision is invalid")
        trainer.pulse_quantum_updates = pulse_decision
        pulse_quantum = (
            trainer._controller_quantum() or float(trainer.pulse_quantum)
        )
        if (
            not math.isfinite(trainer.pulse_quantum_fill)
            or trainer.pulse_quantum_fill < 0.0
            or trainer.pulse_quantum_fill >= pulse_quantum
            or (
                trainer.pulse_quantum_fill > 0.0
                and trainer.pulse_quantum_updates is None
            )
        ):
            raise ValueError("checkpoint pulse quantum is invalid")
        trainer.update_time_ema = float(training.get("update_time_ema", 0.0))
        trainer.observe_time_ema = float(
            training.get("observe_time_ema", 0.0)
        )
        trainer.update_cost_fraction = float(
            training.get("update_cost_fraction", 0.75)
        )
        trainer.last_error_token = int(training.get(
            "last_error_token", 0
        ))
        trainer.error_sampling_stride = max(1, int(training.get(
            "error_sampling_stride", 1
        )))
        trainer._refresh_error_state()
        trainer.training_tokens_seen = int(training.get(
            "training_tokens_seen", trainer.last_error_token
        ))
        trainer._last_lattice_token = int(training.get(
            "last_lattice_token", trainer.training_tokens_seen
        ))
        trainer.accumulated_prediction_mass = float(training.get(
            "accumulated_prediction_mass", 0.0
        ))
        trainer.accumulated_loss_mass = float(training.get(
            "accumulated_loss_mass",
            trainer.accumulated_prediction_mass,
        ))
        if (
            not math.isfinite(trainer.accumulated_prediction_mass)
            or trainer.accumulated_prediction_mass < 0.0
            or not math.isfinite(trainer.accumulated_loss_mass)
            or trainer.accumulated_loss_mass < 0.0
        ):
            raise ValueError("checkpoint optimizer accumulation is invalid")
        if (
            trainer.accumulated_prediction_mass > 0.0
            and trainer.optimizer_quantum <= 0.0
        ):
            raise ValueError("checkpoint accumulation has no optimizer quantum")
        pending_quantum = training.get("pending_optimizer_quantum")
        trainer.pending_optimizer_quantum = (
            None if pending_quantum is None else float(pending_quantum)
        )
        if (
            trainer.pending_optimizer_quantum is not None
            and (
                not math.isfinite(trainer.pending_optimizer_quantum)
                or trainer.pending_optimizer_quantum <= 0.0
            )
        ):
            raise ValueError("checkpoint pending optimizer quantum is invalid")
        accumulation_lr = training.get("accumulation_learning_rate")
        trainer.accumulation_learning_rate = (
            None if accumulation_lr is None else float(accumulation_lr)
        )
        if (
            trainer.accumulation_learning_rate is not None
            and (
                not math.isfinite(trainer.accumulation_learning_rate)
                or trainer.accumulation_learning_rate < 0.0
            )
        ):
            raise ValueError("checkpoint accumulation learning rate is invalid")
        gradients = checkpoint.get("gradients", {})
        if not isinstance(gradients, dict):
            raise ValueError("checkpoint gradients must be a dictionary")
        named_parameters = dict(trainer.model.named_parameters())
        unknown_gradients = set(gradients) - set(named_parameters)
        if unknown_gradients:
            raise ValueError("checkpoint gradients do not match model")
        for name, gradient in gradients.items():
            parameter = named_parameters[name]
            gradient = torch.as_tensor(gradient)
            if gradient.shape != parameter.shape:
                raise ValueError("checkpoint gradient shape differs")
            parameter.grad = gradient.to(
                device=trainer.device, dtype=parameter.dtype
            ).clone()
        if (
            trainer.accumulated_prediction_mass > 0.0
            and trainer.accumulated_loss_mass <= 0.0
        ):
            raise ValueError("checkpoint accumulation has no language mass")
        if trainer.accumulated_prediction_mass > 0.0 and not gradients:
            raise ValueError("checkpoint accumulation has no gradients")
        if trainer.accumulated_prediction_mass <= 0.0 and gradients:
            raise ValueError("checkpoint gradients have no prediction mass")
        controller_fields = (
            "controller_prediction_mass",
            "controller_token_nll",
            "controller_loss_token_mass",
            "controller_loss_bytes",
        )
        controller_values = {
            name: float(training.get(name, 0.0))
            for name in controller_fields
        }
        if any(
            not math.isfinite(value) or value < 0.0
            for value in controller_values.values()
        ):
            raise ValueError("checkpoint controller accumulation is invalid")
        trainer.controller_prediction_mass = controller_values[
            "controller_prediction_mass"
        ]
        trainer.controller_token_nll = controller_values[
            "controller_token_nll"
        ]
        trainer.controller_loss_token_mass = controller_values[
            "controller_loss_token_mass"
        ]
        trainer.controller_loss_bytes = controller_values[
            "controller_loss_bytes"
        ]
        if (
            trainer.controller_prediction_mass <= 0.0
            and any(
                controller_values[name] > 0.0
                for name in controller_fields[1:]
            )
        ):
            raise ValueError("checkpoint controller totals have no mass")
        trainer._set_optimizer_learning_rate()
        trainer.batches = int(training.get("batches", 0))
        trainer.updates = int(training.get("updates", 0))
        trainer.predictions = int(training.get("predictions", 0))
        trainer.total_nll = float(training.get("total_nll", 0.0))
        trainer.loss_bytes = float(training.get("loss_bytes", 0.0))
        trainer.checkpoint_id = str(
            checkpoint.get("model_id")
            or checkpoint.get("checkpoint_id")
            or uuid.uuid4().hex
        )
        trainer.revision_id = str(
            checkpoint.get("revision_id") or uuid.uuid4().hex
        )
        trainer.parent_revision_id = (
            None
            if checkpoint.get("parent_revision_id") is None
            else str(checkpoint.get("parent_revision_id"))
        )
        trainer.root_lineage_id = str(
            checkpoint.get("root_lineage_id") or trainer.checkpoint_id
        )
        trainer._revision_published = True
        trainer.created_at = float(checkpoint.get("created_at", time.time()))
        trainer.description = str(checkpoint.get("description", ""))
        trainer.metadata = dict(checkpoint.get("metadata", {}))
        rng = checkpoint.get("rng", {})
        sampling_state = rng.get("sampling") if isinstance(rng, dict) else None
        if sampling_state is not None:
            trainer.sampling_generator.set_state(
                torch.as_tensor(sampling_state, dtype=torch.uint8, device="cpu")
            )
        return trainer
