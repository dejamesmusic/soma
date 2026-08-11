"""exact trace state and the trace branch mixer network."""

from __future__ import annotations

import math
import numbers
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


PHI = (1.0 + math.sqrt(5.0)) / 2.0


@dataclass(frozen=True)
class TraceBranchConfig:
    """structural parameters for one trace branch mixer."""

    vocab_size: int = 1024
    bands: int = 48
    beta0: float = 1.0
    phi: float = PHI
    symbol_dim: int = 256
    scale_rank: int = 64
    scale_depth: int = 3
    scale_kernel: int = 5
    summaries: int = 8
    model_dim: int = 1024
    trunk_depth: int = 4
    expansion_dim: int = 2048
    prior_smoothing: float = 1.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        integer_fields = (
            "vocab_size",
            "bands",
            "symbol_dim",
            "scale_rank",
            "scale_depth",
            "scale_kernel",
            "summaries",
            "model_dim",
            "trunk_depth",
            "expansion_dim",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value))
                or int(value) != value
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be positive")
        if self.scale_kernel % 2 != 1:
            raise ValueError("scale_kernel must be odd")
        if self.summaries > self.bands:
            raise ValueError("summaries cannot exceed bands")
        if not math.isfinite(self.beta0) or not (0.0 < self.beta0 <= 1.0):
            raise ValueError("beta0 must be in (0, 1]")
        if not math.isfinite(self.phi) or self.phi <= 1.0:
            raise ValueError("phi must be greater than one")
        if (
            not math.isfinite(self.prior_smoothing)
            or self.prior_smoothing <= 0.0
        ):
            raise ValueError("prior_smoothing must be positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        slowest_error_beta = self.beta0 * self.phi ** (-self.bands)
        if slowest_error_beta <= 0.0 or math.exp(-slowest_error_beta) == 1.0:
            raise ValueError(
                "bands and phi exceed float64 temporal resolution"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TraceBranchConfig":
        return cls(**values)


def temporal_lattice(
    config: TraceBranchConfig,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """return beta, retention, injection, and lifetime coordinates."""

    indices = torch.arange(config.bands, dtype=dtype, device=device)
    beta = config.beta0 * torch.pow(
        torch.as_tensor(config.phi, dtype=dtype, device=device), -indices
    )
    retention = torch.exp(-beta)
    injection = -torch.expm1(-beta)
    lifetime = beta.reciprocal()
    return beta, retention, injection, lifetime


# the slowest representable decay: below this, exp(-beta) rounds to 1.0
# in float64 and a band stops integrating entirely.
BETA_RESOLUTION_FLOOR = 1e-14


class TemporalLattice:
    """learned log-time coordinates for the trace bands.

    each band's decay is beta_i = beta0 * phi^(-c_i). the phi lattice
    (c_i = i) is the prior, not the destiny: a controller may deform the
    coordinates toward the corpus's own temporal statistics. invariants
    held at all times: c_0 = 0 (the fastest band stays anchored at the
    token scale), c_last = bands - 1 (adaptation redistributes a conserved
    horizon), coordinates strictly increase (bands never cross), and the
    extended error band stays above float64 temporal resolution."""

    MIN_GAP = 0.25
    MAX_GAP = 4.0
    REFERENCE_TOKENS = 1024.0
    MONITOR_HORIZON_TOKENS = 10.0 * REFERENCE_TOKENS

    def __init__(
        self,
        config: TraceBranchConfig,
        coordinates: torch.Tensor | None = None,
        monitor_density: torch.Tensor | None = None,
    ) -> None:
        self.config = config
        if coordinates is None:
            coordinates = torch.arange(config.bands, dtype=torch.float64)
        coordinates = torch.as_tensor(
            coordinates, dtype=torch.float64, device="cpu"
        ).flatten().clone()
        self._validate(coordinates)
        self.coordinates = coordinates
        if monitor_density is None:
            monitor_density = torch.ones(
                max(0, config.bands - 1), dtype=torch.float64
            )
        monitor_density = torch.as_tensor(
            monitor_density, dtype=torch.float64, device="cpu"
        ).flatten().clone()
        if monitor_density.shape != (max(0, config.bands - 1),):
            raise ValueError("lattice monitor must have one value per gap")
        if (
            not torch.isfinite(monitor_density).all()
            or (monitor_density <= 0.0).any()
        ):
            raise ValueError("lattice monitor must be finite and positive")
        self.monitor_density = monitor_density

    def _resolution_wall(self) -> float:
        return math.log(
            self.config.beta0 / BETA_RESOLUTION_FLOOR, self.config.phi
        )

    def _validate(self, coordinates: torch.Tensor) -> None:
        if coordinates.shape != (self.config.bands,):
            raise ValueError("lattice coordinates must match band count")
        if not torch.isfinite(coordinates).all():
            raise ValueError("lattice coordinates must be finite")
        if abs(float(coordinates[0])) > 1e-12:
            raise ValueError("the fastest band must stay at coordinate zero")
        expected_horizon = float(self.config.bands - 1)
        if abs(float(coordinates[-1]) - expected_horizon) > 1e-9:
            raise ValueError("the temporal horizon must remain conserved")
        if coordinates.numel() > 1 and not bool(
            torch.all(coordinates[1:] > coordinates[:-1])
        ):
            raise ValueError("lattice coordinates must strictly increase")
        # the extended coordinate (error bank's extra slow band) must
        # also stay representable.
        if float(self._extended(coordinates)[-1]) > self._resolution_wall():
            raise ValueError("lattice exceeds float64 temporal resolution")

    @staticmethod
    def _extended(coordinates: torch.Tensor) -> torch.Tensor:
        """append one band slower than the last, continuing local spacing."""

        if coordinates.numel() == 1:
            step = 1.0
        else:
            step = float(coordinates[-1] - coordinates[-2])
        return torch.cat(
            (coordinates, coordinates[-1:] + step)
        )

    def derive(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        extended: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """return beta, retention, injection, lifetime on this lattice."""

        coordinates = (
            self._extended(self.coordinates) if extended else self.coordinates
        )
        coordinates = coordinates.to(dtype=dtype, device=device)
        beta = self.config.beta0 * torch.pow(
            torch.as_tensor(self.config.phi, dtype=dtype, device=device),
            -coordinates,
        )
        retention = torch.exp(-beta)
        injection = -torch.expm1(-beta)
        lifetime = beta.reciprocal()
        return beta, retention, injection, lifetime

    def drift(self) -> torch.Tensor:
        """per-band displacement from the phi prior, in phi exponents."""

        return self.coordinates - torch.arange(
            self.config.bands, dtype=torch.float64
        )

    def interval_coordinates(self, *, extended: bool = True) -> torch.Tensor:
        """physical phi coordinates for adjacent-band residual intervals.

        subtracting one half preserves v13's index convention at the phi
        prior: the interval between prior coordinates i and i + 1 maps to i.
        """

        coordinates = (
            self._extended(self.coordinates) if extended else self.coordinates
        )
        return 0.5 * (coordinates[:-1] + coordinates[1:]) - 0.5

    def deform_(
        self,
        gap_error_energy: torch.Tensor,
        elapsed_tokens: int,
        rate: float,
        elastic: float,
    ) -> None:
        """advance conservative mesh adaptation through lived source time.

        long sparse-observation intervals are integrated as bounded reference
        steps. this makes high decimation equivalent to elapsed time rather
        than one oversized euler update.
        """

        if rate <= 0.0 or self.config.bands < 2 or elapsed_tokens <= 0:
            return
        remaining = int(elapsed_tokens)
        reference = int(self.REFERENCE_TOKENS)
        while remaining > 0:
            step = min(reference, remaining)
            self._deform_step_(gap_error_energy, step, rate, elastic)
            remaining -= step

    def _deform_step_(
        self,
        gap_error_energy: torch.Tensor,
        elapsed_tokens: int,
        rate: float,
        elastic: float,
    ) -> None:
        """one bounded, adiabatic mesh-refinement integration step.

        residual energy is converted to density per unit log-time. each
        interior coordinate compares the density in its left and right
        intervals and moves toward the denser side, contracting that interval
        while expanding its neighbour. endpoints remain fixed, so adaptation
        redistributes resolution without silently changing the horizon.

        every coordinate is rate-limited by its own integration rate. one
        global feasibility scale then keeps every gap inside its structural
        bounds without changing the direction of the proposed motion.
        """

        bands = self.config.bands
        energy = torch.as_tensor(
            gap_error_energy, dtype=torch.float64, device="cpu"
        ).flatten()
        if energy.shape != (bands - 1,):
            raise ValueError("gap error energy must have one value per gap")
        if not torch.isfinite(energy).all() or float(energy.min()) < 0.0:
            raise ValueError("gap error energy must be finite and nonnegative")
        gaps = self.coordinates[1:] - self.coordinates[:-1]
        velocity = torch.zeros_like(self.coordinates)
        total = float(energy.sum())
        if total > 1e-12 and bands > 2:
            observed_density = energy / gaps
            observed_density = observed_density / observed_density.mean().clamp_min(
                1e-300
            )
            # the error bank has already integrated each interval on its own
            # clock. this second filter only rejects controller-quantum
            # transients, so it uses one shared evidence cadence; applying
            # the local beta here again would make slow adaptation scale as
            # beta squared.
            monitor_update = -math.expm1(
                -float(elapsed_tokens) / self.MONITOR_HORIZON_TOKENS
            )
            self.monitor_density.add_(
                monitor_update * (observed_density - self.monitor_density)
            ).clamp_min_(1e-12)
            log_density = torch.log(self.monitor_density)
            # node i separates interval i-1 from interval i. a denser left
            # interval produces negative motion, shrinking the left gap.
            velocity[1:-1] = torch.tanh(
                0.5 * (log_density[1:] - log_density[:-1])
            )

        prior = torch.arange(bands, dtype=torch.float64)
        elastic_drive = (prior - self.coordinates).clamp(-1.0, 1.0)
        velocity.add_(float(elastic) * elastic_drive)

        beta = self.config.beta0 * torch.pow(
            torch.as_tensor(self.config.phi, dtype=torch.float64),
            -self.coordinates,
        )
        # rate is measured per REFERENCE_TOKENS of lived source time. the
        # local factor limits slow coordinates by their integration rate,
        # while the linear elapsed term keeps dynamics invariant to batch
        # partition and controller quantum size.
        adiabatic = (
            float(elapsed_tokens) / self.REFERENCE_TOKENS
        ) * (self.REFERENCE_TOKENS * beta).clamp(max=1.0)
        velocity.mul_(float(rate) * adiabatic)
        velocity[0] = 0.0
        velocity[-1] = 0.0

        gap_velocity = velocity[1:] - velocity[:-1]
        maximum_gap = torch.full_like(gaps, self.MAX_GAP)
        maximum_gap[-1] = min(
            self.MAX_GAP,
            self._resolution_wall() - float(self.coordinates[-1]),
        )
        feasible = 1.0
        shrinking = gap_velocity < 0.0
        if bool(shrinking.any()):
            feasible = min(
                feasible,
                float(((gaps[shrinking] - self.MIN_GAP)
                       / -gap_velocity[shrinking]).min()),
            )
        growing = gap_velocity > 0.0
        if bool(growing.any()):
            feasible = min(
                feasible,
                float(((maximum_gap[growing] - gaps[growing])
                       / gap_velocity[growing]).min()),
            )
        feasible = max(0.0, min(1.0, feasible))
        self.coordinates = self.coordinates + feasible * velocity
        self.coordinates[0] = 0.0
        self.coordinates[-1] = float(bands - 1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "coordinates": self.coordinates.clone(),
            "monitor_density": self.monitor_density.clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        config: TraceBranchConfig,
        state: dict[str, Any],
    ) -> "TemporalLattice":
        return cls(
            config,
            state["coordinates"],
            monitor_density=state.get("monitor_density"),
        )


class TraceState:
    """the exact, nonlearned symbol-by-band present.

    the trace is bit-exact under the lattice trajectory it has lived through.
    when the lattice deforms, previously accumulated mass is carried forward
    from the old coordinates and new evidence follows the new recurrence.
    the resulting transition bias decays with each band's own time constant."""

    def __init__(
        self,
        config: TraceBranchConfig,
        *,
        lattice: "TemporalLattice | None" = None,
        trace: torch.Tensor | None = None,
        prior_counts: torch.Tensor | None = None,
        last_token: int = -1,
        tokens_seen: int = 0,
        source_bytes_seen: float = 0.0,
    ) -> None:
        self.config = config
        self.lattice = lattice if lattice is not None else TemporalLattice(config)

        if prior_counts is None:
            prior_counts = torch.full(
                (config.vocab_size,),
                float(config.prior_smoothing),
                dtype=torch.float64,
            )
        self.prior_counts = prior_counts.detach().to(
            device="cpu", dtype=torch.float64
        ).clone()
        self._validate_counts()

        if trace is None:
            prior = self.prior()
            trace = prior[:, None].expand(
                config.vocab_size, config.bands
            ).clone()
        self.trace = trace.detach().to(device="cpu", dtype=torch.float64).clone()
        self.last_token = int(last_token)
        self.tokens_seen = int(tokens_seen)
        self.source_bytes_seen = float(source_bytes_seen)
        self._validate_trace()

    def _validate_counts(self) -> None:
        if self.prior_counts.shape != (self.config.vocab_size,):
            raise ValueError("prior count shape does not match vocabulary")
        if not torch.isfinite(self.prior_counts).all():
            raise ValueError("prior counts must be finite")
        if float(self.prior_counts.min()) <= 0.0:
            raise ValueError("prior counts must remain positive")

    def _validate_trace(self) -> None:
        expected = (self.config.vocab_size, self.config.bands)
        if self.trace.shape != expected:
            raise ValueError(f"trace shape must be {expected}")
        if not torch.isfinite(self.trace).all():
            raise ValueError("trace state must be finite")
        if float(self.trace.min()) < -1e-12:
            raise ValueError("trace state cannot contain negative mass")
        if not (-1 <= self.last_token < self.config.vocab_size):
            raise ValueError("last token is outside the vocabulary")

    @property
    def retention(self) -> torch.Tensor:
        _, retention, _, _ = self.lattice.derive()
        return retention

    @property
    def injection(self) -> torch.Tensor:
        _, _, injection, _ = self.lattice.derive()
        return injection

    def prior(self) -> torch.Tensor:
        return self.prior_counts / self.prior_counts.sum()

    def clone(self) -> "TraceState":
        return TraceState(
            self.config,
            lattice=self.lattice,
            trace=self.trace,
            prior_counts=self.prior_counts,
            last_token=self.last_token,
            tokens_seen=self.tokens_seen,
            source_bytes_seen=self.source_bytes_seen,
        )

    def advanced(
        self,
        token_ids: torch.Tensor | list[int],
        *,
        source_bytes: float = 0.0,
        update_prior: bool = True,
        prior_weights: torch.Tensor | list[float] | None = None,
    ) -> "TraceState":
        """return the exact state after a contiguous token block."""

        ids = torch.as_tensor(
            token_ids, dtype=torch.long, device="cpu"
        ).flatten()
        if ids.numel() == 0:
            return self.clone()
        if int(ids.min()) < 0 or int(ids.max()) >= self.config.vocab_size:
            raise ValueError("token id is outside the vocabulary")

        count = int(ids.numel())
        _, retention, injection, _ = self.lattice.derive()
        exponent = torch.arange(
            count - 1, -1, -1, dtype=torch.float64
        ).unsqueeze(1)
        weights = injection.unsqueeze(0) * torch.pow(
            retention.unsqueeze(0), exponent
        )
        updated_trace = self.trace * torch.pow(retention, count)
        updated_trace.index_add_(0, ids, weights)

        updated_counts = self.prior_counts.clone()
        if update_prior:
            weights = None
            if prior_weights is not None:
                weights = torch.as_tensor(
                    prior_weights, dtype=torch.float64, device="cpu"
                ).flatten()
                if weights.shape != ids.shape:
                    raise ValueError("prior weights must match token ids")
                if not torch.isfinite(weights).all() or float(weights.min()) < 0:
                    raise ValueError("prior weights must be finite and nonnegative")
            updated_counts.add_(
                torch.bincount(
                    ids,
                    weights=weights,
                    minlength=self.config.vocab_size,
                ).to(torch.float64)
            )

        return TraceState(
            self.config,
            lattice=self.lattice,
            trace=updated_trace,
            prior_counts=updated_counts,
            last_token=int(ids[-1]),
            tokens_seen=self.tokens_seen + count,
            source_bytes_seen=self.source_bytes_seen + float(source_bytes),
        )

    def advance_(
        self,
        token_ids: torch.Tensor | list[int],
        *,
        source_bytes: float = 0.0,
        update_prior: bool = True,
        prior_weights: torch.Tensor | list[float] | None = None,
    ) -> None:
        updated = self.advanced(
            token_ids,
            source_bytes=source_bytes,
            update_prior=update_prior,
            prior_weights=prior_weights,
        )
        self.trace = updated.trace
        self.prior_counts = updated.prior_counts
        self.last_token = updated.last_token
        self.tokens_seen = updated.tokens_seen
        self.source_bytes_seen = updated.source_bytes_seen

    def priors_before(
        self,
        token_ids: torch.Tensor | list[int],
        rows: torch.Tensor | list[int] | None = None,
        *,
        prior_weights: torch.Tensor | list[float] | None = None,
    ) -> torch.Tensor:
        """return the exact empirical prior before each requested token."""

        ids = torch.as_tensor(
            token_ids, dtype=torch.long, device="cpu"
        ).flatten()
        if rows is None:
            rows_cpu = torch.arange(ids.numel(), dtype=torch.long)
        else:
            rows_cpu = torch.as_tensor(
                rows, dtype=torch.long, device="cpu"
            ).flatten()
        if rows_cpu.numel() == 0:
            return torch.empty(
                (0, self.config.vocab_size), dtype=torch.float64
            )
        if int(rows_cpu.min()) < 0 or int(rows_cpu.max()) >= ids.numel():
            raise ValueError("prior rows are outside the token block")
        if rows_cpu.numel() > 1 and not bool(
            torch.all(rows_cpu[1:] > rows_cpu[:-1])
        ):
            raise ValueError("prior rows must be strictly increasing")

        if prior_weights is None:
            weights = torch.ones(ids.shape, dtype=torch.float64)
        else:
            weights = torch.as_tensor(
                prior_weights, dtype=torch.float64, device="cpu"
            ).flatten()
            if weights.shape != ids.shape:
                raise ValueError("prior weights must match token ids")
            if not torch.isfinite(weights).all() or float(weights.min()) < 0:
                raise ValueError("prior weights must be finite and nonnegative")

        counts = self.prior_counts.clone()
        priors = []
        cursor = 0
        for row in rows_cpu.tolist():
            if row > cursor:
                counts.add_(torch.bincount(
                    ids[cursor:row],
                    weights=weights[cursor:row],
                    minlength=self.config.vocab_size,
                ))
            priors.append(counts / counts.sum())
            cursor = row
        return torch.stack(priors)

    def state_dict(self) -> dict[str, Any]:
        return {
            "trace": self.trace.clone(),
            "prior_counts": self.prior_counts.clone(),
            "last_token": self.last_token,
            "tokens_seen": self.tokens_seen,
            "source_bytes_seen": self.source_bytes_seen,
        }

    @classmethod
    def from_state_dict(
        cls,
        config: TraceBranchConfig,
        state: dict[str, Any],
        *,
        lattice: "TemporalLattice | None" = None,
    ) -> "TraceState":
        return cls(
            config,
            lattice=lattice,
            trace=state["trace"],
            prior_counts=state["prior_counts"],
            last_token=state.get("last_token", -1),
            tokens_seen=state.get("tokens_seen", 0),
            source_bytes_seen=state.get("source_bytes_seen", 0.0),
        )


def associative_projected_scan(
    start: torch.Tensor,
    centered_inputs: torch.Tensor,
    retention: torch.Tensor,
    injection: torch.Tensor,
) -> torch.Tensor:
    """produce projected states before each input using an affine prefix scan."""

    if centered_inputs.ndim != 2:
        raise ValueError("centered_inputs must have shape [batch, features]")
    if start.ndim != 2:
        raise ValueError("start must have shape [bands, features]")
    batch, features = centered_inputs.shape
    bands, start_features = start.shape
    if features != start_features:
        raise ValueError("projected feature widths do not match")
    if retention.shape != (bands,) or injection.shape != (bands,):
        raise ValueError("lattice shape does not match projected state")
    if batch == 0:
        return start.new_empty((0, bands, features))

    affine_a = retention.view(1, bands, 1).expand(batch, bands, 1)
    affine_b = (
        injection.view(1, bands, 1)
        * centered_inputs.view(batch, 1, features)
    )

    offset = 1
    while offset < batch:
        right_a = affine_a[offset:]
        right_b = affine_b[offset:]
        left_a = affine_a[:-offset]
        left_b = affine_b[:-offset]
        affine_a = torch.cat(
            (affine_a[:offset], right_a * left_a), dim=0
        )
        affine_b = torch.cat(
            (
                affine_b[:offset],
                right_a * left_b + right_b,
            ),
            dim=0,
        )
        offset *= 2

    after = affine_a * start.unsqueeze(0) + affine_b
    return torch.cat((start.unsqueeze(0), after[:-1]), dim=0)


def selected_projected_scan(
    start: torch.Tensor,
    centered_inputs: torch.Tensor,
    retention: torch.Tensor,
    injection: torch.Tensor,
    prediction_rows: torch.Tensor,
    *,
    regular_stride: int | None = None,
) -> torch.Tensor:
    """produce regularly decimated states without materializing skipped states."""

    rows = prediction_rows.to(
        device=centered_inputs.device, dtype=torch.long
    ).flatten()
    if rows.numel() == 0:
        return start.new_empty((0, *start.shape))
    if int(rows.min()) < 0 or int(rows.max()) >= centered_inputs.shape[0]:
        raise ValueError("prediction rows are outside the input block")
    if regular_stride is None:
        if rows.numel() == 1:
            stride = 1
        else:
            gaps = rows[1:] - rows[:-1]
            stride = int(gaps[0])
        if (
            stride < 1
            or (
                rows.numel() > 1
                and not bool(torch.all(gaps == stride))
            )
        ):
            return associative_projected_scan(
                start, centered_inputs, retention, injection
            ).index_select(0, rows)
    else:
        stride = max(1, int(regular_stride))
        if rows.numel() > 1 and not bool(
            torch.all(rows[1:] - rows[:-1] == stride)
        ):
            raise ValueError("prediction rows do not match regular stride")

    first_row = int(rows[0])
    if first_row:
        prefix_inputs = centered_inputs[:first_row]
        prefix_exponents = torch.arange(
            first_row - 1,
            -1,
            -1,
            device=centered_inputs.device,
            dtype=retention.dtype,
        )
        prefix_weights = injection.unsqueeze(1) * torch.pow(
            retention.unsqueeze(1), prefix_exponents.unsqueeze(0)
        )
        first_state = (
            torch.pow(retention, first_row).unsqueeze(1) * start
            + torch.einsum("ks,sd->kd", prefix_weights, prefix_inputs)
        )
    else:
        first_state = start
    if rows.numel() == 1:
        return first_state.unsqueeze(0)

    # each block is one exact affine transition between retained prediction
    # points. einsum forms only the retained [row, band, feature] states;
    # skipped per-token states never exist as a tensor.
    transitions = int(rows.numel()) - 1
    block_inputs = centered_inputs[
        first_row : first_row + transitions * stride
    ].reshape(
        transitions, stride, centered_inputs.shape[1]
    )
    exponents = torch.arange(
        stride - 1,
        -1,
        -1,
        device=centered_inputs.device,
        dtype=retention.dtype,
    )
    weights = injection.unsqueeze(1) * torch.pow(
        retention.unsqueeze(1), exponents.unsqueeze(0)
    )
    affine_b = torch.einsum("ks,msd->mkd", weights, block_inputs)
    affine_a = torch.pow(retention, stride).view(1, -1, 1).expand(
        transitions, -1, 1
    )

    offset = 1
    while offset < transitions:
        right_a = affine_a[offset:]
        right_b = affine_b[offset:]
        left_a = affine_a[:-offset]
        left_b = affine_b[:-offset]
        affine_a = torch.cat(
            (affine_a[:offset], right_a * left_a), dim=0
        )
        affine_b = torch.cat(
            (
                affine_b[:offset],
                right_a * left_b + right_b,
            ),
            dim=0,
        )
        offset *= 2

    after = affine_a * first_state.unsqueeze(0) + affine_b
    return torch.cat((first_state.unsqueeze(0), after), dim=0)


class ScaleBlock(nn.Module):
    """shared local transport and low-rank mixing on log lifetime."""

    def __init__(
        self,
        symbol_dim: int,
        rank: int,
        kernel_size: int,
        scale_coordinates: torch.Tensor,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.depthwise = nn.Conv1d(
            symbol_dim,
            symbol_dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=symbol_dim,
            bias=False,
        )
        self.down = nn.Linear(symbol_dim, rank, bias=False)
        self.up = nn.Linear(rank, symbol_dim, bias=False)
        self.condition = nn.Linear(3, 2 * symbol_dim, bias=False)
        self.time_gate = nn.Parameter(torch.tensor(0.1))
        self.symbol_gate = nn.Parameter(torch.tensor(0.1))
        self.register_buffer(
            "scale_coordinates", scale_coordinates.clone(), persistent=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.depthwise.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self.depthwise.kernel_size[0]),
        )
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.condition.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        conditioned = self.condition(self.scale_coordinates)
        time_condition, symbol_condition = conditioned.chunk(2, dim=-1)

        time_branch = self.depthwise(
            value.transpose(1, 2)
        ).transpose(1, 2)
        time_branch = time_branch * (
            1.0 + 0.25 * torch.tanh(time_condition).unsqueeze(0)
        )
        value = value + torch.tanh(self.time_gate) * time_branch

        inverse_rms = torch.rsqrt(
            value.square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        normalized = value * inverse_rms
        symbol_branch = (
            self.up(F.silu(self.down(normalized))) / inverse_rms
        )
        symbol_branch = symbol_branch * (
            1.0 + 0.25 * torch.tanh(symbol_condition).unsqueeze(0)
        )
        return value + torch.tanh(self.symbol_gate) * symbol_branch


class ScaleEncoder(nn.Module):
    """organize the projected trace field into fixed-size summaries."""

    def __init__(self, config: TraceBranchConfig) -> None:
        super().__init__()
        _, _, _, lifetime = temporal_lattice(
            config, dtype=torch.float32
        )
        coordinates = self._lattice_coordinates(config.bands, lifetime)

        self.blocks = nn.ModuleList(
            ScaleBlock(
                config.symbol_dim,
                config.scale_rank,
                config.scale_kernel,
                coordinates,
                config.epsilon,
            )
            for _ in range(config.scale_depth)
        )
        self.summary_filter = nn.Conv1d(
            config.symbol_dim,
            config.symbol_dim,
            3,
            padding=1,
            groups=config.symbol_dim,
            bias=False,
        )
        nn.init.dirac_(
            self.summary_filter.weight, groups=config.symbol_dim
        )
        self.register_buffer(
            "pool_weights",
            self._pool_weights(config.bands, config.summaries),
            persistent=False,
        )

    @staticmethod
    def _lattice_coordinates(
        bands: int, lifetime: torch.Tensor
    ) -> torch.Tensor:
        log_lifetime = torch.log(lifetime)
        if bands > 1:
            position = (
                2.0
                * (log_lifetime - log_lifetime[0])
                / (log_lifetime[-1] - log_lifetime[0])
                - 1.0
            )
        else:
            position = torch.zeros_like(log_lifetime)
        left_boundary = 1.0 / (
            1.0 + torch.arange(bands, dtype=torch.float32)
        )
        right_boundary = torch.flip(left_boundary, dims=(0,))
        return torch.stack(
            (position, left_boundary, right_boundary), dim=-1
        )

    @torch.no_grad()
    def refresh_lattice(self, lifetime: torch.Tensor) -> None:
        """re-anchor scale conditioning after the lattice deforms."""

        coordinates = self._lattice_coordinates(
            int(lifetime.numel()), lifetime.to(torch.float32)
        )
        for block in self.blocks:
            block.scale_coordinates.copy_(
                coordinates.to(block.scale_coordinates.device)
            )

    @staticmethod
    def _pool_weights(bands: int, summaries: int) -> torch.Tensor:
        positions = torch.arange(bands, dtype=torch.float32)
        centers = torch.linspace(0.0, float(bands - 1), summaries)
        spacing = max(1.0, float(bands) / float(summaries))
        distance = positions.unsqueeze(0) - centers.unsqueeze(1)
        weights = torch.exp(-0.5 * (distance / spacing).square())
        return weights / weights.sum(dim=1, keepdim=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        value = self.summary_filter(
            value.transpose(1, 2)
        ).transpose(1, 2)
        return torch.einsum("hk,bkd->bhd", self.pool_weights, value)


class ResidualSwiGLU(nn.Module):
    """a magnitude-aware serial composition block."""

    def __init__(
        self,
        model_dim: int,
        expansion_dim: int,
        residual_scale: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.residual_scale = float(residual_scale)
        self.gate = nn.Linear(model_dim, expansion_dim, bias=False)
        self.value = nn.Linear(model_dim, expansion_dim, bias=False)
        self.out = nn.Linear(expansion_dim, model_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.out.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(
            value.square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        normalized = value * inverse_rms
        branch = self.out(
            F.silu(self.gate(normalized)) * self.value(normalized)
        )
        return value + self.residual_scale * branch / inverse_rms


@dataclass
class MixerOutput:
    logits: torch.Tensor
    context: torch.Tensor
    branch: torch.Tensor
    projected: torch.Tensor | None = None
    direct: torch.Tensor | None = None


class TraceBranchMixer(nn.Module):
    """estimate the legal next branch from the exact represented present."""

    species = "soma_v14_adaptive_trace_branch_mixer"

    def __init__(self, config: TraceBranchConfig) -> None:
        super().__init__()
        self.config = config
        _, retention, injection, _ = temporal_lattice(
            config, dtype=torch.float32
        )
        self.register_buffer("retention", retention)
        self.register_buffer("injection", injection)
        self.embedding = nn.Embedding(
            config.vocab_size, config.symbol_dim
        )
        self.scale_encoder = ScaleEncoder(config)
        context_width = (config.summaries + 1) * config.symbol_dim
        self.context = nn.Linear(
            context_width, config.model_dim, bias=False
        )
        residual_scale = 1.0 / math.sqrt(config.trunk_depth)
        self.trunk = nn.ModuleList(
            ResidualSwiGLU(
                config.model_dim,
                config.expansion_dim,
                residual_scale,
                config.epsilon,
            )
            for _ in range(config.trunk_depth)
        )
        self.branch = nn.Linear(
            config.model_dim, config.symbol_dim, bias=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self.config.symbol_dim),
        )
        nn.init.xavier_uniform_(self.context.weight)
        nn.init.zeros_(self.branch.weight)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def refresh_lattice(self, lattice: TemporalLattice) -> None:
        """follow a deformed lattice: scan buffers and scale coordinates.

        the lattice is controller-owned state, not an optimizer parameter;
        the mixer only mirrors it."""

        _, retention, injection, lifetime = lattice.derive(
            dtype=torch.float32
        )
        self.retention.copy_(retention.to(self.retention.device))
        self.injection.copy_(injection.to(self.injection.device))
        self.scale_encoder.refresh_lattice(lifetime)

    def project_state(
        self,
        trace: torch.Tensor,
        prior: torch.Tensor,
        last_token: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = self.embedding.weight.dtype
        device = self.embedding.weight.device
        trace = trace.to(device=device, dtype=dtype)
        prior = prior.to(device=device, dtype=dtype)
        prior_embedding = prior @ self.embedding.weight
        centered_trace = trace - prior.unsqueeze(1)
        projected = centered_trace.transpose(0, 1) @ self.embedding.weight
        if last_token < 0:
            direct = projected.new_zeros(self.config.symbol_dim)
        else:
            direct = self.embedding.weight[int(last_token)] - prior_embedding
        return projected, direct, prior_embedding

    def prepare_batch(
        self,
        trace: torch.Tensor,
        prior: torch.Tensor,
        last_token: int,
        targets: torch.Tensor,
        prediction_rows: torch.Tensor | None = None,
        prediction_stride: int | None = None,
        prediction_priors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected, first_direct, prior_embedding = self.project_state(
            trace, prior, last_token
        )
        targets = targets.to(
            device=self.embedding.weight.device, dtype=torch.long
        )
        embedded_inputs = self.embedding(targets)
        centered_inputs = embedded_inputs - prior_embedding
        scan_start = projected
        scan_inputs = centered_inputs
        direct_inputs = centered_inputs
        first_direct_value = first_direct
        if prediction_priors is not None:
            prediction_priors = torch.as_tensor(
                prediction_priors,
                device=self.embedding.weight.device,
                dtype=self.embedding.weight.dtype,
            )
            expected_rows = (
                targets.numel()
                if prediction_rows is None
                else prediction_rows.numel()
            )
            if prediction_priors.shape != (
                expected_rows, self.config.vocab_size
            ):
                raise ValueError(
                    "prediction priors must match prediction rows and vocabulary"
                )
            scan_start = projected + prior_embedding.unsqueeze(0)
            scan_inputs = embedded_inputs
            direct_inputs = embedded_inputs
            first_direct_value = first_direct + prior_embedding
        if prediction_rows is not None:
            prediction_rows = prediction_rows.to(
                device=centered_inputs.device, dtype=torch.long
            )
            projected_before = selected_projected_scan(
                scan_start,
                scan_inputs,
                self.retention,
                self.injection,
                prediction_rows,
                regular_stride=prediction_stride,
            )
            previous_rows = (prediction_rows - 1).clamp_min(0)
            direct = direct_inputs.index_select(0, previous_rows)
            first = prediction_rows == 0
            direct = torch.where(
                first.unsqueeze(1),
                first_direct_value.unsqueeze(0),
                direct,
            )
        else:
            projected_before = associative_projected_scan(
                scan_start,
                scan_inputs,
                self.retention,
                self.injection,
            )
            if targets.numel() == 1:
                direct = first_direct_value.unsqueeze(0)
            else:
                direct = torch.cat(
                    (
                        first_direct_value.unsqueeze(0),
                        direct_inputs[:-1],
                    ),
                    dim=0,
                )
        if prediction_priors is not None:
            row_prior_embedding = prediction_priors @ self.embedding.weight
            projected_before = (
                projected_before - row_prior_embedding.unsqueeze(1)
            )
            direct = direct - row_prior_embedding
        return projected_before, direct

    def forward_features(
        self,
        projected: torch.Tensor,
        direct: torch.Tensor,
        prior: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | MixerOutput:
        summaries = self.scale_encoder(projected)
        combined = torch.cat(
            (summaries.flatten(start_dim=1), direct), dim=-1
        )
        context = self.context(combined)
        for block in self.trunk:
            context = block(context)
        branch = self.branch(context)
        logits = (
            F.linear(branch, self.embedding.weight)
            / math.sqrt(self.config.symbol_dim)
        )
        prior = prior.to(device=logits.device, dtype=logits.dtype)
        logits = logits + torch.log(
            prior.clamp_min(self.config.epsilon)
        )
        if return_details:
            return MixerOutput(
                logits=logits,
                context=context,
                branch=branch,
                projected=projected,
                direct=direct,
            )
        return logits

    def forward(
        self,
        trace: torch.Tensor,
        prior: torch.Tensor,
        last_token: int,
        targets: torch.Tensor,
        *,
        prediction_rows: torch.Tensor | None = None,
        prediction_stride: int | None = None,
        prediction_priors: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | MixerOutput:
        projected, direct = self.prepare_batch(
            trace,
            prior,
            last_token,
            targets,
            prediction_rows=prediction_rows,
            prediction_stride=prediction_stride,
            prediction_priors=prediction_priors,
        )
        return self.forward_features(
            projected,
            direct,
            prediction_priors if prediction_priors is not None else prior,
            return_details=return_details,
        )

    def advance_projected(
        self,
        projected: torch.Tensor,
        token_id: int,
        prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prior = prior.to(
            device=self.embedding.weight.device,
            dtype=self.embedding.weight.dtype,
        )
        prior_embedding = prior @ self.embedding.weight
        centered = self.embedding.weight[int(token_id)] - prior_embedding
        projected = (
            self.retention.unsqueeze(1) * projected
            + self.injection.unsqueeze(1) * centered.unsqueeze(0)
        )
        return projected, centered
