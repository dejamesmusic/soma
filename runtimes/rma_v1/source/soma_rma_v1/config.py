"""minimal public configuration for soma rma_v1."""

from __future__ import annotations

import math
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path


PHI = (1.0 + math.sqrt(5.0)) / 2.0
# Frontal learning rate as a fraction of the base learning rate.  2.0 diverged
# (cortex 256x2 over a width 256 base), so the dial stops a factor below it.
CORTEX_RATE_OPTIONS = (0.25, 0.5, 1.0)
CORTEX_RATE_CAP = max(CORTEX_RATE_OPTIONS)


# the width rma's dynamics were validated at; wider models scale their updates from it
REFERENCE_WIDTH = 256.0


@dataclass(frozen=True)
class RMAConfig:
    # capacity and temporal extent
    width: int = 256
    bands: int = 32
    input_representation: str = "adjacent_difference"

    # optional frontal composition above exact and temporal bands
    cortex_width: int = 0
    cortex_depth: int = 0
    workspace_tokens: int = 0
    # cortical learning rate relative to the base learning rate
    cortex_rate: float = 1.0
    # exact recent tokens the cortex's sentinel pointer can copy from; used
    # only when a cortex is on (0 disables the pointer)
    readout_window: int = 256
    # a pretrained body soma consults as an expert in its readout ("" for
    # none).  With a body soma reads the body's alphabet and its width is the
    # body's; io2 runs as it does without one.
    body: str = ""

    # full directional movement after one mean-gradient evidence update
    source_gain: float = 0.3

    # one optimizer update integrates this many admitted predictions. source
    # traversal may be much larger when io2 decimates the error stream.
    evidence_quantum: int = 256
    decimation_stride_cap: int = 256
    traversal_gain: float = 1.0
    # Deprecated checkpoint compatibility field.  Residual clocks are part of
    # the predictor's temporal geometry and always follow `bands`.
    residual_bands: int = 0

    # legacy metabolic-controller fields retained only so older checkpoints
    # remain loadable. they no longer affect the learner clock.
    batch_min: int = 8
    batch_max: int = 256

    # source-time metabolism. defaults reproduce the validated controller.
    metabolic_relaxation_tokens: int = 11_520
    surprise_response_tokens: int = 58

    # how the learning rate scales past the validated 256-wide reference: an update moves
    # min(1, 256 / width) ** width_scaling of a 256-wide update. unscaled (0), one update's
    # effect on the logits grows about as width ** 0.6 and models diverge past width ~3000;
    # 0.5 holds it near the 256-wide value (experiments/width_scaling.py). checkpoints saved
    # before this field restore 0, the dynamics they learned under.
    width_scaling: float = 0.5

    @property
    def width_rate(self):
        return min(1.0, REFERENCE_WIDTH / self.width) ** self.width_scaling

    def validate(self):
        if self.width < 1 or self.bands < 4:
            raise ValueError("width must be positive and bands must be at least four")
        if self.input_representation not in {
            "cross_band", "adjacent_difference",
        }:
            raise ValueError("unknown input representation")
        if self.cortex_width < 0 or self.cortex_depth < 0:
            raise ValueError("cortical width and depth cannot be negative")
        if (self.cortex_width == 0) != (self.cortex_depth == 0):
            raise ValueError("cortical width and depth must both be zero or positive")
        if self.workspace_tokens < 0:
            raise ValueError("workspace tokens cannot be negative")
        if self.workspace_tokens and not self.cortex_depth:
            raise ValueError("exact bands require a frontal cortical mixer")
        if self.readout_window < 0:
            raise ValueError("readout_window cannot be negative")
        if not (0.0 < self.cortex_rate <= CORTEX_RATE_CAP):
            raise ValueError(
                f"cortex_rate must be in (0, {CORTEX_RATE_CAP:g}]"
            )
        if self.body:
            from .body import BODY_NAMES
            if self.body not in BODY_NAMES:
                raise ValueError(f"unknown body {self.body!r}")
            if not self.cortex_depth:
                raise ValueError("a body is consulted through the cortex readout")
        if self.source_gain <= 0.0:
            raise ValueError("source_gain must be positive")
        if self.evidence_quantum < 1:
            raise ValueError("evidence_quantum must be positive")
        if self.decimation_stride_cap < 1:
            raise ValueError("decimation_stride_cap must be positive")
        if not math.isfinite(self.traversal_gain) or self.traversal_gain < 0.0:
            raise ValueError("traversal_gain must be finite and non-negative")
        if not (1 <= self.batch_min <= self.batch_max):
            raise ValueError("expected 1 <= batch_min <= batch_max")
        if self.metabolic_relaxation_tokens < 1:
            raise ValueError("metabolic_relaxation_tokens must be positive")
        if self.surprise_response_tokens < 1:
            raise ValueError("surprise_response_tokens must be positive")
        if not math.isfinite(self.width_scaling) or self.width_scaling < 0.0:
            raise ValueError("width_scaling must be finite and non-negative")
        return self

    @property
    def metabolic_range_bands(self):
        return math.log(self.batch_max / self.batch_min, PHI)

    @property
    def controller_bands(self):
        """io2 residual clocks share the predictor's temporal geometry."""
        return self.bands

    @property
    def batch_rise(self):
        # The retained controller advances once per observed-token evidence
        # unit.  This conversion makes a complete cheapward traversal consume
        # the requested number of source tokens.
        return (
            self.metabolic_range_bands * self.batch_min
            / self.metabolic_relaxation_tokens
        )

    @property
    def batch_fall(self):
        observation_quantum = 16
        return (
            self.metabolic_range_bands * observation_quantum
            / self.surprise_response_tokens
        )

    def experiment_namespace(
        self, *, corpora: Path, output: Path, phase_bytes: int,
        eval_bytes: int, device: str, seed: int,
    ):
        self.validate()
        reference_batch = 64
        baseline_horizon = 8192.0
        initial_batch = self.evidence_quantum
        return Namespace(
            corpora=Path(corpora), output=Path(output),
            conditions=["global_clocked"], bands=self.bands,
            controller_bands=self.controller_bands,
            input_representation=self.input_representation,
            token_dim=self.width, aperture_width=self.width,
            language_depth=0, language_expansion=2,
            language_residual_scale=0.1,
            cortex_width=self.cortex_width,
            cortex_depth=self.cortex_depth,
            workspace_tokens=self.workspace_tokens,
            cortex_rate=self.cortex_rate,
            readout_window=self.readout_window,
            cortex_expansion=4,
            batch=initial_batch, batch_mode="fixed",
            batch_controller="io2", batch_min=self.batch_min,
            batch_max=self.batch_max, batch_rise=self.batch_rise,
            batch_fall=self.batch_fall, batch_surprise_threshold=0.05,
            batch_loss_ema=0.01, batch_fast_loss_ema=0.2,
            batch_utility_ema=0.1, batch_utility_baseline_ema=0.005,
            batch_utility_threshold=0.0, batch_utility_relative=False,
            batch_controller_warmup=round(1.5 * baseline_horizon),
            batch_residual_baseline_horizon=baseline_horizon,
            batch_residual_threshold=0.10,
            batch_residual_reduction="l2",
            batch_residual_observation_quantum=16,
            batch_lr_compensation="none",
            batch_learning_law="source_normalized",
            batch_reference=reference_batch,
            phase_bytes=int(phase_bytes), eval_bytes=int(eval_bytes),
            # Residual energy remains a causal, reversible plasticity
            # temperature.  Batch metabolism controls evidence resolution;
            # this amplitude independently anneals movement per source token.
            lens_stride=1, io2_mode="full", io2_error_space="token",
            body_gate=bool(self.body),
            residual_kernel="dense", io2_spectrum="proper",
            io2_sampling="corrected", io2_batch_map="spectral_frequency",
            io2_stride_cap=self.decimation_stride_cap,
            io2_decimation_gain=self.traversal_gain,
            io2_controller_quantum=float(self.evidence_quantum),
            lr=self.source_gain,
            mixer_lr=0.01, schedule="two_speed", slow_coefficient=None,
            dictionary_clock="fast", directional=False,
            match_update_norm=True, parameter_geometry="matrix_sphere",
            bias_ceiling=None, optimizer_kernel="auto",
            device=device, seed=int(seed),
        )

    def public_dict(self):
        return asdict(self)
