"""continual text runtime and checkpoint boundary for soma rma v1."""

from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from . import SPECIES
from .config import PHI, RMAConfig
from .engine import CorticalContext, Experiment, RESIDUAL_HISTORY_LENGTH
from .protocol import TokenCorpus
from .tokenizer import ReferenceTokenizer, TOKENIZER_NAME



RUNTIME_ID = "soma.rma_v1"
RUNTIME_VERSION = "1.8.1"
CHECKPOINT_FORMAT = "soma-rma-v1-pytorch"
CHECKPOINT_VERSION = 1
TEXT_PROTOCOL_ID = "soma-text-turn-v1"
SOURCE_SIGNATURE_BYTES = 64 * 1024
DISK_RESERVE_BYTES = 512 * 1024**2
EPS = 1e-8
# Upper bound on one source block (v14).  Blocks are otherwise sized by the
# predictions an update still needs; this binds only at wide strides.
MAX_SOURCE_BLOCK_TOKENS = 8192


def resolve_device(value="auto") -> torch.device:
    if isinstance(value, torch.device):
        return value
    name = str(value).lower()
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def _hash_range(path: Path, start: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("corpus is shorter than its saved cursor")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _sign_corpus_state(state: dict) -> dict:
    result = dict(state)
    path = Path(result["path"]).expanduser().resolve()
    position = int(result["file_position"])
    if path.stat().st_size < position:
        raise ValueError("corpus is shorter than its saved cursor")
    prefix_length = min(SOURCE_SIGNATURE_BYTES, int(result["file_size"]))
    window_start = max(0, position - SOURCE_SIGNATURE_BYTES)
    result["signature"] = {
        "version": 1,
        "prefix_length": prefix_length,
        "prefix_sha256": _hash_range(path, 0, prefix_length),
        "window_start": window_start,
        "window_end": position,
        "window_sha256": _hash_range(path, window_start, position - window_start),
    }
    return result


def _corpus_matches(path: str | Path, state: dict) -> bool:
    signature = state.get("signature", {})
    if not isinstance(signature, dict) or signature.get("version") != 1:
        return False
    path = Path(path).expanduser().resolve()
    try:
        return (
            path.stat().st_size >= int(state["file_position"])
            and _hash_range(path, 0, int(signature["prefix_length"]))
            == signature["prefix_sha256"]
            and _hash_range(
                path,
                int(signature["window_start"]),
                int(signature["window_end"]) - int(signature["window_start"]),
            ) == signature["window_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _tensor_bytes(value, seen=None) -> int:
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item, seen) for item in value)
    return 0


@dataclass
class BatchMetrics:
    nll: float
    loss_bytes: float
    loss_tokens: float
    top1: float
    top5: float
    source_bytes: int
    tokens: int
    batch: int
    learning_rate: float
    residual_surprise: float

    @property
    def bpb(self) -> float:
        return self.nll / max(EPS, self.loss_bytes * math.log(2.0))


@dataclass
class EvaluationMetrics:
    bpb: float
    token_loss: float
    top1: float
    top5: float
    source_bytes: int
    tokens: int


_CONTROLLER_SCALARS = (
    "io2_ratio", "io2_target_band", "io2_decimation_band", "io2_phase",
    "io2_stride_fraction_phase", "batch_loss_ema", "batch_fast_loss_ema",
    "batch_surprise", "fast_utility", "fast_utility_ema",
    "fast_utility_baseline", "batch_controller_tokens", "residual_surprise",
    "residual_surprise_l2", "residual_surprise_topk",
    "residual_surprise_sum", "residual_surprise_max",
    "residual_surprise_count", "batch_fraction_phase", "io2_ratio_sum",
    "io2_center_sum", "io2_peak_sum", "error_tokens_seen",
    "io2_excess_center", "io2_excess_strength", "io2_stride_sum",
    "io2_observations", "io2_updates", "dynamic_batch_sum",
    "dynamic_batch_count", "dynamic_batch_min", "dynamic_batch_max",
    "optimizer_prediction_mass", "optimizer_loss_mass", "optimizer_source_tokens",
    "controller_token_nll", "controller_prediction_mass",
    "controller_loss_token_mass", "controller_loss_bytes",
)


class SOMA:
    """application-facing rma learner with an exact causal state boundary."""

    species = SPECIES

    def __init__(
        self,
        config: RMAConfig | None = None,
        *,
        device="auto",
        tokenizer_path=None,
        seed=7,
        description="",
    ) -> None:
        self.config = (config or RMAConfig()).validate()
        self.device = resolve_device(device)
        self.tokenizer = ReferenceTokenizer(tokenizer_path)
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        args = self.config.experiment_namespace(
            corpora=Path("."), output=Path("rma.json"), phase_bytes=0,
            eval_bytes=0, device=str(self.device), seed=int(seed),
        )
        self.engine = Experiment(args, "global_clocked", self.tokenizer)
        if self.engine.model.vocab != self.tokenizer.vocab_size:
            raise ValueError("model and tokenizer vocabularies differ")
        self.description = str(description)
        self.checkpoint_id = uuid.uuid4().hex
        self.revision_id = uuid.uuid4().hex
        self.parent_revision_id = None
        self.root_lineage_id = self.checkpoint_id
        self.revision_history = []
        self.forked_from_checkpoint_id = None
        self.forked_from_revision_id = None
        self.created_at = time.time()
        self._revision_published = False
        self._loaded_checkpoint_path: Path | None = None
        self._source_cursor: dict | None = None
        self._last_metrics: BatchMetrics | None = None
        self._last_logits: torch.Tensor | None = None
        self._workspace_ids: list[int] = []
        self._pending_stride: int | None = None
        self._pending_stride_phase = 0.0

    @property
    def net(self):
        return self.engine.model

    @property
    def bytes_seen(self):
        return self.engine.bytes_seen

    @property
    def tokens_seen(self):
        return self.engine.tokens_seen

    @property
    def batch_size(self):
        return self.config.evidence_quantum

    def _source_block_tokens(self) -> int:
        """source block holding the predictions the next update still needs.

        The affine trace scan costs the same however many tokens separate
        predictions, so a block is bounded by evidence, with a source cap
        (as in v14) that only binds at the widest strides.
        """
        engine = self.engine
        self._pending_stride_phase = engine.io2_stride_fraction_phase
        stride = max(1, int(engine._io2_stride()))
        self._pending_stride = stride
        remaining = max(
            1.0,
            self.config.evidence_quantum
            - float(engine.optimizer_prediction_mass),
        )
        predictions = min(
            int(math.ceil(remaining)), self.config.evidence_quantum
        )
        # The remaining evidence is counted in predictions, one per stride
        # tokens.  Size the block to end on the last prediction the update
        # still needs, so blocks do not shrink as the quantum fills.
        first_row = (-int(engine.tokens_seen)) % stride
        needed = first_row + (predictions - 1) * stride + 1
        return max(1, min(needed, MAX_SOURCE_BLOCK_TOKENS))

    def _discard_pending_stride(self) -> None:
        if self._pending_stride is not None:
            self.engine.io2_stride_fraction_phase = self._pending_stride_phase
            self._pending_stride = None

    def params(self) -> int:
        return sum(parameter.numel() for parameter in self.net.parameters())

    def reconfigure_dynamics(self, config: RMAConfig) -> None:
        """Apply editable runtime dynamics without rebuilding learned state.

        Width and temporal bands define the checkpoint tensors and therefore
        cannot change in place.  The remaining public controls are dynamics:
        they are expected to take effect when a checkpoint is continued.
        """
        config = config.validate()
        if config.width != self.config.width or config.bands != self.config.bands:
            raise ValueError(
                "checkpoint width and temporal bands cannot change in continue mode"
            )
        if config.input_representation != self.config.input_representation:
            raise ValueError(
                "input representation can only be chosen for a new model"
            )
        if (
            config.cortex_width != self.config.cortex_width
            or config.cortex_depth != self.config.cortex_depth
            or config.workspace_tokens != self.config.workspace_tokens
            or config.readout_window != self.config.readout_window
        ):
            raise ValueError(
                "cortical shape and workspace can only be chosen for a new model"
            )

        engine = self.engine
        if config.controller_bands != self.config.controller_bands:
            previous_bank = engine.error_bank
            replacement = type(previous_bank)(
                config.controller_bands,
                previous_bank.trace.shape[1],
                PHI,
            )
            overlap = min(
                previous_bank.trace.shape[0], replacement.trace.shape[0]
            )
            replacement.trace[:overlap].copy_(
                previous_bank.trace[:overlap]
            )
            engine.error_bank = replacement
            engine.fused_error_bank = None
            engine.residual_spectrum_baseline = None
            engine.io2_whitened_spectrum = []
            engine.io2_excess_spectrum = []
        self.config = config
        args = engine.args
        args.cortex_rate = config.cortex_rate
        engine.learner.cortex_gain = config.cortex_rate
        args.lr = config.source_gain * config.width_rate
        args.io2_mode = "full"
        args.batch_mode = "fixed"
        args.batch_controller = "io2"
        args.batch = config.evidence_quantum
        args.io2_stride_cap = config.decimation_stride_cap
        args.io2_decimation_gain = config.traversal_gain
        args.io2_controller_quantum = float(config.evidence_quantum)
        engine.io2_decimation_band = max(
            0.0,
            min(math.log(config.decimation_stride_cap, PHI),
                float(engine.io2_decimation_band)),
        )

    def reset_residual_controller(self) -> None:
        """Clear disposable io2 measurements without changing model memory."""
        engine = self.engine
        if engine.fused_error_bank is not None:
            engine.fused_error_bank.traces.zero_()
        engine.error_bank.trace.zero_()
        engine.residual_spectrum_baseline = None
        engine.io2_ratio = 1.0
        engine.io2_target_band = 0.0
        engine.io2_decimation_band = 0.0
        engine.io2_phase = 0.0
        engine.io2_stride_fraction_phase = 0.0
        engine.residual_surprise = 0.0
        engine.residual_surprise_l2 = 0.0
        engine.residual_surprise_topk = 0.0
        engine.residual_surprise_sum = 0.0
        engine.residual_surprise_max = 0.0
        engine.residual_surprise_count = 0
        engine.residual_surprise_history = deque(maxlen=RESIDUAL_HISTORY_LENGTH)
        engine.residual_surprise_l2_history = deque(maxlen=RESIDUAL_HISTORY_LENGTH)
        engine.residual_surprise_topk_history = deque(maxlen=RESIDUAL_HISTORY_LENGTH)
        engine.io2_ratio_sum = 0.0
        engine.io2_center_sum = 0.0
        engine.io2_peak_sum = 0.0
        engine.io2_excess_center = 0.0
        engine.io2_excess_strength = 0.0
        engine.io2_whitened_spectrum = []
        engine.io2_excess_spectrum = []
        engine.io2_stride_sum = 0.0
        engine.io2_observations = 0
        engine.io2_updates = 0
        engine.error_tokens_seen = 0
        engine.batch_controller_tokens = 0
        engine.optimizer_prediction_mass = 0.0
        engine.optimizer_loss_mass = 0.0
        engine.optimizer_source_tokens = 0.0
        engine.controller_token_nll = 0.0
        engine.controller_prediction_mass = 0.0
        engine.controller_loss_token_mass = 0.0
        engine.controller_loss_bytes = 0.0
        engine._control_prediction = None
        engine._control_loss = None
        self._pending_stride = None
        for parameter in self.net.parameters():
            parameter.grad = None

    def _history_windows(self, ids, rows=None):
        width = int(self.config.workspace_tokens)
        if width <= 0:
            return None
        ids = np.asarray(ids, dtype=np.int64)
        rows = (
            np.arange(len(ids), dtype=np.int64)
            if rows is None else np.asarray(rows, dtype=np.int64)
        )
        prefix = np.asarray(self._workspace_ids, dtype=np.int64)
        combined = np.concatenate((prefix, ids))
        positions = len(prefix) + rows
        indices = positions[:, None] + np.arange(-width, 0)[None, :]
        history = np.full(indices.shape, -1, dtype=np.int64)
        valid = indices >= 0
        history[valid] = combined[indices[valid]]
        return torch.from_numpy(history).to(self.device)

    def _current_history(self):
        width = int(self.config.workspace_tokens)
        if width <= 0:
            return None
        values = self._workspace_ids[-width:]
        history = np.full((1, width), -1, dtype=np.int64)
        if values:
            history[0, -len(values):] = values
        return torch.from_numpy(history).to(self.device)

    def _workspace_kept(self) -> int:
        """exact recent ids to retain: the widest reader of recent history."""
        return max(int(self.config.workspace_tokens), self._readout_lags())

    def _readout_lags(self) -> int:
        # The sentinel pointer rides on the cortex and also keys each recent
        # token on the one before it.
        window = int(self.config.readout_window)
        return window + 1 if window and self.config.cortex_depth else 0

    def _recent_grid(self, ids, rows, with_unique=False):
        """[rows, R + 1] ids of the tokens before each row, lag one first.

        With `with_unique`, also the window's distinct ids and each entry's
        index among them, so the pointer projects only tokens present.
        """
        window = self._readout_lags()
        prefix = np.asarray(self._workspace_ids, dtype=np.int64)
        combined = np.concatenate((prefix, np.asarray(ids, dtype=np.int64)))
        positions = len(prefix) + np.asarray(rows, dtype=np.int64)
        indices = positions[:, None] - np.arange(1, window + 1)[None, :]
        grid = np.full(indices.shape, -1, dtype=np.int64)
        valid = indices >= 0
        grid[valid] = combined[indices[valid]]
        if not with_unique:
            return torch.from_numpy(grid).to(self.device)
        unique, inverse = np.unique(grid, return_inverse=True)
        return (
            torch.from_numpy(grid).to(self.device),
            torch.from_numpy(unique).to(self.device),
            torch.from_numpy(inverse.reshape(grid.shape)).to(self.device),
        )

    def _cortical_history(self, ids, rows):
        workspace = self._history_windows(ids, rows)
        if not self._readout_lags():
            return workspace
        recent, unique, inverse = self._recent_grid(ids, rows, with_unique=True)
        return CorticalContext(workspace, recent, unique, inverse)

    def _current_cortical_history(self):
        workspace = self._current_history()
        if not self._readout_lags():
            return workspace
        recent, unique, inverse = self._recent_grid([], [0], with_unique=True)
        return CorticalContext(workspace, recent, unique, inverse)

    def _advance_workspace(self, ids) -> None:
        kept = self._workspace_kept()
        if kept <= 0:
            return
        values = np.asarray(ids, dtype=np.int64).tolist()
        self._workspace_ids = (self._workspace_ids + values)[-kept:]

    def _record_source_cursor(self, reader: TokenCorpus, epoch, epochs) -> None:
        self._source_cursor = {
            "kind": "corpus", "epoch": int(epoch), "epochs": int(epochs),
            "reader": reader.state_dict(),
        }

    def resume_state_for(self, corpus_path) -> dict | None:
        cursor = self._source_cursor
        if not isinstance(cursor, dict) or cursor.get("kind") != "corpus":
            return None
        state = cursor.get("reader")
        if not isinstance(state, dict):
            return None
        requested = Path(corpus_path).expanduser().resolve()
        saved = Path(state.get("path", "")).expanduser().resolve()
        if requested != saved:
            # A continual learner must be free to encounter a new source.
            # Exact verification applies only when resuming the same source.
            return None
        position = int(state.get("start_byte", 0)) + int(state.get("bytes_consumed", 0))
        current_size = requested.stat().st_size
        if position <= 0 or position >= current_size:
            return None
        if not _corpus_matches(requested, state):
            raise ValueError("corpus content differs from the checkpoint cursor")
        result = dict(state)
        result["path"] = str(requested)
        return result

    def _train_batch(
        self, ids, source_bytes, loss_weights, loss_bytes, *, stride=None, update=None,
    ) -> BatchMetrics:
        """one block of learning.  Corpus training lets io2 choose the stride
        and updates at each evidence quantum; chat passes stride 1 and says
        itself when its batch closes."""
        ids = np.asarray(ids, dtype=np.int64)
        weights_np = np.asarray(loss_weights, dtype=np.float32)
        actual_batch = len(ids)
        engine = self.engine
        engine._record_batch_size(actual_batch)
        if stride is None:
            stride = self._pending_stride or engine._io2_stride()
        self._pending_stride = None
        first_row = (-int(engine.tokens_seen)) % stride
        rows = np.arange(first_row, actual_batch, stride, dtype=np.int64)
        if not len(rows):
            with torch.no_grad():
                engine.bank.process_block_select(
                    ids, self.net.input_dictionary, rows=rows
                )
                self._advance_workspace(ids)
            engine.bytes_seen += int(source_bytes)
            engine.tokens_seen += actual_batch
            return BatchMetrics(
                0.0, float(loss_bytes), 0.0, 0.0, 0.0,
                int(source_bytes), actual_batch, actual_batch,
                engine.learner.base_lr, engine.residual_surprise,
            )
        history_ids = self._cortical_history(ids, rows)
        targets = torch.from_numpy(ids[rows]).to(self.device)
        weights = torch.from_numpy(weights_np[rows]).to(self.device)
        # The weights are host data; summing them here avoids a device sync.
        weight_sum = float(weights_np[rows].sum(dtype=np.float64))
        if weight_sum <= 0.0:
            with torch.no_grad():
                engine.bank.process_block_select(
                    ids, self.net.input_dictionary, rows=rows)
                self._advance_workspace(ids)
            engine.bytes_seen += int(source_bytes)
            engine.tokens_seen += actual_batch
            return BatchMetrics(
                0.0, float(loss_bytes), 0.0, 0.0, 0.0,
                int(source_bytes), actual_batch, actual_batch,
                engine.learner.base_lr, engine.residual_surprise,
            )
        # Every source block contributes to the same fixed evidence quantum.
        # Decimation changes the source distance between observations, not the
        # amount of evidence in an optimizer update.
        if engine.optimizer_prediction_mass <= 0.0:
            for parameter in self.net.parameters():
                parameter.grad = None
            # The evidence quantum is a temporal averaging horizon. The
            # accumulated gradient is normalized by its admitted loss mass at
            # the update boundary, so scaling the learning rate by the quantum
            # a second time turns larger horizons into larger rotations.
            engine.learner.base_lr = (
                self.config.source_gain * self.config.width_rate * engine.io2_ratio
            )
        with torch.enable_grad():
            features = engine.bank.process_block_select(
                ids, self.net.input_dictionary, rows=rows
            )
            prediction, loss, _ = engine.prediction_and_loss(
                features, targets, weights, history_ids
            )
        self._advance_workspace(ids)
        with torch.no_grad():
            # Reported quantities stay on the device until one transfer after
            # the optimizer work has been queued.
            if engine._last_rank is not None:
                first, within = engine._last_rank == 0, engine._last_rank < 5
            else:
                top = prediction.topk(min(5, prediction.shape[1]), dim=1).indices
                hits = top == targets[:, None]
                first, within = hits[:, 0], hits.any(1)
            statistics = [
                (engine._last_token_nll * weights).sum(),
                (first.float() * weights).sum(),
                (within.float() * weights).sum(),
            ]
            # Plasticity is annealed by the temporal base alone.  With a
            # cortex attached, the emitted loss is reported but the
            # controller reads the base loss.
            if engine._control_token_nll is not None:
                statistics.append((engine._control_token_nll * weights).sum())
            batch_statistics = torch.stack(statistics)
            self._last_logits = (
                engine._last_row.clone() if engine._last_row is not None
                else prediction[-1].detach().clone()
            )
            # kept for the report's view of the model (listening, surprise);
            # nothing is transferred until a report asks for it
            self._view_batch = (
                ids, rows, engine._last_token_nll.detach(), features.detach()[-32:],
            )
        (loss * weights.sum()).backward()
        prediction_mass = float(len(rows))
        engine.optimizer_prediction_mass += prediction_mass
        engine.optimizer_loss_mass += weight_sum
        engine.optimizer_source_tokens += actual_batch
        update_due = (
            engine.optimizer_prediction_mass + 1e-9
            >= self.config.evidence_quantum
        ) if update is None else bool(update)
        if update_due:
            if update is not None:
                engine.learner.base_lr = self._chat_rate()
            engine.learner.step(
                gradient_scale=1.0 / max(engine.optimizer_loss_mass, EPS)
            )
            engine.io2_updates += 1
            engine.optimizer_prediction_mass = 0.0
            engine.optimizer_loss_mass = 0.0
            engine.optimizer_source_tokens = 0.0
        nll, top1_mass, top5_mass, *control = (
            batch_statistics.cpu().double().tolist()
        )
        top1 = top1_mass / weight_sum
        top5 = top5_mass / weight_sum
        engine.controller_token_nll += control[0] if control else nll
        engine.controller_prediction_mass += weight_sum
        full_weight_sum = float(weights_np.sum())
        sampled_mean_nll = nll / max(weight_sum, EPS)
        estimated_nll = sampled_mean_nll * full_weight_sum
        engine.controller_loss_token_mass += full_weight_sum
        engine.controller_loss_bytes += float(loss_bytes)
        control_token_loss = (
            engine.controller_token_nll
            / max(engine.controller_prediction_mass, EPS)
        )
        control_nats_per_byte = (
            control_token_loss * engine.controller_loss_token_mass
            / max(engine.controller_loss_bytes, EPS)
        )
        engine._apply_io2(
            engine._control_prediction, targets, weights,
            control_token_loss, stride, actual_batch,
            control_due=update_due,
            nats_per_byte=control_nats_per_byte,
        )
        if update_due:
            engine.controller_token_nll = 0.0
            engine.controller_prediction_mass = 0.0
            engine.controller_loss_token_mass = 0.0
            engine.controller_loss_bytes = 0.0
        engine.bytes_seen += int(source_bytes)
        engine.tokens_seen += actual_batch
        metrics = BatchMetrics(
            nll=estimated_nll, loss_bytes=float(loss_bytes),
            loss_tokens=full_weight_sum,
            top1=top1, top5=top5, source_bytes=int(source_bytes),
            tokens=actual_batch, batch=actual_batch,
            learning_rate=float(engine.learner.base_lr),
            residual_surprise=float(engine.residual_surprise),
        )
        self._last_metrics = metrics
        return metrics

    def train(
        self,
        corpus_path,
        epochs=1,
        save_every=0,
        save_path="",
        start_byte=0,
        max_bytes=0,
        report_every=1,
        report_callback: Callable[[dict], None] | None = None,
        dream_every_batches=0,
        dream_length=128,
        dream_temperature=0.8,
        dream_callback: Callable[[str, int, int], None] | None = None,
        dream_stateful=False,
        resume_state=None,
        exact_start=False,
        stop_callback: Callable[[], bool] | None = None,
        **_unused,
    ):
        totals = None
        dream_interval = max(0, int(dream_every_batches or 0))
        next_dream = (
            ((self.engine.learner.steps // dream_interval) + 1) * dream_interval
            if dream_interval else None
        )
        dream_pending = False
        for epoch in range(max(1, int(epochs))):
            totals = {"nll": 0.0, "loss_bytes": 0.0, "loss_tokens": 0.0,
                      "correct": 0.0, "top5": 0.0, "source_bytes": 0,
                      "tokens": 0, "batches": 0,
                      "optimizer_batches": 0}
            started = time.time()
            with TokenCorpus(
                corpus_path, self.tokenizer,
                start_byte=(int(start_byte) if epoch == 0 else 0),
                max_bytes=max_bytes,
                resume_state=(resume_state if epoch == 0 else None),
                exact_start=bool(exact_start and epoch == 0),
            ) as reader:
                self._record_source_cursor(reader, epoch, epochs)
                while not (stop_callback and stop_callback()):
                    block = reader.next_block(
                        self._source_block_tokens(),
                        stop_at_newline=dream_pending,
                    )
                    if block is None:
                        # The stride drawn for a block that never arrived
                        # must not leak into the next source or checkpoint.
                        self._discard_pending_stride()
                        break
                    ids, source_bytes, _ = block
                    previous_steps = int(self.engine.learner.steps)
                    metrics = self._train_batch(
                        ids, source_bytes, reader.last_loss_weights,
                        reader.last_loss_bytes,
                    )
                    completed_batches = max(
                        0, int(self.engine.learner.steps) - previous_steps
                    )
                    self._record_source_cursor(reader, epoch, epochs)
                    totals["nll"] += metrics.nll
                    totals["loss_bytes"] += metrics.loss_bytes
                    totals["loss_tokens"] += metrics.loss_tokens
                    totals["correct"] += metrics.top1 * metrics.loss_tokens
                    totals["top5"] += metrics.top5 * metrics.loss_tokens
                    totals["source_bytes"] += source_bytes
                    totals["tokens"] += len(ids)
                    totals["batches"] += 1
                    totals["optimizer_batches"] += completed_batches
                    if next_dream is not None and self.engine.learner.steps >= next_dream:
                        dream_pending = True
                        while next_dream <= self.engine.learner.steps:
                            next_dream += dream_interval
                    if dream_pending and reader.last_block_ended_at_newline:
                        text = self.dream_text(
                            length=dream_length,
                            temperature=dream_temperature,
                            stateful=bool(dream_stateful),
                        )
                        if dream_callback:
                            dream_callback(text, self.engine.learner.steps, self.bytes_seen)
                        dream_pending = False
                    # A physical source block can contain only a fraction of
                    # an accumulated optimizer batch.  Reporting here used to
                    # construct diagnostics repeatedly while the same batch
                    # was still filling.  Emit on completed learning quanta so
                    # cockpit cadence has the same meaning as `updates`.
                    if (
                        report_every
                        and completed_batches
                        and totals["optimizer_batches"] % int(report_every) == 0
                    ):
                        report = self.training_report(
                            totals, reader, started, epoch + 1, int(epochs)
                        )
                        if report_callback:
                            report_callback(report)
                    if save_every and totals["batches"] % int(save_every) == 0:
                        self.save(save_path)
        if save_path:
            self.save(save_path)
        return totals

    def dream_text(self, length=128, temperature=0.8, *, stateful=False) -> str:
        """sample a progress dream, optionally retaining it in temporal state."""
        saved_trace = None if stateful else self.engine.bank.traces.clone()
        saved_tokens = None if stateful else self.engine.tokens_seen
        saved_workspace = None if stateful else list(self._workspace_ids)
        try:
            self._observe_control("dream")
            return self.generate_text(length=length, temperature=temperature)
        finally:
            if saved_trace is not None:
                self.engine.bank.traces.copy_(saved_trace)
                self.engine.tokens_seen = saved_tokens
                self._workspace_ids = saved_workspace

    def training_report(self, totals, reader, started, epoch=1, epochs=1) -> dict:
        elapsed = max(EPS, time.time() - started)
        loss_bytes = max(EPS, float(totals["loss_bytes"]))
        loss_tokens = max(EPS, float(totals["loss_tokens"]))
        engine = self.engine
        report = {
            "epoch": int(epoch), "epochs": int(epochs),
            "pos": int(reader.bytes_consumed), "total": int(reader.total_bytes),
            "absolute_pos": int(reader.position), "absolute_total": int(reader.limit),
            "bpb": float(totals["nll"] / (loss_bytes * math.log(2.0))),
            "token_loss": float(totals["nll"] / loss_tokens),
            "acc": float(100.0 * totals["correct"] / loss_tokens),
            "top5_acc": float(100.0 * totals["top5"] / loss_tokens),
            "bps": float(totals["source_bytes"] / elapsed),
            "tokens_per_second": float(totals["tokens"] / elapsed),
            "session_bytes": int(totals["source_bytes"]),
            "session_tokens": int(totals["tokens"]),
            "life_bytes": int(self.bytes_seen), "life_tokens": int(self.tokens_seen),
            "updates": int(engine.learner.steps),
            "batch": int(self.batch_size),
            # io2's traversal: phi ** band source tokens between trained predictions
            "traversal_band": float(engine.io2_decimation_band),
            "prediction_stride": int(max(1, round(
                PHI ** engine.io2_decimation_band
            ))),
            "traversal_quantum": float(min(
                self.config.decimation_stride_cap,
                PHI ** engine.io2_decimation_band,
            )),
            "traversal_gain": float(self.config.traversal_gain),
            "evidence_quantum": int(self.config.evidence_quantum),
            "evidence_accumulated": float(engine.optimizer_prediction_mass),
            "controller_bands": int(engine.error_bank.n_bands),
            "source_gain": float(self.config.source_gain),
            "plasticity_ratio": float(engine.io2_ratio),
            # what one update moves now: the setting, scaled for width, times plasticity
            "effective_source_gain": float(
                self.config.source_gain * self.config.width_rate * engine.io2_ratio
            ),
            "effective_learning_rate": float(engine.learner.base_lr),
            "norm_match": float(engine.learner.last_norm_match),
            "cortex_gain": float(engine.learner.cortex_gain),
            "cortex_rate": float(self.config.cortex_rate),
            "pointer_window": max(0, self._readout_lags() - 1),
            "pointer_share": (
                float(self.net.recent_readout.last_share)
                if self.net.recent_readout is not None else 0.0
            ),
            "parameters": self.params(),
        }
        report["session_seconds"] = elapsed
        # bits per byte since the previous report (the session's so far on the first)
        mark = getattr(self, "_report_mark", None)
        nll, nll_bytes = float(totals["nll"]), float(totals["loss_bytes"])
        if mark and mark[0] == started and nll_bytes > mark[2]:
            report["recent_bpb"] = (nll - mark[1]) / ((nll_bytes - mark[2]) * math.log(2.0))
        else:
            report["recent_bpb"] = report["bpb"]
        self._report_mark = (started, nll, nll_bytes)
        report["residual_spectrum"] = list(engine.io2_whitened_spectrum)
        report["residual_excess"] = list(engine.io2_excess_spectrum)
        baseline = engine.residual_spectrum_baseline
        report["residual_baseline"] = (
            [] if baseline is None else baseline.detach().float().cpu().tolist()
        )
        if self.device.type == "mps" and hasattr(torch, "mps"):
            report["mps_allocated_gb"] = float(
                torch.mps.current_allocated_memory() / 1e9)
            report["mps_driver_gb"] = float(
                torch.mps.driver_allocated_memory() / 1e9)
        else:
            report["mps_allocated_gb"] = 0.0
            report["mps_driver_gb"] = 0.0
        logits = self._last_logits
        if logits is not None:
            probability = F.softmax(logits.cpu().float(), dim=0)
            top = probability.topk(min(6, len(probability)))
            report["top_predictions"] = [
                {"token": self.tokenizer.diagnostic_piece(int(index)),
                 "probability": float(value)}
                for value, index in zip(top.values, top.indices)
            ]
            report["finite"] = bool(torch.isfinite(logits).all())
        else:
            report["top_predictions"] = []
            report["finite"] = True
        # Reports are emitted at optimizer boundaries, so one batched norm
        # transfer here is cheap and shows whether retraction holds the radii.
        report["maximum_radius_error"] = self._maximum_radius_error()
        report.update(self._model_view_report())
        return report

    @torch.no_grad()
    def _model_view_report(self, tokens: int = 192) -> dict:
        """What the model is doing, for watching a run. Reporting only.

        listening: for each temporal band, the gain the model applies to it:
        the size of that band's contribution to the first layer of every
        aperture stage (weights times the band's features) divided by the size
        of the band's features, averaged over the last batch's sampled rows and
        normalised to shares (averaged over the stages that read each band).
        Slow-band features are orders of magnitude smaller
        than fast ones by construction, so raw contribution mostly shows the
        input's scale; gain shows what the model has learned to amplify. The
        layers' rows live on a fixed-radius sphere, so gain is a budget the
        model redistributes across timescales. listening_drive keeps the raw
        contribution shares.

        surprise: the last tokens of the batch with the bits each one cost where
        the model predicted it (io2 skips rows by its stride; those carry null).
        """
        view = getattr(self, "_view_batch", None)
        if view is None:
            return {}
        ids, rows, nll, sample = view
        net = self.net
        out = {}
        start = max(0, len(ids) - tokens)
        bits = {}
        for row, value in zip(rows.tolist(), (nll.float().cpu() / math.log(2.0)).tolist()):
            if row >= start:
                bits[row] = round(float(value), 3)
        out["surprise"] = [
            [self._display_piece(int(ids[index])), bits.get(index)]
            for index in range(start, len(ids))
        ]
        if len(sample):
            token_dim, bands = net.token_dim, net.bands
            features = sample.float().reshape(len(sample), token_dim, bands)
            drive = torch.zeros(bands, device=features.device)
            gain = torch.zeros(bands, device=features.device)
            for index, (cutoff, layer) in enumerate(zip(net.cutoffs, net.layers)):
                prior = 0 if index == 0 else net.aperture_width
                weight = layer.weight[:, prior:prior + cutoff * token_dim].float()
                weight = weight.reshape(weight.shape[0], token_dim, cutoff)
                band_input = features[:, :, :cutoff]
                contribution = torch.einsum("adk,rdk->rak", weight, band_input).norm(dim=1)
                drive[:cutoff] += contribution.mean(0)
                gain[:cutoff] += (contribution / band_input.norm(dim=1).clamp_min(1e-12)).mean(0)
            # bands past the first stage's cutoff are read by fewer stages; average per reading
            readings = torch.tensor(
                [sum(1 for cutoff in net.cutoffs if cutoff > band) for band in range(bands)],
                device=features.device, dtype=gain.dtype,
            ).clamp_min(1)
            gain = gain / readings
            for key, values in (("listening", gain), ("listening_drive", drive)):
                total = float(values.sum())
                if total > 0:
                    out[key] = [float(value) / total for value in values.cpu()]
        return out

    def _display_piece(self, token_id: int) -> str:
        """the token as text: word boundaries as spaces, newline bytes as newlines,
        control tokens in brackets"""
        piece = self.tokenizer.diagnostic_piece(token_id)
        if piece == "\\n":
            return "\n"
        if piece.startswith("[") and piece.endswith("]"):
            return piece
        return self.tokenizer.piece(token_id).replace("\u2581", " ")

    def _maximum_radius_error(self) -> float:
        # Measured by the optimizer itself from the norms it gathers for each
        # update, before retraction; a separate scan costs 170 ms on MPS.
        return float(self.engine.learner.last_radius_error)

    @torch.no_grad()
    def evaluate(
        self, corpus_path, max_bytes=0, start_byte=0, carry_state=True,
    ) -> EvaluationMetrics:
        """Held-out metrics on a slice. The traces are state the model has
        accumulated, part of its prior rather than disposable context, so by
        default the slice is read from the current state (`carry_state`);
        `carry_state=False` reads it from zeroed traces, as earlier versions
        did. Either way every state is restored afterwards: evaluating never
        changes the model."""
        saved_trace = self.engine.bank.traces.clone()
        saved_workspace = list(self._workspace_ids)
        totals = dict(nll=0.0, loss_bytes=0.0, loss_tokens=0.0,
                      correct=0.0, top5=0.0, source_bytes=0, tokens=0)
        if not carry_state:
            self.engine.bank.reset()
            self._workspace_ids = []
        try:
            with TokenCorpus(
                corpus_path, self.tokenizer, start_byte=start_byte,
                max_bytes=max_bytes,
            ) as reader:
                while True:
                    block = reader.next_block(max(1, self.config.evidence_quantum))
                    if block is None:
                        break
                    ids, source_bytes, _ = block
                    rows = np.arange(len(ids), dtype=np.int64)
                    history_ids = self._cortical_history(ids, rows)
                    features = self.engine.bank.process_block_select(
                        ids, self.net.input_dictionary, rows=rows
                    )
                    targets = torch.from_numpy(ids).to(self.device)
                    weights = torch.from_numpy(reader.last_loss_weights).to(self.device)
                    languages, _ = self.net.opinions(features, return_logits=False)
                    prediction = self.net.summed_prediction(
                        languages, history_ids
                    )
                    self._advance_workspace(ids)
                    per_token = F.cross_entropy(prediction, targets, reduction="none")
                    weight_sum = float(weights.sum())
                    totals["nll"] += float((per_token * weights).sum())
                    totals["loss_bytes"] += reader.last_loss_bytes
                    totals["loss_tokens"] += weight_sum
                    totals["correct"] += float(((prediction.argmax(1) == targets).float() * weights).sum())
                    top = prediction.topk(min(5, prediction.shape[1]), 1).indices
                    totals["top5"] += float(((top == targets[:, None]).any(1).float() * weights).sum())
                    totals["source_bytes"] += source_bytes
                    totals["tokens"] += len(ids)
        finally:
            self.engine.bank.traces.copy_(saved_trace)
            self._workspace_ids = saved_workspace
        return EvaluationMetrics(
            bpb=totals["nll"] / max(EPS, totals["loss_bytes"] * math.log(2.0)),
            token_loss=totals["nll"] / max(EPS, totals["loss_tokens"]),
            top1=totals["correct"] / max(EPS, totals["loss_tokens"]),
            top5=totals["top5"] / max(EPS, totals["loss_tokens"]),
            source_bytes=int(totals["source_bytes"]), tokens=int(totals["tokens"]),
        )

    def observe_ids(self, ids) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids):
            self.engine.bank._advance_exact(ids)
            self._advance_workspace(ids)
            self.engine.tokens_seen += len(ids)

    def ingest_text(self, text, online=False):
        value = str(text)
        ids = self.tokenizer.encode(value)
        if not len(ids):
            return
        source_bytes = len(value.encode("utf-8"))
        if online:
            self.learn_ids(ids, source_bytes)
        else:
            self.observe_ids(ids)
            self.engine.bytes_seen += source_bytes

    def _observe_control(self, name):
        self.observe_ids([self.tokenizer.control_ids[name]])

    # rows per forward pass while learning a chat batch; a long batch is read
    # in chunks that accumulate into its one update
    CHAT_CHUNK = 512

    def _chat_update_span(self):
        """source tokens one corpus update spans at io2's present density."""
        return self.config.evidence_quantum / self.engine.io2_density()

    def _chat_rate(self):
        """movement for an update closed by chat, scaled by source time.

        In corpus training every update moves source_gain * plasticity and
        io2's stride decides how much source it spans (quantum / density
        tokens).  A chat batch trains every position, so it takes io2's
        density directly: an update spanning n source tokens moves
        n * density / quantum of a corpus update, never more than one.
        """
        engine = self.engine
        fraction = engine.optimizer_source_tokens / self._chat_update_span()
        return self.config.source_gain * self.config.width_rate * engine.io2_ratio * min(1.0, fraction)

    def learn_ids(self, ids, source_bytes, weights=None):
        """chat's learning: training with no decimation and variable batches.

        The span is one batch, trained at every position and closed by one
        update scaled by its source time.  A span longer than a corpus update
        (quantum / io2 density tokens) closes one update per such stretch.
        """
        ids = np.asarray(ids, dtype=np.int64)
        if not len(ids):
            return
        weights = (
            np.ones(len(ids), np.float32) if weights is None
            else np.asarray(weights, np.float32)
        )
        lo_update = 0
        while lo_update < len(ids):
            # io2 keeps observing, so the span is re-read before each update
            remaining = len(ids) - lo_update
            # equal parts, none longer than one corpus update's span
            span = max(1, math.floor(self._chat_update_span()))
            hi_update = lo_update + math.ceil(remaining / math.ceil(remaining / span))
            bounds = list(range(lo_update, hi_update, self.CHAT_CHUNK)) + [hi_update]
            # the closing chunk must carry weight, or its update would not happen
            while len(bounds) > 2 and not weights[bounds[-2]:hi_update].sum() > 0:
                bounds.pop(-2)
            for index, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
                part_bytes = round(source_bytes * (hi - lo) / len(ids))
                self._train_batch(
                    ids[lo:hi], part_bytes, weights[lo:hi], part_bytes,
                    stride=1, update=index == len(bounds) - 2,
                )
            lo_update = hi_update

    def ingest_prompt(self, text, online=False, mark_turn=True):
        if not online:
            if mark_turn:
                self._observe_control("user")
            self.ingest_text(text, online=False)
            self._observe_control("soma")
            return
        # one prompt, one batch: its markers are read but carry no loss, as
        # protocol markers carry none in corpus training
        controls = self.tokenizer.control_ids
        value = str(text)
        body = self.tokenizer.encode(value)
        ids = np.concatenate((
            [controls["user"]] if mark_turn else [], body, [controls["soma"]],
        )).astype(np.int64)
        weights = np.concatenate((
            [0.0] if mark_turn else [], np.ones(len(body)), [0.0],
        )).astype(np.float32)
        self.learn_ids(ids, len(value.encode("utf-8")), weights)

    @torch.no_grad()
    def next_logits(self) -> torch.Tensor:
        features = self.engine.bank.tap(self.net.input_dictionary).unsqueeze(0)
        languages, _ = self.net.opinions(features, return_logits=False)
        prediction = self.net.summed_prediction(
            languages, self._current_cortical_history()
        )
        return prediction[0]

    def _generation_tokens(self, length, temperature, greedy=False):
        if int(length) < 1:
            return
        if temperature <= 0 and not greedy:
            raise ValueError("temperature must be positive")
        forbidden = list(self.tokenizer.generation_forbidden_ids)
        turn_id = self.tokenizer.control_ids["turn"]
        ended = False
        try:
            for _ in range(int(length)):
                logits = self.next_logits()
                logits[forbidden] = float("-inf")
                if greedy:
                    token_id = int(logits.argmax())
                else:
                    probability = F.softmax(logits / float(temperature), dim=0)
                    token_id = int(torch.multinomial(probability, 1))
                self.observe_ids([token_id])
                if token_id == turn_id:
                    ended = True
                    break
                yield token_id
        finally:
            if not ended:
                self.observe_ids([turn_id])

    def generate_ids(self, length=200, temperature=1.0, greedy=False, **_unused):
        return list(self._generation_tokens(length, temperature, greedy))

    def generate_text(self, length=200, temperature=1.0, greedy=False, **kwargs):
        return self.tokenizer.decode(self.generate_ids(length, temperature, greedy, **kwargs))

    def generate(self, length=200, temperature=1.0, **_unused):
        decoder = self.tokenizer.incremental_decoder()
        for token_id in self._generation_tokens(length, temperature):
            yield from decoder.push(token_id)
        yield from decoder.finish()

    def diagnostics(self) -> dict:
        engine = self.engine
        logits = (self._last_logits if self._last_logits is not None
                  else self.next_logits()).cpu()
        probability = F.softmax(logits.float(), dim=0)
        top = probability.topk(min(6, len(probability)))
        return {
            "species": self.species, "parameters": self.params(),
            "input_representation": self.config.input_representation,
            "bytes_seen": self.bytes_seen, "tokens_seen": self.tokens_seen,
            "batch": self.batch_size,
            "controller_bands": int(engine.error_bank.n_bands),
            "traversal_band": float(engine.io2_decimation_band),
            "norm_match": float(engine.learner.last_norm_match),
            "cortex_gain": float(engine.learner.cortex_gain),
            "cortex_rate": float(self.config.cortex_rate),
            "pointer_window": max(0, self._readout_lags() - 1),
            "pointer_share": (
                float(self.net.recent_readout.last_share)
                if self.net.recent_readout is not None else 0.0
            ),
            "workspace_fill": len(self._workspace_ids),
            "maximum_radius_error": self._maximum_radius_error(),
            "finite": all(bool(torch.isfinite(p).all()) for p in self.net.parameters()),
            "top_predictions": [
                {"token": self.tokenizer.diagnostic_piece(int(index)),
                 "probability": float(value)}
                for value, index in zip(top.values, top.indices)
            ],
        }

    def _residual_trace(self):
        if self.engine.fused_error_bank is not None:
            return self.engine.fused_error_bank.traces.detach().cpu()
        return self.engine.error_bank.trace.detach().cpu().T.contiguous()

    def _checkpoint_dict(self, revision_id) -> dict:
        if self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        controller = {name: getattr(self.engine, name) for name in _CONTROLLER_SCALARS}
        controller["residual_spectrum_baseline"] = (
            None if self.engine.residual_spectrum_baseline is None
            else self.engine.residual_spectrum_baseline.clone()
        )
        radii = {
            name: float(self.engine.learner.matrix_radii[id(parameter)])
            for name, parameter in self.net.named_parameters() if parameter.ndim >= 2
        }
        source_cursor = self._source_cursor
        if isinstance(source_cursor, dict) and isinstance(source_cursor.get("reader"), dict):
            source_cursor = dict(source_cursor)
            source_cursor["reader"] = _sign_corpus_state(source_cursor["reader"])
        parent_revision_id = (
            self.revision_id if self._revision_published
            else self.parent_revision_id
        )
        revision_history = list(self.revision_history)
        if self._revision_published:
            previous_record = {
                "checkpoint_id": self.checkpoint_id,
                "revision_id": self.revision_id,
                "parent_revision_id": self.parent_revision_id,
            }
            if not revision_history or revision_history[-1] != previous_record:
                revision_history.append(previous_record)
        revision_history = revision_history[-256:]
        return {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_version": CHECKPOINT_VERSION,
            "runtime_id": RUNTIME_ID, "runtime_version": RUNTIME_VERSION,
            "species": self.species, "config": self.config.public_dict(),
            "description": self.description,
            "checkpoint_id": self.checkpoint_id, "revision_id": revision_id,
            "parent_revision_id": parent_revision_id,
            "root_lineage_id": self.root_lineage_id, "created_at": self.created_at,
            "revision_history": revision_history,
            "forked_from_checkpoint_id": self.forked_from_checkpoint_id,
            "forked_from_revision_id": self.forked_from_revision_id,
            "tokenizer_name": TOKENIZER_NAME,
            "tokenizer_sha256": self.tokenizer.sha256,
            "model": {key: value.detach().cpu() for key, value in self.net.state_dict().items()},
            "pending_gradients": {
                name: parameter.grad.detach().cpu()
                for name, parameter in self.net.named_parameters()
                if parameter.grad is not None
            },
            "matrix_radii": radii,
            "identity_trace": self.engine.bank.traces.clone(),
            "workspace_ids": list(self._workspace_ids),
            "residual_trace": self._residual_trace(),
            "controller": controller,
            "learner": {
                "base_lr": self.engine.learner.base_lr,
                "steps": self.engine.learner.steps,
                "angular_movement": self.engine.learner.angular_movement,
                "norm_match_sum": self.engine.learner.norm_match_sum,
                "last_norm_match": self.engine.learner.last_norm_match,
                "max_norm_match": self.engine.learner.max_norm_match,
            },
            "bytes_seen": self.bytes_seen, "tokens_seen": self.tokens_seen,
            "source_cursor": source_cursor,
            "torch_rng_state": torch.get_rng_state(),
        }

    def save(self, path):
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(f"{path}.lock")
        pointer_path = Path(f"{path}.revision.json")
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            previous = _read_json(pointer_path)
            if previous.get("state") == "pending":
                raise RuntimeError("checkpoint has an unfinished publication")
            same = self._loaded_checkpoint_path == path
            if path.exists() and not same:
                raise FileExistsError("checkpoint exists; load it or save a fork")
            disk_revision = previous.get("revision_id")
            if same and disk_revision and disk_revision != self.revision_id:
                raise RuntimeError("checkpoint changed on disk; save a fork")
            revision = uuid.uuid4().hex
            _write_json_atomic(pointer_path, {
                "version": 1, "state": "pending", "checkpoint_id": self.checkpoint_id,
                "previous_revision_id": self.revision_id, "revision_id": revision,
            })
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                checkpoint = self._checkpoint_dict(revision)
                tensor_bytes = _tensor_bytes(checkpoint)
                required = math.ceil(tensor_bytes * 1.1) + max(DISK_RESERVE_BYTES, tensor_bytes // 2)
                if shutil.disk_usage(path.parent).free < required:
                    raise OSError("not enough free disk space for an atomic checkpoint")
                torch.save(checkpoint, temporary)
                os.replace(temporary, path)
                self.parent_revision_id = checkpoint["parent_revision_id"]
                self.revision_history = checkpoint["revision_history"]
                self.revision_id = revision
                self._revision_published = True
                self._loaded_checkpoint_path = path
                _write_json_atomic(pointer_path, {
                    "version": 1, "state": "complete",
                    "checkpoint_id": self.checkpoint_id, "revision_id": revision,
                })
                self._write_sidecar(path)
            except Exception:
                temporary.unlink(missing_ok=True)
                if previous:
                    _write_json_atomic(pointer_path, previous)
                else:
                    pointer_path.unlink(missing_ok=True)
                raise
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        gc.collect()
        return path

    def _write_sidecar(self, path: Path) -> None:
        existing = _read_json(Path(f"{path}.meta.json"))
        profile = existing.get("profile")
        diagnostics = self.diagnostics()
        sidecar = {
            "sidecar_version": 4, "species": self.species,
            "runtime_id": RUNTIME_ID, "runtime_version": RUNTIME_VERSION,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_id": self.checkpoint_id, "revision_id": self.revision_id,
            "parent_revision_id": self.parent_revision_id,
            "root_lineage_id": self.root_lineage_id,
            "revision_history": self.revision_history,
            "forked_from_checkpoint_id": self.forked_from_checkpoint_id,
            "forked_from_revision_id": self.forked_from_revision_id,
            "description": self.description,
            "created_at": self.created_at, "updated_at": time.time(),
            "parameters": self.params(), "params": self.params(),
            "width": self.config.width, "bands": self.config.bands,
            "input_representation": self.config.input_representation,
            "cortex_width": self.config.cortex_width,
            "cortex_depth": self.config.cortex_depth,
            "workspace_tokens": self.config.workspace_tokens,
            "controller_bands": self.config.controller_bands,
            "source_gain": self.config.source_gain,
            "evidence_quantum": self.config.evidence_quantum,
            "decimation_stride_cap": self.config.decimation_stride_cap,
            "traversal_gain": self.config.traversal_gain,
            "config": self.config.public_dict(),
            "bytes_seen": self.bytes_seen, "tokens_seen": self.tokens_seen,
            "tokenizer": TOKENIZER_NAME, "tokenizer_sha256": self.tokenizer.sha256,
            "lineage": {
                "root": self.root_lineage_id,
                "parent_revision": self.parent_revision_id,
                "forked_from_checkpoint": self.forked_from_checkpoint_id,
                "forked_from_revision": self.forked_from_revision_id,
                "retained_revisions": len(self.revision_history),
            },
            "source_cursor": self._source_cursor,
            "diagnostics": diagnostics,
            "capabilities": [
                "inference.text", "training.continual", "training.resume",
                "learning.online", "state.temporal_trace",
                "composition.cortical", "state.exact_workspace",
                "generation.dream",
            ],
        }
        if isinstance(profile, dict):
            sidecar["profile"] = profile
        _write_json_atomic(Path(f"{path}.meta.json"), sidecar)

    def save_fork(self, path):
        old = (self.checkpoint_id, self.parent_revision_id, self.root_lineage_id,
               self.revision_id, self._loaded_checkpoint_path, self.created_at,
               self._revision_published, self.revision_history,
               self.forked_from_checkpoint_id, self.forked_from_revision_id)
        source_checkpoint_id = self.checkpoint_id
        source_revision_id = self.revision_id
        source_record = {
            "checkpoint_id": source_checkpoint_id,
            "revision_id": source_revision_id,
            "parent_revision_id": self.parent_revision_id,
        }
        history = list(self.revision_history)
        if not history or history[-1] != source_record:
            history.append(source_record)
        self.parent_revision_id = self.revision_id
        self.checkpoint_id = uuid.uuid4().hex
        self.revision_id = uuid.uuid4().hex
        self.revision_history = history[-256:]
        self.forked_from_checkpoint_id = source_checkpoint_id
        self.forked_from_revision_id = source_revision_id
        self._loaded_checkpoint_path = None
        self.created_at = time.time()
        self._revision_published = False
        try:
            return self.save(path)
        except Exception:
            (self.checkpoint_id, self.parent_revision_id, self.root_lineage_id,
             self.revision_id, self._loaded_checkpoint_path, self.created_at,
             self._revision_published, self.revision_history,
             self.forked_from_checkpoint_id, self.forked_from_revision_id) = old
            raise

    @classmethod
    def from_checkpoint(cls, path, device="auto", tokenizer_path=None):
        path = Path(path).expanduser().resolve()
        pointer = _read_json(Path(f"{path}.revision.json"))
        if pointer.get("state") == "pending":
            raise ValueError("checkpoint publication is unfinished")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if checkpoint.get("checkpoint_format") != CHECKPOINT_FORMAT or int(checkpoint.get("checkpoint_version", 0)) != CHECKPOINT_VERSION:
            raise ValueError("unsupported rma checkpoint")
        if pointer.get("state") == "complete" and pointer.get("revision_id") != checkpoint.get("revision_id"):
            raise ValueError("checkpoint revision pointer does not match")
        saved_config = dict(checkpoint["config"])
        # v1.1 and earlier exposed cross-band differences.  Their weights
        # remain valid because the identity trace tensor itself is unchanged.
        saved_config.setdefault("input_representation", "cross_band")
        # saved before width scaling existed: keep the dynamics it learned under
        saved_config.setdefault("width_scaling", 0.0)
        # a body's stride briefly had its own setting; io2 owns it again
        saved_config.pop("body_stride", None)
        if saved_config.get("body"):
            raise ValueError(
                "this model reads through a transformer body, which rma no longer includes "
                "(rma 1.7 and earlier open it)"
            )
        if "evidence_quantum" not in saved_config:
            saved_config["evidence_quantum"] = {
                128: 512, 256: 256, 384: 128, 512: 64,
            }.get(int(saved_config.get("width", 256)), 64)
        model = cls(
            RMAConfig(**saved_config), device=device,
            tokenizer_path=tokenizer_path, description=checkpoint.get("description", ""),
        )
        if model.tokenizer.sha256 != checkpoint.get("tokenizer_sha256"):
            raise ValueError("checkpoint tokenizer differs from runtime tokenizer")
        # A checkpoint whose cortex predates the sentinel pointer is grafted:
        # its weights load as they were and the pointer starts fresh, holding
        # about 0.2% of probability until it learns.  Anything else missing or
        # unexpected is a mismatch.
        missing, unexpected = model.net.load_state_dict(
            checkpoint["model"], strict=False
        )
        grafted = [name for name in missing if name.startswith("recent_readout.")]
        if unexpected or len(grafted) != len(missing):
            raise ValueError(
                "checkpoint does not match this model: "
                f"missing {sorted(set(missing) - set(grafted))}, unexpected {sorted(unexpected)}"
            )
        model.grafted_parameters = grafted
        named = dict(model.net.named_parameters())
        for name, gradient in checkpoint.get("pending_gradients", {}).items():
            if name in named:
                named[name].grad = gradient.to(
                    device=model.device, dtype=named[name].dtype
                )
        for name, radius in checkpoint["matrix_radii"].items():
            model.engine.learner.matrix_radii[id(named[name])] = float(radius)
        model.engine.bank.traces.copy_(checkpoint["identity_trace"])
        workspace = [
            int(value) for value in checkpoint.get("workspace_ids", [])
        ]
        kept = model._workspace_kept()
        model._workspace_ids = workspace[-kept:] if kept else []
        residual = checkpoint["residual_trace"]
        if model.engine.fused_error_bank is not None:
            target = model.engine.fused_error_bank.traces
            source = residual.to(device=model.device, dtype=target.dtype)
            overlap = min(target.shape[1], source.shape[1])
            target.zero_()
            target[:, :overlap].copy_(source[:, :overlap])
        else:
            target = model.engine.error_bank.trace
            source = residual.T.double()
            overlap = min(target.shape[0], source.shape[0])
            target.zero_()
            target[:overlap].copy_(source[:overlap])
        for name, value in checkpoint["controller"].items():
            if name == "residual_spectrum_baseline" and value is not None:
                target_length = model.engine.error_bank.n_bands - 1
                value = value[:target_length].clone()
                if len(value) < target_length:
                    value = F.pad(value, (0, target_length - len(value)))
            setattr(model.engine, name, value)
        if "optimizer_source_tokens" not in checkpoint["controller"]:
            # older checkpoints: estimate the pending update's source span
            model.engine.optimizer_source_tokens = float(
                checkpoint["controller"].get("optimizer_prediction_mass", 0.0)
            ) / model.engine.io2_density()
        learner = checkpoint["learner"]
        for name, value in learner.items():
            setattr(model.engine.learner, name, value)
        model.engine.bytes_seen = int(checkpoint.get("bytes_seen", 0))
        model.engine.tokens_seen = int(checkpoint.get("tokens_seen", 0))
        model.checkpoint_id = checkpoint["checkpoint_id"]
        model.revision_id = checkpoint["revision_id"]
        model.parent_revision_id = checkpoint.get("parent_revision_id")
        model.root_lineage_id = checkpoint.get("root_lineage_id", model.checkpoint_id)
        model.revision_history = list(checkpoint.get("revision_history", []))[-256:]
        model.forked_from_checkpoint_id = checkpoint.get("forked_from_checkpoint_id")
        model.forked_from_revision_id = checkpoint.get("forked_from_revision_id")
        model.created_at = float(checkpoint.get("created_at", time.time()))
        model._source_cursor = checkpoint.get("source_cursor")
        model._loaded_checkpoint_path = path
        model._revision_published = True
        torch.set_rng_state(checkpoint["torch_rng_state"])
        return model

    def load(self, path):
        replacement = self.from_checkpoint(path, device=self.device)
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        return self
