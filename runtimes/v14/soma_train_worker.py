"""isolated training worker for the soma gui.

the gui launches this as a plain child process so torch/mps training state
dies with the process instead of accumulating inside the tkinter app.
"""

import gc
import contextlib
import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path

ORIG_STDOUT = sys.stdout
EVENT_PATH = None


class DiscardOutput:
    """file-like sink for noisy runtime output without retaining it."""

    encoding = "utf-8"

    @staticmethod
    def write(value):
        return len(value)

    @staticmethod
    def flush():
        return None

    @staticmethod
    def isatty():
        return False


DISCARD_OUTPUT = DiscardOutput()
MPS_CACHE_TRIM_INTERVAL_S = 2.0
MPS_CACHE_TRIM_MIN_BYTES = 512 * 1024**2
MPS_CACHE_TRIM_LIVE_RATIO = 0.50
MEMORY_TRIM_FRACTION = 0.75
MEMORY_RECYCLE_FRACTION = 0.75
MEMORY_RECYCLE_GROWTH_BYTES = 3 * 1024**3
MEMORY_RECYCLE_SUSTAINED_CHECKS = 3
MEMORY_STOP_FRACTION = 1.25
MEMORY_EMERGENCY_FRACTION = 1.75
MEMORY_STOP_SUSTAINED_CHECKS = 3


def process_physical_footprint():
    """return the macos task footprint charged by activity monitor."""

    if sys.platform != "darwin":
        return 0
    try:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        task = ctypes.c_uint.in_dll(library, "mach_task_self_").value
        library.task_info.argtypes = [
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        library.task_info.restype = ctypes.c_int
        values = (ctypes.c_uint32 * 100)()
        count = ctypes.c_uint(len(values))
        if library.task_info(task, 22, values, ctypes.byref(count)) != 0:
            return 0
        # task_vm_info.phys_footprint is the uint64 at byte offset 144.
        raw = bytes(values)
        return int.from_bytes(raw[144:152], "little")
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


def system_physical_memory():
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(
            os.sysconf("SC_PAGE_SIZE")
        )
    except (OSError, TypeError, ValueError):
        return 0

def emit(kind, *items):
    line = json.dumps([kind, *items], ensure_ascii=True)
    try:
        print(line, file=ORIG_STDOUT, flush=True)
    except (BrokenPipeError, OSError):
        # The desktop can disappear while this isolated run remains healthy.
        pass
    if kind != "report" and EVENT_PATH is not None:
        try:
            with EVENT_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass


def read_json_dict(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def release_torch_memory(device_type=None):
    # This worker exits immediately after release. The process boundary is the
    # authoritative MPS cleanup; synchronizing a failed device write here can
    # leave teardown blocked indefinitely.
    del device_type
    gc.collect()
    release_cpu_allocator_pages()


def release_cpu_allocator_pages():
    """ask macos malloc zones to return unused pages after large temporaries."""

    if sys.platform != "darwin":
        return 0
    try:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        relief = library.malloc_zone_pressure_relief
        relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        relief.restype = ctypes.c_size_t
        return int(relief(None, 0))
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


def mps_cache_trim_needed(allocated, driver):
    """keep reusable mps scratch bounded relative to the live model."""

    allocated = max(0, int(allocated))
    cached = max(0, int(driver) - allocated)
    allowance = max(
        MPS_CACHE_TRIM_MIN_BYTES,
        int(allocated * MPS_CACHE_TRIM_LIVE_RATIO),
    )
    return cached > allowance


def memory_guard_level(footprint, physical):
    """classify process growth without mistaking unified memory for pressure."""

    footprint = max(0, int(footprint))
    physical = max(0, int(physical))
    if physical <= 0:
        return "normal"
    ratio = footprint / physical
    if ratio >= MEMORY_EMERGENCY_FRACTION:
        return "emergency"
    if ratio >= MEMORY_STOP_FRACTION:
        return "high"
    return "normal"


def memory_recycle_threshold(baseline, physical):
    """bound native backend growth without rejecting a large live model."""

    baseline = max(0, int(baseline))
    physical = max(0, int(physical))
    return max(
        baseline + MEMORY_RECYCLE_GROWTH_BYTES,
        int(physical * MEMORY_RECYCLE_FRACTION),
    )


def recycle_config(config):
    """resume the just-published checkpoint and its verified source cursor."""

    result = dict(config)
    result["checkpoint"] = str(result["save_path"])
    result["fork"] = False
    result["start_byte"] = "0"
    result["start_byte_override"] = False
    return result


def worker_exec_argv(config_path, stop_path):
    script = str(Path(__file__).resolve())
    if getattr(sys, "frozen", False):
        return [
            sys.executable,
            "--run-script",
            script,
            str(config_path),
            str(stop_path),
        ]
    return [sys.executable, script, str(config_path), str(stop_path)]


def supervised_metrics(totals):
    """report only target-bearing language, excluding masked protocol text."""
    loss_sum = float(totals["nll"])
    loss_bytes = float(totals["loss_bytes"])
    loss_tokens = float(totals["loss_tokens"])
    return {
        "nats_per_byte": loss_sum / max(1.0, loss_bytes),
        "token_loss": loss_sum / max(1.0, loss_tokens),
        "loss_bytes": loss_bytes,
    }


class Worker:
    def __init__(self, config_path, stop_path):
        self.config_path = Path(config_path)
        self.stop_path = Path(stop_path)
        self.stop = False
        self.save_on_stop = True
        self.model = None
        self.last_memory_hygiene = 0.0
        self.autosave_interval_s = 0.0
        self.last_save_t = [time.time()]
        self.dream_every_batches = 0
        self.dream_length = "128"
        self.dream_temperature = 0.8
        self.start_byte = 0
        self.resume_state = None
        self.runtime = None
        self.run_id = ""
        self.manifest_path = None
        self.manifest_state = {}
        self.last_manifest_t = 0.0
        self.metrics_path = None
        self.last_metrics_t = 0.0
        self.stopped_without_save = False
        self.final_save_failed = False
        self.memory_guard_tripped = False
        self.memory_high_samples = 0
        self.memory_baseline = 0
        self.memory_recycle_samples = 0
        self.recycle_requested = False
        self.recycle_checkpoint_published = False
        self.validation_path = ""
        self.validation_interval_bytes = 0
        self.validation_max_bytes = 0
        self.next_validation_bytes = 0

    def update_manifest(self, force=False, **changes):
        if self.manifest_path is None:
            return
        now = time.time()
        if not force and now - self.last_manifest_t < 2.0:
            return
        self.manifest_state.update(changes)
        self.manifest_state["heartbeat_at"] = now
        try:
            write_json_atomic(self.manifest_path, self.manifest_state)
            self.last_manifest_t = now
        except OSError:
            pass

    def record_metrics(self, report, now):
        if self.metrics_path is None or now - self.last_metrics_t < 60.0:
            return
        keys = (
            "absolute_pos", "absolute_total", "bpb", "recent_loss",
            "token_loss", "metric_mode", "prediction_coverage", "acc",
            "top5_acc", "bps", "life_bytes", "life_bps", "life_seconds",
            "epoch", "epochs", "lr", "io2", "stride", "byte_stride",
            "update_duty", "pulse_frequency", "update_cost_fraction",
            "optimizer_updates", "optimizer_fill", "radial_pressure",
            "row_spread", "error_energy",
            "training_calls", "source_tokens_per_call",
            "predictions_per_call",
            "mps_allocated_gb", "mps_driver_gb", "mps_cache_gb",
            "process_footprint_gb", "system_memory_gb",
        )
        sample = {
            "kind": "training",
            "timestamp": now,
            "run_id": self.run_id,
            **{key: report.get(key) for key in keys},
        }
        try:
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
            self.last_metrics_t = now
        except OSError as error:
            emit("warning", f"metrics history write failed: {error}")

    def memory_snapshot(self):
        """report live unified-memory ownership without synchronizing mps."""

        result = {
            "mps_allocated_gb": 0.0,
            "mps_driver_gb": 0.0,
            "mps_cache_gb": 0.0,
            "process_footprint_gb": 0.0,
            "system_memory_gb": 0.0,
        }
        scale = float(1024 ** 3)
        result["process_footprint_gb"] = (
            process_physical_footprint() / scale
        )
        result["system_memory_gb"] = system_physical_memory() / scale
        torch = getattr(self.runtime, "torch", None)
        model = self.model
        device = getattr(getattr(model, "device", None), "type", None)
        if torch is None or device != "mps" or not hasattr(torch, "mps"):
            return result
        try:
            allocated = float(torch.mps.current_allocated_memory())
            driver = float(torch.mps.driver_allocated_memory())
        except Exception:
            return result
        result["mps_allocated_gb"] = allocated / scale
        result["mps_driver_gb"] = driver / scale
        result["mps_cache_gb"] = max(0.0, driver - allocated) / scale
        return result

    def run_validation(self, model, reason):
        if not self.validation_path:
            return
        started = time.time()
        emit("validation_start", {
            "reason": reason,
            "path": self.validation_path,
            "bytes_seen": int(model.bytes_seen),
        })
        self.update_manifest(
            force=True,
            status="validating",
            validation_path=self.validation_path,
        )
        try:
            metrics = self.quiet(
                model.evaluate,
                self.validation_path,
                max_bytes=self.validation_max_bytes,
            )
            completed = time.time()
            payload = {
                "kind": "validation",
                "timestamp": completed,
                "run_id": self.run_id,
                "reason": reason,
                "bytes_seen": int(model.bytes_seen),
                "bpb": float(metrics.bpb),
                "top1": float(metrics.top1),
                "tokens": int(metrics.tokens),
                "source_bytes": float(metrics.source_bytes),
                "seconds": float(completed - started),
                "dense": True,
                "estimated_byte_denominator": True,
            }
            if self.metrics_path is not None:
                with self.metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(payload, ensure_ascii=True) + "\n")
            emit("validation", payload)
            self.update_manifest(
                force=True,
                status="running",
                last_validation=payload,
                last_validation_at=completed,
                validation_path=self.validation_path,
            )
        except Exception as error:
            message = f"dense validation failed: {error}"
            emit("warning", message)
            self.update_manifest(
                force=True,
                status="running",
                last_warning=message,
                last_warning_at=time.time(),
                validation_path=self.validation_path,
            )

    def read_stop(self):
        if not self.stop_path.exists():
            return False
        first_request = not self.stop
        self.stop = True
        try:
            data = json.loads(self.stop_path.read_text(encoding="utf-8"))
            self.save_on_stop = bool(data.get("save", True))
        except Exception:
            self.save_on_stop = True
        if first_request:
            self.update_manifest(force=True, status="stopping")
        return True

    def memory_hygiene(self, model=None, force=False):
        """bound abandoned mps workspaces outside the training hot path."""

        now = time.time()
        if (
            not force
            and now - self.last_memory_hygiene < MPS_CACHE_TRIM_INTERVAL_S
        ):
            return False
        self.last_memory_hygiene = now
        if force:
            gc.collect()

        torch = getattr(self.runtime, "torch", None)
        device = getattr(getattr(model, "device", None), "type", None)
        if torch is None or device != "mps" or not hasattr(torch, "mps"):
            return False
        try:
            allocated = torch.mps.current_allocated_memory()
            driver = torch.mps.driver_allocated_memory()
            footprint = process_physical_footprint()
            physical = system_physical_memory()
            pressure = (
                physical > 0
                and footprint >= physical * MEMORY_TRIM_FRACTION
            )
            should_trim = (
                force
                or pressure
                or mps_cache_trim_needed(allocated, driver)
            )
            if should_trim:
                torch.mps.empty_cache()
                if pressure or force:
                    gc.collect()
                    release_cpu_allocator_pages()
                # Safety is based on the irreducible working set, not the
                # reclaimable cache that caused this cleanup.
                allocated = torch.mps.current_allocated_memory()
                driver = torch.mps.driver_allocated_memory()
                footprint = process_physical_footprint()
            level = memory_guard_level(footprint, physical)
            if not force:
                if self.memory_baseline <= 0 and footprint > 0:
                    self.memory_baseline = footprint
                recycle_at = memory_recycle_threshold(
                    self.memory_baseline, physical
                )
                if footprint >= recycle_at:
                    self.memory_recycle_samples += 1
                else:
                    self.memory_recycle_samples = 0
                if level == "high":
                    self.memory_high_samples += 1
                elif level == "normal":
                    self.memory_high_samples = 0
                if (
                    self.memory_recycle_samples
                    >= MEMORY_RECYCLE_SUSTAINED_CHECKS
                    and not self.recycle_requested
                    and not self.stop_path.exists()
                ):
                    self.recycle_requested = True
                    self.stop = True
                    self.save_on_stop = True
                    emit(
                        "info",
                        "refreshing the training engine to release native "
                        "memory; learning will resume automatically",
                    )
            should_stop = (
                level == "emergency"
                or self.memory_high_samples >= MEMORY_STOP_SUSTAINED_CHECKS
            )
            if not force and should_stop and not self.memory_guard_tripped:
                self.memory_guard_tripped = True
                self.stop = True
                self.save_on_stop = True
                footprint_gb = footprint / float(1024**3)
                emit(
                    "warning",
                    "memory safety limit reached at "
                    f"{footprint_gb:.1f}gb; saving and stopping training",
                )
            return bool(should_trim)
        except Exception:
            return False

    def quiet(self, fn, *args, **kwargs):
        # model.train can run for days. StringIO retained every progress line
        # for the lifetime of the run and therefore grew without a bound.
        with contextlib.redirect_stdout(DISCARD_OUTPUT):
            return fn(*args, **kwargs)

    def release_model(self):
        model = self.model
        device = getattr(getattr(model, "device", None), "type", None)
        self.model = None
        if model is not None:
            del model
        release_torch_memory(device)

    def run(self):
        global EVENT_PATH
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.run_id = str(config.get("_run_id") or "")
        manifest_value = config.get("_run_manifest_path")
        event_value = config.get("_run_event_path")
        metrics_value = config.get("_run_metrics_path")
        self.manifest_path = Path(manifest_value) if manifest_value else None
        EVENT_PATH = Path(event_value) if event_value else None
        self.metrics_path = Path(metrics_value) if metrics_value else None
        self.manifest_state = (
            read_json_dict(self.manifest_path)
            if self.manifest_path is not None else {}
        )
        self.update_manifest(
            force=True,
            status="running",
            pid=os.getpid(),
            run_id=self.run_id,
        )
        try:
            import soma_v14 as soma_runtime
        except Exception as e:
            emit("error", f"import failed: {e}")
            self.update_manifest(
                force=True, status="failed", ended_at=time.time(),
                error=f"import failed: {e}")
            emit("done")
            return
        self.runtime = soma_runtime

        failure = None
        try:
            emit("start")
            ckpt = config.get("checkpoint")
            lr_val, lr_auto, lr_base = soma_runtime._parse_auto_or_float(
                config.get("lr", "0.0003"), 0.0003)
            try:
                grad_clip = float(config.get("grad_clip", "1.0"))
            except (TypeError, ValueError):
                grad_clip = 1.0
            try:
                update_interval = max(
                    1, int(config.get("update_interval", 1))
                )
            except (TypeError, ValueError):
                update_interval = 1
            if update_interval != 1:
                raise ValueError(
                    "update interval is fixed at one; io2 controls update frequency"
                )
            try:
                decimation_gain = max(
                    0.0, float(config.get("decimation", 1.0))
                )
            except (TypeError, ValueError):
                decimation_gain = 1.0
            lattice_rate = None
            if config.get("lattice_rate") not in (None, ""):
                try:
                    lattice_rate = min(
                        1.0, max(0.0, float(config["lattice_rate"]))
                    )
                except (TypeError, ValueError):
                    lattice_rate = None

            if ckpt and Path(ckpt).exists():
                model = soma_runtime.SOMA.from_checkpoint(
                    ckpt, device="auto")
                model.description = str(
                    config.get("description", model.description))
                model.lr = lr_val
                model.lr_base = lr_base
                model.lr_auto = lr_auto
                model.grad_clip = grad_clip
                model.trainer.grad_clip = grad_clip
                model.trainer.update_interval = update_interval
                model.trainer.decimation_gain = decimation_gain
                if lattice_rate is not None:
                    model.trainer.lattice_rate = lattice_rate
                lattice_rate = float(model.trainer.lattice_rate)
                model._pulse_duty = (
                    model.trainer.updates / max(1, model.trainer.batches)
                )
                model.batch_size = int(config["batch"])
                model._set_optimizer_lrs()
                if bool(config.get("fork", False)):
                    emit("saving", str(config["save_path"]))
                    self.update_manifest(force=True, status="saving")
                    fork_path = model.save_fork(config["save_path"])
                    emit(
                        "info",
                        f"forked {Path(ckpt).name} as {Path(fork_path).name}",
                    )
                    emit("checkpoint_saved", str(fork_path))
                else:
                    emit("info", f"resumed {Path(ckpt).name}")
            else:
                if lattice_rate is None:
                    lattice_rate = 0.002
                model = soma_runtime.SOMA(
                    n_bands=int(config["bands"]),
                    capacity=str(config.get("capacity", "medium")),
                    base=float(config["base"]),
                    lr=lr_val,
                    lr_base=lr_base,
                    lr_auto=lr_auto,
                    grad_clip=grad_clip,
                    batch_size=int(config["batch"]),
                    description=str(config.get("description", "")),
                    update_interval=update_interval,
                    decimation_gain=decimation_gain,
                    lattice_rate=lattice_rate,
                    device="auto",
                )
                emit("info", "fresh model")
                emit("saving", str(config["save_path"]))
                self.update_manifest(force=True, status="saving")
                initial_path = self.quiet(model.save, config["save_path"])
                emit("checkpoint_saved", str(initial_path))

            if (
                ckpt
                and not bool(config.get("loop", False))
                and not config.get("cycle_files")
                and not bool(config.get("start_byte_override", False))
            ):
                self.resume_state = model.resume_state_for(config["corpus"])
                if self.resume_state is not None:
                    position = int(self.resume_state.get("start_byte", 0)) + int(
                        self.resume_state.get("bytes_consumed", 0)
                    )
                    config["start_byte"] = position
                    emit("info", f"verified checkpoint cursor at byte {position:,}")

            self.model = model
            n_params = self.param_count(model)
            hidden_str = (
                f"symbol={model.config.symbol_dim:,}"
                f" · scale={model.config.scale_rank:,}"
                f" · model={model.config.model_dim:,}"
                f" · depth={model.config.trunk_depth}")

            corpus_name = config.get("corpus_label") or Path(config["corpus"]).name
            summary = {
                "corpus": corpus_name,
                "bands": model.n_bands,
                "capacity": getattr(model, "capacity", "custom"),
                "base": model.base,
                "hidden": hidden_str,
                "params": n_params,
                "active_params": int(getattr(
                    model, "active_params", model.params)()),
                "device": str(model.device),
                "dtype": "float32 · trace64",
                "batch": model.batch_size,
                "lr": getattr(model, "lr", 0.002),
                "lr_base": getattr(model, "lr_base", 0.002),
                "lr_auto": bool(lr_auto),
                "grad_clip": getattr(model, "grad_clip", 1.0),
                "description": getattr(model, "description", ""),
                "plasticity_mode": "io2",
                "update_interval": int(update_interval),
                "decimation_gain": float(decimation_gain),
                "lattice_rate": float(getattr(
                    model.trainer, "lattice_rate", lattice_rate or 0.0)),
                "bytes_seen": model.bytes_seen,
                "tokens_seen": getattr(model, "tokens_seen", 0),
                "tokenizer": getattr(
                    getattr(model, "tokenizer", None), "vocab_size", 0),
                "save_path": Path(config["save_path"]).name,
                "save_path_full": str(Path(config["save_path"])),
                "loop": bool(config.get("loop", False)),
                "run_id": self.run_id,
            }
            emit("summary", summary)
            try:
                last_save_at = Path(config["save_path"]).stat().st_mtime
            except OSError:
                last_save_at = None
            self.update_manifest(
                force=True,
                status="running",
                summary=summary,
                checkpoint_path=str(config["save_path"]),
                last_save_at=last_save_at,
            )

            autosave_min = float(config.get("autosave_min", 0.5))
            self.autosave_interval_s = max(0.0, autosave_min * 60.0)
            self.last_save_t = [time.time()]
            self.dream_every_batches = int(config.get("dream_every_batches", 0))
            self.dream_length = str(config.get("dream_length", "128"))
            self.dream_temperature = float(config.get("dream_temperature", 0.8))
            self.validation_path = str(config.get("validation_path") or "")
            if self.validation_path:
                self.validation_interval_bytes = max(
                    1,
                    int(float(config.get("validation_every_mb", 100))
                        * 1_000_000),
                )
                self.validation_max_bytes = max(
                    1,
                    int(float(config.get("validation_max_mb", 0.25))
                        * 1_000_000),
                )
                self.next_validation_bytes = (
                    int(model.bytes_seen) + self.validation_interval_bytes
                )
                self.update_manifest(
                    force=True,
                    validation_path=self.validation_path,
                    validation_interval_bytes=self.validation_interval_bytes,
                    validation_max_bytes=self.validation_max_bytes,
                )
            self.install_callbacks(model, config)
            if self.validation_path:
                self.run_validation(model, "baseline")

            if bool(config.get("loop", False)):
                self.run_loop(
                    model,
                    config["corpus"],
                    config["save_path"],
                    int(config.get("head_bytes", 50_000_000)),
                    float(config.get("cycle_pause", 60.0)),
                )
            elif config.get("cycle_files"):
                self.run_cycle(
                    model,
                    config["cycle_files"],
                    int(config.get("cycle_count", 1)),
                    config["save_path"],
                )
            else:
                self.run_single(
                    model,
                    config["corpus"],
                    config["save_path"],
                    int(config.get("start_byte", 0)),
                    int(config.get("epochs", 1)),
                )
        except Exception as e:
            failure = f"{type(e).__name__}: {e}"
            emit("error", f"{type(e).__name__}: {e}")
        finally:
            self.release_model()
            restart = (
                self.recycle_requested
                and not failure
                and not self.final_save_failed
                and not self.stopped_without_save
                and not self.stop_path.exists()
            )
            if restart:
                config = recycle_config(config)
                write_json_atomic(self.config_path, config)
                recycle_count = int(
                    self.manifest_state.get("worker_recycles", 0)
                ) + 1
                self.update_manifest(
                    force=True,
                    status="restarting",
                    worker_recycles=recycle_count,
                    checkpoint_path=str(config["save_path"]),
                    error=None,
                )
                emit("worker_recycle", {
                    "count": recycle_count,
                    "checkpoint": str(config["save_path"]),
                })
                argv = worker_exec_argv(self.config_path, self.stop_path)
                os.execv(argv[0], argv)
            else:
                self.update_manifest(
                    force=True,
                    status=(
                        "failed"
                        if failure or self.final_save_failed
                        else (
                            "stopped_without_save"
                            if self.stopped_without_save else "completed"
                        )
                    ),
                    ended_at=time.time(),
                    error=(failure or (
                        "final checkpoint save failed"
                        if self.final_save_failed else None
                    )),
                )
                emit("done")
                try:
                    self.stop_path.unlink()
                except OSError:
                    pass

    def param_count(self, model):
        if hasattr(model, "net") and hasattr(model, "params"):
            return model.params()
        if hasattr(model, "W_list"):
            n_params = sum(W.numel() for W in model.W_list)
            n_params += sum(U.numel() for U in getattr(model, "U_list", []))
            n_params += sum(G.numel() for G in getattr(model, "G_list", []))
            n_params += sum(gb.numel() for gb in getattr(model, "gb_list", []))
            if model.Wd is not None:
                n_params += model.Wd.numel()
            return n_params
        if model.hidden_dim > 0:
            n_params = model.U.numel() + model.W.numel()
            if model.Wd is not None:
                n_params += model.Wd.numel()
            return n_params
        return model.W.numel()

    def install_callbacks(self, model, config):
        ln2 = math.log(2)
        session_t0 = time.time()
        session_bytes_start = model.bytes_seen
        session_updates_start = int(getattr(model.trainer, "updates", 0))
        session_batches_start = int(getattr(model.trainer, "batches", 0))
        session_tokens_start = int(getattr(
            model.trainer, "training_tokens_seen", 0
        ))
        session_predictions_start = int(getattr(
            model.trainer, "predictions", 0
        ))
        last_emit_t = [0.0]
        report_prev = [0.0, 0.0]
        recent_loss_ema = [None]
        min_emit_s = 0.5
        save_path = config["save_path"]
        self.start_byte = int(config.get("start_byte", 0))

        def patched_report(epoch, epochs, *report):
            token_runtime = len(report) == 3
            if token_runtime:
                reader, totals, t0 = report
                pos = int(reader.bytes_consumed)
                total = int(reader.total_bytes)
                loss_sum = float(totals["nll"])
                metrics = supervised_metrics(totals)
                loss_byte_count = metrics["loss_bytes"]
                correct = int(totals["correct"])
                top5 = int(totals.get("top5", 0))
                samples = int(totals["samples"])
                cum_avg = metrics["nats_per_byte"]
                progress_count = loss_byte_count
                token_loss = metrics["token_loss"]
                top5_acc = 100 * top5 / max(1, samples)
            else:
                pos, total, loss_sum, correct, samples, t0 = report
                cum_avg = loss_sum / max(1, samples)
                progress_count = samples
                token_loss = None
                top5_acc = 0.0
            should_stop = self.stop or self.read_stop()
            now = time.time()
            if should_stop or now - last_emit_t[0] >= min_emit_s:
                model._update_diagnostics()
                self.memory_hygiene(model)
                should_stop = should_stop or self.stop
                if progress_count < report_prev[1]:
                    report_prev[:] = [0.0, 0]
                delta_samples = progress_count - report_prev[1]
                delta_loss = loss_sum - report_prev[0]
                recent_loss = (
                    delta_loss / delta_samples
                    if delta_samples > 0 else cum_avg)
                if recent_loss_ema[0] is None:
                    recent_loss_ema[0] = recent_loss
                else:
                    recent_loss_ema[0] += 0.25 * (
                        recent_loss - recent_loss_ema[0])
                report_prev[:] = [loss_sum, progress_count]
                life_dt = max(1e-9, now - session_t0)
                life_bytes = model.bytes_seen - session_bytes_start
                if token_runtime:
                    absolute_pos = int(reader.position)
                    absolute_total = int(reader.limit)
                else:
                    absolute_pos = int(
                        (self.start_byte if epoch == 0 else 0) + pos)
                    absolute_total = int(
                        (self.start_byte if epoch == 0 else 0) + total)
                report_payload = {
                    "pos": int(pos),
                    "total": int(total),
                    "absolute_pos": absolute_pos,
                    "absolute_total": absolute_total,
                    "loss": float(cum_avg),
                    "token_loss": (
                        float(token_loss) if token_loss is not None else None),
                    "accuracy_unit": "token" if token_runtime else "byte",
                    "recent_loss": float(recent_loss_ema[0]),
                    "bpb": float(cum_avg / ln2),
                    "metric_mode": (
                        "dense"
                        if int(getattr(model, "_stride", 1)) == 1
                        else "sampled"
                    ),
                    "prediction_coverage": float(
                        totals.get("selected_prediction_mass", samples)
                        / max(1.0, float(totals.get("loss_tokens", samples)))
                    ) if token_runtime else 1.0,
                    "acc": float(100 * correct / samples if samples > 0 else 0),
                    "top5_acc": float(top5_acc),
                    "bps": float(
                        (pos - getattr(model, "_epoch_bytes_consumed_start", 0))
                        / max(1e-9, now - t0)
                    ),
                    "life_bytes": int(life_bytes),
                    "life_bps": float(life_bytes / life_dt),
                    "life_seconds": float(life_dt),
                    "epoch": int(epoch + 1),
                    "epochs": int(epochs),
                    "lr": float(getattr(model, "lr", 0.002)),
                    "grad_clip": float(getattr(model, "grad_clip", 1.0)),
                    "bytes_per_token": float(getattr(
                        model, "_bytes_per_token", 1.0)),
                    "pulse_duty": float(getattr(
                        model, "_pulse_duty", 1.0)),
                    "pulse_frequency": float(getattr(
                        model, "_pulse_frequency", 1.0)),
                    "update_cost_fraction": float(getattr(
                        model, "_update_cost", 0.0)),
                    "update_duty": float(
                        (int(getattr(model.trainer, "updates", 0))
                         - session_updates_start)
                        / max(
                            1,
                            int(getattr(model.trainer, "batches", 0))
                            - session_batches_start,
                        )
                    ),
                    "training_calls": int(
                        int(getattr(model.trainer, "batches", 0))
                        - session_batches_start
                    ),
                    "source_tokens_per_call": float(
                        (
                            int(getattr(
                                model.trainer, "training_tokens_seen", 0
                            ))
                            - session_tokens_start
                        )
                        / max(
                            1,
                            int(getattr(model.trainer, "batches", 0))
                            - session_batches_start,
                        )
                    ),
                    "predictions_per_call": float(
                        (
                            int(getattr(model.trainer, "predictions", 0))
                            - session_predictions_start
                        )
                        / max(
                            1,
                            int(getattr(model.trainer, "batches", 0))
                            - session_batches_start,
                        )
                    ),
                    "optimizer_updates": int(getattr(
                        model, "_optimizer_updates", 0)),
                    "optimizer_quantum": float(getattr(
                        model.trainer, "optimizer_quantum", 0.0)),
                    "optimizer_fill": float(
                        getattr(
                            model.trainer,
                            "accumulated_prediction_mass",
                            0.0,
                        )
                        / max(
                            1.0,
                            float(getattr(
                                model.trainer,
                                "optimizer_quantum",
                                1.0,
                            )),
                        )
                    ),
                    "radial_pressure": float(getattr(
                        model, "_radial_pressure", 0.0)),
                    "row_spread": float(getattr(
                        model, "_row_spread", 1.0)),
                    "io2": float(getattr(
                        model, "_io2_plasticity", 1.0)),
                    "stride": int(getattr(model, "_stride", 1)),
                    "byte_stride": float(getattr(
                        model, "_byte_stride", 1.0)),
                    "dream_input_tokens": int(getattr(
                        model, "_dream_input_tokens", 0)),
                    "dream_output_tokens": int(getattr(
                        model, "_dream_output_tokens", 0)),
                    "dream_rehearsal_ratio": float(getattr(
                        model, "_dream_rehearsal_ratio", 0.0)),
                    "input_spectrum": getattr(
                        model, "_input_spectral_drive").detach(
                        ).cpu().tolist(),
                    "error_spectrum": getattr(
                        model, "_spectral_drive").detach(
                        ).cpu().tolist(),
                    "weight_spectrum": getattr(
                        model, "_weight_spectral_drive",
                        getattr(model, "_input_spectral_drive")).detach(
                        ).cpu().tolist(),
                    "active_memory": list(getattr(
                        model, "_active_memory", [])),
                    "top_predictions": list(getattr(
                        model, "_top_predictions", [])),
                    "target_band": float(getattr(
                        model, "_spectral_target_band", 0.0)),
                    "input_coherence": float(getattr(
                        model, "_input_coherence", 0.0)),
                    "output_coherence": float(getattr(
                        model, "_output_coherence", 0.0)),
                    "error_energy": float(getattr(
                        model, "_error_energy", 0.0)),
                    "lattice_drift": (
                        model.trainer.lattice_drift()
                        if hasattr(model.trainer, "lattice_drift")
                        else {"max_abs": 0.0, "mean_abs": 0.0}
                    ),
                    "lattice_coordinates": (
                        model.trainer.lattice.coordinates.tolist()
                        if hasattr(model.trainer, "lattice") else []
                    ),
                }
                report_payload.update(self.memory_snapshot())
                emit("report", report_payload)
                self.record_metrics(report_payload, now)
                self.update_manifest(
                    status="running",
                    last_report=report_payload,
                    last_report_at=now,
                )
                if (
                    not should_stop
                    and self.validation_path
                    and int(model.bytes_seen) >= self.next_validation_bytes
                ):
                    while int(model.bytes_seen) >= self.next_validation_bytes:
                        self.next_validation_bytes += (
                            self.validation_interval_bytes)
                    self.run_validation(model, "interval")
                last_emit_t[0] = now
            if should_stop:
                raise KeyboardInterrupt("gui requested stop")

            if (self.autosave_interval_s > 0
                    and now - self.last_save_t[0] >= self.autosave_interval_s):
                try:
                    emit("saving", str(save_path))
                    self.update_manifest(force=True, status="saving")
                    self.quiet(model.save, save_path)
                    self.memory_hygiene(model, force=True)
                    emit("autosaved", save_path)
                    self.update_manifest(
                        force=True,
                        status="running",
                        last_save_at=time.time(),
                        checkpoint_path=str(save_path),
                    )
                except Exception as e:
                    emit("warning", f"autosave failed; retrying: {e}")
                    self.update_manifest(
                        force=True,
                        last_warning=f"autosave failed; retrying: {e}",
                        last_warning_at=time.time(),
                    )
                    # A failed safety save retries within one minute.
                    self.last_save_t[0] = (
                        now - max(0.0, self.autosave_interval_s - 60.0)
                    )
                else:
                    self.last_save_t[0] = now
                    # Treat every successful autosave as a clean native-memory
                    # boundary. MPS can retain transient command buffers and
                    # workspaces even after empty_cache(); replacing the worker
                    # process is the only authoritative release on macOS.
                    self.recycle_checkpoint_published = True
                    self.recycle_requested = True
                    self.stop = True
                    raise KeyboardInterrupt(
                        "autosave training-engine refresh"
                    )

        model._report = patched_report
        model._dream_callback = lambda text, update, seen: emit("dream", {
            "text": text,
            "update": update,
            "bytes_seen": seen,
            "input_tokens": int(getattr(model, "_dream_input_tokens", 0)),
            "output_tokens": int(getattr(model, "_dream_output_tokens", 0)),
        })

    def run_single(self, model, corpus_path, save_path, start_byte=0,
                   epochs=1):
        try:
            self.quiet(model.train,
                corpus_path,
                epochs=max(1, int(epochs)),
                save_every=0,
                save_path="",
                start_byte=start_byte,
                report_every=1,
                dream_every_batches=self.dream_every_batches,
                dream_length=self.dream_length,
                dream_temperature=self.dream_temperature,
                dream_callback=model._dream_callback,
                resume_state=self.resume_state,
            )
        except KeyboardInterrupt:
            self.read_stop()
        self.final_save(model, save_path)

    def run_cycle(self, model, corpus_paths, cycle_count, save_path):
        files = [str(p) for p in corpus_paths]
        for cycle_idx in range(max(1, int(cycle_count))):
            if self.read_stop():
                break
            emit("info", f"cycle {cycle_idx + 1}/{cycle_count} · {len(files)} files")
            for idx, corpus_path in enumerate(files, 1):
                if self.read_stop():
                    break
                emit("info", f"file {idx}/{len(files)} · {Path(corpus_path).name}")
                try:
                    self.quiet(model.train,
                        corpus_path,
                        epochs=1,
                        save_every=0,
                        save_path="",
                        start_byte=0,
                        report_every=1,
                        dream_every_batches=self.dream_every_batches,
                        dream_length=self.dream_length,
                        dream_temperature=self.dream_temperature,
                        dream_callback=model._dream_callback,
                    )
                except KeyboardInterrupt:
                    self.read_stop()
                    break
                except Exception as e:
                    emit("warning", f"cycle file failed: {Path(corpus_path).name}: {e}")
        self.final_save(model, save_path)

    def run_loop(self, model, corpus_path, save_path, head_bytes, cycle_pause):
        tmp_head = Path(save_path).parent / f".{Path(save_path).stem}_head.tmp"

        def snapshot_head():
            corpus = Path(corpus_path)
            if not corpus.exists():
                return None
            n = min(head_bytes, corpus.stat().st_size)
            if n < 1_000_000:
                return None
            written = 0
            with open(corpus, "rb") as src, open(tmp_head, "wb") as dst:
                rem = n
                while rem > 0:
                    chunk = src.read(min(1024 * 1024, rem))
                    if not chunk:
                        break
                    dst.write(chunk)
                    rem -= len(chunk)
                    written += len(chunk)
            return written

        cycle = 0
        while not self.read_stop():
            cycle += 1
            n_written = snapshot_head()
            if n_written is None:
                emit("info", "corpus too small or missing — waiting")
                self.sleep_responsive(cycle_pause)
                continue
            emit("info", f"cycle {cycle} · head={n_written // 1_000_000}M")
            try:
                self.quiet(model.train,
                    str(tmp_head),
                    epochs=1,
                    save_every=0,
                    save_path="",
                    start_byte=0,
                    report_every=1,
                    dream_every_batches=self.dream_every_batches,
                    dream_length=self.dream_length,
                    dream_temperature=self.dream_temperature,
                    dream_callback=model._dream_callback,
                )
            except KeyboardInterrupt:
                self.read_stop()
                break
            except Exception as e:
                emit("warning", f"cycle error: {e}")
                self.sleep_responsive(cycle_pause)
                continue
            finally:
                try:
                    tmp_head.unlink()
                except FileNotFoundError:
                    pass

            if (self.autosave_interval_s > 0
                    and time.time() - self.last_save_t[0] >= self.autosave_interval_s):
                try:
                    emit("saving", str(save_path))
                    self.update_manifest(force=True, status="saving")
                    self.quiet(model.save, save_path)
                    self.memory_hygiene(model, force=True)
                    self.last_save_t[0] = time.time()
                    emit("checkpoint_saved", save_path)
                    self.update_manifest(
                        force=True,
                        status="running",
                        last_save_at=self.last_save_t[0],
                        checkpoint_path=str(save_path),
                    )
                except Exception as e:
                    emit("warning", f"periodic save failed: {e}")
                    self.update_manifest(
                        force=True,
                        last_warning=f"periodic save failed: {e}",
                        last_warning_at=time.time(),
                    )
            if not self.read_stop():
                emit("info", f"pausing {int(cycle_pause)}s before next cycle")
                self.sleep_responsive(cycle_pause)
        self.final_save(model, save_path)

    def sleep_responsive(self, seconds):
        slept = 0.0
        while slept < seconds and not self.read_stop():
            time.sleep(min(1.0, seconds - slept))
            slept += 1.0
            self.update_manifest(status="running")

    def final_save(self, model, save_path):
        self.read_stop()
        if not self.save_on_stop:
            self.stopped_without_save = True
            emit("info", "stopped without saving")
            return
        if (
            self.recycle_requested
            and self.recycle_checkpoint_published
            and not self.stop_path.exists()
        ):
            # The autosave immediately preceding an automatic recycle already
            # contains the verified stream cursor. No learning occurred after
            # it, so avoid a second multi-gigabyte checkpoint write.
            return
        try:
            emit("saving", str(save_path))
            self.update_manifest(force=True, status="saving")
            self.quiet(model.save, save_path)
            self.memory_hygiene(model, force=True)
            emit("checkpoint_saved", save_path)
            self.update_manifest(
                force=True,
                last_save_at=time.time(),
                checkpoint_path=str(save_path),
            )
        except Exception as e:
            self.final_save_failed = True
            emit("error", f"save failed: {e}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: soma_train_worker.py config.json stop.json")
    Worker(sys.argv[1], sys.argv[2]).run()


if __name__ == "__main__":
    main()
