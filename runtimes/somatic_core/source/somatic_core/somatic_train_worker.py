#!/usr/bin/env python3
"""isolated training worker for somatic_core, for the soma harness.

Infrastructure (event journal, manifest, report throttling, autosave, dreams, validation, source cycles, stream
mode, stop with or without saving) is gm-v1's worker (soma_rma_v1's); only the model import differs."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataclasses

from somatic_core.runtime import IDENTITY, SOMA, CoreConfig


def emit(kind, payload=None):
    event = [kind] if payload is None else [kind, payload]
    print(json.dumps(event, ensure_ascii=False), flush=True)


def write_json_atomic(path: Path, value: dict):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


# Continual learning runs indefinitely, so nothing it writes may grow without
# bound.  A thinned journal keeps every other line once it passes its cap: it
# always spans the whole run, at a resolution that halves as the run grows.
METRICS_JOURNAL_BYTES = 8_000_000
DREAM_JOURNAL_BYTES = 4_000_000
EVENT_JOURNAL_BYTES = 16_000_000


def append_thinned(path, line, max_bytes):
    """append one line; past `max_bytes`, keep alternate lines and the last."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
    if path.stat().st_size <= max_bytes:
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = lines[::2]
    if len(lines) % 2 == 0:
        kept.append(lines[-1])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(kept), encoding="utf-8")
    os.replace(temporary, path)


class Worker:
    def __init__(self, config_path, stop_path):
        self.config_path = Path(config_path).resolve()
        self.stop_path = Path(stop_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.manifest_path = Path(self.config.get("_run_manifest_path", ""))
        self.event_path = Path(self.config.get("_run_event_path", ""))
        self.metrics_path = Path(self.config.get("_run_metrics_path", ""))
        self.dreams_path = (
            self.metrics_path.with_name(
                self.metrics_path.name.replace(".metrics.jsonl", ".dreams.jsonl"))
            if str(self.metrics_path) else Path("")
        )
        self.manifest = {
            "version": 1,
            "run_id": self.config.get("_run_id", ""),
            "pid": os.getpid(),
            "status": "starting",
            "started_at": time.time(),
            "heartbeat_at": time.time(),
            "checkpoint_path": self.config.get("save_path", ""),
            "source_path": self.config.get("corpus", ""),
        }
        self.model = None
        self.last_save = time.time()
        self.last_metrics = 0.0
        self.last_report_emit = 0.0
        self.pending_report = None
        self.save_on_stop = True

    def journal(self, event):
        if str(self.event_path):
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            # The GUI tails this journal live, so it rotates instead of
            # thinning: two segments at most.
            if self.event_path.exists() and self.event_path.stat().st_size >= EVENT_JOURNAL_BYTES:
                previous = self.event_path.with_suffix(".previous.jsonl")
                previous.unlink(missing_ok=True)
                os.replace(self.event_path, previous)
            with self.event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def event(self, kind, payload=None):
        event = [kind] if payload is None else [kind, payload]
        emit(kind, payload)
        try:
            self.journal(event)
        except OSError:
            pass

    def update_manifest(self, **changes):
        self.manifest.update(changes)
        self.manifest["heartbeat_at"] = time.time()
        if str(self.manifest_path):
            try:
                self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(self.manifest_path, self.manifest)
            except OSError:
                pass

    def read_stop(self):
        if not self.stop_path.exists():
            return False
        try:
            request = json.loads(self.stop_path.read_text(encoding="utf-8"))
            self.save_on_stop = bool(request.get("save", True))
        except (OSError, json.JSONDecodeError, TypeError):
            self.save_on_stop = True
        return True

    def save(self):
        path = self.config["save_path"]
        self.event("saving", path)
        self.update_manifest(status="saving")
        self.model.save(path)
        self.last_save = time.time()
        self.event("checkpoint_saved", path)
        self.update_manifest(
            status="running", last_save_at=self.last_save,
            checkpoint_path=str(path),
        )

    def report(self, report):
        now = time.time()
        self.pending_report = report
        report_interval = max(
            0.0, float(self.config.get("report_interval_seconds", 0.0)))
        if report_interval and now - self.last_report_emit < report_interval:
            self.update_manifest(status="running")
            self._maybe_autosave(now)
            return
        self._publish_report(report, now)

    def _publish_report(self, report, now=None):
        now = time.time() if now is None else now
        self.event("report", report)
        self.last_report_emit = now
        self.pending_report = None
        self.update_manifest(
            status="running", last_report=report, last_report_at=now,
        )
        if str(self.metrics_path) and now - self.last_metrics >= 60.0:
            try:
                self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
                append_thinned(
                    self.metrics_path, json.dumps({"time": now, **report}) + "\n",
                    METRICS_JOURNAL_BYTES,
                )
                self.last_metrics = now
            except OSError:
                pass
        self._maybe_autosave(now)

    def _maybe_autosave(self, now):
        autosave = 60.0 * float(self.config.get("autosave_min", 30) or 0)
        if autosave > 0 and now - self.last_save >= autosave:
            try:
                self.save()
            except Exception as error:
                self.event("warning", f"autosave failed: {error}")

    def dream(self, text, updates, bytes_seen):
        payload = {
            "text": text, "updates": int(updates),
            "bytes_seen": int(bytes_seen), "time": time.time(),
        }
        self.event("dream", payload)
        # Dreams also keep their own thinned journal, so early dreams stay
        # comparable with late ones after the event journal has rotated.
        if str(self.dreams_path):
            try:
                append_thinned(
                    self.dreams_path,
                    json.dumps(payload, ensure_ascii=False) + "\n",
                    DREAM_JOURNAL_BYTES,
                )
            except OSError:
                pass

    # what a run may set: a new model takes every CoreConfig field the run names; a continuation keeps its identity
    # (a different one is refused by the runtime) and takes the run's dynamics. anything the run leaves unset stays.
    def requested_config(self, base, continuation=False):
        values = base.public_dict()
        fields = {f.name: f for f in dataclasses.fields(CoreConfig)}
        given = dict(self.config)
        if "width" in given and str(given.get("hidden", "")).strip() == "":
            given["hidden"] = given["width"]                     # logOS's cloud agent names the learner width 'width'
        kept = {}
        for key, value in given.items():
            if key in fields and str(value).strip() != "":
                kind = type(values[key])
                v = (str(value).lower() in ("1", "true", "yes")) if kind is bool else kind(value)
                if continuation and key in IDENTITY and v != values[key]:
                    kept[key] = (values[key], v)                 # 1.2.0: a saved model keeps its shape (the form's
                    continue                                     # shape fields describe new models; old sidecars lack some)
                values[key] = v
        if kept:
            self.event("identity_kept", {k: {"model": a, "form": b} for k, (a, b) in kept.items()})
        return CoreConfig(**values).validate()

    def build_model(self):
        checkpoint = str(self.config.get("checkpoint", "")).strip()
        if checkpoint:
            model = SOMA.from_checkpoint(checkpoint, device=self.config.get("device", "auto"))
            model.reconfigure_dynamics(self.requested_config(model.config, continuation=True))
            if self.config.get("fork", False) in (True, "true", "True", 1, "1"):
                self.event("saving", self.config["save_path"])
                model.save_fork(self.config["save_path"])
                self.last_save = time.time()
                self.event("checkpoint_saved", self.config["save_path"])
            elif Path(checkpoint).resolve() != Path(self.config["save_path"]).resolve():
                raise ValueError("continue mode must publish to the loaded checkpoint")
            return model
        return SOMA(
            self.requested_config(CoreConfig()), device=self.config.get("device", "auto"),
            seed=int(self.config.get("seed", 7)),
            description=str(self.config.get("description", "")),
        )

    def validation(self, reason="interval"):
        path = str(self.config.get("validation_path", "")).strip()
        if not path:
            return
        maximum = max(1, int(
            float(self.config.get("validation_max_mb", 0.25)) * 1_000_000))
        self.event("validation_start", {"reason": reason, "path": path})
        self.update_manifest(status="validating")
        started = time.time()
        result = self.model.evaluate(path, max_bytes=maximum)
        payload = {
            "bpb": result.bpb, "token_loss": result.token_loss,
            "top1": result.top1, "top5": result.top5,
            "source_bytes": result.source_bytes, "tokens": result.tokens,
            "seconds": time.time() - started, "timestamp": time.time(),
        }
        self.event("validation", payload)
        self.update_manifest(
            status="running", last_validation=payload,
            last_validation_at=payload["timestamp"],
            validation_path=path,
        )

    def run(self):
        try:
            self.update_manifest(status="starting")
            self.model = self.build_model()
            if getattr(self.model, "grafted_parameters", None):
                self.event(
                    "info",
                    "grafted a fresh sentinel pointer onto this checkpoint's cortex",
                )
            summary = dict(self.model.summary())
            summary.update({
                "run_id": self.config.get("_run_id", ""),
                "checkpoint_id": self.model.checkpoint_id,
                "revision_id": self.model.revision_id,
                "corpus": self.config.get(
                    "corpus_label", Path(self.config["corpus"]).name),
                "save_path": Path(self.config["save_path"]).name,
                "save_path_full": str(self.config["save_path"]),
                "bytes_seen": self.model.bytes_seen,
                "tokens_seen": self.model.tokens_seen,
                "tokenizer": self.model.tokenizer.vocab_size,
                "device": str(self.model.device),
                "dtype": str(next(self.model.net.parameters()).dtype).replace("torch.", ""),
                "loop": bool(self.config.get("loop", False)),
            })
            self.event("start", summary)
            self.event("summary", summary)
            self.update_manifest(status="running", summary=summary)
            if not str(self.config.get("checkpoint", "")).strip():
                # Publish recoverable ground before consuming the first byte.
                self.save()

            validation_interval = int(
                float(self.config.get("validation_every_mb", 100)) * 1_000_000)
            next_validation = (
                self.model.bytes_seen + validation_interval
                if self.config.get("validation_path") and validation_interval > 0
                else None
            )
            base_report = self.report

            def report_with_validation(report):
                nonlocal next_validation
                base_report(report)
                if next_validation is not None and self.model.bytes_seen >= next_validation:
                    self.validation()
                    while next_validation <= self.model.bytes_seen:
                        next_validation += validation_interval

            cycle_files = [
                str(Path(value).expanduser().resolve())
                for value in self.config.get("cycle_files", [])
            ]
            if cycle_files:
                sources = cycle_files * max(1, int(self.config.get("cycle_count", 1)))
            else:
                sources = [str(Path(self.config["corpus"]).expanduser().resolve())]

            stopped = False
            for source_index, source in enumerate(sources):
                if self.read_stop():
                    stopped = True
                    break
                self.event("info", f"reading {Path(source).name}")
                resume = (
                    None if self.config.get("start_byte_override", False)
                    else self.model.resume_state_for(source)
                )
                self.model.train(
                    source,
                    epochs=(int(self.config.get("epochs", 1))
                            if not cycle_files else 1),
                    save_path="",
                    start_byte=(int(self.config.get("start_byte", 0))
                                if source_index == 0 else 0),
                    max_bytes=int(self.config.get("max_bytes", 0)),
                    report_every=max(1, int(self.config.get("report_every", 1))),
                    report_callback=report_with_validation,
                    dream_every_batches=int(
                        self.config.get("dream_every_batches", 64)),
                    dream_length=int(self.config.get("dream_length", 128)),
                    dream_temperature=float(
                        self.config.get("dream_temperature", 0.8)),
                    dream_callback=self.dream,
                    dream_stateful=bool(
                        self.config.get("dream_stateful", False)),
                    resume_state=resume,
                    stop_callback=self.read_stop,
                )
                if self.stop_path.exists():
                    stopped = True
                    break

            # A finite source can finish inside the wall-clock reporting
            # throttle. Publish its last observation before stream mode idles,
            # otherwise the cockpit can appear stranded near byte zero.
            if self.pending_report is not None:
                self._publish_report(self.pending_report)

            # Stream mode follows an append-only file. Each pass begins at the
            # last exact source byte and waits quietly when the producer has
            # not appended new information.
            while bool(self.config.get("loop", False)) and not self.read_stop():
                position = self.model.stream_position()
                source = sources[0]
                size = Path(source).stat().st_size
                if size <= position:
                    self.update_manifest(status="running")
                    time.sleep(min(5.0, float(self.config.get("cycle_pause", 60))))
                    continue
                self.model.train(
                    source, epochs=1, save_path="", start_byte=position,
                    exact_start=True,
                    report_every=max(1, int(self.config.get("report_every", 1))),
                    report_callback=report_with_validation,
                    dream_every_batches=int(
                        self.config.get("dream_every_batches", 64)),
                    dream_length=int(self.config.get("dream_length", 128)),
                    dream_temperature=float(
                        self.config.get("dream_temperature", 0.8)),
                    dream_callback=self.dream,
                    stop_callback=self.read_stop,
                )
                if self.pending_report is not None:
                    self._publish_report(self.pending_report)

            if self.config.get("validation_path") and not stopped:
                self.validation("final")
            if self.pending_report is not None:
                self._publish_report(self.pending_report)
            if self.save_on_stop:
                self.save()
                status = "stopped" if self.stop_path.exists() else "complete"
            else:
                status = "stopped_without_save"
            diagnostics = self.model.diagnostics()
            self.event("done", {"status": status, "diagnostics": diagnostics})
            self.update_manifest(
                status=status, ended_at=time.time(), diagnostics=diagnostics)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self.event("error", message)
            self.event("traceback", traceback.format_exc())
            self.update_manifest(
                status="failed", ended_at=time.time(), error=message)
            raise


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: soma_train_worker.py config.json stop.json")
    Worker(sys.argv[1], sys.argv[2]).run()


if __name__ == "__main__":
    main()
