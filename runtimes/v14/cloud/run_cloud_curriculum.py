#!/usr/bin/env python3
"""build and train the soma cloud curriculum as a resumable state machine."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], root: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def retry(action, attempts: int, status_path: Path, status: dict) -> None:
    for attempt in range(attempts + 1):
        try:
            action()
            return
        except (OSError, RuntimeError, subprocess.CalledProcessError):
            if attempt >= attempts:
                raise
            delay = min(300, 15 * (2 ** attempt))
            atomic_json(status_path, {
                **status,
                "event": "retry",
                "attempt": attempt + 1,
                "delay_seconds": delay,
                "updated_at": time.time(),
            })
            time.sleep(delay)


def ensure_free(path: Path, minimum_gb: float) -> None:
    free = shutil.disk_usage(path).free
    if free < minimum_gb * 1024 ** 3:
        raise RuntimeError(
            f"only {free / 1024**3:.1f}gb free; curriculum paused")


def ensure_build(
        name: str, build: dict, root: Path, data: Path,
        minimum_gb: float) -> None:
    ensure_free(data, minimum_gb)
    if build["kind"] == "dataset":
        run([
            sys.executable, "cloud/build_cloud_corpus.py",
            build["source"], "--root", str(data / "cloud"),
            "--target-gb", str(build["target_gb"]),
            "--chunk-mb", str(build["chunk_mb"]),
            "--min-free-gb", str(minimum_gb),
        ], root)
        return
    if build["kind"] == "conversation":
        expected = data / "conversation_worlds_v4.txt"
        if expected.exists():
            return
        run([
            sys.executable, "build_soma_conversation_worlds.py",
            "--out-dir", str(data),
        ], root)
        cache = data / ".conversation_source_cache"
        if cache.exists():
            shutil.rmtree(cache)
        return
    raise ValueError(f"unknown build kind for {name}: {build['kind']}")


def resolve_files(data: Path, patterns: list[str]) -> list[Path]:
    found = []
    for pattern in patterns:
        found.extend(Path(value) for value in glob.glob(
            str(data / pattern)))
    files = sorted({
        value.resolve() for value in found
        if value.is_file() and not value.name.endswith(".partial")
    })
    if not files:
        raise FileNotFoundError(f"no corpus files for {patterns!r}")
    return files


def recovery_cursor(recovery_path: Path, corpus: Path, checkpoint: Path) -> int:
    value = load_json(recovery_path, {})
    if (value.get("data") == str(corpus.resolve())
            and value.get("checkpoint") == str(checkpoint.resolve())):
        return max(0, int(value.get("position", 0)))
    return 0


def milestone(checkpoint: Path, directory: Path, stage: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{stage}{checkpoint.suffix}"
    temporary = Path(f"{target}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(checkpoint, temporary)
    except OSError:
        shutil.copy2(checkpoint, temporary)
    os.replace(temporary, target)
    return target


def default_state(plan: dict) -> dict:
    return {
        "format": 1,
        "curriculum": plan["name"],
        "stage_index": 0,
        "epoch_index": 0,
        "file_index": 0,
        "completed": False,
        "updated_at": time.time(),
    }


def train_command(
        root: Path, work: Path, checkpoint: Path, corpus: Path,
        stage: dict, plan: dict, cursor: int) -> list[str]:
    model = plan["model"]
    command = [
        sys.executable, "cloud/train_cloud.py",
        "--data", str(corpus),
        "--workdir", str(work),
        "--checkpoint", checkpoint.name,
        "--bands", str(model["bands"]),
        "--batch", str(model["batch"]),
        "--lr", str(model["lr"]),
        "--decimation", str(model.get("decimation", 1.0)),
        "--update-interval", str(model["update_interval"]),
        "--grad-clip", str(model["grad_clip"]),
        "--epochs", "1",
        "--start-byte", str(cursor),
        "--report-bytes", "5000000",
        "--save-bytes", "100000000",
        "--dream-every", str(model["dream_every"]),
        "--dream-tokens", str(model["dream_tokens"]),
        "--stage", stage["name"],
        "--min-free-gb", str(plan["minimum_free_gb"]),
    ]
    if checkpoint.exists():
        command.append("--resume")
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", type=Path,
        default=Path(__file__).with_name("curriculum_v1.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--workdir", type=Path, default=Path("run"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--stop-after", default="")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    work = (root / args.workdir).resolve()
    data = (root / args.data).resolve()
    work.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    state_path = work / "curriculum_state.json"
    status_path = work / "curriculum_status.json"
    state = load_json(state_path, default_state(plan))
    checkpoint = work / "checkpoints" / plan["checkpoint"]
    recovery = work / "logs" / "recovery.json"
    stages = plan["stages"]

    if state.get("completed"):
        print("curriculum already complete")
        return 0

    while state["stage_index"] < len(stages):
        stage = stages[state["stage_index"]]
        build_name = stage["build"]
        build = plan["builds"][build_name]
        status = {
            **state,
            "event": "building",
            "stage": stage["name"],
            "build": build_name,
            "updated_at": time.time(),
        }
        atomic_json(status_path, status)
        retry(
            lambda: ensure_build(
                build_name, build, root, data, plan["minimum_free_gb"]),
            args.max_retries, status_path, status)
        files = resolve_files(data, stage["files"])
        if args.prepare_only:
            print(f"prepared {stage['name']}: {len(files)} files")
            return 0

        for epoch_index in range(state["epoch_index"], int(stage["epochs"])):
            first_file = state["file_index"] if (
                epoch_index == state["epoch_index"]) else 0
            for file_index in range(first_file, len(files)):
                corpus = files[file_index]
                cursor = recovery_cursor(recovery, corpus, checkpoint)
                state.update({
                    "epoch_index": epoch_index,
                    "file_index": file_index,
                    "current_file": str(corpus),
                    "cursor": cursor,
                    "updated_at": time.time(),
                })
                atomic_json(state_path, state)
                atomic_json(status_path, {
                    **state, "event": "training",
                    "stage": stage["name"],
                })
                command = train_command(
                    root, work, checkpoint, corpus, stage, plan, cursor)
                for attempt in range(args.max_retries + 1):
                    try:
                        run(command, root)
                        break
                    except subprocess.CalledProcessError:
                        if attempt >= args.max_retries:
                            raise
                        delay = min(300, 15 * (2 ** attempt))
                        atomic_json(status_path, {
                            **state, "event": "retry",
                            "stage": stage["name"],
                            "attempt": attempt + 1,
                            "delay_seconds": delay,
                        })
                        time.sleep(delay)
                        cursor = recovery_cursor(
                            recovery, corpus, checkpoint)
                        command = train_command(
                            root, work, checkpoint, corpus,
                            stage, plan, cursor)
                state["file_index"] = file_index + 1
                state["cursor"] = 0
                state["updated_at"] = time.time()
                atomic_json(state_path, state)
            state["epoch_index"] = epoch_index + 1
            state["file_index"] = 0
            atomic_json(state_path, state)

        saved = milestone(
            checkpoint, work / "checkpoints" / "milestones", stage["name"])
        state.update({
            "stage_index": state["stage_index"] + 1,
            "epoch_index": 0,
            "file_index": 0,
            "cursor": 0,
            "last_milestone": str(saved),
            "updated_at": time.time(),
        })
        atomic_json(state_path, state)
        atomic_json(status_path, {
            **state, "event": "stage_complete",
            "stage": stage["name"],
        })
        if args.stop_after == stage["name"]:
            return 0

    state["completed"] = True
    state["updated_at"] = time.time()
    atomic_json(state_path, state)
    atomic_json(status_path, {**state, "event": "complete"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
