#!/usr/bin/env python3
"""resumable cloud corpus builder for the soma v13 curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path


sources = {
    "tinystories": {
        "dataset": "roneneldan/TinyStories",
        "config": None,
        "split": "train",
        "text_field": "text",
        "license": "cdla-sharing-1.0",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories",
        "min_chars": 80,
        "max_chars": 100_000,
    },
    "gutenberg": {
        "dataset": "common-pile/project_gutenberg",
        "config": None,
        "split": "train",
        "text_field": "text",
        "license": "public-domain source; retain source manifest",
        "url": "https://huggingface.co/datasets/common-pile/project_gutenberg",
        "min_chars": 2_000,
        "max_chars": 5_000_000,
    },
    "fineweb_edu": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "license": "odc-by",
        "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        "min_chars": 400,
        "max_chars": 250_000,
    },
    "wikipedia": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "text_field": "text",
        "license": "cc-by-sa-3.0 and gfdl",
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia",
        "min_chars": 500,
        "max_chars": 2_000_000,
    },
}

space_re = re.compile(r"[ \t]+")


def atomic_json(path: Path, value: dict) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    try:
        import ftfy
        value = ftfy.fix_text(value)
    except ImportError:
        pass
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = "\n".join(
        space_re.sub(" ", line).rstrip() for line in value.splitlines())
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip()


def acceptable(text: str, minimum: int, maximum: int) -> bool:
    if not minimum <= len(text) <= maximum:
        return False
    sample = text[:10_000].lower()
    if any(marker in sample for marker in (
            "lorem ipsum", "enable javascript", "access denied",
            "cloudflare ray id", "please enable cookies")):
        return False
    if any(marker in text for marker in ("â", "Ã", "Â", "\ufffd")):
        return False
    if sample.count("{") + sample.count("}") > 160:
        return False
    if sample.count("<") + sample.count(">") > 160:
        return False
    return True


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initial_manifest(name: str, spec: dict, chunk_bytes: int) -> dict:
    return {
        "format": 1,
        "source": name,
        "dataset": spec["dataset"],
        "config": spec["config"],
        "split": spec["split"],
        "license": spec["license"],
        "url": spec["url"],
        "chunk_bytes": chunk_bytes,
        "rows_seen": 0,
        "accepted_documents": 0,
        "rejected_documents": 0,
        "total_bytes": 0,
        "files": [],
        "complete_targets": [],
        "updated_at": time.time(),
    }


def load_manifest(path: Path, name: str, spec: dict, chunk_bytes: int) -> dict:
    if not path.exists():
        return initial_manifest(name, spec, chunk_bytes)
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = (spec["dataset"], spec["config"], spec["split"], chunk_bytes)
    actual = (
        value.get("dataset"), value.get("config"), value.get("split"),
        value.get("chunk_bytes"))
    if actual != expected:
        raise ValueError(
            f"manifest settings changed for {name}: {actual!r} != {expected!r}")
    for entry in value.get("files", []):
        path_value = path.parent / entry["name"]
        if not path_value.exists() or path_value.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"missing or truncated corpus chunk: {path_value}")
    return value


def open_dataset(spec: dict, rows_seen: int):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "corpus building requires datasets: pip install datasets") from error
    kwargs = {
        "path": spec["dataset"],
        "split": spec["split"],
        "streaming": True,
    }
    if spec["config"]:
        kwargs["name"] = spec["config"]
    dataset = load_dataset(**kwargs)
    if rows_seen:
        dataset = dataset.skip(rows_seen)
    return dataset


def build(name: str, root: Path, target_bytes: int, chunk_bytes: int) -> dict:
    spec = sources[name]
    output = root / name
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = load_manifest(
        manifest_path, name, spec, chunk_bytes)
    if manifest["total_bytes"] >= target_bytes:
        return manifest

    dataset = open_dataset(spec, int(manifest["rows_seen"]))
    chunk_index = len(manifest["files"])
    partial = output / f"{name}_{chunk_index:04d}.txt.partial"
    final = output / f"{name}_{chunk_index:04d}.txt"
    handle = partial.open("ab")
    chunk_size = partial.stat().st_size
    next_report = manifest["total_bytes"] + 128 * 1024 ** 2

    try:
        for row in dataset:
            manifest["rows_seen"] += 1
            text = clean_text(row.get(spec["text_field"], ""))
            if not acceptable(text, spec["min_chars"], spec["max_chars"]):
                manifest["rejected_documents"] += 1
                continue
            blob = (text + "\n\n").encode("utf-8", errors="replace")
            if chunk_size and chunk_size + len(blob) > chunk_bytes:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
                os.replace(partial, final)
                entry = {
                    "name": final.name,
                    "bytes": final.stat().st_size,
                    "sha256": hash_file(final),
                }
                manifest["files"].append(entry)
                manifest["total_bytes"] += entry["bytes"]
                manifest["updated_at"] = time.time()
                atomic_json(manifest_path, manifest)
                if manifest["total_bytes"] >= target_bytes:
                    break
                chunk_index += 1
                partial = output / f"{name}_{chunk_index:04d}.txt.partial"
                final = output / f"{name}_{chunk_index:04d}.txt"
                handle = partial.open("ab")
                chunk_size = 0

            handle.write(blob)
            chunk_size += len(blob)
            manifest["accepted_documents"] += 1
            if manifest["total_bytes"] + chunk_size >= next_report:
                print(
                    f"{name}: {(manifest['total_bytes'] + chunk_size) / 1024**3:.2f}gb",
                    flush=True)
                next_report += 128 * 1024 ** 2
            if manifest["total_bytes"] + chunk_size >= target_bytes:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
                os.replace(partial, final)
                entry = {
                    "name": final.name,
                    "bytes": final.stat().st_size,
                    "sha256": hash_file(final),
                }
                manifest["files"].append(entry)
                manifest["total_bytes"] += entry["bytes"]
                break
        else:
            if chunk_size:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
                os.replace(partial, final)
                entry = {
                    "name": final.name,
                    "bytes": final.stat().st_size,
                    "sha256": hash_file(final),
                }
                manifest["files"].append(entry)
                manifest["total_bytes"] += entry["bytes"]
    finally:
        if not handle.closed:
            handle.close()

    target_label = str(target_bytes)
    if manifest["total_bytes"] >= target_bytes:
        targets = set(manifest.get("complete_targets", []))
        targets.add(target_label)
        manifest["complete_targets"] = sorted(targets, key=int)
    manifest["updated_at"] = time.time()
    atomic_json(manifest_path, manifest)
    return manifest


def verify(root: Path, name: str) -> None:
    manifest_path = root / name / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = manifest_path.parent / entry["name"]
        if path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"size mismatch: {path}")
        if hash_file(path) != entry["sha256"]:
            raise RuntimeError(f"checksum mismatch: {path}")
    print(f"{name}: {len(manifest['files'])} chunks verified")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=tuple(sources))
    parser.add_argument("--root", type=Path, default=Path("data/cloud"))
    parser.add_argument("--target-gb", type=float, required=True)
    parser.add_argument("--chunk-mb", type=int, default=64)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    args = parser.parse_args()

    free = shutil.disk_usage(args.root.parent if args.root.parent.exists()
                             else Path(".")).free
    if free < args.min_free_gb * 1024 ** 3:
        raise RuntimeError(
            f"only {free / 1024**3:.1f}gb free; refusing corpus build")
    if args.verify:
        verify(args.root, args.source)
        return 0
    manifest = build(
        args.source, args.root.expanduser().resolve(),
        int(args.target_gb * 1024 ** 3),
        int(args.chunk_mb * 1024 ** 2))
    target_bytes = int(args.target_gb * 1024 ** 3)
    if manifest["total_bytes"] < target_bytes:
        raise RuntimeError(
            f"{args.source} ended at {manifest['total_bytes'] / 1024**3:.3f}gb "
            f"before the {args.target_gb:.3f}gb target")
    print(json.dumps({
        "source": args.source,
        "gb": round(manifest["total_bytes"] / 1024 ** 3, 3),
        "chunks": len(manifest["files"]),
        "documents": manifest["accepted_documents"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
