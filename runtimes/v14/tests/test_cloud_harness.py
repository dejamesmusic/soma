import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


root = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("build_cloud_corpus", root / "cloud/build_cloud_corpus.py")
runner = load("run_cloud_curriculum", root / "cloud/run_cloud_curriculum.py")
trainer = load("train_cloud", root / "cloud/train_cloud.py")


class CloudCorpusTests(unittest.TestCase):
    def test_builder_extends_existing_manifest(self):
        rows = [
            {"text": "one " * 50},
            {"text": "two " * 50},
            {"text": "three " * 50},
            {"text": "four " * 50},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            spec = dict(builder.sources["tinystories"])
            spec["min_chars"] = 1
            builder.sources["test"] = spec

            def dataset(_spec, rows_seen):
                return iter(rows[rows_seen:])

            with mock.patch.object(builder, "open_dataset", dataset):
                first = builder.build("test", path, 300, 320)
                second = builder.build("test", path, 900, 320)
            self.assertGreaterEqual(first["total_bytes"], 300)
            self.assertGreaterEqual(second["total_bytes"], 900)
            self.assertEqual(second["rows_seen"], 4)
            self.assertEqual(second["accepted_documents"], 4)
            names = [entry["name"] for entry in second["files"]]
            self.assertEqual(len(names), len(set(names)))

    def test_manifest_detects_truncated_chunk(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            spec = builder.sources["tinystories"]
            manifest_path = path / "manifest.json"
            value = builder.initial_manifest("tinystories", spec, 64)
            value["files"] = [{"name": "bad.txt", "bytes": 3, "sha256": "x"}]
            (path / "bad.txt").write_bytes(b"no")
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                builder.load_manifest(
                    manifest_path, "tinystories", spec, 64)


class CloudRecoveryTests(unittest.TestCase):
    def test_curriculum_uses_io2_decimation(self):
        plan = {
            "minimum_free_gb": 20,
            "model": {
                "bands": 48,
                "batch": 1024,
                "lr": "auto 0.0003",
                "decimation": 1.0,
                "update_interval": 1,
                "grad_clip": 1.0,
                "dream_every": 64,
                "dream_tokens": "128",
            },
        }
        command = runner.train_command(
            root, root / "run", root / "model.pt", root / "data.txt",
            {"name": "test"}, plan, 0)
        self.assertIn("--update-interval", command)
        self.assertIn("--decimation", command)
        self.assertEqual(command[command.index("--decimation") + 1], "1.0")

    def test_checkpoint_rotation_preserves_previous_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "model.pt"
            checkpoint.write_bytes(b"old")
            previous = trainer.rotate_checkpoint(checkpoint)
            temporary = Path(folder) / "model.pt.tmp"
            temporary.write_bytes(b"new")
            temporary.replace(checkpoint)
            self.assertEqual(previous.read_bytes(), b"old")
            self.assertEqual(checkpoint.read_bytes(), b"new")

    def test_recovery_cursor_requires_matching_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            corpus = path / "corpus.txt"
            checkpoint = path / "model.pt"
            corpus.write_text("hello", encoding="utf-8")
            checkpoint.write_bytes(b"x")
            recovery = path / "recovery.json"
            recovery.write_text(json.dumps({
                "data": str(corpus.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "position": 42,
            }), encoding="utf-8")
            self.assertEqual(
                runner.recovery_cursor(recovery, corpus, checkpoint), 42)
            other = path / "other.txt"
            other.write_text("x", encoding="utf-8")
            self.assertEqual(
                runner.recovery_cursor(recovery, other, checkpoint), 0)

    def test_milestone_is_immutable_after_current_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            checkpoint = path / "model.pt"
            checkpoint.write_bytes(b"stage one")
            saved = runner.milestone(checkpoint, path / "milestones", "one")
            replacement = path / "replacement"
            replacement.write_bytes(b"stage two")
            replacement.replace(checkpoint)
            self.assertEqual(saved.read_bytes(), b"stage one")
            self.assertEqual(checkpoint.read_bytes(), b"stage two")


if __name__ == "__main__":
    unittest.main()
