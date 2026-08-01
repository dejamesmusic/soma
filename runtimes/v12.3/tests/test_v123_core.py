import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from soma_gui import (
    _checkpoint_profile, _fmt_bytes, _fmt_tokens, _is_trainable_corpus,
    _read_checkpoint_metadata, _write_checkpoint_profile)

from soma_v12_3 import (
    ProjectedTokenBank, SomaTokenizer, TokenCorpus, TokenSOMA, build_parser,
    TranslatedErrorBank)


class TokenizerTests(unittest.TestCase):
    def test_lossless_roundtrip(self):
        tokenizer = SomaTokenizer()
        text = "<|user|>hello, soma.\nbytes: \x00 \x1e café 日本語<|turn|>"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_incremental_decode_matches_full_decode(self):
        tokenizer = SomaTokenizer()
        text = "ordinary words, café, and 日本語.\n"
        ids = tokenizer.encode(text)
        decoder = tokenizer.incremental_decoder()
        streamed = "".join(decoder.push(token_id) for token_id in ids)
        streamed += decoder.finish()
        self.assertEqual(streamed, tokenizer.decode(ids))

    def test_corpus_emits_fixed_blocks_without_losing_source_bytes(self):
        tokenizer = SomaTokenizer()
        raw = ("one small sentence after another. " * 200).encode()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corpus.txt"
            path.write_bytes(raw)
            corpus = TokenCorpus(path, tokenizer)
            sizes = []
            source_bytes = 0
            try:
                while True:
                    block = corpus.next_block(17)
                    if block is None:
                        break
                    ids, consumed, _ = block
                    sizes.append(len(ids))
                    source_bytes += consumed
            finally:
                corpus.close()
        self.assertTrue(all(size == 17 for size in sizes[:-1]))
        self.assertLessEqual(sizes[-1], 17)
        self.assertEqual(source_bytes, len(raw))

    def test_resume_cursor_uses_aligned_absolute_source_position(self):
        tokenizer = SomaTokenizer()
        raw = ("first sentence.\nsecond sentence after alignment.\n" * 20).encode()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corpus.txt"
            path.write_bytes(raw)
            corpus = TokenCorpus(path, tokenizer, start_byte=3)
            try:
                aligned_start = corpus.start_byte
                block = corpus.next_block(8)
                self.assertIsNotNone(block)
                _, consumed, absolute_position = block
                self.assertGreater(aligned_start, 3)
                self.assertEqual(
                    absolute_position, aligned_start + consumed)
                self.assertEqual(absolute_position, corpus.position)
                self.assertEqual(corpus.limit, len(raw))
            finally:
                corpus.close()


class ProjectedBankTests(unittest.TestCase):
    def test_matches_sequential_reference(self):
        torch.manual_seed(7)
        vocab = 19
        dim = 5
        bands = 4
        embedding = nn.Embedding(vocab, dim)
        bank = ProjectedTokenBank(vocab, dim, bands, 1.618, torch.device("cpu"))
        ids = np.asarray([3, 7, 3, 11, 2, 18, 4], dtype=np.int64)

        trace = torch.zeros(vocab, bands, dtype=torch.float64)
        alpha = bank.alphas_cpu
        decay = bank.decay_cpu
        expected = []
        for token_id in ids:
            projected = trace.T.float() @ embedding.weight
            bandpassed = torch.empty_like(projected)
            bandpassed[:-1] = projected[:-1] - projected[1:]
            bandpassed[-1] = projected[-1]
            expected.append(bandpassed.T.reshape(-1))
            trace.mul_(decay.unsqueeze(0))
            trace[int(token_id)].add_(alpha)

        actual = bank.process_block_select(ids, embedding)
        self.assertTrue(torch.allclose(
            actual, torch.stack(expected), atol=2e-6, rtol=2e-5))
        self.assertTrue(torch.allclose(bank.traces, trace, atol=1e-12, rtol=1e-12))

    def test_dictionary_reinterprets_retained_identity(self):
        torch.manual_seed(11)
        embedding = nn.Embedding(23, 6)
        bank = ProjectedTokenBank(23, 6, 5, 1.618, torch.device("cpu"))
        bank._advance_exact(np.asarray([4, 4, 9, 12], dtype=np.int64))
        features = bank.process_block_select(
            np.asarray([2, 8], dtype=np.int64), embedding)
        features.square().sum().backward()
        self.assertGreater(float(embedding.weight.grad[4].abs().sum()), 0.0)
        self.assertGreater(float(embedding.weight.grad[2].abs().sum()), 0.0)

    def test_cached_projection_matches_exact_identity_update(self):
        torch.manual_seed(13)
        embedding = nn.Embedding(29, 7)
        bank = ProjectedTokenBank(29, 7, 5, 1.618, torch.device("cpu"))
        bank._advance_exact(np.asarray([2, 5, 2, 17], dtype=np.int64))
        reference = ProjectedTokenBank(
            29, 7, 5, 1.618, torch.device("cpu"))
        reference.traces.copy_(bank.traces)
        state = bank.projected_state(embedding)
        generated = np.asarray([8, 3, 21, 8, 1], dtype=np.int64)
        for token_id in generated:
            state = bank.advance_projected(state, token_id, embedding)
        reference._advance_exact(generated)
        exact = reference.projected_state(embedding)
        self.assertTrue(torch.allclose(state, exact, atol=2e-6, rtol=2e-5))


class ErrorBankTests(unittest.TestCase):
    def test_gap_update_matches_repeated_constant_error(self):
        value = torch.tensor([0.25, -0.5, 1.0, -0.1])
        gap = TranslatedErrorBank(6, 4, 1.618)
        repeated = TranslatedErrorBank(6, 4, 1.618)
        gap.advance(value, gaps=7)
        for _ in range(7):
            repeated.advance(value, gaps=1)
        self.assertTrue(torch.allclose(
            gap.trace, repeated.trace, atol=1e-12, rtol=1e-12))


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_profile_is_editable_without_losing_machine_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "sample.pt"
            meta = Path(f"{checkpoint}.meta.json")
            meta.write_text(
                '{"checkpoint_id":"abc","tokens_seen":42}',
                encoding="utf-8")
            profile = _checkpoint_profile(checkpoint)
            profile.update({
                "name": "cassian",
                "tagline": "learns in public",
                "art": "[] soma []",
                "about": "a test profile",
            })
            _write_checkpoint_profile(checkpoint, profile)
            saved = _read_checkpoint_metadata(checkpoint)
            self.assertEqual(saved["checkpoint_id"], "abc")
            self.assertEqual(saved["tokens_seen"], 42)
            self.assertEqual(saved["sidecar_version"], 2)
            self.assertEqual(saved["profile"], profile)

    def test_data_selector_excludes_metadata_and_compressed_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = {
                "corpus.txt": True,
                "enwik9": True,
                "corpus.txt.gz": False,
                "corpus_manifest.txt": False,
                "corpus_readme.txt": False,
                "corpus.json": False,
            }
            for name, included in expected.items():
                path = root / name
                path.touch()
                self.assertEqual(_is_trainable_corpus(path), included)

    def test_training_dreams_are_capped_in_tokens(self):
        model = TokenSOMA(
            n_bands=2, hidden_dim=16, depth=1, batch_size=8,
            lr=0.001, lr_auto=False, device="cpu")
        ordinary_id = next(
            token_id for token_id in model.tokenizer.encode("ordinary words")
            if token_id not in model.tokenizer.generation_forbidden_ids
            and token_id != model.tokenizer.control_ids["turn"])
        model._sample_token = (
            lambda temperature, projected=None: ordinary_id)
        generated = model.generate_ids(
            length=0, temperature=1.0, prelude=False, max_tokens=7)
        self.assertEqual(len(generated), 7)
        turn_id = model.tokenizer.control_ids["turn"]
        self.assertGreater(float(model.bank.traces[turn_id].sum()), 0.0)

    def test_generation_uses_compact_native_boundaries(self):
        model = TokenSOMA(
            n_bands=2, hidden_dim=16, depth=1, batch_size=8,
            lr=0.001, lr_auto=False, device="cpu")
        dream_ids = model.tokenizer.encode("<|dream|>")
        self.assertEqual(len(dream_ids), 1)
        model.ingest_self_state = lambda *args, **kwargs: self.fail(
            "default generation must not inject a state prelude")
        ordinary_id = next(
            token_id for token_id in model.tokenizer.encode("ordinary")
            if token_id not in model.tokenizer.generation_forbidden_ids)
        model._sample_token = (
            lambda temperature, projected=None: ordinary_id)
        model.generate_ids(length=0, max_tokens=1)

    def test_free_generation_length_is_tokens(self):
        model = TokenSOMA(
            n_bands=2, hidden_dim=16, depth=1, batch_size=8,
            lr=0.001, lr_auto=False, device="cpu")
        ordinary_id = next(
            token_id for token_id in model.tokenizer.encode("ordinary")
            if token_id not in model.tokenizer.generation_forbidden_ids)
        calls = []

        def sample(temperature, projected=None):
            calls.append(ordinary_id)
            return ordinary_id

        model._sample_token = sample
        list(model.generate(length=7, temperature=1.0))
        self.assertEqual(len(calls), 7)

        root = Path(__file__).resolve().parents[1]
        gui_source = (root / "soma_gui.py").read_text(encoding="utf-8")
        self.assertIn('"response max tokens"', gui_source)
        self.assertNotIn('"response max bytes"', gui_source)

    def test_experience_units_distinguish_tokens_from_bytes(self):
        self.assertEqual(_fmt_tokens(1_250_000_000), "1.2b")
        self.assertEqual(_fmt_bytes(1024 ** 3), "1.0gb")

    def test_dream_token_cli_and_gui_contract(self):
        args = build_parser().parse_args([
            "train", "corpus.txt", "--dream-tokens", "64"])
        self.assertEqual(args.dream_length, "64")
        root = Path(__file__).resolve().parents[1]
        gui_source = (root / "soma_gui.py").read_text(encoding="utf-8")
        self.assertIn('"dream max tokens"', gui_source)
        self.assertNotIn('"dream max bytes"', gui_source)
        self.assertIn('("rehearsal", "feedback")', gui_source)
        self.assertIn('("maturity", "saturation")', gui_source)

    def test_initial_window_ignores_transient_compact_geometry(self):
        root = Path(__file__).resolve().parents[1]
        gui_source = (root / "soma_gui.py").read_text(encoding="utf-8")
        self.assertIn("self._shell_layout_ready = False", gui_source)
        self.assertIn("self.root.after_idle(self._finish_initial_layout)",
                      gui_source)

    def test_gui_decimation_gain_uses_runtime_config_key(self):
        root = Path(__file__).resolve().parents[1]
        gui_source = (root / "soma_gui.py").read_text(encoding="utf-8")
        self.assertIn(
            'field(metabolism, "decimation gain", "decimation", 2)',
            gui_source,
        )
        self.assertNotIn(
            'field(metabolism, "decimation", "decimation gain", 2)',
            gui_source,
        )

    def test_decimation_setting_is_a_gain_on_io2_output(self):
        def settled_band(gain):
            model = TokenSOMA(
                n_bands=16, hidden_dim=32, depth=1, batch_size=8,
                lr=0.001, lr_auto=False, decimation_range=gain,
                device="cpu")
            model._error_energy = 0.04
            model._error_coherence = 1.0
            model._spectral_target_band = 4.0
            for _ in range(12):
                model._apply_controller(
                    np.log(model.vocab_size), 2.0 * np.log(2.0))
            return model.decimation_band, model._stride

        off_band, off_stride = settled_band(0.0)
        half_band, _ = settled_band(0.5)
        normal_band, _ = settled_band(1.0)
        double_band, _ = settled_band(2.0)
        capped_band, capped_stride = settled_band(100.0)
        self.assertEqual((off_band, off_stride), (0.0, 1))
        self.assertLess(half_band, normal_band)
        self.assertLess(normal_band, double_band)
        self.assertLessEqual(capped_band, np.log(256) / np.log(1.618))
        self.assertEqual(capped_stride, 256)

    def test_row_compressor_rejects_only_outward_capacity_growth(self):
        model = TokenSOMA(
            n_bands=2, hidden_dim=16, depth=1, batch_size=8,
            lr=0.001, lr_auto=False, device="cpu")
        module = model._row_modules()[1]
        ceiling = model._row_ceiling(module.weight)
        with torch.no_grad():
            module.weight.zero_()
            module.weight[:, 0] = 0.9 * ceiling
        model._reset_row_norm_reference()
        reference = module.weight[0].norm().item()
        with torch.no_grad():
            module.weight[:, 0] = 1.02 * ceiling
            module.weight[:, 1] = 0.08 * ceiling
            proposed_direction = module.weight[0] / module.weight[0].norm()
            proposed_norm = module.weight[0].norm().item()
        model._apply_weight_compressor(diagnose=True)
        compressed_norm = module.weight[0].norm().item()
        compressed_direction = module.weight[0] / module.weight[0].norm()
        self.assertGreater(compressed_norm, reference)
        self.assertLess(compressed_norm, proposed_norm)
        self.assertLessEqual(compressed_norm, ceiling)
        self.assertTrue(torch.allclose(
            compressed_direction, proposed_direction, atol=1e-6))
        self.assertGreater(model._row_compression_fraction, 0.0)

        with torch.no_grad():
            module.weight[:, 0] = 0.8 * ceiling
            module.weight[:, 1:] = 0.0
        model._apply_weight_compressor()
        self.assertAlmostEqual(
            module.weight[0].norm().item(), 0.8 * ceiling, places=5)

    def test_io2_throughput_relaxation_uses_one_bpb_boundary(self):
        one_bpb = np.log(2.0)
        self.assertEqual(TokenSOMA._throughput_relaxation(one_bpb), 0.0)
        self.assertEqual(TokenSOMA._throughput_relaxation(2 * one_bpb), 0.0)
        self.assertAlmostEqual(
            TokenSOMA._throughput_relaxation(0.5 * one_bpb), 0.5)
        self.assertEqual(TokenSOMA._throughput_relaxation(0.0), 1.0)

    def test_fresh_models_are_square_but_saved_width_is_preserved(self):
        square = TokenSOMA(
            n_bands=3, depth=2, batch_size=8, device="cpu")
        self.assertEqual(square.hidden_dim, 3 * square.token_dim)
        train_parser = build_parser()._subparsers._group_actions[0].choices[
            "train"]
        self.assertNotIn("--hidden", train_parser._option_string_actions)

        legacy_shape = TokenSOMA(
            n_bands=3, hidden_dim=31, depth=2,
            batch_size=8, device="cpu")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "non_square.pt"
            legacy_shape.save(path)
            loaded = TokenSOMA(
                n_bands=3, batch_size=8, device="cpu").load(path)
        self.assertEqual(loaded.hidden_dim, 31)

    def test_training_reports_ranked_token_and_spectral_diagnostics(self):
        model = TokenSOMA(
            n_bands=4, hidden_dim=32, depth=2, batch_size=16,
            lr=0.001, lr_auto=False, device="cpu")
        model._diagnostic_every = 1
        text = "the small model remembers likely tokens.\n"
        result = model._train_tokens(
            model.tokenizer.encode(text), len(text.encode()), 1)
        self.assertGreaterEqual(result["top5"], result["correct"])
        self.assertEqual(len(model._weight_spectral_drive), 4)
        self.assertTrue(np.isfinite(model._weight_spectral_drive).all())
        self.assertGreater(len(model._top_predictions), 0)
        self.assertGreater(len(model._active_memory), 0)

    def test_checkpoint_roundtrip(self):
        model = TokenSOMA(
            n_bands=3, hidden_dim=32, depth=2, batch_size=16,
            lr=0.001, lr_auto=False, auto_mode="off", device="cpu")
        ids = model.tokenizer.encode("one small step for a token model.\n")
        model._train_tokens(ids, len("one small step for a token model.\n"), 1)
        before = model.net(model.bank.tap(model.net.embedding).unsqueeze(0))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.pt"
            model.save(path)
            first_metadata = _read_checkpoint_metadata(path)
            first_revision = first_metadata["revision_id"]
            custom_profile = dict(first_metadata["profile"])
            custom_profile["name"] = "roundtrip"
            _write_checkpoint_profile(path, custom_profile)
            model.save(path)
            loaded = TokenSOMA(
                n_bands=3, hidden_dim=32, depth=2, batch_size=16,
                device="cpu").load(path)
            after = loaded.net(
                loaded.bank.tap(loaded.net.embedding).unsqueeze(0))
            self.assertTrue(torch.equal(before, after))
            self.assertEqual(loaded.tokenizer.model_sha256,
                             model.tokenizer.model_sha256)
            metadata = _read_checkpoint_metadata(path)
            self.assertEqual(metadata["checkpoint_id"], model.checkpoint_id)
            self.assertNotEqual(metadata["revision_id"], first_revision)
            self.assertEqual(metadata["profile"]["name"], "roundtrip")
            self.assertEqual(loaded.checkpoint_id, model.checkpoint_id)

    def test_initial_v123_scalar_error_checkpoint_migrates(self):
        model = TokenSOMA(
            n_bands=4, hidden_dim=24, depth=2, batch_size=8,
            device="cpu")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scalar.pt"
            model.save(path)
            checkpoint = torch.load(path, map_location="cpu",
                                    weights_only=False)
            checkpoint["runtime_version"] = "v12.3"
            checkpoint.pop("error_bank_kind", None)
            checkpoint["error_trace"] = torch.ones(4, dtype=torch.float64)
            torch.save(checkpoint, path)
            loaded = TokenSOMA(
                n_bands=4, hidden_dim=24, depth=2, batch_size=8,
                device="cpu").load(path)
            self.assertEqual(tuple(loaded.error_bank.trace.shape), (4, 256))
            self.assertEqual(float(loaded.error_bank.trace.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
