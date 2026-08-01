import random
import tempfile
import unittest
from pathlib import Path

import torch

from build_soma_chat_curriculum import (
    CONTROL_PIECES, render_dialogue, validate_protocol, verify_tokenizer)
from prepare_soma_qa import record
from soma_v12_3 import SomaTokenizer, TokenCorpus, TokenSOMA


class ChatCurriculumTests(unittest.TestCase):
    def test_rendered_dialogue_matches_live_chat_order(self):
        rendered = render_dialogue(
            (("hello. my name is james.", "hello, james."),),
            random.Random(7))
        self.assertTrue(rendered.startswith(
            "<|user|>hello. my name is james.<|turn|><|soma|>"
            "hello, james.<|turn|>\n"))
        self.assertTrue(rendered.endswith("hello, james.<|turn|>\n"))
        self.assertNotIn("<|soma_state|>", rendered)

    def test_protocol_validation_rejects_legacy_format(self):
        with tempfile.TemporaryDirectory() as folder:
            valid = Path(folder) / "valid.txt"
            valid.write_text(render_dialogue(
                (("hello", "hello"),), random.Random(3)), encoding="utf-8")
            counts = validate_protocol(valid)
            self.assertEqual(counts["<|turn|>"], 2)

            legacy = Path(folder) / "legacy.txt"
            legacy.write_text(
                valid.read_text(encoding="utf-8") + "\x1e", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_protocol(legacy)

    def test_qa_record_uses_same_native_protocol(self):
        rendered = record(
            "What is two plus two", "Four", random.Random(9), 2)
        self.assertIn("<|user|>What is two plus two?<|turn|><|soma|>", rendered)
        self.assertNotIn("<|soma_state|>", rendered)
        self.assertTrue(rendered.endswith("Four.<|turn|>\n"))
        self.assertNotIn("\x1e", rendered)
        self.assertNotIn("<soma_state>", rendered)

    def test_markers_are_reserved_tokenizer_pieces(self):
        tokenizer = SomaTokenizer()
        for piece in CONTROL_PIECES:
            self.assertEqual(len(tokenizer.encode(piece)), 1)
        model_path = (
            Path(__file__).resolve().parents[1] /
            "tokenizer" / "soma-8k-v1.model")
        self.assertIn("reserved tokenizer pieces", verify_tokenizer(model_path))

    def test_rendered_dialogue_has_only_compact_native_boundaries(self):
        rendered = render_dialogue(
            (("hello", "hello"),), random.Random(11))
        self.assertEqual(
            rendered, "<|user|>hello<|turn|><|soma|>hello<|turn|>\n")

    def test_state_is_perceived_without_direct_online_learning(self):
        model = TokenSOMA(
            n_bands=2, hidden_dim=16, depth=1,
            batch_size=4, device="cpu")
        before_weights = {
            name: value.detach().clone()
            for name, value in model.net.state_dict().items()
        }
        before_trace = model.bank.traces.clone()
        model.ingest_self_state("chat", online=True)
        for name, value in model.net.state_dict().items():
            self.assertTrue(torch.equal(value, before_weights[name]))
        self.assertFalse(torch.equal(model.bank.traces, before_trace))

    def test_corpus_masks_state_but_keeps_recorded_language(self):
        rendered = render_dialogue(
            (("hello james", "hello back"),), random.Random(13))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dialogue.txt"
            path.write_text(rendered, encoding="utf-8")
            tokenizer = SomaTokenizer()
            corpus = TokenCorpus(path, tokenizer)
            ids = []
            weights = []
            try:
                while True:
                    block = corpus.next_block(3)
                    if block is None:
                        break
                    ids.extend(block[0].tolist())
                    weights.extend(corpus.last_loss_weights.tolist())
            finally:
                corpus.close()
        turn_id = tokenizer.control_ids["turn"]
        control_ids = set(tokenizer.control_ids.values())
        self.assertTrue(all(
            abs(weight) < 1e-7 or abs(weight - 0.1) < 1e-6
            for token_id, weight in zip(ids, weights)
            if token_id in control_ids))
        self.assertTrue(any(
            token_id == turn_id and abs(weight - 0.1) < 1e-6
            for token_id, weight in zip(ids, weights)))
        hello_ids = tokenizer.encode("hello james")
        reply_ids = tokenizer.encode("hello back")
        for sequence in (hello_ids, reply_ids):
            start = next(index for index in range(len(ids) - len(sequence) + 1)
                         if ids[index:index + len(sequence)] == sequence.tolist())
            self.assertTrue(all(weight == 1.0
                                for weight in weights[start:start + len(sequence)]))

    def test_state_mask_recovers_when_resume_starts_inside_state(self):
        rendered = (
            "<|user|>hello james<|turn|><|soma|><|soma_state|>\n"
            "source: chat\nbytes_seen: 10\n<|turn|>hello back<|turn|>\n")
        state_offset = rendered.index("bytes_seen:")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dialogue.txt"
            path.write_text(rendered, encoding="utf-8")
            corpus = TokenCorpus(
                path, SomaTokenizer(), start_byte=state_offset)
            try:
                block = corpus.next_block(8)
                self.assertIsNotNone(block)
                ids = block[0].tolist()
                turn_index = ids.index(
                    corpus.tokenizer.control_ids["turn"])
                self.assertTrue(all(
                    weight == 0.0
                    for weight in corpus.last_loss_weights[:turn_index + 1]))
                self.assertTrue(any(
                    weight == 1.0
                    for weight in corpus.last_loss_weights[turn_index + 1:]))
            finally:
                corpus.close()


if __name__ == "__main__":
    unittest.main()
