"""immutable text alphabet for soma rma v1."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from .protocol import CONTROL_NAMES, IncrementalTokenDecoder


TOKENIZER_NAME = "soma-8k-v1"


def default_tokenizer_path() -> Path:
    return Path(__file__).resolve().parent / "tokenizer" / f"{TOKENIZER_NAME}.model"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReferenceTokenizer:
    """the fixed 8k sentencepiece alphabet used by this species."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or default_tokenizer_path()).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"missing tokenizer: {self.path}")
        self.sha256 = _sha256(self.path)
        self.processor = spm.SentencePieceProcessor(model_file=str(self.path))
        self.vocab_size = int(self.processor.vocab_size())
        self.control_ids = {
            name: int(self.processor.piece_to_id(f"<|{name}|>"))
            for name in CONTROL_NAMES
        }
        missing = [
            name for name, token_id in self.control_ids.items()
            if token_id < 0
            or self.processor.id_to_piece(token_id) != f"<|{name}|>"
        ]
        if missing:
            raise ValueError(
                "tokenizer is missing control pieces: " + ", ".join(missing)
            )
        self.generation_forbidden_ids = (
            int(self.processor.unk_id()),
            self.control_ids["soma_state"],
            self.control_ids["user"],
            self.control_ids["soma"],
            self.control_ids["dream"],
        )

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(
            self.processor.encode(str(text), out_type=int), dtype=np.int64
        )

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return self.processor.decode([int(value) for value in ids])

    def piece(self, token_id: int) -> str:
        return self.processor.id_to_piece(int(token_id))

    def diagnostic_piece(self, token_id: int) -> str:
        token_id = int(token_id)
        for name, control_id in self.control_ids.items():
            if token_id == control_id:
                return f"[{name}]"
        piece = self.piece(token_id)
        if piece == "<unk>":
            return "[unknown]"
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            try:
                value = int(piece[3:5], 16)
            except ValueError:
                return piece
            if value == 0x0A:
                return "\\n"
            if value == 0x09:
                return "\\t"
            if value == 0x0D:
                return "\\r"
            if 0x20 <= value <= 0x7E:
                return chr(value)
            return f"[byte {value:02x}]"
        return "_" + piece[1:] if piece.startswith("▁") else piece

    def incremental_decoder(self) -> IncrementalTokenDecoder:
        return IncrementalTokenDecoder(self)
