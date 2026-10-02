"""the body's alphabet: byte-level BPE, read from its tokenizer.json, in pure Python.

Reproduces HuggingFace `tokenizers` for SmolLM2's spec exactly (tests compare them token
for token): added tokens are split out first; every numeric character is isolated
(Digits, individual_digits); each remaining span is cut by the GPT-2 pattern, mapped
byte-for-byte onto printable characters and merged by rank.  Python's `re` has no
\\p{L}/\\p{N}, so those classes are built once from `unicodedata`.

soma's protocol controls (<|user|>, <|turn|>, ...) are carried by the body's reserved
code tokens, which chat text never produces; their literal strings encode to them.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .protocol import CONTROL_NAMES

# soma control name -> the body's reserved token that carries it
CONTROL_CARRIERS = {
    "soma_state": "<repo_name>",
    "turn": "<|im_end|>",
    "user": "<|im_start|>",
    "soma": "<file_sep>",
    "dream": "<filename>",
}
# what the body reads in place of each control: soma's protocol is not the
# body's language, so it sees the conversation as plain text
CONTROL_RENDERINGS = {
    "soma_state": "\n",
    "turn": "\n",
    "user": "\nUser: ",
    "soma": "\nSoma: ",
    "dream": "\n\n",
}
WORD_CACHE = 65_536            # bounded: least recently used words are dropped


def _bytes_to_unicode():
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = printable[:]
    extra = 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            codes.append(256 + extra)
            extra += 1
    return {value: chr(code) for value, code in zip(printable, codes)}


def _ranges(predicate):
    """a regex character class body for every code point satisfying predicate."""
    parts, start = [], None
    for code in range(sys.maxunicode + 2):
        inside = code <= sys.maxunicode and predicate(chr(code))
        if inside and start is None:
            start = code
        elif not inside and start is not None:
            end = code - 1
            parts.append(re.escape(chr(start)) if start == end
                         else f"{re.escape(chr(start))}-{re.escape(chr(end))}")
            start = None
    return "".join(parts)


@lru_cache(maxsize=1)
def _patterns():
    category = unicodedata.category
    letters = _ranges(lambda c: category(c)[0] == "L")
    numbers = _ranges(lambda c: category(c)[0] == "N")
    # Rust's \s is the Unicode White_Space property
    white = "\t\n\x0b\x0c\r \x85\xa0  -     　"
    split = re.compile(
        rf"'s|'t|'re|'ve|'m|'ll|'d| ?[{letters}]+| ?[{numbers}]+"
        rf"| ?[^{white}{letters}{numbers}]+|[{white}]+(?![^{white}])|[{white}]+"
    )
    digit = re.compile(f"[{numbers}]")
    return split, digit


class BodyTokenizer:
    def __init__(self, directory):
        self.directory = Path(directory)
        path = self.directory / "tokenizer.json"
        raw = path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        spec = json.loads(raw)
        model = spec["model"]
        if model["type"] != "BPE":
            raise ValueError("body tokenizer must be byte-level BPE")
        self.vocab = dict(model["vocab"])
        self.pieces = {index: piece for piece, index in self.vocab.items()}
        merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in model["merges"]]
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.added = {token["content"]: int(token["id"]) for token in spec["added_tokens"]}
        for token, index in self.added.items():
            self.pieces[index] = token
        self.vocab_size = max(len(self.vocab), max(self.pieces) + 1)
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}
        self.control_ids = {name: self.added[CONTROL_CARRIERS[name]] for name in CONTROL_NAMES}
        # literal control strings in text (as soma's corpora write them) encode to controls
        self.literal = dict(self.added)
        for name, index in self.control_ids.items():
            self.literal[f"<|{name}|>"] = index
        self.special = re.compile("|".join(re.escape(t) for t in sorted(self.literal, key=len, reverse=True)))
        self.generation_forbidden_ids = tuple(sorted(
            index for token, index in self.added.items()
            if index != self.control_ids["turn"]
        ))
        self._words = OrderedDict()
        # text in which the body hands the turn back: the user control's rendering
        self.handback_text = CONTROL_RENDERINGS["user"].rstrip()
        self.control_renderings = {
            index: self.encode(CONTROL_RENDERINGS[name]).tolist()
            for name, index in self.control_ids.items()
        }

    # ------------------------------------------------------------------ encode

    def _merge(self, word):
        cached = self._words.get(word)
        if cached is not None:
            self._words.move_to_end(word)
            return cached
        symbols = list(word)
        ranks = self.ranks
        while len(symbols) > 1:
            best, best_rank = None, None
            for i in range(len(symbols) - 1):
                rank = ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = i, rank
            if best is None:
                break
            pair = (symbols[best], symbols[best + 1])
            merged, i = [], 0
            while i < len(symbols):           # merge every occurrence of the best pair
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        ids = tuple(self.vocab[s] for s in symbols)
        self._words[word] = ids
        if len(self._words) > WORD_CACHE:
            self._words.popitem(last=False)
        return ids

    def _encode_plain(self, text, out):
        split, digit = _patterns()
        encoder = self.byte_encoder
        position = 0
        spans = []
        for match in digit.finditer(text):                # Digits: isolate each numeral
            if match.start() > position:
                spans.append(text[position:match.start()])
            spans.append(match.group())
            position = match.end()
        if position < len(text):
            spans.append(text[position:])
        for span in spans:
            for word in split.findall(span):
                out.extend(self._merge("".join(encoder[b] for b in word.encode("utf-8"))))

    def encode(self, text) -> np.ndarray:
        text = str(text)
        out = []
        position = 0
        for match in self.special.finditer(text):
            self._encode_plain(text[position:match.start()], out)
            out.append(self.literal[match.group()])
            position = match.end()
        self._encode_plain(text[position:], out)
        return np.asarray(out, dtype=np.int64)

    # ------------------------------------------------------------------ decode

    def _bytes(self, token_id):
        piece = self.pieces.get(int(token_id), "")
        if int(token_id) in self.added.values():
            return piece.encode("utf-8")
        return bytes(self.byte_decoder[c] for c in piece)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()
        controls = set(self.control_ids.values())
        data = b"".join(self._bytes(i) for i in ids if int(i) not in controls)
        return data.decode("utf-8", errors="replace")

    def token_bytes(self, token_id) -> bytes:
        return b"" if int(token_id) in self.control_ids.values() else self._bytes(token_id)

    def piece(self, token_id: int) -> str:
        return self.pieces.get(int(token_id), "?")

    def diagnostic_piece(self, token_id: int) -> str:
        token_id = int(token_id)
        for name, control_id in self.control_ids.items():
            if token_id == control_id:
                return f"[{name}]"
        data = self._bytes(token_id)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return "[byte " + data.hex() + "]"
        if text == "\n":
            return "\\n"
        return "_" + text[1:] if text.startswith(" ") else text

    def incremental_decoder(self):
        return ByteIncrementalDecoder(self)


class ByteIncrementalDecoder:
    """streams text from byte-level tokens, holding back incomplete UTF-8."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pending = bytearray()

    def push(self, token_id: int) -> str:
        self.pending += self.tokenizer.token_bytes(token_id)
        try:
            text = bytes(self.pending).decode("utf-8")
        except UnicodeDecodeError as error:
            if error.reason == "unexpected end of data":
                complete = bytes(self.pending[:error.start])
                self.pending = self.pending[error.start:]
                return complete.decode("utf-8", errors="replace")
            text = bytes(self.pending).decode("utf-8", errors="replace")
        self.pending.clear()
        return text

    def finish(self) -> str:
        text = bytes(self.pending).decode("utf-8", errors="replace")
        self.pending.clear()
        return text
