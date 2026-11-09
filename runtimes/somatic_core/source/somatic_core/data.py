"""The stream: B rows, each reading its own region of a corpus front to back, one segment of T tokens at a time.

Text is read with RMA's segment rule (protocol._read_segment, fixed 2026-09-23): a read ends immediately BEFORE a
space (else after a tab/CR/LF within 1024 bytes, else on a UTF-8 boundary). The tokenizer adds no dummy prefix, so this
keeps each space with its word and makes the token stream identical to encoding the text whole (tests.py).

bpb is exact: every token carries its byte length (the decoded pieces concatenate to the source bytes, tests.py).
Bytes that aren't valid UTF-8 become byte pieces, so any byte stream is read exactly. iMessage-derived text is
never read.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

try:                                                                    # the vendored soma-8k-v1 alphabet
    from .alphabet import ReferenceTokenizer
except ImportError:
    from alphabet import ReferenceTokenizer

DATA = Path.home() / "Library" / "Application Support" / "soma" / "data"
FORBIDDEN = ("imessage",)


def tokenizer():
    return ReferenceTokenizer(Path(__file__).resolve().parent / "tokenizer" / "soma-8k-v1.model")


_INVALID = re.compile("[\udc80-\udcff]+")


def byte_piece0(tok):
    """the id of the byte piece <0x00>; byte b is byte_piece0 + b."""
    sp = tok.processor
    b0 = sp.piece_to_id("<0x00>")
    assert sp.piece_to_id("<0xFF>") == b0 + 255 and all(sp.is_byte(b0 + i) for i in (0, 128, 255))
    return b0


def encode_bytes(tok, raw):
    """token ids for raw bytes, exactly: valid UTF-8 is encoded as text; every byte that isn't valid UTF-8 becomes its
    own byte piece (never a replacement character), so any byte stream round-trips and bpb stays byte-exact."""
    text = raw.decode("utf-8", "surrogateescape")
    if not _INVALID.search(text):
        return tok.encode(text)
    b0, out, pos = byte_piece0(tok), [], 0
    for m in _INVALID.finditer(text):
        if m.start() > pos:
            out.append(tok.encode(text[pos:m.start()]))
        out.append(np.array([b0 + (ord(ch) - 0xDC00) for ch in m.group()], np.int64))
        pos = m.end()
    if pos < len(text):
        out.append(tok.encode(text[pos:]))
    return np.concatenate(out)


def ids_to_bytes(tok, ids):
    """the exact bytes a token sequence stands for (byte pieces are raw bytes; others are their UTF-8 text)."""
    sp, b0 = tok.processor, byte_piece0(tok)
    out = bytearray()
    for i in map(int, ids):
        if b0 <= i < b0 + 256:
            out.append(i - b0)
        else:
            piece = sp.id_to_piece(i)
            if not (piece.startswith("<|") and piece.endswith("|>")) and (sp.is_control(i) or sp.is_unknown(i)):
                continue
            out += piece.replace("\u2581", " ").encode("utf-8")
    return bytes(out)


def byte_lengths(tok):
    sp = tok.processor
    out = np.zeros(sp.get_piece_size(), np.int64)
    for i in range(len(out)):
        piece = sp.id_to_piece(i)
        if sp.is_byte(i):
            out[i] = 1
        elif piece.startswith("<|") and piece.endswith("|>"):
            out[i] = len(piece.encode("utf-8"))                 # control pieces stand for their literal text in a corpus
        elif sp.is_control(i) or sp.is_unknown(i):
            out[i] = 0
        else:
            out[i] = len(piece.replace("▁", " ").encode("utf-8"))
    return out


def read_segment(f, limit, target):
    """RMA's reader: `target` bytes, extended to end immediately before a space."""
    remaining = limit - f.tell()
    if remaining <= 0:
        return b""
    raw = f.read(min(max(256, int(target)), remaining))
    if f.tell() < limit:
        start = f.tell()
        ahead = f.read(min(1025, limit - start))
        cut = ahead.find(b" ", 0, 1024)
        if cut < 0:
            cut = next((i + 1 for i, c in enumerate(ahead[:1024]) if c in b"\t\r\n"), -1)
        if cut < 0:
            cut = min(1024, len(ahead))
            while 0 < cut < len(ahead) and ahead[cut] & 0xC0 == 0x80:
                cut -= 1
        raw += ahead[:cut]
        f.seek(start + cut)
    return raw


class Stream:
    """B rows over [start, end) of one file; row b begins at start + b (end - start) / B and wraps within its region.
    The first read of a row is aligned to the next line start, so rows begin on whole lines."""

    def __init__(self, path, B, tok, start=0.0, end=1.0, read_bytes=4096, byte_range=None, exact=False):
        """exact=True starts every row exactly at its region's first byte (an append-only stream resumes at the exact
        committed byte); otherwise rows start at the next line."""
        path = Path(path)
        assert not any(w in path.name.lower() for w in FORBIDDEN), "never stream iMessage-derived data"
        self.path, self.B, self.tok, self.read_bytes = path, B, tok, read_bytes
        size = path.stat().st_size
        self.lo, self.hi = (int(size * start), int(size * end)) if byte_range is None else \
            (max(0, int(byte_range[0])), min(size, int(byte_range[1])))
        span = (self.hi - self.lo) // B
        self.bounds = [(self.lo + b * span, self.lo + (b + 1) * span) for b in range(B)]
        self.files = [open(path, "rb") for _ in range(B)]
        self.buf = [np.zeros(0, np.int64) for _ in range(B)]
        self.bytes_read = np.zeros(B, np.int64)
        self.wraps = np.zeros(B, np.int64)                               # completed passes over each row's region
        self.exact = exact
        for b, f in enumerate(self.files):
            if exact:
                f.seek(self.bounds[b][0])
            else:
                self._align(b, self.bounds[b][0])

    def _align(self, b, pos):
        """start at the next line (else the next space) within 64 KB, never past the region: some corpora are one line."""
        f = self.files[b]
        f.seek(pos)
        if pos == 0:
            return
        ahead = f.read(1 << 16)
        cut = ahead.find(b"\n")
        cut = cut + 1 if cut >= 0 else ahead.find(b" ")
        f.seek(pos + cut if 0 <= cut and pos + cut < self.bounds[b][1] else pos)

    def next(self, T):
        """the next T tokens of every row, (B, T) int64."""
        out = np.zeros((self.B, T), np.int64)
        for b in range(self.B):
            empty = 0
            while len(self.buf[b]) < T:
                raw = read_segment(self.files[b], self.bounds[b][1], self.read_bytes)
                if not raw:                                   # the region is exhausted: wrap to its start
                    empty += 1
                    assert empty < 3, f"row {b} of {self.path.name} yields no text"
                    self.wraps[b] += 1
                    self._align(b, self.bounds[b][0])
                    continue
                self.bytes_read[b] += len(raw)
                ids = encode_bytes(self.tok, raw)
                self.buf[b] = np.concatenate([self.buf[b], ids])
            out[b], self.buf[b] = self.buf[b][:T], self.buf[b][T:]
        return out

    def state(self):
        return dict(pos=[f.tell() for f in self.files], buf=[b.copy() for b in self.buf],
                    bytes_read=self.bytes_read.copy(), wraps=self.wraps.copy())

    def load(self, st):
        for f, p in zip(self.files, st["pos"]):
            f.seek(p)
        self.buf = [b.copy() for b in st["buf"]]
        self.bytes_read = st["bytes_read"].copy()
        if "wraps" in st:
            self.wraps = st["wraps"].copy()
