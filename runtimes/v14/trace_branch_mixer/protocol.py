"""token protocol and byte-exact streaming corpus for soma v13."""

from __future__ import annotations

from pathlib import Path

import numpy as np


TURN_LOSS_WEIGHT = 0.1
CONTROL_NAMES = ("soma_state", "turn", "user", "soma", "dream")


class IncrementalTokenDecoder:
    """sentencepiece-compatible streaming decode, including byte fallback."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pending_bytes = bytearray()

    def _flush_bytes(self, final: bool = False) -> str:
        if not self.pending_bytes:
            return ""
        try:
            text = bytes(self.pending_bytes).decode("utf-8")
        except UnicodeDecodeError as error:
            if not final and error.reason == "unexpected end of data":
                return ""
            text = bytes(self.pending_bytes).decode(
                "utf-8", errors="replace"
            )
        self.pending_bytes.clear()
        return text

    def push(self, token_id: int) -> str:
        piece = self.tokenizer.piece(token_id)
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            try:
                self.pending_bytes.append(int(piece[3:5], 16))
            except ValueError:
                return self._flush_bytes(final=True) + self.tokenizer.decode(
                    [token_id]
                )
            return self._flush_bytes(final=False)
        return self._flush_bytes(final=True) + self.tokenizer.decode([token_id])

    def finish(self) -> str:
        return self._flush_bytes(final=True)


class TokenCorpus:
    """stream token batches while preserving exact source-byte cursors."""

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        *,
        start_byte: int = 0,
        max_bytes: int = 0,
        turn_loss_weight: float = TURN_LOSS_WEIGHT,
        resume_state: dict | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.tokenizer = tokenizer
        self.turn_loss_weight = float(turn_loss_weight)
        self.file = self.path.open("rb")
        self.file_size = self.path.stat().st_size
        self.start_byte = max(0, min(int(start_byte), self.file_size))
        self.file.seek(self.start_byte)
        if self.start_byte:
            self._align_forward()
        self.start_byte = self.file.tell()
        self.limit = self.file_size
        if max_bytes and int(max_bytes) > 0:
            self.limit = min(
                self.file_size, self.start_byte + int(max_bytes)
            )
        self.bytes_consumed = 0
        self.tokens_consumed = 0
        self._pending: list[dict] = []
        self._pending_tokens = 0
        self._protocol_mode = "plain"
        newline_ids = np.asarray(self.tokenizer.encode("\n"), dtype=np.int64)
        self._newline_id = (
            int(newline_ids[0]) if len(newline_ids) == 1 else None
        )
        self.last_block_ended_at_newline = False
        self.last_loss_weights = np.empty(0, dtype=np.float32)
        self.last_loss_bytes = 0.0
        if resume_state is not None:
            self._restore_state(resume_state)
        elif self.start_byte:
            self._prime_protocol_mode()

    def state_dict(self) -> dict:
        """capture the exact read-ahead and token delivery boundary."""

        pending = []
        for segment in self._pending:
            pending.append({
                "ids": np.asarray(segment["ids"], dtype=np.int64).tolist(),
                "loss_weights": np.asarray(
                    segment["loss_weights"], dtype=np.float32
                ).tolist(),
                "offset": int(segment["offset"]),
                "source_bytes": int(segment["source_bytes"]),
                "committed_bytes": int(segment["committed_bytes"]),
                "committed_loss_bytes": float(
                    segment["committed_loss_bytes"]
                ),
                "boundary_commits": [
                    [int(token_offset), int(byte_offset)]
                    for token_offset, byte_offset in sorted(
                        segment.get("boundary_commits", {}).items()
                    )
                ],
            })
        return {
            "version": 1,
            "path": str(self.path),
            "file_size": int(self.file_size),
            "start_byte": int(self.start_byte),
            "limit": int(self.limit),
            "file_position": int(self.file.tell()),
            "bytes_consumed": int(self.bytes_consumed),
            "tokens_consumed": int(self.tokens_consumed),
            "pending_tokens": int(self._pending_tokens),
            "pending": pending,
            "protocol_mode": str(self._protocol_mode),
        }

    def _restore_state(self, state: dict) -> None:
        if not isinstance(state, dict) or int(state.get("version", 0)) != 1:
            raise ValueError("unsupported corpus resume state")
        limit = int(state.get("limit", -1))
        file_position = int(state.get("file_position", -1))
        start_byte = int(state.get("start_byte", -1))
        bytes_consumed = int(state.get("bytes_consumed", -1))
        tokens_consumed = int(state.get("tokens_consumed", -1))
        if not (
            0 <= start_byte <= file_position <= limit <= self.file_size
            and bytes_consumed >= 0
            and tokens_consumed >= 0
        ):
            raise ValueError("corpus resume coordinates are invalid")
        pending = []
        pending_tokens = 0
        for raw in state.get("pending", []):
            if not isinstance(raw, dict):
                raise ValueError("corpus pending state is invalid")
            ids = np.asarray(raw.get("ids", []), dtype=np.int64)
            weights = np.asarray(
                raw.get("loss_weights", []), dtype=np.float32
            )
            offset = int(raw.get("offset", -1))
            source_bytes = int(raw.get("source_bytes", -1))
            committed_bytes = int(raw.get("committed_bytes", -1))
            committed_loss_bytes = float(
                raw.get("committed_loss_bytes", -1.0)
            )
            boundary_commits = {}
            for item in raw.get("boundary_commits", []):
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError("corpus boundary commits are invalid")
                token_offset, byte_offset = map(int, item)
                if not 0 < token_offset <= len(ids):
                    raise ValueError("corpus boundary token is invalid")
                if not 0 < byte_offset <= source_bytes:
                    raise ValueError("corpus boundary byte is invalid")
                boundary_commits[token_offset] = byte_offset
            if (
                ids.ndim != 1
                or weights.shape != ids.shape
                or not bool(np.isfinite(weights).all())
                or not 0 <= offset <= len(ids)
                or source_bytes < 0
                or not 0 <= committed_bytes <= source_bytes
                or not np.isfinite(committed_loss_bytes)
                or committed_loss_bytes < 0.0
            ):
                raise ValueError("corpus pending segment is invalid")
            pending.append({
                "ids": ids,
                "loss_weights": weights,
                "offset": offset,
                "source_bytes": source_bytes,
                "committed_bytes": committed_bytes,
                "committed_loss_bytes": committed_loss_bytes,
                "boundary_commits": boundary_commits,
            })
            pending_tokens += len(ids) - offset
        if pending_tokens != int(state.get("pending_tokens", -1)):
            raise ValueError("corpus pending token count differs")
        protocol_mode = str(state.get("protocol_mode", ""))
        if protocol_mode not in {"plain", "state", "user", "assistant", "between"}:
            raise ValueError("corpus protocol mode is invalid")
        self.start_byte = start_byte
        self.limit = limit
        self.bytes_consumed = bytes_consumed
        self.tokens_consumed = tokens_consumed
        self._pending = pending
        self._pending_tokens = pending_tokens
        self._protocol_mode = protocol_mode
        self.file.seek(file_position)

    def _prime_protocol_mode(self) -> None:
        position = self.file.tell()
        lookback = min(position, 64 * 1024)
        self.file.seek(position - lookback)
        prefix = self.file.read(lookback).decode("utf-8", errors="replace")
        self.file.seek(position)
        ids = self.tokenizer.encode(prefix)
        if len(ids):
            self._loss_weights(ids)

    def _loss_weights(self, ids: np.ndarray) -> np.ndarray:
        """mask synthetic protocol while retaining recorded language."""

        controls = self.tokenizer.control_ids
        control_ids = set(controls.values())
        weights = np.ones(len(ids), dtype=np.float32)
        mode = self._protocol_mode
        for index, token_id in enumerate(ids):
            token_id = int(token_id)
            if token_id == controls["soma_state"]:
                weights[index] = 0.0
                mode = "state"
                continue
            if mode == "state":
                weights[index] = 0.0
                if token_id == controls["turn"]:
                    mode = "assistant"
                continue
            if token_id in control_ids:
                weights[index] = 0.0
                if token_id == controls["user"]:
                    mode = "user"
                elif token_id in (controls["soma"], controls["dream"]):
                    mode = "assistant"
                elif token_id == controls["turn"]:
                    if mode == "assistant":
                        weights[index] = self.turn_loss_weight
                    mode = "between"
        self._protocol_mode = mode
        return weights

    def _align_forward(self) -> None:
        while self.file.tell() < self.file_size:
            byte = self.file.read(1)
            if not byte or byte in b" \t\r\n":
                return

    @property
    def position(self) -> int:
        return self.start_byte + self.bytes_consumed

    @property
    def total_bytes(self) -> int:
        return max(0, self.limit - self.start_byte)

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "TokenCorpus":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _read_segment(self, target_bytes: int) -> bytes:
        remaining = self.limit - self.file.tell()
        if remaining <= 0:
            return b""
        take = min(max(256, int(target_bytes)), remaining)
        raw = self.file.read(take)
        if self.file.tell() < self.limit:
            extension = bytearray()
            while len(extension) < 1024 and self.file.tell() < self.limit:
                byte = self.file.read(1)
                if not byte:
                    break
                extension.extend(byte)
                if byte in b" \t\r\n":
                    break
            raw += bytes(extension)
        return raw

    def _pending_prefix(self, limit: int) -> np.ndarray:
        pieces = []
        remaining = max(0, int(limit))
        for segment in self._pending:
            if remaining <= 0:
                break
            ids = segment["ids"]
            offset = int(segment["offset"])
            take = min(remaining, len(ids) - offset)
            if take:
                pieces.append(ids[offset : offset + take])
                remaining -= take
        if not pieces:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(pieces)

    def next_block(self, minimum_tokens: int, *, stop_at_newline: bool = False):
        minimum_tokens = max(1, int(minimum_tokens))
        target_bytes = max(1024, minimum_tokens * 4)
        while (
            self._pending_tokens < minimum_tokens
            and self.file.tell() < self.limit
        ):
            raw = self._read_segment(target_bytes)
            if not raw:
                break
            encoded = self.tokenizer.encode(
                raw.decode("utf-8", errors="replace")
            )
            if len(encoded):
                loss_weights = self._loss_weights(encoded)
                boundary_commits = {}
                if self._newline_id is not None:
                    token_boundaries = (
                        np.flatnonzero(encoded == self._newline_id) + 1
                    ).tolist()
                    byte_boundaries = [
                        index + 1
                        for index, value in enumerate(raw)
                        if value == 0x0A
                    ]
                    boundary_commits = {
                        int(token_offset): int(byte_offset)
                        for token_offset, byte_offset in zip(
                            token_boundaries, byte_boundaries
                        )
                    }
                self._pending.append(
                    {
                        "ids": encoded,
                        "loss_weights": loss_weights,
                        "offset": 0,
                        "source_bytes": len(raw),
                        "committed_bytes": 0,
                        "committed_loss_bytes": 0.0,
                        "boundary_commits": boundary_commits,
                    }
                )
                self._pending_tokens += len(encoded)
        if not self._pending_tokens:
            return None

        requested = min(minimum_tokens, self._pending_tokens)
        if stop_at_newline and self._newline_id is not None:
            prefix = self._pending_prefix(requested)
            hits = np.flatnonzero(prefix == self._newline_id)
            if hits.size:
                requested = int(hits[0]) + 1
                # Preserve paragraph boundaries when consecutive newline
                # tokens are already available in the source buffer.
                extended = self._pending_prefix(
                    min(self._pending_tokens, requested + 8)
                )
                while (
                    requested < len(extended)
                    and int(extended[requested]) == self._newline_id
                ):
                    requested += 1
        pieces = []
        loss_pieces = []
        source_bytes = 0
        loss_bytes = 0.0
        remaining = requested
        while remaining > 0:
            segment = self._pending[0]
            ids = segment["ids"]
            loss_weights = segment["loss_weights"]
            old_offset = segment["offset"]
            take = min(remaining, len(ids) - old_offset)
            new_offset = old_offset + take
            pieces.append(ids[old_offset:new_offset])
            loss_pieces.append(loss_weights[old_offset:new_offset])

            target_committed = segment.get("boundary_commits", {}).get(
                new_offset,
                round(segment["source_bytes"] * new_offset / len(ids)),
            )
            source_bytes += target_committed - segment["committed_bytes"]
            target_loss_bytes = (
                segment["source_bytes"]
                * float(loss_weights[:new_offset].sum())
                / len(ids)
            )
            loss_bytes += (
                target_loss_bytes - segment["committed_loss_bytes"]
            )
            segment["committed_bytes"] = target_committed
            segment["committed_loss_bytes"] = target_loss_bytes
            segment["offset"] = new_offset
            remaining -= take
            self._pending_tokens -= take
            if new_offset == len(ids):
                source_bytes += (
                    segment["source_bytes"] - segment["committed_bytes"]
                )
                self._pending.pop(0)

        ids = np.concatenate(pieces)
        self.last_block_ended_at_newline = bool(
            self._newline_id is not None
            and len(ids)
            and int(ids[-1]) == self._newline_id
        )
        self.last_loss_weights = np.concatenate(loss_pieces)
        self.last_loss_bytes = float(loss_bytes)
        self.bytes_consumed += source_bytes
        self.tokens_consumed += requested
        return ids, source_bytes, self.position
