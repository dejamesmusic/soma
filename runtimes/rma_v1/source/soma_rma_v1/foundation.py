"""exact temporal state and bounded activations used by rma v1."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PHI = (1.0 + math.sqrt(5.0)) / 2.0
EPS = 1e-8


def aperture_schedule(n_bands):
    n_bands = max(1, int(n_bands))
    if n_bands == 1:
        return [1]
    levels = max(2, min(4, int(round(n_bands / 8.0))))
    first = max(1, min(n_bands - 1, int(round(0.75 * n_bands))))
    cutoffs = [first]
    for level in range(1, levels - 1):
        remaining_fraction = 0.25 / (2 ** level)
        cutoff = int(round(n_bands * (1.0 - remaining_fraction)))
        minimum = cutoffs[-1] + 1
        maximum = n_bands - (levels - 1 - level)
        cutoffs.append(max(minimum, min(maximum, cutoff)))
    cutoffs.append(n_bands)
    return cutoffs


class BudgetRelu(nn.Module):
    def __init__(self, budget):
        super().__init__()
        self.budget = float(budget)

    def forward(self, value):
        positive = F.relu(value)
        return positive * (
            self.budget / (positive.sum(dim=1, keepdim=True) + EPS)
        )


class ProjectedTokenBank:
    """exact identity traces projected through the current dictionary."""

    EXACT_CHUNK = 16_384

    def __init__(
        self, vocab_size, token_dim, n_bands, base, device,
        representation="cross_band",
    ):
        self.vocab_size = int(vocab_size)
        self.token_dim = int(token_dim)
        self.n_bands = int(n_bands)
        self.base = float(base)
        self.device = torch.device(device)
        if representation not in {"cross_band", "adjacent_difference"}:
            raise ValueError(f"unknown token representation: {representation}")
        self.representation = representation
        alpha = np.asarray(
            [self.base ** -index for index in range(self.n_bands)],
            dtype=np.float64,
        )
        if np.any(alpha <= 0.0) or np.any(1.0 - alpha == 1.0):
            raise ValueError("trace geometry exceeds float64 precision")
        self.alphas_cpu = torch.from_numpy(alpha)
        self.decay_cpu = 1.0 - self.alphas_cpu
        self.traces = torch.zeros(
            self.vocab_size, self.n_bands, dtype=torch.float64
        )
        self.alphas = self.alphas_cpu.float().to(self.device)
        self.decay = self.decay_cpu.float().to(self.device)
        # projected state for tap(), kept between token-by-token observations
        self._tap_cache = None
        self._build_kernels()

    @property
    def n_features(self):
        return self.token_dim * self.n_bands

    def _build_kernels(self):
        self._scan_tables = {}
        length = self.EXACT_CHUNK
        exponent = torch.arange(
            length - 1, -1, -1, dtype=torch.float64
        ).unsqueeze(1)
        self.exact_weights_cpu = self.alphas_cpu.unsqueeze(0) * torch.pow(
            self.decay_cpu.unsqueeze(0), exponent
        )
        self.exact_step_cpu = torch.pow(self.decay_cpu, length)

    def reset(self):
        self.traces.zero_()

    def _projected_state(self, embedding):
        return self.traces.T.float().to(self.device) @ embedding.weight

    def projected_state(self, embedding):
        return self._projected_state(embedding)

    @staticmethod
    def _cross_band(states):
        return torch.cat((states[:-1] - states[1:], states[-1:]), dim=0)

    def _features(self, states):
        if self.representation == "cross_band":
            return self._cross_band(states)
        # The fastest state has injection one: it is the exact last token
        # available to the causal predictor.  Every remaining coordinate is
        # the literal leaky integral of adjacent one-hot differences.  For
        # injection a and retention r, d = (a/r) * (x - z).
        scale = self.alphas[1:] / self.decay[1:]
        shape = (len(scale),) + (1,) * (states.ndim - 1)
        return torch.cat((
            states[0:1],
            (states[0:1] - states[1:]) * scale.reshape(shape),
        ), dim=0)

    # Token-by-token observation (generation) advances a cached projection
    # instead of re-projecting the whole bank; it is re-projected exactly from
    # the float64 traces this often, and whenever the traces or dictionary
    # change any other way (their version counters say so).
    TAP_REFRESH = 256

    def _tap_valid(self, weight):
        cache = self._tap_cache
        return (
            cache is not None and cache[0] is weight
            and cache[1] == weight._version and cache[2] == self.traces._version
        )

    def tap(self, embedding):
        weight = embedding.weight
        if not self._tap_valid(weight) or self._tap_cache[4] >= self.TAP_REFRESH:
            self._tap_cache = [
                weight, weight._version, self.traces._version,
                self._projected_state(embedding).detach(), 0,
            ]
        return self.tap_projected(self._tap_cache[3])

    def tap_projected(self, state):
        return self._features(state).T.reshape(-1)

    def tick(self, token_id):
        self.traces.mul_(self.decay_cpu.unsqueeze(0))
        if int(token_id) >= 0:
            self.traces[int(token_id)].add_(self.alphas_cpu)

    def _advance_exact(self, ids):
        ids_cpu = torch.from_numpy(np.asarray(ids, dtype=np.int64))
        cache = self._tap_cache
        if cache is not None and len(ids_cpu) == 1 and self._tap_valid(cache[0]):
            # one token: the projection advances by the same affine step
            token = int(ids_cpu[0])
            with torch.no_grad():
                cache[3].mul_(self.decay.unsqueeze(1))
                if token >= 0:
                    cache[3].addcmul_(self.alphas.unsqueeze(1), cache[0][token].detach().unsqueeze(0))
            self.traces.mul_(self.decay_cpu.unsqueeze(0))
            if token >= 0:
                self.traces[token].add_(self.alphas_cpu)
            cache[2] = self.traces._version
            cache[4] += 1
            return
        for start in range(0, len(ids_cpu), self.EXACT_CHUNK):
            chunk = ids_cpu[start:start + self.EXACT_CHUNK]
            count = len(chunk)
            if count == self.EXACT_CHUNK:
                step, weights = self.exact_step_cpu, self.exact_weights_cpu
            else:
                step = torch.pow(self.decay_cpu, count)
                exponent = torch.arange(
                    count - 1, -1, -1, dtype=torch.float64
                ).unsqueeze(1)
                weights = self.alphas_cpu.unsqueeze(0) * torch.pow(
                    self.decay_cpu.unsqueeze(0), exponent
                )
            self.traces.mul_(step.unsqueeze(0))
            valid = chunk >= 0
            if bool(valid.any()):
                self.traces.index_add_(0, chunk[valid], weights[valid])

    # Emitted states come from an exact affine scan over the prediction rows,
    # as in the v14.1 trace mixer: the tokens between two predictions
    # collapse into one transition, so a block costs a fixed number of
    # operations however many source tokens it spans.
    MAX_TRANSITIONS = 256

    def _stride_tables(self, stride):
        """float32 views of float64 decay geometry for one prediction stride."""
        tables = self._scan_tables.get(stride)
        if tables is not None:
            return tables
        size = self.MAX_TRANSITIONS
        alpha = self.alphas_cpu.unsqueeze(1)
        decay = self.decay_cpu.unsqueeze(1)
        # injection of the stride inputs into the state after them
        weights = alpha * torch.pow(decay, torch.arange(
            stride - 1, -1, -1, dtype=torch.float64).unsqueeze(0))
        # retention across 0..stride skipped tokens
        powers = torch.pow(decay, torch.arange(
            stride + 1, dtype=torch.float64).unsqueeze(0))
        # transition m reaches row n after n-1-m further strides
        lag = (
            torch.arange(size + 1, dtype=torch.float64).unsqueeze(1)
            - 1 - torch.arange(size, dtype=torch.float64).unsqueeze(0)
        )
        scan = torch.where(
            lag >= 0,
            torch.pow(decay.unsqueeze(2), (stride * lag.clamp_min(0)).unsqueeze(0)),
            torch.zeros((), dtype=torch.float64),
        )
        carry = torch.pow(decay, stride * torch.arange(
            size + 1, dtype=torch.float64).unsqueeze(0))
        tables = tuple(
            value.float().to(self.device)
            for value in (weights, powers, scan, carry)
        )
        if len(self._scan_tables) >= 8:
            self._scan_tables.pop(next(iter(self._scan_tables)))
        self._scan_tables[stride] = tables
        return tables

    def _scan_states(self, values, state, first, stride, count):
        """states before rows first + stride * m, m < count, as [band, row, dim].

        `state` is the projected state before token 0 of `values`.
        """
        groups = []
        position = 0
        while count > 0:
            take = min(count, self.MAX_TRANSITIONS + 1)
            weights, powers, scan, carry = self._stride_tables(stride)
            offset = first - position
            if offset:
                if offset <= stride:
                    prefix = weights[:, stride - offset:]
                    retained = powers[:, offset]
                else:
                    exponent = torch.arange(
                        offset - 1, -1, -1, dtype=torch.float64
                    ).unsqueeze(0)
                    prefix = (self.alphas_cpu.unsqueeze(1) * torch.pow(
                        self.decay_cpu.unsqueeze(1), exponent
                    )).float().to(self.device)
                    retained = torch.pow(
                        self.decay_cpu, offset
                    ).float().to(self.device)
                state = (
                    retained.unsqueeze(1) * state
                    + prefix @ values[position:first]
                )
            if take == 1:
                group = state.unsqueeze(1)
            else:
                transitions = take - 1
                inputs = values[first:first + transitions * stride].reshape(
                    transitions, stride, -1
                )
                injected = torch.einsum("ks,msd->kmd", weights, inputs)
                group = (
                    torch.matmul(scan[:, :take, :transitions], injected)
                    + carry[:, :take].unsqueeze(2) * state.unsqueeze(1)
                )
            groups.append(group)
            position = first + stride * (take - 1)
            state = group[:, -1]
            first = position + stride
            count -= take
        return groups[0] if len(groups) == 1 else torch.cat(groups, dim=1)

    def process_block_select(self, ids, embedding, rows=None):
        ids = np.asarray(ids, dtype=np.int64)
        rows = (
            np.arange(len(ids), dtype=np.int64)
            if rows is None else np.asarray(rows, dtype=np.int64)
        )
        if not len(rows):
            self._advance_exact(ids)
            return torch.empty(
                0, self.n_features,
                dtype=embedding.weight.dtype, device=self.device,
            )
        if rows[0] < 0 or rows[-1] >= len(ids) or np.any(np.diff(rows) <= 0):
            raise ValueError("prediction rows must be increasing and inside the block")
        # Only tokens before the last prediction reach an emitted state.
        ids_device = torch.from_numpy(ids[:rows[-1]]).to(self.device)
        values = embedding(ids_device.clamp_min(0)) * (
            ids_device >= 0
        ).unsqueeze(1)
        state = self._projected_state(embedding)
        gaps = np.diff(rows)
        if not len(gaps) or np.all(gaps == gaps[0]):
            stride = int(gaps[0]) if len(gaps) else 1
            states = self._scan_states(
                values, state, int(rows[0]), stride, len(rows)
            )
        else:
            dense = self._scan_states(
                values, state, int(rows[0]), 1, int(rows[-1] - rows[0]) + 1
            )
            states = dense.index_select(
                1, torch.from_numpy(rows - rows[0]).to(self.device)
            )
        self._advance_exact(ids)
        return self._features(states).permute(1, 2, 0).reshape(len(rows), -1)


class TranslatedErrorBank:
    """geometric traces of signed categorical residual."""

    def __init__(self, n_bands, dimension, base):
        self.n_bands = int(n_bands)
        self.dimension = int(dimension)
        self.base = float(base)
        self.alpha = torch.tensor(
            [self.base ** -index for index in range(self.n_bands)],
            dtype=torch.float64,
        )
        self.decay = 1.0 - self.alpha
        self.trace = torch.zeros(self.n_bands, self.dimension, dtype=torch.float64)

    def reset(self):
        self.trace.zero_()

    def advance(self, values, gaps=1):
        values = values.detach().float().cpu().double()
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if np.isscalar(gaps):
            if not len(values):
                return
            retention = torch.pow(self.decay, max(1, int(gaps)))
            exponent = torch.arange(
                len(values) - 1, -1, -1, dtype=torch.float64
            ).unsqueeze(1)
            weights = (1.0 - retention).unsqueeze(0) * torch.pow(
                retention.unsqueeze(0), exponent
            )
            self.trace.mul_(torch.pow(retention, len(values)).unsqueeze(1)).add_((values.T @ weights).T)
            return
        for value, gap in zip(values, gaps):
            retention = torch.pow(self.decay, max(1, int(gap)))
            self.trace.mul_(retention.unsqueeze(1)).add_((1.0 - retention).unsqueeze(1) * value.unsqueeze(0))

    def bandpass(self):
        output = torch.empty_like(self.trace)
        output[:-1] = self.trace[:-1] - self.trace[1:]
        output[-1] = self.trace[-1]
        return output
