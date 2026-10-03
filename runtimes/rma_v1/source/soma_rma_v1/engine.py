#!/usr/bin/env python3
"""validated residual metabolic aperture learner."""

from __future__ import annotations

import argparse
import json
import math
import warnings
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .foundation import (
    BudgetRelu, ProjectedTokenBank, TranslatedErrorBank, aperture_schedule,
)
from .metal_kernels import (
    ResidualSpectrumState, matrix_sphere_update_, sphere_statistics,
)
from .protocol import TokenCorpus
from .tokenizer import ReferenceTokenizer

PHI = (1.0 + math.sqrt(5.0)) / 2.0
EPS = 1e-8
# Recent residual-surprise observations kept for summaries; nothing grows.
RESIDUAL_HISTORY_LENGTH = 8192


class CorticalContext:
    """what the cortex path may read about the recent past.

    `workspace` is the cortex's exact-window input, a [rows, W] id grid;
    `recent` is a [rows, R + 1] grid of the last token ids, lag one first,
    for the sentinel pointer.
    """

    def __init__(self, workspace=None, recent=None, recent_unique=None,
                 recent_inverse=None):
        self.workspace = workspace
        self.recent = recent
        # distinct ids in `recent` and, for every entry, its index among them:
        # the pointer projects keys only for tokens actually in the window
        self.recent_unique = recent_unique
        self.recent_inverse = recent_inverse


class SentinelPointer(nn.Module):
    """pointer-sentinel mixture over the last R exact tokens.

    Each recent occurrence j is scored against the present state by cosine,
    s_j = tau * cos(q(state), k(token_j, token_before_j)) + b[lag_j], beside a
    sentinel score tau * cos(q, k_sentinel) + offset.  One softmax over the R
    occurrences and the sentinel gives the pointer's weight on each recent
    token and, on the sentinel, the share kept by the model's own prediction:
    p = keep * p_model + sum of pointer weights on each token.  A mixture can
    only move probability, so no token loses more than -log(keep).

    Cosine scores are bounded by tau however far an update rotates the
    matrices, so one step cannot saturate the softmax.  With random initial
    directions the offset holds the first pointer share near 0.2%.  Keys come
    from the detached dictionary, projected once per vocabulary entry and
    gathered by id.  `recent_ids` carries R + 1 lags; the last is only a
    previous-token key.
    """

    def __init__(self, window, token_width, key_width=64, temperature=16.0,
                 sentinel_offset=14.0):
        super().__init__()
        self.window = int(window)
        self.key_width = int(key_width)
        self.temperature = float(temperature)
        self.sentinel_offset = float(sentinel_offset)
        width = int(token_width)
        self.query = nn.Linear(width, self.key_width, bias=False)
        self.key = nn.Linear(width, self.key_width, bias=False)
        self.key_previous = nn.Linear(width, self.key_width, bias=False)
        for layer in (self.query, self.key, self.key_previous):
            nn.init.xavier_uniform_(layer.weight)
        self.sentinel_key = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(1, self.key_width))
        )
        lags = torch.arange(1, self.window + 1, dtype=torch.float32)
        self.lag_bias = nn.Parameter((-torch.log(lags) / torch.log(lags).norm()).unsqueeze(0))
        # Mean probability the pointer took in its last call; a device
        # tensor, read only when a report is made.
        self.last_share = torch.tensor(0.0)

    def forward(self, recent_ids, state, dictionary, norm, unique=None, inverse=None):
        """return pointer weights [rows, R] and the kept share [rows].

        With `unique` and `inverse`, keys are projected only for the distinct
        tokens in the window instead of the whole vocabulary; row by row the
        arithmetic is identical.
        """
        current, previous = recent_ids[:, :self.window], recent_ids[:, 1:self.window + 1]
        if unique is None:
            coordinates = norm(dictionary.weight.detach())
            current_index, previous_index = current.clamp_min(0), previous.clamp_min(0)
        else:
            coordinates = norm(dictionary.weight.detach()[unique.clamp_min(0)])
            current_index = inverse[:, :self.window]
            previous_index = inverse[:, 1:self.window + 1]
        keys = self.key(coordinates)
        previous_keys = self.key_previous(coordinates)
        key = keys[current_index] + previous_keys[previous_index] * (
            previous >= 0
        ).unsqueeze(-1)
        query = F.normalize(self.query(norm(state)), dim=1)
        scores = self.temperature * torch.einsum(
            "rk,rlk->rl", query, F.normalize(key, dim=2)
        ) + self.lag_bias
        scores = scores.masked_fill(current < 0, float("-inf"))
        sentinel = self.temperature * (
            query @ F.normalize(self.sentinel_key, dim=1).T
        ) + self.sentinel_offset
        weights = torch.softmax(torch.cat((scores, sentinel), dim=1), dim=1)
        self.last_share = 1.0 - weights[:, self.window].detach().mean()
        return weights[:, :self.window], weights[:, self.window]

    @staticmethod
    def with_pointer(model_log, weights, keep, recent_ids, targets):
        """log p(target) of the mixture from the model's log p(target)."""
        current = recent_ids[:, :weights.shape[1]]
        pointer = (weights * (current == targets[:, None])).sum(1)
        return torch.logaddexp(
            torch.log(keep.clamp_min(1e-30)) + model_log,
            torch.log(pointer.clamp_min(1e-30)),
        )

    @staticmethod
    def target_log_probability(logits, weights, keep, recent_ids, targets):
        """log p(target) under keep * softmax(logits) + pointer, with no
        vocabulary-sized mixture: one logsumexp, the target's logit and the
        pointer weights on occurrences of the target.  Exact, with exact
        gradients.  Also returns the logsumexp for reuse."""
        normaliser = torch.logsumexp(logits, dim=1)
        model_log = logits.gather(1, targets[:, None])[:, 0] - normaliser
        current = recent_ids[:, :weights.shape[1]]
        pointer = (weights * (current == targets[:, None])).sum(1)
        return torch.logaddexp(
            torch.log(keep.clamp_min(1e-30)) + model_log,
            torch.log(pointer.clamp_min(1e-30)),
        ), normaliser

    @staticmethod
    @torch.no_grad()
    def target_rank(logits, normaliser, weights, keep, recent_ids, target_log,
                    targets, unique=None, inverse=None):
        """exact rank of the target in keep * softmax(logits) + pointer.

        Tokens outside the window hold keep * softmax(logits) and beat the
        target when their logit clears one threshold: a vocabulary-wide
        comparison, no sort.  The window's distinct tokens are then corrected
        with their full mixture probability.
        """
        window = weights.shape[1]
        current = recent_ids[:, :window]
        if unique is None:
            unique, inverse = torch.unique(recent_ids, return_inverse=True)
        slot = inverse[:, :window]
        valid = current >= 0
        present = torch.zeros(
            current.shape[0], unique.shape[0], device=logits.device
        ).scatter_add_(1, slot, valid.to(logits.dtype)) > 0
        mass = torch.zeros_like(present, dtype=weights.dtype).scatter_add_(
            1, slot, weights * valid
        )
        unique_logits = logits[:, unique.clamp_min(0)]
        probability = keep[:, None] * torch.exp(unique_logits - normaliser[:, None]) + mass
        target_probability = target_log.exp()
        threshold = torch.log(target_probability / keep.clamp_min(1e-30)) + normaliser
        is_target = unique[None, :] == targets[:, None]
        target_present = (present & is_target).any(1)
        above = (logits > threshold[:, None]).sum(1)
        above = above - (present & (unique_logits > threshold[:, None])).sum(1)
        own = logits.gather(1, targets[:, None])[:, 0] > threshold
        above = above - (own & ~target_present).long()
        window_above = (
            present & ~is_target & (probability > target_probability[:, None])
        ).sum(1)
        return above + window_above

    @staticmethod
    def mix(logits, weights, keep, recent_ids):
        """log of keep * softmax(logits) + pointer weights gathered by token."""
        current = recent_ids[:, :weights.shape[1]]
        pointer = weights.new_zeros(logits.shape).scatter_add(
            1, current.clamp_min(0), weights * (current >= 0)
        )
        # Clamping keeps log(0) out of the backward pass; a clamped entry
        # contributes exp(-69) and no gradient.
        return torch.logaddexp(
            torch.log(keep.clamp_min(1e-30)).unsqueeze(1) + F.log_softmax(logits, dim=1),
            torch.log(pointer.clamp_min(1e-30)),
        )


class ReadoutTargets(torch.autograd.Function):
    """the cortex's two readout predictions at the target, with analytic
    gradients.

    candidate = base + shift and combined = base + gate * shift, with the base
    logits fixed.  Each loss reads log-softmax at the target alone, so the
    shift's gradient is a row scalar times (onehot - softmax(candidate)) and
    the gate's, from the combined prediction, is the row scalar
    shift_t - E_softmax(combined)[shift], taken in the forward: autograd never
    builds its vocabulary-sized intermediates.  The base keeps its own
    cross-entropy, bit for bit the base-only model's.

    Returns the two target log-probabilities, then the combined logits and
    their logsumexp without gradient (for ranking and the last row).
    """

    @staticmethod
    def forward(ctx, base, shift, gate, targets):
        column = targets[:, None]
        combined = torch.addcmul(base, gate, shift)
        candidate = base + shift
        combined_normaliser = torch.logsumexp(combined, 1)
        candidate_normaliser = torch.logsumexp(candidate, 1)
        shift_target = shift.gather(1, column)[:, 0]
        expected_shift = (
            torch.exp(combined - combined_normaliser[:, None]).mul_(shift).sum(1)
        )
        ctx.gate_shape = gate.shape
        ctx.save_for_backward(
            candidate, candidate_normaliser, shift_target - expected_shift, targets,
        )
        ctx.mark_non_differentiable(combined, combined_normaliser)
        return (
            combined.gather(1, column)[:, 0] - combined_normaliser,
            candidate.gather(1, column)[:, 0] - candidate_normaliser,
            combined, combined_normaliser,
        )

    @staticmethod
    def backward(ctx, combined_grad, candidate_grad, *_):
        candidate, normaliser, slope, targets = ctx.saved_tensors
        shift_grad = None
        if ctx.needs_input_grad[1]:
            # d/dshift of grad * log_softmax(candidate)[target]
            shift_grad = torch.exp(candidate - normaliser[:, None]).mul_(
                -candidate_grad[:, None]
            ).scatter_add_(1, targets[:, None], candidate_grad[:, None])
        return (
            None,
            shift_grad,
            (combined_grad * slope).reshape(-1, 1).sum_to_size(ctx.gate_shape)
            if ctx.needs_input_grad[2] else None,
            None,
        )


class GraftApertureNet(nn.Module):
    """v12.6 apertures exposed as comparable next-token opinions."""

    def __init__(self, vocab, token_dim, bands, *, untied=False,
                 detach_stages=False, mixer=True, language_depth=0,
                 language_expansion=2, language_residual_scale=0.1,
                 aperture_width=None, cortex_width=0, cortex_depth=0,
                 workspace_tokens=0, cortex_expansion=4, readout_window=0):
        super().__init__()
        self.vocab = int(vocab)
        self.token_dim = int(token_dim)
        self.aperture_width = int(aperture_width or token_dim)
        self.bands = int(bands)
        self.cutoffs = aperture_schedule(self.bands)
        self.detach_stages = bool(detach_stages)
        self.input_dictionary = nn.Embedding(self.vocab, self.token_dim)
        self.output_dictionary = (
            nn.Embedding(self.vocab, self.token_dim) if untied else None
        )
        self.layers = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.activations = nn.ModuleList()
        budget = 0.1 * self.aperture_width
        for index, cutoff in enumerate(self.cutoffs):
            prior = 0 if index == 0 else self.aperture_width
            self.layers.append(nn.Linear(
                cutoff * self.token_dim + prior,
                self.aperture_width,
                bias=False,
            ))
            self.heads.append(nn.Linear(
                self.aperture_width, self.token_dim, bias=False
            ))
            self.activations.append(BudgetRelu(budget))
        self.token_bias = nn.Parameter(torch.zeros(self.vocab))
        self.language_blocks = nn.ModuleList([
            LanguageResidualBlock(
                self.token_dim, language_expansion, language_residual_scale
            )
            for _ in range(max(0, int(language_depth)))
        ])
        cortex_spec = (
            int(cortex_width), int(cortex_depth), int(workspace_tokens),
            int(cortex_expansion),
        )
        self.cortex = None
        self.mixer_weights = (
            nn.Parameter(torch.ones(len(self.cutoffs))) if mixer else None
        )
        self.reset_parameters()
        if cortex_spec[0] > 0 and cortex_spec[1] > 0:
            self.cortex = CorticalMixer(
                self.token_dim, len(self.cutoffs), cortex_spec[0],
                cortex_spec[1], cortex_spec[2],
                expansion=cortex_spec[3],
            )
        self.recent_readout = (
            SentinelPointer(readout_window, self.token_dim)
            if self.cortex is not None and int(readout_window) > 0 else None
        )

    @property
    def output_weight(self):
        if self.output_dictionary is None:
            return self.input_dictionary.weight
        return self.output_dictionary.weight

    def reset_parameters(self):
        with torch.no_grad():
            dictionary_std = 0.1 * math.sqrt(256.0 / self.token_dim)
            self.input_dictionary.weight.normal_(std=dictionary_std)
            if self.output_dictionary is not None:
                self.output_dictionary.weight.copy_(
                    self.input_dictionary.weight
                )
            for layer in self.layers:
                layer.weight.normal_()
                target = math.sqrt(layer.weight.shape[1]) * 0.1
                layer.weight.mul_(target / (
                    layer.weight.norm(dim=1, keepdim=True) + EPS
                ))
            for head in self.heads:
                head.weight.normal_()
                # Keep contribution energy in the validated 256-wide units.
                target = math.sqrt(256.0) * 0.1
                head.weight.mul_(target / (
                    head.weight.norm(dim=1, keepdim=True) + EPS
                ))
            self.token_bias.zero_()
            if self.mixer_weights is not None:
                self.mixer_weights.fill_(1.0)

    def opinions(self, features, *, return_logits=True):
        rows = len(features)
        bands = features.reshape(rows, self.token_dim, self.bands)
        hidden = None
        languages, logits = [], []
        for cutoff, layer, activation, head in zip(
            self.cutoffs, self.layers, self.activations, self.heads
        ):
            raw = bands[:, :, :cutoff].reshape(rows, -1)
            if hidden is None:
                stage_input = raw
            else:
                previous = hidden.detach() if self.detach_stages else hidden
                stage_input = torch.cat((previous, raw), dim=1)
            hidden = activation(layer(stage_input))
            language = head(hidden)
            languages.append(language)
            if return_logits:
                logits.append(F.linear(
                    language, self.output_weight, self.token_bias
                ))
        return languages, logits

    def summed_language(self, languages):
        language = torch.stack(languages).sum(0)
        for block in self.language_blocks:
            language = block(language)
        return language

    def base_prediction(self, languages):
        return F.linear(
            self.summed_language(languages),
            self.output_weight,
            self.token_bias,
        )

    def cortical_terms(self, languages, history_ids=None, base_logits=None):
        """the readout's parts before they are combined: base logits, the
        cortex's logit shift and gate, and the pointer's weights.

        Returns (base_logits, shift, gate, weights, keep, context); weights and
        keep are None without a pointer.  combined = base.detach() + gate *
        shift.detach() and candidate = base.detach() + shift.
        """
        language, proposal, gate, weights, keep, context = self._cortical_inputs(
            languages, history_ids
        )
        # Logits are linear in the language, so both cortical predictions are
        # the base logits plus a projection of the proposal: one vocabulary
        # projection instead of two, with the same values and gradients.
        # The base's own learning never flows through either.
        if base_logits is None:
            base_logits = F.linear(language, self.output_weight, self.token_bias)
        shift = F.linear(proposal, self.output_weight.detach())
        return base_logits, shift, gate, weights, keep, context

    def _cortical_inputs(self, languages, history_ids=None):
        """language, cortical proposal and gate, and pointer weights."""
        context = (
            history_ids if isinstance(history_ids, CorticalContext)
            else CorticalContext(workspace=history_ids)
        )
        language = self.summed_language(languages)
        proposal, gate = self.cortex.proposal_and_gate(
            languages, language, context.workspace, self.input_dictionary,
        )
        weights = keep = None
        if self.recent_readout is not None and context.recent is not None:
            weights, keep = self.recent_readout(
                context.recent, language.detach(), self.input_dictionary,
                self.cortex.source_norm, context.recent_unique,
                context.recent_inverse,
            )
        return language, proposal, gate, weights, keep, context

    def cortical_parts(self, languages, history_ids=None, base_logits=None):
        """combined and candidate logits, and the pointer's weights, unmixed.

        Returns (combined, candidate, weights, keep, context).
        """
        base_logits, shift, gate, weights, keep, context = self.cortical_terms(
            languages, history_ids, base_logits
        )
        base_logits = base_logits.detach()
        combined = base_logits + gate * shift.detach()
        candidate = base_logits + shift
        return combined, candidate, weights, keep, context

    def cortical_predictions(self, languages, history_ids=None, base_logits=None):
        """full combined and candidate distributions (log-probabilities when a
        pointer mixes in); used where a whole distribution is needed."""
        combined, candidate, weights, keep, context = self.cortical_parts(
            languages, history_ids, base_logits
        )
        if weights is not None:
            combined = SentinelPointer.mix(combined, weights, keep, context.recent)
            candidate = SentinelPointer.mix(candidate, weights, keep, context.recent)
        return combined, candidate

    def cortical_prediction(self, languages, history_ids=None):
        """the emitted (combined) prediction alone.  Logits are linear in the
        language, so base + gate * shift is one vocabulary projection of
        language + gate * proposal; only the gate receives a gradient, as in
        the combined prediction of `cortical_parts`."""
        language, proposal, gate, weights, keep, context = self._cortical_inputs(
            languages, history_ids
        )
        combined = F.linear(
            language.detach() + gate * proposal.detach(),
            self.output_weight.detach(), self.token_bias.detach(),
        )
        if weights is not None:
            combined = SentinelPointer.mix(combined, weights, keep, context.recent)
        return combined

    def summed_prediction(self, languages, history_ids=None):
        if self.cortex is not None:
            return self.cortical_prediction(languages, history_ids)
        return self.base_prediction(languages)

    def mixed_prediction(self, opinions, *, detach=True):
        values = [value.detach() if detach else value for value in opinions]
        components = [
            values[index] - values[index + 1]
            for index in range(len(values) - 1)
        ] + [values[-1]]
        stacked = torch.stack(components, dim=1)
        return (stacked * self.mixer_weights.view(1, -1, 1)).sum(1)

    def parameter_groups(self):
        stages = [
            list(layer.parameters()) + list(head.parameters())
            for layer, head in zip(self.layers, self.heads)
        ]
        groups = {
            "input_dictionary": [self.input_dictionary.weight],
            "stages": stages,
            "bias": [self.token_bias],
        }
        if self.output_dictionary is not None:
            groups["output_dictionary"] = [self.output_dictionary.weight]
        if self.mixer_weights is not None:
            groups["mixer"] = [self.mixer_weights]
        if self.language_blocks:
            groups["language_blocks"] = [
                list(block.parameters()) for block in self.language_blocks
            ]
        if self.cortex is not None:
            groups["cortex"] = self.cortex.parameter_groups()
            if self.recent_readout is not None:
                groups["cortex"]["readout"] = list(self.recent_readout.parameters())
        return groups


class LanguageResidualBlock(nn.Module):
    """v12.9's post-aperture composition, kept outside temporal state."""

    def __init__(self, dimension, expansion=2, residual_scale=0.1):
        super().__init__()
        hidden = int(dimension) * max(1, int(expansion))
        self.residual_scale = float(residual_scale)
        self.norm = nn.LayerNorm(int(dimension))
        self.up = nn.Linear(int(dimension), hidden, bias=False)
        self.down = nn.Linear(hidden, int(dimension), bias=False)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.xavier_uniform_(self.down.weight)

    def forward(self, value):
        return value + self.residual_scale * self.down(
            F.gelu(self.up(self.norm(value)))
        )


class CorticalBlock(nn.Module):
    """wide present-time composition with no persistent optimizer state."""

    def __init__(self, width, expansion=4, residual_scale=0.1):
        super().__init__()
        width = int(width)
        hidden = width * max(1, int(expansion))
        self.residual_scale = float(residual_scale)
        self.norm = nn.RMSNorm(width, elementwise_affine=False)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.down.weight)

    def forward(self, value):
        normalized = self.norm(value)
        branch = self.down(
            F.silu(self.gate(normalized)) * self.value(normalized)
        )
        return value + self.residual_scale * branch


class CorticalMixer(nn.Module):
    """exact recent bands and nonlinear cross-aperture composition.

    The soma language remains an exact residual path.  A centred two-logit
    trust starts at zero, so enabling a cortex cannot perturb a new model's
    initial predictor before evidence identifies contexts in which its
    residual opinion helps.
    """

    def __init__(self, token_width, apertures, width, depth,
                 workspace_tokens, expansion=4):
        super().__init__()
        self.token_width = int(token_width)
        self.apertures = int(apertures)
        self.width = int(width)
        self.depth = int(depth)
        self.workspace_tokens = int(workspace_tokens)
        sources = self.apertures + self.workspace_tokens
        self.source_norm = nn.RMSNorm(
            self.token_width, elementwise_affine=False
        )
        self.stem = nn.Linear(
            sources * self.token_width, self.width, bias=False
        )
        self.blocks = nn.ModuleList([
            CorticalBlock(self.width, expansion=expansion)
            for _ in range(self.depth)
        ])
        self.output_positive = nn.Linear(
            self.width, self.token_width, bias=False
        )
        self.output_negative = nn.Linear(
            self.width, self.token_width, bias=False
        )
        self.trust = nn.Linear(self.width, 2, bias=False)
        self.last_gate_mean = torch.tensor(0.0)
        self.forced_gate = None
        nn.init.xavier_uniform_(self.stem.weight)
        nn.init.xavier_uniform_(self.output_positive.weight)
        with torch.no_grad():
            self.output_negative.weight.copy_(self.output_positive.weight)
        nn.init.xavier_uniform_(self.trust.weight[:1])
        with torch.no_grad():
            self.trust.weight[1].copy_(self.trust.weight[0])

    def forward(
        self, languages, soma_language, history_ids, dictionary,
        return_candidate=False,
    ):
        proposal, gate = self.proposal_and_gate(
            languages, soma_language, history_ids, dictionary
        )
        base = soma_language.detach()
        combined = base + gate * proposal.detach()
        candidate = base + proposal
        if return_candidate:
            return combined, candidate
        return combined

    def proposal_and_gate(self, languages, soma_language, history_ids, dictionary):
        """the bounded residual proposal and its admission gate.

        The combined language is base + gate * proposal (proposal detached);
        the candidate is base + proposal.  Callers that already hold the base
        logits can project the proposal alone instead of both languages.
        """
        rows = soma_language.shape[0]
        # The cortical learner observes the temporal machine without writing
        # through its feature paths.  Its residual can still improve the
        # shared prediction, while the aperture clocks retain their original
        # learning law and remain a meaningful ungrafted control.
        temporal_sources = torch.stack(languages, dim=1).detach()
        sources = []
        if self.workspace_tokens:
            if history_ids is None:
                history_ids = torch.full(
                    (rows, self.workspace_tokens), -1,
                    dtype=torch.long, device=soma_language.device,
                )
            if tuple(history_ids.shape) != (rows, self.workspace_tokens):
                raise ValueError("cortical history has the wrong shape")
            valid = history_ids >= 0
            safe = history_ids.clamp_min(0)
            history = dictionary(safe).detach()
            history = self.source_norm(history)
            history = history * valid.unsqueeze(-1)
            # Exact bands are ordered by lag: one tick ago first.  The same
            # tokens are also in the temporal traces; this is a second view.
            sources.append(history.flip(1))
        sources.append(self.source_norm(temporal_sources))
        value = torch.cat(sources, dim=1).reshape(rows, -1)
        value = F.silu(self.stem(value))
        for block in self.blocks:
            value = block(value)
        base = soma_language.detach()
        proposal = self._bounded(
            self.output_positive(value) - self.output_negative(value), base
        )
        # Trust observes a candidate; it must not reshape that candidate to
        # make its own decision easier.  Conversely the candidate learns a
        # useful full residual even while trust is still closed.
        trust_logits = self.trust(value.detach())
        # Admission lies in (0, 1): closing trust removes the proposal rather
        # than subtracting it.  For well-open trust tanh(softplus(x)) matches
        # the earlier tanh(x); the proposal starts at zero, so the gate's
        # initial 0.6 leaves a new model's prediction exact.
        gate = torch.tanh(
            F.softplus(trust_logits[:, 1:2] - trust_logits[:, 0:1])
        )
        if self.forced_gate is not None:
            gate = torch.full_like(gate, float(self.forced_gate))
        self.last_gate_mean = gate.detach().mean()
        return proposal, gate

    @staticmethod
    def _bounded(proposal, base):
        """shrink a proposal smoothly so it never outgrows the base language.

        The fixed-radius sphere bounds each matrix's energy but not its gain,
        so an unbounded proposal can grow until it swamps the base.  Below the
        base's size this is close to the identity; above, it saturates at it.
        """
        ratio = proposal.norm(dim=1, keepdim=True) / (
            base.norm(dim=1, keepdim=True) + EPS
        )
        return proposal * torch.rsqrt(1.0 + ratio.square())

    def parameter_groups(self):
        return {
            "stem": list(self.stem.parameters()),
            "blocks": [list(block.parameters()) for block in self.blocks],
            "output": (
                list(self.output_positive.parameters())
                + list(self.output_negative.parameters())
            ),
            "trust": list(self.trust.parameters()),
        }


def schedule_coefficients(name, cutoffs, slow_coefficient=None):
    count = len(cutoffs)
    horizon_slow = PHI ** (-(cutoffs[-1] - cutoffs[0]))
    slow = horizon_slow if slow_coefficient is None else float(slow_coefficient)
    if name == "uniform":
        return [1.0] * count
    if name == "phi":
        return [PHI ** (-index) for index in range(count)]
    if name == "reverse_phi":
        return list(reversed(schedule_coefficients("phi", cutoffs)))
    if name == "horizon":
        origin = cutoffs[0]
        return [PHI ** (-(cutoff - origin)) for cutoff in cutoffs]
    if name == "reverse_horizon":
        return list(reversed(schedule_coefficients("horizon", cutoffs)))
    if name == "linear_horizon":
        if count == 1:
            return [1.0]
        return [
            1.0 + index * (slow - 1.0) / (count - 1)
            for index in range(count)
        ]
    if name == "two_speed":
        return [1.0] + [slow] * (count - 1)
    if name == "all_slow":
        return [slow] * count
    raise ValueError(name)


def frobenius(tensor):
    """Frobenius norm as a sum of squares: an order of magnitude faster than
    torch.norm on MPS for the matrix sizes used here."""
    return tensor.square().sum().sqrt()


class ScheduledSGD:
    def __init__(self, model, base_lr, schedule, *, directional=False,
                 mixer_lr=0.01, match_update_norm=False,
                 slow_coefficient=None, dictionary_clock="fast",
                 parameter_geometry="free", bias_ceiling=None,
                 optimizer_kernel="dense"):
        self.model = model
        self.base_lr = float(base_lr)
        self.coefficients = schedule_coefficients(
            schedule, model.cutoffs, slow_coefficient
        )
        self.dictionary_clock = dictionary_clock
        self.directional = bool(directional)
        self.mixer_lr = float(mixer_lr)
        self.match_update_norm = bool(match_update_norm)
        self.parameter_geometry = str(parameter_geometry)
        self.optimizer_kernel = str(optimizer_kernel)
        self.matrix_radii = {
            id(parameter): float(parameter.detach().norm())
            for parameter in model.parameters() if parameter.ndim >= 2
        }
        self.bias_ceiling = float(
            bias_ceiling
            if bias_ceiling is not None
            else math.sqrt(model.vocab) * math.log(model.vocab)
        )
        self.steps = 0
        self.angular_movement = [0.0] * len(model.cutoffs)
        self.norm_match_sum = 0.0
        self.last_norm_match = 1.0
        self.max_norm_match = 1.0
        self.cortex_trust_coefficient = None
        self.cortex_gain = 1.0
        self.readout_max_angle = 0.01
        # Largest relative distance from its radius that any matrix reached
        # before the last update's retraction, measured from statistics the
        # update already gathers.
        self.last_radius_error = 0.0

    def _effective_gradient(self, parameter, gradient, directional):
        if self.parameter_geometry == "matrix_sphere" and parameter.ndim >= 2:
            direction = parameter / frobenius(parameter).clamp_min(EPS)
            return gradient - (gradient * direction).sum() * direction
        if directional and parameter.ndim >= 2:
            direction = F.normalize(parameter, dim=1)
            return gradient - (
                gradient * direction
            ).sum(1, keepdim=True) * direction
        return gradient

    def _update(self, parameters, rate, directional, gradient_cache=None,
                measure=False, max_angle=None):
        """apply one step; with measure, return the group's rotation angle.

        The angle is the root sum of each matrix's rotation in radians, so a
        stage's movement is comparable between the dense and fused paths.
        """
        movement = 0.0
        with torch.no_grad():
            for parameter in parameters:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient_cache is not None and id(parameter) in gradient_cache:
                    gradient = gradient_cache[id(parameter)]
                else:
                    gradient = self._effective_gradient(
                        parameter, gradient, directional
                    )
                if measure and parameter.ndim >= 2:
                    previous = parameter.detach().clone()
                if (
                    max_angle is not None
                    and self.parameter_geometry == "matrix_sphere"
                    and parameter.ndim >= 2
                ):
                    # Bound one update's rotation: a gradient that dwarfs the
                    # matrix radius would otherwise re-randomise it.
                    radius = self.matrix_radii[id(parameter)]
                    step = rate * float(frobenius(gradient))
                    limit = math.tan(max_angle) * radius
                    if step > limit:
                        gradient = gradient * (limit / step)
                parameter.add_(gradient, alpha=-rate)
                if (
                    self.parameter_geometry == "matrix_sphere"
                    and parameter.ndim >= 2
                ):
                    radius = self.matrix_radii[id(parameter)]
                    norm = frobenius(parameter)
                    self.last_radius_error = max(
                        self.last_radius_error,
                        abs(float(norm) - radius) / max(radius, EPS),
                    )
                    parameter.mul_(radius / norm.clamp_min(EPS))
                elif directional and parameter.ndim >= 2:
                    parameter.copy_(F.normalize(parameter, dim=1))
                elif (
                    self.parameter_geometry == "matrix_sphere"
                    and parameter.ndim < 2
                ):
                    # A common logit offset is an exact softmax gauge.  Remove
                    # it before bounding the remaining functional bias.
                    parameter.sub_(parameter.mean())
                    norm = parameter.norm()
                    if float(norm) > self.bias_ceiling:
                        parameter.mul_(self.bias_ceiling / norm)
                if measure and parameter.ndim >= 2:
                    # The chord between unit directions resolves small
                    # angles that an fp32 cosine rounds to zero.
                    chord = float((
                        previous / frobenius(previous).clamp_min(EPS)
                        - parameter / frobenius(parameter).clamp_min(EPS)
                    ).square().sum().sqrt())
                    movement += (2.0 * math.asin(min(1.0, 0.5 * chord))) ** 2
                parameter.grad = None
        return math.sqrt(movement)

    def _matched_alpha(self, scheduled):
        """match plasticity volume inside one independent clock domain."""
        if not self.match_update_norm:
            return 1.0, None
        gradient_cache = {}
        numerator = None
        denominator = None
        for parameters, coefficient in scheduled:
            for parameter in parameters:
                if parameter.grad is None:
                    continue
                gradient = self._effective_gradient(
                    parameter, parameter.grad, self.directional
                )
                gradient_cache[id(parameter)] = gradient
                square = gradient.detach().square().sum()
                numerator = square if numerator is None else numerator + square
                weighted = coefficient * coefficient * square
                denominator = (
                    weighted if denominator is None
                    else denominator + weighted
                )
        alpha = 1.0
        if denominator is not None:
            ratio = float((numerator / denominator.clamp_min(EPS)).sqrt())
            if math.isfinite(ratio):
                alpha = ratio
        return alpha, gradient_cache

    def _fused_cortical_step(self, cortex, gradient_scale=1.0):
        """the cortical update in fused form: one statistics transfer, then a
        single-pass tangent step and retraction per matrix.

        Numerically it is the per-matrix update: stem, blocks and output share
        their own norm match on phi-spaced depth rates; trust keeps its own
        rate; the pointer is bounded by its rotation cap.
        """
        blocks = cortex["blocks"]
        deepest = PHI ** (-(len(blocks) + 1))
        trust_coefficient = (
            deepest if self.cortex_trust_coefficient is None
            else self.cortex_trust_coefficient
        )
        matched = [(cortex["stem"], 1.0)] + [
            (parameters, PHI ** (-(index + 1)))
            for index, parameters in enumerate(blocks)
        ] + [(cortex["output"], deepest)]
        entries = [
            (parameter, "matched", coefficient)
            for parameters, coefficient in matched
            for parameter in parameters if parameter.grad is not None
        ]
        entries += [
            (parameter, "trust", trust_coefficient)
            for parameter in cortex["trust"] if parameter.grad is not None
        ]
        entries += [
            (parameter, "readout", 1.0)
            for parameter in cortex.get("readout") or []
            if parameter.grad is not None
        ]
        if not entries:
            return
        statistics, _ = sphere_statistics([entry[0] for entry in entries])
        tangents = []
        numerator = denominator = 0.0
        for (_, role, coefficient), (dot, gradient_square, norm_square) in zip(
            entries, statistics
        ):
            tangent_square = max(
                0.0, gradient_square - dot * dot / max(norm_square, EPS)
            )
            tangents.append(tangent_square)
            if role == "matched":
                numerator += tangent_square
                denominator += coefficient * coefficient * tangent_square
        alpha = math.sqrt(numerator / denominator) if denominator > 0.0 else 1.0
        # Statistics are of the unscaled gradients; the scale enters the rate.
        base_rate = self.base_lr * self.cortex_gain * gradient_scale
        limit = math.tan(self.readout_max_angle)
        for (parameter, role, coefficient), (dot, _, norm_square), tangent_square in zip(
            entries, statistics, tangents
        ):
            radius = self.matrix_radii[id(parameter)]
            rate = base_rate * coefficient * (alpha if role == "matched" else 1.0)
            if role == "readout" and rate * math.sqrt(tangent_square) > limit * radius:
                # A softmax gate needs fine steps: rotations of even ten
                # degrees move cosine scores by several units.
                rate = limit * radius / math.sqrt(tangent_square)
            self.last_radius_error = max(
                self.last_radius_error,
                abs(math.sqrt(norm_square) - radius) / max(radius, EPS),
            )
            matrix_sphere_update_(
                parameter, parameter.grad, rate=rate, radius=radius, dot=dot,
                tangent_square=tangent_square, norm_square=norm_square,
            )
            parameter.grad = None

    def _cortical_step(self, groups, gradient_scale=1.0):
        cortex = groups.get("cortex")
        if not cortex:
            return
        if self._cortex_fused():
            self._fused_cortical_step(cortex, gradient_scale)
            return
        # Early composition follows rapidly moving present-time features.
        # Deeper structure integrates those moving representations over
        # progressively longer phi-spaced evidence horizons.  Trust remains
        # fast because it estimates present comparative utility.
        scheduled = [(cortex["stem"], 1.0)]
        readout = cortex.get("readout") or []
        scheduled.extend(
            (parameters, PHI ** (-(index + 1)))
            for index, parameters in enumerate(cortex["blocks"])
        )
        scheduled.append((
            cortex["output"], PHI ** (-(len(cortex["blocks"]) + 1))
        ))
        alpha, gradient_cache = self._matched_alpha(scheduled)
        for parameters, coefficient in scheduled:
            self._update(
                parameters,
                self.base_lr * self.cortex_gain * alpha * coefficient,
                self.directional, gradient_cache,
            )
        if readout:
            # A softmax gate needs fine steps: rotations of even ten degrees
            # move cosine scores by several units and slam it open or shut.
            self._update(
                readout, self.base_lr * self.cortex_gain, self.directional,
                max_angle=self.readout_max_angle,
            )
        trust_coefficient = self.cortex_trust_coefficient
        if trust_coefficient is None:
            trust_coefficient = PHI ** (-(len(cortex["blocks"]) + 1))
        self._update(
            cortex["trust"],
            self.base_lr * self.cortex_gain * trust_coefficient,
            self.directional,
        )

    def step(self, gradient_scale=1.0):
        self.last_radius_error = 0.0
        gradient_scale = float(gradient_scale)
        groups = self.model.parameter_groups()
        # The fused paths apply the scale through their rates, which is exact
        # and saves a full pass over every gradient.  Anything else needs the
        # gradients scaled in place.
        fused = self._metal_ready(groups) and self._cortex_fused()
        if gradient_scale != 1.0 and not fused:
            with torch.no_grad():
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(gradient_scale)
            gradient_scale = 1.0
        fast = self.coefficients[0]
        slow = self.coefficients[-1]
        dictionary_coefficient = fast if self.dictionary_clock == "fast" else slow
        scheduled = [
            (groups["input_dictionary"], dictionary_coefficient),
            *list(zip(groups["stages"], self.coefficients)),
            (groups["bias"], slow),
        ]
        if "output_dictionary" in groups:
            scheduled.append((groups["output_dictionary"], slow))
        for parameters in groups.get("language_blocks", []):
            scheduled.append((parameters, slow))
        # Keep cortical plasticity out of the base norm match.  Adding
        # cortical capacity must not change the temporal learner's clock.
        self._cortical_step(groups, gradient_scale)
        if self._metal_sphere_step(scheduled, groups, gradient_scale):
            return
        alpha, gradient_cache = self._matched_alpha(scheduled)
        self.norm_match_sum += alpha
        self.last_norm_match = alpha
        self.max_norm_match = max(self.max_norm_match, alpha)
        self._update(
            groups["input_dictionary"],
            self.base_lr * alpha * dictionary_coefficient,
            self.directional,
            gradient_cache,
        )
        if "output_dictionary" in groups:
            self._update(
                groups["output_dictionary"], self.base_lr * alpha * slow,
                self.directional,
                gradient_cache,
            )
        for index, (coefficient, parameters) in enumerate(zip(
            self.coefficients, groups["stages"]
        )):
            self.angular_movement[index] += self._update(
                parameters, self.base_lr * alpha * coefficient,
                self.directional,
                gradient_cache,
                measure=True,
            )
        for parameters in groups.get("language_blocks", []):
            self._update(
                parameters, self.base_lr * alpha * slow, self.directional,
                gradient_cache,
            )
        self._update(
            groups["bias"], self.base_lr * alpha * slow, False,
            gradient_cache,
        )
        if "mixer" in groups:
            self._update(groups["mixer"], self.mixer_lr, False)
        self.model.zero_grad(set_to_none=True)
        self.steps += 1

    def _metal_ready(self, groups):
        return (
            self.optimizer_kernel == "auto"
            and self.parameter_geometry == "matrix_sphere"
            and self.match_update_norm
            and next(self.model.parameters()).device.type == "mps"
            and "mixer" not in groups
        )

    def _cortex_fused(self):
        return self.parameter_geometry == "matrix_sphere" and not self.directional

    def _metal_sphere_step(self, scheduled, groups, gradient_scale=1.0):
        if not self._metal_ready(groups):
            return False
        matrices = []
        vectors = []
        for parameters, coefficient in scheduled:
            for parameter in parameters:
                if parameter.grad is None:
                    continue
                if parameter.ndim >= 2:
                    matrices.append((parameter, float(coefficient)))
                else:
                    vectors.append((parameter, float(coefficient)))
        # Sufficient statistics come from one deterministic kernel pass per
        # matrix: fixed-order threadgroup trees, float64 on the host, no
        # atomics (an atomic reducer altered long continual trajectories).
        # The parameter's own norm is among them, so each step retracts from
        # the measured norm and rounding cannot accumulate into drift.  All
        # statistics are of the unscaled gradients; the scale enters the rate.
        vector_squares = (
            torch.stack([
                parameter.grad.detach().square().sum()
                for parameter, _ in vectors
            ]) if vectors else None
        )
        statistics, vector_host = sphere_statistics(
            [parameter for parameter, _ in matrices], extra=vector_squares
        )
        tangent_squares = []
        matrix_dots = []
        norm_squares = []
        numerator = 0.0
        denominator = 0.0
        for (parameter, coefficient), (dot, gradient_square, norm_square) in zip(
            matrices, statistics
        ):
            tangent_square = max(
                0.0, gradient_square - dot * dot / max(norm_square, EPS)
            )
            matrix_dots.append(dot)
            norm_squares.append(norm_square)
            tangent_squares.append(tangent_square)
            numerator += tangent_square
            denominator += coefficient * coefficient * tangent_square
        for (_, coefficient), square in zip(vectors, vector_host):
            numerator += square
            denominator += coefficient * coefficient * square
        alpha = (
            math.sqrt(numerator / denominator)
            if denominator > 0.0 else 1.0
        )
        self.norm_match_sum += alpha
        self.last_norm_match = alpha
        self.max_norm_match = max(self.max_norm_match, alpha)
        stage_of = {
            id(parameter): index
            for index, parameters in enumerate(groups["stages"])
            for parameter in parameters
        }
        stage_movement = [0.0] * len(self.angular_movement)
        for (parameter, coefficient), dot, tangent_square, norm_square in zip(
            matrices, matrix_dots, tangent_squares, norm_squares
        ):
            rate = self.base_lr * alpha * coefficient * gradient_scale
            radius = self.matrix_radii[id(parameter)]
            self.last_radius_error = max(
                self.last_radius_error,
                abs(math.sqrt(norm_square) - radius) / max(radius, EPS),
            )
            matrix_sphere_update_(
                parameter, parameter.grad,
                rate=rate,
                radius=radius,
                dot=dot,
                tangent_square=tangent_square,
                norm_square=norm_square,
            )
            if id(parameter) in stage_of:
                # The retracted tangent step rotates by atan(|step| / radius).
                stage_movement[stage_of[id(parameter)]] += math.atan2(
                    rate * math.sqrt(tangent_square), radius
                ) ** 2
            parameter.grad = None
        for index, movement in enumerate(stage_movement):
            self.angular_movement[index] += math.sqrt(movement)
        with torch.no_grad():
            for parameter, coefficient in vectors:
                parameter.add_(
                    parameter.grad,
                    alpha=-self.base_lr * alpha * coefficient * gradient_scale,
                )
                parameter.sub_(parameter.mean())
                norm = frobenius(parameter)
                if float(norm) > self.bias_ceiling:
                    parameter.mul_(self.bias_ceiling / norm)
                parameter.grad = None
        self.model.zero_grad(set_to_none=True)
        self.steps += 1
        return True


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


class Experiment:
    def __init__(self, args, condition, tokenizer=None):
        self.args = args
        self.condition = condition
        self.device = torch.device(args.device)
        # The runtime owns tokenizer selection; the model vocabulary must be
        # sized from the same tokenizer it will be fed.
        self.tokenizer = tokenizer or ReferenceTokenizer()
        local = condition.startswith("local")
        untied = condition.endswith("untied")
        self.model = GraftApertureNet(
            self.tokenizer.vocab_size, args.token_dim, args.bands,
            untied=untied, detach_stages=local, mixer=local,
            language_depth=args.language_depth,
            language_expansion=args.language_expansion,
            language_residual_scale=args.language_residual_scale,
            aperture_width=(args.aperture_width or args.token_dim),
            cortex_width=getattr(args, "cortex_width", 0),
            cortex_depth=getattr(args, "cortex_depth", 0),
            workspace_tokens=getattr(args, "workspace_tokens", 0),
            cortex_expansion=getattr(args, "cortex_expansion", 4),
            readout_window=getattr(args, "readout_window", 0),
        ).to(self.device)
        schedule = args.schedule
        if condition == "global_uniform" or condition == "local_uniform":
            schedule = "uniform"
        self.learner = ScheduledSGD(
            self.model, args.lr, schedule,
            directional=args.directional, mixer_lr=args.mixer_lr,
            match_update_norm=args.match_update_norm,
            slow_coefficient=args.slow_coefficient,
            dictionary_clock=args.dictionary_clock,
            parameter_geometry=getattr(args, "parameter_geometry", "free"),
            bias_ceiling=getattr(args, "bias_ceiling", None),
            optimizer_kernel=getattr(args, "optimizer_kernel", "dense"),
        )
        if self.model.cortex is not None:
            # The frontal rate is a public ratio to the base learning rate.
            self.learner.cortex_gain = float(getattr(args, "cortex_rate", 1.0))
        self.bank = ProjectedTokenBank(
            self.tokenizer.vocab_size, args.token_dim, args.bands,
            PHI, self.device,
            representation=getattr(
                args, "input_representation", "cross_band"
            ),
        )
        self.local = local
        self.bytes_seen = 0
        self.tokens_seen = 0
        error_dimension = (
            self.tokenizer.vocab_size
            if args.io2_error_space == "token" else args.token_dim
        )
        controller_bands = int(getattr(args, "controller_bands", args.bands))
        self.error_bank = TranslatedErrorBank(
            controller_bands, error_dimension, PHI
        )
        residual_kernel = getattr(args, "residual_kernel", "dense")
        self.fused_error_bank = None
        if (
            residual_kernel != "dense"
            and args.io2_error_space == "token"
            and self.device.type == "mps"
        ):
            self.fused_error_bank = ResidualSpectrumState(
                self.tokenizer.vocab_size,
                controller_bands,
                base=PHI,
                device=self.device,
                quantum=args.batch_residual_observation_quantum,
            )
        self.io2_ratio = 1.0
        self.io2_target_band = 0.0
        self.io2_decimation_band = (
            math.log(max(1.0, args.batch / args.batch_min), PHI)
            if args.batch_mode == "dynamic" else 0.0
        )
        self.io2_phase = 0.0
        self.io2_stride_fraction_phase = 0.0
        self.batch_loss_ema = None
        self.batch_fast_loss_ema = None
        self.batch_surprise = 0.0
        self.fast_utility = 0.0
        self.fast_utility_ema = None
        self.fast_utility_baseline = None
        self.batch_controller_tokens = 0
        self.residual_spectrum_baseline = None
        self.residual_surprise = 0.0
        self.residual_surprise_l2 = 0.0
        self.residual_surprise_topk = 0.0
        self.residual_surprise_sum = 0.0
        self.residual_surprise_max = 0.0
        self.residual_surprise_count = 0
        self.residual_surprise_history = deque(maxlen=RESIDUAL_HISTORY_LENGTH)
        self.residual_surprise_l2_history = deque(maxlen=RESIDUAL_HISTORY_LENGTH)
        self.residual_surprise_topk_history = deque(maxlen=RESIDUAL_HISTORY_LENGTH)
        self.batch_fraction_phase = 0.0
        self.io2_ratio_sum = 0.0
        self.io2_center_sum = 0.0
        self.io2_peak_sum = 0.0
        self.error_tokens_seen = 0
        self.io2_excess_center = 0.0
        self.io2_excess_strength = 0.0
        self.io2_whitened_spectrum = []
        self.io2_excess_spectrum = []
        self.io2_stride_sum = 0.0
        self.io2_observations = 0
        self.io2_updates = 0
        self.dynamic_batch_sum = 0
        self.dynamic_batch_count = 0
        self.dynamic_batch_min = None
        self.dynamic_batch_max = 0
        self._control_prediction = None
        self._control_loss = None
        self._control_token_nll = None
        self._last_token_nll = None
        # the target's exact rank and the last row's full distribution when
        # the loss is taken at the target only
        self._last_rank = None
        self._last_row = None
        # v14 io2 clock: gradients and controller evidence accumulate across
        # decimated source blocks until one fixed prediction quantum is full.
        self.optimizer_prediction_mass = 0.0
        self.optimizer_loss_mass = 0.0
        # source tokens spanned by the update being accumulated
        self.optimizer_source_tokens = 0.0
        self.controller_token_nll = 0.0
        self.controller_prediction_mass = 0.0
        self.controller_loss_token_mass = 0.0
        self.controller_loss_bytes = 0.0

    def _update_surprise_batch_controller(self, loss_value, observed_tokens):
        if self.batch_loss_ema is None:
            self.batch_loss_ema = loss_value
            self.batch_fast_loss_ema = loss_value
            self.batch_surprise = 0.0
        else:
            evidence = observed_tokens / max(1, self.args.batch_min)
            fast_alpha = 1.0 - (
                1.0 - self.args.batch_fast_loss_ema
            ) ** evidence
            self.batch_fast_loss_ema += fast_alpha * (
                loss_value - self.batch_fast_loss_ema
            )
            scale = max(abs(self.batch_loss_ema), EPS)
            self.batch_surprise = (
                self.batch_fast_loss_ema - self.batch_loss_ema
            ) / scale
            upward = self.args.batch_rise * evidence
            downward = self.args.batch_fall * max(
                0.0,
                self.batch_surprise - self.args.batch_surprise_threshold,
            )
            # A short tail is weak evidence.  It may scale an update down, but
            # it must not jerk the evidence horizon merely because its mean is
            # noisy.
            downward *= min(1.0, evidence)
            max_band = math.log(
                self.args.batch_max / self.args.batch_min, PHI
            )
            self.io2_decimation_band = max(0.0, min(
                max_band,
                self.io2_decimation_band + upward - downward,
            ))
            alpha = 1.0 - (1.0 - self.args.batch_loss_ema) ** evidence
            self.batch_loss_ema += alpha * (
                loss_value - self.batch_loss_ema
            )

    def _update_utility_batch_controller(self, observed_tokens):
        evidence = observed_tokens / max(1, self.args.batch_min)
        self.batch_controller_tokens += observed_tokens
        if self.fast_utility_ema is None:
            self.fast_utility_ema = self.fast_utility
            self.fast_utility_baseline = self.fast_utility
            return
        alpha = 1.0 - (1.0 - self.args.batch_utility_ema) ** evidence
        self.fast_utility_ema += alpha * (
            self.fast_utility - self.fast_utility_ema
        )
        calibrating = (
            self.batch_controller_tokens
            <= self.args.batch_controller_warmup
        )
        if calibrating:
            self.fast_utility_baseline = self.fast_utility_ema
            return
        if self.args.batch_utility_relative:
            scale = max(abs(self.fast_utility_baseline), 0.01)
            excess_utility = (
                self.fast_utility_ema - self.fast_utility_baseline
            ) / scale
        else:
            excess_utility = self.fast_utility_ema
        if (
            not self.args.batch_utility_relative
            and self.args.batch_utility_baseline_ema > 0.0
        ):
            excess_utility -= self.fast_utility_baseline
        upward = self.args.batch_rise * evidence
        downward = self.args.batch_fall * max(
            0.0,
            excess_utility - self.args.batch_utility_threshold,
        ) * min(1.0, evidence)
        max_band = math.log(
            self.args.batch_max / self.args.batch_min, PHI
        )
        self.io2_decimation_band = max(0.0, min(
            max_band,
            self.io2_decimation_band + upward - downward,
        ))
        if (
            self.args.batch_utility_baseline_ema > 0.0
            and excess_utility <= self.args.batch_utility_threshold
        ):
            baseline_alpha = 1.0 - (
                1.0 - self.args.batch_utility_baseline_ema
            ) ** evidence
            self.fast_utility_baseline += baseline_alpha * (
                self.fast_utility - self.fast_utility_baseline
            )

    def _update_residual_batch_controller(self, spectrum, observed_tokens):
        spectrum = spectrum.detach().double().cpu()
        self.batch_controller_tokens += observed_tokens
        if self.residual_spectrum_baseline is None:
            self.residual_spectrum_baseline = spectrum.clone()
            return
        evidence = observed_tokens / max(1, self.args.batch_min)
        horizon = max(1.0, self.args.batch_residual_baseline_horizon)
        alpha = 1.0 - math.exp(-observed_tokens / horizon)
        baseline = self.residual_spectrum_baseline
        floor = 0.1 * torch.median(baseline).clamp_min(EPS)
        innovation = torch.clamp(spectrum - baseline, min=0.0)
        self.residual_surprise_l2 = float(
            torch.linalg.vector_norm(innovation)
            / (torch.linalg.vector_norm(baseline + floor) + EPS)
        )
        relative_innovation = innovation / (baseline + floor)
        topk = min(3, relative_innovation.numel())
        self.residual_surprise_topk = float(
            torch.topk(relative_innovation, topk).values.mean()
        )
        self.residual_surprise = (
            self.residual_surprise_l2
            if self.args.batch_residual_reduction == "l2"
            else self.residual_surprise_topk
        )
        self.residual_surprise_history.append(self.residual_surprise)
        self.residual_surprise_l2_history.append(self.residual_surprise_l2)
        self.residual_surprise_topk_history.append(
            self.residual_surprise_topk
        )
        self.residual_surprise_sum += self.residual_surprise
        self.residual_surprise_max = max(
            self.residual_surprise_max, self.residual_surprise
        )
        self.residual_surprise_count += 1
        calibrating = (
            self.batch_controller_tokens
            <= self.args.batch_controller_warmup
        )
        if not calibrating:
            upward = self.args.batch_rise * evidence
            downward = (
                self.args.batch_fall
                * max(
                    0.0,
                    self.residual_surprise
                    - self.args.batch_residual_threshold,
                )
                * min(1.0, evidence)
            )
            max_band = math.log(
                self.args.batch_max / self.args.batch_min, PHI
            )
            self.io2_decimation_band = max(0.0, min(
                max_band,
                self.io2_decimation_band + upward - downward,
            ))
        self.residual_spectrum_baseline.add_(
            spectrum - self.residual_spectrum_baseline,
            alpha=alpha,
        )

    def _next_batch_size(self):
        if self.args.batch_mode == "fixed":
            size = self.args.batch
        else:
            target = self.args.batch_min * PHI ** self.io2_decimation_band
            target = max(self.args.batch_min, min(self.args.batch_max, target))
            lower = int(math.floor(target))
            self.batch_fraction_phase += target - lower
            size = lower
            if self.batch_fraction_phase + EPS >= 1.0:
                size += 1
                self.batch_fraction_phase -= 1.0
        size = max(1, int(size))
        return size

    def _record_batch_size(self, size):
        self.dynamic_batch_sum += size
        self.dynamic_batch_count += 1
        self.dynamic_batch_min = (
            size if self.dynamic_batch_min is None
            else min(self.dynamic_batch_min, size)
        )
        self.dynamic_batch_max = max(self.dynamic_batch_max, size)

    def _io2_stride(self):
        if self.args.io2_mode not in ("lens", "full"):
            return max(1, self.args.lens_stride)
        target = max(1.0, min(
            float(self.args.io2_stride_cap),
            PHI ** self.io2_decimation_band,
        ))
        lower = max(1, int(math.floor(target)))
        upper = min(self.args.io2_stride_cap, lower + 1)
        if lower == upper:
            return lower
        # Choose integer strides so the expected supervised density, rather
        # than the arithmetic mean stride, equals 1 / target.
        upper_probability = (
            (1.0 / lower - 1.0 / target)
            / (1.0 / lower - 1.0 / upper)
        )
        self.io2_stride_fraction_phase += upper_probability
        if self.io2_stride_fraction_phase + EPS >= 1.0:
            self.io2_stride_fraction_phase -= 1.0
            return upper
        return lower

    def io2_density(self):
        """supervised predictions per source token that io2 currently asks for."""
        if self.args.io2_mode not in ("lens", "full"):
            return 1.0 / max(1, self.args.lens_stride)
        return max(
            1.0 / self.args.io2_stride_cap,
            min(1.0, PHI ** (-self.io2_decimation_band)),
        )

    def _io2_update_due(self):
        if self.args.io2_mode != "full":
            return True
        density = self.io2_density()
        if self.io2_observations == 0:
            return True
        self.io2_phase += density
        if self.io2_phase + EPS >= 1.0:
            self.io2_phase = max(0.0, self.io2_phase - 1.0)
            return True
        return False

    def _spectral_frequency_band(self, max_band):
        """Invert the whitened excess frequency centroid into phi time."""
        spectral_excess = torch.as_tensor(
            self.io2_excess_spectrum, dtype=torch.float64
        ).clamp_min(0.0)
        excess_mass = float(spectral_excess.sum())
        if excess_mass <= EPS:
            return float(max_band)
        band_index = torch.arange(
            len(spectral_excess), dtype=torch.float64
        )
        mean_frequency = float((
            spectral_excess * (PHI ** (-band_index))
        ).sum() / excess_mass)
        return float(-math.log(max(mean_frequency, EPS), PHI))

    def _io2_controller_scale(self, observed_tokens):
        if (
            self.args.batch_mode == "fixed"
            or self.args.io2_controller_quantum <= 0
        ):
            return 1.0
        return observed_tokens / self.args.io2_controller_quantum

    def _prepare_learning_rate(self, observed_tokens):
        rate = self.args.lr
        if self.args.batch_learning_law == "source_normalized":
            reference = (
                self.args.batch_reference
                if self.args.batch_reference > 0 else self.args.batch
            )
            rate *= observed_tokens / max(1, reference)
        elif self.args.io2_mode in ("amplitude", "full"):
            if self.args.batch_lr_compensation == "linear":
                rate *= observed_tokens / self.args.batch
            elif self.args.batch_lr_compensation == "tail":
                rate *= min(1.0, observed_tokens / self.args.batch)
        if self.args.io2_mode in ("amplitude", "full"):
            rate *= self.io2_ratio
        self.learner.base_lr = rate

    def _advance_error_bank(self, values, stride):
        if self.args.io2_sampling == "hold" or stride == 1:
            self.error_bank.advance(values, gaps=stride)
            return None
        values = values.detach().float().cpu().double()
        retention = self.error_bank.decay
        injection = self.error_bank.alpha
        sampled_retention = torch.pow(retention, stride)
        gain = torch.sqrt(
            (1.0 - torch.pow(retention, 2 * stride))
            / (1.0 - retention.square() + EPS)
        )
        if len(values):
            # Closed form of len(values) sequential sampled updates:
            # trace <- r^n trace + sum_i (a g) r^(n-1-i) value_i.
            exponent = torch.arange(
                len(values) - 1, -1, -1, dtype=torch.float64
            ).unsqueeze(1)
            weights = (injection * gain).unsqueeze(0) * torch.pow(
                sampled_retention.unsqueeze(0), exponent
            )
            self.error_bank.trace.mul_(
                torch.pow(sampled_retention, len(values)).unsqueeze(1)
            ).add_(weights.T @ values)

        def covariance(left_r, right_r, left_i, right_i):
            return left_i * right_i / (
                1.0 - left_r * right_r + EPS
            )

        dense_variance = covariance(
            retention, retention, injection, injection
        )
        dense_cross = covariance(
            retention[:-1], retention[1:], injection[:-1], injection[1:]
        )
        dense_bandpass = (
            dense_variance[:-1] + dense_variance[1:] - 2.0 * dense_cross
        ).clamp_min(EPS)
        sampled_injection = injection * gain
        sampled_variance = covariance(
            sampled_retention, sampled_retention,
            sampled_injection, sampled_injection,
        )
        sampled_cross = covariance(
            sampled_retention[:-1], sampled_retention[1:],
            sampled_injection[:-1], sampled_injection[1:],
        )
        sampled_bandpass = (
            sampled_variance[:-1] + sampled_variance[1:]
            - 2.0 * sampled_cross
        ).clamp_min(EPS)
        return torch.sqrt(dense_bandpass / sampled_bandpass)

    def _measure_error_spectrum(self, sampling_calibration=None):
        if self.args.io2_spectrum == "proper":
            error_components = (
                self.error_bank.trace[:-1] - self.error_bank.trace[1:]
            )
        else:
            error_components = self.error_bank.bandpass()
        energy = torch.linalg.vector_norm(error_components, dim=1)
        if sampling_calibration is not None:
            if self.args.io2_spectrum == "proper":
                energy = energy * sampling_calibration
            else:
                energy[:-1] = energy[:-1] * sampling_calibration
        # Adjacent ema differences have strongly unequal white-noise gain.
        # Correct that response, including cold-start maturity, before asking
        # which timescale contains excess coherent residual energy.
        alpha = self.error_bank.alpha
        retention = self.error_bank.decay
        elapsed = max(1, self.error_tokens_seen)
        variance = alpha.square() * (
            1.0 - torch.pow(retention.square(), elapsed)
        ) / (1.0 - retention.square() + EPS)
        cross = alpha[:-1] * alpha[1:] * (
            1.0 - torch.pow(retention[:-1] * retention[1:], elapsed)
        ) / (1.0 - retention[:-1] * retention[1:] + EPS)
        white_band_variance = (
            variance[:-1] + variance[1:] - 2.0 * cross
        ).clamp_min(EPS)
        proper_energy = (
            energy if self.args.io2_spectrum == "proper" else energy[:-1]
        )
        whitened = proper_energy / torch.sqrt(white_band_variance)
        noise_floor = torch.median(whitened).clamp_min(EPS)
        excess = torch.clamp(whitened / noise_floor - 1.0, min=0.0)
        excess_total = excess.sum()
        excess_indices = torch.arange(len(excess), dtype=torch.float64)
        self.io2_excess_center = float(
            (excess_indices * excess).sum() / (excess_total + EPS)
        )
        self.io2_excess_strength = float(
            excess_total / max(1, len(excess))
        )
        self.io2_whitened_spectrum = whitened.tolist()
        self.io2_excess_spectrum = excess.tolist()
        total = energy.sum()
        concentration_energy = (
            energy[:-1]
            if self.args.io2_spectrum == "remainder" and len(energy) > 1
            else energy
        )
        probability = concentration_energy / (
            concentration_energy.sum() + EPS
        )
        entropy = -(probability * torch.log(probability + EPS)).sum()
        coherence = 1.0 - entropy / math.log(max(2, len(probability)))
        indices = torch.arange(len(energy), dtype=torch.float64)
        center = (indices * energy).sum() / (total + EPS)
        peak = int(torch.argmax(energy))
        return energy, whitened, total, coherence, center, peak

    def _measure_fused_error_energy(self, squared_energy):
        """apply the dense controller normalization to fused proper bands."""
        energy = torch.sqrt(
            squared_energy.detach().float().cpu().double().clamp_min(0.0)
        )
        alpha = self.error_bank.alpha
        retention = self.error_bank.decay
        elapsed = max(1, self.error_tokens_seen)
        variance = alpha.square() * (
            1.0 - torch.pow(retention.square(), elapsed)
        ) / (1.0 - retention.square() + EPS)
        cross = alpha[:-1] * alpha[1:] * (
            1.0 - torch.pow(retention[:-1] * retention[1:], elapsed)
        ) / (1.0 - retention[:-1] * retention[1:] + EPS)
        white_band_variance = (
            variance[:-1] + variance[1:] - 2.0 * cross
        ).clamp_min(EPS)
        whitened = energy / torch.sqrt(white_band_variance)
        noise_floor = torch.median(whitened).clamp_min(EPS)
        excess = torch.clamp(whitened / noise_floor - 1.0, min=0.0)
        excess_total = excess.sum()
        indices = torch.arange(len(excess), dtype=torch.float64)
        self.io2_excess_center = float(
            (indices * excess).sum() / (excess_total + EPS)
        )
        self.io2_excess_strength = float(
            excess_total / max(1, len(excess))
        )
        self.io2_whitened_spectrum = whitened.tolist()
        self.io2_excess_spectrum = excess.tolist()
        total = energy.sum()
        probability = energy / (total + EPS)
        entropy = -(probability * torch.log(probability + EPS)).sum()
        coherence = 1.0 - entropy / math.log(max(2, len(probability)))
        center = (indices * energy).sum() / (total + EPS)
        peak = int(torch.argmax(energy))
        return energy, whitened, total, coherence, center, peak

    @torch.no_grad()
    def _apply_io2(
        self, prediction, targets, weights, token_loss, stride, observed_tokens,
        *, control_due=True, nats_per_byte=None,
    ):
        if self.args.io2_mode == "off":
            return
        probabilities = F.softmax(prediction.detach().float(), dim=1)
        fused_residual = (
            self.fused_error_bank is not None
            and self.args.batch_mode == "dynamic"
            and self.args.batch_controller == "residual"
            and self.args.io2_error_space == "token"
            and stride == 1
        )
        residual_controller_updated = False
        if fused_residual:
            fused_energies = self.fused_error_bank.advance(
                probabilities, targets, weights
            )
            remaining_tokens = observed_tokens
            observation_quantum = self.args.batch_residual_observation_quantum
            for squared_energy in fused_energies:
                represented_tokens = min(
                    remaining_tokens, observation_quantum
                )
                remaining_tokens -= represented_tokens
                self.error_tokens_seen += represented_tokens
                (
                    energy, whitened, total, coherence, center, peak
                ) = self._measure_fused_error_energy(squared_energy)
                self._update_residual_batch_controller(
                    whitened, represented_tokens
                )
            residual_controller_updated = True
            translated = None
        elif self.args.io2_error_space == "token":
            translated = -probabilities
            translated[torch.arange(len(targets), device=targets.device), targets] += 1.0
            translated = translated * weights.unsqueeze(1)
        else:
            dictionary = self.model.output_weight.detach()
            translated = (
                dictionary.index_select(0, targets)
                - probabilities @ dictionary
            ) * weights.unsqueeze(1)
            dictionary_scale = dictionary.norm(dim=1).mean()
            translated = translated / (dictionary_scale + EPS)
        observation_quantum = self.args.batch_residual_observation_quantum
        if fused_residual:
            pass
        elif (
            self.args.batch_mode == "dynamic"
            and self.args.batch_controller == "residual"
            and observation_quantum > 0
        ):
            remaining_tokens = observed_tokens
            for start in range(0, len(translated), observation_quantum):
                values = translated[start:start + observation_quantum]
                sampling_calibration = self._advance_error_bank(values, stride)
                represented_tokens = min(
                    remaining_tokens, len(values) * max(1, stride)
                )
                remaining_tokens -= represented_tokens
                self.error_tokens_seen += represented_tokens
                (
                    energy, whitened, total, coherence, center, peak
                ) = self._measure_error_spectrum(sampling_calibration)
                self._update_residual_batch_controller(
                    whitened, represented_tokens
                )
            residual_controller_updated = True
        else:
            sampling_calibration = self._advance_error_bank(translated, stride)
            self.error_tokens_seen += observed_tokens
            (
                energy, whitened, total, coherence, center, peak
            ) = self._measure_error_spectrum(sampling_calibration)
        # Error state is updated for every admitted observation, but the
        # actuator changes only at complete optimizer-evidence boundaries.
        if not control_due:
            return
        error_energy = float(total / (total + math.sqrt(len(energy))))
        plasticity = 2.5 * math.sqrt(max(0.0, min(1.0, error_energy))) * (
            0.25 + 0.75 * math.sqrt(max(0.0, min(1.0, float(coherence))))
        )
        chance = math.log(self.model.vocab)
        loss_exponent = max(
            -1.0, min(1.0, (float(token_loss) - chance) / chance)
        )
        loss_scale = PHI ** loss_exponent
        plasticity *= loss_scale
        self.io2_ratio = max(0.02, min(1.0, plasticity))
        self.io2_target_band = float(center)
        if self.args.batch_mode == "dynamic" and self.args.batch_controller in ("surprise", "utility", "residual"):
            if self.args.batch_controller == "surprise":
                self._update_surprise_batch_controller(
                    float(token_loss), observed_tokens
                )
            elif self.args.batch_controller == "utility":
                self._update_utility_batch_controller(observed_tokens)
            elif not residual_controller_updated:
                self._update_residual_batch_controller(
                    whitened, observed_tokens
                )
        elif (
            self.args.io2_mode in ("lens", "full")
            or self.args.batch_mode == "dynamic"
        ):
            max_band = (
                math.log(self.args.batch_max / self.args.batch_min, PHI)
                if self.args.batch_mode == "dynamic"
                else math.log(self.args.io2_stride_cap, PHI)
            )
            if (
                self.args.batch_controller == "io2"
                and self.args.io2_batch_map == "spectral_frequency"
            ):
                # Traversal is a sampling decision.  Convert whitened excess
                # residual mass to its frequency centroid and then invert that
                # frequency back into a coherence-time band.  Plasticity is a
                # separate actuator and must not rescale the time coordinate.
                spectral_band = self._spectral_frequency_band(max_band)
                self.io2_target_band = float(spectral_band)
                target = max(0.0, min(
                    max_band,
                    self.args.io2_decimation_gain * spectral_band,
                ))
            elif (
                self.args.batch_controller == "io2"
                and self.args.io2_batch_map != "historical"
            ):
                if self.args.io2_batch_map == "absolute":
                    absolute_band = self.io2_target_band
                elif self.args.io2_batch_map == "calibrated":
                    absolute_band = -2.381217 + 1.228207 * self.io2_target_band
                else:
                    absolute_band = 1.6258 + 1.0124 * peak
                origin = (
                    math.log(self.args.batch_min, PHI)
                    if self.args.batch_mode == "dynamic" else 0.0
                )
                target = self.args.io2_decimation_gain * (
                    absolute_band - origin
                )
                target = max(0.0, min(max_band, target))
            else:
                native_band = min(
                    max_band,
                    math.log(max(1, self.args.io2_stride_cap), PHI),
                )
                spectral_band = min(native_band, self.io2_target_band)
                if nats_per_byte is None:
                    relaxation = 0.0
                else:
                    relaxation = max(
                        0.0,
                        min(1.0, 1.0 - float(nats_per_byte) / math.log(2.0)),
                    )
                throughput_target = (
                    (1.0 - relaxation) * spectral_band
                    + relaxation * native_band
                )
                raw_target = (1.0 - self.io2_ratio) * throughput_target
                target = max(0.0, min(
                    max_band, self.args.io2_decimation_gain * raw_target
                ))
            # In fixed-evidence mode this branch is entered only when one
            # complete optimizer quantum has accumulated.  Corpus fragments
            # may be much shorter than that quantum, so the size of the final
            # fragment cannot define actuator time.
            controller_scale = self._io2_controller_scale(observed_tokens)
            if target > self.io2_decimation_band:
                target = min(
                    target,
                    self.io2_decimation_band + controller_scale,
                )
            else:
                target = max(
                    target,
                    self.io2_decimation_band - 2.0 * controller_scale,
                )
            self.io2_decimation_band = target
        self.io2_ratio_sum += self.io2_ratio
        self.io2_center_sum += self.io2_target_band
        self.io2_peak_sum += peak
        self.io2_stride_sum += stride
        self.io2_observations += 1

    def prediction_and_loss(
        self, features, targets, weights, history_ids=None
    ):
        languages, opinions = self.model.opinions(
            features, return_logits=self.local
        )
        if self.local:
            local_losses = [
                (F.cross_entropy(logits, targets, reduction="none") * weights).sum()
                / weights.sum()
                for logits in opinions
            ]
            prediction = self.model.mixed_prediction(opinions, detach=True)
            self._control_prediction = prediction.detach()
            token_nll = F.cross_entropy(prediction, targets, reduction="none")
            self._last_token_nll = token_nll.detach()
            self._control_token_nll = None
            mixer_loss = (token_nll * weights).sum() / weights.sum()
            self._control_loss = mixer_loss.detach()
            # Every opinion remains a competent estimate. The detached mixer
            # observes those estimates without corrupting their source clocks.
            loss = torch.stack(local_losses).mean() + mixer_loss
            return prediction, loss, local_losses
        base_prediction = self.model.base_prediction(languages)
        # Annealing belongs to the temporal base.  Optional present-time
        # composition may improve the emitted prediction, but must not alter
        # the clocks or plasticity temperature of the state it reads.
        self._control_prediction = base_prediction.detach()
        self._last_rank = self._last_row = None
        if self.model.cortex is not None:
            base_logits, shift, gate, pointer, keep, context = (
                self.model.cortical_terms(
                    languages, history_ids, base_logits=base_prediction
                )
            )
            # Every loss is read at the target only: no vocabulary-sized
            # mixture or gradient is built beyond the logits themselves.
            combined_log, candidate_log, combined, normaliser = (
                ReadoutTargets.apply(base_logits.detach(), shift, gate, targets)
            )
            prediction = combined
            base_token_nll = F.cross_entropy(base_logits, targets, reduction="none")
            if pointer is not None:
                combined_log = SentinelPointer.with_pointer(
                    combined_log, pointer, keep, context.recent, targets
                )
                candidate_log = SentinelPointer.with_pointer(
                    candidate_log, pointer, keep, context.recent, targets
                )
                # the rank reported is soma's own readout's (with the pointer)
                self._last_rank = SentinelPointer.target_rank(
                    combined, normaliser, pointer.detach(), keep.detach(),
                    context.recent, combined_log.detach(), targets,
                    context.recent_unique, context.recent_inverse,
                )
                with torch.no_grad():
                    self._last_row = SentinelPointer.mix(
                        combined[-1:], pointer[-1:], keep[-1:], context.recent[-1:]
                    )[0]
            token_nll = -combined_log
            candidate_nll = -candidate_log
            predictive_loss = (token_nll * weights).sum() / weights.sum()
            candidate_loss = (candidate_nll * weights).sum() / weights.sum()
            base_loss = (base_token_nll * weights).sum() / weights.sum()
            self._control_loss = base_loss.detach()
            self._control_token_nll = base_token_nll.detach()
            # Numerically this is the runtime predictor's loss. Its gradient
            # is the disjoint union of base competence, full-strength
            # residual learning, and conditional trust.
            loss = (
                predictive_loss
                + base_loss - base_loss.detach()
                + candidate_loss - candidate_loss.detach()
            )
            component_losses = [
                base_loss, candidate_loss, predictive_loss,
            ]
        else:
            prediction = base_prediction
            token_nll = F.cross_entropy(prediction, targets, reduction="none")
            predictive_loss = (token_nll * weights).sum() / weights.sum()
            loss = predictive_loss
            self._control_loss = predictive_loss.detach()
            self._control_token_nll = None
            component_losses = [loss]
        # Callers reporting prequential loss reuse this instead of evaluating
        # the same vocabulary-wide cross-entropy a second time.
        self._last_token_nll = token_nll.detach()
        if self.args.batch_controller == "utility":
            self.fast_utility = self._fast_utility(
                languages, base_prediction, prediction, predictive_loss,
                targets, weights,
            )
        return prediction, loss, component_losses

    @torch.no_grad()
    def _fast_utility(
        self, languages, base_prediction, prediction, predictive_loss,
        targets, weights,
    ):
        """relative loss increase when the fastest aperture is ablated.

        Only the utility batch controller reads this.  It costs one extra
        vocabulary projection, one cross-entropy and a host synchronisation,
        so it is evaluated only when that controller is selected.
        """
        if self.model.language_blocks:
            slow_prediction = self.model.summed_prediction([
                language.detach() for language in languages[1:]
            ])
        else:
            # Each opinion includes the shared bias.  Subtracting the
            # first opinion and restoring that bias is the exact slow-only
            # ablation; it needs one projection of the fast opinion.
            fast_prediction = F.linear(
                languages[0].detach(),
                self.model.output_weight.detach(),
                self.model.token_bias.detach(),
            )
            cortical_residual = (
                prediction.detach() - base_prediction.detach()
            )
            slow_prediction = (
                base_prediction.detach() - fast_prediction
                + self.model.token_bias.detach()
                + cortical_residual
            )
        slow_loss = (
            F.cross_entropy(
                slow_prediction, targets, reduction="none"
            ) * weights
        ).sum() / weights.sum()
        return float(
            (slow_loss - predictive_loss.detach())
            / (predictive_loss.detach().abs() + EPS)
        )

    def train_file(self, path, byte_limit=0):
        # Stale: this loop predates the runtime's evidence-quantum updates. It
        # still scales the rate by tokens / batch_reference and steps every
        # block, which diverges (held-out 14 -> 1367 bpb, 2026-09-27). The
        # runtime (`runtime.SOMA.train`) is the training loop; use it.
        warnings.warn(
            "engine.Experiment.train_file is not the runtime's training loop and "
            "diverges; train through runtime.SOMA (see run_experiment.py)",
            RuntimeWarning, stacklevel=2,
        )
        reader = TokenCorpus(path, self.tokenizer, max_bytes=byte_limit)
        total_nll = 0.0
        total_loss_bytes = 0.0
        total_source_bytes = 0
        batch_sizes = []
        residual_history_start = self.residual_surprise_count
        started = time.perf_counter()
        try:
            while True:
                block = reader.next_block(self._next_batch_size())
                if block is None:
                    break
                ids, source_bytes, _ = block
                actual_batch = len(ids)
                self._record_batch_size(actual_batch)
                batch_sizes.append(actual_batch)
                self._prepare_learning_rate(actual_batch)
                stride = self._io2_stride()
                rows = np.arange(
                    0, len(ids), stride,
                    dtype=np.int64,
                )
                targets = torch.from_numpy(ids[rows]).to(self.device)
                weights = torch.from_numpy(
                    reader.last_loss_weights[rows]
                ).to(self.device)
                if float(weights.sum()) <= 0.0:
                    # Masked protocol/state events still advance temporal
                    # state.  They supply no error and therefore cannot
                    # justify an optimizer or residual-controller update.
                    with torch.no_grad():
                        self.bank.process_block_select(
                            ids, self.model.input_dictionary, rows=rows
                        )
                    total_loss_bytes += reader.last_loss_bytes
                    total_source_bytes += source_bytes
                    self.bytes_seen += source_bytes
                    self.tokens_seen += len(ids)
                    continue
                update_due = self._io2_update_due()
                with torch.set_grad_enabled(update_due):
                    features = self.bank.process_block_select(
                        ids, self.model.input_dictionary, rows=rows
                    )
                    prediction, loss, _ = self.prediction_and_loss(
                        features, targets, weights
                    )
                if update_due:
                    loss.backward()
                    self.learner.step()
                    self.io2_updates += 1
                self._apply_io2(
                    self._control_prediction, targets, weights,
                    self._control_loss, stride,
                    len(ids),
                )
                with torch.no_grad():
                    per_token = F.cross_entropy(
                        prediction, targets, reduction="none"
                    )
                    total_nll += float((per_token * weights).sum())
                total_loss_bytes += reader.last_loss_bytes
                total_source_bytes += source_bytes
                self.bytes_seen += source_bytes
                self.tokens_seen += len(ids)
        finally:
            reader.close()
        elapsed = time.perf_counter() - started
        # The histories are bounded; summarise the observations this call
        # made that they still hold.
        observed = min(
            self.residual_surprise_count - residual_history_start,
            len(self.residual_surprise_history),
        )
        residual_surprises = list(self.residual_surprise_history)[
            len(self.residual_surprise_history) - observed:
        ]
        residual_surprises_l2 = list(self.residual_surprise_l2_history)[
            len(self.residual_surprise_l2_history) - observed:
        ]
        residual_surprises_topk = list(self.residual_surprise_topk_history)[
            len(self.residual_surprise_topk_history) - observed:
        ]
        if residual_surprises:
            surprise_quantiles = {
                str(q): float(np.quantile(residual_surprises, q))
                for q in (0.5, 0.75, 0.9, 0.95, 0.99)
            }
        else:
            surprise_quantiles = {}
        return {
            "bpb_prequential": total_nll / max(total_loss_bytes, EPS) / math.log(2),
            "source_bytes": total_source_bytes,
            "tokens": self.tokens_seen,
            "seconds": elapsed,
            "source_bytes_per_second": total_source_bytes / max(elapsed, EPS),
            "updates": self.learner.steps,
            "batch_mean": sum(batch_sizes) / max(1, len(batch_sizes)),
            "batch_min": min(batch_sizes) if batch_sizes else None,
            "batch_max": max(batch_sizes) if batch_sizes else None,
            "batch_first": batch_sizes[:12],
            "batch_last": batch_sizes[-12:],
            "batch_sizes": batch_sizes,
            "residual_surprise_mean": (
                sum(residual_surprises) / max(1, len(residual_surprises))
            ),
            "residual_surprise_max": (
                max(residual_surprises) if residual_surprises else 0.0
            ),
            "residual_surprise_first": residual_surprises[:12],
            "residual_surprise_last": residual_surprises[-12:],
            "residual_surprise_quantiles": surprise_quantiles,
            "residual_surprises": residual_surprises,
            "residual_surprises_l2": residual_surprises_l2,
            "residual_surprises_topk": residual_surprises_topk,
        }

    @torch.no_grad()
    def evaluate(self, path):
        saved = self.bank.traces.clone()
        self.bank.reset()
        reader = TokenCorpus(
            path, self.tokenizer, max_bytes=self.args.eval_bytes
        )
        total_nll = 0.0
        total_loss_bytes = 0.0
        opinion_nll = [0.0] * len(self.model.cutoffs)
        try:
            while True:
                block = reader.next_block(self.args.batch)
                if block is None:
                    break
                ids, _, _ = block
                features = self.bank.process_block_select(
                    ids, self.model.input_dictionary
                )
                targets = torch.from_numpy(ids).to(self.device)
                weights = torch.from_numpy(reader.last_loss_weights).to(
                    self.device
                )
                languages, opinions = self.model.opinions(features)
                prediction = (
                    self.model.mixed_prediction(opinions, detach=False)
                    if self.local else self.model.summed_prediction(languages)
                )
                total_nll += float((F.cross_entropy(
                    prediction, targets, reduction="none"
                ) * weights).sum())
                for index, logits in enumerate(opinions):
                    opinion_nll[index] += float((F.cross_entropy(
                        logits, targets, reduction="none"
                    ) * weights).sum())
                total_loss_bytes += reader.last_loss_bytes
        finally:
            reader.close()
            self.bank.traces.copy_(saved)
        scale = max(total_loss_bytes, EPS) * math.log(2)
        return {
            "bpb": total_nll / scale,
            "opinion_bpb": [value / scale for value in opinion_nll],
        }

    def diagnostics(self):
        return {
            "cutoffs": self.model.cutoffs,
            "schedule": self.learner.coefficients,
            "parameter_geometry": self.learner.parameter_geometry,
            "optimizer_kernel": self.learner.optimizer_kernel,
            "residual_kernel": (
                "metal" if self.fused_error_bank is not None else "dense"
            ),
            "dictionary_clock": self.learner.dictionary_clock,
            "mixer_weights": (
                self.model.mixer_weights.detach().cpu().tolist()
                if self.model.mixer_weights is not None else None
            ),
            "stage_movement": self.learner.angular_movement,
            "mean_norm_match": (
                self.learner.norm_match_sum / max(1, self.learner.steps)
            ),
            "io2_mode": self.args.io2_mode,
            "io2_ratio": self.io2_ratio,
            "mean_io2_ratio": self.io2_ratio_sum / max(1, self.io2_observations),
            "io2_target_band": self.io2_target_band,
            "mean_io2_target_band": self.io2_center_sum / max(1, self.io2_observations),
            "mean_io2_peak_band": self.io2_peak_sum / max(1, self.io2_observations),
            "io2_excess_center": self.io2_excess_center,
            "io2_excess_strength": self.io2_excess_strength,
            "io2_whitened_spectrum": self.io2_whitened_spectrum,
            "io2_excess_spectrum": self.io2_excess_spectrum,
            "io2_decimation_band": self.io2_decimation_band,
            "mean_io2_stride": self.io2_stride_sum / max(1, self.io2_observations),
            "io2_update_duty": self.io2_updates / max(1, self.io2_observations),
            "batch_mode": self.args.batch_mode,
            "batch_lr_compensation": self.args.batch_lr_compensation,
            "batch_learning_law": self.args.batch_learning_law,
            "batch_reference": (
                self.args.batch_reference
                if self.args.batch_reference > 0 else self.args.batch
            ),
            "mean_dynamic_batch": (
                self.dynamic_batch_sum / max(1, self.dynamic_batch_count)
            ),
            "min_dynamic_batch": self.dynamic_batch_min,
            "max_dynamic_batch": self.dynamic_batch_max,
            "batch_controller": self.args.batch_controller,
            "batch_loss_ema": self.batch_loss_ema,
            "batch_fast_loss_ema": self.batch_fast_loss_ema,
            "batch_surprise": self.batch_surprise,
            "fast_utility": self.fast_utility,
            "fast_utility_ema": self.fast_utility_ema,
            "fast_utility_baseline": self.fast_utility_baseline,
            "batch_controller_tokens": self.batch_controller_tokens,
            "residual_surprise": self.residual_surprise,
            "residual_surprise_l2": self.residual_surprise_l2,
            "residual_surprise_topk": self.residual_surprise_topk,
            "mean_residual_surprise": (
                self.residual_surprise_sum
                / max(1, self.residual_surprise_count)
            ),
            "max_residual_surprise": self.residual_surprise_max,
        }


def run(args, condition):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    experiment = Experiment(args, condition)
    files = {
        name: args.corpora / f"{name}.txt"
        for name in ("a1", "b", "a2", "a_held", "b_held")
    }
    result = {
        "condition": condition,
        "parameters": sum(p.numel() for p in experiment.model.parameters()),
        "args": vars(args) | {"corpora": str(args.corpora), "output": str(args.output)},
        "phases": {},
    }
    for phase in ("a1", "b", "a2"):
        train = experiment.train_file(files[phase], args.phase_bytes)
        result["phases"][phase] = {
            "train": train,
            "a": experiment.evaluate(files["a_held"]),
            "b": experiment.evaluate(files["b_held"]),
            "diagnostics": experiment.diagnostics(),
        }
        print(
            f"{condition}/{phase} A={result['phases'][phase]['a']['bpb']:.4f} "
            f"B={result['phases'][phase]['b']['bpb']:.4f} "
            f"{train['source_bytes_per_second']:.0f} b/s",
            flush=True,
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=[
        "global_uniform", "global_clocked", "local_uniform",
        "local_clocked", "local_clocked_untied",
    ])
    parser.add_argument("--bands", type=int, default=32)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--aperture-width", type=int, default=0)
    parser.add_argument("--language-depth", type=int, default=0)
    parser.add_argument("--language-expansion", type=int, default=2)
    parser.add_argument("--language-residual-scale", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--batch-mode", choices=["fixed", "dynamic"], default="fixed")
    parser.add_argument(
        "--batch-controller",
        choices=["io2", "surprise", "utility", "residual"],
        default="io2",
    )
    parser.add_argument("--batch-min", type=int, default=64)
    parser.add_argument("--batch-max", type=int, default=512)
    parser.add_argument("--batch-rise", type=float, default=0.005)
    parser.add_argument("--batch-fall", type=float, default=2.0)
    parser.add_argument("--batch-surprise-threshold", type=float, default=0.05)
    parser.add_argument("--batch-loss-ema", type=float, default=0.01)
    parser.add_argument("--batch-fast-loss-ema", type=float, default=0.2)
    parser.add_argument("--batch-utility-ema", type=float, default=0.1)
    parser.add_argument("--batch-utility-baseline-ema", type=float, default=0.005)
    parser.add_argument("--batch-utility-threshold", type=float, default=0.0)
    parser.add_argument("--batch-utility-relative", action="store_true")
    parser.add_argument("--batch-controller-warmup", type=int, default=0)
    parser.add_argument(
        "--batch-residual-baseline-horizon", type=float, default=8192.0
    )
    parser.add_argument("--batch-residual-threshold", type=float, default=0.05)
    parser.add_argument(
        "--batch-residual-reduction", choices=["l2", "topk"], default="l2"
    )
    parser.add_argument(
        "--batch-residual-observation-quantum", type=int, default=16
    )
    parser.add_argument(
        "--batch-lr-compensation", choices=["none", "tail", "linear"],
        default="none",
    )
    parser.add_argument(
        "--batch-learning-law",
        choices=["fixed", "source_normalized"],
        default="fixed",
    )
    parser.add_argument("--batch-reference", type=int, default=0)
    parser.add_argument("--phase-bytes", type=int, default=65_536)
    parser.add_argument("--eval-bytes", type=int, default=0)
    parser.add_argument("--lens-stride", type=int, default=1)
    parser.add_argument(
        "--io2-mode", choices=["off", "measure", "amplitude", "lens", "full"],
        default="off",
    )
    parser.add_argument(
        "--io2-error-space", choices=["translated", "token"],
        default="translated",
    )
    parser.add_argument(
        "--residual-kernel", choices=["dense", "auto"], default="dense"
    )
    parser.add_argument(
        "--io2-spectrum", choices=["proper", "remainder"],
        default="proper",
    )
    parser.add_argument(
        "--io2-sampling", choices=["hold", "corrected"], default="hold"
    )
    parser.add_argument(
        "--io2-batch-map",
        choices=["historical", "absolute", "calibrated", "peak"],
        default="historical",
    )
    parser.add_argument("--io2-stride-cap", type=int, default=256)
    parser.add_argument("--io2-decimation-gain", type=float, default=1.0)
    parser.add_argument("--io2-controller-quantum", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--mixer-lr", type=float, default=0.01)
    parser.add_argument("--schedule", choices=[
        "uniform", "phi", "reverse_phi", "horizon", "reverse_horizon",
        "linear_horizon", "two_speed", "all_slow",
    ], default="phi")
    parser.add_argument("--slow-coefficient", type=float, default=None)
    parser.add_argument(
        "--dictionary-clock", choices=["fast", "slow"], default="fast"
    )
    parser.add_argument("--directional", action="store_true")
    parser.add_argument("--match-update-norm", action="store_true")
    parser.add_argument(
        "--parameter-geometry", choices=["free", "matrix_sphere"],
        default="free",
    )
    parser.add_argument("--bias-ceiling", type=float, default=None)
    parser.add_argument(
        "--optimizer-kernel", choices=["dense", "auto"], default="dense"
    )
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": "aperture_clock_graft_v1", "runs": []}
    for condition in args.conditions:
        payload["runs"].append(run(args, condition))
        atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
