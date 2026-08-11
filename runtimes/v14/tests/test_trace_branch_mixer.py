import copy
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from trace_branch_mixer import (
    PHI,
    ReferenceTokenizer,
    TraceBranchConfig,
    TraceBranchMixer,
    TraceBranchTrainer,
    TraceState,
    TranslatedErrorBank,
    associative_projected_scan,
    selected_projected_scan,
    temporal_lattice,
)


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer" / "soma-1k-v1.model"


def tiny_config(vocab_size=31, bands=7):
    return TraceBranchConfig(
        vocab_size=vocab_size,
        bands=bands,
        symbol_dim=16,
        scale_rank=8,
        scale_depth=2,
        scale_kernel=3,
        summaries=3,
        model_dim=32,
        trunk_depth=2,
        expansion_dim=64,
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("bands", 7.5),
        ("bands", float("nan")),
        ("phi", float("inf")),
        ("prior_smoothing", float("nan")),
        ("epsilon", 0.0),
    ),
)
def test_config_rejects_nonrepresentable_values(field, value):
    values = tiny_config().to_dict()
    values[field] = value
    with pytest.raises(ValueError):
        TraceBranchConfig(**values)


def test_config_rejects_bands_beyond_float64_temporal_resolution():
    values = tiny_config().to_dict()
    values.update(bands=78, summaries=4, phi=PHI)
    with pytest.raises(ValueError, match="float64 temporal resolution"):
        TraceBranchConfig(**values)


def test_trainer_initialization_does_not_reseed_global_torch_rng():
    tokenizer = ReferenceTokenizer(TOKENIZER)
    torch.manual_seed(12345)
    state = torch.random.get_rng_state().clone()

    TraceBranchTrainer(
        tiny_config(vocab_size=tokenizer.vocab_size),
        tokenizer,
        device="cpu",
    )

    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize(
    "field,value",
    (
        ("learning_rate", float("nan")),
        ("weight_decay", -1.0),
        ("grad_clip", float("inf")),
        ("decimation_gain", -1.0),
        ("decimation_stride_cap", 257),
        ("pulse_quantum", 1.5),
        ("optimizer_quantum", float("nan")),
    ),
)
def test_trainer_rejects_invalid_runtime_controls(field, value):
    tokenizer = ReferenceTokenizer(TOKENIZER)
    with pytest.raises(ValueError):
        TraceBranchTrainer(
            tiny_config(vocab_size=tokenizer.vocab_size),
            tokenizer,
            device="cpu",
            **{field: value},
        )


def test_exact_phi_lattice_closes_under_adjacent_composition():
    config = tiny_config(bands=12)
    beta, retention, injection, lifetime = temporal_lattice(config)
    assert math.isclose(config.phi, PHI)
    assert torch.allclose(
        beta[:-2], beta[1:-1] + beta[2:], atol=2e-15, rtol=2e-15
    )
    assert torch.allclose(
        retention[:-2],
        retention[1:-1] * retention[2:],
        atol=2e-15,
        rtol=2e-15,
    )
    assert torch.allclose(injection, 1.0 - retention)
    assert torch.allclose(lifetime, beta.reciprocal())


def test_closed_form_trace_batch_matches_token_updates_and_preserves_mass():
    config = tiny_config()
    rng = np.random.default_rng(37)
    ids = rng.integers(0, config.vocab_size, size=513, dtype=np.int64)
    initial = TraceState(config)
    batched = initial.advanced(ids, source_bytes=1000)
    stepped = initial.clone()
    for token_id in ids:
        stepped.advance_([int(token_id)], source_bytes=1000 / len(ids))
    assert torch.allclose(
        batched.trace, stepped.trace, atol=3e-14, rtol=3e-14
    )
    assert torch.allclose(
        batched.trace.sum(dim=0),
        torch.ones(config.bands, dtype=torch.float64),
        atol=2e-13,
        rtol=2e-13,
    )
    assert batched.last_token == int(ids[-1])
    assert batched.tokens_seen == len(ids)
    assert batched.source_bytes_seen == pytest.approx(1000.0)


def test_empirical_priors_are_exact_pre_token_states():
    config = tiny_config(vocab_size=7, bands=5)
    state = TraceState(config)
    ids = torch.tensor([2, 2, 4, 2, 6, 4])
    rows = torch.tensor([0, 1, 3, 5])
    priors = state.priors_before(ids, rows)

    cursor = state.prior_counts.clone()
    expected = []
    previous = 0
    for row in rows.tolist():
        cursor.add_(torch.bincount(
            ids[previous:row], minlength=config.vocab_size
        ))
        expected.append(cursor / cursor.sum())
        previous = row

    assert torch.equal(priors, torch.stack(expected))


def test_prior_weights_exclude_context_without_excluding_trace():
    config = tiny_config(vocab_size=7, bands=5)
    state = TraceState(config)
    ids = torch.tensor([1, 2, 3, 4])
    weights = torch.tensor([1.0, 0.0, 1.0, 0.0])
    advanced = state.advanced(ids, prior_weights=weights)

    assert advanced.tokens_seen == 4
    assert advanced.last_token == 4
    assert advanced.prior_counts[1] == 2.0
    assert advanced.prior_counts[2] == 1.0
    assert advanced.prior_counts[3] == 2.0
    assert advanced.prior_counts[4] == 1.0


def test_associative_scan_matches_serial_projected_recurrence():
    torch.manual_seed(3)
    batch, bands, features = 37, 9, 11
    start = torch.randn(bands, features, dtype=torch.float64)
    centered = torch.randn(batch, features, dtype=torch.float64)
    config = tiny_config(bands=bands)
    _, retention, injection, _ = temporal_lattice(config)
    parallel = associative_projected_scan(
        start, centered, retention, injection
    )
    serial = []
    state = start
    for value in centered:
        serial.append(state)
        state = (
            retention.unsqueeze(1) * state
            + injection.unsqueeze(1) * value.unsqueeze(0)
        )
    assert torch.allclose(
        parallel, torch.stack(serial), atol=2e-13, rtol=2e-13
    )


@pytest.mark.parametrize("stride", [1, 2, 3, 7, 16])
def test_selected_scan_matches_full_scan_at_regular_rows(stride):
    torch.manual_seed(31 + stride)
    batch, bands, features = 113, 9, 11
    start = torch.randn(bands, features, dtype=torch.float64)
    centered = torch.randn(batch, features, dtype=torch.float64)
    config = tiny_config(bands=bands)
    _, retention, injection, _ = temporal_lattice(config)
    rows = torch.arange(0, batch, stride)
    expected = associative_projected_scan(
        start, centered, retention, injection
    ).index_select(0, rows)
    selected = selected_projected_scan(
        start, centered, retention, injection, rows
    )
    assert selected.shape == (len(rows), bands, features)
    assert torch.allclose(selected, expected, atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("stride,phase", [(2, 1), (5, 3), (11, 7)])
def test_selected_scan_matches_full_scan_at_global_phase(stride, phase):
    torch.manual_seed(71 + stride)
    batch, bands, features = 97, 9, 11
    start = torch.randn(bands, features, dtype=torch.float64)
    centered = torch.randn(batch, features, dtype=torch.float64)
    config = tiny_config(bands=bands)
    _, retention, injection, _ = temporal_lattice(config)
    rows = torch.arange(phase, batch, stride)
    expected = associative_projected_scan(
        start, centered, retention, injection
    ).index_select(0, rows)
    selected = selected_projected_scan(
        start,
        centered,
        retention,
        injection,
        rows,
        regular_stride=stride,
    )
    assert torch.allclose(selected, expected, atol=3e-13, rtol=3e-13)


def test_selected_scan_falls_back_for_irregular_rows():
    torch.manual_seed(47)
    batch, bands, features = 29, 5, 7
    start = torch.randn(bands, features, dtype=torch.float64)
    centered = torch.randn(batch, features, dtype=torch.float64)
    config = tiny_config(bands=bands)
    _, retention, injection, _ = temporal_lattice(config)
    rows = torch.tensor([0, 2, 5, 11, 19, 28])
    expected = associative_projected_scan(
        start, centered, retention, injection
    ).index_select(0, rows)
    selected = selected_projected_scan(
        start, centered, retention, injection, rows
    )
    assert torch.allclose(selected, expected, atol=3e-13, rtol=3e-13)


def test_selected_scan_falls_back_for_one_nonzero_row():
    torch.manual_seed(53)
    batch, bands, features = 17, 5, 7
    start = torch.randn(bands, features, dtype=torch.float64)
    centered = torch.randn(batch, features, dtype=torch.float64)
    config = tiny_config(bands=bands)
    _, retention, injection, _ = temporal_lattice(config)
    rows = torch.tensor([11])
    expected = associative_projected_scan(
        start, centered, retention, injection
    ).index_select(0, rows)
    selected = selected_projected_scan(
        start, centered, retention, injection, rows
    )
    assert torch.allclose(selected, expected, atol=3e-13, rtol=3e-13)


def test_selected_scan_preserves_the_dense_scan_gradient():
    torch.manual_seed(59)
    batch, bands, features, stride = 41, 7, 9, 4
    config = tiny_config(bands=bands)
    _, retention, injection, _ = temporal_lattice(config)
    rows = torch.arange(0, batch, stride)

    dense_start = torch.randn(
        bands, features, dtype=torch.float64, requires_grad=True
    )
    dense_inputs = torch.randn(
        batch, features, dtype=torch.float64, requires_grad=True
    )
    sparse_start = dense_start.detach().clone().requires_grad_(True)
    sparse_inputs = dense_inputs.detach().clone().requires_grad_(True)

    dense = associative_projected_scan(
        dense_start, dense_inputs, retention, injection
    ).index_select(0, rows)
    sparse = selected_projected_scan(
        sparse_start,
        sparse_inputs,
        retention,
        injection,
        rows,
        regular_stride=stride,
    )
    dense.square().sum().backward()
    sparse.square().sum().backward()

    assert torch.allclose(sparse, dense, atol=3e-13, rtol=3e-13)
    assert torch.allclose(
        sparse_start.grad, dense_start.grad, atol=3e-12, rtol=3e-12
    )
    assert torch.allclose(
        sparse_inputs.grad, dense_inputs.grad, atol=3e-12, rtol=3e-12
    )


def test_translated_error_bank_matches_serial_updates():
    torch.manual_seed(41)
    config = tiny_config(bands=9)
    values = torch.randn(37, config.symbol_dim, dtype=torch.float64)
    gaps = torch.randint(1, 6, (len(values),), dtype=torch.long)
    vectorized = TranslatedErrorBank(config)
    vectorized.advance(values, gaps)

    serial = torch.zeros_like(vectorized.trace)
    for value, gap in zip(values, gaps):
        retention = torch.pow(vectorized.retention, int(gap)).unsqueeze(1)
        serial = retention * serial + (1.0 - retention) * value
    assert torch.allclose(vectorized.trace, serial, atol=2e-13, rtol=2e-13)
    assert vectorized.bandpass().shape == (
        config.bands, config.symbol_dim
    )


def test_sampled_residual_is_a_causal_importance_weighted_impulse():
    config = tiny_config(bands=9)
    bank = TranslatedErrorBank(config)
    value = torch.ones(config.symbol_dim, dtype=torch.float64)
    bank.advance_impulses(value, [4], importance=4.0)

    expected = (
        4.0 * (1.0 - bank.retention).unsqueeze(1)
        * value.unsqueeze(0)
    )
    assert torch.allclose(bank.trace, expected, atol=2e-15, rtol=2e-15)


def test_dense_impulse_residual_matches_scalar_ema():
    torch.manual_seed(83)
    config = tiny_config(bands=9)
    values = torch.randn(37, config.symbol_dim, dtype=torch.float64)
    bank = TranslatedErrorBank(config)
    bank.advance_impulses(values, torch.ones(37), importance=1.0)

    serial = torch.zeros_like(bank.trace)
    injection = 1.0 - bank.retention
    for value in values:
        serial = (
            bank.retention.unsqueeze(1) * serial
            + injection.unsqueeze(1) * value.unsqueeze(0)
        )
    assert torch.allclose(bank.trace, serial, atol=2e-13, rtol=2e-13)


def test_sampled_residual_calibration_keeps_white_baseline_stable():
    torch.manual_seed(89)
    config = tiny_config(bands=12)
    values = torch.randn(
        65_536, config.symbol_dim, dtype=torch.float64
    )
    measurements = []
    for stride in (1, 16, 256):
        bank = TranslatedErrorBank(config)
        sampled = values[stride - 1 :: stride]
        bank.advance_impulses(
            sampled,
            [stride] * len(sampled),
            band_gain=bank.sampling_gain(stride),
        )
        energy = (
            bank.energy() * bank.bandpass_sampling_calibration(stride)
        )
        probability = energy / energy.sum()
        target = float(
            probability @ torch.arange(config.bands, dtype=torch.float64)
        )
        measurements.append(target)

    assert max(measurements) - min(measurements) < 1.0


def test_prediction_decimation_preserves_the_exact_full_trace():
    tokenizer = ReferenceTokenizer(TOKENIZER)
    config = TraceBranchConfig(
        **{
            **tiny_config(vocab_size=tokenizer.vocab_size).to_dict(),
            "bands": 7,
        }
    )
    trainer = TraceBranchTrainer(config, tokenizer, device="cpu")
    trainer.decimation_band = math.log(4.0, config.phi)
    trainer._set_prediction_stride()
    ids = tokenizer.encode(
        "every source token remains part of the represented present"
    )
    applied_stride = trainer.prediction_stride
    expected = trainer.trace.advanced(ids, source_bytes=73)
    metrics = trainer.train_batch(ids, source_bytes=73, loss_bytes=73)
    assert applied_stride == 4
    assert metrics.predictions == math.ceil(len(ids) / applied_stride)
    assert torch.equal(trainer.trace.trace, expected.trace)
    assert trainer.trace.tokens_seen == len(ids)


def test_observed_context_advances_trace_and_controller_on_one_clock():
    tokenizer = ReferenceTokenizer(TOKENIZER)
    config = tiny_config(vocab_size=tokenizer.vocab_size, bands=9)
    trainer = TraceBranchTrainer(config, tokenizer, device="cpu")
    trainer.error_bank.trace.fill_(1.0)
    trainer._refresh_error_state()
    ids = tokenizer.encode("masked context still occupies lived time")
    expected = torch.pow(
        trainer.error_bank.retention, len(ids)
    ).unsqueeze(1).expand_as(trainer.error_bank.trace)

    trainer.observe(ids, source_bytes=39, update_prior=False)

    assert trainer.trace.tokens_seen == len(ids)
    assert trainer.training_tokens_seen == len(ids)
    assert trainer.last_error_token == len(ids)
    assert torch.allclose(trainer.error_bank.trace, expected)


def test_trailing_decimated_tokens_age_controller_to_block_end():
    tokenizer = ReferenceTokenizer(TOKENIZER)
    config = tiny_config(vocab_size=tokenizer.vocab_size, bands=9)
    trainer = TraceBranchTrainer(config, tokenizer, device="cpu")
    trainer.decimation_band = math.log(4.0, config.phi)
    trainer._set_prediction_stride()
    ids = tokenizer.encode("the controller clock reaches the exact block end")

    trainer.train_batch(ids, source_bytes=48, loss_bytes=48)

    assert trainer.training_tokens_seen == len(ids)
    assert trainer.last_error_token == len(ids)


def test_prediction_grid_keeps_global_phase_across_source_partitions():
    tokenizer = ReferenceTokenizer(TOKENIZER)
    config = tiny_config(vocab_size=tokenizer.vocab_size, bands=9)
    whole = TraceBranchTrainer(
        config, tokenizer, device="cpu", learning_rate=0.0
    )
    partitioned = TraceBranchTrainer(
        config, tokenizer, device="cpu", learning_rate=0.0
    )
    for trainer in (whole, partitioned):
        trainer.prediction_stride = 4
        trainer.pulse_enabled = False
        trainer._apply_io2 = lambda *_args, **_kwargs: None
    ids = np.resize(tokenizer.encode("global prediction phase"), 10)

    whole.train_batch(ids, source_bytes=10, loss_bytes=10)
    partitioned.train_batch(ids[:3], source_bytes=3, loss_bytes=3)
    partitioned.train_batch(ids[3:], source_bytes=7, loss_bytes=7)

    assert whole.predictions == partitioned.predictions == 3


def test_io2_controls_lr_and_the_next_prediction_stride():
    tokenizer = ReferenceTokenizer(TOKENIZER)
    config = tiny_config(vocab_size=tokenizer.vocab_size, bands=12)
    trainer = TraceBranchTrainer(
        config,
        tokenizer,
        device="cpu",
        learning_rate=4e-4,
        lr_auto=True,
    )
    trainer.error_target_band = 8.0
    trainer._io2_ratio = lambda _loss: 0.25
    trainer._apply_io2(token_loss=1.0, nats_per_byte=0.0)
    assert trainer.learning_rate == pytest.approx(1e-4)
    assert trainer.io2_plasticity == pytest.approx(0.25)
    assert trainer.prediction_stride > 1


def test_projected_batch_is_current_embedding_applied_to_every_trace():
    torch.manual_seed(5)
    config = tiny_config()
    model = TraceBranchMixer(config)
    state = TraceState(config).advanced([2, 3, 5, 7, 11])
    targets = torch.tensor([13, 17, 19, 23, 29, 3])
    prior = state.prior()
    projected, direct = model.prepare_batch(
        state.trace, prior, state.last_token, targets
    )

    explicit_projected = []
    explicit_direct = []
    cursor = state.clone()
    for target in targets:
        value, instant, _ = model.project_state(
            cursor.trace, prior, cursor.last_token
        )
        explicit_projected.append(value)
        explicit_direct.append(instant)
        cursor.advance_([int(target)], update_prior=False)

    assert torch.allclose(
        projected,
        torch.stack(explicit_projected),
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        direct,
        torch.stack(explicit_direct),
        atol=2e-6,
        rtol=2e-6,
    )


def test_zero_branch_readout_is_exactly_the_reference_prior():
    torch.manual_seed(7)
    config = tiny_config()
    model = TraceBranchMixer(config)
    state = TraceState(config).advanced([2, 2, 2, 3, 5])
    targets = torch.tensor([7, 11, 13, 17])
    logits = model(
        state.trace, state.prior(), state.last_token, targets
    )
    expected = torch.log(state.prior().float())
    assert torch.allclose(
        logits, expected.unsqueeze(0).expand_as(logits), atol=1e-7
    )


def test_prediction_is_causal_with_respect_to_current_target():
    torch.manual_seed(11)
    config = tiny_config()
    model = TraceBranchMixer(config)
    with torch.no_grad():
        model.branch.weight.normal_(std=0.03)
    state = TraceState(config).advanced([2, 3, 5])
    left = model(
        state.trace,
        state.prior(),
        state.last_token,
        torch.tensor([7, 11, 13]),
    )
    right = model(
        state.trace,
        state.prior(),
        state.last_token,
        torch.tensor([17, 11, 13]),
    )
    assert torch.allclose(left[0], right[0], atol=1e-7)
    assert not torch.allclose(left[1], right[1])


def test_language_gradient_reaches_embedding_scale_and_serial_depth():
    torch.manual_seed(13)
    config = tiny_config()
    model = TraceBranchMixer(config)
    with torch.no_grad():
        model.branch.weight.normal_(std=0.03)
        for block in model.scale_encoder.blocks:
            block.up.weight.normal_(std=0.03)
        for block in model.trunk:
            block.out.weight.normal_(std=0.03)
    state = TraceState(config).advanced([2, 3, 5, 7, 11])
    targets = torch.tensor([13, 17, 19, 23, 29, 3, 5, 7])
    logits = model(
        state.trace, state.prior(), state.last_token, targets
    )
    F.cross_entropy(logits, targets).backward()
    assert model.embedding.weight.grad is not None
    assert float(model.embedding.weight.grad.norm()) > 0.0
    assert float(model.context.weight.grad.norm()) > 0.0
    assert float(model.branch.weight.grad.norm()) > 0.0
    assert all(
        float(block.depthwise.weight.grad.norm()) > 0.0
        and float(block.down.weight.grad.norm()) > 0.0
        and float(block.up.weight.grad.norm()) > 0.0
        for block in model.scale_encoder.blocks
    )
    assert all(
        float(block.gate.weight.grad.norm()) > 0.0
        and float(block.value.weight.grad.norm()) > 0.0
        and float(block.out.weight.grad.norm()) > 0.0
        for block in model.trunk
    )
    assert state.trace.requires_grad is False
    assert state.trace.grad is None


def test_learned_parameter_count_does_not_grow_with_band_count():
    short = TraceBranchMixer(tiny_config(bands=7))
    long_config = TraceBranchConfig(
        **{
            **tiny_config(bands=7).to_dict(),
            "bands": 19,
            "summaries": 3,
        }
    )
    long = TraceBranchMixer(long_config)
    assert short.parameter_count == long.parameter_count


def test_failed_unweighted_batch_does_not_advance_present():
    tokenizer = ReferenceTokenizer(TOKENIZER)
    config = TraceBranchConfig(
        **{
            **tiny_config(vocab_size=tokenizer.vocab_size).to_dict(),
            "bands": 5,
            "summaries": 3,
        }
    )
    trainer = TraceBranchTrainer(config, tokenizer, device="cpu")
    before = trainer.trace.state_dict()
    with pytest.raises(ValueError, match="no weighted targets"):
        trainer.train_batch(
            [2, 3, 5, 7], loss_weights=np.zeros(4, dtype=np.float32)
        )
    assert trainer.trace.tokens_seen == before["tokens_seen"]
    assert torch.equal(trainer.trace.trace, before["trace"])
    assert trainer.updates == 0


def test_generation_cache_and_authoritative_trace_agree():
    tokenizer = ReferenceTokenizer(TOKENIZER)
    config = TraceBranchConfig(
        **{
            **tiny_config(vocab_size=tokenizer.vocab_size).to_dict(),
            "bands": 5,
            "summaries": 3,
        }
    )
    trainer = TraceBranchTrainer(config, tokenizer, device="cpu")
    trainer.observe(tokenizer.encode("a small exact present"))
    before = trainer.trace.clone()
    generated = trainer.generate_ids(8, greedy=True)
    expected = before.advanced(generated, update_prior=False)
    assert torch.allclose(
        trainer.trace.trace, expected.trace, atol=2e-14, rtol=2e-14
    )
    assert torch.equal(
        trainer.trace.prior_counts, before.prior_counts
    )


def test_evaluation_advances_a_private_continuation_only():
    tokenizer = ReferenceTokenizer(TOKENIZER)
    config = TraceBranchConfig(
        **{
            **tiny_config(vocab_size=tokenizer.vocab_size).to_dict(),
            "bands": 5,
            "summaries": 3,
        }
    )
    trainer = TraceBranchTrainer(config, tokenizer, device="cpu")
    trainer.observe(tokenizer.encode("the lived state stays here."))
    before = trainer.trace.state_dict()
    continuation = tokenizer.encode(
        "the evaluation continues elsewhere without changing it."
    )
    metrics = trainer.evaluate_ids(
        continuation, source_bytes=54, batch_size=4
    )
    assert math.isfinite(metrics.bpb)
    assert metrics.tokens == len(continuation)
    assert torch.equal(trainer.trace.trace, before["trace"])
    assert torch.equal(
        trainer.trace.prior_counts, before["prior_counts"]
    )
    assert trainer.trace.last_token == before["last_token"]


def test_checkpoint_roundtrip_restores_exact_next_prediction():
    tokenizer = ReferenceTokenizer(TOKENIZER)
    config = TraceBranchConfig(
        **{
            **tiny_config(vocab_size=tokenizer.vocab_size).to_dict(),
            "bands": 5,
            "summaries": 3,
        }
    )
    trainer = TraceBranchTrainer(config, tokenizer, device="cpu")
    ids = tokenizer.encode(
        "the exact represented present survives a checkpoint boundary."
    )
    trainer.train_batch(ids, source_bytes=64)
    before = trainer.next_logits().detach()
    trace_before = trainer.trace.trace.clone()
    optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "mixer.pt"
        trainer.save(path, metadata={"description": "roundtrip"})
        restored = TraceBranchTrainer.load(
            path, tokenizer, device="cpu"
        )

    after = restored.next_logits().detach()
    assert torch.allclose(after, before, atol=1e-7, rtol=1e-7)
    assert torch.equal(restored.trace.trace, trace_before)
    assert restored.trace.tokens_seen == trainer.trace.tokens_seen
    assert restored.updates == trainer.updates
    assert restored.optimizer.state_dict()["param_groups"] == (
        optimizer_before["param_groups"]
    )
