import math
import tempfile
import json
import subprocess
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

import soma_v14 as soma_v13
import soma_gui
from soma_train_worker import (
    DISCARD_OUTPUT,
    MEMORY_EMERGENCY_FRACTION,
    MPS_CACHE_TRIM_MIN_BYTES,
    MPS_CACHE_TRIM_LIVE_RATIO,
    MEMORY_STOP_FRACTION,
    memory_guard_level,
    memory_recycle_threshold,
    mps_cache_trim_needed,
    recycle_config,
    release_cpu_allocator_pages,
    worker_exec_argv,
    Worker,
)
from soma_profiles import (
    CAPACITY_NAMES,
    DEFAULT_CAPACITY,
    capacity_values,
    default_batch_size,
    estimated_mps_training_bytes,
    infer_capacity,
    mps_fit_status,
    parameter_count,
    recommended_mps_batch,
)
from trace_branch_mixer import (
    ReferenceTokenizer,
    TemporalLattice,
    TokenCorpus,
    TraceBranchConfig,
    TraceBranchTrainer,
    TraceState,
    temporal_lattice,
)


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "tokenizer" / "soma-1k-v1.model"


@pytest.fixture(scope="module")
def _tk_root():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


def _button_harness(command, _root=None):
    if _root is None:
        import tkinter as tk
        _root = tk.Tk()
        _root.withdraw()
    button = soma_gui.RoundedButton(
        _root, "test", command, width=100, height=32)
    button._root = lambda: SimpleNamespace(
        report_callback_exception=lambda *a: None)
    return button


def test_rounded_button_click_fires_command(_tk_root):
    calls = []
    button = _button_harness(lambda: calls.append("called"), _tk_root)
    button._fire()
    assert calls == ["called"]


def test_mps_cache_trim_preserves_a_small_workspace_but_bounds_growth():
    gib = 1024**3
    allocated = int(1.4 * gib)
    allowance = max(
        MPS_CACHE_TRIM_MIN_BYTES,
        int(allocated * MPS_CACHE_TRIM_LIVE_RATIO),
    )
    assert not mps_cache_trim_needed(
        allocated, allocated + allowance
    )
    assert mps_cache_trim_needed(
        allocated, allocated + allowance + 1
    )
    large_live = 4 * gib
    large_allowance = max(
        MPS_CACHE_TRIM_MIN_BYTES,
        int(large_live * MPS_CACHE_TRIM_LIVE_RATIO),
    )
    assert not mps_cache_trim_needed(
        large_live, large_live + large_allowance
    )
    assert mps_cache_trim_needed(
        large_live, large_live + large_allowance + 1
    )


def test_memory_guard_allows_unified_memory_but_catches_runaway_growth():
    physical = 8 * 1024**3
    assert memory_guard_level(int(0.90 * physical), physical) == "normal"
    assert memory_guard_level(
        int(MEMORY_STOP_FRACTION * physical), physical
    ) == "high"
    assert memory_guard_level(
        int(MEMORY_EMERGENCY_FRACTION * physical), physical
    ) == "emergency"


def test_cpu_allocator_relief_is_safe_on_the_current_platform():
    assert release_cpu_allocator_pages() >= 0


def test_memory_recycle_threshold_respects_model_baseline_and_machine_size():
    gib = 1024**3
    assert memory_recycle_threshold(2 * gib, 8 * gib) == 6 * gib
    assert memory_recycle_threshold(10 * gib, 16 * gib) == 13 * gib


def test_recycle_config_resumes_the_published_cursor_without_forking():
    original = {
        "checkpoint": "old.pt",
        "save_path": "current.pt",
        "fork": True,
        "start_byte": "123",
        "start_byte_override": True,
    }
    recycled = recycle_config(original)
    assert recycled["checkpoint"] == "current.pt"
    assert recycled["fork"] is False
    assert recycled["start_byte"] == "0"
    assert recycled["start_byte_override"] is False
    assert original["checkpoint"] == "old.pt"


def test_worker_recycle_command_preserves_worker_arguments():
    argv = worker_exec_argv("config.json", "stop.json")
    assert argv[-2:] == ["config.json", "stop.json"]
    assert str(Path("soma_train_worker.py")) in argv[-3]


def test_autosave_recycle_does_not_write_the_checkpoint_twice(tmp_path):
    worker = Worker(tmp_path / "config.json", tmp_path / "stop.json")
    worker.recycle_requested = True
    worker.recycle_checkpoint_published = True
    worker.save_on_stop = True
    calls = []
    worker.quiet = lambda *args, **kwargs: calls.append((args, kwargs))

    worker.final_save(SimpleNamespace(save=lambda _path: None), tmp_path / "m.pt")

    assert calls == []


def test_managed_process_reaps_finished_child_resources():
    process = soma_gui.ManagedProcess("test")
    assert process.start(["-c", "print('done')"])
    deadline = time.time() + 5
    output = []
    while time.time() < deadline and process.proc is not None:
        output.extend(process.drain())
        time.sleep(0.01)
    output.extend(process.drain())

    assert output == ["done"]
    assert process.proc is None
    assert process._reader is None


def test_quiet_output_sink_never_retains_training_progress():
    payload = "progress" * 1_000
    assert DISCARD_OUTPUT.write(payload) == len(payload)
    assert vars(DISCARD_OUTPUT) == {}
    assert DISCARD_OUTPUT.flush() is None
    assert not DISCARD_OUTPUT.isatty()


def test_rounded_button_click_does_not_fire_when_disabled(_tk_root):
    calls = []
    button = _button_harness(lambda: calls.append("called"), _tk_root)
    button._enabled = False
    button._fire()
    assert calls == []


def test_rounded_button_fires_on_press_without_motion_or_release():
    import tkinter as tk

    root = tk.Tk()
    root.geometry("240x100+20+20")
    calls = []
    button = soma_gui.RoundedButton(
        root, "test", lambda: calls.append("called"),
        width=100, height=32)
    button.pack()
    root.update()
    x = button.winfo_width() // 2
    y = button.winfo_height() // 2
    button.event_generate(
        "<ButtonPress-1>", x=x, y=y,
        rootx=button.winfo_rootx() + x,
        rooty=button.winfo_rooty() + y,
        when="now",
    )
    root.update()
    assert calls == ["called"]
    button.event_generate(
        "<ButtonRelease-1>", x=x, y=y,
        rootx=button.winfo_rootx() + x,
        rooty=button.winfo_rooty() + y,
        when="now",
    )
    root.update()
    assert calls == ["called"]
    root.destroy()


def test_streamed_chat_fragments_are_painted_as_one_chunk():
    app = object.__new__(soma_gui.App)
    app.chat_response_buf = []
    app.logos_response_buf = []
    app._logos_mode = True
    appends = []
    spoken = []
    app._chat_append = lambda text, tag, persist: appends.append(
        (text, tag, persist))
    app.voice = SimpleNamespace(feed=lambda text: spoken.append(text))
    parts = ["one", " ", "chunk"]

    app._flush_chat_chars(parts)

    assert parts == []
    assert appends == [("one chunk", "soma", False)]
    assert app.chat_response_buf == ["one chunk"]
    assert app.logos_response_buf == ["one chunk"]
    assert spoken == ["one chunk"]


def tiny_config(vocab_size):
    return TraceBranchConfig(
        vocab_size=vocab_size,
        bands=7,
        symbol_dim=16,
        scale_rank=8,
        scale_depth=1,
        scale_kernel=3,
        summaries=3,
        model_dim=32,
        trunk_depth=1,
        expansion_dim=64,
    )


def test_capacity_profiles_are_coherent_and_monotonic():
    counts = [parameter_count(name, 48) for name in CAPACITY_NAMES]
    assert counts == sorted(counts)
    assert counts[0] == 28_156_934
    assert counts[1] == 65_508_870
    assert counts[2] == 93_820_422
    assert counts[3] == 221_273_608
    for name in CAPACITY_NAMES:
        values = capacity_values(name, 48)
        assert infer_capacity({"n_bands": 48, **values}) == name


def test_lattice_prior_is_the_phi_lattice():
    config = tiny_config(64)
    lattice = TemporalLattice(config)
    _, retention, injection, _ = lattice.derive()
    _, expected_retention, expected_injection, _ = temporal_lattice(config)
    assert torch.equal(retention, expected_retention)
    assert torch.equal(injection, expected_injection)
    assert float(lattice.drift().abs().max()) == 0.0
    assert torch.equal(
        lattice.interval_coordinates(),
        torch.arange(config.bands, dtype=torch.float64),
    )


def test_lattice_deform_densifies_the_high_error_gap():
    config = tiny_config(64)
    lattice = TemporalLattice(config)
    energy = torch.zeros(config.bands - 1, dtype=torch.float64)
    energy[2] = 1.0
    lattice.deform_(energy, elapsed_tokens=100_000, rate=0.05, elastic=0.1)
    gaps = lattice.coordinates[1:] - lattice.coordinates[:-1]
    assert float(gaps[2]) < 1.0
    assert float(gaps[0]) == 1.0
    assert float(lattice.coordinates[-1]) == float(config.bands - 1)
    assert float(lattice.coordinates[0]) == 0.0
    assert bool(torch.all(gaps > 0.0))


def test_lattice_deformation_is_local_even_for_slow_residuals():
    config = TraceBranchConfig(bands=64, summaries=8)
    lattice = TemporalLattice(config)
    energy = torch.zeros(config.bands - 1, dtype=torch.float64)
    energy[-1] = 1.0
    before = lattice.coordinates.clone()
    lattice.deform_(energy, elapsed_tokens=1024, rate=0.02, elastic=0.1)
    movement = lattice.coordinates - before
    gaps = lattice.coordinates[1:] - lattice.coordinates[:-1]
    assert float(movement[1].abs()) == 0.0
    assert float(movement[-2].abs()) < 1e-6
    assert float(movement[-1].abs()) == 0.0
    assert float(gaps[-1]) < 1.0


def test_lattice_elastic_relaxes_back_to_the_prior():
    config = tiny_config(64)
    lattice = TemporalLattice(config)
    energy = torch.zeros(config.bands - 1, dtype=torch.float64)
    energy[1] = 1.0
    for _ in range(50):
        lattice.deform_(
            energy, elapsed_tokens=1024, rate=0.05, elastic=0.1
        )
    assert float(lattice.drift().abs().max()) > 0.01
    silence = torch.zeros(config.bands - 1, dtype=torch.float64)
    for _ in range(3000):
        lattice.deform_(
            silence, elapsed_tokens=1024, rate=0.05, elastic=0.5
        )
    assert float(lattice.drift().abs().max()) < 0.05


def test_lattice_gaps_stay_inside_clamps_and_resolution_wall():
    config = TraceBranchConfig(bands=64, summaries=8)
    lattice = TemporalLattice(config)
    energy = torch.zeros(config.bands - 1, dtype=torch.float64)
    energy[0] = 1.0
    for _ in range(500):
        lattice.deform_(
            energy, elapsed_tokens=1024, rate=0.2, elastic=0.0
        )
    gaps = lattice.coordinates[1:] - lattice.coordinates[:-1]
    assert float(gaps.min()) >= TemporalLattice.MIN_GAP - 1e-9
    assert float(gaps.max()) <= TemporalLattice.MAX_GAP + 1e-9
    slowest = float(lattice.coordinates[-1]) + float(
        lattice.coordinates[-1] - lattice.coordinates[-2]
    )
    assert slowest <= lattice._resolution_wall() + 1e-9


def test_lattice_bands_cannot_move_faster_than_they_integrate():
    config = TraceBranchConfig(bands=64, summaries=8)
    lattice = TemporalLattice(config)
    # equal discontinuities near the fast and slow ends: the interior nodes
    # receive the same normalized drive but move on their own clocks.
    energy = torch.zeros(config.bands - 1, dtype=torch.float64)
    energy[0] = 1.0
    energy[-1] = 1.0
    before = lattice.coordinates.clone()
    lattice.deform_(energy, elapsed_tokens=1, rate=0.2, elastic=0.0)
    moved = (lattice.coordinates - before).abs()
    assert float(moved[-2]) < 0.02
    assert float(moved[1]) > 5.0 * float(moved[-2])


def test_production_lattice_holds_every_invariant_under_abuse():
    config = TraceBranchConfig(bands=64, summaries=8)
    lattice = TemporalLattice(config)
    energy = torch.zeros(config.bands - 1, dtype=torch.float64)
    for index in (0, 15, 31, 62):
        energy.zero_()
        energy[index] = 1.0
        for _ in range(500):
            lattice.deform_(
                energy, elapsed_tokens=1024, rate=0.2, elastic=0.0
            )
    gaps = lattice.coordinates[1:] - lattice.coordinates[:-1]
    assert float(lattice.coordinates[0]) == 0.0
    assert float(lattice.coordinates[-1]) == 63.0
    assert float(gaps.min()) >= TemporalLattice.MIN_GAP - 1e-12
    assert float(gaps.max()) <= TemporalLattice.MAX_GAP + 1e-12
    assert float(lattice._extended(lattice.coordinates)[-1]) <= (
        lattice._resolution_wall() + 1e-12
    )


def test_lattice_evidence_clock_is_source_partition_stable():
    config = TraceBranchConfig(bands=16, summaries=8)
    energy = torch.zeros(config.bands - 1, dtype=torch.float64)
    energy[4] = 1.0
    whole = TemporalLattice(config)
    split = TemporalLattice(config)
    whole.deform_(energy, elapsed_tokens=1024, rate=0.002, elastic=0.1)
    split.deform_(energy, elapsed_tokens=512, rate=0.002, elastic=0.1)
    split.deform_(energy, elapsed_tokens=512, rate=0.002, elastic=0.1)
    assert torch.allclose(
        whole.monitor_density, split.monitor_density, atol=1e-14, rtol=0.0
    )
    # coordinate integration is nonlinear, but repartitioning one reference
    # interval may only introduce a second-order integration difference.
    assert float((whole.coordinates - split.coordinates).abs().max()) < 2e-4


def test_trace_state_shares_a_live_lattice():
    config = tiny_config(64)
    lattice = TemporalLattice(config)
    state = TraceState(config, lattice=lattice)
    advanced = state.advanced([1, 2, 3])
    assert advanced.lattice is lattice
    assert advanced.clone().lattice is lattice


def test_io2_target_uses_physical_lattice_coordinates():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    config = tiny_config(tokenizer.vocab_size)
    trainer = TraceBranchTrainer(config, tokenizer, device="cpu")
    coordinates = torch.tensor(
        [0.0, 0.5, 1.25, 2.25, 3.5, 4.75, 6.0],
        dtype=torch.float64,
    )
    trainer.lattice = TemporalLattice(config, coordinates)
    trainer.trace.lattice = trainer.lattice
    trainer.error_bank.lattice = trainer.lattice
    trainer.error_bank.refresh_lattice()
    trainer.error_bank.trace.zero_()
    trainer.error_bank.trace[:3, 0] = 1.0
    trainer._refresh_error_state()
    expected = float(trainer.lattice.interval_coordinates()[2])
    assert abs(trainer.error_target_band - expected) < 1e-12


def test_frozen_lattice_rate_keeps_the_phi_prior_through_training():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size), tokenizer, device="cpu"
    )
    ids = tokenizer.encode("the creature walked along the river. " * 30)
    trainer.train_batch(ids[:128], source_bytes=100.0)
    assert float(trainer.lattice.drift().abs().max()) == 0.0


def test_checkpoint_roundtrips_a_deformed_lattice(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size),
        tokenizer,
        device="cpu",
        lattice_rate=0.05,
    )
    ids = tokenizer.encode("the creature walked along the river. " * 60)
    for start in range(0, 256, 64):
        trainer.train_batch(ids[start : start + 64], source_bytes=100.0)
    assert float(trainer.lattice.drift().abs().max()) > 0.0
    path = tmp_path / "deformed.pt"
    trainer.save(path)
    loaded = TraceBranchTrainer.load(path, tokenizer, device="cpu")
    assert torch.equal(
        loaded.lattice.coordinates, trainer.lattice.coordinates
    )
    assert torch.equal(
        loaded.lattice.monitor_density, trainer.lattice.monitor_density
    )
    assert loaded.lattice_rate == 0.05
    assert loaded.trace.lattice is loaded.lattice
    assert loaded.error_bank.lattice is loaded.lattice
    metrics = loaded.train_batch(ids[256:320], source_bytes=100.0)
    assert math.isfinite(metrics.loss)


def test_gui_reads_saved_lattice_rate_from_sidecar_or_checkpoint_state():
    assert soma_gui._checkpoint_lattice_rate({"lattice_rate": 0.023}) == 0.023
    assert soma_gui._checkpoint_lattice_rate({
        "training": {"lattice_rate": 0.017}
    }) == 0.017


def test_atomic_checkpoint_save_requires_payload_and_system_reserve(
        tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    path.write_bytes(b"existing checkpoint")
    checkpoint = {"weights": torch.zeros(1024, dtype=torch.float32)}
    monkeypatch.setattr(
        soma_v13.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1024**3),
    )

    with pytest.raises(OSError, match="not enough free disk space"):
        soma_v13._require_atomic_save_space(path, checkpoint)


def test_adaptive_lattice_resume_is_bit_exact(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size),
        tokenizer,
        device="cpu",
        lattice_rate=0.01,
        optimizer_quantum=64,
    )
    ids = tokenizer.encode("the creature walked along the river. " * 100)
    for start in range(0, 256, 64):
        trainer.train_batch(ids[start : start + 64], source_bytes=100.0)
    path = tmp_path / "adaptive.pt"
    trainer.save(path)
    uninterrupted = TraceBranchTrainer.load(path, tokenizer, device="cpu")
    resumed = TraceBranchTrainer.load(path, tokenizer, device="cpu")
    for start in range(256, 512, 64):
        uninterrupted.train_batch(
            ids[start : start + 64], source_bytes=100.0
        )
        resumed.train_batch(ids[start : start + 64], source_bytes=100.0)
    assert torch.equal(
        uninterrupted.lattice.coordinates, resumed.lattice.coordinates
    )
    assert torch.equal(
        uninterrupted.lattice.monitor_density,
        resumed.lattice.monitor_density,
    )
    assert torch.equal(uninterrupted.trace.trace, resumed.trace.trace)
    for left, right in zip(
        uninterrupted.model.parameters(), resumed.model.parameters()
    ):
        assert torch.equal(left, right)


def test_v13_checkpoint_loads_as_a_phi_prior_embryo(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    config = tiny_config(tokenizer.vocab_size)
    trainer = TraceBranchTrainer(config, tokenizer, device="cpu")
    checkpoint = trainer.checkpoint_dict()
    checkpoint["checkpoint_version"] = 13
    checkpoint["species"] = "soma_v13_trace_branch_mixer"
    del checkpoint["lattice"]["coordinates"]
    for key in ("lattice_rate", "lattice_elastic", "last_lattice_token"):
        del checkpoint["training"][key]
    path = tmp_path / "v13.pt"
    torch.save(checkpoint, path)
    loaded = TraceBranchTrainer.load(path, tokenizer, device="cpu")
    assert torch.equal(
        loaded.lattice.coordinates,
        torch.arange(config.bands, dtype=torch.float64),
    )
    assert loaded.lattice_rate == 0.0


def test_measured_mps_envelope_drives_conservative_auto_batches():
    memory_limit = 5_726_633_984
    assert default_batch_size("small") == 1024
    assert default_batch_size("medium") == 512
    assert default_batch_size("large") == 512
    assert default_batch_size("extra large") == 256
    assert recommended_mps_batch("small", 48, memory_limit) == 1024
    assert recommended_mps_batch("medium", 48, memory_limit) == 512
    assert recommended_mps_batch("large", 48, memory_limit) == 512
    assert mps_fit_status(
        "medium", 48, 1024, memory_limit
    )[0] == "pressure"
    assert mps_fit_status(
        "large", 48, 512, memory_limit
    )[0] == "comfortable"
    assert estimated_mps_training_bytes("large", 64, 512) > (
        estimated_mps_training_bytes("large", 48, 512)
    )


def test_cockpit_reattaches_to_the_exact_live_worker(tmp_path, monkeypatch):
    config = tmp_path / "run.config.json"
    events = tmp_path / "run.events.jsonl"
    stop = tmp_path / "run.stop.json"
    manifest = tmp_path / "run.manifest.json"
    pointer = tmp_path / "active.json"
    config.write_text("{}", encoding="utf-8")
    events.write_text("", encoding="utf-8")
    process = subprocess.Popen([
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        "soma_train_worker.py",
        str(config),
    ])
    try:
        now = time.time()
        soma_gui._write_json_atomic(manifest, {
            "version": 1,
            "run_id": "test-run",
            "status": "running",
            "pid": process.pid,
            "started_at": now,
            "heartbeat_at": now,
            "last_report_at": now,
            "checkpoint_path": str(tmp_path / "model.pt"),
        })
        soma_gui._write_json_atomic(pointer, {
            "manifest_path": str(manifest),
            "event_path": str(events),
            "config_path": str(config),
            "stop_path": str(stop),
            "pid": process.pid,
        })
        monkeypatch.setattr(soma_gui, "ACTIVE_TRAIN_RUN_FILE", pointer)
        worker = soma_gui.TrainWorker()
        assert worker.attached is True
        assert worker.is_running() is True
        worker.stop(save=True)
        assert json.loads(stop.read_text(encoding="utf-8"))["save"] is True
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_new_product_model_defaults_to_medium_capacity_and_pulse():
    assert DEFAULT_CAPACITY == "medium"
    model = soma_v13.SOMA(n_bands=8, profile="small", device="cpu")
    assert model.trainer.pulse_enabled is True
    assert model.trainer.pulse_schedule == "spectral"


@pytest.mark.parametrize("batch_size", [64, 256, 1024])
def test_pulse_clock_uses_the_configured_dense_batch(batch_size):
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=batch_size,
        device="cpu",
    )
    assert model.trainer.pulse_quantum == batch_size
    model.trainer.io2_plasticity = 1.0
    model.trainer.updates = 1
    model.trainer.batches = 2
    model.trainer.pulse_phase = 0.0
    assert model.trainer._schedule_update(batch_size) is True


def test_changing_batch_size_recalibrates_the_pulse_clock():
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=1024,
        device="cpu",
    )
    model.batch_size = 256
    assert model.batch_size == 256
    assert model.trainer.pulse_quantum == 256
    assert model.trainer.optimizer_quantum == 256
    assert model.trainer.learning_rate_base == pytest.approx(
        model.lr_base * 0.25
    )


@pytest.mark.parametrize("value", (0, 1.5, float("nan"), True))
def test_product_rejects_invalid_batch_sizes(value):
    with pytest.raises(ValueError, match="batch size"):
        soma_v13.SOMA(
            n_bands=8,
            profile="small",
            batch_size=value,
            device="cpu",
        )


def test_batch_size_change_finishes_the_existing_learning_quantum():
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=8,
        lr_auto=False,
        device="cpu",
    )
    model.trainer.pulse_enabled = False
    ids = np.resize(model.tokenizer.encode("unfinished quantum"), 4)
    model.trainer.train_batch(ids, source_bytes=4, loss_bytes=4)

    model.batch_size = 16
    assert model.batch_size == 16
    assert model.trainer.optimizer_quantum == 8
    assert model.trainer.pending_optimizer_quantum == 16

    model.trainer.train_batch(ids, source_bytes=4, loss_bytes=4)
    assert model.trainer.updates == 1
    assert model.trainer.optimizer_quantum == 16
    assert model.trainer.pulse_quantum == 16
    assert model.trainer.pending_optimizer_quantum is None


def test_partial_quantum_restores_current_lr_after_captured_step():
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=8,
        lr_auto=False,
        device="cpu",
    )
    model.trainer.pulse_enabled = False
    ids = np.resize(model.tokenizer.encode("captured learning rate"), 8)
    model.trainer.train_batch(ids[:4], source_bytes=4, loss_bytes=4)
    captured = model.trainer.accumulation_learning_rate
    model.lr_base *= 2.0
    model._set_optimizer_lrs()
    assert model.trainer.learning_rate != captured

    model.trainer.train_batch(ids[4:], source_bytes=4, loss_bytes=4)

    assert model.trainer.optimizer.param_groups[0]["lr"] == pytest.approx(
        model.trainer.learning_rate
    )


def test_batch_scaled_lr_conserves_source_time_plasticity_budget():
    budgets = []
    for batch_size in (64, 256, 1024):
        model = soma_v13.SOMA(
            n_bands=8,
            profile="small",
            batch_size=batch_size,
            lr=3e-4,
            lr_base=3e-4,
            device="cpu",
        )
        model.trainer.io2_plasticity = 0.1
        model.trainer._pulse_frequency = lambda: 0.25
        model._set_optimizer_lrs()
        updates_per_reference_span = (1024 / batch_size) * 0.25
        budgets.append(
            updates_per_reference_span
            * model.trainer.learning_rate
        )
    assert budgets == pytest.approx([3e-5, 3e-5, 3e-5])


def test_protocol_controls_exist_and_generation_forbids_roles():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    assert tokenizer.control_ids["turn"] >= 0
    assert tokenizer.control_ids["user"] in tokenizer.generation_forbidden_ids
    assert tokenizer.control_ids["soma"] in tokenizer.generation_forbidden_ids
    assert tokenizer.diagnostic_piece(tokenizer.control_ids["turn"]) == "[turn]"
    newline_id = int(tokenizer.encode("\n")[0])
    assert tokenizer.diagnostic_piece(newline_id) == "\\n"


def test_streaming_corpus_can_end_a_pending_feedback_block_at_newline(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    text = "first record ends here\nsecond record continues much further\n"
    path = tmp_path / "records.txt"
    path.write_text(text, encoding="utf-8")

    with TokenCorpus(path, tokenizer) as corpus:
        first = corpus.next_block(4096, stop_at_newline=True)
        assert first is not None
        assert tokenizer.decode(first[0]).endswith("\n")
        assert corpus.last_block_ended_at_newline is True
        state = corpus.state_dict()
        second = corpus.next_block(4096)

    assert second is not None
    assert tokenizer.decode(second[0]).startswith("second record")
    assert state["bytes_consumed"] == len(
        "first record ends here\n".encode("utf-8")
    )


def test_training_feedback_waits_for_a_source_newline(tmp_path):
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=4,
        lr_auto=False,
        decimation_gain=0.0,
        device="cpu",
    )
    model.trainer.pulse_enabled = False
    path = tmp_path / "feedback-records.txt"
    path.write_text(
        "one\n" * 64,
        encoding="utf-8",
    )
    boundaries = []

    def fake_generate_text(**_kwargs):
        reader = model._source_cursor["reader"]
        boundaries.append(
            int(reader["start_byte"]) + int(reader["bytes_consumed"])
        )
        return "test"

    model.generate_text = fake_generate_text
    model.train(
        path,
        save_path="",
        report_every=0,
        dream_every_batches=1,
        dream_length=1,
    )
    raw = path.read_bytes()
    assert boundaries
    assert all(position == len(raw) or raw[position - 1] == 0x0A
               for position in boundaries)


def test_training_feedback_fires_when_update_clock_crosses_boundary(tmp_path):
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=4,
        lr_auto=False,
        decimation_gain=0.0,
        device="cpu",
    )
    model.trainer.pulse_enabled = False
    model.trainer.updates = 1
    path = tmp_path / "feedback-crossing.txt"
    path.write_text(
        "one source record with enough material to update\n" * 8,
        encoding="utf-8",
    )
    original_train_batch = model.trainer.train_batch

    def train_batch_with_skipped_clock(*args, **kwargs):
        metrics = original_train_batch(*args, **kwargs)
        if metrics.updated:
            model.trainer.updates += 1
        return metrics

    model.trainer.train_batch = train_batch_with_skipped_clock
    dreams = []
    model.generate_text = lambda **_kwargs: "test"
    model.train(
        path,
        save_path="",
        report_every=0,
        dream_every_batches=4,
        dream_length=1,
        dream_callback=lambda *_args: dreams.append(model.trainer.updates),
    )
    assert dreams
    assert dreams[0] >= 4


def test_training_feedback_does_not_wait_forever_for_newline(tmp_path):
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=4,
        lr_auto=False,
        decimation_gain=0.0,
        device="cpu",
    )
    model.trainer.pulse_enabled = False
    path = tmp_path / "newline-free.txt"
    path.write_text(
        "a deliberately newline free source record " * 256,
        encoding="utf-8",
    )
    positions = []

    def fake_generate_text(**_kwargs):
        reader = model._source_cursor["reader"]
        positions.append(
            int(reader["start_byte"]) + int(reader["bytes_consumed"])
        )
        return "test"

    model.generate_text = fake_generate_text
    model.train(
        path,
        save_path="",
        report_every=0,
        dream_every_batches=1,
        dream_length=1,
    )
    assert positions
    assert positions[0] < path.stat().st_size


def test_streaming_corpus_masks_state_and_preserves_cursor(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    text = (
        "<|soma_state|>\nsource: chat\n<|turn|>\n"
        "<|user|>\nhello there\n<|soma|>\nhello james\n<|turn|>\n"
    )
    path = tmp_path / "dialogue.txt"
    path.write_text(text, encoding="utf-8")
    with TokenCorpus(path, tokenizer) as corpus:
        block = corpus.next_block(4096)
        assert block is not None
        ids, source_bytes, position = block
        assert len(ids) == len(corpus.last_loss_weights)
        assert 0.0 < corpus.last_loss_bytes < source_bytes
        assert position == len(text.encode("utf-8"))
        state_id = tokenizer.control_ids["soma_state"]
        state_at = int(np.flatnonzero(ids == state_id)[0])
        assert corpus.last_loss_weights[state_at] == 0.0


def test_streaming_corpus_resume_preserves_read_ahead_tail(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    path = tmp_path / "read-ahead.txt"
    path.write_text(
        ("one exact token tail after another, across every boundary. " * 80),
        encoding="utf-8",
    )
    with TokenCorpus(path, tokenizer) as corpus:
        first = corpus.next_block(7)
        assert first is not None
        state = corpus.state_dict()
        expected = corpus.next_block(19)
        expected_weights = corpus.last_loss_weights.copy()
        expected_loss_bytes = corpus.last_loss_bytes

    with TokenCorpus(path, tokenizer, resume_state=state) as resumed:
        actual = resumed.next_block(19)
        assert actual is not None
        assert np.array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
        assert np.array_equal(resumed.last_loss_weights, expected_weights)
        assert resumed.last_loss_bytes == pytest.approx(expected_loss_bytes)


def test_fully_masked_source_still_reaches_report_and_cursor_hooks(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size),
        tokenizer,
        device="cpu",
        optimizer_quantum=8,
    )
    model = soma_v13.SOMA._from_trainer(trainer, batch_size=8)
    path = tmp_path / "metadata.txt"
    path.write_text(
        "<|soma_state|>\n" + ("masked metadata only " * 200),
        encoding="utf-8",
    )
    reports = []
    model._report = lambda *_args: reports.append(1)

    model.train(path, save_path="", report_every=1)

    assert reports
    assert trainer.batches == 0
    assert model._source_cursor["reader"]["bytes_consumed"] == path.stat().st_size


def test_checkpoint_cursor_verifies_content_and_allows_append(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    path = tmp_path / "lineage.txt"
    path.write_text("stable source prefix " * 200, encoding="utf-8")
    with TokenCorpus(path, tokenizer) as corpus:
        assert corpus.next_block(11) is not None
        reader_state = soma_v13._sign_corpus_state(corpus.state_dict())
    model = soma_v13.SOMA.__new__(soma_v13.SOMA)
    model._source_cursor = {"kind": "corpus", "reader": reader_state}

    with path.open("a", encoding="utf-8") as handle:
        handle.write("new append is outside the represented boundary")
    assert model.resume_state_for(path) is not None

    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="content differs"):
        model.resume_state_for(path)


def test_checkpointed_reader_resumes_exact_training_continuation(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    config = tiny_config(tokenizer.vocab_size)
    uninterrupted = soma_v13.SOMA._from_trainer(
        TraceBranchTrainer(
            config,
            tokenizer,
            device="cpu",
            lr_auto=False,
            pulse_enabled=False,
            optimizer_quantum=8,
        ),
        batch_size=8,
    )
    interrupted = soma_v13.SOMA._from_trainer(
        TraceBranchTrainer(
            config,
            tokenizer,
            device="cpu",
            lr_auto=False,
            pulse_enabled=False,
            optimizer_quantum=8,
        ),
        batch_size=8,
    )
    for model in (uninterrupted, interrupted):
        model.lr_base = 3e-4
        model._set_optimizer_lrs()
    corpus_path = tmp_path / "continuation.txt"
    corpus_path.write_text(
        "exact continuation keeps every token and control event " * 100,
        encoding="utf-8",
    )

    uninterrupted.train(corpus_path, save_path="", report_every=0)

    def stop_after_first_block(*_args):
        raise KeyboardInterrupt

    interrupted._report = stop_after_first_block
    with pytest.raises(KeyboardInterrupt):
        interrupted.train(corpus_path, save_path="", report_every=1)
    checkpoint_path = tmp_path / "continuation.pt"
    interrupted.save(checkpoint_path)
    resumed = soma_v13.SOMA.from_checkpoint(checkpoint_path, device="cpu")
    resume_state = resumed.resume_state_for(corpus_path)
    assert resume_state is not None
    resumed.train(
        corpus_path,
        save_path="",
        report_every=0,
        resume_state=resume_state,
    )

    assert resumed.trainer.updates == uninterrupted.trainer.updates
    assert resumed.trainer.training_tokens_seen == (
        uninterrupted.trainer.training_tokens_seen
    )
    assert torch.equal(
        resumed.trainer.trace.trace,
        uninterrupted.trainer.trace.trace,
    )
    for left, right in zip(
        resumed.trainer.model.parameters(),
        uninterrupted.trainer.model.parameters(),
    ):
        assert torch.equal(left, right)


def test_loss_byte_denominator_is_independent_of_source_bytes():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size),
        tokenizer,
        device="cpu",
    )
    ids = tokenizer.encode("one measured present at a time")
    first = trainer.train_batch(ids, source_bytes=100, loss_bytes=50)
    assert first.updated is True
    assert trainer.updates == 1
    assert trainer.trace.source_bytes_seen == pytest.approx(100.0)
    assert first.bpb == pytest.approx(
        first.token_nll / (50.0 * np.log(2.0))
    )


def test_batch_clocked_update_interval_is_rejected():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    with pytest.raises(ValueError, match="no longer supported"):
        TraceBranchTrainer(
            tiny_config(tokenizer.vocab_size),
            tokenizer,
            device="cpu",
            update_interval=2,
        )


def test_cli_resume_hyperparameters_are_opt_in_overrides():
    parser = soma_v13._build_parser()
    args = parser.parse_args([
        "train", "corpus.txt", "--checkpoint", "model.pt"
    ])
    assert args.batch is None
    assert args.lr is None
    assert args.decimation is None
    assert args.lattice_rate is None
    assert args.grad_clip is None
    assert args.start_byte is None
    assert not hasattr(args, "update_interval")


def test_checkpoint_resume_preserves_exact_next_prediction(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size), tokenizer, device="cpu"
    )
    ids = tokenizer.encode("a checkpoint is a temporal state")
    trainer.train_batch(ids, source_bytes=31, loss_bytes=31)
    path = tmp_path / "v13.pt"
    trainer.save(path)
    expected = trainer.next_logits()
    resumed = TraceBranchTrainer.load(path, tokenizer, device="cpu")
    assert resumed.model.species == "soma_v14_adaptive_trace_branch_mixer"
    assert torch.equal(resumed.error_trace, trainer.error_trace)
    assert torch.allclose(resumed.next_logits(), expected, atol=0.0, rtol=0.0)


def test_soma_checkpoint_restore_reads_payload_once(tmp_path, monkeypatch):
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=16,
        device="cpu",
    )
    path = tmp_path / "single-read.pt"
    model.save(path)

    real_load = torch.load
    reads = []

    def counted_load(*args, **kwargs):
        reads.append(Path(args[0]).resolve())
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counted_load)
    restored = soma_v13.SOMA.from_checkpoint(path, device="cpu")

    assert restored.params() == model.params()
    assert reads == [path.resolve()]


def test_checkpoint_resume_preserves_sampled_continuation(tmp_path):
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=16,
        device="cpu",
    )
    model.ingest_prompt("sampling is part of lived continuation", online=False)
    model.generate_ids(length=7, temperature=0.9)
    corpus_path = tmp_path / "cursor-source.txt"
    corpus_path.write_text("checkpointed pending source " * 100, encoding="utf-8")
    with TokenCorpus(corpus_path, model.tokenizer) as corpus:
        assert corpus.next_block(9) is not None
        model._record_source_cursor(corpus, epoch=0, epochs=1)
    path = tmp_path / "sampled.pt"
    model.save(path)

    uninterrupted = model.generate_ids(length=16, temperature=0.9)
    resumed = soma_v13.SOMA.from_checkpoint(path, device="cpu")
    assert resumed.resume_state_for(corpus_path) is not None
    after_resume = resumed.generate_ids(length=16, temperature=0.9)

    assert after_resume == uninterrupted
    assert torch.equal(
        resumed.trainer.trace.trace, model.trainer.trace.trace
    )


def test_checkpoint_load_rejects_nonfinite_learned_state(tmp_path):
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=16,
        device="cpu",
    )
    path = tmp_path / "nonfinite.pt"
    model.save(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    first = next(iter(checkpoint["model"].values()))
    first.view(-1)[0] = float("nan")
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="non-finite"):
        soma_v13.SOMA.from_checkpoint(path, device="cpu")


def test_failed_save_does_not_advance_live_revision(tmp_path, monkeypatch):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    model = soma_v13.SOMA._from_trainer(
        TraceBranchTrainer(
            tiny_config(tokenizer.vocab_size), tokenizer, device="cpu"
        ),
        batch_size=8,
    )
    revision = model.revision_id

    def fail_save(*_args, **_kwargs):
        raise OSError("injected save failure")

    monkeypatch.setattr(torch, "save", fail_save)
    path = tmp_path / "failed.pt"
    with pytest.raises(OSError, match="injected"):
        model.save(path)

    assert model.revision_id == revision
    assert not path.exists()
    assert not Path(f"{path}.revision.json").exists()


def test_sidecar_exposes_architecture_neutral_ecosystem_contract(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    model = soma_v13.SOMA._from_trainer(
        TraceBranchTrainer(
            tiny_config(tokenizer.vocab_size), tokenizer, device="cpu"
        ),
        batch_size=8,
    )
    path = tmp_path / "ecosystem.pt"
    model.save(path)

    sidecar = json.loads(
        Path(f"{path}.meta.json").read_text(encoding="utf-8")
    )
    ecosystem = sidecar["ecosystem"]
    assert sidecar["sidecar_version"] == 4
    assert sidecar["species"] == model.species
    assert sidecar["runtime_id"] == "soma.v14"
    assert sidecar["checkpoint_format"] == "soma-v14-pytorch"
    assert sidecar["checkpoint_version"] == 14
    assert sidecar["tokenizer_sha256"] == tokenizer.sha256
    assert ecosystem["schema"] == "logos.checkpoint.v1"
    assert ecosystem["identity"]["checkpoint_id"] == model.checkpoint_id
    assert ecosystem["identity"]["revision_id"] == model.revision_id
    assert ecosystem["artifact"]["bytes"] == path.stat().st_size
    assert ecosystem["runtime"]["checkpoint_version"] == 14
    assert ecosystem["interface"]["tokenizer"]["sha256"] == tokenizer.sha256
    assert "training.continual" in ecosystem["capabilities"]


def test_stale_checkpoint_writer_must_fork(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    original = soma_v13.SOMA._from_trainer(
        TraceBranchTrainer(
            tiny_config(tokenizer.vocab_size), tokenizer, device="cpu"
        ),
        batch_size=8,
    )
    path = tmp_path / "owned.pt"
    original.save(path)
    first = soma_v13.SOMA.from_checkpoint(path, device="cpu")
    stale = soma_v13.SOMA.from_checkpoint(path, device="cpu")

    first.ingest_prompt("newer owned state", online=False)
    original_model_id = first.model_id
    root_lineage_id = first.root_lineage_id
    parent_revision = first.revision_id
    first.save(path)
    assert first.model_id == original_model_id
    assert first.parent_revision_id == parent_revision

    with pytest.raises(RuntimeError, match="save as a fork"):
        stale.save(path)
    pointer = soma_v13._read_json_dict(Path(f"{path}.revision.json"))
    assert pointer["state"] == "complete"
    assert pointer["revision_id"] == first.revision_id

    fork_parent = stale.revision_id
    fork_path = tmp_path / "fork.pt"
    stale.save_fork(fork_path)
    fork = soma_v13.SOMA.from_checkpoint(fork_path, device="cpu")
    assert fork.model_id != original_model_id
    assert fork.root_lineage_id == root_lineage_id
    assert fork.parent_revision_id == fork_parent

    pointer["state"] = "pending"
    soma_v13._write_json_atomic(Path(f"{path}.revision.json"), pointer)
    with pytest.raises(ValueError, match="publication is unfinished"):
        soma_v13.SOMA.from_checkpoint(path, device="cpu")


def test_frozen_evaluation_is_invariant_to_batch_partition():
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=16,
        device="cpu",
    )
    ids = model.tokenizer.encode(
        "the same fixed continuation must mean the same thing across "
        "arbitrary delivery partitions. " * 3
    )
    source_bytes = len(model.tokenizer.decode(ids).encode("utf-8"))
    reference = model.trainer.evaluate_ids(
        ids, source_bytes=source_bytes, batch_size=1
    )
    for batch_size in (3, 8, 17, len(ids)):
        candidate = model.trainer.evaluate_ids(
            ids, source_bytes=source_bytes, batch_size=batch_size
        )
        assert candidate.token_nll == pytest.approx(
            reference.token_nll, abs=2e-4, rel=2e-7
        )
        assert candidate.top1 == reference.top1


def test_weighted_evaluation_is_invariant_to_batch_partition():
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=16,
        device="cpu",
    )
    ids = model.tokenizer.encode(
        "weighted evaluation follows the training protocol " * 3
    )
    weights = np.ones(len(ids), dtype=np.float32)
    weights[::5] = 0.0
    weights[3::7] = 0.1
    source_bytes = len(model.tokenizer.decode(ids).encode("utf-8"))
    loss_bytes = source_bytes * float(weights.mean())
    reference = model.trainer.evaluate_ids(
        ids,
        source_bytes=source_bytes,
        batch_size=1,
        loss_weights=weights,
        loss_bytes=loss_bytes,
    )
    for batch_size in (3, 8, 17, len(ids)):
        candidate = model.trainer.evaluate_ids(
            ids,
            source_bytes=source_bytes,
            batch_size=batch_size,
            loss_weights=weights,
            loss_bytes=loss_bytes,
        )
        assert candidate.token_nll == pytest.approx(
            reference.token_nll, abs=2e-4, rel=2e-7
        )
        assert candidate.bpb == pytest.approx(
            reference.bpb, abs=2e-4, rel=2e-7
        )
        assert candidate.top1 == pytest.approx(reference.top1, abs=1e-8)


def test_corpus_evaluation_passes_protocol_weights_and_loss_bytes(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    model = soma_v13.SOMA._from_trainer(
        TraceBranchTrainer(
            tiny_config(tokenizer.vocab_size), tokenizer, device="cpu"
        ),
        batch_size=8,
    )
    corpus_path = tmp_path / "evaluation-dialogue.txt"
    corpus_path.write_text(
        "<|soma_state|>\nprivate state\n<|turn|>\n"
        "<|user|>\nhello\n<|soma|>\nhello back\n<|turn|>\n",
        encoding="utf-8",
    )
    captured = {}

    def capture(ids, **kwargs):
        captured["ids"] = ids
        captured.update(kwargs)
        return "measured"

    model.trainer.evaluate_ids = capture
    assert model.evaluate(corpus_path) == "measured"
    assert len(captured["ids"]) == len(captured["loss_weights"])
    assert float(np.min(captured["loss_weights"])) == 0.0
    assert 0.0 < captured["loss_bytes"] < captured["source_bytes"]


def test_generation_closes_the_turn_even_at_length_limit():
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=16,
        device="cpu",
    )
    model.ingest_prompt("hello", online=False)
    list(model.generate(length=2, temperature=1.0))
    assert model.trainer.trace.last_token == model.tokenizer.control_ids["turn"]


def test_source_delivery_is_bounded_as_decimation_rises():
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=1024,
        device="cpu",
    )
    model.trainer.prediction_stride = 1
    assert model._source_block_tokens() == 1024
    model.trainer.prediction_stride = 4
    assert model._source_block_tokens() == 4093
    model.trainer.prediction_stride = 16
    assert model._source_block_tokens() == soma_v13.MAX_SOURCE_BLOCK_TOKENS
    model.trainer.prediction_stride = 256
    assert model._source_block_tokens() == soma_v13.MAX_SOURCE_BLOCK_TOKENS

    model.batch_size = 16_384
    model.trainer.prediction_stride = 1
    assert model._source_block_tokens() == soma_v13.MAX_SOURCE_BLOCK_TOKENS


def test_short_boundary_block_survives_a_later_decimation_row():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size),
        tokenizer,
        device="cpu",
        pulse_enabled=False,
    )
    trainer.prediction_stride = 32
    trainer.training_tokens_seen = 1
    ids = tokenizer.encode("a\n")[:2]

    metrics = trainer.train_batch(ids, source_bytes=2, loss_bytes=2)

    assert metrics.tokens == len(ids)
    assert metrics.predictions == 1


def test_product_accumulates_short_prompts_to_one_prediction_quantum():
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=8,
        lr_auto=False,
        decimation_gain=0.0,
        device="cpu",
    )
    model.trainer.pulse_enabled = False
    ids = np.resize(
        model.tokenizer.encode("one canonical optimizer quantum"), 8
    )
    first = model.trainer.train_batch(
        ids[:3], source_bytes=3, loss_bytes=3
    )
    second = model.trainer.train_batch(
        ids[3:5], source_bytes=2, loss_bytes=2
    )
    third = model.trainer.train_batch(
        ids[5:8], source_bytes=3, loss_bytes=3
    )

    assert first.updated is False
    assert second.updated is False
    assert third.updated is True
    assert model.trainer.updates == 1
    assert model.trainer.accumulated_prediction_mass == 0.0


def test_canonical_update_is_invariant_to_source_partition():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    config = tiny_config(tokenizer.vocab_size)
    whole = TraceBranchTrainer(
        config,
        tokenizer,
        device="cpu",
        learning_rate=3e-4,
        lr_auto=False,
        decimation_gain=0.0,
        pulse_enabled=False,
        optimizer_quantum=8,
    )
    partitioned = TraceBranchTrainer(
        config,
        tokenizer,
        device="cpu",
        learning_rate=3e-4,
        lr_auto=False,
        decimation_gain=0.0,
        pulse_enabled=False,
        optimizer_quantum=8,
    )
    ids = np.resize(
        tokenizer.encode("partition invariant gradient quantum"), 8
    )

    whole.train_batch(ids, source_bytes=8, loss_bytes=8)
    partitioned.train_batch(ids[:3], source_bytes=3, loss_bytes=3)
    partitioned.train_batch(ids[3:5], source_bytes=2, loss_bytes=2)
    partitioned.train_batch(ids[5:], source_bytes=3, loss_bytes=3)

    assert whole.updates == partitioned.updates == 1
    for left, right in zip(
        whole.model.parameters(), partitioned.model.parameters()
    ):
        assert torch.allclose(left, right, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        whole.trace.trace, partitioned.trace.trace, atol=3e-14, rtol=3e-14
    )


def test_io2_control_quantum_is_invariant_to_source_partition():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    config = tiny_config(tokenizer.vocab_size)
    whole = TraceBranchTrainer(
        config,
        tokenizer,
        device="cpu",
        learning_rate=3e-4,
        decimation_gain=1.0,
        pulse_enabled=False,
        optimizer_quantum=8,
    )
    partitioned = TraceBranchTrainer(
        config,
        tokenizer,
        device="cpu",
        learning_rate=3e-4,
        decimation_gain=1.0,
        pulse_enabled=False,
        optimizer_quantum=8,
    )
    ids = np.resize(tokenizer.encode("one canonical control quantum"), 8)

    whole.train_batch(ids, source_bytes=8, loss_bytes=8)
    partitioned.train_batch(ids[:3], source_bytes=3, loss_bytes=3)
    assert partitioned.controller_prediction_mass == pytest.approx(3.0)
    partitioned.train_batch(ids[3:5], source_bytes=2, loss_bytes=2)
    partitioned.train_batch(ids[5:], source_bytes=3, loss_bytes=3)

    assert whole.controller_prediction_mass == 0.0
    assert partitioned.controller_prediction_mass == 0.0
    assert partitioned.io2_plasticity == pytest.approx(
        whole.io2_plasticity, rel=2e-6, abs=2e-6
    )
    assert partitioned.decimation_band == pytest.approx(
        whole.decimation_band, rel=2e-6, abs=2e-6
    )
    assert partitioned.learning_rate == pytest.approx(
        whole.learning_rate, rel=2e-6, abs=2e-9
    )


def test_pulse_decision_is_held_across_partitioned_quantum():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    config = tiny_config(tokenizer.vocab_size)
    whole = TraceBranchTrainer(
        config,
        tokenizer,
        device="cpu",
        learning_rate=3e-4,
        decimation_gain=0.0,
        pulse_enabled=True,
        optimizer_quantum=8,
        pulse_quantum=8,
    )
    partitioned = TraceBranchTrainer(
        config,
        tokenizer,
        device="cpu",
        learning_rate=3e-4,
        decimation_gain=0.0,
        pulse_enabled=True,
        optimizer_quantum=8,
        pulse_quantum=8,
    )
    for trainer in (whole, partitioned):
        trainer._pulse_frequency = lambda: 0.5
    ids = np.resize(tokenizer.encode("canonical pulse quantum"), 24)

    whole.train_batch(ids[:8], source_bytes=8, loss_bytes=8)
    whole.train_batch(ids[8:16], source_bytes=8, loss_bytes=8)
    whole.train_batch(ids[16:], source_bytes=8, loss_bytes=8)

    partitioned.train_batch(ids[:8], source_bytes=8, loss_bytes=8)
    skipped_a = partitioned.train_batch(
        ids[8:11], source_bytes=3, loss_bytes=3
    )
    assert skipped_a.updated is False
    assert partitioned.pulse_quantum_updates is False
    partitioned.train_batch(ids[11:16], source_bytes=5, loss_bytes=5)
    assert partitioned.pulse_quantum_updates is None
    partitioned.train_batch(ids[16:18], source_bytes=2, loss_bytes=2)
    partitioned.train_batch(ids[18:], source_bytes=6, loss_bytes=6)

    assert whole.updates == partitioned.updates == 2
    assert whole.pulse_phase == pytest.approx(partitioned.pulse_phase)
    for left, right in zip(
        whole.model.parameters(), partitioned.model.parameters()
    ):
        assert torch.allclose(left, right, atol=2e-6, rtol=2e-6)


def test_protocol_loss_weights_do_not_create_fractional_pulse_time():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size),
        tokenizer,
        device="cpu",
        learning_rate=3e-4,
        pulse_enabled=True,
        optimizer_quantum=8,
        pulse_quantum=8,
    )
    ids = np.resize(tokenizer.encode("protocol pulse time"), 16)
    weights = np.ones(8, dtype=np.float32)
    weights[0] = 0.0
    weights[-1] = 0.1

    trainer.train_batch(
        ids[:8], source_bytes=8, loss_bytes=7.1, loss_weights=weights
    )
    assert trainer.pulse_quantum_fill == pytest.approx(0.0)
    assert trainer.pulse_quantum_updates is None
    trainer.train_batch(
        ids[8:], source_bytes=8, loss_bytes=7.1, loss_weights=weights
    )
    assert trainer.pulse_quantum_fill == pytest.approx(0.0)


def test_protocol_loss_weights_do_not_fragment_optimizer_time():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size),
        tokenizer,
        device="cpu",
        learning_rate=3e-4,
        lr_auto=False,
        pulse_enabled=False,
        optimizer_quantum=8,
    )
    ids = np.resize(tokenizer.encode("protocol optimizer time"), 8)
    weights = np.ones(8, dtype=np.float32)
    weights[0] = 0.0
    weights[-1] = 0.1

    metrics = trainer.train_batch(
        ids, source_bytes=8, loss_bytes=7.1, loss_weights=weights
    )

    assert metrics.updated is True
    assert trainer.updates == 1
    assert trainer.accumulated_prediction_mass == pytest.approx(0.0)
    assert trainer.accumulated_loss_mass == pytest.approx(0.0)


def test_partial_gradient_quantum_survives_checkpoint(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    config = tiny_config(tokenizer.vocab_size)
    uninterrupted = TraceBranchTrainer(
        config,
        tokenizer,
        device="cpu",
        learning_rate=3e-4,
        lr_auto=False,
        decimation_gain=0.0,
        pulse_enabled=False,
        optimizer_quantum=8,
    )
    interrupted = TraceBranchTrainer(
        config,
        tokenizer,
        device="cpu",
        learning_rate=3e-4,
        lr_auto=False,
        decimation_gain=0.0,
        pulse_enabled=False,
        optimizer_quantum=8,
    )
    ids = np.resize(
        tokenizer.encode("checkpoint the unfinished gradient"), 8
    )
    uninterrupted.train_batch(ids[:4], source_bytes=4, loss_bytes=4)
    interrupted.train_batch(ids[:4], source_bytes=4, loss_bytes=4)
    path = tmp_path / "partial-gradient.pt"
    interrupted.save(path)
    resumed = TraceBranchTrainer.load(path, tokenizer, device="cpu")

    assert resumed.controller_prediction_mass == pytest.approx(4.0)
    assert resumed.controller_token_nll == pytest.approx(
        interrupted.controller_token_nll
    )

    uninterrupted.train_batch(ids[4:], source_bytes=4, loss_bytes=4)
    resumed.train_batch(ids[4:], source_bytes=4, loss_bytes=4)

    assert resumed.updates == uninterrupted.updates == 1
    for left, right in zip(
        resumed.model.parameters(), uninterrupted.model.parameters()
    ):
        assert torch.equal(left, right)
    assert resumed.io2_plasticity == pytest.approx(
        uninterrupted.io2_plasticity
    )


def test_deferred_batch_quantum_survives_checkpoint(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size),
        tokenizer,
        device="cpu",
        lr_auto=False,
        pulse_enabled=False,
        optimizer_quantum=8,
    )
    ids = np.resize(tokenizer.encode("deferred learning clock"), 8)
    trainer.train_batch(ids[:4], source_bytes=4, loss_bytes=4)
    trainer.request_training_quantum(16)

    path = tmp_path / "deferred-quantum.pt"
    trainer.save(path)
    resumed = TraceBranchTrainer.load(path, tokenizer, device="cpu")
    assert resumed.optimizer_quantum == pytest.approx(8.0)
    assert resumed.pending_optimizer_quantum == pytest.approx(16.0)

    resumed.train_batch(ids[4:], source_bytes=4, loss_bytes=4)
    assert resumed.updates == 1
    assert resumed.optimizer_quantum == pytest.approx(16.0)
    assert resumed.pending_optimizer_quantum is None


def test_repeated_learning_keeps_runtime_state_structurally_bounded():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    config = tiny_config(tokenizer.vocab_size)
    trainer = TraceBranchTrainer(config, tokenizer, device="cpu")
    ids = tokenizer.encode("bounded state can describe an indefinite stream")

    for _ in range(24):
        trainer.train_batch(ids, source_bytes=48, loss_bytes=48)

    assert trainer.trace.trace.shape == (config.vocab_size, config.bands)
    assert trainer.trace.prior_counts.shape == (config.vocab_size,)
    assert trainer.error_trace.shape == (config.bands,)
    assert trainer.error_bank.trace.shape == (
        config.bands + 1, config.symbol_dim
    )
    assert trainer.batches == trainer.updates == 24
    assert len(trainer.optimizer.state) == sum(
        1 for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    for state in trainer.optimizer.state.values():
        assert set(state).issuperset({"step", "exp_avg", "exp_avg_sq"})


def test_soft_radial_compression_only_attenuates_outward_growth():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size), tokenizer, device="cpu"
    )
    parameter = trainer.model.context.weight
    ceiling = trainer._row_ceiling(parameter)
    with torch.no_grad():
        parameter.fill_(0.0)
        parameter[0, 0] = 0.8 * ceiling
        previous = {
            id(parameter): torch.linalg.vector_norm(
                parameter.flatten(start_dim=1), dim=1, keepdim=True
            )
        }
        parameter[0, 0] = 0.9 * ceiling
        parameter[1, 0] = 0.5
        trainer._soft_radial_compress_(previous)

    row_norms = torch.linalg.vector_norm(
        parameter.detach().flatten(start_dim=1), dim=1
    )
    assert 0.8 * ceiling < float(row_norms[0]) < 0.9 * ceiling
    assert float(row_norms[1]) == pytest.approx(0.5)


def test_soft_radial_compression_never_snaps_an_existing_row_to_ceiling():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size), tokenizer, device="cpu"
    )
    parameter = trainer.model.context.weight
    with torch.no_grad():
        parameter[0].zero_()
        parameter[0, 0] = 20.0
        previous = {
            id(parameter): torch.linalg.vector_norm(
                parameter.flatten(start_dim=1), dim=1, keepdim=True
            )
        }
        trainer._soft_radial_compress_(previous)
    assert float(parameter[0].detach().norm()) == pytest.approx(20.0)


def test_radial_compression_bounds_new_growth_and_legacy_inflation():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size), tokenizer, device="cpu"
    )
    parameter = trainer.model.context.weight
    ceiling = trainer._row_ceiling(parameter)

    with torch.no_grad():
        parameter.zero_()
        parameter[0, 0] = 0.99 * ceiling
        previous = {
            id(parameter): torch.linalg.vector_norm(
                parameter.flatten(start_dim=1), dim=1, keepdim=True
            )
        }
        parameter[0, 0] = 1.5 * ceiling
        trainer._soft_radial_compress_(previous)
        assert float(parameter[0].norm()) == pytest.approx(ceiling)

        parameter[0, 0] = 2.0 * ceiling
        previous = {
            id(parameter): torch.linalg.vector_norm(
                parameter.flatten(start_dim=1), dim=1, keepdim=True
            )
        }
        parameter[0, 0] = 3.0 * ceiling
        trainer._soft_radial_compress_(previous)
        assert float(parameter[0].norm()) == pytest.approx(2.0 * ceiling)


def test_scalar_residual_gates_are_stored_inside_finite_tanh_rails():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size), tokenizer, device="cpu"
    )
    gates = [
        parameter
        for parameter in trainer.model.parameters()
        if parameter.ndim == 0 and parameter.requires_grad
    ]
    assert gates
    with torch.no_grad():
        for index, gate in enumerate(gates):
            gate.fill_(100.0 if index % 2 == 0 else -100.0)
        trainer._bound_scalar_gates_()
    assert all(abs(float(gate.detach())) <= 8.0 for gate in gates)


def test_io2_scales_lattice_mobility_without_changing_its_base_rate():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size),
        tokenizer,
        device="cpu",
        lattice_rate=0.02,
        lr_auto=True,
    )
    trainer.io2_plasticity = 0.25
    assert trainer.effective_lattice_rate() == pytest.approx(0.005)
    assert trainer.lattice_rate == pytest.approx(0.02)

    trainer.lr_auto = False
    assert trainer.effective_lattice_rate() == pytest.approx(0.02)


def test_weight_diagnostics_are_finite_and_checkpoint_stable(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size), tokenizer, device="cpu"
    )
    diagnostics = trainer.weight_diagnostics()
    assert 0.0 <= diagnostics["radial_pressure"] <= 1.0
    assert diagnostics["row_spread"] >= 0.0

    path = tmp_path / "radial.pt"
    trainer.save(path)
    resumed = TraceBranchTrainer.load(path, tokenizer, device="cpu")
    assert resumed.radial_compression is True
    assert resumed.radial_ceiling_scale == pytest.approx(0.1)
    assert resumed.radial_floor == pytest.approx(0.01)
    assert resumed.pulse_frequency_exponent == pytest.approx(0.5)


def test_io2_pulse_factors_plasticity_between_time_and_amplitude():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size),
        tokenizer,
        device="cpu",
        pulse_enabled=True,
        pulse_quantum=100,
        pulse_schedule="factor",
    )
    trainer.io2_plasticity = 0.25
    trainer.updates = 1
    trainer.batches = 2
    assert trainer._pulse_frequency() == pytest.approx(0.5)
    assert trainer._schedule_update(100) is False
    trainer._commit_pulse_mass(100)
    trainer.batches += 1
    assert trainer._schedule_update(100) is True

    trainer._io2_ratio = lambda _loss: 0.25
    trainer._apply_io2(1.0, 1.0)
    assert trainer.learning_rate == pytest.approx(
        trainer.learning_rate_base * 0.5
    )


def test_spectral_pulse_uses_residual_clock_and_measured_compute_cost():
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size),
        tokenizer,
        device="cpu",
        pulse_enabled=True,
        pulse_schedule="spectral",
    )
    trainer.io2_plasticity = 0.25
    trainer.error_target_band = 2.0
    trainer.error_coherence = 1.0
    trainer.update_cost_fraction = 1.0
    expected = max(0.25, trainer.config.phi ** -2.0)
    assert trainer._pulse_frequency() == pytest.approx(expected)

    trainer.update_cost_fraction = 0.0
    assert trainer._pulse_frequency() == pytest.approx(1.0)


def test_cockpit_allocation_spectrum_reads_learned_temporal_weights():
    model = soma_v13.SOMA(
        n_bands=8,
        profile="small",
        batch_size=16,
        device="cpu",
    )
    temporal_width = model.config.summaries * model.config.symbol_dim
    with torch.no_grad():
        model.net.context.weight[:, :temporal_width].zero_()
        last = model.config.summaries - 1
        start = last * model.config.symbol_dim
        end = start + model.config.symbol_dim
        model.net.context.weight[:, start:end].fill_(1.0)
    model._update_diagnostics()
    assert len(model._weight_spectral_drive) == model.n_bands
    assert int(model._weight_spectral_drive.argmax()) == model.n_bands - 1
    assert float(model._weight_spectral_drive.mean()) == pytest.approx(1.0)


def test_pre_controller_v13_checkpoint_starts_a_cold_translated_bank(tmp_path):
    tokenizer = ReferenceTokenizer(TOKENIZER_PATH)
    trainer = TraceBranchTrainer(
        tiny_config(tokenizer.vocab_size), tokenizer, device="cpu"
    )
    ids = tokenizer.encode("old checkpoints retain their learned present")
    trainer.train_batch(ids, source_bytes=44, loss_bytes=44)
    checkpoint = trainer.checkpoint_dict()
    checkpoint.pop("translated_error_bank")
    for key in (
        "learning_rate_base",
        "lr_auto",
        "decimation_gain",
        "decimation_stride_cap",
        "decimation_band",
        "prediction_stride",
        "io2_plasticity",
        "last_error_token",
        "training_tokens_seen",
    ):
        checkpoint["training"].pop(key, None)
    checkpoint["lr_auto"] = False
    checkpoint["batch_size"] = 16
    checkpoint["training"]["learning_rate"] = 3e-4
    path = tmp_path / "v13_0.pt"
    torch.save(checkpoint, path)

    resumed = TraceBranchTrainer.load(path, tokenizer, device="cpu")
    assert torch.count_nonzero(resumed.error_bank.trace) == 0
    assert resumed.last_error_token == 0
    assert resumed.training_tokens_seen == 0
    assert resumed.lr_auto is True
    assert resumed.prediction_stride == 1

    wrapped = soma_v13.SOMA.from_checkpoint(path, device="cpu")
    assert wrapped.lr_auto is True
    assert wrapped.lr_base == pytest.approx(3e-4)
