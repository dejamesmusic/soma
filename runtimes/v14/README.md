# soma v14

soma is a local continual language learner. it turns an indefinite token
stream into a fixed-size present, predicts the next token, and can keep
learning without replaying a context window or differentiating through source
time.

v14 retains the v13 learner and makes its temporal instrument adaptive. it
still separates three jobs that earlier builds made one set of weights perform:

1. an exact, nonlearned trace bank remembers which symbols occurred and at
   which geometric timescales;
2. a trace branch mixer organizes that temporal field into a compact present;
3. a serial residual trunk composes the present into a prediction.

the result is a 65.51m parameter default model with a 1,024-token dictionary,
48 exact temporal bands, and a fixed learned shape. phi defines the initial
receiver. persistent io2 residual density may slowly redistribute its interior
band coordinates while the fastest band and total temporal horizon stay fixed.

## the present

for token identity `i` and temporal band `k`, the trace bank stores:

```text
beta_k = phi^-c_k
r_k    = exp(-beta_k)
q_k    = 1 - r_k

t_i,k(n + 1) = r_k t_i,k(n) + q_k [token_n = i]
```

the state is positive, bounded, updated exactly in float64 on cpu, and has no
learned parameters. every band is a different measurement of the same stream.
the slowest lifetime remains `phi^(bands - 1)` tokens. `c_k = k` is the phi
prior; conservative interior motion changes resolution, not horizon.

## the learner

the token identities share a learned coordinate. the runtime projects the
whole trace field through that dictionary, centres every selected pre-token
state against its exact empirical prefix prior, and retains the previous token
as a direct fast channel. the same frozen model therefore sees the same
represented continuation regardless of delivery block boundaries.

three shared scale blocks then mix locally across log-time and through a
96-dimensional symbol bottleneck. twelve fixed gaussian summaries convert the
variable temporal lens into a fixed-size representation. a 1,536-dimensional
context and four residual swiglu blocks perform serial composition. the final
384-dimensional branch is read through the same dictionary used at input,
with the empirical log-prior added to the logits.

training uses ordinary next-token cross entropy and adamw. the exact trace is
committed after each contiguous source block; gradients remain entirely inside
the selected projected presents. generated text changes lived trace state but
does not train the model or alter its source prior. user text can train online.
masked context, controls, and generated text still advance the same lived-time
clock and age controller state without creating residual observations.
weighted prediction gradients accumulate to one configured-batch quantum before
adamw advances, so a short prompt, final fragment, or high-stride source block
cannot masquerade as a full optimizer batch. unfinished accumulation is part of
the checkpoint.

v13.1 translates the signed categorical residual into the same 256-dimensional
symbol coordinates as the model and stores it across `bands + 1` geometric
timescales. io2 reads the residual's energy and concentration to regulate
learning rate and prediction density. the expensive network may inspect every
second, third, or later present while the exact trace continues to ingest every
token. this is compute decimation, not memory decimation.

v13.4 keeps one global prediction phase across source blocks and treats sampled
residuals as causal observations at their actual times. per-band variance and
bandpass calibration keep the controller comparable across stride without
inventing error during skipped intervals. lr and decimation act only after a
canonical quantum of supervised prediction mass, so host block boundaries do
not become controller events.

batch size remains a freely adjustable hardware control. when changed in the
middle of a saved learning quantum, delivery adopts the new limit immediately
while the optimizer and controller finish the old quantum before switching
clocks. the deferred boundary survives save and resume.

v13.1.1 computes those selected presents directly as exact affine blocks. it
never allocates the intermediate projected states that decimation discards.
source delivery is capped at 8,192 tokens per batch, so increasing stride
eventually reduces the prediction batch instead of increasing batch latency or
memory without bound.

v13.2 hardens indefinite learning with a radial compressor. outward row growth
is smoothly attenuated near a fan-in-aware reference radius, then bounded at
that radius while inward movement and changes of direction remain available.
existing rows beyond the radius are never snapped back or globally
renormalized, but cannot inflate farther, so earlier checkpoints resume without
a representational shock.

the live spectrum now shows exact input energy, learned temporal allocation
from the context projection, and translated residual energy. diagnostics are
evaluated only when a visible report is emitted. the residual-clock pulse
scheduler is enabled for new models: it spends full updates where residual
evidence justifies them and preserves dense trace observation between pulses.
its phase and timing state are checkpointed.
one pulse decision is held across every fragment of a canonical supervised-mass
quantum; source partitioning cannot turn a skipped or updated quantum into a
mixture of both. the partial pulse decision and fill are checkpointed.

held-out evaluation is always dense and uses the same protocol mask, weighted
turn objective, empirical-prior rule, and estimated supervised-byte denominator
as training. live training bpb is labelled sampled whenever prediction stride
is greater than one, together with its selected prediction coverage.

## defaults

```text
dictionary             1,024 tokens fixed
bands                  48
base                   phi
capacity               medium
symbol coordinate      384 derived
scale rank             96 derived
scale depth            3 fixed
temporal summaries     12 derived
model width            1,536 derived
composition depth      4 fixed
composition expansion  3,072 derived
parameters             65,508,870
batch                   automatic (medium: 512 predictions)
optimizer quantum       configured automatic batch
pulse quantum           configured batch
learning rate           auto 0.0003
lr reference batch      1,024 predictions
controller              io2
decimation gain         1.0
prediction stride cap   256
gradient clip           1.0
radial compression      soft, enabled
pulse scheduling        spectral, enabled
dream/self-context      off
```

the gui opens in managed mode. a new model asks only for checkpoint identity,
capacity, source, and passes. small is 28.2m parameters, medium is 65.5m, and
large is 93.8m at the managed 48-band lens. phi, learning rate, decimation,
clipping, batch, source cursor, autosave, and dream policy are materialised as
safe settings when training begins; they are not merely hidden stale fields.

an existing checkpoint offers an explicit `continue` or `fork` action.
continue preserves its model id and writes only to the selected artifact. fork
creates a new model id and records the selected revision as its parent. a new
or unrelated checkpoint never overwrites an existing file: the gui assigns a
collision-free name. while a checkpoint is being inspected, training remains
disabled so stale form state cannot be launched.

expert settings reveal temporal bands and base, batch, source cursor,
self-context scheduling, learning rate, decimation gain, gradient clipping,
autosave cadence, and dream temperature. returning to managed mode cancels
hidden self-context overrides. for a resumed production profile, its learning
policy is restored while batch is fitted to the current device.

the managed live view shows estimated bits per source byte with “lower is
better”, whether next-token accuracy is dense or sampled, prediction coverage,
source speed, this-run experience, pass, and current next-token candidates.
`show diagnostics` reveals the learning/controller clocks and the accurately
named trace activity, context-readout allocation, controller residual, and
slow-trace-deviation instruments. sampled values are never presented as dense
measurements.

the entered learning rate is the canonical rate at a 1,024-prediction batch.
the applied rate scales linearly with the configured batch, preserving the
expected source-time plasticity budget when memory requires a smaller batch.

automatic batch is capacity- and device-aware. the current conservative mps
policy is calibrated from isolated production updates with a 15% admission
margin and compares against pytorch's reported recommended memory envelope.
on a 5.73 gb mps envelope it selects 1,024 predictions for small and 512 for
medium or large. the gui labels manual choices as comfortable, pressure, or
unsupported; unsupported shapes do not begin allocation.

## long-run cockpit

each gui training session owns a durable run id and a record under
`state/runs/`: configuration, manifest, lifecycle events, stop request, and a
60-second scalar metrics history. the manifest carries the exact worker pid,
status, heartbeat, latest report, last safe checkpoint time, source,
destination, and any persistent warning. bulky spectra remain live-only.

the setup also accepts a separate validation file. a file whose name contains
`validation`, `heldout`, or `eval` is suggested automatically but remains
optional. managed cadence runs a dense, state-preserving 250 kb continuation
at baseline and every 100 mb of new source experience. validation is removed
from an all-files training cycle, so the cockpit cannot silently train on the
held-out file. cadence and sample size are expert overrides.

a fresh model publishes its first checkpoint before consuming source data.
autosave failures retry within one minute. the cockpit distinguishes live,
busy, checkpoint publication, stopping, stalled, interrupted, and failed
states; it displays checkpoint age, current-pass time remaining, and free
storage, plus the latest dense held-out bpb and exact-token accuracy. validation
has its own visible run state and never masquerades as a training stall. launch
is refused when the destination lacks a conservative atomic-save reserve.

closing the cockpit may leave training running. reopening soma verifies the
exact worker command and pid, restores the latest durable report, tails future
lifecycle events, and controls the same stop channel. if the worker disappeared,
the run is marked interrupted and the app directs the user to the last
published checkpoint. an active training checkpoint can be opened in chat only
as a disposable snapshot; chat cannot write into the trainer-owned lineage.
`run data` reveals the metrics history used for long-horizon analysis.

## use

```bash
./soma gui
./soma train corpus.txt --save model.pt
./soma chat model.pt --online
./soma generate model.pt --length 256
./soma info model.pt
./soma test
```

bare corpus names resolve under `data/`; bare checkpoint names resolve under
`checkpoints/`. the launcher creates its environment and installs dependencies
on first use. the mac app is already self-contained.

## checkpoint boundary

```text
species            soma_v14_adaptive_trace_branch_mixer
checkpoint version 14
runtime version    14.0.0-dev.2
tokenizer          soma-1k-v1
```

v14 checkpoints are intentionally incompatible with pre-v13 soma species.
they persist model and optimizer state, exact trace and source prior, signed
translated residual bank, io2 state, soft radial and pulse state, immutable
tokenizer identity, counters, and checkpoint/revision ids. a human-readable
`.pt.meta.json` sidecar is
written beside every save. v13.0 checkpoints load without conversion; their
new residual controller begins from an empty bank while learned state survives.
v13.4 checkpoints also preserve the model-owned sampling generator, so
stochastic continuation survives save and resume. all runtime checkpoint loads
use pytorch's tensor-only loader. a corpus checkpoint also owns the exact
reader state: raw read-ahead boundary, pending token tail and weights, protocol
mode, byte ownership, source limit, and a content signature. automatic resume
fails closed if the represented corpus prefix changed.

public saves use an inter-process artifact lock and a small revision pointer.
the pointer marks publication pending before replacement and complete only
after the checkpoint is durable at its final path. a stale in-memory copy
cannot overwrite a newer revision; it must be saved under a new lineage path.
continuations keep `model_id`; every saved state records its
`parent_revision_id` and stable `root_lineage_id`. `save_fork` creates a new
model id while retaining the exact parent and root provenance. `checkpoint_id`
remains as the backward-compatible spelling of `model_id`.

## evidence boundary

the production path has passed causal scan, shape, protocol masking,
checkpoint identity, exact-resume, generation-boundary, worker, and mps smoke
tests. in a short development smoke on a 2024 m3 macbook air, io2 prediction
decimation more than doubled immediate source throughput over a dense control
at the same default shape while preserving short-run bpb. this is an
implementation check, not a convergence or comparative efficiency claim.

long-duration convergence and broad held-out language quality remain the v14
experiment. the architecture is now stable enough to run that experiment
without changing its meaning underneath the checkpoint.

## files

- `soma_v14.py` — production runtime and cli
- `trace_branch_mixer/` — exact state, model, protocol, and trainer
- `soma_gui.py` — local training and chat cockpit
- `soma_train_worker.py` — isolated gui training process
- `soma_v14_spec.md` — normative architecture and checkpoint contract
- `logOS_v14_web_handover.md` — upload/parser handover
- `theory/` and `engineering/` — theoretical and implementation background
- `tests/test_soma_v14.py` and `tests/test_trace_branch_mixer.py` — invariants
