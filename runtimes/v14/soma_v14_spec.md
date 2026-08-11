# soma v14 normative specification

## 1. identity

```text
species            soma_v14_adaptive_trace_branch_mixer
checkpoint_version 14
runtime_version    14.0.0-dev.2
tokenizer          soma-1k-v1
vocabulary         1,024
```

v14 accepts frozen v13 trace-branch checkpoints as ancestors. every other
unknown species, checkpoint version, tokenizer hash, lattice, or direct-channel
convention must be rejected.

## 2. source protocol

text is encoded by the immutable sentencepiece model shipped as
`tokenizer/soma-1k-v1.model`. the vocabulary contains private control pieces:

```text
<|soma_state|>  metadata or state block; excluded from supervised loss
<|user|>        user turn begins
<|soma|>        soma turn begins
<|dream|>       autonomous generation begins
<|turn|>        turn boundary and generation stop token
```

ordinary language tokens are supervised. control pieces and state-block tokens
are masked, except an assistant `<|turn|>` boundary carries weight `0.1`.
masked content still occupies lived trace time. the linguistic prior counts
supervised natural-language tokens only; controls, state metadata, and generated
tokens do not alter it. the current supervised-byte denominator is apportioned
from token weights and must be treated as an estimate until byte ownership is
carried directly through tokenization.

## 3. exact temporal state

let `v` be vocabulary size, `k` the number of bands, `x_n` the token at time
`n`, and `phi = (1 + sqrt(5))/2`. every band has a controller-owned log-time
coordinate `c_j`. the phi prior is `c_j = j`; for the current coordinate:

```text
beta_j = beta_0 phi^-c_j, beta_0 = 1
r_j    = exp(-beta_j)
q_j    = -expm1(-beta_j) = 1 - r_j

t_i,j(n + 1) = r_j t_i,j(n) + q_j [x_n = i]
```

`t` has shape `[v,k]`, float64 cpu storage, and no optimizer parameters. it is
initialized from the smoothed empirical prior. positive prior counts are also
stored in float64. the current lifetime is `1/beta_j = phi^c_j`.

for a contiguous block, the implementation must produce every pre-token state
causally and commit the mathematically exact final state only after the model
step succeeds. the associative affine scan is an optimization of this equation,
not a different state transition.

### 3.1 adaptive temporal lattice

the lattice is a conservative adaptive mesh on log time. its endpoints are
structural:

```text
c_0     = 0
c_(k-1) = k - 1
0.25 <= c_(j+1) - c_j <= 4
```

adaptation redistributes a fixed temporal horizon; it cannot create longer
memory as an accidental side effect. the hidden extra error interval must also
remain above float64 decay resolution.

at each io2 prediction quantum, adjacent error-bank amplitude `a_j` is divided
by its current log-time gap `g_j` and normalized to mean one. a checkpointed
monitor `m_j`, initialized to one, follows an exponential 10,240-token evidence
horizon. this separates persistent residual density from a transient spectral
peak without making source partitioning part of the dynamics.

for interior coordinate `i`, the proposed motion is:

```text
drive_i = tanh(0.5 (log m_i - log m_(i-1)))
clock_i = (elapsed_tokens / 1024) min(1, 1024 beta_i)
mobility = lattice_rate plasticity_amplitude(io2)
v_i     = mobility clock_i
          (drive_i + lattice_elastic clamp(i - c_i, -1, 1))
```

the endpoints have zero velocity. one shared feasibility multiplier scales all
`v_i` only when necessary to preserve every gap and the resolution wall. this
keeps motion local, preserves its direction, and prevents any coordinate from
moving faster than its own evidence integrates. the fresh-model gui rate is
`0.002`; this is maximum mobility, not a second independent learning rate.
automatic io2 plasticity scales it in the same register as model learning.
selecting an existing checkpoint restores its saved rate.

the trace is bit-exact under the lattice trajectory it has lived through. a
deformation does not pretend that old evidence was accumulated under the new
filter: old mass is carried forward and its transition bias decays on the
band's own clock. `lattice_rate = 0` is the exact v13 receiver.

## 4. current symbol field

let `e_i in r^d` be the learned token coordinate, `p_i(n)` the empirical token
prior before token `n`, and `e_bar(n) = sum_i p_i(n) e_i`. the centred projected
trace is:

```text
z_j(n) = sum_i (t_i,j(n) - p_i(n)) e_i    shape [k,d]
u_n    = e_(x_(n-1)) - e_bar(n)           shape [d]
```

`z` is temporal evidence; `u` is the direct previous-token channel. the batch
scan advances the uncentred projected trace under the same `r_j,q_j` lattice,
then centres every selected pre-token state against its exact prefix prior.
delivery block boundaries therefore cannot freeze or reset the prior seen by
the learner.

## 5. scale encoder

three shared scale blocks operate on `[k,d]`. each block contains:

1. a depthwise width-five convolution along log lifetime;
2. a low-rank symbol mixer `d -> 64 -> d`;
3. learned conditioning from normalized log lifetime and both boundaries;
4. zero-started residual branches with scalar tanh gates.

the low-rank branch uses inverse-rms coordinates while restoring original
magnitude after mixing:

```text
s  = rsqrt(mean(z^2) + epsilon)
z' = z + tanh(g) up(silu(down(z s))) / s
```

this is not static whitening: magnitude remains available to the output.

a depthwise width-three summary filter is followed by eight fixed normalized
gaussian windows across the band axis. the result has shape `[8,256]`.

## 6. composition and readout

the eight summaries and direct channel are concatenated and projected:

```text
h_0 = context([flatten(summary), u])       shape [1,024]
```

four serial residual swiglu blocks compose the present:

```text
s_l     = rsqrt(mean(h_l^2) + epsilon)
b_l     = out_l(silu(gate_l(h_l s_l)) * value_l(h_l s_l)) / s_l
h_(l+1) = h_l + b_l / sqrt(4)
```

all linear maps are bias-free. the branch projection is zero-initialized:

```text
c      = branch(h_4)                       shape [256]
logit_i = <c,e_i>/sqrt(256) + log(max(p_i,epsilon))
```

the initial model therefore begins at the empirical token prior. input and
output dictionaries are tied.

## 7. learning and compute allocation

the supervised objective is weighted next-token cross entropy. the default
optimizer is adamw:

```text
learning rate  auto 3e-4
weight decay   1e-4
gradient clip  1.0 global norm
```

there is no gradient through source time or through the exact trace update.
the exact trace receives every source token. io2 may select every `s`th
pre-token state for the expensive model forward, objective, and backward pass.
the corpus reader supplies up to `batch_size * s` source tokens, capped at
8,192 tokens per source block. before that cap, the selected model batch stays
near `batch_size`; above it, higher decimation reduces the number of expensive
prediction sites. regularly selected presents are computed directly as affine
stride blocks, so skipped `[band, symbol]` states are never materialized.
weighted mean loss on selected positions estimates full-stream nll for bpb
accounting. whenever stride is greater than one this is a sampled estimate,
not an exact dense score.

the configured batch is also the canonical weighted prediction quantum for one
adamw step. selected token gradients accumulate as unnormalized sums until that
mass is reached, then are divided by the actual accumulated mass exactly once
before clipping and stepping. short prompts, masked spans, final fragments, and
source-cap-limited high strides therefore do not create undersized adam steps.
an unfinished quantum, its captured learning rate, and its parameter gradients
are checkpoint state. source delivery requests only the remaining prediction
mass wherever possible.

batch size may change at any stop point. it changes the hardware delivery limit
immediately. if a learning quantum is unfinished, its original optimizer,
controller, and pulse boundary remains authoritative until completion; the new
quantum is adopted at the next clean boundary. that deferred transition is
checkpoint state, so changing hardware shape cannot discard gradients or make
stopping position part of the learning rule.

io2 accumulates selected nll, supervised token mass, prediction mass, and loss
bytes over the same canonical quantum. learning-rate and decimation actuation
therefore occur on supervised prediction boundaries rather than once per host
batch. the accumulated controller totals are checkpoint state.

batch-clocked `update_interval` skipping is not part of v14. it is rejected
rather than allowing source partitioning to select a different learner. io2
pulse scheduling is the sole update-frequency controller.

after each optimizer step, the production runtime applies a soft radial
compressor to every learned matrix row. for fan-in `f`:

```text
ceiling       = max(1, 0.1 sqrt(f))
ratio         = previous_row_norm / ceiling
outward_gain  = max(0.01, 1 / (1 + ratio^4))
```

only outward row-norm change is multiplied by `outward_gain`. an existing row
below the reference ceiling cannot cross it. an existing checkpoint row above
the ceiling is not clipped or renormalized, but cannot grow farther. inward
changes and angular motion remain available, and adamw weight decay remains
active. scalar residual gates are passed through `tanh`, so their functional
gain remains in `[-1, 1]`; their raw stored values are additionally clamped to
`[-8, 8]`, already inside the numerically saturated regime.

generated tokens update trace state but not weights or prior counts. a user
prompt may update both when online learning is enabled. generation terminates
on `<|turn|>`; if the length limit is reached, the runtime commits that boundary
itself so every turn remains structurally closed.

automatic dream/self-context generation is off by default because it mutates
the authoritative lived trace. when explicitly enabled, its cadence is counted
in completed optimizer updates rather than source delivery batches, and its
model-owned sampling state is checkpointed.

evaluation predicts every token position and applies the same protocol loss
weights and linguistic-prior exclusions as training. its bpb denominator uses
the corpus reader's weighted allocation of segment bytes; this is consistent
with training but remains an estimate because sentencepiece does not expose an
exact byte span for every normalized token. decimated live training metrics
must be labelled sampled and report selected prediction coverage.

the managed live surface labels bpb as an estimated source-byte measure and
states that lower is better. top-1 and top-5 labels change to `sampled` whenever
stride exceeds one. the default readout is limited to loss, accuracy, coverage,
source speed, this-run experience, pass, and next-token candidates. optional
diagnostics expose learning rate, controller drive, prediction spacing, update
duty and fill, row pressure/spread, and prediction-error energy. the spectral
instrument is labelled `trace activity`, `context readout allocation`, and
`controller residual`; slow-band token deviations are not called memory.

## 8. translated residual bank and io2

for each selected prediction, the categorical residual is translated through
the tied token dictionary:

```text
p       = softmax(logits)
expected = sum_i p_i e_i
observed = e_target
c       = (observed - expected) / mean_i ||e_i||
```

the signed correction advances a float64 geometric bank with
`k+1` bands on the source-token clock. adjacent differencing produces the `k`
visible residual bands. the hidden extra-slow band prevents the final visible
band from becoming a forced low-pass remainder.

sampled corrections are causal point observations: elapsed source time decays
the bank before the current residual is injected. a new residual is never held
backward across the skipped interval. each band receives the stationary-variance
gain implied by the current stride, and adjacent-band energy is calibrated to
the dense white-residual baseline. prediction rows retain one global phase from
the lived-token clock instead of restarting at every source block. frequencies
above the sampled nyquist limit remain unrecoverable; their aliasing therefore
pushes the controller conservatively toward denser observation rather than
manufacturing slow residual structure.

let `a_j` be the l2 energy of visible residual interval `j`,
`pi_j = a_j / sum a`, and `u_j = (c_j + c_(j+1))/2 - 1/2`. `u_j` equals
the legacy integer index at the phi prior, but remains a physical log-time
coordinate after deformation. io2 derives:

```text
error energy      = sum(a) / (sum(a) + sqrt(k))
error coherence   = 1 - h(pi) / log(k)
error band        = sum_j u_j pi_j
plasticity        = clipped function(error energy, coherence, current loss)
effective lr      = base lr * plasticity
```

the default base learning rate is `3e-4`. plasticity is bounded to `[0.02,1]`.
loss relative to chance may scale the raw plasticity by at most `phi` in either
direction.

io2 also maps residual timescale and mastery into a continuous decimation band.
the realized prediction stride is `round(phi^band)`, capped at 256. decimation
gain scales the target band and defaults to one. the controller may rise by at
most one band or fall by at most two bands per canonical prediction quantum,
scaled by the actual accumulated mass. this allows rapid attention without
making host delivery fragments into control events. because trace advancement
stays dense, decimation spends less compute without deleting events from
memory.

the pulse scheduler may factor total io2 plasticity `p`
between optimizer frequency `f` and update amplitude `p/f`. spectral mode uses:

```text
characteristic = phi^(-error_band)
f = max(p, 1 - update_cost * error_coherence * (1 - characteristic))
```

`update_cost` is an ema estimate of the fraction of an updated batch spent
beyond observation-only work. the pulse and optimizer quanta follow the
configured weighted-prediction batch. residual time advances with every token
that changes the lived trace, including masked context, controls, and generated
tokens; those spans age residual state without inventing a residual
observation. actuation waits for canonical supervised mass.
phase, timing emas, and schedule are checkpointed. spectral pulse scheduling
is enabled for new models because the compute saving can be reinvested in model
capacity; resumed checkpoints preserve their saved choice.
the update/skip decision is made once at the start of a canonical supervised
prediction quantum and held across all of its source fragments. pulse fill and
the active decision are checkpointed, so fragmenting or interrupting a quantum
cannot change which targets receive gradients.

## 9. default shape

```text
bands                 48
base                  phi
capacity              medium
symbol_dim            384
scale_rank            96
scale_depth           3
scale_kernel          5
summaries             12
model_dim             1,536
trunk_depth           4
expansion_dim         3,072
parameters            65,508,870
batch_size            automatic (medium: 512 on the reference mps envelope)
pulse_quantum         batch_size
learning_rate         auto 3e-4
lr_reference_batch    1,024
controller            io2
decimation_gain       1
stride_cap            256
radial_compression    enabled
pulse_scheduling      spectral, enabled
dream/self-context    off
```

small, medium, and large are coherent capacity profiles. bands remain the
temporal lens; capacity derives symbol width, scale rank, temporal summaries,
model width, composition depth, and expansion. trace storage is `v*k*8` bytes
plus small auxiliary state.

the configured learning rate is defined at a reference batch of 1,024 dense
predictions. the applied optimizer base is `lr * batch_size / 1024`. together
with pulse amplitude `p/f`, this keeps expected parameter movement over a
fixed number of predicted presents approximately invariant to batch size and
pulse frequency.

standard-mode batch is a hardware policy, not an architecture choice. mps
admission uses measured working-set buckets for each production capacity, a
15% margin, lens scaling, and `torch.mps.recommended_max_memory()`. choices at
or below 70% of that envelope are comfortable, 70–90% are pressure, and above
90% are unsupported. on the 5.73 gb reference envelope the automatic batches
are small 1,024, medium 512, and large 512. research mode may override a
comfortable default, but the gui blocks unsupported admission before model
allocation.

standard gui mode is an executable policy rather than a presentation filter.
for a new model it materialises the 48-band phi lattice, automatic learning
rate, full-density decimation gain, gradient protection, device-fitted batch,
managed source cursor, 30-minute autosave, and disabled self-context before
launch. for a resumed production checkpoint it restores the saved learning
policy, fits batch to the current device, and keeps optional self-context off.
expert mode exposes the underlying overrides; returning to standard mode
removes hidden runtime-only overrides before launch.

checkpoint action is explicit. `continue` retains `model_id` and can publish
only to the selected artifact. `fork` calls the lineage operation before
training, creates a new `model_id`, retains `root_lineage_id`, and records the
selected `revision_id` as `parent_revision_id`. new or unrelated artifacts are
assigned collision-free filenames rather than overwriting existing lineages.
training admission remains disabled while checkpoint metadata and saved policy
are loading.

## 10. checkpoint contract

the `.pt` checkpoint contains:

```text
species, checkpoint_version, runtime_version
model_id, checkpoint_id, revision_id, parent_revision_id, root_lineage_id
created_at, saved_at, description
config
lattice.coordinates, lattice.monitor_density
lattice.retention, lattice.injection, lattice.direct_channel
tokenizer.sha256, tokenizer.filename
model, optimizer, gradients, trace, error_trace, translated_error_bank, rng,
training, metadata, source_cursor
```

`model_id` identifies one owned life and survives ordinary continuation;
`checkpoint_id` is its backward-compatible alias. `revision_id` identifies one
saved state and changes atomically per save. ordinary continuation records the
previous revision as `parent_revision_id`. an explicit fork creates a new model
id, retains `root_lineage_id`, and records the source revision as parent.
`trace` includes the full float64 trace, prior counts, last token, tokens seen,
and source bytes seen. save uses a temporary file followed by atomic replace.
`rng.sampling` stores the model-owned portable sampling generator, so sampled
generation and dream state continue exactly after save/resume.
`gradients` plus the training accumulation fields preserve an unfinished
optimizer quantum without forcing a smaller final update.
`source_cursor` stores the reader's exact raw read-ahead position, pending token
tail and loss weights, protocol mode, byte commitments, source limit, and a
sha-256 signature of the represented corpus boundary. it is part of the model
revision rather than gui state. automatic resume verifies the local corpus and
refuses a changed prefix; appending beyond the represented boundary is safe.

the adjacent `.revision.json` pointer is the publication guard. saves take an
inter-process artifact lock, compare the owned on-disk revision, publish a
pending candidate, atomically replace the `.pt`, and then mark the candidate
complete. loaders reject pending or mismatched publications. a stale writer
cannot replace a newer complete revision.

the adjacent `.pt.meta.json` sidecar is display metadata, not sufficient for
resume. parsers must treat the `.pt` file as untrusted and should use the
sidecar wherever tensor loading is unnecessary. runtime tensor loading must use
pytorch's `weights_only=true` path and then validate identity, lattice, shape,
dtype, and finite state before use.

## 11. invariants

1. all prediction states precede their target token.
2. chunked scan and scalar recurrence produce the same represented state.
3. source state commits only after a finite model result.
4. save/resume produces the same next logits and sampled continuation.
5. tokenizer identity and temporal lattice cannot silently change.
6. state blocks can affect context but cannot reduce supervised loss directly.
7. generated tokens cannot change learned weights or empirical source prior.
8. every generated turn ends in exactly one turn boundary.
9. prediction decimation cannot change the final exact trace state.
10. source block size is bounded independently of prediction stride.
11. the visible residual spectrum has exactly `k` bands; its hidden boundary
    band is never presented as model evidence.
12. radial compression cannot reduce a row merely because it already exceeds
    its reference ceiling, and no optimizer step can increase it farther.
13. frozen-weight predictions are invariant, within model-dtype scan tolerance,
    to source delivery partitioning.
14. every token entering the trace advances the controller's lived-time clock.
15. source fragments smaller than the optimizer quantum cannot advance adamw's
    step counter until their combined weighted prediction mass reaches it.
16. changing batch size cannot reinterpret or discard an unfinished learning
    quantum; the new clock begins only after the old boundary completes.
16. save/resume during an unfinished optimizer quantum produces the same next
    parameter update as uninterrupted accumulation.
17. prediction sampling phase is a function of the checkpointed lived-token
    clock and cannot reset at source delivery boundaries.
18. a sampled residual cannot be assigned backward to the interval preceding
    its observation.
19. io2 actuation cannot occur merely because a source fragment ended.
20. save/resume with a pending tokenizer tail produces the same subsequent
    tokens and parameter updates as uninterrupted corpus reading.
21. batch size cannot change while optimizer or controller prediction mass is
    unfinished.
22. every fragment of one pulse quantum shares one update/skip decision.

## 12. cockpit and run contract

the desktop process does not own the lifetime of a trainer. each launch creates
a durable run id and five records: configuration, manifest, lifecycle event
journal, scalar metric history, and cooperative stop request. the manifest is
atomically replaced and contains worker pid, status, heartbeat, latest report,
last safe checkpoint time, source, destination, mode, and persistent warning.
the metric history records an analysis-ready scalar sample every 60 seconds;
large spectral arrays are excluded. records are typed as `training` or
`validation`.

an optional validation source must be a distinct file. when training cycles all
data files, the selected validation file is removed from the training set.
managed cadence evaluates a dense, private continuation of at most 250 kb at
baseline and every 100 mb of subsequent source experience. evaluation changes
neither lived trace nor learned state. its result records estimated bpb,
weighted exact-token accuracy, tokens, source bytes, runtime, and the model's
source experience at evaluation. cadence and sample size are expert controls.

fresh models publish revision zero before source consumption. checkpoint
publication is an explicit `saving` state, autosave failure retries within one
minute, and a final-save failure makes the run failed rather than completed.
the cockpit reserves enough destination storage for atomic checkpoint
replacement before launch and continuously displays storage remaining,
checkpoint age, report health, current-pass eta, and the latest dense held-out
result with its age.

closing the cockpit may detach without stopping the trainer. on restart the app
reattaches only when both pid and worker command match the recorded run. it
hydrates the last durable report, follows new lifecycle events and manifest
reports, and writes to the same cooperative stop channel. a missing live worker
is marked interrupted; recovery always begins from the last complete model
revision, whose checkpoint contains the exact reader and optimizer state.

while training owns a checkpoint, chat may load the last published revision as
a disposable context snapshot. online mutation is disabled and the chat copy
cannot save over the active lineage. this keeps inspection separate from
ownership without preventing qualitative checks during a long run.
