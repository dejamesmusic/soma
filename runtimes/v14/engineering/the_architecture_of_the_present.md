# the architecture of the present

## an engineering design for a trace-based continual branch predictor

status: reference design for the first theory-led implementation

companion document:
[the geometry of the present](../theory/the_geometry_of_the_present.md)

## abstract

the theory defines soma as a predictor over a self-affine trace representation
of symbolic history. the trace bank is the authoritative temporal state. each
new symbol applies one known affine contraction. the learned model estimates
the probability of the next symbol-labelled branch.

the engineering problem is to preserve that object without paying for a dense
learned transformation over every symbol-timescale coordinate at every token.

the proposed architecture has five parts:

1. an exact, nonlearned trace bank with geometrically spaced exponential rates;
2. a learned symbol embedding used to project the full trace state without
   replacing it;
3. a shared log-scale encoder that compresses the projected trace field into a
   small set of scale summaries;
4. a conventional serial residual trunk that supplies compositional capacity
   within the present;
5. a tied branch-energy readout that produces ordinary next-symbol
   probabilities.

the authoritative trace state is retained in cpu float64. the learner operates
in device float32 or bfloat16. projected traces are recomputed at every weight
update boundary, so changing symbol geometry is applied consistently to the
entire remembered state. no learned recurrence is added, no gradient modifies
the trace bank, and no controller changes the language objective.

the reference architecture is called the **trace branch mixer**.

## 1. engineering goals

the implementation must:

- preserve the exact causal trace recurrence;
- represent long decay horizons without losing updates to device precision;
- keep model parameters and dominant matrix work from growing with
  `v × k`;
- permit real serial composition after temporal evidence has been organized;
- retain magnitude information;
- produce a legal next-symbol distribution;
- use the same state transition during training, generation, and chat;
- support indefinite checkpoint continuation;
- keep trace advancement separate from optimizer sampling;
- expose enough internal structure to test the theory without making
  diagnostics part of the learning rule.

the first implementation should be small enough to inspect and strict enough
to falsify. production conveniences may follow only after the reference path
is correct.

## 2. complete data path

```text
token ids
   |
   v
authoritative float64 trace state
   |
   v
current learned symbol projection
   |
   v
projected trace scan over the contiguous batch
   |
   v
shared log-scale encoder
   |
   v
scale summaries + instantaneous channel
   |
   v
serial compositional trunk
   |
   v
tied branch-energy readout
   |
   v
next-token distribution
```

temporal state exists only in the trace bank. every operation below it is an
instantaneous function of the represented present.

## 3. symbol process and tokenizer

the tokenizer defines the alphabet. it is part of the model rather than an
interchangeable preprocessing detail.

the reference implementation uses:

- a fixed tokenizer file;
- a finite vocabulary `v`;
- an explicit unknown or byte-fallback mechanism;
- reserved control symbols only when their semantics are documented;
- a tokenizer hash stored in every checkpoint.

a 1,024-symbol dictionary is the initial reference because:

- the complete trace state remains small;
- output softmax is inexpensive;
- token sequences are substantially shorter than byte sequences;
- the symbol embedding and output matrix remain cheap;
- behavior can be compared with existing soma token runs.

the architecture scales linearly with vocabulary in:

- trace-state memory;
- symbol embedding parameters;
- final output scoring.

no architectural claim depends on exactly 1,024 symbols.

## 4. exact temporal lattice

### 4.1 direct present

the most recent observed symbol is retained as an explicit instantaneous
channel:

```text
x_direct = e_(y_(t-1)).
```

this avoids pretending that any finite exponential lifetime is equivalent to
the immediate past.

### 4.2 exponential bands

choose:

```text
beta_k = beta_0 phi^(-k)
rho_k = exp(-beta_k)
a_k = 1 - rho_k.
```

the reference choice is:

```text
beta_0 = 1.
```

the first exponential therefore has an e-fold lifetime of one token:

```text
tau_0 = 1 / beta_0 = 1.
```

the remaining lifetimes are:

```text
tau_k = phi^k.
```

the direct channel and first exponential intentionally overlap. one is an
exact boundary observation; the other begins the closed contraction lattice.

### 4.3 horizon selection

for target lifetime `tau_max`:

```text
k
  = 1 + ceil(log_phi(tau_max / tau_0)).
```

bands determine temporal range and resolution, not learned parameter count.

the reference local profile uses approximately 40 to 48 exponential bands.
the exact number must be written into the checkpoint.

## 5. authoritative trace state

store:

```text
q in float64 cpu memory with shape [v, k].
```

the trace update is:

```text
q_next[:, k]
  = rho_k q[:, k] + a_k e_y.
```

the state is:

- nonlearned;
- detached from autograd;
- persistent across batches;
- saved in checkpoints;
- advanced through every observed token;
- never reconstructed from hidden activations.

for `v = 1,024` and `k = 48`, the state occupies:

```text
1,024 * 48 * 8 bytes
  = 393,216 bytes.
```

float64 trace storage is negligible beside model and optimizer memory.

### 5.1 why cpu float64

very slow bands receive injections much smaller than float32 relative
precision. repeated float32 additions can disappear even though their
cumulative contribution should become meaningful.

the cpu state is authoritative because:

- mac mps does not support float64;
- the state is small;
- slow-band fidelity matters more than raw trace arithmetic speed;
- batch updates can be vectorized and overlapped with device work.

the learner receives a float32 projection of the state. this does not erase
accumulated slow evidence: the float64 state is reprojected at every optimizer
boundary.

### 5.2 exact batch update

for a batch of `b` observed tokens:

```text
y_0, ..., y_(b-1),
```

the final state is:

```text
q_end[:, k]
  = rho_k^b q_start[:, k]
  + a_k
    sum_(j=0 to b-1)
      rho_k^(b-1-j) e_(y_j).
```

this update requires:

- one decay of each trace column;
- one weighted scatter-add per token and band.

it does not require a dense `b × v × k` tensor.

the cpu update can run concurrently with device training on the current
batch, provided synchronization completes before the next batch begins.

## 6. learned symbol projection

let:

```text
e in real^(v x d_symbol)
```

be a learned symbol embedding.

choose a reference symbol distribution `pi`, stored as cumulative or smoothed
online counts. define:

```text
e_pi = pi^transpose e.
```

the centered projected trace at band `k` is:

```text
h_t[k]
  = (q_t[:, k] - pi)^transpose e.
```

the instantaneous channel is:

```text
h_direct
  = e[y_(t-1)] - e_pi.
```

centering places the projected evidence in the tangent coordinates described
by the theory. the context-free symbol prior is retained separately in the
readout.

### 6.1 the full trace remains recoverable by width

the projection is a learned rank constraint:

```text
v -> d_symbol.
```

when:

```text
d_symbol >= v
```

and `e` has full rank, the symbol projection can preserve the complete trace
plane. smaller widths trade decoder capacity for speed and memory.

this is a model approximation boundary, not a change to the authoritative
temporal representation.

### 6.2 current geometry is applied to remembered history

the embedding changes during learning. historical events must be interpreted
through the current embedding rather than permanently encoded using the
weights that happened to exist when they arrived.

therefore:

1. store `q`, not a persistent trace of learned embeddings;
2. at each optimizer boundary, project `q_start` through the current `e`;
3. discard the projected cache after the weights change;
4. reproject on the next batch.

this prevents a weight-dependent temporal state from replacing the fixed trace
representation.

## 7. projected batch scan

during one optimizer batch the embedding is fixed. the projected trace obeys:

```text
h_(t+1)[k]
  = rho_k h_t[k]
    + a_k (e[y_t] - e_pi).
```

starting from:

```text
h_start[k]
  = (q_start[:, k] - pi)^transpose e,
```

this recurrence produces exactly the same projected features as projecting
every intermediate full trace state through the current embedding.

the training path therefore avoids materializing:

```text
[batch, vocabulary, bands].
```

it materializes:

```text
[batch, bands, d_symbol].
```

### 7.1 gradient semantics

the trace state and its recurrence coefficients receive no gradients.

gradients may reach the shared embedding through the projected scan. this is
the derivative of:

```text
q_t^transpose e
```

with respect to `e`; it does not make `q_t` learnable or propagate credit into
past trace states.

no gradient crosses an optimizer boundary. the authoritative trace is
detached and reprojected after every parameter update.

### 7.2 implementation stages

the correctness implementation may use a straightforward token scan.

the production implementation should use:

- a fused device scan;
- an associative scan where supported;
- or a custom metal/cuda kernel.

the recurrence is diagonal across bands and vectorized across embedding
channels, making fusion straightforward.

slow-band movement below device relative precision may remain constant within
one batch. accumulated movement reappears at the next float64 reprojection.
this is appropriate when its true lifetime is many orders of magnitude longer
than the batch.

## 8. the log-scale encoder

the projected trace field has shape:

```text
[batch, bands, d_symbol].
```

applying a full `d_symbol × d_symbol` matrix independently at every band and
layer would make computation grow linearly with the temporal lens. the
reference encoder instead uses a depthwise-separable graph over log-time.

each scale block contains:

1. a shared depthwise convolution over neighbouring band indices;
2. a low-rank channel mixer shared at every band;
3. a residual dry path;
4. a small scale-conditioning signal;
5. no persistent state.

### 8.1 scale-local transport

for projected field `h`:

```text
u
  = h
  + gate_time
    depthwise_conv_logtime(h).
```

the same convolution kernel is used at every interior band. finite boundaries
use explicit masks rather than silently treating padding as evidence.

this operation supplies approximate translation symmetry along log lifetime.

### 8.2 shared symbol-feature mixing

with interaction rank `r`:

```text
v
  = u
  + gate_symbol
    w_up activation(w_down u).
```

where:

```text
w_down has shape [d_symbol, r]
w_up has shape [r, d_symbol].
```

the matrices are shared across bands. complexity is:

```text
order(batch * bands * d_symbol * r)
```

rather than:

```text
order(batch * bands * d_symbol^2).
```

gated linear units may replace the single activation after the reference
implementation is verified.

### 8.3 absolute scale information

language is not assumed to be exactly scale-invariant. spelling and discourse
do not obey identical dynamics merely because their lifetimes are related by
dilation.

the encoder therefore receives a small scale-conditioning vector derived
from:

```text
log(tau_k)
```

or an equivalent band embedding. it modulates shared operations through
low-rank gains rather than allocating an independent dense matrix to every
band.

this yields:

- shared local structure;
- explicit absolute scale;
- parameter growth much smaller than independent band networks.

### 8.4 scale compression

after several shared blocks, a strided local convolution compresses `k` bands
into `h_scale` overlapping summaries:

```text
s in real^(h_scale x d_symbol).
```

these outputs are called scale summaries rather than coherence modes. the
network may use them to approximate coherent observables, but no clean
eigendecomposition is assumed.

the scale encoder is the only component whose activation cost grows
materially with the number of bands.

## 9. context assembly

flatten the scale summaries and append the instantaneous channel:

```text
c_raw
  = concat(
      flatten(s),
      h_direct
    ).
```

project into trunk width:

```text
c_0 = w_context c_raw.
```

this is the learned compression boundary between organized temporal evidence
and unrestricted within-present composition.

its input width is:

```text
(h_scale + 1) d_symbol.
```

its output width is:

```text
d_model.
```

increasing the number of trace bands does not increase the serial trunk width.

## 10. serial compositional trunk

the trunk is an ordinary residual multilayer perceptron operating on the
current context:

```text
c_(l+1)
  = c_l
  + residual_l(c_l).
```

the reference residual uses a swiglu-style branch:

```text
residual_l(c)
  = scale_l
    w_out_l(
      silu(w_gate_l n(c))
      elementwise_multiply
      w_value_l n(c)
    ).
```

where `n` is confined to the residual branch. the dry path remains
unnormalized.

### 10.1 preserving magnitude

plain normalization can make the nonlinear branch blind to evidence
magnitude. the reference design exposes the pre-normalization rms scalar to
the branch:

```text
r = rms(c)
n = c / max(r, epsilon)
```

and permits:

```text
residual_l(c)
  = r
    scale_l
    f_l(n, log(r)).
```

this preserves direction conditioning without declaring magnitude irrelevant.
the first code draft may use the simpler branch-only rms form while retaining
the dry path and reporting the rms values.

### 10.2 role of depth

depth does not add temporal memory. it composes interactions among evidence
already present in the trace representation.

in a local linearization, deeper layers add longer computational paths. they
allow combinations of symbol and scale evidence that cannot be represented by
one low-rank projection.

the trunk is deliberately conventional. previous attempts to replace
composition entirely with temporal machinery changed the problem rather than
solving it.

### 10.3 initialization

residual branches use:

- variance-preserving weight initialization;
- residual scale approximately proportional to `1 / sqrt(depth)`;
- a zero-initialized final branch readout.

zero learned readout must recover the context-free reference distribution
rather than random logits.

## 11. tied branch-energy readout

project the final context back into symbol-embedding space:

```text
r_branch
  = w_branch c_final
```

with:

```text
w_branch in real^(d_symbol x d_model).
```

score every candidate symbol using the same embedding that interprets the
trace:

```text
score_y
  = inner_product(e[y], r_branch)
    / sqrt(d_symbol).
```

let `pi` be the smoothed context-free symbol distribution. logits are:

```text
logit_y
  = log(pi_y + epsilon)
    + score_y.
```

and:

```text
p_theta(y | q)
  = softmax(logits)_y.
```

this corresponds to transition energy:

```text
u_theta(q, f_y(q))
  = -log(pi_y + epsilon)
    - score_y(q).
```

the source state enters through `r_branch`. the candidate legal branch enters
through `e[y]`.

### 11.1 why tie input and output symbols

weight tying:

- uses one learned geometry for remembered and candidate symbols;
- makes branch scoring linear in vocabulary;
- reduces parameters;
- allows changes to symbol geometry to affect both interpretation and action;
- keeps every candidate associated with the translation used by its legal
  successor map.

the output rank is bounded by `d_symbol`. increasing `d_symbol` is the direct
way to relax this constraint.

an untied residual output may be tested later, but it is not part of the
reference architecture.

## 12. objective and optimization

### 12.1 language objective

the sole reference objective is weighted next-token cross-entropy:

```text
loss
  = -mean_t log p_theta(y_t | q_t).
```

source bytes are retained for universal bpb reporting:

```text
bpb
  = total_token_nll / (source_bytes log(2)).
```

the following do not alter the definition of the language gradient:

- residual spectra;
- band energies;
- coherence diagnostics;
- generated feedback;
- self-predicted motor channels.

the v13.1 production runtime uses residual spectra and an io2 controller only
to choose the scalar learning rate and which predictive positions instantiate
that same objective. it does not add an auxiliary language target.

### 12.2 optimizer

the first implementation uses a conventional optimizer:

- adamw;
- float32 optimizer parameters on mps;
- bfloat16 or float32 compute on supported cuda devices;
- global gradient-norm clipping;
- no hard periodic row projection;
- a small, explicit weight decay selected independently of trace timing.

the purpose is to test the architecture without embedding another theory
inside its optimizer.

v13.1 surrounds this optimizer with io2. its signed error bank is detached from
backpropagation, so the controller changes plasticity and compute allocation
without redefining the representation or objective.

### 12.3 continual batches

training batches are contiguous segments of the stream.

for each batch:

1. freeze current parameters for the batch;
2. project `q_start` through the current symbol embedding;
3. construct predictive projected traces before each selected target;
4. run the model and ordinary cross-entropy on those selected states;
5. advance authoritative `q` through every source token;
6. apply one optimizer update;
7. discard projected caches;
8. begin the next batch by reprojecting the updated trace state.

there is no random context window and no reset between batches.

## 13. decimation and compute policy

decimation is not part of the sensory representation.

the trace bank advances through every token. an optional optimizer stride
selects which predictive positions contribute expensive model work and
gradients.

the dense architectural reference begins with:

```text
optimizer_stride = 1.
```

the v13.1 production policy is io2 lens decimation. it uses a translated signed
residual bank to choose a stride while preserving every source event in the
exact trace. any sparse policy must:

- preserve complete trace advancement;
- test and report possible corpus-period locking under deterministic stride;
- report source tokens, selected predictions, and optimizer updates
  separately;
- be evaluated at equal wall clock as well as equal source exposure.

no model-controlled stride is included. io2 is an external deterministic
controller whose state is checkpointed with the learner.

## 14. coherence support

coherence is a property to recover and measure, not a target to impose by
name.

the model supports coherent modes through:

- multiple exponential lifetimes;
- signed learned symbol embeddings;
- local shared mixing along log-time;
- nonlinear serial composition;
- unrestricted real hidden channels capable of representing conjugate
  two-dimensional phase pairs.

complex arithmetic is unnecessary for the first implementation. a real
network can represent a complex mode through paired channels.

### 14.1 optional detached monitor

an off-by-default diagnostic may retain a small random projection of trunk
contexts:

```text
z_t = random_projection(c_t).
```

periodically estimate:

```text
c_00 = mean(z_t z_t^transpose)
c_10 = mean(z_(t+1) z_t^transpose).
```

a regularized low-rank estimate:

```text
a_probe = c_10 inverse_regularized(c_00)
```

can expose approximate mode magnitudes and phases.

this monitor:

- is detached from training;
- has a strict fixed memory budget;
- runs infrequently;
- makes no controller decision;
- is labeled as an empirical local linearization.

it exists to test whether the learned state develops coherent observables, not
to manufacture them.

## 15. generation and interaction

generation uses exactly the training state transition:

1. compute `p_theta(y | q)`;
2. sample or select `y`;
3. apply `q <- f_y(q)`;
4. repeat.

when weights remain fixed, cache the projected trace state:

```text
h_next[k]
  = rho_k h[k]
    + a_k (e[y] - e_pi).
```

this avoids reprojecting the full trace for every generated token.

after any optimizer update, invalidate the cache and reproject authoritative
`q`.

generated tokens update the trace because they are events in the model's
experienced stream. they do not induce weight updates by default. externally
observed user tokens may be trained online only under an explicit interaction
policy.

## 16. checkpoint contract

every checkpoint must contain:

- architecture version;
- tokenizer bytes or immutable tokenizer hash;
- vocabulary size;
- `beta_0`, phi value, band count, `rho`, and `a`;
- direct-channel convention;
- authoritative float64 trace state;
- symbol-prior counts and smoothing rule;
- model parameters;
- optimizer state when resumable training is requested;
- source tokens and source bytes seen;
- optimizer updates and selected prediction count;
- corpus cursor metadata;
- numerical dtypes;
- generation delimiter and control-symbol definitions;
- user-editable description stored separately from immutable structure.

loading must reject:

- tokenizer mismatch;
- vocabulary mismatch;
- temporal-lattice mismatch;
- missing authoritative trace state for a continuing checkpoint;
- incompatible projection or trunk shapes.

silent reinterpretation is not permitted.

## 17. memory and compute

let:

```text
v = vocabulary size
k = exponential bands
d = symbol embedding width
r = shared scale-interaction rank
h = retained scale summaries
w = serial trunk width
l_scale = scale encoder depth
l_trunk = serial trunk depth
m = swiglu expansion width.
```

### 17.1 parameter scaling

approximate learnable parameters:

```text
symbol embedding:
  v d

scale encoder:
  l_scale (kernel_width d + 2 d r)

context projection:
  ((h + 1) d) w

serial trunk:
  l_trunk (3 w m)

branch projection:
  w d
```

the output embedding is tied and adds no second `v d` matrix.

most parameters live in the serial trunk. temporal horizon changes state and
stem work without exploding model width.

### 17.2 activation scaling

dominant stored activations:

```text
projected trace:
  batch * k * d

scale summaries:
  batch * h * d

serial trunk:
  batch * l_trunk * w
```

the implementation never stores:

```text
batch * v * k
```

dense trace activations.

### 17.3 compute scaling

projected scan:

```text
order(batch * k * d)
```

scale encoder:

```text
order(
  batch * l_scale * k * d * r
)
```

serial trunk:

```text
order(
  batch * l_trunk * w * m
)
```

output:

```text
order(batch * v * d).
```

the dominant dense matrix work is independent of band count.

## 18. reference profiles

these are engineering starting points, not theoretical optima.

### small verification

```text
vocabulary             1,024
bands                   32
symbol width            128
scale rank              32
scale blocks            2
scale summaries         4
trunk width             512
trunk depth             3
swiglu expansion        1,024
```

purpose:

- correctness;
- cpu and mps comparison;
- batched versus token execution;
- short corpus learning.

### local reference

```text
vocabulary             1,024
bands                   48
symbol width            256
scale rank              64
scale blocks            3
scale summaries         8
trunk width             1,024
trunk depth             4
swiglu expansion        2,048
batch                    512
```

approximate parameters:

```text
28 to 32 million
```

depending on scale-conditioning and gating details.

approximate persistent trace state:

```text
0.4 mb.
```

projected trace activation at batch 512:

```text
512 * 48 * 256 * 4 bytes
  approximately 24 mb.
```

this is the first serious local-training profile.

### capacity reference

increase:

- trunk width;
- trunk depth;
- swiglu expansion;
- symbol width when output rank becomes limiting.

increase bands only when a longer operational trace horizon is justified.

capacity scaling should primarily enlarge the serial trunk rather than
replicate independent models at every timescale.

## 19. runtime pipeline

### 19.1 training worker

```text
read contiguous token batch
        |
        +--> launch float64 cpu trace-end update
        |
        v
project q_start through current embedding
        |
        v
device projected scan
        |
        v
scale encoder
        |
        v
serial trunk
        |
        v
tied branch logits
        |
        v
cross-entropy and backward
        |
        v
optimizer step
        |
        v
join cpu trace update
        |
        v
commit q_end and checkpoint counters
```

the authoritative state is committed only after both device and cpu work
complete successfully.

### 19.2 failure atomicity

if the optimizer step or trace update fails:

- do not advance the committed corpus cursor;
- do not partially replace `q`;
- do not report the batch as seen;
- restore the last committed optimizer and trace boundary when resuming.

periodic checkpoint files are written atomically through temporary files and
rename.

## 20. correctness tests required before learning claims

### trace tests

- token-by-token float64 recurrence equals closed-form batch update;
- zero initialization has the expected sub-normalized mass;
- prior initialization preserves unit mass;
- exact-beta rates satisfy:

```text
rho_k
  approximately_equals
  rho_(k+1) rho_(k+2)
```

to float64 tolerance;
- slow bands continue changing beyond float32 relative precision.

### projection tests

- projected scan equals explicit:

```text
q_t^transpose e
```

at every position;
- changing `e` and reprojecting changes all historical symbol coordinates
  consistently;
- no gradient reaches authoritative `q`;
- batched projection gradients equal explicit dense-reference gradients.

### model tests

- zero branch readout predicts the smoothed prior;
- logits form a normalized next-symbol distribution;
- every predicted distribution defines one legal expected successor;
- scale padding cannot create evidence;
- scale encoder parameter count does not grow as `k d^2`;
- generation and teacher-forced state transitions agree.

### continuation tests

- save and resume produces identical next logits;
- corpus cursor and trace state commit together;
- changing batch boundaries without changing weight updates preserves trace
  states;
- tokenizer or lattice mismatch fails visibly;
- long repeated load, train, stop, and resume cycles do not retain stale model
  instances or device allocations.

## 21. performance work after correctness

optimization order:

1. fuse the projected diagonal trace scan;
2. overlap cpu float64 state update with device training;
3. fuse depthwise scale convolution and low-rank channel mixer;
4. use compiled swiglu trunk kernels;
5. avoid device synchronization in reporting;
6. compute diagnostics from detached summaries at low frequency;
7. reuse generation caches while weights remain fixed;
8. profile tokenizer, corpus io, and checkpoint writing separately from model
   compute.

do not:

- lower trace precision silently;
- cache learned projected traces across parameter updates;
- decimate trace advancement;
- replace exact rates with approximate powers for convenience;
- make reporting tensors retain autograd graphs;
- store per-token full vocabulary trace planes.

## 22. answers to the theory document

### how is the contraction lattice represented?

by exact geometric `beta`, float64 `rho` and `a`, an explicit direct channel,
and immutable checkpoint metadata.

### how are finite endpoints handled?

the present has a separate direct channel. finite scale boundaries are masked
explicitly. exact global scale equivariance is not claimed.

### how is the branch kernel parameterized?

by a trace-conditioned serial context and tied candidate-symbol energies.

### how are symbol and scale interactions shared?

symbol geometry is shared through one embedding. scale interactions use
depthwise local transport and low-rank feature mixing shared across log-time,
with small absolute-scale conditioning.

### how is probability mass preserved?

the trace recurrence preserves normalized columns after normalized
initialization. learned features operate outside the authoritative state.
output softmax defines a legal branch distribution.

### how does precision bound memory?

the authoritative trace uses float64. band count is selected so injections and
state changes remain representable over the intended horizon. device
projections are refreshed from that state after every optimizer update.

### how is coherence exposed?

through multiple decay coordinates, scale-local mixing, and serial nonlinear
composition. a detached low-rank monitor may estimate local operator modes,
but coherence is not inserted as a training target.

### how are training and generation aligned?

both use the same legal trace transition. only the source of the next symbol
differs.

### how does compute scale?

trace and stem work grow approximately linearly with bands. dominant dense
trunk work depends on chosen model width and depth, not temporal horizon.

### what remains falsifiable?

the embedding rank, scale encoder, summary count, tied output, serial depth,
and exact-beta lattice are architectural hypotheses. each can fail to extract
the information retained by the trace state.

## 23. architectural thesis

> the trace branch mixer retains the complete finite-precision symbol trace as
> a small, exact, nonlearned state. it applies the current learned symbol
> geometry to that state at every optimizer boundary, organizes temporal
> evidence through shared operations on uniform log lifetime, compresses it
> into a fixed-size present, and gives ordinary serial depth the task of
> composing that present into energies over legal successor symbols. temporal
> range, symbol geometry, and compositional capacity therefore have separate
> controls and separate costs.

this is the engineering bet to be implemented in the third deliverable.
