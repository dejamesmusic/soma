# soma rma_v1

`rma` means residual metabolic aperture.

## fixed algorithm

```text
token stream
  -> exact phi identity traces
  -> present-token anchor + decayed adjacent-token differences
  -> tied symbol coordinates
  -> nested cumulative apertures
  -> additive language revisions
  -> tied categorical readout

optional cortical composition
  <- exact recent lag bands, newest first
  <- every unsummed aperture opinion
  -> wide nonlinear residual blocks
  -> zero-start residual gate

categorical prediction residual
  -> phi residual traces
  -> white-noise-corrected io2 spectrum
  -> prediction-stride and plasticity controller

all learned matrices
  -> two source-time clocks
  -> tangent sgd
  -> fixed global matrix energy
```

all persistent state is bounded. matrix magnitude is fixed; learning is directional
movement and redistribution of a finite representational energy. categorical bias is
zero-centred and bounded. no optimizer moments are retained.

the input representation is a signed transition spectrum. if `x_t` is the one-hot
token event, every temporal band integrates `x_t - x_(t-1)` before learning. the
fastest coordinate exposes the exact most recently observed token; slower coordinates
expose leaky adjacent differences in their literal band amplitude. this is an invertible
change of coordinates from the original low-pass token traces when the present token is
known, but it makes arrival positive, departure negative, and stationary input silent.
the persistent checkpoint tensor remains the bounded float64 identity trace field; the
transition spectrum is projected from it without adding state or parameters.

when exact bands are enabled, they overlap rather than partition history. the most
recent `w` tokens are available to the cortical mixer as full-resolution lag bands
ordered `t-1 ... t-w`, and every token also enters the temporal trace field on arrival,
so the base predictor sees the whole stream whether or not a cortex is attached. an
earlier partitioned layout, which withheld the last `w` tokens from the base, cost
0.195 bits per byte against no cortex at all; the overlapping layout gains 0.030.

new models use `adjacent_difference`. checkpoints created by runtime 1.1 and earlier
are explicitly restored as `cross_band`, preserving the coordinates in which their
weights learned. input representation cannot change during continuation.

## exposed scale and metabolic controls

- `width`: learned symbol and aperture capacity;
- `bands`: temporal extent, with aperture cutoffs derived from the phi geometry;
- `cortex_width` and `cortex_depth`: optional present-time nonlinear capacity over
  the unsummed aperture opinions;
- `workspace_tokens`: the number of full-resolution exact lag bands available to the
  optional frontal cortical mixer, alongside the full temporal field;
- the cortical proposal is bounded smoothly so it never exceeds the base language
  state in size, and trust admits it through tanh(softplus(x)), which lies in (0, 1):
  closing trust removes the proposal instead of subtracting it. the fixed-radius
  sphere bounds each matrix's energy but not its gain, and without the bound a
  768-wide cortex at rate 1.0 diverged on two of three fresh seeds within about 75
  updates; with it, all were stable;
- `cortex_rate`: frontal learning rate as a fraction of the base learning rate, one of
  0.25, 0.5 or 1.0 (default and cap). with a 256 x 2 cortex over a width-256 base, 1.0
  was best held-out on two seeds; 1.25 was already erratic and 2.0 diverged. the base
  controller reads only the base loss, so the rate cannot change the base clock;
- `readout_window`: exact recent tokens the cortex's sentinel pointer can copy from
  (default 256; used only when a cortex is on). each of the last `R` tokens is scored
  against the base's present state by cosine; one softmax over those scores and a
  sentinel splits probability between the model's own prediction and the recent
  tokens. a mixture only moves probability, so no token loses more than the share the
  pointer holds. the pointer's matrices rotate at most 0.01 radians per update:
  coarser steps slam its softmax open or shut. at width 256 with the controller off, a
  256-token window took held-out gain over the base from -0.042 (cortex alone) to
  -0.098 bpb on two seeds, cutting 2.8-3.4 bits from content words seen earlier in
  the text. a checkpoint whose cortex predates the pointer is grafted on load: its
  weights are kept and a fresh pointer starts holding about 0.2% of probability.
  training reads every readout loss at the target alone: log p(target) of the mixture
  is one logsumexp, the target's logit and the pointer's weight on the target, and
  the cortex's gradients are row scalars times (onehot - softmax), written directly
  (`ReadoutTargets`). no vocabulary-sized mixture or gradient is built, the reported
  top-1/top-5 come from the target's exact rank, and pointer keys are projected only
  for the distinct tokens in the window. values and gradients match the full mixture;
  at a 49k vocabulary this took a training step from 758 to 1275 tokens/s;
- `body`: a small pretrained transformer soma consults ("" for none; the bundled body is
  `smollm2-135m`, Apache-2.0, 134.5m frozen parameters in `bodies/`). soma then reads
  the body's alphabet (`body_tokenizer.py`, byte-level BPE matching HuggingFace token for
  token), its width is the body's (576), its dictionary starts as the body's embeddings
  and keeps learning, and a cortex is required. the body reads every token soma reads
  and offers its prediction as an expert in soma's readout, admitted by a learned gate
  from soma's language and the body's own confidence: p(x) = (1 - g) p_soma(x) +
  g (1 - P_soma(controls)) p_body(x | text), while each control keeps soma's own
  probability. controls are soma's protocol, so the body reads them as text ("\nUser: ",
  "\nSoma: ", "\n") and never prices them; when the body writes "\nUser:" itself it is
  handing the turn back, and generation ends the turn there. the gate carries a fixed
  prior of +4 (a new soma starts with about 2% of the vote, like the pointer's sentinel
  offset) and soma earns trust as it learns. training reads each loss at the target only
  (the body's log-probability of the true token is all a mixture's loss needs). the
  body's opinion before a token depends only on the tokens just before it, so for learning
  it holds no state: it reads fresh windows, batched, each starting at least 256 tokens
  before the first position it serves and serving up to 2,048 (1.125 reads a token), and
  only at the positions io2 trains on. a gap between trained rows wider than 256 starts a
  fresh window, so a row costs min(gap, 257) tokens of reading: below that the rows'
  contexts overlap and every token is computed once, which is why an update's body time
  grows with io2's stride (at stride 32 it reads about 7,400 tokens, 1.5 s of a 1.8 s
  update). on conversation text, windows seeing 256-2,304 tokens scored 0.8827 bpb
  against 0.8938 for 1,024-2,048 from the earlier cache, and soma with the body trained
  at 4,300 tokens/s against 1,830. its whole state is its last 256 stream tokens, kept in the checkpoint;
  generation syncs a bounded cache from them and steps it. on MPS its decode step is fused (one-token matvecs at memory bandwidth, SwiGLU in the
  gate/up pass, rotary embedding with the cache write, residual-add with RMSNorm):
  3.9 ms a token against 34 ms in HuggingFace. on 1m tokens of real messages the body alone scored 1.187 bpb and soma with the
  body 1.145, the gain growing from 0.9% in the first fifth to 6.5% in the last;
- chat learns as training does, with two differences: no decimation and variable
  batches. each prompt (with its markers, which carry no loss) is one batch trained at
  every position and closed by one update, scaled by source time: in corpus training every
  update moves `source_gain` x plasticity and io2's stride sets how much source it spans
  (quantum / density tokens), so a chat update spanning n source tokens moves
  n x density / quantum of a corpus update, and a prompt longer than that span closes one
  equal update per span. io2 keeps observing chat, so its density and plasticity are the
  live ones. with a body, io2 runs exactly as without one; decimation leaves the mixture no
  worse than the body alone once the gate's prior holds a sparsely trained soma nearly shut
  (at cap 32 it had been 2.7% worse without the prior);
- `source_gain`: full directional movement of one optimizer update;
- `evidence_quantum`: fixed batch of admitted predictions per optimizer update;
- `decimation_stride_cap`: maximum source-token distance between admitted errors;
- `traversal_gain`: dilation of the residual-derived coordinate on the phi axis.

the slow clock, evidence-normalized update law, activation budget, matrix radii,
whitening response, and controller source-time conversion are derived invariants rather
than independent user tuning knobs.

the application exposes the audited width points 128, 256, 384, and 512, plus explicit
scale options 640, 768, and 1024. the audited points contain approximately 3m, 10m,
21m, and 36m parameters at 32 bands. symbol width and aperture width move together because the matched-parameter
unbalanced controls were weaker. width changes the default batch size;
source gain remains invariant because every update uses the mean admitted
gradient.

the cortical path is an experimental graft rather than a replacement. it reads detached
aperture and workspace features, so its representation gradient cannot rewrite their
temporal clocks. the original summed aperture language retains its own loss and norm-match
domain. the residual candidate learns independently; a sample-conditioned two-opinion
trust readout decides whether to admit it. identical initial trust directions make the
gate exactly zero without introducing zero-radius matrices. the candidate itself is the
difference of two identical directional readouts, so it also begins as an exact zero and
learns by rotation rather than presenting a random residual to trust. cortical gain is
derived as `min(1, 64 / cortical_width)`, preserving the validated angular learning quantum
as quadratic cortical capacity scales. width, depth, and workspace are checkpoint shape
choices and cannot change during continuation. short controls have not shown a predictive
advantage, so the application leaves the graft off by default.

## kernel boundary

the dense torch implementation remains the arithmetic reference. the first metal kernel
keeps the full categorical residual field on device and fuses residual construction,
phi-trace advancement, adjacent differencing, and vocabulary reduction. it returns only
band energies at the controller observation quantum.

later kernels may fuse trace projection and cumulative aperture prefixes, but may not
change prediction or controller arithmetic.

the first retained generation fuses the exact categorical residual sensor and compact
sphere update. stable mps reductions are deliberately retained for optimizer statistics;
an all-metal atomic reduction was faster but changed long continual trajectories.

## application state

the application checkpoint contains exactly the state needed to continue the same
causal learner:

- learned parameters and the fixed radius assigned to every learned matrix;
- the float64 identity trace field;
- the categorical residual trace field;
- io2 band, fractional stride phase, residual state, and controller clocks;
- accumulated gradients and prediction mass at a partial batch boundary;
- source-normalized sgd counters and the exact signed corpus read-ahead cursor;
- tokenizer identity, lineage identity, and torch sampling state.

reports, plots, dream text, and diagnostic histories are observations. they are written
to the run journal and do not become unbounded model state.

## reversible annealing

source traversal and parameter temperature are independent actuators. physical blocks
and optimizer updates retain a fixed batch size. io2 admits every phi-spaced
prediction while all intervening tokens still traverse the temporal state. gradients
accumulate until the fixed quantum is complete, so decimation spends less prediction
compute and covers more source without redefining the learner's update.

the io2 measurement also supplies a reversible plasticity amplitude. normalized
residual energy, spectral coherence, and loss relative to chance produce a bounded
ratio between 0.02 and 1.0. predictable streams cool the directional learner, while
coherent new error can reopen movement after a domain change. optional cortical
composition does not control this temperature.

traversal uses the frequency centroid of the whitened positive residual excess.
if `e_k` is excess mass in band `k`, its effective coordinate is
`-log_phi(sum(e_k phi^-k) / sum(e_k))`. traversal gain multiplies that
coordinate before exponentiation, so the nominal stride is `phi^(gain * band)`.
plasticity changes response amplitude and never rescales the temporal coordinate.

## application reporting

the live report is expressed in source time. it includes prequential bits per source
byte, token loss and accuracy, source bytes and tokens per second, fixed batch size, prediction stride, io2 band, normalized residual surprise,
effective source gain, plasticity ratio, update rate, norm-match compensation, updates,
and session source experience.

the io2 residual bank uses exactly the predictor band count, keeping measurement
and representation on one temporal geometry.
its visualization contains one fewer passband than traces because every value is an
adjacent-trace difference. the dense float64 controller remains the reference path;
the existing fixed-width fused residual kernel is not used for this scalable bank.

periodic dreams are bounded token generations. diagnostic dreams snapshot and restore
the temporal trace and lived-token counter; stateful dreams are an explicit alternative.
