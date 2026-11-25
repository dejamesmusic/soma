# stochastic transport collapses every long path to rank one

james blight · 2026-09-15 · version 1 · result · verdict: open

## current evidence note

editorial review 23 november 2026, for this specific notebook version. the original title and experiment text below are historical.

evaluation protocol: **analytical_and_synthetic, single_stream_prequential**. small controls and few seeds; hyperparameters incompletely swept; language correction necessary but not sufficient.

cost/accounting scope: no present-runtime efficiency claim.

[structured evidence and available material](/api/evidence.json) · [developments](/developments)


## original summary

nonnegative column-stochastic transport conserves mass while destroying identity: contrast decays as tau^L and the nonnegativity clamp absorbs support, so at initialization every path longer than two edges has numerical rank one. an identity-anchored correction restores full rank and recovers every synthetic lag and the de bruijn control, but does not move language, because on this corpus no single lag beyond three carries information. selecting transport by the arriving symbol takes a conjunctive task from chance to exact. transport conserves; boundaries compute.

published: 2026-09-27 05:23:40.728091+00:00

evidence: contraction: arithmetic, any seed · recoveries: small streams, 2-3 seeds · language: 382,066 ticks, seed 7

[latest version](/api/developments/transport-contraction.md)

this document reports a measurement, an explanation it forces on several
recorded null results, and a correction with its own controls. the measurement
is arithmetic rather than empirical: it reads the parameterization and needs no
corpus and no exposure.

open_questions.md item 1 already asks whether "positive
transport loses directional contrast before the relevant event can arrive", and
the reconstruction audit already
reports loss of directional contrast in trained long paths. this document
converts that suspicion into a number, states how large it is, and shows what
the field recovers when it is removed.

## 1. the quantity

for a nonnegative column-stochastic map `W` the dobrushin (ergodic) coefficient

```text
\tau(W)=\tfrac12\max_{i,j}\lVert We_i-We_j\rVert_1\in[0,1]
```

measures how much of the input's identity survives. `tau = 1` means every pair
of inputs stays exactly as far apart as it started; the map is a permutation of
coordinates and nothing is lost. `tau = 0` means every input produces the same
output: mass arrives, identity does not. the coefficient is submultiplicative,

```text
\tau(AB)\le\tau(A)\,\tau(B),
```

so a path of `L` learned stochastic edges loses contrast geometrically in its
own length.

this is the precise sense in which **conserving mass is not conserving
information**. for nonnegative stochastic transport the two are not merely
different, they are opposed: `l1` mass is preserved exactly at every depth
while contrast decays as `tau^L`. section 4 of
current_architecture.md establishes the first
property. this document measures the second.

`soma_spacetime/contraction.py` implements the measurement. it never
materializes a composed edge map; it propagates a sampled block of one-hot
columns along the path, so the sampled result is a lower bound on the true
coefficient, which is the safe direction.

## 2. the measurement

the four configurations recorded in
confirmed_results.md, "extended simplestories depth and
vocabulary suite", at their own seed and their own float32 runtime:

| configuration | width | rank | mean per-edge tau | first path below float32 epsilon |
|---|---|---|---|---|
| short_1k, paths 2,3,5,8 | 1024 | 32 | 0.052 | 5 |
| deep_1k, paths 2,3,5,8,13,21 | 1024 | 64 | 0.033 | 3 |
| medium_4k, paths 2,3,5,8,13 | 4096 | 32 | 0.051 | 3 |
| short_8k, paths 2,3,5,8 | 8192 | 32 | 0.051 | 5 |

float32 epsilon is `1.19e-07`. the deep_1k configuration, at initialization,
reported as the coefficient, the ratio of the second to the first singular
value of the composed path map, and its numerical rank:

```
path  2   tau 4.56e-05   s2/s1 1.59e-05   rank 56
path  3   tau 6.34e-08   s2/s1 2.46e-08   rank  1
path  5   tau 1.30e-13   s2/s1 5.12e-14   rank  1
path  8   tau 2.16e-16   s2/s1 9.70e-16   rank  1
path 13   tau 2.38e-16   s2/s1 9.88e-16   rank  1
path 21   tau 2.54e-16   s2/s1 7.70e-16   rank  1
```

**every path longer than two edges has numerical rank one.** the composed map
is `pi g^T`: one fixed output direction with a per-symbol scalar gain. such a
path still delivers mass, and that mass still varies with the input, but it can
report only a scalar - never which of 1024 symbols arrived. the reverse field
along it is the transpose of the same rank-one map, so the local coincidence it
produces is a fixed outer product with no symbol-specific direction either. the
path cannot transmit, therefore cannot learn, therefore cannot learn to
transmit.

`2.2e-16` is float64 epsilon: composed in the runtime's own float32 these
products underflow entirely.

### a second, independent failure: absorbed inputs

weights are nonnegative and the projection clamps at zero. zero is an absorbing
state for a nonnegative coordinate whose gradient is proportional to the
activation it carries, so each projection is a ratchet that can only remove
support. training a width-128 rank-32 field for 6,000 ticks and measuring the
fraction of input symbols whose column carries no mass at all down each path:

| tick | L=2 | L=3 | L=5 | L=8 | L=13 | L=21 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0% | 0% | 0% | 0% | 0% |
| 500 | 56% | 37% | 50% | 51% | 36% | 44% |
| 2000 | 62% | 55% | 53% | 59% | 54% | 53% |
| 6000 | 64% | 50% | 52% | 52% | 56% | 57% |

over half the vocabulary stops being transported at all, on every path, within
a few thousand ticks. the same run's coefficient for path 2 degrades from
`3.0e-04` to `2.0e-16`: under training even the shortest long path ends at
rank one.

the instantaneous rank factorization is the main cause. an unfactorized dense
edge has `tau` near `0.37` at width 1024; factorizing it as two stochastic maps
through a rank-32 bottleneck drops it to `0.05`. factorization was verified
temporally neutral - it inserts no node and no tick - but it is not neutral in
contrast, and that was not measured.

## 3. what this forces

every depth comparison in this repository was made between models whose long
paths transmitted nothing. specifically:

- the headline null, "the two 1k curves remained essentially identical over
  382,066 observations", compared a model in which paths 5 and 8 are below
  runtime epsilon against one in which paths 3 through 21 are. the extra depth
  in the deeper model was not weakly useful; it was arithmetically absent.
- "its combined local evidence was only `0.033`, against `0.774` for the signed
  fast pair", and the per-path evidence `0.00268, 0.00178, 0.00160, 0.00146`
  decreasing in length, is the same decay read off the loss.
- "the positive long field carries activation while losing most of its
  directional contrast" is exactly `mass = 1, tau = 0`.
- the de bruijn probe, the lag-7 and lag-13 memory probes, and the
  disappearance of the lag-7 advantage at 100,000 steps are all the same fact.
- the three rejected bounded-negation variants, the inverse rail and the fixed
  identity mixtures do not act on `tau`, so none of them could have helped.
- the fast signed pair dominates because it is the only path of length one, and
  therefore the only one that pays `tau` once.

these are not results about temporal distance. they are results about a
parameterization. **the architecture's central claim has not yet been tested.**

## 4. the correction

spend the contrast budget once per path instead of once per edge. on every
interior edge - every edge except the one entering the output boundary -

```text
W_e \;\rightarrow\; (1-\varepsilon_e)\,\frac{I}{n_{\text{out}}(\text{src})}
\;+\;\varepsilon_e\,W_e .
```

this is a convex combination of two valid outgoing-simplex configurations, so
the joint outgoing simplex, forward `l1` conservation and reverse `l-infinity`
nonexpansiveness hold exactly and unchanged. the anchor is a scaled identity,
so it is its own transpose and the reverse field is treated identically for
free. there is no extra matmul and no new parameter: one scaled add per
direction. the factors sit behind the blend, so the chain rule carries
`epsilon` into both factor gradients, and transport learns slowly while the
exit edge learns at full rate.

setting `eps_e = b/(L-1)` for a path of length `L` holds the per-path
coefficient near `exp(-b)` independent of distance. at `b = 1`, width 1024,
rank 64:

```
path    2   3.17e-02
path    3   1.55e-02
path    5   1.34e-02
path    8   1.22e-02
path   13   1.22e-02
path   21   1.17e-02        invariants 2.4e-07
```

flat across every distance, against `2.5e-16` before, and every path keeps its
full numerical rank 64 instead of rank 1. the anchor is also a support floor:
because an identity component guarantees every input coordinate retains mass,
the projection cannot clamp a symbol out of the geometry. the same 6,000-tick
training run that drops 50-64% of inputs under the implemented parameterization
drops **0% at every path throughout**, while its coefficient stays flat and
drifts slightly upward:

| tick | L=2 | L=3 | L=5 | L=8 | L=13 | L=21 |
|---|---|---|---|---|---|---|
| 0 | 5.4e-02 | 2.7e-02 | 2.1e-02 | 2.1e-02 | 1.9e-02 | 1.9e-02 |
| 6000 | 1.2e-01 | 3.1e-02 | 2.2e-02 | 2.1e-02 | 1.9e-02 | 1.7e-02 |

### how far this holds under training

measured with the corrected instrument at the scale configuration - width 1024,
rank 64, paths `2,3,5,8,13,21`, 400,000 simplestories bytes, about 153,000
ticks, admission on, `lr = 0.3`:

| | tau per path | numerical rank per path | inputs dropped |
|---|---|---|---|
| implemented, after training | `3e-16` throughout | `1,1,1,1,1,1` | 31%, 16%, 8%, 16%, 6%, 16% |
| conserving, after training | `3.8e-03` to `2.1e-05` | `64,63,42,38,35,35` | `0%` at every path |

the correction holds in the sense that matters: rank 35-64 against rank one,
no inputs dropped against 6-31% dropped, and a coefficient four to eleven
orders of magnitude higher. under the implemented parameterization **even path
2 finishes at rank one** on real tokens.

it does not hold perfectly. the conserving field's coefficient falls from
`1.2e-02` at initialization to `2.1e-05` at path 21, and its rank erodes from
64 to 35. so the identity anchor is a floor on contraction and an exact floor
on support, but it is **not** a floor on rank: long exposure still erodes
degrees of freedom, just far more slowly. a larger budget, or a rank floor in
the projection itself, is the obvious next thing to test and has not been
tested. an earlier 6,000-tick measurement at width 128 showed no decay at all,
which was simply too short to see this.

`eps = 1` recovers the
implemented field exactly: over 500 learning ticks the subclass and the
original agree to `0.0e+00` on probability, both factors and both field states.
`soma_spacetime/conserving.py` implements it;
`experiments/transport_contraction_audit.py` reproduces the table.

## 5. what the correction recovers

### synthetic lag, positive field, identical geometry

vocabulary 8, paths `2,3,4,6,9`, a codeword stream of period `L` resampled every
four periods so the lag is required and cannot be memorized. chance is `3.000`
bits and accuracy `0.125`; the ceiling is `0.750` bits and accuracy `0.750`.
three seeds, 6,000 ticks, the same counterpropagating field and local
coincidence law throughout.

| eps | lag 2 acc | lag 4 acc | lag 9 acc |
|---|---|---|---|
| 1.00 (implemented) | 0.698 | 0.256 | 0.200 |
| 0.50 | 0.740 | 0.736 | 0.760 |
| 0.25 | 0.742 | 0.741 | 0.765 |
| 0.10 | 0.741 | 0.745 | 0.769 |
| 0.00 (pure delay) | 0.740 | 0.746 | 0.769 |

the implemented transport solves lag 2 and dies from lag 4 onward. every
`eps < 1` reaches the ceiling at every lag tested.

the control that isolates the cause freezes the interior instead of learning
it. freezing it at its dense random initialization stays at chance; freezing it
at the identity reaches the ceiling. the difference is not learnability, it is
`tau`.

### the repository's own de bruijn control

from confirmed_results.md: "the positive temporal field
failed to distinguish correct length-eight from wrong length-seven paths even
with admission disabled. this is a capacity result for that positive field".

de bruijn-3, `0,0,0,1,0,1,1,1`, period 8 with exactly balanced one-step
successors. admission disabled, three seeds, 40,000 ticks.

| transport | paths `2,4,8` | paths `2,4,7` | separation |
|---|---|---|---|
| implemented | 1.5956 | 1.6574 | 0.062 |
| conserving, budget 1 | 1.3414 | 1.5420 | 0.201 |
| conserving, eps 0.1 | 1.3387 | 1.5368 | 0.198 |

the capacity result was a contrast result.

## 5b. the passing control was already the corrected architecture

the delayed-xor control in
confirmed_results.md is the strongest positive result
in the repository: 27 of 27 configurations with the required history present
reached 100% held-out accuracy, worst case `0.010 bpb`, while all 21 without it
stayed at chance. the reconstruction audit
records how it was actually run: it "stores literal one-hot delay slots and
trains a gelu reader over all slots".

literal one-hot delay slots are transport at `tau = 1`. a reader over all slots
is computation at the boundary. that control is therefore not evidence about
the positive stochastic field at all - it is a measurement of the corrected
architecture, and it passed cleanly. the same audit states the consequence
directly: "extra positive transport steps alone do not supply the
compositional operation tested in the earliest tanh/gelu controls."

the two facts belong together. where distance was carried losslessly and the
computation was done at the boundary, the object worked at `0.010 bpb`. where
the computation was distributed into the transport, the object reached chance.

## 6. the readout is a second, separate boundary

conserving transport lets a path carry a symbol. it does not let the output
prefer that path. the joint outgoing simplex divides input mass among `P`
paths and the output is a normalized sum, so a single correct path is bounded
at `p(correct) <= 1/P + (1-1/P)/V` however good it is. accuracy reaches its
ceiling while confidence cannot.

a per-path scalar on the exit map fixes this, and **where it reads matters more
than what it does**. at a path's entry the forward field is the symbol arriving
now and the reverse field is the residual from `L-1` ticks ago: the two are
`2L-1` events apart, the coincidence is causally misaligned, and a gate driven
from it selects the wrong path and makes the result worse. at the exit the
forward field left the input `L-1` ticks ago and the reverse field is the
current residual. these are the aligned pair, and

```text
\frac{\partial \mathcal L}{\partial\beta_p}=\langle r_t,\;S_p f_p\rangle
```

is exactly path `p`'s contribution to the present error. vocabulary 8, paths
`2,3,5,8,13`, two seeds:

| gate | lag 5 bits | mass share | lag 13 bits | mass share |
|---|---|---|---|---|
| none | 2.115 | uniform | 2.033 | uniform |
| at the entry | 2.309 | wrong path | 2.116 | wrong path |
| at the exit | 1.551 | 0.93 on path 5 | 1.567 | 0.94 on path 13 |

on the de bruijn control with both corrections, 40,000 ticks:

| | paths `2,4,8` | paths `2,4,7` | separation |
|---|---|---|---|
| implemented | 1.5968 | 1.6632 | 0.066 |
| conserving | 1.3414 | 1.5420 | 0.201 |
| conserving + exit gate | 0.8217 | 1.4954 | 0.674 |

with the correct distance available the gate places `2.997` of `3.0` on the
length-8 path; with only wrong distances it falls back to length 2 and reports
that no path carries the structure. the gate is therefore also a readable
instrument for which distances a corpus actually uses.

what the exit gate preserves: `sum(beta) = P`, so each path delivers `beta_p/P`
and total forward mass arriving at the output is still exactly 1. what it
relaxes: the per-source outgoing simplex at the exit nodes, and with it the
reverse `l-infinity` bound, by at most `max(beta)`. this is the one invariant
traded, and it is traded for the only place the field can express confidence
rather than direction alone.

## 7. boundary of interpretation

- the contraction measurement is arithmetic and holds for any seed and any
  exposure. it is not an empirical claim.
- the synthetic and de bruijn recoveries are measured on small streams with
  few seeds.
- `b = 1` and `eps` were not tuned; a sweep over the contrast budget has not
  been run.
- the exit gate's learning rate was not tuned, and its interaction with the
  admission gate is untested.
- the language results are reported in section 8 and are weaker than the
  synthetic ones; read that section before treating the correction as
  sufficient.

## 8. language: the correction is necessary and it is not sufficient

the same protocol as the recorded simplestories suite: first one million
simplestories bytes, 1k tokenizer, 382,066 ticks, `lr = 0.3`, summed updates at
a fixed interval of 55, seed 7, float32 on mps. the only variable is the
interior transport. `eps = 1` reproduces the implemented field bit-for-bit:
over 500 learning ticks the two agree to `0.0e+00` on probability, both factors
and both field states.

| model | paths | parameters | bpb | final ten windows |
|---|---|---|---|---|
| deep, implemented | 2,3,5,8,13,21 | 8.91m | `3.48758` | `3.47987` |
| deep, conserving | 2,3,5,8,13,21 | 8.91m | `3.48756` | `3.48006` |
| deep, conserving + exit gate | 2,3,5,8,13,21 | 8.91m | `3.48755` | `3.48007` |
| short, conserving + exit gate | 2,3,5,8 | 4.46m | `3.48755` | `3.47983` |

four runs spanning both corrections and both depths lie within `0.00003 bpb`
of each other, and the 4.46m short model matches the 8.91m deep one. two
further cells of the two-by-two (deep implemented with the gate, short
conserving without it) were started and then killed to keep this machine inside
its two-process memory budget; the corpus measurement below makes them
uninformative.

the exit gate's own reading is the most useful line in the table. on the de
bruijn control it places `2.997` of `3.0` on the length-8 path. here it settles
at `1.26, 0.66, 1.09, 0.98, 0.99, 1.02` - very nearly uniform, with `sum(beta)`
exact to `0.0`. asked which distance carries structure in this corpus, the gate
answers "none of them", and section 8.1 shows it is right.

the recorded deep_1k result on this protocol was `3.48950`, so the baseline
reproduces. the correction changes the fifth decimal place and nothing else.
mean long-path evidence, by path:

```
implemented   0.000653 0.000458 0.00109 0.000874 0.000901 0.000902
conserving    0.001023 0.000773 0.000979 0.000945 0.000947 0.000956
```

restoring full rank on every path raised per-path evidence by well under a
thousandth of a nat and did not move the loss. **this is a real negative
result and it is not explained by transport.**

### 8.1 why: the corpus has nothing at those lags

`experiments/corpus_lag_information.py` trains no field. it fits
`P(w_t | w_{t-k})` on a training prefix and measures held-out bits, which
bounds what any path of length `k` can contribute to a readout that is additive
over lags, however perfect its transport.

| lag | held-out bits | gain over unigram |
|---|---|---|
| 1 | 4.9976 | **+3.1993** |
| 2 | 6.8148 | +1.3821 |
| 3 | 7.7252 | +0.4717 |
| 5 | 8.4060 | −0.2091 |
| 8 | 8.6193 | −0.4224 |
| 13 | 8.7022 | −0.5053 |
| 21 | 8.7285 | −0.5316 |

beyond lag three, conditioning on a single lag is *worse* than not conditioning
at all: the table is pure estimation noise. and the conjunctive question, with
the estimator given every advantage - backed off to the lag-1 model, mixing
weight chosen on held-out data, 79-96% of contexts seen:

| second lag, given lag 1 | best lambda | gain |
|---|---|---|
| 2 | 0.40 | +0.1655 |
| 3 | 0.10 | +0.0127 |
| 5 | 0.00 | 0.0000 |
| 8 | 0.00 | 0.0000 |
| 13 | 0.00 | 0.0000 |
| 21 | 0.00 | 0.0000 |

the optimal weight on the second lag is exactly zero from lag five onward.

so the recorded depth null was over-determined. the transport was rank one,
**and** on this corpus at this exposure there was nothing at those distances
for a lag-additive readout to collect. fixing one without the other cannot
move the loss, which is exactly what the two runs above show.

this converts open question 1 from a search into a measurement. run the corpus
probe first; only spend exposure on depth where the number is not zero. the
probe takes seconds and would have preempted the 382,066-tick suite.

the caveat matters and cuts the other way: this estimator is pairwise. genuine
long-range structure in language is not a relation between two token positions,
it is a relation between the next token and an accumulated state. the table
bounds what *this architecture's representation class* can collect. it does not
bound language.

## 9. what this says the architecture is

three of the four parts are load-bearing and survive everything measured here.

- **the learner.** counterpropagating fields with a local coincidence law,
  verified against autograd to `1.4e-15`, with no stored activations, no
  unrolling and no causal identifiers. nothing in this document touches it.
- **the clock.** distance through the graph is elapsed time. the delayed-xor
  control settles this: 27 of 27 configurations with the history present reach
  100%, all 21 without it stay at chance.
- **the spacing.** phi, because `phi^2 = phi + 1` is the unique point where a
  geometric progression is also an additive recurrence, so the delay set is at
  once maximally non-resonant between scales and exactly compositional along
  them.

the fourth part is where the design went wrong, and it went wrong in a specific
and correctable way. nonnegative column-stochastic edge maps conserve `l1` mass
exactly, and that is what made the invariants clean. but for such maps mass and
information are not merely different quantities, they are opposed: contrast
decays as `tau^L` while mass is preserved exactly, and the nonnegativity clamp
is an absorbing ratchet that removes support. the architecture spent its
information budget on transport, where it buys nothing, and arrived at the
boundary with rank one.

the principle the measurements point to:

> **transport conserves. boundaries compute.**

a path's job is to deliver the past intact. contrast is a budget and it should
be spent once per path, at the boundary where the forward and reverse fields
are causally aligned - which is the exit, and only the exit. the passing
delayed-xor control was already built this way and nobody noticed: literal
one-hot delay slots are `tau = 1` transport, and a reader over all slots is
boundary computation. it reached `0.010 bpb` where the positive field reaches
chance.

concretely, the object underneath all of this is:

1. a phi-spaced set of contrast-preserving delays, anchored near the identity so
   that neither composition nor the nonnegativity clamp can erase a symbol;
2. one learned map per delay, at its exit;
3. one learned scalar per delay, also at its exit, so the output can express
   which distance it believes rather than averaging all of them;
4. the counterpropagating local coincidence rule as the learner, unchanged.

### the ceiling this leaves, and the one operation that escapes it

with 1-4 the model is `p_hat ~ sum_L beta_L M_L e_{v(t-L)}`: a learned mixture
of lag-conditional maps. it is additive over lags, so it is a bag of lagged
unigrams and it cannot represent a conjunction. section 8 measured the exact
size of that ceiling on this corpus, and it is reached by lags 1, 2 and 3.

the one operation that escapes it, and the one that every other part of this
architecture is already compatible with, is to make the transport map depend on
what is arriving:

```text
W \;\rightarrow\; W_{v_t}.
```

a selection among column-stochastic matrices is still column-stochastic, so
nonnegativity, `l1` conservation and the reverse bound all survive untouched,
and the content at depth `k` becomes a product of input-selected operators - a
conjunction of the last `k` symbols rather than their superposition. this is
the same move as the kronecker field operator that was alive in the trace-bank
work.

this one is measured too. `experiments/selective_transport_probe.py` runs the
lfsr `x_t = x_(t-2) XOR x_(t-5)`, period 31, in which every single lag is
exactly balanced: no lag alone predicts anything and only the conjunction of
lags 2 and 5 does, so a readout additive over lags is pinned at `1.000` bits by
construction however perfect its transport. paths `2,3,5`, width 2, conserving
transport and the exit gate in every condition, three seeds:

| transport | bits | accuracy |
|---|---|---|
| additive, eps = 0 | 1.0229 | 0.338 |
| additive, eps = 1 | 1.6793 | 0.484 |
| selective by local content, eps = 1 | 1.4652 | 0.482 |
| selective by local content, eps = 0.5 | 0.9919 | 0.570 |
| **selective by arriving symbol, eps = 1** | **0.0001** | **1.0000** |
| **selective by arriving symbol, eps = 0.5** | **0.0000** | **1.0000** |

chance to exact, on every seed. one detail matters and is not cosmetic:
selecting by the content already on the edge does **not** work, because after
one map that content is no longer a clean symbol index. selecting by the symbol
arriving at the input boundary does. that is a third boundary broadcast, which
is a genuine addition to the architecture's locality - though it is the same
kind of object as the residual already broadcast to every output edge and the
one-hot already broadcast to every input edge, rather than a new kind.

note also that the exit gate stays near uniform here (`1.09, 0.68, 1.24`): the
conjunction is distributed across paths rather than carried by one of them, so
the gate correctly declines to choose. the two mechanisms are independent.

the order is not negotiable. a multiplicative operation on a substrate that
transmits rank one is worthless, which is why every earlier attempt at
composition and at bounded negation failed without saying why. fix
transmission, then let the boundary weight what it receives, then multiply.


## attached evidence

- [conserving.py](/api/developments/transport-contraction/files/conserving.py?version=1) · code · 11293 bytes · sha256 `5badcd468f3bb7fafca7198bb4ee6b21752be4f2d3a169e4352305f49d895cee`
- [contraction.py](/api/developments/transport-contraction/files/contraction.py?version=1) · code · 11813 bytes · sha256 `dd45ec577a0d00fcac6dde2a7ff2968d54441e02057806b53efa1c5bb8dc5bdc`
- [corpus_lag_information.py](/api/developments/transport-contraction/files/corpus_lag_information.py?version=1) · code · 4215 bytes · sha256 `9006c82e110a789d2d12207c1709795dce65071694109d359fe87a7eac747b2f`
- [selective_transport_probe.py](/api/developments/transport-contraction/files/selective_transport_probe.py?version=1) · code · 7448 bytes · sha256 `a7905facfcac21fa9be06a6a5eed057403c9c898b278067eb6d6cd9fe24a3bf6`
- [transport_contraction_audit.py](/api/developments/transport-contraction/files/transport_contraction_audit.py?version=1) · code · 2853 bytes · sha256 `f6199fd3d3a9672a9fa0110c8fec4d56b2e0f2589ec77d32ec92a7147ab53143`
