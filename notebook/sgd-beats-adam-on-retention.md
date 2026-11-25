# plain sgd forgets 2.5x less than adam at equal acquisition, and three benchmarks decided their own answers

james blight · 2026-09-16 · version 1 · result · verdict: alive

## current evidence note

editorial review 23 november 2026, for this specific notebook version. the original title and experiment text below are historical.

evaluation protocol: **continual_training_with_domain_probes**. two seeds; schedule preferences depend on benchmark and representation; three benchmark confounds documented; current somatic core uses adamw.

cost/accounting scope: no universal optimiser or hardware-efficiency claim; cost not itemised.

[structured evidence and available material](/api/evidence.json) · [developments](/developments)


## original summary

on a five-domain language stream (1.5m tokens, single pass, no replay, no task labels) plain sgd with one scalar per layer matched adam's acquisition and forgot 2.5x less (backward transfer 0.125 vs 0.314 bits); on a conflicting-worlds control it dominated on both axes. forgetting is a step to a floor rather than an erosion, so mechanisms that only change how far weights move cannot protect anything. three of the five benchmarks used were miscalibrated in ways that fixed their own answers; the lessons are recorded.

published: 2026-09-27 05:23:50.626165+00:00

evidence: 2-3 seeds per configuration · two benchmarks with opposite optimal schedules · not yet at scale

[latest version](/api/developments/sgd-beats-adam-on-retention.md)

session of 2026-09-15/16. everything below was measured in this directory
except the soma_spacetime work, which lives in
`~/spectral_v2/soma_spacetime/docs/transport_contraction_2026-09-15.md`.

read section 4 before trusting sections 2 and 3. three of the five benchmarks
used during this session were miscalibrated in ways that determined their own
answers, and the pattern is more useful than most of the results.

---

## 1. what holds

### 1.1 plain sgd beats adam on retention at equal acquisition

the most portable result of the session, and it does not depend on getting the
layerwise schedule right.

**five-domain language stream** (simplestories, conversation_worlds,
childrens_books, BNC, MovieCorpus; 3 cycles, 15 phases, 1.5M tokens, single
pass, no replay, no task labels, 2 seeds):

| | ACC | BWT |
|---|---|---|
| adam, base 3e-4 | 5.8482 ±0.0151 | 0.3137 ±0.0323 |
| sgd uniform, base 3e-4 | 5.8576 ±0.0165 | **0.1251 ±0.0165** |

same acquisition within noise, **2.5x less forgetting**. the gap is 0.189 bits
against a pooled standard error of 0.026.

**conflicting-worlds control** (section 4.3; 2 seeds):

| | PLAS | STAB |
|---|---|---|
| adam, base 3e-3 | 2.0394 ±0.0330 | 0.4751 ±0.0386 |
| sgd r=0.5, base 3e-3 | **1.9595 ±0.0924** | **0.2170 ±0.0523** |

dominates on both axes; the STAB gap is about 5 sigma.

two benchmarks, opposite optimal schedules, same conclusion: a per-parameter
second-moment normaliser costs roughly half the retention and buys nothing.
this is the claim worth scaling.

### 1.2 a per-parameter normaliser and a layerwise schedule are mutually exclusive

not measured, structural. adam rescales each parameter's step by its own
gradient history, so a layer set slow has its step scaled straight back up and
the schedule becomes decorative. if the learning rate is meant to be a unit of
time, adam is a clock that resets itself. every sgd run here uses **one scalar
per layer per step** (gradient rms) instead, which leaves relative gradients
inside a layer untouched and makes lr comparable across depth.

### 1.3 forgetting is a step to a floor, not an erosion

across every retention curve measured (`schedule_map.py` probes A repeatedly
through the B phase):

```
7.81 7.81 8.09 8.19 8.24 8.29 8.24 8.33 8.32 8.28 8.21
```

every configuration jumps within one or two probes and then sits. what a
schedule changes is **the height of the floor**, not the rate of decay.

this reframes what any mechanism can do. it is not damage accumulating that
could be slowed - it is the distance between where task A's solution sits and
where task B's solution sits. a mechanism can shorten that distance. it cannot
"protect" anything, because there is no ongoing erosion to protect against.
this is why every protection mechanism in section 2 failed, and why they had to.

### 1.4 the optimal schedule tracks where the learnable structure is

three benchmarks gave three different answers about the ordering:

| benchmark | input space | best ordering |
|---|---|---|
| language A->B, 60k tokens | 1024 tokens -> 64 dims, real similarity structure | input-fast (r>1) |
| language stream, 1.5M tokens | same | uniform / traded |
| conflicting worlds | 16 clean symbol indices, nothing to learn | readout-fast (r<1) |

that is not noise. where the input problem is hard, plasticity spent at the
embedding buys a lot; where the input is already a clean index, it is wasted and
the function lives entirely in the readout.

**there is no universal ratio.** the "optimal per-layer ratio ~4" figure from
mid-session is an artefact of one task at one depth and should not be carried
forward.

---

## 2. what does not hold

each of these was built, given a control that could kill it, and killed.

| mechanism | result |
|---|---|
| benna-fusi consolidation ladder behind each weight | lands exactly on the baseline's acquisition/forgetting frontier. it is a learning rate. a linear ladder low-passes the weight trajectory and cannot distinguish consistent evidence from inconsistent, which is the entire content of "consolidation". their capacity result needs a bounded stochastic synapse that does not survive transfer to a continuous weight. |
| per-weight coherence gating (omega = m^2/v, the local analogue of EWC/SI) | also on the frontier. omega measures 0.002-0.011 and **does not change across rho = 1e-2, 1e-3, 1e-4** - flat over three decades of averaging window. at rho=0.01 that matches the white-noise prediction rho/2 exactly; at 1e-4 it is 120x above it. so there is real per-weight drift, but it saturates at ~0.6% of the jitter and never cleans up. no importance statistic built by averaging the local coincidence has any purchase, at any timescale. |
| weight decay inside a simplex-projected field | strict no-op. bit-identical to baseline at all rates, because the projection renormalises it away on the same step. any consolidation mechanism that works by scaling the efficacy is dead inside a simplex constraint. |
| slow bands as broadcast gains (the neuromodulator attribution) | won by 0.085 nats at 6k tokens and reversed by 50k. |

the common shape: **all of these are isotropic in the sense that matters.** they
change how far weights move, not which way. forgetting is caused by moving in a
direction with a component along the old function, so a scalar - wherever it is
attached - slides along one line trading old loss for new gain. james's
counter-argument is correct that per-layer rates are a block-diagonal
preconditioner and therefore *do* rotate the joint direction; that is why
section 1.4 has a real effect where sections 2's mechanisms do not.

---

## 3. the numbers, with their scope

### 3.1 schedule map (unbounded, 60k tokens, A->B, 3 seeds)

`schedule_map.py`. parameters constant, scale decoupled from shape:
`lr_g = base * S^(g/(G-1) - 1/2)`, g=0 readout.

| S | per-layer ratio | B acquired | forgetting |
|---|---|---|---|
| 1 | 1.00 | 6.8413 ±0.0473 | 0.6858 ±0.0469 |
| 11.3 | 1.83 | 6.7742 ±0.0536 | 0.5881 ±0.0437 |
| 64 | 2.83 | 6.6590 ±0.0601 | 0.4777 ±0.0161 |
| 256 | 4.00 | 6.6021 ±0.0581 | 0.4237 ±0.0124 |
| 1024 | 5.66 | 6.6109 ±0.0413 | 0.4127 ±0.0501 |
| 4096 | 8.00 | 6.6562 ±0.0345 | 0.4631 ±0.0441 |

input-fast dominated uniform on both axes, with a located turnaround at ratio 8.
**scope: unbounded learner, one corpus pair, one shift, 60k tokens.** does not
reproduce under bounding + decay + constant parameters across depth (section 4.2).

### 3.2 schedule shape at matched total plasticity (3 seeds)

same endpoints, same geometric-mean rate, different distribution in between:

| shape | B acquired (base 3e-4) | forgetting |
|---|---|---|
| geometric (equal ratio per layer) | **5.8614 ±0.0163** | 0.6508 ±0.0700 |
| linear (equal difference) | 5.9072 ±0.0243 | 0.6518 ±0.0557 |
| step_input | 5.9365 ±0.0111 | 0.6322 ±0.0966 |
| step_readout | 5.9791 ±0.0288 | 0.6960 ±0.0511 |

geometric best on acquisition at both scales tested (t ~ 6 against step_readout
at 3e-4, directional at 1e-4), with forgetting indistinguishable across shapes.
`step_readout` - one slow readout, everything else uniformly fast - is the
*worst* shape, so this is not a two-speed system.

**this is the only evidence in the session that the lattice does work the
ordering alone does not.** it is also the least replicated: one spread value,
one corpus pair, three seeds.

### 3.3 total plasticity is not held constant by holding the mean rate constant

with rms-normalised steps the applied update for group g has l2 norm exactly
`lr_g * sqrt(n_g)`, so lifetime path length through parameter space is analytic:

| r | path/step vs r=1 |
|---|---|
| 1 | 1.00x |
| 2 | 1.27x |
| 4 | 3.10x |
| 8 | 9.93x |

**at fixed base, r=4 spends 3.1x the total weight movement of r=1.** any
fixed-base comparison across r is therefore unfair, and every such comparison
made before this was measured is suspect. `bounded_stream.py` and
`conflicting_worlds.py` record `path_per_step`; use it.

at matched path length on the language stream (`r=4@3e-4` path 1.900 vs
`r=1@1e-3` path 2.042), geometric dominated uniform by 0.29 bits acquisition
and 0.48 bits retention. at lower path lengths they merely trade.

---

## 4. the benchmark problem

the most useful output of the session. **three of five benchmarks decided their
own answers**, and the failures are instructive.

### 4.1 a combined metric locates the optimum wherever the easier term dominates

the first conflicting-worlds design reported ACC averaged over all worlds. but
conflicting contexts cannot be retained by a label-free learner at all, so ACC
was mostly a measurement of how well the *shared* structure was learned - which
low learning rates always win. uniform's optimum was pinned in the no-damage
regime **by construction**.

the fix is to never average acquisition and retention into one number:

- **PLAS** bits on the world just trained, measured right after training it
- **STAB** rise in bits on contexts every world agrees about

this applies backwards. the five-corpus language stream shows **negative BWT**
at low rates (-0.42, -0.34, -0.17): training later domains *improved* earlier
ones. five english corpora through one tokenizer share so much structure that
there is little to forget, so the low-rate half of that grid is not measuring
retention at all.

### 4.2 constraints change results

bounding and decay are not hygiene. the section 3.1 ridge was measured on an
**unbounded** learner; under row-norm bounding, decay, and parameters held
constant across depth, the optimum moved to r ~ 1-2 and nothing dominated.

but that test was also broken, in two ways worth recording:
- coupled decay (lambda_g proportional to lr_g) gave the embedding
  **0.05 decay per step** at G=7, r=8 - a twenty-step memory. the fast end was
  being erased as fast as it learned. cap it.
- a global radius of 1.0 bound only the embedding and crushed it 8x while four
  of five groups never approached it. that is a one-layer handicap masquerading
  as a global constraint.

james's framing is the right one: **bounding is a numerical guard against
precision breakdown, not a regulariser.** it should be per-layer, scaled to each
layer's own initialisation norm, generous enough never to bind in normal
operation, and instrumented. implemented that way (`language_stream.py`,
guard = 20x init row norm) it **bound zero times across 1.5M tokens** at every
configuration. costless and pointless in equal measure at this scale - which is
the correct behaviour for a guard.

### 4.3 conflict design biases the answer

permuted *labels* favour a fast readout. permuted *inputs* favour a fast
embedding. either decides the ordering question by construction.

`conflicting_worlds.py` conflicts in the output conditional over a shared,
trivial 16-symbol input space - which is structurally the permuted-labels case.
it returned readout-fast, as designed to. **the r=0.5 result in section 1.1 is
admissible as an adam comparison and not as evidence about ordering.**

a neutral benchmark needs conflict on both sides: domains that disagree about
the output mapping *and* about which inputs are equivalent. not yet built.

### 4.4 short horizons measure transient adaptation

three separate mechanisms won at 6k-60k tokens and reversed by 50k-1.5M: the
neuromodulator attribution, the layerwise ridge, and the first long-stream read.
**anything measured at 1e4-1e5 tokens on this problem is measuring adaptation
speed, not steady-state behaviour.**

### 4.5 verify the harness varied what you claim it varied

a seed loop that passed `--seeds 1` while looping in the shell reported "three
seeds" that were the same seed three times, caught only because the standard
deviations were exactly 0.0000. print realised per-layer rates and the seed in
every row.

---

## 5. reproduce

```bash
PY=~/spectral_v2/soma_v14/.packaging-venv/bin/python3   # torch 2.13, numpy 1.26

# conflicting-worlds control: PLAS/STAB frontier, ~7s per run
$PY experiments/conflicting_worlds.py --name test --V 16 --hidden 16 --d 16 \
    --r 1 --base 0.001 --conflict 0.75 --phase 80000 --held 20000 --cycles 3 \
    --out /tmp/t.json

# five-domain language stream: ACC/BWT, 1.5M tokens, ~140s per run
$PY experiments/language_stream.py --name test --r 1 --base 0.0003 \
    --phase 100000 --cycles 3 --out /tmp/t.json

# live report while a queue runs
experiments/report.sh --loop
```

`results/` holds the json for every run quoted above. machine constraint: 8 GB,
at most 2 torch or 4 numpy processes at once.

---

## 6. where to pick up

1. **scale the adam comparison.** section 1.1 is the result that does not depend
   on resolving the ordering, held on two benchmarks with opposite optimal
   schedules, and is a claim about standard practice. it needs a real model and
   a real streaming benchmark with published baselines.
2. **build the two-sided conflict benchmark** (4.3). until it exists, no ordering
   result is admissible.
3. **replicate the shape result** (3.2). it is the only evidence for a graded
   lattice and it rests on one spread value and three seeds.
4. **section 1.4 as the law**, if it survives: the schedule should follow where
   the function's difficulty sits, which is measurable per task rather than
   assumed. that is a weaker claim than a universal constant and a much more
   defensible one.


## attached evidence

- [conflicting_worlds.py](/api/developments/sgd-beats-adam-on-retention/files/conflicting_worlds.py?version=1) · code · 9477 bytes · sha256 `6d66dea771c5eaf4c52256004f33fd546f0e57a91d5236e0c07b91de6d25d408`
- [language_stream.py](/api/developments/sgd-beats-adam-on-retention/files/language_stream.py?version=1) · code · 8533 bytes · sha256 `312464ea29f5462a5ba3495314f58c095a51c641c14757973af68b271cb9dd73`
