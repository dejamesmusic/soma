# a residual component learner selects the one predictive distinction and keeps it

james blight · 2026-09-20 · version 1 · result · verdict: alive

## current evidence note

editorial review 23 november 2026, for this specific notebook version. the original title and experiment text below are historical.

evaluation protocol: **synthetic_prequential_selection_and_held_out**. estimator-rate floor; zero original null admissions is not nominal calibration; coordinate-dependent sparsity; no demonstrated automatic open-ended measurement construction.

cost/accounting scope: small finite dictionaries; candidate shadow work O(M) per event, not open-ended language-scale compute.

[structured evidence and available material](/api/evidence.json) · [developments](/developments)


## original summary

a globally evaluated, non-network learner admits a component only when it buys future prequential code. on transparent binary streams it selected the single generating distinction in 5/5 seeds, made no false admission among up to 528 simultaneous noise candidates in 60 null runs, and returned to a previous regime 0.314 bits per event cheaper than a fresh learner (95% interval 0.287 to 0.341), where dense controls transferred nothing or negatively.

published: 2026-09-27 05:23:47.356718+00:00

evidence: 5-20 seeds per condition · small binary sources · components drawn from a declared finite dictionary

[latest version](/api/developments/residual-component-learner.md)

## result in one sentence

a globally evaluated, non-network component learner identified the single predictive
distinction in transparent streams, rejected up to 528 simultaneous noise candidates,
and retained useful information across an a-b-a intervention.

## selection and intervention

five seeds used 20,000 training and 10,000 held-out symbols. the dictionary contained
eight exact lags and all 28 pair products. the residual learner admitted one component
on each structured source and none on the null.

| source | selected component | selection | held-out bits | accuracy |
|---|---|---:|---:|---:|
| independent noise | none | 5/5 | 1.005 | 49.7% |
| noisy lag 1 | `lag_1` | 5/5 | 0.402 | 92.1% |
| noisy parity 1,2 | `lag_1*lag_2` | 5/5 | 0.399 | 92.1% |
| noisy parity 2,5 | `lag_2*lag_5` | 5/5 | 0.400 | 92.1% |

the admission decision used future prequential code gain. residual correlation remained
a proposal signal and diagnostic; it did not authorize learning.

## family-size stress

twenty independent null streams were tested at each dictionary size.

| candidates available globally | false admissions | held-out bits |
|---:|---:|---:|
| 36 | 0/20 runs | 1.004 |
| 136 | 0/20 runs | 1.003 |
| 528 | 0/20 runs | 1.005 |

this is empirical support rather than a universal false-discovery guarantee. longer
horizons, repeated residual resets, dependent candidate families, and adversarial nulls
remain to be tested.

## component availability

a cubic autoregressive parity source was tested against quadratic and cubic dictionaries.
without the generating cubic measurement, the learner found real lower-order shadows
created by the recurrence, usually `lag_7` and sometimes `lag_2*lag_3`, and reached 0.748
held-out bits. with cubic measurements available it selected `lag_1*lag_2*lag_4` first in
10/10 runs and reached 0.407 bits.

global error can select among retained distinctions. it cannot directly recover a
distinction absent from the measurement algebra, although the stream may contain
predictive surrogate distinctions. open-ended composition therefore remains a central
problem.

## literal binary event basis

the initial suite used signed coordinates `s=2x-1`. this is invertible, but it makes
some boolean relations artificially sparse. the full suite was therefore repeated with
literal arrival indicators `x in {0,1}` as the primitive features.

| source | held-out bits | admitted components |
|---|---:|---:|
| independent noise | 1.005 | 0 |
| noisy lag 1 | 0.400 | 2 |
| noisy parity 1,2 | 0.430 | 5--6 |
| noisy parity 2,5 | 0.400 | 3 |

delayed parity consistently admitted `x_2*x_5`, `x_2`, and `x_5`, the three nonconstant
terms needed to express the boolean relation in arrival coordinates. the signed version
expressed the same relation as one product. the result survives, but component sparsity
is coordinate-dependent. event arrival at a declared tick is therefore the theorem's
primitive; signed contrasts are derived measurements whose convenience must be stated.

## a-b-a retention

ten runs used 8,000-symbol segments: noisy lag 1, noisy parity 2,5, then noisy lag 1.
return savings compare the first 512 symbols of the returning a segment with an identical
fresh learner on that same segment.

| learner | a return | fresh a | return savings |
|---|---:|---:|---:|
| dense atomic | 0.458 | 0.456 | -0.002 bits/token |
| dense quadratic | 0.561 | 0.491 | -0.069 bits/token |
| residual components | 0.526 | 0.841 | **0.314 bits/token** |

the residual learner retained both true components in all 10 runs. its a-return cost was
higher than its settled cost of 0.413 bits, so live coefficients still suffered partial
interference. the result supports retained structure and return savings, not perfect
retention.

across the ten paired runs, residual-component return savings had standard deviation
0.044 and a 95% mean interval of `[0.287, 0.341]` bits per event. the dense atomic interval
included zero; the dense quadratic control showed reliably negative transfer.

## power curve

the one-component signed-coordinate test swept corruption toward chance over 20 seeds
per condition and 20,000 observations per run. available information is
`n * (1 - h2(noise))`.

| noise | effect `1-2e` | available bits | atomic detection | parity detection |
|---:|---:|---:|---:|---:|
| 0.38 | 0.24 | 839 | 20/20 | 20/20 |
| 0.42 | 0.16 | 371 | 20/20 | 20/20 |
| 0.45 | 0.10 | 145 | 20/20 | 20/20 |
| 0.47 | 0.06 | 52 | 0/20 | 6/20 |
| 0.48 | 0.04 | 23 | 0/20 | 0/20 |
| 0.49 | 0.02 | 6 | 0/20 | 0/20 |
| 0.50 | 0.00 | 0 | 0/20 | 0/20 |

there were no irrelevant admissions in the 400 structured power runs. detection delay
rose from a median 384 observations at 8% noise to 4,288--6,304 observations at 45%
noise. the transition between 52 and 145 available bits demonstrates finite power rather
than a rule that merely rejects everything.

at 45% noise the selected component was real, but held-out loss remained near one bit.
the fixed active-component learning rate is too noisy in this regime. component discovery
and coefficient tracking are therefore distinct mechanisms; passing the admission test
does not guarantee that a fixed-rate online coefficient will exploit a weak component.

### estimator floor

the apparent power cliff was primarily a constant-step estimator floor. at 6% effect,
changing only the shadow coefficient rate produced:

| shadow rate | atomic detection | parity detection |
|---:|---:|---:|
| 0.025 | 0/20 | 5/20 |
| 0.0125 | 5/20 | 18/20 |
| 0.00625 | 15/20 | 18/20 |
| 0.003125 | 16/20 | 20/20 |

with 200,000 observations, the atomic component remained 0/10 at rate 0.025 and reached
10/10 at 0.00625. this confirms an asymptotic regret floor at the original rate rather
than insufficient exposure. parity eventually reached 10/10 at the original rate, but
its median admission was 17,568 observations versus 6,400 at the lower rate.

polyak averaging the fast shadow coefficient improved detection at the unchanged 0.025
rate: atomic 6% detection rose from 1/30 to 12/30 and parity from 15/30 to 22/30. at 4%
effect, parity rose from 0/30 to 7/30. it reduces the floor but does not remove the need
to tune or automate estimator timescale.

the family-alpha sweep showed that the declared penalty is not setting the current
frontier. alpha values `0.01, 0.05, 0.1, 0.5` each produced zero admissions in 50 null
runs for both fast and polyak shadows. this is evidence of conservative miscalibration,
not successful nominal type-i calibration.

### no-regret shadow and penalty ablation

a scalar online-newton shadow removes the arbitrary constant-step clock. at unchanged
penalty it produced:

| effect | fast sgd | polyak sgd | online newton |
|---:|---:|---:|---:|
| atomic 6% | 1/30 | 12/30 | 24/30 |
| atomic 4% | 0/30 | 1/30 | 3/30 |
| parity 6% | 15/30 | 22/30 | 30/30 |
| parity 4% | 0/30 | 7/30 | 19/30 |

newton also made the family price observable: with the half-log parameter surcharge,
alpha 0.5 produced 3 admissions in 50 null runs while lower alpha values produced none.

removing the explicit `0.5 log2(1+n)` surcharge increased atomic 6% power from 25/30 to
28/30 and parity 4% power from 10/30 to 24/30 on a fresh seed set. at alpha 0.01 the null
changed from 0/50 to 1/50; at alpha 0.5 it changed from 1/50 to 11/50. online-newton
regret already pays much of the scalar estimation cost, so the extra half-log term is
substantially double charging it. exact nominal control remains unproven: a proper
mixture likelihood or e-process should replace the current mdL-style approximation.

### averaging and retention

polyak coefficient shadows slightly improved stationary held-out loss: 0.4039 to 0.4018
at strong signal and 1.0014 to 0.9932 at 10% effect. unconditional averaging failed on
a-b-a. returning-a loss was 1.242 for the global average versus 0.563 for the fast
coefficient, and its late loss remained 0.622 versus 0.412.

a global slow coefficient blends incompatible regimes. slow state can preserve a
component opinion, but it must be selected or conditioned by current evidence rather
than used unconditionally. discovery averaging and continual retention are related
timescale problems, not one interchangeable estimator.

## candidate compute

the current implementation maintains a counterfactual predictor for every inactive
candidate, requiring `O(M)` state and work per event.

| candidates | terms | ticks/second |
|---:|---|---:|
| 36 | pairwise, 8 lags | 70,008 |
| 528 | pairwise, 32 lags | 41,636 |
| 2,080 | pairwise, 64 lags | 18,313 |
| 5,488 | cubic, 32 lags | 7,762 |

this is already load-bearing before token identities and temporal scales multiply the
dictionary. exhaustive shadow prediction cannot construct an open-ended measurement
algebra. candidate_search.md records the staged alternative and
the pure-synergy constraint it must preserve.

the first proposal-recall control found that the original coherence ema was too fast.
at 6% effect its effective 200-event horizon kept the truth in a beam of one for only
6--8% of checkpoints. cumulative coherence raised occupancy to 85% for the atomic rule
and 92% for parity, with 100% final top-one recall. at 4% effect a cumulative beam of four
had 100% final recall on both tasks. proposal screening therefore needs evidence clocks
commensurate with authorization; nonstationary use will require a small bank of proposal
horizons rather than one ema or one permanent cumulative score.

## claim boundary

the experiments establish a mechanism on small binary sources where the relevant
components exist in a declared finite dictionary. they do not establish that this learner
can construct its own adequate measurement algebra, scale candidate search, solve
multiclass normalization, or compete on language. the dense controls also use a common
fixed sgd rate and are diagnostic rather than optimally tuned baselines.

the next decisive experiment is temporal measurement: hold this admission law fixed and
compare exact lags, adjacent differences, linear delay scales, powers of two, fibonacci
delays, and phi fading traces. only that matched test can connect this learner back to the
distinction theorem's geometric claim.


## attached evidence

- [benchmark_candidates.py](/api/developments/residual-component-learner/files/benchmark_candidates.py?version=1) · code · 1639 bytes · sha256 `1adb3fbf7ebb6d4f9137835f363995cec32488c9f399fb4a59df666b1b624c2b`
- [run_beam_recall.py](/api/developments/residual-component-learner/files/run_beam_recall.py?version=1) · code · 3631 bytes · sha256 `c162bd93abbeb523a897ac24f09c1c58cea29fbd46a929b02981d1e9f5186a73`
- [run_binary_basis.py](/api/developments/residual-component-learner/files/run_binary_basis.py?version=1) · code · 1866 bytes · sha256 `5a4c03bc61c849d0fd631bd2799a3b046a8a7eadf6734a06b865bf1e776960e1`
- [run_continual.py](/api/developments/residual-component-learner/files/run_continual.py?version=1) · code · 3921 bytes · sha256 `2176b19635bc8233129b6854e3f8f12e8994d449a7c282b1b278a4df85020fa5`
- [run_estimator_sweep.py](/api/developments/residual-component-learner/files/run_estimator_sweep.py?version=1) · code · 3524 bytes · sha256 `5e531e22742bcff474e90d5b119056258998de427b862cb5073b981fc65599d8`
- [run_penalty_ablation.py](/api/developments/residual-component-learner/files/run_penalty_ablation.py?version=1) · code · 2251 bytes · sha256 `58cdefa0c8b07d87c4d6afc1dbdf6bb50aaf95cef937e8184619c26f359567db`
- [run_polyak.py](/api/developments/residual-component-learner/files/run_polyak.py?version=1) · code · 2795 bytes · sha256 `0e726fb7a9dc74a0cdeef549dc0149c126d413d0384ca75f716faee08ffb5d54`
- [run_power.py](/api/developments/residual-component-learner/files/run_power.py?version=1) · code · 5810 bytes · sha256 `328a682392b69b1e1a3dcb4585b7522c78f9659e8d4a767d565e868f1f780f32`
- [run_stress.py](/api/developments/residual-component-learner/files/run_stress.py?version=1) · code · 3519 bytes · sha256 `55cc28daebd5f0fb5d774dbc0fd61c783f904fbea5601c99f369d85a2ab50482`
- [run_synthetic.py](/api/developments/residual-component-learner/files/run_synthetic.py?version=1) · code · 14094 bytes · sha256 `359d6ba16a9b56397253ad6e33a01cbcb0e68bb0520ab76ea30a3e0b12d6d0b1`
- [run_tracking.py](/api/developments/residual-component-learner/files/run_tracking.py?version=1) · code · 3828 bytes · sha256 `97e2c96957f287cd4d35f8f0c1650c01fd13d0de2a166082383366d049c57d00`
