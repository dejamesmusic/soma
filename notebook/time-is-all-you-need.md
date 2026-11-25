# time is all you need: temporal translation and the credit assignment problem

james blight · 2026-03-01 · version 1 · paper · verdict: reference

## current evidence note

editorial review 23 november 2026, for this specific notebook version. the original title and experiment text below are historical.

evaluation protocol: **not_an_experiment**. not a universal proof of unlimited history or a uniquely optimal phi schedule; current runtime changed subsequently.

cost/accounting scope: not an experimental cost claim.

[structured evidence and available material](/api/evidence.json) · [developments](/developments)


## original summary

credit assignment is ill-posed until temporal history is rendered into a representation where responsibility is locally computable. causal, bounded, continuous, locally interpretable translation admits essentially one class of linear time-invariant finite-state summaries: exponentially decaying measurements at geometrically spaced timescales, spaced by the golden ratio. soma instantiates it and reaches 1.87 bits per byte under streaming constraints with bounded memory.

published: 2026-09-27 05:23:36.297287+00:00

evidence: peer reviewed · aaai 2026

[latest version](/api/developments/time-is-all-you-need.md)

## abstract

Learning algorithms assume meaningful input — state, context, relational structure. A continuous stream provides none of this. Before a system can learn from time, it must translate time into state. We argue that the central obstacle to embodied continual learning is not inadequate optimization but inadequate translation: credit assignment is ill-posed until temporal history has been rendered into a representation where responsibility is locally computable. We characterize the constraints on temporal translation — causal, bounded, continuous, locally interpretable — and show they admit essentially one solution class among linear, time-invariant, finite-state summaries: exponentially decaying measurements at geometrically spaced timescales. When spaced by the golden ratio to maximize incommensurability, this decomposition provides a minimal temporal language in which the past is present and credit assignment becomes tractable. We instantiate this in the Spectral Online Machine Architecture (SOMA), demonstrating continuous adaptation without catastrophic forgetting, under fixed resources, with no replay buffer and no sequence storage, reaching 1.87 bits per byte under streaming constraints with bounded memory. The architecture satisfies a requirement that theories of consciousness increasingly emphasize: temporal integration must be intrinsic to the system's state, not externalized as retrievable data. A system with translated time *is* its history; it does not merely *have* access to it. The failure of continual learning is not a failure of learning rules. It is a failure to give them a language for time.

## what the paper argues

- the translation problem
- what can be retained from infinity
- from measurement to language
- translation enables credit assignment
- biological foundations
- the spectral online machine architecture
- demonstrations
- temporal translation and the free energy principle
- conclusion
- optimality of golden ratio spacing
- complete algorithm specification

## figures

![Adaptation to distribution shift. New statistics accumulate while old patterns fade at trace-determined rates. Re-exposure yields faster acquisition than initial learning.](/api/developments/time-is-all-you-need/files/figure1.png?version=1)

![SOMA waveform prediction after 500 steps. Slow traces establish broad structure; fast traces have not yet accumulated sufficient statistics. Resolution emerges hierarchically.](/api/developments/time-is-all-you-need/files/figure2.png?version=1)

![SOMA waveform prediction after 10,000 steps. Fine structure has filled in as medium and fast traces converge. Multiple resolutions predicted simultaneously, with no interference between timescales.](/api/developments/time-is-all-you-need/files/figure3.png?version=1)

![Trace bank visualization. Each row is a timescale; columns are input symbols. Fast traces (top) capture immediate structure; slow traces (bottom) encode long-run statistics. The multi-scale decomposition is the temporal language.](/api/developments/time-is-all-you-need/files/figure4.png?version=1)

## where the work went next

every later entry in this notebook tests some part of this argument: which parts of the
trace bank survive control, what depth and transport add, and what continual learning
costs. the verdicts are recorded there, including where the paper's framing did not hold.


## attached evidence

- [figure1.png](/api/developments/time-is-all-you-need/files/figure1.png?version=1) · figure · 262818 bytes · sha256 `af384fac9b0b972ca26f8a7faae8722294fa9b37ace175e2081f782af138814b`
- [figure2.png](/api/developments/time-is-all-you-need/files/figure2.png?version=1) · figure · 224755 bytes · sha256 `9b327021d4649cdae244d01dc41875061d12633dc86833a9862f57b533c57b2e`
- [figure3.png](/api/developments/time-is-all-you-need/files/figure3.png?version=1) · figure · 153253 bytes · sha256 `86e4c1a2b5cfff2b8fc3e7eeb69aea3f9a239294af16e651731534a5f24753b4`
- [figure4.png](/api/developments/time-is-all-you-need/files/figure4.png?version=1) · figure · 65199 bytes · sha256 `4d85f32c03e7b11c69094f060f8aca352b67f18c544ed4532982eb4bbff4e9e6`
