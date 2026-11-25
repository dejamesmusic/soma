# lineage

every architecture soma has published, in order: what it was, when, and where its evidence is. versions are tags in
this repository; results are entries in the [notebook](notebook). each runtime's own words are quoted from its
published record.

## the hypothesis

learning need not stop at deployment. a bank of exponentially decaying traces at geometrically spaced rates is a
bounded, causal summary of the past at every timescale: a spectral basis for learning over time. a model built on it
can keep updating its weights from what it reads and hears, with state that stays bounded. being tested: adaptation,
retention, reuse of experience. not established: a general advantage over transformers, unlimited retention, reliable
tool use. ([time is all you need](notebook/time-is-all-you-need.md), [where the evidence stands](notebook/where-the-evidence-stands.md))

## runtimes

| version | released | tag | what it was |
|---|---|---|---|
| v8 | 2026-04 | `v8` | the spectral online machine architecture: exponentially decayed traces as a spectral basis, published as one script and its specification. |
| v10 | 2026-05 | `v10` | a geometrically spaced trace bank; the script bundle, with metal, torch and triton paths. |
| v12, v12.2 | 2026-06 to 07 | `v12.2` | "the serial trace learner": byte stream, trace bank, serial hidden layers, next-byte belief. |
| v12.3 | 2026-09 | `v12.3.17` | "token-native continual spectral learner": an 8k sentencepiece dictionary feeds a token trace bank; a serial budget-residual mlp composes over the projected present. |
| v14 | 2026-09 | `v14.0.0-dev.2` | "adaptive trace-branch mixer": the exact float64 trace bank, organised by a learned trace-branch mixer into a 1,536-dim present; the temporal lens becomes an adaptive lattice on log time. |
| rma v1 | 2026-10 | `rma-v1-1.6.3` to `rma-v1-1.8.1` | "residual metabolic aperture": an 8k-token alphabet read through exact phi-spaced traces and nested cumulative apertures; bounded state, fixed matrix energy, no optimizer moments. |
| somatic core | 2026-11 | `somatic-core-1.0.0` to `somatic-core-1.6.0` | "bounded evidence across timescales, read by an attention learner": context–successor counts at six decaying timescales plus an all-time rung, a 65,536-token copy ring, a causal attention learner over 1,024 tokens fed by the store in every block. 1.5 adds search and sandboxed python as skills; 1.6 adds lanes (many conversations on one model at once). |

v8, v10 and v12 were published on github; from v12.3 on, releases are published by logOS, and this repository holds
each one exactly as logOS serves it (`tools/verify.py`). v12.3, v14 and rma v1 are posthumous: kept and downloadable, no
longer developed.

## the notebook

| when | entry | verdict |
|---|---|---|
| 2026-03 | [time is all you need](notebook/time-is-all-you-need.md): the original manuscript | reference |
| 2026-09-15 | [stochastic transport collapses long paths](notebook/transport-contraction.md): conserving mass doesn't conserve information | open |
| 2026-09-16 | [sgd beats adam on retention](notebook/sgd-beats-adam-on-retention.md): one sgd setting forgot 2.5x less (two seeds) | alive |
| 2026-09-20 | [residual component learner](notebook/residual-component-learner.md): admit a distinction only if it pays in future bits | alive |
| 2026-10-16 | [records and banks](notebook/records-and-banks.md): keeping an event versus learning a relationship | alive |
| 2026-11-08 | [somatic core](notebook/somatic-core.md): counts plus a small reader | alive |
| 2026-11-15 | [somatic core at 1 gb](notebook/somatic-core-1gb.md): attention with the store in every block, against transformers | alive |
| 2026-11-19 | [soma-1b: a new frontier](notebook/soma-1b-a-new-frontier.md): 1.06b parameters, 1.01b tokens | alive |
| 2026-11-24 | [the weights are the model](notebook/the-weights-are-the-model.md) | alive |
| 2026-11-25 | [soma-1b, 1.4 billion tokens in](notebook/soma-1b-1-4-billion-tokens.md) | open |
| 2026-11-25 | [lanes](notebook/lanes.md) | alive |
| 2026-11-25 | [the arithmetic of a shared model](notebook/the-arithmetic-of-a-shared-model.md) | open |
| 2026-11-25 | [a language for learning machines](notebook/a-language-for-learning-machines.md) | alive |
| 2026-11-25 | [where the evidence stands](notebook/where-the-evidence-stands.md) | reference |

machine-readable protocols, exposure and gaps for each result: [`notebook/evidence.json`](notebook/evidence.json).
the notebook here is a copy; logOS keeps every version of every entry at its own address.

## what isn't here

- versions between those published (v9, v11, v13 and the experiments between rma v1 and somatic core) were never
  released. where a notebook entry reports one, the entry is the record.
- the code behind each notebook result is being added under `research/`, pinned to the entries that cite it. where
  the exact code no longer exists, the entry will say so rather than reconstruct it silently.
- weights and datasets: weights are on logOS; datasets are public sources named in each entry.

## how this repository was assembled

on 2026-11-26, from the public `soma`, `soma_v10` and `soma_v12` repositories (their commits replayed with their own
messages and dates) and every runtime release logOS publishes, each checked against its published sha256 before it
was unpacked. a commit's author date is the release date; its commit date is when it was added here.
