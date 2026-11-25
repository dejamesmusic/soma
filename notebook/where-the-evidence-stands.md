# where the evidence stands, november 2026

james blight · 2026-11-25 · version 1 · note · verdict: reference

## current evidence note

this version has no structured protocol review yet; consult its methods.


## original summary

what has been shown, what hasn't, and how the design got here, in one place. soma learns from a changing stream with bounded memory: a store of context–successor counts across timescales, and a causal attention learner (1,024-token window) whose weights keep updating. at 1 GB, still learning, it scored 1.165 bits per byte against 1.178 for a frozen 161M transformer, one seed each. soma-1b reached 0.925 in its final training window, with no matched 1b transformer. a fact stated once was partly recalled millions of tokens later. not established: a general advantage over transformers, solved forgetting, reliable tool use.

published: 2026-11-25 09:48:00.126877+00:00

evidence: summary of the entries linked below, each with its own protocol; machine-readable protocols at /api/evidence.json

[latest version](/api/developments/where-the-evidence-stands.md)

each result below belongs to its own experiment. protocols, exposure, seeds and gaps for each: `/api/evidence.json`.

## the hypothesis

learning need not stop at deployment. a model can keep updating its weights from what it reads and hears. being tested:
adaptation, retention, reuse of experience. unknown: general superiority, indefinite retention, robust tool use.

## language at 1 GB

learn the first 1 GB of fineweb-edu, score the next 50 MB. lower is better.

| model | learner parameters | bits per byte |
|---|---:|---:|
| somatic core, 1024 × 12 | 178 million | 1.165 |
| transformer, 1024 × 12 | 161 million | 1.178 |
| somatic core, 768 × 12 | 106 million | 1.188 |
| transformer, 768 × 12 | 92 million | 1.193 |

- one seed. near-shape, not equal parameters or equal compute.
- soma kept learning while scoring; the transformers were frozen.
- the 0.013 lead is within untested run-to-run variation.
- about 14.4 hours for soma, about 2 for the transformer, with different timing scopes.
- with four training passes, in a smaller experiment, the transformer led.

[somatic core at 1 GB](/developments/somatic-core-1gb)

## soma-1b

- 1.058 billion learner parameters, 1.012 billion tokens, one h100, about 27.5 hours, about $121 reported.
- final 40-million-token training window: 0.925. whole run: 1.022. training-stream scores, not held-out.
- eight distant stretches shared one learner. not a globally unseen-token evaluation.
- no matched 1b transformer.

[soma-1b: a new frontier](/developments/soma-1b-a-new-frontier) · [1.4 billion tokens in](/developments/soma-1b-1-4-billion-tokens)

## memory beyond the window

- a four-digit fact stated once cost about 19–19.5 bits to predict 1–3 million tokens later; never stated, about 35.
- partial retention, not recall. beyond the 65,536-token copy ring, recall needs a short cue: count keys cover 1–3
  tokens. slot competition and decay limit it.

## how the design got here

| stage | what it tested | what survived |
|---|---|---|
| [temporal translation](/developments/time-is-all-you-need) | a bounded temporal representation of the past, in decaying traces | the question |
| v12.3, v14, rma v1 | ways to read temporal evidence, allocate capacity, control updates | historical runtimes, still downloadable |
| [transport](/developments/transport-contraction) | mixing along long paths | conserving mass doesn't conserve information |
| [sgd vs adam](/developments/sgd-beats-adam-on-retention) | optimisers and forgetting | one sgd setting forgot 2.5× less (two seeds); not universal |
| [residual components](/developments/residual-component-learner) | admit a distinction only if it pays in future bits | worked on small finite dictionaries |
| [records and banks](/developments/records-and-banks) | exact storage beside learned association | the split between keeping an event and learning a relationship |
| [somatic core](/developments/somatic-core) | counts plus a small reader | the store |
| [at 1 GB](/developments/somatic-core-1gb) | attention, with the store in every block | the current design |

## the record

- entries are versioned. corrections say what changed; earlier versions stay citable.
- signed releases are immutable.
- a reported result, source availability and independent reproduction are separate facts. missing evidence is
  labelled, not invented.


