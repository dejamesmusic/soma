# records and banks: exact storage beside exposure-normalised associative memory, in place of a kv cache

james blight · 2026-10-16 · version 2 · result · verdict: alive

## current evidence note

editorial review 23 november 2026, for this specific notebook version. the original title and experiment text below are historical.

evaluation protocol: **synthetic_held_out, language_held_out_and_recall_probes**. associative recall and alias degrade under load; near-perfect literal recall depends on separate exact record; absolute two-hop accuracy limited; learned-key language-scale comparison not established; version 2 corrects the attribution of the plateau escape.

cost/accounting scope: state-ratio comparison, not a demonstrated universal throughput or language-scale efficiency gain.

[structured evidence and available material](/api/evidence.json) · [developments](/developments)


## original summary

separate exact historical storage from learned associative memory, then train their arbitration so exact recall cannot suppress the learning of reusable relationships. on controlled tasks, an exact record plus delta-rule banks with φ-spaced write steps and an exposure-normalised read recalls exactly at every tested load, retrieves by meaning, and chains lookups better than attention, with 20-79x less sequence state than a kv cache. the banks learn reliably (5 of 5 seeds, as attention does with the same short-sequence start) once each band's read is normalised by its own write step with bounded gain; exposure normalisation adds capacity on top. v2 corrects v1, which credited the plateau escape to exposure normalisation.

published: 2026-10-16 06:45:48.090413+00:00

evidence: controlled synthetic tasks · 1-3 seeds per arm · byte-level needles to 1 mb · not yet language

[latest version](/api/developments/records-and-banks.md)

sessions of 2026-10-13 to 10-16. code in `~/logOS/records_banks/` and `~/logOS/sieve/runtime/`;
results in `~/logOS/closed_system/results/fork_text/{runtime,rbnet}/`. the full spec and theory is
`records_banks/RECORDS_AND_BANKS.md`.

## the idea

attention's kv cache does two jobs with one expensive store:
- it says **exactly** what happened;
- it finds what is **relevantly similar** to now.

it keeps a dense key and value per token per layer because the context is its only working memory.

we split the jobs:

- a **record** keeps the past exactly: the tokens themselves, plus an index from recent context to the
  position that followed it. it replays what came next. it cannot generalise, and it does not interfere.
- **banks** keep learned associations in fixed-size state: delta-rule memories, one per head.
  - each head writes at its own φ-spaced step, so the heads cover different horizons;
  - each reads by dividing what it stored toward the query by how much it has been exposed toward it.
  - they generalise. they blur under load.
- a small gate **arbitrates**, trained on its own loss so that exact recall cannot suppress the banks'
  learning.

the contribution is not that raw bytes are cheaper than vectors. it is the combination:
- normalised multi-horizon banks;
- each component protected while it learns;
- the measured trade between capability and state.

indexed exact retrieval beside neural prediction is not new (for example infini-gram); the delta rule
is established (deltanet). what is ours is how those parts are made to work together.

## why it works (derived)

- **a bank is an online least-squares memory.** the delta rule S ← S + βk(v − Sᵀk)ᵀ is a gradient step on
  ‖Sᵀk − v‖². it forgets only along the key being written, so memories fade by competition, not by clock.
- **the exposure-normalised read.** the delta rule is linear in the values, with key-only coefficients.
  - so S = Σᵢ k̃ᵢ vᵢᵀ, and an exposure memory m run by the same rule with value 1 is Σᵢ k̃ᵢ.
  - Sᵀq / mᵀq is then exactly a normalised weighted average of the stored values: the shape of attention's
    output, which unnormalised linear memories lack.
  - for one key written n times at any step it returns v itself. this is the dense-key form of the exposure
    accounting (u + c/m) from our count-memory and soma resonance work.
- **correction (v2): what the read does.** the clamped read works as two regimes.
  - where exposure toward the query is unresolved (most reads early in training), it returns Sᵀq/(ε·a_h): the raw read
    normalised by the band's write step, so every band is heard and gets a gradient.
  - where exposure is resolved, it returns the exposure-normalised value.
  - the step-normalised raw read alone, with no exposure memory, also learns. a ridge-regularised read that silences
    unresolved reads does not.
  - so step normalisation with bounded gain is what lets the banks learn; exposure normalisation adds capacity. v1
    credited the plateau escape to exposure normalisation, and that was wrong in emphasis.
- **the floor is part of the model.** the implemented read is Sᵀq / (relu(mᵀq) + ε·a_h). with signed keys the
  exposure can be negative, so the floor sets a gain law, not just numerical safety. a query opposite to a
  single write returns −5·v at ε 0.1. the reliability results below are consistent with controlling that gain.

## what we measured

**byte level, exact record + exact-keyed banks, from scratch on 32 mb of fineweb.**
- needles: 24 random characters stated once, asked again at a distance; chance is 143-192 bits.
- the attention control is a 3.5m-parameter byte transformer with a 1,024-byte window, trained on the same 32 mb. it
  never learned to copy random strings even inside its window, so this table shows reach beyond any window, not a
  win inside it.

| system | held-out bpb | answer bits @256 b | @4 kb | @64 kb | @1 mb |
|---|---|---|---|---|---|
| byte transformer (3.5m, 1 kb window) | 2.196 | 242 | 243 | 249 | 250 |
| records + banks | 1.865 | 10.0 | 10.8 | 9.5 | 9.4 |
| records + banks + count memory | 1.742 | 9.8 | 10.5 | 10.0 | 11.1 |
| banks only | 1.876 | 142 | 162 | 195 | 228 |

for this architecture's kv format, a 4,194,304-byte span would need about 17 gb of fp16 kv. the record covers it
with 4 mib of bytes and a 4 mib index. both grow linearly with span; the cost per position differs enormously.
the 17 gb is specific to this architecture, not a lower bound for every attention implementation.

**learned-key banks against attention, inside matched sequence lengths.**
- tasks: recall (pairs given once, queried later), alias (queried under a different surface form of the key:
  retrieval by meaning) and two-hop (a→b, b→c given; a→c asked).
- models: 4 layers, d 128, 8 heads, trained from scratch.
- state is per sequence: banks 13,312 numbers including exposure and convolution buffers; attention's kv cache
  (+ the same buffers) 266,752 at t 256 and 1,053,184 at t 1,024. that is 20x and 79x.

| t 1,024 (accuracy at 16 / 64 / 128 / 192 facts) | recall | alias | two-hop |
|---|---|---|---|
| attention (+ short convolutions) | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 30 / 9 / 4 / 3 |
| banks (φ steps + exposure-normalised read) | 92 / 61 / 39 / 19 | 93 / 61 / 38 / 19 | 34 / 12 / 7 / 5 |
| records + banks | 99.9 / 99.6 / 99.4 / 99.6 | 93 / 61 / 38 / 19 | 33 / 12 / 6 / 4 |

at t 256 (accuracy at 8 / 16 / 32 / 48 / 64 facts):
- banks: recall 100 / 100 / 95 / 80 / 64; two-hop 44 / 31 / 20 / 15 / 4.
- attention: two-hop 23 / 12 / 6 / 5 / 4.
- plain φ banks, and deltanet with all heads at full step, stayed on the plateau at 1-3%.
- deltanet with the normalised read but no φ steps also stalled, on two seeds.

## what held, what didn't

- **the record makes recall exact at every tested load.** the banks supply retrieval by meaning; alias tracks
  recall exactly in every bank run. the two mechanisms are complementary.
- **the record must be a separate expert.** fed in as an input it answered recall for free, the banks never
  learned to retrieve, and alias and two-hop stayed at 0. with its own loss, optimiser and clipping, the
  network's weights are identical with and without it (difference exactly 0).
- **banks beat attention at two-hop** at both lengths. why is not yet explained.
- **failures:**
  - alias capacity falls with load (to 19% at 192 facts);
  - two-hop is low for everyone at 4 layers;
  - training is sensitive to initialisation, and so is attention's in this setup. over three rounds (26 runs):
    - the read floor ε is the main control: at 0.1 the hard seeds fail, fused or not; at 0.5 with a short-sequence
      start they broke through in 4 of 4 runs, by step 1,000-1,250;
    - flat write steps at the same mean succeed in 1 of 2;
    - 5 seeds of the selected configuration (ε 0.5, short-sequence start): 5 of 5 broke through by step 1,000-1,500;
      attention with the same start, 5 of 5 by 1,000-1,250;
    - two-hop at 8 facts: banks 43-48% against attention 19-24%, on all five seeds;
    - per-head diagnostics show the floor is the operating regime (~70% of first-layer reads are floor-dominated
      even when training works), and the failures begin with a ~100x gain shock in the slowest heads.
- **not done:** the learned-key model has not been compared with attention on language, or over millions of
  tokens. we make no claim that it has matched full attention there.

## the claim, as far as it goes

exact records and exposure-normalised associative banks provide complementary retrieval capabilities with
substantially less sequence state than a conventional kv cache, on controlled tasks.


## attached evidence

- [gate_eval.py](/api/developments/records-and-banks/files/gate_eval.py?version=2) · code · 3662 bytes · sha256 `d282e9a3376551e6aed220fd78a2c893665e322cdf57fb856f5ea38731f8becf`
- [rbnet.py](/api/developments/records-and-banks/files/rbnet.py?version=2) · code · 25898 bytes · sha256 `02841e0f337515346eb4edfa0e702e2a20a4357c56e1b84c106594669d293a13`
- [record.c](/api/developments/records-and-banks/files/record.c?version=2) · code · 5692 bytes · sha256 `c3f9d6cee0be6fc8c7b6a95230d6c3baca7862157e6ccd9f99e96acb02425a32`
