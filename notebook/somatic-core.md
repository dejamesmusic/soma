# somatic core: a continual learner with no context window, that beats same-stream transformers

james blight · 2026-11-08 · version 1 · result · verdict: alive

## current evidence note

editorial review 23 november 2026, for this specific notebook version. the original title and experiment text below are historical.

evaluation protocol: **single_stream_prequential**. historical no-window headline does not describe current somatic core; finite copy ring even in this design; single seeds and limited corpora.

cost/accounting scope: no general matched-compute claim; timings and state belong to this historical implementation.

[structured evidence and available material](/api/evidence.json) · [developments](/developments)


## original summary

attention reads a window of the past exactly and nothing outside it: information goes from fully available to absent at a fixed position, and the window's cost grows with its length. somatic core has no window. its present is a bounded store holding, for every recent short context, what followed it on a ladder of φ-spaced timescales from 4 to 64k tokens, plus an exact copy model and a small learned reader that reads the store's whole back-off chain. evidence fades with age along smooth curves, and by competition for a fixed number of slots, never at an edge. learning one stream in order with fixed-size state, it scored 0.15 bits per byte below a same-tokenizer transformer trained on the identical stream at 6m bytes of simple stories, and 0.14 below the better of two at 20m bytes of scifi. most of the lead is on text that recurs, from beyond 128 tokens back and smoothly out to a million, with no step anywhere. the lead narrows with data (0.24 to 0.14 over 2.5m to 17.5m bytes).

published: 2026-11-09 01:11:15.287187+00:00

evidence: one stream, in order, prequential · simplestories 6m bytes, scifi 20m bytes · one seed · against bpe transformers on the identical token stream · receipts 1, 3–6 run; told-once recall pending · longer runs against our own continual learners: conversation worlds 25m bytes, simple stories 300m bytes

[latest version](/api/developments/somatic-core.md)

sessions of 2026-11-05 to 11-08. code in `~/spectral_v2/somatic_core/`; the research arc is in
`soma_gm_v1/investigations/2026-11-05-the-bet/results.md`, the engineering log in `somatic_core/RESULTS.md`.

## the short version

a transformer's context is a window: the last n tokens, read exactly, and nothing before them. that is a cliff.
a fact one token inside the window is fully available; one token outside, it is gone. the window is also the
expensive part: its key-value cache grows with its length, and it learns nothing between uses.

somatic core has no window. its whole state is fixed in size, and it holds:
- for each short context it has seen recently (the last 1, 2 or 3 tokens), what followed it, counted on seven
  timescales from about 4 to 64,000 tokens plus a slowest one;
- an exact copy model over the last 64k tokens;
- a small learned network that reads all of that, plus the last 16 tokens.

old evidence fades along a smooth curve set by the timescales, and by competition for a fixed number of slots. it
never falls off at a position. it learns continuously from one stream, in order, scoring every token before
learning from it.

on the same stream with the same tokenizer, it beats a transformer trained the standard way:

| | somatic core | bpe transformer, same stream | difference |
|---|---|---|---|
| simple stories, window ending 5.5m bytes | **1.102** | 1.248 (7.4m params) | −0.146 |
| scifi magazines, window ending 17.5m bytes | **1.608** | 1.769 (7.4m params) | −0.161 |
| scifi, the larger transformer, same window | **1.608** | 1.751 (17.0m params) | −0.143 |

bits per byte, prequential, lower is better. one seed each.

## what attention's window is, and what it costs

reading a window is exact: every token in it is available at full resolution, and attention can find any of them
by content. three costs come with it:

1. **the cliff.** information is either inside the window or absent. we measured it earlier in this project, on a
   borrowed smollm2 body reading a 256-byte window. an invented name told once and asked later in different words
   cost 5.78 bits at 256 bytes back. at 1 kb, once it had left the window, it cost 11.98 bits, no better than a
   model that never saw it (`2026-11-05-the-bet/results.md`).
2. **state grows with reach.** the key-value cache is linear in the window, per layer and per head. reach is
   bought with memory.
3. **the window learns nothing.** what the model knows about the last hour is what fits in the window. anything
   older has to be in the weights already, and in the standard recipe the weights are frozen.

on language at this scale, attention mostly uses the near past. the same transformers' loss by context length
flattens after about 32 tokens:

| bits per byte by tokens of context | 0-8 | 8-32 | 32-128 | 128 |
|---|---|---|---|---|
| simple stories, 5.5m bytes | 1.362 | 1.260 | 1.237 | 1.257 |
| scifi, 17.5m bytes | 1.837 | 1.764 | 1.765 | 1.746 |

so the window's reach is mostly unused here, while its cliff and its cost remain.

## the object: one lattice

a predictor of sequences can be laid out on two axes:
- **key length:** how much of the recent past selects the evidence;
- **value timescale:** how far back the evidence under that key is averaged.

| cell | what it is in a standard model |
|---|---|
| short key, slow value | the weights: all-time statistics of what follows what |
| middle key, middle value | n-gram and cache statistics |
| long key, fast value | attention as copying (an induction head): what followed the last occurrence of this context |

a transformer covers this lattice implicitly: its weights hold the slow end, and its window holds the fast end
at full resolution, up to the edge. somatic core covers it explicitly and continuously:
- **keys:** exact contexts of 1, 2 and 3 tokens, each owning a node in a fixed-size pool;
- **values:** each node's successors, counted on a φ ladder of decays (τ = φ^3, φ^7, … φ^23 ≈ 4 to 64k tokens) plus a
  slowest rung that halves every ~8m tokens;
- **the long-key, fast-value corner:** a copy model that finds the last earlier occurrence of the current 3+ tokens
  in a 64k-token ring and predicts what followed it.

**reading.** each timescale gives a full prediction by backing off along the chain of contexts (Witten-Bell). the
timescales are combined in log space, with a weight per (timescale, deepest context present) learned online.

**the learned reader.** exact keys cannot share between different contexts that mean the same thing. a small
network does that. it reads the last 16 tokens and, for every level of the chain and every timescale, the
expected embedding of the next token under that chain distribution. it trains on its own loss. a gate learned
online mixes it with the exact reader, by confidence.

**everything is bounded:**
- pools, slots, the pair store, the ring and the network are fixed in size;
- rows of the learned matrices cannot grow past a soft ceiling (the v14 radial compressor);
- counts are bounded by their decay, and the slowest rung halves;
- every clock is rebased.

a checkpoint is the whole state of a life, about the same size on day one as on day one hundred.

## why there is no cliff (derived)

a count at timescale τ_b is an exponential trace: evidence of age a contributes (1 − α_b)^a with α_b = φ^(−e_b).
the reader combines rungs with weights w_b in log space, so evidence of age a enters through

    Σ_b w_b · (1 − α_b)^a

a sum of exponentials with φ-spaced rates. it is smooth in a, has no zero, and falls off over the whole range
4 to 64k tokens rather than at one position. this is the trace bank of *time is all you need*, applied to the
values of an exact memory.

two things do cut, and they are not positions:
- **competition:** pools and pair slots are fixed. a context or successor that has not been used lately is evicted
  by sampled least-use. forgetting is by disuse, not by age.
- **the copy model's ring:** verbatim copying reaches 64k tokens back (about 240 kb of text). past that, what
  survives is what the counts kept, not the sequence.

## an ocean, not a cliff (argued)

the rungs are the same statistics resolved against differently paced backgrounds. that makes them
positional encodings of a kind: they encode age, not index.
- **where time sets relevance,** the fast rungs keep it: the scene, the story, what was just said.
- **over long enough horizons,** statistics become averages. a rung slower than the stream's coherence time
  predicts the same thing as the next slower one. there is a timescale K past which more bands do not change the
  statistics.
- **past K, time no longer encodes salience.** the information survives decay because it does not depend on it,
  so it can be held in space rather than time, and space here is the weights.

the slowest backgrounds are what the learner should absorb, and the store's fast rungs keep what the weights cannot
yet hold. so the transition is not an edge where information disappears. it is a shore, where evidence passes from
time-resolved counts into time-free weights as it stops depending on time. this is the hypothesis of *time is all
you need* (weights are the slowest summary, the kv cache the fastest), made literal in one store and one learner.

it also says what does not cross. a fact told once is an event, not an average. it lives at the fast end (the fast
rungs and the copy ring) and is kept for as long as they keep it. the weights take what repeats.

two predictions follow, and both are measurable (receipts 5 and 6 below):
- adjacent rungs converge beyond some τ*;
- the slow rungs' marginal value falls as a run goes on, as the learner takes over what they hold.

at 20m bytes neither has happened yet (receipts 5 and 6). the learner moves toward the slowest rung, but the
rungs have not converged by 64k tokens and their value is still growing.

**cost per token is independent of reach.** a lookup and an update touch only the 3 active contexts and their
successors, about 300 cells. evidence 60,000 tokens old costs nothing to keep and nothing to read. attention's
cost to see 60,000 tokens back is a 60,000-token cache.

## what we measured

**setup.**
- one stream, read in order, never repeated; every 256-token tick is scored before anything learns from it.
- the soma-8k-v1 bpe tokenizer, the same one as the transformer controls. it was trained offline, so its
  vocabulary is prior knowledge for both.
- transformers are trained the standard way on the identical token stream: 128-token blocks, every position scored
  before its adam step.

**simple stories, 6m bytes** (window bpb at 5.5m):

| | store alone | learner alone | final |
|---|---|---|---|
| bpe transformer, l4 d256 (7.4m params) | | | 1.248 |
| bpe transformer, l6 d384 (17m params) | | | 1.257 |
| somatic core (11.6m params + a 210 mb store) | 1.217 | 1.154 | **1.102** |

**scifi magazines, 20m bytes** (ocr'd 1950s-60s magazines; window bpb):

| bytes | somatic core final | its learner alone | transformer l4 d256 | transformer l6 d384 | lead over the better |
|---|---|---|---|---|---|
| 2.5m | 1.895 | 2.069 | 2.149 | 2.136 | −0.24 |
| 5m | 1.748 | 1.836 | 1.975 | 1.973 | −0.23 |
| 10m | 1.626 | 1.683 | 1.818 | 1.809 | −0.18 |
| 17.5m | **1.608** | 1.658 | 1.769 | 1.751 | **−0.14** |

**what it took** (each step measured on simple stories):
- **units:** exact contexts keyed on bytes reached 1.366; keyed on bpe tokens, 1.198. byte contexts straddle word
  boundaries and fragment the statistics.
- **the reader's features:** the chain's expected next-token embedding instead of its log-distribution took the
  learner alone from 1.531 to 1.409.
- **exactness inside the tick:** each position counts the earlier positions of its own tick, so tick size no
  longer changes what the store sees. a bigger tick then gave the learner better steps, 1.409 to 1.315.
- **bounds:** the learner alone went from 1.198 to 1.154 with the bounds on. we have not separated which part.

## what held, what didn't

- **learned parts must train on their own loss** (three instances). fitted to the residual of the exact reader,
  the network's gradient is mostly noise. trained on the mixture's loss, it collapses.
- **width did not pay at these sizes.** learner width 2048 equals 1024 at 6m bytes. depth 3 equals depth 1
  through 20m bytes of scifi (cumulative 1.686 vs 1.681). width 2048 trails 1024 at 20m bytes of scifi (17.5m window
  1.618 vs 1.608, its learner 1.679 vs 1.658) and costs 122 ms per tick against 84. the data are not yet enough for
  capacity to pay.
- **the lead narrows with data,** about 0.035 bits per doubling over 2.5m to 17.5m bytes of scifi. this is the
  known pattern: count models lead early and transformers catch up. extrapolating naively, they meet near 300m
  bytes against the larger transformer, 400m against the smaller. whether the learned reader's share keeps growing past that is the open question.
- **what attention does that this does not, yet:** retrieve a one-off event by content from far away when the
  preceding words differ.
  - exact keys only match the same last 1-3 tokens;
  - the copy model only matches a repeated 3-token run;
  - the learner sees 16 tokens.

  a name introduced once 2,000 tokens ago, asked about in new words, is on the fast shore. it reaches the predictor
  only through what followed the same short contexts. what repeats, the weights are meant to carry (the ocean
  above). receipt 2 measures how far the fast shore reaches.

## longer runs, against our own continual learners

the transformer comparison stops at 20m bytes. two longer runs from the soma harness's run records compare somatic
core with the continual learners before it, on the same files, at the same source position. bits per byte,
prequential, per window.

**conversation worlds** (dialogue), from byte 0:

| bytes | somatic core (33m params) | soma gm v1 | rma v1 (51m params) |
|---|---|---|---|
| 2.5m | **1.016** | 1.351 | 1.275 |
| 5m | **0.965** | 1.251 | 1.226 |
| 10m | **0.892** | 1.210 | 1.133 |
| 20m | **0.830** | 1.073 | 1.102 |
| 25m | **0.799** | 1.031 | 1.100 |

gm v1 got to 0.80 somewhere between 100m and 250m bytes; somatic core was there at 25m.

**simple stories**, somatic core after 114m bytes of conversation worlds, then 300m bytes of stories:

| source position | somatic core (33m) | rma v1 (105m, after 0.55b bytes) | rma v1 (105m, after 8.7b bytes) |
|---|---|---|---|
| 0–25m | 1.007 | 1.779 | 1.150 |
| 150–175m | **0.904** | 1.252 | 0.971 |
| 275–300m | **0.904** | 1.157 | — |

- rma ran with io2 choosing its own decimation. that trades bits for speed (it read about five times faster), so this
  compares the systems as they run, not the readers alone.
- on the same stream, an rma head over a frozen, pretrained 135m-parameter transformer body (smollm2) reached
  0.745. that is the gap the real comparison below has to close.
- somatic core flattened at about 0.90 from 75m bytes. its bounded state was full: the order-3 context pool
  turned over every ~2m tokens, the order-2 pool every ~8m, and the all-time rung halved every ~4m. the curve is
  the level its bounds allowed, which says the next scale to buy is the store's.

## receipts

one instrumented 20m-byte scifi run of somatic core (h1024, the same configuration and the same final numbers as
the table above), measured before each tick is learned from, against the two transformers' per-token
probabilities on the identical token stream. analysis over the second half (10m–20m bytes). scripts:
`somatic_core/tests/receipts.py`, `receipt_distance.py`.

### 1. bits by distance to the last occurrence

for every token, the distance back to the last time the same 3-token run (two tokens of context and the token
itself) appeared. bits per byte in each band:

| distance (tokens) | share of bytes | somatic core | transformer l6 d384 | lead |
|---|---|---|---|---|
| 1–32 | 0.7% | 0.60 | 1.54 | −0.94 |
| 33–128 | 1.5% | 0.50 | 1.33 | −0.83 |
| 129–1k | 3.3% | 0.68 | 1.23 | −0.55 |
| 1k–8k | 4.7% | 0.90 | 1.12 | −0.22 |
| 8k–64k | 7.5% | 1.06 | 1.21 | −0.15 |
| 64k–1m | 22.5% | 1.27 | 1.52 | −0.25 |
| over 1m | 12.6% | 1.64 | 1.77 | −0.12 |
| never seen | 47.3% | 2.07 | 2.10 | −0.03 |

- **somatic core declines smoothly with distance.** there is no step anywhere, including at 64k, where both the
  copy ring and the 64k rung end. with 6-token runs the curve is the same shape (0.36 at 33–128, 0.58 at 8k–64k,
  0.73 at 64k–1m, 1.67 never seen).
- **its lead lives in recurrence.** weighting each band by its bytes, 90% of the 0.145 lead comes from tokens whose
  run occurred before, and 76% from runs last seen more than 128 tokens back, beyond the transformer's context.
  39% comes from 64k–1m tokens back alone. on runs never seen before the two are level (2.07 against 2.10).

**the transformer has no cliff to show here, because it does not copy.** we split each short-distance band by
whether the earlier occurrence fell inside the transformer's visible context. its 128-token blocks make that
depend only on where the block boundary falls. inside and outside are level (33–64 tokens: 1.36 inside, 1.33
outside; 65–128: 1.34 and 1.31). at 20m bytes and this size, the transformer has not learned to copy from its
context, and its own context curve agrees: tokens past the 8th buy 0.07 bits and past the 32nd nothing. so these
transformers lose on recurrence because they do not use their window, not because of where it ends. the cliff
itself is measured on a trained model (smollm2 with a 256-token window: 5.78 bits at 256 bytes, 11.98 at 1k).
a fair same-stream cliff needs a transformer that has learned induction. that is a larger run than this one.

**the real comparison** is a fully trained transformer, frozen, against a fully trained somatic core that is still
learning online. the transformer has a window it can use and nothing past it. somatic core has no window, and its
memory is fixed in size and still being written. that is where the cliff and the memory cost per token of reach can
be compared fairly. only long runs can set it up.

### 3. what the long rungs hold

at read time, every rung past τ is replaced by rung τ, in both the exact mix and the learner's input. cost in bits
per byte:

| collapsed past | at 2.5m bytes | at 10m | at 20m |
|---|---|---|---|
| 199 tokens | +0.49 | +0.65 | +0.66 |
| 1.4k | +0.30 | +0.41 | +0.42 |
| 9.3k | +0.17 | +0.27 | +0.28 |
| 64k | +0.07 | +0.15 | +0.16 |

by distance, each collapse costs exactly in the bands beyond its τ and nowhere before it. collapsed past 1.4k
tokens, the 1k–8k band goes from 0.90 to 1.37 and the 129–1k band is unchanged. each rung holds the recurrences
at its own timescale. collapsing past 199 tokens is the closest thing to giving somatic core the transformer's
window. it costs 0.66 bits per byte, more than four times the whole lead.

(the collapse also gives the learner inputs it never trained on, so the learner's share of the cost is an upper
bound. the exact mix alone is in distribution.)

### 4. state is constant

over the 20m-byte run, gpu memory stays at 535–536 mb from the first window to the last. process memory varies
between 640 and 910 mb with no trend. the store, learner and every counter are fixed in size and bounded
(`somatic_core/bounds.py`).

### 5 and 6. the ocean: not reached at this scale

- **adjacent rungs do not converge by 64k tokens.** the divergence between adjacent rungs' predictions,
  KL(p_slower ‖ p_faster) in bits, falls by about 30% per rung, smoothly: 3.11, 2.27, 1.64, 1.14, 0.80. from 64k to
  the all-time rung it is 2.04 and rising (0.79 at 2.5m bytes). the all-time rung is also the best single rung
  (1.87 bits per byte alone at 17.5m, 64k: 1.97). on this stream at 20m bytes, statistics are still changing with
  timescale at 64k tokens. no K has been found yet.
- **the slow rungs' value grows with data.** it does not fall: the cost of collapsing past 64k doubles over the run.
- **the learner does move toward the slow rungs.** its divergence from the all-time rung falls from 2.58 to 2.12
  bits over the run, and rises from the fast ones (from 6.2 to 7.6 at τ = 4).

so the ocean is a hypothesis this run neither confirms nor refutes. the learner is moving toward the shore, but
20m bytes are not enough for it to arrive, and the long counts are still where the long-range value lies. a
longer run, or a stream with a shorter coherence time, is needed.

### 2. told once, asked again (pending)

the bet's fact stream (invented names in natural prose), asked at lags from 32 tokens to 64k, once in the same
wording and once paraphrased. prediction: verbatim recall holds to the ring's reach, and paraphrased recall is
weak, which bounds the claim above.

## prior art

the base is context mixing: ppm (cleary & witten 1984), witten-bell escape, paq and cmix's mixed context models,
cache language models (kuhn & de mori 1990; grave et al. 2017), and exact or nearest-neighbour memories
interpolated with neural models (knn-lm, khandelwal et al. 2020; infini-gram, liu et al. 2024). we use them as
the floor.

what this adds, to be checked against the literature before any claim:
- every context's continuation held at φ-spaced timescales, with no window;
- a learned reader that reads the exact model's whole back-off chain, trained on its own loss;
- the whole system as a continual learner with fixed-size, self-contained state, judged prequentially.

it is meant to pick up where counts and context mixing leave off, without a cliff and inside a bounded,
continuous state.

## the claim, as far as it goes

somatic core has no context window and no positional horizon for the statistics it keeps. evidence fades smoothly
with age and by disuse, and its state and cost per token do not grow with reach. learning one stream in order,
it beats bpe transformers trained on the identical stream at 6m and 20m bytes, on two corpora, with one seed.
almost all of its lead is on text that recurs, mostly from beyond 128 tokens back and smoothly to a million.
the same-stream transformers lose that by not copying at all, so a same-stream cliff against a transformer that
copies is still owed. told-once recall (pending) will say how much of attention's retrieval it replaces, and the
narrowing lead says the learned side has to scale for the result to hold at larger data.


## attached evidence

- [core2.py](/api/developments/somatic-core/files/core2.py?version=1) · code · 20522 bytes · sha256 `6de0bbf5f5417953901682b3712e5ab9b2b08920d1fbb55c9a7f51f275b5f2cc`
- [core3.py](/api/developments/somatic-core/files/core3.py?version=1) · code · 23269 bytes · sha256 `3051bccbf22ee3fa640d0536eaa3afd1baa998134e2d36d778edd1e2d3e2f956`
- [RESULTS.md](/api/developments/somatic-core/files/RESULTS.md?version=1) · log · 9942 bytes · sha256 `23178acc313809ef238e8b404f6c9b58a26b63e4c5eb99805fff750aaacfef8e`
- [THEORY.md](/api/developments/somatic-core/files/THEORY.md?version=1) · paper · 20713 bytes · sha256 `0e45206840a19971e1ca9893b174bf8f3443e3ca6bdc5b6498836ea608ad4c00`
