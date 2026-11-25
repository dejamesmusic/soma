# somatic core at 1 GB: still learning, it beats a transformer trained the standard way

james blight · 2026-11-15 · version 2 · result · verdict: alive

## current evidence note

editorial review 23 november 2026, for this specific notebook version. the original title and experiment text below are historical.

evaluation protocol: **single_stream_prequential, frozen_held_out**. different parameter counts and scoring regimes; small lead relative to observed run variation; transformer led in a smaller four-pass experiment; offline baseline hyperparameters tuned at a smaller size.

cost/accounting scope: published approximate training-time comparison, not matched compute; the soma run record spans training plus the scored next 50 MB, while the transformer run record seconds field covers evaluation only; its approximately 2h training estimate is separate.

[structured evidence and available material](/api/evidence.json) · [developments](/developments)

[numeric results and source digests](/api/evidence/somatic-core-1gb-results.json)


## original summary

somatic core is a continual learner: one stream, in order, every token scored before it is learned from, with state that is fixed in size from birth. it pairs an exact count store (what followed every recent 1-, 2- and 3-token context, kept on six timescales from about 11 to 170,000 tokens plus an all-time rung, and an exact copy model over the last 65,536 tokens) with an attention learner over the last 1,024 tokens that reads the store at every block. after one pass over 1 GB of fineweb-edu it scores 1.165 bits per byte on the next 50 MB while still learning, against 1.178 for a 161M-parameter transformer and 1.193 for a 92M one, both trained offline for one epoch on the same 1 GB and then frozen. one seed each. against a transformer learning online on the same stream the gap is wide (1.29 against 1.68 at 1 GB, at 40M). it recalls a fact told once millions of tokens later, rereads text at half the cost, and its checkpoint is now 3.9 GiB at 178M learner parameters. it is about 7x slower to train than the offline transformer.

published: 2026-11-15 08:30:49.724565+00:00

evidence: fineweb-edu 10BT shard 000, bytes 0–1000 MB learned once in order, 1000–1050 MB scored · prequential, weights moving during scoring · transformers trained offline 1 epoch on the same 1000 MB then frozen, same tokenizer, same scored bytes (shard sha256 3d35760b…) · one seed per model · RTX 3090

[latest version](/api/developments/somatic-core-1gb.md)

work of 2026-11-09 to 11-15. code in `~/spectral_v2/somatic_core/`; every run has a record (config, code digest,
seed, shard hash) and its windowed scores.

## the short version

somatic core learns from one stream of text, in order, forever. every token is scored before the model learns from
it, the weights never stop moving, and the state is a fixed size from the first byte.

after reading 1 GB of fineweb-edu once, it predicts the next 50 MB at **1.165 bits per byte**, while still learning.
a 161M-parameter transformer trained the standard way on the same 1 GB (offline, shuffled, one epoch, then frozen)
scores **1.178** on the same bytes; a 92M one scores **1.193**.

| model, after 1 GB | bits per byte on the next 50 MB |
|---|---|
| **somatic core, 1024 wide × 12 deep (178M learner parameters), learning throughout** | **1.165** |
| transformer 161M (12 × 1024), offline 1 epoch, frozen | 1.178 |
| somatic core, 768 × 12 (106M) | 1.188 |
| transformer 92M (12 × 768), offline 1 epoch, frozen | 1.193 |
| transformer 41M, offline 1 epoch, frozen | 1.239 |
| somatic core with the earlier learner, 40M | 1.293 |
| transformer 41M, learning online on the same stream | 1.680 |

one seed per model. the comparison mixes two regimes on purpose, because they are the two ways the models are meant
to be used: the transformer is trained and then frozen; somatic core never stops. the version where both learn
during the scored text, and where both are frozen, is next (below).

## what it is

two parts, and a gate between them.

**an exact store.** for every recent context of 1, 2 or 3 tokens, it keeps what followed, as counts per (context,
successor) pair. each count is kept on six timescales, decaying with time constants of about 11, 76, 520, 3,600, 24,000
and 170,000 tokens (φ^5 to φ^25), plus an all-time rung that halves when it fills. alongside it, an exact copy model
over the last 65,536 tokens. the store has a fixed number of slots; contexts and pairs compete for them, and the
least-used are evicted. nothing in it grows.

**a learner.** a causal transformer over the last 1,024 tokens. the store's whole back-off chain (every order, every
timescale) is projected and injected into every block, so the learner reads the store wherever it attends.

**a gate** mixes the store's exact prediction with the learner's.

**the tick.** 512 tokens at a time: score them all, then learn from them (AdamW, decoupled weight decay 0.002, a fixed
rate). that's the only training. there is no pretraining phase, no replay buffer and no task boundary.

a correction to the first somatic core post: it said "no context window". the adopted core has one, 1,024 tokens
of attention. what it doesn't have is a cliff: past the window, the store still holds evidence, fading along smooth
curves rather than vanishing at a position.

## what moved it from losing to winning

at 1 GB on 10 November, the standard transformer led: 1.178 against our 1.358. three things closed it.

1. **the store feeds every attention block** (rather than only the input). at 100 MB this took the best one-pass
   result from 1.442 to 1.415.
2. **slower forgetting.** the learner's memory timescale is set by learning rate × weight decay. at 768 × 12,
   weight decay 0.03 → 0.008 → 0.002 gave 1.240 → 1.204 → 1.188 at 1 GB.
3. **width at a lower rate.** 1024 × 12 at rate 1e-4 gave 1.165. at 2e-4 it had trailed 768 × 12.

## the rest of the evidence

**learn 100 MB once, then predict the next 20 MB** (the transformer trained offline, one epoch, tuned):
somatic core 40M **1.424**, 106M **1.415**; transformer 41M 1.606, 161M 1.602.

**more passes.** with repeats the transformer catches up: level at two passes, the transformer ahead at four
(1.351 against 1.411). it extracts far more from a second look at the same data.

**against a strong single-pass compressor** (paq8px -8, one CPU core, the same bytes): on new text at 80–100 MB,
1.471 against paq8px's 1.490. paq8px still leads cumulatively at 100 MB (1.520 against 1.549), because of somatic
core's slower start.

**told once.** a 4-digit fact inserted once into the stream, then asked for 1,000, 1M and 3M tokens later: 12, 19
and 19.5 bits on the four digits, against ~40, 35 and 35 for a fact never told. a transformer learning online shows
no recall. past the copy model's 65k tokens, recall rides on 3-token contexts, so the cue has to sit within about
three tokens of the answer.

**rereading.** the same 10 MB read again after 10 MB of other text: 1.86 bits per byte the first time, 0.88 the
second. a transformer learning online: 2.46, then 2.25. somatic core keeps the text; the transformer keeps the
domain.

**state.** the 1 GB checkpoint is 3.9 GiB: learner weights 0.7, optimiser state 1.4, the store the rest. it was 7.5
until this week: the store's successor-slot tables were 97% empty (the median context has one successor), and a
paged layout gives the same answers bit for bit at half the size.

## what didn't work, and what's honestly unknown

- **one seed, and wider run-to-run spread than we first said.** every headline number is one run. at 40M we saw
  repeats within ±0.002; at 768 × 12, three runs with the same store state (same config, same data) span 1.384 to
  1.394 at 140 MB, a spread of 0.010, from GPU nondeterminism compounding through learning. the 1 GB lead over the
  161M transformer is 0.013, so it is not yet outside that spread. three seeds at 1 GB are the first hardening runs.
  (corrected 15 november: this bullet first said runs repeat to about ±0.002.)
- **"near shape", not equal parameters.** our 178M counts the store's projections; the 161M transformer's learning
  rate and weight decay were tuned at 41M, not at 161M.
- **regimes.** the transformer is frozen during the scored text; somatic core learns. both-learning (dynamic
  evaluation of the trained transformer) and both-frozen are the next runs.
- **cost.** learning online is slower: 1 GB took 14.4 hours at 1024 × 12 against about 2 for the offline 161M
  transformer on the same GPU. width is almost free in wall-clock (768 and 1024 take the same time); the store's CPU
  work is the bottleneck.
- **one corpus, one scale.** nothing above 1 GB yet. count models often lead early and lose late; we haven't shown
  this one doesn't.
- **the rate.** the best runs use a fixed learning rate, and no adaptive controller we've tried is reliably better: io2 (our
  residual-spectrum controller) by 0.004 at 1 GB, at the same average rate; and a family that compares the error with
  the learner's own recent moves, which turned out to measure the text's correlation length rather than whether the
  rate is right. a controller that sets the rate without tuning and without a schedule is open.
- **φ isn't special.** any log-spaced ladder of timescales with ratios from 2 to 25 ties; linear spacing and two
  rungs lose 0.045–0.05.

## where this goes next

the open question is scale, and it's a question of compute. what we'd do, in order:

1. **harden the 1 GB result**: three seeds; both evaluation regimes; a transformer given our store through the same
   gate (if it matches us, the union isn't the contribution); store size against bits per byte; enwik8, so the
   numbers sit beside published ones.
2. **the scaling grid**: learner sizes from 10M to 600M at 0.1, 1 and 10 GB, each beside an online transformer, an
   offline one, and paq8px. the win condition is written down before the runs: somatic core's curve stays below the
   online transformer's at matched compute, and approaches the frozen transformer's while still learning. if the gap
   closes and inverts with scale, we publish where.
3. **the demo**: a model that runs and keeps learning on an 8 GB laptop. you talk to it, teach it a fact, feed it a
   book, ask for the fact back.
4. **a preprint** with the protocol (prequential, bounded, retention and told-once measures, fixed public streams)
   and the harness, so anyone can put another model through it.

the bottleneck for 2 is the store's CPU work and wall-clock (each 10 GB point is days on one GPU), not money per
run. we're deciding how far to push it.


## attached evidence

- [core2.py](/api/developments/somatic-core-1gb/files/core2.py?version=2) · code · 31497 bytes · sha256 `4775f4712af880983fcf3619ad50c9a701c71fcacd7b49ac8e796994cf3c0f85`
- [core3.py](/api/developments/somatic-core-1gb/files/core3.py?version=2) · code · 47490 bytes · sha256 `5c5cb051b7790619eaa75441a089a0f5ab5f3db5e76025dcb211623e7d926bef`
- [runtime.py](/api/developments/somatic-core-1gb/files/runtime.py?version=2) · code · 42558 bytes · sha256 `6e90dba58c750cbd0393c3e68025440a5be52d3304797f117ca100acc7d18d0e`
