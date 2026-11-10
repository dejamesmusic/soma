# somatic core 1.1.0: runtime specification

*Runtime `soma.somatic_core`, species `somatic_core`, checkpoint format `somatic-core-pytorch` (checkpoint version 1),
tokenizer soma-8k-v1 (8192 tokens). Theory: `THEORY.md`. Measurements: `RESULTS.md`.*

somatic core predicts the next token of one stream, then learns from it, forever, in a state of fixed size. It has no
attention and no context window: what it keeps of the past is the exact statistics of what followed what, at a ladder of
timescales from about 4 tokens to all-time, read by a learned network.

## 1. One tick

The stream is read in ticks of `chunk` tokens (default 512). Each tick does the following, in order:

1. **Score.** Every token in the tick is predicted, and its bits are counted, *before anything learns from it*. Every
   reported bpb is prequential: one stream, in order, weights moving.
2. **Learn.** The learner, the mix and the gate take one step on the tick. The store then counts the tick's tokens.

The exact path is exact *inside* the tick too. A token sees every earlier token of the same tick, as though the store
were updated one token at a time. So the store's predictions do not depend on the tick size; only the learner's update
cadence does.

## 2. State

| part | what it holds | size |
|---|---|---|
| contexts, orders 1–3 | the last 1, 2 and 3 tokens; one node per distinct context. Order 1 has a node per token; orders 2 and 3 are bounded pools (`p2`, `p3`) that recycle the stalest node | `p2`, `p3` nodes |
| successor slots | per node, up to `k1` / `k2` / `k3` successor tokens | nodes × slots (CPU) |
| pair counts | per (context, successor) pair, a count on each rung of the ladder | `np1`, `np2`, `np3` pairs × 7 rungs (GPU) |
| node totals, clocks | per node, a total on each rung, and the clock of its last update (decay is lazy, per node) | nodes × 7 |
| order 0 | per token, a count on each rung | 8192 × 7 |
| match model | the stream's recent tokens (`ring`), for copying a repeat of length ≥ `minl` | `ring` |
| learner | the learned reader (§4) and its AdamW state | ≈ 16 bytes per parameter |
| mix, gate | rung weights per deepest order (4 × 7), match weights (4 × 6), a gate per key | small |
| io2 | the controller's residual bank (§6) | 17 × width, float64 |

**The ladder.** Rung *b* decays each count by 1 − φ^(−e_b) per token, with e = `ladder` = 3, 7, 11, 15, 19, 23. The
timescales are τ ≈ 4, 29, 199, 1364, 9349 and 64079 tokens. A seventh rung is all-time: it never decays, and it halves
when its total passes `alltime_cap` tokens.

## 3. The exact reader

For each rung separately:
- a Witten–Bell back-off runs down the chain order 3 → 2 → 1 → 0 → uniform;
- each order's escape weight comes from its distinct-successor count and its total;
- the all-time rung counts at least one type whenever a node has any count (added in 0.1.3).

This gives seven distributions over the next token. They are mixed **in log space** by weights chosen by the deepest
order present. The match model adds a learned bump on the token it predicts. The result is `p_exact`.

## 4. The learned reader

A per-position network. It does not mix across positions, and it has no recurrence. Its inputs:
- the last `window` tokens (16), embedded (`emb`, 64);
- for every order and rung, the exact reader's chain projected through a learned token projection (`proj`, 16), with
  log totals.

It has `depth` layers of width `hidden` (after the first two, pre-LayerNorm residual blocks), a final LayerNorm
(`final_norm`, added in 1.0.1), and an output layer onto 8192 tokens. It is trained on its own loss with AdamW:
decoupled weight decay `wd` on its matrices, gradient clipping at 1 (§7). Its output is `p_learned`.

**The prediction.** p = s · p_exact + (1 − s) · p_learned. The gate s is learned per key (deepest order, match length,
entropy bucket).

## 5. Learning per tick

- **The learner:** one AdamW step on the tick's mean loss, at io2's learning rate.
- **The rung and match weights:** one Adam step (`mix_adam`).
- **The gate:** a closed-form step toward whichever reader did better (`gate_lr`).
- **The store:** counts the tick's tokens. A recycled node, and its pairs, read as empty from the tick it is recycled.

**Decimation (optional; off by default).** With `learn_mode`:
- `skip`: the optimiser steps on one tick in `learn_stride`.
- `accumulate`: the gradient is averaged over `learn_stride` ticks.
- `select`: every `learn_stride`-th prediction gives a gradient, with one step per io2 quantum.
- `io2`: like `select`, but io2's decimation band chooses the stride, dithered so the supervised density is exact.

In every mode the learner still predicts every token. In tests on a fresh model at 20 MB, every mode cost 0.04–0.05
bpb, for 17–21% more throughput. That is why the default is no decimation.

## 6. io2

The production controller, ported from soma v14:
1. The learner's residual, translated into its own coordinates (W_outᵀ(onehot − p) / mean row norm), drives a bank of
   16 φ-spaced bands.
2. The bank's energy, coherence and spectrum set the plasticity ratio, once per quantum of 512 predictions.
3. The learning rate is `lr_base` × ratio.
4. io2 also computes a decimation band (reported as `io2_band`), which is used only when `learn_mode` = `io2`.

## 7. Nothing grows

Every number the learner carries is bounded. The checkpoint is fixed-size for a given shape.
- **Learner matrices** (`bound_mode` = `wd`, the default from 1.1): AdamW's decoupled weight decay. Each step shrinks a
  weight by lr·wd and moves it by at most about lr, so each weight settles within about 1/wd of zero (10 at the
  default). Gradients are clipped at norm 1.
- **Learner vectors** (biases, LayerNorm gains): no decay; clamped to ±8.
- **`bound_mode` = `radial`** (1.0.x, and every checkpoint from before 1.1): soft radial compression of every row.
  Outward growth is scaled by max(0.01, 1/(1 + r⁴)), where r is the row's norm over its ceiling, max(1, 0.1 √fan_in);
  vectors take the same rule with an RMS ceiling of 2.
- **`bound_mode` = `sphere`** (experimental): every matrix keeps its starting Frobenius norm and only turns.
  Angle-capped steps, no optimiser moments.
- **Gate and mix weights:** clamped to ±8.
- **Adam's step count:** capped.
- **Counts:** bounded by decay. The all-time rung halves at `alltime_cap`.
- **Clocks:** rebased at 2³⁰ tokens.
- **Pools, slots, pairs and the ring:** fixed size.

## 8. Settings

**Shape** is fixed when a model is created; a continuation keeps it.

| setting | default (new model) | meaning |
|---|---|---|
| `hidden` | 2048 | learner width (the cloud agent's `width` is accepted as an alias) |
| `depth` | 3 | learner layers |
| `final_norm` | 1 | a LayerNorm before the output; checkpoints from before 1.0.1 open with 0 |
| `bound_mode` | wd | how the learner is bounded (§7): `wd`, `radial` or `sphere`; checkpoints from before 1.1 open with `radial` |
| `p2`, `p3` | 1048576, 2097152 | order-2 and order-3 context pools |
| `np1`, `np2`, `np3` | 1048576, 4194304, 8388608 | pair pools per order |
| `alltime_cap` | 33554432 | the all-time rung halves when its total passes this many tokens |
| `k1`, `k2`, `k3` | 256, 32, 16 | successor slots per node |
| `ladder` | 3,7,11,15,19,23 | rung exponents, φ^e tokens |
| `window`, `emb`, `proj` | 16, 64, 16 | learner inputs |
| `ring`, `minl` | 65536, 3 | match model |

**Dynamics** may change on a continuation.

| setting | default | meaning |
|---|---|---|
| `chunk` | 512 | tokens per tick |
| `lr_base` | 0.003 | io2's base learning rate |
| `mix_adam`, `gate_lr`, `wd` | 0.003, 0.02, 0.1 | mix, gate, weight decay (checkpoints from before 1.1 keep 1e-4) |
| `learn_stride`, `learn_mode` | 1, skip | decimation (§5) |
| `tf32` | 1 | CUDA only: fp32 matmuls on tensor cores |
| `learner_bf16` | 0 | CUDA only: the learner in bf16 autocast, with weights and Adam in fp32 |

**Sizes at the defaults:** 41.5M learner parameters, a ~1.9 GB checkpoint and ~1.1 GB of GPU memory. On an Apple M3,
about 2.8k tokens/s (11.5k bytes/s).

## 9. Checkpoint

A `torch.save` dict. Its identity fields:
- `checkpoint_format`, `checkpoint_version`, `runtime_id`, `runtime_version`, `species`, `tokenizer_sha256`;
- `checkpoint_id`: stable for the model's life;
- `revision_id`, with `parent_revision_id` and `revision_history` (the last 256): the lineage;
- `root_lineage_id` and `forked_from_*`: forks.

Plain metadata (`bytes_seen`, `tokens_seen`, `parameters`, `hidden_dim`, `depth`, `config`) is readable without
unpacking the state. The state is in `core` (the store, learner, optimisers, mix, gate, io2 and decimation phase),
`stream_ctx` / `chat_ctx` (match histories), `sampler`, `source_cursor` and `life`.

A `.meta.json` sidecar carries the config and the source cursor (path, file position, size). A continuation on the same
file resumes at the exact byte.

**Compatibility:** 1.1.0 opens checkpoints from 0.1.0–0.1.5, 1.0.0, 1.0.1 and 1.1.0. A checkpoint without `final_norm`
in its config opens with it off; one without `bound_mode` opens with `radial` and keeps its own `wd`. Both are
unchanged.

**1.0.1.** Without the final LayerNorm the learner's residual stream is unnormalised. On a fresh model, Adam's first
steps on the zero-initialised block outputs push it to norms in the thousands, with the learner's own loss in the
hundreds of nats. At 2048×3 that is a transient over the first ~50 ticks; at 4096×4 it diverged, and io2's
plasticity exponent overflowed. 1.0.1 adds the norm for new models and clamps io2's exponent.

**1.1.0.** A four-arm A/B on the same 20 MB stream (fineweb-edu 10 MB, scifi 5 MB, fineweb-edu 5 MB; 2048×3, fresh,
prequential) put plain AdamW weight decay ahead of the radial compressor:

| bounding | lifetime bpb | learner alone, first 10 MB |
|---|---|---|
| radial (1.0.1) | 1.8035 | 2.260 |
| wd 1e-2 | 1.7936 | 2.208 |
| wd 1e-1 (1.1 default) | 1.7928 | 2.206 |
| sphere SGD | 1.8398 | 2.545 |

wd 1e-1 was also ~8% faster (no per-row norm pass), and the only Adam arm whose weights and residual stream shrank over
its life. 1.1 also replaces two advanced-index lookups in the learner's features with `index_select`, the same maths
with a far cheaper backward on CUDA (1.5× faster ticks on an RTX 3090).

## 10. Reporting

- **Headline:** `bpb` (prequential, final mix).
- **Stats:** `bps` (source bytes/s), `life_bytes`, `bpb_exact` (the store alone), `bpb_learner` (the learner alone),
  `lr`, `radial_pressure`, `o3_pair_fill`.
- **Also reported:**
  - `gate_exact`;
  - per-order fill, recycles and overflows;
  - halvings;
  - io2's ratio, energy, coherence and band;
  - `residual_spectrum` / `residual_baseline` / `residual_excess` by band;
  - the surprise of recent tokens and the top predictions.
