# somatic core: runtime specification (1.2.1; what 1.5.0 adds is §13)

*Runtime `soma.somatic_core`, species `somatic_core`, checkpoint format `somatic-core-pytorch` (checkpoint version 1),
tokenizer soma-8k-v1 (8192 tokens). Theory: `THEORY.md`. Measurements: `RESULTS.md`.*

somatic core predicts the next token of one stream, then learns from it, forever, in a state of fixed size. Its memory
has no context window: what it keeps of the past is the exact statistics of what followed what, at a ladder of
timescales from about 11 tokens to all-time. A learned network reads that memory. Since 1.2 the network is a causal
attention learner over the last 1024 tokens, which reads the store at every block; evidence older than its window
reaches it only through the store, and fades there along smooth curves, never at an edge.

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
| successor slots | per node, up to `k1` / `k2` / `k3` successor tokens; since 1.2 in pages of `slot_page` (16) slots from a shared pool, so a node holds only the pages it uses | pages × 16 (CPU) |
| pair counts | per (context, successor) pair, a count on each rung of the ladder | `np1`, `np2`, `np3` pairs × 7 rungs (GPU) |
| node totals, clocks | per node, a total on each rung, and the clock of its last update (decay is lazy, per node) | nodes × 7 |
| order 0 | per token, a count on each rung | 8192 × 7 |
| match model | the stream's recent tokens (`ring`), for copying a repeat of length ≥ `minl` | `ring` |
| learner | the learned reader (§4) and its AdamW state | ≈ 16 bytes per parameter |
| match index | the order-2/3 context pools' key → node index (numpy open addressing since 1.2.1) | ~70 MB at the defaults |
| mix, gate | rung weights per deepest order (4 × 7), match weights (4 × 6), a gate per key | small |
| io2 | the controller's residual bank (§6; legacy models only) | 17 × width, float64 |

**The ladder.** Rung *b* decays each count by 1 − φ^(−e_b) per token, with e = `ladder` = 5, 9, 13, 17, 21, 25 (1.2;
3, 7, … 23 before). The timescales are τ ≈ 11, 76, 521, 3571, 24476 and 167761 tokens. A seventh rung is all-time: it never decays, and it halves
when its total passes `alltime_cap` tokens.

## 3. The exact reader

For each rung separately:
- a Witten–Bell back-off runs down the chain order 3 → 2 → 1 → 0 → uniform;
- each order's escape weight comes from its distinct-successor count and its total;
- the all-time rung counts at least one type whenever a node has any count (added in 0.1.3).

This gives seven distributions over the next token. They are mixed **in log space** by weights chosen by the deepest
order present. The match model adds a learned bump on the token it predicts. The result is `p_exact`.

## 4. The learned reader

**The attention learner** (`learner` = `attn`, the default since 1.2). A causal transformer of `depth` pre-LayerNorm
blocks, width `hidden`, `att_heads` heads and learned absolute positions, over the last `att_ctx` (1024) tokens. Each
tick predicts its `chunk` positions at the end of a full window, so every position it is trained on sees `att_ctx`
tokens behind it.

The store enters at every block (`att_inject` = `every`). For each predicted position, the exact reader's chain at every
order and rung is projected through a learned token projection (`proj`, 256), with log totals; a LayerNorm and a
linear cell map that read to width `hidden`, and it is added to that position's residual stream before each block.
A final LayerNorm and an output layer onto 8192 tokens give `p_learned`. It is trained on its own loss with AdamW at a
fixed rate `lr` (2e-4) after a linear `warmup` of 200 steps; decoupled weight decay `wd` (0.002) on its matrices;
gradients clipped at 1 (§7).

**The legacy learner** (`learner` = `mlp`, 1.0–1.1, and every checkpoint from before 1.2). A per-position network with no
mixing across positions and no recurrence. Its inputs are the last `window` tokens (16), embedded (`emb`, 64), and the
same chain projections (`proj`, 16). It has `depth` layers of width `hidden` (after the first two, pre-LayerNorm
residual blocks) and a final LayerNorm (`final_norm`, added in 1.0.1).

**The prediction.** p = s · p_exact + (1 − s) · p_learned. The gate s is learned per key (deepest order, match length,
entropy bucket).

## 5. Learning per tick

- **The learner:** one AdamW step on the tick's mean loss, at the fixed rate `lr` (legacy models: io2's rate). Its
  gradients are freed as soon as the step is taken (1.2.1).
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

Off for new models since 1.2 (`io2` = 0), which learn at a fixed rate. Models from 1.0–1.1 keep it. The controller, ported from soma v14:
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

## 8. Chat and generation

Chat, dreams and evaluation read with a training tick's geometry, because the attention learner has only ever predicted
behind a full window.
- **The window.** A read anchors on the last `att_ctx` − `chunk` + 1 tokens, as a tick's first prediction sees them,
  and grows by one position per new token until it is full; then it re-anchors.
- **The cache (1.2.1).** Each block's keys and values for the window are kept, so a generated token costs one position
  of work, not a full window. The cache lives on the conversation's context and is dropped whenever the weights move.
  It equals a full forward to float order (6e-6 in logits).
- **A new chat** starts from the end of the stream the model learned from, plus a blank line, not from an empty window.

## 9. Execution on a Mac (1.2.1)

Two settings, both on by default on Apple silicon (`-1` = auto) and off elsewhere:
- `att_kernel`: attention by a Metal kernel (FlashAttention-2 style, fp32, head dim 64, causal). It never stores the
  T × T score matrix; it equals PyTorch's attention to ≤ 2e-6.
- `att_ckpt`: each attention block is recomputed in the backward pass instead of keeping its activations.

Saves write the store and weights directly, without a second copy (a save's extra memory went from 2.8 GB to 0.5 GB
at 768). On an 8 GB M-series Mac the 768×12 model trains at about 2.1k bytes/s with a peak near 6 GB, loads in
3.7 GB, and chats at 19–31 ms per token.

## 10. Settings

**Shape** is fixed when a model is created; a continuation keeps it.

| setting | default (new model) | meaning |
|---|---|---|
| `learner` | attn | `attn` (§4) or the legacy `mlp` |
| `hidden`, `depth` | 768, 12 | learner width and blocks (the cloud agent's `width` is accepted as an alias) |
| `att_heads`, `att_ctx`, `att_inject` | 12, 1024, every | the attention learner's heads, window and where the store enters |
| `final_norm` | 1 | a LayerNorm before the output; checkpoints from before 1.0.1 open with 0 |
| `bound_mode` | wd | how the learner is bounded (§7): `wd`, `radial` or `sphere`; checkpoints from before 1.1 open with `radial` |
| `p2`, `p3` | 1048576, 2097152 | order-2 and order-3 context pools |
| `np1`, `np2`, `np3` | 1048576, 8388608, 16777216 | pair pools per order |
| `k1`, `k2`, `k3` | 2048, 256, 128 | successor slots per node |
| `slot_page` | 16 | slots per page (0: one dense row per node, as before 1.2) |
| `alltime_cap` | 33554432 | the all-time rung halves when its total passes this many tokens |
| `ladder` | 5,9,13,17,21,25 | rung exponents, φ^e tokens |
| `window`, `emb`, `proj` | 16, 64, 256 | learner inputs (`window` and `emb`: legacy learner only) |
| `ring`, `minl` | 65536, 3 | match model |

**Dynamics** may change on a continuation.

| setting | default | meaning |
|---|---|---|
| `chunk` | 512 | tokens per tick |
| `lr`, `warmup` | 0.0002, 200 | the learner's fixed rate, and the steps over which it ramps up |
| `io2`, `lr_base` | 0, 0.003 | io2 and its base rate (legacy models) |
| `mix_adam`, `gate_lr`, `wd` | 0.003, 0.02, 0.002 | mix, gate, weight decay (checkpoints from before 1.1 keep 1e-4) |
| `att_kernel`, `att_ckpt` | -1, -1 | Metal attention kernel and block recompute (§9); -1 = on for Apple silicon |
| `learn_stride`, `learn_mode` | 1, skip | decimation (§5) |
| `tf32` | 1 | CUDA only: fp32 matmuls on tensor cores |
| `learner_bf16` | 0 | CUDA only: the learner in bf16 autocast, with weights and Adam in fp32 |

Settings marked dev in `runtime.py` (rate controllers, rehearsal, skip contexts) are experiments, not part of the
release.

**Sizes at the defaults:** 106M learner parameters, a ~3.4 GB checkpoint (0.8 GB of it the store).

## 11. Checkpoint

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

**Compatibility:** 1.2.1 opens checkpoints from 0.1.0–0.1.5, 1.0.0, 1.0.1, 1.1.0, 1.2.0 and 1.2.1. Settings a
checkpoint's config lacks take the values its own release used (the legacy learner, io2, dense slots, the old ladder),
so an older model opens unchanged. A checkpoint without `final_norm` opens with it off; one without `bound_mode` opens
with `radial` and keeps its own `wd`. A continuation keeps the model's shape even when the request differs.

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

**1.2.0.** The attention learner is adopted (§4), with the store's slots ×8, pair pools ×2, the ladder at φ^5…φ^25,
`proj` 256, paged slots and a fixed rate. After one pass over 1 GB of fineweb-edu, a 1024×12 model scored 1.165
bits per byte on the next 50 MB while still learning, against 1.178 for a 161M-parameter transformer trained offline for
one epoch on the same 1 GB and then frozen (one seed each). The 768×12 model scored 1.199 on the same 50 MB.

**1.2.1.** The Mac release (§8–9): the chat cache, the Metal attention kernel, block recompute on by default, saves
without a second copy, and a numpy index for the context pools (909 MiB of Python objects became about 70 MB at 768,
with identical answers). On an 8 GB Mac, training the 768 model went from 1.44 s to 0.89 s a tick (0.6k to 2.1k
bytes/s) and its peak from 7.7 GB to 6.1 GB; chat went from about 100 ms to 19–31 ms a token. Same maths throughout.

## 12. Reporting

- **Headline:** `bpb` (prequential, final mix).
- **Stats:** `bps` (source bytes/s), `life_bytes`, `bpb_exact` (the store alone), `bpb_learner` (the learner alone),
  `lr`, `gate_exact`, `o3_pair_fill`.
- **Also reported:**
  - `radial_pressure` (legacy models);
  - per-order fill, recycles and overflows;
  - halvings;
  - io2's ratio, energy, coherence and band;
  - `residual_spectrum` / `residual_baseline` / `residual_excess` by band;
  - the surprise of recent tokens and the top predictions.

## 13. Reading turns and skills (1.5.0)

Nothing in the architecture, the state or the checkpoint changes; 1.5.0 changes what happens at two tokens.

- **Reading turns** (James, Nov 22). While a file is read, wherever the model's most likely next token is its own turn
  (`<|soma|>`), it takes the turn exactly there: `<|soma|>`, a reply of up to its length, `<|turn|>`, then
  `<|soma_state|>`, and the file goes on from that token. Each tick is checked before it is learned: the same tick is
  read without learning on a copy of the stream (an exact copy of its matcher; the store and weights untouched), and the
  first file position whose argmax is `<|soma|>` is taken. A position is taken at most once. The check costs one extra
  forward pass per tick, with no backward. A score never interrupts; force read (an engine setting)
  reads straight through.
- **Skills.** While it writes, wherever its most likely next token is `<|soma_state|>` (a file begins), the model has
  asked for a file. What it wrote since its turn began (or its last file) decides what it gets: a link it has read →
  that page as text, with its links; a code block at its end → what the code printed (python in a webassembly sandbox,
  no network, 10 s, 256 MB); anything else → search results for its last line. The answer is read in as a file and
  `<|soma|>` gives it the turn back; files don't count toward its length; at most 16 asks a turn. The crawler reaches
  only public addresses (checked on the address connected to, every redirect), honours robots.txt, takes one request a
  second per site, and follows only links the model has read (typed to it, in a file, a page or search results), so it
  can't carry a conversation out inside a link it made up. Caps: 3 fetches or searches a turn, 30 an hour, 10 runs a
  turn. Skills run where the model runs, and are off unless an engine turns them on (`enable_skills`).
- **The vocabulary is unchanged**: both triggers use controls the tokenizer already has, so every model can learn them.

