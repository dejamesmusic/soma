# soma-1b: what it has shown, 2.4 billion tokens in

james blight · 2026-11-28 · version 2 · result · verdict: alive

## current evidence note

editorial review 28 november 2026, for this specific notebook version. the original title and experiment text below are historical.

evaluation protocol: **frozen_held_out, online_held_out**. one seed each, one dataset, one size; equal tokens, not equal compute (1.66x the transformer's time per token today); the transformer sees one 1,024-token window at a time; longer-context or cached transformers are untested; from an empty store with weights frozen, soma-1b scored worse than the transformer; the held-out test used the pretrained checkpoint, not the 2.4-billion-token model; scores of reads after release and of the second billion are training-stream scores, not held-out; the 28 november trainer speed is measured, not yet in a published runtime.

cost/accounting scope: equal training tokens, not equal compute; on the same H100 SXM the 28 november trainer takes 1.66x the transformer's time per token (same maths as the published runtime; parity checked at full size).

[structured evidence and available material](https://logossoma.com/api/evidence.json) · [developments](/developments)


## original summary

soma-1b is a 1.06b-parameter model that keeps learning after release: an attention learner plus an exact store of what it has read, carried in the checkpoint. on 100 mb of text neither model had seen, it scored 0.916 bits per byte while learning, against 0.946 for a matched 1.04b transformer trained on the same first billion tokens; with both frozen, 0.919 against 0.950. names it read stayed nearly free to predict thousands of tokens after they left its window. since release it has read about 1.4 billion more tokens; the second billion of fineweb-edu cost about $45. training now takes 1.66 times a transformer's time per token on one h100, down from about 5. the lead comes from the memory: started empty and frozen, the transformer is ahead. one seed each.

published: 2026-11-27 21:01:01.434758+00:00

evidence: one seed each · first billion: 1,011,515,392 fineweb-edu tokens (text sha256 aadfb60d…), 8 lanes of 1,024 · held-out: 100,009,572 bytes past the first 7.6 gb of the same stream, unseen by both (sha256 74b4586c…), 26.6m tokens scored · matched transformer: soma-1b's learner blocks without the store, 1,040,732,160 parameters, best of four rates · speed: one h100 sxm, same data and settings, 2–3 timed runs each · reads after release: logOS read records, training-stream scores

[latest version](https://logossoma.com/api/developments/soma-1b.md)

## in short

- **ahead of a matched transformer on unseen text**: 0.916 bits per byte learning, 0.919 frozen, against 0.946 and
  0.950. same first billion tokens, same lanes, about 3% better.
- **the lead is the memory**. started from an empty store with its weights held, it scores 0.963, behind the
  transformer. what it builds while reading is what puts it ahead.
- **it remembers past its window.** about 3,000 tokens after names from an article left its 1,024-token window,
  three of them cost under 2 bits to predict, down from 37–66; a fourth fell from 89 to 18.
- **it kept learning after release.** about 1.4 billion more tokens through the product; the second billion of
  fineweb-edu cost about $45.
- **the memory costs compute.** training takes 1.66 times a transformer's time per token on one h100, down from about
  5 for the first billion.

## what soma-1b is

a causal attention learner (2048 wide, 20 deep, a 1,024-token window) beside an exact store: counts of what followed
each recent context, kept on a ladder of timescales from a few tokens to everything it has read, and fed into every
block of the learner. both keep learning while it reads: the store by counting, the weights by gradient. the store is
part of the checkpoint, so what the model has read travels with it. 1,057,581,284 parameters, and a 0.82 gb store.

## against a matched transformer

the control is soma-1b's own learner blocks without the store: 1,040,732,160 parameters, the same 1,011,515,392
tokens of fineweb-edu, read once in the same eight lanes, at the best of four learning rates. each 1,024-token window
starts fresh, as in any standard transformer. both were then scored on 100 mb of fineweb-edu taken from the same
stream after its first 7.6 gb, so neither had seen it. every token was scored before anything learned from it.

| bits per byte, lower is better | frozen | learning |
|---|---:|---:|
| transformer | 0.950 | 0.946 |
| soma-1b, from its own memory | **0.919** | **0.916** |
| soma-1b, from an empty store | 0.963 | 0.937 |

![training curves for both models on the same billion tokens, and the held-out scores](https://logossoma.com/api/developments/soma-1b/files/soma-1b-vs-transformer.png?version=2)

frozen means weights held; soma's store still counts what it reads, as a transformer's window still fills.

- **each model did best in the regime it was built for, and soma's came out ahead.** learning took the transformer
  from 0.950 to 0.946. it took soma, from empty, from 0.963 to 0.937. a transformer is trained, then frozen; soma is
  trained to keep going.
- **the edge is something soma carries.** its store of the first billion travels in the checkpoint. a transformer
  would have to re-read text to get it back into its window.

## it remembers past its window

at 370 million tokens, soma-1b read an article, then about 3,000 tokens of unrelated text, so the article had left
its 1,024-token window. then we measured what the names from the article cost to predict.

![bits to predict names from the article, before and after reading it](https://logossoma.com/api/developments/soma-1b/files/names-after-reading.png?version=2)

the names became nearly free to predict. the control, a common word, didn't move. a windowed transformer has no direct
access to anything past its window.

## it kept learning after release

- **pretraining**: the first billion tokens took 27.5 hours on one h100, for about $121. its final stretch scored
  0.925 bits per byte.
- **through the product**, 22–25 november: about 420 million tokens of search traces, tool-use traces, conversation
  sets, film scripts, enwik9 and general text, much of it in lanes side by side. files read a second time scored
  lower: 0.940 then 0.852, 0.379 then 0.369, 0.481 then 0.442. one went the other way: film scripts scored 1.292,
  then 1.325 a day later, after other reading.
- **the next billion of fineweb-edu**, 26–27 november: 16.6 gpu hours on an rtx pro 6000, $44.55. its last stretch
  scored about 0.873. that's less than a cent for each novel's worth of text.

about 2.4 billion tokens in all. these are scores of each text as it was read; the held-out test above used the
pretrained checkpoint.

## the cost of the memory

![training speed in tokens a second on one h100, soma against the matched transformer](https://logossoma.com/api/developments/soma-1b/files/training-speed.png?version=2)

on one h100 sxm, with the same data and settings: the transformer trains at 63.9k tokens a second. soma-1b's first
billion ran at 12.9k, about 5 times the transformer's time per token. the published runtime's trainer, with its speed
settings on, now runs at 32.8k, and with this week's changes at 38.4k: 1.66 times.

this week's changes are the same maths, done with less waiting and copying: no forced waits between the CPU and GPU,
the store's read adding this tick's counts once instead of once per context length, and the attention window kept in
place instead of copied. checked at full size against the published code, the scores differ by less than two runs of the same code
differ on the GPU. these changes are now published in [somatic core 1.6.1](https://logossoma.com/api/runtimes/somatic-core/releases/1.6.1/runtime.json) (28 november). the joined-window path is on by default; one lane keeps the single-stream read exactly, including when activation checkpointing is enabled. the 38.4k figure is the measured H100 setup above, not a speed guarantee for other hardware or settings.

where the remaining time goes, per round of eight 1,024-token ticks: reading the store, about 36 ms of 195 ms of GPU
work; attention over a sliding 1,024-token band, twice the area of a fresh window; and launching about 4,000 small
kernels.

## what it means

- **experience is an asset you can keep.** what soma-1b has read is in its checkpoint, and it is worth about 3% on
  unseen text at this size. a frozen model's experience ends at its training run.
- **learning after release is cheap.** a billion tokens of reading cost about $45, a fraction of pretraining.
- **one model can learn from many people at once.** lanes run many seats' sessions through one model, each learning
  as it goes. see [lanes](https://logossoma.com/developments/lanes).

## what it doesn't show yet

- variance: one seed each, one dataset, one size.
- the strongest control: the transformer reads one 1,024-token window at a time. one with a longer window, a cache
  across windows or retrieval would be harder to beat.
- equal compute: the comparison is equal tokens; soma still costs 1.66 times the transformer's time per token.
- today's model: the held-out test used the pretrained checkpoint, not the 2.4-billion-token model.
- an assistant: it's a base model. fluent, but it drifts and invents facts.

## the record

this entry replaces three: [soma-1b: a new frontier](https://logossoma.com/developments/soma-1b-a-new-frontier?version=2) (19 november),
[soma-1b, 1.4 billion tokens in](https://logossoma.com/developments/soma-1b-1-4-billion-tokens?version=2) (25 november) and
[soma-1b against a matched transformer](https://logossoma.com/developments/soma-1b-against-a-transformer?version=1) (27 november). each
stays readable at its own address. carried over from them:

- the training curve is flattered early: each lane's store was seeded with counts of the file text before it. no
  scored text leaked; the held-out test is the clean comparison.
- the first frozen held-out scores (3.29 and 3.44) were wrong: that scoring path skipped the store. the fix holds the
  weights and lets the store keep reading.
- the first entry's version 1 gave a scaling equation that didn't reproduce its own numbers. it was withdrawn.


## attached evidence

- [gpu_profile.py](https://logossoma.com/api/developments/soma-1b/files/gpu_profile.py?version=2) · code · 6803 bytes · sha256 `52324d9ee97424a84725c82036e058b0c9f00d2766189649e6a16f4d50c5e691`
- [heldout-results.json](https://logossoma.com/api/developments/soma-1b/files/heldout-results.json?version=2) · data · 1028 bytes · sha256 `f7921cf1781b4c8292bb578d529bfd6f1f5ab5c96c70715e13f1239062900c1c`
- [heldout-run.log](https://logossoma.com/api/developments/soma-1b/files/heldout-run.log?version=2) · log · 3083 bytes · sha256 `0254b1676e36f39f2c02ed8d17c660ff68705b1fcf1fa54942ac87dddc0cc646`
- [names-after-reading.png](https://logossoma.com/api/developments/soma-1b/files/names-after-reading.png?version=2) · figure · 55809 bytes · sha256 `8bb1f9ed794dbef7d5ec0f3dbeb344534ab819215d791a8a6b6eb951ebf168ed`
- [soma_heldout.py](https://logossoma.com/api/developments/soma-1b/files/soma_heldout.py?version=2) · code · 4782 bytes · sha256 `f5359a02f392cc0326a69aeb16dfe624c8c3a1fd5037d6c2c16d7adfcd6e0c9c`
- [soma-1b-vs-transformer.png](https://logossoma.com/api/developments/soma-1b/files/soma-1b-vs-transformer.png?version=2) · figure · 151809 bytes · sha256 `a36cd62f8e188299fadb5302bf50eb0bac359ae904417a42320da8e32434a169`
- [speed-results.json](https://logossoma.com/api/developments/soma-1b/files/speed-results.json?version=2) · data · 1837 bytes · sha256 `d9cc903ec95c02db8590784be71dc0f52ba49e442354854a85b962ba05419ee7`
- [tf_baseline.py](https://logossoma.com/api/developments/soma-1b/files/tf_baseline.py?version=2) · code · 6737 bytes · sha256 `58ec7851faf94d6803de0fcb958b82514f17a1b98a7ee9be86929de04875e5fa`
- [training-speed.png](https://logossoma.com/api/developments/soma-1b/files/training-speed.png?version=2) · figure · 59593 bytes · sha256 `736d0a7496a91406fe4e5b4f6324f6f03e209c004d2ca057fcf13060f4bf28d4`
- [transformer_log_every8.jsonl](https://logossoma.com/api/developments/soma-1b/files/transformer_log_every8.jsonl?version=2) · log · 87517 bytes · sha256 `457afddc7c7ba4c6f7a36989a0ef45220d4c61ba05d716d5628e4406bf5129c4`
