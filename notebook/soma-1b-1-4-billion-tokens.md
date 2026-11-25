# soma-1b, 1.4 billion tokens in

james blight · 2026-11-25 · version 2 · result · verdict: open

## current evidence note

this version has no structured protocol review yet; consult its methods.


## original summary

since its pretraining (1.01 billion tokens of fineweb-edu, 0.925 bits per byte in its final window), soma-1b has read about 1.6 GB more through the product itself: search traces, tool-use traces, conversation worlds, dialogue curricula, film scripts, enwik9 and a general corpus, much of it side by side in lanes. about 1.43 billion tokens in all. files read a second time scored lower than the first time; one scored higher after the others. these are scores of the stream as it was read, not held-out tests. the pretrained checkpoint is kept; the model now lives in a new data center.

published: 2026-11-25 20:48:43.618748+00:00

evidence: logOS read records for soma-1b, nov 22–25 · bits per byte reported by each read as it ran, learning on · h100 sxm and h100 nvl, runpod · pretraining as in soma-1b: a new frontier

[latest version](/api/developments/soma-1b-1-4-billion-tokens.md)

reads of 2026-11-22 to 11-25, from logOS's records. each figure is what the read reported while it ran, learning on.

## the short version

pretraining gave soma-1b 1.01 billion tokens. since then it has kept learning the way it's meant to: from what people
give it to read. about 1.6 GB more, roughly 420 million tokens, for about 1.43 billion in all.

## what it read

| file | read | MB | bits per byte |
|---|---|---:|---:|
| search traces v1 (40,000 turns) | nov 22 | 128.4 | 0.462 |
| tool-use traces | nov 22 | 148.4 | 0.379 |
| conversation worlds v4 | nov 22 (stopped) | 83.9 | 0.481 |
| allcombined | nov 23 | 178.7 | 0.940 |
| tool-use traces, again | nov 24 | 148.4 | 0.369 |
| allcombined, again | nov 24 | 178.7 | 0.852 |
| conversation worlds v4, again | nov 24 (stopped at the cap) | 255.5 | 0.442 |
| meaning curriculum 02, dialogue worlds | nov 24 | 205.5 | 1.207 |
| enwik9 | nov 24 (stopped at the cap) | 141.9 | 0.977 |
| moviecorpus | nov 24 | 17.0 | 1.292 |
| moviecorpus, again | nov 25 | 17.0 | 1.325 |

the six reads of nov 24 ran side by side, in lanes, overnight on one h100: 8.35 hours, 947 MB, about $36.65.

## what it shows

- **second readings score lower.** allcombined 0.940 then 0.852. tool-use traces 0.379 then 0.369. conversation
  worlds 0.481 then 0.442. the model kept something of each.
- **one went the other way.** moviecorpus scored 1.292, then 1.325 a day later, after other reading in between.
- **corpora differ.** the traces are regular and score low; film scripts and dialogue curricula score high. scores
  across files are not comparable.

## where it is now

- it lives on a volume in eur-is-1. it moved there through logOS on nov 25, 34.4 GB with its stores, in two hops that
  finished on their own.
- the pretrained checkpoint, as pretraining left it, is kept on logOS. the complete pretraining log is archived.

## what's honestly unknown

- **no held-out test.** every number above is the stream scored as it was read. a second reading's lower score could be
  learning, or memory of the first. a held-out test would tell them apart.
- **interference.** moviecorpus rising after other reading is one observation, not a measurement of forgetting.
- **conversation.** with nothing useful in its store, it drifts between english and not, and falls into loops. given
  context, it does better. how much context it needs is not measured.


