# lanes: one model, many seats, all learning at once

james blight · 2026-11-25 · version 1 · engineering · verdict: alive

## current evidence note

this version has no structured protocol review yet; consult its methods.


## original summary

a soma model can now hold several conversations and reads at once on one gpu, every one of them updating the same weights. each conversation keeps its own store; the network is shared. on an h100, soma-1b runs 24 lanes in 73 GB, learning at 15.6k tokens a second across them and writing about 8 tokens a second to each seat when all 24 write together. on an rtx pro 6000, 22 lanes, two thirds of the learning speed, twice the writing speed per seat. confirmed in production with two accounts, two conversations and two reads on one model at the same time. in production the lanes still share one stream's throughput, and reading through the product costs several times what pretraining does: that is the next work.

published: 2026-11-25 09:47:49.526309+00:00

evidence: soma-1b (1.06B) and a fresh model of its shape · tick 256 · lanes_bench, warm · h100 sxm and rtx pro 6000 server, runpod · production machine logs and read records, nov 22–25

[latest version](/api/developments/lanes.md)

work of 2026-11-22 to 11-25. runtime somatic core 1.6.0; engine 2.1.x.

## the short version

a model runs on a clock. on every tick, every lane that has something to do takes 256 tokens: a person's words, a
file being read, or the model's own reply. all lanes learn on the same tick, from their own tokens, into the same
weights. each lane holds one conversation's store: its recent context and counts. the network is shared.

so one model can talk to many people at once, and everything any of them teaches it goes into the one set of weights.

## what a lane is

- **a seat** is a person who can use a model. **a lane** is one conversation or read running at once. seats queue for
  lanes: one queue, the first free lane.
- **a lane** is a chat store plus a 256-token tick's working memory. the store is the conversation's; the weights are
  everyone's.
- **idle lanes are masked**: they take no compute and nothing is learned from them.
- **each store keeps its own clock**, counted in its own tokens. a conversation nobody is using doesn't decay while
  others talk.
- **chats come first.** a file read never takes the last free lane.
- **when every lane is busy**, the least recently used idle conversation is saved to disk and swapped out.

## memory and speed

soma-1b, tick 256, every lane reading and then writing, on an h100 (80 GB):

| lanes | peak memory | learning, all lanes | writing, per seat |
|---:|---:|---:|---:|
| 8 | 33.8 GB | 11.5k tok/s | 20.6 tok/s |
| 16 | 53.3 GB | 11.7k tok/s | 10.3 tok/s |
| 20 | 63.6 GB | 14.6k tok/s | |
| 22 | 68.3 GB | | |
| 24 | 73.2 GB | 15.6k tok/s | 8 tok/s |
| 26 | 76.4 GB | too tight | |

a fresh model of the same shape on an rtx pro 6000 (96 GB, blackwell):

| lanes | peak memory | learning, all lanes | writing, per seat |
|---:|---:|---:|---:|
| 16 | 62.5 GB | 9.6k tok/s | 17.4 tok/s |
| 20 | 75.2 GB | 9.7k tok/s | 17.0 tok/s |
| 22 | 81.5 GB | 9.9k tok/s | 15.6 tok/s |
| 24 | out of memory | | |

each lane costs about 2.4 GB on the h100 and about 3.2 GB on the pro 6000. the pro 6000 learns at about two thirds of
the h100's speed, which is the same per dollar, and writes to each seat about twice as fast.

## what makes it correct

- one lane is bit-identical to the single-stream runtime: the same bits, the same state, the same next distribution.
- three people in three lanes: each store ends exactly as it would have alone.
- batched decoding matches decoding one at a time, to float order.

## in production

- nov 24: two accounts, two conversations and two file reads on soma-1b, all at once.
- the overnight run of nov 24: six reads side by side for 8.35 hours, 947 MB, about 8.4k tokens a second in all. one
  read alone runs at about the same total speed. in production, the lanes split one stream's throughput rather than
  adding to it.
- so reading through the product cost about $146 per billion tokens, against $25–40 in the training harness.

## around it

- **auto read**: an owner can give some lanes to the model's own files. those lanes read the whole folder, laid end to
  end, on a loop, evenly spread, each keeping its store until a seat needs the lane. they never count as activity, so a
  machine still stops when nobody is there.
- **warm hours**: a model can keep hours when its machine is already up, started a little before. inside them it
  neither idles out nor hits the 8-hour cap.

## what's honestly unknown

- why a lane costs a third more memory on the pro 6000 than on the h100.
- how many seats share a lane in real use. we assume six to ten.
- how fast a swapped-out conversation comes back.
- why production lanes don't add throughput the way the benchmark's do.

## where this goes next

close the production throughput gap; fit more lanes per gpu (idle stores in host memory, smaller chat stores);
capture fixed-size ticks as cuda graphs, so more lanes stay quick.


