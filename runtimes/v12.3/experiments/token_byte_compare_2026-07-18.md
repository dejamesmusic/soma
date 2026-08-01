# v12.3.1 token transition result

## question

at the same parameter count and source data, does the fixed token alphabet
improve source-domain learning and wall-clock throughput over v12.2 bytes?

## controlled setup

```text
corpus               meaning_curriculum_01.txt
training source      first 3,000,000 bytes
held-out source      next 500,000 bytes
bands                12
depth                3
batch                512 model symbols
lr                   fixed 0.001
controllers          off
decimation           off / stride 1
row ceiling          off
seed                  7
device                apple mps
```

parameter matching:

```text
v12.2 byte    hidden 858    4,327,752 parameters
v12.3 token   hidden 512    4,333,570 parameters
```

## result

| runtime | train source bytes/s | held-out nats/source byte | held-out bpb |
|---|---:|---:|---:|
| v12.2 byte | 14,970 | 1.615 | 2.330 |
| v12.3.1 token | **34,721** | **1.436** | **2.072** |

v12.3.1 was 2.32x faster in source traversal and reduced held-out code length
by 11.1%. the token model used 0.2995 tokens per source byte on the held-out
slice. quality is numerically unchanged from the initial v12.3 run. translated
256-wide residual telemetry costs about 8% throughput at this unusually small
shape and should be a smaller fraction of larger-model compute.

the v12.2 sample after training was `ivartion:`. the v12.3 sample already
contained multi-clause word-level text. neither short sample is a fluency test;
it is supporting evidence that the intended first-layer work moved into the
shared dictionary.

## systems finding

the first token reader returned variable-size batches at whitespace seams.
apple mps compiled many graph shapes and resident memory rose to 11.5 gb. the
run was stopped. the production reader now keeps bounded read-ahead and emits
exact fixed token batches except at corpus end. after the correction, resident
memory stayed approximately 380-405 mb through the 4.33m-parameter token half.

## conclusion

the transition cleared both gates required for production integration:

1. source-normalized held-out quality improved at matched parameters.
2. source-byte throughput improved materially without accumulating memory.

the test does not establish long-run fluency or prove that 8,192 is the global
optimal vocabulary size. it does justify v12.3 as the next production species.
