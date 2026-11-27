# soma

soma is a continual learning architecture. it uses a bank of exponentially decaying context traces to provide a
spectral basis for learning over time, alongside an attention network whose weights keep updating as it reads.

this repository is everything [logOS](https://logossoma.com) makes public, in one place: every runtime version, the
lab notebook, the paper, and the language logOS speaks. the weights live on logOS.

| | |
|---|---|
| [`runtimes/`](runtimes) | every published version, from v8 to somatic core 1.6.1. each release is a tag |
| [`LINEAGE.md`](LINEAGE.md) | what each version tested, what survived, and where the evidence is |
| [`notebook/`](notebook) | the developments notebook, as published, with the evidence records |
| [`paper/`](paper) | time is all you need: the original manuscript |
| [`platform/`](platform) | the logOS language, the `logos` command line, and how programs and agents use it |
| [`tools/verify.py`](tools/verify.py) | check any release here against logOS's signed record |

## check it

every runtime release logOS serves is here, unpacked, at its own tag, with logOS's record of it beside the source.

```
python3 tools/verify.py somatic-core-1.6.1
python3 tools/verify.py --all
```

for each release: the hashes agree with logOS's record; the signature verifies (somatic core from 1.4.0); the bundle
logOS serves today unpacks to exactly the tag's files. nothing is taken on this repository's word.

## the current runtime

somatic core: bounded context–successor counts at six decaying timescales plus an all-time rung, a 65,536-token copy
ring, and a causal attention learner over a 1,024-token window that the store feeds in every block. with learning on,
its weights keep updating. specification: [`runtimes/somatic_core/SPEC.md`](runtimes/somatic_core/SPEC.md).

run it in [soma](https://logossoma.com/download) on mac or windows, or on a gpu rented through your own runpod account
from [logOS](https://logossoma.com).

## worth knowing

- results are in the notebook, each with its protocol, exposure and gaps. one seed is one seed.
- the code that produced each notebook result is being added under `research/`, pinned to the entries that cite it.
  until an entry names a revision here, its source is the release it used and its own methods section.
- research software, provided as it is, without warranty.

MIT · [logossoma.com](https://logossoma.com) · [llms.txt](https://logossoma.com/llms.txt)
