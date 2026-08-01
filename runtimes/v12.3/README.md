# soma v12.3.17

soma is a local continual language learner. give it text, stop it, resume it,
change the corpus, talk to it, and keep training the same checkpoint. there is
no context window, replay buffer, kv cache, or training sequence stored behind
the trace bank.

v12.3 keeps soma's temporal core and changes its language alphabet. v12.3.17
finishes that translation across training, reporting, corpus control and local
speech, gives online conversation an explicit learning boundary, and completes
the responsive cockpit and checkpoint identity panel. instead of
spending the network on reconstructing spelling byte by byte, every model uses
the same lossless 8,192-token dictionary. the model can spend its learned
capacity on composition while byte fallback preserves arbitrary input.

v12.3.7 replaced periodic hard row clipping with radial plasticity
compression. rows progressively lose outward magnitude growth near their
capacity boundary while inward and directional learning remain available.
existing v12.3 token checkpoints load without conversion.

v12.3.10 makes configured decimation a continuous gain on io2's native
decision. values below one attenuate its selected band; values above one
amplify it toward the unchanged stride cap. the legacy checkpoint field is
retained, but old values above one now use the new gain semantics.

```text
text
  -> immutable soma-8k-v1 tokenizer
  -> exact geometric token trace bank
  -> current learned 256-wide token dictionary
  -> serial budget-residual mlp
  -> shared dictionary readout
  -> next-token distribution
```

## install

the mac app is in `dist/soma.dmg`. drag soma to applications and open it.

the script bundle uses:

```sh
./soma gui
./soma train corpus.txt --save model.pt
./soma chat model.pt
```

on first launch the script bundle creates a virtual environment and installs
the dependencies in `requirements.txt`.

## production defaults

```text
bands       16
hidden      4096
depth       4
base        1.6180
batch       512 tokens
lr          auto 0.002
controller  io2
decimation  1
gradclip    1
dreams      every 64 batches, up to 128 tokens
```

this is approximately 70.3m parameters. fresh-model width is intrinsic:
`hidden = 256 x bands`. capacity has two exposed axes: spectral resolution
(`bands`) and compositional depth (`depth`). saved checkpoints retain their
original width when resumed.

## what changed

### one shared language alphabet

`tokenizer/soma-8k-v1.model` is shipped with every runtime. it uses lexical
sentencepiece bpe, identity normalization and complete byte fallback. it does
not contain cross-word phrase pieces. its sha256 is written into every
checkpoint and must match at load time.

### exact token time

the trace bank remains an exact, non-learned geometric compression of the
observed stream. it stores token identity by band. at every batch the retained
identity traces are projected through the *current* learned dictionary, so a
dictionary update immediately reinterprets the compressed past without bptt or
stale embedded traces.

### budget-residual composition

the first learned map sees `256 x bands` features. later layers compose within
the same present through gated residual transforms. the output is mapped back
to 256 language coordinates and scored against the same learned token
dictionary used at input.

### source and token diagnostics

the gui reports nats per original source byte and bits per byte, not nats per
token. this keeps runs comparable across alphabets. top-1 and top-5 accuracy are
token accuracy. the live view also exposes input, first-layer weight and error
spectra, active trace-memory tokens, likely next tokens and depth saturation.

## checkpoint compatibility

v12.3 checkpoints use species `soma_v12_3_token_residual`. v12.2 byte
checkpoints are intentionally not load-compatible: the output alphabet and
first learned dictionary are different. v12.2 remains available in its own
folder and artifacts.

v12.3 checkpoints include:

- architecture and controller settings
- shared dictionary and residual network weights
- adam state
- exact token traces and translated 256-wide residual traces
- source bytes and tokens seen
- tokenizer name and sha256
- description and a sidecar `.meta.json` for logOS

## continual behavior

- training data is streamed; the corpus is not loaded into memory.
- batches have fixed token shapes so mps/cuda graph caches remain bounded.
- native protocol pieces are perceived without becoming prediction targets.
  io2 and language metrics exclude them.
- online chat learns the user's external words. generated replies update the
  trace state without reinforcing soma's own samples.
- recorded dialogue keeps both speakers as learnable external language while
  masking synthetic state; reply-ending turn markers receive light supervision.
- chat and dream generation lengths, batch size and trace time are tokens.
- the gui reports the latest feedback ratio as generated dream tokens over
  external corpus tokens since the previous dream.
- dreams begin with `<|dream|>` and close with `<|turn|>` instead of injecting
  a textual telemetry prelude into the fast trace bands.
- gui stride reports both token stride and estimated source-byte stride.
- training runs in a child process in the app, so stopping releases all
  accelerator state at the operating-system boundary.
- checkpoints save atomically.
- corpus cursor positions are tracked per checkpoint and data file.
- one corpus can run for explicit epochs, or all data files can cycle for the
  selected number of epochs.

## local speech loop

the mac app can complete the conversation loop without an api or speech
service. `speak` records one utterance and places an editable transcript in the
ordinary chat input. recognition is forced through apple's on-device path and
refuses to run when the selected language cannot be guaranteed local; there is
no network fallback. silence ends an utterance automatically, or the user can
stop it manually.

`speak replies` reads generated sentences through the local macos system
voice. speech is queued independently, so generation and the interface do not
wait for playback. stopping a response also stops its voice. with online chat
enabled, a spoken prompt follows exactly the same continual-learning path as a
typed prompt.

microphone and speech-recognition permission are requested on first use.

## measured transition

on the same first 3m source bytes of `meaning_curriculum_01.txt`, with both
controllers disabled and parameter-matched at 4.33m:

| runtime | held-out bpb | source bytes/s | parameters |
|---|---:|---:|---:|
| v12.2 byte | 2.330 | 14,970 | 4.328m |
| v12.3.1 token | **2.072** | **34,721** | 4.334m |

the next untouched 500k source bytes were used for evaluation. the harness is
`experiments/token_byte_compare.py`.

## files

- `soma_v12_3.py` - production runtime
- `soma_gui.py` - mac interface
- `soma_voice.py` - local speech queue and helper bridge
- `soma_train_worker.py` - isolated app training worker
- `tokenizer/` - immutable shared language alphabet
- `soma_v12_spec.md` - complete architecture and checkpoint specification
- `cloud/train_cloud.py` - observable resumable cloud trainer
- `soma_loop.py` - perpetual rolling-corpus trainer
