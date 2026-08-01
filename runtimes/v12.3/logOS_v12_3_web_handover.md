# logOS v12.3.17 web handover

## production identity

the current downloadable runtime is **soma v12.3.17 token residual**.

```text
species:          soma_v12_3_token_residual
runtime_version:  v12.3.17
architecture:     token spectral residual mlp
tokenizer:        soma-8k-v1
vocabulary:       8192
token_dim:        256
```

v12.3 checkpoints are not compatible with v12.2 byte checkpoints. do not label
v12.2 models as current or imply that the website can convert them.

## product description

soma is a mac app for training a language model locally and continuously. a
user chooses text, starts a model, watches it learn, stops whenever they like,
then resumes the same checkpoint on the same or a different corpus. chat can
continue updating the model. the artifact is never considered finished.

v12.3 uses one shared, lossless 8k token dictionary so the network can learn
composition rather than relearn spelling in every checkpoint. time is still
held by soma's geometric trace bank, not a context window or kv cache.

preferred plain-language line:

> train your own ai, on your own machine, for free, for as long as you like.

avoid calling soma a transformer. it borrows a fixed token vocabulary, tied
input/output dictionary and residual depth, but its memory and continual
learning mechanism are the trace bank.

## checkpoint parser

prefer the adjacent `<checkpoint>.meta.json`; it avoids loading a large torch
file in the web process. parse these fields:

```text
species
runtime_version
architecture
checkpoint_id
revision_id
tokenizer_name
tokenizer_sha256
vocab_size
token_dim
n_bands
hidden_dim
depth
base
batch_size
description
bytes_seen
tokens_seen
lr
lr_base
grad_clip
decimation_range
decimation_gain
decimation_band
bytes_per_token
byte_stride
params
row_clip_fraction
row_compression_fraction
layer_maturity
saved_at
```

the sidecar also contains a user-owned `profile` object:

```json
{
  "profile": {
    "name": "my soma",
    "tagline": "local continual language model",
    "art": "three lines of optional monospace art",
    "about": "a human-written description of this checkpoint"
  }
}
```

the app preserves `profile` whenever it saves the checkpoint. all other
sidecar fields are runtime-owned and may be refreshed on every save. accept
the `.pt` checkpoint by itself, or the `.pt` plus its optional adjacent
`.pt.meta.json`. when both are uploaded, display the profile as checkpoint
branding and retain the machine fields for validation.

required upload validation:

1. `species == soma_v12_3_token_residual`
2. `runtime_version` begins with `v12.3`
3. `tokenizer_name == soma-8k-v1`
4. tokenizer hash is present
5. architecture dimensions and parameter count are positive

display the tokenizer name and vocabulary alongside shape. reject mismatched
species with a clear “requires soma v12.3” compatibility message.

## metrics and copy

v12.3 reports:

- **nats per source byte** for learning quality
- **bits per byte** for cross-architecture code length
- **top-1 and top-5 token accuracy**, which must not be labelled byte accuracy
- **source bytes/s** for throughput
- source bytes and tokens seen
- token stride and estimated source-byte stride

the gui default is 16 bands, hidden 4096, depth 4, approximately 70.3m
parameters. fresh-model hidden width is always `256 x bands`; users control
capacity through bands and depth. io2 is intrinsic rather than selectable.

## download artifacts

publish these exact current artifacts from `dist/`:

```text
soma.dmg
soma-v12.3.17-script-bundle.zip
soma-script-bundle.zip
```

the script bundle includes the tokenizer. a runtime without
`tokenizer/soma-8k-v1.model` is incomplete.

## conceptual staging

stage the website explanation in this order:

1. **promise:** train your own ai locally.
2. **loop:** feed it text, watch it learn, talk to it, keep going.
3. **ownership:** checkpoint and training history stay with the user.
4. **difference:** no fixed training session and no finished artifact.
5. **mechanism:** a trace bank compresses elapsed time into the present.
6. **v12.3 detail:** a shared token dictionary makes language composition
   visible sooner without replacing the temporal architecture.

the first viewport should not require understanding spectral learning,
continual learning or tokenization. those ideas belong in progressively deeper
sections for users who ask why the app can do this.

## claims discipline

supported:

- local training from fresh weights on apple silicon
- pause, resume and corpus switching
- bounded streaming data memory
- continual trace state without a context window or kv cache
- parameter-matched v12.3 improvement over v12.2 in the included experiment

do not claim:

- transformer-equivalent general intelligence
- guaranteed fluency for every shape or corpus
- zero forgetting
- compatibility with arbitrary tokenizers
- cryptographic privacy when a user enables the logOS web bridge
