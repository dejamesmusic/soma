# logOS v12.2 web handover

## current product claim

soma is a local continuous learning app. the plain-language claim should be:

train your own language model locally, continuously, from scratch, in a normal mac app.

avoid leading with architecture. lead with the loop:

- choose text
- train a model
- watch it learn
- save the checkpoint
- talk to it
- keep training it
- upload/share it through logOS

## current architecture

the shipped runtime is `soma v12.2 serial`.

high-level shape:

- one trace bank turns byte history into a fixed present.
- a serial mlp/backprop stack composes over that present.
- no backprop through time.
- no transformer attention or context window.
- io2 is the production controller for plasticity and decimation.
- row-norm ceiling prevents long-run weight drift.
- dreams provide live qualitative training samples.
- dreams/chat generations are preceded by a trainable `<soma_state>` prelude
  containing runtime stats such as lr, io2, decimation, stride, and coherence.

the website should describe this as local continuous learning, not as a transformer alternative first. transformer comparisons are useful deeper down the page, but the first landing page job is comprehension.

## checkpoint metadata

new v12.2 checkpoints include these relevant fields:

```text
species: soma_v12_2_serial
architecture: soma v12.2 serial trace mlp
runtime_version: v12.2
description: user-editable checkpoint description
n_bands
hidden_dim
depth
base
batch_size
lr
lr_base
lr_auto
grad_clip
row_norm
row_norm_mult
row_norm_every
auto_mode
decimation_range
decimation_band
bytes_seen
checkpoint_history
```

fresh v12.2 models use `auto_mode: io2` and `decimation_range: 1`. this is the
normalized full spectral range, not a one-band cap.

logOS upload should parse and prefer `description` from the checkpoint file. if a user supplies a description in the website upload form, the website can treat that as an override, but the app-authored checkpoint description should be the default.

display recommendations:

- name
- creator
- description
- architecture / runtime version
- bands, hidden, depth
- bytes seen
- size
- checkpoint id
- upload date

## compatibility

v12.2 is the production line. older v10/v11/v12.1 copy should not be presented as current.

website copy should not imply old checkpoints are compatible with the new runtime. if upload parsing detects a different `species`, mark it as legacy or reject it with a clear message.

recommended species allowlist for new uploads:

```text
soma_v12_2_serial
```

## user comprehension priorities

stage the page like this:

1. plain claim: train a local ai that keeps learning.
2. short app workflow: data -> train -> dream -> chat -> save/share.
3. screenshots/gifs of the app training and dreams.
4. checkpoint gallery/logOS: models as living artifacts.
5. technical explanation: trace bank, serial learner, io2, continuous learning.
6. download section: mac app first, script bundle second.

avoid making the landing page feel like a paper. the concept is deep, but the first conversion point is practical curiosity.

## copy consistency

preferred terms:

- soma: the local training app/runtime.
- logOS: checkpoint sharing and remote chat layer.
- checkpoint: saved model artifact.
- dream: generation during training.
- io2: automatic controller.
- decimation: sampling/metabolic rate.

avoid:

- calling v12.2 a transformer.
- implying context-window retrieval.
- promising polished assistant behavior from fresh checkpoints.
- hiding that early models produce partial/growing language.

## upload behavior

for v12.2 checkpoint upload:

1. read checkpoint with torch on the backend.
2. extract metadata fields without loading model tensors into gpu memory.
3. compute file hash/checkpoint id.
4. use checkpoint `description` as default upload description.
5. show parsed architecture stats before final publish.

backend parser should use cpu map loading:

```python
torch.load(path, map_location="cpu", weights_only=False)
```

do not run inference during upload. parsing should be metadata-only.

## current deliverables

current app and bundles are produced from:

```text
/Users/jamesblight/spectral_v2/soma_v12.1
```

release files:

```text
dist/soma.dmg
dist/soma-v12.2-script-bundle.zip
dist/soma-script-bundle.zip
```

## open website improvements

- add a short “what happens when you train” section using real dream logs.
- make checkpoint pages show bytes seen and architecture shape.
- add a compatibility badge: v12.2 serial.
- add a small note that checkpoint descriptions can now be authored in the app.
- when explaining chat/dream behavior, mention that soma can condition on a
  small text description of its own current state before generating.
- make upload parsing robust to missing optional fields.
- keep technical pages available but not required for first-time comprehension.
