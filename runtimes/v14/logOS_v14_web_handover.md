# logos v14 checkpoint handover

this species-specific handover is subordinate to
`logOS_ecosystem_strategy.md` and `logOS_runtime_submission_standard.md`. use
the universal ecosystem envelope for admission, identity, lineage, and runtime
routing. use this document only for deeper v14 validation and display.

v14 is a reference runtime submission, not a privileged architecture. its
checkpoint adapter, package, permissions, fixtures, and verification results
should be represented through the same contracts available to any third-party
runtime. no universal parser rule may be derived solely from v14 internals.

## parser decision

v14 is a checkpoint species:

```text
soma_v14_adaptive_trace_branch_mixer
```

do not apply tensor-key requirements such as `weights`, `traces + net`, `w_0`,
`traces_0`, `w`, or `organs` in the universal upload path. those names describe
particular architectures and will incorrectly reject other valid species.

new v14 sidecars include `ecosystem.schema = logos.checkpoint.v1`. route them
to runtime id `soma.v14` and checkpoint format `soma-v14-pytorch`. sidecars
without that envelope remain valid through the v14 adapter.

## safe upload path

accept the checkpoint as:

```text
model.pt
model.pt.meta.json   optional but strongly preferred
```

use the json sidecar for website display and validation. do not deserialize an
untrusted pytorch checkpoint in the web process. when no sidecar is supplied,
store the checkpoint as an opaque artifact and either request the sidecar or
extract metadata in a disposable, network-isolated worker with strict resource
limits.

## sidecar schema

required for a fully described v14 listing:

```text
sidecar_version       4
ecosystem             logos.checkpoint.v1 envelope
species               soma_v14_adaptive_trace_branch_mixer
architecture          soma_v14_adaptive_trace_branch_mixer
checkpoint_version    14
runtime_id            soma.v14
runtime_version       14.0.0-dev.2
checkpoint_format     soma-v14-pytorch
capacity              small | medium | large | custom
n_bands               integer
base                  number
token_dim             integer
scale_rank            integer
scale_depth           3
summaries             integer
model_dim             integer
language_depth        integer
language_expansion    integer
batch_size            integer
lr                    number
lr_base               number
lr_reference_batch    integer, currently 1024
lr_batch_scale        number
lr_auto               boolean
decimation_gain       number
lattice_rate          number
lattice_drift         object with max_abs and mean_abs
lattice_coordinates   array of n_bands numbers
decimation_band       number
prediction_stride     integer
io2_plasticity        number
grad_clip             number
update_interval       integer
radial_compression    boolean
radial_ceiling_scale number
radial_floor          number
pulse_enabled         boolean
pulse_quantum         integer, normally equal to batch_size
pulse_schedule        spectral | factor
bytes_seen            integer
tokens_seen           integer
params                integer
checkpoint_id         string
revision_id           string
tokenizer             soma-1k-v1
tokenizer_sha256      tokenizer artifact digest
tokenizer_vocab       1024
```

optional/editorial fields:

```text
description
name
tagline
art
about
weight_spectrum
radial_pressure
row_spread
```

preserve unknown fields when a user edits or replaces a profile. the app merges
existing editorial fields into the next sidecar save.

## v14 adapter validation

these checks grant v14-specific described/runnable eligibility after universal
admission. they are not global checkpoint requirements.

reject the v14 claim when:

- species/architecture is not `soma_v14_adaptive_trace_branch_mixer`;
- runtime major version is not `14`;
- tokenizer is not `soma-1k-v1` or vocabulary is not `1024`;
- tokenizer digest is absent or does not match the registered tokenizer;
- ids are absent or malformed;
- dimensions, counts, or file size are non-positive;
- the declared checkpoint exceeds service limits.

an artifact that fails this adapter may still remain preserved. do not mark it
runnable with v14 unless the adapter and isolated runtime smoke test pass.

do not reject an upload merely because band count differs from 48. the learned
shape is fixed while the exact temporal lens is configurable.

## lineage behavior

use `checkpoint_id` as the stable model identity and `revision_id` as the
immutable artifact identity.

- same checkpoint id + new revision id: another save in the same life;
- new checkpoint id: a distinct model;
- duplicate revision id: the same artifact and should be idempotent.

never infer identity from filename alone.

## display copy

recommended concise shape line:

```text
v14 adaptive trace branch mixer · 48 bands · d1024 · depth 4 · 28.2m parameters
```

recommended experience line:

```text
<bytes_seen> source bytes · <tokens_seen> tokens · soma-1k-v1
```

call `bands` the temporal lens. call `language_depth` composition depth. call
`prediction_stride` the current prediction stride and `decimation_gain` the
controller range. v14 is io2-controlled: every source token enters the exact
trace while io2 may sample fewer presents for model work. do not call that
trace decimation, consolidation, an organ system, or v12 compatibility.

`weight_spectrum` is the mean-one learned temporal allocation inferred from the
context projection. `radial_pressure` is the mean fraction of future outward
row growth the soft compressor would attenuate, and `row_spread` is the global
p99/median learned-row norm ratio. pulse fields describe the saved optimizer
schedule; new v14 models normally use `pulse_enabled=true` with `spectral`.

## download pairing

serve the `.pt` checkpoint as the primary download and the sidecar beside it.
the app can load the checkpoint without the sidecar, but keeping both preserves
branding and lets logos inspect it without executing pytorch serialization.

## reference runtime registration

register the v14 application and script bundle as immutable platform packages
under `runtime_id = soma.v14`. record their server-computed digests, source
commit or source snapshot, build recipe, dependency lock, specification,
tokenizer assets, license, declared permissions, and platform requirements.

v14 should pass the common runtime suite for inspect, text inference, bounded
training, save/reload, resume, stop, and cleanup. v14-specific architecture
checks remain an additional adapter suite; they do not replace the common
suite and do not become requirements for another species.
