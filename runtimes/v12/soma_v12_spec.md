# soma v12.2 spec

v12.2 is the serial-backprop version of soma.

## architecture

the trace bank collapses the byte stream into a fixed spectral present:

```text
x_t -> trace_bank(x_<=t) -> 256 * bands features
```

v12.2 then applies serial depth inside that present:

```text
trace features -> hidden -> hidden ... -> byte logits
```

training uses ordinary backprop through the feedforward stack for the current
batch only. temporal memory remains entirely in the trace bank. there is no
backprop through time, no trace bank over hidden activations, and no learned
state carried between batches except the weights themselves.

checkpoints are a new species:

```text
species = soma_v12_2_serial
architecture = serial_trace_mlp
```

v12.1 checkpoints do not load into v12.2.

## current fresh defaults

```text
bands 20
base 1.6180
hidden 1536
depth 3
batch 512
auto_mode io2
lr auto 0.001
decimation_range 1
decimation_stride_cap 512
grad_clip 1.0
row_norm auto x4 every 100 batches
dream_every_batches 100
dream_length auto 300
dream_temperature 1.0
description saved in checkpoint metadata
```

## why this replaced v12.1

short comparisons showed that serial depth over one trace-bank present learned
faster and generated cleaner language than the belief-stack runtime at smaller
parameter counts. it also strips several controller knobs that were useful
experimentally but made the shipping app harder to understand.

the production surface is therefore simpler:

- spectral memory lives in the trace bank.
- composition lives in serial hidden layers.
- io2 is the production controller for both lr and decimation.
- model and wallclock modes remain available as experimental comparisons.
- grad clipping remains a fixed safety rail.
- row norm ceiling softly bounds weight growth after optimizer steps.
- checkpoint descriptions are saved directly in the `.pt` file for logOS.
- dreams remain the main live qualitative readout during training.
- generation is preceded by a trainable `<soma_state>` text prelude.

training corpora are opened with `numpy.memmap`, so the runtime can work over
large files without holding the whole corpus in ram.

## controller

the production controller is `io2`.

it controls:

- effective lr, when `lr_auto` is true, from error-bank energy and coherence,
  adjusted by loss relative to byte-level chance.
- decimation band from the same input, output, and residual coherence state.
- actual stride, capped by `decimation_stride_cap` so simple low-loss corpora
  cannot force runaway throughput.

io2 depends on the trace-bank input, the model output distribution, and the
residual/error trace. its default decimation value is `1`, the normalized full
spectral range. the alternate model and wallclock modes use an explicit
twelve-band range; off uses zero.

`model` is the experimental motor-controlled alternative. it uses reserved byte
`0x11` as a relative speed signal, feeds realized sampling state back into the
trace bank, and applies an attention-budget loss. it remains available for
research, but is not the shipping default.

it does not directly control grad clipping. clipping remains an explicit
numerical safety limit for backprop.

online chat prompt ingestion uses the same strided/controller path as corpus
training. context-only chat still advances the trace bank without weight
updates.

the gui maintains a corpus cursor for single-corpus training. if a run is
stopped before end of file, `start byte` is updated to the last reported
absolute corpus position. if the run finishes normally, it resets to `0`.
selecting or editing the corpus also resets the cursor; app restart preserves
the saved value.

## self-state prelude

before dreams and chat generation, v12.2 feeds a bounded text block through
the trace bank with online weight updates:

```text
<soma_state>
source: dream|chat|generation
bytes_seen: ...
lr: ...
decimation: ...
stride: ...
io2: ...
input_coherence: ...
output_coherence: ...
</soma_state>
```

this is deliberate training data, not generated text. it gives the model a
stable, parseable sensory description of its own runtime state before it
switches into generation.

v12.2 also uses ascii `0x1e` record separator as a private turn boundary for
interactive runtime streams. prompts, self-state records, and generated turns
advance the trace bank through this byte. it is not displayed to users and is
not inserted into corpus files.

dream timing is still batch-count based, but the batch before a dream tries to
end at the next newline inside its normal batch window. this prevents dreams
from regularly cutting through the middle of a sentence or paragraph while
keeping dream cadence tied to training progress.

## row norm ceiling

v12.2 uses adamw with no weight decay. gradients are clipped, but optimizer
steps can still slowly increase row norms over long continuous runs.

the row norm ceiling is checked periodically after optimizer steps:

```text
ceiling = sqrt(fan_in) * 0.1 * row_norm_mult
```

the default is `row_norm auto`, `row_norm_mult 4.0`, and
`row_norm_every 100`. rows below the ceiling are untouched. rows above it are
scaled back to the ceiling. this keeps logits away from saturation without
changing checkpoint format or forcing old checkpoints to be rewritten on load.
