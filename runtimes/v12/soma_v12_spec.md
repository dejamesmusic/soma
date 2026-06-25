# soma v12 spec

v12 keeps the stacked trace-bank architecture and now defaults to the io2
input/output coherence controller.

## spectral mode

the model still predicts the next byte from the trace bank. after prediction,
the byte residual is written into a second trace bank. bandpassed residual
energy is then used for two jobs:

- scale each band's gradient contribution by the coherent error energy at that
  band.
- choose the current decimation target from the residual spectrum center.

`spectral` also keeps global loss-level scaling for lr and max_change.

`full spectrum` keeps the residual-spectrum band scaling and decimation, but
removes the global loss-level scaling.

`io2` is experimental. it keeps residual-spectrum band scaling, then compares:

- input coherence: how structured the trace-bank present is.
- output coherence: how concentrated the model's predicted distribution is.
- residual concentration: how sharply error is localized across bands.
- progress: whether loss is moving.

the controller learns densely when the input looks structured and the model is
underfit or confidently wrong. it traverses/skips more aggressively when the
input looks incoherent or loss is below the homeostatic target.

## current fresh defaults

```text
bands 50
base 1.6180
hidden 1024
layers 3
auto_mode io2
lr auto
max_change auto
batch 256
decimation_auto true
decimation_range 1.0
dream_every_batches 50
dream_length auto 200
dream_temperature 1.0
```

explicit io2 syntax is also accepted:

```text
lr io2 1.0
max_change io2 1.0
auto_mode io2
```

## retired but compatible fields

these fields remain in the checkpoint loader so older experiments can resume,
but they are not part of the fresh gui path:

- direct_readout
- scale_gate
- clock
- weight_decay
