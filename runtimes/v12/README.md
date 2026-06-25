░▒▓ soma v12 ▓▒░

soma v12 is the current spectral trace learner.

fresh defaults:

```text
50 bands · base 1.6180 · hidden 1024 · layers 3
auto mode io2 · lr auto · max_change auto
batch 256 · decimation range 1.0
autosave every 30 minutes in the app
dream every 50 batches · dream length auto 200 · temperature 1.0
```

the gui intentionally hides retired experimental knobs for fresh models:

- direct readout
- scale gate
- clock
- weight decay

older checkpoints that contain those fields still load.

## controller modes

fresh models default to `auto mode io2` with `lr auto` and
`max_change auto`. in this form, `auto` means "use the selected auto mode";
switching `auto mode` to `spectral`, `full spectrum`, or `progress` changes
the controller without editing the lr/max_change fields.

available auto-style lr/max_change strings:

- `spectral 1.0` — residual spectrum controls band effort and decimation;
  loss level scales global lr/max_change.
- `full spectrum 1.0` — residual spectrum controls band effort and
  decimation; global lr/max_change stay at the requested value.
- `progress 1.0` — lr/max_change follow short-term loss movement.
- `io2 1.0` — experimental input/output coherence controller. it compares
  trace-bank input coherence, model output coherence, residual concentration,
  and loss movement to choose lr/max_change and decimation.

you can still enter explicit controller strings such as `io2 1.0` or
`full spectrum 0.5` directly in the `lr` and `max_change` fields.

## terminal use

run:

```sh
./soma
```

then choose `train`, enter a corpus, and enter a checkpoint name. bare names
resolve into local folders:

- corpora: `data/`
- checkpoints: `checkpoints/`
- stream scripts: `streams/`
- stream output corpora: `data/streams/`

absolute paths still work, which is what you want on rented gpu storage.

for the current v12 setup in terminal, use:

```text
config: custom
bands: 50
range: 1.6180
hidden: 1024
layers: 3
auto mode: io2
lr: auto
max_change: auto
batch: 256
decimation range: 1.0
dream every batches: 50
dream length: auto 200
dream temperature: 1.0
```

the express path (`enter=demisa`) uses the same defaults.

## mac app build

```sh
macos/build_app.sh
```

outputs land in `dist/`:

- `soma.app`
- `soma.dmg`
- `soma-script-bundle.zip` when the script bundle is refreshed
