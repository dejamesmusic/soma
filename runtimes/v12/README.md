░▒▓ soma v12.2 ▓▒░

soma v12.2 is the serial trace learner.

the core shape is:

```text
byte stream -> trace bank -> serial hidden layers -> next-byte belief
```

the trace bank still creates the fixed spectral present. the learned model now
uses normal serial depth over that present: one projection from the trace bank,
then depth-1 hidden transforms, then a byte readout. backprop stays inside the
current trace-bank batch; there is no bptt and no learned memory between steps.

v12.2 checkpoints are a new species and are not load-compatible with v12.1.

## fresh defaults

```text
20 bands · base 1.6180 · hidden 1536 · depth 3
batch 512 · auto mode io2 · lr auto 0.001
decimation range 1 · max stride 512 · gradclip 1.0 · row norm auto x4 every 100 batches
autosave every 30 minutes in the app
dream every 100 batches · dream length auto 300 · temperature 1.0
```

the gui keeps the controller knobs that have proven useful:

- `lr`: use `auto 0.001` for progress-controlled plasticity, or a number for fixed lr.
- `auto mode`: `io2` is the production default. it uses the input, output, and residual trace banks to set plasticity and decimation. `wallclock` and `model` remain available as bounded research controllers; `off` keeps dense, fixed-rate training.
- `decimation`: is mode-aware. `io2` defaults to normalized range `1`; `wallclock` and `model` default to explicit range `12`; `off` defaults to `0`. changing auto mode in the app updates this value deliberately rather than carrying an incompatible range across modes.
- the hard stride cap is 512 bytes by default, preventing experimental modes from accelerating into unstable kilobyte-plus sampling jumps.
- row norm ceiling: enabled by default in scripts as `auto x4` every 100 batches; it only clips rows that drift above a soft norm ceiling.
- `description`: saved inside checkpoints for logOS uploads and local inspection.

large corpora are read through a memory map in v12.2, so long runs do not need
to copy the full corpus into ram before training.

before dreams and chat generations, soma now writes a small `<soma_state>`
prelude into the trace bank and trains on it. the block includes current lr,
motor/io2 state, decimation, stride, coherence stats, rowclip, and model shape. this gives
the model a stable text description of its own state before it speaks.

interactive turns are separated internally with ascii `0x1e` record separator.
this byte is not printed in chat or dreams and should not appear in ordinary
corpora. it gives soma a private, learnable boundary between user prompts,
self-state records, and model speech.

the old model-controlled decimation path uses one active reserved motor byte:

- `0x11`: decimation control. higher relative motor salience means higher
  decimation, faster corpus movement, and fewer updates per corpus byte.
- `0x12`: reserved legacy motor byte. it remains masked so old motor traces do
  not become ordinary language targets, but it no longer controls decimation.

motor-byte rows are not treated as ordinary text targets, but motor logits
remain inside the full 256-way language softmax for normal text. this makes
control compete with language prediction instead of living in a separate loss
geometry. in model mode, soma reads the `0x11` logit as a z-score relative to
the non-motor byte field. neutral output is below the midpoint; the motor byte
must become salient relative to the field before it means "go faster." that
relative salience is mapped onto the configured decimation range. the actual
sampled stride chosen for that batch is then normalized through the band
geometry and injected back into the trace bank as the next motor-channel input.
the model therefore sees its realized control history, not just its raw
intention.

the configured range also defines the attention budget threshold. with
`decimation 12`, the threshold is band `6`. budget accounting is based on the
sampled stride actually used by the batch, converted back into band space:
sampled strides below threshold spend budget nonlinearly, while sampled strides
above it recharge budget with diminishing returns. a small differentiable
opportunity-cost term is added to every language batch when the model asks for
dense attention, so the whole network feels the resource cost while still being
judged primarily on language prediction. this cost is strongly
surprise-discounted: stable or falling loss makes dense attention expensive,
while even modest upward loss movement against the slow baseline makes dense
attention cheaper. the budget floor is intentionally soft and nonlinear: it
mostly stays out of the way until
attention is heavily depleted. the model still only chooses speed; the config
determines how expensive attention is. model mode
also treats decimation as a sampling precision setting. low decimation gives
tight, nearly stride-1 sampling. higher decimation broadens the sampled stride
distribution in stride space. the chosen decimation band defines the maximum
stride, while higher decimation lowers the sampled minimum toward `1`, so high
decimation becomes diffuse sampling across many possible strides rather than one
stable fast stride. if the active motor byte is emitted during generation, it is
hidden from visible text and applied as an internal decimation action.

the install data folder includes `soma_motor_seed.txt`, a small curriculum
containing real `0x11` bytes for bootstrapping this control habit.

online chat prompt learning uses the same strided/controller path as corpus
training, so long prompts behave like normal training text instead of one giant
uncontrolled batch.

the wallclock research mode has a simple rationale: soma does not decimate its perception. every
raw byte updates the trace bank. decimation only selects which trace-bank
states receive a gradient, trading update density against source traversal per
real machine second. on the measured cpu frontiers, a stable mean stride near
eight beat both dense updates and much sparser policies. wallclock starts there
and adds a tiny, slow log-stride probe; only a persistent correlation between
that probe and loss descent per second can move the centre. it is bounded and
rate-limited, so it explores the local frontier without abrupt sampling jumps.

in the gui, stopping a single-corpus training run mid-file updates `start byte`
to the last reported corpus position. a normal end-of-file finish resets it to
`0`. selecting a corpus from the data menu or editing the corpus field also
resets it to `0`; restarting the app leaves the saved value alone.

dreams still happen every configured number of batches, but the dream-trigger
batch now tries to stop at a newline inside its normal batch window before the
self-state prelude and generation begin. this gives dream samples cleaner
language boundaries without decoupling them from training progress.

the gui strips the older experimental knobs from fresh training:

- max change
- direct readout
- scale gate
- clock
- weight decay

## terminal use

run:

```sh
./soma
```

bare filenames resolve into local folders:

- corpora: `data/`
- checkpoints: `checkpoints/`
- stream scripts: `streams/`
- stream output corpora: `data/streams/`

absolute paths still work, which is what you want on rented gpu storage.

example:

```sh
./soma train meaning_curriculum_01.txt \
  --save v12_2.pt \
  --bands 20 --hidden 1536 --depth 3 \
  --batch 512 \
  --description "trained on meaning curriculum" \
  --auto-mode io2 --lr "auto 0.001" \
  --decimation 1 --max-stride 512 --grad-clip 1.0 \
  --row-norm auto --row-norm-mult 4.0 --row-norm-every 100 \
  --dream-every 100 \
  --dream-length "auto 300"
```

## mac app build

```sh
macos/build_app.sh
```

outputs land in `dist/`:

- `soma.app`
- `soma.dmg`
- `soma-v12.2-script-bundle.zip`
- `soma-script-bundle.zip`

## implementation files

- [soma_v12_2.py](/Users/jamesblight/spectral_v2/soma_v12.1/soma_v12_2.py)
- [soma_gui.py](/Users/jamesblight/spectral_v2/soma_v12.1/soma_gui.py)
- [soma_v12_spec.md](/Users/jamesblight/spectral_v2/soma_v12.1/soma_v12_spec.md)
