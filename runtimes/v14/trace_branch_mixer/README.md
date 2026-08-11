# trace branch mixer

this folder is the production v13 implementation of
[the architecture of the present](../engineering/the_architecture_of_the_present.md).

the implementation keeps four boundaries explicit:

1. `TraceState` is the exact, nonlearned float64 symbol-by-band present.
2. `associative_projected_scan` applies the current symbol geometry across a
   contiguous batch without storing full trace planes.
3. `TraceBranchMixer` organizes log-scale evidence and gives composition to a
   conventional serial residual trunk.
4. `TraceBranchTrainer` owns optimization, atomic state commits, generation,
   and checkpoint validation.

run the small reference profile:

```bash
python -m trace_branch_mixer.train data.txt \
  --profile small \
  --device auto \
  --save checkpoints/trace-mixer.pt
```

v13.1 adds the proven io2 control loop around this learner. a signed residual
is translated through the tied symbol dictionary and accumulated in an exact
`[bands + 1, symbol_dim]` error bank. io2 uses its energy and temporal
concentration to regulate effective learning rate and prediction density.
decimation selects expensive prediction sites only; `TraceState` still sees
every source token. replay, generated-token learning, and model-controlled
motor channels remain outside the production path.

v13.1.1 evaluates regularly decimated presents with an exact block-affine scan
and bounds source delivery. skipped projected states are not allocated, and a
high stride reduces model work rather than growing one batch without limit.

v13.2 adds checkpoint-compatible radial compression and truthful geometry
diagnostics. outward matrix-row growth is progressively attenuated and cannot
cross its capacity boundary. checkpoint rows already beyond that boundary are
not snapped down, but cannot inflate farther. the amber cockpit spectrum is
learned temporal allocation from the context weights, not another view of the
current activation. expensive diagnostics are throttled to visible report
emission.

the trainer also contains a disabled-by-default residual-clock pulse scheduler.
it is checkpointed for research, but dense io2 plasticity remains production
behavior until sparse updates match its learning per source byte.
