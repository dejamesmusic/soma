# soma v12.3.17 specification

## 1. identity

```text
runtime version  v12.3.17
species          soma_v12_3_token_residual
tokenizer        soma-8k-v1
vocabulary       8,192
token width      256
```

v12.3.17 is a token-native continual spectral learner. its temporal state is a
fixed geometric token trace bank and a translated residual trace bank.
gradients are confined to the current batch; there is no backpropagation
through time.

## 2. fixed tokenizer

all v12.3 models use the same checked-in sentencepiece model.

```text
model type             bpe
vocabulary             8,192
normalization          identity
byte fallback          yes
split by whitespace    yes
bos/eos/pad            absent
```

the tokenizer is lossless for arbitrary byte content via byte fallback. bpe
pieces do not cross whitespace boundaries, so the dictionary represents
lexical and morphological structure rather than cached phrases.

reserved protocol pieces are:

```text
<|soma_state|>  <|turn|>  <|user|>  <|soma|>  <|dream|>
```

the tokenizer model sha256 is a checkpoint compatibility field. a different
dictionary is a different model species even if every tensor shape matches.

## 3. token trace bank

let `v = 8192`, `k` be the number of bands, and `s_t` the token observed at
time `t`. band `j` has:

```text
alpha_j = base^(-j)
decay_j = 1 - alpha_j
```

the exact identity trace is:

```text
t_t[i,j] = decay_j * t_(t-1)[i,j] + alpha_j * 1[s_t = i]
```

prediction uses the pre-token state `t_(t-1)`. adjacent low-pass traces are
differenced to isolate temporal bands:

```text
b[:,j]   = t[:,j] - t[:,j+1]       j < k-1
b[:,k-1] = t[:,k-1]
```

the bank is non-learned, causal and bounded. it stores `v x k` float64 values
on cpu. for the default 16 bands this is about 1 mb.

band time is measured in tokens. source-domain time is reported separately;
the runtime maintains a moving bytes/token estimate for translating token
stride into approximate source-byte stride.

### current-dictionary projection

let `e` be the learned `v x 256` token dictionary. the retained token identity
is projected at the beginning of every batch:

```text
p[j,d] = sum_i t[i,j] * e[i,d]
```

then bandpassing is applied over `j` and the result is flattened to a
`256k`-wide present. because identity rather than old embeddings is retained,
changing `e` immediately reinterprets all compressed past visible in the bank.
this is the central v12.3 compatibility between a learned dictionary and
soma's non-bptt temporal membrane.

block processing uses the closed form of the recurrence. tests compare it to
literal sequential ticking, including gradients into tokens already retained
in the bank.

## 4. learned forward path

with hidden width `h` and depth `d >= 1`:

```text
x_0 = flatten(projected bandpass present)       length 256k
h_0 = budget_relu(input(x_0))                   length h

for l = 0 .. d-2:
    g_l     = sigmoid(residual_logit_l)
    h_(l+1) = budget_relu(h_l + g_l * u_l(h_l))

language = language_map(h_(d-1))                length 256
logits   = language @ e^T + token_bias          length 8192
probs    = softmax(logits)
```

all linear transforms are bias-free except the final token bias. the input and
output token dictionary is tied. residual gates initialize at `0.1`, so added
depth begins as a small correction to an already usable path.

budget relu is:

```text
budget_relu(z) = relu(z) * (0.1h / (sum(relu(z)) + epsilon))
```

it fixes total positive activation energy while leaving the network to choose
its distribution across coordinates.

### parameter count

```text
dictionary       8192 * 256
input            (256k) * h
residual depth   (d - 1) * h * h
language map     h * 256
token bias       8192
residual gates   d - 1
```

fresh models define `h = 256k`; hidden width is not an independent production
configuration axis. default `k=16, h=4096, d=4` is approximately 70.3m
parameters. the first learned map therefore preserves the dimensionality of
the projected trace-bank present before residual composition begins. checkpoint
loading retains a saved non-square width for compatibility.

## 5. objective and updates

the learned network minimizes next-token cross entropy:

```text
loss_t = -log p(s_t | pre-token trace state)
```

adamw is used with zero weight decay. global gradient norm is clipped at the
configured `gradclip`. no gradient crosses a batch boundary, but the exact
trace state does.

row magnitude is governed by a radial plasticity compressor. every configured
interval, each row in the dictionary, input, residual and language maps
compares its proposed norm with its norm at the previous interval:

```text
ceiling(w) = 4 * 0.1 * sqrt(fan_in)
r = previous_norm / ceiling
radial_gain = clamp(1 - r^4, 0, 1)
target_norm = previous_norm + radial_gain * max(0, proposed_norm - previous_norm)
```

inward norm changes pass unchanged. the proposed row direction also passes
unchanged, so a row at capacity remains free to rotate and reorganize. only
outward radial growth is progressively compressed, with the ceiling retained
as an emergency bound. the reported `compression` percentage is the fraction
of proposed outward norm growth rejected at the latest diagnostic interval.
the final token bias is not compressed.

## 6. decimation

decimation selects gradient observations, not trace observations. every token
advances the bank. at stride `q`, loss and backprop are evaluated at every
`q`th token while the corpus reader supplies approximately `batch * q` tokens.
the model therefore sees continuous time but spends fewer updates per source
byte.

error traces use `q` as the elapsed interval between sampled residuals. the
reported byte stride is `q * ema(source_bytes / tokens)`.

the hard stride cap is 256. the configured decimation value is a gain applied
after io2 chooses its raw spectral target:

```text
effective_band = clamp(decimation_gain * io2_band, 0, log_base(256))
stride = round(base ^ effective_band)
```

`1` preserves io2's native decision, `0` is dense training, values below `1`
attenuate decimation, and values above `1` amplify it toward the same safety
cap. because stride is exponential in band, gain has a strongly nonlinear
effect on corpus throughput while leaving io2's evidence and lr path intact.
the historical checkpoint field `decimation_range` is retained as an alias for
compatibility. old checkpoints storing `1` retain their controller scale;
values above `1` are intentionally reinterpreted as gain rather than literal
band caps.

training nats/byte under stride are estimated from sampled token loss and token
density. evaluation always scores every token.

## 7. io2 controller

the categorical residual is translated through the tied token dictionary:

```text
r_t = e[target_t] - sum_j p_t[j] * e[j]
    = e^T (one_hot(target_t) - p_t)
```

this signed 256-dimensional residual is normalized by mean dictionary norm and
retained in a geometric trace bank with the same bands. at decimated samples,
the recurrence uses `decay^stride`, preserving elapsed token-time. its
bandpassed vector norm gives:

- total residual energy
- spectral concentration
- dominant residual band

io2 converts energy, concentration and loss relative to token chance
`log(8192)` into a bounded plasticity ratio. that ratio:

- scales automatic learning rate from its configured base
- moves the decimation target
- advances depth saturation

depth saturation gradually protects the shared dictionary and early transforms
as competence rises. the final language readout remains fully plastic. groups
are:

```text
dictionary + input -> residual blocks -> language map + token bias
```

io2 is the production controller and is not a user-facing mode selection.
automatic lr uses a default base of `0.002`; decimation gain defaults to `1`.
gradient clipping remains a static safety boundary rather than a live
controller statistic.

the absolute throughput-relaxation term is measured in source-domain bits:

```text
relaxation = clamp(1 - bits_per_byte, 0, 1)
```

there is therefore no low-loss mastery bonus at or above one bit per source
byte. below one bpb, the bonus rises linearly while residual spectrum and io2
plasticity continue to determine the final decimation target.

## 8. metrics

the production headline metrics are in the source domain:

```text
nats/byte = total deterministic-token nll / original utf-8 source bytes
bpb       = nats/byte / ln(2)
```

this is a code-length comparison against byte models because tokenization is
fixed and deterministic. the gui reports top-1 and top-5 next-token accuracy.
throughput is original source bytes per wall-clock second. diagnostic reports
also include the input trace spectrum, first-layer weight allocation by band,
residual spectrum, active trace-memory tokens and likely next tokens.

## 9. generation protocol

chat uses explicit user, soma and turn pieces. dreams begin with the single
native `<|dream|>` piece. every generation closes with `<|turn|>`, whether the
model emits it itself or reaches the configured cap. these compact protocol
pieces advance the trace bank but are not direct prediction targets.

live online learning follows the source of each token:

```text
user language       perceive + learn
self-state/protocol perceive only
soma generation     perceive only
```

the user's words are external observations. a generated token is soma's own
sample; scoring it as truth has zero expected learning signal and reinforces
sampling noise on individual turns. generation therefore changes the retained
present but never updates weights.

recorded dialogue is external data on both sides, so user and soma language are
both scored. synthetic state and role markers are masked. the turn marker that
ends a recorded soma reply has weight 0.1, enough to teach yielding without
making immediate silence the easiest part of the corpus. masked tokens are also
excluded from io2, bpb and accuracy accounting.

free generation masks the unknown token and internal protocol labels. the turn
piece remains available as the learned yield-floor action. chat and dream
generation lengths are both generated-token caps. the learned turn piece can
end either form of generation early. dream cadence remains measured in training
batches, so input and feedback are in the same language: the runtime records
external tokens consumed since the last dream, generated dream tokens, and
their ratio. the gui consumes decoded increments and can stop between tokens.

dream generation remains perceive-only. its effect is therefore not direct
gradient training on sampled text: it changes the retained present against
which later external observations are learned. the one-token dream boundary
keeps the measured feedback ratio focused on generated language rather than
repeated telemetry syntax.

generation projects the retained identity bank once, advances that projected
state recursively while weights are fixed, then commits the generated token
sequence to the exact identity bank in one closed-form block. translated error
traces decay across self-generation because no external residual is observed.

## 10. streaming and memory

corpora are read incrementally and tokenized at whitespace-aligned segments.
the reader holds bounded token read-ahead and emits fixed-size token batches.
the final batch alone may be short. fixed graph shapes prevent an unbounded
mps/cuda compilation cache during long runs.

the app performs training in a child process. stopping training destroys that
process, releasing model, optimizer and accelerator caches even if a framework
allocator retains internal state.

## 11. checkpoint contract

checkpoints are atomic torch files with an adjacent `.meta.json`. required
compatibility fields include:

```text
species, runtime_version, checkpoint_id, revision_id,
tokenizer_name, tokenizer_sha256,
vocab_size, token_dim, n_bands, hidden_dim, depth, base
```

state includes network and optimizer tensors, token and error traces,
controller state, source bytes and tokens seen, description and history.
`checkpoint_id` remains stable for the life of a model; `revision_id` changes
on each successful save.

the sidecar has two ownership domains. runtime fields are authoritative
machine metadata and are refreshed on save. the `profile` object is
human-owned and preserved:

```json
{
  "profile": {
    "name": "my soma",
    "tagline": "local continual language model",
    "art": "optional monospace art",
    "about": "editable checkpoint notes"
  }
}
```

the app exposes this profile in the models and training views. it may be
edited independently of the tensor checkpoint and uploaded beside the `.pt`.

corpus resume positions are app-local and keyed by checkpoint filename plus
corpus filename. a cursor advances only after the matching checkpoint save
succeeds, and token-reader alignment is included in its absolute byte
position.

v12.3 checkpoints load directly. scalar error traces from the initial v12.3
release are reset once because they do not share the translated residual's
geometry; learned weights and exact token traces are preserved.

v12.2 byte checkpoints do not load into v12.3. they remain valid in the v12.2
runtime; conversion would not preserve the learned input or output alphabet.

## 12. validation

`tests/test_v123_core.py` verifies:

1. tokenizer roundtrip including protocol, control bytes and unicode
2. fixed-block corpus reading without dropped source bytes
3. block trace recurrence against literal sequential time
4. gradient reinterpretation of retained token identity
5. cached generation-state recurrence against exact identity updates
6. gap-aware residual recurrence against repeated elapsed time
7. incremental sentencepiece decoding including byte fallback
8. exact save/load logits and metadata
9. state perception without direct state-token weight updates
10. protocol-aware corpus masking across arbitrary block boundaries

`experiments/token_byte_compare.py` provides the parameter-matched source-byte
comparison against v12.2.
