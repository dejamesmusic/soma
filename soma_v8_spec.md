 
░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░       
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                        soma                         ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓              algorithm specification                ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓      v8 · spectral online machine architecture      ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░

░░░░░░░░░░░░░░░░░░░░░░░░▒▓ definitions ▓▒░░░░░░░░░░░░░░░░░░░░░░░░

trace · a scalar value summarizing the recent history of a single
byte value at a single timescale. decays toward zero every 
timestep; increased when its byte is observed. trace E_{i,k} 
encodes how recently and how frequently byte i has appeared, as 
measured at timescale k.

band · one timescale index k in the range 0 to K-1. band 0 is 
fastest (shortest memory). band K-1 is slowest (longest memory). 
each band has decay rate α_k = 1/base^k.

base · a scalar r > 1 determining geometric spacing between 
adjacent bands. default is the golden ratio φ = (1+√5)/2 ≈ 1.618. 
alternatively computed from a desired maximum memory window W 
via r = W^(1/(K-1)).

bandpass feature · the difference between a trace at band k and 
the trace at band k+1 for the same byte channel, isolating 
temporal frequency content between those timescales. the slowest 
band's feature is its trace value directly.

feature vector · concatenation of all bandpass features across 
all 256 byte channels and all K bands. dimensionality 256 × K. 
the system's complete representation of the past at any moment.

confidence · a per-band scalar between 0 and 1 scaling each 
band's gradient contribution. determined by the relationship 
between the band's timescale and observation resolution.

decimation band · an integer d in [0, K-1] setting observation 
resolution during training. the system observes one byte every 
base^d bytes. band k confidence = min(1, base^(k-d)).

stride · bytes between observations. stride = base^d rounded to 
nearest integer. at decimation band 0, stride is 1.

budget normalization · after relu, hidden activations are 
rescaled so their l1 norm equals a fixed constant. total 
activation is conserved: increasing one unit decreases others.

weight normalization · after every update, each row of every 
weight matrix is rescaled to a fixed l2 norm. learning is 
directional only.

clipped update · each weight's change per step is clamped to 
max_change × |current value|. no weight changes by more than a 
fixed fraction of its magnitude per step.

░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓ soma ▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░

░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░       
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                     trace bank                      ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░

the trace bank is a matrix of shape (256, K) in 64-bit floating 
point on cpu. each row is one byte value. each column is one 
temporal band.

on observing byte b:

    all channels decay
    E_{i,k} ← (1 - α_k) × E_{i,k}

    observed channel increased
    E_{b,k} ← E_{b,k} + α_k

all 256 channels decay every step. only the channel matching the 
observed byte is increased. traces encode both recency and 
frequency at each timescale.

64-bit precision is required because slow-band decay rates 
approach 1.0 (e.g. 0.99999999917 at band 45 with base φ) and 
round to exactly 1.0 in 32-bit, freezing the trace.

░░░░░░░░░░░░░░░░░░░░▒▓ bandpass features ▓▒░░░░░░░░░░░░░░░░░░░░░░

computed from traces by adjacent differencing:

    B_{i,k}   = E_{i,k} - E_{i,k+1}     for k < K-1
    B_{i,K-1} = E_{i,K-1}               slowest band

each feature isolates activity in one byte channel at the 
frequency between two adjacent timescales. the full feature 
vector is concatenated to length 256 × K, cast from 64-bit to 
32-bit, and transferred to the compute device.

░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓ soma ▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░

░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░       
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                    forward path                     ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░

maps a feature vector to a probability distribution over the next 
byte. contains all learnable parameters. the trace bank has none.

 › with hidden layer (H > 0)

given feature vector x of length 256K:

1. z = U × x — U has shape (H, 256K)
2. h = relu(z)
3. h_norm = h × (budget / (sum(h) + ε)) — budget = H × 0.1
4. logits = W × h_norm — W has shape (256, H)
5. optional: logits = logits + Wd × x — Wd has shape (256, 256K)
6. probabilities = softmax(logits)

 › without hidden layer (H = 0)

logits = W × x — W has shape (256, 256K). 
probabilities = softmax(logits).

░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓ soma ▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░

░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░       
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                      training                       ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░

training processes a corpus sequentially from beginning to end.

 › stride 1

the corpus is divided into contiguous blocks. for each block:

1. trace bank processes the block, computing the feature vector 
at every byte position. trace state advances to reflect the full 
block.
2. forward path computes logits for all positions simultaneously. 
intermediate values are cached for gradient computation.
3. cross-entropy error is computed against the true bytes. the 
error signal is the softmax output with 1.0 subtracted at the 
true byte position.
4. weight updates are applied using this error signal and the 
cached values.

 › stride > 1

the system observes one byte every stride positions. between 
observations, the trace bank advances through skipped bytes using 
a closed-form computation. at each observation, the feature 
vector and target byte are recorded. when enough samples 
accumulate, forward path and weight updates process them as a 
batch.

░░░░░░░░░░░░░░░░░░░░░░▒▓ weight updates ▓▒░░░░░░░░░░░░░░░░░░░░░░░

all gradients are computed analytically from the forward path 
equations and cached intermediate values. no automatic 
differentiation is used.

 › W update (not band-indexed)

    grad_W = (errors^T × h_norm) / batch_size

applied as a clipped update, then weight decay, then row 
normalization.

 › U update (band-indexed)

the gradient passes backward through budget normalization 
(off-diagonal jacobian terms due to the quotient) and relu.

    grad_U = (grad_hidden^T × features) / batch_size

applied per-band: for each band k, the columns of U 
corresponding to that band are updated with the gradient scaled 
by band k's confidence. after all bands, weight decay and row 
normalization are applied to all of U.

 › Wd update (band-indexed, if present)

    grad_Wd = (errors^T × features) / batch_size

applied per-band with confidence scaling, same as U.

░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓ soma ▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░

░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░       
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                     decimation                      ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░

controlled by decimation_band d.

    stride       = round(base^d)
    confidence_k = min(1, base^(k-d))

bands k ≥ d have confidence 1.0. bands k < d have confidence 
base^(k-d) < 1.0.

every band updates every step. confidence is a scalar multiplier 
on the gradient at the moment of application.

░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓ soma ▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░

░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░       
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                     generation                      ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░

autoregressive, with no weight updates:

1. compute the feature vector from the current trace bank state.
2. pass through the forward path to get logits.
3. divide logits by temperature.
4. softmax to get probabilities.
5. sample one byte.
6. feed the sampled byte into the trace bank.
7. repeat.

the sampled byte enters the trace bank as memory. the trace bank 
reflects both prior training data and the model's own output. no 
learning occurs during generation because the model's own 
samples carry no external error signal.

░░░░░░░░░░░░░░░░░░░░░░▒▓ prompt ingestion ▓▒░░░░░░░░░░░░░░░░░░░░░

> without online learning · the trace bank advances through 
the prompt bytes using the closed-form computation. no features 
are computed and no weights are updated. trace state reflects 
the prompt content.

> with online learning · each prompt byte is processed 
sequentially. for each byte: the feature vector is computed, 
the forward path produces a prediction, error is computed, and 
weights are updated.

░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓ soma ▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░

░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░       
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                     checkpoints                     ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░

the complete state is: all weight matrices (U, W, Wd), the trace 
bank (256, K) matrix, normalization targets, hyperparameters, and 
bytes seen.

░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                        soma                         ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░