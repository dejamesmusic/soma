# the geometry of the present

## trace representations, symbolic dynamics, and predictive coherence

status: theoretical foundation for the next soma architecture

## abstract

soma models an indefinitely continuing symbolic process. its central
representation is a bank of exponentially decayed symbol traces. this document
defines the resulting mathematical object, states what the representation
preserves and discards, and identifies the function a learned model must
approximate.

the trace update is a symbol-labelled affine contraction. the set of states
reachable from infinite histories is therefore the attractor of an iterated
function system. every history is both a path through that attractor and a
superposition of symbol events over fixed temporal signatures. from any
reachable state, the next observation selects one of a finite set of legal
successor maps. prediction is consequently the estimation of the next branch
of the trace dynamics.

the stream and its trace dynamics define a markov operator over observables of
the reachable state. its coherent modes may have complex eigenvalues. their
magnitudes determine persistence, their arguments determine temporal phase,
and their coupling to the next branch determines predictive relevance. the
trace bank supplies fixed real decay coordinates with which a learner may
resolve these modes; it does not guarantee that they are resolved.

geometric spacing gives constant relative resolution over decay time. phi is
selected uniquely if this spacing is additionally required to satisfy exact
two-neighbour multiplicative closure of decay envelopes. this is a conditional
theorem, not a claim that phi is universally optimal for language.

the resulting thesis is:

> a trace bank embeds symbolic history as an address on a self-affine
> reachable set. the next event selects a legal outgoing branch. the learned
> model approximates the conditional branch kernel by recovering predictively
> coherent interactions among fixed logarithmic decay coordinates.

## 1. scope

this document separates four objects:

1. the world: an indefinitely continuing symbolic stochastic process;
2. the representation: a finite bank of causal exponential traces;
3. the ideal predictor: the conditional distribution of the next branch given
   the represented present;
4. the learned model: a finite approximation to that conditional branch
   kernel.

the distinction is essential. the trace bank may omit predictive information.
the learned model may fail to extract information that the trace bank retains.
irreducible uncertainty may remain even when both are ideal.

the document makes no claim that a finite trace bank is sufficient for every
symbolic process, that language possesses a finite coherence time, or that one
particular neural implementation is minimal. it establishes the geometry that
any such implementation receives and the constraints it should not violate
without reason.

## 2. the observed process

let the alphabet contain `v` symbols:

```text
a = {1, ..., v}.
```

the observation at source time `t` is represented by a one-hot vector:

```text
x_t = e_{y_t} in {e_1, ..., e_v}.
```

the complete history available before observing `y_t` is:

```text
h_t = (..., y_{t-2}, y_{t-1}).
```

the unrestricted bayes predictor is:

```text
p_h(y | h_t) = probability(y_t = y | h_t).
```

the process need not be finite-order markov, stationary, ergodic, or generated
by a finite latent state. those properties may be introduced later as explicit
assumptions, but they are not built into the trace representation.

## 3. the trace representation

choose `k` retentions:

```text
0 <= rho_k < 1
```

and injections:

```text
a_k = 1 - rho_k.
```

collect the retentions into a diagonal matrix:

```text
d = diag(rho_0, ..., rho_(k-1))
```

and the injections into a vector:

```text
a = (a_0, ..., a_(k-1)).
```

the trace state before observing `y_t` is a matrix:

```text
q_t in real^(v x k).
```

observing symbol `y_t` applies:

```text
q_(t+1) = q_t d + e_(y_t) a^transpose.
```

column `k` therefore obeys:

```text
q_(t+1)[:, k]
    = rho_k q_t[:, k] + a_k e_(y_t).
```

if a column initially sums to one, it remains a probability distribution over
symbols. if initialized at zero, its mass after `t` observations is:

```text
1 - rho_k^t.
```

zero initialization is consequently a sub-normalized boundary condition.
initialization from a prior distribution gives normalized columns from the
first observation.

### 3.1 exact event-orbit expansion

unrolling the recurrence gives:

```text
q_t
  = q_0 d^t
  + sum_(n=1 to t) e_(y_(t-n)) a^transpose d^(n-1).
```

for an infinite past, or after the initial condition has decayed:

```text
q_t
  = sum_(n>=1) e_(y_(t-n)) tensor r(n),
```

where the temporal signature of an event at age `n` is:

```text
r(n)
  = (a_0 rho_0^(n-1), ...,
     a_(k-1) rho_(k-1)^(n-1)).
```

this identity is exact.

every historical event contributes the outer product of:

- one symbolic basis vector;
- one point on a universal temporal orbit.

the trace state is their linear superposition.

### 3.2 sampled generating functions

for symbol `s`, define the causal history-generating function:

```text
f_(s,t)(z)
  = sum_(n>=1) indicator(y_(t-n) = s) z^(n-1).
```

then:

```text
q_t[s, k] = a_k f_(s,t)(rho_k).
```

the trace bank is therefore a finite sampling of one positive-coefficient
generating function per symbol. the symbol rows are not independent: at every
age exactly one symbol coefficient is active.

writing:

```text
rho_k = exp(-beta_k)
```

relates the representation to a sampled discrete laplace transform. this is
related to, but not identical with, scale-invariant laplace memory models that
also provide translation and approximate inverse machinery. see
[shankar, singh, and howard](https://arxiv.org/abs/1503.03322).

## 4. the reachable trace set

for each symbol `y`, define the affine map:

```text
f_y(q) = q d + e_y a^transpose.
```

for any two states:

```text
f_y(q) - f_y(q')
  = (q - q') d.
```

under a compatible matrix norm:

```text
norm(f_y(q) - f_y(q'))
  <= rho_max norm(q - q'),
```

where:

```text
rho_max = max_k rho_k < 1.
```

every map is therefore contractive.

standard iterated-function-system theory implies a unique nonempty compact
reachable set `r` satisfying:

```text
r = union_(y in a) f_y(r).
```

the foundational result is due to
[hutchinson](https://iumj.org/article/2972/).

the correct general term for soma's represented world is:

> the reachable trace attractor.

it should not yet be called a smooth manifold. depending on retentions,
alphabet size, overlap, initialization, and precision, the reachable set may
be self-affine, overlapping, locally full-dimensional, or singular.

### 4.1 histories as addresses

an infinite symbolic past provides an address for a point in `r`:

```text
q_t
  = ... f_(y_(t-1)) composed_with
        f_(y_(t-2)) composed_with ... (q_origin).
```

two histories may have:

- distinct trace states;
- trace states separated by less than available numerical precision;
- exactly the same trace state when affine addresses overlap.

the trace map is therefore a coding map from symbolic histories into the
reachable attractor. the degree to which that coding is injective is a
property to be proved for a selected lattice and precision, not assumed from
dimension counting.

### 4.2 finite precision is part of the representation

an exact real number may encode arbitrarily detailed history while remaining
impossible to recover stably. the operational state is not an abstract real
matrix but:

```text
q_t_tilde = c_epsilon(q_t),
```

where `c_epsilon` represents floating-point precision, quantization, state
noise, clipping, and any other numerical channel.

histories whose states become indistinguishable after `c_epsilon` belong to
the same operational equivalence class. this is the representation soma can
actually use.

## 5. legal successor dynamics

from state `q`, observing symbol `y` produces exactly:

```text
f_y(q) = q d + e_y a^transpose.
```

the legal actual successors are the `v` vertices:

```text
{f_1(q), ..., f_v(q)}.
```

if the next symbol has distribution:

```text
p in simplex(v),
```

the expected successor is:

```text
f_p(q)
  = sum_y p_y f_y(q)
  = q d + p a^transpose.
```

the convex hull of legal successors is an affine simplex of dimension at most
`v - 1`. this is the successor fibre over `q`.

the same distribution `p` must explain expected motion at every band:

```text
f_p(q)[:, k] - rho_k q[:, k] = a_k p.
```

this is an exact consistency law.

it does not reduce the dimension of an ordinary `v`-logit next-symbol output:
every next-symbol distribution already defines one legal expected successor.
its value is conceptual and structural. any model that predicts a full future
trace plane should be projected onto this fibre, and any transition energy
should distinguish the source-transition pair rather than an unconstrained
destination alone.

## 6. what must be predicted

the trace representation induces the ideal branch kernel:

```text
p_star(y | q)
  = probability(y_t = y | q_t = q).
```

the next-symbol problem is:

> given the current address on the reachable trace attractor, estimate which
> symbol-labelled contraction will occur next.

the ideal model is therefore a conditional probability kernel from the
reachable set to the alphabet:

```text
p_star : r -> simplex(v).
```

a learned model:

```text
p_theta : r -> simplex(v)
```

approximates this kernel.

an energy representation may write:

```text
p_theta(y | q)
  proportional_to
  exp(-u_theta(q, f_y(q))).
```

including both source and destination prevents an unproved path-independence
assumption. with unrestricted `u_theta`, this remains a reparameterization of
ordinary classification. any efficiency must come from justified restrictions
on the function family, not from the successor notation itself.

## 7. exact predictive-loss decomposition

let `h` denote full history, `q_tilde` the operational trace state, and `y`
the next symbol. if:

```text
y -> h -> q_tilde
```

forms the expected information channel, then:

```text
expected[-log p_theta(y | q_tilde)]
  = entropy(y | h)
  + information(y ; h | q_tilde)
  + expected_q_tilde[
      kl(
        probability(y | q_tilde)
        ||
        p_theta(y | q_tilde)
      )
    ].
```

the terms are:

1. irreducible uncertainty in the stream;
2. predictive information unavailable from the operational trace state;
3. available predictive information not extracted by the learned model.

this is exact. it separates representation failure from approximation and
optimization failure.

the terms are not directly observable without estimating flexible conditional
distributions. held-out probes can compare representations and bound useful
differences, but they do not provide effortless access to the exact bayes
decomposition.

for a bounded model family `g`, define operational predictive deficiency:

```text
deficiency_(epsilon,g)
  = inf_(p in g)
      expected[-log p(y | q_tilde)]
    - entropy(y | h).
```

this quantity includes finite precision and bounded readout complexity. it is
the relevant notion of practical sufficiency.

## 8. geometry of the trace state

### 8.1 ambient geometry

when normalized, each trace column belongs to the symbol simplex:

```text
q[:, k] in simplex(v).
```

the ambient state space is therefore:

```text
simplex(v)^k.
```

the reachable attractor is a subset of this product. the ambient space is not
the same object as the reachable set.

### 8.2 reference and tangent coordinates

the normalized simplex does not have the zero matrix as an intrinsic origin.
choose a reference distribution `pi`, such as the stationary symbol marginal,
and define:

```text
q_reference = pi 1^transpose
z = q - q_reference.
```

every normalized column of `z` has zero total mass:

```text
sum_s z[s, k] = 0.
```

`z` lies in the product of simplex tangent spaces. this is the correct linear
space for signed deviations, band differences, and mass-conserving learned
corrections.

zero initialization is a computational boundary condition, not this
information-geometric reference.

### 8.3 possible metrics

the representation does not uniquely dictate one metric.

a weighted native metric is:

```text
distance_linear(q, q')^2
  = sum_k w_k norm(q[:, k] - q'[:, k])_2^2.
```

for geometric spacing, equal `w_k` corresponds to equal quadrature weight per
unit of log decay time.

the local fisher metric on the product simplex is:

```text
g_q(u, v)
  = sum_k w_k sum_s u[s,k] v[s,k] / q[s,k].
```

it measures distinguishability of nearby probability fields. its mathematical
naturalness does not imply that fisher-whitened values should be fed directly
to a learner. representation geometry and effective optimization coordinates
are separate questions.

the trace feature map also induces an event kernel:

```text
kappa((s,n), (s',m))
  = indicator(s = s')
    sum_k w_k r_k(n) r_k(m).
```

this kernel measures similarity between historical events after trace
compression. distance between histories under the corresponding feature norm
is exactly distance between their trace states under the weighted linear
metric.

### 8.4 temporal scale-space

define the exponential rate and e-fold lifetime:

```text
beta_k = -log(rho_k)
tau_k = 1 / beta_k.
```

geometric rates:

```text
beta_(k+1) = r beta_k
```

produce geometric lifetimes:

```text
tau_(k+1) = tau_k / r.
```

the band index is therefore a uniform coordinate in log lifetime. finite
endpoints and a separate instantaneous channel break exact dilation symmetry,
so a practical finite bank is approximately rather than perfectly
scale-equivariant.

## 9. the conditional phi theorem

### 9.1 axioms

assume:

1. decay rates have a constant positive ratio:

```text
beta_(k+1) = r beta_k, with 0 < r < 1;
```

2. the decay envelope at one scale is the pointwise product of the next two
   slower envelopes:

```text
exp(-beta_k n)
  = exp(-beta_(k+1) n)
    exp(-beta_(k+2) n)
```

for every nonnegative `n`.

### 9.2 derivation

the closure condition implies:

```text
beta_k = beta_(k+1) + beta_(k+2).
```

substituting the geometric ratio gives:

```text
1 = r + r^2.
```

the unique positive solution below one is:

```text
r = (sqrt(5) - 1) / 2 = 1 / phi.
```

equivalently:

```text
beta_(k+1) = beta_k / phi
tau_(k+1) = phi tau_k
rho_k = rho_(k+1) rho_(k+2).
```

phi is therefore the unique constant-ratio decay lattice satisfying exact
two-neighbour multiplicative closure.

### 9.3 precise conclusion

this theorem does not prove that:

- language is optimally represented by phi;
- two-neighbour closure is required by nature;
- phi minimizes finite-bank approximation error;
- phi gives the best hardware efficiency.

it proves:

> if constant relative scale resolution and exact local multiplicative
> closure are selected as axioms, phi is forced.

that is a defensible mathematical reason to treat phi as a principled prior.
its empirical optimality remains open.

### 9.4 exact and approximate implementations

the current-style choice:

```text
a_k = phi^(-k)
rho_k = 1 - a_k
```

does not produce exact geometric rates because:

```text
beta_k = -log(1 - phi^(-k)).
```

for slow bands:

```text
beta_k approximately_equals phi^(-k),
```

so closure is asymptotic.

the exact rate construction is:

```text
beta_k = beta_0 phi^(-k)
rho_k = exp(-beta_k)
a_k = 1 - exp(-beta_k).
```

an instantaneous present should be represented separately, because no finite
positive lifetime is identical to direct observation.

## 10. coherence time

trace lifetime and coherence time are different.

trace lifetime:

> how long the measuring coordinate retains an event.

coherence time:

> how long a relationship in the observed process remains predictively
> aligned.

a slow trace may retain information that has ceased to be useful. a fast
trace may participate in a rule that remains stable for the lifetime of the
language.

### 10.1 the branch-transition operator

the true branch kernel and legal successor maps define an operator on
observables `f` over the reachable set:

```text
(p f)(q)
  = sum_y p_star(y | q) f(f_y(q)).
```

this is a markov or koopman-type expectation operator: it propagates
observables forward while the dual transfer operator propagates
distributions. operator formulations of dynamics descend from
[koopman](https://terpconnect.umd.edu/~cjarzyns/CHEM-CHPH-PHYS_703_Spr_20/resources/Koopman-1931.pdf).

### 10.2 coherent modes

if an observable `psi` satisfies:

```text
p psi = lambda psi,
```

then:

```text
expected[
  psi(q_(t+n))
  |
  q_t
]
  = lambda^n psi(q_t).
```

write:

```text
lambda = magnitude exp(i omega).
```

for `0 < magnitude < 1`:

```text
coherence_time(lambda)
  = -1 / log(magnitude)
```

and, when `omega` is nonzero:

```text
period(lambda)
  = 2 pi / abs(omega).
```

the magnitude gives e-fold persistence. the argument gives temporal phase.
complex-conjugate eigenmodes represent damped oscillatory structure.

the complete operator may be non-normal or possess continuous spectrum, so
not every behavior decomposes into a finite set of clean eigenmodes. the
eigenmode picture is exact when suitable eigenfunctions exist and otherwise
serves as one component of a broader spectral analysis.

### 10.3 correlation definition

under an invariant stream measure `mu`, the lagged coherence of centered
observables `u` and `v` is:

```text
c_(u,v)(n)
  = inner_product_mu(u, p^n v).
```

integrated or squared correlation may define an observable-specific coherence
time. no single scalar coherence time characterizes all of language.

different structures may remain coherent over radically different intervals:

- local orthography;
- morphology;
- syntax;
- conversational role;
- topic;
- narrative state;
- stable facts and conventions.

### 10.4 predictive relevance

persistence is not sufficient. a mode matters to the learner only if it also
changes the next-branch distribution.

the useful spectrum is therefore not every long-lived mode of the stream, but
the subset coupled to:

```text
p_star(y | q).
```

a stable but next-symbol-independent feature consumes representational energy
without reducing predictive loss. a short-lived feature may be highly
valuable.

### 10.5 relationship to the trace lattice

the trace retentions `rho_k` are fixed real poles selected before observing
the process. the coherent modes `lambda_j` belong to the process and its
branch operator.

the trace bank does not contain the process spectrum by declaration. it
provides a logarithmic real basis from which a learned system may approximate
predictive modes.

this distinction explains why scheduling parameter updates directly from
trace lifetimes is not generally justified:

```text
trace lifetime != coherence time of a learned interaction.
```

an interaction involving a slow band may change quickly. an interaction among
fast coordinates may describe a rule that changes slowly. update timing
belongs to the statistic being estimated, not automatically to the lifetime
of either input coordinate.

## 11. the smith-chart analogy

a smith chart maps normalized complex impedance to a bounded reflection
coefficient:

```text
gamma = (z - z_0) / (z + z_0).
```

its radius and angle expose mismatch magnitude and phase. the original
construction was introduced by
[phillip h. smith](https://www.worldradiohistory.com/Archive-Electronics/30s/Electronics-1939-01.pdf).

the raw trace field is real and nonnegative. it is not itself a smith chart,
and no physical impedance has yet been defined.

two precise smith-like structures do appear:

1. every exponential trace has complex frequency response:

```text
h_k(exp(i omega))
  = a_k / (1 - rho_k exp(-i omega));
```

2. the spectrum of the branch-transition operator lies within an appropriate
   bounded spectral region, with complex modes carrying persistence and phase.

the second is closest to the intended intuition:

- radial position corresponds to coherence;
- angular position corresponds to oscillation or phase;
- the trace lattice supplies radial resolution;
- learned symbol interactions supply nontrivial angular structure.

this remains an analogy until a reference impedance and reflection map are
derived from the predictive dynamics. the operator spectrum already provides
the needed magnitude-phase language without pretending that trace
probabilities are electrical impedances.

## 12. what the learned model represents

the trace state carries transient evidence. learned parameters carry stable
regularities of the branch kernel.

weights are not additional samples of history. they approximate relationships
among trace coordinates that remain useful across many histories.

one defensible interpretation is:

> a learned interaction records evidence that a set of trace coordinates
> participates in a common predictively coherent mode.

this interpretation is conditional rather than definitional. an arbitrary
weight matrix need not align with operator eigenfunctions. an architecture
must make such alignment accessible without requiring it by assertion.

### 12.1 intrinsic and computational graphs

the intrinsic transition graph is defined by the representation:

- a vertex is a reachable trace state;
- each symbol labels one outgoing edge;
- edge `y` maps `q` to `f_y(q)`;
- a history is a path;
- the branch kernel assigns probabilities to outgoing edges.

the computational graph is chosen by engineering:

- vertices may be symbol-timescale activations;
- edges may mix symbols, scales, or both;
- depth may compose interactions within the represented present.

the intrinsic graph is exact. the computational graph is a hypothesis about
how to approximate its branch probabilities efficiently.

### 12.2 path superposition

in a linearized layered model, the contribution from an input coordinate to
an output is a sum over computational paths:

```text
contribution(path)
  = input_activation
    product_(edge in path) weight_edge.
```

nonlinear gates make path contributions depend on the current state.

this gives precise content to the idea that symbol interactions possess a
superposition of possible computational paths. it is a property of the chosen
function approximator, not an automatic theorem of the trace representation.

## 13. symmetries and obligations inherited by engineering

the theory does not yet select one architecture. it does establish obligations
and candidate symmetries.

### exact obligations

1. causality: prediction uses only `q_t`, formed before `y_t`;
2. probability: outputs define a distribution over legal next symbols;
3. successor consistency: any predicted trace motion corresponds to one
   shared next-symbol distribution across bands;
4. precision disclosure: claimed memory is bounded by actual numerical
   representation;
5. objective integrity: ordinary next-symbol likelihood remains the scoring
   rule unless a different objective is explicitly justified.

### principled candidate symmetries

1. symbol relabelling: without external semantics, arbitrary symbol indices
   have no natural order;
2. log-scale locality: neighbouring bands differ by a fixed relative
   lifetime;
3. approximate band translation: away from finite boundaries, shifting band
   index approximates temporal dilation;
4. mass conservation: learned movements in normalized symbol planes should
   remain in simplex tangent spaces before projection;
5. transparent initialization: an untrained learned path should not destroy
   the known trace state.

these symmetries can constrain an architecture, but their usefulness must be
distinguished from mathematical necessity. learned magnitude cannot be
discarded merely because directional normalization is aesthetically clean.

## 14. what is proved and what remains hypothesis

### exact results

- the trace recurrence and event-orbit expansion;
- the sampled generating-function identity;
- contraction of every finite-band symbol update;
- existence of a unique compact reachable attractor;
- symbol histories as addresses on that attractor;
- the legal successor vertices and successor fibre;
- the predictive-loss decomposition;
- the conditional phi closure theorem;
- eigenmode persistence and phase when branch-operator eigenfunctions exist.

### open mathematical questions

- topology, dimension, overlap, and separation of the reachable attractor for
  practical lattices;
- injectivity and conditioning of the history coding map;
- finite-precision predictive sufficiency;
- approximation error from a finite logarithmic rate lattice;
- regularity of the branch kernel on the reachable set;
- spectral structure of realistic symbolic branch operators;
- conditions under which trace coordinates efficiently resolve their
  predictive coherent modes;
- whether phi spacing minimizes any relevant approximation bound.

### architectural hypotheses

- scale-local parameter sharing can approximate the branch kernel more
  efficiently than anonymous dense mixing;
- a computational graph over symbol-timescale coordinates can recover useful
  coherent modes;
- transparent, mass-aware interaction paths improve continual stability;
- long coherence modes can be learned without destroying fast predictive
  structure;
- the resulting model reaches lower practical deficiency per parameter and
  per unit of compute than less structured alternatives.

none of these follows solely from the exact trace geometry.

## 15. the theoretical model of soma

the complete theoretical picture is:

```text
indefinite symbolic history
        |
        v
symbol-labelled affine contractions
        |
        v
reachable self-affine trace attractor
        |
        v
operational finite-precision state
        |
        v
conditional branch kernel
        |
        v
predictively coherent operator modes
        |
        v
finite learned approximation
        |
        v
distribution over the next legal branch
```

the trace bank compresses the past into a present address.

phi supplies a conditional principle for arranging temporal contractions.

the branch operator describes the coherent dynamics of the represented world.

the learned model approximates the distribution over legal successor maps.

the model succeeds when its operational representation retains the predictive
coherence of the stream and its finite parameterization extracts that
coherence before it becomes numerically or statistically inaccessible.

## 16. central thesis

> soma is a continual predictor over a self-affine trace representation of
> symbolic history. each observation advances the state through one
> symbol-labelled affine contraction. prediction is inference over the next
> legal contraction. the trace lattice supplies fixed logarithmic decay
> coordinates; the stream supplies coherent modes; learned interactions
> approximate the branch kernel that couples those modes to future symbols.
> phi is uniquely selected when geometric rate spacing is required to obey
> local two-neighbour multiplicative closure. the quality of soma is therefore
> determined separately by operational trace sufficiency and by the ability
> of its learned model to recover predictive coherence from that
> representation.

## 17. consequences for the next documents

the engineering document must answer:

1. how the exact contraction lattice is represented numerically;
2. how the instantaneous channel and finite endpoints are handled;
3. how the branch kernel is parameterized;
4. how symbol and scale interactions are constrained and shared;
5. how probability mass and transparent initialization are preserved;
6. how finite precision limits the slowest useful band;
7. how the model exposes predictive coherence without assuming a finite clean
   eigendecomposition;
8. how training, batching, and generation preserve the same causal state
   transition;
9. how compute scales with vocabulary, bands, interaction rank, and
   compositional depth;
10. which claims remain falsifiable implementation hypotheses.

the code document must then implement those engineering decisions directly,
without adding mechanisms that have no stated theoretical role.

## references

- john e. hutchinson, [fractals and self
  similarity](https://iumj.org/article/2972/), 1981.
- b. o. koopman, [hamiltonian systems and transformations in hilbert
  space](https://terpconnect.umd.edu/~cjarzyns/CHEM-CHPH-PHYS_703_Spr_20/resources/Koopman-1931.pdf),
  1931.
- karthik h. shankar, inder singh, and marc w. howard,
  [neural mechanism to simulate a scale-invariant future
  timeline](https://arxiv.org/abs/1503.03322), 2015.
- lyudmila grigoryeva and juan-pablo ortega,
  [universal discrete-time reservoir computers with stochastic inputs and
  linear readouts using non-homogeneous state-affine
  systems](https://arxiv.org/abs/1712.00754), 2017.
- phillip h. smith, [transmission line
  calculator](https://www.worldradiohistory.com/Archive-Electronics/30s/Electronics-1939-01.pdf),
  1939.
