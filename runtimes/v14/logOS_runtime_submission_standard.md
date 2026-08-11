# logos runtime submission standard

## status

proposal for `logos.runtime.v1`.

this document defines the minimum contract by which any person or organization
may publish an intelligent architecture and an implementation of it on logos.
it standardizes identity, integrity, interfaces, permissions, and evidence. it
does not standardize tensors, learning rules, topology, tokenization, modality,
or a definition of intelligence.

## the promise

logos makes four distinct promises:

1. it can preserve unfamiliar architecture and runtime artifacts without
   executing them;
2. it can identify every release and package by exact bytes and provenance;
3. it can state compatibility only when a named test observed it;
4. it can keep old runtimes available without treating them as obsolete minds.

submission is open. execution and recommendation are graduated trust states.

## submission unit

a complete submission may contain:

```text
runtime.json                 required
specification.md             required
license                      required
source snapshot or commit    strongly preferred
build recipe                 strongly preferred
dependency lock / sbom       required for executable review
platform package(s)          optional at initial publication
checkpoint fixture(s)        required for compatibility verification
protocol fixture(s)          required for interface verification
```

a specification without executable packages can still be published. a package
without a specification, identity, license, and declared permissions cannot be
described or executed.

## manifest

`runtime.json` uses `schema: logos.runtime.v1` and contains:

- stable `runtime_id` and immutable `release_id`;
- publisher identity and signing key where available;
- version, source repository, immutable source commit, and license;
- species ids and permanent specifications;
- one record per platform package, including media type, byte length, digest,
  signature, and minimum operating requirements;
- checkpoint format ids and versions;
- declared capabilities;
- protocols and modalities;
- entrypoint and adapter contract;
- cpu, memory, disk, accelerator, network, filesystem, and subprocess needs;
- tokenizer, vocabulary, schema, or other inseparable assets;
- migration tools, if any, as separate capabilities.

logos recomputes all artifact lengths and hashes. a release is immutable after
publication. corrections create a new release and may supersede, but never
rewrite, the old record.

## adapter contract

an executable runtime exposes a small host adapter rather than requiring logos
to understand its architecture. the first protocol should support json request
and response envelopes for declared operations:

```text
inspect(checkpoint) -> identity, format, interfaces, resource estimate
infer(checkpoint, fixture) -> bounded output
train(checkpoint, fixture, budget) -> candidate revision
resume(candidate, fixture, budget) -> candidate revision
stop(run) -> acknowledgement and bounded exit
```

only declared operations are invoked. paths are capability-scoped. stdout,
stderr, exit status, resource use, and produced artifacts are captured. the
adapter is not allowed to reinterpret a failed operation as a passed test.

## trust states

### preserved

bytes, manifest, publisher claim, and digest are recorded. no code runs.

### described

the common contract, specification, license, interfaces, privileges, and
formats are complete. claims remain publisher claims.

### reproducible

the submitted package was reproduced from declared source and recipe, or a
transparent exception explains why it was not.

### reviewed

static inspection, dependency policy, signing, and permission review pass.

### executable

the exact package digest starts and exits inside the logos isolation boundary
without undeclared access.

### compatible

the exact package digest passes a named suite against named checkpoint and
protocol fixtures.

### recommended

maintenance, documentation, evidence, user experience, and community history
justify editorial prominence. recommendation is never implied by execution.

## isolation requirements

verification workers must default to:

- no network;
- no user secrets or host home directory;
- read-only runtime and fixture mounts;
- one writable disposable workspace;
- bounded cpu, ram, accelerator memory, disk, subprocesses, output, and time;
- termination from outside the runtime;
- complete artifact and resource accounting;
- destruction after the verification result is committed.

platforms that cannot provide this boundary may preserve and distribute a
runtime but cannot grant automated executable status. community attestations
must remain visibly distinct from logos verification.

## compatibility

compatibility is a relation among four immutable objects:

```text
runtime release + platform + checkpoint revision + verification suite
```

it is never inferred from version order, filename, species similarity, or the
publisher's assertion. a newer runtime does not inherit compatibility. a
checkpoint does not become invalid when its runtime becomes posthumous.

## ownership and governance

- publisher identity and ownership transfers are signed records;
- runtime ids are stable namespaces and cannot be recycled;
- releases may be active, maintained, posthumous, quarantined, or unavailable;
- quarantine blocks execution and, where necessary, download without erasing
  the historical record;
- verification criteria and suite source are public and versioned;
- failed reviews include reasons and an appeal path;
- soma runtimes pass through the same manifest and verification machinery as
  third-party runtimes.

## first implementation

1. register v12.3 and v14 as reference runtime submissions;
2. accept third-party specifications and inert package artifacts;
3. expose publisher, species, runtime, release, package, and checkpoint pages;
4. implement manifest validation and immutable digesting;
5. add manual review before any third-party package can execute;
6. build the common isolated inspect/infer/train/resume/stop suite;
7. publish every badge with its exact evidence tuple.

## boundary

logos does not decide which geometry can think. it decides whether an artifact
is identifiable, whether an implementation is safely runnable, and whether a
capability claim has evidence. that narrow authority is what permits the field
to remain genuinely broad.
