# logOS ecosystem strategy

## purpose

logOS should become a public registry for architectures, executable runtimes,
and the persistent learned lives they produce. it is not a file browser for
whichever soma runtime is newest, and it is not limited to architectures that
descend from soma.

the larger proposition is that democratic training cannot require every voter
to be the same species. a person should be able to publish both a new way for
intelligence to take form and the checkpoints that have lived through it.
logOS supplies the conventions that let unfamiliar forms meet without making
their internal anatomy universal.

the governing distinction is:

- a **checkpoint** is an immutable revision of a learned state;
- a **lineage** is the continuing identity that connects revisions and forks;
- a **species** describes an architecture and its state transition;
- a **runtime** is the executable implementation that can inspect, run, and
  possibly continue training that species;
- a **runtime release** is one immutable, platform-specific implementation of
  that runtime;
- a **protocol** describes how a checkpoint receives and emits information;
- a **capability** is a tested operation, not an inference from tensor names;
- a **verification** is a reproducible observation made by a named test suite
  against one runtime release, checkpoint revision, and platform.

logOS should regulate the contract around those objects without requiring
their tensors, programs, learning rules, or state representations to share a
shape. architecture is relevant to routing and discovery, but it should not be
the universal admission test. soma v12.3 and v14 are the first reference
species, not privileged definitions of a model.

## core policy

the constitutional rule is:

> publication is open. execution is earned. recommendation is evidence-based.

anyone may submit a species description, runtime manifest, immutable runtime
release, or checkpoint. publication records what the author claims and what
bytes were submitted. logOS verification records what the system has safely
observed. recommendation remains a separate editorial or community judgment.

the right to submit a runtime is not permission for that runtime to execute on
logOS infrastructure or a user's machine. those permissions arrive only
through explicit trust and verification states.

logOS should accept an artifact when it can answer five questions:

1. what immutable artifact is this?
2. what life and lineage does it belong to?
3. which runtime claims to understand it?
4. what interfaces and capabilities does that runtime expose?
5. what has logOS independently verified?

none of those questions require `weights`, `net`, `traces`, `organs`, a fixed
tokenizer, or any particular neural architecture.

quality must remain separate from admission. a checkpoint may be valid and
poorly trained. fluency, loss, safety, and benchmark results belong in evidence
and reputation, not in the file-format gate.

a runtime submission must answer a parallel set of questions:

1. who published this implementation and under what license?
2. what exact source and package bytes constitute this release?
3. which species, formats, protocols, and platforms does it claim?
4. what resources and privileges does it require?
5. which of those claims has logOS independently reproduced?

## admission without architectural requirements

use progressive admission states rather than one valid/invalid decision.

### 1. preserved

the uploaded bytes are stored immutably and addressed by a server-computed
sha-256 digest. the manifest is syntactically valid, ids are well formed, file
size is within policy, and moderation/provenance requirements are satisfied.

logOS does not execute the artifact to reach this state.

### 2. described

the artifact has an architecture-neutral ecosystem manifest containing
identity, lineage, runtime requirements, interface, tokenizer contract,
capabilities, and experience. unknown fields are preserved.

legacy adapters may synthesize this envelope from v12.3 and other known
sidecars. the checkpoint itself must not be rewritten merely to modernize its
listing.

### 3. runtime available

the required runtime exists in the runtime registry for at least one platform,
and its immutable package digest is known. this means “there is a maintained
way to open it,” not yet that the artifact has passed a live test.

### 4. runnable

a disposable, network-isolated worker has loaded the artifact with the declared
runtime and completed a bounded inference smoke test.

### 5. trainable

the worker has also completed a bounded resume test: load, consume a small
fixture, update, save a new revision, reload it, and verify identity and state
continuity. continual-training claims should receive this badge only after this
test.

### 6. evidenced

benchmarks, corpus declarations, sample outputs, and community reports may be
attached as evidence. they never rewrite the immutable artifact record.

the site can therefore preserve an unfamiliar species immediately while being
precise about what it does and does not know.

## universal checkpoint manifest

new runtimes should emit an adjacent json sidecar. existing species keep their
native fields, but add one common envelope:

```json
{
  "sidecar_version": 4,
  "ecosystem": {
    "schema": "logos.checkpoint.v1",
    "identity": {
      "checkpoint_id": "stable life id",
      "revision_id": "immutable revision id",
      "parent_revision_id": "nullable parent revision",
      "root_lineage_id": "root life id"
    },
    "artifact": {
      "filename": "model.pt",
      "media_type": "application/x-pytorch-checkpoint",
      "bytes": 123456789
    },
    "runtime": {
      "id": "soma.v14",
      "version": "14.0.0-dev.2",
      "checkpoint_format": "soma-v14-pytorch",
      "checkpoint_version": 14
    },
    "model": {
      "species": "soma_v14_adaptive_trace_branch_mixer",
      "parameters": 93820422,
      "modalities": ["text"]
    },
    "interface": {
      "input": "text",
      "output": "text",
      "protocol": "soma-text-turn-v1",
      "tokenizer": {
        "id": "soma-1k-v1",
        "sha256": "...",
        "vocabulary": 1024
      }
    },
    "capabilities": [
      "inference.text",
      "training.continual",
      "training.resume",
      "learning.online",
      "state.temporal_trace"
    ],
    "experience": {
      "source_bytes": 409639444,
      "tokens": 156459526
    }
  }
}
```

the upload service computes and stores `artifact_sha256`; it must not trust a
digest claimed by the uploader. the same applies to upload byte length.

the envelope deliberately excludes architecture-specific dimensions. those
remain in the species section of the sidecar and may be interpreted by a
species adapter. this keeps the common schema small enough to survive future
architectures.

## identity and lineage

identity must never depend on a filename.

- `checkpoint_id` is the stable identity of one continuing model.
- `revision_id` identifies one immutable save.
- `parent_revision_id` forms the revision edge.
- `root_lineage_id` groups forks that share an origin.
- the server-computed artifact digest identifies the exact uploaded bytes.

the database should reject a collision where one `revision_id` is presented
with different bytes or incompatible identity fields. uploading identical
bytes again should be idempotent.

a normal save creates a new revision under the same checkpoint id. a fork
creates a new checkpoint id, retains the root lineage id, and points its parent
revision at the source revision. deletion should tombstone a listing rather
than silently erase lineage edges.

## runtime compatibility

compatibility is a many-to-many registry backed by tests, not a string
comparison and not an architectural guess.

each immutable runtime release should publish:

```json
{
  "runtime_id": "soma.v12.3",
  "runtime_version": "12.3.17",
  "status": "maintained",
  "platforms": ["macos-arm64"],
  "package_sha256": "...",
  "checkpoint_formats": [
    {
      "id": "soma-v12.3-token-residual",
      "versions": [1],
      "capabilities": [
        "inference.text",
        "training.continual",
        "training.resume",
        "learning.online"
      ]
    }
  ]
}
```

the registry, rather than the checkpoint, is authoritative about future
compatibility. a checkpoint can truthfully identify the runtime and format
that created it; it cannot know which future runtimes will load it.

resolve a download in this order:

1. exact runtime id and version, when available;
2. another registry release with a passing compatibility result for the exact
   checkpoint format/version;
3. preserved-only download with a clear “runtime unavailable” label.

never silently open a checkpoint with the newest runtime. never imply that a
newer species supersedes the learned state of an older one.

## open runtime publication

runtime publication is a first-class logOS workflow, not an administrator-only
exception. every submission uses `logos.runtime.v1` and produces an immutable
runtime release record.

minimum manifest:

```json
{
  "schema": "logos.runtime.v1",
  "identity": {
    "runtime_id": "org.example.my-runtime",
    "release_id": "immutable release id",
    "version": "1.0.0",
    "publisher_id": "verified publisher id"
  },
  "architecture": {
    "species": ["org.example.my-species"],
    "specification": "artifact or permanent url"
  },
  "source": {
    "repository": "https://example.invalid/repository",
    "commit": "full immutable commit",
    "license": "spdx identifier"
  },
  "packages": [
    {
      "platform": "macos-arm64",
      "media_type": "application/x-apple-diskimage",
      "filename": "runtime.dmg",
      "sha256": "server verified digest",
      "bytes": 123456
    }
  ],
  "execution": {
    "entrypoint": "declared adapter entrypoint",
    "network": false,
    "filesystem": "workspace-only",
    "accelerators": ["cpu", "mps"],
    "minimum_memory_bytes": 8589934592
  },
  "checkpoint_formats": [
    {
      "id": "org.example.checkpoint",
      "versions": [1],
      "capabilities": ["inference.text", "training.resume"]
    }
  ],
  "interfaces": [
    {
      "protocol": "logos.text-turn.v1",
      "input": "text",
      "output": "text"
    }
  ]
}
```

the uploader's hashes are claims until logOS recomputes them. package bytes,
source snapshots, specifications, licenses, tokenizer assets, build recipes,
and dependency manifests should remain attached to the immutable release.
changing any executable byte creates a new release id.

### runtime admission states

- **preserved** — manifest and package bytes are immutable, hashed, scanned,
  and retained; nothing has executed;
- **described** — species, interfaces, formats, privileges, source, license,
  and platform requirements satisfy the common contract;
- **reproducible** — logOS or an accepted builder reproduced the package from
  the declared source and recipe, or recorded why reproduction is unavailable;
- **reviewed** — static review and dependency policy have passed;
- **executable** — the release completed a no-network, resource-bounded
  runtime smoke test in isolation;
- **compatible** — named checkpoint fixtures passed declared load, inference,
  save, and resume tests;
- **recommended** — evidence, maintenance, documentation, and user experience
  justify surfacing it to new users.

a runtime may be publicly discoverable while only preserved or described. the
site must make that state unmistakable and must not offer one-click execution.

### verification contract

verification suites are themselves immutable, versioned artifacts. a result
must name the runtime release, platform image, suite version, checkpoint
fixture, resource limits, logs, and timestamp. badges expire only when policy
says so; they are never silently transferred to a different package digest.

the common suite should test, where claimed:

1. package integrity and entrypoint discovery;
2. bounded startup with no undeclared network or filesystem access;
3. checkpoint inspection without mutation;
4. bounded inference through the declared protocol;
5. save/reload identity continuity;
6. bounded training and resume continuity;
7. stop/cancellation behavior and resource cleanup;
8. declared cpu, memory, accelerator, and disk envelopes.

architectures may publish additional test suites. they cannot replace the
common safety and identity suite with self-authored evidence.

## runtime lifecycle

use neutral lifecycle labels:

- **recommended** — default for new users;
- **maintained** — supported and receiving fixes;
- **posthumous** — frozen, preserved, and still downloadable;
- **superseded build** — replaced by a security or packaging fix that retains
  the same checkpoint compatibility;
- **unavailable** — artifact preserved but no executable package is hosted.

posthumous must not mean invalid. a posthumous runtime is an environment
preservation promise: immutable package, tokenizer assets, specification,
license, platform, digest, and known limitations remain together.

security fixes should produce a new runtime build with a new package digest.
the old package remains identified but may be blocked from automatic download.

## security boundary

pytorch checkpoints and third-party runtimes are executable-risk surfaces.

- never deserialize an uploaded checkpoint in the web process;
- compute hashes while streaming the upload;
- inspect only json sidecars in the request path;
- perform known-runtime validation in an ephemeral worker with no network,
  read-only runtime files, strict cpu/ram/time/output limits, and disposable
  storage;
- accept third-party runtime submissions as inert, immutable artifacts;
- never execute an unreviewed package merely because it is publicly listed;
- only reviewed, isolated runtime packages may receive executable or
  compatible badges;
- keep user claims, runtime claims, and logOS-verified facts visibly distinct.

the ecosystem is runtime-open and execution-gated. this is not a compromise on
openness: preservation and publication do not require code execution, while
verified execution is a separate capability with a deliberately higher bar.

## governance

logOS should govern conventions and evidence, not architectural taste.

- soma receives no schema privileges unavailable to another species;
- common schemas change through versioned proposals with migration guidance;
- verification suites and badge criteria are public and versioned;
- publisher identity, package signing, ownership transfer, and revocation are
  explicit records;
- security quarantine removes execution and distribution privileges without
  erasing the historical identity of an artifact;
- rejected executable status is appealable and includes a machine-readable
  reason;
- benchmark authorship and conflicts of interest remain visible;
- popularity does not become proof of capability;
- a species may define its own vocabulary, state, learning rule, and ontology
  while still obeying common identity, integrity, interface, and safety rules.

## v12.3 preservation

v12.3 should remain a first-class maintained species and may remain the public
default while v14 matures.

the server adapter can synthesize the universal envelope from the existing
v12.3 sidecar:

```text
runtime.id                    soma.v12.3
runtime.version               runtime_version
runtime.checkpoint_format     soma-v12.3-token-residual
runtime.checkpoint_version    1
model.species                 species
model.parameters              params
model.modalities              [text]
interface.protocol            soma-text-turn-v1
interface.tokenizer.id        tokenizer_name
interface.tokenizer.sha256    tokenizer_sha256
interface.tokenizer.vocabulary vocab_size
identity.checkpoint_id        checkpoint_id
identity.revision_id          revision_id
identity.parent_revision_id   null when unavailable
identity.root_lineage_id      checkpoint_id when unavailable
experience.source_bytes       bytes_seen
experience.tokens             tokens_seen
```

this is a view-layer migration. do not alter or reserialize v12.3 checkpoints.
the v12.3 runtime page should carry its own app, script bundle, tokenizer,
specification, release notes, and compatibility statement.

## site information architecture

### checkpoint page

lead with the individual model:

- name and profile;
- species and lineage;
- experience;
- verified capabilities;
- compatible runtime and platform;
- artifact revision and digest;
- evidence and sample outputs;
- download checkpoint;
- download/open with the correct runtime.

architecture details should be expandable rather than used as the primary
identity.

### species page

explain the architecture, protocol, tokenizer, known runtime implementations,
population, representative checkpoints, and compatibility boundaries.

### runtime page

show publisher, source, license, lifecycle and trust states, platforms, package
digests, supported checkpoint formats, specification, release notes,
permissions, downloads, and verification results. unverified packages must
never resemble one-click trusted installers.

### architecture page

lead with the idea rather than its package: specification, state and learning
model, protocols, runtime implementations, verification suites, population,
representative checkpoints, forks, and open questions. this is where a new
architecture enters public intellectual life.

### submit runtime

provide a guided flow for publisher identity, species registration, manifest
validation, source and license declaration, immutable package upload,
permission review, fixture attachment, and verification status. submission
must work before automated execution exists; a runtime can begin as a
preserved specification and package.

### lineage page

show revisions and forks as a graph. revisions belong to a life; species and
runtime changes should appear only through an explicit migration/fork record,
never by relabelling the old checkpoint.

### discovery

filters should include modality, capability, species, runtime availability,
platform, parameter range, experience, and verification state. “recommended”
may rank results but should not hide maintained or posthumous species.

## suggested data model

minimum server entities:

- `artifacts`: digest, byte length, storage key, media type, scan status;
- `checkpoints`: checkpoint id, species id, root lineage id, owner;
- `revisions`: revision id, checkpoint id, parent revision id, artifact digest,
  manifest, created/saved/uploaded times;
- `species`: species id, description, protocol ids, adapter id;
- `publishers`: identity, signing keys, verification and moderation state;
- `runtimes`: stable runtime id, publisher, species relations;
- `runtime_releases`: release id, version, source commit, manifest, lifecycle
  and trust states;
- `runtime_packages`: release, platform, artifact digest, signature, scan and
  reproduction status;
- `runtime_formats`: runtime release, checkpoint format/version, declared
  capabilities;
- `verification_suites`: immutable suite id, version, author, fixtures and
  policy class;
- `verification_runs`: revision, runtime release, test suite version, result,
  metrics, logs, timestamp;
- `profiles`: mutable user-owned presentation fields kept separate from the
  immutable revision manifest.

## api behavior

recommended endpoints:

```text
post /api/checkpoints/inspect
post /api/checkpoints
post /api/species
post /api/runtimes
post /api/runtimes/:runtime_id/releases
post /api/runtime-releases/:release_id/packages
post /api/runtime-releases/:release_id/verify
get  /api/checkpoints/:checkpoint_id
get  /api/revisions/:revision_id
get  /api/revisions/:revision_id/download
get  /api/species/:species_id
get  /api/runtimes
get  /api/runtimes/:runtime_id/:version
get  /api/compatibility?revision_id=...
```

upload is a transaction:

1. stream artifact and compute digest;
2. parse the sidecar only;
3. normalize through the universal envelope or a legacy adapter;
4. validate ids and reject contradictory collisions;
5. commit immutable artifact/revision records;
6. queue runtime verification;
7. publish immediately as preserved/described, then add verification badges
   asynchronously.

## rollout

### phase 1 — preserve both known species

- keep v12.3 public and downloadable;
- add v12.3 and v14 runtime records;
- normalize both sidecars into one internal manifest;
- remove architecture-specific tensor-key requirements from the upload path;
- add species/runtime/verification labels to checkpoint pages.

### phase 2 — immutable reference runtimes and verification

- store package digests and platform artifacts;
- run load, inference, save, and resume smoke tests in isolation;
- display runnable/trainable badges backed by verification records;
- resolve every checkpoint download to the correct runtime.

### phase 3 — open runtime publication

- add species and lineage pages;
- expose posthumous runtime downloads;
- permit unknown species to be preserved with unverified status;
- accept `logos.runtime.v1` submissions from verified publishers;
- publish specifications and inert runtime packages before execution review;
- add review, reproducibility, permission, and verification states;
- expose architecture and runtime submission pages.

### phase 4 — plural ecosystem

- permit reviewed third-party runtimes in isolated verification workers;
- publish common and architecture-authored verification suites;
- support signed package updates, revocation, and ownership transfer;
- allow checkpoints to migrate across runtimes only through explicit,
  evidenced migration records;
- consider federation once artifact identity and verification semantics are
  stable.

## acceptance criteria

the first implementation is complete when:

1. a v12.3 checkpoint uploads without modification and resolves to v12.3;
2. a v14 checkpoint uploads through `ecosystem.schema` and resolves to v14;
3. the same revision uploaded twice is idempotent;
4. a revision-id collision with different bytes is rejected;
5. an unknown species with a valid manifest is preserved but not marked
   runnable;
6. the web process never deserializes checkpoint tensors;
7. every runnable/trainable badge names the runtime and verification result
   that justifies it;
8. posthumous runtimes remain discoverable and downloadable;
9. unknown manifest fields survive profile edits and re-publication;
10. filenames never determine identity, species, or compatibility;
11. a third-party runtime can be preserved and described without being
    executed;
12. no executable badge can migrate between package digests;
13. a reviewed third-party runtime can prove compatibility with its own
    checkpoint format through the same public suite used by soma;
14. runtime permissions and provenance are visible before download;
15. soma-specific fields never appear in the universal runtime gate.

## language for the public site

recommended framing:

> logOS is where architectures are published, runtimes are verified, and
> learned lives continue.

> species describe how a model is built. lineage describes what it has lived.

> anyone may submit a form of intelligence. verification says what it can
> demonstrably do.

> new architectures do not erase old learning. each checkpoint opens with the
> runtime that knows its form.

avoid “legacy model,” “unsupported architecture,” or “invalid checkpoint” when
the precise state is posthumous, runtime unavailable, or not yet verified.

## decision

logOS should be permissive about form and strict about claims.

that is the balance that allows architectural democracy without turning the
site into an unsafe file dump: preserve broadly, identify immutably, publish
openly, execute conservatively, verify capabilities, and never confuse the
newest or most familiar runtime with the only valid kind of mind.
