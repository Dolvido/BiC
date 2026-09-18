# Durable ownership of repeated curriculum cycles

Implementation and design, 2026-09-17. The goal is a bounded restart path for admitted cycles while
preserving learned weights, full AdamW state, exclusions and retention evidence.
This does not authorize adopting formal-study weights or establish learning
benefit. Existing frozen study sources remain unchanged.

## Implemented interface

`experiments/foundation_cycle_owner.py` supplies `CycleOwner.create(directory,
parent=..., retention=..., evaluation_banks=..., plan_config=..., root_seed=...,
loop_options=..., device="cpu")`, `CycleOwner.load(path, expected_sha256=...,
min_generation=..., device="cpu")`, and `owner.run(max_updates=...,
max_cycles=..., deadline=...)`. At least one finite allowance is mandatory.
`max_cycles` counts newly completed cycles in that call; the monotonic deadline
is absolute. The owner follows deterministic fresh balanced plans. This is not
an adaptive or self-directed learning policy.

Update exhaustion can finish the current evaluation panel, but permits no new
plan or optimizer update. A zero-cycle allowance does no work. Completed cycles
are atomically folded into an explicit boundary state; later calls may admit the
next child, so a resumed boundary cannot count the same completed cycle twice.
The returned `owner_sha256` is the next exact trusted head pin.

Each invocation publishes an exclusive intent before work and an atomic complete
accounting receipt afterward. Missing, malformed, truncated or unknown-work
receipts block ordinary continuation. `load(...,
recovery_policy="acknowledge_unknown")` explicitly records uncertainty in the
owner and a separate immutable recovery artifact, preserving prior evidence.
An existing run lock must be independently resolved; this policy never removes
one automatically. It never infers zero lost work or silently retries a failed
invocation. Intent/receipt inspection grows with invocation count, while restart
loads one current parent capsule and at most one active plan index.

## Parent capsule: persist verified evidence once

The implemented capsule supplies a versioned, explicit persistence boundary around the process-owned
`CompletedParent`; keep its ordinary pickle/sidecar constructor unavailable:

```python
receipt = export_parent_capsule(parent, new_path)
parent = load_parent_capsule(
    new_path, expected_sha256=receipt["sha256"],
    device=device, evaluation_banks=exact_banks,
)
```

Export accepts only an exact `CompletedParent` minted by a validated completed
loop or a previously authenticated capsule load. It checks the token's sources
before and after serialization, publishes to a new path without replacing any
existing capsule, and returns the exact file-byte SHA256. The exported artifact
contains one terminal learner, its bounded completion/retention metadata, the
full current training transcript inventory and cumulative protected inventory.
It contains fixed-size predecessor identity/envelope digests, not ancestor
capsules or recursively nested historical learners.

The loader's trust root is the caller-pinned exact artifact hash from a known
export. A digest copied out of that capsule, its neighboring receipt, or an
untrusted mutable index is not an independent pin. A hash identifies bytes; it
does not by itself prove that those bytes came from a validated execution.

Read one immutable byte image, authenticate its hash, then deserialize those
same bytes with `weights_only=True` on CPU. Reject unsupported schemas, unknown
fields, noncanonical values and size/shape limits before exposing a token. There
is no directory discovery, latest-file selection, automatic migration or fallback
to a raw checkpoint. The loader remints a process-owned token only after all
checks succeed; detached public values and nonserializability remain unchanged.

## Loader checks and the exact claim

The capsule is trusted for the historical facts already established by its
exporting process. The loader independently checks its current compatibility and
internal invariants:

- Exact capsule/token/learner schema; current source closure and runtime match;
  canonical architecture/configuration, objective, optimizer recipe and bank
  specifications. Direct source hashing includes trainer, provider, capsule
  exporter/loader and their existing dependencies without a circular call between
  capsule and cycle modules. Nested dependency hashing still repeats file reads;
  the [efficiency profile](FOUNDATION_PREPARATION_EFFICIENCY.md#repeated-cycle-source-check-overhead)
  measures that separate cost.
- Recompute all metadata, weight, optimizer, index and transcript-list digests.
  Require terminal local cursor equal the admitted index's bundle count, no
  unfinished request/evaluation, and exact completion metadata.
- For a cycle parent, `lifetime = base + local_cursor`; the captured parent's
  `base_updates` for its successor equals that lifetime. Ordinals increment once.
  Original variant parents have lifetime equal to their completed local cursor.
- All weights and optimizer tensors are finite CPU tensors with permitted
  dtypes/shapes and no gradients. Optimizer groups equal the captured canonical
  recipe; state covers exactly their parameter IDs, every AdamW step equals
  lifetime, the two moments agree in shape/dtype, and second moments are
  nonnegative. Tensor layout and alias semantics established at live capture are
  inherited through the pin; do not claim to reconstruct them without a trusted
  captured layout or an architecture template.
- Recompute the final producer using `foundation_loop._producer` /
  `checkpoint_digest`. This differs from the capsule's tensor-tree digest; do not
  substitute one digest algorithm for the other. Evaluation `updates` remain
  cycle-local even when optimizer steps are lifetime values.
- Require exactly every declared role/name bank, normal control, the captured
  provider identity, final local cursor and actual final weight producer.
  Independently canonicalize the caller's exact banks and validate metric counts,
  denominators and progress with the existing metric contracts. Validate retained
  references against the same bank identities and their historical producer
  coordinates. Neither numeric consistency nor a pin reruns model predictions.
- Require sorted unique transcript inventories with exact counts/digests; current
  training inventory matches the authenticated index digest and is included in
  cumulative exclusions. Prior exclusion preservation was authenticated during
  live capture and is inherited from the pinned export. No role can be converted
  from development, retention or audit into training during import.

These checks do not reproduce past gradients, recreate old model predictions, or
independently regenerate historical curricula. That provenance is inherited from
the known pinned export. Therefore no historical model, old optimizer execution,
or recursive old-plan index construction is required. If independent canonical
replay is separately required, rebuild only this capsule's latest finite parent
plan/index under its exact sources; that is an additional verification mode, not
an implied property of a capsule load.

The runtime memory/time is independent of the number of ancestor models, but not
of all training history: exact cumulative transcript exclusions grow with unique
transcripts and must still be read and checked. Do not call the entire capsule
constant-size. No source-version migration is implicit; archived old source
inspection does not make a new runtime compatible with an old capsule.

## One owner envelope, immutable capsules

The durable cycle owner owns one atomic envelope containing:

```text
schema, source/runtime/options
generation and local/lifetime coordinates
active parent capsule {relative_path, file_sha256, identity_sha256}
active admitted data references and exact evaluation identities
active FoundationLoop snapshot
lifetime retention fold and its authenticated producers
retained accounting, transition identity and publication state
```

Keep the inner `FoundationLoop` unbound to a separate disk path. It may commit
internally in memory, but only the owner publishes durable active state. A mutable
HEAD pointing to a separately overwritten child checkpoint is unsafe: a crash
between those two writes can destroy the child version named by HEAD. If separate
child artifacts are needed, make them immutable and publish the owner pointer last.

At a cycle boundary, require a completed inner loop, capture its parent token,
fold lifetime retention against matching banks, and export its capsule first.
Then atomically publish a boundary envelope with no active child. Only if the
caller's allowance remains does a later operation admit the next complete disjoint
plan against cumulative exclusions, create its provider and initial evaluation
state, and publish the active envelope. A failure before either replacement leaves
the last published envelope authoritative. Candidate changes do not mutate the
committed image. Unreferenced capsules are harmless orphaned artifacts and are
never adopted automatically.

Use the existing publication discipline: one owner/run lock, stale-writer digest
checks, flush/fsync, replace, and post-error determination of whether old or new
bytes reached disk. Uncertain publication requires explicit reload. On restart,
load the one pinned parent capsule, rebuild only the active child's admitted
index/provider, and restore the inner loop from the owner's snapshot. Do not
reconstruct earlier ancestors to satisfy a child constructor.

## What the caller pins on restart

Expose `CycleOwner.load(path, expected_sha256=owner_head_sha256, ...)`. The
authenticated owner envelope transitively pins its parent capsule and active data
artifacts. Every successful durable save returns a new owner-head digest for the
trusted caller to retain. A direct capsule import instead requires the capsule's
own independently supplied digest.

A pin to generation zero does not authenticate arbitrary later successors merely
because they contain hash links. Authenticating an evolving head needs the exact
current head pin from a trusted caller/store, or a separately designed authenticated
append log. Neither a self-hashed HEAD file nor an editable neighboring receipt
provides that authority. An old valid pinned head may intentionally replay an old
state; `min_generation` rejects a parent cycle below the caller's supplied floor.
Do not promise rollback detection without such outside state. Intentionally
restoring an older envelope while keeping a later recovery artifact can fail on
that artifact's exclusive publication. This rare rollback path remains fail-closed;
it requires explicit inspection and does not silently replace recovery evidence.

## Measured owner validation

`runs/foundation-cycle-owner-validation-local/attempt-001/report.json` records
10 passing tests in 435.709 seconds: six publication/policy checks and four actual
tiny CPU integration checks, using one PyTorch thread and a separately pinned
completed cycle-one fixture. All 88 frozen formal-study source hashes still match.

The automatic run completed two fresh cycles (132 updates) in 240.984 seconds,
stopped at the boundary at lifetime update 198, then admitted cycle four only on
a later bounded call. The preserved result reaches local update 4 in cycle four,
lifetime update 202. Resumed and independently restored branches matched exact
weights, optimizer moments, evidence and lifetime-retention state after two more
updates. Instrumented restart loaded one capsule and built zero indexes at the
boundary or exactly one active index during a cycle.

Across all test branches, instrumentation recorded 140 physical optimizer calls
and 840 training episode exposures. This includes two deliberately discarded
updates after failed publication and two updates on the comparison branch; the
preserved owner retains 136 updates beyond its 66-update origin. Thirty-seven
banks (74 evaluation episodes) were scored, including one deliberate foreign
inner-loop evaluation subsequently rejected by the owner's entry guard. No
unknown optimizer work occurred. The failure test restored the previous complete
weights, moments and evidence. Separate interruption fixtures established that
missing/truncated receipts block continuation and preserve their evidence through
explicit uncertainty acknowledgement.

The main automatic run reports 40.859 seconds of training, 11.722 of scoring,
6.109 of subsequent admission, 10.203 of completion, and 44.966 of checkpoint
publication. Its larger total also includes identity/source/snapshot checks and
other coordination work. These tiny-fixture timings establish execution costs,
not performance advantage. A separate CLI check authenticated the preserved
active owner in 11.5 seconds and executed a zero-cycle allowance in 2.797 seconds,
with zero training/scoring and unchanged owner bytes. Its additional invocation
files and receipt are listed separately from the ten-test artifact inventory.

The preserved owner pin is
`62049b4417663732f8be6dbfc145d49321e5f132b73cd71f69a284eee26aaaf6`;
the tested owner source pin is
`413c8388a7dd55de0fab0b4341d86791fc94126339692606897685173cf7abcf`.

## Bounded validation

Before relying on restart, test capsule roundtrip, detached state, wrong pin,
changed bytes, unsupported schema/source/runtime, malformed finite tensors or
moments, local/lifetime confusion, final producer mismatch, missing/relabelled
banks and exclusion inventory tampering. Test a three-cycle uninterrupted versus
restart path, each boundary's exact weights/moments and lineage, retention fold,
and failures before/after capsule publication and owner replacement. Assert that
restart authenticates only one parent capsule and constructs only the active
child index, never an ancestor chain. Corrupt fixtures must be tested at the
validator boundary as well as through pin mismatch, so a correct hash check does
not mask missing invariant checks.

All are execution/integrity tests. Teacher-free transfer, skill retention,
curriculum quality, useful self-direction and measured resource benefit require
separate prospective learning evaluations. The original design audit performed
no neural work. Subsequent owner implementation validation is recorded separately
under `runs/foundation-cycle-owner-validation-local/`; it uses tiny CPU fixtures
and no formal-study data, learned weights or GPU work.
