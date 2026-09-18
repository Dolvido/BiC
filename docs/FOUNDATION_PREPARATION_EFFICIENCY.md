# Shared curriculum preparation: measured costs and next revision

The cost checkpoint below describes preparation. Later data verification and
calibration completed; the [main continuation](FOUNDATION_VARIANT_RECOVERY.md)
records the current phase. No efficiency prototype below was inserted into that
sealed comparison.

The architecture comparison completed preparation at 05:01:09 UTC on September
17, 2026. Its invocation took 1,621.016 seconds; the nested data-preparation
receipt reports 1,581.938 seconds. Independent data verification then started.
These are CPU preparation costs, with no formal optimizer updates yet recorded
at this checkpoint. The execution ledger (archive reference: `../runs/variant-study-execution-local/execution.json`)
is authoritative for later phases. The current study's 88 source files remain
frozen; the work below does not alter its lesson stream or acceptance rules.

## Curriculum compactness and evidence storage

The prepared calibration and main plans contain 48,960 and 294,912 distinct
training episodes, respectively. Their plan JSON files total 514,949 bytes.
Versioned generators and seeds reconstruct those lessons; full training rows
are not stored as the curriculum. Six learners per phase reuse each exact stream,
so the formal budget of 2,063,232 exposures is not that many unique lessons.

The data manifest nevertheless seals 628,054,541 bytes of artifacts. Historical
exact-transcript protection contains 1,277,856 digests, repeated in several
stage-specific protection files; stored evaluation rows and their metadata also
contribute. Compact lesson recipes therefore do not establish indefinitely
compact verification storage. These sizes and counts come from the completed
data manifest (archive reference: `../runs/variant-study-local/data/manifest.json`) and its bound stage
files, before the separate independent verification finishes. Neither a bounded
replay recipe nor a bounded validation cache solves growing transcript history.

Names-only admission repaired two calibration pairs and eight main pairs, with
one retry each. Their recorded programs, values and supervision remain unchanged;
the pending independent verifier must reproduce that first-safe ordering.

## Small non-neural CPU profile

A separate fresh fixture covered all six depths and all three history lengths,
with all three domains and 32 episodes per domain. Eighteen bundles produced
1,728 episode instances per pass. One ordinary pass and one instrumented repeat
materialized 3,456 instances total, with identical row-evidence digests. No formal
data, model or optimizer was used. See the profile report (archive reference: `../runs/foundation-cpu-profile-local/attempt-20260917T050119441312Z/report.json`).

The ordinary pass took 2.172 seconds: 1.469 seconds for materialization and 0.703
for row evidence. The instrumented repeat took 7.687 seconds and used a warm
generator cache. It observed 4,320 public pair-generation calls for 864 actual
pairs. Each actual pair is copied once for materialization and twice again in
each of two complete-pair validation passes. Every public call deep-copies both
variants. The profiler recorded 6,963,000 recursive `deepcopy` calls, with 7.107
cumulative seconds. These are instrumented call counts and nested timings;
they cannot be projected as a percentage saving in ordinary execution.

The first efficiency candidate is therefore reducing redundant expected-value
copies while preserving canonical validation. A private bounded cache of
immutable canonical row bytes can obtain truth through the unchanged generator
and independent oracles, then compare submitted rows without exposing mutable
cached objects. Strict recipe types, source identity, full pair invariants and
malformed-input rejection must remain intact. Public generation must still return
detached rows.

## Implemented standalone candidate and timing evidence

The new [canonical validator](../experiments/foundation_canonical_validation.py)
provides `CanonicalValidator.validate_pairs` for finite batches. It retains only
immutable expected row bytes, with explicit entry and serialized-byte limits.
Canonical truth comes through the unchanged public generator; candidate rows or
supplied hashes never establish truth. Exact recipe types prevent cache-key
aliases such as Boolean/integer equality. Source checks run before and after each
batch; a failed check clears the cache and invalidates the instance. It has no
persistent cache loader and is not connected to the running study.

All 13 focused checks (archive reference: `../runs/foundation-canonical-validation-local/attempt-001/report.json`)
passed in the first captured run. They cover all domains/depths/lengths, admission
roles, naming overrides, metadata/semantic tampering, caller mutation, original
JSON normalization, eviction, byte limits and source failures. Independent review
identified the unreadable-source failure case before testing; the corrected
implementation and its regression checks passed. The tests made 171 canonical
generator calls, attempted 491 candidate rows and validated 456 rows/225 pairs;
malformed cases explain the difference. Four cache evictions occurred. No neural
or optimizer work ran.

An independent unprofiled benchmark (archive reference: `../runs/foundation-canonical-benchmark-local/attempt-20260917T051005419977Z/report.json`)
used another fresh fixture of 864 pairs (1,728 episode instances), covering the
same depth/length/domain grid. Two opposite execution orders compared cold then
warm validation, clearing only the separate benchmark process's generator and
candidate caches between conditions. Every check accepted the unchanged fixture.

| Validation condition | Original, mean seconds | Candidate, mean seconds |
|---|---:|---:|
| Cold | 1.118 | 0.824 |
| Warm | 0.655 | 0.0995 |

Each mean contains two timing observations, not independent learning trials.
Cold means cleared recipe caches, not cold process, import or operating-system
caches. Candidate source checks are included once before and after its whole
864-pair batch; this ratio does not describe separate one-pair API calls.
Constructor/identity setup, common outer source checks, fixture hashing and
initial fixture construction are excluded from the timed intervals. The eight
validation batches visited 6,912 pairs/13,824 rows. Each
candidate instance retained 864 entries/8,840,904 encoded bytes, with zero
evictions under its 1,024-entry/32-MiB limits; generator caches and Python object
overhead are additional memory. The warm fixture fits entirely in that cache,
so this timing does not predict the hit rate of a much larger stream.

This supports a local validation speedup on the measured fixture. It does not
establish faster full preparation, training or useful learning per hour. The
active study remains unchanged. A future explicit integration must preserve
serial lesson, admission, anchor and prefix identities and measure the complete
path before claiming an end-to-end benefit.

Parallel bundle generation remains a later option if measurements justify its
startup, communication and memory cost. Acceptance must retain numeric bundle,
declared family and pair order, one global transcript set, the first safe naming
retry and original anchor order. Worker completion order must not change lessons
or receipts. No multiprocessing or validation shortcut has been inserted into
the running study.

## Repeated-cycle source-check overhead

The subsequent [repeated-cycle validation](FOUNDATION_CYCLE_IMPLEMENTATION.md)
measured about seven seconds inside each tiny child cycle's practice calls but
outside its retained training-step intervals. This is an observed overhead
interval, not a pure-gradient comparison. The automatic owner's two new
66-update cycles took 240.984 seconds, including admission, scoring, state
validation and publication. Its check suite and repeated-cycle fixtures remain
separate from the running GPU learning experiment.

A source-call profile (archive reference: `../runs/foundation-cycle-source-profile-local/attempt-001/report.json`)
then instrumented exactly ten calls to each current source-identity entry point:

| Entry point | Distinct files | Reads per call | Instrumented wall time for ten calls |
|---|---:|---:|---:|
| Cycle trainer | 81 | 1,056 | 0.969 seconds |
| Cycle owner | 83 | 1,058 | 0.947 seconds |

Both repeat 975 reads beyond one read per distinct file per call; 43 files are
read 18 times each. Nested source functions repeatedly hash overlapping sets
and their unions. The owner adds only two reads to the trainer's work. The
profile performed no model construction, checkpoint loading, inference,
optimizer work or CUDA initialization, and changed no implementation sources.

An isolated source inventory prototype (archive reference: `../runs/foundation-source-inventory-validation-local/attempt-001/README.md`)
now passes 19 pure filesystem checks. It reads each declared source once per
check, freshly validates the guard itself, and scans declared directory membership
before and after hashing. Failed or interrupted checks permanently invalidate the
instance. No file-content cache survives a check. All returned logical maps match
the earlier pinned 81-file trainer and 83-file owner maps.

| Entry point | Legacy reads | Inventory reads, including guard | Legacy warm median | Inventory warm median |
|---|---:|---:|---:|---:|
| Cycle trainer | 1,056 | 82 | 63.515 ms | 20.143 ms |
| Cycle owner | 1,058 | 84 | 70.317 ms | 22.606 ms |

Each median covers ten unprofiled calls with alternating execution order; counted
reads were measured separately. Constructor startup, including its complete check,
took 20.963 and 22.286 ms respectively. OS caches were uncontrolled. These roughly
3.1-fold improvements describe source checks only. The prototype is not integrated
into learning, capsules or the owner, and performed no neural work.

The guard preserves the declared nonrecursive membership scope and rejects
symlinks, reparse paths and nonregular sources more strictly than the historical
closures. It does not establish historical execution provenance or an atomic
filesystem snapshot. A future integration must explicitly version the affected
contracts and thread one checked inventory through nested operations; replacing
only the outer owner's function would leave duplicate work underneath. Existing
source-bound capsules must not be silently migrated. Complete admission, training,
scoring and publication costs still need measurement after integration. All 88
frozen formal-study files matched before and after prototype validation.
