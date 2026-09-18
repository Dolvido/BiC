# Foundation practice in a continuing English loop

The design was written without reading foundation pilot scores. The separate
[practice provider](../experiments/foundation_provider.py) is now implemented and
passed 15 CPU tests, including exact continuation and failure handling. It binds
original admission protection separately from expanded training/evaluation
protection and validates the entire admitted training plan before model creation.
The separate [durable controller](../experiments/foundation_loop.py) now passes
14 focused tests, including actual CPU continuation through partial evaluation
and pending practice. Neither component changes the frozen pilot. Adaptive lesson
choice still requires a distinct scheduling contract and evidence of benefit.

## Existing boundaries

- `FoundationTrainer.step()` consumes the next immutable bundle in its declared
  order: three complete-pair family microbatches, unchanged objectives, one
  optimizer update. Its snapshot already binds curriculum version, plan/order,
  protected transcripts, source identities, weights, AdamW state, consumed-ID
  digests and exposure counts. Failures poison further use until restoration.
  See `experiments/foundation_training.py:148`, `:209`, `:319` and `:329`.
- `FoundationBank` authenticates the actual foundation curriculum and inherits
  observation-only scoring and BOS-only generated replies. Its bank identity
  preserves `bic-shared-foundation-v1`; `foundation_metrics.validate_metrics`
  checks the canonical denominators without relabeling it as composition data.
  See `experiments/foundation_evaluation.py:20` and
  `experiments/foundation_metrics.py:171`.
- `EnglishLoop` already joins evaluation, allocation, practice, retention
  references and atomic controller/learner publication. However, it requires a
  concrete `RealizationBackend`, accesses private evaluation rows and realization
  stream fields, validates metrics through capacity/composition helpers, and
  constructs a RealizationBackend during load. Its schedule is three family
  names per update, not a foundation lesson ID. See
  `experiments/english_loop.py:87`, `:143`, `:167`, `:331` and `:456`.
- `RealizationBackend` supplies useful chunk/deadline/rollback accounting, but
  instantiates `RealizationTrainer` and composition `PreparedBank` directly.
  Passing foundation rows to that class is not an integration API. See
  `experiments/realization_backend.py:92`, `:157`, `:303` and `:420`.

## Small provider contract

`FoundationPracticeProvider` wraps an unchanged FoundationTrainer
and prepared FoundationBanks. The outer controller remains the sole durable
writer. Provider methods return isolated JSON/tensor-safe values:

| Member | Required meaning |
|---|---|
| `identity` | Provider schema, actual curriculum version, admitted plan and admission-receipt hashes, protection hash, source closure, model/optimizer recipe and execution profile. |
| `specs()` | Named development and retention banks, their exact identities, and typed cell metadata: family, causal depth, primitive mechanism and turn length. Both roles require `dev` admission; role and curriculum version remain distinct fields. |
| `status()` | Updates, curriculum cursor, remaining finite work, pending request, per-family/per-cell microbatches and actual episode/token/byte exposures. Resource limits are explicit capabilities, not assumptions about a realization `seen` set. |
| `evaluate(role, names, control)` | Existing teacher-free scoring and raw per-bank counts/rates; no optimizer, lesson consumption or provider-state changes. Include measured cost and weight identity. |
| `validate_evaluation(response, expected_weights_sha256=..., expected_updates=...)` | Foundation-specific bank/role/version/config/denominator validation, then a common progress vector. Historical validation requires both producing-state arguments from an authenticated controller record; it does not independently rerun that historical model. |
| `practice(request, max_updates, deadline)` | Consume a bounded, validated request; report completed, failed, retained and uncertain work separately. No internal durable checkpoint publication. |
| `snapshot()` / `restore(payload)` | Full trainer plus pending-request state, with strict cursor/exposure/identity checks. Restore is transactional; publication uncertainty cannot silently resume. |

The common progress vector can retain paired action, paired generated reply,
known accuracy and unsupported ASK, with their denominators. Keep final-query,
per-turn and action/reply-agreement evidence available for interpretation rather
than reducing every cell to a scalar. Provider validation must call foundation
validators; changing a version string to satisfy an older checker is forbidden.
Parsed programs, lesson metadata, targets and allocator decisions stay outside
the student's observation inputs.

## First integration: prescribed practice

Use requests such as
`{kind: "prescribed_prefix", provider_sha256, plan_sha256, order, start_cursor, stop_cursor}`.
The provider accepts only the next prefix of its existing order. A prescribed
policy allocates a finite next window; evaluation records progress and retention
alarms but cannot reorder lessons, repeat bundles or shorten the declared budget
based on scores. Every update already covers all three domains. Earlier-depth
rehearsal remains whatever the admitted plan actually specifies; arbitrary
window boundaries must not be advertised as having an unverified depth floor.

The controller follows this transaction pattern:

1. Evaluate the fixed development and retention queue, resumably one bank at a
   time, at one bound weight identity.
2. Record complete vectors and persistent retention references; issue the next
   prescribed window.
3. Practice its exact bundle prefix. Advance policy state only by completed work
   present in the candidate learner snapshot.
4. Atomically publish weights, optimizer, admitted-plan identity, request/cursor,
   prescribed-window state, partial evaluation queue, retention references and accounting
   in one envelope. On failure restore the last verified envelope; preserve
   physical/discarded/unknown work independently from retained counters.

The load path selects an explicitly registered provider schema and requires
matching external plan/banks/protection; it must not import a class named by
untrusted checkpoint metadata. Source/profile/identity drift fails closed.
Initial preparation, canonical replay on restore, evaluation and checkpoint I/O
remain visible costs. They are not GPU optimization time.

This is a separate engineering run. The current frozen matched pilot has its
own evaluation schedule and must not acquire intermediate feedback. Comparing
teaching orders stays distinct from demonstrating adaptive curriculum benefit.

## Implemented durable API

`FoundationLoop(provider, window_updates=16, chunk_updates=4, history_limit=32)`
requires a compatible zero-update provider without a pending request. The caller
first constructs `FoundationPracticeProvider` with the exact admitted plan,
order, seed, model configuration, development/retention banks, original
`admission_protected_transcripts`, expanded `protected_transcripts` and admission
receipt. The historical exclusion set cannot be inferred from the visible banks.

- `run(max_updates=None, max_evaluations=None, deadline=None)` requires at least
  one explicit bound. Update allowances are integers from 0 through 4,096;
  evaluation allowances are 0 through 1,000,000 banks. A deadline is a finite
  absolute `time.monotonic()` value. Time limits are checked between banks and
  updates, rather than interrupting neural operations. An evaluation-only bound
  permits zero practice. `max_updates=0` still permits baseline or final scoring.
- `snapshot()` returns a detached tensor/JSON envelope containing the full
  provider, optimizer, pending prefix, loop phase and window, partial evaluation
  queue, producing cursor/weights, bounded history, persistent per-cell
  references and retained-lineage work counts. Historical response identities
  and denominators are validated without treating their own claimed hashes as
  independent producing-state evidence.
- `save(path)` binds the owner to one new envelope path. It writes a temporary
  file, flushes/fsyncs it, and atomically replaces the owned envelope. Existing
  paths require explicit loading. `FoundationLoop.load(path, fresh_provider)`
  authenticates the saved options and provider/source/runtime identity and
  restores the full envelope. It does not adopt arbitrary pilot checkpoints.
- `status` is a property exposing the phase, retained cursor, evaluation cursor,
  completed evaluation count and prescribed window. After the final finite
  prefix it remains in evaluation until every development and retention bank is
  scored, then reports `complete`; it never recycles the plan. A publication or
  external-state ambiguity reports `reload_required` with unknown progress.
- `last_report` distinguishes completed provider updates, physical optimizer
  calls, attempted/completed episode exposures, returned scoring banks,
  retained/durable/discarded deltas, and unknown optimizer or scoring work.
  Failed chunks are captured before rollback clears the provider report.

Single-writer guards cover the envelope directory and provider object. Public
entry points compare the complete provider state with the last owned envelope;
foreign practice or weight changes cannot be silently adopted. A definite failed
publication restores the last committed provider and controller together. If an
exception occurs after replacement, the actual file hash determines whether the
new envelope published, the old one remains, or explicit reload is required.
Evaluation responses are committed one bank at a time and remain bound to one
producing weight/cursor throughout their queue.

Retention comparisons preserve paired action, paired generated reply, known
accuracy and unsupported ASK for every bank. Best references persist when old
history records are dropped; retained alarms are recomputed against a bounded
reference anchor during load. Neither aggregate improvement nor a different
cell's gain suppresses a recorded regression. These are observations, not
mastery thresholds, promotion rules or adaptive lesson decisions.

Full admission/loaded-envelope validation remains a trust-boundary operation.
Exact response-content caches are bounded, bind provider/source identity and
producing state, and are empty on a new owner. Private commit snapshots share
immutable historical responses while copying every mutable container. Public
snapshots are detached. These reduce repeated validation/copying but do not
establish home-scale throughput: **each bank commit still serializes the full
learner/history envelope**, and load/rollback regenerates consumed canonical
lessons. The window/chunk defaults are engineering choices, not measured optimal
settings. Invocation reports expose evaluation, practice, checkpoint and rollback
wall intervals; components are included subsets of total invocation time.

Retained-lineage accounting is not a complete lifetime compute ledger. Failed
and discarded work survives in the invocation's in-memory report, but a hard
process kill can leave unreported work because no durable started-work marker is
implemented. Unknown completion stays unknown. No GPU run, tutor, background
execution, audit input, autonomous extension or adaptive policy is included in
this component's validation.

## Later integration: genuinely adaptive curriculum

The existing `english_allocation` policy selects families. Relabeling depth or
operator cells as families would violate its three-family contract and the
foundation trainer's fixed cursor. The clean extension is a new deterministic
lesson scheduler over a prospectively admitted finite pool of **joint-domain
bundles**, preserving one microbatch per domain per update. It chooses reusable
dependency depth/mechanism/length, not bespoke domain-specific repairs.

That extension needs a new trainer/provider accepting explicit admitted bundle
IDs, with a persisted selection stream and per-cell counters. It cannot invoke
the current fixed-order trainer with a fabricated cursor. Repeated rehearsal of
an admitted lesson must be explicit and counted; fresh naming/value generation
would be a separately versioned capability with its own admission and resume
state. Depth-one COPY and ADVANCE remain separate measured mechanisms; depth
comes from final-query ancestry, not utterance count.

Declare feasible minimum coverage for retained primitive/depth/length cells per
completed window, reserving those slots before allocating the remainder. Use
development-only observed progress per measured shared-window cost, preserve
every regression/unsupported-ASK alarm, and use deterministic ties and cold
start. This remains an engineered observational heuristic, not causal estimates
of cell efficiency or evidence of learned self-direction. No mastery thresholds,
automatic promotion or audit-driven allocation are introduced here.

An external tutor may later propose typed lesson requests outside the student;
the same canonical provider admission must independently verify them. Neither
tutor text, explanations nor hidden solutions enter policy observations. Tutor
benefit and decreasing dependence require their own comparison.

## Evidence required before use

The integration requires exact direct-trainer/provider behavior, split
resume across partial evaluation and practice, strict plan/protection/role/version
and counter checks, and preserved failure/publication accounting. A later adaptive
scheduler must restore coverage debt and decisions exactly; scores must never
change the prescribed baseline schedule. The provider evidence below covers its
in-memory scope. The controller's CPU evidence follows; adaptive validation
remains separate.

The provider tests now establish the prescribed in-memory boundary: actual
direct-trainer equivalence, pending-prefix resume, teacher-free evaluation with
unchanged learner state, canonical role/version/protection admission, strict
historical producing-state bindings and explicit uncertainty after failures.
The final suite passed 15 checks in 24.329 seconds; all author iterations together
used 32 actual CPU optimizer calls, 204 returned candidate episodes, 200 neural
attempts and 196 completed-microbatch episode exposures. Those include deliberate
failure work and are separate from all formal GPU budgets. Provider snapshots
carry complete learner and pending-request state; the outer controller still
owns durable publication and rollback, with the lifetime-accounting limitation
described above.

The final controller suite passed **14 tests in 21.326 seconds**: nine finite,
non-neural control fixtures and five tiny canonical CPU integration tests. They
cover exact direct-provider equivalence and AdamW continuation across a saved
one-bank evaluation and partly consumed prefix, final evaluation with no update
allowance, tampered producer/reference/count rejection, stale ownership, definite
and ambiguous publication failures, partial neural failure and optimizer
uncertainty. The test fixtures use no pilot inputs or scores.

That final run used 11 actual CPU optimizer calls; 72 drawn episodes, 70 neural
attempts and 68 completed-microbatch episode exposures; and 48 successfully
returned scoring banks covering 96 episodes. A prior integration run failed only
in test normalization of integer AdamW keys after nine actual optimizer calls;
its work remains counted. Across both neural test runs, totals are **20 optimizer
calls, 132 drawn episodes, 128 neural attempts, 124 completed-microbatch episode
exposures, and 84 returned scoring banks/168 episodes**. Both runs deliberately
include one post-optimizer exception whose completion the controller correctly
reports as uncertain, even though the test instrument observed the arithmetic.
An earlier non-neural fixture-clock failure added no learning/scoring work.

Captured console excerpts, failure descriptions, final hashes and accounting
are preserved under `runs/foundation-loop-validation-local/receipt.json`. They
were preserved from the completed tool outputs without rerunning tests; these
are engineering evidence and are separate from all formal study GPU budgets.
