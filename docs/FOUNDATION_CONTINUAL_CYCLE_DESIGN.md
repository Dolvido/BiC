# Continuing foundation learning across finite cycles

Design audit, 2026-09-17. This is an additive implementation proposal, not an
execution proof, learning result, or adoption of any study checkpoint. The active
flat/hierarchical study and all sources in its launch gate remain unchanged.

## Smallest useful next component

Add `experiments/foundation_cycle_training.py`: an authenticated completed-parent
capture and a cycle-aware trainer. Leave the current loop, provider, plan,
curriculum, evaluation code and English controller unchanged. The first API is:

```python
parent = CompletedParent.from_loop(completed_loop,
                                  training_transcripts=parent_transcripts)
child = CycleFoundationTrainer(
    new_plan, order="curriculum", parent=parent,
    device=device,
    admission_protected_transcripts=history,
    protected_transcripts=expanded_protection,
    admission_receipt=new_receipt, plan_index=new_index,
)
saved = child.snapshot()
# Resume reconstructs the same parent token and new-plan index, then supplies
# payload=saved. It does not call the finite trainer's restore on changed fields.
```

The child derives architecture, seed, model configuration and learning rate from
the authenticated parent; callers do not choose a replacement learning contract.
`capture_completed_parent(...)` may be a convenience alias for `from_loop`.

The parent object is process-owned, immutable, detached from caller tensors, and
nonserializable, like `AuthenticatedPlanIndex`. A serialized claim of completion
does not replace its factory. This protects ordinary API boundaries, not against
arbitrary Python modifying object internals.

Implement this component before an automatic scheduler. It can
prove faithful continuation of learning state without inventing an adaptive
learning policy. A separate cycle provider connects it to the existing
`FoundationLoop` for durable ownership. The first implementation supports one
completed ordinary variant loop handing off to one fresh cycle. Repeated cycles
and reconstruction of the parent chain on process restart require further work.

## Parent admission

The first factory supports an actual `FoundationLoop` containing an authenticated
variant provider, after exact envelope restoration or ordinary owned execution.
Call `loop.snapshot()` to check ownership, current disk identity, source identity,
complete state/history and provider state. Require all of:

- `phase == "complete"`; learner cursor equals its full admitted plan length;
  provider pending request and partial evaluation are both absent.
- The loop's full final evaluation exists and validates against the final weight
  producer. Exhausted training without the last evaluation is insufficient.
- Exact parent envelope/provider/learner schema, source identities, architecture,
  model configuration, optimizer settings and completed input evidence.
- The supplied sorted, unique training transcript list hashes to the process
  index's `transcript_sha256` and has its `unique_transcripts` count. A hash alone
  cannot supply the actual exclusions needed by the next admission.

Capture the exact weights, every AdamW tensor and parameter group, final lifetime
updates, prior exclusion union, authenticated parent training transcripts,
evaluation role/specification map, exposure evidence, retention references, and
lineage. Bind a durable parent to the verified envelope file digest. An in-memory
parent must be labeled as such and cannot claim a durable completed receipt.

The running architecture study produces study checkpoint envelopes, not loop
envelopes. The initial factory must reject them. A later, explicit study adapter
must bind the selected endpoint to the completed study protocol, job and worker
receipts, checkpoint verification, journal, file hashes and completed summary.
The adapter must restore the original endpoint under its original plan/index and
recipe before exporting state. Choosing an endpoint is explicit; neither the
summary's comparison screen nor a last-written checkpoint promotes a model.

## Exact subclass boundary

`CycleFoundationTrainer` can subclass `VariantFoundationTrainer` and retain its
`_build` plus the inherited `FoundationTrainer.step` unchanged. The step selects
`schedule[self.cursor]`, folds only current-plan evidence, and lets AdamW advance
its own moments. It does not require optimizer step to equal the local cursor.
The conflicting assumptions occur in snapshot/restore, not gradient arithmetic.

Before calling `super().__init__(..., payload=None)`, establish a separate frozen
`_cycle_sources` identity. Override `_assert_sources` to call the variant check
and then check this extra closure. Do not replace the superclass `_sources` map:
its constructor and checks expect exactly the variant closure. After construction,
extend the recipe with the new schema, parent identity and cycle metadata.
Construction may create seeded tensors temporarily; those are not the child
origin when a nonempty parent is imported.

Override `snapshot` and `restore`; add `base_updates` and `lifetime_updates`.
Preserve the inherited local `updates` property and explicitly document it.
A new schema needs its own strict restore preflight. The
current restore uses module-level schemas and assumes seeded zero-update weights,
so passing a rewritten old payload through it is invalid.

| Field | Required meaning |
| --- | --- |
| `cursor` | Completed updates in this cycle's finite new plan |
| `base_updates` | Authenticated parent's lifetime updates; immutable this cycle |
| `lifetime_updates` | `base_updates + cursor` |
| `updates` property | Current-cycle updates, preserving the finite trainer API |
| `evidence` | Exact new-index replay at the local cursor |
| `timing` | Current-cycle retained timing only |
| parent receipt | Prior evidence/costs, weights/optimizer identity and lineage |

Validate each AdamW `step` against lifetime updates, never the local cursor. With
a positive parent base, cursor zero still requires complete optimizer state.
At cursor zero, all weights, tied aliases, `exp_avg`, `exp_avg_sq`, `step` and
parameter groups must equal the authenticated parent exactly. At later cursors,
retain structural/finite/dtype/shape/moment checks and authenticated input replay;
this does not independently reproduce historical gradient arithmetic.

Require unchanged architecture, parameter topology/order, token/configuration
contract, objective, clipping and optimizer configuration, including learning
rate in version one. Do not reinitialize moments, reset their steps, reinterpret
the parent seed, change parameter groups, or migrate architectures implicitly.
Preflight candidate state before swapping it; retain failed-state poisoning and
transactional recovery semantics.

## Disjoint curriculum and persistent roles

Define `H_next = parent.protected_transcripts union parent.training_transcripts`.
Require next-cycle admission protection to include all of `H_next`; expanded
trainer protection additionally includes every newly reserved evaluation bank.
Use existing `repair_plan`, `prepare_training` and `AuthenticatedPlanIndex` to
authenticate the complete next plan against these sets before a child model is
accepted. Retain both original admission and expanded protection identities.

A new seed alone is not proof of freshness. Bind the new plan, receipt, complete
training digest list, data-source hashes and actual coverage/exposure manifest.
Retain all prior role exclusions even if a bank stops being scored. Development,
retention and sealed audit roles cannot be relabeled into training. Historical
audits do not become unseen again under a new name. A consumed audit becomes
historical evidence; future selection needs a separately reserved audit.

Initial cycle execution requires the same full development/retention bank
specifications and identities for comparable measurements. Supporting additional
banks requires a later explicit version. Exact
transcript disjointness establishes new realizations only; it does not establish
new words, unseen semantic procedures or general reasoning transfer.

## Loop and retention integration is a second component

Add a distinct cycle provider deriving from `FoundationPracticeProvider` through
the variant provider, with a versioned constructor/identity/restore. Its trainer
factory constructs the cycle-aware trainer and authenticates the same parent on
reload. The current provider constructor uses that overridable factory, but its
source/schema checks still require the same separate-extra-closure pattern and
new snapshot/restore schema. Preserve prescribed-prefix requests, scoring,
pending-request validation and physical-work accounting.

The existing `FoundationLoop` can keep local `cursor`, producer `updates`, window
boundaries and retained-work counts. Its provider identity must additionally bind
the parent and lifetime base. External reports must show both local and lifetime
updates; label old loop `updates` as cycle-local. This is a new cycle, not a
checkpoint whose completed work was erased.

Do not insert prior-cycle references into the existing loop's reference map: its
validator requires the current provider identity and local completed boundaries.
The first provider exposes the authenticated parent final evaluation and prior
reference map separately, without claiming cross-cycle alarms. A later cycle owner
must compare matching bank identities across cycles using producer identity
`(cycle_id, lifetime_updates, weights_sha256)`. Carry prior best references as well
as the parent endpoint, so forgetting earlier than the latest cycle is visible.
Fresh cycle-local baseline measurements must precede new practice. A complete
continual loop is not finished until this lifetime retention comparison is owned
and committed together with cycle transitions.

Use one durable owner. Publish a new child envelope to a new path with an explicit
parent hash; never rewrite the parent or reset its loop. Resuming returns to that
child's committed local cursor. A transition receipt distinguishes admission,
publication and completed-cycle states so a crash cannot launch a duplicate child
silently. Retained lineage counts and invocation physical work remain distinct;
an envelope alone cannot account for work lost in a hard process kill.

## Curriculum, tutor and control boundaries

One current update already practices all three families through a shared model,
using serial microbatch gradient accumulation and one optimizer update. This is
interleaved multi-domain learning, not simultaneous independent writers to the
same weights. Parallel preparation/validation can help; shared model updates need
one ordered owner.

Version one completes each admitted fixed schedule. It does not extend the list
mid-cycle, skip prescribed slots, replace pending work or adapt lessons from audit
scores. `RealizationBackend` provides the useful `base_updates + cursor` pattern;
its stream/curriculum schema is different and cannot directly load these weights
or replace the foundation provider. `EnglishLoop` supplies an explicit allocation
heuristic, not a drop-in adaptive controller for foundation plans.

Later curriculum work should declare reusable concepts, prerequisites, balanced
practice across domains, earlier-skill rehearsal, uncertainty handling and
independent transfer objectives prospectively. Avoid constructing a growing set of
lessons that merely imitate the current diagnostic failures. The present grammar
and finite color/count/switch worlds remain narrow, even across many cycles.

An external tutor can propose lesson recipes, explanations and broad curricula;
its outputs require independent truth/consistency checks before admission. Keep
tutor text and supervision out of evaluation. Record tutor calls, assistance
levels and cost; progressively reduce assistance only under a declared rule with
teacher-free retention and new-task transfer checks. Ethical, beneficial
self-direction requires its own explicit task/reward constraints and evaluations;
neither elapsed training nor this continuation mechanism establishes it.

## Validation order

1. Model-free contract tests: reject incomplete/fake parents, mismatched plan or
   transcript list, dropped exclusions, role changes, stale sources, changed
   architecture/settings, reused cycle identity and mutated returned objects.
2. Small CPU execution tests after the active study's resource reservation:
   exact parent-to-child weights and all optimizer moments; cursor-zero import;
   first child step at parent lifetime plus one; split/save/restore continuation;
   zero-work operations; malformed state leaves live state intact.
3. Provider/loop tests: parent completion requirement, initial child evaluation,
   local versus lifetime counters, inherited retention, pending-request resume,
   stale writers, publication failures and no duplicate successor on restart.
4. Separately budgeted strict-runtime proof before GPU use. Current finite-run
   proofs do not automatically prove cycle handoffs or a changed source version.
5. Only then a prospective learning comparison with matched data/work budgets,
   multiple initializations, no tutor at evaluation, old-skill retention, fresh
   realizations, held procedure/task transfer, ASK/calibration and cost reporting.

Steps 1-4 establish execution correctness. Step 5 can test learning benefit within
the declared tasks. Neither establishes general intellect by itself. No neural
tests or GPU work were performed for this design audit.
