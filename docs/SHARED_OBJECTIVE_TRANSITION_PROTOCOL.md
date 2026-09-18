# Shared state objective transition: prospective acquisition comparison

This document proposes the fixed comparison and decision rules for review before
launch freeze. It does not authorize an unrecorded change to a completed study,
an existing continuation recipe, or the home learner. No result from this new
comparison has been inspected when setting these rules.

The preceding sustained study showed continued, late acquisition across all
three families, while exact-fit acquisition remained weak and structural-program
transfer remained zero. That positive trajectory makes continued weight-0.3
training an essential control. The hypothesis is that removing the state
auxiliary objective may improve general acquisition at this learned state; its
benefit or harmful-gradient explanation is not assumed.

## Fixed parent and two objective branches

Both branches start from the **same curriculum research checkpoint at update
13,000** in `runs/sustained-composition-local/attempt-001`. They receive exactly
the same complete incoming model, tied weights, AdamW parameter groups, moments,
per-parameter step counters, evidence cursor and cumulative accounting.

Pinned source artifacts:

| Artifact | SHA-256 |
| --- | --- |
| Curriculum update-13,000 checkpoint | `4780cd241a17516003e2e162b45d2fb3072d9f533e68586edb4e562de4b412fa` |
| Completed parent launch | `4bd912311f64cf25939aa7eb437bfd09e265d23ed2b9ab7911e32ef237cbe40d` |
| Completed parent summary | `fb46f0972dd37d5d273dfee51a017b63fc22c2e0d6856cd65ba2746421da7246` |
| Completed parent terminal marker | `db150cd061e6876e11495163a12fe7d2f2655aae000207faf8859a3fb45f1335` |
| Independent parent recount | `7ac87b22735316769293576f9e90996f773a1b2effbf4a6b5169c7f76b1c4acf` |
| Reused complementary data manifest | `f5a71eafc49e0b99f9fa893b0c1f9385df26b618e12c79a99b9dd2f7747de896` |

- **Control, 0.3 → 0.3:** retain the original sequence objective plus 0.3 times
  the existing shared-state loss.
- **Zero auxiliary, 0.3 → 0:** retain the identical sequence objective and
  architecture, with auxiliary weight zero. Use the kernel's established
  zero-weight path, which skips the auxiliary graph entirely.

The initial native action logits, reply outputs, full weights and optimizer
state must match across branches. Both baseline evaluations must reproduce all
16 native banks from the parent's fixed final endpoint. Parameter initialization,
learning rate, optimizer configuration, clipping, strict FP32 runtime, data order,
microbatch size and native inference are unchanged. No new seed-dependent model
initialization, optimizer reset or architecture is part of the intervention.

Weight zero is not a context-detached auxiliary decoder. In the zero branch,
`state_alias_embedding` and `state_decoder` parameters receive no gradients;
their weights, AdamW moments and individual AdamW step counters must remain
bitwise equal to the update-13,000 parent at every saved endpoint. They stay in
the model and optimizer groups. Native parameter updates and their optimizer
counters continue normally. The control continues updating its auxiliary
parameters. The intervention therefore removes auxiliary training and its
shared-encoder contribution, not merely the displayed scalar loss.

## Explicit transition identity and restart contract

Use a new, additive `experiments/shared_objective_continuation.py` boundary.
Keep the original continuation and trainer sources unchanged. A transition must
authenticate the source checkpoint bytes before data-only deserialization,
retain the original origin identity and complete source recipe, and explicitly
bind the source weight 0.3 and destination weight 0.3 or zero. The new envelope
must distinguish source identity, declared objective change, and resulting
kernel recipe. It must never pretend a zero-weight descendant is an unchanged
old 0.3 continuation.

The control also receives an explicit 0.3 → 0.3 transition record, so both arms
cross the same admission boundary. Subsequent restores preserve that recorded
transition; they must not silently retarget the objective or append another
transition. Checkpoint source pins, wrapper source pins and incoming tensor,
optimizer, cursor, evidence and accounting identities are required.

The additive bridge's public boundary is
`SharedObjectiveContinuation.from_continuation(raw, expected_sha256=...,
expected_identity=..., auxiliary_weight=0.3|0.0, device=...)` for this one explicit
transition. Its own `from_snapshot` requires the exact archive SHA and
`expected_transition_sha256`; it does not accept a replacement objective.
`step(prepared, state_targets=..., deadline=...)` and `snapshot()` retain the
established prepared-bundle interface. The new envelope is
`bic-shared-objective-continuation-v1`, with a separately identified
`bic-shared-objective-transition-v1` record. Bind the old archive, identity,
recipe, bridge costs and starting evidence/accounting alongside the new recipe
and source map. Any data-only metadata decode or legacy snapshot used in
admission is physical work and must be counted; nested timings must not be
summed twice.

Preserve all inherited work, state-work and cost counters. Record new physical
work and transition/restore/snapshot costs separately. Exact startup and restart
checks must include full optimizer state, not only model weight digests. Focused
equivalence checks compare the wrapper with the ordinary supported kernel at
each declared weight; the zero branch additionally proves auxiliary parameters
and optimizer state remain unchanged after an update. A failed boundary or
unknown optimizer outcome is preserved and cannot be reported as a completed
matched comparison.

## Matched practice and compute allowance

Both branches use the **curriculum** arm's exact ordered 648-record schedule from
the pinned data manifest, repeated four times. The old atomic control schedule
is not used. Each branch receives 2,592 additional updates and 248,832 repeated
episode exposures, reaching lifetime update **15,592**. Every update uses 32
episodes from each of color, count and switch, with one synchronized optimizer
update after three family forwards/backwards.

Per branch the extension comprises:

| Lesson kind | Updates | Episode exposures |
| --- | ---: | ---: |
| Atomic basis | 144 | 13,824 |
| Complementary composed meanings | 720 | 69,120 |
| Earlier definitions | 432 | 41,472 |
| Broad replay | 1,296 | 124,416 |
| Total | 2,592 | 248,832 |

Repeated source records and targets remain unchanged. Only the top-level
consumed bundle/evidence cursor advances from 13,000 through 15,591. Every step
binds source position `relative_index % 648`, replay pass
`1 + relative_index // 648`, and global consumed cursor `13000 + relative_index`.
Record all actual target checking and preparation work. No new lesson bank,
teacher call, raw-example authoring, cache integration, schedule selection or
family-specific allocation is part of this experiment.

The paired total is **5,184 updates, 497,664 episode exposures and 15,552 family
forwards/backwards**. Expected new auxiliary readouts/objectives are 7,776 each
for control and zero for the zero branch. Do not charge a skipped readout as
physical neural work or erase its inherited historical counts.

Evaluate at additional updates **0, 648, 1,296 and 2,592**. Save and restore the
complete state before each trained endpoint evaluation: two initial transitions,
six subsequent full restores and six snapshots, or eight model restorations
overall. Use the fixed final endpoint for all decisions. Retain every earlier
endpoint to describe acquisition, forgetting and instability; do not select a
better earlier checkpoint after seeing scores.

One inclusive **3,600-second worker deadline** covers both sequential branches,
worker setup, training, evaluation, restoration, cleanup and durable summary
publication. An immutable completion marker must bind the summary SHA and elapsed
time measured after publication; both summary and marker must report completion
within the worker allowance. The small marker's publication is separately
reported administrative bookkeeping. CPU freeze, focused validation and pure
recount costs are separate. There is no automatic retry, score-dependent early
stop, release promotion, overlapping model job or hidden budget extension.

Compare equal exposure first, and separately report wall time, steps per second,
peak allocated/reserved GPU memory and state-head computation saved. Lower
execution cost alone is not an acquisition benefit.

## Native measurements and primary objective-effect screen

Use all **16 unchanged banks**: four complementary fit/development panels,
four basis panels, three earlier-definition panels and five broad banks.
Evaluation supplies only normal observation text to the native policy. Neither
teacher nor state-head outputs provide answers. Generate replies from BOS and
retain raw action predictions, exact reply tokens, targets and query identities
outside model inputs.

The primary measure is `all_query_pair_both`: every query in both counterfactual
episodes must have the correct native action and exact free reply. For each of
the 12 complementary cells (fit/development × binding/revision × three families),
let its final effect be the zero-auxiliary accuracy minus control accuracy.
Each fit cell has 20 pairs and each development cell 40 pairs.

The **general acquisition effect** passes only if all of these hold:

1. The unweighted mean of the 12 cell effects is at least **five percentage
   points**.
2. None of the 12 cell effects is negative.
3. The six-cell mean effect is strictly positive separately for fit and for
   development.
4. The four-cell mean effect is strictly positive separately for color, count
   and switch.

Use exact integer counts and rational comparisons, without rounding before a
threshold check. Do not pool unequal fit/development denominators. These rules
test a broad objective effect without allowing one subject or training-fit alone
to carry the result. This is a fixed single-parent engineering screen, not a
statistical confidence or replication claim.

Report all cell counts at every endpoint, each branch's change from the common
13,000 parent, split and family mean effects, and action-only/reply-only/joint
counts. Separately report known/unknown and destination queries, prior-value
questions and action/reply agreement. Easy unchanged-state questions cannot
substitute for execution accuracy. The control's continued learning is evidence,
not a failure of the experiment: the treatment must be compared with that
measured trajectory rather than with a frozen parent alone.

## Absolute capability, transfer, retention and admissibility

Keep the following distinct from the new primary objective-effect claim:

- **Complementary acquisition:** at least 75% complete pairs in every one of the
  12 complementary fit/development family cells, evaluated separately per arm.
- **Original basis capability:** at least 75% complete pairs in every one of the
  12 basis binding/revision/composition/sequence family cells, per arm.
- **Comparative basis transfer:** final zero-minus-control equal-cell mean of
  the six basis composition/sequence cells at least five points, with no
  regression in any of those six cells. This remains a separately reported
  secondary claim; it is not the primary acquisition measure of this study.
- **Within-branch transfer progress:** report six-cell changes against the
  common 13,000 parent, earlier curriculum 10,408 parent, original 9,760 parent
  and fixed historical 9,112 reference. Eligibility requires strictly positive
  progress against the original shared 9,760 parent.

For each arm, retention allows at most five percentage points of loss in every
one of 96 native checks against **each** of four references:

1. The common immediate curriculum update-13,000 parent.
2. The previous curriculum update-10,408 research parent.
3. The original shared update-9,760 research parent.
4. The fixed historical update-9,112 reference.

Each set comprises 75 broad-bank family joint/known/unknown action/reply checks,
nine earlier-definition complete-pair checks, and 12 basis complete-pair checks.
Report all four sets (384 checks per arm), without replacing a historical
reference with whichever newer learner is convenient. The original 9,760 and
historical 9,112 references are already bound by the completed preceding study;
derive the 10,408 curriculum reference from its verified own-parent metrics.

Research capability eligibility per arm requires all complementary acquisition
checks, all basis capability checks, all four retention sets and positive
composition/sequence progress against the original 9,760 parent. Report that
eligibility separately from the primary comparative objective effect and the
secondary paired transfer claim. A mechanism effect can be informative while
both learners still fail absolute capability or retention. A result is never
automatically adopted, and an operationally valid checkpoint is not evidence of
capability eligibility.

The home learner remains at update **9,328** on its separate lineage. This study
neither compares against nor replaces it. Any later adoption requires a separate
prospective comparison with the then-current home state and fresh evaluation.

## Verification and interpretation limits

Authenticate the source closure, original parent chain, declared transition,
data, full restart evidence, source schedule and every endpoint commit. Recount
raw native predictions independently. Expected evaluation coverage remains
128 bank endpoints, 208 score files and 35,232 episode evaluations. All16 baseline
metrics must equal the parent's final metrics and each other. Actual per-arm
work must reflect the different state-head execution counts while preserving
identical episode exposure and original native forward/backward counts.

These are repeatedly observed development banks and exact-fit subsets, not a
fresh blind test. Structural programs may be functionally equivalent in some
domains. Neither success nor failure establishes general English competence,
learned self-direction, or a universal architecture limit.

Entity targets are invariant to some rule-definition changes, but a shared EOS
representation can encode both rule meaning and entity values. An improvement
after removing the auxiliary objective would establish an effect of this
specific intervention from this parent; it would not by itself prove gradient
conflict or explain the mechanism. Both branches inherit the same AdamW history
and retain global gradient clipping at norm 1. Removing the state loss changes
the total gradient and can change clipping, so update magnitude and optimizer
dynamics can mediate any effect; it does not uniquely identify representational
interference. A null or negative effect would bound this
intervention and exposure budget. No architecture change, detached-gradient
variant, new teacher curriculum or extra repetition may be silently substituted
for the declared comparison.
