# Shared learning-rate continuation from the completed objective study

## Question and evidence

Does reducing the learning rate from 0.0003 to 0.0001 improve native acquisition
or preservation of earlier abilities during continued whole-curriculum practice?
Keep the auxiliary objective at 0.3 in both branches. This is one fixed local
comparison, not a rate sweep, architectural change or home-checkpoint adoption.

The preceding objective comparison failed the declared benefit screen for
auxiliary removal: its equally weighted 12-cell effect was -2.0833 percentage
points. With the unchanged objective, practiced revision improved from 17 to
39/60 complete pairs, while development revision changed 35/34/58/54 out of 120
and development binding 11/10/20/10 out of 120. Transfer remained almost absent
and retention failed. Continued acquisition therefore remains possible, while
stability deserves a shared test. These observations do not establish that the
current rate is excessive or that either learner has reached a plateau.

Earlier rate evidence contained fitting/transfer tradeoffs. A lower rate is not
assumed superior. AdamW's decoupled weight decay also scales with its learning
rate, so any effect belongs to this whole rate intervention with inherited
optimizer history, not to an isolated gradient-noise or representation mechanism.

This protocol fixes the comparison before its launch. Implementation, numerical
validation and an authenticated freeze must complete before training starts.
It neither extends an earlier run nor alters its criteria after seeing results.

## One common research parent

Use the final **control** of `runs/shared-objective-local/attempt-001`, at native
update **15,592**, for both branches. Choosing that final parent follows the
failed auxiliary-removal comparison; no earlier endpoint is selected for having
a better development score. Both branches begin with the same complete weights,
AdamW moments and individual steps, evidence and historical accounting.

| Evidence | SHA-256 |
| --- | --- |
| Completed objective launch | `f6670f922676306500d6f97e91c83a2dbdb171f2ae3293cebecc8fcf623d804e` |
| Completed objective summary | `6edaf731cd155e3906727a31c4215b48213c424e03d16af781be6ceb63110c2e` |
| Completion marker | `dcc734c69f424c2b5058ef5d5fb4467413433562fc9094059a540915e3dbdb3d` |
| Independent raw recount | `b1edee2751047dd3be93e6e5e32f9c9e6e51f531653476cb478f66178703e417` |
| `execution/checkpoints/control-2592.pt` | `fe13decc662f63f89089c3d9fc628ae74c77d2c6924bc20cfbe28e94e279f3c9` |
| Native weight digest | `5bb595e19f81c60cc77ffa5ca098acc72421789b6b6287e97e0d7e71b36a35f1` |

The checkpoint path above is relative to the completed objective directory.
Its snapshot precedes that study's final restoration/evaluation; the endpoint
commit includes later administrative restore costs. Preserve this distinction
when authenticating historical accounting rather than fabricating equality.

## Explicit rate transition

Branches are `control` (0.0003 to 0.0003) and `lower` (0.0003 to 0.0001).
Both use an explicit authenticated rate-transition envelope. Keep the incoming
objective transition as historical provenance. Change only the rate field in
each optimizer parameter group and the corresponding declared recipe fields.
All other optimizer settings, weights, moments, individual step values, evidence,
model structure, objective terms and clipping threshold remain equal initially.
The optimizer-state difference allowed at this boundary is exactly the rate.

An additive `SharedRateContinuation` adapter must bind the new kernel recipe,
source closure and transition identity; directly mutating a guarded optimizer
is invalid. Old continuation classes and frozen sources remain unchanged.
The adapter must support its own full-state snapshot and authenticated reload.
Keep native and auxiliary accounting cumulative and report new physical work
separately. Report nested archive loads, model/optimizer construction and snapshot
work without adding nested timers to their enclosing durations.

Before launch, require metadata rejection checks and exact CPU/CUDA numerical
proofs covering both rates, their first updates, full-state reload, and further
updates against the corresponding direct kernel references. A changed rate is
not expected to match the unchanged-rate update; each must match its own declared
reference. Preserve failed proof attempts and their costs.

## Fixed teaching, computation and exposure

Reuse `runs/complementary-composition-data-local/attempt-001/manifest.json`, SHA
`f5a71eafc49e0b99f9fa893b0c1f9385df26b618e12c79a99b9dd2f7747de896`.
Both branches repeat its original **curriculum** schedule, in the same order,
for four complete passes of 648 updates. No new examples, tutor requests or
curriculum choices are introduced. Original source-bundle identity remains
verifiable while consumption cursors advance from 15,592.

Each branch receives 2,592 updates: 144 basis, 720 complementary composition,
432 earlier-definition and 1,296 broad replay updates. Every update contains
32 episodes from each of color, count and switch, with micro-batch size 32.
Each branch therefore consumes 248,832 episode exposures. The total physical
plan is **5,184 updates, 497,664 exposures, 15,552 family forwards and backwards,
and 15,552 auxiliary readouts/objectives**. These are repeated exposures to
existing lessons, not counts of new independently generated examples.

Retain the native 2,264,725-parameter SharedStateStudent, original preparation
and data loaders, state-target construction, objective weight 0.3, gradient clip 1,
strict FP32 runtime, no TF32, and one CPU/inter-op thread. Do not integrate the
new JSON cache, packed-lesson variants or source-read optimizations in this
comparison. Their isolated timing evidence cannot substitute for native parity
and end-to-end throughput evidence.

Run control then lower sequentially, with no overlapping model jobs. The fixed
worker allowance is **3,600 seconds**, inclusive of imports, setup, authentication,
training, scoring, snapshots, restorations, cleanup and durable summary
publication. Freeze/setup cost outside the worker is reported separately.
Completion-marker publication is separate administrative cost. Check deadlines
between operations; do not claim an individual operation can be preempted.
An overrun or failure preserves its receipts and does not trigger an automatic
retry, allowance extension or replacement run.

## Endpoints and native evaluation

Score the same 16 banks at relative updates **0,648,1296,2592** for each branch.
At each nonzero endpoint, save and reload full state before scoring. This gives
six snapshots and eight logical restorations, including both initial transitions.
The final lifetime cursor is **18,184** per branch. Authenticate all 16 baseline
metrics against the common 15,592 parent's final scores before proceeding.

Expected scoring remains 128 bank endpoints, 208 raw score files and 35,232
episode evaluations. Independently recount all native predictions and all 5,184
step reports. Learning targets and auxiliary state labels never become policy
inputs. Inference uses native observations, actions and freely generated replies;
the external tutor supplies no evaluation answers.

Report every fixed endpoint, with fit/development, family, known/unknown,
destination/prior-value and action/reply/joint metrics kept separate. Definition
success requires every declared question in both counterfactual episodes to have
the correct action and exact freely generated reply. Broad-bank focal-pair
success is an easier, different metric and must not be pooled with this one.

## Prospective comparisons and absolute capability

The primary acquisition-effect screen is the final **lower-minus-control**
comparison across the 12 complementary fit/development x binding/revision x
family complete-pair cells. Use exact rational arithmetic and equal cell weights:

1. The 12-cell mean effect is at least five percentage points.
2. No individual cell regresses.
3. The six-cell mean effect is strictly positive in fit and development separately.
4. The four-cell mean effect is strictly positive in each family separately.

Report small, mixed and negative differences even if this screen fails. A failed
acquisition-effect screen does not establish equal rates or rule out a retention
tradeoff. Report preservation of earlier abilities independently: all per-check
changes and failed-check counts for each branch against every reference below.
Do not relabel reduced loss, fewer failures in one subject, or an earlier favorable
endpoint as passing the primary acquisition screen.

The secondary basis-transfer screen compares the six composition/sequence family
cells: final lower-minus-control equal-cell mean at least five percentage points,
with no cell regression. Also report each branch's transfer change against every
reference. These finite, reused development panels provide engineering evidence,
not a statistical replication or fresh general-English audit.

Absolute complementary acquisition requires at least 75% complete pairs in all 12
cells. Absolute original basis capability requires at least 75% in all 12 binding,
revision, composition and sequence family cells. Keep those levels separate from
comparative rate effects; one can improve without reaching useful capability.

## Five retained references and selection limits

Allow at most five percentage points of loss in every one of 96 native retention
checks against **each** of these fixed references:

1. Common immediate control parent at 15,592.
2. Prior common curriculum parent at 13,000.
3. Earlier curriculum parent at 10,408.
4. Original shared research parent at 9,760.
5. Fixed historical reference at 9,112.

Each set retains the same 75 broad-bank family joint/known/unknown action/reply
checks, nine earlier-definition complete-pair checks and 12 basis checks. This
is **480 checks per branch**. Authenticate older references through the verified
parent chain; do not replace a failing historical reference with a newer one.

Research capability eligibility requires all complementary acquisition and basis
capability cells, all five retention sets, and strictly positive six-cell basis
composition/sequence progress against the original 9,760 parent. Operationally
valid archives remain distinct from scientifically eligible candidates.

There is no automatic adoption, further-rate sweep or task-specific repair.
The home learner remains separately retained at 9,328. Any later adoption needs
a prospective comparison against the then-current home learner and fresh
evaluation. Neither this study nor a successful numerical proof demonstrates
general English competence, learned self-direction or a home-hardware ceiling.
