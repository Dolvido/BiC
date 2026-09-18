# Requested-word selection diagnostic

## Question and decision

The completed [rate comparison](SHARED_RATE_RESULTS.md) established a conditional
acquisition benefit for rate 0.0001, with 479/480 retention checks passing, but
unfamiliar program composition remained 0/96 and sequence 1/96. Actual practiced
binding continued improving late. These observations justify neither a capacity
ceiling nor another blind rate sweep.

This new, inference-only diagnostic asks whether changing **only the requested
word in the first Apply instruction** changes native behavior according to the
two meanings already defined in the same context. Definitions, initial facts,
argument aliases, question literals and the second Apply instruction remain
identical between pair members. This tests a missing behavioral contrast; it
cannot alone distinguish internal binding storage, selection, execution or
answer-production mechanisms.

If the lower-rate learner improves this contrast consistently across families,
continued shared acquisition remains a live explanation and control. If it
fails this contrast despite gains on definition-change tests, review one shared
word-to-program representation intervention before spending on another broad
practice extension. Success on this contrast with weak program composition
instead prioritizes reuse of selected meanings. These are interpretation branches,
not post hoc thresholds, promotion criteria or automatic training extensions.
No individual weak cell receives special lessons or revised acceptance rules.

## Fixed candidate states

Evaluate exactly three saved states, in this order:

1. The common incoming objective-control parent at lifetime update 15,592.
2. The final unchanged-rate control at lifetime update 18,184.
3. The final lower-rate branch at lifetime update 18,184.

The rate launch SHA-256 is
`057a6a06932749fd712dd10d32b9e68e6625d2db510aa739f8302af4a6bc21cf`;
its completed summary is
`6345f2128674853ea052712e590660914787defcf75ad75544ae2da59c0231ed`;
completion is
`45543fc1cf65a3d2305fd313f13f8895d20137b82c179e5e3096987f032c43a3`;
and verified independent recount is
`922679ce9b665c836cd7f21b8af6aa5de0d0c23545da6e7d69ff30bb7e8e2943`.
The preparation binds these records and their exact checkpoint/weight pins.
No intermediate checkpoint or home checkpoint is selected using diagnostic scores.
The home learner at 9,328 is separate and remains unchanged.

## Full factorial bank

Use the existing complementary binding skeleton: two definitions, two initial
facts, a first application at zero-based turn 4, immediate source/destination
questions at turns 5/6, a second application with swapped arguments at turn 7,
then its source/destination questions, an unknown-entity question and a further
source question. There are exactly 12 turns and six questions per episode.
The second application always requests the atomic word in both pair members.

One word denotes a composed meaning from
`CD, DC, SD, DD, SS, DS, DCD, CDD, SSD, SDD`; the other denotes `C`, `S` or `D`.
Thus the context retains the practiced one-composed-plus-one-atomic form. `C`
copies source to destination, `S` advances source, and `D` advances destination.
This bank introduces no `SC`, `CS`, `SCS` or `CSS` definition.

The complete bank has **720 pairs / 1,440 episodes**, comprising:

- Three families: color, count and switch.
- Ten composed codes and three atomic codes.
- Both definition orders.
- Two initial relationships: destination equals source, or is one advance ahead.
- Both probe-reference variants: each immediate question's literal is the
  corresponding outcome under one of the two requested meanings.

The explicit probe-reference factor balances opposite answers when a role changes.
Within each pair, variant 0 requests the composed word and variant 1 the atomic
word. Every other visible turn is byte-identical. Both probe-reference settings
share aliases, nonce words and base values; changing definition order also leaves
those assignments fixed. Use a fixed seed, **853018101**, the existing `dev`
nonce vocabulary, deterministic alias/argument assignments and bounded values.
Do not use the reserved audit vocabulary or choose examples using model outputs.

Keep semantically invariant cases rather than filtering or resampling them.
The expected inventory is 620 state-changing pairs (220 color, 220 count, 180
switch) and 100 invariant controls (20 color, 20 count, 60 switch). Preparation
must derive and verify these counts from both interpreters, preserving the full
720-cell inventory and all semantic metadata. A disagreement stops preparation.
Never relabel an invariant case as an informative opposite-answer pair.

Every pair is checked by the independent typed and visible-English interpreters
already used for complementary lessons. They must agree on every action and
causal state prefix. Validate exact one-word visible difference, shared questions,
all query targets and changed-role labels; reject malformed pairs, out-of-range
values and context overflow. The seed and cell coordinates identify reproducible
examples. This is new evaluation text within a reused grammar and vocabulary,
not new natural English or an independently sampled external benchmark.

## Observations, raw evidence and metrics

Only packed visible utterances and a BOS reply prefix enter native inference.
Programs, target actions, expected replies, pair coordinates, word labels,
changed-role labels and oracle states remain evaluation metadata. No auxiliary
state or program decoder supplies an answer. Decode replies freely from BOS.
Retain raw four-class logits and generated reply token bytes for all six questions.

The primary descriptive measure is **complete immediate-pair success**: both
immediate questions must have correct action and exact free reply in both
members. Report it separately for each family's state-changing cases and as the
equal-weight mean of those three rates. Also report the full-bank result and
invariance controls with their own denominators. Do not pool away the larger
switch invariance fraction.

Report action-only and reply-only immediate pair results; source/destination
role pairs; opposite-answer changed-role pairs; all-six-question complete pairs;
and slices by definition order, composed/atomic code, initial relationship and
probe reference. Pairwise checkpoint differences use the identical examples and
fixed final states. Provide all three checkpoints, not only the best score.
These correlated factorial cases are not 720 independent natural-language samples.

An analytic policy producing identical predictions for both normalized pair
members has zero complete immediate-pair success on state-changing cases: at
least one shared question has opposite targets. Report that limit as an analytic
check, not as a measured neural ablation. High invariance/unknown accuracy cannot
replace the changed-case primary measure. There is no new eligibility gate,
significance claim or automatic adoption in this diagnostic.

## Local execution and accounting

Preparation publishes immutable canonical JSON, cell inventory, source snapshots,
input pins and a launch record before inference. Record preparation wall/CPU
cost and generation/admission work. Pure synthetic tests may construct fixtures
before the formal bank; they are recorded separately and use no model.

The one declared worker runs exclusively on the local RTX 5080, using the same
strict FP32 runtime as the admitted rate comparison. Its allowance is **600 wall
seconds**, including model/weight loading, bank preparation, inference, recount,
cleanup and durable summary publication. A terminal marker binds the summary
and verifies the inclusive deadline; marker publication is separately reported.
Preserve failed attempts and partial work. Do not restart or enlarge the bound
solely because observation times out.

Load exactly three checkpoint archives and construct exactly three native models.
Verify envelope lineage, full weight layout/ties and expected weight digests.
Do not construct optimizers, restore training owners, train or mutate archives.
Batch size is 32: 45 batches per candidate, **135 native encoder calls**,
**4,320 encoded episode instances**, **51,840 all-turn free-reply contexts** and
**25,920 scored query records** overall. Count decoder recurrent work from actual
calls, including separate BOS work, rather than estimating it from query count.
Record attempts and completions, GPU memory, wall/CPU time, model state before
and after, data admission, archive bytes and all durable score-file pins.
Optimizer updates, backwards, auxiliary calls and teacher/network calls are zero.

The existing home project ownership lock and authoritative process inventory
prevent overlapping learning jobs. No cloud service or download is involved.
After terminal completion, independently verify the saved raw records against
the frozen bank and recount every reported metric without model inference.
Pure tests and independent source review precede the formal worker; no separate
training proof is needed because this diagnostic has no optimizer transition.

## Goal remains broader

This diagnostic improves the choice of the next shared learning intervention.
It does not itself teach English, establish tutor benefit, implement learned
curriculum selection, integrate compact home storage or establish a home compute
ceiling. The operational offline tutor setup and user continuation instructions
remain available; useful native transfer, retention and increasing independence
remain requirements of the active project goal.
