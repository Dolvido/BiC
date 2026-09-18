# Shared foundations before longer reasoning chains

The generator, prescribed-order trainer, training-evidence preparation and
evaluation-bank builder are implemented. The matched pilot is complete; its
[results](FOUNDATION_PILOT_RESULTS.md) show weak primitive acquisition in both
orders and poor, unstable paired replies. The completed [capacity study](CAPACITY_STUDY_RESULTS.md) supplied only
development evidence to a prospective selection rule, choosing width **192** and
learning rate **0.001** for both fresh learners. Every candidate's worst-cell
paired score was zero; this configuration choice promotes no trained checkpoint.
Strict local GPU continuation proofs passed. An initial preparation failure on
historical transcript collisions was diagnosed and resolved through a recorded
names-only admission amendment. Both revised arms trained and passed verification;
the original failure remains preserved below.

The hypothesis is that systematic practice on reusable state operations may
help a shared learner acquire longer combinations. Existing composition recipes
require both copy and advance in the final causal chain. Earlier simple
questions provide some foundation practice, but their distribution is incidental.
The new generator makes that teaching progression explicit across every domain.
This is a candidate route toward general learning, not evidence of general
intellect or proof that curriculum is the only bottleneck.
The intended benefit is accurate, reusable and retained skills at measured local
cost. These symbolic state worlds make no claim about BiC's subjective experience.

## One mechanism across domains

[Foundation recipes](../experiments/foundation_curriculum.py) reuse the unchanged
English grammar and independent abstract and English interpreters. They teach
setting, copying, updating and querying state in the color, count and switch
worlds. Causal depth counts copy/update transformations in the final value's
dependency chain. Depth zero teaches a direct fact; depth one teaches a single
transformation; depths two through five combine copying and updating.

Every depth fits each 8-, 10- and 12-turn context bucket. Disjoint distractors
vary surrounding context without lengthening the causal chain. These particular
distractors do not test interfering overwrites. Each complete pair changes one
earlier fact while keeping the final question identical and reversing its known
answer. Both independent interpreters must agree with all supplied targets.

Compact recipes hold procedure, naming and value seeds; exact regeneration
authenticates observations, targets and metadata. Packing uses the existing
observation-only byte path with an explicit foundation pair validator. Depth,
parsed state, recipes and answers never enter model observations. The causal
sequence model and learning objective need no change for this data format.

[FoundationTrainer](../experiments/foundation_training.py) consumes one immutable
bundle per optimizer update: three independent domain microbatches, equal thirds
of the existing objective, then one clipped AdamW update. Its checkpoint includes
weights, optimizer, cursor, source/recipe identities, consumed-input hashes and
actual exposure counts. Restore regenerates the consumed prefix to authenticate
inputs; it does not reproduce earlier gradient arithmetic. Failed steps require
restoration. An owning runner must provide atomic saves and account for discarded
physical work. This trainer executes a prescribed plan, not adaptive allocation.

## Preserve what an unseen test actually means

Composed ancestry identities retain their original hashes and partitions.
Every supervised training composition must belong to the legacy training
partition; development cannot expose a legacy audit composition, even in an
earlier question. Primitive anchors are explicitly shared and do not count as
withheld structure. Their metadata distinguishes shared admission from their
legacy hash partition.

At depth two, the two mixed-operation motifs are copy then update (legacy train)
and update then copy (legacy audit). There is no distinct legacy development
motif at this depth. An unavailable request fails; familiar-pattern development
probes must explicitly request the training structural partition. New names and
values can test realization transfer without being called new procedures.

The grammar and value domains are finite. Switch parity and the color cycle can
simplify long chains, so unseen syntactic ancestry does not mean a new algorithm.
The generator does not read old study banks. New bank preparation must separately
exclude historical transcripts and reserve all new evaluation data; a legacy
training-partition motif used in an earlier familiar-pattern audit is not made
pristine again by a new recipe version.

[Training evidence](../experiments/foundation_evidence.py) materializes the plan
before optimization, rejects exact transcript collisions, records actual
domain/depth/operator/length and byte/label mixtures, and retains compact
references to admitted anchors. The [evaluation builder](../experiments/foundation_banks.py)
authenticates those references and the complete training-transcript list. Its
fresh panel regenerates the same admitted procedure with changed visible names
and anonymous value-event transcripts, preserving every query ancestry. Depth-one
copy and advance are separate cells. Held-composition panels use legacy dev
depths three through five or audit depths two through five. All panels cover all
three domains and lengths. Complete pairs are rejected on overlap with training,
caller exclusions, or any other bank produced in the call; bounded retries never
fall back to another split or mechanism. Historical exclusions and source/data
authentication remain the runner's responsibility. Anonymous transcript identity
is syntactic and does not establish semantic novelty.

## Compare order while holding the lessons fixed

The [plan builder](../experiments/foundation_plan.py) produces two schedules over
the identical set of complete three-domain microbatch bundles. One schedule
progresses from shallow to deeper stages with earlier-depth rehearsal. The other
shuffles that prefix. Both finish with the exact same mixed-practice suffix.
Length rotates within each depth, rather than being a fixed label for depth.

Each bundle's procedure, names and values depend on its immutable ID, not when
it is consumed. Reordering therefore cannot change realizations. Bundle IDs do
not guarantee unique English text: evidence preparation measures collisions,
exact byte/label/depth mixtures and reserved-data overlap before a study starts.
The builder's default small plan remains an API example.

The proposed pilot uses **192 updates in each of six stages**, followed by an
identical **384-update mixed suffix**: **1,536 updates per arm**. With 32 episodes
per domain microbatch and three domains per update, each arm consumes **147,456
episode exposures**; together they consume 3,072 updates and 294,912 exposures.
The curriculum arm advances through depths zero to five with earlier-depth
rehearsal. The mixed arm shuffles the same prefix bundles. Both start afresh from
the same initialization and use the selected width-192, four-layer, four-head
model with feedforward width 768 and rate 0.001. No capacity checkpoint supplies trained weights.
The lesson multiset, objective and exposures match; elapsed time need not match.

This comparison tests teaching order within a curriculum containing foundation
practice. It does not isolate the benefit of adding foundation data relative to
composition-only practice. Earlier study checkpoints are historical context,
not matched control arms. One shared initialization makes this a descriptive
pilot. Before launch, the runner must bind the admitted banks, source closure,
budgets and exclusions, pass a matching strict local runtime/continuation check,
and preserve preparation, training, scoring and discarded-work costs separately.

[FoundationBank](../experiments/foundation_evaluation.py) authenticates one
family/depth/length cell, then reuses the existing scorer, paired metrics,
BOS-only free replies, and blank/reset-history controls. Measure primitive
acquisition and retention, fresh name/value transfer, known answers and
unsupported ASK, and withheld composed decisions separately. Repeat promising
results across seeds before capability claims. The new data and evaluator are
not yet connected to the continuing English controller or an external tutor;
there is no model promotion. The completed study
can inform teaching choices for a broader learner; completing these finite
families does not complete the self-directed home-learning objective.

The earlier generator, plan and evaluator validation passed **13, 9 and 8 focused
CPU tests**, respectively. The repository at that point passed **743 tests in 117.045 seconds**,
with the running capacity study's 71 source hashes unchanged.
Integration evidence (archive reference: `../runs/english-loop-validation-local/integration.json`)
These tests check curriculum truth, admission, matched recipes and evaluation
boundaries; they do not validate the new pilot's GPU execution or measure learning
benefit. Expanded trainer/evidence/bank checks and the pilot runtime receipt must
be reported separately.

## Implemented pilot and execution evidence

The [pilot runner](../experiments/foundation_study.py) and
[result pipeline](../experiments/foundation_results.py) now implement preparation,
two one-shot training arms, independent CPU replay/state verification, and later
scoring. The runner requires all lessons and evaluation banks, historical exclusions,
initial weights, sources, budgets and execution proof to be bound before training. Training saves
each official stage boundary and records per-step physical work; failures remain
visible and are not silently retried. Both completed endpoints precede scoring.
Development curves include all stage boundaries; fitting, audit and blank/reset
controls use the endpoints. The continuing controller and external tutor remain
separate integration work. The first formal preparation attempt stopped before
gradients and remains preserved. Diagnosis (archive reference: `../runs/foundation-admission-diagnosis-local/report.json`)
found three complete pairs already present in the historical training stream,
with no within-plan transcript repeats. The admission amendment (archive reference: `../runs/foundation-planning-local/admission-amendment.json`)
adds a bounded, deterministic names-only repair before source/data freeze.
It preserves typed programs, values, all query ancestries, targets, replies and
teaching schedules; both arms use the same admitted plan. The original plan and
amended plan retain separate hashes. All rejected attempts are recorded, and
reserved observations are never removed. The revised admission passed: three
pairs changed through four retry candidates, with all 147,456 accepted episode
transcripts unique. Preparation took 757.609 seconds. Both matched arms completed
1,536 updates under a 79-source frozen protocol (archive reference: `../runs/foundation-study-local-v2/protocol.json`).
Canonical CPU verification authenticated all 16 saved states and lesson streams
in 2,859.985 seconds. Evaluation followed in 171.281 seconds. The
[count-backed report](FOUNDATION_PILOT_RESULTS.md) preserves fitting, fresh/held
transfer, all stage boundaries, retention references and controls. No checkpoint
was promoted; the selected next hypothesis changes one shared representation.

All 101 foundation checks passed in the latest full repository run. That run
contained 813 passes and one Windows connection-reset error in a preexisting UI
test; a single isolated rerun passed all six UI tests. These separate outcomes
are preserved in the [checkpoint notes](LOCAL_LEARNING_CHECKPOINT.md).

Strict local GPU probes passed on the earlier source revision at all three supported widths (archive reference: `../runs/foundation-runtime-validation-local/coverage.json`):
96, 192 and 256. Each passed exact initialization, midpoint, reload, continuation,
independent repeat and consumed-input comparisons. Four separate workers per
width performed **144 updates and 13824 episode exposures** in total, outside the
learning pilot. Peak allocated CUDA memory ranged from about 970 to 1493 MiB per
worker. Widths 192 and 256 were exercised in parallel after the capacity GPU
pipeline finished. These checks cover the declared 16-update mixed workload at
rate .001 on the recorded local runtime; another configuration needs a matching
proof. Parent execution records beside the coverage file separate process
observations from nested worker/step timings. This runtime proof is not a
learning-benefit result. The amended admission path requires a new selected-width
proof that actually trains overridden pairs before and after a restart; earlier
proof directories and their complete source snapshots remain unchanged. The new
width-192 admission proof (archive reference: `../runs/foundation-runtime-validation-local-v2/w192/probe.json`)
passed all exact comparisons after 48 separate updates/4,608 episodes. Sixteen
executed bundles contained renamed color pairs, eight before and eight after the
restart midpoint. This is distinct from the old 144-update proof set.

A later efficiency candidate is a shared immutable store of admitted, packed
lesson bundles. The current live path repeats canonical generation/validation;
generation timing does not include all packing work. Measure that cost before
changing the path, and require exact input, gradient, optimizer and continuation
equivalence. Cache construction must be charged once as shared work, with each
learner retaining its own schedule and optimizer. A separate opt-in
[recipe cache](../experiments/foundation_recipe_cache.py) now caches canonical
rows and reconstructs anchors with bounded storage, full plan/source identity
and isolated copies. Nine correctness tests passed. It is not wired into this
pilot, caches no tensors or learner state, and its counterbalanced benchmark has
not run; no speed or learning benefit is claimed.

The separate [practice-provider integration](FOUNDATION_LOOP_INTEGRATION.md)
preserves the actual curriculum identity for future continuing loops. The
[interpretation guide](FOUNDATION_PILOT_INTERPRETATION.md) was written before any
pilot scores were read and keeps subsequent choices at the level of shared
teaching, representation or retention mechanisms.
