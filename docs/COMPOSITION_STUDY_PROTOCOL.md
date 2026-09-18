# Shared compositions and continual learning: prospective protocol

Prospective specification before source and bank freeze. The preparation manifest
records the realized banks, source hashes and all selected recipes before
optimization. This document does not authorize changing a frozen study after
seeing its results.

The [consolidation comparison](CONSOLIDATION_STUDY_RESULTS.md) established a
useful retention effect from replay and better fitting of the broader bank with
more training. It did not establish arithmetic acquisition: paired answers
remained weak, and aggregate retention concealed losses on individual panels.
This study tests learnability and interference on shared compositions before
introducing an adaptive practice selector or promoting a regional checkpoint.

## One procedure, three value domains

Generate programs through one procedure over typed locations and four operations:
set a value, copy a current value, advance a value, and query equality. The three
families are **color**, **count** and **switch**. Their semantics are:

| Domain | Stored value | Advance |
|---|---|---|
| Color | red, green, blue or yellow | Explicit cyclic successor, including yellow to red |
| Count | Bounded nonnegative integer | Explicit signed increment; reject invalid programs rather than silently clamp |
| Switch | off or on | Toggle |

The text states the applicable operation unambiguously. Copy captures the source
value at that event; changing the source later does not change the destination.
Copying an unknown source makes the destination unknown, and advancing an unknown
value leaves it unknown. Unknown is distinct from false. These semantics apply
equally to every family; family identity must not select a separate narrative
skeleton, event-position recipe or answer head.

Episodes contain 8, 10 or 12 turns. Structure, values and naming have separate
deterministic seeds. Sample irrelevant statements, earlier queries, source and
destination reuse, and query positions independently of family. The final
query's actual causal ancestry must contain both a copy and an advance. Earlier
unknown queries occur at varied nonfinal positions. Record target mixtures by
family, length and query position; enforce the declared unknown coverage during
bank preparation rather than infer it from aggregate accuracy.

Every episode belongs to a complete counterfactual pair. Change exactly one
earlier set fact, retain all remaining observation text and query wording, and
require opposite known final answers. Paired variants therefore balance final
DENY and ALLOW. The final query is always known under this contract: final-turn
uncertainty is **not measured**. Unknown performance concerns the nonfinal probes.
The changing fact and answer must not be encoded in an input row index, variant
identifier, metadata channel or teacher prefix.

The abstract interpreter executes typed events. A separate text interpreter
reads the rendered English and reconstructs its own state. Both must agree on
every target, including earlier queries, before a row is admitted. Rendering
must not pass abstract state to the text interpreter. Only observed text bytes
enter the learner; target labels, replies, dependency signatures, family labels,
split identities and parsed state remain evaluation/training metadata.

## What is withheld

Partition the **actual versioned causal ancestry of the final query**, not the
whole transcript. Trace reads through the value version copied at each event,
discard overwritten and irrelevant facts, and normalize names and constants.
Retain copy/advance order and canonical source/destination role reuse. Altering
distractors, earlier queries or surface aliases must not move the same ancestry
into another partition.

Use disjoint training, development and sealed audit ancestry signatures, with
separate full-program identities for reproducibility. Inspect every supervised
known query, not only the final query: a training program is admissible only when
every query ancestry containing both copy and advance belongs to the training
partition. Shorter primitive ancestries are exempt. Record each query's actual
ancestry identity, partition and composed/known status. This protects supervised
compositions; it does not withhold every unqueried subexpression.
Development programs also reject any known composed query in the audit partition,
including earlier queries. Across every domain and length, development final
composed identities must not overlap any supervised training composition; audit
final identities must not overlap any supervised training or development
composition. Other audit queries can have mixed familiarity, so the final pair
metric supplies the clean held-composition endpoint.
All constituent primitive
operations appear in training. The withheld boundary is a dependency composition,
not a previously unseen primitive or a claim of an unseen algorithm. Distinct
chains may be semantically equivalent, especially even/odd switch advances.
Report both signature and full-program counts; repeated seeds or renderings do
not enlarge the structural count.

Freeze world/value/naming recipes and actual row identities before optimization.
Reject whole pairs when either normalized member leaks across required roles.
Familiar-structure/new-name panels may intentionally reuse their declared
structure, but must be labeled separately from new-ancestry panels. Reusing known
words under new mappings is not unseen-vocabulary learning. Family rotation can
also introduce value vocabulary, such as digits, so report that lexical boundary
instead of attributing every difference to abstract transfer.

Training contains 256 complete pairs per family and length: 4,608 episodes. Each
development and audit group contains two panels for every family/length, with
64 pairs per panel: 2,304 episodes per group. `seen` panels regenerate the first
64 training structural recipes with fresh value and naming seeds. They contain
familiar sampled motifs with fresh renderings; anonymous-world repetition is
allowed, especially in the Boolean domain. `composed` panels request the role's
development or audit structural partition. Exact full observation transcripts
must be globally disjoint across all banks, with whole-pair rejection on overlap.

The seed namespaces are 70,000,000 for training, 80,000,000 for development and
90,000,000 for audit. Add the global family index times 1,000,000 and turn count
times 10,000. Composed evaluation panels additionally add 300,000. Candidate
seeds increase deterministically; naming seeds add 100,000,000 and value seeds
add 200,000,000. Seen panels retain the original training structure seed while
using the new role's candidate seed for names and values. Record every accepted
recipe, exact-transcript rejection, development/audit-query rejection, actual
motif/program/map counts, target counts by turn and UTF-8/context sizes.

## Matched exposure, different order

Use two rotations: **color introduced late** and **count introduced late**. The
other two domains are the old families. Initialize identical learners and use
the same per-family ordered draw streams, complete pairs, bank contents and
optimizer recipe. A single joint learner is shared by both comparisons, because
joint training is identical regardless of the rotation label. Each rotation has
its own sequential learner with replay. The two comparisons therefore share a
baseline and are not independent replications. No adaptive selector is used.

Every optimizer step is the arithmetic mean of three separately computed
32-episode objectives. Retain the same objective, clipping and normalization in
both conditions. This avoids changing loss scale through an extra replay term.
Replay draws count toward the old-family quota rather than add unmatched data.

| Condition | Updates | Three 32-episode microbatches |
|---|---:|---|
| Sequential, old phase | 1,800 | Alternate old families; all three microbatches come from the selected family |
| Sequential, new phase | 1,800 | Two late-family microbatches and one alternating old-family microbatch |
| Joint | 3,600 | One microbatch from each family per step |

Each main run therefore consumes exactly **3,600 microbatches / 115,200 episodes
per family**. An old family receives 2,700 microbatches in the first phase and
900 through replay in the second. The late family receives 3,600 in the second
phase. Total: 3,600 updates and 345,600 episodes per run.
Reconstruct final sampler states and byte/token exposure to
prove the two schedules consumed the same examples, including replay. Do not
claim equal dedicated GPU time: ordering can change padding, batching and runtime.

Use initialization seed **2801**, learning rate **0.001**, and the default
753,610-parameter SequenceStudent with `max_turns=12` and `max_positions=1024`.
Changing the allowed turn count does not add parameters. Preparation must check
every actual encoded episode fits the context limit and report its maximum
length; no observation truncation is permitted. Per-family samplers use seed
**3801** and a fixed global family index independent of which family is late.
Every family has separate 8-, 10- and 12-turn training banks, each containing
256 complete pairs. A microbatch first draws a length bucket uniformly, then
draws 16 complete pairs with replacement from that bucket. Preserve the exact
ordered per-family stream across schedules and the corresponding fresh control.

Joint learning sees the late family from the start. This comparison isolates
ordering under matched overall exposure; it is not an equal-pretraining few-shot
transfer comparison. Compare curves by both optimizer updates and cumulative
family exposures, with that availability difference explicit.

For each rotation, include one fresh late-family-only control: 1,200 optimizer
steps, three late-family microbatches per step, the same 3,600-microbatch stream.
This matches late-family exposure but not total computation, old-family exposure
or the number of optimizer updates during the sequential new phase. It is a
learnability reference, not an independently matched causal estimate of transfer.
There is no separate post-study adaptation sweep.

Measure development at the following matched late-family exposure budgets;
preserve the corresponding checkpoints for the final sealed evaluation:

| Late-family episodes | Joint updates | Sequential new-phase updates | Fresh updates |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 384 | 12 | 6 | 4 |
| 1,536 | 48 | 24 | 16 |
| 6,144 | 192 | 96 | 64 |
| 24,576 | 768 | 384 | 256 |
| 115,200 | 3,600 | 1,800 | 1,200 |

Sequential total updates add 1,800 to the new-phase column. Its zero-exposure
checkpoint already contains old-family learning, unlike joint/fresh initialization.
Curve areas therefore combine starting competence with subsequent acquisition;
they do not isolate a learned optimization speed.

Initial scope is one initialization, two rotations, three main runs and two fresh
controls: five physical runs named `joint`, `seq-color`, `seq-count`, `fresh-color`
and `fresh-count`. The planned total is **13,200 optimizer updates and 1,267,200 repeated
episode exposures**. Replication requires a new declared seed cohort; one-seed
differences are descriptive evidence.

A disposable local execution probe used seed 2800 for two updates (192 episode
exposures) on separate tiny training banks. Its weights are discarded, and it
does not select a learning rate or checkpoint. Record it separately from the
five-run study budget; its short timing is not a throughput or capability claim.

## Evaluation and integrity

Development data may measure progress but may not select a more favorable final
checkpoint. Preserve the sequential checkpoint at update 1,800 and every fixed
endpoint. After all main runs and controls finish, evaluate the sealed audit on
the saved before/after checkpoints to measure old-family retention. Do not inspect
intermediate audit results while other study endpoints remain unfinished.

Report each family, episode length and structural/naming panel separately:

- Action accuracy with confusion matrices and explicit known/unknown denominators.
- Final opposite-answer pair accuracy, with complete pair counts.
- Earlier and later query positions, including known later pairs where eligible.
- Free reply correctness and action/reply agreement, without teacher prefixes.
- Probability error and unknown handling; absent denominators remain absent.
- Retention changes per panel, including the largest regressions concealed by a macro.

Blank-text and reset-history controls retain original canonical targets. Reset
history presents each original utterance alone. Verify exact CPU serialization
and restart behavior, unchanged checkpoint files during scoring, and absence of
teacher labels or interpreter state from model inputs. Authenticated immutable
bank caching is allowed only if included in the frozen implementation and tested
for equivalence and mutation isolation; it cannot waive canonical admission.

Freeze current source hashes, copied source hashes, bank content/order, model
configuration, initial weight digest and expected exposure streams. Resume must
preserve optimizer moments, update counters and independent per-family samplers.
Report training, validation, packing, inference and generation timing separately
where measured. Do not infer dedicated GPU-hours from concurrent worker intervals.

## Decision boundary

No automatic promotion or numerical winner is predeclared. A useful result must
show paired dependence on the changed fact and preserve earlier abilities;
aggregate accuracy or correct uncertainty alone is insufficient. If joint
learning succeeds while sequential replay fails, investigate interference across
shared mechanisms. If both fail despite adequate training fit, investigate
composition coverage and shared representation before giving an adaptive
controller more freedom. If both acquire and retain, replicate across seeds
before testing development-driven allocation against the fixed schedule.

This remains a diagnostic local sequence learner. Regional integration and
progressive independence from an external teacher are separate capabilities to
validate. The curriculum concerns inert simulated objects, with exact beneficial
learning/retention objectives and no real-world actions or harmful evolutionary
selection. Fresh rows do not erase the fact that earlier audits informed this
study's design.
