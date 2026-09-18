# Shared decision teaching: completed local results

The training-only decision head **failed its predeclared benefit screen**. No
checkpoint was promoted and confirmation seeds were not triggered. The original
motor policy still failed later decisions requiring changes to remembered state.
Both trained learners reached the same final graph score as fresh initialization,
with zero complete opposite-answer pairs. The broad home-learning objective
remains open; this closes one mechanism comparison.

The [prospective protocol](COGNITIVE_CREDIT_PROTOCOL.md),
full audit (archive reference: `../runs/general-credit-local/audit-2301/report.json`),
exact summary (archive reference: `../runs/general-credit-local/audit-2301/summary.json`) and
decision record (archive reference: `../runs/general-credit-local/decision.json`) preserve the evidence.
All reported percentages below concern decision actions unless explicitly labeled
as generated replies. Macro scores give each family, or each graph level, equal
weight. A correct pair requires both opposite answers at the same question
position; it does not mean both complete conversations were solved.

## Fixed comparison and compute

Both 796,552-parameter models began at seed 2301 with matching weights and
complete-pair sample streams. They differed only in the training-only auxiliary
decision-loss weight, 0 versus 0.3. Official actions, generated replies and state
updates used the existing regional pathway. Binding, arithmetic and conditional
logic at levels 1/2 each received 600 updates: 1,800 batch-64 updates and 115,200
repeated episode exposures per arm. Graphs were absent from this main training.
Fixed endpoints were used; the audit did not select a favorable earlier model.

The two main workers shared one RTX 5080. Their training intervals summed to
644.346 seconds; the longer worker's total interval was 331.188 seconds. These
are worker measurements, not dedicated GPU-hours or a measured whole-study
wall-clock duration. Isolated copies of both endpoints and a fresh initialization
then received 16 graph updates each, totaling 48 updates, 3,072 repeated support
exposures and 8.329 training seconds. The 64 graph support episodes, sample
stream and zero auxiliary weight were identical across adaptation candidates.

The audit was prepared before both initial main checkpoints. It verified source,
bank, initialization, exposure and endpoint constraints. The original checkpoints
were unchanged; exact CPU conversation continuation passed for all three audited
candidates. These checks establish experiment integrity, not capability.

## Main development decisions

| Family | Control query | Auxiliary query | Control pairs | Auxiliary pairs | Control later known | Auxiliary later known |
|---|---:|---:|---:|---:|---:|---:|
| Variable binding | 54.95% | 57.03% | 0/71 | 0/71 | 40/128 (31.25%) | 48/128 (37.50%) |
| Arithmetic updates | 66.67% | 66.67% | 0/128 | 0/128 | 64/128 (50.00%) | 64/128 (50.00%) |
| Conditional logic | 62.50% | 37.50% | 64/73 (87.67%) | 0/73 | 0/96 | 0/96 |
| Equal-family mean | 61.37% | 53.73% | 29.22% | 0.00% | 27.08% | 29.17% |

“Later known” selects an episode's last yes/no question only when its ordinal
among all questions is at least two. This excludes episodes whose only known
question is the first. On that slice, **both arms solved zero complete opposite
pairs in every trained family**: 0/7 binding, 0/64 arithmetic and 0/9 conditional.
The auxiliary arm's binding gain was 6.25 points, but the equal-family later-known
gain was only 2.08 points and did not extend to a second subject.

The apparently strong control conditional pair score came entirely from the
initial questions. Their additional switches are neutral, so those answers only
require the first switch. Both arms answered ASK to all 96 later known
conditional questions, including the intervention outcomes, and got none right.
Both answered DENY to all 128 later known arithmetic questions, obtaining 50%
individual accuracy but no paired correctness. A question-position baseline
already obtains 66.67% arithmetic accuracy on this curriculum. These scores do
not establish composition or successful state updating.

The fixed training-fit diagnostic also showed weak learning on already available
lessons. Control/auxiliary binding query accuracy was 47.27%/49.09%, arithmetic
66.67%/66.67%, and conditional 62.50%/37.50%. Binding and arithmetic paired
accuracy remained zero. This warrants testing a stronger common representation
and learning procedure; it does not isolate a proven causal bottleneck.

Generated replies and uncertainty remained imperfect. On main development,
control/auxiliary exact query-reply accuracy was 43.49%/53.39% for binding,
66.67%/66.67% for arithmetic and 44.14%/44.14% for conditional logic. Conditional
ASK precision was 25% with 100% recall for both arms: asking on known questions
was a substantial failure. Better action/reply agreement alone would not resolve
wrong decisions. The audit retains class scores, Brier scores, reply agreement,
query positions and their denominators.

## Learning the withheld graph subject

| Graph support updates | Control query | Auxiliary query | Fresh query |
|---|---:|---:|---:|
| 0 | 45.02% | 37.40% | 46.68% |
| 1 | 45.02% | 38.67% | 25.00% |
| 4 | 45.41% | 44.73% | 52.34% |
| 16 | 53.32% | 53.32% | 53.32% |
| Normalized learning-curve area | 48.32% | 46.96% | 49.11% |

Relative to fresh initialization, curve area was **0.80 points lower** for the
control and **2.15 points lower** for the auxiliary learner. Prior training
therefore supplied no measured graph acquisition benefit in this experiment.
The endpoint was 54.30% query accuracy at level 2 and 52.34% at level 3 for all
three candidates. Each had 128/256 correct later known questions at each level,
but **0/128 correct opposite-answer pairs at each level**. The fresh learner
briefly solved one level-3 pair at update 4; that did not persist.

Blanking all English at the endpoint yielded 28.32%/52.93%/0.00% query accuracy
for control/auxiliary/fresh. Resetting state between turns yielded
28.32%/28.32%/46.68%. All these controls had zero paired accuracy. The auxiliary
learner's nearly unchanged blank-text score illustrates why raw query accuracy
cannot substitute for contextual paired success. These controls preserve the
original labels and do not count as new independent task draws.

## Retention and advanced operations

| Equal-family measure | Control before graph learning | Control after | Auxiliary before | Auxiliary after |
|---|---:|---:|---:|---:|
| Retention query accuracy | 58.59% | 51.56% | 50.61% | 24.83% |
| Retention paired accuracy | 28.07% | 5.26% | 0.00% | 0.00% |
| Advanced query accuracy | 51.04% | 40.76% | 42.71% | 20.70% |
| Advanced paired accuracy | 24.24% | 7.20% | 0.00% | 0.00% |

Retention query accuracy fell by 7.03 points for the control and 25.78 points for
the auxiliary learner. Neither preserved earlier behavior through graph learning.
Later-known individual scores sometimes rose while later-known paired scores
stayed zero; changes in preferred answer class cannot establish retained
reasoning. Advanced level 3 adds copying in binding/arithmetic and three-input
conditional rules. It includes new operations and language, so it is not a pure
depth-extrapolation test.

The auxiliary arm failed the required main paired gains, cross-family later-known
gains, conditional query preservation, and multiple before/after retention
requirements. Its graph non-regression checks passed relative to the weak control;
this does not outweigh the other failures or show an advantage over fresh.

## Limits and next decision

This was one initialization, one withheld-family rotation and one graph support
draw. Many questions are correlated within finite six-turn templates; episode
counts are not independent experimental replications. The curriculum shares
words and incidental entity pairs across splits, despite disjoint primary pairs
and exact-transcript checks. Its oracle validates meaning and labels, not
curriculum completeness or freedom from every heuristic. These inspected audits
are now known diagnostics; changed learners or curricula need prospective new
evaluation evidence. No external teacher was consulted for policy answers.

The next comparison tests the common sequence representation before expanding
the same failed objective. A 753,610-parameter causal sequence baseline, raw-text
packer, separate reply decoder, matched objective/trainer and teacher-free
evaluator are implemented. A short local benchmark (archive reference: `../runs/sequence-baseline-local/benchmark-initialized.json`)
measured 64.17 sequence versus 8.44 regional updates/second at batch 64, with
578.44 versus 205.54 MiB peak allocated CUDA memory. That approximately 7.60-fold
throughput difference is **not learning efficiency**: the probe ran only twelve
FP32 optimizer steps after three warmups per model and scored no capability.
Its binding/arithmetic/conditional mixture differs from the new formal study.
The original probe, before corrected embedding initialization, is preserved
separately as historical evidence.
No sequence capability study or checkpoint promotion is established by it.

The [sequence design](SEQUENCE_BASELINE_DESIGN.md) requires a new matched learning
protocol with declared development budgets, exposure and elapsed-time views,
retention, fresh-init adaptation and context controls. This diagnostic model
bypasses regional routing; success would need separate regional integration
evidence. The [home-learning roadmap](HOME_LEARNING_ROADMAP.md) still includes
broader curricula, consolidation, self-direction, declining external-teacher
dependence and a usable saved learner. A failed local screen closes neither
those obligations nor the user's larger objective.
