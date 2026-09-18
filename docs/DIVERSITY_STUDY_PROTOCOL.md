# General curriculum diversity: prospective local comparison

This next experiment tests a shared curriculum mechanism across binding,
directed paths and conditional rules. The diagnostic causal sequence learner,
its objectives and optimizer remain unchanged. Arithmetic is withheld from main
optimization and development scoring, then learned in a separate adaptation
phase. The previously completed studies remain intact. This is a prospective
one-seed development comparison, not a capability claim or regional BiC release.

The curriculum has a new version. It constructs anonymous event programs,
evaluates them using an abstract interpreter, renders raw English through a
bijective naming map, and independently checks the rendered text using the
existing reviewed interpreter. Only observed English enters the student.
Reference answers, world programs, partitions, seeds and family identifiers
remain outside inference context. The generated worlds are six-turn restricted
lessons with four response actions; they do not represent general English.

## Four matched conditions

| Arm | Distinct anonymous world pairs per family | Naming maps |
|---|---:|---:|
| w32-n1 | 32 | 1 |
| w32-n8 | 32 | 8 |
| w256-n1 | 256 | 1 |
| w256-n8 | 256 | 8 |

World pools alternate levels 1 and 2, with distinct anonymous fingerprints.
Naming randomness is independent of world randomness. Every arm contains
2,048 logical pair slots per family. Slot `i` chooses world
`(i mod 256) mod world_count` and naming map
`floor(i / 256) mod naming_count`. Low-diversity arms therefore intentionally
repeat examples. All arms use the same pair-slot draw stream. Both variants
of each counterfactual pair always share one naming map, questions and event
order; at least one known question changes answer.

The two factors are **finite bank diversity**, not continual world refresh.
The preparation record counts actual unique transcripts, anonymous pair IDs,
complete alias mappings, target classes and targets by turn. Distinct naming
seeds alone are not evidence of distinct mappings. Anonymous program identity
is syntactic, not a proof of semantic equivalence or independent samples.
Larger world pools may change class and operation frequencies. This is a test
of the declared diversity treatment, not an isolated causal claim about
semantic novelty.

Each fresh arm starts from seed **2601**, using the unchanged 753,610-parameter
sequence model. AdamW uses learning rate **0.001**, previously selected in the
architecture study; there is no new learning-rate search. Each arm receives
**3,600 updates**, batch size **64**, with a fixed interleaved schedule across
the three families: 1,200 updates and 76,800 repeated episode exposures per
family. Objectives remain balanced query cross-entropy, 0.25 acknowledgement,
0.1 per-turn reply-byte and 0.1 per-turn observation-byte losses. No external
teacher or pretrained weights participate.

All four endpoints are fixed at 3,600 updates. Development scoring at every
600 updates is descriptive; it does not choose a checkpoint or change practice.
Training-fit scoring uses the same first 32 familiar worlds under the first
naming map in every arm. Their exposure frequencies differ as declared above.

## Frozen partitions and separate generalization panels

Anonymous pair fingerprints determine world partitions. Ordered primary name
pairs have separate training/development/audit partitions. Training admission
requires **both** partitions and the row's admission role to be training.
Evaluation can cross them explicitly:

- **Names:** the first 32 training worlds, known to every arm, each rendered
  twice with naming combinations withheld from training.
- **Worlds:** 64 unfamiliar anonymous worlds per family under the first
  familiar training map.
- **Both:** the same 64 unfamiliar worlds with withheld naming combinations.

These are 128 episodes per family and panel. Development and later audit use
different world/name partitions. Level-3 audit uses another 64 distinct worlds
per trained family. Level numbers describe generator difficulty and may change
operations as well as depth. Paired fingerprints alone do not prevent one
member appearing with different counterfactual partners. Whole candidate pairs
are therefore rejected when either individual alias-normalized transcript
overlaps an earlier world pool. Selection proceeds training, development,
retained, advanced; the naming panel deliberately reuses its declared familiar
worlds. Arithmetic query pools also exclude support and preceding query
variants. These exclusions and their counts are frozen before training. Exact
rendered transcript overlap is additionally rejected across training and
evaluation. Semantic equivalence beyond consistent renaming can still overlap.
Incidental aliases and pairs can still overlap; primary-pair holdouts do not
establish wholly unfamiliar vocabulary.

All bank rows, source files and identities are frozen before optimization.
Preparation verifies training/evaluation transcript separation, paired truth,
mapping counts and actual diversity. Main workers never optimize or evaluate
arithmetic. Its procedural grammar is already known to the project, so this
is a within-run subject holdout, not an untouched scientific benchmark.

## Adaptation and retention audit

The audit opens only after all four intact fixed endpoints exist. It evaluates
all four and one common fresh seed-2601 control; no winner is selected first.
Each receives a fresh AdamW optimizer at 0.001, sampling seed **3601**, and the
same 64 level-1/2 arithmetic support worlds (128 episodes) from the training
partition. Cumulative budgets are **0, 1, 4, 16 and 64** updates of 64 episodes.
No replay is added in this comparison.

At every budget, score independent arithmetic levels 2 and 3 (128 world pairs,
256 episodes each). Score old-subject audit panels and level-3 worlds before
and after adaptation. At the endpoint, blank-observation and reset-history
controls measure dependence on available context. A serialized CPU restart
must reproduce actions and generated replies exactly. Parent checkpoints must
remain unchanged.

Report per-family/panel query accuracy, opposite-answer pairs, later known
decisions, uncertainty behavior, calibration, freely generated replies and
answer/action agreement. Report query, pair and later-known adaptation curve
areas relative to the common fresh control, alongside retention losses. Sparse
trapezoidal areas are descriptive; ordinary gradient adaptation is not evidence
of a trained meta-learning algorithm. Aggregate scores must not hide failures
on individual subjects or later questions.

The main budget is **14,400 updates / 921,600 episode exposures**. Adaptation
adds **320 updates / 20,480 exposures**, for **14,720 updates / 942,080 repeated
episode exposures**. Shared-GPU worker times are not dedicated GPU-hours. Record
measured worker training time, orchestration times and memory separately.

There is no automatic promotion or retrospective numerical pass/fail gate.
Use the results to choose broader curriculum, representation, output-alignment
or retention work. Repeat a promising mechanism across seeds and subject
rotations before making a robust capability claim. The objective remains a
useful local learner that directs practice, retains knowledge and becomes
less dependent on external tutoring.
