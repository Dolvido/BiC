# General learning: completed local cycle, September 16, 2026

Development now measures reusable learning across subjects, instead of extending
the earlier sequence of English-specific repairs. The first completed comparison
does **not** establish a broadly better learner. Mixed practice produced a small
new-subject learning benefit in one recurrent model, but the benefit did not
extend to reasoning after interventions and came with substantial forgetting.
The added episodic-attention mechanism failed its prospective benefit screen in
both curriculum orders. No checkpoint was promoted.

The next research priority is a shared improvement to how decision feedback
trains the reasoning pathway, measured across the entire curriculum and against
retention. Additional hours on this unchanged memory configuration are not the
selected next step. This is a hypothesis to test, not an identified causal cure.

## Implemented scope

The new experimental learner accepts raw English bytes, carries caller-owned
regional activity across turns, and optionally retrieves learned token
representations from up to eight utterances. Two shared attention hops feed the
existing hippocampal pathway. Both memory modes have **796,292 parameters** and
identical initial weights; the control retrieves only the current utterance but
still carries ordinary regional state. The response remains routed through BiC's
regional brain. No parsed facts, answer labels, task identity, or teacher output
enter the inference memory.

The compact curriculum covers variable binding, graph reachability, arithmetic
updates, and conditional logic. Six-turn lessons are reconstructed from version,
seed, family, split, and level. An independent interpreter verifies their truth
before training. Numeric observations are zero. The
[curriculum inventory](COGNITIVE_CURRICULUM.md) documents 5,184 canonical episodes,
5,166 distinct transcripts, complete opposite-premise pairs, and the finite
grammar's limits. All prepared bank groups had zero exact transcript overlap.

Training and audit runners freeze sources, banks, sample streams, budgets and
initialization before optimization. They preserve model, optimizer and sampler
state, refuse changed recipes, and keep adapted copies separate. Final evaluation
uses no LLM or external service. All computation in this cycle was local.

This experiment fixes curriculum order to isolate its effects. It is not a
learned research director or an integrated replacement for the earlier
self-directed symbolic loop. Those loops and all released checkpoints remain
separate. The newly implemented general learner's competence must improve before
automatic curriculum allocation or evolutionary selection can be justified by
its scores.

## Frozen comparison

The [prospective protocol](GENERAL_LEARNING_PROTOCOL.md) trained binding, graphs
and arithmetic at levels 1/2. Conditional logic was entirely withheld from main
training. Four workers shared the RTX 5080: episodic/recurrent memory crossed
with interleaved/blocked subject order. Every arm used seed 2201, 1,800 updates,
batch 64 and the same 600 minibatches per subject. The actual within-family
sample fingerprints match across arms.

All arms used the same query classification, acknowledgement, reply-byte, and
causal input-next-byte objectives. No subject-specific semantic head was added.
Each trained on 3,072 generated episodes, containing 3,064 distinct transcripts,
with repeated sampling. No final-bank score selected a checkpoint or changed
training. All four runs reached the planned endpoint before the audit began.

Development endpoint scores are equal-family averages. Paired accuracy requires
both opposite answers at the same query position; it is not conversation success.

| Memory / order | Query accuracy | Paired accuracy | Binding query | Graph query | Arithmetic query |
|---|---:|---:|---:|---:|---:|
| Episodic / interleaved | 57.99% | 0.00% | 51.82% | 55.47% | 66.67% |
| Episodic / blocked | 42.45% | 0.00% | 30.99% | 29.69% | 66.67% |
| Recurrent / interleaved | 55.82% | 3.39% | 41.93% | 55.47% | 70.05% |
| Recurrent / blocked | 50.65% | 0.52% | 37.24% | 48.05% | 66.67% |

Interleaving improved endpoint macro query accuracy by 15.54 points in the
episodic model and 5.16 points in the recurrent model. This single initialization
does not establish a universal curriculum advantage. Query averages also hide
weak dependence on the relevant premise: all four binding pair scores are zero;
the recurrent/interleaved arithmetic pair score is only 10.16%.

The full checkpoint-by-family matrix is saved in every run's report. In the
episodic blocked arm, binding and graph query accuracy fell 22.40 and 25.78 points
from their best earlier development checkpoints. Recurrent interleaving reached
20.31% arithmetic paired accuracy at updates 900/1,200, fell to zero at 1,500,
and ended at 10.16%. A favorable intermediate checkpoint is not the endpoint.

## Learning the withheld subject

Each endpoint and two fresh initialization controls received the same 64
conditional-logic support episodes, with a new AdamW optimizer and matched
sample stream. Evaluations occurred after 0, 1, 4 and 16 updates; the last point
therefore used 1,024 repeated support exposures. The final query banks each
contain 256 episodes, 512 questions, and primary alias compositions disjoint
from training support. Level 3 adds a third logical input.

| Candidate | Query at 0 | At 1 | At 4 | At 16 | Paired at 16 | Normalized adaptation area |
|---|---:|---:|---:|---:|---:|---:|
| Episodic / interleaved | 18.55% | 18.55% | 43.55% | 37.50% | 0.00% | 37.38% |
| Episodic / blocked | 12.50% | 37.50% | 37.50% | 37.50% | 0.00% | 36.72% |
| Recurrent / interleaved | 45.61% | 43.95% | 37.40% | 62.50% | 73.26% | 47.89% |
| Recurrent / blocked | 18.95% | 37.50% | 37.50% | 32.81% | 0.00% | 35.16% |
| Fresh episodic | 43.55% | 43.95% | 50.20% | 37.50% | 0.00% | 44.45% |
| Fresh recurrent | 43.55% | 43.95% | 37.50% | 37.50% | 0.00% | 38.49% |

The sparse area is a trapezoidal summary, with most weight on the interval from
4 to 16 updates. It is not a densely measured learning-speed curve. Prior mixed
practice improved the recurrent arm's area by 9.39 points over its fresh
control, while the other three trained candidates did worse than their
corresponding fresh controls. This limited signal merits explanation, not
promotion.

| Candidate | Level-2 query / pair | Level-3 query / pair | Generated reply exact, L2 / L3 |
|---|---:|---:|---:|
| Episodic / interleaved | 37.50% / 0.00% | 37.50% / 0.00% | 44.14% / 43.75% |
| Episodic / blocked | 37.50% / 0.00% | 37.50% / 0.00% | 44.14% / 43.75% |
| Recurrent / interleaved | 62.50% / 70.33% | 62.50% / 76.19% | 70.31% / 72.66% |
| Recurrent / blocked | 32.23% / 0.00% | 33.40% / 0.00% | 42.19% / 41.02% |
| Fresh episodic | 37.50% / 0.00% | 37.50% / 0.00% | 25.00% / 25.00% |
| Fresh recurrent | 37.50% / 0.00% | 37.50% / 0.00% | 0.00% / 0.00% |

**The favorable paired score does not demonstrate multi-input intervention
reasoning.** A saved, explicitly post-hoc
query-position breakdown (archive reference: `../runs/general-learning-local/audit/query-position-breakdown.json`)
shows that the recurrent/interleaved model's decision head answered all 256
initial-state questions correctly in each level, then predicted ASK for all
256 final questions. Its known post-intervention decision accuracy was
**0/192 in each level**.
The remaining 64 final questions concerned an undefined lamp and correctly
received ASK. All 128 correct counterfactual pairs per level came from the
initial question; it solved zero of 54/40 eligible post-intervention pairs.

In these initial states, every input after the first is neutral for its operator
(ON for ALL, OFF for ANY), so the first switch alone determines the lamp.
Consequently higher paired scores on
level 3 do not establish composition across three inputs. The differing paired
denominators explain why level 3 has the higher percentage. This finding is
recorded without changing the frozen curriculum or rerunning a repaired test.

For this model, ASK recall was 100% but precision only 25% at both levels; it
overused uncertainty. Known-answer macro accuracy was about 57.1%. Brier scores
were 0.478/0.473; generated replies and discrete actions also disagreed on many
questions. Generated replies got 91/192 (47.40%) and 105/192 (54.69%) known
post-intervention questions correct, while recognizing only 13/64 and 11/64
undefined-lamp cases. They do not share the decision head's exact failure
pattern, and their near-half correctness on known outcomes does not establish
the intended rule computation. Action and language results remain separate.
Blank text and resetting state between turns each reduced query
accuracy to 12.5% and paired accuracy to zero. Those controls demonstrate use
of text and carried state in the measured behavior, not complete reasoning.
Full denominators, confusion matrices, reply agreement and all controls are
retained in the audit report (archive reference: `../runs/general-learning-local/audit/report.json`).

## Retention and broader transfer

Retention uses a separately seeded development bank from the three original
subjects. Advanced transfer uses final-split level-3 episodes: longer graph
paths and new copying operations for binding/arithmetic. The copying conditions
also introduce syntax, so they are not pure depth tests.

| Candidate | Retained query before / after adaptation | Advanced query before / after | Advanced paired before / after |
|---|---:|---:|---:|
| Episodic / interleaved | 56.25% / 31.99% | 51.43% / 31.84% | 0.00% / 0.00% |
| Episodic / blocked | 40.32% / 39.84% | 42.71% / 38.02% | 0.67% / 0.00% |
| Recurrent / interleaved | 58.68% / 41.28% | 52.34% / 35.35% | 1.82% / 1.82% |
| Recurrent / blocked | 45.79% / 44.01% | 45.38% / 44.27% | 0.26% / 2.56% |

The recurrent/interleaved arm lost 12.24 points on binding, 19.92 on graphs and
20.05 on arithmetic after adaptation. There was no rehearsal during this fixed
fast-adaptation experiment. These results support treating consolidation and
retention as part of learning quality, rather than selecting a learner solely
for its latest-task score.

Both prospective episodic-benefit screens failed. Interleaved episodic memory
was 25 points worse on adapted query accuracy than its recurrent control;
blocked episodic memory gained only 4.69 points and had zero correct pairs.
Neither met the combined transfer, retention and uncertainty requirements.
The complete checks are in summary.json (archive reference: `../runs/general-learning-local/summary.json`).
No extra initializations were run to confirm an episodic benefit that this
screen did not find.

## General mechanism diagnosis and next cycle

A read-only CPU diagnostic used 192 fresh train-split episodes across the three
trained families, with fixed seed 12,000,000 and no optimizer updates. It measured
opposite-premise representation differences and query-only gradients. Neither
audit questions nor their support set entered this diagnostic. All checkpoint
digests remained unchanged.

Changed premises produced distinct encoder activity, but substantially smaller
differences reached query decisions. For example, episodic/interleaved binding
relative separation was 0.01175 at retrieval, 0.000710 at prefrontal activity and
0.0000534 at motor output. Arithmetic retrieval separation was already very
small. Query gradients were connected; there was no universally severed pathway
or universal absence of gradients. Byte-level differences do not establish
semantic understanding, and this association does not isolate a causal cause.
See pathway-diagnostic.json (archive reference: `../runs/general-learning-local/pathway-diagnostic.json`).

The selected next experiment is a **single shared decision-learning auxiliary**:
compare a training-only 64-to-4 readout from prefrontal activity at zero versus
one predeclared nonzero loss weight. Both arms instantiate the same extra head;
all official inference and scoring discard it and use the original brain/motor
path. It supplies the same supervision in every subject, with no parsed facts
or subject-specific readouts. The hypothesis is that shorter decision-training
paths may improve the shared representation. Only measured improvement through
the original inference path would support it.

Freeze a new protocol before that run, use matched exposure and compute records,
rotate the withheld family, and examine the full retention/transfer matrix.
These already-inspected conditional banks are now known benchmarks. Keep
mechanism-required query slices visible so easy initial-state queries cannot
mask failed later reasoning. Changes to the learner and benchmark must be
versioned separately. A later self-directed scheduler should allocate practice
using broad development progress while reserving prior-subject rehearsal;
it must not inspect final questions or relax success criteria.

This next experiment has been selected and documented, **not run**. The current
cycle provides the baseline and diagnostic evidence needed to make it concrete.
Ethical scope remains inert local worlds, truthful uncertainty, preserved
abilities and useful learning per compute. No real-world actions, manipulation,
distress-based objectives or autonomous source rewriting are introduced.

## Compute, integrity and reproduction

Main training completed 7,200 updates and 460,800 repeated episode exposures.
Six adaptation copies added 96 updates and 6,144 repeated support exposures.
All work used the local RTX 5080/CPU with PyTorch 2.11.0+cu128. Main workers
recorded 2,407.845 aggregate training seconds; the longest worker wall interval
was 614.406 seconds. Four workers overlapped on one GPU, so their sum is neither
dedicated GPU-hours nor elapsed wall time. Adaptation optimizer work totaled
13.078 seconds, with evaluation recorded separately in candidate reports.

The short standalone benchmark measured batch-64 CUDA throughput of 6.74 updates/s
for recurrent and 8.26 for episodic mode, with 264.60/486.88 MiB peak allocated
memory. These are twelve-step measurements, not evidence that episodic retrieval
is generally faster. CPU throughput was 3.31/2.09 updates/s. Concurrent study
timings and coarse common-worker-time checkpoints are recorded separately; a
matched dedicated-GPU efficiency claim is not supported.

The full regression suite passed **391 tests**. All six adapted copies passed
bitwise CPU conversation-state save/reload continuation. Main and adapted
checkpoint digests remained unchanged during evaluation/diagnosis; no claim of
bitwise CUDA resumption is made. All training and audit jobs are complete.

From the project root, with a **new** study directory:

```text
python -m experiments.audit_cognitive_transfer --prepare runs/NEW-STUDY/audit
python -m experiments.train_cognitive --output runs/NEW-STUDY/episodic-interleaved --memory episodic --schedule interleaved
python -m experiments.train_cognitive --output runs/NEW-STUDY/episodic-blocked --memory episodic --schedule blocked
python -m experiments.train_cognitive --output runs/NEW-STUDY/recurrent-interleaved --memory recurrent --schedule interleaved
python -m experiments.train_cognitive --output runs/NEW-STUDY/recurrent-blocked --memory recurrent --schedule blocked
python -m experiments.audit_cognitive_transfer --runs runs/NEW-STUDY/episodic-interleaved runs/NEW-STUDY/episodic-blocked runs/NEW-STUDY/recurrent-interleaved runs/NEW-STUDY/recurrent-blocked --output runs/NEW-STUDY/audit --device cuda
python -m experiments.summarize_cognitive_study --study runs/NEW-STUDY
python -m experiments.inspect_cognitive_transfer --audit runs/NEW-STUDY/audit
python -m unittest discover -s tests -q
```

On this Windows installation use `.venv/Scripts/python.exe` as the interpreter.
The first command must precede all initial checkpoints. Existing completed
protocols, source snapshots, checkpoints and results are preserved under
`runs/general-learning-local/`. Reproduction on these inspected banks is a
known-benchmark rerun, not new held-out evidence. This cycle has one training
initialization, one withheld-family rotation and one adaptation support draw;
it is exploratory evidence, not a general-intelligence capability claim.
