# Calibrated architecture comparison: completed local results

All nine calibration runs, three main runs and six audit adaptations completed
on the local RTX 5080. There is measurable progress, but **no checkpoint is
promoted**. The sequence learner improved later conditional decisions and lost
less earlier-task accuracy during adaptation; the recurrent learner acquired
arithmetic more strongly. Neither supplied consistently strong competence,
transfer and retention across the tested subjects.

The sequence learner made a substantial advance on later conditional decisions,
but it did not improve every trained subject. Its familiar-example fit also
substantially exceeds development performance. This one-seed result does not
establish a robust architecture advantage or broad intelligence. Unlike the
preceding credit study, this protocol defined **no numerical promotion or
pass/fail benefit screen**. No promotion is an evidence-based development
decision, not a claim that an undeclared threshold was failed. The
full audit (archive reference: `../runs/sequence-study-local/audit/report.json`),
exact summary (archive reference: `../runs/sequence-study-local/audit/summary.json`) and
decision record (archive reference: `../runs/sequence-study-local/decision.json`) preserve the outcome.

## Prospective comparison and calibration

The [protocol](SEQUENCE_STUDY_PROTOCOL.md) compares recurrent regional BiC,
episodic regional BiC and an independent causal sequence learner. The two
regional models have 796,552 parameters each; the sequence model has 753,610.
The sequence model directly reads actions from its causal representation and
does not retain regional routing. Its performance must not be attributed to
regional BiC or to the separately implemented integration prototype.

Sources and recipes were frozen at **2026-09-16 20:24:15 UTC**. All architectures
receive raw observed English and the same declared action, acknowledgement,
reply-byte and within-utterance next-byte objectives. No family IDs, parsed
facts, reference replies or external-teacher answers enter inference context.
Binding, graphs and conditional logic at levels 1/2 are the trained subjects;
arithmetic is withheld from both calibration and main training.

Each architecture tested three learning rates for 600 updates from seed 2401.
Selection used the fixed final development average of family-macro paired
correctness and last-known-later-query accuracy, with lower-rate tie-breaking.
The selection record (archive reference: `../runs/sequence-study-local/selection.json`) preserves all
nine endpoints and selected **0.001 for every architecture**.

| Architecture | Score at 0.0003 | Score at 0.001 | Score at 0.003 |
|---|---:|---:|---:|
| Recurrent regional | 31.56% | 38.85% | 30.69% |
| Episodic regional | 31.38% | 33.69% | 31.73% |
| Causal sequence | 49.13% | 52.51% | 44.83% |

These are selection scores, not overall accuracy. All main models then started
fresh at seed 2501, without calibration weights or optimizer moments, and ran
3,600 batch-64 updates: 1,200 updates per family, 230,400 repeated episodes per
architecture. Matched complete-pair streams and fixed endpoints prevent choosing
favorable training snapshots. Calibration totals 5,400 updates/345,600 exposures;
main training totals 10,800 updates/691,200 exposures. The six audit candidates
add 384 updates/24,576 exposures, giving **16,584 updates and 1,061,376 repeated
episode exposures** across the complete study.

## Main development results

Source endpoint reports: recurrent (archive reference: `../runs/sequence-study-local/main/recurrent/report.json`),
episodic (archive reference: `../runs/sequence-study-local/main/episodic/report.json`),
sequence (archive reference: `../runs/sequence-study-local/main/sequence/report.json`).
Main development has 128 episodes per family. A paired success requires both
opposite answers at one question position, not two wholly correct conversations.

| Architecture | Subject | Query correct | Opposite pairs correct | Last known later query correct | Later opposite pairs |
|---|---|---:|---:|---:|---:|
| Recurrent | Binding | 234/384 (60.94%) | 23/71 (32.39%) | 61/128 (47.66%) | 2/7 |
| Recurrent | Graph | 151/256 (58.98%) | 0/64 | 64/128 (50.00%) | 0/64 |
| Recurrent | Conditional | 176/256 (68.75%) | 64/74 (86.49%) | 28/96 (29.17%) | 0/10 |
| Episodic | Binding | 240/384 (62.50%) | 27/71 (38.03%) | 66/128 (51.56%) | 2/7 |
| Episodic | Graph | 142/256 (55.47%) | 0/64 | 64/128 (50.00%) | 0/64 |
| Episodic | Conditional | 173/256 (67.58%) | 64/74 (86.49%) | 24/96 (25.00%) | 0/10 |
| Sequence | Binding | 221/384 (57.55%) | 22/71 (30.99%) | 63/128 (49.22%) | 1/7 |
| Sequence | Graph | 127/256 (49.61%) | 6/64 (9.38%) | 63/128 (49.22%) | 6/64 |
| Sequence | Conditional | 213/256 (83.20%) | 73/74 (98.65%) | 74/96 (77.08%) | 9/10 |

“Last known later” selects each episode's last yes/no question only when its
ordinal among all questions is at least two. The equal-family development means
are:

| Architecture | Query accuracy | Paired accuracy | Last known later accuracy |
|---|---:|---:|---:|
| Recurrent | 62.89% | 39.63% | 42.27% |
| Episodic | 61.85% | 41.50% | 42.19% |
| Sequence | 63.45% | 46.34% | 58.51% |

The sequence model's conditional result extends past the easy first question:
it solved 9/10 later opposite pairs versus zero for either regional model.
However, it solved fewer binding pairs and had lower graph query accuracy than
both regional models. Its six graph pairs show some counterfactual sensitivity,
while 9.38% paired correctness remains weak. Neither raw accuracy nor an
equal-family mean establishes consistent benefit across subjects.

Actions and generated replies also differ. Both regional models generated the
correct reply for all 96 known later conditional questions, while their official
actions were correct for only 28 and 24. The sequence model generated 79/96
correct replies and chose 74/96 correct actions on that slice. Correct text
therefore cannot be substituted for correct decisions. The full endpoint reports
retain class scores, uncertainty, reply agreement and query-position counts.

## Fitting, generalization and curriculum diversity

The fixed training-fit diagnostic uses 256 episodes per family: the first 128
rows of each 512-row training level block. It is a sampled familiar-bank
diagnostic, not a full training-set evaluation or checkpoint-selection rule.

| Architecture | Training-fit macro query | Development macro query | Training-fit macro pairs | Development macro pairs |
|---|---:|---:|---:|---:|
| Recurrent | 64.47% | 62.89% | 43.56% | 39.63% |
| Episodic | 63.52% | 61.85% | 44.74% | 41.50% |
| Sequence | 91.67% | 63.45% | 74.70% | 46.34% |

Sequence training-fit query/pair scores are 94.92%/88.03% for binding,
82.03%/36.72% for graphs and 98.05%/99.34% for conditional logic. Graph final-known
fit remains 171/256 (66.80%); strong average fitting does not mean every
reasoning operation was acquired. Regional graph fitting has zero paired
success, and their later conditional paired success also remains zero.

The [diversity diagnostic](COGNITIVE_DIVERSITY_DIAGNOSTIC.md) verifies 1,024 unique
training transcripts per family, with all 12 aliases and all 66 training primary
pairs. Main exposure averages 75 presentations per bank row. After consistent
first-occurrence alias normalization, these become 1,010 binding, 682 graph and
only 32 conditional transcripts. All 128 conditional development episodes share
a normalized transcript with training, versus 2/128 binding and 58/128 graph.
Exact train/development transcript overlap is zero.

Alias-normalized transcript identity is a syntactic comparison, **not proven
semantic-world identity or independent-sample count**. New seeds may chiefly
change naming in a finite repertoire. Binding development also contains many
new non-name combinations, so its gap cannot be assigned to names alone. These
observations support a proposed matched world-diversity × naming-diversity
experiment; that proposal is not frozen or launched as part of this study.

## Local compute and engineering validation

| Main worker | Training interval | Reported worker interval | Peak allocated CUDA memory |
|---|---:|---:|---:|
| Recurrent | 719.702 s | 749.640 s | 277.10 MiB |
| Episodic | 719.877 s | 746.422 s | 501.74 MiB |
| Sequence | 110.206 s | 115.688 s | 783.44 MiB |

These workers shared one GPU. Their reported intervals exclude construction and
setup and are not independent dedicated-GPU measurements. Calibration training
intervals sum to 773.187 seconds, and main training intervals sum to 1,549.785
seconds. Audit adaptation adds 36.999 training seconds and runs candidates
sequentially. Summed worker intervals are not elapsed study time or dedicated
GPU-hours. The execution record (archive reference: `../runs/sequence-study-local/execution.json`)
records observed phase envelopes: calibration 416 seconds, main training 766
seconds and audit 238 seconds. First calibration launch to observed audit
completion was **23 minutes 58 seconds**, including inter-phase gaps and
completion polling delay. This is an observed orchestration envelope, not a
precise dedicated-GPU measurement.

Memory peaks cover each worker invocation, including evaluation. Regional
evaluation uses whole banks while sequence evaluation uses chunks of 64, so
these are not matched training-only VRAM comparisons. The separate
initialized short probe (archive reference: `../runs/sequence-baseline-local/benchmark-initialized.json`)
also establishes execution speed, not learning efficiency. Useful learning per
compute hour requires retained, transferable behavior alongside throughput.

The validation history (archive reference: `../runs/sequence-study-local/validation.json`) preserves
the initial local-HTTP connection-reset failure, successful targeted reruns and
the 443-test prelaunch pass. After adding the separate regional integration
prototype, all 453 tests passed in 41.475 seconds. That prototype has 1,309,188
parameters and ten focused tests; it is **untrained**, outside the frozen
three-architecture study, and supplies no integration capability evidence.

## Learning the withheld arithmetic subject

Each architecture's main endpoint and its own fresh seed-2501 initialization
adapted to the same **128 support episodes**, using fresh optimizer moments,
learning rate 0.001 and matched pair draws. The fixed budgets were 0, 1, 4, 16
and 64 updates. The following area averages level-2/3 query accuracy, integrates
trapezoids over those update counts and divides by 64. The 16-to-64 interval
therefore receives most of the weight.

| Architecture | Pretrained query-curve area | Fresh query-curve area | Prior-learning gain | Final pretrained query | Final fresh query |
|---|---:|---:|---:|---:|---:|
| Recurrent | 60.77% | 50.35% | +10.42 points | 65.59% | 58.17% |
| Episodic | 52.42% | 50.47% | +1.95 points | 52.31% | 58.40% |
| Sequence | 55.83% | 51.91% | +3.91 points | 60.77% | 50.68% |

All three pretrained models had positive query-area gains against their own
fresh controls. This is ordinary gradient adaptation, not evidence that a
meta-learning algorithm was trained. The episodic model's positive area does
not imply a better endpoint: it finished below its fresh control. Descriptive
paired-area differences, computed on the same curve, were +13.58 points
recurrent, +0.34 episodic and **−1.36 sequence**. The sequence query-area gain
therefore does not extend to the paired learning-curve summary.

At update 64, each query level contains 256 episodes, 256 opposite-answer
question pairs and 128 later-question pairs. Each cell below gives
**pretrained / fresh** results.

| Architecture and level | Query accuracy | Opposite pairs correct | Last known later correct | Later opposite pairs correct |
|---|---:|---:|---:|---:|
| Recurrent level 2 | 73.96% / 66.54% | 79/256 / 0/256 | 147/256 / 128/256 | 23/128 / 0/128 |
| Recurrent level 3 | 57.23% / 49.80% | 69/256 / 0/256 | 116/256 / 128/256 | 8/128 / 0/128 |
| Episodic level 2 | 66.54% / 66.80% | 3/256 / 2/256 | 127/256 / 128/256 | 3/128 / 0/128 |
| Episodic level 3 | 38.09% / 50.00% | 0/256 / 1/256 | 67/256 / 128/256 | 0/128 / 0/128 |
| Sequence level 2 | 72.53% / 71.48% | 80/256 / 90/256 | 148/256 / 150/256 | 46/128 / 42/128 |
| Sequence level 3 | 49.02% / 29.88% | 37/256 / 47/256 | 90/256 / 2/256 | 0/128 / 0/128 |

Level 2's last question follows an increase and decrease. Level 3 introduces
copying the accumulated count to another name and increasing that copy, an
operation and wording combination absent from support. Neither sequence
candidate solved any complete level-3 later pair. Their earlier paired successes
occur after the initial increase; those questions already require an update,
so they must not be described as mere initial-value lookup. They still do not
establish successful copy-and-update composition.

Level-3 arithmetic has **no unknown targets**. Its ASK recall is undefined,
not zero or perfect. The pretrained sequence and episodic candidates nevertheless
predicted ASK on 54/512 and 124/512 questions respectively. Level-2 ASK recall
was 98.44% sequence, 99.61% recurrent and 100% episodic; these easier unknown
decisions cannot compensate for weak known paired correctness.

## Retention, advanced operations and context controls

| Pretrained learner | Retention query before → after | Retention pairs before → after | Advanced query before → after | Advanced pairs before → after |
|---|---:|---:|---:|---:|
| Recurrent | 62.85% → 35.26% | 41.49% → 8.75% | 62.89% → 38.67% | 39.39% → 7.02% |
| Episodic | 61.50% → 43.34% | 42.91% → 25.16% | 61.20% → 40.23% | 39.16% → 14.22% |
| Sequence | 62.85% → 60.53% | 43.35% → 37.73% | 48.96% → 48.05% | 29.34% → 24.25% |

Recurrent arithmetic acquisition cost **27.58 points** of retained query
accuracy, versus 18.16 episodic and 2.32 sequence. The sequence model lost less
on that mean, but still lost 5.62 paired points. Its retained graph later pairs
fell from 12/128 to 4/128. Its conditional later pairs were 15/31 before
adaptation and 11/31 after; this independent retained bank also shows why the
development result of 9/10 should not be generalized as a stable 90% ability.

Before adaptation, **both regional models freely generated all 192/192 correct
known later conditional replies**, despite action accuracies of 33.85% recurrent
and 29.69% episodic and action paired correctness of 0/31. After arithmetic
adaptation, their correct replies collapsed to 5/192 (2.60%) and 45/192 (23.44%).
Sequence replies changed from 149/192 (77.60%) to 140/192 (72.92%). The regional
pathways therefore contained enough information to express these known outcomes
before adaptation; weak actions cannot be described as absence of the learned
concept. Generic action/language alignment, routing and retention require
separate measurement alongside representation quality.

Advanced level 3 includes copy operations in binding, longer graph paths and
higher-arity conditional rules. It is not uniformly a depth-only holdout.
Sequence advanced binding later pairs stayed 0/20, while advanced graph later
pairs fell 16/128 to 3/128. Thus less aggregate forgetting is meaningful progress
relative to these regional controls, but does not establish dependable retained
composition or superior advanced transfer.

| Pretrained arithmetic endpoint | Ordinary query / pairs | Blank-English query / pairs | Reset-history query / pairs |
|---|---:|---:|---:|
| Recurrent | 65.59% / 28.91% | 41.67% / 0% | 41.67% / 0% |
| Episodic | 52.31% / 0.59% | 0% / 0% | 16.67% / 0% |
| Sequence | 60.77% / 22.85% | 16.67% / 0% | 41.34% / 0% |

All six candidates, including the fresh controls, had zero paired accuracy
under either control. Context contributes to the ordinary paired results,
without making those results sufficient. The audit verified unchanged base
checkpoints and **exact CPU continuation for all six candidates** after
serialization at turn three. Sequence sessions preserve raw observation prefixes
and recompute them; regional sessions preserve regional/episodic activity.
These are different state mechanisms and do not imply equal inference cost.

## Decision and next work

No checkpoint is promoted. The protocol did not define an automatic numerical
promotion gate, and this report does not invent one retrospectively. Evidence
supports further development: some later conditional decisions now generalize,
prior training can improve arithmetic acquisition, and the sequence model loses
less average earlier-task accuracy. Evidence also leaves serious limits:
uneven subjects, weak copy composition, naming/combination gaps, forgetting and
uncertainty errors. No architecture is the consistently stronger learner across
all reported outcomes.

Next, freeze a broad comparison of curriculum diversity, shared representation,
generic output alignment and retention under matched budgets. The [diversity proposal](COGNITIVE_DIVERSITY_DIAGNOSTIC.md)
separates world refresh from consistent renaming while protecting new held-out
boundaries. The untrained regional sequence adapter permits a later test of
stronger comprehension and credit assignment through the existing motor pathway;
its motivation is not proof that the older learner lacked the represented
concept. It must earn its
own training, transfer and retention evidence; implementation and passing tests
are not an integration result. Avoid promoting a fluent reply or favorable
aggregate while the corresponding decisions or retained skills remain weak.

This experiment has one fresh main initialization per architecture, one held-out
subject rotation and one support draw. Architecture, access to raw history and
decision routing differ together; this is a comparison of complete learner
designs, not isolation of one causal mechanism. All local jobs for this study
have finished. Its inspected audits are known evidence for future development.
The
[home-learning objective](HOME_LEARNING_ROADMAP.md) remains open beyond these
finite tasks, with broader learning, retention, self-direction, usable saved
state and progressive external-teacher independence still required.
