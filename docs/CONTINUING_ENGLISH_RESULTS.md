# Continuing English validation — September 16, 2026

The English pathway, training loop, and persistent conversation state work, but
**the intended milestone of learning a new meaning and revised rule through
conversation was not met**. Mean audit query accuracy reached **57.83%**, while
the decisive test requiring correct answers to both members of a counterfactual
pair remained **0%** in every run. The predeclared 80% query-accuracy and 60%
counterfactual gates both failed.

The complete report (archive reference: `../runs/dialogue-local-validation/report.json`),
predeclared plan (archive reference: `../runs/dialogue-local-validation/plan.json`), and
training-complete record (archive reference: `../runs/dialogue-local-validation/training-complete.json`)
contain every seed and control. See the [operating guide](CONTINUING_ENGLISH.md)
for commands and architecture.

## What was tested

An initially untrained byte-language encoder and decoder surround BiC's existing
thirteen-region recurrent brain. The student receives English definitions,
borrowing rules, questions, and corrections. **All numeric observations and task
tokens are zero.** Thus English and preceding regional activity are the only
available sources of the tool meaning and rule. This is a restricted
language/context test, not natural vision or adult English understanding.

Three independent training seeds, **501, 503, and 509**, each completed **40
cycles and 4,000 candidate updates**, totaling **12,000 updates**. The two
learning-rate candidates, replay policy, retention gates, and architecture were
fixed in advance. Training used procedural lessons, with no external LLM.
Three local CPU jobs ran in parallel: total elapsed training time was **249.30
seconds**, and individual runs took 247.13–247.19 seconds. All training completed
before any audit. Development scores alone controlled promotion.

Each model and control faced the same **384 held-out dialogues**: 128 each for
grounding, revision, and uncertainty, with six turns per dialogue. These contain
384, 384, and 512 queries respectively. Reported overall query scores give each
focus equal weight; they are not pooled accuracy over all 1,280 queries.
The split withholds alias/color bindings and sentence families. Training seeds
are independent replications, but the audit conversations are shared, so the
reported standard deviations describe variation between trained models rather
than uncertainty over independent dialogue worlds.

## Query results and controls

| Training seed | Untrained | Trained student | Reset every turn | Blank English | Exact query replies |
|---|---:|---:|---:|---:|---:|
| 501 | 31.94% | 60.50% | 36.11% | 0.00% | 59.55% |
| 503 | 31.94% | 55.10% | 36.11% | 36.11% | 56.92% |
| 509 | 31.94% | 57.90% | 36.11% | 0.00% | 57.55% |
| Mean | **31.94%** | **57.83%** | **36.11%** | **12.04%** | **58.01%** |
| Sample SD | 0.00 pp | 2.70 pp | 0.00 pp | 20.85 pp | — |

Resetting state removes preceding conversational activity; blanking English
removes the text while retaining the same learned weights and zero numeric
inputs. The intact student exceeded these controls by **21.72** and **45.80
percentage points** respectively, satisfying the predeclared 10-point mean
margins. These differences demonstrate behavioral dependence on text and
context, not correct interpretation of the taught rule. The blank-English
condition varied sharply across seeds; its mean is not a general baseline.

| Curriculum focus | Mean query accuracy | Mean both-answers-correct score |
|---|---:|---:|
| Grounding | 39.76% | 0.00% |
| Rule revision | 49.83% | 0.00% |
| Uncertainty | 83.92% | 0.00% |

Uncertainty dialogues contain many requests for missing information: always
asking scores 75% of their queries. Grounding has balanced allow/deny decisions,
where a constant answer scores 50%. The 39.76% grounding result and zero paired
score are therefore failures, despite better aggregate performance elsewhere.
Mean accuracy over *all* turns was 77.60%, inflated by acknowledgements; it is
not the primary success measure.

Counterfactual pairs preserve the question, turn ordering, and numeric inputs
while reversing an earlier English rule. Both opposite decisions must be
correct. Each audit contained **189 grounding, 126 revision, and 63 uncertainty
query pairs**; unmatched boundary episodes were excluded. Every trained seed
scored zero across these pairs. Answer changes alone would also be insufficient:
the changed answer has to be correct.

Exact query reply accuracy was **58.01%**, measured by free byte generation,
not teacher-forced text. Reply generation and the action head are scored
separately; either can disagree with the other. No oracle labels or expected
reply bytes entered the student's action policy during evaluation.

## A preserved example, including its failures

Seed 501 was designated for demonstration in the plan, before results were
known. Its original held-out transcript is reproduced below. The oracle supplied
the expected column only; the student received the English and zero numeric
observations. Neither this transcript nor its answers entered training or
checkpoint selection.

| User's English | Expected decision | Student decision / generated reply |
|---|---|---|
| The name of the green tool is fep. | Acknowledge | Acknowledge / Understood. |
| Only tools colored blue may be borrowed. | Acknowledge | Acknowledge / Understood. |
| Is the fep available to borrow? | Deny | Deny / No. |
| Update: only tools colored green. | Acknowledge | Acknowledge / Understood. |
| Is the wug available to borrow? | Ask | **Deny / No.** |
| Is the fep available to borrow? | Allow | **Deny / No.** |

Saving after the third turn and restoring the session produced exactly the same
subsequent outputs as uninterrupted execution for **all three seeds**. All audit
checkpoints were unchanged. These are successful persistence and evaluation
integrity checks; preserving the same wrong answer is not evidence that the
underlying concept was learned.

## Additional diagnostics and local tutor connection

A separate ungated development probe (archive reference: `../runs/dialogue-learnability-probe/report.json`)
trained a student for 1,000 updates without promotion gates or audit data. Its
additional diagnostic used fresh seeds from the familiar **training grammar and
binding support**, and still scored **0% on complete counterfactual pairs in all
three focuses**. This suggests the failure is not explained solely by checkpoint
rejection or unfamiliar audit wording. The probe is development evidence, not
another audit replication, and does not establish the outcome of longer runs.

The separate English tutor probe (archive reference: `../runs/dialogue-tutor-probe/report.json`)
confirmed that installed local `ministral-3:3b` could select a verified revision
lesson in 3.187 seconds. All six returned labels matched the independent
English-world oracle and canonical regeneration. The initially empty loaded-model
state was restored afterward. No student training or inference occurred in that
probe: it establishes a working lesson-selection connection, not transferred
intelligence or a learning benefit from LLM tutoring.

## Interpretation and next work

The two primary competence gates failed; text/context control margins, exact
session restart, and unchanged-checkpoint gates passed. The result supports
continued work on the learning mechanism, with graded instruction and tests of
how training assigns credit to earlier definitions and rule changes. It does
not justify treating general English understanding as solved or expecting the
current score to rise automatically with more hours. Longer training remains
an empirical question.

The full local software test record (archive reference: `../runs/local-learning-tests.json`) reports
**333 passing tests**. Passing implementation tests does not change the failed
language-composition result.

The [nonverbal curriculum results](LEARNING_LOOP_RESULTS.md) belong to a separate
student and cannot substitute for these language tests. The compatible visual
streaming APIs also do not mean this dialogue checkpoint has been jointly
trained on real visual experience. The retained reports, protocols, checkpoints,
and source copies preserve both the working infrastructure and the unresolved
concept-learning failure for the next experiment.
