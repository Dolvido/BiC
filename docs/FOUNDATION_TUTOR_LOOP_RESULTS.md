# Local external-tutor loop: completed pilot

September 17, 2026. The complete local pipeline ran on the RTX 5080: verified
practice, learner updates, an actual external LLM curriculum decision, unaided
evaluation, two full optimizer restarts, and learning with the tutor withdrawn.
This is a working integration milestone. The resulting learner still has weak
English reasoning and is not promoted as a capable or independent checkpoint.

## What executed

| Phase | Lifetime updates | Curriculum source | New tutor chats |
|---|---:|---|---:|
| Initial cycle | 0–216 | Procedural curriculum | 0 |
| Tutor-selected cycle | 216–432 | Local `ministral-3:3b` selected curriculum order | 1 before this phase |
| Withdrawal cycle | 432–648 | Procedural curriculum, teacher disconnected | 0 |

All three cycles used fresh, previously admitted lessons across color, count and
switch, depths zero through five, and eight-, ten- and twelve-turn contexts.
The same learner accumulated **648 GPU updates and 62,208 training episode
exposures**. Both new learner processes restored exact weights and full AdamW
state before further learning. Evaluation scored **6,552 episode exposures**,
including 65,520 freely generated turn replies, without teacher answers.

The tutor saw only aggregate development counts. It chose between two balanced
orders of the same lesson inventory; it did not generate labels or direct the
student at inference time. Its single accepted chat used 2,522 prompt tokens
and nine output tokens. The decision interval was 4.156 seconds; model identity
was verified before and after. Withdrawal made zero API calls. The chosen order
was also the procedural default, so this run supplies no causal estimate of
tutor benefit.

An initial restart refused the checkpoint because JSON changed the optimizer's
`betas` metadata from a tuple to a list. This happened before model construction
or learning. A targeted repair preserved strict weight and optimizer comparisons
and passed both JSON round-trip regressions. Explicit recovery reused the
completed 216-update checkpoint, original evaluations and accepted tutor choice;
none was repeated. The original failure and all costs remain recorded. See the
[recovery record](FOUNDATION_TUTOR_LOOP_RECOVERY.md).

## Learning results

The constant development bank contains 360 changed-fact pairs. Paired action
requires correct native decisions in both versions; paired reply requires exact
English replies in both. Joint success requires both modalities in both versions.

| Updates | Paired action | Paired reply | Joint paired success | Known-query action | Known-query reply |
|---:|---:|---:|---:|---:|---:|
| 0 | 2/360 | 0/360 | 0/360 | 17.02% | 0.00% |
| 216 | 2/360 | 6/360 | 1/360 | 30.46% | 28.58% |
| 432 | 17/360 | 31/360 | 2/360 | 28.77% | 27.89% |
| 648 | 28/360 | 19/360 | 1/360 | 37.56% | 32.98% |

The final reserved audit combined fresh realizations and held compositions:

| Family | Paired action | Paired English reply | Joint paired success |
|---|---:|---:|---:|
| Color | 9/132 | 17/132 | 0/132 |
| Count | 0/132 | 0/132 | 0/132 |
| Switch | 19/132 | 10/132 | 0/132 |
| **Total** | **28/396 (7.07%)** | **27/396 (6.82%)** | **0/396** |

On individual known audit questions, native actions were correct on 665/1,756
(37.87%) and exact English replies on 585/1,756 (33.31%). On unknown questions,
correct actions were 1,212/1,964 (61.71%) and replies 1,482/1,964 (75.46%). The
learner also asked unnecessarily on 25.17% of known questions through its action
output and 36.16% through its English output. Action/reply agreement was 63.58%.
High unknown-answer scores therefore do not establish accurate comprehension.

Withdrawal continued learning, but did not demonstrate successful independence.
On its new development bank, paired actions rose from 24/360 to 31/360, while
paired replies fell from 41/360 to 23/360 and joint success from 7/360 to 1/360.
Known-query actions improved from 28.33% to 37.63%; unknown-query actions fell
from 81.57% to 61.80%. These mixed changes, together with weak audit performance,
do not justify extending unchanged training and calling the result general
intelligence. No independent learned curriculum policy was trained in this run.

## Efficiency and next engineering decision

Data preparation took 583.656 seconds wall / 575.109 seconds CPU. It admitted
62,208 unique training transcripts and 2,952 reserved transcripts against the
explicit historical inventory and each other. Its 1,127,444 returned canonical
rows include verification copies and reconstruction, not additional lessons.

The two orchestration invocations took 698.782 seconds combined. Successful
learning workers accounted for 680.703 seconds within those intervals; the
rejected restore worker added 4.594 seconds. The full interval from original
orchestration start to final completion, including repair, was 1,138.704 seconds,
inside the original 3,600-second allowance. These are wall times, not measured
GPU-kernel hours. Peak learner CUDA allocation was approximately 1,500 MiB.

Across successful workers, rebuilding and verifying lesson bundles took
285.658 seconds, **42.0% of worker wall time**. Learner-step calls took 348.776
seconds, including their own packing, verification and neural work. Evaluation
took 13.533 seconds. This identifies a concrete efficiency target: preserve
verified compact curricula while reducing repeated reconstruction and validation.
It does not establish a measured speedup for an optimization not yet implemented.

The next learning work should target shared factual binding, composition and
agreement between decisions and English, across the whole curriculum. The tutor
currently selects lesson order; richer verified teaching and a matched procedural
comparison remain needed before claiming that LLM tutoring improves BiC or makes
it progressively independent. Keep acquisition, retention and unaided transfer
as the acceptance criteria rather than tutoring activity or fluent-looking text.

## Preserved evidence

Independent pure-data checks recounted all saved raw evaluation groups and cell
summaries: 1,440 records in the initial cycle and 5,112 in the remaining cycles.
All counts, denominators and producer identities matched. They performed no
new inference, checkpoint loading or learning. A report-envelope comparison
error in the first final-analysis attempt was corrected and its cost retained;
it did not affect the recorded learner results.

- [Prospective pilot contract](FOUNDATION_TUTOR_LOOP_PILOT.md).
- Completed data preparation (archive reference: `../runs/foundation-tutor-loop-data-local/attempt-001/preparation.json`).
- Original invocation and retained failure (archive reference: `../runs/foundation-tutor-loop-local/attempt-001/orchestration/receipt.json`).
- Completed recovered invocation (archive reference: `../runs/foundation-tutor-loop-local/attempt-002/orchestration/receipt.json`), SHA256 `7542a4d4c6b35832cecc5f18af17a3ab45afa2b7e4fd3f9e85bc46b1c743e4b8`.
- Final experimental checkpoint (archive reference: `../runs/foundation-tutor-loop-local/attempt-002/orchestration/worker-2/checkpoint.pt`), SHA256 `4ebbc584bdd22d17a4fbc858d361df86a027ff1fc9e9b8496613c2358bd29da8`.
- JSON restart regression (archive reference: `../runs/foundation-layout-continuation-json-validation-local/attempt-001/report.json`).
- Independent result recount (archive reference: `../runs/foundation-tutor-loop-analysis-local/analysis-001.json`), SHA256 `927d97d7ad7b0fe9ada56d0246e417b67b9fbf084a2aa511297ebc0df5e21462`.

The original CPU bridge proof used five tiny updates; the JSON regression used
three. Their eight CPU updates and 48 episode exposures are separate validation
work and are not included in the 648-update learning result. No live teacher
request or completed GPU training was repeated during recovery.
