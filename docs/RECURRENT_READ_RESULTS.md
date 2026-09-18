# Repeated reading: completed local learning comparison

September 17, 2026. Adding three tied reads of the conversation did not improve
the learner in this fixed single-seed comparison. Both branches ended at zero
jointly correct action-and-English pairs on fresh development, retained
development and the training-fit sample. The candidate also lost 9.09 percentage
points of known quantity decisions and used 8.13% more measured training time.
It failed the prospective screen; neither checkpoint is promoted.

## What ran

The existing learner and the recurrent reader started with exactly equal
parameter tensors. The reader reused its final encoder block for three additional
causal reads, adding computation without parameters or privileged inputs. Both
received the same original objective and the same three admitted curriculum
cycles: **648 updates and 62,208 episode exposures per branch**, or **1,296
updates and 124,416 episode exposures total**. All updates completed, with no
unknown optimizer outcomes or retries. Each branch retained three complete
weights-and-AdamW checkpoints.

The new prepared-batch path matched the original trainer exactly in focused CPU
and RTX 5080 checks: packed inputs, losses, weights, AdamW state and consumed
evidence. These checks establish correctness, not an end-to-end speed claim.
The recurrent reader's six tests and independent source review also passed.

The fixed [protocol](RECURRENT_READ_PILOT.md) preceded learning and scores. The
fresh development bank excluded 1,701,000 authenticated historical, current
training and previously reserved transcripts. Exact text was excluded; the
finite grammar, vocabulary and semantic patterns still overlap. Training-fit
rows are actual early lessons, not a diagnostic of only the latest batch.

## Final unaided results

Each development pair changes a fact and requires opposite known answers. Joint
success requires correct native actions and exact freely generated English
replies on both members. The tutor supplied no answers during evaluation.

| Final fresh development | Existing learner | Repeated-reading learner |
|---|---:|---:|
| Correct action pairs | 28/360 | 25/360 |
| Correct English-reply pairs | 8/360 | 0/360 |
| Jointly correct pairs | 0/360 | 0/360 |
| Known-query actions | 584/1,588 (36.78%) | 548/1,588 (34.51%) |
| Known-query English replies | 545/1,588 (34.32%) | 517/1,588 (32.56%) |
| Unknown-query actions | 1,231/1,794 (68.62%) | 1,282/1,794 (71.46%) |
| Unknown-query English replies | 1,359/1,794 (75.75%) | 1,430/1,794 (79.71%) |
| Action/reply agreement | 2,282/3,382 (67.47%) | 2,130/3,382 (62.98%) |

All three families had zero final joint pairs out of 120 each. Known quantity
actions fell from 216/528 to 168/528 (40.91% to 31.82%). Increased unknown-answer
accuracy did not compensate for the decline in known answers. Overall query
accuracy alone would therefore give a misleading impression of improvement.

| Fresh joint pairs by update | Existing learner | Repeated-reading learner |
|---|---:|---:|
| 0 | 0/360 | 0/360 |
| 216 | 0/360 | 0/360 |
| 432 | 4/360 | 2/360 |
| 648 | 0/360 | 0/360 |

At the endpoint, both branches also had 0/54 joint training-fit pairs and 0/360
joint pairs on reused development. The small intermediate gains did not survive
the final cycle. This remains weak acquisition and retention, not evidence of
adult English comprehension or learned independence.

## Cost and verification

| Measured work | Seconds |
|---|---:|
| Shared data packaging and fresh evaluation construction | 347.516 |
| Freezing the launch | 3.812 |
| Complete paired execution | 418.687 |
| Total of these phases | 770.015 |
| Existing learner training including per-step preparation | 153.464 |
| Repeated-reader training including per-step preparation | 165.945 |

The final two rows are parts of execution, not additional costs. The recurrent
candidate used 12.481 extra training seconds and gained zero joint development
pairs. Peak allocated GPU memory was 1,666.62 MiB. Costs exclude earlier reusable
data admission, engineering and focused tests; the saved launch records inherited
data preparation separately. This experiment is matched by parameters and lesson
exposure, not by compute time.

The pure post-run analysis checked saved artifact digests and frozen sources,
then recounted all **24 bank endpoints and 12,384 evaluation episode records**.
No second model evaluation was performed. The run made **zero LLM calls**: it
isolated learner computation. The separate [tutor pilot](FOUNDATION_TUTOR_LOOP_RESULTS.md)
already exercised an external LLM order choice and withdrawal; rich tutoring
and a beneficial transfer of knowledge remain unproven.

## Next decision

Do not extend this repeated-reader configuration unchanged. The result rejects
this specific extra-computation intervention at this seed and budget; it does
not prove that all iterative architectures fail or identify a unique bottleneck.

The next controlled change is shared teaching of entity bindings and current
state across color, count and switch. A single training-only decoder should
answer entity-conditioned state questions from observation-derived features,
with targets verified from each causal prefix. The same decoder parameters
should serve every entity and family. Compare positive supervision against a
zero-weight control with identical initialization, lessons and parameter budget.
Keep parsed state out of inference inputs and preserve independent actions and
free English replies. The earlier unshared binding scaffold's poor transfer
means that better auxiliary accuracy alone cannot justify promotion.

That contrast tests the full teaching intervention, not factorization separately
from additional supervision. The current foundation generator avoids interfering
overwrites: changed initial facts test causal sensitivity, but do not establish
reliable state revision in changing worlds. Extend that coverage explicitly in
a subsequent curriculum comparison rather than imply it is already measured.

Records: `runs/recurrent-read-pilot-local/attempt-001/`.

- Launch SHA256: `6a8b20d560b76399dc970e90d96e9477e4fd96856f0a28ebb57db36767f8a287`
- Execution summary SHA256: `06d210f7044934b65685f409ec5ef715a3d7d69be76da104ea24568230045b97`
- Verified analysis SHA256: `8026469b297da5f00adcdb6f8e49d1227983a4934dd210ab12613af33142551f`
