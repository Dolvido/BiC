# Repeated-reading learning pilot

This is a prospective local learning comparison, fixed before either branch is
trained or scored. Its purpose is to test a reusable computation change across
the complete current foundation curriculum, rather than tune individual weak
questions. It is an exploratory single-seed screen, not evidence of general
intelligence, broad English comprehension, or learned self-direction.

## Change and control

Compare the existing causal sequence learner with a tied recurrent reader.
Both have exactly the same parameters and initial weights. The reader performs
three additional reads from each utterance's end position into its causal byte
memory, reusing the final encoder block. These reads receive no parsed facts,
answers, task identifiers, or privileged causal annotations. The ordinary byte
encoder and observation objective are unchanged. Actions and replies continue
to have separate readouts, so agreement alone cannot demonstrate improvement.

Both branches use width 192, four encoder layers, four heads, feedforward width
768, 1,024 positions, at most 12 turns, 128 input bytes and 32 output bytes;
initialization seed 852204001; AdamW learning rate 0.003; gradient clipping at
1; and the original action + 0.1 reply + 0.1 observation objective. Execution
uses the existing strict FP32 profile on the local RTX 5080.

## Lessons and execution

Use the three already admitted original-layout tutor-pilot plans, in canonical
curriculum order: 648 updates per branch, with 32 episodes from each of color,
count and switch per update. Each branch receives 62,208 episode exposures.
All depths 0–5 and lengths 8, 10 and 12 remain in scope. Materialize each
admitted bundle once, check its full expected evidence, and publish its digest.
The two branches consume identical rows using the same prepared-batch optimizer
kernel; they alternate which branch executes first on each update.

Checkpoint and evaluate at updates 0, 216, 432 and 648, saving model and full
optimizer state at the three nonzero endpoints. Finish the fixed schedule
regardless of intermediate scores, unless execution fails or the 3,600-second
execution allowance expires. Retain partial evidence on failure. The data
preparation time is reported separately and included in the overall project
cost. No checkpoint is automatically promoted.

## Evaluation

Before training, create a development bank with seed 852202001 and four pairs
per cell using the existing development partitions (720 episodes). Exclude
the authenticated historical transcript union, all 62,208 training transcripts
and the 2,952 previously reserved tutor-pilot transcripts. This is exact-text
exclusion within a finite grammar; semantic and template overlap remain.

Also evaluate a balanced training-fit sample containing the first observed
pair for each family × depth × length (54 pairs, 108 episodes), and the
previous pilot's retention bank (720 episodes), explicitly identified as reused
development material. Each branch starts fresh: this bank tracks retention
across this run, not retention of a transferred earlier checkpoint.

Use native actions and freely generated replies with no tutor available.
Report jointly correct opposite-answer pairs, action and reply pairs separately,
known-answer accuracy, unknown-answer accuracy, unsupported abstention, and
action/reply agreement, overall and per family. Preserve all raw records.

## Decision fixed before scores

The final 648-update endpoint is the primary comparison. A candidate is worth
a larger multi-seed study only if it gains at least 5 percentage points in
jointly correct fresh-development pairs overall, does not lose joint pairs in
any family, and improves total joint training-fit pairs. Neither head may lose
more than 5 percentage points of known-answer or unknown-answer accuracy in
any development family, or more than 5 points of overall joint retention pairs.

Report preparation, training, evaluation and total wall time, peak GPU memory,
and the incremental correct pairs per additional training second. A slower
candidate is not an efficiency improvement simply because it receives the
same number of updates. A positive screen justifies replication and a matched
compute comparison; a negative screen does not justify scaling it unchanged.

The external tutor remains available in the established loop but makes no
choice in this controlled comparison. This run tests the learner, not tutor
benefit. The previous tutor pilot demonstrated one bounded order-selection
call and teacher withdrawal; it did not demonstrate rich tutoring or learned
independence.
