# Shared semantic reply learning: prospective local comparison

Recorded 2026-09-17 after the complete architecture, shared-feature and fixed
gradient results, before implementing or training this comparison. This is one
curriculum-wide objective intervention. It is not selected from weak cells and
does not establish general intellect outside the declared finite grammar.

## Decision and hypothesis

The learned hierarchy failed its matched screen. Historical representation
probes did not consistently improve on initialization and all fixed probe fits
were unconverged. Original first-byte and full-reply correctness were identical
on every measured group. The completed gradient inspection found query–reply
alignment in every endpoint/group/parameter partition, with median weighted
reply/query norm ratios of about 0.0053–0.0073. Query was the largest individual
weighted component everywhere. These observations do not support describing
auxiliary losses as generally dominant or opposed to decision learning.

Test whether allocating reply learning equally across semantic answer classes
and actual utterances improves common representations and useful behavior. The
old pooled-token reduction allocates learning according to class frequency and
reply length as well as errors. Changing that allocation is a hypothesis; the
gradient measurements do not establish its causal benefit.

## Two explicit objectives

Both arms use the same fresh flat sequence model and the same action, observation,
decoder, clipping and AdamW paths. The baseline computes the original
`sequence_objective` without arithmetic changes. Its reply loss is the mean over
turn positions of the pooled non-PAD token cross entropy across the family batch.

The `balanced_reply` arm changes only the reply reduction:

1. Compute ordinary cross entropy over all 259 output tokens, ignoring PAD.
2. For each actual episode-turn, average its non-PAD target-token losses,
   including EOS.
3. Average those utterance losses separately within each present semantic action
   class: deny, allow, unknown/ask and acknowledgement.
4. Average the present class means. The weights sum to one.

The outer family average and weights remain unchanged: equal present query-class
action cross entropy plus 0.25 acknowledgement cross entropy, plus **0.1 reply**
and **0.1 observation-byte** loss. All original reply tokens and trainable paths
remain active. No first-byte-only loss, constrained decoder, extra readout,
teacher answer at inference, gradient surgery or observation-loss deletion is
introduced. Class balancing uses existing training supervision only. It jointly
changes class and length allocation; this comparison does not separate those two
parts of the semantic normalization rule.

## Matched learning and local budget

Use the already authenticated `runs/variant-study-local/data` calibration and
main plans, admissions, inventories and banks, bound to its completed independent
data-verification receipt. A current load checks its immutable files; a real
process-owned `AuthenticatedPlanIndex` is constructed once per stage and reused
across that stage's fresh jobs. No persistent index is trusted. Reusing this
preparation avoids generating another nearly identical curriculum for an
objective comparison. Canonical lesson materialization remains charged.

The previously viewed development and audit banks are **reused benchmarks**.
Their original composition partitions remain meaningful, but they are not a new
untouched confirmatory test after these development decisions. No training
receives any of their transcripts. Any promising result needs a fresh independent
confirmation before a capability or adoption claim.

- Model: flat, width 192, four layers, four heads, feedforward 768, 12 maximum
  turns; 2,221,738 parameters, identical tensors within each paired seed.
- Calibration: fresh seed 8471, each objective at rates 0.0003, 0.001 and 0.003;
  exactly 510 updates per run and the complete existing calibration schedule.
- Select one rate per objective using the existing complete 90-cell development
  rule: equal domains/cells, mean paired actions, paired replies and
  target-balanced known correctness; round to 12 decimals, then lower rate.
- Main: fresh paired seeds 8472, 8473 and 8474 at the selected arm-specific rates;
  exactly 3,072 updates per run, with identical ordered curriculum and mixed tail.
- Each update uses three family microbatches of 32 episodes: 96 episodes.
- Calibration checkpoints: 0 and 510. Main checkpoints: 0, 384, 768, 1152,
  1536, 1920, 2304, 2688 and 3072.
- Total formal maximum: **21,492 updates and 2,063,232 episode exposures**.
  Validation fixtures and setup/scoring costs are additional and separately
  recorded. There is no automatic extension, extra candidate or retry.
- Local MSI RTX 5080 only, existing strict FP32 deterministic runtime, one CPU
  thread per worker. One process reuses its stage index through training,
  verification and scoring. Jobs run sequentially, baseline then candidate for
  each ascending calibration rate or main seed; simultaneous live learners and
  new data-prefetch machinery are outside this comparison. Record
  actual wall/CPU time, CUDA peaks, preparation, training, restores and scoring;
  this matches updates/exposures, not computation. A nonpreemptive 14,400-second
  allowance for each explicitly launched phase is checked at operation boundaries.

No model-selection scores are inspected while that stage is still training.
After all stage endpoints are sealed, verify exact job/source/data/checkpoint
identities, canonical consumed prefixes and the complete step journals before
scoring. Preserve every declared checkpoint and full family/cell/length/depth
breakdown; the 3,072 endpoint is primary, not the best observed checkpoint.

## Validation before formal gradients

Use an additive versioned trainer and checkpoint schema with an exact objective
recipe and source closure; do not edit frozen historical trainers or implicitly
migrate old checkpoints. A small, declared CPU fixture must establish baseline
arithmetic equivalence, the candidate's analytic reduction, active gradients,
and exact split/save/restore/continuation for both objective IDs. Reject
cross-objective or altered-recipe restores before swapping live learner state.
Then perform one bounded selected-width CUDA continuation proof using the same
strict runtime, covering both objectives and all three proposed rates. Record
physical validation work separately; do not tune learning settings from it.

Fixtures and probes establish only their tested execution boundary. An
interrupted phase retains its work receipt; no missing terminal record is treated
as zero work and no unknown partial attempt is automatically replayed.

The initial CPU fixture uses one admitted 66-bundle plan with two episodes per
family microbatch and fresh width8/layer1 models. Its ceiling is eight completed
optimizer updates: two original-baseline steps, two new-baseline steps plus one
resumed-baseline step, and two candidate steps plus one resumed-candidate step.
A deliberate interrupted family pass and one synthetic analytic reply fixture
are separate checks. Bound neural attempts at 30 family forwards, 60 episode
passes and 30 backwards, with at most 12 model/template constructions. Record
actual counters. It reads no historical checkpoints and performs no GPU work.

The selected-width CUDA proof has six objective/rate cases. Each uses six
reference updates, a separate three-update prefix with save/strict restore and
three-update continuation, and six repeat updates: **108 physical updates and
10,368 episode exposures** in total. A single admitted 126-bundle fixture plan
uses seed 847000001, stage20/final6, micro32, rehearsal4 and order seed8470;
the first six curriculum updates exercise 8/10/12 turns twice. Fresh model seed
847001 is fixed. Compare weights, complete optimizer state, recipe, cursor and
canonical evidence after every matching step, excluding timing only. The proof
is within one strict local process and does not establish cross-process recovery
for this new objective. Its 14,400-second ceiling includes setup and comparison.

## Readout and decision

Report both channels separately and together on fitting diagnostics, all
development checkpoints and the full reused main benchmark: fresh primitives,
familiar compositions and held composition partitions. Include every family's
paired actions/replies, primitive operator, depth and length; known-target
correctness, unknown asking, unsupported asking, generated parseability, and
blank/reset-history controls. Retention is the fixed-bank trajectory through
continued training, not an isolated no-practice forgetting experiment.

Use the same conservative architecture-study screen for this development
comparison: primary fresh-primitive and held-composition panels must improve in
every paired seed on the action/reply mean; pooled family/modality differences
must all be nonnegative, and no candidate seed/family/modality may be zero.
Report the complete screen and all regressions even if the aggregate improves.
Uncertainty and retention outcomes remain necessary review evidence and cannot
be traded away silently for capability. This screen alone authorizes no automatic
checkpoint promotion. Three seeds replicate initialization on shared banks;
they are not three independent curricula, and no general-intellect claim follows.

If the candidate fails, preserve the negative evidence and choose a new shared
learning hypothesis from the complete results. If promising, validate on fresh
lessons and the broader independence/retention requirements before connecting it
to the existing repeated-cycle controller. Keep the external tutor separate;
this comparison has zero external LLM calls and zero teacher inference assistance.

## Prior evidence

- [Architecture results](FOUNDATION_VARIANT_RESULTS.md).
- [Shared-feature diagnostic](FOUNDATION_SHARED_DIAGNOSTIC_RESULTS.md).
- [Fixed gradient protocol](FOUNDATION_GRADIENT_DIAGNOSTIC.md) and
  completed gradient result (archive reference: `../runs/foundation-gradient-diagnostic-local/attempt-001/result.json`),
  SHA256 `875b8d6cfbd189f720fa08d00bd33a6b228c4d7f5e0560c75300b450294cbd92`.
