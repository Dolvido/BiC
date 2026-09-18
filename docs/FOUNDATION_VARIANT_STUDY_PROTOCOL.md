# Prospective comparison: flat bytes versus learned utterance summaries

This protocol follows the completed [foundation pilot](FOUNDATION_PILOT_RESULTS.md).
Both previous orders fitted primitives poorly. The single hypothesis is that
parallel local byte encoding followed by causal attention over learned utterance
summaries improves shared acquisition and transfer under a home-computer budget.
It is not a diagnosis of the old failure. Optimization, objective balance,
exposure, binding and output instability remain competing explanations.

## Matched learners and intentional differences

Compare explicit `flat` and `hierarchical` variants at width 192, four blocks,
four heads, feedforward width 768, maximum 12 turns and the original byte limits.
Each paired seed gives every corresponding parameter the exact same initial
tensor. Both have 2,221,738 parameters, identical action/reply heads, AdamW,
gradient clipping and the unchanged three-domain sequence objective.

The hierarchy applies two shared causal blocks independently to each observed
utterance in a real batch, gathers its learned EOS state, then applies two
causal blocks over turns. Global byte positions are preserved. No parser,
domain tag, target, teacher representation or reply prefix enters the encoder.
Local next-byte prediction remains causal. See the
[candidate's limits](FOUNDATION_HIERARCHICAL_ENCODER_CANDIDATE.md).

This changes compression, cross-turn depth (four to two blocks), auxiliary
receptive field and gradient routing together. Equal parameters, updates and
data do not mean equal computation. Report measured end-to-end time, nested
materialization/step time, preparation, verification, scoring and GPU memory.

Both variants use the same fixed curriculum order: ascending depth with
rehearsal every four updates, followed by a common mixed tail. This is a
controlled choice to test one representation; the earlier one-seed order result
does not establish a winning schedule. No old trained weights are reused.

## Finite budget and calibration

| Phase | Plan seed | Stage updates × six | Mixed tail | Model seeds | Rates per architecture |
|---|---:|---:|---:|---|---|
| Calibration | 841000001 | 64 | 126 | 8461 | .0003, .001, .003 |
| Main | 842000001 | 384 | 768 | 8462, 8463, 8464 | Independently selected once |

Each update uses one 32-episode complete-pair microbatch from each of color,
count and switch: 96 episodes/update. Ordering seeds are 8410 and 8420.
Calibration is six fresh 510-update jobs; main is six fresh 3,072-update jobs.
The total formal budget is **21,492 updates / 2,063,232 episode exposures**.
Runtime proofs, engineering tests, failures and scoring are separate work.
No automatic extension, retry, early stopping or checkpoint promotion is allowed.
Before any preparation or gradients, implementation preflight corrected the
initial proposed 128-update calibration tail to 126: the unchanged planner
requires equal coverage of six depths. No plan semantics were changed.

Calibration ranks only the completed endpoint on the same 90 development cells:
63 fresh realizations and 27 held-development compositions. For each cell,
average final paired action accuracy, final paired generated-reply accuracy,
and target-balanced known action accuracy (mean accuracy for targets 0 and 1).
Average cells within each domain, then the three domains equally. Round this
engineered scalar to 12 decimal places; an exact tie chooses the lower learning
rate. Preserve all three domain/modality/ASK vectors. The ranking is an explicit
tuning heuristic, not a mastery score. All six calibration endpoints and their
canonical verification must complete before any calibration scoring. Select a
rate separately for each architecture using the same rule and budget. This short
calibration is approximately one sixth of the main horizon and may not choose
the best rate for longer training; choices remain frozen regardless.

Calibration scores never use main development or audit banks. Main checkpoints
are 0, 384, 768, 1,152, 1,536, 1,920, 2,304, 2,688 and 3,072. All six main
endpoints and checkpoint verification must finish before their scoring.
The primary checkpoint is **3,072**; no best-checkpoint selection.

## Accurate curriculum and reserved observations

Use the unchanged compact foundation recipe grammar and independent canonical
validators. The pinned completed foundation summary authenticates all 424 input
files and 79 sources. Historical exclusions are its protected observations
union its actual training transcripts; this includes the older study history.

Admit calibration first against history, then main against history plus
calibration training. Only the existing first-safe deterministic names-only
repair is allowed. Preserve procedures, values, targets, replies and schedules.
Every rejected admission candidate remains recorded. Generate evaluation after
both training plans exist, excluding history, both training streams, and every
previously admitted evaluation transcript.

Evaluation seeds are 843000001 (calibration development), 844000001 (main
development), and 845000001 (main audit). Development has 16 complete pairs/cell;
audit has 32. Main fitting evidence uses the first 16 canonical admitted anchors
per cell; it is a capped endpoint sample of actual training, not a training
census. Main fresh-audit generation can use 32 anchors/cell. Preserve every
domain, depth, primitive operator and 8/10/12-turn length.

Independent data verification reconstructs new admission, training manifests,
anchors and banks once before formal gradients. Each process builds a canonical,
source-bound plan index before model creation; it is never loaded from a cache
file. Later checkpoint restores reuse that authenticated index to avoid
regenerating every consumed prefix. Source changes or altered artifacts fail
closed. Historical checkpoint hashing is not repeated at every ordinary load;
the frozen historical gate and actual transcript dependencies remain bound.

## Outcomes and interpretation

Report exact paired action/reply counts separately for actual fitting anchors,
fresh primitive answers, fresh composed answers and held compositions. Preserve
known-answer accuracy, unknown ASK recall, unsupported ASK, per-target
confusions, generated-reply agreement and every length/operator cell. Aggregate
query accuracy cannot substitute for known-answer or opposite-pair success.

The primary acquisition panel is fresh audit depth 0 and depth 1 (direct,
COPY, ADVANCE); transfer is held audit composition at depths 2–5. A *promising
comparison* screen requires, in **both** panels:

1. The mean of paired action/reply rates improves for the hierarchy in each of
   the three paired initialization seeds.
2. Across seeds, no domain × modality (three domains × action/reply) has a
   negative paired-rate difference.
3. The hierarchy has nonzero paired action and paired reply success in every
   domain in every seed.

This is a replication/noncollapse screen, not mastery, statistical significance,
general intelligence or a deployment threshold. A zero or weak operator still
limits acquisition claims even when the screen passes. Report all differences,
including when the screen fails. Three seeds measure initialization variation
on one fixed lesson stream and do not replicate the data-generating process.

For retention, use the prescribed post-stage checkpoint 384 × (depth + 1),
then 2,304 and 3,072; never use each cell's best checkpoint. Report the additional
2,688 midpoint and both modalities' common-tail changes. Continued mixed training
is not a no-training forgetting control. Known/unknown response tradeoffs and
retention deterioration must accompany any positive endpoint comparison.

At the final main checkpoint, score blank-input and reset-history controls on
the entire audit set with both action and BOS-only generated replies. They
establish dependence on available input/history, not mastery. All models use
the same evaluation batches and raw context limits.

## Execution and practical limits

A separate strict local CUDA proof must cover both architectures and all three
rates before formal preparation/launch. It compares independent repeats,
mid-run optimizer reloads, inputs and scoring under the declared runtime.
The experiment uses one GPU and may run the two architecture workers in
parallel; overlapping wall times must not be summed as dedicated GPU hours.

Each worker reuses its authenticated index across its three fresh learners.
Official checkpoints bind architecture, initialization, rate, plan, sources,
runtime, weights, optimizer, consumed evidence and complete work journals.
CPU verification validates these states and canonical input evidence; it does
not reproduce every past gradient. Scoring binds the exact verified weights.

The grammar remains finite and synthetic. Held syntactic ancestries share
primitive anchors and can have semantic equivalences. Distractors, English
variety, grounded knowledge, useful computer work, learned curriculum choice,
ethical self-modification and progressive external-teacher independence remain
larger objectives. This comparison introduces no autonomous model promotion.
