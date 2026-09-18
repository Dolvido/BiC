# Shared reply-objective comparison: completed results

The balanced-reply recipe did **not** meet the fixed adoption screen. All three
criteria failed on both primary panels. Preserve both recipes as experimental
checkpoints; neither is an accepted general learner. The broad home-learning
goal remains open.

The [frozen protocol](FOUNDATION_OBJECTIVE_STUDY_PROTOCOL.md) compared the same
flat observation-only learner and identical admitted lessons, changing the reply
loss reduction to equal semantic classes and equal utterances within each class.
Each recipe received its own fixed development calibration: baseline learning
rate 0.003, balanced reply 0.001. This is a comparison of calibrated recipes,
not a fixed-rate causal isolation of the loss reduction.

## Verified completion

All six main jobs completed 3,072 updates. Canonical consumed-prefix verification,
checkpoint restores and the declared evaluation finished before score inspection.
The dispatcher completed on September 17, 2026 at 12:13:20 UTC.

- Dispatcher (archive reference: `../runs/foundation-objective-study-local/execution.json`), SHA256
  `e478ed2dc8d0e2bcc79b8b0259ea222687256e73fb2929d5aabb9228f759627b`.
- Main terminal receipt (archive reference: `../runs/foundation-objective-study-local/study/main/stage.json`), SHA256
  `6a2936f07bbc93565261c55852f860db8b02896df79eb26a88ea2ffc7cd20d79`.
- Official summary (archive reference: `../runs/foundation-objective-study-local/study/main/summary.json`), SHA256
  `b98b7fc1fed56b343076cd44f0ce833e8dd9e424d66f9ecbf030182f35ef2dac`.
- Independent descriptive analysis (archive reference: `../runs/foundation-objective-analysis-local/completed-analysis-001.json`), SHA256
  `571c5ced98d78063e2d97deb672bb166860927bbcfa9d0aa8d17c35d84923da5`.

The source-pinned analyzer reauthenticated all stage artifacts and score coverage,
reproduced the official screen exactly, and rehashed inputs before publishing.
It performed no model imports, checkpoint deserialization, inference or training.
All detailed cells, nine development checkpoints, controls and cost records are
retained in that analysis. A second read-only review confirmed the decision.

## Paired learning and transfer

Entries are successful action pairs / successful reply pairs. Each successful
pair requires both opposite-answer members to be correct. Fresh primitives have
864 pairs per job; each composed panel has 1,152. Fit uses 1,008 sampled actual
training pairs, including 144 direct-fact pairs; it is not a training census.

| Recipe / seed | Fresh primitives | Familiar compositions | Held compositions | Sampled fit | Direct-fact fit |
|---|---:|---:|---:|---:|---:|
| Baseline 8472 | 71 / 2 | 46 / 0 | 51 / 1 | 64 / 2 | 10 / 0 |
| Balanced 8472 | 71 / 4 | 51 / 22 | 57 / 28 | 64 / 16 | 12 / 0 |
| Baseline 8473 | 33 / 63 | 49 / 104 | 55 / 86 | 42 / 77 | 5 / 10 |
| Balanced 8473 | 60 / 26 | 55 / 55 | 49 / 61 | 50 / 41 | 8 / 6 |
| Baseline 8474 | 48 / 59 | 81 / 87 | 69 / 73 | 54 / 77 | 4 / 11 |
| Balanced 8474 | 92 / 106 | 123 / 132 | 129 / 128 | 116 / 125 | 19 / 18 |

The candidate regressed on the action/reply mean in seed 8473 for both primary
panels. Pooled fresh-primitive color replies fell from 75 to 71 of 864 pairs;
pooled held-composition count actions fell from 1 to 0 of 1,152. Candidate
seed/family/modality zero results remained. Thus all three criteria fail in each
primary panel, despite positive aggregate gains and a stronger third seed.

Across all three audit panels and seeds, each family has 3,168 pairs:

| Family | Baseline action / reply | Balanced action / reply |
|---|---:|---:|
| Color | 204 / 291 | 353 / 325 |
| Count | 4 / 0 | 6 / 6 |
| Switch | 295 / 184 | 328 / 231 |

Weak direct-fact fit means the limitation cannot be treated solely as transfer
to new compositions. Blank-input and reset-history controls achieve zero
successful pairs; that demonstrates dependence on observations/history, not
mastery of them.

## Accuracy, uncertainty and retention

Across audit seeds, ordinary known-answer correctness rises from 29.1–29.8%
to 41.9–43.9%, and unsupported asking falls from 38.2–39.7% to 21.0–25.9%.
Action/reply agreement falls from 68.8–85.1% to 50.0–63.7%. Unknown asking is
mixed across seeds. Every generated query reply remains parseable; syntactic
success does not establish correct coordinated decisions and replies.

The fixed final interval, updates 2,688 to 3,072, shows a shared stability
concern. These changes use all 90 development cells. Known correctness is the
equal-cell target-balanced metric; asking changes use pooled query counts.
All values below are percentage-point changes, not relative percentages.

| Recipe / seed | Target-balanced known | Unknown asking | Unsupported asking |
|---|---:|---:|---:|
| Baseline 8472 | −13.28 | +29.95 | +25.09 |
| Baseline 8473 | −12.13 | +27.68 | +23.14 |
| Baseline 8474 | −10.64 | +25.11 | +20.15 |
| Balanced 8472 | −2.96 | +8.07 | +3.41 |
| Balanced 8473 | −10.72 | +24.07 | +17.76 |
| Balanced 8474 | −3.55 | +11.98 | +8.55 |

These are fixed-endpoint observations under continued mixed training. They do
not isolate forgetting, stream-distribution changes, insufficient exposure or
retrieval failure. No best observed checkpoint replaces the declared endpoint.

## Cost and next decision

The complete formal study used 21,492 optimizer updates and 2,063,232 episode
exposures, including calibration. Dispatcher elapsed time was 9,967.5 seconds
(2.77 hours), enclosing preparation, training, verification and scoring. This
is not a claim about exclusive GPU time. Baseline main training took
1,123.6–1,126.8 seconds/job; balanced reply took 1,147.9–1,152.8 seconds/job.
The separate prerequisite runtime proof used 108 updates and 10,368 exposures.
Earlier studies and engineering validation are additional costs.

The next shared-learning hypothesis is **stable acquisition and use of state
across the curriculum**. The completed
sealed-metadata analysis (archive reference: `../runs/foundation-acquisition-analysis-local/metadata-001.json`),
SHA256 `e15f0127da0be0f257aa93f53fce4f5693ef5953db614d624828c0946e79dddd`,
examines twelve fixed descriptive windows without importing a model or generating
lessons. It finds equal depth coverage in both 384-update tail halves: 64 bundles
per depth and 12,288 episodes per family. Unknown query targets change only from
30,020/57,818 (51.9215%) to 30,180/58,024 (52.0130%). All 9,216 family microbatches
contain all three query classes. A large aggregate target-mixture jump is not
supported; these frequencies are not the action loss's equal-class weights.

The endpoints occupy different phases of the recurring depth/length pattern:
the last bundles at 2,688 and 3,072 both have depth 5, but 8 and 10 turns,
respectively. That is a possible recency covariate, not proof of an order effect.
The metadata does not contain per-query causal distances. A prospective
all-family contextual-sensitivity diagnostic can test controlled binding,
renaming and history placement. Any actual learning intervention then needs a
matched unchanged-training control. Do not select another architecture, loss
or subject-specific patch as though the mechanism were already established.

Parallel lesson preparation remains a separate execution-efficiency candidate.
Its CUDA identity check and a complete-cost throughput measurement are required
before adoption; neither engineering check would prove a capability gain.

All benchmark banks here were previously viewed. The three seeds replicate
initialization on the same lessons and banks, not independent curricula. There
were no external LLM calls or teacher assistance at evaluation. Broader grounded
skills, useful tutor independence, adaptive-curriculum benefit, sustained
retention and the complete home-learning loop remain unproven.
