# Foundation-order pilot: completed results

The prescribed curriculum produced more correct paired actions at the final
checkpoint than the mixed order, but **neither learner established reliable
primitive acquisition or usable paired English replies**. The common mixed tail
reduced paired reply correctness broadly. These results motivate one shared
representation experiment; they do not support promotion or autonomous lesson
selection.

This report uses the completed, verified
summary (archive reference: `../runs/foundation-study-local-v2/evaluation/summary.json`), SHA256
`188b0d026c4debce596f3cca5a1b64f6f5f0e0e3992dd4174904a33b6c3e87c2`.
The analysis rehashed all **424 summary-bound input files and 79 frozen sources**;
every hash matched. It derives counts only, without inference, checkpoint
deserialization, training or another canonical stream replay. See the
analysis provenance (archive reference: `../runs/foundation-analysis-local-v2/provenance.json`).

## Matched comparison and limits

Both fresh learners used the same width-192 model initialization and learning
rate .001, 1,536 updates, and 147,456 episode exposures. Each update contained one
32-episode complete-pair microbatch from each of color, count and switch. Both
orders consumed the same admitted lessons, targets, values and names. Sparse
naming-only admission repairs preserved procedures, supervision and schedules.

The first 1,152 updates differed in order: ascending causal depth with prescribed
rehearsal versus a shuffled prefix. The final 384 updates had the same mixed
suffix. Intermediate equal-update comparisons therefore have different depth
exposures by design; at 1,152 both have consumed the same prefix multiset. The
tail includes further training and cannot serve as a no-training retention
control.

This is **one shared initialization and one finite order comparison**. It does
not isolate the effect of adding primitive lessons, since both arms received
them. Cells, checkpoints and pairs are not independent seed replications.
Fresh examples change names and values together; their effects are not separated
here. Composed panels test specified held syntactic ancestries, with shared
primitive anchors. Color cycles and switch parity allow semantic equivalences;
the held boundary is not evidence of learning a new algorithm. Distractors have
disjoint roles, so adversarial interference robustness is untested.

The [prospective interpretation guide](FOUNDATION_PILOT_INTERPRETATION.md) was
written before reading these scores. No retrospective winner threshold or
checkpoint selection is introduced.

## Endpoint fitting and transfer

Each entry below is **correct final opposite pairs / total pairs**. Both members
must be correct. Reply results use generated replies with BOS-only decoder
prefixes. Fitting anchors are capped samples of actually observed training
realizations, not a census of training or a fitting curve.

| Evidence | Curriculum action | Curriculum reply | Mixed action | Mixed reply |
|---|---:|---:|---:|---:|
| Observed fitting anchors | 118/1,008 | 12/1,008 | 60/1,008 | 0/1,008 |
| Fresh audit realizations | 100/1,008 | 10/1,008 | 62/1,008 | 3/1,008 |
| Held-composition audit | 67/576 | 9/576 | 37/576 | 1/576 |

Fresh panels include direct facts, COPY, ADVANCE and depths 2–5. Held audit
composition covers depths 2–5; the development held panel covers depths 3–5
because depth 2 has no separate development motif. Their totals are therefore
different and must not be compared as identical banks.

The audit action advantage is descriptive: curriculum has 167/1,584 correct
final pairs versus 99/1,584, while paired replies remain 19/1,584 versus 4/1,584.
Unpaired final-known action accuracy is similar: 1,516/3,168 versus 1,526/3,168.
The pair requirement reveals failure to follow opposite premises that an
approximately half-correct single-answer rate can conceal.

All nine primitive domain/operator groups are shown below. **Every entry has
48 pairs**: three lengths with 16 pairs each. Each value is *action-correct /
reply-correct*, not a fraction whose denominator is the second number.

| Domain and primitive | Curriculum fitting A/R | Mixed fitting A/R | Curriculum fresh audit A/R | Mixed fresh audit A/R |
|---|---:|---:|---:|---:|
| Color: direct fact | 9 / 0 | 8 / 0 | 6 / 0 | 4 / 0 |
| Color: COPY | 8 / 1 | 1 / 0 | 7 / 0 | 4 / 0 |
| Color: ADVANCE | 11 / 0 | 4 / 0 | 7 / 1 | 2 / 0 |
| Count: direct fact | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 |
| Count: COPY | 0 / 1 | 0 / 0 | 2 / 0 | 0 / 0 |
| Count: ADVANCE | 0 / 0 | 0 / 0 | 0 / 1 | 0 / 2 |
| Switch: direct fact | 6 / 0 | 9 / 0 | 6 / 0 | 5 / 0 |
| Switch: COPY | 8 / 1 | 3 / 0 | 7 / 0 | 7 / 0 |
| Switch: ADVANCE | 7 / 1 | 3 / 0 | 4 / 0 | 8 / 0 |

Across these groups, primitive fitting reaches only 49/432 paired actions and
4/432 replies for curriculum, versus 28/432 and 0/432 for mixed. Poor held
composition therefore cannot yet be assigned to a composition-specific defect.
One additional correct pair in an individual 16-pair cell changes its rate by
6.25 percentage points; small differences are not precise effect estimates.

For context, known-answer correctness on the entire fresh audit panel is
1,691/4,710 versus 1,669/4,710; unsupported ASK occurs on 1,719 versus 1,656 of
those known questions. On held composition, known correctness is 917/2,436
versus 858/2,436 and unsupported ASK 809 versus 856. Correct ASK on truly unknown
queries is much higher: 4,168/4,932 versus 4,050/4,932 on fresh audit, and
2,048/2,454 versus 2,039/2,454 on held composition. These different denominators
must remain visible rather than being hidden in aggregate query accuracy.

## All development checkpoints

The fixed development set has 90 cells and **1,440 final pairs** per checkpoint.
The table gives action-correct and reply-correct counts separately. All cells,
including every length and operator, are retained in the
development count table (archive reference: `../runs/foundation-analysis-local-v2/development_cells.csv`).

| Updates | Curriculum action | Curriculum reply | Mixed action | Mixed reply |
|---:|---:|---:|---:|---:|
| 0 | 39 | 0 | 39 | 0 |
| 192 | 165 | 0 | 50 | 71 |
| 384 | 94 | 36 | 8 | 119 |
| 576 | 116 | 65 | 76 | 73 |
| 768 | 72 | 14 | 116 | 77 |
| 960 | 77 | 8 | 78 | 114 |
| 1,152 | 105 | 160 | 132 | 112 |
| 1,536 | 147 | 17 | 85 | 1 |

The relative order effect changes through training and differs between actions
and replies. It is not a steady curriculum-acquisition advantage. Count paired
actions remain essentially zero across every depth and length, while color and
switch fluctuate. At the endpoint, the curriculum's 147 action pairs comprise
71 color, zero count and 76 switch pairs; mixed has 31, zero and 54. Each domain
has 480 pairs in this combined development panel.

## Fixed-reference retention and the common tail

The curriculum's declared post-stage reference for depth `d` is update
`192 × (d + 1)`, not that cell's best checkpoint. Exact per-cell action, reply,
known/unknown, earlier-opposite-pair and unsupported-ASK counts at that reference,
1,152 and 1,536 are preserved in
retention_references.csv (archive reference: `../runs/foundation-analysis-local-v2/retention_references.csv`).
It contains all 180 arm/cell records. Mixed uses the same calendar references for
comparison; those are not completion times for its shuffled depth practice.

The curriculum's fresh primitive references are already weak. Across all three
domains and lengths, each row below has 144 pairs; values are A/R correct counts.

| Primitive | Declared reference | At 1,152 | At 1,536 |
|---|---:|---:|---:|
| Direct fact, reference 192 | 8 / 0 | 9 / 11 | 12 / 0 |
| COPY, reference 384 | 9 / 4 | 11 / 17 | 15 / 0 |
| ADVANCE, reference 384 | 9 / 5 | 7 / 17 | 9 / 1 |

This does not establish earlier reliable primitive skills followed by wholesale
forgetting. Some paired replies emerge later and subsequently deteriorate.

During the identical final mixed suffix, curriculum fresh action pairs rise
77→101/1,008 and held-development actions 28→46/432, while replies fall
109→12/1,008 and 51→5/432. Mixed actions fall 92→58/1,008 and 40→27/432;
replies fall 77→1/1,008 and 35→0/432.

The reply decline spans all context lengths. Curriculum fresh reply counts at
8/10/12 turns change **51/35/23 → 7/4/1**, each length denominator 336. Mixed
changes **36/26/15 → 0/1/0**. Action length patterns vary rather than showing a
simple monotonic penalty: curriculum fresh 8-turn action pairs fall 41→26/336
while 10-turn and 12-turn counts rise 31→39 and 5→36.

Every post-training development checkpoint has 13,038/13,038 parseable query
replies. During the common tail, overall query-reply correctness nevertheless
rises from 7,530→7,767/13,038 for curriculum and 7,606→7,736 for mixed while
final paired replies deteriorate. Known action correctness falls
2,497→2,292/6,474 and 2,413→2,258; unsupported ASK rises 1,836→2,341 and
1,974→2,290 on those known questions. Correct ASK on unknown queries rises
4,919→5,499/6,564 and 4,917→5,388. Thus aggregate gains can hide the failure to
answer known opposite premises.

These patterns support behavioral instability and response-bias hypotheses.
They do not identify the responsible operation, prove constant-output collapse,
or establish that a hidden acquired skill survived the decline. Different
action/reply decision boundaries remain possible; this is not simply a
parseability problem.

## Controls

For both arms, blank-text and reset-history audit controls each yield **zero
final action pairs and zero final reply pairs out of 1,584**. They establish
dependence of the observed paired successes on available input/history, not
mastery. Reset-history aggregate query accuracy remains near one half because
unknown questions can be answered ASK while known questions fail. Control
denominators and full count vectors remain in
endpoint_cells.csv (archive reference: `../runs/foundation-analysis-local-v2/endpoint_cells.csv`).

## Compute and verification costs

The formal comparison retained **3,072 optimizer updates and 294,912 episode
exposures**, with no failed/discarded formal updates reported. Each arm received
49,152 episodes per domain. Actual observation-byte/token and reply-target
exposures match between arms and are retained in
counts.json (archive reference: `../runs/foundation-analysis-local-v2/counts.json`).

| Recorded scope | Seconds |
|---|---:|
| Successful formal preparation | 757.609 |
| Curriculum worker wall, including setup | 622.359 |
| Mixed worker wall, including setup | 619.234 |
| Canonical verification wall | 2,859.985 |
| Evaluation wall, including preparation/scoring/validation | 171.281 |

The workers overlapped. Adding their wall times does not measure elapsed wall
time or dedicated GPU hours. Curriculum retained step time is 557.805 seconds,
including 230.088 seconds of lesson materialization; mixed is 553.595 including
227.434 seconds. Materialization is therefore approximately **41% of retained
step time** in each arm and must not be added again. Packing and other work are
not fully isolated by that measurement. Canonical verification is a substantial
separate cost; its time is not neural training.

Execution proofs, engineering tests, the preserved failed original preparation,
and subsequent loop/component validation have separate records. They are not
included in the 3,072 formal updates and cannot be silently treated as free or
folded into a capability result. This report's derivation used only file hashing
and arithmetic; its own cost is recorded in the analysis provenance. It did not
repeat the expensive canonical verification.

## One next hypothesis, without promotion

Select the prospective guide's **weak primitive fitting and fresh acquisition**
branch: compare the current flat byte encoder with one **learned
utterance-summary encoder feeding causal cross-utterance reasoning**. Apply the
same shared mechanism to all domains, operators, depths and lengths. Keep raw
observations, canonical teaching targets, independent evaluation and both action
and generated-reply measures. No domain-specific head, parser/oracle input,
teacher reply prefix or external LLM policy belongs in the encoder.

The [candidate design](FOUNDATION_HIERARCHICAL_ENCODER_CANDIDATE.md) separates
local causal byte encoding from causal attention over learned utterance vectors.
Implementation preparation is now assigned, but **no new formal training
comparison has been selected or launched**. It still needs independent causal
and restart tests, a frozen hypothesis and recipe, symmetric development-only
calibration, fresh initialization and held examples. Equal parameters/exposure
would not imply equal computation; actual preparation, training, scoring and
memory costs must be measured.

The hypothesis is not a diagnosis. Competing explanations include objective and
gradient imbalance, optimization under this budget, inadequate exposure to
independent realizations, alias/value binding, and insufficient memory or credit
assignment. Summarization may itself discard details needed for exact binding;
changing the auxiliary byte predictor's receptive field also changes gradient
routing. Those limits must be declared instead of attributing any future gain
to compression alone.

There is no automatic promotion, general-intellect claim or claim of learned
self-direction. The useful outcome of this pilot is a verified negative result
on broad acquisition, a limited count-backed order effect, and one shared next
comparison rather than a collection of subject-specific repairs.

## Derived evidence

- development_cells.csv (archive reference: `../runs/foundation-analysis-local-v2/development_cells.csv`):
  1,440 arm/checkpoint/cell rows with exact counts, identity and agreement rates.
- development_overview.csv (archive reference: `../runs/foundation-analysis-local-v2/development_overview.csv`):
  all 16 arm/checkpoint totals.
- retention_references.csv (archive reference: `../runs/foundation-analysis-local-v2/retention_references.csv`):
  180 cells' prescribed references and both later checkpoints.
- endpoint_cells.csv (archive reference: `../runs/foundation-analysis-local-v2/endpoint_cells.csv`):
  720 endpoint fitting/audit/control rows, preserving all groups and lengths.
- counts.json (archive reference: `../runs/foundation-analysis-local-v2/counts.json`),
  derive.py (archive reference: `../runs/foundation-analysis-local-v2/derive.py`) and
  provenance.json (archive reference: `../runs/foundation-analysis-local-v2/provenance.json`):
  reproducible arithmetic, source-summary identity, verified input manifest and
  hashes of every derived artifact. No raw predictions are duplicated.
