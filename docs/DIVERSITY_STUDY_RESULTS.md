# Curriculum diversity study — completed September 16, 2026

**Broader naming practice improved generalization within the trained subjects,
but this comparison did not establish better acquisition of the withheld subject
or adequate retention. No checkpoint was promoted.** All four fixed main runs
and five arithmetic adaptations completed locally. The study ran 14,720 updates
and 942,080 repeated episode exposures. Its
[prospective protocol](DIVERSITY_STUDY_PROTOCOL.md) defined no numerical promotion
gate; this is an evidence-based development decision after the complete audit.

Across the three old-subject audit panels, the broadest bank reached 62.05%
macro query accuracy before adaptation versus 52.05% for the smallest bank.
The outcome depended strongly on which factor changed: more naming maps helped
withheld-name panels, while more worlds helped new worlds under familiar names.
Later paired decisions remained weak, and every pretrained arm had a lower
integrated arithmetic query learning curve than fresh initialization. The next
step is a general continual-learning and representation comparison using varied,
verified lessons and explicit retention, not a subject-specific repair.

The verified summary (archive reference: `../runs/diversity-study-local/audit/summary.json`),
complete audit (archive reference: `../runs/diversity-study-local/audit/report.json`) and
decision record (archive reference: `../runs/diversity-study-local/decision.json`) retain exact
metrics and provenance. This one-seed result is not a robust capability claim.

## What the comparison tests

The same 753,610-parameter causal sequence learner receives a two-by-two
curriculum comparison: 32 or 256 distinct anonymous counterfactual world pairs
per family, rendered through one or eight independent bijective naming maps.
These are finite banks, not continuous generation during optimization. Every
arm has 2,048 logical pair slots per family and the same slot sampling stream.
Lower-diversity arms intentionally repeat worlds and names more frequently.

Binding, directed paths and conditional logic are trained at levels 1 and 2.
Every arm starts from seed 2601 and receives 3,600 updates with batch size 64
and learning rate 0.001. The endpoints are fixed; development measurements do
not select checkpoints. Model, objectives and optimizer are unchanged across
arms. The 14,400 main updates represent 921,600 repeated episode exposures.
No pretrained weights or external teacher participate in this comparison.

The new curriculum constructs anonymous event programs and independently
checks their rendered English using the existing text interpreter. Both
counterfactual variants have identical questions and event order, with one
changed statement and at least one opposite known answer. Programs, labels,
replies, family names and provenance remain outside the observed input stream.
The model sees only the six utterances of raw English, in causal order.

## Separate world and naming panels

Development and later retention audits distinguish familiar worlds rendered
with withheld primary name combinations, unfamiliar worlds under a familiar
mapping, and unfamiliar worlds with withheld naming combinations. Each panel
has 128 episodes per trained family. A common training-fit bank uses the first
32 worlds and first naming map, familiar to every arm but presented at different
frequencies. Level-3 audit worlds provide a separate composition assessment.

Anonymous pair fingerprints and primary ordered name pairs have independent
partitions. A paired fingerprint alone cannot prevent the same individual
transcript appearing with a different counterfactual partner. Preparation
therefore rejects a whole novel-world pair if either alias-normalized member
overlaps an earlier world pool, and separately checks exact rendered transcript
separation. Naming-only panels deliberately reuse their declared training
worlds. These checks establish the stated syntactic boundaries; they do not
prove semantic equivalence classes or statistically independent worlds.

## Adaptation, retention and controls

After the four main endpoints exist, each arm and a common fresh seed-2601
control learn from the same 128 arithmetic support episodes. They receive
fresh optimizers, the same sampling seed and cumulative budgets of 0, 1, 4,
16 and 64 updates. Independent arithmetic levels 2 and 3 each contain 256
episodes. The five adaptations added 320 updates and 20,480 repeated exposures,
making the complete study 14,720 updates and 942,080 exposures.

Arithmetic learning curves must be interpreted alongside retained old-subject
panels, advanced compositions, opposite-answer pairs, later known decisions,
uncertainty, freely generated replies and action/reply agreement. Blank-text
and reset-history controls test dependence on the available context. CPU
checkpoint restarts must reproduce actions and free replies, with parent
checkpoints unchanged. Ordinary gradient adaptation is not a demonstrated
meta-learning algorithm.

## Actual diversity and fitted behavior

All arms contained 4,096 logical episode slots per family. The actual distinct
rendered transcript counts were 64/512/512/4,096 for binding and graphs in arm
order below; conditional counts were 64/512/511/4,088. Conditional anonymous
individual transcript counts were 64 for the 32-world pool and 509 for the
256-world pool. Some distinct pairs therefore share individual members inside
training. The complete counts and class distributions are preserved in the
diversity record (archive reference: `../runs/diversity-study-local/diversity.json`).

Each family received 76,800 episode exposures. Nominal average presentations per
world-and-name variant were 1,200 / 150 / 150 / 18.75 respectively; duplicate
individual transcripts can receive more. The common fit subset is the same
32 worlds under the first mapping, not an estimate over each entire bank.

| Arm | Common fit query | Common fit pairs | Dev names query | Dev worlds query | Dev both query |
|---|---:|---:|---:|---:|---:|
| w32-n1 | 100.00% | 100.00% | 49.70% | 65.15% | 43.71% |
| w32-n8 | 99.22% | 97.06% | 71.53% | 60.29% | 52.43% |
| w256-n1 | 99.48% | 98.04% | 51.95% | 79.38% | 49.48% |
| w256-n8 | 87.67% | 55.52% | 60.63% | 77.21% | 55.03% |

The smallest bank was perfectly fitted but transferred poorly. The broadest
bank was less completely fitted under the same exposure budget: its graph fit
paired accuracy was 23.53%, compared with at least 94% in the other arms. These facts
support both diversity and optimization questions; they do not identify one
cause or justify unlimited training on a fixed bank.

## Old-subject audit before arithmetic adaptation

Query percentages below use equal family weighting. The names panel reuses
familiar worlds; worlds uses a familiar naming map; both changes both factors.

| Arm | Names | Worlds | Both | All three panels |
|---|---:|---:|---:|---:|
| w32-n1 | 48.26% | 65.15% | 42.75% | 52.05% |
| w32-n8 | 69.36% | 60.16% | 52.00% | 60.50% |
| w256-n1 | 50.00% | 77.26% | 47.35% | 58.20% |
| w256-n8 | 56.21% | 75.48% | 54.47% | 62.05% |

At 32 worlds, eight naming maps improved audit names by 21.09 percentage
points and both by 9.24 points, while worlds fell 4.99 points. At 256 worlds,
names improved 6.21 points and both 7.12 points, while worlds fell 1.78 points.
More worlds improved the familiar-name worlds panel by 12.11 points with one
map and 15.32 points with eight. More diversity was therefore not uniformly
better across panels, families or paired decisions.

The following counts preserve the difficult slices. A paired decision counts
only when both opposite-answer variants are correct. Later-known queries exclude
unknown targets; conditional later queries are not all changed-outcome interventions.

Before adaptation, action/reply agreement on the joint-new panel was
98.44%/98.44%/99.22% for w32-n8 and 98.96%/93.75%/99.61% for w256-n8
(binding/graph/conditional). The sequence learner usually said and did the same
wrong thing; alignment alone is not the primary explanation of these errors.

### Familiar worlds, withheld naming combinations

| Arm | Family | Query correct | Opposite pairs | Later known correct | Later opposite pairs |
|---|---|---:|---:|---:|---:|
| w32-n1 | Binding | 217/384 | 16/76 | 75/128 | 3/18 |
| w32-n1 | Graph | 141/256 | 6/68 | 62/128 | 6/64 |
| w32-n1 | Conditional | 85/256 | 4/68 | 18/72 | 3/30 |
| w32-n8 | Binding | 301/384 | 47/76 | 107/128 | 9/18 |
| w32-n8 | Graph | 171/256 | 13/68 | 68/128 | 13/64 |
| w32-n8 | Conditional | 161/256 | 25/68 | 43/72 | 13/30 |
| w256-n1 | Binding | 180/384 | 11/76 | 55/128 | 2/18 |
| w256-n1 | Graph | 169/256 | 10/68 | 66/128 | 10/64 |
| w256-n1 | Conditional | 95/256 | 2/68 | 18/72 | 2/30 |
| w256-n8 | Binding | 214/384 | 21/76 | 68/128 | 7/18 |
| w256-n8 | Graph | 177/256 | 3/68 | 63/128 | 3/64 |
| w256-n8 | Conditional | 112/256 | 12/68 | 20/72 | 7/30 |

### Unfamiliar worlds, familiar naming map

| Arm | Family | Query correct | Opposite pairs | Later known correct | Later opposite pairs |
|---|---|---:|---:|---:|---:|
| w32-n1 | Binding | 218/384 | 26/69 | 59/106 | 3/12 |
| w32-n1 | Graph | 163/256 | 11/66 | 72/128 | 11/64 |
| w32-n1 | Conditional | 192/256 | 29/74 | 53/80 | 13/33 |
| w32-n8 | Binding | 180/384 | 14/69 | 45/106 | 0/12 |
| w32-n8 | Graph | 169/256 | 12/66 | 69/128 | 12/64 |
| w32-n8 | Conditional | 173/256 | 30/74 | 48/80 | 16/33 |
| w256-n1 | Binding | 281/384 | 39/69 | 55/106 | 3/12 |
| w256-n1 | Graph | 178/256 | 17/66 | 73/128 | 17/64 |
| w256-n1 | Conditional | 228/256 | 48/74 | 63/80 | 18/33 |
| w256-n8 | Binding | 286/384 | 39/69 | 51/106 | 2/12 |
| w256-n8 | Graph | 172/256 | 6/66 | 66/128 | 6/64 |
| w256-n8 | Conditional | 217/256 | 40/74 | 59/80 | 15/33 |

### Unfamiliar worlds and withheld naming combinations

| Arm | Family | Query correct | Opposite pairs | Later known correct | Later opposite pairs |
|---|---|---:|---:|---:|---:|
| w32-n1 | Binding | 170/384 | 7/69 | 45/106 | 0/12 |
| w32-n1 | Graph | 139/256 | 2/66 | 57/128 | 2/64 |
| w32-n1 | Conditional | 76/256 | 2/74 | 11/80 | 2/33 |
| w32-n8 | Binding | 173/384 | 10/69 | 47/106 | 2/12 |
| w32-n8 | Graph | 166/256 | 7/66 | 65/128 | 7/64 |
| w32-n8 | Conditional | 118/256 | 19/74 | 25/80 | 7/33 |
| w256-n1 | Binding | 163/384 | 8/69 | 39/106 | 1/12 |
| w256-n1 | Graph | 170/256 | 4/66 | 62/128 | 4/64 |
| w256-n1 | Conditional | 85/256 | 3/74 | 16/80 | 2/33 |
| w256-n8 | Binding | 191/384 | 16/69 | 41/106 | 4/12 |
| w256-n8 | Graph | 178/256 | 6/66 | 66/128 | 6/64 |
| w256-n8 | Conditional | 113/256 | 11/74 | 28/80 | 7/33 |

High-world/high-name graph accuracy can exceed its paired performance: on
new worlds with familiar names it answered 172/256 queries but solved only
6/66 opposite pairs, including 6/64 later pairs. On the joint-new panel its
binding later pairs were 4/12 and conditional later pairs 7/33. These are
limited gains in restricted tasks, not broad reliable reasoning.

## Withheld arithmetic learning and retention

Normalized sparse trapezoidal areas use budgets 0, 1, 4, 16 and 64. Query
accuracy is a macro over the two query levels; area differences are percentage
points against the same fresh seed. All pretrained query-area differences are
negative. Their final query endpoints are higher than the deteriorated fresh
endpoint, so the final score alone would give an incomplete account of learning.

Support uses one training naming map; queries combine new audit worlds and audit
naming combinations, and arithmetic introduces previously untrained digit bytes.
This is a combined transfer challenge, not isolated arithmetic acquisition.
Its absolute scores cannot be compared with the previous v2 audit as regression,
because the curriculum and support distribution changed.

| Arm | Query area | Area vs fresh | Pair area | Final query | Old query before → after | Old pair before → after |
|---|---:|---:|---:|---:|---:|---:|
| w32-n1 | 26.23% | -10.05 pp | 0.02% | 31.35% | 52.05% → 42.80% | 16.13% → 4.17% |
| w32-n8 | 29.95% | -6.33 pp | 0.24% | 28.81% | 60.50% → 46.35% | 27.50% → 19.57% |
| w256-n1 | 27.95% | -8.33 pp | 0.00% | 28.74% | 58.20% → 40.54% | 22.33% → 5.92% |
| w256-n8 | 33.70% | -2.58 pp | 0.07% | 34.08% | 62.05% → 46.77% | 24.06% → 15.51% |
| fresh | 36.28% | — | 0.14% | 25.03% | — | — |

Fresh initialization also failed paired reasoning: its initially higher curve
included constant-class decisions on balanced known targets. Negative transfer
relative to this control does not imply that the fresh learner mastered arithmetic.

At 64 updates, **every candidate, including fresh, solved 0/128 later-known
arithmetic pairs at both levels**. Level-2 query pairs were 0/256 except
w32-n8 at 1/256; level-3 query pairs were 0/256 except w32-n8 and w256-n8 at
1/256. These near-zero paired results dominate any claim of numerical competence.

| Arm | Level-2 query | Level-2 later known | Level-3 query | Level-3 later known | Unsupported level-3 ASK |
|---|---:|---:|---:|---:|---:|
| w32-n1 | 309/768 | 89/256 | 115/512 | 99/256 | 228/512 |
| w32-n8 | 357/768 | 110/256 | 57/512 | 5/256 | 156/512 |
| w256-n1 | 302/768 | 94/256 | 93/512 | 85/256 | 242/512 |
| w256-n8 | 354/768 | 104/256 | 113/512 | 9/256 | 164/512 |
| fresh | 272/768 | 70/256 | 75/512 | 64/256 | 326/512 |

Level-3 arithmetic has zero true unknown targets, so ASK recall is undefined
and the final column counts errors. The first arithmetic question already
follows an increase or decrease; the later question adds copied-state composition.

All four pretrained arms forgot earlier tasks. For the broadest bank, mean old
query accuracy fell 15.28 points and paired accuracy fell 8.55 points. Its
mean later-known accuracy rose only 0.80 points while broad performance fell,
illustrating why a favorable slice cannot stand in for retention. In its joint-new
panel, binding/graph/conditional later pairs changed 4/12→1/12, 6/64→1/64 and
7/33→1/33. No replay was included; the result does not test a retention mechanism.

## Advanced composition, replies and controls

Before adaptation, level-3 later paired decisions remained sparse:

| Arm | Binding | Graph | Conditional | Advanced macro query before → after adaptation |
|---|---:|---:|---:|---:|
| w32-n1 | 1/22 | 5/64 | 4/36 | 34.77% → 30.60% |
| w32-n8 | 1/22 | 8/64 | 5/36 | 43.62% → 35.29% |
| w256-n1 | 0/22 | 2/64 | 0/36 | 41.02% → 25.13% |
| w256-n8 | 0/22 | 2/64 | 2/36 | 44.79% → 37.37% |

Trained reply outputs were largely parseable, but correctness and agreement
could deteriorate. For w256-n8 on old graph panels, exact reply accuracy fell
22.66–25.39 points and action/reply agreement fell 29.69–37.89 points. In the
familiar-name conditional world panel, its exact reply accuracy fell
86.33%→58.59%. Parseable output is not retained understanding. Generated strings
were not retained in these reports, so a claim of constant-reply collapse is
not supported. At the arithmetic endpoint the pretrained models generated
parseable replies on every query; fresh did so on only 221/768 level-2 and
161/512 level-3 queries. This retained output format is distinct from learning
the correct numeric relation.

At the final arithmetic budget, resetting history reduced every pretrained
arm to 16.67% macro query accuracy and zero later-known accuracy. Blank-text
query accuracy was 11.59%, 26.89%, 5.86% and 2.08% in arm order. All blank and
reset controls had zero opposite-pair accuracy. Context affects outputs; this
control does not demonstrate correct use of the full context.

## Execution, validation and next decision

All four main jobs and the audit exited successfully on the local RTX 5080.
The observed execution envelope was 371 seconds (6 minutes 11 seconds):
267 seconds for main jobs and 104 for the audit, including completion polling.
Main worker training intervals sum to 819.063 seconds; adaptation training
adds 6.388 seconds. These concurrent/sparse timing records are not dedicated
GPU-hours or a measure of useful learning efficiency. Main peak allocated CUDA
memory ranged from 840.10 to 843.49 MiB, including cached banks and evaluation;
driver and other process memory are excluded. See the
execution record (archive reference: `../runs/diversity-study-local/execution.json`).

All 48 frozen sources and checkpoint digests were verified. All five serialized
CPU restarts exactly reproduced actions and generated replies; base checkpoints
remained unchanged. The full suite passed **494 tests in 37.347 seconds** and
`git diff --check` passed. The
validation record (archive reference: `../runs/diversity-study-local/validation.json`) preserves the
earlier 487-test pass. These engineering checks do not add capability evidence.

**Decision: no promotion; continue the broader research objective.** Preserve
the useful curriculum separation between worlds and naming, but test learning
across a wider, accurately verified set of operations with balanced replay or
consolidation and shared representations. Match new-subject exposures and account
explicitly for replay computation, or declare a fixed-compute comparison. Include
a longer-budget broad-bank control before attributing incomplete fitting to model
capacity. Compare fresh and pretrained learners across subject rotations and seeds,
tracking whole paired decisions and retention. A next protocol should distinguish
acquisition from forgetting and use fresh holdouts. Do not tune arithmetic-specific heads or reuse this
audit as untouched evidence. The independent regional adapter remains untrained.

## Interpretation boundaries

- This is one initialization and one withheld subject. The protocol defines
  no automatic promotion or retrospective numerical pass/fail gate.
- More world diversity can also change target and operation frequencies. The
  comparison estimates the declared bank treatment; it does not isolate one
  causal explanation for generalization.
- The English grammar is finite. Withheld naming combinations reuse familiar
  aliases and bytes, rather than introducing an unknown vocabulary.
- Conditional later questions sometimes follow irrelevant interventions or
  unchanged values. Aggregate later accuracy is not a uniform test of
  recomputing an altered outcome.
- Level 3 is not a pure scalar difficulty increase. Binding and arithmetic
  introduce copy-plus-update compositions; other families change path length
  or rule arity. Arithmetic level 3 has no unknown targets, unlike the lower
  support/query levels. Its digit bytes are absent from main-subject text.
- Syntactic world identity does not establish semantic novelty. Some different
  pairs share a single member within a training pool; actual transcript counts
  and declared cross-pool exclusions must accompany world counts.
- Throughput and repeated exposures are resource measurements, not useful
  learning per compute hour. Concurrent worker times cannot be summed into
  dedicated GPU-hours.

The regional sequence adapter remains untrained and outside this comparison.
The broader objective remains a usable local learner that chooses practice,
acquires reusable skills, retains knowledge and becomes progressively less
dependent on an external teacher. Restricted lesson scores do not complete it.
