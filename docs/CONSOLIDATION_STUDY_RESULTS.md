# Continued fitting and replay — completed September 16, 2026

**Replay substantially reduced forgetting, and additional fitting helped the
broader world bank. Neither established reliable acquisition of the withheld
subject. No checkpoint was promoted.** Both continuations and all nine audits
completed locally. Ordinary adaptation lost 8.17–17.16 percentage points of old
query accuracy; replay limited the aggregate loss to 0.10–2.18 points, using extra
computation and old examples. Every pretrained candidate still solved zero
advanced arithmetic later-known pairs out of 128.

The [frozen protocol](CONSOLIDATION_STUDY_PROTOCOL.md) defined no numerical
promotion gate. The verified summary (archive reference: `../runs/consolidation-study-local/audit/summary.json`),
full audit (archive reference: `../runs/consolidation-study-local/audit/report.json`) and
decision record (archive reference: `../runs/consolidation-study-local/decision.json`) preserve the
complete evidence. This is one initialization and one held-out subject. It
supports a retention component, not a general capability or completed home-learning goal.

## Matched conditions and actual panels

The 753,610-parameter sequence learner is unchanged. Two original eight-map
banks, with 32 or 256 anonymous world pairs per family, were continued from
3,600 to 14,400 cumulative updates using their original optimizer and sampler
states. Each short/long parent then received ordinary or replay adaptation,
plus a common fresh ordinary control: nine candidates. All used fresh AdamW
at 0.001 for adaptation and the same 64 new episodes per optimizer step.
Replay added 32 old episodes at loss weight 0.5, using exactly 256 original
complete pairs per old family. The support draw stream and new-example byte
exposure were identical across candidates. Replay was extra compute, not a
compute-matched advantage. No external teacher or pretrained model participated.

New development and audit pools were frozen before continuation. Novel-world
pools reject prior normalized individual transcripts, not only paired IDs.
Naming-only panels intentionally reuse familiar support worlds. The arithmetic
manifest records these actual counts:

| Query panel | Episodes | Anonymous world pairs | Distinct naming maps | Query targets (no / yes / unknown) |
|---|---:|---:|---:|---|
| names | 128 | 32 | 64 | 128 / 128 / 128 |
| worlds | 256 | 128 | 1 | 256 / 256 / 256 |
| both | 256 | 128 | 128 | 256 / 256 / 256 |
| advanced | 256 | 128 | 128 | 256 / 256 / 0 |

**Protocol wording correction:** “two withheld naming maps” meant two
renderings of each familiar world, not two maps globally. The names panel
actually has 64 full mapping identities across 64 rendered pairs; both and
advanced each have 128. These are the frozen data; no training or evidence was
changed to make this clarification.

Advanced arithmetic reserved previously unused development-partition worlds
with audit admission and audit names, sealed from optimization and development
scoring. The older audit-world partition was nearly exhausted. This explicit
reservation retained prior transcript exclusions, but highlights finite grammar
coverage rather than an open-ended curriculum. Names reuse familiar vocabulary;
arithmetic digit bytes were absent from original main training. Level 3 changes
copy/update composition and has no unknown targets. Some conditional changes
are irrelevant or leave outcomes unchanged. Scores should not be compared with
earlier, different curricula as regression on the same benchmark.

## More fitting helped the broader bank

Fit uses the common first 32 worlds under one familiar map, not the whole bank.
Development uses the new frozen names/worlds/both panels with equal-bank macros.

| Parent bank | Fit query before → after | Fit pairs before → after | Development query before → after | Development pairs before → after |
|---|---:|---:|---:|---:|
| w32-n8 | 99.22% → 99.22% | 97.06% → 97.06% | 60.30% → 60.05% | 27.14% → 25.41% |
| w256-n8 | 87.67% → 96.88% | 55.52% → 88.44% | 62.57% → 65.83% | 24.56% → 37.23% |

For the broad bank, common-fit graph pairs improved from 23.53% to 94.12%.
This demonstrates that the earlier fixed endpoint was not fully fitted. The
broader development paired score also improved, but remaining generalization
gaps persist. The smaller bank was already almost perfectly fitted and showed
no consistent development gain from the longer endpoint. More hours helped a
measured bottleneck here; they did not by themselves establish reusable abstraction.

## Replay protected aggregate retention

Each adaptation ran 256 updates, or 16,384 new episodes. Query areas are normalized
linear-update trapezoids over 0/4/16/64/256; they combine starting performance
and adaptation rather than isolating learning speed. Macros weight the four
arithmetic panels equally, despite different denominators. Old-task macros
weight the nine family/panel banks equally.

| Candidate | Query area | Pair area | Final query | Final pairs | Old query before → after | Old pairs before → after |
|---|---:|---:|---:|---:|---:|---:|
| w32-n8-short-ordinary | 47.68% | 0.07% | 47.18% | 0.20% | 59.87% → 42.71% | 26.03% → 4.01% |
| w32-n8-short-replay | 44.53% | 0.05% | 44.55% | 0.00% | 59.87% → 59.77% | 26.03% → 28.57% |
| w32-n8-long-ordinary | 47.91% | 0.08% | 46.65% | 0.10% | 62.11% → 53.94% | 29.08% → 12.40% |
| w32-n8-long-replay | 49.20% | 0.29% | 49.93% | 0.78% | 62.11% → 61.46% | 29.08% → 25.14% |
| w256-n8-short-ordinary | 47.59% | 0.08% | 48.13% | 0.20% | 62.01% → 53.57% | 24.73% → 5.25% |
| w256-n8-short-replay | 47.99% | 0.04% | 49.61% | 0.10% | 62.01% → 59.82% | 24.73% → 21.50% |
| w256-n8-long-ordinary | 49.66% | 0.05% | 50.94% | 0.00% | 65.45% → 57.02% | 35.44% → 16.02% |
| w256-n8-long-replay | 50.19% | 0.01% | 51.50% | 0.00% | 65.45% → 64.60% | 35.44% → 32.74% |
| fresh-ordinary | 47.68% | 2.64% | 50.08% | 6.05% | — | — |

Replay preserved much more old-task accuracy than ordinary adaptation in all
four parent comparisons. It also largely preserved free replies: the broad-long
parent’s old-task reply accuracy changed 65.10%→64.45% with replay, versus
65.10%→56.80% without it. These gains cost 8,192 extra old episodes per replay
candidate. They do not demonstrate better intrinsic transfer at equal compute.

**Aggregate retention still hides losses.** Broad-long replay retained
65.45%→64.60% old query accuracy and 35.44%→32.74% old paired accuracy. Within
its familiar-world/new-name binding panel, query and reply accuracy each fell
8.07 points, and paired accuracy fell 14.47 points (47.37%→32.89%). Retaining
the mean is insufficient evidence that every earlier ability survives.

## Arithmetic paired reasoning remained weak

Ordinary pretrained query-area differences versus fresh were +0.01, +0.23,
−0.08 and+1.98 percentage points for small-short, small-long, broad-short and
broad-long respectively. Every pretrained pair area remained below fresh.
Replay changed acquisition inconsistently: its query-area differences versus
ordinary adaptation were −3.16, +1.29, +0.40 and+0.53 points. Retention is the
clearer benefit in this comparison.

All eight pretrained final paired macros were 0–0.78%, compared with 6.05% for
fresh. The fresh model was still weak, including only 1/128 advanced later-known
pairs. All eight pretrained candidates scored 0/128 on that slice. The best
pretrained later counts on names/worlds/both were only 1/64,1/128 and 2/128,
achieved together by small-long replay. High query accuracy therefore cannot
be treated as reliable counterfactual or numerical reasoning.

For example, compare broad-long replay with fresh at the final endpoint:

| Panel | Broad-long replay query | Its opposite pairs | Its later pairs | Fresh query | Fresh opposite pairs | Fresh later pairs |
|---|---:|---:|---:|---:|---:|---:|
| names | 205/384 | 0/128 | 0/64 | 189/384 | 7/128 | 1/64 |
| worlds | 512/768 | 0/256 | 0/128 | 485/768 | 24/256 | 7/128 |
| both | 402/768 | 0/256 | 0/128 | 365/768 | 10/256 | 3/128 |
| advanced | 172/512 | 0/256 | 0/128 | 207/512 | 14/256 | 1/128 |

The 66.67% worlds score includes every unknown answer correct (256/256),
but only 50% of known answers and zero opposite pairs. Known yes/no targets
had identical aggregate prediction counts. The model learned useful uncertainty
and surface distinctions without reliably responding to the changed numeric
fact. Advanced broad-long replay also emitted 165 unsupported ASK actions on
512 queries, where no true unknown targets exist; ASK recall is undefined there.

## Controls, verification and measured resources

Blank and reset controls for broad-long replay yielded 14.06% and 25.00% query
accuracy, respectively, with zero opposite-pair accuracy. Context affects
outputs but these controls do not prove correct use of the critical facts.
Free replies were evaluated independently of action labels. Parseability or
action/reply agreement is not correctness; generated strings were not retained
for a claim about constant-output collapse.

All 53 frozen sources, checkpoint identities, optimizer/sampler continuations
and exact sampled byte counts were verified. All nine serialized CPU restarts
reproduced actions and free replies exactly. Parent checkpoints remain unchanged.
The validation record (archive reference: `../runs/consolidation-study-local/validation.json`)
records the final **525-test pass in 50.449 seconds**, preserving the earlier
Windows HTTP transport failures and retries. The summary helper also passed
eight focused tests. A tuple/list normalization in that
read-only helper reconciled JSON optimizer metadata with the saved checkpoint;
no frozen training source or evidence changed.

| New work in this study | Optimizer updates | Episode exposures | Observation UTF-8 byte exposures |
|---|---:|---:|---:|
| Two continuations, excluding parent history |21,600|1,382,400|206,415,041|
| Nine adaptations: new subject |2,304|147,456|24,000,210|
| Extra old examples in four replay conditions |0 additional|32,768|4,886,022|

Total new work was 23,904 optimizer updates and 1,562,624 repeated episode
exposures. Each old buffer held 1,536 episodes:3,946,397/3,946,965 canonical JSON
bytes for the small/broad parent, including 229,546/229,161 observation text bytes.
The support bank used 5,301,852 canonical JSON bytes and 333,340 text bytes.
Seed recipes describe the curriculum compactly, but stored buffers and exposure
counts remain part of the actual resource cost.

The observed local execution envelope was 666 seconds (11 minutes 6 seconds):
406 for continuations and 260 for the audit. The two continuation worker training
intervals sum to 659.627 seconds; synchronized adaptation training adds 60.391
seconds. Continuation peak allocated CUDA memory was 838.27–841.79 MiB.
Internal continuation invocation timers exclude setup and final scoring/writing;
external intervals include those phases and completion observation delays.
Concurrent intervals are not dedicated GPU-hours. See the
execution record (archive reference: `../runs/consolidation-study-local/execution.json`).

## Next development decision

Preserve replay as a useful retention component and the evidence that longer
fitting can help a broader curriculum. Do not promote these checkpoints as
broadly capable. The next prospective comparison should teach shared compositions
and compare joint interleaving with sequential replay at matched per-family
exposure, across at least two withheld-family rotations. Paired known decisions
and per-family retention must constrain any later progress-based selector so
unknown recognition or aggregate scores cannot stand in for acquired skills.

Expand the finite curriculum explicitly before claiming long-horizon learning.
The 1.31M-parameter regional adapter remains untrained. The broader goal remains
a usable local BiC that directs practice, acquires reusable abilities, preserves
gains and becomes less dependent on an external teacher. Ordinary replay and
gradient adaptation are not demonstrations of biological consolidation, learned
meta-learning or general intelligence. Previously inspected audits remain
historical evidence, and subsequent changes require fresh prospective boundaries.
