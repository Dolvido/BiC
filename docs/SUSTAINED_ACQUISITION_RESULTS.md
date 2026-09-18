# Sustained acquisition results

The completed continuation passed the declared acquisition screen and both
transfer screens, with positive joint-pair progress in all three families.
Development rose from 149/324 to 257/324 joint pairs (+33.33 percentage points);
original-layout transfer rose from 71/216 to 136/216 (+30.09 points), and varied
transfer from 64/216 to 125/216 (+28.24 points). This is the first such combined
pass in this sequence of shared-state studies. It supports sustained shared
learning in the tested grammar, without automatic promotion or a broad-English
capability claim.

The [protocol](SHARED_ACQUISITION_PROTOCOL.md) continued the completed procedural
parent at update 2,160, preserving the architecture, full AdamW, learning rate
0.0003, original sequence objective and 0.3 training-only state supervision.
There were no teacher requests or family-specific allocations. The single worker
finished at update 7,816 after 1,744.750 wall seconds, stopping at its declared
final-reserve boundary within the 1,800-second allowance. No intermediate score
selected the endpoint.

## Native learning and transfer

Joint success requires both opposite-answer members to be correct through both
the action head and freely generated English reply. Every scheduled endpoint is
below; column denominators are fixed pair counts.

| Lifetime update | Development /324 | Transfer original /216 | Transfer varied /216 | Fit /54 | Retention /360 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,160 | 149 | 71 | 64 | 24 | 152 |
| 2,808 | 168 | 82 | 78 | 30 | 188 |
| 3,456 | 196 | 94 | 94 | 31 | 199 |
| 4,104 | 193 | 92 | 96 | 33 | 202 |
| 4,752 | 215 | 116 | 112 | 35 | 239 |
| 5,400 | 228 | 121 | 123 | 38 | 241 |
| 6,048 | 247 | 124 | 137 | 40 | 258 |
| 6,696 | 236 | 129 | 127 | 45 | 262 |
| 7,344 | 262 | 141 | 138 | 46 | 274 |
| 7,816 | 257 | 136 | 125 | 43 | 286 |

The final endpoint was not the best observed endpoint: development fell from
262 to 257 pairs, original transfer from 141 to 136, and varied transfer from
138 to 125 after update 7,344. All final comparisons still passed their
prospective baseline-relative screens. The final varied-layout gap was
-5.09 percentage points against original layout; the gap was not monotonic.

Family joint-pair counts, restored baseline -> final:

| Bank | Color | Count | Switch |
| --- | ---: | ---: | ---: |
| Development | 64 -> 94/108 | 19 -> 67/108 | 66 -> 96/108 |
| Transfer original | 33 -> 58/72 | 1 -> 28/72 | 37 -> 50/72 |
| Transfer varied | 29 -> 53/72 | 2 -> 27/72 | 33 -> 45/72 |
| Training fit | 11 -> 17/18 | 2 -> 10/18 | 11 -> 16/18 |
| Retention | 65 -> 110/120 | 17 -> 68/120 | 70 -> 108/120 |

Counting transfer remains incomplete: 28/72 original and 27/72 varied pairs,
compared with 58/72 and 53/72 for color. These gains do not establish reliable
composition across the tested procedures. The varied bank changed legal layout
for 210/216 paired parents, including 126 nonterminal anchors; the remaining
six pairs retained the original layout.

Known/unknown query correctness, baseline -> final. Each denominator counts
individual queries, not opposite-answer pairs. Replies were decoded freely
from BOS without target answers or the training-only state decoder.

| Bank | Known action | Known free reply | Unknown action | Unknown free reply |
| --- | ---: | ---: | ---: | ---: |
| Development | 1030 -> 1336/1440 | 1005 -> 1333/1440 | 1268 -> 1610/1626 | 1295 -> 1608/1626 |
| Transfer original | 599 -> 785/918 | 590 -> 786/918 | 691 -> 900/912 | 701 -> 896/912 |
| Transfer varied | 567 -> 772/918 | 553 -> 767/918 | 590 -> 870/912 | 570 -> 869/912 |
| Training fit | 201 -> 241/252 | 189 -> 239/252 | 212 -> 263/264 | 219 -> 263/264 |
| Retention | 1123 -> 1460/1592 | 1099 -> 1467/1592 | 1352 -> 1763/1784 | 1400 -> 1759/1784 |

Final development action/reply agreement was 3,045/3,066; generated replies were
parseable on all 3,066 development queries (also true at baseline). Unsupported
ASK on known development questions fell from 124 to 4/1,440 for actions and
153 to 4/1,440 for replies. Parseability alone is not semantic correctness:
counting pairs and transfer still expose errors. Exact per-family known/unknown,
nonanchor and other native counts are retained in the analysis (archive reference: `../runs/sustained-acquisition-local/attempt-001/analysis.json`).

## Work, caching and cost

There were **5,656 new optimizer updates and 542,976 episode exposures**:
eight complete passes over the unchanged 648-bundle catalogue, plus 472 bundles.
The catalogue contains 62,208 unique original training rows; this invocation
created zero new training rows. All 16,968 family forwards, backwards and state
objectives completed. There were nine full-state restorations and nine new
snapshots, zero unknown optimizer outcomes and zero automatic retries. Each
endpoint evaluated 2,340 episodes; ten endpoints produced 23,400 evaluation
episode passes and 106,180 query records.

| Measured worker component | Wall seconds |
| --- | ---: |
| Training, including prepared-token consumption | 971.596 |
| Lesson preparation, including resident-reader work | 669.698 |
| Native evaluation | 52.486 |
| Evaluation-bank preparation | 5.266 |
| Source/input authentication | 2.825 |
| Full-state restoration | 2.638 |
| New snapshots | 0.813 |

The private resident cache loaded 648 archives once (493,782,632 serialized
bytes, 470.91 MiB), then served 5,008 hits without eviction. It decoded 62,208
rows and computed their state targets once per cached image. Nested reader costs
were 57.656 seconds archive loading, 603.263 seconds owner preparation/packing,
2.382 seconds targets and 0.157 seconds copy work. These are already included
in lesson preparation and must not be added again. The original prepared-owner
path remained in use; this run does not causally measure its speed against
packed-v2 or an uncached run.

Peak host working set was 4,911,431,680 bytes (4.57 GiB), below the 8 GiB resource
limit. Peak GPU allocation/reservation was 1,572,006,400 / 4,682,940,416 bytes
(1.46 / 4.36 GiB). The 512 MiB cache cap applies to serialized archive bytes,
not the larger in-memory Python objects or total process memory.

The worker used 1,629.78125 CPU seconds. Fresh evaluation-data preparation cost
44.781 wall / 44.171875 CPU seconds separately: together these two phases cost
1,789.531 wall seconds. Launch metadata freeze added 1.579 seconds. Engineering
checks are separate: resident-reader validation 1.016 seconds, controller tests
0.016 seconds, and transfer-data tests 1.328 seconds. The separate packed-v2 GPU
proof took 14.375 seconds overall and was not adopted: its short matched timing
slice was 5.033 seconds packed versus 4.826 reference. None of these proof costs
is hidden inside, or evidence of benefit from, the new learning updates.

## Verification, limits and next decision

The independent stdlib recount authenticated all 150 raw groups and all 50
bank-endpoints, independently counted 23,400 episodes and 106,180 queries, and
reconciled the 29,010-event journal with all 5,656 physical updates. It matched
the primary metrics and screens exactly, taking 3.703 wall / 3.6875 CPU seconds
with no model, checkpoint deserialization, generation or teacher call.

Fresh development/transfer banks were frozen before training; fit and retention
were explicitly reused. Transfer parents use the audit generator partition,
but fixed repeated observation makes this a development diagnostic, not a
pristine final audit. The finite English grammar, familiar aliases/value ranges,
outcome-informed procedural parent and single continuing learner limit the
claim. This is neither an independent architecture comparison nor evidence of
external-tutor benefit or learned self-direction.

The next separately frozen experiment is the first real continuous tutor
campaign (archive reference: `../runs/continuous-tutor-campaign-local/attempt-001/launch.json`) from
the update-7,816 checkpoint. It pairs compact verified authored teaching with
procedural teaching, then common tutor-free withdrawal; branch adoption is
subject to the prospective native benefit and fixed-reference retention guards.
Its first real cycle result is pending. The earlier authored comparison's
failed-benefit result remains unchanged.

Authenticated records:

| Record | SHA-256 |
| --- | --- |
| Launch (archive reference: `../runs/sustained-acquisition-local/attempt-001/launch.json`) | `5ab38001fd31c740b47fe4e6e7b26119d39dce019f7cf2b1d3322c46f1053525` |
| Completed summary (archive reference: `../runs/sustained-acquisition-local/attempt-001/execution/summary.json`) | `3c3a78c7cf30da2bf366e2d353dd15dc61003b6fa69fc55e0251176a51e98983` |
| Primary analysis (archive reference: `../runs/sustained-acquisition-local/attempt-001/analysis.json`) | `66939972a8a71875720b1325534b08e13003405ec3215ee1933b02c942f93154` |
| Independent recount (archive reference: `../runs/sustained-acquisition-analysis-local/attempt-001/independent-analysis.json`) | `73d88c7c36f0a17464ea91f82d8d48fc7d07e2f8fa7ba02e8d817203593338b4` |
| Evaluation manifest (archive reference: `../runs/shared-acquisition-transfer-data-local/attempt-001/manifest.json`) | `3ce2402df0edfb54fca9f5dd01aa42e16a37b4b3f27b0adf8884bc24449d5c2a` |
| Final checkpoint (archive reference: `../runs/sustained-acquisition-local/attempt-001/execution/checkpoints/00007816.pt`) | `1b9c570dbd117415fb365d02614ea0076f703cc4bee301b6c09bf987ded42407` |
