# Fresh realizations of shared procedures — completed results

Fresh verified realizations improved aggregate query accuracy and probability quality, but reduced final opposite-answer paired correctness in every domain/panel macro. Neither learner acquired reliable reusable composition. **No checkpoint was promoted.** This is one initialization of a small learner under a fixed training recipe, not evidence that varied practice is intrinsically harmful or that one architectural bottleneck has been proved.

The [frozen protocol](REALIZATION_STUDY_PROTOCOL.md), verified summary (archive reference: `../runs/realization-study-local/audit/summary.json`), complete report (archive reference: `../runs/realization-study-local/audit/report.json`) and development decision (archive reference: `../runs/realization-study-local/decision.json`) preserve the comparison. Both training jobs and both audits finished locally on the RTX 5080.

## Comparison and actual exposure

Both arms used a freshly initialized 753,610-parameter causal byte learner, learning rate 0.001 and 3,600 optimizer updates. Each update combined one 32-episode microbatch from each of color, count and switch. Each arm consumed 115,200 episodes per domain: 345,600 per arm and **691,200 across 7,200 retained updates**. A separate two-update/192-episode execution probe discarded its weights and is excluded.

The arms shared ordered procedure and length draws, initial weights, objective and optimizer recipe. The fixed arm repeated the original examples. The fresh arm used the original realization on its first occurrence, then independently verified names and values. This changed increments, query truth and byte lengths as well as naming. Updates, episodes and structural draws were matched; targets, bytes, padding, FLOPs and wall time were not. There was no teacher or network training.

## Withheld results

Each row below combines 64 complete pairs at each of 8, 10 and 12 turns: 192 pairs/384 episodes. A final pair succeeds only when both opposite known answers are correct after one earlier fact changes, with the same questions and event order. Query accuracy also includes earlier unknown probes. Percentages are equal-length macros; exact per-position and per-target counts remain in the summary.

| Panel | Domain | Query fixed | Query fresh | Final pairs fixed | Final pairs fresh |
|---|---|---:|---:|---:|---:|
| `name_only` | color | 53.39% | 61.57% | 53/192 (27.60%) | 14/192 (7.29%) |
| `name_only` | count | 57.78% | 65.41% | 41/192 (21.35%) | 0/192 (0.00%) |
| `name_only` | switch | 55.11% | 61.85% | 65/192 (33.85%) | 25/192 (13.02%) |
| `value_only` | color | 48.06% | 61.06% | 32/192 (16.67%) | 7/192 (3.65%) |
| `value_only` | count | 62.19% | 63.85% | 37/192 (19.27%) | 0/192 (0.00%) |
| `value_only` | switch | 59.91% | 63.70% | 52/192 (27.08%) | 24/192 (12.50%) |
| `both` | color | 46.65% | 59.15% | 36/192 (18.75%) | 7/192 (3.65%) |
| `both` | count | 53.48% | 63.96% | 43/192 (22.40%) | 0/192 (0.00%) |
| `both` | switch | 49.11% | 63.65% | 39/192 (20.31%) | 30/192 (15.62%) |
| `composed` | color | 44.03% | 57.15% | 38/192 (19.79%) | 8/192 (4.17%) |
| `composed` | count | 48.61% | 60.38% | 25/192 (13.02%) | 0/192 (0.00%) |
| `composed` | switch | 49.34% | 59.81% | 46/192 (23.96%) | 35/192 (18.23%) |

Across all panels, query accuracy rose **52.31% → 61.80%** and known-query accuracy rose 46.29% → 50.11%, while final paired accuracy fell **22.01% → 6.51%**. Mean Brier score improved from 0.8326 to 0.4458. Better average probabilities and unknown-query handling did not yield dependable sensitivity to the changed fact. Fresh count had zero complete final action pairs in every panel; these aggregates do not establish constant ASK/DENY predictions or identify the exact per-pair error pattern.

This was also a learning-curve result: fresh final-pair area was below fixed in all twelve domain/panel comparisons. Areas use the prospectively saved 0/300/900/1,800/3,600-update checkpoints and normalized trapezoids over episodes per domain. They are sparse curve summaries, not independently observed learning rates.

| Withheld composed domain | Final-pair area fixed | Final-pair area fresh |
|---|---:|---:|
| color | 15.45% | 5.56% |
| count | 6.60% | 0.00% |
| switch | 22.57% | 15.39% |

Free-running replies did not rescue the result. Overall final reply-pair accuracy fell 22.01% → 4.12%; on withheld compositions, fixed/fresh reply pairs were color 40/192 versus 12/192, count 21/192 versus 0/192, and switch 47/192 versus 14/192. Fresh switch action/reply agreement was only 81.46–83.16% across panels, versus 97.94–98.55% for fixed; this alignment loss is an additional issue, not an explanation of every reasoning failure. Blank-English and reset-history controls scored zero final action and reply pairs in both arms. Passing these controls shows that the measured successes depend on context, not that the context was interpreted reliably.

## Fitting versus generalization

Both fitting probes cover all 2,304 admitted procedure recipes and 4,608 episodes; every original was encountered. The latest probe selects the most recently consumed realization of each recipe. It is an occurrence-selected training sample, not an unbiased estimate over the fresh stream. Fixed latest and initial rows are identical.

| Domain | Fixed initial/latest pairs | Fresh initial pairs | Fresh latest observed pairs |
|---|---:|---:|---:|
| color | 750/768 (97.66%) | 41/768 (5.34%) | 48/768 (6.25%) |
| count | 436/768 (56.77%) | 0/768 (0.00%) | 0/768 (0.00%) |
| switch | 746/768 (97.14%) | 105/768 (13.67%) | 122/768 (15.89%) |

The fixed learner strongly fitted repeated color/switch examples but transferred poorly even when only names or values changed. The fresh learner also poorly fitted its latest encountered realizations across domains. Thus freshness alone did not resolve acquisition at this model, objective, exposure and optimization recipe. Representation, capacity, optimization and teaching progression remain competing explanations. This joint schedule measures concurrent maintenance; it does not measure late-task retention, few-shot acquisition or an adaptive practice policy.

## Canonical curriculum, factor controls and boundaries

The compact immutable generator shares set/copy/advance/query procedures across three typed domains. Independent abstract and English interpreters verify the answers; the learner receives observation bytes only. Episodes have 8, 10 or 12 turns. Final queries are always known and causally depend on the changed fact; final uncertainty is therefore untested. Dependency motifs are unary chains, not general multi-input reasoning or unseen semantic algorithms.

New banks contain 4,608 training, 4,608 development and 4,608 audit episodes, with 13,824 distinct complete observed transcripts. The familiar panels use the same first 64 training procedures per domain/length:

- `name_only` preserves the exact original value seed and changes the actual naming map.
- `value_only` preserves the exact original naming map and changes the actual values.
- `both` combines those same changed coordinates. All three panels are accepted or rejected together.
- `composed` uses the withheld final-query ancestry partition and a new realization.

The final ancestry signatures exclude irrelevant events and names/constants, preserve versioned copy semantics, and are shared across domains. Current training exposes 511 known composed-query identities; development exposes 366, including 145 final composed identities. The 133 audit final composed identities are disjoint from current training/development supervised composed queries. Development rejects earlier audit-partition queries. These are syntactic dependency boundaries; historical audit motifs can recur in the finite grammar, and aliases use familiar vocabulary. Distinct transcripts do not prove distinct semantic skills.

All 9,216 transcripts from the preceding composition study were excluded. The frozen protected index contains 18,432 retired and new evaluation transcripts. Fixed training used 4,608 unique transcripts and accepted 340,992 repeated transcript exposures, averaging 75 exposures per original. Fresh training consumed 345,600 unique transcripts, with 29 whole-pair retries: one protected collision and 28 seen collisions, zero accepted duplicates and no exhaustion. Reserved first-occurrence originals were protected from later refreshes. Actual recipe, naming, value and per-turn target counts are retained in the protocol manifest (archive reference: `../runs/realization-study-local/protocol.json`) and verified stream statistics.

| Sampled all-turn targets | Fixed | Fresh |
|---|---:|---:|
| DENY | 406,284 | 407,707 |
| ALLOW | 320,244 | 318,821 |
| ASK | 454,026 | 454,026 |
| ACK | 2,265,846 | 2,265,846 |

Each arm had 172,800 final opposite-answer pairs. Including eligible earlier questions gives 317,332 opposite-answer query pairs for fixed and 316,073 for fresh. This small target-mixture difference is real and is not concealed by the matched episode count.

## Integrity and local cost

All 64 frozen source hashes, banks, protected index, saved optimizer states, sample streams and endpoint training trees passed verification. Independently regenerated streams matched every saved checkpoint; that replay authenticates curriculum consumption, not a reproduction of all AdamW arithmetic. Both CPU restarts were exact. The full suite passed **629 tests in 69.633 seconds**. The validation (archive reference: `../runs/realization-study-local/validation.json`) and execution (archive reference: `../runs/realization-study-local/execution.json`) records preserve scope.

| Resource | Fixed | Fresh |
|---|---:|---:|
| Observed UTF-8 bytes | 105,397,017 | 105,350,327 |
| Synchronized retained training-step seconds | 727.874 | 833.513 |
| Generation/authentication seconds, already included above | 213.840 | 315.869 |
| Peak allocated CUDA memory | 1,384.35 MiB | 1,439.18 MiB |

Total observation exposure was 224,532,944 tokens / 210,747,344 UTF-8 bytes. Worker step intervals sum to 1,561.387 seconds, including 529.709 seconds of generation/authentication; do not add that subset twice. Audit worker time sums to 83.562 seconds. The observed training-start-to-audit-completion envelope was 24 minutes 29 seconds, including verification, orchestration gaps and polling. These are not dedicated GPU-hours; allocator peaks exclude driver and other processes. Both workers completed without interruption. Speed alone is not useful learning efficiency.

## Backend progress and next development decision

The separate [raw-English backend](RAW_ENGLISH_LOOP_DESIGN.md) now supports bounded scheduled training, atomic learner/optimizer/stream/cursor commits, exact committed-state resume, failed-work accounting and development-only per-bank paired/reply/known/ASK progress vectors. Its 17 focused CPU tests are engineering evidence; the study did not train through an autonomous controller. Deadlines are cooperative and can overrun by an in-flight step plus current snapshot/save cost. The uniqueness index grows with training, so indefinite bounded storage is still unimplemented.

A later, separate backend GPU check restored the serialized state exactly, but
continued weights and optimizer moments differed from an independently trained
uninterrupted reference. Another uninterrupted run also differed. The input
streams and schedule matched throughout; numerical CUDA training equivalence
was not established. These 96 updates/9,216 episodes are excluded from the study
totals. The [backend record](RAW_ENGLISH_LOOP_DESIGN.md#cuda-checkpoint-recovery-and-numerical-repeatability)
preserves the failed comparison and repeat control. Future capacity comparisons
also need a declared, validated GPU execution profile.

The selected next direction (archive reference: `../runs/realization-study-local/decision.json`) is a prospective width-only comparison of the existing four-layer, four-head shared learner: widths 96/192/256, feedforward width four times model width, with 753,610 / 2,221,738 / 3,692,010 parameters. These are CPU parameter counts, not trained results. The study is **not frozen or launched**: it still requires equal development-only learning-rate search, fresh initialization, new evaluation boundaries and declared budgets. Use identical accepted fresh realizations and measure extra compute explicitly. If all widths still fail fresh fitting, test shared representation or primitive-to-composition teaching before further scaling. A favorable size effect would not prove a unique cause.

No numerical promotion gate was predeclared, and none has been invented after the result. Current audits are now inspected and retired as pristine selection evidence. Adaptive allocation, external-teacher benefit and declining dependence, long-run transcript storage, broader grounding and a unified useful checkpoint remain open work. The regional sequence adapter remains untrained. The [broad home-learning goal](HOME_LEARNING_ROADMAP.md) remains active; this completed study does not establish general intellect or a completed autonomous learner.
