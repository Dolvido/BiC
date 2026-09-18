# Shared-state learning rates: completed broad-data result

The slower rate **failed the prospective native-policy screen**. At 648 updates, AdamW 0.003 solved 1/360 development counterfactual pairs jointly; 0.0003 solved 0/360. Both solved 0/54 actual training-fit pairs and 0/360 retention pairs. Slower learning improved known-query action accuracy in every family and several state-readout measures, but did not produce reliable combined action/reply answers or general comprehension.

The verified native recount (archive reference: `../runs/shared-state-rate-pilot-local/attempt-001/analysis.json`) authenticated and recounted all 24 bank endpoints and 12,384 evaluation episode records from the completed execution (archive reference: `../runs/shared-state-rate-pilot-local/attempt-001/execution/summary.json`), without another neural run. The [prospective main protocol](SHARED_STATE_RATE_PILOT.md) and [separate state diagnostic](SHARED_STATE_RATE_DIAGNOSTIC.md) remain unchanged.

## Fixed comparison and evaluation scope

Both arms were the unchanged SharedStateStudent and SharedStateKernel with **2,264,725 parameters**, identical full initial weights from seed 852604001, the original action/reply/observation objective plus state-loss weight 0.3, gradient clip 1 and strict local FP32 on the RTX 5080. Architecture was width 192, four layers/heads and feedforward 768. The sole assigned training difference was AdamW rate: fast 0.003 versus slow 0.0003. State targets remained supervision, never policy inputs.

Each arm consumed the same 648 original-layout bundles, 32 episodes per family per update, with first-arm execution alternating. This was one pass over the existing cache containing three graded curriculum cycles, not separate rehearsal of the fit sample. Both arms completed 62,208 training episode exposures. There were zero LLM calls, no early stopping by score and no substitution of an earlier checkpoint.

All evaluation data were **reused and previously observed by the project**: 720 development episodes, 108 actual training-fit episodes and 720 retention episodes from the entity-retrieval overlay. Development remained outside these learners' training, but it was not pristine validation. The screen's inherited `fresh_joint_gain` key does not change that fact. No new data preparation was performed. The finite grammar, shared vocabulary, single initialization and limited curriculum constrain conclusions about broader language, memory or independence.

## Native policy results

Joint pair success requires correct native actions and exact freely generated replies on both opposite-answer anchor queries. Auxiliary state predictions never supply policy answers.

| Final bank | Fast joint | Slow joint |
| --- | ---: | ---: |
| Reused development | 1/360 | 0/360 |
| Development color | 0/120 | 0/120 |
| Development count | 0/120 | 0/120 |
| Development switch | 1/120 | 0/120 |
| Actual training-fit | 0/54 | 0/54 |
| Reused retention | 0/360 | 0/360 |

The overall development gain, strict gains in each family and training-fit improvement conditions all failed. Every known/unknown regression margin and the retention margin passed, but those checks cannot replace the missing improvements. The fixed-budget result does not rule out a longer run, other rates or different optimization settings.

All predetermined trajectories are shown below. A/R/joint denotes action-pair, reply-pair and joint-pair correct counts with the common denominator shown separately.

| Updates | Bank | Denominator | Fast A/R/joint | Slow A/R/joint |
| ---: | --- | ---: | ---: | ---: |
| 0 | Development | 360 | 0 / 0 / 0 | 0 / 0 / 0 |
| 0 | Training-fit | 54 | 0 / 0 / 0 | 0 / 0 / 0 |
| 0 | Retention | 360 | 0 / 0 / 0 | 0 / 0 / 0 |
| 216 | Development | 360 | 14 / 23 / 0 | 18 / 0 / 0 |
| 216 | Training-fit | 54 | 1 / 3 / 0 | 3 / 0 / 0 |
| 216 | Retention | 360 | 20 / 25 / 3 | 20 / 0 / 0 |
| 432 | Development | 360 | 22 / 11 / 1 | 34 / 0 / 0 |
| 432 | Training-fit | 54 | 4 / 3 / 0 | 6 / 0 / 0 |
| 432 | Retention | 360 | 27 / 12 / 1 | 31 / 0 / 0 |
| 648 | Development | 360 | 13 / 9 / 1 | 14 / 0 / 0 |
| 648 | Training-fit | 54 | 2 / 2 / 0 | 2 / 0 / 0 |
| 648 | Retention | 360 | 7 / 12 / 0 | 17 / 0 / 0 |

The slow arm had zero reply-pair successes on every bank at every endpoint. Its development action-pair score fell from 34/360 at 432 to 14/360 at 648. Individual known development actions followed **500/1,594 → 718/1,594 → 638/1,594** at updates 216/432/648. This trajectory does not support assuming that simply extending the same training will produce monotonic improvement.

| Final development slice | Fast action | Slow action | Fast reply | Slow reply |
| --- | ---: | ---: | ---: | ---: |
| Color known | 158/528 | 195/528 | 174/528 | 188/528 |
| Color unknown | 377/598 | 399/598 | 391/598 | 452/598 |
| Count known | 201/538 | 240/538 | 187/538 | 211/538 |
| Count unknown | 399/596 | 437/596 | 435/596 | 460/596 |
| Switch known | 176/528 | 203/528 | 175/528 | 161/528 |
| Switch unknown | 434/598 | 405/598 | 462/598 | 507/598 |

Known-action gains were +7.01 points for color, +7.25 for count and +5.11 for switch. Overall known actions rose from 535/1,594 (33.56%) to 638/1,594 (40.03%); known replies rose from 536/1,594 to 560/1,594, with switch replies declining. On nonanchor known questions, actions rose from 188/874 to 285/874 and replies from 195/874 to 212/874. These are factual readout improvements within the tested bank, not accurate general comprehension.

Across all 3,386 development queries, action/reply agreement rose from 2,125/3,386 to 2,271/3,386. All replies parsed in both arms. Unsupported action ASK on known questions fell from 465/1,594 to 400/1,594, while unsupported reply ASK increased from 519/1,594 to 557/1,594. Aggregate unknown-answer or agreement gains should not obscure that action and English still fail the opposite-answer pair test.

## Final training-state diagnostic

The declared diagnostic inspected only both final checkpoints on the same 108 actual training-fit rows. Its raw output (archive reference: `../runs/shared-state-rate-diagnostic-local/attempt-001/diagnostic.json`), execution receipt (archive reference: `../runs/shared-state-rate-diagnostic-local/attempt-001/receipt.json`) and independent pure verification (archive reference: `../runs/shared-state-rate-diagnostic-local/attempt-001/verification.json`) preserve and recount predictions, targets, known probabilities and true-value ranks. Both model states remained unchanged. This is a training-sample readout, not held-out state transfer, and it cannot override the native screen.

| State readout | Fast full known exact | Slow full known exact | Fast conditional value | Slow conditional value |
| --- | ---: | ---: | ---: | ---: |
| Color | 1/880 | 36/880 | 189/880 (21.48%) | 418/880 (47.50%) |
| Count | 0/880 | 0/880 | 49/880 (5.57%) | 65/880 (7.39%) |
| Switch | 9/880 | 62/880 | 481/880 (54.66%) | 538/880 (61.14%) |
| Overall | 10/2,640 (0.38%) | 98/2,640 (3.71%) | 719/2,640 (27.23%) | 1,021/2,640 (38.67%) |

Conditional value is argmax over all 106 nonzero classes, with no family mask. Mean true-value rank improved from 2.6375 to 1.8307 for color, 16.2307 to 10.7625 for count and 1.4534 to 1.3886 for switch. The descriptive fit-buffer majority references (archive reference: `../runs/shared-state-fit-local/fit-buffer-baselines.json`) are respectively 30.57%, 6.25% and 54.77%; they are post hoc references, not policy inputs or a prospective screen. The slow arm exceeded those references, but count's conditional improvement was small and full known-state accuracy remained zero.

Indeed, a direct count of the saved predictions found that **both models predicted unknown for all 4,320 count-family states**, including all 880 actually known count values. This coexists with improved count conditional accuracy/rank because the total nonzero probability is distributed among many values: an unknown full argmax does not prove there are no encoded facts.

Across families, full unknown-state accuracy declined from 10,221/10,320 (99.04%) to 9,503/10,320 (92.08%). With the separate strict `P(known)>0.5` rule, known sensitivity was 1,845/2,640 versus 1,567/2,640; unknown specificity was 3,764/10,320 versus 4,868/10,320. Overall knownness Brier scores were 0.249214 and 0.248554. Balanced state loss fell from 1.561198 to 1.413544, with lower loss in every family. These metrics establish changed readout behavior; they do not establish successful state use by the policy, comprehension or a uniquely identified causal bottleneck.

## Work, validation and cost

The main comparison completed exactly **1,296 physical/retained updates**, **124,416 training episode exposures**, 3,888 family forwards/backwards, two model constructions and six full model/AdamW snapshots. Each arm made 1,944 auxiliary readouts/objectives. State labels were computed once for each of the 648 shared bundles. All 24 native bank endpoints, 72 raw evaluation groups and 12,384 evaluation episode exposures were retained. There were no uncertain optimizer outcomes, retries or added training after scores.

The separate diagnostic used two model constructions, two checkpoint images, one bank image, nine target packs over 108 rows/1,080 turns, 18 BOS-only forwards, 18 auxiliary calls, 216 episode passes and 25,920 state predictions. It performed zero updates, backwards, free-reply rollouts or tutor calls. Main training and the diagnostic both completed on their first invocation.

| Newly incurred stage | Wall seconds | CPU seconds |
| --- | ---: | ---: |
| Corrected static contract validation | 0.015 | 0.000000 |
| Source/data freeze | 2.297 | 2.062500 |
| Entire main execution | **423.063** | 394.187500 |
| Final state diagnostic | 4.172 | 3.843750 |

One prior static-validation invocation failed with `NameError: SEED`: its extractor omitted tuple-assignment constants. The validator was corrected without changing production code; the preserved validation receipt (archive reference: `../runs/shared-state-rate-validation-local/attempt-001/report.json`) records that failed invocation and the successful check. Both avoided importing the modules under test or loading archives and performed zero neural work. The 0.015-second value covers the corrected check; no separate duration is recorded for the failed invocation. Existing model/kernel tests were not repeated or counted as new validation evidence.

**New data-preparation cost was zero.** The overlay's 64.235 seconds are historical work from the prior entity-retrieval comparison, not a new charge here. Re-reading data and validating banks during this execution are included in main time. Main operation totals include 57.583 seconds of data loads, 148.679 of batch preparation, 175.450 of learner operations, **2.959 of shared target preparation**, 23.658 of evaluation, 3.688 of bank validation and 0.219 of snapshots. These subdivisions must not be added again to the enclosing 423.063 seconds.

| Completed-event per-arm time | Fast | Slow |
| --- | ---: | ---: |
| Learner steps, seconds | 87.697 | 86.866 |
| Batch preparation, seconds | 74.932 | 72.461 |
| Training including preparation, seconds | 162.629 | 159.327 |
| Training episode exposures/second, including preparation | 382.51 | 390.44 |
| Native evaluation, seconds | 12.079 | 11.517 |

Journal event timings exclude completion-event writing overhead and therefore differ slightly from enclosing operation totals. The measured slow-arm training/preparation difference was −3.302 seconds, with one fewer final joint development pair. This is not a replicated speed comparison; architectures and exposure budgets were identical. Shared target preparation belongs to neither arm's incremental cost. Peak main GPU allocation was 1,610,770,944 bytes (1,536.15 MiB); peak reservation was 4,758,437,888 bytes (4,538 MiB). Diagnostic peak allocation was 184,517,120 bytes. Pure recount and this documentation used no neural work.

## Preserved pins and next decision

| Artifact | SHA-256 |
| --- | --- |
| Pilot `launch.json` | `bcfdfb8f5b369d4591988e7b7db26321256cd6a9db263003979b3d10619fd3f0` |
| Main `execution/summary.json` | `e268e983015ef56eb10b14d864e82566e2a102c0de0e156b3d5d03f314548931` |
| Native `analysis.json` | `3154d66a8fc91d5745723e9cab7edb5dac2bb82ea9809b706aea3339d9555434` |
| Static validation `report.json` | `110ee774e7d50738005ed9e5081d38615df3e4a98860aac2cc564ee3deb94499` |
| State `diagnostic.json` | `0fa826a650c2de86ced742218b5c233dd86183dc429a3e8da1f846fa63e0461e` |
| State `receipt.json` | `8d9dd73dd3df95a2ba75dc26d6b596a23289ed21f9c319760b79411e70b9bc79` |
| Independent state `verification.json` | `4f5f466ef00bb4ce1b3dec419387c6a09552d06a2dbe1c6354f458c1a22961f1` |

The next selected work is a [separately frozen paired continuation](SHARED_STATE_CONTINUATION_DESIGN.md), after an exact SharedStateContinuation restore proof and an explicit replay-ID adapter. Both final 648-update learners retain their own complete AdamW state and original rates for two additional full-cache passes, targeting lifetime 1,944 updates. Replayed episodes remain repeated exposures, not new lessons. A new development bank is evaluated at the restored 648 baseline and fixed 1,296/1,944 endpoints alongside the explicitly reused development, fit and retention banks; concrete launch settings and acceptance rules must be frozen separately.

The original completed run stays closed, and neither rate is adopted as a winner. The slower arm's partial factual/state-readout gains justify testing further broad acquisition, but the nonmonotonic trajectory prevents extrapolating success. No per-family patch, teacher call, automatic extension or checkpoint promotion follows from this result.
