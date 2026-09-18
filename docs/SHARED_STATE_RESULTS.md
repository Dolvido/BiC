# Shared causal-state teaching: completed local result

The completed comparison **failed its prospective screen**. At the fixed 648-update endpoint, the state-supervised learner solved 4/360 fresh development counterfactual pairs jointly, versus 0/360 for the control. All four successes were switch problems. Both learners solved 0/54 actual training-fit pairs. This does not support adopting the intervention or extending this configuration unchanged.

The full execution is attempt-001 (archive reference: `../runs/shared-state-pilot-local/attempt-001/execution/summary.json`), with a pure recount (archive reference: `../runs/shared-state-pilot-local/attempt-001/analysis.json`) of all 24 recorded bank endpoints and 12,384 evaluation episode exposures. The recount performed no new neural evaluation. The [prospective protocol](SHARED_STATE_PILOT.md) and [separate diagnostic protocol](SHARED_STATE_DIAGNOSTIC.md) remain unchanged.

## What was compared

Two identically initialized models used the flat width-192, four-layer policy plus the same shared entity-state decoder. Seed was 852304001, AdamW learning rate 0.003, gradient clipping 1, and execution used strict FP32 on the local RTX 5080. Each arm received the same 648 cached original-layout bundles: 62,208 episode exposures across color, count and switch, depths 0–5, and 8/10/12 turns. Which arm updated first alternated. Original action, English reply and observation losses were unchanged.

The candidate added auxiliary loss weight 0.3; the control used zero and skipped that graph. State targets came only from the visible English prefix after each turn, for all 12 aliases and the same 107 classes. For each example×turn, known and unknown entries were averaged separately and the present groups weighted equally; those losses were averaged over examples and turns, then across the three families. State labels never entered native policy inference. This tests the whole added teaching intervention, not a separately isolated decoder factorization.

Development used 720 newly generated episodes excluded against the authenticated 1,701,720-transcript union. Training-fit reused 108 actual training episodes; retention reused 720 prior development episodes. These remain finite-English, shared-vocabulary tasks: transcript exclusions do not establish natural-language independence or exclude semantic equivalents. The fixed training curriculum also avoids interfering overwrites; target-code tests of overwrites are not evidence that the learner mastered them. There were **zero LLM calls**.

## Native policy result

A joint pair requires correct native actions **and** exact freely generated replies on both counterfactual anchor queries. Auxiliary state decoding was not used to answer them.

| Fixed final evaluation | Control | State-supervised | Change |
| --- | ---: | ---: | ---: |
| Fresh development, all families | 0/360 | 4/360 | +1.11 percentage points |
| Development: color | 0/120 | 0/120 | 0 |
| Development: count | 0/120 | 0/120 | 0 |
| Development: switch | 0/120 | 4/120 | +3.33 percentage points |
| Reused actual training-fit | 0/54 | 0/54 | 0 |
| Reused retention | 0/360 | 5/360 | +1.39 percentage points |

The screen required at least a 5-point overall development gain, strictly more correct joint pairs in **each** family, strictly better total training-fit joint performance, no known/unknown action or reply loss exceeding 5 points in any development family, and no overall joint retention loss exceeding 5 points. The overall gain, color/count improvements and training-fit conditions failed. Count-family unknown-query action accuracy also fell from 470/616 (76.30%) to 432/616 (70.13%): **−6.17 points**, exceeding the allowed regression. The screen is exploratory and single-seed; even a pass would only support replication, not promotion.

The full predetermined progression shows deterioration after 432 updates:

| Updates per arm | Development control/state, /360 | Training-fit control/state, /54 | Retention control/state, /360 |
| ---: | ---: | ---: | ---: |
| 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 216 | 8 / 2 | 1 / 0 | 9 / 3 |
| 432 | 11 / 6 | 3 / 2 | 16 / 6 |
| 648 | 0 / 4 | 0 / 0 | 0 / 5 |

Both arms finished below their 432-update scores on every bank. No earlier checkpoint was selected or substituted for the prospective final endpoint. This trajectory is descriptive evidence of unstable acquisition/retention under this training configuration; it does not isolate the cause.

## Training-state diagnostic

The diagnostic was declared before final scores, then run after the main process and primary recount finished. It loaded only the two final checkpoints and scored the 108 actual training-fit rows. Nine family×turn groups per model produced 18 native BOS-only forwards, 18 auxiliary calls, 216 episode passes and 25,920 entity-state predictions. There were no updates, free reply rollouts, or development/audit state targets. Both full model states, evaluation modes and absent gradients were checked unchanged.

| State-supervised auxiliary head | Exact known values | Exact unknown states | All exact states | Balanced loss |
| --- | ---: | ---: | ---: | ---: |
| Color | 7/880 | 3,391/3,440 | 3,398/4,320 | 1.336740 |
| Count | 0/880 | 3,440/3,440 | 3,440/4,320 | 2.621645 |
| Switch | 0/880 | 3,438/3,440 | 3,438/4,320 | 0.993597 |
| Overall | **7/2,640 (0.27%)** | **10,269/10,320 (99.51%)** | **10,276/12,960 (79.29%)** | **1.650661** |

The candidate predicted unknown for 12,902/12,960 states (99.55%). An always-unknown classifier would score 10,320/12,960, or **79.63%**, higher than its 79.29% overall exact accuracy. Its auxiliary argmax therefore remains almost entirely unknown, despite the known/unknown-balanced training loss. The control's unused auxiliary head obtained 6/2,640 known values, 326/10,320 unknown states and balanced loss 4.604268; it is an untrained reference, not an independently state-trained competitor.

This does not show that a successful state decoder merely failed to communicate its facts to the native policy: accurate known-state decoding was itself not obtained on the training sample. It also **does not establish that the encoder contains no latent facts**, measure state transfer, or uniquely locate a causal bottleneck. Soft probabilities can improve loss while top-class predictions remain unhelpful. The independent raw recount also found that all seven known successes occur at the first prefix. Its recount receipt (archive reference: `../runs/shared-state-pilot-local/attempt-001/independent-diagnostic-review.json`) preserves the full count checks; no model was rerun. Raw predictions, targets and exact counts are preserved in diagnostic.json (archive reference: `../runs/shared-state-pilot-local/attempt-001/diagnostic.json`).

## Work and cost

The main comparison completed **1,296 physical and retained optimizer updates**, **124,416 training episode exposures**, 3,888 family forwards/backwards, two model constructions and six full model/AdamW snapshots. State targets were derived once per each of the 648 shared bundles. No updates or training lessons were added after viewing scores.

| Stage | Wall seconds | CPU seconds |
| --- | ---: | ---: |
| New development data preparation | 61.438 | 60.000000 |
| Source/data launch freeze | 2.234 | 2.000000 |
| Complete main execution | 418.250 | 390.484375 |
| Separate training-state diagnostic | 4.203 | 3.812500 |
| Target/loss CPU proof | 1.359 | 1.109375 |
| Student CPU proof | 2.197613 | 1.718750 |
| Training-kernel CPU proof | 3.141 | 2.640625 |
| Production-configuration GPU equivalence proof | 3.375 | 2.843750 |

These are recorded stage boundaries, including their imports, verification and publication where specified in each receipt. They do not charge earlier experiments' training-cache construction again, and are not a total project compute bill. Primary recount and this documentation are non-neural; no separate elapsed receipt was recorded for the recount.

The main peak allocated GPU memory was 1,611,475,456 bytes (**1,536.82 MiB**); peak reserved memory was 4,857,004,032 bytes. Main wall time includes 56.242 seconds of data reads, 150.229 of batch preparation, 170.637 of learner steps, 2.654 of shared target generation, 24.032 of evaluation, and the remaining authentication, runtime, checkpoints, publication and orchestration. These measured subdivisions are descriptive and must not be added again to the enclosing main time.

The data receipt records 39,765 canonical generator calls/79,530 returned rows, including reconstruction and admission work; these are not counts of new distinct training lessons. Its archive-load counter is incomplete: it recorded 3 explicit loads, while 217 delegated helper loads also occurred, for **220 actual archive loads**. All are already included in the 61.438-second enclosing preparation time. The frozen receipt is retained with this accounting caveat rather than rewritten.

All four focused proofs passed on their first invocation. Target tests covered 228 rows/2,304 turns and both independent interpreters, with no model work. Student tests used three tiny constructions, 12 forwards, three backwards and two updates. Kernel CPU tests used two tiny constructions, nine forwards/backwards and three updates. The GPU proof used two production-size constructions, six forwards/backwards, 192 episode exposures and two updates, establishing exact zero-weight baseline losses, weights, AdamW and consumed evidence; the auxiliary parameters remained unchanged with absent gradients. These are correctness checks, not general learning or throughput claims.

## Artifact pins

Paths below are relative to the repository root; all identifiers are SHA-256 of the preserved file bytes.

| Artifact | SHA-256 |
| --- | --- |
| `runs/shared-state-pilot-local/attempt-001/launch.json` | `100f899ec4d42748fe2c4a7bb89a79415af695759f11c2c3f6fe117fba37d0e5` |
| `runs/shared-state-pilot-local/attempt-001/execution/summary.json` | `4dd146676f035e409c1befc6b963767e926a4657e83ad7263a57040a961dbb04` |
| `runs/shared-state-pilot-local/attempt-001/analysis.json` | `ccc0e9e514e5a9af2c1be2720c9be111b9fc517f39abc3731e5575c8d1a6b618` |
| `runs/shared-state-pilot-local/attempt-001/diagnostic.json` | `b20e1568bb0ef63531e341a9c40117b22afe4133d47c929de1188c4521d4b64c` |
| `runs/shared-state-pilot-local/attempt-001/diagnostic-receipt.json` | `b0b05e1d1b47a98f6fc009c827b04482b65fa4051d1b0c47bd0fc643413623d5` |
| `runs/shared-state-data-local/attempt-001/manifest.json` | `266dfe643074b8d66b65b1e506f6fc726fa6cb75f2a532b04232ec87bf30eb75` |
| `runs/shared-state-data-local/attempt-001/preparation.json` | `07abfc4213602d9e4cd054e5924f741fcc7529ffa19c87737ef009bbb5b44273` |
| `runs/shared-state-pilot-local/attempt-001/execution/checkpoints/control-0648.pt` | `e8085ed4739bc0bbee72d9cb80581939139bf48b553491ff081911c6f6cbba15` |
| `runs/shared-state-pilot-local/attempt-001/execution/checkpoints/state-0648.pt` | `b2602762c4bbbc15b401d65332c7d702ac93f0df953498b29339a45426ec48c5` |
| `runs/shared-state-targets-validation-local/attempt-001/report.json` | `c76c5d91f654f80c18983b7bc6792d5996f09d627f7412127a5661e796f41dcd` |
| `runs/shared-state-student-validation-local/attempt-001/report.json` | `0d8c686e5c7ad7e79af8ad81b2e8b487f8d8cef81d8868c94b7df0ea0132c488` |
| `runs/shared-state-training-validation-local/attempt-001/report.json` | `4da5bd24803668153b2042c266cba57051cc441fc54ac9adf37d86b7230730b1` |
| `runs/shared-state-training-gpu-validation-local/attempt-001/report.json` | `26731ccd517a7a26f820f74dea9afb578f78e9f76141d0da75a8a4d9fad34ac7` |

## Next

The next step is the prospective [fixed-buffer trainability diagnostic](SHARED_STATE_FIT_PROTOCOL.md): four identically initialized learners fit only the same 108 training rows, comparing joint versus state-only objectives at learning rates 0.003 and 0.0003. Fixed endpoints and separate knownness, conditional-value and native-policy metrics will test whether this shared learner can fit the buffer and help narrow representation versus optimization questions. This is a training-fit diagnostic, not a generalization claim or checkpoint-selection exercise. The broader goal remains reliable general learning and increasing independence from the external tutor, rather than selecting isolated family gains from the failed broad-data screen.
