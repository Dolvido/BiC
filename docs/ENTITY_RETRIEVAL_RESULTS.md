# Causal entity retrieval: completed local result

The comparison **failed its prospective screen**. At the fixed 648-update endpoint, both learners solved **0/360 fresh development pairs** and **0/54 actual training-fit pairs** jointly. Retention was 2/360 for the baseline and 0/360 for retrieval. The candidate also exceeded the allowed regression in switch-family unknown replies. This run does not support adopting the intervention.

The completed execution (archive reference: `../runs/entity-retrieval-pilot-local/attempt-001/execution/summary.json`) and verified pure recount (archive reference: `../runs/entity-retrieval-pilot-local/attempt-001/analysis.json`) preserve all 24 bank endpoints and 12,384 evaluation episode records. The recount did not run another model. The [prospective protocol](ENTITY_RETRIEVAL_PILOT.md) fixed the endpoint and screen before training; no earlier checkpoint replaced the final result.

## Comparison and limits

The baseline was the existing SharedStateStudent. The candidate added learned causal retrieval for all 12 aliases and learned attention from each turn's EOS state to those entity vectors. Both native action and free-reply context used the fused state; auxiliary teaching reused the same retrieved vectors. No parsed state, family label, target, hard entity routing or teacher answer entered native inference. The observation-language branch retained its original path.

Both arms used the original sequence objective plus **0.3 times the same state loss**, AdamW learning rate 0.003, clip 1, seed 852504001, width 192, four layers, four heads and strict local FP32. Every common initial tensor matched exactly. They received the same 648 original-layout bundles, 32 episodes from each of color/count/switch per update, with first-arm order alternating. State targets were computed once per bundle and shared between arms.

The baseline had **2,264,725 parameters**; retrieval had **2,568,277**, an additional 303,552 (+13.40%): 154,944 for entity retrieval and 148,608 for native fusion. This tests the whole architecture change, including added parameters and computation, rather than an isolated parameter-matched mechanism.

Development comprised 720 new episodes from seed 852502001, excluding the recorded-study union of 1,702,440 transcripts. Training-fit reused 108 actual training episodes; retention reused 720 previous development episodes. The transcript census does not exclude semantic equivalents or claim to cover unit-test fixtures. Shared vocabulary, finite English grammar, the single initialization and a curriculum without interfering overwrites limit the inference. There were **zero LLM calls**, and the experiment did not test learned self-direction or persistent memory.

## Native policy results

A joint pair requires both counterfactual anchor questions to receive correct native actions **and** exact freely generated replies. The auxiliary 107-class decoder was unused during these evaluations.

| Final bank | Baseline joint pairs | Retrieval joint pairs | Change |
| --- | ---: | ---: | ---: |
| Fresh development | 0/360 | 0/360 | 0 points |
| Development color | 0/120 | 0/120 | 0 points |
| Development count | 0/120 | 0/120 | 0 points |
| Development switch | 0/120 | 0/120 | 0 points |
| Reused actual training-fit | 0/54 | 0/54 | 0 points |
| Reused retention | 2/360 | 0/360 | −0.56 points |

The screen required at least a 5-point overall development gain, strict joint improvement in every family and in training-fit, no development family known/unknown action or reply loss exceeding 5 points, and no overall joint retention loss exceeding 5 points. The first five improvement checks failed. Switch unknown-reply accuracy fell from **482/598 to 448/598 (80.60% to 74.92%; −5.6856 points)**, failing that regression check too. Retention stayed within its allowed loss margin. Passing one margin does not offset a failed primary condition.

All predetermined trajectories follow. Each triplet is **action-pair / reply-pair / joint-pair correct counts**, with the common denominator shown separately.

| Updates | Bank | Denominator | Baseline A/R/joint | Retrieval A/R/joint |
| ---: | --- | ---: | ---: | ---: |
| 0 | Development | 360 | 0 / 0 / 0 | 0 / 0 / 0 |
| 0 | Training-fit | 54 | 0 / 0 / 0 | 0 / 0 / 0 |
| 0 | Retention | 360 | 0 / 0 / 0 | 0 / 0 / 0 |
| 216 | Development | 360 | 0 / 0 / 0 | 32 / 0 / 0 |
| 216 | Training-fit | 54 | 0 / 0 / 0 | 9 / 0 / 0 |
| 216 | Retention | 360 | 1 / 0 / 0 | 25 / 0 / 0 |
| 432 | Development | 360 | 17 / 19 / 0 | 18 / 12 / 0 |
| 432 | Training-fit | 54 | 6 / 2 / 0 | 0 / 3 / 0 |
| 432 | Retention | 360 | 21 / 20 / 1 | 9 / 14 / 1 |
| 648 | Development | 360 | 28 / 18 / 0 | 33 / 0 / 0 |
| 648 | Training-fit | 54 | 0 / 2 / 0 | 8 / 0 / 0 |
| 648 | Retention | 360 | 17 / 26 / 2 | 30 / 0 / 0 |

No development or training-fit joint pair was solved at any recorded endpoint. Retrieval's separate reply-pair successes at 432 disappeared by 648 on all three banks. A read-only count of its authenticated final development records found **zero Yes replies** across 3,386 queries: 1,567 No, 1,772 ASK and 47 acknowledgment replies. An opposite-answer pair necessarily contains a Yes target, so this output pattern prevents any successful reply pair. The increased final action-pair counts therefore did not produce reliable combined answers.

Final development known/unknown slices retain exact denominators:

| Family / query kind | Baseline action | Retrieval action | Baseline reply | Retrieval reply |
| --- | ---: | ---: | ---: | ---: |
| Color known | 203/528 | 179/528 | 197/528 | 189/528 |
| Color unknown | 308/598 | 355/598 | 376/598 | 395/598 |
| Count known | 219/538 | 201/538 | 199/538 | 191/538 |
| Count unknown | 351/596 | 404/596 | 432/596 | 426/596 |
| Switch known | 201/528 | 188/528 | 149/528 | 185/528 |
| Switch unknown | 381/598 | 409/598 | 482/598 | 448/598 |

All three families improved unknown-action counts while losing known-action counts. Across all 3,386 development queries, action/reply agreement rose from 1,961/3,386 to 2,343/3,386, but total exact replies fell from 1,835/3,386 to 1,834/3,386. Every reply parsed in both arms (3,386/3,386). Agreement and parseability alone therefore provide no evidence of improved factual correctness. On the 1,594 known queries, unsupported action ASK increased from 354 to 460; unsupported reply ASK was 505 versus 503. Other known questions, excluding anchors, scored 261/874 versus 214/874 on actions and 201/874 versus 214/874 on replies. Full family/operator/depth counts remain in the saved records and recount.

## Final training-state diagnostic

The separate [declared diagnostic](ENTITY_RETRIEVAL_DIAGNOSTIC.md) completed after the main comparison. It loaded only the two final checkpoints and used the 108 actual training-fit rows. It performed 18 BOS-only native forwards, 18 auxiliary calls, 216 episode passes and 25,920 state predictions, with no gradients, optimizer, free reply rollouts, held-out state targets or teacher calls. Both complete model states remained unchanged. The raw diagnostic (archive reference: `../runs/entity-retrieval-diagnostic-local/attempt-001/diagnostic.json`), receipt (archive reference: `../runs/entity-retrieval-diagnostic-local/attempt-001/receipt.json`) and independent pure recount (archive reference: `../runs/entity-retrieval-diagnostic-local/attempt-001/verification.json`) preserve and check probabilities, true-value ranks, targets and predictions.

| Training-fit state readout | Baseline known exact | Retrieval known exact | Baseline conditional value | Retrieval conditional value | Descriptive majority value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Color | 2/880 | 0/880 | 189/880 (21.48%) | 202/880 (22.95%) | 269/880 (30.57%) |
| Count | 0/880 | 0/880 | 35/880 (3.98%) | 31/880 (3.52%) | 55/880 (6.25%) |
| Switch | 4/880 | 0/880 | 462/880 (52.50%) | 482/880 (54.77%) | 482/880 (54.77%) |
| Overall | 6/2,640 | 0/2,640 | 686/2,640 (25.98%) | 715/2,640 (27.08%) | 806/2,640 (30.53%) |

Conditional value selects the largest of **all 106 nonzero classes**, without a family mask; it is separate from the full 107-class argmax. The majority column is a post hoc family-specific descriptive reference computed from this same buffer, not a policy input, new fitted model or prospective selection rule. Its preserved baseline receipt (archive reference: `../runs/shared-state-fit-local/fit-buffer-baselines.json`) gives the class census. Equal switch accuracy does not by itself establish identical predictions or a learned state algorithm.

Retrieval predicted unknown as its full argmax for **all 12,960 states**: 10,320/10,320 unknown states correct and 0/2,640 known values correct. Baseline unknown exact was 10,255/10,320. Balanced state loss was 1.564869 for baseline and 1.577494 for retrieval. The same readout did not acquire accurate known-state decoding on the sampled training episodes.

An unknown argmax does **not** establish that the representation contains no facts. Summed known probability can exceed unknown probability while being spread over multiple value classes. With the separately declared strict `P(known) > 0.5` rule, baseline detected 2,344/2,640 known states but correctly rejected only 1,963/10,320 unknown states; retrieval detected 1,765/2,640 and rejected 4,029/10,320. Overall knownness Brier scores were 0.251087 and 0.245507. These probabilities show why argmax, knownness and conditional value must be interpreted separately; the conditional values still provide no strong binding evidence on this buffer. This diagnostic does not prove an absence of latent facts, isolate optimization versus architecture, or measure held-out state transfer.

## Physical work and cost

The main run completed **1,296 physical and retained optimizer updates**, **124,416 training episode exposures**, 3,888 family forwards/backwards, two model constructions and six complete model/AdamW checkpoints. Each arm made 1,944 auxiliary readouts/objectives. The 648 shared target preparations were not duplicated per arm. No optimizer outcome was unknown, and no automatic retry or score-dependent extension occurred.

The 72 saved evaluation groups sum to 432 native forwards, 12,384 episode passes, 123,840 BOS contexts and 123,840 freely decoded contexts. BOS used 123,840 decoder row-steps separately from 3,343,680 free-generation row-steps. These are physical evaluation counts, not additional training.

| Recorded stage | Wall seconds | CPU seconds |
| --- | ---: | ---: |
| New development data preparation | 64.235 | 62.750000 |
| Launch freeze | 2.219 | 2.031250 |
| Entire main execution | 431.578 | 400.359375 |
| Final training-state diagnostic | 4.297 | 3.859375 |
| Model CPU proof | 1.594 | 1.406250 |
| Kernel CPU proof | 2.969 | 2.468750 |
| Production-shape GPU proof | 3.593 | 2.984375 |

The main enclosing time includes 57.358 seconds of data loading, 147.522 of batch preparation, 184.211 of training operations, **3.153 of shared target preparation**, 24.795 of evaluation, 3.703 of bank validation, 0.280 of snapshots and the remaining authentication, publication and orchestration. These subdivisions are already included in 431.578 seconds and must not be added again.

| Per-arm completed-event timing | Baseline | Retrieval |
| --- | ---: | ---: |
| Learner steps, seconds | 87.201 | 95.845 |
| Batch preparation, seconds | 71.693 | 74.412 |
| Training including preparation, seconds | 158.894 | 170.257 |
| Training episode exposures/second, including preparation | 391.51 | 365.38 |
| Native evaluation, seconds | 11.248 | 13.515 |

These per-arm times come from journal completion events and exclude completion-event writing overhead; they can differ slightly from the enclosing operation totals. Retrieval cost **11.363 extra seconds (+7.15%)** for training including preparation, producing **zero additional final development joint pairs**. Shared target preparation is common to both arms and is excluded from this incremental comparison. The timings are observations from this interleaved run, not replicated hardware benchmarks. Peak main allocated GPU memory was **1,709,407,744 bytes (1,630.22 MiB)**; peak reserved memory was 4,762,632,192 bytes (4,542 MiB). The separate diagnostic peaked at 185,734,656 allocated bytes.

Data preparation recorded 39,766 canonical generation calls and 79,532 returned rows, including reconstruction/admission work rather than that many distinct new lessons. Its four explicit archive loads plus 217 delegated loads give **221 actual loads**, all included in the 64.235-second stage. Previous construction of the reused training cache is not charged again. Pure recount and documentation perform no neural work; no separate recount elapsed receipt was recorded. These are stage costs, not a total project compute bill.

All three new proof invocations passed on their first attempt. Six model CPU cases used three tiny models, 15 completed forwards (29 rows), four rejected forwards, two backwards and zero updates. They checked causal prefixes, padding, row isolation, shared initialization, alias equivariance, native slot permutation invariance, cached retrieval identity and gradient paths. Kernel CPU validation used three tiny models, nine forwards/backwards, 18 episode exposures and three updates; baseline weights, complete AdamW, losses and evidence matched the previous kernel exactly. It reconstructed three existing canonical parents. The GPU proof used two production-size models, six forwards/backwards, 192 episode exposures and two updates on one cached bundle, with zero lesson generation. Candidate retrieval, native fusion and shared encoder/decoder gradients were finite and nonzero in the checks; targets stayed outside model inputs. These are correctness checks, not evidence of general learned competence. The frozen 88-source gate and copied proof sources remained unchanged.

## Artifact pins and next decision

The following SHA-256 values identify the preserved bytes. The launch binds protocol/source snapshots, data and proof inputs; its checkpoints remain research artifacts with no automatic promotion.

| Artifact | SHA-256 |
| --- | --- |
| Pilot `launch.json` | `f8dd7c778d1f7202506e3d8d45256e974caec8a5f1e2716935abcd5b888eb514` |
| Main `execution/summary.json` | `c870610de58ce61e9126c6c47c377a72b32e75c41cd6adc5e58f16eae7c1f1b9` |
| Verified `analysis.json` | `69698503fd94e2e08819275bda59d7c621e72dff233d6ba889fcb299914814eb` |
| Data `manifest.json` | `7c0d1f208f59a183cf5c052c991c842c2358107a3c40c84f4fc36ba42fa63612` |
| Data `preparation.json` | `3052a0acdcc1c401f205d14a09b2ffe0d3bd37ee79c11619685c975aa543d13c` |
| Model CPU proof | `21bde3783dc61e12f4cb6daa79b002b8e0c177ba661e6f53a293af4ae4104a5c` |
| Kernel CPU proof | `e4c6e1fd1121237fc7487f58107988cfb22dc27c022669c57f90818426183736` |
| Kernel GPU proof | `bd5dc303c6ca39e6af919e6babb22c42853633a50239b9628d3933e0c2d41b27` |
| Final state `diagnostic.json` | `7afc6366fbe8cafff1516ca07185d69d098709aff6a246d28423539174238774` |
| State diagnostic `receipt.json` | `d38ee2783e8e9fa343e5cb35f596dc1ca7bb0cead0912adf3df4ec881d1d271f` |
| Independent state `verification.json` | `474e1781d71ab636234e622ee3227aaf9e0703e0fdbae3f61c91cf82ec76057b` |

The next approved comparison is the prospective [broad learning-rate diagnostic](SHARED_STATE_RATE_PILOT.md): keep SharedStateStudent, auxiliary weight 0.3, the full broad 648-bundle schedule and seed 852604001 fixed, and compare AdamW rates 0.003 and 0.0003. It reuses the now-observed development bank explicitly, adds no model architecture, and does not rehearse the small fit sample separately. Earlier small-buffer trainability motivates this remaining optimization contrast but does not predict broad-data success. The completed evidence rules out a successful screen for the retrieval setting tested here; it does not establish that retrieval is generally ineffective or prescribe a family-specific repair. No prior learner or checkpoint is promoted.
