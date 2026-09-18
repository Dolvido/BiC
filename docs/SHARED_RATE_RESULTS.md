# Shared learning-rate transition results

Public distribution: [compact aggregate evidence](PUBLIC_EVIDENCE.md) supplies
recountable trajectories, acquisition cells and retention checks. Original artifact
hashes below refer to the full local archive, which is not distributed.

The lower learning rate **0.0001 passed the predeclared acquisition-effect screen**, improving the equally weighted 12-cell mean by **24.3750 percentage points** over unchanged 0.0003. Every acquisition cell was nonregressing; fit, development and every family mean improved. Lower also passed **479/480 retention checks**, versus **465/480** for control.

This was a conditional research improvement, not capability eligibility. Lower passed only **6/12 absolute acquisition cells** and **3/12 original basis capability cells**, versus control **4/12** and **2/12**. Its one retention failure still violated the fixed rule. The separate program-transfer screen failed. Neither branch is eligible or adopted; the home learner remains separately retained at **9,328**.

This document uses only the verified independent JSON report and the preserved validation/recount facts supplied with it. It performs no raw rescoring, inference, training, checkpoint decoding, generation or teacher call.

## Matched comparison and authenticated evidence

Both branches started from the exact final control checkpoint of the objective study at **15,592**, with equal incoming model weights, complete AdamW moments and individual steps, evidence and historical accounting. Control explicitly retained learning rate 0.0003; lower explicitly changed it to 0.0001. Auxiliary weight remained **0.3 in both**. Each consumed the same ordered curriculum of 648 admitted bundles four times: 2,592 new updates per arm, finishing at lifetime **18,184**.

The fixed endpoint order in every trajectory below is **0 / 648 / 1,296 / 2,592** additional updates, corresponding to lifetime **15,592 / 16,240 / 16,888 / 18,184**. The independent reader verified all 16 baseline metrics against the same parent, full-state restart evidence, all step reports, source/data identities and the final publication deadline.

| Evidence | SHA-256 |
| --- | --- |
| Verified independent report (archive reference: `../runs/shared-rate-analysis-local/attempt-002/independent-analysis.json`) | `922679ce9b665c836cd7f21b8af6aa5de0d0c23545da6e7d69ff30bb7e8e2943` |
| Frozen launch (archive reference: `../runs/shared-rate-local/attempt-001/launch.json`) | `057a6a06932749fd712dd10d32b9e68e6625d2db510aa739f8302af4a6bc21cf` |
| Completed summary (archive reference: `../runs/shared-rate-local/attempt-001/execution/summary.json`) | `6345f2128674853ea052712e590660914787defcf75ad75544ae2da59c0231ed` |
| Inclusive completion marker (archive reference: `../runs/shared-rate-local/attempt-001/execution/completion.json`) | `45543fc1cf65a3d2305fd313f13f8895d20137b82c179e5e3096987f032c43a3` |
| [Rate bridge](../experiments/shared_rate_continuation.py) | `b175b25dde588611d6e06219ebd85b5c1de6e39416794a2ce17a475a9ea8c7d2` |
| [Study runner](../experiments/shared_rate_run.py) | `578df65a2a925b88ecc20a141eda82a18e832bc427f22329b4eb1ae2dbf5ade5` |
| [Prospective protocol](../docs/SHARED_RATE_TRANSITION_PROTOCOL.md) | `0ccc7f969f7f692448f170dc8bc4187f698d1b735610069bcddf908827dba038` |
| Corrected independent reader (archive reference: `../runs/shared-rate-analysis-local/recount_v2.py`) | `e719926804e498ddda5100ec951efb33dded62b5aebabbd5aedd0e2fc0103226` |

Common parent checkpoint SHA: `fe13decc662f63f89089c3d9fc628ae74c77d2c6924bc20cfbe28e94e279f3c9`. Reused data manifest SHA: `f5a71eafc49e0b99f9fa893b0c1f9385df26b618e12c79a99b9dd2f7747de896`. The [prospective next-decision memo](SHARED_RATE_NEXT_DECISION.md) was written before reading this study's scores. Completed predecessor studies and this fixed final comparison retain their original criteria.

## All 16 bank trajectories

Entries are successful pair counts. Definition, basis and complementary rows require correct native action and exact free reply on **every query in both counterfactual episodes** (`all_query_pair_both`). The final five broad rows use the known-anchor pair metric (`anchor_pair_both`), an easier and different criterion. Their transfer scores cannot be pooled with definition-program composition/sequence.

| Bank | Pairs per endpoint | Control 0.0003 | Lower 0.0001 |
| --- | --- | --- | --- |
| basis_binding | 96 | 9 / 13 / 8 / 16 | 9 / 21 / 28 / 27 |
| basis_revision | 96 | 80 / 82 / 88 / 81 | 80 / 92 / 90 / 89 |
| basis_composition | 96 | 1 / 0 / 0 / 0 | 1 / 0 / 0 / 0 |
| basis_sequence | 96 | 0 / 0 / 0 / 0 | 0 / 1 / 0 / 1 |
| complementary_fit_binding | 60 | 11 / 15 / 18 / 15 | 11 / 28 / 30 / 43 |
| complementary_fit_revision | 60 | 39 / 35 / 39 / 46 | 39 / 50 / 54 / 55 |
| complementary_dev_binding | 120 | 10 / 27 / 31 / 27 | 10 / 50 / 55 / 62 |
| complementary_dev_revision | 120 | 54 / 68 / 69 / 89 | 54 / 88 / 88 / 97 |
| definition_binding | 96 | 93 / 95 / 91 / 95 | 93 / 96 / 96 / 96 |
| definition_revision | 96 | 96 / 95 / 92 / 96 | 96 / 96 / 96 / 96 |
| definition_composition | 96 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| dev | 324 | 268 / 273 / 266 / 271 | 268 / 274 / 279 / 278 |
| retention | 360 | 277 / 287 / 288 / 281 | 277 / 300 / 298 / 296 |
| train_fit | 54 | 52 / 52 / 51 / 52 | 52 / 53 / 53 / 53 |
| transfer_original | 216 | 144 / 155 / 144 / 149 | 144 / 154 / 160 / 160 |
| transfer_varied | 216 | 148 / 156 / 134 / 141 | 148 / 164 / 164 / 160 |

## Twelve acquisition cells and the fixed screens

The primary cells are complementary fit/development x binding/revision x family. Fit rows are actual practiced examples. Development is held-out text from the same restricted grammar, reused and observed repeatedly. Each cell has equal weight regardless of its pair denominator.

| Cell | Family | Pairs | Control trajectory | Lower trajectory | Final lower-control pp |
| --- | --- | --- | --- | --- | --- |
| fit_binding | color | 20 | 3 / 2 / 6 / 5 | 3 / 9 / 8 / 14 | +45.0000 |
| fit_binding | count | 20 | 4 / 4 / 6 / 5 | 4 / 8 / 10 / 15 | +50.0000 |
| fit_binding | switch | 20 | 4 / 9 / 6 / 5 | 4 / 11 / 12 / 14 | +45.0000 |
| fit_revision | color | 20 | 13 / 12 / 13 / 15 | 13 / 17 / 18 / 19 | +20.0000 |
| fit_revision | count | 20 | 12 / 9 / 11 / 14 | 12 / 16 / 16 / 16 | +10.0000 |
| fit_revision | switch | 20 | 14 / 14 / 15 / 17 | 14 / 17 / 20 / 20 | +15.0000 |
| dev_binding | color | 40 | 3 / 6 / 9 / 6 | 3 / 15 / 19 / 21 | +37.5000 |
| dev_binding | count | 40 | 3 / 10 / 11 / 6 | 3 / 12 / 14 / 14 | +20.0000 |
| dev_binding | switch | 40 | 4 / 11 / 11 / 15 | 4 / 23 / 22 / 27 | +30.0000 |
| dev_revision | color | 40 | 19 / 21 / 24 / 33 | 19 / 31 / 33 / 33 | +0.0000 |
| dev_revision | count | 40 | 13 / 20 / 21 / 21 | 13 / 24 / 20 / 26 | +12.5000 |
| dev_revision | switch | 40 | 22 / 27 / 24 / 35 | 22 / 33 / 35 / 38 | +7.5000 |

The final effect was **+24.3750 pp** across all 12 cells, **+30.8333 pp** on fit and **+17.9167 pp** on development. Family means were color **+25.6250 pp**, count **+23.1250 pp**, switch **+24.3750 pp**. All 12 nonregression checks, both positive split means, all three positive family means and the >=5 pp aggregate condition passed.

Absolute >=75% acquisition remained incomplete: lower passed the three fit-revision cells, count fit-binding, and color/switch development-revision, **6/12** in total. Control passed color/switch revision on fit and development, **4/12**. No development-binding family met 75%. Original basis capability passed only the three revision cells in lower (**3/12**) and color/switch revision in control (**2/12**).

The secondary six-cell basis composition/sequence effect was **+0.5208 pp**, below the required +5 pp, despite all six nonregression checks passing. Lower solved one color sequence pair and no composition pairs; control solved neither. Against the common 15,592 parent, lower's six-cell mean gain was **zero**: the parent's one color composition success was replaced by one color sequence success. Against original 9,760, lower's mean gain was +0.5208 pp; control's was zero. This isolated pair does not establish reusable program execution.

Eligibility required all acquisition and capability cells, positive original-parent transfer and all five retention sets. Both arms failed; `eligible_arms` is empty, no arm was selected, and automatic adoption is false.

## Continued practice, late learning and program reuse

The final interval spans 1,296 updates, twice each earlier interval. The following percentage-point changes therefore describe unequal-length intervals rather than equally spaced slopes.

| Bank | Control full pp | Control late pp | Lower full pp | Lower late pp |
| --- | --- | --- | --- | --- |
| fit_binding | +6.6667 | -5.0000 | +53.3333 | +21.6667 |
| fit_revision | +11.6667 | +11.6667 | +26.6667 | +1.6667 |
| dev_binding | +14.1667 | -3.3333 | +43.3333 | +5.8333 |
| dev_revision | +29.1667 | +16.6667 | +35.8333 | +7.5000 |

Lower fit binding still increased **30 -> 43/60** in the late interval, and its three families increased **8/10/12 -> 14/15/14 out of 20**. Lower development binding increased **55 -> 62/120**; revision increased **54 -> 55/60 fit** and **88 -> 97/120 development**. Continued late acquisition is inconsistent with asserting that learning has stopped. Control also improved late development revision, while its fit binding and development binding fell from the middle endpoint; a single aggregate slope would conceal that difference.

Lower's final fit binding was 43/60 (71.67%), versus development 62/120 (51.67%); fit revision was 55/60 (91.67%), versus development 97/120 (80.83%). These gaps coexist with essentially absent basis composition/sequence and zero prior-definition composition. Learning the practiced complementary procedures more accurately is real progress, but it is not yet evidence that their meanings are reliably reused in unfamiliar multi-step programs.

Native complementary action/reply/state losses fell across every replay pass in both arms and were lower in the lower-rate arm. This supports continuing acquisition within the observed budget; it is not a replacement for exact native pair success.

The remaining binding gap is not merely action/reply disagreement. Lower finished with known action/reply correctness **574/600 and 578/600 on fit**, and **1,106/1,200 and 1,110/1,200 on development**. Its complete known-query scores were 572/600 and 1,097/1,200. Argument-role trajectories below require both action and exact free reply; endpoint order is unchanged.

| Binding query role | Questions | Control trajectory | Lower trajectory |
| --- | --- | --- | --- |
| Fit source | 360 | 284 / 309 / 319 / 318 | 284 / 324 / 327 / 341 |
| Fit destination | 240 | 188 / 205 / 213 / 207 | 188 / 222 / 227 / 231 |
| Development source | 720 | 554 / 607 / 619 / 603 | 554 / 654 / 652 / 663 |
| Development destination | 480 | 346 / 387 / 396 / 392 | 346 / 423 / 425 / 434 |

Lower revision finished at **239/240 source and 116/120 destination on fit**, **472/480 source and 221/240 destination on development**. Prior-value answers were 240/240 and 479/480; unknown answers were perfect at the final endpoint. High prior-value/unknown scores do not prove complete instruction execution. These aggregates cannot isolate Apply-word selection, meaning resolution, argument binding, execution or reply generation as the cause.

## Retention against all five references

Each reference imposes the same 96 checks: 75 broad-bank family joint/known/unknown action/reply checks, nine prior-definition all-query pair checks and 12 basis checks. Every loss must remain within 5 pp. Historical 9,112 is a fixed study reference; it is not the separately retained home learner at 9,328.

| Reference | Control passing checks | Lower passing checks |
| --- | --- | --- |
| common15592 | 93/96 | 96/96 |
| prior13000 | 93/96 | 96/96 |
| previous10408 | 94/96 | 96/96 |
| original9760 | 93/96 | 95/96 |
| historical9112 | 92/96 | 96/96 |

Lower passed **479/480** checks, but its sole failure was **basis revision / switch against original 9,760: 32 -> 30/32, a 6.25 pp loss**. This exceeds the 5 pp limit; the result is not rounded into a pass. Control passed **465/480**, failing 15 checks. All per-check changes remain in the linked verified JSON report. No failing reference is removed or replaced.

## Work, cost and preserved validation history

| Work | Actual |
| --- | --- |
| Physical trajectories / new updates | 2 / 5,184 total, 2,592 per arm |
| Episode exposures | 497,664 total, 248,832 per arm |
| Family forward / backward calls | 15,552 each total |
| Auxiliary readouts / objectives | 15,552 each total, 7,776 per arm |
| Exposure per family per arm | 82,944 |
| Full-state restorations / snapshots | 8 / 6 |
| Teacher calls / newly generated lessons / unknown optimizer outcomes | 0 / 0 / 0 |
| Independent evaluation recount | 128 bank endpoints, 208 raw files, 35,232 episodes |
| Independent update verification | All 5,184 step reports |

Each arm consumed 13,824 basis, 69,120 complementary, 41,472 prior-definition and 124,416 broad-replay episode exposures. They are repeated exposures to existing admitted lessons, not newly generated unique examples. Both auxiliary heads remained active: historical 46,776 state readouts/objectives advanced to 54,552, and all native/auxiliary optimizer steps advanced to lifetime 18,184. Eight auxiliary inspections each bound 20 parameter/AdamW tensors, 515,864 bytes; the initial fingerprints agreed and trained fingerprints changed in both arms. There was no zero-weight frozen-head condition in this study.

| Cost | Wall seconds | CPU seconds |
| --- | --- | --- |
| Data-only freeze and one parent metadata decode | 2.484 | 2.359375 |
| Worker through cleanup and durable summary publication | 2,090.141 | 1,982.765625 |
| Preserved failed first independent recount | 16.890 | 16.812500 |
| Corrected successful independent recount | 16.594 | 16.562500 |

The worker total includes 0.094 seconds of summary publication. Completion-marker publication is separate administrative work and is not assigned an invented duration here. Peak GPU allocation was 1,557,689,856 bytes (1,485.53 MiB), with 4,164,943,872 bytes (3,972 MiB) reserved. Enclosing and nested restore times overlap and are not summed twice. Earlier data preparation, prior learners and runtime proofs remain separate historical costs; the table is not a claim to include the entire project's cost.

Both CPU and CUDA numerical proofs passed their first invocation, validating full incoming state, rate-specific updates and reload; they establish numerical correctness, not learning benefit. Their authenticated report/marker pins remain in the independent report's input inventory.

The first recount (archive reference: `../runs/shared-rate-analysis-local/attempt-001/independent-analysis.json`), SHA `950a9e38b5916f7d0f91fb329bb7f7f461a6c996c7e2c62fd8af91231977dae0`, failed because its reader expected nested snapshot counters absent from the frozen producer. Exact counters were present at the outer rate-restore level. The original reader/failure stayed unchanged; additive `recount_v2.py` validated the real nested status/cursor/timing shape while retaining outer counters. Eight pure tests passed, then the second saved-evidence recount verified. No training or inference was rerun, and both recount costs are shown above.

## Next research decision and limits

The final lower-rate branch is a **conditional research baseline** for further diagnosis. It gives a broad, predeclared acquisition benefit and better preservation at equal exposure, while retaining the same auxiliary objective and curriculum. It is not an adopted home checkpoint: absolute acquisition, original capability and one historical retention check still fail.

The next recommended step is to **isolate Apply-word selection before paying for a new rule head**. A prospectively specified native-policy diagnostic should test whether changing only the applied word changes behavior according to its defined meaning, with definitions and facts held fixed, checking both words and source/destination roles across all families with action and free replies. Any controlled change needs independent target validation; no parser or auxiliary answer enters policy inputs. Current aggregates motivate this hypothesis but do not establish its cause. This is a research recommendation, not an executed diagnostic or automatic training extension.

The result is conditional on one inherited AdamW history and reused schedule. Effective weight decay also scales with rate, so the gain does not uniquely identify a representation or gradient-noise mechanism. Repeatedly observed banks and a finite grammar do not establish general English, learned self-direction or tutor benefit. Late fit-binding gains do not support a plateau claim; finite checkpoints establish no home-hardware ceiling. Neither a broad-bank transfer gain nor one basis sequence pair proves reusable program learning.

Any home adoption requires a separate prospective fresh comparison with the then-current home learner, covering unaided capability and retention. The home lineage stays unchanged at **9,328**. No automatic extension, promotion or family-specific repair follows.
