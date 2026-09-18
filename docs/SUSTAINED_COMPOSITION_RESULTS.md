# Sustained composition: completed acquisition comparison

The completed comparison is valid evidence of continued learning, but **neither branch passes the research screen or qualifies for adoption**. The curriculum branch improved practiced and development composition acquisition across all three families, with its largest complete-pair gains in the final two passes. It still failed every complementary acquisition cell, all held-out basis composition/sequence cells, and retention checks. The home learner remains separately retained at update **9,328**.

This report uses only the authenticated independent recount (archive reference: `../runs/sustained-composition-analysis-local/attempt-001/independent-analysis.json`). It performs no additional scoring, checkpoint decoding, training, teacher call or lesson generation. The [prospective protocol](SUSTAINED_COMPOSITION_PROTOCOL.md) fixed all endpoints and decision rules before execution.

## Comparison and admissible conclusions

Each branch continued its own update-10,408 checkpoint, including full AdamW state. The two starting checkpoints already differed because of the preceding comparison. Both replayed their exact 648-update schedule four times, reaching update 13,000. This is continuation of the original curriculum comparison, not a new randomized experiment from equal weights. Every branch baseline reproduced all 16 of its preceding final panels.

The curriculum branch received 720 additional composed teaching updates (69,120 episode exposures), 144 atomic teaching updates, 432 earlier-definition updates and 1,296 broad replay updates. Control received 864 atomic teaching updates and the same earlier-definition/replay counts. These are repeated exposures to existing lessons; the study generated no new examples.

The completion marker admits the run as a complete, within-budget execution and its checkpoints as usable research archives. Its operational `continuation_eligible: true` is **not** the scientific eligibility result: both scientific `eligible` flags are false. No checkpoint was selected automatically.

## All fixed native trajectories

Each sequence below is the count at additional updates **0 / 648 / 1,296 / 2,592**. Definition and complementary rows require correct native actions and exact freely generated replies on every question in both counterfactual episodes. The five broad banks use their existing joint focal-pair metric; their denominators and easier success criterion must not be conflated with complete definition pairs.

| Bank | Pair denominator | Control counts | Curriculum counts |
| --- | ---: | --- | --- |
| Basis binding | 96 | 31 / 41 / 46 / 45 | 3 / 8 / 3 / 5 |
| Basis revision | 96 | 89 / 91 / 89 / 91 | 55 / 66 / 54 / 76 |
| Basis composition | 96 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Basis sequence | 96 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Complementary fit binding | 60 | 0 / 0 / 0 / 0 | 0 / 1 / 0 / 8 |
| Complementary fit revision | 60 | 3 / 2 / 1 / 1 | 7 / 5 / 7 / 17 |
| Complementary development binding | 120 | 0 / 0 / 0 / 1 | 0 / 0 / 2 / 11 |
| Complementary development revision | 120 | 2 / 5 / 3 / 2 | 10 / 12 / 9 / 35 |
| Earlier definition binding | 96 | 96 / 94 / 92 / 95 | 90 / 96 / 93 / 95 |
| Earlier definition revision | 96 | 96 / 94 / 91 / 93 | 93 / 95 / 92 / 96 |
| Earlier definition composition | 96 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Broad development | 324 | 269 / 271 / 264 / 254 | 263 / 261 / 270 / 261 |
| Broad retention | 360 | 280 / 291 / 279 / 284 | 280 / 278 / 278 / 283 |
| Broad training-fit | 54 | 51 / 49 / 49 / 50 | 50 / 50 / 53 / 50 |
| Broad structural transfer | 216 | 158 / 146 / 145 / 148 | 141 / 140 / 140 / 138 |
| Broad varied-position transfer | 216 | 156 / 148 / 143 / 148 | 145 / 138 / 150 / 150 |

Basis composition and sequence stayed **0/32 in every family at every endpoint** in both branches. Their six-cell final curriculum-minus-control mean is zero, below the required five-point benefit. The earlier definition-composition panel also remained zero. Better performance on broad structural transfer is a different task and does not establish new definition-program transfer.

## Acquisition by family and query

Family triplets are **color, count, switch**. Each fit family has 20 pairs; each development family has 40. The acquisition rule requires at least 75% in every one of these 12 cells. Both branches failed all 12.

| Branch / complementary bank | 0 | 648 | 1,296 | 2,592 |
| --- | --- | --- | --- | --- |
| control / fit binding | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| control / fit revision | 1, 1, 1 | 1, 1, 0 | 1, 0, 0 | 0, 1, 0 |
| control / development binding | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 1, 0 |
| control / development revision | 1, 0, 1 | 1, 2, 2 | 1, 0, 2 | 1, 0, 1 |
| curriculum / fit binding | 0, 0, 0 | 0, 0, 1 | 0, 0, 0 | 2, 3, 3 |
| curriculum / fit revision | 2, 3, 2 | 2, 2, 1 | 2, 3, 2 | 6, 6, 5 |
| curriculum / development binding | 0, 0, 0 | 0, 0, 0 | 0, 1, 1 | 3, 3, 5 |
| curriculum / development revision | 4, 2, 4 | 4, 3, 5 | 3, 5, 1 | 13, 8, 14 |

Curriculum final practiced binding is 8/60 (13.33%) and revision 17/60 (28.33%); development is 11/120 (9.17%) and 35/120 (29.17%). Thus incomplete acquisition occurs on actual training rows, not solely on new texts. The gains are present in every family, but no family approaches the complete-pair requirement.

The following curriculum query counts preserve the distinction between recalling an unchanged value and executing an update. Each count requires both action and free reply to be correct.

| Complementary bank | Known queries: start → final | Destination queries: start → final | Final destination by family |
| --- | --- | --- | --- |
| Complementary fit binding | 350/600 → 458/600 | 122/240 → 177/240 | 62/80, 57/80, 58/80 |
| Complementary fit revision | 474/600 → 519/600 | 63/120 → 74/120 | 24/40, 24/40, 26/40 |
| Complementary development binding | 673/1200 → 843/1200 | 209/480 → 301/480 | 102/160, 99/160, 100/160 |
| Complementary development revision | 947/1200 → 1043/1200 | 114/240 → 147/240 | 50/80, 43/80, 54/80 |

Revision prior-value questions remained perfect: 240/240 on fit and 480/480 on development, at both start and finish. Final unknown actions and replies were perfect in each family of all four complementary banks. Neither fact substitutes for correct composed destinations. Action/reply agreement improved on all four banks, while destination errors remain substantial; agreement alone is insufficient.

## Retention and eligibility

Each reference contributes 96 checks: 75 broad-bank family joint/known/unknown action/reply checks, nine earlier-definition family complete-pair checks, and 12 basis family complete-pair checks. The maximum allowed loss is five percentage points per check.

| Reference | Control failed checks /96 | Curriculum failed checks /96 |
| --- | ---: | ---: |
| Own immediate update-10,408 parent | 7 | 3 |
| Original shared research update-9,760 parent | 1 | 4 |
| Fixed historical update-9,112 reference | 6 | 7 |

Against its immediate parent, curriculum lost 7.41 points on broad development switch pairs, 5.56 points on broad training-fit count pairs, and 6.94 points on varied-position switch pairs. Against the original shared parent it also failed basis color binding, basis switch revision and count known-action structural-transfer checks. Against the historical reference, count structural-transfer joint pairs fell 15.28 points, with known actions/replies down 7.19/7.84 points. These are reported failures, not targets for family-specific repair.

Control failed seven immediate-parent checks despite improving atomic binding from 31 to 45/96. Its broad development total fell 269→254/324 and broad structural transfer 158→148/216. Curriculum broad retention increased only 280→283/360 while its family-level failures remained. Aggregate improvement does not cancel a failed retention condition.

The original basis-capability screen also failed: control passed only its three revision cells; curriculum passed only color and switch revision. All six composition/sequence gains against each of the three references are zero. Consequently neither branch satisfies acquisition, capability, positive shared-parent transfer gain and all retention requirements together. The historical 9,112 reference is preserved for experimental comparability; it is not the current 9,328 home learner.

Final family counts for the remaining banks are provided below in color/count/switch order. The independent report retains every family array at every endpoint.

| Bank | Control final | Curriculum final |
| --- | --- | --- |
| Basis binding | 17/32, 13/32, 15/32 | 1/32, 3/32, 1/32 |
| Basis revision | 32/32, 29/32, 30/32 | 25/32, 21/32, 30/32 |
| Basis composition | 0/32, 0/32, 0/32 | 0/32, 0/32, 0/32 |
| Basis sequence | 0/32, 0/32, 0/32 | 0/32, 0/32, 0/32 |
| Earlier definition binding | 32/32, 32/32, 31/32 | 31/32, 32/32, 32/32 |
| Earlier definition revision | 32/32, 30/32, 31/32 | 32/32, 32/32, 32/32 |
| Earlier definition composition | 0/32, 0/32, 0/32 | 0/32, 0/32, 0/32 |
| Broad development | 93/108, 65/108, 96/108 | 98/108, 68/108, 95/108 |
| Broad retention | 101/120, 75/120, 108/120 | 106/120, 70/120, 107/120 |
| Broad training-fit | 17/18, 16/18, 17/18 | 18/18, 14/18, 18/18 |
| Broad structural transfer | 55/72, 34/72, 59/72 | 59/72, 25/72, 54/72 |
| Broad varied-position transfer | 56/72, 35/72, 57/72 | 61/72, 34/72, 55/72 |

## Learning rate of progress and next decision

Losses continued to decrease on the same ordered cache. These are mean training losses during each pass, not loss measurements on a fixed frozen model.

| Replay pass | Curriculum composed action loss | Curriculum composed total loss | Control atomic action loss | Control atomic total loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.300370 | 0.421544 | 0.063107 | 0.117361 |
| 2 | 0.272477 | 0.383245 | 0.046495 | 0.091711 |
| 3 | 0.247880 | 0.352171 | 0.040236 | 0.079233 |
| 4 | 0.227675 | 0.325919 | 0.035739 | 0.070082 |

Absolute per-pass loss reductions became smaller, but **complete-pair acquisition did not exhibit a demonstrated plateau**. The largest gains arrived between 1,296 and 2,592 additional updates: curriculum fit binding 0→8, fit revision 7→17, development binding 2→11 and development revision 9→35. That interval contains two passes, unlike the first two intervals; even after acknowledging the longer interval, describing acquisition as flat would be inaccurate. No claim about eventual convergence follows from four endpoints.

The [prewritten decision plan](SUSTAINED_COMPOSITION_NEXT_DECISION.md) therefore supports retaining a fixed continued-practice control before changing architecture. The smallest informative next mechanism comparison is **the same curriculum update-13,000 research parent with state auxiliary weight 0.3 versus an explicit transition to weight zero**, using identical admitted examples, updates and endpoints. This keeps the continuing positive trajectory as the control and tests auxiliary influence without adding model parameters. A context-detached auxiliary is a distinct, more specific alternative that continues training the auxiliary head; it must not be mislabeled as the weight-zero intervention.

Any changed objective needs a new authenticated transition identity that preserves incoming model weights and full optimizer state. The existing exact-continuation recipe must remain unchanged. Prespecify cross-family fit, transfer and retention measurements, retain the three historical comparisons and the immediate research parent, and keep the home lineage separate. A benefit from auxiliary removal would support a conditional objective effect, not prove that invariant entity targets caused harmful gradients: the same EOS representation can encode rule information in other directions. This report recommends a study; it does not declare its implementation or execution complete.

The present result warrants neither another architecture change nor a claim that more practice cannot help. It establishes weak but continuing acquisition, absent complete-pair structural transfer on these panels, and retention tradeoffs. Repeatedly observed finite-grammar development banks are not a fresh audit of general English competence.

## Physical work, time and evidence

- Additional work: 5,184 physical updates, 497,664 episode exposures, 15,552 family forwards/backwards, eight exact full-state restorations and six snapshots; zero unknown optimizer outcomes and zero teacher calls.
- Each branch consumed 248,832 exposures, equally split into 82,944 per family. Existing lessons and evaluation rows were reused.
- Independent recount: 128 bank endpoints, 208 score files, 35,232 evaluated episodes, 182,480 queries and all 5,184 step reports. It also authenticated 504 existing JSON learning images and 15 prior broad-bank inventory files; these were not new lessons or new model evaluations.
- Peak allocated GPU memory: 1,557,689,856 bytes (1,485.53 MiB); peak reserved: 4,164,943,872 bytes (3,972 MiB).

| Stage | Wall seconds | CPU seconds |
| --- | ---: | ---: |
| CPU freeze, including two metadata-only checkpoint loads | 2.406 | 2.15625 |
| Inclusive worker through cleanup and durable summary publication | 2,174.875 | 2,056.59375 |
| Completion-marker administrative publication, separately reported by launcher | 0.015 | 0 |
| Independent raw-record recount | 17.656 | 17.203125 |

The saved summary stops its timer before its own publication (2,174.781 wall seconds); the completion marker includes that publication and is the correct inclusive worker time. Its 0.094-second summary-publication component is already included. The inherited data-preparation cost is historical, not repeated here. Engineering and earlier validation costs are not a complete project compute bill.

| Artifact | SHA-256 |
| --- | --- |
| Launch | `4bd912311f64cf25939aa7eb437bfd09e265d23ed2b9ab7911e32ef237cbe40d` |
| Completed summary | `fb46f0972dd37d5d273dfee51a017b63fc22c2e0d6856cd65ba2746421da7246` |
| Completion marker | `db150cd061e6876e11495163a12fe7d2f2655aae000207faf8859a3fb45f1335` |
| Independent recount | `7ac87b22735316769293576f9e90996f773a1b2effbf4a6b5169c7f76b1c4acf` |
| Reused data manifest | `f5a71eafc49e0b99f9fa893b0c1f9385df26b618e12c79a99b9dd2f7747de896` |

Execution artifacts are under `runs/sustained-composition-local/attempt-001`; the independent report is under `runs/sustained-composition-analysis-local/attempt-001`. No released checkpoint or home learner was replaced.
