# Fixed-buffer shared-state trainability result

The completed four-arm diagnostic shows that the existing native learner **can fit this small training buffer**: `joint_slow`, using learning rate 0.0003, answered all 516 query turns correctly in both action and generated reply at the fixed 432-update endpoint. It solved 54/54 counterfactual anchor pairs, compared with 39/54 for `joint_fast` at 0.003. However, **all four arms failed the predeclared dense-state adequate-fitting conditions**. Native query fitting and dense entity-state decoding are distinct outcomes.

This is repeated fitting of 108 already-seen training episodes, not a generalization result. The fitting experiment itself used no development or audit evaluation. A separately declared, subsequently completed reused-development check found only 10/360 joint pairs for `joint_slow`, versus 23/360 for `joint_fast`; its results appear below. No LLM was called and no checkpoint was promoted. Memorization remains a possible explanation. The fixed [protocol](SHARED_STATE_FIT_PROTOCOL.md), completed execution (archive reference: `../runs/shared-state-fit-local/attempt-001/execution/summary.json`), and verified pure recount (archive reference: `../runs/shared-state-fit-local/attempt-001/analysis.json`) retain the fitting evidence.

## Comparison and admission

Four fresh SharedStateStudents began with identical full parameters, seed 852404001, using the unchanged width-192/four-layer/four-head/feedforward-768 architecture and shared 107-class auxiliary decoder. They used strict FP32 on the same local RTX 5080, AdamW defaults except the declared learning rate, and gradient clipping at 1. No weights from the failed broad-data pilot were reused.

All conditions made the same teacher-forced native forward and state readout. Joint arms differentiated the original sequence objective plus 0.3 times the state loss; state-only arms differentiated only 0.3 times the state loss. Each update averaged the three family losses before a single optimizer step. State-only native output heads received no direct objective, so their native scores are descriptive rather than an adoption comparison.

The authenticated source was the previous 108-row actual training-fit bank: twelve episodes per family×turn-length group, across color/count/switch and lengths 8/10/12. Inputs and causal targets were prepared once and reused. Every prefix included all 12 aliases; labels stayed outside native observation inputs. The auxiliary loss retained equal known/unknown group means per example×turn, then equal example/turn and family averaging. The archive also contained other banks; those were decoded and discarded without scoring or generating state targets.

Each arm completed 432 updates, rotating lengths 8→10→12, for 144 passes over the buffer and 15,552 episode exposures. The first arm rotated each round, giving each condition equal ordering positions. All models finished the predetermined schedule; intermediate checkpoints were not substituted for the final endpoint.

## Native policy fitting

Joint anchor-pair correctness requires both counterfactual anchors to have correct actions and exact freely generated replies. It is narrower than correctness on every query turn.

| Arm | Learning rate | Differentiated objective | Joint pairs at 0 | At 216 | At 432 |
| --- | ---: | --- | ---: | ---: | ---: |
| `joint_fast` | 0.003 | Original + 0.3 state | 0/54 | 38/54 | 39/54 |
| `joint_slow` | 0.0003 | Original + 0.3 state | 0/54 | 54/54 | 54/54 |
| `state_fast` | 0.003 | 0.3 state only | 0/54 | 0/54 | 0/54 |
| `state_slow` | 0.0003 | 0.3 state only | 0/54 | 0/54 | 0/54 |

**The 216-update `joint_slow` result was not complete query fitting.** It had 516/516 correct query actions, but only 252/516 correct replies: all 252 known-query replies were correct and **0/264 unknown-query replies** were correct. Thus its 54/54 known-anchor pairs concealed a systematic failure elsewhere. By the fixed 432-update endpoint, both heads were correct on all 516 query turns, including all 264 unknown queries and all 144 known queries outside the anchors.

| Final native measure | `joint_fast` action / reply | `joint_slow` action / reply | `state_fast` action / reply | `state_slow` action / reply |
| --- | ---: | ---: | ---: | ---: |
| All query turns, /516 | 491 / 487 | 516 / 516 | 94 / 0 | 98 / 0 |
| Known query turns, /252 | 231 / 227 | 252 / 252 | 92 / 0 | 89 / 0 |
| Other known queries excluding anchors, /144 | 137 / 134 | 144 / 144 | 56 / 0 | 54 / 0 |
| Unknown query turns, /264 | 260 / 260 | 264 / 264 | 2 / 0 | 9 / 0 |

At the final endpoint, `joint_fast` solved 18/18 color, 3/18 count and 18/18 switch anchor pairs; `joint_slow` solved 18/18 in each family. The slower rate improved native training fitting under this fixed setup. This does not establish its transfer, retention, compute efficiency on fresh lessons, or independence from a tutor.

## Dense state readouts

All readouts use the same 107-class logits. “Known exact” requires the ordinary full argmax to equal the actual value. “Conditional value” chooses among all 106 nonzero classes, with **no family mask**. Knownness uses the fixed rule P(known)=1−P(class 0)>0.5; sensitivity is correct detection of known entries, and specificity is correct rejection of unknown entries. The latter is not the same as unknown winning the 107-way argmax.

The prospective adequate-fitting definition required, **in each family at update 432**, sensitivity and specificity ≥95%, conditional value accuracy ≥90%, and ordinary known exact accuracy ≥80%. All four arms failed; no threshold was relaxed after seeing scores.

| Arm | Family | Known exact | Conditional value | Known sensitivity | Unknown specificity |
| --- | --- | ---: | ---: | ---: | ---: |
| `joint_fast` | color | 207/880 (23.52%) | 536/880 (60.91%) | 482/880 (54.77%) | 2884/3440 (83.84%) |
| `joint_fast` | count | 378/880 (42.95%) | 555/880 (63.07%) | 653/880 (74.20%) | 2940/3440 (85.47%) |
| `joint_fast` | switch | 336/880 (38.18%) | 622/880 (70.68%) | 603/880 (68.52%) | 2556/3440 (74.30%) |
| `joint_slow` | color | 607/880 (68.98%) | 732/880 (83.18%) | 788/880 (89.55%) | 3012/3440 (87.56%) |
| `joint_slow` | count | 681/880 (77.39%) | 820/880 (93.18%) | 772/880 (87.73%) | 3122/3440 (90.76%) |
| `joint_slow` | switch | 616/880 (70.00%) | 758/880 (86.14%) | 787/880 (89.43%) | 3107/3440 (90.32%) |
| `state_fast` | color | 717/880 (81.48%) | 761/880 (86.48%) | 827/880 (93.98%) | 3039/3440 (88.34%) |
| `state_fast` | count | 748/880 (85.00%) | 764/880 (86.82%) | 871/880 (98.98%) | 3295/3440 (95.78%) |
| `state_fast` | switch | 618/880 (70.23%) | 715/880 (81.25%) | 771/880 (87.61%) | 2990/3440 (86.92%) |
| `state_slow` | color | 620/880 (70.45%) | 760/880 (86.36%) | 819/880 (93.07%) | 3051/3440 (88.69%) |
| `state_slow` | count | 657/880 (74.66%) | 833/880 (94.66%) | 786/880 (89.32%) | 3183/3440 (92.53%) |
| `state_slow` | switch | 683/880 (77.61%) | 784/880 (89.09%) | 808/880 (91.82%) | 2992/3440 (86.98%) |

Each family contributes 880 known and 3,440 unknown dense targets. For context, the fixed-buffer majority-value conditional baselines are 30.57% for color, 6.25% for count, and 54.77% for switch. These are repeated prefix targets, not independent examples.

| Arm, final pooled readout | Known exact | Conditional value | Known sensitivity | Unknown specificity | Mean true-value rank | Brier score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `joint_fast` | 34.89% | 64.89% | 65.83% | 81.20% | 1.5754 | 0.145429 |
| `joint_slow` | 72.12% | 87.50% | 88.90% | 89.54% | 1.1568 | 0.092711 |
| `state_fast` | 78.90% | 84.85% | 93.52% | 90.35% | 1.1811 | 0.065377 |
| `state_slow` | 74.24% | 90.04% | 91.40% | 89.40% | 1.1186 | 0.093905 |

Pooled success cannot replace the per-family criteria: for example, `state_slow` exceeds 90% pooled conditional accuracy but misses it for color and switch. True-value rank is 1 plus the number of strictly greater nonzero logits; argmax ties select the lowest class index. The preserved state records contain targets, both argmax predictions, ranks and probabilities for all endpoints; exact confusion counts, probability means, ordinary unknown accuracy and complete trajectories remain in the execution/recount artifacts.

The lower joint rate substantially improved fitting of both the native queries and the dense targets. Removing the native objective at the faster rate also improved dense-state fitting relative to `joint_fast`, but no arm achieved the declared adequate dense fit. These observations make optimization and objective interaction useful hypotheses; they do not uniquely identify a cause or establish an architectural ceiling. The perfect final native training queries demonstrate some trainability while leaving dense-prefix coverage and transfer unresolved.

## Work, cost and verification

The run completed **1,728 synchronized optimizer updates**, **62,208 training episode exposures**, and **5,184 family training forwards/backwards and auxiliary readouts**, with zero unknown optimizer outcomes. Four fresh models and eight full model/AdamW checkpoints were retained. Nine target packs covered the 108 distinct buffer rows once. Auxiliary evaluation used 108 native forwards/readouts and 1,296 episode passes; native policy evaluation independently used 108 forwards and another 1,296 episode passes, including free replies. The state records contain 155,520 entity predictions. No new training lessons were sampled; canonical admission reconstructed 54 parent pairs and materialized 108 layout rows, recorded in `bank_validation_work`. No LLM was called.

| Recorded stage or cost | Wall seconds | CPU seconds |
| --- | ---: | ---: |
| Source/input freeze, separate from execution | 1.453 | 1.140625 |
| Entire execution | **136.531** | **106.109375** |
| Training operations, included above | 118.909 | 91.906250 |
| Native evaluation operations, included above | 9.672 | 7.578125 |
| Auxiliary evaluation operations, included above | 1.138 | 0.890625 |
| Checkpoint operations, included above | 0.640 | 0.562500 |
| Pure postrun recount, separate | 0.734 | Not separately recorded |
| Four pure validation tests, separate | 0.000 recorded | 0.015625 |

Peak GPU allocation was **662,683,136 bytes (631.9839 MiB)**. Initial setup took 3.937 seconds within the enclosing execution. Operation categories are not all disjoint: publication includes checkpoint writes nested inside checkpoint operations. Neither those nested costs nor setup are added again to the full 136.531 seconds. The validation wall clock recorded zero at its available resolution; that does not mean no computation occurred. Earlier cache-generation and existing-model proof costs remain in their original receipts rather than being charged again here.

Four pure metric/threshold tests passed, with zero Torch imports, tensors, models, bank loads, GPU calls or updates. Static source review checked objective gradients, exact matching initialization, rotations, knownness/conditional/native separation and deadline/failure handling before launch. The frozen runner/source closures, runtime and input pins were authenticated. The pure recount verified all 108 native groups, 12 state endpoints and integer fitting decisions; it used an explicit 1e−12 absolute/relative tolerance only for floating aggregates whose summation order changed when JSON keys were sorted. It performed no new neural work. No old source, receipt or checkpoint was rewritten.

## Frozen artifact identities

Paths are relative to the repository root; hashes are SHA-256 of preserved bytes.

| Artifact | SHA-256 |
| --- | --- |
| `docs/SHARED_STATE_FIT_PROTOCOL.md` | `89295887a370080d2a0853fb6458c81c0f38f3ebe3e855e9f3cb7424275adef5` |
| `experiments/shared_state_fit.py` | `0de6121556fb64149dd695673cd26536aa8bdcc485289eb6ddacb24869011348` |
| `runs/shared-state-fit-local/attempt-001/launch.json` | `1b8a920caf81d3c0dbc02a447adc0480505577192632003a913df147003642e1` |
| `runs/shared-state-fit-local/attempt-001/execution/summary.json` | `9e4de8297cf917d54d58306fb1337ea407089cfef5ea19eee1f274286b383f5c` |
| `runs/shared-state-fit-local/attempt-001/analysis.json` | `1298704c5d36885d3769ac7b39f8ecda5a7f4f016ba0fab236ae852c821b132e` |
| `runs/shared-state-fit-validation-local/attempt-001/report.json` | `1a4114f05aa239ce0cff8a968cbac33c0989bc44c1501338704d36671188fcd3` |
| `runs/shared-state-fit-local/fit-buffer-baselines.json` | `2f224be654f6686b107908d77000dc110b6340d27404ceaa85fbd448b1557c88` |
| `runs/shared-state-fit-local/attempt-001/execution/checkpoints/joint_fast-0432.pt` | `fcd63e31f8c4a86fd70f67e30c138b85a10216d1660eb12d48a7fdd7277b8184` |
| `runs/shared-state-fit-local/attempt-001/execution/checkpoints/joint_slow-0432.pt` | `8c4cf03c4165186c13fc27156d87815449467b847d25a6ab40b39debaf210af5` |
| `runs/shared-state-fit-local/attempt-001/execution/checkpoints/state_fast-0432.pt` | `d2ed86fb4f4bf20b15658bb8507db1c4c7730bd459ebda3711e22e85aac62622` |
| `runs/shared-state-fit-local/attempt-001/execution/checkpoints/state_slow-0432.pt` | `29d317cf35baf2defb3fa34ee4ed7a220c6aef31ec0d8624878951b67b42aefd` |

The exact shared initial full-state digest was `a666ad47ec13b476a2dc1dd9cee24f621a202f3204bb5004222ab303d4a34b28`. Source data manifest and bank pins remain `266dfe643074b8d66b65b1e506f6fc726fa6cb75f2a532b04232ec87bf30eb75` and `c42c117d04070cae026e6372615bcb044b493ff41f80603a268dc8aff93e4efb`, respectively.

## Completed transfer check on reused development

After the fitting result, the separate [transfer protocol](SHARED_STATE_FIT_TRANSFER.md) was fixed before these checkpoints were scored on the selected bank. It compared **both predetermined final 432-update joint checkpoints**, without further training, on all 720 episodes of the existing development bank. State-only arms were excluded by their training objective, not by development scores. Native actions and replies were evaluated without the auxiliary head or parsed state as input.

The bank had been inspected in the older shared-state pilot, so it is **reused development data, not a pristine audit or independent replication**. Direct comparison of full observed-text tuples found 720 unique development transcripts, 108 unique fitting transcripts, and **zero overlap**. Shared finite grammar, vocabulary, patterns and semantic equivalents still limit the claim.

| Joint action-and-reply anchor pairs | `joint_fast` | `joint_slow` |
| --- | ---: | ---: |
| Practiced training-fit bank | 39/54 (72.22%) | 54/54 (100%) |
| Reused development, all families | **23/360 (6.39%)** | **10/360 (2.78%)** |
| Development: color | 11/120 | 4/120 |
| Development: count | 0/120 | 0/120 |
| Development: switch | 12/120 | 6/120 |

Perfect practiced-query performance therefore did **not** carry over to this disjoint bank. The slower rate is not justified as a general winner: it fitted the training sample better, yet produced fewer joint pairs on this reused development check. Neither rate produced strong transfer, and no count-family pair was jointly correct in either arm. This does not prove purely rote fitting or an inability of the architecture to generalize.

| Reused-development query slice | `joint_fast` action / reply | `joint_slow` action / reply |
| --- | ---: | ---: |
| All queries, /3,392 | 1,292 / 1,269 | 1,391 / 1,268 |
| Known queries, /1,580 | 384 / 372 | 288 / 214 |
| Other known queries excluding anchors, /860 | 162 / 162 | 98 / 68 |
| Unknown queries, /1,812 | 908 / 897 | 1,103 / 1,054 |

The slower model's higher overall action count came with more correct abstentions on unknown queries and fewer correct answers to known queries. Its known-query action/reply accuracies were 18.23%/13.54%, versus 24.30%/23.54% for the faster model; unknown-query accuracies were 60.87%/58.17%, versus 50.11%/49.50%. Overall accuracy alone would obscure that tradeoff.

| Family and arm | Known action / reply | Unknown action / reply |
| --- | ---: | ---: |
| Color, `joint_fast` | 118/528 / 113/528 | 272/596 / 265/596 |
| Color, `joint_slow` | 79/528 / 55/528 | 365/596 / 338/596 |
| Count, `joint_fast` | 132/528 / 127/528 | 333/616 / 334/616 |
| Count, `joint_slow` | 107/528 / 83/528 | 349/616 / 333/616 |
| Switch, `joint_fast` | 134/524 / 132/524 | 303/600 / 298/600 |
| Switch, `joint_slow` | 102/524 / 76/524 | 389/600 / 383/600 |

The transfer invocation completed in **9.859 wall seconds / 7.953125 CPU seconds**, with peak GPU allocation **496,107,520 bytes (473.125 MiB)**. It loaded one bank archive and two pinned final checkpoints, constructed two models, and completed **54 native forwards / 1,440 episode passes**. It made **zero optimizer, backward, auxiliary-head or tutor calls** and verified unchanged full model states. Canonical admission reconstructed 360 parent pairs/materialized 720 layout rows; this validation work is recorded separately and included in the enclosing wall time. No new training lessons were sampled.

The completed transfer receipt (archive reference: `../runs/shared-state-fit-transfer-local/attempt-001/summary.json`) preserves all raw action/reply groups, counts, transcript identities, validation work, runtime and costs. The verified pure recount (archive reference: `../runs/shared-state-fit-transfer-local/attempt-001/analysis.json`) checked all 18 groups and 1,440 episode records without new neural work, taking a separate 0.359 wall seconds. Key transfer pins are:

| Artifact | SHA-256 |
| --- | --- |
| `docs/SHARED_STATE_FIT_TRANSFER.md` | `2b2d9fa6a2fbb6a659220d4caaf12c5db04a9cd2c43dbf4bfe35c688e769ac75` |
| `experiments/shared_state_fit_transfer.py` | `76a5abb06e6d667394eb4eaf2ec7ce2815210cddf9a5bd788144bf9c133bbe26` |
| `runs/shared-state-fit-transfer-local/attempt-001/summary.json` | `21ae5907b36140cf383e3acc31699eda6dd217122405025d828f2f89895ebb9d` |
| `runs/shared-state-fit-transfer-local/attempt-001/analysis.json` | `8082daf93bce0315dba3b3ad8fd87f710c293f201e1b0e2327e8648562c48f88` |

## Next

The selected next intervention is [shared causal entity retrieval with native fusion](CAUSAL_ENTITY_RETRIEVAL_DESIGN.md). It will let learned entity queries retrieve from the visible prefix and let the common action/reply path consult those vectors. The comparison will retain matched lessons and supervision across all three families. No additional small-buffer fit run is selected, and the slower learning rate is not adopted as a general winner. This is a design for the next local comparison; no retrieval candidate has been implemented or trained yet. The broader goal remains active.
