# Shared-state continuation: broad acquisition with important regressions

The slower learner acquired substantially better native behavior after two more
passes over the broad training cache: fresh-development joint counterfactual
success rose **0 → 58 → 143 of 360 pairs** at lifetime updates 648/1296/1944.
The faster learner ended at 0/360. Nevertheless, **both within-arm acquisition
screens and the final between-arm screen failed**. The slower learner lost too
much unknown-answer accuracy in color and switch; it is an exploratory research
parent, not an accepted checkpoint or evidence of general comprehension.

The completed execution (archive reference: `../runs/shared-state-continuation-pilot-local/attempt-001/execution/summary.json`),
native recount (archive reference: `../runs/shared-state-continuation-pilot-local/attempt-001/analysis.json`)
and independent recount (archive reference: `../runs/shared-state-continuation-pilot-local/attempt-001/independent-analysis.json`)
cover all 24 bank endpoints and 13,608 raw evaluation episode records. The
[prospective protocol](SHARED_STATE_CONTINUATION_PROTOCOL.md) remains unchanged.
There was no score-dependent stopping, extension, earlier-checkpoint selection
or teacher call.

## What continued

Both predetermined final 648-update checkpoints from the
[preceding rate study](SHARED_STATE_RATE_RESULTS.md) resumed with complete weights,
AdamW moments/groups, original learning rates, recipe, evidence and cumulative
accounting. Each SharedStateStudent retained 2,264,725 parameters, width 192,
four layers/heads, feedforward 768, original sequence objective plus 0.3 causal
state loss, clip 1 and strict local FP32. Fast remained 0.003 and slow 0.0003;
there was no optimizer reset or new model initialization treatment.

Each arm received 1,296 new updates and 124,416 repeated episode exposures,
ending at lifetime 1,944 updates and 186,624 exposures. The cache still contains
only 62,208 distinct original training rows. Global consumed IDs 648–1943 map to
two complete passes over the same 648 archives; nested canonical lesson identities
remain intact. First-arm execution alternated, and state targets were computed
once for each shared bundle. Those labels never entered policy inputs.

The new development bank has 720 episodes, seed 852702001, excluded from 1,703,160
recorded-study transcripts. Previous development 720, actual training-fit 108 and
retention 720 were explicitly reused. Restored 648 metrics on all three reused
banks exactly matched their original endpoint before any new update. The fresh
bank tests unseen transcripts in the same finite grammar and vocabulary; it does
not establish broader English, novel semantics or independence.

## All native trajectories

A/R/joint are correct paired native actions, exact freely generated replies, and
both together on the two opposite-answer anchor questions. State-head predictions
never supply these answers.

| Lifetime updates | Bank | Pair denominator | Fast A/R/joint | Slow A/R/joint |
| ---: | --- | ---: | ---: | ---: |
| 648 | Fresh development | 360 | 12 / 15 / 2 | 12 / 0 / 0 |
| 648 | Previous development | 360 | 13 / 9 / 1 | 14 / 0 / 0 |
| 648 | Training-fit | 54 | 2 / 2 / 0 | 2 / 0 / 0 |
| 648 | Retention | 360 | 7 / 12 / 0 | 17 / 0 / 0 |
| 1296 | Fresh development | 360 | 18 / 0 / 0 | 93 / 78 / 58 |
| 1296 | Previous development | 360 | 19 / 0 / 0 | 98 / 88 / 62 |
| 1296 | Training-fit | 54 | 0 / 0 / 0 | 9 / 10 / 6 |
| 1296 | Retention | 360 | 21 / 1 / 1 | 103 / 86 / 70 |
| 1944 | Fresh development | 360 | 14 / 2 / 0 | 151 / 168 / 143 |
| 1944 | Previous development | 360 | 18 / 4 / 1 | 162 / 177 / 148 |
| 1944 | Training-fit | 54 | 1 / 0 / 0 | 22 / 19 / 18 |
| 1944 | Retention | 360 | 23 / 1 / 0 | 152 / 166 / 140 |

Slow final joint correctness was 73/120 color,13/120 count and 57/120 switch on
fresh development; fast scored 0/120 in each. Overall slow known-query actions
rose 637 → 856 → 1099/1586 and known replies 576 → 732 → 1080/1586.
Nonanchor known actions rose 295 → 613/866 and replies 225 → 587/866 from 648
to 1944. Acquisition therefore extended beyond the selected final anchor.

| Fresh-development final slice | Fast action | Slow action | Fast reply | Slow reply |
| --- | ---: | ---: | ---: | ---: |
| Color known | 181/526 | 385/526 | 182/526 | 364/526 |
| Color unknown | 408/594 | 346/594 | 438/594 | 360/594 |
| Count known | 169/528 | 337/528 | 183/528 | 336/528 |
| Count unknown | 449/594 | 473/594 | 474/594 | 470/594 |
| Switch known | 171/532 | 377/532 | 152/532 | 380/532 |
| Switch unknown | 438/586 | 358/586 | 498/586 | 388/586 |

The final slow-versus-fast screen passed its joint-gain, every-family joint,
training-fit and retention conditions. It failed color unknown actions/replies
(−10.44/−13.13 percentage points) and switch unknown actions/replies
(−13.65/−18.77 points). Those losses exceed the fixed 5-point margin.

Against its own restored baseline, slow passed all acquisition improvement
conditions but failed color unknown actions/replies (−9.43/−16.33 points) and
switch unknown replies (−17.41 points). Switch unknown actions declined 3.92
points and remained inside the margin. Fast failed the acquisition improvements,
count known-action margin and switch known-reply margin. The separate reused
previous-development nonregression check also failed for slow color unknown
actions/replies and switch unknown replies. These are distinct failed screens,
not interchangeable reasons to accept or reject one aggregate score.

All 3360 fresh-development query replies parsed in each final model. Final
action/reply agreement was 2110/3360 fast versus 3024/3360 slow (90%); agreement
can include jointly wrong answers. Slow unsupported ASK on known questions fell
from 370 to 102/1586 for actions and 528 to 144/1586 for replies. Its overall unknown
actions fell 1234→1177/1774 and replies 1417→1218/1774. The result combines better
factual answering with worse abstention on some subjects, rather than uniform
accuracy or reliable calibration.

## Final training-state diagnostic

The raw diagnostic (archive reference: `../runs/shared-state-continuation-diagnostic-local/attempt-001/diagnostic.json`),
receipt (archive reference: `../runs/shared-state-continuation-diagnostic-local/attempt-001/receipt.json`)
and independent verification (archive reference: `../runs/shared-state-continuation-diagnostic-local/attempt-001/verification.json`)
inspect only both final 1944 checkpoints on 108 actual training-fit rows. This
is auxiliary readout on training examples, not held-out state transfer or policy
inference. Models remained unchanged and there was no learning or free decoding.

| State readout | Fast known exact | Slow known exact | Fast conditional value | Slow conditional value |
| --- | ---: | ---: | ---: | ---: |
| Color | 13/880 | 243/880 | 237/880 (26.93%) | 492/880 (55.91%) |
| Count | 0/880 | 41/880 | 57/880 (6.48%) | 241/880 (27.39%) |
| Switch | 26/880 | 382/880 | 433/880 (49.20%) | 580/880 (65.91%) |
| Overall | 39/2640 | 666/2640 | 727/2640 (27.54%) | 1313/2640 (49.73%) |

Conditional value uses all 106 nonzero classes, with no family mask. Relative to
its 648 endpoint, slow conditional counts improved 418→492 color,65→241 count
and 538→580 switch; full known exact improved 98→666/2640 overall. Count mean
true-value rank improved 10.7625→6.4693, and full known count decoding became
nonzero. These are meaningful partial gains, but slow still decoded only 41/880
known count states exactly. The earlier all-unknown full argmax did not imply
absence of conditional value information.

At 1944, full unknown-state accuracy was 9976/10320 fast and 8757/10320 slow.
Under the distinct strict `P(known)>0.5` rule, sensitivity was 1516/2640 versus
1975/2640 and specificity 5267/10320 versus 6644/10320; Brier scores were 0.239619
and 0.197954. Balanced state loss was 1.512656 versus 1.099204. Native ASK behavior,
the 107-class argmax and binary knownness are different readouts and must not be
collapsed into one capability claim. This comparison also does not isolate the
causal contribution of auxiliary supervision, since both arms retained it.

## Exact continuation, work and cost

CPU and production-GPU restart proofs both passed complete next-step weights,
AdamW, losses, evidence and work equality. CPU proof used 6 model constructions,
3 updates and 9 forwards/backwards (18 episode exposures), including rejected
restore cases. GPU proof used 2 models,3 updates and 9 forwards/backwards (288
exposures). The independent replay proof verified identical packed inputs at
source/consumed IDs 0/648/1296 and rejected malformed mappings without learning.
All three proofs and the main run completed their first invocation.

| New operation | Wall seconds | CPU seconds |
| --- | ---: | ---: |
| CPU restart proof | 3.859 | 3.437500 |
| GPU restart proof | 4.344 | 3.703125 |
| Replay proof | 3.031 | 2.750000 |
| Source-only runner validation | 0.016 | 0.000000 |
| Fresh evaluation preparation | 65.922 | 64.546875 |
| Source/data/checkpoint freeze | 2.609 | 2.375000 |
| Entire continuation execution | **1225.703** | **1161.343750** |
| Final state diagnostic | 5.031 | 4.515625 |

Preparation retained 648 original training records, made 222 authenticated archive
reads, and counted 39,767 generator calls/79,534 returned canonical rows. These
include reconstructed validation parents, not new training lessons. It published
only the fresh evaluation overlay, with zero neural work.

Main physical work was 2592 new retained/synchronized updates,248,832 training
episode exposures and 7776 family forwards/backwards, two restored models, four
saved complete continuation checkpoints and 1296 shared target computations.
Lifetime totals per arm include the inherited 648 updates:1944 updates,186,624
exposures and 5832 forwards/backwards. Restored accounting is not counted as new
learning. Native evaluation covered 24 bank endpoints and 13,608 episode records.
The final state diagnostic added 18 forwards,18 auxiliary calls,216 episode
exposures and 25,920 state predictions; it constructed 2 models and no optimizer.

Main enclosing operation times were 285.212 seconds for replay loads,140.205 for
replay copying/remapping,294.707 for preparation/packing,452.008 for learner
steps,25.500 for evaluation and 5.553 for shared state-target preparation. These
are subdivisions of 1225.703 seconds, not additional costs. Replay read 987,565,264
archive bytes in 1296 loads. Load/remap plus packing consumed about 58.75% of total
wall time; learner operations consumed about 36.88%. The next efficiency work
should target CPU cache handling before attributing this elapsed time to GPU
capacity.

Completed-event per-arm training including packing was 371.603 seconds fast and
370.739 slow; evaluation was 12.545 and 12.861 seconds. Event timing excludes some
event-writing overhead, so these sums differ slightly from enclosing operation
totals. Peak main GPU allocation was 1,613,584,384 bytes and reservation 4,745,854,976
bytes. Diagnostic peak allocation was 184,517,120 bytes. Historical cache creation
and the original 648-update study are not charged as new work here.

## Interpretation and next controlled work

The observed gains support a shared acquisition/optimization explanation for
part of the prior failure: the unchanged slower architecture acquired native
joint behavior across all three families with further broad practice. The fast
arm did not. This does not establish a universally optimal rate, guarantee more
training will continue helping, or prove a particular internal state mechanism.
There is one initialization and a repeated finite training inventory.

Slow 1944 is a defensible **exploratory tutor-comparison parent** because it now
has substantial native behavior to improve and preserve. Choose it explicitly
as an outcome-informed research decision, then clone the same full weights and
AdamW state into both tutor and procedural branches. Do not describe this as
promotion or screen passage. Keep matched whole-curriculum coverage/replay,
unaided inference, every family's known/unknown and nonanchor checks, retention,
and a common teacher-withdrawal phase. A count-only or unknown-only repair would
not answer the shared teaching question.

The [verified tutor design](VERIFIED_TUTOR_CURRICULUM_DESIGN.md) and
[chapter compiler](VERIFIED_TUTOR_CURRICULUM_COMPILER.md) provide the next
engineering path: consume verified tutor-chosen examples and compare them with
procedural choices under the same fixed slots and budgets. An accepted teacher
response or choice identical to the default is not evidence of teacher benefit.
Any future cache optimization should preserve immutable authentication and prove
that reused packed tensors, supervision and per-consumption identities remain
exactly equivalent. No architecture change or further automatic run follows
from this completed record.

## Artifact pins

| Artifact | SHA-256 |
| --- | --- |
| Launch | `aea983ab379c26d054a2b80f736bdab46144ff374c2abbb0545af17f09bd3e81` |
| Execution summary | `c15821dbd629a39286961f360c2310ab72697e2ac7f50351174fcced41233773` |
| Native analysis | `7e9ed7725d3411364e36d623a17b3050c37626d8de350f7149949e088971a83d` |
| Independent native analysis | `62701a8fe725e4fd61781be4c4e64484e67f3b49a31f75ef0346f58186bae83a` |
| Final state diagnostic | `fb8d40c210b8d909a8d3e292e1a80ea0ca75d98e1580b05bafc890731408e240` |
| Diagnostic receipt | `eb5c9ebb904914678fbe312bc274bcd0f05a9a23b195949920735f548434a8be` |
| Independent state verification | `38b9ae77e43b391796df4f13c029ee314a5f1d4b025d477e9b9196098d43f36c` |
| Fresh data manifest | `34ac898adce7446ce9edcd67e821fa42c2d5217883ccb3f51f0272225cab3795` |
| Fresh data preparation | `f4a96e7e621e73d7421d8c391ca40d33178a2fa57e3c8ec056fc8580d2b60e4c` |
