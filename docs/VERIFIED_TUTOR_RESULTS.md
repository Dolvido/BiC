# Verified tutor curriculum results

The tutor loop is operational, but this experiment did **not demonstrate a tutor
advantage**. After common withdrawal, tutor-guided teaching finished at 158/360
joint development pairs versus 154/360 for procedural teaching: +1.11 percentage
points, below the required five-point gain. Fit and retention favored the
procedural arm, and the tutor arm failed withdrawal nonregression. Neither
checkpoint is promoted.

## What was compared

The [protocol](VERIFIED_TUTOR_PROTOCOL.md) compared two exact full-model/AdamW
copies of the slow 1944-update parent from the
[continuation study](SHARED_STATE_CONTINUATION_RESULTS.md). That parent's joint
development score reached 143/360 while the fast endpoint remained at zero, but
unknown-answer regressions failed its acceptance screen. Selecting it was
explicitly outcome-informed development selection, not promotion.

Both branches retained the same SharedStateStudent, native objective plus 0.3
state loss, learning rate, runtime, slot order, seed allocation and coverage.
Each received 108 teaching updates and 108 common replay withdrawal updates:
432 new updates and 41,472 training episode exposures altogether, with 32 episodes
per family per update. Every chapter covered all 18 depth/length cells across
color, count and switch. This was a whole-curriculum contrast, not extra practice
for selected weak cells.

The local external LLM supplied only menu-constrained chapter choices; canonical
oracles independently generated and verified the examples and supervision.
Teacher prose, labels and auxiliary state targets never entered policy inference.
Evaluation used native actions and freely generated replies on previously
observed development 720, original training-fit 108 and retention 720 episode banks.
These reused banks are not independent confirmation data.

## Actual teacher choice and compiled contrast

The first author slot returned HTTP 400 after the service loaded and warmed the
model. Its error body was lost, so the exact cause is unconfirmed. The preserved
receipt records one attempted chat, no response completion,2.500 wall seconds,
and unknown generation/token work; it was not zero teacher compute.

A separately declared [v 2 transport request](VERIFIED_TUTOR_RECOVERY.md), using a
portable array schema and the same exact positional validation, completed and
was accepted. This was an explicit recovery, not an automatic retry. The pinned
local Ministral-3:3 b digest matched before/after; `keep_alive=0` was sent. It took
3.109 wall seconds and reported 1,058 prompt/160 generated tokens. Across both
slots: two attempted chats, one completion,5.609 measured request seconds, with
first-attempt generation work still unknown.

Using one-based chapter numbers, chapters 1-2 remained fixed replay. The accepted
fresh choices were chapter 3 balanced/rename, chapter 4 copy_emphasis/revalue,
chapter 5 change_emphasis/rename, and chapter 6 balanced/independent. Rename retains
parent values; revalue retains aliases. The procedural condition used
balanced/independent throughout. Pool names indicate relative copy counts in
existing admitted procedures; they introduce no new grammar, operator or causal
law, and some pools overlap.

The completed compilation changed exactly 54/108 teaching bundles in actual
learner-visible contents; the remaining 54 matched. Withdrawal images were
identical across arms. Each compiled phase contained 10,368 episodes/103,680 turns.
Equal episode counts and upper limits did not mean identical token exposures:

| Compiled condition | Observation bytes | Observation tokens | Reply bytes | Reply target tokens |
|---|---:|---:|---:|---:|
| procedural | 2,934,329 | 3,141,689 | 1,296,064 | 1,399,744 |
| tutor | 2,934,182 | 3,141,542 | 1,298,515 | 1,402,195 |
| withdrawal | 2,932,694 | 3,140,054 | 1,292,558 | 1,396,238 |

Withdrawal describes one cache consumed once by each arm. Token counts include
existing boundary tokens, not padding; teacher tokens are separate.

## Native outcomes

Entries are action-pair / reply-pair / joint-pair correctness. Joint success
requires both opposite-answer members correct in both output channels.
Denominators:360 pairs for development/retention and 54 for training-fit.
Update 0 is the restored parent;108 ends teaching;216 ends withdrawal.

| Relative updates | Bank | Procedural A / R / joint | Tutor A / R / joint |
|---:|---|---:|---:|
| 0 | dev | 151 / 168 / 143 | 151 / 168 / 143 |
| 0 | train_fit | 22 / 19 / 18 | 22 / 19 / 18 |
| 0 | retention | 152 / 166 / 140 | 152 / 166 / 140 |
| 108 | dev | 141 / 151 / 129 | 156 / 146 / 130 |
| 108 | train_fit | 24 / 23 / 20 | 24 / 23 / 22 |
| 108 | retention | 133 / 140 / 120 | 142 / 154 / 131 |
| 216 | dev | 167 / 176 / 154 | 172 / 168 / 158 |
| 216 | train_fit | 26 / 26 / 24 | 26 / 21 / 21 |
| 216 | retention | 157 / 169 / 152 | 159 / 156 / 140 |

An independent stdlib recount authenticated 54 raw groups,18 endpoints,9,288
episodes and 43,512 queries, reproducing every metric/screen; baseline raw records
were identical. It took 1.360 wall/1.343750 CPU seconds with zero neural work.

Both branches lost development joint pairs during teaching, then recovered during
the identical withdrawal curriculum. Final family joint development pairs were
procedural/tutor: color 70/70, count 18/15, switch 66/73, each out of 120. The final
comparative screen failed the minimum overall gain, strict improvement in every
family, and training-fit improvement requirements. All final comparative
known/unknown five-point margins and overall retention margin passed, but those
do not override failed acquisition criteria. At 108 the comparative screen also
failed; tutor count known replies were 302/528 versus 329/528 (-5.11 points).

Both baseline-to-final acquisition screens failed: procedural gained 11/360 joint
pairs (+3.06 points), tutor 15/360 (+4.17), below five points; color joint success
fell from 73/120 to 70/120 in both arms.

Final development slices below show procedural -> tutor with exact denominators.
Nonanchor known excludes the designated paired anchor.

| Family | Known A | Known R | Unknown A | Unknown R | Nonanchor known A | Nonanchor known R |
|---|---:|---:|---:|---:|---:|---:|
| color | 380 -> 389 / 526 | 370 -> 364 / 526 | 435 -> 449 / 594 | 446 -> 473 / 594 | 196 -> 199 / 286 | 183 -> 177 / 286 |
| count | 334 -> 329 / 528 | 332 -> 312 / 528 | 505 -> 510 / 594 | 510 -> 528 / 594 | 195 -> 196 / 288 | 193 -> 185 / 288 |
| switch | 417 -> 429 / 532 | 414 -> 404 / 532 | 452 -> 447 / 586 | 464 -> 476 / 586 | 236 -> 241 / 292 | 229 -> 216 / 292 |

Overall known actions were 1131 -> 1147/1586, known replies 1116 -> 1080/1586; unknown
actions 1392 -> 1406/1774 and unknown replies 1420 -> 1477/1774. The tutor favored unknown
answers and some actions while generated known replies remained below control;
its final known-reply score merely matched the parent. Unsupported ASK on known
queries was 142 -> 149/1586 for actions and 160 -> 215/1586 for replies. All 3360 query
replies were parseable; action/reply agreement was 3099 -> 3050/3360. Neither
parseability nor agreement establishes semantic correctness.

## Restart and withdrawal

Four exact full-state restorations and four saved snapshots completed. The
worker made zero teacher calls and ended both branches at global cursor 2160,
with no unknown optimizer outcome or automatic retry. All 432 updates,1,296 family
forwards/backwards and state readouts, and 9,288 evaluation episode passes completed.

The procedural arm passed all 36 withdrawal known/unknown five-point checks.
The tutor arm failed three:

| Tutor slice, teaching -> withdrawal | Counts | Change |
|---|---:|---:|
| Retention/color known reply |380 -> 351 /536|-5.41 pp|
| Training-fit/switch known action |70 -> 64 /84|-7.14 pp|
| Training-fit/switch known reply |67 -> 62 /84|-5.95 pp|

The benefit condition is false: the example contrast is nonzero, but the final
comparative and tutor withdrawal screens fail. This demonstrates the executed
connection from external recipe through verified examples, learning, unaided
evaluation and restart. It does not demonstrate beneficial teaching or learned
self-direction; withdrawal was imposed by the caller.

## Preserved preparation failures and cost

The first compilation completed 108 procedural and 84 tutor bundles before rejecting
a protected transcript. Its 102.031 wall seconds,101.359375 CPU seconds,
68,532 canonical calls and 137,064 returned rows remain charged preparation work;
no partial output was silently adopted. A focused diagnosis identified both
switch members at zero-based slot 84/pair 2/depth 4/eight turns, under rename.
Different seeds/IDs can yield the same English within a finite alias vocabulary.

The [additive correction](VERIFIED_TUTOR_COMPILATION_RECOVERY.md) preserves valid
candidate-zero bytes and allows at most 64 deterministic candidates per fresh
family pair. Both members pass atomically; exclusions and selected transformations
remain fixed. Rename varies names only, revalue values only; independent retries
vary names while preserving candidate-zero values. Replay is unchanged, and
exhaustion fails. Repair can change one family's aliases; identical aliases across
families are not promised.

The completed production preparation rejected four tutor candidates across three
switch pairs, all protected: slot 84 offsets 2 (attempts 0,1),7 (attempt 0),8 (attempt 0).
Accepted attempts were 2,1,1 respectively. Procedural/withdrawal had no rejections.
The narrow fixture protected only the two original collision texts and accepted
attempt 1; full preparation correctly rejected that candidate against its complete
protected set. No exclusion was relaxed.

The diagnosis cost 2.312 wall/2.328125 CPU seconds,35 canonical calls/70 rows and
one protected archive read. Six focused v 2 tests passed first invocation in
2.328 wall/2.093750 CPU seconds:1,149 real canonical calls/2,298 rows plus 67 mock
calls, including v 1 image/evidence parity, actual collision repair, atomicity,
finite exhaustion and replay refusal. All 88 formal/source pins stayed unchanged.
The v 2 author passed nine mocked tests in 0.313 wall/0.250 CPU seconds. These
engineering checks made no learner, GPU or live teacher calls.

| Separately recorded live phase | Wall seconds | CPU seconds |
|---|---:|---:|
| Inventory preparation |41.735|41.328125|
| Failed author request |2.500|0.046875|
| Accepted author request |3.109|0.046875|
| Failed compilation |102.031|101.359375|
| Completed compilation |172.203|171.171875|
| Learner worker |175.422|163.468750|
| Sum of recorded intervals |497.000|477.421875|

This sum excludes historical parent/data work, engineering checks, diagnosis,
analysis and gaps between operations; it is not one measured continuous duration.
Both production compilations together used 184,024 canonical calls/368,048 returned
rows. Nested regeneration is not unique learner exposure.

Inside the worker, training took 72.205 seconds, packing 45.915, archive reads 28.203,
evaluation 16.953, bank validation 3.500, targets 1.385, restoration 1.015 and
snapshotting 0.312. These are portions of 175.422 seconds, not additional costs.
Peak allocated/reserved GPU memory was 1,622,219,264/4,053,794,816 bytes. The worker
had 326 outer archive reads (324 lessons, one bank and one parent), plus four
restoration decodes: five checkpoint deserializations total. Bank admission
included 774 canonical parent regenerations and 1,548 typed/1,548 English oracle
calls. Packing plus archive reads exceeded training time, an identified CPU
efficiency opportunity rather than evidence to change the learning objective.

## Scope and evidence

This was one outcome-selected parent, one accepted recipe and reused banks within
a finite English grammar. It establishes neither general English comprehension,
new causal concepts nor explanation learning. The modest uneven development gain,
worse fit/retention and withdrawal failures do not support adopting this teacher
recipe or either checkpoint. No score-selected intermediate endpoint or extra
teacher request was used to obtain a favorable result.

- Parent checkpoint SHA256: `3036f6b6bad4288f1f8d462cbd50c96914ff2aa7710107bf49f5f6fee4c726b9`; weights `f2c20245d85c4ac1483117defafab1530e447fb9a08389b51dca34dd5d7cfb41`.
- Unknown first author (archive reference: `../runs/verified-tutor-author-local/attempt-001/uncommitted.json`): `6284cc0d41cf1dbab7c8eae77fb8e90e286ee645f0bdc484a3bb1896e3a3d150`; accepted result (archive reference: `../runs/verified-tutor-author-local/attempt-002/result.json`): `3c5f601f8fe44a85763b8cbbd712a70e28123743a86339bb6f322cc64bbdebf0`.
- Inventory manifest (archive reference: `../runs/verified-tutor-inventory-local/attempt-001/manifest.json`): `6777954dd97141a09fd2a2fa5e664dbdbbf092efe3d3e587fbf91b109be6b088`; completed compilation manifest (archive reference: `../runs/verified-tutor-data-local/attempt-002/manifest.json`): `a35f6b3ab0c2a104c9038e28e4cd58178c59e14343e341565415f24054170419`.
- Failed compilation receipt (archive reference: `../runs/verified-tutor-data-local/attempt-001/preparation.json`): `35b26ab660acca727dad4806ed377d8d51b1670da1c5a093e40e78978a2f386a`; completed receipt (archive reference: `../runs/verified-tutor-data-local/attempt-002/preparation.json`): `41e498b71730bace8a475df819dff550a6b734193f9e94854ce58855d0ccf6fe`.
- Collision diagnosis (archive reference: `../runs/verified-tutor-collision-diagnosis-local/attempt-001/report.json`): `3d488133c159e0cb9ba680829917c6d5cbf454d3e5f64784a118d9d4dd788645`; compiler v2 proof (archive reference: `../runs/verified-tutor-curriculum-v2-validation-local/attempt-001/report.json`): `d4cc0f62e90284f0f1f103a24f07d9a8b1df0b28739ee6679048d2c7db349d81`.
- Launch (archive reference: `../runs/verified-tutor-run-local/attempt-001/launch.json`): `e5879d9e433240cd82ffbf131941a452fb6897ad75c3ec925f8dce2c10840d9b`; completed summary (archive reference: `../runs/verified-tutor-run-local/attempt-001/execution/summary.json`): `23e6286ee25b8f52c974e18d692670619cf9c200c7e3d53117099f854f7695ea`.
- Pure analysis (archive reference: `../runs/verified-tutor-run-local/attempt-001/analysis.json`): `bc6973b4af20c487d6c12558b3b494390a8d6d951d54e0b37754a97d2195f755`; independent recount (archive reference: `../runs/verified-tutor-analysis-local/attempt-001/independent-analysis.json`): `7bb8e4376713ff06f8a20b6cca88b0002ac45d9d3b3484979e9bc27c427b08ed`.
