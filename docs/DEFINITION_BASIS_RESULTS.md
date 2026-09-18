# Shared operations and two-word instruction learning

The fixed 648-update experiment completed and passed independent raw-record
verification. It learned most one-word revisions, but did not establish reliable
selection between two defined words or transfer to unfamiliar combinations.
Seven old-skill paired checks exceeded the declared five-point loss allowance.
The update-9,760 candidate remains a separate research branch; the retained
campaign learner is still update 9,112.

The model, full optimizer, rate and objective continued unchanged. Training used
216 basis updates, 108 prior-definition updates and 324 broad-practice updates:
62,208 episode exposures, equal across color, count and switch. The new compact
basis catalogue contains 10,368 rows (10,364 distinct transcripts); its two
passes are explicitly repeated practice. No teacher participated in this run.

The primary metric requires every native action and freely generated reply to
be correct for all questions in both counterfactual episodes. Each panel has
96 pairs. These are the fixed reported endpoints, not a selected best checkpoint:

| Definition panel | Parent | Step 216 | Step 432 | Step 648 |
|---|---:|---:|---:|---:|
| Two-word basis binding | 0 | 0 | 9 | 10 |
| Basis revision | 0 | 10 | 62 | 78 |
| Unfamiliar two-instruction composition | 0 | 0 | 0 | 0 |
| Unfamiliar three-instruction sequence | 0 | 0 | 0 | 0 |
| Earlier one-word binding | 0 | 83 | 94 | 92 |
| Earlier one-word revision | 0 | 90 | 90 | 91 |
| Earlier composition panel | 0 | 0 | 0 | 0 |

Final basis binding was 5/32 color, 4/32 count and 1/32 switch. Revision was
25/32, 21/32 and 32/32 respectively. Composition and sequence remained 0/32 in
every family. The declared combined capability milestone failed. Prior
definition panels improved from this parent's zero complete-pair baseline;
they do not prove preservation of the separate earlier definition probe's
weights or abilities.

| Original diagnostic bank, joint focal pairs | Parent | Final |
|---|---:|---:|
| Development | 271/324 | 257/324 |
| Retention | 299/360 | 290/360 |
| Training fit | 49/54 | 48/54 |
| Unfamiliar compositions | 153/216 | 136/216 |
| Changed question positions | 146/216 | 141/216 |

Seven of the fifteen bank/family joint-pair retention checks failed. All sixty
known/unknown action/reply checks passed, illustrating why aggregate query
accuracy alone is insufficient. The largest joint decline was count transfer,
36/72 to 25/72 (15.28 percentage points). The nine prior-definition complete-pair
retention checks passed against their zero baseline.

Physical work was 648 optimizer updates, 1,944 family forwards and backwards,
four exact restorations and three snapshots, with no unknown optimizer outcome.
The run took 296.172 wall seconds and 282 CPU seconds. Independent analysis
verified 48 bank endpoints, 88 raw score files and 14,736 episode records without
loading a checkpoint or making a model call. It took 2.953 wall seconds and
2.921875 CPU seconds.

A separate pure analysis joined all four binding endpoints to the admitted
examples. At the final endpoint, 789/960 known queries had both action and reply
correct (82.19%), but only 10/96 pairs were completely correct. Accuracy for the
contrasted word and the other word was similar: 82.81% and 81.77%. Definition
position did not consistently favor the latest definition; errors occurred
across all three contrasts and families. Of 171 incorrect joint queries, 136
had both action and reply wrong. These are shared decision errors, without
evidence yet for a specific architectural cause or simple second-word neglect.

The diagnosis used 768 episode records and 4,608 query records, without new
inference. Its report is `runs/definition-basis-diagnostic-local/attempt-001/analysis.json`,
SHA256 `9d864174b666447bb2a4b30034cec1925799d183a01b1f768ba3fbc3e7380ea2`.
That raw-score analysis alone could not distinguish underfitting from transfer
to new nonce words. A subsequent fixed-checkpoint diagnostic scored 288 exact
training episodes per panel, taken from three previously trained images each.
Complete-pair correctness was 19/144 for binding (13.19%) and 131/144 for revision
(90.97%). Binding family counts were 10/48 color, 6/48 count and 3/48 switch;
revision counts were 45/48, 38/48 and 48/48. Poor binding therefore also occurs
on these measured training rows. Novel vocabulary alone cannot explain it;
the subset does not establish a particular architectural or optimization cause.

The diagnostic performed one exact restoration, eighteen inference forwards,
576 episode evaluations and 3,456 query evaluations, with no training. It took
3.797 wall seconds and 3.46875 CPU seconds. Both score files completed, but the
final receipt retained a failure because its Windows path separators violated
the pin verifier's canonical naming rule. A separate pure verification checked
all unchanged bytes, source inventories, raw answers and metric arithmetic;
inference was not repeated. That verification took 0.297 wall seconds and
0.296875 CPU seconds. Its report is
`runs/definition-basis-fit-diagnostic-analysis-local/attempt-001/verification.json`,
SHA256 `9af78a7cf03152db55a94f03dd067cc78d94bfd18bcb59ff69b5849125c3e28a`.

The next bounded test compares
the tutor's alternating binding/revision progression against a fixed order,
preserving the same complete lesson/target multiset and full starting state.
Its teaching advantage remains an experimental question.

Evidence:

- Run: `runs/definition-basis-learning-local/attempt-001`.
- Launch SHA256: `4fd471d4b5efc9357a1eccd6ac86c1867c4f863d6ddd86a443bca22c04d82415`.
- Summary SHA256: `5f247097e43e8a4f31d209a9a0601731111da3b69de67e13941c5ebbded071cb`.
- Independent report: `runs/definition-basis-analysis-local/attempt-001/independent-analysis.json`.
- Independent SHA256: `80044620c009dcb4714f8cbc0e90dc438c4107e98edc88008c3e2e96c75bbe49`.
- Data manifest SHA256: `3ae5d0d93685675c247c68aaa7a87254abc9ad32e59782b1e34f353c6b5e285b`.

The first CPU packing integration probe admitted all four banks and one lesson,
then stopped because its harness reused an owner with an outstanding token.
That failure is preserved; a separate probe admitted only the remaining lesson
with a fresh owner. No model work occurred in either check. The complete GPU
run used normal single-use token consumption and completed without that error.
