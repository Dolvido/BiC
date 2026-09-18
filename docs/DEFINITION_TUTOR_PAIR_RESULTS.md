# External tutor chapter order and independent practice

The local teacher's accepted curriculum order was executed on the native learner
and compared with a fixed order. Both branches completed the same 108 teaching
updates and 108 common tutor-free practice updates, starting from identical
update-9,760 weights and full AdamW state. Independent raw-record verification
passed. Neither branch met the transfer and retention requirements; the retained
campaign learner remains update 9,112.

The local `ministral-3:3b` teacher alternated each binding chapter with its
matching revision chapter. The procedural order placed all binding chapters
before all revision chapters. Exactly 72 of 108 teaching positions differed.
Both received the same complete lesson, input and target multiset, equally
covering color, count and switch. Correctness came from the verified curriculum;
the teacher received aggregate development counts and the permitted chapter menu.

Each metric below counts completely correct action and freely generated reply
sequences for both members of a counterfactual pair. Every panel has 96 pairs.

| Panel | Shared parent | Procedural teaching | Procedural final | Tutor teaching | Tutor final |
|---|---:|---:|---:|---:|---:|
| Two-word binding | 10 | 6 | 20 | 7 | 21 |
| Rule revision | 78 | 73 | 82 | 57 | 83 |
| Unfamiliar composition | 0 | 0 | 0 | 0 | 0 |
| Unfamiliar sequence | 0 | 0 | 0 | 0 | 0 |
| Earlier one-word binding | 92 | 35 | 93 | 21 | 94 |
| Earlier one-word revision | 91 | 36 | 92 | 27 | 90 |
| Earlier composition | 0 | 0 | 0 | 0 | 0 |

The primary composition/sequence difference was zero percentage points. The
tutor's one additional binding and revision pair are intermediate results, not
evidence of improved transfer. Neither arm passed the separate capability
milestone. Both lost older definition ability during isolated teaching and
recovered much of it during common mixed practice, supporting continued broad
rehearsal throughout future curricula.

Of 96 retention checks against the current research parent, the procedural arm
failed four and the tutor arm failed five. Against the fixed retained reference,
they failed six and seven respectively. The procedural arm also failed one of
twelve teaching-to-withdrawal checks: count binding fell from 5/32 to 3/32.
The tutor arm passed those twelve checks. Both remain ineligible because shared
composition did not improve and overall retention failed. No candidate was
automatically adopted.

Physical work was 432 optimizer updates, 41,472 episode exposures, 1,296 family
forwards and backwards, six full-state restorations and four snapshots. There
were no unknown optimizer outcomes. Both trajectories together took 225.625
wall seconds and 212.53125 CPU seconds. Peak allocated GPU memory was
1,557,689,856 bytes; peak reserved memory was 3,454,009,344 bytes. The worker made
no teacher calls. The prior author request was one completed local call, costing
1,394 prompt and 66 output tokens; it is not counted again as worker activity.

Independent verification checked 72 bank endpoints, 132 raw score files, 22,104
episode records, 110,940 query records and 432 step reports. It also checked
actual lesson order, target identity, restoration evidence and physical work.
Verification took 4.640 wall seconds and 4.625 CPU seconds, without model calls,
generation, teacher calls or checkpoint deserialization.

The experiment establishes a working external-author-to-native-learning path,
including full-state restart and independent practice. It does not demonstrate
a tutoring advantage, broad English comprehension or learned self-direction.
The development banks have been used repeatedly; they are not fresh audits.

The next curriculum must improve operation-order coverage while preserving
held-out combinations. New basis lessons contain atomic definitions, but older
definition rehearsal already includes `copy; advance destination`, using the
same semicolon syntax. Therefore this learner has seen multi-clause syntax;
zero transfer cannot be attributed simply to an unseen punctuation or combinator.
Its compound training coverage is narrow, and it has not practiced the target
source-changing compositions. The original banks and result remain unchanged.
A new version should teach complementary operation orders with balanced
prior-skill practice and reserve existing target combinations for transfer.
Poor two-word fit on actual training examples remains a separate acquisition
limitation, documented in `DEFINITION_BASIS_RESULTS.md`.

Evidence:

- Protocol: `docs/DEFINITION_TUTOR_PAIR_PROTOCOL.md`.
- Run: `runs/definition-tutor-pair-local/attempt-001`.
- Launch SHA256: `44c6bfa9471a6e57ea8195fbe0c0c9f1c5657a6afa5875e8d2e4b02621fdaca7`.
- Summary SHA256: `20d86778b313a444d38a29b10b17c8e00fe9669fe1ffb3a66c68f323acaf5413`.
- Paired-data manifest SHA256: `04d2ede4fd25f8982dff607a5fa32c237493ebb16dc1e5c4021b8a9995c415d2`.
- Independent report: `runs/definition-tutor-pair-analysis-local/attempt-001/independent-analysis.json`.
- Independent SHA256: `359296b2ace00ddf61a7ea7ff7517ec94cc510f5c10bd9549fc5fed5ab9a10b5`.
- Accepted teacher result SHA256: `e9d7cf4ad140df1a4e225a4cbbf99a607a2baa82a2105a15e9a9c94423626192`.

Focused worker, data and independent arithmetic tests passed. Two worker
validation harness failures occurred before any test ran and remain preserved;
neither performed neural work. The experiment itself completed on its first
execution, and the independent recount verified it on its first invocation.
