# Shared context diagnostic: completed results

The trained learners remain weak even when the relevant causal chain ends
immediately before the known question. Moving it farther away does not produce
a consistent all-family deterioration. This does not support selecting a
near-to-far teaching schedule as the next intervention. No learner was promoted.

The new grammar also exposes poor uncertainty handling: the unknown question
is last, whereas the original foundation grammar always ends with a known
counterfactual question. All three baseline models answer zero of these 2,016
unknown questions correctly through their action output. This is an observed
generalization failure under the new grammar, not a causal identification of
position shortcuts or the source of the earlier training failure.

## Scope and completed execution

The [prospective protocol](FOUNDATION_CONTEXT_DIAGNOSTIC_PROTOCOL.md) fixed
2,016 episodes: three semantic families, seven operation/depth groups, eight
base pairs per group, three placements and two consistent naming conditions.
Each episode has ten statements followed by a known and an unknown question.
Only the initial causal fact changes within an opposite-answer pair. Placement
preserves statement text, values and total bytes; renaming preserves role
relations and byte lengths. The original intermediate query is absent.

All six completed objective endpoints and their three common initial states
were evaluated on the local RTX 5080. The nine states were fixed before scoring;
none was replaced by a better historical checkpoint. Native actions and freely
generated query replies receive separate scores. Targets, recipes and oracle
state never enter model inputs. All parameter digests remain unchanged.

- Frozen launch (archive reference: `../runs/foundation-context-diagnostic-local/attempt-001/launch.json`):
  `2511467a862262a0004480838e2465324a2f27ec1818c90c9427b61f69fe02ba`.
- Completed inference (archive reference: `../runs/foundation-context-diagnostic-local/attempt-001/inference/receipt.json`):
  `0f6e26de65eab8468f16ffed28dcc946d2a07dd106140d422f4c48406d7efc6d`.
- Independent arithmetic analysis (archive reference: `../runs/foundation-context-analysis-local/analysis-001.json`):
  `ee27f90164b012409a5a9eede67bb4341c181b0e68a4fc14343b2e147975d313`.

Inference completed on September 17, 2026 at 14:24:19 UTC: 18,144 episode
evaluations, 36,288 scored query contexts, 567 sequence forwards and 9,688 free
decoder recurrent calls covering 620,032 row steps. The native sequence calls
also process 217,728 BOS-only reply contexts. There were nine model constructions
and nine weights-only archive loads, zero backward passes, optimizer updates or
tutor calls, and no unmatched work attempts.

Elapsed inference cost was 20.750 seconds, with 17.703125 CPU seconds. Peak CUDA
allocation was 551,263,744 bytes; reserved memory peaked at 2,013,265,920 bytes.
Preparation cost another 1.422 seconds and generated exactly 2,016 episodes.
Inference admission revalidated the saved bank without generating replacements.
The independent stdlib analyzer took 3.156 seconds, reauthenticated inputs,
decoded raw replies and reproduced every metric and matched transition table.
It performed no model imports, archive deserialization, generation or inference.
These measurements are diagnostic cost, not training throughput.

## Paired comprehension

A successful pair requires both opposite-answer members to be correct. Entries
are action pairs / reply pairs; every row has 1,008 pairs. The three untrained
states scored zero in both outputs.

| Trained endpoint | Action pairs | Reply pairs |
|---|---:|---:|
| Baseline 8472 | 64 | 0 |
| Balanced reply 8472 | 87 | 13 |
| Baseline 8473 | 91 | 74 |
| Balanced reply 8473 | 62 | 26 |
| Baseline 8474 | 41 | 61 |
| Balanced reply 8474 | 65 | 82 |

All groups and both naming conditions are included below. Each cell contains
112 pairs; the three entries are gap 0, gap 2 and gap 4. They are matched
conditions, not independent repetitions.

| Endpoint | Color: action/reply | Count: action/reply | Switch: action/reply |
|---|---|---|---|
| Baseline 8472 | 19/0, 12/0, 23/0 | 0/0, 0/0, 1/0 | 3/0, 5/0, 1/0 |
| Balanced 8472 | 15/3, 15/4, 17/5 | 0/0, 0/1, 0/0 | 11/0, 15/0, 14/0 |
| Baseline 8473 | 16/18, 18/15, 12/27 | 0/0, 0/0, 0/0 | 15/6, 16/6, 14/2 |
| Balanced 8473 | 14/9, 18/8, 22/8 | 0/0, 0/0, 0/1 | 6/0, 1/0, 1/0 |
| Baseline 8474 | 0/17, 0/15, 0/17 | 0/0, 0/0, 0/0 | 23/3, 11/2, 7/7 |
| Balanced 8474 | 18/23, 13/18, 19/28 | 2/0, 0/0, 2/0 | 3/3, 2/5, 6/5 |

The direct-fact count group scores zero paired actions and replies even at gap
zero in all six models (16 pairs/model). This observation is included with the
full matrix; it is not a request for a separate arithmetic repair.

## Unknown answers and naming

Each endpoint sees 2,016 unknown questions at the final turn. Entries count
correct unknown actions / exact freely generated unknown replies:

| Endpoint | Correct unknown actions / replies |
|---|---:|
| Baseline 8472 | 0 / 12 |
| Balanced 8472 | 71 / 30 |
| Baseline 8473 | 0 / 0 |
| Balanced 8473 | 0 / 0 |
| Baseline 8474 | 0 / 0 |
| Balanced 8474 | 131 / 133 |

Renaming changes the known action on 33–285 of 1,008 matched episodes across
the six trained models. The exact paired transitions, reply changes and all
family/group conditions are retained in the analysis. Neither unchanged
predictions nor a naming effect by itself establishes comprehension: a constant
wrong answer can be perfectly invariant.

## Next shared learning comparison

Do not launch the conditional near-to-far order comparison: its prerequisite
of useful shared near-context competence and a consistent distance decline was
not established. Do not infer that the architecture lacks a representation or
that a new auxiliary objective must solve the problem.

Prioritize a prospective **causally varied lesson-layout curriculum** against
the unchanged layout. Existing training already varies interior distractors
and some questions, but always appends its designated known anchor last.
The new candidate should preserve each lesson's exact events, wording, targets
and every query's read-version ancestry while varying valid query placement
across all families and depths. It needs an explicit anchor identity/position
and a new schema; the existing final-turn validators and scorers cannot be
silently reused. Feasible layouts depend on dependency depth and turn budget.

This tests a general teaching hypothesis across the curriculum. It is not yet
implemented learning evidence, and it is not a claim that position alone caused
the failures. Keep complete matched controls, new frozen lesson seeds, native
actions and replies, old-grammar retention, fresh layouts and accurate unknown
answers. Keep tutor selection and parallel execution as separate comparisons.

The bank was designed after observing earlier results. It changes query density
and grammar, and moves absolute position together with distance. Three seeds
share the same diagnostic examples. General English comprehension, independent
learning, useful self-direction and the complete external-tutor loop remain
unproven.

## Software validation

The generator passed four CPU tests using one 252-episode fixture and sixteen
malformed variants, with Torch imports blocked. The evaluator passed eight
separate mocked checks: twelve synthetic records, twelve fake descriptors,
five malformed cases, four fake models and seven fake sequence calls. One
deliberately failing decoder accounts for its unmatched mock attempt. No real
neural work or diagnostic generation occurred in that mocked suite. All
production inference work is recorded above, separately from these checks.
