# Completed recurring tutor and practice campaign

The campaign completed seven paired cycles and retained a learner at update
9,112, up from 7,816. Its selected development result improved from 257/324 to
271/324 joint pairs, with higher final fit, retention and both transfer scores.
No tutor branch was selected and no tutor learning advantage was demonstrated.
The final owner exited successfully at its declared budget boundary, leaving
cycle eight authored and compiled but untrained.

Each cycle started both branches from the same full model and AdamW state,
provided 108 teaching updates and 108 common tutor-free practice updates, and
evaluated native actions plus freely generated replies at relative updates
0, 108 and 216. Neither evaluation nor the training worker contacted the tutor.
The [fixed policy](CONTINUOUS_TUTOR_CAMPAIGN.md) required a five-point tutor
development advantage, family and transfer checks, and retention against both
the current parent and the fixed campaign-start learner. Selection is an
engineered continuation rule, not model promotion or learned self-direction.

The following trajectory describes the **selected learner**, not the better
score chosen separately for each bank. Joint correctness requires both the
native action and generated reply to be correct for both members of each
opposite-answer focal-question pair.

| Completed cycle | Selected update | Dev /324 | Fit /54 | Retention /360 | Transfer original /216 | Transfer varied /216 |
|---|---:|---:|---:|---:|---:|---:|
| Start | 7,816 | 257 | 43 | 286 | 136 | 125 |
| 1: retain parent | 7,816 | 257 | 43 | 286 | 136 | 125 |
| 2: procedural | 8,032 | 260 | 48 | 279 | 144 | 140 |
| 3: procedural | 8,248 | 263 | 48 | 283 | 145 | 145 |
| 4: procedural | 8,464 | 265 | 48 | 279 | 138 | 143 |
| 5: procedural | 8,680 | 270 | 50 | 284 | 145 | 136 |
| 6: procedural | 8,896 | 260 | 50 | 286 | 145 | 143 |
| 7: procedural | 9,112 | 271 | 49 | 299 | 153 | 146 |

Cycle one changed 72/108 teaching bundles. Its procedural and tutor final
development scores were 267/324 and 264/324; neither met the complete selection
and retention conditions. The proposed procedural continuation lost color
retention from 110/120 to 100/120, beyond the five-point allowance, so the original
parent was retained. Cycles two through seven passed the parent-retention guard.

The actual teacher contrast was limited. Cycles 2, 4, 5 and 6 exhausted the
declared 64-candidate fresh-realization limit and used the explicit procedural
fallback. Their accepted recipes were preserved but not applied. Cycle 3 applied
the accepted recipe, but it was the procedural default. These five cycles had
identical teaching archives and identical branch answers at all three endpoints;
both physical branches were nevertheless run and are fully counted.

Cycle 7 again changed 72/108 teaching bundles. Its tutor endpoint scored 262/324
development, 50/54 fit, 293/360 retention, 142/216 original transfer and 147/216
varied transfer, versus the selected procedural endpoint in the table. It failed
the required development advantage, color and switch development joint checks,
retention joint comparison and original-transfer comparison. This is evidence
against a demonstrated advantage for these actual tutor proposals, not a test
of every possible form of external teaching.

Count reasoning remains the clearest family limitation. Final selected joint
scores were:

| Family | Dev | Fit | Retention | Transfer original | Transfer varied |
|---|---:|---:|---:|---:|---:|
| Color | 103/108 | 18/18 | 113/120 | 56/72 | 53/72 |
| Count | 65/108 | 13/18 | 74/120 | 36/72 | 31/72 |
| Switch | 103/108 | 18/18 | 112/120 | 61/72 | 62/72 |

The full known/unknown action/reply counts and every intermediate teaching and
withdrawal endpoint remain in the independent cycle reports. The finite
template and composition banks were repeatedly used for development and
selection. These are not pristine audit results, broad English comprehension,
or evidence that the learner can choose its own curriculum. The separate
[definition-basis experiment](DEFINITION_BASIS_PROTOCOL.md) tests English-defined
operations on a distinct branch; equal update counts never identify equal
learners.

Across the seven workers there were **3,024 physical optimizer updates,
290,304 episode exposures, 9,072 family forwards/backwards, 28 restores and
28 snapshots**, with zero unknown optimizer outcomes. The retained lineage
advanced by 1,296 updates: rejected and unselected branch work still costs
compute. Eight unique external teacher requests completed, using 8,460 prompt
tokens and 1,246 generated tokens. Their total recorded request wall time was
25.001 seconds, nested within the owner intervals.

| Preserved owner attempt | Outcome | Wall seconds | Owner CPU seconds | New worker updates |
|---|---|---:|---:|---:|
| 001 | Fresh-only admission refusal before training | 130.156 | 124.703125 | 0 |
| 002 | Cycle 1 complete; cycle 2 proposal exhausted | 460.953 | 258.015625 | 432 |
| 003 | Cycles 2-7 complete; budget stop before cycle 8 worker | 2,412.062 | 1,196.843750 | 2,592 |
| Total | All attempts retained | 3,003.171 | 1,579.562500 | 3,024 |

Launch preparation added 39.109 wall seconds and 37.890625 CPU seconds outside
those owner intervals. The seven worker intervals totalled 1,364.767 wall
seconds, including training, evaluation, restore and saving; this time is
already inside owner wall time. Worker-process CPU was separately 1,283.046875
seconds. Teacher-service CPU and engineering/testing time are not included in
that CPU sum. No GPU utilization or isolated GPU-compute duration is inferred.

The first failed compilation cost 118.422 seconds and disclosed 48 repeated
training examples; the additive recovery classified them as practice without
relaxing held-out exclusion. The second failed compilation cost 117.781 seconds.
The four retained finite-exhaustion proposal subreceipts total 223.484 seconds,
including the second failure's nested proposal work; these figures are already
inside the owner costs and must not be added again. All failures and old frozen
sources remain available through the [recovery record](CONTINUOUS_TUTOR_CAMPAIGN_FALLBACK.md).

Cycle 8 compiled successfully with 54/108 differing teaching bundles, at a
recorded compilation cost of 182.984 seconds. Its recipe was applied to lesson
artifacts only: **no cycle-8 worker, optimizer updates or evaluation ran**.
The final budget check had 596.360 seconds remaining, below the required
600-second worker allowance plus 60-second final reserve. The saved compiled
state can support an explicitly budgeted future continuation without repeating
its teacher request or compilation; none is counted as completed here.

Seven independent raw recounts verified 630 score groups, 210 bank endpoints,
98,280 episode records and 445,956 query records, plus the exact selection,
retention, work and committed-state transitions. A terminal aggregation
authenticated those receipts, all three owner attempts, eight author responses,
retained failed proposals, source closure and pending cycle-8 artifacts. It
decoded no model archives and made no model, generation or teacher calls.
The original cycle-1 recount's metadata reconstruction failure is preserved.
The first terminal aggregation also preserved a 0.625-second missing-context
failure in its report-only caller; the corrected aggregation passed in 1.219
wall seconds and 1.09375 CPU seconds without repeating old raw recounts.

Primary records and pins:

- Terminal owner summary: `runs/continuous-tutor-campaign-local/attempt-003/execution/summary.json`, SHA256 `76cf485b6f010c0d0c43f5ad17d7397829f6b05bb7896673a96f2adb5fe8ac8e`.
- Final compiled state: `runs/continuous-tutor-campaign-local/attempt-003/execution/states/0031.json`, SHA256 `063b69f666df92435035ffcef5bf2e8c68eebe12b1a929d7562fc7666b6f1fe4`.
- Selected checkpoint: `runs/continuous-tutor-campaign-local/attempt-003/execution/cycles/000007/worker/execution/checkpoints/withdrawal-procedural.pt`, SHA256 `1b2a6f7bf84deebe64571145ac0aca72beac2d5a6f75e1532803809eb8da702d`.
- Cycle-7 independent recount: `runs/continuous-tutor-campaign-analysis-local/cycle-000007-attempt-001/independent-analysis.json`, SHA256 `029e0a07a6b6a6cf4bd2eb767c00d6b7aa5d99a2b414a114d56c31681696feed`.
- Terminal aggregate, including all seven report pins and three recovery-cost records: `runs/continuous-tutor-campaign-analysis-local/terminal-attempt-002/terminal-analysis.json`, SHA256 `3bdccfbb3dc4f6eaa6bda10e3d17361dcac13fb1a6ec12818472b530f4690157`.
