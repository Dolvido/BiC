# Shared-learner width: prospective protocol

This is a new local experiment following the completed realization comparison.
That comparison did not establish that insufficient capacity caused weak fresh
learning. This study asks whether increasing encoder width, with symmetric rate
calibration and the same verified curriculum stream, improves fitting and
generalization across three domains. It does not select a deployable BiC policy.
Freeze this protocol, source closure, runtime profile and banks before formal
calibration. No audit result may select a rate, endpoint or continuation budget.

## Intervention and budgets

Compare the existing raw-byte SequenceStudent at widths **96, 192 and 256**.
Keep four encoder layers, four attention heads, feed-forward width four times
encoder width, maximum 12 turns and maximum context 1,024 tokens. Parameter
counts are **753,610 / 2,221,738 / 3,692,010**, respectively. Preserve the existing
action/reply/next-byte objectives, normalization, clipping and AdamW recipe.
All arms use fresh realizations and jointly learn color, count and switch.

Each update averages three 32-episode microbatch losses, one per domain. Each
domain's independent sampler selects an 8-, 10- or 12-turn bank uniformly and
then 16 complete pairs with replacement. Use the same ordered structural and
realization draws for every width and rate within a phase. Shape-dependent
initialization shares a declared seed, not identical parameter tensors.

| Phase | Runs | Updates per run | Model / sampler seed |
|---|---:|---:|---|
| Development-only rate calibration | Three widths × three rates | 600 | 3100 / 4100 |
| Main comparison, newly initialized | Three widths | 7,200 | 3101 / 4101 |

Calibration rates are **0.0003, 0.001 and 0.003**. No weights or optimizer moments
transfer from calibration to the main phase. Each main arm receives 230,400
episode exposures per domain, 691,200 total. Formal work totals **27,000 retained
updates and 2,592,000 episode exposures**: 5,400 calibration updates and 21,600
main updates. Count declared execution probes and discarded work separately.

Matching covers consumed examples, truths, bytes and optimizer-update counts
within a phase, subject to successful exact stream verification. It does not
match parameter count, optimizer work, FLOPs, memory or wall time. Equal rate
grids and budgets do not establish equally optimal training for every width.

## Calibration selection

At exactly update 600, score every calibration development panel with teacher-free
actions and greedy replies. For each of the nine domain × length cells, average
equally across its four panels the mean of final action-pair accuracy and final
free-reply-pair accuracy. Final pairs have opposite known answers.

Rank each width's three rates lexicographically by:

1. The minimum of these nine cell scores.
2. Their equal-cell mean.
3. Equal-cell, equal-panel known-query accuracy.
4. Negative equal-cell, equal-panel unsupported-ASK rate on known queries.

Compare count-derived values rounded to 12 decimal places. Remaining ties prefer
**0.001, then 0.0003, then 0.003**. Preserve every selection component and original
denominator. Report action and reply scores separately, all domain/length/panel
cells, unknown recall, confusion, action/reply agreement and probability error.
Easy unknown queries cannot win the leading selection criteria. A zero worst-cell
score remains zero; do not omit a difficult cell or change the rule after scoring.

This single development endpoint chooses a rate only. Main outcomes use the
declared 7,200-update endpoint regardless of development scores. Single-seed
selection remains noisy and is not an estimate of each width's optimal capacity.

## Data and protected boundaries

Use new main bank namespaces **210,000,000 / 220,000,000 / 230,000,000** for
training/development/audit and separate calibration namespaces
**240,000,000 / 250,000,000** for training/development. Each main domain/length
training bank has 256 pairs; calibration has 64 pairs. Evaluation has four
panels (`name_only`, `value_only`, `both`, `composed`) for every domain and length,
with 64 pairs per main bank and 32 per calibration bank.

The familiar panels preserve an initial sampled procedure and change actual
naming, actual anonymous value realization, or both. Changed naming coordinates
are matched between `name_only` and `both`; changed value coordinates are matched
between `value_only` and `both`. A value change can alter increments and query
truths. The composed panel withholds final supervised copy/advance ancestry under
the existing finite partition rule. Every known composed training query, not only
the final query, must have training ancestry. Development excludes audit ancestry.

Independent abstract and English interpreters must agree. Each complete pair
differs in exactly one earlier set statement and has opposite known final
answers. Only observation bytes enter the learner; semantic state, provenance,
family identities, targets and teacher reply prefixes remain outside its inputs.
No truncation is allowed. Authenticate canonical rows before targets can train
the model or contribute to evaluation.

Authenticate historical bank hashes and earlier endpoint cumulative transcript
digest sets before using them as exclusions. Reject whole pairs if either
individual observation transcript is protected. Keep all new initial and
evaluation transcripts disjoint. Protect both phases' evaluation banks and the
other phase's initial training bank during fresh generation. During preparation,
purely replay the prescribed 600-update calibration stream with sampler 4100,
without training or model scoring. Store its planned consumed transcript set and
add it to main protection before freezing. All nine calibration jobs must later
match this same planned stream exactly before main optimization begins. Record
the protection union's actual count/hash; repeated calibration runs do not create
nine times as many distinct protected examples.

Record actual distinct transcripts, procedures, query ancestries, naming/value
realizations, truth mixtures by position, bytes/tokens, context maxima and
collision retries. Stop on frozen search-bound exhaustion. Repeated finite
ancestry motifs and Boolean states are expected; historical transcript exclusion
does not establish new algorithms, unseen vocabulary or semantic independence.

## Runtime and evidence gate

The previous separate backend probe loaded a checkpoint exactly but did not
produce bitwise-identical CUDA trajectories in repeated uninterrupted or resumed
executions under its default settings. It did not isolate a kernel cause or an
additional reload effect. This study must declare a new runtime reproducibility
profile and pass recorded probes before source freeze or formal calibration.

Record software/device versions, precision and determinism settings, execution
path controls and probe seeds/data. Test repeat execution and split continuation
from the identical saved parent under the intended runtime. Cover each proposed
width. Compare weights, AdamW state, stream, sampler, occurrences and cursor,
excluding only explicitly identified timing fields. If checks fail, stop before
the formal study; do not relabel numerical drift as exact continuation. Probe
passing establishes reproducibility for the tested cases, not every possible
future workload or a performance benefit. Record all probe cost separately.
The declared width-96 probe uses 32 uninterrupted updates, 16 plus reload plus
16, and an independent 32-update repeat: 96 physical updates. Widths 192 and 256
each use 16 uninterrupted, 8 plus reload plus 8, and an independent 16-update
repeat: 48 updates per width. Compare the saved midpoint learning states exactly
before attributing endpoint differences to continuation. Verify actual coverage
of all three length buckets. Total planned profile-probe work is **192 updates
and 18,432 episode exposures**, additional to formal study work. There are no
undeclared retries. A final coverage receipt binds all three per-width receipts
and artifacts; preparation verifies their flags and matching runtime profiles.

Freeze copied dependency sources, protocol/bank identities, model configurations,
initial-weight digests, sampler rules and expected structural exposures. Exact
resume includes optimizer moments and the complete fresh-stream collision and
occurrence state. Preserve original checkpoint files and verify them on read.
Run through the existing transactional backend, committing every 16 updates and
at normal stops. Main scheduling uses finite blocks of at most 600 updates;
checkpoint payloads bind the scheduled block and cursor to the learner state.
Record backend overhead and discarded work separately when observable.

After all main endpoints complete, reconstruct each complete consumed stream
on CPU from the frozen generator, checking every saved scoring boundary. Bind
the final report and latest learner to the same final checkpoint, including the
full optimizer tree, sampler state, stream digest and exposures. Replaying input
generation does not reproduce optimization or independently verify AdamW values.
Do not release any sealed audit result until all widths pass these gates.

## Evaluation and interpretation

Save development/checkpoint evidence at updates **0, 600, 1,800, 3,600 and 7,200**.
After the gate, evaluate these same five boundaries on all frozen audit panels.
At each final endpoint also evaluate the common initial-realization bank and
each arm's latest actually consumed realization per recipe, with occurrence
coverage reported. These fitting diagnostics are bounded samples; latest fit
is not unbiased full-stream accuracy. No audit-driven early stopping or selection.

Report final opposite-answer action/reply pairs, earlier eligible paired queries,
query positions, known/unknown denominators, unsupported ASK, confusion,
action/reply agreement and probability error by panel, domain and length. Keep
absolute scores and regressions visible beside width differences. Sparse curve
areas are descriptive, not precise time-to-learning estimates. Final queries
are always known, so final-turn uncertainty is unmeasured.

At each endpoint run blank-text and reset-history controls against original
targets; reset presents each utterance alone. Verify CPU restart actions and
free replies exactly, and confirm no evaluation changes weights or files.
Concurrent maintenance of three domains is not a sequential forgetting study.

Record preparation, fresh generation/validation, optimization-containing step
time, scoring, peak memory and observed wall intervals. A measured generation
component is a subset of step time, not an additional cost to sum. Concurrent
worker intervals cannot be presented as dedicated GPU wall time. Include all
nine calibration runs when comparing project compute, and report interrupted
discarded work wherever known. Immutable authenticated caching can improve
execution efficiency without weakening row or source checks.

This single-seed width comparison cannot establish a general scaling law,
statistical significance, broad intelligence or unique capacity causation.
Representation and optimization change with width as well. A useful next result
requires reported gains in known paired answers and free replies across domains
and lengths, with costs and regressions exposed. There is no numerical promotion
gate, automatic policy replacement or autonomous selector in this study. Work
uses inert simulated objects and introduces no external actions or selection
pressure to pursue intellect at the expense of welfare or accuracy.
Earlier studies provide motivation, not a matched additional arm: the new
runtime profile, banks, initialization and longer budget differ from them.
