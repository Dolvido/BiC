# Continued fitting and replay: prospective local study

This comparison tests whether more fitting of the broader curriculum and a
generic replay mechanism improve acquisition and retention. It follows the
[completed diversity study](DIVERSITY_STUDY_RESULTS.md), which found improved
naming generalization but weak paired reasoning and substantial forgetting.
The causal sequence learner remains a diagnostic model, separate from the
untrained regional integration. No result or checkpoint promotion is assumed.

## Four parent endpoints, two adaptation conditions

Use the two eight-map diversity arms, `w32-n8` and `w256-n8`. Preserve their
completed 3,600-update checkpoints. Continue separate copies to 14,400 total
updates using each original training bank, optimizer moments, family schedule
and sampler states. This adds 10,800 updates per arm. The model architecture,
753,610 parameters, objectives and learning rate 0.001 remain unchanged. There
is no new calibration or learning-rate search.

The four parents are therefore 32/256 anonymous worlds per family crossed with
3,600/14,400 cumulative updates; all have eight fixed naming maps. Longer
training changes exposure, not curriculum breadth. It provides a controlled
fitting comparison before attributing the broadest bank's earlier incomplete
graph fitting to model capacity. Original checkpoints remain intact. Fixed
development measurements describe progress; they do not select an endpoint.

Adapt every parent under both ordinary training and replay, plus one common
fresh seed-2601 ordinary control: nine candidates in total. Every adaptation
starts a fresh AdamW optimizer at 0.001 and sampling seed 3701. The sole new
subject is arithmetic. Ordinary and replay conditions receive the same complete
new-example pairs, in the same draw order, and exactly 64 new episodes per step.

Replay adds 32 old episodes from one of the three old families per step, with
balanced interleaving and independent pair samplers. Replay draws never advance
the new-example sampler. The loss is the unchanged new-example sequence loss
plus **0.5 times** the unchanged old-example loss. Both gradients feed one
optimizer update and the same clipping rule. Replay is additional computation
and additional exposure; this is **not a compute-matched comparison**.

The old buffer contains at most 256 complete canonical training pairs per
family, selected deterministically from that parent's original bank. It does
not admit development or audit rows. Record actual pair identities, transcript
counts, canonical storage bytes and raw observation bytes. Both parents supply
256 pairs per family: the smaller covers 32 worlds and the larger covers 256,
each across all eight naming maps. The same buffer is used for that bank's
short and long parent endpoints. The study does not claim biological systems
consolidation or a trained meta-learning algorithm.

## Accurate compact lessons and fresh boundaries

Reuse the frozen v3 six-turn event-program curriculum, its independent abstract
and English interpreters, exact row authentication, complete counterfactual
pairs and separate world/name partitions. Only raw observed English enters the
policy. Teacher labels, replies, parsed states, family identifiers and provenance
remain outside the observation stream. Evaluation uses freely generated replies
from BOS only, without an external teacher or pretrained model.

Prepare new development, retained and advanced old-subject pools before any
continuation or adaptation. Exclude alias-normalized individual transcripts
from prior world pools when constructing new worlds, rejecting whole pairs when
either member overlaps. Do not rely on pair identity alone: one transcript can
belong to multiple valid counterfactual pairs. Naming-only panels intentionally
reuse their declared familiar worlds. Preserve exact exclusions, names, recipes,
source hashes and bank identities in the preparation manifest.

Old-subject development and retention distinguish three panels per family:
familiar worlds with withheld naming combinations; new worlds under a familiar
mapping; and new worlds with withheld naming combinations. Advanced old-subject
worlds use level 3. Arithmetic support uses 128 level-1/2 world pairs rendered
through eight training maps: 2,048 support episodes. Its query panels are:

| Panel | Query episodes | Boundary |
|---|---:|---|
| Names | 128 | Familiar 32 level-2 support worlds, two withheld naming maps |
| Worlds | 256 | 128 new level-2 worlds, a familiar support naming map |
| Both | 256 | The same new level-2 worlds, withheld naming maps |
| Advanced | 256 | 128 new level-3 worlds, withheld naming maps |

Report all four separately. Equal-panel macros weight these panels equally,
not by query or episode count. The names panel tests a declared familiar-world
boundary; it must not be described as unseen worlds. Support-to-query exclusions
apply to new worlds and preserve complete pairs.

The advanced arithmetic panel uses previously unused level-3 worlds from the
generator's development world partition, sealed for this audit, with audit row
admission and audit naming maps. They are not scored for development or used for
optimization. Prior single-variant exclusions still apply. The older level-3
audit world partition was nearly exhausted: this finite grammar has only about
672 possible level-3 programs. The declared partition choice supplies fresh
unused evidence; it does not make the curriculum open-ended. Future broader
studies will require an explicitly versioned expansion of these compositions.

These are restricted synthetic lessons. World fingerprints are syntactic
identities, not proof of semantic equivalence. Withheld naming combinations
reuse familiar vocabulary. Arithmetic digit bytes were absent from original
main-subject text. Level 3 changes copy/update compositions and target mixtures;
it is not a pure scalar increase in difficulty. Arithmetic level 3 has no true
unknown targets. Some conditional interventions are irrelevant or leave outcomes
unchanged, so later-query accuracy is not uniformly changed-outcome reasoning.

## Fixed audit and resource accounting

Score arithmetic at cumulative budgets **0, 4, 16, 64 and 256** updates. Before
and after adaptation, score every old-subject development, retained and advanced
bank, including generated replies. At 256 updates, run blank-text and reset-history
controls on all four arithmetic panels. Verify that a serialized CPU restart
reproduces actions and generated replies exactly. Preserve parent checkpoint
digests and save complete optimizer/sampler snapshots for the new candidates.

Report query accuracy, opposite-answer paired decisions, later-known decisions
and their pairs, uncertainty, calibration, free reply correctness and action/reply
agreement. Retain per-panel/family denominators. Sparse trapezoidal query, pair
and later-known curve areas are descriptive; compare both areas and final
endpoints, because early scores and later deterioration can change their ranking.
Absent denominators remain absent, rather than being scored as zero.

Keep resource quantities separate:

- Two continuations: 21,600 additional optimizer updates and 1,382,400 old-subject
  episode exposures. Earlier parent histories are not counted again.
- Nine adaptations: 2,304 optimizer updates and 147,456 new-subject exposures.
- Four replay candidates: 32,768 additional old-subject exposures, with no extra
  optimizer steps but extra forward/backward work.
- Total new work: 23,904 optimizer updates and 1,562,624 repeated episode
  exposures across the declared phases, including replay.

Record support and replay observation/reply byte and token exposures, buffer
storage, synchronized training time and total wall time separately. Training
time includes both losses and optimizer work; no unmeasured per-component GPU
time is inferred. Extension invocation time starts after setup and excludes the
final report-scoring pass; external process intervals include it. Concurrent
worker intervals are not dedicated GPU-hours.
Throughput does not establish useful learning efficiency.

## Decision boundary and broader objective

There is no numerical promotion gate or automatic winner. This is one parent
initialization, one withheld subject and a fixed replay weight. A favorable
result supports replication across seeds and withheld-family rotations; it does
not establish general competence. Earlier inspected audits remain historical
evidence, not untouched future benchmarks. Changes after reviewing this audit
require a new prospective protocol and new holdouts.

Judge whether the new ability is acquired while earlier abilities survive,
including paired decisions and generated language. The recent sequence learner
often agreed with its own wrong replies before adaptation, so output alignment
alone is not the primary hypothesis here. Regional routing/alignment remains a
separate unresolved integration question.

The user's objective remains a useful local learner with reusable knowledge,
self-directed practice, durable gains and progressive independence from an
external teacher. Local compute, accurate lessons and beneficial retention guide
this work. All lessons concern inert simulated objects; no real-world actions or
harmful evolutionary selection participate. Passing these restricted tasks or
using more GPU time does not complete the broader objective.
