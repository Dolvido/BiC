# General learning: first local comparison

This prospective study replaces a sequence of task-specific English repairs
with a common learner tested across several verified mechanisms. Its aim is to
measure whether shared experience improves transfer, retention, and the speed
of acquiring another kind of task. High scores on these small synthetic worlds
alone do not establish general intelligence, unrestricted English, or autonomous
scientific discovery. The earlier English prototypes remain separate reference
checkpoints.

## Curriculum and student boundary

Use four families of six-turn English episodes:

| Family | Mechanism to learn | Generated uncertainty case |
|---|---|---|
| `variable_binding` | Store, copy, and revise a named value | A referenced value is not established |
| `graph_reachability` | Compose directed links into a path | No path is known and the world has not been declared complete |
| `arithmetic_updates` | Carry an initial amount through additions and subtractions | A queried entity has no established count |
| `conditional_logic` | Apply ALL/ANY conditions after switch interventions | An undefined lamp has no rule |

All use the same four actions and corresponding reply strings: yes, no, ask for
information, and acknowledge. The student receives raw English and zero-valued
nonlanguage observations. Family identifiers, difficulty tags, generator states,
oracle answers, and split membership are outside policy inputs. A learned
episodic-attention mechanism may retrieve learned representations of earlier
utterances through the regional brain; it may not retrieve parsed facts,
labels, or reference answers. No external tutor participates in inference.

The independent simulator verifies every transcript and target. A counterfactual
pair preserves the queries, wording choices, ordering, and irrelevant inputs,
while changing one relevant statement so that a known answer reverses. For
graphs, that statement contains two edges whose destinations are exchanged;
this is not a one-edge change. The exchange preserves each node's in/out degree
and, at levels 2 and 3, the queried terminal edge. It does not exhaustively
balance whole-graph motifs. Sample whole pairs from even seed boundaries.
Report individual query accuracy and the fraction of eligible **query-position
pairs** for which both opposite answers are correct. This counterfactual metric
does not require every answer in both six-turn conversations to be correct.
Reject inconsistent, ambiguous, truncated, or unsupported lessons before
training. Missing facts must not silently become false facts.

The generated conditional-logic lessons supply all switch values. Exactly a
quarter of their final queries ask about an undefined lamp, so the uncertainty
test concerns a missing rule, not partial Boolean knowledge. The oracle also
supports unspecified switch values, but this bank does not measure that ability.
The arithmetic amounts and operations come from small finite ranges. Successful
performance must therefore be described as learning these bounded worlds.

Keep the curriculum compact: generator rules, reviewed grammar templates,
world seeds, manifests, and bounded replay descriptors replace growing stores
of duplicate text. Parallel workers generate independent episodes and isolated
candidate students; each conversation retains its own recurrent state. Parallel
exposures count as compute and data exposure, not additional independent
research replications.

## Frozen first cycle

The first held-out family is `conditional_logic`. Train on the other three
families at levels 1 and 2 only. Level 3 tests longer graph paths, higher-arity
logic, and new copy operations for bindings and arithmetic. The latter two are
an operation-and-language shift, not merely greater depth. This is one
held-out-family rotation, chosen before results;
successful confirmation must rotate the held-out family. Do not select the
easiest held-out family after viewing results.

Compare the following four arms, starting from student initialization seed
2201. Use one parameter layout and identical initial weights: a byte-GRU
utterance encoder of width 128 feeding a regional brain of width 64. Record the
actual parameter count rather than treating its approximate size as measured.

| Student | Training order |
|---|---|
| Recurrent regional student; attention sees the current utterance only | Interleaved families |
| Recurrent regional student; attention sees the current utterance only | Blocked families |
| Same student; attention sees the causal utterance history | Interleaved families |
| Same student; attention sees the causal utterance history | Blocked families |

The comparison separates access to episodic history from curriculum order.
The recurrent state still crosses conversational turns in the current-utterance
control; it is not a context-free classifier. Both modes retain the same learned
attention parameters. No family-specific output heads or task-specific repair
losses are introduced. Use query action loss plus 0.25 acknowledgement loss,
0.1 reply-byte loss, and 0.1 generic causal next-byte prediction loss within the
current input utterance in both modes. This language-model objective predicts
the next byte from its prefix; it does not predict the next conversational
utterance. Keep learning rate, clipping, and target weighting identical.
Extra arithmetic and memory access in the episodic mode must still be measured.

Default budget: **1,800 optimizer updates, batch size 64, AdamW learning rate
0.003**, checkpoint every 300 updates. No score-based early stopping or
best-checkpoint selection. If a machine limit interrupts an arm, preserve its
state and compare the largest common completed update; do not describe a partial
run as the planned endpoint. Stop immediately on nonfinite loss or an integrity
failure, and report that run as failed. Each full arm presents 115,200 episodes,
generally with repetitions. A protocol manifest may freeze a benchmark-informed
budget adjustment before any optimization; changing it after scores are visible
is a new study. Keep adjusted updates divisible by three.

Construct one seeded list of complete-pair minibatches for each family, using
matched world banks balanced over levels 1 and 2: seed 2,000,000 plus 100,000
per family index and 10,000 per level, with 512 episodes per family/level.
A minibatch contains one
family. Interleaving cycles families 0, 1, 2; blocking uses 600 consecutive
updates per family. Both consume the **same exposure multiset**, with the same
within-family sample stream. The exposure list includes repeated bank examples;
equal exposure does not imply that every presentation is a distinct lesson.
State resets between episodes,
not between turns. Evaluate at the two block boundaries as well as ordinary
checkpoints. In confirmation runs, rotate the blocked family order and report
the last-family advantage instead of mistaking it for general retention.

Freeze before optimization: source hashes, model settings and parameter counts,
all seeds, ordered training exposure fingerprints, development and final query
fingerprints, adaptation support fingerprints, grammar versions, operation
depths, split membership, episode counts, scoring rules, and this selection
rule. Save optimizer, per-family samplers, model weights, total updates, and
family update counts so an interrupted cycle resumes its sampling stream.
This fixed-order study regenerates immutable banks from their manifests; it
does not yet contain an adaptive replay policy or learned curriculum scheduler.

## Development, final transfer, and adaptation

Maintain separate training, development, and final records. Primary ordered
alias-pair support follows the verified generator's declared partition: train
residues 0/1, development 2, final 3. Incidental entity pairs may overlap and
all aliases belong to a shared finite vocabulary. This withholds specified
primary compositions, not every possible combination or previously unseen words.
Count exact transcript
overlap; exclude a whole pair if either transcript overlaps its training or
adaptation support bank. Hash these exclusions before scoring models.

At fixed checkpoints, evaluate **all** trained families, rather than only the
family most recently practiced. Development uses split `dev`, 64 episodes per
family/level at levels 1 and 2, combined by family, from seed 4,000,000 with the
same family/level offsets. These checkpoints do not evaluate level 3 or the
held-out family. Grammar-only changes, new random seeds,
and zero exact overlap do not by themselves establish a new reasoning mechanism.
Do not feed development queries or their answers into replay.

Final evaluation occurs only after all four arms complete the fixed endpoint.
Freeze it before optimization with `experiments.audit_cognitive_transfer
--prepare DIR`. The final bank uses split `audit`: 256 episodes per trained
family at level 3 from seed 6,000,000, and 256 held-out-family episodes at each
of levels 2 and 3 from seed 8,000,000, with the family/level offsets above.
Report generated counterfactual groups and distinct transcripts as well as
generated rows; different seeds are not necessarily different semantic worlds. The final
bank must not guide architecture changes, data selection, or checkpoint choice;
after it has been inspected, it is a known benchmark for subsequent cycles.

For the entirely held-out family, report zero-shot performance and adaptation
after **1, 4, and 16 optimizer updates**. Use one fixed support set of 64 episodes
(32 complete pairs), split `train`, restricted to levels 1 and 2, with disjoint query
compositions and transcripts. Each update samples 64 episodes with replacement
as whole pairs from that same support bank.
Support starts at seed 7,000,000 plus 10,000 per level, with 32 episodes per
level. The curve therefore spends 0, 64, 256, and 1,024 support exposures. Evaluate
separately on at least 256 level-2 query episodes and 256 level-3 query episodes.
Start every candidate's adaptation from its own unchanged endpoint, reset
optimizer moments, and use sampler seed 3101, learning rate 0.003, and the same
support order. These adapted copies never overwrite the parent checkpoint.

Include fresh random initialization in both memory modes, using the same starting
seed, architecture, support, and adaptation budget. This determines whether prior
learning improves on learning the small task from scratch. Score only the held-out query set;
support-set memorization is not transfer. Test retention before and after
adaptation using all three original families, split `dev`, 64 episodes per
family/level at levels 1 and 2, from seed 9,000,000 with the same offsets.
These are known development-family conditions. Also measure each trained
family's level-3 final transfer before and after adaptation. The first cycle
uses one frozen support draw; run
at least three disjoint support draws in confirmation, retaining each paired
comparison rather than treating repeated queries as independent subjects.

This is a **measurement of fast adaptation**, not proof that the optimizer itself
has learned or that a meta-learning objective was implemented. Holding out a
family may also change grammar and vocabulary; record those changes, and avoid
attributing the entire gap to a single reasoning operation. A later isolated
mechanism claim needs matched linguistic coverage.

## Scores, controls, and the next decision

Report equal-family macro query accuracy, eligible complete-pair accuracy,
allow/deny macro accuracy, ASK recall and precision, each family's results,
generated reply accuracy, and action/reply disagreement. Include denominators.
Also report query log loss or Brier score; confidence without correctness is
not an improvement. Never let a frequent ASK or acknowledgement label hide
failure on known answers.

For each family, forgetting is its best earlier development query accuracy
minus its endpoint accuracy, floored at zero. Retain the full checkpoint-by-
family matrix. Report adaptation area under the query-accuracy-versus-update
curve, the fixed 16-update endpoint, and the gain over the same architecture's
fresh-initialization curve. Neither a single favorable intermediate checkpoint
nor an averaged score that hides a collapsing family meets the screen.
The audit's adaptation macro averages level 2 and level 3 equally. Its normalized
area uses trapezoids between only 0, 1, 4, and 16 updates, so the long final
interval has the largest weight; it is not a densely measured learning curve.
Report both level-specific curves beside this aggregate. Generated reply scores
are collected at update 16, while earlier adaptation points score actions only.
Action/reply agreement can occur when both outputs are wrong and must be read
alongside correctness.

At adaptation update 16, evaluate reset-between-turns and blank-English controls
on the held-out query banks, leaving labels unchanged. The separately trained
current-utterance mode supplies the matched memory-access comparison; scrambled
retrieval is not implemented in this first audit. Check finite values,
unchanged weights during evaluation, and exact CPU save/reload continuation.
Record any CUDA restart tolerance separately from a bitwise guarantee.

The first-cycle **exploratory benefit screen** requires at least a 10 percentage
point gain in held-out-family query accuracy after 16 adaptation updates over
the corresponding current-utterance memory control with the same curriculum
order. It also requires positive complete-pair accuracy without degradation,
an improved adaptation curve, familiar-family macro query/pair scores within
2 points of that control, no individual family's query score more than 5 points
lower, and ASK precision/recall each within 2 points where the evaluated bank has
the corresponding denominator. Apply both macro retention limits and the
per-family retention limit **before and after adaptation**, each against the
matched current-utterance control at the same phase. At adaptation update 16,
check ASK precision and recall separately on each held-out query level. Both
models must have the respective denominator; a missing denominator is unmeasured
and does not pass that screen. Examine every transfer
cell and each support draw even if the aggregate screen passes. If both
attention orders pass, prefer the higher held-out adaptation curve; break a
tie within 1 point by lower measured training time. This rule selects further
research, not a general-capability champion.

A promising comparison is repeated with initialization seeds 2203 and 2209
and the same frozen budget. Report means and sample standard deviations across
initializations; shared test scenarios are not independent benchmark replicas.
Confirmatory development rotates all four held-out families and retains the
parameter-matched control before claiming a broadly useful mechanism. General
capability claims retain the repository's requirement for at least five seeds.

No automatic promotion follows a screening gain. The existing experimental
competence targets remain useful minimums within each evaluated family: 80%
query accuracy, 60% complete-pair accuracy, 75% allow/deny macro accuracy, and
a 10-point complete-pair advantage over blank-text and reset-state controls.
They must be accompanied by transfer, retention, and meaningful adaptation
benefit. These targets are research criteria, not definitions of intelligence.

If no arm improves the held-out adaptation curve, do not expand training hours
on the same configuration. Inspect common failures across families and propose
one reusable change to representation, memory, credit assignment, or learning
procedure. Freeze a new protocol and rerun the broad matrix. A future learned
neural key/value memory is a candidate mechanism; an oracle memory table or
task-specific parser at inference would answer a different question.

## Compute, self-direction, and ethical scope

Use the local RTX 5080 and CPU only. Record wall time, aggregate worker seconds,
updates, repeated exposures, parameter counts, bank sizes, and distinct audit
transcripts. The separate short benchmark records update throughput and peak
allocated CUDA memory. The training runner records elapsed training and overall
worker times; it does not measure FLOPs, energy, peak memory over the full study,
or standalone inference latency. Optimizer/replay byte accounting and a dedicated
common-compute performance comparison are not implemented in this first cycle.
Consequently the primary comparison is equal updates and exposure, not equal
compute. Four concurrent CUDA workers contend for one GPU; summed worker
intervals are not dedicated GPU-hours or elapsed wall time. Report this
concurrency before drawing an efficiency conclusion. The smallest experiment
that distinguishes the hypotheses takes precedence over saturating the GPU.

This first cycle fixes curriculum order to identify its effect. Subsequent
self-directed cycles may allocate lessons by held-out development progress and
uncertainty, while reserving equal minimum exposure across established families
and bounded replay. Every cycle evaluates the same broad retention/transfer
matrix and records the policy's choice, its evidence, and the resulting gain per
compute. The scheduler cannot inspect final queries, rewrite scoring rules, or
declare success by lowering gates. Distinguish engineered scheduling from
learned self-direction and measured meta-learning.

Evolutionary proposals, if added, remain bounded offline candidates. They may
vary a declared learner or curriculum setting, but may not modify oracle truth,
final-bank membership, audit checks, resource limits, or promotion criteria.
Preserve the last validated checkpoint and provenance for every candidate.
Evaluate benefit through accurate learning, useful uncertainty, retained ability,
and resource efficiency, not parameter growth or intellect alone.

The worlds contain inert objects, amounts, links, and switches. They require no
harmful real-world actions, manipulation of people, or simulated distress as a
training signal. Nothing in this benchmark establishes consciousness or welfare
experience. Ethical selection here means truthful evidence, bounded local
experiments, reversible changes, and capabilities developed for beneficial use.
