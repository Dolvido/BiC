# Preserving learning while overlapping lesson preparation

Recorded 2026-09-17 after the standalone CPU preparation checks, while the fixed
objective main comparison is running. No main objective scores have been read.
This is an execution-efficiency change, independent of that comparison's learning
outcome. It introduces no curriculum, objective, model-size or inference change.
The active study and its 95 launch pins remain unchanged.

## Integration contract

Use an additive versioned trainer with the same flat model and the two explicit
objective IDs, `baseline` and `balanced_reply`. Its only change to learning is
acquiring already prepared CPU tensors and transferring them to the declared
device. Preserve the original three separate family forwards/backwards, each
loss divided by three, one gradient clip and one AdamW update. No mixed precision,
asynchronous optimizer, merged microbatches, source-content cache or different
lesson stream is introduced.

The real process-owned plan index, exact plan/protection/configuration, full
canonical evidence and immutable source identities remain required. Preparation
holds the current bundle plus at most one next bundle. It cannot supply a teacher
answer as inference input. Semantic labels and reply prefixes stay in their
existing training-supervision paths.

Starting preparation requires a finite stop cursor and time allowance. The unit
of nonpreemptive learner work is one complete three-family update: check the
allowance after acquisition, validation and the first transfer, immediately
before the first forward. After that boundary, finish the existing family,
backward, clip and optimizer order. A deadline prevents another update; any
overrun and producer shutdown work remain visible. CPU preparation may finish its
already active operation after stopping, but cannot authorize further learning.

A snapshot drains/closes the preparation owner and refuses an unjoined producer.
Transient queues and live tensor leases never enter the checkpoint. Restarting
preparation after a snapshot or restore requires a new explicit bounded invocation
and charges any repeated preparation. A new checkpoint schema and exact recipe
bind the preparation policy and source closure. Historical objective snapshots
are not accepted implicitly. The existing strict weight, alias, complete AdamW,
canonical prefix and timing checks may be reused through a private envelope
adaptation after validating the new public schema and recipe.

Invalid restoration must leave the learner's weights, optimizer, cursor and
canonical evidence unchanged. Explicit queue drainage is a separate observable
side effect. Failed neural work poisons the trainer until explicit strict restore;
the report retains attempted/completed family work and actual or unknown physical
optimizer work. Releasing prepared storage never certifies a successful update.

Keep CPU preparation time and exposures separate from learner step time and
exposures. Overlapping work is not additive wall time. The old checkpoint field
for materialization included inside a step is zero in the new recipe; the new
timing scope explains this. Preparation invocation reports, including discarded
and incomplete work, must survive the surrounding invocation ledger. No speed
claim may omit setup, guards, packing, transfers, checkpoint drainage or shutdown.

## Declared CPU validation before any GPU work

Use one newly constructed, admitted 66-bundle fixture, microbatch two episodes
per family, including a deliberate naming-only admission repair and one real
shared index. Use width 8, one layer, two heads, feedforward 16, and the existing
12-turn/input/reply/position limits. All six depths and three lengths must appear.
Initial admission and index reconstruction/scan are additional preparation work.

For each objective, run the original synchronous trainer for all 66 updates.
Compare it with the new trainer for 33 updates, snapshot/strict restore, then
33 more updates. Compare every matched step's exact weights, full optimizer
state, cursor, consumed canonical evidence and loss fields. Timing and the
explicitly different execution/source recipe are excluded only from the cross-
implementation trajectory comparison. A checkpoint reload itself must preserve
its own complete saved learner state.

The core comparison uses **264 physical optimizer updates, 792 family forwards,
1,584 episode forwards and 792 backwards**. Allow at most four additional
prescribed successful updates for interrupted-family restoration and preflight
refusal checks; total ceiling **268 updates**. Bound all attempted family
forwards at 810, attempted episode forwards at 1,620, backwards at 810 and
model/template constructions at 20. The test source must declare its exact
smaller workload before execution; these ceilings are not permission to add
unplanned tests or retries.

Check changed schema/objective/source/index/protection, wrong or missing live
preparation, an exhausted allowance before the first forward, a deliberate
partial-family failure, poisoned continuation refusal and exact explicit
recovery. Source changes are simulated at the guard return boundary; no frozen
file is modified. Record preparation/lease counts separately, and preserve any
failed attempt rather than silently rerunning it. CUDA must remain uninitialized.
CPU development concurrent with the formal study is disclosed and cannot provide
an isolated timing or training-efficiency claim.

The initial fixed test workload uses plan seed 831909001, order seed 831909002,
model seed 831909799 and learning rate 0.001 for both objectives. Its exact
ceiling is **265 updates and 11 model/template constructions**: 797 attempted
family forwards (1,594 episode attempts), 796 entered native family forwards
(1,592 episode forwards) and 796 backwards. The deliberate second-family
interruption fails before entering that native forward. After the separate
admission/index setup, bound returned preparation at 269 canonical bundles
(1,614 episodes) and 807 family packing calls (1,614 packed episode passes).

Three of those prepared bundles add no learning: two are queued and discarded
when snapshot drains a live owner, and one reaches the first-family transfer
before the fresh pre-forward allowance check refuses it. These explicitly test
the integration boundaries beyond the standalone owner's earlier CPU proof.
The source and complete launch manifest are frozen before this single run.

## Subsequent GPU proof and measured benefit

After the current formal GPU comparison finishes, fix one bounded local CUDA
proof before executing it. It must cover both objective IDs, all six depths and
three lengths, strict FP32 runtime, synchronous versus prepared trajectories,
repeatability and split/save/restore. Record the exact model, seeds, rates,
lessons, update/episode ceilings and allowance in a separate frozen run contract.
No settings are selected from partial trajectories. A within-process proof does
not establish cross-process determinism.

Only after exact CUDA trajectories pass, predeclare a small matched throughput
comparison with identical lessons, initialization, objectives and updates, a
counterbalanced execution order and a meaningful improvement threshold. Include
all invocation setup, the real protection inventory, queue waiting, transfers,
validation, saves and shutdown; distinguish shared prerequisite costs from
per-invocation work. Repeated full protection checks and Python contention may
erase overlap gains. No adoption follows from a fast inner loop alone.

This document authorizes neither a new main learning experiment nor a checkpoint
promotion. Learning benefit, retained transfer, external-tutor independence and
useful self-direction still require their own completed evidence.
