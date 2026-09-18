# Foundation variant execution proof

This prospective engineering proof is separate from any capability comparison.
It does not select a learner or rate and does not authorize its own GPU launch.
Both architectures must pass all three rates before this proof can support a
new explicitly versioned comparison. Prior proofs retain their original scope.

The fixed CUDA workload uses width 192, four blocks, four heads, feedforward
width 768, the existing byte/reply capacities and twelve-turn capacity. Cases
are flat and hierarchical at learning rates .0003, .001 and .003, with fresh
seed 9911 initializations and the unchanged three-domain foundation objective.
Each update consumes three independent 32-episode microbatches. No pretrained
weights or formal study data are used.

A new 66-bundle plan uses seed 991000001 and mixed ordering seed zero. Its first
eight bundles exercise 8-, 10- and 12-turn inputs and depths 0, 1, 3 and 4; this
is not coverage of all six depths. The original color pair zero in every
executed bundle is protected, forcing deterministic naming-only admission
before and after the split at update three. Three additional canonical held
evaluation pairs (color/depth0/8turns, count/depth1/10turns,
switch/depth2/12turns on a familiar structure) are protected from the entire
admitted plan. These tiny banks test execution consistency, not competence.

Each process constructs and authenticates one in-memory plan index, reused for
all six cases. Construction includes admission reconstruction and a separate
full admitted-plan scan; it is recorded separately from training steps. Index
identity, original and expanded protection, frozen sources, inputs and strict
runtime settings are bound throughout. There is no persistent index loader.

Run separate processes in this fixed order after review:

1. `prepare`: freeze the plan, admission, protected banks, six CPU initial
   snapshots, source snapshots and protocol.
2. `uninterrupted`: eight updates per case, saving update three and eight.
3. `split-first`: three updates per case.
4. `resumed`: load the split snapshot exactly, then five updates per case.
5. `repeat`: another independent eight updates per case.
6. `verify`: CPU-only canonical/index, checkpoint, journal and saved-output
   verification; no additional inference or optimizer work.

The four workers perform **144 optimizer updates and 13,824 episode exposures**
in total: 24 updates per architecture/rate case. Sources and all workers are
one-shot; failures and started-step intents remain visible, with no automatic
retry. Dangling starts and unreported partial work remain uncertain.

Checkpoint comparisons require exact seeded initial tensors, midpoint and
endpoint weights, AdamW states, consumed evidence and recipe. Independent
trajectory comparisons ignore only the two explicitly named timing values in
the trainer's `timing` field; their consistency with step journals is checked.
Immediate serialized reload includes timing and must match the complete state.

At each saved midpoint/endpoint, workers perform actual BOS-only observation
forwards and greedy free-reply generation on the three protected banks. They
compare action logits, first-reply logits, generated tokens and derived metrics
exactly. Outputs retain their producing weight hash, cursor and bank identities.
Labels are consulted only after prediction. The resumed midpoint is evaluated
again before any new update. Across the six cases, this adds 252 encoder episode
passes and 2,520 free-reply turns, separate from training exposure. No numerical
accuracy gate or learning claim is attached to these outputs.

Strict CUDA settings must be configured before initialization in each worker.
Each worker records setup/restore, index construction, evaluation, step and wall
cost plus peak allocated/reserved memory. Materialization is inside step time;
setup, restore, evaluation and writes are inside worker wall. These nested times
must not be added twice or called dedicated GPU hours. Process startup lies
outside function wall measurements. Evaluation intents separately preserve
failed or incomplete forward/generation work.

Public APIs are `prepare(directory)`, `worker(directory, name)`,
`verify(directory)` and `load_proof(directory, config)`. The latter accepts only
a complete authenticated `passed` CUDA receipt matching all six cases and the
declared configuration. Programmatic tiny CPU fixtures can shorten the workload
and use a smaller four-block model; they return `passed_cpu_engineering_only`
and can never satisfy the CUDA proof gate. No live study or frozen source is
modified by this component.

The first `load_proof` call in a process performs full canonical and checkpoint
authentication. A single-entry in-memory JSON cache can serve later calls for
the same directory, configuration and proof hash only after rechecking every
artifact and current source hash before and after the call. Returned data are
isolated copies; no tensor state or persistent cache is trusted. First-load
authentication time and subsequent file/source recheck time are different
setup costs and must be measured as such.
