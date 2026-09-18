# Causal entity retrieval: prospective paired pilot

Fixed before the learning comparison. The preceding repeated-reading and shared
state pilots failed their screens. The subsequent fixed-buffer diagnostic showed
perfect native training fitting at a smaller learning rate but poor transfer.
This pilot tests learned retrieval of separate entity representations and their
availability to native decisions and replies, as specified in the
[implementation design](CAUSAL_ENTITY_RETRIEVAL_DESIGN.md). It tests the whole
architecture intervention, including additional parameters and compute.

## Learners

The control is the existing SharedStateStudent. The candidate constructs the
same base model first, then adds causal entity retrieval and native fusion.
Initialize both with seed 852504001 and require exact equality of every common
parameter. Report all candidate-only parameters; this is not parameter-matched.
Both use width 192, 4 layers, 4 heads, feedforward 768, 1024 positions, maximum
12 turns, input 128 bytes and output 32 bytes. Keep AdamW learning rate 0.003,
gradient clipping 1, and the established strict local FP32 execution profile.

For each turn, candidate entity queries are a learned projection of each fixed
32-dimensional alias embedding plus a learned projection of the causal EOS
state. Multihead attention uses projected byte keys and values once per forward;
only positions at or before that turn's EOS are visible. Input validation and
masking exclude padding and future bytes. The projected read is residual-added
to the entity query and normalized. All 12 aliases are queried at every prefix.
A complete alias permutation may be used only to test equivariance/invariance.

A learned projection of the native EOS state queries projected entity keys and
values. Residual-add the projected read to that EOS state and normalize. Both
the action head and free-reply context derive from this same fused state. The
raw causal encoder states and tied observation-language objective stay on their
original path. No zero gate, detached retrieval or privileged state routing is
used. The existing shared 107-class decoder receives the entity vector and its
alias embedding. Training consumes vectors already computed by native forward,
without performing retrieval twice.

Both arms use the original action/reply/observation objective plus 0.3 times the
same prefix-state objective. Targets derive from visible English after each
turn, with questions not asserting values and unknown-copy semantics preserved.
Average known/unknown state group losses per example and turn, then use equal
family weighting and one update after the three families. No parser, family
label, state target, tutor response or external symbolic answer enters inference.
The auxiliary 107-class decoder is not called by policy evaluation; learned
entity retrieval and fusion are native candidate inference components.

## Lessons and work

Reuse all 648 immutable original-layout training bundles, 32 episodes per family
per update, for 62,208 training episode exposures per arm. Alternate which arm
updates first. Derive state targets once per bundle and share them. Stop at
648 updates per arm (1,296 total), regardless of intermediate scores. Evaluate
at 0/216/432/648; save complete model/AdamW checkpoints at nonzero endpoints.
The main execution allowance is 3,600 seconds including its setup, evaluation
and saving. Check deadlines between operations, preserve partial work on failure,
and do not retry, extend or change settings automatically.

Prepare a new 720-episode development bank with seed 852502001, excluding the
verified recorded-study union of 1,702,440 transcripts. This adds the previous
shared-state pilot's 720 development transcripts to its prior exclusion union.
Use the same 108 actual training-fit and 720 retention episodes, explicitly
labeled reused. Unit-test fixtures and semantic equivalents are not claimed to
be excluded by this recorded-study transcript census. The finite grammar and
shared vocabulary limit any conclusion about wider English.

Use native actions and freely generated replies, starting at BOS with the tutor
disconnected. Report all 24 bank endpoints and retain all 12,384 evaluation episode
records, with joint opposite-answer pairs, separate action/reply pairs, every
family's known/unknown and nonanchor questions, unsupported ASK, agreement and
retention. An aggregate unknown-answer gain must not hide worse factual answers.
Do not choose an earlier checkpoint or rehearse the fit sample separately.

## Decision and validation

At the fixed 648-update endpoint, apply the existing shared_state_screen exactly:
require at least 5 percentage points of overall fresh joint-pair improvement,
strict joint improvement in each of color/count/switch, and strictly better total
joint training-fit performance. No known/unknown action or reply slice in any
family may lose more than 5 points; overall joint retention may not lose more
than 5 points. Apply thresholds to exact counts. A positive single-seed result
supports replication and a matched-compute check, not checkpoint promotion.

Before launch, use focused small CPU tests for causal prefixes, padding,
cross-row isolation, common initialization, alias permutation, native slot-order
invariance, cached retrieval reuse, and gradients from auxiliary/native losses.
Check the new baseline kernel against the original state-teaching kernel and run
one production-shape update per arm on an existing cached bundle. Keep every
validation invocation and its physical work visible. These correctness checks
are not learning evidence.

Pin the model, kernel, runner, data overlay, pure report and screen, all inputs
and validation receipts. Recount saved raw predictions after completion without
another neural run. Report actual parameter counts, preparation, training,
evaluation and checkpoint costs, throughput, and peak allocated/reserved GPU
memory. Common target preparation is a shared cost, not an incremental candidate
teaching cost. Include delegated archive reads in the data preparation ledger.

No LLM calls occur in this comparison. External tutoring, learned self-direction,
persistent cross-session memory, reliable overwrites and broad English remain
separate open requirements. The larger home-learning objective stays active.
