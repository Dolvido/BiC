# Next shared learning intervention: causal entity retrieval

Selected after the completed shared-state and fixed-buffer/transfer experiments.
This is an implementation design, not a trained capability or a frozen launch.
The broad home-learning goal remains active. Local implementation, verification
and the paired learning run are the next work; no additional user approval is needed.

## Evidence and question

The original shared-state head inferred every entity through MLP(EOS, alias).
Its broad-data pilot failed the prospective screen. Repeating encoder reads also
failed. A smaller learning rate let the same learner fit every one of 516 query
turns on 108 rehearsed episodes, but only 10/360 development pairs transferred,
versus 23/360 at the faster rate. The reused-development check is described in
[the completed results](SHARED_STATE_FIT_RESULTS.md). These observations support
investigating a shared representation mechanism, without uniquely identifying
the cause of failed generalization or establishing a learning-rate winner.

The next question is whether explicit learned retrieval for entities, available
to both the teaching readout and the native answer path, improves transfer across
color, quantity and switch reasoning. This targets a common operation across
subjects. No individual failing example or family receives a dedicated repair.

## Candidate

Keep the raw-byte causal encoder and observation objective. At every turn, each
of the twelve fixed alias queries attends learned keys/values from the observed
byte prefix. Mask all positions beyond that turn's EOS and all padding. Earlier
byte states and queries must themselves be causal. Use one shared state decoder
on the resulting entity vectors, retaining the existing prefix targets and the
0.3 auxiliary loss. Do not use a per-family mask or head.

The native EOS representation forms a learned query over these entity vectors.
Fuse the retrieved vector into the shared action/reply representation through a
learned residual projection and normalization, so both native outputs can consult
the same learned representation. Preserve freely generated replies at evaluation.
The parser, oracle, entity-state labels, family identifier and hard-coded query
routing never enter native inference. All alias queries are made at every prefix;
future occurrence cannot select which entities are represented.

Project byte keys/values once per forward and avoid copying them for every turn
and alias. Report actual model parameters and measured memory/cost. The package
adds parameters and computation; a positive result would support the complete
retrieval-plus-fusion intervention, not parameter-matched causal isolation.

This is causal entity retrieval, not a claim of persistent cross-session memory,
learned symbolic execution, novel-name competence or robust revision. The current
curriculum lacks interfering overwrites. Expand those tests separately if the
shared mechanism becomes useful. Twelve aliases describe this experiment's world,
not a proposed permanent limit on BiC's concepts.

## Planned comparison

Use the existing SharedStateStudent with auxiliary weight 0.3 as the control and
the retrieval/fusion learner with the same supervision as the candidate. Fix one
common learning rate at 0.003 to retain the preceding broad-data control setting;
the tiny-buffer slow-rate result did not justify adopting it as a general winner.
Use a fresh paired initialization seed 852504001, width 192 and the existing strict
FP32 local runtime. Retain identical common encoder/answer parameters where the
architecture permits, document extra parameter initialization, and alternate arm
execution order. The final launch must freeze this configuration before scores.

Reuse the 648 cached training bundles, 62,208 episodes per arm, with evaluation
at updates 0/216/432/648. Do not add a separate tiny-buffer calibration or choose
an intermediate checkpoint. Prepare a fresh 720-episode development bank with
seed 852502001 and authenticated exclusion of all prior observed transcripts,
including the previous shared-state development bank. Explicitly label the 108
actual training-fit and 720 retention episodes as reused. Training-fit rows are
measurements, not a separately rehearsed curriculum for this comparison.

Report every family's native action/reply/joint pair scores, known and unknown
queries, unsupported abstention, nonanchor questions and retention. Preserve the
fixed final-step screen: at least 5 percentage points of overall fresh joint-pair
improvement, strict joint improvement in every family, better training-fit joint
performance, and no greater than 5-point known/unknown head regression in any
family or overall joint-retention regression. Even a pass requires replication;
it does not promote a checkpoint or establish broad English understanding.

Before launch, test prefix causality, padding, alias-query permutation, native
slot-order invariance and gradients reaching the shared encoder/retriever. Run
one bounded production-shape update check. Reuse validated target and native
scoring code. Record setup, data preparation, training, evaluation, checkpoints,
parameter counts and peak GPU allocation/reservation. Do not silently extend or
retry based on observed learning scores. Keep the LLM external; this comparison
makes no tutor calls and cannot establish richer LLM teaching benefit.
