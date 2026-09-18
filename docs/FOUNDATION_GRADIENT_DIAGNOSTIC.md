# Fixed gradient inspection after the shared-feature diagnostic

Recorded before gradient measurements, 2026-09-17. The completed shared-feature
diagnostic leaves the shared-learning explanation unresolved: trained probe
scores have no consistent all-family advantage over initialization, every probe
is unconverged under its fixed residual criterion, and original reply errors
already occur at the first byte. Execute the conditional inspection specified
in `FOUNDATION_SHARED_LEARNING_DIAGNOSTIC.md`; do not retune those probes.

Use only the historical curriculum and mixed 1,536-update states and the sealed
fitting dataset from `runs/foundation-shared-diagnostic-local/attempt-002`.
Authenticate those exact files and the current sources/runtime afresh. Select
the first four complete pair slots in every one of its 63 cells. No new examples,
evaluation examples, scoring inputs or selected weak cells enter this inspection.

For each endpoint, group the three families by their common depth/operator/turn
suffix, in canonical sorted order. There are 21 groups of 24 episodes: three
family forwards of eight episodes each. Across both endpoints this is exactly
**42 groups, 126 family forwards, 1,008 episode forwards, and 168 component
gradient evaluations**. Reuse each group's forward graph for four separate
`autograd.grad` calls. No fifth total-objective backward, clipping, optimizer,
gradient step, new learner continuation or model promotion occurs.

The four unweighted components are equal-present-query-class cross entropy,
ACK cross entropy, reply loss and observation-byte loss. Reproduce the existing
`sequence_objective` arithmetic: reply and observation losses pool token losses
across the family batch separately at each turn position and then average turn
positions. They are not a global token mean or a mean of individual episode-turn
means. Compute each family separately and average the three family components
before differentiation. Apply the existing weights **1, 0.25, 0.1, 0.1** when
reporting the actual objective's component magnitudes.

Report every group's four raw/weighted norms, six pairwise cosines, and the
query-versus-weighted-auxiliary-sum cosine. Measure shared attention-block
parameters and the tied token/input-readout embedding separately, with one entry
per parameter identity. Zero-norm cosines are null. Preserve caller `.grad`,
parameter flags, modes and exact weight bytes; interruption records completed
work and retains its original exception type. Report any incomplete group rather
than inventing zero work.

The local `cuda:0` run has a **900-second nonpreemptive wall allowance**, checked
before each endpoint operation. A durable intent precedes each operation.
Record complete phase wall/CPU costs, actual forward/backward work and CUDA peak
memory. This allowance is additional to the earlier 70.032-second feature run.
No automatic retry or dataset/weight selection occurs after viewing gradients.

These are unpreconditioned local gradients before clipping. Their alignment
does not reveal AdamW's moment-preconditioned update direction, establish a
unique cause of weak learning or prove that deleting/reweighting a loss helps.
Use the complete results to choose at most one shared prospective training
comparison, preserving action/reply learning, uncertainty and retention across
all families. Every candidate needs measured benefit; capability alone is not
an automatic reason to adopt a change.
