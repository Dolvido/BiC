# Retention across curriculum cycles

`LifetimeRetention` preserves the best observed paired-action, generated-reply,
known-answer and unsupported-ASK counts for each fixed development/retention
bank. Higher is better for the first three measures; lower is better for
unsupported ASK. An improved field cannot hide another field's regression.
Equal counts retain the earlier producing state. Undefined denominators stay
undefined and do not create an improvement or regression.

The public input is an authenticated `CompletedParent`, after the finite loop's
final evaluation. The first record requires original cycle one. Advancing requires
the next ordinal, the exact predecessor identity, continuous lifetime updates and
identical bank roles, names and content identities. Local evaluation updates are
converted to lifetime coordinates using the completed parent's inherited count.
Every reference also retains its cycle, provider, weights and parent identities.

The fold considers the best observations anywhere inside each completed cycle,
not just its final scores. It carries earlier best references into later cycles
and compares each current endpoint against those retained bests. It keeps one
best and one endpoint per bank/metric, plus current alarms and digest links;
there is no recursively growing list of past responses or learner tensors.
This bounded metric record does not make cumulative transcript exclusions
constant-size: exact exclusions still grow with distinct lessons.

The object is immutable and returns detached JSON snapshots. Pinned restore
authenticates a detached snapshot before checking its sources, lineage endpoint,
bank identities, count/rate consistency, producer ranges and alarms against the
current parent. Its expected snapshot digest must come from a trusted caller or
an authenticated outer envelope. A self-declared hash in untrusted metadata does
not establish historical truth. Old neural predictions are not recomputed.

Ten pure count and lineage checks (archive reference: `../runs/foundation-lifetime-retention-validation-local/attempt-002/report.json`)
passed, including a 100-cycle synthetic fold, mixed gains/losses, within-cycle
best retention, bank changes, missing denominators, invalid lineage, corrupted
records, detached state and caller mutation during restore. Independent review
found a hash-before-copy race; the implementation now copies before authenticating,
and the added regression passes. These tests perform no model forward or optimizer
work. Actual completed-parent integration is separately exercised by the repeated
cycle validation; synthetic fixtures alone do not establish execution authenticity.
The 20-check repeated-cycle run (archive reference: `../runs/foundation-cycle-repeated-validation-local/attempt-001/report.json`)
subsequently passed with both architectures through three completed cycles. It
preserved seven flat and three hierarchical alarms at the final tiny-fixture
endpoint, including through pinned record restoration. These are engineering
fixture observations, not comparative learning outcomes.

These alarms are observations, not mastery thresholds, promotion decisions,
adaptive lesson choices or proof of useful learning. The durable cycle owner must
commit retention together with cycle transitions. See the
[owner design](FOUNDATION_CYCLE_OWNER_DESIGN.md) for atomic publication and restart.
