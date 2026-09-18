# Explicit completed-parent restart capsule

`export_parent_capsule(parent, path)` accepts only the exact `CompletedParent`
captured from an actually completed, validated foundation loop. It writes one
new file exclusively and returns its SHA-256, parent identity hash, byte count,
and measured cost. Existing destinations cannot be overwritten. Preserve the
returned digest in trusted caller state.

`load_parent_capsule(path, expected_sha256=..., device="cpu", evaluation_banks=...)`
authenticates one immutable byte read against that externally supplied pin before
loading its single learner image with `torch.load(..., weights_only=True)`.
It returns an exact immutable `CompletedParent` usable by the next cycle.
The digest must come from the trusted export, not an untrusted sidecar located
beside a purported capsule. Pinning arbitrary invented bytes does not establish
actual historical execution. Capsule metadata does not certify itself.

The fixed-member, uncompressed ZIP is never extracted to disk. Its source,
identity, metadata, exclusion lists, and one current learner image are bound by
the full-file pin and internal member hashes. Duplicate or extra members,
compression, duplicate JSON fields, changed sources/runtime, malformed lineage,
inconsistent counters, omitted exclusions, or changed bank identities fail.
The default size limit is 2 GiB. Existing v1 parent tokens are not migrated.

The loader builds exactly one current architecture template on CPU, with no
forward pass or optimizer update, to check parameter names, shapes, dtypes,
tied weights, complete AdamW moments, parameter ordering, and lifetime step
counters. It canonicalizes the caller's same development/retention banks and
checks final and retained-reference metric counts, progress, producer hashes,
and local update coordinates. Current training transcripts must match the
captured index count/hash and belong to the preserved exclusion union.

Historical phase completion, admitted plan execution, the full loop-envelope
digest, and past gradient/prediction provenance are inherited from the trusted
captured export. The compact capsule cannot independently reconstruct these
facts from metadata alone. Loading does not regenerate old plans, construct old
indexes, load ancestral learners, repeat lessons, recompute old scores, promote
a model, or change source/runtime identities.

One current learner and digest lineage replace recursive ancestor tensor loads.
Exact transcript exclusions still grow with unique historical data; total
capsule size is therefore not constant. The current cycle can continue to use
its authenticated plan index normally after restart.

The loader records observational cost in `load_parent_capsule.last_report`:
wall/CPU seconds, one current CPU-template duration, bytes, and zero ancestor
models/index constructions. This process-local latest-call report is not a trust
input and is not a concurrent task ledger. Export costs are returned directly.

`tests/test_foundation_parent_capsule.py` uses synthetic containers and tiny
tensors solely to exercise codec and structural rejection paths. These fixtures
are explicitly not evidence of completed training. Actual token export/load and
subsequent cycle continuation are validated in the repeated-cycle integration
fixture, using its existing completed loops and no additional training job.
All 15 pure codec checks passed. The subsequent
20-check repeated-cycle suite (archive reference: `../runs/foundation-cycle-repeated-validation-local/attempt-001/report.json`)
also passed all six actual parent roundtrips and continued training from the
loaded tokens. Observed loads took 0.719–0.937 seconds on these tiny CPU fixtures,
with one current model template, no ancestor models and no plan indexes. This
is a measured import cost, not a projection of full-size training throughput.
