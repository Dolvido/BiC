# Verified chapter compiler: bounded implementation

The additive [compiler](../experiments/verified_tutor_curriculum.py) implements the
recipe boundary proposed in [the tutor design](VERIFIED_TUTOR_CURRICULUM_DESIGN.md).
It provides no teacher transport, learner, checkpoint restoration, filesystem
publication or benefit claim. The caller authenticates admitted parent inventory,
protected transcripts and the contract, and persists each returned image once.

`compile_curriculum(recipe, admitted_parent_inventory=..., protected_transcripts=...,
coverage_contract=...)` returns `manifest`, `images`, `provenance`, and `receipt`.
Each image contains only `bundle`, `expected_evidence`, and
`expected_evidence_sha256`, compatible with PreparedLayoutOwner. Global bundle
IDs are the contract's `start_cursor` plus slot index. Original plans and nested
canonical identities are preserved, with newly realized lessons identified by
their actual new canonical recipes. They are not relabeled as old plan outputs.

Teacher recipes contain exactly a schema and positional chapters, each with
`motif` and `realization`. No family selection, answer labels, code or explanation
text is accepted. Contract and recipe validators, the procedural default and
authoring JSON schema/catalogue helpers import only the standard library.

## Fixed and chosen components

The caller fixes one to six balanced chapters. Each contains the same positive
coverage of all 18 depth/length cells, with all three families in every update.
Update order, microbatch size, global IDs, seed, literal-replay slots, earlier-depth
rehearsal floors, per-cell replay floors, protection digest and exposure/time/work
ceilings are fixed before authoring. No teacher allocation can remove a subject,
cell or reserved replay slot. Up to 108 bundles and microbatch32 are supported.

The inventory contains unmodified admitted foundation plans and shared parent
descriptors: plan digest, original bundle/pair index, and three family-specific
canonical pair digests. Named motif pools are explicit caller-frozen lists of
these descriptors. The compiler verifies actual selected parents by regeneration,
not a pool's prose name. Pool semantics and overlaps must be described from their
actual canonical procedures; a relative copy-heavy pool is not a new grammar.
Every menu choice must contain enough distinct parents for every eligible slot.

For teaching slots, `independent` changes naming and value realization seeds;
`rename` changes names while retaining original value seed; `revalue` changes
values while retaining names. Purpose-separated seeds depend on the fixed caller
seed, slot and pair position, never family or performance. Each family uses the
same selected procedure. Literal replay ignores the teacher choice and preserves
the original admitted parent. No stochastic retry searches for a passing lesson.

All supplied protected text hashes are forbidden, including during replay. The
caller must distinguish allowed prior training from held-out/protected rows.
Fresh realized transcripts must also be distinct within the compilation; literal
replay may repeat across updates and is counted as repeated exposure. Duplicate
parents within one microbatch are rejected. If an otherwise valid menu choice
collides with protection or exceeds a limit, compilation fails and the caller
receives a rejection receipt, never a partial accepted curriculum.

## Verification and limits

Every selected parent, new realization and layout is canonically validated.
Typed and English truth checks compare all action targets and canonical replies;
known/unknown questions and opposite paired answers are required. Actual family,
depth, turns, train admission, operators, target classes and byte/token exposures
are recorded. Token ceilings are the same fixed limits for matched arms; different
legal realizations need not have equal actual byte counts. Neither parsed state
nor provenance becomes a model input.

The work receipt counts every `generate_pair` invocation, including calls through
validators, with attempts/completions/returned rows; those rows are reconstructed
work, not distinct lessons. It separately records explicit compiler oracle calls
and the layout validator's counters. Cached generator-internal oracle checks are
not misrepresented as explicit calls. Exceptions, including interrupts, retain
the partial receipt and restore the temporary generator hook. A source recheck
precedes successful return. Caller publication and later CPU packing costs are
outside compilation and must be reported separately.

The focused unittest declaration is six methods, one synthetic admitted-plan
metadata fixture and only 18 selected original parent descriptors across three
families. Five successful 18-bundle micro2 compilations exercise determinism,
all three realization modes and literal replay. The whole invocation is bounded
by 3,000 generator calls / 6,000 returned rows, 18 exact existing-evidence
comparisons, six CPU pack calls / 12 packed episodes, and zero model construction,
forward, backward, optimizer, CUDA or teacher work. Malformed menus/coverage,
protected text, altered parent pins, actual token limits and interruption fail
closed. These fixtures establish compiler compatibility and rejection behavior,
not historical admission or learning benefit. No production curriculum or teacher
request is executed by this proof.
