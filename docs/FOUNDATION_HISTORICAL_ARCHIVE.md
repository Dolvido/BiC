# Historical foundation archive verification

`experiments.foundation_historical_archive` is an additive, standard-library-only
reader for completed historical evidence. It is independent of the active formal
variant study. No existing loader, gate, protocol, evidence, or sealed source is
changed by adding it.

The completed pilot summary is pinned by its caller to SHA-256
`188b0d026c4debce596f3cca5a1b64f6f5f0e0e3992dd4174904a33b6c3e87c2`.
It binds 424 absolute input paths and 79 canonical relative source paths. Both
each original live source and its `runs/foundation-study-local-v2/source/` copy
already occur in that input manifest with the same expected bytes.

The verifier authenticates the summary before using its contents. The caller
also declares the expected input/source counts, schemas, version, arms, and
checkpoint boundaries through `ExpectedEvidence`. Small test fixtures use
explicit smaller expectations; metadata does not select its own expected count
or identity. The historical CLI requires explicit summary hash and counts, and
uses the historical schema, version, arms, and checkpoint expectations.

The protocol must itself be a summary-bound, unchanged input. Its source map
must equal the summary integrity source map exactly. Its contract must match
the summary and caller expectations. The preparation, verification, evaluation,
and both arm completion records must be summary-bound, completed, and linked to
the original protocol (and evaluation to the original verification). This reads
their declared completion fields; it does not redo their original work.

Only the exact original source input at `repository_root/<declared source>` may
be remapped to `<historical directory>/source/<declared source>`. Both input
entries must already exist in the pinned manifest with the source map's digest.
There is no caller-supplied arbitrary remapping map or fallback search. Every
other input remains bound to its original absolute path and digest. Changed or
deleted live source files can therefore coexist with valid historical bytes.
Current source equivalence is not asserted.

Relative names reject traversal, empty segments, alternate separators, drive
names, alternate data streams, trailing dots/spaces, and reserved Windows names.
Input paths use the native platform's canonical absolute spelling. Duplicate
JSON members, case aliases, symlinks, junctions, remapping chains, and hardlink
aliases are rejected. The only allowed shared physical target is the intentional
source-to-archive remapping to the same already-declared archive path.

An example from the repository root, using its Python environment:

```powershell
.venv\Scripts\python.exe -B -m experiments.foundation_historical_archive `
  runs/foundation-study-local-v2 `
  --repository-root . `
  --summary-sha256 188b0d026c4debce596f3cca5a1b64f6f5f0e0e3992dd4174904a33b6c3e87c2 `
  --input-count 424 --source-count 79 `
  --receipt runs/foundation-historical-archive-validation-local/new-check.json
```

The receipt destination must not already exist and must be outside the historical
evidence directory. These checks precede evidence verification and writes. The verifier itself writes
nothing; only the CLI writes the new receipt. The receipt retains the original
input and source maps and lists every original source path, archive target, and
expected digest. It records 424 logical bindings, 79 remappings, 345 unchanged
input bindings, and 345 distinct physical input files for the historical pilot.
Hashes are computed by streaming bytes, with no checkpoint deserialization,
archived-code imports, tensor operations, model work, or GPU use. Wall time, CPU
time, and streamed input bytes are recorded; summary and verifier-source reads
are excluded from the input-byte count. Its own source SHA-256 is bound in the
receipt and checked again before success.

This is point-in-time authentication of the caller's selected historical
evidence. The summary is rehashed and file identity/size/time metadata rechecked
before success. This is not an atomic filesystem snapshot or protection against
an adversarial concurrent writer. Keep evidence quiescent during checks. It
does not replay lessons, regenerate banks, recompute metrics, redo training,
prove model validity, grant automatic promotion, or migrate evidence. Absolute
non-source paths still prevent relocation to another machine/root. The current
formal study continues to use its existing verification and source closure;
adopting this reader in a future protocol requires a separate explicit change.

Tests use only tiny temporary files and cover source evolution, archive and
ordinary-input tampering, unbound protocols/archive copies, source-map mismatch,
protocol/completion identity, count expectations, path/JSON aliases, missing or
incomplete evidence, symlink escape, and hardlinks. Validation receipts and exact
test output are stored under `runs/foundation-historical-archive-validation-local/`.
