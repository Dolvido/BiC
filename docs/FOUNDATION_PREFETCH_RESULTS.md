# Equivalence of overlapped lesson preparation

On 2026-09-17, the additive prefetched trainer passed six bounded CPU checks.
The purpose is to overlap lesson preparation with learning while preserving
the existing learner. A subsequent local CUDA proof also passed, as recorded
below. Throughput improvement and learning benefit remain unmeasured.

For each of the baseline and balanced-reply objectives, the check compared
66 synchronous updates with 33 prefetched updates, strict snapshot restoration,
and another 33 prefetched updates. Every matched step had identical weights,
complete AdamW state, cursor, canonical lesson evidence and all four loss fields.
The schedule covered all six depths and three sequence lengths. One additional
restored update matched after a deliberate partial-family interruption, giving
133 exact step comparisons across the two objectives.

The remaining checks rejected incompatible checkpoints and lesson identities,
prevented continuation after failed work, discarded two queued lessons when
saving, and refused an expired allowance after the first transfer but before
any forward. All nine preparation invocations finished with their workers
joined; queues never entered the saved learner. These were same-process CPU
checks using a small model and one fixture, with no evaluation or capability claim.

The successful run performed exactly 265 optimizer updates, 796 native family
forwards and backwards, 1,592 episode forwards and 11 model/template constructions.
There were 797 attempted family forwards: the deliberate interruption stopped
one attempt before entering the native model. Separate from initial admission
and index construction, lesson preparation returned 269 bundles / 1,614 episodes
and completed 807 family packing calls. Preparation counts are not learner
exposures. Wall time was 129.744 seconds and process CPU time 141.359 seconds;
the formal GPU study was running concurrently, so this is not an isolated speed
measurement. CUDA remained uninitialized and all 105 pinned files matched before
and after the run.

Evidence: successful receipt (archive reference: `../runs/foundation-prefetched-training-validation-local/attempt-002/report.json`),
SHA256 `2edb9d74f01dd69dd72709e2a3622ffe94b20ecb1ec6aaa06464ef9e60981343`;
frozen launch (archive reference: `../runs/foundation-prefetched-training-validation-local/attempt-002/launch.json`),
SHA256 `222b4cb4637eb7e4a76a2b016ceef49dcda336c6126a0f46e0f66f34c5107f59`.
The source snapshots preserve the exact trainer, test and prospective contract.

The first failed attempt (archive reference: `../runs/foundation-prefetched-training-validation-local/README.md`)
remains part of the cost: 66 CPU updates and 10.057 wall seconds before a
test-only dictionary-container mismatch. Correcting the detached test-state
copy did not change the trainer or weaken numerical comparison. Across both
attempts there were 331 physical CPU optimizer updates; neither run trained
an adopted learner or changed the active formal experiment.

Only a subsequent matched, complete-cost throughput comparison can justify
using this path in longer training. The
[completed learning comparison](FOUNDATION_OBJECTIVE_RESULTS.md) independently
determines the next change to BiC's shared learning mechanism.

The [prospective CUDA contract](FOUNDATION_PREFETCH_CUDA_PROTOCOL.md) now fixes
396 updates covering both objectives, every curriculum depth/length, a saved
continuation and a fresh repeat. Its separate parent-termination gate passed
seven pure checks, including a read-only query confirming the validation process
was live on Windows. No Torch module was imported. The
gate receipt (archive reference: `../runs/foundation-prefetch-gate-validation-local/attempt-001/report.json`)
records 0.128 wall seconds and zero neural work. This verifies the tested launch
boundary; it did not itself execute the CUDA trajectory proof.

The GPU probe and gated launcher are now implemented and source-reviewed.
Five additional small CPU checks passed for the exact comparison and journal
helpers in 1.391 wall seconds. Guards recorded zero model, forward, backward,
optimizer or actual-probe calls, and CUDA remained uninitialized. The real
launcher also refused the active parent as intended in 0.047 seconds, creating
no GPU proof directory. The execution handoff (archive reference: `../runs/foundation-prefetched-runtime-validation-local/README.md`)
records source pins, receipts and the completed attempts.

## Completed local CUDA identity proof

After the formal study completed and its learning results were analyzed,
attempt 001 passed the fresh parent-process gate but failed while preserving
source files: a Windows temporary filename exceeded the normal path limit.
The failure occurred before runtime setup, data construction or neural work.
Its enclosing cost was 5.235 wall seconds / 2.328125 process CPU seconds, with
zero optimizer updates and no unknown physical work. The failed evidence remains
in `runs/foundation-prefetched-runtime-validation-local/attempt-001`.

A path-only wrapper in the new probe preserved the existing exclusive
publication behavior and used Windows extended paths. Four actual local checks
reproduced the old error, verified source/JSON/scalar-archive/journal paths up to
409 characters, preserved relative artifact names and refused overwrites.
They used 1.75 wall seconds / 1.234375 CPU seconds, with zero model/data/optimizer
work and CUDA uninitialized. The
repair receipt (archive reference: `../runs/foundation-prefetched-path-validation-local/attempt-001/report.json`)
has SHA256 `0785edd5ab1c61e120f5a89fd12533ae5d73af56ed828c9c6297c3a1a9a6576b`.
Historical trainer, owner, objective helper and protocol sources were unchanged.

The separately launched attempt 002 passed on September 17 at 13:53:56 UTC:

- 396 physical optimizer updates, 1,188 completed family forwards/backwards,
  and 38,016 learner episode exposures across both objectives.
- All six depths and three lengths in the 66-bundle fixture; synchronous
  reference, 33 + strict saved restore + 33 prepared updates, and a fresh
  66-update repeat for each objective.
- 272 matching state comparisons, using typed byte digests for weights,
  complete optimizer state, cursor and lesson evidence; exact aggregate/family
  losses and direct full seven-field equality at saved midpoint boundaries.
- Eight fresh models plus two restore templates. All six preparation owners
  joined; no unknown work, deadline overrun, evaluation or automatic promotion.
- Enclosing launcher time 455.875 wall seconds / 478.546875 process CPU seconds.
  The per-step copy/hash instrumentation makes these unsuitable speed results.

Proof receipt (archive reference: `../runs/foundation-prefetched-runtime-validation-local/attempt-002/proof/probe.json`),
SHA256 `8719ca91494c45d21b93b55186408e3e6b8c1e0e9b83576cbd399e9341e9d249`;
launcher receipt (archive reference: `../runs/foundation-prefetched-runtime-validation-local/attempt-002/launcher-result.json`),
SHA256 `a1a8b9ddeba788aef276f786503c280d340770e11ee60285e30a65611437ef13`.
An independent verification (archive reference: `../runs/foundation-prefetched-runtime-validation-local/attempt-002/verification.json`)
rehashed all 97 proof artifacts and 118 launch inputs, checked all recorded
comparisons, and validated 1,979 chained journal events with 853 completed
operations and no pending operations. Its SHA256 is
`d5d4d3a15cdf0c9b0c26cf774d777329ce2abf5ccafd7fc4d47a760f3e8d165b`.
It performed no neural work or tensor deserialization.

This proves the tested execution equivalence, not faster or better learning.
Before adoption, measure full setup, real protection-inventory handling,
training, checkpoints and shutdown costs against the synchronous path.
