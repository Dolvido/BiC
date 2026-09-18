# Fixed CUDA equivalence proof for prepared lessons

Declared September 17, 2026 after the six CPU integration checks passed and
before the objective study's main results were inspected. This checks the
execution of the new preparation path. It neither selects a learning objective
nor measures capability, throughput, tutor independence or curriculum benefit.

## Prerequisites and execution boundary

The active local objective comparison must finish successfully first. A launcher
must authenticate the completed dispatcher and main-stage receipts, then confirm
through the operating system that their recorded processes and the main launcher
are no longer live. Unknown process status is a refusal. The launcher must not
initialize CUDA before this gate. No automatic waiting, retry, continuation or
extension is included; an interrupted proof remains a charged failed attempt.

The prerequisite CPU receipt is
`runs/foundation-prefetched-training-validation-local/attempt-002/report.json`,
SHA256 `2edb9d74f01dd69dd72709e2a3622ffe94b20ecb1ec6aaa06464ef9e60981343`.
It passed six checks with 265 updates; the preceding failed test attempt's
66 updates remain recorded separately. A new caller-pinned source manifest must
include the exact trainer, preparation owner, probe, this contract and their
dependencies. Preserve source snapshots and immutable operation records.

Use only the local RTX 5080 at `cuda:0`, strict FP32, deterministic operations,
TF32 disabled and one CPU intra-op/inter-op thread. Configure the existing strict
profile before CUDA initialization and verify it throughout. No CPU fallback,
external tutor, inference scoring or historical learner load is permitted.

## Fixed workload

Construct one fresh plan with seed **831910001**, ordering seed **831910002**,
stage updates **10**, final updates **6**, rehearsal interval **2** and microbatch
**32 episodes per family**. Protect the two transcripts from original bundle 0,
color family, pair 0, then admit/repair the plan and build one actual authenticated
index. The complete 66-bundle curriculum must visit each of the six depths and
three sequence lengths. Initial protection generation, admission repair and the
index's admission reconstruction and scan are distinct preparation costs.

Use model seed **831910799**, width **192**, four layers, four heads, feedforward
**768**, maximum 12 turns and the current 128-input/32-reply/1,024-position limits:
2,221,738 parameters. Test both fixed objectives at the already selected
calibration rates: baseline **0.003**, balanced reply **0.001**. These rates are
execution cases, not a new rate comparison or a choice based on main results.

For each objective, execute exactly these paths:

1. Original synchronous training for 66 updates.
2. Prepared training for 33 updates, an actual CPU checkpoint file save/load,
   strict restore into a fresh trainer, and 33 further prepared updates.
3. A fresh prepared repeat for all 66 updates.

The exact successful workload is **396 physical optimizer updates**,
**1,188 family forwards and backwards**, **38,016 episode forwards**, and
**10 model/template constructions** (including the two restore templates).
After initial admission/index setup, this requires 396 returned canonical
bundles, 1,188 family packing calls and 38,016 packed episode passes. No extra
failure injection or evaluation is included. Setup and failed/unknown work
must be recorded separately rather than hidden in these successful counts.

## What must agree

At every matched step, compare weights, complete AdamW state, cursor and
canonical consumed evidence through typed SHA256 byte digests of detached CPU
copies. Include tensor dtype, shape, layout, strides, offset and gradient flag;
do not silently cast or quantize. This avoids retaining approximately 1.7 GiB
of reference states per objective. Record the expected and actual component
digests. Compare all four loss scalars and ordered family evidence exactly.

Cross-implementation comparisons exclude only their deliberately different
execution schema, preparation/source recipe and timing. Check the common model,
objective, rate, plan/index and seeded initialization separately. A loaded
midpoint and its strict restored snapshot must match directly across all seven
saved fields, including their own complete recipe and timing. Verify final
consumed evidence against the real index's full 66-bundle prefix.

Do not call the prepared trainer's snapshot after every update: snapshots drain
the queue. Use detached copies for per-step comparison. Actual snapshots occur
only at declared boundaries; preparation windows stop exactly at 33 or 66.
Every invocation must have a retained final preparation report. The saved
learner cannot contain queues, live leases or a carried-over time allowance.

## Costs, failure and interpretation

The complete invocation has one **900-second** allowance. Check it before each
new operation and immediately before the first forward. After the preparation
constructor returns, clamp both the trainer's caller deadline and the producer's
internal deadline to that same absolute deadline. Hold the producer's condition
while changing its deadline and wake its waiters; record both original and
effective deadlines. A started three-family optimizer update remains
nonpreemptive. Already active construction, CPU preparation, synchronization and
shutdown may overrun the allowance; report that work. Once the constructor
returns, no fresh preparation operation or learner update may begin after expiry.

Journal attempted and completed work durably. A throwing optimizer, missing
terminal record or unmatched operation intent leaves physical work explicitly
unknown. On failure, close all preparation owners with a bounded join, preserve
their reports and any unjoined work, then write a failed receipt. Keep checkpoint
hashes, the complete artifact inventory, source/runtime checks, peak CUDA memory,
wall and process CPU time, and preparation costs separate from learner exposures.

Passing establishes only this same-process, same-machine trajectory and saved
restart boundary. It does not establish a process restart or performance gain.
Only afterward may a separately declared matched throughput comparison decide
whether the path improves complete training-cycle cost. No promotion or change
to the main learning study follows automatically.
