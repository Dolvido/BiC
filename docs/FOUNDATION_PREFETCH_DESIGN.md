# Bounded preparation ahead of foundation training

Recorded 2026-09-17. Prospective throughput design, not an implemented or adopted trainer change. The active objective study remains unchanged. Before this new file was created, its absence was asserted against all 95 entries of `runs/foundation-objective-study-local/launch.json`, SHA256 `e477b1f6dd4d4f787288bcd6defaba73359949ef67867012caa20c2a81c2c24a`. The destination was also absent on disk. No model, test or benchmark work was performed to write this document.

The proposed change is **one-bundle-ahead CPU preparation**, overlapping deterministic generation, validation and packing with the consumer's existing three-family training update. It changes when input preparation happens. It must preserve the exact lessons, family order, tensor values and learning arithmetic. Whether Python execution, host contention and device work overlap enough to improve throughput is unmeasured.

Implementation status: the standalone preparation owner now exists in
`experiments/foundation_prepared_bundles.py`. Its ten bounded CPU checks passed
once, including exact packed tensor values and layouts across all six depths
and all three lengths, repaired naming admission, source/tensor refusal,
deadline/failure refusal and bounded shutdown. The
validation record (archive reference: `../runs/foundation-prepared-validation-local/README.md`)
separates preparation work from learning. This owner is not connected to a
trainer; CUDA learning equivalence and end-to-end throughput remain untested.
The current objective comparison remains unchanged.

## Completed cost evidence

These are the six completed architecture-study main workers, each with 3,072 updates and 294,912 episode exposures. Values below are seconds. Each `steps.jsonl` was checked against its completed sibling receipt's `journal_sha256`. Scores are unnecessary for this cost analysis.

| Worker | Worker wall | Setup | Sum step wall | Sum retained step time | Materialization included in step | Step-wall minus step-time | Materialization / worker wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat, seed 8462 | 1,272.750 | 10.266 | 1,247.501 | 1,171.650 | 508.961 | 75.851 | 39.99% |
| flat, seed 8463 | 1,216.953 | 10.594 | 1,192.930 | 1,120.859 | 487.498 | 72.071 | 40.06% |
| flat, seed 8464 | 1,218.781 | 10.688 | 1,193.492 | 1,122.314 | 477.997 | 71.178 | 39.22% |
| hierarchical, seed 8462 | 1,271.172 | 10.203 | 1,246.471 | 1,170.811 | 515.004 | 75.660 | 40.51% |
| hierarchical, seed 8463 | 1,190.375 | 10.594 | 1,167.213 | 1,096.387 | 488.194 | 70.826 | 41.01% |
| hierarchical, seed 8464 | 1,224.156 | 10.609 | 1,198.802 | 1,127.214 | 485.919 | 71.588 | 39.69% |

`materialization_seconds` encloses `_materialize_validated_bundle` and `_rows_evidence`. It excludes the subsequent per-family packing and its validation. Retained step time encloses materialization, packing, forwards/backwards, clipping, AdamW, synchronization and the post-update source check. Step wall additionally includes the initial source preflight and pre-step device synchronization. Therefore neither subtraction is a measurement of GPU computation or source hashing alone. Materialization represents 42.59–44.53% of retained step time; its sum is 2,963.573 seconds across workers, not recoverable elapsed time.

The flat and hierarchical worker processes ran concurrently. Their first-to-last main windows were approximately 06:41:52–07:43:43 and 06:41:47–07:43:15 UTC on 2026-09-17. The summed worker wall of 7,394.187 seconds is not elapsed wall time or GPU-hours. No utilization trace was collected. Peak allocated device memory, approximately 1,502–1,503 MiB flat and 906–907 MiB hierarchical, does not measure GPU utilization.

Each architecture process built one authenticated plan index reused by all three seeds. Those construction costs were 988.359 seconds flat and 982.906 seconds hierarchical, outside the worker wall above. Their admission/scan components were 486.359/494.844 and 485.390/490.188 seconds respectively. Each index contained 3,107,603 bytes of records and 7,156,527 bytes of prefix payload. A prepared queue must not disguise new whole-plan preparation as free index reuse. The existing objective dispatcher already reuses one process index across its jobs.

Receipts live at `runs/variant-study-local/main/<architecture>/<architecture>-seed<seed>/receipt.json`; journals are sibling `steps.jsonl` files. Exact receipt and journal pins used for the measurements are:

| Worker | Receipt SHA256 | Journal SHA256 |
|---|---|---|
| flat 8462 | `3cbe2105713d754405330b4347af8bc9a294d8630dd82fba684fb023bc11d666` | `03a65de348c2c2a739cf81c7efcdf0703c377474cd9e345d2d8e45cf16a12144` |
| flat 8463 | `00bfd6ab2d9c4c25da7c5ba0e266ce153a5527b8e86c8457b988ae786e814c8f` | `caffa7ca746f9bb44385eed60be613c6c01337e9a7bcbf7a465fb7de464dfafe` |
| flat 8464 | `6381d78f905bf04644097142bdfd7b2ff1d84077d780db0387611e87f179f01c` | `b1ea27311896ca08cd1ceb425e5fb3d6dc6e2e4e43b6f3dd177cf8dee324cfb7` |
| hierarchical 8462 | `1b640c3bc5137868319a0be62f2e8072f11212028d1d17e9b20e0e09238d68b9` | `1fe1de35e8b0d9c9eb271605d7f2b86fdafecac4f7ea740e7e03ad5dc53b428e` |
| hierarchical 8463 | `d1ccc92fccda0b25bf347a4061f75b4bd102ce291e69f4cce3f7e8cce6d2a169` | `0afafe38c83986d7289a15ecbbd89d3eabe1539cd8c3a4dc92ce6aba0509b5cd` |
| hierarchical 8464 | `7836e2394019302f1a80b07c0e8ca9ddff7c45d4017cc0dda4efe431d2fd1cff` | `5962f98f50782d7e303bc2103ab0efef7a38827605eaba926efe36b376e30443` |

## Exact integration boundary

Use one same-process CPU producer thread and one training consumer. Permit only the current bundle and one following bundle, including an in-progress item in this bound. Do not cache the full plan's tensors. The producer owns a detached admitted plan and prepares all three families of a scheduled bundle with the unchanged sequence:

1. `_materialize_validated_bundle(plan, bundle_id)`;
2. `_rows_evidence(plan, bundle_id, rows, protected_set)`;
3. `pack_composition_episodes(..., device="cpu", training=True, pair_validator=foundation_curriculum.validate_pair, ...)` using the exact consumer packing limits.

Retain exact raw row/recipe evidence and complete adjacent canonical pairs. No model, CUDA allocation, gradient or optimizer work belongs in the producer. Keep the existing source/protection/admission checks; preparation must not bypass the packer's canonical pair validation.

Require the actual exact-type `AuthenticatedPlanIndex` admitted for the same plan, source closure and protection inventory. Initially obtain expected bundle evidence once through its public `replay(order, len(plan["schedules"][order]), include_bundles=True)` and retain a detached table indexed by the scheduled bundle ID. Charge this setup and memory explicitly. Do not call full-prefix replay at every consumption and do not treat a caller-authored sidecar as authenticated evidence. A narrower constant-time public record accessor could be a later separately reviewed index change.

Every produced item binds the index identity, plan identity, order, cursor, bundle ID, complete protection identity, packing limits, source/runtime identity and tensor/evidence digests. Before accepting it, the consumer checks that these match its exact expected next update and that the packed tensors remain unmodified. Ownership must prevent an externally returned mutable tensor from silently replacing queued content. A saved digest is not permission to skip fresh source or index guards.

An additive, versioned trainer entry point would replace only input acquisition and packing around the current `ObjectiveFoundationTrainer.step` materialization/per-family packing region. Consumption transfers the identical CPU integer/Boolean tensors to the declared device and then retains **color, count, switch** as three separate forwards/backwards, each loss divided by three, followed by the existing single global clip and single AdamW update. No family concatenation, new padding, floating-point dtype change, objective reduction change, asynchronous model update or altered kernel arithmetic is part of this proposal. Both currently defined objective IDs must remain supported.

The consumer retains fresh source checks at the existing pre-step and post-optimizer boundaries and the same checkpoint/source/membership guards. Producer guards may add checks. This proposal neither adopts a source-content cache nor removes repeated checks. The 70.8–75.9 second preflight/synchronization gaps suggest a separate possible cost, but no isolated source-check measurement establishes its size.

## Failure, shutdown and allowance accounting

Preparation and training are separate work categories. Record preparation attempts/completions, regenerated rows/episodes, validation and packing completion, per-item identity, producer CPU/wall time, queue wait, transfer time, queued/discarded items, and peak queue memory. Preparation is not a consumed lesson or optimizer update. Preserve the existing physical, retained, attempted and completed learner work fields without relabeling them.

The consumed cursor advances only with the same successful optimizer/source/evidence commit as the synchronous trainer. Reject a missing, duplicate, stale, out-of-order, incorrectly protected or modified item before model work. A producer exception is retained and poisons the queue; it is not an instruction to regenerate under a new seed, retry or skip to another bundle. Failures after neural work preserve its actual or explicitly unknown work counts and require the existing explicit restore policy.

The caller supplies finite update/cycle/deadline bounds. Never schedule preparation beyond the invocation's allowed next updates, and check allowance again before each training update. On stop, deadline or failure, cease new preparation and training, then close and join the producer. At most the already-running preparation operation can finish nonpreemptively; record and discard it if no longer usable. Account for that shutdown overrun explicitly. Do not continue training merely because a prepared item is available.

The queue is transient and absent from learner checkpoints. An explicit restore starts from the authenticated committed cursor and rebuilds preparation, charging repeated work. A durable invocation/operation intent and completed receipt must distinguish known completed work from interrupted or hard-killed preparation/training. Missing or malformed completion evidence means unknown physical work; checkpoint rollback cannot turn it into zero work. No automatic replay or adoption of orphan prepared artifacts follows a stale intent.

## Minimum evidence before adoption

1. **Input identity:** compare synchronous and prepared raw rows, recipes, complete pair ordering, exposure evidence and every packed CPU tensor, including shapes, padding, masks, target IDs and decoder inputs. Cover all six depths, all three turn lengths and repaired admission naming overrides. Exercise explicit small bounds and early close without admitting an extra item.
2. **Execution identity:** under the declared same hardware/runtime profile, compare every step's weights, full AdamW moments/steps and consumed evidence for both objective IDs. Include uninterrupted, split/strict-restore and repeated runs. Keep the family sequence and reduction order exact. A same-process proof must not be advertised as a cross-process determinism guarantee.
3. **Refusal and accounting:** reject source/membership drift, changed index/protection/packing limits, wrong/stale/out-of-order or mutated items, producer exceptions and consumer failures. Cover zero-update allowance, allowance exhaustion, deadline during preparation, exception during consumption and interrupted receipt publication. No retry, cursor advance or hidden neural exposure may occur on refusal.
4. **Measured benefit:** compare complete synchronous/prepared invocations under the same hardware and competing load. Report startup/index/table preparation, CPU time, actual lesson preparation, queue waits, host/device transfer, device synchronization, checkpoint I/O, shutdown/discards, memory and end-to-end wall time. Keep learned endpoints exact. Predeclare a small fixed work budget and a meaningful throughput threshold before the measurement; neither is selected by this document.

A passed identity proof establishes execution equivalence only within its tested scope. Adoption additionally requires an end-to-end improvement after all preparation and shutdown costs. The GIL, two concurrent workers and extra ownership/digest checks may erase the expected overlap. Even a proven speedup would provide more efficient execution, not evidence of broader learning, independent tutoring, beneficial self-direction or improved transfer.
