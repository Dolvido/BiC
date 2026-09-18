# Home learning storage audit

The two completed, comparable home cycles occupy **396.670 and 396.787 MiB each**. Their principal storage costs are published lessons, full learner/AdamW checkpoints, and raw research scores. The implemented compact chapter representation has an exact reconstruction proof, but it is **not integrated into the home learning loop**. Nothing was deleted, compressed, moved, or reclaimed in this audit.

The filesystem snapshot was taken on **2026-09-18 at 02:39:22 UTC** (September 17, 22:39 Eastern). All sizes below are logical file lengths; MiB means 1,048,576 bytes. These are neither allocated disk blocks nor Python/GPU memory measurements.

## Scope and current home state

The census covers 21 explicit scopes, **16,105 files and 3,364,387,875 bytes (3.133 GiB)**. Every path is counted once. Invocation overhead excludes the cycle subtrees counted separately. The receipt contains each root/exclusion and a category subtotal; `files.tsv` records every counted path and length. Symbolic links/reparse points would be excluded and reported; none were encountered.

Included home cycles are completed cycle 8 under `continuous-tutor-campaign-local/attempt-003`, completed cycle 9 and prepared cycle 10 under invocation `47e81115884747ffac12e6b9ca76e8d0`. Shared stores and completed paired studies are listed below. The entire live rate study, unlisted historical cycles, virtual environment, Git, installed tutor weights and unrelated machine files are excluded. This is a selected learning-artifact census, not a project-wide disk total.

The authenticated home profile retains **9,328 lifetime updates after nine completed cycles**, with cycle 10 already compiled. Invocation `47e...` records two completed cycles, 864 new physical worker updates and two teacher calls. The subsequent `b781...` invocation records zero new updates or teacher calls and retains the same pending stage. Its extra files are orchestration metadata, not another learning cycle.

Cycles 8 and 9 both began at update 9,112 and completed two branches of 108 teaching plus 108 withdrawal updates: **432 physical updates per cycle**, 96 episodes per update, with the same evaluation shape. That makes their storage footprints comparable. It does not imply equal learning outcomes, equal retained progress, or a permanent growth rate. Pending cycle 10 has no completed worker and is excluded from the per-completed-cycle estimate.

## Measured footprint

| Completed-cycle component | Cycle 8 files / MiB | Cycle 9 files / MiB |
|---|---:|---:|
| Published lesson archives | 324 / 235.570 | 324 / 235.565 |
| Full checkpoints | 4 / 104.888 | 4 / 104.888 |
| Raw research score files | 90 / 43.476 | 90 / 43.477 |
| Protection/admission/provenance | 5 / 5.527 | 5 / 5.634 |
| Source snapshots | 90 / 1.332 | 90 / 1.332 |
| Teacher request/response records | 3 / 0.017 | 3 / 0.017 |
| Logs and other metadata | 22 / 5.860 | 22 / 5.875 |
| **Total** | **538 / 396.670** | **538 / 396.787** |

Prepared cycle 10 occupies **162.658 MiB in 238 files**, including 216 lesson archives. Invocation `47e...` overhead outside cycles 9/10 is 4.959 MiB; the later no-learning invocation is 0.948 MiB; the home profile is 13,638 bytes.

| Shared immutable store or completed study, each `attempt-001` | Files | MiB |
|---|---:|---:|
| Verified tutor inventory | 9 | 129.771 |
| Shared acquisition/transfer banks | 13 | 18.633 |
| Recurrent-read broad lesson cache | 730 | 488.805 |
| Shared-state continuation data | 11 | 22.897 |
| Original definition data | 112 | 65.295 |
| Definition basis data | 112 | 74.579 |
| Copied definition chapter catalogue | 113 | 70.473 |
| Compact referenced chapter catalogue | 3 | 0.484 |
| Complete compact reconstruction proof | 1 | 0.023 |
| Complementary composition data | 186 | 123.126 |
| Prior verified tutor paired worker | 157 | 140.253 |
| Prior definition tutor paired worker | 673 | 185.215 |
| Prior complementary paired worker | 1,617 | 295.656 |
| Prior sustained composition paired worker | 5,508 | 315.103 |
| Prior shared-objective paired worker | 5,511 | 316.181 |

These stores have different workloads and purposes; their totals are not interchangeable cycle-growth samples. Across the complete selected census, lesson images account for 1,416.536 MiB, 34 checkpoints for 891.695 MiB, raw scores for 547.011 MiB, and source snapshots for 11.087 MiB. Per-step receipts contribute 50.323 MiB across 12,096 files. A compact lesson representation does not replace the evidence recorded in scores, optimizer checkpoints or receipts.

## Existing sharing and concrete duplication

Each completed home manifest contains 432 lesson references but only 324 physical paths: both branches already reference one common 108-image withdrawal set. The normal procedural and tutor teaching streams are still published separately. The publisher's relevant boundary is `experiments/verified_tutor_cycle_data_v3.py:497-514`: it reuses procedural paths for an explicit fallback and shares withdrawal records, but otherwise saves each stream's images. The worker writes full checkpoints at `experiments/continuous_tutor_worker.py:464` and raw scores at line 348; these are separate research artifacts, not copied curriculum payloads.

Using **recorded SHA256 values in compiled manifests, whose current file hashes are preserved in the receipt**, plus observed equal file lengths:

- Cycle 8 has 54 same-hash extra teaching copies, totaling **41,185,470 bytes**; cycle 9 has 36, totaling **27,473,620 bytes**.
- Across the two completed cycles' 648 physical lesson files, 414 distinct recorded hashes remain. The 234 extra copies total **178,453,794 bytes (170.187 MiB)**, including cross-cycle duplicates. This is an alternative combined total, not an amount to add to the within-cycle figures.
- Neither completed cycle has identical complete ordered teaching streams. Partial duplicate blocks cannot justify coalescing updates after learner histories diverge.
- Pending cycle 10 does have identical complete ordered teaching streams and already uses the same procedural paths for both, leaving 216 physical lesson files and no same-hash extra copies within that cycle. Whether a future worker can perform equivalent branches once is a separate compute-equivalence question requiring identical starting full state, runtime/objective and complete ordered inputs, with honest physical-work accounting. This census establishes no such runtime proof.

The audit did **not** rehash or deserialize the archives. Duplicate-byte findings are manifest evidence, not a fresh verification of every archive's content. They also do not identify semantically equal examples whose archive metadata differs.

## Compact curriculum proof versus integration

The chapter catalogue copies 108 regrouped images totaling **72,853,828 bytes** from existing basis lessons. `experiments/referenced_definition_lessons.py:163` builds a reference manifest; `ChapterLessonStore` at line 194 resolves it using authenticated source images and detached reconstruction, preserving row/target/evidence identity and original chapter indices. Its **489,721-byte manifest** is 0.6722% of the copied image bytes. The source basis lessons and their provenance remain required.

The preserved complete proof resolved all 108 images exactly in 14.938 seconds, with 324 source-image loads under an eight-entry cache. This historical reconstruction cost is not an end-to-end speed result. See `docs/REFERENCE_LESSON_STORAGE.md` and the proof pin below. Both the copied catalogue and compact catalogue still exist; no saving has been realized on disk.

The home entry point, campaign v3, worker and v3 data publisher do not use `ChapterLessonStore`. Their home lesson artifacts are compiled `.pt` images, so the chapter JSON reference adapter is not a drop-in replacement. The smallest storage follow-up is a **versioned content-addressed lesson publisher**: keep per-cycle order, original coordinates, evidence, source and reserved-split bindings in manifests, while identical authenticated archive bytes have one durable object. Existing artifacts should remain immutable. Prove exact public-loader content/evidence compatibility before using this in the home worker. This targets measured duplicate bytes without requiring regeneration or weakening admission. Source snapshot sharing is smaller in the measured cycles and can wait.

## Capacity, limits and evidence

At the snapshot, drive C had **1,610,446,802,944 free bytes (1,499.845 GiB)**. A transparent planning allowance of **512 MiB per similarly shaped completed paired cycle** exceeds the larger observed cycle by about 29%. Reserving half of the currently free disk gives arithmetic room for **1,499 such allowances**; a nearer 100-cycle allocation would use 50 GiB. These are conditional storage budgets, not supported forecasts of thousands of future iterations. Only two comparable cycles were measured; protection history, new curricula, checkpoints, failed attempts and unrelated disk use can grow. Re-measure before extending a storage budget. The live study can also change free space after this snapshot.

The inventory used only file metadata and 41 distinct selected metadata JSON files. It opened no lesson/checkpoint archive, imported no provider/model and made no network or training calls. A first census failed on a Windows long-path `lstat`; its source and failure record are preserved. The same scope then completed using extended paths only at I/O boundaries. The successful census recorded 1.596 seconds wall / 1.59375 CPU before receipt publication; these are audit costs, not a performance benchmark.

Reproduction artifacts are under `runs/home-storage-analysis-local/attempt-001`:

- `receipt.json`: `b76f036145160a0de4c9554f771c6ab70c2d6133982e8dfc38d7bdfc28b4a76a`; includes the exact scopes, source/metadata pins, file-list digest, duplicate groups and free-space snapshot.
- `files.tsv`: complete path/category/logical-length listing, with no ancestor double counting.
- `failure-001.json` and `failed-inventory-source.py`: preserved first census failure.

The immutable compact manifest is `f3971f53e6b6ea6d597789c0627cac85b668f20364fd7fe4af52d027158296ce`; its full reconstruction proof is `25fc538f3eaa64e7af51293d1c2b8a50064920d5887ee7915013300b33d5db4b`. The current home profile pin is `24b9dfb0143e11613806627fe7b1bf7280b219c2b004f7597c4ee78b262c45cc` and was unchanged across the census.
