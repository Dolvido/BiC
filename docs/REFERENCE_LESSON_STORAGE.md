# Reference lesson storage

The completed reference catalogue preserves the exact 108 previously admitted chapter images without storing another copy of their episode rows and prefix targets. `experiments/referenced_definition_lessons.py` records source-image pins, pair coordinates, original chapter indices, evidence and expected reconstructed-image digests. Its bounded `ChapterLessonStore.resolve(index)` returns detached images with the original chapter identity. A training caller must separately remap the consumption cursor through its admitted interface.

The compact manifest is **489,721 bytes**, compared with **72,853,828 bytes** for the 108 copied chapter images: 0.6722% of that image storage, or approximately 148.77 times smaller. This comparison excludes shared source lessons and provenance, which remain required. No files were deleted and no disk space has been reclaimed.

Compilation completed in 3.547 seconds wall / 3.546875 seconds CPU, reading 113 JSON files (108 chapter images plus five metadata files). The independent full reconstruction proof matched all 108 original canonical image hashes, including rows, targets and evidence, in **14.938 seconds wall / 14.84375 seconds CPU**. With an eight-source-image cache, it loaded 324 source images, reconstructed 5,184 family pairs / 10,368 rows, and recorded 1,404 cache hits and 316 evictions. Peak cached serialized bytes were 5,404,305; this is not a resident-memory measurement. Both operations used zero canonical generation, prefix interpretation, archive decoding, models or teacher calls.

Evidence:

- Compact manifest: `runs/referenced-definition-lessons-local/attempt-001/manifest.json`, SHA256 `f3971f53e6b6ea6d597789c0627cac85b668f20364fd7fe4af52d027158296ce`.
- Compilation receipt: same directory, `preparation.json`, SHA256 `610534d32c64251a88f7b56e56c5a117f90cf6f31aa2839f8da109ae854e3e49`.
- Full proof: `runs/referenced-definition-lessons-analysis-local/attempt-001/verification.json`, SHA256 `25fc538f3eaa64e7af51293d1c2b8a50064920d5887ee7915013300b33d5db4b`.
- Resolver source: SHA256 `f84c97a13979b737fc001dcb518a18f2c76592685987d0070d80d1d08f007ac3`.

This proves exact reconstruction and a smaller reusable lesson representation. It does not prove faster training, and the resolver is not integrated into a learning controller. Required research scores, checkpoints and failure receipts have a separate purpose and remain preserved.

## Measured repeated-practice costs

The completed complementary comparison (`runs/complementary-composition-local/attempt-001/execution/summary.json`, SHA256 `b384bb7aa293dff92d7774c9f6f11552e5d8488a5a2df38a29a123945d2f1e2e`) took 568.235 wall seconds for 1,296 updates. Learner steps account for 182.749 seconds, native scoring for 57.772 seconds, and eight restorations for 1.861 seconds. Owner preparation-plus-consumption reports sum to 152.642 seconds, but token consumption runs inside learner steps: these totals overlap and must not be added as independent components. The 385.486 seconds outside learner steps cannot be attributed entirely to packing or I/O. Input reading, target preparation, publication, bank construction, source checks and cleanup were not separately timed.

The run nevertheless records definite repeated work: 648 definition JSON loads, 648 training replay archive loads (plus one evaluation-bank archive), and 1,944 family replay-target preparations covering 62,208 rows. It published 1,516 artifacts without a separate publication timer. `complementary_composition_run.run` reloads each lesson occurrence; `PreparedLayoutOwner.prepare` packs every admitted update, and its consumption checks copy and authenticate tensors.

For four passes of the candidate's existing 648-update schedule, a bounded private cache of authenticated images and CPU prefix targets could reduce 1,296 definition JSON occurrences to 324 distinct images, and 1,296 replay archive occurrences to 108 distinct images. The latter would reduce repeated target preparation from 3,888 family calls to 324. These are operation counts, not predicted time savings. The cache would retain immutable source/evidence pins and clone per-use data, changing only the top-level consumption cursor and issuing a fresh public owner token for every update. Packing would still occur 2,592 times. Definitions must reuse their pinned provider targets; the legacy English state parser does not interpret definitions.

This is the smallest prospective reuse boundary that leaves the frozen learner and continuation recipe intact. Existing `sustained_replay.ResidentReplay` demonstrates the archive/target cache pattern but is specific to its old provider. Reusing packed tensors additionally requires an explicit provider-aware extension to the versioned public packed-image API and a full next-update weights/AdamW/evidence equivalence proof. Neither integration is implemented here. A future matched inclusive timing check must count setup, authentication, cache misses, packing, publications and shutdown; the eight-entry reconstruction result alone provides no speed claim.
