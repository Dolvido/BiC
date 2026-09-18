# Shared-state continuation: evaluation data declaration

This declaration precedes fresh data generation for the separately frozen
[continuation study](SHARED_STATE_CONTINUATION_DESIGN.md). The fixed fresh
development seed is **852702001**. Preparation constructs no learner, optimizer
or teacher and performs no training or evaluation inference.

The additive implementation is
[`shared_state_continuation_data.py`](../experiments/shared_state_continuation_data.py).
Its schema is `bic-shared-state-continuation-data-v1`; `prepare(directory)` creates
one exclusive output directory, and `load_manifest(directory,
expected_manifest_sha256=...)` authenticates the completed overlay. The CLI is
`python -B -m experiments.shared_state_continuation_data DIRECTORY`.

## Immutable inputs and unchanged training

The original recurrent-read manifest is
`runs/recurrent-read-data-local/attempt-001/manifest.json`, SHA-256
`c420a43a3a45f63347288dda7db48c71b3e7fa183b4a1681c2ec89eabeb08654`.
All **648 original training records** remain exactly equal to that manifest.
Their paths stay relative to `parent_data_directory`; no training archive is
written or regenerated. These records describe 62,208 original episode rows.
Repeated consumption and global replay IDs belong to the subsequent training
runner, not this overlay.

The previous overlay is
`runs/entity-retrieval-data-local/attempt-001/manifest.json`, SHA-256
`7c0d1f208f59a183cf5c052c991c842c2358107a3c40c84f4fc36ba42fa63612`.
The earlier shared-state manifest, used to reconstruct exclusions, has SHA-256
`266dfe643074b8d66b65b1e506f6fc726fa6cb75f2a532b04232ec87bf30eb75`.
Each bank archive is authenticated before weights-only CPU decoding. Existing
source closures and the declaration bytes are checked before completion.

## Four evaluation banks

| Bank | Role | Rows / complete pairs | Provenance |
|---|---|---:|---|
| `dev` | dev | 720 / 360 | Newly generated with seed 852702001 |
| `previous_dev` | dev | 720 / 360 | Exact entity-retrieval development rows |
| `train_fit` | train_fit | 108 / 54 | Existing actual-training sample |
| `retention` | dev | 720 / 360 | Existing reused development bank |

The bank archive has 2,268 rows and 1,134 complete pairs. Existing original-layout
generation and independent admission are reused. Fresh development keeps the
same balanced family/operator/depth/turn-length design. Reused bank contents are
copied exactly; fitting and retention contents must also equal the original
cache's banks. No auxiliary state targets are produced here.

Preparation reconstructs and checks the previous **1,702,440** excluded transcript
hashes against the pinned entity-overlay exclusion digest. Its actual 720
development transcripts must be distinct and disjoint from that union, giving
**1,703,160** excluded recorded-study transcripts. Fresh rows must be mutually
distinct and disjoint from the resulting union. This is exact observed-text
exclusion within the existing finite grammar, not a claim of new grammar or
independence from every unrecorded historical experiment.

## Work and failure accounting

The fixed allowance is 600 seconds from preparation entry, including input
authentication, archive decoding, canonical generation/admission and artifact
writes up to manifest publication. There is one invocation and no automatic
retry. A terminal failure receipt is retained; a manifest alone does not make a
preparation usable. Final receipt serialization follows the measured work.

Successful preparation decodes **222 existing archives**: five direct reads
(earlier and previous overlay banks, training/reserved/history inventories),
plus 217 delegated reads (the original bank archive and the first 216 cached
training bundles). Attempts and completions are recorded separately, including
failures. The work ledger's five direct safe-load counts overlap the direct
counter and must not be added again. JSON metadata reads and byte hashing are
included in elapsed/CPU time, not called archive decodes. The other 432 original
training archives remain pinned records and are authenticated when consumed;
preparation does not claim to decode all 648.

The first 216 bundles supply the authenticated 20,736-transcript first-cycle
training inventory required by the existing evaluation builder. Canonical
generation can reconstruct training parents while building/admitting the new
bank; every such generator attempt, completion and returned row is counted by
the existing tracking ledger. Returned/reconstructed rows are not new unique
training lessons. The receipt also records layout validation work, adapted
evaluation pairs, wall time and CPU time. Guards reject model construction,
model forward, backward, optimizer construction/step and CUDA initialization;
teacher calls are absent. This data preparation does not imply any learner
improvement, checkpoint adoption or tutor connection.
