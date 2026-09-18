# Explicit recovery of the first local tutor pilot

The first invocation completed cycle zero: 216 GPU updates, 20,736 training
episode exposures, initial and final retention evaluations, and a full weights
and AdamW checkpoint. The local `ministral-3:3b` tutor then returned an accepted
`curriculum` choice in one chat. Its reported prompt length was 2,522 tokens;
the decision took 4.156 seconds, including model loading and identity checks.

The next worker rejected the checkpoint's caller-provided identity before
constructing a model or performing any forward pass or optimizer update. The
original failure, worker receipts, raw response, decision, evaluations and
checkpoint remain preserved in `runs/foundation-tutor-loop-local/attempt-001`.
The first orchestration took 232.969 seconds. Its completed worker took 220.344
seconds and its rejected restore worker took 4.594 seconds.

The exact mismatch was the AdamW `betas` metadata: `(0.9, 0.999)` in the native
checkpoint became `[0.9, 0.999]` in the JSON receipt and worker specification.
The full identity has identical canonical JSON content. The targeted bridge
repair compares JSON identity metadata in canonical JSON form while preserving
strict typed comparisons of tensors, optimizer state, shapes, dtypes and values.
Regression validation must exercise actual JSON round-trips for both restore
routes, since the original CPU checks passed identities directly in memory.

Both targeted JSON round-trip regression tests passed on their first invocation.
They used three tiny CPU updates, nine family forwards/backwards and eighteen
episode exposures, with zero CUDA or tutor calls. Exact weights and full AdamW
state matched across both restore routes. The receipt is
`runs/foundation-layout-continuation-json-validation-local/attempt-001/report.json`
(SHA256 `3240df9484f4b5c9420513e2f8a0d0a9e0182cbef08f6aed8cf3430aa3d0c9c8`).

Recovery must authenticate and reuse that completed parent and accepted tutor
decision. It must preserve their original producing launch identities. It must
not repeat cycle zero, its evaluations, the live tutor call or the rejected
worker's known-zero learning work. A new invocation records the targeted repair,
the old and new source versions, and all inherited costs explicitly.

The learning recipe, three data plans, initialization, fixed evaluation banks,
216-update cycle lengths and final tutor withdrawal remain unchanged. Continue
from lifetime update 216 to 432, then restart and continue without a tutor to
648. Keep the original orchestration deadline rather than resetting the
3,600-second allowance. Any repair validation work is reported separately.

This recovery is a concrete response to a diagnosed integration failure under
the user's existing authorization for local development and training. It is
not an automatic replay of uncertain training or tutor work.
