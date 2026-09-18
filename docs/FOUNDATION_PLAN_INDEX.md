# Authenticated foundation plan index

This implemented, opt-in component retains canonical evidence for a finite foundation plan in the current process. It is not connected to the frozen pilot, its verifier, the trainer or the practice provider. No full-plan benchmark or speed improvement has been measured.

The implementation is [foundation_plan_index.py](../experiments/foundation_plan_index.py). Its API is:

```python
index = AuthenticatedPlanIndex(
    admitted_plan,
    admission_protected_transcripts=historical_transcript_hashes,
    protected_transcripts=expanded_transcript_hashes,
    admission_receipt=canonical_admission_receipt,
)
evidence = index.replay("curriculum", cursor)
with_records = index.replay("mixed", cursor, include_bundles=True)
identity = index.identity
construction = index.construction
```

`replay` returns the same evidence shape as the original `foundation_training.replay_evidence`. With `include_bundles=True`, it returns `{"evidence": ..., "bundles": ...}` in the requested teaching order. `identity` binds the complete admitted plan, admission receipt, historical and expanded protection, curriculum version, source closure, transcript set, bundle records and order-specific prefix evidence. `construction` reports separate admission, admitted-plan scan and prefix-fold times within construction wall time, plus logical admission counts and retained payload sizes. Payload sizes exclude Python/container overhead; construction time excludes imports.

## Why retain authenticated evidence

The frozen pilot verifies two teaching orders, each with 1,536 bundles. Its verifier performs a complete journal replay and restores checkpoints at updates 0, 192, 384, 576, 768, 960, 1,152 and 1,536. Each restore independently replays its consumed prefix.

| Source-derived work across both arms | Bundle visits |
| --- | ---: |
| Complete journal replays | 3,072 |
| Checkpoint-prefix replays | 11,136 |
| Total | 14,208 |
| Distinct admitted bundles | 1,536 |

These replay paths revisit **9.25 times** the number of distinct bundles: 1,363,968 episode slots versus 147,456 unique admitted episodes, at 96 episodes per bundle. The comparison **excludes admission reconstruction, evaluation-bank generation and anchor reconstruction**. It counts source-level bundle visits, not cache misses, neural work, elapsed-time savings or a demonstrated speedup. Existing generator caches can reuse some computation.

The relevant paths are [replay_evidence](../experiments/foundation_training.py), [checkpoint restoration](../experiments/foundation_training.py), and [formal verification](../experiments/foundation_results.py).

## Authentication and limits

Construction first independently regenerates deterministic naming admission against the original historical protection, including any retries, and compares the supplied plan and receipt. It then performs a separate complete scan of admitted bundles. That scan validates canonical pairs, labels, replies, declared cells, exact byte counts, expanded protection and global transcript uniqueness. Historical protection must be a subset of expanded protection. These are two construction passes; construction is not free.

The index derives per-bundle evidence through the unchanged `_rows_evidence` helper and folds it through the unchanged `_accumulate` logic for both orders. It retains only immutable JSON-byte records and prefix evidence. Caller inputs are isolated; every returned dictionary is a fresh copy. Source identities are checked before and after construction or lookup boundaries. Tests cover drift during construction and during lookup.

There is no persistent loader; serialization of the index is rejected. A saved file and a matching self-supplied hash would establish file consistency, not canonical lesson truth. A new process must authenticate its own index. This is also not a security boundary against arbitrary code modifying Python internals. Memory use is finite but grows with the declared plan and its prefix count; it is not an indefinitely compact learning memory.

## Validation and future integration

The first focused run passed **9 tests in 12.260 seconds**, with no failures or errors and identical before/after source hashes. It used tiny canonical 66-bundle plans, both teaching orders and naming overrides. Tests compare stage, partial and endpoint prefixes with original replay; reject corrupt admission, protection, labels, duplicate transcripts and source drift; check mutation isolation; and prove repeated lookups do not regenerate or readmit lessons. No model, inference, optimizer, formal pilot data or full 1,536-bundle benchmark was used.

Captured evidence: report (archive reference: `../runs/foundation-plan-index-validation-local/attempt-001/report.json`), test log (archive reference: `../runs/foundation-plan-index-validation-local/attempt-001/stderr.txt`), stdout (archive reference: `../runs/foundation-plan-index-validation-local/attempt-001/stdout.txt`), source hashes before (archive reference: `../runs/foundation-plan-index-validation-local/attempt-001/sources-before.json`), source hashes after (archive reference: `../runs/foundation-plan-index-validation-local/attempt-001/sources-after.json`).

Validated module SHA256: `61f65d97017eac7ddee882aa4ae3c427d027b71e41aa6dcf667545596d4073d1`.

Future use requires explicit, versioned trainer/verifier integration and new source binding. The integration must match the index to the learner's exact plan and protection, establish cached-versus-original evidence equality at all required prefixes, preserve every weight/optimizer/timing and transactional restore check, and validate continuation under the declared runtime. Existing frozen sources, proofs and receipts remain unchanged. This component supplies input-evidence reuse; it demonstrates no learning benefit, autonomous allocation, promotion decision or broader competence.
