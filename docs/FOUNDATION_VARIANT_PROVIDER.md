# Architecture-aware foundation provider

`experiments/foundation_variant_provider.py` adds an opt-in
`VariantFoundationPracticeProvider` for explicit `flat` or `hierarchical`
`VariantFoundationTrainer` instances. It is an engineering adapter, not a model
promotion, a demonstrated learning improvement or an automatic curriculum
extension. The active architecture study and all 88 frozen sources are unchanged.

The adapter subclasses the existing provider and works with the unchanged
`FoundationLoop`. Prescribed lesson requests, partial-prefix accounting,
teacher-free evaluation, historical evidence checks and the loop's single atomic
envelope remain the existing implementations. The new provider schema is
`bic-foundation-variant-practice-provider-v1`. Its identity binds the explicit
architecture/version, sources, runtime, configuration, original admission
protection, expanded protection, canonical evaluation specifications and trainer
recipe. Old provider/learner checkpoints and cross-architecture payloads are
rejected, even when initial tensor inventories match.

Both `development` and `retention` must contain canonical development rows for
all three domains. Every evaluation transcript must be protected from training.
Scoring supplies raw observations and a BOS-only reply prefix; replies are
generated freely. The existing evaluation-response schema is reused because its
semantics are unchanged. Its `provider_sha256` binds the new architecture
identity; separate weight-digest and update-count fields bind the producing state.

A complete `AuthenticatedPlanIndex` authenticates original admission and expanded
protection **before model construction**. An explicitly supplied index must be
the real immutable object constructed in the current process with exactly the
same identities. No serialized cache metadata is accepted as authentication.
Candidate restores reuse the index, fully validate learner/optimizer and pending
state, then recheck source/runtime identity before replacing live state. Caller
payloads are detached before validation.

For an independently admitted engineering plan and reserved development banks:

```python
from experiments.foundation_plan_index import AuthenticatedPlanIndex
from experiments.foundation_variant_provider import VariantFoundationPracticeProvider
from experiments.foundation_loop import FoundationLoop

index = AuthenticatedPlanIndex(
    plan, admission_receipt=admission_receipt,
    admission_protected_transcripts=original_protection,
    protected_transcripts=expanded_protection,
)

def new_provider():
    return VariantFoundationPracticeProvider(
        plan, architecture="hierarchical", seed=8513, config=config,
        evaluation_banks=evaluation_banks,  # development and retention
        admission_receipt=admission_receipt,
        admission_protected_transcripts=original_protection,
        protected_transcripts=expanded_protection,
        plan_index=index, device="cpu",
    )

loop = FoundationLoop(new_provider(), window_updates=16,
                      chunk_updates=4, history_limit=32)
loop.save("engineering-loop/envelope.pt")
loop.run(max_updates=4)
del loop
resumed = FoundationLoop.load("engineering-loop/envelope.pt", new_provider())
```

The caller owns the output location, finite practice budget and prepared data.
An existing envelope is loaded explicitly, never silently replaced. Partial
evaluation queues and pending practice prefixes are saved alongside full
optimizer/controller state. The inherited loop still requires final evaluation
at plan exhaustion; this adapter does not recycle a completed plan.

The first captured validation passed **10 tests in 47.647 seconds** (49.359 seconds
including process startup). The receipt (archive reference: `../runs/foundation-variant-provider-validation-local/iteration-20260917T045611381324Z/receipt.json`),
physical-work ledger (archive reference: `../runs/foundation-variant-provider-validation-local/iteration-20260917T045611381324Z/physical-work.json`)
and test log (archive reference: `../runs/foundation-variant-provider-validation-local/iteration-20260917T045611381324Z/stderr.txt`)
record both architecture comparisons, pending resume, partial evaluation reload,
failed publication rollback, publication-after-replace reconciliation, detached
restore inputs and rejection before live-state mutation.

Actual CPU work was 30 trainer-step attempts, 28 completed optimizer calls,
180 drawn episodes, 176 neural-attempted episodes and 172 completed-microbatch
episodes. Two deliberate failures occurred after the second encoder forward;
neither reached an optimizer update. The completed-update total also includes
two updates discarded after definite publication failures. There were 88
training encoder calls and 38 evaluation encoder calls, all completed; 38 banks
scored 76 episodes. The ledger separately records 126 reply-decoder `forward`
calls covering 2,376 turn-sequence instances. Those decoder instances are not
additional training episodes or a count of every internal generation operation.
No optimizer completion was unknown. All 88 architecture-study, 71 capacity and
79 prior-foundation source hashes matched before and after the run. No formal
data or learned weights were used, and no GPU work ran.

Provider source SHA256:
`2e80dedac35220dc0ebdbd3fc5bdc78b59134bdd8a1b73385c96d42a7fe24bd5`.
Tests SHA256:
`1ca8950c6256c766610df889b58112bf70309cd92230f97cf83c2fe4469f24fc`.

`setup_report` separates index admission/checking, bank preparation and initial
trainer setup; these are included intervals, not additive wall times. Reused
index construction time is historical. Index reuse removes repeated canonical
prefix materialization, but candidate model/optimizer construction, scoring and
full-envelope writes remain real costs. No speedup was measured. The new adapter
has tiny CPU integration evidence only: prior direct-trainer or old-loop CUDA
proofs do not establish CUDA equivalence for this newly composed path.
