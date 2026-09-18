# Verified tutor curriculum: implemented integration and remaining limits

The [first authored-curriculum comparison](VERIFIED_TUTOR_RESULTS.md) is complete.
An external local LLM selected a six-chapter recipe, the canonical compiler
verified its lessons, and matched native learners completed 432 updates in total.
Both saved and restored full weights/AdamW before a common tutor-free continuation;
the learning worker made zero teacher calls. Actual examples differed in 54 of
108 teaching batches. Final development joint pairs were 158/360 for the tutor
branch versus 154/360 for procedural teaching, but retention and withdrawal
regressions failed the declared benefit screen. No checkpoint was promoted.

The [v2 author](../experiments/verified_tutor_author_v2.py) and
[v2 compiler](../experiments/verified_tutor_curriculum_v2.py) preserve the original
failed HTTP request and incomplete compilation as separate recorded attempts.
Their focused tests and the completed local run establish this engineering loop.
They do not establish a useful tutoring advantage, general English comprehension
or learned self-direction. The design below records the implementation rationale
and the continuing boundary between external authorship and native learning.

## Implemented boundary

The [completed local tutor pilot](FOUNDATION_TUTOR_LOOP_RESULTS.md) connected one real local LLM request to continued learning, unaided evaluation, optimizer restart and caller-controlled withdrawal. It completed 648 learner updates. The single accepted choice was the procedural default, so it supplies no causal estimate of tutor benefit. The final audit had zero joint successes in 396 opposite-answer pairs.

The earlier [adviser](../experiments/foundation_tutor_adviser.py) accepts only a `profile_id`. `_catalogue()` verifies whole-curriculum coverage, rehearsal and equal exposure budgets; `_chat_payload()` prohibits new lessons, facts and labels. [That loop's `request_for()`](../experiments/foundation_tutor_loop_run.py) offers `curriculum` or `mixed` order on the same original-layout inventory. The new completed integration adds bounded chapter choices and verified fresh naming/value realizations. Neither path learns arbitrary explanations or delegates truth to the LLM.

`decide()` / `load_decision()` already provide the useful persistence boundary: immutable request intent before a call, externally pinned decision reuse, preserved response bytes, explicit unknown interrupted requests with no automatic replay, local model digest checks before/after generation, `keep_alive=0`, teacher cost and zero-call withdrawal. Preserve these behaviors in a new versioned authoring adapter.

Older [LocalTutor.explain()](../brain_in_computer/curriculum_tutor.py) selects exact registered narration, a follow-up and an existing verified practice packet. [LearningLoop](../brain_in_computer/learning_loop.py) appends `practice_examples`; explanation strings remain metadata rather than learned inputs. [DialogueTutor.teach()](../brain_in_computer/dialogue_tutor.py) chooses one of two procedurally generated dialogues; [DialogueLoop](../brain_in_computer/dialogue_learning.py) appends that episode. Neither path supplies arbitrary authored explanations to the current SharedStateStudent.

## Proposed minimum restart adapter

The current [rate pilot checkpoint](../experiments/shared_state_rate_pilot.py) has outer schema `bic-shared-state-rate-pilot-v1` and an inner [SharedStateKernel](../experiments/shared_state_training.py) snapshot with schema `bic-shared-state-training-v1`. The latter contains `recipe`, `weights`, full `optimizer`, `cursor`, `evidence` and `accounting` (including `state_work`). The existing [LayoutContinuation](../experiments/foundation_layout_continuation.py) reconstructs FoundationLayoutTrainer/SequenceStudent and admits different schemas. It cannot correctly resume this checkpoint by relabeling it.

Add a small `SharedStateContinuation` wrapper in a new module, initially supporting only the explicitly declared rate-pilot producer and its own continuation envelope:

```python
SharedStateContinuation.from_checkpoint(
    checkpoint_bytes, *, expected_sha256, expected_identity, device="cpu")
SharedStateContinuation.from_snapshot(
    snapshot_bytes, *, expected_sha256, expected_identity, device="cpu")
continuation.step(prepared, *, state_targets, deadline=None)
continuation.snapshot()
```

Expose the existing `model`, `optimizer`, `cursor`, `evidence`, `recipe`, `accounting` and `last_report` interfaces. Keep the original kernel recipe intact; the outer continuation schema separately binds adapter sources, origin and restore cost. Do not mint an old CycleOwner token or claim historical execution was replayed.

Minimum restore contract:

1. The caller authenticates the completed producing launch/summary and supplies the checkpoint pin plus expected producer schema, arm, step, architecture, config, learning rate, auxiliary weight, runtime, weight digest and recipe/evidence identity. Hash the single immutable byte image before `torch.load(..., weights_only=True, map_location="cpu")`.
2. Require the original objective, auxiliary weight 0.3 and unchanged source/runtime contract. Construct one same-configuration SharedStateStudent and AdamW template. Validate all finite weight keys, shapes/dtypes and tied token/observation aliases. Check exact optimizer parameter order/groups, complete first/second moments, finite matching moment shapes/dtypes, nonnegative second moments and every AdamW step equal to the committed cursor. Normalize only the declared JSON metadata representation so tuple/list serialization does not repeat the prior beta-field restart failure; keep tensor and numeric checks strict.
3. Strictly load weights and full optimizer state, synchronize, and compare the complete CPU snapshot and recorded weight digest before exposing the learner. Restore original evidence, cumulative work, state-work and cost counters; report restore work separately. This validates compatibility of a trusted pinned checkpoint, not its past gradients.
4. Delegate each update to `SharedStateKernel.step()`, preserving its poisoning, physical-update accounting and no-retry behavior. New bundles have monotonically increasing global IDs, while their canonical parent recipes retain their actual provenance. Compute deterministic state targets once from admitted English and pass them only as `state_targets`, never to `model.forward()`.

Use the current [PreparedLayoutOwner](../experiments/foundation_layout_prepared.py) for packing admitted bundles. Snapshot first, then publish the owner/loop commit last; persist the accepted curriculum pin with the checkpoint. On restart, reuse that choice without contacting the teacher. An unresolved teacher intent or uncertain optimizer outcome requires explicit recovery, not implicit repetition. The implemented [SharedStateContinuation](../experiments/shared_state_continuation.py) now passes exact next-step weights, AdamW, losses and evidence comparisons on CPU and the local RTX 5080. This establishes the tested restoration path, not learned capability.

## Proposed compact authoring and compilation

The smallest compatible extension is **tutor-authored chapter recipes**, not unrestricted prose accepted as training truth:

```python
author_curriculum(directory, *, request, expected_request_sha256,
                  expected_result_sha256=None, deadline=None)
compile_curriculum(recipe, *, admitted_parent_inventory,
                   protected_transcripts, coverage_contract)
# -> immutable canonical bundle descriptors, evidence, cost/rejection receipt
```

The request binds the restored parent, development-only aggregate evidence, allowed operator vocabulary and a caller-frozen coverage/exposure budget. The response contains a short list of family-independent chapters: causal operator-chain motifs, worked-example/contrast progression, and naming/value variation. Apply each chapter across color, count and switch, rather than requesting extra practice for individual weak cells. The teacher never supplies answers, oracle state, source code, held-out examples or withdrawal decisions.

For the first version, the compiler resolves chapter constraints to existing admitted canonical parent recipes and bounded fresh naming/value seeds. It rejects unsupported motifs instead of silently changing them. Use [foundation_curriculum.generate_pair()/validate_pair()/query_ancestries()](../experiments/foundation_curriculum.py), the independent typed and English oracles in [composition_curriculum](../experiments/composition_curriculum.py), and [foundation_layout_curriculum.validate_pair()](../experiments/foundation_layout_curriculum.py). All outputs must regenerate exactly and obey train/structure admission and transcript exclusions before packing. Freeze the compiled inventory once, and reuse its authenticated bytes/compact descriptors rather than calling the teacher or reconstructing history during updates.

This preserves the existing student, loss, state-target parser and prepared-bundle interface. The learner consumes the compiled English examples and their independently derived supervision. Any optional tutor rationale is stored as untrusted metadata and is **not** claimed to have taught the learner. Genuine new explanatory text is outside this minimal version: the current canonical validator rejects altered text, and [pack_composition_episodes()](../experiments/composition_data.py) fixes replies to four canonical responses. Teaching explanations would require a separately declared episode/verification contract, not appending an unchecked `explanation` field.

The compiler must check actual materialized family/depth/length/operator coverage, opposite-answer pairs, known/unknown presence, replay/rehearsal floors and total episode/token bounds. Replay is caller-admitted prior training data, never development/audit examples. The same constraints and compiler apply to the procedural control. Parent checkpoint, authored recipe, compiled inventory and consumed evidence all receive distinct identities.

## Minimal matched comparison and withdrawal

Start two branches from the same authenticated parent weights **and full AdamW**. One receives the accepted tutor-authored chapter recipe; the other receives the fixed procedural recipe. Use the same compiler, eligible parent pool, seed allocation, coverage, replay floor, settings and training exposure/token budget. Constrain lesson ordering to the same declared policy so the intended contrast is the authored chapter content/progression rather than an incidental `curriculum` versus `mixed` switch. Record actual differences between the compiled curricula. If the teacher chooses the default or compiles to the same inventory/order, report zero treatment contrast; do not interpret it as a benefit test.

Freeze unaided evaluation before either branch learns. Compare joint counterfactual action/reply correctness, each family's known/unknown and nonanchor factual answers, retention, transfer and total teacher-plus-learner cost. Score freely generated replies with the existing [PreparedLayoutBank](../experiments/foundation_layout_evaluation.py); no teacher outputs, state targets or auxiliary decoder answers enter policy evaluation. A good explanation or accepted request is not an outcome measure.

After the fixed teaching interval, both branches continue for the same caller-prescribed interval with the tutor withdrawn and a common procedural/replay schedule. Require zero API calls and evaluate again. This measures retention and continued learning after external help ends; it does not establish a learned curriculum policy or general independence. The existing order-only pilot, rejected architecture results and current rate study remain unchanged, and no checkpoint is promoted by this design.
