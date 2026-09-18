# Small prospective shared-learning diagnostic

Recorded **2026-09-17 07:15 UTC, before current main-result scoring**. This is a
proposal only: no diagnostic implementation, training, inference, or checkpoint
inspection was performed. The owner will decide whether to execute it after the
completed main comparison. No frozen source, formal ranking, or budget changes.

The historical pilot's weak fitting in every family does not establish that
understanding is intact behind a poor reply decoder. Conversely, low final
accuracy cannot locate the failure in representations, readouts, objectives, or
optimization. Use one shared diagnostic across color, count, and switch.

## First pass: cached representations and a semantic first-byte check

Use three predetermined historical states: the pilot's common initialization
and **both** curriculum/mixed endpoints at 1,536 updates. Select no winning arm,
intermediate checkpoint, or current formal-study learner.

1. Prepare two new, separately seeded diagnostic datasets, each containing eight
   complete opposite-answer pairs in every one of the 63 family/depth/operator/
   length cells from `foundation_banks.expected_fresh_cells()`. Use fit seed
   **916170001** and evaluation seed **916170002**, deriving per-cell candidate
   seeds by canonical SHA-256 of role, cell, and attempt. Fit examples use the
   train-admissible grammar; evaluation uses fresh names/values in that same
   grammar (`split="dev"`, train composition ancestry). Both members stay in one
   split. Deterministically reject transcript collisions with all historical and
   formal-study exclusions and with the other diagnostic role. Never convert a
   reserved development/audit bank into fitting data. Seal both datasets and
   their inventories before model work; bounded admission failure stops the job.
2. Freeze each entire learner. Cache its EOS encoder states and 128-dimensional
   `production_context`, along with original action and BOS-only decoder logits.
   Fit one shared four-class linear readout on each representation, across all
   families. Use the original action weighting: equal present query classes plus
   0.25 acknowledgement, averaging equal-cell/equal-family contributions. Use
   one fixed full-batch convex solver, maximum 200 iterations, and report its
   convergence; no rate search, cell-specific head, or evaluation-based stopping.
   Evaluate once on the untouched diagnostic evaluation split.
3. Run the identical refits on the matched initialization features. Also repeat
   each fitting run with one fixed label permutation within family/turn-kind,
   preserving class counts; evaluate against the true evaluation labels. These
   controls expose easy input shortcuts, finite-sample fitting, and a probe that
   succeeds without learned representations. They require no new encoder pass.
4. Compare original action decisions, original decoder first bytes, and complete
   generated replies. Canonical replies are `No.`, `Yes.`,
   `I need more information.`, and `Understood.`: **N/Y/I/U identify the four
   classes at the first byte**. That decision receives BOS during both training
   and evaluation, before any correct reply prefix. Score the unrestricted
   259-symbol argmax against the expected first byte; do not force a four-letter
   vocabulary. Generate complete replies from cached reply contexts for the
   evaluation split only, with the unchanged BOS-only decoder.

This requires **6,048 encoder episode passes**: three states × two splits ×
1,008 episodes. The small readout fits reuse cached features on CPU. Full reply
generation covers 3,024 evaluation episodes / 30,240 turns. No encoder or
original learner parameter changes. Record actual preparation, forward, decoder,
readout-fit, and end-to-end wall/CPU/GPU cost; do not promise an hourly cost before
measurement or occupy the GPU while the formal jobs are running.

## Denominators and interpretation

Each evaluation state has **504 final pairs**, 168 per family; 216 primitive
pairs and 288 composed pairs. Every cell has eight pairs, and every primitive
family/operator group has 24 across three lengths. Paired success requires both
members correct, separately for actions, first bytes, and full replies.
Unpaired final-known accuracy has 1,008 episodes. Other known/unknown query,
unsupported-ASK, and acknowledgement rates retain their actual counted turn
denominators. Report all cells and families plus equal-family aggregates; no
selective remediation or aggregate-only success claim. Probe predictions are
four-class diagnostics, not generated English replies.

| Observation | Supports | Does not establish |
|---|---|---|
| EOS linear refit improves fresh paired decisions beyond original head, initialization, and shuffled-label controls | Useful information is already linearly available; readout fitting, drift, or calibration is a plausible bottleneck | A unique optimizer bug, intact general understanding, or an architecture winner |
| EOS probe succeeds but reply-context probe does not | The learned reply projection may restrict accessible answer information | That information is absent rather than nonlinearly encoded |
| Reply-context probe succeeds but original BOS first byte fails | A decoder semantic-readout/optimization problem remains plausible | That later teacher-forcing exposure bias explains the first decision |
| BOS first byte succeeds but complete reply fails | Later decoding, termination, or exact-output stability needs attention | Reliable full replies, or that first-byte success generalizes beyond this grammar |
| Refits fit their training examples but fail fresh evaluation | Readout adaptation memorized or failed to transfer | A simple readout-only repair |
| Neither trained representation yields a useful probe under this bounded fit | Prioritize shared learning/representation/objective/optimization investigation | Absence of all semantic information, or a specific causal diagnosis |

Only if the first pass leaves the shared-learning branch unresolved, consider
one additional fixed gradient inspection: four fitting pairs per cell at both
historical endpoints, grouped into 21 matched three-family batches per endpoint.
Separate query, acknowledgement, reply, and observation-byte gradients; measure
weighted norms and cosine alignment on shared blocks and the tied token/input
readout embedding, before clipping. This has **zero optimizer updates** and no
per-cell intervention. Large opposing auxiliary gradients identify local tension,
not proof that deleting or reweighting a loss improves learning. Any causal loss
ablation would be a separate prospective experiment.

## Exact local evidence and authentication

- Historical trust anchor: `runs/foundation-study-local-v2/evaluation/summary.json`,
  SHA-256 `188b0d026c4debce596f3cca5a1b64f6f5f0e0e3992dd4174904a33b6c3e87c2`.
  Authenticate its protocol and checkpoint bindings using the historical archive
  reader; no old stream replay is needed. Read only
  `curriculum/checkpoint-000000.pt`, `curriculum/checkpoint-001536.pt`, and
  `mixed/checkpoint-001536.pt` under that historical directory. Confirm both
  recorded initializations match before counting the common state once.
- Exclusion inputs: historical `protected.pt` and `training-transcripts.pt`, plus
  `runs/variant-study-local/data/manifest.json` and its authenticated `history.pt`
  and calibration/main `training-transcripts.pt` and `protected.pt`. Read data
  identities only; no current main checkpoint or score is a diagnostic input.
- Mechanism definitions: `experiments/sequence_training.py:sequence_objective`,
  `experiments/cognitive_credit.py:balanced_query_loss`,
  `experiments/sequence_student.py:SequenceStudent.forward`,
  `experiments/foundation_curriculum.py:generate_pair`,
  `experiments/foundation_banks.py:expected_fresh_cells`,
  `experiments/composition_evaluation.py:PreparedBank.score`, and
  `experiments/foundation_metrics.py`. Hash the diagnostic wrapper, all invoked
  source dependencies, datasets, exact model bytes, and resulting feature cache.

The prior motivation is recorded in `FOUNDATION_VARIANT_IMPLEMENTATION.md`,
`FOUNDATION_PILOT_RESULTS.md`, and `CAPACITY_STUDY_RESULTS.md`. This diagnostic
tests accessible information and local training signals within an already known
finite grammar. It does not test new algorithms or establish useful autonomous
learning. No result automatically selects a loss change, architecture, or model.

## Pre-execution addendum — 2026-09-17 07:48 UTC

This addendum resolves implementation details in the earlier proposal; it does
not revise that historical record or report an executed diagnostic. Only source
text, installed solver code and data/protocol byte identities were inspected.
No checkpoint, current main score, model, solver or test was executed for this
review. Execution remains conditional on the completed main comparison.

### Exact fitted-probe contract

- Cache the normalized encoder states gathered from `context_states` at
  `eos_positions`, not every byte position or merely the last padded position.
  Cache `production_context` separately. Use the checkpoint's exact configuration,
  `model.eval()`, observation-only inputs and a `[batch, turns, 1]` BOS-only
  decoder prefix. Group encoder batches by actual turn count; padding must not
  create supervised turns. A cache entry binds state digest, dataset digest,
  pair/row/turn coordinates, representation name, dtype and tensor bytes.
- Each state/representation gets one four-class affine probe `z W.T + b`, shared
  across all cells and families. Convert cached features to CPU float64. Compute
  an unweighted fit-turn mean and population standard deviation for each feature,
  using only the fitting split. For standard deviation at most `1e-12`, force
  that column to zero in both splits; otherwise use `(x - mean) / std`. Reuse
  exactly those statistics for evaluation and the shuffled-label control. Never
  normalize from evaluation data or choose statistics using labels.
- Define the fitting objective exactly. In each of the 63 cells, average CE
  separately over each present query class 0, 1 and 2, then average those class
  means; add `0.25` times mean CE over acknowledgement class 3. Average the 63
  cell losses, giving each family one third of the data loss. Add
  `0.5 * 1e-3 * (sum(W**2) + sum(b**2))`. The data coefficients total `1.25`;
  do not divide again by `1.25`, number of turns, feature width or parameter
  count. This matches the original class/ACK loss form with diagnostic equal-cell
  weighting; it does not reproduce the historical curriculum's cell frequencies.
  Positive L2 on both weights and bias fixes the softmax gauge and gives a finite
  unique optimum. These numerical constants are prospective choices, not tuned
  values or an assertion that they are optimal.
- Initialize every probe's weights and bias to zero. Set CPU intra/inter-op
  threads to one before computation and enable deterministic algorithms. Pin
  the runtime; deterministic execution is promised only within that declared
  environment, not across arbitrary library versions or hardware. Use one
  `torch.optim.LBFGS.step` with `lr=1`, `max_iter=200`, `max_eval=999`,
  `history_size=20`, `tolerance_grad=1e-7`, `tolerance_change=1e-12`, and
  `line_search_fn="strong_wolfe"`. No restarts, rate search or solver retry.
- A closure-owned counter must refuse a call before performing objective or
  gradient work when 999 such evaluations have completed. Reserve at most one
  additional objective/gradient evaluation for the final fit-only residual
  check: at most **1,000 per probe**, or **12,000 across 12 probes**. Count actual
  started/completed evaluations independently of optimizer state. The inspected
  installed solver is PyTorch `2.11.0+cu128`; `torch/optim/lbfgs.py` has SHA256
  `bce32f11cc29073610d5422a29d55864cbb15cd2a7c847aa5fabbf18ffa6eead`.
  Its line search makes an initial evaluation in addition to its iteration
  allowance, so `max_eval` alone is not a strict objective-work cap.
- A closure exception can leave an unevaluated trial displacement installed because the solver's
  `_directional_evaluate` does not restore parameters in a `finally` block.
  On hard-cap interruption, solver error or nonfinite values, discard that fit,
  record its incomplete status and actual work, and produce no evaluation
  predictions. Do not select a fallback trial, retry or expand the budget. On normal
  solver return, evaluate the returned parameters once within the reserved
  allowance. A finite normal-return endpoint may be scored descriptively even
  when it exhausted the iteration allowance, with an explicit unconverged flag.
  Call a fit converged only if it returned normally and the finite
  final gradient's maximum absolute component is at most `1e-7`. A normal
  return or small loss change alone is not convergence. Report cap/error status,
  final objective, gradient residual, solver iterations and closure work.
- Twelve probes means three states × two representations × true/shuffled
  fitting labels. Probe optimization and its extra supervised examples are
  recorded as fitted-probe work; original learner optimizer updates remain zero.
  Failure of an unconverged probe is inconclusive about available information.

### Shuffle, dataset admission and byte scoring

Replace the proposal's family-level shuffle with one fixed permutation within
each **cell and turn kind** (`turn["kind"]`: `query` or `statement`, as canonically
validated). Enumerate fitting positions canonically by pair slot,
member 0/1 and turn index. Deterministically order that group's positions by
SHA256 of `["foundation-shared-probe-shuffle-v1", 916170003, cell, kind,
pair_slot, member, turn_index]`, breaking digest ties by the original coordinates,
and assign that ordered source-label list to canonical destination positions.
Reuse the permutation across all three states and both representations; record
its digest and changed-label counts. Acknowledgements remain constant and are
not an informative shuffled control. Class counts within each weighted cell
remain unchanged. Keep true canonical rows/replies untouched: shuffle only the
detached fitting-label array after canonical admission. Evaluation labels always
remain true, and no shuffled rows enter the curriculum validator.

Candidate seed derivation must include the declared role's root seed, cell,
pair slot, attempt and a separate procedure/naming/value purpose. Use the
canonical JSON digest and reduce to `0, 2**63)`. Allow at most 1,000 attempts
per pair. `generate_pair` has no primitive operator argument: reject candidates
unless `training_cell` matches the requested copy/advance/direct cell exactly.
Require `structure_split="shared"` at depths 0/1 and `"train"` at depths 2–5,
with fitting `split="train"` and evaluation `split="dev"`. Validate every pair
with `foundation_curriculum.validate_pair` and each role's exact inventory.
Reject within-role and cross-role transcript duplicates as well as exclusions.

The existing `build_evaluation` expects authenticated training-plan anchors and
also prepares held panels; it is not the builder for this new fit/evaluation
sample. Use the canonical generator and admission helpers explicitly. Independent
procedure/name/value seeds establish independently sampled fresh realizations,
not proof of new vocabulary, disjoint value alphabets or a matched same-procedure
intervention. Record program/name/anonymous-value overlap. The diagnostic claim
is generalization to new samples of the known train-admissible grammar; if a
matched realization-only test is desired, it needs a separately frozen pairing
contract rather than being silently substituted during admission.

The first byte's expected token ID is `ord(letter) + ByteCodec.BYTE_OFFSET`
(`BYTE_OFFSET=3`), not its raw ASCII value. PAD/BOS/EOS and all other token
argmaxes count as incorrect; do not strip special tokens before scoring this
decision. The unrestricted BOS-step logits and the first generated token should
agree under the same frozen model/context. Later replies use the existing
`_generate_reply_tokens` and `_exact_replies` semantics, including EOS and trailing
PAD; display decoding alone is not an exact-output scorer. Correct N/Y/I/U is a
semantic class diagnostic, not evidence of correct subsequent words. No correct
reply prefix reaches the encoder or this first decision.

### Provenance, work limits and remaining launch gates

Call `foundation_historical_archive.verify_archive` with the pinned historical
summary and `ExpectedEvidence(input_count=424, source_count=79, ...)`. It hashes
all bound historical files and authenticates their records; it does not
deserialize models, execute archived code, prove current source equivalence or
restore a historical trainer. “Read only three checkpoints” above means only
three selected weight deserializations, not only three checkpoint files hashed.
Check the two recorded initialization weight digests against the protocol's
common initialization digest; bind the one loaded initial model to that digest.

Before loading selected weights, authenticate one immutable byte image for each
checkpoint against its archive-receipt digest and pass those same bytes to
`weights_only=True`. Validate the exact envelope, arm/update/configuration,
recorded weight digest, finite tensor keys/shapes/dtypes and strict state loading.
Compare all invoked model, packer, decoder and objective repository source files
with their authenticated historical copies. Any difference needs an explicit
reviewed compatibility decision or an isolated historical runtime; archive
authentication alone is not permission to bypass a trainer's source contract.
The diagnostic neither restores the historical optimizer nor mints a continuation
token. Pin current diagnostic dependencies, installed solver bytes, runtime and
feature artifacts separately, and check model weight digests before/after work.

The formal data manifest must have the independently supplied expected hash
`d0170189ffa373c68492972df83ed7e39aaa1c40f2cc5c9520b89454f0672be6`, also recorded
in the formal protocol's data bindings. Authenticate the manifest bytes and its
named transcript inventories before decoding those inventories. A manifest's
own neighboring preparation receipt is not an external trust anchor;
`foundation_variant_data.load` alone does not establish one. Union the historical
protected/training inventories with formal history and both stages'
protected/training inventories, recording every file and union digest. This
limited reader need not deserialize formal bank observations, checkpoints or
scores, and must not call `verify()` to regenerate current study data.

The stated 6,048 encoder episodes, 3,024 evaluation episodes and 30,240 evaluation
turns are correct planned sample counts. They are not fixed FLOPs or elapsed
cost: record padded byte positions, actual batch sizes, decoder recurrent steps,
reply tokens attempted and early termination. Decode only from cached contexts;
calling `PreparedBank.score` again would add encoder work. Enforce the declared
12-probe iteration/evaluation ceilings and a separate caller wall allowance;
an operation already executing may finish after its deadline. A partial run is
reported as partial, without automatic replay. Cache and decoder passes require
their own intent/completion accounting, including failed attempts.

If the conditional gradient inspection is later authorized, predeclare exactly
42 matched three-family forward groups across the two endpoints, 1,008 total
episode forwards. Reproduce per-family query CE plus `0.25` ACK, `0.1` reply
and `0.1` observation losses, then the original mean across three families.
Reply and observation losses are means of per-turn token means, not one global
token mean. Use four component gradients from the same forward graph where
possible; record actual backward evaluations, with zero optimizer calls.
Deduplicate the tied token/observation-head weight by parameter identity when
forming norms/cosines; define zero-norm cosines as null. Pin batches and weights
before/after inspection. This work is additional to the first-pass counts.

No numerical solver choice remains for execution-time tuning. The remaining
decisions are whether the main results justify this diagnostic, whether archived
forward dependencies match, and what finite caller wall allowance to grant.
Any unresolved gate stops the launch rather than silently changing the contract.

## Isolated solver implementation — 2026-09-17 08:09 UTC

`experiments/foundation_representation_probe.py` now implements the fixed cached-
feature solver as `fit_probe(features, labels, cells=..., kinds=...,
coordinates=..., cell_inventory=..., shuffle=False)` and
`fit_true_and_shuffled(...)`. A result exposes detached `report`, `normalization`
and `coefficients`, plus `logits(evaluation_features)` and
`predict(evaluation_features)`. Evaluation labels are not API inputs. The default
inventory contains all 63 production cells in sorted order; an explicit smaller
inventory is permitted for synthetic tests. This module validates array structure,
not lesson provenance; the diagnostic caller still owns cache authentication and
canonical data admission.

The caller configures CPU intra/inter-op threads to one and strict deterministic
algorithms before tensor work. Import and fitting do not silently change process
settings. Ambient `no_grad` is supported through explicit gradient scopes; ambient
`inference_mode` is refused before fitting preparation. Original
`KeyboardInterrupt`/`SystemExit` exceptions propagate with detached `probe_report`
accounting, plus `completed_probe_reports` when the second paired fit interrupts.
No interrupted fit supplies coefficients or predictions.

All eight focused synthetic tests passed in **1.068 seconds**, **2.765 seconds**
including the test process, recorded in
`runs/foundation-representation-probe-validation-local/attempt-001/report.json`
(SHA256 `58e7b967ca73055eedcdafc945b3f8e1aaaf7b853485f0f4d48019cf7f4ffad1`).
They cover weighting/ridge, shuffle counts and coordinate identity, fit-only
normalization, source-input preservation, separable synthetic signal, hard-cap
discard, ordinary versus interrupted solver returns, and ambient runtime modes.

The ledger records 10 probe-step attempts, five normally completed step calls,
64 internal solver iterations, 1,077 closure attempts / 1,075 completed closures,
and 1,081 objective attempts / 1,080 completed objective-and-gradient evaluations.
Five completed evaluations were the reserved final checks. These totals include
the deliberately forced 999-closure cap, injected failures and interruptions;
one separate direct objective-forward comparison is also recorded. The shuffled
synthetic control returned a finite endpoint with gradient residual
`3.742893600616618e-7`, correctly labelled unconverged under the fixed `1e-7`
criterion; its descriptive result did not trigger retuning or a retry.

The tested solver source SHA256 is
`9eb6bb46a321395e41bfe8d36a85a9a3807a4b202fb6cd53bdd24b303cdf399c`.
Its source, tests and installed solver bytes stayed unchanged during validation;
all 88 frozen formal-study source hashes still match. Source-learner calls,
original learner updates, historical/formal evidence loads and GPU work were
all zero. This validates a reusable diagnostic component, not the historical
BiC diagnostic or a learning claim; the earlier launch gates still apply.

## Component preparation — 2026-09-17, during main scoring

The conditional diagnostic remains unlaunched. Software preparation and fixture
checks do not choose a learner or establish a learning result.

The [source compatibility preflight (archive reference: `../runs/foundation-shared-diagnostic-preflight-local/attempt-001/README.md`)
passed: 18 transitive core dependencies, 29 including admission/metric references,
and all 79 historically declared repository sources match their authenticated
archive bytes. JSON records agree on the three predetermined checkpoint pins,
configurations and weight digests. No checkpoint was opened or deserialized.
Single-byte-image authentication and safe tensor validation still occur only at
the later authorized diagnostic load; runtime equivalence is not implied by
repository byte equality.

The new `foundation_diagnostic_data.py` builder passed
eight fixture checks (archive reference: `../runs/foundation-diagnostic-data-validation-local/attempt-001/README.md`).
Its separate small fixture covers every declared cell and checks canonical truth,
pair completeness, role/ancestry boundaries, disjoint transcripts, deterministic
collision resampling, bounded failure and caller-input isolation. It uses different
root seeds and one pair per cell; no production diagnostic dataset was generated.
All 88 formal-study sources remained unchanged. The report accounts for repeated
canonical generation separately from learner exposures, of which there were zero.

## Execution decision after the complete comparison — 2026-09-17

The [completed six-run comparison](FOUNDATION_VARIANT_RESULTS.md) failed every
predeclared screen criterion in both primary panels, with poor shared fitting
and unstable paired action/reply retention. Execute this already specified
diagnostic on its three predetermined historical states. It will not directly
diagnose the newly tested hierarchy, choose a winning state or change a learner.

The first actual run has a **3,600-second nonpreemptive wall allowance**, checked
before each operation, on local `cuda:0` for frozen features/replies and one CPU
thread for probes. No automatic retry, solver retuning or model promotion is
allowed. Preserve completed and interrupted work in a durable phase journal.
Twelve attempted probes with a discarded/nonfinite endpoint yield a **partial**
diagnostic; a finite normal-return unconverged probe is reported descriptively
with its residual and explicit lack of convergence. Any unexpected failure
stops the runner and leaves the original evidence intact.

The prior full historical archive verification is retained as historical evidence.
The input reader freshly authenticates the required JSON bindings, all 79
historical archived/live sources, the 88 formal sources, diagnostic dependencies,
seven transcript inventory byte images and each of the three selected checkpoint
byte images at its one actual load. This explicitly replaces a redundant full
424-input reread at launch; it makes no new claim about unneeded historical files.
The selected immutable bytes, strict weight/configuration checks and current
runtime remain mandatory. This change concerns read-only verification cost and
does not alter samples, selected learners, budgets or scoring.

The detached feature/decoder component passed eight focused CPU tests on one
tiny fresh model: 12 attempted encoder episodes, 1,241 generated decoder row-steps
and 88 separately counted BOS row-steps. The pure scorer passed eight tests with
no model work. The run journal and source guard passed five pure tests, including
durable intent, partial interruption, elapsed allowance, source changes and
refusal to overwrite an existing attempt. These are component checks, not the
historical diagnostic or evidence of a learning improvement.
