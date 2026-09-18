# v0.2 status: language feasibility and untrained interfaces

## What advanced

- Added independent posterior-temporal comprehension and inferior-frontal production networks, plus a learned semantic bridge to the existing core.
- Added a fixed UTF-8 codec, causal/padding-aware sequence interfaces, and an optional additive language-context pathway into temporal processing.
- Added inference-only `language-check` and analytical `language-plan` commands.
- Documented native English feasibility, home-computer resource constraints, biological design limits, and a staged teaching/evaluation curriculum.

The default combined network has **469,133 parameters**, including the unchanged **162,810-parameter symbolic core**. New language/bridge weights are **untrained**. The delivered checkpoint contains the original symbolic core only.

## Verification performed

**43 tests passed without optimizer updates.** The executed files were `test_model.py`, `test_tasks.py`, `test_language.py`, `test_resources.py`, and `test_readiness.py`. Tests cover causal prefix processing, padding, independent parameters, gradient connectivity, input validation, optional-context timing, language-to-core influence, core-to-production influence, lesion blocking, resource arithmetic, and read-only CLI behavior.

Gradient checks only computed derivatives at random initialization. They did not apply an optimizer step or change model weights. Readiness tests additionally forbid backward calls, optimizer construction/steps, checkpoint writes, and entry into the old Trainer. The final verification run also blocked AdamW construction and checkpoint saves across all selected tests.

The older `test_training.py` was intentionally not rerun in this turn because it performs symbolic training updates. Its passing v0.1 results remain historical evidence in `docs/RESULTS.md` and `experiments/verification.json`; they are not newly claimed v0.2 validation.

The original v0.1 source was loaded directly from its saved release archive and compared against the new core using the same checkpoint and 32 generated episodes with six distractors. All five output heads and all regional activities were **bitwise identical** when language context was omitted. A separate test confirms explicit zero context also preserves the old computation.

The original reference checkpoint's SHA-256 remains:

```text
830e0ec6bd74b51b66acb7017c4c1bae2f95e3503335f84136e2597a636fa153
```

Recorded results:

- `experiments/language_readiness.json`: shapes, parameter counts, finite outputs, unchanged weights, and causal influence checks.
- `experiments/language_budgets.json`: pure arithmetic estimates with training duration left unknown.
- `experiments/v02_compatibility.json`: exact comparison against original v0.1 core computation.
- `experiments/v02_verification.json`: actual test count, scope, and no-training status.

## What was not performed

**No English training, optimizer updates, model downloads, tokenizer fitting, or language-checkpoint saving occurred.** Existing core weights were read for compatibility and inference checks only. The implementation has not acquired English comprehension, meaningful sentence generation, persistent conversation, or a demonstrated benefit from episodic word memory.

## What remains before an English pilot

The teaching design is concrete, but a first learning run still needs an observable object/action environment and encoder; training/validation/test data with held-out roles and compositions; language-aware losses/checkpoints; a free-running generation evaluator; and explicit persistent-memory/reset semantics. A training-throughput and peak-memory measurement on the target PC should set the eventual corpus and time budget.

The biology-inspired priority is meaningful association and useful retrieval, followed by communicative action and retention. The larger language system must earn its complexity through these experiments. Neural-region labels and sensitivity to random text do not establish those capabilities.
