# Local curriculum learning loop

BiC can now choose registered lessons, ask for verified help, train independent
candidates, compare their development performance, and save improvements for
later sessions. Everything needed for the default loop runs locally. This is a
complete loop inside a small symbolic world; it is not general intelligence or
open-ended English conversation.

## Run and continue

Use the project's Python environment from the `brain-in-computer` directory.
On Windows, `.venv\Scripts\python.exe` can replace `python` in these commands.
`Start-BiC-Curriculum.cmd` provides the local launcher.

```powershell
python -m brain_in_computer.learning_loop run --output local-results/curriculum-01 --hours 0.25
python -m brain_in_computer.learning_loop status --output local-results/curriculum-01
python -m brain_in_computer.learning_loop resume --output local-results/curriculum-01 --hours 1
python -m brain_in_computer.learning_loop audit --output local-results/curriculum-01 --count 256
```

`run` requires an empty output directory. `resume` grants additional wall-clock
hours and preserves the saved configuration. The deadline is checked at safe
boundaries; an update, evaluation, or save already underway can finish after it.
Ctrl+C preserves the last completed cycle. Do not run two writers against the
same output directory. `status` reads the saved report without loading weights.

CPU is the default. A new run may use `--device cuda` when the local PyTorch
environment supports the GPU. `--candidates 2 --workers 2` is the default; CPU
candidates can train concurrently, while a single GPU executes candidates
serially to control memory use. Each training batch also contains independent
episodes. Parallel branches do not average incompatible weights.

Useful run options include `--steps 100`, `--seed 101`, and
`--selection round_robin` for a comparison experiment. Changing these settings
requires a new run. Default settings are recorded in `protocol.json`.

## What learns

Each run initializes a separate BiC student with the existing thirteen-region
architecture. Its neural weights learn from examples. It contains no pretrained
LLM and does not replace existing BiC checkpoints. The default student uses a
hidden size of 32; task observations have three time steps, 32 visual features,
four auditory features, four body features, a skill token, and zero feedback.

Visual inputs describe up to three simulated objects. They are numeric scene
features, not photographs. The skill token identifies the requested task;
English prompts and explanations are readable lesson records, not inputs that
demonstrate English comprehension.

| Skill | Exact task | Prerequisite |
|---|---|---|
| Color | Identify the cued object's color | None |
| Shape | Identify the cued object's shape | None |
| Spatial | Find its direction relative to another object | Color |
| Count | Count zero to three present objects | Shape |
| Compare count | Compare first and final object counts | Count |
| Delayed recall | Recall a color hidden in the two later observations | Color |

The simulator independently determines every answer. Lesson validation rejects
altered observations, targets, explanations, and provenance. Reproducible seeds
produce fresh practice without retaining an expanding collection of scenes.

## How the loop directs itself

1. Measure current development performance and unlock prerequisite-ready skills.
2. Choose topics using weaknesses, recent learning progress, practice cost, and
   time since practice. This is an engineered policy, not learned curiosity.
3. Generate new verified scenes, rehearse older skills, and identify a difficult
   example. Incorrect or uncertain answers trigger a registered help question.
4. Train isolated candidates with separate optimizer state. Small variations in
   learning rate and replay fraction explore alternative learning strategies.
5. Evaluate candidates on the same fresh development examples and fixed
   retention examples. Promote only candidates that pass the benefit gates.
6. Save the accepted student, controller state, rehearsal descriptors, and
   exploration state; continue until the invocation's time budget is spent.

Current topics receive fresh lessons each cycle. Other previously practiced
skills rehearse a mixture of stored examples and newly generated variants,
reducing repeated fitting to a small fixed reservoir.

Unpromoted exploration branches can continue learning across cycles, allowing
them to recover from temporary regressions without replacing the accepted
student. Continued exploration is not permission to bypass promotion checks.

Promotion considers accuracy, cross-entropy loss, and probability error measured
by multiclass Brier score. Previously demonstrated abilities receive retention
checks against fixed best-achieved anchors, so many small accepted regressions
cannot silently erase a skill. Brier score alone does not prove calibration.
Candidate efficiency and elapsed compute are recorded separately.

## Tutor and evaluation boundaries

The procedural tutor is the default and needs no model service. For a locally
installed Ollama model, start a new run with `--tutor-model MODEL_NAME`.
The adapter contacts only the local service and does not download models. It
accepts only selections from exact verified explanations and registered
follow-up questions, plus a small verified practice packet contrasting possible
answers. Packet examples enter training only after independent validation.
Unsupported or failed responses fall back to the procedural tutor. The tutor
cannot invent training labels or alter the curriculum.

Training, development, and audit use separate random streams, disjoint nuisance
identity ranges, and distinct prompt families. Development scores guide topic
selection and promotion, so they are not independent final evidence. The
explicit `audit` command evaluates the accepted student and its untrained
baseline with the tutor off. Audit results never enter selection or training.
Repeated model/seed audit output is protected from overwrite. Avoid using audit
results to tune the same experiment: that would compromise their independence.

These partitions measure transfer within the six known rules, including a
nuisance shift. They do not establish new concept discovery, natural vision,
language understanding, or unrestricted reasoning. Measured validation results
belong in `LEARNING_LOOP_RESULTS.md`, separately from this operating guide.

## Persistence, evolution, and extension

`latest.pt` is the resumable checkpoint; `initial.pt` preserves the baseline.
`status.json` summarizes progress, and `protocol.json` records configuration,
environment, curriculum fingerprint, source fingerprint, and policy. Audit
reports are separate JSON artifacts. Checkpoint replacement is atomic.
Rehearsal stores bounded per-skill seed descriptors, not full lesson corpora;
recent cycle history is capped at 128 entries. These bounds keep storage from
growing with every learning iteration.

Evolution is deliberately narrow: candidates may vary learning rate and replay
fraction. The objectives, oracle, curriculum, evaluation rules, promotion gates,
and source code remain fixed. The policy favors verified competence, retention,
reliable probabilities, and compute efficiency. Tasks involve inert simulated
objects, with no pain, deprivation, coercion, suffering objectives, or real-world
actions. These design constraints are not a measurement of subjective welfare.

To extend the world, review and version the generator, independent oracle,
canonical explanations, prerequisite rules, and split tests together. Start a
new experiment after changing protected source or curriculum; resume rejects
such changes. Generated text cannot install a new task, execute code, rewrite
objectives, or promote itself. More compute permits more iterations within the
reviewed world; it does not guarantee increasing intelligence or remove its
representational limits.

Implementation is divided among `curriculum.py` (verified lessons),
`curriculum_tutor.py` (bounded help), `learning_student.py` (weight updates and
evaluation), and `learning_loop.py` (selection, persistence, and command line).
