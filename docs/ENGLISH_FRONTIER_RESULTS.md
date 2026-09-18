# Local English training frontier — September 16, 2026

This follow-up uses the MSI RTX 5080 machine to choose the next development
step from actual training. It preserves the released, symbolic, and earlier
English checkpoints. The teacher is disconnected throughout this study.

**Result:** elementary comprehension teaching resolves the familiar-task
learning failure, but flexible language transfer still fails. Seven completed
branches performed **12,000 optimizer updates and 1,536,000 dialogue exposures**
(repeated presentations of the small lesson bank), using 2,571.12 aggregate
worker-training seconds. All training jobs have finished. No checkpoint was
promoted as a general English learner. Machine-readable summary (archive reference: `../runs/english-frontier-local/summary.json`)
and next-development decision (archive reference: `../runs/english-frontier-local/decision.json`)
record the evidence and resulting priority.

## Measured results

At the common 1,000-update point, seed1101 has equal batch size, student
initialization, training bank, and dialogue-exposure count across these arms:

| Training method | Familiar query accuracy | Both reversed-rule answers correct |
|---|---:|---:|
| Original learner | 66.09% | 0.00% |
| Paired examples and balanced query loss | 65.47% | 0.00% |
| Elementary text teaching | 99.69% | 100.00% |
| Text plus causal memory teaching | 98.91% | 97.92% |
| Memory teaching with balanced binding loss | 99.69% | 99.48% |

These are the frozen seed180000 development matrix scores. The first two arms
then stopped under the declared stall rule; the others continued to 2,000.
Text teaching alone is enough to resolve this familiar-task failure in the
tested seed. Adding memory heads is not established as superior.

Full scaffolding was repeated from three independent initializations. At the
fixed 2,000-update endpoint, the separate seed360000 development evaluation
gave these means and sample standard deviations across initializations:

| Evaluation condition | Query accuracy | Complete-pair accuracy | Generated query replies correct |
|---|---:|---:|---:|
| Familiar meanings and wording | 99.82% ± 0.12 pp | 99.83% ± 0.30 pp | 99.69% |
| Familiar meanings, different wording | 51.43% ± 11.33 pp | 11.98% ± 11.28 pp | 47.89% |
| Unseen name–color combinations, familiar wording | 47.11% ± 5.40 pp | 4.43% ± 4.69 pp | 50.21% |
| Unseen combinations and wording | 47.40% ± 5.49 pp | 5.12% ± 4.33 pp | 44.84% |

All three models pass the familiar-task gates. Resetting state or blanking
English yields **zero complete-pair accuracy** in that condition for all three.
All three fail every transfer slice's overall competence gate. These results
demonstrate learned use of familiar English context, not general English or
reliable understanding of newly assigned meanings. The three seeds share the
evaluation scenarios; their standard deviation is not a population confidence
interval or three independent benchmark sets.

The final familiar slice includes two exact training transcripts among 384.
A uniform follow-up removes both members of any overlapping pair, independent
of model answers: 382 dialogues and 381 reversed-rule query pairs remain.
The three-model mean query and pair scores still round to **99.82% and 99.83%**.
All transfer slices have zero exact training-transcript overlap. See the
deduplication check (archive reference: `../runs/english-frontier-local/novel-transcript-check.json`).
The familiar template and meaning support remain known after deduplication.

The targeted binding-loss follow-up improved its **known-binding readout** from
86.80% to 97.73% at 2,000 updates, while unknown-binding accuracy was 99.29%
(99.68% in its matched baseline). It did **not** meet its predeclared transfer
screen: unseen-binding pair accuracy improved only 0.52 percentage points on
the screening matrix, below the required 10 points. On the separate final
diagnostic bank it scored 4.43%, versus 9.64% for matched seed1101 full
scaffolding. Improving an auxiliary readout did not establish better policy
transfer. The early 750-update readout score of 10.84% was an intermediate
training finding, not the final result or proof that the state lacked a fact.

All seven models passed exact CPU session save/reload continuation and remained
unchanged by evaluation. **352 regression tests passed** in 27.499 seconds,
including causal supervision, training/evaluation separation, and exact CPU
optimizer/sampler/head continuation. CUDA training occurred as reported;
bitwise CUDA restart has not been established. Test record (archive reference: `../runs/english-frontier-local/tests.json`)
and source verification (archive reference: `../runs/english-frontier-local/source-verification.json`)
are saved alongside the runs.

## Development decision

Keep elementary role/name/color teaching as a prerequisite, and keep paired
context tests in the promotion criteria. Do not allocate another large run to
the unchanged loss or infer transfer from familiar accuracy. The compute-only
and pairing/balancing controls did not resolve the original failure.

The next controlled change is a **shared name-conditioned binding readout**:
teach a common operation for retrieving a named fact, instead of independent
color classifiers for each fixed name. This is a hypothesis to test, not an
implemented or demonstrated solution. Keep its supervision outside policy
inputs. Then separately vary verified wording coverage, reserving new
combinations before training. Require transfer gains, familiar-skill retention,
accurate uncertainty, and useful learning per unit compute before widening
the curriculum or scaling runs. Full scaffolding's transfer often deteriorated
as familiar loss approached zero, so final-step improvement on the same bank
is insufficient evidence to keep training it.

The released visual, symbolic, and experimental English checkpoints remain
separate. This study improves the research teaching and diagnostic machinery;
it does not merge those abilities or replace the original learning launchers.

## What changed

The unchanged student struggled even to memorize four paired conversations.
An independent 400-update probe scored 50% query accuracy and 0% complete-pair
accuracy on those same eight training dialogues. A temporary mean-pooling
encoder probe also scored 0% complete pairs. These small diagnostics motivated
better teaching before a larger model or longer unattended training.

New, removable training heads teach sentence role, mentioned name, mentioned
color, the current permitted color, and the name–color facts stated so far.
Targets are generated from verified training text, causally, outside the model.
The existing byte encoder and thirteen-region student still produce actions
and response bytes. No parser, symbolic memory table, auxiliary head, target,
or LLM supplies an answer during inference. These are engineered instructional
signals, not evidence that BiC invented a new learning algorithm.

## Design

Each branch uses the same 3,072 six-turn dialogues, with 1,024 per focus
(grounding, revision, uncertainty). There are 3,036 distinct exact transcripts.
Batch size is 128; initialization seeds are explicit; AdamW learning rate is
0.003. Optimizer, sampler, student, and auxiliary-head states are checkpointed
every 250 updates. The candidate banks and source hashes are frozen before
optimization. Checkpoint files and complete histories remain in
`runs/english-frontier-local/`.

- **Baseline:** original loss and individual-dialogue sampling, seed1101.
- **Paired and balanced:** whole counterfactual pairs and equal query-class
  loss weights, seed1101. This changes two things jointly.
- **Text scaffolding:** paired/balanced training plus current-utterance
  role/name/color targets, seed1101.
- **Full scaffolding:** adds causal permission and binding readouts, seeds
  1101, 1103, and 1109.
- **Balanced binding supervision:** same full scaffolding, with equal aggregate
  weight for known and unknown binding targets, seed1101. This follow-up was
  declared after the first study exposed an imbalance; see the
  [binding protocol](ENGLISH_BINDING_PROTOCOL.md).

Branches target 2,000 updates. Baseline and paired branches stop after four
consecutive checkpoints with zero paired correctness on familiar support.
Compare their common 1,000-update checkpoint with the other branches; their
shorter final compute is not an equal-budget 2,000-update comparison. The
single-seed ablations are exploratory, while full scaffolding has three
initializations. CUDA and CPU implementations can differ numerically.

The diagnostic matrix separates wording transfer from binding transfer:

| Names paired with colors | Wording | Purpose |
|---|---|---|
| Familiar combinations | Familiar templates | Can the learner solve the taught task? |
| Familiar combinations | Different templates | Does wording transfer? |
| Unseen combinations | Familiar templates | Can it use a newly assigned meaning? |
| Unseen combinations | Different templates | Can both transfers work together? |

These are **known development benchmarks**, not a pristine final audit. New
seeds and zero transcript overlap do not make their finite grammar new. The
training-time matrix uses seed180000 and 64 dialogues per focus; none of its
four slices overlaps training transcripts. The separate support probe contains
two training transcripts among 192 episodes. Final diagnostic evaluation uses
seed360000 and 128 dialogues per focus, with overlap explicitly counted.

Each final slice includes 384 dialogues and scores both members of each
opposite-rule query pair. Controls remove carried state or English while
preserving labels. Greedy response bytes are scored separately from actions.
Competence requires at least 80% query accuracy, 60% complete-pair accuracy,
60% in every focus, 75% allow/deny macro accuracy, and a 10-point paired
advantage over both controls. These gates do not establish general English.

## Hardware and efficiency

The measured machine has an RTX 5080 with 16 GB VRAM. An isolated benchmark of
the existing six-turn learner measured 1,974 dialogue exposures/second on CUDA
versus 790 on CPU at batch256, about 2.5 times as many. At batch32, CPU was
faster (361 versus 293). The study uses batch128 to retain more optimizer
updates per unit time while benefiting from the GPU. This is a small-model
benchmark, not a ceiling on machine capacity.

Independent training processes run in parallel; their measured training
seconds therefore sum to more than elapsed wall time. Reported exposure
counts count repeated presentations, not distinct lessons. No cloud training,
new model download, or paid external service is used.

## Reproduce and inspect

Run from the repository with its local Python environment:

```powershell
.\.venv\Scripts\python.exe -m experiments.train_english_frontier --output runs/my-scaffold --arm scaffold --seed 1101 --device cuda
.\.venv\Scripts\python.exe -m experiments.train_english_binding --output runs/my-binding-comparison
.\.venv\Scripts\python.exe -m experiments.audit_english_frontier --checkpoint runs/my-scaffold/latest.pt --output runs/my-scaffold/audit.json --save-session
.\.venv\Scripts\python.exe -m experiments.inspect_english_scaffold --checkpoint runs/my-scaffold/latest.pt --output runs/my-scaffold/readouts.json
```

Use a different output directory for a changed protocol. Repeating the same
training command resumes compatible saved states; it does not extend its frozen
planned update count. The experimental binding runner makes a scoped,
process-local substitution of a training-only loss class; it never changes the
active frontier source or the inference model. Do not invoke different wrappers
concurrently in threads of one process. Separate processes are isolated.

Saved `candidate-session.pt` files are unpromoted research prototypes. The
existing English launcher and released visual checkpoint remain unchanged.
