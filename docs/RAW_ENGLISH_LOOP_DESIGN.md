# Raw-English learning backend: integration boundary

The new [backend](../experiments/realization_backend.py) provides budgeted,
restartable training and development evaluation around the existing sequence
learner. It is **not a complete autonomous learner**, a promoted checkpoint or
evidence that the current curriculum teaches broad intelligence. It is separate
from the completed realization comparison and does not change its frozen sources,
protocol, samples or results. Its implementation validation includes CPU tests
and the separate CUDA checks described below.
The [17 focused tests](../tests/test_realization_backend.py) cover exact learning
state resume, consumed-batch rollback, failures before and after publication,
deadline/save accounting, stale writers, audit-ancestry exclusion and progress
denominators. They passed in 5.028 seconds; all 64 live-study source hashes remained
unchanged. This is engineering evidence, not an adaptive-learning result.

### CUDA checkpoint recovery and numerical repeatability

The GPU checkpoint probe (archive reference: `../runs/realization-backend-validation-local/probe.json`)
used 32 uninterrupted updates versus 16 updates, save/load, then 16 more.
The saved state loaded exactly, and both final runs consumed the exact same
realizations, sampler states, exposure counts and schedule. Their weights and
AdamW moments differed, with maximum absolute weight difference 0.00517823.
The strict numerical comparison therefore failed; it has not been relabeled
as a passing GPU continuation test.

A separately declared uninterrupted repeat (archive reference: `../runs/realization-backend-validation-local/repeat-control/probe.json`)
also differed from the first uninterrupted run, with maximum difference
0.00949482. This shows that reload is not necessary for numerical divergence
under these settings. It does not identify the responsible operation or exclude
an additional restart effect. CPU continuation tests passed, but bitwise CUDA
training equivalence is not guaranteed by this backend.

These checks consumed 96 physical updates and 9,216 episodes in total, separate
from the formal study; they performed no capability evaluation or promotion.
All 64 frozen study sources and study inputs stayed unchanged. The recorded
Torch 2.11/CUDA 12.8 settings had deterministic algorithms disabled and no
`CUBLAS_WORKSPACE_CONFIG`. Future comparisons must declare their numerical
execution settings and validate that profile before training. PyTorch documents
opt-in deterministic algorithms and their possible performance cost in its
[reproducibility guidance](https://docs.pytorch.org/docs/2.11/notes/randomness.html);
that general guidance does not diagnose this particular discrepancy.

### Subsequent strict-profile execution checks

The separately declared [execution profile](../experiments/execution_profile.py)
configures deterministic algorithms before CUDA initialization, rejects settings
drift, disables TF32, and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Its
[seven focused tests](../tests/test_execution_profile.py) cover explicit startup,
late configuration, conflicting settings and rejection without silent repair or
CPU fallback. The profile source SHA256 is
`dcf995cb99a5611a168f4b3284004b773de4a7a555eb7a5f84796cf3575a5d73`.

The coverage receipt (archive reference: `../runs/capacity-validation-local/coverage.json`) binds
successful checks for widths
96 (archive reference: `../runs/capacity-validation-local/strict-profile/probe.json`),
192 (archive reference: `../runs/capacity-validation-local/w192/probe.json`) and
256 (archive reference: `../runs/capacity-validation-local/w256/probe.json`). Each used the same
four-layer architecture and verified exact initial and saved midpoint state,
checkpoint reload, subsequent continuation, and an independent uninterrupted
repeat across separate CUDA processes. Weights, AdamW state and the consumed
curriculum state matched; explicitly declared timing fields were excluded.
Every domain consumed all three 8-, 10- and 12-turn length buckets.

Width 96 used a 32-update endpoint, while widths 192 and 256 used 16-update
endpoints. The uninterrupted, split and repeat paths consumed **192 physical
updates and 18,432 episode exposures** together, outside formal learning-study
budgets. These were short execution checks on a tiny canonical bank, with no
capability evaluation or model promotion. They establish repeatability for the
recorded local cases, not a guarantee for longer training, every input shape,
different hardware or other software releases. The earlier failed default-profile
receipts remain unchanged; this later engineering result does not erase them or
demonstrate improved general learning.

## Implemented interface

`RealizationBackend` accepts canonical training banks, fixed/fresh realization
mode, protected transcript digests, and named evaluation banks under
`development` and `retention`. Both evaluation roles require canonical `dev`
rows. Audit rows are rejected. All supplied evaluation transcripts are added to
training exclusions, and their exact identities/configuration enter the backend
checkpoint contract.
Canonical `dev` admission alone is insufficient: every known composed query,
including earlier questions, is checked for forbidden audit ancestry.

It wraps `RealizationTrainer` and `PreparedBank`, preserving their raw-observation
input boundary. Parsed state, provenance, targets and family names remain outside
model observations. Reply evaluation starts from BOS only. There are no tutor,
network, download, promotion or architecture-selection calls.

The caller supplies a finite schedule: one three-family microbatch tuple per
optimizer update. Families may repeat, so a future controller can allocate
practice or replay through the same operation. A schedule contains at most 4,096
updates. Pending work cannot be silently replaced; after it is exhausted, a
caller may install a new schedule. The backend chooses no lesson priorities.

```python
import time
from experiments.realization_backend import RealizationBackend

# train_banks, protected and dev/retention rows must already be authenticated
# for a new declared run. Use a separate output directory from frozen studies.
backend = RealizationBackend(
    train_banks, mode="fresh", protected_transcripts=protected,
    evaluation_banks={"development": development, "retention": retention},
    device="cpu",
)
report = backend.train_chunk(
    [("color", "count", "switch")] * 32,
    max_updates=32,
    deadline=time.monotonic() + 60,
    commit_interval=16,
    checkpoint_path="runs/new-backend/latest.pt",
)
progress = backend.evaluate("development")
```

`snapshot()` and `restore(payload)` operate in memory. `save(path)` and
`RealizationBackend.load(path, train_banks, protected_transcripts=...,
evaluation_banks=...)` persist and recover the same learner. Loading reconstructs
the configuration and optimizer recipe from the checkpoint while checking the
supplied bank/exclusion/evaluation identities. Continue a pending schedule with
`train_chunk(max_updates=..., deadline=...)`, without supplying it a second time.

## Transaction and budget semantics

Each checkpoint contains the learner weights, AdamW moments, exact consumed
realization stream, samplers, occurrences, collision records, exposures and
curriculum schedule/cursor in one atomic file. Cursor validation checks optimizer
updates and consumed family counts. The checkpoint also binds the dependency
source hashes, including this adapter, and prepared evaluation identities.
Snapshots are isolated copies.

The default commit interval is 16 completed updates. Installing a schedule commits
its cursor before training. Subsequent commits occur at the interval and at a
normal deadline/update-bound stop. A same-directory temporary file is flushed
and atomically replaces the checkpoint; a writer lock and last-loaded file digest
prevent a stale backend from overwriting a newer writer. An existing file must
be explicitly loaded rather than treated as a new run.
If an interruption occurs immediately after atomic publication, the backend
checks the actual file digest and retains the newly published commit. It does
not roll memory back behind a newer durable checkpoint. If publication cannot
be determined, further work is blocked pending an explicit reload.

If a step consumes part of a batch and fails, or a commit fails, the backend
restores the last committed weights, optimizer, consumed stream and cursor.
Successful updates since that commit are **discarded**, not reported as retained
training. `last_report` exposes discarded completed updates, failed step attempts,
their measured step time and rollback time even when an exception escapes. The
caller must retain that report if it needs a durable physical-work ledger. A hard
process or machine termination can lose this uncommitted accounting; exact
physical work after the last durable checkpoint is then unknown.

`deadline` is an absolute monotonic timestamp; `max_updates` is a per-call bound.
At least one is required. Wall-budget checks include source validation, generation,
packing, optimizer work, snapshots and saves. No new step starts after the deadline.
An in-flight step and its final commit may finish afterward; the returned report
includes measured wall time and overrun. This is a safe-boundary budget, not a
hard real-time deadline or a fixed maximum number of overrun seconds.

The report separates step, commit and rollback timing, with total invocation wall
time including all overhead. CUDA steps are synchronized around measurement;
step time also includes generation and validation, so it is not isolated GPU
time. Checkpoints retain the accumulated timing of retained
steps, while complete invocation and discarded-work timing are caller-visible
reports. Saving every update is optional and is not the default. The small CUDA
checks establish execution and the repeatability limit above, not a sustained
capacity or throughput frontier.

## Progress is a vector, not a promotion score

`evaluate(role, names=..., batch_size=..., control=...)` returns each named bank's
full evidence and a separate progress vector. It preserves:

- Final opposite-answer pair correctness and denominators, plus other eligible pairs.
- Known-query correctness and unknown recall/precision with their actual counts.
- Unsupported ASK predictions on known questions.
- Free query replies, final reply pairs, parseability and action/reply agreement.
- Query loss, probability error and per-turn paired/query results.

There is no aggregate mastery number, candidate winner or automatic promotion.
Blank and reset-history controls retain the original targets. Evaluation neither
updates weights nor consumes curriculum samples. A future controller must define
prospective selection/retention rules over this evidence, preserve difficult
domain/length slices and compare allocation against a fixed-practice baseline.
Correct unknown responses must not substitute for acquired known-state reasoning.

## Remaining work toward the user's objective

The older symbolic loop already has wall-hour budgets, isolated candidates,
development selection, bounded per-skill rehearsal and optional local tutoring.
Its measured benefit concerns six numeric simulator tasks. Those selection rules
and accuracy thresholds are not evidence of a useful raw-English controller.
The backend supplies the execution boundary. A separate
[EnglishLoop controller](ENGLISH_LOOP_CONTROLLER.md) now adds fixed joint and
engineered progress allocation, minimum family coverage, per-cell best-observed
retention references, and atomic learner/decision-state resume. Those references
are noisy observations, not mastery thresholds. CPU engineering validation does
not establish adaptive learning benefit, learned self-direction, or a complete
home learner; a matched comparison with fixed practice is still required.

The fresh stream stores compact recipe/occurrence information, but its set of
previously encountered transcript hashes still grows with consumed examples.
The recipe set, grammar, value domains and context capacity are also finite.
Long runs need an explicit storage/index policy or a verified namespace design
that preserves exclusion guarantees; simply dropping old hashes would weaken
the current contract. This adapter is not indefinitely bounded memory.

The existing local tutor selects verified symbolic practice packets. Its
explanation strings are metadata, not demonstrated free-form teaching to this
learner. A future English packet provider must admit canonical training lessons
through independent verification, preserve procedural fallback and keep the tutor
outside policy evaluation. Compare tutor-off, procedural selection and local-LLM
selection under declared learner/teacher costs. Declining help requests alone do
not establish growing independence; measure newly acquired, retained skills with
and without assistance.

Still separate are broader curriculum mechanisms and wording, grounding in
observations/actions, longer usable memory, the untrained regional sequence
adapter, a nontechnical continuing-learner interface and a unified checkpoint.
Candidate evolution must preserve truth, evaluation boundaries and useful
retained behavior rather than optimize capability growth alone. The full
[home-learning objective](HOME_LEARNING_ROADMAP.md) stays open beyond success
on these synthetic families.
