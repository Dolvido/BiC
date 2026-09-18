# Budgeted raw-English practice controller

[EnglishLoop](../experiments/english_loop.py) joins the existing raw-English
backend to an [explicit allocation policy](../experiments/english_allocation.py).
It can measure development and retention cells, choose the next practice window,
train within caller budgets, and resume the learner and decision state together.
This is an engineering prototype. Its allocation rule is designed by us; a
learning benefit over fixed joint practice has not been demonstrated. It does
not promote checkpoints, call an external tutor, change its code, or establish
broad intelligence. The larger home-learning objective remains open.

## Admission and use

Supply your own authenticated canonical composition banks. Training has the
shape `train_banks[family][turn_count] = complete_adjacent_pairs`, with the three
admitted families `color`, `count`, and `switch`. Each evaluation role is a mapping
of names such as `seen/color/t8` to complete canonical pairs from one family and
one length. Both `development` and `retention` must cover every admitted family.
The backend verifies canonical provenance, training admission, pair integrity,
and observation-only input boundaries. Evaluation requires `dev` admission and
rejects every supervised composed query whose ancestry belongs to the audit
partition, including earlier queries.

Keep sealed study audits outside this controller. Evaluation transcripts are
automatically protected from training; pass any additional historical or reserved
transcript digests explicitly. Callers still own the prospective bank design and
its structural boundaries. Canonical authentication does not establish that a
small synthetic grammar represents general English competence.

The following is an API example, not a command to run a study. The bank variables
must already contain your admitted data; it does not load any formal study bank.

```python
from pathlib import Path
import time

from experiments.english_allocation import AllocationConfig
from experiments.english_loop import EnglishLoop
from experiments.realization_backend import RealizationBackend
from experiments.sequence_student import SequenceConfig

evaluation_banks = {"development": development_banks, "retention": retention_banks}
backend = RealizationBackend(
    train_banks,
    mode="fresh",
    protected_transcripts=reserved_transcript_digests,
    evaluation_banks=evaluation_banks,
    device="cpu",
    config=SequenceConfig(max_turns=12),
)
loop = EnglishLoop(
    backend,
    AllocationConfig(mode="joint", window_updates=16, min_family_microbatches=8),
    max_seen_transcripts=100_000,
    history_limit=32,
)
checkpoint = Path("runs/my-english-loop/loop.pt")
loop.save(checkpoint)
report = loop.run(max_updates=16, deadline=time.monotonic() + 60)

# Resume in the same declared source/runtime configuration and with the same banks.
loop = EnglishLoop.load(
    checkpoint, train_banks,
    protected_transcripts=reserved_transcript_digests,
    evaluation_banks=evaluation_banks,
    device="cpu",
)
```

The controller takes exclusive ownership of an unbound backend: do not train,
restore, or save that backend independently afterward. An existing envelope
requires `load`; `save` refuses to overwrite one as a new run. Changing banks,
sources, allocation options, or the declared execution profile requires a new
declared run. CUDA use requires configuring the strict execution profile before
CUDA initialization; there is no silent CPU fallback. A recorded profile and an
exact serialized reload do not guarantee identical future arithmetic on every
device or workload.

## What determines practice

Every optimizer update contains three family microbatches. `joint` keeps equal
practice across families and is the baseline against which allocation benefit
must be tested. `progress` starts jointly, then uses the latest two complete
observations. In each development cell it takes the smallest change among final
opposite-answer action pairs, final reply pairs, and known-query accuracy, and
penalizes an increase in unsupported ASK. Family scores average those cell
changes and divide by the same measured window cost. This is observational
progress per shared window second, **not causal efficiency of a family**.

Each development and retention cell also retains its best observed rates for
those three measures and its lowest unsupported-ASK rate. Regressions beyond the
declared tolerance take precedence over positive progress. The policy ranks
affected families by their worst cell shortfall and cycles remaining slots among
them. With no alarm and positive progress, it favors the highest score; with no
positive progress, it returns to joint practice. These references are noisy
best-observed anchors, not validated mastery thresholds. A family floor applies
to every *completed* window; an interrupted prefix may still owe coverage, which
is visible in `status`.

Full per-bank evidence preserves paired counts, reply correctness, known and
unknown queries, unsupported ASK, per-turn results, and calibration measures.
Aggregate accuracy cannot replace this vector. With `mode="fresh"`, revisiting a
family means new realizations of its admitted recipes. The family floor is
fresh rehearsal, not exact-example replay and not a guarantee of retention.

## Commit, resume, and budgets

One atomic envelope contains weights, optimizer state, consumed realization and
sampler state, the backend schedule/cursor, allocator counters and plan, partial
evaluation cursor/results, and bounded history. Training commits at most 16
updates at a time; each evaluated bank commits separately. Resume continues a
pending schedule or evaluation without intentionally repeating committed work.
Stored complete and partial evidence is revalidated against canonical bank
denominators; an instance-local cache stores at most 128 exact validated response
digests. Stale writers, changed sources/runtime, and mismatched counters are
rejected.

If an operation or save fails, recovery uses the last committed envelope. If
publication succeeded before an interruption, the file digest determines which
envelope was actually published. Unresolvable publication or failed rollback
blocks continuation until explicit reload; retained/discarded counts become
unknown, and `status` distinguishes in-memory updates from verified state.

`run` requires an update limit, an absolute monotonic deadline, or both.
`run_for_hours(hours)` provides the deadline form. A zero update budget starts no
scoring or training. Limits are cooperative: an in-flight training step or bank
score and its current snapshot/save can finish after the deadline. Evaluation
may remain pending when the update limit is reached. This is not a hard real-time
deadline or a durable lifetime compute-hour budget.

The invocation report includes wall time from public `run` entry, including lock
wait, validation, training, scoring, commits, and recovery. Failed operation times
are included; their separate fields are subsets, not additional costs. Retained
policy window cost includes successful backend training-call and evaluator
intervals, excluding outer envelope writes, orchestration, and rolled-back work.
The in-memory `last_report` distinguishes retained work, discarded completed
updates, failed attempts, and uncertainty. A hard process death can leave
unrecorded physical work; durable counters must not be presented as its total.

## Storage and validation limits

The allocator stores the latest two complete observations, compact best-observed
cell references, and a caller-bounded window history. The learner's exact
seen-transcript digest index still grows. `max_seen_transcripts` conservatively
reserves room before another chunk and stops at its finite bound; it does not
provide indefinitely compact learning. Snapshot cost grows with retained state.
A disk-backed or compacted novelty index, longer-lived memory/consolidation,
broader grounded tasks, external tutor admission, and measured decreasing tutor
dependence remain separate work.

All **17 controller tests and 12 allocation tests passed**, including the updated
joint/progress continuation case. The subsequent full integration suite passed
**743 tests in 117.045 seconds**; all 71 live capacity-study source hashes stayed
unchanged. Recorded integration evidence (archive reference: `../runs/english-loop-validation-local/integration.json`)

The [controller tests](../tests/test_english_loop.py) verify canonical admission,
exact weights/AdamW/stream/cursor and decision continuation for tiny CPU learners,
atomic publication and rollback, partial evaluation resume, corrupted saved
evidence rejection, failed-work accounting, and resource/deadline stops. Measured
wall times and their normalized progress magnitudes are excluded from exact
continuation comparisons. [Allocation tests](../tests/test_english_allocation.py)
also check deterministic policy decisions and coverage/reference accounting.

Earlier focused controller checks consumed 66 completed updates and 398 drawn
episodes, including failed/discarded work. The full-suite controller subset added
35 completed updates and 212 drawn episodes: **101 updates and 610 episodes**
across these controller checks. Other existing tests perform separate small CPU
training; this is not a physical-work total for the entire repository suite.
The full GPU controller path has not been validated. These checks do not compare
learning quality, establish useful self-direction, or demonstrate broad retention.
No new GPU experiment, external tutor use or formal study result is claimed here.
