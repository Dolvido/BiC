# Offline home learner: operational and development handoff

The current priority is local, offline learning with the installed English tutor,
a repeatable home training command, and clear continuation instructions. The
larger research objective is useful reusable learning with retention and good
benefit per compute hour. General English comprehension and autonomous general
improvement have not been demonstrated.

## User entry point

The user-facing folder is:

`<BiC user folder>`

It contains `Start Offline BiC.cmd` and `OFFLINE BIC - READ ME.md`. The command
uses this project's installed `.venv\Scripts\python.exe` and
`experiments.offline_home`, which delegates actual training to the unchanged
`experiments.home_learning` launcher and `continuous_tutor_campaign_v3` owner.
The offline front end has no separate learner, optimizer or selection policy.

The durable profile is `runs/home-learning-local/session.json`. Always read its
current pinned campaign and last invocation; do not infer the retained learner
from the newest experiment directory or this document's historical counts.

The exact installed tutor is `ministral-3:3b`, digest
`f04aa1c738f64e13c625b82ae92504fc0260fa6723b509ed1ece0fa188179b1d`.
The actual setup check found 2,953,840,808 bytes of local model data, GGUF
Q4_K_M. The transport uses only `127.0.0.1:11434` and rejects remote model metadata.
No download or cloud fallback is configured. This does not change network access
for the Ollama application or other processes; disconnecting the computer's
internet connection is compatible with the intended workflow.

Before a new session, open installed Ollama and run setup check. The home launcher
checks current process ownership and refuses overlapping project learners,
unresolved previous work, changed frozen sources or mismatched checkpoints.
Status does not load models or contact services. Training requires CUDA on this
MSI RTX 5080 machine. The existing environment and model are already installed.

The time argument is the controller's allowance, not an exact wall-clock promise:
local checks/freeze run separately, and whole-stage reservations can end a
session early. No silent restart or retries are allowed after unknown work.

## What the retained loop does

The local LLM selects constrained chapter recipes based on development evidence.
An independent compiler verifies teaching semantics and excludes protected
evaluation examples. Two native branches practice, receive common tutor-free
practice, restore full model/AdamW state and generate their own actions/replies
for evaluation. Engineered progress/retention rules choose a continuation or
retain the parent. Tutor output cannot execute code or directly modify weights.

Training covers restricted English instructions over color, count and switches.
It does not yet accept arbitrary prose, internet content, documents or live chat
as trustworthy teaching data. Tutor usefulness is measured against ordinary
practice; it must not be assumed from successful API calls.

Initial home profile validation succeeded at update 9,112 after seven completed
cycles, with cycle eight already authored/compiled. The inherited physical work
was 3,024 updates and eight completed tutor requests; retained lineage gain was
1,296 updates from the campaign's original update-7,816 parent. These counters
have different meanings and must not be added or substituted for each other.

The actual bounded home validation then completed cycles eight and nine:
864 physical updates, 82,944 episode exposures, and two new local tutor requests
(for cycles nine and ten). The controller used 770.969 seconds of a 1,260-second
allowance; launcher setup and completion brought its measured total to 772.891
seconds. Both completed cycles were independently recounted. Cycle eight kept
the parent; cycle nine retained the procedural continuation at update **9,328**
after all 128 retention conditions passed. The tutor branch did not meet its
benefit threshold. Selected joint development/retention changed from 271/324 and
299/360 to 272/324 and 300/360; these are modest development changes, not broad
English acquisition or a fresh audit.

Cycle ten's local tutor decision and compiled lessons are saved. The controller
stopped before starting its worker because insufficient stage allowance remained.
A separate actual front-end invocation with a 144-second allowance verified the
menu-to-child resume/ownership path and budget boundary. It performed zero
updates and zero teacher requests, preserving the committed state byte-for-byte.
The current profile points to that completed boundary check and has no unresolved
home invocation. That saved state is ready to resume once other project learners
have finished. It does not establish that the computer is idle: separate research
and timing jobs can run without changing the saved home profile.
Use Status and the launcher's process check before starting a home session.

Setup evidence: `runs/offline-home-setup-local/attempt-001/report.json`, SHA256
`32909e03d771da061c92af57a602bf5be7b8cf55d6c5fda52f66ce47ef5c2164`.
It pins the actual invocations, independent cycle reports, teacher results and
front-end sources. The GUI command's Check setup/Status/Exit path was exercised
on the actual Windows machine; six mocked front-end tests and eight focused
home-launcher tests passed. Preserved failures are documented in the receipts.

## Continue development safely

1. Read the current home profile, latest invocation receipt, pinned campaign
   launch/summary/committed state, and `runs/foundation-planning-local/continuation-variant-v1.json`.
2. If an invocation is pending, inspect its receipts and actual process identity
   before doing anything else. Preserve known work and ambiguous outcomes. Never
   delete an unexplained lock or repeat an uncertain teacher/GPU request.
3. Read `docs/CONTINUOUS_TUTOR_RESULTS.md`, `docs/DEFINITION_BASIS_RESULTS.md`,
   `docs/DEFINITION_TUTOR_PAIR_RESULTS.md` and
   `docs/COMPLEMENTARY_COMPOSITION_RESULTS.md` and
   `docs/SUSTAINED_COMPOSITION_RESULTS.md` for measured capability limits.
4. Independently verify completed new v3 cycles with
   `runs/continuous-tutor-campaign-analysis-local/recount_v2.py`, pinned SHA
   `1dd1b15a3083f11b0b7b20c50efe11fd69a86ff78b5bdaf1af8beabd2981066f`.
   It needs the exact launch, trained-before state, decision, ready-after state
   and worker-summary pins. Do not reuse the historical terminal reader unchanged:
   its earlier attempt/cycle list is fixed.
   On Windows, invoke this unchanged reader through its extended absolute
   `\\?\C:\...\recount_v2.py` path and pass `--campaign` through the same extended
   absolute prefix. New home invocation paths can exceed the ordinary Windows
   path limit. Cycle nine's initial path failure and successful path-only retry
   are preserved under `runs/offline-home-analysis-local/`; the raw artifacts and
   reader mathematics were unchanged.
5. Use actual acquisition, transfer, retention and compute results to select the
   next general learning experiment. Keep its held-out examples out of teaching
   and avoid repeatedly tuning individual weak cells.

The completed complementary comparison made 1,296 physical updates in 568.235
seconds; neither branch qualified for retention. Its fixed longer continuation
then completed 5,184 physical updates and 497,664 episode exposures in
2,174.875 seconds, with zero new teacher requests. Both completion artifacts and
the independent recount passed. The report is `docs/SUSTAINED_COMPOSITION_RESULTS.md`;
the raw run remains `runs/sustained-composition-local/attempt-001`, launch SHA256
`4bd912311f64cf25939aa7eb437bfd09e265d23ed2b9ab7911e32ef237cbe40d`.
Curriculum practiced binding/revision improved to 8/60 and 17/60; development
reached 11/120 and 35/120 across the three families. Complete held-out composition
and sequence stayed zero. Both branches failed retention and adoption screens.
The home checkpoint remains separately retained at update 9,328.

`docs/SUSTAINED_COMPOSITION_NEXT_DECISION.md` records the conditional next
learning experiments before inspecting this trial's new scores. The measured
late gains support a continued-practice control. The next prospective protocol,
`docs/SHARED_OBJECTIVE_TRANSITION_PROTOCOL.md`, compares the same curriculum
update-13,000 parent and lessons with auxiliary objective weight 0.3 versus zero.
Its transition adapter passed nine metadata checks and exact CPU/CUDA
continuation proofs, including full AdamW and frozen auxiliary-state checks.
Those proofs performed nine tiny CPU updates and eight actual-parent GPU updates,
respectively. The runner passed ten pure checks and independent static review;
the independent reader passed eight pure checks. The fixed study has completed
under `runs/shared-objective-local/attempt-001`, launch SHA256
`f6670f922676306500d6f97e91c83a2dbdb171f2ae3293cebecc8fcf623d804e`.
It performed all 5,184 updates in 2,139.969 seconds within the 3,600-second worker
allowance. Summary SHA256 is
`6edaf731cd155e3906727a31c4215b48213c424e03d16af781be6ceb63110c2e`;
completion SHA256 is
`dcc734c69f424c2b5058ef5d5fb4467413433562fc9094059a540915e3dbdb3d`.
The first independent recount verified all 128 bank endpoints, 208 score files
and 5,184 step reports; its SHA256 is
`b1edee2751047dd3be93e6e5e32f9c9e6e51f531653476cb478f66178703e417`.
Removing the auxiliary failed the declared benefit screen (-2.0833 percentage
points overall). Revision acquisition improved with continued practice, but
complete transfer remained nearly absent and both branches failed retention.
Keep the 0.3 objective as baseline and the home learner at 9,328. See
`docs/SHARED_OBJECTIVE_RESULTS.md` for every trajectory and the next learning
decision. Check the actual process/session and planning record before starting
other work; completing this study does not imply that later timing jobs are idle.

`experiments/referenced_definition_lessons.py` produced a 489,721-byte chapter
index instead of another 72,853,828-byte copied image collection. All 108 chapters
were reconstructed with exactly the original image hashes. Existing files were
preserved. This adapter is not integrated into training and has no measured
speedup; its cache bounds entry count, not total Python RAM.

The additive `experiments/immutable_lesson_cache.py` passed eight focused pure
tests and the actual lesson-input comparison, but was slower than ordinary
loading: control 83.859 versus 34.063 seconds; curriculum 108.593 versus 34.844
seconds, including boundaries and common verification. It is not integrated.
The fixed-order receipt is `runs/immutable-lesson-cache-analysis-local/attempt-001/receipt.json`;
it does not establish end-to-end training throughput. A separate bytes-backed
version, `experiments/immutable_lesson_bytes_cache.py`, passed nine synthetic
tests and its first actual-data comparison. All 5,184 returned images and both
ordered streams matched. Inclusive JSON-path times were control 32.641/26.953
seconds and curriculum 33.109/30.766 seconds for reference/cache respectively.
The receipt SHA256 is
`d3893067aba92257d5fdfc3d6d798df98247622b6c9adb65c46e6a78ea7f8c04`.
These isolated, fixed-order savings do not establish training throughput; no
cache integration has occurred. A separate profiled two-update diagnostic
confirmed 1,639 source opens per update and substantial copying work. Its
receipt SHA256 is
`47bbbe87a2e16c9e8b16528f8edfabbaa5ca1808f0c626faad8dade84013dcf9`.
See `docs/SHARED_STEP_ATTRIBUTION_RESULTS.md` for costs and limitations.

The next learning direction is fixed in `docs/SHARED_RATE_TRANSITION_PROTOCOL.md`:
same final objective-control parent at 15,592, rate 0.0003 versus 0.0001, auxiliary
weight 0.3 in both, unchanged whole curriculum and original preparation, four
passes per arm. The explicit rate-transition adapter is implemented and passed
nine pure metadata tests plus independent source review. Its validation receipt
is `runs/shared-rate-continuation-validation-local/attempt-001/report.json`, SHA256
`777e98d6e8982be6992f49e2edfd536bfb7a3d85503a88a3ca8693bfa65795f5`.
Both numerical proofs now passed on their first invocations: CPU eight updates
in 15.937 inclusive seconds, and RTX 5080 eight updates in 34.625 seconds.
Weights, optimizer moments and individual steps, evidence and historical work
matched the respective references exactly through two updates and a restart at
each rate. These are validation updates, not retained learning or a rate benefit.
CPU report SHA256:
`2a7bd8c5a6eca436ac8d7a094671ddda16922002d13c05a6218e23f562b07f9c`;
CUDA report SHA256:
`c40b842e6a19158112c7ac8597c967b15d8914740efbd9604b80b4c9f70e375a`.
Both corresponding completion markers passed their 180-second inclusive bounds.
The independent reader passed eight synthetic tests and the saved-parent JSON
preflight, including all 16 metrics and five retention references. The runner
passed all 11 focused tests. The comparison freeze completed with one metadata
archive decode, zero models/updates and 2.484 wall seconds. Launch SHA256 is
`057a6a06932749fd712dd10d32b9e68e6625d2db510aa739f8302af4a6bc21cf`.
Independent admission of actual frozen parent metadata passed; receipt SHA256
`4691d19901bb5ab44d9f4a18b6d30f3616d0978bdeff0a3b267de9c45834c180`.
The single declared comparison started on the RTX 5080 on September 17 at
22:13 EDT, exec session 37135, worker PID 24116 (with venv bootstrap PID 31624).
It has one inclusive 3,600-second allowance for 2,592 new updates per arm.
That process is now terminal, exit 0; do not restart it. It completed all 5,184
updates in 2,090.141 inclusive wall seconds (1,982.765625 CPU seconds).
Summary SHA256: `6345f2128674853ea052712e590660914787defcf75ad75544ae2da59c0231ed`;
completion: `45543fc1cf65a3d2305fd313f13f8895d20137b82c179e5e3096987f032c43a3`.
The first independent reader failed because its nested objective snapshot schema
incorrectly expected rate-kernel counters. The frozen producer's counters exist
at the outer rate report. Original reader and failure remain preserved. Additive
`recount_v2.py` passed eight focused tests, retained exact outer counters, and
verified all 128 bank endpoints, 208 score files and 5,184 physical step reports.
Verified analysis: `runs/shared-rate-analysis-local/attempt-002/independent-analysis.json`,
SHA256 `922679ce9b665c836cd7f21b8af6aa5de0d0c23545da6e7d69ff30bb7e8e2943`.
No training or model inference was repeated for the correction.
Lower rate passed the predefined primary effect screen at +24.375 percentage
points across all twelve acquisition cells; no primary cell regressed. It passed
479/480 retention checks, failing switch basis revision versus original 9,760
(32 to 30/32, -6.25 pp). Absolute acquisition and basis capability remain unmet;
composition 0/96 and sequence 1/96 fail the transfer screen. Neither branch is
eligible for adoption; home remains 9,328. See `docs/SHARED_RATE_RESULTS.md`.
The next question is a small cross-family diagnostic isolating Apply-word
selection with definitions/facts fixed, before another costly training mechanism.
The planning record preserves exact proof, reader, completion and failure pins.

The completed `docs/HOME_STORAGE_AUDIT.md` measured 16,105 files / 3.133 GiB in
21 selected, nonoverlapping scopes; this is not the entire repository. Comparable
completed home cycles 8/9 use 415,938,670 / 416,061,625 logical bytes. Manifests
record 178,453,794 duplicate lesson bytes across them, without fresh archive
rehashing. Compact chapter storage has an exact reconstruction proof but is not
integrated into home training. Nothing was deleted or reclaimed. Storage receipt:
`b76f036145160a0de4c9554f771c6ab70c2d6133982e8dfc38d7bdfc28b4a76a`.
Two cycles establish an observed footprint, not a long-run growth forecast.

The prospective `docs/SHARED_RATE_NEXT_DECISION.md` keeps the fixed criteria and
defines when further repetition would fail to answer a new question. During the
run, source-only reviews in `docs/SHARED_CURRICULUM_MECHANISM_AUDIT.md` and
`docs/SHARED_PROGRAM_MECHANISM_AUDIT.md` found no code-established truncation,
causal-label or policy-input blocker. They distinguish the existing entity-value
supervision from word-to-program binding and identify a missing isolated Apply-word
selection contrast. They read no live scores and establish no causal diagnosis.
Any subsequent mechanism or diagnostic must follow the completed fixed results.

Frozen source closures and original receipts are immutable research evidence.
Additive experiments can reuse their public APIs; changing a pinned implementation
requires an explicit new version/lineage, not silently relabeling an old launch.
All files under `sources/` are read-only project references. Keep the project,
profile, referenced datasets and full checkpoints together for resume.
