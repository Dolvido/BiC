# Foundation architecture comparison: implementation and validation

This records engineering readiness on 17 September 2026. The six-case local
GPU execution proof **passed every comparison**, including generated outputs.
Its execution ledger (archive reference: `../runs/variant-gpu-execution-local/execution.json`) preserves
all six invocations. Formal preparation began at **04:34:08 UTC**, with 88 source
identities sealed in the launch gate (archive reference: `../runs/foundation-planning-local/variant-launch-gate.json`).
Preparation completed at **05:01:09 UTC**, taking 1,621.016 seconds; independent
data verification completed at **05:37:31 UTC**. All six calibration endpoints,
verification and development selection completed by **06:01:36 UTC**, selecting
learning rate **0.003** for each architecture. Following an interruption before
any main training job began, an explicit [recovery](FOUNDATION_VARIANT_RECOVERY.md)
preserved completed phases and launched only the remaining work at **06:22:43 UTC**.
The current study ledger (archive reference: `../runs/variant-study-recovery-local/attempt-001/execution.json`)
records progress. Verified main results remain pending. Execution checks alone
do not establish acquisition, transfer or mastery.

The [prospective protocol](FOUNDATION_VARIANT_STUDY_PROTOCOL.md) declares one
flat-versus-learned-utterance-summary hypothesis: equal parameter inventory,
paired initialization, lessons, order and development search, with measured
compute reported separately. Its finite formal budget is **21,492 optimizer
updates / 2,063,232 episode exposures**. Calibration has six 510-update jobs;
main training has six 3,072-update jobs. The calibration tail was corrected
from 128 to 126 before preparation because the existing planner requires a
multiple of six. No formal result informed that correction.

The implementation consists of:

- [Hierarchical student](../experiments/hierarchical_sequence_student.py): two
  shared causal byte blocks per utterance, followed by two causal blocks over
  learned EOS summaries. It retains the flat model's width-192 inventory of
  2,221,738 parameters and initial tensors, decoder and heads. Observation
  boundaries are the only packing information supplied to the encoder. Its
  changed receptive fields and gradient paths are explicit in the
  [candidate rationale](FOUNDATION_HIERARCHICAL_ENCODER_CANDIDATE.md).
- [Variant trainer](../experiments/foundation_variant_training.py) and
  [authenticated plan index](../experiments/foundation_plan_index.py): unchanged
  foundation step objective, AdamW and clipping; explicit architecture,
  admission, protection and source identities; full learner/optimizer/cursor
  restoration. An index authenticates the entire plan before model construction
  and supplies immutable consumed-prefix evidence. Old checkpoints are not
  silently migrated.
- [Data boundary](../experiments/foundation_variant_data.py): pinned historical
  evidence, deterministic names-only admission, globally disjoint training and
  reserved evaluation transcripts, canonical fitting anchors, and independent
  regeneration before gradients. The full historical gate hashes 424 inputs
  and 79 sources; ordinary loads authenticate the pinned summary and its two
  transcript dependencies plus all new artifacts.
- [Count analysis](../experiments/foundation_variant_analysis.py): complete
  90-cell development ranking, lower-rate tie rule, and the declared two-panel,
  three-seed descriptive comparison screen. It preserves domain/modality/ASK
  evidence and requires upstream canonical metric and producer authentication.
- [Runtime proof](../experiments/foundation_variant_runtime_probe.py): separate
  processes compare uninterrupted, resumed and independently repeated full
  learner states plus teacher-free outputs for both architectures at all three
  rates. [Study driver](../experiments/foundation_variant_study.py): immutable
  phase records, all-job completion and checkpoint verification before scoring,
  development-only selection, explicit score producers and preserved failures.

The latest focused suites passed **66 checks across seven separate suites**;
this is not a claim that one combined 66-test invocation ran.

| Component | Latest passing checks | Captured evidence |
|---|---:|---|
| Model | 11 | Report (archive reference: `../runs/hierarchical-sequence-validation-local/attempt-001/report.json`) |
| Trainer | 8 | Receipt (archive reference: `../runs/foundation-variant-training-validation-local/iteration-20260917T040503084285Z/receipt.json`), physical work (archive reference: `../runs/foundation-variant-training-validation-local/iteration-20260917T040503084285Z/physical-work.json`) |
| Data | 10 | Receipt (archive reference: `../runs/foundation-variant-data-validation-local/iteration-20260917T042136357370Z/receipt.json`) |
| Pure JSON analysis | 12 | Receipt (archive reference: `../runs/foundation-variant-analysis-validation-local/receipt.json`) |
| Runtime probe | 13 | Latest report (archive reference: `../runs/foundation-variant-runtime-validation-local/attempt-003/report.json`), all-attempt accounting (archive reference: `../runs/foundation-variant-runtime-validation-local/validation.json`) |
| Study lifecycle | 4 | Latest receipt (archive reference: `../runs/foundation-variant-study-validation-local/iteration-20260917T042511772109Z/receipt.json`) |
| Study phase gates | 8 | Receipt (archive reference: `../runs/foundation-variant-study-gates-validation-local/iteration-20260917T042847774943Z/receipt.json`) |

Receipts bind the tested source revisions and captured logs. Checks include
causal/session/decoder isolation, identical initialization, exact CPU optimizer
continuation, admission and artifact tampering, complete phase gates, and
failure accounting. Independent source review found no remaining material
blocker in the data, ranking or phase boundaries. No comparison scores were
used in these reviews.

CPU work is recorded separately from prospective formal training:

- Model tests performed **6 optimizer updates / 12 optimizer episode exposures**,
  included within 27 completed forwards / 39 episode passes. Ten additional
  forward attempts were rejected or deliberately failed.
- Trainer tests performed **13 optimizer updates**, with 84 episodes drawn,
  82 neural attempts and 80 completed microbatch episodes. The deliberate
  partial failure explains the differing counts; optimizer completion was not
  unknown in that suite.
- The three runtime-probe test attempts cumulatively performed **108 updates /
  648 training episode exposures**, plus **756 evaluation episode passes /
  7,560 generated-reply turns**. Earlier failures remain preserved: a Windows
  path-length failure in a copied tamper fixture, then a missing JSON import
  in the new proof-cache path. The latest 13-check attempt passed. Its probe
  source is `f3f386f5ba9bf386784e49fe1f01a1644b08fba2a6b234d58cce83434189c616`.
- Study lifecycle iterations performed **807 updates / 4,842 episode exposures**
  cumulatively, including three deliberate journal-publication failures after
  completed updates. The four preserved invocations are listed below.
- Data, ranking and phase-gate tests performed no neural work. The first gate
  fixture lacked a temporary directory; its failed receipt (archive reference: `../runs/foundation-variant-study-gates-validation-local/iteration-20260917T042817968017Z/receipt.json`)
  remains alongside the passing correction.

| Study lifecycle invocation | Outcome | Actual CPU updates / episodes |
|---|---|---:|
| 04:19:28 (archive reference: `../runs/foundation-variant-study-validation-local/iteration-20260917T041928996695Z/receipt.json`) | Missing fixture directory | 0 / 0 |
| 04:19:42 (archive reference: `../runs/foundation-variant-study-validation-local/iteration-20260917T041942631704Z/receipt.json`) | Complete-plan guard rejected the prefix fixture | 13 / 78 |
| 04:20:28 (archive reference: `../runs/foundation-variant-study-validation-local/iteration-20260917T042028358537Z/receipt.json`) | Four checks passed on complete tiny plans | 397 / 2,382 |
| 04:25:11 (archive reference: `../runs/foundation-variant-study-validation-local/iteration-20260917T042511772109Z/receipt.json`) | Four checks passed after driver hardening | 397 / 2,382 |

The already completed whole-loop GPU proof (archive reference: `../runs/foundation-loop-runtime-validation-local/w192/proof.json`)
is separate evidence: **24 updates / 2,304 training exposures / 1,152 scored
episode exposures**, including pending evaluation and pending practice reloads.
It does not replace the new architecture/rate proof. The completed
six-case proof (archive reference: `../runs/variant-gpu-local/w192/probe.json`) performed **144 updates /
13,824 training exposures**, plus 252 evaluation episode passes / 2,520 generated
reply turns, under [its own protocol](FOUNDATION_VARIANT_RUNTIME_PROTOCOL.md).
All nine exact comparisons passed in each case; CPU verification took 33.328
seconds. No failed or unknown physical work was reported. Neither proof supplies
capability evidence.

The opt-in [variant provider](FOUNDATION_VARIANT_PROVIDER.md) now connects either
explicit architecture to the unchanged complete prescribed loop. Its first ten
focused CPU checks passed: optimizer and pending-practice continuation, partial
evaluation recovery, and atomic publication. This is separate engineering
evidence, with no formal-data use, model promotion or new CUDA guarantee.

Preparation and verification costs remain material. The process-owned index
avoids regenerating consumed lessons at each restore, but construction still
authenticates admission and scans the admitted plan. The bounded proof cache
performs full authentication on first use and rechecks receipt, artifact and
source hashes before and after reuse. The study resolves calibration selection
once per public phase, then retains dependency hash checks rather than repeatedly
regenerating every calibration metric. Home-scale throughput improvement from
these changes remains unmeasured. Model setup, index construction, restoration,
checkpoint I/O, scoring and end-to-end wall time must be reported; materialization
is nested in step time and must not be added twice.
The [preparation efficiency record](FOUNDATION_PREPARATION_EFFICIENCY.md) now
measures repeated validation copies and a separate bounded canonical-byte cache.
Its 13 checks and small validation benchmark do not change this running study.

## Interpreting the pending learning comparison

A read-only review before any new learning scores identified two unresolved
shared questions. The unchanged [objective](../experiments/sequence_training.py)
balances action queries across the present answer classes, then adds statement
acknowledgement, generated-reply and next-byte losses. Complete opposite-answer
pairs enter each batch, but their queries receive independent cross-entropy;
there is no explicit complete-pair objective. Reply supervision includes
statement acknowledgements and is not answer-class balanced. Coefficients alone
do not establish which loss dominates the shared encoder's gradients. The
hierarchical encoder also changes the auxiliary loss's receptive field and
gradient path, so the comparison cannot isolate compression from those effects.

The [action and reply readouts](../experiments/sequence_student.py) share the
encoder but have separate parameters and no agreement objective. The decoder
trains with correct preceding reply bytes and is scored from BOS only. This
ordinary training/evaluation difference is explicit; evaluation supplies no
correct reply prefix. The [pilot's](FOUNDATION_PILOT_RESULTS.md) fully parseable
but weak paired replies establish a semantic performance problem. Its weak
paired action fitting means that an intact understanding hidden behind a poor
reply readout has not been demonstrated either.

Consistent gains across domains, actions, generated replies and retained skills
would support this representation under the fixed objective. Action-only gains
would leave semantic reply stability unresolved. Failure of both models would
not identify a unique cause or establish that changing loss weights will help.
The declared ranking, budgets and comparison screen remain unchanged. Neither
outcome establishes independent acquisition of new skills with an external
teacher disconnected; that remains a separate requirement in the
[home-learning roadmap](HOME_LEARNING_ROADMAP.md).
