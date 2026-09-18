# Brain in Computer (BiC)

### Studying independent learning with a small neural learner and an external local tutor

**Research status: September 18, 2026.** BiC has an operational offline teaching,
practice, evaluation and resume loop on a home RTX 5080 computer. Its latest
controlled experiment found a substantial improvement in restricted English
instruction learning. Reliable composition of unfamiliar procedures, useful
tutor advantage and learned curriculum selection remain open research problems.

[Try the CPU demo](docs/PUBLIC_DEMO.md) · [Results](docs/SHARED_RATE_RESULTS.md)
· [Check the evidence](docs/PUBLIC_EVIDENCE.md) · [MIT license](LICENSE)

## Abstract

Can a small learner acquire reusable procedures from accurate, compact lessons,
retain earlier abilities, and become less dependent on an external language-model
tutor? BiC investigates this question using controlled English environments,
native neural action and reply generation, explicit curriculum verification and
matched continuation experiments. The tutor proposes bounded lesson choices;
it does not answer evaluation questions or become the learner's inference engine.

The current sequence research learner has **2,264,725 parameters**. In the latest
comparison, two continuations received identical lessons and inherited optimizer
state, differing only in learning rate. Rate 0.0001 outperformed 0.0003 by
**24.375 percentage points** across twelve equally weighted acquisition cells.
The lower-rate branch passed **479 of 480 retention checks**, but failed the full
capability requirements: unfamiliar composition scored **0/96** complete pairs
and sequence **1/96**. This supports improved acquisition under the tested
conditions, while exposing the gap between practicing procedures and reusing
their meanings. The project preserves unsuccessful comparisons and measures
retained capability separately from training activity.

## 1. Current system

| Component | Current evidence |
| --- | --- |
| Offline learning loop | Local tutor, verified recipes, matched practice branches, tutor-free evaluation, retention checks and full-state resume |
| Retained home learner | Nine completed cycles; lifetime update 9,328; tenth curriculum authored and compiled |
| Native English research | Causal byte encoder, action head and generated short replies; latest research descendants at update 18,184 |
| Earlier interactive labs | Miniature desktop and visual teaching interfaces, using separately preserved release weights |
| Public distribution | Source, CPU demo, identified research weights, compact aggregate evidence and focused checks; full campaign archives are not included |

These are separate saved states. A newer experiment does not automatically
replace the home learner or upgrade the interactive labs. The historical desktop manifest
still selects older v0.6 weights; the Python package version is v0.7.0.

## 2. Architecture and learning loop

The sequence learner uses four causal attention layers of width 192 over UTF-8
bytes. Each turn's representation feeds a four-class action head and a learned
128-dimensional context for a recurrent reply decoder. A shared, training-only
entity-state readout supplies an additional supervised objective. Its predicted
states are never substituted for native answers during evaluation.

The action classes correspond to **No**, **Yes**, **I need more information**, and
an acknowledgment. Replies are generated from a beginning-of-sequence token and
checked against expected bytes. The model receives visible instruction history;
executable programs and answer labels belong to the curriculum and evaluator,
not to its policy inputs. See the
[native-pathway audit](docs/SHARED_PROGRAM_MECHANISM_AUDIT.md) and
[continuation architecture](docs/SHARED_STATE_CONTINUATION_RESULTS.md).

~~~mermaid
flowchart LR
    T[External local LLM tutor] --> R[Bounded lesson recipe]
    R --> V[Semantic and split checks]
    V --> P[Native learner practice]
    C[Matched procedural practice] --> E[Tutor-free evaluation]
    P --> E
    E --> G[Progress and retention checks]
    G --> S[Saved learner and optimizer]
    S --> P
    E --> T
~~~

The installed tutor is **ministral-3:3b**, served by Ollama through
**127.0.0.1:11434**. The home workflow has no download or cloud fallback.
The controller provides development evidence and accepts only supported
curriculum choices. The tutor cannot execute code, set evaluation answers or
directly edit learner weights. Curriculum selection and continuation rules are
currently engineered; learned self-direction has not been demonstrated.

BiC also contains an earlier brain-inspired modular architecture and visual
labs. Those experiments motivate the broader project, but their capabilities
and scores must not be attributed to the current sequence learner. See the
[original architecture](docs/ARCHITECTURE.md) and
[historical README](README.legacy.md).

## 3. Curriculum: small worlds with checkable meanings

Current research uses colors, bounded counts and switches. English instructions
define temporary word meanings, set facts, apply operations, revise definitions
and ask questions. Shared primitives copy a source value or advance a source or
destination. Composition changes their order and combines them.

This example illustrates supported syntax and intended semantics; it is not a
claim that a particular checkpoint answered this example correctly:

~~~text
Define jaskel for count: copy source to destination; advance destination once.
Set the count of dax to 17.
Set the count of wug to 30.
Apply jaskel to count from dax to wug.
Is the count of wug 18?
~~~

Independent typed and visible-English interpreters verify teaching targets and
state histories. Paired evaluation examples expose whether a relevant change
affects the learner's answer. Defined-word metadata, oracle states and program
structure remain outside neural inputs.

The principal instruction metric requires **every question in both members of a
pair to receive both the correct native action and exact generated reply**.
Query accuracy, unknown-answer accuracy and the easier broad-bank anchor metric
are reported separately. They cannot stand in for complete procedural
understanding. Fit examples are actual practiced material; development text is
held out from training within the same finite grammar. Repeated observation of
development banks limits claims of fresh generalization.

The [curriculum audit](docs/SHARED_CURRICULUM_MECHANISM_AUDIT.md) documents grammar,
context limits, labels and functional equivalences between some programs.
Compiler correctness is evidence about lessons, not proof that the learner has
acquired their meanings.

## 4. Latest experiment: learning-rate continuation

Both branches began at lifetime update 15,592 with identical model weights,
AdamW moments and individual step counters. Each consumed the same ordered
648-bundle curriculum four times: **2,592 new updates per branch**. The
auxiliary-state objective stayed at weight 0.3. Only the declared learning rate
changed: 0.0003 versus 0.0001. Evaluation endpoints and acceptance conditions were
fixed before the run. See the [protocol](docs/SHARED_RATE_TRANSITION_PROTOCOL.md).

| Complete instruction pairs | Common start | Final 0.0003 | Final 0.0001 |
| --- | ---: | ---: | ---: |
| Practiced binding | 11/60 | 15/60 | **43/60** |
| Practiced revision | 39/60 | 46/60 | **55/60** |
| Development binding | 10/120 | 27/120 | **62/120** |
| Development revision | 54/120 | 89/120 | **97/120** |
| Unfamiliar basis composition | 1/96 | 0/96 | 0/96 |
| Unfamiliar basis sequence | 0/96 | 0/96 | 1/96 |

The twelve acquisition cells cross fit/development, binding/revision and the
three families. Their equally weighted lower-minus-control mean was **+24.375 pp**,
with no cell regression. Fit improved by +30.8333 pp and development by +17.9167 pp
relative to control. This is a between-branch effect, not the before/after gain
in a single row above.

Retention passed **479/480 checks** in lower and **465/480** in control. Lower's
remaining failure was switch revision against an earlier reference: 32 to 30
correct pairs out of 32, a 6.25-point loss exceeding the fixed five-point limit.
Only 6/12 absolute acquisition cells and 3/12 basis capability cells passed in
lower. Neither branch qualified for adoption. Continued late binding gains also
mean the experiment does not establish a learning plateau.

This is one inherited optimization history. AdamW decay scales with rate, and
the result is not an independent multi-seed replication. It does not identify a
unique internal mechanism. Full trajectories, individual cells, retention
references and accounting are in the [verified results](docs/SHARED_RATE_RESULTS.md).

## 5. Tutor loop and earlier findings

The persistent home campaign has completed nine cycles and retains update 9,328.
Across its selected trajectory, broad development pairs increased from 257/324
to 272/324 and retention pairs from 286/360 to 300/360. These are selected,
repeatedly observed development results using the broad-bank anchor metric,
not the complete instruction metric in the table above.

The campaign has **not demonstrated a reliable tutor learning advantage**.
Procedural continuations supplied the retained gains. Some teacher choices
produced identical streams or explicit fallbacks; their physical computation
is still counted. A successful local LLM request does not establish learning
benefit. See the [campaign results](docs/CONTINUOUS_TUTOR_RESULTS.md) and subsequent
two-cycle [home validation](docs/OFFLINE_HOME_HANDOFF.md).

| Question | Finding |
| --- | --- |
| Remove auxiliary state supervision? | Failed the benefit screen: -2.0833 pp overall. Keep weight 0.3 as baseline. [Results](docs/SHARED_OBJECTIVE_RESULTS.md) |
| Add recurrent rereading or entity retrieval? | Neither bounded comparison established the intended benefit. [Rereading](docs/RECURRENT_READ_RESULTS.md), [retrieval](docs/ENTITY_RETRIEVAL_RESULTS.md) |
| Continue shared curriculum practice? | Acquisition improved while composition and retention remained limiting. [Sustained practice](docs/SUSTAINED_COMPOSITION_RESULTS.md) |
| Store lessons more compactly? | Exact reconstruction passed for a referenced chapter catalogue; home integration remains outstanding. [Storage audit](docs/HOME_STORAGE_AUDIT.md) |

## 6. Compute and reproducibility

The latest experiment ran locally on an MSI computer with an AMD Ryzen 9 9900X,
approximately 32 GB RAM and an NVIDIA RTX 5080 with 16 GB VRAM. The recorded
runtime used Python 3.12.14 and PyTorch 2.11.0+cu128 with explicit deterministic
FP32 settings. This is a tested runtime, not a cross-platform reproducibility claim.

| Latest paired experiment | Measured amount |
| --- | ---: |
| Physical optimizer updates | 5,184 |
| Episode exposures | 497,664; repeated exposures, not unique examples |
| Inclusive worker time | 2,090.141 seconds, approximately 34 minutes 50 seconds |
| Peak allocated GPU memory | Approximately 1.45 GiB |
| Peak reserved GPU memory | Approximately 3.88 GiB |
| Tutor calls in this comparison | 0 |
| Independently recounted evidence | 128 bank endpoints, 208 raw score files, 5,184 step reports |

Worker time includes loading, training, evaluation, restarts, cleanup and durable
summary publication. Separate preparation, numerical proofs and analysis costs
are in the result report; this is not the total project cost.

Checkpoints preserve complete learner and optimizer state. Frozen experiments
bind sources, data, recipes, evaluations and checkpoints by hashes. Independent
saved-record recounts check the results. The first rate recount found a checker
schema error; its failed receipt was preserved, the checker corrected, and saved
evidence recounted without repeating training. Correctness checks and learning
criteria are separate.

Two comparable completed home cycles used approximately **416 MB each** of logical
file storage. That is an observed footprint, not a long-run growth forecast.
Loader prototypes expose optimization opportunities, but no end-to-end increase
in retained learning per compute hour from those changes has been demonstrated.
The practical home-compute frontier remains unmeasured.

### Try the current learner on CPU

The repository includes an inference-only export of the lower-rate research
candidate at update 18,184. It is the candidate reported above, which failed the
full adoption criteria. The demo presents fixed English scenarios and the model's
native actions and generated replies, including mistakes. It uses no LLM,
internet access, symbolic answer engine or optimizer.

From a source checkout, create a Python 3.12 environment and install dependencies:

~~~powershell
git clone https://github.com/Dolvido/BiC.git
cd BiC
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-cpu.txt -e .
python -m examples.english_demo
python -m examples.english_demo --interactive
~~~

On Linux/macOS, create the environment with `python3.12 -m venv .venv` and
activate it with `source .venv/bin/activate`. Provisioning dependencies may need
internet access; the demo itself is offline. The commands run from the checkout
because the research modules in `experiments/` are not a standalone installed
application. See the [demo guide](docs/PUBLIC_DEMO.md) for the exact command,
checkpoint provenance, limits and observed output. In the recorded demo, the
defined-copy scenario answers all three queries correctly; the color, count and
composition examples each answer only one of three. Those failures are included.

Validate the public evidence and maintained source checks:

~~~text
python tools/verify_evidence.py
python tools/validate_release.py
~~~

The [validation record](docs/RELEASE_VALIDATION.md) states what was tested.
The selected suite is deliberate: historical integration tests depend on the
excluded local archives. A passing source check does not replicate the training
study. The [public evidence guide](docs/PUBLIC_EVIDENCE.md) explains which
aggregate calculations can be checked and which raw artifacts are absent.

### Continue the configured home campaign

The author’s existing MSI installation has a separate, initialized campaign and
local tutor. Its **Start Offline BiC.cmd** menu supports setup checks, status and
hour-budgeted training. The [home guide](docs/HOME_CURRENT_LEARNING.md) and
[handoff](docs/OFFLINE_HOME_HANDOFF.md) record the continuation procedure.
Those commands require the original session, exact model and full archive; they
are not fresh-clone setup instructions. Keep one learning job active at a time.

The source includes loopback-only Ollama transport and campaign/resume logic,
but neither the tutor model nor the complete campaign history is distributed.
A portable fresh campaign initializer and compact long-running storage remain
release work. The public inference checkpoint cannot resume optimizer history.

The earlier visual interfaces also require separate historical checkpoints from
`config/checkpoints.json`; those weights are not included. Their
[guide](docs/PROJECT_GUIDE.md) and [historical README](README.legacy.md) remain
available as research history.

## 7. Limitations

BiC studies a finite English grammar and small simulated worlds. General
conversation, arbitrary-document learning, reliable unfamiliar program composition
and general operating-system autonomy have not been demonstrated.
Biological inspiration motivates architecture; the experiments do not establish
anatomical fidelity, human-like intelligence or consciousness.

The tutor remains external, with constrained authority and independently checked
lessons. Evolutionary or curriculum-search extensions should be judged by useful
native learning, retention, robustness and resource cost. Higher selected scores
alone are insufficient; failed candidates and compute must remain visible.
No result establishes learned self-modification or resolves ethical questions
about hypothetical future autonomous evolution.

## 8. Next milestones

1. Finish the [requested-word diagnostic](docs/APPLY_WORD_DIAGNOSTIC_PROTOCOL.md):
   a fixed 720-pair comparison changing only the requested word, with invariant
   controls and three saved candidates evaluated without training. Its protocol
   is included; unfinished implementation is excluded from this release. It has not run.
2. Use that evidence to choose one shared acquisition or program-reuse experiment,
   preserving whole-curriculum controls and historical retention checks.
3. Replicate the gain on fresh examples and an independent initialization or
   lineage before presenting it as robust.
4. Extend the inference demo into a measured learning demonstration with
   before/after behavior and full training-state resume.
5. Package a fresh offline training initializer, full experiment reproduction
   artifacts and a compact curriculum store suitable for many cycles.
6. Establish useful tutor benefit, increasingly learned curriculum choice and
   retained transfer per total compute hour. These remain scientific milestones.

The [release-readiness record](docs/GITHUB_RELEASE_READINESS.md) separates a
presentable research release from the broader home-learning objective.
A public research artifact can be valuable while these scientific questions remain open.

## Evidence index

| Topic | Entry point |
| --- | --- |
| Latest learning result | [Shared-rate results](docs/SHARED_RATE_RESULTS.md) |
| Operational offline learner | [Current home learner](docs/HOME_CURRENT_LEARNING.md) |
| Exact setup and continuation | [Offline handoff](docs/OFFLINE_HOME_HANDOFF.md) |
| Tutor comparison | [Campaign results](docs/CONTINUOUS_TUTOR_RESULTS.md) |
| Curriculum validity | [Curriculum audit](docs/SHARED_CURRICULUM_MECHANISM_AUDIT.md) |
| Compute and storage | [Storage audit](docs/HOME_STORAGE_AUDIT.md), [step costs](docs/SHARED_STEP_ATTRIBUTION_RESULTS.md) |
| Development record | [Home-learning roadmap](docs/HOME_LEARNING_ROADMAP.md) |
| Earlier visual release | [v0.6 results](docs/V06_RESULTS.md), [v0.7 feedback results](docs/V07_RESULTS.md) |

This is a project research report, not a peer-reviewed paper. Numbers refer to the
linked frozen experiments and their stated metrics. This public source snapshot
is licensed under [MIT](LICENSE), including the supplied BiC inference weights.
Third-party libraries and the separately installed tutor retain their own licenses.
The research prototype is not a peer-reviewed or independently replicated result.

