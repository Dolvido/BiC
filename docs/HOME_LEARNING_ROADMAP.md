# BiC: a broadly useful learner on the home computer

The objective remains a usable BiC that acquires reusable knowledge and skills,
chooses useful practice, corrects mistakes, retains earlier abilities and becomes
progressively independent of an external teacher. Training and validation stay
on the local MSI computer. Accurate learning, useful transfer, retention and
benefit per compute hour guide development. Passing the current toy tasks,
running a large parameter count or keeping the GPU busy does not complete that
objective. No fixed benchmark establishes general intellect.

This roadmap preserves the scope in [RESEARCH_PLAN.md](RESEARCH_PLAN.md) while
updating the route using the local evidence. It is a sequence of decisions and
deliverables, not a prediction of when broad competence will emerge. The current
learner, teaching data and optimization all impose measured limitations; the
home computer's practical learning frontier has not been established.

## Starting point: implemented versus demonstrated

| Existing work | Evidence and remaining boundary |
|---|---|
| Thirteen-region neural core, byte language interfaces, caller-owned state and local checkpoints | These are working engineering components; anatomical names do not establish biological fidelity or useful specialization. |
| Released visual/desktop controller and shared-weight association experiments | V0.6 reports 94.42% joint success on its fresh desktop wording and 98.83% on fresh memory objects. Tasks, language and simulated actions are restricted; consolidation and controller experiments remain distinct. [V0.6 results](V06_RESULTS.md) |
| Symbolic feedback adaptation | V0.7 reached 97.88% late reward across five seeds. This is an imitation-trained recurrent strategy with frozen evaluation weights, supplied categories and action mapping, separate from the visual controller. It is not a general online learning algorithm. [V0.7 results](V07_RESULTS.md) |
| Resumable curriculum selection, parallel candidates, compact replay and promotion checks | The six-skill symbolic loop reached 93.19% mean audit accuracy across three seeds versus 89.32% for round robin. It uses numeric observations and an engineered selection policy. [Loop results](LEARNING_LOOP_RESULTS.md) |
| External local tutor and learner continuation | A complete local pilot ran 648 GPU updates over 62,208 training episodes, one accepted tutor curriculum choice, two exact weights/AdamW restarts, and a final tutor-free cycle. The final audit had zero jointly correct action-and-English pairs out of 396. The loop operates; tutor benefit, broad comprehension and learned self-direction remain unproven. [Completed tutor-loop results](FOUNDATION_TUTOR_LOOP_RESULTS.md) |
| Shared iterative reading and prepared lessons | A completed 648-update-per-branch comparison reused the final encoder block for three additional causal reads. Both branches ended at 0/360 joint fresh-development pairs; the candidate lost 9.09 points on known quantity actions and cost 8.13% more training time. The prepared-batch path matched the original trainer exactly on CPU and GPU. No checkpoint was promoted. The following shared-state comparison is recorded below. [Repeated-reading results](RECURRENT_READ_RESULTS.md) |
| Shared causal-state teaching | The matched comparison completed 1,296 updates in total. Added state supervision reached 4/360 fresh joint pairs versus 0/360 for the control, all gains in switches, and failed its prospective screen. The training-only decoder recovered only 7/2,640 known state values and mostly predicted unknown. No checkpoint was promoted; the following trainability and transfer checks narrowed the next question. [Shared-state results](SHARED_STATE_RESULTS.md) |
| Fixed-buffer shared learning | Four matched conditions completed 1,728 updates on 108 practiced episodes. At the lower learning rate, joint training reached 54/54 paired problems and 516/516 actions and generated replies at the final endpoint; the faster joint learner solved 39/54 pairs. All conditions still failed the stricter dense-state fitting target. A separate reused-development check found only 10/360 joint pairs for the slower learner versus 23/360 for the faster learner, so perfect practice did not establish transfer. [Trainability results](SHARED_STATE_FIT_RESULTS.md) |
| Learned causal entity retrieval and native fusion | Both arms completed 648 updates and ended at 0/360 joint development pairs and 0/54 training-fit pairs. Retrieval cost 7.15% more training time and added 13.40% parameters; it failed the screen and was not adopted. Its final dense readout classified all 12,960 fit states as unknown, with conditional values near constant priors. The next comparison holds architecture and curriculum fixed while testing two learning rates. [Retrieval results](ENTITY_RETRIEVAL_RESULTS.md) |
| Broad acquisition at two learning rates | Both unchanged learners completed 648 updates. A tenfold smaller rate improved known development actions from 535/1,594 to 638/1,594 and conditional training-state values from 719/2,640 to 1,021/2,640, but final joint pairs were 0/360 versus 1/360, with 0/54 training-fit pairs in both arms. Neither passed the screen. The following exact continuation study tests two additional complete cache passes, with fresh evaluation. [Rate results](SHARED_STATE_RATE_RESULTS.md) |
| Exact continued acquisition | Both learners resumed full weights and AdamW for 1,296 additional updates each. The slower learner improved fresh joint native pairs from 0/360 to 143/360, with gains in all three families and known actions at 1,099/1,586. Unknown-answer regressions still failed its acceptance screen. It is the exploratory parent for a matched tutor study, not a promoted checkpoint. CPU replay and packing consumed 58.75% of the 20.43-minute run. [Continuation results](SHARED_STATE_CONTINUATION_RESULTS.md) |
| Verified external chapter authorship and withdrawal | The local LLM selected a compact recipe that changed 54/108 teaching batches. Two identical full-state learners completed 108 teaching and 108 common tutor-free updates each, including exact optimizer restarts. Final joint development pairs were 158/360 tutor versus 154/360 procedural; tutor retention was 140/360 versus 152/360. The benefit and tutor-withdrawal checks failed. The engineering loop works, but this recipe was not adopted and neither checkpoint was promoted. [Authored-tutor results](VERIFIED_TUTOR_RESULTS.md) |
| English-defined instructions | An unchanged native learner completed 216 mixed definition/replay updates in 115.25 seconds. Independent recount confirmed 95/96 complete unfamiliar-word application pairs and 96/96 revision pairs, but 0/96 withheld composition pairs and excessive color/count retention loss. Each episode has one verb, so multiword semantic binding remains untested. The candidate is preserved without automatic adoption. [Definition results](DEFINITION_LEARNING_RESULTS.md) |
| Restricted English learning | Elementary training scaffolds solved familiar borrowing dialogues, but transferred poorly to different name bindings and wording. These are separate unpromoted prototypes. [English results](ENGLISH_FRONTIER_RESULTS.md) |
| A shared 796,292-parameter learner over four procedural subjects | The first general study found no useful episodic-attention benefit. One recurrent arm improved a new-subject score, but solved no known post-intervention decisions and forgot earlier abilities. No checkpoint was promoted. [General results](GENERAL_LEARNING_RESULTS.md) |
| Shared decision teaching with a training-only readout | The 796,552-parameter matched comparison failed its benefit screen. Later-known paired correctness remained zero in all three trained subjects; graph acquisition showed no advantage over fresh initialization and retention failed. [Credit results](COGNITIVE_CREDIT_RESULTS.md) |
| A 753,610-parameter causal sequence learner | Its completed matched study improved conditional action decisions and reduced mean forgetting relative to regional controls, but showed large fitting/development gaps, weak advanced composition and no consistent cross-subject advantage. No checkpoint was promoted. [Sequence results](SEQUENCE_STUDY_RESULTS.md) |
| Anonymous worlds separated from consistent naming | The completed four-condition finite-bank study improved trained-subject naming generalization, but found no integrated withheld-subject learning advantage and substantial forgetting. No checkpoint was promoted; replay/consolidation was not tested in that study. [Diversity results](DIVERSITY_STUDY_RESULTS.md) |
| Continued fitting and bounded replay | Longer broad-bank fitting improved development paired decisions; replay substantially reduced aggregate forgetting at extra compute. Withheld-subject paired reasoning remained weak, and some old slices still deteriorated. No checkpoint was promoted. [Replay results](CONSOLIDATION_STUDY_RESULTS.md) |
| Shared compositions and matched ordering | Joint versus sequential/replay practice had mixed domain effects. Color/switch fitted exact training rows but transferred poorly even to fresh realizations of familiar motifs; count was poorly fitted. All 578 engineering tests passed, but no checkpoint was promoted. [Composition results](COMPOSITION_STUDY_RESULTS.md) |
| Continually fresh verified realizations | With matched procedure/episode draws, aggregate query accuracy improved but final paired correctness fell in all domain/panel macros. Fresh latest-observed fitting was also weak. No checkpoint was promoted; the cause is not uniquely established. [Realization results](REALIZATION_STUDY_RESULTS.md) |
| A transactional raw-English learning backend and practice controller | The backend has 17 focused CPU tests. A separate controller now joins fixed joint or engineered progress allocation, per-cell retention references and learner/decision-state commits. Adaptive learning benefit, teacher benefit and indefinitely compact transcript storage remain unproven. [Controller design](ENGLISH_LOOP_CONTROLLER.md) |
| CUDA checkpoint recovery validation | Default settings produced numerical divergence despite exact reload and lesson streams. A separately declared strict profile subsequently passed midpoint/reload/continuation/repeat checks at all three widths; these short probes do not guarantee arbitrary future workloads. [Validation evidence](RAW_ENGLISH_LOOP_DESIGN.md#subsequent-strict-profile-execution-checks) |
| Shared learner capacity | The completed width96/192/256 comparison found mixed gains, weak fitting and almost no arithmetic paired success. A prospective development rule selected width192/rate0.001 for a fresh next experiment; every worst-cell score was zero and no checkpoint was promoted. [Results](CAPACITY_STUDY_RESULTS.md) |
| Explicit shared foundations | The completed matched pilot found weak primitive fitting and fresh acquisition. Ordered teaching led endpoint paired actions in one initialization, but paired replies remained poor and deteriorated during the common tail. No checkpoint was promoted. [Results](FOUNDATION_PILOT_RESULTS.md) |
| A 1,309,188-parameter regional sequence adapter | Implements causal comprehension through existing regional motor/reply paths, explicit bounded state and ten focused tests. It remains untrained, outside the completed comparison, with no integration capability evidence. [Sequence design](SEQUENCE_BASELINE_DESIGN.md) |

The software now supports controlled experiments. A single checkpoint that
combines these abilities, learns broadly through ordinary teaching and preserves
its gains remains unproven. Integration must demonstrate that combined behavior,
rather than merely place separate demonstrations behind one launcher.

The [completed sustained acquisition](SUSTAINED_ACQUISITION_RESULTS.md) passed
the acquisition and both transfer screens with joint-pair gains in every family.
Development improved from 149/324 to 257/324; original/varied transfer improved
from 71/216 and 64/216 to 136/216 and 125/216. The run made 5,656 updates over repeated
whole-curriculum examples in 1,744.750 seconds, preserving full AdamW through
nine restorations. Counting transfer remains 28/72 and 27/72; finite grammar,
repeatedly observed banks and the single continuing learner limit the result.
No checkpoint was automatically promoted.

The [continuous tutor campaign](CONTINUOUS_TUTOR_RESULTS.md) completed seven
cycles from that update-7,816 learner and retained update 9,112. Six cycles
accepted procedural continuation; no tutor branch demonstrated sufficient
benefit. Across the preserved attempts, physical work was 3,024 updates,
290,304 exposures and eight completed teacher requests. The eighth curriculum
was compiled, but its worker did not start because insufficient campaign time
remained. The controller is engineered; its operation does not establish learned
self-direction or a tutoring advantage.

The next [shared-operation basis probe](DEFINITION_BASIS_RESULTS.md) added 648
updates to a separate research branch. It reached 10/96 complete two-word pairs
and 78/96 revision pairs, with zero unfamiliar composition/sequence pairs and
excessive older-skill loss. A measured training subset also showed weak binding:
19/144 complete pairs versus 131/144 revision pairs. This is not solely a failure
to transfer to unfamiliar words.

The [external chapter-order comparison](DEFINITION_TUTOR_PAIR_RESULTS.md) then
executed a real local tutor's order against a fixed order from that same research
state. Both received identical lessons and common independent practice, totaling
432 updates in 225.625 seconds. Final binding was 21/96 tutor versus 20/96
procedural; revision was 83/96 versus 82/96. Both remained at zero composition
and sequence pairs and failed retention. The retained learner at that stage was 9,112.
The completed [complementary-composition comparison](COMPLEMENTARY_COMPOSITION_RESULTS.md)
broadened operation-order coverage while retaining whole-curriculum practice.
It added 1,296 physical updates in 568.235 seconds. Final atomic binding/revision
was 31/96 and 89/96 for control, versus 3/96 and 55/96 for the broader curriculum.
Both remained at zero complete held-out composition/sequence pairs. The added
curriculum also failed acquisition on practiced examples (binding 0/60, revision
7/60), and both branches failed retention. The independently recounted
[sustained comparison](SUSTAINED_COMPOSITION_RESULTS.md) then completed 5,184
physical updates in 2,174.875 seconds. Curriculum practiced binding/revision
improved to 8/60 and 17/60; development reached 11/120 and 35/120, with gains
across all three families. Complete held-out composition/sequence remained zero,
and both branches failed retention and adoption screens. Late gains support
continued learning; they do not establish a plateau or broad English competence.
The [shared-objective comparison](SHARED_OBJECTIVE_RESULTS.md) has now completed
and passed independent raw-record recount: 5,184 physical updates in 2,139.969
seconds from the same update-13,000 parent and lessons. Removing the auxiliary
state objective failed its prospective effect screen: zero-minus-control was
-2.0833 percentage points across the 12 equally weighted acquisition cells
(fit -7.5, development +3.3333). Keep weight 0.3 as the research baseline.
Control practiced revision improved 17 to 39/60 and development revision 35 to
54/120, while binding remained weak and unstable. Basis composition reached only
1/96 in control and 0/96 without the auxiliary; sequence remained zero. Neither
arm passed any complete acquisition cell or all retention requirements, so
neither replaces the home learner. Further shared-practice and retention work
remains warranted; these finite trajectories establish neither a global plateau
nor a unique architectural cause. Prior-definition rehearsal already taught some
multi-clause syntax.

The current priority is a usable offline home setup using the already installed
local English tutor and retained learner, with explicit instructions for continued
practice. The [home launcher](HOME_CURRENT_LEARNING.md) resumes the existing
campaign with a requested time allowance and preserves completed stages and
pending lessons. Its offline acceptance checks are complete. The actual validation
finished two further cycles with 864 physical updates and two local tutor calls.
Independent recount retained the procedural continuation at update 9,328, with
all retention conditions passing and small development gains. The tenth
curriculum is compiled for the next session. The Windows menu's setup/status
path and a real child resume at a budget boundary also passed. The
[offline handoff](OFFLINE_HOME_HANDOFF.md) provides current evidence and user
instructions. This makes the local research workflow usable; it does not settle
general English learning or establish a useful tutor advantage.

The [packed-lesson reuse design](PACKED_LESSON_REUSE_DESIGN.md) has a passing
full-state equivalence proof. Its six-image, two-pass timing check took 5.033
seconds packed-v2 versus 4.826 seconds reference, so no speedup or adoption is
claimed. The completed sustained run used the original continuation/preparation
path with a bounded resident replay cache. A separate immutable JSON lesson cache
also preserved exact inputs but took 83.859/108.593 seconds versus
34.063/34.844 seconds for ordinary control/curriculum loading, including checks.
It was not integrated. A new bytes-backed version passed nine synthetic tests;
its first actual-data comparison preserved every returned image and reduced
isolated inclusive JSON-path time by 17.43% for control and 7.08% for curriculum.
This is not an end-to-end training speedup, and the cache remains unintegrated.
A separate two-update profile observed 1,639 source-file opens per step and
substantial copying costs; its instrumentation overhead is unmeasured. See the
[cost results](SHARED_STEP_ATTRIBUTION_RESULTS.md). The
[next learning protocol](SHARED_RATE_TRANSITION_PROTOCOL.md) compares unchanged
0.0003 and lower 0.0001 rates from the common final control parent, retaining
objective 0.3 and the original full curriculum and data path. Its explicit rate
transition adapter passed nine metadata tests and independent source review;
its CPU and RTX 5080 numerical proofs now both pass exact incoming-state,
two-update and save/reload comparisons at both rates. Each proof performed eight
validation updates; these are not new retained learning. The independent reader
passed eight synthetic tests and an actual saved-parent metadata preflight.
The runner passed all 11 focused tests, and the comparison's source and inputs
are frozen. Independent admission of the actual saved parent metadata passed.
The comparison started on the RTX 5080 on September 17 at 22:13 EDT, with four
full curriculum passes per arm and one inclusive 3,600-second allowance.
The comparison completed 5,184 physical updates in 2,090.141 inclusive seconds.
The [verified results](SHARED_RATE_RESULTS.md) show a +24.375 percentage-point
lower-rate advantage across the twelve acquisition cells, with no cell regression.
Lower passed 479/480 retention checks, but only 6/12 absolute acquisition and
3/12 basis capability checks; composition was 0/96 and sequence 1/96. Neither
branch qualifies for adoption. The retained home learner remains at 9,328.
The initial independent reader failed on a nested report schema mismatch; an
additive correction passed eight focused tests and verified all saved evidence.
No training was repeated. The next shared question isolates requested-word
selection while holding definitions/facts fixed, before another training mechanism.
The [storage audit](HOME_STORAGE_AUDIT.md) measured two comparable home cycles
at about 416 MB each. Compact reconstruction is proven but remains unintegrated;
long-run useful learning per total compute hour is still unestablished.
Further work remains
tied to useful shared acquisition, retention and transfer beyond the current grammar.

## Development sequence

### 1. Test shared decision learning and memory — first comparison completed

The [selected control comparison](COGNITIVE_CREDIT_RESULTS.md) is complete: a
training-only prefrontal readout at weights zero and 0.3, with all official
decisions through the unchanged regional/motor pathway. Binding, arithmetic and
conditional logic were trained; graphs were reserved for adaptation. Both arms
used the frozen samples and fixed 1,800-update endpoints. The added objective
failed its screen, and no checkpoint or confirmation run was selected.

Continue judging common mechanisms across subjects: known decisions after changes,
complete opposite-answer pairs, calibrated uncertainty, generated replies,
adaptation speed and retention. Record gradients and representation differences
as diagnostics, without treating them as proof of a causal explanation. Preserve
the earlier failed comparisons. Binding and arithmetic were poorly fitted even
on the training diagnostic, and later-known conditional decisions were all
wrong. Graph adaptation matched fresh initialization at its endpoint while
earlier skills deteriorated. This motivates the sequence comparison below; it
does not establish a unique cause or justify more hours on the failed objective.

### 2. Establish a stronger general sequence-learning baseline — first comparison completed

A small causal attention learner is now implemented from scratch with raw
observations, explicit legal context, the shared supervision objectives and
separate generated replies. Its data packer, trainer and teacher-free evaluator
have focused validation. The [completed comparison](SEQUENCE_STUDY_RESULTS.md)
includes nine calibration runs, three fresh main runs and six withheld-arithmetic
adaptations, totaling 16,584 updates. Sequence conditional action decisions
improved and its mean forgetting was smaller; recurrent arithmetic acquisition
was stronger. Neither design was consistently superior across subjects and
outputs. No checkpoint was promoted, and the study defined no automatic
numerical promotion screen. All jobs in that study finished; its full suite
passed 453 tests, including the separate untrained regional adapter.

The regional models could already generate every known later conditional reply
correctly on the retained bank before adaptation, despite poor action decisions;
those replies then deteriorated sharply. A next comparison must therefore test
generic output alignment and retention as well as shared representations and
curriculum diversity. The [sequence design](SEQUENCE_BASELINE_DESIGN.md) keeps
the diagnostic model distinct from the untrained regional integration. Give
each new architecture a declared, comparable development-search budget, and
measure a small capacity ladder only when learning evidence justifies it.

Report parameter-matched, exposure/update-matched and measured-compute comparisons
separately. Match available history and memory capacity; disclose additional
readouts, caches and pretrained components. The earlier V0.7 GRU comparison used
similar parameters and updates, while BiC took about 9.45 times longer on that
recorded host. It does not settle the broader architecture question. Keep regional
complexity where repeatable accuracy, transfer, retention or efficiency supports
it. Model size is a means to test a bottleneck, not a capability score. Repeat
promising screens across initializations and family rotations; retain the
research plan's minimum of five training seeds for capability claims.

### 3. Broaden the curriculum and learn across tasks

The [sequence development diagnostic](COGNITIVE_DIVERSITY_DIAGNOSTIC.md)
motivated the now-completed [finite-bank diversity study](DIVERSITY_STUDY_RESULTS.md).
Separating anonymous worlds from naming improved the ability to measure each
generalization challenge. Broader naming practice helped held-name panels;
more worlds helped novel worlds under familiar names. The broadest bank still
underfit graphs at the fixed endpoint, and all four arms had negative integrated
withheld-subject learning differences against fresh initialization. All lost old
query accuracy after adaptation. All jobs in that study finished; neither
this evidence nor passing tests completes the broad objective.

The subsequent [continued fitting and replay study](CONSOLIDATION_STUDY_RESULTS.md)
completed those controls. More exposure improved broad-bank fitting and development;
replay retained much more old performance than ordinary adaptation. Neither
established reliable arithmetic paired reasoning. No checkpoint was promoted.
The subsequent [shared-composition study](COMPOSITION_STUDY_RESULTS.md) matched
per-family example streams across joint and sequential/replay schedules and two
late-family rotations. Schedule effects were mixed and absolute paired competence
remained weak. A post-study diagnostic distinguished strong exact-bank color/switch
fitting with poor fresh-realization transfer from count underfitting. Neither
forgetting alone nor one universal capacity explanation accounts for this evidence.

The [fixed-versus-fresh realization comparison](REALIZATION_STUDY_RESULTS.md)
is now complete. Fresh verified naming and values improved overall query and
probability scores while lowering final paired correctness in all twelve
domain/panel macros. Fresh also poorly fitted the latest observed realizations;
fixed color/switch fitted repeats strongly but transferred poorly. The intervention
changed value realizations, increments and targets as well as names, with matched
structural draws and episodes but different byte/runtime costs. It tested
concurrent maintenance, not late-task retention or adaptive scheduling.

The [completed capacity comparison](CAPACITY_STUDY_RESULTS.md) tested widths
96/192/256 within the same four-layer, four-head observation path and objective:
753,610 / 2,221,738 / 3,692,010 parameters. Equal development rate search and
matched fresh streams produced mixed gains; fitting remained weak and all widths
scored zero held-composition count action/reply pairs. This does not isolate
capacity, representation, optimization or lesson difficulty as the cause.
The prospective development rule chose width192/rate0.001 for the next fresh
experiment, with every candidate's worst-cell score still zero. No trained
checkpoint is promoted or reused. Test a shared teaching or representation
mechanism before further scaling, and repeat promising results across seeds
before capability claims or adaptive allocation.

A new [shared-foundation curriculum](FOUNDATION_CURRICULUM_DESIGN.md) now makes
primitive and composed teaching explicit across every domain, without changing
the model or objective. Its candidate comparison holds the exact lesson multiset
fixed while changing teaching order and preserving a shared mixed finish. The
generator, trainer, evidence and evaluator are implemented. The latest 814-test
run had 813 passes and one existing UI connection-reset error; all six UI tests
passed separately on rerun. All 101 foundation checks passed in that full run.
Separate three-width GPU proofs used 144 updates/13,824 episodes for execution
validation of the earlier source revision. Formal foundation preparation failed
before gradients on three complete pairs overlapping historical training. A
recorded names-only admission amendment preserves their problems and answers.
The new selected-width execution proof and bank/source freeze passed; both
matched learners completed training, verification and scoring. The
[results](FOUNDATION_PILOT_RESULTS.md) show weak acquisition even on practiced
primitives, with 167/1,584 versus 99/1,584 final audit action pairs and only
19/1,584 versus 4/1,584 reply pairs. The order advantage is descriptive and varies
across checkpoints. The next shared hypothesis is a learned utterance-summary
encoder feeding causal reasoning across turns, with equal calibration and fresh
initialization. Its [prospective comparison](FOUNDATION_VARIANT_STUDY_PROTOCOL.md)
entered formal preparation at 04:34:08 UTC on September 17, after 66 focused CPU
checks and a passing six-case GPU proof. It uses three paired initialization
seeds. Data verification and six calibration jobs are complete, selecting rate
0.003 for each architecture. The [main continuation](FOUNDATION_VARIANT_RECOVERY.md)
preserves the sealed design and completed work after an interruption before
main training began. The [completed main results](FOUNDATION_VARIANT_RESULTS.md)
fail every prospective screen criterion on both primary panels. Hierarchical
memory use is lower, but shared paired acquisition remains weak and counting
actions are zero on every run's fitting sample. No model is promoted; extending
either unchanged loop is not supported by these results.
The complete prescribed foundation loop passed eight strict GPU restart/repeat
comparisons. Its separate 24 engineering updates establish tested continuation,
not learned self-direction or a benefit from adaptive practice.
A versioned provider now connects both candidate architectures to the unchanged
prescribed loop; ten focused CPU checks passed, with no formal weights adopted.
Its own combined loop/architecture GPU proof remains future validation.
The version-two [fresh-cycle handoff](FOUNDATION_CYCLE_IMPLEMENTATION.md) now
has 20 passing CPU checks, including three complete cycles for each architecture.
It keeps full weights and optimizer state, protects earlier training/evaluation
transcripts, and restores from a single current-parent capsule without replaying
ancestor models. A separate lifetime record preserves within-cycle and earlier
best results; exact exclusion inventories still grow with unique lessons.
The [automatic durable owner](FOUNDATION_CYCLE_OWNER_DESIGN.md) passed ten further
CPU checks and now executes fresh cycles under explicit compute bounds through
the [local runner](FOUNDATION_CYCLE_RUNNER.md). A prospective adaptive curriculum
comparison remains necessary for useful self-directed learning. The
[cycle-boundary adviser design](FOUNDATION_CYCLE_ADVISER_DESIGN.md) specifies
balanced whole-curriculum choices, a matched benefit test and explicit
compatibility constraints for the existing English allocator. The
[small shared-learning diagnostic](FOUNDATION_SHARED_DIAGNOSTIC_RESULTS.md)
is complete. Its historical trained-state probes do not consistently outperform
initialization; all twelve fixed fits are unconverged, and original semantic
reply errors already occur at the first byte. These historical states cannot
directly diagnose the new hierarchy. The [fixed gradient inspection](FOUNDATION_GRADIENT_RESULTS.md)
also completed: answer learning dominates individual weighted components, while
reply gradients align but remain small. Broad auxiliary conflict is unsupported.
The selected [shared objective comparison](FOUNDATION_OBJECTIVE_STUDY_PROTOCOL.md)
balances full reply learning across semantic answer classes and utterances,
with unchanged objective coefficients, matched lessons and three fresh paired
initializations. Reused evaluation banks are development benchmarks, not a new
untouched confirmation. Five bounded CPU checks and the six-case CUDA continuation
proof passed; the latter used 108 updates and 10,368 episode exposures. The formal
calibration and three-seed comparison completed locally on September 17. The
[completed results](FOUNDATION_OBJECTIVE_RESULTS.md) fail all three fixed
criteria on both primary panels. Known-answer accuracy improves and unsupported
asking falls, but paired gains do not hold across seeds, direct-fact fit remains
weak, and every run loses known correctness during the final fixed interval.
No checkpoint was promoted. The next shared investigation concerns stable
acquisition and use of state across the curriculum, including stream-wide
ordering/distribution and controlled binding/history effects.
This tests reusable state operations across domains, while transfer, retention,
external-teacher independence and broader grounded skills remain open goals.

Separately, the [overlapped preparation trainer](FOUNDATION_PREFETCH_RESULTS.md)
passed six CPU checks preserving exact learning trajectories, restart and
failure recovery for both objectives. Its subsequent 396-update local CUDA
identity proof passed, including saved continuation and fresh repeats over all
depths and lengths. A complete-cost speed benefit remains untested; it has not
been adopted or supplied a new capability result. The prospective
[contextual-sensitivity diagnostic](FOUNDATION_CONTEXT_RESULTS.md) has now
completed all nine model states. Paired comprehension stays weak even with
nearby evidence; no consistent shared distance decline emerged. Unknown-last
questions also expose poor generalization beyond the original grammar. The
next candidate varies causal-valid lesson layouts across the whole curriculum
while preserving each query's truth and read-version ancestry. It remains a
prospective teaching comparison, not a demonstrated learning improvement.

Keep world/name/symbol novelty separate when interpreting the earlier arithmetic
audits. Output alignment also remains measurable: the latest fresh switch model
had substantially weaker action/reply agreement, while many earlier sequence
errors were shared by action and reply. Neither alignment alone nor one aggregate
score explains reusable learning. Regional routing remains a distinct untrained
integration question.

Develop families around reusable operations: binding and revision, relations,
quantification, numerical state, causal intervention, composition, induction and
short action sequences. Expand combinations, distractors, wording, history length
and sensory grounding in explicitly versioned stages. Separate linguistic novelty
from new mechanisms when interpreting transfer. Reserve whole families and
compositions; rotate the withheld family instead of selecting a favorable one.

Use compact world descriptions, generator versions, seeds and bounded replay
recipes. Recompute truth with independent simulators; reject ambiguity and
unsupported statements. Subject names and parsed world state remain outside
the student. Validate difficult query slices and simple competing heuristics
before training. Inspecting an audit retires its pristine status for future
development. This extends the curriculum as a coherent learning program, rather
than repeatedly optimizing one observed test failure.

Fast adaptation alone is not meta-learning. A later meta-learning comparison
must actually train across support/query learning episodes and test on untouched
task families. Compare it with ordinary shared pretraining under matched budgets.
Require better acquisition of new tasks while retaining older ones.

### 4. Consolidate knowledge and connect modalities

Bounded replay now has a useful single-seed retention result in these restricted
English worlds. It still left some old-family slices worse and did not solve the
new subject. Compare it with consolidation and ordinary joint training using complete
old-task/new-task matrices. Measure how quickly new facts become durable, what
survives activity resets and restarts, interference from conflicting lessons,
and performance after examples or teacher assistance are removed. Count replay
bytes and exposures as part of the budget.

Bring visual grounding, language, feedback adaptation and action prediction into
a common saved learner through tested interfaces. Preserve released checkpoints
as references. The previous visual associations show a small route from exemplars
into shared weights; they do not establish consolidation across the new broad
curriculum. Longer sequences and action-conditioned worlds are needed before
claiming planning or general computer use.

### 5. Integrate the self-directed learning loop

The new [raw-English backend](RAW_ENGLISH_LOOP_DESIGN.md) implements the shared
engineering boundary: canonical observation-only training, a persisted finite
schedule and cursor, atomic optimizer/stream commits, rollback and per-bank
paired/reply/known/ASK development vectors. It has 17 focused CPU tests. The new
[controller](ENGLISH_LOOP_CONTROLLER.md) adds engineered allocation and a fixed
joint baseline, family coverage floors, best-observed per-cell retention
references, and atomic decisions together with learner state. These are
implemented mechanisms whose learning benefit has not been measured. Deadlines
may overrun by the active operation and current save cost; an explicit transcript
bound stops further work, while indefinitely compact novelty storage remains open.

Reuse the existing loop's resumable state, candidate isolation, compact replay,
retention checks and source/data fingerprints for the stronger learner. Let its
development evidence select useful practice while maintaining minimum coverage
of established subjects. Compare engineered progress selection with fixed
curricula; identify separately any scheduling policy that is itself learned.
Require opposite-answer paired decisions and per-family retention before using
aggregate accuracy as the selector's progress signal. The latest study showed
that correct unknown answers can mask failed known counterfactual reasoning.

The [verified tutor curriculum design](VERIFIED_TUTOR_CURRICULUM_DESIGN.md) is now
implemented: exact model/optimizer restart, compact tutor-selected chapter
recipes, independently checked examples and a matched procedural control. The
[completed teaching/withdrawal experiment](VERIFIED_TUTOR_RESULTS.md) measured
158/360 final joint development pairs for tutor teaching versus 154/360 for
procedural teaching; it failed the declared tutor-benefit and withdrawal checks.
The later [recurring campaign](CONTINUOUS_TUTOR_RESULTS.md) and
[offline home validation](OFFLINE_HOME_HANDOFF.md) completed nine cycles in
total, retaining the procedural learner at update 9,328. Cycle ten's authored
and compiled lessons are ready to resume. The initial service rejection remains
in the historical evidence; it is no longer an unresolved integration blocker.
These runs establish an operating external-tutor loop, not a reliable tutoring
advantage or a learner-trained curriculum policy.

Keep the LLM external. It may suggest lessons, explanations or experiments that
pass independent checks; it supplies neither policy answers nor hidden world
state during evaluation. Measure accepted lesson quality, student gain per
teacher/compute cost and the assistance required for new learning. Independence
requires successful learning and action with the tutor disconnected, across
successively broader tasks. Local teacher generation can run separately from
student training when memory contention warrants it.

Evolution, if useful, proposes bounded candidate changes to a declared learner
or learning procedure. Preserve the previous accepted checkpoint and evaluate
truthfulness, uncertainty, retention and practical benefit. Candidates cannot
rewrite truth rules, inspect final answers or lower their acceptance criteria.
Learning objectives must avoid distress, coercion or harmful external actions;
capability growth alone is not a sufficient benefit criterion.

### 6. Deliver a usable checkpoint and measure the home frontier

Provide a nontechnical local interface for teaching, correcting, continuing and
resuming the same learner. Show what it has actually learned, when information
is missing and when an experiment is still unvalidated. Verify save/reload of
weights, optimizer, curriculum/replay policy and conversation state. A usable
release needs dependable behavior in its declared scope, resource limits,
recovery after interruption, and reproducible evaluation of the combined skills.

Measure a practical resource frontier with real training and retained transfer,
not allocation-only tests. Vary sequence length, batch size, representation size,
retrieval capacity and training precision in small declared comparisons. Record
unique lessons, repeated bytes/exposures, useful improvement per wall hour,
inference latency, peak RAM/VRAM, checkpoint/replay storage and background
contention. Include a CPU reference and leave the desktop responsive. Parallel
lesson generation and independent candidates can accelerate research; repeated
queries or workers are not extra independent evidence.

The local inventory (archive reference: `../runs/general-credit-local/home-hardware-inventory.json`)
is an MSI Aegis ZS2 with a Ryzen 9 9900X (12 cores/24 threads),
an RTX 5080 reporting 16,303 MiB, and about 31.05 GiB of Windows-visible physical
RAM. The recorded local benchmark (archive reference: `../runs/general-learning-local/benchmark.json`)
used PyTorch 2.11.0+cu128 and measured 264.60/486.88 MiB peak allocated GPU memory
for twelve batch-64 recurrent/episodic updates. These small samples establish
execution, not an upper capacity or a guaranteed learning rate. Earlier
175.7-million-parameter profiles in [HOME_COMPUTE.md](HOME_COMPUTE.md) ran on a
different CPU host and must not be presented as training results on this MSI.

The newer initialized sequence probe (archive reference: `../runs/sequence-baseline-local/benchmark-initialized.json`)
measured 64.17 sequence versus 8.44 regional updates/second at batch 64, with
578.44 versus 205.54 MiB peak allocated CUDA memory. It used only twelve measured
FP32 updates per model on binding, arithmetic and conditional worlds. Its speed
advantage is not a learning-efficiency result and cannot predict capability or
longer-study throughput. The [new protocol](SEQUENCE_STUDY_PROTOCOL.md) compares
three architectures after equal rate-calibration budgets and rotates arithmetic
into the withheld subject. Its [completed learning evidence](SEQUENCE_STUDY_RESULTS.md)
now records positive query-area gains over fresh for each architecture, alongside
uneven paired gains, weak advanced composition and retention losses. The result
does not establish a general architecture winner or exhaust the home frontier.

Completion remains tied to the user's broad home-learning objective and explicit
evidence about useful independence, transfer, retention and operation. Individual
studies close when their declared comparisons finish; the larger objective stays
open through failed studies and incremental milestones. Each continuation should
leave an honest capability record, preserved checkpoints and a concrete next
decision based on learning evidence.

## Curriculum cautions established for the credit comparison

The CPU-only distribution report (archive reference: `../runs/general-credit-local/curriculum-distributions.json`)
validated its frozen train/dev recipes: base seeds 20,000,000/24,000,000; family
order binding, arithmetic, conditional; levels 1/2; 512/64 episodes per family
and level. It covers 3,456 episodes and 9,216 questions with zero exact train/dev
transcript overlap. It generates no held-out graph or audit questions.

Arithmetic's fixed query positions admit 66.67% query accuracy from position
majorities while solving no opposite-answer pairs. Conditional initial questions
only require the first switch because other inputs are neutral; copying that
switch then always asking on the last question can score 62.5% while solving no
known intervention outcomes. Report those outcomes separately. Binding positions
vary, but all three subjects use a finite grammar, small values, shared aliases
and six-turn skeletons. These limits remain part of the frozen experiment's
interpretation. Data changes require a new prospective protocol, not silent
edits after observing an arm's scores.
