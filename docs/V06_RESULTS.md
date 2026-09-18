# BiC v0.6 results

V0.6 repairs the observed greeting regression, demonstrates visual associations
stored in shared neural weights, and provides one launcher for both local labs.
The experiments remain small, controlled learning tasks. No local LLM was
downloaded or invoked; the [teacher feasibility study](LOCAL_LLM_TEACHER.md)
describes a proposed future experiment.

## Targeted rehearsal and retention

The released regional controller starts from v0.5 and retains its **558,467
parameters**, including the frozen visual encoder. Each update combines 16
regional-memory examples with 32 older desktop examples. Desktop reply tasks
receive fixed priorities derived from their initial development errors; greetings
receive the highest priority. This is a practical rehearsal schedule inspired by
protecting older learning, with no claim that it reproduces biological sleep.

The run executes **1,000 updates**. Every 200 updates, it measures desktop and
memory development action-plus-reply success and selects the earliest checkpoint
with the highest minimum of those two scores. The selected checkpoint is **step
400**. Data preparation takes 7.80 seconds; training plus development evaluation
takes 81.63 seconds on the execution host's CPU with one Torch thread. These are
measured host times, not estimates for the user's GPU.

The earlier v0.4/v0.5 reserved desktop wording is now deliberately rehearsed.
Its improved score measures repair on a known problem. A separate bank of 16
complete sentence templates is fixed before training and excluded from all
rehearsal templates, including after substitution of colors, positions, and
words. A separate set of 256 glyph identities excludes every recorded v0.4 and
v0.5 identity partition. These new identities are used only in final evaluation.

Both checkpoints receive identical evaluation cases. Each desktop condition has
448 episodes, balanced across seven tasks; memory has 1,280 episodes. Joint
success requires the correct action outcome and exact freely generated reply in
the same episode.

| Paired evaluation | v0.5 before | v0.6 after | Interpretation |
|---|---:|---:|---|
| Earlier desktop wording, joint success | 95.98% | **100.00%** | Rehearsed restoration diagnostic |
| Earlier greeting wording, joint success | 81.25% | **100.00%** | 64 episodes in each model |
| Fresh desktop sentences, action success | 95.54% | **98.44%** | New complete wording, familiar task meanings |
| Fresh desktop sentences, exact reply | 83.48% | **94.87%** | Free byte generation |
| Fresh desktop sentences, joint success | 81.70% | **94.42%** | 448 paired episodes |
| Fresh greeting sentences, joint success | 29.69% | **100.00%** | 64 episodes in each model |
| Fresh memory objects, action accuracy | 99.06% | **99.38%** | Familiar five-relation instruction families |
| Fresh memory objects, exact reply | **98.98%** | 98.83% | Small regression remains |
| Fresh memory objects, joint success | 98.75% | **98.83%** | 1,280 paired episodes |

Fresh desktop joint success by task is 100% for color clicks, position clicks,
recall, and greetings; 93.75% for color descriptions; 96.88% for typing; and
70.31% for clarification. The latter remains a clear weakness: unfamiliar
unsupported requests can elicit an unrelated reply. Some typing cases also stop
before completing the requested field contents. The aggregate improvement does
not establish reliable open-ended conversation.

This is one repair seed and one combined schedule. There is no matched-budget
comparison isolating which part of the new rehearsal schedule causes the gain.
The new wording is distinct as complete sentences, but uses familiar words and
task meanings. Memory object identities are new; the five instruction meanings
and earlier reserved memory-phrase family are familiar. Finite desktop scene
states and procedural rendering families can overlap earlier experience.

Evidence: protocol (archive reference: `../runs/retention-v06/protocol.json`),
initial development errors (archive reference: `../runs/retention-v06/initial_development.json`),
training and selection (archive reference: `../runs/retention-v06/training_report.json`), and
paired final evaluation (archive reference: `../runs/retention-v06/final_evaluation.json`).

## Associations transferred into shared neural weights

A separate **56,824-parameter** network reads the bytes of a taught name through
shared embedding, GRU, and projection weights and predicts a visual embedding.
Training targets come from the fast episodic memory's frozen visual exemplars.
At inference, this network owns no exemplar bank, object IDs, or per-name visual
vector table. Its known-name set remains an explicit symbolic vocabulary gate.

The experiment teaches 16 associations, trains for 1,000 updates, then teaches
16 additional associations and trains two copies for another 1,000 updates.
One copy replays old examples; the other receives only new examples. Both receive
16 new examples per update. Replay adds 16 old examples per update, so total
examples and compute are not matched. There are two independently initialized
seeds, 32 names per seed, and four teaching renderings per name. Every arm uses
its fixed final update; evaluation does not select weights or training duration.

The **v0.5 regional controller and visual encoder stay frozen** in this
experiment. Predicted visual similarities enter the existing hippocampal pathway;
the regional motor network selects the action and the byte decoder generates the
reply. These measurements are separate from the repaired v0.6 controller above.

Each seed has 2,560 evaluation episodes, equally split between older and newer
associations. Half require a valid selection; the other half require STOP for an
unknown name, absent object, duplicate match, or impossible direction. The table
averages the two equally sized seed results.

| Recall condition | Older associations: action | Newer associations: action | Older associations: joint | Newer associations: joint |
|---|---:|---:|---:|---:|
| Original episodic exemplars | 99.18% | 98.98% | 98.40% | 98.44% |
| Shared neural recall, with replay | **99.34%** | **99.34%** | **98.71%** | **98.71%** |
| Shared neural recall, without replay | 57.03% | 99.34% | 53.83% | 98.75% |
| Random initial recall weights | 50.00% | 50.00% | 50.00% | 50.00% |
| Erased recall weights | 50.00% | 50.00% | 50.00% | 50.00% |

Random and erased recall weights succeed on **0% of valid selections**. Their
50% aggregate score reflects the balanced STOP cases. After learning additional
names without replay, older valid-selection accuracy is only 15.94%; with replay
it is 99.22%. This supports dependence on learned shared weights and the benefit
of rehearsing older associations in this experiment.

Names and object identities in this experiment are taught before testing. Only
rendering seeds and arrangements differ; the procedural appearance family is
familiar. This is retention of taught associations in neural weights, not the
ability to infer an arbitrary name-to-object mapping without being taught it.

The learned network is larger than the original 8,192 stored exemplar floats:
56,824 FP32 parameters versus 8,192 FP32 target values, before metadata. No
compression advantage is claimed. It is a new shared recall network, not a
demonstration of transfer into the existing cortical-region weights or a complete
model of biological systems consolidation.

### Restart and evidence checks

The original sealed evaluation reports that source-memory files were still
present in later execution sessions despite training having recorded successful
deletion. That report is preserved. A supplemental verification explicitly erases
their contents and repeats the same fixed 32 cases per seed, without retraining
or choosing new cases. **All 64 actions are correct** in fresh OS processes,
with unchanged checkpoints and episodic construction, lookup, and teaching
forbidden. The workers receive candidate pixels and quoted instructions, not
procedural identities or target actions.

A further supplement checks 100 public-API evidence cases across every case
category, age group, condition, and seed. Cached similarities match public
inference within 1e-6; the largest observed difference is approximately 1.2e-7.
All 25,600 condition-specific known-name flags match actual memory membership.
The evaluator now derives that flag directly from membership and enforces frozen
source/checkpoint hashes for future runs. The exact historical training/evaluation
driver is preserved as `frozen_runner.txt`, matching its recorded protocol hash.
These corrections change neither trained weights nor the original sealed result.

Evidence: protocol (archive reference: `../runs/consolidation-v06/protocol.json`),
training report (archive reference: `../runs/consolidation-v06/training_report.json`),
original final evaluation (archive reference: `../runs/consolidation-v06/final_evaluation.json`),
restart supplement (archive reference: `../runs/consolidation-v06/restart_verification.json`), and
public evidence parity (archive reference: `../runs/consolidation-v06/evidence_parity.json`).
See [CONSOLIDATION.md](CONSOLIDATION.md) for the experimental API.

## Release verification and usability

The complete source test suite passes **146 tests in 11.119 seconds**; see the
[test log](../experiments/v06_tests.log). Independent review also verifies the
retention selection rule, sentence and identity exclusions, target isolation,
joint-metric semantics, and protocol/checkpoint hashes. The consolidation review
verifies saved-state contents, preserved historical source, public evidence
parity, and the restart supplement's link to the unchanged original report.

`python -m brain_in_computer doctor` checks local readiness, and
`python -m brain_in_computer launch` starts both labs using the release manifest.
The default teaching UI keeps rapid episodic storage; using it does not train
shared neural recall. The [project guide](PROJECT_GUIDE.md) documents the layout,
memory files, ports, and troubleshooting. Release HTTP checks are recorded in
[launcher_v06_verification.json](../experiments/launcher_v06_verification.json).
The fixed demonstration passes all 10 desktop and 11 regional-memory
action-plus-reply cases, including restart without reteaching, changed object
pixels, and an unknown name. All 16 structural checks and 32 expected HTTP outcomes
pass, and the launched child processes are reaped. Saved memory and checkpoint
bytes remain unchanged by queries and restart. This is interface integration
evidence; browser rendering has not been visually verified.

The earlier 175.7-million-parameter CPU scaling benchmark remains a short
optimizer workload, not this release's trained model. Neither this release nor
the proposed local LLM study establishes the user's RTX 5080 computation limit.
