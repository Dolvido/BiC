# V0.5 measured results

BiC now uses persistent visual associations through its existing regional
controller. On the new sealed two-by-two-board test, the delivered model chose
the correct action in **1,272/1,280 cases (99.38%)** and produced both the correct
action and exact freely generated reply in **1,267/1,280 (98.98%)**. Rehearsing old
desktop examples greatly reduced forgetting, although some older phrasing
regressed. These are controlled task results, not general English competence.

## New memory-guided actions

The task is to select a taught object or its immediate neighbor to the left,
right, above, or below. Missing names, absent objects, duplicate referents, and
out-of-board neighbors should produce STOP. Four raw visual similarities and a
known-name flag enter a learned adapter and the hippocampal output
transformation. Existing prefrontal, basal-ganglia, and motor networks select the
action; a byte decoder generates the reply. Neither a chosen target index nor
an acceptance-threshold decision enters this policy.

The sealed set uses **256 new glyph identities** excluded from all earlier
encoder partitions and the new controller's training/development identities.
Its normalized instruction templates are also held out. The model receives one
taught exemplar per query. Appearance families, grid positions, and the five
relations are familiar task structure. Exact quoted names are symbolic memory
addresses masked before neural comprehension; new names do not establish neural
vocabulary acquisition. Protocol (archive reference: `../runs/regional-memory-v05/protocol.json`),
[architecture and use](REGIONAL_MEMORY.md).

| Sealed case type | Cases | Correct action | Exact generated reply |
|---|---:|---:|---:|
| Valid cell selection | 640 | **99.38%** | 99.22% |
| Untaught name: STOP | 160 | **100.00%** | 100.00% |
| Taught object absent: STOP | 160 | **97.50%** | 97.50% |
| Duplicate referent: STOP | 160 | **100.00%** | 98.13% |
| Neighbor outside board: STOP | 160 | **100.00%** | 100.00% |
| All cases | 1,280 | **99.38%** | **99.06%** |

Half the cases need a click, so always stopping scores 50% overall and 0% on
valid selections. The eight action errors were **four false clicks on absent
objects** and **four refusals on valid requests**. Correct STOP behavior in all
sampled ambiguous cases is not a guarantee: three such cases still generated
an incorrect reply. Action accuracy and response accuracy should remain
separate measures. Full evaluation and examples (archive reference: `../runs/regional-memory-v05/final_evaluation.json`).

Considering only valid selections, direct finding succeeded in 126/128 cases,
left and above in 127/128 each, and right and below in 128/128 each. These counts
were reconstructed from the complete eight-error log and the frozen category
balance; no further inference or training was performed.
Reporting supplement (archive reference: `../runs/regional-memory-v05/reporting_supplement.json`).

## Rehearsal and older-task retention

Both arms began with identical v0.4 computer/language weights and recall-adapter
initialization. Each completed 3,000 new-task updates of 24 examples, using
identical new-task sample and wording streams. The replay arm additionally
rehearsed 16 old desktop examples per update. This controls new-task exposure,
not total data or computation.

The table uses the **3,000-update checkpoints for both arms**. Each old-task
condition contains 64 episodes for each of seven tasks, 448 in total. Joint
success requires both correct behavior and an exact freely generated reply.

| Matched evaluation | Original v0.4 | V0.5 with replay | V0.5 without replay |
|---|---:|---:|---:|
| New sealed action accuracy | Not evaluated | **99.38%** | 98.83% |
| New sealed joint success | Not evaluated | **98.98%** | 98.36% |
| Old familiar-phrasing joint success | 99.78% | **99.11%** | 19.64% |
| Old reserved-phrasing joint success | 98.88% | **95.98%** | 13.62% |

The same updated computer/language networks perform the old tasks; the
evaluation does not switch back to a frozen v0.4 controller. The desktop CLI can
load them from the delivered regional checkpoint.

**Retention is incomplete.** On the older reserved-phrasing test, greeting joint
success fell from **100% to 81.25%** (52/64), despite replay. Examples include
answering `hey again` with `please rephrase.` and producing an unrelated color
reply to `greetings there`. Other failures include a color click and incorrect
color descriptions. Typing joint success on these matched samples changed from
92.19% to 93.75%. The overall retention result should not hide the greeting
regression or imply every old capability improved.

Development selection used the mean of new-task action accuracy and old-task
joint accuracy. It selected **step 3,000 with replay** and **step 500 without
replay**. That earlier no-replay checkpoint reached 90.31% new action accuracy,
57.37% old familiar joint success, and 50.67% old reserved-phrasing joint success.
Those selected-checkpoint numbers answer a different question from the matched
3,000-update comparison above. Replay's selected and final checkpoint files
have different metadata/hashes but exactly identical model tensors.

## Causal controls on the delivered model

| Intervention | All-case action accuracy | Valid-selection action accuracy |
|---|---:|---:|
| Intact model | **99.38%** | **99.38%** |
| Remove complete external recall | 50.00% | 0.00% |
| Silence hippocampus | 50.00% | 0.00% |
| Silence temporal-language region | 50.00% | 0.00% |
| Empty instruction | 12.50% | 25.00% |
| Zero similarity evidence | 50.00% | 0.00% |
| Permute similarity positions | 43.44% | 0.00% |

Removing recall or silencing hippocampus leaves correct abstentions but no
correct valid selections. The successful policy therefore depends on that
pathway. Empty instructions and a temporal-language lesion differ because an
empty string still receives the network's learned text representation.

The report's `zero_encoder` control directly zeros cached similarity scores,
retaining name availability; it does not blank raw images or retrain an encoder.
`shuffled_memory` permutes the four score positions as `[2, 3, 0, 1]` with the
scoring target unchanged; it tests score-to-cell correspondence, not a shuffled
exemplar bank. These are acute interventions in one fitted model, not proof of
anatomical equivalence or necessity of all thirteen regions.

## Interface, persistence, and implementation checks

A fixed real HTTP demonstration completed **50/50 correct actions and 50/50
correct neural replies**. It covered 20 commands after teaching, 20 after
restarting into a different operating-system process, five untaught-name
requests, and five absent-object requests. The four object identities, names,
seed, and requests were fixed before evaluating the model. All object pixels
and their order changed after restart, with **zero reteaching calls**. The
memory, encoder, and controller files remained byte-identical. All 18 structural
checks passed. [HTTP verification](../experiments/regional_ui_verification.json).

This demonstration shares the new sealed identity pool and is a deployment
check, not an independent generalization estimate. Its
recorded illustration (archive reference: `../runs/regional-memory-v05/ui_demo.png`) presents HTTP
data; it is not a browser screenshot. Browser rendering was not verified.

The computer HTTP interface also passed four greeting, color selection,
last-click recall, and typing probes using the same v0.5 checkpoint.
[Computer interface verification](../experiments/computer_ui_v05_verification.json).
These four probes do not negate the broader greeting regression above.

The full suite completed **121 tests in 12.368 seconds, all passing**.
[Test log](../experiments/v05_tests.log). Independent checks established that
action-loss gradients reach the recall adapter, hippocampal output, prefrontal
cortex, basal ganglia, and motor cortex; replacing the motor logits controls the
public action; hippocampal lesions remove external recall; and checkpoint
restoration preserves shared computer state. An additional archived-v0.4
compatibility probe found exact equality for all 23 returned tensors when recall
was omitted. [Compatibility record](../experiments/v05_compatibility.json).

## Compute and biological interpretation

The delivered model contains **558,467 parameters**: 376,995 eligible for
optimization and 181,472 in the frozen glyph encoder. The new recall adapter
adds only 288 parameters to the existing computer and visual-association
networks. Parameter eligibility does not mean every auxiliary head received a
task-learning signal.

On the available CPU host, using one thread, training plus development checks
took **203.0 seconds with replay** and **109.0 seconds without replay**. Preparing
the cached training/development evidence took **10.7 seconds** separately.
Training report (archive reference: `../runs/regional-memory-v05/training_report.json`). This is a
small learned system; the 175.7-million-parameter workload in v0.4 was a separate
execution benchmark. This run did not benchmark the user's Ryzen/RTX desktop,
measure its GPU ceiling, or exhaust home-computer capacity.

The biological inspiration is rapid association feeding a slower, changing
controller, with rehearsal reducing interference from new learning. Stored
exemplars remain external, the encoder stays frozen, and global backpropagation
updates the controller. The experiment demonstrates neither sleep nor transfer
of taught names into cortical weights. It uses one training seed per arm and a
finite family of synthetic tasks. The greeting regression motivates better
retention testing; removing stored exemplars while testing retained knowledge
would be necessary for a subsequent consolidation claim.

Delivered checkpoint SHA-256:

```text
6041ee317d6ef9c8cba6b85154f8e6b675a6cee4ec26138bc210bfb2f79057a3
```
