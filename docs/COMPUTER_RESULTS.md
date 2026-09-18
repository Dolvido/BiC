# BiC v0.3 computer-use results

The trained proof of concept achieved **88.3% joint action-and-reply success with familiar wording**, falling to **52.7% with held-out wording**. It can perform several small computer tasks and generate short replies, but its language generalization and color grounding remain unreliable. These results concern a controlled pixel desktop, not general intelligence, unrestricted conversation, or native operating-system control.

The delivered checkpoint (archive reference: `../runs/computer/checkpoint.pt`) has SHA-256:

```text
3a9bb01ddbd1313d16c12e4dee136bcc95f5a398a0d32a8b73756ddaf0e4ec6b
```

The final evaluation (archive reference: `../runs/computer/final_evaluation.json`) contains exact counts, per-task confidence intervals, settings, and sampled failures. The [evaluation script](../experiments/evaluate_computer_poc.py) reproduces its six conditions.

## What was trained

The model has **376,707 parameters**: 13 separately parameterized core regions, four interface networks for pixels, comprehension, production, and the semantic bridge, plus an auxiliary visual readout. The computer branch started from random initialization. It uses no downloaded language model, external AI service, or inherited v0.1 symbolic checkpoint weights.

The model receives 32 × 32 RGB screenshots of a 96 × 96 rendered desktop, UTF-8 prompt bytes, cursor position, field focus, and elapsed action steps. Its 11 actions select four predefined button locations, focus the field, enter `h`, `i`, `o`, or `k`, backspace, or stop. It cannot predict arbitrary pointer coordinates or use a general keyboard.

Training used 2,100 procedural demonstrations, yielding 3,519 action states and 2,100 completed-response examples. Action imitation, word-weighted next-byte prediction, visual naming supervision, and a small next-observation loss jointly trained the networks. Scene labels enter losses only; targets, task identifiers, teacher actions, and button metadata are not policy inputs. This is supervised learning, not reinforcement learning.

The delivered weights contain **9,000 updates**: 5,000 initial updates and 4,000 refinement updates. Refinement loaded the first-stage checkpoint (archive reference: `../runs/computer-development/perception-stage.pt`), restarted AdamW, increased stop-sampling probability to 0.35, and increased text-loss weight from 1 to 3. An earlier unsuccessful 2,500-update attempt was discarded and contributes no weights. The first-stage report (archive reference: `../runs/computer-development/perception_stage.json`) and refinement report (archive reference: `../runs/computer/report.json`) preserve this lineage.

Combined recorded training-loop time was **458.6 seconds**, excluding demonstration collection, evaluation, and the discarded attempt. This was one training seed, 31, on Linux with one CPU thread, Python 3.12.14, and PyTorch 2.14.0+cpu. No GPU or Windows performance was measured; these timings are not a benchmark of the user's Ryzen/RTX desktop. See the [runtime record](../experiments/v03_runtime.json) and [README](../README.md) for installation and two-stage commands. The README explains why the revised sampler prevents an exact replay of the original first-stage random sequence.

## Evaluation and results

Each condition contains **896 episodes: 128 for each of seven task families**. The model chooses every action, observes its consequences, and freely generates response bytes. No teacher corrects an evaluation trajectory. Action success requires the task outcome and an explicit stop; query tasks require stopping without other actions. Reply accuracy is exact match to the canonical response, and joint success requires both. Thus action success alone can be high for a question even when its answer is wrong.

The generator has 21 training template patterns that instantiate 51 distinct prompt strings. The test set has 14 held-out patterns instantiating 34 strings. Wording is held out, but task meanings, many words, and finite scene combinations overlap. Independent evaluation seeds do not create disjoint scene sets; scenes can repeat. Development used fresh scenes with training wording. Final conditions use seed 900031 after development decisions stopped.

All values below are percentages. Equal task counts make the macro averages equal to pooled episode rates. The 95% Wilson intervals describe joint success across sampled episodes; they do not measure training-seed variability, broad English competence, or uncertainty over unfamiliar applications.

| Condition | Action | Reply | Joint | Joint 95% interval |
|---|---:|---:|---:|---:|
| Familiar wording | 94.4 | 88.3 | 88.3 | 86.0–90.2 |
| Held-out wording | 83.1 | 53.5 | 52.7 | 49.4–55.9 |
| Layout/appearance shift | 94.6 | 88.7 | 88.7 | 86.5–90.6 |
| Pixels blanked | 75.1 | 59.6 | 40.7 | 37.6–44.0 |
| Prompt blanked | 10.5 | 6.4 | 6.4 | 4.9–8.2 |
| Hippocampus lesioned | 90.6 | 82.7 | 82.7 | 80.1–85.0 |

The shift moves button rectangles by up to two pixels and slightly changes background and button colors. Its similar score shows tolerance to that small combined perturbation; it does not establish transfer to a new interface. Blanking and lesion conditions otherwise use familiar wording and the standard layout.

| Task family | Familiar action | Familiar reply | Familiar joint | Held-out joint |
|---|---:|---:|---:|---:|
| Click a color | 60.9 | 60.9 | 60.9 | 62.5 |
| Click a position | 100.0 | 100.0 | 100.0 | 100.0 |
| Describe a position | 100.0 | 58.6 | 58.6 | 21.9 |
| Type `hi` or `ok` | 100.0 | 100.0 | 100.0 | 82.8 |
| Report last-click swatch | 100.0 | 98.4 | 98.4 | 45.3 |
| Greeting | 100.0 | 100.0 | 100.0 | 3.1 |
| Clarification examples | 100.0 | 100.0 | 100.0 | 53.1 |

Color selection and description are the main familiar-wording weaknesses, with recurring green/yellow confusions. Unseen descriptions sometimes provoke clicks instead of answers. Unseen greetings largely fail despite perfect familiar greeting scores. Finite clarification examples do not provide reliable recognition of arbitrary unsupported requests. The strong typing result covers only two short strings, with focus and elapsed-step cues that can support memorized sequences.

## Biology, controls, and observed conversation

Blanking pixels reduces joint success from 88.3% to 40.7%; blanking prompts reduces it to 6.4%. This demonstrates dependence on both supplied channels under these interventions, not comprehensive understanding. Remaining performance can use prompt patterns, body signals, and task priors. Blanking also creates an input distribution absent from training.

The hippocampal lesion reduces joint success to 82.7%, showing sensitivity to that pathway. It does not establish that the biological partition is advantageous: no parameter-matched retrained architecture comparison was run here. Last-click answers read a visible swatch and do not demonstrate episodic memory. Neural state is recomputed over up to three observations; only desktop state persists across turns. The gradient report also identifies unused classifier/value heads and an auditory input weight without observed nonzero gradients. Region names and parameter eligibility do not prove every component learned a useful biological function.

The recorded six-turn session (archive reference: `../runs/computer/demo.json`), with a visual record (archive reference: `../runs/computer/demo.png`), preserves an actual mistake. BiC greeted the user, clicked red, and answered “last was red.” Asked for the top-left button's color, it replied **“it is green.” although that button was yellow**. It then entered `hi` and replaced it with `ok`. Before entering `hi`, it issued an unnecessary backspace while the empty field was unfocused; that event had no text effect. The session was not corrected by a teacher. Ordinary chat runs inference and does not train model weights.

## Verification and remaining scope

The [verification record](../experiments/v03_verification.json) and [test log](../experiments/v03_tests.log) report **82 passing tests**. The original symbolic checkpoint remained unchanged. Separate [HTTP integration checks](../experiments/computer_ui_verification.json) passed with the actual saved policy, covering HTML, visible-state screenshots, chat requests, manual input, and rejected invalid requests. Cloud-browser access to the loopback interface was blocked with `ERR_BLOCKED_BY_CLIENT`; browser visual rendering was not verified.

This release is a trainable, inspectable language-and-action experiment with a local conversation interface. Broader progress requires reliable color grounding, new language structures, novel object names, hidden-state memory tasks, independent training replications, and transfer to richer interfaces. The present results do not meet the original general-intelligence ambition.
