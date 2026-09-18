# BiC v0.7: measured feedback adaptation with frozen weights

Date: 2026-09-16. This is the completed optional pilot described in
[ADAPTATION_PROTOCOL.md](ADAPTATION_PROTOCOL.md). The complete numerical record
is runs/adaptation-v07/report.json (archive reference: `../runs/adaptation-v07/report.json`).

The learned BiC policy adapted to a hidden rewarded category and recovered after
an unannounced change. Across five training seeds, its late reward was
**97.8809% ± 0.5914%**, and its reward over all choices was **90.6484% ± 0.8195%**.
Resetting its state before every choice reduced late reward to
62.2314% ± 10.0704%; removing reward information reduced it to
25.0195% ± 0.4720%. Both predeclared reward thresholds and all recorded
restart/unchanged-weight checks passed.

This demonstrates fast adaptation through recurrent activity on a small symbolic
task. The policy learned offline by imitating a scripted candidate-elimination
strategy. Its weights did not learn during evaluation. It did not acquire new
object concepts, learn from reward alone, consolidate new knowledge, or integrate
this behavior into the existing v0.6 controller.

## What was trained and tested

Each 40-choice episode presents the same four categorical objects in shuffled
cells. One category is rewarded, then a different category becomes rewarded at
an undisclosed step. Agent inputs contain only the visible layout and the previous
selected category and reward. The agent receives no target, step number, phase,
reversal flag, or teacher candidate set. A supplied category-to-cell lookup turns
the neural category output into a cell selection; visual recognition and spatial
action mapping are not learned here.

The five training seeds were 11, 23, 37, 51, and 71. Each architecture received the
same 2,048 pregenerated trajectories for a given seed, 400 optimizer updates,
batch size 64, and optimizer settings. Training labels were uniform distributions
over categories still consistent with the teacher's observed choice/reward
history, never one-hot labels for the hidden rewarded category. Demonstrations
mixed 25% random choices with 75% candidate-set choices.

Checkpoints were selected using 128 development episodes, at updates 100, 200,
300, and 400, by highest development late reward with an earliest-update tie
break. All ten model selections finished before sealed evaluation. BiC seed 11
selected update 300; the other four BiC seeds and all five GRU seeds selected
update 400. There were two earlier development-only smoke runs, for seed 11 at
100 and 400 updates. Their saved protocols used the same learning settings, and
neither directory contains sealed test results; no hyperparameter change followed
those smoke runs.

Every reported condition used the **same 256 sealed scenarios**, seeds 2,000,000
through 2,000,255. They contain six layouts withheld from the 18-layout
training/development set. The reversal step is uniformly sampled from 8 through
32, compared with 12 through 28 during training and 10 through 30 in development.
These are 256 shared scenarios, not 1,280 independently drawn test scenarios.
The same scenario is run by each of the five trained models and their controls.
Withheld layouts exercise the supplied interface; since outputs denote
categories, successful performance does not establish learned spatial transfer.

The saved protocol (archive reference: `../runs/adaptation-v07/protocol.json`),
training records (archive reference: `../runs/adaptation-v07/training-summary.json`),
source hashes (archive reference: `../runs/adaptation-v07/source-hashes.json`), and source snapshot
preserve the exact implementation and settings used.

## Reward results and all controls

Neural results below are the mean ± sample standard deviation of five seed-level
means. Values are percentages, rounded to four decimal places; the JSON report
retains full precision. The scripted and random policies were each evaluated
once on the shared 256 scenarios, so they have no across-training-seed SD.
These SDs are descriptive variation across training runs, not confidence
intervals, and individual choices are not independent training replications.

| Policy / condition | All 40 choices | Late, both phases | Initial-phase late | Post-reversal late |
|---|---:|---:|---:|---:|
| BiC, persistent state | 90.6484 ± 0.8195 | 97.8809 ± 0.5914 | 98.7305 ± 0.3829 | 97.0312 ± 1.0245 |
| BiC, reset each choice | 59.8691 ± 9.5052 | 62.2314 ± 10.0704 | 74.5898 ± 17.6427 | 49.8730 ± 3.2546 |
| BiC, reward removed | 25.0312 ± 0.2896 | 25.0195 ± 0.4720 | 25.0293 ± 0.2643 | 25.0098 ± 0.7229 |
| GRU, persistent state | 58.0840 ± 5.9000 | 63.1299 ± 7.7550 | 82.2168 ± 11.9790 | 44.0430 ± 4.7483 |
| GRU, reset each choice | 64.1211 ± 6.3480 | 68.9697 ± 8.3989 | 73.7598 ± 5.6973 | 64.1797 ± 12.0354 |
| GRU, reward removed | 25.3652 ± 2.1686 | 25.5811 ± 2.3952 | 26.2402 ± 1.8337 | 24.9219 ± 3.5737 |
| Scripted candidate learner | 91.3184 | 98.1689 | 98.5840 | 97.7539 |
| Seeded uniform random | 25.5078 | 25.1465 | 26.1230 | 24.1699 |

Late reward averages the last eight choices of each stable phase, with equal
weight for the two phases. The first four choices form the early windows.
At the shortest phase length of eight choices, the late window contains the
entire phase, including acquisition; early and late windows overlap in short
phases and are not independent samples. This also explains why even the
scripted reference's late reward is below 100%.

For persistent BiC, reward in the first four choices was 62.4609% ± 1.0509% during
initial acquisition and 46.4453% ± 4.4219% after reversal. Reward over the complete
post-reversal phase was 86.1962% ± 2.0008%. The phase-level values, rather than
only the high late score, retain the cost of discovering and relearning the rule.

Reset-state controls retain the immediate previous category and reward; they
remove longer recurrent history. Reward-removed controls preserve the previous
category and presence flags while setting the reward component to zero. These
are interventions on the selected models, not independently retrained controls.

## Recovery and repeated mistakes

Recovery latency counts choices after reversal through the third success in the
first streak of three consecutive successes. Its minimum is three. If no such
streak occurs, the recorded value is the second-phase length plus one, and the
episode receives a failure indicator. Thus the reported latency includes capped
failure values; it is not a mean restricted to successful recoveries or an
estimate of uncensored time to recovery. A three-success streak also does not
guarantee all later choices remain correct.

A repeated error is a failed choice of a category that had already failed
earlier in the same true stable phase, including when successes intervened.
The evaluator uses the hidden phase boundary only for scoring. It never resets
the agent at that boundary. The repeated-error rate divides the count by that
phase's number of choices, then averages over scenarios and training seeds.

| Policy / condition | Recovery choices | Recovery failure, % | Repeated errors per second phase | Repeated-error rate, % |
|---|---:|---:|---:|---:|
| BiC, persistent state | 5.2375 ± 0.3049 | 0.2344 ± 0.5241 | 0.2414 ± 0.3321 | 1.5227 ± 2.1260 |
| BiC, reset each choice | 12.0984 ± 0.4863 | 49.6875 ± 3.3535 | 8.4812 ± 0.6982 | 43.0892 ± 3.1282 |
| BiC, reward removed | 20.3695 ± 0.0472 | 99.9219 ± 0.1747 | 11.5281 ± 0.0745 | 56.5415 ± 0.4465 |
| GRU, persistent state | 14.0883 ± 0.4386 | 54.7656 ± 5.5477 | 10.5930 ± 0.6131 | 50.5963 ± 3.9235 |
| GRU, reset each choice | 10.2516 ± 1.6217 | 34.4531 ± 12.7310 | 6.2602 ± 2.0305 | 31.5349 ± 9.7173 |
| GRU, reward removed | 16.0461 ± 0.5875 | 75.0781 ± 3.5737 | 13.7969 ± 0.6231 | 70.4828 ± 3.3498 |
| Scripted candidate learner | 4.9531 | 0.0000 | 0.0000 | 0.0000 |
| Seeded uniform random | 18.0664 | 81.6406 | 11.5391 | 56.9594 |

Persistent BiC had three recovery failures among 1,280 model-scenario pairs, all
from training seed 37. The scenarios are shared across models, so these pairs
must not be treated as 1,280 independent scenarios. Initial acquisition had zero
recovery failures and zero repeated errors in all five persistent BiC runs.

The paired BiC difference against reset-state was **35.6494 ± 9.6324 percentage
points** of late reward, **6.8609 ± 0.3648 fewer** capped recovery choices, and
**8.2398 ± 0.5589 fewer** repeated errors per second phase. These controls support
a useful role for recurrent history in this trained BiC policy. The GRU's
reset-state condition performed better than its persistent condition after
reversal, so persistence was not universally helpful in the learned models.

## Five seeds, selection, and gates

The table reports persistent models' full and late reward and both BiC ablations'
late reward. All numbers are percentages. The complete per-seed metric sets for
both architectures and all conditions are preserved in the JSON report and
individual `*-test.json` files.

| Seed | BiC full | BiC late | BiC reset late | BiC reward-removed late | GRU full | GRU late |
|---|---:|---:|---:|---:|---:|---:|
| 11 | 89.9609 | 97.5586 | 65.5273 | 24.4141 | 63.8281 | 70.9229 |
| 23 | 91.2305 | 98.4131 | 60.6934 | 24.7559 | 60.0391 | 65.2344 |
| 37 | 89.5703 | 97.0459 | 46.8506 | 25.5615 | 55.2832 | 58.9111 |
| 51 | 91.3184 | 98.4375 | 74.6582 | 25.4150 | 62.0605 | 68.7012 |
| 71 | 91.1621 | 97.9492 | 63.4277 | 24.9512 | 49.2090 | 51.8799 |

The predeclared primary thresholds were mean late reward at least 75% and at
least 15 percentage points above the same-weights reward-removed control.
The measured values were 97.880859375% and a paired gain of
72.861328125 percentage points; both passed. The SD of that paired gain was
0.814603284 percentage points. There was no threshold requiring superiority
over the GRU, and no post-hoc metric substituted for the primary gate.

The interactive lab uses **BiC seed 23, update 400**, selected by the highest
development late reward, 99.90234375%, with seed-order ties as specified in the
lab selection rule (archive reference: `../runs/adaptation-v07/lab-selection-rule.json`). This was
not a selection of the best sealed score. The
selected-model record (archive reference: `../runs/adaptation-v07/selected-model.json`) binds the
demonstration checkpoint to its original file hash. The five-seed results above
remain the capability report rather than the selected demonstration's score.

## Resources, persistence, and audit

The BiC model has 19,566 parameters; the nearest GRU configuration has 19,367.
Each received the same corpus, batches, and 400-update budget. On this machine,
using CPU execution with one PyTorch thread, training-loop elapsed time was
**30.6431 ± 0.0599 seconds** for BiC and **3.2435 ± 0.0727 seconds** for the GRU.
These timings include development evaluations and checkpoint writes, but exclude
corpus creation and sealed evaluation. BiC took about **9.45 times** as long.
The comparison matched parameters and updates, not elapsed compute; it does not
establish that the architecture is generally superior or more efficient. The
weaker GRU was not subsequently retuned or given a larger compute budget.

For all ten selected neural checkpoints, restoring saved activity and environment
state after choice 13 preserved every subsequent action, reward, and observation,
and the final neural-state tensors matched exactly. Model hashes were unchanged
through test evaluation and the restart check. The separate state tests also
check bitwise next-output continuity after save/load into a fresh same-weights
CPU model. No cross-device bitwise guarantee is inferred.

An independent read-only audit recomputed every metric from all 32 saved test
files: 8,192 model-scenario records and 327,680 reward decisions. It checked the
reward traces against the seeded hidden rules, sample SDs, paired differences,
development-only checkpoint choices, frozen source hashes, and the selected
checkpoint hash. All checks passed. This was an audit of saved results, with no
retraining, tuning, or additional evaluation rollout.

The source strategy remains slightly better than persistent BiC: 98.1689% versus
97.8809% late reward, with no repeated errors for the scripted learner. The
demonstrated milestone is therefore a neural approximation of an explicit causal
feedback strategy, with useful memory and verified save/resume. It is not an
independently discovered general learning algorithm. These four known categories,
one reversal per short episode, deterministic rewards, supplied symbol identities,
and supplied action mapping leave substantial work before broader adaptation.
The original v0.6 checkpoint has not been jointly trained with this model, and
retention in a combined old-and-new-skill model remains unmeasured.
