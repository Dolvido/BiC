# BiC v0.7 feedback-adaptation pilot: prospective protocol

Status: protocol fixed before the sealed comparison. Date: 2026-09-16.
This file specifies an optional experiment, not an achieved capability or a
replacement for the existing v0.6 release. Results belong in a separate report.

## Question and scope

Can a small, freshly trained BiC use its recurrent state to select a rewarded
object category, and recover when that category changes without announcement?
Training imitates a causal, hand-designed candidate-set policy. During evaluation
all model weights are frozen: any adaptation occurs through recurrent state and
observed feedback. This is supervised imitation of a feedback-dependent strategy,
or a small meta-learning experiment. It is not reinforcement learning, online
weight learning, consolidation, novel-object recognition, or human intellect.

The existing v0.6 checkpoints, interfaces, and skills remain unchanged. Success
in this separate model does not establish integration with those skills or their
retention after future joint training.

## Observable task and hidden state

Each episode contains 40 choices. Four distinct categorical objects occupy four
cells; each category appears exactly once. The layout changes between choices.
Exactly one category is rewarded. Its identity changes once to a different
category at a hidden reversal step, independently of the policy's choices.

The public observation contains only:

- `layout`: the four visible category IDs in cell order;
- `previous_category`: the category actually selected on the preceding choice,
  or absent on the first choice;
- `reward`: the observed binary reward for that preceding choice, or absent on
  the first choice.

The input encoding uses a 16-value cell-by-category visual encoding, a four-value
previous-category encoding, an explicit feedback-presence flag, and the previous
reward. Unused channels are zero or declared constants. Neither the hidden
rewarded category nor the step, reversal time, phase, seed, remaining duration,
or candidate-set teacher state is an agent input. The previous choice and reward
are delivered only after that choice, so the current reward cannot select its
own action. Evaluation chooses a category and maps it to its visible cell.
This category-to-cell lookup is supplied by code. The learned policy can solve
the feedback problem while ignoring layout, so withheld-layout success does
not demonstrate learned spatial reasoning or perception.

Layout randomness must use a stream independent of the hidden rule schedule and
the agent's actions. Changing only the hidden rule schedule must leave the layout
stream unchanged. The reversal creates no episode reset, special token, change
flag, feedback-presence change, or state reset. A recurrent network can count
steps even without a time input; variable reversal times reduce this shortcut
but do not establish transfer to arbitrary switching processes.

## Frozen data, budgets, and checkpoint selection

| Item | Fixed value |
|---|---|
| Training seeds | 11, 23, 37, 51, 71 |
| Episodes | 40 choices each |
| Layout partition | Shuffle all 24 permutations with split seed 7001; 18 training/development layouts and six sealed layouts |
| Training corpus | 2,048 pregenerated trajectories per training seed; the same corpus for both architectures |
| Training environment seeds | `seed_index * 10000 + episode_index`, with zero-based seed and episode indices |
| Teacher action RNG | Separate from the environment and layout streams; record its seeds |
| Training reversal | Uniform integer from 12 through 28, inclusive |
| Training budget | 400 optimizer updates, batch size 64; sample trajectories uniformly with replacement |
| Optimization | AdamW; learning rate 0.003, weight decay 0.0001, gradient norm clipped at 1.0; mean soft-target cross-entropy over all choices |
| Development | 128 episodes, seeds in the 1,000,000 range; training-layout partition; reversal uniformly from 10 through 30 |
| Sealed evaluation | 256 episodes, seeds in the 2,000,000 range; only the six withheld layouts; reversal uniformly from 8 through 32 |
| Checkpoint candidates | After updates 100, 200, 300, and 400 |
| Selection rule | Highest development mean late reward; select the earliest checkpoint on an exact tie |
| Evaluation action | Argmax category; choose the lowest category ID on an exact tie |

Record the exact seed derivation, split permutations, observation encoding,
optimizer settings, corpus hash, parameter counts, source hashes, and selected
update in machine-readable artifacts. Architecture pairs and control conditions
must use the same evaluation episode seeds, layouts, and hidden rule schedules.
The development and sealed episode sets are separate from the training corpus.
No sealed episode, outcome, or layout is used to select a checkpoint or change a
hyperparameter. The six layouts exercise the interface under rearrangement of
known categories, not new categories, learned spatial transfer, or perception.

Development-only smoke runs may expose implementation defects or justify a
revision before the sealed comparison. Record every such change and its reason,
then freeze the final settings before looking at sealed outcomes. A changed
protocol after seeing sealed results requires a new, explicitly separate test;
the original results must remain available. Running the same sealed test again
after tuning is not a fresh confirmation.

## Causal training targets

The teacher begins with all four categories possible. A positive reward narrows
the set to the selected category. A negative reward removes the selected
category. If this contradicts the remaining singleton and empties the set, the
teacher starts a new candidate set containing the other three categories. This
rule uses only the public history; it does not receive the true rule or a
reversal signal. If a reversal occurs while the candidate set still contains
several categories, the teacher must handle the resulting uncertainty through
later observations rather than inspect the hidden state.

At each choice, the supervised target is a uniform distribution over the current
candidate set. It is not a one-hot label for the hidden rewarded category.
Pregenerated trajectories mix 25% uniformly random category choices with 75%
choices uniform over the causal candidate set. Keep this exploration RNG
separate from environment randomness. Both trainable models see identical
observations, teacher distributions, and trajectory budgets within each seed.

This is teacher-forced trajectory training. At evaluation each model supplies
its own choices and receives their real consequences. A distribution gap between
teacher and learner trajectories is a possible failure mode, not evidence of
environment leakage. Student roll-ins are outside this frozen pilot; adding them
requires a documented protocol revision before sealed evaluation.

## Models and controls

Train a fresh 13-region BiC with hidden width 16 and a monolithic GRU whose hidden
width gives the nearest practical total parameter count. Report actual counts,
optimization settings, update counts, and elapsed time. Equal parameter and
trajectory budgets do not guarantee equal compute or equal tuning effort.

Evaluate the following conditions:

- **Persistent BiC and persistent GRU:** retain state throughout each episode,
  including the reversal; reset at the next episode.
- **Reset state each choice:** use each selected model's unchanged weights but
  initialize recurrent state before every choice. Retain the same immediate
  previous-choice and reward observation; this isolates memory beyond one step.
- **Remove reward feedback:** use unchanged weights and persistent state, but
  remove reward information without giving any hidden-state cue. Keep the
  visible layout, previous choice, and ordinary first-step presence convention.
  Record the precise masking operation. This is an inference intervention, not
  a separately optimized reward-free baseline.
- **Candidate-set learner:** the same causal algorithm that supplies training
  targets, operating on its own observed history, with the declared action rule.
  It is an engineered reference and the source strategy, not an independent
  discovery by BiC.
- **Uniform random choices:** a seeded policy with expected reward 0.25.

Reset-state policies can learn to repeat a previously rewarded choice. High late
reward alone therefore does not prove a benefit from longer memory. Neither an
advantage of BiC over the GRU nor a particular region's biological function is
assumed.

## Metrics and declared interpretation

Compute metrics from complete on-policy evaluation traces. A phase is the stable
segment before or after the hidden reversal. The evaluator may use phase
boundaries for scoring; those boundaries remain unavailable to the agent.

- **Full reward:** mean binary reward over all 40 choices.
- **Early reward:** mean reward in the first four choices of each phase, reported
  separately for initial acquisition and reversal recovery.
- **Late reward:** mean reward in the last eight choices of each phase, reported
  separately. The checkpoint score and primary gate average those two phase
  means equally. At the shortest test phases, early and late windows overlap;
  report this rather than treating them as independent samples.
- **Post-reversal reward:** mean reward over the complete second phase.
- **Recovery latency:** number of post-reversal choices through the third
  consecutive rewarded choice in the first such streak. If no streak occurs,
  record `second_phase_length + 1` and a separate failure indicator. Report
  recovery failure rate alongside the capped latency; a conditional mean among
  successes alone would hide failures.
- **Repeated errors:** a failed selection of a category already observed to fail
  earlier in the same stable phase. Report mean counts by phase and a rate with
  its explicit denominator. The hidden phase boundary is used only to score;
  it must not reset the policy's memory or candidate set.

The primary pilot gate is mean persistent-BiC late reward at least 0.75 and at
least 0.15 above its same-weights reward-removed control. Average episode metrics
within each training seed first, then give the five seed results equal weight.
Report every seed, the mean and sample standard deviation, and paired differences
against controls. These are descriptive results from five trained models; do not
treat individual trials as independent training replications.

The longer-memory hypothesis is assessed using acquisition, repeated errors,
and recovery latency against reset-state controls. There is no predeclared
minimum late-reward margin against reset-state and no post-hoc replacement of
the primary gate with whichever metric improves. Report mixed or null findings
plainly. Compare BiC with the GRU under the same protocol without requiring or
presuming architectural superiority.

## State integrity and release checks

Persistent state must include every recurrent tensor and the bounded hippocampal
history needed to continue a trajectory. State belongs to the caller, not a
shared mutable model global. Interleaved episodes must remain independent.
Specify explicit graph-detachment boundaries during training; serialization
must contain no retained computation graph.

Before interpreting evaluation results, require:

1. One uninterrupted trajectory and the same trajectory split across calls have
   equivalent outputs under declared deterministic settings.
2. Saving and restoring state yields bitwise-identical subsequent outputs and
   choices on the same device and configuration; report any platform limitation
   explicitly rather than substituting an undisclosed tolerance.
3. An independently reset episode reproduces the initial-state behavior; no
   history leaks between episodes or batches.
4. Counterfactual tests change hidden rules while preserving public observations
   and verify that encoding and teacher targets remain identical until observed
   feedback differs.
5. The ordinary v0.6 path and its selected checkpoint are unchanged. Report
   regression checks separately from the new optional model's scores.

Passing the primary gate and integrity checks would establish only that this
small learned policy uses feedback to adapt on this declared categorical task.
It would not establish general-purpose learning, autonomous consolidation,
language comprehension, human-like cognition, or improvement to v0.6 skills.
