# Architecture and general learning: calibrated comparison

Compare recurrent regional BiC, episodic regional BiC, and an independent causal
sequence learner. The sequence model is a diagnostic reference, not a BiC
checkpoint or evidence that regional abilities have been integrated. This study
tests supervised learning, transfer, and retention in finite verified worlds;
it does not define or demonstrate general intelligence.

## Learning-rate calibration and fresh main runs

Train the same three subjects: `variable_binding`, `graph_reachability`, and
`conditional_logic`, at levels 1 and 2. Entirely withhold `arithmetic_updates`
from calibration and main training. The learners receive raw observed English,
never family identity, labels, parser output, teacher replies, or oracle memory
as inference input. All use the shared action, acknowledgement, reply-byte, and
within-utterance next-byte objectives, with their existing declared weights.

Before any calibration update, freeze all sources, banks, seeds, rates, scoring
rules, and audit boundaries. For each architecture, run learning rates
**0.0003, 0.001, and 0.003**, each from fresh seed **2401** for exactly **600
updates**, batch **64**. Select the rate using the fixed final development score:

```
0.5 * mean_family(counterfactual_query_pair_accuracy)
+ 0.5 * mean_family(last_known_later_query_accuracy)
```

The last-known-later query is each episode's last known yes/no query, included
only when its ordinal among all queries is at least two. A missing denominator
cannot receive a favorable score. Ties select the lower rate. Save all nine
calibration endpoints and their scores; do not choose intermediate checkpoints.
This finite grid calibrates a short training budget, not each architecture's
best achievable result.

Each main learner then starts **fresh from seed 2501**, without calibration
weights or optimizer moments. Use its selected rate for **3,600 updates**,
batch **64**, checkpoint every **600**. Interleave the three families in the
same order, with matched complete-pair sample streams and exposure fingerprints.
Each main run consumes 230,400 repeated episodes. Architectures need not have
identical tensors or parameter counts; report the measured counts and both
exposure and compute costs. Equal exposure is not equal compute.

No score-based early stopping or post-hoc checkpoint choice is permitted.
Interruptions preserve the latest state; unfinished runs cannot enter final
audit as full-budget candidates. Nonfinite training or failed provenance checks
are reported as failures rather than hidden exclusions.

## Frozen banks

Family order is binding, graph, conditional. For the table's base seed, add
100,000 per family index and 10,000 per level. A single arithmetic bank uses
family index zero. All counts preserve complete even/odd counterfactual pairs.

| Bank | Base seed | Split | Levels | Episodes per family/level |
|---|---:|---|---|---:|
| Calibration and main training | 32,000,000 | train | 1, 2 | 512 |
| Rate selection and main development | 33,000,000 | dev | 1, 2 | 64 |
| Original-subject retention | 34,000,000 | dev | 1, 2 | 128 |
| Advanced original-subject transfer | 35,000,000 | audit | 3 | 256 |
| Arithmetic adaptation support | 36,000,000 | train | 1, 2 | 64 |
| Arithmetic final queries | 37,000,000 | audit | 2, 3 | 256 |

The arithmetic support bank has **128 episodes**, not 64. Adaptation batches
sample 64 episodes with replacement as whole pairs, so exposure counts are
distinct from support size. Query banks each contain 256 episodes before any
whole-pair overlap exclusions. Record exact fingerprints, distinct transcripts,
label counts, and all exclusions before any final model scores are inspected.

The v2 curriculum remains unchanged. Splits withhold primary ordered entity
pairs while retaining a shared vocabulary and allowing incidental pair overlap.
Level 3 means longer paths or higher-arity rules in some families, and new copy
operations/syntax in binding and arithmetic. It is not uniformly a pure depth
test. Neutral extra switches make initial conditional queries solvable from the
first switch alone. Report later-query behavior explicitly. Fresh seeds on this
inspected finite grammar are new scenarios, not wholly new research benchmarks.
Arithmetic also introduces digit bytes absent from the three training subjects.
Its held-out scores therefore mix subject transfer with adaptation to unfamiliar
input symbols; they do not isolate a purely conceptual arithmetic mechanism.

## Withheld-family adaptation and controls

Only after all nine calibration runs and all three selected-rate main runs
finish may the audit inspect arithmetic performance. Adapt isolated copies of
each main endpoint and its own fresh seed-2501 initialization. Reset optimizer
moments for all six copies, use that architecture's selected learning rate,
sampler seed **3501**, and the same 128-episode support bank.

Score cumulative budgets **0, 1, 4, 16, and 64 updates**, corresponding to 0,
64, 256, 1,024, and 4,096 repeated support exposures. Intermediate query scores
never change the schedule, objective, support set, or endpoint. All action and
reply evaluations start with BOS-only decoder prefixes; free replies are
generated separately. The student never receives reference reply prefixes in
evaluation. Original training checkpoints remain unchanged.

Report level-2 and level-3 curves separately and as equal-level macro averages.
Use normalized trapezoidal area over the observed update axis, dividing by 64;
the 16-to-64 interval dominates this sparse summary. Compare each pretrained
curve with its own architecture's fresh curve at the same selected rate.
Architecture-specific rates mean these are equal-update comparisons, not equal
step-size trajectories or direct evidence of a learned optimizer.

Measure all trained families before and after adaptation on the retained and
advanced banks. Record query accuracy, query-position counterfactual pairs,
last-known-later accuracy and paired correctness, per-family and per-position
known/unknown results, ASK precision/recall and denominators, log loss, Brier
score, freely generated reply accuracy, and action/reply agreement. Correct
actions and correct language are separate outcomes. Pair accuracy does not
require an entire conversation to be correct; agreement can mean both outputs
are wrong.

At update 64, blank English and reset history between utterances, preserving
canonical labels. Reset-history presents each unchanged utterance alone; it
does not replace the input with zeros. Verify exact CPU restart after turn
three. Regional models persist regional/episodic state; the sequence model
persists the observed prefix and recomputes it after restoration. Report that
difference rather than claiming identical session mechanisms or inference cost.

## Interpretation and next decisions

The calibration score chooses only a rate. It is not a promotion score or proof
of general learning. Show every family and later-query slice, final-budget
results, earlier-subject forgetting, and fresh-control comparisons. A gain
confined to initial conditional questions, an ASK-heavy strategy, or an unused
language/action output cannot establish a generally better architecture.

Treat architecture differences as exploratory: one calibration seed, one main
seed, one held-out family, one support draw, and shared scenario banks. A
promising result needs replication and rotated family holdouts before a robust
architecture claim. No automatic promotion follows aggregate query accuracy.
The established requirements for substantial per-family query/pair competence,
context dependence, useful uncertainty, and retention remain visible even when
one learner is relatively better than another.

Record the full cost of **all nine calibration runs**, three main runs, and six
adaptations, alongside evaluation time. Distinguish wall time from summed
worker intervals and document GPU concurrency. A faster or smaller reference
may inform implementation choices, but it is not automatically a replacement
for BiC. Keep all previous frozen studies and accepted checkpoints intact.

Use local compute only. Training worlds remain inert names, lamps, graphs, and
amounts. Candidate selection values accurate learning, retained abilities,
truthful uncertainty, and efficiency. No harmful external action, simulated
distress, autonomous alteration of audit truth, or unsupported sentience claim
is part of this study.
