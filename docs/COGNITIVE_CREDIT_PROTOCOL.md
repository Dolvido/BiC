# Shared decision teaching: prospective local comparison

This follow-up tests one learning mechanism across subjects. A disposable
prefrontal decision readout supplies a shorter supervised training path; every
official answer still comes from the original regional brain and motor head.
The preceding episodic-attention study did not identify a causal bottleneck.
This is a new hypothesis, not a repair known to work or an intelligence claim.

## Frozen learner and exposure

Both arms instantiate the same recurrent-memory student and the same learned
64-to-4 auxiliary head. Compare auxiliary loss weights **0.0 and 0.3**; retain
the shared query-class, acknowledgement, reply-byte, and causal input-next-byte
objectives. The auxiliary objective is query-only, answer-class-balanced cross
entropy using the existing action labels, with the same rule in every family.
Acknowledgements do not enter that auxiliary loss. No parsed facts or family-specific semantic readouts enter
the student. The auxiliary head is ignored by all official scoring, generated
replies, and inference-state transitions.

Train `variable_binding`, `arithmetic_updates`, and `conditional_logic` at
levels 1 and 2. Reserve **`graph_reachability`** completely from main training.
Use seed **2301**, **1,800 interleaved updates**, batch **64**, learning rate
**0.003**, and checkpoints every **300** updates. Each arm consumes the same
600 seeded complete-pair minibatches per family, with identical initial weights
and exposure fingerprints. The base policy and auxiliary initialization must
both match. No score-based early stopping or best-checkpoint selection is used.

The existing v2 procedural curriculum is preserved. Its finite grammar, shared
vocabulary, and incidental entity-pair overlap limit interpretation. Conditional
initial states use neutral additional switches, so their first query alone does
not establish Boolean composition. Advanced binding and arithmetic introduce
copy operations and syntax; they are not pure depth extrapolations. This study
changes the learning objective without silently repairing the benchmark.

## Frozen banks and audit boundary

For each seed below, add 100,000 per family index and 10,000 per level. A
single held-out family uses family index zero. Complete pairs start at even
episode seeds. Canonical bank manifests preserve raw text, verified targets,
source hashes, sample streams, and exact transcript counts before optimization.

| Bank | Base seed | Split | Levels | Episodes per family/level |
|---|---:|---|---|---:|
| Main training, three families | 20,000,000 | train | 1, 2 | 512 |
| Main development, three families | 24,000,000 | dev | 1, 2 | 64 |
| Advanced transfer, three families | 26,000,000 | audit | 3 | 256 |
| Graph adaptation support | 27,000,000 | train | 1, 2 | 32 |
| Graph final queries | 28,000,000 | audit | 2, 3 | 256 |
| Retention, three families | 29,000,000 | dev | 1, 2 | 64 |

The three-family order is binding, arithmetic, conditional logic. Primary
ordered entity pairs are disjoint across train/dev/audit supports; incidental
pairs and words need not be. Remove a whole evaluation pair if either exact
transcript overlaps its training/adaptation support, recording all exclusions
before scoring. Fresh seeds on this previously inspected grammar are new
scenarios, not evidence from a previously unknown task family or language.

Preparation must precede both initial training checkpoints. The audit refuses
incomplete arms, mismatched budgets, different initializations/exposures, changed
source or bank hashes, and accidental graph training. It runs only after both
fixed endpoints complete. Evaluation may inspect intermediate adaptation budgets
but cannot change the support, endpoint, objective, or candidate from those
answers. Source and checkpoint hashes remain unchanged by evaluation.

## Adaptation and official scoring

Create isolated copies of both endpoints and one fresh seed-2301 initialization.
All three adapt to the same 64 graph support episodes with **auxiliary weight
zero**, fresh AdamW moments, learning rate 0.003, sampler seed **3301**, and
batch 64. Sample complete pairs with replacement. Record cumulative budgets
**0, 1, 4, and 16** updates: 0, 64, 256, and 1,024 support exposures.
Never overwrite the general-training checkpoints.

Score graph levels 2 and 3 separately and as an equal-level macro average.
The normalized adaptation area uses trapezoids between the four fixed update
counts, divided by 16; its last interval receives the most weight. Compare it
with the fresh-initialization floor as well as the matched zero-auxiliary arm.
This measures few-step transfer under an ordinary optimizer, not an implemented
meta-learning algorithm.

Before adaptation, rescore the fixed main development banks using the original
policy. Measure three-family retention and advanced transfer both before and
after adaptation. Report query and paired correctness, per-answer-class scores,
ASK precision/recall and denominators, query loss, Brier score, exact freely
generated replies, and action/reply agreement. A correct generated reply and a
correct action are separate outputs; agreement can also mean both are wrong.

The audit exposes per-turn and per-query-ordinal known/unknown slices. A paired
score requires both opposite answers at a given query position, not both entire
conversations. Also define **last known later query** precisely: select each
episode's last query whose target is yes/no; include it only if its ordinal
among *all* queries is at least two. Episodes whose only known query is the
first contribute no observation to that slice. This slice prevents gains only
on an easy first question from hiding later decision failures.

At update 16, run blank-English and reset-between-turn controls while preserving
labels. Verify exact CPU continuation after serializing turn-three activity,
including the full model's checkpoint identity. CUDA numerical reproducibility
is a separate question; CPU restart does not imply bitwise CUDA training resume.

## Predeclared exploratory benefit screen

Compare weight 0.3 with weight 0.0 at the fixed endpoint. All requirements below
must pass; missing denominators cannot count as a measured success.

- Main development family-macro paired accuracy improves by at least **10
  percentage points**, with paired gains of at least **5 points in at least
  two of three families**. No family's overall query accuracy falls by more
  than **5 points**.
- Main development last-known-later-query accuracy improves by at least **5
  points as an equal-family mean** and by at least **5 points in at least two
  families**. No family's score on that slice falls by more than **5 points**.
- Graph adaptation area is no more than **2 points lower**. At update 16,
  last-known-later-query accuracy and paired accuracy are each no more than
  **2 points lower in either graph level**. A first-query gain cannot compensate
  for worse final path decisions.
- Retention macro query and paired accuracy are each no more than **2 points
  lower**, **both before and after adaptation**, versus the corresponding
  zero-weight arm at the same phase. No retained family's overall query score
  falls by more than **5 points** relative to that control at either phase.

Report uncertainty, reply, advanced-operation, and control results even when
the screen fails; do not replace these requirements with a favorable subset.
Passing selects confirmation only. Repeat a promising matched comparison at
seeds **2303 and 2309**, retaining the same declared banks and budgets and
disclosing that their scenario set is shared. Broader claims require additional
held-out-family rotations and independent scenario designs, not just more
initializations on this one split.

No model is promoted from this exploratory screen. Keep the established
per-family competence, useful-uncertainty, retention, and context-control gates
visible. If the new head learns while the original motor policy does not improve,
the intervention has improved an unused readout rather than the learner's
official behavior. If only the first query improves, the mechanism has not met
the predeclared broad-benefit criterion.

## Compute, ethics, and scope

Use the approved local RTX 5080 and CPU. Record repeated exposures, unique bank
transcripts, parameter counts, training time, total worker time, adaptation time,
and any concurrency. Equal updates/exposure is not automatically equal compute;
the auxiliary gradient path has a cost. Avoid dedicated-GPU efficiency claims
from contended elapsed worker times. No cloud training, external acting, or
autonomous rewriting of source, curriculum truth, audit membership, or gates is
part of this comparison.

The examples concern inert names, amounts, switches, lamps, and graphs. Candidate
selection prioritizes accurate decisions, retained abilities, useful uncertainty,
and learning efficiency. Training does not introduce simulated distress, claims
of sentience, or harmful real-world objectives. The prior validated checkpoints
and all frozen evidence remain available for rollback and comparison.
