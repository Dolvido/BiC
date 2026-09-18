# Shared learning under causal lesson-layout diversity

Prospective protocol. No training for this comparison has started. The
curriculum adapter, research trainer and evaluator are being implemented in
separate modules; they do not alter the completed studies or the existing cycle
owner. The broader external-tutor learning loop remains a separate integration
deliverable.

## Question

Does varying causally valid lesson layouts improve shared comprehension beyond
the fixed final-known-query convention, while preserving known reasoning and
earlier skills across the curriculum?

The [completed context diagnostic](FOUNDATION_CONTEXT_RESULTS.md) found weak
near-context comprehension and poor unknown-last generalization. It did not
establish a unique training failure or a common distance effect. Existing
foundation teaching already varies interior questions and distractors, but
always appends its designated known opposite-answer anchor last. This study
changes that presentation restriction across all families, rather than adding
an arithmetic-specific objective or training only an observed weak cell.

## Matched teaching conditions

Both arms consume the same ordered canonical parent bundles, complete
counterfactual pairs, literal sentences, values, names, targets, replies,
statement order and query read-version ancestries. The new layout schema binds
the canonical parent recipes and row hashes, layout seed, shared permutation,
source identities and explicit original/current query indices. The designated
known anchor retains its identity and opposite answers even when nonterminal.
Old validators and final-turn scorers remain unchanged.

- **Original:** identity permutation, preserving the current teaching layout.
- **Varied:** choose deterministic causal-valid query placements. Exact original
  write versions, including copy ancestry and overwrite effects, must remain
  unchanged in both counterfactual members. Coincident answers after a cycle or
  parity reversal do not authorize a move.

Varied layouts enumerate feasible terminal kinds: known anchor last; a
structurally still-unassigned question last; and a statement last only when
every query has a legal slot before the final statement. Select by layout seed
modulo the feasible kind count, then choose remaining legal slots and within-slot
query order deterministically. This choice is independent of target truth,
values, names and family. The same pair permutation is used in both members.
Record actual support and exposure; do not impose impossible uniform quotas.
At depth five with eight turns, six required statements plus two questions
preclude a terminal statement. Some varied draws can equal the original layout.

Both the typed-effect oracle and independent English oracle must reproduce all
targets after movement. Every query keeps its ancestry partition, not merely
the main anchor; supervised audit compositions remain excluded from training.
All transformed turns, including private observation placeholders, are copies.
Model inputs contain only visible text and causal boundaries.

The first preparation attempt found an exact transcript collision introduced
by a varied layout, before any neural work. The
[recorded preparation amendment](FOUNDATION_LAYOUT_PREPARATION.md) admits both
views together and, on collision, deterministically rebinds only the names in
the shared canonical pair. All problem facts, targets, causal dependencies and
layout seeds stay fixed. Both teaching arms use the same repaired parent stream;
the previous failed work and every rejected naming attempt remain counted.

## First bounded learning screen

Use one fresh paired initialization, a flat width-192 sequence student with
four layers, four attention heads, feedforward width 768, 1,024 positions,
twelve-turn capacity, 128 input bytes and 32 output bytes. Keep the existing
baseline objective, AdamW recipe, clipping and learning rate 0.003 unchanged.
There is no per-arm learning-rate selection and no added auxiliary loss.

The loss function remains unchanged, but this is not a pure isolation of
absolute position. Its reply and observation terms average token losses within
turn positions. Reordering questions changes those per-position mixtures and
can therefore change implicit example weighting. Record target counts, reply
target tokens and input bytes by position in each arm. Interpret any result as
the effect of the complete teaching-layout intervention.

The common canonical plan has six 96-update depth stages, earlier-depth
rehearsal every fourth update, and a 288-update mixed finish. Lengths 8/10/12
rotate within depth. Every update consumes 32 episodes from each of color,
count and switch before one optimizer step. Each arm completes 864 updates:
1,728 total updates and 165,888 learner episode exposures. The mixed finish
contains complete depth/length cycles. The order of canonical bundle IDs is
identical between arms; only the within-episode query layouts differ.

Fixed intended seeds are model 852020799, canonical plan 852020001, plan
ordering 852020002, layout namespace 852020003, development 852021001 and
audit 852022001. Freeze their exact use, source closure, data hashes, parameter
initialization recipe, runtime, checkpoint cadence and resource allowance in
the executable launch manifest before constructing a model or inspecting a
score. Do not start training with an incomplete launch contract.

Save full learner/optimizer snapshots at updates 0, 144, 288, 432, 576, 720 and
864 in each arm. Run original first and varied second; report this fixed order
as a possible timing confound. Both are fresh constructions from seed 852020799;
their initial weight digests must agree. Record the first construction's actual
digest; no extra initialization probe is required. Score that shared state once
on normal fitting and development banks. At updates 144 through 720, score
normal development banks. At 864, score normal fitting, development and audit
banks, plus blank-text and reset-history development controls. No intermediate
checkpoint is substituted for the fixed endpoint. This comparison does not
claim tested optimizer resumption.

The training process has a fixed 5,400-second allowance including preparation
of evaluation tensors, training, repeated admission, scoring, snapshots and
publication. Data generation has separate declared phase bounds and its entire
cost is reported in addition. A deadline is checked between operations; a single
operation may finish after it. An incomplete process is reported as incomplete,
with no automatic continuation or replacement run. All teacher calls are zero.

Use the existing strict FP32 local RTX 5080 runtime, one CPU intra/inter-op
thread and a single optimizer owner. This study uses the synchronous path.
Record preparation/admission, repeated validation, packing, forward/backward,
checkpoint, evaluation and total elapsed costs. Equal utterance totals alone
do not establish equal execution cost. Source or data changes require a new
experiment boundary, not an implicit continuation.

## Readout and decision

Prepare paired canonical and varied-layout evaluation views before training,
with complete opposite-answer pairs, new realizations of taught procedures and
reserved composed procedures across all families and lengths. Preserve their
matching world/target relations and exact transcript exclusions. Bind all
protected data and diagnostic/history exclusions in the preparation receipt.
Teacher answers, hidden state and parsed recipes stay outside policy inputs.

Score the explicitly identified anchor, all other known questions, unknown
questions, freely generated replies, action/reply agreement and unsupported
asking. Preserve all family/depth/layout conditions, literal outputs and raw
counts. Include original-layout retention and blank/reset-history controls.
Do not substitute a better intermediate checkpoint for the fixed endpoint.

The intended benefit is better known counterfactual actions **and** replies
across the families, with accurate unknown responses and retention. Improvement
confined to final unknown questions is adaptation to a presentation convention,
not the requested broader acquisition. The launch must specify exact count
denominators and the descriptive screen before scores are viewed. A promising
single paired initialization requires new paired seeds before any general
learning claim. It is not evidence of broad English competence or a learned
scheduling policy.

Each evaluation cell contains eight opposite-answer pairs. Each layout view
contains 504 fresh-procedure pairs (168 per family), 216 held-composition
development pairs (72 per family), or 288 held-composition audit pairs (96 per
family). Fitting has 504 pairs per layout. These are paired layout views of the
same parent problems, not independent extra problems. Query denominators depend
on the generated procedures and are frozen from the admitted banks before
training. The fitting bank intentionally reuses exact training examples.

The single-seed descriptive screen passes only if every condition holds:

1. For both audit panels (fresh and held composition) and both evaluation
   layouts, varied training improves pooled anchor-pair actions and exact
   replies by at least five percentage points over original training.
2. In each audit panel/layout/family, neither anchor-pair measure regresses;
   each family improves both measures somewhere across those conditions.
3. For each audit panel/layout, unknown-query actions, unknown exact replies
   and action/reply agreement do not regress. Known questions other than the
   anchor are reported separately and must not regress where present.
4. On development, varied training's endpoint normal anchor-pair actions and
   replies exceed both blank and reset controls for each family and layout,
   pooled across fresh and held problems.
5. For each development family/layout, loss from the best earlier observed
   normal score to the endpoint is at most five percentage points greater than
   the corresponding loss under original training, separately for paired
   actions and replies. Report the complete trajectories and absolute losses.

Thresholds are prospective engineering decisions, not a statistical confidence
claim. No screen outcome automatically promotes a checkpoint or establishes
general intellect. Report per-cell counts even when pooled screens fail. A
failed screen informs the next learning decision without permitting the tutor
to inspect audit answers or silently target those examples.

## Validation before training

The layout adapter's first CPU scope is 54 distinct canonical pairs across
three families, six depths and three lengths: 108 unique base episodes. Request
four layouts per pair (original and varied seeds 0/1/2), producing 216 layout
pairs/432 rows. Compact-recipe validation reconstructs additional copies of
those parents and layouts; count that work separately. Include at most sixteen
malformed variants and two tiny handwritten typed relation fixtures checking
parity coincidences and copied values surviving source overwrite. Block Torch
imports. Preserve all failures and actual counters.

Trainer and evaluator checks require separately declared bounded scopes before
execution. Mocked scoring tests are not neural validation. Real model calls,
optimizer updates, discarded work and uncertain partial operations must be
counted explicitly. Failed execution never silently resumes or repeats.
