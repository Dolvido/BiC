# Shared curriculum mechanism audit

This source-only review found **no code-established semantic, admission,
truncation or policy-input blocker** in the current definition/basis/complementary
path. It does identify limits on what the measured behavior can establish.
The review used source and completed study documents only, while the fixed rate
comparison was running. No live rate scores, lesson contents, checkpoints,
generators, models or tutor services were accessed or executed. No frozen file
was changed.

## What the learner actually receives

The providers implement a finite English instruction language: define/revise a
nonce word, set a value, apply the word to two entities, and ask a yes/no/unknown
question. C copies source to destination; S advances source; D advances
destination. The native model receives UTF-8 observation bytes and turn
boundaries. Program trees, recipe identities, query kinds, tested rule words,
rule versions and prefix labels are verification or supervision fields.

The relevant boundaries agree:

- Separate typed and visible-English interpreters must agree on answers,
  prefix states and mutation provenance before publication. Query literals do
  not assign facts; revision changes a rule without retroactively changing
  entity values. Twelve-alias labels remain causal after each whole utterance.
  Anchors must have opposite known answers across the pair. See
  `experiments/definition_basis_curriculum.py:92`, `:140`, `:205`, `:301` and
  `experiments/complementary_composition_curriculum.py:86`, `:134`, `:298`.
- Canonical admission precedes stored prefix-label production; runtime loading
  authenticates the unchanged image, row/evidence identity and aligned targets.
  The prepared owner then checks exact admitted pairs and consumer coordinates;
  it does not pretend to regenerate new semantics on every use. See
  `experiments/complementary_composition_data.py:128`, `:144`, `:168`, `:176`
  and `experiments/foundation_layout_prepared.py:90`, `:143`.
- Oversize input raises rather than truncates: 128 UTF-8 bytes per utterance,
  twelve turns and the configured 1,024-position context. BOS/EOS positions are
  included in the provider's context check. Right padding and causal attention
  cannot expose later observation tokens to an earlier turn's output. See
  `brain_in_computer/language.py:59`, `experiments/composition_data.py:28`,
  `experiments/sequence_student.py:87`, `:145`.
- Replies are teacher-forced only in the separate training decoder. Native
  evaluation packs observations, supplies BOS, and freely generates replies;
  reply targets do not enter the observation encoder or action head. See
  `experiments/composition_data.py:129`, `experiments/sequence_student.py:163`,
  `experiments/definition_basis_evaluation.py:292`, `:395`, `:405`.

## Structural reuse and observable limits

Training and development nonce vocabularies are disjoint; complementary lessons
deliberately reuse the basis split vocabularies. They do not introduce globally
new words. Basis training covers C/S, S/D and C/D contrasts; complementary
training adds CD/DC, SD/DD, SS/DS, DCD/CDD and SSD/SDD. SC, CS, SCS and CSS remain
absent as trained definition strings. See `definition_basis_curriculum.py:27`,
`:30`, `:64` and `complementary_composition_curriculum.py:27`, `:30`, `:134`
under `experiments/`.

These are not universal holdouts of mathematical functions. From the actual
semantics, with initial state `(s,d)` and advancement `a`:

| Program | Final source, destination |
| --- | --- |
| SC | `a(s), a(s)` |
| CS | `a(s), s` |
| SCS | `a(a(s)), a(s)` |
| CSS | `a(a(s)), s` |

DC equals C, SD equals DS, and DCD equals CD. For binary switches, two advances
cancel: held-out CSS equals practiced C/DC, and held-out SCS equals practiced
CD/DCD. This was already declared in
[COMPLEMENTARY_COMPOSITION_DESIGN.md](COMPLEMENTARY_COMPOSITION_DESIGN.md).
Opposite-anchor checks avoid treating equivalent functions as a discriminating
pair, but successful switch sequence behavior alone would not establish a new
function. Order-sensitive reuse and equivalence-preserving reuse are distinct
claims.

Two-word binding randomizes definition order and tests both applications with
swapped entity roles. Its counterfactual pair changes the contrasted definition,
not which word is requested by an otherwise identical Apply instruction
(`definition_basis_curriculum.py:224`, `:281`; complementary equivalent `:224`,
`:277`). Thus existing word/position slices are useful behavioral evidence,
but do not isolate causal selection by the application word. Auxiliary labels
describe entity values, not the stored word-to-program map. Equal value labels
before application do not imply the hidden state cannot retain different rules
(`shared_state_student.py:56`, `shared_state_targets.py:98`). This is an
observability limit, not proof of destructive gradients.

Every definition pair requires all six queries in both episodes to have both
correct actions and exact free replies. Easier prior-value and unknown queries
can inflate aggregate accuracy; the all-query criterion and role slices already
expose this distinction (`definition_basis_evaluation.py:187`). Completed basis
fit, chapter-order, complementary and objective results already establish weak
practiced binding, some revision acquisition, and unreliable transfer. They do
not establish a capacity ceiling, a latest-definition shortcut, or harmful
auxiliary supervision. In particular, auxiliary removal failed its prospective
benefit screen. Repeating those proposals would duplicate existing work; see
[DEFINITION_BASIS_RESULTS.md](DEFINITION_BASIS_RESULTS.md),
[DEFINITION_TUTOR_PAIR_RESULTS.md](DEFINITION_TUTOR_PAIR_RESULTS.md), and
[SHARED_OBJECTIVE_RESULTS.md](SHARED_OBJECTIVE_RESULTS.md).

## Actual tutor authority and two discriminating questions

The definition author can permute six immutable chapters, with each binding
chapter before its matching revision. It cannot change lessons, targets,
within-chapter order, exposures, operations or explanations
(`experiments/definition_tutor_chapters.py:57`, `:80`, `:89`;
`experiments/definition_tutor_author.py:152`). The separate recurring campaign
author chooses allowed motif pools and alias/value realizations; it likewise
cannot write arbitrary instructional English or ground-truth answers
(`experiments/verified_tutor_author_v2.py:113`). These are real external
curriculum decisions, but not unrestricted English tutoring or learned native
self-direction. The fixed grammar and four reply forms do not measure arbitrary
English understanding.

Two general mechanism questions remain distinguishable from prior aggregate
analyses, for a separately declared future diagnostic:

1. With both definitions and initial facts fixed, does changing only the word
   in an Apply instruction produce the corresponding opposite behavior across
   all three families? This isolates rule selection from the existing
   definition-change counterfactual and should retain both-argument questions.
2. Once practiced acquisition is adequate, does the native learner preserve
   answers under a functionally equivalent program rewrite while changing them
   under a non-equivalent order change, with nonce vocabulary and bindings
   matched? Report equivalence and novel-function cases separately by family,
   followed by tutor-free retention. This avoids interpreting a switch identity
   as wholly new semantic execution.

Neither question changes the running study, supplies a new acceptance gate, or
justifies a curriculum patch before its fixed results are known.
