# Shared contextual-sensitivity diagnostic

Status: prospective diagnostic design, not a learning result. No diagnostic
model evaluation or training has been launched. The separate parallel-preparation
CUDA identity proof is unrelated to this diagnostic's capability readout.

## Question and limits

The [completed objective comparison](FOUNDATION_OBJECTIVE_RESULTS.md) found weak
simple-fact fit and a late accuracy decline in every run. Sealed metadata shows
no large aggregate query-label shift between the mixed-tail halves. This does
not establish insufficient capacity, forgetting, a retrieval defect or an
ordering effect.

Test whether completed models' predictions change with controlled placement of
irrelevant statements and consistent role renaming. An observed effect is
conditional inference sensitivity in this new grammar. It does not identify the
cause of training failure, establish a better learning method, or authorize
checkpoint promotion. Preserve every condition and model result.

## Explicit diagnostic grammar

Use all three existing semantic families and seven groups: depth-0 direct;
depth-1 copy; depth-1 advance; composed depths 2, 3, 4 and 5. Each episode has
exactly twelve turns:

- Turns 1–10 contain the complete causal chain and irrelevant set statements.
- Turn 11 is the known question, with opposite answers in the counterfactual pair.
- Turn 12 asks about an unassigned role and must have the unknown answer.

For depth d, the causal chain contains d+1 statements. The remaining 9−d
statements concern disjoint roles and cannot affect either question. For gap g
in {0, 2, 4}, put 9−d−g distractions before the intact chain and g after it.
This leaves all causal evidence intact even at depth 5. For example:

| Gap | Depth-5 layout before the two fixed questions |
|---:|---|
| 0 | D1 D2 D3 D4 S T1 T2 T3 T4 T5 |
| 2 | D1 D2 S T1 T2 T3 T4 T5 D3 D4 |
| 4 | S T1 T2 T3 T4 T5 D1 D2 D3 D4 |

Within a placement comparison, preserve every statement's literal text, values,
names, operations, total bytes and answers. Change only placement. Between
counterfactual members, change only S, the initial causal fact. The literal known
question stays the same and its truth must flip. Renaming is a separate matched
condition; role mappings must be consistent, injective and preserve role-name
byte lengths to avoid adding a name-length confound.

This is a new two-query grammar. It omits the canonical intermediate known query
explicitly. Canonical depth-5 episodes leave only three distractions at twelve
turns and cannot implement the proposed four-gap condition unchanged. Do not
reuse their recipe IDs or claim unchanged query density. Pair scoring anchors
the known question at turn 11; the last turn is deliberately unknown.

## Validation and provenance

The additive generator and validator must check the complete trace with both
the abstract state oracle and independent English parser/oracle. Check exact
turn count, fixed question positions, absence of assigned unknown roles,
disjoint distractions, valid causal-chain order, opposite known answers, the
single changed initial fact, matching placement multisets and renaming maps.
All statement targets are acknowledgments. Provenance identifies the new
grammar, generator source, fixed group, pair seed, placement and naming condition.
No hidden state, canonical target or oracle answer enters the learner input.

Initial CPU validation is bounded to one pair seed across all conditions:
7 groups × 3 families × 3 gaps × 2 naming conditions × 2 members = 252 episodes,
plus at most sixteen deliberately malformed validation variants. Record actual
generation, validation, wall/CPU work and failures. No model, Torch or CUDA work
belongs to this fixture validation.

## Planned inference scope

The eventual bank uses eight pairs per group/family before the placement/naming
cross-product: 1,008 pairs, or 2,016 episodes per model. Fix exact seeds, bank
hashes, model-weight hashes, evaluator source and runtime before model inference.
No scores may be inspected while choosing those inputs.

Evaluate all six completed objective endpoints and one common initialization
for each of their three seeds: nine model states, 18,144 episode evaluations
and 36,288 scored query turns. Restore authenticated weights only for inference;
there are no gradients, optimizer updates, tutor answers or model selection.
The evaluator must use native actions and freely generated replies, report
their separate paired correctness, known and unknown accuracy, action/reply
agreement, and paired placement/renaming differences. Retain all raw counts and
family/group/seed breakdowns. Do not replace a fixed endpoint with a better
historical checkpoint.

The bank seed is fixed at 851910001, distinct from the generator test fixture
851900001. Preparation will generate the 2,016 episodes once and record the
validator work separately. Subsequent bank admission may validate those saved
rows again, but will not generate replacements. No exact-transcript exclusion
against the historical inventory is claimed. These are development diagnostics
designed after viewing earlier results.

For each seed 8472, 8473 and 8474, evaluate the authenticated step-zero baseline
checkpoint, baseline step 3,072 and balanced-reply step 3,072, in that order.
Both recipes' verified step-zero parameter digests agree. Each state receives
a separate flat width-192 model with four layers, four heads, feedforward 768,
1,024 positions, twelve turns, 128 input bytes and 32 output bytes. There are
nine model constructions and nine weights-only archive loads. Archives contain
optimizer tensors, which are deserialized as part of the existing format but
are never restored into an optimizer or used for learning.

Use the parent's exact strict FP32 local runtime, one CPU intra/inter-op thread,
and CUDA device zero. Batch size is 32, giving exactly 567 native sequence
forwards across the nine states. All twelve turns are observed causally, with
BOS-only reply prefixes. Free reply generation selects only the two query
contexts (indices 10 and 11); statement reply generation is omitted. The reply
decoder processes rows independently, so this selection changes neither query
prefixes nor their contexts. Count actual free-decoder recurrent calls and row
steps; the bound is 33 recurrent calls per batch, at most 18,711 calls and
1,197,504 row steps. The initial BOS decoder operation inside each native
sequence forward is separate, covering twelve turns per episode.

Inference has a fixed 1,800-second allowance starting before authentication,
including setup, loading, bank admission, scoring and final integrity checks.
Check expiry before new model/batch work. Active operations and final evidence
publication are nonpreemptive; record any overrun. A failed or interrupted
invocation is preserved and has no automatic retry or implicit continuation.
Freeze source, bank, checkpoint archive/parameter and runtime identities in a
launch manifest before any score is inspected. This is an inference diagnostic,
not a throughput benchmark or a learner adoption screen.

The generator's four CPU checks have passed with 252 generated episodes and
sixteen malformed cases. The evaluator adds a separate, declared validation
scope using mock models and synthetic scoring records, with no production
lesson generation, real learner forward, optimization or CUDA. Its exact work
and outcome are recorded separately before preparing the production bank.

## Decision after the diagnostic

Do not infer a training mechanism from an inference-only effect. Any subsequent
learning intervention needs a separate prospective protocol and an unchanged
training control across every family. A matched complete-bundle multiset under
different fixed orders can test an order effect; equal additional exposure can
measure continued learning. Align scheduled endpoints to complete depth/length
cycles and retain all declared checkpoints. Exact parent-state branching and
changed schedules require a new experiment boundary, not an implicit resume of
the completed formal study. The wider goal remains sustained acquisition,
transfer, retention and decreasing dependence on the external tutor.
