# Fresh realizations of shared procedures: prospective protocol

Freeze this specification, implementation, banks and deterministic stream rules
before optimization. This is a new local comparison informed by the completed
[composition study](COMPOSITION_STUDY_RESULTS.md); its earlier evidence remains
unchanged. No automatic promotion or audit-driven checkpoint selection is allowed.

The earlier study fitted its exact color/switch examples well but performed poorly
with fresh names and values. Count also had a substantial exact-example fitting
deficit. Those observations motivate a shared curriculum intervention; they do
not identify a particular naming, arithmetic or architectural failure.

## Intervention and matching

Compare two newly initialized **joint-schedule** learners:

| Arm | Realization of a sampled procedure |
|---|---|
| `fixed` | Reuse that procedure's admitted initial names and values. |
| `fresh` | Use its initial realization on first occurrence; on later occurrences generate independently verified names and values. |

Both use the existing shared set/copy/advance/query grammar in color, count and
switch, with 8-, 10- and 12-turn episodes. Keep the sampled full procedure,
operation order, roles, actual query ancestries and counterfactual construction
fixed when realizing a procedure again. Values can change increments, equality
queries and known-answer targets. Reject a whole pair unless the independent
abstract and English interpreters agree and changing exactly one earlier set
statement flips the known final answer. Unknown queries remain nonfinal.

Each arm runs **3,600 AdamW updates** at learning rate **0.001**, initialization
seed **2901**, and sampler seed **3901**. Use the unchanged **753,610-parameter**
SequenceStudent with `max_turns=12` and `max_positions=1024`. Every update averages
three 32-episode microbatch objectives, one per domain, using the same objective,
loss normalization and clipping as the preceding composition study.

Each domain has three length banks with 256 complete pairs each. The independent
per-domain sampler selects a length uniformly, then 16 pairs with replacement.
Both arms consume the identical ordered length/procedure draws. Generation uses
a separate deterministic seed stream and cannot advance this sampling stream.
Each arm consumes 115,200 episodes per domain and 345,600 in total; the two-arm
budget is **7,200 updates and 691,200 episode exposures**. This matches updates,
episodes and procedures, **not observation bytes, targets, increments, padding,
generation cost or dedicated GPU time**. Record those realized differences.

A separate disposable CUDA execution probe used two fresh-stream updates
(192 episode exposures), model seed 2900, sampler seed 3900 and a separate tiny
training recipe namespace beginning at 265,000,000. Its weights were discarded;
it did not select a rate, model or checkpoint. Record its cost separately and
do not treat its short execution timing as steady training throughput.

Fresh rendering retries protected or previously observed exact transcripts and
reserved initial realizations without replacing the sampled procedure. Record
occurrence counts, accepted seed/retry identities and rejection totals. If an
admissible new realization cannot be found within the frozen search bound, stop
and report exhaustion; do not relax the boundary after seeing scores. Repeated
Boolean states or semantically equivalent chains are not new abstract skills.

## Banks and boundaries

Use seed namespaces **170,000,000 / 180,000,000 / 190,000,000** for initial training,
development and sealed audit, with the bank manifest recording every accepted
procedure, naming/value seed and row identity. Training contains 4,608 episodes.
Each development/audit group contains 64 complete pairs for every panel, domain
and length: **4,608 episodes per group**.

| Panel | Procedure | Names | Values |
|---|---|---|---|
| `name_only` | Familiar initial training recipe | Changed actual mapping | Exact initial value seed |
| `value_only` | Familiar initial training recipe | Exact initial naming seed | Changed actual anonymous realization |
| `both` | Familiar initial training recipe | Changed mapping | Changed anonymous realization |
| `composed` | Withheld final ancestry partition | New realization | New realization |

The three familiar panels use the same first 64 admitted training recipes per
domain and length. Changes must concern actual realized content, not merely seed
numbers. A value change can also change textual increments and query targets;
the panel is not an isolated test of initial stored values. `name_only` and `both`
share the changed naming coordinate; `value_only` and `both` share the changed
value coordinate. Reject or accept each six-row factor triplet together.
Comparing marginal panels does not by itself establish an additive naming/value
mechanism.

Reject whole pairs if either member's complete observation transcript duplicates
any protected individual transcript. The historical protection set comprises
all **9,216 rows of the previous composition study**, loaded only after its bank
hash matches its frozen protocol. Also exclude exact duplicates across all new
initial and evaluation banks. During fresh training protect these historical
transcripts, all new development/audit transcripts, previously encountered fresh
transcripts and initial realizations reserved for first occurrence. Record exact
protection counts and hashes. This does not claim exhaustive exclusion against
every earlier project grammar or every possible semantic equivalent.

Retain the existing ancestry-partition rule: every supervised known training
query containing copy and advance must be in the training partition. Development
rejects known composed queries from the audit partition. The final composed
audit query has an ancestry absent from this study's supervised training and
development queries. Earlier audit questions may have mixed familiarity.
These finite syntactic partitions and some motifs were examined in earlier
studies; the new study does **not** establish never-before-evaluated algorithms,
globally new motifs, unseen vocabulary or semantic novelty.

Freeze actual bank counts, unique procedures/ancestries/naming maps/realizations,
targets by query position, byte/token sizes and maximum contexts. No truncation
is permitted. Only observed text bytes enter the learner. Canonical state,
targets, family/split identities, ancestry metadata and teacher replies cannot
enter policy inputs.

## Evaluation and fitting diagnostics

Save checkpoints and development scores at updates **0, 300, 900, 1,800 and 3,600**.
Use the declared endpoint regardless of development outcomes. Open no sealed
audit until **both endpoints** are complete and their source, initialization,
optimizer, sampler and entire draw/realization evidence has been authenticated.
Then evaluate those same five checkpoints on the frozen audit panels. Shared
initial weights may reuse one verified initial evaluation if explicitly recorded.

At the final checkpoint also score two descriptive fitting probes:

1. The shared full initial-realization bank, with an independent check of which
   original procedures were actually encountered in each arm.
2. Each arm's most recently encountered realization for every sampled procedure,
   reconstructed from its occurrence and retry record. Report missing procedures
   rather than substitute unseen rows. When all procedures were encountered,
   each probe contains 4,608 episodes with matching structural coverage.

The latter is a bounded occurrence-selected training sample, not the full fresh
stream or an unbiased estimate of generalization. Initial-bank fit is also not
fresh-distribution performance. These probes do not select weights or modify
the prospective transfer scores. Reconstruct the consumed stream once per arm,
checking all saved checkpoint boundaries, before releasing any audit score.

Report per domain, length and panel: query accuracy, known/unknown denominators,
confusion matrices, final opposite-answer pairs, eligible earlier paired queries,
query-position results, free-reply correctness/pairs, action/reply agreement and
probability error. Each audit bank has 64 final pairs, always known and balanced;
final-turn uncertainty is therefore unmeasured. Keep unknown denominators
explicit and do not substitute aggregate query accuracy for causal paired success.
Curve areas are descriptive sparse checkpoint summaries, not precise learning
speed measurements. Report endpoint differences and any regressions as well.

Run final blank-text and reset-history controls against original targets; reset
history presents each original utterance alone. Verify exact CPU save/reload
outputs and free replies. Check weights and checkpoint files remain unchanged.
One shared three-domain schedule measures concurrent maintenance; it is not a
new late-domain forgetting or few-shot transfer experiment.

## Integrity, cost and interpretation

Freeze the source dependency closure, source copies, protocol, banks, initial
weight digest, sampler configuration and final structural-draw expectations.
After both endpoints finish, independently reconstruct all five realized-stream
checkpoint boundaries from the frozen generator and compare their actual
digests, occurrences and counters before audit access; this reconstruction is
not a second optimization run or a pretraining measurement of audit outcomes. Exact
resume includes optimizer moments, occurrence/retry state, collision-protection
state and counters. Authenticate every realized row independently before it can
provide a training target. Cached immutable canonical banks may be reused with
identity checks and mutation isolation; caching cannot waive validation.

Keep memory bounded by recipe/occurrence records and compact transcript digests,
rather than retaining the complete fresh stream as full rows or computation
graphs. Record preparation, generation/validation, scoring and peak memory where
measured. Synchronized step time includes generation, validation, packing and
optimization; its measured generation component is a subset, not an additional
cost to sum again. It is not isolated GPU optimizer time. Count interrupted
discarded work separately when reconstructible; retained checkpoint work is not
necessarily total physical compute. Include CPU generation overhead in efficiency
judgments.

This is one initialization with two arms. A useful improvement requires stronger
known paired answers and free replies across domains and lengths, with reported
tradeoffs in count fitting, unknown handling and concurrent maintenance. No
numerical promotion threshold, adaptive curriculum selector or architectural
diagnosis follows from this single study. Replication and regional integration
remain separate steps. Training concerns inert simulated objects; no external
actions or harmful selection pressures are introduced.
