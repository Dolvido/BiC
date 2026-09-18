# Learned utterance summaries: candidate rationale

The rationale below was prepared before reading any foundation pilot scores.
The completed [pilot](FOUNDATION_PILOT_RESULTS.md) subsequently matched this
branch of the [prospective interpretation guide](FOUNDATION_PILOT_INTERPRETATION.md):
both orders were weak on primitive fitting and fresh primitive answers. The
separate candidate now passes 11 focused CPU tests, and the
[prospective comparison](FOUNDATION_VARIANT_STUDY_PROTOCOL.md) entered formal
preparation after its six-case GPU proof passed. This pattern does not uniquely
identify representation as the cause; the frozen pilot remains unchanged.

The hypothesis is that separating byte comprehension from reasoning across
utterances could shorten useful learning paths. It must be tested across color,
count and switch with the same shared mechanism, rather than repairing a failed
subject or individual wording pattern. General English, grounded knowledge,
retention and teacher independence remain wider, unproven objectives.

## Smallest proposed change

The existing [SequenceStudent](../experiments/sequence_student.py#L145) applies
four causal blocks to the complete observation-byte sequence, then gathers each
EOS representation for actions and replies. Its embeddings, blocks and shared
heads are defined at [lines 58–74](../experiments/sequence_student.py#L58).

Reuse that parameter inventory in a new, separately versioned model:

1. Apply the first **two causal byte blocks within each utterance**. Their weights
   are shared across every utterance, domain and context position. Pack utterances
   into the batch dimension so independent local encoding can run in parallel.
2. Gather one learned width-sized vector at each utterance's EOS. The EOS state
   sees that complete utterance; it is not a parsed fact or supplied state vector.
3. Apply the remaining **two causal blocks over the ordered utterance vectors**.
   A turn's decision can use its current utterance and summaries of earlier ones.
4. Keep the current action head, reply-context projection and English decoder.
   Keep the existing final normalization parameters; apply that same normalization
   to local byte states for the auxiliary readout and to final turn states for
   action/reply production. This proposal adds no extra learned normalization,
   pooling query, turn embedding, gate or domain-specific head.

Preserve the original global byte-position embeddings while repacking, rather
than silently substituting a new positional scheme. The current
[boundary validator](../experiments/sequence_student.py#L117) derives utterance
beginnings from observed BOS/EOS positions. These boundaries already exist in
the [observation packer](../experiments/composition_data.py#L30).

Only observed BOS/byte/EOS IDs, their padding masks, lengths and boundary
positions enter the encoder. Labels, replies, family/depth IDs, recipes, oracle
state, symbolic aliases and external-teacher representations remain outside it.
Reply prefixes stay in the separate decoder. Keep the same raw context, turn,
input-byte and output-byte limits; compression must not grant the candidate
additional history. Blank/reset controls retain their existing meanings.

The new representation must have an explicit checkpoint/config identity. An old
checkpoint's tensors must not be silently reinterpreted as a validated new
architecture. Likewise, a returned local byte state must not be described as the
old full-history byte representation to another consumer, such as a regional
adapter.

## Matching parameters does not match computation

Repartitioning the existing four blocks, while retaining all embeddings and
heads and adding no parameters, is intended to preserve the exact parameter
inventory. At the current width-192 configuration, the reference inventory is
2,221,738 parameters; verify the implemented count and every intended shared
tensor before freezing a comparison. Use identical initial values for
corresponding tensors where possible, fresh learners, identical admitted lesson
streams and order, and equal development-only calibration budgets.

For a sequence of `T` utterances with byte-token lengths `n_t` and total length
`L`, the attention interaction term changes from roughly `4 L²` to
`2 Σ n_t² + 2 T²`. Projection and feedforward work also change: the latter two
blocks operate on `T` vectors instead of `L` tokens. Actual batching adds padding
and packing costs. Merely applying a block-diagonal mask to a dense `L × L`
kernel would not realize the proposed local-encoding savings.

The primary comparison could match parameters, lessons, targets, updates and
development search, while reporting measured forward/backward time, packing
cost, memory, bytes and end-to-end local wall cost separately. It would **not**
be compute-matched. An additional equal-wall-budget comparison would generally
change exposure and needs its own prospective declaration. No speed or learning
efficiency gain is assumed from the interaction-count formula.

## Auxiliary loss and compression limits

The current [objective](../experiments/sequence_training.py#L20) combines action
teaching with reply and observation-byte losses, each weighted 0.1. Preserve its
targets and per-turn weighting. The packer masks
[EOS-to-next-BOS transitions and padding](../experiments/composition_data.py#L134).

Attach the tied next-byte head to causal **local byte states**, retaining the
head's existing parameters. This shortens its receptive field: the present flat
encoder can use earlier utterances, whereas the proposed local byte predictor
cannot. The auxiliary gradient also passes through fewer encoder blocks. Thus
the candidate tests hierarchical representation and credit routing together;
it does not isolate compression alone, despite an unchanged loss formula.
Never broadcast a completed EOS summary backward into predictions for earlier
bytes: that would reveal the bytes being predicted.

The two-local-plus-two-turn arrangement also reduces cross-utterance block depth
from four to two. It therefore changes interaction depth and credit routing as
well as compression. This might make primitive learning easier while limiting
longer dependencies; outcomes at every curriculum depth must remain visible.
Matching the parameter count does not isolate the effect of compression.

One finite vector per utterance is a real information bottleneck. It may help
organize reusable facts and reduce long byte-level interactions, but it may also
discard exact aliases, numerical detail, relationships or ambiguity needed by a
later question. Utterance boundaries provide an existing structural bias, not
evidence that each sentence contains one independent fact. The present short,
finite grammar cannot establish that this compression works for ordinary
English or more densely informative utterances.

Competing explanations for weak primitive scores include insufficient or poorly
allocated exposure, optimization settings, multi-objective gradient balance,
alias/value binding, and disagreement between action and reply heads. A positive
result would support this combined inductive bias under the tested recipe; a
negative result would not prove that learned summaries cannot work.

## Meaningful checks before any launch

- **Causality and separation:** perturb future utterances and later bytes without
  changing earlier permissible logits; verify full-prefix equivalence and that
  decoder prefixes cannot affect actions or observation representations. Earlier
  next-byte predictions must not see the completed current utterance. Existing
  [causality tests](../tests/test_sequence_student.py#L73) provide useful reference
  contracts, with additional checks for the local-to-turn boundary.
- **Packing and learned paths:** verify variable byte lengths, padding, blank
  turns, session isolation and exact EOS ordering through repacking. Keep the
  action, reply and byte-loss tensor/target contracts. Confirm gradients from a
  later decision reach earlier learned summaries and the shared byte encoder;
  no detached cache, parser, target-fed shortcut or domain-specific branch may
  substitute for that path. Check the exact parameter inventory separately.
- **Continuation and cost:** require exact tested CPU optimizer/input/cursor
  continuation and a new workload-specific strict GPU repeat/reload proof before
  formal use. Measure packing and training cost on identical frozen inputs.
  These engineering checks establish neither successful learning nor transfer.

Formal use requires fresh reserved evaluation data, a frozen comparison plan and
its workload-specific GPU proof. The original pre-result note involved no model
execution. Subsequent implementation evidence is recorded separately under
hierarchical validation (archive reference: `../runs/hierarchical-sequence-validation-local/attempt-001/`):
11 CPU checks and six optimizer updates, without a learning-benefit claim.
