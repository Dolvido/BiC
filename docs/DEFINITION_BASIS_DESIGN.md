# Ground shared operations before testing new combinations

The first definition experiment learned the two taught meanings and their
revisions, but solved no complete held-out composition pair. Those compositions
changed the source; every trained definition left it unchanged. The experiment
therefore combined a new argument-role effect with a new instruction order.
It also defined only one verb per episode, permitting a strategy that ignored
the word and executed the latest definition. The next curriculum addresses
these general experimental limitations across all three subjects.

Teach the same three operations in color, count and switch worlds:

| Operation | Meaning |
|---|---|
| C | Copy the source value to the destination. |
| S | Advance the source once. |
| D | Advance the destination once. |

Balance C/S, S/D and C/D definition contrasts. For C/S initialize different
source and destination values so copying is observable. For C/D initialize
equal values: when switch values are opposite, copying and advancing the
destination produce the same result. Independently executed semantics must
establish each opposite-answer question; labels cannot force a difference.

Binding episodes define two nonce verbs in randomized order. After both
definitions, they apply the contrasted word, then apply the other word with
the entity roles reversed. Counterfactual members change only the contrasted
definition. Questions before the second application identify the first rule's
effect, and later questions check the other rule and both entity roles. This
tests rule-word discrimination as well as instruction execution. Training and
evaluation use separate, previously unused verb vocabularies.

Revision episodes check that redefining a word does not retroactively change
facts, then apply the revised operation with reversed entity roles. Query
metadata identifies the tested application and word, including an argument
that the operation leaves unchanged. Mutation provenance is distinct from
which rule is being tested. Neither metadata field enters neural inputs.

Withhold S;C and C;S from new training and compare their different destination
effects after the shared primitives are grounded. Their hypothesis has already
been inspected, so new examples are development transfer rather than a pristine
new hypothesis. A separately reserved sequence-length panel may compare
S;C;S with C;S;S; their destination values differ after one advancement even in
the switch family. Neither reserved sequence enters training.

Retain practice on the earlier copy/copy-then-advance-destination definitions
and on the original broad curriculum. All subjects keep equal weighting. The
next experiment should preserve complete all-question paired accuracy as its
explicit primary definition metric and retain the existing old-skill checks.
It must distinguish the definition-trained update-8,032 candidate from the
separate campaign's update-8,032 learner; their identities and histories differ.

The basis provider is additive. It preserves the public learner interface,
model, full optimizer, rate and objective while supplying independently verified
English and prefix supervision. Current constraints remain twelve turns,
twelve entities, bounded values and four reply forms. Data and GPU runs will be
declared separately after the provider's semantic checks; this document is a
design, not a report of newly trained ability.
