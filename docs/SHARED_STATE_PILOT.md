# Shared causal-state teaching pilot

Prospective local comparison fixed before learning or model scores. The previous
repeated-reading comparison failed its screen. This experiment tests whether
dense, shared teaching of entity state improves independent reasoning across
all three current foundation subjects. It tests the teaching intervention as
a whole, not factorization separately from adding supervision.

## Learners and teaching

Both arms instantiate the existing flat SequenceStudent plus the same auxiliary
decoder and start with identical weights, initialization seed 852304001. The
policy is width 192, four layers, four heads, feedforward 768, position capacity
1024, maximum 12 turns, input 128 bytes and output 32 bytes. AdamW uses learning
rate 0.003 and gradient clipping at 1, with the existing strict FP32 execution profile
on the local RTX 5080. Original action, reply and observation losses are unchanged.

The auxiliary decoder concatenates each normalized EOS representation with a
learned 32-dimensional entity-query embedding and uses one shared 128-unit GELU
hidden layer. Its 107 output classes are unknown, four colors, counts 0–99, and
switch off/on. All 12 aliases use the same decoder; no separate name or family
classifier, per-family output mask, or privileged policy input is introduced.
The native policy forward remains the original forward. The auxiliary method
is never called by policy evaluation.

Generate targets from the observed English prefix after each turn. A question
does not assign its proposed value; copying an unknown source clears the
destination; copies preserve values through subsequent source changes; unknown
updates remain unknown. Dense targets cover the same fixed 12 aliases at every
prefix, so future occurrence never selects an entity query. Verify query/ACK
labels against both existing interpreters and check state semantics explicitly
in focused tests. The parsed state and targets enter the training loss only.

For each example and turn, compute 107-class cross entropy, average known entries separately
from unknown entries and give the two present groups equal weight. Average turn
losses equally across examples and turns, then preserve the original equal three-family weighting. The control
uses auxiliary weight 0 and skips its auxiliary graph; the candidate uses 0.3.
No weight sweep or development-based adjustment occurs in this comparison.

## Lessons and independent evaluation

Reuse all 648 immutable original-layout bundles from the completed repeated-read
data cache in canonical curriculum order. Both arms receive 62,208 episodes across
all three families, depths 0–5 and lengths 8/10/12. Derive state targets once per
bundle and share them across arms. Alternate which arm updates first. Run 648
updates per arm and evaluate at 0/216/432/648; save complete model/AdamW checkpoints
at each nonzero endpoint. The fixed execution allowance is 3,600 seconds; stop on
execution failure or allowance exhaustion and retain partial evidence. Do not
stop or extend according to intermediate scores.

Prepare a new 720-episode development bank with seed 852302001 and four pairs per
cell. Exclude the authenticated 1,701,720-transcript union including the previous
comparison's fresh development examples. Reuse the 108-episode balanced actual
training-fit sample and 720-episode retention bank, explicitly labeling them as
reused. Fresh text is not independent natural-language coverage: this remains
a finite grammar with common vocabulary, patterns and semantic equivalents.

Score native actions and unconstrained generated English replies with no tutor
or auxiliary state decoder. Preserve raw records and report joint counterfactual
pairs, separate action/reply pairs, known/unknown performance, unsupported
abstention, agreement and retention, overall and by family. Better auxiliary
classification or agreement alone is not evidence of comprehension.

## Decision and scope

At 648 updates, require at least 5 percentage points of overall fresh joint-pair
improvement, strictly more joint pairs in **each** of color/count/switch, and
strictly more total joint training-fit pairs. No head may lose more than 5 points
of known or unknown accuracy in any development family; overall joint retention
may not lose more than 5 points. Use exact counts to apply these thresholds.
Passing this exploratory single-seed screen justifies replication and a matched
compute comparison, not checkpoint promotion. Report all preparation, execution,
training and evaluation costs and GPU memory. A negative result does not justify
extending this configuration unchanged.

The current curriculum avoids interfering overwrites. Testing target code on
overwrites does not establish a learner's ability to handle them. The initial
comparison keeps the curriculum fixed to isolate teaching; broader revision and
interaction require explicit later curriculum coverage.

The LLM remains external and makes no calls in this comparison. The established
tutor loop has demonstrated order selection and withdrawal, but richer tutoring,
beneficial knowledge transfer, broad English comprehension and learned topic
selection remain open project requirements. The broad home-computer goal remains
active regardless of this bounded experiment's outcome.
