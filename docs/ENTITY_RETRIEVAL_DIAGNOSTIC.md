# Entity-retrieval training-state diagnostic

Declared while the paired entity-retrieval learning run was active, before the
root agent viewed any of its model scores. Initial scores may already have been
computed internally. This separate declaration does not change the frozen
primary screen, choose checkpoints, or authorize promotion.

After the main run completes and its primary predictions have been recounted,
inspect both predetermined final 648-update checkpoints: baseline and retrieval.
Use only the 108 authenticated actual training-fit episodes already present in
the run's bank archive. Decoding that archive also decodes its other banks;
discard those without scoring or constructing state targets. There are no
development, retention or audit state targets in this diagnostic.

Group by family and actual turn count: nine groups of twelve episodes. Derive
the existing causal-prefix targets once per group, including all twelve aliases
at every prefix, and reuse those labels for both models. Labels are supervision
for scoring only and never native model inputs. Use the original strict local
FP32 runtime and load exact full weights from the source-pinned checkpoints
with weights-only deserialization. Build two models and verify unchanged full
weights, evaluation modes and absent gradients after scoring.

Each group receives one native forward with BOS-only decoder prefixes. The
baseline uses its existing EOS/alias state decoder; the candidate decodes the
cached entity vectors from that same native forward. Do not repeat candidate
retrieval to obtain auxiliary logits. This gives exactly 18 native forwards,
18 auxiliary calls, 216 episode passes and 25,920 entity-state predictions.
Make zero optimizer, backward, free-reply or tutor calls.

Preserve targets, ordinary 107-class argmax, conditional argmax over all 106
nonzero classes, true-value rank and P(known)=1-P(class 0). Do not apply a family
mask. Rank is one plus the number of strictly greater nonzero logits; argmax
ties select the lowest class index. Report exact known/unknown class accuracy,
conditional known-value accuracy, knownness confusion using P(known)>0.5,
probability means, Brier score and mean known-value rank, overall and by family.
Reuse the unchanged, previously tested pure state_metrics implementation.
Sort group keys before recounting so JSON key ordering does not change floating
accumulation. Also report the existing balanced state loss, weighted by actual
example-turn counts when pooling groups.

Authenticate the caller-pinned completed main summary and launch, current main
source closure, final checkpoint images, data manifest and bank, and this
diagnostic's own source/protocol hashes. Bind the two extra imports needed by
state_metrics to their already-verified prior fit-launch source identities.
Use an exclusive output directory and a 180-second allowance covering setup,
scoring, verification and publication. Check deadlines between operations;
an in-flight operation may finish after the boundary. Preserve partial work,
wall/CPU cost and peak GPU memory on failure; do not retry or extend silently.

This describes learned representation on training examples. Better auxiliary
accuracy does not establish native use, generalization or comprehension. Poor
readout does not establish that the encoder contains no facts. Conditional
value accuracy must remain separate from knownness and the ordinary full
argmax, avoiding a misleading all-unknown accuracy headline. No diagnostic
threshold overrides the main paired screen or promotes a checkpoint.
