# Shared-state learning-rate training-state diagnostic

Declared before the paired broad-data learning-rate run begins. This separate
diagnostic does not change its frozen native-policy screen, choose checkpoints,
extend training or authorize promotion.

After the main run completes and its primary predictions have been recounted,
inspect both predetermined final 648-update checkpoints: fast (AdamW 0.003) and
slow (AdamW 0.0003). Both are the unchanged SharedStateStudent trained with the
original sequence objective plus 0.3 state loss. Authenticate the main schema
`bic-shared-state-rate-pilot-v1`, common seed 852604001, declared arm rates and
weights. Check each outer checkpoint rate and the learner recipe's saved AdamW
parameter groups against the launch contract. The serialized optimizer groups
must equal the recipe groups. No optimizer is constructed or restored to a live
optimizer object during this diagnostic.

Use only the same 108 authenticated actual training-fit episodes in the main
bank archive. Deserializing that archive also deserializes its other banks;
discard them without scoring or constructing state targets. There are no
development, retention or audit state targets in this diagnostic. The primary
comparison's development bank is reused and previously observed, not pristine
validation; this training-only diagnostic does not change that scope.

Group by family and actual turn count: nine groups of twelve episodes across
color/count/switch and 8/10/12 turns. Derive the existing causal-prefix state
targets once per group, for all twelve aliases at every prefix, and reuse those
labels for both models. Labels are scoring supervision only and never native
model inputs. Use the exact main strict local FP32 runtime and weights-only
deserialization of authenticated full checkpoints. Construct two models and
check unchanged complete weights, evaluation modes and absent gradients after
scoring each one.

Each group receives one native forward with BOS-only decoder prefixes, followed
by the existing shared EOS/alias state decoder. The complete work is exactly
two constructions, two checkpoint loads, one bank load, nine target packs over
108 rows/1,080 turns, 18 native forwards, 18 auxiliary calls, 216 episode passes
and 25,920 entity-state predictions. Perform zero optimizer calls, backward
passes, free-reply rollouts or tutor calls. Generate no new curriculum rows.

Preserve targets, ordinary 107-class argmax, conditional argmax over all 106
nonzero classes, true-value rank and P(known)=1-P(class 0). Do not apply a family
mask. Rank is one plus the number of strictly greater nonzero logits; argmax
ties select the lowest class index. Report exact known/unknown class accuracy,
conditional known-value accuracy, knownness confusion using P(known)>0.5,
probability means, Brier score and mean known-value rank, overall and by family.
Use the unchanged, previously tested pure state_metrics implementation from the
authenticated earlier fit study. Sort group keys before recounting so JSON key
ordering does not alter floating accumulation. Report the existing balanced
state loss, weighted by actual example-turn counts when pooling groups.

Authenticate caller-supplied SHA-256 pins for the completed main summary and
launch, this diagnostic source and this protocol. Also authenticate the main
source closure, final checkpoint images, data manifest and bank. Preserve the
metric-origin launch and the exact shared_state_fit/shared_state_pilot source
pins inherited from the frozen entity-retrieval diagnostic; do not substitute
a newly edited metric implementation.

Use an exclusive output directory and a fixed 180-second allowance including
setup, scoring, verification and publication. Check deadlines between
operations; an in-flight operation may finish after the boundary. Preserve
actual and attempted work, wall/CPU cost and peak GPU memory on failure. Do not
retry or extend silently. The CLI requires the output directory,
--main-directory, --summary-sha256, --launch-sha256, --source-sha256 and
--protocol-sha256. Main launch/summary pins are supplied after that run exists;
the diagnostic declaration precedes its scores.

This measures representation readout on sampled training examples. Better
auxiliary accuracy does not establish native use, transfer or comprehension;
poor readout does not establish that the encoder contains no facts. Keep
conditional-value accuracy, knownness and full argmax separate, avoiding an
all-unknown accuracy headline. An unsuccessful slower arm is evidence only for
that rate at 648 updates, not against longer learning or every lower-rate
setting. No diagnostic threshold overrides the main screen or promotes a model.
