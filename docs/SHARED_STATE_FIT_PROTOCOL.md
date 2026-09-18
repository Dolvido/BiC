# Shared-state trainability diagnostic

Prospective local diagnostic following the failed shared-state pilot. Frozen before
new model construction or training. This asks whether the existing shared learner
can fit a small, fixed, balanced teaching buffer, and how this depends on update
size and competition among objectives. It does not test generalization or promote
a checkpoint. All four conditions finish at the declared endpoint irrespective
of intermediate scores.

## Fixed comparison

Use the unchanged SharedStateStudent, including its 107-class shared state head,
and the previous width 192 / 4 layers / 4 heads / feedforward 768 configuration. Position capacity 1024,
max turns 12, input 128 bytes, output 32 bytes. Four fresh models start with exactly
the same full weights, seed 852404001. No weights from the failed pilot are reused.
The original raw-English policy, targets, label meanings and state loss remain
unchanged. AdamW retains its defaults except the declared learning rate; clipping
is 1 and execution uses the existing strict local FP32 profile.

| Arm | Learning rate | Differentiated loss per family |
|---|---:|---|
| joint_fast |0.003| original sequence objective + 0.3 state loss |
| joint_slow |0.0003| original sequence objective + 0.3 state loss |
| state_fast |0.003|0.3 state loss only |
| state_slow |0.0003|0.3 state loss only |

Each update averages the three family losses before one optimizer step. All arms
perform the same teacher-forced native forward and auxiliary readout. Original
native losses can be recorded in the state-only arms but do not contribute to
their gradients. Native output heads can remain untrained in those arms; their
policy scores are descriptive and are not a transfer/adoption criterion.

## Data and work

Only the authenticated 108 actual training rows already used in the prior fit
bank are admitted: twelve rows for each of color/count/switch crossed with lengths
8/10/12. The source is runs/shared-state-data-local/attempt-001, manifest SHA256
266dfe643074b8d66b65b1e506f6fc726fa6cb75f2a532b04232ec87bf30eb75 and bank SHA256
c42c117d04070cae026e6372615bcb044b493ff41f80603a268dc8aff93e4efb.
The previous launch/summary and source identities authenticate provenance.
No new lessons, tutor calls, development targets or audit targets are used.
Decoding the existing archive also decodes its other banks; discard them without
selecting rows or scoring them. No capability claim treats these reused rows as
unseen data.

Rotate lengths 8, 10, 12 in that order. At each update use all 12 rows per family for
that length: 36 episode exposures per update. Fix 432 updates per arm, giving 144
passes over the buffer, 15,552 exposures per arm, 1,728 total updates and 62,208 total
training episode exposures. Rotate the first arm in each four-arm round so order
does not systematically favor one condition. Prepare and validate targets once,
pack inputs once and reuse the same immutable tensors. Labels remain outside the
native policy's observation input.

Evaluate all 108 rows at updates 0, 216, 432, with gradients disabled. For all 4 arms
and all 3 endpoints perform 9 auxiliary forwards and 9 native policy forwards,
so 1,296 episode passes for each evaluation route, 2,592 combined. The native route
uses the existing evaluator with BOS-only start and freely generated replies.
Preserve raw records, original paired-action/reply/joint metrics and family
breakdowns. Training uses 5,184 family forwards/backwards and 1,728 optimizer calls.
Save all four complete model/AdamW checkpoints at 216 and 432; do not select an
intermediate checkpoint as the final result.

The run has a 900-second wall allowance covering setup, training, evaluation and
saving after invocation. Check the deadline between operations; an in-flight
operation can finish after the boundary. Preserve failure/partial counters and
stop on error; do not silently retry, extend or change settings. Record source,
protocol and input hashes, runtime, setup/training/evaluation/checkpoint cost,
peak GPU allocation and equal initialization. The local RTX 5080 is the only
training device. This is one initialization, not an independent replication.

## Diagnostic readouts and decision

Report the original 107-way argmax separately on known and unknown targets. Also
record conditional value top-1 over all 106 nonzero classes and true-value rank
for known targets, with no family mask. Rank is 1 plus the number of strictly
greater nonzero logits; argmax breaks ties at the lowest class index. Record P(known)=1-P(class 0), its mean on
both target groups, binary knownness confusion at the fixed threshold P(known)>0.5,
and Brier score. Preserve targets, predictions, ranks and probabilities for pure
recount. Report metrics per family and overall; pooled accuracy is insufficient.
The existing flat cross entropy already decomposes algebraically into knownness
and conditional value terms. Merely changing argmax or rewriting this same loss
would not establish improved binding.

For describing adequate fitting, require at 432 updates, in each family: knownness
sensitivity and specificity at least 95%, conditional known-value top-1 at least 90%,
and original 107-way known exact at least 80%. These diagnostic thresholds are fixed
before results. Also report the complete continuous metrics and trajectories;
failing a threshold does not establish that the architecture is incapable.

If both objective types fit, the next question is broad exposure and transfer.
If state-only fits and joint does not at the same rate, competing objectives or
their optimization becomes a candidate cause. If a lower rate alone fits, update
size is implicated under this fixed setup. If neither fits, investigate shared
representation/retrieval or teaching structure before another broad-data run.
These are hypotheses, not uniquely identified causes. Success may be memorization.
No score here establishes English comprehension, broad intelligence, useful LLM
teaching or independence. The larger home-learning objective remains active.
