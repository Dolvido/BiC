# Binding-loss comparison after the English frontier study

**Completed:** the 2,000-update run improved known-binding readout accuracy but
failed its transfer-benefit screen. See [the measured results](ENGLISH_FRONTIER_RESULTS.md)
and `runs/english-frontier-local/binding-balanced-1101/`. The prospective
design below records the comparison as declared before its outcome.

The bounded local `binding_balanced` experiment has launched; its result is
pending. Its frozen protocol (archive reference: `../runs/english-frontier-local/binding-balanced-1101/protocol.json`)
records the arm, budgets, source hashes, and dataset settings. The standalone
[runner](../experiments/train_english_binding.py) reuses the original frontier
trainer without changing its source or the student architecture. This document
states the comparison and its limits, not an outcome.

## Evidence motivating the comparison

The seed-1101 scaffold branch reached approximately 97.4% complete-pair
correctness on the familiar training support by 750 updates, while the
unseen-binding/familiar-wording diagnostic scored approximately 6.25%.
The contemporaneous auxiliary inspection reported 100% alias and color
recognition, 100% permitted-color recognition, about 10.84% known-binding
recognition, and about 98.51% unknown-binding recognition. These intermediate
results motivate a controlled experiment; they do not establish a final result
or select a checkpoint for deployment.

The binding loss currently averages eight aliases after every turn. Most aliases
have not been defined, so unknown targets dominate. The readout is also an
independent five-class classifier for each alias: colors outside that alias's
training support never receive positive supervision for that output row.
Balancing known and unknown targets addresses the first issue without claiming
to solve the second.

A weak auxiliary classifier does **not** establish that the recurrent state
lacks a fact. It establishes failure of this jointly trained readout on its
stated distribution. Policy correctness, exact context-dependent answer changes,
and conditional transfer measurements remain necessary evidence.

## Frozen comparison

Compare the existing `scaffold` branch with one new `binding_balanced` branch.
Both start from the same seed-1101 student initialization and seed-1102 auxiliary
initialization. Do not warm-start the new arm from an intermediate checkpoint.
Use the same 3,072 canonical training dialogues: 1,024 per focus, starting at
120000, 130000, and 140000. Keep the same paired sampler and its random stream.

Both arms use 2,000 optimizer updates, batch size 128, AdamW at 0.003, gradient
clipping at 1.0, and checkpoints every 250 updates. Keep all sentence/state
heads, parameters, label generation, query-class balancing, acknowledgement
weight, decoder weight, and auxiliary weight unchanged. The objective remains:

```
total = action + 0.1 * reply + 0.5 * auxiliary
auxiliary = mean(role, alias, color, permitted, bindings)
```

Change only the reduction of the binding head's ordinary five-class cross
entropy. For each turn, define valid known positions as masked targets unequal
to `UNKNOWN_COLOR`, and valid unknown positions as masked targets equal to it.
Average loss within each nonempty group, then average the group means:

```
binding_loss = mean(nonempty_groups(mean(cross_entropy(group))))
```

When both groups exist, their weights are one half each. When one group is
empty, the other retains its full mean; when both are empty, use a differentiable
zero. Do not add a second binding loss or accidentally change its weight relative
to the other four heads. Preserve the ordinary masking contract, including
ignoring invalid labels at masked positions.

Both branches use the local CUDA device. The runner defaults to a 1,800-second
elapsed-time limit, checked between chunks, plus final evaluation. It records
the stopping reason and elapsed time; its protocol does not currently save the
`max_seconds` argument. A time-limited partial run is not a 2,000-update
comparison. Compare the largest common completed checkpoint if either arm
stops early, and label the mismatch explicitly. There is no automatic champion
promotion.

## Evaluation and interpretation

Retain all checkpoints and the established four-slice diagnostic bank at seed
180000, 64 dialogues per focus. It separately tests familiar/unseen binding
support and familiar/unseen phrase support. These are known development
benchmarks, including when episode seeds or exact transcripts are new; they
are not pristine audit evidence. Phrase contrasts share source worlds; binding
contrasts use their separate canonical generators.

The current manifest check found zero exact transcript overlaps between the
training bank and the familiar/familiar matrix slice. The separate familiar
support probe at seeds 240000, 250000, and 260000 contains two training
transcripts among its 192 episodes. Preserve this disclosure. Use the matrix
slice as the cleaner familiar-support comparison; do not silently relabel the
support probe as strictly held out. Neither absence of transcript overlap nor a
fresh seed establishes new templates or new binding support.

At each checkpoint the existing runner collects query and complete-pair
accuracy, pair counts, query loss, and per-target/per-kind metrics on the
support probe and four diagnostic slices. It does not run the auxiliary
inspector at every checkpoint. Final evaluation adds freely generated reply
scores and reset-state/blank-text controls on the familiar support probe, with
unchanged-parameter checks. The final study evaluation supplies per-focus
results; these should not be inferred from pooled checkpoint accuracy.

The separate auxiliary inspector reports known/unknown binding accuracy,
denominators, per-alias known-binding accuracy, permitted-color recognition,
and sentence-field recognition by role on familiar training support only.
It does not establish unseen-binding or unfamiliar-phrase readout accuracy,
retention by delay, or action/reply agreement. Those would require additional
diagnostics. Its parser-derived labels remain outside student inference, and
canonical training admission remains unchanged.

The decision endpoint is update 2,000, not the best checkpoint selected after
viewing the curves. Treat this as a one-seed exploratory comparison. Consider
balancing promising if unseen-binding/familiar-phrase complete-pair accuracy
improves by at least 10 percentage points over the matched scaffold baseline,
while familiar/familiar query and complete-pair accuracy each fall by no more
than 2 points and ASK accuracy falls by no more than 2 points. Inspect all
phrase-transfer results and generated replies even if this screen passes.
The existing competence gates of 80% query accuracy, 60% complete-pair accuracy,
60% query accuracy in each focus, 75% allow/deny macro accuracy, and a 10-point
paired advantage over controls remain unchanged. Passing this development
screen is not an independent audit or sufficient for automatic promotion.

If only known-binding auxiliary accuracy improves, the intervention has repaired
its training target but has not established better policy transfer. If familiar
policy accuracy improves without unseen-binding transfer, scaling this loss
alone is not supported. If phrase transfer alone remains weak, investigate the
sentence representation separately. A promising policy-transfer result should
be replicated with new initializations and the same frozen endpoint before
claiming robustness.

## Isolated implementation and checks

`experiments/train_english_binding.py` defines `BalancedScaffoldHeads`, a
training-only subclass with identical parameter layout. It changes only the
binding-loss reduction. Scoped, process-local substitutions let the unchanged
frontier trainer instantiate this subclass and let its existing runner call the
new training wrapper. Substitutions restore on exit, including failures. The
experiment must not share a process with concurrent training threads; separate
processes retain their ordinary classes and functions.

The fixed arm tag is `binding_balanced`. The wrapper includes its own SHA-256 in
the original source manifest, and the existing protocol comparison rejects
resume under changed source or settings. It runs in an isolated output
directory. No existing trainer or student source needed editing, and no
auxiliary classifier or parsed target enters the action policy at inference.

The dedicated test file covers an imbalanced known/unknown example against
independent group-mean calculations; known-only, unknown-only, fully masked,
and masked-invalid-label cases; unchanged nonbinding losses and parameter
layout; exact CPU continuation of student weights, auxiliary heads, optimizer,
and sampler; substitution restoration after failures; held-out training
rejection; and inclusion of the new runner's source hash. Existing scaffold
tests cover the separation of auxiliary targets from student inference.

## Subsequent factorized binding readout, only if warranted

If balancing improves known-binding recognition without meaningful transfer,
test a separate `shared_binding_scaffold` arm. Keep the policy architecture and
student inputs unchanged. Replace only the disposable unshared binding readout
with a learned alias-query embedding concatenated with the recurrent activity,
followed by a shared small nonlinear five-color/unknown classifier:

```
binding_logits[alias] = shared_classifier(concat(recurrent_activity,
                                               learned_alias_query[alias]))
```

Every alias passes through the same classifier, so color-output parameters can
receive positive supervision from different aliases. This encourages reusable
decoding without guaranteeing compositional memory or preventing alias-specific
memorization. Keep the balanced reduction fixed if comparing against the balanced
arm, and disclose the auxiliary parameter-count change. Never pass the queried
alias embedding or parsed facts into the policy; these remain supervision-only
readouts that disappear at inference.

This is a later hypothesis, not part of the balancing comparison. Reinitialize
from the same student seed and matched budget, preserve the same data and
diagnostic matrix, and compare policy transfer as well as readout accuracy.
Do not infer better reasoning merely because a more expressive probe can decode
more facts. A separate frozen-state probe comparison can help distinguish a
better readout from a changed representation, but its probe-training data and
evaluation data must be separated and disclosed.
