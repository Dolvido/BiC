# Shared representation diagnostic results

The completed diagnostic does not support a readout-only explanation of weak
shared learning. New readouts recover some decisions, but trained representations
do not consistently beat matched untrained features. The prescribed follow-up is
the small gradient inspection across all families before selecting a new training
comparison. No learner weights changed or were promoted.

The run used the historical common initialization and both predetermined
1,536-update endpoints. It does not directly diagnose the newer hierarchical
architecture. Two newly generated datasets each contain 504 complete pairs,
168 per family, with 216 primitive and 288 composed pairs. Both sets use the
known train-admissible grammar. Exact transcripts are disjoint from each other
and every authenticated historical/formal exclusion. Program identities still
overlap in three cases; this is fresh sampling, not new algorithms.

## Paired decisions and controls

Counts below require both opposite-answer members correct. Each fitting and
evaluation entry has denominator **504 pairs**. A probe is one shared four-class
affine readout over all families, not generated English. The untrained control
receives the same extra fitting labels as trained states.

| State | Readout | Fitting pairs | Evaluation pairs | Evaluation color / count / switch, each /168 |
|---|---|---:|---:|---:|
| Initial | Original action | 13 | 13 | 11 / 0 / 2 |
| Curriculum | Original action | 47 | 57 | 28 / 0 / 29 |
| Mixed | Original action | 21 | 29 | 6 / 0 / 23 |
| Initial | EOS true labels | 117 | 65 | 28 / 3 / 34 |
| Initial | EOS shuffled labels | 42 | 27 | 10 / 1 / 16 |
| Curriculum | EOS true labels | 102 | 77 | 39 / 1 / 37 |
| Curriculum | EOS shuffled labels | 22 | 22 | 12 / 1 / 9 |
| Mixed | EOS true labels | 78 | 58 | 23 / 0 / 35 |
| Mixed | EOS shuffled labels | 39 | 34 | 11 / 1 / 22 |
| Initial | Reply context, true labels | 106 | 99 | 48 / 9 / 42 |
| Initial | Reply context, shuffled labels | 36 | 28 | 8 / 2 / 18 |
| Curriculum | Reply context, true labels | 82 | 63 | 31 / 1 / 31 |
| Curriculum | Reply context, shuffled labels | 20 | 12 | 6 / 1 / 5 |
| Mixed | Reply context, true labels | 70 | 53 | 25 / 0 / 28 |
| Mixed | Reply context, shuffled labels | 25 | 26 | 10 / 2 / 14 |

True-label probes beat their own shuffled-label controls, including at untrained
initialization. The curriculum EOS probe improves on its original action head,
but only modestly exceeds the initial EOS probe; counting remains at 1/168 pairs.
Both trained reply-context probes trail the initial reply-context probe. These
are single historical endpoints and one common sampled dataset, not replicated
causal effects of training.

All displayed fitting and evaluation scores use true canonical labels, including
the shuffled-control fitting rows. Shuffled targets are used only for fitting
the corresponding control readout, not for evaluating its predictions.

**Every probe returned a finite but unconverged endpoint after its fixed 200
iterations.** Final maximum absolute gradient components range from
`5.4752131599613185e-6` to `5.858094790883285e-4`, above the prescribed `1e-7`.
No iteration limit, rate or regularization was retuned. These descriptive results
show what this bounded fitting procedure recovered; poor scores do not prove
that a representation lacks relevant information.

## Reply semantics

| State | Original first-byte fitting pairs /504 | Evaluation first-byte pairs /504 | Evaluation complete-reply pairs /504 |
|---|---:|---:|---:|
| Initial | 0 | 0 | 0 |
| Curriculum | 7 | 6 | 6 |
| Mixed | 0 | 0 | 0 |

First-byte and complete-reply correctness counts agree in every reported group.
Independent cached decoding also agrees with the BOS-only first-token argmax on
all **30,240 evaluation turns**. The first byte distinguishes No, Yes, I-need-help
and Understood without receiving any correct reply prefix. Thus later word
generation is not the measured endpoint explanation for these semantic failures.
The refitted reply-context readouts recover more paired decisions, but the
stronger initialization control prevents interpreting that as intact learned
general understanding.

The true probes also trade uncertainty behavior for known-query accuracy:

| State / representation | Known-query correct /2288 | Unknown ASK /2626 | Unsupported ASK on known /2288 |
|---|---:|---:|---:|
| Initial / EOS | 34.40% | 55.48% | 18.84% |
| Curriculum / EOS | 44.62% | 69.61% | 20.41% |
| Mixed / EOS | 38.16% | 71.06% | 27.10% |
| Initial / reply context | 36.63% | 55.45% | 19.10% |
| Curriculum / reply context | 43.53% | 70.30% | 20.28% |
| Mixed / reply context | 38.29% | 74.11% | 28.54% |

Original curriculum/mixed action heads answer 34.70%/34.44% of known queries
correctly while correctly asking on 84.54%/81.65% of unknown queries. Readout
refitting therefore cannot be declared an unqualified behavior improvement.

## Cost and integrity

The successful local run finished in **70.032 wall seconds**, **67.296875 process
CPU seconds**, with **293.65 MiB peak allocated / 862 MiB reserved CUDA memory**.
Wall phases include validation and copies; these are not active GPU-kernel hours.
Input authentication took 13.343 seconds and preparing/sealing both datasets
16.157 seconds. The six paired true/shuffled fitting phases took about 7.78 seconds.

- 6,048 attempted and completed frozen encoder episode passes.
- 60,480 separate BOS decoder row-steps.
- 30,240 generated reply turns and 836,640 physical generated decoder row-steps.
- 12 probe fits, 2,400 internal solver iterations, 2,597 objective/gradient
  evaluations and 26,177,760 supervised turn exposures inside those evaluations.
- 241,920 detached probe prediction rows.
- Zero source-learner backward evaluations or optimizer updates.

All selected checkpoint weight digests match before/after use. Three safe
checkpoint deserializations loaded only the specified historical states; no
optimizer was restored. All 159 launch-source pins and all completed artifact
hashes were checked, including the 88 unchanged formal-study sources.

Attempt 001 failed in 2.078 seconds during a wrapper runtime lookup, before any
diagnostic operation. That evidence is preserved. An explicit one-line startup
repair was verified without model work; attempt 002 used unchanged samples,
states, settings and limits. This startup cost is additional to the successful
run and is not hidden inside its timing.

## Next experiment

Subsequent work: the fixed [gradient inspection](FOUNDATION_GRADIENT_RESULTS.md)
described below has completed. Query and reply gradients aligned in every
measured group; the reply contribution was small, and broad auxiliary conflict
was unsupported. A separate prospective
[reply-objective comparison](FOUNDATION_OBJECTIVE_STUDY_PROTOCOL.md) is now
running. No new learning benefit has been established.

The shared-learning explanation remains unresolved. Execute the already
specified zero-update gradient inspection: first four fitting pairs in every
cell, both historical endpoints, 42 matched three-family groups and 1,008
episode forwards. Compare query, acknowledgement, reply and observation-byte
gradients using the original family and turn weighting. Any conflicting or
disproportionate gradients will motivate a separate prospective causal training
comparison; they cannot by themselves prove that removing a loss helps.

The next improvement must be common across families, preserve uncertainty and
retention, and improve useful learning per local compute hour. No per-cell lesson
patch, architecture promotion or claim of general intellect follows from this
diagnostic.

## Evidence

- Completed run and actual costs (archive reference: `../runs/foundation-shared-diagnostic-local/attempt-002/execution.json`).
- Full result (archive reference: `../runs/foundation-shared-diagnostic-local/attempt-002/result.json`),
  SHA256 `f25a262d4b88d084e8c6d09017ee85d91ae9ab0bb71c98722f0796a3dd883b18`:
  all cells, families, primitive operators, depths, lengths, turns, confusion,
  convergence residuals and original/probe scores for both roles.
- Artifact bindings (archive reference: `../runs/foundation-shared-diagnostic-local/attempt-002/artifacts.json`)
  and execution/recovery record (archive reference: `../runs/foundation-shared-diagnostic-local/README.md`).
- [Prospective diagnostic](FOUNDATION_SHARED_LEARNING_DIAGNOSTIC.md) and
  [completed architecture comparison](FOUNDATION_VARIANT_RESULTS.md).
