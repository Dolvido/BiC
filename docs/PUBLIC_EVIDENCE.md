# Public evidence and what it verifies

The repository includes a compact, derived aggregate of the completed learning-rate
comparison in [shared-rate-aggregate.json](../evidence/shared-rate-aggregate.json).
It contains integer successes and denominators for all 16 bank trajectories, all
12 acquisition cells, and the metrics used by every retention check against five
fixed references. It includes failures as well as improvements.

Run the arithmetic check from the repository root with Python 3.10 or later:

```console
python tools/verify_evidence.py
```

No PyTorch installation, GPU, model, tutor, internet connection or training is
needed. The command exits successfully only when the exported counts are valid,
family totals agree with overall totals, compared denominators match, both arms
share the starting metrics, and the recomputed screens agree with the reported
outcomes. It prints all failed retention checks. Arithmetic uses exact fractions;
the 6.25 percentage-point retention loss is not rounded into a pass.

The recomputed results are:

| Measure | Control, 0.0003 | Lower rate, 0.0001 |
| --- | ---: | ---: |
| Acquisition cells at least 75% | 4/12 | 6/12 |
| Original basis capability cells at least 75% | 2/12 | 3/12 |
| Retention checks passed | 465/480 | 479/480 |
| Eligible for adoption | No | No |

Lower minus control is **+24.375 percentage points** across the equally weighted
12 acquisition cells. Its fit and development means are +30.8333 and +17.9167
points. The separate six-cell composition/sequence effect is +0.5208 points,
below the required +5-point improvement. Neither arm was selected or adopted.

## Provenance

This file is a **projection of an original report**, not the original report or
a complete experiment archive. The source report had status `verified` after a
separate local recount of saved evaluation records and training ledgers.

| Artifact | SHA-256 |
| --- | --- |
| Public aggregate | `be4cb587e298739cd745a6b3189bb92ed4ac3e71f1d8acba518f983128bcbb3f` |
| Original independent report | `922679ce9b665c836cd7f21b8af6aa5de0d0c23545da6e7d69ff30bb7e8e2943` |
| Original frozen launch | `057a6a06932749fd712dd10d32b9e68e6625d2db510aa739f8302af4a6bc21cf` |
| Original completed summary | `6345f2128674853ea052712e590660914787defcf75ad75544ae2da59c0231ed` |

If the original report is separately available, the verifier can authenticate
its exact bytes and require that its deterministic projection equals the public
aggregate:

```console
python tools/verify_evidence.py --source-report path/to/independent-analysis.json
```

Without that optional file, the output explicitly reports
`original_report_authenticated: false`. The stored source hash identifies an
artifact; by itself it does not authenticate the underlying experimental claims.

## What remains outside this check

The public aggregate does not include raw episode predictions, full optimizer
archives, or the complete historical input closure. The standard command verifies
the published arithmetic and internal consistency. It cannot independently
rescore predictions, prove that source records are complete, reconstruct training
history, or rerun the original controlled comparison.

These scores come from one inherited optimizer history, a finite English grammar
and repeatedly observed evaluation banks. Development examples are held-out text
within that grammar. The measured improvement does not establish general English
comprehension, open-ended program execution, learned self-direction, reliable
tutor benefit or a result replicated across random seeds.

For all trajectories, study costs and the original decision rules, see
[the full results discussion](SHARED_RATE_RESULTS.md) and
[the prospective protocol](SHARED_RATE_TRANSITION_PROTOCOL.md).
