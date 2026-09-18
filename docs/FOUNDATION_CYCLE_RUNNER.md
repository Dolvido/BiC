# Bounded local continuation

`experiments.foundation_cycle_run` resumes a caller-pinned `CycleOwner` under an
explicit update, completed-cycle or wall-time allowance. It uses the owner's
existing curriculum, evaluation banks, model and optimizer. There is no latest
checkpoint discovery, architecture selection, teacher call or model promotion.

From the repository's local Python environment:

```powershell
python -m experiments.foundation_cycle_run --owner <owner-directory>/owner.pt --expected-sha256 <trusted-current-owner-sha256> --receipt <separate-results-directory>/invocation-001.json --hours 1
```

Paths and the hash above are placeholders, not a command to run against a
particular checkpoint. The creator or preceding successful invocation supplies
the exact current owner hash. Store that pin under the caller's control. A hash
copied from an editable neighboring file does not independently establish its
provenance. The runner returns the resulting owner hash in its result and stdout;
it never chooses one from a directory automatically.

`--max-updates` and `--max-cycles` can be combined with `--hours`; the first
exhausted allowance stops continuation. Updates count work completed by that
invocation. Cycles count new completed boundaries, without recounting an earlier
boundary on restart. The wall clock includes loading. Loading, admission,
scoring and publication are nonpreemptive, so an allowance is a stopping boundary
between operations rather than a hard process deadline. `--max-cycles 0` permits
an authenticated no-learning check. An exhausted update allowance can still
finish an active evaluation panel according to the owner's contract.

The default device is CPU with one intra-operation thread. `--threads` and
`--interop-threads` explicitly configure the saved thread counts; omitting the
latter retains the fresh process's inter-operation setting. CPU execution keeps
the process's other settings unless `--strict-profile` is supplied. An explicit
CUDA device such as `--device cuda:0` always configures the declared strict profile before loading or
initializing CUDA. All saved runtime and source fields must still match exactly;
there is no automatic migration. These options do not establish a new GPU
reproducibility guarantee. Preserve the saved device's exact spelling: `cuda`
and `cuda:0` remain distinct capsule identities. Reusing a formal-study checkpoint requires an explicit
compatible, validated import, not a path passed to this runner.

The result path must be new and outside the owner directory. A complete,
exclusive invocation intent is published before loading; the result is also
published as complete JSON without overwriting existing evidence. The owner
separately journals actual training, scoring, retained and discarded work. A
crashed CLI may have no result even when the owner durably recorded progress.
Inspect that evidence and recover the exact trusted owner identity before
continuing; do not automatically retry from the old pin.

Missing, incomplete or explicitly unknown owner work requires the owner's
separate recovery policy. `--recovery-policy acknowledge_unknown` records an
explicit acceptance of that uncertainty without converting it to zero. Existing
run locks are resolved independently using actual process identity. This option
is a programmatic recovery choice, not an instruction to seek new user permission.

Eleven focused CLI tests passed with a fake owner and mocked runtime setup,
including finite bounds, deadline accounting, pin forwarding, runtime setup
before load, failure reporting, exclusive publication and preventing writes
inside owner state. Review found missing explicit runtime setup after the first
eight tests, then an explicit `cuda:0` spelling gap after the ten-test revision;
all revisions and receipts are preserved. No run performed model construction,
forward passes, CUDA initialization or optimizer updates. The
current receipt (archive reference: `../runs/foundation-cycle-cli-validation-local/attempt-003/report.json`)
does not substitute for actual repeated-cycle owner tests or a GPU proof.

An actual separate-process CLI check (archive reference: `../runs/foundation-cycle-cli-validation-local/attempt-004/actual-cli.json`)
then loaded the preserved owner from its supplied pin and stopped at
`--max-cycles 0`. The saved owner hash and 202 lifetime updates stayed unchanged;
there were zero new updates, forward/scoring attempts or evaluation episodes.
Loading took 11.500 seconds and the bounded invocation 2.797 seconds, 14.297
seconds total excluding Python import/startup. These are observed tiny-fixture
costs, not evidence that restart overhead is negligible. The owner loaded its
current parent and active index, without reconstructing ancestral learners.

The runner executes prescribed broad practice. Adaptive selection, useful
transfer, retained ability and declining dependence on an external tutor remain
learning outcomes that require prospective measurement.
