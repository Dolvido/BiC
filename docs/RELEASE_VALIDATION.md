# Public source validation

The public checkout has a small, explicit validation command that uses the
bundled inference weights and requires no historical `runs/` directory, Ollama
service, or GPU:

```sh
python -m pip install -c constraints-cpu.txt torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-cpu.txt -e .
python -B tools/validate_release.py
```

Run this from the repository root in a Python environment with the project
dependencies installed. Installation may need network access; the checks do not
contact a live tutor or download models. An optional `--report result.json` writes
a machine-readable receipt. The validator verifies that imported project modules
come from this checkout, which catches accidentally using another editable copy.

## What is checked

| Test file | Cases | Coverage |
| --- | ---: | --- |
| `test_launcher.py` | 8 | Path validation, read-only diagnostics, child-process cleanup |
| `test_offline_home.py` | 6 | Menu, budget input, source/model pins, mocked process delegation |
| `test_home_learning.py` | 8 | Retained identity, locks, resume boundaries, failure handling |
| `test_definition_tutor_author.py` | 9 | Mocked local tutor transport, durable responses, checked recipes, fallback |
| `test_definition_curriculum.py` | 10 | Hand-computed copy/revision/composition answers, causal prefixes, pair validation |
| `test_shared_rate_continuation.py` | 9 | Synthetic checkpoint metadata, optimizer rate boundaries, accounting |
| `test_shared_rate_run.py` | 11 | Matched replay coordinates, parent eligibility, budget/completion contracts |
| `test_public_demo.py` | 9 | Bundled weights, CPU inference, oracle isolation, input limits, immutable parameters |

The command also compiles Python source in memory and checks five CLI help paths:
the main application, launcher diagnostics, offline menu, home continuation, and
rate-study runner. It does not launch those workflows.

It then runs `python tools/verify_evidence.py` to recompute the published aggregate
arithmetic, and the bundled CPU demo with `--json --threads 2`. The demo runs all
four fixed scenarios (27 turns), including unsuccessful predictions; these are
illustrations rather than an accuracy benchmark. Each command has a 30-second
timeout. Neither command trains the learner or calls the tutor.

## Recorded local result

On 18 September 2026, the isolated public release checkout passed **70 tests, with
zero failures, errors, or skips**. It passed syntax checks for 452 Python files
and all five CLI help checks. The machine used Windows, Python 3.12.14, PyTorch
2.11.0+cu128, NumPy 2.5.3, and Pillow 12.3.0. Aggregate verification and all 27 CPU
demo turns also completed. The complete check took 6.878 seconds. See
[the JSON receipt](release-validation.json) for the exact invocation result and
imported-module inventory.

The interpreter and dependencies were reused from the development environment;
project source was resolved from the separate public checkout. This is a clean
source validation result, not a claim of a fresh dependency installation or Linux
execution. No historical run directory was copied for these checks.

## CI and limits

[GitHub Actions](../.github/workflows/tests.yml) runs the same command on Linux with
Python 3.12 and CPU PyTorch 2.11.0. That wheel is available from the official
[PyTorch CPU index](https://download.pytorch.org/whl/cpu/torch/).
[The CPU constraints](../constraints-cpu.txt) also pin NumPy 2.5.3 and Pillow
12.3.0. The committed local receipt is separate from the current Actions result.

These checks validate selected contracts and curriculum semantics. They do not
replicate the learning-rate study, measure tutor benefit, validate the complete
historical test collection, or prove end-to-end training from a fresh checkout.
Some historical tests intentionally require unpublished run artifacts. Use the
explicit command above for the public source check; unrestricted test discovery
is not the supported release validation command.
