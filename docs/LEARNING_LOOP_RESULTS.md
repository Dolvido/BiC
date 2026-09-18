# Local curriculum validation — September 16, 2026

The self-directed curriculum loop reached **93.19% mean audit accuracy** across
three training seeds, compared with **25.93% before training**. The matched
round-robin scheduler reached **89.32%**. These are results for six simulated
tasks with numeric observations and skill tokens, not English comprehension or
general intelligence. Spatial reasoning remains the clearest weakness.

The full evidence is in the validation report (archive reference: `../runs/curriculum-local-validation/report.json`),
with the training-complete record (archive reference: `../runs/curriculum-local-validation/training-complete.json`)
and predeclared plan (archive reference: `../runs/curriculum-local-validation/plan.json`).

## Protocol and compute

Both schedulers used training seeds 101, 202, and 303; the same thirteen-region
BiC architecture with hidden size 32; two candidate branches; and the same
training, retention, and promotion rules. Each run completed **120 cycles and
24,000 candidate updates**, for **144,000 updates across six runs**. This matches
the cycle/update budget, rather than claiming identical wall-clock cost or
identical lesson streams. Topic selection changes the lessons encountered.

Training used the procedural tutor. Each cycle generated fresh practice and
mixed older examples with fresh variants for retention. Candidate selection used
development data only. All six training jobs finished before any final audit.
The audit evaluated the accepted checkpoint, with no tutor involvement, against
the corresponding untrained checkpoint. Every audited checkpoint remained
unchanged by evaluation.

Audit seed 800000001 supplied **512 examples per skill**, or **3,072 scenarios
per model**. All six models faced the same scenarios. There are three paired
training-seed replications, not three independent audit datasets. The standard
deviations below describe variation across training seeds; they are not
confidence intervals over fresh worlds.

The six runs took **531.55 seconds total elapsed training time** with three jobs
in parallel. Individual runs took 263.06–265.67 seconds, approximately 4.4
minutes, on the local CPU configuration. Concurrent candidate runtimes overlap;
their sum should not be interpreted as independent elapsed compute hours.
The recorded per-run cap was 0.2 hours, and every run reached its cycle limit.

## Audit scores

| Training seed | Untrained | Progress selection | Round robin | Paired difference |
|---|---:|---:|---:|---:|
| 101 | 27.96% | 93.42% | 88.83% | +4.59 pp |
| 202 | 23.70% | 91.60% | 91.73% | −0.13 pp |
| 303 | 26.14% | 94.53% | 87.40% | +7.13 pp |
| Mean | **25.93%** | **93.19%** | **89.32%** | **+3.86 pp** |
| Sample SD | — | 1.48 pp | 2.21 pp | — |

Progress selection improved each seed by 65.46–68.39 percentage points over its
untrained student. It exceeded round robin in two of three paired runs. This
small experiment supports the implemented learning mechanism; it does not
establish general superiority of the progress scheduler.

| Skill | Progress selection, mean | Round robin, mean |
|---|---:|---:|
| Color | 95.90% | 96.55% |
| Shape | 94.86% | 89.26% |
| Spatial | **71.35%** | **57.75%** |
| Count | 98.89% | 96.48% |
| Compare count | 99.09% | 97.92% |
| Delayed recall | 99.02% | 97.98% |

Spatial accuracy under progress selection ranged from **60.35% to 86.33%**.
A high overall average therefore does not mean every skill is mastered.
The comparison-count task has three valid labels; the other tasks have four.
Uniform guessing over each task's valid labels averages **26.39%**.

Mean multiclass Brier probability error was **0.0957** for progress selection,
**0.1557** for round robin, and **0.7546** for the untrained baseline; lower is
better. This is a probability-error measure, not proof of calibration.

The predeclared gates passed: progress-selection mean accuracy at least 75%,
every progress seed improving at least 20 percentage points over its initial
checkpoint, and no checkpoint changes during audit. These gates do not certify
open-ended learning or performance beyond the tested curriculum.

## Local tutor probe and earlier development work

A separate local tutor probe (archive reference: `../runs/loop-tutor-probe/report.json`) used the
installed `ministral-3:3b` model. Both of two requests returned accepted selections
of verified explanations, follow-up questions, and practice packets. All eight
returned practice examples passed independent verification. The requests took
2.875 and 0.312 seconds respectively, including the first request's model load.
The model loaded by the probe was unloaded afterward.

That probe performed **no student training**. It demonstrates that the local
tutor connection and bounded practice-selection interface work. It provides no
evidence that LLM tutoring improves learning compared with procedural tutoring;
the six-run validation used procedural tutoring throughout.

Earlier small-corpus probes used development examples and exposed overfitting
from repeatedly training on a tiny fixed set. Those exploratory observations
informed fresh practice and rehearsal design. They remain development evidence
and are not pooled with, or relabeled as, the final audit.

## Reproducibility and remaining scope

Each run retains its initial and accepted checkpoints, configuration, audit
report, and copies of the frozen source files. For example, the
progress-101 protocol (archive reference: `../runs/curriculum-local-validation/progress-101/protocol.json`)
records provenance, while its source directory (archive reference: `../runs/curriculum-local-validation/progress-101/source`)
preserves the implementation used. The report records source, curriculum,
checkpoint, and model hashes. Source changes require a new experiment.

The audit tests new deterministic scenes and a disjoint nuisance-identity range
within the same six task rules. Prompts are display metadata in this experiment.
The results do not measure photographs, natural speech, unrestricted English,
new scientific concepts, subjective welfare, or autonomous source-code changes.
The separately developed English dialogue integration needs its own evidence
for meaning learned through text, correction, uncertainty, and persistent
conversation. More hours alone cannot establish those capabilities.
