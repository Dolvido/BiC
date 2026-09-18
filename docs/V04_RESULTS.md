# V0.4 measured results

## Grounded English and computer actions

The **376,707-parameter** computer model retains v0.3's inference architecture. Training adds counterfactual lessons crossing each color layout with every instruction target, broader phrasing, and training-only referent supervision of regional representations. Correct actions, task identities, and referent labels remain outside inference inputs. The experiment combines several changes; it does not isolate which caused improvement.

The run executed 5,000 additional updates in **308.3 seconds** on one CPU thread. Development selection chose step **3,000**; delivered weights therefore contain the inherited 9,000 v0.3 updates plus 3,000 new updates. The 2,465-parameter referent head is discarded. Its optimizer state is incidental training metadata, not an inference component. Training record (archive reference: `../runs/grounding-v04/report.json`)

Each condition contains 128 episodes per task, 896 total. Joint success requires correct executed behavior and an exact freely generated reply.

| Matched evaluation condition | Frozen v0.3 | V0.4 |
|---|---:|---:|
| Original v0.3 familiar phrasing | 88.62% | **99.78%** |
| Expanded v0.4 familiar phrasing | 60.94% | **99.78%** |
| New sealed phrasing | 56.70% | **98.44%** |
| Sealed phrasing with shifted appearance/layout | 57.14% | **98.55%** |

The expanded bank includes phrases v0.3 never trained on; the first row is the direct familiar-wording retention comparison. On new phrasing, v0.4 action success is 99.22% and reply accuracy 98.88%. Typing remains the weakest task at 91.41% joint success; color selection reaches 97.66%. Full matched evaluation (archive reference: `../runs/grounding-v04/final_evaluation.json`)

Blanking images reduces v0.4 expanded-familiar joint success from 99.78% to **46.32%**; blanking instructions reduces it to **6.25%**. These controls support dependence on both inputs. They do not prove each of the 13 regions is necessary. Image-independent greetings and fixed-position clicks explain some surviving performance.

The old v0.3 test is now development data. New sealed phrases were excluded from optimization and checkpoint selection. All 24 color permutations and the shifted geometry family occur in v0.4 training; finite scene states may recur. Generalization demonstrated here concerns held-out phrasing within the same miniature task family, not unseen interfaces or task meanings. Results come from one training seed, not a population of independently trained models.

Delivered checkpoint SHA-256:

```text
5081e6c4767bfa27ca14ca398c6b8193a0ba56b5ec97e18e6160963cd57fd680
```

## Persistent associations

A separately trained **181,472-parameter** glyph encoder completed 1,000 updates in **83.3 seconds** on one CPU thread. Each new name receives one exemplar; teaching does not optimize encoder weights.

| Fresh identity test | Correct when known/present | Reject unknown/absent | Overall |
|---|---:|---:|---:|
| Find among four candidates | 99.61% | 100.00% | **99.80%** |
| Name among 32 stored labels | 97.66% | 97.66% | **97.66%** |

Each row has 512 queries, equally divided between known/present and unknown/absent cases. Overall accuracy therefore includes successful abstention. Random-encoder overall scores are 55.86%/51.56%; raw grayscale comparisons score 59.57%/47.66%. Each baseline receives its own validation-calibrated thresholds. Source report (archive reference: `../runs/associative/report.json`)

Sixteen earlier names remained correct after adding 32 names and saving/reloading all 48. This is frozen-encoder exemplar retention, not neural consolidation. Confidence is cosine similarity, not probability. Multiple plausible matches abstain.

A separate real HTTP trial taught four names, stopped the server, then started a second operating-system process. **4/4 names** were found with changed object order and pixels, with zero teaching calls after restart. Memory and encoder files remained byte-identical. This demonstrates persistence for those examples, not general retention under weight updates. [Two-process verification](../experiments/teaching_ui_verification.json)

An initial observed test became development evidence after correcting baseline calibration coverage and ambiguity handling. The final 128 glyph identities exclude training, validation, and that initial test; final thresholds were selected on validation only. Synthetic crops are already separated objects, and labels are symbolic addresses. Read [ASSOCIATIVE_MEMORY.md](ASSOCIATIVE_MEMORY.md) before interpreting this as word learning.

Compute profiles through 175.7 million parameters measure execution, not learned competence; see [HOME_COMPUTE.md](HOME_COMPUTE.md). The next experiments should test regional integration, retention during weight updates, and new task families with fresh evaluation partitions.

## Verification

`python -m unittest discover -s tests -q` completed **109 tests in 10.129 seconds, all passing**. Tests include inference/teacher separation, prompt partition boundaries, persistence/hash rejection, public memory API versus scoring parity, and fresh-process restoration. HTTP behavior and frontend JavaScript syntax passed the teaching integration check. Browser rendering was not verified. [Test log](../experiments/v04_tests.log)

The computer HTTP interface also passed greeting, color selection, last-click recall, and typing checks. [Interface verification](../experiments/computer_ui_v04_verification.json). Recorded scenes show six correct computer-lab turns (archive reference: `../runs/grounding-v04/demo.png`) and teaching/restart recall (archive reference: `../runs/associative/restart_demo.png`). These are execution demonstrations, not additional held-out benchmark estimates.
