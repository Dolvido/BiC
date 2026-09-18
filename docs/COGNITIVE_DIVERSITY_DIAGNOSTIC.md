# Curriculum diversity: development diagnostic

The completed sequence endpoint fits its familiar binding and conditional
examples substantially better than development examples. Graph fitting remains
incomplete. This supports testing curriculum diversity and consistent naming
as shared learning hypotheses, without assigning a proven cause or modifying
the running comparison.

This diagnostic reads only the main sequence report (archive reference: `../runs/sequence-study-local/main/sequence/report.json`)
and regenerates the frozen **main training/development** recipes. It loads no
model, computes no new predictions and accesses no withheld audit bank. Exact
metrics, counts, normalization method and SHA-256 provenance are saved in
diversity-diagnostic.json (archive reference: `../runs/sequence-study-local/diversity-diagnostic.json`).
These are development findings, not a final transfer or promotion decision.

## Fitting versus development

The seed-2501 sequence learner completed 3,600 updates, with 1,200 per family.
The fixed training-fit diagnostic evaluates only the first 128 rows of each
512-row level block: 256 episodes per family. Development contains 64 episodes
per level, or 128 per family. Both use levels 1/2. Training-fit scoring excludes
reply generation; the table compares action decisions in both banks.

| Subject | Training-fit query accuracy | Development query accuracy | Training-fit opposite pairs | Development opposite pairs |
|---|---:|---:|---:|---:|
| Binding | 729/768 (94.92%) | 221/384 (57.55%) | 125/142 (88.03%) | 22/71 (30.99%) |
| Graph reachability | 420/512 (82.03%) | 127/256 (49.61%) | 47/128 (36.72%) | 6/64 (9.38%) |
| Conditional logic | 502/512 (98.05%) | 213/256 (83.20%) | 151/152 (99.34%) | 73/74 (98.65%) |

A pair requires both opposite answers at one question position. The high
conditional pair score is dominated by initial questions, whose additional
switches are neutral. Later decisions must therefore remain visible:

| Subject | Training-fit last known later question | Development last known later question | Training-fit later pairs | Development later pairs |
|---|---:|---:|---:|---:|
| Binding | 238/256 (92.97%) | 63/128 (49.22%) | 11/14 | 1/7 |
| Graph reachability | 171/256 (66.80%) | 63/128 (49.22%) | 47/128 | 6/64 |
| Conditional logic | 183/192 (95.31%) | 74/96 (77.08%) | 23/24 | 9/10 |

The later slice selects each episode's last known yes/no question only when its
ordinal among all questions is at least two. The record also preserves all
query-ordinal counts. Graph first-question accuracy is 249/256 (97.27%) on the
training subset, while final-question accuracy is 171/256. Thus its 82.03%
overall fit does not imply that path decisions are nearly solved. Fitting and
generalization limitations can coexist.

## What the fixed banks contain

Recipes use training/development base seeds **32,000,000 / 33,000,000**, family
order binding, graph, conditional, plus 100,000 per family index and 10,000 per
level. All regenerated canonical fingerprints match the frozen
dataset record (archive reference: `../runs/sequence-study-local/datasets.json`). Each family has
1,024 distinct training transcripts in 512 counterfactual groups and 128 distinct
development transcripts in 64 groups. Exact train/development transcript overlap
is zero.

Each training family covers all **12 aliases** and all **66 allowed primary
ordered pairs**. Each development family covers all 12 aliases and 28 of its
30 allowed primary pairs. These partitioned primary roles do not prevent
incidental entity-pair overlap elsewhere in a sentence or world.

Each main family receives 76,800 repeated episode exposures: an average of
**75 presentations per training-bank row**. Sampling is with replacement, so
individual rows need not appear exactly 75 times. The main run starts fresh;
its weights do not inherit the earlier calibration updates.

| Subject | Distinct training transcripts after alias normalization | Distinct development transcripts after alias normalization | Development episodes whose normalized transcript appears in training |
|---|---:|---:|---:|
| Binding | 1,010 | 128 | 2/128 |
| Graph reachability | 682 | 114 | 58/128 |
| Conditional logic | 32 | 32 | 128/128 |

Normalization scans the six raw utterances in order and replaces each exact
whole-word alias by `ENTITY0`, `ENTITY1`, and so on at first occurrence, using
that replacement consistently through the episode. It preserves colors,
numbers, operators, punctuation, statement order, question order and all other
wording. Targets and metadata do not enter this transcript comparison.

**An alias-normalized transcript is not a proven semantic-world identity.**
This method detects literal transcript equivalence under consistent renaming;
it does not merge reordered equivalent facts, broader graph isomorphisms,
redundant statements or interchangeable color symbols. Neither its unique
counts nor ordinary episode counts establish independent samples.

The conditional bank already contains all 32 normalized patterns found in
development. Fresh procedural seeds in that family can chiefly change naming
without expanding its reasoning repertoire. Binding development also changes
many non-name combinations, so its gap cannot be attributed to naming alone.

## Proposed shared experiment, not yet frozen or launched

Test two factors under equal exposures: **fixed versus refreshed abstract-world
draws**, crossed with **bounded versus refreshed naming maps**. Separate the
world generator from its rendering so independent random streams can hold world
draws identical across naming conditions and naming draws identical across
world conditions. Keep architecture, objectives, family/level balance and
optimizer-update budget fixed. This is a proposal, not a modification to the
[current protocol](SEQUENCE_STUDY_PROTOCOL.md).

Apply each bijective naming map consistently across every utterance and both
members of a counterfactual pair. Preserve statements, questions and truth, then
independently verify the rendered examples. A renamed example needs new explicit
provenance; it must not be mislabeled as an unchanged canonical v2 episode.

Freeze a new training partition and separate fresh-world, withheld-name-combination
and combined evaluation panels before optimization. Arbitrary alias permutations
can cross the existing primary-pair partition, so training maps must be
constrained to permitted training primary roles. Existing inspected development
or audit material cannot silently regain pristine status through renaming.

Count exact and normalized diversity, unique naming assignments, repetitions,
bytes and elapsed compute separately. If a family's structural repertoire is
already exhausted, report that limitation; a genuine test of greater conditional
world diversity requires a prospectively declared broader generator. Better
performance would support the tested curriculum hypothesis within its scope.
It would not identify a unique bottleneck, demonstrate general intelligence,
or complete the broader home-learning objective.
