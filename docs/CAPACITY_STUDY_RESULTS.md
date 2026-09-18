# Shared raw-English capacity study: completed local results

Increasing width did not establish reliable shared-program learning. Width 192 improved some counterfactual answers, especially switch actions and color replies, but all three widths almost completely failed final arithmetic pairs. This is a completed local comparison with one initialization per width, not evidence for checkpoint promotion or general intelligence.

The verified summary (archive reference: `../runs/capacity-study-local/audit/summary.json`) authenticated all 71 frozen sources, 127 input files, all nine calibration jobs, three main endpoints, three audits, saved optimizer states, canonical streams, and scoring denominators. The dispatcher completed all 16 processes with exit code zero at 2026-09-17 02:08:42 UTC. The summary ran once, completed with exit code zero, and performed no neural inference or training; its capture receipt (archive reference: `../runs/capacity-summary-validation-20260917T0209133944802Z/receipt.json`) records 222.65 seconds. Summary SHA256: `9a63ef7b836a2b2c1cba858cd66d030348081e933b19763bb594fdf9d14da8cb`.

## Comparison and selection

All models used four layers, four attention heads, feedforward width four times model width, the same byte inputs, objectives, three domains, fresh realization streams, and 7,200 main updates. Widths 96/192/256 have 753,610/2,221,738/3,692,010 parameters. Each received the same three-rate development calibration; all selected learning rate 0.001. Input episodes, targets, and byte exposures match across main arms. FLOPs, initialization shapes, optimization geometry, and elapsed costs do not.

The next pilot's prospective rule (archive reference: `../runs/foundation-planning-local/prospective.json`) was recorded at 01:50:09 UTC, before the capacity pipeline completed and before capacity scores were read for that decision. It ranks final development checkpoints lexicographically by: worst domain/length cell's mean final paired action/reply score across the four panels; overall mean paired score; known-answer accuracy; negative unsupported-ASK rate. Components are rounded to 12 decimals; an exact tie prefers smaller width. The chosen width keeps its independently calibrated rate.

| Width | Worst cell | Mean paired action/reply | Known accuracy | Unsupported ASK |
|---|---:|---:|---:|---:|
| 96 | 0% | 3.26% | 60.04% | 1.41% |
| 192 | 0% | 7.29% | 58.34% | 1.44% |
| 256 | 0% | 6.71% | 57.03% | 4.61% |

The recorded choice (archive reference: `../runs/foundation-planning-local/capacity-choice.json`) is width **192**, learning rate **0.001**. Every worst-cell score is zero. This chooses a common fresh configuration for the next experiment; no trained capacity checkpoint is promoted or reused, and audit results are not selection inputs.

## Sealed audit

Each width was scored on 36 identical cells: four panels × three domains × three lengths. Every cell has 64 opposite final-answer pairs. Paired success requires both members correct; generated replies are free-running and scored separately from actions. Known accuracy and ASK rates below pool their actual question denominators; paired percentages also equal the cell macro because all pair denominators match. Unsupported ASK means predicting unknown on a known-answer question.

| Width | Domain | Action pairs / 768 | Reply pairs / 768 | Known accuracy | ASK recall | Unsupported ASK |
|---|---|---:|---:|---:|---:|---:|
| 96 | color | 76 (9.90%) | 4 (0.52%) | 57.50% | 92.28% | 2.96% |
| 96 | count | 2 (0.26%) | 0 | 60.48% | 94.18% | 1.04% |
| 96 | switch | 26 (3.39%) | 27 (3.52%) | 59.83% | 91.86% | 1.19% |
| 192 | color | 70 (9.11%) | 102 (13.28%) | 58.61% | 93.06% | 1.63% |
| 192 | count | 1 (0.13%) | 1 (0.13%) | 58.92% | 95.09% | 1.68% |
| 192 | switch | 138 (17.97%) | 49 (6.38%) | 57.06% | 92.10% | 1.31% |
| 256 | color | 81 (10.55%) | 68 (8.85%) | 54.26% | 90.24% | 6.94% |
| 256 | count | 1 (0.13%) | 1 (0.13%) | 56.63% | 91.23% | 4.68% |
| 256 | switch | 62 (8.07%) | 84 (10.94%) | 57.46% | 88.10% | 3.44% |

Overall action/reply pair success was **4.51%/1.35%, 9.07%/6.60%, and 6.25%/6.64%**, respectively, out of 2,304 pairs per width. Each audit has 15,914 questions: 9,800 known and 6,114 unknown. Equal-cell query accuracy was 72.25%/71.86%/69.24%, while known accuracy was only 59.36%/58.27%/56.21%. Good recognition of earlier unknown questions does not establish success on final dependencies; final questions are always known.

All widths scored **zero action and zero reply pairs out of 192** on held composed count ancestry, including zero at each length. Width 192's composed switch action score was 35/192, versus 8/192 for width 96, but its composed color action score was 15/192 versus 21/192. Its color actions also fall from 46/256 at eight turns to 8/256 at twelve turns across panels, while color reply pairs rise from 14/256 to 50/256. Actions, replies, domains, and lengths therefore cannot be collapsed into a uniform capacity improvement.

Blank-text and reset-history controls produced zero final action/reply pairs and zero known-answer accuracy. All three CPU restart checks were exact. These establish useful input/history dependence and engineering consistency, not correct reusable algorithms.

## Fitting, progress, and cost

Both fit probes contain 4,608 episodes; all 2,304 recipes and initial realizations were encountered. Initial and latest-observed fitting probes are selected training examples, not the complete refreshed stream or unbiased generalization samples.

| Width | Initial action/reply pairs | Latest action/reply pairs |
|---|---:|---:|
| 96 | 5.03% / 1.17% | 5.03% / 1.65% |
| 192 | 10.94% / 7.81% | 10.68% / 8.20% |
| 256 | 6.47% / 5.99% | 7.81% / 6.12% |

Learning was not monotonic. At 600/1,800/3,600/7,200 updates, width 192's audit action pairs were 8.98/2.43/6.12/9.07%; reply pairs were 10.24/1.56/1.39/6.60%. Earlier checkpoints are descriptive curves, not retrospectively selected winners. The low fitting scores mean this result does not isolate generalization, representation, optimization, or lesson difficulty as the cause.

The complete study retained **27,000 updates and 2,592,000 episode exposures**, including 5,400 calibration updates. No discarded completed updates or failed steps were recorded. Main retained step times were 1,594/2,263/2,299 seconds; worker invocation times were 1,819/2,547/2,611 seconds. Generation/authentication accounted for 603/701/696 seconds **inside** those step intervals. Peak allocated CUDA memory was 1,488/1,917/2,208 MiB. Different concurrent workloads prevent interpreting those observations as isolated throughput scaling.

The dispatcher spanned 1 hour 50 minutes 9 seconds. Summed training invocation intervals were 8,878.55 seconds and audit workers 211.58 seconds; overlapping worker sums are not elapsed time or dedicated GPU hours. Setup, development, commits, and generation are nested costs and must not be added again. Preparation took 114.78 seconds; the earlier strict-profile engineering probes used 192 updates/18,432 episodes outside the formal totals. The independent summary's CPU time is also separate.

## Next bounded experiment and limits

The predeclared next comparison uses two fresh width-192 learners for **1,536 updates each**, with exactly the same immutable lesson multiset and a common mixed tail. It compares ascending primitive-to-composed practice with a shuffled prefix, retaining earlier depths and testing fresh realizations and withheld supervised ancestry. It is a comparison of lesson order within foundation practice, not a composition-only control or a new capability claim. Its bank/source freeze and strict execution proof are separate from this capacity study.

This capacity result has one initialization per width, no statistical replication, a finite grammar and previously inspected ancestry partitions, and no test of unseen semantic algorithms, open-ended English, welfare, or self-directed general learning. Strict repeatability evidence is workload/runtime-specific. Neither additional width nor training time alone established the learning foundation; the next pilot should measure primitive acquisition, retention, and composed transfer before attributing failure to any one mechanism.
