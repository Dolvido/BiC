# Measuring home computation

## Current local evidence, September 16, 2026

The actual local RTX 5080 with 16,303 MiB of reported VRAM is now measured.
Completed English and general-learning studies used its CUDA optimizer; those
records supersede the earlier absence of desktop measurements, without changing
the historical results below. The current capacity experiment spans about
0.75 to 3.69 million parameters; a separate 1.31M-parameter integration prototype remains
untrained. They have not established the machine's learning or capacity limit.

The new [shared-capacity comparison](CAPACITY_STUDY_PROTOCOL.md) is running with
at most two GPU workers. Its fixed budget is 27,000 updates and 2,592,000 episode
exposures, including all nine calibration jobs and three main jobs. All 71 sources
were frozen after a 668-test pass. Preparation took 114.782 seconds, including
bank authentication and pure calibration-stream reconstruction; that is outside
training. Costs and learning outcomes remain pending in the
execution ledger (archive reference: `../runs/capacity-study-local/execution.json`).

Separate strict-profile validation passed at every width using 192 physical
updates and 18,432 episodes. The largest observed probe allocation was
1,757.89 MiB; reservation, other processes and driver memory are additional.
These short repeatability checks do not establish sustained throughput or
learning efficiency. [Profile evidence](RAW_ENGLISH_LOOP_DESIGN.md#subsequent-strict-profile-execution-checks)

The preceding [fixed-versus-fresh comparison](REALIZATION_STUDY_RESULTS.md)
completed 7,200 updates and 691,200 episode exposures, totaling 224,532,944
observation tokens / 210,747,344 UTF-8 bytes. Fixed and fresh worker step intervals
were 727.874 and 833.513 seconds; generation/authentication accounted for 213.840
and 315.869 seconds **within** those intervals. Their summed 1,561.387 seconds
must not have generation added a second time. Audit worker intervals sum to
83.562 seconds. Allocator peaks were 1,384.35 and 1,439.18 MiB, excluding other
processes and driver memory.

The observed first-training-launch to final-audit-completion envelope was
24 minutes 29 seconds, including stream reconstruction, orchestration gaps and
polling. Both workers completed without interruption. These overlapping worker
intervals and observed envelopes are not dedicated GPU-hours. A separate
execution probe used two updates/192 episodes and discarded its weights.
Execution record (archive reference: `../runs/realization-study-local/execution.json`)

Fresh training consumed 345,600 distinct transcripts versus 4,608 originals
repeated in the fixed arm. It retried 29 whole-pair collisions and admitted no
duplicates. More data did not establish better learning efficiency: aggregate
query accuracy improved while every domain/panel final-pair macro worsened.
All 629 tests passed and both CPU restarts were exact; no checkpoint was promoted.
The width96/192/256 comparison above uses 753,610 / 2,221,738 / 3,692,010
parameters and fresh protected banks. Its runtime settings, seeds and budgets
differ from this earlier study, so their scores are not a matched width contrast.

The separate [budgeted backend](RAW_ENGLISH_LOOP_DESIGN.md) has 17 focused CPU
tests for committed-state resume, rollback and progress boundaries. Save time
counts against its cooperative wall budget, but an active step plus current
snapshot/save can overrun a deadline. Its transcript digest index grows with
training, so neither fixed lifetime checkpoint size nor indefinite bounded
storage is established. No new autonomous-training benefit is claimed.

Separate backend GPU validation consumed 96 updates/9,216 episodes: an
uninterrupted 32-update run, 16+save/load+16, and one additional uninterrupted
32-update control. The first comparison used 11.924 synchronized step seconds;
the extra control used 6.501 seconds. Allocator peaks were at most 1,111.25 MiB.
Saved-state reload and curriculum continuity were exact, but final weights and
AdamW moments varied both with and without reload. This is a numerical
repeatability limit under the recorded defaults, not an established checkpoint
data-loss defect. These execution probes are outside the main study and provide
no capability result. [Probe details](RAW_ENGLISH_LOOP_DESIGN.md#cuda-checkpoint-recovery-and-numerical-repeatability)

The preceding [shared-composition comparison](COMPOSITION_STUDY_RESULTS.md) retained
13,200 optimizer updates, 1,267,200 repeated episode exposures and 393,019,622
observed UTF-8 bytes across three main runs and two fresh controls. The schedules
matched per-family draws, while fresh controls matched only their trained family.
Durable worker optimizer intervals sum to 2,345.312 seconds; audit worker intervals
sum to 37.829 seconds. These overlapping intervals are not elapsed wall time or
dedicated GPU-hours. Final-invocation allocator peaks ranged from 764.11 to
1,431.59 MiB, excluding other processes and driver memory.

Five simultaneous workers raised combined reported GPU use to 15,865/16,303 MiB.
Both fresh controls were deliberately paused without an out-of-memory failure,
then resumed from saved update 64 and finished their fixed 1,200-update endpoints.
Discarded uncheckpointed work is unknown: 0–192 updates and 0–18,432 episodes per
control, excluded from the retained totals and durable timers. A disposable probe
used another two updates/192 episodes and discarded its weights. Invocation times
and allocator peaks cover only final invocations after resume. The later read-only
training-fit diagnostic took 15.25 seconds, including 5.766 seconds of preparation,
and performed no optimization. Execution record (archive reference: `../runs/composition-study-local/execution.json`)

That study finished with all 578 tests passing. Neither schedule acquired reliable
composition; exact-bank fitting and fresh-realization generalization were distinct
limitations. It motivated the now-completed realization comparison above, with byte,
target-mixture and runtime differences reported. This evidence
does not establish useful general learning efficiency or the machine's frontier.

The preceding [continued fitting and replay study](CONSOLIDATION_STUDY_RESULTS.md)
completed 23,904 additional optimizer updates and 1,562,624 repeated episode
exposures. Two continuations added 21,600 updates/1,382,400 old-subject exposures;
nine adaptations added 2,304 updates/147,456 new-subject exposures, with another
32,768 old replay exposures and their extra forward/backward computation.
The observed execution envelope was 666 seconds (11 minutes 6 seconds), including
orchestration and completion-observation delays: 406 continuation and 260 audit
seconds. Concurrent continuation training intervals sum to 659.627 seconds;
synchronized adaptation training adds 60.391 seconds. These are not dedicated
GPU-hours. Internal continuation invocation timers exclude final scoring and
writing; the external execution record (archive reference: `../runs/consolidation-study-local/execution.json`)
includes those phases. Continuation peak allocated CUDA memory was
838.27–841.79 MiB. All jobs finished and the final full suite passed 525 tests.
Replay improved retention at matched new
exposure and extra compute; unreliable paired acquisition prevents interpreting
that as broad learning efficiency or a completed hardware frontier.

The preceding [curriculum diversity comparison](DIVERSITY_STUDY_RESULTS.md)
completed four concurrent main runs and five subsequent adaptations: 14,720
updates and 942,080 repeated episode exposures. The observed execution envelope
was 371 seconds (6 minutes 11 seconds), including completion polling: 267 main
and 104 audit seconds. Main training intervals summed to 819.063 seconds;
adaptation training added 6.388 seconds. These overlapping intervals are not
dedicated GPU-hours. Main peak allocated CUDA memory was 840.10–843.49 MiB,
including cached banks and scoring but excluding driver/other process memory.
The execution record (archive reference: `../runs/diversity-study-local/execution.json`) preserves
the timing scope. All jobs finished, its full suite passed 494 tests, and no checkpoint was
promoted. Naming generalization improved, but integrated new-subject learning
did not improve over fresh initialization and retention deteriorated. Fast
execution therefore remains distinct from useful learning efficiency.

The isolated decision-learning benchmark (archive reference: `../runs/general-credit-local/benchmark.json`)
used 796,552 parameters, batch 64, three warmup updates and twelve measured
updates. The zero/0.3 auxiliary-loss arms measured 6.51/6.45 updates per second
and 207.68 MiB peak CUDA allocated memory each. These short probes exclude
data preparation, evaluation and checkpoint writing; allocator peaks exclude
driver and other process memory. They are throughput measurements on temporary
fresh learners, not capability evidence.

The completed [decision-credit comparison](COGNITIVE_CREDIT_RESULTS.md) ran
3,600 main updates and 230,400 repeated episode exposures. Its two concurrent
workers summed to 644.346 training seconds; the longer worker interval was
331.188 seconds. Subsequent graph adaptation added 48 updates, 3,072 repeated
support exposures and 8.329 training seconds. Summed worker intervals are not
dedicated GPU-hours. The extra teaching loss failed its benefit screen and
neither trained learner acquired graphs better than fresh initialization.

The newer sequence execution probe (archive reference: `../runs/sequence-baseline-local/benchmark-initialized.json`)
used sequential regional/sequence jobs, batch 64, three warmups and twelve FP32
updates. The regional control (796,552 parameters) measured 8.44 updates/second
and 205.54 MiB peak allocated CUDA memory. The causal sequence model (753,610)
measured 64.17 updates/second and 578.44 MiB. This approximately 7.60-fold
throughput difference is limited to this short workload; it is not measured
learning efficiency. The probe used binding, arithmetic and conditional worlds,
excluded evaluation/checkpoint time and ran no capability audit. The
earlier probe (archive reference: `../runs/sequence-baseline-local/benchmark.json`) predates the
sequence initialization correction and remains historical evidence. A new
[calibrated learning comparison](SEQUENCE_STUDY_RESULTS.md) has now completed
all nine calibration runs, three fresh main runs and six audit adaptations:
16,584 updates and 1,061,376 repeated episode exposures. This is a separate
workload from the short sequential probe. Sources and recipes were frozen at
2026-09-16 20:24:15 UTC. The observed first-launch-to-audit-completion envelope
was 23 minutes 58 seconds, including phase gaps and polling delay. Concurrent
training intervals sum to 773.187 seconds for calibration and 1,549.785 for main
training; sequential audit adaptations add 36.999 training seconds. These sums
are not dedicated GPU-hours. Main worker intervals exclude construction/setup,
and memory peaks include differing evaluation batch sizes; consult the
execution record (archive reference: `../runs/sequence-study-local/execution.json`) and results for
scope. All jobs for that study finished. Its validation passed 453 tests, including
the untrained integration adapter; no checkpoint was promoted.

The preceding [general-learning comparison](GENERAL_LEARNING_RESULTS.md)
completed four concurrent workers and 7,200 main updates on this computer.
It did not find a broadly better learner, so increasing parameter count or
runtime alone is not yet supported as the solution. The
[home-learning roadmap](HOME_LEARNING_ROADMAP.md) keeps representation quality,
new-subject learning, retention, practical usability and measured resource
scaling as separate requirements of the ongoing objective.

## Earlier profiles and their hardware

The v0.5 trained regional-memory agent contains **558,467 parameters**, of which
181,472 belong to its frozen visual encoder. On the recorded CPU host, 3,000
updates plus development checks took **203.0 seconds with rehearsal** and
**109.0 seconds without rehearsal**, with another 10.7 seconds for shared data
preparation. Each update used 24 new examples; the rehearsal arm additionally
used 16 old examples. These are single-thread CPU measurements, not GPU timings,
and process-memory peaks were not separately measured for this experiment.
Training record (archive reference: `../runs/regional-memory-v05/training_report.json`)

The larger profiles below are the earlier v0.4 computation experiment; their
size and memory use must not be attributed to the trained v0.5 checkpoint.

The released tools measure actual optimizer updates at different regional widths. The largest completed profile has **175,735,747 parameters**. This establishes execution of a short synthetic workload on the recorded host; it is neither a trained language model nor the maximum model a home computer can run.

These earlier measurements used Linux, an AMD EPYC 9V74 host, PyTorch 2.14.0+cpu, float32, and one CPU thread. The Ryzen 9 9900X / RTX 5080 desktop was **not measured in that older run**; newer actual desktop evidence is listed above.

| Profile | Parameters | Batch | Mean update time | Process peak RAM, GiB |
|---|---:|---:|---:|---:|
| small | 376,707 | 8 | 0.025 s | 0.357 |
| medium | 1,369,027 | 8 | 0.048 s | 0.375 |
| large | 3,619,779 | 8 | 0.067 s | 0.417 |
| home | 13,941,699 | 8 | 0.201 s | 0.601 |
| stretch | 44,407,747 | 8 | 0.841 s | 1.127 |
| extended | 175,735,747 | 1 | 3.170 s | 3.284 |

The first five profiles measure three updates after two warmups; extended measures two after one warmup. Every profile uses a fresh process. Inputs are three 32 × 32 images, a short prompt, and 13 decoder steps. Timing includes forward/backward passes, loss calculation, clipping, and AdamW. It excludes dataset generation, rendering, evaluation, and checkpoint writing. These short, potentially contended samples cannot predict a full training run precisely. Scale measurements (archive reference: `../runs/compute/cpu_scale.json`), extended measurement (archive reference: `../runs/compute/cpu_extended.json`)

RAM is process-lifetime peak resident memory, including Python/PyTorch. Analytical weights/gradients/Adam estimates exclude activations and temporary buffers. GPU allocator peaks, when measured, describe a separate memory pool. No mixed precision or quantized training was used.

## Measure your desktop

Use the project's Python environment. For GPU execution, select a compatible build through the [official PyTorch installer](https://pytorch.org/get-started/locally/), then verify detection:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
python -m brain_in_computer benchmark --device cuda --profiles small medium large home --batch-size 8 --measured-steps 20 --output runs/compute/my-gpu.json
```

Once measured, examine `cuda_peak_allocated_bytes`, `cuda_peak_reserved_bytes`, step times, and failure statuses before increasing the workload:

```bash
python -m brain_in_computer benchmark --device cuda --profiles stretch extended --batch-size 1 --max-parameters 200000000 --measured-steps 10 --output runs/compute/my-gpu-larger.json
```

Use `--device cpu --threads 1` for a CPU comparison. The default 50-million-parameter cap skips extended; raising it is explicit. A parameter cap limits model allocation, not process memory, and is not a fit guarantee. Each worker also has a configurable timeout. Save separate reports when varying batch size, history, or hardware.

For an actual larger *learning experiment*, the general computer trainer accepts dimensions:

```bash
python -m brain_in_computer computer-train --hidden-size 96 --language-hidden-size 192 --embedding-size 96 --visual-features 96 --batch-size 32 --steps 5000 --device cuda --output runs/medium-learning
```

This starts new weights on the older general curriculum; it does not reproduce the v0.4 counterfactual training driver. With `--initialize-from`, configuration comes from the checkpoint. Compare held-out competence and retention per unit of computation before expanding again. Current evidence supports further measured experiments, not an upper bound on intelligence or a claim that home computation is exhausted.
