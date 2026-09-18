# v0.1 measured results

These are measurements from the supplied code and checkpoints, not projected capabilities. This is one training seed on deliberately small synthetic tasks. It establishes a working learner and evaluation apparatus; it does not establish general intelligence or validate the brain analogy.

## Reference training

- Model: 13 independently parameterized regional networks, 162,810 parameters, hidden width 48.
- Environment: Linux CPU, Python 3.12.14, PyTorch 2.14.0+cpu, one Torch intra-op thread.
- Training: 600 AdamW updates, batch size 32, learning rate 0.002, training delay 2.
- Seeds: model 7, training stream 8, replay stream 9, reference evaluation 100007.
- Memory: eight within-episode memory slots; 256 stored training episodes; 25% rehearsal after the first batch.
- Work: 19,200 training episode presentations, including 14,408 fresh episodes and 4,792 replay samples.
- Training wall time: **11.87 seconds**, including periodic checkpoint writes. Initialization and final evaluation are outside this timer. This is the execution environment's measurement, not a benchmark of the intended Windows home computer.

Every row below uses 512 independently generated test episodes, with four equiprobable response classes and 25% random-choice accuracy. Initial and final evaluations use the same fixed evaluation stream; it is separate from training and is not rehearsed. This run used a fixed 600-update budget without selecting a checkpoint based on test scores.

| Task | Untrained | Trained, 2 distractors | Trained, 6 distractors |
|---|---:|---:|---:|
| Visual symbol match | 25.39% | 100% | 100% |
| Auditory symbol match | 27.93% | 100% | 100% |
| Instruction-based channel selection | 24.41% | 100% | 100% |
| Delayed first-symbol recall | 26.76% | 100% | 100% |
| Cross-modal modulo-four rule | 22.85% | 100% | 100% |

The generators can repeat simple symbolic combinations encountered during training. These scores do not measure novel natural tasks, natural language, rich perception, or systematic compositional transfer. Observing zero mistakes on 512 samples is not proof of zero underlying error. Reported JSON standard errors are plug-in sampling estimates and become zero at perfect observed accuracy; they are not confidence bounds or training-seed uncertainty estimates.

Source results: `runs/reference/report.json`, `initial_eval.json`, `final_eval.json`, `longer_delay_eval.json`, and `checkpoint.pt`.

## Matched recurrent baseline

The comparator is one GRU with the same inputs and five output heads. Its width was selected to match total parameters: **163,047**, 0.146% above the modular model. It used the same 600 updates, objectives, optimizer settings, batch composition, data seed, and replay policy. The script checked that final task-generator and replay RNG states matched.

| Model | Mean task accuracy, delay 2 | Mean task accuracy, delay 6 | Training time |
|---|---:|---:|---:|
| Regional cluster | 100% | 100% | 11.87 s |
| Monolithic GRU | 96.99% | 97.27% | 3.71 s |

The difference is entirely the cross-modal rule: GRU accuracy was 84.96% at delay 2 and 86.33% at delay 6; the other four tasks reached 100%. The modular model took approximately 3.2 times as long in these separately measured runs. Its timing includes periodic serialization, while the baseline training timer excludes serialization. This is an indicative accuracy/compute tradeoff, not a controlled throughput claim.

One optimizer setting may favor one architecture; equal parameter/update counts are not equal compute or memory capacity. Multiple initialization seeds, tuned controls, harder splits, and equal-compute comparisons are required before claiming architectural superiority.

Source: `experiments/baseline_comparison.json`; reproducible script `experiments/compare_baseline.py`. `runs/baseline/gru_weights.pt` is an inference state dictionary, not a resumable `Trainer` checkpoint.

## Region lesions and memory stress

A separate diagnostic evaluation uses seed 300007 and 256 episodes per task. It applies a lesion for the entire sequence and performs no retraining. Intact accuracy was 100%.

| Region disabled | Mean accuracy across five tasks |
|---|---:|
| Visual cortex | 48.05% |
| Auditory cortex | 63.98% |
| Somatosensory cortex | 98.98% |
| Temporal/language | 44.06% |
| Parietal association | 94.14% |
| Thalamus | 38.36% |
| Prefrontal cortex | 75.08% |
| Hippocampus | 100% |
| Basal ganglia | 97.89% |
| Motor cortex | 25.08% |
| Cerebellum | 100% |
| Amygdala | 88.44% |
| Hypothalamus | 98.52% |

**Interpretation:** the model is sensitive to several regional pathways, but this curriculum does not establish a performance benefit from hippocampus or cerebellum. Their removal can change logits without changing the winning response. Working memory can solve delayed recall, so success on that task is not proof of episodic-memory use. Other regions may also be redundant or compensate for missing signals.

Sensory lesions do not remove all sensory information: raw prediction errors enter the cerebellum through an explicitly documented external pathway. Lesion performance is therefore not equivalent to blindness or deafness. Nor does it establish that a region is necessary after retraining.

In the same stress test, delayed recall scored 100% with 2, 6, and 14 distractors. The 14-distractor case exceeds the eight-slot hippocampal bank; recurrent state can retain information after a memory vector is evicted. Longer horizons, association interference, and novel rules remain untested.

The diagnostic process reached approximately **342 MiB peak resident memory**, including Python/PyTorch, loaded optimizer state, and replay. This is an observed diagnostic peak, not a training-memory bound. The trained checkpoint is about 2.7 MiB. Wider models, much longer differentiable sequences, or raw image/audio replay can require substantially more memory.

Source: `experiments/diagnostics.json`; script `experiments/run_diagnostics.py`.

## Operational verification

All **24 automated tests passed**. The suite checks independent parameter ownership, finite gradient flow into every region, output contracts, ablation disconnection, causal feedback timing, absence of future-input influence, isolation between episodes and batch members, bounded differentiable hippocampal memory, task-label correctness, input validation, replay ownership/capacity, and exact CPU checkpoint continuation.

The resume test verifies the next optimizer update's metrics, weights, optimizer tensors, next training examples, and rehearsal state against uninterrupted execution. Evaluation is tested to preserve weights, gradients, training mode, and RNG state. The one-observation loss case is covered so an absent transition cannot produce a NaN prediction loss.

Windows/CUDA execution, naturalistic tasks, continual-learning benefit, independent cross-episode episodic retrieval, and general intelligence have not been demonstrated. The next experiments are defined in `RESEARCH_PLAN.md`.
