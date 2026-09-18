# Sequence baseline — locally evaluated, broad capability unproven

The [decision-credit study](COGNITIVE_CREDIT_RESULTS.md) failed its broad-benefit screen. The subsequent comparison tested whether a stronger shared sequence representation improves learning before extending the same failed objective. The capacity-control model, observation-only data packer, matched objective/trainer and teacher-free evaluator are implemented and tested. Short local optimizer probes establish execution; the completed learning study supplies the separate capability evidence and its limits. Its [prospective protocol](SEQUENCE_STUDY_PROTOCOL.md) specified three architectures, equal learning-rate calibration budgets, fresh main runs and a withheld arithmetic subject.

**Current status:** the [complete comparison and audit](SEQUENCE_STUDY_RESULTS.md) finished all 16,584 planned calibration, main and adaptation updates. All architectures selected learning rate 0.001. Sequence improved later conditional decisions and lost less mean retained accuracy, while recurrent BiC acquired arithmetic more strongly; neither established consistently broad competence. No checkpoint was promoted, and the protocol defined no automatic numerical promotion screen. The validation record (archive reference: `../runs/sequence-study-local/validation.json`) now records 453 passing tests, including the separate untrained regional adapter. All jobs for this study have finished.

## One reusable sequence learner

Encode the six observed English utterances as one sequence of UTF-8 byte IDs, retaining explicit BOS/EOS boundaries. A causal Transformer can compute all six utterance representations in one training pass: each utterance's EOS state can attend to its current and earlier observations, never later ones. This supplies a learned sequence representation with shared attention, residual updates and shared feedforward layers across every family.

Use the same four-way response head for all families, the same 128-unit byte reply GRU as the existing learner, and a tied byte-embedding output head for observation prediction. The four-way head is a common response interface, not separate domain classifiers. Give neither encoder nor state family IDs, parsed facts, rule labels, oracle outputs, counterfactual-group IDs or external LLM features.

Keep the existing objective: query-class-balanced action cross entropy, 0.25 acknowledgement cross entropy, 0.1 reply-byte cross entropy and 0.1 observation-next-byte cross entropy. Sample counterfactual pairs using the same independent family streams. All learned weights start fresh.

**Answer isolation:** concatenate only observed utterance bytes. Earlier correct replies must not enter later context. Extract action logits at each utterance EOS before response decoding. Teacher-forced reply prefixes enter the separate reply GRU only; generated replies also stay outside the observation history. Score greedy replies in addition to action choices. Mask observation prediction at utterance boundaries and padding; an input-only sequence must never accidentally include response targets.

## Capacity options measured on the CPU

These exact parameter counts were measured by instantiating the specified PyTorch modules, without a forward or training pass. Every option has a 259-entry byte embedding, 1,024 learned position embeddings, pre-normalized `TransformerEncoderLayer` blocks with biases, ReLU feedforward layers and zero dropout, final LayerNorm, tied observation-byte readout, one four-way action head, a linear/LayerNorm/tanh bridge to the existing reply GRU, and that reply GRU. Shared embedding/output weights are counted once.

| Option | Width | Layers | Attention heads | Feedforward width | Total parameters |
|---|---:|---:|---:|---:|---:|
| Capacity control | 96 | 4 | 4 | 384 | 753,610 |
| Approximately 1M | 128 | 4 | 4 | 512 | 1,144,682 |
| Approximately 4M | 256 | 4 | 4 | 1,024 | 3,692,010 |
| Approximately 16M | 512 | 5 | 8 | 2,048 | 16,657,642 |

The reply GRU contributes 169,571 parameters to every option. The implemented capacity control verifies the **753,610** count. It has fewer parameters than CognitiveStudent's 796,292 or the credit-control model's 796,552. This makes it useful for testing whether a gain can be explained solely by adding parameters. The larger options remain module-inventory counts, not trained models or measured capacity limits.

Installed locally: PyTorch **2.11.0+cu128**, compiled for CUDA **12.8**, with `scaled_dot_product_attention` available. That does not establish which GPU attention kernel a particular mask/dtype selects.

## Implemented preparation and measured execution

The implementation is separated into [sequence_student.py](../experiments/sequence_student.py), [sequence_data.py](../experiments/sequence_data.py), [sequence_training.py](../experiments/sequence_training.py) and [sequence_evaluation.py](../experiments/sequence_evaluation.py). Packing retains only observed English with explicit delimiters; action labels, decoder prefixes and byte targets remain separate supervision. It rejects oversized inputs instead of truncating them. The model validates context boundaries and supports blank observations, separate sessions and serialized observation history. Training authenticates canonical lessons and preserves complete-pair sampling and optimizer state. Evaluation supports generated replies, opposite-answer pairs, later-known slices and blank/history-reset controls.

Before the formal study freeze, token and position embeddings were initialized with standard deviation 0.02 to avoid an oversized tied observation-readout loss. The zero-update CPU check (archive reference: `../runs/sequence-initialization-check.json`) records initial loss components on fresh binding, graph and conditional train worlds. This is initialization validation, not learning evidence.

The initialized local probe (archive reference: `../runs/sequence-baseline-local/benchmark-initialized.json`) measured sequential jobs on the RTX 5080: three warmups and twelve FP32 updates per model, batch 64, learning rate 0.001. Its subjects were **binding, arithmetic and conditional**, unlike the forthcoming study's binding, graph and conditional mixture.

| Probe model | Parameters | Updates/second | Episode exposures/second | Peak allocated CUDA memory |
|---|---:|---:|---:|---:|
| Regional credit control | 796,552 | 8.44 | 540.08 | 205.54 MiB |
| Causal sequence | 753,610 | 64.17 | 4,106.95 | 578.44 MiB |

The sequence probe processed 106,500 observation tokens in its measured updates. Changing future text left the tested earlier CUDA logits unchanged (maximum difference 0.0). Evaluation and checkpoint time are excluded; allocated memory is not total process/driver VRAM. The approximately 7.60-fold throughput difference is a short-workload result, **not learning efficiency or a general architecture speed claim**. The original probe (archive reference: `../runs/sequence-baseline-local/benchmark.json`), before the initialization correction, remains historical evidence. Neither probe includes a capability audit or a selected checkpoint.

## Distinguish diagnosis from regional integration

**Diagnostic monolithic baseline:** the causal encoder's EOS representation directly supplies the common action head and reply bridge. This deliberately bypasses BiC's thirteen-region routing. Success would demonstrate a more learnable mechanism on the verified curriculum; it would not establish that BiC's existing regional learner acquired that ability.

**Candidate regional integration:** replace the per-utterance GRU representation with the learned causal sequence representation, project evidence into the existing temporal-language and hippocampal interfaces, and retain the current motor action path and region-derived reply context. Carry both sequence context and BrainState explicitly. No direct encoder-to-answer route is allowed in this comparison. Initially use the same encoder width/depth as a successful diagnostic baseline and measure the complete parameter count.

This interface now has an **untrained 1,309,188-parameter implementation** in [regional_sequence_student.py](../experiments/regional_sequence_student.py), with ten focused CPU tests. It maps contextual token/EOS states into the existing 128-wide comprehension interface, keeps the regional action and reply paths, stores observation-only history with regional state, and rejects capacity overflow. It is outside the completed frozen study and has no learning or transfer result. Regional models in that study could express known conditional outcomes correctly in generated replies despite wrong actions, so an integration experiment must examine generic output alignment and credit assignment as well as representation quality; existing concepts must not be presumed absent.

If only the monolithic learner succeeds, regional integration remains an unsolved engineering constraint. Preserve that finding rather than claiming the baseline's performance as regional BiC performance. Conversely, architecture identity alone should not exclude a stronger, verified learning mechanism from the project's development path.

## Bounded, interpretable comparison

The [development diversity diagnostic](COGNITIVE_DIVERSITY_DIAGNOSTIC.md)
records the completed sequence endpoint's fixed training-fit versus development
gap and the current banks' naming/structural limits. Its proposed factorial
curriculum experiment is not part of the frozen architecture comparison and
uses no withheld audit evidence.

1. Start with the **753,610-parameter capacity control**, recurrent regional and episodic regional learners. The new protocol gives each architecture three learning rates and 600 updates per rate at seed 2401, then a fresh 3,600-update main run at seed 2501 using its selected rate. Freeze fresh training/support recipes, schedules, compute limits and an untouched audit before calibration. Previously inspected audits are known diagnostics.
2. Compare equal lesson exposures and optimizer updates, including a fixed validation-only learning-rate selection budget for each architecture. A single unsuitable inherited learning rate would not fairly test the new sequence representation. Report examples, input/reply tokens, updates, elapsed time and peak allocated GPU memory.
3. Also compare the common elapsed-time budget. Equal updates and equal time are separate comparisons; self-attention can process visible turns in parallel, while the regional model advances its state sequentially. Record the actual examples completed in each comparison.
4. Lead with opposite-premise paired correctness, unfamiliar combinations, a held-out family's learning curve against fresh initialization, and retention. Include text-blanked and history-reset controls. Confirm promising screens across at least three seeds and multiple family rotations; retain the research plan's minimum of five training seeds for capability claims. A training-loss reduction alone does not qualify.
5. If the small baseline learns broadly, test regional integration before scaling. If capacity remains plausible, advance to 1.14M, then 3.69M; attempt 16.66M only after a measured benefit and a local throughput/memory benchmark justify it. Do not run a broad architecture sweep.

Use an effective batch of 64, with gradient accumulation if needed. Six 128-byte utterances plus delimiters fit below the proposed 1,024-position limit. Start GPU microbatches conservatively and measure complete forward/backward updates before reserving a longer run. Mixed precision and efficient attention are options to benchmark, not assumed speedups. At 16.66M parameters, FP32 weights, gradients and two Adam moments alone are roughly 254 MiB; activations and attention workspaces will determine the practical limit on the 16 GB 5080.

Before training, test causal masking, padding, answer isolation, separate-session batching, prefix versus incremental inference agreement, checkpoint restoration and exact CPU optimizer/sampler resume. Serialize either observation history or validated attention caches together with configuration and a matching weight digest. Keep experimental checkpoints separate from accepted BiC checkpoints. The outcome must remain a statement about measured learning and transfer, not a declaration of general intelligence.
