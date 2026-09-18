# Architecture and extension contract

This document describes the symbolic core retained from v0.1. V0.2's optional language interface and its untrained status are detailed in `LANGUAGE_ARCHITECTURE.md`.

## Computational partition

`Brain.regions` is a PyTorch `ModuleDict` of 13 neural networks. Every trainable parameter belongs to one region. The default model uses 48-dimensional regional messages, four response classes, and eight hippocampal memory slots. The model is a single-process neural cluster, not a distributed service deployment.

The region classes are in `regions.py`; the explicit execution schedule and directed pathways are in `model.py`. `CONNECTOME` records `(source, destination, observation_step_delay)`. A delay of zero denotes an earlier computation within the current observation step; one denotes the previous observation step. This is discrete algorithmic scheduling, not physiological conduction time. The graph has 41 inter-region pathways. External inputs are separately declared by `EXTERNAL_INPUTS`; `inspect` reports both inventories.

Each region computes its own learned transformation. At a high level:

\[
h_t^r = f_{\theta_r}(x_t^r,\{h_t^q\}_{q\in I_r},\{h_{t-1}^q\}_{q\in D_r},s_{t-1}^r).
\]

The independent parameter sets are `theta_r`. Current-step inputs `I_r` and delayed inputs `D_r` are deliberately selected, not an all-to-all network. Some regional computations concatenate messages, while thalamic and hippocampal networks use learned attention. Global backpropagation assigns credit through these pathways; region-specific local learning rules are not implemented.

## One observation step

1. Visual and auditory encoders process their own signals with previous prefrontal feedback. The body encoder includes previous motor activity. The instruction encoder includes previous prefrontal context.
2. Parietal association fuses these encodings. Amygdala and hypothalamus compute modulation signals. Thalamus uses learned content attention to route five candidate messages.
3. Hippocampus creates a gated memory vector, appends it to the bounded episode memory, and attends over the retained vectors. The current observation can therefore be read immediately. Prefrontal GRU state integrates routed information, recall, instruction, and modulation, plus the previous cerebellar state.
4. Basal ganglia produces selection features, action gates, and an unconstrained scalar confidence estimate. Motor cortex produces response logits from prefrontal state, thalamic routing, selection, and previous cerebellar activity.
5. Cerebellum predicts the next raw observation from current prefrontal/motor/body representations and the current raw observation minus its previous prediction. Its state feeds prefrontal and motor processing on the next step.

That last error input is a separate sensory pathway. Disabling visual cortex does not blind the complete system. A proper modality-deprivation experiment must zero the raw channel everywhere while preserving target labels. Inference lesions are interventions outside the training distribution; they establish sensitivity, not anatomical necessity or benefit compared with retrained alternatives.

## Observation and output contract

`Brain.forward(observations, ablate=(), return_activity=False)` expects a dictionary. Every floating tensor must match the model's device and floating dtype; all inputs must be finite. Tokens must be valid vocabulary indices.

V0.2 additionally accepts the keyword `language_context`, an optional `[batch, time, hidden_size]` floating tensor added to temporal-language activity. Omitting it or supplying zeros preserves the original core. The producer is responsible for ensuring that context contains no future information.

| Input | Default shape | Meaning in built-in tasks |
|---|---|---|
| `visual` | `[batch, time, 8]` | Four one-hot symbol values plus four noise values |
| `auditory` | `[batch, time, 8]` | Independent auditory-channel symbol and noise |
| `body` | `[batch, time, 4]` | Normalized time, start flag, query flag, constant |
| `tokens` | `[batch, time]`, int64 | Stable symbolic instruction |
| `feedback` | `[batch, time, 2]` | Reserved external feedback; zeros in v0.1 tasks |

| Output | Default shape | Training target |
|---|---|---|
| `logits` | `[batch, time, 4]` | Correct response, final position only |
| `visual_logits` | `[batch, time, 4]` | Present visual symbol |
| `auditory_logits` | `[batch, time, 4]` | Present auditory symbol |
| `prediction` | `[batch, time, 20]` | Concatenated next visual, auditory, body observation |
| `value` | `[batch, time]` | Final selected-response correctness, detached target |
| `region_activity` | Dictionary of `[batch, time, 48]` | Optional diagnostic values |

There is no softmax on response logits; the loss applies cross-entropy. `value` is unconstrained and is not a calibrated probability. Future sensory vocabulary sizes may need separate recognition-head dimensions rather than the present shared four-action dimension.

`Batch` keeps labels and task IDs separate from observations. Earlier action targets and absent sensory targets are `-100`, ignored by the corresponding loss. Built-in sequences have `time = delay + 2`. The model accepts shorter sequences, and next-observation loss is zero when no transition is available.

## State and learning

Each forward call initializes recurrent state and hippocampal memory to zero/empty. Individual batch rows do not share state. The public API currently handles complete episodes; it does not provide a persistent streaming `step()` session. Longer external interactions should first be represented as explicit episodes; a future streaming API must expose reset and state ownership.

The hippocampal memory bank is a bounded FIFO of learned vectors, with differentiable writes and reads within the episode. The first cue may be evicted beyond its capacity. Other recurrent pathways can still remember it, so success on a long sequence does not by itself prove hippocampal retrieval.

The trainer minimizes:

\[
L=L_{response}+0.15(L_{visual}+L_{auditory})+0.03L_{next-observation}+0.05L_{confidence}.
\]

AdamW updates all regions; gradient norm is clipped to 1.0. Defaults are learning rate 0.002, weight decay 0.0001, batch size 32, and seed 7. The default curriculum uniformly draws five tasks. After the first batch, up to one quarter of each batch is sampled from the bounded replay reservoir; the remaining examples are fresh. Reservoir replacement approximates uniform coverage of seen training episodes. Test examples never enter it.

Replay retains CPU copies of complete training episodes and their targets. It is capped by episode count and fixed observation dimensions/delay within a run, not by an arbitrary user-supplied byte limit. The default size is small. Scaling image/audio inputs requires explicit storage budgeting or compression.

Checkpoints contain versioned config, weights, AdamW state, completed-step count, rehearsal reservoir, training-stream RNG, replay RNG, global Torch RNG, optional CUDA RNG, and logged metrics. CPU continuation was tested for an exact next update. Bitwise equality across different operating systems, Torch versions, GPU architectures, or device switches is not promised.

## Extension points

- **Tasks:** extend `tasks.py` or supply `Batch` objects with the same contract. Define training/evaluation splits and baseline scores before training. Changing the built-in generator changes the experiment and must be recorded.
- **Regions:** add a parameter-owning module, invoke it explicitly in `Brain.forward`, declare its pathways/external inputs, define its state lifetime, and support outgoing-value ablation. Verify gradients and causal sensitivity.
- **Sensors:** resize `BrainConfig` inputs and replace the relevant encoders. The current built-in task generator and recognition losses assume four symbolic classes; a new modality requires appropriate data and objectives.
- **Learning:** add a loss in `learning_loss` only when its target is available without future-information leakage. Reward-driven learning needs an environment interaction loop and return/advantage estimates; the current confidence head is insufficient.
- **Persistent retrieval:** introduce explicit write/read policies, stable keys, bounded storage, reset semantics, and a cue-based evaluation independent of PFC recurrence. Do not treat the trainer's labeled replay buffer as inference-time memory.
- **Planning:** add action-conditioned transition, reward, and termination models plus candidate action rollouts. The current predictor does not perform counterfactual search.

See `RESEARCH_PLAN.md` for proposed measurable advancement gates and primary research references.

## V0.7 continuation

`Brain.forward` still starts independent sequences with fresh state for backward
compatibility. `Brain.forward_with_state(observations, state=None, ...)` returns
both outputs and a caller-owned `BrainState`, allowing later chunks to continue
the same recurrent computation. State carries prefrontal, motor, cerebellar,
prediction, and bounded hippocampal tensors. It is not a module global and adds
no learned parameters or checkpoint keys. `state.detached()` explicitly cuts
training history; tensor-only export/import supports saved activity.

The optional `adaptive_agent` and `adaptation` modules use this API for a
four-category feedback task. This is a fresh small model, separate from the
desktop/language checkpoint. The policy is trained through causal scripted
strategy imitation and then evaluated with fixed weights. See
[ADAPTATION_GUIDE.md](ADAPTATION_GUIDE.md) and
[ADAPTATION_PROTOCOL.md](ADAPTATION_PROTOCOL.md).

The architecture description above records the original core design; its
independent-call reset behavior remains the default API, not a restriction of
the new explicit-state API.
