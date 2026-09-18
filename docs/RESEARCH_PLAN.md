# Brain in Computer: research scope and advancement gates

V0.2 adds untrained language interfaces. Its feasibility assessment and teaching sequence are in `ENGLISH_FEASIBILITY.md`; implemented scope and verification are in `LANGUAGE_ARCHITECTURE.md` and `V02_STATUS.md`. The v0.1 capability results below remain limited to symbolic tasks.

The goal is a trainable, broadly capable system that can run on a home computer. Version 1 is an experiment toward that goal: a small Python/PyTorch network composed of 13 separately parameterized macro-region modules. It is not human-level general intelligence, a whole-brain reconstruction, or evidence that human-level learning will emerge from more training. No result on the supplied symbolic tasks establishes the requested ultimate outcome.

## What the biological inspiration means

The anatomical names specify intended engineering responsibilities. They do not establish anatomical fidelity or prove that a module learned its intended function. This first partition groups many structures together and omits others, including olfactory, gustatory, and detailed brainstem circuits. Extending it toward a selected anatomical atlas requires an explicit inventory and justification for each added subdivision.

The research motivation is testable. Recurrent Independent Mechanisms demonstrated useful specialization and generalization with recurrent modules and restricted attention-based communication on their benchmarks. This motivates comparing modular communication with a matched monolithic network; this implementation does not reproduce RIMs. [Goyal et al., *Recurrent Independent Mechanisms*](https://arxiv.org/abs/1909.10893).

The following are engineering analogies, not a neuroscience reference atlas:

| Region module | Intended analogy and v1 mechanism | Missing capability or evidence |
|---|---|---|
| Visual | Encodes symbolic visual-channel observations | Images, object recognition, visual hierarchy |
| Auditory | Encodes symbolic auditory-channel observations | Waveforms, acoustic features, speech |
| Somatosensory | Encodes body-state inputs | Physical touch and learned proprioception |
| Temporal/language | Encodes instruction/context symbols | Natural-language understanding or generation |
| Parietal | Combines modality representations | Learned geometry and spatial reference frames |
| Thalamus | Learns gating/routing of messages | Detailed nuclei, anatomical connectivity, timing |
| Prefrontal | Maintains recurrent task/working state | Deliberate multi-step planning |
| Hippocampus | Learns attention over within-episode memory | Durable, autonomous cross-episode retrieval |
| Basal ganglia | Contributes learned selection/gating | Validated reinforcement-learning action selection |
| Motor | Produces discrete response outputs | Embodied motor skills and feedback control |
| Cerebellum | Supplies a learned predictive auxiliary | Action-conditioned forward control model |
| Amygdala | Supplies learned salience-related modulation | Validated affective learning; emotional experience |
| Hypothalamus | Supplies learned state modulation | Closed-loop physiological needs or homeostasis |

Messages are continuous learned tensors passed through an explicit computational graph; recurrent state carries information between discrete steps. These are functional communication analogies. The model does not simulate spikes, conduction delays, cell types, neurotransmitters, oscillations, or local biological learning. Joint backpropagation is the v1 training method. Future asynchronous updates or local objectives must be compared against this simple reference before adoption.

## What the initial experiments establish

The five symbolic tasks cover visual response matching, auditory response matching, instruction-based channel selection, delayed recall, and a modulo-four cross-modal rule. A predictive auxiliary provides an additional learning objective. These tasks exercise interfaces, gradient flow, routing, and elementary memory; a small finite symbol space permits shortcuts and memorization.

Within-episode memory and trainer replay are separate mechanisms. The former can influence current inference; the latter stores examples for later training. Replay is a plausible research direction because van de Ven and colleagues demonstrated continual-learning benefits from a particular brain-inspired **generative** replay method. Our stored-example rehearsal is a simpler, different mechanism, and its benefit requires its own experiment. [van de Ven et al., *Brain-inspired replay for continual learning*](https://www.nature.com/articles/s41467-020-17866-2).

## Measurable advancement gates

These are proposed future acceptance targets, not reported achievements. Freeze splits, budgets, and thresholds before running comparisons. Use at least five training seeds for capability claims, with per-task scores and uncertainty intervals.

| Gate | Experiment | Acceptance target |
|---|---|---|
| 1. Reproducible learning | Train/evaluate the symbolic suite on separate generated episodes; verify save/reload and all-region gradient flow | Every task exceeds its majority/chance baseline; target at least 90% per task, with reproducible evaluation and bounded memory |
| 2. Useful specialization | Compare parameter-matched GRU, ungated routing, retrained region removal, and inference lesions | Identify a repeatable task-selective benefit; retain architectural complexity only when supported by accuracy, transfer, or efficiency |
| 3. Episodic retrieval | Query previously stored arbitrary associations after resetting working state; introduce distractors and interference | At least 80% retrieval on held-out associations and at least 15 percentage points over disabled retrieval; report capacity curves |
| 4. Sequential learning | Learn tasks A through E in blocks; compare no replay, fixed-budget replay, and joint training | Average final forgetting below 10 percentage points while remaining within 5 points of joint-training accuracy; fix replay bytes and updates |
| 5. Grounded sensors | Replace symbols with small images and audio clips; hold out objects/styles and speakers/noise conditions | Both modalities exceed simple encoder baselines on declared distribution shifts; never report random-split scores as transfer |
| 6. Rules and sequences | Train primitive instructions, then test withheld compositions and sequences twice the training length | At least 80% exact-sequence accuracy on each held-out family, and a measurable benefit over the matched recurrent baseline |
| 7. Embodied learning | Add a small partially observed gridworld with actions, goals, rewards, and unseen layouts | At least 80% held-out task success within a fixed interaction budget; report failure modes and comparison policies |
| 8. Planning | Learn action-conditioned transitions and reward/termination predictions; evaluate imagined action sequences before choosing | Planning improves held-out success over the same policy without planning under equal compute; compare multiple rollout depths and model error |

For forgetting, record the full accuracy matrix: after every training block, evaluate every task learned so far. Per-task forgetting is its best earlier accuracy minus final accuracy; report the mean alongside final accuracy and learning speed. Test labels and examples must never enter replay or model selection.

Held-out compositions matter because the SCAN study found that successful recurrent-network generalization on easier splits did not ensure systematic compositional generalization. [Lake and Baroni, *Generalization without systematicity*](https://arxiv.org/abs/1711.00350). Planning requires more than prediction loss: Dreamer learned behavior through imagined trajectories in a learned latent world model. That offers a concrete later comparison, not an implemented v1 capability. [Hafner et al., *Dream to Control*](https://arxiv.org/abs/1912.01603).

## Home-computer scaling

Keep a CPU reference configuration with hidden width 48 and a small parameter budget. Record actual parameter counts, peak RAM/VRAM, update latency, inference latency, and wall time on the machine used. The local Ryzen 9/RTX 5080/32 GB desktop is now measured hardware: completed optimizer and learning studies are recorded in [HOME_COMPUTE.md](HOME_COMPUTE.md). Those finite workloads do not establish its useful learning limit. Verify the installed PyTorch build and a real CUDA operation before enabling GPU training; retain CPU execution as a reference. Declare and validate the numerical execution profile separately from learning-quality claims.

Scale bottlenecks individually: sensor encoders first, then bounded persistent retrieval, action-conditioned prediction, and tested planning. Preserve versioned message shapes, checkpoint compatibility, explicit memory limits, and ablation controls. Pretrained encoders may accelerate later work, but report their external training provenance separately from learning performed here. More regions, parameters, or anatomical resemblance are not acceptance criteria for general intelligence; demonstrated transfer, retention, and learning efficiency are.
