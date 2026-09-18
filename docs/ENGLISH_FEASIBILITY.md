# Teaching Brain in Computer English: feasibility and proposed curriculum

**Feasibility verdict:** teaching a small native system controlled, grounded English is a credible home-computer research project. Reliable open-ended conversation from scratch is a much harder goal; human-level general intelligence is not an established outcome of this architecture. Start with words that refer to objects, actions, memories, and goals the system can actually distinguish.

**Status:** this document proposes future work. No English training or pretrained-model download was performed for it. Parameter ranges below are planning candidates, not implemented model sizes or demonstrated capability.

## What the existing result establishes

V0.1 has 13 regional networks and 162,810 parameters. Its five tiny symbolic tasks exercise selected sensory, instruction, and memory pathways. Its instruction tokens are identifiers; its four response classes are not English generation. Success on those tasks provides a useful software baseline, without establishing language understanding. Removing hippocampal output did not reduce the reported curriculum accuracy, so those results do not establish that episodic retrieval is functionally useful yet.

## Feasible paths on Luke's computer

Luke's reported desktop has a Ryzen 9 9900X and 32 GB system RAM. The standard RTX 5080 has **16 GB dedicated GPU memory**; system RAM does not add to that pool at GPU speed. This is a specification-based assessment, not a benchmark of Luke's machine. [NVIDIA RTX 5080 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/)

| Path | Candidate language-module scale | Practical interpretation |
|---|---:|---|
| Native grounded pilot | 1–10 million parameters | Recommended first experiment: restricted instructions, descriptions, questions, and short replies in a controlled environment. |
| Richer native restricted English | 10–100 million | Plausible capacity experiment for simple stories and longer dialogue within a limited domain; resource fit does not guarantee learning. |
| Imported language ability | Approximately 1–3 billion pretrained parameters | Optional later frontend with frozen weights or small trainable adapters; imports substantial prior learning rather than demonstrating native English acquisition. |
| Frontier-style general English from scratch | No credible home training target | Do not use this as the first milestone or promise it as a result of adding brain regions. |

The **TinyStories** study demonstrates that models below ten million parameters can learn coherent text in a deliberately restricted story distribution. It supports attempting a small language pilot; it does not establish unrestricted dialogue, human developmental equivalence, or success for our regional architecture. [Eldan and Li, 2023](https://arxiv.org/abs/2305.07759)

A pretrained alternative such as **SmolLM3-3B** already has English capability. Its authors report 11.2 trillion pretraining tokens, illustrating the large prior investment that an apparently small downloaded model contains. Integration would still need grounding and causal tests so that the frontend cannot supply plausible answers while ignoring the regional system. [Official SmolLM3 model card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)

For planning, FP32 weights, gradients, and two Adam moments total approximately **16 bytes per trainable parameter**. This gives about 160 MB at 10 million parameters, 1.6 GB at 100 million, and 48 GB at three billion, **before activations, buffers, other regions, and framework overhead**. These are arithmetic estimates in decimal units, not measured peak allocations. Mixed precision and optimizer implementations change the accounting.

Inference is different: three billion two-byte weights alone occupy about 6 GB; ideal four-bit storage is 1.5 GB before quantization metadata, temporary buffers, and attention caches. This does not imply that full training fits. Frozen quantized weights plus adapters reduce training-state requirements, while activation memory still matters. [QLoRA](https://arxiv.org/abs/2305.14314)

Begin with short contexts, small batches, and a compact vocabulary. Stream larger text collections from disk. Estimate elapsed training time only after measuring tokens per second and peak memory on the actual configuration; parameter count alone cannot supply a defensible hours-or-days estimate.

## Biology as an architectural constraint

Human language relies on interacting frontal and temporal areas with overlapping comprehension and production functions. A rigid “Wernicke understands; Broca speaks” map is too simple. Use posterior temporal comprehension and inferior frontal production as **engineering biases**, with shared representations and reciprocal communication. Language proficiency must be evaluated separately from broader reasoning. [Fedorenko, Piantadosi, and Gibson, 2024](https://colala.berkeley.edu/papers/fedorenko2024language.pdf)

The proposed regional responsibilities are:

| Functional component | Learning objective and communication |
|---|---|
| Temporal language network | Encode ordered text; exchange word and sentence representations with association regions and frontal language processing. |
| Frontal language network | Generate a reply conditioned on a communicative goal, selected referents, and retrieved evidence. |
| Visual/parietal association | Represent objects, attributes, locations, and action roles; connect language to observations. |
| Prefrontal and action-selection networks | Maintain the current goal and conversational state; decide whether to answer, act, or request clarification. |
| Hippocampal memory | Bind a new name or event to its context and retrieve it from a later cue. |
| Cortical learning with replay | Consolidate recurring knowledge while retaining previously learned tasks. |

Rapid episodic storage combined with slower interleaved learning is inspired by complementary learning systems theory. A bounded associative memory and replay are engineering approximations, not claims of physiological fidelity. [McClelland, McNaughton, and O'Reilly, 1995](https://pubmed.ncbi.nlm.nih.gov/7624455/)

Written English is the first interface because it makes learning and testing simpler. Auditory recognition and speech articulation would later become separately trained networks connected to the same linguistic and grounded representations. Bytes, subwords, GRUs, and backpropagation are computational choices, not literal biological mechanisms.

## The teaching sequence

1. **Prepare an observable world and honest tests.** Define a small world containing objects, colors, positions, movements, and explicit unknown information. Keep observations separate from teacher labels. Create training, validation, and sealed test partitions before optimization. Record example generation rules and data provenance; use only authorized text sources.

2. **Teach shared reference.** Pair an observed object with “red cube,” train recognition from text, and train descriptions from observations. Start with a limited vocabulary and short utterances. A tokenizer represents spelling; it does not teach meaning. Use distinct objectives for understanding and generation, and ensure both directions are tested.

3. **Teach composition and actions.** Progress to “move the red cube left,” comparisons, negation, and relations. Swap colors, objects, positions, and participant roles while balancing target frequencies. “The red cube is left of the blue cube” and the reversed relation must require different representations. Hold out whole combinations and some sentence structures, not merely random examples. SCAN research shows why ordinary random splits can conceal poor compositional generalization. [Lake and Baroni, 2018](https://arxiv.org/abs/1711.00350)

4. **Teach episodic words and facts.** Introduce a novel name once, insert distractors, then ask about it. Change name-to-object mappings between episodes so that memorizing a fixed vocabulary table cannot solve the task. Test cue-based retrieval after resetting working state while preserving permitted episodic memory. Compare intact memory, empty memory, and shuffled memory. A causal benefit must be demonstrated before crediting hippocampal learning.

5. **Teach purposeful communication.** Give the speaker information the listener lacks. Reward successful identification or action, alongside teacher examples of short replies. Include ambiguous requests, corrections, and questions whose answers are absent. Teach clarification and “I don't know” using explicitly labeled conditions. Begin with supervised imitation; introduce reinforcement learning only after an interactive task and independently checkable outcome exist.

6. **Consolidate and broaden.** Interleave earlier tasks during later lessons. Measure forgetting before and after replay. Expand vocabulary, paraphrases, and short stories only after grounded tests succeed. Keep text fluency objectives alongside action and retrieval objectives so that better prose cannot conceal worsening task performance. Speech becomes a separate later milestone.

## Evidence required before calling it English communication

Report separate scores for **grammar**, **meaning**, **compositional transfer**, **episodic recall**, **communicative success**, and **retention**. Free generation must use its own previous tokens; teacher-forced next-token accuracy alone is insufficient. Include target-role reversals, unseen combinations, longer delays, unknown answers, and held-out wording. Report sample counts, uncertainty, and results across multiple seeds.

Compare against a simple language-only baseline and a parameter-matched integrated model. Remove or shuffle particular inputs and messages to identify actual dependence. Test-time lesions reveal sensitivity; retrained comparisons are needed to assess architectural benefit. If removing grounded observations or episodic memory leaves relevant answers intact, inspect leakage or shortcuts before scaling.

The first useful demonstration would be modest and concrete: show a new object, teach its name, ask the system to act on a novel instruction about it, and receive a correct short explanation after a delay. No English capability is claimed until that behavior is learned and survives the held-out tests.
