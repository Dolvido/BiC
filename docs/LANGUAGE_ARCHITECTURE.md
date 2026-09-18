# v0.2 language interfaces: implemented scope

**Historical scope:** this document records the untrained v0.2 interface release. V0.3 reuses these components in a separate, authorized computer-use training pipeline with pixel observations and free-running replies; see [COMPUTER_USE_POC.md](COMPUTER_USE_POC.md) and [COMPUTER_RESULTS.md](COMPUTER_RESULTS.md). The readiness command described here still initializes random language weights and performs no training.

This version prepares pathways for future language learning. It does not speak or understand English. All newly introduced language weights are randomly initialized whenever the readiness command runs. No pretrained model, trained tokenizer, language corpus download, language optimizer, or trained language checkpoint is included.

## Biological design commitments

The brain-inspired goal is to connect language to perceptual association, working memory, goals, episodic retrieval, and action. Frontal and temporal names describe engineering biases within a distributed system. They do not assert that comprehension and production live in two isolated human areas. The scientific motivation and teaching plan are in `ENGLISH_FEASIBILITY.md`.

Two ideas guide the new interface: input language should influence the regional state used to act, and output language should depend on what that state represents. The code makes both dependencies possible. It does not demonstrate that the representations have acquired meaning, that an internal state is truthful, or that a model cannot learn a shortcut.

## Networks and signal paths

| Component | Parameters, default | Implemented computation |
|---|---:|---|
| Existing 13-region brain | 162,810 | The v0.1 recurrent regional network, with an optional context input |
| `posterior_temporal` | 91,072 | Its own byte embedding and causal GRU; summarizes an incoming utterance |
| `semantic_bridge` | 25,104 | Two separately parameterized projections: language-to-core and core-to-production |
| `inferior_frontal` | 190,147 | Its own byte embedding, context-conditioned GRU, and next-byte output head |
| Total | 469,133 | Interface prototype, with 306,323 new language/bridge parameters |

`posterior_temporal` and `inferior_frontal` do not share parameter tensors. The bridge is an engineering interface, not an additional anatomical claim. The default language hidden width is 128; embedding width is 64. Larger widths are configurable for future experiments; more parameters do not imply more learned capability before training.

The forward schedule is:

1. Encode an entire utterance whose content is known before the episode begins. Packed sequence processing excludes right-padding from the text state.
2. Map that state to the core's hidden width, then provide it as an additive input to the existing temporal-language network at each observation step. Existing top-down prefrontal feedback still operates inside the core.
3. Run the sensory/action episode. All other regional processing proceeds through the core's explicit pathways.
4. Concatenate the **final prefrontal, parietal, and hippocampal activities**, and map them into a production context.
5. Condition a causal production GRU on that context and preceding output bytes; return next-byte logits.

There is no direct input-text-to-production connection outside the regional core. Disabling the core's temporal-language network blocks all incoming text context from influencing the production head. This is a structural property. A future learned system could still encode the whole prompt in the core or ignore sensory facts, so randomized observations, role reversals, and evidence-removal tests remain necessary.

The current design is **utterance, then episode, then response**. It does not interleave listening, acting, and speaking online. The comprehension encoder does not yet receive recurrent top-down feedback from the broader brain while reading. Full reciprocal language interaction is a future architectural extension, not an implemented claim.

## Text representation

`ByteCodec` uses a fixed UTF-8 mapping: PAD=0, BOS=1, EOS=2, and byte values offset by 3, for 259 symbols. Encoding preserves punctuation, case, and valid Unicode text. Limits count bytes, not characters. Input and output limits default to 256 and 128 bytes respectively, excluding special framing tokens.

The codec has no training phase and teaches no meanings. Oversized inputs raise an error rather than being silently truncated. Decoding invalid UTF-8 raises by default; lossy display requires an explicit `errors="replace"`. Subword tokenization may be more efficient for a later corpus, but choosing or fitting a tokenizer must be recorded as part of that experiment.

## Python interface

```python
import torch
from brain_in_computer.model import Brain
from brain_in_computer.language import ByteCodec, LanguageBrain
from brain_in_computer.tasks import TaskStream

model = LanguageBrain(Brain()).eval()  # Random language and core weights here.
observations = TaskStream(123).sample(1, delay=0).observations
observations["tokens"].zero_()  # Avoid using a legacy task ID as a language shortcut.
ids, lengths = model.prepare_inputs(["Look at the red cube."])
decoder_prefix = torch.tensor([[ByteCodec.BOS]])
with torch.no_grad():
    result = model(observations, ids, lengths, decoder_prefix)
print(result["language_logits"].shape)  # [1, 1, 259]; not an English answer.
```

The synthetic observations in this example do **not** depict a red cube. They are an interface fixture only. A scene encoder and teacher-labeled grounded data are still needed.

`LanguageBrain.forward(observations, text_ids, text_lengths, decoder_input_ids, ablate=())` returns the core outputs plus:

| Field | Shape |
|---|---|
| `language_logits` | `[batch, output_prefix_length, 259]` |
| `concept_context` | `[batch, core_hidden]` |
| `production_context` | `[batch, language_hidden]` |
| `comprehension_states` | `[batch, input_length, language_hidden]` |

All rows are independent. Core recurrent state and hippocampal memory reset per forward call. Conversation history does not persist automatically. The language wrapper exposes the core's existing region lesions; it does not add a complete lesion API for the three new components.

Input text must have BOS, byte IDs, EOS, and contiguous right padding. Decoder inputs must start with BOS and contain only a preceding output prefix. A future trainer must shift targets correctly and mask padded positions; the module cannot inspect the origin of a caller-supplied tensor to prevent target leakage. Logits at padded decoder positions have no supervised meaning.

## Compatibility and limits

The core `Brain.forward` accepts optional `language_context` with shape `[batch, observation_time, core_hidden]`. `None` and an all-zero context preserve the v0.1 computation. Existing state-dictionary keys and parameter shapes are unchanged, and the original core checkpoint remains readable. The old `Trainer` still handles only symbolic-core models; it does not save, restore, or optimize `LanguageBrain`.

`language-check` runs inference only and verifies unchanged weights and actual input influence. `language-plan` estimates parameter-storage arithmetic only. Neither is a language-learning benchmark. The pilot capacity range in the feasibility report is a future experiment; the smaller default configuration here is designed for inexpensive interface testing.

Before training English, implement a richer observation/action world, a corpus contract with fixed validation/test splits, a clear generation objective and free-running evaluator, persistent cross-turn state/reset semantics, and a language-aware checkpoint format. Before claiming episodic word learning, demonstrate that retrieval helps after working state has been reset. These are substantive next steps, not hidden capabilities already present in v0.2.
