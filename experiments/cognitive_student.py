"""Domain-independent neural episodic retrieval through the regional brain.

Only observed bytes and learned regional activity enter retrieval. Supervised
answers and the decoder's teacher-forcing inputs cannot enter the memory path.
This experimental model leaves all released model/checkpoint sources untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

import torch
from torch import Tensor, nn

from brain_in_computer.language import ByteCodec, LanguageBrain, LanguageConfig
from brain_in_computer.learning_student import _INITIALIZATION_LOCK, _integer
from brain_in_computer.model import Brain, BrainConfig, BrainState


@dataclass(frozen=True)
class EpisodicState:
    """Caller-owned activity; rows remain separate sessions across calls."""

    brain_state: BrainState
    token_memory: tuple[Tensor, ...] = ()
    token_masks: tuple[Tensor, ...] = ()

    def detached(self):
        return EpisodicState(self.brain_state.detached(),
            tuple(value.detach() for value in self.token_memory),
            tuple(value.detach() for value in self.token_masks))


class CognitiveStudent(LanguageBrain):
    """Two shared retrieval hops; all answers still pass through the brain.

    Both modes have identical parameters and initial weights. ``episodic``
    retrieves current and earlier utterances, ``recurrent`` current text only;
    both preserve ordinary BrainState. Stored tensors are bounded by eight
    turns, not detached implicitly, and contain no targets or decoder bytes.
    Do not carry training activity across optimizer updates.
    """

    memory_turns = 8
    retrieval_hops = 2

    def __init__(self, *, memory_mode="episodic"):
        if memory_mode not in ("episodic", "recurrent"):
            raise ValueError("memory_mode must be episodic or recurrent")
        super().__init__(Brain(BrainConfig(hidden_size=64, visual_dim=32,
            auditory_dim=4, body_dim=4, vocab_size=1, num_actions=4, memory_slots=8)),
            LanguageConfig(hidden_size=128, embedding_size=32,
                           max_input_bytes=128, max_output_bytes=32))
        self.memory_mode = memory_mode
        self.retrieval_query = nn.Linear(192, 128)
        self.retrieval_attention = nn.MultiheadAttention(128, 4, dropout=0., batch_first=True)
        self.retrieval_update = nn.GRUCell(128, 128)
        self.retrieval_norm = nn.LayerNorm(128)
        self.retrieval_projection = nn.Linear(128, 64)
        self.retrieval_age = nn.Embedding(self.memory_turns, 128)
        self.observation_predictor = nn.Linear(128, ByteCodec.VOCAB_SIZE)

    def parameter_counts(self):
        result = super().parameter_counts()
        result["episodic_memory"] = sum(parameter.numel()
            for name, parameter in self.named_parameters() if name.startswith("retrieval_"))
        result["observation_prediction"] = sum(p.numel() for p in self.observation_predictor.parameters())
        return result

    def _validate_episodic_state(self, state, batch):
        if not isinstance(state, EpisodicState):
            raise ValueError("state must be EpisodicState or None")
        self.brain._validate_state(state.brain_state, batch)
        if (not isinstance(state.token_memory, tuple) or not isinstance(state.token_masks, tuple)
                or len(state.token_memory) != len(state.token_masks)
                or len(state.token_memory) > self.memory_turns):
            raise ValueError("episodic memory requires matching bounded tuples")
        reference = next(self.parameters())
        for memory, mask in zip(state.token_memory, state.token_masks):
            if (not isinstance(memory, Tensor) or memory.ndim != 3
                    or memory.shape[0] != batch or memory.shape[2] != self.config.hidden_size
                    or not 2 <= memory.shape[1] <= self.config.max_input_bytes + 2
                    or memory.device != reference.device or memory.dtype != reference.dtype
                    or not torch.isfinite(memory).all()):
                raise ValueError("invalid token memory shape, dtype, device or finite values")
            if (not isinstance(mask, Tensor) or mask.shape != memory.shape[:2]
                    or mask.dtype != torch.bool or mask.device != reference.device):
                raise ValueError("invalid token memory mask")

    def forward(self, observations, text_ids, text_lengths, decoder_input_ids, *, ablate=()):
        output, _ = self.forward_with_state(observations, text_ids, text_lengths,
                                           decoder_input_ids, ablate=ablate)
        return output

    def forward_with_state(self, observations, text_ids, text_lengths,
                           decoder_input_ids, state=None, *, ablate=()):
        if isinstance(ablate, str):
            raise ValueError("ablate must be an iterable of region names")
        ablate = frozenset(ablate)
        batch, time = self.brain._validate_observations(observations)
        self._validate_text(text_ids, text_lengths, batch)
        self._validate_decoder(decoder_input_ids, batch)
        state = EpisodicState(self.brain.initial_state(batch)) if state is None else state
        self._validate_episodic_state(state, batch)
        comprehension, final_text = self.posterior_temporal(text_ids, text_lengths)
        positions = torch.arange(text_ids.shape[1], device=text_ids.device)[None, :]
        mask = (positions > 0) & (positions < text_lengths[:, None] - 1)
        memories = (*state.token_memory, comprehension)[-self.memory_turns:]
        masks = (*state.token_masks, mask)[-self.memory_turns:]
        selected = memories if self.memory_mode == "episodic" else memories[-1:]
        selected_masks = masks if self.memory_mode == "episodic" else masks[-1:]
        values = torch.cat(selected, dim=1)
        valid = torch.cat(selected_masks, dim=1)
        ages = torch.cat([torch.full((value.shape[1],), len(selected) - index - 1,
            dtype=torch.long, device=text_ids.device) for index, value in enumerate(selected)])
        keys = values + self.retrieval_age(ages)[None, :, :]
        # A zero sentinel handles empty/blank utterances without a softmax over
        # an entirely masked row. It is available only when no byte is visible.
        empty = ~valid.any(dim=1, keepdim=True)
        values = torch.cat((values, values.new_zeros(batch, 1, 128)), dim=1)
        keys = torch.cat((keys, keys.new_zeros(batch, 1, 128)), dim=1)
        valid = torch.cat((valid, empty), dim=1)
        previous = state.brain_state.previous_prefrontal
        if "prefrontal_cortex" in ablate:
            previous = torch.zeros_like(previous)
        query = self.retrieval_query(torch.cat((final_text, previous), dim=-1)).tanh()
        for _ in range(self.retrieval_hops):
            read, _ = self.retrieval_attention(query[:, None, :], keys, values,
                                               key_padding_mask=~valid, need_weights=False)
            query = self.retrieval_norm(self.retrieval_update(read[:, 0], query))
        memory_context = self.retrieval_projection(query).tanh()[:, None, :].expand(-1, time, -1)
        concept = self.semantic_bridge.comprehend(final_text)
        output, brain_state = self.brain.forward_with_state(observations, state.brain_state,
            ablate=ablate, return_activity=True, memory_context=memory_context,
            language_context=concept[:, None, :].expand(-1, time, -1))
        activity = output["region_activity"]
        production = self.semantic_bridge.formulate(torch.cat([activity[name][:, -1]
            for name in ("prefrontal_cortex", "parietal_association", "hippocampus")], dim=-1))
        return {**output, "concept_context": concept, "production_context": production,
            "comprehension_states": comprehension,
            "observation_language_logits": self.observation_predictor(comprehension),
            "language_logits": self.inferior_frontal(decoder_input_ids, production)}, \
            EpisodicState(brain_state, memories, masks)

    def _state_config(self):
        return torch.tensor([1, int(self.memory_mode == "episodic"), self.memory_turns,
            self.retrieval_hops, *asdict(self.brain.config).values(), *asdict(self.config).values()],
            dtype=torch.int64)

    @staticmethod
    def _digest_tensor(checkpoint_sha256):
        if not isinstance(checkpoint_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA256 hex digest")
        return torch.tensor(list(bytes.fromhex(checkpoint_sha256)), dtype=torch.uint8)

    def state_to_dict(self, state, *, checkpoint_sha256):
        """Export independent CPU tensors, bound to the caller's weight digest."""
        if (not isinstance(state, EpisodicState) or not isinstance(state.brain_state, BrainState)
                or not isinstance(state.brain_state.previous_prefrontal, Tensor)
                or state.brain_state.previous_prefrontal.ndim != 2):
            raise ValueError("state must have a two-dimensional batch")
        self._validate_episodic_state(state, state.brain_state.previous_prefrontal.shape[0])
        return {"config": self._state_config(), "checkpoint_sha256": self._digest_tensor(checkpoint_sha256),
            "brain_state": self.brain.state_to_dict(state.brain_state),
            "token_memory": [value.detach().cpu().clone() for value in state.token_memory],
            "token_masks": [value.detach().cpu().clone() for value in state.token_masks]}

    def state_from_dict(self, payload, *, checkpoint_sha256):
        """Restore activity only for the expected configuration and weight digest."""
        expected = {"config", "checkpoint_sha256", "brain_state", "token_memory", "token_masks"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("invalid episodic state fields")
        for name, actual in (("config", self._state_config()),
                             ("checkpoint_sha256", self._digest_tensor(checkpoint_sha256))):
            value = payload[name]
            if (not isinstance(value, Tensor) or value.device.type != "cpu"
                    or value.dtype != actual.dtype or not torch.equal(value, actual)):
                raise ValueError(f"episodic state {name} mismatch")
        memories, masks = payload["token_memory"], payload["token_masks"]
        if (not isinstance(memories, list) or not isinstance(masks, list)
                or len(memories) != len(masks) or len(memories) > self.memory_turns):
            raise ValueError("invalid saved memory lists")
        reference = next(self.parameters())
        for value in memories:
            if (not isinstance(value, Tensor) or value.device.type != "cpu"
                    or value.dtype != reference.dtype or not torch.isfinite(value).all()):
                raise ValueError("invalid saved memory values")
        for value in masks:
            if not isinstance(value, Tensor) or value.device.type != "cpu" or value.dtype != torch.bool:
                raise ValueError("invalid saved memory masks")
        brain_state = self.brain.state_from_dict(payload["brain_state"])
        state = EpisodicState(brain_state,
            tuple(value.to(reference.device).clone() for value in memories),
            tuple(value.to(reference.device).clone() for value in masks))
        self._validate_episodic_state(state, brain_state.previous_prefrontal.shape[0])
        return state


def build_cognitive_student(seed, device="cpu", memory_mode="episodic"):
    """Fresh small neural learner, without changing caller RNG state."""
    _integer("seed", seed)
    if seed >= 2**63:
        raise ValueError("seed must be < 2**63")
    with _INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = CognitiveStudent(memory_mode=memory_mode)
    return model.to(device)
