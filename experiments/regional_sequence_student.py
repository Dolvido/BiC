"""Untrained causal observation encoder integrated into BiC's regional paths.

This preparation is not a trained checkpoint or a capability claim. Only the
sequence encoder is retained: action selection and reply production use the
existing regional model. Caller-owned history contains observed bytes only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from brain_in_computer.language import ByteCodec
from brain_in_computer.learning_student import _INITIALIZATION_LOCK, _integer
from brain_in_computer.model import BrainState
from experiments.cognitive_student import CognitiveStudent
from experiments.sequence_student import SequenceConfig, SequenceStudent


@dataclass(frozen=True)
class RegionalSequenceState:
    """A session batch's regional activity and bounded raw observation prefix."""

    brain_state: BrainState
    text_history: tuple[Tensor, ...] = ()
    text_lengths: tuple[Tensor, ...] = ()

    def detached(self):
        return RegionalSequenceState(self.brain_state.detached(),
            tuple(value.detach() for value in self.text_history),
            tuple(value.detach() for value in self.text_lengths))


class ObservationSequenceEncoder(nn.Module):
    """Reuse the diagnostic's causal encoder, retaining none of its heads."""

    def __init__(self, config):
        super().__init__()
        source = SequenceStudent(config)
        self.config = source.config
        for name in ("tokens", "positions", "blocks", "norm"):
            setattr(self, name, getattr(source, name))

    def forward(self, token_ids, eos_positions):
        valid = SequenceStudent._validate_inputs(self, token_ids, eos_positions, None, None)
        width = token_ids.shape[1]
        positions = torch.arange(width, device=token_ids.device)
        context = self.tokens(token_ids) + self.positions(positions)[None]
        causal = torch.ones(width, width, dtype=torch.bool, device=token_ids.device).triu(1)
        for block in self.blocks:
            context = block(context, src_mask=causal, src_key_padding_mask=~valid, is_causal=True)
        return self.norm(context).masked_fill(~valid[:, :, None], 0.)


class RegionalSequenceStudent(CognitiveStudent):
    """Shared causal representations enter temporal and hippocampal pathways.

    The existing regional brain, retrieval, motor head and regional reply
    bridge remain. Retrieval reads only current-utterance contextual states;
    their causal encoder can attend to the retained observation prefix. The
    byte GRU is replaced rather than kept as an unused parameter budget.

    History is bounded by ``sequence_config.max_turns`` and ``max_positions``.
    Exceeding either limit raises instead of silently truncating observations;
    the caller must reset explicitly. Default capacity fits six maximum-length
    turns. No module-owned session state or KV cache exists; each call
    recomputes the prefix. Do not carry live regional graphs
    across optimizer steps; detached/saved state is an explicit graph boundary.
    """

    def __init__(self, sequence_config=None):
        config = SequenceConfig() if sequence_config is None else sequence_config
        if not isinstance(config, SequenceConfig):
            raise ValueError("sequence_config must be SequenceConfig")
        if config.max_input_bytes != 128 or config.max_output_bytes != 32:
            raise ValueError("regional language interface requires input/output limits 128/32")
        super().__init__(memory_mode="recurrent")
        self.sequence_config = config
        self.posterior_temporal = ObservationSequenceEncoder(config)
        self.sequence_projection = nn.Sequential(nn.Linear(config.width, 128),
                                                nn.LayerNorm(128), nn.Tanh())

    def parameter_counts(self):
        return {**super().parameter_counts(), "sequence_projection":
                sum(parameter.numel() for parameter in self.sequence_projection.parameters())}

    def _validate_sequence_state(self, state, batch):
        if not isinstance(state, RegionalSequenceState):
            raise ValueError("state must be RegionalSequenceState or None")
        self.brain._validate_state(state.brain_state, batch)
        if (not isinstance(state.text_history, tuple) or not isinstance(state.text_lengths, tuple)
                or len(state.text_history) != len(state.text_lengths)
                or len(state.text_history) > self.sequence_config.max_turns):
            raise ValueError("observation history requires matching bounded tuples")
        for ids, lengths in zip(state.text_history, state.text_lengths):
            self._validate_text(ids, lengths, batch)
        if state.text_lengths and (torch.stack(state.text_lengths).sum(0)
                                  > self.sequence_config.max_positions).any():
            raise ValueError("observation history exceeds position capacity")

    def _prefix(self, state, text_ids, text_lengths):
        # Copies keep returned history independent of mutable caller inputs.
        history = (*state.text_history, text_ids.clone())
        lengths = (*state.text_lengths, text_lengths.clone())
        if len(history) > self.sequence_config.max_turns:
            raise ValueError("observation history exceeds turn capacity; reset explicitly")
        if (torch.stack(lengths).sum(0) > self.sequence_config.max_positions).any():
            raise ValueError("observation history exceeds position capacity; reset explicitly")
        turn_lengths = torch.stack(lengths, dim=1)
        eos_positions = turn_lengths.cumsum(1) - 1
        joined = torch.cat(history, dim=1)
        valid = joined.ne(ByteCodec.PAD)
        width = int(turn_lengths.sum(1).max())
        packed = joined.new_zeros(joined.shape[0], width)
        # PAD contributes zero, including repeated indices between utterances.
        packed.scatter_add_(1, (valid.long().cumsum(1) - 1).clamp_min(0), joined)
        return packed, eos_positions, history, lengths

    def forward_with_state(self, observations, text_ids, text_lengths,
                           decoder_input_ids, state=None, *, ablate=()):
        if isinstance(ablate, str):
            raise ValueError("ablate must be an iterable of region names")
        ablate = frozenset(ablate)
        batch, time = self.brain._validate_observations(observations)
        self._validate_text(text_ids, text_lengths, batch)
        self._validate_decoder(decoder_input_ids, batch)
        state = RegionalSequenceState(self.brain.initial_state(batch)) if state is None else state
        self._validate_sequence_state(state, batch)
        packed, eos, history, lengths = self._prefix(state, text_ids, text_lengths)
        context = self.posterior_temporal(packed, eos)
        offsets = eos[:, -1] + 1 - text_lengths
        positions = torch.arange(text_ids.shape[1], device=text_ids.device)[None]
        indices = (offsets[:, None] + positions).clamp_max(context.shape[1] - 1)
        current = context.gather(1, indices[:, :, None].expand(-1, -1, context.shape[-1]))
        comprehension = self.sequence_projection(current).masked_fill(
            (positions >= text_lengths[:, None])[:, :, None], 0.)
        final_text = comprehension.gather(1,
            (text_lengths - 1)[:, None, None].expand(-1, 1, 128))[:, 0]
        valid = (positions > 0) & (positions < text_lengths[:, None] - 1)
        values = comprehension
        keys = values + self.retrieval_age.weight[0][None, None]
        # Preserve CognitiveStudent's finite blank-text fallback.
        empty = ~valid.any(1, keepdim=True)
        values = torch.cat((values, values.new_zeros(batch, 1, 128)), dim=1)
        keys = torch.cat((keys, keys.new_zeros(batch, 1, 128)), dim=1)
        valid = torch.cat((valid, empty), dim=1)
        previous = state.brain_state.previous_prefrontal
        if "prefrontal_cortex" in ablate:
            previous = torch.zeros_like(previous)
        query = self.retrieval_query(torch.cat((final_text, previous), dim=-1)).tanh()
        for _ in range(self.retrieval_hops):
            read, _ = self.retrieval_attention(query[:, None], keys, values,
                                               key_padding_mask=~valid, need_weights=False)
            query = self.retrieval_norm(self.retrieval_update(read[:, 0], query))
        memory_context = self.retrieval_projection(query).tanh()[:, None].expand(-1, time, -1)
        concept = self.semantic_bridge.comprehend(final_text)
        output, brain_state = self.brain.forward_with_state(observations, state.brain_state,
            ablate=ablate, return_activity=True, memory_context=memory_context,
            language_context=concept[:, None].expand(-1, time, -1))
        activity = output["region_activity"]
        production = self.semantic_bridge.formulate(torch.cat([activity[name][:, -1]
            for name in ("prefrontal_cortex", "parietal_association", "hippocampus")], dim=-1))
        return {**output, "concept_context": concept, "production_context": production,
            "comprehension_states": comprehension,
            "observation_language_logits": self.observation_predictor(comprehension),
            "language_logits": self.inferior_frontal(decoder_input_ids, production)}, \
            RegionalSequenceState(brain_state, history, lengths)

    def _state_config(self):
        # Model/schema identifier and fail-on-capacity policy are
        # part of compatibility, alongside every architectural configuration.
        return torch.tensor([2, 1, self.retrieval_hops, self.memory_turns,
            *asdict(self.sequence_config).values(), *asdict(self.brain.config).values(),
            *asdict(self.config).values()], dtype=torch.int64)

    def state_to_dict(self, state, *, checkpoint_sha256):
        """Detached independent CPU tensors, bound to config and weight digest."""
        if (not isinstance(state, RegionalSequenceState) or not isinstance(state.brain_state, BrainState)
                or not isinstance(state.brain_state.previous_prefrontal, Tensor)
                or state.brain_state.previous_prefrontal.ndim != 2):
            raise ValueError("state must have a two-dimensional batch")
        self._validate_sequence_state(state, state.brain_state.previous_prefrontal.shape[0])
        return {"config": self._state_config(), "checkpoint_sha256": self._digest_tensor(checkpoint_sha256),
            "brain_state": self.brain.state_to_dict(state.brain_state),
            "text_history": [value.detach().cpu().clone() for value in state.text_history],
            "text_lengths": [value.detach().cpu().clone() for value in state.text_lengths]}

    def state_from_dict(self, payload, *, checkpoint_sha256):
        expected = {"config", "checkpoint_sha256", "brain_state", "text_history", "text_lengths"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("invalid regional sequence state fields")
        for name, actual in (("config", self._state_config()),
                             ("checkpoint_sha256", self._digest_tensor(checkpoint_sha256))):
            value = payload[name]
            if (not isinstance(value, Tensor) or value.device.type != "cpu"
                    or value.dtype != actual.dtype or not torch.equal(value, actual)):
                raise ValueError(f"regional sequence state {name} mismatch")
        history, lengths = payload["text_history"], payload["text_lengths"]
        if (not isinstance(history, list) or not isinstance(lengths, list)
                or len(history) != len(lengths) or len(history) > self.sequence_config.max_turns):
            raise ValueError("invalid saved observation history lists")
        for value in history + lengths:
            if not isinstance(value, Tensor) or value.device.type != "cpu" or value.dtype != torch.long:
                raise ValueError("saved observation history must use CPU long tensors")
        brain_payload = payload["brain_state"]
        if not isinstance(brain_payload, dict) or any(
                not isinstance(value, Tensor) or value.device.type != "cpu"
                for value in brain_payload.values()):
            raise ValueError("saved regional activity must use CPU tensors")
        brain_state = self.brain.state_from_dict(brain_payload)
        device = next(self.parameters()).device
        state = RegionalSequenceState(brain_state,
            tuple(value.detach().to(device).clone() for value in history),
            tuple(value.detach().to(device).clone() for value in lengths))
        self._validate_sequence_state(state, brain_state.previous_prefrontal.shape[0])
        return state


def build_regional_sequence_student(seed, device="cpu", sequence_config=None):
    """Initialize a fresh candidate without consuming caller RNG state."""
    _integer("seed", seed)
    if seed >= 2**63:
        raise ValueError("seed must be < 2**63")
    with _INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = RegionalSequenceStudent(sequence_config)
    return model.to(device)
