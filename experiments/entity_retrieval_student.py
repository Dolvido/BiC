"""Causal entity retrieval shared by auxiliary teaching and native answers.

Only observation bytes and their turn boundaries enter forward. Every fixed
alias is queried at every prefix. Labels, parsers and hard entity routing are
absent. This is prefix retrieval, not persistent or symbolic state memory.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from brain_in_computer.language import ByteCodec
from brain_in_computer.learning_student import _INITIALIZATION_LOCK, _integer
from experiments.shared_state_student import SharedStateStudent, STATE_ALIAS_WIDTH
from experiments.shared_state_student import source_hashes as shared_source_hashes


ARCHITECTURE = "sequence-causal-entity-retrieval-v1"


def source_hashes():
    result = shared_source_hashes()
    result["experiments/entity_retrieval_student.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


class EntityRetrievalStudent(SharedStateStudent):
    """Shared alias queries retrieve separate vectors from causal byte memory.

    Production forward always retrieves all twelve entities in vocabulary order.
    Standalone retrieval permits a complete permutation only, for checking slot
    equivariance. Byte keys/values are projected once, without turn/alias copies.
    Native answers attend the entity vectors, while the unchanged observation
    readout uses the original normalized full-byte states.
    """

    def __init__(self, config=None):
        # Construct the complete shared-state baseline before any new modules.
        super().__init__(config)
        width = self.config.width
        self.entity_alias_query = nn.Linear(STATE_ALIAS_WIDTH, width)
        self.entity_eos_query = nn.Linear(width, width)
        self.entity_key = nn.Linear(width, width)
        self.entity_value = nn.Linear(width, width)
        self.entity_readout = nn.Linear(width, width)
        self.entity_norm = nn.LayerNorm(width)
        self.native_entity_query = nn.Linear(width, width)
        self.native_entity_key = nn.Linear(width, width)
        self.native_entity_value = nn.Linear(width, width)
        self.native_entity_readout = nn.Linear(width, width)
        self.native_entity_norm = nn.LayerNorm(width)

    def parameter_counts(self):
        return {**super().parameter_counts(),
            "entity_retrieval": sum(p.numel() for module in (self.entity_alias_query,
                self.entity_eos_query, self.entity_key, self.entity_value,
                self.entity_readout, self.entity_norm) for p in module.parameters()),
            "native_entity_fusion": sum(p.numel() for module in (self.native_entity_query,
                self.native_entity_key, self.native_entity_value, self.native_entity_readout,
                self.native_entity_norm) for p in module.parameters())}

    def _turn_states(self, context_states, eos_positions):
        reference = self.state_alias_embedding.weight
        if (not isinstance(context_states, torch.Tensor) or context_states.ndim != 3
                or context_states.shape[0] < 1 or not 2 <= context_states.shape[1] <= self.config.max_positions
                or context_states.shape[2] != self.config.width
                or context_states.device != reference.device or context_states.dtype != reference.dtype):
            raise ValueError("model-device floating byte states with the configured width required")
        if (not isinstance(eos_positions, torch.Tensor) or eos_positions.ndim != 2
                or eos_positions.shape[0] != context_states.shape[0]
                or not 1 <= eos_positions.shape[1] <= self.config.max_turns
                or eos_positions.dtype != torch.long or eos_positions.device != reference.device
                or bool((eos_positions < 1).any()) or bool((eos_positions >= context_states.shape[1]).any())
                or bool((eos_positions[:, 1:] <= eos_positions[:, :-1]).any())):
            raise ValueError("increasing per-example EOS positions inside byte states required")
        return context_states.gather(1, eos_positions[:, :, None].expand(-1, -1, self.config.width))

    def _alias_indices(self, alias_indices):
        reference = self.state_alias_embedding.weight
        expected = torch.arange(reference.shape[0], device=reference.device, dtype=torch.long)
        if alias_indices is None:
            return expected
        if (not isinstance(alias_indices, torch.Tensor) or alias_indices.shape != expected.shape
                or alias_indices.dtype != torch.long or alias_indices.device != reference.device
                or not torch.equal(alias_indices.sort().values, expected)):
            raise ValueError("alias indices must be a complete fixed-vocabulary permutation")
        return alias_indices

    def retrieve_entities(self, context_states, eos_positions, *, alias_indices=None):
        """Return [B,T,12,width]; each query sees bytes only through its EOS.

        A standalone caller must supply causal byte states. EOS positions imply
        the contiguous valid prefix; all trailing positions are masked, even if
        their supplied feature values are nonzero. Production forward separately
        validates the actual BOS/byte/EOS token grammar and padding.
        """
        turns = self._turn_states(context_states, eos_positions)
        indices = self._alias_indices(alias_indices)
        batch, length, width = context_states.shape
        count, aliases, heads = turns.shape[1], indices.numel(), self.config.heads
        head_width = width // heads
        query = (self.entity_eos_query(turns)[:, :, None, :]
                 + self.entity_alias_query(self.state_alias_embedding(indices))[None, None, :, :])
        q = query.reshape(batch, count * aliases, heads, head_width).transpose(1, 2)
        # K/V remain [B,H,L,D/H]; neither is repeated across turns or aliases.
        k = self.entity_key(context_states).reshape(batch, length, heads, head_width).transpose(1, 2)
        v = self.entity_value(context_states).reshape(batch, length, heads, head_width).transpose(1, 2)
        positions = torch.arange(length, device=context_states.device)
        valid = positions[None, :] <= eos_positions[:, -1, None]
        blocked = ((positions[None, None, :] > eos_positions[:, :, None]) | ~valid[:, None, :])
        blocked = blocked[:, :, None, :].expand(-1, -1, aliases, -1).reshape(batch, count * aliases, length)
        scores = torch.matmul(q, k.transpose(-2, -1)) * head_width ** -.5
        attention = scores.masked_fill(blocked[:, None, :, :], float("-inf")).softmax(dim=-1)
        read = torch.matmul(attention, v).transpose(1, 2).reshape(batch, count, aliases, width)
        return self.entity_norm(query + self.entity_readout(read))

    def _fuse_entities(self, turn_states, entity_states):
        batch, turns, aliases, width = entity_states.shape
        heads, head_width = self.config.heads, width // self.config.heads
        q = self.native_entity_query(turn_states).reshape(batch * turns, 1, heads, head_width).transpose(1, 2)
        k = self.native_entity_key(entity_states).reshape(batch * turns, aliases, heads, head_width).transpose(1, 2)
        v = self.native_entity_value(entity_states).reshape(batch * turns, aliases, heads, head_width).transpose(1, 2)
        attention = (torch.matmul(q, k.transpose(-2, -1)) * head_width ** -.5).softmax(dim=-1)
        read = torch.matmul(attention, v).transpose(1, 2).reshape(batch, turns, width)
        return self.native_entity_norm(turn_states + self.native_entity_readout(read))

    def _decode_entities(self, entity_states, *, alias_indices=None):
        indices = self._alias_indices(alias_indices)
        reference = self.state_alias_embedding.weight
        if (not isinstance(entity_states, torch.Tensor) or entity_states.ndim != 4
                or entity_states.shape[0] < 1 or not 1 <= entity_states.shape[1] <= self.config.max_turns
                or entity_states.shape[2:] != (indices.numel(), self.config.width)
                or entity_states.device != reference.device or entity_states.dtype != reference.dtype):
            raise ValueError("model-device [batch,turns,12,width] entity states required")
        names = self.state_alias_embedding(indices)[None, None, :, :].expand(*entity_states.shape[:2], -1, -1)
        return self.state_decoder(torch.cat((entity_states, names), dim=-1))

    def state_logits(self, context_states, eos_positions, *, alias_indices=None):
        """Standalone state check; training should consume cached forward output."""
        entities = self.retrieve_entities(context_states, eos_positions, alias_indices=alias_indices)
        return self._decode_entities(entities, alias_indices=alias_indices)

    def state_logits_from_output(self, output, eos_positions):
        """Decode cached forward entities without repeating byte retrieval."""
        if (not isinstance(output, dict) or "context_states" not in output or "entity_states" not in output
                or not isinstance(output.get("entity_eos_positions"), torch.Tensor)):
            raise ValueError("forward output including cached entities and EOS identity required")
        turns = self._turn_states(output["context_states"], eos_positions)
        if not torch.equal(output["entity_eos_positions"], eos_positions):
            raise ValueError("cached entity EOS positions differ from requested positions")
        entities = output["entity_states"]
        if not isinstance(entities, torch.Tensor) or entities.shape[:2] != turns.shape[:2]:
            raise ValueError("cached entity states must match the requested batch and turn count")
        return self._decode_entities(entities)

    def forward(self, token_ids, eos_positions, decoder_input_ids, *, valid_mask=None, lengths=None):
        valid = self._validate_inputs(token_ids, eos_positions, valid_mask, lengths)
        batch, width = token_ids.shape
        turns = eos_positions.shape[1]
        self._validate_decoder(decoder_input_ids, batch, turns)
        positions = torch.arange(width, device=token_ids.device)
        context = self.tokens(token_ids) + self.positions(positions)[None, :, :]
        causal = torch.ones(width, width, dtype=torch.bool, device=token_ids.device).triu(1)
        for block in self.blocks:
            context = block(context, src_mask=causal, src_key_padding_mask=~valid, is_causal=True)
        context = self.norm(context).masked_fill(~valid[:, :, None], 0.)
        entity_states = self.retrieve_entities(context, eos_positions)
        turn_context = self._fuse_entities(self._turn_states(context, eos_positions), entity_states)
        production = self.reply_context(turn_context)
        reply_logits = self.inferior_frontal(decoder_input_ids.flatten(0, 1), production.flatten(0, 1))
        return {"logits": self.action_head(turn_context),
            "language_logits": reply_logits.reshape(batch, turns, decoder_input_ids.shape[2], ByteCodec.VOCAB_SIZE),
            "observation_language_logits": self.observation_head(context),
            "production_context": production, "context_states": context, "entity_states": entity_states,
            "entity_eos_positions": eos_positions.detach().clone()}


def build_entity_retrieval_student(seed, device="cpu", config=None):
    """Common initialization first, extra modules afterward; preserve caller RNG."""
    _integer("seed", seed)
    if seed >= 2**63:
        raise ValueError("seed must be < 2**63")
    with _INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = EntityRetrievalStudent(config)
    return model.to(device)
