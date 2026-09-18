"""Separate learned-utterance-summary candidate; no trained capability claim.

Two shared causal byte blocks encode utterances in parallel, then two causal
turn blocks process their learned EOS summaries. Parameter inventory and seeded
initial values match SequenceStudent. Interaction depth, computation and the
auxiliary byte receptive field change, so this does not isolate compression.
Only observed token IDs and their existing boundaries enter the encoder.

Standard tensor state_dict/load_state_dict remain low-level PyTorch operations,
not checkpoint authentication. Public weight envelopes require this architecture
version. A trainer must additionally bind its complete recipe and source identity.
"""
from __future__ import annotations

from dataclasses import asdict

import torch
from torch import Tensor

from brain_in_computer.language import ByteCodec
from brain_in_computer.learning_student import _INITIALIZATION_LOCK, _integer
from experiments.sequence_student import SequenceConfig, SequenceStudent


ARCHITECTURE = "bic-hierarchical-sequence-v1"
WEIGHT_SCHEMA = "bic-hierarchical-sequence-weights-v1"


class HierarchicalSequenceStudent(SequenceStudent):
    """Observation-only hierarchy with the existing shared action/reply heads.

    All batch rows contain the same 1..max_turns complete utterances. There is
    no mutable session state or parser. The local auxiliary readout cannot see
    earlier utterances; final actions/replies can use preceding EOS summaries.
    Local states must not be treated as the flat learner's full-history states.
    """

    architecture = ARCHITECTURE

    def __init__(self, config=None):
        if config is not None and not isinstance(config, SequenceConfig):
            raise ValueError("config must be SequenceConfig")
        config = config or SequenceConfig()
        if config.layers != 4:
            raise ValueError("hierarchical sequence architecture requires exactly four blocks")
        super().__init__(config)

    def weight_checkpoint(self):
        """Portable architecture-tagged weights; not a complete training state."""
        return {"schema": WEIGHT_SCHEMA, "architecture": ARCHITECTURE,
                "config": asdict(self.config),
                "weights": {name: value.detach().cpu().clone()
                            for name, value in self.state_dict().items()}}

    def load_weight_checkpoint(self, payload):
        """Reject wrong architecture/config before any low-level tensor copying.

        This validates a weight envelope, not its training history or provenance.
        Optimizer/cursor/recipe/source validation belongs to the owning trainer.
        """
        if (type(payload) is not dict or set(payload) != {"schema", "architecture", "config", "weights"}
                or payload["schema"] != WEIGHT_SCHEMA or payload["architecture"] != ARCHITECTURE
                or type(payload["config"]) is not dict
                or any(type(value) is not int for value in payload["config"].values())
                or payload["config"] != asdict(self.config)):
            raise ValueError("hierarchical weight schema/architecture/config differs")
        expected, weights = self.state_dict(), payload["weights"]
        if type(weights) is not dict or set(weights) != set(expected):
            raise ValueError("hierarchical checkpoint parameter names differ")
        for name, reference in expected.items():
            value = weights[name]
            if (not isinstance(value, Tensor) or value.shape != reference.shape
                    or value.dtype != reference.dtype or value.device.type != "cpu"
                    or value.requires_grad or not bool(torch.isfinite(value).all())):
                raise ValueError("hierarchical checkpoint requires finite matching CPU tensors")
        aliases = {}
        for name, parameter in self.named_parameters(remove_duplicate=False):
            first = aliases.setdefault(id(parameter), name)
            if not torch.equal(weights[name], weights[first]):
                raise ValueError("hierarchical checkpoint tied parameter aliases differ")
        self.load_state_dict(weights, strict=True)
        return self

    def forward(self, token_ids, eos_positions, decoder_input_ids, *, valid_mask=None, lengths=None):
        """Return the common objective keys and explicitly named local states.

        local_byte_states [B,L,D] are normalized local causal byte states,
        zeroed on right padding. utterance_summaries [B,T,D] are unnormalized
        local EOS states. turn_context_states [B,T,D] are normalized outputs of
        the causal turn blocks. No ``context_states`` compatibility alias is
        supplied: these byte states have a different receptive field.
        """
        valid = self._validate_inputs(token_ids, eos_positions, valid_mask, lengths)
        batch, width = token_ids.shape
        turns = eos_positions.shape[1]
        self._validate_decoder(decoder_input_ids, batch, turns)
        device, hidden = token_ids.device, self.config.width
        beginnings = torch.cat((torch.zeros_like(eos_positions[:, :1]), eos_positions[:, :-1] + 1), dim=1)
        local_lengths = (eos_positions - beginnings + 1).reshape(-1)
        local_width = int(local_lengths.max())
        offsets = torch.arange(local_width, device=device)
        absolute = beginnings.reshape(-1, 1) + offsets[None, :]
        local_valid = offsets[None, :] < local_lengths[:, None]
        # Repack integer observations, not differentiable hidden states. Each
        # utterance becomes an independent batch row, keeping global positions.
        session = torch.arange(batch, device=device).repeat_interleave(turns)
        packed_ids = token_ids[session[:, None], absolute.clamp_max(width - 1)]
        packed_ids = packed_ids.masked_fill(~local_valid, ByteCodec.PAD)
        position_ids = absolute.masked_fill(~local_valid, 0)
        local = self.tokens(packed_ids) + self.positions(position_ids)
        local_causal = torch.ones(local_width, local_width, dtype=torch.bool, device=device).triu(1)
        for block in self.blocks[:2]:
            local = block(local, src_mask=local_causal, src_key_padding_mask=~local_valid, is_causal=True)

        # EOS indices are unique: no repeated floating-point destinations in
        # repacking/backpropagation. No scatter_add or overwriting scatter is used.
        summary_indices = torch.arange(batch * turns, device=device) * local_width + local_lengths - 1
        summaries = local.reshape(-1, hidden).index_select(0, summary_indices).reshape(batch, turns, hidden)
        turn_context = summaries
        turn_causal = torch.ones(turns, turns, dtype=torch.bool, device=device).triu(1)
        for block in self.blocks[2:]:
            turn_context = block(turn_context, src_mask=turn_causal, is_causal=True)
        turn_context = self.norm(turn_context)

        normalized_local = self.norm(local).masked_fill(~local_valid[:, :, None], 0.)
        positions = torch.arange(width, device=device)[None, :].expand(batch, -1)
        turn_index = (positions[:, :, None] > eos_positions[:, None, :]).sum(dim=-1).clamp_max(turns - 1)
        starts = beginnings.gather(1, turn_index)
        source_index = (torch.arange(batch, device=device)[:, None] * turns + turn_index) * local_width + positions - starts
        flattened = normalized_local.reshape(-1, hidden)
        # Every requested PAD position receives its own constant zero row. This
        # keeps the entire inverse index one-to-one, including padded sessions.
        padding_indices = flattened.shape[0] + torch.arange(batch * width, device=device).reshape(batch, width)
        source_index = torch.where(valid, source_index, padding_indices)
        extended = torch.cat((flattened, local.new_zeros(batch * width, hidden)), dim=0)
        byte_states = extended.index_select(0, source_index.reshape(-1)).reshape(batch, width, hidden)

        production = self.reply_context(turn_context)
        replies = self.inferior_frontal(decoder_input_ids.flatten(0, 1), production.flatten(0, 1))
        return {"logits": self.action_head(turn_context),
                "language_logits": replies.reshape(batch, turns, decoder_input_ids.shape[2], ByteCodec.VOCAB_SIZE),
                "observation_language_logits": self.observation_head(byte_states),
                "production_context": production, "local_byte_states": byte_states,
                "utterance_summaries": summaries, "turn_context_states": turn_context}


def build_hierarchical_sequence_student(seed, device="cpu", config=None):
    """Match the flat model's initial tensors without consuming caller RNG."""
    _integer("seed", seed)
    if seed >= 2**63:
        raise ValueError("seed must be < 2**63")
    with _INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = HierarchicalSequenceStudent(config)
    return model.to(device)
