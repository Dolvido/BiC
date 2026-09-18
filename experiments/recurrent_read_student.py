"""Tied iterative query reads over the sequence learner's causal byte memory.

The ordinary encoder and observation-byte objective are unchanged. Extra reads
reuse the last encoder block's parameters on EOS queries, with fixed pre-last
block memory. They add computation, not parameters or privileged input. This is
an experimental learner; shared features do not guarantee correct decisions.
"""
from __future__ import annotations

import torch

from brain_in_computer.language import ByteCodec
from brain_in_computer.learning_student import _INITIALIZATION_LOCK, _integer
from experiments.sequence_student import SequenceStudent


ARCHITECTURE = "sequence-tied-recurrent-read-v1"


class RecurrentReadStudent(SequenceStudent):
    """One ordinary encoder pass followed by ``read_passes - 1`` EOS reads.

    A pass count of one calls the unchanged baseline forward. No module or
    parameter is added, so the baseline state dict is strictly compatible.
    Callers must record read_passes separately from that state dict.
    """

    def __init__(self, config=None, *, read_passes=4):
        _integer("read_passes", read_passes, minimum=1)
        super().__init__(config)
        self.read_passes = read_passes

    def forward(self, token_ids, eos_positions, decoder_input_ids, *, valid_mask=None, lengths=None):
        if self.read_passes == 1:
            return super().forward(token_ids, eos_positions, decoder_input_ids,
                                   valid_mask=valid_mask, lengths=lengths)

        valid = self._validate_inputs(token_ids, eos_positions, valid_mask, lengths)
        batch, width = token_ids.shape
        turns = eos_positions.shape[1]
        self._validate_decoder(decoder_input_ids, batch, turns)
        positions = torch.arange(width, device=token_ids.device)
        context = self.tokens(token_ids) + self.positions(positions)[None, :, :]
        causal = torch.ones(width, width, dtype=torch.bool, device=token_ids.device).triu(1)
        for block in self.blocks[:-1]:
            context = block(context, src_mask=causal, src_key_padding_mask=~valid, is_causal=True)

        last = self.blocks[-1]
        # Match the pre-norm last block's key/value input. Keep its graph: reply
        # and action gradients must reach the byte encoder through every read.
        memory = last.norm1(context)
        context = last(context, src_mask=causal, src_key_padding_mask=~valid, is_causal=True)
        queries = context.gather(1, eos_positions[:, :, None].expand(-1, -1, self.config.width))
        # EOS positions differ between examples. A triangular T x L mask would
        # be wrong: each turn can see every valid byte through its own EOS.
        blocked = (positions[None, None, :] > eos_positions[:, :, None]) | ~valid[:, None, :]
        blocked = blocked[:, None, :, :].expand(-1, self.config.heads, -1, -1)
        blocked = blocked.reshape(batch * self.config.heads, turns, width)
        for _ in range(self.read_passes - 1):
            attended = last.self_attn(last.norm1(queries), memory, memory,
                                      attn_mask=blocked, need_weights=False)[0]
            queries = queries + last.dropout1(attended)
            feedforward = last.linear2(last.dropout(last.activation(last.linear1(last.norm2(queries)))))
            queries = queries + last.dropout2(feedforward)

        # Keep the original full-byte context for the unchanged observation
        # auxiliary. Refined EOS states are exclusively the shared decision input.
        context = self.norm(context).masked_fill(~valid[:, :, None], 0.)
        turn_context = self.norm(queries)
        production = self.reply_context(turn_context)
        reply_logits = self.inferior_frontal(decoder_input_ids.flatten(0, 1), production.flatten(0, 1))
        return {"logits": self.action_head(turn_context),
                "language_logits": reply_logits.reshape(batch, turns, decoder_input_ids.shape[2], ByteCodec.VOCAB_SIZE),
                "observation_language_logits": self.observation_head(context),
                "production_context": production, "context_states": context}


def build_recurrent_read_student(seed, device="cpu", config=None, *, read_passes=4):
    """Use the baseline initialization sequence without consuming caller RNG."""
    _integer("seed", seed)
    if seed >= 2**63:
        raise ValueError("seed must be < 2**63")
    _integer("read_passes", read_passes, minimum=1)
    with _INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = RecurrentReadStudent(config, read_passes=read_passes)
    return model.to(device)
