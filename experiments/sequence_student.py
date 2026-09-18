"""Untrained causal sequence diagnostic, deliberately outside regional BiC.

The encoder receives observation-only BOS/byte/EOS sequences. Reply prefixes
enter a separate decoder and cannot affect actions or observation history.
No parser, task identifier, label, oracle or pretrained model enters inference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from brain_in_computer.language import ByteCodec, InferiorFrontal, LanguageConfig
from brain_in_computer.learning_student import _INITIALIZATION_LOCK, _integer


@dataclass(frozen=True)
class SequenceConfig:
    width: int = 96
    layers: int = 4
    heads: int = 4
    feedforward: int = 384
    max_positions: int = 1024
    max_turns: int = 6
    max_input_bytes: int = 128
    max_output_bytes: int = 32

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads:
            raise ValueError("width must divide evenly across attention heads")
        if self.max_positions < 2:
            raise ValueError("position capacity must fit at least BOS and EOS")


class SequenceStudent(nn.Module):
    """Causal attention over visible observations, with one shared reply policy.

    All batch rows contain the same number of complete turns, 1..max_turns.
    The full training batch has six. A single causal pass supplies every EOS
    representation. Outputs named ``logits`` and ``language_logits`` concern
    the common four-way response interface; there are no family-specific heads.

    This is a diagnostic monolithic learner, not a regional BiC promotion. It
    owns no mutable session history: callers supply the observation prefix.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is not None and not isinstance(config, SequenceConfig):
            raise ValueError("config must be SequenceConfig")
        self.config = config or SequenceConfig()
        c = self.config
        self.codec = ByteCodec(c.max_input_bytes)
        self.tokens = nn.Embedding(ByteCodec.VOCAB_SIZE, c.width, padding_idx=ByteCodec.PAD)
        self.positions = nn.Embedding(c.max_positions, c.width)
        # A tied byte readout must not inherit Embedding's unit-scale weights:
        # they otherwise create extreme next-byte logits before learning.
        nn.init.normal_(self.tokens.weight, mean=0., std=.02)
        nn.init.normal_(self.positions.weight, mean=0., std=.02)
        with torch.no_grad():
            self.tokens.weight[ByteCodec.PAD].zero_()
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(c.width, c.heads,
            c.feedforward, dropout=0., activation="relu", batch_first=True, norm_first=True)
            for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.width)
        self.observation_head = nn.Linear(c.width, ByteCodec.VOCAB_SIZE)
        self.observation_head.weight = self.tokens.weight
        self.action_head = nn.Linear(c.width, 4)
        self.reply_context = nn.Sequential(nn.Linear(c.width, 128), nn.LayerNorm(128), nn.Tanh())
        self.inferior_frontal = InferiorFrontal(LanguageConfig(hidden_size=128,
            embedding_size=32, max_input_bytes=c.max_input_bytes,
            max_output_bytes=c.max_output_bytes))

    def parameter_counts(self):
        # The observation readout reuses the input embedding; count it once.
        return {**{name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in (("token_embedding", self.tokens), ("position_embedding", self.positions),
                ("causal_blocks", self.blocks), ("final_norm", self.norm),
                ("action_head", self.action_head), ("reply_context", self.reply_context),
                ("reply_decoder", self.inferior_frontal))},
            "observation_readout_bias": self.observation_head.bias.numel()}

    def _validate_inputs(self, token_ids, eos_positions, valid_mask, lengths):
        reference = next(self.parameters())
        if (not isinstance(token_ids, Tensor) or token_ids.ndim != 2 or token_ids.shape[0] < 1
                or not 2 <= token_ids.shape[1] <= self.config.max_positions
                or token_ids.dtype != torch.long or token_ids.device != reference.device):
            raise ValueError("token_ids must be nonempty [batch, positions] long on the model device")
        if not ((token_ids >= 0) & (token_ids < ByteCodec.VOCAB_SIZE)).all():
            raise ValueError("token_ids contain invalid byte vocabulary IDs")
        batch, width = token_ids.shape
        valid = token_ids.ne(ByteCodec.PAD)
        if ((~valid).cumsum(dim=1).gt(0) & valid).any():
            raise ValueError("observations must use contiguous right padding")
        actual_lengths = valid.sum(dim=1)
        if valid_mask is not None and (not isinstance(valid_mask, Tensor)
                or valid_mask.shape != token_ids.shape or valid_mask.dtype != torch.bool
                or valid_mask.device != reference.device or not torch.equal(valid_mask, valid)):
            raise ValueError("valid_mask must match non-PAD observation positions")
        if lengths is not None and (not isinstance(lengths, Tensor) or lengths.shape != (batch,)
                or lengths.dtype != torch.long or lengths.device != reference.device
                or not torch.equal(lengths, actual_lengths)):
            raise ValueError("lengths must match complete non-PAD observations")
        if (not isinstance(eos_positions, Tensor) or eos_positions.ndim != 2
                or eos_positions.shape[0] != batch or eos_positions.dtype != torch.long
                or eos_positions.device != reference.device
                or not 1 <= eos_positions.shape[1] <= self.config.max_turns):
            raise ValueError("eos_positions must be [batch, 1..max_turns] long on the model device")
        if (not ((eos_positions >= 1) & (eos_positions < width)).all()
                or (eos_positions[:, 1:] <= eos_positions[:, :-1]).any()
                or not torch.equal(eos_positions[:, -1], actual_lengths - 1)):
            raise ValueError("EOS positions must be increasing and end the observed sequence")
        beginnings = torch.cat((torch.zeros_like(eos_positions[:, :1]), eos_positions[:, :-1] + 1), dim=1)
        byte_lengths = eos_positions - beginnings - 1
        if not ((byte_lengths >= 0) & (byte_lengths <= self.config.max_input_bytes)).all():
            raise ValueError("utterance byte lengths exceed configured limits")
        expected_eos, expected_bos = torch.zeros_like(valid), torch.zeros_like(valid)
        expected_eos.scatter_(1, eos_positions, True)
        expected_bos.scatter_(1, beginnings, True)
        if (not torch.equal(token_ids.eq(ByteCodec.EOS), expected_eos)
                or not torch.equal(token_ids.eq(ByteCodec.BOS), expected_bos)):
            raise ValueError("each observed turn must be exactly BOS, bytes, EOS")
        return valid

    def _validate_decoder(self, decoder_input_ids, batch, turns):
        reference = next(self.parameters())
        if (not isinstance(decoder_input_ids, Tensor) or decoder_input_ids.ndim != 3
                or decoder_input_ids.shape[:2] != (batch, turns)
                or not 1 <= decoder_input_ids.shape[2] <= self.config.max_output_bytes + 1
                or decoder_input_ids.dtype != torch.long or decoder_input_ids.device != reference.device):
            raise ValueError("decoder_input_ids must be [batch, turns, reply_prefix] long on the model device")
        ids = decoder_input_ids.flatten(0, 1)
        if (not ((ids >= 0) & (ids < ByteCodec.VOCAB_SIZE)).all()
                or not ids[:, 0].eq(ByteCodec.BOS).all() or ids[:, 1:].eq(ByteCodec.BOS).any()):
            raise ValueError("reply prefixes require exactly one initial BOS and valid byte IDs")
        padded = ids.eq(ByteCodec.PAD).cumsum(dim=1).gt(0)
        after_eos = (ids.eq(ByteCodec.EOS).cumsum(dim=1) - ids.eq(ByteCodec.EOS).long()).gt(0)
        if (padded & ids.ne(ByteCodec.PAD)).any() or (after_eos & ids.ne(ByteCodec.PAD)).any():
            raise ValueError("reply prefixes require right padding and only PAD after EOS")

    def forward(self, token_ids, eos_positions, decoder_input_ids, *, valid_mask=None, lengths=None):
        """Read observation bytes causally; decode each reply separately.

        Returns action logits [B,T,4], reply logits [B,T,R,259], observation
        next-byte logits [B,L,259], reply contexts [B,T,128], and context states
        [B,L,width]. Predictions at right-padded observation positions should be
        ignored by the objective; context_states there are explicitly zeroed.
        """
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
        turn_context = context.gather(1, eos_positions[:, :, None].expand(-1, -1, self.config.width))
        production = self.reply_context(turn_context)
        reply_logits = self.inferior_frontal(decoder_input_ids.flatten(0, 1), production.flatten(0, 1))
        return {"logits": self.action_head(turn_context),
            "language_logits": reply_logits.reshape(batch, turns, decoder_input_ids.shape[2], ByteCodec.VOCAB_SIZE),
            "observation_language_logits": self.observation_head(context),
            "production_context": production, "context_states": context}


def build_sequence_student(seed, device="cpu", config=None):
    """Initialize without consuming caller RNG state or loading pretrained data."""
    _integer("seed", seed)
    if seed >= 2**63:
        raise ValueError("seed must be < 2**63")
    with _INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = SequenceStudent(config)
    return model.to(device)
