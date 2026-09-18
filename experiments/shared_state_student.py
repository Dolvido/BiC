"""Training-only, shared entity-conditioned state supervision for SequenceStudent.

Native forward is inherited unchanged. The auxiliary decoder receives learned
observation states and a fixed alias vocabulary, never parsed state or labels.
Its loss weight and verified targets belong to the training caller.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from brain_in_computer.learning_student import _INITIALIZATION_LOCK, _integer
from experiments.cognitive_curriculum import ALIASES, COLORS
from experiments.sequence_student import SequenceStudent


ARCHITECTURE = "sequence-shared-entity-state-v1"
STATE_CLASS_COUNT = 107
STATE_ALIAS_WIDTH = 32
STATE_LABEL_SCHEMA = {"unknown": 0, "colors": {name: index + 1 for index, name in enumerate(COLORS)},
                      "count_offset": 5, "count_min": 0, "count_max": 99,
                      "switch_off": 105, "switch_on": 106}


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("experiments/shared_state_student.py", "experiments/sequence_student.py",
             "experiments/cognitive_curriculum.py", "brain_in_computer/__init__.py",
             "brain_in_computer/learning_student.py", "brain_in_computer/language.py",
             "brain_in_computer/model.py", "brain_in_computer/regions.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


class SharedStateStudent(SequenceStudent):
    """The baseline policy plus an auxiliary decoder shared across all aliases.

    ``context_states`` must be the already-normalized byte states returned by
    native forward. Every alias uses the same decoder and all 107 classes;
    there is no family mask, per-name head, or auxiliary call during inference.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.state_alias_embedding = nn.Embedding(len(ALIASES), STATE_ALIAS_WIDTH)
        self.state_decoder = nn.Sequential(nn.Linear(self.config.width + STATE_ALIAS_WIDTH, 128),
                                           nn.GELU(), nn.Linear(128, STATE_CLASS_COUNT))

    def parameter_counts(self):
        return {**super().parameter_counts(),
                "state_alias_embedding": sum(value.numel() for value in self.state_alias_embedding.parameters()),
                "state_decoder": sum(value.numel() for value in self.state_decoder.parameters())}

    def state_logits(self, context_states, eos_positions, *, alias_indices=None):
        """Return [B,T,12,107], or a requested ordering of fixed alias indices.

        Production training uses the default full vocabulary. The optional
        index vector permits checking that alias-query order has no effect on
        another query; it never enters native forward or the policy readouts.
        """
        reference = self.state_alias_embedding.weight
        if (not isinstance(context_states, torch.Tensor) or context_states.ndim != 3
                or context_states.shape[0] < 1 or context_states.shape[1] < 1
                or context_states.shape[2] != self.config.width
                or context_states.device != reference.device or context_states.dtype != reference.dtype):
            raise ValueError("model-device floating byte states with the configured width required")
        if (not isinstance(eos_positions, torch.Tensor) or eos_positions.ndim != 2
                or eos_positions.shape[0] != context_states.shape[0]
                or not 1 <= eos_positions.shape[1] <= self.config.max_turns
                or eos_positions.dtype != torch.long or eos_positions.device != context_states.device
                or bool((eos_positions < 0).any()) or bool((eos_positions >= context_states.shape[1]).any())
                or bool((eos_positions[:, 1:] <= eos_positions[:, :-1]).any())):
            raise ValueError("increasing per-example EOS positions inside byte states required")
        if alias_indices is None:
            alias_indices = torch.arange(len(ALIASES), dtype=torch.long, device=context_states.device)
        if (not isinstance(alias_indices, torch.Tensor) or alias_indices.ndim != 1
                or not 1 <= alias_indices.numel() <= len(ALIASES)
                or alias_indices.dtype != torch.long or alias_indices.device != context_states.device
                or bool((alias_indices < 0).any()) or bool((alias_indices >= len(ALIASES)).any())):
            raise ValueError("a nonempty model-device vector of fixed alias vocabulary indices required")
        turn_states = context_states.gather(1, eos_positions[:, :, None].expand(-1, -1, self.config.width))
        queries = self.state_alias_embedding(alias_indices)
        batch, turns = eos_positions.shape
        names = alias_indices.numel()
        features = torch.cat((turn_states[:, :, None, :].expand(-1, -1, names, -1),
                              queries[None, None, :, :].expand(batch, turns, -1, -1)), dim=-1)
        return self.state_decoder(features)


def build_shared_state_student(seed, device="cpu", config=None):
    """Initialize baseline first and auxiliary second, preserving caller RNG."""
    _integer("seed", seed)
    if seed >= 2**63:
        raise ValueError("seed must be < 2**63")
    with _INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = SharedStateStudent(config)
    return model.to(device)
