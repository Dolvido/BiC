"""Training-only readouts for elementary English and causal dialogue memory.

The readouts consume learned activity, never parsed facts. ``causal_targets``
parses independently verified training lessons outside the policy forward pass.
Neither these heads nor their targets belong in inference checkpoints; runners
may save the heads' ordinary state_dict separately to resume their optimizer.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .dialogue_curriculum import (
    ALIASES, COLORS, NUM_TURNS, parse_sentence, validate_dialogue,
)


ROLE_NAMES = ("definition", "rule", "correction", "query")
DEFINITION, RULE, CORRECTION, QUERY = range(4)
NONE_ALIAS = len(ALIASES)
UNKNOWN_COLOR = len(COLORS)
HEAD_NAMES = ("role", "alias", "color", "permitted", "bindings")


def causal_targets(episodes: Sequence[Mapping], device="cpu") -> dict:
    """Return canonical train-only labels, with shape [turn, batch, ...].

    ``role``, ``alias``, ``color``, and ``permitted`` are int64 [6, batch];
    ``bindings`` is int64 [6, batch, 8], in ``ALIASES`` order. Color IDs follow
    ``COLORS``. NONE_ALIAS and UNKNOWN_COLOR denote absent utterance fields or
    knowledge not established yet. Unknown is a supervised category, not a
    missing label. ``masks`` maps each name to an all-true bool tensor of the
    same shape; a runner can mask selected examples without changing labels.

    Utterance labels describe only the current text. Memory labels include the
    current turn's update, but never a later definition or rule correction.
    The input episodes and their observations are never modified.
    """
    if (not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes))
            or not episodes):
        raise ValueError("scaffolding requires a nonempty sequence of train dialogues")
    for episode in episodes:
        if not isinstance(episode, Mapping) or episode.get("split") != "train":
            raise ValueError("scaffolding accepts only train dialogues")
        validate_dialogue(episode)

    rows = {name: [] for name in HEAD_NAMES}
    for episode in episodes:
        values = {name: [] for name in HEAD_NAMES}
        bindings = [UNKNOWN_COLOR] * len(ALIASES)
        permitted = UNKNOWN_COLOR
        for turn in episode["turns"]:
            kind, facts = parse_sentence(turn["text"])
            alias = ALIASES.index(facts["alias"]) if "alias" in facts else NONE_ALIAS
            color = COLORS.index(facts["color"]) if "color" in facts else UNKNOWN_COLOR
            if kind == "definition":
                bindings[alias] = color
            elif kind in ("rule", "correction"):
                permitted = color
            values["role"].append(ROLE_NAMES.index(kind))
            values["alias"].append(alias)
            values["color"].append(color)
            values["permitted"].append(permitted)
            values["bindings"].append(bindings.copy())
        for name in HEAD_NAMES:
            rows[name].append(values[name])
    result = {name: torch.tensor(values, dtype=torch.long, device=device).transpose(0, 1).contiguous()
              for name, values in rows.items()}
    result["masks"] = {name: torch.ones_like(value, dtype=torch.bool)
                       for name, value in result.items()}
    return result


class ScaffoldHeads(nn.Module):
    """Disposable auxiliary classifiers; no facts flow back into policy inputs.

    ``forward(output)`` reads ``concept_context`` [batch, concept_size] and,
    when include_state=True, the final prefrontal and hippocampal activities
    concatenated to [batch, state_size]. Returned logits have shapes [batch,4]
    (role), [batch,9] (alias), [batch,5] (color/permitted), and [batch,8,5]
    (bindings). The module adds no hooks or parameters to the student.

    ``losses(output, targets, turn)`` returns one masked mean cross entropy per
    enabled head. ``loss`` averages those means, giving the binding head the
    same weight as one utterance head. Fully masked heads contribute zero.
    Targets/masks retain a leading six-turn dimension, even after a runner
    samples their batch dimension. Heads and optimizer have ordinary PyTorch
    state_dicts and can be restored independently of the student.
    """

    def __init__(self, *, include_state: bool = True, concept_size: int = 32,
                 state_size: int = 64):
        super().__init__()
        if type(include_state) is not bool:
            raise ValueError("include_state must be boolean")
        if any(type(size) is not int or size < 1 for size in (concept_size, state_size)):
            raise ValueError("head input sizes must be positive integers")
        self.include_state = include_state
        self.role = nn.Linear(concept_size, len(ROLE_NAMES))
        self.alias = nn.Linear(concept_size, len(ALIASES) + 1)
        self.color = nn.Linear(concept_size, len(COLORS) + 1)
        if include_state:
            self.permitted = nn.Linear(state_size, len(COLORS) + 1)
            self.bindings = nn.Linear(state_size, len(ALIASES) * (len(COLORS) + 1))

    def forward(self, output: Mapping) -> dict[str, Tensor]:
        concept = output["concept_context"]
        result = {name: getattr(self, name)(concept) for name in ("role", "alias", "color")}
        if self.include_state:
            activity = output["region_activity"]
            state = torch.cat((activity["prefrontal_cortex"][:, -1],
                               activity["hippocampus"][:, -1]), dim=-1)
            result["permitted"] = self.permitted(state)
            result["bindings"] = self.bindings(state).reshape(-1, len(ALIASES), len(COLORS) + 1)
        return result

    def losses(self, output: Mapping, targets: Mapping, turn: int) -> dict[str, Tensor]:
        """Read activity and score the selected causal turn, without mutation."""
        if type(turn) is not int or not 0 <= turn < NUM_TURNS:
            raise ValueError("turn must be an integer in [0, 6)")
        result = {}
        for name, logits in self(output).items():
            labels, mask = targets[name][turn], targets["masks"][name][turn]
            if labels.shape != logits.shape[:-1] or mask.shape != labels.shape:
                raise ValueError(f"{name} labels and masks must match readout shape")
            if labels.dtype != torch.long or mask.dtype != torch.bool:
                raise ValueError("labels must be int64 and masks boolean")
            # Invalid masked labels cannot enter CE. Zero active labels retain
            # a differentiable zero loss, avoiding NaNs and empty reductions.
            safe_labels = labels.masked_fill(~mask, 0)
            terms = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                    safe_labels.reshape(-1), reduction="none")
            result[name] = (terms * mask.reshape(-1)).sum() / mask.sum().clamp_min(1)
        return result

    def loss(self, output: Mapping, targets: Mapping, turn: int) -> Tensor:
        """Mean of enabled per-head losses; scale externally in the objective."""
        return torch.stack(tuple(self.losses(output, targets, turn).values())).mean()
