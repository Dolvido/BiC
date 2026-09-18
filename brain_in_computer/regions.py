"""Small, separately parameterized networks with brain-inspired functional roles.

These are engineering abstractions, not models of cells or anatomical fidelity.
Every inter-region input is supplied explicitly by :class:`Brain`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def mlp(input_size: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size), nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), nn.Tanh(),
    )


class SensoryCortex(nn.Module):
    """Encode a sensory channel with delayed prefrontal feedback."""

    def __init__(self, input_size: int, hidden_size: int, symbols: int):
        super().__init__()
        self.encoder = mlp(input_size, hidden_size)
        self.feedback = nn.Linear(hidden_size, hidden_size, bias=False)
        self.classifier = nn.Linear(hidden_size, symbols)

    def forward(self, observation: Tensor, prefrontal: Tensor) -> tuple[Tensor, Tensor]:
        activity = torch.tanh(self.encoder(observation) + 0.2 * self.feedback(prefrontal))
        return activity, self.classifier(activity)


class SomatosensoryCortex(nn.Module):
    """Combine body state, external feedback, and previous motor activity."""

    def __init__(self, body_size: int, hidden_size: int):
        super().__init__()
        self.encoder = mlp(body_size + 2 + hidden_size, hidden_size)

    def forward(self, body: Tensor, feedback: Tensor, previous_motor: Tensor) -> Tensor:
        return self.encoder(torch.cat((body, feedback, previous_motor), dim=-1))


class TemporalLanguage(nn.Module):
    """Embed a small instruction vocabulary; this is not natural language understanding."""

    def __init__(self, vocabulary_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.encoder = mlp(2 * hidden_size, hidden_size)

    def forward(self, token: Tensor, prefrontal: Tensor) -> Tensor:
        return self.encoder(torch.cat((self.embedding(token), prefrontal), dim=-1))


class ParietalAssociation(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fusion = mlp(4 * hidden_size, hidden_size)

    def forward(self, visual: Tensor, auditory: Tensor, body: Tensor, language: Tensor) -> Tensor:
        return self.fusion(torch.cat((visual, auditory, body, language), dim=-1))


class Amygdala(nn.Module):
    """Learn a salience signal; no claim of emotion or subjective experience."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.salience = mlp(3 * hidden_size, hidden_size)

    def forward(self, visual: Tensor, auditory: Tensor, body: Tensor) -> Tensor:
        return self.salience(torch.cat((visual, auditory, body), dim=-1))


class Hypothalamus(nn.Module):
    """Generate bounded gain control from encoded body state."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.drive = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())

    def forward(self, body: Tensor) -> Tensor:
        return self.drive(body)


class Thalamus(nn.Module):
    """Content-dependent routing over five explicit incoming pathways."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Linear(2 * hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.salience_bias = nn.Linear(hidden_size, 5)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(hidden_size)

    def forward(
        self, sources: Tensor, prefrontal: Tensor, language: Tensor,
        salience: Tensor, arousal: Tensor,
    ) -> Tensor:
        query = self.query(torch.cat((prefrontal, language), dim=-1))
        scores = torch.einsum("bmh,bh->bm", self.key(sources), query) / self.scale
        weights = (scores + self.salience_bias(salience)).softmax(dim=-1)
        routed = torch.einsum("bm,bmh->bh", weights, self.value(sources))
        return torch.tanh(self.output(routed)) * (0.5 + arousal)


class Hippocampus(nn.Module):
    """Differentiable, bounded, within-episode key/value memory.

    Writes retain their computation graph for training. The caller supplies only
    earlier writes; the current write is appended before reading. Nothing persists
    between independent forward calls, and no task labels are stored here.
    """

    def __init__(self, hidden_size: int, memory_slots: int):
        super().__init__()
        self.memory_slots = memory_slots
        self.write = nn.Linear(3 * hidden_size, hidden_size)
        self.write_gate = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(3 * hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(hidden_size)

    def forward(
        self, association: Tensor, language: Tensor, body: Tensor,
        prefrontal: Tensor, salience: Tensor, memory: tuple[Tensor, ...],
        *, external_context: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        candidate = torch.tanh(self.write(torch.cat((association, language, body), dim=-1)))
        candidate = candidate * torch.sigmoid(self.write_gate(salience))
        updated = (*memory, candidate)[-self.memory_slots:]
        bank = torch.stack(updated, dim=1)
        query = self.query(torch.cat((association, language, prefrontal), dim=-1))
        scores = torch.einsum("bmh,bh->bm", self.key(bank), query) / self.scale
        retrieved = torch.einsum("bm,bmh->bh", scores.softmax(dim=-1), self.value(bank))
        # Optional persistent recall shares the learned hippocampal output
        # transformation. It supplies evidence, never a selected motor action.
        if external_context is not None:
            retrieved = retrieved + external_context
        return torch.tanh(self.output(retrieved)), updated


class PrefrontalCortex(nn.Module):
    """Recurrent working state, updated once per observation step."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.working_memory = nn.GRUCell(6 * hidden_size, hidden_size)

    def forward(
        self, routed: Tensor, memory: Tensor, language: Tensor,
        previous_cerebellum: Tensor, salience: Tensor, arousal: Tensor, state: Tensor,
    ) -> Tensor:
        inputs = torch.cat((routed, memory, language, previous_cerebellum, salience, arousal), dim=-1)
        return self.working_memory(inputs, state)


class BasalGanglia(nn.Module):
    """Action-selection features, action gates, and a trainable value head."""

    def __init__(self, hidden_size: int, actions: int):
        super().__init__()
        self.selector = mlp(3 * hidden_size, hidden_size)
        self.action_gate = nn.Linear(hidden_size, actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, prefrontal: Tensor, memory: Tensor, salience: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        activity = self.selector(torch.cat((prefrontal, memory, salience), dim=-1))
        return activity, torch.tanh(self.action_gate(activity)), self.value_head(activity).squeeze(-1)


class MotorCortex(nn.Module):
    def __init__(self, hidden_size: int, actions: int):
        super().__init__()
        self.controller = mlp(4 * hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, actions)

    def forward(
        self, prefrontal: Tensor, routed: Tensor, selection: Tensor,
        previous_cerebellum: Tensor, gate: Tensor,
    ) -> tuple[Tensor, Tensor]:
        activity = self.controller(torch.cat((prefrontal, routed, selection, previous_cerebellum), dim=-1))
        logits = self.action_head(activity) * (1.0 + 0.5 * gate)
        return activity, logits


class Cerebellum(nn.Module):
    """Learn next-observation prediction and feed forward-model state back later."""

    def __init__(self, observation_size: int, hidden_size: int):
        super().__init__()
        self.model = mlp(3 * hidden_size + observation_size, hidden_size)
        self.prediction_head = nn.Linear(hidden_size, observation_size)

    def forward(self, prefrontal: Tensor, motor: Tensor, body: Tensor, prediction_error: Tensor) -> tuple[Tensor, Tensor]:
        activity = self.model(torch.cat((prefrontal, motor, body, prediction_error), dim=-1))
        return activity, self.prediction_head(activity)
