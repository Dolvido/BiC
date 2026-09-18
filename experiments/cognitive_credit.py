"""Domain-independent, training-only decision credit at prefrontal activity.

The extra readout has no route into policy logits, replies, or persistent state.
Both experimental arms instantiate it; only its training-loss weight differs.
Previously frozen experiment/core sources are reused without modification.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from brain_in_computer.dialogue_student import ACK, ByteCodec, _run_turn, _select
from brain_in_computer.learning_student import _INITIALIZATION_LOCK, _check_finite_tree, _integer
from experiments.cognitive_student import CognitiveStudent
from experiments.train_cognitive import CognitiveTrainer


class CreditStudent(CognitiveStudent):
    """A disposable shared decision head; the official learner is unchanged."""

    def __init__(self, *, memory_mode="recurrent"):
        super().__init__(memory_mode=memory_mode)
        self.credit_head = nn.Linear(self.brain.config.hidden_size, self.brain.config.num_actions)

    def parameter_counts(self):
        return {**super().parameter_counts(),
                "decision_credit": sum(parameter.numel() for parameter in self.credit_head.parameters())}

    def forward_with_state(self, *args, **kwargs):
        output, state = super().forward_with_state(*args, **kwargs)
        output["auxiliary_logits"] = self.credit_head(output["region_activity"]["prefrontal_cortex"][:, -1])
        return output, state


def build_credit_student(seed, device="cpu", memory_mode="recurrent"):
    """Initialize the exact seeded base learner, then its independent readout."""
    _integer("seed", seed)
    if seed >= 2**63:
        raise ValueError("seed must be < 2**63")
    with _INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = CreditStudent(memory_mode=memory_mode)
    return model.to(device)


def balanced_query_loss(logits, labels):
    """Equal weight per present query class; acknowledgement is excluded."""
    losses = [F.cross_entropy(logits[labels == target], labels[labels == target])
              for target in range(ACK) if bool((labels == target).any())]
    if not losses:
        raise ValueError("training episodes need a scored query")
    return torch.stack(losses).mean()


class CreditTrainer(CognitiveTrainer):
    """Identical family streams and objectives plus optional short-path credit.

    The inherited constructor authenticates canonical training banks and creates
    the same independent family samplers. Encoding is model-independent byte
    preprocessing; replacing the fresh learner does not alter those tensors.
    Optimizer state and auxiliary parameters are saved by inherited snapshot().
    """

    def __init__(self, banks, *, seed=2201, memory_mode="recurrent", device="cpu",
                 batch_size=64, learning_rate=.003, aux_weight=0., payload=None):
        if (type(aux_weight) not in (int, float) or not math.isfinite(aux_weight)
                or aux_weight < 0):
            raise ValueError("aux_weight must be finite and nonnegative")
        super().__init__(banks, seed=seed, memory_mode=memory_mode, device=device,
                         batch_size=batch_size, learning_rate=learning_rate)
        self.aux_weight = float(aux_weight)
        self.model = build_credit_student(seed, device=device, memory_mode=memory_mode)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.recipe = {**self.recipe, "model": "bic-cognitive-credit-v1", "aux_weight": self.aux_weight}
        if payload is not None:
            _check_finite_tree(payload, "training_checkpoint")
            if payload.get("recipe") != self.recipe:
                raise ValueError("checkpoint training recipe differs")
            self.model.load_state_dict(payload["weights"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer"])
            if set(payload["samplers"]) != set(self.generators):
                raise ValueError("checkpoint family sampler set differs")
            for family, generator in self.generators.items():
                generator.set_state(payload["samplers"][family])
            self.updates = payload["updates"]
            self.family_updates = dict(payload["family_updates"])

    def step(self, family):
        if family not in self.encoded:
            raise ValueError("family not admitted into this training phase")
        device = next(self.model.parameters()).device
        pairs = torch.randint(len(self.banks[family]) // 2, (self.batch_size // 2,),
                              generator=self.generators[family])
        indices = (pairs[:, None] * 2 + torch.arange(2)[None]).flatten().to(device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        state = None
        logits, auxiliary, labels, replies, observations = [], [], [], [], []
        for full_batch in self.encoded[family]:
            batch = _select(full_batch, indices)
            output, state = _run_turn(self.model, batch, state)
            logits.append(output["logits"][:, -1])
            auxiliary.append(output["auxiliary_logits"])
            labels.append(batch["targets"])
            replies.append(F.cross_entropy(output["language_logits"].flatten(0, 1),
                batch["reply_targets"].flatten(), ignore_index=ByteCodec.PAD))
            observations.append(F.cross_entropy(output["observation_language_logits"][:, :-1].flatten(0, 1),
                batch["text_ids"][:, 1:].flatten(), ignore_index=ByteCodec.PAD))
        logits, auxiliary, labels = torch.cat(logits), torch.cat(auxiliary), torch.cat(labels)
        action = balanced_query_loss(logits, labels)
        if bool((labels == ACK).any()):
            action = action + .25 * F.cross_entropy(logits[labels == ACK], labels[labels == ACK])
        auxiliary_loss = balanced_query_loss(auxiliary, labels)
        reply, observation = torch.stack(replies).mean(), torch.stack(observations).mean()
        loss = action + .1 * reply + .1 * observation + self.aux_weight * auxiliary_loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite cognitive training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        self.family_updates[family] += 1
        query = labels.ne(ACK)
        return {"loss": loss.item(), "action_loss": action.item(), "reply_loss": reply.item(),
            "observation_language_loss": observation.item(), "family": family,
            "aux_weight": self.aux_weight, "auxiliary_query_loss": auxiliary_loss.item(),
            "auxiliary_query_accuracy": float(auxiliary.detach().argmax(-1)[query].eq(labels[query]).float().mean())}
