"""Matched complete-pair optimizer for the diagnostic causal sequence learner.

This is preparation for a new study, not a trained or promoted BiC checkpoint.
Loss components retain the regional learner's per-turn weighting.
"""
from dataclasses import asdict
import math

import torch
from torch.nn import functional as F

from brain_in_computer.language import ByteCodec
from brain_in_computer.learning_student import _check_finite_tree, _cpu_copy
from experiments.cognitive_credit import balanced_query_loss
from experiments.sequence_data import pack_cognitive_episodes
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.train_cognitive import fingerprint_rows


def sequence_objective(output, batch):
    """Equal query-class CE, .25 ACK, .1 reply and .1 input-byte losses."""
    inputs, supervision = batch["inputs"], batch["supervision"]
    labels = supervision["action_targets"]
    logits = output["logits"]
    action = balanced_query_loss(logits.flatten(0, 1), labels.flatten())
    acknowledgement = labels.eq(3)
    if bool(acknowledgement.any()):
        action = action + .25 * F.cross_entropy(logits[acknowledgement], labels[acknowledgement])
    reply_targets = supervision["reply_targets"]
    reply_tokens = F.cross_entropy(output["language_logits"].flatten(0, 2),
        reply_targets.flatten(), ignore_index=ByteCodec.PAD, reduction="none").reshape_as(reply_targets)
    reply_mask = reply_targets.ne(ByteCodec.PAD)
    reply = (reply_tokens.sum(dim=(0, 2)) / reply_mask.sum(dim=(0, 2)).clamp_min(1)).mean()
    observation_targets = supervision["observation_next_byte_targets"]
    token_loss = F.cross_entropy(output["observation_language_logits"].transpose(1, 2),
        observation_targets, ignore_index=ByteCodec.PAD, reduction="none")
    ends = inputs["eos_positions"]
    begins = torch.cat((torch.zeros_like(ends[:, :1]), ends[:, :-1] + 1), dim=1)
    positions = torch.arange(token_loss.shape[1], device=token_loss.device)[None, :]
    observations = []
    for turn in range(ends.shape[1]):
        mask = ((positions >= begins[:, turn:turn + 1]) & (positions < ends[:, turn:turn + 1])
                & observation_targets.ne(ByteCodec.PAD))
        observations.append((token_loss * mask).sum() / mask.sum().clamp_min(1))
    observation = torch.stack(observations).mean()
    return {"loss": action + .1 * reply + .1 * observation,
            "action_loss": action, "reply_loss": reply, "observation_language_loss": observation}


class SequenceTrainer:
    def __init__(self, banks, *, seed=2401, device="cpu", config=None, batch_size=64,
                 learning_rate=.001, payload=None):
        if type(batch_size) is not int or batch_size < 2 or batch_size % 2:
            raise ValueError("batch size must contain complete pairs")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not banks:
            raise ValueError("at least one training family is required")
        self.model = build_sequence_student(seed, device=device, config=config)
        if self.model.config.max_turns < 6:
            raise ValueError("canonical training requires six turns")
        self.banks, self.batch_size = banks, batch_size
        self.encoded = {}
        for family, rows in banks.items():
            if not rows or len(rows) % 2 or any(row["family"] != family for row in rows):
                raise ValueError("canonical banks must contain complete pairs from their named family")
            for index in range(0, len(rows), 2):
                if rows[index]["counterfactual_group"] != rows[index + 1]["counterfactual_group"]:
                    raise ValueError("adjacent rows must be a complete counterfactual pair")
            self.encoded[family] = pack_cognitive_episodes(rows, device=device, training=True,
                max_input_bytes=self.model.config.max_input_bytes,
                max_context_tokens=self.model.config.max_positions,
                max_reply_bytes=self.model.config.max_output_bytes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.generators = {family: torch.Generator().manual_seed(seed + index * 7919)
                           for index, family in enumerate(sorted(banks))}
        self.recipe = {"model": "bic-sequence-diagnostic-v1", "config": asdict(self.model.config),
            "banks": {family: fingerprint_rows(rows) for family, rows in banks.items()},
            "batch_size": batch_size, "learning_rate": learning_rate,
            "objective": "same per-class action and per-turn reply/within-utterance byte weighting"}
        self.updates = 0
        self.family_updates = {family: 0 for family in banks}
        if payload is not None:
            _check_finite_tree(payload, "sequence_checkpoint")
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

    def snapshot(self):
        return {"recipe": self.recipe, "weights": _cpu_copy(self.model.state_dict()),
            "optimizer": _cpu_copy(self.optimizer.state_dict()),
            "samplers": {family: generator.get_state().clone() for family, generator in self.generators.items()},
            "updates": self.updates, "family_updates": dict(self.family_updates)}

    def step(self, family):
        if family not in self.encoded:
            raise ValueError("family not admitted into this training phase")
        pairs = torch.randint(len(self.banks[family]) // 2, (self.batch_size // 2,),
                              generator=self.generators[family])
        device = next(self.model.parameters()).device
        indices = (pairs[:, None] * 2 + torch.arange(2)[None]).flatten().to(device)
        batch = {group: {key: value.index_select(0, indices) for key, value in fields.items()}
                 for group, fields in self.encoded[family].items()}
        # Trim only trailing PAD; observation history and all true bytes remain.
        width = int(batch["inputs"]["lengths"].max())
        for key in ("token_ids", "valid_mask"):
            batch["inputs"][key] = batch["inputs"][key][:, :width]
        for key in ("observation_next_byte_targets", "observation_next_byte_mask"):
            batch["supervision"][key] = batch["supervision"][key][:, :width]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(**batch["inputs"],
                            decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
        losses = sequence_objective(output, batch)
        if not torch.isfinite(losses["loss"]):
            raise ValueError("nonfinite sequence objective")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        self.family_updates[family] += 1
        return {"family": family, **{key: float(value.detach()) for key, value in losses.items()},
            "observation_tokens": int(batch["inputs"]["valid_mask"].sum()),
            "reply_target_tokens": int(batch["supervision"]["reply_target_mask"].sum())}
