"""Small, causal synthetic tasks for testing a modular learning system.

Exact batch schema (B=batch size, T=delay+2):

* observations['visual']: float32 [B,T,8], one-hot symbol in columns 0:4.
* observations['auditory']: float32 [B,T,8], one-hot symbol in columns 0:4.
* observations['body']: float32 [B,T,4], normalized time, start, query, one.
* observations['tokens']: int64 [B,T], a stable instruction for the episode.
* observations['feedback']: float32 [B,T,2], always zero (no answer leakage).
* targets: int64 [B,T], final action 0..3; earlier positions are -100.
* visual_targets, auditory_targets: int64 [B,T], the observed symbol 0..3,
  or -100 for an absent symbol. These are auxiliary labels, never inputs.
* task_ids: int64 [B], indices into TASK_NAMES.

Token vocabulary: 0 blank; 1 copy final visual; 2 copy final auditory;
3 attend final visual; 4 attend final auditory; 5 remember first visual;
6 add final visual and auditory modulo four; 7 reserved. Columns 4:8 of
present sensory symbols contain independent uniform noise in [-0.02,0.02].
The delayed task has random intermediate distractors and no final visual
symbol, so copying a final visual observation cannot solve it.

Each stream owns a CPU torch.Generator. Use distinct seeds for training and
evaluation, and a larger evaluation delay to test memory beyond training.
Independent random test draws can share simple symbol combinations with
training: these tasks do not establish strong compositional generalization,
human-level competence, or general intelligence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


TASK_NAMES = (
    "visual_match",
    "auditory_match",
    "rule_switch",
    "delayed_match",
    "cross_modal",
)


@dataclass
class Batch:
    """Observations and labels, kept separate to prevent answer leakage."""

    observations: dict[str, Tensor]
    targets: Tensor
    visual_targets: Tensor
    auditory_targets: Tensor
    task_ids: Tensor

    def to(self, device: torch.device | str) -> Batch:
        """Return a batch with every tensor on ``device``."""
        return Batch(
            observations={key: value.to(device) for key, value in self.observations.items()},
            targets=self.targets.to(device),
            visual_targets=self.visual_targets.to(device),
            auditory_targets=self.auditory_targets.to(device),
            task_ids=self.task_ids.to(device),
        )


class TaskStream:
    """Reproducible stream of fresh episodes; sampling advances local RNG state."""

    def __init__(self, seed: int = 0) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def sample(
        self,
        batch_size: int = 32,
        tasks: Sequence[str] | None = None,
        delay: int = 2,
        device: torch.device | str = "cpu",
    ) -> Batch:
        """Draw a uniform mixture of the requested tasks.

        ``delay`` counts intervening distractor steps between first cue and
        final query. Zero is valid and produces a two-step episode. Sampling
        always happens on CPU before transferring the complete batch.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not isinstance(delay, int) or isinstance(delay, bool):
            raise TypeError("delay must be an integer")
        if delay < 0:
            raise ValueError("delay must be nonnegative")
        if tasks is None:
            task_names = TASK_NAMES
        else:
            if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
                raise TypeError("tasks must be a sequence of task names")
            task_names = tuple(tasks)
            if not task_names:
                raise ValueError("tasks must contain at least one task name")
            if any(not isinstance(name, str) for name in task_names):
                raise TypeError("every task name must be a string")
            unknown = [name for name in task_names if name not in TASK_NAMES]
            if unknown:
                raise ValueError(f"unknown task name(s): {unknown}")
            if len(set(task_names)) != len(task_names):
                raise ValueError("tasks must not contain duplicate names")

        steps = delay + 2
        chosen = torch.randint(len(task_names), (batch_size,), generator=self.generator)
        allowed_ids = torch.tensor([TASK_NAMES.index(name) for name in task_names], dtype=torch.long)
        task_ids = allowed_ids[chosen]

        # Independent draws matter: an ignored modality must not reveal the
        # attended symbol, and a distractor must not predict an earlier cue.
        visual_symbols = torch.randint(4, (batch_size, steps), generator=self.generator)
        auditory_symbols = torch.randint(4, (batch_size, steps), generator=self.generator)
        attend_auditory = torch.randint(2, (batch_size,), generator=self.generator).bool()

        visual = torch.zeros(batch_size, steps, 8)
        auditory = torch.zeros(batch_size, steps, 8)
        visual.scatter_(2, visual_symbols.unsqueeze(-1), 1.0)
        auditory.scatter_(2, auditory_symbols.unsqueeze(-1), 1.0)
        visual[:, :, 4:] = torch.rand(batch_size, steps, 4, generator=self.generator) * 0.04 - 0.02
        auditory[:, :, 4:] = torch.rand(batch_size, steps, 4, generator=self.generator) * 0.04 - 0.02

        visual_targets = visual_symbols.clone()
        auditory_targets = auditory_symbols.clone()
        final_actions = torch.empty(batch_size, dtype=torch.long)
        instructions = torch.empty(batch_size, dtype=torch.long)

        for task_id, name in enumerate(TASK_NAMES):
            selected = task_ids == task_id
            if name == "visual_match":
                instructions[selected] = 1
                final_actions[selected] = visual_symbols[selected, -1]
            elif name == "auditory_match":
                instructions[selected] = 2
                final_actions[selected] = auditory_symbols[selected, -1]
            elif name == "rule_switch":
                instructions[selected] = 3 + attend_auditory[selected].long()
                final_actions[selected] = torch.where(
                    attend_auditory[selected],
                    auditory_symbols[selected, -1],
                    visual_symbols[selected, -1],
                )
            elif name == "delayed_match":
                instructions[selected] = 5
                final_actions[selected] = visual_symbols[selected, 0]
                visual[selected, -1, :] = 0.0
                visual_targets[selected, -1] = -100
            elif name == "cross_modal":
                instructions[selected] = 6
                final_actions[selected] = (
                    visual_symbols[selected, -1] + auditory_symbols[selected, -1]
                ) % 4

        body = torch.zeros(batch_size, steps, 4)
        body[:, :, 0] = torch.arange(steps, dtype=torch.float32) / (steps - 1)
        body[:, 0, 1] = 1.0
        body[:, -1, 2] = 1.0
        body[:, :, 3] = 1.0
        targets = torch.full((batch_size, steps), -100, dtype=torch.long)
        targets[:, -1] = final_actions

        return Batch(
            observations={
                "visual": visual,
                "auditory": auditory,
                "body": body,
                "tokens": instructions[:, None].expand(-1, steps).clone(),
                "feedback": torch.zeros(batch_size, steps, 2),
            },
            targets=targets,
            visual_targets=visual_targets,
            auditory_targets=auditory_targets,
            task_ids=task_ids,
        ).to(device)
