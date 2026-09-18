"""Bounded reservoir of complete training episodes for offline rehearsal.

Labels remain training targets; they are never fed into the brain's observations.
This is not inference-time autobiographical retrieval.
"""

import torch
from .tasks import Batch


class EpisodeReplay:
    def __init__(self, capacity=256, seed=0):
        if capacity < 0:
            raise ValueError("capacity must be nonnegative")
        self.capacity = capacity
        self.seen = 0
        self.rows = []
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.rows)

    def add(self, batch):
        for i in range(len(batch.task_ids)):
            self.seen += 1
            if not self.capacity:
                continue
            slot = len(self.rows) if len(self.rows) < self.capacity else int(
                torch.randint(self.seen, (), generator=self.generator))
            if slot >= self.capacity:
                continue
            row = {
                "observations": {k: v[i:i+1].detach().cpu().clone()
                                 for k, v in batch.observations.items()},
                **{k: getattr(batch, k)[i:i+1].detach().cpu().clone()
                   for k in ("targets", "visual_targets", "auditory_targets", "task_ids")},
            }
            if slot == len(self.rows):
                self.rows.append(row)
            else:
                self.rows[slot] = row

    def sample(self, count, device="cpu"):
        if not self.rows or count < 1:
            raise ValueError("sample requires stored episodes and positive count")
        indices = torch.randint(len(self.rows), (count,), generator=self.generator)
        rows = [self.rows[int(i)] for i in indices]
        return Batch(
            observations={k: torch.cat([r["observations"][k] for r in rows]).to(device)
                          for k in rows[0]["observations"]},
            **{k: torch.cat([r[k] for r in rows]).to(device)
               for k in ("targets", "visual_targets", "auditory_targets", "task_ids")},
        )

    def state_dict(self):
        return {"capacity": self.capacity, "seen": self.seen, "rows": self.rows,
                "rng": self.generator.get_state()}

    def load_state_dict(self, state):
        self.capacity, self.seen, self.rows = state["capacity"], state["seen"], state["rows"]
        self.generator.set_state(state["rng"].cpu())


def concatenate(a, b):
    return Batch(
        observations={k: torch.cat([a.observations[k], b.observations[k]])
                      for k in a.observations},
        **{k: torch.cat([getattr(a, k), getattr(b, k)])
           for k in ("targets", "visual_targets", "auditory_targets", "task_ids")},
    )
