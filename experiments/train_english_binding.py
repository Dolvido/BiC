"""Isolated experiment balancing known and unknown binding supervision.

This runner keeps the frozen frontier implementation intact. A process-local
temporary class substitution changes only the training-only binding loss;
student architecture, policy inputs, other losses and evaluation are unchanged.
Do not run this scoped experiment concurrently in threads of the same process.
Its own source hash joins the frontier source manifest for restart provenance.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from brain_in_computer import dialogue_scaffolding
from brain_in_computer.dialogue_scaffolding import (
    ScaffoldHeads, UNKNOWN_COLOR,
)
from experiments import train_english_frontier as frontier


ARM = "binding_balanced"
_FRONTIER_TRAIN_CHUNK = frontier.train_chunk
_FRONTIER_SOURCE_FINGERPRINTS = frontier.source_fingerprints


class BalancedScaffoldHeads(ScaffoldHeads):
    """Give established bindings and unknown bindings equal aggregate weight.

    When both strata are present, binding CE is half the mean over known
    bindings plus half the mean over unknown bindings. A single present stratum
    receives full weight. Fully masked bindings yield differentiable zero.
    Other head losses and all model/readout parameters remain unchanged.
    """

    def losses(self, output, targets, turn):
        result = super().losses(output, targets, turn)
        if "bindings" not in result:
            return result
        activity = output["region_activity"]
        state = torch.cat((activity["prefrontal_cortex"][:, -1],
                           activity["hippocampus"][:, -1]), dim=-1)
        labels = targets["bindings"][turn]
        mask = targets["masks"]["bindings"][turn]
        logits = self.bindings(state).reshape(*labels.shape, UNKNOWN_COLOR + 1)
        safe_labels = labels.masked_fill(~mask, 0)
        terms = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                safe_labels.reshape(-1), reduction="none").reshape_as(labels)
        known = mask & labels.ne(UNKNOWN_COLOR)
        unknown = mask & labels.eq(UNKNOWN_COLOR)
        known_count, unknown_count = known.sum(), unknown.sum()
        known_mean = (terms * known).sum() / known_count.clamp_min(1)
        unknown_mean = (terms * unknown).sum() / unknown_count.clamp_min(1)
        strata = known_count.gt(0).long() + unknown_count.gt(0).long()
        result["bindings"] = (known_mean + unknown_mean) / strata.clamp_min(1)
        return result


def train_chunk(payload, episodes, *, seed, steps, batch_size, device,
                arm=ARM, lr=.003):
    """Reuse the frozen trainer, substituting its disposable readouts locally."""
    if arm != ARM:
        raise ValueError(f"this experimental wrapper requires arm={ARM!r}")
    with patch.object(dialogue_scaffolding, "ScaffoldHeads", BalancedScaffoldHeads):
        return _FRONTIER_TRAIN_CHUNK(payload, episodes, seed=seed, steps=steps,
            batch_size=batch_size, device=device, arm="scaffold", lr=lr)


def source_fingerprints():
    fingerprints = _FRONTIER_SOURCE_FINGERPRINTS()
    path = Path(__file__).resolve()
    root = path.parents[1]
    fingerprints[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprints


def run(args):
    if args.arm != ARM:
        raise ValueError(f"this experimental runner requires arm={ARM!r}")
    # These substitutions are restored even if validation, training or writing
    # fails. Other processes continue using the unchanged frontier source.
    with patch.object(frontier, "train_chunk", train_chunk), \
            patch.object(frontier, "source_fingerprints", source_fingerprints):
        return frontier.run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--lr", type=float, default=.003)
    parser.set_defaults(arm=ARM)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.checkpoint_every, args.max_seconds, args.lr) <= 0:
        parser.error("budgets and learning rate must be positive")
    run(args)


if __name__ == "__main__":
    main()
