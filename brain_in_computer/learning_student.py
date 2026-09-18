"""Weight learning for the compact, locally verified curriculum.

Each example is an independent three-observation episode. Only its final action
is supervised; explanatory text and labels are never added to observations.
Recurrent activity is reset by Brain.forward, so evaluation cannot write memory.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Mapping, Sequence

import torch
from torch.nn import functional as F

from .model import Brain, BrainConfig


_INITIALIZATION_LOCK = threading.Lock()
_WIDTHS = {"visual": 32, "auditory": 4, "body": 4, "feedback": 2}
_SEQUENCE_LENGTH = 3


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite_number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def build_student(seed: int, hidden_size: int = 32, device="cpu") -> Brain:
    """Create a separate thirteen-region student, preserving the CPU RNG state.

    All initial parameters are made on CPU inside a serialized RNG fork before
    moving the model. No CUDA generator is seeded or otherwise modified.
    """
    _integer("seed", seed)
    if seed >= 2**63:
        raise ValueError("seed must be < 2**63")
    _integer("hidden_size", hidden_size, 1)
    config = BrainConfig(hidden_size=hidden_size, visual_dim=32, auditory_dim=4,
                         body_dim=4, vocab_size=16, num_actions=4, memory_slots=4)
    with _INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        student = Brain(config)
    return student.to(device)


def encode_examples(examples: Sequence[Mapping], device="cpu") -> tuple[dict, torch.Tensor]:
    """Validate the curriculum's strict JSON observation contract and batch it."""
    if not examples:
        raise ValueError("examples must not be empty")
    rows = {name: [] for name in (*_WIDTHS, "tokens")}
    targets = []
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError(f"example {index} must be a mapping")
        target = example.get("target")
        if type(target) is not int or not 0 <= target < 4:
            raise ValueError(f"example {index} target must be an integer in [0, 3]")
        if not isinstance(example.get("skill"), str) or not example["skill"]:
            raise ValueError(f"example {index} requires a nonempty skill")
        observation = example.get("observations")
        if not isinstance(observation, Mapping) or set(observation) != set(rows):
            raise ValueError(f"example {index} observation channels must be {tuple(rows)}")
        for name, width in _WIDTHS.items():
            try:
                value = torch.as_tensor(observation[name], dtype=torch.float32)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError(f"example {index} {name} must be a numeric array") from error
            if value.shape != (_SEQUENCE_LENGTH, width) or not torch.isfinite(value).all():
                raise ValueError(f"example {index} {name} must be finite with shape (3, {width})")
            rows[name].append(value)
        tokens = observation["tokens"]
        if (not isinstance(tokens, (list, tuple)) or len(tokens) != _SEQUENCE_LENGTH
                or any(type(token) is not int or not 0 <= token < 16 for token in tokens)):
            raise ValueError(f"example {index} tokens must contain three integers in [0, 15]")
        rows["tokens"].append(torch.tensor(tokens, dtype=torch.long))
        targets.append(target)
    observations = {name: torch.stack(values).to(device) for name, values in rows.items()}
    return observations, torch.tensor(targets, dtype=torch.long, device=device)


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probabilities = logits.softmax(dim=-1)
    total = targets.numel()
    correct = int((probabilities.argmax(dim=-1) == targets).sum().item())
    # Multiclass Brier: sum over all four classes, then mean over examples.
    brier = (probabilities - F.one_hot(targets, num_classes=4)).square().sum(dim=-1).mean()
    loss = F.cross_entropy(logits, targets)
    if not torch.isfinite(loss) or not torch.isfinite(brier):
        raise ValueError("student produced nonfinite evaluation metrics")
    return {"accuracy": correct / total, "loss": float(loss.item()),
            "brier": float(brier.item()), "correct": correct, "total": total}


def evaluate_student(model: Brain, examples: Sequence[Mapping]) -> dict:
    """Return held-out scores without changing weights, activity, or mode flags.

    Brier score measures probability error; it is not a claim of calibration.
    """
    device = next(model.parameters()).device
    observations, targets = encode_examples(examples, device)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            logits = model(observations)["logits"][:, -1]
            if not torch.isfinite(logits).all():
                raise ValueError("student produced nonfinite evaluation logits")
            result = _metrics(logits, targets)
            result["per_skill"] = {}
            for skill in sorted({example["skill"] for example in examples}):
                indices = torch.tensor([index for index, example in enumerate(examples)
                                        if example["skill"] == skill], device=device)
                result["per_skill"][skill] = _metrics(logits[indices], targets[indices])
        return result
    finally:
        # Direct flag restoration preserves even deliberately mixed submodule modes.
        for module, training in modes:
            module.training = training


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _check_finite_tree(value, name):
    if isinstance(value, torch.Tensor):
        if value.device.type == "meta" or not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite materialized tensors")
    elif isinstance(value, Mapping):
        for item in value.values():
            _check_finite_tree(item, name)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _check_finite_tree(item, name)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must contain finite numbers")


def train_candidate(initial_state_dict: Mapping, examples: Sequence[Mapping], *,
                    seed: int, hidden_size: int = 32, steps: int, batch_size: int,
                    learning_rate: float, replay_fraction: float = 0.25,
                    replay_examples: Sequence[Mapping] = (), device="cpu",
                    deadline: float | None = None, optimizer_state: Mapping | None = None,
                    sampler_state: torch.Tensor | None = None) -> dict:
    """Train an isolated candidate with verified train labels and optional replay.

    Replay is sampled independently per batch position with replay_fraction
    probability, or disabled when no replay data exist. The local generator
    state and AdamW moments permit exact continuation for an unchanged corpus.
    An absolute time.monotonic deadline is checked before each whole update;
    an in-progress update is completed atomically and can overrun the deadline.
    """
    started = time.monotonic()
    _integer("steps", steps)
    _integer("batch_size", batch_size, 1)
    _finite_number("learning_rate", learning_rate)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    _finite_number("replay_fraction", replay_fraction)
    if not 0 <= replay_fraction <= 1:
        raise ValueError("replay_fraction must be in [0, 1]")
    if deadline is not None:
        _finite_number("deadline", deadline)
    for example in (*examples, *replay_examples):
        if not isinstance(example, Mapping) or example.get("split") != "train":
            raise ValueError("training and replay accept only split='train'; evaluation leakage rejected")
    observations, targets = encode_examples(examples, device)
    replay_count = len(replay_examples)
    if replay_count:
        replay_observations, replay_targets = encode_examples(replay_examples, device)
        observations = {name: torch.cat((value, replay_observations[name]))
                        for name, value in observations.items()}
        targets = torch.cat((targets, replay_targets))
    model = build_student(seed, hidden_size, device)
    _check_finite_tree(initial_state_dict, "initial_state_dict")
    model.load_state_dict(initial_state_dict, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if optimizer_state is not None:
        _check_finite_tree(optimizer_state, "optimizer_state")
        optimizer.load_state_dict(_cpu_copy(optimizer_state))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if sampler_state is not None:
        generator.set_state(sampler_state.detach().cpu().clone())
    updates = examples_seen = replay_examples_seen = 0
    loss_value = None
    model.train()
    for _ in range(steps):
        if deadline is not None and time.monotonic() >= deadline:
            break
        indices = torch.randint(len(examples), (batch_size,), generator=generator)
        if replay_count:
            replay_mask = torch.rand(batch_size, generator=generator) < replay_fraction
            draw_count = int(replay_mask.sum().item())
            indices[replay_mask] = (torch.randint(replay_count, (draw_count,), generator=generator)
                                    + len(examples))
        else:
            draw_count = 0
        indices = indices.to(device)
        batch = {name: value[indices] for name, value in observations.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)["logits"][:, -1]
        loss = F.cross_entropy(logits, targets[indices])
        if not torch.isfinite(loss):
            raise ValueError("candidate produced nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        updates += 1
        examples_seen += batch_size
        replay_examples_seen += draw_count
        loss_value = float(loss.detach().item())
    _check_finite_tree(model.state_dict(), "trained_state_dict")
    _check_finite_tree(optimizer.state_dict(), "trained_optimizer_state")
    return {"state_dict": _cpu_copy(model.state_dict()),
            "optimizer_state": _cpu_copy(optimizer.state_dict()),
            "sampler_state": generator.get_state().clone(),
            "training_seconds": time.monotonic() - started, "updates": updates,
            "examples_seen": examples_seen, "replay_examples_seen": replay_examples_seen,
            "loss": loss_value,
            "deadline_reached": deadline is not None and time.monotonic() >= deadline}
