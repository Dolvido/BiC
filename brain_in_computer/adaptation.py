"""A small feedback-learning world, separate from the neural implementation.

Four already identified categorical objects occupy four shuffled cells. This is
a symbolic adaptation task, not visual concept acquisition. A single hidden
reward-rule reversal tests whether an agent uses action-contingent feedback.

Only ``RuleWorld.observe()`` is an agent input. Snapshots, evaluator methods, and
history deliberately contain privileged labels and must remain outside policies.
``CandidateLearner`` is a scripted causal learning strategy; imitating it is not
reward-only reinforcement learning and does not itself establish neural learning.
"""
from __future__ import annotations

import copy
from itertools import permutations
import random

import torch
from torch import Tensor
from torch.nn import functional as F


CATEGORIES = (0, 1, 2, 3)
WORLD_SCHEMA = "bic-rule-world-v1"
LEARNER_SCHEMA = "bic-candidate-learner-v1"
OBSERVATION_KEYS = frozenset(("layout", "previous_category", "reward"))


def _integer(value, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    return value


def _category(value, name: str) -> int:
    value = _integer(value, name)
    if value not in CATEGORIES:
        raise ValueError(f"{name} must be in 0..3")
    return value


def _layout(value) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("layout must contain four categorical objects")
    for item in value:
        _category(item, "layout category")
    if set(value) != set(CATEGORIES):
        raise ValueError("layout must contain each category exactly once")
    return tuple(value)


def all_layouts() -> list[tuple[int, int, int, int]]:
    """All 24 cell-to-category permutations in a stable lexical order."""
    return list(permutations(CATEGORIES))


def split_layouts(seed: int = 7001) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Return deterministic disjoint 18-layout train and 6-layout test sets.

    This splits spatial arrangements, not category identities or task semantics.
    It does not mutate Python's or PyTorch's global random-number state.
    """
    _integer(seed, "seed")
    layouts = all_layouts()
    random.Random(seed).shuffle(layouts)
    return layouts[:18], layouts[18:]


def _validate_observation(observation: dict) -> tuple[tuple[int, ...], int | None, float | None]:
    if not isinstance(observation, dict) or set(observation) != OBSERVATION_KEYS:
        raise ValueError("public observation must contain only layout, previous_category, and reward")
    layout = _layout(observation["layout"])
    previous = observation["previous_category"]
    reward = observation["reward"]
    if previous is None:
        if reward is not None:
            raise ValueError("feedback without a previous choice is invalid")
    else:
        _category(previous, "previous_category")
        if type(reward) not in (int, float) or reward not in (0, 1):
            raise ValueError("a previous choice requires finite binary feedback")
        reward = float(reward)
    return layout, previous, reward


def _json_rng_state(rng: random.Random) -> list:
    version, internal, gaussian = rng.getstate()
    return [version, list(internal), gaussian]


class RuleWorld:
    """Seeded finite episode with a hidden category rewarded before/after a switch.

    ``reversal_step`` is zero-based: action indices below it use the first rule;
    the action at that index is the first governed by the second rule. It must
    be strictly inside the episode. If omitted it is sampled from 16..24, which
    requires horizon >= 25. Explicit switch steps permit shorter episodes.

    Layout RNG is independent of the RNG choosing the rule and switch time.
    Layouts therefore do not change with actions, feedback, or reversal timing.
    There is no implicit reset, and no reset or phase flag appears at reversal.
    """

    def __init__(self, seed: int, horizon: int = 40, reversal_step: int | None = None,
                 allowed_layouts: list | tuple | None = None):
        _integer(seed, "seed")
        _integer(horizon, "horizon")
        if horizon < 2:
            raise ValueError("horizon must be at least two actions")
        if reversal_step is None:
            if horizon < 25:
                raise ValueError("horizon must be >= 25 for the default 16..24 reversal range")
        else:
            _integer(reversal_step, "reversal_step")
            if not 1 <= reversal_step < horizon:
                raise ValueError("reversal_step must lie strictly inside the episode")
        if allowed_layouts is None:
            allowed_layouts = all_layouts()
        if not isinstance(allowed_layouts, (list, tuple)) or not allowed_layouts:
            raise ValueError("allowed_layouts must be a nonempty list of permutations")
        layouts = tuple(_layout(layout) for layout in allowed_layouts)
        if len(set(layouts)) != len(layouts):
            raise ValueError("allowed_layouts must not contain duplicates")

        # A separate string-seeded RNG prevents hidden schedule draws from
        # shifting the visible layout stream or depending on agent behavior.
        rule_rng = random.Random(f"bic-rule-world-v1:{seed}")
        self._initial_target = rule_rng.randrange(4)
        self._reversed_target = (self._initial_target + rule_rng.randrange(1, 4)) % 4
        self._reversal_step = rule_rng.randint(16, 24) if reversal_step is None else reversal_step
        self._seed, self._horizon = seed, horizon
        self._allowed_layouts = layouts
        self._rng = random.Random(seed)
        self._layout = self._rng.choice(self._allowed_layouts)
        self._step_index = 0
        self._previous_category = None
        self._reward = None
        self._history = []

    @property
    def done(self) -> bool:
        """Episode termination for the outer evaluator; absent from observations."""
        return self._step_index >= self._horizon

    def observe(self) -> dict:
        """Return a fresh public observation containing past feedback only."""
        return {"layout": list(self._layout), "previous_category": self._previous_category,
                "reward": self._reward}

    def evaluator_phase(self) -> str:
        """Privileged metric label for the next action (or final terminal phase)."""
        return "before" if self._step_index < self._reversal_step else "after"

    def evaluator_target_category(self) -> int:
        """Privileged oracle label for scoring, never a policy observation."""
        return self._initial_target if self._step_index < self._reversal_step else self._reversed_target

    def evaluator_state(self) -> dict:
        """Privileged state for auditing; explicitly not an agent input."""
        return {"step_index": self._step_index, "horizon": self._horizon,
                "reversal_step": self._reversal_step, "phase": self.evaluator_phase(),
                "target_category": self.evaluator_target_category(),
                "initial_target": self._initial_target, "reversed_target": self._reversed_target,
                "done": self.done}

    def export_history(self) -> list[dict]:
        """Privileged scoring history, returned by deep copy."""
        return copy.deepcopy(self._history)

    def step(self, action: int) -> tuple[dict, float]:
        """Choose a cell, receive binary reward, then present the next layout."""
        _category(action, "action cell")
        if self.done:
            raise RuntimeError("episode is finished; create a new world for another episode")
        chosen = self._layout[action]
        target = self.evaluator_target_category()
        reward = float(chosen == target)
        self._history.append({"step_index": self._step_index, "layout": list(self._layout),
                              "action": action, "chosen_category": chosen, "reward": reward,
                              "phase": self.evaluator_phase(), "target_category": target})
        self._previous_category, self._reward = chosen, reward
        self._step_index += 1
        # A terminal observation follows the same transition rule as all others.
        # An extra layout draw cannot signal the hidden switch or reset a policy.
        self._layout = self._rng.choice(self._allowed_layouts)
        return self.observe(), reward

    def snapshot(self) -> dict:
        """JSON-friendly evaluator checkpoint, including hidden state and RNG.

        Do not supply this object to an agent. Python random.Random's versioned
        state is preserved; restoration is intended for the same Python runtime.
        """
        return {"schema": WORLD_SCHEMA, "seed": self._seed, "horizon": self._horizon,
                "reversal_step": self._reversal_step,
                "allowed_layouts": [list(layout) for layout in self._allowed_layouts],
                "step_index": self._step_index, "layout": list(self._layout),
                "previous_category": self._previous_category, "reward": self._reward,
                "initial_target": self._initial_target, "reversed_target": self._reversed_target,
                "rng_state": _json_rng_state(self._rng), "history": self.export_history()}

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "RuleWorld":
        """Restore an exact trajectory; inconsistent or altered snapshots fail.

        Replaying the saved actions validates scene, reward, hidden-rule, and RNG
        consistency before the saved RNG state is restored explicitly. Replay is
        deterministic and never changes global RNGs or the caller's dictionary.
        """
        expected_keys = {"schema", "seed", "horizon", "reversal_step", "allowed_layouts",
                         "step_index", "layout", "previous_category", "reward", "initial_target",
                         "reversed_target", "rng_state", "history"}
        if not isinstance(snapshot, dict) or set(snapshot) != expected_keys or snapshot["schema"] != WORLD_SCHEMA:
            raise ValueError("unsupported or malformed rule-world snapshot")
        world = cls(snapshot["seed"], snapshot["horizon"], snapshot["reversal_step"], snapshot["allowed_layouts"])
        step = _integer(snapshot["step_index"], "step_index")
        if not 0 <= step <= world._horizon:
            raise ValueError("snapshot step_index is outside the episode")
        history = snapshot["history"]
        if not isinstance(history, list) or len(history) != step:
            raise ValueError("snapshot history must match the completed action count")
        for entry in history:
            if not isinstance(entry, dict) or "action" not in entry:
                raise ValueError("malformed snapshot history entry")
            world.step(entry["action"])
        # Comparing canonical JSON also distinguishes True from 1 and integer
        # values from floats in fields that would compare equal in Python.
        import json
        try:
            actual = json.dumps(snapshot, sort_keys=True, allow_nan=False)
            expected = json.dumps(world.snapshot(), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("snapshot must be finite JSON data") from error
        if actual != expected:
            raise ValueError("snapshot state, history, or RNG does not match its deterministic trajectory")
        version, internal, gaussian = snapshot["rng_state"]
        world._rng.setstate((version, tuple(internal), gaussian))
        return world


class CandidateLearner:
    """Scripted elimination baseline using only the public last-choice feedback.

    Positive feedback fixes the candidate category. Negative feedback excludes
    it; contradicting a singleton (or exhausting candidates) starts a fresh
    candidate set excluding that category. No hidden switch detector is used.
    """

    def __init__(self):
        self._candidates = set(CATEGORIES)

    def probabilities(self, observation: dict) -> list[float]:
        """Process public feedback and return a uniform distribution over categories."""
        _, previous, reward = _validate_observation(observation)
        if previous is not None:
            if reward == 1.0:
                self._candidates = {previous}
            else:
                self._candidates.discard(previous)
                if not self._candidates:
                    self._candidates = set(CATEGORIES) - {previous}
        probability = 1.0 / len(self._candidates)
        return [probability if category in self._candidates else 0.0 for category in CATEGORIES]

    def choose_category(self, observation: dict) -> int:
        """Update once and choose the lowest remaining category deterministically."""
        probabilities = self.probabilities(observation)
        return next(category for category, probability in enumerate(probabilities) if probability > 0)

    def act(self, observation: dict) -> int:
        """Update once, then map the lowest candidate category to its current cell."""
        probabilities = self.probabilities(observation)
        category = next(category for category, probability in enumerate(probabilities) if probability > 0)
        return observation["layout"].index(category)

    def snapshot(self) -> dict:
        return {"schema": LEARNER_SCHEMA, "candidates": sorted(self._candidates)}

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "CandidateLearner":
        if (not isinstance(snapshot, dict) or set(snapshot) != {"schema", "candidates"}
                or snapshot["schema"] != LEARNER_SCHEMA):
            raise ValueError("unsupported candidate-learner snapshot")
        values = snapshot["candidates"]
        if not isinstance(values, list) or not values:
            raise ValueError("candidate set must be a nonempty list")
        for value in values:
            _category(value, "candidate")
        if values != sorted(set(values)):
            raise ValueError("candidate set must be unique and sorted")
        learner = cls()
        learner._candidates = set(values)
        return learner


def encode_observations(observations: list[dict] | tuple[dict, ...], device="cpu") -> dict[str, Tensor]:
    """Encode only public fields as one-step core inputs; no time or phase channel.

    visual [B,1,16]: cell-major category one-hot (four cells by four categories).
    auditory [B,1,4]: previous chosen category one-hot; all zero initially.
    body [B,1,4]: [constant one, feedback present, zero, zero].
    feedback [B,1,2]: [previous reward or zero, feedback present].
    tokens [B,1]: long zeros, for a single-symbol vocabulary.

    BrainConfig dimensions are visual16/auditory4/body4/vocab1/actions4.
    Neural policy action indices may represent categories; the experiment must
    map a chosen category to the visible cell before calling RuleWorld.step.
    """
    if not isinstance(observations, (list, tuple)) or not observations:
        raise ValueError("observations must be a nonempty list or tuple")
    validated = [_validate_observation(observation) for observation in observations]
    layouts = torch.tensor([row[0] for row in validated], dtype=torch.long, device=device)
    previous = torch.tensor([0 if row[1] is None else row[1] for row in validated], dtype=torch.long, device=device)
    present = torch.tensor([row[1] is not None for row in validated], dtype=torch.float32, device=device)
    rewards = torch.tensor([0.0 if row[2] is None else row[2] for row in validated], dtype=torch.float32, device=device)
    batch = len(validated)
    body = torch.zeros(batch, 1, 4, dtype=torch.float32, device=device)
    body[:, 0, 0] = 1
    body[:, 0, 1] = present
    return {"visual": F.one_hot(layouts, 4).to(torch.float32).reshape(batch, 1, 16),
            "auditory": (F.one_hot(previous, 4).to(torch.float32) * present[:, None]).unsqueeze(1),
            "body": body, "feedback": torch.stack((rewards, present), dim=-1).unsqueeze(1),
            "tokens": torch.zeros(batch, 1, dtype=torch.long, device=device)}
