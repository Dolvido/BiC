"""Compact, reproducible lessons with exact simulator-owned answers.

This is a closed symbolic world, not a test of image recognition, English
comprehension, or general knowledge. Prompts explain lessons to a human/tutor;
the student receives a skill token and observations, never the prompt or answer.
The train/dev/audit partitions use different random streams, phrase families,
and nuisance identity ranges. They test transfer across those shifts, not unseen
concepts. Audit examples must not be used for fitting or candidate selection.

Each visual frame has three nine-channel object slots: presence, four color
indicators, two shape bits, x, y. Channels 27..31 are nuisance identity,
background brightness, cued slot / 2, comparison slot / 2, and time / 2.
Objects are simulated inert tokens; there is no suffering or real-world action.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping


VERSION = "symbolic-grounding-v1"
VISUAL_DIM = 32
AUDITORY_DIM = 4
BODY_DIM = 4
TIME_STEPS = 3
NUM_ACTIONS = 4
SPLITS = ("train", "dev", "audit")
SKILLS = ("color", "shape", "spatial", "count", "compare_count", "delayed_recall")
COLORS = ("red", "green", "blue", "yellow")
SHAPES = ("circle", "square", "triangle", "star")
DIRECTIONS = ("left", "right", "below", "above")
IDENTITY_RANGES = {"train": (1, 999_999), "dev": (3_000_001, 3_999_999), "audit": (7_000_001, 7_999_999)}
_IDENTITY_SCALE = 10_000_000
_FIELDS = {"id", "skill", "split", "seed", "observations", "target", "prompt", "explanation"}
_PROVENANCE = "Exact facts of the local inert-object simulator; no teacher-inferred labels."

CURRICULUM = {
    "color": {
        "prerequisites": (),
        "description": "Identify the color of the cued object in the final scene.",
        "question": "What color is the cued object?",
        "oracle": "Decode its four mutually exclusive color indicators.",
        "source": _PROVENANCE,
        "valid_targets": (0, 1, 2, 3),
    },
    "shape": {
        "prerequisites": (),
        "description": "Identify the shape of the cued object in the final scene.",
        "question": "What shape is the cued object?",
        "oracle": "Decode its two binary shape features, low bit first.",
        "source": _PROVENANCE,
        "valid_targets": (0, 1, 2, 3),
    },
    "spatial": {
        "prerequisites": ("color",),
        "description": "Find the cued object's direction relative to the comparison object.",
        "question": "Where is the cued object relative to the comparison object?",
        "oracle": "Use the dominant coordinate displacement; x grows right and y grows up.",
        "source": _PROVENANCE,
        "valid_targets": (0, 1, 2, 3),
    },
    "count": {
        "prerequisites": ("shape",),
        "description": "Count occupied object slots in the final scene, from zero to three.",
        "question": "How many objects are present?",
        "oracle": "Sum the three binary presence indicators.",
        "source": _PROVENANCE,
        "valid_targets": (0, 1, 2, 3),
    },
    "compare_count": {
        "prerequisites": ("count",),
        "description": "Compare the final object count with the first scene: fewer, equal, or more.",
        "question": "Are there fewer, the same number, or more objects than in the first scene?",
        "oracle": "Compare sums of presence indicators in the first and final frames: 0 fewer, 1 equal, 2 more.",
        "source": _PROVENANCE,
        "valid_targets": (0, 1, 2),
    },
    "delayed_recall": {
        "prerequisites": ("color",),
        "description": "Remember a cued color across two observations in which it is hidden.",
        "question": "What color did the cued object have in the first scene?",
        "oracle": "Decode the first-frame cued object's color; later frames contain no color for that object.",
        "source": _PROVENANCE,
        "valid_targets": (0, 1, 2, 3),
    },
}

# These families are provenance/display metadata, not a language-learning input.
_PROMPTS = {
    "train": ("Practice: {question}", "Observe carefully. {question}"),
    "dev": ("Development exercise — {question}", "Check this new scene: {question}"),
    "audit": ("Independent examination. {question}", "Held-out challenge — {question}"),
}


def curriculum_digest() -> str:
    """Fingerprint the versioned registry, layout, and split policy."""
    payload = {
        "version": VERSION,
        "curriculum": CURRICULUM,
        "skills": SKILLS,
        "prompts": _PROMPTS,
        "identity_ranges": IDENTITY_RANGES,
        "dimensions": [TIME_STEPS, VISUAL_DIM, AUDITORY_DIM, BODY_DIM, NUM_ACTIONS],
        "colors": COLORS,
        "shapes": SHAPES,
        "directions": DIRECTIONS,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _check_request(skill: str, seed: int, split: str) -> None:
    if not isinstance(skill, str) or skill not in CURRICULUM:
        raise ValueError(f"Unknown curriculum skill: {skill!r}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(split, str) or split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")


def _rng(skill: str, seed: int, split: str) -> random.Random:
    key = _canonical_json([VERSION, skill, split, seed]).encode("utf8")
    return random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))


def _object(rng: random.Random) -> list[float]:
    color, shape = rng.randrange(4), rng.randrange(4)
    return [1.0] + [float(color == c) for c in range(4)] + [
        float(shape & 1), float((shape >> 1) & 1),
        round(rng.uniform(-0.9, 0.9), 6), round(rng.uniform(-0.9, 0.9), 6),
    ]


def _count(frame: list[float]) -> int:
    return sum(int(frame[slot * 9]) for slot in range(3))


def _cue(frame: list[float]) -> int:
    return int(round(frame[29] * 2))


def _decode_color(frame: list[float]) -> int:
    start = _cue(frame) * 9
    colors = frame[start + 1:start + 5]
    if frame[start] != 1 or colors.count(1.0) != 1 or sum(colors) != 1:
        raise ValueError("The cued object must have exactly one visible color")
    return colors.index(1.0)


def oracle(example: Mapping[str, object]) -> int:
    """Derive a label from observations alone, without consulting supplied text/target.

    The public oracle checks the input schema, while :func:`validate_example`
    additionally authenticates the full generated lesson and its provenance.
    """
    if not isinstance(example, Mapping):
        raise ValueError("example must be a mapping")
    skill = example.get("skill")
    if not isinstance(skill, str) or skill not in CURRICULUM:
        raise ValueError("Unknown skill")
    observations = example.get("observations")
    _validate_observations(observations, skill)
    frames = observations["visual"]
    first, last = frames[0], frames[-1]
    if skill in ("color", "delayed_recall"):
        return _decode_color(first if skill == "delayed_recall" else last)
    if skill == "shape":
        start = _cue(last) * 9
        if last[start] != 1:
            raise ValueError("The cued object must be present")
        return int(last[start + 5]) + 2 * int(last[start + 6])
    if skill == "spatial":
        a, b = _cue(last) * 9, int(round(last[30] * 2)) * 9
        if a == b or last[a] != 1 or last[b] != 1:
            raise ValueError("Spatial comparison needs two distinct present objects")
        dx, dy = last[a + 7] - last[b + 7], last[a + 8] - last[b + 8]
        if abs(dx) == abs(dy):
            raise ValueError("Ambiguous spatial direction")
        return (0 if dx < 0 else 1) if abs(dx) > abs(dy) else (2 if dy < 0 else 3)
    if skill == "count":
        return _count(last)
    before, after = _count(first), _count(last)
    return 0 if after < before else 2 if after > before else 1


def _validate_observations(observations: object, skill: str) -> None:
    if type(observations) is not dict or set(observations) != {"visual", "auditory", "body", "tokens", "feedback"}:
        raise ValueError("Observation keys must exactly match the five student input channels")
    for channel, width in (("visual", VISUAL_DIM), ("auditory", AUDITORY_DIM), ("body", BODY_DIM), ("feedback", 2)):
        rows = observations[channel]
        if type(rows) is not list or len(rows) != TIME_STEPS:
            raise ValueError(f"{channel} must have exactly {TIME_STEPS} time steps")
        for row in rows:
            if type(row) is not list or len(row) != width:
                raise ValueError(f"{channel} rows must have width {width}")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in row):
                raise ValueError(f"{channel} must contain finite numeric JSON values")
    tokens = observations["tokens"]
    if type(tokens) is not list or len(tokens) != TIME_STEPS or any(type(token) is not int or token != SKILLS.index(skill) for token in tokens):
        raise ValueError("tokens must contain only the selected skill's integer index")
    for t, frame in enumerate(observations["visual"]):
        if frame[29] not in (0.0, 0.5, 1.0) or frame[30] not in (0.0, 0.5, 1.0) or frame[31] != t / 2:
            raise ValueError("Invalid object cue or observation time")
        for slot in range(3):
            obj = frame[slot * 9:slot * 9 + 9]
            if any(value not in (0.0, 1.0) for value in obj[:7]):
                raise ValueError("Object presence, color, and shape features must be binary")
            if obj[0] == 0.0 and any(obj[1:]):
                raise ValueError("Absent object slots must be empty")
            if obj[0] == 1.0:
                hidden = skill == "delayed_recall" and t > 0 and slot == _cue(frame)
                if sum(obj[1:5]) != (0 if hidden else 1):
                    raise ValueError("Present object has invalid color indicators")
                if any(abs(value) > 1 for value in obj[7:9]):
                    raise ValueError("Object coordinates must stay inside the simulated scene")
    if any(any(row) for row in observations["feedback"]):
        raise ValueError("Feedback must be zero before the student answers")


def _explanation(skill: str, observations: dict[str, list], target: int) -> str:
    if skill == "color":
        return f"The cued object in the final scene is {COLORS[target]}."
    if skill == "shape":
        return f"The cued object in the final scene is a {SHAPES[target]}."
    if skill == "spatial":
        return f"The cued object is {DIRECTIONS[target]} of the comparison object in the final scene."
    if skill == "count":
        return f"Exactly {target} of the three object slots are occupied in the final scene."
    if skill == "compare_count":
        first, last = observations["visual"][0], observations["visual"][-1]
        relation = ("fewer objects", "the same number of objects", "more objects")[target]
        return f"The first scene contains {_count(first)} objects and the final scene contains {_count(last)} objects, so the final scene has {relation}."
    return f"The cued object is {COLORS[target]} in the first scene; its color is hidden in both later scenes."


def _generate_one(skill: str, seed: int, split: str) -> dict:
    rng = _rng(skill, seed, split)
    cue, comparison = rng.sample(range(3), 2)
    base_objects = [_object(rng) for _ in range(3)]
    frames = [[value for obj in base_objects for value in obj] for _ in range(TIME_STEPS)]
    if skill == "spatial":
        direction = rng.randrange(4)
        anchor_x, anchor_y = rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)
        major, minor = rng.uniform(0.35, 0.55), rng.uniform(-0.12, 0.12)
        dx, dy = ((-major, minor), (major, minor), (minor, -major), (minor, major))[direction]
        for frame in frames:
            frame[comparison * 9 + 7:comparison * 9 + 9] = [round(anchor_x, 6), round(anchor_y, 6)]
            frame[cue * 9 + 7:cue * 9 + 9] = [round(anchor_x + dx, 6), round(anchor_y + dy, 6)]
    if skill in ("count", "compare_count"):
        if skill == "count":
            counts = [rng.randrange(4)] * TIME_STEPS
        else:
            relation = rng.randrange(3)
            pairs = [(before, after) for before in range(4) for after in range(4)
                     if (0 if after < before else 2 if after > before else 1) == relation]
            before, after = rng.choice(pairs)
            counts = [before, rng.randrange(4), after]
        for frame, count in zip(frames, counts):
            present = set(rng.sample(range(3), count))
            for slot in range(3):
                if slot not in present:
                    frame[slot * 9:slot * 9 + 9] = [0.0] * 9
    if skill == "delayed_recall":
        # Later color observations carry zero information about the hidden color.
        for frame in frames[1:]:
            frame[cue * 9 + 1:cue * 9 + 5] = [0.0] * 4
    identity = rng.randint(*IDENTITY_RANGES[split]) / _IDENTITY_SCALE
    background = round(rng.random(), 6)
    for t, frame in enumerate(frames):
        frame.extend([identity, background, cue / 2, comparison / 2, t / 2])
    observations = {
        "visual": frames,
        "auditory": [[0.0] * AUDITORY_DIM for _ in range(TIME_STEPS)],
        "body": [[0.0] * BODY_DIM for _ in range(TIME_STEPS)],
        "tokens": [SKILLS.index(skill)] * TIME_STEPS,
        "feedback": [[0.0, 0.0] for _ in range(TIME_STEPS)],
    }
    result = {"skill": skill, "split": split, "seed": seed, "observations": observations}
    target = oracle(result)
    key = _canonical_json([VERSION, skill, split, seed]).encode("utf8")
    result.update({
        "id": hashlib.sha256(key).hexdigest(),
        "target": target,
        "prompt": rng.choice(_PROMPTS[split]).format(question=CURRICULUM[skill]["question"]),
        "explanation": _explanation(skill, observations, target),
    })
    return result


def generate(skill: str, seed: int, count: int, split: str = "train") -> list[dict]:
    """Return verified examples with per-record seeds ``seed .. seed+count-1``.

    A replay descriptor needs only skill and seed (plus this registry's version).
    Generation is independent of batching and of global Python random state.
    """
    _check_request(skill, seed, split)
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")
    return [_generate_one(skill, seed + index, split) for index in range(count)]


def validate_example(example: object) -> bool:
    """Reject malformed, mislabeled, or altered lessons; return True if canonical."""
    if type(example) is not dict or set(example) != _FIELDS:
        raise ValueError("Example fields must exactly match the curriculum schema")
    _check_request(example["skill"], example["seed"], example["split"])
    if type(example["target"]) is not int or example["target"] not in CURRICULUM[example["skill"]]["valid_targets"]:
        raise ValueError("Invalid target")
    if any(type(example[field]) is not str for field in ("id", "prompt", "explanation")):
        raise ValueError("Provenance and lesson text fields must be strings")
    if oracle(example) != example["target"]:
        raise ValueError("Target disagrees with the independent scene oracle")
    canonical = _generate_one(example["skill"], example["seed"], example["split"])
    if example != canonical:
        raise ValueError("Lesson disagrees with its deterministic provenance")
    return True
