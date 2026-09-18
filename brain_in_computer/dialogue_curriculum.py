"""Verified six-turn English conversations for a small inert-tool world.

English is the *only* alias/rule channel. All nonlanguage observations, including
task tokens and feedback, are zero. The oracle is a tiny explicit grammar and
state machine, never a teacher model. This is a language-and-memory experiment,
not unrestricted English understanding or evidence of human-level intellect.

Train/dev/audit withhold entire alias-to-color bindings, and therefore every
alias/color/rule composition involving that binding. Each split has all colors,
aliases, and action labels, but different phrase families. This deliberately
tests composition and wording together; it does not claim unseen-word transfer.
Even/odd seed pairs share question text, ordering, and nonlanguage inputs while
reversing which known tool may be borrowed. A query alone cannot solve a pair.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping


VERSION = "english-tool-dialogue-v1"
NUM_TURNS = 6
MAX_TEXT_BYTES = 128
DENY, ALLOW, ASK, ACK = range(4)
TARGET_NAMES = ("deny", "allow", "ask", "ack")
REPLIES = ("No.", "Yes.", "I need more information.", "Understood.")
COLORS = ("red", "green", "blue", "yellow")
ALIASES = ("dax", "wug", "fep", "zot", "blick", "toma", "kiv", "nup")
SPLITS = ("train", "dev", "audit")
FOCI = ("grounding", "revision", "uncertainty")
FOCUSES = ("mixed",) + FOCI
KINDS = ("statement",) + FOCI
_RESIDUES = {"train": (0, 1), "dev": (2,), "audit": (3,)}
_EPISODE_FIELDS = {"id", "seed", "split", "focus", "version", "counterfactual_group", "turns"}
_TURN_FIELDS = {"text", "observations", "target", "reply", "kind"}
_WIDTHS = {"visual": 32, "auditory": 4, "body": 4, "feedback": 2}

DIALOGUE_CURRICULUM = {
    "grounding": {"prerequisites": (), "description": "Bind a tool name to a color and apply an English borrowing rule."},
    "revision": {"prerequisites": ("grounding",), "description": "Update the borrowing decision when the rule changes."},
    "uncertainty": {"prerequisites": ("grounding",), "description": "Request information when the tool meaning or rule is missing."},
}

_PHRASES = {
    "train": {
        "definition": ("A {alias} is the {color} tool.", "{alias} names the {color} tool."),
        "rule": ("Only {color} tools may be borrowed.", "You may borrow only {color} tools."),
        "correction": ("Now only {color} tools may be borrowed.", "Rule update: borrow only {color} tools."),
        "query": ("May I borrow the {alias}?", "Can I borrow the {alias}?"),
    },
    "dev": {
        "definition": ("The {color} tool is a {alias}.", "Call the {color} tool {alias}."),
        "rule": ("Borrow only tools that are {color}.", "The rule allows only {color} tools."),
        "correction": ("New rule: only {color} tools.", "Now borrow only tools that are {color}."),
        "query": ("Is borrowing the {alias} allowed?", "May the {alias} be borrowed?"),
    },
    "audit": {
        "definition": ("A {color} tool is called {alias}.", "The name of the {color} tool is {alias}."),
        "rule": ("Only tools colored {color} may be borrowed.", "Borrowing is allowed only for {color} tools."),
        "correction": ("Update: only tools colored {color}.", "Borrowing now allows only {color} tools."),
        "query": ("Is the {alias} available to borrow?", "Does the rule let me borrow the {alias}?"),
    },
}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def curriculum_digest() -> str:
    return hashlib.sha256(_json({"version": VERSION, "curriculum": DIALOGUE_CURRICULUM,
                                "phrases": _PHRASES, "bindings": _RESIDUES,
                                "colors": COLORS, "aliases": ALIASES,
                                "replies": REPLIES, "turns": NUM_TURNS}).encode()).hexdigest()


def _check_request(seed: int, split: str, focus: str) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(split, str) or split not in SPLITS:
        raise ValueError("unknown dialogue split")
    if not isinstance(focus, str) or focus not in FOCUSES:
        raise ValueError("unknown dialogue focus")


def _observations() -> dict:
    return {**{name: [[0.0] * width] for name, width in _WIDTHS.items()}, "tokens": [0]}


def allowed_bindings(split: str) -> tuple[tuple[str, str], ...]:
    """Reviewed split assignments; no alias/color binding appears in two splits."""
    if split not in SPLITS:
        raise ValueError("unknown dialogue split")
    return tuple((alias, color) for a, alias in enumerate(ALIASES)
                 for c, color in enumerate(COLORS) if (a + c) % 4 in _RESIDUES[split])


# Compile the finite, reviewed English grammar independently of episode state.
# Placeholder captures carry their own names, so reversed word orders retain
# the same meaning. No arbitrary code or instructions can extend this grammar.
def _grammar() -> tuple:
    patterns = []
    for phrases in _PHRASES.values():
        for kind, templates in phrases.items():
            for template in templates:
                pattern = re.escape(template)
                pattern = pattern.replace(re.escape("{alias}"), r"(?P<alias>[a-z]{2,8})")
                pattern = pattern.replace(re.escape("{color}"), "(?P<color>" + "|".join(COLORS) + ")")
                patterns.append((kind, re.compile(pattern)))
    return tuple(patterns)


_GRAMMAR = _grammar()


def parse_sentence(text: str) -> tuple[str, dict[str, str]]:
    """Parse one supported sentence, rejecting unsupported/unverified claims."""
    if not isinstance(text, str) or not 0 < len(text.encode("utf8")) <= MAX_TEXT_BYTES:
        raise ValueError("dialogue text must contain 1..128 UTF-8 bytes")
    for kind, pattern in _GRAMMAR:
        match = pattern.fullmatch(text)
        if match:
            return kind, match.groupdict()
    raise ValueError("sentence is outside the reviewed dialogue grammar")


def dialogue_oracle(episode: Mapping) -> list[int]:
    """Compute replies using only the preceding English, never target/metadata.

    Definitions bind names; a rule replaces the currently permitted color.
    Queries require both a known name and a rule. Missing either yields ASK.
    This parser is for generation and validation, not student inference.
    """
    if not isinstance(episode, Mapping) or not isinstance(episode.get("turns"), list):
        raise ValueError("dialogue must contain a list of turns")
    aliases, permitted, targets = {}, None, []
    for turn in episode["turns"]:
        if not isinstance(turn, Mapping):
            raise ValueError("each turn must be a mapping")
        kind, facts = parse_sentence(turn.get("text"))
        if kind == "definition":
            aliases[facts["alias"]] = facts["color"]
            answer = ACK
        elif kind in ("rule", "correction"):
            permitted = facts["color"]
            answer = ACK
        else:
            color = aliases.get(facts["alias"])
            answer = ASK if color is None or permitted is None else ALLOW if color == permitted else DENY
        targets.append(answer)
    return targets


def _generate_one(seed: int, split: str, focus: str) -> dict:
    pair = _json([VERSION, split, focus, seed // 2]).encode()
    digest = hashlib.sha256(pair).hexdigest()
    rng = random.Random(int(digest, 16))
    scenario = rng.choice(FOCI) if focus == "mixed" else focus
    bindings = allowed_bindings(split)
    alias_a, color_a = rng.choice(bindings)
    alias_b, color_b = rng.choice(tuple((alias, color) for alias, color in bindings
                                      if alias != alias_a and color != color_a))
    permitted = color_a if seed % 2 == 0 else color_b
    changed = color_b if seed % 2 == 0 else color_a
    phrases = _PHRASES[split]

    def sentence(kind, alias=alias_a, color=color_a):
        return rng.choice(phrases[kind]).format(alias=alias, color=color)

    definition_a = (sentence("definition"), "statement")
    definition_b = (sentence("definition", alias_b, color_b), "statement")
    rule = (sentence("rule", color=permitted), "statement")
    query_a = (sentence("query"), "grounding")
    query_b = (sentence("query", alias_b), "grounding")
    if scenario == "grounding":
        events = rng.choice((
            [definition_a, rule, query_a, definition_b, query_b, query_a],
            [rule, definition_a, query_a, definition_b, query_b, query_b],
            [definition_a, definition_b, rule, query_a, query_b, query_a],
            [definition_b, rule, query_b, definition_a, query_a, query_b],
        ))
    elif scenario == "revision":
        first = [definition_a, rule]
        rng.shuffle(first)
        correction = (sentence("correction", color=changed), "statement")
        # Identical question before and after correction is essential evidence.
        events = first + [query_a, correction, (query_a[0], "revision")]
        events.insert(rng.randrange(NUM_TURNS), (query_b[0], "uncertainty"))
    else:
        first = [definition_a, rule]
        rng.shuffle(first)
        unknown = (query_a[0], "uncertainty")
        events = [unknown, first[0], unknown, first[1], query_a]
        events.insert(rng.randrange(NUM_TURNS), (query_b[0], "uncertainty"))
    result = {
        "id": hashlib.sha256(_json([VERSION, seed, split, focus]).encode()).hexdigest(),
        "seed": seed, "split": split, "focus": focus, "version": VERSION,
        "counterfactual_group": digest,
        "turns": [{"text": text, "observations": _observations(), "kind": kind} for text, kind in events],
    }
    for turn, target in zip(result["turns"], dialogue_oracle(result)):
        turn.update(target=target, reply=REPLIES[target])
    return result


def generate_dialogues(seed: int, count: int, split: str = "train", focus: str = "mixed") -> list[dict]:
    """Generate independent conversations; record i always uses seed+i.

    Adjacent even/odd seeds form counterfactual pairs. For paired evaluation use
    an even starting seed and even count. Replay needs only seed and focus under
    the frozen version. Every episode starts with an empty semantic state.
    """
    _check_request(seed, split, focus)
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")
    return [_generate_one(seed + i, split, focus) for i in range(count)]


def validate_dialogue_inputs(episode: object) -> bool:
    """Check the input/output schema without invoking the semantic oracle.

    Evaluation can use this structural check without a tutor/parser dependency.
    Provenance and answer correctness require :func:`validate_dialogue` before
    admitting a lesson into training. Observations are intentionally all zero.
    """
    if type(episode) is not dict or set(episode) != _EPISODE_FIELDS:
        raise ValueError("unexpected dialogue fields")
    _check_request(episode["seed"], episode["split"], episode["focus"])
    if episode["version"] != VERSION:
        raise ValueError("unsupported dialogue version")
    if any(type(episode[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", episode[key]) is None
           for key in ("id", "counterfactual_group")):
        raise ValueError("invalid dialogue identity")
    turns = episode["turns"]
    if type(turns) is not list or len(turns) != NUM_TURNS:
        raise ValueError("dialogues require exactly six turns")
    for turn in turns:
        if type(turn) is not dict or set(turn) != _TURN_FIELDS:
            raise ValueError("unexpected dialogue turn fields")
        if type(turn["text"]) is not str or not 0 < len(turn["text"].encode("utf8")) <= MAX_TEXT_BYTES:
            raise ValueError("invalid dialogue text length")
        if type(turn["target"]) is not int or not 0 <= turn["target"] < len(TARGET_NAMES):
            raise ValueError("invalid dialogue target")
        if turn["reply"] != REPLIES[turn["target"]] or turn["kind"] not in KINDS:
            raise ValueError("invalid reply or diagnostic kind")
        observations = turn["observations"]
        if type(observations) is not dict or set(observations) != set(_WIDTHS) | {"tokens"}:
            raise ValueError("unexpected dialogue observation channels")
        tokens = observations["tokens"]
        if type(tokens) is not list or len(tokens) != 1 or type(tokens[0]) is not int or tokens[0] != 0:
            raise ValueError("dialogue task tokens must be zero")
        for name, width in _WIDTHS.items():
            rows = observations[name]
            if type(rows) is not list or len(rows) != 1 or type(rows[0]) is not list or len(rows[0]) != width:
                raise ValueError("dialogue observations must have exactly one correctly sized row")
            if any(type(value) not in (float, int) or not math.isfinite(value) or value != 0 for value in rows[0]):
                raise ValueError("dialogue nonlanguage inputs must be finite zero values")
    return True


def validate_dialogue(episode: object) -> bool:
    """Reject unverified text, labels, metadata, or noncanonical provenance."""
    validate_dialogue_inputs(episode)
    if dialogue_oracle(episode) != [turn["target"] for turn in episode["turns"]]:
        raise ValueError("dialogue targets disagree with the English-only oracle")
    if episode != _generate_one(episode["seed"], episode["split"], episode["focus"]):
        raise ValueError("dialogue disagrees with deterministic provenance")
    return True
