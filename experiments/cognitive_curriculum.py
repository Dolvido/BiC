"""Compact, verified worlds for four kinds of elementary reasoning.

Only English enters the student: all numeric observations and task tokens are
zero. The interpreter belongs to lesson construction/verification, never policy
inference. Every adjacent even/odd seed pair changes one statement, retaining the
questions and event order; at least one known answer changes truth value.
For graphs, that statement swaps the destinations of two independently named
links, preserving every node's in/out degree while changing connectivity.

Splits withhold ordered *primary* entity pairs. Incidental pairs and vocabulary
can overlap. Levels are available in every split: a depth holdout is a runner
decision, not a claim made by this generator. These are small procedural worlds,
not a test or definition of general intelligence.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping

VERSION = "bic-cognitive-worlds-v2"
DENY, ALLOW, ASK, ACK = range(4)
REPLIES = ("No.", "Yes.", "I need more information.", "Understood.")
ALIASES = ("dax", "wug", "fep", "zot", "blick", "toma", "kiv", "nup", "ral", "sog", "vex", "pim")
COLORS = ("red", "green", "blue", "yellow")
FAMILIES = ("variable_binding", "graph_reachability", "arithmetic_updates", "conditional_logic")
SPLITS = ("train", "dev", "audit")
FAMILY_METADATA = {
    "variable_binding": {"description": "Remember, revise and copy temporary name-to-color bindings.",
                         "levels": ["bind", "replace", "copy then replace the source"]},
    "graph_reachability": {"description": "Follow directed links; distinguish unknown from false after explicit closure.",
                           "levels": ["one link", "two-link path", "three-link path"]},
    "arithmetic_updates": {"description": "Track a count through independently stated changes.",
                           "levels": ["one increase", "increase then decrease", "copy and update a derived count"]},
    "conditional_logic": {"description": "Apply an exact Boolean rule and recompute after an intervention.",
                          "levels": ["one input", "two inputs", "three inputs"]},
}
_RESIDUES = {"train": (0, 1), "dev": (2,), "audit": (3,)}
_WIDTHS = {"visual": 32, "auditory": 4, "body": 4, "feedback": 2}
_FIELDS = {"id", "seed", "split", "family", "requested_family", "level", "version", "counterfactual_group", "turns"}
_TURN_FIELDS = {"text", "observations", "target", "reply", "kind"}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def curriculum_digest() -> str:
    """Recipe identity; file hashes should also be recorded by experiment runners."""
    return _digest({"version": VERSION, "families": FAMILY_METADATA, "aliases": ALIASES,
                    "colors": COLORS, "splits": _RESIDUES, "replies": REPLIES})


def allowed_entity_pairs(split: str) -> tuple[tuple[str, str], ...]:
    if split not in SPLITS:
        raise ValueError("unknown cognitive split")
    return tuple((a, b) for i, a in enumerate(ALIASES) for j, b in enumerate(ALIASES)
                 if a != b and (i + j) % 4 in _RESIDUES[split])


def _observations() -> dict:
    return {**{name: [[0.0] * width] for name, width in _WIDTHS.items()}, "tokens": [0]}


_NAME = "(?:" + "|".join(ALIASES) + ")"
_COLOR = "(?:" + "|".join(COLORS) + ")"
_GRAMMAR = (
    ("color_set", rf"(?P<name>{_NAME}) is (?:now )?(?P<color>{_COLOR})\."),
    ("color_copy", rf"Set (?P<name>{_NAME}) to the color of (?P<source>{_NAME})\."),
    ("color_query", rf"Is (?P<name>{_NAME}) (?P<color>{_COLOR})\?"),
    ("link", rf"There is a link from (?P<source>{_NAME}) to (?P<destination>{_NAME})\."),
    ("link_pair", rf"There are links from (?P<source>{_NAME}) to (?P<destination>{_NAME}) and from (?P<other_source>{_NAME}) to (?P<other_destination>{_NAME})\."),
    ("close_links", r"These are all the links\."),
    ("path_query", rf"Can a path follow links from (?P<source>{_NAME}) to (?P<destination>{_NAME})\?"),
    ("count_set", rf"(?P<name>{_NAME}) starts with a count of (?P<number>[0-9]{{1,2}})\."),
    ("count_add", rf"Increase the count of (?P<name>{_NAME}) by (?P<number>[0-9]{{1,2}})\."),
    ("count_subtract", rf"Decrease the count of (?P<name>{_NAME}) by (?P<number>[0-9]{{1,2}})\."),
    ("count_copy", rf"Set the count of (?P<name>{_NAME}) to the count of (?P<source>{_NAME})\."),
    ("count_query", rf"Is the count of (?P<name>{_NAME}) (?P<number>[0-9]{{1,2}})\?"),
    ("switch_set", rf"Switch (?P<name>{_NAME}) is (?P<value>on|off)\."),
    ("switch_pair", rf"Switch (?P<name>{_NAME}) and switch (?P<other>{_NAME}) are (?P<value>on|off)\."),
    ("intervention", rf"Set switch (?P<name>{_NAME}) to (?P<value>on|off)\."),
    ("copy_rule", rf"Lamp (?P<name>{_NAME}) copies switch (?P<inputs>{_NAME})\."),
    ("boolean_rule", rf"Lamp (?P<name>{_NAME}) is on exactly when (?P<operator>all|any) of (?P<inputs>{_NAME}(?:, {_NAME}){{1,2}}) are on\."),
    ("lamp_query", rf"Is lamp (?P<name>{_NAME}) on\?"),
)
_PATTERNS = tuple((kind, re.compile(pattern)) for kind, pattern in _GRAMMAR)


def parse_cognitive_sentence(text: str) -> tuple[str, dict[str, str]]:
    """Parse a reviewed statement; unsupported or overlong English is rejected."""
    if type(text) is not str or not 0 < len(text.encode("utf8")) <= 128:
        raise ValueError("cognitive English must contain 1..128 UTF-8 bytes")
    for kind, pattern in _PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            return kind, match.groupdict()
    raise ValueError("English lies outside the reviewed cognitive grammar")


_FAMILY_KINDS = {
    "variable_binding": {"color_set", "color_copy", "color_query"},
    "graph_reachability": {"link", "link_pair", "close_links", "path_query"},
    "arithmetic_updates": {"count_set", "count_add", "count_subtract", "count_copy", "count_query"},
    "conditional_logic": {"switch_set", "switch_pair", "intervention", "copy_rule", "boolean_rule", "lamp_query"},
}


def cognitive_oracle(episode: Mapping) -> list[int]:
    """Evaluate text causally from an empty world, without reading answer labels.

    Undefined colors/counts and undetermined lamps yield ASK. A missing path
    yields ASK until links are explicitly declared complete, then DENY. Boolean
    rules use three-valued logic: one false ALL input proves false; one true ANY
    input proves true, even when another input is not yet known. Copying an
    unknown color erases the target's established color rather than inventing it.
    """
    if not isinstance(episode, Mapping) or episode.get("family") not in FAMILIES:
        raise ValueError("oracle requires one supported cognitive family")
    if not isinstance(episode.get("turns"), list):
        raise ValueError("oracle requires a list of turns")
    colors, counts, switches, rules, links = {}, {}, {}, {}, {}
    closed = False
    answers = []

    def truth(value):
        return ASK if value is None else ALLOW if value else DENY

    def reachable(source, destination):
        pending, seen = list(links.get(source, ())), set()
        while pending:
            node = pending.pop()
            if node == destination:
                return True
            if node not in seen:
                seen.add(node)
                pending.extend(links.get(node, ()))
        return False

    for turn in episode["turns"]:
        if not isinstance(turn, Mapping):
            raise ValueError("oracle turns must be mappings")
        kind, facts = parse_cognitive_sentence(turn.get("text"))
        if kind not in _FAMILY_KINDS[episode["family"]]:
            raise ValueError("sentence does not belong to the declared family")
        answer = ACK
        name = facts.get("name")
        if kind == "color_set":
            colors[name] = facts["color"]
        elif kind == "color_copy":
            colors[name] = colors.get(facts["source"])
        elif kind == "color_query":
            answer = truth(None if colors.get(name) is None else colors[name] == facts["color"])
        elif kind in ("link", "link_pair"):
            if closed:
                raise ValueError("links cannot be added after explicit closure")
            links.setdefault(facts["source"], set()).add(facts["destination"])
            if kind == "link_pair":
                links.setdefault(facts["other_source"], set()).add(facts["other_destination"])
        elif kind == "close_links":
            closed = True
        elif kind == "path_query":
            answer = ALLOW if reachable(facts["source"], facts["destination"]) else DENY if closed else ASK
        elif kind == "count_set":
            counts[name] = int(facts["number"])
        elif kind == "count_copy":
            if facts["source"] in counts:
                counts[name] = counts[facts["source"]]
            else:
                counts.pop(name, None)
        elif kind in ("count_add", "count_subtract"):
            if name in counts:
                counts[name] += int(facts["number"]) * (1 if kind == "count_add" else -1)
                if counts[name] < 0:
                    raise ValueError("a count cannot become negative")
        elif kind == "count_query":
            answer = truth(None if name not in counts else counts[name] == int(facts["number"]))
        elif kind in ("switch_set", "switch_pair", "intervention"):
            switches[name] = facts["value"] == "on"
            if kind == "switch_pair":
                switches[facts["other"]] = switches[name]
        elif kind in ("copy_rule", "boolean_rule"):
            inputs = tuple(facts["inputs"].split(", "))
            if len(inputs) != len(set(inputs)):
                raise ValueError("a Boolean rule cannot repeat an input")
            rules[name] = (facts.get("operator", "all"), inputs)
        elif kind == "lamp_query":
            if name not in rules:
                answer = ASK
            else:
                operator, inputs = rules[name]
                values = [switches.get(item) for item in inputs]
                if operator == "all":
                    value = False if False in values else None if None in values else True
                else:
                    value = True if True in values else None if None in values else False
                answer = truth(value)
        answers.append(answer)
    return answers


def _check(seed: int, split: str, family: str, level: int) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(split) is not str or split not in SPLITS:
        raise ValueError("unknown cognitive split")
    if type(family) is not str or family not in ("mixed", *FAMILIES):
        raise ValueError("unknown cognitive family")
    if type(level) is not int or not 1 <= level <= 3:
        raise ValueError("cognitive level must be 1, 2 or 3")


def _generate_one(seed: int, split: str, requested_family: str, level: int) -> dict:
    group = _digest([VERSION, split, requested_family, level, seed // 2])
    rng = random.Random(int(group, 16))
    family = rng.choice(FAMILIES) if requested_family == "mixed" else requested_family
    a, b = rng.choice(allowed_entity_pairs(split))
    c, d, e, f = rng.sample([alias for alias in ALIASES if alias not in (a, b)], 4)
    even = seed % 2 == 0
    if family == "variable_binding":
        first, other, third, fourth = rng.sample(COLORS, len(COLORS))
        events = [f"{a} is {first if even else other}.", f"{b} is {third}."]
        rng.shuffle(events)
        events += [f"Is {a} {first}?"]
        if level == 1:
            queried, expected = rng.choice(((a, first), (b, third), (c, fourth)))
            if rng.choice((False, True)):
                expected = rng.choice(tuple(color for color in COLORS if color != expected))
            events += [f"{c} is {fourth}.", f"Is {queried} {expected}?"]
            events.insert(rng.randrange(6), f"Is {d} {rng.choice(COLORS)}?")
        elif level == 2:
            queried, expected = rng.choice(((a, fourth), (b, third)))
            if rng.choice((False, True)):
                expected = rng.choice(tuple(color for color in COLORS if color != expected))
            events += [f"{a} is now {fourth}.", f"Is {queried} {expected}?"]
            events.insert(rng.randrange(6), f"Is {d} {rng.choice(COLORS)}?")
        else:
            queried, expected = rng.choice(((a, third), (b, third), (c, first), (d, fourth)))
            if rng.choice((False, True)):
                expected = rng.choice(tuple(color for color in COLORS if color != expected))
            events += [f"Set {c} to the color of {a}.", f"Set {a} to the color of {b}.", f"Is {queried} {expected}?"]
    elif family == "graph_reachability":
        # At levels 2/3 both variants keep the queried source's outgoing degree,
        # destination's incoming degree and terminal edge. Truth requires
        # matching the internal links rather than a final-edge direction cue.
        source, destination = (a, b) if rng.choice((False, True)) else (b, a)
        edges = ([(source, destination), (c, d), (d, e), (e, f)] if level == 1 else
                 [(source, c), (c, destination), (d, e), (e, f)] if level == 2 else
                 [(source, c), (c, d), (d, destination), (e, f)])
        broken_index = rng.choice((0, 1)) if level == 3 else 0
        distractor_index = rng.choice((2, 3)) if level == 2 else 3
        unchanged = [edge for index, edge in enumerate(edges) if index not in (broken_index, distractor_index)]
        if not even:
            path_edge, distractor_edge = edges[broken_index], edges[distractor_index]
            edges[broken_index] = (path_edge[0], distractor_edge[1])
            edges[distractor_index] = (distractor_edge[0], path_edge[1])
        # Four edges fit in three fact turns through one two-link sentence.
        # The swapped edges share that sentence, so exactly one statement
        # differs. Every node and its in/out degree survive the edge swap.
        # Unlabelled whole-graph motifs are not claimed to be fully balanced.
        coupled = [edges[broken_index], edges[distractor_index]]
        rng.shuffle(coupled)
        groups = [coupled, *[[edge] for edge in unchanged]]
        rng.shuffle(groups)
        events = []
        for group_edges in groups:
            start, end = group_edges[0]
            if len(group_edges) == 1:
                events.append(f"There is a link from {start} to {end}.")
            else:
                other_start, other_end = group_edges[1]
                events.append(f"There are links from {start} to {end} and from {other_start} to {other_end}.")
        query_position = rng.randint(1, 3)
        established = [edge for group_edges in groups[:query_position] for edge in group_edges if edge in unchanged]
        if established and rng.choice((False, True)):
            early_source, early_destination = rng.choice(established)
        else:
            early_source = rng.choice((destination, f))
            early_destination = rng.choice(tuple(node for node in (a, b, c, d, e, f) if node != early_source))
        events.insert(query_position, f"Can a path follow links from {early_source} to {early_destination}?")
        events += ["These are all the links.", f"Can a path follow links from {source} to {destination}?"]
    elif family == "arithmetic_updates":
        initial, increment, decrement, last = rng.randint(4, 8), rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 3)
        total = initial + increment
        events = [f"{a} starts with a count of {initial if even else initial + 1}.",
                  f"Increase the count of {a} by {increment}.", f"Is the count of {a} {total}?"]
        if level == 1:
            events += [f"{b} starts with a count of {last}.", f"Is the count of {c} {last}?"]
        elif level == 2:
            total -= decrement
            events += [f"Decrease the count of {a} by {decrement}.", f"Is the count of {b} {last}?"]
        else:
            total += last
            events += [f"Set the count of {b} to the count of {a}.", f"Increase the count of {b} by {last}."]
        events += [f"Is the count of {b if level == 3 else a} {total}?"]
    else:
        operator = rng.choice(("all", "any"))
        neutral = "on" if operator == "all" else "off"
        intervention = rng.choice(("on", "off"))
        intervened = a if level == 1 else rng.choice((b, c)[:level - 1])
        rule = (f"Lamp {d} copies switch {a}." if level == 1 else
                f"Lamp {d} is on exactly when {operator} of {', '.join((a, b, c)[:level])} are on.")
        queried_lamp = e if (seed // 2) % 4 == 0 else d
        events = [f"Switch {a} is {'on' if even else 'off'}.",
                  f"Switch {b} and switch {c} are {neutral}.", rule,
                  f"Is lamp {d} on?", f"Set switch {intervened} to {intervention}.", f"Is lamp {queried_lamp} on?"]
    episode = {"id": _digest([VERSION, seed, split, requested_family, level]), "seed": seed,
               "split": split, "family": family, "requested_family": requested_family, "level": level,
               "version": VERSION, "counterfactual_group": group,
               "turns": [{"text": text, "observations": _observations()} for text in events]}
    for turn, target in zip(episode["turns"], cognitive_oracle(episode)):
        turn.update(target=target, reply=REPLIES[target], kind="statement" if target == ACK else family)
    return episode


def generate_cognitive(seed: int, count: int, split: str = "train", family: str = "mixed", level: int = 1) -> list[dict]:
    """Generate complete pairs from a compact seed/family/level recipe.

    Use exact family requests when equal family representation is needed;
    ``mixed`` is seeded random sampling. Count zero is allowed for empty banks.
    """
    _check(seed, split, family, level)
    if seed % 2 or type(count) is not int or count < 0 or count % 2:
        raise ValueError("cognitive banks require an even seed and nonnegative even count")
    return [_generate_one(seed + index, split, family, level) for index in range(count)]


def validate_cognitive(episode: object) -> bool:
    """Authenticate exact recipe, all labels and the zero observation boundary."""
    if type(episode) is not dict or set(episode) != _FIELDS:
        raise ValueError("unexpected cognitive episode fields")
    _check(episode["seed"], episode["split"], episode["requested_family"], episode["level"])
    if episode["family"] not in FAMILIES or episode["version"] != VERSION:
        raise ValueError("invalid cognitive family or version")
    if any(type(episode[key]) is not str or re.fullmatch("[0-9a-f]{64}", episode[key]) is None
           for key in ("id", "counterfactual_group")):
        raise ValueError("invalid cognitive identity")
    if type(episode["turns"]) is not list or len(episode["turns"]) != 6:
        raise ValueError("cognitive episodes require exactly six turns")
    for turn in episode["turns"]:
        if type(turn) is not dict or set(turn) != _TURN_FIELDS:
            raise ValueError("unexpected cognitive turn fields")
        parse_cognitive_sentence(turn["text"])
        if type(turn["target"]) is not int or not 0 <= turn["target"] < 4:
            raise ValueError("invalid cognitive target")
        if turn["reply"] != REPLIES[turn["target"]] or turn["kind"] != ("statement" if turn["target"] == ACK else episode["family"]):
            raise ValueError("invalid cognitive reply or kind")
        observations = turn["observations"]
        if type(observations) is not dict or set(observations) != {*_WIDTHS, "tokens"}:
            raise ValueError("unexpected cognitive observations")
        if type(observations["tokens"]) is not list or observations["tokens"] != [0] or type(observations["tokens"][0]) is not int:
            raise ValueError("cognitive tokens must be exactly integer zero")
        for name, width in _WIDTHS.items():
            rows = observations[name]
            if type(rows) is not list or len(rows) != 1 or type(rows[0]) is not list or len(rows[0]) != width:
                raise ValueError("cognitive observations must have the fixed one-row shape")
            if any(type(value) not in (int, float) or not math.isfinite(value) or value != 0 for value in rows[0]):
                raise ValueError("cognitive observations must be finite zero values")
    if cognitive_oracle(episode) != [turn["target"] for turn in episode["turns"]]:
        raise ValueError("cognitive labels disagree with the English-only oracle")
    if episode != _generate_one(episode["seed"], episode["split"], episode["requested_family"], episode["level"]):
        raise ValueError("cognitive episode disagrees with deterministic provenance")
    return True


def cognitive_bank(seed: int, count_per_family: int = 32, *, split: str = "dev", level: int = 1) -> dict[str, list[dict]]:
    """Return balanced family slices with nonoverlapping seed blocks."""
    return {family: generate_cognitive(seed + index * count_per_family, count_per_family, split, family, level)
            for index, family in enumerate(FAMILIES)}
