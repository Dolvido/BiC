"""V3 procedural event worlds with independently seeded bijective names.

The abstract interpreter and the frozen v2 English interpreter must agree.
Partition identities normalize names in ordered transcripts, not world meaning:
different equivalent descriptions need not receive the same identity. Nothing
here changes v2 generators, admission rules, checkpoints or completed studies.

The partition unit is a counterfactual PAIR. Two different pairs can share one
individual transcript; studies requiring unseen individual episodes must also
reject normalized individual transcript overlap when selecting their banks.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import random
import re

from experiments.cognitive_curriculum import (
    ALIASES, COLORS, FAMILIES, REPLIES, allowed_entity_pairs, cognitive_oracle,
)

VERSION = "bic-diverse-worlds-v3"
DENY, ALLOW, ASK, ACK = range(4)
SPLITS = ("train", "dev", "audit")
ROLES = tuple(f"x{i}" for i in range(len(ALIASES)))
_ROLE_PATTERN = re.compile(r"\bx(?:[0-9]|1[01])\b")
_WORLD_FIELDS = {"version", "family", "seed", "level", "requested_split", "attempt",
                 "primary_roles", "programs", "world_fingerprint", "world_partition"}
_ROW_FIELDS = {"version", "id", "family", "level", "split", "world_partition", "name_split",
               "world_seed", "world_requested_split", "world_attempt", "world_fingerprint",
               "naming_seed", "variant", "counterfactual_group", "turns"}
_FAMILY_OPS = {
    "variable_binding": {"color_set", "color_copy", "color_query"},
    "graph_reachability": {"links", "close_links", "path_query"},
    "arithmetic_updates": {"count_set", "count_copy", "count_add", "count_subtract", "count_query"},
    "conditional_logic": {"switch_set", "switch_pair", "intervention", "rule", "lamp_query"},
}


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("curriculum records must be finite JSON values") from error


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")


def _split(split):
    if type(split) is not str or split not in SPLITS:
        raise ValueError("split must be train, dev or audit")


def _event(op, **values):
    return {"op": op, **values}


def abstract_oracle(program, family):
    """Independent causal truth simulation, with no rendered English parser."""
    if family not in FAMILIES or not isinstance(program, (list, tuple)):
        raise ValueError("unsupported abstract program")
    colors, counts, switches, lamps = {}, {}, {}, {}
    edges, closed, answers = set(), False, []
    for event in program:
        if type(event) is not dict or event.get("op") not in _FAMILY_OPS[family]:
            raise ValueError("operation outside family")
        op, answer = event["op"], ACK
        name = event.get("name")
        if op == "color_set":
            colors[name] = event["color"]
        elif op == "color_copy":
            colors[name] = colors.get(event["source"])
        elif op == "color_query":
            answer = ASK if colors.get(name) is None else int(colors[name] == event["color"])
        elif op == "count_set":
            counts[name] = event["number"]
        elif op == "count_copy":
            if event["source"] in counts:
                counts[name] = counts[event["source"]]
            else:
                counts.pop(name, None)
        elif op in ("count_add", "count_subtract"):
            if name in counts:
                counts[name] += event["number"] * (1 if op == "count_add" else -1)
                if counts[name] < 0:
                    raise ValueError("negative count")
        elif op == "count_query":
            answer = ASK if name not in counts else int(counts[name] == event["number"])
        elif op == "links":
            if closed:
                raise ValueError("links after closure")
            edges.update(tuple(edge) for edge in event["edges"])
        elif op == "close_links":
            closed = True
        elif op == "path_query":
            # Transitive closure is deliberately separate from the text oracle's DFS.
            reachable = set(edges)
            while True:
                extended = reachable | {(a, d) for a, b in reachable for c, d in reachable if b == c}
                if extended == reachable:
                    break
                reachable = extended
            answer = ALLOW if (event["source"], event["destination"]) in reachable else DENY if closed else ASK
        elif op in ("switch_set", "switch_pair", "intervention"):
            switches[name] = event["value"]
            if op == "switch_pair":
                switches[event["other"]] = event["value"]
        elif op == "rule":
            lamps[name] = (event["operator"], tuple(event["inputs"]))
        elif op == "lamp_query":
            if name not in lamps:
                answer = ASK
            else:
                operator, inputs = lamps[name]
                states = [switches.get(item) for item in inputs]
                if operator == "all":
                    answer = DENY if any(value is False for value in states) else ASK if any(value is None for value in states) else ALLOW
                elif operator == "any":
                    answer = ALLOW if any(value is True for value in states) else ASK if any(value is None for value in states) else DENY
                else:
                    raise ValueError("unknown Boolean operator")
        answers.append(answer)
    return answers


def _sentence(event, names):
    op = event["op"]
    name = names[event["name"]] if "name" in event else None
    source = names[event["source"]] if "source" in event else None
    if op == "color_set":
        return f"{name} is {'now ' if event.get('revision', False) else ''}{event['color']}."
    if op == "color_copy":
        return f"Set {name} to the color of {source}."
    if op == "color_query":
        return f"Is {name} {event['color']}?"
    if op == "count_set":
        return f"{name} starts with a count of {event['number']}."
    if op == "count_copy":
        return f"Set the count of {name} to the count of {source}."
    if op in ("count_add", "count_subtract"):
        return f"{'Increase' if op == 'count_add' else 'Decrease'} the count of {name} by {event['number']}."
    if op == "count_query":
        return f"Is the count of {name} {event['number']}?"
    if op == "links":
        edges = event["edges"]
        if len(edges) == 1:
            a, b = edges[0]
            return f"There is a link from {names[a]} to {names[b]}."
        a, b = edges[0]
        c, d = edges[1]
        return f"There are links from {names[a]} to {names[b]} and from {names[c]} to {names[d]}."
    if op == "close_links":
        return "These are all the links."
    if op == "path_query":
        return f"Can a path follow links from {source} to {names[event['destination']]}?"
    if op in ("switch_set", "switch_pair", "intervention"):
        value = "on" if event["value"] else "off"
        if op == "switch_pair":
            return f"Switch {name} and switch {names[event['other']]} are {value}."
        return f"Set switch {name} to {value}." if op == "intervention" else f"Switch {name} is {value}."
    if op == "rule":
        inputs = [names[item] for item in event["inputs"]]
        if len(inputs) == 1:
            return f"Lamp {name} copies switch {inputs[0]}."
        return f"Lamp {name} is on exactly when {event['operator']} of {', '.join(inputs)} are on."
    if op == "lamp_query":
        return f"Is lamp {name} on?"
    raise ValueError("unrenderable operation")


def world_fingerprint(world_pair):
    """Name-independent syntactic pair identity, invariant to variant order."""
    programs = world_pair.get("programs") if isinstance(world_pair, dict) else None
    if type(programs) is not list or len(programs) != 2 or any(len(p) != 6 for p in programs):
        raise ValueError("world requires two six-turn programs")
    normalized = []
    for program in programs:
        aliases = {}
        def replace(match):
            if match[0] not in aliases:
                aliases[match[0]] = f"ENTITY{len(aliases)}"
            return aliases[match[0]]
        normalized.append([_ROLE_PATTERN.sub(replace, _sentence(event, dict(zip(ROLES, ROLES)))) for event in program])
    return _digest({"version": VERSION, "programs": sorted(normalized)})


def world_split(world_pair):
    residue = int(world_fingerprint(world_pair), 16) % 4
    return "train" if residue < 2 else "dev" if residue == 2 else "audit"


def naming_map(naming_seed, name_split="train"):
    """A bijection independent of world, family and difficulty; x0/x1 are primary."""
    _seed(naming_seed)
    _split(name_split)
    rng = random.Random(int(_digest([VERSION, "names", naming_seed, name_split]), 16))
    first, second = rng.choice(allowed_entity_pairs(name_split))
    rest = [name for name in ALIASES if name not in (first, second)]
    rng.shuffle(rest)
    return dict(zip(ROLES, [first, second, *rest]))


def _binding(rng, level):
    initial, changed = rng.sample(COLORS, 2)
    first = _event("color_set", name="x0", color=initial)
    events = [first, _event("color_set", name="x1", color=rng.choice(COLORS))]
    rng.shuffle(events)
    events.append(_event("color_query", name="x0", color=initial))
    if level == 3:
        copied = rng.choice(("x1", "x2"))
        revised = rng.choice(("x0", copied))
        queried = rng.choice(("x0", copied, "x3"))
        events += [_event("color_copy", name=copied, source="x0"),
                   _event("color_set", name=revised, color=rng.choice(COLORS), revision=True),
                   _event("color_query", name=queried, color=rng.choice((initial, changed)))]
        other = deepcopy(events)
        other[events.index(first)]["color"] = changed
        return [events, other]
    if level == 2 and rng.choice((False, True)):
        events.append(_event("color_copy", name=rng.choice(("x0", "x1", "x2")), source=rng.choice(("x0", "x1", "x3"))))
    else:
        events.append(_event("color_set", name=rng.choice(("x0", "x1", "x2")), color=rng.choice(COLORS), revision=level > 1))
    events.append(_event("color_query", name=rng.choice(("x0", "x1", "x2")), color=rng.choice(COLORS)))
    events.insert(rng.randrange(6), _event("color_query", name=rng.choice(("x3", "x4")), color=rng.choice(COLORS)))
    other = deepcopy(events)
    other[events.index(first)]["color"] = changed
    return [events, other]


def _arithmetic(rng, level):
    start, delta, second = rng.randint(10, 25), rng.randint(1, 7), rng.randint(1, 3)
    operation = rng.choice(("count_add", "count_subtract"))
    total = start + (delta if operation == "count_add" else -delta)
    events = [_event("count_set", name="x0", number=start), _event(operation, name="x0", number=delta),
              _event("count_query", name="x0", number=total)]
    if level == 3:
        events += [_event("count_copy", name="x1", source="x0"), _event("count_add", name="x1", number=second),
                   _event("count_query", name="x1", number=total + second)]
    else:
        if rng.choice((False, True)):
            mutation = _event("count_copy", name="x1", source="x0") if level == 2 else _event("count_set", name="x1", number=second)
            queried, expected = "x1", total if level == 2 else second
        else:
            mutation = _event("count_add", name="x0", number=second)
            queried, expected = "x0", total + second
        events += [mutation, _event("count_query", name=queried, number=expected + rng.choice((0, 1)))]
        unknown = "x2" if mutation["name"] == "x1" else "x1"
        events.insert(rng.randrange(6), _event("count_query", name=unknown, number=rng.randint(0, 25)))
    other = deepcopy(events)
    next(event for event in other if event["op"] == "count_set" and event["name"] == "x0")["number"] += 1
    return [events, other]


def _graph(rng, level):
    path = [("x0", "x1")] if level == 1 else [("x0", "x2"), ("x2", "x1")] if level == 2 else [("x0", "x2"), ("x2", "x3"), ("x3", "x1")]
    edges = list(path)
    choices = [(a, b) for a in ROLES[:6] for b in ROLES[:6] if a != b and (a, b) not in edges]
    edges += rng.sample(choices, 4 - len(edges))
    broken = rng.randrange(len(path))
    distractor = rng.randrange(len(path), 4)
    coupled = [edges[broken], edges[distractor]]
    rng.shuffle(coupled)
    facts = [_event("links", edges=[list(edge) for edge in coupled])]
    facts += [_event("links", edges=[list(edge)]) for index, edge in enumerate(edges) if index not in (broken, distractor)]
    rng.shuffle(facts)
    early_source, early_destination = rng.sample(ROLES[:6], 2)
    events = facts[:]
    events.insert(rng.randint(1, 3), _event("path_query", source=early_source, destination=early_destination))
    events += [_event("close_links"), _event("path_query", source="x0", destination="x1")]
    other = deepcopy(events)
    compound = next(event for event in other if event["op"] == "links" and len(event["edges"]) == 2)
    compound["edges"][0][1], compound["edges"][1][1] = compound["edges"][1][1], compound["edges"][0][1]
    return [events, other]


def _conditional(rng, level):
    switches = ["x0", "x1", "x3"]
    pair = rng.sample(switches, 2)
    remaining = next(name for name in switches if name not in pair)
    facts = [_event("switch_pair", name=pair[0], other=pair[1], value=rng.choice((False, True))),
             _event("switch_set", name=remaining, value=rng.choice((False, True)))]
    changed = rng.randrange(2)
    altered = facts[changed]
    inputs = rng.sample(switches, level)
    facts.append(_event("rule", name="x2", operator=rng.choice(("all", "any")) if level > 1 else "all", inputs=inputs))
    rng.shuffle(facts)
    query_names = ("x2", "x2", "x2", "x0", "x1", "x3", "x4")
    events = facts + [_event("lamp_query", name=rng.choice(query_names)),
        _event("intervention", name=rng.choice((*switches, "x4")), value=rng.choice((False, True))),
        _event("lamp_query", name=rng.choice(query_names))]
    other = deepcopy(events)
    other[events.index(altered)]["value"] = not altered["value"]
    return [events, other]


def _pair_truth(programs, family):
    if len(programs) != 2 or any(len(program) != 6 for program in programs):
        return None
    changed = [index for index, (a, b) in enumerate(zip(*programs)) if a != b]
    if len(changed) != 1 or programs[0][changed[0]]["op"].endswith("query"):
        return None
    if family == "graph_reachability":
        for program in programs:
            edges = [tuple(edge) for event in program if event["op"] == "links" for edge in event["edges"]]
            if len(set(edges)) != len(edges):
                return None
    answers = [abstract_oracle(program, family) for program in programs]
    if not any({a, b} == {DENY, ALLOW} for a, b in zip(*answers)):
        return None
    return answers


@lru_cache(maxsize=8192)
def _world_cached(family, seed, level, requested_split):
    builders = dict(zip(FAMILIES, (_binding, _graph, _arithmetic, _conditional)))
    for attempt in range(4096):
        rng = random.Random(int(_digest([VERSION, family, seed, level, attempt]), 16))
        programs = builders[family](rng, level)
        if _pair_truth(programs, family) is None:
            continue
        world = {"version": VERSION, "family": family, "seed": seed, "level": level,
                 "requested_split": requested_split, "attempt": attempt,
                 "primary_roles": ["x0", "x1"], "programs": programs}
        world["world_fingerprint"] = world_fingerprint(world)
        world["world_partition"] = world_split(world)
        if requested_split is None or world["world_partition"] == requested_split:
            return world
    raise ValueError("no admissible world within deterministic attempt budget")


def generate_world_pair(family, seed, level=2, split=None):
    """Generate a verified anonymous pair; optional split rejection is deterministic."""
    if type(family) is not str or family not in FAMILIES:
        raise ValueError("unknown family")
    _seed(seed)
    if type(level) is not int or level not in (1, 2, 3):
        raise ValueError("level must be 1, 2 or 3")
    if split is not None:
        _split(split)
    return deepcopy(_world_cached(family, seed, level, split))


def _verify_world(world):
    if type(world) is not dict or set(world) != _WORLD_FIELDS:
        raise ValueError("unexpected world provenance")
    expected = generate_world_pair(world["family"], world["seed"], world["level"], world["requested_split"])
    if _json(world) != _json(expected):
        raise ValueError("world differs from deterministic provenance")


def _observations():
    return {**{name: [[0.0] * width] for name, width in (("visual", 32), ("auditory", 4), ("body", 4), ("feedback", 2))}, "tokens": [0]}


def _render(world, naming_seed, split, name_split):
    _split(split)
    _split(name_split)
    names = naming_map(naming_seed, name_split)
    if split == "train" and (world["world_partition"] != "train" or name_split != "train"):
        raise ValueError("train admission requires train world and name partitions")
    answers = _pair_truth(world["programs"], world["family"])
    group = _digest([VERSION, world["world_fingerprint"], names, split, name_split])
    rows = []
    for variant, program in enumerate(world["programs"]):
        turns = [{"text": _sentence(event, names), "observations": _observations(), "target": target,
                  "reply": REPLIES[target], "kind": "statement" if target == ACK else world["family"]}
                 for event, target in zip(program, answers[variant])]
        row = {"version": VERSION, "family": world["family"], "level": world["level"], "split": split,
            "world_partition": world["world_partition"], "name_split": name_split,
            "world_seed": world["seed"], "world_requested_split": world["requested_split"],
            "world_attempt": world["attempt"], "world_fingerprint": world["world_fingerprint"],
            "naming_seed": naming_seed, "variant": variant, "counterfactual_group": group, "turns": turns}
        if cognitive_oracle(row) != answers[variant]:
            raise ValueError("independent abstract and English oracles disagree")
        row["id"] = _digest(row)
        rows.append(row)
    return rows


def render_world_pair(world_pair, naming_seed, split="train", name_split=None):
    """Render a shared bijection, with independent world/name evaluation panels."""
    _verify_world(world_pair)
    return _render(world_pair, naming_seed, split, split if name_split is None else name_split)


def validate_diverse(row):
    """Authenticate exact v3 provenance, admission, both oracles and every field."""
    if type(row) is not dict or set(row) != _ROW_FIELDS:
        raise ValueError("unexpected diverse row fields")
    if type(row["variant"]) is not int or row["variant"] not in (0, 1):
        raise ValueError("variant must be integer zero or one")
    world = generate_world_pair(row["family"], row["world_seed"], row["level"], row["world_requested_split"])
    expected = _render(world, row["naming_seed"], row["split"], row["name_split"])[row["variant"]]
    if _json(row) != _json(expected):
        raise ValueError("row differs from canonical v3 provenance")
    return True


def validate_pair(rows):
    if not isinstance(rows, (list, tuple)) or len(rows) != 2:
        raise ValueError("exactly two adjacent pair members are required")
    for row in rows:
        validate_diverse(row)
    if ([row["variant"] for row in rows] != [0, 1]
            or rows[0]["counterfactual_group"] != rows[1]["counterfactual_group"]):
        raise ValueError("pair requires matching canonical variants zero then one")
    left, right = rows
    if any(left[key] != right[key] for key in _ROW_FIELDS - {"id", "variant", "turns"}):
        raise ValueError("pair members require identical world and naming provenance")
    changed = [index for index, (a, b) in enumerate(zip(left["turns"], right["turns"])) if a["text"] != b["text"]]
    if len(changed) != 1 or left["turns"][changed[0]]["target"] != ACK or right["turns"][changed[0]]["target"] != ACK:
        raise ValueError("counterfactual changes exactly one statement")
    if not any({a["target"], b["target"]} == {DENY, ALLOW} for a, b in zip(left["turns"], right["turns"])):
        raise ValueError("counterfactual must flip at least one known answer")
    return True


def generate_diverse(seed, count, split="train", family="mixed", level=2, world_split=None, name_split=None):
    """Convenience complete-pair bank; factorial callers should compose the two APIs."""
    _seed(seed)
    if type(count) is not int or count < 2 or count % 2:
        raise ValueError("count must be positive and contain complete pairs")
    _split(split)
    if family != "mixed" and family not in FAMILIES:
        raise ValueError("unknown family")
    world_partition = split if world_split is None else world_split
    naming_partition = split if name_split is None else name_split
    rows = []
    for index in range(count // 2):
        subject = FAMILIES[(seed + index) % len(FAMILIES)] if family == "mixed" else family
        world = generate_world_pair(subject, seed + 2 * index, level, world_partition)
        rows.extend(render_world_pair(world, seed + 2 * index + 1, split, naming_partition))
    return rows
