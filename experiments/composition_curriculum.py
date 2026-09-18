"""Shared typed state compositions with held-out supervised dependency motifs.

One procedure sampler serves color, count and switch domains. Final-query
version ancestry, not distractor wording, owns the structural partition.
Identities describe syntactic dependency chains, not semantic equivalence or
unseen algorithms. Only final queries must be known; unknown probes are earlier.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import random
import re

from experiments.cognitive_curriculum import ALIASES, REPLIES


VERSION = "bic-shared-composition-v1"
FAMILIES = ("color", "count", "switch")
SPLITS = ("train", "dev", "audit")
COLORS = ("red", "green", "blue", "yellow")
DENY, ALLOW, ASK, ACK = range(4)
ROLES = tuple(f"x{i}" for i in range(len(ALIASES)))
_FIELDS = {"version", "id", "family", "split", "variant", "counterfactual_group",
           "recipe", "structure_id", "structure_partition", "program_id", "query_ancestries", "turns"}


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("finite canonical JSON required") from error


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")


def _split(value):
    if type(value) is not str or value not in SPLITS:
        raise ValueError("partition must be train, dev or audit")


def naming_map(seed):
    """A whole-vocabulary bijection independent of values, family and structure."""
    _seed(seed)
    names = list(ALIASES)
    random.Random(int(_hash([VERSION, "names", seed]), 16)).shuffle(names)
    return dict(zip(ROLES, names))


def _partition(identity):
    bucket = int(identity, 16) % 5
    return "train" if bucket < 3 else "dev" if bucket == 3 else "audit"


def _normalized(events):
    """Keep ordered reads/writes and role reuse, dropping all typed constants."""
    roles, result = {}, []
    def role(name):
        if name not in roles:
            roles[name] = f"v{len(roles)}"
        return roles[name]
    for event in events:
        op = event["op"]
        if op == "copy":
            result.append([op, "read", role(event["source"]), "write", role(event["name"])])
        elif op == "advance":
            result.append([op, "read_write", role(event["name"])])
        elif op == "set":
            result.append([op, "write", role(event["name"])])
        elif op == "query":
            result.append([op, "read", role(event["name"])])
        else:
            raise ValueError("unknown procedure operation")
    return result


def _ancestry_records(program):
    """Trace immutable value versions, so copies survive later source writes."""
    cells, nodes, records = {}, {}, []
    for index, event in enumerate(program):
        op, name = event["op"], event["name"]
        if op == "set":
            nodes[index] = (event, None)
            cells[name] = index
        elif op in ("copy", "advance"):
            parent = cells.get(event["source"] if op == "copy" else name)
            nodes[index] = (event, parent)
            cells[name] = index if parent is not None else None
        elif op == "query":
            node = cells.get(name)
            chain, root = [], None
            while node is not None:
                root = node
                prior, node = nodes[node]
                chain.append(prior)
            chain.reverse()
            known = bool(chain)
            signature = _normalized([*chain, event]) if known else None
            identity = _hash([VERSION, "query-ancestry", signature]) if known else None
            operations = {step["op"] for step in chain}
            records.append({"turn_index": index, "known": known,
                "composed": {"copy", "advance"}.issubset(operations),
                "structure_id": identity, "structure_partition": _partition(identity) if known else None,
                "root_index": root, "signature": signature})
        else:
            raise ValueError("unknown procedure operation")
    return records


def _public_ancestries(program):
    return [{key: value for key, value in row.items() if key not in ("root_index", "signature")}
            for row in _ancestry_records(program)]


def query_ancestries(row_or_program):
    """Return every supervised query's causal identity; no family labels enter it."""
    program = _parse_row(row_or_program) if isinstance(row_or_program, dict) else row_or_program
    return _public_ancestries(program)


def structure_id(row_or_program):
    """Final-query ancestry identity, invariant to irrelevant events and names."""
    program = _parse_row(row_or_program) if isinstance(row_or_program, dict) else row_or_program
    records = _ancestry_records(program)
    if not records or records[-1]["turn_index"] != len(program) - 1 or not records[-1]["known"]:
        raise ValueError("final event must query a known value")
    return records[-1]["structure_id"]


def _value_valid(family, value):
    return ((family == "color" and type(value) is str and value in COLORS)
            or (family == "count" and type(value) is int and 0 <= value <= 99)
            or (family == "switch" and type(value) is bool))


def _abstract_run(program, family):
    states, answers, query_values = {}, [], {}
    for index, event in enumerate(program):
        op, name = event["op"], event["name"]
        answer = ACK
        if op == "set":
            states[name] = event["value"]
        elif op == "copy":
            states[name] = states.get(event["source"])
        elif op == "advance":
            value = states.get(name)
            if value is not None:
                if family == "color":
                    value = COLORS[(COLORS.index(value) + 1) % len(COLORS)]
                elif family == "count":
                    value += event["amount"]
                    if not 0 <= value <= 99:
                        raise ValueError("count outside bounded world")
                else:
                    value = not value
                states[name] = value
        elif op == "query":
            value = states.get(name)
            query_values[index] = value
            answer = ASK if value is None else int(value == event.get("value"))
        else:
            raise ValueError("unknown operation")
        answers.append(answer)
    return answers, query_values


def abstract_oracle(program, family):
    """Typed simulator; never calls the English parser or interpreter."""
    if family not in FAMILIES or not isinstance(program, (list, tuple)) or not program:
        raise ValueError("unknown family or empty program")
    for event in program:
        if type(event) is not dict:
            raise ValueError("events must be dictionaries")
        op = event.get("op")
        fields = {"op", "name"} | ({"value"} if op in ("set", "query") else {"source"} if op == "copy" else {"amount"})
        if op not in ("set", "copy", "advance", "query") or set(event) != fields:
            raise ValueError("invalid event fields")
        if type(event["name"]) is not str or (op == "copy" and type(event["source"]) is not str):
            raise ValueError("event roles must be strings")
        if op in ("set", "query") and not _value_valid(family, event["value"]):
            raise ValueError("value outside typed domain")
        if op == "advance" and (type(event["amount"]) is not int or (event["amount"] not in (-3, -2, -1, 1, 2, 3) if family == "count" else event["amount"] != 1)):
            raise ValueError("invalid advance amount")
    return _abstract_run(program, family)[0]


def _sentence(event, family, names):
    op, name = event["op"], names[event["name"]]
    if op == "copy":
        return f"Copy the {family} of {names[event['source']]} to {name}."
    if op == "advance":
        if family == "color":
            return f"Move the color of {name} one step in cycle red, green, blue, yellow."
        if family == "switch":
            return f"Toggle the switch of {name}."
        amount = event["amount"]
        return f"{'Increase' if amount > 0 else 'Decrease'} the count of {name} by {abs(amount)}."
    value = ("on" if event["value"] else "off") if family == "switch" else str(event["value"])
    if op == "set":
        return f"Set the {family} of {name} to {value}."
    if op == "query":
        return f"Is the {family} of {name} {value}?"
    raise ValueError("unrenderable event")


# This parser and interpreter are independent of _abstract_run and its value
# representation: switch truth is stored as English strings, color uses a map.
_SET = re.compile(r"Set the (color|count|switch) of ([a-z]+) to ([a-z]+|[0-9]+)\.")
_COPY = re.compile(r"Copy the (color|count|switch) of ([a-z]+) to ([a-z]+)\.")
_QUERY = re.compile(r"Is the (color|count|switch) of ([a-z]+) ([a-z]+|[0-9]+)\?")
_COLOR = re.compile(r"Move the color of ([a-z]+) one step in cycle red, green, blue, yellow\.")
_COUNT = re.compile(r"(Increase|Decrease) the count of ([a-z]+) by ([1-3])\.")
_SWITCH = re.compile(r"Toggle the switch of ([a-z]+)\.")


def parse_sentence(text):
    """Exact finite English grammar; returns a family and generic event."""
    if type(text) is not str or not text or len(text.encode("utf-8")) > 128:
        raise ValueError("English input must be nonempty and at most 128 UTF-8 bytes")
    match = _SET.fullmatch(text) or _QUERY.fullmatch(text)
    if match:
        family, name, literal = match.groups()
        if family == "color":
            if literal not in ("red", "green", "blue", "yellow"):
                raise ValueError("unknown color")
            value = literal
        elif family == "switch":
            if literal not in ("on", "off"):
                raise ValueError("unknown switch state")
            value = literal == "on"
        else:
            if not literal.isdigit() or str(int(literal)) != literal or not 0 <= int(literal) <= 99:
                raise ValueError("invalid bounded count")
            value = int(literal)
        event = {"op": "set" if text.startswith("Set") else "query", "name": name, "value": value}
    elif match := _COPY.fullmatch(text):
        family, source, name = match.groups()
        event = {"op": "copy", "name": name, "source": source}
    elif match := _COLOR.fullmatch(text):
        family, event = "color", {"op": "advance", "name": match[1], "amount": 1}
    elif match := _COUNT.fullmatch(text):
        family, event = "count", {"op": "advance", "name": match[2], "amount": int(match[3]) * (1 if match[1] == "Increase" else -1)}
    elif match := _SWITCH.fullmatch(text):
        family, event = "switch", {"op": "advance", "name": match[1], "amount": 1}
    else:
        raise ValueError("unsupported composition English")
    if event["name"] not in ALIASES or (event["op"] == "copy" and event["source"] not in ALIASES):
        raise ValueError("name outside reviewed vocabulary")
    return family, event


def _parse_row(row):
    if not isinstance(row, dict) or row.get("family") not in FAMILIES:
        raise ValueError("row requires a declared typed domain")
    parsed = [parse_sentence(turn["text"]) for turn in row["turns"]]
    if any(family != row["family"] for family, _ in parsed):
        raise ValueError("mixed typed domains in an episode")
    return [event for _, event in parsed]


def english_oracle(row):
    """Read only English; metadata answers and programs never determine truth."""
    program = _parse_row(row)
    domain, memory, answers = row["family"], {}, []
    next_color = {"red": "green", "green": "blue", "blue": "yellow", "yellow": "red"}
    for event in program:
        op, name = event["op"], event["name"]
        if op == "set":
            memory[name] = ("on" if event["value"] else "off") if domain == "switch" else event["value"]
        elif op == "copy":
            if event["source"] in memory:
                memory[name] = memory[event["source"]]
            else:
                memory.pop(name, None)
        elif op == "advance":
            if name in memory:
                if domain == "color":
                    memory[name] = next_color[memory[name]]
                elif domain == "switch":
                    memory[name] = "off" if memory[name] == "on" else "on"
                else:
                    updated = memory[name] + event["amount"]
                    if updated < 0 or updated > 99:
                        raise ValueError("English count leaves bounded world")
                    memory[name] = updated
        else:
            literal = ("on" if event["value"] else "off") if domain == "switch" else event["value"]
            answers.append(ASK if name not in memory else ALLOW if memory[name] == literal else DENY)
            continue
        answers.append(ACK)
    return answers


def _sample_procedure(rng, turns):
    length = rng.randint(3, turns - 3)
    operations = [rng.choice(("copy", "advance")) for _ in range(length - 1)]
    if set(operations) != {"copy", "advance"}:
        return None
    anchor = {"op": "set", "name": "x0", "anchor": True}
    program, current = [anchor], "x0"
    for op in operations:
        if op == "copy":
            destination = rng.choice([role for role in ROLES[:6] if role != current])
            program.append({"op": op, "name": destination, "source": current})
            current = destination
        else:
            program.append({"op": op, "name": current})
    known_position = rng.randint(1, len(program))
    known_name = program[known_position - 1]["name"]
    program.insert(known_position, {"op": "query", "name": known_name})
    program.insert(rng.randrange(len(program) + 1), {"op": "query", "name": "x6"})
    while len(program) < turns - 1:
        op = rng.choice(("set", "copy", "advance", "query"))
        event = {"op": op, "name": rng.choice(ROLES[:6])}
        if op == "copy":
            event["source"] = rng.choice(ROLES[:7])
        program.insert(rng.randrange(len(program) + 1), event)
    program.append({"op": "query", "name": current})
    changed = next(index for index, event in enumerate(program) if event.get("anchor"))
    program[changed].pop("anchor")
    return program, changed


@lru_cache(maxsize=8192)
def _procedure(seed, turns, requested_partition):
    for attempt in range(8192):
        rng = random.Random(int(_hash([VERSION, "procedure", seed, turns, attempt]), 16))
        candidate = _sample_procedure(rng, turns)
        if candidate is None:
            continue
        program, changed = candidate
        records = _ancestry_records(program)
        final = records[-1]
        if (not final["known"] or not final["composed"] or final["root_index"] != changed
                or final["structure_partition"] != requested_partition
                or not any(row["known"] for row in records[:-1])
                or not any(not row["known"] for row in records[:-1])):
            continue
        if requested_partition == "train" and any(row["composed"] and row["structure_partition"] != "train" for row in records):
            continue
        return {"program": program, "changed_index": changed, "attempt": attempt,
                "structure_id": final["structure_id"], "structure_partition": requested_partition,
                "program_id": _hash([VERSION, "whole-procedure", _normalized(program)]),
                "query_ancestries": _public_ancestries(program)}
    raise ValueError("no valid shared procedure within the deterministic attempt bound")


def _random_value(rng, family):
    return rng.choice(COLORS) if family == "color" else rng.randint(30, 60) if family == "count" else rng.choice((False, True))


def _instantiate(procedure, family, value_seed):
    rng = random.Random(int(_hash([VERSION, "values", family, value_seed]), 16))
    programs = [deepcopy(procedure["program"])]
    for event in programs[0]:
        if event["op"] == "set":
            event["value"] = _random_value(rng, family)
        elif event["op"] == "advance":
            event["amount"] = rng.choice((-3, -2, -1, 1, 2, 3)) if family == "count" else 1
    programs.append(deepcopy(programs[0]))
    original = programs[0][procedure["changed_index"]]["value"]
    changed = rng.choice([color for color in COLORS if color != original]) if family == "color" else original + 1 if family == "count" else not original
    programs[1][procedure["changed_index"]]["value"] = changed
    first = _abstract_run(programs[0], family)[1]
    second = _abstract_run(programs[1], family)[1]
    for index, event in enumerate(programs[0]):
        if event["op"] != "query":
            continue
        if index == len(programs[0]) - 1:
            value = rng.choice((first[index], second[index]))
        elif first[index] is not None and rng.choice((False, True)):
            value = first[index]
        else:
            value = _random_value(rng, family)
        programs[0][index]["value"] = programs[1][index]["value"] = value
    answers = [abstract_oracle(program, family) for program in programs]
    if {answers[0][-1], answers[1][-1]} != {DENY, ALLOW}:
        raise ValueError("final counterfactual must change a known answer")
    return programs, answers


def _zero_observations():
    return {**{name: [[0.0] * width] for name, width in (("visual", 32), ("auditory", 4), ("body", 4), ("feedback", 2))}, "tokens": [0]}


@lru_cache(maxsize=8192)
def _generate_cached(family, seed, split, turns, naming_seed, value_seed, requested_partition):
    procedure = _procedure(seed, turns, requested_partition)
    programs, answers = _instantiate(procedure, family, value_seed)
    names = naming_map(naming_seed)
    recipe = {"seed": seed, "turns": turns, "naming_seed": naming_seed, "value_seed": value_seed,
              "structure_split": requested_partition, "procedure_attempt": procedure["attempt"]}
    group = _hash([VERSION, family, split, recipe])
    rows = []
    for variant, program in enumerate(programs):
        row = {"version": VERSION, "family": family, "split": split, "variant": variant,
            "counterfactual_group": group, "recipe": deepcopy(recipe),
            "structure_id": procedure["structure_id"], "structure_partition": requested_partition,
            "program_id": procedure["program_id"], "query_ancestries": deepcopy(procedure["query_ancestries"]),
            "turns": [{"text": _sentence(event, family, names), "observations": _zero_observations(),
                "target": target, "reply": REPLIES[target], "kind": "statement" if target == ACK else "query"}
                for event, target in zip(program, answers[variant])]}
        if english_oracle(row) != answers[variant] or query_ancestries(row) != row["query_ancestries"]:
            raise ValueError("abstract and independent English interpretations disagree")
        row["id"] = _hash(row)
        rows.append(row)
    return rows


def generate_pair(family, seed, *, split="train", turns=8, naming_seed=None, value_seed=None, structure_split=None):
    """Generate a complete exact counterfactual pair from compact independent seeds."""
    if type(family) is not str or family not in FAMILIES:
        raise ValueError("family must be color, count or switch")
    _seed(seed)
    _split(split)
    if type(turns) is not int or turns not in (8, 10, 12):
        raise ValueError("turn count must be 8, 10 or 12")
    naming_seed = seed if naming_seed is None else naming_seed
    value_seed = seed if value_seed is None else value_seed
    _seed(naming_seed)
    _seed(value_seed)
    requested_partition = split if structure_split is None else structure_split
    _split(requested_partition)
    if split == "train" and requested_partition != "train":
        raise ValueError("training admission requires training structural partition")
    return deepcopy(_generate_cached(family, seed, split, turns, naming_seed, value_seed, requested_partition))


def validate_row(row):
    if type(row) is not dict or set(row) != _FIELDS or type(row["variant"]) is not int or row["variant"] not in (0, 1):
        raise ValueError("invalid composition row fields or variant")
    recipe = row["recipe"]
    required = {"seed", "turns", "naming_seed", "value_seed", "structure_split", "procedure_attempt"}
    if type(recipe) is not dict or set(recipe) != required:
        raise ValueError("exact compact procedure provenance required")
    expected = generate_pair(row["family"], recipe["seed"], split=row["split"], turns=recipe["turns"],
        naming_seed=recipe["naming_seed"], value_seed=recipe["value_seed"], structure_split=recipe["structure_split"])[row["variant"]]
    if _json(row) != _json(expected):
        raise ValueError("row differs from canonical composition provenance")
    return True


validate_composition = validate_row


def validate_pair(rows):
    if not isinstance(rows, (list, tuple)) or len(rows) != 2:
        raise ValueError("exactly two complete pair members required")
    for row in rows:
        validate_row(row)
    if [row["variant"] for row in rows] != [0, 1] or any(rows[0][key] != rows[1][key] for key in _FIELDS - {"id", "variant", "turns"}):
        raise ValueError("matching canonical variants zero then one required")
    changed = [(a, b) for a, b in zip(rows[0]["turns"], rows[1]["turns"]) if a["text"] != b["text"]]
    if len(changed) != 1 or changed[0][0]["target"] != ACK or changed[0][1]["target"] != ACK:
        raise ValueError("exactly one initial statement must change")
    if {row["turns"][-1]["target"] for row in rows} != {DENY, ALLOW}:
        raise ValueError("final known question must flip")
    return True
