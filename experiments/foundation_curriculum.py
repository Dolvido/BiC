"""Untrained candidate curriculum: shared primitives to longer dependency chains.

The existing English grammar and independent abstract/English oracles are reused
unchanged. ``depth`` counts copy/advance transformations on the final immutable
value ancestry, not utterances or distracting operations. All depths 0..5 fit
all 8/10/12-turn buckets. Distractions touch disjoint roles; this does NOT test
robustness to interfering overwrites. Unary chains also permit parity/cycle
shortcuts. Structural novelty is finite syntactic novelty, not new algorithms.

Composed identities retain the legacy composition-v1 hashes AND partitions.
Every supervised train composition must be legacy train; development may use
legacy train/dev but never legacy audit. Noncomposed primitives are explicitly
shared. Their top-level structure_partition='train' means train-admissible, NOT
held structural novelty; per-query metadata separately records 'shared' and the
legacy partition. At depth two copy->advance is legacy train and advance->copy
is legacy audit: there is no depth-two legacy dev motif and no fallback.

This generator does not read historical bank files. Excluding prior transcripts,
or train-partition motifs previously used in familiar-motif audits, belongs in
new bank preparation. No parsed state, recipe, depth or answer enters model
inputs: use the existing packer with explicit ``pair_validator=validate_pair``.
No training, model selection, tutor or automatic promotion occurs here.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import random

from experiments import composition_curriculum as legacy


VERSION = "bic-shared-foundation-v1"
FAMILIES = legacy.FAMILIES
DEPTHS = tuple(range(6))
TURN_BUCKETS = (8, 10, 12)
MAX_ATTEMPTS = 2048
REPLIES = legacy.REPLIES
_FIELDS = {"version", "id", "family", "split", "variant", "counterfactual_group", "recipe",
           "structure_id", "structure_partition", "primitive_shared", "depth", "program_id",
           "query_ancestries", "turns"}
_RECIPE = {"seed", "depth", "turns", "naming_seed", "value_seed", "structure_split", "procedure_attempt"}


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("finite canonical JSON required") from error


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _seed(value, name="seed"):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(f"{name} must be an integer in [0,2**63)")


def _program(row_or_program):
    if type(row_or_program) is dict:
        family = row_or_program.get("family")
        if family not in FAMILIES or type(row_or_program.get("turns")) is not list:
            raise ValueError("typed English row required")
        parsed = [legacy.parse_sentence(turn["text"]) for turn in row_or_program["turns"]]
        if any(domain != family for domain, _ in parsed):
            raise ValueError("mixed typed domains are not admitted")
        return [event for _, event in parsed]
    if type(row_or_program) not in (list, tuple) or not row_or_program:
        raise ValueError("nonempty program required")
    return row_or_program


def query_ancestries(row_or_program):
    """Legacy identity plus explicit shared-primitive admission for EVERY query."""
    records = []
    for original in legacy._ancestry_records(_program(row_or_program)):
        shared = original["known"] and not original["composed"]
        records.append({"turn_index": original["turn_index"], "known": original["known"],
            "composed": original["composed"], "structure_id": original["structure_id"],
            "legacy_structure_partition": original["structure_partition"],
            "structure_partition": "shared" if shared else original["structure_partition"],
            "primitive_shared": shared,
            "depth": sum(step[0] in ("copy", "advance") for step in original["signature"])
                     if original["known"] else None})
    return records


def final_depth(row_or_program):
    program = _program(row_or_program)
    records = query_ancestries(program)
    if not records or records[-1]["turn_index"] != len(program) - 1 or not records[-1]["known"]:
        raise ValueError("final event must query a known value")
    return records[-1]["depth"]


def structure_id(row_or_program):
    final_depth(row_or_program)
    return query_ancestries(row_or_program)[-1]["structure_id"]


def _admissible(records, split):
    allowed = {"train"} if split == "train" else {"train", "dev"} if split == "dev" else {"train", "dev", "audit"}
    return all(not row["composed"] or row["legacy_structure_partition"] in allowed for row in records)


def _sample(rng, depth, turns):
    operations = [rng.choice(("copy", "advance")) for _ in range(depth)]
    if depth >= 2 and set(operations) != {"copy", "advance"}:
        return None
    program = [{"op": "set", "name": "x0", "anchor": True}]
    current = "x0"
    for op in operations:
        if op == "copy":
            destination = rng.choice([f"x{i}" for i in range(6) if f"x{i}" != current])
            program.append({"op": op, "name": destination, "source": current})
            current = destination
        else:
            program.append({"op": op, "name": current})
    # An intermediate known question is possible except depth5/turns8. It reads
    # a genuine prefix, never a supplied intermediate answer or state trace.
    if turns >= depth + 4:
        at = rng.randint(1, len(program))
        program.insert(at, {"op": "query", "name": program[at - 1]["name"]})
    program.insert(rng.randrange(len(program) + 1), {"op": "query", "name": "x6"})
    while len(program) < turns - 1:
        # These roles never connect to the main chain or the always-unknown x6.
        event = {"op": rng.choice(("set", "query")), "name": f"x{rng.randrange(7, 12)}"}
        program.insert(rng.randrange(len(program) + 1), event)
    program.append({"op": "query", "name": current})
    changed = next(index for index, event in enumerate(program) if event.pop("anchor", False))
    return program, changed


@lru_cache(maxsize=4096)
def _procedure(seed, depth, turns, split, requested_partition):
    if depth == 2 and requested_partition == "dev":
        raise ValueError("no depth-two legacy dev motif exists; no partition fallback")
    for attempt in range(MAX_ATTEMPTS):
        rng = random.Random(int(_hash([VERSION, "procedure", seed, depth, turns, attempt]), 16))
        candidate = _sample(rng, depth, turns)
        if candidate is None:
            continue
        program, changed = candidate
        records = query_ancestries(program)
        final = records[-1]
        # A familiar train-structure probe regenerates the SAME program under
        # dev/audit admission; changing only admission cannot loosen its prefixes.
        admission = "train" if requested_partition == "train" else split
        if (final["depth"] != depth or final["structure_partition"] != requested_partition
                or not _admissible(records, admission)):
            continue
        return {"program": program, "changed_index": changed, "attempt": attempt,
                "structure_id": final["structure_id"],
                "structure_partition": "train" if final["primitive_shared"] else final["structure_partition"],
                "primitive_shared": final["primitive_shared"],
                "program_id": _hash([VERSION, "whole-procedure", legacy._normalized(program)]),
                "query_ancestries": records}
    raise ValueError(f"no admitted foundation procedure within {MAX_ATTEMPTS} deterministic attempts")


def _values(family):
    return legacy.COLORS if family == "color" else tuple(range(100)) if family == "count" else (False, True)


def _random_initial(rng, family):
    return rng.randint(30, 60) if family == "count" else rng.choice(_values(family))


def _instantiate(procedure, family, value_seed):
    rng = random.Random(int(_hash([VERSION, "values", family, value_seed]), 16))
    first = deepcopy(procedure["program"])
    for event in first:
        if event["op"] == "set":
            event["value"] = _random_initial(rng, family)
        elif event["op"] == "advance":
            event["amount"] = rng.choice((-3, -2, -1, 1, 2, 3)) if family == "count" else 1
    second = deepcopy(first)
    index = procedure["changed_index"]
    original = first[index]["value"]
    second[index]["value"] = original + 1 if family == "count" else rng.choice([v for v in _values(family) if v != original])
    first_values = legacy._abstract_run(first, family)[1]
    second_values = legacy._abstract_run(second, family)[1]
    for index, event in enumerate(first):
        if event["op"] != "query":
            continue
        value = first_values[index]
        if index == len(first) - 1:
            literal = rng.choice((value, second_values[index]))
        elif value is None:
            literal = _random_initial(rng, family)
        elif rng.choice((False, True)):
            literal = value
        else:
            literal = rng.choice([v for v in _values(family) if v != value])
        first[index]["value"] = second[index]["value"] = literal
    programs = (first, second)
    answers = [legacy.abstract_oracle(program, family) for program in programs]
    if {answer[-1] for answer in answers} != {legacy.DENY, legacy.ALLOW}:
        raise ValueError("counterfactual final query must have opposite known answers")
    return programs, answers


@lru_cache(maxsize=4096)
def _generate(family, seed, depth, turns, split, naming_seed, value_seed, requested_partition):
    procedure = _procedure(seed, depth, turns, split, requested_partition)
    programs, answers = _instantiate(procedure, family, value_seed)
    names = legacy.naming_map(naming_seed)
    recipe = {"seed": seed, "depth": depth, "turns": turns, "naming_seed": naming_seed,
              "value_seed": value_seed, "structure_split": requested_partition,
              "procedure_attempt": procedure["attempt"]}
    group = _hash([VERSION, family, split, recipe])
    rows = []
    for variant, program in enumerate(programs):
        row = {"version": VERSION, "family": family, "split": split, "variant": variant,
            "counterfactual_group": group, "recipe": deepcopy(recipe), "depth": depth,
            **{key: deepcopy(procedure[key]) for key in ("structure_id", "structure_partition", "primitive_shared", "program_id", "query_ancestries")},
            "turns": [{"text": legacy._sentence(event, family, names), "observations": legacy._zero_observations(),
                       "target": target, "reply": REPLIES[target], "kind": "statement" if target == legacy.ACK else "query"}
                      for event, target in zip(program, answers[variant])]}
        if (legacy.english_oracle(row) != answers[variant] or query_ancestries(row) != row["query_ancestries"]
                or final_depth(row) != depth):
            raise ValueError("independent English truth or ancestry differs")
        row["id"] = _hash(row)
        rows.append(row)
    return rows


def generate_pair(family, seed, *, depth, turns=8, split="train", naming_seed=None,
                  value_seed=None, structure_split=None):
    """Generate two canonical observations differing in one causal initial fact.

    Depth0/1 use only structure_split=None/'shared'. At depth>=2 the requested
    legacy partition defaults to split. Familiar development probes may request
    'train'; audit-partition compositions are never admitted as train or dev.
    Different lengths do not assert matching programs, only matching depth.
    """
    if type(family) is not str or family not in FAMILIES:
        raise ValueError("unknown typed family")
    _seed(seed)
    if type(depth) is not int or depth not in DEPTHS:
        raise ValueError("depth must be an integer from zero through five")
    if type(turns) is not int or turns not in TURN_BUCKETS:
        raise ValueError("turn bucket must be eight, ten or twelve")
    if type(split) is not str or split not in legacy.SPLITS:
        raise ValueError("admission split must be train, dev or audit")
    naming_seed = seed if naming_seed is None else naming_seed
    value_seed = seed if value_seed is None else value_seed
    _seed(naming_seed, "naming_seed"); _seed(value_seed, "value_seed")
    requested = ("shared" if depth < 2 else split) if structure_split is None else structure_split
    if (type(requested) is not str or (requested != "shared" if depth < 2 else requested not in legacy.SPLITS)):
        raise ValueError("primitive anchors require shared scope; compositions require a legacy partition")
    if depth >= 2 and (split == "train" and requested != "train" or split == "dev" and requested == "audit"):
        raise ValueError("requested composition violates admission partition")
    return deepcopy(_generate(family, seed, depth, turns, split, naming_seed, value_seed, requested))


def validate_row(row):
    """Authenticate all observations, metadata and targets by exact regeneration."""
    if (type(row) is not dict or set(row) != _FIELDS or type(row["variant"]) is not int
            or row["variant"] not in (0, 1) or type(row["primitive_shared"]) is not bool):
        raise ValueError("canonical foundation row fields/variant differ")
    recipe = row["recipe"]
    if type(recipe) is not dict or set(recipe) != _RECIPE:
        raise ValueError("compact foundation recipe fields differ")
    if type(recipe["procedure_attempt"]) is not int or not 0 <= recipe["procedure_attempt"] < MAX_ATTEMPTS:
        raise ValueError("procedure attempt must be a bounded integer")
    expected = generate_pair(row["family"], recipe["seed"], depth=recipe["depth"], turns=recipe["turns"],
        split=row["split"], naming_seed=recipe["naming_seed"], value_seed=recipe["value_seed"],
        structure_split=recipe["structure_split"])[row["variant"]]
    if _json(row) != _json(expected):
        raise ValueError("row differs from canonical foundation provenance")
    return True


def validate_pair(rows):
    if type(rows) not in (list, tuple) or len(rows) != 2:
        raise ValueError("exactly two canonical pair members required")
    for row in rows:
        validate_row(row)
    if [row["variant"] for row in rows] != [0, 1] or any(
            _json(rows[0][key]) != _json(rows[1][key]) for key in _FIELDS - {"id", "variant", "turns"}):
        raise ValueError("matching canonical variants zero then one required")
    changed = [(left, right) for left, right in zip(rows[0]["turns"], rows[1]["turns"]) if left["text"] != right["text"]]
    if (len(changed) != 1 or any(turn["target"] != legacy.ACK for turn in changed[0])
            or {row["turns"][-1]["target"] for row in rows} != {legacy.DENY, legacy.ALLOW}):
        raise ValueError("pair requires one changed statement and opposite known final answers")
    return True
