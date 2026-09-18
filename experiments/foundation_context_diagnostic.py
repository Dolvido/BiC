"""Inference-only controlled context bank; never a canonical training lesson.

Twelve turns contain ten statements, a known query at index 10, and an unknown
query at index 11. The original foundation intermediate query is deliberately
absent. An intact causal chain moves relative to the same disjoint set-only
distractions; it is neither shortened nor supplied as an inference-side trace.
Renaming uses a derangement of ten existing three-byte aliases. These finite
unary worlds permit cycle/parity shortcuts. Sensitivity here is not evidence of
a causal learning defect, general intellect, or an untouched capability test.

Only this module and the two existing, independent procedural/English-oracle
modules are imported. The caller owns source/data admission, historical exact
transcript exclusions, execution budgets, and observation-only model packing.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random

from experiments import composition_curriculum as oracle


VERSION = "bic-context-sensitivity-v1"
BANK_SCHEMA = "bic-context-sensitivity-bank-v1"
FAMILIES = oracle.FAMILIES
GROUPS = ("direct", "copy", "advance", "composed_d2", "composed_d3",
          "composed_d4", "composed_d5")
GAPS = (0, 2, 4)
NAMINGS = (0, 1)
KNOWN_QUERY_TURN = 10
UNKNOWN_QUERY_TURN = 11
ALIASES = tuple(name for name in oracle.ALIASES if len(name.encode("ascii")) == 3)
ROLES = tuple(f"r{i}" for i in range(10))
SCOPE = ("Inference-only two-query diagnostic; no intermediate known query. "
         "Gap changes position and distance jointly. Renaming changes spellings, "
         "not role relations or byte lengths. No training-mechanism or mastery claim.")
_ROW_FIELDS = {"version", "id", "family", "split", "variant", "recipe",
               "comparison_group", "counterfactual_group", "operator_group", "depth",
               "gap", "naming_condition", "known_query_turn", "unknown_query_turn",
               "causal_statement_indices", "turns"}
_RECIPE_FIELDS = {"seed", "group", "pair_index", "gap", "naming"}
_TURN_FIELDS = {"text", "observations", "target", "reply", "kind"}
_WORK_FIELDS = ("blueprints_constructed", "generated_rows", "row_validation_attempts",
                "validated_rows", "pair_validation_attempts", "validated_pairs",
                "comparison_validation_attempts", "validated_comparisons",
                "bank_validation_attempts", "validated_banks", "abstract_oracle_calls",
                "english_oracle_calls", "abstract_value_runs")


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("finite canonical JSON required") from error


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf8")).hexdigest()


def _same(actual, expected, message):
    if _json(actual) != _json(expected):
        raise ValueError(message)


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


@dataclass
class WorkLedger:
    """Counts calls/work in this module, not arbitrary process-wide operations."""
    counts: dict = field(default_factory=lambda: dict.fromkeys(_WORK_FIELDS, 0))

    def report(self):
        return deepcopy(self.counts)


def _tick(work, name, count=1):
    if work is not None:
        if type(work) is not WorkLedger or set(work.counts) != set(_WORK_FIELDS):
            raise ValueError("unmodified WorkLedger required")
        work.counts[name] += count


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ["experiments/foundation_context_diagnostic.py",
             "experiments/composition_curriculum.py", "experiments/cognitive_curriculum.py"]
    initializer = "experiments/__init__.py"
    if (root / initializer).exists():
        names.append(initializer)
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _rng(seed, *purpose):
    return random.Random(int(_hash([VERSION, seed, *purpose]), 16))


def _depth(group):
    if type(group) is not str or group not in GROUPS:
        raise ValueError("declared operator group required")
    return 0 if group == "direct" else 1 if group in ("copy", "advance") else int(group[-1])


def _names(seed, group, pair_index, naming):
    names = list(ALIASES)
    if len(names) != 10 or len(set(names)) != 10:
        raise ValueError("ten distinct existing three-byte aliases required")
    _rng(seed, "names", group, pair_index).shuffle(names)
    if naming == 1:
        names = names[1:] + names[:1]
    return dict(zip(ROLES, names))


def _value(rng, family):
    return (rng.choice(oracle.COLORS) if family == "color" else
            rng.randint(30, 60) if family == "count" else rng.choice((False, True)))


def _abstract(program, family, work):
    _tick(work, "abstract_oracle_calls")
    return oracle.abstract_oracle(program, family)


def _blueprint(seed, family, group, pair_index, work=None):
    """Reconstruct typed recipes, not rendered episode rows or historical banks."""
    _integer(seed, "seed", 0, 2**63 - 1)
    _integer(pair_index, "pair_index", 0, 63)
    depth = _depth(group)
    if type(family) is not str or family not in FAMILIES:
        raise ValueError("declared family required")
    _tick(work, "blueprints_constructed")
    rng = _rng(seed, "procedure", group, pair_index)
    if depth == 0:
        operations = []
    elif depth == 1:
        operations = [group]
    else:
        operations = ["copy", "advance"] + [rng.choice(("copy", "advance"))
                                             for _ in range(depth - 2)]
        rng.shuffle(operations)
    values = _rng(seed, "values", family, group, pair_index)
    initial = _value(values, family)
    alternative = (values.choice([v for v in oracle.COLORS if v != initial])
                   if family == "color" else initial + 1 if family == "count" else not initial)
    chain = [{"op": "set", "name": "r0", "value": initial}]
    current, next_role = "r0", 1
    for operation in operations:
        if operation == "copy":
            destination = f"r{next_role}"
            chain.append({"op": "copy", "name": destination, "source": current})
            current, next_role = destination, next_role + 1
        else:
            chain.append({"op": "advance", "name": current,
                          "amount": values.choice((-3, -2, -1, 1, 2, 3)) if family == "count" else 1})
    if next_role > 5:
        raise ValueError("causal chain exceeds disjoint role allocation")
    other = deepcopy(chain)
    other[0]["value"] = alternative
    # Typed value evolution is used only in construction. Validation independently
    # executes the complete typed program and the rendered English interpreter.
    finals = []
    for candidate in (chain, other):
        query = {"op": "query", "name": current, "value": initial}
        _abstract([*candidate, query], family, work)
        _tick(work, "abstract_value_runs")
        finals.append(oracle._abstract_run([*candidate, query], family)[1][len(candidate)])
    if type(finals[0]) is not type(finals[1]) or finals[0] == finals[1]:
        raise ValueError("counterfactual causal values must remain distinct")
    literal = values.choice(finals)
    distractors = [{"op": "set", "name": f"r{5 + (i % 4)}", "value": _value(values, family)}
                   for i in range(9 - depth)]
    queries = [{"op": "query", "name": current, "value": literal},
               {"op": "query", "name": "r9", "value": _value(values, family)}]
    return (chain, other), distractors, queries


def _layout(chains, distractions, queries, variant, gap):
    prefix = len(distractions) - gap
    return deepcopy(distractions[:prefix] + chains[variant] + distractions[prefix:] + queries)


def _identity(seed, family, group, pair_index, gap=None, naming=None):
    base = [VERSION, "comparison", seed, family, group, pair_index]
    return _hash(base if gap is None else [*base, "pair", gap, naming])


def _make_rows(seed, family, group, pair_index, work):
    chains, distractions, queries = _blueprint(seed, family, group, pair_index, work)
    depth = _depth(group)
    rows = []
    for gap in GAPS:
        for naming in NAMINGS:
            names = _names(seed, group, pair_index, naming)
            for variant in (0, 1):
                program = _layout(chains, distractions, queries, variant, gap)
                targets = _abstract(program, family, work)
                prefix = len(distractions) - gap
                row = dict(version=VERSION, family=family, split="diagnostic", variant=variant,
                    recipe=dict(seed=seed, group=group, pair_index=pair_index, gap=gap, naming=naming),
                    comparison_group=_identity(seed, family, group, pair_index),
                    counterfactual_group=_identity(seed, family, group, pair_index, gap, naming),
                    operator_group=group, depth=depth, gap=gap, naming_condition=naming,
                    known_query_turn=KNOWN_QUERY_TURN, unknown_query_turn=UNKNOWN_QUERY_TURN,
                    causal_statement_indices=list(range(prefix, prefix + depth + 1)),
                    turns=[dict(text=oracle._sentence(event, family, names),
                        observations=oracle._zero_observations(), target=target,
                        reply=oracle.REPLIES[target], kind="statement" if target == oracle.ACK else "query")
                        for event, target in zip(program, targets)])
                row["id"] = _hash(row)
                rows.append(row)
                _tick(work, "generated_rows")
    return rows


def validate_row(row, *, work=None):
    _tick(work, "row_validation_attempts")
    if type(row) is not dict or set(row) != _ROW_FIELDS:
        raise ValueError("exact diagnostic row fields required")
    recipe = row["recipe"]
    if type(recipe) is not dict or set(recipe) != _RECIPE_FIELDS:
        raise ValueError("exact diagnostic recipe fields required")
    seed, group, pair_index, gap, naming = (recipe[k] for k in ("seed", "group", "pair_index", "gap", "naming"))
    _integer(gap, "gap", 0, 4)
    _integer(naming, "naming", 0, 1)
    _integer(row["variant"], "variant", 0, 1)
    if gap not in GAPS:
        raise ValueError("declared gap required")
    depth, family = _depth(group), row["family"]
    chains, distractions, queries = _blueprint(seed, family, group, pair_index, work)
    program = _layout(chains, distractions, queries, row["variant"], gap)
    names = _names(seed, group, pair_index, naming)
    metadata = dict(version=VERSION, split="diagnostic", operator_group=group, depth=depth,
        gap=gap, naming_condition=naming, known_query_turn=10, unknown_query_turn=11,
        comparison_group=_identity(seed, family, group, pair_index),
        counterfactual_group=_identity(seed, family, group, pair_index, gap, naming),
        causal_statement_indices=list(range(len(distractions)-gap, len(distractions)-gap+depth+1)))
    _same({k: row[k] for k in metadata}, metadata, "diagnostic metadata differs")
    turns = row["turns"]
    if type(turns) is not list or len(turns) != 12 or any(type(t) is not dict or set(t) != _TURN_FIELDS for t in turns):
        raise ValueError("exactly twelve canonical diagnostic turns required")
    typed_targets = _abstract(program, family, work)
    parsed = []
    for turn, event, target in zip(turns, program, typed_targets):
        _same(turn, dict(text=oracle._sentence(event, family, names),
            observations=oracle._zero_observations(), target=target, reply=oracle.REPLIES[target],
            kind="statement" if target == oracle.ACK else "query"), "turn differs from typed diagnostic recipe")
        parsed_family, parsed_event = oracle.parse_sentence(turn["text"])
        if parsed_family != family:
            raise ValueError("mixed-family English is forbidden")
        parsed.append(parsed_event)
    _tick(work, "english_oracle_calls")
    _same(oracle.english_oracle(row), typed_targets, "independent English and abstract truth differ")
    _same(_abstract(parsed, family, work), typed_targets, "parsed typed truth differs")
    if typed_targets[:10] != [oracle.ACK] * 10 or typed_targets[10] not in (oracle.DENY, oracle.ALLOW) or typed_targets[11] != oracle.ASK:
        raise ValueError("fixed known/unknown query anchors differ")
    if sum(len(t["text"].encode("utf8")) + 2 for t in turns) > 1024:
        raise ValueError("diagnostic input exceeds the existing 1024-token context")
    _same(row["id"], _hash({k: v for k, v in row.items() if k != "id"}), "row identity differs")
    _tick(work, "validated_rows")
    return True


def validate_pair(rows, *, work=None):
    _tick(work, "pair_validation_attempts")
    if type(rows) not in (list, tuple) or len(rows) != 2:
        raise ValueError("two adjacent counterfactual rows required")
    for row in rows:
        validate_row(row, work=work)
    if [r["variant"] for r in rows] != [0, 1] or rows[0]["counterfactual_group"] != rows[1]["counterfactual_group"]:
        raise ValueError("ordered variants from the same diagnostic pair required")
    changed = [i for i, (a, b) in enumerate(zip(rows[0]["turns"], rows[1]["turns"])) if a["text"] != b["text"]]
    if changed != [rows[0]["causal_statement_indices"][0]]:
        raise ValueError("only the initial causal fact may change")
    if {r["turns"][10]["target"] for r in rows} != {oracle.DENY, oracle.ALLOW}:
        raise ValueError("known query at index 10 must have opposite answers")
    _tick(work, "validated_pairs")
    return True


def validate_comparison(rows, *, work=None):
    _tick(work, "comparison_validation_attempts")
    if type(rows) not in (list, tuple) or len(rows) != 12:
        raise ValueError("complete gap/naming/variant comparison required")
    for start in range(0, 12, 2):
        validate_pair(rows[start:start+2], work=work)
    expected = [(g, n, v) for g in GAPS for n in NAMINGS for v in (0, 1)]
    if [(r["gap"], r["naming_condition"], r["variant"]) for r in rows] != expected or len({r["comparison_group"] for r in rows}) != 1:
        raise ValueError("comparison condition coverage/order differs")
    lookup = {(r["gap"], r["naming_condition"], r["variant"]): r for r in rows}
    for naming in NAMINGS:
        for variant in (0, 1):
            reference = lookup[(0, naming, variant)]
            for gap in GAPS:
                row = lookup[(gap, naming, variant)]
                if Counter(t["text"] for t in row["turns"][:10]) != Counter(t["text"] for t in reference["turns"][:10]):
                    raise ValueError("gap manipulation changed the statement multiset")
                _same(row["turns"][10:], reference["turns"][10:], "gap manipulation changed fixed queries")
                _same([t["target"] for t in row["turns"]], [t["target"] for t in reference["turns"]], "gap manipulation changed answers")
    for gap in GAPS:
        for variant in (0, 1):
            a, b = (lookup[(gap, naming, variant)] for naming in NAMINGS)
            _same([len(t["text"].encode("utf8")) for t in a["turns"]],
                  [len(t["text"].encode("utf8")) for t in b["turns"]], "renaming changed byte lengths")
    _tick(work, "validated_comparisons")
    return True


def validate_bank(bank, *, work=None):
    _tick(work, "bank_validation_attempts")
    if type(bank) is not dict or set(bank) != {"schema", "config", "scope", "rows", "rows_sha256"}:
        raise ValueError("exact diagnostic bank fields required")
    config = bank["config"]
    if type(config) is not dict or set(config) != {"seed", "pairs_per_group"}:
        raise ValueError("exact bank config required")
    _integer(config["seed"], "seed", 0, 2**63-1)
    _integer(config["pairs_per_group"], "pairs_per_group", 1, 64)
    rows = bank["rows"]
    if bank["schema"] != BANK_SCHEMA or bank["scope"] != SCOPE or type(rows) is not list or len(rows) != 252 * config["pairs_per_group"]:
        raise ValueError("bank schema/scope/coverage differs")
    expected = [(f, g, i) for f in FAMILIES for g in GROUPS for i in range(config["pairs_per_group"])]
    for offset, (family, group, pair_index) in enumerate(expected):
        comparison = rows[12*offset:12*(offset+1)]
        validate_comparison(comparison, work=work)
        recipe = comparison[0]["recipe"]
        if (comparison[0]["family"], recipe["group"], recipe["pair_index"], recipe["seed"]) != (family, group, pair_index, config["seed"]):
            raise ValueError("canonical bank comparison order/seed differs")
    transcripts = [_hash([t["text"] for t in row["turns"]]) for row in rows]
    if len(set(transcripts)) != len(rows) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate diagnostic rows/transcripts")
    _same(bank["rows_sha256"], _hash(rows), "bank row digest differs")
    _tick(work, "validated_banks")
    return True


def build_bank(*, seed, pairs_per_group=8, work=None):
    """Build and validate exactly 252*pairs_per_group new diagnostic episodes."""
    _integer(seed, "seed", 0, 2**63-1)
    _integer(pairs_per_group, "pairs_per_group", 1, 64)
    rows = []
    for family in FAMILIES:
        for group in GROUPS:
            for pair_index in range(pairs_per_group):
                rows.extend(_make_rows(seed, family, group, pair_index, work))
    bank = dict(schema=BANK_SCHEMA, config=dict(seed=seed, pairs_per_group=pairs_per_group),
                scope=SCOPE, rows=rows, rows_sha256=_hash(rows))
    validate_bank(bank, work=work)
    return bank
