"""Causal query-placement adapter; never alters a canonical foundation lesson.

This separate schema preserves complete turns and statement order. Queries may
move only where their exact original immutable read version is unchanged in
both counterfactual members. Equal answers or normalized ancestry alone do not
authorize a move. The original opposite-answer query is an explicit anchor;
it need not be the last query or the last turn in this schema.

No Torch, neural work, training, historical-bank access, or automatic promotion.
Use an explicit pair_validator=validate_pair when admitting these rows elsewhere.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from experiments import foundation_curriculum as foundation


VERSION = "bic-foundation-layout-v1"
LAYOUTS = ("original", "varied")
_FIELDS = {"version", "id", "family", "split", "variant", "counterfactual_group",
           "recipe", "structure_id", "structure_partition", "primitive_shared", "depth",
           "program_id", "query_ancestries", "queries", "anchor", "layout_summary", "turns"}
_COUNTERS = ("materialize_pair_calls", "validate_pair_calls", "canonical_validation_calls",
             "canonical_parent_regeneration_calls", "layout_pair_materializations",
             "layout_episodes_materialized", "slot_signature_comparisons",
             "query_signature_checks", "permutation_checks", "typed_oracle_calls",
             "english_oracle_calls")


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("finite canonical JSON required") from error


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


@dataclass
class WorkLedger:
    counts: dict = field(default_factory=lambda: dict.fromkeys(_COUNTERS, 0))

    def report(self):
        return deepcopy(self.counts)


def _tick(work, name, count=1):
    if work is not None:
        if type(work) is not WorkLedger or set(work.counts) != set(_COUNTERS):
            raise ValueError("unmodified layout WorkLedger required")
        work.counts[name] += count


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("experiments/cognitive_curriculum.py", "experiments/composition_curriculum.py",
             "experiments/foundation_curriculum.py", "experiments/foundation_layout_curriculum.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _query_id(pair, index):
    return _hash([VERSION, "original-query", pair[0]["counterfactual_group"], index])


def _write(cells, nodes, event, identity):
    """Version IDs refer to original statement indices, never new positions.

    Unknown writes retain their effect ancestry too. An unknown advance is
    conservatively represented by an effect version; equal unknown truth does
    not erase preceding effects. Canonical foundation chains are known.
    """
    op, name = event["op"], event["name"]
    if op not in ("set", "copy", "advance"):
        raise ValueError("typed statement required")
    parent = None if op == "set" else cells.get(event["source"] if op == "copy" else name)
    nodes[identity] = dict(event=deepcopy(event), parent=parent)
    cells[name] = identity


def _signature(cells, nodes, query):
    version = cells.get(query["name"])
    current, ancestry = version, []
    while current is not None:
        node = nodes[current]
        ancestry.append(dict(original_statement_index=current, event=deepcopy(node["event"])))
        current = node["parent"]
    ancestry.reverse()
    return dict(read_version=version, versions=ancestry, query=deepcopy(query))


def _trace(program, identities):
    cells, nodes, queries = {}, {}, {}
    for event, identity in zip(program, identities):
        if event["op"] == "query":
            queries[identity] = _signature(cells, nodes, event)
        else:
            _write(cells, nodes, event, identity)
    return queries


def _legal_slots(pair, work):
    """Slots are counts of preceding statements, including 0 and all-statements."""
    programs = [foundation._program(row) for row in pair]
    statement_ids = [i for i, event in enumerate(programs[0]) if event["op"] != "query"]
    query_ids = [i for i, event in enumerate(programs[0]) if event["op"] == "query"]
    if any([event["op"] for event in program] != [event["op"] for event in programs[0]] for program in programs[1:]):
        raise ValueError("counterfactual operation order differs")
    slots = {identity: set(range(len(statement_ids) + 1)) for identity in query_ids}
    for program in programs:
        original = _trace(program, range(len(program)))
        cells, nodes = {}, {}
        for slot in range(len(statement_ids) + 1):
            for identity in query_ids:
                _tick(work, "slot_signature_comparisons")
                if _signature(cells, nodes, program[identity]) != original[identity]:
                    slots[identity].discard(slot)
            if slot < len(statement_ids):
                identity = statement_ids[slot]
                _write(cells, nodes, program[identity], identity)
    if any(not options for options in slots.values()):
        raise ValueError("an original query has no exact-version placement")
    return statement_ids, {identity: sorted(options) for identity, options in slots.items()}


def _permutation(pair, layout, seed, slots, statements):
    turns = len(pair[0]["turns"])
    final_slot = len(statements)
    signatures = [_trace(foundation._program(row), range(turns)) for row in pair]
    absent = [identity for identity, options in slots.items()
              if final_slot in options and all(member[identity]["read_version"] is None for member in signatures)]
    if not absent:
        raise ValueError("canonical foundation must retain a terminal-valid absent query")
    feasible = ["anchor_last", "unknown_last"]
    if all(any(slot < final_slot for slot in options) for options in slots.values()):
        feasible.append("statement_last")
    if layout == "original":
        return list(range(turns)), "original", feasible
    # This key excludes targets, typed values, surface names and family labels.
    key = [VERSION, "causal-layout", seed, pair[0]["program_id"]]
    anchor = turns - 1
    policy = feasible[seed % len(feasible)]
    terminal = (anchor if policy == "anchor_last" else
                min(absent, key=lambda identity: _hash([*key, "terminal", identity])) if policy == "unknown_last" else None)
    buckets = {slot: [] for slot in range(final_slot + 1)}
    for identity, options in slots.items():
        if policy == "statement_last":
            options = [slot for slot in options if slot < final_slot]
        slot = final_slot if identity == terminal else options[int(_hash([*key, "slot", identity]), 16) % len(options)]
        buckets[slot].append(identity)
    result = []
    for slot in range(final_slot + 1):
        queries = sorted(buckets[slot], key=lambda identity: _hash([*key, "order", identity]))
        if terminal in queries:
            queries.remove(terminal)
            queries.append(terminal)
        result.extend(queries)
        if slot < final_slot:
            result.append(statements[slot])
    return result, policy, feasible


def _check_relation(pair, rows, permutation, work=None):
    """Independent semantic checks beyond exact canonical regeneration."""
    size = len(pair[0]["turns"])
    _tick(work, "permutation_checks")
    if (type(permutation) is not list or len(permutation) != size
            or any(type(index) is not int for index in permutation)
            or set(permutation) != set(range(size))):
        raise ValueError("complete integer event permutation required")
    original_statements = [index for index, event in enumerate(foundation._program(pair[0])) if event["op"] != "query"]
    if [index for index in permutation if index in original_statements] != original_statements:
        raise ValueError("statement relative order changed")
    for parent, row in zip(pair, rows):
        expected_turns = [parent["turns"][index] for index in permutation]
        if _json(row["turns"]) != _json(expected_turns):
            raise ValueError("complete original turns or shared permutation changed")
        if Counter(map(_json, parent["turns"])) != Counter(map(_json, row["turns"])):
            raise ValueError("full event multiset changed")
        before_program, after_program = foundation._program(parent), foundation._program(row)
        before = _trace(before_program, range(size))
        after = _trace(after_program, permutation)
        for identity, signature in before.items():
            _tick(work, "query_signature_checks")
            if signature != after.get(identity):
                raise ValueError("query exact read-version ancestry changed")
        expected_targets = [turn["target"] for turn in expected_turns]
        _tick(work, "typed_oracle_calls")
        if foundation.legacy.abstract_oracle(after_program, row["family"]) != expected_targets:
            raise ValueError("typed effect oracle changed a target")
        _tick(work, "english_oracle_calls")
        if foundation.legacy.english_oracle(row) != expected_targets:
            raise ValueError("English oracle changed a target")
        prior_ancestries = {record["turn_index"]: record for record in foundation.query_ancestries(before_program)}
        current_ancestries = foundation.query_ancestries(after_program)
        for record in current_ancestries:
            original = prior_ancestries[permutation[record["turn_index"]]]
            if {k: v for k, v in record.items() if k != "turn_index"} != {k: v for k, v in original.items() if k != "turn_index"}:
                raise ValueError("query semantic partition or ancestry changed")
        admission = "train" if parent["recipe"]["structure_split"] == "train" else parent["split"]
        if not foundation._admissible(current_ancestries, admission):
            raise ValueError("query no longer has canonical semantic admission")
        if any(row[key] != parent[key] for key in ("family", "split", "variant", "depth", "structure_id", "structure_partition", "primitive_shared")):
            raise ValueError("canonical anchor semantic scope changed")
    anchor_index = permutation.index(size - 1)
    if {row["turns"][anchor_index]["target"] for row in rows} != {foundation.legacy.DENY, foundation.legacy.ALLOW}:
        raise ValueError("original known anchor no longer has opposite answers")


def _materialize(pair, layout, seed, work):
    statements, slots = _legal_slots(pair, work)
    permutation, policy, feasible = _permutation(pair, layout, seed, slots, statements)
    recipe = dict(layout=layout, seed=seed, base_version=foundation.VERSION,
        base_recipe=deepcopy(pair[0]["recipe"]), base_pair_sha256=_hash(pair),
        base_row_sha256=[_hash(row) for row in pair], base_ids=[row["id"] for row in pair],
        base_counterfactual_group=pair[0]["counterfactual_group"],
        source_sha256=source_hashes(), permutation=permutation,
        feasible_terminal_kinds=feasible, selected_terminal_kind=policy)
    group = _hash([VERSION, pair[0]["family"], pair[0]["split"], recipe])
    rows = []
    for parent in pair:
        program = foundation._program(parent)
        signatures = _trace(program, range(len(program)))
        turns = [deepcopy(parent["turns"][index]) for index in permutation]
        reordered = [program[index] for index in permutation]
        anchor_original = len(program) - 1
        row = {key: deepcopy(parent[key]) for key in ("family", "split", "variant", "structure_id",
            "structure_partition", "primitive_shared", "depth")}
        row.update(version=VERSION, counterfactual_group=group, recipe=deepcopy(recipe), turns=turns,
            program_id=_hash([VERSION, "whole-procedure", foundation.legacy._normalized(reordered)]),
            query_ancestries=foundation.query_ancestries(reordered),
            queries=[dict(query_id=_query_id(pair, identity), original_turn_index=identity,
                turn_index=permutation.index(identity), version_signature=deepcopy(signatures[identity]),
                valid_statement_slots=deepcopy(slots[identity])) for identity in sorted(signatures)],
            anchor=dict(query_id=_query_id(pair, anchor_original), original_turn_index=anchor_original,
                turn_index=permutation.index(anchor_original), structure_id=parent["structure_id"],
                depth=parent["depth"], version_signature_sha256=_hash(signatures[anchor_original])),
            layout_summary=dict(terminal_kind=policy, changed=permutation != list(range(len(program))),
                anchor_nonterminal=permutation[-1] != anchor_original, statements=len(statements),
                queries=len(slots), terminal_query_id=_query_id(pair, permutation[-1]) if permutation[-1] in slots else None))
        row["id"] = _hash(row)
        rows.append(row)
        _tick(work, "layout_episodes_materialized")
    _tick(work, "layout_pair_materializations")
    _check_relation(pair, rows, permutation, work)
    return rows


def _options(layout, seed):
    if type(layout) is not str or layout not in LAYOUTS:
        raise ValueError("layout must be original or varied")
    foundation._seed(seed, "layout seed")


def materialize_pair(base_pair, *, layout="original", seed=0, work=None):
    """Authenticate an existing canonical pair and copy it into the new schema.

    Original preserves turn order exactly. Varied selects seed modulo feasible
    terminal kinds: anchor-last, unknown-last, and statement-last only when all
    queries can precede the last statement. Remaining slot/order draws are
    deterministic and preserve exact versions. No impossible quota is imposed.
    Some feasible varied draws can equal the original; `layout_summary.changed`
    records this honestly. Callers must count actual terminal-kind exposure.
    """
    _options(layout, seed)
    _tick(work, "materialize_pair_calls")
    _tick(work, "canonical_validation_calls")
    foundation.validate_pair(base_pair)
    return _materialize(deepcopy(list(base_pair)), layout, seed, work)


def validate_pair(rows, *, work=None):
    """Exact compact-recipe regeneration plus independent causal/effect checks."""
    _tick(work, "validate_pair_calls")
    if (type(rows) not in (list, tuple) or len(rows) != 2
            or any(type(row) is not dict or set(row) != _FIELDS or row["version"] != VERSION for row in rows)
            or [row["variant"] for row in rows] != [0, 1]
            or any(type(row["variant"]) is not int for row in rows)):
        raise ValueError("two ordered layout-schema variants required")
    recipe = rows[0]["recipe"]
    fields = {"layout", "seed", "base_version", "base_recipe", "base_pair_sha256", "base_row_sha256",
              "base_ids", "base_counterfactual_group", "source_sha256", "permutation",
              "feasible_terminal_kinds", "selected_terminal_kind"}
    if type(recipe) is not dict or set(recipe) != fields or recipe["base_version"] != foundation.VERSION:
        raise ValueError("exact compact canonical-parent and layout recipe required")
    _options(recipe["layout"], recipe["seed"])
    base_recipe = recipe["base_recipe"]
    if type(base_recipe) is not dict or set(base_recipe) != foundation._RECIPE:
        raise ValueError("compact canonical foundation recipe required")
    _tick(work, "canonical_parent_regeneration_calls")
    pair = foundation.generate_pair(rows[0]["family"], base_recipe["seed"], depth=base_recipe["depth"],
        turns=base_recipe["turns"], split=rows[0]["split"], naming_seed=base_recipe["naming_seed"],
        value_seed=base_recipe["value_seed"], structure_split=base_recipe["structure_split"])
    expected = _materialize(pair, recipe["layout"], recipe["seed"], work)
    if _json(list(rows)) != _json(expected):
        raise ValueError("layout rows differ from canonical parent and deterministic placement provenance")
    return True
