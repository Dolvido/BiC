"""Prospective recipe/byte/label evidence without model inference or training.

Training recipes are materialized once in bundle-ID order before either teaching
order runs. Exact transcript collisions and protected data are rejected. Compact
anchor references retain actual admitted procedures for fresh-realization tests.
The caller must freeze this evidence and its source/data provenance before use.
"""
from __future__ import annotations

import copy
import hashlib
import json

from experiments import foundation_curriculum as curriculum
from experiments import foundation_plan as planning
from experiments.composition_training import COUNTS, _row_counts
from experiments.realization_banks import transcript_digest
from experiments.train_cognitive import fingerprint_rows


SCHEMA = "bic-foundation-evidence-v1"


def json_digest(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("finite JSON evidence required") from error
    return hashlib.sha256(encoded).hexdigest()


def transcript_set(values):
    if type(values) not in (list, tuple, set, frozenset) or any(
            type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in values):
        raise ValueError("canonical lowercase transcript SHA256 values required")
    if len(values) != len(set(values)):
        raise ValueError("duplicate transcript digests are not canonical evidence")
    return set(values)


def _cell(row):
    depth = row["depth"]
    if depth == 0:
        mechanism = "direct"
    elif depth == 1:
        from experiments.composition_curriculum import _ancestry_records
        signature = _ancestry_records(curriculum._program(row))[-1]["signature"]
        transformations = [step[0] for step in signature if step[0] in ("copy", "advance")]
        if len(transformations) != 1:
            raise ValueError("primitive depth does not match final causal mechanism")
        mechanism = transformations[0]
    else:
        mechanism = "composed"
    return f"{row['family']}/d{depth}/{mechanism}/t{len(row['turns'])}"


def training_cell(row):
    curriculum.validate_row(row)
    return _cell(row)


def _empty_counts():
    return {**dict.fromkeys(COUNTS, 0), "final_opposite_pairs": 0,
            "target_counts": dict.fromkeys(map(str, range(4)), 0),
            "query_target_counts": dict.fromkeys(map(str, range(3)), 0)}


def row_counts(rows):
    """Count already admitted complete-pair rows; performs no inference."""
    result = _empty_counts()
    for row, counts in zip(rows, _row_counts(rows)):
        for name in COUNTS:
            result[name] += counts[name]
        for turn in row["turns"]:
            label = str(turn["target"])
            result["target_counts"][label] += 1
            if turn["target"] != 3:
                result["query_target_counts"][label] += 1
    result["final_opposite_pairs"] = len(rows)//2
    return result


def _add(destination, source):
    for name, value in source.items():
        if isinstance(value, dict):
            for key, count in value.items():
                destination[name][key] += count
        else:
            destination[name] += value


def reconstruct_anchor(plan, reference):
    planning.validate_plan(plan)
    if (type(reference) is not dict or set(reference) != {"bundle_id", "pair_index", "family", "pair_sha256"}
            or type(reference["family"]) is not str or reference["family"] not in curriculum.FAMILIES):
        raise ValueError("canonical anchor reference required")
    planning._integer(reference["pair_index"], "anchor pair index", 0, plan["config"]["micro_batch_size"]//2-1)
    rows = planning._materialize_validated_bundle(plan, reference["bundle_id"])[reference["family"]]
    start = 2*reference["pair_index"]
    pair = rows[start:start+2]
    if fingerprint_rows(pair) != reference["pair_sha256"]:
        raise ValueError("anchor differs from admitted training recipe")
    return pair


def prepare_training(plan, *, protected_transcripts=(), anchor_limit=64):
    planning.validate_plan(plan)
    planning._integer(anchor_limit, "anchor_limit", 1, 4096)
    plan = copy.deepcopy(plan)
    protected = transcript_set(protected_transcripts)
    seen, bundles, anchors, cells = set(), {}, {}, {}
    totals = {family: _empty_counts() for family in curriculum.FAMILIES}
    for bundle_id in range(len(plan["bundles"])):
        materialized = planning._materialize_validated_bundle(plan, bundle_id)
        record = {"family_sha256": {}, "family_counts": {}}
        for family, rows in materialized.items():
            record["family_sha256"][family] = fingerprint_rows(rows)
            record["family_counts"][family] = row_counts(rows)
            _add(totals[family], record["family_counts"][family])
            for offset in range(0, len(rows), 2):
                pair = rows[offset:offset+2]
                curriculum.validate_pair(pair)
                cell = _cell(pair[0])
                if cell != _cell(pair[1]):
                    raise ValueError("counterfactual pair changed the learning cell")
                for row in pair:
                    digest = transcript_digest(row)
                    if digest in protected or digest in seen:
                        raise ValueError("training transcript collides with protected or already admitted observations")
                    seen.add(digest)
                _add(cells.setdefault(cell, _empty_counts()), row_counts(pair))
                references = anchors.setdefault(cell, [])
                if len(references) < anchor_limit:
                    references.append({"bundle_id": bundle_id, "pair_index": offset//2,
                        "family": family, "pair_sha256": fingerprint_rows(pair)})
        bundles[str(bundle_id)] = record
    schedule_totals = {}
    for order, ids in plan["schedules"].items():
        phases = {phase: {family: _empty_counts() for family in curriculum.FAMILIES}
                  for phase in ("foundation", "mixed")}
        for bundle_id in ids:
            phase = plan["bundles"][str(bundle_id)]["phase"]
            for family, counts in bundles[str(bundle_id)]["family_counts"].items():
                _add(phases[phase][family], counts)
        schedule_totals[order] = phases
    if schedule_totals["curriculum"] != schedule_totals["mixed"]:
        raise ValueError("planned teaching orders differ in phase exposure budgets")
    transcripts = sorted(seen)
    manifest = {"schema": SCHEMA, "curriculum_version": curriculum.VERSION,
        "plan_sha256": json_digest(plan), "protected_count": len(protected),
        "protected_sha256": json_digest(sorted(protected)), "anchor_limit": anchor_limit,
        "bundles": bundles, "anchors": anchors, "cells": cells, "totals": totals,
        "schedule_totals": schedule_totals, "unique_transcripts": len(transcripts),
        "transcript_sha256": json_digest(transcripts)}
    return manifest, transcripts
