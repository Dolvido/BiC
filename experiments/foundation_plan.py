"""Matched, compact lesson recipes for a candidate foundation-order comparison.

This prepares lessons; it does not train, choose a winning model or freeze a
study. Both orders consume the exact same bundle IDs and final mixed suffix.
Before any study, callers must authenticate generated banks, reserve evaluation
and historical transcripts, record actual byte/label mixtures, and bind sources.
"""
from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy


SCHEMA = "bic-foundation-plan-v1"
ADMISSION_SCHEMA = "bic-foundation-plan-admission-v1"
MAX_ADMISSION_ATTEMPTS = 1000
# Admission extensions must never change the original procedure/value streams.
_SEED_NAMESPACE = "bic-foundation-plan-v1"
DEPTHS = tuple(range(6))
TURN_BUCKETS = (8, 10, 12)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("finite JSON plan required") from error


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def build_plan(*, seed, stage_updates=24, final_updates=48, micro_batch_size=32,
               rehearsal_every=4, ordering_seed=0, admission=None):
    """Six depth stages with earlier-depth rehearsal and a shared mixed finish.

    Every bundle contains one microbatch from each of the three domains. Stage
    zero teaches foundations; later stages reserve every Nth bundle for earlier
    depths, in round-robin order. Enough rehearsal slots to revisit every earlier
    depth are required. Length rotates separately within each depth. The mixed
    arm shuffles only the staged prefix, with no change to lesson recipes.
    """
    _integer(seed, "seed", 0, 2**63-1)
    _integer(ordering_seed, "ordering_seed", 0, 2**63-1)
    _integer(stage_updates, "stage_updates", 1, 4096)
    _integer(final_updates, "final_updates", 6, 24576)
    _integer(micro_batch_size, "micro_batch_size", 2, 512)
    _integer(rehearsal_every, "rehearsal_every", 2, 4096)
    if micro_batch_size % 2 or final_updates % len(DEPTHS):
        raise ValueError("complete pair microbatches and equal final depth coverage required")
    if stage_updates // rehearsal_every < len(DEPTHS)-1:
        raise ValueError("each depth stage needs enough rehearsal slots for all earlier depths")
    config = {"seed": seed, "stage_updates": stage_updates, "final_updates": final_updates,
              "micro_batch_size": micro_batch_size, "rehearsal_every": rehearsal_every,
              "ordering_seed": ordering_seed}
    bundles, prefix, suffix = {}, [], []
    occurrences = dict.fromkeys(DEPTHS, 0)

    def append(depth, stage, phase, destination):
        index = len(bundles)
        turns = TURN_BUCKETS[occurrences[depth] % len(TURN_BUCKETS)]
        occurrences[depth] += 1
        bundles[str(index)] = {"id": index, "depth": depth, "turns": turns,
                               "stage": stage, "phase": phase}
        destination.append(index)

    for stage in DEPTHS:
        earlier = 0
        for slot in range(stage_updates):
            rehearse = stage > 0 and (slot+1) % rehearsal_every == 0
            depth = earlier % stage if rehearse else stage
            earlier += int(rehearse)
            append(depth, stage, "foundation", prefix)
    for index in range(final_updates):
        append(DEPTHS[index % len(DEPTHS)], None, "mixed", suffix)
    shuffled = list(prefix)
    random.Random(ordering_seed).shuffle(shuffled)
    result = {"schema": SCHEMA, "config": config, "bundles": bundles,
            "schedules": {"curriculum": prefix+suffix, "mixed": shuffled+suffix},
            "scope": "Same complete-pair bundle multiset and identical final mixed phase. Only prefix order differs; stage/depth/recipe metadata never enter model input. This tests order within added foundation practice, not benefit of adding foundations versus composition-only data."}
    if admission is not None:
        result["admission"] = _validate_admission(admission, result)
    return result


def _pair_location(plan, bundle_id, family, pair_index):
    from experiments.foundation_curriculum import FAMILIES
    _integer(bundle_id, "bundle_id", 0, len(plan["bundles"])-1)
    if type(family) is not str or family not in FAMILIES:
        raise ValueError("canonical foundation family required")
    _integer(pair_index, "pair_index", 0, plan["config"]["micro_batch_size"]//2-1)
    return f"{bundle_id}/{family}/{pair_index}"


def _validate_admission(admission, plan):
    """Validate the recipe envelope; its protected-set proof is owned by admission."""
    if (type(admission) is not dict or set(admission) != {
            "schema", "protected_sha256", "protected_count", "realization_attempts"}
            or admission["schema"] != ADMISSION_SCHEMA):
        raise ValueError("canonical foundation admission fields required")
    digest = admission["protected_sha256"]
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("canonical protected transcript digest required")
    _integer(admission["protected_count"], "protected_count", 0, 2**63-1)
    attempts = admission["realization_attempts"]
    if type(attempts) is not dict:
        raise ValueError("realization attempts must be a canonical mapping")
    for key, attempt in attempts.items():
        parts = key.split("/") if type(key) is str else []
        if len(parts) != 3 or not parts[0].isdecimal() or not parts[2].isdecimal():
            raise ValueError("canonical bundle/family/pair admission key required")
        expected = _pair_location(plan, int(parts[0]), parts[1], int(parts[2]))
        if key != expected:
            raise ValueError("noncanonical admission index spelling")
        _integer(attempt, "realization attempt", 1, MAX_ADMISSION_ATTEMPTS)
    return deepcopy(admission)


def validate_plan(plan):
    fields = {"schema", "config", "bundles", "schedules", "scope"}
    if (type(plan) is not dict or set(plan) not in (fields, fields | {"admission"})
            or type(plan["config"]) is not dict):
        raise ValueError("foundation plan fields differ")
    try:
        expected = build_plan(**plan["config"], admission=plan.get("admission"))
    except TypeError as error:
        raise ValueError("foundation plan configuration fields differ") from error
    if _json(plan) != _json(expected):
        raise ValueError("foundation plan differs from its canonical recipe")
    return True


def _seed(config, bundle_id, pair_index, purpose):
    value = [_SEED_NAMESPACE, config["seed"], bundle_id, pair_index, purpose]
    return int(hashlib.sha256(_json(value).encode()).hexdigest(), 16) % (2**63)


def materialize_bundle(plan, bundle_id):
    """Recreate one three-domain update without mutable RNG or occurrence state.

    Purpose-separated naming/value seeds are fixed by ID, so reordering cannot
    change the examples. ID uniqueness does not guarantee unique English text:
    a prospective bank builder must check exact collisions and reserved data.
    """
    validate_plan(plan)
    return _materialize_validated_bundle(plan, bundle_id)


def materialize_pair(plan, bundle_id, family, pair_index, attempt=0):
    """Probe one naming-only candidate, independently of an admitted override.

    Zero always recreates the original pair. Positive bounded attempts change
    only its naming seed; procedure and value seeds, labels and replies remain
    unchanged. Normal bundle materialization applies the frozen override map.
    """
    validate_plan(plan)
    return _materialize_validated_pair(plan, bundle_id, family, pair_index, attempt)


def _materialize_validated_pair(plan, bundle_id, family, pair_index, attempt=0):
    from experiments.foundation_curriculum import generate_pair, validate_pair
    _pair_location(plan, bundle_id, family, pair_index)
    _integer(attempt, "realization attempt", 0, MAX_ADMISSION_ATTEMPTS)
    bundle, config = plan["bundles"][str(bundle_id)], plan["config"]
    purpose = "names" if attempt == 0 else f"admission-names-{attempt}"
    pair = generate_pair(family, _seed(config, bundle_id, pair_index, "procedure"),
        depth=bundle["depth"], turns=bundle["turns"], split="train",
        naming_seed=_seed(config, bundle_id, pair_index, purpose),
        value_seed=_seed(config, bundle_id, pair_index, "values"))
    validate_pair(pair)
    return pair


def _materialize_validated_bundle(plan, bundle_id):
    """Internal fast path for an owning caller's already validated isolated plan.

    This skips whole-plan reconstruction, not canonical lesson validation.
    Owners must bind the admitted plan/source identity before training or replay.
    Public callers should use materialize_bundle.
    """
    from experiments.foundation_curriculum import FAMILIES

    _integer(bundle_id, "bundle_id", 0, len(plan["bundles"])-1)
    config = plan["config"]
    attempts = plan.get("admission", {}).get("realization_attempts", {})
    result = {}
    for family in FAMILIES:
        rows = []
        for pair_index in range(config["micro_batch_size"] // 2):
            attempt = attempts.get(f"{bundle_id}/{family}/{pair_index}", 0)
            pair = _materialize_validated_pair(plan, bundle_id, family, pair_index, attempt)
            rows.extend(pair)
        result[family] = rows
    return result
