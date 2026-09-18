"""Deterministic pre-training admission by sparse, names-only pair replacement.

Neither scores nor a learner participate. Bundle, family and pair traversal is
fixed; a whole pair keeps its original rendering unless either transcript is
protected or already admitted. Every retry preserves the complete typed program
and all supervision. The report is deterministic; callers own elapsed timing.
This establishes exact transcript exclusion, not new algorithms or semantics.
"""
from __future__ import annotations

import copy

from experiments import foundation_curriculum as curriculum
from experiments import foundation_plan as planning
from experiments.foundation_evidence import json_digest, transcript_set
from experiments.realization_banks import transcript_digest


SCHEMA = "bic-foundation-admission-v1"
MAX_ATTEMPTS = 1000


class AdmissionError(ValueError):
    """Failed admission with isolated, deterministic partial evidence."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = copy.deepcopy(receipt)


def _meaning(pair):
    """Remove only aliases and their provenance, preserving typed operations."""
    result = []
    for row in pair:
        inverse = {alias: role for role, alias in
                   curriculum.legacy.naming_map(row["recipe"]["naming_seed"]).items()}
        program = copy.deepcopy(curriculum._program(row))
        for event in program:
            for field in ("name", "source"):
                if field in event:
                    event[field] = inverse[event[field]]
        value = {key: copy.deepcopy(item) for key, item in row.items()
                 if key not in ("id", "counterfactual_group", "turns")}
        value["recipe"].pop("naming_seed")
        value["turns"] = [{**{key: copy.deepcopy(item) for key, item in turn.items()
                              if key != "text"}, "typed_event": event}
                          for turn, event in zip(row["turns"], program)]
        result.append(value)
    return json_digest(result)


def _identity(pair):
    return {"pair_sha256": json_digest(pair),
            "transcript_sha256": [transcript_digest(row) for row in pair],
            "naming_seed": pair[0]["recipe"]["naming_seed"]}


def _reasons(hashes, protected, seen):
    return {"protected": sorted(set(hashes) & protected),
            "already_admitted": sorted(set(hashes) & seen),
            "within_pair_duplicate": len(set(hashes)) != 2}


def _collision(reasons):
    return bool(reasons["protected"] or reasons["already_admitted"]
                or reasons["within_pair_duplicate"])


def repair_plan(base_plan, protected_transcripts=()):
    """Return a canonical admitted plan and exact deterministic receipt.

    Original pairs are visited once in ascending bundle ID, declared family
    order and pair offset. Failed attempts never enter ``seen``. A maximum of
    1000 alternative naming seeds is considered per collided pair. No input is
    mutated; exhaustion exposes the full partial receipt on AdmissionError.
    """
    planning.validate_plan(base_plan)
    if "admission" in base_plan:
        raise ValueError("repair requires an original plan without admission")
    plan = copy.deepcopy(base_plan)
    protected = transcript_set(protected_transcripts)
    protected_hash = json_digest(sorted(protected))
    counts = dict.fromkeys(("bundles", "original_pairs", "original_episode_candidates",
        "candidate_pairs", "candidate_episodes", "retry_candidates", "rejected_candidate_pairs",
        "original_collision_pairs", "rerendered_pairs", "unchanged_pairs", "accepted_pairs",
        "accepted_episodes", "unique_accepted_transcripts"), 0)
    receipt = {"schema": SCHEMA, "status": "preparing",
        "original_plan_sha256": json_digest(plan),
        "protected_sha256": protected_hash, "protected_count": len(protected),
        "traversal": "ascending bundle ID, declared family order, ascending pair index",
        "family_order": list(curriculum.FAMILIES), "max_naming_attempts": MAX_ATTEMPTS,
        "counts": counts, "rejections": [], "overrides": [],
        "accepted_stream_sha256": json_digest([SCHEMA, "accepted_pairs"]),
        "scope": "Exact observation-transcript exclusion only. Procedures, values, ancestry, supervision and teaching orders remain fixed. No model, scores, optimization, admission dropping or semantic-novelty claim."}
    seen, overrides = set(), {}
    for bundle_id in range(len(plan["bundles"])):
        rows = planning._materialize_validated_bundle(plan, bundle_id)
        if type(rows) is not dict or set(rows) != set(curriculum.FAMILIES):
            raise ValueError("materializer must return every declared family")
        for family in curriculum.FAMILIES:
            examples = rows[family]
            if type(examples) is not list or len(examples) != plan["config"]["micro_batch_size"]:
                raise ValueError("materializer pair count differs from plan")
            for offset in range(0, len(examples), 2):
                original = examples[offset:offset+2]
                curriculum.validate_pair(original)
                bundle = plan["bundles"][str(bundle_id)]
                if any(row["family"] != family or row["split"] != "train"
                       or row["structure_partition"] != "train"
                       or row["depth"] != bundle["depth"]
                       or len(row["turns"]) != bundle["turns"] for row in original):
                    raise ValueError("materialized pair differs from declared training cell")
                ref = {"bundle_id": bundle_id, "family": family, "pair_index": offset//2}
                key = f"{bundle_id}/{family}/{offset//2}"
                initial = _identity(original)
                counts["original_pairs"] += 1
                counts["original_episode_candidates"] += 2
                candidate, accepted, meaning = original, None, None
                for attempt in range(MAX_ATTEMPTS+1):
                    if attempt:
                        candidate = planning.materialize_pair(plan, bundle_id, family,
                                                              offset//2, attempt=attempt)
                        curriculum.validate_pair(candidate)
                        counts["retry_candidates"] += 1
                    counts["candidate_pairs"] += 1
                    counts["candidate_episodes"] += 2
                    identity = initial if attempt == 0 else _identity(candidate)
                    if attempt and _meaning(candidate) != meaning:
                        receipt.update(status="failed", failure="naming retry changed typed lesson",
                                       failed_reference={**ref, "attempt": attempt})
                        raise AdmissionError("naming retry changed program, values, ancestry or supervision", receipt)
                    reasons = _reasons(identity["transcript_sha256"], protected, seen)
                    if not _collision(reasons):
                        accepted = identity
                        break
                    counts["rejected_candidate_pairs"] += 1
                    if attempt == 0:
                        counts["original_collision_pairs"] += 1
                        meaning = _meaning(original)
                    receipt["rejections"].append({**ref, "attempt": attempt,
                                                   **identity, "reasons": reasons})
                if accepted is None:
                    receipt.update(status="failed", failure="naming retry budget exhausted",
                                   failed_reference=ref)
                    raise AdmissionError("foundation naming admission exhausted 1000 attempts", receipt)
                if attempt:
                    overrides[key] = attempt
                    counts["rerendered_pairs"] += 1
                    receipt["overrides"].append({**ref, "attempt": attempt,
                        "original": initial, "accepted": accepted,
                        "typed_lesson_sha256": meaning,
                        "typed_program_values_ancestry_targets_replies_unchanged": True})
                else:
                    counts["unchanged_pairs"] += 1
                seen.update(accepted["transcript_sha256"])
                counts["accepted_pairs"] += 1
                counts["accepted_episodes"] += 2
                counts["unique_accepted_transcripts"] = len(seen)
                receipt["accepted_stream_sha256"] = json_digest([
                    receipt["accepted_stream_sha256"], ref, accepted["pair_sha256"]])
        counts["bundles"] += 1
    admission = {"schema": planning.ADMISSION_SCHEMA, "protected_sha256": protected_hash,
                 "protected_count": len(protected), "realization_attempts": overrides}
    admitted = planning.build_plan(**plan["config"], admission=admission)
    if {key: value for key, value in admitted.items() if key != "admission"} != plan:
        raise ValueError("admission unexpectedly changed the original lesson plan")
    receipt.update(status="admitted", admitted_plan_sha256=json_digest(admitted),
                   accepted_transcripts_sha256=json_digest(sorted(seen)))
    return admitted, receipt


def authenticate_admission(base_plan, admitted_plan, protected_transcripts=(), receipt=None):
    """Recreate canonical admission once at verification, never in a step loop."""
    planning.validate_plan(admitted_plan)
    expected_plan, expected_receipt = repair_plan(base_plan, protected_transcripts)
    if json_digest(admitted_plan) != json_digest(expected_plan):
        raise ValueError("admitted plan differs from deterministic first-safe naming attempts")
    if receipt is not None and json_digest(receipt) != json_digest(expected_receipt):
        raise ValueError("admission receipt differs from canonical reconstruction")
    return expected_receipt
