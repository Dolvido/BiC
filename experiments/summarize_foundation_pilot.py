"""Project completed, sealed foundation evidence without replay or inference.

Only the standard library is imported. Canonical admission, optimizer restoration
and prediction validation belong to the already completed frozen pipeline. This
helper rechecks its file bindings and arithmetic; it does not repeat those jobs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


SCHEMA = "bic-foundation-pilot-summary-v1"
RESULTS_SCHEMA = "bic-foundation-results-v1"
VERSION = "bic-shared-foundation-v1"
ARMS = ("curriculum", "mixed")
METRICS = ("query_accuracy", "known_accuracy", "final_accuracy", "final_pair_accuracy",
           "opposite_pair_accuracy", "query_reply_accuracy", "final_reply_pair_accuracy")
EXTRA = ("ask_precision", "ask_recall", "action_reply_agreement", "query_loss", "brier_score", "unsupported_ask_rate")
PHYSICAL = ("physical_optimizer_updates", "drawn_episode_exposures",
            "neural_attempted_episode_exposures", "completed_microbatch_episode_exposures")
SCORE_COUNTS = ("scored_episodes", "scored_turns", "encoder_episode_instances", "free_reply_turns")
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    def reject(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=reject)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def same(a, b, message):
    if digest(a) != digest(b):
        raise ValueError(message)


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be finite and nonnegative")
    return value


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def ratio(value, correct, total, name):
    integer(correct, name); integer(total, name)
    if correct > total or (total == 0 and value is not None) or (total and (
            type(value) not in (int, float) or not math.isfinite(value)
            or not math.isclose(value, correct/total, rel_tol=0., abs_tol=1e-7))):
        raise ValueError(name + " count/rate disagreement")


def relative(root, name):
    if type(name) is not str or Path(name).is_absolute():
        raise ValueError("relative bound path required")
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("bound path escapes its evidence directory")
    return path


def load_complete(directory):
    """Read scores only after every completed gate exists; authenticate files."""
    directory = Path(directory).resolve()
    required = ("protocol.json", "preparation.json", "verification.json", "evaluation/report.json",
                *(f"{arm}/receipt.json" for arm in ARMS), *(f"evaluation/{arm}.json" for arm in ARMS))
    if any(not (directory/name).is_file() for name in required):
        raise ValueError("both completed arms, verification and evaluation are required")
    protocol = read(directory/"protocol.json")
    contract = protocol["contract"]
    if (contract["schema"] != "bic-foundation-order-pilot-v2" or contract["version"] != VERSION
            or contract["arms"] != list(ARMS) or contract["automatic_promotion"] is not False
            or contract["evaluation_during_training"] is not False):
        raise ValueError("foundation pilot contract differs")
    steps = contract["checkpoints"]
    if (not steps or any(type(step) is not int for step in steps) or steps[0] != 0
            or steps != sorted(set(steps)) or steps[-1] != contract["updates"]):
        raise ValueError("official checkpoint boundaries differ")
    verified = read(directory/"verification.json")
    if (verified.get("schema") != RESULTS_SCHEMA or verified.get("status") != "completed"
            or verified.get("automatic_promotion") is not False
            or verified.get("neural_training_or_inference") is not False
            or set(verified.get("arms", {})) != set(ARMS)):
        raise ValueError("completed two-arm CPU verification required")
    for arm in ARMS:
        receipt = read(directory/arm/"receipt.json")
        if (receipt.get("status") != "completed" or receipt.get("retained_updates") != steps[-1]
                or receipt.get("physical_work_unknown") is not False or receipt.get("automatic_promotion") is not False
                or receipt.get("schema") != contract["schema"] or receipt.get("arm") != arm
                or verified["arms"][arm].get("exact_official_checkpoint_restores") is not True):
            raise ValueError("completed known-work arm and exact checkpoint verification required")
        same(receipt, verified["arms"][arm]["receipt"], "verified training receipt changed")
    report = read(directory/"evaluation/report.json")
    if (report.get("schema") != RESULTS_SCHEMA or report.get("status") != "completed"
            or report.get("automatic_promotion") is not False or report.get("base_checkpoints_unchanged") is not True
            or set(report.get("results", {})) != set(ARMS) or set(report.get("arm_files_sha256", {})) != set(ARMS)):
        raise ValueError("completed unchanged two-arm evaluation required")

    checked = {}
    def bind(path, expected=None):
        path = Path(path).resolve()
        if str(path) not in checked:
            checked[str(path)] = file_hash(path)
        actual = checked[str(path)]
        if expected is not None and actual != expected:
            raise ValueError("bound file changed: " + str(path))
        return actual

    protocol_hash = bind(directory/"protocol.json")
    if set(protocol["files_sha256"]) != {"plan.json", "lesson-admission.json", "training-manifest.json",
            "bank-admission.json", "capacity-choice.json", "banks.pt", "protected.pt", "training-transcripts.pt"}:
        raise ValueError("complete frozen foundation input set required")
    same(verified["source_sha256"], protocol["source_sha256"], "verified source closure differs")
    for name, sha in protocol["source_sha256"].items():
        bind(relative(ROOT, name), sha)
        bind(relative(directory/"source", name), sha)
    expected_inputs = {"protocol.json", "preparation.json", *protocol["files_sha256"]}
    expected_checkpoints = {f"checkpoint-{step:06d}.pt" for step in steps}
    for arm in ARMS:
        expected_inputs.update({f"{arm}/receipt.json", f"{arm}/steps.jsonl",
                                *(f"{arm}/{name}" for name in expected_checkpoints)})
    if set(verified["input_file_sha256"]) != expected_inputs:
        raise ValueError("verified input manifest omits or adds files")
    for name, sha in verified["input_file_sha256"].items():
        bind(relative(directory, name), sha)
    for name, sha in protocol["files_sha256"].items():
        if verified["input_file_sha256"][name] != sha:
            raise ValueError("protocol input differs from verified manifest")
    if verified["protocol_sha256"] != protocol_hash or verified["proof_sha256"] != protocol["proof_sha256"]:
        raise ValueError("verified protocol/proof binding differs")
    preparation = read(directory/"preparation.json")
    if (preparation.get("status") != "prepared" or preparation["protocol_sha256"] != protocol_hash
            or preparation.get("neural_training_or_inference") is not False):
        raise ValueError("completed non-neural preparation required")
    for path, sha in protocol["capacity_choice"]["files_sha256"].items():
        bind(path, sha)
    proof_directory = Path(protocol["proof_directory"])
    proof = read(proof_directory/"probe.json")
    if (digest(proof) != protocol["proof_sha256"] or proof.get("status") != "passed"
            or proof.get("admission", {}).get("mode") != "naming_overrides"
            or type(proof["admission"].get("executed_prefix_overridden_pairs")) is not int
            or proof["admission"]["executed_prefix_overridden_pairs"] < 1
            or not proof.get("flags") or not all(value is True for value in proof["flags"].values())):
        raise ValueError("bound admitted execution proof differs")
    same(proof["config"], protocol["config"], "proof configuration differs")
    same(proof["execution_profile"], protocol["execution_profile"], "proof runtime differs")
    if proof["learning_rate"] != protocol["learning_rate"]:
        raise ValueError("proof rate differs")
    bind(proof_directory/"probe.json")
    for name, sha in proof["artifact_sha256"].items():
        bind(relative(proof_directory, name), sha)
    verification_hash = bind(directory/"verification.json")
    for evidence in (report, *report["results"].values()):
        if (evidence.get("schema") != RESULTS_SCHEMA or evidence.get("status") != "completed"
                or evidence.get("automatic_promotion") is not False
                or evidence["verification_sha256"] != verification_hash or evidence["protocol_sha256"] != protocol_hash):
            raise ValueError("completed evaluation binding differs")
        same(evidence["input_file_sha256"], verified["input_file_sha256"], "scored inputs differ")
        same(evidence["execution_profile"], protocol["execution_profile"], "scoring runtime differs")
    for arm in ARMS:
        row = verified["arms"][arm]
        receipt = row["receipt"]
        if set(row["weights_sha256"]) != {str(step) for step in steps} or set(receipt["checkpoints"]) != expected_checkpoints:
            raise ValueError("official checkpoint set differs")
        if row["weights_sha256"]["0"] != protocol["initial_weights_sha256"]:
            raise ValueError("initial verified weights differ")
        if receipt["protocol_sha256"] != protocol_hash or receipt["journal_sha256"] != verified["input_file_sha256"][f"{arm}/steps.jsonl"]:
            raise ValueError("training receipt input binding differs")
        same(receipt["execution_profile"], protocol["execution_profile"], "training runtime differs")
        for name, sha in receipt["checkpoints"].items():
            if sha != verified["input_file_sha256"][f"{arm}/{name}"]:
                raise ValueError("checkpoint binding differs")
        scored = report["results"][arm]
        if scored["arm"] != arm or [point["updates"] for point in scored["development_curve"]] != steps:
            raise ValueError("complete official development curve required")
        for point in scored["development_curve"]:
            if point["weights_sha256"] != row["weights_sha256"][str(point["updates"])]:
                raise ValueError("scored development checkpoint differs")
        bind(directory/f"evaluation/{arm}.json", report["arm_files_sha256"][arm])
        same(read(directory/f"evaluation/{arm}.json"), scored, "embedded evaluation arm differs")
    bind(directory/"evaluation/report.json")
    return protocol, verified, report, preparation, checked


def cell(name):
    found = re.fullmatch(r"(fit|fresh|composed)/(color|count|switch)/d([0-5])/(direct|copy|advance|composed)/t(8|10|12)", name)
    if not found:
        raise ValueError("foundation cell name differs")
    panel, family, depth, operator, turns = found.groups()
    allowed = ("direct",) if depth == "0" else ("copy", "advance") if depth == "1" else ("composed",)
    if operator not in allowed:
        raise ValueError("operator does not match causal depth")
    return panel, family, "d"+depth, operator, turns


def expected_cells(role):
    result = set()
    for family in ("color", "count", "switch"):
        for turns in (8, 10, 12):
            for depth in range(6):
                for operator in (("direct",) if depth == 0 else ("copy", "advance") if depth == 1 else ("composed",)):
                    result.add(f"{'fit' if role == 'train_fit' else 'fresh'}/{family}/d{depth}/{operator}/t{turns}")
                if role != "train_fit" and depth >= (3 if role == "dev" else 2):
                    result.add(f"composed/{family}/d{depth}/composed/t{turns}")
    return result


def pooled(rows):
    pairs = ("final_action_pairs", "final_reply_pairs", "all_opposite_action_pairs",
             "final_known_action", "known", "query", "query_reply", "unknown")
    return {key: {field: sum(row["counts"][key][field] for row in rows) for field in ("correct", "total")}
            for key in pairs} | {"unsupported_ask_count": sum(row["unsupported_ask_count"] for row in rows)}


def project_metrics(metrics, role, control, config):
    """Cheap cached-score consistency and projection; no canonical regeneration."""
    rows = metrics["per_bank"]
    if not rows:
        raise ValueError("nonempty scored cells required")
    cells, groups = {}, {}
    for name, row in rows.items():
        panel, family, depth, operator, turns = cell(name)
        identity = row["bank"]
        if (identity["version"] != VERSION or identity["role"] != role or identity["config"] != config
                or identity["episodes"] != row["episodes"] or identity["turns"] != int(turns)
                or row["turns_per_episode"] != int(turns) or row["control"] != control
                or row["free_running_replies"] is not True or row["teacher_used_for_policy"] is not False
                or row["decoder_prefix"] != "BOS only"):
            raise ValueError("cached observation/reply boundary or bank identity differs")
        n = integer(row["episodes"], "episodes")
        if not n or n % 2 or row["final_pairs"]["total"] != n//2 or row["final_total"] != n:
            raise ValueError("complete final-known pair denominator differs")
        if row["known_total"] < n:
            raise ValueError("known questions must include every final question")
        counts = {"final_action_pairs": deepcopy(row["final_pairs"]),
            "final_reply_pairs": {"correct": row["final_reply_pair_correct"], "total": n//2},
            "all_opposite_action_pairs": {"correct": row["opposite_pair_correct"], "total": row["opposite_pair_total"]},
            "final_known_action": {"correct": row["final_correct"], "total": n},
            "known": {"correct": row["known_correct"], "total": row["known_total"]},
            "query": {"correct": row["query_correct"], "total": row["query_total"]},
            "query_reply": {"correct": row["query_reply_correct"], "total": row["query_total"]},
            "unknown": {"correct": row["ask_correct"], "total": row["ask_true"]}}
        for group, metric in (("final_action_pairs", "final_pair_accuracy"), ("final_reply_pairs", "final_reply_pair_accuracy"),
                ("all_opposite_action_pairs", "opposite_pair_accuracy"),
                ("final_known_action", "final_accuracy"), ("known", "known_accuracy"), ("query", "query_accuracy"),
                ("query_reply", "query_reply_accuracy"), ("unknown", "ask_recall")):
            ratio(row[metric], **counts[group], name=metric)
        if row["query_total"] != row["known_total"]+row["ask_true"] or row["query_correct"] != row["known_correct"]+row["ask_correct"]:
            raise ValueError("known/unknown query counts differ")
        ratio(row["ask_precision"], row["ask_correct"], row["ask_predicted"], "ASK precision")
        unsupported = row["ask_predicted"]-row["ask_correct"]
        integer(unsupported, "unsupported ASK")
        if unsupported > row["known_total"]:
            raise ValueError("unsupported ASK exceeds known questions")
        value = {"bank": deepcopy(identity), "panel": panel, "family": family, "depth": int(depth[1:]),
            "operator": operator, "turns": int(turns), "counts": counts,
            "unsupported_ask_count": unsupported,
            "ask_predicted": row["ask_predicted"], "reply_parseable_queries": row["reply_parseable_queries"],
            "rates": {key: row[key] for key in (*METRICS, *EXTRA[:-1])},
            "per_target": deepcopy(row["per_target"]), "by_turn": deepcopy(row["by_turn"]),
            "confusion_matrix": deepcopy(row["confusion_matrix"])}
        value["rates"]["unsupported_ask_rate"] = unsupported/row["known_total"]
        cells[name] = value
        groups.setdefault(f"{panel}/{family}/{operator}", []).append((name, value))
    macro = {}
    for key in METRICS:
        macro[key] = sum(row[key] for row in rows.values())/len(rows)
        if not math.isclose(metrics["macro_"+key], macro[key], rel_tol=0., abs_tol=1e-12):
            raise ValueError("equal-cell macro differs")
    return {"macro_equal_cells": macro, "pooled_counts": pooled(list(cells.values())), "cells": cells,
        "by_panel_family_mechanism": {name: {"cell_names": sorted(key for key, _ in group),
            "macro_equal_cells": {key: sum(row["rates"][key] for _, row in group)/len(group) for key in METRICS},
            "pooled_counts": pooled([row for _, row in group])} for name, group in groups.items()}}


def _score_piece(piece, role, control, protocol, admission, identities):
    metrics = piece["metrics"]
    if set(metrics["per_bank"]) != expected_cells(role):
        raise ValueError("scoring omitted a declared domain/depth/operator/length cell")
    projection = project_metrics(metrics, role, control, protocol["config"])
    counts = dict.fromkeys(SCORE_COUNTS, 0)
    score_seconds = 0.
    for name, row in metrics["per_bank"].items():
        if row["episodes"] != 2*protocol["contract"]["pairs_per_evaluation_cell"]:
            raise ValueError("prescribed pair count differs")
        key = (role, name)
        if key in identities:
            same(row["bank"], identities[key], "bank changed across curves or controls")
        identities[key] = deepcopy(row["bank"])
        if role != "train_fit":
            declared = admission[role]["banks"][name]
            if (row["bank"]["sha256"] != declared["sha256"] or row["episodes"] != declared["episodes"]
                    or row["query_total"] != sum(declared["query_target_counts"].values())
                    or row["known_total"] != sum(declared["query_target_counts"][str(k)] for k in (0, 1))
                    or row["ask_true"] != declared["query_target_counts"]["2"]
                    or row["final_pairs"]["total"] != declared["final_opposite_pair_total"]):
                raise ValueError("cached scores differ from frozen bank admission counts")
        episodes, turns = row["episodes"], row["episodes"]*row["turns_per_episode"]
        for field, value in zip(SCORE_COUNTS, (episodes, turns, turns if control == "reset" else episodes, turns)):
            counts[field] += value
        score_seconds += number(row["seconds"], "bank score seconds")
    cost = piece["cost"]
    for key, value in counts.items():
        if integer(cost[key], key) != value:
            raise ValueError("scoring exposure cost differs")
    if (cost["control"] != control or not math.isclose(cost["bank_score_seconds_included_in_wall"], score_seconds,
            rel_tol=1e-10, abs_tol=1e-7) or score_seconds > number(cost["wall_seconds"], "score wall")+1e-7):
        raise ValueError("nested scoring cost differs")
    return {**projection, "cost": deepcopy(cost)}


def summarize(directory):
    started = time.monotonic()
    helper_hash = file_hash(__file__)
    directory = Path(directory).resolve()
    protocol, verified, report, preparation, inputs = load_complete(directory)
    admission = read(directory/"bank-admission.json")
    output, identities = {}, {}
    same(verified["arms"][ARMS[0]]["endpoint_evidence"]["exposures"],
         verified["arms"][ARMS[1]]["endpoint_evidence"]["exposures"], "matched arm exposures differ")
    for arm in ARMS:
        scored = report["results"][arm]
        curves = [{"updates": point["updates"], "weights_sha256": point["weights_sha256"],
            **_score_piece(point, "dev", "normal", protocol, admission, identities)} for point in scored["development_curve"]]
        fit = _score_piece(scored["train_fit"], "train_fit", "normal", protocol, admission, identities)
        audit = _score_piece(scored["audit"], "audit", "normal", protocol, admission, identities)
        if set(scored["controls"]) != {"blank", "reset"}:
            raise ValueError("both endpoint controls required")
        controls = {name: _score_piece(piece, "audit", name, protocol, admission, identities)
                    for name, piece in scored["controls"].items()}
        pieces = [*curves, fit, audit, *controls.values()]
        cost = scored["cost"]
        for key in SCORE_COUNTS:
            if cost[key] != sum(piece["cost"][key] for piece in pieces):
                raise ValueError("arm scoring exposure sum differs")
        score_wall = sum(piece["cost"]["wall_seconds"] for piece in pieces)
        if not math.isclose(cost["score_wall_seconds_included_in_arm_wall"], score_wall, rel_tol=1e-10, abs_tol=1e-7) or score_wall > number(scored["wall_seconds"], "arm score wall")+1e-7:
            raise ValueError("nested arm scoring wall differs")
        journal, receipt = verified["arms"][arm]["journal"], verified["arms"][arm]["receipt"]
        for key in PHYSICAL:
            expected = protocol["contract"]["updates"] * (1 if key == PHYSICAL[0] else 3*protocol["contract"]["plan_options"]["micro_batch_size"])
            if integer(journal[key], key) != expected or receipt[key] != expected:
                raise ValueError("verified physical totals disagree with retained plan")
        if not (number(journal["materialization_seconds_included_in_step"], "generation time")
                <= number(journal["retained_step_seconds"], "step time")
                <= number(journal["step_invocation_seconds"], "step invocation")
                <= number(receipt["wall_seconds"], "training wall")+1e-7):
            raise ValueError("nested training cost differs")
        output[arm] = {"development_curve": curves, "endpoint_train_fit": fit, "endpoint_audit": audit,
            "endpoint_controls": controls, "training_cost": {"journal": deepcopy(journal),
                "worker_wall_seconds": receipt["wall_seconds"], "setup_seconds_included": receipt["setup_seconds"],
                "peak_cuda_allocated_mib": receipt["peak_cuda_allocated_mib"], "peak_cuda_reserved_mib": receipt["peak_cuda_reserved_mib"],
                "exposures_by_domain": deepcopy(verified["arms"][arm]["endpoint_evidence"]["exposures"])},
            "evaluation_cost": {**deepcopy(cost), "arm_wall_seconds": scored["wall_seconds"]}}
    if sum(report["results"][arm]["wall_seconds"] for arm in ARMS)+number(report["preparation_seconds"], "scoring preparation") > number(report["wall_seconds"], "total evaluation wall")+1e-7:
        raise ValueError("evaluation component time exceeds total wall")
    # Catch mutation during this read-only projection, without any model replay.
    for path, sha in inputs.items():
        if file_hash(path) != sha:
            raise ValueError("bound evidence changed during projection")
    if file_hash(__file__) != helper_hash:
        raise ValueError("summary helper source changed during projection")
    return {"schema": SCHEMA, "status": "completed_descriptive", "automatic_promotion": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "contract": deepcopy(protocol["contract"]),
        "config": deepcopy(protocol["config"]), "learning_rate": protocol["learning_rate"], "arms": output,
        "cost": {"retained_optimizer_updates": sum(output[arm]["training_cost"]["journal"][PHYSICAL[0]] for arm in ARMS),
            "completed_microbatch_episode_exposures": sum(output[arm]["training_cost"]["journal"][PHYSICAL[-1]] for arm in ARMS),
            "preparation_seconds": number(preparation["wall_seconds"], "preparation wall"),
            "verification_seconds": number(verified["wall_seconds"], "verification wall"),
            "evaluation_wall_seconds": report["wall_seconds"],
            "evaluation_preparation_seconds_included": report["preparation_seconds"],
            "helper_wall_seconds": time.monotonic()-started},
        "integrity": {"helper_sha256": helper_hash, "input_file_sha256": inputs,
            "source_sha256": deepcopy(protocol["source_sha256"]), "canonical_replay_repeated": False,
            "neural_training_or_inference": False},
        "interpretation": {"initializations": 1, "selection_or_promotion": False,
            "final_queries": "All final questions are known and oppositely paired; earlier questions include unknown probes.",
            "aggregation": "Macro rates weight measured cells equally. Pooled integer counts retain their own denominators; every domain, depth, mechanism and length stays visible.",
            "retention": "Development curves are post-training evaluations of saved checkpoints. They permit per-cell retention analysis; a pooled gain is not uniform retention.",
            "scope": "Ordering within the identical admitted foundation lesson multiset, with a common mixed tail. One initialization; descriptive only, no general-intelligence or autonomous-learning claim.",
            "cost": "Materialization is included in step time, setup/steps/saves in worker wall, scoring pieces/preparation in evaluation wall. Overlapping worker sums are not elapsed time or dedicated GPU hours. Separate execution probes and failed preparation are outside formal retained totals.",
            "validation": "File bindings and cached arithmetic rechecked. Canonical admission, full optimizer restores and prediction provenance rely on the authenticated completed pipeline; no fresh replay or inference."}}


def write_summary(directory):
    path = Path(directory).resolve()/"evaluation/summary.json"
    if path.exists():
        raise FileExistsError("foundation completion summary already exists")
    value = summarize(directory)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    value = write_summary(args.directory)
    print(json.dumps({"status": value["status"], "output": str(Path(args.directory)/"evaluation/summary.json"), "cost": value["cost"]}))


if __name__ == "__main__":
    main()
