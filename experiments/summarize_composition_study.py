"""Read-only evidence verification and descriptive composition comparisons.

No result or checkpoint is opened until all five training reports, every fixed
checkpoint and all five audit results exist. This helper never scores a model,
trains a model, selects a checkpoint or promotes a candidate.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_loop import run_lock
from brain_in_computer.learning_student import _check_finite_tree
from experiments import composition_study as study
from experiments.composition_banks import bank_manifest, verify_boundaries
from experiments.composition_evaluation import METRICS
from experiments.composition_training import CompositionTrainer, COUNTS
from experiments.sequence_student import build_sequence_student
from experiments.train_cognitive import atomic_json


SCHEMA = "bic-composition-summary-v1"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _json_equal(left, right):
    def canonical(value):
        # Checkpoint buckets have integer keys; JSON protocols have string
        # keys. Normalize before sorting so 8/10/12 cannot acquire different
        # numeric-versus-lexical order across those representations.
        normalized = json.loads(json.dumps(value, allow_nan=False))
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return canonical(left) == canonical(right)


def _mean(values):
    return sum(values) / len(values) if values and all(value is not None for value in values) else None


def compact(metrics):
    """Keep denominators, target mixtures, replies and per-turn results intact."""
    rows = copy.deepcopy(metrics["per_bank"])
    for row in rows.values():
        row.pop("seconds", None)
    grouped = {}
    for name, row in rows.items():
        panel, family, length = name.split("/")
        grouped.setdefault(panel, {}).setdefault(family, {"per_length": {}})["per_length"][length] = row
    for families in grouped.values():
        for group in families.values():
            group["macro"] = {metric: _mean([row[metric] for row in group["per_length"].values()])
                              for metric in METRICS}
    return {"per_panel_family": grouped,
            "macro": {key: copy.deepcopy(value) for key, value in metrics.items() if key.startswith("macro_")}}


def curve_areas(curve, family):
    """Normalized sparse trapezoidal areas, equally weighting available lengths."""
    xs = [point["held_episode_exposure"] for point in curve]
    if (len(xs) < 2 or xs[0] != 0 or xs[-1] <= 0 or any(type(x) is not int for x in xs)
            or any(right <= left for left, right in zip(xs, xs[1:]))):
        raise ValueError("curve requires increasing held-episode exposure beginning at zero")
    groups = [compact(point["metrics"])["per_panel_family"] for point in curve]
    panels = set(groups[0])
    if any(set(group) != panels or any(family not in group[panel] for panel in panels) for group in groups):
        raise ValueError("curve panel or family differs")
    result = {}
    for panel in sorted(panels):
        lengths = set(groups[0][panel][family]["per_length"])
        if any(set(group[panel][family]["per_length"]) != lengths for group in groups):
            raise ValueError("curve length mixture differs")
        result[panel] = {}
        for metric in METRICS:
            values = [group[panel][family]["macro"][metric] for group in groups]
            result[panel][metric] = (sum((b - a) * (left + right) / 2 for a, b, left, right in
                zip(xs, xs[1:], values, values[1:])) / xs[-1]) if all(value is not None for value in values) else None
    return result


def _difference(left, right):
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError("comparison axes differ")
        return {name: _difference(left[name], right[name]) for name in left}
    return None if left is None or right is None else left - right


def _metric_values(metrics, families=None):
    groups = compact(metrics)["per_panel_family"]
    return {panel: {family: {"macro": group["macro"], "per_length": {
        length: {metric: row[metric] for metric in METRICS} for length, row in group["per_length"].items()}}
        for family, group in values.items() if families is None or family in families}
        for panel, values in groups.items()}


def _select_metrics(metrics, families):
    selected = {name: row for name, row in metrics["per_bank"].items() if name.split("/")[1] in families}
    return {"per_bank": selected, **{f"macro_{metric}": _mean([row[metric] for row in selected.values()])
                                   for metric in METRICS}}


def _comparison(left, right, family):
    if [row["held_episode_exposure"] for row in left["curve"]] != [row["held_episode_exposure"] for row in right["curve"]]:
        raise ValueError("comparison held exposure differs")
    return {"area_difference": _difference(curve_areas(left["curve"], family), curve_areas(right["curve"], family)),
        "endpoint_difference": _difference(_metric_values(left["final"], {family}),
                                             _metric_values(right["final"], {family}))}


def assemble_summary(protocol, main, audit):
    if set(main) != set(study.JOBS) or set(audit["results"]) != set(study.JOBS):
        raise ValueError("all five candidates are required")
    results = audit["results"]
    for family in ("color", "count"):
        expected = main["joint"]["exposures"][family]
        if any(main[job]["exposures"][family] != expected for job in (f"seq-{family}", f"fresh-{family}")):
            raise ValueError("matched held-family episode and byte exposures differ")
    for job in ("seq-color", "seq-count"):
        if main[job]["exposures"] != main["joint"]["exposures"]:
            raise ValueError("main schedules must match every family exposure")
    summary = {"schema": SCHEMA, "study_schema": protocol["schema"], "automatic_promotion": False,
        "interpretation": protocol["interpretation"],
        "comparison_note": "Descriptive single-initialization comparisons. Both rotations reuse one common joint run; they are not independent joint replications. Fresh controls match held-family samples, not total updates or compute.",
        "area_note": "Sparse trapezoidal areas normalized by final held-family episode exposure; equal weighting across actual lengths. Areas combine initial performance and subsequent learning and do not isolate learning speed. Sequential zero exposure already includes the old-domain phase and its continued optimizer state.",
        "metric_note": "Seen panels use familiar structural recipes with fresh values/names; composed panels withhold final supervised ancestry signatures. Final answers are always known and opposite; final uncertainty is unmeasured. Missing replies or absent ASK denominators remain null, not zero. Query target mixtures differ by bank; denominators and per-target counts are retained.",
        "time_note": "training_seconds sums synchronized optimizer work in the durable checkpoint lineage, excluding preparation, scoring and any uncheckpointed work discarded by interruption; sums across workers are not elapsed wall time. Invocation time begins after trainer and development-bank preparation and includes development scoring and saves, but covers only the final invocation after any resume. Peak CUDA allocation also covers only that final invocation.",
        "interruption_note": "The two fresh controls were paused to reduce GPU memory pressure and resumed from their durable checkpoints. Retained update/exposure counts remain exact; discarded uncheckpointed work and its optimizer time cannot be reconstructed from these reports. Reported compute is not total physical work across interrupted invocations.",
        "main": {}, "audit": {}, "rotations": {}}
    for job, report in main.items():
        summary["main"][job] = {key: copy.deepcopy(report[key]) for key in
            ("updates", "family_microbatches", "exposures", "training_seconds", "invocation_seconds_after_preparation",
             "peak_cuda_allocated_mib", "weights_sha256")}
        summary["main"][job]["development_curve"] = [{**{key: copy.deepcopy(point[key]) for key in
            ("updates", "family_microbatches", "training_seconds")}, "metrics": compact(point["development"])}
            for point in report["history"]]
    for job, result in results.items():
        families = study.FAMILIES if job == "joint" else (job.split("-", 1)[1],)
        summary["audit"][job] = {"curve": [{"updates": point["updates"],
            "held_episode_exposure": point["held_episode_exposure"], "weights_sha256": point["weights_sha256"],
            "metrics": compact(point["metrics"])} for point in result["curve"]],
            "areas_by_family": {family: curve_areas(result["curve"], family) for family in families},
            "final": compact(result["final"]),
            "controls": {name: compact(value) for name, value in result["controls"].items()},
            "cpu_restart": copy.deepcopy(result["cpu_restart"]), "seconds": result["seconds"]}
        before = result["retained_before_new_phase"]
        if before is not None:
            old = set(study.FAMILIES) - set(families)
            summary["audit"][job]["retention"] = {"before": compact(before),
                "after": compact(_select_metrics(result["final"], old)),
                "change": _difference(_metric_values(result["final"], old), _metric_values(before, old))}
    for family in ("color", "count"):
        sequential, fresh = results[f"seq-{family}"], results[f"fresh-{family}"]
        summary["rotations"][family] = {"common_joint_reference": "joint",
            "sequential_minus_joint": _comparison(sequential, results["joint"], family),
            "sequential_minus_fresh": _comparison(sequential, fresh, family),
            "joint_minus_fresh": _comparison(results["joint"], fresh, family)}
    summary["compute"] = {"optimizer_updates": sum(row["updates"] for row in main.values()),
        "accounting_scope": "retained checkpoint lineage; excludes unknown discarded interrupted work",
        "microbatches": sum(sum(row["family_microbatches"].values()) for row in main.values()),
        "sampled_exposures": {key: sum(counts[key] for row in main.values() for counts in row["exposures"].values())
                              for key in COUNTS},
        "worker_training_seconds": sum(row["training_seconds"] for row in main.values()),
        "worker_invocation_seconds_after_preparation": sum(row["invocation_seconds_after_preparation"] for row in main.values()),
        "audit_worker_seconds": sum(row["seconds"] for row in results.values())}
    return summary


def _required(directory):
    files = [directory / name for name in ("protocol.json", "banks.pt")]
    files += [directory / "audit" / name for name in ("report.json", "evaluation-started.json")]
    files += [directory / "audit" / f"{job}.json" for job in study.JOBS]
    files += [directory / "main" / job / name for job in study.JOBS for name in
        ("latest.pt", "report.json", *(f"checkpoint-{step:06d}.pt" for step in sorted({0, *study.curve_steps(job)})))]
    return files


def _opposite_pair_denominators(banks):
    """Count eligible opposite known targets directly from admitted canonical rows."""
    return {name: [sum({rows[index]["turns"][turn]["target"],
                            rows[index + 1]["turns"][turn]["target"]} == {0, 1}
                       for index in range(0, len(rows), 2))
                   for turn in range(len(rows[0]["turns"]))] for name, rows in banks.items()}


def _validate_metrics(metrics, names, manifests, *, role, control="normal", replies=True,
                      pair_denominators=None):
    from experiments.composition_curriculum import VERSION
    if set(metrics.get("per_bank", {})) != set(names):
        raise ValueError("metric bank set differs")
    for name, row in metrics["per_bank"].items():
        manifest = manifests[name]
        query = sum(sum(turn[str(target)] for target in range(3)) for turn in manifest["target_counts_by_turn"])
        known = sum(sum(turn[str(target)] for target in range(2)) for turn in manifest["target_counts_by_turn"])
        identity = {"sha256": manifest["sha256"], "version": VERSION,
            "episodes": manifest["episodes"], "turns": manifest["turns"], "role": role, "config": asdict(study.CONFIG)}
        if (row["bank"] != identity or row["episodes"] != manifest["episodes"]
                or row["turns_per_episode"] != manifest["turns"] or row["query_total"] != query
                or row["known_total"] != known or row["final_total"] != manifest["episodes"]
                or row["final_pairs"]["total"] != manifest["complete_pairs"]
                or row["teacher_used_for_policy"] is not False or row["decoder_prefix"] != "BOS only"
                or row["free_running_replies"] is not replies or row["control"] != control):
            raise ValueError("metric denominator, bank or inference boundary differs")
        checks = [("query_accuracy", row["query_correct"], query),
            ("known_accuracy", row["known_correct"], known),
            ("final_accuracy", row["final_correct"], manifest["episodes"]),
            ("final_pair_accuracy", row["final_pairs"]["correct"], manifest["complete_pairs"]),
            ("opposite_pair_accuracy", row["opposite_pair_correct"], row["opposite_pair_total"])]
        if replies:
            checks += [("query_reply_accuracy", row["query_reply_correct"], query),
                ("final_reply_pair_accuracy", row["final_reply_pair_correct"], manifest["complete_pairs"])]
        elif any(row[key] is not None for key in ("query_reply_accuracy", "final_reply_pair_accuracy", "action_reply_agreement")):
            raise ValueError("unmeasured reply metrics must remain null")
        confusion = row["confusion_matrix"]
        if (not isinstance(confusion, list) or len(confusion) != 4
                or any(not isinstance(values, list) or len(values) != 4 for values in confusion)
                or any(type(value) is not int or value < 0 for values in confusion for value in values)):
            raise ValueError("query confusion counts differ")
        target_totals = [sum(turn[str(target)] for turn in manifest["target_counts_by_turn"]) for target in range(3)]
        if ([sum(values) for values in confusion] != target_totals + [0]
                or sum(confusion[target][target] for target in range(3)) != row["query_correct"]
                or confusion[0][0] + confusion[1][1] != row["known_correct"]
                or set(row["per_target"]) != {"0", "1", "2"}):
            raise ValueError("query target mixture differs")
        for target, total in enumerate(target_totals):
            score = row["per_target"][str(target)]
            if score["total"] != total or score["correct"] != confusion[target][target]:
                raise ValueError("per-target denominator differs")
            expected = score["correct"] / total if total else None
            if (score["accuracy"] is not None if expected is None else
                    score["accuracy"] is None or not math.isclose(score["accuracy"], expected, abs_tol=1e-12)):
                raise ValueError("per-target accuracy differs")
        ask_total, ask_prediction, ask_correct = target_totals[2], sum(values[2] for values in confusion), confusion[2][2]
        if (row["ask_true"], row["ask_predicted"], row["ask_correct"]) != (ask_total, ask_prediction, ask_correct):
            raise ValueError("ASK denominator differs")
        for ask_metric, denominator in (("ask_precision", ask_prediction), ("ask_recall", ask_total)):
            expected = ask_correct / denominator if denominator else None
            if (row[ask_metric] is not None if expected is None else
                    row[ask_metric] is None or not math.isclose(row[ask_metric], expected, abs_tol=1e-12)):
                raise ValueError("ASK rate differs")
        if len(row["by_turn"]) != manifest["turns"]:
            raise ValueError("per-turn metric length differs")
        expected_pairs = None if pair_denominators is None else pair_denominators[name]
        if expected_pairs is not None and len(expected_pairs) != manifest["turns"]:
            raise ValueError("canonical per-turn pair denominator length differs")
        for index, turn in enumerate(row["by_turn"]):
            total = sum(manifest["target_counts_by_turn"][index][str(target)] for target in range(3))
            if (type(turn["turn"]) is not int or turn["turn"] != index
                    or type(turn["total"]) is not int or turn["total"] != total
                    or type(turn["correct"]) is not int or not 0 <= turn["correct"] <= total):
                raise ValueError("per-turn query denominator differs")
            if (turn["accuracy"] is not None if total == 0 else
                    type(turn["accuracy"]) not in (int, float) or
                    not math.isclose(turn["accuracy"], turn["correct"] / total, rel_tol=0, abs_tol=1e-7)):
                raise ValueError("per-turn query accuracy differs")
            pair_total, pair_correct = turn["opposite_pair_total"], turn["opposite_pair_correct"]
            if (type(pair_total) is not int or type(pair_correct) is not int
                    or not 0 <= pair_correct <= pair_total <= manifest["complete_pairs"]
                    or pair_correct * 2 > turn["correct"]
                    or expected_pairs is not None and pair_total != expected_pairs[index]):
                raise ValueError("per-turn opposite-pair denominator differs")
        if (sum(turn["total"] for turn in row["by_turn"]) != row["query_total"]
                or sum(turn["correct"] for turn in row["by_turn"]) != row["query_correct"]
                or sum(turn["opposite_pair_total"] for turn in row["by_turn"]) != row["opposite_pair_total"]
                or sum(turn["opposite_pair_correct"] for turn in row["by_turn"]) != row["opposite_pair_correct"]
                or row["by_turn"][-1]["opposite_pair_total"] != row["final_pairs"]["total"]
                or row["by_turn"][-1]["opposite_pair_correct"] != row["final_pairs"]["correct"]
                or row["by_turn"][-1]["correct"] != row["final_correct"]):
            raise ValueError("per-turn counts disagree with global metrics")
        if replies and (type(row["reply_parseable_queries"]) is not int
                or not 0 <= row["reply_parseable_queries"] <= query
                or type(row["action_reply_agreement"]) not in (int, float)
                or not 0 <= row["action_reply_agreement"] <= 1):
            raise ValueError("reply parseability or agreement denominator differs")
        for metric, correct, total in checks:
            if (type(correct) is not int or type(total) is not int or not 0 <= correct <= total
                    or (row[metric] is not None if total == 0 else
                        type(row[metric]) not in (int, float) or
                        not math.isclose(row[metric], correct / total, rel_tol=0, abs_tol=1e-7))):
                raise ValueError("metric count and accuracy differ")
    for metric in METRICS:
        expected = _mean([row[metric] for row in metrics["per_bank"].values()])
        actual = metrics[f"macro_{metric}"]
        if (actual is not None if expected is None else
                actual is None or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("macro weighting differs")


def verify_completed(directory):
    directory = Path(directory)
    required = _required(directory)
    if any(not path.is_file() for path in required):
        raise ValueError("summary requires all five completed runs, checkpoints and five completed audits")
    hashes = {path.relative_to(directory).as_posix(): study.file_hash(path) for path in required}
    protocol = study.load_protocol(directory)
    if (protocol["jobs"] != list(study.JOBS) or protocol["seed"] != study.SEED
            or protocol["sampler_seed"] != study.SAMPLER_SEED or protocol["config"] != asdict(study.CONFIG)
            or protocol["learning_rate"] != study.RATE or protocol["micro_batch_size"] != study.MICRO
            or protocol["microbatches_per_update"] != 3 or protocol["family_order"] != list(study.FAMILIES)
            or protocol["automatic_promotion"] is not False
            or protocol["held_episode_budgets"] != list(study.EXPOSURES)
            or protocol["updates"] != {job: study.total_updates(job) for job in study.JOBS}
            or protocol["curve_steps"] != {job: study.curve_steps(job) for job in study.JOBS}
            or protocol["final_family_microbatches"] != {job: study.expected_counts(job, study.total_updates(job))
                                                        for job in study.JOBS}):
        raise ValueError("fixed study protocol differs")
    main_hashes = {name: sha for name, sha in hashes.items() if name.startswith("main/")}
    marker = {"schema": study.SCHEMA, "protocol_sha256": hashes["protocol.json"], "input_files_sha256": main_hashes}
    audit = _read(directory / "audit" / "report.json")
    if (_read(directory / "audit" / "evaluation-started.json") != marker or audit.get("inputs") != marker
            or audit.get("protocol") != protocol or audit.get("schema") != study.SCHEMA
            or audit.get("base_checkpoints_unchanged") is not True or audit.get("automatic_promotion") is not False
            or set(audit.get("results", {})) != set(study.JOBS)):
        raise ValueError("completed audit provenance differs")
    banks = torch.load(directory / "banks.pt", map_location="cpu", weights_only=True)
    manifests = bank_manifest(banks)
    if manifests != protocol["bank_manifest"] or verify_boundaries(banks) != protocol["bank_diagnostics"]["boundaries"]:
        raise ValueError("canonical bank manifest or boundaries differ")
    pair_denominators = {group: _opposite_pair_denominators(banks[group]) for group in ("development", "audit")}
    evidence = study.SampleEvidence(banks["train"])
    initial_digest = checkpoint_digest(build_sequence_student(study.SEED, config=study.CONFIG))
    if initial_digest != protocol["initial_weights_sha256"]:
        raise ValueError("prescribed initial tensor digest differs")
    expected_streams = {}
    for family in study.FAMILIES:
        state, buckets, exposures = evidence.at(family, 3600)
        expected_streams[family] = {"sampler_sha256": hashlib.sha256(bytes(state.tolist())).hexdigest(),
            "bucket_microbatches": buckets, "exposures": exposures}
    if not _json_equal(expected_streams, protocol["expected_final_family_streams"]):
        raise ValueError("prospective final-family exposure streams differ")
    main, digests = {}, {}
    for job in study.JOBS:
        folder = directory / "main" / job
        report = _read(folder / "report.json")
        saved = torch.load(folder / "latest.pt", map_location="cpu", weights_only=True)
        study.validate_snapshot(saved, protocol, job, evidence)
        admitted = ({job.split("-", 1)[1]: banks["train"][job.split("-", 1)[1]]}
                    if job.startswith("fresh-") else banks["train"])
        trainer = CompositionTrainer(admitted, config=study.CONFIG, payload=saved["training"])
        if (report.get("schema") != study.SCHEMA or report["job"] != job or report["protocol"] != protocol
                or report["updates"] != study.total_updates(job) or saved["training"]["updates"] != study.total_updates(job)
                or report["checkpoint_file_sha256"] != hashes[f"main/{job}/latest.pt"]
                or report["weights_sha256"] != checkpoint_digest(trainer.model)
                or report["history"] != saved["history"] or report["training_seconds"] != saved["training_seconds"]
                or report["family_microbatches"] != saved["training"]["family_microbatches"]
                or report["exposures"] != saved["training"]["exposures"] or report["heldout_evaluation_performed"] is not False):
            raise ValueError("completed training report differs from checkpoint")
        steps = sorted({0, *study.curve_steps(job)})
        if [point["updates"] for point in report["history"]] != steps:
            raise ValueError("development checkpoint schedule differs")
        digests[job] = {}
        for step in steps:
            checkpoint = torch.load(folder / f"checkpoint-{step:06d}.pt", map_location="cpu", weights_only=True)
            study.validate_snapshot(checkpoint, protocol, job, evidence)
            trainer._restore(checkpoint["training"])
            if (checkpoint["training"]["updates"] != step
                    or checkpoint["history"] != [point for point in report["history"] if point["updates"] <= step]
                    or checkpoint["training_seconds"] != checkpoint["history"][-1]["training_seconds"]):
                raise ValueError("fixed checkpoint step/history differs")
            digests[job][step] = checkpoint_digest(trainer.model)
        if digests[job][0] != initial_digest or digests[job][steps[-1]] != report["weights_sha256"]:
            raise ValueError("initial or final fixed checkpoint weights differ")
        for point in report["history"]:
            if point["family_microbatches"] != study.expected_counts(job, point["updates"]):
                raise ValueError("development family count differs")
            _validate_metrics(point["development"], banks["development"], manifests["development"], role="dev", replies=False,
                              pair_denominators=pair_denominators["development"])
        main[job] = report
        result = audit["results"][job]
        if (result != _read(directory / "audit" / f"{job}.json") or result.get("schema") != study.SCHEMA
                or result["job"] != job or result["inputs"] != marker or result["automatic_promotion"] is not False
                or [point["updates"] for point in result["curve"]] != study.curve_steps(job)
                or [point["held_episode_exposure"] for point in result["curve"]] != list(study.EXPOSURES)
                or result["cpu_restart"].get("exact") is not True or result["cpu_restart"].get("device") != "cpu"
                or result["cpu_restart"].get("weights_sha256") != report["weights_sha256"]
                or set(result["controls"]) != {"blank", "reset"}):
            raise ValueError("completed candidate audit identity differs")
        selected = {name for name in banks["audit"] if job == "joint" or name.split("/")[1] == job.split("-", 1)[1]}
        for point in result["curve"]:
            if point["weights_sha256"] != digests[job][point["updates"]]:
                raise ValueError("audit curve tensor digest differs")
            _validate_metrics(point["metrics"], selected, manifests["audit"], role="audit",
                              pair_denominators=pair_denominators["audit"])
        _validate_metrics(result["final"], banks["audit"], manifests["audit"], role="audit",
                          pair_denominators=pair_denominators["audit"])
        for control, scores in result["controls"].items():
            _validate_metrics(scores, selected, manifests["audit"], role="audit", control=control,
                              pair_denominators=pair_denominators["audit"])
        if job.startswith("seq-"):
            _validate_metrics(result["retained_before_new_phase"], set(banks["audit"]) - selected,
                              manifests["audit"], role="audit", pair_denominators=pair_denominators["audit"])
        elif result["retained_before_new_phase"] is not None:
            raise ValueError("retention-before phase is only defined for sequential runs")
        del trainer
    result = assemble_summary(protocol, main, audit)
    if result["compute"]["optimizer_updates"] != 13200 or result["compute"]["sampled_exposures"]["episodes"] != 1267200:
        raise ValueError("completed study compute budget differs")
    result["verification"] = {"input_file_sha256": hashes, "source_sha256": protocol["source_sha256"],
        "verified_source_files": len(protocol["source_sha256"]), "initial_weights_sha256": initial_digest,
        "checkpoint_weights_sha256": digests, "helper_sha256": study.file_hash(__file__),
        "bank_boundaries": protocol["bank_diagnostics"]["boundaries"],
        "no_neural_inference_or_training_performed": True}
    study.load_protocol(directory)
    if any(study.file_hash(directory / name) != sha for name, sha in hashes.items()):
        raise RuntimeError("study evidence changed while constructing summary")
    _check_finite_tree(result, "composition summary")
    return result


def summarize(directory):
    directory = Path(directory)
    if any(not path.is_file() for path in _required(directory)):
        raise ValueError("summary requires all five completed runs, checkpoints and five completed audits")
    with run_lock(directory / "audit"):
        result = verify_completed(directory)
        atomic_json(directory / "audit" / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = summarize(args.study)
    print(json.dumps({"compute": result["compute"], "verified_source_files": result["verification"]["verified_source_files"]}, indent=2))


if __name__ == "__main__":
    main()
