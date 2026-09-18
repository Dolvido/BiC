"""Compact evidence from a completed architecture study; never selects weights."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.train_cognitive import atomic_json


def compact(metrics):
    result = {key: metrics[key] for key in (
        "macro_query_accuracy", "macro_pair_accuracy", "macro_later_known_accuracy")}
    result["per_bank"] = {name: {"query_accuracy": row["query_accuracy"],
        "paired_accuracy": row["counterfactual_accuracy"],
        "paired_total": row["counterfactual_query_pairs"],
        "later_known": row["positions"]["last_known_later_query"],
        "ask_precision": row["ask_precision"], "ask_recall": row["ask_recall"],
        "ask_true": row["ask_true"], "ask_predicted": row["ask_predicted"],
        "brier_score": row["brier_score"],
        "query_reply_exact_accuracy": row["query_reply_exact_accuracy"],
        "action_reply_agreement": row["action_reply_agreement"]}
        for name, row in metrics["per_bank"].items()}
    return result


def curve_areas(curve):
    keys = ("macro_query_accuracy", "macro_pair_accuracy", "macro_later_known_accuracy")
    return {key: sum((right["updates"] - left["updates"]) * (right[key] + left[key]) / 2
        for left, right in zip(curve, curve[1:])) / curve[-1]["updates"] for key in keys}


def summarize(directory):
    directory = Path(directory)
    audit_path = directory / "audit" / "report.json"
    audit = json.loads(audit_path.read_text(encoding="utf8"))
    calibration = [json.loads(path.read_text(encoding="utf8"))
                   for path in sorted((directory / "calibration").glob("*/report.json"))]
    main = [json.loads(path.read_text(encoding="utf8"))
            for path in sorted((directory / "main").glob("*/report.json"))]
    if len(calibration) != 9 or len(main) != 3 or len(audit["results"]) != 6:
        raise ValueError("summary requires all nine calibration, three main and six adaptation results")
    result = {"schema": "bic-sequence-study-summary-v1",
        "audit_report_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "selected_rates": audit["inputs"]["selected_rates"],
        "calibration_updates": sum(row["updates"] for row in calibration),
        "calibration_exposures": sum(row["examples"] for row in calibration),
        "calibration_worker_training_seconds": sum(row["training_seconds"] for row in calibration),
        "main_updates": sum(row["updates"] for row in main),
        "main_exposures": sum(row["examples"] for row in main),
        "main_worker_training_seconds": sum(row["training_seconds"] for row in main),
        "adaptation_updates": 6 * 64, "adaptation_exposures": 6 * 64 * 64,
        "adaptation_worker_training_seconds": sum(row["training_seconds"] for row in audit["results"].values()),
        "timing_note": "Concurrent main/calibration workers share one RTX 5080; summed worker intervals "
            "are not dedicated GPU-hours. Audit adaptations run sequentially. Training timing excludes "
            "most scoring and serialization. Reported main/calibration wall intervals start after bank "
            "preparation and trainer construction; they are not full process lifetimes.",
        "main": {row["protocol"]["architecture"]: {
            "parameters": row["parameters"], "learning_rate": row["protocol"]["learning_rate"],
            "training_seconds": row["training_seconds"], "wall_seconds": row["wall_seconds"],
            "peak_cuda_allocated_mib": row["peak_cuda_allocated_mib"],
            "peak_cuda_scope": row["peak_cuda_scope"],
            "development": compact(row["development"]),
            "training_fit_diagnostic": compact(row["training_fit_diagnostic"])} for row in main},
        "candidates": {}, "prior_learning_gain_over_fresh_auc": audit["prior_learning_gain_over_fresh_auc"],
        "base_checkpoints_unchanged": audit["base_checkpoints_unchanged"],
        "automatic_promotion": False}
    for name, row in audit["results"].items():
        result["candidates"][name] = {
            "curve": [{"updates": step["updates"], **compact(step)} for step in row["curve"]],
            "descriptive_curve_areas": curve_areas(row["curve"]),
            "normalized_adaptation_auc": row["normalized_adaptation_auc"],
            "retention_before": compact(row["retention_before"]),
            "retention_after": compact(row["retention_after"]),
            "advanced_before": compact(row["advanced_before"]),
            "advanced_after": compact(row["advanced_after"]),
            "controls_at_64": {key: compact(value) for key, value in row["controls_at_64"].items()},
            "cpu_restart": row["cpu_restart"]}
    result["descriptive_prior_learning_area_gains"] = {architecture: {
        metric: result["candidates"][f"{architecture}-pretrained"]["descriptive_curve_areas"][metric]
            - result["candidates"][f"{architecture}-fresh"]["descriptive_curve_areas"][metric]
        for metric in ("macro_query_accuracy", "macro_pair_accuracy", "macro_later_known_accuracy")}
        for architecture in result["selected_rates"]}
    atomic_json(directory / "audit" / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    result = summarize(parser.parse_args().study)
    print(json.dumps({"selected_rates": result["selected_rates"],
        "main": {key: {name: value["development"][name] for name in (
            "macro_query_accuracy", "macro_pair_accuracy", "macro_later_known_accuracy")}
            for key, value in result["main"].items()},
        "prior_learning_gain_over_fresh_auc": result["prior_learning_gain_over_fresh_auc"]}, indent=2))


if __name__ == "__main__":
    main()
