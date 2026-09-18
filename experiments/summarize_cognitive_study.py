"""Summarize the fixed general-learning study; never select checkpoints or train."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def summarize(directory):
    directory = Path(directory)
    audit = read(directory / "audit/report.json")
    results = audit["results"]
    main = {f"{mode}-{order}": read(directory / f"{mode}-{order}/report.json")
            for mode in ("episodic", "recurrent") for order in ("interleaved", "blocked")}
    familiar, forgetting, common_compute = {}, {}, {}
    common_seconds = min(row["training_seconds"] for row in main.values())
    for key, report in main.items():
        familiar[key] = {name: {metric: row[metric] for metric in
            ("query_accuracy", "counterfactual_accuracy", "query_reply_exact_accuracy")}
            for name, row in report["development"]["per_family"].items()}
        forgetting[key] = {family: max(0., max(item["development"]["per_family"][family]["query_accuracy"]
            for item in report["history"]) - final["query_accuracy"])
            for family, final in report["development"]["per_family"].items()}
        elapsed, selected = 0., None
        for row in report["history"]:
            elapsed += row["training_seconds"]
            if elapsed <= common_seconds + 1e-6:
                selected = {"updates": row["updates"], "worker_training_seconds": elapsed,
                    "query": row["development"]["macro_query_accuracy"],
                    "pairs": row["development"]["macro_pair_accuracy"]}
        common_compute[key] = selected

    screens = {}
    for order in ("interleaved", "blocked"):
        candidate, control = results[f"episodic-{order}"], results[f"recurrent-{order}"]
        ce, be = candidate["curve"][-1], control["curve"][-1]
        requirements = {
            "heldout_query_gain_at_least_10pp": ce["macro_query_accuracy"] - be["macro_query_accuracy"] >= .10 - 1e-9,
            "heldout_pairs_positive_and_not_worse": ce["macro_pair_accuracy"] > 0 and ce["macro_pair_accuracy"] >= be["macro_pair_accuracy"] - 1e-9,
            "adaptation_auc_improves": candidate["normalized_adaptation_auc"] > control["normalized_adaptation_auc"],
        }
        # Apply the retention requirement before and after adaptation. A missing
        # uncertainty denominator cannot count as a passing measured result.
        for phase in ("retention_before", "retention_after"):
            c, b = candidate[phase], control[phase]
            requirements[f"{phase}_macro_query_within_2pp"] = c["macro_query_accuracy"] >= b["macro_query_accuracy"] - .02 - 1e-9
            requirements[f"{phase}_macro_pairs_within_2pp"] = c["macro_pair_accuracy"] >= b["macro_pair_accuracy"] - .02 - 1e-9
            requirements[f"{phase}_every_family_within_5pp"] = all(row["query_accuracy"] >= b["per_bank"][family]["query_accuracy"] - .05 - 1e-9 for family, row in c["per_bank"].items())
        for bank, row in ce["per_bank"].items():
            for metric in ("ask_precision", "ask_recall"):
                c, b = row[metric], be["per_bank"][bank][metric]
                requirements[f"{bank}_{metric}_measured_within_2pp"] = None if c is None or b is None else c >= b - .02 - 1e-9
        screens[order] = {"passed": all(value is True for value in requirements.values()),
            "requirements": requirements,
            "query_gain": ce["macro_query_accuracy"] - be["macro_query_accuracy"],
            "pair_gain": ce["macro_pair_accuracy"] - be["macro_pair_accuracy"],
            "auc_gain": candidate["normalized_adaptation_auc"] - control["normalized_adaptation_auc"]}

    compact = {}
    for key, result in results.items():
        compact[key] = {"curve": [{name: row[name] for name in
            ("updates", "support_exposures", "macro_query_accuracy", "macro_pair_accuracy")}
            for row in result["curve"]],
            "heldout_by_level": {level: [{"updates": row["updates"], **{
                metric: row["per_bank"][level][metric] for metric in
                ("query_accuracy", "query_total", "counterfactual_accuracy", "counterfactual_query_pairs",
                 "allow_deny_macro_accuracy", "ask_precision", "ask_recall", "ask_true", "ask_predicted",
                 "brier_score", "query_reply_exact_accuracy", "action_reply_agreement")}}
                for row in result["curve"]] for level in result["curve"][0]["per_bank"]},
            "adaptation_auc": result["normalized_adaptation_auc"],
            "retained_query_before": result["retention_before"]["macro_query_accuracy"],
            "retained_query_after": result["retention_after"]["macro_query_accuracy"],
            "retained_pairs_before": result["retention_before"]["macro_pair_accuracy"],
            "retained_pairs_after": result["retention_after"]["macro_pair_accuracy"],
            "retention_delta_by_family": result["retention_delta_by_family"],
            "advanced_query_before": result["trained_family_transfer_before"]["macro_query_accuracy"],
            "advanced_pairs_before": result["trained_family_transfer_before"]["macro_pair_accuracy"],
            "advanced_query_after": result["trained_family_transfer_after"]["macro_query_accuracy"],
            "advanced_pairs_after": result["trained_family_transfer_after"]["macro_pair_accuracy"],
            "controls_at_16": {name: {"query": row["macro_query_accuracy"],
                "pairs": row["macro_pair_accuracy"], "per_level": {level: {
                    "query": value["query_accuracy"], "pairs": value["counterfactual_accuracy"]}
                    for level, value in row["per_bank"].items()}}
                for name, row in result["controls_at_16"].items()},
            "cpu_restart_exact": result["cpu_restart"]["exact"]}
    return {"schema": "bic-general-learning-summary-v1",
        "audit_report_sha256": hashlib.sha256((directory / "audit/report.json").read_bytes()).hexdigest(),
        "parameters_per_arm": {key: value["parameters"] for key, value in main.items()},
        "main_updates": sum(value["updates"] for value in main.values()),
        "main_repeated_episode_exposures": sum(value["examples"] for value in main.values()),
        "main_worker_training_seconds": sum(value["training_seconds"] for value in main.values()),
        "main_max_worker_wall_seconds": max(value["wall_seconds"] for value in main.values()),
        "adaptation_updates": len(results) * 16, "adaptation_repeated_episode_exposures": len(results) * 16 * 64,
        "adaptation_worker_training_seconds": sum(value["training_seconds"] for value in results.values()),
        "timing_note": "Four main workers shared one RTX 5080; worker wall time is not independent GPU-hours. Short standalone benchmark is separately recorded.",
        "common_worker_training_seconds_ceiling": common_seconds,
        "latest_checkpoint_within_common_time": common_compute,
        "familiar_endpoint": familiar, "development_forgetting": forgetting,
        "candidates": compact, "prior_learning_gain_over_fresh_auc": audit["prior_learning_gain_over_fresh_auc"],
        "episodic_benefit_screens": screens,
        "screen_interpretation": "Single initialization, one withheld family, one support draw. Screens select confirmatory research only; no model promotion.",
        "base_checkpoints_unchanged": audit["base_checkpoints_unchanged"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.study)
    path = args.study / "summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({"output": str(path), "main_updates": result["main_updates"],
        "episodic_benefit_screens": {key: value["passed"] for key, value in
            result["episodic_benefit_screens"].items()}, "base_checkpoints_unchanged":
            result["base_checkpoints_unchanged"]}))


if __name__ == "__main__":
    main()
