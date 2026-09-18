"""Condense the completed credit study without training or selecting endpoints."""
import argparse
import json
from pathlib import Path

from experiments.audit_cognitive_transfer import file_hash


def compact_bank(value):
    return {name: value[name] for name in ("macro_query_accuracy", "macro_pair_accuracy",
        "macro_later_known_accuracy")} | {"per_family": {family: {
            "query_accuracy": row["query_accuracy"], "pairs": row["counterfactual_accuracy"],
            "pair_total": row["counterfactual_query_pairs"],
            "later_known": row["positions"]["last_known_later_query"],
            "ask_precision": row["ask_precision"], "ask_recall": row["ask_recall"],
            "ask_true": row["ask_true"], "ask_predicted": row["ask_predicted"],
            "brier": row["brier_score"], "reply_exact": row["query_reply_exact_accuracy"],
            "action_reply_agreement": row["action_reply_agreement"]}
            for family, row in value["per_bank"].items()}}


def summarize(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf8"))
    main = {row["key"]: json.loads((Path(row["path"]) / "report.json").read_text(encoding="utf8"))
            for row in report["inputs"]["runs"]}
    candidates = {}
    for name, result in report["results"].items():
        candidates[name] = {phase: compact_bank(result[phase]) for phase in
            ("main_development", "retention_before", "retention_after", "advanced_before", "advanced_after")}
        candidates[name].update(
            adaptation_curve=[{"updates": row["updates"], **compact_bank(row)} for row in result["curve"]],
            adaptation_auc=result["normalized_adaptation_auc"],
            controls={key: compact_bank(value) for key, value in result["controls_at_16"].items()},
            retention_change={family: result["retention_after"]["per_bank"][family]["query_accuracy"] -
                row["query_accuracy"] for family, row in result["retention_before"]["per_bank"].items()},
            cpu_restart_exact=result["cpu_restart"]["exact"])
        if name in main:
            record = main[name]
            candidates[name].update(parameters=record["parameters"], main_updates=record["updates"],
                main_worker_training_seconds=record["training_seconds"],
                main_worker_wall_seconds=record["wall_seconds"],
                training_fit_diagnostic=record["training_fit_diagnostic"],
                development_forgetting={family: max(0., max(row["development"]["per_family"][family]["query_accuracy"]
                    for row in record["history"]) - value["query_accuracy"])
                    for family, value in record["development"]["per_family"].items()})
    return {"schema": "bic-cognitive-credit-summary-v1", "seed": report["manifest"]["seed"],
        "audit_report_sha256": file_hash(directory / "report.json"),
        "main_updates": sum(row["updates"] for row in main.values()),
        "main_repeated_episode_exposures": sum(row["examples"] for row in main.values()),
        "main_worker_training_seconds": sum(row["training_seconds"] for row in main.values()),
        "longest_main_worker_wall_seconds": max(row["wall_seconds"] for row in main.values()),
        "adaptation_updates": len(candidates) * 16,
        "adaptation_repeated_episode_exposures": len(candidates) * 16 * 64,
        "adaptation_worker_training_seconds": sum(row["training_seconds"] for row in report["results"].values()),
        "timing_note": "Two main workers shared one RTX 5080; summed worker time is not dedicated GPU-hours.",
        "candidates": candidates, "benefit_screen": report["benefit_screen"],
        "prior_learning_gain_over_fresh_auc": report["prior_learning_gain_over_fresh_auc"],
        "base_checkpoints_unchanged": report["base_checkpoints_unchanged"],
        "official_scoring_uses_auxiliary_logits": report["official_scoring_uses_auxiliary_logits"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.audit)
    path = args.audit / "summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({"output": str(path), "seed": result["seed"],
        "main_updates": result["main_updates"], "benefit_screen_passed": result["benefit_screen"]["passed"]}))


if __name__ == "__main__":
    main()
