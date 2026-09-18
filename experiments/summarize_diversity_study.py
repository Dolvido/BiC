"""Read-only evidence summary after all four endpoints and five audits finish.

No model is evaluated or trained here. Differences are descriptive, from one
seed and finite banks, without a significance test or automatic promotion.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_loop import run_lock
from brain_in_computer.learning_student import _check_finite_tree
from experiments import diversity_study as study
from experiments.sequence_student import build_sequence_student
from experiments.train_cognitive import atomic_json


AREA_KEYS = ("macro_query_accuracy", "macro_pair_accuracy", "macro_later_known_accuracy")
EFFECT_KEYS = ("query_accuracy", "paired_accuracy", "later_known_accuracy", "later_known_pair_accuracy",
               "query_reply_exact_accuracy", "action_reply_agreement")
PANELS = ("names", "worlds", "both")


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values and None not in values else None


def _difference(after, before):
    return None if after is None or before is None else after - before


def compact(metrics):
    """Keep denominators and missing values, including later known pairs."""
    result = {key: metrics[key] for key in AREA_KEYS}
    result["per_bank"] = {}
    for name, row in metrics["per_bank"].items():
        later = copy.deepcopy(row["positions"]["last_known_later_query"])
        total, parsed = row["query_total"], row["reply_parseable_query_count"]
        result["per_bank"][name] = {
            "query_accuracy": row["query_accuracy"], "query_correct": row["query_correct"], "query_total": total,
            "paired_accuracy": row["counterfactual_accuracy"], "paired_correct": row["counterfactual_correct"],
            "paired_total": row["counterfactual_query_pairs"], "later_known": later,
            "later_known_accuracy": later["accuracy"], "later_known_pair_accuracy": later["pairs"]["accuracy"],
            "query_reply_exact_accuracy": row["query_reply_exact_accuracy"],
            "action_reply_agreement": row["action_reply_agreement"],
            "reply_parseable_query_count": parsed,
            "reply_parseable_query_rate": parsed / total if parsed is not None and total else None,
            "unparseable_query_replies": total - parsed if parsed is not None else None,
            "ask_precision": row["ask_precision"], "ask_recall": row["ask_recall"],
            "ask_true": row["ask_true"], "ask_predicted": row["ask_predicted"],
            "brier_score": row["brier_score"]}
    return result


def panels(metrics):
    rows = compact(metrics)["per_bank"]
    result = {}
    for panel in PANELS:
        members = {name.split("/", 1)[1]: value for name, value in rows.items()
                   if name.startswith(panel + "/")}
        if not members:
            raise ValueError(f"missing diversity panel {panel}")
        result[panel] = {"per_family": members,
            "macro": {key: _mean(row[key] for row in members.values()) for key in EFFECT_KEYS}}
    if sum(len(row["per_family"]) for row in result.values()) != len(rows):
        raise ValueError("unrecognized diversity panel bank")
    return result


def metric_change(after, before):
    if set(after) != set(before):
        raise ValueError("comparison bank sets differ")
    return {name: {key: _difference(after[name][key], before[name][key]) for key in EFFECT_KEYS}
            for name in after}


def factorial_effects(panel_metrics):
    """High-minus-low fraction differences at each fixed other factor."""
    comparisons = {"naming_at_32_worlds": ("w32-n8", "w32-n1"),
        "naming_at_256_worlds": ("w256-n8", "w256-n1"),
        "worlds_at_1_naming_map": ("w256-n1", "w32-n1"),
        "worlds_at_8_naming_maps": ("w256-n8", "w32-n8")}
    result = {}
    for name, (high, low) in comparisons.items():
        result[name] = {"higher_diversity": high, "lower_diversity": low, "panels": {}}
        for panel in PANELS:
            a, b = panel_metrics[high][panel], panel_metrics[low][panel]
            result[name]["panels"][panel] = {
                "macro_difference": {key: _difference(a["macro"][key], b["macro"][key]) for key in EFFECT_KEYS},
                "per_family_difference": metric_change(a["per_family"], b["per_family"])}
    return result


def curve_areas(curve):
    updates = [row["updates"] for row in curve]
    if (len(updates) < 2 or updates[0] != 0 or any(type(value) is not int for value in updates)
            or any(right <= left for left, right in zip(updates, updates[1:]))):
        raise ValueError("curve must start at zero and have strictly increasing integer updates")
    result = {}
    for key in AREA_KEYS:
        values = [row[key] for row in curve]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("AUC requires finite observed accuracy values in [0,1]")
        result[key] = sum((b["updates"] - a["updates"]) * (a[key] + b[key]) / 2
                          for a, b in zip(curve, curve[1:])) / updates[-1]
    return result


def assemble_summary(protocol, main, audit):
    """Pure report calculations; filesystem verification is in summarize()."""
    if set(main) != set(study.ARMS) or set(audit["results"]) != {*study.ARMS, "fresh"}:
        raise ValueError("summary requires four main and five audit results")
    result = {"schema": "bic-diversity-study-summary-v1", "seed": protocol["seed"],
        "interpretation": protocol["interpretation"], "automatic_promotion": False,
        "effect_note": "Single-seed descriptive high-minus-low differences in accuracy fractions; "
            "no significance test. World pools can differ in class/composition frequencies. "
            "Repeated exposure counts are not independent examples or semantic equivalence classes.",
        "timing_note": "Summed worker training intervals are not dedicated GPU-hours. Training seconds "
            "exclude most scoring/serialization. Invocation seconds start after setup and can cover only "
            "the latest resumed invocation; reported peak scope is preserved per arm.",
        "reply_note": "Reply exact accuracy, parseability and action agreement quantify degradation. "
            "Generated strings were not retained, so constant-reply collapse cannot be established here.",
        "fit_note": "Fit diagnostics use the common first 32 worlds and first naming map, not each full bank; "
            "fit replies were not scored. Each development panel uses the same family macro weighting.",
        "main": {}, "adaptation": {}, "base_checkpoints_unchanged": audit["base_checkpoints_unchanged"]}
    panel_metrics = {}
    for arm, row in main.items():
        development, fitting = panels(row["development"]), compact(row["fit_subset"])
        panel_metrics[arm] = development
        fit_gaps = {panel: metric_change(values["per_family"], fitting["per_bank"])
                    for panel, values in development.items()}
        result["main"][arm] = {"updates": row["updates"], "episode_exposures": row["episodes"],
            "training_seconds": row["training_seconds"],
            "current_invocation_seconds_after_setup": row["current_invocation_seconds_after_setup"],
            "peak_cuda_allocated_mib": row["peak_cuda_allocated_mib"], "peak_scope": row["peak_scope"],
            "development": development, "fit_subset": fitting, "development_minus_fit": fit_gaps,
            "weights_sha256": row["weights_sha256"]}
    result["development_factorial_effects"] = factorial_effects(panel_metrics)
    retained_panels = {}
    for name, row in audit["results"].items():
        curve = row["curve"]
        areas = curve_areas(curve)
        before, after = compact(row["retention_before"]), compact(row["retention_after"])
        changes = metric_change(after["per_bank"], before["per_bank"])
        final = compact(curve[-1])
        result["adaptation"][name] = {"curve": [{"updates": point["updates"],
            "episode_exposures": point["exposures"], **compact(point)} for point in curve],
            "normalized_areas": areas, "updates": curve[-1]["updates"],
            "episode_exposures": curve[-1]["exposures"], "training_seconds": row["training_seconds"],
            "final_later_known_by_level": {level: scores["later_known"] for level, scores in final["per_bank"].items()},
            "retention_before": before, "retention_after": after, "retention_change": changes,
            "retention_macro_change": {key: _difference(after[key], before[key]) for key in AREA_KEYS},
            "reply_diagnostics": {bank: {"before_exact_accuracy": before["per_bank"][bank]["query_reply_exact_accuracy"],
                "after_exact_accuracy": scores["query_reply_exact_accuracy"],
                "exact_accuracy_change": changes[bank]["query_reply_exact_accuracy"],
                "action_agreement_change": changes[bank]["action_reply_agreement"],
                "before_parseable_rate": before["per_bank"][bank]["reply_parseable_query_rate"],
                "after_parseable_rate": scores["reply_parseable_query_rate"],
                "unparseable_query_replies_after": scores["unparseable_query_replies"]}
                for bank, scores in after["per_bank"].items()},
            "advanced_before": compact(row["advanced_before"]), "advanced_after": compact(row["advanced_after"]),
            "controls_at_final_budget": {key: compact(value) for key, value in row["controls"].items()},
            "cpu_restart": copy.deepcopy(row["cpu_restart"]),
            "initial_weights_sha256": row["initial_weights_sha256"], "final_weights_sha256": row["final_weights_sha256"]}
        if name != "fresh":
            retained_panels[name] = panels(row["retention_before"])
    result["audit_factorial_effects_before_adaptation"] = factorial_effects(retained_panels)
    fresh = result["adaptation"]["fresh"]["normalized_areas"]
    result["prior_learning_area_gains_vs_fresh"] = {arm: {
        key: result["adaptation"][arm]["normalized_areas"][key] - fresh[key] for key in AREA_KEYS}
        for arm in study.ARMS}
    result["compute"] = {"main_updates": sum(row["updates"] for row in main.values()),
        "main_episode_exposures": sum(row["episodes"] for row in main.values()),
        "main_worker_training_seconds": sum(row["training_seconds"] for row in main.values()),
        "adaptation_updates": sum(row["updates"] for row in result["adaptation"].values()),
        "adaptation_episode_exposures": sum(row["episode_exposures"] for row in result["adaptation"].values()),
        "adaptation_training_seconds": sum(row["training_seconds"] for row in result["adaptation"].values())}
    return result


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _weight_digest(model, weights):
    _check_finite_tree(weights, "checkpoint weights")
    model.load_state_dict(weights, strict=True)
    return checkpoint_digest(model)


def _verify_recipe(training, protocol, banks, seed, config):
    recipe = training["recipe"]
    if (recipe["model"] != "bic-diversity-sequence-v1" or recipe["banks"] != banks
            or recipe["seed"] != seed or recipe["config"] != asdict(config)
            or recipe["batch_size"] != protocol["batch_size"]
            or recipe["learning_rate"] != protocol["learning_rate"] or recipe["gradient_clip"] != 1.
            or recipe["optimizer"]["name"] != "AdamW"):
        raise ValueError("checkpoint training recipe differs from the fixed study")
    groups = training["optimizer"]["param_groups"]
    if len(groups) != 1 or {key: value for key, value in groups[0].items() if key != "params"} != recipe["optimizer"]["options"]:
        raise ValueError("checkpoint optimizer options differ from its recipe")


def verify_completed(directory):
    """Validate completed files and provenance, without inference or bank scoring."""
    directory = Path(directory)
    names = (*study.ARMS, "fresh")
    required = [directory / "protocol.json", directory / "audit" / "report.json",
                directory / "audit" / "evaluation-started.json"]
    required += [directory / "main" / arm / name for arm in study.ARMS
                 for name in ("report.json", "latest.pt", "initial.pt")]
    required += [directory / "audit" / filename for name in names
                 for filename in (f"{name}.json", f"{name}-adapted.pt")]
    if any(not path.is_file() for path in required):
        raise ValueError("summary requires all four completed main runs and all five completed audits")
    # This gate precedes any study-result reading or checkpoint deserialization.
    hashes = {path.relative_to(directory).as_posix(): study.file_hash(path) for path in required}
    protocol = study.load_protocol(directory)
    audit = _read(directory / "audit" / "report.json")
    parents = {arm: hashes[f"main/{arm}/latest.pt"] for arm in study.ARMS}
    marker = {"schema": study.SCHEMA, "parent_sha256": parents,
              "protocol_sha256": hashes["protocol.json"]}
    if (_read(directory / "audit" / "evaluation-started.json") != marker
            or audit.get("schema") != study.SCHEMA or audit.get("inputs") != marker
            or set(audit.get("results", {})) != set(names)
            or audit.get("base_checkpoints_unchanged") is not True
            or audit.get("automatic_promotion") is not False):
        raise ValueError("completed audit provenance differs")
    main, family_exposures = {}, {}
    verifier = build_sequence_student(protocol["seed"])
    if checkpoint_digest(verifier) != protocol["initial_weights_sha256"]:
        raise ValueError("fresh initialization differs from the frozen protocol")
    for arm in study.ARMS:
        path = directory / "main" / arm
        row = _read(path / "report.json")
        saved = torch.load(path / "latest.pt", map_location="cpu", weights_only=True)
        _check_finite_tree(saved, "main checkpoint")
        study.validate_resume(saved, protocol, arm)
        _verify_recipe(saved["training"], protocol, protocol["banks"]["train"][arm],
                       protocol["seed"], verifier.config)
        if (row.get("schema") != study.SCHEMA or row.get("protocol") != protocol or row.get("arm") != arm
                or row.get("updates") != protocol["steps"] or saved["training"]["updates"] != protocol["steps"]
                or row.get("episodes") != protocol["steps"] * protocol["batch_size"]
                or row.get("checkpoint_file_sha256") != parents[arm]
                or row.get("heldout_evaluation_performed") is not False
                or row.get("training_seconds") != saved["training_seconds"]
                or saved["training"]["recipe"]["banks"] != protocol["banks"]["train"][arm]
                or _weight_digest(verifier, saved["training"]["weights"]) != row.get("weights_sha256")):
            raise ValueError("completed main endpoint differs from its report or protocol")
        initial = torch.load(path / "initial.pt", map_location="cpu", weights_only=True)
        if _weight_digest(verifier, initial["weights"]) != protocol["initial_weights_sha256"]:
            raise ValueError("saved initial weights differ")
        family_exposures[arm] = {family: {"updates": count, "episode_exposures": count * protocol["batch_size"]}
                                for family, count in saved["training"]["family_updates"].items()}
        main[arm] = row
    for name, row in audit["results"].items():
        if row != _read(directory / "audit" / f"{name}.json") or row.get("name") != name or row.get("inputs") != marker:
            raise ValueError("individual audit result differs from completed aggregate")
        expected_initial = protocol["initial_weights_sha256"] if name == "fresh" else main[name]["weights_sha256"]
        budgets = protocol["audit_budgets"]
        if (row["initial_weights_sha256"] != expected_initial
                or [point["updates"] for point in row["curve"]] != budgets
                or any(point["exposures"] != point["updates"] * protocol["batch_size"] for point in row["curve"])
                or set(row["controls"]) != {"blank_text", "reset_history"}):
            raise ValueError("audit initialization, adaptation budgets or controls differ")
        areas = curve_areas(row["curve"])
        if set(row["areas"]) != set(AREA_KEYS) or any(
                not math.isclose(row["areas"][key], value, rel_tol=0, abs_tol=1e-12) for key, value in areas.items()):
            raise ValueError("reported adaptation areas disagree with recorded curve")
        adapted = torch.load(directory / "audit" / f"{name}-adapted.pt", map_location="cpu", weights_only=True)
        _check_finite_tree(adapted, "adapted checkpoint")
        _verify_recipe(adapted, protocol, protocol["banks"]["support"], protocol["audit_seed"], verifier.config)
        if (adapted["updates"] != budgets[-1] or adapted["family_updates"] != {protocol["heldout_family"]: budgets[-1]}
                or adapted["recipe"]["banks"] != protocol["banks"]["support"]
                or adapted["recipe"]["seed"] != protocol["audit_seed"]
                or adapted["recipe"]["batch_size"] != protocol["batch_size"]
                or _weight_digest(verifier, adapted["weights"]) != row["final_weights_sha256"]
                or row["cpu_restart"].get("exact") is not True
                or row["cpu_restart"].get("checkpoint_sha256") != row["final_weights_sha256"]):
            raise ValueError("adapted checkpoint or CPU restart provenance differs")
        generator = torch.Generator().manual_seed(protocol["audit_seed"])
        for _ in range(budgets[-1]):
            torch.randint(protocol["support_episodes"] // 2, (protocol["batch_size"] // 2,), generator=generator)
        if set(adapted["samplers"]) != {protocol["heldout_family"]} or not torch.equal(
                adapted["samplers"][protocol["heldout_family"]], generator.get_state()):
            raise ValueError("adaptation exposure stream differs")
    summary = assemble_summary(protocol, main, audit)
    expected_gains = summary["prior_learning_area_gains_vs_fresh"]
    if audit["prior_learning_area_gains"] != expected_gains:
        raise ValueError("reported prior-learning gains differ from recorded curves")
    summary["main_family_exposures"] = family_exposures
    summary["verification"] = {"input_file_sha256": hashes,
        "source_sha256": protocol["source_sha256"], "banks_file_sha256": protocol["banks_file_sha256"],
        "summary_builder_sha256": study.file_hash(__file__),
        "no_inference_or_training_performed": True}
    # Catch changes during verification/calculation before permitting a write.
    study.load_protocol(directory)
    if any(study.file_hash(directory / name) != sha for name, sha in hashes.items()):
        raise RuntimeError("study files changed while building summary")
    _check_finite_tree(summary, "summary")
    return summary


def summarize(directory):
    directory = Path(directory)
    if not (directory / "audit" / "report.json").is_file():
        raise ValueError("completed audit report required before summarizing")
    with run_lock(directory / "audit"):
        summary = verify_completed(directory)
        atomic_json(directory / "audit" / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    result = summarize(parser.parse_args().study)
    print(json.dumps({"compute": result["compute"],
        "prior_learning_area_gains_vs_fresh": result["prior_learning_area_gains_vs_fresh"]}, indent=2))


if __name__ == "__main__":
    main()
