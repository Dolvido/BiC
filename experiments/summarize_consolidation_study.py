"""Verified descriptive consolidation summary after both extensions and all audits.

This module performs no neural inference, optimization or checkpoint selection.
Replay comparisons match new exposure while counting their extra old data and
compute. Previous audit evidence informed the design; no significance claim is
made from this single-seed, single-held-subject study.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_loop import run_lock
from brain_in_computer.learning_student import _check_finite_tree
from experiments import consolidation_study as study
from experiments import summarize_diversity_study as previous_summary
from experiments.consolidation_banks import bank_manifest
from experiments.consolidation_evaluation import _bank_storage
from experiments.consolidation_evaluation import SCHEMA as EVALUATION_SCHEMA
from experiments.consolidation_training import ConsolidationTrainer, REPLAY_SEED_OFFSET
from experiments.diversity_training import DiversityTrainer
from experiments.sequence_student import build_sequence_student
from experiments.train_cognitive import atomic_json, fingerprint_rows


AREA_KEYS = previous_summary.AREA_KEYS
COUNTS = ("episodes", "observation_tokens", "observation_bytes", "reply_target_tokens", "reply_target_bytes")
QUERY_PANELS = ("names", "worlds", "both", "advanced")
PARENTS = tuple(f"{arm}-{stage}" for arm in study.ARMS for stage in ("short", "long"))
JOBS = tuple(f"{parent}-{mode}" for parent in PARENTS for mode in ("ordinary", "replay")) + ("fresh-ordinary",)


def compact(metrics):
    result = previous_summary.compact(metrics)
    for name, row in metrics["per_bank"].items():
        result["per_bank"][name].update(ask_correct=row["ask_correct"],
            per_target=copy.deepcopy(row["per_target"]),
            confusion_matrix_true_rows_predicted_columns=copy.deepcopy(row["confusion_matrix_true_rows_predicted_columns"]),
            query_reply_denominator=row["query_total"], action_reply_agreement_denominator=row["query_total"])
    return result


def panel_areas(curve):
    result = {}
    for panel in QUERY_PANELS:
        points = []
        for point in curve:
            row = point["per_bank"][panel]
            points.append({"updates": point["updates"], "macro_query_accuracy": row["query_accuracy"],
                "macro_pair_accuracy": row["counterfactual_accuracy"],
                "macro_later_known_accuracy": row["positions"]["last_known_later_query"]["accuracy"]})
        result[panel] = previous_summary.curve_areas(points)
    return result


def _minus(after, before):
    return None if after is None or before is None else after - before


def metric_change(after, before):
    return {"macro": {key: _minus(after[key], before[key]) for key in AREA_KEYS},
            "per_bank": previous_summary.metric_change(after["per_bank"], before["per_bank"])}


def _comparison(after, before):
    for key in (f"support_{name}" for name in COUNTS):
        if after["exposures"][key] != before["exposures"][key]:
            raise ValueError("comparison requires exactly matched new episode and byte exposure")
    return {"matched_new_exposure": True,
        "query_area_difference": {key: after["areas"][key] - before["areas"][key] for key in AREA_KEYS},
        "query_panel_area_difference": {panel: {key: after["panel_areas"][panel][key]
            - before["panel_areas"][panel][key] for key in AREA_KEYS} for panel in QUERY_PANELS},
        "query_endpoint_difference": metric_change(after["curve"][-1], before["curve"][-1]),
        "old_retention_endpoint_difference": metric_change(after["retention_after"], before["retention_after"]),
        "retention_change_difference": {"macro": {key: _minus(after["retention_change"]["macro"][key],
            before["retention_change"]["macro"][key]) for key in AREA_KEYS},
            "per_bank": previous_summary.metric_change(after["retention_change"]["per_bank"],
                                                       before["retention_change"]["per_bank"])},
        "extra_exposure": {key: after["exposures"][key] - before["exposures"][key] for key in after["exposures"]},
        "training_seconds_difference": after["training_seconds"] - before["training_seconds"]}


def assemble_summary(protocol, extensions, audit):
    """Pure descriptive calculations, separated from provenance verification."""
    if set(extensions) != set(study.ARMS) or set(audit["results"]) != set(JOBS):
        raise ValueError("summary requires two completed extensions and nine audits")
    result = {"schema": "bic-consolidation-study-summary-v1", "automatic_promotion": False,
        "interpretation": protocol["interpretation"],
        "bank_denominators": copy.deepcopy(protocol["banks"]),
        "comparison_note": "Replay adds old examples and computation at matched new-example exposure and optimizer updates. "
            "A replay gain is not intrinsic transfer improvement or a compute-matched advantage. Longer pretraining also "
            "adds optimization/exposure. Ordinary pretrained-versus-fresh comparisons are reported separately.",
        "evidence_note": "Single main seed, one held-out subject, descriptive differences only. Earlier audit evidence "
            "informed this design; fresh rows do not erase that design adaptation. No automatic promotion.",
        "reply_note": "Exact reply accuracy, parseability and action agreement can reveal degradation; recorded scores "
            "cannot establish constant-output collapse because generated strings were not retained. Actions and free-running "
            "replies use distinct readouts: agreement is not correctness, and parseability is not answer accuracy.",
        "aggregation_note": "Query macros weight names/worlds/both/advanced panels equally, not by query count. Old-task "
            "macros weight named banks equally. Target mixtures and eligible pair denominators differ across panels; "
            "compare like panels and retain per-target scores. Naming-map counts are actual full-bank counts in bank_denominators.",
        "area_note": "AUC is a normalized linear-update trapezoid over the fixed sparse budgets, not a densely observed "
            "learning curve or an endpoint score. Area differences combine initial performance and subsequent adaptation; "
            "they do not isolate learning speed. Endpoint known-later pair scores are reported separately.",
        "missing_value_note": "Null means an absent measured denominator or an unmeasured quantity, including fit replies; it is not zero.",
        "timing_note": "Summed concurrent extension worker intervals are not dedicated GPU-hours. Adaptation records "
            "synchronized complete update intervals including both losses; wall includes evaluation/restart/setup. "
            "Extension invocation seconds start after setup and exclude final report scoring and writing. "
            "Replay-component GPU time was not measured separately.",
        "extension": {}, "adaptation": {}, "base_checkpoints_unchanged": audit["base_checkpoints_unchanged"]}
    for arm, row in extensions.items():
        curve = [{"updates": point["updates"],
            "additional_updates": point["updates"] - protocol["parent_updates"],
            "training_seconds": point["training_seconds"], "development": compact(point["development"]),
            **({"common_fit": compact(point["common_fit"])} if "common_fit" in point else {})}
            for point in row["history"]]
        result["extension"][arm] = {"updates": row["updates"], "additional_updates": row["additional_updates"],
            "additional_episode_exposures": row["additional_episode_exposures"],
            "training_seconds": row["training_seconds"], "invocation_seconds_after_setup": row["invocation_seconds_after_setup"],
            "peak_cuda_allocated_mib": row["peak_cuda_allocated_mib"], "peak_scope": row["peak_scope"],
            "curve": curve, "development": compact(row["development"]), "common_fit": compact(row["common_fit"]),
            "development_change": metric_change(compact(row["development"]), curve[0]["development"]),
            "common_fit_change": metric_change(compact(row["common_fit"]), curve[0]["common_fit"]),
            "fit_note": "Same familiar first 32 worlds/first naming map in both arms; not the full training bank. Fit replies unscored."}
    for name, row in audit["results"].items():
        before, after = compact(row["retention_before"]), compact(row["retention_after"])
        points = [{"updates": point["updates"], "optimizer_updates": point["optimizer_updates"],
                   "exposures": copy.deepcopy(point["exposures"]), **compact(point)} for point in row["curve"]]
        result["adaptation"][name] = {"parent": row["parent"], "mode": row["mode"],
            "optimizer_updates": row["optimizer_updates"], "curve": points,
            "query_macro_weighting": row["query_macro_weighting"],
            "areas": previous_summary.curve_areas(row["curve"]), "panel_areas": panel_areas(row["curve"]),
            "endpoint_later_known_pairs": {panel: scores["later_known"]["pairs"]
                for panel, scores in points[-1]["per_bank"].items()},
            "development_before": compact(row["development_before"]), "development_after": compact(row["development_after"]),
            "retention_before": before, "retention_after": after, "retention_change": metric_change(after, before),
            "advanced_before": compact(row["advanced_before"]), "advanced_after": compact(row["advanced_after"]),
            "controls_at_final_budget": {key: compact(value) for key, value in row["controls_at_256"].items()},
            "cpu_restart": copy.deepcopy(row["cpu_restart"]), "exposures": copy.deepcopy(row["exposures"]),
            "support_storage": copy.deepcopy(row["support_storage"]), "replay_storage": copy.deepcopy(row["replay_storage"]),
            "training_seconds": row["training_seconds"], "wall_seconds": row["wall_seconds"], "time_scope": row["time_scope"],
            "initial_weights_sha256": row["initial_weights_sha256"], "final_weights_sha256": row["final_weights_sha256"]}
    values = result["adaptation"]
    result["replay_minus_ordinary"] = {parent: _comparison(values[f"{parent}-replay"], values[f"{parent}-ordinary"])
                                       for parent in PARENTS}
    result["long_minus_short_at_fixed_replay_condition"] = {arm: {mode:
        _comparison(values[f"{arm}-long-{mode}"], values[f"{arm}-short-{mode}"])
        for mode in ("ordinary", "replay")} for arm in study.ARMS}
    result["ordinary_pretrained_minus_common_fresh"] = {parent:
        _comparison(values[f"{parent}-ordinary"], values["fresh-ordinary"]) for parent in PARENTS}
    result["compute"] = {"extension_optimizer_updates": sum(row["additional_updates"] for row in extensions.values()),
        "extension_episode_exposures": sum(row["additional_episode_exposures"] for row in extensions.values()),
        "extension_worker_training_seconds": sum(row["training_seconds"] for row in extensions.values()),
        "adaptation_optimizer_updates": sum(row["optimizer_updates"] for row in values.values()),
        "adaptation_training_seconds": sum(row["training_seconds"] for row in values.values()),
        "adaptation_exposures": {key: sum(row["exposures"][key] for row in values.values())
                                 for key in values["fresh-ordinary"]["exposures"]}}
    return result


def sampled_counts(banks, seed, batch_size, family_updates, *, skip_updates=None):
    """Reconstruct exact UTF-8 exposure from the verified CPU sampler streams.

    Pure counts on frozen text, not language-model inference. Whole-turn BOS/EOS
    add12 observation tokens per episode; shifted replies add6 EOS targets.
    ``skip_updates`` excludes the earlier parent phase from extension totals.
    """
    if set(banks) != set(family_updates) or (skip_updates is not None and set(skip_updates) != set(banks)):
        raise ValueError("sampler bank/counter sets differ")
    totals, states = dict.fromkeys(COUNTS, 0), {}
    for index, family in enumerate(sorted(banks)):
        count = family_updates[family]
        skip = 0 if skip_updates is None else skip_updates[family]
        if type(count) is not int or type(skip) is not int or not 0 <= skip <= count:
            raise ValueError("invalid sampler reconstruction counts")
        rows = banks[family]
        if not rows or len(rows) % 2 or type(batch_size) is not int or batch_size < 2 or batch_size % 2:
            raise ValueError("sampling requires complete pairs")
        per_pair = torch.tensor([[sum(len(turn[field].encode("utf8")) for row in rows[start:start + 2]
            for turn in row["turns"]) for field in ("text", "reply")] for start in range(0, len(rows), 2)], dtype=torch.long)
        hits = torch.zeros(len(rows) // 2, dtype=torch.long)
        generator = torch.Generator().manual_seed(seed + index * 7919)
        ones = torch.ones(batch_size // 2, dtype=torch.long)
        for update in range(count):
            pairs = torch.randint(len(rows) // 2, (batch_size // 2,), generator=generator)
            if update >= skip:
                hits.scatter_add_(0, pairs, ones)
        observation_bytes, reply_bytes = (hits[:, None] * per_pair).sum(0).tolist()
        episodes = (count - skip) * batch_size
        row = {"episodes": episodes, "observation_bytes": observation_bytes, "reply_target_bytes": reply_bytes,
               "observation_tokens": observation_bytes + episodes * 12, "reply_target_tokens": reply_bytes + episodes * 6}
        for key in COUNTS:
            totals[key] += row[key]
        states[family] = generator.get_state()
    return totals, states


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _json_equal(left, right):
    # torch checkpoints retain tuple-valued AdamW betas; JSON represents them
    # as lists. Compare their complete canonical JSON content, not containers.
    return study.prior.digest(left) == study.prior.digest(right)


def _samplers_equal(actual, expected):
    return set(actual) == set(expected) and all(torch.equal(actual[name], value) for name, value in expected.items())


def _counts_at(protocol, banks, replay, updates):
    support_counts, support_states = sampled_counts(banks["support"], protocol["adaptation_seed"], protocol["batch_size"],
                                                   {name: updates for name in banks["support"]})
    names = sorted(replay or {})
    replay_updates = {name: updates // len(names) + int(index < updates % len(names)) for index, name in enumerate(names)}
    replay_counts, replay_states = sampled_counts(replay or {}, protocol["adaptation_seed"] + REPLAY_SEED_OFFSET,
                                                  protocol["replay_batch_size"], replay_updates)
    return {**{f"support_{key}": value for key, value in support_counts.items()},
            **{f"replay_{key}": value for key, value in replay_counts.items()}}, support_states, replay_states


def verify_completed(directory):
    directory = Path(directory)
    milestones = list(range(study.PARENT_UPDATES * 2, study.TOTAL_UPDATES + 1, study.PARENT_UPDATES))
    required = [directory / filename for filename in ("protocol.json", "banks.pt", "replay.pt")]
    required += [directory / "audit" / filename for filename in ("report.json", "evaluation-started.json")]
    required += [directory / "extension" / arm / filename for arm in study.ARMS
                 for filename in ("latest.pt", "report.json", *(f"checkpoint-{step:06d}.pt" for step in milestones))]
    required += [directory / "audit" / filename for name in JOBS for filename in (f"{name}.json", f"{name}-adapted.pt")]
    if any(not path.is_file() for path in required):
        raise ValueError("summary requires both completed extensions, their milestones and all nine completed audits")
    # No result JSON or checkpoint is read before the all-files completion gate.
    hashes = {path.relative_to(directory).as_posix(): study.prior.file_hash(path) for path in required}
    protocol = study.load_protocol(directory)
    if protocol["extension_checkpoints"] != milestones:
        raise ValueError("extension checkpoint contract differs")
    if study.parent_evidence(protocol["parent"]["directory"]) != protocol["parent"]:
        raise ValueError("original parent evidence differs")
    candidates, endpoint_hashes = study.verify_endpoints(directory, protocol)
    marker = {"schema": study.SCHEMA, "protocol_sha256": hashes["protocol.json"], "endpoint_files_sha256": endpoint_hashes}
    audit = _read(directory / "audit" / "report.json")
    if (_read(directory / "audit" / "evaluation-started.json") != marker or audit.get("inputs") != marker
            or audit.get("protocol") != protocol or audit.get("schema") != study.SCHEMA
            or set(audit.get("results", {})) != set(JOBS) or audit.get("base_checkpoints_unchanged") is not True
            or audit.get("automatic_promotion") is not False):
        raise ValueError("completed audit provenance differs")
    banks = torch.load(directory / "banks.pt", map_location="cpu", weights_only=True)
    replay = torch.load(directory / "replay.pt", map_location="cpu", weights_only=True)
    if bank_manifest(banks) != protocol["banks"] or {arm: {family: {"episodes": len(rows), "sha256": fingerprint_rows(rows)}
            for family, rows in groups.items()} for arm, groups in replay.items()} != protocol["replay_banks"]:
        raise ValueError("frozen bank manifests differ")
    _, original_banks = study.prior.read_banks(protocol["parent"]["directory"])
    model = build_sequence_student(protocol["seed"])
    candidate_digests = {"fresh": checkpoint_digest(model)}
    for name, weights in candidates.items():
        model.load_state_dict(weights, strict=True)
        candidate_digests[name] = checkpoint_digest(model)
    extensions, extension_counts, milestone_digests = {}, {}, {}
    for arm in study.ARMS:
        path = directory / "extension" / arm
        report = _read(path / "report.json")
        saved = torch.load(path / "latest.pt", map_location="cpu", weights_only=True)
        study.validate_extension(saved, protocol, arm)
        trainer = DiversityTrainer(original_banks["train"][arm], seed=protocol["seed"],
            batch_size=protocol["batch_size"], learning_rate=protocol["learning_rate"], payload=saved["training"])
        if (report.get("schema") != study.SCHEMA or report["protocol"] != protocol or report["arm"] != arm
                or report["additional_updates"] != protocol["additional_updates_per_arm"]
                or report["additional_episode_exposures"] != protocol["additional_updates_per_arm"] * protocol["batch_size"]
                or report["history"] != saved["history"] or report["training_seconds"] != saved["training_seconds"]
                or checkpoint_digest(trainer.model) != report["weights_sha256"]
                or report["history"][0]["updates"] != protocol["parent_updates"]
                or report["history"][-1]["updates"] != protocol["total_updates"]):
            raise ValueError("extension report differs from its final checkpoint")
        original = torch.load(Path(protocol["parent"]["directory"]) / "main" / arm / "latest.pt", map_location="cpu", weights_only=True)
        trainer._restore(original["training"])
        counts, states = sampled_counts(original_banks["train"][arm], protocol["seed"], protocol["batch_size"],
            saved["training"]["family_updates"], skip_updates=original["training"]["family_updates"])
        if not _samplers_equal(saved["training"]["samplers"], states):
            raise ValueError("extension reconstructed exposure stream differs")
        extension_counts[arm], milestone_digests[arm] = counts, {}
        for step in milestones:
            checkpoint = torch.load(path / f"checkpoint-{step:06d}.pt", map_location="cpu", weights_only=True)
            _check_finite_tree(checkpoint, "extension milestone")
            study.validate_extension(checkpoint, protocol, arm)
            if checkpoint["training"]["updates"] != step or checkpoint["history"] != [row for row in report["history"] if row["updates"] <= step]:
                raise ValueError("extension milestone count/history differs")
            trainer._restore(checkpoint["training"])
            milestone_digests[arm][str(step)] = checkpoint_digest(trainer.model)
        if milestone_digests[arm][str(protocol["total_updates"])] != report["weights_sha256"]:
            raise ValueError("final milestone differs from completed endpoint")
        extensions[arm] = report
        del trainer
    count_cache = {}
    for name, row in audit["results"].items():
        parent, mode = name.rsplit("-", 1)
        old = replay[parent.rsplit("-", 1)[0]] if mode == "replay" else None
        if (row != _read(directory / "audit" / f"{name}.json") or row.get("schema") != EVALUATION_SCHEMA
                or row.get("inputs") != marker or row.get("name") != name
                or row.get("parent") != parent or row.get("mode") != mode or row.get("condition") != mode
                or row.get("adapted_file_sha256") != hashes[f"audit/{name}-adapted.pt"]
                or row["initial_weights_sha256"] != candidate_digests[parent]
                or row["optimizer_reset"] is not True or row["parent_weights_unchanged"] is not True
                or row["automatic_promotion"] is not False or row["seed"] != protocol["adaptation_seed"]
                or row["budgets"] != protocol["adaptation_budgets"] or row["optimizer_updates"] != protocol["adaptation_budgets"][-1]
                or row["new_batch_size"] != protocol["batch_size"] or row["learning_rate"] != protocol["learning_rate"]
                or row["replay_batch_size"] != (protocol["replay_batch_size"] if old else 0)
                or row["replay_weight"] != (protocol["replay_weight"] if old else 0)
                or row["controls_at_updates"] != protocol["adaptation_budgets"][-1]
                or set(row["controls_at_256"]) != {"blank_text", "reset_history"}):
            raise ValueError("adaptation report identity or fixed budget differs")
        saved = torch.load(directory / "audit" / f"{name}-adapted.pt", map_location="cpu", weights_only=True)
        trainer = ConsolidationTrainer(banks["support"], replay_banks=old, seed=protocol["adaptation_seed"],
            batch_size=protocol["batch_size"], replay_batch_size=protocol["replay_batch_size"],
            replay_weight=protocol["replay_weight"], learning_rate=protocol["learning_rate"], payload=saved)
        if (not _json_equal(saved["recipe"], row["recipe"]) or saved["updates"] != row["optimizer_updates"]
                or saved["exposures"] != row["exposures"] or checkpoint_digest(trainer.model) != row["final_weights_sha256"]
                or row["cpu_restart"].get("exact") is not True or row["cpu_restart"].get("device") != "cpu"
                or row["cpu_restart"].get("checkpoint_sha256") != row["final_weights_sha256"]
                or row["support_storage"] != _bank_storage(banks["support"]) or row["replay_storage"] != _bank_storage(old)
                or [point["updates"] for point in row["curve"]] != protocol["adaptation_budgets"]):
            raise ValueError("adapted checkpoint, storage or restart evidence differs")
        for point in row["curve"]:
            key = (parent.rsplit("-", 1)[0] if old else None, point["updates"])
            if key not in count_cache:
                count_cache[key] = _counts_at(protocol, banks, old, point["updates"])
            counts, support_states, replay_states = count_cache[key]
            if point["exposures"] != counts or point["optimizer_updates"] != point["updates"]:
                raise ValueError("recorded adaptation byte/episode exposure differs from sampler reconstruction")
        if not _samplers_equal(saved["samplers"], support_states) or not _samplers_equal(saved["replay_samplers"], replay_states):
            raise ValueError("adaptation sampler stream differs")
        if saved["exposures"] != counts:
            raise ValueError("final adaptation exposure counts differ")
        areas = previous_summary.curve_areas(row["curve"])
        if set(row["areas"]) != set(AREA_KEYS) or any(not math.isclose(row["areas"][key], value, rel_tol=0, abs_tol=1e-12)
                                                    for key, value in areas.items()):
            raise ValueError("reported adaptation areas differ")
        del trainer
    result = assemble_summary(protocol, extensions, audit)
    result["extension_exposures"] = extension_counts
    result["compute"]["extension_sampled_counts"] = {key: sum(row[key] for row in extension_counts.values()) for key in COUNTS}
    result["extension_count_note"] = "Exact UTF-8/token exposure reconstructed from verified continuation sampler states and frozen text; parent-phase draws excluded."
    result["verification"] = {"input_file_sha256": hashes, "endpoint_files_sha256": endpoint_hashes,
        "source_sha256": protocol["source_sha256"], "verified_source_files": len(protocol["source_sha256"]),
        "parent_evidence": protocol["parent"], "milestone_weight_sha256": milestone_digests,
        "helper_sha256": {"summary": study.prior.file_hash(__file__), "compact_metrics": study.prior.file_hash(previous_summary.__file__)},
        "no_neural_inference_or_training_performed": True}
    study.load_protocol(directory)
    if any(study.prior.file_hash(directory / name) != sha for name, sha in hashes.items()) or any(
            study.prior.file_hash(path) != sha for path, sha in endpoint_hashes.items()):
        raise RuntimeError("study evidence changed while constructing summary")
    _check_finite_tree(result, "consolidation summary")
    return result


def summarize(directory):
    directory = Path(directory)
    if not (directory / "audit" / "report.json").is_file():
        raise ValueError("completed audit report required before summary")
    with run_lock(directory / "audit"):
        result = verify_completed(directory)
        atomic_json(directory / "audit" / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    torch.set_num_threads(1)
    result = summarize(parser.parse_args().study)
    print(json.dumps({"compute": result["compute"], "verified_source_files": result["verification"]["verified_source_files"]}, indent=2))


if __name__ == "__main__":
    main()
