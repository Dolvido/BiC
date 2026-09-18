"""Verified descriptive fixed/fresh summaries, after every replay and audit.

No numerical evidence is read before the complete-file gate. Recorded metrics
are checked against canonical denominators; no neural inference or optimization
is performed. The full stream has already been replayed by the sealed verifier.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_loop import run_lock
from brain_in_computer.learning_student import _check_finite_tree
from experiments import realization_study as study
from experiments import summarize_composition_study as metrics_helper
from experiments.composition_evaluation import METRICS
from experiments.realization_banks import (
    _stats, bank_manifest, historical_evidence, protected_transcripts, verify_boundaries,
)
from experiments.realization_training import COUNTS, stream_evidence
from experiments.sequence_student import build_sequence_student
from experiments.train_cognitive import atomic_json


SCHEMA = "bic-realization-summary-v1"
EXTRA_METRICS = ("ask_precision", "ask_recall", "action_reply_agreement", "query_loss", "brier_score")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _same(left, right):
    if not metrics_helper._json_equal(left, right):
        raise ValueError("summary evidence differs")


def _same_tensors(left, right):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError("endpoint optimizer/checkpoint tensor state differs")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError("endpoint checkpoint fields differ")
        for key in left:
            _same_tensors(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise ValueError("endpoint checkpoint sequence differs")
        for first, second in zip(left, right):
            _same_tensors(first, second)
    elif left != right:
        raise ValueError("endpoint checkpoint values differ")


def _validate_endpoint_binding(latest, checkpoint, replay_total):
    """Bind complete endpoint state and its actual stream to sealed evidence."""
    _same(stream_evidence(latest["realization"]), replay_total)
    _same_tensors(latest, checkpoint)


def compact(metrics):
    result = metrics_helper.compact(metrics)
    for families in result["per_panel_family"].values():
        for group in families.values():
            for key in EXTRA_METRICS:
                values = [row[key] for row in group["per_length"].values()]
                group["macro"][key] = sum(values) / len(values) if all(value is not None for value in values) else None
    return result


def _curve(curve):
    return [{**point, "held_episode_exposure": point["episodes_per_family"]} for point in curve]


def _metric_values(metrics):
    return {panel: {family: {"macro": group["macro"], "per_length": {
        length: {key: row[key] for key in (*METRICS, *EXTRA_METRICS)}
        for length, row in group["per_length"].items()}} for family, group in families.items()}
        for panel, families in compact(metrics)["per_panel_family"].items()}


def assemble_summary(protocol, main, audit, timings):
    if set(main) != set(study.JOBS) or set(audit["results"]) != set(study.JOBS) or set(timings) != set(study.JOBS):
        raise ValueError("both complete candidates required")
    fixed, fresh = main["fixed"]["stream"], main["fresh"]["stream"]
    for field in ("sampler_sha256", "occurrences", "family_microbatches", "bucket_microbatches", "structural_stream_sha256"):
        _same(fixed[field], fresh[field])
    for family in study.FAMILIES:
        for field in ("episodes", "turns"):
            if fixed["exposures"][family][field] != fresh["exposures"][family][field]:
                raise ValueError("matched structural exposure differs")
    results = audit["results"]
    if [point["episodes_per_family"] for point in results["fixed"]["curve"]] != [point["episodes_per_family"] for point in results["fresh"]["curve"]]:
        raise ValueError("matched curve exposure differs")
    summary = {"schema": SCHEMA, "study_schema": protocol["schema"], "automatic_promotion": False,
        "interpretation": protocol["interpretation"],
        "comparison_note": "One initialization; fresh minus fixed is a combined name/value diversity intervention. Procedures, schedule, optimizer updates and episode exposures match. Targets, increments, byte lengths, padding and generation costs may differ. Marginal name/value panels do not establish an additive mechanism.",
        "area_note": "Sparse trapezoidal areas normalized by final per-family episode exposure, with equal length weighting. These checkpoint summaries combine initial performance and subsequent learning; they do not isolate learning speed.",
        "fit_note": "Initial fit scores the common original realization bank; latest fit scores the most recently consumed realization per observed recipe. These are occurrence-selected fitting probes, not the complete fresh stream or unbiased generalization. Coverage and missing recipes remain explicit.",
        "metric_note": "Preserve panel, family, length, known/ASK denominators, earlier eligible pairs, final pairs and free replies. Final queries are always known/opposite. Missing metrics stay null. Historically inspected syntactic motifs and aliases are not new semantic algorithms or vocabulary.",
        "time_note": "Synchronized durable step seconds already include generation, validation, packing and optimization. Generation/validation and rejected-candidate times are nested subsets, not extra costs. Summed concurrent worker intervals are not elapsed wall time or isolated GPU time. Invocation/peak values cover their measured invocation; discarded uncheckpointed work would not be recovered by these counters.",
        "main": {}, "audit": {}, "fresh_minus_fixed": {}}
    for job, report in main.items():
        summary["main"][job] = {key: copy.deepcopy(report[key]) for key in
            ("updates", "exposures", "training_step_seconds", "step_time_scope", "invocation_seconds_after_preparation",
             "peak_cuda_allocated_mib", "weights_sha256", "stream")}
        summary["main"][job]["generation_timing_subset"] = copy.deepcopy(timings[job])
        summary["main"][job]["development_curve"] = [{"updates": point["updates"],
            "episodes_per_family": point["updates"] * study.MICRO,
            "training_step_seconds": point["training_step_seconds"], "metrics": compact(point["development"])}
            for point in report["history"]]
    for job, result in results.items():
        summary["audit"][job] = {"curve": [{**{key: copy.deepcopy(point[key]) for key in
            ("updates", "episodes_per_family", "weights_sha256")}, "metrics": compact(point["metrics"])}
            for point in result["curve"]],
            "areas_by_family": {family: metrics_helper.curve_areas(_curve(result["curve"]), family)
                                for family in study.FAMILIES},
            "final": compact(result["final"]), "initial_realization_fit": compact(result["initial_realization_fit"]),
            "latest_observed_fit": compact(result["latest_observed_fit"]),
            "latest_observed_metadata": copy.deepcopy(result["latest_observed_metadata"]),
            "controls": {control: compact(scores) for control, scores in result["controls"].items()},
            "cpu_restart": copy.deepcopy(result["cpu_restart"]), "seconds": result["seconds"]}
    summary["fresh_minus_fixed"] = {
        "endpoint_difference": metrics_helper._difference(_metric_values(results["fresh"]["final"]), _metric_values(results["fixed"]["final"])),
        "area_difference_by_family": metrics_helper._difference(summary["audit"]["fresh"]["areas_by_family"], summary["audit"]["fixed"]["areas_by_family"]),
        "actual_exposure_difference": {family: {key: fresh["exposures"][family][key] - fixed["exposures"][family][key] for key in COUNTS}
                                       for family in study.FAMILIES},
        "step_seconds_difference": main["fresh"]["training_step_seconds"] - main["fixed"]["training_step_seconds"]}
    summary["compute"] = {"retained_optimizer_updates": sum(row["updates"] for row in main.values()),
        "sampled_exposures": {key: sum(counts[key] for row in main.values() for counts in row["exposures"].values()) for key in COUNTS},
        "worker_training_step_seconds": sum(row["training_step_seconds"] for row in main.values()),
        "generation_validation_seconds_already_in_steps": sum(row["generation_validation_seconds"] for row in timings.values()),
        "rejected_candidate_seconds_already_in_generation": sum(row["rejected_candidate_seconds"] for row in timings.values()),
        "worker_invocation_seconds_after_preparation": sum(row["invocation_seconds_after_preparation"] for row in main.values()),
        "audit_worker_seconds": sum(row["seconds"] for row in results.values()),
        "scope": "retained study checkpoint lineage; separate disposable execution probe excluded"}
    return summary


def _required(directory):
    return [directory / name for name in ("protocol.json", "banks.pt", "protected.pt")] + study._required(directory) + [
        directory / "verification" / f"{job}.json" for job in study.JOBS] + [directory / "audit" / name for name in
            ("evaluation-started.json", "report.json", *(f"{job}.json" for job in study.JOBS))]


def _validate_scores(scores, rows, manifests, *, role, control="normal", replies=True):
    metrics_helper._validate_metrics(scores, rows, manifests, role=role, control=control, replies=replies,
                                    pair_denominators=metrics_helper._opposite_pair_denominators(rows))


def verify_completed(directory):
    directory = Path(directory)
    required = _required(directory)
    if any(not path.is_file() for path in required):
        raise ValueError("summary requires both endpoints, all checkpoints, both complete replays and both completed audits")
    hashes = {path.relative_to(directory).as_posix(): study.file_hash(path) for path in required}
    helper_hashes = {"summary": study.file_hash(__file__), "metric_validator": study.file_hash(metrics_helper.__file__)}
    protocol = study.load_protocol(directory)
    files = {name: sha for name, sha in hashes.items() if name.startswith("main/")}
    marker = {"schema": study.SCHEMA, "protocol_sha256": hashes["protocol.json"], "input_files_sha256": files,
        "verification_files_sha256": {job: hashes[f"verification/{job}.json"] for job in study.JOBS}}
    audit = _read(directory / "audit" / "report.json")
    if (audit.get("schema") != study.SCHEMA or audit.get("protocol") != protocol or audit.get("inputs") != marker
            or audit.get("base_checkpoints_unchanged") is not True or audit.get("automatic_promotion") is not False
            or set(audit.get("results", {})) != set(study.JOBS)
            or _read(directory / "audit" / "evaluation-started.json") != marker):
        raise ValueError("completed audit provenance differs")
    banks, protected = study._read_banks(directory)
    manifests = bank_manifest(banks)
    if (manifests != protocol["bank_manifest"] or verify_boundaries(banks) != protocol["bank_diagnostics"]["boundaries"]
            or historical_evidence() != protocol["bank_diagnostics"]["historical_evidence"]
            or protected != protected_transcripts(banks) or len(protected) != protocol["protected_transcripts"]):
        raise ValueError("canonical banks, historical exclusions or protected transcripts differ")
    _same(study.structural_evidence(banks["train"], study.TOTAL), protocol["expected_final_structural_streams"])
    initial_digest = checkpoint_digest(build_sequence_student(study.SEED, config=study.CONFIG))
    if initial_digest != protocol["initial_weights_sha256"]:
        raise ValueError("initial tensor digest differs")
    initial_rows = {f"initial/{family}/t{turns}": rows for family, buckets in banks["train"].items() for turns, rows in buckets.items()}
    initial_manifest = {name: _stats(rows) for name, rows in initial_rows.items()}
    main, timings, verified_digests = {}, {}, {}
    for job in study.JOBS:
        replay = _read(directory / "verification" / f"{job}.json")
        if (replay.get("schema") != study.SCHEMA or replay.get("job") != job or replay.get("input_files_sha256") != files
                or replay.get("protocol_sha256") != hashes["protocol.json"]
                or replay.get("neural_training_or_audit_performed") is not False
                or set(replay["replayed_streams"]) != {str(step) for step in study.STEPS}
                or set(replay["weights_sha256"]) != {str(step) for step in study.STEPS}):
            raise ValueError("sealed stream replay provenance differs")
        folder = directory / "main" / job
        report = _read(folder / "report.json")
        final = torch.load(folder / "latest.pt", map_location="cpu", weights_only=True)
        study.validate_snapshot(final, protocol, job, banks)
        trainer = study._trainer(banks, protected, job, payload=final["training"])
        if (trainer.updates != study.TOTAL or report["updates"] != study.TOTAL or report["job"] != job
                or report["schema"] != study.SCHEMA or report["protocol"] != protocol
                or report["heldout_evaluation_performed"] is not False
                or report["checkpoint_file_sha256"] != hashes[f"main/{job}/latest.pt"]
                or report["weights_sha256"] != checkpoint_digest(trainer.model)
                or report["exposures"] != final["training"]["exposures"]
                or report["training_step_seconds"] != final["training_step_seconds"]):
            raise ValueError("completed endpoint report differs")
        _same(report["stream"], trainer.stream_evidence())
        _same(trainer.stream_evidence(), replay["replayed_streams"][str(study.TOTAL)])
        _same(report["history"], final["history"])
        if [point["updates"] for point in report["history"]] != list(study.STEPS):
            raise ValueError("development history schedule differs")
        timings[job] = copy.deepcopy(final["training"]["realization"]["timing"])
        if not (0 <= timings[job]["rejected_candidate_seconds"] <= timings[job]["generation_validation_seconds"] <= report["training_step_seconds"]):
            raise ValueError("generation timing is not a subset of durable step time")
        verified_digests[job] = {}
        for step in study.STEPS:
            saved = torch.load(folder / f"checkpoint-{step:06d}.pt", map_location="cpu", weights_only=True)
            study.validate_snapshot(saved, protocol, job, banks)
            trainer._restore(saved["training"])
            if step == study.TOTAL:
                _validate_endpoint_binding(final["training"], saved["training"], replay["replayed_streams"][str(study.TOTAL)])
            if trainer.updates != step or saved["training_step_seconds"] != saved["history"][-1]["training_step_seconds"]:
                raise ValueError("checkpoint update/timing differs")
            _same(saved["history"], [point for point in report["history"] if point["updates"] <= step])
            _same(trainer.stream_evidence(), replay["replayed_streams"][str(step)])
            _same(saved["history"][-1]["stream"], replay["replayed_streams"][str(step)])
            digest = checkpoint_digest(trainer.model)
            if digest != replay["weights_sha256"][str(step)]:
                raise ValueError("verified replay checkpoint tensor digest differs")
            verified_digests[job][str(step)] = digest
        if verified_digests[job]["0"] != initial_digest or verified_digests[job][str(study.TOTAL)] != report["weights_sha256"]:
            raise ValueError("initial/final fixed checkpoint differs")
        for point in report["history"]:
            _validate_scores(point["development"], banks["development"], manifests["development"], role="dev", replies=False)
        result = audit["results"][job]
        if (result != _read(directory / "audit" / f"{job}.json") or result.get("schema") != study.SCHEMA
                or result["job"] != job or result["inputs"] != marker or result["automatic_promotion"] is not False
                or [point["updates"] for point in result["curve"]] != list(study.STEPS)
                or [point["episodes_per_family"] for point in result["curve"]] != [step * study.MICRO for step in study.STEPS]
                or result["cpu_restart"].get("exact") is not True or result["cpu_restart"].get("device") != "cpu"
                or result["cpu_restart"].get("weights_sha256") != report["weights_sha256"]
                or set(result["controls"]) != {"blank", "reset"} or result["final"] != result["curve"][-1]["metrics"]):
            raise ValueError("completed candidate audit identity differs")
        for point in result["curve"]:
            if point["weights_sha256"] != verified_digests[job][str(point["updates"])]:
                raise ValueError("audit curve tensor digest differs")
            _validate_scores(point["metrics"], banks["audit"], manifests["audit"], role="audit")
        for control, scores in result["controls"].items():
            _validate_scores(scores, banks["audit"], manifests["audit"], role="audit", control=control)
        latest_banks, metadata = trainer.latest_observed_banks()
        _same(metadata, result["latest_observed_metadata"])
        latest_rows = {f"latest/{family}/t{turns}": rows for family, buckets in latest_banks.items() for turns, rows in buckets.items()}
        latest_manifest = {name: _stats(rows) for name, rows in latest_rows.items()}
        _validate_scores(result["initial_realization_fit"], initial_rows, initial_manifest, role="train_fit")
        _validate_scores(result["latest_observed_fit"], latest_rows, latest_manifest, role="train_fit")
        main[job] = report
        del trainer
    result = assemble_summary(protocol, main, audit, timings)
    if result["compute"]["retained_optimizer_updates"] != 7200 or result["compute"]["sampled_exposures"]["episodes"] != 691200:
        raise ValueError("retained study update/exposure budget differs")
    result["verification"] = {"input_file_sha256": hashes, "source_sha256": protocol["source_sha256"],
        "verified_source_files": len(protocol["source_sha256"]), "helper_sha256": helper_hashes,
        "checkpoint_weights_sha256": verified_digests, "historical_evidence": protocol["bank_diagnostics"]["historical_evidence"],
        "protected_transcript_count": len(protected), "bank_boundaries": protocol["bank_diagnostics"]["boundaries"],
        "stream_verification": "Every saved optimizer state was structurally validated and restored; sampler/occurrence/digest/statistics states matched sealed one-pass canonical replay. Replay does not reproduce AdamW optimization.",
        "no_neural_inference_or_training_performed": True}
    study.load_protocol(directory)
    if (any(study.file_hash(directory / name) != sha for name, sha in hashes.items())
            or study.file_hash(__file__) != helper_hashes["summary"]
            or study.file_hash(metrics_helper.__file__) != helper_hashes["metric_validator"]
            or historical_evidence() != protocol["bank_diagnostics"]["historical_evidence"]):
        raise RuntimeError("evidence changed while constructing summary")
    _check_finite_tree(result, "realization summary")
    return result


def summarize(directory):
    directory = Path(directory)
    if any(not path.is_file() for path in _required(directory)):
        raise ValueError("summary requires both endpoints, all checkpoints, both complete replays and both completed audits")
    with run_lock(directory / "audit"):
        result = verify_completed(directory)
        atomic_json(directory / "audit" / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = summarize(args.study)
    print(json.dumps({"compute": result["compute"], "verified_source_files": result["verification"]["verified_source_files"]}, indent=2))


if __name__ == "__main__":
    main()
