"""Frozen arithmetic-family audit for the calibrated architecture comparison.

The orchestrator is imported only inside functions to avoid source-discovery
circular imports. Evaluation uses each architecture's original action/decoder.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import io
import json
from pathlib import Path
import time

import torch

from brain_in_computer.dialogue_student import (
    ByteCodec, _generate_reply_tokens, _run_turn, checkpoint_digest, encode_dialogues,
)
from brain_in_computer.learning_loop import run_lock
from brain_in_computer.learning_student import _check_finite_tree, _cpu_copy
from experiments.audit_cognitive_transfer import (
    bank_manifest, exclude_overlapping_pairs, file_hash, transcript,
)
from experiments.cognitive_credit import build_credit_student
from experiments.sequence_data import pack_observations
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.train_cognitive import atomic_checkpoint, atomic_json, fingerprint_rows


SCHEMA = "bic-sequence-study-audit-v1"
BUDGETS = (0, 1, 4, 16, 64)
RATES = (.0003, .001, .003)
ARCHITECTURES = ("recurrent", "episodic", "sequence")
MAIN_SEED, CALIBRATION_SEED, ADAPT_SEED = 2501, 2401, 3501


def study():
    return importlib.import_module("experiments.sequence_study")


def build_model(architecture, seed, device="cpu"):
    if architecture == "sequence":
        return build_sequence_student(seed, device=device)
    if architecture not in ("recurrent", "episodic"):
        raise ValueError("unknown study architecture")
    return build_credit_student(seed, device=device, memory_mode=architecture)


def source_hashes():
    hashes = dict(study().source_hashes())
    root = Path(__file__).resolve().parents[1]
    hashes[Path(__file__).relative_to(root).as_posix()] = file_hash(__file__)
    return hashes


def prepare_banks():
    api = study()
    support = api.make_banks(seed=36_000_000, count=64, split="train", families=(api.HELDOUT_FAMILY,))
    excluded = {transcript(row) for row in support[api.HELDOUT_FAMILY]}
    queries, exclusions = {}, {}
    for level in (2, 3):
        rows = api.make_banks(seed=37_000_000, count=256, split="audit",
            levels=(level,), families=(api.HELDOUT_FAMILY,))[api.HELDOUT_FAMILY]
        queries[f"level_{level}"], exclusions[f"level_{level}"] = exclude_overlapping_pairs(rows, excluded)
    return {"support": support, "query": queries,
        "main_development": api.make_banks(seed=33_000_000, count=64, split="dev"),
        "retained": api.make_banks(seed=34_000_000, count=128, split="dev"),
        "advanced": api.make_banks(seed=35_000_000, count=256, split="audit", levels=(3,))}, exclusions


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        api = study()
        banks, exclusions = prepare_banks()
        contract = {"schema": SCHEMA, "source_sha256": source_hashes(),
            "architectures": list(ARCHITECTURES), "calibration_rates": list(RATES),
            "calibration_seed": CALIBRATION_SEED, "calibration_updates": 600,
            "main_seed": MAIN_SEED, "main_updates": 3600, "batch_size": 64,
            "main_checkpoint_every": 600, "train_seed": 32_000_000,
            "train_count_per_family_level": 512, "train_levels": [1, 2],
            "training_families": list(api.TRAIN_FAMILIES), "heldout_family": api.HELDOUT_FAMILY,
            "dev_seed": 33_000_000, "adaptation_seed": ADAPT_SEED,
            "adaptation_updates": list(BUDGETS), "support_episodes": 128,
            "adaptation_rate": "each architecture's selected calibration rate, also used for its fresh floor",
            "selection_rule": "final 600-update development mean of macro pair accuracy and macro "
                "last-known-later-query accuracy; ties choose lower learning rate",
            "banks": bank_manifest(banks), "support_overlap_whole_pair_exclusions": exclusions,
            "interpretation": "One calibration seed and main seed, one withheld family/support draw. "
                "Independent sequence learner is a diagnostic reference, not regional BiC capability. "
                "No automatic promotion or general-intelligence claim."}
        path = directory / "manifest.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf8"))
            if {key: value for key, value in previous.items() if key != "prepared_utc"} != contract:
                raise ValueError("prepared source or audit contract changed")
            load_prepared(directory)
            return previous
        contract["prepared_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(directory / "frozen_banks.json", banks)
        atomic_json(path, contract)
        return contract


def load_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    banks = json.loads((directory / "frozen_banks.json").read_text(encoding="utf8"))
    if manifest.get("schema") != SCHEMA or manifest["source_sha256"] != source_hashes():
        raise ValueError("audit source mismatch")
    if manifest["banks"] != bank_manifest(banks):
        raise ValueError("audit bank mismatch")
    return manifest, banks


def calibration_score(metrics):
    pair = metrics["macro_pair_accuracy"]
    later = metrics["macro_later_known_accuracy"]
    if pair is None or later is None:
        raise ValueError("calibration score requires measured pair and later-query denominators")
    if not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in (pair, later)):
        raise ValueError("invalid calibration scores")
    return (pair + later) / 2


def choose_rate(scores):
    """Finite predeclared grid; exact numerical ties prefer the smaller rate."""
    if set(scores) != set(RATES):
        raise ValueError("all three declared learning rates are required")
    if not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in scores.values()):
        raise ValueError("invalid calibration score")
    return min(scores, key=lambda rate: (-scores[rate], rate))


def verify_run(path, manifest, *, phase, architecture, rate):
    path = Path(path).resolve()
    api = study()
    root = path.parents[1]
    frozen = api.load_protocol(root)
    protocol = json.loads((path / "protocol.json").read_text(encoding="utf8"))
    report = json.loads((path / "report.json").read_text(encoding="utf8"))
    saved = torch.load(path / "latest.pt", map_location="cpu", weights_only=True)
    steps, seed = (600, CALIBRATION_SEED) if phase == "calibration" else (3600, MAIN_SEED)
    checks = {"schema": api.SCHEMA, "architecture": architecture, "phase": phase, "seed": seed,
        "learning_rate": rate, "steps": steps, "batch_size": 64, "checkpoint_every": 600,
        "study": frozen}
    study_checks = {"architectures": list(ARCHITECTURES), "learning_rates": list(RATES),
        "calibration_seed": CALIBRATION_SEED, "main_seed": MAIN_SEED,
        "calibration_steps": 600, "main_steps": 3600, "batch_size": 64, "checkpoint_every": 600,
        "training_families": manifest["training_families"], "heldout_family": manifest["heldout_family"],
        "train_levels": [1, 2], "train_seed": 32_000_000,
        "train_count_per_family_level": 512, "dev_seed": 33_000_000,
        "dev_count_per_family_level": 64, "schedule": "interleaved", "tutor": "off",
        "arithmetic_main_access": False}
    if (any(protocol.get(key) != value for key, value in checks.items())
            or any(frozen.get(key) != value for key, value in study_checks.items())):
        raise ValueError("run settings differ from the prepared study")
    if (report["updates"] != steps or saved["training"]["updates"] != steps
            or report["stop_reason"] != "planned_updates" or report["heldout_evaluation_performed"]
            or report["protocol"] != protocol or saved["protocol"] != protocol
            or saved.get("schema") != api.SCHEMA or report["examples"] != steps * 64):
        raise ValueError("run is incomplete or has inconsistent provenance")
    expected_counts = {family: steps // 3 for family in manifest["training_families"]}
    if saved["training"]["family_updates"] != expected_counts or report["family_updates"] != expected_counts:
        raise ValueError("family update budgets differ")
    if protocol["source_sha256"] != manifest["source_sha256"] or protocol["source_sha256"] != source_hashes():
        raise ValueError("training source changed")
    draws = api.exposure_hashes(seed, steps=steps, batch_size=64, lessons=512)
    if protocol["within_family_draw_sha256"] != draws or frozen[f"{phase}_draw_sha256"] != draws:
        raise ValueError("sample streams differ from the declared seed and exposure budget")
    if (path / "initial.pt").stat().st_mtime < datetime.fromisoformat(manifest["prepared_utc"]).timestamp():
        raise ValueError("audit preparation must precede every calibration/main initial checkpoint")
    model = build_model(architecture, seed)
    expected_initial = checkpoint_digest(model)
    initial = torch.load(path / "initial.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(initial["weights"])
    if checkpoint_digest(model) != expected_initial:
        raise ValueError("run did not start from its own declared fresh initialization")
    _check_finite_tree(saved["training"]["weights"], "audited_weights")
    model.load_state_dict(saved["training"]["weights"])
    if checkpoint_digest(model) != report["checkpoint_sha256"]:
        raise ValueError("checkpoint and reported weights disagree")
    train = api.make_banks(seed=32_000_000, count=512, split="train")
    dev = api.make_banks(seed=33_000_000, count=64, split="dev")
    datasets = json.loads((path / "datasets.json").read_text(encoding="utf8"))
    expected_datasets = {"training": {name: fingerprint_rows(rows) for name, rows in train.items()},
                         "development": {name: fingerprint_rows(rows) for name, rows in dev.items()}}
    if (datasets != expected_datasets or report["datasets"] != datasets
            or json.loads((root / "datasets.json").read_text(encoding="utf8")) != datasets):
        raise ValueError("training or development bank identity differs")
    return {"architecture": architecture, "phase": phase, "rate": rate, "path": str(path),
            "checkpoint_sha256": file_hash(path / "latest.pt"), "weights": saved["training"]["weights"],
            "protocol": protocol, "report": report, "training_banks": train}


def verify_study(directory, manifest):
    directory = Path(directory)
    selections = study().select_rates(directory, persist=False)
    calibration, selected, exposure_by_phase = [], {}, {}
    for architecture in ARCHITECTURES:
        scores = {}
        for rate in RATES:
            path = directory / "calibration" / f"{architecture}-lr-{rate:g}"
            record = verify_run(path, manifest, phase="calibration", architecture=architecture, rate=rate)
            scores[rate] = calibration_score(record["report"]["development"])
            calibration.append(record)
        selected[architecture] = choose_rate(scores)
    if selections["choices"] != selected:
        raise ValueError("selected rates disagree with the fixed calibration rule")
    main = [verify_run(directory / "main" / architecture, manifest,
        phase="main", architecture=architecture, rate=selected[architecture]) for architecture in ARCHITECTURES]
    for record in calibration + main:
        phase = record["phase"]
        draws = record["protocol"]["within_family_draw_sha256"]
        if phase in exposure_by_phase and exposure_by_phase[phase] != draws:
            raise ValueError("architectures or rates received different sample streams")
        exposure_by_phase[phase] = draws
    return calibration, main, selected


def fresh_adaptation(architecture, weights, banks, *, rate, device="cpu"):
    trainer = study().make_trainer(architecture, banks, seed=ADAPT_SEED, device=device,
        batch_size=64, learning_rate=rate, payload=None)
    trainer.model.load_state_dict(weights, strict=True)
    if trainer.optimizer.state or trainer.updates:
        raise RuntimeError("adaptation must reset optimizer moments and update count")
    return trainer


def regional_restart(weights, architecture, episode):
    """Persist full regional state with the complete (unused head included) model."""
    model = build_model(architecture, MAIN_SEED)
    model.load_state_dict(weights)
    model.eval()
    digest = checkpoint_digest(model)
    with torch.inference_mode():
        batches = encode_dialogues(model, [episode])
        decoder = torch.full((1, 1), ByteCodec.BOS, dtype=torch.long)
        state = None
        for batch in batches[:3]:
            _, state = _run_turn(model, batch, state, decoder=decoder)
        stream = io.BytesIO()
        torch.save({"weights": _cpu_copy(model.state_dict()),
                    "state": model.state_to_dict(state, checkpoint_sha256=digest)}, stream)
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
        restored = build_model(architecture, MAIN_SEED)
        restored.load_state_dict(payload["weights"])
        restored.eval()
        restarted = restored.state_from_dict(payload["state"], checkpoint_sha256=checkpoint_digest(restored))
        for batch in batches[3:]:
            a, state = _run_turn(model, batch, state, decoder=decoder)
            b, restarted = _run_turn(restored, batch, restarted, decoder=decoder)
            if any(not torch.equal(a[key], b[key]) for key in ("logits", "language_logits", "production_context")):
                raise RuntimeError("regional CPU continuation differs")
            if not torch.equal(_generate_reply_tokens(model, a["production_context"]),
                               _generate_reply_tokens(restored, b["production_context"])):
                raise RuntimeError("regional CPU generated continuation differs")
    return {"exact": True, "serialized_after_turn": 3, "checked_remaining_turns": 3,
            "device": "cpu", "checkpoint_sha256": digest, "includes_auxiliary_weights": True,
            "state_kind": "regional and episodic activity"}


def sequence_restart(model, episode):
    """The sequence model restores an observed prefix, not a hidden-state cache."""
    weights = _cpu_copy(model.state_dict())
    original = build_sequence_student(MAIN_SEED, config=model.config)
    original.load_state_dict(weights)
    original.eval()
    digest = checkpoint_digest(original)
    texts = [turn["text"] for turn in episode["turns"]]
    stream = io.BytesIO()
    torch.save({"weights": weights, "config": asdict(model.config), "observed_prefix": texts[:3],
                "checkpoint_sha256": digest}, stream)
    stream.seek(0)
    payload = torch.load(stream, map_location="cpu", weights_only=True)
    restored = build_sequence_student(MAIN_SEED, config=SequenceConfig(**payload["config"]))
    restored.load_state_dict(payload["weights"])
    restored.eval()
    if checkpoint_digest(restored) != payload["checkpoint_sha256"]:
        raise RuntimeError("saved sequence prefix is bound to different weights")
    prefix = payload["observed_prefix"]
    with torch.inference_mode():
        for end in range(4, 7):
            prefix = [*prefix, texts[end - 1]]
            inputs_a = pack_observations([texts[:end]], max_context_tokens=model.config.max_positions,
                                        max_input_bytes=model.config.max_input_bytes)
            inputs_b = pack_observations([prefix], max_context_tokens=model.config.max_positions,
                                        max_input_bytes=model.config.max_input_bytes)
            decoder = torch.full((1, end, 1), ByteCodec.BOS, dtype=torch.long)
            a = original(**inputs_a, decoder_input_ids=decoder)
            b = restored(**inputs_b, decoder_input_ids=decoder)
            if any(not torch.equal(a[key][:, -1], b[key][:, -1]) for key in ("logits", "language_logits", "production_context")):
                raise RuntimeError("sequence CPU continuation differs")
            if not torch.equal(_generate_reply_tokens(original, a["production_context"][:, -1]),
                               _generate_reply_tokens(restored, b["production_context"][:, -1])):
                raise RuntimeError("sequence CPU reply continuation differs")
    return {"exact": True, "device": "cpu", "serialized_after_turn": 3,
            "checked_remaining_turns": 3, "checkpoint_sha256": digest,
            "state_kind": "observed prefix; recomputed after restart, no hidden-state cache"}


def adapt_candidate(architecture, weights, banks, *, rate, device="cpu"):
    api = study()
    started = time.monotonic()
    trainer = fresh_adaptation(architecture, weights, banks["support"], rate=rate, device=device)
    initial_digest = checkpoint_digest(trainer.model)
    main = api.evaluate(architecture, trainer.model, banks["main_development"], replies=True)
    retained_before = api.evaluate(architecture, trainer.model, banks["retained"], replies=True)
    advanced_before = api.evaluate(architecture, trainer.model, banks["advanced"], replies=True)
    curve, training_seconds = [], 0.
    for budget in BUDGETS:
        while trainer.updates < budget:
            start = time.monotonic()
            trainer.step(api.HELDOUT_FAMILY)
            training_seconds += time.monotonic() - start
        metrics = api.evaluate(architecture, trainer.model, banks["query"], replies=True)
        curve.append({"updates": budget, "support_exposures": budget * 64, **metrics})
    retained_after = api.evaluate(architecture, trainer.model, banks["retained"], replies=True)
    advanced_after = api.evaluate(architecture, trainer.model, banks["advanced"], replies=True)
    controls = {name: api.evaluate(architecture, trainer.model, banks["query"], replies=True, **kwargs)
        for name, kwargs in (("blank_text", {"blank_text": True}), ("reset_history", {"reset_history": True}))}
    auc = sum((right["updates"] - left["updates"]) *
        (right["macro_query_accuracy"] + left["macro_query_accuracy"]) / 2
        for left, right in zip(curve, curve[1:])) / 64
    snapshot = trainer.snapshot()
    if architecture == "sequence":
        restart = sequence_restart(trainer.model, banks["query"]["level_2"][0])
    else:
        restart = regional_restart(snapshot["weights"], architecture, banks["query"]["level_2"][0])
    return {"architecture": architecture, "learning_rate": rate, "adaptation_seed": ADAPT_SEED,
        "optimizer_reset": True, "support_episodes": 128, "curve": curve,
        "normalized_adaptation_auc": auc, "main_development": main,
        "retention_before": retained_before, "retention_after": retained_after,
        "advanced_before": advanced_before, "advanced_after": advanced_after,
        "controls_at_64": controls, "cpu_restart": restart, "training_seconds": training_seconds,
        "wall_seconds": time.monotonic() - started, "initial_weight_sha256": initial_digest,
        "final_weight_sha256": checkpoint_digest(trainer.model)}, snapshot


def audit(directory, *, device="cpu"):
    torch.set_num_threads(1)
    directory = Path(directory)
    output = directory / "audit"
    with run_lock(output):
        manifest, banks = load_prepared(output)
        calibration, main, selected = verify_study(directory, manifest)
        train_text = {transcript(row) for rows in main[0]["training_banks"].values() for row in rows}
        exclusions = {}
        for group in ("main_development", "retained", "advanced"):
            exclusions[group] = {}
            for name, rows in banks[group].items():
                banks[group][name], exclusions[group][name] = exclude_overlapping_pairs(rows, train_text)
        def brief(record):
            return {key: record[key] for key in ("architecture", "phase", "rate", "path", "checkpoint_sha256")}
        inputs = {"schema": SCHEMA, "device": device, "manifest_sha256": file_hash(output / "manifest.json"),
            "selection_sha256": file_hash(directory / "selection.json"), "selected_rates": selected,
            "calibration": list(map(brief, calibration)), "main": list(map(brief, main)),
            "exclusions": exclusions, "evaluated_banks": bank_manifest(banks)}
        marker = output / "evaluation-started.json"
        if marker.exists() and json.loads(marker.read_text(encoding="utf8")) != inputs:
            raise ValueError("audit already started with different models or settings")
        atomic_json(marker, inputs)
        if (output / "report.json").exists():
            return json.loads((output / "report.json").read_text(encoding="utf8"))
        candidates = [(record["architecture"], "pretrained", record["weights"]) for record in main]
        candidates.extend((architecture, "fresh", _cpu_copy(build_model(architecture, MAIN_SEED).state_dict()))
                          for architecture in ARCHITECTURES)
        results = {}
        for architecture, origin, weights in candidates:
            key = f"{architecture}-{origin}"
            path = output / f"{key}.json"
            if path.exists():
                results[key] = json.loads(path.read_text(encoding="utf8"))
                continue
            result, saved = adapt_candidate(architecture, weights, banks, rate=selected[architecture], device=device)
            result["origin"] = origin
            atomic_checkpoint(output / f"{key}-adapted.pt", saved)
            atomic_json(path, result)
            results[key] = result
            print(json.dumps({"candidate": key, "query_at_64": result["curve"][-1]["macro_query_accuracy"],
                "pairs_at_64": result["curve"][-1]["macro_pair_accuracy"],
                "later_known_at_64": result["curve"][-1]["macro_later_known_accuracy"],
                "adaptation_auc": result["normalized_adaptation_auc"]}), flush=True)
        for record in calibration + main:
            if file_hash(Path(record["path"]) / "latest.pt") != record["checkpoint_sha256"]:
                raise RuntimeError("calibration or main checkpoint changed during audit")
        if source_hashes() != manifest["source_sha256"]:
            raise RuntimeError("source changed during audit")
        report = {"schema": SCHEMA, "manifest": manifest, "inputs": inputs, "results": results,
            "prior_learning_gain_over_fresh_auc": {architecture:
                results[f"{architecture}-pretrained"]["normalized_adaptation_auc"] -
                results[f"{architecture}-fresh"]["normalized_adaptation_auc"] for architecture in ARCHITECTURES},
            "base_checkpoints_unchanged": True, "automatic_promotion": False,
            "interpretation": manifest["interpretation"]}
        atomic_json(output / "report.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, help="audit subdirectory to freeze before calibration")
    parser.add_argument("--study", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.prepare:
        if args.study:
            parser.error("prepare and study evaluation are separate phases")
        manifest = prepare(args.prepare)
        print(json.dumps({"prepared": str(args.prepare), "support_episodes": manifest["support_episodes"]}))
    else:
        if not args.study:
            parser.error("evaluation requires --study")
        report = audit(args.study, device=args.device)
        print(json.dumps({"complete": True, "candidates": len(report["results"]), "automatic_promotion": False}))


if __name__ == "__main__":
    main()
