"""Longer fitting and matched-new-exposure rehearsal, preserving prior evidence."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_loop import run_lock
from brain_in_computer.learning_student import _cpu_copy, _check_finite_tree
from experiments import diversity_study as prior
from experiments.diversity_training import DiversityTrainer
from experiments.diversity_evaluation import evaluate_diverse_banks
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.train_cognitive import atomic_checkpoint, atomic_json, fingerprint_rows

SCHEMA = "bic-consolidation-study-v1"
ARMS = ("w32-n8", "w256-n8")
PARENT_UPDATES, TOTAL_UPDATES = 3600, 14400
ADAPT_SEED, ADAPT_BUDGETS = 3701, (0, 4, 16, 64, 256)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = [*prior.source_hashes(), *("experiments/" + name + ".py" for name in (
        "consolidation_banks", "consolidation_training", "consolidation_evaluation", "consolidation_study")),
        "docs/CONSOLIDATION_STUDY_PROTOCOL.md"]
    return {name: prior.file_hash(root / name) for name in names}


def parent_evidence(parent):
    parent = Path(parent)
    protocol = prior.load_protocol(parent)
    records = {}
    for arm in ARMS:
        path = parent / "main" / arm
        report = json.loads((path / "report.json").read_text(encoding="utf8"))
        saved = torch.load(path / "latest.pt", map_location="cpu", weights_only=True)
        prior.validate_resume(saved, protocol, arm)
        if (saved["training"]["updates"] != PARENT_UPDATES or report["updates"] != PARENT_UPDATES
                or report["checkpoint_file_sha256"] != prior.file_hash(path / "latest.pt")
                or report["protocol"] != protocol or report["arm"] != arm or report["heldout_evaluation_performed"]):
            raise ValueError("intact original completed parent required")
        model = build_sequence_student(prior.SEED)
        model.load_state_dict(saved["training"]["weights"], strict=True)
        if checkpoint_digest(model) != report["weights_sha256"]:
            raise ValueError("parent tensor digest mismatch")
        records[arm] = {"checkpoint_sha256": prior.file_hash(path / "latest.pt"),
            "report_sha256": prior.file_hash(path / "report.json"),
            "weights_sha256": checkpoint_digest(model), "updates": PARENT_UPDATES,
            "training_bank_sha256": protocol["banks"]["train"][arm]}
    return {"directory": str(parent.resolve()), "protocol_sha256": prior.file_hash(parent / "protocol.json"),
            "banks_sha256": prior.file_hash(parent / "banks.pt"), "records": records}


def prepare(directory, parent):
    from experiments.consolidation_banks import prepare_banks, build_replay_banks, bank_manifest
    directory, parent = Path(directory), Path(parent)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        if (directory / "protocol.json").exists():
            return load_protocol(directory)
        evidence = parent_evidence(parent)
        _, old = prior.read_banks(parent)
        banks, diagnostics = prepare_banks(old, with_diagnostics=True)
        replay = {arm: build_replay_banks(old["train"][arm], max_pairs=256) for arm in ARMS}
        atomic_checkpoint(directory / "banks.pt", banks)
        atomic_checkpoint(directory / "replay.pt", replay)
        protocol = {"schema": SCHEMA, "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_hashes(), "parent": evidence,
            "arms": list(ARMS), "seed": prior.SEED, "parent_updates": PARENT_UPDATES,
            "total_updates": TOTAL_UPDATES, "additional_updates_per_arm": TOTAL_UPDATES - PARENT_UPDATES,
            "extension_checkpoints": [7200, 10800, 14400], "batch_size": 64, "learning_rate": .001,
            "extension_optimizer": "Original parent AdamW state, family samplers, weights and schedule continue unchanged.",
            "adaptation_seed": ADAPT_SEED, "adaptation_budgets": list(ADAPT_BUDGETS),
            "adaptation_optimizer": "Fresh AdamW, same rate and matched new-example draws across candidates.",
            "replay_batch_size": 32, "replay_weight": .5,
            "replay_rule": "Sorted old-family round robin, complete pairs from compact authenticated original training buffers; one combined backward and optimizer step.",
            "replay_objective": "unchanged new-batch objective + 0.5 * unchanged replay-batch objective",
            "replay_budget_note": "64 new examples per update in both modes; rehearsal adds 32 old examples and a second forward pass. Matched new exposure and updates, NOT matched compute.",
            "model": "Unchanged default independent SequenceStudent, not regional BiC.",
            "banks": bank_manifest(banks), "bank_selection_diagnostics": diagnostics,
            "banks_file_sha256": prior.file_hash(directory / "banks.pt"),
            "replay_file_sha256": prior.file_hash(directory / "replay.pt"),
            "replay_banks": {arm: {family: {"episodes": len(rows), "sha256": fingerprint_rows(rows)}
                                    for family, rows in groups.items()} for arm, groups in replay.items()},
            "automatic_promotion": False,
            "selection": "Fixed final extension checkpoints; audit all four short/long parents with and without replay plus a common fresh no-replay control.",
            "interpretation": "One main seed and one held subject; exploratory consolidation and fitting-budget evidence. New naming/world panels separate transfer challenges. Old audit evidence informed this study; fresh rows do not erase design adaptation. No automatic numerical promotion screen or claim of general intelligence."}
        root = Path(__file__).resolve().parents[1]
        for name in protocol["source_sha256"]:
            target = directory / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
        atomic_json(directory / "protocol.json", protocol)
        return protocol


def load_protocol(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf8"))
    if protocol.get("schema") != SCHEMA or protocol["source_sha256"] != source_hashes():
        raise ValueError("consolidation source changed")
    if any(prior.file_hash(directory / "source" / name) != sha for name, sha in protocol["source_sha256"].items()):
        raise ValueError("frozen consolidation source changed")
    for stem in ("banks", "replay"):
        if prior.file_hash(directory / f"{stem}.pt") != protocol[f"{stem}_file_sha256"]:
            raise ValueError("frozen consolidation bank changed")
    parent = Path(protocol["parent"]["directory"])
    if (prior.file_hash(parent / "protocol.json") != protocol["parent"]["protocol_sha256"]
            or prior.file_hash(parent / "banks.pt") != protocol["parent"]["banks_sha256"]):
        raise ValueError("parent protocol or banks changed")
    for arm, record in protocol["parent"]["records"].items():
        for filename, key in (("latest.pt", "checkpoint_sha256"), ("report.json", "report_sha256")):
            if prior.file_hash(parent / "main" / arm / filename) != record[key]:
                raise ValueError("original parent changed")
    return protocol


def validate_extension(saved, protocol, arm):
    if saved.get("schema") != SCHEMA or saved.get("protocol") != protocol or saved.get("arm") != arm:
        raise ValueError("extension resume contract differs")
    training = saved["training"]
    count = training["updates"]
    if type(count) is not int or not PARENT_UPDATES <= count <= TOTAL_UPDATES:
        raise ValueError("extension updates outside declared range")
    if training["family_updates"] != {family: count // 3 + int(index < count % 3)
                                      for index, family in enumerate(prior.FAMILIES)}:
        raise ValueError("extension schedule differs")
    expected = prior.expected_samplers(count)
    if set(training["samplers"]) != set(expected) or any(
            not torch.equal(training["samplers"][family], value) for family, value in expected.items()):
        raise ValueError("extension sampler stream differs")
    if training["recipe"]["banks"] != protocol["parent"]["records"][arm]["training_bank_sha256"]:
        raise ValueError("extension training bank identity differs")
    recipe = training["recipe"]
    if (recipe["model"] != "bic-diversity-sequence-v1" or recipe["seed"] != prior.SEED
            or recipe["batch_size"] != 64 or recipe["learning_rate"] != .001
            or recipe["config"] != asdict(SequenceConfig())):
        raise ValueError("extension model or optimization recipe differs")


def extend(directory, arm, device="cuda"):
    if arm not in ARMS:
        raise ValueError("unknown extension arm")
    directory = Path(directory)
    protocol = load_protocol(directory)
    parent = Path(protocol["parent"]["directory"])
    _, old = prior.read_banks(parent)
    banks = torch.load(directory / "banks.pt", map_location="cpu", weights_only=True)
    output = directory / "extension" / arm
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        latest = output / "latest.pt"
        saved = torch.load(latest, map_location="cpu", weights_only=True) if latest.exists() else None
        if saved:
            validate_extension(saved, protocol, arm)
            initial = saved["training"]
        else:
            initial = torch.load(parent / "main" / arm / "latest.pt", map_location="cpu", weights_only=True)["training"]
        trainer = DiversityTrainer(old["train"][arm], seed=prior.SEED, device=device,
            learning_rate=protocol["learning_rate"], payload=initial)
        history = saved["history"] if saved else []
        seconds = saved["training_seconds"] if saved else 0.
        start = time.monotonic()
        fit = {family: rows[:64] for family, rows in old["train"][arm].items()}
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        if not saved:
            history.append({"updates": PARENT_UPDATES, "training_seconds": 0.,
                "development": evaluate_diverse_banks(trainer.model, banks["development"], score_replies=False),
                "common_fit": evaluate_diverse_banks(trainer.model, fit, score_replies=False)})
        def snapshot():
            return {"schema": SCHEMA, "protocol": protocol, "arm": arm,
                    "training": trainer.snapshot(), "history": history, "training_seconds": seconds}
        while trainer.updates < TOTAL_UPDATES:
            tick = time.monotonic()
            last = trainer.step(prior.FAMILIES[trainer.updates % 3])
            seconds += time.monotonic() - tick
            if trainer.updates % 1200 == 0:
                metrics = evaluate_diverse_banks(trainer.model, banks["development"], score_replies=False)
                row = {"updates": trainer.updates, "training_seconds": seconds, "development": metrics,
                       "loss": last["loss"]}
                if trainer.updates % 3600 == 0:
                    row["common_fit"] = evaluate_diverse_banks(trainer.model, fit, score_replies=False)
                history.append(row)
                load_protocol(directory)
                atomic_checkpoint(latest, snapshot())
                if trainer.updates % 3600 == 0:
                    atomic_checkpoint(output / f"checkpoint-{trainer.updates:06d}.pt", snapshot())
                atomic_json(output / "progress.json", {"arm": arm, "history": history})
                print(json.dumps({"arm": arm, "updates": trainer.updates, "query": metrics["macro_query_accuracy"],
                    "pairs": metrics["macro_pair_accuracy"], "later": metrics["macro_later_known_accuracy"]}), flush=True)
        load_protocol(directory)
        atomic_checkpoint(latest, snapshot())
        atomic_json(output / "report.json", {"schema": SCHEMA, "protocol": protocol, "arm": arm,
            "updates": trainer.updates, "additional_updates": trainer.updates - PARENT_UPDATES,
            "additional_episode_exposures": (trainer.updates - PARENT_UPDATES) * 64,
            "training_seconds": seconds, "invocation_seconds_after_setup": time.monotonic() - start,
            "development": evaluate_diverse_banks(trainer.model, banks["development"]),
            "common_fit": evaluate_diverse_banks(trainer.model, fit, score_replies=False), "history": history,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
            "peak_scope": "Training and evaluation after encoding, including cached banks; driver overhead excluded.",
            "weights_sha256": checkpoint_digest(trainer.model), "checkpoint_file_sha256": prior.file_hash(latest),
            "heldout_evaluation_performed": False})


def verify_endpoints(directory, protocol):
    directory = Path(directory)
    parent = Path(protocol["parent"]["directory"])
    candidates, files = {}, {}
    verifier = build_sequence_student(prior.SEED)
    for arm in ARMS:
        for stage in ("short", "long"):
            path = parent / "main" / arm if stage == "short" else directory / "extension" / arm
            report = json.loads((path / "report.json").read_text(encoding="utf8"))
            saved = torch.load(path / "latest.pt", map_location="cpu", weights_only=True)
            if stage == "long":
                validate_extension(saved, protocol, arm)
                if report["protocol"] != protocol or report["updates"] != TOTAL_UPDATES:
                    raise ValueError("complete fixed long endpoint required")
            expected = PARENT_UPDATES if stage == "short" else TOTAL_UPDATES
            if (saved["training"]["updates"] != expected or report["updates"] != expected
                    or report["arm"] != arm or report["heldout_evaluation_performed"]
                    or report["checkpoint_file_sha256"] != prior.file_hash(path / "latest.pt")):
                raise ValueError("completed endpoint changed")
            _check_finite_tree(saved, "endpoint")
            verifier.load_state_dict(saved["training"]["weights"], strict=True)
            if checkpoint_digest(verifier) != report["weights_sha256"]:
                raise ValueError("endpoint tensor digest differs")
            key = f"{arm}-{stage}"
            candidates[key] = saved["training"]["weights"]
            files[str((path / "latest.pt").resolve())] = prior.file_hash(path / "latest.pt")
            files[str((path / "report.json").resolve())] = prior.file_hash(path / "report.json")
    return candidates, files


def audit(directory, device="cuda"):
    from experiments.consolidation_evaluation import adapt_candidate
    directory = Path(directory)
    protocol = load_protocol(directory)
    output = directory / "audit"
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        candidates, files = verify_endpoints(directory, protocol)
        banks = torch.load(directory / "banks.pt", map_location="cpu", weights_only=True)
        replay = torch.load(directory / "replay.pt", map_location="cpu", weights_only=True)
        marker = {"schema": SCHEMA, "protocol_sha256": prior.file_hash(directory / "protocol.json"),
                  "endpoint_files_sha256": files}
        marker_path = output / "evaluation-started.json"
        if marker_path.exists() and json.loads(marker_path.read_text(encoding="utf8")) != marker:
            raise ValueError("audit inputs changed")
        atomic_json(marker_path, marker)
        jobs = [(name, mode, weights) for name, weights in candidates.items() for mode in ("ordinary", "replay")]
        jobs.append(("fresh", "ordinary", _cpu_copy(build_sequence_student(prior.SEED).state_dict())))
        results = {}
        for parent_name, mode, weights in jobs:
            name = f"{parent_name}-{mode}"
            path = output / f"{name}.json"
            if path.exists():
                result = json.loads(path.read_text(encoding="utf8"))
                if result["inputs"] != marker or result["name"] != name:
                    raise ValueError("cached audit provenance differs")
                if prior.file_hash(output / f"{name}-adapted.pt") != result["adapted_file_sha256"]:
                    raise ValueError("cached adapted checkpoint changed")
            else:
                arm = parent_name.rsplit("-", 1)[0]
                result, saved = adapt_candidate(weights, banks,
                    replay_banks=replay[arm] if mode == "replay" else None, device=device, seed=ADAPT_SEED)
                result.update(name=name, parent=parent_name, mode=mode, inputs=marker)
                atomic_checkpoint(output / f"{name}-adapted.pt", saved)
                result["adapted_file_sha256"] = prior.file_hash(output / f"{name}-adapted.pt")
                atomic_json(path, result)
                print(json.dumps({"candidate": name, "query_auc": result["areas"]["macro_query_accuracy"],
                    "pair_auc": result["areas"]["macro_pair_accuracy"],
                    "retention_before": result["retention_before"]["macro_query_accuracy"],
                    "retention_after": result["retention_after"]["macro_query_accuracy"]}), flush=True)
            results[name] = result
        if any(prior.file_hash(path) != sha for path, sha in files.items()):
            raise RuntimeError("parent/extension evidence changed during audit")
        load_protocol(directory)
        atomic_json(output / "report.json", {"schema": SCHEMA, "protocol": protocol, "inputs": marker,
            "results": results, "base_checkpoints_unchanged": True, "automatic_promotion": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent", default="runs/diversity-study-local")
    parser.add_argument("--phase", choices=("prepare", "extend", "audit"), required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "prepare":
        result = prepare(args.output, args.parent)
        print(json.dumps({"prepared": True, "sources": len(result["source_sha256"])}))
    elif args.phase == "extend":
        extend(args.output, args.arm, args.device)
    else:
        audit(args.output, args.device)


if __name__ == "__main__":
    main()
