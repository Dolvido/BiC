"""Matched fixed versus refreshed realizations, with sealed local evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_loop import run_lock
from brain_in_computer.learning_student import _check_finite_tree
from experiments import composition_study as preceding
from experiments.composition_curriculum import FAMILIES
from experiments.composition_evaluation import PreparedBank, evaluate_banks, verify_cpu_restart
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.train_cognitive import atomic_checkpoint, atomic_json


SCHEMA = "bic-realization-study-v1"
JOBS = ("fixed", "fresh")
SEED, SAMPLER_SEED, RATE = 2901, 3901, .001
TOTAL, MICRO = 3600, 32
STEPS = (0, 300, 900, 1800, 3600)
CONFIG = SequenceConfig(max_turns=12)
file_hash = preceding.file_hash


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = [*preceding.source_hashes(), "experiments/realization_banks.py",
             "experiments/realization_training.py", "experiments/realization_study.py",
             "docs/REALIZATION_STUDY_PROTOCOL.md"]
    return {name: file_hash(root / name) for name in names}


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _same(left, right):
    if _canonical(left) != _canonical(right):
        raise ValueError("deterministic evidence differs")


def structural_evidence(banks, updates):
    if type(updates) is not int or not 0 <= updates <= TOTAL:
        raise ValueError("update count outside fixed study")
    result = {}
    for family in FAMILIES:
        generator = torch.Generator().manual_seed(SAMPLER_SEED + sorted(FAMILIES).index(family) * 7919)
        buckets = sorted(banks[family])
        counts = dict.fromkeys(buckets, 0)
        occurrences = {turns: torch.zeros(len(banks[family][turns]) // 2, dtype=torch.long) for turns in buckets}
        for _ in range(updates):
            turns = buckets[int(torch.randint(len(buckets), (1,), generator=generator))]
            pairs = torch.randint(len(occurrences[turns]), (MICRO // 2,), generator=generator)
            occurrences[turns] += torch.bincount(pairs, minlength=len(occurrences[turns]))
            counts[turns] += 1
        result[family] = {"sampler_sha256": hashlib.sha256(bytes(generator.get_state().tolist())).hexdigest(),
                          "bucket_microbatches": counts,
                          "occurrences": {turns: value.tolist() for turns, value in occurrences.items()},
                          "episodes": updates * MICRO,
                          "turns": sum(turns * value * MICRO for turns, value in counts.items())}
    return _canonical(result)


def prepare(directory):
    from experiments.realization_banks import prepare_banks, bank_manifest, protected_transcripts
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        if (directory / "protocol.json").exists():
            return load_protocol(directory)
        preceding.load_protocol("runs/composition-study-local")
        banks, diagnostics = prepare_banks(with_diagnostics=True)
        protected = sorted(protected_transcripts(banks))
        atomic_checkpoint(directory / "banks.pt", banks)
        atomic_checkpoint(directory / "protected.pt", protected)
        protocol = _canonical({
            "schema": SCHEMA, "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_hashes(), "banks_file_sha256": file_hash(directory / "banks.pt"),
            "protected_file_sha256": file_hash(directory / "protected.pt"),
            "protected_transcripts": len(protected),
            "bank_manifest": bank_manifest(banks), "bank_diagnostics": diagnostics,
            "jobs": list(JOBS), "seed": SEED, "sampler_seed": SAMPLER_SEED,
            "config": asdict(CONFIG), "learning_rate": RATE, "micro_batch_size": MICRO,
            "microbatches_per_update": 3, "family_order": list(FAMILIES),
            "initial_weights_sha256": checkpoint_digest(build_sequence_student(SEED, config=CONFIG)),
            "updates": TOTAL, "checkpoints": list(STEPS),
            "expected_final_structural_streams": structural_evidence(banks["train"], TOTAL),
            "initial_realization": "Fresh occurrence zero uses the fixed original; later occurrences refresh independently of structural RNG.",
            "comparison": "Same ordered procedures, family/length draws, objective, optimizer and update/episode budget. Actual realizations, targets, increments, bytes and generation cost can differ.",
            "selection": "Fixed endpoints. No audit scoring until both endpoints and all saved stream evidence pass verification.",
            "fit": "Endpoint shared initial-realization probe and latest actually observed realization per recipe, each labeled with observed coverage.",
            "interpretation": "Single initialization; combined name/value diversity intervention. New transcripts do not make historically inspected finite ancestry motifs or semantic algorithms novel.",
            "automatic_promotion": False})
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
        raise ValueError("realization study source changed")
    contract = {"jobs": list(JOBS), "seed": SEED, "sampler_seed": SAMPLER_SEED,
                "config": asdict(CONFIG), "learning_rate": RATE, "micro_batch_size": MICRO,
                "microbatches_per_update": 3, "family_order": list(FAMILIES),
                "updates": TOTAL, "checkpoints": list(STEPS), "automatic_promotion": False}
    if any(protocol.get(name) != value for name, value in contract.items()):
        raise ValueError("frozen study contract differs")
    if any(file_hash(directory / "source" / name) != sha for name, sha in protocol["source_sha256"].items()):
        raise ValueError("frozen source changed")
    for name, field in (("banks.pt", "banks_file_sha256"), ("protected.pt", "protected_file_sha256")):
        if file_hash(directory / name) != protocol[field]:
            raise ValueError("frozen banks or exclusions changed")
    return protocol


def _required(directory):
    return [Path(directory) / "main" / job / name for job in JOBS
            for name in ("latest.pt", "report.json", *(f"checkpoint-{step:06d}.pt" for step in STEPS))]


def _completed_inputs(directory):
    directory = Path(directory)
    required = _required(directory)
    if any(not path.is_file() for path in required):
        raise ValueError("both completed endpoints and every fixed checkpoint are required")
    return {path.relative_to(directory).as_posix(): file_hash(path) for path in required}


def _read_banks(directory):
    directory = Path(directory)
    return (torch.load(directory / "banks.pt", map_location="cpu", weights_only=True),
            torch.load(directory / "protected.pt", map_location="cpu", weights_only=True))


def _trainer(banks, protected, job, device="cpu", payload=None):
    from experiments.realization_training import RealizationTrainer
    if job not in JOBS:
        raise ValueError("unknown realization candidate")
    return RealizationTrainer(banks["train"], mode=job, protected_transcripts=protected, seed=SEED,
                              sampler_seed=SAMPLER_SEED, device=device, micro_batch_size=MICRO,
                              learning_rate=RATE, config=CONFIG, payload=payload)


def validate_snapshot(saved, protocol, job, banks):
    if saved.get("schema") != SCHEMA or saved.get("protocol") != protocol or saved.get("job") != job:
        raise ValueError("snapshot provenance differs")
    training = saved["training"]
    updates = training["updates"]
    expected = structural_evidence(banks["train"], updates)
    if training["family_microbatches"] != dict.fromkeys(FAMILIES, updates):
        raise ValueError("joint schedule family counts differ")
    for family, row in expected.items():
        if (hashlib.sha256(bytes(training["samplers"][family].tolist())).hexdigest() != row["sampler_sha256"]
                or _canonical(training["bucket_microbatches"][family]) != row["bucket_microbatches"]
                or _canonical(training["realization"]["occurrences"][family]) != row["occurrences"]
                or training["exposures"][family]["episodes"] != row["episodes"]
                or training["exposures"][family]["turns"] != row["turns"]):
            raise ValueError("structural schedule evidence differs")
    _check_finite_tree(saved, "realization snapshot")


def train(directory, job, device="cuda"):
    directory = Path(directory)
    protocol = load_protocol(directory)
    banks, protected = _read_banks(directory)
    output = directory / "main" / job
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=True) if (output / "latest.pt").exists() else None
        if saved:
            validate_snapshot(saved, protocol, job, banks)
        trainer = _trainer(banks, protected, job, device, saved["training"] if saved else None)
        prepared = {name: PreparedBank(rows, role="dev", config=CONFIG) for name, rows in banks["development"].items()}
        history = saved["history"] if saved else []
        seconds = saved["training_step_seconds"] if saved else 0.
        started = time.monotonic()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        def snapshot():
            return {"schema": SCHEMA, "protocol": protocol, "job": job, "training": trainer.snapshot(),
                    "history": history, "training_step_seconds": seconds}
        def save_point():
            metrics = evaluate_banks(trainer.model, prepared, score_replies=False)
            history.append({"updates": trainer.updates, "development": metrics,
                            "training_step_seconds": seconds, "stream": trainer.stream_evidence()})
            load_protocol(directory)
            saved = snapshot()
            atomic_checkpoint(output / f"checkpoint-{trainer.updates:06d}.pt", saved)
            atomic_checkpoint(output / "latest.pt", saved)
            print(json.dumps({"job": job, "updates": trainer.updates,
                              "development_final_pairs": metrics["macro_final_pair_accuracy"]}), flush=True)
        if not saved:
            save_point()
        while trainer.updates < TOTAL:
            if device == "cuda":
                torch.cuda.synchronize()
            tick = time.monotonic()
            result = trainer.step(tuple(FAMILIES))
            if device == "cuda":
                torch.cuda.synchronize()
            seconds += time.monotonic() - tick
            if trainer.updates in STEPS:
                save_point()
            elif trainer.updates % 100 == 0:
                load_protocol(directory)
                atomic_checkpoint(output / "latest.pt", snapshot())
                print(json.dumps({"job": job, "updates": trainer.updates, "loss": result["loss"]}), flush=True)
        final = snapshot()
        validate_snapshot(final, protocol, job, banks)
        load_protocol(directory)
        atomic_checkpoint(output / "latest.pt", final)
        atomic_json(output / "report.json", {"schema": SCHEMA, "job": job, "protocol": protocol,
            "updates": trainer.updates, "stream": trainer.stream_evidence(),
            "exposures": trainer.exposures, "history": history, "training_step_seconds": seconds,
            "step_time_scope": "Synchronized durable steps include generation, validation, packing and optimizer work; these are concurrent worker intervals.",
            "invocation_seconds_after_preparation": time.monotonic() - started,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
            "weights_sha256": checkpoint_digest(trainer.model),
            "checkpoint_file_sha256": file_hash(output / "latest.pt"), "heldout_evaluation_performed": False})


def verify(directory, job):
    """Replay all accepted/rejected realization draws before any neural audit."""
    from experiments.realization_training import replay_evidence
    if job not in JOBS:
        raise ValueError("unknown realization candidate")
    directory = Path(directory)
    protocol = load_protocol(directory)
    files = _completed_inputs(directory)
    if checkpoint_digest(build_sequence_student(SEED, config=CONFIG)) != protocol["initial_weights_sha256"]:
        raise ValueError("prescribed initialization differs from frozen digest")
    banks, protected = _read_banks(directory)
    _same(structural_evidence(banks["train"], TOTAL), protocol["expected_final_structural_streams"])
    output = directory / "verification"
    output.mkdir(parents=True, exist_ok=True)
    (output / job).mkdir(exist_ok=True)
    with run_lock(output / job):
        report = json.loads((directory / "main" / job / "report.json").read_text())
        final = torch.load(directory / "main" / job / "latest.pt", map_location="cpu", weights_only=True)
        validate_snapshot(final, protocol, job, banks)
        trainer = _trainer(banks, protected, job, payload=final["training"])
        if (trainer.updates != TOTAL or report["updates"] != TOTAL or report["protocol"] != protocol
                or report["job"] != job or report["heldout_evaluation_performed"] is not False
                or report["weights_sha256"] != checkpoint_digest(trainer.model)
                or report["checkpoint_file_sha256"] != files[f"main/{job}/latest.pt"]
                or report["training_step_seconds"] != final["training_step_seconds"]
                or report["exposures"] != final["training"]["exposures"]):
            raise ValueError("completed endpoint report differs")
        _same(report["history"], final["history"])
        _same(report["stream"], trainer.stream_evidence())
        expected = replay_evidence(banks["train"], job, protected, STEPS,
                                   sampler_seed=SAMPLER_SEED, micro_batch_size=MICRO)
        weights = {}
        for step in STEPS:
            saved = torch.load(directory / "main" / job / f"checkpoint-{step:06d}.pt", map_location="cpu", weights_only=True)
            validate_snapshot(saved, protocol, job, banks)
            trainer._restore(saved["training"])
            if trainer.updates != step:
                raise ValueError("fixed checkpoint step differs")
            _same(trainer.stream_evidence(), expected[step])
            _same(saved["history"][-1]["stream"], expected[step])
            if (saved["history"] != [item for item in final["history"] if item["updates"] <= step]
                    or saved["training_step_seconds"] != saved["history"][-1]["training_step_seconds"]
                    or [item["updates"] for item in saved["history"]] != [value for value in STEPS if value <= step]):
                raise ValueError("checkpoint development schedule differs")
            weights[str(step)] = checkpoint_digest(trainer.model)
        if weights["0"] != protocol["initial_weights_sha256"] or weights[str(TOTAL)] != report["weights_sha256"]:
            raise ValueError("initial or final tensor digest differs")
        if _completed_inputs(directory) != files:
            raise RuntimeError("training evidence changed during stream verification")
        load_protocol(directory)
        result = {"schema": SCHEMA, "job": job, "protocol_sha256": file_hash(directory / "protocol.json"),
                  "input_files_sha256": files, "weights_sha256": weights,
                  "replayed_streams": expected, "neural_training_or_audit_performed": False}
        atomic_json(output / f"{job}.json", result)
        print(json.dumps({"verified_streams": job}), flush=True)
        return result


def audit(directory, device="cuda"):
    directory = Path(directory)
    protocol = load_protocol(directory)
    files = _completed_inputs(directory)
    verification = {}
    for job in JOBS:
        path = directory / "verification" / f"{job}.json"
        if not path.is_file():
            raise ValueError("both complete stream replays are required before audit")
        value = json.loads(path.read_text())
        if (value["schema"] != SCHEMA or value["job"] != job
                or value["input_files_sha256"] != files
                or value["protocol_sha256"] != file_hash(directory / "protocol.json")):
            raise ValueError("stream verification provenance differs")
        verification[job] = value
    banks, protected = _read_banks(directory)
    prepared = {name: PreparedBank(rows, role="audit", config=CONFIG) for name, rows in banks["audit"].items()}
    initial_prepared = {f"initial/{family}/t{turns}": PreparedBank(rows, role="train_fit", config=CONFIG)
                        for family, buckets in banks["train"].items() for turns, rows in buckets.items()}
    marker = {"schema": SCHEMA, "protocol_sha256": file_hash(directory / "protocol.json"),
              "input_files_sha256": files,
              "verification_files_sha256": {job: file_hash(directory / "verification" / f"{job}.json") for job in JOBS}}
    output = directory / "audit"
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        if (output / "evaluation-started.json").exists() and json.loads((output / "evaluation-started.json").read_text()) != marker:
            raise ValueError("sealed evaluation input changed")
        atomic_json(output / "evaluation-started.json", marker)
        results = {}
        for job in JOBS:
            path = output / f"{job}.json"
            if path.exists():
                result = json.loads(path.read_text())
                if result["inputs"] != marker or result["job"] != job:
                    raise ValueError("cached audit provenance differs")
                results[job] = result
                continue
            started = time.monotonic()
            model = build_sequence_student(SEED, device=device, config=CONFIG)
            curve = []
            for step in STEPS:
                saved = torch.load(directory / "main" / job / f"checkpoint-{step:06d}.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(saved["training"]["weights"], strict=True)
                if checkpoint_digest(model) != verification[job]["weights_sha256"][str(step)]:
                    raise ValueError("verified checkpoint tensor digest differs")
                curve.append({"updates": step, "episodes_per_family": step * MICRO,
                              "weights_sha256": checkpoint_digest(model), "metrics": evaluate_banks(model, prepared)})
            final = torch.load(directory / "main" / job / "latest.pt", map_location="cpu", weights_only=True)
            trainer = _trainer(banks, protected, job, payload=final["training"])
            latest_banks, latest_metadata = trainer.latest_observed_banks()
            latest_prepared = {f"latest/{family}/t{turns}": PreparedBank(rows, role="train_fit", config=CONFIG)
                              for family, buckets in latest_banks.items() for turns, rows in buckets.items() if rows}
            result = {"schema": SCHEMA, "job": job, "inputs": marker, "curve": curve,
                      "final": curve[-1]["metrics"],
                      "initial_realization_fit": evaluate_banks(model, initial_prepared),
                      "latest_observed_fit": evaluate_banks(model, latest_prepared),
                      "latest_observed_metadata": latest_metadata,
                      "controls": {control: evaluate_banks(model, prepared, control=control) for control in ("blank", "reset")},
                      "cpu_restart": verify_cpu_restart(model, banks["audit"][next(iter(prepared))][0]),
                      "seconds": time.monotonic() - started, "automatic_promotion": False}
            atomic_json(path, result)
            results[job] = result
            del trainer, model
            print(json.dumps({"audited": job}), flush=True)
        load_protocol(directory)
        if _completed_inputs(directory) != files or any(file_hash(directory / "verification" / f"{job}.json") != sha
                for job, sha in marker["verification_files_sha256"].items()):
            raise RuntimeError("evidence changed during audit")
        atomic_json(output / "report.json", {"schema": SCHEMA, "protocol": protocol, "inputs": marker,
                    "results": results, "base_checkpoints_unchanged": True, "automatic_promotion": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("prepare", "train", "verify", "audit"), required=True)
    parser.add_argument("--job", choices=JOBS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "prepare":
        result = prepare(args.output)
        print(json.dumps({"prepared": True, "source_files": len(result["source_sha256"])}))
    elif args.phase == "train":
        train(args.output, args.job, args.device)
    elif args.phase == "verify":
        verify(args.output, args.job)
    else:
        audit(args.output, args.device)


if __name__ == "__main__":
    main()
