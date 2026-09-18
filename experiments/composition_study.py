"""Prospective matched-exposure composition study on local compute only."""
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
from experiments import consolidation_study as preceding
from experiments.composition_curriculum import FAMILIES
from experiments.composition_evaluation import PreparedBank, evaluate_banks, verify_cpu_restart
from experiments.composition_training import CompositionTrainer
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.train_cognitive import atomic_checkpoint, atomic_json


SCHEMA = "bic-composition-study-v1"
JOBS = ("joint", "seq-color", "seq-count", "fresh-color", "fresh-count")
SEED, SAMPLER_SEED, RATE = 2801, 3801, .001
TOTAL, OLD_PHASE, MICRO = 3600, 1800, 32
EXPOSURES = (0, 384, 1536, 6144, 24576, 115200)
CONFIG = SequenceConfig(max_turns=12)
file_hash = preceding.prior.file_hash


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = [*preceding.source_hashes(), *("experiments/" + name + ".py" for name in
        ("composition_curriculum", "composition_data", "composition_training", "composition_evaluation",
         "composition_banks", "composition_study")), "docs/COMPOSITION_STUDY_PROTOCOL.md"]
    return {name: file_hash(root / name) for name in names}


def total_updates(job):
    if job not in JOBS:
        raise ValueError("unknown composition candidate")
    return 1200 if job.startswith("fresh-") else TOTAL


def families_at(job, update):
    if type(update) is not int or not 0 <= update < total_updates(job):
        raise ValueError("update outside candidate schedule")
    if job == "joint":
        return tuple(FAMILIES)
    late = job.split("-", 1)[1]
    if job.startswith("fresh-"):
        return (late,) * 3
    old = [family for family in FAMILIES if family != late]
    return (old[update % 2],) * 3 if update < OLD_PHASE else (late, late, old[(update - OLD_PHASE) % 2])


def expected_counts(job, updates):
    if type(updates) is not int or not 0 <= updates <= total_updates(job):
        raise ValueError("candidate update count differs")
    if job == "joint":
        return dict.fromkeys(FAMILIES, updates)
    late = job.split("-", 1)[1]
    if job.startswith("fresh-"):
        return {late: updates * 3}
    old = [family for family in FAMILIES if family != late]
    early, later = min(updates, OLD_PHASE), max(0, updates - OLD_PHASE)
    return {late: later * 2, **{family: ((early + 1 - index) // 2) * 3 + (later + 1 - index) // 2
                              for index, family in enumerate(old)}}


def curve_steps(job):
    total_updates(job)
    if job == "joint":
        return [value // MICRO for value in EXPOSURES]
    if job.startswith("seq-"):
        return [OLD_PHASE + value // (2 * MICRO) for value in EXPOSURES]
    return [value // (3 * MICRO) for value in EXPOSURES]


def prepare(directory):
    from experiments.composition_banks import prepare_banks, bank_manifest
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        if (directory / "protocol.json").exists():
            return load_protocol(directory)
        # Completed preceding evidence remains intact; no predecessor weights are reused.
        preceding.load_protocol(Path("runs/consolidation-study-local"))
        banks, diagnostics = prepare_banks(with_diagnostics=True)
        atomic_checkpoint(directory / "banks.pt", banks)
        evidence = SampleEvidence(banks["train"])
        expected_streams = {}
        for family in FAMILIES:
            state, buckets, exposures = evidence.at(family, 3600)
            expected_streams[family] = {"sampler_sha256": hashlib.sha256(bytes(state.tolist())).hexdigest(),
                                        "bucket_microbatches": buckets, "exposures": exposures}
        protocol = {"schema": SCHEMA, "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_hashes(), "banks_file_sha256": file_hash(directory / "banks.pt"),
            "bank_manifest": bank_manifest(banks), "bank_diagnostics": diagnostics,
            "jobs": list(JOBS), "seed": SEED, "sampler_seed": SAMPLER_SEED,
            "learning_rate": RATE, "config": asdict(CONFIG), "micro_batch_size": MICRO,
            "initial_weights_sha256": checkpoint_digest(build_sequence_student(SEED, config=CONFIG)),
            "expected_final_family_streams": expected_streams,
            "microbatches_per_update": 3, "family_order": list(FAMILIES),
            "updates": {job: total_updates(job) for job in JOBS},
            "curve_steps": {job: curve_steps(job) for job in JOBS}, "held_episode_budgets": list(EXPOSURES),
            "final_family_microbatches": {job: expected_counts(job, total_updates(job)) for job in JOBS},
            "objective": "Mean of three unchanged sequence_objective microbatch losses; one clip and AdamW update.",
            "optimizer_phase_boundary": "AdamW state continues through the sequential phase boundary; no reset.",
            "joint_reference": "One common joint run reused in both rotations; these are not independent joint replications.",
            "selection": "Fixed endpoints only; audit after all five runs finish; no automatic promotion.",
            "automatic_promotion": False,
            "interpretation": "One initialization, two typed-domain rotations, synthetic supervised compositions. "
                "Main schedules match every ordered family draw, including replay. Fresh controls match held-family "
                "exposure but not total updates/compute. Causal ancestry signatures are syntactic, not semantic algorithms."}
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
        raise ValueError("composition study source changed")
    if any(file_hash(directory / "source" / name) != sha for name, sha in protocol["source_sha256"].items()):
        raise ValueError("frozen source changed")
    if file_hash(directory / "banks.pt") != protocol["banks_file_sha256"]:
        raise ValueError("frozen banks changed")
    return protocol


class SampleEvidence:
    """Independent exact count/state reconstruction from immutable lesson text."""

    def __init__(self, banks):
        self.banks, self.cache = banks, {}
        self.tables = {}
        for family, buckets in banks.items():
            self.tables[family] = {}
            for turns, rows in buckets.items():
                self.tables[family][turns] = torch.tensor([[sum(len(turn[field].encode("utf8"))
                    for row in rows[index:index + 2] for turn in row["turns"])
                    for field in ("text", "reply")] for index in range(0, len(rows), 2)], dtype=torch.long)

    def at(self, family, microbatches):
        key = family, microbatches
        if key not in self.cache:
            generator = torch.Generator().manual_seed(SAMPLER_SEED + sorted(FAMILIES).index(family) * 7919)
            buckets = sorted(self.banks[family])
            bucket_counts = dict.fromkeys(buckets, 0)
            sums = torch.zeros(2, dtype=torch.long)
            turns_seen = 0
            for _ in range(microbatches):
                turns = buckets[int(torch.randint(len(buckets), (1,), generator=generator))]
                pairs = torch.randint(len(self.banks[family][turns]) // 2, (MICRO // 2,), generator=generator)
                sums += self.tables[family][turns].index_select(0, pairs).sum(0)
                bucket_counts[turns] += 1
                turns_seen += MICRO * turns
            observation, reply = sums.tolist()
            counts = {"episodes": microbatches * MICRO, "turns": turns_seen,
                "observation_tokens": observation + 2 * turns_seen, "observation_bytes": observation,
                "reply_target_tokens": reply + turns_seen, "reply_target_bytes": reply}
            self.cache[key] = generator.get_state(), bucket_counts, counts
        return self.cache[key]


def validate_snapshot(saved, protocol, job, evidence):
    if saved.get("schema") != SCHEMA or saved.get("protocol") != protocol or saved.get("job") != job:
        raise ValueError("candidate snapshot provenance differs")
    training = saved["training"]
    counts = expected_counts(job, training["updates"])
    if training["family_microbatches"] != counts:
        raise ValueError("candidate schedule counters differ")
    if (training["recipe"]["config"] != asdict(CONFIG) or training["recipe"]["seed"] != SEED
            or training["recipe"]["sampler_seed"] != SAMPLER_SEED or training["recipe"]["learning_rate"] != RATE
            or training["recipe"]["micro_batch_size"] != MICRO):
        raise ValueError("candidate neural recipe differs")
    for family, count in counts.items():
        state, buckets, exposures = evidence.at(family, count)
        if (not torch.equal(training["samplers"][family], state)
                or training["bucket_microbatches"][family] != buckets or training["exposures"][family] != exposures):
            raise ValueError("candidate draw stream or sampled exposure differs")
    _check_finite_tree(saved, "composition snapshot")


def train(directory, job, device="cuda"):
    total_updates(job)
    directory = Path(directory)
    protocol = load_protocol(directory)
    banks = torch.load(directory / "banks.pt", map_location="cpu", weights_only=True)
    admitted = {job.split("-", 1)[1]: banks["train"][job.split("-", 1)[1]]} if job.startswith("fresh-") else banks["train"]
    evidence = SampleEvidence(banks["train"])
    output = directory / "main" / job
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=True) if (output / "latest.pt").exists() else None
        if saved:
            validate_snapshot(saved, protocol, job, evidence)
        trainer = CompositionTrainer(admitted, seed=SEED, sampler_seed=SAMPLER_SEED,
                                     learning_rate=RATE, micro_batch_size=MICRO, device=device, config=CONFIG,
                                     payload=saved["training"] if saved else None)
        prepared = {name: PreparedBank(rows, role="dev", config=CONFIG) for name, rows in banks["development"].items()}
        history = saved["history"] if saved else []
        seconds = saved["training_seconds"] if saved else 0.
        checkpoints = {0, *curve_steps(job)}
        started = time.monotonic()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        def snapshot():
            return {"schema": SCHEMA, "protocol": protocol, "job": job, "training": trainer.snapshot(),
                    "history": history, "training_seconds": seconds}
        def save_point():
            metrics = evaluate_banks(trainer.model, prepared, score_replies=False)
            history.append({"updates": trainer.updates, "family_microbatches": dict(trainer.family_microbatches),
                            "development": metrics, "training_seconds": seconds})
            load_protocol(directory)
            payload = snapshot()
            atomic_checkpoint(output / f"checkpoint-{trainer.updates:06d}.pt", payload)
            atomic_checkpoint(output / "latest.pt", payload)
            print(json.dumps({"job": job, "updates": trainer.updates,
                              "development_final_pairs": metrics["macro_final_pair_accuracy"]}), flush=True)
        if not saved:
            save_point()
        while trainer.updates < total_updates(job):
            if device == "cuda":
                torch.cuda.synchronize()
            tick = time.monotonic()
            loss = trainer.step(families_at(job, trainer.updates))["loss"]
            if device == "cuda":
                torch.cuda.synchronize()
            seconds += time.monotonic() - tick
            if trainer.updates in checkpoints:
                save_point()
            elif trainer.updates % 300 == 0:
                load_protocol(directory)
                atomic_checkpoint(output / "latest.pt", snapshot())
                print(json.dumps({"job": job, "updates": trainer.updates, "loss": loss}), flush=True)
        load_protocol(directory)
        final = snapshot()
        validate_snapshot(final, protocol, job, evidence)
        atomic_checkpoint(output / "latest.pt", final)
        atomic_json(output / "report.json", {"schema": SCHEMA, "job": job, "protocol": protocol,
            "updates": trainer.updates, "family_microbatches": trainer.family_microbatches,
            "exposures": trainer.exposures, "training_seconds": seconds, "history": history,
            "invocation_seconds_after_preparation": time.monotonic() - started,
            "weights_sha256": checkpoint_digest(trainer.model), "checkpoint_file_sha256": file_hash(output / "latest.pt"),
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
            "heldout_evaluation_performed": False})


def audit(directory, device="cuda"):
    directory = Path(directory)
    protocol = load_protocol(directory)
    required = [directory / "main" / job / name for job in JOBS
                for name in ("latest.pt", "report.json", *(f"checkpoint-{step:06d}.pt" for step in sorted({0, *curve_steps(job)})))]
    if any(not path.is_file() for path in required):
        raise ValueError("all five completed candidates and curve checkpoints are required")
    files = {path.relative_to(directory).as_posix(): file_hash(path) for path in required}
    banks = torch.load(directory / "banks.pt", map_location="cpu", weights_only=True)
    evidence = SampleEvidence(banks["train"])
    # Validate all initializations and full optimizer endpoints before any held-out score.
    verifiers = {}
    initial_digest = checkpoint_digest(build_sequence_student(SEED, config=CONFIG))
    if initial_digest != protocol["initial_weights_sha256"]:
        raise ValueError("prescribed initialization differs from prospective digest")
    for job in JOBS:
        report = json.loads((directory / "main" / job / "report.json").read_text())
        saved = torch.load(directory / "main" / job / "latest.pt", map_location="cpu", weights_only=True)
        validate_snapshot(saved, protocol, job, evidence)
        if (saved["training"]["updates"] != total_updates(job) or report["updates"] != total_updates(job)
                or report["protocol"] != protocol or report["job"] != job or report["heldout_evaluation_performed"]
                or report["checkpoint_file_sha256"] != files[f"main/{job}/latest.pt"]):
            raise ValueError("fixed completed endpoint differs")
        admitted = {job.split("-", 1)[1]: banks["train"][job.split("-", 1)[1]]} if job.startswith("fresh-") else banks["train"]
        verifier = CompositionTrainer(admitted, config=CONFIG, payload=saved["training"])
        if checkpoint_digest(verifier.model) != report["weights_sha256"]:
            raise ValueError("endpoint report tensor digest differs")
        initial = torch.load(directory / "main" / job / "checkpoint-000000.pt", map_location="cpu", weights_only=True)
        validate_snapshot(initial, protocol, job, evidence)
        if initial["training"]["updates"] != 0:
            raise ValueError("initial checkpoint must precede optimization")
        verifier._restore(initial["training"])
        if checkpoint_digest(verifier.model) != initial_digest:
            raise ValueError("candidates must share the prescribed initialization")
        verifiers[job] = verifier
    prepared = {name: PreparedBank(rows, role="audit", config=CONFIG) for name, rows in banks["audit"].items()}
    output = directory / "audit"
    output.mkdir(parents=True, exist_ok=True)
    marker = {"schema": SCHEMA, "protocol_sha256": file_hash(directory / "protocol.json"), "input_files_sha256": files}
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
            verifier = verifiers.pop(job)
            model = build_sequence_student(SEED, device=device, config=CONFIG)
            curve = []
            started = time.monotonic()
            for step, exposure in zip(curve_steps(job), EXPOSURES):
                saved = torch.load(directory / "main" / job / f"checkpoint-{step:06d}.pt", map_location="cpu", weights_only=True)
                validate_snapshot(saved, protocol, job, evidence)
                if saved["training"]["updates"] != step:
                    raise ValueError("curve checkpoint step differs")
                verifier._restore(saved["training"])
                model.load_state_dict(saved["training"]["weights"], strict=True)
                selected = prepared if job == "joint" else {name: bank for name, bank in prepared.items()
                                                           if name.split("/")[1] == job.split("-", 1)[1]}
                curve.append({"updates": step, "held_episode_exposure": exposure,
                    "weights_sha256": checkpoint_digest(model), "metrics": evaluate_banks(model, selected)})
            final_metrics = curve[-1]["metrics"] if job == "joint" else evaluate_banks(model, prepared)
            retained_before = None
            if job.startswith("seq-"):
                saved = torch.load(directory / "main" / job / f"checkpoint-{OLD_PHASE:06d}.pt", map_location="cpu", weights_only=True)
                before = build_sequence_student(SEED, device=device, config=CONFIG)
                before.load_state_dict(saved["training"]["weights"], strict=True)
                retained_before = evaluate_banks(before, {name: bank for name, bank in prepared.items()
                                            if name.split("/")[1] != job.split("-", 1)[1]})
                del before
            controls = {name: evaluate_banks(model, selected, control=name) for name in ("blank", "reset")}
            result = {"schema": SCHEMA, "job": job, "inputs": marker, "curve": curve,
                "final": final_metrics, "retained_before_new_phase": retained_before, "controls": controls,
                "cpu_restart": verify_cpu_restart(model, banks["audit"][next(iter(selected))][0]),
                "seconds": time.monotonic() - started, "automatic_promotion": False}
            atomic_json(path, result)
            results[job] = result
            print(json.dumps({"audited": job}), flush=True)
            del verifier, model
        load_protocol(directory)
        if any(file_hash(directory / name) != sha for name, sha in files.items()):
            raise RuntimeError("candidate evidence changed during audit")
        atomic_json(output / "report.json", {"schema": SCHEMA, "protocol": protocol, "inputs": marker,
            "results": results, "base_checkpoints_unchanged": True, "automatic_promotion": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("prepare", "train", "audit"), required=True)
    parser.add_argument("--job", choices=JOBS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "prepare":
        protocol = prepare(args.output)
        print(json.dumps({"prepared": True, "source_files": len(protocol["source_sha256"])}))
    elif args.phase == "train":
        train(args.output, args.job, args.device)
    else:
        audit(args.output, args.device)


if __name__ == "__main__":
    main()
