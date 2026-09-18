"""Prospective, matched architecture study on raw-text cognitive curricula.

Calibration sees development banks only. Arithmetic remains withheld until
all main endpoints exist. This diagnostic is not a promotion of a new BiC.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_loop import run_lock
from brain_in_computer.learning_student import _cpu_copy
from experiments.cognitive_credit import CreditTrainer
from experiments.cognitive_curriculum import VERSION, generate_cognitive
from experiments.sequence_training import SequenceTrainer
from experiments.train_cognitive import atomic_checkpoint, atomic_json, fingerprint_rows

SCHEMA = "bic-sequence-architecture-study-v1"
TRAIN_FAMILIES = ("variable_binding", "graph_reachability", "conditional_logic")
HELDOUT_FAMILY = "arithmetic_updates"
ARCHITECTURES = ("recurrent", "episodic", "sequence")
LEARNING_RATES = (.0003, .001, .003)
CALIBRATION_SEED, MAIN_SEED = 2401, 2501
CALIBRATION_STEPS, MAIN_STEPS = 600, 3600


def make_banks(*, seed=32_000_000, count=512, split="train", levels=(1, 2), families=TRAIN_FAMILIES):
    return {family: [row for level in levels for row in generate_cognitive(
        seed + index * 100_000 + level * 10_000, count, split=split, family=family, level=level)]
        for index, family in enumerate(families)}


def make_trainer(architecture, banks, *, seed, device="cpu", batch_size=64,
                 learning_rate=.001, payload=None):
    if architecture not in ARCHITECTURES:
        raise ValueError("unknown study architecture")
    common = dict(seed=seed, device=device, batch_size=batch_size,
                  learning_rate=learning_rate, payload=payload)
    if architecture == "sequence":
        return SequenceTrainer(banks, **common)
    return CreditTrainer(banks, memory_mode=architecture, aux_weight=0., **common)


def evaluate(architecture, model, banks, *, replies=True, blank_text=False, reset_history=False):
    if architecture not in ARCHITECTURES:
        raise ValueError("unknown study architecture")
    if architecture == "sequence":
        from experiments.sequence_evaluation import evaluate_banks
        return evaluate_banks(model, banks, score_replies=replies, blank_text=blank_text,
                              reset_history=reset_history)
    from experiments.audit_cognitive_credit import score_banks
    return score_banks(model, banks, replies=replies, blank_text=blank_text,
                       reset_each_turn=reset_history)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = [str(path.relative_to(root)).replace("\\", "/")
             for path in sorted((root / "brain_in_computer").glob("*.py"))]
    names += ["experiments/" + name + ".py" for name in (
        "cognitive_curriculum", "cognitive_student", "train_cognitive",
        "cognitive_credit", "train_cognitive_credit", "audit_cognitive_credit",
        "audit_cognitive_transfer", "sequence_student", "sequence_data",
        "sequence_evaluation", "sequence_training", "sequence_study", "audit_sequence_study")]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def exposure_hashes(seed, *, steps, batch_size=64, lessons=512):
    result = {}
    for index, family in enumerate(sorted(TRAIN_FAMILIES)):
        generator = torch.Generator().manual_seed(seed + index * 7919)
        digest = hashlib.sha256()
        for _ in range(steps // len(TRAIN_FAMILIES)):
            digest.update(torch.randint(lessons, (batch_size // 2,), generator=generator).numpy().tobytes())
        result[family] = digest.hexdigest()
    return result


def prepare(directory):
    """Freeze all code and bank identities before any calibration optimization."""
    from experiments.audit_sequence_study import prepare as prepare_audit
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        protocol = {"schema": SCHEMA, "source_sha256": source_hashes(),
            "architectures": list(ARCHITECTURES), "learning_rates": list(LEARNING_RATES),
            "calibration_seed": CALIBRATION_SEED, "main_seed": MAIN_SEED,
            "calibration_steps": CALIBRATION_STEPS, "main_steps": MAIN_STEPS,
            "checkpoint_every": 600, "batch_size": 64, "train_count_per_family_level": 512,
            "train_seed": 32_000_000, "dev_seed": 33_000_000, "dev_count_per_family_level": 64,
            "training_families": list(TRAIN_FAMILIES), "heldout_family": HELDOUT_FAMILY,
            "train_levels": [1, 2], "curriculum_version": VERSION,
            "selection_rule": "Final calibration development mean of macro counterfactual accuracy and "
                "macro last-known-later-query accuracy; exact ties choose lower learning rate.",
            "calibration_draw_sha256": exposure_hashes(CALIBRATION_SEED, steps=CALIBRATION_STEPS),
            "main_draw_sha256": exposure_hashes(MAIN_SEED, steps=MAIN_STEPS),
            "objective": "query-class-balanced CE + .25 ACK CE + .1 per-turn reply CE + .1 per-turn input-byte CE",
            "schedule": "interleaved", "tutor": "off", "arithmetic_main_access": False,
            "interpretation": "Exploratory representation/routing comparison, one main seed; calibrated at "
                "a short fixed budget. Sequence and episodic have raw observation history; recurrent "
                "retains compressed state. Different architectures and context paths are a package, "
                "not an isolated causal mechanism. No general-intellect or deployment claim."}
        path = directory / "protocol.json"
        if path.exists():
            prior = json.loads(path.read_text(encoding="utf8"))
            if {key: value for key, value in prior.items() if key != "prepared_utc"} != protocol:
                raise ValueError("study code or protocol changed; use a new directory")
            prepare_audit(directory / "audit")
            return prior
        protocol["prepared_utc"] = datetime.now(timezone.utc).isoformat()
        banks, dev = make_banks(), make_banks(seed=33_000_000, count=64, split="dev")
        atomic_json(directory / "datasets.json", {
            "training": {key: fingerprint_rows(rows) for key, rows in banks.items()},
            "development": {key: fingerprint_rows(rows) for key, rows in dev.items()}})
        root = Path(__file__).resolve().parents[1]
        for name in protocol["source_sha256"]:
            target = directory / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
        # Audit preparation must also succeed before study admits a worker.
        prepare_audit(directory / "audit")
        atomic_json(path, protocol)
        return protocol


def load_protocol(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf8"))
    if protocol.get("schema") != SCHEMA or protocol["source_sha256"] != source_hashes():
        raise ValueError("study source or schema changed")
    for name, digest in protocol["source_sha256"].items():
        if hashlib.sha256((directory / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen study source changed")
    return protocol


def calibration_score(metrics):
    pair, later = metrics["macro_pair_accuracy"], metrics["macro_later_known_accuracy"]
    if pair is None or later is None or not all(math.isfinite(v) and 0 <= v <= 1 for v in (pair, later)):
        raise ValueError("selection needs finite paired and later-query scores")
    return (pair + later) / 2


def choose_rate(candidates):
    if len(candidates) != len(LEARNING_RATES) or sorted(row["learning_rate"] for row in candidates) != list(LEARNING_RATES):
        raise ValueError("all three unique frozen calibration rates are required")
    return min(candidates, key=lambda row: (-calibration_score(row["development"]), row["learning_rate"]))


def calibration_path(directory, architecture, rate):
    return Path(directory) / "calibration" / f"{architecture}-lr-{rate:g}"


def select_rates(directory, *, persist=True):
    directory = Path(directory)
    protocol = load_protocol(directory)
    with run_lock(directory) if persist else nullcontext():
        choices, evidence = {}, {}
        for architecture in ARCHITECTURES:
            candidates = []
            for rate in LEARNING_RATES:
                path = calibration_path(directory, architecture, rate)
                report = json.loads((path / "report.json").read_text(encoding="utf8"))
                if (report["updates"] != CALIBRATION_STEPS or report["stop_reason"] != "planned_updates"
                        or report["protocol"]["phase"] != "calibration"
                        or report["protocol"]["architecture"] != architecture
                        or report["protocol"]["learning_rate"] != rate
                        or report["protocol"]["study"] != protocol
                        or report["heldout_evaluation_performed"]):
                    raise ValueError("incomplete or mismatched calibration")
                candidates.append({"learning_rate": rate, "development": report["development"],
                    "score": calibration_score(report["development"]),
                    "report_sha256": hashlib.sha256((path / "report.json").read_bytes()).hexdigest()})
            chosen = choose_rate(candidates)
            choices[architecture] = chosen["learning_rate"]
            evidence[architecture] = candidates
        selected = {"schema": SCHEMA, "choices": choices, "calibration": evidence,
                    "rule": protocol["selection_rule"], "source_sha256": protocol["source_sha256"]}
        path = directory / "selection.json"
        if not persist and not path.exists():
            raise ValueError("select calibration rates before launching main workers")
        if path.exists() and json.loads(path.read_text(encoding="utf8")) != selected:
            raise ValueError("selection is immutable once written")
        if persist:
            atomic_json(path, selected)
        return selected


def validate_resume(saved, protocol):
    if saved.get("schema") != SCHEMA or saved.get("protocol") != protocol:
        raise ValueError("resume protocol differs")
    updates = saved["training"]["updates"]
    if type(updates) is not int or not 0 <= updates <= protocol["steps"]:
        raise ValueError("resume updates outside planned run")
    expected = {family: updates // 3 + int(index < updates % 3) for index, family in enumerate(TRAIN_FAMILIES)}
    if saved["training"]["family_updates"] != expected:
        raise ValueError("resume family counts differ from interleaved schedule")


def train_run(directory, architecture, *, phase, rate, device="cuda", max_seconds=3600.):
    if architecture not in ARCHITECTURES or phase not in ("calibration", "main") or rate not in LEARNING_RATES:
        raise ValueError("run outside frozen study grid")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("time budget must be finite and positive")
    torch.set_num_threads(1)
    directory = Path(directory)
    study = load_protocol(directory)
    seed, steps = (CALIBRATION_SEED, CALIBRATION_STEPS) if phase == "calibration" else (MAIN_SEED, MAIN_STEPS)
    if phase == "main":
        selected = select_rates(directory, persist=False)
        if selected["choices"][architecture] != rate:
            raise ValueError("main rate must match frozen calibration selection")
    path = calibration_path(directory, architecture, rate) if phase == "calibration" else directory / "main" / architecture
    path.mkdir(parents=True, exist_ok=True)
    with run_lock(path):
        protocol = {"schema": SCHEMA, "study": study, "phase": phase, "architecture": architecture,
            "seed": seed, "steps": steps, "learning_rate": rate, "device": device,
            "batch_size": 64, "checkpoint_every": 600, "max_seconds": max_seconds,
            "source_sha256": study["source_sha256"],
            "within_family_draw_sha256": study[f"{phase}_draw_sha256"],
            "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}
        protocol_path = path / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf8")) != protocol:
            raise ValueError("run protocol changed")
        atomic_json(protocol_path, protocol)
        latest = path / "latest.pt"
        saved = torch.load(latest, map_location="cpu", weights_only=True) if latest.exists() else None
        if saved:
            validate_resume(saved, protocol)
        banks, dev = make_banks(), make_banks(seed=33_000_000, count=64, split="dev")
        datasets = {"training": {key: fingerprint_rows(rows) for key, rows in banks.items()},
                    "development": {key: fingerprint_rows(rows) for key, rows in dev.items()}}
        if datasets != json.loads((directory / "datasets.json").read_text(encoding="utf8")):
            raise ValueError("training/development bank changed")
        atomic_json(path / "datasets.json", datasets)
        trainer = make_trainer(architecture, banks, seed=seed, device=device, learning_rate=rate,
                              payload=saved["training"] if saved else None)
        history, wall_seconds, training_seconds = (saved["history"], saved["wall_seconds"], saved["training_seconds"]) if saved else ([], 0., 0.)
        if not saved:
            atomic_checkpoint(path / "initial.pt", {"weights": _cpu_copy(trainer.model.state_dict())})
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started, chunk_seconds, last = time.monotonic(), 0., {}
        stop_reason = "planned_updates"
        while trainer.updates < steps:
            if time.monotonic() - started + wall_seconds >= max_seconds:
                stop_reason = "compute_budget"
                break
            family = TRAIN_FAMILIES[trainer.updates % 3]
            begin = time.monotonic()
            last = trainer.step(family)
            chunk_seconds += time.monotonic() - begin
            if trainer.updates % 600 == 0:
                metrics = evaluate(architecture, trainer.model, dev, replies=False)
                training_seconds += chunk_seconds
                row = {"updates": trainer.updates, "examples": trainer.updates * 64,
                       "training_seconds": chunk_seconds, "development": metrics, **last}
                history.append(row)
                chunk_seconds = 0.
                payload = {"schema": SCHEMA, "protocol": protocol, "training": trainer.snapshot(),
                    "history": history, "wall_seconds": wall_seconds + time.monotonic() - started,
                    "training_seconds": training_seconds}
                if source_hashes() != study["source_sha256"]:
                    raise RuntimeError("source changed during training")
                atomic_checkpoint(latest, payload)
                atomic_checkpoint(path / f"checkpoint-{trainer.updates:06d}.pt", payload)
                atomic_json(path / "progress.json", {"updates": trainer.updates, "history": history})
                print(json.dumps({"architecture": architecture, "phase": phase, "lr": rate,
                    "updates": trainer.updates, "training_seconds": row["training_seconds"],
                    "loss": row["loss"], "query": metrics["macro_query_accuracy"],
                    "pairs": metrics["macro_pair_accuracy"], "later": metrics["macro_later_known_accuracy"]}), flush=True)
        training_seconds += chunk_seconds
        if source_hashes() != study["source_sha256"]:
            raise RuntimeError("source changed during training")
        atomic_checkpoint(latest, {"schema": SCHEMA, "protocol": protocol, "training": trainer.snapshot(),
            "history": history, "wall_seconds": wall_seconds + time.monotonic() - started,
            "training_seconds": training_seconds})
        final = evaluate(architecture, trainer.model, dev, replies=True)
        fit = evaluate(architecture, trainer.model,
            {key: rows[:128] + rows[512:640] for key, rows in banks.items()}, replies=False) if phase == "main" else None
        atomic_json(path / "report.json", {"protocol": protocol, "datasets": datasets, "history": history,
            "updates": trainer.updates, "family_updates": trainer.family_updates,
            "examples": trainer.updates * 64, "training_seconds": training_seconds,
            "wall_seconds": wall_seconds + time.monotonic() - started, "stop_reason": stop_reason,
            "parameters": sum(parameter.numel() for parameter in trainer.model.parameters()),
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
            "peak_cuda_scope": "Entire current worker invocation, including evaluation; regional evaluates "
                "whole banks whereas sequence uses paired chunks of 64. Not matched training-only VRAM.",
            "development": final, "training_fit_diagnostic": fit,
            "checkpoint_sha256": checkpoint_digest(trainer.model), "heldout_evaluation_performed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", required=True, choices=("prepare", "calibration", "select", "main"))
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-seconds", type=float, default=3600.)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "prepare":
        print(json.dumps(prepare(args.output)))
    elif args.phase == "select":
        print(json.dumps(select_rates(args.output)["choices"]))
    else:
        if args.architecture is None:
            parser.error("training requires --architecture")
        rates = LEARNING_RATES if args.phase == "calibration" else (select_rates(args.output, persist=False)["choices"][args.architecture],)
        for rate in rates:
            train_run(args.output, args.architecture, phase=args.phase, rate=rate,
                      device=args.device, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
