"""Frozen matched study of a shared training-only decision-learning objective.

The final graph family is withheld throughout main optimization. Previous
experiments and source snapshots are immutable reference studies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_student import _cpu_copy
from brain_in_computer.learning_loop import run_lock
from experiments.cognitive_credit import CreditTrainer
from experiments.cognitive_curriculum import VERSION, generate_cognitive
from experiments.train_cognitive import (
    assess, atomic_checkpoint, atomic_json, fingerprint_rows,
    source_hashes as previous_source_hashes,
)


SCHEMA = "bic-credit-training-v1"
TRAIN_FAMILIES = ("variable_binding", "arithmetic_updates", "conditional_logic")
HELDOUT_FAMILY = "graph_reachability"


def make_banks(*, seed=20_000_000, count=512, split="train", levels=(1, 2), families=TRAIN_FAMILIES):
    return {family: [row for level in levels for row in generate_cognitive(
        seed + index * 100_000 + level * 10_000, count, split=split, family=family, level=level)]
        for index, family in enumerate(families)}


def exposure_hashes(seed, *, steps, batch_size, lessons):
    result = {}
    for index, family in enumerate(sorted(TRAIN_FAMILIES)):
        generator = torch.Generator().manual_seed(seed + index * 7919)
        digest = hashlib.sha256()
        for _ in range(steps // len(TRAIN_FAMILIES)):
            pairs = torch.randint(lessons, (batch_size // 2,), generator=generator)
            digest.update(pairs.numpy().tobytes())
        result[family] = digest.hexdigest()
    return result


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    result = previous_source_hashes()
    for name in ("experiments/cognitive_credit.py", "experiments/train_cognitive_credit.py"):
        result[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return result


def run(args):
    torch.set_num_threads(1)
    if args.steps < 3 or args.steps % 3 or args.checkpoint_every < 1:
        raise ValueError("planned updates must divide equally across three subjects")
    if args.batch_size < 2 or args.batch_size % 2 or args.lessons < 2 or args.lessons % 2:
        raise ValueError("batch and lesson sizes must contain complete pairs")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("time budget must be finite and positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        protocol = {"schema": SCHEMA, "seed": args.seed, "aux_weight": args.aux_weight,
            "memory_mode": "recurrent", "schedule": "interleaved", "steps": args.steps,
            "batch_size": args.batch_size, "checkpoint_every": args.checkpoint_every,
            "max_seconds": args.max_seconds, "learning_rate": args.lr, "device": args.device,
            "training_families": list(TRAIN_FAMILIES), "heldout_family": HELDOUT_FAMILY,
            "train_levels": [1, 2], "train_bank_seed": 20_000_000,
            "train_count_per_family_level": args.lessons, "dev_seed": 24_000_000,
            "within_family_draw_sha256": exposure_hashes(args.seed, steps=args.steps,
                batch_size=args.batch_size, lessons=args.lessons),
            "source_sha256": source_hashes(), "curriculum_version": VERSION,
            "tutor": "off", "semantic_auxiliary_heads": False,
            "decision_auxiliary": "training-only Linear(64,4) from final prefrontal activity; query-balanced CE",
            "inference": "unchanged regional action and byte decoder paths; auxiliary logits never select a response",
            "objective": "query-balanced CE + .25 ACK CE + .1 reply CE + .1 input next-byte CE + aux_weight * query-balanced auxiliary CE",
            "heldout_rule": "No graph-family evaluation, lesson, or score during main training",
            "comparison": "same initial parameters and exact exposure/order; only auxiliary loss coefficient differs",
            "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        path = directory / "protocol.json"
        if path.exists() and json.loads(path.read_text(encoding="utf8")) != protocol:
            raise ValueError("source/protocol changed; use a new run directory")
        atomic_json(path, protocol)
        root = Path(__file__).resolve().parents[1]
        for name, expected in protocol["source_sha256"].items():
            target = directory / "source" / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((root / name).read_bytes())
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError("frozen source snapshot mismatch")
        banks = make_banks(count=args.lessons)
        dev = make_banks(seed=24_000_000, count=64, split="dev")
        datasets = {"training": {key: fingerprint_rows(rows) for key, rows in banks.items()},
                    "development": {key: fingerprint_rows(rows) for key, rows in dev.items()}}
        atomic_json(directory / "datasets.json", datasets)
        latest = directory / "latest.pt"
        saved = torch.load(latest, map_location="cpu", weights_only=True) if latest.exists() else None
        if saved:
            if saved.get("schema") != SCHEMA or saved.get("protocol") != protocol:
                raise ValueError("saved checkpoint protocol differs from this run")
            updates = saved["training"]["updates"]
            if type(updates) is not int or not 0 <= updates <= args.steps:
                raise ValueError("saved update count lies outside this run")
            expected = {family: updates // 3 + int(index < updates % 3)
                        for index, family in enumerate(TRAIN_FAMILIES)}
            if saved["training"]["family_updates"] != expected:
                raise ValueError("saved family counts disagree with the interleaved schedule")
        trainer = CreditTrainer(banks, seed=args.seed, device=args.device, memory_mode="recurrent",
            batch_size=args.batch_size, learning_rate=args.lr, aux_weight=args.aux_weight,
            payload=saved["training"] if saved else None)
        history, wall_seconds, training_seconds = (saved["history"], saved["wall_seconds"], saved["training_seconds"]) if saved else ([], 0., 0.)
        if not saved:
            atomic_checkpoint(directory / "initial.pt", {"weights": _cpu_copy(trainer.model.state_dict())})
        started = time.monotonic()
        deadline = started + max(0., args.max_seconds - wall_seconds)
        last, chunk_seconds = {}, 0.
        stop_reason = "planned_updates"
        while trainer.updates < args.steps:
            if time.monotonic() >= deadline:
                stop_reason = "compute_budget"
                break
            family = TRAIN_FAMILIES[trainer.updates % len(TRAIN_FAMILIES)]
            begin = time.monotonic()
            last = trainer.step(family)
            chunk_seconds += time.monotonic() - begin
            if trainer.updates % args.checkpoint_every == 0 or trainer.updates == args.steps:
                metrics = assess(trainer.model, dev)
                training_seconds += chunk_seconds
                row = {"updates": trainer.updates, "examples": trainer.updates * args.batch_size,
                    "training_seconds": chunk_seconds, "development": metrics, **last}
                history.append(row)
                chunk_seconds = 0.
                payload = {"schema": SCHEMA, "protocol": protocol, "training": trainer.snapshot(),
                    "history": history, "wall_seconds": wall_seconds + time.monotonic() - started,
                    "training_seconds": training_seconds}
                if source_hashes() != protocol["source_sha256"]:
                    raise RuntimeError("source changed during training")
                atomic_checkpoint(latest, payload)
                atomic_checkpoint(directory / f"checkpoint-{trainer.updates:06d}.pt", payload)
                atomic_json(directory / "progress.json", {"updates": trainer.updates, "history": history})
                print(json.dumps({"seed": args.seed, "aux_weight": args.aux_weight,
                    "updates": trainer.updates, "seconds": row["training_seconds"], "loss": row["loss"],
                    "auxiliary_query_loss": row.get("auxiliary_query_loss"),
                    "query": metrics["macro_query_accuracy"], "pairs": metrics["macro_pair_accuracy"],
                    "family_pairs": {key: value["counterfactual_accuracy"] for key, value in metrics["per_family"].items()}}), flush=True)
        training_seconds += chunk_seconds
        if source_hashes() != protocol["source_sha256"]:
            raise RuntimeError("source changed during training")
        payload = {"schema": SCHEMA, "protocol": protocol, "training": trainer.snapshot(), "history": history,
            "wall_seconds": wall_seconds + time.monotonic() - started, "training_seconds": training_seconds}
        atomic_checkpoint(latest, payload)
        final = assess(trainer.model, dev, replies=True)
        # A fixed training-bank diagnostic distinguishes weak fitting from a
        # gap on withheld entity combinations. It cannot select the endpoint.
        training_fit = assess(trainer.model, {key: rows[:128] + rows[args.lessons:args.lessons + 128]
            for key, rows in banks.items()}) if args.lessons >= 128 else assess(trainer.model, banks)
        atomic_json(directory / "report.json", {"protocol": protocol, "datasets": datasets, "history": history,
            "updates": trainer.updates, "family_updates": trainer.family_updates,
            "examples": trainer.updates * args.batch_size, "training_seconds": training_seconds,
            "wall_seconds": wall_seconds + time.monotonic() - started, "stop_reason": stop_reason,
            "parameters": sum(parameter.numel() for parameter in trainer.model.parameters()),
            "development": final, "training_fit_diagnostic": training_fit,
            "checkpoint_sha256": checkpoint_digest(trainer.model), "heldout_evaluation_performed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aux-weight", type=float, choices=(0., .3), required=True)
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=300)
    parser.add_argument("--lessons", type=int, default=512)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--lr", type=float, default=.003)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
