"""Matched local study of reusable memory and cross-family learning order.

All families use raw English, zero numeric task channels and the same action /
byte-generation losses. There are no family-specific semantic readout heads.
The held-out family is never evaluated or rehearsed by this training runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from brain_in_computer.dialogue_student import (
    ACK, ByteCodec, _run_turn, _select, checkpoint_digest, encode_dialogues,
    evaluate_dialogues,
)
from brain_in_computer.learning_student import _cpu_copy, _check_finite_tree
from brain_in_computer.learning_loop import run_lock
from experiments.cognitive_curriculum import generate_cognitive, validate_cognitive
from experiments.cognitive_student import build_cognitive_student


SCHEMA = "bic-general-learning-v1"
TRAIN_FAMILIES = ("variable_binding", "graph_reachability", "arithmetic_updates")
HELDOUT_FAMILY = "conditional_logic"


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def fingerprint_rows(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def family_at(update: int, total: int, schedule: str) -> str:
    if schedule not in ("interleaved", "blocked") or total % len(TRAIN_FAMILIES):
        raise ValueError("schedule must have equal integer family budgets")
    if not 0 <= update < total:
        raise ValueError("update outside planned run")
    index = update % len(TRAIN_FAMILIES) if schedule == "interleaved" else update // (total // len(TRAIN_FAMILIES))
    return TRAIN_FAMILIES[index]


def make_banks(*, seed=2_000_000, count=512, split="train", levels=(1, 2), families=TRAIN_FAMILIES):
    return {family: [row for level in levels for row in generate_cognitive(
        seed + index * 100_000 + level * 10_000, count, split=split, family=family, level=level)]
        for index, family in enumerate(families)}


def exposure_hashes(seed, *, steps, batch_size, lessons):
    """Freeze each family's sample stream independently of curriculum order."""
    result = {}
    for index, family in enumerate(sorted(TRAIN_FAMILIES)):
        generator = torch.Generator().manual_seed(seed + index * 7919)
        digest = hashlib.sha256()
        for _ in range(steps // len(TRAIN_FAMILIES)):
            # Two levels, each with lessons rows; hence lessons complete pairs.
            pairs = torch.randint(lessons, (batch_size // 2,), generator=generator)
            digest.update(pairs.numpy().tobytes())
        result[family] = digest.hexdigest()
    return result


class CognitiveTrainer:
    """One optimizer; independent matched sampler stream for each task family."""

    def __init__(self, banks, *, seed=2201, memory_mode="episodic", device="cpu",
                 batch_size=64, learning_rate=.003, payload=None):
        if type(batch_size) is not int or batch_size < 2 or batch_size % 2:
            raise ValueError("batch size must contain complete pairs")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        self.banks, self.batch_size = banks, batch_size
        if not banks:
            raise ValueError("at least one training family is required")
        for family, rows in banks.items():
            if not rows or len(rows) % 2:
                raise ValueError("training banks require complete pairs")
            for row in rows:
                if row["split"] != "train" or row["family"] != family:
                    raise ValueError("only canonical train rows in the named family are admitted")
                validate_cognitive(row)
            for index in range(0, len(rows), 2):
                if rows[index]["counterfactual_group"] != rows[index + 1]["counterfactual_group"]:
                    raise ValueError("adjacent examples must be counterfactual pairs")
        self.model = build_cognitive_student(seed, device=device, memory_mode=memory_mode)
        self.recipe = {"banks": {key: fingerprint_rows(rows) for key, rows in banks.items()},
                       "memory_mode": memory_mode, "batch_size": batch_size, "learning_rate": learning_rate}
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.generators = {family: torch.Generator().manual_seed(seed + index * 7919)
                           for index, family in enumerate(sorted(banks))}
        self.updates = 0
        self.family_updates = {family: 0 for family in banks}
        if payload is not None:
            _check_finite_tree(payload, "training_checkpoint")
            if payload.get("recipe") != self.recipe:
                raise ValueError("checkpoint training recipe differs")
            self.model.load_state_dict(payload["weights"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer"])
            if set(payload["samplers"]) != set(self.generators):
                raise ValueError("checkpoint family sampler set differs")
            for family, generator in self.generators.items():
                generator.set_state(payload["samplers"][family])
            self.updates = payload["updates"]
            self.family_updates = dict(payload["family_updates"])
        self.encoded = {family: encode_dialogues(self.model, rows) for family, rows in banks.items()}

    def snapshot(self):
        return {"recipe": self.recipe, "weights": _cpu_copy(self.model.state_dict()),
                "optimizer": _cpu_copy(self.optimizer.state_dict()),
                "samplers": {family: generator.get_state() for family, generator in self.generators.items()},
                "updates": self.updates, "family_updates": dict(self.family_updates)}

    def step(self, family):
        if family not in self.encoded:
            raise ValueError("family not admitted into this training phase")
        device = next(self.model.parameters()).device
        pairs = torch.randint(len(self.banks[family]) // 2, (self.batch_size // 2,), generator=self.generators[family])
        indices = (pairs[:, None] * 2 + torch.arange(2)[None]).flatten().to(device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        state = None  # Never retain encoder graphs across optimizer updates.
        logits, labels, replies, observations = [], [], [], []
        for full_batch in self.encoded[family]:
            batch = _select(full_batch, indices)
            output, state = _run_turn(self.model, batch, state)
            logits.append(output["logits"][:, -1])
            labels.append(batch["targets"])
            replies.append(F.cross_entropy(output["language_logits"].flatten(0, 1),
                batch["reply_targets"].flatten(), ignore_index=ByteCodec.PAD))
            observations.append(F.cross_entropy(output["observation_language_logits"][:, :-1].flatten(0, 1),
                batch["text_ids"][:, 1:].flatten(), ignore_index=ByteCodec.PAD))
        logits, labels = torch.cat(logits), torch.cat(labels)
        query_losses = [F.cross_entropy(logits[labels == target], labels[labels == target])
                        for target in range(ACK) if bool((labels == target).any())]
        if not query_losses:
            raise ValueError("training episodes need a scored query")
        action = torch.stack(query_losses).mean()
        if bool((labels == ACK).any()):
            action = action + .25 * F.cross_entropy(logits[labels == ACK], labels[labels == ACK])
        reply = torch.stack(replies).mean()
        observation = torch.stack(observations).mean()
        loss = action + .1 * reply + .1 * observation
        if not torch.isfinite(loss):
            raise ValueError("nonfinite cognitive training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        self.family_updates[family] += 1
        return {"loss": loss.item(), "action_loss": action.item(), "reply_loss": reply.item(),
                "observation_language_loss": observation.item(), "family": family}


def assess(model, banks, *, replies=False, **controls):
    before = checkpoint_digest(model)
    results = {family: evaluate_dialogues(model, rows, score_replies=replies, **controls)
               for family, rows in banks.items()}
    if checkpoint_digest(model) != before:
        raise RuntimeError("evaluation changed model parameters")
    return {"per_family": results,
            "macro_query_accuracy": sum(row["query_accuracy"] for row in results.values()) / len(results),
            "macro_pair_accuracy": sum(row["counterfactual_accuracy"] or 0 for row in results.values()) / len(results)}


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "experiments/cognitive_student.py", root / "experiments/cognitive_curriculum.py",
             *sorted((root / "brain_in_computer").glob("*.py"))]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run(args):
    torch.set_num_threads(1)
    if args.steps < 3 or args.steps % 3 or args.checkpoint_every < 1:
        raise ValueError("planned updates must divide equally across three families")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("time budget must be finite and positive")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        protocol = {"schema": SCHEMA, "seed": args.seed, "memory_mode": args.memory,
            "schedule": args.schedule, "steps": args.steps, "batch_size": args.batch_size,
            "checkpoint_every": args.checkpoint_every, "max_seconds": args.max_seconds,
            "learning_rate": args.lr, "device": args.device, "training_families": TRAIN_FAMILIES,
            "heldout_family": HELDOUT_FAMILY, "train_levels": [1, 2], "train_bank_seed": 2_000_000,
            "train_count_per_family_level": args.lessons, "dev_seed": 4_000_000,
            "within_family_draw_sha256": exposure_hashes(args.seed, steps=args.steps,
                batch_size=args.batch_size, lessons=args.lessons),
            "source_sha256": source_hashes(), "tutor": "off", "semantic_auxiliary_heads": False,
            "inputs": "raw English bytes and zero numeric channels; explicit learned state",
            "objective": "equal query-class CE + .25 ACK CE + .1 reply byte CE + .1 observation next-byte CE",
            "comparison": "same initialization, per-family examples and optimizer updates; family order differs",
            "heldout_rule": "No heldout-family evaluation, selection, or replay in this runner",
            "safety_scope": "inert generated objects, graphs, quantities and switches; no external actions",
            "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        protocol = json.loads(json.dumps(protocol))
        path = directory / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
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
        dev = make_banks(seed=4_000_000, count=64, split="dev")
        manifest = {"training": {key: fingerprint_rows(rows) for key, rows in banks.items()},
                    "development": {key: fingerprint_rows(rows) for key, rows in dev.items()}}
        atomic_json(directory / "datasets.json", manifest)
        latest = directory / "latest.pt"
        saved = torch.load(latest, map_location="cpu", weights_only=True) if latest.exists() else None
        trainer = CognitiveTrainer(banks, seed=args.seed, memory_mode=args.memory, device=args.device,
            batch_size=args.batch_size, learning_rate=args.lr, payload=saved["training"] if saved else None)
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
            family = family_at(trainer.updates, args.steps, args.schedule)
            t = time.monotonic()
            last = trainer.step(family)
            chunk_seconds += time.monotonic() - t
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
                print(json.dumps({"mode": args.memory, "schedule": args.schedule, "updates": trainer.updates,
                    "seconds": row["training_seconds"], "loss": row["loss"],
                    "query": metrics["macro_query_accuracy"], "pairs": metrics["macro_pair_accuracy"],
                    "family_pairs": {k: v["counterfactual_accuracy"] for k, v in metrics["per_family"].items()}}), flush=True)
        training_seconds += chunk_seconds
        if source_hashes() != protocol["source_sha256"]:
            raise RuntimeError("source changed during training")
        payload = {"schema": SCHEMA, "protocol": protocol, "training": trainer.snapshot(), "history": history,
                   "wall_seconds": wall_seconds + time.monotonic() - started, "training_seconds": training_seconds}
        atomic_checkpoint(latest, payload)
        final = assess(trainer.model, dev, replies=True)
        atomic_json(directory / "report.json", {"protocol": protocol, "datasets": manifest, "history": history,
            "updates": trainer.updates, "family_updates": trainer.family_updates,
            "examples": trainer.updates * args.batch_size, "training_seconds": training_seconds,
            "wall_seconds": wall_seconds + time.monotonic() - started, "stop_reason": stop_reason,
            "parameters": sum(p.numel() for p in trainer.model.parameters()), "development": final,
            "checkpoint_sha256": checkpoint_digest(trainer.model), "heldout_evaluation_performed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory", choices=("episodic", "recurrent"), required=True)
    parser.add_argument("--schedule", choices=("interleaved", "blocked"), required=True)
    parser.add_argument("--seed", type=int, default=2201)
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
