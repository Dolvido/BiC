"""Two-seed, fixed-budget transfer of episodic examples into shared recall weights.

Run `train` then `evaluate`. The latter refuses to replace sealed results. Names
and identities are taught before testing; only their appearance seeds are held
out. Original v0.5 controller/vision weights remain frozen in every condition.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F

from brain_in_computer.associative import AssociativeMemory, distinct_patterns, render_glyph
from brain_in_computer.consolidation import ConsolidatedMemory, NeuralRecall, RecallConfig, replay_examples
from brain_in_computer.regional_memory import load_regional_checkpoint
from brain_in_computer.training import atomic_json
from experiments.train_regional_memory import RELATIONS, TEMPLATES, CELL_NAMES, relation_target, evaluate_memory

NAMES = ("dax", "wug", "toma", "kiki", "fep", "blick", "zup", "noba",
         "mivo", "pazz", "veko", "lumi", "sorp", "zindle", "quavo", "pelmet",
         "ruxin", "bondu", "yempa", "dorvek", "seltu", "navo", "pimble", "zoska",
         "vempi", "kuldo", "raspen", "tivna", "jorbu", "pondek", "lazmi", "falno")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def teach(encoder, names, patterns, seed):
    bank = AssociativeMemory(encoder)
    for index, (name, pattern) in enumerate(zip(names, patterns)):
        for view in range(4):
            bank.teach(name, render_glyph(pattern, seed + index * 4 + view))
    return bank


def fit(recall, names, targets, steps, seed, old=None):
    """Each arm gets 16 new targets/update; replay adds 16 old targets/update."""
    rng = random.Random(seed)
    replay_rng = random.Random(seed + 10000)
    optimizer = torch.optim.AdamW(recall.parameters(), lr=.002, weight_decay=.00001)
    started = time.perf_counter()
    history = []
    recall.train()
    for step in range(steps):
        indexes = rng.choices(range(len(names)), k=16)
        batch_names = [names[i] for i in indexes]
        batch_targets = targets[indexes]
        if old:
            old_names, old_targets = old
            old_indexes = replay_rng.choices(range(len(old_names)), k=16)
            batch_names += [old_names[i] for i in old_indexes]
            batch_targets = torch.cat((batch_targets, old_targets[old_indexes]))
        prediction = recall(batch_names)
        loss = (1 - (prediction * batch_targets).sum(-1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(recall.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == steps:
            history.append({"step": step + 1, "training_cosine_loss": float(loss.detach())})
    return {"seconds": time.perf_counter() - started, "updates": steps, "history": history}


def train(args):
    output = Path(args.output)
    if (output / "protocol.json").exists():
        raise ValueError("output contains a frozen protocol; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    agent = load_regional_checkpoint(args.regional)
    encoder_checkpoint = torch.load(args.encoder, weights_only=True)
    excluded = {p for key, values in encoder_checkpoint.items() if key.endswith("identities") for p in values}
    prior = json.loads(Path("runs/regional-memory-v05/protocol.json").read_text())
    excluded.update(p for values in prior["identity_partitions"].values() for p in values)
    sessions = []
    for seed in args.seeds:
        patterns = distinct_patterns(96, seed, excluded)
        excluded.update(patterns)
        sessions.append({"seed": seed, "names": list(NAMES), "taught_patterns": patterns[:32],
                         "distractor_patterns": patterns[32:], "teaching_seed": seed + 10000,
                         "sealed_view_seed": seed + 1000000, "sealed_spec_seed": seed + 2000000})
    protocol = {"schema": "bic-consolidation-experiment-v1", "arguments": vars(args),
                "runner_sha256": sha(__file__), "module_sha256": sha(Path(__file__).resolve().parents[1] / "brain_in_computer/consolidation.py"),
                "regional_sha256": sha(args.regional), "encoder_sha256": sha(args.encoder),
                "network": asdict(RecallConfig()), "sessions": sessions,
                "stages": ["16 taught names", "16 additional names, with or without original-example replay"],
                "selection": "Fixed final update in all stages; no development or sealed selection.",
                "training": "Four visual exemplars/name. Loss is cosine distance from shared byte/GRU/MLP output to an exemplar's frozen visual embedding. AdamW lr=.002, weight_decay=.00001; gradient clip 1.",
                "comparison": "Both second-stage arms receive 16 new examples/update. Rehearsal adds 16 old examples/update, so compute and total examples are not matched.",
                "sealed_controls": ["episodic", "with_replay", "without_replay", "initial", "erased"],
                "scope": "Identities and names are taught during this experiment. Appearance seeds are held out; procedural appearance families and relationship grammar are familiar. Exact known-name gating remains symbolic.",
                "sealed_used_for_tuning": False}
    atomic_json(output / "protocol.json", protocol)
    protocol_hash = sha(output / "protocol.json")
    reports = []
    for session in sessions:
        seed = session["seed"]
        folder = output / f"seed-{seed}"
        folder.mkdir()
        bank = teach(agent.encoder, NAMES, session["taught_patterns"], session["teaching_seed"])
        source_file = folder / "temporary-source-memory.json"
        bank.save(source_file)
        old = replay_examples(bank, NAMES[:16])
        new = replay_examples(bank, NAMES[16:])
        torch.manual_seed(seed)
        initial = NeuralRecall()
        student = copy.deepcopy(initial)
        ConsolidatedMemory(initial, agent.encoder, NAMES).save(folder / "initial.pt", {"protocol_sha256": protocol_hash})
        first = fit(student, *old, args.steps, seed + 1)
        ConsolidatedMemory(student, agent.encoder, NAMES[:16]).save(folder / "stage-one.pt", {"protocol_sha256": protocol_hash})
        row = {"seed": seed, "stage_one": first, "arms": {}, "recall_parameters": sum(p.numel() for p in student.parameters()),
               "episodic_target_floats": 32 * 4 * agent.encoder.config.embedding_size,
               "initial_sha256": sha(folder / "initial.pt"), "stage_one_sha256": sha(folder / "stage-one.pt")}
        for arm in ("with_replay", "without_replay"):
            child = copy.deepcopy(student)
            row["arms"][arm] = fit(child, *new, args.steps, seed + 2, old=old if arm == "with_replay" else None)
            path = folder / f"{arm}.pt"
            ConsolidatedMemory(child, agent.encoder, NAMES).save(path, {"protocol_sha256": protocol_hash, "arm": arm})
            row["arms"][arm]["checkpoint_sha256"] = sha(path)
            print(seed, arm, json.dumps(row["arms"][arm]), flush=True)
        del bank, old, new
        source_file.unlink()
        row["source_memory_file_deleted"] = not source_file.exists()
        reports.append(row)
    atomic_json(output / "training_report.json", {"protocol_sha256": protocol_hash, "sessions": reports,
                                                 "sealed_evaluated": False})


def sealed_specs(session):
    """All five case categories; valid actions and STOP each comprise one half."""
    rng = random.Random(session["sealed_spec_seed"])
    specs = []
    counter = 0
    for index, (name, query) in enumerate(zip(session["names"], session["taught_patterns"])):
        for repetition in range(2):
            for relation in RELATIONS:
                for category in ("valid", "valid", "valid", "valid", "absent", "ambiguous", "boundary", "unknown_label"):
                    chosen = relation if not (category == "boundary" and relation == "find") else "left"
                    eligible = [cell for cell in range(4) if (relation_target(cell, chosen) == 10) == (category == "boundary")]
                    reference = rng.choice(eligible)
                    objects = rng.sample(session["distractor_patterns"], 4)
                    if category != "absent":
                        objects[reference] = query
                    if category == "ambiguous":
                        objects[rng.choice([cell for cell in range(4) if cell != reference])] = query
                    target = relation_target(reference, chosen) if category == "valid" else 10
                    label = name if category != "unknown_label" else "untaught-" + name
                    specs.append({"name": label, "objects": objects, "known": category != "unknown_label",
                                  "object_seeds": [session["sealed_view_seed"] + counter * 4 + i for i in range(4)],
                                  "prompt": rng.choice(TEMPLATES["sealed"][chosen]).format(name=label),
                                  "relation": chosen, "category": category, "target": target,
                                  "reply": "cannot select." if target == 10 else f"selected {CELL_NAMES[target]}.",
                                  "stage": "old" if index < 16 else "new"})
                    counter += 1
    return specs


@torch.no_grad()
def score_data(memory, specs, embeddings):
    if isinstance(memory, ConsolidatedMemory):
        predictions = memory.recall([s["name"] for s in specs])
        scores = (embeddings * predictions[:, None]).sum(-1).clamp(-1, 1)
    else:
        scores = torch.stack([(embeddings[i] @ torch.stack(memory._entries[s["name"]]).T).max(-1).values
                              if s["name"] in memory.labels else torch.zeros(4) for i, s in enumerate(specs)]).clamp(-1, 1)
    known = torch.tensor([[float(s["name"] in memory.labels)] for s in specs])
    scores *= known
    return {"specs": specs, "scores": scores, "known": known, "targets": torch.tensor([s["target"] for s in specs])}


def worker(args):
    """Fresh process receives only images/instructions and frozen checkpoints."""
    from unittest.mock import patch
    torch.set_num_threads(1)
    agent = load_regional_checkpoint(args.regional)
    memory = ConsolidatedMemory.load(args.student, agent.encoder)
    batch = torch.load(args.inputs, weights_only=True)
    with patch.object(AssociativeMemory, "__init__", side_effect=AssertionError("episodic construction forbidden")), \
         patch.object(AssociativeMemory, "evidence", side_effect=AssertionError("episodic lookup forbidden")), \
         patch.object(AssociativeMemory, "teach", side_effect=AssertionError("reteaching forbidden")):
        rows = [memory.act(agent, text, pixels) for text, pixels in zip(batch["instructions"], batch["pixels"])]
    atomic_json(args.worker_output, {"pid": os.getpid(), "rows": rows, "episodic_access_forbidden": True})


def evaluate(args):
    output = Path(args.output)
    if (output / "final_evaluation.json").exists():
        raise ValueError("sealed result already exists; preserve it")
    protocol = json.loads((output / "protocol.json").read_text())
    options = protocol["arguments"]
    bindings = {options["regional"]: protocol["regional_sha256"], options["encoder"]: protocol["encoder_sha256"],
                __file__: protocol["runner_sha256"],
                str(Path(__file__).resolve().parents[1] / "brain_in_computer/consolidation.py"): protocol["module_sha256"]}
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError(f"frozen experiment dependency changed: {path}")
    training = json.loads((output / "training_report.json").read_text())
    if training["protocol_sha256"] != sha(output / "protocol.json"):
        raise ValueError("training report belongs to another protocol")
    for row in training["sessions"]:
        folder = output / f"seed-{row['seed']}"
        paths = {folder / "initial.pt": row["initial_sha256"], folder / "stage-one.pt": row["stage_one_sha256"]}
        paths.update({folder / f"{arm}.pt": values["checkpoint_sha256"] for arm, values in row["arms"].items()})
        for path, expected in paths.items():
            if sha(path) != expected:
                raise ValueError(f"frozen student checkpoint changed: {path}")
    agent = load_regional_checkpoint(options["regional"])
    result = {"protocol_sha256": sha(output / "protocol.json"), "sessions": [], "sealed_used_for_tuning": False}
    for session in protocol["sessions"]:
        seed = session["seed"]
        folder = output / f"seed-{seed}"
        specs = sealed_specs(session)
        pieces = []
        with torch.no_grad():
            for offset in range(0, len(specs), 32):
                group = specs[offset:offset + 32]
                pixels = torch.stack([torch.stack([render_glyph(p, s) for p, s in zip(row["objects"], row["object_seeds"])]) for row in group])
                pieces.append(agent.encoder(pixels.flatten(0, 1)).reshape(len(group), 4, -1))
        embeddings = torch.cat(pieces)
        row = {"seed": seed, "conditions": {}}
        for arm in protocol["sealed_controls"]:
            if arm == "episodic":
                memory = teach(agent.encoder, NAMES, session["taught_patterns"], session["teaching_seed"])
            else:
                memory = ConsolidatedMemory.load(folder / ("with_replay.pt" if arm == "erased" else f"{arm}.pt"), agent.encoder)
                if arm == "erased":
                    with torch.no_grad():
                        for parameter in memory.recall.parameters():
                            parameter.zero_()
            data = score_data(memory, specs, embeddings)
            row["conditions"][arm] = {}
            for stage in ("old", "new"):
                indexes = [i for i, s in enumerate(specs) if s["stage"] == stage]
                part = {"specs": [specs[i] for i in indexes], "scores": data["scores"][indexes],
                        "known": data["known"][indexes], "targets": data["targets"][indexes]}
                row["conditions"][arm][stage] = evaluate_memory(agent, part)
            print(seed, arm, {stage: values["action_accuracy"] for stage, values in row["conditions"][arm].items()}, flush=True)
            del memory
        # Neither a bank nor its pixels enter the fresh process; exact rendered
        # images are supplied, and procedural IDs/targets remain in this parent.
        chosen = [next(s for s in specs if s["name"] == name and s["category"] == "valid" and s["relation"] == "find") for name in NAMES]
        with tempfile.TemporaryDirectory(prefix="bic-consolidation-restart-") as directory:
            inputs, answers = Path(directory) / "inputs.pt", Path(directory) / "outputs.json"
            torch.save({"instructions": [s["prompt"] for s in chosen],
                        "pixels": torch.stack([torch.stack([render_glyph(p, v) for p, v in zip(s["objects"], s["object_seeds"])]) for s in chosen])}, inputs)
            started = time.perf_counter()
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--regional", str(Path(options["regional"]).resolve()),
                            "--student", str((folder / "with_replay.pt").resolve()), "--inputs", str(inputs), "--worker-output", str(answers)],
                           cwd=directory, check=True, timeout=90, capture_output=True, text=True)
            fresh = json.loads(answers.read_text())
        correct = sum(answer["action"] == spec["target"] for answer, spec in zip(fresh["rows"], chosen))
        row["fresh_process"] = {"parent_pid": os.getpid(), "worker_pid": fresh["pid"], "distinct_process": fresh["pid"] != os.getpid(),
                                "correct_actions": correct, "episodes": len(chosen), "seconds": time.perf_counter() - started,
                                "source_memory_file_absent": not (folder / "temporary-source-memory.json").exists(),
                                "episodic_construction_lookup_and_teaching_forbidden": fresh["episodic_access_forbidden"],
                                "inputs": "candidate RGB pixels and quoted natural-language instructions only",
                                "answers": fresh["rows"]}
        result["sessions"].append(row)
    result["limits"] = ["32 arbitrary names per training seed; two seeds. Names and object identities are trained, and only render seeds are held out.",
                        "Known-name rejection remains an exact symbolic set; no claim of open vocabulary language learning.",
                        "The regional controller and glyph encoder are frozen v0.5 components. This is transfer into shared neural recall weights, not transfer into the existing cortical-region weights.",
                        "Four fixed crops, five familiar relations and familiar procedural appearance families; no general vision or English.",
                        "Neural recall may use more storage than the episodic exemplar bank; no compression advantage is claimed.",
                        "Replay adds old examples and compute; equal new-example updates, not equal total compute."]
    atomic_json(output / "final_evaluation.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "worker"))
    parser.add_argument("--output", default="runs/consolidation-v06")
    parser.add_argument("--regional", default="runs/regional-memory-v05/checkpoint.pt")
    parser.add_argument("--encoder", default="runs/associative/encoder.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[6101, 6102])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--student")
    parser.add_argument("--inputs")
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if min(args.steps, args.threads) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("counts must be positive and seeds unique")
    torch.set_num_threads(args.threads)
    {"train": train, "evaluate": evaluate, "worker": worker}[args.command](args)


if __name__ == "__main__":
    main()
