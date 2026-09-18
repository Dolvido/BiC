"""Frozen-protocol regional association and replay experiment for BiC v0.5.

Procedural identities/relations below generate teacher labels and score outcomes;
only rendered images, the user's utterance and pixel-derived memory evidence are
policy inputs. The existing glyph encoder is frozen throughout this experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.nn import functional as F

from brain_in_computer.associative import distinct_patterns, render_glyph, load_encoder_checkpoint
from brain_in_computer.training import atomic_json

RELATIONS = ("find", "left", "right", "above", "below")
CASE_TYPES = ("valid", "unknown_label", "absent", "ambiguous", "boundary")
CELL_NAMES = ("top left", "top right", "bottom left", "bottom right")
NAMES = {
    "train": ("dax", "wug", "toma", "kiki", "fep", "blick", "zup", "noba"),
    "development": ("mivo", "tulip", "pazz", "veko", "lumi", "sorp"),
    "sealed": ("zindle", "quavo", "pelmet", "ruxin", "bondu", "yempa", "dorvek", "seltu"),
}
TEMPLATES = {
    "train": {
        "find": ('find "{name}"', 'click "{name}"', 'select the object called "{name}"', 'please select "{name}"'),
        "left": ('click the object left of "{name}"', 'select the object to the left of "{name}"', 'find the object left of "{name}"', 'please click left of "{name}"'),
        "right": ('click the object right of "{name}"', 'select the object to the right of "{name}"', 'find the object right of "{name}"', 'please click right of "{name}"'),
        "above": ('click the object above "{name}"', 'select the object above "{name}"', 'find the object above "{name}"', 'please click above "{name}"'),
        "below": ('click the object below "{name}"', 'select the object below "{name}"', 'find the object below "{name}"', 'please click below "{name}"'),
    },
    "development": {
        "find": ('please find "{name}"', 'could you select "{name}"'),
        "left": ('please find the object left of "{name}"', 'could you click the object left of "{name}"'),
        "right": ('please find the object right of "{name}"', 'could you click the object right of "{name}"'),
        "above": ('please find the object above "{name}"', 'could you click the object above "{name}"'),
        "below": ('please find the object below "{name}"', 'could you click the object below "{name}"'),
    },
    "sealed": {
        "find": ('would you find "{name}"', 'please click the object called "{name}"', 'could you find "{name}"'),
        "left": ('would you select the object left of "{name}"', 'please choose the object left of "{name}"', 'could you select the object to the left of "{name}"'),
        "right": ('would you select the object right of "{name}"', 'please choose the object right of "{name}"', 'could you select the object to the right of "{name}"'),
        "above": ('would you select the object above "{name}"', 'please choose the object above "{name}"', 'could you select the object above "{name}"'),
        "below": ('would you select the object below "{name}"', 'please choose the object below "{name}"', 'could you select the object below "{name}"'),
    },
}


def relation_target(reference: int, relation: str) -> int:
    if relation == "find":
        return reference
    row, col = divmod(reference, 2)
    dr, dc = {"left": (0, -1), "right": (0, 1), "above": (-1, 0), "below": (1, 0)}[relation]
    row, col = row + dr, col + dc
    return 2 * row + col if 0 <= row < 2 and 0 <= col < 2 else 10


def partitions(encoder_checkpoint: str, seed: int) -> dict:
    previous = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
    excluded = sorted(set(p for key, values in previous.items() if key.endswith("identities") for p in values))
    result = {"excluded_v04_identities": excluded}
    occupied = list(excluded)
    for index, (split, count) in enumerate((("train", 512), ("development", 128), ("sealed", 256))):
        result[split] = distinct_patterns(count, seed + index, occupied)
        occupied.extend(result[split])
    return result


def build_specs(patterns: list[int], split: str, seed: int, count: int) -> list[dict]:
    """Exactly half valid; other half evenly split over four STOP causes."""
    if count % 40:
        raise ValueError("case count must be divisible by 40 for balance")
    rng = random.Random(seed)
    specs = []
    for index in range(count):
        category = ("valid",) * 4 + CASE_TYPES[1:]
        kind = category[index % 8]
        relation = RELATIONS[(index // 8) % 5]
        if kind == "boundary" and relation == "find":
            relation = RELATIONS[1 + (index // 40) % 4]
        query, *distractors = rng.sample(patterns, 5)
        eligible = [cell for cell in range(4) if relation_target(cell, relation) < 4]
        if kind == "boundary":
            eligible = [cell for cell in range(4) if relation_target(cell, relation) == 10]
        reference = eligible[(index // 40) % len(eligible)] if kind in ("valid", "boundary") else rng.randrange(4)
        objects = distractors[:4]
        if kind != "absent":
            objects[reference] = query
        if kind == "ambiguous":
            other = rng.choice([cell for cell in range(4) if cell != reference])
            objects[other] = query
        name = rng.choice(NAMES[split])
        target = relation_target(reference, relation) if kind == "valid" else 10
        specs.append({"query": query, "objects": objects, "support_seed": rng.randrange(2**31),
                      "object_seeds": [rng.randrange(2**31) for _ in range(4)],
                      "known": kind != "unknown_label", "relation": relation, "category": kind,
                      "name": name, "prompt": rng.choice(TEMPLATES[split][relation]).format(name=name),
                      "target": target, "reply": "cannot select." if target == 10 else f"selected {CELL_NAMES[target]}.",
                      "reference": reference})
    rng.shuffle(specs)
    return specs


@torch.no_grad()
def encode_specs(specs: list[dict], encoder) -> dict:
    """Cache only sensory computations; no teacher target enters the builder."""
    score_chunks = []
    for offset in range(0, len(specs), 64):
        group = specs[offset:offset + 64]
        supports = torch.stack([render_glyph(s["query"], s["support_seed"]) for s in group])
        crops = torch.stack([torch.stack([render_glyph(p, seed) for p, seed in zip(s["objects"], s["object_seeds"])]) for s in group])
        query = encoder(supports)
        candidates = encoder(crops.flatten(0, 1)).reshape(len(group), 4, -1)
        scores = (candidates * query[:, None]).sum(-1).clamp(-1, 1)
        score_chunks.append(scores)
    scores = torch.cat(score_chunks)
    known = torch.tensor([[float(s["known"])] for s in specs])
    return {"scores": scores, "known": known,
            "targets": torch.tensor([s["target"] for s in specs]), "specs": specs}


def normalized_prompts(specs: list[dict], rng=None) -> list[str]:
    from brain_in_computer.regional_memory import normalize_instruction
    prompts = [s["prompt"] if rng is None else rng.choice(TEMPLATES["train"][s["relation"]]).format(name=s["name"]) for s in specs]
    return [normalize_instruction(prompt)[1] for prompt in prompts]


def byte_loss(prediction, decoder, targets):
    from brain_in_computer.language import ByteCodec
    active = targets != ByteCodec.PAD
    weights = torch.ones_like(targets, dtype=prediction.dtype)
    weights[:, 0] = 8
    weights[decoder == ord(" ") + ByteCodec.BYTE_OFFSET] = 8
    losses = F.cross_entropy(prediction[active], targets[active], reduction="none")
    return (losses * weights[active]).sum() / weights[active].sum()


@torch.no_grad()
def evaluate_memory(agent, data, *, control="intact", batch_size=64, generate=True):
    agent.eval()
    device = next(agent.parameters()).device
    predictions, replies = [], []
    for offset in range(0, len(data["specs"]), batch_size):
        specs = data["specs"][offset:offset + batch_size]
        scores = data["scores"][offset:offset + batch_size].clone().to(device)
        known = data["known"][offset:offset + batch_size].clone().to(device)
        prompts = normalized_prompts(specs)
        ablate = ()
        blank_memory = control == "blank_memory"
        if control == "blank_language":
            prompts = [""] * len(specs)
        elif control == "hippocampus_lesion":
            ablate = ("hippocampus",)
        elif control == "temporal_language_lesion":
            ablate = ("temporal_language",)
        elif control == "zero_encoder":
            scores.zero_()
        elif control == "shuffled_memory":
            # A fixed derangement breaks the score-to-cell correspondence while
            # preserving each example's score distribution and known-label bit.
            scores = scores[:, [2, 3, 0, 1]]
        output = agent.forward_evidence(scores, known, prompts, ablate=ablate, blank_memory=blank_memory)
        predictions.extend(output["logits"][:, -1].argmax(-1).tolist())
        if generate:
            replies.extend(agent.respond_evidence(scores, known, prompts, ablate=ablate, blank_memory=blank_memory))
    correct = [prediction == s["target"] for prediction, s in zip(predictions, data["specs"])]
    reply_ok = [reply.strip() == s["reply"] for reply, s in zip(replies, data["specs"])] if generate else []
    result = {"episodes": len(correct), "control": control, "action_accuracy": sum(correct) / len(correct),
              "reply_exact": sum(reply_ok) / len(reply_ok) if reply_ok else None,
              "joint_success": sum(a and b for a, b in zip(correct, reply_ok)) / len(correct) if reply_ok else None,
              "all_stop_accuracy": sum(s["target"] == 10 for s in data["specs"]) / len(correct),
              "categories": {}, "relations": {}, "failure_examples": []}
    for field, values in (("category", CASE_TYPES), ("relation", RELATIONS)):
        destination = result["categories" if field == "category" else "relations"]
        for value in values:
            indexes = [i for i, s in enumerate(data["specs"]) if s[field] == value]
            destination[value] = {"episodes": len(indexes), "action_accuracy": sum(correct[i] for i in indexes) / len(indexes),
                                  "reply_exact": sum(reply_ok[i] for i in indexes) / len(indexes) if reply_ok else None}
    for index, (ok, spec) in enumerate(zip(correct, data["specs"])):
        if not ok and len(result["failure_examples"]) < 12:
            result["failure_examples"].append({"prompt": spec["prompt"], "category": spec["category"],
                                               "expected_action": spec["target"], "action": predictions[index],
                                               "reply": replies[index] if replies else None,
                                               "cosine_scores": data["scores"][index].tolist()})
    return result


def replay_loss(computer, actions, replies, rng, batch_size, device):
    from experiments.advance_grounding import render_prompt
    # Uniform action-class sampling protects rare keys and explicit STOP.
    action_groups = actions
    half = batch_size // 2
    first = [rng.choice(rng.choice(action_groups)) for _ in range(half)]
    second = [rng.choice(rng.choice(replies)) for _ in range(batch_size - half)]
    rows = first + second
    pixels = torch.stack([s["pixels"] for s in rows]).to(device, dtype=torch.float32).div_(255)
    body = torch.stack([s["body"] for s in rows]).to(device)
    prompts = [render_prompt(s["task"], s["target"], rng, "train") for s in rows]
    decoder, target_bytes = computer.prepare_reply([s["reply"] or "" for s in rows])
    target_bytes[:half].zero_()
    result = computer(pixels, body, prompts, decoder)
    action = F.cross_entropy(result["logits"][:half, -1], torch.tensor([s["action"] for s in first], device=device))
    language = byte_loss(result["language_logits"], decoder, target_bytes)
    scene_targets = torch.tensor([s["scene"] for s in rows], device=device)
    scene = torch.stack([F.cross_entropy(part, scene_targets[:, i]) for i, part in enumerate(result["scene_logits"].split((4, 4, 4, 4, 5, 21), -1))]).mean()
    return action + 3 * language + .3 * scene


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_arm(args, label, train_data, development, replay, protocol_hash):
    from brain_in_computer.regional_memory import load_regional_checkpoint, save_regional_checkpoint
    from experiments.advance_grounding import evaluate
    torch.manual_seed(args.seed)
    agent = load_regional_checkpoint(Path(args.output) / "initial-checkpoint.pt", args.device)
    optimizer = torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad], lr=args.lr, weight_decay=.0001)
    sampler = random.Random(args.seed + 30)
    phrase_rng = random.Random(args.seed + 31)
    replay_rng = random.Random(args.seed + 32)
    output = Path(args.output) / label
    output.mkdir(parents=True, exist_ok=True)
    history, best_score, best_step = [], -1, 0
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        agent.train()
        agent.encoder.eval()
        indexes = sampler.choices(range(len(train_data["specs"])), k=args.batch_size)
        specs = [train_data["specs"][i] for i in indexes]
        scores = train_data["scores"][indexes].to(args.device)
        known = train_data["known"][indexes].to(args.device)
        targets = train_data["targets"][indexes].to(args.device)
        decoder, target_bytes = agent.computer.prepare_reply([s["reply"] for s in specs])
        optimizer.zero_grad(set_to_none=True)
        prediction = agent.forward_evidence(scores, known, normalized_prompts(specs, phrase_rng), decoder)
        action_loss = F.cross_entropy(prediction["logits"][:, -1], targets)
        language_loss = byte_loss(prediction["language_logits"], decoder, target_bytes)
        new_loss = action_loss + 3 * language_loss
        old_loss = replay_loss(agent.computer, *replay, replay_rng, args.replay_batch_size, args.device) if label == "with_replay" else new_loss.new_zeros(())
        loss = new_loss + args.replay_weight * old_loss
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in agent.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == args.steps:
            row = {"step": step, "elapsed_seconds": time.perf_counter() - started, "loss": float(loss.detach()),
                   "new_action_loss": float(action_loss.detach()), "new_language_loss": float(language_loss.detach()),
                   "replay_loss": float(old_loss.detach())}
            history.append(row)
            print(label, json.dumps(row), flush=True)
        if step % args.evaluate_every == 0 or step == args.steps:
            new_eval = evaluate_memory(agent, development)
            old_eval = evaluate(agent.computer, seed=args.seed + 500, per_task=32, split="development")
            score = (new_eval["action_accuracy"] + old_eval["macro_joint_success"]) / 2
            row = {"step": step, "new_tasks": new_eval, "old_desktop": old_eval, "selection_score": score}
            atomic_json(output / f"development-{step}.json", row)
            print(f"{label} dev {step}: new={new_eval['action_accuracy']:.4f}, old={old_eval['macro_joint_success']:.4f}, combined={score:.4f}", flush=True)
            if score > best_score:
                best_score, best_step = score, step
                save_regional_checkpoint(output / "checkpoint.pt", agent, metadata={"arm": label, "selected_step": step, "protocol_sha256": protocol_hash})
    # A common final update is retained for the matched-update replay contrast.
    # Selected checkpoints are separate so development selection cannot silently
    # make the comparison use different numbers of new-task updates.
    save_regional_checkpoint(output / "final-checkpoint.pt", agent, metadata={"arm": label, "step": args.steps, "protocol_sha256": protocol_hash})
    report = {"arm": label, "new_task_updates": args.steps, "new_examples_per_update": args.batch_size,
              "replay_examples_per_update": args.replay_batch_size if label == "with_replay" else 0,
              "selected_step": best_step, "development_selection_score": best_score, "history": history,
              "training_and_development_seconds": time.perf_counter() - started,
              "parameter_count": sum(p.numel() for p in agent.parameters()),
              "trainable_parameter_count": sum(p.numel() for p in agent.parameters() if p.requires_grad),
              "checkpoint_sha256": sha256(output / "checkpoint.pt"),
              "final_checkpoint_sha256": sha256(output / "final-checkpoint.pt")}
    atomic_json(output / "report.json", report)
    return report


def train(args):
    from brain_in_computer.computer_use.environment import TASKS
    from brain_in_computer.computer_use.training import load_computer_checkpoint
    from brain_in_computer.regional_memory import RegionalMemoryAgent, save_regional_checkpoint
    from experiments.advance_grounding import build_curriculum
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("output already has a frozen protocol; use a new output directory")
    identities = partitions(args.encoder, args.seed + 100)
    protocol = {"experiment": "regional memory and old-task rehearsal v0.5", "arguments": vars(args),
                "runner_sha256": sha256(__file__), "initialize_sha256": sha256(args.initialize), "encoder_sha256": sha256(args.encoder),
                "identity_partitions": identities, "templates": TEMPLATES, "label_names": NAMES,
                "case_mix": {"valid": .5, "unknown_label": .125, "absent": .125, "ambiguous": .125, "boundary": .125},
                "seeds": {"training_cases": args.seed + 200, "development_cases": args.seed + 201, "sealed_cases": args.seed + 202},
                "counts": {"train": 3200, "development": 640, "sealed": 1280},
                "selection": "Earliest maximum of mean(new-development action accuracy, old-v04-development joint accuracy), every evaluate_every updates; no sealed data opened during training.",
                "retention_contrast": "Both final update checkpoints evaluated on matched old tasks and new tasks; replay adds extra old examples/compute, not extra new-task updates. One training seed per arm.",
                "evidence": "Four raw pixel-embedding cosine similarities plus whether exact quoted label exists; label masked as <name> before neural comprehension. No relation, threshold decision, target index, identity or action supplied as an observation.",
                "biological_scope": "Fast explicit exemplar binding provides learned evidence to a neural hippocampal pathway; rehearsal imitates a role of experience replay. This is not hippocampal anatomy, offline sleep or transfer from episodic memory to cortical storage.",
                "sealed_controls": ["blank_memory", "hippocampus_lesion", "temporal_language_lesion", "blank_language", "zero_encoder", "shuffled_memory"],
                "sealed_used_for_selection": False}
    atomic_json(output / "protocol.json", protocol)
    protocol_hash = sha256(output / "protocol.json")
    print(f"Protocol frozen: {protocol_hash}", flush=True)
    torch.manual_seed(args.seed)
    initial_agent = RegionalMemoryAgent(load_computer_checkpoint(args.initialize), load_encoder_checkpoint(args.encoder))
    save_regional_checkpoint(output / "initial-checkpoint.pt", initial_agent, metadata={"protocol_sha256": protocol_hash, "role": "identical initialization for both arms"})
    del initial_agent
    started = time.perf_counter()
    encoder = load_encoder_checkpoint(args.encoder)
    training = encode_specs(build_specs(identities["train"], "train", args.seed + 200, 3200), encoder)
    development = encode_specs(build_specs(identities["development"], "development", args.seed + 201, 640), encoder)
    actions, replies, _ = build_curriculum(args.seed + 300, 3, variations=2, other_episodes=100)
    replay = ([ [s for s in actions if s["action"] == action] for action in range(11)],
              [ [s for s in replies if s["task"] == task] for task in TASKS])
    data_seconds = time.perf_counter() - started
    print(f"Precomputed training and development evidence in {data_seconds:.1f}s; sealed cases not rendered.", flush=True)
    reports = {}
    for arm in ("with_replay", "without_replay"):
        reports[arm] = train_arm(args, arm, training, development, replay, protocol_hash)
    atomic_json(output / "training_report.json", {"protocol_sha256": protocol_hash, "data_preparation_seconds": data_seconds,
                                                 "arms": reports, "sealed_evaluated": False})
    print("Both arms selected on development; run evaluate once for sealed comparisons.", flush=True)


def final_evaluation(args):
    from brain_in_computer.computer_use.training import load_computer_checkpoint, evaluate_computer
    from brain_in_computer.regional_memory import load_regional_checkpoint
    from experiments.advance_grounding import evaluate
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    if (output / "final_evaluation.json").exists():
        raise ValueError("sealed result already exists; preserve it instead of silently repeating evaluation")
    protocol = json.loads((output / "protocol.json").read_text())
    training = json.loads((output / "training_report.json").read_text())
    options = protocol["arguments"]
    encoder = load_encoder_checkpoint(options["encoder"])
    sealed = encode_specs(build_specs(protocol["identity_partitions"]["sealed"], "sealed", protocol["seeds"]["sealed_cases"], protocol["counts"]["sealed"]), encoder)
    results = {"protocol_sha256": sha256(output / "protocol.json"), "sealed_used_for_selection": False,
               "matched_updates": {}, "development_selected": {}, "controls": {}}
    baseline = load_computer_checkpoint(options["initialize"], args.device)
    results["baseline_v04"] = {"checkpoint_sha256": sha256(options["initialize"]),
                               "old_familiar": evaluate_computer(baseline, seed=options["seed"] + 700, episodes_per_task=64, split="train"),
                               "old_sealed_phrasing": evaluate(baseline, seed=options["seed"] + 701, per_task=64, split="sealed")}
    for arm in ("with_replay", "without_replay"):
        for checkpoint_name, result_key in (("final-checkpoint.pt", "matched_updates"), ("checkpoint.pt", "development_selected")):
            path = output / arm / checkpoint_name
            agent = load_regional_checkpoint(path, args.device)
            row = {"checkpoint_sha256": sha256(path), "new_sealed": evaluate_memory(agent, sealed),
                   "old_familiar": evaluate_computer(agent.computer, seed=options["seed"] + 700, episodes_per_task=64, split="train"),
                   "old_sealed_phrasing": evaluate(agent.computer, seed=options["seed"] + 701, per_task=64, split="sealed")}
            results[result_key][arm] = row
            print(result_key, arm, "new", row["new_sealed"]["action_accuracy"], "old", row["old_sealed_phrasing"]["macro_joint_success"], flush=True)
    selected = load_regional_checkpoint(output / "with_replay" / "checkpoint.pt", args.device)
    for control in protocol["sealed_controls"]:
        results["controls"][control] = evaluate_memory(selected, sealed, control=control)
        print("control", control, results["controls"][control]["action_accuracy"], flush=True)
    results["limits"] = [
        "A two-by-two board, five instructed relationships, and one click or STOP; no general computer interaction, object detection, or open-ended English.",
        "Glyph identities and normalized prompt templates are disjoint across train/development/sealed; all relations, positions and appearance-generation families occur in training.",
        "Quoted labels use exact symbolic exemplar lookup; masked new label names do not demonstrate neural vocabulary acquisition.",
        "Only one training seed per arm; replay consumes extra examples and compute. The comparison controls new-task updates, not total compute.",
        "The visual encoder remains frozen; the recurrent regional controller and language networks learn to use its retrieval scores.",
        "Ablations are acute interventions without retraining; they demonstrate dependence in this fitted system, not biological equivalence.",
        "Procedural teacher identities/relation labels are used to render inputs and score targets only; neither reaches policy observations.",
    ]
    atomic_json(output / "final_evaluation.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--output", default="runs/regional-memory-v05")
    parser.add_argument("--initialize", default="runs/grounding-v04/checkpoint.pt")
    parser.add_argument("--encoder", default="runs/associative/encoder.pt")
    parser.add_argument("--seed", type=int, default=5051)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--replay-batch-size", type=int, default=16)
    parser.add_argument("--evaluate-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=.0004)
    parser.add_argument("--replay-weight", type=float, default=1.)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.replay_batch_size, args.evaluate_every, args.threads) < 1 or args.replay_batch_size < 2:
        parser.error("positive counts and replay batch of at least two required")
    (train if args.command == "train" else final_evaluation)(args)


if __name__ == "__main__":
    main()
