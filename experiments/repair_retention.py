"""Targeted rehearsal of known regressions, with fresh evaluation partitions.

The earlier v0.4/v0.5 wording tests are development evidence now. They may be
rehearsed and are reported as restoration diagnostics, never as new held-out
proof. New desktop phrases and new memory-object identities are fixed before
optimization. No local LLM is downloaded or invoked by this experiment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F

from brain_in_computer.associative import distinct_patterns
from brain_in_computer.computer_use.environment import TASKS
from brain_in_computer.regional_memory import load_regional_checkpoint, save_regional_checkpoint
from brain_in_computer.training import atomic_json
from experiments import advance_grounding as desktop
from experiments import train_regional_memory as memory


FRESH_PHRASES = {
    "click_color": ("could you please choose the {color} button", "please click the button colored {color}"),
    "click_position": ("could you please choose the {position} button", "please click on the button at {position}"),
    "describe_position": ("could you please tell me the color at {position}", "tell me which color the {position} button is"),
    "type_word": ("could you please write {word}", "please put the word {word} in the field"),
    "recall": ("could you please tell me what color you clicked", "tell me which color you selected last"),
    "greet": ("hello once again bic", "hi again friend", "greetings again bic", "good morning my friend"),
    "clarify": ("could you please paint the sky", "please build me a spaceship"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repair_phrases():
    phrases = {task: list(dict.fromkeys(text for split in desktop.V04_TEMPLATES.values()
                                      for text in split[task])) for task in TASKS}
    phrases["greet"] = list(dict.fromkeys(phrases["greet"] + [
        f"{opening} {ending}" for opening in ("hello", "hi", "hey", "greetings", "good morning", "good evening")
        for ending in ("there", "again", "bic", "friend")]))
    for task in TASKS:
        if set(phrases[task]) & set(FRESH_PHRASES[task]):
            raise ValueError("fresh phrases must be disjoint from rehearsal phrases")
    return phrases


def evaluate_desktop(computer, seed, split, per_task=32):
    if split != "v06_fresh":
        return desktop.evaluate(computer, seed=seed, per_task=per_task, split=split)
    banks = {**desktop.V04_TEMPLATES, "v06_fresh": FRESH_PHRASES}
    with patch.object(desktop, "V04_TEMPLATES", banks):
        return desktop.evaluate(computer, seed=seed, per_task=per_task, split=split)


def rehearse(computer, actions, replies, phrases, task_weights, rng):
    # Twelve action states plus twenty terminal reply states per update. The
    # priority is per task and comes from pre-training development errors.
    action_rows = [rng.choice(rng.choice(actions)) for _ in range(12)]
    tasks = rng.choices(list(TASKS), weights=task_weights, k=20)
    rows = action_rows + [rng.choice(replies[task]) for task in tasks]
    device = next(computer.parameters()).device
    pixels = torch.stack([row["pixels"] for row in rows]).to(device, dtype=torch.float32) / 255
    body = torch.stack([row["body"] for row in rows]).to(device)
    prompts = [rng.choice(phrases[row["task"]]).format(color=row["target"],
               position=row["target"].replace("_", " "), word=row["target"]) for row in rows]
    decoder, target_bytes = computer.prepare_reply([row["reply"] or "" for row in rows])
    target_bytes[:12].zero_()
    output = computer(pixels, body, prompts, decoder)
    action_loss = F.cross_entropy(output["logits"][:12, -1],
                                 torch.tensor([row["action"] for row in action_rows], device=device))
    text_loss = memory.byte_loss(output["language_logits"], decoder, target_bytes)
    target_scene = torch.tensor([row["scene"] for row in rows], device=device)
    scene_loss = torch.stack([F.cross_entropy(part, target_scene[:, i]) for i, part in
                             enumerate(output["scene_logits"].split((4, 4, 4, 4, 5, 21), -1))]).mean()
    return action_loss + 3 * text_loss + .3 * scene_loss


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        raise ValueError("Use a new output directory; this one already contains a protocol")
    v05 = json.loads(Path("runs/regional-memory-v05/protocol.json").read_text())
    excluded = sorted(set(p for values in v05["identity_partitions"].values() for p in values))
    test_ids = distinct_patterns(256, args.seed + 71, excluded)
    phrases = repair_phrases()
    protocol = {"schema": "bic-retention-repair-v1", "arguments": vars(args),
                "runner_sha256": sha256(__file__), "initial_sha256": sha256(args.initialize),
                "repair_phrases": phrases, "fresh_desktop_phrases": FRESH_PHRASES,
                "fresh_memory_identities": test_ids, "excluded_identity_count": len(excluded),
                "steps": args.steps, "memory_examples_per_step": 16, "desktop_examples_per_step": 32,
                "priority_rule": "Fixed task weights 1+8*(1-initial development reply accuracy); no test errors used",
                "selection": "Earliest highest min(memory-development joint, desktop-development joint), every200steps",
                "earlier_wording_tests_are_now_development_evidence": True,
                "fresh_test_used_for_selection": False,
                "comparison_scope": "Before/after one targeted-rehearsal run; no causal advantage claimed over other replay schedules",
                "local_llm_invoked": False}
    atomic_json(root / "protocol.json", protocol)
    agent = load_regional_checkpoint(args.initialize, args.device)
    cache_encoder = copy.deepcopy(agent.encoder).cpu().eval()
    initial = evaluate_desktop(agent.computer, args.seed + 100, "development")
    task_weights = [1 + 8 * (1 - initial["tasks"][task]["reply_exact"]) for task in TASKS]
    atomic_json(root / "initial_development.json", {"desktop": initial, "task_weights": dict(zip(TASKS, task_weights))})
    started = time.perf_counter()
    ids = v05["identity_partitions"]
    training = memory.encode_specs(memory.build_specs(ids["train"], "train", args.seed + 200, 1600), cache_encoder)
    development = memory.encode_specs(memory.build_specs(ids["development"], "development", args.seed + 201, 640), cache_encoder)
    actions, replies, _ = desktop.build_curriculum(args.seed + 300, 3, variations=2, other_episodes=100)
    action_groups = [[row for row in actions if row["action"] == action] for action in range(11)]
    reply_groups = {task: [row for row in replies if row["task"] == task] for task in TASKS}
    data_seconds = time.perf_counter() - started
    rng = random.Random(args.seed + 1)
    rehearsal_rng = random.Random(args.seed + 2)
    optimizer = torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad], lr=.00025, weight_decay=.0001)
    best, chosen, history = -1., 0, []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        agent.train()
        indexes = rng.choices(range(len(training["specs"])), k=16)
        specs = [training["specs"][i] for i in indexes]
        decoder, targets = agent.computer.prepare_reply([spec["reply"] for spec in specs])
        prediction = agent.forward_evidence(training["scores"][indexes].to(args.device),
                    training["known"][indexes].to(args.device), memory.normalized_prompts(specs, rng), decoder)
        new_loss = F.cross_entropy(prediction["logits"][:, -1], training["targets"][indexes].to(args.device))
        new_loss = new_loss + 3 * memory.byte_loss(prediction["language_logits"], decoder, targets)
        old_loss = rehearse(agent.computer, action_groups, reply_groups, phrases, task_weights, rehearsal_rng)
        loss = new_loss + old_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if step % 200 == 0 or step == args.steps:
            old_eval = evaluate_desktop(agent.computer, args.seed + 100, "development")
            new_eval = memory.evaluate_memory(agent, development)
            score = min(old_eval["macro_joint_success"], new_eval["joint_success"])
            row = {"step": step, "desktop": old_eval, "memory": new_eval, "selection_score": score,
                   "elapsed_seconds": time.perf_counter() - started}
            atomic_json(root / f"development-{step}.json", row)
            history.append({"step": step, "old_joint": old_eval["macro_joint_success"],
                            "greeting": old_eval["tasks"]["greet"]["joint_success"],
                            "memory_joint": new_eval["joint_success"], "elapsed_seconds": row["elapsed_seconds"]})
            print(json.dumps(history[-1]), flush=True)
            if score > best:
                best, chosen = score, step
                save_regional_checkpoint(root / "checkpoint.pt", agent,
                    metadata={"selected_step": step, "protocol_sha256": sha256(root / "protocol.json"),
                              "purpose": "targeted-rehearsal retention repair"})
    atomic_json(root / "training_report.json", {"selected_step": chosen, "executed_steps": args.steps,
                "data_seconds": data_seconds, "training_and_development_seconds": time.perf_counter()-started,
                "history": history, "checkpoint_sha256": sha256(root / "checkpoint.pt"), "fresh_test_opened": False})


def evaluate(args):
    torch.set_num_threads(1)
    root = Path(args.output)
    protocol = json.loads((root / "protocol.json").read_text())
    if sha256(__file__) != protocol["runner_sha256"]:
        raise ValueError("Experiment runner changed after protocol freeze")
    if (root / "final_evaluation.json").exists():
        raise ValueError("Preserve the existing final evaluation")
    if sha256(protocol["arguments"]["initialize"]) != protocol["initial_sha256"]:
        raise ValueError("Initial checkpoint changed after protocol freeze")
    training_report = json.loads((root / "training_report.json").read_text())
    if sha256(root / "checkpoint.pt") != training_report["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after training")
    previous = load_regional_checkpoint(protocol["arguments"]["initialize"], args.device)
    current = load_regional_checkpoint(root / "checkpoint.pt", args.device)
    cache_encoder = copy.deepcopy(previous.encoder).cpu().eval()
    seed = protocol["arguments"]["seed"]
    fresh = memory.encode_specs(memory.build_specs(protocol["fresh_memory_identities"], "sealed",
                               seed + 501, 1280), cache_encoder)
    result = {"protocol_sha256": sha256(root / "protocol.json"), "fresh_test_used_for_selection": False,
              "local_llm_invoked": False, "models": {}}
    for name, agent, path in (("v05_before", previous, protocol["arguments"]["initialize"]),
                              ("v06_after", current, root / "checkpoint.pt")):
        row = {"checkpoint_sha256": sha256(path),
               "restoration_diagnostic": evaluate_desktop(agent.computer, 5752, "sealed", 64),
               "fresh_desktop_phrases": evaluate_desktop(agent.computer, seed + 502, "v06_fresh", 64),
               "fresh_memory_objects": memory.evaluate_memory(agent, fresh)}
        result["models"][name] = row
        print(name, "restored", row["restoration_diagnostic"]["macro_joint_success"],
              "fresh-desktop", row["fresh_desktop_phrases"]["macro_joint_success"],
              "memory", row["fresh_memory_objects"]["joint_success"], flush=True)
    result["limits"] = ["Old reserved wording was rehearsed; restoration is a diagnostic, not independent generalization.",
                        "New desktop wording is disjoint from rehearsal; memory objects are fresh but memory instruction patterns are the earlier v0.5 reserved family.",
                        "One repair seed; no claim of superiority to an equally budgeted alternative schedule.",
                        "The five memory relations and seven desktop task meanings remain unchanged."]
    atomic_json(root / "final_evaluation.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--output", default="runs/retention-v06")
    parser.add_argument("--initialize", default="runs/regional-memory-v05/checkpoint.pt")
    parser.add_argument("--seed", type=int, default=6061)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("steps must be positive")
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
