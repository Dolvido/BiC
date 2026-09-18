"""Reproducible v0.4 grounding experiment; no changes to the v0.3 policy API.

The old v0.3 tests are development data now. V04_TEMPLATES['sealed'] is never
sampled for optimization or development selection. At inference, both compared
models receive only rendered pixels, body measurements, and an English prompt.
The referent heads below are training-only teachers of regional representations;
their labels and outputs are never passed into the policy or response generator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from itertools import permutations
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.nn import functional as F

from brain_in_computer.computer_use.environment import (
    ACTION_NAMES, COLORS, POSITIONS, TASKS, Goal, MiniDesktop,
    PROMPT_TEMPLATES, make_episode, oracle_action, goal_success,
)
from brain_in_computer.computer_use.model import ComputerBrain, ComputerConfig
from brain_in_computer.computer_use.training import (
    ComputerTrainConfig, history_tensors, tensor_batch, visible_labels,
    load_computer_checkpoint, save_computer_checkpoint, evaluate_computer,
)
from brain_in_computer.language import ByteCodec
from brain_in_computer.training import atomic_json


EXTRA_TRAIN = {
    "click_color": ("please press the {color} button", "could you click {color}", "select {color}", "choose {color}", "please select the {color} button", "would you click the {color} button", "click on {color}", "press {color} now"),
    "click_position": ("please press the {position} button", "could you click {position}", "select {position}", "choose {position}", "please select the {position} button", "would you click the button at {position}", "click on the {position} button", "press {position} now"),
    "describe_position": ("please name the {position} color", "what is the color at {position}", "tell me the color at {position}", "which color is at {position}", "please identify the color at {position}", "what color do you see at {position}", "name the {position} button color", "describe the color at {position}"),
    "type_word": ("please enter {word}", "type {word} in the field", "write {word}", "could you type {word}", "please write {word} in the field", "enter the word {word}", "type {word} now", "would you enter {word}"),
    "recall": ("what color did you click", "please tell me the last clicked color", "which color was clicked", "what was clicked last", "please name the last color", "show me the last clicked color", "what is your last clicked color", "tell me what color you clicked"),
    "greet": ("hi there", "hey there", "good afternoon", "good night", "hello bic", "hi bic", "greetings bic", "hello again"),
    "clarify": ("paint the ocean", "build a castle", "move the sun", "cook dinner", "find a dragon", "make a spaceship", "sing me a song", "please fly to mars"),
}

V04_TEMPLATES = {
    "train": {task: tuple(dict.fromkeys(sum((list(PROMPT_TEMPLATES[split][task]) for split in PROMPT_TEMPLATES), []) + list(EXTRA_TRAIN[task]))) for task in TASKS},
    "development": {
        "click_color": ("please click on {color}", "could you select the {color} button"),
        "click_position": ("please click on the {position} button", "could you select the button at {position}"),
        "describe_position": ("please tell me the color at {position}", "could you name the color at {position}"),
        "type_word": ("could you write {word} in the field", "please enter the word {word}"),
        "recall": ("please tell me what color you clicked", "could you show the last clicked color"),
        "greet": ("good morning bic", "hello there bic"),
        "clarify": ("could you cook dinner", "please build a castle"),
    },
    "sealed": {
        "click_color": ("would you please press {color}", "select the {color} button now", "please choose {color}"),
        "click_position": ("would you please press {position}", "select the {position} button now", "please choose {position}"),
        "describe_position": ("would you tell me the color at {position}", "please identify the {position} button color", "name the color of the button at {position}"),
        "type_word": ("would you please type {word}", "write the word {word} in the field", "please fill the field with {word}"),
        "recall": ("would you tell me your last clicked color", "please name the color you clicked last", "which color did you select last"),
        "greet": ("good evening bic", "greetings there", "hey again"),
        "clarify": ("would you find a dragon", "please paint the moon", "could you sing a song"),
    },
}


def render_prompt(task, target, rng, split="train"):
    return rng.choice(V04_TEMPLATES[split][task]).format(
        color=target, position=target.replace("_", " "), word=target)


def episode(rng, task, split="train", shift=False):
    env, goal = make_episode(rng, task=task, split="train", shift=shift)
    return env, replace(goal, prompt=render_prompt(task, goal.target, rng, split))


def goal_for(env, task, target):
    if task == "click_color":
        reply = f"clicked {target}."
    elif task in ("click_position", "describe_position"):
        color = env.button_colors[POSITIONS.index(target)]
        reply = f"clicked {color}." if task == "click_position" else f"it is {color}."
    else:
        raise ValueError(task)
    return Goal(task, "", target, reply)


def referent_labels(env, goal):
    color, position = 4, 4
    if goal.task == "click_color":
        color = COLORS.index(goal.target)
        position = env.button_colors.index(goal.target)
    elif goal.task in ("click_position", "describe_position"):
        position = POSITIONS.index(goal.target)
        color = COLORS.index(env.button_colors[position])
    elif goal.task == "recall" and env.last_clicked is not None:
        color = COLORS.index(env.last_clicked)
    return [TASKS.index(goal.task), color, position]


def build_curriculum(seed, history_size, variations=8, other_episodes=400):
    """Counterfactual lessons cross all 24 permutations with all four targets.

    All screens in a contrast group are the same; only the instruction target
    changes. Across groups the same instruction encounters every color layout.
    Screen storage is uint8; conversion to normalized float occurs on sampling.
    """
    rng = random.Random(seed)
    actions, replies, paired_replies = [], [], []

    def record_episode(env, goal):
        history = [env.observe()]
        for _ in range(10):
            action = oracle_action(env, goal)
            pixels, body = history_tensors(history, history_size)
            actions.append({"pixels": (pixels * 255).round().to(torch.uint8), "body": body,
                            "task": goal.task, "target": goal.target, "action": action,
                            "reply": None, "scene": visible_labels(env), "referent": referent_labels(env, goal)})
            env.step(action)
            history.append(env.observe())
            if env.terminated:
                break
        if not goal_success(env, goal):
            raise RuntimeError("Curriculum teacher failed")
        pixels, body = history_tensors(history, history_size)
        replies.append({"pixels": (pixels * 255).round().to(torch.uint8), "body": body,
                        "task": goal.task, "target": goal.target, "action": 10,
                        "reply": goal.expected_reply, "scene": visible_labels(env),
                        "referent": referent_labels(env, goal)})
        return len(replies)-1

    for layout in permutations(COLORS):
        for variation in range(variations):
            base = MiniDesktop(rng.randrange(2**32), shift=variation % 4 == 3)
            base.reset(layout, rng.choice((None, *COLORS)), rng.choice(("", "hi", "ok", "h", "o")))
            for task in ("click_color", "click_position", "describe_position"):
                paired = []
                for target in (COLORS if task == "click_color" else POSITIONS):
                    env = base.clone()
                    paired.append(record_episode(env, goal_for(env, task, target)))
                paired_replies.append(paired)
    for task in TASKS[3:]:
        for index in range(other_episodes):
            env, goal = episode(rng, task, shift=index % 4 == 3)
            record_episode(env, goal)
    return actions, replies, paired_replies


@torch.no_grad()
def evaluate(model, seed=4049001, per_task=64, split="development", shift=False, blank_pixels=False, blank_prompt=False):
    rng = random.Random(seed)
    model.eval()
    results, failures = {}, []
    for task in TASKS:
        successes = [0, 0, 0]
        for offset in range(0, per_task, 32):
            pairs = [episode(rng, task, split, shift) for _ in range(min(32, per_task-offset))]
            envs, goals = zip(*pairs)
            histories = [[env.observe()] for env in envs]
            prompts = ["" if blank_prompt else goal.prompt for goal in goals]
            traces = [[] for _ in pairs]
            for _ in range(8):
                pixels, body = tensor_batch(histories, model)
                if blank_pixels:
                    pixels.zero_()
                prefix = torch.full((len(pairs), 1), ByteCodec.BOS, dtype=torch.long, device=pixels.device)
                chosen = model(pixels, body, prompts, prefix)["logits"][:, -1].argmax(-1).tolist()
                for index, (env, action) in enumerate(zip(envs, chosen)):
                    if not env.terminated:
                        env.step(action)
                        histories[index].append(env.observe())
                        traces[index].append(action)
                if all(env.terminated for env in envs):
                    break
            pixels, body = tensor_batch(histories, model)
            if blank_pixels:
                pixels.zero_()
            generated = model.respond(pixels, body, prompts)
            for env, goal, reply, trace in zip(envs, goals, generated, traces):
                action_ok = bool(goal_success(env, goal))
                reply_ok = reply.strip() == goal.expected_reply
                successes = [x + int(y) for x, y in zip(successes, (action_ok, reply_ok, action_ok and reply_ok))]
                if (not action_ok or not reply_ok) and sum(x["task"] == task for x in failures) < 3:
                    failures.append({"task": task, "prompt": goal.prompt, "expected": goal.expected_reply,
                                     "reply": reply, "action_success": action_ok, "actions": [ACTION_NAMES[a] for a in trace]})
        results[task] = dict(zip(("action_success", "reply_exact", "joint_success"), (n/per_task for n in successes)))
        results[task]["episodes"] = per_task
    return {"seed": seed, "split": split, "layout_shift": shift, "blank_pixels": blank_pixels,
            "blank_prompt": blank_prompt, "teacher_used_for_policy": False, "free_running_replies": True,
            "tasks": results, "failures": failures,
            **{f"macro_{metric}": sum(row[metric] for row in results.values())/len(TASKS)
               for metric in ("action_success", "reply_exact", "joint_success")}}


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed+1)
    sampler = torch.Generator().manual_seed(args.seed+2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.initialize:
        model = load_computer_checkpoint(args.initialize, device=args.device)
    else:
        model = ComputerBrain(ComputerConfig(hidden_size=args.hidden, language_hidden_size=args.language_hidden,
                                             embedding_size=64, visual_features=64)).to(args.device)
    # Separate auxiliary weights are discarded by the standard inference loader.
    referent_head = nn.Linear(3*model.config.hidden_size, 17).to(args.device)
    optimizer = torch.optim.AdamW([*model.parameters(), *referent_head.parameters()], lr=args.lr, weight_decay=0.0001)
    config = ComputerTrainConfig(seed=args.seed, batch_size=args.batch_size, learning_rate=args.lr,
                                 steps=args.steps, demonstrations=1, language_weight=3.0)
    print("Building counterfactual scene curriculum...", flush=True)
    actions, replies, pairs = build_curriculum(args.seed+10, model.config.max_history, args.variations, args.other_episodes)
    config.demonstrations = len(replies)
    groups = [[i for i, row in enumerate(actions) if row["action"] == action] for action in range(11)]
    weights = torch.tensor([0.2 if i == 10 else 0.08 for i in range(11)])
    reply_groups = [[i for i, row in enumerate(replies) if row["task"] == task] for task in TASKS]
    print(f"Parameters={model.parameter_counts()['total']:,}; action states={len(actions)}; replies={len(replies)}", flush=True)
    initial = evaluate(model, seed=args.seed+20000, per_task=32, split="train")
    atomic_json(output / "initial.json", initial)
    history, best_score, best_step = [], -1, 0
    start = time.perf_counter()
    for step in range(1, args.steps+1):
        model.train()
        n_actions = args.batch_size//2
        classes = torch.multinomial(weights, n_actions, replacement=True, generator=sampler).tolist()
        a_rows = [actions[rng.choice(groups[c])] for c in classes]
        r_rows = [replies[i] for i in rng.choice(pairs)]
        while len(r_rows) < args.batch_size-n_actions:
            task_group = rng.choice(reply_groups)
            r_rows.append(replies[rng.choice(task_group)])
        r_rows = r_rows[:args.batch_size-n_actions]
        rows = a_rows + r_rows
        pixels = torch.stack([r["pixels"] for r in rows]).to(args.device, dtype=torch.float32).div_(255)
        body = torch.stack([r["body"] for r in rows]).to(args.device)
        prompts = [render_prompt(r["task"], r["target"], rng) for r in rows]
        decoder, targets = model.prepare_reply([r["reply"] or "" for r in rows])
        targets[:n_actions].fill_(0)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(pixels, body, prompts, decoder)
        action_targets = torch.tensor([r["action"] for r in a_rows], device=args.device)
        action_loss = F.cross_entropy(prediction["logits"][:n_actions, -1], action_targets)
        active = targets != 0
        byte_weights = torch.ones_like(targets, dtype=pixels.dtype)
        byte_weights[:, 0] = 8
        byte_weights[decoder == ord(" ")+3] = 8
        byte_loss = F.cross_entropy(prediction["language_logits"][active], targets[active], reduction="none")
        language_loss = (byte_loss*byte_weights[active]).sum()/byte_weights[active].sum()
        scene_targets = torch.tensor([r["scene"] for r in rows], device=args.device)
        scene_loss = torch.stack([F.cross_entropy(part, scene_targets[:, i])
                                  for i, part in enumerate(prediction["scene_logits"].split((4, 4, 4, 4, 5, 21), -1))]).mean()
        states = prediction["region_activity"]
        semantic = torch.cat([states[name][:, -1] for name in ("prefrontal_cortex", "parietal_association", "hippocampus")], -1)
        ref_targets = torch.tensor([r["referent"] for r in rows], device=args.device)
        referent_loss = torch.stack([F.cross_entropy(part, ref_targets[:, i])
                                     for i, part in enumerate(referent_head(semantic).split((7, 5, 5), -1))]).mean()
        loss = action_loss + 3*language_loss + 0.3*scene_loss + args.referent_weight*referent_loss
        loss.backward()
        nn.utils.clip_grad_norm_([*model.parameters(), *referent_head.parameters()], 1, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == args.steps:
            metrics = {"step": step, "seconds": time.perf_counter()-start,
                       "loss": float(loss.detach()), "action_loss": float(action_loss.detach()),
                       "language_loss": float(language_loss.detach()), "scene_loss": float(scene_loss.detach()),
                       "referent_loss": float(referent_loss.detach())}
            history.append(metrics)
            print(json.dumps(metrics), flush=True)
        if step % args.evaluate_every == 0 or step == args.steps:
            development = evaluate(model, seed=args.seed+21000, per_task=32, split="development")
            familiar = evaluate(model, seed=args.seed+22000, per_task=32, split="train")
            score = (development["macro_joint_success"] + familiar["macro_joint_success"])/2
            atomic_json(output / f"development-{step}.json", {"development": development, "familiar": familiar})
            print(f"Development step={step}: familiar={familiar['macro_joint_success']:.3f}, phrasing={development['macro_joint_success']:.3f}", flush=True)
            if score > best_score:
                best_score, best_step = score, step
                save_computer_checkpoint(output / "checkpoint.pt", model, config, optimizer, sampler, step, history)
    # Inference checkpoint is the development-selected model, never test-selected.
    result = {"experiment": "counterfactual grounded language v0.4", "arguments": vars(args),
              "initialization_sha256": hashlib.sha256(Path(args.initialize).read_bytes()).hexdigest() if args.initialize else None,
              "parameter_counts": model.parameter_counts(), "model_config": asdict(model.config),
              "action_states": len(actions), "reply_states": len(replies), "history": history,
              "training_seconds": time.perf_counter()-start, "selected_step": best_step, "selection_score": best_score,
              "training_templates": V04_TEMPLATES["train"], "development_templates": V04_TEMPLATES["development"],
              "test_used_for_selection": False, "old_v03_test_is_development_data": True,
              "auxiliary_heads_used_during_inference": False,
              "limits": ["All 24 color permutations occur in training; evaluation uses fresh procedural draws that can overlap prior states, plus disjoint sealed phrasing.",
                         "Shifted layouts occur in 25% of training variations; shifted evaluation is not an unseen geometry family.",
                         "The combined intervention does not isolate effects of counterfactual lessons, new phrases, auxiliary loss, or additional updates.",
                         "Eleven actions and the original miniature desktop remain the full interaction scope."],
              "biology": "Distinct regional networks; counterfactual grounding and local referent errors. No claim of anatomical equivalence."}
    atomic_json(output / "report.json", result)
    print(f"Saved selected step {best_step} to {output / 'checkpoint.pt'}", flush=True)


def final_evaluation(args):
    torch.set_num_threads(args.threads)
    paths = {"v04": args.checkpoint, "v03": args.baseline}
    result = {"sealed_templates": V04_TEMPLATES["sealed"], "comparison": {}, "test_used_for_selection": False,
              "scope": {"all_color_permutations_in_training": True, "shift_family_in_v04_training": True,
                        "sealed_shifted_meaning": "Fresh shifted-state draws and unseen phrasing, not unseen geometry; exact state overlap is possible.",
                        "legacy_familiar_meaning": "Original v0.3 training phrasing on fresh matched scenes.",
                        "familiar_meaning": "Expanded v0.4 training phrasing; some phrases were unseen by the frozen v0.3 model."}}
    for label, path in paths.items():
        model = load_computer_checkpoint(path, args.device)
        scores = {"checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        scores["legacy_familiar"] = evaluate_computer(model, seed=4049001, episodes_per_task=args.per_task, split="train")
        print(label, "legacy_familiar", scores["legacy_familiar"]["macro_joint_success"], flush=True)
        for name, options in (("familiar", {"split": "train"}), ("sealed_phrasing", {"split": "sealed"}),
                              ("sealed_shifted", {"split": "sealed", "shift": True}),
                              ("blank_pixels", {"split": "train", "blank_pixels": True}),
                              ("blank_prompt", {"split": "train", "blank_prompt": True})):
            scores[name] = evaluate(model, seed=4049001, per_task=args.per_task, **options)
            print(label, name, scores[name]["macro_joint_success"], flush=True)
        result["comparison"][label] = scores
    atomic_json(Path(args.output), result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--initialize", default="runs/computer/checkpoint.pt")
    fit.add_argument("--output", default="runs/grounding-v04")
    fit.add_argument("--seed", type=int, default=4041)
    fit.add_argument("--steps", type=int, default=5000)
    fit.add_argument("--batch-size", type=int, default=32)
    fit.add_argument("--lr", type=float, default=0.0004)
    fit.add_argument("--variations", type=int, default=8)
    fit.add_argument("--other-episodes", type=int, default=400)
    fit.add_argument("--hidden", type=int, default=96)
    fit.add_argument("--language-hidden", type=int, default=160)
    fit.add_argument("--referent-weight", type=float, default=1.0)
    fit.add_argument("--evaluate-every", type=int, default=1000)
    score = sub.add_parser("evaluate")
    score.add_argument("--checkpoint", default="runs/grounding-v04/checkpoint.pt")
    score.add_argument("--baseline", default="runs/computer/checkpoint.pt")
    score.add_argument("--output", default="runs/grounding-v04/final_evaluation.json")
    score.add_argument("--per-task", type=int, default=128)
    for command in (fit, score):
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        final_evaluation(args)


if __name__ == "__main__":
    main()
