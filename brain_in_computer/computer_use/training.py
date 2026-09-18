"""Imitation learning and closed-loop evaluation in the pixel desktop.

The procedural teacher is used for training labels and evaluation scoring only.
The inference controller accepts no Goal and never invokes the teacher.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time
from itertools import product
import hashlib

import torch
from torch.nn import functional as F

from ..language import ByteCodec
from ..training import atomic_json
from .environment import ACTION_NAMES, TASKS, COLORS, MiniDesktop, make_episode, oracle_action, goal_success
from .model import ComputerBrain, ComputerConfig

SCHEMA = "bic-computer-use-v2"
VISIBLE_TEXTS = ("", "h", "i", "o", "k", *("".join(pair) for pair in product("hiok", repeat=2)))


@dataclass
class ComputerTrainConfig:
    seed: int = 31
    demonstrations: int = 2100
    batch_size: int = 32
    learning_rate: float = 0.001
    steps: int = 5000
    stop_fraction: float = 0.35
    language_weight: float = 3.0

    def __post_init__(self):
        for name in ("seed", "demonstrations", "batch_size", "steps"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if min(self.demonstrations, self.batch_size, self.steps) < 1:
            raise ValueError("demonstrations, batch_size, and steps must be positive")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 to include action and reply examples")
        if not 0 < self.learning_rate < float("inf"):
            raise ValueError("learning_rate must be finite and positive")
        if not 0 < self.stop_fraction < 1 or not 0 < self.language_weight < float("inf"):
            raise ValueError("stop_fraction must lie in (0,1) and language_weight must be finite and positive")


def history_tensors(history, size):
    recent = history[-size:]
    padded = [recent[0]] * (size-len(recent)) + recent
    return torch.stack([x["pixels"] for x in padded]), torch.stack([x["body"] for x in padded])


def tensor_batch(histories, model):
    pairs = [history_tensors(history, model.config.max_history) for history in histories]
    device = next(model.parameters()).device
    return torch.stack([x[0] for x in pairs]).to(device), torch.stack([x[1] for x in pairs]).to(device)


def build_demonstrations(config, history_size):
    rng = random.Random(config.seed + 1)
    actions, replies = [], []
    for index in range(config.demonstrations):
        env, goal = make_episode(rng, task=TASKS[index % len(TASKS)], split="train")
        history = [env.observe()]
        for _ in range(10):
            action = oracle_action(env, goal)
            pixels, body = history_tensors(history, history_size)
            actions.append({"pixels": pixels, "body": body, "prompt": goal.prompt,
                            "action": action, "reply": None, "task": goal.task,
                            "scene": visible_labels(env)})
            env.step(action)
            history.append(env.observe())
            if env.terminated:
                break
        if not goal_success(env, goal):
            raise RuntimeError(f"Teacher failed {goal.task}: {goal.prompt}")
        pixels, body = history_tensors(history, history_size)
        replies.append({"pixels": pixels, "body": body, "prompt": goal.prompt,
                        "action": ACTION_NAMES.index("stop"), "reply": goal.expected_reply,
                        "task": goal.task, "scene": visible_labels(env)})
    return actions, replies


def visible_labels(env):
    """Training annotations of visible objects; never an inference input."""
    return [*(COLORS.index(color) for color in env.button_colors),
            0 if env.last_clicked is None else COLORS.index(env.last_clicked)+1,
            VISIBLE_TEXTS.index(env.text)]


@torch.no_grad()
def run_turn(model, env, prompt, max_actions=8, *, ablate=(), blank_pixels=False, blank_prompt=False):
    """Act and speak using only pixels, body state, and the user's words.

    Scene state persists in env across turns; neural state is recomputed over a
    bounded recent history. A visible status swatch supplies last-click evidence.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a nonempty instruction or question")
    if type(max_actions) is not int or not 1 <= max_actions <= 10:
        raise ValueError("max_actions must be between 1 and 10")
    normalized = prompt.strip().lower()
    model.language.codec.encode(normalized, max_bytes=128)
    env.begin_turn()
    history, frames, actions = [env.observe()], [env.render()], []
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        for _ in range(max_actions):
            pixels, body = tensor_batch([history], model)
            if blank_pixels:
                pixels.zero_()
            words = ["" if blank_prompt else normalized]
            prefix = torch.full((1, 1), ByteCodec.BOS, dtype=torch.long, device=pixels.device)
            output = model(pixels, body, words, prefix, ablate=ablate)
            action = int(output["logits"][0, -1].argmax())
            env.step(action)
            actions.append(action)
            history.append(env.observe())
            frames.append(env.render())
            if env.terminated:
                break
        pixels, body = tensor_batch([history], model)
        if blank_pixels:
            pixels.zero_()
        reply = model.respond(pixels, body, ["" if blank_prompt else normalized], ablate=ablate)[0]
    finally:
        for module, mode in modes:
            module.training = mode
    return {"reply": reply, "actions": actions, "terminated": env.terminated,
            "frames": frames, "prompt": normalized, "used_teacher": False}


@torch.no_grad()
def evaluate_computer(model, *, seed=900031, episodes_per_task=64, split="train", shift=False,
                      ablate=(), blank_pixels=False, blank_prompt=False, batch_size=32):
    if episodes_per_task < 1 or batch_size < 1:
        raise ValueError("evaluation sizes must be positive")
    rng = random.Random(seed)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    results, failures = {}, []
    started = time.perf_counter()
    try:
        for task in TASKS:
            correct_action = correct_reply = correct_both = count = 0
            for offset in range(0, episodes_per_task, batch_size):
                pairs = [make_episode(rng, task=task, split=split, shift=shift)
                         for _ in range(min(batch_size, episodes_per_task-offset))]
                environments = [p[0] for p in pairs]
                goals = [p[1] for p in pairs]
                histories = [[env.observe()] for env in environments]
                words = ["" if blank_prompt else goal.prompt.lower() for goal in goals]
                traces = [[] for _ in pairs]
                for _ in range(8):
                    pixels, body = tensor_batch(histories, model)
                    if blank_pixels:
                        pixels.zero_()
                    prefix = torch.full((len(pairs), 1), ByteCodec.BOS, dtype=torch.long, device=pixels.device)
                    selected = model(pixels, body, words, prefix, ablate=ablate)["logits"][:, -1].argmax(-1).cpu().tolist()
                    for i, (env, action) in enumerate(zip(environments, selected)):
                        if not env.terminated:
                            env.step(action)
                            histories[i].append(env.observe())
                            traces[i].append(action)
                    if all(env.terminated for env in environments):
                        break
                pixels, body = tensor_batch(histories, model)
                if blank_pixels:
                    pixels.zero_()
                generated = model.respond(pixels, body, words, ablate=ablate)
                for env, goal, reply, trace in zip(environments, goals, generated, traces):
                    action_ok = bool(goal_success(env, goal))
                    reply_ok = reply.strip() == goal.expected_reply
                    correct_action += action_ok
                    correct_reply += reply_ok
                    correct_both += action_ok and reply_ok
                    count += 1
                    if (not action_ok or not reply_ok) and sum(f["task"] == task for f in failures) < 3:
                        failures.append({"task": task, "prompt": goal.prompt,
                                         "expected_reply": goal.expected_reply, "reply": reply,
                                         "action_success": action_ok, "actions": [ACTION_NAMES[a] for a in trace]})
            results[task] = {"episodes": count, "action_success": correct_action/count,
                             "reply_exact": correct_reply/count, "joint_success": correct_both/count,
                             "action_success_count": correct_action, "reply_exact_count": correct_reply,
                             "joint_success_count": correct_both}
    finally:
        for module, mode in modes:
            module.training = mode
    return {"seed": seed, "prompt_split": split, "layout_shift": shift,
            "ablate": list(ablate), "blank_pixels": blank_pixels, "blank_prompt": blank_prompt,
            "teacher_used_for_policy": False, "free_running_replies": True,
            "macro_action_success": sum(x["action_success"] for x in results.values())/len(results),
            "macro_reply_exact": sum(x["reply_exact"] for x in results.values())/len(results),
            "macro_joint_success": sum(x["joint_success"] for x in results.values())/len(results),
            "tasks": results, "failures": failures, "evaluation_seconds": time.perf_counter()-started}


def save_computer_checkpoint(path, model, config, optimizer, generator, step, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"schema": SCHEMA, "model_config": asdict(model.config), "train_config": asdict(config),
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "sample_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "step": step, "history": history}, temporary)
    temporary.replace(path)


def load_computer_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema") == "bic-regional-memory-v1":
        # The memory experiment updates these same computer/language weights.
        # Expose them to the existing desktop UI and evaluation commands too.
        from ..regional_memory import load_regional_checkpoint
        return load_regional_checkpoint(path, device=device).computer
    if saved.get("schema") != SCHEMA:
        raise ValueError("Expected a computer-use checkpoint, not a v0.1 symbolic checkpoint")
    model = ComputerBrain(ComputerConfig(**saved["model_config"])).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model


def train_computer(config=None, model_config=None, output="runs/computer", device="cpu", log_every=250,
                   initialize_from=None):
    config = config or ComputerTrainConfig()
    source = None
    if initialize_from:
        source = torch.load(initialize_from, map_location="cpu", weights_only=True)
        if source.get("schema") != SCHEMA:
            raise ValueError("Initialization checkpoint must use the computer model schema")
        if model_config is None:
            model_config = ComputerConfig(**source["model_config"])
    model_config = model_config or ComputerConfig()
    if log_every < 1:
        raise ValueError("log_every must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    model = ComputerBrain(model_config).to(device)
    initialization = None
    if initialize_from:
        if source.get("schema") != SCHEMA or source["model_config"] != asdict(model_config):
            raise ValueError("Initialization checkpoint must match the computer model schema and configuration")
        model.load_state_dict(source["model"], strict=True)
        initialization = {"path": str(initialize_from), "sha256": hashlib.sha256(Path(initialize_from).read_bytes()).hexdigest(),
                          "source_steps": source["step"], "optimizer_restarted": True}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(config.seed+2)
    print("Building procedural demonstrations from visible desktop interactions...", flush=True)
    action_data, reply_data = build_demonstrations(config, model_config.max_history)
    action_groups = [[i for i, row in enumerate(action_data) if row["action"] == action]
                     for action in range(len(ACTION_NAMES))]
    action_groups = [group for group in action_groups if group]
    class_weights = torch.tensor([config.stop_fraction if action_data[group[0]]["action"] == 10
                                   else (1-config.stop_fraction)/10 for group in action_groups])
    print(f"Demonstrations: {len(action_data)} action states, {len(reply_data)} completed responses", flush=True)
    initial = evaluate_computer(model, episodes_per_task=16, seed=config.seed+10000)
    atomic_json(output / "initial_eval.json", initial)
    history, trained_names = [], set()
    start = time.perf_counter()
    for step in range(1, config.steps+1):
        model.train()
        half = config.batch_size//2
        classes = torch.multinomial(class_weights, config.batch_size-half, replacement=True, generator=generator).tolist()
        sample_a = [action_groups[c][int(torch.randint(len(action_groups[c]), (1,), generator=generator))]
                    for c in classes]
        sample_r = torch.randint(len(reply_data), (half,), generator=generator).tolist()
        rows = [action_data[i] for i in sample_a] + [reply_data[i] for i in sample_r]
        pixels = torch.stack([r["pixels"] for r in rows]).to(device)
        body = torch.stack([r["body"] for r in rows]).to(device)
        words = [r["prompt"].lower() for r in rows]
        action_targets = torch.tensor([r["action"] for r in rows], device=device)
        decoder, targets = model.prepare_reply([r["reply"] or "" for r in rows])
        for i, row in enumerate(rows):
            if row["reply"] is None:
                targets[i].fill_(ByteCodec.PAD)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(pixels, body, words, decoder)
        n_actions = len(sample_a)
        action_loss = F.cross_entropy(prediction["logits"][:n_actions, -1], action_targets[:n_actions])
        active = targets != ByteCodec.PAD
        # Word beginnings carry most semantic distinctions in these short
        # replies. Upweight them so punctuation and predictable suffixes cannot
        # conceal a wrong color/intent. All preceding bytes remain teacher input.
        weights = torch.ones_like(targets, dtype=pixels.dtype)
        weights[:, 0] = 8
        weights[decoder == ord(" ") + ByteCodec.BYTE_OFFSET] = 8
        byte_loss = F.cross_entropy(prediction["language_logits"][active], targets[active], reduction="none")
        language_loss = (byte_loss * weights[active]).sum() / weights[active].sum()
        scene_targets = torch.tensor([r["scene"] for r in rows], device=device)
        scene_parts = prediction["scene_logits"].split((4, 4, 4, 4, 5, 21), dim=-1)
        scene_loss = torch.stack([F.cross_entropy(part, scene_targets[:, index])
                                  for index, part in enumerate(scene_parts)]).mean()
        # A small next-observation objective keeps the predictive region grounded
        # in real consecutive encoded screenshots; target features are detached.
        predictive_loss = action_loss * 0
        if model_config.max_history > 1:
            with torch.no_grad():
                vision = model.retina(pixels[:, 1:].reshape(-1, 3, 32, 32)).reshape(len(rows), model_config.max_history-1, -1)
                future = torch.cat((vision, pixels.new_zeros(len(rows), model_config.max_history-1, 8), body[:, 1:]), -1)
            predictive_loss = F.mse_loss(prediction["prediction"][:, :-1], future)
        loss = action_loss + config.language_weight * language_loss + 0.5 * scene_loss + 0.03 * predictive_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite computer learning loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if step == 1 or step == config.steps:
            trained_names.update(name for name, p in model.named_parameters() if p.grad is not None and bool(p.grad.ne(0).any()))
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == config.steps:
            metrics = {"step": step, "loss": float(loss.detach()), "action_loss": float(action_loss.detach()),
                       "language_loss": float(language_loss.detach()), "prediction_loss": float(predictive_loss.detach()),
                       "scene_loss": float(scene_loss.detach()),
                       "teacher_state_action_accuracy": float(prediction["logits"][:n_actions, -1].argmax(-1).eq(action_targets[:n_actions]).float().mean()),
                       "elapsed_seconds": time.perf_counter()-start}
            history.append(metrics)
            print(f"step={step} action_loss={metrics['action_loss']:.3f} text_loss={metrics['language_loss']:.3f} "
                  f"teacher_action_acc={metrics['teacher_state_action_accuracy']:.3f}", flush=True)
            save_computer_checkpoint(output / "checkpoint.pt", model, config, optimizer, generator, step, history)
    elapsed = time.perf_counter()-start
    # Validation uses fresh scenes with training-distribution wording. Final test
    # wording/layout results are run separately, after development choices stop.
    validation = evaluate_computer(model, seed=config.seed+20000, episodes_per_task=64, split="train")
    report = {"version": "0.3.0", "method": "action-reweighted imitation, weighted next-byte generation, and supervised visual naming",
              "train_config": asdict(config), "model_config": asdict(model_config),
              "initialization": initialization,
              "parameter_counts": model.parameter_counts(), "training_seconds": elapsed,
              "demonstration_action_states": len(action_data), "demonstration_replies": len(reply_data),
              "initial": initial, "validation": validation, "history": history,
              "parameters_with_nonzero_gradients_observed": sorted(trained_names),
              "parameters_without_observed_nonzero_gradients": [n for n,p in model.named_parameters() if n not in trained_names],
              "english_training_performed": True, "real_os_access": False,
              "notes": ["Training state accuracy uses teacher states; reported task success uses model-driven closed-loop rollouts.",
                        "Replies are freely generated; their correctness is scored independently from actions.",
                        "Visual naming annotations train an auxiliary head; they are never inference inputs.",
                        "This is a small pixel simulation, not a general desktop/browser benchmark."]}
    atomic_json(output / "report.json", report)
    atomic_json(output / "validation_eval.json", validation)
    print(f"Validation: actions={validation['macro_action_success']:.3f} replies={validation['macro_reply_exact']:.3f} "
          f"joint={validation['macro_joint_success']:.3f}; {elapsed:.1f}s training", flush=True)
    return model, report
