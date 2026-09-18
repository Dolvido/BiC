"""Supervised multi-task learning, rehearsal, evaluation, and resumable state."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time

import torch
import torch.nn.functional as F

from .model import Brain, BrainConfig
from .replay import EpisodeReplay, concatenate
from .tasks import TASK_NAMES, TaskStream
from . import __version__

SCHEMA_VERSION = 1


@dataclass
class TrainConfig:
    seed: int = 7
    batch_size: int = 32
    delay: int = 2
    learning_rate: float = 0.002
    replay_capacity: int = 256
    replay_fraction: float = 0.25
    curriculum: str = "mixed"
    stage_steps: int = 200

    def __post_init__(self):
        for name in ("seed", "batch_size", "delay", "replay_capacity", "stage_steps"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.batch_size < 1 or self.delay < 0 or self.stage_steps < 1:
            raise ValueError("batch_size/stage_steps must be positive; delay nonnegative")
        if not 0 <= self.replay_fraction < 1 or self.replay_capacity < 0:
            raise ValueError("invalid replay capacity or fraction")
        if not 0 < self.learning_rate < float("inf"):
            raise ValueError("learning rate must be finite and positive")
        if self.curriculum not in ("mixed", "staged"):
            raise ValueError("curriculum must be mixed or staged")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def classification_loss(logits, targets):
    active = targets.ne(-100)
    return F.cross_entropy(logits[active], targets[active]) if active.any() else logits.sum() * 0


def learning_loss(outputs, batch):
    action = classification_loss(outputs["logits"], batch.targets)
    sensory = (classification_loss(outputs["visual_logits"], batch.visual_targets)
               + classification_loss(outputs["auditory_logits"], batch.auditory_targets))
    actual = torch.cat([batch.observations[k] for k in ("visual", "auditory", "body")], -1)
    prediction = (F.mse_loss(outputs["prediction"][:, :-1], actual[:, 1:])
                  if actual.shape[1] > 1 else outputs["prediction"].sum() * 0)
    # Confidence critic for the selected action, not an RL return or dopamine model.
    correctness = outputs["logits"][:, -1].detach().argmax(-1).eq(batch.targets[:, -1]).float()
    value = F.mse_loss(outputs["value"][:, -1], correctness)
    total = action + 0.15 * sensory + 0.03 * prediction + 0.05 * value
    return total, {"action": float(action.detach()), "sensory": float(sensory.detach()),
                   "prediction": float(prediction.detach()), "value": float(value.detach())}


@torch.no_grad()
def evaluate(model, *, seed=100007, batches=8, batch_size=64, delay=2, device="cpu", ablate=()):
    if batches < 1 or batch_size < 1:
        raise ValueError("evaluation counts must be positive")
    was_training = model.training
    model.eval()
    stream = TaskStream(seed)
    results = {}
    try:
        for task in TASK_NAMES:
            correct, count = 0, 0
            for _ in range(batches):
                batch = stream.sample(batch_size, tasks=(task,), delay=delay, device=device)
                output = model(batch.observations, ablate=ablate)
                correct += int(output["logits"][:, -1].argmax(-1).eq(batch.targets[:, -1]).sum())
                count += batch_size
            results[task] = {"accuracy": correct/count, "correct": correct, "episodes": count,
                             "standard_error": (correct/count*(1-correct/count)/count)**0.5}
    finally:
        model.train(was_training)
    return {"seed": seed, "delay": delay, "ablate": list(ablate), "random_chance": 0.25,
            "macro_accuracy": sum(r["accuracy"] for r in results.values()) / len(results),
            "tasks": results,
            "interpretation": "Fresh random episodes from the same symbolic generators; not evidence of general intelligence."}


class Trainer:
    def __init__(self, brain_config=None, train_config=None, device="cpu"):
        self.config = train_config or TrainConfig()
        self.device = torch.device(device)
        torch.manual_seed(self.config.seed)
        self.model = Brain(brain_config or BrainConfig()).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate,
                                          weight_decay=0.0001)
        self.stream = TaskStream(self.config.seed + 1)
        self.replay = EpisodeReplay(self.config.replay_capacity, self.config.seed + 2)
        self.step = 0
        self.history = []

    def train_step(self):
        self.model.train()
        cfg = self.config
        tasks = None if cfg.curriculum == "mixed" else (
            TASK_NAMES[min(self.step // cfg.stage_steps, len(TASK_NAMES)-1)],)
        rehearsal_count = int(cfg.batch_size * cfg.replay_fraction) if len(self.replay) else 0
        fresh = self.stream.sample(cfg.batch_size-rehearsal_count, tasks=tasks,
                                   delay=cfg.delay, device=self.device)
        batch = concatenate(fresh, self.replay.sample(rehearsal_count, self.device)) if rehearsal_count else fresh
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(batch.observations)
        loss, components = learning_loss(outputs, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss; checkpoint has not been overwritten")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.replay.add(fresh)
        self.step += 1
        return {"step": self.step, "loss": float(loss.detach()), "gradient_norm": float(grad_norm),
                "accuracy": float(outputs["logits"][:, -1].detach().argmax(-1).eq(batch.targets[:, -1]).float().mean()),
                **components}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": SCHEMA_VERSION, "brain_config": asdict(self.model.config),
                   "train_config": asdict(self.config), "model": self.model.state_dict(),
                   "optimizer": self.optimizer.state_dict(), "step": self.step,
                   "stream_rng": self.stream.generator.get_state(), "replay": self.replay.state_dict(),
                   "torch_rng": torch.get_rng_state(), "history": self.history,
                   "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []}
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint schema")
        trainer = cls(BrainConfig(**state["brain_config"]), TrainConfig(**state["train_config"]), device)
        trainer.model.load_state_dict(state["model"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.step = state["step"]
        trainer.stream.generator.set_state(state["stream_rng"])
        trainer.replay.load_state_dict(state["replay"])
        trainer.history = state["history"]
        torch.set_rng_state(state["torch_rng"])
        if trainer.device.type == "cuda" and state["cuda_rng"]:
            if len(state["cuda_rng"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        return trainer


def train_run(trainer, steps, output, log_every=100):
    if steps < 1 or log_every < 1:
        raise ValueError("steps and log_every must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    eval_options = {"device": trainer.device, "seed": trainer.config.seed + 100000,
                    "delay": trainer.config.delay}
    initial = evaluate(trainer.model, **eval_options)
    atomic_json(output / "initial_eval.json", initial)
    started = time.perf_counter()
    start_step = trainer.step
    try:
        for i in range(steps):
            metrics = trainer.train_step()
            if i == 0 or trainer.step % log_every == 0 or i == steps-1:
                trainer.history.append(metrics)
                print(f"step={metrics['step']:5d} loss={metrics['loss']:.4f} train_acc={metrics['accuracy']:.3f}", flush=True)
            if trainer.step % log_every == 0:
                trainer.save(output / "checkpoint.pt")
    except KeyboardInterrupt:
        # A signal may occur partway through a step; keep that distinct from the last periodic checkpoint.
        trainer.save(output / "interrupted.pt")
        print("Interrupted. Saved interrupted.pt (step may be partial); last checkpoint.pt remains available.", flush=True)
        raise
    elapsed = time.perf_counter()-started
    trainer.save(output / "checkpoint.pt")
    final = evaluate(trainer.model, **eval_options)
    longer = evaluate(trainer.model, **{**eval_options, "delay": trainer.config.delay+4})
    atomic_json(output / "final_eval.json", final)
    atomic_json(output / "longer_delay_eval.json", longer)
    report = {"version": __version__, "device": str(trainer.device), "python": platform.python_version(),
              "torch": str(torch.__version__), "threads": torch.get_num_threads(),
              "parameters": sum(p.numel() for p in trainer.model.parameters()),
              "region_parameters": trainer.model.region_parameter_counts(),
              "start_step": start_step, "end_step": trainer.step, "training_seconds": elapsed,
              "brain_config": asdict(trainer.model.config), "train_config": asdict(trainer.config),
              "initial": initial, "final": final, "longer_delay": longer, "history": trainer.history}
    atomic_json(output / "report.json", report)
    print(f"Saved {output / 'checkpoint.pt'}; evaluation accuracy={final['macro_accuracy']:.3f}; training_seconds={elapsed:.1f}", flush=True)
    return report
