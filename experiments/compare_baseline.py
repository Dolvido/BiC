"""Compare a parameter-matched GRU against the modular model on identical data.

Example (reuses an already-trained modular checkpoint):
    python experiments/compare_baseline.py --checkpoint runs/reference/checkpoint.pt \
        --steps 600 --output runs/comparison

Use --train-modular instead of --checkpoint to train both models from scratch.
This is a single-seed engineering comparison, not evidence for or against AGI.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import platform
import sys
import time

# Allow the example above from an unpacked source tree without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from brain_in_computer.model import BrainConfig
from brain_in_computer.training import Trainer, TrainConfig, atomic_json, evaluate


class MonolithicGRU(nn.Module):
    """One recurrent network receiving exactly the same observation channels.

    All action, sensory-classification, prediction, and confidence heads read
    the same hidden state. The embedding is an input adapter, not a region.
    Recurrence resets on every call, as it does in Brain.forward.
    """

    def __init__(self, config: BrainConfig, hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        input_size = config.observation_dim + 2 + config.hidden_size
        self.recurrent = nn.GRU(input_size, hidden_size, batch_first=True)
        self.action = nn.Linear(hidden_size, config.num_actions)
        self.visual = nn.Linear(hidden_size, config.num_actions)
        self.auditory = nn.Linear(hidden_size, config.num_actions)
        self.prediction = nn.Linear(hidden_size, config.observation_dim)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, observations, ablate=()):
        if ablate:
            raise ValueError("Named brain-region ablations do not apply to the monolithic baseline")
        sensory = [observations[k] for k in ("visual", "auditory", "body", "feedback")]
        inputs = torch.cat((*sensory, self.embedding(observations["tokens"])), dim=-1)
        state, _ = self.recurrent(inputs)
        return {
            "logits": self.action(state),
            "visual_logits": self.visual(state),
            "auditory_logits": self.auditory(state),
            "prediction": self.prediction(state),
            "value": self.value(state).squeeze(-1),
        }


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def baseline_parameter_count(config, hidden_size):
    inputs = config.observation_dim + 2 + config.hidden_size
    outputs = 3 * config.num_actions + config.observation_dim + 1
    embedding = config.vocab_size * config.hidden_size
    # GRU input/recurrent weights and two biases, followed by five affine heads.
    return embedding + 3 * hidden_size * (inputs + hidden_size + 2) + outputs * (hidden_size + 1)


def matching_width(config, target):
    upper = 1
    while baseline_parameter_count(config, upper) < target:
        upper *= 2
    return min(range(1, upper + 1), key=lambda width: abs(baseline_parameter_count(config, width) - target))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_steps(trainer, steps, log_every):
    synchronize(trainer.device)
    started = time.perf_counter()
    history = []
    for i in range(steps):
        metrics = trainer.train_step()
        if i == 0 or trainer.step % log_every == 0 or i == steps - 1:
            history.append(metrics)
            print(f"{type(trainer.model).__name__} step={trainer.step} loss={metrics['loss']:.4f} "
                  f"train_accuracy={metrics['accuracy']:.3f}", flush=True)
    synchronize(trainer.device)
    return time.perf_counter() - started, history


def tensor_digest(tensor):
    return hashlib.sha256(bytes(tensor.cpu().tolist())).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_pair(modular, baseline, config, args):
    options = {"seed": config.seed + 100000, "batches": args.eval_batches,
               "batch_size": args.eval_batch_size, "device": args.device}
    scores = {}
    for label, delay in (("trained_delay", config.delay), ("longer_delay", config.delay + 4)):
        scores[label] = {}
        for name, model in (("modular", modular), ("gru", baseline)):
            synchronize(next(model.parameters()).device)
            started = time.perf_counter()
            result = evaluate(model, **options, delay=delay)
            synchronize(next(model.parameters()).device)
            scores[label][name] = {**result, "evaluation_seconds": time.perf_counter() - started}
        scores[label]["modular_minus_gru_accuracy"] = (
            scores[label]["modular"]["macro_accuracy"] - scores[label]["gru"]["macro_accuracy"])
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--train-modular", action="store_true")
    parser.add_argument("--steps", type=int, help="Total updates; must equal checkpoint step count (default: its count, or 600)")
    parser.add_argument("--output", type=Path, default=Path("runs/comparison"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, help="Fresh modular training seed; checkpoint mode inherits its seed")
    parser.add_argument("--hidden-size", type=int, help="Fresh modular hidden width; checkpoint mode inherits its width")
    args = parser.parse_args(argv)
    for name in ("threads", "eval_batches", "eval_batch_size", "log_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.checkpoint and (args.seed is not None or args.hidden_size is not None):
        parser.error("Checkpoint mode inherits --seed and --hidden-size; omit those options")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu or install a compatible PyTorch build")
    torch.set_num_threads(args.threads)
    output = args.output
    modular_training_seconds = None
    modular_history = []
    try:
        if args.checkpoint:
            modular_trainer = Trainer.load(args.checkpoint, args.device)
            steps = args.steps if args.steps is not None else modular_trainer.step
            if steps != modular_trainer.step or steps < 1:
                parser.error("Baseline total updates must equal the modular checkpoint's positive cumulative step count")
            source_info = {"checkpoint": str(args.checkpoint), "sha256": file_digest(args.checkpoint),
                           "mode": "existing_checkpoint"}
        else:
            steps = args.steps if args.steps is not None else 600
            config = TrainConfig(seed=args.seed if args.seed is not None else 7)
            brain_config = BrainConfig(hidden_size=args.hidden_size if args.hidden_size is not None else 48)
            modular_trainer = Trainer(brain_config, config, args.device)
            modular_training_seconds, modular_history = run_steps(modular_trainer, steps, args.log_every)
            output.mkdir(parents=True, exist_ok=True)
            modular_trainer.save(output / "modular_checkpoint.pt")
            source_info = {"checkpoint": str(output / "modular_checkpoint.pt"),
                           "sha256": file_digest(output / "modular_checkpoint.pt"), "mode": "fresh_training"}

        config = modular_trainer.config
        brain_config = modular_trainer.model.config
        target_count = parameter_count(modular_trainer.model)
        width = matching_width(brain_config, target_count)

        # Trainer owns its independent task and rehearsal RNGs. Reusing its
        # train_step ensures identical batching, labels, replay, objective,
        # optimizer settings, and clipping without copying the learning code.
        baseline_trainer = Trainer(brain_config, TrainConfig(**asdict(config)), args.device)
        torch.manual_seed(config.seed)
        baseline_trainer.model = MonolithicGRU(brain_config, width).to(args.device)
        baseline_trainer.optimizer = torch.optim.AdamW(
            baseline_trainer.model.parameters(), lr=config.learning_rate, weight_decay=0.0001)
        actual_count = parameter_count(baseline_trainer.model)
        if actual_count != baseline_parameter_count(brain_config, width):
            raise RuntimeError("Baseline parameter-count formula does not match the constructed model")
        baseline_seconds, baseline_history = run_steps(baseline_trainer, steps, args.log_every)
        stream_match = torch.equal(modular_trainer.stream.generator.get_state(), baseline_trainer.stream.generator.get_state())
        replay_match = (modular_trainer.replay.seen == baseline_trainer.replay.seen
                        and torch.equal(modular_trainer.replay.generator.get_state(), baseline_trainer.replay.generator.get_state()))
        if not stream_match or not replay_match:
            raise RuntimeError("Training/replay RNG states differ: the checkpoint does not match a fresh run of this configuration")
        scores = evaluate_pair(modular_trainer.model, baseline_trainer.model, config, args)
        relative_difference = (actual_count - target_count) / target_count
        report = {
            "interpretation": "Single-seed engineering check on synthetic tasks, not a research conclusion or an AGI assessment.",
            "limitations": ["Same optimizer settings are not independently tuned for each architecture.",
                            "Equal update and parameter budgets do not imply equal compute or equal recurrent-state capacity.",
                            "Random test episodes can share symbolic combinations with training; this is not compositional transfer.",
                            "Reported per-task standard errors cover evaluation sampling, not training-seed variation.",
                            "An existing checkpoint must have been trained with this code and unchanged settings for historical data parity."],
            "reference": source_info,
            "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                         "device": args.device, "threads": torch.get_num_threads()},
            "seed_controls": {"initialization_seed": config.seed, "training_stream_seed": config.seed + 1,
                              "replay_seed": config.seed + 2, "evaluation_seed": config.seed + 100000,
                              "training_rng_match": stream_match, "replay_rng_and_seen_match": replay_match,
                              "final_stream_rng_sha256": tensor_digest(baseline_trainer.stream.generator.get_state())},
            "training_config": asdict(config), "modular_config": asdict(brain_config),
            "total_updates_each": steps,
            "modular": {"parameters": target_count, "training_seconds": modular_training_seconds,
                        "training_time_note": "Not measured for an existing checkpoint" if args.checkpoint else "Measured in this run",
                        "history": modular_history},
            "gru": {"hidden_size": width, "embedding_size": brain_config.hidden_size, "parameters": actual_count,
                    "relative_parameter_difference": relative_difference, "within_ten_percent": abs(relative_difference) <= 0.1,
                    "training_seconds": baseline_seconds, "history": baseline_history},
            "evaluation": scores,
        }
        output.mkdir(parents=True, exist_ok=True)
        # This is an inference artifact, not a resumable Trainer checkpoint.
        baseline_path = output / "gru_weights.pt"
        temporary = baseline_path.with_name(baseline_path.name + ".tmp")
        torch.save({"architecture": "MonolithicGRU", "brain_config": asdict(brain_config),
                    "hidden_size": width, "model": baseline_trainer.model.state_dict(),
                    "train_config": asdict(config), "total_updates": steps}, temporary)
        temporary.replace(baseline_path)
        atomic_json(output / "comparison.json", report)
        print(f"Saved {output / 'comparison.json'}; parameters modular={target_count}, gru={actual_count}; "
              f"trained-delay accuracy modular={scores['trained_delay']['modular']['macro_accuracy']:.3f}, "
              f"gru={scores['trained_delay']['gru']['macro_accuracy']:.3f}", flush=True)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
