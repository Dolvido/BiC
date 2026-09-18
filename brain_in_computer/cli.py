"""Local command-line training and evaluation; no cloud credentials required."""

import argparse
from dataclasses import asdict
import json
import sys

import torch

from .model import Brain, BrainConfig, REGION_NAMES
from .tasks import TASK_NAMES, TaskStream
from .training import Trainer, TrainConfig, atomic_json, evaluate, train_run


def compute_options(parser):
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=2, help="CPU intra-op threads; small models often prefer 1-2")


def parser():
    p = argparse.ArgumentParser(prog="brain-in-computer", description="Train and inspect a small brain-inspired neural cluster.", allow_abbrev=False)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check installation and active release files; performs no training")
    sub.add_parser("launch", help="Start both local labs using the active release manifest")
    inspect = sub.add_parser("inspect", help="Show actual networks, sizes, and pathways", allow_abbrev=False)
    inspect.add_argument("--hidden-size", type=int, default=48)
    inspect.add_argument("--checkpoint")
    compute_options(inspect)
    train = sub.add_parser("train", help="Train a new randomly initialized model", allow_abbrev=False)
    train.add_argument("--steps", type=int, default=600)
    train.add_argument("--hidden-size", type=int, default=48)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--delay", type=int, default=2)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--learning-rate", type=float, default=0.002)
    train.add_argument("--replay-capacity", type=int, default=256)
    train.add_argument("--replay-fraction", type=float, default=0.25)
    train.add_argument("--curriculum", choices=("mixed", "staged"), default="mixed")
    train.add_argument("--stage-steps", type=int, default=200)
    train.add_argument("--output", default="runs/first")
    train.add_argument("--log-every", type=int, default=100)
    compute_options(train)
    resume = sub.add_parser("resume", help="Continue saved weights, optimizer, task stream, and rehearsal", allow_abbrev=False)
    resume.add_argument("--checkpoint", required=True)
    resume.add_argument("--steps", type=int, default=600, help="Additional optimizer steps")
    resume.add_argument("--output", default="runs/resumed")
    resume.add_argument("--log-every", type=int, default=100)
    compute_options(resume)
    ev = sub.add_parser("evaluate", help="Test a saved model on fresh generated episodes", allow_abbrev=False)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--seed", type=int, default=100007)
    ev.add_argument("--batches", type=int, default=8)
    ev.add_argument("--batch-size", type=int, default=64)
    ev.add_argument("--delay", type=int, default=2)
    ev.add_argument("--ablate", nargs="+", choices=REGION_NAMES, default=[])
    ev.add_argument("--output")
    compute_options(ev)
    demo = sub.add_parser("demo", help="Show model decisions for individual symbolic tasks", allow_abbrev=False)
    demo.add_argument("--checkpoint", required=True)
    demo.add_argument("--task", choices=TASK_NAMES, default="delayed_match")
    demo.add_argument("--delay", type=int, default=2)
    demo.add_argument("--count", type=int, default=8)
    demo.add_argument("--seed", type=int, default=200007)
    compute_options(demo)
    budget = sub.add_parser("language-plan", help="Estimate language parameter budgets; performs no training", allow_abbrev=False)
    budget.add_argument("--parameters", nargs="+", type=int, default=[1000000, 10000000, 100000000, 3000000000])
    budget.add_argument("--corpus-tokens", type=int)
    budget.add_argument("--epochs", type=int, default=1)
    budget.add_argument("--measured-tokens-per-second", type=float)
    budget.add_argument("--output")
    compute_options(budget)
    readiness = sub.add_parser("language-check", help="Probe random language networks without updating weights", allow_abbrev=False)
    readiness.add_argument("--checkpoint", help="Optional existing SYMBOLIC core checkpoint; language stays untrained")
    readiness.add_argument("--language-hidden-size", type=int, default=128)
    readiness.add_argument("--embedding-size", type=int, default=64)
    readiness.add_argument("--seed", type=int, default=17)
    readiness.add_argument("--output")
    compute_options(readiness)
    computer_train = sub.add_parser("computer-train", help="Train a pixel desktop agent and grounded text replies from demonstrations", allow_abbrev=False)
    computer_train.add_argument("--steps", type=int, default=5000)
    computer_train.add_argument("--demonstrations", type=int, default=2100)
    computer_train.add_argument("--batch-size", type=int, default=32)
    computer_train.add_argument("--seed", type=int, default=31)
    computer_train.add_argument("--learning-rate", type=float, default=0.001)
    computer_train.add_argument("--stop-fraction", type=float, default=0.35)
    computer_train.add_argument("--language-weight", type=float, default=3.0)
    computer_train.add_argument("--initialize-from", help="Start from these computer weights with a new optimizer")
    computer_train.add_argument("--output", default="runs/new-computer")
    computer_train.add_argument("--hidden-size", type=int, help="Core region width; defaults to 48 for a new model")
    computer_train.add_argument("--language-hidden-size", type=int, help="Language width; defaults to 96")
    computer_train.add_argument("--embedding-size", type=int, help="Byte embedding width; defaults to 48")
    computer_train.add_argument("--visual-features", type=int, help="Image feature width; defaults to 32")
    computer_train.add_argument("--max-history", type=int, help="Observation history; defaults to 3")
    computer_train.add_argument("--log-every", type=int, default=250)
    compute_options(computer_train)
    computer_evaluate = sub.add_parser("computer-evaluate", help="Evaluate a computer-use checkpoint in closed-loop desktop episodes", allow_abbrev=False)
    computer_evaluate.add_argument("--checkpoint", required=True, help="Computer-use checkpoint; symbolic checkpoints are incompatible")
    computer_evaluate.add_argument("--episodes-per-task", type=int, default=64)
    computer_evaluate.add_argument("--seed", type=int, default=900031)
    computer_evaluate.add_argument("--split", choices=("train", "validation", "test"), default="test")
    computer_evaluate.add_argument("--shift", action="store_true", help="Evaluate small button-position shifts")
    computer_evaluate.add_argument("--blank-pixels", action="store_true")
    computer_evaluate.add_argument("--blank-prompt", action="store_true")
    computer_evaluate.add_argument("--ablate", nargs="+", choices=REGION_NAMES, default=[])
    computer_evaluate.add_argument("--output")
    compute_options(computer_evaluate)
    computer_chat = sub.add_parser("computer-chat", help="Interact with a saved agent in its simulated desktop", allow_abbrev=False)
    computer_chat.add_argument("--checkpoint", required=True, help="Computer-use checkpoint; symbolic checkpoints are incompatible")
    computer_chat.add_argument("--seed", type=int, default=0)
    compute_options(computer_chat)
    computer_serve = sub.add_parser("computer-serve", help="Open a local browser interface to a saved desktop agent", allow_abbrev=False)
    computer_serve.add_argument("--checkpoint", required=True, help="Computer-use checkpoint; symbolic checkpoints are incompatible")
    computer_serve.add_argument("--host", default="127.0.0.1", help="Loopback host for the local interface")
    computer_serve.add_argument("--port", type=int, default=8765)
    compute_options(computer_serve)
    teaching = sub.add_parser("memory-serve", help="Teach names to visual objects and retain associations across restarts", allow_abbrev=False)
    teaching.add_argument("--checkpoint", required=True, help="Trained associative visual encoder checkpoint")
    teaching.add_argument("--regional-checkpoint", help="Optional trained regional controller using the same visual encoder")
    teaching.add_argument("--memory", default="runs/personal-memory.json")
    teaching.add_argument("--host", default="127.0.0.1")
    teaching.add_argument("--port", type=int, default=8766)
    teaching.add_argument("--seed", type=int, default=7)
    compute_options(teaching)
    benchmark = sub.add_parser("benchmark", help="Measure actual optimizer time and process memory across model sizes", allow_abbrev=False)
    benchmark.add_argument("--output", default="runs/benchmarks/home.json")
    benchmark.add_argument("--batch-size", type=int, default=8)
    benchmark.add_argument("--warmup-steps", type=int, default=2)
    benchmark.add_argument("--measured-steps", type=int, default=3)
    benchmark.add_argument("--max-parameters", type=int, default=50000000)
    benchmark.add_argument("--max-seconds", type=float, default=120.0)
    benchmark.add_argument("--profiles", nargs="+", choices=("small", "medium", "large", "home", "stretch", "extended"))
    compute_options(benchmark)
    return p


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in ("doctor", "launch"):
        from .launcher import main as launcher_main
        return launcher_main(arguments)
    p = parser()
    args = p.parse_args(arguments)
    if args.threads < 1:
        p.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable. Use --device cpu, or install a compatible CUDA PyTorch build.")
    try:
        if args.command == "computer-train":
            from .computer_use.training import ComputerTrainConfig, train_computer
            from .computer_use.model import ComputerConfig
            config = ComputerTrainConfig(**{key: getattr(args, key) for key in ComputerTrainConfig.__dataclass_fields__})
            supplied = {key: getattr(args, key) for key in ComputerConfig.__dataclass_fields__ if getattr(args, key) is not None}
            model_config = ComputerConfig(**supplied) if supplied else None
            train_computer(config=config, output=args.output, device=device, log_every=args.log_every,
                           initialize_from=args.initialize_from, model_config=model_config)
        elif args.command == "computer-evaluate":
            from .computer_use.training import evaluate_computer, load_computer_checkpoint
            model = load_computer_checkpoint(args.checkpoint, device=device)
            result = evaluate_computer(
                model, seed=args.seed, episodes_per_task=args.episodes_per_task,
                split=args.split, shift=args.shift, ablate=args.ablate,
                blank_pixels=args.blank_pixels, blank_prompt=args.blank_prompt,
            )
            if args.output:
                atomic_json(args.output, result)
            print(json.dumps(result, indent=2))
        elif args.command == "computer-chat":
            from .computer_use.environment import ACTION_NAMES, MiniDesktop
            from .computer_use.training import load_computer_checkpoint, run_turn
            model = load_computer_checkpoint(args.checkpoint, device=device)
            env = MiniDesktop(seed=args.seed)
            print("Simulated desktop chat. Enter /reset for a new scene or /quit to leave.")
            while True:
                try:
                    prompt = input("You: ").strip()
                except EOFError:
                    print()
                    break
                if prompt == "/quit":
                    break
                if prompt == "/reset":
                    env.reset()
                    print("Desktop reset.")
                    continue
                if not prompt:
                    continue
                try:
                    result = run_turn(model, env, prompt)
                except ValueError as error:
                    print(str(error), file=sys.stderr)
                    continue
                print(f"Brain: {result['reply']}")
                print("Actions: " + ", ".join(ACTION_NAMES[action] for action in result["actions"]))
        elif args.command == "computer-serve":
            from .computer_use.ui import serve
            serve(args.checkpoint, host=args.host, port=args.port, device=device)
        elif args.command == "memory-serve":
            from .teaching_ui import serve
            serve(args.checkpoint, memory_file=args.memory, host=args.host, port=args.port,
                  device=device, seed=args.seed, regional_checkpoint=args.regional_checkpoint)
        elif args.command == "benchmark":
            from .benchmark import benchmark_suite
            result = benchmark_suite(args.output, device=device, threads=args.threads,
                                     profiles=args.profiles,
                                     batch_size=args.batch_size, warmup_steps=args.warmup_steps,
                                     measured_steps=args.measured_steps, max_parameters=args.max_parameters,
                                     max_seconds=args.max_seconds)
            print(json.dumps(result, indent=2))
        elif args.command == "language-plan":
            from .resources import estimate_language_budget
            result = {"language_training_performed": False,
                      "estimates": [estimate_language_budget(n, corpus_tokens=args.corpus_tokens,
                                    epochs=args.epochs, measured_tokens_per_second=args.measured_tokens_per_second)
                                    for n in args.parameters]}
            if args.output:
                atomic_json(args.output, result)
            print(json.dumps(result, indent=2))
        elif args.command == "language-check":
            from .language import LanguageConfig
            from .language_readiness import readiness_report
            config = LanguageConfig(hidden_size=args.language_hidden_size, embedding_size=args.embedding_size)
            result = readiness_report(args.checkpoint, config, device=device, seed=args.seed)
            if args.output:
                atomic_json(args.output, result)
            print(json.dumps(result, indent=2))
        elif args.command == "inspect":
            model = Trainer.load(args.checkpoint, device).model if args.checkpoint else Brain(BrainConfig(hidden_size=args.hidden_size))
            result = {"config": asdict(model.config), "parameters": sum(p.numel() for p in model.parameters()),
                      "regions": model.region_parameter_counts(), "connectome": model.connectome,
                      "external_inputs": model.external_inputs}
            print(json.dumps(result, indent=2))
        elif args.command == "train":
            cfg = TrainConfig(**{k: getattr(args, k) for k in TrainConfig.__dataclass_fields__})
            train_run(Trainer(BrainConfig(hidden_size=args.hidden_size), cfg, device), args.steps, args.output, args.log_every)
        elif args.command == "resume":
            train_run(Trainer.load(args.checkpoint, device), args.steps, args.output, args.log_every)
        elif args.command == "evaluate":
            trainer = Trainer.load(args.checkpoint, device)
            if args.seed in (trainer.config.seed, trainer.config.seed+1):
                p.error("Choose an evaluation seed different from the model and training-data seeds")
            result = evaluate(trainer.model, seed=args.seed, batches=args.batches,
                              batch_size=args.batch_size, delay=args.delay, device=device, ablate=args.ablate)
            if args.output:
                atomic_json(args.output, result)
            print(json.dumps(result, indent=2))
        elif args.command == "demo":
            trainer = Trainer.load(args.checkpoint, device)
            trainer.model.eval()
            batch = TaskStream(args.seed).sample(args.count, tasks=(args.task,), delay=args.delay, device=device)
            with torch.no_grad():
                probabilities = trainer.model(batch.observations)["logits"][:, -1].softmax(-1).cpu()
            for i, probs in enumerate(probabilities):
                print(json.dumps({"task": args.task, "episode": i+1,
                                  "visual_sequence": batch.observations["visual"][i, :, :4].cpu().tolist(),
                                  "auditory_sequence": batch.observations["auditory"][i, :, :4].cpu().tolist(),
                                  "instruction": batch.observations["tokens"][i].cpu().tolist(),
                                  "prediction": int(probs.argmax()), "expected": int(batch.targets[i, -1]),
                                  "probabilities": [round(float(x), 4) for x in probs]}))
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
