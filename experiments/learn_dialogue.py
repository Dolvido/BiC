"""Freeze a local English learning protocol, then audit all models after training."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from brain_in_computer.dialogue_curriculum import generate_dialogues
from brain_in_computer.dialogue_learning import DialogueConfig, DialogueLoop, FOCI, fingerprint
from brain_in_computer.dialogue_student import DialogueSession
from brain_in_computer.learning_loop import atomic_json


def _train(job):
    directory, config, cycles, hours = job
    torch.set_num_threads(1)
    loop = DialogueLoop(directory, DialogueConfig(**config))
    with (directory / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        loop.run(hours, max_cycles=cycles)
    return {"seed": loop.config.seed, "directory": str(directory), "cycle": loop.state["cycle"],
            "promotions": loop.state["promotions"], "updates": loop.state["updates"],
            "wall_seconds": loop.state["wall_seconds"],
            "development_query_accuracy": loop.state["development"]["query_accuracy"]}


def train(output, cycles=40, hours=.12, jobs=3):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    configs = [asdict(DialogueConfig(seed=s)) for s in (501, 503, 509)]
    plan = {"schema": "bic-English-validation-v1", "configs": configs, "cycles": cycles,
            "per_run_hours": hours, "jobs": jobs, "source": fingerprint(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "audit_seed": 900_000_001, "audit_count_per_focus": 128,
            "gates": {"mean_query_accuracy": .8, "mean_counterfactual_accuracy": .6,
                      "text_and_memory_query_margin": .10, "all_resume_exact": True},
            "demonstration_seed": 501,
            "meaning": "Restricted language/context test, not adult English. Report every seed. "
                       "Primary counterfactual score requires both opposite answers correct. "
                       "All three training runs finish before audit; no audit-based model/tutor selection."}
    atomic_json(output / "plan.json", plan)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(_train, [(output / f'seed-{c["seed"]}', c, cycles, hours) for c in configs]))
    result = {"runs": results, "wall_seconds": time.monotonic() - started,
              "all_cycles_completed": all(r["cycle"] == cycles for r in results)}
    atomic_json(output / "training-complete.json", result)
    print(json.dumps(result, indent=2))


def evaluate(output):
    torch.set_num_threads(1)
    output = Path(output)
    if (output / "report.json").exists():
        raise FileExistsError("English audit already completed")
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    training = json.loads((output / "training-complete.json").read_text(encoding="utf-8"))
    if plan["source"] != fingerprint():
        raise ValueError("English source changed since plan")
    rows = []
    for row in training["runs"]:
        loop = DialogueLoop(row["directory"], resume=True)
        report = loop.audit(seed=plan["audit_seed"], count=plan["audit_count_per_focus"])
        # A held-out dialogue checks continuation, not model selection. Neither
        # the transcript nor its labels ever returns to replay or the controller.
        episode = generate_dialogues(plan["audit_seed"], 1, split="audit", focus="revision")[0]
        session = DialogueSession(loop.model())
        transcript = []
        for turn in episode["turns"][:3]:
            transcript.append({"text": turn["text"], "expected_action": turn["target"],
                               "student": session.step(turn["observations"], turn["text"])})
        path = Path(row["directory"]) / "audit-session.pt"
        session.save(path)
        restored = DialogueSession.load(path, device=loop.config.device)
        exact = True
        for turn in episode["turns"][3:]:
            a = session.step(turn["observations"], turn["text"])
            b = restored.step(turn["observations"], turn["text"])
            exact = exact and a == b
            transcript.append({"text": turn["text"], "expected_action": turn["target"], "student": a})
        rows.append({**row, "audit": report, "restart_exact": exact, "transcript": transcript})
    summary = {}
    for condition in ("student", "reset_each_turn", "blank_english", "initial"):
        values = [r["audit"][condition]["query_accuracy"] for r in rows]
        summary[condition] = {"mean_query_accuracy": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "per_focus": {f: statistics.mean(r["audit"][condition]["per_focus"][f]["query_accuracy"] for r in rows) for f in FOCI},
            "mean_counterfactual_accuracy": statistics.mean(r["audit"][condition]["per_focus"][f]["counterfactual_accuracy"]
                                                              for r in rows for f in FOCI),
            "mean_query_reply_exact_accuracy": statistics.mean(r["audit"][condition]["per_focus"][f]["query_reply_exact_accuracy"]
                                                                for r in rows for f in FOCI)}
    learned = summary["student"]
    result = {"plan": plan, "training": training, "runs": rows, "summary": summary,
              "gates": {"query_accuracy": learned["mean_query_accuracy"] >= plan["gates"]["mean_query_accuracy"],
                        "counterfactual_accuracy": learned["mean_counterfactual_accuracy"] >= plan["gates"]["mean_counterfactual_accuracy"],
                        "text_and_memory_margin": all(learned["mean_query_accuracy"] - summary[c]["mean_query_accuracy"] >=
                                                          plan["gates"]["text_and_memory_query_margin"]
                                                          for c in ("reset_each_turn", "blank_english")),
                        "all_resume_exact": all(r["restart_exact"] for r in rows),
                        "all_checkpoints_unchanged": all(r["audit"]["checkpoint_unchanged"] for r in rows)}}
    atomic_json(output / "report.json", result)
    print(json.dumps({"summary": summary, "gates": result["gates"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    (train if args.command == "train" else evaluate)(args.output)


if __name__ == "__main__":
    main()
