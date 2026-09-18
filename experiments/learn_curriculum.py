"""Prospective local comparison; finish all training before opening any audit."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from brain_in_computer.curriculum import SKILLS, CURRICULUM
from brain_in_computer.learning_loop import LearningLoop, LoopConfig, atomic_json, source_digest


def _train(job):
    directory, config, cycles, hours = job
    torch.set_num_threads(1)
    loop = LearningLoop(directory, LoopConfig(**config))
    loop.run(hours, max_cycles=cycles, emit=False)
    return {"directory": str(directory), "seed": loop.config.seed, "selection": loop.config.selection,
            "cycles": loop.state["cycle"], "promotions": loop.state["promotions"],
            "updates": loop.state["updates"], "wall_seconds": loop.state["wall_seconds"],
            "candidate_seconds": loop.state["candidate_seconds"],
            "development_accuracy": loop.state["development"]["accuracy"]}


def train(output, *, seeds=(101, 202, 303), cycles=120, hours=.2, jobs=3):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    configurations = [asdict(LoopConfig(seed=seed, selection=condition))
                      for condition in ("progress", "round_robin") for seed in seeds]
    plan = {"schema": "bic-curriculum-validation-v1", "seeds": list(seeds), "cycles": cycles,
            "per_run_max_hours": hours, "jobs": jobs, "configurations": configurations,
            "source_sha256": source_digest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "audit_seed": 800_000_001, "audit_count_per_skill": 512,
            "gates": {"mean_progress_accuracy": .75, "each_progress_seed_gain_over_initial": .20},
            "interpretation": "Three seed-level replications; shared audit scenarios. Round-robin compares "
                              "the scheduler at equal cycle/update budgets, not unequal architecture/tutor assistance. "
                              "Procedural tutoring isolates student learning; live LLM packet validation is separate. "
                              "No assumed progress-policy superiority. Audit only after every training job completes."}
    atomic_json(output / "plan.json", plan)
    workload = [(output / f'{c["selection"]}-{c["seed"]}', c, cycles, hours) for c in configurations]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        results = []
        for result in pool.map(_train, workload):
            results.append(result)
            print(json.dumps(result), flush=True)
            atomic_json(output / "training-progress.json", results)
    summary = {"runs": results, "total_wall_seconds": time.monotonic() - started,
               "all_cycles_completed": all(r["cycles"] == cycles for r in results)}
    atomic_json(output / "training-complete.json", summary)
    return summary


def evaluate(output):
    torch.set_num_threads(1)
    output = Path(output)
    if (output / "report.json").exists():
        raise FileExistsError("validation evidence already exists")
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    training = json.loads((output / "training-complete.json").read_text(encoding="utf-8"))
    if plan["source_sha256"] != source_digest():
        raise ValueError("source changed since prospective training plan")
    rows = []
    for record in training["runs"]:
        loop = LearningLoop(record["directory"], resume=True)
        audit = loop.audit(seed=plan["audit_seed"], count=plan["audit_count_per_skill"])
        rows.append({**record, "audit": audit})
    conditions = {}
    for name in ("progress", "round_robin"):
        selected = [r for r in rows if r["selection"] == name]
        values = [r["audit"]["student"]["accuracy"] for r in selected]
        conditions[name] = {"mean_accuracy": statistics.mean(values),
                            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                            "per_skill": {s: statistics.mean(r["audit"]["student"]["per_skill"][s]["accuracy"]
                                                             for r in selected) for s in SKILLS},
                            "mean_initial_accuracy": statistics.mean(r["audit"]["untrained_baseline"]["accuracy"]
                                                                      for r in selected)}
    progress = [r for r in rows if r["selection"] == "progress"]
    differences = [next(r for r in rows if r["selection"] == "progress" and r["seed"] == seed)["audit"]["student"]["accuracy"] -
                   next(r for r in rows if r["selection"] == "round_robin" and r["seed"] == seed)["audit"]["student"]["accuracy"]
                   for seed in plan["seeds"]]
    report = {"schema": plan["schema"], "plan": plan, "training": training, "runs": rows,
              "conditions": conditions, "paired_progress_minus_round_robin": differences,
              "mean_paired_difference": statistics.mean(differences),
              "uniform_valid_label_chance": statistics.mean(1 / len(CURRICULUM[s]["valid_targets"]) for s in SKILLS),
              "gates": {
                  "mean_progress_accuracy": conditions["progress"]["mean_accuracy"] >= plan["gates"]["mean_progress_accuracy"],
                  "each_progress_seed_gain_over_initial": all(r["audit"]["accuracy_gain"] >= plan["gates"]["each_progress_seed_gain_over_initial"]
                                                               for r in progress),
                  "all_checkpoints_unchanged_by_audit": all(r["audit"]["checkpoint_unchanged"] for r in rows)}}
    atomic_json(output / "report.json", report)
    print(json.dumps({"conditions": conditions, "gates": report["gates"]}, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--cycles", type=int, default=120)
    parser.add_argument("--hours", type=float, default=.2)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "train":
        return train(args.output, cycles=args.cycles, hours=args.hours, jobs=args.jobs)
    return evaluate(args.output)


if __name__ == "__main__":
    main()
