"""Recount saved evaluation records and apply the prospective pilot screen."""
import argparse
import json
from pathlib import Path

from experiments.foundation_tutor_loop_data import digest, native, read, write
from experiments.foundation_layout_evaluation import score_records
from experiments.recurrent_read_screen import compare

ROOT = Path(__file__).resolve().parents[1]


def analyze(directory):
    directory = Path(directory).resolve()
    summary = read(directory / "execution/summary.json")
    launch = read(directory / "launch.json")
    if summary["status"] != "completed" or digest(directory / "launch.json") != summary["launch_sha256"]:
        raise ValueError("completed, matching launch and execution required")
    for name, pin in launch["source_sha256"].items():
        if digest(ROOT / name) != pin:
            raise ValueError("frozen screen or supporting source changed: " + name)
    if set(summary["evaluations"]) != {"baseline", "recurrent"}:
        raise ValueError("both fixed arms required")
    for name, pin in summary["artifact_sha256"].items():
        if digest(directory / "execution" / name) != pin:
            raise ValueError("saved result artifact changed: " + name)
    if digest(directory / "execution/events.jsonl") != summary["journal_sha256"]:
        raise ValueError("execution journal changed")
    recounted, evaluation_episodes = 0, 0
    for arm, endpoints in summary["evaluations"].items():
        if set(endpoints) != {"0", "216", "432", "648"}:
            raise ValueError("complete fixed evaluation endpoints required")
        for step, banks in endpoints.items():
            if set(banks) != {"dev", "train_fit", "retention"}:
                raise ValueError("all three fixed banks required")
            for name, saved in banks.items():
                records = []
                for group in saved["groups"].values():
                    path = ROOT / group["path"]
                    if digest(path) != group["sha256"]:
                        raise ValueError("raw evaluation group changed")
                    scored = read(path)
                    if score_records(scored["raw_records"], expected_bank=scored["bank"]) != scored["metrics"]:
                        raise ValueError("group arithmetic differs")
                    records.extend(scored["raw_records"])
                if score_records(records) != saved["metrics"]:
                    raise ValueError("pooled arithmetic differs")
                recounted += 1
                evaluation_episodes += len(records)
    if recounted != 24 or evaluation_episodes != 12384:
        raise ValueError("complete prospective evaluation census required")
    endpoints = {arm: {name: record["metrics"] for name, record in steps["648"].items()}
                 for arm, steps in summary["evaluations"].items()}
    screen = compare(endpoints["baseline"], endpoints["recurrent"])
    timing = {arm: dict(training=0., training_preparation=0., evaluation=0.) for arm in endpoints}
    for line in native(directory / "execution/events.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("event") == "complete" and entry.get("kind") in ("training", "training_preparation", "evaluation"):
            arm = entry["identity"].split("/")[0]
            timing[arm][entry["kind"]] += entry["wall_seconds"]
    for costs in timing.values():
        costs["training_including_preparation"] = costs["training"] + costs["training_preparation"]
    extra_seconds = timing["recurrent"]["training_including_preparation"] - timing["baseline"]["training_including_preparation"]
    joint = screen["differences"]["dev/overall/anchor_pair_both"]
    extra_correct = joint["candidate"] - joint["baseline"]
    result = dict(status="verified", summary_sha256=digest(directory / "execution/summary.json"),
        launch_sha256=summary["launch_sha256"], recounted_bank_endpoints=recounted,
        evaluation_episode_exposures=evaluation_episodes, screen=screen, endpoint_metrics=endpoints,
        timing=timing, execution_seconds=summary["wall_seconds"],
        data_preparation_seconds=launch["preparation"]["wall_seconds"],
        freeze_seconds=launch["freeze_cost"]["wall_seconds"],
        peak_gpu_allocated_bytes=summary["cuda_peak_allocated_bytes"],
        incremental_correct_pairs=extra_correct, incremental_training_seconds=extra_seconds,
        incremental_correct_pairs_per_extra_training_second=(extra_correct / extra_seconds if extra_seconds > 0 else None),
        independent_neural_evaluation=False, new_model_work=0, teacher_calls=0,
        scope="Pure recount of recorded predictions, not another model evaluation; one exploratory initialization.")
    write(directory / "analysis.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    analyze(parser.parse_args().directory)
