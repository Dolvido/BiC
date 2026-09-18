"""Measure the current English learner on local CPU and RTX hardware."""
import argparse
import gc
import json
from pathlib import Path
import platform
import time

import torch

from brain_in_computer.dialogue_curriculum import generate_dialogues
from brain_in_computer.dialogue_student import build_dialogue_student, train_dialogue_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    episodes = generate_dialogues(8_000, 512, "train", "mixed")
    initial = build_dialogue_student(1101).state_dict()
    report = {"python": platform.python_version(), "torch": str(torch.__version__),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "cuda": torch.version.cuda, "cpu_threads": 1, "measurements": [],
              "scope": "Current six-turn English model; no final evaluation or model selection by quality"}
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        for batch_size in (32, 128, 256):
            warm = train_dialogue_candidate(initial, episodes, seed=1101, steps=5,
                                            batch_size=batch_size, device=device)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.monotonic()
            result = train_dialogue_candidate(warm["state_dict"], episodes, seed=1102,
                     optimizer_state=warm["optimizer_state"], steps=30, batch_size=batch_size, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            seconds = time.monotonic() - start
            row = {"device": device, "batch_size": batch_size, "updates": result["updates"],
                   "seconds": seconds, "updates_per_second": result["updates"] / seconds,
                   "dialogues_per_second": result["episodes_seen"] / seconds,
                   "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
                   "finite_loss": result["loss"]}
            report["measurements"].append(row)
            (path / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
            del warm, result
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
