"""Short throughput measurement for the broad cognitive learner, no selection."""
import argparse
import json
from pathlib import Path
import time

import torch

from experiments.train_cognitive import CognitiveTrainer, TRAIN_FAMILIES, make_banks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    banks = make_banks(count=32)
    rows = []
    for mode in ("recurrent", "episodic"):
        for device in ("cpu", "cuda"):
            trainer = CognitiveTrainer(banks, seed=2201, memory_mode=mode, device=device, batch_size=64)
            for i in range(3):
                trainer.step(TRAIN_FAMILIES[i % 3])
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            for i in range(12):
                trainer.step(TRAIN_FAMILIES[i % 3])
            if device == "cuda":
                torch.cuda.synchronize()
            seconds = time.monotonic() - start
            row = {"mode": mode, "device": device, "batch_size": 64, "steps": 12,
                   "seconds": seconds, "updates_per_second": 12 / seconds,
                   "dialogues_per_second": 12 * 64 / seconds,
                   "max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del trainer
            if device == "cuda":
                torch.cuda.empty_cache()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
        "measurements": rows, "not_capability_evidence": True}, indent=2))


if __name__ == "__main__":
    main()
