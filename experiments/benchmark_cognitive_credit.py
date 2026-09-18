"""Short isolated compute probe; these temporary learners are never audited."""
import argparse
import json
from pathlib import Path
import time

import torch

from experiments.cognitive_credit import CreditTrainer
from experiments.train_cognitive_credit import TRAIN_FAMILIES, make_banks, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    banks = make_banks(seed=21_000_000, count=32)
    results = []
    for coefficient in (0., .3):
        trainer = CreditTrainer(banks, seed=2301, aux_weight=coefficient, device="cuda", batch_size=64)
        for update in range(3):
            trainer.step(TRAIN_FAMILIES[update % 3])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        for update in range(12):
            trainer.step(TRAIN_FAMILIES[update % 3])
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        results.append({"aux_weight": coefficient, "updates": 12, "seconds": seconds,
            "updates_per_second": 12 / seconds, "episode_exposures_per_second": 768 / seconds,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
        del trainer
        torch.cuda.empty_cache()
    report = {"source_sha256": source_hashes(), "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__), "batch_size": 64, "warmup_updates_per_arm": 3,
        "not_capability_evidence": True, "measurements": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
