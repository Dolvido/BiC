"""Reproduce inference lesions and memory-length stress tests without training.

Run from the project root: python experiments/run_diagnostics.py --checkpoint runs/reference/checkpoint.pt
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from brain_in_computer.model import REGION_NAMES
from brain_in_computer.training import Trainer, atomic_json, evaluate


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="experiments/diagnostics.json")
    p.add_argument("--seed", type=int, default=300007)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--delays", nargs="+", type=int, default=[2, 6, 14])
    args = p.parse_args()
    if args.threads < 1:
        p.error("threads must be positive")
    torch.set_num_threads(args.threads)
    trainer = Trainer.load(args.checkpoint)
    if args.seed in (trainer.config.seed, trainer.config.seed+1):
        p.error("use an independent evaluation seed")
    start = time.perf_counter()
    options = {"seed": args.seed, "batches": args.batches, "delay": trainer.config.delay}
    intact = evaluate(trainer.model, **options)
    lesions = {name: evaluate(trainer.model, **options, ablate=(name,)) for name in REGION_NAMES}
    memory = {str(delay): evaluate(trainer.model, **{**options, "delay": delay}) for delay in args.delays}
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    except ImportError:
        peak_bytes = None
    result = {"seed": args.seed, "trained_steps": trainer.step,
              "parameters": sum(p.numel() for p in trainer.model.parameters()),
              "intact": intact, "region_lesions": lesions, "delay_sweep": memory,
              "diagnostic_seconds": time.perf_counter()-start, "process_peak_rss_bytes": peak_bytes,
              "notes": ["Inference lesions are out-of-distribution interventions, not retrained controls.",
                        "Raw observation error also enters cerebellum. Lesioning a sensory region does not remove its modality.",
                        "Memory slots bound within-episode reads; working state can also retain a cue.",
                        "Peak RSS includes the Python/PyTorch runtime and loaded optimizer/replay, not just model tensors."]}
    atomic_json(args.output, result)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
