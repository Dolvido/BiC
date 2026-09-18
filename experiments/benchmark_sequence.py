"""Isolated local optimizer throughput and CUDA causal-prefix sanity check."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from brain_in_computer.language import ByteCodec
from experiments.cognitive_credit import CreditTrainer
from experiments.sequence_data import pack_observations
from experiments.sequence_training import SequenceTrainer
from experiments.train_cognitive_credit import TRAIN_FAMILIES, make_banks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    banks = make_banks(seed=31_000_000, count=32)
    results = []
    for name, constructor in (("regional", CreditTrainer), ("sequence", SequenceTrainer)):
        trainer = constructor(banks, seed=2401, device="cuda", batch_size=64, learning_rate=.001)
        for index in range(3):
            trainer.step(TRAIN_FAMILIES[index % 3])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        tokens = 0
        for index in range(12):
            metrics = trainer.step(TRAIN_FAMILIES[index % 3])
            tokens += metrics.get("observation_tokens", 0)
        torch.cuda.synchronize()
        seconds = time.monotonic() - begin
        result = {"model": name, "parameters": sum(parameter.numel() for parameter in trainer.model.parameters()),
            "updates": 12, "seconds": seconds, "updates_per_second": 12 / seconds,
            "episode_exposures_per_second": 768 / seconds,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "observation_tokens": tokens if tokens else None}
        if name == "sequence":
            model = trainer.model.eval()
            with torch.inference_mode():
                first = pack_observations([["A fact.", "A future observation."], ["Another fact.", "Something later."]], device="cuda")
                changed = pack_observations([["A fact.", "Entirely changed future."], ["Another fact.", "Unrelated words."]], device="cuda")
                decoder = torch.full((2, 2, 1), ByteCodec.BOS, dtype=torch.long, device="cuda")
                a = model(**first, decoder_input_ids=decoder)
                b = model(**changed, decoder_input_ids=decoder)
                difference = float((a["logits"][:, 0] - b["logits"][:, 0]).abs().max())
                torch.testing.assert_close(a["logits"][:, 0], b["logits"][:, 0], rtol=1e-5, atol=1e-5)
                result["cuda_future_turn_isolation"] = {"passed": True, "max_logit_difference": difference,
                    "relative_tolerance": 1e-5, "absolute_tolerance": 1e-5}
        results.append(result)
        print(json.dumps(result), flush=True)
        del trainer
        if name == "sequence":
            del model
        torch.cuda.empty_cache()
    root = Path(__file__).resolve().parents[1]
    paths = ("experiments/sequence_student.py", "experiments/sequence_data.py", "experiments/sequence_training.py", "experiments/benchmark_sequence.py")
    report = {"source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths},
        "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "batch_size": 64,
        "warmup_updates_per_model": 3, "training_learning_rate": .001, "not_capability_evidence": True,
        "measurement_scope": "12 FP32 optimizer steps per model, sequential jobs, three canonical train families, no evaluation or checkpoint timing",
        "measurements": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
