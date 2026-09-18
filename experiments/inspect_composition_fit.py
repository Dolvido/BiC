"""Post-study fitting diagnostic; never changes the sealed comparison or selects weights."""
import json
from pathlib import Path
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_loop import run_lock
from experiments import composition_study as study
from experiments.composition_evaluation import PreparedBank, evaluate_banks
from experiments.sequence_student import build_sequence_student
from experiments.train_cognitive import atomic_json


def inspect(directory):
    directory = Path(directory)
    protocol = study.load_protocol(directory)
    summary_path = directory / "audit" / "summary.json"
    if not summary_path.is_file():
        raise ValueError("completed verified study summary required")
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    evidence = summary["verification"]["input_file_sha256"]
    if any(study.file_hash(directory / name) != sha for name, sha in evidence.items()):
        raise ValueError("completed study evidence changed")
    if summary["verification"]["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("verified source provenance differs")
    summary_hash = study.file_hash(summary_path)
    banks = torch.load(directory / "banks.pt", map_location="cpu", weights_only=True)
    started = time.monotonic()
    prepared = {f"train/{family}/t{turns}": PreparedBank(rows, role="train_fit", config=study.CONFIG)
                for family, buckets in banks["train"].items() for turns, rows in buckets.items()}
    preparation_seconds = time.monotonic() - started
    results = {}
    for job in study.JOBS:
        saved = torch.load(directory / "main" / job / "latest.pt", map_location="cpu", weights_only=True)
        model = build_sequence_student(study.SEED, device="cuda", config=study.CONFIG)
        model.load_state_dict(saved["training"]["weights"], strict=True)
        digest = checkpoint_digest(model)
        if digest != summary["main"][job]["weights_sha256"]:
            raise ValueError("endpoint tensor digest differs")
        selected = prepared if not job.startswith("fresh-") else {
            name: bank for name, bank in prepared.items() if name.split("/")[1] == job.split("-", 1)[1]}
        results[job] = {"weights_sha256": digest, "metrics": evaluate_banks(model, selected)}
        if checkpoint_digest(model) != digest:
            raise RuntimeError("diagnostic changed weights")
        del model
        print(json.dumps({"fit_scored": job}), flush=True)
    study.load_protocol(directory)
    if study.file_hash(summary_path) != summary_hash or any(study.file_hash(directory / name) != sha for name, sha in evidence.items()):
        raise RuntimeError("study evidence changed during fitting diagnostic")
    return {
        "schema": "bic-composition-post-study-fit-v1",
        "scope": "Descriptive post-study training-bank fitting; not part of the prospective audit, not additional training, and not checkpoint selection.",
        "interpretation": "Training and held-out banks differ in contents and target mixture; fitting gaps do not uniquely identify a mechanism or establish generalization.",
        "source_sha256": protocol["source_sha256"], "script_sha256": study.file_hash(__file__),
        "verified_summary_sha256": summary_hash, "input_file_sha256": evidence,
        "training_updates": 0, "automatic_promotion": False, "device": "cuda",
        "preparation_seconds": preparation_seconds, "total_seconds": time.monotonic() - started,
        "results": results}


if __name__ == "__main__":
    torch.set_num_threads(1)
    directory = Path("runs/composition-study-local")
    output = directory / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        atomic_json(output / "train-fit.json", inspect(directory))
