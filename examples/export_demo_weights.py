"""Export only native model weights from a trusted shared-rate checkpoint.

This utility is for maintainers with the original research archive. The public
demo already includes its exported asset and does not need that archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments.sequence_student import SequenceConfig
from experiments.shared_state_student import ARCHITECTURE, build_shared_state_student


SCHEMA = "bic-public-inference-v1"
SOURCE_ARCHIVE_SHA256 = "ad39a15737834e20203209261cd58dc9f70773a849914fc463ab705bcaf70637"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_weights(checkpoint: Path, output: Path) -> dict:
    source_digest = sha256_file(checkpoint)
    if source_digest != SOURCE_ARCHIVE_SHA256:
        raise ValueError("this release exporter requires the recorded lower-2592 research archive")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    learner = payload["learner"]
    recipe = learner["recipe"]
    if recipe["architecture"] != ARCHITECTURE:
        raise ValueError("expected the shared entity-state research architecture")
    config = SequenceConfig(**recipe["config"])
    model = build_shared_state_student(seed=0, config=config).eval()
    model.load_state_dict(learner["weights"], strict=True)
    digest = checkpoint_digest(model)
    if digest != payload["weights_sha256"]:
        raise ValueError("checkpoint's declared native weights digest differs")
    provenance = {
        "source_archive_name": checkpoint.name,
        "source_archive_sha256": source_digest,
        "study": "shared-rate-local/attempt-001",
        "branch": "lower",
        "lifetime_updates": payload["lifetime_updates"],
        "study_updates": payload["rate_updates"],
        "status": "Research candidate; failed the complete promotion gate.",
    }
    artifact = {
        "schema": SCHEMA,
        "architecture": ARCHITECTURE,
        "config": recipe["config"],
        "weights": model.state_dict(),
        "weights_sha256": digest,
        "provenance": provenance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    manifest = {
        "schema": SCHEMA,
        "asset": output.name,
        "asset_sha256": sha256_file(output),
        "asset_bytes": output.stat().st_size,
        "weights_sha256": digest,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "architecture": ARCHITECTURE,
        "config": recipe["config"],
        "provenance": provenance,
        "export_contents": "Model tensors and inference configuration only; no optimizer, tutor model, lessons, or session state.",
        "export_runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_weights(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
