"""Evaluate validated tutor sample data without a tutor or optimizer.

This is an integration sample diagnostic, not held-out generalization evidence.
All paths are explicit. Existing reports and release directories are protected.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
import time
from unittest.mock import patch

from experiments.local_tutor_validation import (
    GRAMMAR, RELATIONS, canonical_json, strict_json, validate_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("valid", "unknown_label", "absent", "ambiguous", "boundary")


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def network_disabled():
    """Reject Python socket connections and datagrams throughout evaluation."""
    def deny(*args, **kwargs):
        raise RuntimeError("Network calls are disabled during offline tutor evaluation")
    with patch.object(socket.socket, "connect", deny), \
         patch.object(socket.socket, "connect_ex", deny), \
         patch.object(socket.socket, "sendto", deny), \
         patch.object(socket, "create_connection", deny), \
         patch.object(socket, "getaddrinfo", deny):
        yield


def validated_lessons(protocol_path, accepted_path):
    # Importing this helper does not call its optional Ollama client.
    from experiments.local_tutor_pilot import read_protocol

    protocol = read_protocol(protocol_path)
    specs = {spec["lesson_id"]: spec for spec in protocol["specs"]}
    records, seen = [], set()
    with Path(accepted_path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line_number > protocol["count"] or len(line.encode("utf-8")) > 32768:
                raise ValueError("Accepted file exceeds the frozen protocol budget")
            record = strict_json(line)
            if not isinstance(record, dict) or record.get("accepted") is not True:
                raise ValueError(f"Line {line_number}: not an accepted lesson record")
            identifier = record.get("lesson_id")
            if not isinstance(identifier, str) or identifier not in specs or identifier in seen:
                raise ValueError(f"Line {line_number}: unknown or duplicate lesson ID")
            seen.add(identifier)
            candidate = {key: record.get(key) for key in ("lesson_id", "instruction", "relation")}
            decision = validate_candidate(candidate, specs[identifier])
            if decision.get("accepted") is not True:
                raise ValueError(f"Line {line_number}: revalidation failed: {decision.get('reason')}")
            if set(record) != set(decision) | {"spec", "envelope_sha256"}:
                raise ValueError(f"Line {line_number}: unexpected accepted-record fields")
            if any(record[key] != value for key, value in decision.items()):
                raise ValueError(f"Line {line_number}: validated decision or hash mismatch")
            expected = dict(specs[identifier], prompt=decision["instruction"],
                            **{key: decision[key] for key in ("target", "reply", "category")})
            # Canonical comparison rejects bool/int and other permissive Python equalities.
            if canonical_json(record["spec"]) != canonical_json(expected):
                raise ValueError(f"Line {line_number}: simulator evidence or targets changed")
            if not isinstance(record["envelope_sha256"], str) or not re.fullmatch(
                    r"[0-9a-f]{64}", record["envelope_sha256"]):
                raise ValueError(f"Line {line_number}: malformed envelope hash")
            records.append(record)
    return protocol, records


def metrics(rows):
    count = len(rows)
    action = sum(row["action_correct"] for row in rows)
    reply = sum(row["reply_exact"] for row in rows)
    joint = sum(row["joint_correct"] for row in rows)
    return {
        "episodes": count,
        "action_correct": action, "action_accuracy": action / count if count else None,
        "reply_exact_count": reply, "reply_exact_accuracy": reply / count if count else None,
        "joint_correct": joint, "joint_accuracy": joint / count if count else None,
        "all_stop_accuracy": sum(row["target"] == 10 for row in rows) / count if count else None,
        "invalid_grid_actions": sum(4 <= row["action"] < 10 for row in rows),
    }


def summarize(rows):
    return {
        "overall": metrics(rows),
        "valid_selections": metrics([row for row in rows if row["category"] == "valid"]),
        "categories": {value: metrics([row for row in rows if row["category"] == value])
                       for value in CATEGORIES},
        "relations": {value: metrics([row for row in rows if row["relation"] == value])
                      for value in RELATIONS},
        "phrase_families": {value: metrics([row for row in rows if row["phrase_family"] == value])
                           for value in GRAMMAR},
    }


def output_location(output, inputs):
    path = Path(output).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError("--output must name a new .json file")
    if path in {Path(item).resolve() for item in inputs}:
        raise ValueError("Output cannot replace an input")
    if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / "runs"):
        raise ValueError("Outputs inside the project must be under runs/")
    # A new filename inside an old release would still alter release evidence.
    for checkpoint in (ROOT / "runs").rglob("*.pt"):
        if path.is_relative_to(checkpoint.parent):
            raise ValueError(f"Output is inside an existing checkpoint directory: {checkpoint.parent}")
    return path


def evaluate(*, protocol, accepted, checkpoint, output, device="cpu", threads=1, batch_size=16):
    if device not in ("cpu", "cuda"):
        raise ValueError("Choose cpu or cuda explicitly")
    if type(threads) is not int or threads < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("threads and batch-size must be positive integers")
    paths = {key: Path(value).resolve() for key, value in (
        ("protocol", protocol), ("accepted", accepted), ("checkpoint", checkpoint))}
    destination = output_location(output, paths.values())
    initial_hashes = {key: sha_file(path) for key, path in paths.items()}
    started = time.perf_counter()

    # No API/service query is needed, including to check whether Ollama is running.
    with network_disabled():
        import torch
        from brain_in_computer.regional_memory import load_regional_checkpoint
        from experiments.train_regional_memory import encode_specs, normalized_prompts

        torch.set_num_threads(threads)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu or a compatible Torch build")
        frozen_protocol, records = validated_lessons(paths["protocol"], paths["accepted"])
        agent = load_regional_checkpoint(paths["checkpoint"], device="cpu")
        source_weights = {key: tensor.detach().clone() for key, tensor in agent.state_dict().items()}
        specifications = [record["spec"] for record in records]
        tick = time.perf_counter()
        # Frozen visual encoding remains on CPU, independent of inference backend.
        evidence = encode_specs(specifications, agent.encoder.cpu().eval()) if records else None
        encoding_seconds = time.perf_counter() - tick
        agent = agent.to(device).eval()
        predictions = []
        tick = time.perf_counter()
        with torch.inference_mode():
            for offset in range(0, len(records), batch_size):
                batch = records[offset:offset + batch_size]
                specs = specifications[offset:offset + batch_size]
                scores = evidence["scores"][offset:offset + batch_size].to(device)
                known = evidence["known"][offset:offset + batch_size].to(device)
                prompts = normalized_prompts(specs)
                result = agent.forward_evidence(scores, known, prompts)
                actions = result["logits"][:, -1].argmax(-1).tolist()
                replies = agent.respond_evidence(scores, known, prompts)
                for index, (record, action, reply, normalized) in enumerate(
                        zip(batch, actions, replies, prompts)):
                    spec = record["spec"]
                    action_ok, reply_ok = action == spec["target"], reply == spec["reply"]
                    predictions.append({
                        "lesson_id": record["lesson_id"], "instruction": record["instruction"],
                        "normalized_instruction": normalized,
                        "phrase_family": record["phrase_family"], "relation": spec["relation"],
                        "category": spec["category"], "target": spec["target"], "action": action,
                        "expected_reply": spec["reply"], "reply": reply,
                        "action_correct": action_ok, "reply_exact": reply_ok,
                        "joint_correct": action_ok and reply_ok,
                        "cosine_scores": scores[index].tolist(), "known_name": bool(known[index].item()),
                    })
        if device == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - tick
        unchanged = all(torch.equal(value.detach().cpu(), source_weights[key])
                        for key, value in agent.state_dict().items())
        if not unchanged:
            raise RuntimeError("Model state changed during inference")
        final_hashes = {key: sha_file(path) for key, path in paths.items()}
        if final_hashes != initial_hashes:
            raise RuntimeError("Input files changed during evaluation")
        report = {
            "schema": "bic-tutor-sample-evaluation-v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "integration_sample_diagnostic",
            "held_out_generalization_claim": False, "learning_improvement_claim": False,
            "training_performed": False, "weights_unchanged": unchanged,
            "inference": {
                "device": device, "visual_encoding_device": "cpu", "threads": threads,
                "batch_size": batch_size, "torch_version": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "teacher_requests": 0, "teacher_required": False,
                "network_guard": "Python socket connections, DNS, and datagrams blocked",
                "teacher_process_state": "not queried; report does not assert Ollama service is stopped",
                "reply_metric": "Exact canonical string equality, without whitespace stripping",
                "encoding_seconds": encoding_seconds, "inference_seconds": inference_seconds,
            },
            "provenance": {
                "inputs": {key: {"path": str(path), "sha256": initial_hashes[key]}
                           for key, path in paths.items()},
                "source_sha256": {name: sha_file(ROOT / name) for name in (
                    "experiments/evaluate_tutor_sample.py", "experiments/local_tutor_validation.py",
                    "experiments/local_tutor_pilot.py", "experiments/train_regional_memory.py",
                    "brain_in_computer/regional_memory.py", "brain_in_computer/associative.py")},
                "protocol_source_hashes_verified": True,
                "accepted_rows_revalidated": len(records),
                "frozen_protocol_count": frozen_protocol["count"],
                "generation_envelope_hashes": {record["lesson_id"]: record["envelope_sha256"]
                                              for record in records},
                "envelope_hash_scope": "Retained from importer; raw envelopes are not inputs to this evaluator",
            },
            "coverage": {
                "accepted": len(records), "protocol_total": frozen_protocol["count"],
                "omitted_lesson_ids": [spec["lesson_id"] for spec in frozen_protocol["specs"]
                                       if spec["lesson_id"] not in {r["lesson_id"] for r in records}],
            },
            "metrics": summarize(predictions), "predictions": predictions,
            "limits": [
                "These are accepted generation fixtures, not a held-out benchmark.",
                "The tutor copied approved grammar; this does not measure novel lesson quality.",
                "No optimizer ran, so this cannot demonstrate learning from the tutor.",
                "Identity labels score simulator outcomes only; inference receives normalized language and sensory evidence.",
                "An always-STOP policy can score well when many sample cases require STOP; inspect valid selections.",
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation handles a file appearing after the initial check.
    with destination.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2, ensure_ascii=False, allow_nan=False)
        target.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        report = evaluate(**vars(args))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
    print(json.dumps({"output": str(args.output), "purpose": report["purpose"],
                      "coverage": report["coverage"], "metrics": report["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

