"""Read-only CPU diagnosis of learned English scaffold recognition and memory.

This uses fresh seeds from the familiar training support, not a held-out audit.
No pretrained model, tutor, optimizer step, or checkpoint selection is involved.
Run with --checkpoint PATH [--output REPORT.json].
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from brain_in_computer.dialogue_curriculum import ALIASES, COLORS, FOCI, generate_dialogues
from brain_in_computer.dialogue_scaffolding import ROLE_NAMES, UNKNOWN_COLOR, ScaffoldHeads, causal_targets
from brain_in_computer.dialogue_student import (
    ByteCodec, _run_turn, build_dialogue_student, checkpoint_digest, encode_dialogues,
)


def module_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        data = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf8"))
        digest.update(str((tuple(data.shape), str(data.dtype))).encode("ascii"))
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def accuracy(predicted, expected, mask=None):
    if mask is None:
        mask = torch.ones_like(expected, dtype=torch.bool)
    total = int(mask.sum())
    correct = int(((predicted == expected) & mask).sum())
    return {"accuracy": correct / total if total else None, "correct": correct, "total": total}


def category_metrics(predicted, expected, names):
    return {**accuracy(predicted, expected),
            "per_class": {name: accuracy(predicted, expected, expected == index)
                          for index, name in enumerate(names)}}


def inspect(checkpoint, *, seed=480_000, count_per_focus=64, batch_size=64):
    """Return metrics and unchanged hashes; all student inference is no-grad CPU."""
    if (type(seed) is not int or seed < 0
            or any(type(value) is not int or value < 1 for value in (count_per_focus, batch_size))):
        raise ValueError("seed must be nonnegative; counts and batch size must be positive")
    checkpoint_path = Path(checkpoint)
    checkpoint_bytes = checkpoint_path.read_bytes()
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    payload = saved.get("training", saved)
    if "state_dict" not in payload or "heads_state" not in payload:
        raise ValueError("checkpoint requires learned student state_dict and heads_state")
    include_state = "permitted.weight" in payload["heads_state"]
    # Initialization is discarded, and its RNG effects are contained.
    with torch.random.fork_rng(devices=[]):
        model = build_dialogue_student(0, device="cpu")
        heads = ScaffoldHeads(include_state=include_state)
    model.load_state_dict(payload["state_dict"], strict=True)
    heads.load_state_dict(payload["heads_state"], strict=True)
    model.eval()
    heads.eval()
    before = {"student": checkpoint_digest(model), "heads": module_digest(heads)}
    episodes = [episode for index, focus in enumerate(FOCI)
                for episode in generate_dialogues(seed + index * 10_000, count_per_focus, "train", focus)]
    targets = causal_targets(episodes, device="cpu")
    predictions = {name: [] for name in ("role", "alias", "color")}
    if include_state:
        predictions.update(permitted=[], bindings=[])
    with torch.inference_mode():
        for start in range(0, len(episodes), batch_size):
            sample = episodes[start:start + batch_size]
            batches = encode_dialogues(model, sample)
            collected = {name: [] for name in predictions}
            state = None
            for batch in batches:
                bos = torch.full((len(sample), 1), ByteCodec.BOS, dtype=torch.long)
                output, state = _run_turn(model, batch, state, decoder=bos)
                for name, logits in heads(output).items():
                    collected[name].append(logits.argmax(dim=-1))
            for name, values in collected.items():
                predictions[name].append(torch.stack(values))
    predictions = {name: torch.cat(values, dim=1) for name, values in predictions.items()}
    metrics = {
        "role": category_metrics(predictions["role"], targets["role"], ROLE_NAMES),
        "alias": category_metrics(predictions["alias"], targets["alias"], (*ALIASES, "none")),
        "color": category_metrics(predictions["color"], targets["color"], (*COLORS, "none")),
        "sentence_fields_by_role": {
            role: {name: accuracy(predictions[name], targets[name], targets["role"] == index)
                   for name in ("role", "alias", "color")}
            for index, role in enumerate(ROLE_NAMES)},
    }
    if include_state:
        for name in ("permitted", "bindings"):
            predicted, expected = predictions[name], targets[name]
            metrics[name] = {**category_metrics(predicted, expected, (*COLORS, "unknown")),
                             "known": accuracy(predicted, expected, expected != UNKNOWN_COLOR),
                             "unknown": accuracy(predicted, expected, expected == UNKNOWN_COLOR)}
        metrics["bindings"]["per_alias_known"] = {
            alias: accuracy(predictions["bindings"][:, :, index], targets["bindings"][:, :, index],
                            targets["bindings"][:, :, index] != UNKNOWN_COLOR)
            for index, alias in enumerate(ALIASES)}
    after = {"student": checkpoint_digest(model), "heads": module_digest(heads)}
    if after != before:
        raise RuntimeError("read-only scaffold inspection changed parameters")
    return {
        "schema": "bic-scaffold-inspection-v1", "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_file_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "updates": saved.get("updates"), "device": "cpu", "include_state": include_state,
        "support": {"split": "train", "fresh_seed": seed, "seed_stride_per_focus": 10_000,
                    "count_per_focus": count_per_focus, "episodes": len(episodes),
                    "transcripts_sha256": hashlib.sha256(json.dumps(episodes, sort_keys=True,
                        separators=(",", ":")).encode("utf8")).hexdigest()},
        "metrics": metrics, "parameter_hashes_before": before, "parameter_hashes_after": after,
        "parameters_unchanged": True, "teacher_used_for_policy": False,
        "limits": "Familiar training-support diagnostic; auxiliary label recognition is not action correctness, "
                  "unfamiliar-language transfer, or a pristine audit.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=480_000)
    parser.add_argument("--count-per-focus", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = inspect(args.checkpoint, seed=args.seed, count_per_focus=args.count_per_focus,
                     batch_size=args.batch_size)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf8")
        temporary.replace(path)
    print(text)


if __name__ == "__main__":
    main()
