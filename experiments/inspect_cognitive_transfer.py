"""Post-hoc query-position breakdown of frozen audit endpoints; no training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from brain_in_computer.dialogue_student import ByteCodec, _run_turn, _generate_reply_tokens, _exact_replies, checkpoint_digest, encode_dialogues
from experiments.cognitive_student import build_cognitive_student
from experiments.audit_cognitive_transfer import file_hash, load_prepared


def inspect(directory):
    directory = Path(directory)
    manifest, banks = load_prepared(directory)
    torch.set_num_threads(1)
    report = {"post_hoc": True, "purpose": "Disaggregate already-inspected final endpoints, not select checkpoints or change the benchmark.",
        "audit_manifest_sha256": file_hash(directory / "manifest.json"),
        "inspection_source_sha256": file_hash(__file__), "candidates": {}}
    completed = json.loads((directory / "report.json").read_text(encoding="utf8"))
    for name, record in completed["results"].items():
        path = directory / f"{name}-adapted.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = build_cognitive_student(manifest["initial_seed"], memory_mode=record["memory_mode"])
        model.load_state_dict(saved["weights"])
        model.eval()
        before = checkpoint_digest(model)
        if before != record["final_weight_sha256"]:
            raise ValueError("adapted endpoint digest mismatch")
        result = {}
        with torch.inference_mode():
            for level, episodes in banks["query"].items():
                state, rows = None, []
                for turn, batch in enumerate(encode_dialogues(model, episodes)):
                    decoder = torch.full((len(episodes), 1), ByteCodec.BOS, dtype=torch.long)
                    output, state = _run_turn(model, batch, state, decoder=decoder)
                    target = batch["targets"]
                    if not bool(target.ne(3).any()):
                        continue
                    pred = output["logits"][:, -1].argmax(-1)
                    generated = _generate_reply_tokens(model, output["production_context"])
                    reply_correct = _exact_replies(generated, batch["reply_targets"])
                    masks = {"all": target.ne(3), "known": target.lt(2), "unknown": target.eq(2)}
                    row = {"turn_one_based": turn + 1, "slices": {}}
                    for key, mask in masks.items():
                        total = int(mask.sum())
                        correct = int((mask & pred.eq(target)).sum())
                        row["slices"][key] = {"correct": correct, "total": total,
                            "accuracy": correct / total if total else None,
                            "reply_exact_correct": int((mask & reply_correct).sum()),
                            "reply_exact_accuracy": float(reply_correct[mask].float().mean()) if total else None,
                            "predicted_action_counts": torch.bincount(pred[mask], minlength=4).tolist()}
                    eligible = target[0::2].lt(2) & target[1::2].lt(2) & target[0::2].ne(target[1::2])
                    correct = eligible & pred[0::2].eq(target[0::2]) & pred[1::2].eq(target[1::2])
                    row["pairs"] = {"correct": int(correct.sum()), "total": int(eligible.sum())}
                    rows.append(row)
                result[level] = rows
        if before != checkpoint_digest(model):
            raise RuntimeError("inspection changed weights")
        report["candidates"][name] = {"checkpoint_sha256": before, "weights_unchanged": True, "levels": result}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.audit)
    path = args.audit / "query-position-breakdown.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({"output": str(path), "candidates": len(report["candidates"]), "weights_unchanged": True}))


if __name__ == "__main__":
    main()
