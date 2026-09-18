"""Read-only train-support diagnosis of neural information and credit pathways.

One balanced query-only backward pass per family, without optimizer steps. This
measures associations and gradient access, not causal sufficiency or competence.
It neither reads nor constructs the prospective audit/support banks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from brain_in_computer.dialogue_student import (
    ACK, ByteCodec, _run_turn, checkpoint_digest, encode_dialogues,
)
from experiments.cognitive_curriculum import generate_cognitive, validate_cognitive
from experiments.cognitive_student import build_cognitive_student
from experiments.train_cognitive import TRAIN_FAMILIES, fingerprint_rows, source_hashes


SEED = 12_000_000
COUNT_PER_LEVEL = 32
STAGES = ("encoder_final", "retrieval_projection", "hippocampus", "prefrontal", "motor")


def _mean(values):
    return sum(values) / len(values) if values else None


def _separation(left, right):
    left, right = left.detach().float(), right.detach().float()
    distance = float(torch.linalg.vector_norm(left - right))
    norm = float((torch.linalg.vector_norm(left) + torch.linalg.vector_norm(right)) / 2)
    cosine = float(1 - F.cosine_similarity(left[None], right[None], eps=1e-12)[0])
    if not bool(left.any()) and not bool(right.any()):
        cosine = 0.
    return {"l2": distance, "relative_l2": distance / max(norm, 1e-12),
            "cosine_distance": cosine, "mean_feature_norm": norm}


def _summarize_separations(rows):
    return {"comparisons": len(rows), **{key: _mean([row[key] for row in rows])
            for key in ("l2", "relative_l2", "cosine_distance", "mean_feature_norm")}}


def _parameter_gradients(model):
    groups = {"encoder": [], "retrieval": [], "semantic_bridge": [],
              "observation_predictor": [], "reply_decoder": []}
    groups.update({name: [] for name in model.brain.region_names})
    for name, parameter in model.named_parameters():
        if name.startswith("posterior_temporal."):
            group = "encoder"
        elif name.startswith("retrieval_"):
            group = "retrieval"
        elif name.startswith("semantic_bridge."):
            group = "semantic_bridge"
        elif name.startswith("observation_predictor."):
            group = "observation_predictor"
        elif name.startswith("inferior_frontal."):
            group = "reply_decoder"
        else:
            group = name.split(".")[2]
        groups[group].append(parameter)
    return {name: {"parameter_count": sum(p.numel() for p in parameters),
        "with_gradient_count": sum(p.numel() for p in parameters if p.grad is not None),
        "gradient_l2": sum(float(p.grad.detach().double().square().sum())
                           for p in parameters if p.grad is not None) ** .5}
        for name, parameters in groups.items()}


def diagnose_family(model, episodes):
    """Measure one canonical train bank, restoring weights, modes and gradients."""
    if not episodes or any(row.get("split") != "train" for row in episodes):
        raise ValueError("diagnosis accepts fresh canonical train rows only")
    for row in episodes:
        validate_cognitive(row)
    if len(episodes) % 2:
        raise ValueError("diagnosis requires complete adjacent pairs")
    for index in range(0, len(episodes), 2):
        if episodes[index]["counterfactual_group"] != episodes[index + 1]["counterfactual_group"]:
            raise ValueError("diagnosis requires complete adjacent pairs")
    before = checkpoint_digest(model)
    modes = [(module, module.training) for module in model.modules()]
    previous_gradients = [(parameter, parameter.grad) for parameter in model.parameters()]
    captured = {name: [] for name in (*STAGES, "encoder_tokens")}
    handles = []

    def remember(name, tensor):
        if tensor.requires_grad:
            tensor.retain_grad()
        captured[name].append(tensor)

    def encoder_hook(module, inputs, output):
        remember("encoder_tokens", output[0])
        remember("encoder_final", output[1])

    handles.append(model.posterior_temporal.register_forward_hook(encoder_hook))
    handles.append(model.retrieval_projection.register_forward_hook(
        lambda module, inputs, output: remember("retrieval_projection", output)))
    for stage, region in (("hippocampus", "hippocampus"), ("prefrontal", "prefrontal_cortex"),
                          ("motor", "motor_cortex")):
        handles.append(model.brain.regions[region].register_forward_hook(
            lambda module, inputs, output, name=stage:
                remember(name, output[0] if isinstance(output, tuple) else output)))
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            state, logits, targets = None, [], []
            batches = encode_dialogues(model, episodes)
            for batch in batches:
                decoder = torch.full((len(episodes), 1), ByteCodec.BOS, dtype=torch.long,
                                     device=batch["text_ids"].device)
                output, state = _run_turn(model, batch, state, decoder=decoder)
                logits.append(output["logits"][:, -1])
                targets.append(batch["targets"])
            all_logits, labels = torch.cat(logits), torch.cat(targets)
            terms = [F.cross_entropy(all_logits[labels == target], labels[labels == target])
                     for target in range(ACK) if bool((labels == target).any())]
            if not terms:
                raise ValueError("diagnostic bank has no queries")
            loss = torch.stack(terms).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite diagnostic query loss")
            loss.backward()
        query_comparisons = {name: [] for name in (*STAGES, "logits")}
        premise_comparisons = []
        query_gradient_rows = {name: [] for name in (*STAGES, "encoder_tokens")}
        premise_gradients = {name: [] for name in ("encoder_final", "encoder_tokens")}
        predictions = all_logits.detach().argmax(-1).reshape(6, len(episodes))
        paired_correct = 0
        for left in range(0, len(episodes), 2):
            right = left + 1
            for turn, (a, b) in enumerate(zip(episodes[left]["turns"], episodes[right]["turns"])):
                if a["text"] != b["text"]:
                    feature = captured["encoder_final"][turn]
                    premise_comparisons.append(_separation(feature[left], feature[right]))
                    for name in premise_gradients:
                        gradient = captured[name][turn].grad
                        for row in (left, right):
                            premise_gradients[name].append(0. if gradient is None else
                                float(torch.linalg.vector_norm(gradient[row].detach().float())))
                if a["target"] == ACK or b["target"] == ACK or a["target"] == b["target"]:
                    continue
                if a["text"] != b["text"]:
                    raise ValueError("opposite-answer queries must share their current text")
                for name in query_comparisons:
                    feature = logits[turn] if name == "logits" else captured[name][turn]
                    # Memory context is tanh(projected query); gradients below
                    # are for its preactivation, the hooked module output.
                    if name == "retrieval_projection":
                        feature = feature.tanh()
                    query_comparisons[name].append(_separation(feature[left], feature[right]))
                for name in query_gradient_rows:
                    gradient = captured[name][turn].grad
                    for row in (left, right):
                        query_gradient_rows[name].append(0. if gradient is None else
                            float(torch.linalg.vector_norm(gradient[row].detach().float())))
                paired_correct += int(predictions[turn, left] == a["target"]
                                      and predictions[turn, right] == b["target"])
        total = len(query_comparisons["logits"])
        query = labels.ne(ACK)
        result = {"episodes": len(episodes), "bank_sha256": fingerprint_rows(episodes),
            "query_only_loss": float(loss.detach()), "query_count": int(query.sum()),
            "query_accuracy": float(all_logits.detach().argmax(-1)[query].eq(labels[query]).float().mean()),
            "opposite_query_pairs": total, "both_answers_correct": paired_correct,
            "pair_accuracy": paired_correct / total if total else None,
            "changed_premise_encoder_separation": _summarize_separations(premise_comparisons),
            "opposite_query_stage_separation": {name: _summarize_separations(rows)
                                                for name, rows in query_comparisons.items()},
            "query_stage_gradient_mean_row_l2": {name: _mean(values) for name, values in query_gradient_rows.items()},
            "changed_premise_encoder_gradient_mean_row_l2": {name: _mean(values)
                                                              for name, values in premise_gradients.items()},
            "query_only_parameter_gradients": _parameter_gradients(model),
            "checkpoint_sha256_before": before, "checkpoint_sha256_after": checkpoint_digest(model)}
        if result["checkpoint_sha256_after"] != before:
            raise RuntimeError("diagnosis changed weights")
        return result
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training
        for parameter, gradient in previous_gradients:
            parameter.grad = gradient


def diagnose_model(model, *, seed=SEED, count_per_level=COUNT_PER_LEVEL):
    return {family: diagnose_family(model, [row for level in (1, 2)
        for row in generate_cognitive(seed + index * 100_000 + level * 10_000,
            count_per_level, split="train", family=family, level=level)])
        for index, family in enumerate(TRAIN_FAMILIES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--count-per-level", type=int, default=COUNT_PER_LEVEL)
    args = parser.parse_args()
    if args.seed < 0 or args.seed % 2 or args.count_per_level < 2 or args.count_per_level % 2:
        parser.error("seed and positive count must be even")
    if args.output.exists():
        parser.error("output exists; use a new diagnostic output path")
    torch.set_num_threads(1)
    report = {"schema": "bic-pathway-diagnostic-v1", "seed": args.seed,
        "count_per_level": args.count_per_level, "levels": [1, 2], "split": "train",
        "device": "cpu", "optimizer_steps": 0, "heldout_banks_accessed": False,
        "source_sha256": {**source_hashes(), "experiments/diagnose_cognitive_pathway.py":
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "notes": ["Associations and gradient access do not establish causal sufficiency.",
            "Same-current-query encoder separation should be numerical zero by design.",
            "Query stage gradients include later queries through recurrent state.",
            "Retrieval separation uses tanh context; its gradient uses projection preactivation.",
            "Gradient scales depend on representation scale and query-loss normalization."], "runs": {}}
    for directory in args.runs:
        completed = json.loads((directory / "report.json").read_text(encoding="utf8"))
        if completed["stop_reason"] != "planned_updates" or completed["updates"] != completed["protocol"]["steps"]:
            raise ValueError("diagnosis requires completed training runs")
        if completed["protocol"]["source_sha256"] != source_hashes():
            raise ValueError("training source mismatch")
        checkpoint = directory / "latest.pt"
        file_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["protocol"] != completed["protocol"] or saved["training"]["updates"] != completed["updates"]:
            raise ValueError("checkpoint protocol/update mismatch")
        model = build_cognitive_student(completed["protocol"]["seed"],
                                       memory_mode=completed["protocol"]["memory_mode"])
        model.load_state_dict(saved["training"]["weights"], strict=True)
        if checkpoint_digest(model) != completed["checkpoint_sha256"]:
            raise ValueError("final checkpoint weight mismatch")
        results = diagnose_model(model, seed=args.seed, count_per_level=args.count_per_level)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != file_digest:
            raise RuntimeError("diagnosis changed saved checkpoint")
        report["runs"][str(directory.resolve())] = {"checkpoint_file_sha256": file_digest,
            "checkpoint_sha256": checkpoint_digest(model), "families": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
