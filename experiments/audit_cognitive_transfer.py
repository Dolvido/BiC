"""Freeze and audit one unseen-family adaptation study after all four runs finish.

Preparation constructs and hashes the banks without evaluating any learner.
The audit rejects incomplete/mismatched runs and never loads their optimizer
moments into adaptation. Its fixed endpoints are not checkpoint selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from brain_in_computer.dialogue_student import (
    ACK, ByteCodec, _action_metrics, _exact_replies, _generate_reply_tokens,
    _run_turn, checkpoint_digest, encode_dialogues,
)
from brain_in_computer.learning_loop import run_lock
from brain_in_computer.learning_student import _check_finite_tree, _cpu_copy
from experiments.cognitive_curriculum import generate_cognitive
from experiments.cognitive_student import build_cognitive_student
from experiments.train_cognitive import (
    CognitiveTrainer, HELDOUT_FAMILY, TRAIN_FAMILIES, atomic_checkpoint,
    atomic_json, fingerprint_rows, make_banks, source_hashes,
)


SCHEMA = "bic-cognitive-family-transfer-v1"
BUDGETS = (0, 1, 4, 16)
ADAPT_SEED = 3101
INITIAL_SEED = 2201


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_source_hashes():
    result = source_hashes()
    root = Path(__file__).resolve().parents[1]
    result[Path(__file__).relative_to(root).as_posix()] = file_hash(__file__)
    return result


def transcript(row):
    return tuple(turn["text"] for turn in row["turns"])


def exclude_overlapping_pairs(rows, excluded):
    """Remove both pair members by transcript membership, independent of scores."""
    if len(rows) % 2:
        raise ValueError("an evaluation bank must contain complete pairs")
    result, removed = [], []
    for index in range(0, len(rows), 2):
        pair = rows[index:index + 2]
        if pair[0]["counterfactual_group"] != pair[1]["counterfactual_group"]:
            raise ValueError("evaluation pairs are not adjacent")
        if any(transcript(row) in excluded for row in pair):
            removed.extend(row["id"] for row in pair)
        else:
            result.extend(pair)
    if not result:
        raise ValueError("overlap exclusion left no evaluation examples")
    return result, removed


def prepare_banks():
    support = make_banks(seed=7_000_000, count=32, families=(HELDOUT_FAMILY,))
    excluded = {transcript(row) for row in support[HELDOUT_FAMILY]}
    queries, exclusions = {}, {}
    for level in (2, 3):
        rows = generate_cognitive(8_000_000 + level * 10_000, 256,
            split="audit", family=HELDOUT_FAMILY, level=level)
        queries[f"level_{level}"], exclusions[f"level_{level}"] = exclude_overlapping_pairs(rows, excluded)
    retained = make_banks(seed=9_000_000, count=64, split="dev")
    transfer = make_banks(seed=6_000_000, count=256, split="audit", levels=(3,))
    return {"support": support, "query": queries, "retained": retained,
            "trained_family_transfer": transfer}, exclusions


def bank_manifest(banks):
    return {group: {name: {"sha256": fingerprint_rows(rows), "episodes": len(rows),
        "distinct_transcripts": len({transcript(row) for row in rows}),
        "counterfactual_groups": len({row["counterfactual_group"] for row in rows}),
        "turn_target_counts": {str(target): sum(turn["target"] == target
            for row in rows for turn in row["turns"]) for target in range(4)}}
        for name, rows in values.items()} for group, values in banks.items()}


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        banks, exclusions = prepare_banks()
        contract = {"schema": SCHEMA, "source_sha256": audit_source_hashes(),
            "heldout_family": HELDOUT_FAMILY, "training_families": list(TRAIN_FAMILIES),
            "initial_seed": INITIAL_SEED, "adaptation_seed": ADAPT_SEED,
            "adaptation_updates": list(BUDGETS), "batch_size": 64, "learning_rate": .003,
            "support_seed": 7_000_000, "query_seed": 8_000_000, "retention_seed": 9_000_000,
            "support_levels": [1, 2], "query_levels": [2, 3], "query_split": "audit",
            "trained_family_transfer_seed": 6_000_000,
            "trained_family_transfer_level": 3, "trained_family_transfer_split": "audit",
            "planned_main_updates": 1800, "required_arms": [[mode, order]
                for mode in ("episodic", "recurrent") for order in ("interleaved", "blocked")],
            "banks": bank_manifest(banks), "whole_pair_exclusions": exclusions,
            "interpretation": "One previously untrained family, one support draw, shared vocabulary. "
                "Primary ordered entity-pair supports are disjoint; incidental pairs can overlap. "
                "Not general intelligence or a learned meta-optimizer.",
            "evaluation_rule": "All four complete runs before first heldout evaluation; fixed endpoints; "
                "AdamW reset; same support/sampler; no heldout feedback during general training."}
        path = directory / "manifest.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf8"))
            previous = {key: value for key, value in saved.items() if key != "prepared_utc"}
            if previous != contract:
                raise ValueError("prepared source or bank contract changed; use a new study directory")
            load_prepared(directory)
            return saved
        contract["prepared_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(directory / "banks.json", banks)
        atomic_json(path, contract)
        return contract


def load_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    banks = json.loads((directory / "banks.json").read_text(encoding="utf8"))
    if manifest.get("schema") != SCHEMA or manifest["source_sha256"] != audit_source_hashes():
        raise ValueError("prepared audit source mismatch")
    if manifest["banks"] != bank_manifest(banks):
        raise ValueError("prepared audit bank mismatch")
    return manifest, banks


def verify_runs(paths, manifest):
    """Validate completion and provenance before allowing any heldout scoring."""
    if len(paths) != 4:
        raise ValueError("exactly four completed runs are required")
    prepared = datetime.fromisoformat(manifest["prepared_utc"]).timestamp()
    seen, records, comparison = set(), [], None
    initial_digest = None
    root = Path(__file__).resolve().parents[1]
    for value in paths:
        path = Path(value).resolve()
        protocol = json.loads((path / "protocol.json").read_text(encoding="utf8"))
        report = json.loads((path / "report.json").read_text(encoding="utf8"))
        checkpoint_path = path / "latest.pt"
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        cell = (protocol["memory_mode"], protocol["schedule"])
        if cell in seen or list(cell) not in manifest["required_arms"]:
            raise ValueError("runs must cover each memory-by-order arm once")
        seen.add(cell)
        if (protocol["seed"] != manifest["initial_seed"]
                or protocol["steps"] != manifest["planned_main_updates"]
                or report["updates"] != protocol["steps"]
                or saved["training"]["updates"] != protocol["steps"]
                or report["stop_reason"] != "planned_updates"
                or report["heldout_evaluation_performed"]
                or saved["protocol"] != protocol or report["protocol"] != protocol):
            raise ValueError("run is incomplete or its study protocol disagrees")
        expected_counts = {family: protocol["steps"] // len(TRAIN_FAMILIES) for family in TRAIN_FAMILIES}
        if (saved["training"]["family_updates"] != expected_counts
                or report["family_updates"] != expected_counts):
            raise ValueError("run did not complete equal family update budgets")
        _check_finite_tree(saved["training"]["weights"], "audited_weights")
        if protocol["source_sha256"] != source_hashes():
            raise ValueError("training source changed since the run")
        if (tuple(protocol["training_families"]) != TRAIN_FAMILIES
                or protocol["heldout_family"] != HELDOUT_FAMILY
                or protocol["train_levels"] != [1, 2]):
            raise ValueError("heldout-family or difficulty leakage in training protocol")
        if (path / "initial.pt").stat().st_mtime < prepared:
            raise ValueError("heldout manifest must precede initial training checkpoint")
        for name, digest in protocol["source_sha256"].items():
            if file_hash(path / "source" / name) != digest or file_hash(root / name) != digest:
                raise ValueError("frozen training source snapshot mismatch")
        match = {key: val for key, val in protocol.items() if key not in ("memory_mode", "schedule")}
        if comparison is not None and match != comparison:
            raise ValueError("four arms have different training budgets or other settings")
        comparison = match
        model = build_cognitive_student(protocol["seed"], memory_mode=cell[0])
        initial = torch.load(path / "initial.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(initial["weights"])
        digest = checkpoint_digest(model)
        expected = checkpoint_digest(build_cognitive_student(protocol["seed"], memory_mode=cell[0]))
        if digest != expected or (initial_digest is not None and digest != initial_digest):
            raise ValueError("arms do not share their declared fresh initialization")
        initial_digest = digest
        model.load_state_dict(saved["training"]["weights"])
        if checkpoint_digest(model) != report["checkpoint_sha256"]:
            raise ValueError("reported final weights disagree with checkpoint")
        datasets = json.loads((path / "datasets.json").read_text(encoding="utf8"))
        train_banks = make_banks(seed=protocol["train_bank_seed"],
                                count=protocol["train_count_per_family_level"])
        if datasets["training"] != {key: fingerprint_rows(rows) for key, rows in train_banks.items()}:
            raise ValueError("training bank fingerprint mismatch")
        records.append({"key": "-".join(cell), "path": str(path),
            "checkpoint_sha256": file_hash(checkpoint_path), "memory_mode": cell[0],
            "weights": saved["training"]["weights"], "training_banks": train_banks})
    return records


def evaluate_bank(model, rows, *, replies=False, reset_each_turn=False, blank_text=False):
    """Single pass with uncertainty denominators, Brier score, and paired answers."""
    before = checkpoint_digest(model)
    modes = [(module, module.training) for module in model.modules()]
    logits, labels, generated_correct, reply_actions = [], [], [], []
    try:
        model.eval()
        with torch.inference_mode():
            state = None
            for batch in encode_dialogues(model, rows, blank_text=blank_text):
                decoder = torch.full((len(rows), 1), ByteCodec.BOS, dtype=torch.long,
                                     device=batch["text_ids"].device)
                output, state = _run_turn(model, batch, None if reset_each_turn else state, decoder=decoder)
                logits.append(output["logits"][:, -1])
                labels.append(batch["targets"])
                if replies:
                    generated = _generate_reply_tokens(model, output["production_context"])
                    generated_correct.append(_exact_replies(generated, batch["reply_targets"]))
                    from experiments.cognitive_curriculum import REPLIES
                    actions = []
                    for tokens in generated:
                        try:
                            reply = model.codec.decode(tokens.tolist())
                        except (ValueError, UnicodeError):
                            reply = None
                        actions.append(REPLIES.index(reply) if reply in REPLIES else -1)
                    reply_actions.append(torch.tensor(actions, device=decoder.device))
            logits, labels = torch.cat(logits), torch.cat(labels)
            predictions, query = logits.argmax(-1), labels.ne(ACK)
            qlogits, qlabels, qpred = logits[query], labels[query], predictions[query]
            result = _action_metrics(qlogits, qlabels)
            result.update(query_accuracy=result["accuracy"], query_total=result["total"],
                          query_correct=result["correct"], query_loss=result["loss"])
            confusion = torch.bincount(qlabels * 4 + qpred, minlength=16).reshape(4, 4)
            ask_true, ask_pred, ask_correct = int(confusion[2].sum()), int(confusion[:, 2].sum()), int(confusion[2, 2])
            per_target = {str(target): _action_metrics(qlogits[qlabels == target], qlabels[qlabels == target])
                          for target in range(3)}
            allow_deny = [per_target[str(target)]["accuracy"] for target in (0, 1)]
            probabilities = qlogits.softmax(-1)
            result.update(confusion_matrix_true_rows_predicted_columns=confusion.cpu().tolist(),
                per_target=per_target, ask_true=ask_true, ask_predicted=ask_pred,
                ask_correct=ask_correct, ask_recall=ask_correct / ask_true if ask_true else None,
                ask_precision=ask_correct / ask_pred if ask_pred else None,
                allow_deny_macro_accuracy=sum(allow_deny) / 2 if None not in allow_deny else None,
                brier_score=float((probabilities - F.one_hot(qlabels, 4)).square().sum(-1).mean()),
                blank_text=blank_text, reset_each_turn=reset_each_turn,
                query_reply_exact_accuracy=None, action_reply_agreement=None,
                reply_parseable_query_count=None)
            pred_matrix, target_matrix = predictions.reshape(6, len(rows)), labels.reshape(6, len(rows))
            groups = {}
            for index, row in enumerate(rows):
                groups.setdefault(row["counterfactual_group"], []).append(index)
            total = correct = 0
            for members in groups.values():
                if len(members) != 2:
                    raise ValueError("evaluation requires exactly two rows in every counterfactual group")
                left, right = members
                eligible = (target_matrix[:, left].ne(ACK) & target_matrix[:, right].ne(ACK)
                            & target_matrix[:, left].ne(target_matrix[:, right]))
                total += int(eligible.sum())
                correct += int((eligible & pred_matrix[:, left].eq(target_matrix[:, left])
                                & pred_matrix[:, right].eq(target_matrix[:, right])).sum())
            result.update(counterfactual_query_pairs=total, counterfactual_correct=correct,
                          counterfactual_accuracy=correct / total if total else None)
            if replies:
                reply_pred = torch.cat(reply_actions)[query]
                result.update(query_reply_exact_accuracy=float(torch.cat(generated_correct)[query].float().mean()),
                    action_reply_agreement=float(reply_pred.eq(qpred).float().mean()),
                    reply_parseable_query_count=int(reply_pred.ge(0).sum()))
            return result
    finally:
        for module, training in modes:
            module.training = training
        if checkpoint_digest(model) != before:
            raise RuntimeError("audit evaluation changed student weights")


def score_banks(model, banks, **options):
    results = {key: evaluate_bank(model, rows, **options) for key, rows in banks.items()}
    return {"per_bank": results,
            "macro_query_accuracy": sum(row["query_accuracy"] for row in results.values()) / len(results),
            "macro_pair_accuracy": sum(row["counterfactual_accuracy"] or 0 for row in results.values()) / len(results)}


def fresh_adaptation_trainer(weights, support, memory_mode, device="cpu"):
    trainer = CognitiveTrainer(support, seed=ADAPT_SEED, memory_mode=memory_mode,
                               device=device, batch_size=64, learning_rate=.003)
    trainer.model.load_state_dict(weights, strict=True)
    if trainer.optimizer.state or trainer.updates:
        raise RuntimeError("adaptation must begin without optimizer history")
    return trainer


def verify_cpu_restart(weights, memory_mode, episode):
    """Serialize at turn three; require bitwise continuation on the CPU."""
    model = build_cognitive_student(INITIAL_SEED, memory_mode=memory_mode)
    model.load_state_dict(weights)
    model.eval()
    digest = checkpoint_digest(model)
    batches = encode_dialogues(model, [episode])
    with torch.inference_mode():
        state = None
        decoder = torch.full((1, 1), ByteCodec.BOS, dtype=torch.long)
        for batch in batches[:3]:
            _, state = _run_turn(model, batch, state, decoder=decoder)
        stream = io.BytesIO()
        torch.save({"weights": _cpu_copy(model.state_dict()),
                    "state": model.state_to_dict(state, checkpoint_sha256=digest)}, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
        restored = build_cognitive_student(INITIAL_SEED, memory_mode=memory_mode)
        restored.load_state_dict(saved["weights"])
        restored.eval()
        restarted = restored.state_from_dict(saved["state"], checkpoint_sha256=checkpoint_digest(restored))
        for batch in batches[3:]:
            left, state = _run_turn(model, batch, state, decoder=decoder)
            right, restarted = _run_turn(restored, batch, restarted, decoder=decoder)
            for field in ("logits", "language_logits", "production_context"):
                if not torch.equal(left[field], right[field]):
                    raise RuntimeError(f"CPU restart changed {field}")
            if not torch.equal(_generate_reply_tokens(model, left["production_context"]),
                               _generate_reply_tokens(restored, right["production_context"])):
                raise RuntimeError("CPU restart changed generated replies")
    return {"exact": True, "device": "cpu", "serialized_after_turn": 3,
            "checked_remaining_turns": 3, "checkpoint_sha256": digest}


def adapt_candidate(weights, memory_mode, banks, *, device="cpu"):
    started = time.monotonic()
    trainer = fresh_adaptation_trainer(weights, banks["support"], memory_mode, device)
    base_digest = checkpoint_digest(trainer.model)
    retained_before = score_banks(trainer.model, banks["retained"])
    transfer_before = score_banks(trainer.model, banks["trained_family_transfer"])
    curve = []
    training_seconds = 0.
    for budget in BUDGETS:
        while trainer.updates < budget:
            start = time.monotonic()
            trainer.step(HELDOUT_FAMILY)
            training_seconds += time.monotonic() - start
        metrics = score_banks(trainer.model, banks["query"], replies=budget == BUDGETS[-1])
        curve.append({"updates": budget, "support_exposures": budget * 64, **metrics})
    retained_after = score_banks(trainer.model, banks["retained"], replies=True)
    transfer_after = score_banks(trainer.model, banks["trained_family_transfer"], replies=True)
    controls = {name: score_banks(trainer.model, banks["query"], **option)
        for name, option in (("blank_text", {"blank_text": True}),
                             ("reset_each_turn", {"reset_each_turn": True}))}
    area = sum((right["updates"] - left["updates"]) *
        (right["macro_query_accuracy"] + left["macro_query_accuracy"]) / 2
        for left, right in zip(curve, curve[1:])) / BUDGETS[-1]
    final_weights = _cpu_copy(trainer.model.state_dict())
    result = {"memory_mode": memory_mode, "initial_weight_sha256": base_digest,
        "optimizer_reset": True, "adaptation_seed": ADAPT_SEED, "support_episodes": 64,
        "curve": curve, "normalized_adaptation_auc": area,
        "retention_before": retained_before, "retention_after": retained_after,
        "trained_family_transfer_before": transfer_before,
        "trained_family_transfer_after": transfer_after,
        "retention_delta_by_family": {family: retained_after["per_bank"][family]["query_accuracy"] -
            retained_before["per_bank"][family]["query_accuracy"] for family in TRAIN_FAMILIES},
        "controls_at_16": controls, "training_seconds": training_seconds,
        "wall_seconds": time.monotonic() - started,
        "cpu_restart": verify_cpu_restart(final_weights, memory_mode, banks["query"]["level_2"][0]),
        "final_weight_sha256": checkpoint_digest(trainer.model)}
    return result, trainer.snapshot()


def audit(paths, directory, *, device="cpu"):
    torch.set_num_threads(1)
    directory = Path(directory)
    with run_lock(directory):
        manifest, banks = load_prepared(directory)
        runs = verify_runs(paths, manifest)
        # Exclusions are deterministic and fixed before any student is scored.
        training_transcripts = {transcript(row) for run in runs for rows in run["training_banks"].values() for row in rows}
        exclusions = {}
        for group in ("retained", "trained_family_transfer"):
            exclusions[group] = {}
            for family, rows in banks[group].items():
                banks[group][family], exclusions[group][family] = exclude_overlapping_pairs(rows, training_transcripts)
        inputs = {"schema": SCHEMA, "manifest_sha256": file_hash(directory / "manifest.json"),
            "device": device, "runs": [{key: row[key] for key in ("key", "path", "checkpoint_sha256")} for row in runs],
            "whole_pair_exclusions": exclusions, "evaluated_banks": bank_manifest(banks)}
        marker = directory / "evaluation-started.json"
        if marker.exists() and json.loads(marker.read_text(encoding="utf8")) != inputs:
            raise ValueError("audit already began with different runs or settings")
        atomic_json(marker, inputs)
        report_path = directory / "report.json"
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf8"))
        candidates = [(row["key"], row["memory_mode"], row["weights"]) for row in runs]
        candidates.extend((f"fresh-{mode}", mode, _cpu_copy(build_cognitive_student(INITIAL_SEED,
            memory_mode=mode).state_dict())) for mode in ("episodic", "recurrent"))
        results = {}
        for key, mode, weights in candidates:
            path = directory / f"{key}.json"
            if path.exists():
                results[key] = json.loads(path.read_text(encoding="utf8"))
                continue
            result, saved = adapt_candidate(weights, mode, banks, device=device)
            result["candidate"] = key
            atomic_checkpoint(directory / f"{key}-adapted.pt", saved)
            atomic_json(path, result)
            results[key] = result
            print(json.dumps({"candidate": key, "zero_shot": result["curve"][0]["macro_query_accuracy"],
                "adapted_query": result["curve"][-1]["macro_query_accuracy"],
                "adapted_pairs": result["curve"][-1]["macro_pair_accuracy"],
                "adaptation_auc": result["normalized_adaptation_auc"]}), flush=True)
        for row in runs:
            if file_hash(Path(row["path"]) / "latest.pt") != row["checkpoint_sha256"]:
                raise RuntimeError("audit changed a general-training checkpoint")
        if audit_source_hashes() != manifest["source_sha256"]:
            raise RuntimeError("audit source changed during evaluation")
        report = {"schema": SCHEMA, "manifest": manifest, "inputs": inputs, "results": results,
            "base_checkpoints_unchanged": True,
            "interpretation": manifest["interpretation"],
            "prior_learning_gain_over_fresh_auc": {row["key"]: results[row["key"]]["normalized_adaptation_auc"] -
                results[f"fresh-{row['memory_mode']}"]["normalized_adaptation_auc"] for row in runs}}
        atomic_json(report_path, report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.prepare:
        if args.runs or args.output:
            parser.error("--prepare is a separate phase; omit --runs and --output")
        manifest = prepare(args.prepare)
        print(json.dumps({"prepared": str(args.prepare), "schema": manifest["schema"],
                          "heldout_family": manifest["heldout_family"], "banks": manifest["banks"]}))
    else:
        if not args.runs or not args.output:
            parser.error("audit requires --runs and the previously prepared --output directory")
        report = audit(args.runs, args.output, device=args.device)
        print(json.dumps({"complete": True, "candidates": len(report["results"]),
                          "base_checkpoints_unchanged": report["base_checkpoints_unchanged"]}))


if __name__ == "__main__":
    main()
