"""Train and evaluate a small glyph encoder plus rapid persistent associations.

Run from project root: python -m experiments.train_associative --steps 1000
No test identities are used for optimization or threshold selection. Auxiliary
occupancy targets belong only to the procedural training curriculum.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from brain_in_computer.associative import (
    AssociativeMemory, ENCODER_SCHEMA, GlyphEncoder, distinct_patterns,
    encoder_hash, render_glyph,
)


def embed_patterns(model, patterns, seeds, raw=False):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(patterns), 128):
            pixels = torch.stack([render_glyph(p, s) for p, s in zip(patterns[start:start+128], seeds[start:start+128])])
            if raw:
                pixels = pixels.mean(dim=1).flatten(1)
                pixels = pixels - pixels.amin(dim=1, keepdim=True)
                chunks.append(F.normalize(pixels, dim=-1))
            else:
                chunks.append(model(pixels))
    return torch.cat(chunks)


def retrieval_cases(model, patterns, seed, cases=512, raw=False):
    rng = random.Random(seed)
    support, targets, candidates, seeds = [], [], [], []
    for i in range(cases):
        choice = rng.sample(patterns, 5)
        present = i % 2 == 0
        support.append(choice[0])
        group = choice[:4] if present else choice[1:5]
        rng.shuffle(group)
        candidates.extend(group)
        targets.append(group.index(choice[0]) if present else -1)
        seeds.extend([rng.randrange(2**31) for _ in range(4)])
    queries = embed_patterns(model, support, [rng.randrange(2**31) for _ in support], raw)
    objects = embed_patterns(model, candidates, seeds, raw).reshape(cases, 4, -1)
    scores = (objects * queries.unsqueeze(1)).sum(dim=-1)
    return scores, torch.tensor(targets)


def naming_cases(model, patterns, seed, bank_size=32, repetitions=8, raw=False):
    rng = random.Random(seed)
    known = patterns[:bank_size]
    unknown = patterns[bank_size:2*bank_size]
    bank = embed_patterns(model, known, [rng.randrange(2**31) for _ in known], raw)
    queries, targets = [], []
    for _ in range(repetitions):
        for i, p in enumerate(known + unknown):
            queries.append(p)
            targets.append(i if i < bank_size else -1)
    embeddings = embed_patterns(model, queries, [rng.randrange(2**31) for _ in queries], raw)
    return embeddings @ bank.T, torch.tensor(targets)


def decisions(scores, threshold, margin):
    values, indexes = scores.topk(2, dim=-1)
    accepted = (values[:, 0] >= threshold) & (values[:, 1] < threshold) & (values[:, 0] - values[:, 1] >= margin) & (values[:, 0] - values[:, 1] > 1e-7)
    return torch.where(accepted, indexes[:, 0], -1)


def metrics(scores, targets, threshold, margin):
    predicted = decisions(scores, threshold, margin)
    known, accepted = targets >= 0, predicted >= 0
    right = predicted == targets
    return {"episodes": len(targets), "overall_accuracy": right.float().mean().item(),
            "known_accuracy": right[known].float().mean().item(),
            "unknown_rejection_rate": right[~known].float().mean().item(),
            "unknown_false_accept_rate": accepted[~known].float().mean().item(),
            "coverage": accepted.float().mean().item(),
            "known_coverage": accepted[known].float().mean().item(),
            "conditional_accuracy": right[accepted].float().mean().item() if accepted.any() else None,
            "forced_choice_known_accuracy": (scores[known].argmax(-1) == targets[known]).float().mean().item()}


def calibrate(*cases):
    best = (-1, None)
    # Empirical validation quantiles cover random encoders whose cosine scores
    # cluster near 1, as well as trained and raw-pixel score distributions.
    values = torch.cat([scores.topk(2, dim=-1).values for scores, _ in cases])
    thresholds = sorted(set([-1., 1.] + torch.quantile(values.flatten(), torch.linspace(0, 1, 129)).tolist()))
    margins = sorted(set([0.] + torch.quantile(values[:, 0] - values[:, 1], torch.linspace(0, 1, 33)).tolist()))
    for threshold in thresholds:
        for margin in margins:
            correct = sum((decisions(scores, threshold, margin) == target).float().mean().item() for scores, target in cases) / len(cases)
            if correct > best[0]:
                best = correct, {"threshold": threshold, "margin": margin}
    return {**best[1], "validation_macro_accuracy": best[0], "method": "validation-quantile grid maximizing balanced present/absent overall accuracy over retrieval and 32-label naming; reject multiple above-threshold matches; cosine score is not a probability"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--output", default="runs/associative")
    parser.add_argument("--evaluate-existing", help="Reuse existing encoder weights; perform no optimization")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.batch_size > 512:
        parser.error("positive steps and batch size 2..512 required")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train = distinct_patterns(512, args.seed)
    validation = distinct_patterns(128, args.seed + 1, train)
    development_test = distinct_patterns(128, args.seed + 2, train + validation)
    test = distinct_patterns(128, args.seed + 3, train + validation + development_test)
    model = GlyphEncoder()
    random_model = GlyphEncoder()
    random_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    started = time.perf_counter()
    history = []
    previous_training_seconds = None
    if args.evaluate_existing:
        previous = torch.load(args.evaluate_existing, map_location="cpu", weights_only=True)
        if previous.get("schema") != ENCODER_SCHEMA or any(previous["training"][key] != getattr(args, key) for key in ("seed", "steps", "batch_size")):
            raise ValueError("existing checkpoint does not match requested training configuration")
        model.load_state_dict(previous["model"])
        history = previous["history"]
        previous_training_seconds = history[-1]["elapsed_seconds"]
    for step in range(1, 1 if args.evaluate_existing else args.steps + 1):
        patterns = rng.sample(train, args.batch_size)
        first = torch.stack([render_glyph(p, rng.randrange(2**31)) for p in patterns])
        second = torch.stack([render_glyph(p, rng.randrange(2**31)) for p in patterns])
        pixels = torch.cat((first, second))
        targets = torch.tensor([[(p >> bit) & 1 for bit in range(16)] for p in patterns], dtype=torch.float32).repeat(2, 1)
        embeddings, occupancy = model.encode(pixels)
        logits = embeddings @ embeddings.T / .12
        logits.fill_diagonal_(-1e9)
        matches = (torch.arange(2 * args.batch_size) + args.batch_size) % (2 * args.batch_size)
        contrastive = F.cross_entropy(logits, matches)
        reconstruction = F.binary_cross_entropy_with_logits(occupancy, targets)
        loss = contrastive + reconstruction
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            row = {"step": step, "loss": loss.item(), "contrastive": contrastive.item(),
                   "occupancy_bce": reconstruction.item(), "elapsed_seconds": time.perf_counter() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
    model.eval()
    train_seconds = previous_training_seconds if previous_training_seconds is not None else time.perf_counter() - started
    val_retrieval = retrieval_cases(model, validation, args.seed + 10000)
    val_naming = naming_cases(model, validation, args.seed + 11000)
    calibration = calibrate(val_retrieval, val_naming)
    print(json.dumps({"calibration": calibration}), flush=True)
    checkpoint = {"schema": ENCODER_SCHEMA, "config": asdict(model.config),
                  "model": model.state_dict(), "encoder_hash": encoder_hash(model),
                  "calibration": calibration, "training": vars(args),
                  "train_identities": train, "validation_identities": validation,
                  "development_test_identities": development_test,
                  "test_identities": test, "history": history}
    torch.save(checkpoint, output / "encoder.pt")
    report = {"schema": "bic-associative-evaluation-v1", "seed": args.seed,
              "training": vars(args), "training_seconds_cpu_one_thread": train_seconds,
              "parameters": sum(p.numel() for p in model.parameters()),
              "encoder_hash": checkpoint["encoder_hash"], "calibration": calibration,
              "identity_split_counts": {"train": len(train), "validation": len(validation), "reserved_development_test": len(development_test), "final_test": len(test)},
              "identity_splits_disjoint": len(set(train + validation + development_test + test)) == len(train + validation + development_test + test),
              "test_protocol": "Final identities use seed+3 and exclude train, validation, and initial observed development-test identities (seed+2). Calibration was frozen from validation before final evaluation. No optimizer updates followed the initial test; a validation-grid coverage and multiple-match ambiguity correction preceded this fresh final test.",
              "history": history,
              "notes": ["4x4 synthetic glyph patches, not natural images or general English.",
                        "One teacher-selected label and one visual exemplar per new object; labels are symbolic addresses.",
                        "Color, cell size, and translation vary independently between teaching and query images.",
                        "Image embedding is learned; persistent memory is a bounded exemplar bank, not end-to-end neural consolidation.",
                        "Final test identities were disjoint and unused during optimizer training and threshold calibration.",
                        "Confidence is cosine similarity, not a probability. Unknown rejection is distribution-specific."]}
    for tag, baseline, raw in (("trained", model, False), ("random_encoder", random_model, False), ("raw_grayscale_pixels", model, True)):
        if tag == "trained":
            cal = calibration
        else:
            cal = calibrate(retrieval_cases(baseline, validation, args.seed + 10000, raw=raw), naming_cases(baseline, validation, args.seed + 11000, raw=raw))
        retrieval = retrieval_cases(baseline, test, args.seed + 20000, raw=raw)
        naming = naming_cases(baseline, test, args.seed + 21000, raw=raw)
        report[tag] = {"calibration": cal,
                       "retrieval_four_candidates": metrics(*retrieval, cal["threshold"], cal["margin"]),
                       "naming_32_labels": metrics(*naming, cal["threshold"], cal["margin"])}
        print(json.dumps({tag: report[tag]}), flush=True)
    memory = AssociativeMemory(model, calibration["threshold"], calibration["margin"])
    for i, p in enumerate(test[:16]):
        memory.teach(f"novel_{i}", render_glyph(p, args.seed + 30000 + i))
    queries = [render_glyph(p, args.seed + 31000 + i) for i, p in enumerate(test[:16])]
    before = [memory.name(q) for q in queries]
    for i, p in enumerate(test[16:48], start=16):
        memory.teach(f"novel_{i}", render_glyph(p, args.seed + 30000 + i))
    after = [memory.name(q) for q in queries]
    memory.save(output / "demonstration_memory.json")
    reloaded = AssociativeMemory.load(output / "demonstration_memory.json", model)
    restored = [reloaded.name(q) for q in queries]
    report["retention"] = {
        "initial_labels": 16, "added_labels": 32, "final_labels": len(memory.labels),
        "before_correct": sum(r["label"] == f"novel_{i}" for i, r in enumerate(before)),
        "after_correct": sum(r["label"] == f"novel_{i}" for i, r in enumerate(after)),
        "restored_correct": sum(r["label"] == f"novel_{i}" for i, r in enumerate(restored)),
        "save_load_exact_decisions": after == restored,
        "fresh_process": False,
        "note": "Frozen perception preserves exemplar vectors; additional competing labels can still cause retrieval interference."}
    report["duplicate_candidate_control"] = memory.find("novel_0", [queries[0], queries[0]])
    report["transformed_duplicate_candidate_control"] = memory.find("novel_0", [render_glyph(test[0], 991), render_glyph(test[0], 992)])
    report["unknown_label_control"] = memory.find("never_taught", [queries[0]])
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"retention": report["retention"], "report": str(output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
