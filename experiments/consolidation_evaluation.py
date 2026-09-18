"""Teacher-free prospective adaptation and retention audit for v3 sequence banks.

This helper writes no files and never chooses a checkpoint. Replay has extra
forward/backward computation at matched new-example exposure, not matched compute.
The caller owns freezing sources/banks and checking parent checkpoint files.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_student import _check_finite_tree, _cpu_copy
from experiments.audit_sequence_study import sequence_restart
from experiments.diversity_evaluation import (
    evaluate_diverse_banks, validate_diverse_evaluation_rows,
)


SCHEMA = "bic-consolidation-evaluation-v1"
BUDGETS = (0, 4, 16, 64, 256)
RATE, BATCH_SIZE, REPLAY_BATCH_SIZE, REPLAY_WEIGHT = .001, 64, 32, .5
HELDOUT = "arithmetic_updates"
OLD_FAMILIES = ("variable_binding", "graph_reachability", "conditional_logic")
QUERY_PANELS = ("names", "worlds", "both", "advanced")
METRICS = ("macro_query_accuracy", "macro_pair_accuracy", "macro_later_known_accuracy")


def normalized_areas(curve):
    """Sparse trapezoidal areas; missing denominators fail instead of becoming zero."""
    if not isinstance(curve, (list, tuple)) or len(curve) < 2:
        raise ValueError("an adaptation curve requires at least two endpoints")
    budgets = [point.get("updates") for point in curve]
    if (any(type(value) is not int or value < 0 for value in budgets)
            or budgets[0] != 0 or any(a >= b for a, b in zip(budgets, budgets[1:]))):
        raise ValueError("curve budgets must increase strictly from zero")
    for point in curve:
        for metric in METRICS:
            value = point.get(metric)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("every curve metric requires a finite measured denominator")
    return {metric: sum((right["updates"] - left["updates"]) *
                (left[metric] + right[metric]) / 2 for left, right in zip(curve, curve[1:]))
            / budgets[-1] for metric in METRICS}


def _validate_banks(banks):
    required = {"development", "retained", "advanced", "support", "query"}
    if not isinstance(banks, Mapping) or set(banks) != required:
        raise ValueError("audit requires development, retained, advanced, support and query banks")
    if not isinstance(banks["support"], Mapping) or set(banks["support"]) != {HELDOUT}:
        raise ValueError("support must contain only the declared held-out family")
    if not isinstance(banks["query"], Mapping) or set(banks["query"]) != set(QUERY_PANELS):
        raise ValueError("query must preserve names/worlds/both/advanced panels")
    for group in ("development", "retained", "advanced", "query"):
        if not isinstance(banks[group], Mapping) or not banks[group]:
            raise ValueError("each evaluation group requires named complete banks")
        for rows in banks[group].values():
            # Authenticate every original row before any optimizer or control.
            validate_diverse_evaluation_rows(rows, BATCH_SIZE)
            if any(row["split"] == "train" for row in rows):
                raise ValueError("evaluation rows require an evaluation admission role")
            allowed = {HELDOUT} if group == "query" else set(OLD_FAMILIES)
            if any(row["family"] not in allowed for row in rows):
                raise ValueError("evaluation family violates the held-out subject boundary")


def _bank_storage(banks):
    if banks is None:
        return {"episodes": 0, "canonical_json_bytes": 0, "observation_utf8_bytes": 0,
                "per_family": {}}
    per_family = {family: {
        "episodes": len(rows),
        "complete_pair_slots": len(rows) // 2,
        "distinct_rendered_pair_groups": len({row["counterfactual_group"] for row in rows}),
        "observation_utf8_bytes": sum(len(turn["text"].encode("utf-8")) for row in rows for turn in row["turns"]),
    } for family, rows in banks.items()}
    return {"episodes": sum(value["episodes"] for value in per_family.values()),
        "canonical_json_bytes": len(json.dumps(banks, sort_keys=True, separators=(",", ":"),
                                                 allow_nan=False).encode("utf-8")),
        "observation_utf8_bytes": sum(value["observation_utf8_bytes"] for value in per_family.values()),
        "per_family": per_family}


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def adapt_candidate(weights, banks, replay_banks=None, device="cpu", seed=3701):
    """Return ``(result, trainer_snapshot)`` for one fixed prospective endpoint.

    ``weights`` is an ordinary sequence state_dict. The optimizer is always
    fresh. Supplied weights and banks remain caller-owned and unchanged. The
    four query panels retain their full metrics/denominators; macro scores weight
    panels equally, despite different query counts. Every evaluation generates
    free replies from BOS only through the frozen v3 evaluator.
    """
    from experiments.consolidation_training import ConsolidationTrainer

    started = time.monotonic()
    _validate_banks(banks)
    if (replay_banks is not None and (not isinstance(replay_banks, Mapping)
            or set(replay_banks) != set(OLD_FAMILIES))):
        raise ValueError("replay must contain all and only the three old training families")
    _check_finite_tree(weights, "initial_adaptation_weights")
    parent_weights = _cpu_copy(weights)
    trainer = ConsolidationTrainer(banks["support"], replay_banks=replay_banks,
        seed=seed, device=device, batch_size=BATCH_SIZE, replay_batch_size=REPLAY_BATCH_SIZE,
        replay_weight=REPLAY_WEIGHT, learning_rate=RATE)
    trainer.model.load_state_dict(weights, strict=True)
    model_device = next(trainer.model.parameters()).device
    initial_digest = checkpoint_digest(trainer.model)
    evaluate = lambda group, **options: evaluate_diverse_banks(
        trainer.model, banks[group], batch_size=BATCH_SIZE, score_replies=True, **options)
    development_before = evaluate("development")
    retained_before, advanced_before = evaluate("retained"), evaluate("advanced")
    curve, training_seconds = [], 0.
    for budget in BUDGETS:
        while trainer.updates < budget:
            _synchronize(model_device)
            tick = time.monotonic()
            trainer.step(HELDOUT)
            _synchronize(model_device)
            training_seconds += time.monotonic() - tick
        curve.append({"updates": trainer.updates, "optimizer_updates": trainer.updates,
                      "exposures": dict(trainer.exposures), **evaluate("query")})
    development_after = evaluate("development")
    retained_after, advanced_after = evaluate("retained"), evaluate("advanced")
    controls = {name: evaluate("query", **options) for name, options in (
        ("blank_text", {"blank_text": True}), ("reset_history", {"reset_history": True}))}
    restart = sequence_restart(trainer.model, banks["query"]["worlds"][0])
    snapshot = trainer.snapshot()
    if trainer.exposures["support_episodes"] != BUDGETS[-1] * BATCH_SIZE:
        raise RuntimeError("new-example exposure budget differs from the protocol")
    expected_replay = BUDGETS[-1] * REPLAY_BATCH_SIZE if replay_banks is not None else 0
    if trainer.exposures["replay_episodes"] != expected_replay:
        raise RuntimeError("replay exposure budget differs from the protocol")
    if set(parent_weights) != set(weights) or any(
            not torch.equal(parent_weights[key], weights[key].detach().cpu()) for key in parent_weights):
        raise RuntimeError("adaptation changed the caller's parent weights")
    result = {"schema": SCHEMA, "condition": "replay" if replay_banks is not None else "ordinary",
        "seed": seed, "learning_rate": RATE, "optimizer_reset": True,
        "optimizer_updates": trainer.updates, "new_batch_size": BATCH_SIZE,
        "replay_batch_size": REPLAY_BATCH_SIZE if replay_banks is not None else 0,
        "replay_weight": REPLAY_WEIGHT if replay_banks is not None else 0.,
        "budgets": list(BUDGETS), "curve": curve, "areas": normalized_areas(curve),
        "query_macro_weighting": "equal names/worlds/both/advanced panels, not pooled query counts",
        "development_before": development_before, "development_after": development_after,
        "retention_before": retained_before, "retention_after": retained_after,
        "advanced_before": advanced_before, "advanced_after": advanced_after,
        "controls_at_256": controls, "controls_at_updates": trainer.updates,
        "exposures": dict(trainer.exposures), "support_storage": _bank_storage(banks["support"]),
        "replay_storage": _bank_storage(replay_banks), "recipe": snapshot["recipe"],
        "cpu_restart": restart, "initial_weights_sha256": initial_digest,
        "final_weights_sha256": checkpoint_digest(trainer.model), "parent_weights_unchanged": True,
        "training_seconds": training_seconds, "wall_seconds": time.monotonic() - started,
        "time_scope": "Training covers synchronized optimizer steps including sampling and both losses; wall includes validation, packing, scoring, snapshot and CPU restart. Per-component GPU times are not measured.",
        "comparison_scope": "Matched new-example exposure and optimizer updates; replay adds computation and old examples. Not a compute-matched comparison.",
        "automatic_promotion": False}
    return result, snapshot
