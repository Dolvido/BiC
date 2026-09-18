"""Count-authenticated metrics and prospective development-only rate selection.

No model inference, training, source mutation or checkpoint selection occurs here.
Canonical bank admission belongs to the caller; this module binds metric evidence
to those exact rows and validates every denominator again.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math

from experiments.composition_evaluation import METRICS
from experiments.sequence_student import SequenceConfig
from experiments.summarize_composition_study import (
    _opposite_pair_denominators, _validate_metrics, study as _previous_study,
)
from experiments.train_cognitive import fingerprint_rows

WIDTHS = (96, 192, 256)
FAMILIES = ("color", "count", "switch")
LENGTHS = (8, 10, 12)
PANELS = ("name_only", "value_only", "both", "composed")
RATES = {"r0003": .0003, "r001": .001, "r003": .003}
TIE_PREFERENCE = (.001, .0003, .003)
EXPECTED_BANKS = frozenset(f"{panel}/{family}/t{turns}"
                         for panel in PANELS for family in FAMILIES for turns in LENGTHS)
RULE = ("At update600, maximize rounded12-decimal lexicographic tuple: minimum9-cell "
        "paired action/reply mean, macro9-cell paired action/reply mean, equal-cell/panel "
        "known accuracy, negative equal-cell/panel unsupported ASK on known queries; "
        "remaining ties prefer .001, .0003, .003. No model promotion.")


def _finite(value, *, minimum=0., maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < minimum or maximum is not None and value > maximum):
        raise ValueError("finite metric within its declared range required")
    return value


def _ratio(correct, total):
    if type(correct) is not int or type(total) is not int or total <= 0 or not 0 <= correct <= total:
        raise ValueError("positive canonical denominator and bounded integer count required")
    return correct / total


def aggregate_rows(per_bank):
    """Equal-bank means; a partially unmeasured metric remains unmeasured."""
    if not isinstance(per_bank, dict) or not per_bank:
        raise ValueError("nonempty named metric banks required")
    result = {"per_bank": deepcopy(per_bank)}
    for metric in METRICS:
        values = [row[metric] for row in per_bank.values()]
        for value in values:
            if value is not None:
                _finite(value, maximum=1.)
        result[f"macro_{metric}"] = (sum(values) / len(values)
                                      if all(value is not None for value in values) else None)
    return result


def validate_metrics(metrics, rows, manifests, config, role, control="normal", replies=True):
    """Validate actual config before adapting a private copy to the old checker.

    The previous validator's globals and the caller's evidence remain unchanged.
    Its canonical target/pair/ASK/reply checks are retained for every width.
    """
    expected_config = asdict(config) if isinstance(config, SequenceConfig) else asdict(SequenceConfig(**config))
    if role not in ("dev", "audit", "train_fit") or control not in ("normal", "blank", "reset") or type(replies) is not bool:
        raise ValueError("explicit valid scoring role, control and reply flag required")
    if (not isinstance(rows, dict) or not rows or set(rows) != set(manifests)
            or set(metrics.get("per_bank", {})) != set(rows)):
        raise ValueError("metric, row and manifest bank sets differ")
    for name, canonical in rows.items():
        if not canonical or len(canonical) % 2:
            raise ValueError("canonical complete pairs required")
        turns = len(canonical[0]["turns"])
        if any(len(row["turns"]) != turns for row in canonical):
            raise ValueError("bank has mixed episode lengths")
        counts = [{str(target): sum(row["turns"][turn]["target"] == target for row in canonical)
                   for target in range(4)} for turn in range(turns)]
        facts = {"sha256": fingerprint_rows(canonical), "episodes": len(canonical),
                 "complete_pairs": len(canonical) // 2, "turns": turns, "target_counts_by_turn": counts}
        if any(manifests[name].get(key) != value for key, value in facts.items()):
            raise ValueError("manifest differs from exact canonical rows")
        row = metrics["per_bank"][name]
        if row.get("bank", {}).get("config") != expected_config:
            raise ValueError("recorded metric model config differs from actual configuration")
        for key in METRICS:
            if row[key] is not None:
                _finite(row[key], maximum=1.)
        _finite(row["query_loss"])
        _finite(row["brier_score"], maximum=2. + 1e-6)
        if "seconds" in row:
            _finite(row["seconds"])
    adapted = deepcopy(metrics)
    for row in adapted["per_bank"].values():
        row["bank"]["config"] = asdict(_previous_study.CONFIG)
    _validate_metrics(adapted, rows, manifests, role=role, control=control, replies=replies,
                      pair_denominators=_opposite_pair_denominators(rows))
    for row in metrics["per_bank"].values():
        if replies:
            # Exact canonical replies must parse; two exact final replies are
            # needed for each reported successful final reply pair.
            if (row["query_reply_correct"] > row["reply_parseable_queries"]
                    or 2 * row["final_reply_pair_correct"] > row["query_reply_correct"]
                    or row["action_reply_agreement"] * row["query_total"]
                    > row["reply_parseable_queries"] + max(1e-5, 1e-7 * row["query_total"])):
                raise ValueError("reply correctness/agreement exceeds possible parseable evidence")
        elif any(row[key] is not None for key in ("query_reply_correct", "final_reply_pair_correct", "reply_parseable_queries")):
            raise ValueError("unmeasured reply counts must remain null")
    return True


def _calibration_details(metrics):
    per_bank = metrics.get("per_bank", {})
    if set(per_bank) != EXPECTED_BANKS:
        raise ValueError("calibration requires all36 panel/domain/length banks")
    cells = {}
    for family in FAMILIES:
        for turns in LENGTHS:
            components = {}
            for panel in PANELS:
                row = per_bank[f"{panel}/{family}/t{turns}"]
                pair_total = row["final_pairs"]["total"]
                action = _ratio(row["final_pairs"]["correct"], pair_total)
                reply = _ratio(row["final_reply_pair_correct"], pair_total)
                known = _ratio(row["known_correct"], row["known_total"])
                confusion = row["confusion_matrix"]
                if (not isinstance(confusion, list) or len(confusion) != 4
                        or any(not isinstance(line, list) or len(line) != 4 for line in confusion)
                        or any(type(value) is not int or value < 0 for line in confusion for value in line)
                        or sum(confusion[0]) + sum(confusion[1]) != row["known_total"]
                        or confusion[0][0] + confusion[1][1] != row["known_correct"]):
                    raise ValueError("calibration known confusion/counts differ")
                unsupported_count = confusion[0][2] + confusion[1][2]
                unsupported = _ratio(unsupported_count, row["known_total"])
                for key, value in (("final_pair_accuracy", action), ("final_reply_pair_accuracy", reply),
                                   ("known_accuracy", known)):
                    _finite(row[key], maximum=1.)
                    if not math.isclose(row[key], value, rel_tol=0, abs_tol=1e-7):
                        raise ValueError("calibration score differs from integer counts")
                if row.get("free_running_replies") is not True or row.get("teacher_used_for_policy") is not False or row.get("decoder_prefix") != "BOS only" or row.get("control") != "normal":
                    raise ValueError("calibration requires normal teacher-free free-reply evidence")
                components[panel] = {"action_pairs": deepcopy(row["final_pairs"]),
                    "reply_pairs": {"correct": row["final_reply_pair_correct"], "total": pair_total},
                    "action_pair_accuracy": action, "reply_pair_accuracy": reply,
                    "paired_action_reply_mean": (action + reply) / 2,
                    "known": {"correct": row["known_correct"], "total": row["known_total"], "accuracy": known},
                    "unsupported_ask_on_known": {"count": unsupported_count, "total": row["known_total"], "rate": unsupported}}
            cells[f"{family}/t{turns}"] = {"panels": components,
                "paired_action_reply_mean": sum(p["paired_action_reply_mean"] for p in components.values()) / 4,
                "known_accuracy": sum(p["known"]["accuracy"] for p in components.values()) / 4,
                "unsupported_ask_rate": sum(p["unsupported_ask_on_known"]["rate"] for p in components.values()) / 4}
    values = list(cells.values())
    rank = tuple(round(value, 12) for value in (
        min(cell["paired_action_reply_mean"] for cell in values),
        sum(cell["paired_action_reply_mean"] for cell in values) / 9,
        sum(cell["known_accuracy"] for cell in values) / 9,
        -sum(cell["unsupported_ask_rate"] for cell in values) / 9))
    return rank, cells


def calibration_rank(metrics):
    return _calibration_details(metrics)[0]


def select_rates(results):
    expected = {f"w{width}-{label}": (width, rate) for width in WIDTHS for label, rate in RATES.items()}
    if not isinstance(results, dict) or set(results) != set(expected):
        raise ValueError("all9 declared width/rate calibration jobs are required")
    candidates = {}
    for job, (width, rate) in expected.items():
        result = results[job]
        if type(result["width"]) is not int or result["width"] != width or type(result["rate"]) not in (int, float) or result["rate"] != rate:
            raise ValueError("calibration job width/rate identity differs")
        config = asdict(SequenceConfig(width=width, feedforward=4 * width, max_turns=12))
        if any(row.get("bank", {}).get("config") != config
               for row in result["metrics"].get("per_bank", {}).values()):
            raise ValueError("calibration width/config evidence differs")
        rank, cells = _calibration_details(result["metrics"])
        candidates[job] = {"width": width, "rate": rate, "rank": list(rank), "cells": cells}
    choices = {}
    for width in WIDTHS:
        possible = [job for job, row in candidates.items() if row["width"] == width]
        chosen = max(possible, key=lambda job: (tuple(candidates[job]["rank"]),
                                               -TIE_PREFERENCE.index(candidates[job]["rate"])))
        choices[str(width)] = {"job": chosen, **{key: deepcopy(candidates[chosen][key]) for key in ("width", "rate", "rank")}}
    return {"selected": choices, "candidates": candidates, "tie_preference": list(TIE_PREFERENCE),
            "round_decimal_places": 12, "rule": RULE, "automatic_promotion": False}


def _json_equal(left, right):
    # Saved Torch metadata may use integer bucket keys; JSON necessarily uses
    # strings. Normalize first so numeric-versus-lexical sorting cannot differ.
    def encode(value):
        normalized = json.loads(json.dumps(value, allow_nan=False))
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return encode(left) == encode(right)


def validate_audit_result(result, *, job, inputs, steps, config, expected_weights,
                          audit_rows, initial_rows, latest_rows, latest_metadata):
    """Authenticate a fresh or cached width audit without performing inference.

    Row arguments are flat named banks, e.g. ``initial/color/t8``. The caller
    reconstructs latest observed rows/metadata from the verified endpoint first.
    ``expected_weights`` binds every step to the earlier replay-verification
    receipt. This validator has no authority to open a sealed audit itself.
    """
    from experiments.realization_banks import _stats

    if (not isinstance(steps, (tuple, list)) or not steps
            or any(type(step) is not int or step < 0 for step in steps)
            or list(steps) != sorted(set(steps))):
        raise ValueError("audit requires ordered unique nonnegative checkpoint steps")
    normalized_weights = {str(key): value for key, value in expected_weights.items()}
    if (len(normalized_weights) != len(expected_weights)
            or set(normalized_weights) != {str(step) for step in steps}):
        raise ValueError("verified weight map does not cover every audit checkpoint")
    expected_keys = {"schema", "job", "inputs", "curve", "final", "initial_fit",
        "latest_observed_fit", "latest_observed_metadata", "controls", "cpu_restart",
        "seconds", "automatic_promotion"}
    if (not isinstance(result, dict) or set(result) != expected_keys
            or result.get("schema") != "bic-shared-capacity-study-v1"
            or result.get("job") != job or not _json_equal(result.get("inputs"), inputs)
            or result.get("automatic_promotion") is not False):
        raise ValueError("audit result identity, fields or source bindings differ")
    _finite(result["seconds"])
    if result["seconds"] <= 0:
        raise ValueError("audit elapsed runtime must be positive")
    curve = result["curve"]
    if (not isinstance(curve, list) or len(curve) != len(steps)
            or any(not isinstance(point, dict) or set(point) != {
                "updates", "episodes_per_family", "weights_sha256", "metrics"} for point in curve)):
        raise ValueError("audit curve is incomplete or malformed")
    audit_manifest = {name: _stats(rows) for name, rows in audit_rows.items()}
    for step, point in zip(steps, curve):
        if (type(point["updates"]) is not int or point["updates"] != step
                or type(point["episodes_per_family"]) is not int
                or point["episodes_per_family"] != step * 32
                or point["weights_sha256"] != normalized_weights[str(step)]):
            raise ValueError("audit curve checkpoint, exposure or weight identity differs")
        validate_metrics(point["metrics"], audit_rows, audit_manifest, config, "audit")
    if not _json_equal(result["final"], curve[-1]["metrics"]):
        raise ValueError("audit final metrics differ from the last fixed checkpoint")
    for key, rows in (("initial_fit", initial_rows), ("latest_observed_fit", latest_rows)):
        manifests = {name: _stats(bank) for name, bank in rows.items()}
        validate_metrics(result[key], rows, manifests, config, "train_fit")
    if not _json_equal(result["latest_observed_metadata"], latest_metadata):
        raise ValueError("latest fitting probe differs from the actual consumed endpoint")
    if not isinstance(result["controls"], dict) or set(result["controls"]) != {"blank", "reset"}:
        raise ValueError("both declared audit controls are required")
    for control, metrics in result["controls"].items():
        validate_metrics(metrics, audit_rows, audit_manifest, config, "audit", control=control)
    restart = result["cpu_restart"]
    expected_turns = len(next(iter(audit_rows.values()))[0]["turns"])
    if (not isinstance(restart, dict) or restart.get("exact") is not True
            or restart.get("device") != "cpu"
            or restart.get("weights_sha256") != normalized_weights[str(steps[-1])]
            or type(restart.get("utterances")) is not int or restart["utterances"] != expected_turns):
        raise ValueError("CPU restart is not exact for the verified endpoint")
    return True
