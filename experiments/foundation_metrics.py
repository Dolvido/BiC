"""Canonical foundation metric evidence without inference or version relabeling.

Numeric consistency cannot prove that cached predictions came from a particular
checkpoint. The owning runner must additionally bind source, bank and model hashes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
import re

from experiments.foundation_curriculum import FAMILIES, VERSION, validate_pair
from experiments.foundation_evidence import json_digest, training_cell
from experiments.composition_evaluation import METRICS
from experiments.sequence_student import SequenceConfig


SCHEMA = "bic-foundation-metric-summary-v1"
EXTRA_METRICS = ("ask_precision", "ask_recall", "action_reply_agreement", "query_loss", "brier_score", "unsupported_ask_rate")


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer")


def _finite(value, name, maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
            or maximum is not None and value > maximum):
        raise ValueError(name + " must be finite and within range")


def _ratio(value, correct, total, name):
    _integer(correct, name + " correct"); _integer(total, name + " total")
    if correct > total:
        raise ValueError(name + " correct exceeds its denominator")
    if total == 0:
        if value is not None:
            raise ValueError(name + " absent denominator must remain null")
    else:
        _finite(value, name, 1.)
        if not math.isclose(value, correct/total, rel_tol=0., abs_tol=1e-7):
            raise ValueError(name + " rate differs from counts")


def _same(left, right):
    if json.dumps(left, sort_keys=True, allow_nan=False) != json.dumps(right, sort_keys=True, allow_nan=False):
        raise ValueError("foundation metric identity differs")


def _cell(name):
    if type(name) is not str:
        raise ValueError("named foundation cell required")
    found = re.fullmatch(r"([^/]+)/([^/]+)/d([0-5])/(direct|copy|advance|composed)/t(8|10|12)", name)
    if not found:
        raise ValueError("bank names must be panel/family/depth/operator/length")
    panel, family, depth, operator, turns = found.groups()
    if family not in FAMILIES or operator not in (("direct",) if depth == "0" else ("copy", "advance") if depth == "1" else ("composed",)):
        raise ValueError("foundation cell mechanism differs from depth or family")
    return panel, family, "d"+depth, operator, turns


def _canonical(rows, name, config, role):
    _cell(name)
    if type(rows) is not list or not rows or len(rows) % 2:
        raise ValueError("complete canonical foundation pairs required")
    split = "train" if role == "train_fit" else role
    groups, cell = set(), name.split("/", 1)[1]
    for index in range(0, len(rows), 2):
        pair = rows[index:index+2]
        validate_pair(pair)
        if pair[0]["counterfactual_group"] in groups:
            raise ValueError("duplicate foundation pair")
        groups.add(pair[0]["counterfactual_group"])
        if any(row["split"] != split or training_cell(row) != cell for row in pair):
            raise ValueError("canonical bank role, mechanism or cell differs")
    turns = len(rows[0]["turns"])
    if turns > config.max_turns or any(
            sum(len(turn["text"].encode("utf8"))+2 for turn in row["turns"]) > config.max_positions
            or any(len(turn["text"].encode("utf8")) > config.max_input_bytes
                   or len(turn["reply"].encode("utf8")) > config.max_output_bytes for turn in row["turns"])
            for row in rows):
        raise ValueError("canonical observations or replies exceed recorded model capacity")
    identity = {"sha256": json_digest(rows), "version": VERSION, "episodes": len(rows),
                "turns": turns, "role": role, "config": asdict(config)}
    targets = [[row["turns"][turn]["target"] for row in rows] for turn in range(turns)]
    counts = [[values.count(target) for target in range(4)] for values in targets]
    opposite = [sum({values[index], values[index+1]} == {0, 1} for index in range(0, len(rows), 2)) for values in targets]
    return identity, counts, opposite


def _validate_row(row, identity, counts, opposite, control):
    _same(row["bank"], identity)
    n, turns = identity["episodes"], identity["turns"]
    target_totals = [sum(values[target] for values in counts) for target in range(3)]
    q, known, pairs = sum(target_totals), sum(target_totals[:2]), n//2
    expected = {"episodes": n, "turns_per_episode": turns, "query_total": q, "known_total": known,
                "final_total": n, "opposite_pair_total": sum(opposite), "ask_true": target_totals[2]}
    for key, value in expected.items():
        _integer(row[key], key)
        if row[key] != value:
            raise ValueError("canonical " + key + " denominator differs")
    if (row["control"] != control or row["teacher_used_for_policy"] is not False
            or row["free_running_replies"] is not True or row["decoder_prefix"] != "BOS only"):
        raise ValueError("foundation teacher-free reply boundary differs")
    final = row["final_pairs"]
    if type(final) is not dict or set(final) != {"correct", "total"} or final["total"] != pairs:
        raise ValueError("canonical final pair denominator differs")
    for key, correct, total in (
            ("query_accuracy", row["query_correct"], q), ("known_accuracy", row["known_correct"], known),
            ("final_accuracy", row["final_correct"], n), ("final_pair_accuracy", final["correct"], final["total"]),
            ("opposite_pair_accuracy", row["opposite_pair_correct"], row["opposite_pair_total"]),
            ("query_reply_accuracy", row["query_reply_correct"], q),
            ("final_reply_pair_accuracy", row["final_reply_pair_correct"], pairs)):
        _ratio(row[key], correct, total, key)
    confusion = row["confusion_matrix"]
    if type(confusion) is not list or len(confusion) != 4 or any(type(values) is not list or len(values) != 4 for values in confusion):
        raise ValueError("four-way confusion matrix required")
    for values in confusion:
        for value in values: _integer(value, "confusion count")
    if ([sum(values) for values in confusion] != target_totals + [0]
            or sum(confusion[target][target] for target in range(3)) != row["query_correct"]
            or confusion[0][0]+confusion[1][1] != row["known_correct"]):
        raise ValueError("confusion target/known/query counts differ")
    if type(row["per_target"]) is not dict or set(row["per_target"]) != {"0", "1", "2"}:
        raise ValueError("all canonical query-target cells required")
    for target in range(3):
        score = row["per_target"][str(target)]
        if score["total"] != target_totals[target] or score["correct"] != confusion[target][target]:
            raise ValueError("per-target confusion counts differ")
        _ratio(score["accuracy"], score["correct"], score["total"], "per-target")
    for key, value in (("ask_predicted", sum(values[2] for values in confusion)), ("ask_correct", confusion[2][2])):
        _integer(row[key], key)
        if row[key] != value:
            raise ValueError("ASK confusion counts differ")
    _ratio(row["ask_precision"], row["ask_correct"], row["ask_predicted"], "ASK precision")
    _ratio(row["ask_recall"], row["ask_correct"], row["ask_true"], "ASK recall")
    by_turn = row["by_turn"]
    if type(by_turn) is not list or len(by_turn) != turns:
        raise ValueError("per-turn counts missing")
    for index, score in enumerate(by_turn):
        for key in ("turn", "total", "correct", "opposite_pair_total", "opposite_pair_correct"):
            _integer(score[key], "per-turn " + key)
        total, pt, pc = sum(counts[index][:3]), opposite[index], score["opposite_pair_correct"]
        if score["turn"] != index or score["total"] != total or score["opposite_pair_total"] != pt:
            raise ValueError("canonical per-turn query/pair denominator differs")
        _ratio(score["accuracy"], score["correct"], total, "per-turn")
        if not max(0, score["correct"]-(total-pt)) <= pc <= min(pt, score["correct"]//2):
            raise ValueError("per-turn correct queries cannot yield these complete pairs")
    if (sum(score["correct"] for score in by_turn) != row["query_correct"]
            or sum(score["opposite_pair_correct"] for score in by_turn) != row["opposite_pair_correct"]
            or by_turn[-1]["correct"] != row["final_correct"]
            or by_turn[-1]["opposite_pair_correct"] != final["correct"]):
        raise ValueError("per-turn and global correct counts differ")
    _integer(row["reply_parseable_queries"], "parseable replies")
    if not row["query_reply_correct"] <= row["reply_parseable_queries"] <= q:
        raise ValueError("reply correctness/parseability counts differ")
    if not max(0, row["query_reply_correct"]-(q-pairs)) <= row["final_reply_pair_correct"] <= min(pairs, row["query_reply_correct"]//2):
        raise ValueError("complete reply pairs are inconsistent with correct replies")
    _finite(row["action_reply_agreement"], "action/reply agreement", 1.)
    agreement = round(row["action_reply_agreement"]*q)
    if agreement > row["reply_parseable_queries"] or not math.isclose(row["action_reply_agreement"], agreement/q, rel_tol=0., abs_tol=1e-7):
        raise ValueError("action/reply agreement has no possible query count")
    _finite(row["query_loss"], "query loss")
    _finite(row["brier_score"], "Brier score", 2.+1e-6)
    _finite(row["seconds"], "scoring time")


def validate_metrics(metrics, rows, config, role, control="normal"):
    """Authenticate canonical rows and their actual foundation metric identity.

    Scores must include free-running replies. All counts are checked without
    invoking a model. Sources/checkpoints remain the caller's separate boundary.
    """
    if role not in ("train_fit", "dev", "audit") or control not in ("normal", "blank", "reset"):
        raise ValueError("explicit scoring role/control required")
    config = config if isinstance(config, SequenceConfig) else SequenceConfig(**config)
    if (type(rows) is not dict or not rows or type(metrics) is not dict
            or set(metrics) != {"per_bank", *("macro_"+key for key in METRICS)}
            or type(metrics["per_bank"]) is not dict or set(rows) != set(metrics["per_bank"])):
        raise ValueError("exact named banks and evaluator macro fields required")
    try:
        json.dumps(metrics, allow_nan=False)
        for name, canonical in rows.items():
            identity, counts, opposite = _canonical(canonical, name, config, role)
            _validate_row(metrics["per_bank"][name], identity, counts, opposite, control)
        for key in METRICS:
            expected = sum(row[key] for row in metrics["per_bank"].values())/len(rows)
            actual = metrics["macro_"+key]
            _finite(actual, "macro "+key, 1.)
            if not math.isclose(actual, expected, rel_tol=0., abs_tol=1e-12):
                raise ValueError("equal-bank macro weighting differs")
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("malformed foundation metric evidence") from error
    return True


def _mean(values):
    known = [value for value in values if value is not None]
    return sum(known)/len(known) if known else None


def compact(metrics):
    """Group already validated scores; keep full counts at every length.

    Scalar macros equally weight measured cells; null ASK precision/recall cells
    are omitted only from those descriptive means. Pooled denominators remain
    visible and are not replaced by macro rates.
    """
    result = {"schema": SCHEMA, "version": VERSION,
              "macro": {key: metrics["macro_"+key] for key in METRICS},
              "per_panel_family_depth_operator": {}}
    for name, source in metrics["per_bank"].items():
        if source.get("bank", {}).get("version") != VERSION:
            raise ValueError("compact requires validated foundation metrics, preserving their real version")
        panel, family, depth, operator, turns = _cell(name)
        row = deepcopy(source)
        row["unsupported_ask_count"] = row["ask_predicted"]-row["ask_correct"]
        row["unsupported_ask_rate"] = row["unsupported_ask_count"]/row["known_total"]
        group = result["per_panel_family_depth_operator"].setdefault(panel, {}).setdefault(family, {}).setdefault(depth, {}).setdefault(operator, {"per_length": {}})
        if turns in group["per_length"]:
            raise ValueError("duplicate foundation summary cell")
        group["per_length"][turns] = row
    for families in result["per_panel_family_depth_operator"].values():
        for depths in families.values():
            for operators in depths.values():
                for group in operators.values():
                    rows = list(group["per_length"].values())
                    group["macro"] = {key: _mean([row[key] for row in rows]) for key in (*METRICS, *EXTRA_METRICS)}
                    group["pooled_counts"] = {
                        "final_pairs": {key: sum(row["final_pairs"][key] for row in rows) for key in ("correct", "total")},
                        "final_reply_pairs": {"correct": sum(row["final_reply_pair_correct"] for row in rows), "total": sum(row["final_pairs"]["total"] for row in rows)},
                        "known": {"correct": sum(row["known_correct"] for row in rows), "total": sum(row["known_total"] for row in rows)},
                        "query": {"correct": sum(row["query_correct"] for row in rows), "total": sum(row["query_total"] for row in rows)},
                        "unknown": {"correct": sum(row["ask_correct"] for row in rows), "total": sum(row["ask_true"] for row in rows)},
                        "unsupported_ask_count": sum(row["unsupported_ask_count"] for row in rows)}
    return result
