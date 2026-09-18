"""Pure ENGINEERED allocation policy for development-only English learning.

This is an observational heuristic, not learned self-direction or promotion.
Progress is measured per shared-window work second, not causal family efficiency.
All families retain a caller-declared microbatch floor in every completed window.
Best-observed cell references keep unresolved regressions visible; they are not
mastery thresholds. No model, tutor, audit bank, clock or random generator is used.

Transitions return isolated JSON trees. ``plan`` never consumes work; the owning
transaction must ``advance`` by the backend's *retained* update delta. It must
also bind these counters to its learner snapshot. Only the two latest complete
observations and compact cell references are retained here; an external ledger
can preserve earlier full observations. Each update contains three microbatches.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math


SCHEMA = "bic-engineered-english-allocation-v1"
ROLES = ("development", "retention")
FIELDS = ("paired_action", "paired_reply", "known", "unsupported_ask")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name, minimum=0., maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < minimum or maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its finite numeric range")


def _copy(value):
    def check(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) in (list, tuple):
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise ValueError("allocation evidence/state must be finite JSON values with string keys")
    check(value)
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _same(left, right):
    # JSON distinguishes booleans from count integers; Python dict equality does not.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class AllocationConfig:
    mode: str = "joint"
    families: tuple[str, ...] = ("color", "count", "switch")
    window_updates: int = 16
    min_family_microbatches: int = 8
    regression_tolerance: float = 0.0

    def __post_init__(self):
        if self.mode not in ("joint", "progress"):
            raise ValueError("allocation mode must be joint or progress")
        if (type(self.families) is not tuple or len(self.families) != 3
                or any(type(name) is not str or not name for name in self.families)
                or len(set(self.families)) != 3):
            raise ValueError("three distinct ordered family names required")
        _integer(self.window_updates, "window_updates", 1)
        _integer(self.min_family_microbatches, "min_family_microbatches", 1)
        if self.window_updates > 4096 or self.min_family_microbatches > self.window_updates:
            raise ValueError("window exceeds backend capacity or family floor exceeds budget")
        _number(self.regression_tolerance, "regression_tolerance", maximum=1.)


def _config(value):
    if type(value) is AllocationConfig:
        return value
    if type(value) is not dict or set(value) != set(asdict(AllocationConfig())):
        raise ValueError("allocation config fields differ")
    return AllocationConfig(**{**value, "families": tuple(value["families"])})


def _specs(config, bank_specs):
    specs = _copy(bank_specs)
    if type(specs) is not dict or set(specs) != set(ROLES):
        raise ValueError("complete development and retention bank roles required; no audit")
    common_config = None
    for role, banks in specs.items():
        if type(banks) is not dict or not banks:
            raise ValueError("named banks required for both roles")
        families = set()
        for name, spec in banks.items():
            if not name or type(spec) is not dict or set(spec) != {"family", "identity"}:
                raise ValueError("bank specification needs family and exact identity")
            if spec["family"] not in config.families:
                raise ValueError("bank family is not admitted")
            families.add(spec["family"])
            identity = spec["identity"]
            if (type(identity) is not dict or set(identity) !=
                    {"sha256", "version", "episodes", "turns", "role", "config"}
                    or identity["role"] != "dev"):
                raise ValueError("prepared bank identity must have development admission")
            digest = identity["sha256"]
            if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("bank SHA256 differs")
            if type(identity["version"]) is not str or not identity["version"]:
                raise ValueError("bank version required")
            _integer(identity["episodes"], "bank episodes", 2)
            _integer(identity["turns"], "bank turns", 2)
            if identity["episodes"] % 2:
                raise ValueError("complete paired bank required")
            model_config = identity["config"]
            if type(model_config) is not dict or not model_config:
                raise ValueError("bank model configuration required")
            for value in model_config.values():
                _integer(value, "model configuration", 1)
            if common_config is None:
                common_config = model_config
            if model_config != common_config:
                raise ValueError("bank model configurations differ")
        if families != set(config.families):
            raise ValueError(f"{role} must cover every admitted family")
    return specs


def _rate(value, correct, total, name):
    _integer(total, name + " total")
    _integer(correct, name + " correct")
    if correct > total:
        raise ValueError(name + " correct exceeds total")
    if total == 0:
        if value is not None:
            raise ValueError(name + " empty denominator must have null rate")
    else:
        _number(value, name, maximum=1.)
        if not math.isclose(value, correct / total, rel_tol=0., abs_tol=1e-7):
            raise ValueError(name + " rate differs from counts")


def _vector(row, identity):
    """Validate count consistency, retaining the backend's complete progress vector.

    The controller separately authenticates denominators against canonical rows;
    this pure policy never parses English or calls an oracle.
    """
    if (not _same(row["bank"], identity) or row["control"] != "normal"
            or row["free_running_replies"] is not True or row["teacher_used_for_policy"] is not False
            or row["decoder_prefix"] != "BOS only"):
        raise ValueError("bank identity, role/config or teacher-free evidence boundary differs")
    n, turns = identity["episodes"], identity["turns"]
    for name in ("episodes", "turns_per_episode", "query_correct", "query_total", "known_correct", "known_total",
                 "final_correct", "final_total", "opposite_pair_correct", "opposite_pair_total", "ask_true",
                 "ask_predicted", "ask_correct", "query_reply_correct", "final_reply_pair_correct", "reply_parseable_queries"):
        _integer(row[name], name)
    if row["episodes"] != n or row["turns_per_episode"] != turns or row["final_total"] != n:
        raise ValueError("episode/turn denominator differs")
    q, known, pairs = row["query_total"], row["known_total"], row["final_pairs"]
    if not 0 < known <= q <= n * turns or q != known + row["ask_true"]:
        raise ValueError("query/known denominator differs")
    if type(pairs) is not dict or set(pairs) != {"correct", "total"} or pairs["total"] != n // 2:
        raise ValueError("final paired denominator differs")
    checks = (("query_accuracy", row["query_correct"], q), ("known_accuracy", row["known_correct"], known),
              ("final_accuracy", row["final_correct"], n), ("final_pair_accuracy", pairs["correct"], pairs["total"]),
              ("opposite_pair_accuracy", row["opposite_pair_correct"], row["opposite_pair_total"]),
              ("query_reply_accuracy", row["query_reply_correct"], q),
              ("final_reply_pair_accuracy", row["final_reply_pair_correct"], pairs["total"]),
              ("ask_precision", row["ask_correct"], row["ask_predicted"]),
              ("ask_recall", row["ask_correct"], row["ask_true"]))
    for name, correct, total in checks:
        _rate(row[name], correct, total, name)
    confusion = row["confusion_matrix"]
    if type(confusion) is not list or len(confusion) != 4 or any(type(r) is not list or len(r) != 4 for r in confusion):
        raise ValueError("query confusion matrix shape differs")
    for values in confusion:
        for value in values:
            _integer(value, "confusion count")
    if (sum(map(sum, confusion)) != q or sum(confusion[3]) != 0
            or sum(confusion[0]) + sum(confusion[1]) != known
            or sum(confusion[i][i] for i in range(3)) != row["query_correct"]
            or confusion[0][0] + confusion[1][1] != row["known_correct"]
            or sum(confusion[2]) != row["ask_true"] or confusion[2][2] != row["ask_correct"]
            or sum(values[2] for values in confusion) != row["ask_predicted"]):
        raise ValueError("query confusion matrix/counts differ")
    if set(row["per_target"]) != {"0", "1", "2"}:
        raise ValueError("per-target evidence missing")
    for i in range(3):
        target = row["per_target"][str(i)]
        if target["total"] != sum(confusion[i]) or target["correct"] != confusion[i][i]:
            raise ValueError("per-target counts differ")
        _rate(target["accuracy"], target["correct"], target["total"], "per-target")
    by_turn = row["by_turn"]
    if type(by_turn) is not list or len(by_turn) != turns:
        raise ValueError("per-turn evidence missing")
    for index, turn in enumerate(by_turn):
        _integer(turn["turn"], "turn index")
        if turn["turn"] != index:
            raise ValueError("turn index differs")
        _rate(turn["accuracy"], turn["correct"], turn["total"], "per-turn query")
        pc, pt = turn["opposite_pair_correct"], turn["opposite_pair_total"]
        _integer(pc, "turn pair correct"); _integer(pt, "turn pair total")
        if turn["total"] > n or not pc <= pt <= n // 2 or 2 * pc > turn["correct"]:
            raise ValueError("per-turn pair denominator differs")
    if (sum(t["total"] for t in by_turn) != q or sum(t["correct"] for t in by_turn) != row["query_correct"]
            or sum(t["opposite_pair_total"] for t in by_turn) != row["opposite_pair_total"]
            or sum(t["opposite_pair_correct"] for t in by_turn) != row["opposite_pair_correct"]
            or by_turn[-1]["total"] != n or by_turn[-1]["correct"] != row["final_correct"]
            or by_turn[-1]["opposite_pair_total"] != pairs["total"]
            or by_turn[-1]["opposite_pair_correct"] != pairs["correct"]):
        raise ValueError("per-turn/global evidence does not reconcile")
    false_ask = row["ask_predicted"] - row["ask_correct"]
    if not 0 <= false_ask <= known or row["reply_parseable_queries"] > q:
        raise ValueError("unsupported ASK or parsed-reply count differs")
    _number(row["action_reply_agreement"], "action/reply agreement", maximum=1.)
    for name in ("query_loss", "brier_score", "seconds"):
        _number(row[name], name)
    return {"final_pairs": pairs, "final_pair_accuracy": row["final_pair_accuracy"],
            "opposite_pairs": {"correct": row["opposite_pair_correct"], "total": row["opposite_pair_total"]},
            "known_queries": {"correct": row["known_correct"], "total": known, "accuracy": row["known_accuracy"]},
            "unknown_queries": {"correct": row["ask_correct"], "total": row["ask_true"],
                                "recall": row["ask_recall"], "precision": row["ask_precision"]},
            "unsupported_ask": {"count": false_ask, "known_total": known, "rate": false_ask / known},
            "query_replies": {"correct": row["query_reply_correct"], "total": q, "accuracy": row["query_reply_accuracy"]},
            "final_reply_pairs": {"correct": row["final_reply_pair_correct"], "total": pairs["total"],
                                  "accuracy": row["final_reply_pair_accuracy"]},
            "action_reply_agreement": row["action_reply_agreement"],
            "parseable_query_replies": row["reply_parseable_queries"], "query_loss": row["query_loss"],
            "brier_score": row["brier_score"], "by_turn": by_turn}


def _evidence(evidence, specs):
    data = _copy(evidence)
    if type(data) is not dict or set(data) != set(ROLES):
        raise ValueError("complete development/retention evidence required; no audit")
    vectors = {}
    for role in ROLES:
        response = data[role]
        if (response["role"] != role or response["automatic_promotion"] is not False
                or set(response["per_bank"]) != set(specs[role]) or set(response["progress"]) != set(specs[role])):
            raise ValueError("evidence role/bank set differs")
        _number(response["wall_seconds"], "evaluation wall_seconds")
        vectors[role] = {name: _vector(response["per_bank"][name], spec["identity"])
                         for name, spec in specs[role].items()}
        if not _same(response["progress"], vectors[role]):
            raise ValueError("backend progress vector differs from metric evidence")
    return data, _copy(vectors)


def _rates(vector):
    return dict(zip(FIELDS, (vector["final_pair_accuracy"], vector["final_reply_pairs"]["accuracy"],
                            vector["known_queries"]["accuracy"], vector["unsupported_ask"]["rate"])))


def _denominators(vector):
    return (vector["final_pairs"]["total"], vector["opposite_pairs"]["total"], vector["known_queries"]["total"],
            vector["unknown_queries"]["total"], vector["query_replies"]["total"],
            tuple((turn["total"], turn["opposite_pair_total"]) for turn in vector["by_turn"]))


def _counts(schedule, families):
    return {family: sum(row.count(family) for row in schedule) for family in families}


def _decision(config, specs, latest, previous, reference):
    scores = dict.fromkeys(config.families, 0.)
    changes, alarms, observed = {}, [], dict.fromkeys(config.families, 0)
    for role in ROLES:
        changes[role] = {}
        for name, vector in latest["vectors"][role].items():
            family = specs[role][name]["family"]
            current = _rates(vector)
            delta = {key: 0. if previous is None else current[key] - _rates(previous["vectors"][role][name])[key]
                     for key in FIELDS}
            changes[role][name] = delta
            if role == "development" and previous is not None:
                scores[family] += min(delta[key] for key in FIELDS[:3]) - max(0., delta["unsupported_ask"])
                observed[family] += 1
            for key in FIELDS:
                best = reference[role][name][key]
                shortfall = current[key] - best if key == "unsupported_ask" else best - current[key]
                if shortfall > config.regression_tolerance:
                    alarms.append({"role": role, "bank": name, "family": family, "metric": key,
                                   "observed": current[key], "reference": best, "shortfall": shortfall})
    if previous is not None:
        scores = {family: scores[family] / observed[family] / latest["window_seconds"] for family in config.families}
    selected = list(config.families)
    if previous is None:
        reason = "joint_cold_start"
    elif config.mode == "joint":
        reason = "fixed_joint"
    elif alarms:
        worst = {family: max((a["shortfall"] for a in alarms if a["family"] == family), default=0.)
                 for family in config.families}
        selected = sorted((family for family in config.families if worst[family] > 0.),
                          key=lambda family: (-worst[family], config.families.index(family)))
        reason = "retention_alarm"
    elif max(scores.values()) > 0.:
        selected = [family for family in config.families if scores[family] == max(scores.values())]
        reason = "observed_progress"
    else:
        reason = "joint_no_positive_progress"
    slots = list(config.families) * config.min_family_microbatches
    slots += [selected[i % len(selected)] for i in range(3 * config.window_updates - len(slots))]
    schedule = [slots[i:i + 3] for i in range(0, len(slots), 3)]
    return schedule, {"policy": "ENGINEERED", "reason": reason, "selected_families": selected,
        "progress_per_shared_window_second": scores, "cell_changes": changes, "alarms": alarms,
        "full_window_family_microbatches": _counts(schedule, config.families), "automatic_promotion": False}


def initialize(config, bank_specs):
    """Bind caller-declared configuration and exact development bank identities."""
    config = _config(config)
    return {"schema": SCHEMA, "config": _copy(asdict(config)), "bank_specs": _specs(config, bank_specs),
            "window_index": 0, "cursor": 0, "total_updates": 0,
            "family_microbatches": dict.fromkeys(config.families, 0), "schedule": [], "decision": None,
            "latest": None, "previous": None, "reference": {}}


def _validate(state):
    state = _copy(state)
    if type(state) is not dict or state.get("schema") != SCHEMA:
        raise ValueError("allocation state schema differs")
    config = _config(state["config"])
    specs = _specs(config, state["bank_specs"])
    if set(state) != set(initialize(config, specs)):
        raise ValueError("allocation state fields differ")
    for name in ("window_index", "cursor", "total_updates"):
        _integer(state[name], name)
    if (state["cursor"] > config.window_updates
            or state["total_updates"] != state["window_index"] * config.window_updates + state["cursor"]):
        raise ValueError("window cursor/update accounting differs")
    counts = state["family_microbatches"]
    if type(counts) is not dict or set(counts) != set(config.families):
        raise ValueError("family accounting differs")
    for count in counts.values():
        _integer(count, "family microbatches")
    if sum(counts.values()) != 3 * state["total_updates"]:
        raise ValueError("family/update accounting differs")
    if state["latest"] is None:
        if state != initialize(config, specs):
            raise ValueError("unobserved state contains consumed work")
        return state
    for key in ("latest", "previous"):
        observation = state[key]
        if observation is None:
            continue
        if type(observation) is not dict or set(observation) != {"window_index", "window_seconds", "evidence", "vectors"}:
            raise ValueError("observation state fields differ")
        _integer(observation["window_index"], "observation window")
        _number(observation["window_seconds"], "observed work seconds")
        if (observation["window_seconds"] == 0) != (observation["window_index"] == 0):
            raise ValueError("only initial observation has zero work seconds")
        _, vectors = _evidence(observation["evidence"], specs)
        if not _same(observation["vectors"], vectors):
            raise ValueError("stored vectors differ from evidence")
    latest, previous = state["latest"], state["previous"]
    if (latest["window_index"] != state["window_index"] or
            (previous is None) != (state["window_index"] == 0) or
            previous is not None and previous["window_index"] != state["window_index"] - 1):
        raise ValueError("observation sequence differs")
    reference = state["reference"]
    if type(reference) is not dict or set(reference) != set(ROLES):
        raise ValueError("cell references missing")
    for role in ROLES:
        if set(reference[role]) != set(specs[role]):
            raise ValueError("reference bank identity differs")
        for name, values in reference[role].items():
            if set(values) != set(FIELDS):
                raise ValueError("reference vector fields differ")
            for metric, value in values.items():
                _number(value, "reference rate", maximum=1.)
            for observation in (latest, previous):
                if observation is None:
                    continue
                rates = _rates(observation["vectors"][role][name])
                if any(values[k] > rates[k] if k == "unsupported_ask" else values[k] < rates[k] for k in FIELDS):
                    raise ValueError("reference is inconsistent with observed cells")
            if previous is not None and _denominators(latest["vectors"][role][name]) != _denominators(previous["vectors"][role][name]):
                raise ValueError("observation bank denominators changed")
    schedule, decision = _decision(config, specs, latest, previous, reference)
    if not _same(state["schedule"], schedule) or not _same(state["decision"], decision):
        raise ValueError("stored schedule/decision differs from engineered policy")
    current = _counts(schedule[:state["cursor"]], config.families)
    completed = {family: counts[family] - current[family] for family in config.families}
    if (any(value < state["window_index"] * config.min_family_microbatches for value in completed.values())
            or state["window_index"] == 0 and any(completed.values())):
        raise ValueError("completed window family floor accounting differs")
    return state


def restore(payload, config, bank_specs):
    """Reject changed identities/configuration before returning a detached tree."""
    try:
        state = _validate(payload)
        contract = initialize(config, bank_specs)
        if state["config"] != contract["config"] or state["bank_specs"] != contract["bank_specs"]:
            raise ValueError("allocation resume contract differs")
        return state
    except (KeyError, TypeError, AttributeError, ZeroDivisionError) as error:
        raise ValueError("malformed allocation state") from error


def snapshot(state):
    try:
        return restore(state, state["config"], state["bank_specs"])
    except (KeyError, TypeError) as error:
        raise ValueError("malformed allocation state") from error


def observe(state, evidence_by_role, *, window_seconds):
    """Close a full consumed window (or admit initial evidence), then plan anew.

    ``window_seconds`` is caller-measured completed-window work time. It is a
    shared cost, not per-family attribution. Evaluation wall times remain in the
    evidence separately; the caller must declare its chosen accounting scope.
    """
    state = snapshot(state)
    config = _config(state["config"])
    _number(window_seconds, "window_seconds")
    initial = state["latest"] is None
    if initial:
        if window_seconds != 0:
            raise ValueError("initial evidence requires zero work seconds")
    elif state["cursor"] != config.window_updates or window_seconds <= 0:
        raise ValueError("observe requires a complete window and positive measured cost")
    try:
        evidence, vectors = _evidence(evidence_by_role, state["bank_specs"])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("incomplete or malformed development evidence") from error
    previous = state["latest"]
    if previous is not None:
        for role in ROLES:
            for name in vectors[role]:
                if _denominators(vectors[role][name]) != _denominators(previous["vectors"][role][name]):
                    raise ValueError("observation bank denominators changed")
        state["window_index"] += 1
    reference = {}
    for role in ROLES:
        reference[role] = {}
        for name, vector in vectors[role].items():
            rates = _rates(vector)
            old = state["reference"].get(role, {}).get(name, rates)
            reference[role][name] = {key: (min if key == "unsupported_ask" else max)(rates[key], old[key]) for key in FIELDS}
    latest = {"window_index": state["window_index"], "window_seconds": window_seconds,
              "evidence": evidence, "vectors": vectors}
    schedule, decision = _decision(config, state["bank_specs"], latest, previous, reference)
    state.update(cursor=0, previous=previous, latest=latest, reference=reference, schedule=schedule, decision=decision)
    return snapshot(state)


def status(state):
    state = snapshot(state)
    config = _config(state["config"])
    current = _counts(state["schedule"][:state["cursor"]], config.families)
    return {"has_baseline": state["latest"] is not None, "window_index": state["window_index"],
            "cursor": state["cursor"], "window_updates": config.window_updates,
            "window_complete": state["latest"] is not None and state["cursor"] == config.window_updates,
            "family_microbatches": _copy(state["family_microbatches"]), "total_updates": state["total_updates"],
            "coverage_debt": {family: max(0, config.min_family_microbatches - current[family]) for family in config.families}}


def plan(state, max_updates=None):
    state = snapshot(state)
    if state["latest"] is None:
        raise ValueError("initial development/retention evidence required before planning")
    if max_updates is not None:
        _integer(max_updates, "max_updates")
    remaining = state["schedule"][state["cursor"]:]
    return {"schedule": remaining if max_updates is None else remaining[:max_updates],
            "full_schedule": state["schedule"], "decision": state["decision"],
            "window_index": state["window_index"], "cursor": state["cursor"]}


def advance(state, updates):
    """Account only complete retained backend updates; never reserve future work."""
    state = snapshot(state)
    _integer(updates, "retained updates")
    if state["latest"] is None or state["cursor"] + updates > len(state["schedule"]):
        raise ValueError("retained updates exceed the observed window plan")
    consumed = state["schedule"][state["cursor"]:state["cursor"] + updates]
    counts = _counts(consumed, _config(state["config"]).families)
    state["family_microbatches"] = {family: value + counts[family] for family, value in state["family_microbatches"].items()}
    state["cursor"] += updates
    state["total_updates"] += updates
    return snapshot(state)
