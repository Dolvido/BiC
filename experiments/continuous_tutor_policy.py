"""Fixed, JSON-persistable campaign decisions; not a learned curriculum policy.

Caller owns artifact authentication, durable commit, actual execution/deadlines,
and restoration of the unchanged parent recipe/full AdamW. No I/O or transport.
Branch envelopes contain status, parent_checkpoint_sha256, recipe_sha256,
cycle_spec, evaluation_inputs (role -> bank SHA), common_withdrawal_sha256,
actual_work (teaching/withdrawal phase counts), teaching_metrics (three roles),
and final_metrics (five roles). Metrics are native score_records dictionaries.
"""
from fractions import Fraction
import hashlib
import json
import math

SCHEMA = "bic-continuous-tutor-policy-v1"
FAMILIES = ("color", "count", "switch")
CORE_ROLES = ("dev", "train_fit", "retention")
ROLES = CORE_ROLES + ("transfer_original", "transfer_varied")
QUERY_METRICS = ("known_action", "known_reply", "unknown_action", "unknown_reply")
METRICS = ("anchor_pair_both",) + QUERY_METRICS


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _pin(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("explicit lowercase SHA-256 identity required")
    return value


def cycle_spec(campaign_seed, cycle_index):
    """Fresh detached spec: same schedule/seeds for both branches, no allocations."""
    if any(type(v) is not int or v < 0 for v in (campaign_seed, cycle_index)):
        raise ValueError("nonnegative integer campaign seed and cycle index required")
    seeds = {phase: int(_digest([SCHEMA, campaign_seed, cycle_index, phase])[:16], 16) % (2**63)
             for phase in ("teaching", "withdrawal", "evaluation")}
    return dict(schema=SCHEMA, campaign_seed=campaign_seed, cycle_index=cycle_index,
        seeds=seeds, families=list(FAMILIES), micro_batch_size=32,
        slots=[dict(depth=d, turns=t) for _ in range(6) for d in range(6) for t in (8, 10, 12)],
        teaching_updates=108, withdrawal_updates=108, common_withdrawal=True,
        architecture="sequence-shared-entity-state-v1", learning_rate=0.0003,
        auxiliary_weight=0.3, objective_id="unchanged-sequence-objective-three-family-mean-v1",
        config=dict(width=192, layers=4, heads=4, feedforward=768, max_positions=1024,
                    max_turns=12, max_input_bytes=128, max_output_bytes=32),
        restoration="unchanged public SharedStateContinuation; preserve parent recipe and full AdamW")


def stage_decision(*, budget_seconds, elapsed_seconds, required_seconds, final_reserve_seconds):
    """Admission only: caller supplies the entire stage bound and enforces time.

    Decimal representations determine exact boundaries. A final-save stage may
    explicitly use reserve zero; other stages must retain the declared reserve.
    """
    values = (budget_seconds, elapsed_seconds, required_seconds, final_reserve_seconds)
    try:
        valid = all(type(v) in (int, float) and math.isfinite(v) for v in values)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("finite numeric wall times required")
    budget, elapsed, required, reserve = map(lambda v: Fraction(str(v)), values)
    if budget <= 0 or elapsed < 0 or required <= 0 or not 0 <= reserve <= budget:
        raise ValueError("positive budget/stage bound and valid elapsed/reserve required")
    remaining = max(Fraction(0), budget - elapsed)
    allowed = remaining >= required + reserve
    return dict(schema=SCHEMA, start=allowed, budget_seconds=budget_seconds,
        elapsed_seconds=elapsed_seconds, required_seconds=required_seconds,
        final_reserve_seconds=final_reserve_seconds, remaining_seconds=float(remaining),
        reason="stage_and_reserve_fit" if allowed else "insufficient_remaining_wall_budget")


def _metrics(value, roles):
    if type(value) is not dict or set(value) != set(roles):
        raise ValueError("complete fixed native evaluation roles required")
    for role in roles:
        endpoint = value[role]
        if set(endpoint["by_family"]) != set(FAMILIES):
            raise ValueError("complete three-family native metrics required")
        for group in [endpoint["overall"]] + [endpoint["by_family"][f] for f in FAMILIES]:
            for metric in METRICS:
                pair = group["counts"][metric]
                if (type(pair["count"]) is not int or type(pair["total"]) is not int
                        or not 0 <= pair["count"] <= pair["total"] or pair["total"] <= 0):
                    raise ValueError("nonempty exact integer counts required")
            for kind in ("known", "unknown"):
                if group["counts"][kind+"_action"]["total"] != group["counts"][kind+"_reply"]["total"]:
                    raise ValueError("native modalities must use the same query denominator")
        for metric in METRICS:
            for key in ("count", "total"):
                if endpoint["overall"]["counts"][metric][key] != sum(
                        endpoint["by_family"][f]["counts"][metric][key] for f in FAMILIES):
                    raise ValueError("overall counts must equal all-family sums")


def _branch(value, parent):
    if value["status"] != "completed" or _pin(value["parent_checkpoint_sha256"]) != parent:
        raise ValueError("completed branch from the caller-pinned same parent required")
    _pin(value["recipe_sha256"]); _pin(value["common_withdrawal_sha256"])
    spec = value["cycle_spec"]
    if _encoded(spec) != _encoded(cycle_spec(spec["campaign_seed"], spec["cycle_index"])):
        raise ValueError("unchanged whole-curriculum cycle specification required")
    if set(value["evaluation_inputs"]) != set(ROLES):
        raise ValueError("all evaluation bank pins required")
    for pin in value["evaluation_inputs"].values():
        _pin(pin)
    expected = dict(synchronized_optimizer_updates=108, retained_episodes=10368,
        unknown_optimizer_outcomes=0, family_episode_exposures={f: 3456 for f in FAMILIES})
    if set(value["actual_work"]) != {"teaching", "withdrawal"} or any(
            _encoded(work) != _encoded(expected) for work in value["actual_work"].values()):
        raise ValueError("both complete matched actual phase exposures and zero unknown updates required")
    _metrics(value["teaching_metrics"], CORE_ROLES)
    _metrics(value["final_metrics"], ROLES)


def select_continuation(procedural, tutor, *, expected_parent_identity):
    """Return a branch choice, never promotion; authenticate inputs before calling.

    expected_parent_identity is the exact checkpoint-byte SHA, not a label.
    recipe_sha256 identifies the unchanged inherited recipe, equal in both arms.
    Bank pins cover every endpoint; common_withdrawal_sha256 binds ordered lessons.
    Incomplete/malformed comparisons raise rather than silently selecting a branch.
    """
    parent = _pin(expected_parent_identity)
    try:
        for branch in (procedural, tutor):
            _branch(branch, parent)
        for key in ("recipe_sha256", "cycle_spec", "evaluation_inputs", "common_withdrawal_sha256", "actual_work"):
            if _encoded(procedural[key]) != _encoded(tutor[key]):
                raise ValueError("paired branch identity or actual exposure mismatch: "+key)
        checks, differences = {}, {}

        def check(key, left, right, role, metric, minimum, family=None):
            def pair(values):
                group = values[role]["overall"] if family is None else values[role]["by_family"][family]
                return group["counts"][metric]
            a, b = pair(left), pair(right)
            if a["total"] != b["total"]:
                raise ValueError("matched evaluation denominators required")
            change = Fraction(b["count"] - a["count"], a["total"])
            checks[key] = change >= minimum
            differences[key] = dict(before=a["count"], after=b["count"], total=a["total"],
                change_numerator=change.numerator, change_denominator=change.denominator,
                minimum_numerator=minimum.numerator, minimum_denominator=minimum.denominator)

        left, right = procedural["final_metrics"], tutor["final_metrics"]
        for role in ROLES:
            check(role+"/joint", left, right, role, "anchor_pair_both",
                  Fraction(1, 20) if role == "dev" else Fraction(0))
            for family in FAMILIES:
                if role == "dev":
                    check(role+"/"+family+"/joint", left, right, role, "anchor_pair_both", Fraction(0), family)
                for metric in QUERY_METRICS:
                    key = role+"/"+family+"/"+metric
                    check(key, left, right, role, metric, -Fraction(1, 20), family)
        for role in CORE_ROLES:
            for family in FAMILIES:
                for metric in QUERY_METRICS:
                    check("withdrawal/"+role+"/"+family+"/"+metric,
                        tutor["teaching_metrics"], right, role, metric, -Fraction(1, 20), family)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("malformed branch envelope or native metrics") from error
    selected = "tutor" if all(checks.values()) else "procedural"
    return dict(schema=SCHEMA, selected_arm=selected, tutor_criteria_passed=all(checks.values()),
        checks=checks, differences=differences, parent_checkpoint_sha256=parent,
        cycle_spec_sha256=_digest(procedural["cycle_spec"]),
        inputs_sha256=_digest(dict(procedural=procedural, tutor=tutor)),
        automatic_promotion=False, learned_policy=False,
        scope="Fixed research continuation choice from authenticated native counts; no broader independence claim.")


def protect_retention(choice, *, candidates, current_parent, reference_parent):
    """Keep the current parent if the chosen branch forgets either reference.

    The caller authenticates the fixed campaign-start reference and current
    parent metrics on the same five banks. This does not undo physical work,
    select another candidate, or assert improvement when the parent is retained.
    """
    try:
        if (type(choice) is not dict or choice.get("schema") != SCHEMA
                or choice.get("selected_arm") not in ("procedural", "tutor")
                or type(candidates) is not dict or set(candidates) != {"procedural", "tutor"}):
            raise ValueError("a paired policy choice and both native candidate endpoints required")
        for value in (*candidates.values(), current_parent, reference_parent):
            _metrics(value, ROLES)
        selected = choice["selected_arm"]
        candidate = candidates[selected]
        checks, differences = {}, {}
        minimum = -Fraction(1, 20)

        def check(reference_name, reference, role, metric, family=None):
            def pair(endpoint):
                group = endpoint[role]["overall"] if family is None else endpoint[role]["by_family"][family]
                return group["counts"][metric]
            before, after = pair(reference), pair(candidate)
            if before["total"] != after["total"]:
                raise ValueError("retention guard requires unchanged evaluation denominators")
            change = Fraction(after["count"]-before["count"], before["total"])
            key = "/".join((reference_name, role, family or "overall", metric))
            checks[key] = change >= minimum
            differences[key] = dict(before=before["count"], after=after["count"], total=before["total"],
                change_numerator=change.numerator, change_denominator=change.denominator,
                minimum_numerator=minimum.numerator, minimum_denominator=minimum.denominator)

        for name, reference in (("current_parent", current_parent), ("campaign_start", reference_parent)):
            check(name, reference, "retention", "anchor_pair_both")
            for family in FAMILIES:
                check(name, reference, "retention", "anchor_pair_both", family)
                for role in ROLES:
                    for metric in QUERY_METRICS:
                        check(name, reference, role, metric, family)
        result = json.loads(_encoded(choice))
        failed = [key for key, passed in checks.items() if not passed]
        result["retention_guard"] = dict(passed=not failed, proposed_arm=selected,
            checks=checks, differences=differences, failed_checks=failed,
            inputs_sha256=_digest(dict(candidates=candidates, current_parent=current_parent,
                                      reference_parent=reference_parent)),
            reason="retention_limits_satisfied" if not failed else "retain_current_parent_after_retention_loss")
        if failed:
            result["selected_arm"] = "parent"
        return result
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("malformed retention choice or native endpoints") from error
