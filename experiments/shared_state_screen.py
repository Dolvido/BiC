"""Pure final-endpoint screen fixed before the shared-state pilot scores."""
from fractions import Fraction


def compare(baseline, candidate):
    """Accept role -> score_records metrics; return exact count-based criteria."""
    checks, deltas = {}, {}

    def difference(role, metric, family=None):
        group = "overall" if family is None else "by_family"
        left, right = baseline[role][group], candidate[role][group]
        if family is not None:
            left, right = left[family], right[family]
        a, b = left["counts"][metric], right["counts"][metric]
        if a["total"] != b["total"] or a["total"] <= 0:
            raise ValueError("matched nonempty evaluation denominators required")
        change = Fraction(b["count"] - a["count"], a["total"])
        key = "/".join((role, family or "overall", metric))
        deltas[key] = dict(baseline=a["count"], candidate=b["count"],
                           total=a["total"], percentage_points=float(100 * change))
        return change

    checks["fresh_joint_gain_at_least_5_points"] = difference("dev", "anchor_pair_both") >= Fraction(1, 20)
    checks["training_fit_joint_improves"] = difference("train_fit", "anchor_pair_both") > 0
    checks["retention_joint_loss_at_most_5_points"] = difference("retention", "anchor_pair_both") >= -Fraction(1, 20)
    families = ("color", "count", "switch")
    for role in ("dev", "train_fit", "retention"):
        if set(baseline[role]["by_family"]) != set(families) or set(candidate[role]["by_family"]) != set(families):
            raise ValueError("complete three-family endpoints required")
    for family in families:
        checks[family + "_joint_strictly_improves"] = difference("dev", "anchor_pair_both", family) > 0
        for metric in ("known_action", "known_reply", "unknown_action", "unknown_reply"):
            checks[family + "_" + metric + "_loss_at_most_5_points"] = difference("dev", metric, family) >= -Fraction(1, 20)
    return dict(passed=all(checks.values()), checks=checks, differences=deltas,
                scope="Single-seed exploration only; positive screen requires replication and matched-compute comparison.",
                automatic_promotion=False)
