"""Pure prospective-screen arithmetic: synthetic counts, no lesson/model work."""
from copy import deepcopy
import unittest

from experiments import foundation_layout_study as study


WORK = dict(constructed_resultsets=0, synthetic_panel_summaries=0,
    synthetic_family_summaries=0, mutated_case_variants=0, summarize_attempts=0,
    summarize_completions=0, expected_denominator_rejections=0)


def _ratio(count, total):
    return dict(count=count, total=total, rate=count/total if total else None)


def _family(pair_score, varied):
    WORK["synthetic_family_summaries"] += 1
    other, unknown = (60, 160) if varied else (50, 150)
    known = 2*pair_score+(100-pair_score)//2+other
    counts = {name: _ratio(count, total) for name, count, total in (
        ("query_action", known+unknown, 500), ("query_reply", known+unknown, 500),
        ("known_action", known, 300), ("known_reply", known, 300),
        ("other_known_action", other, 100), ("other_known_reply", other, 100),
        ("unknown_action", unknown, 200), ("unknown_reply", unknown, 200),
        ("action_reply_agreement", 325 if varied else 300, 500), ("reply_parseable", 500, 500),
        ("known_unsupported_ask", 20, 300), ("known_reply_unsupported_ask", 20, 300),
        ("anchor_pair_action", pair_score, 100), ("anchor_pair_reply", pair_score, 100),
        ("anchor_pair_both", pair_score, 100))}
    return dict(episodes=200, pairs=100, query_turns=500, counts=counts)


def _refresh(panel):
    """Build independent pooled counts directly from the three family records."""
    families = panel["metrics"]["by_family"]
    first = next(iter(families.values()))
    total = {key: sum(value[key] for value in families.values()) for key in ("episodes", "pairs", "query_turns")}
    total["counts"] = {key: _ratio(sum(value["counts"][key]["count"] for value in families.values()),
        sum(value["counts"][key]["total"] for value in families.values())) for key in first["counts"]}
    panel["metrics"]["overall"] = total


def _passing_results():
    WORK["constructed_resultsets"] += 1
    results = {}
    schedules = {"original": {144: 20, 288: 25, 432: 30, 576: 35, 720: 40, 864: 40},
                 "varied": {144: 22, 288: 30, 432: 40, 576: 45, 720: 50, 864: 50}}
    for spec in study.evaluation_schedule():
        role_maps = {}
        for role in spec["roles"]:
            role_maps[role] = {}
            for view in study.ARMS:
                role_maps[role][view] = {}
                for name in (("fit",) if role == "train_fit" else ("fresh", "composed")):
                    score = (10 if spec["arm"] == "shared" else 20 if spec["control"] != "normal"
                             else schedules[spec["arm"]][spec["step"]])
                    panel = dict(metrics=dict(by_family={family: _family(score, spec["arm"] == "varied")
                                                        for family in study.FAMILIES}))
                    _refresh(panel)
                    role_maps[role][view][name] = panel
                    WORK["synthetic_panel_summaries"] += 1
        results[spec["id"]] = dict(by_role=role_maps)
    return results


def _panel(results, arm, role, view, panel, *, step=864, control="normal"):
    return results[f"{arm}-{step:06d}-{control}"]["by_role"][role][view][panel]


def _change(panel, family, metric, count, total=None):
    counts = panel["metrics"]["by_family"][family]["counts"]
    old = counts[metric]
    counts[metric] = _ratio(count, old["total"] if total is None else total)
    for suffix in ("action", "reply"):
        if metric == "anchor_pair_"+suffix:
            known = 2*count+(100-count)//2+counts["other_known_"+suffix]["count"]
            counts["known_"+suffix] = _ratio(known, 300)
        counts["query_"+suffix] = _ratio(counts["known_"+suffix]["count"]+counts["unknown_"+suffix]["count"], 500)
    both = min(counts["anchor_pair_both"]["count"], *(counts[name]["count"] for name in study.PAIR_METRICS))
    counts["anchor_pair_both"] = _ratio(both, 100)
    _refresh(panel)


def _score(results):
    WORK["summarize_attempts"] += 1
    result = study.summarize(results)
    WORK["summarize_completions"] += 1
    return result


class FoundationLayoutStudyScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.passing = _passing_results()

    def variant(self):
        WORK["mutated_case_variants"] += 1
        return deepcopy(self.passing)

    def test_passing_screen_covers_all_five_rules_and_trajectories(self):
        result = _score(self.passing)
        self.assertTrue(result["screen_passed"])
        self.assertEqual(result["rule_passed"], {str(rule): True for rule in range(1, 6)})
        self.assertEqual(len(result["checks"]), 91)
        self.assertEqual(len(result["development_trajectories"]), 12)
        for trajectory in result["development_trajectories"].values():
            self.assertEqual([p["step"] for p in trajectory["varied"]["trajectory"]], list(study.STEPS))

    def test_pooled_gain_cannot_hide_family_regression(self):
        results = self.variant()
        panel = _panel(results, "varied", "audit", "original", "fresh")
        for family, count in (("color", 39), ("count", 60), ("switch", 60)):
            for metric in study.PAIR_METRICS:
                _change(panel, family, metric, count)
        result = _score(results)
        self.assertTrue(result["rule_passed"]["1"])
        self.assertFalse(result["rule_passed"]["2"])
        self.assertFalse(result["screen_passed"])

    def test_action_gain_without_reply_gain_fails(self):
        results = self.variant()
        panel = _panel(results, "varied", "audit", "varied", "composed")
        for family in study.FAMILIES:
            _change(panel, family, "anchor_pair_reply", 40)
        result = _score(results)
        self.assertFalse(result["rule_passed"]["1"])
        self.assertTrue(result["rule_passed"]["2"])
        self.assertTrue(any(check["passed"] and check["condition"] == "varied/composed/anchor_pair_action"
                            for check in result["checks"] if check["rule"] == 1))

    def test_unknown_regression_fails_despite_known_gains(self):
        results = self.variant()
        panel = _panel(results, "varied", "audit", "varied", "fresh")
        for family in study.FAMILIES:
            _change(panel, family, "unknown_action", 149)
        result = _score(results)
        self.assertTrue(result["rule_passed"]["1"])
        self.assertFalse(result["rule_passed"]["3"])
        self.assertFalse(result["screen_passed"])

    def test_equal_blank_control_does_not_count_as_history_use(self):
        results = self.variant()
        for name in ("fresh", "composed"):
            panel = _panel(results, "varied", "dev", "original", name, control="blank")
            for metric in study.PAIR_METRICS:
                _change(panel, "color", metric, 50)
        result = _score(results)
        self.assertFalse(result["rule_passed"]["4"])
        self.assertTrue(result["rule_passed"]["5"])
        self.assertFalse(result["screen_passed"])

    def test_excess_forgetting_uses_earlier_peak_and_fixed_endpoint(self):
        results = self.variant()
        for name in ("fresh", "composed"):
            panel = _panel(results, "varied", "dev", "varied", name, step=720)
            for metric in study.PAIR_METRICS:
                _change(panel, "switch", metric, 65)
        result = _score(results)
        self.assertFalse(result["rule_passed"]["5"])
        self.assertTrue(result["rule_passed"]["1"])
        detail = result["development_trajectories"]["varied/switch/anchor_pair_action"]
        self.assertEqual(detail["varied"]["absolute_loss"], .15)
        self.assertEqual(detail["varied"]["endpoint_rate"], .5)
        self.assertFalse(result["screen_passed"])

    def test_matched_denominator_mismatch_raises(self):
        results = self.variant()
        panel = _panel(results, "varied", "audit", "original", "fresh")
        _change(panel, "color", "anchor_pair_action", 50, total=101)
        with self.assertRaisesRegex(ValueError, "denominators differ"):
            _score(results)
        WORK["expected_denominator_rejections"] += 1

    def test_exact_five_percentage_points_pass_but_one_count_below_fails(self):
        results = self.variant()
        for view in study.ARMS:
            for name in ("fresh", "composed"):
                panel = _panel(results, "varied", "audit", view, name)
                for family in study.FAMILIES:
                    for metric in study.PAIR_METRICS:
                        _change(panel, family, metric, 45)
        boundary = _score(results)
        self.assertTrue(boundary["screen_passed"])
        self.assertTrue(all(check["delta"] == .05 for check in boundary["checks"] if check["rule"] == 1))
        # Reuse the same synthetic object for one additional boundary case.
        WORK["mutated_case_variants"] += 1
        _change(_panel(results, "varied", "audit", "original", "fresh"), "color", "anchor_pair_reply", 44)
        below = _score(results)
        self.assertFalse(below["rule_passed"]["1"])
        self.assertFalse(below["screen_passed"])


if __name__ == "__main__":
    unittest.main()
