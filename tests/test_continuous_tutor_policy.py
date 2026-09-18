"""Eight pure stdlib tests; synthetic exact counts only, no data or model work."""
from collections import Counter
from copy import deepcopy
import json
import unittest

from experiments import continuous_tutor_policy as policy

PARENT = "a" * 64


def endpoint(joint=50):
    families = {}
    for family in policy.FAMILIES:
        counts = {name: dict(count=200 if name.startswith("known") else 50,
                            total=400 if name.startswith("known") else 100)
                  for name in policy.QUERY_METRICS}
        counts["anchor_pair_both"] = dict(count=joint, total=100)
        families[family] = dict(counts=counts)
    value = dict(by_family=families, overall=dict(counts={}))
    aggregate(value)
    return value


def aggregate(value):
    for name in policy.METRICS:
        value["overall"]["counts"][name] = {
            key: sum(value["by_family"][family]["counts"][name][key] for family in policy.FAMILIES)
            for key in ("count", "total")}


def branches():
    work = dict(synchronized_optimizer_updates=108, retained_episodes=10368,
        unknown_optimizer_outcomes=0, family_episode_exposures={family: 3456 for family in policy.FAMILIES})
    control = dict(status="completed", parent_checkpoint_sha256=PARENT,
        recipe_sha256="b"*64, cycle_spec=policy.cycle_spec(1234, 2),
        evaluation_inputs={role: (str(i+1)*64) for i, role in enumerate(policy.ROLES)},
        common_withdrawal_sha256="c"*64,
        actual_work={phase: deepcopy(work) for phase in ("teaching", "withdrawal")},
        teaching_metrics={role: endpoint() for role in policy.CORE_ROLES},
        final_metrics={role: endpoint() for role in policy.ROLES})
    tutor = deepcopy(control)
    tutor["final_metrics"]["dev"] = endpoint(55)
    return control, tutor


class ContinuousTutorPolicyTests(unittest.TestCase):
    def choose(self, control, tutor):
        return policy.select_continuation(control, tutor, expected_parent_identity=PARENT)

    def test_fixed_spec_deterministic_seeds_and_detached_json(self):
        first = policy.cycle_spec(1234, 2)
        self.assertEqual(first, json.loads(json.dumps(first)))
        self.assertEqual(first, policy.cycle_spec(1234, 2))
        self.assertNotEqual(first["seeds"], policy.cycle_spec(1234, 3)["seeds"])
        self.assertNotEqual(first["seeds"], policy.cycle_spec(1235, 2)["seeds"])
        self.assertEqual(len(set(first["seeds"].values())), 3)
        for cycle in range(64):
            self.assertTrue(all(0 <= seed < 2**63 for seed in policy.cycle_spec(853002001, cycle)["seeds"].values()))
        counts = Counter((slot["depth"], slot["turns"]) for slot in first["slots"])
        self.assertEqual(counts, {(d, t): 6 for d in range(6) for t in (8, 10, 12)})
        self.assertEqual((first["teaching_updates"], first["withdrawal_updates"]), (108, 108))
        self.assertEqual((first["learning_rate"], first["auxiliary_weight"]), (0.0003, 0.3))
        first["slots"][0]["depth"] = 99
        self.assertEqual(policy.cycle_spec(1234, 2)["slots"][0]["depth"], 0)
        for seed, index in ((True, 1), (1, -1), (1.0, 2)):
            with self.subTest(seed=seed, index=index), self.assertRaises(ValueError):
                policy.cycle_spec(seed, index)

    def test_exact_five_point_boundary_and_durable_result(self):
        control, tutor = branches()
        before = deepcopy((control, tutor))
        result = self.choose(control, tutor)
        self.assertEqual(result["selected_arm"], "tutor")
        self.assertEqual(result["differences"]["dev/joint"]["change_denominator"], 20)
        self.assertFalse(result["automatic_promotion"])
        self.assertFalse(result["learned_policy"])
        self.assertEqual(result, json.loads(json.dumps(result)))
        self.assertEqual((control, tutor), before)
        # Cached decimal fields cannot overturn exact integer-count arithmetic.
        tutor["final_metrics"]["dev"]["overall"]["counts"]["anchor_pair_both"]["value"] = -1000
        self.assertEqual(self.choose(control, tutor)["selected_arm"], "tutor")
        tutor["final_metrics"]["dev"]["by_family"]["color"]["counts"]["anchor_pair_both"]["count"] -= 1
        aggregate(tutor["final_metrics"]["dev"])
        self.assertEqual(self.choose(control, tutor)["selected_arm"], "procedural")

    def test_slice_losses_and_withdrawal_exact_boundary(self):
        for role in policy.ROLES:
            for metric in policy.QUERY_METRICS:
                with self.subTest(role=role, metric=metric):
                    control, tutor = branches()
                    pair = tutor["final_metrics"][role]["by_family"]["color"]["counts"][metric]
                    pair["count"] -= pair["total"] // 20
                    aggregate(tutor["final_metrics"][role])
                    self.assertEqual(self.choose(control, tutor)["selected_arm"], "tutor")
                    pair["count"] -= 1
                    aggregate(tutor["final_metrics"][role])
                    self.assertEqual(self.choose(control, tutor)["selected_arm"], "procedural")
        # Withdrawal failure can occur while the final cross-arm test passes.
        control, tutor = branches()
        tutor["teaching_metrics"]["retention"]["by_family"]["switch"]["counts"]["known_reply"]["count"] = 221
        aggregate(tutor["teaching_metrics"]["retention"])
        result = self.choose(control, tutor)
        self.assertTrue(result["checks"]["retention/switch/known_reply"])
        self.assertFalse(result["checks"]["withdrawal/retention/switch/known_reply"])
        self.assertEqual(result["selected_arm"], "procedural")

    def test_joint_fit_retention_transfers_and_family_gates(self):
        for role in policy.ROLES[1:]:
            with self.subTest(role=role):
                control, tutor = branches()
                tutor["final_metrics"][role] = endpoint(49)
                self.assertEqual(self.choose(control, tutor)["selected_arm"], "procedural")
        control, tutor = branches()
        for family, count in zip(policy.FAMILIES, (49, 63, 63)):
            tutor["final_metrics"]["dev"]["by_family"][family]["counts"]["anchor_pair_both"]["count"] = count
        aggregate(tutor["final_metrics"]["dev"])
        result = self.choose(control, tutor)
        self.assertTrue(result["checks"]["dev/joint"])
        self.assertFalse(result["checks"]["dev/color/joint"])
        self.assertEqual(result["selected_arm"], "procedural")

    def test_invalid_counts_denominators_and_missing_roles(self):
        for count, total in ((-1, 100), (101, 100), (1, 0), (True, 100), (1, 100.0)):
            with self.subTest(count=count, total=total):
                control, tutor = branches()
                tutor["final_metrics"]["dev"]["by_family"]["count"]["counts"]["known_action"] = dict(count=count, total=total)
                with self.assertRaises(ValueError): self.choose(control, tutor)
        for case in ("paired_denominator", "withdrawal_denominator", "overall_sum", "missing_role", "missing_family"):
            with self.subTest(case=case):
                control, tutor = branches()
                if case in ("paired_denominator", "withdrawal_denominator"):
                    field = "final_metrics" if case == "paired_denominator" else "teaching_metrics"
                    value = tutor[field]["dev"]
                    for metric in ("known_action", "known_reply"):
                        value["by_family"]["color"]["counts"][metric]["total"] += 1
                    aggregate(value)
                elif case == "overall_sum": tutor["final_metrics"]["dev"]["overall"]["counts"]["known_action"]["count"] += 1
                elif case == "missing_role": del tutor["final_metrics"]["transfer_varied"]
                else: del tutor["final_metrics"]["dev"]["by_family"]["switch"]
                with self.assertRaises(ValueError): self.choose(control, tutor)

    def test_foreign_parent_recipe_bank_and_cycle_rejected(self):
        for key, value in (("parent_checkpoint_sha256", "d"*64), ("recipe_sha256", "d"*64),
                           ("common_withdrawal_sha256", "d"*64), ("status", "failed"), ("status", "active")):
            with self.subTest(key=key, value=value):
                control, tutor = branches(); tutor[key] = value
                with self.assertRaises(ValueError): self.choose(control, tutor)
        for field in ("evaluation_inputs", "cycle_spec"):
            control, tutor = branches()
            if field == "evaluation_inputs": tutor[field]["dev"] = "d"*64
            else: tutor[field]["learning_rate"] = 0.003
            with self.assertRaises(ValueError): self.choose(control, tutor)
        control, tutor = branches()
        control["parent_checkpoint_sha256"] = tutor["parent_checkpoint_sha256"] = "d"*64
        with self.assertRaises(ValueError): self.choose(control, tutor)

    def test_actual_exposures_complete_and_known_in_both_phases(self):
        for arm in (0, 1):
            for phase in ("teaching", "withdrawal"):
                for field in ("synchronized_optimizer_updates", "retained_episodes", "unknown_optimizer_outcomes", "family_episode_exposures"):
                    with self.subTest(arm=arm, phase=phase, field=field):
                        pair = branches(); work = pair[arm]["actual_work"][phase]
                        if field == "family_episode_exposures": work[field]["switch"] -= 1
                        elif field == "unknown_optimizer_outcomes": work[field] = 1
                        else: work[field] -= 1
                        with self.assertRaises(ValueError): self.choose(*pair)
        control, tutor = branches(); del tutor["actual_work"]["withdrawal"]
        with self.assertRaises(ValueError): self.choose(control, tutor)

    def test_wall_budget_and_final_reserve_boundaries(self):
        def decide(budget=10, elapsed=7, required=2, reserve=1):
            return policy.stage_decision(budget_seconds=budget, elapsed_seconds=elapsed,
                required_seconds=required, final_reserve_seconds=reserve)
        self.assertTrue(decide()["start"])
        self.assertFalse(decide(elapsed=7.000001)["start"])
        self.assertFalse(decide(elapsed=11)["start"])
        self.assertTrue(decide(budget=0.3, elapsed=0.1, required=0.1, reserve=0.1)["start"])
        self.assertTrue(decide(elapsed=9, required=1, reserve=0)["start"])
        for kwargs in (dict(budget=0), dict(budget=float("inf")), dict(elapsed=-1),
                       dict(required=0), dict(required=float("nan")), dict(reserve=-1),
                       dict(reserve=11), dict(budget=True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): decide(**kwargs)

    def test_retention_clear_pass_and_exact_loss_boundary(self):
        control, tutor = branches(); choice = self.choose(control, tutor)
        candidates = dict(procedural=control["final_metrics"], tutor=tutor["final_metrics"])
        parent = deepcopy(control["final_metrics"])
        candidates["tutor"]["retention"]["by_family"]["color"]["counts"]["anchor_pair_both"]["count"] -= 5
        aggregate(candidates["tutor"]["retention"])
        saved = deepcopy((choice, candidates, parent))
        result = policy.protect_retention(choice, candidates=candidates, current_parent=parent, reference_parent=parent)
        self.assertEqual(result["selected_arm"], "tutor")
        self.assertTrue(result["retention_guard"]["passed"])
        self.assertEqual(len(result["retention_guard"]["checks"]), 128)
        self.assertEqual((choice, candidates, parent), saved)
        self.assertEqual(result, json.loads(json.dumps(result)))
        # Bad/missing selected-bank denominators must refuse, not become a pass.
        broken = deepcopy(parent); del broken["transfer_varied"]
        with self.assertRaises(ValueError):
            policy.protect_retention(choice, candidates=candidates, current_parent=broken, reference_parent=parent)
        broken = deepcopy(parent)
        for metric in ("unknown_action", "unknown_reply"):
            broken["dev"]["by_family"]["count"]["counts"][metric]["total"] += 1
        aggregate(broken["dev"])
        with self.assertRaises(ValueError):
            policy.protect_retention(choice, candidates=candidates, current_parent=broken, reference_parent=parent)

    def test_retention_tutor_failure_keeps_parent_not_other_candidate(self):
        control, tutor = branches(); choice = self.choose(control, tutor)
        parent = deepcopy(control["final_metrics"])
        candidates = dict(procedural=control["final_metrics"], tutor=tutor["final_metrics"])
        candidates["tutor"]["transfer_varied"]["by_family"]["switch"]["counts"]["unknown_reply"]["count"] -= 6
        aggregate(candidates["tutor"]["transfer_varied"])
        result = policy.protect_retention(choice, candidates=candidates, current_parent=parent, reference_parent=parent)
        self.assertEqual(result["selected_arm"], "parent")
        self.assertEqual(result["retention_guard"]["proposed_arm"], "tutor")
        self.assertIn("current_parent/transfer_varied/switch/unknown_reply", result["retention_guard"]["failed_checks"])

    def test_retention_procedural_failure_keeps_parent(self):
        control, tutor = branches(); tutor["final_metrics"]["dev"] = endpoint()
        choice = self.choose(control, tutor); self.assertEqual(choice["selected_arm"], "procedural")
        parent = deepcopy(control["final_metrics"])
        for family in policy.FAMILIES:
            control["final_metrics"]["retention"]["by_family"][family]["counts"]["anchor_pair_both"]["count"] -= 6
        aggregate(control["final_metrics"]["retention"])
        result = policy.protect_retention(choice, candidates=dict(procedural=control["final_metrics"], tutor=tutor["final_metrics"]),
            current_parent=parent, reference_parent=parent)
        self.assertEqual(result["selected_arm"], "parent")
        self.assertIn("current_parent/retention/overall/anchor_pair_both", result["retention_guard"]["failed_checks"])

    def test_fixed_campaign_reference_blocks_cumulative_forgetting(self):
        control, tutor = branches(); choice = self.choose(control, tutor)
        current = deepcopy(control["final_metrics"]); reference = deepcopy(current)
        # Each local change is only three points; total loss from start is six.
        for value, count in ((reference, 53), (current, 50), (tutor["final_metrics"], 47)):
            value["dev"]["by_family"]["color"]["counts"]["unknown_action"]["count"] = count
            aggregate(value["dev"])
        result = policy.protect_retention(choice, candidates=dict(procedural=control["final_metrics"], tutor=tutor["final_metrics"]),
            current_parent=current, reference_parent=reference)
        checks = result["retention_guard"]["checks"]
        self.assertTrue(checks["current_parent/dev/color/unknown_action"])
        self.assertFalse(checks["campaign_start/dev/color/unknown_action"])
        self.assertEqual(result["selected_arm"], "parent")


if __name__ == "__main__":
    unittest.main()
