"""Synthetic traces for a pure engineered policy; no trained model or study scores."""
import copy
from dataclasses import asdict
import hashlib
import json
import unittest

import torch

from experiments.composition_evaluation import prediction_metrics
from experiments.english_allocation import (
    AllocationConfig, advance, initialize, observe, plan, restore, snapshot, status,
)
from experiments.realization_backend import progress_vector
from experiments.sequence_student import SequenceConfig


FAMILIES = ("color", "count", "switch")


def fixture_specs():
    return {role: {f"{family}/{cell}": {"family": family, "identity": {
        "sha256": hashlib.sha256(f"{role}/{family}/{cell}".encode()).hexdigest(),
        "version": "synthetic-fixture-v1", "episodes": 20, "turns": 2, "role": "dev",
        "config": asdict(SequenceConfig(max_turns=12)),
    }} for family in FAMILIES for cell in ("a", "b")} for role in ("development", "retention")}


def fixture_evidence(specs, changes=None):
    """Independent hand predictions, not generated labels based on policy decisions."""
    changes = {} if changes is None else changes
    result = {}
    for role, banks in specs.items():
        rows = {}
        for name, spec in banks.items():
            good, reply_good, false_ask = changes.get((role, name), (3, 3, 0))
            targets = torch.full((20, 2), 2, dtype=torch.long)
            targets[:, -1] = torch.arange(20) % 2
            prediction = targets.clone()
            prediction[2 * good:, -1] = 1 - prediction[2 * good:, -1]
            if false_ask:
                prediction[-false_ask:, -1] = 2
            logits = torch.full((20, 2, 4), -2.)
            logits.scatter_(-1, prediction.unsqueeze(-1), 2.)
            reply_correct = torch.ones((20, 2), dtype=torch.bool)
            reply_correct[2 * reply_good:, -1] = False
            reply_actions = targets.clone()
            reply_actions[~reply_correct] = 1 - reply_actions[~reply_correct]
            row = prediction_metrics(logits, targets, reply_correct=reply_correct, reply_actions=reply_actions)
            row.update(bank=copy.deepcopy(spec["identity"]), control="normal", free_running_replies=True,
                       teacher_used_for_policy=False, decoder_prefix="BOS only", seconds=.01)
            rows[name] = row
        result[role] = {"role": role, "per_bank": rows,
                        "progress": {name: progress_vector(row) for name, row in rows.items()},
                        "wall_seconds": .1, "automatic_promotion": False}
    return result


class EnglishAllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.specs = fixture_specs()
        cls.evidence = fixture_evidence(cls.specs)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def state(self, mode="progress", **options):
        config = AllocationConfig(mode=mode, window_updates=4, min_family_microbatches=2, **options)
        return observe(initialize(config, self.specs), self.evidence, window_seconds=0)

    def next_window(self, state, changes=None, seconds=2.):
        return observe(advance(state, status(state)["window_updates"] - status(state)["cursor"]),
                       fixture_evidence(self.specs, changes), window_seconds=seconds)

    def test_cold_start_joint_and_exact_budget_without_plan_consumption(self):
        initial = initialize(AllocationConfig(), self.specs)
        self.assertFalse(status(initial)["has_baseline"])
        with self.assertRaisesRegex(ValueError, "initial"):
            plan(initial)
        state = self.state()
        before = snapshot(state)
        self.assertEqual(plan(state)["schedule"], [list(FAMILIES)] * 4)
        self.assertEqual(plan(state, max_updates=2)["schedule"], [list(FAMILIES)] * 2)
        self.assertEqual(plan(state, max_updates=0)["schedule"], [])
        self.assertEqual(state, before)
        self.assertEqual(plan(state)["decision"]["reason"], "joint_cold_start")
        self.assertEqual(sum(plan(state)["decision"]["full_window_family_microbatches"].values()), 12)
        with self.assertRaises(ValueError):
            plan(state, max_updates=True)

    def test_progress_uses_all_three_metrics_and_shared_window_cost(self):
        changes = {("development", f"color/{cell}"): (5, 5, 0) for cell in ("a", "b")}
        changes.update({("development", f"count/{cell}"): (8, 3, 0) for cell in ("a", "b")})
        state = self.next_window(self.state(), changes, seconds=2.)
        decision = plan(state)["decision"]
        self.assertEqual(decision["reason"], "observed_progress")
        self.assertEqual(decision["selected_families"], ["color"])
        self.assertAlmostEqual(decision["progress_per_shared_window_second"]["color"], .1)
        self.assertEqual(decision["progress_per_shared_window_second"]["count"], 0.)
        self.assertEqual(plan(state)["full_schedule"], [list(FAMILIES)] * 2 + [["color"] * 3] * 2)
        slower = self.next_window(self.state(), changes, seconds=4.)
        self.assertAlmostEqual(plan(slower)["decision"]["progress_per_shared_window_second"]["color"], .05)
        self.assertEqual(plan(slower)["full_schedule"], plan(state)["full_schedule"])

    def test_aggregate_improvement_cannot_hide_cell_regression_or_reply_loss(self):
        changes = {("development", "color/a"): (2, 2, 0), ("development", "color/b"): (9, 9, 0),
                   ("development", "count/a"): (9, 9, 0), ("development", "count/b"): (9, 9, 0),
                   ("retention", "switch/a"): (3, 1, 0)}
        state = self.next_window(self.state(), changes)
        decision = plan(state)["decision"]
        self.assertGreater(decision["progress_per_shared_window_second"]["color"], 0.)
        self.assertEqual(decision["reason"], "retention_alarm")
        self.assertEqual(decision["selected_families"], ["switch", "color"])
        keys = {(a["role"], a["bank"], a["metric"]) for a in decision["alarms"]}
        self.assertIn(("development", "color/a", "paired_action"), keys)
        self.assertIn(("retention", "switch/a", "paired_reply"), keys)
        self.assertTrue(all(count >= 2 for count in decision["full_window_family_microbatches"].values()))
        unchanged = self.next_window(state, changes)
        self.assertEqual(plan(unchanged)["decision"]["alarms"], decision["alarms"])

    def test_unsupported_ask_increase_remains_explicit_alarm_despite_accuracy_gains(self):
        changes = {("development", "color/a"): (5, 5, 2),
                   ("development", "count/a"): (9, 9, 0), ("development", "count/b"): (9, 9, 0)}
        state = self.next_window(self.state(), changes)
        decision = plan(state)["decision"]
        self.assertEqual(decision["selected_families"], ["color"])
        alarm = [a for a in decision["alarms"] if a["metric"] == "unsupported_ask"]
        self.assertEqual(len(alarm), 1)
        self.assertEqual(alarm[0]["shortfall"], .1)
        vector = state["latest"]["vectors"]["development"]["color/a"]
        self.assertEqual(vector["unsupported_ask"], {"count": 2, "known_total": 20, "rate": .1})

    def test_joint_baseline_ignores_scores_but_preserves_alarm_evidence(self):
        original = self.state(mode="joint")
        first = self.next_window(original)
        changed = self.next_window(original, {("retention", "count/a"): (0, 0, 3),
                                              ("development", "color/a"): (10, 10, 0)})
        self.assertEqual(plan(first)["schedule"], plan(changed)["schedule"])
        self.assertEqual(plan(changed)["schedule"], [list(FAMILIES)] * 4)
        self.assertEqual(plan(changed)["decision"]["reason"], "fixed_joint")
        self.assertTrue(plan(changed)["decision"]["alarms"])

    def test_floor_debt_json_resume_and_partial_continuation_are_exact(self):
        state = self.next_window(self.state(), {("development", "color/a"): (5, 5, 0)})
        partial = advance(state, 1)
        self.assertEqual(status(partial)["coverage_debt"], dict.fromkeys(FAMILIES, 1))
        self.assertFalse(status(partial)["window_complete"])
        config = AllocationConfig(mode="progress", window_updates=4, min_family_microbatches=2)
        resumed = restore(json.loads(json.dumps(snapshot(partial))), config, self.specs)
        self.assertEqual(plan(resumed), plan(partial))
        resumed = advance(advance(resumed, 1), 2)
        uninterrupted = advance(state, 4)
        self.assertEqual(resumed, uninterrupted)
        self.assertEqual(status(resumed)["total_updates"], 8)
        self.assertEqual(status(resumed)["family_microbatches"], {"color": 12, "count": 6, "switch": 6})
        self.assertEqual(status(resumed)["coverage_debt"], dict.fromkeys(FAMILIES, 0))
        self.assertTrue(status(resumed)["window_complete"])
        self.assertEqual(plan(resumed)["schedule"], [])
        self.assertEqual(advance(resumed, 0), resumed)

    def test_invalid_config_and_illegal_transitions_fail_without_mutation(self):
        for options in ({"mode": "learned"}, {"window_updates": True}, {"window_updates": 4097},
                        {"window_updates": 2, "min_family_microbatches": 3},
                        {"min_family_microbatches": 0}, {"regression_tolerance": float("nan")},
                        {"families": ("color", "color", "switch")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                AllocationConfig(**options)
        state = self.state()
        original = copy.deepcopy(state)
        for updates in (-1, True, 5, 1.5):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                advance(state, updates)
        with self.assertRaisesRegex(ValueError, "complete window"):
            observe(state, self.evidence, window_seconds=1.)
        for seconds in (0., -1., float("nan"), float("inf"), True):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                observe(advance(state, 4), self.evidence, window_seconds=seconds)
        self.assertEqual(state, original)

    def test_evidence_rejects_missing_nan_roles_identity_config_and_oracle_metadata(self):
        mutations = [lambda e: e.pop("retention"),
                     lambda e: e.update(audit=e["development"]),
                     lambda e: e["retention"].update(role="development"),
                     lambda e: e["development"]["per_bank"]["color/a"].pop("known_accuracy"),
                     lambda e: e["development"]["per_bank"]["color/a"].update(query_loss=float("nan")),
                     lambda e: e["development"]["per_bank"]["color/a"].update(teacher_used_for_policy=True),
                     lambda e: e["development"]["per_bank"]["color/a"].update(free_running_replies=False),
                     lambda e: e["development"]["per_bank"]["color/a"]["bank"].update(role="audit"),
                     lambda e: e["development"]["per_bank"]["color/a"]["bank"].update(sha256="0" * 64),
                     lambda e: e["development"]["per_bank"]["color/a"]["bank"]["config"].update(width=192)]
        initial = initialize(AllocationConfig(), self.specs)
        for mutation in mutations:
            evidence = copy.deepcopy(self.evidence)
            mutation(evidence)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                observe(initial, evidence, window_seconds=0)

    def test_count_rate_by_turn_and_redundant_vector_tampering_rejected(self):
        def mutate_row(key, value):
            return lambda e: e["development"]["per_bank"]["color/a"].update({key: value})
        mutations = [mutate_row("query_correct", True), mutate_row("known_accuracy", .9),
                     mutate_row("ask_predicted", 100), mutate_row("reply_parseable_queries", 41),
                     lambda e: e["development"]["per_bank"]["color/a"]["by_turn"][1].update(accuracy=1.),
                     lambda e: e["development"]["per_bank"]["color/a"]["by_turn"][1].update(opposite_pair_total=99),
                     lambda e: e["development"]["progress"]["color/a"].update(final_pair_accuracy=.9)]
        initial = initialize(AllocationConfig(), self.specs)
        for mutation in mutations:
            evidence = copy.deepcopy(self.evidence)
            mutation(evidence)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                observe(initial, evidence, window_seconds=0)

    def test_restore_contract_and_schedule_cursor_tampering_rejected(self):
        state = self.state()
        config = AllocationConfig(mode="progress", window_updates=4, min_family_microbatches=2)
        specs = copy.deepcopy(self.specs)
        specs["development"]["color/a"]["identity"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "contract"):
            restore(state, config, specs)
        with self.assertRaisesRegex(ValueError, "contract"):
            restore(state, AllocationConfig(mode="joint", window_updates=4, min_family_microbatches=2), self.specs)
        for mutation in (lambda s: s.update(cursor=1), lambda s: s["schedule"][0].__setitem__(0, "count"),
                         lambda s: s["family_microbatches"].update(color=True),
                         lambda s: s["decision"].update(reason="promoted")):
            payload = copy.deepcopy(state)
            mutation(payload)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                restore(payload, config, self.specs)

    def test_boolean_aliases_cannot_masquerade_as_count_or_config_integers(self):
        specs = copy.deepcopy(self.specs)
        for banks in specs.values():
            for spec in banks.values():
                spec["identity"]["config"]["layers"] = 1
        evidence = fixture_evidence(specs)
        initial = initialize(AllocationConfig(), specs)
        # Python considers True == 1; exact identity validation must not.
        malformed = copy.deepcopy(evidence)
        malformed["development"]["per_bank"]["color/a"]["bank"]["config"]["layers"] = True
        with self.assertRaises(ValueError):
            observe(initial, malformed, window_seconds=0)
        malformed = copy.deepcopy(evidence)
        # An unsupported ASK count is zero in this fixture; False must not alias it.
        malformed["development"]["progress"]["color/a"]["unsupported_ask"]["count"] = False
        with self.assertRaises(ValueError):
            observe(initial, malformed, window_seconds=0)
        with self.assertRaises(ValueError):
            snapshot({})

    def test_outputs_are_independent_and_explicit_tolerance_is_recorded(self):
        state = self.state(regression_tolerance=.15)
        changed = self.next_window(state, {("retention", "color/a"): (2, 2, 0)})
        self.assertEqual(plan(changed)["decision"]["alarms"], [])
        self.assertEqual(changed["config"]["regression_tolerance"], .15)
        saved = snapshot(changed)
        returned = plan(changed)
        returned["full_schedule"][0][0] = "invalid"
        returned["decision"]["alarms"].append({})
        saved["latest"]["vectors"]["development"]["color/a"]["final_pairs"]["correct"] = 99
        self.assertEqual(changed, snapshot(changed))
        self.assertEqual(plan(changed)["full_schedule"][0], list(FAMILIES))
        self.assertEqual(changed["latest"]["vectors"]["development"]["color/a"]["final_pairs"]["correct"], 3)


if __name__ == "__main__":
    unittest.main()
