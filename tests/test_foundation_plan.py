"""Prospective lesson-order mechanics; no model inference or optimizer work."""
from collections import Counter
import copy
import hashlib
import json
import random
import unittest
from unittest.mock import patch

import torch

from experiments.foundation_plan import (
    ADMISSION_SCHEMA, build_plan, materialize_bundle, materialize_pair, validate_plan,
)


class FoundationPlanTests(unittest.TestCase):
    def plan(self, **options):
        return build_plan(**{"seed": 401000001, "ordering_seed": 401000002,
                             "micro_batch_size": 2, **options})

    def test_orders_use_exact_same_unique_bundle_multiset(self):
        plan = self.plan()
        self.assertTrue(validate_plan(plan))
        total = 6 * 24 + 48
        self.assertEqual(set(plan["bundles"]), {str(index) for index in range(total)})
        for schedule in plan["schedules"].values():
            self.assertEqual(len(schedule), total)
            self.assertEqual(set(schedule), set(range(total)))
            self.assertEqual(len(schedule), len(set(schedule)))
        self.assertEqual(Counter(plan["schedules"]["curriculum"]),
                         Counter(plan["schedules"]["mixed"]))
        self.assertNotEqual(plan["schedules"]["curriculum"][:144],
                            plan["schedules"]["mixed"][:144])
        for key, bundle in plan["bundles"].items():
            self.assertEqual(int(key), bundle["id"])

    def test_shared_final_phase_has_equal_depth_coverage(self):
        plan = self.plan()
        first = plan["schedules"]["curriculum"][-48:]
        self.assertEqual(first, plan["schedules"]["mixed"][-48:])
        bundles = [plan["bundles"][str(index)] for index in first]
        self.assertEqual([bundle["depth"] for bundle in bundles], list(range(6)) * 8)
        self.assertTrue(all(bundle["phase"] == "mixed" and bundle["stage"] is None for bundle in bundles))

    def test_ascending_stages_rehearse_all_earlier_depths_in_order(self):
        plan = self.plan()
        for stage in range(6):
            ids = plan["schedules"]["curriculum"][24 * stage:24 * (stage + 1)]
            bundles = [plan["bundles"][str(index)] for index in ids]
            earlier = 0
            rehearsed = []
            for slot, bundle in enumerate(bundles, start=1):
                self.assertEqual(bundle["stage"], stage)
                self.assertEqual(bundle["phase"], "foundation")
                if stage and slot % 4 == 0:
                    expected = earlier % stage
                    earlier += 1
                    rehearsed.append(expected)
                else:
                    expected = stage
                self.assertEqual(bundle["depth"], expected)
            if stage:
                self.assertEqual(earlier, 6)
                self.assertEqual(set(rehearsed), set(range(stage)))

    def test_length_rotation_is_per_depth_across_stage_and_final_boundaries(self):
        plan = self.plan()
        for depth in range(6):
            lengths = [plan["bundles"][str(index)]["turns"]
                       for index in plan["schedules"]["curriculum"]
                       if plan["bundles"][str(index)]["depth"] == depth]
            self.assertEqual(lengths, [(8, 10, 12)[index % 3] for index in range(len(lengths))])
            counts = Counter(lengths)
            self.assertEqual(set(counts), {8, 10, 12})
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_ordering_seed_changes_only_prefix_order_and_does_not_consume_global_rng(self):
        rng = random.getstate()
        first, second = self.plan(ordering_seed=100), self.plan(ordering_seed=101)
        self.assertEqual(random.getstate(), rng)
        self.assertEqual(first["bundles"], second["bundles"])
        self.assertEqual(first["schedules"]["curriculum"], second["schedules"]["curriculum"])
        self.assertEqual(first["schedules"]["mixed"][-48:], second["schedules"]["mixed"][-48:])
        self.assertNotEqual(first["schedules"]["mixed"][:144], second["schedules"]["mixed"][:144])
        self.assertEqual(first, self.plan(ordering_seed=100))

    def test_resume_suffix_and_materialized_recipes_do_not_depend_on_traversal(self):
        plan = self.plan()
        saved = json.loads(json.dumps(plan))
        self.assertTrue(validate_plan(saved))
        for arm, schedule in plan["schedules"].items():
            for consumed in (0, 1, 24, 143, 144, len(schedule)):
                self.assertEqual(saved["schedules"][arm][consumed:], schedule[consumed:])
        chosen = next(bundle["id"] for bundle in plan["bundles"].values()
                      if bundle["depth"] == 3 and bundle["turns"] == 10)
        expected = materialize_bundle(plan, chosen)
        materialize_bundle(plan, 0)  # An intervening different lesson cannot advance recipe RNG.
        self.assertEqual(materialize_bundle(saved, chosen), expected)
        self.assertEqual(materialize_bundle(self.plan(ordering_seed=47), chosen), expected)
        self.assertEqual(plan, saved)
        self.assertIn(chosen, plan["schedules"]["curriculum"])
        self.assertIn(chosen, plan["schedules"]["mixed"])

    def test_invalid_configuration_is_rejected_before_any_generation(self):
        invalid = {
            "seed": (-1, 2**63, True, 1.5, None),
            "ordering_seed": (-1, 2**63, False, "1"),
            "stage_updates": (0, 4097, 19, 24.0, True),
            "final_updates": (0, 5, 7, 24577, 48.0, True),
            "micro_batch_size": (0, 1, 3, 514, 2.0, True),
            "rehearsal_every": (0, 1, 5, 4097, 4.0, True),
        }
        with patch("experiments.foundation_curriculum.generate_pair", side_effect=AssertionError("generation started")):
            for key, values in invalid.items():
                for value in values:
                    with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                        self.plan(**{key: value})

    def test_tampered_recipe_or_schedule_is_rejected_before_materialization(self):
        original = self.plan()

        def tamper(which, plan):
            if which == "schema":
                plan["schema"] = "unrecognized"
            elif which == "scope":
                plan["scope"] = "automatic promotion"
            elif which == "extra":
                plan["oracle_state"] = {}
            elif which == "missing_config":
                plan["config"].pop("seed")
            elif which == "extra_config":
                plan["config"]["hidden_curriculum"] = 1
            elif which == "changed_budget":
                plan["config"]["stage_updates"] = 28
            elif which == "duplicated_slot":
                plan["schedules"]["mixed"][1] = plan["schedules"]["mixed"][0]
            elif which == "final_reordered":
                plan["schedules"]["mixed"][-2:] = plan["schedules"]["mixed"][-2:][::-1]
            elif which == "missing_bundle":
                plan["bundles"].pop("0")
            else:
                field, value = {"id": ("id", 999), "depth": ("depth", 5), "turns": ("turns", 12),
                                "stage": ("stage", 2), "phase": ("phase", "mixed")}[which]
                plan["bundles"]["0"][field] = value

        cases = ("schema", "scope", "extra", "missing_config", "extra_config", "changed_budget",
                 "duplicated_slot", "final_reordered", "missing_bundle", "id", "depth", "turns", "stage", "phase")
        with patch("experiments.foundation_curriculum.generate_pair", side_effect=AssertionError("unvalidated generation")):
            for case in cases:
                with self.subTest(case=case):
                    changed = copy.deepcopy(original)
                    tamper(case, changed)
                    with self.assertRaises(ValueError):
                        validate_plan(changed)
                    with self.assertRaises(ValueError):
                        materialize_bundle(changed, 0)
            for bundle_id in (-1, len(original["bundles"]), True, 0.0, "0", None):
                with self.subTest(bundle_id=bundle_id), self.assertRaises(ValueError):
                    materialize_bundle(original, bundle_id)

    def test_generated_lessons_pack_only_observations_and_complete_pairs(self):
        from experiments.composition_data import pack_composition_episodes, pack_observations
        from experiments.foundation_curriculum import FAMILIES, final_depth, validate_pair
        plan = self.plan()
        for depth, turns in ((0, 8), (1, 10), (5, 12)):
            chosen = next(bundle["id"] for bundle in plan["bundles"].values()
                          if (bundle["depth"], bundle["turns"]) == (depth, turns))
            lesson = materialize_bundle(plan, chosen)
            self.assertEqual(set(lesson), set(FAMILIES))
            for family, rows in lesson.items():
                self.assertEqual(len(rows), 2)
                self.assertTrue(validate_pair(rows))
                self.assertTrue(all(row["family"] == family and final_depth(row) == depth for row in rows))
                self.assertTrue(all(len(row["turns"]) == turns for row in rows))
                self.assertEqual({row["turns"][-1]["target"] for row in rows}, {0, 1})
                packed = pack_composition_episodes(rows, training=True, pair_validator=validate_pair, max_turns=12)
                raw = pack_observations([[turn["text"] for turn in row["turns"]] for row in rows], max_turns=12)
                self.assertEqual(set(packed["inputs"]), set(raw))
                for key in raw:
                    torch.testing.assert_close(packed["inputs"][key], raw[key], rtol=0, atol=0)
                self.assertEqual(tuple(packed["supervision"]["action_targets"].shape), (2, turns))

    def admission(self, attempts=None):
        return {"schema": ADMISSION_SCHEMA, "protected_sha256": "a" * 64,
                "protected_count": 1, "realization_attempts": attempts or {}}

    def test_optional_admission_preserves_original_plan_and_materializations_exactly(self):
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode()).hexdigest()
        plan = self.plan(micro_batch_size=4)
        # Captured before the admission extension; protects the v1 seed namespace.
        self.assertEqual(digest(plan), "603fbaeeb2048347438aeb1fd7cd8bfd5bc714695b2683bb249c0696671da1f3")
        self.assertEqual(digest(materialize_bundle(plan, 0)), "6eaf09ffbd08bed7e0023f249825f7f5165b454d9a7a1df3464be6bf6ed67c99")
        self.assertEqual(digest(materialize_bundle(plan, 96)), "fe4e8a552564d5cf0328a6842a64411fb677c6892fca5f018a1fa20b1dc7fe43")
        self.assertEqual(plan, self.plan(micro_batch_size=4, admission=None))
        empty = self.plan(micro_batch_size=4, admission=self.admission())
        self.assertTrue(validate_plan(empty))
        self.assertEqual({k: v for k, v in empty.items() if k != "admission"}, plan)
        self.assertEqual(materialize_bundle(empty, 96), materialize_bundle(plan, 96))

    def test_only_selected_complete_pair_is_rerendered_without_changing_truth_or_procedure(self):
        from experiments.foundation_curriculum import validate_pair
        original = self.plan(micro_batch_size=4)
        admission = self.admission({"96/color/1": 1})
        admitted = self.plan(micro_batch_size=4, admission=admission)
        admission["realization_attempts"]["96/color/1"] = 2
        self.assertEqual(admitted["admission"]["realization_attempts"]["96/color/1"], 1)
        before, after = materialize_bundle(original, 96), materialize_bundle(admitted, 96)
        self.assertEqual(before["count"], after["count"])
        self.assertEqual(before["switch"], after["switch"])
        self.assertEqual(before["color"][:2], after["color"][:2])
        self.assertNotEqual(before["color"][2:], after["color"][2:])
        self.assertTrue(validate_pair(after["color"][2:]))
        self.assertEqual(materialize_bundle(admitted, 0), materialize_bundle(original, 0))
        self.assertEqual(admitted["config"], original["config"])
        self.assertEqual(admitted["bundles"], original["bundles"])
        self.assertEqual(admitted["schedules"], original["schedules"])
        for left, right in zip(before["color"][2:], after["color"][2:]):
            self.assertEqual({k: v for k, v in left.items() if k not in ("id", "recipe", "turns", "counterfactual_group")},
                             {k: v for k, v in right.items() if k not in ("id", "recipe", "turns", "counterfactual_group")})
            self.assertEqual({k: v for k, v in left["recipe"].items() if k != "naming_seed"},
                             {k: v for k, v in right["recipe"].items() if k != "naming_seed"})
            self.assertNotEqual(left["recipe"]["naming_seed"], right["recipe"]["naming_seed"])
            self.assertEqual([{k: v for k, v in turn.items() if k != "text"} for turn in left["turns"]],
                             [{k: v for k, v in turn.items() if k != "text"} for turn in right["turns"]])

    def test_candidate_zero_remains_original_and_override_reconstructs_from_saved_plan(self):
        from experiments.foundation_evidence import reconstruct_anchor
        from experiments.train_cognitive import fingerprint_rows
        baseline = self.plan(micro_batch_size=4)
        admitted = self.plan(micro_batch_size=4, admission=self.admission({"96/color/1": 1000}))
        self.assertEqual(materialize_pair(admitted, 96, "color", 1, attempt=0),
                         materialize_bundle(baseline, 96)["color"][2:])
        pair = materialize_pair(baseline, 96, "color", 1, attempt=1000)
        self.assertEqual(materialize_bundle(admitted, 96)["color"][2:], pair)
        saved = json.loads(json.dumps(admitted))
        self.assertTrue(validate_plan(saved))
        self.assertEqual(materialize_bundle(saved, 96), materialize_bundle(admitted, 96))
        reference = {"bundle_id": 96, "family": "color", "pair_index": 1,
                     "pair_sha256": fingerprint_rows(pair)}
        self.assertEqual(reconstruct_anchor(saved, reference), pair)
        wrong = {**reference, "pair_sha256": fingerprint_rows(materialize_pair(baseline, 96, "color", 1))}
        with self.assertRaises(ValueError):
            reconstruct_anchor(saved, wrong)

    def test_admission_envelope_rejects_noncanonical_fields_indices_and_attempts(self):
        valid = self.admission({"96/color/0": 1})
        invalid = []
        for key in valid:
            changed = copy.deepcopy(valid); changed.pop(key); invalid.append(changed)
        invalid += [{**valid, "extra": 1}, {**valid, "schema": "unknown"}]
        invalid += [{**valid, "protected_count": value} for value in (-1, True, 1.0, "1", 2**63)]
        invalid += [{**valid, "protected_sha256": value} for value in (None, "A" * 64, "g" * 64, "a" * 63)]
        invalid += [{**valid, "realization_attempts": value} for value in (None, [], "96/color/0")]
        for key in (96, "96/color", "96/color/0/extra", "096/color/0", "96/color/00",
                    "+96/color/0", "-1/color/0", "192/color/0", "96/color/1", "96/other/0", "９６/color/0"):
            invalid.append({**valid, "realization_attempts": {key: 1}})
        for attempt in (0, -1, 1001, True, 1.0, "1", None):
            invalid.append({**valid, "realization_attempts": {"96/color/0": attempt}})
        with patch("experiments.foundation_curriculum.generate_pair", side_effect=AssertionError("generation started")):
            for candidate in invalid:
                with self.subTest(candidate=candidate):
                    with self.assertRaises(ValueError):
                        self.plan(admission=candidate)
                    tampered = self.plan(); tampered["admission"] = candidate
                    with self.assertRaises(ValueError):
                        materialize_bundle(tampered, 96)
            tampered = self.plan(); tampered["admission"] = None
            with self.assertRaises(ValueError):
                validate_plan(tampered)

    def test_candidate_materialization_validates_indices_attempts_and_plan_before_generation(self):
        plan = self.plan()
        cases = ((True, "color", 0, 0), (-1, "color", 0, 0), (192, "color", 0, 0),
                 (0, "other", 0, 0), (0, None, 0, 0), (0, "color", True, 0),
                 (0, "color", 1, 0), (0, "color", 0, -1), (0, "color", 0, 1001),
                 (0, "color", 0, True), (0, "color", 0, 0.0))
        with patch("experiments.foundation_curriculum.generate_pair", side_effect=AssertionError("generation started")):
            for args in cases:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    materialize_pair(plan, *args)
            changed = copy.deepcopy(plan); changed["schedules"]["mixed"][0] = -1
            with self.assertRaises(ValueError):
                materialize_pair(changed, 0, "color", 0)


if __name__ == "__main__":
    unittest.main()
