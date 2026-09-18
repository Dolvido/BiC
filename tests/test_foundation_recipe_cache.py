"""Recipe/cache mechanics only; no model, optimizer or performance benchmark."""
import copy
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_plan as planning
from experiments import foundation_recipe_cache as module
from experiments.foundation_evidence import reconstruct_anchor
from experiments.train_cognitive import fingerprint_rows


class FoundationRecipeCacheTests(unittest.TestCase):
    def plan(self, **options):
        return planning.build_plan(**{"seed": 704000001, "ordering_seed": 704000002,
            "stage_updates": 10, "rehearsal_every": 2, "final_updates": 6,
            "micro_batch_size": 4, **options})

    def reference(self, plan, bundle=0, family="color", pair=0):
        rows = planning.materialize_bundle(plan, bundle)[family][2*pair:2*pair+2]
        return {"bundle_id": bundle, "family": family, "pair_index": pair,
                "pair_sha256": fingerprint_rows(rows)}

    def test_cached_bundles_and_observation_packing_equal_canonical_rows(self):
        from experiments.composition_data import pack_composition_episodes
        from experiments.foundation_curriculum import validate_pair
        plan = self.plan()
        cache = module.FoundationRecipeCache(plan, max_bundles=3)
        chosen = [0, next(int(k) for k, v in plan["bundles"].items()
                          if v["depth"] == 4 and v["turns"] == 10)]
        for bundle in chosen:
            expected = planning.materialize_bundle(plan, bundle)
            for _ in range(2):
                actual = cache.get_bundle(bundle)
                self.assertEqual(actual, expected)
                self.assertEqual(fingerprint_rows(actual), fingerprint_rows(expected))
                for family in expected:
                    first = pack_composition_episodes(expected[family], training=True,
                                                       pair_validator=validate_pair, max_turns=12)
                    second = pack_composition_episodes(actual[family], training=True,
                                                        pair_validator=validate_pair, max_turns=12)
                    self.assertEqual(set(first["inputs"]), set(second["inputs"]))
                    for key in first["inputs"]:
                        torch.testing.assert_close(first["inputs"][key], second["inputs"][key], rtol=0, atol=0)
        self.assertEqual(cache.stats["materialized_bundles"], 2)
        self.assertEqual(cache.stats["misses"], 2)
        self.assertEqual(cache.stats["hits"], 2)

    def test_anchor_reconstruction_reuses_one_bundle_and_retains_exact_hashes(self):
        plan = self.plan()
        references = [self.reference(plan, family="color", pair=0),
                      self.reference(plan, family="count", pair=1)]
        expected = [reconstruct_anchor(plan, ref) for ref in references]
        cache = module.FoundationRecipeCache(plan)
        with patch.object(planning, "_materialize_validated_bundle",
                          wraps=planning._materialize_validated_bundle) as materialize:
            actual = [cache.reconstruct_anchor(ref) for ref in references]
        self.assertEqual(actual, expected)
        self.assertEqual(materialize.call_count, 1)
        self.assertEqual(cache.stats["materialized_bundles"], 1)
        self.assertEqual(cache.stats["hits"], 1)
        self.assertEqual(plan["schedules"], self.plan()["schedules"])

    def test_input_plan_returned_rows_anchor_and_statistics_are_isolated(self):
        plan = self.plan()
        original = copy.deepcopy(plan)
        reference = self.reference(plan)
        cache = module.FoundationRecipeCache(plan)
        identity = cache.plan_sha256
        plan["bundles"]["0"]["depth"] = 5
        rows = cache.get_bundle(0)
        rows["color"][0]["turns"][0]["text"] = "mutated"
        rows["count"].clear()
        pair = cache.reconstruct_anchor(reference)
        pair[0]["recipe"]["seed"] = 0
        pair[1]["turns"] = []
        stats = cache.stats; stats["hits"] = 999
        sources = cache.source_sha256; sources.clear()
        self.assertEqual(cache.plan_sha256, identity)
        self.assertEqual(cache.get_bundle(0), planning.materialize_bundle(original, 0))
        self.assertEqual(cache.reconstruct_anchor(reference), reconstruct_anchor(original, reference))
        self.assertNotEqual(cache.stats["hits"], 999)
        self.assertTrue(cache.source_sha256)

    def test_capacity_is_bounded_and_eviction_is_least_recently_used(self):
        cache = module.FoundationRecipeCache(self.plan(), max_bundles=2)
        for bundle in (0, 1, 0, 2, 1):
            cache.get_bundle(bundle)
            self.assertLessEqual(cache.stats["entries"], 2)
        self.assertEqual({key: cache.stats[key] for key in ("entries", "hits", "misses",
            "materialized_bundles", "evictions")},
            {"entries": 2, "hits": 1, "misses": 4, "materialized_bundles": 4, "evictions": 2})

    def test_full_order_override_and_protection_plan_identities_are_separate(self):
        baseline = self.plan()
        order = self.plan(ordering_seed=704000003)
        admission = {"schema": planning.ADMISSION_SCHEMA, "protected_sha256": "a"*64,
                     "protected_count": 1, "realization_attempts": {"0/color/0": 1}}
        override = self.plan(admission=admission)
        other_protection = self.plan(admission={**admission, "protected_sha256": "b"*64,
                                               "protected_count": 2})
        cache = module.FoundationRecipeCache(baseline, max_bundles=4)
        identities, rows = [], []
        for plan in (baseline, order, override, other_protection):
            identities.append(cache.admit(plan))
            rows.append(cache.get_bundle(0))
        self.assertEqual(len(set(identities)), 4)
        self.assertEqual(rows[0], rows[1])
        self.assertNotEqual(rows[0]["color"][:2], rows[2]["color"][:2])
        self.assertEqual(rows[2], rows[3])
        self.assertEqual(cache.stats["misses"], 4)
        self.assertEqual(cache.stats["entries"], 4)
        cache.admit(baseline)
        self.assertEqual(cache.get_bundle(0), rows[0])
        self.assertEqual(cache.stats["hits"], 1)

    def test_invalid_admission_and_indices_fail_without_changing_current_plan(self):
        plan = self.plan()
        for size in (0, -1, True, 1.0, 4097):
            with self.subTest(size=size), self.assertRaises(ValueError):
                module.FoundationRecipeCache(plan, max_bundles=size)
        cache = module.FoundationRecipeCache(plan)
        before = cache.plan_sha256
        bad = copy.deepcopy(plan); bad["schedules"]["mixed"][0] = -1
        with patch.object(planning, "_materialize_validated_bundle", side_effect=AssertionError("generation")):
            with self.assertRaises(ValueError):
                cache.admit(bad)
            for bundle in (-1, 66, True, 0.0, "0"):
                with self.subTest(bundle=bundle), self.assertRaises(ValueError):
                    cache.get_bundle(bundle)
        self.assertEqual(cache.plan_sha256, before)
        self.assertEqual(cache.stats["materialized_bundles"], 0)
        self.assertEqual(cache.stats["plan_admissions"], 1)

    def test_anchor_schema_indices_and_original_hash_reject_for_repaired_pair(self):
        baseline = self.plan()
        original = self.reference(baseline)
        admission = {"schema": planning.ADMISSION_SCHEMA, "protected_sha256": "a"*64,
                     "protected_count": 1, "realization_attempts": {"0/color/0": 1}}
        repaired = self.plan(admission=admission)
        cache = module.FoundationRecipeCache(repaired)
        ref = self.reference(repaired)
        self.assertEqual(cache.reconstruct_anchor(ref), reconstruct_anchor(repaired, ref))
        bad = [original, {**ref, "extra": True}, {k: v for k, v in ref.items() if k != "family"},
               {**ref, "family": "other"}, {**ref, "pair_index": True}, {**ref, "pair_index": 2},
               {**ref, "bundle_id": -1}, {**ref, "pair_sha256": "0"*64}]
        for item in bad:
            with self.subTest(reference=item), self.assertRaises(ValueError):
                cache.reconstruct_anchor(item)

    def test_source_drift_rejects_hits_misses_anchors_and_plan_switches(self):
        plan = self.plan(); ref = self.reference(plan)
        cache = module.FoundationRecipeCache(plan)
        cache.get_bundle(0)
        old = cache.source_sha256
        drift = {**old, "experiments/foundation_curriculum.py": "0"*64}
        before = cache.stats
        with patch.object(module, "source_hashes", return_value=drift):
            for action in (lambda: cache.get_bundle(0), lambda: cache.get_bundle(1),
                           lambda: cache.reconstruct_anchor(ref), lambda: cache.admit(plan)):
                with self.assertRaisesRegex(ValueError, "source identity changed"):
                    action()
        self.assertEqual(cache.stats["source_rejections"], 4)
        for key in ("entries", "hits", "misses", "materialized_bundles", "plan_admissions"):
            self.assertEqual(cache.stats[key], before[key])

    def test_mid_materialization_source_drift_does_not_publish_candidate_rows(self):
        plan = self.plan(); cache = module.FoundationRecipeCache(plan)
        old = cache.source_sha256; drift = {**old, "experiments/foundation_plan.py": "0"*64}
        with patch.object(module, "source_hashes", side_effect=[old, drift]):
            with self.assertRaisesRegex(ValueError, "source identity changed"):
                cache.get_bundle(0)
        self.assertEqual(cache.stats["entries"], 0)
        self.assertEqual(cache.stats["materialized_bundles"], 1)
        self.assertEqual(cache.stats["failed_materializations"], 1)
        self.assertEqual(cache.stats["misses"], 1)


if __name__ == "__main__":
    unittest.main()
