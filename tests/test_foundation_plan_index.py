"""Small canonical input checks only: no model, inference or optimizer work."""
import copy
from dataclasses import FrozenInstanceError
import pickle
import unittest
from unittest.mock import patch

from experiments import foundation_admission as admission
from experiments import foundation_curriculum as curriculum
from experiments import foundation_plan as planning
from experiments import foundation_plan_index as indexed
from experiments import foundation_training as training
from experiments.foundation_evidence import json_digest
from experiments.realization_banks import transcript_digest


class FoundationPlanIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = planning.build_plan(seed=951000001, stage_updates=10,
            final_updates=6, micro_batch_size=2, rehearsal_every=2,
            ordering_seed=951000002)
        cls.history = sorted({transcript_digest(planning.materialize_pair(
            cls.base, bundle, "color", 0)[0]) for bundle in (0, 65)})
        cls.plan, cls.receipt = admission.repair_plan(cls.base, cls.history)
        cls.protected = sorted([*cls.history, "f"*64])
        cls.index = cls.make_index()

    @classmethod
    def make_index(cls, **changes):
        options = {"admission_protected_transcripts": cls.history,
                   "protected_transcripts": cls.protected,
                   "admission_receipt": cls.receipt}
        options.update(changes)
        return indexed.AuthenticatedPlanIndex(cls.plan, **options)

    def test_both_orders_match_original_replay_at_stage_and_partial_boundaries(self):
        self.assertEqual(set(self.plan["admission"]["realization_attempts"]),
                         {"0/color/0", "65/color/0"})
        for order in indexed.ORDERS:
            for cursor in (0, 1, 10, 20, 30, 40, 50, 60, 63, 66):
                with self.subTest(order=order, cursor=cursor):
                    expected = training.replay_evidence(self.plan, order, cursor,
                        self.protected, include_bundles=True)
                    self.assertEqual(self.index.replay(order, cursor, include_bundles=True), expected)
                    self.assertEqual(self.index.replay(order, cursor), expected["evidence"])
        left, right = (self.index.replay(order, 66) for order in indexed.ORDERS)
        self.assertEqual(left["exposures"], right["exposures"])
        self.assertNotEqual(left["consumed_rows_sha256"], right["consumed_rows_sha256"])

    def test_identity_counts_and_construction_cost_scopes(self):
        identity, report = self.index.identity, self.index.construction
        self.assertEqual(identity["plan_sha256"], json_digest(self.plan))
        self.assertEqual(identity["admission_receipt_sha256"], json_digest(self.receipt))
        self.assertEqual(identity["admission_protection_sha256"], json_digest(self.history))
        self.assertEqual(identity["expanded_protection_sha256"], json_digest(self.protected))
        self.assertEqual(identity["unique_transcripts"], 396)
        self.assertEqual(report["admitted_scan_bundles"], 66)
        self.assertEqual(report["admitted_scan_episodes"], 396)
        self.assertEqual(report["admission_counts"], self.receipt["counts"])
        self.assertEqual(report["admission_counts"]["rerendered_pairs"], 2)
        for family, counts in report["target_counts"].items():
            self.assertEqual(sum(counts.values()), self.index.replay("mixed", 66)["exposures"][family]["turns"])
        seconds = ("admission_seconds", "admitted_plan_scan_seconds", "prefix_fold_seconds")
        self.assertTrue(all(report[key] >= 0 for key in seconds))
        self.assertLessEqual(sum(report[key] for key in seconds), report["construction_wall_seconds"])
        self.assertGreater(report["record_payload_bytes"], 0)
        self.assertGreater(report["prefix_payload_bytes"], 0)
        for name in ("foundation_plan_index", "foundation_admission", "foundation_evidence",
                     "foundation_training", "foundation_plan", "foundation_curriculum",
                     "composition_curriculum", "composition_training", "realization_banks"):
            self.assertIn(f"experiments/{name}.py", identity["source_sha256"])

    def test_input_and_output_aliases_cannot_mutate_index(self):
        plan, receipt = copy.deepcopy(self.plan), copy.deepcopy(self.receipt)
        history, protection = list(self.history), list(self.protected)
        before = copy.deepcopy((plan, receipt, history, protection))
        index = indexed.AuthenticatedPlanIndex(plan, admission_receipt=receipt,
            admission_protected_transcripts=history, protected_transcripts=protection)
        self.assertEqual((plan, receipt, history, protection), before)
        expected = index.replay("curriculum", 10, include_bundles=True)
        plan["schedules"]["curriculum"].clear(); receipt["counts"].clear()
        history.clear(); protection.clear()
        returned = index.replay("curriculum", 10, include_bundles=True)
        returned["evidence"]["exposures"].clear(); returned["bundles"][0]["families"].clear()
        identity, report = index.identity, index.construction
        identity["source_sha256"].clear(); report["target_counts"].clear()
        self.assertEqual(index.replay("curriculum", 10, include_bundles=True), expected)
        self.assertTrue(index.identity["source_sha256"])
        self.assertTrue(index.construction["target_counts"])
        with self.assertRaises(FrozenInstanceError):
            index._records = ()
        self.assertTrue(all(type(record) is bytes for record in index._records))
        with self.assertRaisesRegex(TypeError, "process-owned"):
            pickle.dumps(index)

    def test_repeated_lookups_do_not_rematerialize_or_reauthenticate(self):
        with patch.object(indexed, "_materialize_validated_bundle", side_effect=AssertionError("materialized")), \
             patch.object(planning, "_materialize_validated_bundle", side_effect=AssertionError("materialized")), \
             patch.object(curriculum, "generate_pair", side_effect=AssertionError("generated")), \
             patch.object(admission, "authenticate_admission", side_effect=AssertionError("readmitted")), \
             patch.object(training, "replay_evidence", side_effect=AssertionError("replayed")):
            for _ in range(2):
                for order in indexed.ORDERS:
                    for cursor in (0, 7, 34, 66):
                        result = self.index.replay(order, cursor, include_bundles=True)
                        self.assertEqual(result["evidence"]["cursor"], cursor)
                        self.assertEqual(len(result["bundles"]), cursor)

    def test_protection_and_receipt_are_not_self_asserted_cache_keys(self):
        accepted = planning.materialize_bundle(self.plan, 65)["color"][0]
        with self.assertRaisesRegex(ValueError, "protected exact transcript"):
            self.make_index(protected_transcripts=sorted({*self.protected, transcript_digest(accepted)}))
        with self.assertRaisesRegex(ValueError, "included in expanded"):
            self.make_index(protected_transcripts=[])
        with self.assertRaisesRegex(ValueError, "first-safe"):
            self.make_index(admission_protected_transcripts=[])
        changed = copy.deepcopy(self.receipt)
        changed["counts"]["accepted_pairs"] += 1
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.make_index(admission_receipt=changed)
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.make_index(admission_receipt=None)
        for values in (["G"*64], [self.history[0]]*2, "0"*64):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.make_index(protected_transcripts=values)

    def test_changed_plan_rejected_before_materialization(self):
        changed = copy.deepcopy(self.plan)
        changed["bundles"]["0"]["depth"] = 1
        with patch.object(indexed, "_materialize_validated_bundle", side_effect=AssertionError("materialized")):
            with self.assertRaisesRegex(ValueError, "canonical recipe"):
                indexed.AuthenticatedPlanIndex(changed,
                    admission_protected_transcripts=self.history,
                    protected_transcripts=self.protected, admission_receipt=self.receipt)

    def test_scan_rejects_corrupt_label_and_global_duplicate(self):
        materialize = indexed._materialize_validated_bundle

        def corrupt(plan, bundle_id):
            rows = materialize(plan, bundle_id)
            if bundle_id == 0:
                turn = rows["color"][0]["turns"][-1]
                turn["target"] = 1-turn["target"]
            return rows

        with patch.object(indexed, "_materialize_validated_bundle", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "canonical foundation provenance"):
                self.make_index()

        def duplicate(plan, bundle_id):
            # Bundles 0 and 3 are both depth0/8turn cells; each row is canonical.
            return materialize(plan, 0 if bundle_id == 3 else bundle_id)

        with patch.object(indexed, "_materialize_validated_bundle", side_effect=duplicate):
            with self.assertRaisesRegex(ValueError, "duplicate exact transcript"):
                self.make_index()

    def test_source_drift_rejects_every_lookup_and_construction(self):
        sources = indexed.source_hashes()
        changed = dict(sources)
        changed["experiments/foundation_plan_index.py"] = "0"*64
        with patch.object(indexed, "source_hashes", return_value=changed):
            for read in (lambda: self.index.identity, lambda: self.index.construction,
                         lambda: self.index.replay("mixed", 0)):
                with self.assertRaisesRegex(RuntimeError, "source identity"):
                    read()
        with patch.object(indexed, "source_hashes", side_effect=[sources, changed]):
            with self.assertRaisesRegex(RuntimeError, "during construction"):
                self.make_index()
        with patch.object(indexed, "source_hashes", side_effect=[sources, changed]):
            with self.assertRaisesRegex(RuntimeError, "source identity"):
                self.index.replay("curriculum", 66)

    def test_cursor_order_and_result_shape_boundaries(self):
        for cursor in (-1, 67, True, 1., None):
            with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                self.index.replay("mixed", cursor)
        for order in ("unknown", None, [], True):
            with self.subTest(order=order), self.assertRaises(ValueError):
                self.index.replay(order, 0)
        for include in (0, 1, None, "yes"):
            with self.subTest(include=include), self.assertRaises(ValueError):
                self.index.replay("mixed", 0, include_bundles=include)
        self.assertEqual(self.index.replay("mixed", 0, include_bundles=True),
                         {"evidence": training._empty_evidence(), "bundles": []})


if __name__ == "__main__":
    unittest.main()
