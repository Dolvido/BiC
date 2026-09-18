"""Names-only admission evidence; no learner, inference or optimizer work."""
import copy
import json
import random
import unittest
from unittest.mock import patch

from experiments import foundation_admission as admission
from experiments import foundation_curriculum as curriculum
from experiments import foundation_plan as planning
from experiments.foundation_evidence import json_digest
from experiments.realization_banks import transcript_digest


class FoundationAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = planning.build_plan(seed=701000001, stage_updates=10, final_updates=6,
            micro_batch_size=2, rehearsal_every=2, ordering_seed=701000002)
        cls.original = planning.materialize_pair(cls.base, 0, "color", 0)
        cls.first = planning.materialize_pair(cls.base, 0, "color", 0, attempt=1)
        # Protect just one member: the complete pair must be replaced. Include
        # the first candidate to verify deterministic first-safe retry choice.
        cls.protected = sorted({transcript_digest(cls.original[0]), transcript_digest(cls.first[0])})
        cls.admitted, cls.receipt = admission.repair_plan(cls.base, cls.protected)

    def test_empty_protection_keeps_originals_and_exact_budget(self):
        admitted, report = admission.repair_plan(self.base)
        self.assertEqual(admitted["admission"]["realization_attempts"], {})
        self.assertEqual({key: value for key, value in admitted.items() if key != "admission"}, self.base)
        self.assertEqual(report["counts"]["accepted_pairs"], 66*3)
        self.assertEqual(report["counts"]["accepted_episodes"], 66*3*2)
        self.assertEqual(report["counts"]["unique_accepted_transcripts"], 66*3*2)
        self.assertEqual(report["counts"]["candidate_episodes"], 66*3*2)
        self.assertEqual(report["rejections"], [])
        for bundle_id in (0, 32, 65):
            self.assertEqual(planning.materialize_bundle(admitted, bundle_id),
                             planning.materialize_bundle(self.base, bundle_id))

    def test_first_safe_whole_pair_and_sparse_names_only_override(self):
        attempts = self.admitted["admission"]["realization_attempts"]
        self.assertEqual(set(attempts), {"0/color/0"})
        selected = attempts["0/color/0"]
        self.assertGreaterEqual(selected, 2)
        for attempt in range(selected):
            pair = planning.materialize_pair(self.base, 0, "color", 0, attempt=attempt)
            self.assertTrue(set(map(transcript_digest, pair)) & set(self.protected))
        actual = planning.materialize_bundle(self.admitted, 0)["color"]
        self.assertFalse(set(map(transcript_digest, actual)) & set(self.protected))
        self.assertEqual(actual, planning.materialize_pair(self.base, 0, "color", 0, attempt=selected))
        self.assertEqual(admission._meaning(actual), admission._meaning(self.original))
        self.assertNotEqual(actual[0]["recipe"]["naming_seed"], self.original[0]["recipe"]["naming_seed"])
        self.assertEqual(self.admitted["bundles"], self.base["bundles"])
        self.assertEqual(self.admitted["schedules"], self.base["schedules"])

    def test_complete_rejection_evidence_and_candidate_cost_counts(self):
        report = self.receipt
        selected = self.admitted["admission"]["realization_attempts"]["0/color/0"]
        self.assertEqual([row["attempt"] for row in report["rejections"]], list(range(selected)))
        self.assertTrue(all(row["reasons"]["protected"] for row in report["rejections"]))
        counts = report["counts"]
        self.assertEqual(counts["rerendered_pairs"], 1)
        self.assertEqual(counts["original_collision_pairs"], 1)
        self.assertEqual(counts["retry_candidates"], selected)
        self.assertEqual(counts["candidate_pairs"], counts["original_pairs"]+selected)
        self.assertEqual(counts["candidate_episodes"], 2*counts["candidate_pairs"])
        self.assertEqual(counts["rejected_candidate_pairs"], selected)
        self.assertEqual(counts["unchanged_pairs"]+counts["rerendered_pairs"], counts["accepted_pairs"])
        self.assertEqual(report["protected_sha256"], json_digest(self.protected))
        self.assertEqual(report["protected_count"], len(self.protected))
        self.assertEqual(report["original_plan_sha256"], json_digest(self.base))
        self.assertEqual(report["admitted_plan_sha256"], json_digest(self.admitted))
        self.assertTrue(all(row["typed_program_values_ancestry_targets_replies_unchanged"]
                            for row in report["overrides"]))
        self.assertNotIn("seconds", json.dumps(report))

    def test_all_depths_and_domains_keep_values_program_and_every_query(self):
        chosen = {row["depth"]: row["id"] for row in self.base["bundles"].values()}
        for depth, bundle_id in chosen.items():
            for family in curriculum.FAMILIES:
                with self.subTest(depth=depth, family=family):
                    original = planning.materialize_pair(self.base, bundle_id, family, 0)
                    candidate = planning.materialize_pair(self.base, bundle_id, family, 0, attempt=3)
                    self.assertEqual(admission._meaning(original), admission._meaning(candidate))
                    for first, second in zip(original, candidate):
                        self.assertEqual(first["query_ancestries"], second["query_ancestries"])
                        self.assertEqual([turn["target"] for turn in first["turns"]],
                                         [turn["target"] for turn in second["turns"]])
                        self.assertEqual([turn["reply"] for turn in first["turns"]],
                                         [turn["reply"] for turn in second["turns"]])

    def test_json_roundtrip_authentication_and_isolated_deterministic_inputs(self):
        plan, protected = copy.deepcopy(self.base), list(self.protected)
        before, rng = copy.deepcopy(plan), random.getstate()
        actual, report = admission.repair_plan(plan, protected)
        self.assertEqual(actual, self.admitted)
        self.assertEqual(report, self.receipt)
        self.assertEqual(plan, before)
        self.assertEqual(protected, self.protected)
        self.assertEqual(random.getstate(), rng)
        self.assertEqual(admission.authenticate_admission(plan, json.loads(json.dumps(actual)),
            protected_transcripts=protected, receipt=json.loads(json.dumps(report))), report)
        actual["schedules"]["mixed"].clear()
        report["rejections"].clear()
        self.assertEqual(plan, before)
        self.assertTrue(self.receipt["rejections"])

    def test_changed_attempt_and_receipt_are_rejected(self):
        changed = copy.deepcopy(self.admitted)
        changed["admission"]["realization_attempts"]["0/color/0"] += 1
        with self.assertRaisesRegex(ValueError, "first-safe"):
            admission.authenticate_admission(self.base, changed, self.protected, self.receipt)
        for change in ("counts", "rejections", "scope"):
            changed = copy.deepcopy(self.receipt)
            if change == "counts": changed["counts"]["accepted_pairs"] = True
            elif change == "rejections": changed["rejections"].pop()
            else: changed["scope"] += " modified"
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "receipt"):
                admission.authenticate_admission(self.base, self.admitted, self.protected, changed)

    def test_invalid_base_and_protection_reject_before_materialization(self):
        bad_plan = copy.deepcopy(self.base)
        bad_plan["bundles"]["0"]["depth"] = True
        with patch.object(planning, "_materialize_validated_bundle", side_effect=AssertionError("materialized")):
            for protected in ([True], [None], ["G"*64], ["0"*64, "0"*64], "0"*64):
                with self.subTest(protected=protected), self.assertRaises(ValueError):
                    admission.repair_plan(self.base, protected)
            for plan in (bad_plan, self.admitted):
                with self.assertRaises(ValueError): admission.repair_plan(plan)

    def test_wrong_protection_identity_cannot_authenticate(self):
        with self.assertRaisesRegex(ValueError, "first-safe"):
            admission.authenticate_admission(self.base, self.admitted, (), self.receipt)

    def test_retry_exhaustion_is_bounded_and_preserves_partial_evidence(self):
        original = copy.deepcopy(self.original)
        with patch.object(planning, "materialize_pair", return_value=original) as candidate:
            with self.assertRaises(admission.AdmissionError) as caught:
                admission.repair_plan(self.base, [transcript_digest(original[0])])
        self.assertEqual(candidate.call_count, 1000)
        report = caught.exception.receipt
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["counts"]["accepted_pairs"], 0)
        self.assertEqual(report["counts"]["candidate_pairs"], 1001)
        self.assertEqual(len(report["rejections"]), 1001)
        self.assertEqual(report["rejections"][-1]["attempt"], 1000)
        self.assertEqual(report["failed_reference"], {"bundle_id": 0, "family": "color", "pair_index": 0})
        self.assertEqual(original, self.original)
        self.assertNotIn("admission", self.base)

    def test_canonical_but_changed_value_seed_is_not_a_naming_retry(self):
        original = self.original[0]
        recipe = original["recipe"]
        candidate = curriculum.generate_pair("color", recipe["seed"], depth=recipe["depth"],
            turns=recipe["turns"], naming_seed=recipe["naming_seed"]+1,
            value_seed=recipe["value_seed"]+1, structure_split=recipe["structure_split"])
        self.assertTrue(curriculum.validate_pair(candidate))
        with patch.object(planning, "materialize_pair", return_value=candidate):
            with self.assertRaisesRegex(admission.AdmissionError, "program, values") as caught:
                admission.repair_plan(self.base, [transcript_digest(original)])
        self.assertEqual(caught.exception.receipt["counts"]["accepted_pairs"], 0)
        self.assertEqual(caught.exception.receipt["counts"]["retry_candidates"], 1)

    def test_seen_protection_and_intrapair_reasons_do_not_mutate_sets(self):
        protected, seen = {"a"}, {"b"}
        self.assertEqual(admission._reasons(["a", "b"], protected, seen),
            {"protected": ["a"], "already_admitted": ["b"], "within_pair_duplicate": False})
        self.assertTrue(admission._collision(admission._reasons(["c", "c"], protected, seen)))
        self.assertFalse(admission._collision(admission._reasons(["c", "d"], protected, seen)))
        self.assertEqual(protected, {"a"})
        self.assertEqual(seen, {"b"})


if __name__ == "__main__":
    unittest.main()
