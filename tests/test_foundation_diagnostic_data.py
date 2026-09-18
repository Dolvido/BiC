"""Small untrained fixtures; deliberately different seeds from the diagnostic."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from experiments import foundation_diagnostic_data as data


ROOTS = {"fit": 941170001, "evaluation": 941170002}
CELL = "color/d0/direct/t8"


class DiagnosticDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.banks, cls.receipt = data.build_pair_of_splits(excluded_transcripts=(), roots=ROOTS,
            pairs_per_cell=1)

    def test_complete_balanced_canonical_inventory_and_disjoint_roles(self):
        self.assertFalse(self.receipt["production_contract"])
        sets = []
        for role, split in (("fit", "train"), ("evaluation", "dev")):
            banks, receipt = self.banks[role], self.receipt["roles"][role]
            self.assertEqual(set(banks), set(data.CELLS))
            self.assertEqual(len(banks), 63)
            self.assertEqual(receipt["counts"]["accepted_pairs"], 63)
            self.assertEqual(receipt["counts"]["accepted_episodes"], 126)
            self.assertEqual(sum(row["turns"] for row in receipt["banks"].values()), 1260)
            for cell, pair in banks.items():
                self.assertTrue(data.curriculum.validate_pair(pair))
                self.assertEqual([row["variant"] for row in pair], [0, 1])
                self.assertEqual({row["turns"][-1]["target"] for row in pair}, {0, 1})
                for row in pair:
                    self.assertEqual(row["split"], split)
                    self.assertEqual(data.training_cell(row), cell)
                    self.assertEqual(row["recipe"]["structure_split"], "shared" if row["depth"] < 2 else "train")
            hashes = {data.transcript_digest(row) for pair in banks.values() for row in pair}
            self.assertEqual(sorted(hashes), receipt["transcript_sha256"])
            self.assertEqual(len(hashes), 126)
            self.assertEqual(receipt["banks_sha256"], data.json_digest(banks))
            counts = receipt["counts"]
            self.assertEqual(counts["candidate_pairs"], counts["accepted_pairs"] +
                             counts["rejected_wrong_cell"] + counts["rejected_collision"])
            self.assertEqual(counts["generated_episode_candidates"], 2 * counts["candidate_pairs"])
            sets.append(hashes)
        self.assertFalse(sets[0] & sets[1])
        self.assertEqual(self.receipt["roles"]["evaluation"]["excluded_count"], 126)

    def test_seed_coordinates_are_stable_distinct_and_bounded(self):
        first = data.candidate_seeds("fit", ROOTS["fit"], CELL, 0, 0)
        self.assertEqual(first, data.candidate_seeds("fit", ROOTS["fit"], CELL, 0, 0))
        self.assertEqual(len(set(first.values())), 3)
        self.assertTrue(all(type(value) is int and 0 <= value < 2**63 for value in first.values()))
        for role, root, slot, attempt in (("evaluation", ROOTS["fit"], 0, 0),
                ("fit", ROOTS["evaluation"], 0, 0), ("fit", ROOTS["fit"], 1, 0), ("fit", ROOTS["fit"], 0, 1)):
            self.assertNotEqual(first, data.candidate_seeds(role, root, CELL, slot, attempt))

    def test_collision_resamples_whole_pair_and_keeps_caller_inputs(self):
        blocked = [data.transcript_digest(self.banks["fit"][CELL][0])]
        original = deepcopy(blocked)
        banks, receipt = data.build_split("fit", excluded_transcripts=blocked,
            root_seed=ROOTS["fit"], pairs_per_cell=1, cells=[CELL])
        self.assertEqual(blocked, original)
        self.assertGreaterEqual(receipt["counts"]["rejected_collision"], 1)
        self.assertFalse(set(blocked) & set(receipt["transcript_sha256"]))
        self.assertTrue(data.curriculum.validate_pair(banks[CELL]))
        repeated = data.build_split("fit", excluded_transcripts=blocked,
            root_seed=ROOTS["fit"], pairs_per_cell=1, cells=[CELL])
        self.assertEqual((banks, receipt), repeated)

    def test_canonical_failure_returns_no_admitted_pair(self):
        with patch.object(data.curriculum, "validate_pair", side_effect=ValueError("canonical refusal")):
            with self.assertRaises(data.DiagnosticAdmissionError) as caught:
                data.build_split("fit", excluded_transcripts=(), root_seed=ROOTS["fit"],
                                 pairs_per_cell=1, cells=[CELL])
        receipt = caught.exception.receipt
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["counts"]["candidate_pairs"], 1)
        self.assertEqual(receipt["counts"]["generated_episode_candidates"], 2)
        self.assertEqual(receipt["counts"]["canonical_pair_checks"], 0)
        self.assertEqual(receipt["counts"]["accepted_pairs"], 0)

    def test_fixed_attempt_exhaustion_preserves_failure_counts(self):
        seeds = data.candidate_seeds("fit", ROOTS["fit"], CELL, 0, 0)
        blocked = [data.transcript_digest(self.banks["fit"][CELL][0])]
        with patch.object(data, "MAX_ATTEMPTS_PER_PAIR", 2), patch.object(data, "candidate_seeds", return_value=seeds):
            with self.assertRaises(data.DiagnosticAdmissionError) as caught:
                data.build_split("fit", excluded_transcripts=blocked, root_seed=ROOTS["fit"],
                                 pairs_per_cell=1, cells=[CELL])
        receipt = caught.exception.receipt
        self.assertEqual(receipt["max_attempts_per_pair"], 2)
        self.assertEqual(receipt["counts"]["candidate_pairs"], 2)
        self.assertEqual(receipt["counts"]["rejected_collision"], 2)
        self.assertEqual(receipt["counts"]["accepted_pairs"], 0)
        self.assertEqual(receipt["status"], "failed")

    def test_second_role_failure_preserves_completed_first_role_work(self):
        fit_receipt = deepcopy(self.receipt["roles"]["fit"])
        failure = data.DiagnosticAdmissionError("stopped", {"status": "failed", "counts": {"candidate_pairs": 2}})
        with patch.object(data, "build_split", side_effect=[(self.banks["fit"], fit_receipt), failure]):
            with self.assertRaises(data.DiagnosticAdmissionError) as caught:
                data.build_pair_of_splits(excluded_transcripts=(), roots=ROOTS, pairs_per_cell=1)
        self.assertEqual(caught.exception.receipt["roles"]["fit"], fit_receipt)
        self.assertEqual(caught.exception.receipt["roles"]["evaluation"]["counts"]["candidate_pairs"], 2)
        fit_receipt["counts"]["accepted_pairs"] = -1
        self.assertEqual(caught.exception.receipt["roles"]["fit"]["counts"]["accepted_pairs"], 63)

    def test_invalid_or_ambiguous_inputs_are_rejected(self):
        bad_options = ({"root_seed": True}, {"pairs_per_cell": True}, {"pairs_per_cell": 0},
                       {"cells": []}, {"cells": [CELL, CELL]}, {"cells": ["color/d2/advance/t8"]})
        for options in bad_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                data.build_split("fit", excluded_transcripts=(), **options)
        h = data.transcript_digest(self.banks["fit"][CELL][0])
        with self.assertRaises(ValueError):
            data.build_split("fit", excluded_transcripts=[h, h])
        with self.assertRaises(ValueError):
            data.build_pair_of_splits(excluded_transcripts=(), roots={"fit": ROOTS["fit"]})
        with patch.object(data, "build_split") as build:
            with self.assertRaises(ValueError):
                data.build_pair_of_splits(excluded_transcripts=(), roots={"fit": ROOTS["fit"], "evaluation": -1})
            build.assert_not_called()

    def test_outer_cell_inventory_is_snapshotted_before_first_role(self):
        cells, observed = [CELL], []
        fit = self.banks["fit"][CELL]
        evaluation = self.banks["evaluation"][CELL]
        def fake_build(role, **kwargs):
            observed.append(kwargs["cells"])
            cells.append("count/d0/direct/t8")
            rows = fit if role == "fit" else evaluation
            return {CELL: rows}, {"transcript_sha256": [data.transcript_digest(row) for row in rows],
                                 "production_contract": False}
        with patch.object(data, "build_split", side_effect=fake_build):
            data.build_pair_of_splits(excluded_transcripts=(), roots=ROOTS, pairs_per_cell=1, cells=cells)
        self.assertEqual(observed, [(CELL,), (CELL,)])


if __name__ == "__main__":
    unittest.main()
