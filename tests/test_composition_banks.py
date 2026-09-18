"""Prospective bank boundaries without model evaluation or training."""
import copy
import unittest

from experiments import composition_banks as banks
from experiments import composition_curriculum as curriculum


class CompositionBanksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values, cls.diagnostics = banks.prepare_banks(
            train_pairs=8, panel_pairs=4, with_diagnostics=True)

    def test_counts_actual_lengths_and_training_admission(self):
        self.assertEqual(set(self.values), {"train", "development", "audit"})
        self.assertEqual(set(self.values["train"]), set(banks.FAMILIES))
        for family, buckets in self.values["train"].items():
            self.assertEqual(set(buckets), set(banks.LENGTHS))
            for turns, rows in buckets.items():
                self.assertEqual(len(rows), 16)
                for row in rows:
                    self.assertEqual(row["family"], family)
                    self.assertEqual(len(row["turns"]), turns)
                    self.assertEqual(row["split"], "train")
                    self.assertEqual(row["structure_partition"], "train")
        for group in ("development", "audit"):
            self.assertEqual(len(self.values[group]), 18)
            self.assertTrue(all(len(rows) == 8 for rows in self.values[group].values()))

    def test_seen_panels_keep_first_training_recipes_with_fresh_values_and_names(self):
        for group, split in (("development", "dev"), ("audit", "audit")):
            for family in banks.FAMILIES:
                for turns in banks.LENGTHS:
                    old = self.values["train"][family][turns]
                    seen = self.values[group][f"seen/{family}/t{turns}"]
                    for index in range(0, len(seen), 2):
                        self.assertEqual(old[index]["structure_id"], seen[index]["structure_id"])
                        self.assertEqual(old[index]["program_id"], seen[index]["program_id"])
                        self.assertEqual(old[index]["recipe"]["seed"], seen[index]["recipe"]["seed"])
                        for seed in ("naming_seed", "value_seed"):
                            self.assertNotEqual(old[index]["recipe"][seed], seen[index]["recipe"][seed])
                        self.assertEqual(seen[index]["split"], split)
                        self.assertEqual(seen[index]["structure_partition"], "train")

    def test_global_supervised_motif_boundary_includes_every_query(self):
        boundary = self.diagnostics["boundaries"]
        training = set(boundary["training_known_composed_query_ids"])
        development = set(boundary["development_known_composed_query_ids"])
        self.assertFalse(training & set(boundary["development_final_composed_ids"]))
        self.assertFalse((training | development) & set(boundary["audit_final_composed_ids"]))
        for rows in self.values["development"].values():
            for row in rows:
                self.assertFalse(any(query["structure_partition"] == "audit"
                    for query in banks.known_composed_queries(row)))
        self.assertTrue(boundary["supervised_composed_query_boundary_verified"])

    def test_exact_transcripts_are_globally_disjoint_and_pairs_intact(self):
        all_banks = [rows for buckets in self.values["train"].values() for rows in buckets.values()]
        all_banks += list(self.values["development"].values()) + list(self.values["audit"].values())
        observations = []
        for rows in all_banks:
            observations.extend(map(banks.transcript, rows))
            for index in range(0, len(rows), 2):
                self.assertTrue(curriculum.validate_pair(rows[index:index + 2]))
        self.assertEqual(len(observations), len(set(observations)))
        self.assertEqual(len(observations), self.diagnostics["boundaries"]["unique_exact_transcripts"])

    def test_manifest_reports_actual_coverage_targets_and_capacity(self):
        manifest = banks.bank_manifest(self.values)
        for family in banks.FAMILIES:
            for turns in banks.LENGTHS:
                row = manifest["train"][family][str(turns)]
                self.assertEqual(row["episodes"], 16)
                self.assertEqual(row["complete_pairs"], 8)
                self.assertEqual(row["target_counts_by_turn"][-1], {"0": 8, "1": 8, "2": 0, "3": 0})
                self.assertEqual(row["distinct_final_structures"], len(row["final_structure_ids"]))
                self.assertEqual(row["distinct_known_composed_queries"], len(row["known_composed_query_ids"]))
                self.assertLessEqual(row["max_context_tokens"], 1024)
                self.assertLessEqual(row["max_utterance_bytes"], 128)
                self.assertEqual(row["observation_tokens"] - row["observation_utf8_bytes"], 16 * turns * 2)

    def test_reproduction_and_mutation_isolation(self):
        again, diagnostic = banks.prepare_banks(train_pairs=8, panel_pairs=4, with_diagnostics=True)
        self.assertEqual(again, self.values)
        self.assertEqual(diagnostic, self.diagnostics)
        again["train"]["color"][8][0]["turns"][0]["text"] = "changed"
        self.assertNotEqual(again, self.values)

    def test_boundary_recheck_rejects_duplicate_or_tampered_observations(self):
        duplicate = copy.deepcopy(self.values)
        rows = duplicate["train"]["color"][8]
        rows[2:4] = copy.deepcopy(rows[:2])
        with self.assertRaisesRegex(ValueError, "transcript reused"):
            banks.verify_boundaries(duplicate)
        tampered = copy.deepcopy(self.values)
        tampered["audit"]["composed/count/t12"][0]["turns"][-1]["target"] = 2
        with self.assertRaises(ValueError):
            banks.verify_boundaries(tampered)

    def test_invalid_preparation_counts_fail_before_generation(self):
        for values in ({"train_pairs": True}, {"panel_pairs": 0},
                       {"train_pairs": 2, "panel_pairs": 3}, {"with_diagnostics": 1}):
            with self.assertRaises(ValueError):
                banks.prepare_banks(**values)

    def test_boundary_recheck_requires_all_nonempty_declared_panels(self):
        for operation in (
                lambda b: b["audit"].pop("composed/color/t8"),
                lambda b: b["train"]["switch"].pop(12),
                lambda b: b["development"].update({"seen/count/t10": []})):
            changed = copy.deepcopy(self.values)
            operation(changed)
            with self.assertRaises(ValueError):
                banks.verify_boundaries(changed)


if __name__ == "__main__":
    unittest.main()
