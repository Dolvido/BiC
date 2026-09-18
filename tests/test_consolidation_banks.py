"""Original-only replay selection and prospectively sealed fresh panel boundaries."""
from collections import Counter
from copy import deepcopy
import unittest

from experiments import consolidation_banks as banks
from experiments.diverse_curriculum import render_world_pair
from experiments.diversity_study import world_pool, normalized_transcript


class ConsolidationReplayTests(unittest.TestCase):
    def source(self):
        return {family: [row for name in banks.TRAIN_NAME_SEEDS[:4]
            for world in world_pool(family, 811_000 + index * 1000, 8, "train")
            for row in render_world_pair(world, name)]
            for index, family in enumerate(banks.FAMILIES)}

    def test_replay_covers_worlds_levels_names_and_retains_exact_original_rows(self):
        source = self.source()
        original = deepcopy(source)
        selected = banks.build_replay_banks(source, max_pairs=8)
        for family, rows in selected.items():
            self.assertEqual(len(rows), 16)
            self.assertEqual(len({row["world_fingerprint"] for row in rows}), 8)
            self.assertEqual(Counter(row["level"] for row in rows), {1: 8, 2: 8})
            self.assertEqual(Counter(row["naming_seed"] for row in rows), dict.fromkeys(banks.TRAIN_NAME_SEEDS[:4], 4))
            for row in rows:
                self.assertIn(row, original[family])
            self.assertEqual(len({row["counterfactual_group"] for row in rows}), 8)
        self.assertEqual(source, original)
        selected[banks.FAMILIES[0]][0]["turns"][0]["text"] = "changed returned copy"
        self.assertEqual(source, original)

    def test_logical_repetition_does_not_enlarge_replay_or_change_selection(self):
        source = self.source()
        unique = banks.build_replay_banks(source, max_pairs=1000)
        repeated = banks.build_replay_banks({key: rows * 3 for key, rows in source.items()}, max_pairs=1000)
        self.assertEqual(unique, repeated)
        self.assertTrue(all(len(rows) == 64 for rows in unique.values()))

    def test_replay_rejects_nontraining_provenance_and_tampered_duplicates(self):
        source = self.source()
        family = banks.FAMILIES[0]
        bad = deepcopy(source)
        bad[family][-1]["turns"][0]["text"] += " "
        with self.assertRaises(ValueError):
            banks.build_replay_banks(bad)
        nontrain = deepcopy(source)
        world = world_pool(family, 92, 1, "dev")[0]
        nontrain[family][:2] = render_world_pair(world, 93, split="dev", name_split="dev")
        with self.assertRaisesRegex(ValueError, "training"):
            banks.build_replay_banks(nontrain)
        duplicate_variant = deepcopy(source)
        duplicate_variant[family][1] = deepcopy(duplicate_variant[family][0])
        with self.assertRaises(ValueError):
            banks.build_replay_banks(duplicate_variant)

    def test_manifest_distinguishes_worlds_maps_pairs_and_repeated_transcripts(self):
        replay = banks.build_replay_banks(self.source(), max_pairs=16)
        manifest = banks.bank_manifest({"replay": replay})["replay"]
        for family, rows in replay.items():
            value = manifest[family]
            self.assertEqual(value["episodes"], 32)
            self.assertEqual(value["world_pairs"], 8)
            self.assertEqual(value["naming_maps"], 4)
            self.assertEqual(value["counterfactual_pairs"], 16)
            self.assertEqual(sum(value["levels"].values()), len(rows))
            self.assertEqual(value["unique_normalized_transcripts"], len({normalized_transcript(row) for row in rows}))

    def test_bad_parent_contracts_and_pair_limits_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                banks.build_replay_banks({}, max_pairs=value)
        with self.assertRaises(ValueError):
            banks.build_replay_banks({})
        with self.assertRaises(ValueError):
            banks.prepare_banks({})


class ConsolidationFreshPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Regenerate the historical recipe, never depend on a saved result or model.
        from experiments.diversity_study import prepare_banks as parent_recipe
        cls.parent = parent_recipe()
        cls.prepared, cls.diagnostics = banks.prepare_banks(cls.parent, with_diagnostics=True)

    def normalized(self, rows):
        return {normalized_transcript(row) for row in rows}

    def test_counts_admission_and_reserved_unused_world_partition(self):
        prepared = self.prepared
        self.assertEqual({group: len(prepared[group]) for group in prepared},
            {"development": 9, "retained": 9, "advanced": 3, "support": 1, "query": 4})
        support = prepared["support"][banks.HELDOUT]
        self.assertEqual(len(support), 2048)
        self.assertEqual(len({row["world_fingerprint"] for row in support}), 128)
        self.assertEqual(len({row["naming_seed"] for row in support}), 8)
        self.assertEqual(Counter(row["level"] for row in support), {1: 1024, 2: 1024})
        self.assertTrue(all(row["split"] == row["world_partition"] == row["name_split"] == "train" for row in support))
        self.assertEqual({name: len(rows) for name, rows in prepared["query"].items()},
                         {"names": 128, "worlds": 256, "both": 256, "advanced": 256})
        advanced = prepared["query"]["advanced"]
        self.assertTrue(all(row["world_partition"] == "dev" and row["split"] == "audit"
                            and row["name_split"] == "audit" and row["level"] == 3 for row in advanced))
        self.assertIn("25,600", self.diagnostics["advanced_arithmetic_partition"]["reason"])

    def test_novel_world_pools_exclude_all_prior_and_earlier_new_individual_transcripts(self):
        banned = self.normalized(list(banks._parent_rows(self.parent)))
        for family in banks.FAMILIES:
            for group in ("development", "retained"):
                actual = self.normalized(self.prepared[group][f"worlds/{family}"])
                self.assertFalse(actual & banned)
                self.assertEqual(actual, self.normalized(self.prepared[group][f"both/{family}"]))
                banned.update(actual)
            actual = self.normalized(self.prepared["advanced"][family])
            self.assertFalse(actual & banned)
            banned.update(actual)
        support = self.normalized(self.prepared["support"][banks.HELDOUT])
        self.assertFalse(support & banned)
        banned.update(support)
        familiar = self.normalized(self.prepared["query"]["names"])
        self.assertTrue(familiar <= support)
        self.assertEqual(self.normalized(self.prepared["query"]["worlds"]),
                         self.normalized(self.prepared["query"]["both"]))
        for panel in ("worlds", "advanced"):
            actual = self.normalized(self.prepared["query"][panel])
            self.assertFalse(actual & banned)
            banned.update(actual)

    def test_all_renderings_are_exactly_new_even_when_worlds_are_familiar(self):
        prior = {banks._transcript(row) for row in banks._parent_rows(self.parent)}
        for values in self.prepared.values():
            for rows in values.values():
                actual = {banks._transcript(row) for row in rows}
                self.assertFalse(actual & prior)
                prior.update(actual)
        for family in banks.FAMILIES:
            common = {row["world_fingerprint"] for row in self.parent["train"]["w32-n8"][family]}
            for group in ("development", "retained"):
                rows = self.prepared[group][f"names/{family}"]
                self.assertEqual({row["world_fingerprint"] for row in rows}, common)


if __name__ == "__main__":
    unittest.main()
