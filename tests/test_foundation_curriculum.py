"""CPU-only canonical foundation generation and raw-observation packing tests."""
from copy import deepcopy
import json
import random
import unittest
from unittest.mock import patch

import torch

from experiments import composition_curriculum as legacy
from experiments import foundation_curriculum as curriculum
from experiments.composition_data import pack_composition_episodes


class FoundationCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_every_domain_depth_and_length_has_exact_counterfactual_truth(self):
        for family in curriculum.FAMILIES:
            for depth in curriculum.DEPTHS:
                for turns in curriculum.TURN_BUCKETS:
                    with self.subTest(family=family, depth=depth, turns=turns):
                        pair = curriculum.generate_pair(family, 402000100 + depth, depth=depth, turns=turns)
                        self.assertTrue(curriculum.validate_pair(pair))
                        self.assertEqual({row["turns"][-1]["target"] for row in pair}, {0, 1})
                        changed = [index for index, (a, b) in enumerate(zip(pair[0]["turns"], pair[1]["turns"]))
                                   if a["text"] != b["text"]]
                        self.assertEqual(len(changed), 1)
                        self.assertTrue(pair[0]["turns"][changed[0]]["text"].startswith("Set the "))
                        for row in pair:
                            self.assertEqual(row["version"], curriculum.VERSION)
                            self.assertEqual(len(row["turns"]), turns)
                            self.assertEqual(curriculum.final_depth(row), depth)
                            self.assertEqual(curriculum.query_ancestries(row), row["query_ancestries"])
                            self.assertEqual(curriculum.structure_id(row), row["structure_id"])
                            expected = [turn["target"] for turn in row["turns"]]
                            self.assertEqual(legacy.english_oracle(row), expected)
                            parsed = [legacy.parse_sentence(turn["text"])[1] for turn in row["turns"]]
                            self.assertEqual(legacy.abstract_oracle(parsed, family), expected)
                            self.assertIn(2, expected[:-1])
                            for turn in row["turns"]:
                                self.assertLessEqual(len(turn["text"].encode("utf-8")), 128)
                                self.assertEqual(turn["observations"]["tokens"], [0])
                                self.assertTrue(all(value == 0 for name, vectors in turn["observations"].items()
                                                    if name != "tokens" for vector in vectors for value in vector))

    def test_primitives_are_shared_not_falsely_withheld_structures(self):
        for depth in (0, 1):
            for split in ("train", "dev", "audit"):
                row = curriculum.generate_pair("color", 402001000 + depth, depth=depth, split=split)[0]
                self.assertEqual(row["split"], split)
                self.assertTrue(row["primitive_shared"])
                self.assertEqual(row["structure_partition"], "train")
                self.assertEqual(row["recipe"]["structure_split"], "shared")
                final = row["query_ancestries"][-1]
                self.assertEqual(final["structure_partition"], "shared")
                self.assertTrue(final["primitive_shared"])
                old_final = legacy.query_ancestries(row)[-1]
                self.assertEqual(final["legacy_structure_partition"], old_final["structure_partition"])
                self.assertEqual(final["structure_id"], old_final["structure_id"])
                with self.assertRaisesRegex(ValueError, "shared"):
                    curriculum.generate_pair("color", 402001000 + depth, depth=depth, split=split, structure_split=split)

    def test_depth_one_covers_copy_and_advance_with_one_shared_interface(self):
        seen = set()
        for seed in range(402002000, 402002050):
            row = curriculum.generate_pair("count", seed, depth=1)[0]
            program = [legacy.parse_sentence(turn["text"])[1] for turn in row["turns"]]
            signature = legacy._ancestry_records(program)[-1]["signature"]
            seen.update(event[0] for event in signature if event[0] in ("copy", "advance"))
        self.assertEqual(seen, {"copy", "advance"})

    def test_legacy_partition_identity_and_all_query_admission_are_preserved(self):
        train, dev, audit = set(), set(), set()
        for depth in range(2, 6):
            for turns in curriculum.TURN_BUCKETS:
                for split, seen in (("train", train), ("dev", dev), ("audit", audit)):
                    if depth == 2 and split == "dev":
                        continue
                    for seed in range(402003000, 402003008):
                        row = curriculum.generate_pair("switch", seed, depth=depth, turns=turns, split=split)[0]
                        self.assertFalse(row["primitive_shared"])
                        self.assertEqual(row["structure_partition"], split)
                        old = legacy.query_ancestries(row)
                        for new, prior in zip(row["query_ancestries"], old):
                            self.assertEqual(new["structure_id"], prior["structure_id"])
                            self.assertEqual(new["legacy_structure_partition"], prior["structure_partition"])
                            if new["composed"]:
                                if split == "train":
                                    self.assertEqual(new["legacy_structure_partition"], "train")
                                    train.add(new["structure_id"])
                                elif split == "dev":
                                    self.assertNotEqual(new["legacy_structure_partition"], "audit")
                        seen.add(row["structure_id"])
        self.assertFalse(train & audit)
        self.assertFalse(dev & audit)

    def test_unavailable_depth_two_dev_has_no_fallback_or_relabel(self):
        with self.assertRaisesRegex(ValueError, "no depth-two legacy dev motif"):
            curriculum.generate_pair("color", 402004000, depth=2, split="dev")
        familiar = curriculum.generate_pair("color", 402004000, depth=2, split="dev", structure_split="train")
        self.assertEqual(familiar[0]["structure_partition"], "train")
        self.assertEqual(familiar[0]["split"], "dev")
        with self.assertRaisesRegex(ValueError, "admission partition"):
            curriculum.generate_pair("count", 402004000, depth=2, split="dev", structure_split="audit")
        with self.assertRaisesRegex(ValueError, "admission partition"):
            curriculum.generate_pair("switch", 402004000, depth=3, split="train", structure_split="dev")

    def test_earlier_audit_query_is_rejected_even_with_training_final_ancestry(self):
        # Legacy advance->copy is audit, but the extra copy makes the final
        # ancestry train. Final-only validation would silently leak the earlier query.
        program = [{"op": "set", "name": "x0"}, {"op": "advance", "name": "x0"},
                   {"op": "copy", "name": "x2", "source": "x0"}, {"op": "query", "name": "x2"},
                   {"op": "copy", "name": "x3", "source": "x2"}, {"op": "query", "name": "x6"},
                   {"op": "set", "name": "x7"}, {"op": "query", "name": "x3"}]
        records = curriculum.query_ancestries(program)
        self.assertEqual(records[0]["legacy_structure_partition"], "audit")
        self.assertEqual(records[-1]["legacy_structure_partition"], "train")
        self.assertFalse(curriculum._admissible(records, "train"))
        self.assertFalse(curriculum._admissible(records, "dev"))
        self.assertTrue(curriculum._admissible(records, "audit"))
        with patch.object(curriculum, "MAX_ATTEMPTS", 3), patch.object(curriculum, "_sample", return_value=(program, 0)) as sampled:
            with self.assertRaisesRegex(ValueError, "within 3 deterministic attempts"):
                curriculum._procedure(402005000, 3, 8, "dev", "train")
            self.assertEqual(sampled.call_count, 3)

    def test_domain_names_values_and_familiar_admission_are_independent_factors(self):
        rows = [curriculum.generate_pair(family, 402006000, depth=4, turns=12)[0] for family in curriculum.FAMILIES]
        for key in ("structure_id", "program_id", "query_ancestries", "recipe", "depth"):
            self.assertEqual(rows[0][key], rows[1][key])
            self.assertEqual(rows[1][key], rows[2][key])
        original = curriculum.generate_pair("count", 402006100, depth=4, turns=12)
        renamed = curriculum.generate_pair("count", 402006100, depth=4, turns=12, naming_seed=402006101)
        revalued = curriculum.generate_pair("count", 402006100, depth=4, turns=12, value_seed=402006102)
        admitted = curriculum.generate_pair("count", 402006100, depth=4, turns=12, split="dev", structure_split="train")
        for pair in (renamed, revalued, admitted):
            self.assertTrue(curriculum.validate_pair(pair))
            for key in ("structure_id", "program_id", "query_ancestries", "depth"):
                self.assertEqual(pair[0][key], original[0][key])
        self.assertEqual([t["target"] for t in renamed[0]["turns"]], [t["target"] for t in original[0]["turns"]])
        self.assertNotEqual([t["text"] for t in renamed[0]["turns"]], [t["text"] for t in original[0]["turns"]])
        self.assertNotEqual([t["text"] for t in revalued[0]["turns"]], [t["text"] for t in original[0]["turns"]])
        self.assertEqual(original[0]["turns"], admitted[0]["turns"])

    def test_depth_tracks_immutable_ancestry_not_total_operations_or_turns(self):
        for turns in curriculum.TURN_BUCKETS:
            row = curriculum.generate_pair("count", 402007000, depth=1, turns=turns)[0]
            self.assertEqual(curriculum.final_depth(row), 1)
            self.assertGreater(len(row["turns"]), curriculum.final_depth(row) + 2)
        program = [{"op": "set", "name": "x0"}, {"op": "advance", "name": "x0"},
                   {"op": "copy", "name": "x1", "source": "x0"}, {"op": "set", "name": "x0"},
                   {"op": "advance", "name": "x7"}, {"op": "query", "name": "x1"}]
        self.assertEqual(curriculum.final_depth(program), 2)
        program.insert(-1, {"op": "set", "name": "x1"})
        self.assertEqual(curriculum.final_depth(program), 0)

    def test_packer_explicit_validator_keeps_only_text_in_inputs(self):
        pair = curriculum.generate_pair("color", 402008000, depth=1)
        packed = pack_composition_episodes(pair, training=True, pair_validator=curriculum.validate_pair)
        self.assertEqual(set(packed["inputs"]), {"token_ids", "valid_mask", "lengths", "eos_positions"})
        self.assertEqual(tuple(packed["supervision"]["action_targets"].shape), (2, 8))
        with self.assertRaises(ValueError):
            pack_composition_episodes(pair, training=True)  # legacy admission is not interchangeable
        modified = deepcopy(pair)
        for row in modified:
            row["depth"] = 999
            row["recipe"] = {"secret": "metadata is not neural input"}
            for turn in row["turns"]:
                turn["target"] = 0
                turn["reply"] = curriculum.REPLIES[0]
        # Structural packing does not consult the oracle; explicit validation is
        # the required training admission boundary. Manipulated labels stay out.
        with patch.object(curriculum, "validate_pair", side_effect=AssertionError("oracle during packing")), \
             patch.object(legacy, "english_oracle", side_effect=AssertionError("oracle during packing")):
            other = pack_composition_episodes(modified, training=False)
        for name in packed["inputs"]:
            self.assertTrue(torch.equal(packed["inputs"][name], other["inputs"][name]))
        with self.assertRaises(ValueError):
            pack_composition_episodes(modified, training=True, pair_validator=curriculum.validate_pair)

    def test_strict_types_bounds_and_scope_reject_boolean_aliases(self):
        for options in ({"depth": True}, {"depth": -1}, {"depth": 6}, {"depth": 1, "turns": True},
                        {"depth": 1, "turns": 9}, {"depth": 1, "naming_seed": False},
                        {"depth": 1, "value_seed": float("nan")}, {"depth": 1, "split": "validation"},
                        {"depth": 1, "structure_split": "audit"}, {"depth": 3, "structure_split": "shared"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                curriculum.generate_pair("color", 402009000, **options)
        for seed in (True, -1, 2**63, 1.2, None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                curriculum.generate_pair("color", seed, depth=0)
        with self.assertRaises(ValueError):
            curriculum.generate_pair("new_subject", 402009000, depth=0)

    def test_regeneration_rejects_metadata_text_targets_and_cross_pair_tampering(self):
        pair = curriculum.generate_pair("count", 402010000, depth=3)
        mutations = [lambda r: r.update(depth=True), lambda r: r.update(variant=False),
                     lambda r: r.update(primitive_shared=True), lambda r: r.update(version=legacy.VERSION),
                     lambda r: r["recipe"].update(procedure_attempt=True),
                     lambda r: r["recipe"].update(value_seed=r["recipe"]["value_seed"] + 1),
                     lambda r: r["turns"][-1].update(target=2, reply=curriculum.REPLIES[2]),
                     lambda r: r["turns"][0].update(text=r["turns"][0]["text"] + " "),
                     lambda r: r["turns"][0]["observations"].update(tokens=[1]),
                     lambda r: r["query_ancestries"][-1].update(legacy_structure_partition="audit"),
                     lambda r: r.update(structure_id="0" * 64), lambda r: r.update(extra="unbound metadata")]
        for mutation in mutations:
            row = deepcopy(pair[0]); mutation(row)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                curriculum.validate_row(row)
        for rows in ([pair[0]], pair[::-1], [pair[0], pair[0]],
                     [pair[0], curriculum.generate_pair("count", 402010001, depth=3)[1]]):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                curriculum.validate_pair(rows)

    def test_generator_cache_and_caller_rng_are_isolated(self):
        before = random.getstate()
        pair = curriculum.generate_pair("switch", 402011000, depth=5, turns=12)
        original = json.dumps(pair, sort_keys=True)
        pair[0]["turns"][0]["observations"]["visual"][0][0] = 999
        pair[0]["query_ancestries"][-1]["depth"] = 999
        regenerated = curriculum.generate_pair("switch", 402011000, depth=5, turns=12)
        self.assertEqual(json.dumps(regenerated, sort_keys=True), original)
        self.assertEqual(before, random.getstate())
        self.assertTrue(curriculum.validate_pair(json.loads(original)))

    def test_exhaustion_is_finite_and_never_returns_relabeled_fallback(self):
        with patch.object(curriculum, "MAX_ATTEMPTS", 4), patch.object(curriculum, "_sample", return_value=None) as sampled:
            with self.assertRaisesRegex(ValueError, "within 4 deterministic attempts"):
                curriculum.generate_pair("color", 402012000, depth=4)
            self.assertEqual(sampled.call_count, 4)


if __name__ == "__main__":
    unittest.main()
