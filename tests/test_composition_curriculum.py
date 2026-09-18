"""Independent typed truth and causally partitioned shared composition recipes."""
from copy import deepcopy
import json
import random
import unittest
from unittest.mock import patch

from experiments import composition_curriculum as curriculum


def event(op, name, **fields):
    return {"op": op, "name": name, **fields}


def text_row(family, texts):
    return {"family": family, "turns": [{"text": text} for text in texts]}


class CompositionTruthTests(unittest.TestCase):
    def test_color_cycle_copy_snapshot_and_unknown_erasure(self):
        program = [event("set", "a", value="yellow"), event("copy", "b", source="a"),
            event("advance", "b", amount=1), event("set", "a", value="blue"),
            event("query", "b", value="red"), event("copy", "b", source="missing"),
            event("advance", "b", amount=1), event("query", "b", value="red")]
        row = text_row("color", ["Set the color of dax to yellow.", "Copy the color of dax to wug.",
            "Move the color of wug one step in cycle red, green, blue, yellow.",
            "Set the color of dax to blue.", "Is the color of wug red?", "Copy the color of fep to wug.",
            "Move the color of wug one step in cycle red, green, blue, yellow.", "Is the color of wug red?"])
        expected = [3, 3, 3, 3, 1, 3, 3, 2]
        self.assertEqual(curriculum.abstract_oracle(program, "color"), expected)
        self.assertEqual(curriculum.english_oracle(row), expected)

    def test_count_updates_and_copied_versions(self):
        program = [event("set", "a", value=8), event("advance", "a", amount=-3),
            event("copy", "b", source="a"), event("advance", "a", amount=2),
            event("query", "b", value=5), event("query", "a", value=5)]
        row = text_row("count", ["Set the count of dax to 8.", "Decrease the count of dax by 3.",
            "Copy the count of dax to wug.", "Increase the count of dax by 2.",
            "Is the count of wug 5?", "Is the count of dax 5?"])
        self.assertEqual(curriculum.abstract_oracle(program, "count"), [3, 3, 3, 3, 1, 0])
        self.assertEqual(curriculum.english_oracle(row), [3, 3, 3, 3, 1, 0])
        for initial, amount in ((0, -1), (99, 1)):
            with self.assertRaises(ValueError):
                curriculum.abstract_oracle([event("set", "a", value=initial), event("advance", "a", amount=amount)], "count")
        with self.assertRaises(ValueError):
            curriculum.english_oracle(text_row("count", ["Set the count of dax to 0.", "Decrease the count of dax by 1."]))

    def test_switch_toggling_unknown_advance_and_copy(self):
        program = [event("advance", "a", amount=1), event("query", "a", value=False),
            event("set", "a", value=True), event("copy", "b", source="a"),
            event("advance", "a", amount=1), event("query", "b", value=True),
            event("advance", "b", amount=1), event("query", "b", value=True)]
        row = text_row("switch", ["Toggle the switch of dax.", "Is the switch of dax off?",
            "Set the switch of dax to on.", "Copy the switch of dax to wug.",
            "Toggle the switch of dax.", "Is the switch of wug on?", "Toggle the switch of wug.", "Is the switch of wug on?"])
        self.assertEqual(curriculum.abstract_oracle(program, "switch"), [3, 2, 3, 3, 3, 1, 3, 0])
        self.assertEqual(curriculum.english_oracle(row), [3, 2, 3, 3, 3, 1, 3, 0])

    def test_text_oracle_does_not_call_abstract_oracle_or_use_claimed_answers(self):
        row = text_row("count", ["Set the count of dax to 5.", "Is the count of dax 5?"])
        for turn in row["turns"]:
            turn.update(target=0, reply="wrong", observations={"secret": 17})
        with patch.object(curriculum, "_abstract_run", side_effect=AssertionError("shared simulator")):
            self.assertEqual(curriculum.english_oracle(row), [3, 1])
        with patch.object(curriculum, "parse_sentence", side_effect=AssertionError("English parser")):
            self.assertEqual(curriculum.abstract_oracle([event("set", "a", value=3), event("query", "a", value=3)], "count"), [3, 1])

    def test_exact_grammar_rejects_ambiguous_unknown_and_mixed_domain_text(self):
        for text in ("Set the count of dax to 05.", "Set the count of dax to 100.",
                     "Set the color of dax to violet.", "Set the switch of dax to maybe.",
                     "Set the count of stranger to 5.", "Is the color of dax red? ", "x" * 129, None):
            with self.assertRaises(ValueError):
                curriculum.parse_sentence(text)
        with self.assertRaises(ValueError):
            curriculum.english_oracle(text_row("color", ["Set the count of dax to 3."]))


class CompositionStructureTests(unittest.TestCase):
    def chain(self):
        return [event("set", "a", value=4), event("copy", "b", source="a"),
                event("advance", "b", amount=2), event("query", "b", value=6)]

    def test_ancestry_ignores_distractors_queries_constants_and_names(self):
        chain = self.chain()
        decorated = [event("set", "unused", value=99), chain[0],
            event("query", "unused", value=99), chain[1], event("set", "a", value=17),
            event("advance", "unused", amount=-1), chain[2], chain[3]]
        self.assertEqual(curriculum.structure_id(chain), curriculum.structure_id(decorated))
        renamed = [event("set", "p", value="red"), event("copy", "q", source="p"),
                   event("advance", "q", amount=1), event("query", "q", value="green")]
        self.assertEqual(curriculum.structure_id(chain), curriculum.structure_id(renamed))

    def test_ancestry_tracks_copy_snapshots_overwrites_and_role_reuse(self):
        chain = self.chain()
        overwrite = [chain[0], chain[1], event("set", "b", value=4), chain[2], chain[3]]
        self.assertNotEqual(curriculum.structure_id(chain), curriculum.structure_id(overwrite))
        self.assertFalse(curriculum.query_ancestries(overwrite)[0]["composed"])
        erased = [chain[0], chain[1], event("copy", "b", source="missing"), chain[2], chain[3]]
        self.assertFalse(curriculum.query_ancestries(erased)[0]["known"])
        with self.assertRaises(ValueError):
            curriculum.structure_id(erased)
        reused = [chain[0], chain[1], event("copy", "a", source="b"), event("advance", "a", amount=1), event("query", "a", value=5)]
        distinct = [chain[0], chain[1], event("copy", "c", source="b"), event("advance", "c", amount=1), event("query", "c", value=5)]
        self.assertNotEqual(curriculum.structure_id(reused), curriculum.structure_id(distinct))

    def test_shared_procedure_is_identical_across_domains(self):
        for turns in (8, 10, 12):
            for split in curriculum.SPLITS:
                rows = [curriculum.generate_pair(family, 51, turns=turns, split=split)[0] for family in curriculum.FAMILIES]
                for key in ("structure_id", "program_id", "query_ancestries", "recipe"):
                    self.assertEqual(rows[0][key], rows[1][key])
                    self.assertEqual(rows[1][key], rows[2][key])

    def test_all_supervised_train_compositions_are_disjoint_from_held_final_motifs(self):
        train_ids, held_ids = set(), set()
        for turns in (8, 10, 12):
            for seed in range(100):
                row = curriculum.generate_pair("color", seed, turns=turns)[0]
                for query in row["query_ancestries"]:
                    if query["known"] and query["composed"]:
                        self.assertEqual(query["structure_partition"], "train")
                        train_ids.add(query["structure_id"])
                for split in ("dev", "audit"):
                    held = curriculum.generate_pair("count", seed, turns=turns, split=split)[0]
                    self.assertEqual(held["structure_partition"], split)
                    held_ids.add(held["structure_id"])
        self.assertFalse(train_ids & held_ids)
        self.assertGreater(len(train_ids), 100)
        self.assertGreater(len(held_ids), 100)


class CompositionGenerationTests(unittest.TestCase):
    def test_generated_pairs_truth_provenance_and_zero_observations(self):
        for family in curriculum.FAMILIES:
            for turns in (8, 10, 12):
                for split in curriculum.SPLITS:
                    for seed in (3, 19, 85):
                        pair = curriculum.generate_pair(family, seed, turns=turns, split=split)
                        self.assertTrue(curriculum.validate_pair(pair))
                        self.assertEqual({row["turns"][-1]["target"] for row in pair}, {0, 1})
                        self.assertEqual(len(pair[0]["turns"]), turns)
                        for row in pair:
                            self.assertEqual(curriculum.english_oracle(row), [turn["target"] for turn in row["turns"]])
                            self.assertEqual(curriculum.structure_id(row), row["structure_id"])
                            self.assertEqual(curriculum.query_ancestries(row), row["query_ancestries"])
                            self.assertIn(2, [turn["target"] for turn in row["turns"][:-1]])
                            self.assertTrue(any(turn["target"] in (0, 1) for turn in row["turns"][:-1]))
                            for turn in row["turns"]:
                                self.assertLessEqual(len(turn["text"].encode()), 128)
                                self.assertEqual(turn["observations"]["tokens"], [0])
                                self.assertTrue(all(v == 0 for k,x in turn["observations"].items() if k != "tokens" for v in x[0]))
                        changed = [(a,b) for a,b in zip(pair[0]["turns"],pair[1]["turns"]) if a["text"] != b["text"]]
                        self.assertEqual(len(changed),1)
                        self.assertTrue(changed[0][0]["text"].startswith("Set the "))

    def test_unknown_positions_vary_and_final_probe_is_known(self):
        for turns in (8, 10, 12):
            positions = set()
            final_ids = set()
            for seed in range(160):
                row = curriculum.generate_pair("switch", seed, turns=turns)[0]
                positions.update(i for i,t in enumerate(row["turns"]) if t["target"] == 2)
                final_ids.add(row["structure_id"])
            self.assertEqual(positions, set(range(turns - 1)))
            self.assertGreater(len(final_ids), 10)

    def test_value_naming_and_admission_factors_leave_structure_fixed(self):
        original = curriculum.generate_pair("count", 114, turns=12)
        renamed = curriculum.generate_pair("count", 114, turns=12, naming_seed=991)
        revalued = curriculum.generate_pair("count", 114, turns=12, value_seed=992)
        familiar_eval = curriculum.generate_pair("count", 114, turns=12, split="audit", structure_split="train")
        for pair in (renamed, revalued, familiar_eval):
            self.assertTrue(curriculum.validate_pair(pair))
            self.assertEqual(pair[0]["structure_id"], original[0]["structure_id"])
            self.assertEqual(pair[0]["program_id"], original[0]["program_id"])
        self.assertEqual([[t["target"] for t in r["turns"]] for r in renamed], [[t["target"] for t in r["turns"]] for r in original])
        self.assertEqual(set(curriculum.naming_map(991)), set(curriculum.ROLES))
        self.assertEqual(set(curriculum.naming_map(991).values()), set(curriculum.ALIASES))
        with self.assertRaises(ValueError):
            curriculum.generate_pair("color", 4, split="train", structure_split="audit")

    def test_exact_authentication_rejects_metadata_truth_and_sensor_tampering(self):
        original = curriculum.generate_pair("color", 101)[0]
        mutations = [lambda r:r.update(extra="hidden"), lambda r:r.update(variant=True),
            lambda r:r.update(structure_partition="dev"), lambda r:r.update(structure_id="bad"),
            lambda r:r["recipe"].update(value_seed=9), lambda r:r["recipe"].update(procedure_attempt=True),
            lambda r:r["query_ancestries"][-1].update(structure_partition="audit"),
            lambda r:r["turns"][0].update(text=r["turns"][0]["text"]+" "),
            lambda r:r["turns"][-1].update(target=not bool(r["turns"][-1]["target"])),
            lambda r:r["turns"][0].update(reply="wrong"),
            lambda r:r["turns"][0]["observations"]["tokens"].__setitem__(0, True),
            lambda r:r["turns"][0]["observations"]["visual"][0].__setitem__(0,float("nan"))]
        for mutate in mutations:
            bad=deepcopy(original)
            mutate(bad)
            with self.assertRaises(ValueError):
                curriculum.validate_row(bad)
        pair=curriculum.generate_pair("color",101)
        for bad in ([pair[1],pair[0]],[pair[0],pair[0]],[pair[0]]):
            with self.assertRaises(ValueError):
                curriculum.validate_pair(bad)

    def test_deterministic_json_independent_storage_and_invalid_requests(self):
        random.seed(22)
        state=random.getstate()
        pair=curriculum.generate_pair("color",103)
        self.assertEqual(state,random.getstate())
        self.assertEqual(pair,curriculum.generate_pair("color",103))
        self.assertEqual(pair,json.loads(json.dumps(pair)))
        untouched=deepcopy(pair[1])
        pair[0]["turns"][0]["observations"]["body"][0][0]=8
        self.assertEqual(pair[1],untouched)
        self.assertNotEqual(pair,curriculum.generate_pair("color",103))
        for values in ({"seed":True},{"seed":-1},{"turns":6},{"turns":True},{"family":"graph"},
                       {"split":"test"},{"value_seed":-1},{"naming_seed":False},{"structure_split":"test"}):
            with self.assertRaises(ValueError):
                curriculum.generate_pair(**{"family":"count","seed":0,**values})


if __name__ == "__main__":
    unittest.main()
