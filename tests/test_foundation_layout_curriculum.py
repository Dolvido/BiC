"""Bounded CPU-only layout checks; do not run without the study work receipt.

One class fixture creates54 distinct canonical pairs (108 distinct episodes),
and four layouts per pair. Validation reconstructs additional copies of those
same parents and layouts; the shared ledger and call counter retain that work.
No Torch, neural forwards, optimizer, historical-bank or training operations.
"""
from collections import Counter
from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from experiments import foundation_layout_curriculum as layout


WORK = layout.WorkLedger()
TEST_WORK = dict(distinct_canonical_pair_draws=0, canonical_generate_pair_calls=0,
    canonical_rows_returned=0, explicitly_requested_layout_pairs=0,
    malformed_variants=0, handwritten_typed_pair_fixtures=0)


class FoundationLayoutCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.caller_rng = random.getstate()
        original_generate = layout.foundation.generate_pair
        def counted(*args, **kwargs):
            TEST_WORK["canonical_generate_pair_calls"] += 1
            pair = original_generate(*args, **kwargs)
            TEST_WORK["canonical_rows_returned"] += len(pair)
            return pair
        cls.generation_patch = patch.object(layout.foundation, "generate_pair", side_effect=counted)
        cls.generation_patch.start()
        cls.addClassCleanup(cls.generation_patch.stop)
        cls.base, cls.rows = {}, {}
        for family in layout.foundation.FAMILIES:
            for depth in layout.foundation.DEPTHS:
                for turns in layout.foundation.TURN_BUCKETS:
                    key = (family, depth, turns)
                    cls.base[key] = layout.foundation.generate_pair(family, 918240000 + depth*16 + turns,
                        depth=depth, turns=turns, naming_seed=918241000 + depth*16 + turns)
                    TEST_WORK["distinct_canonical_pair_draws"] += 1
                    for mode, seed in (("original", 0), ("varied", 0), ("varied", 1), ("varied", 2)):
                        cls.rows[(*key, mode, seed)] = layout.materialize_pair(cls.base[key], layout=mode, seed=seed, work=WORK)
                        TEST_WORK["explicitly_requested_layout_pairs"] += 1

    def test_original_control_is_exact_and_cannot_masquerade_as_foundation(self):
        for key, base in self.base.items():
            adapted = self.rows[(*key, "original", 0)]
            self.assertEqual(adapted[0]["recipe"]["permutation"], list(range(key[2])))
            for parent, row in zip(base, adapted):
                self.assertEqual(row["turns"], parent["turns"])
                self.assertEqual(row["version"], layout.VERSION)
                self.assertNotEqual(row["id"], parent["id"])
                self.assertEqual(row["anchor"]["turn_index"], key[2]-1)
            with self.assertRaises(ValueError):
                layout.foundation.validate_pair(adapted)

    def test_every_layout_regenerates_with_all_query_truth_and_version_preserved(self):
        for key, rows in self.rows.items():
            self.assertTrue(layout.validate_pair(rows, work=WORK))
            base = self.base[key[:3]]
            permutation = rows[0]["recipe"]["permutation"]
            self.assertEqual(permutation, rows[1]["recipe"]["permutation"])
            statement_ids = [index for index, turn in enumerate(base[0]["turns"]) if turn["kind"] == "statement"]
            self.assertEqual([index for index in permutation if index in statement_ids], statement_ids)
            for parent, row in zip(base, rows):
                self.assertEqual(Counter(map(layout._json, parent["turns"])), Counter(map(layout._json, row["turns"])))
                before = layout._trace(layout.foundation._program(parent), range(len(parent["turns"])))
                after = layout._trace(layout.foundation._program(row), permutation)
                self.assertEqual(before, after)
                self.assertEqual(layout.foundation.legacy.english_oracle(row), [turn["target"] for turn in row["turns"]])
                for query in row["queries"]:
                    identity, current = query["original_turn_index"], query["turn_index"]
                    self.assertEqual(permutation[current], identity)
                    self.assertEqual(query["version_signature"], before[identity])
                    self.assertEqual(row["turns"][current], parent["turns"][identity])
                anchor = row["anchor"]["turn_index"]
                self.assertEqual(row["turns"][anchor], parent["turns"][-1])
            self.assertEqual({row["turns"][row["anchor"]["turn_index"]]["target"] for row in rows}, {0, 1})

    def test_terminal_feasibility_and_depth_five_eight_turn_limit(self):
        for key in self.base:
            controls = self.rows[(*key, "original", 0)]
            support = controls[0]["recipe"]["feasible_terminal_kinds"]
            self.assertEqual(support[:2], ["anchor_last", "unknown_last"])
            self.assertIn(len(support), (2, 3))
            selected = set()
            for seed in range(3):
                rows = self.rows[(*key, "varied", seed)]
                kind = rows[0]["recipe"]["selected_terminal_kind"]
                selected.add(kind)
                self.assertEqual(kind, support[seed % len(support)])
                for row in rows:
                    final = row["turns"][-1]
                    if kind == "anchor_last":
                        self.assertEqual(row["anchor"]["turn_index"], key[2]-1)
                        self.assertIn(final["target"], (0, 1))
                    elif kind == "unknown_last":
                        self.assertEqual(final["target"], 2)
                        self.assertLess(row["anchor"]["turn_index"], key[2]-1)
                        identity = row["recipe"]["permutation"][-1]
                        self.assertIsNone(next(q for q in row["queries"] if q["original_turn_index"] == identity)["version_signature"]["read_version"])
                    else:
                        self.assertEqual(final["kind"], "statement")
                        self.assertIsNone(row["layout_summary"]["terminal_query_id"])
            self.assertEqual(selected, set(support))
            if key[1:] == (5, 8):
                self.assertEqual(support, ["anchor_last", "unknown_last"])
                row = self.rows[(*key, "varied", 1)][0]
                self.assertEqual(row["layout_summary"]["statements"], 6)
                self.assertEqual(row["layout_summary"]["queries"], 2)
                self.assertEqual(row["anchor"]["turn_index"], 6)
                anchor = next(q for q in row["queries"] if q["original_turn_index"] == 7)
                self.assertEqual(anchor["valid_statement_slots"], [6])

    def test_layout_selection_is_shared_across_families_and_values(self):
        for depth in layout.foundation.DEPTHS:
            for turns in layout.foundation.TURN_BUCKETS:
                for seed in range(3):
                    rows = [self.rows[(family, depth, turns, "varied", seed)][0] for family in layout.foundation.FAMILIES]
                    for other in rows[1:]:
                        self.assertEqual(rows[0]["recipe"]["permutation"], other["recipe"]["permutation"])
                        self.assertEqual(rows[0]["recipe"]["feasible_terminal_kinds"], other["recipe"]["feasible_terminal_kinds"])

    def test_copy_ownership_private_metadata_and_caller_rng(self):
        key = ("count", 5, 12)
        base = self.base[key]
        previous = layout._json(base)
        materialized = self.rows[(*key, "varied", 1)]
        saved = deepcopy(materialized)
        self.assertEqual(self.caller_rng, random.getstate())
        try:
            materialized[0]["turns"][0]["observations"]["visual"][0][0] = 99
            materialized[0]["recipe"]["base_recipe"]["seed"] = 0
            materialized[0]["queries"][0]["version_signature"]["query"]["name"] = "changed"
            self.assertEqual(layout._json(base), previous)
            self.assertNotEqual(materialized[0], saved[0])
            self.assertEqual(materialized[1], saved[1])
        finally:
            self.rows[(*key, "varied", 1)] = saved

    def test_sixteen_malformed_variants_fail_closed(self):
        base = self.rows[("color", 5, 8, "varied", 1)]
        mutations = [
            lambda rows: rows[0].update(version=layout.foundation.VERSION),
            lambda rows: rows[0].update(variant=False),
            lambda rows: rows.reverse(),
            lambda rows: rows[0].update(extra="unbound"),
            lambda rows: rows[0]["recipe"].update(seed=True),
            lambda rows: rows[0]["recipe"].update(layout="audit"),
            lambda rows: rows[0]["recipe"]["permutation"].__setitem__(0, True),
            lambda rows: rows[0]["recipe"].update(base_pair_sha256="0"*64),
            lambda rows: rows[0]["recipe"]["source_sha256"].update({"extra.py": "0"*64}),
            lambda rows: rows[0]["recipe"].update(feasible_terminal_kinds=["statement_last"]),
            lambda rows: rows[0]["anchor"].update(turn_index=7),
            lambda rows: rows[0]["queries"][0]["version_signature"].update(read_version=999),
            lambda rows: rows[0]["query_ancestries"][0].update(legacy_structure_partition="audit"),
            lambda rows: rows[0]["turns"][0]["observations"].update(tokens=[1]),
            lambda rows: rows[0]["turns"].reverse(),
            lambda rows: rows[1]["turns"].__setitem__(0, deepcopy(rows[0]["turns"][0])),
        ]
        for mutate in mutations:
            rows = deepcopy(base)
            mutate(rows)
            TEST_WORK["malformed_variants"] += 1
            with self.assertRaises(ValueError):
                layout.validate_pair(rows, work=WORK)

    def test_coincident_switch_parity_does_not_authorize_new_read_version(self):
        # Deliberately not a canonical lesson; exercises the relation check's
        # exact-version criterion when both independent target vectors agree.
        programs = []
        for initial in (False, True):
            programs.append([dict(op="set", name="dax", value=initial),
                dict(op="query", name="dax", value=False),
                dict(op="advance", name="dax", amount=1),
                dict(op="advance", name="dax", amount=1),
                dict(op="query", name="wug", value=True),
                dict(op="query", name="dax", value=False)])
        pair = []
        for variant, program in enumerate(programs):
            answers = layout.foundation.legacy.abstract_oracle(program, "switch")
            pair.append(dict(family="switch", variant=variant, turns=[dict(
                text=layout.foundation.legacy._sentence(event, "switch", {"dax": "dax", "wug": "wug"}),
                target=answer, reply=layout.foundation.REPLIES[answer]) for event, answer in zip(program, answers)]))
        TEST_WORK["handwritten_typed_pair_fixtures"] += 1
        permutation = [0, 2, 3, 1, 4, 5]
        rows = [dict(row, turns=[deepcopy(row["turns"][i]) for i in permutation]) for row in pair]
        for row in rows:
            self.assertEqual(layout.foundation.legacy.english_oracle(row), [t["target"] for t in row["turns"]])
        with self.assertRaisesRegex(ValueError, "exact read-version"):
            layout._check_relation(pair, rows, permutation, WORK)

    def test_copy_version_survives_later_source_overwrite(self):
        rows = []
        for value in (False, True):
            program = [dict(op="set", name="dax", value=value),
                dict(op="copy", source="dax", name="wug"),
                dict(op="set", name="dax", value=not value),
                dict(op="query", name="wug", value=False)]
            rows.append(dict(family="switch", turns=[dict(text=layout.foundation.legacy._sentence(
                event, "switch", {"dax": "dax", "wug": "wug"})) for event in program]))
        TEST_WORK["handwritten_typed_pair_fixtures"] += 1
        statements, slots = layout._legal_slots(rows, WORK)
        self.assertEqual(statements, [0, 1, 2])
        self.assertEqual(slots[3], [2, 3])


if __name__ == "__main__":
    unittest.main()
