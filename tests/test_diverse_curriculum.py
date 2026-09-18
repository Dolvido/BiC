"""Independent truth, counterfactuals and exact v3 admission boundaries."""
from collections import Counter
from copy import deepcopy
import json
import random
import unittest

import torch

from experiments.cognitive_curriculum import cognitive_oracle, validate_cognitive
from experiments.diverse_curriculum import (
    ACK, ALLOW, ASK, DENY, ALIASES, FAMILIES, ROLES, VERSION,
    abstract_oracle, allowed_entity_pairs, generate_diverse,
    generate_world_pair, naming_map, render_world_pair,
    validate_diverse, validate_pair, world_fingerprint, world_split,
)
from experiments.sequence_data import pack_cognitive_episodes


def event(op, **fields):
    return {"op": op, **fields}


class DiverseTruthTests(unittest.TestCase):
    def check_truth(self, family, program, sentences, expected):
        self.assertEqual(abstract_oracle(program, family), expected)
        self.assertEqual(cognitive_oracle({"family": family, "turns": [{"text": s} for s in sentences]}), expected)

    def test_binding_snapshot_copy_and_unknown_source(self):
        self.check_truth("variable_binding", [
            event("color_set", name="x0", color="red"),
            event("color_copy", name="x1", source="x0"),
            event("color_set", name="x0", color="blue"),
            event("color_query", name="x1", color="red"),
            event("color_copy", name="x1", source="x2"),
            event("color_query", name="x1", color="red"),
        ], ["dax is red.", "Set wug to the color of dax.", "dax is blue.",
            "Is wug red?", "Set wug to the color of fep.", "Is wug red?"],
            [ACK, ACK, ACK, ALLOW, ACK, ASK])

    def test_arithmetic_updates_copy_and_missing_count(self):
        self.check_truth("arithmetic_updates", [
            event("count_set", name="x0", number=7),
            event("count_subtract", name="x0", number=2),
            event("count_copy", name="x1", source="x0"),
            event("count_add", name="x0", number=3),
            event("count_query", name="x1", number=5),
            event("count_query", name="x2", number=0),
        ], ["dax starts with a count of 7.", "Decrease the count of dax by 2.",
            "Set the count of wug to the count of dax.", "Increase the count of dax by 3.",
            "Is the count of wug 5?", "Is the count of fep 0?"], [ACK, ACK, ACK, ACK, ALLOW, ASK])
        with self.assertRaisesRegex(ValueError, "negative"):
            abstract_oracle([event("count_set", name="x0", number=0),
                             event("count_subtract", name="x0", number=1)], "arithmetic_updates")

    def test_graph_transitivity_cycles_and_closed_world(self):
        self.check_truth("graph_reachability", [
            event("links", edges=[["x0", "x1"], ["x1", "x2"]]),
            event("path_query", source="x0", destination="x2"),
            event("path_query", source="x2", destination="x0"),
            event("links", edges=[["x2", "x1"]]),
            event("close_links"),
            event("path_query", source="x2", destination="x0"),
        ], ["There are links from dax to wug and from wug to fep.",
            "Can a path follow links from dax to fep?", "Can a path follow links from fep to dax?",
            "There is a link from fep to wug.", "These are all the links.",
            "Can a path follow links from fep to dax?"], [ACK, ALLOW, ASK, ACK, ACK, DENY])

    def test_boolean_mixed_values_and_intervention(self):
        self.check_truth("conditional_logic", [
            event("switch_pair", name="x0", other="x1", value=False),
            event("switch_set", name="x3", value=True),
            event("rule", name="x2", operator="all", inputs=["x0", "x3"]),
            event("lamp_query", name="x2"),
            event("intervention", name="x0", value=True),
            event("lamp_query", name="x2"),
        ], ["Switch dax and switch wug are off.", "Switch zot is on.",
            "Lamp fep is on exactly when all of dax, zot are on.", "Is lamp fep on?",
            "Set switch dax to on.", "Is lamp fep on?"], [ACK, ACK, ACK, DENY, ACK, ALLOW])
        missing = [event("switch_set", name="x0", value=False),
                   event("rule", name="x2", operator="all", inputs=["x0", "x1"]),
                   event("lamp_query", name="x2"),
                   event("rule", name="x2", operator="any", inputs=["x0", "x1"]),
                   event("lamp_query", name="x2")]
        self.assertEqual(abstract_oracle(missing, "conditional_logic"), [ACK, ACK, DENY, ACK, ASK])


class DiverseGenerationTests(unittest.TestCase):
    def test_all_families_partitions_levels_are_verified_complete_pairs(self):
        for family in FAMILIES:
            for level in (1, 2, 3):
                for split in ("train", "dev", "audit"):
                    for seed in (90, 172, 241):
                        with self.subTest(family=family, level=level, split=split, seed=seed):
                            world = generate_world_pair(family, seed, level, split)
                            pair = render_world_pair(world, 70, split)
                            self.assertTrue(validate_pair(pair))
                            self.assertEqual(world["world_partition"], split)
                            self.assertEqual(world["world_fingerprint"], world_fingerprint(world))
                            for variant, row in enumerate(pair):
                                self.assertEqual(row["version"], VERSION)
                                self.assertEqual([t["target"] for t in row["turns"]], abstract_oracle(world["programs"][variant], family))
                                self.assertTrue(all(len(t["text"].encode()) <= 128 for t in row["turns"]))
                                for role in ("x0", "x1"):
                                    self.assertIn(naming_map(70, split)[role], " ".join(t["text"] for t in row["turns"]))
                            for a, b in zip(pair[0]["turns"], pair[1]["turns"]):
                                if a["target"] != ACK:
                                    self.assertEqual(a["text"], b["text"])

    def test_determinism_independent_storage_and_serialization(self):
        random.seed(400)
        before = random.getstate()
        original = generate_world_pair("variable_binding", 95, 3, "train")
        rows = render_world_pair(original, 20)
        self.assertEqual(before, random.getstate())
        self.assertEqual(rows, render_world_pair(original, 20))
        self.assertEqual(rows, json.loads(json.dumps(rows)))
        original["programs"][0][0]["color"] = "bad"
        self.assertNotEqual(original, generate_world_pair("variable_binding", 95, 3, "train"))
        rows[0]["turns"][0]["observations"]["visual"][0][0] = 8
        self.assertEqual(rows[1]["turns"][0]["observations"]["visual"][0][0], 0)

    def test_names_are_bijections_and_independent_of_world(self):
        support = {split: set(allowed_entity_pairs(split)) for split in ("train", "dev", "audit")}
        for split in support:
            for seed in range(20):
                names = naming_map(seed, split)
                self.assertEqual(set(names), set(ROLES))
                self.assertEqual(set(names.values()), set(ALIASES))
                self.assertIn((names["x0"], names["x1"]), support[split])
            for other in support:
                if split != other:
                    self.assertFalse(support[split] & support[other])
        for family in FAMILIES:
            world = generate_world_pair(family, 13, 2, "train")
            first, second = render_world_pair(world, 99), render_world_pair(world, 133)
            self.assertEqual(first[0]["world_fingerprint"], second[0]["world_fingerprint"])
            self.assertEqual([[t["target"] for t in r["turns"]] for r in first], [[t["target"] for t in r["turns"]] for r in second])
            self.assertNotEqual(first[0]["counterfactual_group"], second[0]["counterfactual_group"])

    def test_world_identity_ignores_variant_order_and_consistent_role_renaming(self):
        world = generate_world_pair("conditional_logic", 12, 2)
        renamed = deepcopy(world)
        role_map = dict(zip(ROLES, reversed(ROLES)))
        def rename(value):
            if isinstance(value, str):
                return role_map.get(value, value)
            if isinstance(value, list):
                return [rename(v) for v in value]
            if isinstance(value, dict):
                return {k: rename(v) for k, v in value.items()}
            return value
        renamed["programs"] = rename(list(reversed(renamed["programs"])))
        renamed.update(level=500, seed=888)
        self.assertEqual(world_fingerprint(renamed), world_fingerprint(world))
        self.assertEqual(world_split(renamed), world_split(world))
        with self.assertRaises(ValueError):
            render_world_pair(renamed, 10, "dev")

    def test_crossed_evaluation_panels_but_strict_train_admission(self):
        worlds = {split: generate_world_pair("conditional_logic", 72, 2, split) for split in ("train", "dev", "audit")}
        for world_split_name, world in worlds.items():
            for name_split in worlds:
                for admission in ("dev", "audit"):
                    rows = render_world_pair(world, 12, admission, name_split)
                    self.assertTrue(validate_pair(rows))
                    self.assertEqual(rows[0]["world_partition"], world_split_name)
                    self.assertEqual(rows[0]["name_split"], name_split)
                if (world_split_name, name_split) != ("train", "train"):
                    with self.assertRaises(ValueError):
                        render_world_pair(world, 12, "train", name_split)

    def test_exact_authentication_rejects_text_labels_sensors_and_metadata_tampering(self):
        row = generate_diverse(50, 2, family="variable_binding")[0]
        changes = [lambda r: r.update(world_seed=r["world_seed"] + 1),
                   lambda r: r.update(world_partition="audit"), lambda r: r.update(name_split="dev"),
                   lambda r: r.update(world_fingerprint="0" * 64), lambda r: r.update(id="bad"),
                   lambda r: r.update(extra="secret"), lambda r: r.update(variant=True),
                   lambda r: r["turns"][0].update(text=r["turns"][0]["text"] + " "),
                   lambda r: r["turns"][0].update(target=True), lambda r: r["turns"][0].update(reply="yes"),
                   lambda r: r["turns"][0]["observations"]["visual"][0].__setitem__(0, float("nan")),
                   lambda r: r["turns"][0]["observations"]["tokens"].__setitem__(0, True)]
        for change in changes:
            bad = deepcopy(row)
            change(bad)
            with self.assertRaises(ValueError):
                validate_diverse(bad)
        with self.assertRaises(ValueError):
            validate_cognitive(row)
        pair = generate_diverse(50, 2, family="variable_binding")
        for bad in ([pair[1], pair[0]], [pair[0], pair[0]], [pair[0]], [pair[0], generate_diverse(52, 2, family="variable_binding")[1]]):
            with self.assertRaises(ValueError):
                validate_pair(bad)

    def test_encoder_inputs_are_observations_only_and_frozen_packer_is_unchanged(self):
        rows = generate_diverse(135, 4, family="conditional_logic")
        packed = pack_cognitive_episodes(rows, training=False)
        changed = deepcopy(rows)
        for row in changed:
            row.update(family="different", id="irrelevant", world_fingerprint="oracle_secret")
            for turn in row["turns"]:
                turn["target"] = (turn["target"] + 1) % 4
        # The structural packer requires canonical reply spellings even for eval.
        from experiments.cognitive_curriculum import REPLIES
        for row in changed:
            for turn in row["turns"]:
                turn["reply"] = REPLIES[turn["target"]]
        other = pack_cognitive_episodes(changed, training=False)
        for key in packed["inputs"]:
            self.assertTrue(torch.equal(packed["inputs"][key], other["inputs"][key]))
        with self.assertRaises(ValueError):
            pack_cognitive_episodes(rows, training=True)

    def test_conditional_capacity_and_mixed_compositions(self):
        for level in (1, 2, 3):
            worlds = [generate_world_pair("conditional_logic", seed, level, "train") for seed in range(400)]
            self.assertGreaterEqual(len({world["world_fingerprint"] for world in worlds}), 128)
            orders, values, operators, queried, intervention_names = set(), set(), set(), set(), set()
            for world in worlds:
                for program in world["programs"]:
                    orders.add(tuple(e["op"] for e in program[:3]))
                    values.add(tuple(e["value"] for e in program[:3] if e["op"].startswith("switch")))
                    operators.add(next(e["operator"] for e in program if e["op"] == "rule"))
                    queried.add(program[-1]["name"])
                    intervention_names.add(program[-2]["name"])
            self.assertEqual(len(orders), 6)
            self.assertEqual(values, {(False, False), (False, True), (True, False), (True, True)})
            self.assertEqual(operators, {"all"} if level == 1 else {"all", "any"})
            self.assertGreater(len(queried), 1)
            self.assertGreater(len(intervention_names), 1)

    def test_graph_degree_preserving_counterfactuals_and_advanced_binding_composition(self):
        for level in (1, 2, 3):
            for seed in range(60):
                world = generate_world_pair("graph_reachability", seed, level, "train")
                edges = [[tuple(edge) for event in program if event["op"] == "links" for edge in event["edges"]] for program in world["programs"]]
                self.assertEqual([len(set(e)) for e in edges], [4, 4])
                for direction in (0, 1):
                    self.assertEqual(Counter(e[direction] for e in edges[0]), Counter(e[direction] for e in edges[1]))
        for seed in range(20):
            program = generate_world_pair("variable_binding", seed, 3)["programs"][0]
            self.assertEqual([e["op"] for e in program[-3:]], ["color_copy", "color_set", "color_query"])
            self.assertTrue(program[-2]["revision"])

    def test_invalid_requests_fail_without_silent_coercion(self):
        for options in ({"seed": True}, {"seed": -1}, {"count": 1}, {"count": False},
                        {"level": True}, {"level": 4}, {"family": "magic"}, {"split": "test"}):
            values = {"seed": 1, "count": 2, **options}
            with self.assertRaises(ValueError):
                generate_diverse(**values)


if __name__ == "__main__":
    unittest.main()
