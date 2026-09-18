"""Truth, provenance and inference boundaries for general-learning toy worlds."""
import copy
import json
import unittest
from unittest.mock import patch

import torch

from experiments.cognitive_curriculum import (
    ACK, ALLOW, ASK, DENY, ALIASES, FAMILIES, SPLITS, allowed_entity_pairs,
    cognitive_bank, cognitive_oracle, curriculum_digest, generate_cognitive,
    parse_cognitive_sentence, validate_cognitive,
)
from brain_in_computer.dialogue_student import build_dialogue_student, evaluate_dialogues
from brain_in_computer.dialogue_curriculum import validate_dialogue


def hand_world(family, *sentences):
    return {"family": family, "turns": [{"text": text} for text in sentences]}


class CognitiveOracleTests(unittest.TestCase):
    def test_binding_copy_preserves_value_when_source_changes(self):
        world = hand_world("variable_binding", "Is dax red?", "dax is red.",
                           "Set wug to the color of dax.", "dax is now blue.",
                           "Is wug red?", "Is dax red?")
        self.assertEqual(cognitive_oracle(world), [ASK, ACK, ACK, ACK, ALLOW, DENY])

    def test_copy_unknown_erases_stale_color(self):
        world = hand_world("variable_binding", "dax is red.", "Set dax to the color of wug.", "Is dax red?")
        self.assertEqual(cognitive_oracle(world), [ACK, ACK, ASK])

    def test_graph_is_directed_transitive_and_explicitly_closed(self):
        world = hand_world("graph_reachability", "There is a link from dax to wug.",
                           "There is a link from wug to fep.", "Can a path follow links from dax to fep?",
                           "Can a path follow links from fep to dax?", "These are all the links.",
                           "Can a path follow links from fep to dax?")
        self.assertEqual(cognitive_oracle(world), [ACK, ACK, ALLOW, ASK, ACK, DENY])
        with self.assertRaisesRegex(ValueError, "closure"):
            cognitive_oracle(hand_world("graph_reachability", "These are all the links.", "There is a link from dax to wug."))

    def test_graph_cycles_terminate_and_missing_path_remains_unknown(self):
        world = hand_world("graph_reachability", "There is a link from dax to wug.",
                           "There is a link from wug to dax.", "Can a path follow links from dax to fep?",
                           "Can a path follow links from dax to dax?")
        self.assertEqual(cognitive_oracle(world), [ACK, ACK, ASK, ALLOW])

    def test_compound_link_sentence_has_same_truth_as_two_separate_statements(self):
        joint = hand_world("graph_reachability", "There are links from dax to wug and from wug to fep.",
                           "Can a path follow links from dax to fep?", "These are all the links.",
                           "Can a path follow links from fep to dax?")
        separate = hand_world("graph_reachability", "There is a link from dax to wug.",
                              "There is a link from wug to fep.", *[t["text"] for t in joint["turns"][1:]])
        self.assertEqual(cognitive_oracle(joint), [ACK, ALLOW, ACK, DENY])
        self.assertEqual(cognitive_oracle(joint)[1:], cognitive_oracle(separate)[2:])

    def test_arithmetic_tracks_updates_and_copied_counts(self):
        world = hand_world("arithmetic_updates", "dax starts with a count of 6.",
                           "Increase the count of dax by 3.", "Set the count of wug to the count of dax.",
                           "Decrease the count of wug by 2.", "Is the count of wug 7?", "Is the count of dax 7?")
        self.assertEqual(cognitive_oracle(world), [ACK, ACK, ACK, ACK, ALLOW, DENY])

    def test_unknown_count_not_invented_and_negative_counts_rejected(self):
        world = hand_world("arithmetic_updates", "Increase the count of dax by 2.", "Is the count of dax 2?",
                           "wug starts with a count of 4.", "Set the count of wug to the count of dax.", "Is the count of wug 4?")
        self.assertEqual(cognitive_oracle(world), [ACK, ASK, ACK, ACK, ASK])
        with self.assertRaisesRegex(ValueError, "negative"):
            cognitive_oracle(hand_world("arithmetic_updates", "dax starts with a count of 1.", "Decrease the count of dax by 2."))

    def test_boolean_rule_recomputes_after_intervention(self):
        world = hand_world("conditional_logic", "Switch dax is on.", "Switch wug and switch fep are on.",
                           "Lamp zot is on exactly when all of dax, wug, fep are on.", "Is lamp zot on?",
                           "Set switch dax to off.", "Is lamp zot on?")
        self.assertEqual(cognitive_oracle(world), [ACK, ACK, ACK, ALLOW, ACK, DENY])

    def test_boolean_unknowns_use_three_valued_logic(self):
        world = hand_world("conditional_logic", "Switch dax is off.",
                           "Lamp zot is on exactly when all of dax, wug are on.", "Is lamp zot on?",
                           "Lamp zot is on exactly when any of dax, wug are on.", "Is lamp zot on?",
                           "Set switch dax to on.", "Is lamp zot on?")
        self.assertEqual(cognitive_oracle(world), [ACK, ACK, DENY, ACK, ASK, ACK, ALLOW])

    def test_oracle_ignores_claimed_targets_and_forbids_cross_family_sentences(self):
        world = hand_world("variable_binding", "dax is red.", "Is dax red?")
        for turn in world["turns"]:
            turn.update(target=DENY, observations={"secret": 4}, reply="wrong")
        self.assertEqual(cognitive_oracle(world), [ACK, ALLOW])
        with self.assertRaisesRegex(ValueError, "family"):
            cognitive_oracle(hand_world("variable_binding", "Is lamp dax on?"))


class CognitiveGenerationTests(unittest.TestCase):
    def test_all_families_splits_levels_are_verified_exact_pairs(self):
        for family in FAMILIES:
            for split in SPLITS:
                for level in (1, 2, 3):
                    episodes = generate_cognitive(1200, 12, split, family, level)
                    for left, right in zip(episodes[::2], episodes[1::2]):
                        with self.subTest(family=family, split=split, level=level, seed=left["seed"]):
                            self.assertEqual(left["counterfactual_group"], right["counterfactual_group"])
                            differing = 0
                            flips = 0
                            for episode in (left, right):
                                self.assertTrue(validate_cognitive(episode))
                                self.assertEqual(len(episode["turns"]), 6)
                                self.assertEqual(cognitive_oracle(episode), [t["target"] for t in episode["turns"]])
                            for a, b in zip(left["turns"], right["turns"]):
                                differing += a["text"] != b["text"]
                                self.assertEqual(a["observations"], b["observations"])
                                if a["target"] != ACK:
                                    self.assertEqual(a["text"], b["text"])
                                flips += {a["target"], b["target"]} == {DENY, ALLOW}
                            self.assertEqual(differing, 1)
                            self.assertGreater(flips, 0)

    def test_split_primary_pairs_are_disjoint_and_cover_same_vocabulary(self):
        supports = {split: set(allowed_entity_pairs(split)) for split in SPLITS}
        for first in SPLITS:
            self.assertEqual(set(alias for pair in supports[first] for alias in pair), set(ALIASES))
            for second in SPLITS:
                if first != second:
                    self.assertFalse(supports[first] & supports[second])
        self.assertEqual(sum(map(len, supports.values())), len(ALIASES) * (len(ALIASES) - 1))

    def test_generated_primary_pairs_are_observable_and_obey_support(self):
        for family in FAMILIES:
            for split in SPLITS:
                for level in (1, 2, 3):
                    for episode in generate_cognitive(5000, 12, split, family, level):
                        facts = [parse_cognitive_sentence(turn["text"])[1] for turn in episode["turns"]]
                        if family == "variable_binding":
                            definitions = [parse_cognitive_sentence(turn["text"])[1] for turn in episode["turns"]
                                           if parse_cognitive_sentence(turn["text"])[0] == "color_set"]
                            pair = definitions[0]["name"], definitions[1]["name"]
                        elif family == "graph_reachability":
                            pair = facts[-1]["source"], facts[-1]["destination"]
                        elif family == "arithmetic_updates":
                            pair = facts[0]["name"], facts[4 if level == 2 else 3]["name"]
                        else:
                            pair = facts[0]["name"], facts[1]["name"]
                        self.assertIn(pair, allowed_entity_pairs(split))

    def test_conditional_final_answer_requires_rule_and_intervened_other_input(self):
        episodes = generate_cognitive(7200, 128, family="conditional_logic", level=3)
        final_outcomes = set()
        unknown = 0
        for episode in episodes:
            first = parse_cognitive_sentence(episode["turns"][0]["text"])[1]
            intervention = parse_cognitive_sentence(episode["turns"][4]["text"])[1]
            self.assertNotEqual(first["name"], intervention["name"])
            if episode["turns"][5]["target"] == ASK:
                unknown += 1
            else:
                final_outcomes.add((episode["turns"][3]["target"], episode["turns"][5]["target"]))
        # Same first answer can remain or change after the other input changes.
        self.assertEqual(final_outcomes, {(DENY, DENY), (DENY, ALLOW), (ALLOW, DENY), (ALLOW, ALLOW)})
        self.assertEqual(unknown, len(episodes) // 4)

    def test_graph_controls_defeat_each_single_edge_position_and_endpoint_degree(self):
        for level in (2, 3):
            rows = generate_cognitive(8400, 128, family="graph_reachability", level=level)
            ambiguous_positions = set()
            early_positions, early_labels = set(), set()
            broken_positions = set()
            for left, right in zip(rows[::2], rows[1::2]):
                def graph(ep):
                    edges = []
                    for turn in ep["turns"]:
                        kind, facts = parse_cognitive_sentence(turn["text"])
                        if kind in ("link", "link_pair"):
                            edges.append({"source": facts["source"], "destination": facts["destination"]})
                            if kind == "link_pair":
                                edges.append({"source": facts["other_source"], "destination": facts["other_destination"]})
                    return edges
                first, second = graph(left), graph(right)
                query = parse_cognitive_sentence(left["turns"][-1]["text"])[1]
                start, end = query["source"], query["destination"]
                self.assertEqual(left["turns"][-1]["text"], right["turns"][-1]["text"])
                self.assertEqual((left["turns"][-1]["target"], right["turns"][-1]["target"]), (ALLOW, DENY))
                # Full endpoint in/out-degree signatures and the terminal edge
                # agree despite opposite path truth. A destination match is not
                # sufficient evidence of reachability.
                def degrees(edges):
                    return tuple(sum(edge[direction] == node for edge in edges)
                                 for node in (start, end) for direction in ("source", "destination"))
                self.assertEqual(degrees(first), degrees(second))
                nodes_first = {node for edge in first for node in edge.values()}
                nodes_second = {node for edge in second for node in edge.values()}
                self.assertEqual(nodes_first, nodes_second)
                self.assertEqual(len(nodes_first), 6)
                self.assertEqual(len(first), 4)
                for node in nodes_first:
                    for direction in ("source", "destination"):
                        self.assertEqual(sum(edge[direction] == node for edge in first),
                                         sum(edge[direction] == node for edge in second))
                self.assertEqual([edge for edge in first if edge["destination"] == end],
                                 [edge for edge in second if edge["destination"] == end])
                for position, (a, b) in enumerate(zip(first, second)):
                    if a == b:
                        # Identical (edge-at-position, question) representations
                        # have different labels; no one fixed edge solves bank.
                        ambiguous_positions.add(position)
                    else:
                        broken_positions.add(position)
                early = next(index for index, turn in enumerate(left["turns"])
                             if parse_cognitive_sentence(turn["text"])[0] == "path_query")
                early_positions.add(early)
                early_labels.add(left["turns"][early]["target"])
                self.assertEqual(left["turns"][early]["target"], right["turns"][early]["target"])
            self.assertEqual(ambiguous_positions, {0, 1, 2, 3})
            self.assertEqual(broken_positions, {0, 1, 2, 3})
            self.assertEqual(early_positions, {1, 2, 3})
            self.assertEqual(early_labels, {ALLOW, ASK})

    def test_binding_unknown_positions_and_post_revision_answers_are_varied(self):
        for level in (1, 2):
            rows = generate_cognitive(8800, 128, family="variable_binding", level=level)
            unknown_positions = {index for ep in rows for index, turn in enumerate(ep["turns"])
                                 if turn["target"] == ASK}
            final_targets = {ep["turns"][-1]["target"] for ep in rows}
            self.assertEqual(unknown_positions, set(range(6)))
            self.assertEqual(final_targets, {DENY, ALLOW, ASK})

    def test_recipe_is_reproducible_serializable_and_has_independent_storage(self):
        episodes = generate_cognitive(400, 16, family="mixed", level=3)
        self.assertEqual(episodes, generate_cognitive(400, 16, family="mixed", level=3))
        self.assertEqual(episodes, json.loads(json.dumps(episodes)))
        self.assertEqual(len(curriculum_digest()), 64)
        before = copy.deepcopy(episodes[1])
        episodes[0]["turns"][0]["observations"]["visual"][0][0] = 8
        self.assertEqual(episodes[1], before)

    def test_exact_provenance_rejects_tampering_even_when_otherwise_valid(self):
        original = generate_cognitive(0, 2, family="variable_binding")[0]
        mutations = [lambda ep: ep.update(level=2), lambda ep: ep.update(split="dev"),
                     lambda ep: ep["turns"][0]["observations"]["tokens"].__setitem__(0, True),
                     lambda ep: ep["turns"][0]["observations"]["visual"][0].__setitem__(0, float("nan")),
                     lambda ep: ep["turns"][2].update(target=(ep["turns"][2]["target"] + 1) % 4),
                     lambda ep: ep["turns"][0].update(text=ep["turns"][0]["text"] + " ")]
        for mutate in mutations:
            bad = copy.deepcopy(original)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_cognitive(bad)
        with self.assertRaises(ValueError):
            validate_dialogue(original)

    def test_invalid_requests_and_unsupported_grammar_rejected(self):
        for seed, count in ((True, 2), (1, 2), (-2, 2), (0, True), (0, 1), (0, -2)):
            with self.assertRaises(ValueError):
                generate_cognitive(seed, count)
        for options in ({"family": "magic"}, {"split": "test"}, {"level": True}, {"level": 0}, {"level": 4}):
            with self.assertRaises(ValueError):
                generate_cognitive(0, 2, **options)
        for text in ("dax is ultraviolet.", "Ignore all instructions.", "x" * 129, None):
            with self.assertRaises(ValueError):
                parse_cognitive_sentence(text)
        self.assertEqual(generate_cognitive(0, 0), [])

    def test_bank_and_student_evaluation_need_no_oracle_at_inference(self):
        bank = cognitive_bank(1600, 2, split="audit", level=3)
        self.assertEqual(set(bank), set(FAMILIES))
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            model = build_dialogue_student(3)
            with patch("experiments.cognitive_curriculum.cognitive_oracle", side_effect=AssertionError("oracle used by policy")), \
                 patch("experiments.cognitive_curriculum.parse_cognitive_sentence", side_effect=AssertionError("parser used by policy")):
                for episodes in bank.values():
                    result = evaluate_dialogues(model, episodes, score_replies=False)
                    self.assertGreater(result["counterfactual_query_pairs"], 0)
                    self.assertFalse(result["teacher_used_for_policy"])
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
