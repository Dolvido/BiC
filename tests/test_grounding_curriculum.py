"""Checks the evidence boundaries of the v0.4 grounding experiment."""

from itertools import product
import unittest
from unittest.mock import patch

import torch

from experiments.advance_grounding import V04_TEMPLATES, build_curriculum, evaluate
from brain_in_computer.computer_use.environment import COLORS, POSITIONS, TASKS
from brain_in_computer.computer_use.model import ComputerBrain, ComputerConfig


class GroundingCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_expanded_prompt_splits_are_disjoint(self):
        sentences = {}
        for split, bank in V04_TEMPLATES.items():
            self.assertEqual(set(bank), set(TASKS))
            sentences[split] = set()
            for templates in bank.values():
                for template, color, position, word in product(templates, COLORS, POSITIONS, ("hi", "ok")):
                    sentence = template.format(color=color, position=position.replace("_", " "), word=word)
                    self.assertLessEqual(len(sentence.encode("utf8")), 128)
                    sentences[split].add(sentence)
        self.assertFalse(sentences["train"] & sentences["development"])
        self.assertFalse(sentences["train"] & sentences["sealed"])
        self.assertFalse(sentences["development"] & sentences["sealed"])

    def test_counterfactual_groups_cover_every_layout_and_target(self):
        actions, replies, groups = build_curriculum(42, history_size=3, variations=1, other_episodes=1)
        self.assertEqual(len(groups), 24*3)
        self.assertEqual(len({tuple(replies[group[0]]["scene"][:4]) for group in groups}), 24)
        for group in groups:
            rows = [replies[index] for index in group]
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({tuple(row["scene"][:4]) for row in rows}), 1)
            task = rows[0]["task"]
            self.assertEqual({row["target"] for row in rows}, set(COLORS if task == "click_color" else POSITIONS))
        self.assertTrue(all(row["pixels"].dtype == torch.uint8 for row in actions+replies))

    def test_closed_loop_evaluation_does_not_call_teacher(self):
        model = ComputerBrain(ComputerConfig(hidden_size=8, language_hidden_size=8, embedding_size=4, visual_features=4))
        with patch("experiments.advance_grounding.oracle_action", side_effect=AssertionError("teacher leaked")):
            result = evaluate(model, seed=23, per_task=1, split="development")
        self.assertFalse(result["teacher_used_for_policy"])
        self.assertTrue(result["free_running_replies"])
        self.assertEqual(set(result["tasks"]), set(TASKS))


if __name__ == "__main__":
    unittest.main()
