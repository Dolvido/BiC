"""Functional checks for the closed desktop; no model training occurs here."""

from itertools import combinations, product
import json
import random
import unittest

import numpy as np
import torch

from brain_in_computer.computer_use.environment import (
    ACTION_NAMES, BUTTON_CENTERS, COLORS, FIELD_CENTER, MAX_STEPS, POSITIONS,
    PROMPT_TEMPLATES, TASKS, Goal, MiniDesktop, goal_success, make_episode, oracle_action,
)


class ComputerEnvironmentTests(unittest.TestCase):
    def test_pixels_and_body_are_only_observations(self):
        env = MiniDesktop(7)
        observation = env.observe()
        self.assertEqual(set(observation), {"pixels", "body"})
        self.assertEqual(tuple(observation["pixels"].shape), (3, 32, 32))
        self.assertEqual(tuple(observation["body"].shape), (4,))
        for value in observation.values():
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(((value >= 0) & (value <= 1)).all())
        self.assertEqual(env.render().size, (96, 96))
        self.assertEqual(env.render().mode, "RGB")
        env.click(95, 0)
        self.assertEqual(env.observe()["body"][:2].tolist(), [1.0, 0.0])

    def test_click_hit_testing_and_keyboard_effects(self):
        env = MiniDesktop()
        env.reset(COLORS)
        env.key("h")
        self.assertEqual(env.text, "")
        env.click(*FIELD_CENTER)
        self.assertTrue(env.focused)
        env.key("h")
        env.key("i")
        env.key("o")
        self.assertEqual(env.text, "hi")
        env.key("backspace")
        self.assertEqual(env.text, "h")
        env.click(*BUTTON_CENTERS[1])
        self.assertFalse(env.focused)
        self.assertEqual(env.last_clicked, "blue")
        env.key("k")
        self.assertEqual(env.text, "h")
        env.click(0, 0)
        self.assertEqual(env.last_clicked, "blue")
        self.assertEqual(env.action_history[-1]["hit"], "background")

    def test_fixed_actions_hit_buttons_under_geometry_shift(self):
        for seed in range(20):
            env = MiniDesktop(seed, shift=True)
            for position, color in enumerate(env.button_colors):
                env.begin_turn()
                env.step(position)
                self.assertEqual(env.last_clicked, color)
                self.assertEqual(env.action_history[-1]["position"], POSITIONS[position])

    def test_turn_stop_timeout_and_persistent_visible_state(self):
        env = MiniDesktop(8)
        env.reset(COLORS, last_clicked="yellow", text="ok")
        env.step(4)
        env.step(10)
        with self.assertRaises(RuntimeError):
            env.step(0)
        colors = list(env.button_colors)
        env.begin_turn()
        self.assertEqual(env.button_colors, colors)
        self.assertEqual(env.text, "ok")
        self.assertEqual(env.last_clicked, "yellow")
        self.assertTrue(env.focused)
        self.assertEqual(env.steps, 0)
        self.assertEqual(env.action_history, [])
        self.assertFalse(env.terminated)
        self.assertFalse(env.stop_requested)
        for _ in range(MAX_STEPS):
            env.step(5)
        self.assertTrue(env.terminated)
        self.assertFalse(env.stop_requested)
        self.assertFalse(goal_success(env, Goal("type_word", "type ok", "ok", "typed ok.")))

    def test_pointer_and_action_validation(self):
        env = MiniDesktop()
        for coordinates in ((-1, 0), (96, 0), (95.5, 0), (0, 96), (float("nan"), 2), (0, float("inf")), (True, 2)):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                env.click(*coordinates)
        for action in (-1, 11, True, 1.0, "stop"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
        for key in ("hi", "x", "enter", None):
            with self.subTest(key=key), self.assertRaises(ValueError):
                env.key(key)
        self.assertEqual(env.steps, 0)

    def test_deterministic_render_and_json_roundtrip(self):
        first, second = MiniDesktop(33, shift=True), MiniDesktop(33, shift=True)
        self.assertTrue(np.array_equal(first.render(), second.render()))
        first.step(2)
        first.step(4)
        first.step(7)
        payload = json.loads(json.dumps(first.state_dict()))
        second.load_state_dict(payload)
        self.assertTrue(np.array_equal(first.render(), second.render()))
        self.assertTrue(torch.equal(first.observe()["body"], second.observe()["body"]))
        self.assertEqual(json.dumps(first.state_dict(), sort_keys=True), json.dumps(second.state_dict(), sort_keys=True))
        cloned = first.clone()
        cloned.key("k")
        self.assertEqual(first.text, "o")
        self.assertEqual(cloned.text, "ok")
        first.reset()
        second.reset()
        self.assertTrue(np.array_equal(first.render(), second.render()))

    def test_all_oracle_tasks_succeed_across_splits_and_shift(self):
        generator = random.Random(171)
        for split, shift, task in product(PROMPT_TEMPLATES, (False, True), TASKS):
            for _ in range(12):
                with self.subTest(split=split, shift=shift, task=task):
                    env, goal = make_episode(generator, task, split, shift)
                    self.assertEqual(goal.task, task)
                    self.assertEqual(goal.expected_reply, goal.expected_reply.lower())
                    self.assertTrue(goal.expected_reply.endswith("."))
                    while not env.terminated:
                        action = oracle_action(env, goal)
                        self.assertIn(action, range(len(ACTION_NAMES)))
                        env.step(action)
                    self.assertTrue(goal_success(env, goal))
                    self.assertLess(env.steps, MAX_STEPS)

    def test_queries_reject_side_effects_and_clicks_require_correct_trace(self):
        env = MiniDesktop()
        env.reset(COLORS)
        click = Goal("click_color", "click red", "red", "clicked red.")
        env.step(1)
        env.step(0)
        env.step(10)
        self.assertFalse(goal_success(env, click))
        env.begin_turn()
        env.step(10)
        self.assertFalse(goal_success(env, click))
        env.begin_turn()
        env.step(4)
        env.step(10)
        self.assertFalse(goal_success(env, Goal("greet", "hello", "", "hello.")))

    def test_prompt_splits_are_string_disjoint_and_values_have_coverage(self):
        expanded = {}
        for split, tasks in PROMPT_TEMPLATES.items():
            prompts = set()
            for templates in tasks.values():
                for template, color, position, word in product(templates, COLORS, POSITIONS, ("hi", "ok")):
                    prompts.add(template.format(color=color, position=position.replace("_", " "), word=word))
            expanded[split] = prompts
        for first, second in combinations(expanded, 2):
            self.assertFalse(expanded[first] & expanded[second])
        generator = random.Random(7)
        for task, desired in (("click_color", set(COLORS)), ("click_position", set(POSITIONS)),
                              ("describe_position", set(POSITIONS)), ("type_word", {"hi", "ok"})):
            values = {make_episode(generator, task)[1].target for _ in range(100)}
            self.assertEqual(values, desired)


if __name__ == "__main__":
    unittest.main()
