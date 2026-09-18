"""Causality, serialization, and input-contract tests for symbolic adaptation."""
import copy
import json
import random
import unittest

import torch

from brain_in_computer.adaptation import (
    CandidateLearner, RuleWorld, all_layouts, encode_observations, split_layouts,
)


class RuleWorldTests(unittest.TestCase):
    def test_layouts_are_disjoint_complete_and_seeded_without_global_rng_mutation(self):
        state = random.getstate()
        train, test = split_layouts(7001)
        self.assertEqual(random.getstate(), state)
        self.assertEqual((len(train), len(test)), (18, 6))
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train + test), set(all_layouts()))
        self.assertEqual((train, test), split_layouts(7001))
        self.assertNotEqual((train, test), split_layouts(7002))

    def test_same_seed_same_trajectory_and_independent_global_rng(self):
        state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
        left, right = RuleWorld(123), RuleWorld(123)
        self.assertEqual(left.snapshot(), right.snapshot())
        self.assertEqual(random.getstate(), state)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        layouts = set()
        for index in range(40):
            self.assertEqual(left.observe(), right.observe())
            layout = left.observe()["layout"]
            self.assertEqual(sorted(layout), [0, 1, 2, 3])
            layouts.add(tuple(layout))
            self.assertEqual(left.step(index % 4), right.step(index % 4))
        self.assertGreater(len(layouts), 1)
        self.assertTrue(left.done)

    def test_single_hidden_rule_change_is_exact_and_feedback_is_delayed(self):
        world = RuleWorld(31, horizon=8, reversal_step=4)
        first = world.evaluator_target_category()
        self.assertEqual(world.observe()["previous_category"], None)
        self.assertEqual(world.observe()["reward"], None)
        targets = []
        for step in range(8):
            before = world.observe()
            self.assertEqual(set(before), {"layout", "previous_category", "reward"})
            targets.append(world.evaluator_target_category())
            cell = before["layout"].index(first)
            after, reward = world.step(cell)
            self.assertEqual(reward, 1.0 if step < 4 else 0.0)
            self.assertEqual(after["previous_category"], first)
            self.assertEqual(after["reward"], reward)
        self.assertEqual(targets[:4], [first] * 4)
        self.assertEqual(len(set(targets[4:])), 1)
        self.assertNotEqual(first, targets[4])
        self.assertEqual([row["phase"] for row in world.export_history()], ["before"] * 4 + ["after"] * 4)

    def test_no_layout_or_flag_cue_from_reversal_schedule_or_actions(self):
        early = RuleWorld(5, horizon=12, reversal_step=3)
        late = RuleWorld(5, horizon=12, reversal_step=8)
        self.assertEqual(early.observe(), late.observe())
        self.assertEqual(early.evaluator_target_category(), late.evaluator_target_category())
        for step in range(12):
            # Paired worlds have the same visible scene even when their hidden
            # rules, action histories, and rewards differ.
            self.assertEqual(early.observe()["layout"], late.observe()["layout"])
            self.assertEqual(set(early.observe()), {"layout", "previous_category", "reward"})
            early.step(step % 4)
            late.step((step + 1) % 4)
        self.assertNotEqual(early.export_history(), late.export_history())

    def test_default_reversal_is_seeded_variable_and_inside_documented_range(self):
        switches = {RuleWorld(seed).evaluator_state()["reversal_step"] for seed in range(100)}
        self.assertEqual(switches, set(range(16, 25)))

    def test_layout_subset_and_external_mutations_do_not_modify_world(self):
        train, _ = split_layouts()
        world = RuleWorld(7, allowed_layouts=train)
        initial = world.snapshot()
        observation = world.observe()
        observation["layout"][0] = 99
        train[0] = (99, 99, 99, 99)
        self.assertEqual(initial, world.snapshot())
        original_allowed = {tuple(layout) for layout in initial["allowed_layouts"]}
        for _ in range(40):
            self.assertIn(tuple(world.observe()["layout"]), original_allowed)
            world.step(0)
        history = world.export_history()
        history[0]["layout"][0] = 99
        self.assertNotEqual(world.export_history(), history)

    def test_json_snapshot_resume_has_exact_rng_and_future_trajectory(self):
        world = RuleWorld(91, horizon=12, reversal_step=6)
        for action in [3, 2, 0, 1, 2]:
            world.step(action)
        frozen = json.loads(json.dumps(world.snapshot()))
        restored = RuleWorld.from_snapshot(frozen)
        self.assertEqual(world.snapshot(), restored.snapshot())
        for action in [0, 3, 2, 1, 0, 3, 2]:
            self.assertEqual(world.observe(), restored.observe())
            self.assertEqual(world.step(action), restored.step(action))
            self.assertEqual(world.snapshot(), restored.snapshot())
        terminal = RuleWorld.from_snapshot(json.loads(json.dumps(world.snapshot())))
        self.assertTrue(terminal.done)
        before = terminal.snapshot()
        with self.assertRaises(RuntimeError):
            terminal.step(0)
        self.assertEqual(before, terminal.snapshot())

    def test_initial_and_default_switch_snapshots_restore_without_implicit_reset(self):
        world = RuleWorld(14)
        restored = RuleWorld.from_snapshot(json.loads(json.dumps(world.snapshot())))
        self.assertEqual(world.snapshot(), restored.snapshot())
        for _ in range(22):
            world.step(1)
        restored = RuleWorld.from_snapshot(world.snapshot())
        self.assertEqual(restored.evaluator_state()["step_index"], 22)

    def test_invalid_world_inputs_and_actions_are_rejected_before_mutation(self):
        for args in ({"seed": True}, {"seed": 1.2}, {"seed": 0, "horizon": True},
                     {"seed": 0, "horizon": 1}, {"seed": 0, "horizon": 10},
                     {"seed": 0, "reversal_step": 0}, {"seed": 0, "reversal_step": 40},
                     {"seed": 0, "reversal_step": 3.0}, {"seed": 0, "allowed_layouts": []},
                     {"seed": 0, "allowed_layouts": [[0, 1, 2, 2]]},
                     {"seed": 0, "allowed_layouts": [[False, 1, 2, 3]]},
                     {"seed": 0, "allowed_layouts": [[0, 1, 2, 3]] * 2}):
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                RuleWorld(**args)
        world = RuleWorld(1)
        before = world.snapshot()
        for action in (True, -1, 4, 0.0, "0", None):
            with self.subTest(action=action), self.assertRaises((TypeError, ValueError)):
                world.step(action)
            self.assertEqual(before, world.snapshot())

    def test_corrupt_snapshot_hidden_labels_feedback_and_rng_fail_closed(self):
        world = RuleWorld(8)
        world.step(2)
        for field, replacement in (("initial_target", 99), ("step_index", True),
                                   ("previous_category", 99), ("reward", float("nan")),
                                   ("rng_state", [3, [], None]), ("history", [])):
            altered = copy.deepcopy(world.snapshot())
            altered[field] = replacement
            with self.subTest(field=field), self.assertRaises((TypeError, ValueError)):
                RuleWorld.from_snapshot(altered)
        altered = world.snapshot()
        altered["history"][0]["reward"] = 1 - altered["history"][0]["reward"]
        with self.assertRaises(ValueError):
            RuleWorld.from_snapshot(altered)


class CandidateLearnerTests(unittest.TestCase):
    @staticmethod
    def observation(previous=None, reward=None, layout=(3, 2, 1, 0)):
        return {"layout": list(layout), "previous_category": previous, "reward": reward}

    def test_feedback_eliminates_candidates_and_maps_category_to_cell(self):
        learner = CandidateLearner()
        self.assertEqual(learner.probabilities(self.observation()), [.25] * 4)
        self.assertEqual(learner.act(self.observation()), 3)
        self.assertEqual(learner.probabilities(self.observation(0, 0.0)), [0, 1 / 3, 1 / 3, 1 / 3])
        self.assertEqual(learner.choose_category(self.observation(1, 0.0)), 2)
        self.assertEqual(learner.probabilities(self.observation(3, 1.0)), [0, 0, 0, 1])
        self.assertEqual(learner.act(self.observation(3, 1.0)), 0)

    def test_contradicted_singleton_resets_only_from_negative_feedback(self):
        learner = CandidateLearner()
        learner.probabilities(self.observation(2, 1.0))
        self.assertEqual(learner.snapshot()["candidates"], [2])
        self.assertEqual(learner.probabilities(self.observation(2, 0.0)), [1 / 3, 1 / 3, 0, 1 / 3])
        self.assertEqual(learner.probabilities(self.observation(2, 0.0)), [1 / 3, 1 / 3, 0, 1 / 3])

    def test_learner_json_resume_preserves_choices_through_world_reversal(self):
        world, learner = RuleWorld(41, horizon=12, reversal_step=6), CandidateLearner()
        for _ in range(5):
            world.step(learner.act(world.observe()))
        copied_world = RuleWorld.from_snapshot(json.loads(json.dumps(world.snapshot())))
        copied_learner = CandidateLearner.from_snapshot(json.loads(json.dumps(learner.snapshot())))
        while not world.done:
            action = learner.act(world.observe())
            self.assertEqual(action, copied_learner.act(copied_world.observe()))
            self.assertEqual(world.step(action), copied_world.step(action))
        self.assertEqual(learner.snapshot(), copied_learner.snapshot())

    def test_scripted_baseline_learns_both_rules_with_at_most_six_total_misses(self):
        for seed in range(20):
            world, learner = RuleWorld(seed), CandidateLearner()
            rewards = []
            while not world.done:
                _, reward = world.step(learner.act(world.observe()))
                rewards.append(reward)
            self.assertGreaterEqual(sum(rewards), 34)
            self.assertEqual(rewards[-1], 1.0)

    def test_invalid_snapshots_and_observations_do_not_change_baseline(self):
        learner = CandidateLearner()
        original = learner.snapshot()
        for observation in ({**self.observation(), "target": 0}, self.observation(None, 1),
                            self.observation(0, None), self.observation(True, 0),
                            self.observation(0, float("inf"))):
            with self.assertRaises((ValueError, TypeError)):
                learner.probabilities(observation)
            self.assertEqual(learner.snapshot(), original)
        for values in ([], [0, 0], [2, 1], [True], [4]):
            with self.assertRaises((ValueError, TypeError)):
                CandidateLearner.from_snapshot({"schema": original["schema"], "candidates": values})


class ObservationEncodingTests(unittest.TestCase):
    def test_exact_shapes_values_and_feedback_absence(self):
        rows = [{"layout": [2, 0, 3, 1], "previous_category": None, "reward": None},
                {"layout": [0, 3, 1, 2], "previous_category": 2, "reward": 0.0},
                {"layout": [1, 2, 0, 3], "previous_category": 3, "reward": 1.0}]
        encoded = encode_observations(rows)
        for key, width in (("visual", 16), ("auditory", 4), ("body", 4), ("feedback", 2)):
            self.assertEqual(tuple(encoded[key].shape), (3, 1, width))
            self.assertEqual(encoded[key].dtype, torch.float32)
        self.assertEqual(tuple(encoded["tokens"].shape), (3, 1))
        self.assertEqual(encoded["tokens"].dtype, torch.long)
        self.assertEqual(int(encoded["tokens"].sum()), 0)
        self.assertEqual(encoded["visual"].reshape(3, 4, 4).argmax(-1).tolist(), [r["layout"] for r in rows])
        self.assertTrue(torch.all(encoded["visual"].reshape(3, 4, 4).sum(-1) == 1))
        self.assertEqual(encoded["auditory"].squeeze(1).tolist(), [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.assertEqual(encoded["body"].squeeze(1).tolist(), [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])
        self.assertEqual(encoded["feedback"].squeeze(1).tolist(), [[0, 0], [0, 1], [1, 1]])

    def test_inputs_have_no_absolute_time_phase_or_future_outcome(self):
        early = RuleWorld(21, horizon=10, reversal_step=2)
        late = RuleWorld(21, horizon=10, reversal_step=8)
        for _ in range(2):
            early.step(0)
            late.step(0)
        self.assertNotEqual(early.evaluator_phase(), late.evaluator_phase())
        self.assertEqual(early.observe(), late.observe())
        a, b = encode_observations([early.observe()]), encode_observations([late.observe()])
        self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))
        leaked = dict(early.observe(), reversal_step=2)
        with self.assertRaises(ValueError):
            encode_observations([leaked])

    def test_encoding_rejects_invalid_public_values(self):
        valid = RuleWorld(0).observe()
        for observation in ({**valid, "layout": [0, 0, 2, 3]}, {**valid, "reward": True},
                            {**valid, "previous_category": 2}, {**valid, "step": 0},
                            {**valid, "layout": [0.0, 1, 2, 3]}):
            with self.assertRaises((ValueError, TypeError)):
                encode_observations([observation])
        for observations in ([], {}, "bad"):
            with self.assertRaises(ValueError):
                encode_observations(observations)


if __name__ == "__main__":
    unittest.main()
