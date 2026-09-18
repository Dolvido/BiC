"""Causal data-contract checks for the synthetic task stream."""

import unittest

import torch

from brain_in_computer.tasks import TASK_NAMES, Batch, TaskStream


class TaskStreamTests(unittest.TestCase):
    def assert_batches_equal(self, left: Batch, right: Batch) -> None:
        self.assertEqual(left.observations.keys(), right.observations.keys())
        for key in left.observations:
            self.assertTrue(torch.equal(left.observations[key], right.observations[key]), key)
        for key in ("targets", "visual_targets", "auditory_targets", "task_ids"):
            self.assertTrue(torch.equal(getattr(left, key), getattr(right, key)), key)

    def test_shapes_types_and_no_feedback_leakage(self):
        batch = TaskStream(12).sample(batch_size=32, delay=3)
        for name, shape in {
            "visual": (32, 5, 8), "auditory": (32, 5, 8),
            "body": (32, 5, 4), "tokens": (32, 5), "feedback": (32, 5, 2),
        }.items():
            tensor = batch.observations[name]
            self.assertEqual(tuple(tensor.shape), shape)
            self.assertEqual(tensor.dtype, torch.long if name == "tokens" else torch.float32)
            self.assertEqual(tensor.device.type, "cpu")
        self.assertTrue(torch.all(batch.observations["feedback"] == 0))
        self.assertEqual(tuple(batch.task_ids.shape), (32,))
        self.assertEqual(batch.task_ids.dtype, torch.long)
        for label in (batch.targets, batch.visual_targets, batch.auditory_targets):
            self.assertEqual(tuple(label.shape), (32, 5))
            self.assertEqual(label.dtype, torch.long)
        self.assertTrue(torch.all(batch.targets[:, :-1] == -100))
        self.assertTrue(torch.all((batch.targets[:, -1] >= 0) & (batch.targets[:, -1] < 4)))
        tokens = batch.observations["tokens"]
        self.assertTrue(torch.equal(tokens, tokens[:, :1].expand_as(tokens)))

    def test_each_task_matches_actual_observation(self):
        stream = TaskStream(8)
        for task_id, name in enumerate(TASK_NAMES):
            with self.subTest(task=name):
                batch = stream.sample(512, tasks=[name], delay=4)
                obs = batch.observations
                visual = obs["visual"][..., :4].argmax(dim=-1)
                auditory = obs["auditory"][..., :4].argmax(dim=-1)
                tokens = obs["tokens"][:, -1]
                self.assertTrue(torch.all(batch.task_ids == task_id))
                if name == "visual_match":
                    expected, token = visual[:, -1], 1
                elif name == "auditory_match":
                    expected, token = auditory[:, -1], 2
                elif name == "rule_switch":
                    expected = torch.where(tokens == 4, auditory[:, -1], visual[:, -1])
                    self.assertEqual(set(tokens.tolist()), {3, 4})
                    token = None
                elif name == "delayed_match":
                    expected, token = visual[:, 0], 5
                    self.assertTrue(torch.all(obs["visual"][:, -1] == 0))
                    self.assertTrue(torch.all(batch.visual_targets[:, -1] == -100))
                    self.assertGreater(int((visual[:, 1] != visual[:, 0]).sum()), 200)
                else:
                    expected, token = (visual[:, -1] + auditory[:, -1]) % 4, 6
                self.assertTrue(torch.equal(batch.targets[:, -1], expected))
                if token is not None:
                    self.assertTrue(torch.all(tokens == token))
                self.assertEqual(set(expected.tolist()), {0, 1, 2, 3})
                visible = batch.visual_targets != -100
                self.assertTrue(torch.equal(batch.visual_targets[visible], visual[visible]))
                self.assertTrue(torch.equal(batch.auditory_targets, auditory))

    def test_rng_is_reproducible_local_and_advances(self):
        stream_a, stream_b = TaskStream(17), TaskStream(17)
        torch.manual_seed(5)
        global_state = torch.random.get_rng_state().clone()
        first_a, first_b = stream_a.sample(), stream_b.sample()
        self.assert_batches_equal(first_a, first_b)
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))
        second_a, second_b = stream_a.sample(), stream_b.sample()
        self.assert_batches_equal(second_a, second_b)
        self.assertFalse(torch.equal(first_a.observations["visual"], second_a.observations["visual"]))
        different = TaskStream(18).sample()
        self.assertFalse(torch.equal(first_a.observations["visual"], different.observations["visual"]))

    def test_symbols_are_independently_drawn(self):
        batch = TaskStream(39).sample(4096, tasks=["rule_switch"], delay=2)
        v, a = batch.visual_targets, batch.auditory_targets
        # Large deterministic sample: every joint pair occurs in each setting.
        # This catches tied modalities, tied time steps, and instruction leakage.
        self.assertEqual(set((4 * v[:, -1] + a[:, -1]).tolist()), set(range(16)))
        self.assertEqual(set((4 * v[:, 0] + v[:, -1]).tolist()), set(range(16)))
        for instruction in (3, 4):
            selected = batch.observations["tokens"][:, 0] == instruction
            self.assertEqual(set(v[selected, -1].tolist()), {0, 1, 2, 3})
            self.assertEqual(set(a[selected, -1].tolist()), {0, 1, 2, 3})
        equality_rate = (v[:, -1] == a[:, -1]).float().mean().item()
        self.assertGreater(equality_rate, 0.20)
        self.assertLess(equality_rate, 0.30)

    def test_longer_delay_and_body_cues(self):
        for delay in (0, 2, 12):
            with self.subTest(delay=delay):
                batch = TaskStream(21).sample(16, tasks=["delayed_match"], delay=delay)
                body = batch.observations["body"]
                self.assertEqual(body.shape[1], delay + 2)
                self.assertTrue(torch.all(body[:, 0, 0] == 0))
                self.assertTrue(torch.all(body[:, -1, 0] == 1))
                self.assertTrue(torch.all(body[:, 0, 1] == 1))
                self.assertTrue(torch.all(body[:, 1:, 1] == 0))
                self.assertTrue(torch.all(body[:, -1, 2] == 1))
                self.assertTrue(torch.all(body[:, :-1, 2] == 0))
                self.assertTrue(torch.all(body[:, :, 3] == 1))
                self.assertTrue(torch.equal(batch.targets[:, -1], batch.visual_targets[:, 0]))

    def test_subset_and_transfer(self):
        batch = TaskStream().sample(128, tasks=("cross_modal", "auditory_match"))
        self.assertEqual(set(batch.task_ids.tolist()), {1, 4})
        moved = batch.to(torch.device("cpu"))
        self.assert_batches_equal(batch, moved)
        self.assertIsNot(batch, moved)
        self.assertIsNot(batch.observations, moved.observations)

    def test_invalid_inputs(self):
        for value in (True, 1.5, "1"):
            with self.subTest(seed=value), self.assertRaises(TypeError):
                TaskStream(value)
        stream = TaskStream()
        for kwargs, exception in (
            ({"batch_size": 0}, ValueError),
            ({"batch_size": -2}, ValueError),
            ({"batch_size": True}, TypeError),
            ({"batch_size": 2.5}, TypeError),
            ({"delay": -1}, ValueError),
            ({"delay": False}, TypeError),
            ({"delay": 1.0}, TypeError),
            ({"tasks": []}, ValueError),
            ({"tasks": ["missing"]}, ValueError),
            ({"tasks": "visual_match"}, TypeError),
            ({"tasks": [1]}, TypeError),
            ({"tasks": ["visual_match", "visual_match"]}, ValueError),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(exception):
                stream.sample(**kwargs)


if __name__ == "__main__":
    unittest.main()
