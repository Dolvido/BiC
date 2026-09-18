"""Operational checks for bounded rehearsal, resumption, and pure evaluation."""

from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest

import torch

from brain_in_computer.model import BrainConfig
from brain_in_computer.replay import EpisodeReplay
from brain_in_computer.tasks import TASK_NAMES, TaskStream
from brain_in_computer.training import TrainConfig, Trainer, evaluate, learning_loss


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.previous_rng = torch.get_rng_state().clone()

    def tearDown(self):
        torch.set_rng_state(self.previous_rng)

    def assert_nested_equal(self, expected, actual):
        """Compare saved operational state, including tensors, exactly."""
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor)
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertEqual(expected.device, actual.device)
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(expected), type(actual))
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def assert_batch_equal(self, expected, actual):
        self.assert_nested_equal(vars(expected), vars(actual))

    def make_trainer(self):
        return Trainer(
            BrainConfig(hidden_size=16),
            TrainConfig(seed=51, batch_size=8, delay=1, replay_capacity=10,
                        replay_fraction=0.25, curriculum="staged", stage_steps=1),
            device="cpu",
        )

    def test_single_observation_has_finite_loss_without_prediction_target(self):
        trainer = self.make_trainer()
        batch = trainer.stream.sample(8, delay=0)
        batch.observations = {k: value[:, -1:] for k, value in batch.observations.items()}
        for name in ("targets", "visual_targets", "auditory_targets"):
            setattr(batch, name, getattr(batch, name)[:, -1:])
        output = trainer.model(batch.observations)
        loss, components = learning_loss(output, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(components["prediction"], 0.0)
        loss.backward()

    def test_replay_is_bounded_and_owns_stored_examples(self):
        replay = EpisodeReplay(capacity=5, seed=9)
        stream = TaskStream(11)
        for _ in range(4):
            batch = stream.sample(17, tasks=("visual_match",), delay=1)
            replay.add(batch)
            self.assertEqual(len(replay), 5)
            # Later caller mutations must not corrupt the saved observations
            # or targets used for rehearsal.
            for tensor in batch.observations.values():
                tensor.zero_()
            batch.targets.fill_(-99)
        sampled = replay.sample(64)
        self.assertEqual(tuple(sampled.targets.shape), (64, 3))
        self.assertTrue(torch.all(sampled.observations["visual"][..., :4].sum(-1) == 1))
        self.assertTrue(torch.all(sampled.observations["auditory"][..., :4].sum(-1) == 1))
        self.assertTrue(torch.all(sampled.targets[:, :-1] == -100))
        self.assertTrue(torch.all((sampled.targets[:, -1] >= 0) & (sampled.targets[:, -1] < 4)))
        self.assertEqual(replay.state_dict()["seen"], 68)
        self.assertEqual(len(replay), 5)

    def test_replay_roundtrip_restores_sampling_and_future_reservoir_updates(self):
        replay = EpisodeReplay(capacity=7, seed=97)
        stream = TaskStream(98)
        replay.add(stream.sample(31, delay=1))
        replay.sample(5)  # Saving must preserve an already advanced RNG.
        saved = io.BytesIO()
        torch.save(replay.state_dict(), saved)
        saved.seek(0)
        restored = EpisodeReplay(capacity=1, seed=999)
        restored.load_state_dict(torch.load(saved, weights_only=True))
        self.assertEqual(len(restored), 7)
        self.assert_batch_equal(replay.sample(29), restored.sample(29))
        more = stream.sample(37, delay=1)
        replay.add(more)
        restored.add(more)
        self.assertEqual(len(restored), 7)
        self.assert_nested_equal(replay.state_dict(), restored.state_dict())
        self.assert_batch_equal(replay.sample(29), restored.sample(29))

    def test_disabled_replay_stores_nothing_and_empty_sampling_is_rejected(self):
        replay = EpisodeReplay(capacity=0)
        replay.add(TaskStream().sample(12))
        self.assertEqual(len(replay), 0)
        with self.assertRaises(ValueError):
            replay.sample(1)
        active = EpisodeReplay(capacity=3)
        with self.assertRaises(ValueError):
            active.sample(1)
        active.add(TaskStream().sample(4))
        with self.assertRaises(ValueError):
            active.sample(0)

    def test_checkpoint_resume_produces_identical_next_update(self):
        trainer = self.make_trainer()
        for _ in range(2):
            trainer.history.append(trainer.train_step())
        self.assertEqual(trainer.step, 2)
        self.assertGreater(len(trainer.replay), 0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            trainer.save(checkpoint)
            saved_history = deepcopy(trainer.history)
            expected_metrics = trainer.train_step()
            expected_model = deepcopy(trainer.model.state_dict())
            expected_optimizer = deepcopy(trainer.optimizer.state_dict())
            expected_replay = deepcopy(trainer.replay.state_dict())
            expected_stream_rng = trainer.stream.generator.get_state().clone()
            expected_global_rng = torch.get_rng_state().clone()

            restored = Trainer.load(checkpoint, device="cpu")
            self.assertEqual(restored.step, 2)
            self.assertEqual(restored.history, saved_history)
            self.assertEqual(restored.config, trainer.config)
            self.assertEqual(restored.model.config, trainer.model.config)
            actual_metrics = restored.train_step()
            self.assertEqual(actual_metrics, expected_metrics)
            self.assertEqual(restored.step, 3)
            self.assert_nested_equal(expected_model, restored.model.state_dict())
            self.assert_nested_equal(expected_optimizer, restored.optimizer.state_dict())
            self.assert_nested_equal(expected_replay, restored.replay.state_dict())
            self.assertTrue(torch.equal(expected_stream_rng, restored.stream.generator.get_state()))
            self.assertTrue(torch.equal(expected_global_rng, torch.get_rng_state()))
            self.assert_batch_equal(trainer.stream.sample(8, delay=1), restored.stream.sample(8, delay=1))
            self.assert_batch_equal(trainer.replay.sample(11), restored.replay.sample(11))

    def test_evaluate_preserves_model_training_state_and_training_randomness(self):
        trainer = self.make_trainer()
        trainer.train_step()
        for training_mode in (True, False):
            with self.subTest(training_mode=training_mode):
                trainer.model.train(training_mode)
                before_model = deepcopy(trainer.model.state_dict())
                before_gradients = [None if p.grad is None else p.grad.clone()
                                    for p in trainer.model.parameters()]
                before_optimizer = deepcopy(trainer.optimizer.state_dict())
                before_replay = deepcopy(trainer.replay.state_dict())
                before_stream_rng = trainer.stream.generator.get_state().clone()
                before_global_rng = torch.get_rng_state().clone()
                before_step = trainer.step
                result = evaluate(trainer.model, seed=100051, batches=1,
                                  batch_size=8, delay=5, device="cpu")
                self.assertEqual(trainer.model.training, training_mode)
                self.assert_nested_equal(before_model, trainer.model.state_dict())
                self.assert_nested_equal(before_optimizer, trainer.optimizer.state_dict())
                self.assert_nested_equal(before_replay, trainer.replay.state_dict())
                self.assertTrue(torch.equal(before_stream_rng, trainer.stream.generator.get_state()))
                self.assertTrue(torch.equal(before_global_rng, torch.get_rng_state()))
                self.assertEqual(trainer.step, before_step)
                self.assert_nested_equal(before_gradients, [p.grad for p in trainer.model.parameters()])
                self.assertEqual(set(result["tasks"]), set(TASK_NAMES))
                self.assertTrue(0 <= result["macro_accuracy"] <= 1)
                self.assertTrue(all(item["episodes"] == 8 for item in result["tasks"].values()))
                repeated = evaluate(trainer.model, seed=100051, batches=1,
                                    batch_size=8, delay=5, device="cpu")
                self.assertEqual(result, repeated)


if __name__ == "__main__":
    unittest.main()
