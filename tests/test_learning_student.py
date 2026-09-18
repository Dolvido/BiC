"""Isolation, data boundaries, resume fidelity, and actual student learning."""

import copy
import time
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.learning_student import (
    build_student, encode_examples, evaluate_student, train_candidate,
)


def examples(count=12, split="train"):
    result = []
    for index in range(count):
        target = index % 2
        visual = [[0.0] * 32 for _ in range(3)]
        for row in visual:
            row[target] = 1.0
        result.append({"id": f"{split}-{index}", "skill": "copy", "split": split,
                       "target": target, "prompt": "copy the symbol", "explanation": str(target),
                       "observations": {"visual": visual, "auditory": [[0.0] * 4 for _ in range(3)],
                                        "body": [[0.0] * 4 for _ in range(3)],
                                        "feedback": [[0.0] * 2 for _ in range(3)], "tokens": [1, 1, 1]}})
    return result


class LearningStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def train(self, state, corpus=None, **kwargs):
        options = dict(seed=71, hidden_size=8, steps=2, batch_size=8,
                       learning_rate=0.005, replay_fraction=0.25)
        options.update(kwargs)
        return train_candidate(state, examples() if corpus is None else corpus, **options)

    def test_initialization_and_training_preserve_global_rng_and_are_deterministic(self):
        before = torch.random.get_rng_state().clone()
        first = build_student(22, hidden_size=8)
        second = build_student(22, hidden_size=8)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]))
        run_a = self.train(first.state_dict())
        run_b = self.train(first.state_dict())
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for name, value in run_a["state_dict"].items():
            self.assertTrue(torch.equal(value, run_b["state_dict"][name]))

    def test_real_weight_updates_reduce_loss_and_learn_heldout_rule(self):
        model = build_student(22, hidden_size=8)
        initial = {name: value.clone() for name, value in model.state_dict().items()}
        before = evaluate_student(model, examples(split="test"))
        trained = self.train(initial, steps=150, batch_size=12, learning_rate=0.01)
        model.load_state_dict(trained["state_dict"])
        after = evaluate_student(model, examples(split="test"))
        self.assertLess(after["loss"], before["loss"] * 0.5)
        self.assertEqual(after["accuracy"], 1.0)
        self.assertEqual(trained["updates"], 150)
        self.assertEqual(trained["examples_seen"], 1800)
        self.assertTrue(any(not torch.equal(value, trained["state_dict"][name])
                            for name, value in initial.items()))

    def test_evaluation_preserves_parameters_modes_and_gradients(self):
        model = build_student(22, hidden_size=8)
        model.train()
        model.regions["visual_cortex"].eval()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        modes = [module.training for module in model.modules()]
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        result = evaluate_student(model, examples(split="audit"))
        self.assertEqual(result["total"], 12)
        self.assertEqual(result["per_skill"]["copy"]["total"], 12)
        self.assertGreaterEqual(result["brier"], 0.0)
        self.assertLessEqual(result["brier"], 2.0)
        self.assertEqual(modes, [module.training for module in model.modules()])
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(state[name], value))
        for parameter in model.parameters():
            self.assertTrue(torch.equal(parameter.grad, torch.ones_like(parameter)))

    def test_test_audit_and_missing_splits_cannot_enter_training_or_replay(self):
        state = build_student(22, hidden_size=8).state_dict()
        for split in ("test", "audit", "validation", None):
            with self.subTest(split=split):
                invalid = examples(split=split)
                with self.assertRaisesRegex(ValueError, "leakage"):
                    self.train(state, invalid)
                with self.assertRaisesRegex(ValueError, "leakage"):
                    self.train(state, replay_examples=invalid)

    def test_optimizer_and_sampler_resume_matches_uninterrupted_learning(self):
        state = build_student(22, hidden_size=8).state_dict()
        replay = examples(8)
        full = self.train(state, steps=6, replay_examples=replay)
        part = self.train(state, steps=3, replay_examples=replay)
        resumed = self.train(part["state_dict"], steps=3, replay_examples=replay,
                             optimizer_state=part["optimizer_state"], sampler_state=part["sampler_state"])
        for name, value in full["state_dict"].items():
            self.assertTrue(torch.equal(value, resumed["state_dict"][name]), name)
        self.assertEqual(full["replay_examples_seen"],
                         part["replay_examples_seen"] + resumed["replay_examples_seen"])

    def test_zero_updates_and_expired_deadline_preserve_the_champion(self):
        state = build_student(22, hidden_size=8).state_dict()
        for options in ({"steps": 0}, {"deadline": time.monotonic() - 1}):
            result = self.train(state, **options)
            self.assertEqual(result["updates"], 0)
            self.assertEqual(result["examples_seen"], 0)
            self.assertIsNone(result["loss"])
            for name, value in state.items():
                self.assertTrue(torch.equal(value, result["state_dict"][name]))
                self.assertNotEqual(value.data_ptr(), result["state_dict"][name].data_ptr())

    def test_labels_and_explanations_never_become_inputs(self):
        corpus = examples()
        original, _ = encode_examples(corpus)
        changed = copy.deepcopy(corpus)
        for example in changed:
            example["target"] = 3
            example["prompt"] = "the answer is 3"
            example["explanation"] = "the answer is 3"
        altered, targets = encode_examples(changed)
        for name, value in original.items():
            self.assertTrue(torch.equal(value, altered[name]))
        self.assertTrue(torch.equal(targets, torch.full_like(targets, 3)))

    def test_nonfinite_and_invalid_examples_are_rejected(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            corpus = examples()
            corpus[0]["observations"]["visual"][0][0] = invalid
            with self.assertRaises(ValueError):
                encode_examples(corpus)
        for invalid in (True, 0.3, -1, 4):
            corpus = examples()
            corpus[0]["target"] = invalid
            with self.assertRaises(ValueError):
                encode_examples(corpus)
        for invalid in (True, 0.3, -1, 16):
            corpus = examples()
            corpus[0]["observations"]["tokens"][0] = invalid
            with self.assertRaises(ValueError):
                encode_examples(corpus)
        with self.assertRaises(ValueError):
            encode_examples([])
        state = build_student(22, hidden_size=8).state_dict()
        next(iter(state.values())).flatten()[0] = float("nan")
        with self.assertRaises(ValueError):
            self.train(state)

    def test_finite_logits_with_overflowing_loss_are_rejected(self):
        model = build_student(22, hidden_size=8)
        extreme = torch.tensor([3e38, -3e38, 0.0, 0.0]).repeat(12, 3, 1)
        self.assertTrue(torch.isfinite(extreme).all())
        with patch.object(model, "forward", return_value={"logits": extreme}):
            with self.assertRaisesRegex(ValueError, "nonfinite evaluation metrics"):
                evaluate_student(model, examples(split="audit"))

    def test_replay_fraction_controls_sampling_and_invalid_values_are_rejected(self):
        state = build_student(22, hidden_size=8).state_dict()
        for fraction, expected in ((0.0, 0), (1.0, 16)):
            run = self.train(state, replay_examples=examples(4), replay_fraction=fraction)
            self.assertEqual(run["replay_examples_seen"], expected)
        for options in ({"replay_fraction": float("nan")}, {"learning_rate": float("inf")},
                        {"learning_rate": 0}, {"batch_size": 0}, {"steps": -1},
                        {"deadline": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.train(state, **options)


if __name__ == "__main__":
    unittest.main()
