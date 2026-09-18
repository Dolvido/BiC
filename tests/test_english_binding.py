"""Binding balance changes only training supervision and remains resumable."""
import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from brain_in_computer import dialogue_scaffolding
from brain_in_computer.dialogue_curriculum import generate_dialogues
from brain_in_computer.dialogue_scaffolding import ScaffoldHeads, UNKNOWN_COLOR, causal_targets
from brain_in_computer.dialogue_student import build_dialogue_student, encode_dialogues, _run_turn
from experiments import train_english_binding as binding
from experiments import train_english_frontier as frontier


class EnglishBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fixture(self):
        episodes = generate_dialogues(100, 2, "train", "grounding")
        model = build_dialogue_student(99)
        output, _ = _run_turn(model, encode_dialogues(model, episodes)[0], None)
        targets = causal_targets(episodes)
        base, balanced = ScaffoldHeads(), binding.BalancedScaffoldHeads()
        balanced.load_state_dict(base.state_dict())
        return output, targets, base, balanced

    def test_nonbinding_losses_and_parameter_layout_are_unchanged(self):
        output, targets, base, balanced = self.fixture()
        expected, actual = base.losses(output, targets, 0), balanced.losses(output, targets, 0)
        self.assertEqual(tuple(base.state_dict()), tuple(balanced.state_dict()))
        for name in expected.keys() - {"bindings"}:
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)

    def test_binding_loss_weights_strata_equally_instead_of_each_slot(self):
        output, targets, _, balanced = self.fixture()
        # One known binding and fifteen unknowns make per-slot and per-stratum
        # means distinguishable, while a fixed bias makes expectations exact.
        targets["bindings"][0].fill_(UNKNOWN_COLOR)
        targets["bindings"][0, 0, 0] = 0
        with torch.no_grad():
            balanced.bindings.weight.zero_()
            balanced.bindings.bias.copy_(torch.tensor([0., 0., 0., 0., 3.] * 8))
        logits = balanced(output)["bindings"]
        terms = F.cross_entropy(logits.reshape(-1, 5), targets["bindings"][0].reshape(-1),
                                reduction="none").reshape(2, 8)
        known = targets["bindings"][0].ne(UNKNOWN_COLOR)
        expected = .5 * (terms[known].mean() + terms[~known].mean())
        actual = balanced.losses(output, targets, 0)["bindings"]
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertGreater(abs(float((actual - terms.mean()).detach())), .1)

    def test_empty_strata_renormalize_and_all_masked_is_finite_zero(self):
        output, targets, _, balanced = self.fixture()
        for color in (0, UNKNOWN_COLOR):
            with self.subTest(color=color):
                labels = copy.deepcopy(targets)
                labels["bindings"][0].fill_(color)
                logits = balanced(output)["bindings"]
                expected = F.cross_entropy(logits.reshape(-1, 5), labels["bindings"][0].reshape(-1))
                actual = balanced.losses(output, labels, 0)["bindings"]
                torch.testing.assert_close(expected, actual)
                self.assertTrue(bool(torch.isfinite(actual)))
        targets["masks"]["bindings"][0].fill_(False)
        targets["bindings"][0].fill_(-999)
        loss = balanced.losses(output, targets, 0)["bindings"]
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(balanced.bindings.weight.grad).all())
        self.assertEqual(float(balanced.bindings.weight.grad.abs().sum()), 0.)

    def test_cpu_resume_preserves_student_heads_optimizer_and_sampling_exactly(self):
        initial = {"state_dict": build_dialogue_student(99).state_dict()}
        episodes = generate_dialogues(100, 8)
        options = dict(seed=99, batch_size=4, device="cpu")
        whole = binding.train_chunk(initial, episodes, steps=2, **options)
        first = binding.train_chunk(initial, episodes, steps=1, **options)
        resumed = binding.train_chunk(first, episodes, steps=1, **options)

        def same_tree(left, right):
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    same_tree(left[key], right[key])
            elif isinstance(left, (tuple, list)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    same_tree(a, b)
            else:
                self.assertEqual(left, right)

        for key in ("state_dict", "heads_state", "optimizer_state", "sampler_state"):
            same_tree(whole[key], resumed[key])
        self.assertIs(dialogue_scaffolding.ScaffoldHeads, ScaffoldHeads)

    def test_experimental_substitutions_restore_on_failure_and_source_is_hashed(self):
        original_train, original_sources = frontier.train_chunk, frontier.source_fingerprints
        args = type("Args", (), {"arm": binding.ARM})()
        with patch.object(frontier, "run", side_effect=RuntimeError("test")):
            with self.assertRaisesRegex(RuntimeError, "test"):
                binding.run(args)
        self.assertIs(frontier.train_chunk, original_train)
        self.assertIs(frontier.source_fingerprints, original_sources)
        with self.assertRaisesRegex(ValueError, "train"):
            binding.train_chunk({"state_dict": {}}, generate_dialogues(100, 2, "dev"),
                                seed=99, steps=1, batch_size=2, device="cpu")
        self.assertIs(dialogue_scaffolding.ScaffoldHeads, ScaffoldHeads)
        fingerprints = binding.source_fingerprints()
        path = Path(binding.__file__).resolve()
        self.assertEqual(fingerprints["experiments/train_english_binding.py"],
                         hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertIn("experiments/train_english_frontier.py", fingerprints)


if __name__ == "__main__":
    unittest.main()
