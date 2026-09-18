"""Shared decision credit has no inference shortcut and resumes exactly."""
import copy
import unittest

import torch

from brain_in_computer.dialogue_student import _run_turn, encode_dialogues
from experiments.cognitive_credit import CreditTrainer, balanced_query_loss, build_credit_student
from experiments.cognitive_curriculum import generate_cognitive
from experiments.cognitive_student import build_cognitive_student
from experiments.train_cognitive import CognitiveTrainer


class CognitiveCreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def bank(self):
        return {family: generate_cognitive(1200 + index * 100, 4, family=family)
                for index, family in enumerate(("variable_binding", "arithmetic_updates"))}

    def same_tree(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.same_tree(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.same_tree(a, b)
        else:
            self.assertEqual(left, right)

    def test_seeded_base_initialization_matches_and_both_arms_have_same_parameters(self):
        before = torch.random.get_rng_state().clone()
        credit = build_credit_student(37)
        base = build_cognitive_student(37, memory_mode="recurrent")
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, credit.state_dict()[name]))
        self.assertEqual(credit.parameter_counts()["decision_credit"], 260)
        self.assertEqual(sum(credit.parameter_counts().values()), sum(p.numel() for p in credit.parameters()))
        control = CreditTrainer(self.bank(), seed=37, batch_size=2, aux_weight=0.)
        supervised = CreditTrainer(self.bank(), seed=37, batch_size=2, aux_weight=.3)
        self.same_tree(control.model.state_dict(), supervised.model.state_dict())
        self.assertEqual(control.model.parameter_counts(), supervised.model.parameter_counts())

    def test_auxiliary_head_mutation_or_removal_cannot_change_official_inference(self):
        model = build_credit_student(37)
        episodes = generate_cognitive(1200, 2, family="variable_binding")
        batches = encode_dialogues(model, episodes)
        reference = build_cognitive_student(37, memory_mode="recurrent")
        reference.load_state_dict({key: value for key, value in model.state_dict().items()
                                   if not key.startswith("credit_head.")})
        a_state = b_state = None
        for batch in batches:
            original, a_state = _run_turn(model, batch, a_state)
            stripped, b_state = _run_turn(reference, batch, b_state)
            for name in ("logits", "language_logits", "production_context", "observation_language_logits"):
                self.assertTrue(torch.equal(original[name], stripped[name]))
            self.same_tree(model.brain.state_to_dict(a_state.brain_state),
                           reference.brain.state_to_dict(b_state.brain_state))
        before, old_state = _run_turn(model, batches[0], None)
        with torch.no_grad():
            model.credit_head.weight.fill_(42.)
            model.credit_head.bias.fill_(-10.)
        after, new_state = _run_turn(model, batches[0], None)
        self.assertFalse(torch.equal(before["auxiliary_logits"], after["auxiliary_logits"]))
        for name in ("logits", "language_logits", "production_context"):
            self.assertTrue(torch.equal(before[name], after[name]))
        self.same_tree(model.brain.state_to_dict(old_state.brain_state),
                       model.brain.state_to_dict(new_state.brain_state))
        self.same_tree(old_state.token_memory, new_state.token_memory)

    def test_auxiliary_query_loss_reaches_prefrontal_and_excludes_ack(self):
        model = build_credit_student(37)
        episodes = generate_cognitive(1200, 2, family="variable_binding")
        batch = encode_dialogues(model, episodes)[0]
        output, _ = _run_turn(model, batch, None)
        activity = output["region_activity"]["prefrontal_cortex"]
        activity.retain_grad()
        loss = balanced_query_loss(output["auxiliary_logits"], torch.tensor([0, 1]))
        loss.backward()
        self.assertGreater(float(activity.grad.abs().sum()), 0.)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.brain.regions["prefrontal_cortex"].parameters()), 0.)
        logits = torch.randn(4, 4, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 3])
        expected = .5 * (torch.nn.functional.cross_entropy(logits[:2], labels[:2])
                         + torch.nn.functional.cross_entropy(logits[2:3], labels[2:3]))
        actual = balanced_query_loss(logits, labels)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        self.assertEqual(float(logits.grad[3].abs().sum()), 0.)

    def test_zero_weight_preserves_original_training_update(self):
        banks = self.bank()
        original = CognitiveTrainer(banks, seed=37, batch_size=2, memory_mode="recurrent")
        control = CreditTrainer(banks, seed=37, batch_size=2, aux_weight=0.)
        expected = original.step("variable_binding")
        actual = control.step("variable_binding")
        for key in ("loss", "action_loss", "reply_loss", "observation_language_loss"):
            self.assertEqual(expected[key], actual[key])
        for name, value in original.model.state_dict().items():
            torch.testing.assert_close(value, control.model.state_dict()[name], rtol=0, atol=0)

    def test_all_weights_optimizer_and_family_samplers_resume_exactly(self):
        for weight in (0., .3):
            with self.subTest(weight=weight):
                banks = self.bank()
                full = CreditTrainer(banks, seed=37, batch_size=2, aux_weight=weight)
                full.step("variable_binding")
                saved = full.snapshot()
                full.step("arithmetic_updates")
                resumed = CreditTrainer(banks, seed=37, batch_size=2, aux_weight=weight, payload=saved)
                metrics = resumed.step("arithmetic_updates")
                self.same_tree(full.snapshot(), resumed.snapshot())
                self.assertIn("auxiliary_query_loss", metrics)
                self.assertIn("auxiliary_query_accuracy", metrics)
                with self.assertRaisesRegex(ValueError, "recipe"):
                    CreditTrainer(banks, seed=37, batch_size=2, aux_weight=1 - weight, payload=saved)

    def test_canonical_boundary_and_invalid_weight_rejected(self):
        family = "variable_binding"
        with self.assertRaisesRegex(ValueError, "train"):
            CreditTrainer({family: generate_cognitive(1200, 2, split="dev", family=family)}, batch_size=2)
        bad = self.bank()
        bad[family][0]["turns"][0]["target"] = 2
        with self.assertRaises(ValueError):
            CreditTrainer(bad, batch_size=2)
        for weight in (-1., float("inf"), float("nan"), True):
            with self.assertRaisesRegex(ValueError, "aux_weight"):
                CreditTrainer(self.bank(), batch_size=2, aux_weight=weight)


if __name__ == "__main__":
    unittest.main()
