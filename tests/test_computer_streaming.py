"""Persistent screen/text episodes retain the existing ComputerBrain checkpoint."""

import copy
import io
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from brain_in_computer.computer_use.model import ComputerBrain, ComputerConfig
from brain_in_computer.language import ByteCodec


class ComputerStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(129)
        self.model = ComputerBrain(ComputerConfig(hidden_size=8, language_hidden_size=12,
                                                 embedding_size=6, visual_features=8)).double()
        self.pixels = torch.rand(2, 1, 3, 32, 32, dtype=torch.float64)
        self.body = torch.rand(2, 1, 4, dtype=torch.float64)
        self.prompts = ["Remember the red button.", "Remember the blue button."]
        self.decoder, self.targets = self.model.prepare_reply(["Done.", "OK."])

    def assert_tree_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for name, value in left.items():
            if isinstance(value, dict):
                self.assert_tree_equal(value, right[name])
            else:
                self.assertTrue(torch.equal(value, right[name]), name)

    def call(self, state=None, *, pixels=None, prompts=None, model=None, decoder=None, **options):
        return (self.model if model is None else model).forward_with_state(
            self.pixels if pixels is None else pixels, self.body,
            self.prompts if prompts is None else prompts,
            self.decoder if decoder is None else decoder, state, **options,
        )

    def test_fresh_and_explicit_zero_state_match_legacy_outputs_without_new_keys(self):
        saved = copy.deepcopy(self.model.state_dict())
        with torch.no_grad():
            original = self.model(self.pixels, self.body, self.prompts, self.decoder)
            fresh, _ = self.call()
            zero, _ = self.call(self.model.brain.initial_state(2))
        self.assert_tree_equal(original, fresh)
        self.assert_tree_equal(original, zero)
        self.assert_tree_equal(saved, self.model.state_dict())
        recreated = ComputerBrain(self.model.config).double()
        incompatibilities = recreated.load_state_dict(saved, strict=True)
        self.assertEqual(incompatibilities.missing_keys, [])
        self.assertEqual(incompatibilities.unexpected_keys, [])
        with torch.no_grad():
            loaded, _ = self.call(model=recreated)
        self.assert_tree_equal(original, loaded)

    def test_prior_screens_and_language_change_later_actions_and_response_context(self):
        with torch.no_grad():
            _, state = self.call()
            _, changed_screen_state = self.call(pixels=1 - self.pixels)
            _, changed_language_state = self.call(prompts=["Remember green.", "Remember yellow."])
            question = ["Which button was it?", "Which button was it?"]
            continued, _ = self.call(state, prompts=question)
            screen_changed, _ = self.call(changed_screen_state, prompts=question)
            language_changed, _ = self.call(changed_language_state, prompts=question)
            reset, _ = self.call(prompts=question)
            for key in ("logits", "production_context", "language_logits"):
                for altered in (screen_changed, language_changed, reset):
                    self.assertFalse(torch.equal(continued[key], altered[key]), key)

    def test_serialized_activity_with_same_checkpoint_resumes_bitwise(self):
        with torch.no_grad():
            _, state = self.call()
            before = self.model.brain.state_to_dict(state)
            expected, expected_state = self.call(state, prompts=["Now click it.", "Now click it."])
            buffer = io.BytesIO()
            torch.save({"weights": self.model.state_dict(), "activity": before}, buffer)
            buffer.seek(0)
            payload = torch.load(buffer, weights_only=True, map_location="cpu")
            restarted = ComputerBrain(self.model.config).double()
            restarted.load_state_dict(payload["weights"], strict=True)
            restored = restarted.brain.state_from_dict(payload["activity"])
            actual, actual_state = self.call(restored, model=restarted, prompts=["Now click it.", "Now click it."])
            self.assert_tree_equal(expected, actual)
            self.assert_tree_equal(self.model.brain.state_to_dict(expected_state),
                                   restarted.brain.state_to_dict(actual_state))
            self.assert_tree_equal(before, self.model.brain.state_to_dict(state))

    def force_reply_token(self, token):
        with torch.no_grad():
            readout = self.model.language.inferior_frontal.readout
            readout.weight.zero_()
            readout.bias.fill_(-20)
            readout.bias[ByteCodec.PAD] = 30
            readout.bias[ByteCodec.BOS] = 25
            readout.bias[token] = 20

    def test_stateful_response_runs_core_once_masks_specials_and_restores_modes(self):
        self.force_reply_token(ord("h") + ByteCodec.BYTE_OFFSET)
        self.model.train()
        self.model.retina.eval()
        modes = [module.training for module in self.model.modules()]
        with torch.no_grad():
            _, state = self.call()
            _, expected = self.call(state, decoder=torch.full((2, 1), ByteCodec.BOS, dtype=torch.long))
        saved_state = self.model.brain.state_to_dict(state)
        with patch.object(self.model.brain, "forward_with_state", wraps=self.model.brain.forward_with_state) as core:
            replies, actual = self.model.respond_with_state(
                self.pixels, self.body, self.prompts, state, max_new_bytes=3)
        self.assertEqual(core.call_count, 1)
        self.assertEqual(replies, ["hhh", "hhh"])
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assert_tree_equal(self.model.brain.state_to_dict(expected), self.model.brain.state_to_dict(actual))
        self.assert_tree_equal(saved_state, self.model.brain.state_to_dict(state))
        self.assertFalse(actual.previous_prefrontal.requires_grad)

    def test_stateful_response_matches_legacy_and_byte_budget_does_not_advance_state(self):
        self.force_reply_token(ord("k") + ByteCodec.BYTE_OFFSET)
        old_calls = []
        hook = self.model.brain.register_forward_hook(lambda *_: old_calls.append(1))
        try:
            legacy = self.model.respond(self.pixels, self.body, self.prompts, max_new_bytes=3)
        finally:
            hook.remove()
        self.assertEqual(old_calls, [1])
        replies, state = self.model.respond_with_state(self.pixels, self.body, self.prompts, max_new_bytes=3)
        _, shorter_state = self.model.respond_with_state(self.pixels, self.body, self.prompts, max_new_bytes=1)
        self.assertEqual(replies, legacy)
        self.assert_tree_equal(self.model.brain.state_to_dict(state), self.model.brain.state_to_dict(shorter_state))
        self.force_reply_token(ByteCodec.EOS)
        ended, _ = self.model.respond_with_state(self.pixels, self.body, self.prompts, max_new_bytes=3)
        self.assertEqual(ended, ["", ""])

    def test_future_turn_loss_reaches_prior_pixels_language_bridge_and_retina(self):
        pixels = self.pixels.clone().requires_grad_()
        _, state = self.call(pixels=pixels)
        output, _ = self.call(state, prompts=["What was on screen?", "What was on screen?"])
        F.cross_entropy(output["language_logits"].flatten(0, 1), self.targets.flatten(),
                        ignore_index=ByteCodec.PAD).backward()
        self.assertIsNotNone(pixels.grad)
        self.assertGreater(float(pixels.grad.abs().sum()), 0.0)
        for module in (self.model.retina, self.model.language.posterior_temporal,
                       self.model.language.semantic_bridge, self.model.brain):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_invalid_state_or_ablation_restores_modes_and_does_not_change_weights(self):
        self.model.train()
        self.model.retina.eval()
        modes = [module.training for module in self.model.modules()]
        before = copy.deepcopy(self.model.state_dict())
        for options in ({"state": {}}, {"state": self.model.brain.initial_state(1)}, {"ablate": ("unknown",)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.model.respond_with_state(self.pixels, self.body, self.prompts, **options)
            self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assert_tree_equal(before, self.model.state_dict())


if __name__ == "__main__":
    unittest.main()
