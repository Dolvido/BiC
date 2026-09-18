"""Independent causal and persistence checks for regional recall integration.

These checks use small random networks. Learned capability belongs in the held-
out experiment report; these tests establish the routes that can produce it.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from brain_in_computer.associative import (
    AssociativeMemory, EncoderConfig, GlyphEncoder, render_glyph,
)
from brain_in_computer.computer_use.model import ComputerBrain, ComputerConfig
from brain_in_computer.computer_use.training import load_computer_checkpoint
from brain_in_computer.regional_memory import (
    MemoryControlConfig, RegionalMemoryAgent, load_regional_checkpoint,
    normalize_instruction, save_regional_checkpoint,
)


class RegionalMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(951)
        self.computer = ComputerBrain(ComputerConfig(
            hidden_size=12, language_hidden_size=16, embedding_size=8,
            visual_features=8,
        ))
        self.encoder = GlyphEncoder(EncoderConfig(16))
        self.agent = RegionalMemoryAgent(self.computer, self.encoder,
                                         MemoryControlConfig(steps=2)).eval()
        self.scores = torch.tensor([[.15, .95, -.2, .35],
                                    [.85, -.1, .2, .4]])
        self.known = torch.ones(2, 1)
        self.prompts = ["find <name>", "choose the object left of <name>"]

    @staticmethod
    def gradient_sum(module):
        gradients = [parameter.grad for parameter in module.parameters()
                     if parameter.grad is not None]
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError("nonfinite gradient")
        return sum(float(gradient.abs().sum()) for gradient in gradients)

    def observations(self, time=3):
        brain = self.computer.brain
        return {
            "visual": torch.rand(2, time, brain.config.visual_dim),
            "auditory": torch.rand(2, time, brain.config.auditory_dim),
            "body": torch.rand(2, time, brain.config.body_dim),
            "feedback": torch.zeros(2, time, 2),
            "tokens": torch.zeros(2, time, dtype=torch.long),
        }

    def test_no_recall_and_zero_recall_preserve_legacy_core_exactly(self):
        observations = self.observations()
        with torch.no_grad():
            omitted = self.computer.brain(observations, return_activity=True)
            explicit = self.computer.brain(observations, return_activity=True,
                                           memory_context=None)
            zero = self.computer.brain(observations, return_activity=True,
                                       memory_context=torch.zeros(2, 3, 12))
        for key in ("logits", "visual_logits", "auditory_logits", "prediction", "value"):
            self.assertTrue(torch.equal(omitted[key], explicit[key]), key)
            self.assertTrue(torch.equal(omitted[key], zero[key]), key)
        for name, activity in omitted["region_activity"].items():
            self.assertTrue(torch.equal(activity, zero["region_activity"][name]), name)

    def test_future_recall_cannot_alter_past_actions(self):
        observations = self.observations()
        recall = torch.randn(2, 3, 12)
        changed = recall.clone()
        changed[:, -1] += torch.randn(2, 12)
        with torch.no_grad():
            first = self.computer.brain(observations, memory_context=recall)
            second = self.computer.brain(observations, memory_context=changed)
        self.assertTrue(torch.equal(first["logits"][:, :-1], second["logits"][:, :-1]))
        self.assertFalse(torch.equal(first["logits"][:, -1], second["logits"][:, -1]))

    def test_hippocampal_lesion_removes_external_recall_causally(self):
        observations = self.observations()
        first_context = torch.randn(2, 3, 12)
        second_context = first_context + torch.randn(2, 3, 12)
        with torch.no_grad():
            intact = [self.computer.brain(observations, memory_context=context,
                                          return_activity=True)
                      for context in (first_context, second_context)]
            lesioned = [self.computer.brain(observations, memory_context=context,
                                            return_activity=True, ablate=("hippocampus",))
                        for context in (first_context, second_context)]
        self.assertFalse(torch.equal(intact[0]["logits"], intact[1]["logits"]))
        self.assertTrue(torch.equal(lesioned[0]["logits"], lesioned[1]["logits"]))
        self.assertEqual(int(torch.count_nonzero(lesioned[0]["region_activity"]["hippocampus"])), 0)

    def test_recall_validation_rejects_wrong_shape_dtype_and_nonfinite_values(self):
        observations = self.observations()
        for context in (torch.zeros(2, 12), torch.zeros(2, 2, 12),
                        torch.zeros(2, 3, 11), torch.zeros(2, 3, 12).double(),
                        torch.full((2, 3, 12), float("nan"))):
            with self.subTest(shape=context.shape, dtype=context.dtype):
                with self.assertRaises(ValueError):
                    self.computer.brain(observations, memory_context=context)

    def test_action_learning_uses_shared_hippocampal_and_motor_networks(self):
        output = self.agent.forward_evidence(self.scores, self.known, self.prompts)
        self.assertEqual(tuple(output["logits"].shape), (2, 2, 11))
        self.assertEqual(set(output["region_activity"]), set(self.computer.brain.region_names))
        F.cross_entropy(output["logits"][:, -1], torch.tensor([1, 10])).backward()
        for name in ("hippocampus", "prefrontal_cortex", "basal_ganglia", "motor_cortex"):
            self.assertGreater(self.gradient_sum(self.computer.brain.regions[name]), 0, name)
        self.assertGreater(self.gradient_sum(self.computer.brain.regions["hippocampus"].output), 0)
        self.assertGreater(self.gradient_sum(self.computer.language.posterior_temporal), 0)
        # At least one added parameter must receive the action gradient. Shared
        # computer and frozen visual encoder parameters do not satisfy this.
        shared_ids = {id(parameter) for parameter in self.computer.parameters()}
        shared_ids.update(id(parameter) for parameter in self.encoder.parameters())
        added = [parameter for parameter in self.agent.parameters()
                 if id(parameter) not in shared_ids]
        self.assertTrue(added)
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in added
                               if parameter.grad is not None), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.encoder.parameters()))

    def test_blank_memory_removes_scores_and_known_flag(self):
        with torch.no_grad():
            first = self.agent.forward_evidence(self.scores, self.known, self.prompts,
                                                 blank_memory=True)
            second = self.agent.forward_evidence(-self.scores, torch.zeros_like(self.known),
                                                  self.prompts, blank_memory=True)
        self.assertTrue(torch.equal(first["logits"], second["logits"]))
        self.assertTrue(torch.equal(first["language_logits"], second["language_logits"]))

    def test_language_lesion_removes_instruction_dependence(self):
        other = ["choose the object below <name>", "find <name>"]
        with torch.no_grad():
            intact = self.agent.forward_evidence(self.scores, self.known, self.prompts)
            altered = self.agent.forward_evidence(self.scores, self.known, other)
            blocked = self.agent.forward_evidence(self.scores, self.known, self.prompts,
                                                   ablate=("temporal_language",))
            blocked_other = self.agent.forward_evidence(self.scores, self.known, other,
                                                         ablate=("temporal_language",))
        self.assertFalse(torch.equal(intact["logits"], altered["logits"]))
        self.assertTrue(torch.equal(blocked["logits"], blocked_other["logits"]))

    def test_retrieval_supplies_all_scores_without_using_existing_decision(self):
        crops = torch.stack([render_glyph(pattern, 770 + i)
                             for i, pattern in enumerate((31823, 51191, 28665, 47421))])
        memory = AssociativeMemory(self.encoder)
        memory.teach("dax", crops[2])
        with patch.object(AssociativeMemory, "find", side_effect=AssertionError("find bypass")), \
             patch.object(AssociativeMemory, "_decision", side_effect=AssertionError("decision bypass")):
            scores, known = self.agent.prepare_evidence(memory, "dax", crops)
            missing_scores, missing_known = self.agent.prepare_evidence(memory, "untaught", crops)
        with torch.no_grad():
            expected = self.encoder(crops) @ self.encoder(crops[2:3]).T
        self.assertEqual(scores.numel(), 4)
        self.assertTrue(torch.allclose(scores.flatten(), expected.flatten(), atol=1e-6))
        self.assertEqual(known.numel(), 1)
        self.assertEqual(float(known.flatten()[0]), 1)
        self.assertEqual(float(missing_known.flatten()[0]), 0)
        self.assertEqual(int(torch.count_nonzero(missing_scores)), 0)

    def test_motor_logits_remain_the_only_public_action_source(self):
        crops = torch.stack([render_glyph(pattern, 110 + i)
                             for i, pattern in enumerate((31823, 51191, 28665, 47421))])
        memory = AssociativeMemory(self.encoder)
        memory.teach("dax", crops[0])
        calls = []

        def replace_logits(module, arguments, output):
            calls.append(1)
            logits = torch.full_like(output[1], -100)
            logits[:, 2] = 100
            return output[0], logits

        handle = self.computer.brain.regions["motor_cortex"].register_forward_hook(replace_logits)
        try:
            with patch.object(AssociativeMemory, "find", side_effect=AssertionError("find bypass")):
                result = self.agent.act(memory, 'find "dax"', crops)
        finally:
            handle.remove()
        self.assertTrue(calls)
        self.assertEqual(result["index"], 2)
        self.assertFalse(result["stopped"])

    def test_exact_label_normalization_preserves_instruction_words(self):
        label, normalized = normalize_instruction('choose the object left of "dax"')
        self.assertEqual(label, "dax")
        self.assertIn("left of", normalized)
        self.assertIn("<name>", normalized)
        self.assertNotIn("dax", normalized)
        other_label, other_normalized = normalize_instruction('choose the object left of "wug"')
        self.assertEqual(other_label, "wug")
        self.assertEqual(normalized, other_normalized)

    def test_checkpoint_restoration_preserves_logits_without_reteaching(self):
        with torch.no_grad():
            expected = self.agent.forward_evidence(self.scores, self.known, self.prompts)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "regional.pt"
            save_regional_checkpoint(path, self.agent, metadata={"purpose": "route test"})
            with patch.object(AssociativeMemory, "teach", side_effect=AssertionError("reteaching")):
                restored = load_regional_checkpoint(path).eval()
                desktop = load_computer_checkpoint(path).eval()
                with torch.no_grad():
                    actual = restored.forward_evidence(self.scores, self.known, self.prompts)
            for name, value in self.computer.state_dict().items():
                self.assertTrue(torch.equal(value, desktop.state_dict()[name]), name)
        self.assertTrue(torch.equal(expected["logits"], actual["logits"]))
        self.assertTrue(torch.equal(expected["language_logits"], actual["language_logits"]))

    def test_unsupported_precision_is_rejected_before_encoder_mutation(self):
        for computer_dtype, encoder_dtype in ((torch.float64, torch.float32),
                                              (torch.float32, torch.float64)):
            computer = ComputerBrain(ComputerConfig(
                hidden_size=12, language_hidden_size=16, embedding_size=8,
                visual_features=8,
            )).to(dtype=computer_dtype)
            encoder = GlyphEncoder(EncoderConfig(16)).to(dtype=encoder_dtype)
            before = {name: value.clone() for name, value in encoder.state_dict().items()}
            with self.subTest(computer=computer_dtype, encoder=encoder_dtype):
                with self.assertRaises(ValueError):
                    RegionalMemoryAgent(computer, encoder)
                self.assertEqual(next(encoder.parameters()).dtype, encoder_dtype)
                self.assertTrue(all(parameter.requires_grad for parameter in encoder.parameters()))
                self.assertTrue(encoder.training)
                self.assertTrue(all(torch.equal(before[name], value)
                                    for name, value in encoder.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
