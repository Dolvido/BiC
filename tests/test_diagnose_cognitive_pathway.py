"""Read-only pathway measurements preserve model and training boundaries."""
import unittest

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments.cognitive_curriculum import generate_cognitive
from experiments.cognitive_student import build_cognitive_student
from experiments.diagnose_cognitive_pathway import diagnose_family, diagnose_model, STAGES


class CognitivePathwayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_fresh_cpu_measurements_preserve_weights_modes_gradients_and_hooks(self):
        model = build_cognitive_student(9)
        parameter = next(model.parameters())
        gradient = torch.ones_like(parameter)
        parameter.grad = gradient
        model.train()
        model.posterior_temporal.eval()
        modes = [module.training for module in model.modules()]
        digest = checkpoint_digest(model)
        results = diagnose_model(model, seed=12_000_000, count_per_level=2)
        self.assertEqual(checkpoint_digest(model), digest)
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertIs(parameter.grad, gradient)
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))
        self.assertEqual(len(results), 3)
        for row in results.values():
            self.assertEqual(row["checkpoint_sha256_before"], row["checkpoint_sha256_after"])
            self.assertGreater(row["opposite_query_pairs"], 0)
            self.assertGreater(row["changed_premise_encoder_separation"]["l2"], 0.)
            self.assertLess(row["opposite_query_stage_separation"]["encoder_final"]["l2"], 1e-6)
            self.assertEqual(set(row["opposite_query_stage_separation"]), {*STAGES, "logits"})
            gradients = row["query_only_parameter_gradients"]
            self.assertGreater(gradients["encoder"]["gradient_l2"], 0.)
            self.assertGreater(gradients["motor_cortex"]["gradient_l2"], 0.)
            self.assertEqual(gradients["reply_decoder"]["gradient_l2"], 0.)
            self.assertEqual(gradients["observation_predictor"]["gradient_l2"], 0.)

    def test_diagnostic_rejects_heldout_and_incomplete_pairs(self):
        model = build_cognitive_student(9)
        with self.assertRaisesRegex(ValueError, "train"):
            diagnose_family(model, generate_cognitive(12_000_000, 2, split="audit", family="variable_binding"))
        with self.assertRaisesRegex(ValueError, "pairs"):
            diagnose_family(model, generate_cognitive(12_000_000, 2, family="variable_binding")[:1])


if __name__ == "__main__":
    unittest.main()
