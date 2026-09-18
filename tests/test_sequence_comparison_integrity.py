"""Independent comparison checks: matched sampling and causal training credit."""
import unittest

import torch
from torch.nn import functional as F

from experiments.cognitive_curriculum import generate_cognitive
from experiments.sequence_data import pack_observations
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.sequence_training import SequenceTrainer
from experiments.train_cognitive import CognitiveTrainer


class SequenceComparisonIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_all_three_architectures_draw_identical_complete_pairs_per_family(self):
        families = ("variable_binding", "graph_reachability", "conditional_logic")
        banks = {family: generate_cognitive(40_000_000 + index * 10_000, 8, family=family)
                 for index, family in enumerate(families)}
        trainers = [CognitiveTrainer(banks, seed=2501, memory_mode=mode, batch_size=4)
                    for mode in ("recurrent", "episodic")]
        trainers.append(SequenceTrainer(banks, seed=2501, batch_size=4,
            config=SequenceConfig(width=16, layers=1, heads=2, feedforward=32)))
        for family in families:
            for _ in range(7):
                draws = [torch.randint(len(banks[family]) // 2, (2,),
                                      generator=trainer.generators[family]) for trainer in trainers]
                self.assertTrue(torch.equal(draws[0], draws[1]))
                self.assertTrue(torch.equal(draws[0], draws[2]))
                indices = (draws[0][:, None] * 2 + torch.arange(2)[None]).flatten()
                for left, right in zip(indices[::2], indices[1::2]):
                    self.assertEqual(banks[family][int(left)]["counterfactual_group"],
                                     banks[family][int(right)]["counterfactual_group"])

    def test_earlier_action_loss_has_zero_gradient_to_future_observation_occurrences(self):
        model = build_sequence_student(2501,
            config=SequenceConfig(width=16, layers=2, heads=2, feedforward=32))
        inputs = pack_observations([["What is already known?", "This arrives later.", "An even later observation."]])
        captured = []

        def capture(module, values, output):
            output.retain_grad()
            captured.append(output)

        handle = model.tokens.register_forward_hook(capture)
        try:
            output = model(**inputs, decoder_input_ids=torch.ones(1, 3, 1, dtype=torch.long))
            F.cross_entropy(output["logits"][:, 0], torch.tensor([1])).backward()
        finally:
            handle.remove()
        boundary = int(inputs["eos_positions"][0, 0]) + 1
        self.assertGreater(float(captured[0].grad[:, :boundary].abs().sum()), 0.)
        self.assertEqual(float(captured[0].grad[:, boundary:].abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
