import copy
import unittest

import torch

from experiments.cognitive_curriculum import generate_cognitive
from experiments.train_cognitive import CognitiveTrainer, TRAIN_FAMILIES, family_at


class CognitiveTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_schedules_preserve_per_family_exposures(self):
        for mode in ("interleaved", "blocked"):
            rows = [family_at(i, 18, mode) for i in range(18)]
            self.assertEqual({f: rows.count(f) for f in TRAIN_FAMILIES}, {f: 6 for f in TRAIN_FAMILIES})
        self.assertEqual([family_at(i, 6, "blocked") for i in range(6)],
                         [f for f in TRAIN_FAMILIES for _ in range(2)])
        with self.assertRaises(ValueError):
            family_at(0, 7, "interleaved")

    def test_optimizer_sampler_and_neural_weights_resume_exactly(self):
        family = TRAIN_FAMILIES[0]
        banks = {family: generate_cognitive(10, 4, family=family)}
        for mode in ("episodic", "recurrent"):
            with self.subTest(mode=mode):
                full = CognitiveTrainer(banks, seed=2, memory_mode=mode, batch_size=2)
                full.step(family)
                saved = full.snapshot()
                full.step(family)
                resumed = CognitiveTrainer(banks, seed=2, memory_mode=mode, batch_size=2, payload=saved)
                resumed.step(family)
                self.assertEqual(full.family_updates, resumed.family_updates)
                for key, value in full.model.state_dict().items():
                    torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
                self.assertTrue(torch.equal(full.generators[family].get_state(), resumed.generators[family].get_state()))
                self.assertEqual(full.optimizer.state_dict()["param_groups"], resumed.optimizer.state_dict()["param_groups"])
                for pid, state in full.optimizer.state_dict()["state"].items():
                    for key, value in state.items():
                        torch.testing.assert_close(value, resumed.optimizer.state_dict()["state"][pid][key], rtol=0, atol=0)

    def test_training_rejects_heldout_tampering_and_changed_resume_recipe(self):
        family = TRAIN_FAMILIES[0]
        heldout = {family: generate_cognitive(10, 4, family=family, split="dev")}
        with self.assertRaisesRegex(ValueError, "train"):
            CognitiveTrainer(heldout, batch_size=2)
        rows = generate_cognitive(10, 4, family=family)
        tampered = copy.deepcopy(rows)
        tampered[0]["turns"][-1]["target"] = 3
        with self.assertRaises(ValueError):
            CognitiveTrainer({family: tampered}, batch_size=2)
        saved = CognitiveTrainer({family: rows}, batch_size=2).snapshot()
        with self.assertRaisesRegex(ValueError, "recipe"):
            CognitiveTrainer({family: rows}, batch_size=4, payload=saved)
        with self.assertRaisesRegex(ValueError, "recipe"):
            CognitiveTrainer({family: generate_cognitive(20, 4, family=family)}, batch_size=2, payload=saved)


if __name__ == "__main__":
    unittest.main()
