"""Resume and held-out boundaries for local training experiments."""
import unittest

import torch

from brain_in_computer.dialogue_curriculum import generate_dialogues
from brain_in_computer.dialogue_student import build_dialogue_student
from experiments.train_english_frontier import train_chunk


class EnglishFrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_resume_preserves_weights_optimizer_and_sampling(self):
        episodes = generate_dialogues(100, 8)
        initial = {"state_dict": build_dialogue_student(99).state_dict()}
        for arm in ("baseline", "paired", "scaffold", "text_scaffold"):
            with self.subTest(arm=arm):
                options = dict(seed=99, batch_size=4, device="cpu", arm=arm)
                whole = train_chunk(initial, episodes, steps=2, **options)
                first = train_chunk(initial, episodes, steps=1, **options)
                resumed = train_chunk(first, episodes, steps=1, **options)
                for key in whole["state_dict"]:
                    torch.testing.assert_close(whole["state_dict"][key], resumed["state_dict"][key], rtol=0, atol=0)
                self.assertTrue(torch.equal(whole["sampler_state"], resumed["sampler_state"]))
                if "heads_state" in whole:
                    for key in whole["heads_state"]:
                        torch.testing.assert_close(whole["heads_state"][key], resumed["heads_state"][key], rtol=0, atol=0)

    def test_paired_training_rejects_incomplete_pairs_and_heldout(self):
        initial = {"state_dict": build_dialogue_student(99).state_dict()}
        options = dict(seed=99, steps=1, batch_size=4, device="cpu", arm="paired")
        with self.assertRaisesRegex(ValueError, "pairs"):
            train_chunk(initial, generate_dialogues(101, 4), **options)
        with self.assertRaisesRegex(ValueError, "train"):
            train_chunk(initial, generate_dialogues(100, 4, "dev"), **options)


if __name__ == "__main__":
    unittest.main()
