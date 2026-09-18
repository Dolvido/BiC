"""The baseline preserves loss weighting and complete resumable learning state."""
import unittest

import torch
from torch.nn import functional as F

from brain_in_computer.language import ByteCodec
from experiments.cognitive_curriculum import generate_cognitive
from experiments.sequence_data import pack_cognitive_episodes
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.sequence_training import SequenceTrainer, sequence_objective


class SequenceTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def config(self):
        return SequenceConfig(width=16, layers=1, heads=2, feedforward=32)

    def banks(self):
        return {family: generate_cognitive(30_000_000 + index * 10_000, 4, family=family)
                for index, family in enumerate(("variable_binding", "arithmetic_updates"))}

    def compare(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.compare(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.compare(a, b)
        else:
            self.assertEqual(left, right)

    def test_parallel_objective_matches_independent_per_turn_weighting(self):
        rows = self.banks()["variable_binding"]
        batch = pack_cognitive_episodes(rows, training=True)
        model = build_sequence_student(77, config=self.config())
        output = model(**batch["inputs"], decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
        actual = sequence_objective(output, batch)
        labels = batch["supervision"]["action_targets"]
        logits = output["logits"]
        action = torch.stack([F.cross_entropy(logits[labels.eq(target)], labels[labels.eq(target)])
            for target in range(3) if bool(labels.eq(target).any())]).mean()
        action = action + .25 * F.cross_entropy(logits[labels.eq(3)], labels[labels.eq(3)])
        replies, observations = [], []
        eos = batch["inputs"]["eos_positions"]
        obs_targets = batch["supervision"]["observation_next_byte_targets"]
        for turn in range(6):
            replies.append(F.cross_entropy(output["language_logits"][:, turn].flatten(0, 1),
                batch["supervision"]["reply_targets"][:, turn].flatten(), ignore_index=ByteCodec.PAD))
            predicted, targets = [], []
            for row in range(len(rows)):
                start = 0 if turn == 0 else int(eos[row, turn - 1]) + 1
                end = int(eos[row, turn])
                predicted.append(output["observation_language_logits"][row, start:end])
                targets.append(obs_targets[row, start:end])
            observations.append(F.cross_entropy(torch.cat(predicted), torch.cat(targets)))
        expected = action + .1 * torch.stack(replies).mean() + .1 * torch.stack(observations).mean()
        torch.testing.assert_close(actual["loss"], expected, rtol=1e-6, atol=1e-6)

    def test_resume_matches_optimizer_weights_and_independent_family_streams(self):
        trainer = SequenceTrainer(self.banks(), seed=77, config=self.config(), batch_size=2)
        trainer.step("variable_binding")
        saved = trainer.snapshot()
        trainer.step("arithmetic_updates")
        resumed = SequenceTrainer(self.banks(), seed=77, config=self.config(), batch_size=2, payload=saved)
        resumed.step("arithmetic_updates")
        self.compare(trainer.snapshot(), resumed.snapshot())
        with self.assertRaisesRegex(ValueError, "recipe"):
            SequenceTrainer(self.banks(), seed=77, config=self.config(), batch_size=2,
                            learning_rate=.003, payload=saved)
        with self.assertRaisesRegex(ValueError, "admitted"):
            trainer.step("graph_reachability")


if __name__ == "__main__":
    unittest.main()
