"""Variable-length packing, byte boundaries and canonical admission."""
import copy
import unittest
from unittest.mock import Mock

import torch

from brain_in_computer.language import ByteCodec
from experiments.cognitive_curriculum import REPLIES
from experiments.composition_data import pack_composition_episodes, pack_observations
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.sequence_training import sequence_objective


def synthetic_rows(turns=8):
    zero = {name: [[0.] * width] for name, width in
            (("visual", 32), ("auditory", 4), ("body", 4), ("feedback", 2))}
    zero["tokens"] = [0]
    return [{"split": "train", "structure_partition": "train", "family": "fixture",
             "variant": row, "turns": [{"text": f"é {index}" if row else "",
                "observations": copy.deepcopy(zero), "target": index % 4,
                "reply": REPLIES[index % 4]} for index in range(turns)]} for row in range(2)]


class CompositionDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_every_actual_turn_count_and_exact_utf8_boundaries(self):
        for turns in range(2, 13):
            rows = synthetic_rows(turns)
            packed = pack_composition_episodes(rows)
            inputs, labels = packed["inputs"], packed["supervision"]
            self.assertEqual(inputs["eos_positions"].shape, (2, turns))
            self.assertEqual(labels["action_targets"].shape, (2, turns))
            for b, row in enumerate(rows):
                start = 0
                for t, turn in enumerate(row["turns"]):
                    end = int(inputs["eos_positions"][b, t])
                    expected = ByteCodec().encode(turn["text"])
                    self.assertEqual(inputs["token_ids"][b, start:end + 1].tolist(), expected)
                    self.assertEqual(labels["observation_next_byte_targets"][b, start:end + 1].tolist(), expected[1:] + [0])
                    reply = ByteCodec().encode(turn["reply"])[1:]
                    self.assertEqual(labels["reply_targets"][b, t, :len(reply)].tolist(), reply)
                    start = end + 1
                self.assertEqual(start, int(inputs["lengths"][b]))
                self.assertTrue(inputs["token_ids"][b, start:].eq(0).all())
                self.assertTrue(labels["observation_next_byte_targets"][b, start:].eq(0).all())

    def test_only_text_enters_inputs_and_evaluation_never_calls_validator(self):
        rows = synthetic_rows(12)
        changed = copy.deepcopy(rows)
        for row in changed:
            row.update(family="elsewhere", recipe={"answer": "secret"}, split="audit")
            for turn in row["turns"]:
                turn["target"] = (turn["target"] + 1) % 4
                turn["reply"] = REPLIES[turn["target"]]
        forbidden = Mock(side_effect=AssertionError("oracle entered evaluation"))
        first = pack_composition_episodes(rows, pair_validator=forbidden)
        second = pack_composition_episodes(changed, pair_validator=forbidden)
        forbidden.assert_not_called()
        self.assertEqual(set(first["inputs"]), {"token_ids", "valid_mask", "lengths", "eos_positions"})
        for name in first["inputs"]:
            self.assertTrue(torch.equal(first["inputs"][name], second["inputs"][name]))
        self.assertFalse(torch.equal(first["supervision"]["action_targets"], second["supervision"]["action_targets"]))

    def test_training_requires_complete_canonical_pairs_and_train_structure(self):
        rows = synthetic_rows()
        validator = Mock(return_value=True)
        pack_composition_episodes(rows, training=True, pair_validator=validator)
        validator.assert_called_once_with(rows)
        for field in ("split", "structure_partition"):
            bad = copy.deepcopy(rows)
            bad[0][field] = "audit"
            with self.assertRaisesRegex(ValueError, "train"):
                pack_composition_episodes(bad, training=True, pair_validator=validator)
        with self.assertRaisesRegex(ValueError, "pairs"):
            pack_composition_episodes(rows[:1], training=True, pair_validator=validator)
        with self.assertRaisesRegex(ValueError, "admission"):
            pack_composition_episodes(rows, training=True, pair_validator=lambda _: False)

    def test_overflow_shapes_and_sensor_information_are_rejected(self):
        for text in ("é" * 65, "x" * 129):
            for blank in (True, False):
                with self.assertRaises(ValueError):
                    pack_observations([[text]], blank_text=blank)
        with self.assertRaisesRegex(ValueError, "context"):
            pack_observations([["abc"] * 12], max_context_tokens=59)
        for raw in ([], [[]], [["a"] * 13], [["a"], ["a", "b"]], [[None]]):
            with self.assertRaises(ValueError):
                pack_observations(raw)
        for rows in (synthetic_rows(1), synthetic_rows(13), [synthetic_rows(2)[0], synthetic_rows(3)[0]]):
            with self.assertRaises(ValueError):
                pack_composition_episodes(rows)
        rows = synthetic_rows()
        rows[0]["turns"][0]["observations"]["body"][0][0] = 1
        with self.assertRaisesRegex(ValueError, "zero"):
            pack_composition_episodes(rows)

    def test_blank_reset_and_caller_independence(self):
        rows = synthetic_rows(12)
        original = copy.deepcopy(rows)
        packed = pack_composition_episodes(rows, blank_text=True)
        self.assertEqual(packed["inputs"]["token_ids"][0].tolist(), [1, 2] * 12)
        self.assertEqual(packed["supervision"]["observation_next_byte_targets"][0].tolist(), [2, 0] * 12)
        raw = pack_observations([["Outside every curriculum grammar."]])
        self.assertEqual(raw["eos_positions"].shape, (1, 1))
        packed["inputs"]["token_ids"].zero_()
        packed["supervision"]["action_targets"].zero_()
        self.assertEqual(rows, original)

    def test_existing_sequence_objective_trains_actual_twelve_turn_batch(self):
        model = build_sequence_student(2801, config=SequenceConfig(width=16, layers=1,
            heads=2, feedforward=32, max_turns=12))
        batch = pack_composition_episodes(synthetic_rows(12))
        output = model(**batch["inputs"], decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
        self.assertEqual(output["logits"].shape, (2, 12, 4))
        loss = sequence_objective(output, batch)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.tokens.weight.grad.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
