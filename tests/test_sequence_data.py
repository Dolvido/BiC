"""Exact observation packing, causal target alignment and admission boundaries."""
import copy
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.language import ByteCodec
from experiments.cognitive_curriculum import REPLIES, generate_cognitive
from experiments.sequence_data import pack_cognitive_episodes, pack_observations


class SequencePackingTests(unittest.TestCase):
    def episodes(self, split="train"):
        return (generate_cognitive(2000, 2, split, "variable_binding", 1)
                + generate_cognitive(3000, 2, split, "conditional_logic", 2))

    def assert_inputs_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_exact_text_reconstruction_and_separate_supervision(self):
        rows = self.episodes()
        result = pack_cognitive_episodes(rows, training=True)
        self.assertEqual(set(result), {"inputs", "supervision"})
        inputs, supervision = result["inputs"], result["supervision"]
        self.assertEqual(set(inputs), {"token_ids", "valid_mask", "lengths", "eos_positions"})
        self.assertEqual(inputs["eos_positions"].shape, (4, 6))
        self.assertEqual(supervision["action_targets"].shape, (4, 6))
        codec = ByteCodec()
        for index, row in enumerate(rows):
            start = 0
            for turn_index, turn in enumerate(row["turns"]):
                end = int(inputs["eos_positions"][index, turn_index])
                ids = inputs["token_ids"][index, start:end + 1]
                self.assertEqual(ids.tolist(), codec.encode(turn["text"]))
                self.assertEqual(codec.decode(ids.tolist()), turn["text"])
                self.assertEqual(int(supervision["action_targets"][index, turn_index]), turn["target"])
                reply = supervision["reply_decoder_input_ids"][index, turn_index]
                self.assertEqual(codec.decode(reply.tolist()), turn["reply"])
                start = end + 1
            self.assertEqual(start, int(inputs["lengths"][index]))

    def test_next_byte_targets_mask_boundaries_but_retain_last_byte_to_eos(self):
        rows = self.episodes()
        result = pack_cognitive_episodes(rows)
        inputs, labels = result["inputs"], result["supervision"]
        targets = labels["observation_next_byte_targets"]
        codec = ByteCodec()
        for index, episode in enumerate(rows):
            start = 0
            for turn_index, turn in enumerate(episode["turns"]):
                ids = codec.encode(turn["text"])
                end = int(inputs["eos_positions"][index, turn_index])
                self.assertEqual(targets[index, start:end + 1].tolist(), ids[1:] + [ByteCodec.PAD])
                self.assertEqual(int(targets[index, end - 1]), ByteCodec.EOS)
                self.assertEqual(int(targets[index, end]), ByteCodec.PAD)
                if turn_index < 5:
                    self.assertEqual(int(inputs["token_ids"][index, end + 1]), ByteCodec.BOS)
                start = end + 1
            self.assertTrue(targets[index, start:].eq(ByteCodec.PAD).all())
        self.assertTrue(torch.equal(labels["observation_next_byte_mask"], targets.ne(ByteCodec.PAD)))

    def test_reply_shift_and_masks(self):
        result = pack_cognitive_episodes(self.episodes())
        supervision = result["supervision"]
        decoder, targets = supervision["reply_decoder_input_ids"], supervision["reply_targets"]
        self.assertEqual(decoder.shape, targets.shape)
        self.assertEqual(decoder.shape[:2], (4, 6))
        self.assertTrue(decoder[:, :, 0].eq(ByteCodec.BOS).all())
        self.assertTrue(torch.equal(decoder[:, :, 1:], targets[:, :, :-1]))
        self.assertTrue(torch.equal(supervision["reply_target_mask"], targets.ne(ByteCodec.PAD)))
        for index, row in enumerate(self.episodes()):
            for turn, episode_turn in enumerate(row["turns"]):
                expected = ByteCodec().encode(episode_turn["reply"])[1:]
                self.assertEqual(targets[index, turn, :len(expected)].tolist(), expected)
                self.assertTrue(targets[index, turn, len(expected):].eq(ByteCodec.PAD).all())

    def test_padding_lengths_and_eos_positions_handle_unicode_and_unequal_texts(self):
        rows = [["a", "é🙂"], ["longer words", ""]]
        packed = pack_observations(rows)
        self.assertEqual(packed["lengths"].tolist(), [11, 16])
        self.assertEqual(packed["eos_positions"].tolist(), [[2, 10], [13, 15]])
        self.assertEqual(packed["valid_mask"].sum(1).tolist(), [11, 16])
        self.assertTrue(packed["token_ids"][0, 11:].eq(ByteCodec.PAD).all())
        self.assertFalse(packed["valid_mask"][0, 11:].any())
        self.assertEqual(packed["token_ids"].dtype, torch.long)
        self.assertEqual(packed["valid_mask"].dtype, torch.bool)
        self.assertEqual(packed["token_ids"].device.type, "cpu")

    def test_blank_english_preserves_labels_and_turn_boundaries(self):
        rows = self.episodes()
        original = pack_cognitive_episodes(rows)
        blank = pack_cognitive_episodes(rows, blank_text=True)
        self.assertEqual(blank["inputs"]["token_ids"].shape, (4, 12))
        self.assertEqual(blank["inputs"]["token_ids"][0].tolist(), [1, 2] * 6)
        self.assertEqual(blank["inputs"]["eos_positions"][0].tolist(), [1, 3, 5, 7, 9, 11])
        for key in ("action_targets", "reply_decoder_input_ids", "reply_targets", "reply_target_mask"):
            self.assertTrue(torch.equal(original["supervision"][key], blank["supervision"][key]))
        self.assertEqual(blank["supervision"]["observation_next_byte_targets"][0].tolist(), [2, 0] * 6)

    def test_evaluation_has_no_oracle_and_encoder_ignores_supervision_and_metadata(self):
        rows = self.episodes("audit")
        changed = copy.deepcopy(rows)
        for episode in changed:
            episode.update(family="ignored metadata", id="ignored identity", seed=-99)
            for turn in episode["turns"]:
                turn["target"] = (turn["target"] + 1) % 4
                turn["reply"] = REPLIES[turn["target"]]
        with patch("experiments.sequence_data.validate_cognitive", side_effect=AssertionError("validator entered inference")), \
             patch("experiments.cognitive_curriculum.cognitive_oracle", side_effect=AssertionError("oracle entered inference")), \
             patch("experiments.cognitive_curriculum.parse_cognitive_sentence", side_effect=AssertionError("parser entered inference")):
            first = pack_cognitive_episodes(rows)
            second = pack_cognitive_episodes(changed)
            arbitrary = pack_observations([["This input is outside every curriculum grammar."]])
        self.assert_inputs_equal(first["inputs"], second["inputs"])
        self.assertFalse(torch.equal(first["supervision"]["action_targets"], second["supervision"]["action_targets"]))
        self.assertEqual(arbitrary["eos_positions"].shape, (1, 1))

    def test_training_requires_canonical_train_provenance(self):
        for split in ("dev", "audit"):
            with self.assertRaisesRegex(ValueError, "train"):
                pack_cognitive_episodes(self.episodes(split), training=True)
        rows = self.episodes()
        rows[0]["turns"][0]["target"] = (rows[0]["turns"][0]["target"] + 1) % 4
        rows[0]["turns"][0]["reply"] = REPLIES[rows[0]["turns"][0]["target"]]
        with self.assertRaises(ValueError):
            pack_cognitive_episodes(rows, training=True)

    def test_single_turn_reset_rows_preserve_text_without_fake_episode_metadata(self):
        rows = self.episodes()
        single = [[turn["text"]] for episode in rows for turn in episode["turns"]]
        inputs = pack_observations(single)
        self.assertEqual(inputs["eos_positions"].shape, (24, 1))
        for index, text_row in enumerate(single):
            end = int(inputs["eos_positions"][index, 0])
            self.assertEqual(ByteCodec().decode(inputs["token_ids"][index, :end + 1].tolist()), text_row[0])

    def test_inputs_and_outputs_do_not_mutate_or_alias_caller_rows(self):
        rows = self.episodes()
        before = copy.deepcopy(rows)
        first, second = pack_cognitive_episodes(rows, training=True), pack_cognitive_episodes(rows, training=True)
        first["inputs"]["token_ids"].fill_(0)
        first["supervision"]["action_targets"].fill_(0)
        self.assertEqual(rows, before)
        self.assertTrue(second["inputs"]["token_ids"].ne(0).any())
        self.assertTrue(second["supervision"]["action_targets"].ne(0).any())

    def test_overflow_invalid_shapes_and_nonzero_channels_fail(self):
        for text in ("x" * 129, "é" * 65):
            for blank in (False, True):
                with self.assertRaises(ValueError):
                    pack_observations([[text]], blank_text=blank)
        with self.assertRaisesRegex(ValueError, "context"):
            pack_observations([["abcd", "efgh"]], max_context_tokens=11)
        for rows in ([], ["flat text"], [[]], [["a"], ["b", "c"]], [["a"] * 7], [[3]]):
            with self.assertRaises(ValueError):
                pack_observations(rows)
        for options in ({"blank_text": 1}, {"max_input_bytes": True}, {"max_context_tokens": 0}):
            with self.assertRaises(ValueError):
                pack_observations([["a"]], **options)
        episodes = self.episodes()
        episodes[0]["turns"][0]["observations"]["visual"][0][0] = 1
        with self.assertRaisesRegex(ValueError, "zero"):
            pack_cognitive_episodes(episodes)
        with self.assertRaises(ValueError):
            pack_cognitive_episodes(self.episodes(), max_reply_bytes=1)


if __name__ == "__main__":
    unittest.main()
