"""Random-initialization wiring checks; no language training or weight updates."""

import unittest
from dataclasses import fields

import torch
from torch.nn import functional as F

from brain_in_computer.language import ByteCodec, LanguageBrain, LanguageConfig
from brain_in_computer.model import Brain, BrainConfig
from brain_in_computer.tasks import TaskStream


class ByteCodecTests(unittest.TestCase):
    def test_utf8_roundtrip_empty_unicode_and_right_padding(self):
        codec = ByteCodec()
        texts = ["", "Hello.", "caf\u00e9 \U0001f600", "\x00\n"]
        ids, lengths = codec.batch_encode(texts)
        self.assertEqual(ids.dtype, torch.long)
        for row, text, length in zip(ids, texts, lengths.tolist()):
            self.assertEqual(codec.decode(row), text)
            self.assertEqual(length, len(text.encode("utf-8")) + 2)
            self.assertTrue((row[length:] == ByteCodec.PAD).all())
        self.assertEqual(codec.encode(""), [ByteCodec.BOS, ByteCodec.EOS])
        self.assertEqual(codec.encode("\x00"), [ByteCodec.BOS, 3, ByteCodec.EOS])
        self.assertEqual(ByteCodec.VOCAB_SIZE, 259)

    def test_explicit_byte_limits_do_not_silently_truncate(self):
        codec = ByteCodec(max_bytes=3)
        self.assertEqual(codec.decode(codec.encode("abc")), "abc")
        for text in ("abcd", "\U0001f600"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "No truncation"):
                codec.encode(text)
        self.assertEqual(codec.decode(codec.encode("\U0001f600", max_bytes=4)), "\U0001f600")
        for invalid in (0, -1, True, 2.5):
            with self.subTest(limit=invalid), self.assertRaises(ValueError):
                ByteCodec(invalid)
        for invalid in ([], "plain string", [None]):
            with self.subTest(texts=invalid), self.assertRaises(ValueError):
                codec.batch_encode(invalid)

    def test_invalid_generated_utf8_requires_explicit_replacement(self):
        codec = ByteCodec()
        tokens = [ByteCodec.BOS, 255 + ByteCodec.BYTE_OFFSET, ByteCodec.EOS]
        with self.assertRaises(UnicodeDecodeError):
            codec.decode(tokens)
        self.assertEqual(codec.decode(tokens, errors="replace"), "\ufffd")
        for invalid in ([-1], [259], [True], [1.0]):
            with self.subTest(ids=invalid), self.assertRaises(ValueError):
                codec.decode(invalid)
        with self.assertRaises(ValueError):
            codec.decode([3], errors="ignore")


class LanguageBrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(170)
        self.brain = Brain(BrainConfig(hidden_size=12, memory_slots=3))
        self.config = LanguageConfig(hidden_size=16, embedding_size=8, max_input_bytes=24, max_output_bytes=16)
        self.model = LanguageBrain(self.brain, self.config)
        self.observations = TaskStream(971).sample(2, delay=2).observations
        self.ids, self.lengths = self.model.prepare_inputs(["ab", "cdef"])
        self.decoder = torch.tensor([[1, 100, 101, 102], [1, 110, 111, 112]], dtype=torch.long)

    def call(self, *, ids=None, lengths=None, decoder=None, observations=None, ablate=()):
        return self.model(
            self.observations if observations is None else observations,
            self.ids if ids is None else ids,
            self.lengths if lengths is None else lengths,
            self.decoder if decoder is None else decoder,
            ablate=ablate,
        )

    def test_output_shapes_and_separate_network_parameters(self):
        output = self.call()
        self.assertEqual(tuple(output["language_logits"].shape), (2, 4, ByteCodec.VOCAB_SIZE))
        self.assertEqual(tuple(output["concept_context"].shape), (2, 12))
        self.assertEqual(tuple(output["production_context"].shape), (2, 16))
        self.assertEqual(tuple(output["comprehension_states"].shape), (2, 6, 16))
        self.assertEqual(tuple(output["logits"].shape), (2, 4, 4))
        counts = self.model.parameter_counts()
        self.assertEqual(set(counts), {"brain", "posterior_temporal", "inferior_frontal", "semantic_bridge"})
        self.assertTrue(all(count > 0 for count in counts.values()))
        self.assertEqual(sum(counts.values()), sum(p.numel() for p in self.model.parameters()))
        pointers = []
        for module in (self.brain, self.model.posterior_temporal, self.model.inferior_frontal, self.model.semantic_bridge):
            pointers.extend(p.untyped_storage().data_ptr() for p in module.parameters())
        self.assertEqual(len(pointers), len(set(pointers)))

    def test_gradient_connectivity_without_any_parameter_update(self):
        before = {name: tensor.clone() for name, tensor in self.model.state_dict().items()}
        output = self.call()
        # Synthetic byte IDs exercise backward connectivity only. There is no
        # optimizer, learning step, trained checkpoint, or competence metric.
        targets = torch.tensor([[107, 108, 109, 2], [120, 121, 122, 2]])
        loss = F.cross_entropy(output["language_logits"].flatten(0, 1), targets.flatten())
        loss.backward()
        for name, module in (
            ("comprehension", self.model.posterior_temporal),
            ("production", self.model.inferior_frontal),
            ("bridge", self.model.semantic_bridge),
            ("temporal core", self.brain.regions["temporal_language"]),
            ("prefrontal core", self.brain.regions["prefrontal_cortex"]),
            ("hippocampal core", self.brain.regions["hippocampus"]),
        ):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients), name)
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0, name)
        for name, tensor in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], tensor), name)

    def test_decoder_is_autoregressive_and_response_cannot_change_brain_episode(self):
        altered = self.decoder.clone()
        altered[:, 2:] += 40
        with torch.no_grad():
            original, changed = self.call(), self.call(decoder=altered)
        self.assertTrue(torch.equal(original["language_logits"][:, :2], changed["language_logits"][:, :2]))
        self.assertFalse(torch.equal(original["language_logits"][:, 2:], changed["language_logits"][:, 2:]))
        self.assertTrue(torch.equal(original["logits"], changed["logits"]))
        for name in original["region_activity"]:
            self.assertTrue(torch.equal(original["region_activity"][name], changed["region_activity"][name]), name)

    def test_comprehension_states_are_causal_and_padding_is_excluded(self):
        first_ids, first_lengths = self.model.prepare_inputs(["abcx", "abcx"])
        second_ids, second_lengths = self.model.prepare_inputs(["abcy", "abcz"])
        with torch.no_grad():
            first = self.call(ids=first_ids, lengths=first_lengths)
            second = self.call(ids=second_ids, lengths=second_lengths)
            extended = self.call(ids=F.pad(self.ids, (0, 4), value=ByteCodec.PAD))
            ordinary = self.call()
        self.assertTrue(torch.equal(first["comprehension_states"][:, :4], second["comprehension_states"][:, :4]))
        self.assertFalse(torch.equal(first["concept_context"], second["concept_context"]))
        for key in ("concept_context", "production_context", "language_logits", "logits"):
            self.assertTrue(torch.equal(ordinary[key], extended[key]), key)
        self.assertEqual(int(torch.count_nonzero(extended["comprehension_states"][:, 6:])), 0)
        self.assertEqual(int(torch.count_nonzero(ordinary["comprehension_states"][0, 4:])), 0)

    def test_text_enters_core_and_has_no_direct_production_bypass(self):
        changed_ids, changed_lengths = self.model.prepare_inputs(["wx", "yzab"])
        with torch.no_grad():
            ordinary = self.call()
            changed = self.call(ids=changed_ids, lengths=changed_lengths)
            lesion = self.call(ablate=("temporal_language",))
            changed_lesion = self.call(ids=changed_ids, lengths=changed_lengths, ablate=("temporal_language",))
        self.assertFalse(torch.equal(ordinary["logits"], changed["logits"]))
        self.assertFalse(torch.equal(ordinary["language_logits"], changed["language_logits"]))
        self.assertTrue(torch.equal(lesion["logits"], changed_lesion["logits"]))
        self.assertTrue(torch.equal(lesion["language_logits"], changed_lesion["language_logits"]))
        self.assertEqual(int(torch.count_nonzero(lesion["region_activity"]["temporal_language"])), 0)

    def test_production_depends_on_observations_and_no_future_observation_enters_past_core(self):
        observations = {name: tensor.clone() for name, tensor in self.observations.items()}
        observations["visual"][:, -1] += 2.0
        with torch.no_grad():
            original, altered = self.call(), self.call(observations=observations)
        self.assertTrue(torch.equal(original["logits"][:, :-1], altered["logits"][:, :-1]))
        self.assertFalse(torch.equal(original["production_context"], altered["production_context"]))
        self.assertFalse(torch.equal(original["language_logits"], altered["language_logits"]))

    def test_calls_reset_state_and_batch_members_are_independent(self):
        with torch.no_grad():
            original = self.call()
            self.call(decoder=self.decoder[:, :1])
            repeated = self.call()
            single = self.model({name: tensor[:1] for name, tensor in self.observations.items()}, self.ids[:1], self.lengths[:1], self.decoder[:1])
        self.assertTrue(torch.equal(original["language_logits"], repeated["language_logits"]))
        torch.testing.assert_close(original["language_logits"][:1], single["language_logits"])

    def test_config_validation_and_model_dtype(self):
        for field in fields(LanguageConfig):
            for invalid in (0, -1, True, 1.5):
                with self.subTest(field=field.name, value=invalid), self.assertRaises(ValueError):
                    LanguageConfig(**{field.name: invalid})
        model = LanguageBrain(Brain(BrainConfig(hidden_size=12, memory_slots=3)).double(), self.config)
        observations = {name: value if name == "tokens" else value.double() for name, value in self.observations.items()}
        output = model(observations, self.ids, self.lengths, self.decoder)
        self.assertEqual(output["language_logits"].dtype, torch.float64)

    def test_text_shape_lengths_vocabulary_and_right_padding_are_validated(self):
        invalid_pairs = [
            (self.ids.float(), self.lengths), (self.ids[:, :0], self.lengths),
            (self.ids[:1], self.lengths), (self.ids, self.lengths.float()),
            (self.ids, self.lengths[:, None]), (self.ids, torch.tensor([1, 6])),
            (self.ids, torch.tensor([7, 6])),
            (F.pad(self.ids, (0, 25)), self.lengths),
        ]
        for row, column, value in ((0, 0, 3), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 3, 4), (0, 5, 7), (0, 1, 259), (0, 1, -1)):
            ids = self.ids.clone()
            ids[row, column] = value
            invalid_pairs.append((ids, self.lengths))
        for index, (ids, lengths) in enumerate(invalid_pairs):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.call(ids=ids, lengths=lengths)

    def test_decoder_boundaries_and_padding_are_validated(self):
        invalid = [self.decoder.float(), self.decoder[:1], self.decoder[:, :0], F.pad(self.decoder, (0, 14))]
        for column, value in ((0, 3), (1, 1), (1, 0), (1, 2), (1, -1), (1, 259)):
            ids = self.decoder.clone()
            ids[0, column] = value
            invalid.append(ids)
        for index, decoder in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.call(decoder=decoder)
        # Shorter teacher-forcing rows can contain EOS followed only by padding.
        valid = torch.tensor([[1, 2, 0, 0], [1, 90, 91, 92]])
        self.assertTrue(torch.isfinite(self.call(decoder=valid)["language_logits"]).all())
        self.assertEqual(tuple(self.call(decoder=self.decoder[:, :1])["language_logits"].shape), (2, 1, 259))


if __name__ == "__main__":
    unittest.main()
