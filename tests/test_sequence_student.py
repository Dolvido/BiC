"""Causal observation-only sequence baseline boundaries, without training."""
import copy
import unittest

import torch
from torch.nn import functional as F

from brain_in_computer.language import ByteCodec
from experiments.sequence_student import SequenceConfig, build_sequence_student


class SequenceStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self):
        return build_sequence_student(37, config=SequenceConfig(width=16, layers=2, heads=2, feedforward=32))

    def inputs(self, rows, *, extra_padding=0, reply="Yes."):
        codec = ByteCodec()
        encoded, endings = [], []
        for turns in rows:
            sequence, eos = [], []
            for text in turns:
                sequence.extend(codec.encode(text))
                eos.append(len(sequence) - 1)
            encoded.append(sequence)
            endings.append(eos)
        width = max(map(len, encoded)) + extra_padding
        tokens = torch.zeros(len(rows), width, dtype=torch.long)
        for index, sequence in enumerate(encoded):
            tokens[index, :len(sequence)] = torch.tensor(sequence)
        prefix = torch.tensor(codec.encode(reply)[:-1], dtype=torch.long)
        return {"token_ids": tokens, "eos_positions": torch.tensor(endings),
                "decoder_input_ids": prefix[None, None, :].expand(len(rows), len(rows[0]), -1).clone(),
                "valid_mask": tokens.ne(ByteCodec.PAD),
                "lengths": torch.tensor([len(sequence) for sequence in encoded])}

    def test_default_count_tied_weights_and_reproducible_seed_preserve_rng(self):
        before = torch.random.get_rng_state().clone()
        first, second = build_sequence_student(37), build_sequence_student(37)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in first.parameters()), 753610)
        self.assertEqual(sum(first.parameter_counts().values()), 753610)
        self.assertIs(first.tokens.weight, first.observation_head.weight)
        self.assertAlmostEqual(float(first.tokens.weight[1:].detach().std()), .02, delta=.001)
        self.assertAlmostEqual(float(first.positions.weight.detach().std()), .02, delta=.001)
        self.assertEqual(float(first.tokens.weight[ByteCodec.PAD].detach().abs().sum()), 0.)
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]))

    def test_shapes_and_observation_only_action_gradient_path(self):
        model = self.model()
        batch = self.inputs([["A.", "B.", "C.", "D.", "E.", "F."],
                             ["One.", "Two.", "Three.", "Four.", "Five.", "Six."]])
        output = model(**batch)
        self.assertEqual(output["logits"].shape, (2, 6, 4))
        self.assertEqual(output["language_logits"].shape, (2, 6, 5, 259))
        self.assertEqual(output["production_context"].shape, (2, 6, 128))
        self.assertEqual(output["observation_language_logits"].shape, (*batch["token_ids"].shape, 259))
        F.cross_entropy(output["logits"].flatten(0, 1), torch.tensor([0, 1, 2, 3, 0, 1] * 2)).backward()
        self.assertGreater(float(model.tokens.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.blocks[0].self_attn.in_proj_weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.blocks[-1].linear2.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in model.inferior_frontal.parameters()))

    def test_future_utterances_cannot_change_earlier_decisions_or_token_states(self):
        model = self.model()
        left = self.inputs([["A then B.", "What follows A?", "C is here."]])
        right = self.inputs([["A then B.", "What follows A?", "D is gone."]])
        a, b = model(**left), model(**right)
        torch.testing.assert_close(a["logits"][:, :2], b["logits"][:, :2], rtol=0, atol=1e-6)
        end = int(left["eos_positions"][0, 1]) + 1
        torch.testing.assert_close(a["context_states"][:, :end], b["context_states"][:, :end], rtol=0, atol=1e-6)
        torch.testing.assert_close(a["language_logits"][:, :2], b["language_logits"][:, :2], rtol=0, atol=1e-6)

    def test_full_context_matches_prefix_decisions_and_next_byte_states(self):
        model = self.model()
        turns = ["A then B.", "B then C.", "What follows A?", "C then D."]
        full = model(**self.inputs([turns]))
        for length in range(1, len(turns)):
            inputs = self.inputs([turns[:length]])
            prefix = model(**inputs)
            torch.testing.assert_close(full["logits"][:, :length], prefix["logits"], rtol=0, atol=1e-6)
            torch.testing.assert_close(full["context_states"][:, :inputs["token_ids"].shape[1]],
                                       prefix["context_states"], rtol=0, atol=1e-6)

    def test_padding_blank_text_and_other_sessions_are_isolated(self):
        model = self.model()
        normal = self.inputs([["A.", "What?"], ["X.", "Why??"]])
        padded = self.inputs([["A.", "What?"], ["X.", "Why??"]], extra_padding=9)
        changed = self.inputs([["A.", "What?"], ["Y.", "Where"]])
        first, second, third = model(**normal), model(**padded), model(**changed)
        torch.testing.assert_close(first["logits"], second["logits"], rtol=0, atol=1e-6)
        torch.testing.assert_close(first["logits"][0], third["logits"][0], rtol=0, atol=1e-6)
        self.assertEqual(float(second["context_states"][~padded["valid_mask"]].detach().abs().sum()), 0.)
        blank = model(**self.inputs([["", ""], ["", ""]]))
        self.assertTrue(torch.isfinite(blank["logits"]).all())

    def test_decoder_prefix_changes_cannot_change_actions_or_observation_context(self):
        model = self.model()
        a = model(**self.inputs([["A then B.", "What follows A?"]], reply="Yes."))
        b = model(**self.inputs([["A then B.", "What follows A?"]], reply="No."))
        for name in ("logits", "production_context", "context_states", "observation_language_logits"):
            self.assertTrue(torch.equal(a[name], b[name]))
        bos_only = self.inputs([["A then B.", "What follows A?"]])
        bos_only["decoder_input_ids"] = torch.ones(1, 2, 1, dtype=torch.long)
        self.assertEqual(model(**bos_only)["language_logits"].shape, (1, 2, 1, 259))

    def test_inference_attention_fast_path_preserves_prefix_and_future_isolation(self):
        model = self.model().eval()
        with torch.inference_mode():
            full = model(**self.inputs([["A then B.", "B then C.", "C then D."]]))
            prefix = model(**self.inputs([["A then B.", "B then C."]], extra_padding=7))
            changed = model(**self.inputs([["A then B.", "B then C.", "D then C."]]))
        torch.testing.assert_close(full["logits"][:, :2], prefix["logits"], rtol=0, atol=1e-6)
        torch.testing.assert_close(full["logits"][:, :2], changed["logits"][:, :2], rtol=0, atol=1e-6)

    def test_validation_rejects_padding_boundaries_and_inconsistent_metadata(self):
        model = self.model()
        original = self.inputs([["A.", "B."]], extra_padding=2)
        mutations = (
            lambda batch: batch["token_ids"].__setitem__((0, 2), ByteCodec.PAD),
            lambda batch: batch["eos_positions"].__setitem__((0, 0), 0),
            lambda batch: batch["valid_mask"].__setitem__((0, 0), False),
            lambda batch: batch["lengths"].__setitem__(0, 2),
            lambda batch: batch["decoder_input_ids"].__setitem__((0, 0, 0), ByteCodec.PAD),
        )
        for mutate in mutations:
            bad = copy.deepcopy(original)
            mutate(bad)
            with self.assertRaises(ValueError):
                model(**bad)
        for options in ({"width": True}, {"width": 17, "heads": 4}, {"layers": 0}):
            with self.assertRaises(ValueError):
                SequenceConfig(**options)


if __name__ == "__main__":
    unittest.main()
