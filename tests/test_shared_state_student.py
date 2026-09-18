"""Tiny CPU state-decoder checks with synthetic text and explicit actual work."""
import json
import unittest

import torch
from torch.nn import functional as F

from brain_in_computer.language import ByteCodec
from experiments.cognitive_curriculum import ALIASES, COLORS, REPLIES
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.sequence_training import sequence_objective
from experiments.shared_state_student import STATE_LABEL_SCHEMA, build_shared_state_student


class SharedStateStudentTests(unittest.TestCase):
    accounting = dict(model_constructions=0, native_forward_calls=0, native_forward_rows=0,
                      native_forward_turns=0, state_decoder_calls=0, state_query_vectors=0,
                      rejected_state_calls=0, backward_calls=0, optimizer_steps=0,
                      generated_curriculum_rows=0)

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config = SequenceConfig(width=16, layers=2, heads=2, feedforward=32)
        before = torch.random.get_rng_state().clone()
        cls.baseline = build_sequence_student(852310001, config=cls.config)
        cls.accounting["model_constructions"] += 1
        cls.model = build_shared_state_student(852310001, config=cls.config)
        cls.accounting["model_constructions"] += 1
        cls.replica = build_shared_state_student(852310001, config=cls.config)
        cls.accounting["model_constructions"] += 1
        cls.rng_preserved = torch.equal(before, torch.random.get_rng_state())

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        print("SHARED_STATE_CPU_ACCOUNTING=" + json.dumps(cls.accounting, sort_keys=True))

    def inputs(self, rows, *, padding=0):
        codec, sequences, endings = ByteCodec(), [], []
        for turns in rows:
            sequence, eos = [], []
            for text in turns:
                sequence.extend(codec.encode(text))
                eos.append(len(sequence) - 1)
            sequences.append(sequence); endings.append(eos)
        tokens = torch.zeros(len(rows), max(map(len, sequences)) + padding, dtype=torch.long)
        for index, sequence in enumerate(sequences):
            tokens[index, :len(sequence)] = torch.tensor(sequence)
        return dict(token_ids=tokens, eos_positions=torch.tensor(endings),
                    decoder_input_ids=torch.full((len(rows), len(rows[0]), 1), ByteCodec.BOS, dtype=torch.long),
                    valid_mask=tokens.ne(ByteCodec.PAD), lengths=torch.tensor(list(map(len, sequences))))

    def native(self, model, inputs):
        output = model(**inputs)
        self.accounting["native_forward_calls"] += 1
        self.accounting["native_forward_rows"] += inputs["token_ids"].shape[0]
        self.accounting["native_forward_turns"] += inputs["eos_positions"].numel()
        return output

    def states(self, output, positions, **kwargs):
        try:
            result = self.model.state_logits(output["context_states"], positions, **kwargs)
        except ValueError:
            self.accounting["rejected_state_calls"] += 1
            raise
        self.accounting["state_decoder_calls"] += 1
        self.accounting["state_query_vectors"] += result.numel() // result.shape[-1]
        return result

    def test_auxiliary_gradient_reaches_encoder_without_policy_readouts(self):
        self.model.train(); self.model.zero_grad(set_to_none=True)
        inputs = self.inputs([["dax is red.", "Is dax red?"], ["wug is blue.", "Is wug red?"]])
        output = self.native(self.model, inputs)
        logits = self.states(output, inputs["eos_positions"])
        targets = torch.zeros(2, 2, len(ALIASES), dtype=torch.long)
        targets[0, :, ALIASES.index("dax")] = 1
        targets[1, :, ALIASES.index("wug")] = 3
        self.accounting["backward_calls"] += 1
        F.cross_entropy(logits.flatten(0, 2), targets.flatten()).backward()
        required = [self.model.tokens.weight, self.model.norm.weight,
                    self.model.state_alias_embedding.weight, self.model.state_decoder[0].weight,
                    self.model.state_decoder[2].weight]
        required.extend(block.self_attn.in_proj_weight for block in self.model.blocks)
        for parameter in required:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        for module in (self.model.action_head, self.model.reply_context, self.model.inferior_frontal):
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
        self.assertIsNone(self.model.observation_head.bias.grad)

    def test_factory_rng_base_weights_and_native_forward_are_exact(self):
        self.assertTrue(self.rng_preserved)
        for name, value in self.baseline.state_dict().items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[name]), name)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, self.replica.state_dict()[name]), name)
        self.assertEqual(sum(self.model.parameter_counts().values()), sum(p.numel() for p in self.model.parameters()))
        self.assertEqual(self.model.state_alias_embedding.weight.shape, (12, 32))
        self.assertEqual(STATE_LABEL_SCHEMA["colors"], dict(zip(COLORS, range(1, 5))))
        self.assertEqual((STATE_LABEL_SCHEMA["count_offset"], STATE_LABEL_SCHEMA["switch_off"],
                          STATE_LABEL_SCHEMA["switch_on"]), (5, 105, 106))
        calls = []
        handle = self.model.state_decoder.register_forward_hook(lambda *args: calls.append(True))
        inputs = self.inputs([["dax is red.", "Is dax red?"], ["wug is blue.", "Is wug red?"]])
        try:
            for training in (True, False):
                self.baseline.train(training); self.model.train(training)
                with torch.inference_mode(not training):
                    baseline = self.native(self.baseline, inputs)
                    actual = self.native(self.model, inputs)
                self.assertEqual(set(baseline), set(actual))
                for name in baseline:
                    self.assertTrue(torch.equal(baseline[name], actual[name]), name)
        finally:
            handle.remove()
        self.assertEqual(calls, [])

    def test_future_prefix_and_padding_do_not_change_earlier_state_queries(self):
        self.model.eval()
        rows = [["dax is red.", "Is dax red?", "dax is blue."],
                ["wug is green.", "Is wug green?", "wug is red."]]
        variants = [rows, [row[:2] + ["Something else."] for row in rows], [row[:2] for row in rows], rows]
        scores = []
        with torch.inference_mode():
            for index, variant in enumerate(variants):
                inputs = self.inputs(variant, padding=7 if index == 3 else 0)
                scores.append(self.states(self.native(self.model, inputs), inputs["eos_positions"]))
        for other in scores[1:]:
            torch.testing.assert_close(scores[0][:, :2], other[:, :2], rtol=0, atol=2e-6)
        torch.testing.assert_close(scores[0], scores[3], rtol=0, atol=2e-6)

    def test_invalid_state_queries_are_rejected(self):
        states = {"context_states": torch.zeros(1, 6, self.config.width)}
        for positions in (torch.tensor([1, 3]), torch.tensor([[1., 3.]]),
                          torch.tensor([[1, 6]]), torch.tensor([[3, 1]])):
            with self.assertRaises(ValueError):
                self.states(states, positions)
        with self.assertRaises(ValueError):
            self.states(states, torch.tensor([[1, 3]]), alias_indices=torch.tensor([12]))

    def test_shared_alias_queries_are_permutation_equivariant(self):
        self.model.eval()
        inputs = self.inputs([["dax is red.", "Is dax red?"], ["wug is blue.", "Is wug red?"]])
        permutation = torch.arange(len(ALIASES) - 1, -1, -1)
        subset = torch.tensor([3, 3, 0])
        with torch.inference_mode():
            output = self.native(self.model, inputs)
            all_names = self.states(output, inputs["eos_positions"])
            permuted = self.states(output, inputs["eos_positions"], alias_indices=permutation)
            repeated = self.states(output, inputs["eos_positions"], alias_indices=subset)
        self.assertEqual(all_names.shape, (2, 2, 12, 107))
        torch.testing.assert_close(permuted, all_names[:, :, permutation], rtol=0, atol=2e-6)
        torch.testing.assert_close(repeated, all_names[:, :, subset], rtol=0, atol=2e-6)

    def test_zero_auxiliary_weight_preserves_base_gradients_and_first_step(self):
        # Zero weight means the auxiliary branch is unused, as at inference.
        # The model does not own a loss-weight policy or construct that branch.
        inputs = self.inputs([["dax is red.", "Is dax red?"], ["wug is blue.", "Is wug red?"]])
        labels = torch.tensor([[3, 1], [3, 0]])
        replies = [[ByteCodec().encode(REPLIES[int(label)]) for label in row] for row in labels]
        width = max(len(reply) - 1 for row in replies for reply in row)
        decoder = torch.zeros(2, 2, width, dtype=torch.long)
        targets = torch.zeros_like(decoder)
        for b, row in enumerate(replies):
            for t, reply in enumerate(row):
                decoder[b, t, :len(reply) - 1] = torch.tensor(reply[:-1])
                targets[b, t, :len(reply) - 1] = torch.tensor(reply[1:])
        inputs["decoder_input_ids"] = decoder
        observed = torch.zeros_like(inputs["token_ids"])
        observed[:, :-1] = inputs["token_ids"][:, 1:]
        batch = dict(inputs=inputs, supervision=dict(action_targets=labels,
                     reply_targets=targets, observation_next_byte_targets=observed))
        optimizers = []
        for model in (self.baseline, self.model):
            model.train(); model.zero_grad(set_to_none=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
            optimizers.append(optimizer)
            output = self.native(model, inputs)
            self.accounting["backward_calls"] += 1
            sequence_objective(output, batch)["loss"].backward()
        for name, parameter in self.baseline.named_parameters():
            other = dict(self.model.named_parameters())[name]
            self.assertEqual(parameter.grad is None, other.grad is None, name)
            if parameter.grad is not None:
                self.assertTrue(torch.equal(parameter.grad, other.grad), name)
        self.assertTrue(all(p.grad is None for p in self.model.state_alias_embedding.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.state_decoder.parameters()))
        for model, optimizer in zip((self.baseline, self.model), optimizers):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); self.accounting["optimizer_steps"] += 1
        for name, value in self.baseline.state_dict().items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
