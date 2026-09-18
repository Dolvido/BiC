"""Small synthetic CPU boundaries; no curriculum generation or optimizer."""
import copy
import json
import unittest

import torch

from brain_in_computer.language import ByteCodec
from experiments.recurrent_read_student import RecurrentReadStudent, build_recurrent_read_student
from experiments.sequence_student import SequenceConfig, build_sequence_student


class RecurrentReadStudentTests(unittest.TestCase):
    accounting = {"model_constructions": 0, "forward_attempts": 0, "completed_forwards": 0,
                  "rejected_forwards": 0, "forward_rows": 0, "forward_turns": 0,
                  "forward_observation_bytes": 0, "gradient_enabled_forwards": 0,
                  "backward_calls": 0, "optimizer_updates": 0, "generated_curriculum_rows": 0}

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config = SequenceConfig(width=16, layers=2, heads=2, feedforward=32)
        before = torch.random.get_rng_state().clone()
        cls.baseline = build_sequence_student(852110001, config=cls.config)
        cls.accounting["model_constructions"] += 1
        cls.one = build_recurrent_read_student(852110001, config=cls.config, read_passes=1)
        cls.accounting["model_constructions"] += 1
        cls.four = build_recurrent_read_student(852110001, config=cls.config, read_passes=4)
        cls.accounting["model_constructions"] += 1
        cls.rng_preserved = torch.equal(before, torch.random.get_rng_state())

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        print("RECURRENT_READ_CPU_ACCOUNTING=" + json.dumps(cls.accounting, sort_keys=True))

    def inputs(self, rows, *, padding=0, reply="Yes."):
        codec, sequences, endings = ByteCodec(), [], []
        for turns in rows:
            sequence, eos = [], []
            for text in turns:
                sequence.extend(codec.encode(text))
                eos.append(len(sequence) - 1)
            sequences.append(sequence)
            endings.append(eos)
        tokens = torch.zeros(len(rows), max(map(len, sequences)) + padding, dtype=torch.long)
        for index, sequence in enumerate(sequences):
            tokens[index, :len(sequence)] = torch.tensor(sequence)
        prefix = torch.tensor([ByteCodec.BOS] if reply is None else codec.encode(reply)[:-1])
        return {"token_ids": tokens, "eos_positions": torch.tensor(endings),
                "decoder_input_ids": prefix[None, None, :].expand(len(rows), len(rows[0]), -1).clone(),
                "valid_mask": tokens.ne(ByteCodec.PAD), "lengths": torch.tensor(list(map(len, sequences)))}

    def run_model(self, model, inputs, *, grad=False):
        counts = type(self).accounting
        counts["forward_attempts"] += 1
        try:
            with torch.set_grad_enabled(grad):
                output = model(**inputs)
        except ValueError:
            counts["rejected_forwards"] += 1
            raise
        counts["completed_forwards"] += 1
        counts["forward_rows"] += inputs["token_ids"].shape[0]
        counts["forward_turns"] += inputs["eos_positions"].numel()
        counts["forward_observation_bytes"] += int(inputs["token_ids"].ge(3).sum())
        counts["gradient_enabled_forwards"] += int(grad)
        return output

    def close(self, left, right):
        torch.testing.assert_close(left, right, rtol=0, atol=2e-6)

    def test_one_pass_is_exact_baseline_and_parameters_are_identical(self):
        self.assertTrue(self.rng_preserved)
        for model in (self.one, self.four):
            self.assertEqual(self.baseline.parameter_counts(), model.parameter_counts())
            self.assertIs(model.tokens.weight, model.observation_head.weight)
            self.assertEqual(set(self.baseline.state_dict()), set(model.state_dict()))
            for key, value in self.baseline.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        inputs = self.inputs([["A is on.", "Is A on?"], ["B is off.", "Is B on?"]], padding=3)
        for training in (True, False):
            self.baseline.train(training)
            self.one.train(training)
            with torch.inference_mode(not training):
                baseline = self.run_model(self.baseline, inputs, grad=training)
                one = self.run_model(self.one, inputs, grad=training)
            self.assertEqual(set(baseline), set(one))
            for key in baseline:
                self.assertTrue(torch.equal(baseline[key], one[key]), key)

    def test_future_and_prefix_isolation_in_training_and_inference(self):
        rows = [["A is on.", "Is A on?", "A is off."], ["B is off.", "Is B on?", "B is on."]]
        changed = [row[:2] + ["Unseen."] for row in rows]
        for training in (True, False):
            self.four.train(training)
            with torch.inference_mode(not training):
                full = self.run_model(self.four, self.inputs(rows), grad=training)
                future = self.run_model(self.four, self.inputs(changed), grad=training)
                prefix = self.run_model(self.four, self.inputs([row[:2] for row in rows]), grad=training)
            for key in ("logits", "language_logits", "production_context"):
                self.close(full[key][:, :2], future[key][:, :2])
                self.close(full[key][:, :2], prefix[key])

    def test_padding_blank_turns_and_other_batch_rows_are_isolated(self):
        self.four.eval()
        rows = [["A.", "What?"], ["Longer input.", "Why?"]]
        with torch.inference_mode():
            ordinary = self.run_model(self.four, self.inputs(rows))
            padded_inputs = self.inputs(rows, padding=7)
            padded = self.run_model(self.four, padded_inputs)
            changed = self.run_model(self.four, self.inputs([rows[0], ["Other.", "Where?"]]))
            blank = self.run_model(self.four, self.inputs([["", ""], ["", ""]]))
        for key in ("logits", "language_logits", "production_context"):
            self.close(ordinary[key], padded[key])
            self.close(ordinary[key][0], changed[key][0])
        self.assertEqual(float(padded["context_states"][~padded_inputs["valid_mask"]].abs().sum()), 0.)
        self.assertTrue(all(torch.isfinite(value).all() for value in blank.values()))

    def test_bos_and_reply_prefix_do_not_enter_decision_or_memory(self):
        self.four.eval()
        rows = [["A is on.", "Is A on?"]]
        with torch.inference_mode():
            yes = self.run_model(self.four, self.inputs(rows, reply="Yes."))
            no = self.run_model(self.four, self.inputs(rows, reply="No."))
            bos = self.run_model(self.four, self.inputs(rows, reply=None))
        for key in ("logits", "production_context", "context_states", "observation_language_logits"):
            self.assertTrue(torch.equal(yes[key], no[key]), key)
            self.assertTrue(torch.equal(yes[key], bos[key]), key)
        self.close(yes["language_logits"][:, :, 0], bos["language_logits"][:, :, 0])
        self.close(no["language_logits"][:, :, 0], bos["language_logits"][:, :, 0])
        self.assertEqual(bos["language_logits"].shape, (1, 2, 1, ByteCodec.VOCAB_SIZE))

    def test_fixed_memory_shared_block_mask_and_gradient_path(self):
        self.four.train()
        self.four.zero_grad(set_to_none=True)
        inputs = self.inputs([["A.", "What?"], ["Long fact.", "Why?"]], padding=2)
        calls, pre_last, normalized = [], [], []
        last = self.four.blocks[-1]
        capture_memory = last.register_forward_pre_hook(lambda module, args: pre_last.append(args[0]))
        capture_attention = last.self_attn.register_forward_pre_hook(
            lambda module, args, kwargs: calls.append((args, kwargs)), with_kwargs=True)
        capture_normalized = self.four.norm.register_forward_hook(
            lambda module, args, output: normalized.append(output))
        try:
            output = self.run_model(self.four, inputs, grad=True)
        finally:
            capture_memory.remove()
            capture_attention.remove()
            capture_normalized.remove()
        self.assertEqual(len(pre_last), 1)
        self.assertEqual(len(calls), 4)
        expected_memory = calls[0][0][1]
        positions = torch.arange(inputs["token_ids"].shape[1])
        blocked = ((positions[None, None, :] > inputs["eos_positions"][:, :, None])
                   | ~inputs["valid_mask"][:, None, :])
        blocked = blocked[:, None].expand(-1, self.config.heads, -1, -1).reshape(
            len(inputs["token_ids"]) * self.config.heads, inputs["eos_positions"].shape[1], -1)
        for args, kwargs in calls[1:]:
            self.assertEqual(args[0].shape[:2], inputs["eos_positions"].shape)
            self.assertIs(args[1], args[2])
            self.assertIs(args[1], calls[1][0][1])
            self.assertTrue(torch.equal(args[1], expected_memory))
            self.assertTrue(args[0].requires_grad and args[1].requires_grad)
            self.assertTrue(torch.equal(kwargs["attn_mask"], blocked))
            self.assertFalse(kwargs["need_weights"])
        # The full-byte auxiliary stays exactly the ordinary encoder path.
        self.assertEqual(len(normalized), 2)
        self.assertTrue(torch.equal(output["context_states"],
                                    normalized[0].masked_fill(~inputs["valid_mask"][:, :, None], 0.)))
        self.accounting["backward_calls"] += 1
        (output["logits"].square().mean() + output["language_logits"].square().mean()).backward()
        for parameter in (self.four.tokens.weight, self.four.blocks[0].self_attn.in_proj_weight,
                          last.self_attn.in_proj_weight, last.linear2.weight,
                          self.four.action_head.weight, self.four.reply_context[0].weight,
                          self.four.inferior_frontal.readout.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_invalid_passes_and_inputs_fail_before_neural_work(self):
        for passes in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                RecurrentReadStudent(self.config, read_passes=passes)
        original = self.inputs([["A.", "B."]], padding=2)
        mutations = (lambda row: row["eos_positions"].__setitem__((0, 0), 0),
                     lambda row: row["valid_mask"].__setitem__((0, 0), False),
                     lambda row: row["lengths"].__setitem__(0, 2),
                     lambda row: row["decoder_input_ids"].__setitem__((0, 0, 0), ByteCodec.PAD))
        for mutate in mutations:
            invalid = copy.deepcopy(original)
            mutate(invalid)
            with self.assertRaises(ValueError):
                self.run_model(self.four, invalid)


if __name__ == "__main__":
    unittest.main()
