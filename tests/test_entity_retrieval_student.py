"""Small synthetic CPU checks; no curriculum generation or optimizer work."""
import copy
import json
import unittest

import torch
from torch.nn import functional as F

from brain_in_computer.language import ByteCodec
from experiments.entity_retrieval_student import build_entity_retrieval_student
from experiments.sequence_student import SequenceConfig
from experiments.shared_state_student import build_shared_state_student


class EntityRetrievalStudentTests(unittest.TestCase):
    accounting = dict(model_constructions=0, forward_attempts=0, completed_forwards=0,
        rejected_forwards=0, forward_rows=0, forward_turns=0, byte_key_projections=0,
        byte_value_projections=0, state_decoder_calls=0, state_query_vectors=0,
        native_fusions=0, backward_calls=0, rejected_helper_calls=0,
        optimizer_steps=0, generated_curriculum_rows=0)

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config = SequenceConfig(width=16, layers=2, heads=4, feedforward=32)
        before = torch.random.get_rng_state().clone()
        cls.baseline = build_shared_state_student(852510001, config=cls.config)
        cls.model = build_entity_retrieval_student(852510001, config=cls.config)
        cls.replica = build_entity_retrieval_student(852510001, config=cls.config)
        cls.accounting["model_constructions"] += 3
        cls.rng_preserved = torch.equal(before, torch.random.get_rng_state())
        cls.hooks = []
        def count(name):
            def hook(module, args, output):
                cls.accounting[name] += 1
            return hook
        def states(module, args, output):
            cls.accounting["state_decoder_calls"] += 1
            cls.accounting["state_query_vectors"] += output.numel() // output.shape[-1]
        for model in (cls.baseline, cls.model, cls.replica):
            cls.hooks.append(model.state_decoder.register_forward_hook(states))
        for model in (cls.model, cls.replica):
            for module, name in ((model.entity_key, "byte_key_projections"),
                    (model.entity_value, "byte_value_projections"),
                    (model.native_entity_norm, "native_fusions")):
                cls.hooks.append(module.register_forward_hook(count(name)))

    @classmethod
    def tearDownClass(cls):
        for hook in cls.hooks:
            hook.remove()
        torch.set_num_threads(cls.threads)
        print("ENTITY_RETRIEVAL_CPU_ACCOUNTING=" + json.dumps(cls.accounting, sort_keys=True))

    def inputs(self, rows, *, padding=0, reply="Yes."):
        codec, sequences, endings = ByteCodec(), [], []
        for turns in rows:
            sequence, eos = [], []
            for text in turns:
                sequence.extend(codec.encode(text)); eos.append(len(sequence) - 1)
            sequences.append(sequence); endings.append(eos)
        tokens = torch.zeros(len(rows), max(map(len, sequences)) + padding, dtype=torch.long)
        for index, sequence in enumerate(sequences):
            tokens[index, :len(sequence)] = torch.tensor(sequence)
        decoder = torch.tensor(codec.encode(reply)[:-1])[None, None, :].expand(len(rows), len(rows[0]), -1).clone()
        return dict(token_ids=tokens, eos_positions=torch.tensor(endings), decoder_input_ids=decoder,
                    valid_mask=tokens.ne(ByteCodec.PAD), lengths=torch.tensor(list(map(len, sequences))))

    def native(self, model, inputs):
        self.accounting["forward_attempts"] += 1
        try:
            result = model(**inputs)
        except ValueError:
            self.accounting["rejected_forwards"] += 1
            raise
        self.accounting["completed_forwards"] += 1
        self.accounting["forward_rows"] += inputs["token_ids"].shape[0]
        self.accounting["forward_turns"] += inputs["eos_positions"].numel()
        return result

    def assert_gradient(self, parameters):
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_auxiliary_and_native_gradients_share_encoder_and_retriever(self):
        self.model.train()
        inputs = self.inputs([["dax is red.", "Is dax red?"], ["wug is blue.", "Is wug red?"]])
        for auxiliary in (True, False):
            self.model.zero_grad(set_to_none=True)
            output = self.native(self.model, inputs)
            if auxiliary:
                logits = self.model.state_logits_from_output(output, inputs["eos_positions"])
                targets = torch.zeros(2, 2, 12, dtype=torch.long)
                targets[:, :, 0] = 1
                loss = F.cross_entropy(logits.flatten(0, 2), targets.flatten())
            else:
                loss = output["logits"].square().mean() + output["language_logits"].square().mean()
            self.accounting["backward_calls"] += 1
            loss.backward()
            self.assert_gradient([self.model.tokens.weight, self.model.blocks[0].self_attn.in_proj_weight,
                self.model.blocks[-1].self_attn.in_proj_weight, self.model.norm.weight,
                self.model.state_alias_embedding.weight, self.model.entity_alias_query.weight,
                self.model.entity_eos_query.weight, self.model.entity_key.weight,
                self.model.entity_value.weight, self.model.entity_readout.weight])
            if auxiliary:
                self.assert_gradient([self.model.state_decoder[0].weight, self.model.state_decoder[2].weight])
                for module in (self.model.action_head, self.model.reply_context, self.model.inferior_frontal,
                               self.model.native_entity_query, self.model.native_entity_readout):
                    self.assertTrue(all(p.grad is None for p in module.parameters()))
            else:
                self.assert_gradient([self.model.native_entity_query.weight, self.model.native_entity_key.weight,
                    self.model.native_entity_value.weight, self.model.native_entity_readout.weight,
                    self.model.action_head.weight, self.model.reply_context[0].weight,
                    self.model.inferior_frontal.readout.weight])
                self.assertTrue(all(p.grad is None for p in self.model.state_decoder.parameters()))
            self.assertIsNone(self.model.observation_head.bias.grad)

    def test_cached_state_readout_reuses_entities_and_keys_values_once(self):
        self.model.eval()
        inputs = self.inputs([["A.", "B."], ["A longer turn.", "Other."]], padding=3)
        counts = dict(self.accounting)
        with torch.inference_mode():
            output = self.native(self.model, inputs)
            cached = self.model.state_logits_from_output(output, inputs["eos_positions"])
            self.assertEqual(self.accounting["byte_key_projections"] - counts["byte_key_projections"], 1)
            self.assertEqual(self.accounting["byte_value_projections"] - counts["byte_value_projections"], 1)
            direct = self.model.state_logits(output["context_states"], inputs["eos_positions"])
        self.assertTrue(torch.equal(cached, direct))
        self.assertEqual(cached.shape, (2, 2, 12, 107))
        self.assertEqual(self.accounting["byte_key_projections"] - counts["byte_key_projections"], 2)
        self.assertEqual(self.accounting["byte_value_projections"] - counts["byte_value_projections"], 2)

    def test_factory_preserves_rng_common_weights_and_raw_observation_path(self):
        self.assertTrue(self.rng_preserved)
        for name, value in self.baseline.state_dict().items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[name]), name)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, self.replica.state_dict()[name]), name)
        for model in (self.baseline, self.model):
            self.assertEqual(sum(model.parameter_counts().values()), sum(p.numel() for p in model.parameters()))
        inputs = self.inputs([["dax is red.", "Is dax red?"], ["wug is blue.", "Is wug red?"]])
        state_calls = self.accounting["state_decoder_calls"]
        for training in (True, False):
            self.baseline.train(training); self.model.train(training)
            with torch.inference_mode(not training):
                baseline = self.native(self.baseline, inputs)
                actual = self.native(self.model, inputs)
            for name in ("context_states", "observation_language_logits"):
                self.assertTrue(torch.equal(baseline[name], actual[name]), name)
            self.assertEqual(actual["entity_states"].shape, (2, 2, 12, 16))
        self.assertEqual(self.accounting["state_decoder_calls"], state_calls)

    def test_future_padding_other_rows_and_reply_prefix_are_isolated(self):
        self.model.eval()
        rows = [["dax is red.", "Is dax red?", "dax is blue."],
                ["wug is green.", "Is wug green?", "wug is red."]]
        variants = [rows, [row[:2] + ["A changed future statement."] for row in rows],
                    [row[:2] for row in rows], rows,
                    [rows[0], ["Different.", "Second changed.", "Third changed."]], rows]
        outputs = []
        with torch.inference_mode():
            for index, variant in enumerate(variants):
                inputs = self.inputs(variant, padding=7 if index == 3 else 0, reply="No." if index == 5 else "Yes.")
                outputs.append(self.native(self.model, inputs))
        for index, other in enumerate(outputs[1:], start=1):
            batch = slice(0, 1) if index == 4 else slice(None)
            for name in ("logits", "production_context", "entity_states"):
                torch.testing.assert_close(outputs[0][name][batch, :2], other[name][batch, :2], rtol=0, atol=3e-6)
            if index != 5:
                torch.testing.assert_close(outputs[0]["language_logits"][batch, :2],
                                           other["language_logits"][batch, :2], rtol=0, atol=3e-6)
        # Observation padding stays exactly zero despite learned readout biases.
        inputs = self.inputs(rows, padding=7)
        self.assertTrue(outputs[3]["context_states"][~inputs["valid_mask"]].eq(0).all())

    def test_invalid_inputs_and_cached_positions_fail_closed(self):
        self.model.eval()
        original = self.inputs([["A.", "B."]], padding=2)
        mutations = (lambda row: row["eos_positions"].__setitem__((0, 0), 0),
                     lambda row: row["valid_mask"].__setitem__((0, 0), False),
                     lambda row: row["lengths"].__setitem__(0, 2),
                     lambda row: row["decoder_input_ids"].__setitem__((0, 0, 0), ByteCodec.PAD))
        before = self.accounting["byte_key_projections"]
        for mutate in mutations:
            invalid = copy.deepcopy(original); mutate(invalid)
            with self.assertRaises(ValueError):
                self.native(self.model, invalid)
        self.assertEqual(self.accounting["byte_key_projections"], before)
        with torch.inference_mode():
            output = self.native(self.model, original)
        changed = original["eos_positions"].clone(); changed[0, 0] += 1
        cases = [lambda: self.model.state_logits_from_output(output, changed),
                 lambda: self.model.state_logits_from_output({"context_states": output["context_states"]}, original["eos_positions"]),
                 lambda: self.model.retrieve_entities(output["context_states"], original["eos_positions"], alias_indices=torch.tensor([0])),
                 lambda: self.model.retrieve_entities(output["context_states"], original["eos_positions"], alias_indices=torch.zeros(12, dtype=torch.long))]
        before = self.accounting["byte_key_projections"]
        for call in cases:
            with self.assertRaises(ValueError):
                call()
            self.accounting["rejected_helper_calls"] += 1
        self.assertEqual(self.accounting["byte_key_projections"], before)

    def test_slot_permutation_and_direct_prefix_mask(self):
        self.model.eval()
        inputs = self.inputs([["A.", "The end."], ["Some words.", "B."]], padding=4)
        permutation = torch.arange(11, -1, -1)
        with torch.inference_mode():
            output = self.native(self.model, inputs)
            original = output["entity_states"]
            permuted = self.model.retrieve_entities(output["context_states"], inputs["eos_positions"], alias_indices=permutation)
            torch.testing.assert_close(permuted, original[:, :, permutation], rtol=0, atol=2e-6)
            logits = self.model.state_logits_from_output(output, inputs["eos_positions"])
            permuted_logits = self.model._decode_entities(permuted, alias_indices=permutation)
            torch.testing.assert_close(permuted_logits, logits[:, :, permutation], rtol=0, atol=2e-6)
            turns = self.model._turn_states(output["context_states"], inputs["eos_positions"])
            torch.testing.assert_close(self.model._fuse_entities(turns, original),
                                       self.model._fuse_entities(turns, permuted), rtol=0, atol=2e-6)
            altered = output["context_states"].clone()
            for row, eos in enumerate(inputs["eos_positions"][:, 0]):
                altered[row, int(eos) + 1:] = 20.
            changed = self.model.retrieve_entities(altered, inputs["eos_positions"])
            torch.testing.assert_close(changed[:, 0], original[:, 0], rtol=0, atol=2e-6)
            padded = output["context_states"].clone(); padded[~inputs["valid_mask"]] = -30.
            changed_padding = self.model.retrieve_entities(padded, inputs["eos_positions"])
            torch.testing.assert_close(changed_padding, original, rtol=0, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
