"""CPU-only boundary tests for the untrained regional sequence candidate."""
import copy
import io
import unittest

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments.cognitive_student import EpisodicState, build_cognitive_student
from experiments.regional_sequence_student import (
    RegionalSequenceState, build_regional_sequence_student,
)
from experiments.sequence_data import pack_observations
from experiments.sequence_student import SequenceConfig


class RegionalSequenceStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self, **overrides):
        config = dict(width=16, layers=1, heads=4, feedforward=32, max_positions=256)
        config.update(overrides)
        return build_regional_sequence_student(83, sequence_config=SequenceConfig(**config))

    def inputs(self, model, texts, replies=None):
        batch = len(texts)
        observations = {name: torch.zeros(batch, 1, width) for name, width in
                        (("visual", 32), ("auditory", 4), ("body", 4), ("feedback", 2))}
        observations["tokens"] = torch.zeros(batch, 1, dtype=torch.long)
        ids, lengths = model.prepare_inputs(texts)
        decoder, _ = model.codec.batch_encode(replies or ["Yes."] * batch)
        return observations, ids, lengths, decoder[:, :-1]

    def assert_state_equal(self, model, left, right):
        left_brain = model.brain.state_to_dict(left.brain_state)
        right_brain = model.brain.state_to_dict(right.brain_state)
        for name in left_brain:
            self.assertTrue(torch.equal(left_brain[name], right_brain[name]), name)
        self.assertEqual(len(left.text_history), len(right.text_history))
        for a, b in zip(left.text_history + left.text_lengths, right.text_history + right.text_lengths):
            self.assertTrue(torch.equal(a, b))

    def test_seed_base_initialization_parameter_count_and_encoder_only(self):
        rng = torch.random.get_rng_state().clone()
        model = build_regional_sequence_student(83)
        repeated = build_regional_sequence_student(83)
        original = build_cognitive_student(83, memory_mode="recurrent")
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(checkpoint_digest(model), checkpoint_digest(repeated))
        for name, value in original.state_dict().items():
            if not name.startswith("posterior_temporal."):
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        self.assertEqual(set(model.posterior_temporal._modules), {"tokens", "positions", "blocks", "norm"})
        self.assertEqual(sum(model.parameter_counts().values()), sum(p.numel() for p in model.parameters()))
        self.assertEqual(sum(model.parameter_counts().values()), 1_309_188)
        self.assertEqual(len(model.brain.region_names), 13)

    def test_decoder_prefixes_cannot_change_official_policy_or_history(self):
        model = self.model()
        first, left = model.forward_with_state(*self.inputs(model, ["A follows B."], ["Yes."]))
        second, right = model.forward_with_state(*self.inputs(model, ["A follows B."], ["No."]))
        for name in ("logits", "concept_context", "production_context", "observation_language_logits"):
            self.assertTrue(torch.equal(first[name], second[name]), name)
        self.assert_state_equal(model, left, right)
        self.assertFalse(torch.equal(first["language_logits"], second["language_logits"]))

    def test_causal_prefix_equivalence_and_future_byte_isolation(self):
        model = self.model().eval()
        short = pack_observations([["A before B."]])
        long = pack_observations([["A before B.", "B before C."]])
        altered = pack_observations([["A before B.", "D after E."]])
        with torch.inference_mode():
            prefix = model.posterior_temporal(short["token_ids"], short["eos_positions"])
            complete = model.posterior_temporal(long["token_ids"], long["eos_positions"])
            changed = model.posterior_temporal(altered["token_ids"], altered["eos_positions"])
            torch.testing.assert_close(prefix, complete[:, :prefix.shape[1]], rtol=1e-5, atol=2e-6)
            torch.testing.assert_close(prefix, changed[:, :prefix.shape[1]], rtol=1e-5, atol=2e-6)
            left, _ = model.forward_with_state(*self.inputs(model, ["A then B."]))
            right, _ = model.forward_with_state(*self.inputs(model, ["A then C."]))
        torch.testing.assert_close(left["observation_language_logits"][:, :8],
                                   right["observation_language_logits"][:, :8], rtol=0, atol=0)

    def test_batch_session_and_padding_isolation_with_caller_owned_history(self):
        model = self.model().eval()
        with torch.no_grad():
            first_inputs = self.inputs(model, ["A.", "A much longer unrelated premise."])
            _, together = model.forward_with_state(*first_inputs)
            first_inputs[1].zero_()
            self.assertNotEqual(int(together.text_history[0][0, 0]), 0)
            _, alone = model.forward_with_state(*self.inputs(model, ["A."]))
            result, _ = model.forward_with_state(*self.inputs(model, ["Why?", "Unrelated question?"]), together)
            independent, _ = model.forward_with_state(*self.inputs(model, ["Why?"]), alone)
            for name in ("logits", "production_context"):
                torch.testing.assert_close(result[name][:1], independent[name], rtol=1e-5, atol=2e-6)
            changed = tuple(value.clone() for value in together.text_history)
            changed[0][1, 1] = model.codec.BYTE_OFFSET + ord("Z")
            modified = RegionalSequenceState(together.brain_state, changed, together.text_lengths)
            other, _ = model.forward_with_state(*self.inputs(model, ["Why?", "Unrelated question?"]), modified)
            self.assertTrue(torch.equal(result["logits"][0], other["logits"][0]))
            fresh, _ = model.forward_with_state(*self.inputs(model, ["Why?"]))
            baseline, _ = model.forward_with_state(*self.inputs(model, ["Why?"]))
            self.assertTrue(torch.equal(fresh["logits"], baseline["logits"]))

    def test_prior_raw_bytes_receive_query_gradients_without_regional_history(self):
        model = self.model()
        with torch.no_grad():
            _, previous = model.forward_with_state(*self.inputs(model, ["Z"]))
        state = RegionalSequenceState(model.brain.initial_state(1), previous.text_history, previous.text_lengths)
        output, _ = model.forward_with_state(*self.inputs(model, ["a?"]), state)
        output["logits"].square().mean().backward()
        prior_token_gradient = model.posterior_temporal.tokens.weight.grad[ord("Z") + model.codec.BYTE_OFFSET]
        self.assertGreater(float(prior_token_gradient.abs().sum()), 0.)
        for module in (model.sequence_projection, model.retrieval_projection,
                       model.semantic_bridge, model.brain.regions["motor_cortex"]):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0.)

    def test_retrieval_and_actions_remain_on_existing_regional_paths(self):
        model = self.model()
        inputs = self.inputs(model, ["Is A before B?"])
        before, _ = model.forward_with_state(*inputs)
        lesioned, _ = model.forward_with_state(*inputs, ablate=("hippocampus",))
        with torch.no_grad():
            model.retrieval_projection.bias.add_(10.)
        after, _ = model.forward_with_state(*inputs)
        changed_lesion, _ = model.forward_with_state(*inputs, ablate=("hippocampus",))
        self.assertFalse(torch.equal(before["logits"], after["logits"]))
        self.assertTrue(torch.equal(lesioned["logits"], changed_lesion["logits"]))
        self.assertTrue(torch.equal(lesioned["production_context"], changed_lesion["production_context"]))
        motor_lesion, _ = model.forward_with_state(*inputs, ablate=("motor_cortex",))
        self.assertEqual(float(motor_lesion["logits"].abs().sum()), 0.)
        # With both text entry regions silenced, text cannot bypass them.
        disabled = ("temporal_language", "hippocampus")
        a, _ = model.forward_with_state(*inputs, ablate=disabled)
        b, _ = model.forward_with_state(*self.inputs(model, ["Completely different observation."]), ablate=disabled)
        self.assertTrue(torch.equal(a["logits"], b["logits"]))
        self.assertTrue(torch.equal(a["production_context"], b["production_context"]))

    def test_empty_text_is_finite_and_capacity_overflow_does_not_truncate(self):
        model = self.model(max_turns=2, max_positions=16)
        output, state = model.forward_with_state(*self.inputs(model, ["", ""]))
        self.assertTrue(torch.isfinite(output["logits"]).all())
        self.assertTrue(torch.isfinite(output["language_logits"]).all())
        _, state = model.forward_with_state(*self.inputs(model, ["", ""]), state)
        with self.assertRaisesRegex(ValueError, "turn capacity"):
            model.forward_with_state(*self.inputs(model, ["", ""]), state)
        with self.assertRaisesRegex(ValueError, "position capacity"):
            model.forward_with_state(*self.inputs(model, ["x" * 15]))
        _, prior = model.forward_with_state(*self.inputs(model, ["x" * 8]))
        with self.assertRaisesRegex(ValueError, "position capacity"):
            model.forward_with_state(*self.inputs(model, ["y" * 8]), prior)
        self.assertEqual(len(prior.text_history), 1)
        self.assertEqual(len(state.text_history), 2)

    def test_six_maximum_length_turns_fit_without_history_loss(self):
        model = self.model(max_positions=1024).eval()
        state = None
        with torch.no_grad():
            for turn in range(6):
                texts = [chr(65 + turn) * 128, "" if turn % 2 else "short"]
                inputs = self.inputs(model, texts)
                output, state = model.forward_with_state(*inputs, state)
                self.assertEqual(len(state.text_history), turn + 1)
                self.assertEqual(int(state.text_lengths[-1][0]), 130)
                self.assertTrue(torch.equal(state.text_history[-1], inputs[1]))
                self.assertTrue(torch.isfinite(output["logits"]).all())
        self.assertEqual(sum(int(length[0]) for length in state.text_lengths), 780)
        self.assertEqual(int(state.text_history[0][0, 1]), ord("A") + model.codec.BYTE_OFFSET)

    def test_exact_cpu_checkpoint_and_tensor_only_state_continuation(self):
        model = self.model()
        _, state = model.forward_with_state(*self.inputs(model, ["A before B.", "Q before R."]))
        _, state = model.forward_with_state(*self.inputs(model, ["B before C.", "R before S."]), state)
        digest = checkpoint_digest(model)
        payload = model.state_to_dict(state, checkpoint_sha256=digest)
        for value in payload["text_history"] + payload["text_lengths"] + list(payload["brain_state"].values()):
            self.assertEqual(value.device.type, "cpu")
            self.assertFalse(value.requires_grad)
        first = state.text_history[0].clone()
        payload["text_history"][0].zero_()
        self.assertTrue(torch.equal(first, state.text_history[0]))
        payload = model.state_to_dict(state, checkpoint_sha256=digest)
        stream = io.BytesIO()
        torch.save({"weights": model.state_dict(), "state": payload}, stream)
        stream.seek(0)
        loaded = torch.load(stream, weights_only=True)
        restored = build_regional_sequence_student(7, sequence_config=model.sequence_config)
        restored.load_state_dict(loaded["weights"], strict=True)
        resumed = restored.state_from_dict(loaded["state"], checkpoint_sha256=checkpoint_digest(restored))
        loaded["state"]["text_history"][0].zero_()
        self.assert_state_equal(model, state, resumed)
        detached = state.detached()
        self.assertIsNone(detached.brain_state.previous_prefrontal.grad_fn)
        inputs = self.inputs(model, ["Is A before C?", "Is Q before S?"])
        expected, expected_state = model.forward_with_state(*inputs, state)
        actual, actual_state = restored.forward_with_state(*inputs, resumed)
        for name in ("logits", "language_logits", "observation_language_logits"):
            self.assertTrue(torch.equal(expected[name], actual[name]), name)
        self.assert_state_equal(model, expected_state, actual_state)

    def test_state_rejects_incompatible_types_config_digest_and_corruption(self):
        model = self.model()
        _, state = model.forward_with_state(*self.inputs(model, ["A."]))
        digest = checkpoint_digest(model)
        payload = model.state_to_dict(state, checkpoint_sha256=digest)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            model.state_from_dict(payload, checkpoint_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "config"):
            self.model(max_turns=2).state_from_dict(payload, checkpoint_sha256=digest)
        with self.assertRaisesRegex(ValueError, "RegionalSequenceState"):
            model.forward_with_state(*self.inputs(model, ["A."]), EpisodicState(model.brain.initial_state(1)))
        for name in ("shape", "dtype", "bos", "nan"):
            corrupt = copy.deepcopy(payload)
            if name == "shape":
                corrupt["text_lengths"][0] = torch.tensor([[4]])
            elif name == "dtype":
                corrupt["text_history"][0] = corrupt["text_history"][0].float()
            elif name == "bos":
                corrupt["text_history"][0][0, 0] = 0
            else:
                corrupt["brain_state"]["previous_prefrontal"][0, 0] = float("nan")
            with self.assertRaises(ValueError):
                model.state_from_dict(corrupt, checkpoint_sha256=digest)


if __name__ == "__main__":
    unittest.main()
