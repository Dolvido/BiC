"""Reusable episodic retrieval retains causal and session boundaries."""
import copy
import io
import unittest

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments.cognitive_student import EpisodicState, build_cognitive_student


class CognitiveStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def inputs(self, model, texts, replies=None):
        batch = len(texts)
        observations = {name: torch.zeros(batch, 1, width) for name, width in
                        (("visual", 32), ("auditory", 4), ("body", 4), ("feedback", 2))}
        observations["tokens"] = torch.zeros(batch, 1, dtype=torch.long)
        ids, lengths = model.prepare_inputs(texts)
        decoder, _ = model.codec.batch_encode(replies or ["Yes."] * batch)
        return observations, ids, lengths, decoder[:, :-1]

    def assert_same_state(self, model, left, right):
        for name, value in model.brain.state_to_dict(left.brain_state).items():
            self.assertTrue(torch.equal(value, model.brain.state_to_dict(right.brain_state)[name]))
        for a, b in zip(left.token_memory + left.token_masks, right.token_memory + right.token_masks):
            self.assertTrue(torch.equal(a, b))

    def test_modes_share_parameters_initialization_and_do_not_consume_rng(self):
        before = torch.random.get_rng_state().clone()
        episodic = build_cognitive_student(37)
        recurrent = build_cognitive_student(37, memory_mode="recurrent")
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(episodic.state_dict().keys(), recurrent.state_dict().keys())
        for name, value in episodic.state_dict().items():
            self.assertTrue(torch.equal(value, recurrent.state_dict()[name]))
        self.assertEqual(sum(episodic.parameter_counts().values()),
                         sum(p.numel() for p in episodic.parameters()))
        self.assertTrue(200000 <= sum(episodic.parameter_counts().values()) <= 1000000)
        self.assertEqual(len(episodic.brain.region_names), 13)

    def test_decoder_bytes_cannot_change_policy_memory_or_observation_prediction(self):
        model = build_cognitive_student(37)
        left = self.inputs(model, ["A is before B.", "B is before C."], ["Yes.", "No."])
        right = self.inputs(model, ["A is before B.", "B is before C."], ["No.", "Yes."])
        first, state_a = model.forward_with_state(*left)
        second, state_b = model.forward_with_state(*right)
        for name in ("logits", "production_context", "observation_language_logits"):
            self.assertTrue(torch.equal(first[name], second[name]))
        self.assert_same_state(model, state_a, state_b)

    def test_next_byte_prediction_has_no_future_byte_access(self):
        model = build_cognitive_student(37)
        left, _ = model.forward_with_state(*self.inputs(model, ["A then B."]))
        right, _ = model.forward_with_state(*self.inputs(model, ["A then C."]))
        # BOS plus the common seven-byte prefix precede the changed byte.
        self.assertTrue(torch.equal(left["observation_language_logits"][:, :8],
                                    right["observation_language_logits"][:, :8]))

    def test_episodic_gradients_reach_previous_encoder_states_and_blank_is_finite(self):
        model = build_cognitive_student(37)
        _, initial = model.forward_with_state(*self.inputs(model, ["A is before B."]))
        memory = initial.token_memory[0]
        memory.retain_grad()
        # Erase the ordinary recurrence to isolate the episodic gradient path.
        state = EpisodicState(model.brain.initial_state(1), initial.token_memory, initial.token_masks)
        output, _ = model.forward_with_state(*self.inputs(model, ["Is A before B?"]), state)
        output["logits"].square().mean().backward()
        self.assertGreater(float(memory.grad.abs().sum()), 0.)
        self.assertGreater(float(model.posterior_temporal.embedding.weight.grad.abs().sum()), 0.)
        for mode in ("episodic", "recurrent"):
            empty_model = build_cognitive_student(37, memory_mode=mode)
            output, state = empty_model.forward_with_state(*self.inputs(empty_model, ["", ""]))
            self.assertTrue(torch.isfinite(output["logits"]).all())
            self.assertFalse(bool(state.token_masks[0].any()))

    def test_control_ignores_previous_token_storage_and_rows_remain_isolated(self):
        for mode in ("episodic", "recurrent"):
            model = build_cognitive_student(37, memory_mode=mode)
            _, state = model.forward_with_state(*self.inputs(model, ["A then B.", "C then D."]))
            changed = state.token_memory[0].clone()
            changed[1] = changed[1] + 10.
            altered = EpisodicState(state.brain_state, (changed,), state.token_masks)
            inputs = self.inputs(model, ["What follows A?", "What follows C?"])
            normal, _ = model.forward_with_state(*inputs, state)
            modified, _ = model.forward_with_state(*inputs, altered)
            self.assertTrue(torch.equal(normal["logits"][0], modified["logits"][0]))
            if mode == "recurrent":
                self.assertTrue(torch.equal(normal["logits"], modified["logits"]))
            else:
                self.assertFalse(torch.equal(normal["logits"][1], modified["logits"][1]))

    def test_memory_enters_through_hippocampus_and_is_silenced_by_lesion(self):
        model = build_cognitive_student(37)
        _, state = model.forward_with_state(*self.inputs(model, ["A then B."]))
        altered = EpisodicState(state.brain_state, (state.token_memory[0] + 10.,), state.token_masks)
        inputs = self.inputs(model, ["What follows A?"])
        normal, _ = model.forward_with_state(*inputs, state, ablate=("hippocampus",))
        modified, _ = model.forward_with_state(*inputs, altered, ablate=("hippocampus",))
        self.assertTrue(torch.equal(normal["logits"], modified["logits"]))
        self.assertEqual(float(normal["region_activity"]["hippocampus"].abs().sum()), 0.)

    def test_eviction_detach_and_exact_tensor_only_state_restart(self):
        model = build_cognitive_student(37)
        state, observed = None, []
        for turn in range(10):
            output, state = model.forward_with_state(*self.inputs(model, [f"Turn {turn}."]), state)
            observed.append(output["comprehension_states"])
        self.assertEqual(len(state.token_memory), 8)
        for actual, expected in zip(state.token_memory, observed[-8:]):
            self.assertIs(actual, expected)
        detached = state.detached()
        self.assertTrue(all(value.grad_fn is None for value in detached.token_memory))
        digest = checkpoint_digest(model)
        payload = model.state_to_dict(state, checkpoint_sha256=digest)
        for value in payload["token_memory"]:
            self.assertEqual(value.device.type, "cpu")
            self.assertFalse(value.requires_grad)
        before = state.token_memory[0].clone()
        payload["token_memory"][0].zero_()
        self.assertTrue(torch.equal(before, state.token_memory[0]))
        payload = model.state_to_dict(state, checkpoint_sha256=digest)
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        loaded = torch.load(stream, weights_only=True)
        restored = build_cognitive_student(37)
        next_state = restored.state_from_dict(loaded, checkpoint_sha256=digest)
        loaded["token_memory"][0].zero_()
        self.assertTrue(torch.equal(next_state.token_memory[0], state.token_memory[0]))
        inputs = self.inputs(model, ["What happened earlier?"])
        expected, expected_state = model.forward_with_state(*inputs, state)
        actual, actual_state = restored.forward_with_state(*inputs, next_state)
        self.assertTrue(torch.equal(expected["logits"], actual["logits"]))
        self.assertTrue(torch.equal(expected["language_logits"], actual["language_logits"]))
        self.assert_same_state(model, expected_state, actual_state)

    def test_saved_state_rejects_digest_config_corruption_and_invalid_values(self):
        model = build_cognitive_student(37)
        _, state = model.forward_with_state(*self.inputs(model, ["A then B."]))
        digest = checkpoint_digest(model)
        payload = model.state_to_dict(state, checkpoint_sha256=digest)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            model.state_from_dict(payload, checkpoint_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "config"):
            build_cognitive_student(37, memory_mode="recurrent").state_from_dict(payload, checkpoint_sha256=digest)
        for failure in ("nan", "shape", "mask"):
            bad = copy.deepcopy(payload)
            if failure == "nan":
                bad["token_memory"][0][0, 0, 0] = float("nan")
            elif failure == "shape":
                bad["token_memory"][0] = bad["token_memory"][0][:, :, :-1]
            else:
                bad["token_masks"][0] = bad["token_masks"][0].float()
            with self.assertRaises(ValueError):
                model.state_from_dict(bad, checkpoint_sha256=digest)


if __name__ == "__main__":
    unittest.main()
