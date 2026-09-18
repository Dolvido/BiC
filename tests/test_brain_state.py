"""Behavioral checks for caller-owned state and uninterrupted computation."""
import copy
from dataclasses import replace
import io
import unittest

import torch

from brain_in_computer.model import Brain, BrainConfig, BrainState, REGION_NAMES


class BrainStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(709)
        self.model = Brain(BrainConfig(hidden_size=8, visual_dim=4, auditory_dim=3,
                                      body_dim=2, memory_slots=3)).double()
        self.observations = {
            "visual": torch.randn(3, 7, 4, dtype=torch.float64),
            "auditory": torch.randn(3, 7, 3, dtype=torch.float64),
            "body": torch.randn(3, 7, 2, dtype=torch.float64),
            "feedback": torch.randn(3, 7, 2, dtype=torch.float64),
            "tokens": torch.randint(8, (3, 7)),
        }

    def chunk(self, start, stop, observations=None):
        return {key: value[:, start:stop] for key, value in
                (self.observations if observations is None else observations).items()}

    def assert_outputs_close(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            if isinstance(left[key], dict):
                self.assert_outputs_close(left[key], right[key])
            else:
                torch.testing.assert_close(left[key], right[key], rtol=1e-10, atol=1e-12)

    def join_outputs(self, parts):
        return {
            key: ({region: torch.cat([part[key][region] for part in parts], dim=1)
                   for region in parts[0][key]} if isinstance(parts[0][key], dict)
                  else torch.cat([part[key] for part in parts], dim=1))
            for key in parts[0]
        }

    def test_full_sequence_equals_chunks_with_contexts_and_memory_eviction(self):
        language = torch.randn(3, 7, 8, dtype=torch.float64)
        recall = torch.randn(3, 7, 8, dtype=torch.float64)
        for ablate in ((), ("hippocampus",), ("prefrontal_cortex", "cerebellum")):
            with self.subTest(ablate=ablate), torch.no_grad():
                full, end_full = self.model.forward_with_state(
                    self.observations, ablate=ablate, return_activity=True,
                    language_context=language, memory_context=recall)
                parts, state = [], None
                for start, stop in ((0, 1), (1, 3), (3, 7)):
                    part, state = self.model.forward_with_state(
                        self.chunk(start, stop), state, ablate, return_activity=True,
                        language_context=language[:, start:stop], memory_context=recall[:, start:stop])
                    parts.append(part)
                self.assert_outputs_close(full, self.join_outputs(parts))
                self.assert_outputs_close(self.model.state_to_dict(end_full), self.model.state_to_dict(state))
                self.assertEqual(len(state.memory), 0 if "hippocampus" in ablate else 3)

    def test_default_forward_resets_without_module_or_supplied_state_mutation(self):
        weights_before = {key: value.clone() for key, value in self.model.state_dict().items()}
        buffers_before = tuple(self.model.named_buffers())
        with torch.no_grad():
            baseline = self.model(self.observations)
            initial = self.model.initial_state(3)
            saved = self.model.state_to_dict(initial)
            _, continuing = self.model.forward_with_state(self.observations, initial)
            self.model.forward_with_state(self.observations, continuing)
            self.assert_outputs_close(self.model(self.observations), baseline)
            self.assert_outputs_close(self.model.state_to_dict(initial), saved)
        self.assertEqual(tuple(self.model.named_buffers()), buffers_before)
        self.assert_outputs_close(self.model.state_dict(), weights_before)

    def test_batch_sessions_are_independent_across_chunks(self):
        with torch.no_grad():
            _, state = self.model.forward_with_state(self.chunk(0, 3))
            reference, _ = self.model.forward_with_state(self.chunk(3, 7), state)
            changed_state = self.model.state_from_dict(self.model.state_to_dict(state))
            changed_state.previous_prefrontal[1:] += 4
            changed_state.memory[0][1:] -= 4
            changed = self.chunk(3, 7)
            changed = {key: value.clone() for key, value in changed.items()}
            changed["visual"][1:] += 3
            changed["tokens"][1:] = 0
            result, _ = self.model.forward_with_state(changed, changed_state)
            self.assert_outputs_close({k: v[:1] for k, v in reference.items()},
                                      {k: v[:1] for k, v in result.items()})
            self.assertFalse(torch.equal(reference["logits"][1:], result["logits"][1:]))

    def test_state_roundtrip_resumes_all_outputs_and_has_independent_storage(self):
        with torch.no_grad():
            _, state = self.model.forward_with_state(self.chunk(0, 4))
            expected, expected_state = self.model.forward_with_state(self.chunk(4, 7), state)
            payload = self.model.state_to_dict(state)
            stream = io.BytesIO()
            torch.save(payload, stream)
            stream.seek(0)
            loaded = torch.load(stream, weights_only=True, map_location="cpu")
            restarted = Brain(self.model.config).double()
            restarted.load_state_dict(self.model.state_dict(), strict=True)
            restored = restarted.state_from_dict(loaded)
            result, result_state = restarted.forward_with_state(self.chunk(4, 7), restored)
            self.assert_outputs_close(expected, result)
            self.assert_outputs_close(self.model.state_to_dict(expected_state),
                                      self.model.state_to_dict(result_state))
            for name in expected:
                self.assertTrue(torch.equal(expected[name], result[name]), name)
            for name, value in self.model.state_to_dict(expected_state).items():
                self.assertTrue(torch.equal(value, restarted.state_to_dict(result_state)[name]), name)
            prior = state.previous_prefrontal.clone()
            loaded["previous_prefrontal"].zero_()
            self.assertTrue(torch.equal(state.previous_prefrontal, prior))
            self.assertTrue(torch.equal(restored.previous_prefrontal, prior))
            payload["memory"].zero_()
            self.assertTrue(any(torch.count_nonzero(item) for item in state.memory))

    def test_empty_memory_roundtrip_and_model_dtype_conversion(self):
        zero = self.model.initial_state(2)
        data = self.model.state_to_dict(zero)
        self.assertEqual(data["memory"].shape, (2, 0, 8))
        floating_model = copy.deepcopy(self.model).float()
        restored = floating_model.state_from_dict(data)
        self.assertEqual(restored.previous_prefrontal.dtype, torch.float32)
        self.assertEqual(restored.memory, ())

    def test_invalid_state_shape_dtype_device_finite_and_bound_are_rejected(self):
        valid = self.model.initial_state(3)
        bad = [
            {}, replace(valid, previous_motor=torch.zeros(2, 8, dtype=torch.float64)),
            replace(valid, previous_prediction=torch.zeros(3, 8, dtype=torch.float64)),
            replace(valid, previous_prefrontal=torch.zeros(3, 8)),
            replace(valid, previous_cerebellum=torch.zeros(3, 8, dtype=torch.float64, device="meta")),
            replace(valid, previous_motor=torch.full((3, 8), float("nan"), dtype=torch.float64)),
            replace(valid, memory=[valid.previous_motor]),
            replace(valid, memory=(valid.previous_motor,) * 4),
            replace(valid, memory=(torch.zeros(3, 7, dtype=torch.float64),)),
            replace(valid, memory=(torch.full((3, 8), float("inf"), dtype=torch.float64),)),
        ]
        for state in bad:
            with self.subTest(state_type=type(state)), self.assertRaises(ValueError):
                self.model.forward_with_state(self.observations, state)
        for batch in (True, 0, -1, 1.2):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                self.model.initial_state(batch)

    def test_saved_state_rejects_schema_shape_and_nonfinite_data(self):
        data = self.model.state_to_dict(self.model.initial_state(3))
        invalid = [
            dict(data, extra=torch.tensor(0)),
            dict(data, schema_version=torch.tensor(2)),
            dict(data, schema_version=torch.tensor(1.0)),
            dict(data, memory=torch.zeros(3, 4, 8, dtype=torch.float64)),
            dict(data, memory=torch.zeros(3, 0, 8)),
            dict(data, previous_motor=torch.full((3, 8), float("inf"), dtype=torch.float64)),
        ]
        for record in invalid:
            with self.subTest(keys=list(record)), self.assertRaises(ValueError):
                self.model.state_from_dict(record)

    def test_gradient_continuity_matches_full_sequence(self):
        other = copy.deepcopy(self.model)
        whole = {key: value.clone().requires_grad_(key != "tokens") for key, value in self.observations.items()}
        split = {key: value.clone().requires_grad_(key != "tokens") for key, value in self.observations.items()}
        output = self.model(whole)
        loss = output["logits"][:, 3:].square().sum() + output["prediction"][:, 3:].square().sum()
        loss.backward()
        _, state = other.forward_with_state(self.chunk(0, 3, split))
        tail, _ = other.forward_with_state(self.chunk(3, 7, split), state)
        (tail["logits"].square().sum() + tail["prediction"].square().sum()).backward()
        self.assertGreater(float(split["visual"].grad[:, :3].abs().sum()), 0)
        for key in ("visual", "auditory", "body", "feedback"):
            torch.testing.assert_close(whole[key].grad, split[key].grad, rtol=1e-9, atol=1e-11)
        for (name, first), (_, second) in zip(self.model.named_parameters(), other.named_parameters()):
            with self.subTest(parameter=name):
                if first.grad is None:
                    self.assertIsNone(second.grad)
                else:
                    torch.testing.assert_close(first.grad, second.grad, rtol=1e-9, atol=1e-11)

    def test_explicit_detachment_blocks_prior_observation_gradients(self):
        prefix = {key: value[:, :3].clone().requires_grad_(key != "tokens")
                  for key, value in self.observations.items()}
        _, state = self.model.forward_with_state(prefix)
        self.assertTrue(state.previous_prefrontal.requires_grad)
        detached = state.detached()
        self.assertFalse(detached.previous_prefrontal.requires_grad)
        self.assertTrue(all(not item.requires_grad for item in detached.memory))
        output, _ = self.model.forward_with_state(self.chunk(3, 7), detached)
        output["logits"].square().sum().backward()
        self.assertIsNone(prefix["visual"].grad)

    def test_new_lesion_silences_carried_regional_outputs(self):
        with torch.no_grad():
            _, state = self.model.forward_with_state(self.chunk(0, 3))
            original = self.model.state_to_dict(state)
            output, next_state = self.model.forward_with_state(
                self.chunk(3, 7), state, ablate=REGION_NAMES, return_activity=True)
            for key, value in output.items():
                for tensor in value.values() if isinstance(value, dict) else (value,):
                    self.assertEqual(torch.count_nonzero(tensor), 0)
            self.assertEqual(next_state.memory, ())
            self.assert_outputs_close(original, self.model.state_to_dict(state))


if __name__ == "__main__":
    unittest.main()
