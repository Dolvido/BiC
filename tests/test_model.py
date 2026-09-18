"""Behavioral checks for trainability, causal messaging, and regional lesions."""

import unittest
from dataclasses import fields

import torch

from brain_in_computer.model import Brain, BrainConfig, CONNECTOME, REGION_NAMES
from brain_in_computer.regions import Hippocampus
from brain_in_computer.tasks import TaskStream
from brain_in_computer.training import learning_loss


class BrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(31)
        self.model = Brain(BrainConfig(hidden_size=12, memory_slots=3))
        self.batch = TaskStream(53).sample(16, delay=3)

    def observations_copy(self):
        return {key: value.clone() for key, value in self.batch.observations.items()}

    def test_thirteen_regions_have_distinct_trainable_parameters(self):
        self.assertEqual(len(REGION_NAMES), 13)
        self.assertEqual(set(self.model.regions), set(REGION_NAMES))
        identities, storage = set(), set()
        counts = self.model.region_parameter_counts()
        for name, region in self.model.regions.items():
            with self.subTest(region=name):
                parameters = list(region.parameters())
                self.assertGreater(counts[name], 0)
                self.assertEqual(counts[name], sum(p.numel() for p in parameters))
                for parameter in parameters:
                    self.assertTrue(parameter.requires_grad)
                    self.assertNotIn(id(parameter), identities)
                    self.assertNotIn(parameter.untyped_storage().data_ptr(), storage)
                    identities.add(id(parameter))
                    storage.add(parameter.untyped_storage().data_ptr())
        self.assertEqual(sum(counts.values()), sum(p.numel() for p in self.model.parameters()))
        self.assertEqual(len(CONNECTOME), len(set(CONNECTOME)))
        for source, destination, delay in CONNECTOME:
            self.assertIn(source, REGION_NAMES)
            self.assertIn(destination, REGION_NAMES)
            self.assertIn(delay, (0, 1))

    def test_loss_reaches_every_region_with_finite_nonzero_gradients(self):
        outputs = self.model(self.batch.observations)
        loss, components = learning_loss(outputs, self.batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in components.values()))
        loss.backward()
        for name, region in self.model.regions.items():
            with self.subTest(region=name):
                gradients = [p.grad for p in region.parameters()]
                self.assertTrue(all(g is not None for g in gradients))
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.0)

    def test_output_contract_and_all_regional_ablations(self):
        expected_shapes = {
            "logits": (16, 5, 4), "visual_logits": (16, 5, 4),
            "auditory_logits": (16, 5, 4), "prediction": (16, 5, 20),
            "value": (16, 5),
        }
        with torch.no_grad():
            baseline = self.model(self.batch.observations, return_activity=True)
            self.assertEqual(set(baseline), {*expected_shapes, "region_activity"})
            for key, shape in expected_shapes.items():
                self.assertEqual(tuple(baseline[key].shape), shape)
                self.assertTrue(torch.isfinite(baseline[key]).all())
            self.assertEqual(set(baseline["region_activity"]), set(REGION_NAMES))
            for name, activity in baseline["region_activity"].items():
                self.assertEqual(tuple(activity.shape), (16, 5, 12), name)
                self.assertTrue(torch.isfinite(activity).all(), name)
            for name in REGION_NAMES:
                with self.subTest(region=name):
                    lesioned = self.model(self.batch.observations, ablate=(name,), return_activity=True)
                    self.assertEqual(int(torch.count_nonzero(lesioned["region_activity"][name])), 0)
                    self.assertFalse(torch.equal(baseline["logits"], lesioned["logits"]))
                    for key in expected_shapes:
                        self.assertTrue(torch.isfinite(lesioned[key]).all())
                    head = {"visual_cortex": "visual_logits", "auditory_cortex": "auditory_logits",
                            "motor_cortex": "logits", "cerebellum": "prediction", "basal_ganglia": "value"}.get(name)
                    if head:
                        self.assertEqual(int(torch.count_nonzero(lesioned[head])), 0)
            complete = self.model(self.batch.observations, ablate=REGION_NAMES, return_activity=True)
            for key in expected_shapes:
                self.assertEqual(int(torch.count_nonzero(complete[key])), 0)
            for activity in complete["region_activity"].values():
                self.assertEqual(int(torch.count_nonzero(activity)), 0)

    def test_ablated_regions_are_disconnected_from_learning_loss(self):
        for name in REGION_NAMES:
            with self.subTest(region=name):
                self.model.zero_grad(set_to_none=True)
                output = self.model(self.batch.observations, ablate=(name,))
                loss, _ = learning_loss(output, self.batch)
                loss.backward()
                for parameter in self.model.regions[name].parameters():
                    self.assertTrue(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0)

    def test_future_observations_cannot_change_past_outputs(self):
        altered = self.observations_copy()
        for name, tensor in altered.items():
            if name == "tokens":
                tensor[:, 3:] = (tensor[:, 3:] + 1) % self.model.config.vocab_size
            else:
                tensor[:, 3:] += 0.7
        with torch.no_grad():
            first = self.model(self.batch.observations, return_activity=True)
            second = self.model(altered, return_activity=True)
        for key in ("logits", "visual_logits", "auditory_logits", "prediction", "value"):
            self.assertTrue(torch.equal(first[key][:, :3], second[key][:, :3]), key)
        for name in REGION_NAMES:
            self.assertTrue(torch.equal(first["region_activity"][name][:, :3], second["region_activity"][name][:, :3]), name)
        self.assertFalse(torch.equal(first["logits"][:, 3:], second["logits"][:, 3:]))

    def test_feedback_pathways_have_one_observation_step_delay(self):
        with torch.no_grad():
            first = self.model(self.batch.observations, return_activity=True)
            for source, destination in (
                ("prefrontal_cortex", "visual_cortex"),
                ("prefrontal_cortex", "auditory_cortex"),
                ("prefrontal_cortex", "temporal_language"),
                ("motor_cortex", "somatosensory_cortex"),
            ):
                with self.subTest(source=source, destination=destination):
                    second = self.model(self.batch.observations, ablate=(source,), return_activity=True)
                    before, after = first["region_activity"][destination], second["region_activity"][destination]
                    self.assertTrue(torch.equal(before[:, 0], after[:, 0]))
                    self.assertFalse(torch.equal(before[:, 1:], after[:, 1:]))
            no_cerebellum = self.model(self.batch.observations, ablate=("cerebellum",))
            self.assertTrue(torch.equal(first["logits"][:, 0], no_cerebellum["logits"][:, 0]))
            self.assertFalse(torch.equal(first["logits"][:, 1:], no_cerebellum["logits"][:, 1:]))

    def test_forward_calls_reset_state_and_batch_members_are_independent(self):
        parameters = {k: v.clone() for k, v in self.model.state_dict().items()}
        with torch.no_grad():
            first = self.model(self.batch.observations)
            self.model(TaskStream(200).sample(5, delay=8).observations)
            again = self.model(self.batch.observations)
            single = self.model({k: v[:1] for k, v in self.batch.observations.items()})
            altered = self.observations_copy()
            for name in ("visual", "auditory", "body", "feedback"):
                altered[name][1:] += 0.5
            altered["tokens"][1:] = 0
            changed_batch = self.model(altered)
        for key in first:
            self.assertTrue(torch.equal(first[key], again[key]), key)
            torch.testing.assert_close(first[key][:1], single[key])
            self.assertTrue(torch.equal(first[key][:1], changed_batch[key][:1]), key)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(parameters[name], value), name)

    def test_config_and_input_validation(self):
        for field in fields(BrainConfig):
            for invalid in (0, -1, True, 1.5, "4"):
                with self.subTest(field=field.name, value=invalid), self.assertRaises(ValueError):
                    BrainConfig(**{field.name: invalid})
        for name in self.batch.observations:
            with self.subTest(missing=name):
                obs = self.observations_copy()
                del obs[name]
                with self.assertRaises(ValueError):
                    self.model(obs)
            with self.subTest(wrong_shape=name):
                obs = self.observations_copy()
                obs[name] = obs[name][:, :-1]
                with self.assertRaises(ValueError):
                    self.model(obs)
            with self.subTest(wrong_dtype=name):
                obs = self.observations_copy()
                obs[name] = obs[name].float() if name == "tokens" else obs[name].double()
                with self.assertRaises(ValueError):
                    self.model(obs)
        for bad in ("visual_cortex", ("unknown",)):
            with self.subTest(ablate=bad), self.assertRaises(ValueError):
                self.model(self.batch.observations, ablate=bad)
        for axis in (0, 1):
            with self.subTest(empty_axis=axis):
                obs = {k: v[:0] if axis == 0 else v[:, :0] for k, v in self.batch.observations.items()}
                with self.assertRaises(ValueError):
                    self.model(obs)

    def test_nonfinite_observations_and_out_of_vocabulary_tokens_are_rejected(self):
        for name in ("visual", "auditory", "body", "feedback"):
            for invalid in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(channel=name, value=invalid):
                    obs = self.observations_copy()
                    obs[name][0, 0, 0] = invalid
                    with self.assertRaises(ValueError):
                        self.model(obs)
        for invalid in (-1, self.model.config.vocab_size):
            with self.subTest(token=invalid):
                obs = self.observations_copy()
                obs["tokens"][0, 0] = invalid
                with self.assertRaises(ValueError):
                    self.model(obs)

    def test_custom_dimensions_and_model_dtype(self):
        config = BrainConfig(hidden_size=7, visual_dim=3, auditory_dim=5, body_dim=2,
                             vocab_size=9, num_actions=6, memory_slots=2)
        model = Brain(config).double()
        observations = {
            "visual": torch.randn(2, 3, 3, dtype=torch.float64),
            "auditory": torch.randn(2, 3, 5, dtype=torch.float64),
            "body": torch.randn(2, 3, 2, dtype=torch.float64),
            "feedback": torch.randn(2, 3, 2, dtype=torch.float64),
            "tokens": torch.randint(9, (2, 3)),
        }
        outputs = model(observations)
        self.assertEqual(tuple(outputs["logits"].shape), (2, 3, 6))
        self.assertEqual(tuple(outputs["prediction"].shape), (2, 3, 10))
        self.assertTrue(all(t.dtype == torch.float64 for t in outputs.values()))

    def test_hippocampal_memory_is_bounded_and_retains_write_gradients(self):
        memory_region = Hippocampus(hidden_size=6, memory_slots=3)
        memory, associations = (), []
        for step in range(5):
            association = torch.randn(2, 6, requires_grad=True)
            associations.append(association)
            inputs = [torch.randn(2, 6) for _ in range(4)]
            recalled, memory = memory_region(association, *inputs, memory)
            self.assertEqual(len(memory), min(step + 1, 3))
            self.assertTrue(all(write.requires_grad for write in memory))
        recalled.square().sum().backward()
        for old in associations[:2]:
            self.assertIsNone(old.grad)
        for retained in associations[2:]:
            self.assertIsNotNone(retained.grad)
            self.assertTrue(torch.isfinite(retained.grad).all())
            self.assertGreater(float(retained.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
