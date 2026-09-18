"""Read-only structural checks: these tests never train either model pathway."""

from contextlib import ExitStack, contextmanager, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.language import ByteCodec, LanguageBrain, LanguageConfig
from brain_in_computer.language_readiness import readiness_report
from brain_in_computer.model import Brain, BrainConfig
from brain_in_computer.tasks import TaskStream


@contextmanager
def prohibit_training_and_model_saving():
    """Fail visibly if any readiness probe crosses the inference boundary."""
    operations = (
        "torch.optim.AdamW.__init__",
        "torch.optim.AdamW.step",
        "torch.optim.Optimizer.step",
        "torch.Tensor.backward",
        "torch.save",
        "brain_in_computer.training.Trainer.__init__",
        "brain_in_computer.training.Trainer.load",
        "brain_in_computer.training.Trainer.train_step",
    )
    with ExitStack() as stack:
        for operation in operations:
            stack.enter_context(patch(operation, side_effect=AssertionError(
                f"Readiness must not call {operation}")))
        yield


class ReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.previous_rng = torch.get_rng_state().clone()
        self.training_guard = prohibit_training_and_model_saving()
        self.training_guard.__enter__()

    def tearDown(self):
        self.training_guard.__exit__(None, None, None)
        torch.set_rng_state(self.previous_rng)

    def assert_outputs_equal(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for name in expected:
            if isinstance(expected[name], dict):
                self.assert_outputs_equal(expected[name], actual[name])
            else:
                self.assertTrue(torch.equal(expected[name], actual[name]), name)

    def assert_weights_equal(self, expected, model):
        self.assert_outputs_equal(expected, model.state_dict())
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_readiness_with_reference_checkpoint_is_read_only_and_preserves_rng(self):
        checkpoint = Path(__file__).resolve().parents[1] / "runs" / "reference" / "checkpoint.pt"
        if not checkpoint.exists():
            self.skipTest("Reference symbolic checkpoint is not included in this checkout")
        before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        before_rng = torch.get_rng_state().clone()
        report = readiness_report(checkpoint=checkpoint, device="cpu", seed=17)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(before_hash, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        self.assertEqual(report["symbolic_core_checkpoint_sha256"], before_hash)
        self.assertFalse(report["language_training_performed"])
        self.assertEqual(report["english_optimizer_steps"], 0)
        self.assertFalse(report["language_checkpoint_saved"])
        self.assertFalse(report["model_downloaded"])
        self.assertTrue(report["checks"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["output_shapes"]["language_logits"], [2, 1, ByteCodec.VOCAB_SIZE])
        self.assertEqual(report["total_parameters"], sum(report["parameter_counts"].values()))

    def test_fresh_readiness_and_cli_use_no_training_or_save(self):
        from brain_in_computer.cli import main

        before_rng = torch.get_rng_state().clone()
        captured = io.StringIO()
        with redirect_stdout(captured):
            code = main(["language-check", "--device", "cpu", "--threads", "1",
                         "--language-hidden-size", "32", "--embedding-size", "16"])
        self.assertEqual(code, 0)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        report = json.loads(captured.getvalue())
        self.assertIsNone(report["symbolic_core_checkpoint_sha256"])
        self.assertFalse(report["language_training_performed"])
        self.assertTrue(all(report["checks"].values()))
        captured = io.StringIO()
        with redirect_stdout(captured):
            code = main(["language-plan", "--parameters", "1000000", "--threads", "1"])
        self.assertEqual(code, 0)
        plan = json.loads(captured.getvalue())
        self.assertFalse(plan["language_training_performed"])
        self.assertIsNone(plan["estimates"][0]["estimated_training_seconds"])

    def test_language_context_validation_and_none_preserve_core_behavior(self):
        torch.manual_seed(123)
        model = Brain(BrainConfig(hidden_size=16)).eval()
        observations = TaskStream(99).sample(2, delay=2).observations
        before = deepcopy(model.state_dict())
        with torch.no_grad():
            baseline = model(observations, return_activity=True)
            omitted = model(observations, return_activity=True, language_context=None)
            zero_context = model(observations, return_activity=True,
                                 language_context=torch.zeros(2, 4, 16))
        self.assert_outputs_equal(baseline, omitted)
        self.assert_outputs_equal(baseline, zero_context)
        invalid_contexts = (
            [],
            torch.zeros(2, 4, 15),
            torch.zeros(2, 3, 16),
            torch.zeros(2, 4, 16, dtype=torch.float64),
            torch.zeros(2, 4, 16, device="meta"),
            torch.full((2, 4, 16), float("nan")),
            torch.full((2, 4, 16), float("inf")),
        )
        for context in invalid_contexts:
            with self.subTest(context=str(type(context))), self.assertRaises(ValueError):
                model(observations, language_context=context)
        self.assert_weights_equal(before, model)

    def test_future_context_cannot_change_past_core_outputs(self):
        torch.manual_seed(124)
        model = Brain(BrainConfig(hidden_size=16)).eval()
        observations = TaskStream(100).sample(2, delay=3).observations
        context = torch.randn(2, 5, 16)
        changed = context.clone()
        changed[:, 3:] += torch.randn_like(changed[:, 3:]) * 4
        with torch.no_grad():
            first = model(observations, language_context=context, return_activity=True)
            second = model(observations, language_context=changed, return_activity=True)
        for key in ("logits", "visual_logits", "auditory_logits", "prediction", "value"):
            self.assertTrue(torch.equal(first[key][:, :3], second[key][:, :3]), key)
        for region in first["region_activity"]:
            self.assertTrue(torch.equal(first["region_activity"][region][:, :3],
                                        second["region_activity"][region][:, :3]), region)
        self.assertFalse(torch.equal(first["logits"][:, 3:], second["logits"][:, 3:]))

    def test_temporal_ablation_blocks_all_context_influence_on_core(self):
        torch.manual_seed(125)
        model = Brain(BrainConfig(hidden_size=16)).eval()
        observations = TaskStream(101).sample(2, delay=2).observations
        with torch.no_grad():
            context_free = model(observations, ablate=("temporal_language",), return_activity=True)
            with_context = model(observations, ablate=("temporal_language",), return_activity=True,
                                 language_context=torch.randn(2, 4, 16))
        self.assert_outputs_equal(context_free, with_context)

    def test_language_forward_and_decoder_are_causal_without_weight_updates(self):
        torch.manual_seed(126)
        model = LanguageBrain(Brain(BrainConfig(hidden_size=16)),
                              LanguageConfig(hidden_size=32, embedding_size=16)).eval()
        observations = TaskStream(102).sample(2, delay=1).observations
        ids, lengths = model.prepare_inputs(["Look at the cube.", "Wait."])
        decoder = torch.tensor([
            [ByteCodec.BOS, 100, 101, 102, 103],
            [ByteCodec.BOS, 110, 111, 112, 113],
        ], dtype=torch.long)
        changed_decoder = decoder.clone()
        changed_decoder[:, 3:] += 9
        before = deepcopy(model.state_dict())
        before_rng = torch.get_rng_state().clone()
        with torch.no_grad():
            first = model(observations, ids, lengths, decoder)
            second = model(observations, ids, lengths, changed_decoder)
        self.assertTrue(torch.equal(first["language_logits"][:, :3], second["language_logits"][:, :3]))
        self.assertFalse(torch.equal(first["language_logits"][:, 3:], second["language_logits"][:, 3:]))
        self.assertTrue(torch.equal(first["logits"], second["logits"]))
        self.assertTrue(torch.equal(first["production_context"], second["production_context"]))
        self.assert_weights_equal(before, model)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
