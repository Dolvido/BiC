"""Controller, teacher boundary, real optimization, and checkpoint integration."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.computer_use.environment import ACTION_NAMES, COLORS, TASKS, MiniDesktop
from brain_in_computer.computer_use.model import ComputerBrain, ComputerConfig
from brain_in_computer.computer_use.training import (
    ComputerTrainConfig, VISIBLE_TEXTS, build_demonstrations, evaluate_computer, history_tensors,
    load_computer_checkpoint, run_turn, tensor_batch, train_computer, visible_labels,
)
from brain_in_computer.language import ByteCodec


TRAINING_MODULE = "brain_in_computer.computer_use.training"


class ComputerTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(61)
        self.config = ComputerConfig(hidden_size=8, language_hidden_size=12,
                                     embedding_size=6, visual_features=8, max_history=3)
        self.model = ComputerBrain(self.config)
        self.env = MiniDesktop(seed=51)

    def force_action_and_empty_reply(self, action):
        """Use an actual network with deterministic output heads, not a teacher."""
        with torch.no_grad():
            readout = self.model.brain.regions["motor_cortex"].action_head
            readout.weight.zero_()
            readout.bias.fill_(-10)
            readout.bias[action] = 10
            readout = self.model.language.inferior_frontal.readout
            readout.weight.zero_()
            readout.bias.fill_(-10)
            readout.bias[ByteCodec.EOS] = 10

    def modes(self):
        return [(name, module.training) for name, module in self.model.named_modules()]

    def test_history_padding_keeps_recent_screens_in_order_and_does_not_alias(self):
        observations = []
        for step in range(4):
            observations.append({"pixels": torch.full((3, 32, 32), step / 10),
                                 "body": torch.full((4,), step / 10)})
        pixels, body = history_tensors(observations[:1], 3)
        self.assertEqual(tuple(pixels.shape), (3, 3, 32, 32))
        self.assertEqual(tuple(body.shape), (3, 4))
        self.assertTrue(torch.equal(pixels, pixels[:1].expand_as(pixels)))
        pixels, body = history_tensors(observations, 3)
        for index in range(3):
            self.assertTrue(torch.equal(pixels[index], observations[index + 1]["pixels"]))
            self.assertTrue(torch.equal(body[index], observations[index + 1]["body"]))
        batched_pixels, batched_body = tensor_batch([observations, observations[:1]], self.model)
        self.assertEqual(tuple(batched_pixels.shape), (2, 3, 3, 32, 32))
        self.assertEqual(tuple(batched_body.shape), (2, 3, 4))
        batched_pixels.zero_()
        batched_body.zero_()
        self.assertAlmostEqual(float(observations[-1]["pixels"].mean()), 0.3, places=6)
        self.assertAlmostEqual(float(observations[-1]["body"].mean()), 0.3, places=6)

    def test_demonstrations_are_reproducible_visible_histories_with_separate_labels(self):
        config = ComputerTrainConfig(demonstrations=7, batch_size=2, steps=1)
        actions, replies = build_demonstrations(config, history_size=3)
        repeated_actions, repeated_replies = build_demonstrations(config, history_size=3)
        self.assertEqual(len(replies), 7)
        self.assertGreaterEqual(len(actions), 7)
        self.assertEqual({row["task"] for row in replies}, set(TASKS))
        for rows, repeated in ((actions, repeated_actions), (replies, repeated_replies)):
            self.assertEqual(len(rows), len(repeated))
            for row, other in zip(rows, repeated):
                self.assertEqual(set(row), {"pixels", "body", "prompt", "action", "reply", "task", "scene"})
                self.assertEqual(tuple(row["pixels"].shape), (3, 3, 32, 32))
                self.assertEqual(tuple(row["body"].shape), (3, 4))
                self.assertTrue(torch.isfinite(row["pixels"]).all())
                self.assertTrue(((row["pixels"] >= 0) & (row["pixels"] <= 1)).all())
                self.assertTrue(torch.equal(row["pixels"], other["pixels"]))
                self.assertTrue(torch.equal(row["body"], other["body"]))
                for key in ("prompt", "action", "reply", "task", "scene"):
                    self.assertEqual(row[key], other[key])
                self.assertIsInstance(row["prompt"], str)
                self.assertIn(row["action"], range(len(ACTION_NAMES)))
                self.assertEqual(len(row["scene"]), 6)
                self.assertEqual(set(row["scene"][:4]), set(range(4)))
                self.assertIn(row["scene"][4], range(5))
                self.assertIn(row["scene"][5], range(21))
        self.assertTrue(all(row["reply"] is None for row in actions))
        self.assertTrue(all(isinstance(row["reply"], str) and row["reply"] for row in replies))
        self.assertTrue(all(row["action"] == ACTION_NAMES.index("stop") for row in replies))

    def test_scene_annotations_match_visible_button_swatch_and_text_state(self):
        self.assertEqual(len(VISIBLE_TEXTS), 21)
        self.assertEqual(len(set(VISIBLE_TEXTS)), 21)
        self.assertEqual(set(VISIBLE_TEXTS), {"", *"hiok", *(a+b for a in "hiok" for b in "hiok")})
        arranged = tuple(reversed(COLORS))
        self.env.reset(button_colors=arranged, last_clicked=None, text="")
        before = self.env.state_dict()
        annotations = visible_labels(self.env)
        self.assertEqual([COLORS[i] for i in annotations[:4]], list(arranged))
        self.assertEqual(annotations[4], 0)
        self.assertEqual(VISIBLE_TEXTS[annotations[5]], "")
        self.assertEqual(self.env.state_dict(), before)
        for color in COLORS:
            self.env.last_clicked = color
            for text in VISIBLE_TEXTS:
                self.env.text = text
                annotations = visible_labels(self.env)
                self.assertEqual(COLORS[annotations[4]-1], color)
                self.assertEqual(VISIBLE_TEXTS[annotations[5]], text)
        self.assertEqual(set(self.env.observe()), {"pixels", "body"})

    def test_run_turn_uses_only_observations_normalized_prompt_and_generated_prefix(self):
        self.force_action_and_empty_reply(ACTION_NAMES.index("stop"))
        self.env.reset(last_clicked="red", text="hi")
        self.model.train()
        self.model.retina.eval()
        modes = self.modes()
        captured = []
        handle = self.model.register_forward_pre_hook(
            lambda module, args, kwargs: captured.append((args, kwargs)), with_kwargs=True)
        try:
            with patch(f"{TRAINING_MODULE}.oracle_action", side_effect=AssertionError("teacher used")), \
                 patch(f"{TRAINING_MODULE}.goal_success", side_effect=AssertionError("scorer used")), \
                 patch(f"{TRAINING_MODULE}.visible_labels", side_effect=AssertionError("scene labels used")):
                result = run_turn(self.model, self.env, "  HELLO  ")
        finally:
            handle.remove()
        self.assertFalse(result["used_teacher"])
        self.assertEqual(result["prompt"], "hello")
        self.assertEqual(result["reply"], "")
        self.assertEqual(result["actions"], [ACTION_NAMES.index("stop")])
        self.assertTrue(result["terminated"])
        self.assertEqual(len(result["frames"]), 2)
        self.assertEqual(self.env.last_clicked, "red")
        self.assertEqual(self.env.text, "hi")
        self.assertEqual(self.modes(), modes)
        self.assertGreaterEqual(len(captured), 2)
        for args, kwargs in captured:
            self.assertEqual(len(args), 4)
            pixels, body, words, prefix = args
            self.assertEqual(tuple(pixels.shape), (1, 3, 3, 32, 32))
            self.assertEqual(tuple(body.shape), (1, 3, 4))
            self.assertEqual(words, ["hello"])
            self.assertTrue(torch.equal(prefix, torch.tensor([[ByteCodec.BOS]])))
            self.assertEqual(set(kwargs), {"ablate"})
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_nonstopping_policy_is_limited_by_controller_and_environment(self):
        self.force_action_and_empty_reply(0)
        with patch(f"{TRAINING_MODULE}.oracle_action", side_effect=AssertionError("teacher used")):
            for ceiling in (1, 3, 8, 10):
                with self.subTest(ceiling=ceiling):
                    result = run_turn(self.model, self.env, "click red", max_actions=ceiling)
                    self.assertEqual(result["actions"], [0] * ceiling)
                    self.assertEqual(len(result["frames"]), ceiling + 1)
                    self.assertEqual(self.env.steps, ceiling)
                    self.assertEqual(result["terminated"], ceiling == 10)
                    self.assertFalse(self.env.stop_requested)

    def test_blanking_reaches_action_and_response_inputs_without_destroying_scene(self):
        self.force_action_and_empty_reply(ACTION_NAMES.index("stop"))
        captured = []
        handle = self.model.register_forward_pre_hook(
            lambda module, args: captured.append(tuple(args)))
        try:
            result = run_turn(self.model, self.env, "hello", blank_pixels=True, blank_prompt=True,
                              ablate=("hippocampus",))
        finally:
            handle.remove()
        self.assertGreaterEqual(len(captured), 2)
        for pixels, body, words, prefix in captured:
            self.assertEqual(int(torch.count_nonzero(pixels)), 0)
            self.assertEqual(words, [""])
            self.assertGreater(int(torch.count_nonzero(body)), 0)
        self.assertEqual(result["prompt"], "hello")
        self.assertGreater(float(self.env.observe()["pixels"].mean()), 0)

    def test_controller_validates_prompt_and_action_budget_before_touching_scene(self):
        self.env.step(0)
        before = self.env.state_dict()
        for invalid in (0, 11, -1, True, 1.5, "2"):
            with self.subTest(max_actions=invalid), self.assertRaises(ValueError):
                run_turn(self.model, self.env, "hello", max_actions=invalid)
            self.assertEqual(self.env.state_dict(), before)
        for prompt in ("", "   ", None, 5, "x" * 129):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                run_turn(self.model, self.env, prompt)
            self.assertEqual(self.env.state_dict(), before)

    def test_evaluation_never_invokes_teacher_and_scores_free_responses(self):
        self.force_action_and_empty_reply(ACTION_NAMES.index("stop"))
        self.model.train()
        self.model.retina.eval()
        modes = self.modes()
        with patch(f"{TRAINING_MODULE}.oracle_action", side_effect=AssertionError("teacher used")), \
             patch(f"{TRAINING_MODULE}.visible_labels", side_effect=AssertionError("scene labels used")):
            report = evaluate_computer(self.model, episodes_per_task=2, batch_size=1,
                                       split="test", shift=True)
        self.assertEqual(self.modes(), modes)
        self.assertFalse(report["teacher_used_for_policy"])
        self.assertTrue(report["free_running_replies"])
        self.assertEqual(report["prompt_split"], "test")
        self.assertTrue(report["layout_shift"])
        self.assertEqual(set(report["tasks"]), set(TASKS))
        self.assertEqual(report["macro_reply_exact"], 0)
        self.assertEqual(report["macro_joint_success"], 0)
        for task, metrics in report["tasks"].items():
            self.assertEqual(metrics["episodes"], 2)
            self.assertEqual(metrics["reply_exact_count"], 0)
            self.assertEqual(metrics["joint_success_count"], 0)
            if task in ("describe_position", "recall", "greet", "clarify"):
                self.assertEqual(metrics["action_success_count"], 2)
        for failure in report["failures"]:
            self.assertEqual(failure["reply"], "")
            self.assertEqual(failure["actions"], ["stop"])
        json.dumps(report, allow_nan=False)

    def test_evaluation_timeout_never_counts_as_success(self):
        self.force_action_and_empty_reply(0)
        with patch(f"{TRAINING_MODULE}.oracle_action", side_effect=AssertionError("teacher used")):
            report = evaluate_computer(self.model, episodes_per_task=1, batch_size=1)
        self.assertEqual(report["macro_action_success"], 0)
        self.assertEqual(len(report["failures"]), len(TASKS))
        for failure in report["failures"]:
            self.assertEqual(failure["actions"], ["click_top_left"] * 8)

    def test_inference_restores_submodule_modes_after_failure(self):
        self.model.train()
        self.model.retina.eval()
        self.model.language.posterior_temporal.eval()
        modes = self.modes()
        with self.assertRaises(ValueError):
            run_turn(self.model, self.env, "hello", ablate=("not_a_region",))
        self.assertEqual(self.modes(), modes)
        with self.assertRaises(ValueError):
            evaluate_computer(self.model, episodes_per_task=1, ablate=("not_a_region",))
        self.assertEqual(self.modes(), modes)

    def test_real_tiny_optimizer_run_changes_weights_and_checkpoint_reloads(self):
        # Closed-loop evaluation has its own real-controller tests above. Keep
        # this optimization test small by replacing only the large evaluation runs.
        evaluation = {"macro_action_success": 0.0, "macro_reply_exact": 0.0,
                      "macro_joint_success": 0.0}
        for history_size in (1, 3):
            with self.subTest(history_size=history_size), tempfile.TemporaryDirectory() as directory:
                model_config = ComputerConfig(hidden_size=8, language_hidden_size=12,
                                              embedding_size=6, visual_features=8, max_history=history_size)
                config = ComputerTrainConfig(seed=27, demonstrations=7, batch_size=4, steps=2)
                torch.manual_seed(config.seed)
                initial = ComputerBrain(model_config)
                initial_state = {name: tensor.clone() for name, tensor in initial.state_dict().items()}
                with patch(f"{TRAINING_MODULE}.evaluate_computer", return_value=evaluation), \
                     redirect_stdout(io.StringIO()):
                    trained, report = train_computer(config, model_config, output=directory, log_every=1)
                for prefix in ("retina.", "scene_readout.", "language.posterior_temporal.", "language.inferior_frontal.",
                               "language.brain.regions.prefrontal_cortex.", "language.brain.regions.motor_cortex."):
                    self.assertTrue(any(not torch.equal(initial_state[name], tensor)
                                        for name, tensor in trained.state_dict().items() if name.startswith(prefix)), prefix)
                self.assertEqual(len(report["history"]), 2)
                self.assertTrue(all(torch.isfinite(torch.tensor(row["loss"])) for row in report["history"]))
                self.assertTrue(all(torch.isfinite(torch.tensor(row["scene_loss"])) for row in report["history"]))
                self.assertTrue(report["parameters_with_nonzero_gradients_observed"])
                self.assertTrue(report["english_training_performed"])
                self.assertFalse(report["real_os_access"])
                checkpoint = Path(directory) / "checkpoint.pt"
                self.assertTrue(checkpoint.is_file())
                self.assertFalse(checkpoint.with_name("checkpoint.pt.tmp").exists())
                loaded = load_computer_checkpoint(checkpoint)
                self.assertFalse(loaded.training)
                self.assertEqual(loaded.config, trained.config)
                for name, tensor in trained.state_dict().items():
                    self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]), name)
                pixels, body = tensor_batch([[self.env.observe()]], loaded)
                prefix = torch.tensor([[ByteCodec.BOS]])
                trained.eval()
                with torch.no_grad():
                    expected = trained(pixels, body, ["hello"], prefix)
                    actual = loaded(pixels, body, ["hello"], prefix)
                for key in ("logits", "language_logits", "prediction", "scene_logits"):
                    self.assertTrue(torch.equal(expected[key], actual[key]), key)
                saved = torch.load(checkpoint, weights_only=True)
                self.assertEqual(saved["step"], 2)
                self.assertEqual(saved["schema"], "bic-computer-use-v2")
                self.assertTrue(saved["optimizer"]["state"])
                self.assertEqual(saved["sample_rng"].dtype, torch.uint8)
                self.assertEqual(json.loads((Path(directory) / "report.json").read_text())["history"], report["history"])

    def test_checkpoint_schema_and_training_integer_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pt"
            torch.save({"schema_version": 1, "model": {}}, path)
            with self.assertRaises(ValueError):
                load_computer_checkpoint(path)
        for field in ("seed", "demonstrations", "batch_size", "steps"):
            for value in (True, 1.5, "2"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    ComputerTrainConfig(**{field: value})
        for arguments in ({"batch_size": 1}, {"batch_size": 0}, {"steps": 0},
                          {"demonstrations": 0}, {"learning_rate": 0},
                          {"learning_rate": float("nan")}, {"learning_rate": float("inf")}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ComputerTrainConfig(**arguments)


if __name__ == "__main__":
    unittest.main()
