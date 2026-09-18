"""CPU replay objectives, exposure boundaries and deterministic continuation."""
import copy
import io
import unittest
from unittest.mock import patch

import torch

from experiments.consolidation_training import ConsolidationTrainer, _draw
from experiments.diverse_curriculum import generate_diverse
from experiments.diversity_training import DiversityTrainer
from experiments.sequence_student import SequenceConfig
from experiments.sequence_training import sequence_objective


class ConsolidationTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def config(self):
        return SequenceConfig(width=16, layers=1, heads=2, feedforward=32)

    def banks(self):
        support = {"arithmetic_updates": generate_diverse(58_000_000, 4, family="arithmetic_updates")}
        replay = {family: generate_diverse(59_000_000 + index * 1000, 4, family=family)
                  for index, family in enumerate(("variable_binding", "graph_reachability"))}
        return support, replay

    def trainer(self, *, replay=True, support_banks=None, replay_banks=None, **overrides):
        support, old = self.banks()
        options = dict(seed=3701, batch_size=2, replay_batch_size=2, config=self.config())
        options.update(overrides)
        return ConsolidationTrainer(support if support_banks is None else support_banks,
            replay_banks=(old if replay_banks is None else replay_banks) if replay else None, **options)

    def compare(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.compare(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.compare(a, b)
        else:
            self.assertEqual(left, right)

    def test_no_replay_exactly_matches_frozen_trainer_updates_and_sampler(self):
        support, _ = self.banks()
        old = DiversityTrainer(support, seed=3701, batch_size=2, config=self.config())
        new = self.trainer(replay=False)
        for _ in range(2):
            expected = old.step("arithmetic_updates")
            actual = new.step("arithmetic_updates")
            self.assertEqual(actual["loss"], expected["loss"])
            for name in ("loss", "action_loss", "reply_loss", "observation_language_loss"):
                self.assertEqual(actual["support"][name], expected[name])
            for name in ("weights", "optimizer", "samplers", "updates", "family_updates"):
                self.compare(old.snapshot()[name], new.snapshot()[name])
            self.assertIsNone(actual["replay"])
        self.assertEqual(new.exposures["support_episodes"], 4)
        self.assertTrue(all(value == 0 for key, value in new.exposures.items() if key.startswith("replay_")))

    def test_replay_keeps_support_draws_identical_and_rotates_only_old_families(self):
        plain, replay = self.trainer(replay=False), self.trainer()
        order = []
        for _ in range(3):
            with patch.object(plain.model, "forward", wraps=plain.model.forward) as a, \
                    patch.object(replay.model, "forward", wraps=replay.model.forward) as b:
                left = plain.step("arithmetic_updates")
                right = replay.step("arithmetic_updates")
            self.assertEqual(a.call_count, 1)
            self.assertEqual(b.call_count, 2)
            self.compare(a.call_args.kwargs, b.call_args_list[0].kwargs)
            self.compare(plain.generators["arithmetic_updates"].get_state(), replay.generators["arithmetic_updates"].get_state())
            self.assertEqual(left["exposures"]["support_observation_bytes"], right["exposures"]["support_observation_bytes"])
            order.append(right["replay_family"])
        self.assertEqual(order, ["graph_reachability", "variable_binding", "graph_reachability"])
        self.assertEqual(replay.replay_family_updates, {"variable_binding": 1, "graph_reachability": 2})
        self.assertEqual(replay.exposures["support_episodes"], 6)
        self.assertEqual(replay.exposures["replay_episodes"], 6)

    def test_combined_loss_shared_gradients_single_clip_and_exact_byte_counts(self):
        trainer = self.trainer()
        def draw(family, old=False):
            generator = torch.Generator()
            generators = trainer.replay_generators if old else trainer.generators
            generator.set_state(generators[family].get_state())
            return _draw((trainer.replay_encoded if old else trainer.encoded)[family],
                         (trainer.replay_banks if old else trainer.banks)[family], generator, 2, "cpu")
        new, old = draw("arithmetic_updates"), draw("graph_reachability", True)
        reference = copy.deepcopy(trainer.model)
        def objective(batch):
            return sequence_objective(reference(**batch["inputs"],
                decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"]), batch)["loss"]
        new_loss = objective(new)
        new_loss.backward()
        new_gradients = [parameter.grad.clone() for parameter in reference.parameters()]
        reference.zero_grad(set_to_none=True)
        old_loss = objective(old)
        old_loss.backward()
        old_gradients = [parameter.grad.clone() for parameter in reference.parameters()]
        self.assertGreater(float(reference.tokens.weight.grad.abs().sum()), 0.)
        expected = [a + .5 * b for a, b in zip(new_gradients, old_gradients)]
        original_clip = torch.nn.utils.clip_grad_norm_
        def capture(parameters, *args, **kwargs):
            parameters = list(parameters)
            for parameter, gradient in zip(parameters, expected):
                torch.testing.assert_close(parameter.grad, gradient, rtol=2e-5, atol=2e-6)
            return original_clip(parameters, *args, **kwargs)
        with patch("torch.nn.utils.clip_grad_norm_", side_effect=capture) as clipping, \
                patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as update:
            result = trainer.step("arithmetic_updates")
        self.assertEqual(clipping.call_count, 1)
        self.assertEqual(update.call_count, 1)
        self.assertAlmostEqual(result["loss"], float(new_loss.detach() + .5 * old_loss.detach()), places=6)
        self.assertEqual(result["support"]["loss"], float(new_loss.detach()))
        self.assertEqual(result["replay"]["loss"], float(old_loss.detach()))
        for source, batch in (("support", new), ("replay", old)):
            counts = result["exposures"]
            self.assertEqual(counts[f"{source}_observation_bytes"], int(batch["inputs"]["token_ids"].ge(3).sum()))
            self.assertEqual(counts[f"{source}_observation_tokens"] - counts[f"{source}_observation_bytes"], 24)
            self.assertEqual(counts[f"{source}_reply_target_tokens"] - counts[f"{source}_reply_target_bytes"], 12)

    def test_exact_tensor_only_resume_includes_round_robin_optimizer_samplers_and_exposures(self):
        trainer = self.trainer()
        trainer.step("arithmetic_updates")
        saved = trainer.snapshot()
        stream = io.BytesIO()
        torch.save(saved, stream)
        stream.seek(0)
        resumed = self.trainer(payload=torch.load(stream, weights_only=True))
        for _ in range(2):
            self.compare(trainer.step("arithmetic_updates"), resumed.step("arithmetic_updates"))
            self.compare(trainer.snapshot(), resumed.snapshot())
        saved["exposures"]["support_episodes"] = 999
        saved["recipe"]["replay_weight"] = 9.
        saved["replay_samplers"]["graph_reachability"].zero_()
        self.compare(trainer.snapshot(), resumed.snapshot())
        plain = self.trainer(replay=False)
        restored = self.trainer(replay=False, payload=plain.snapshot())
        self.compare(plain.step("arithmetic_updates"), restored.step("arithmetic_updates"))

    def test_caller_banks_are_immutable_and_old_optimizer_gradients_are_cleared(self):
        support, replay = self.banks()
        trainer = self.trainer(support_banks=support, replay_banks=replay)
        reference = self.trainer()
        support["arithmetic_updates"][0]["turns"][0]["text"] = "changed"
        replay["variable_binding"].clear()
        for parameter in trainer.model.parameters():
            parameter.grad = torch.full_like(parameter, 100.)
        self.compare(trainer.step("arithmetic_updates"), reference.step("arithmetic_updates"))
        self.compare(trainer.snapshot(), reference.snapshot())

    def test_zero_weight_replay_still_reports_extra_forward_and_byte_exposure(self):
        trainer = self.trainer(replay_weight=0.)
        plain = self.trainer(replay=False)
        with patch.object(trainer.model, "forward", wraps=trainer.model.forward) as forward:
            actual = trainer.step("arithmetic_updates")
        expected = plain.step("arithmetic_updates")
        self.assertEqual(forward.call_count, 2)
        self.assertEqual(actual["loss"], actual["support"]["loss"])
        self.assertEqual(actual["loss"], expected["loss"])
        self.assertGreater(actual["exposures"]["replay_observation_bytes"], 0)
        self.assertEqual(actual["exposures"]["replay_episodes"], 2)
        for name in ("weights", "optimizer", "samplers"):
            self.compare(trainer.snapshot()[name], plain.snapshot()[name])

    def test_invalid_banks_recipe_counts_and_nonfinite_optimizer_rejected(self):
        support, replay = self.banks()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.trainer(replay_banks=support)
        for kind in ("heldout", "wrong_label", "duplicate_variant"):
            bad = copy.deepcopy(replay)
            if kind == "heldout":
                bad["variable_binding"] = generate_diverse(60_000_000, 4, split="dev", family="variable_binding")
            elif kind == "wrong_label":
                bad["variable_binding"][0]["turns"][-1]["target"] = 3
            else:
                bad["variable_binding"][1] = copy.deepcopy(bad["variable_binding"][0])
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.trainer(replay_banks=bad)
        trainer = self.trainer()
        trainer.step("arithmetic_updates")
        payload = trainer.snapshot()
        for options in ({"replay_weight": .2}, {"replay_batch_size": 4}, {"seed": 3702}, {"replay": False}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.trainer(payload=payload, **options)
        for kind in ("round_robin", "exposure", "sampler", "nan"):
            bad = copy.deepcopy(payload)
            if kind == "round_robin":
                bad["replay_family_updates"]["variable_binding"] += 1
            elif kind == "exposure":
                bad["exposures"]["support_observation_bytes"] += 1
            elif kind == "sampler":
                bad["replay_samplers"]["variable_binding"] = torch.zeros(2)
            else:
                next(iter(bad["optimizer"]["state"].values()))["exp_avg"].flatten()[0] = float("nan")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.trainer(payload=bad)
        for options in ({"replay_weight": -1}, {"replay_weight": float("nan")}, {"replay_batch_size": 3}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.trainer(**options)


if __name__ == "__main__":
    unittest.main()
