"""Matched family draws, real variable-turn objectives and exact CPU continuation."""
import copy
import io
import unittest
from unittest.mock import patch

import torch

from experiments.composition_curriculum import FAMILIES, generate_pair
from experiments.composition_training import CompositionTrainer
from experiments.sequence_student import SequenceConfig
from experiments.sequence_training import sequence_objective


class CompositionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.banks = {family: {turns: sum((generate_pair(family, 710000 + index * 1000 + turns * 10 + pair,
            turns=turns) for pair in range(2)), []) for turns in (8, 10, 12)}
            for index, family in enumerate(sorted(FAMILIES))}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def trainer(self, banks=None, **options):
        config = SequenceConfig(width=16, layers=1, heads=2, feedforward=32, max_turns=12)
        defaults = {"seed": 2801, "sampler_seed": 3801, "micro_batch_size": 2, "config": config}
        defaults.update(options)
        return CompositionTrainer(self.banks if banks is None else banks, **defaults)

    def compare(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for name in left:
                self.compare(left[name], right[name])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.compare(a, b)
        else:
            self.assertEqual(left, right)

    def test_joint_sequential_and_subset_preserve_ordered_family_samples(self):
        names = sorted(self.banks)
        joint, sequential = self.trainer(), self.trainer()
        held = names[-1]  # A nonzero global offset catches subset reindexing.
        fresh = self.trainer({held: self.banks[held]})
        for name in joint.model.state_dict():
            self.compare(joint.model.state_dict()[name], fresh.model.state_dict()[name])
        joint_draws = {name: [] for name in names}
        for _ in range(3):
            for report in joint.step(tuple(names))["microbatches"]:
                joint_draws[report["family"]].append((report["turns"], report["pair_indices"], report["exposures"]))
        for name in names:
            reports = sequential.step((name,) * 3)["microbatches"]
            self.assertEqual(joint_draws[name], [(row["turns"], row["pair_indices"], row["exposures"]) for row in reports])
            self.compare(joint.generators[name].get_state(), sequential.generators[name].get_state())
            self.assertEqual(joint.exposures[name], sequential.exposures[name])
        reports = fresh.step((held,) * 3)["microbatches"]
        self.assertEqual(joint_draws[held], [(row["turns"], row["pair_indices"], row["exposures"]) for row in reports])
        self.assertEqual(joint.exposures[held], fresh.exposures[held])

    def test_three_unchanged_losses_accumulate_then_clip_and_update_once(self):
        trainer, reference = self.trainer(), self.trainer()
        families = tuple(sorted(self.banks))
        reference.optimizer.zero_grad(set_to_none=True)
        expected_losses, expected_reports = [], []
        for family in families:
            batch, turns, pairs, counts = reference._draw(family)
            output = reference.model(**batch["inputs"], decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
            losses = sequence_objective(output, batch)
            expected_losses.append(float(losses["loss"].detach()))
            (losses["loss"] / 3).backward()
            expected_reports.append((turns, pairs, counts))
            self.assertEqual(batch["inputs"]["token_ids"].shape[0], 2)
            self.assertEqual(counts["observation_bytes"], int(batch["inputs"]["token_ids"].ge(3).sum()))
            self.assertEqual(counts["turns"], 2 * turns)
        expected_gradients = [parameter.grad.clone() for parameter in reference.model.parameters()]
        self.assertGreater(float(reference.model.tokens.weight.grad.abs().sum()), 0.)
        original_clip = torch.nn.utils.clip_grad_norm_
        def capture(parameters, *args, **kwargs):
            parameters = list(parameters)
            for parameter, gradient in zip(parameters, expected_gradients):
                torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
            return original_clip(parameters, *args, **kwargs)
        with patch("torch.nn.utils.clip_grad_norm_", side_effect=capture) as clipping, \
                patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as update:
            actual = trainer.step(families)
        self.assertEqual(clipping.call_count, 1)
        self.assertEqual(update.call_count, 1)
        self.assertEqual(actual["loss"], sum(expected_losses) / 3)
        for report, expected in zip(actual["microbatches"], expected_reports):
            self.assertEqual((report["turns"], report["pair_indices"], report["exposures"]), expected)
        original_clip(reference.model.parameters(), 1., error_if_nonfinite=True)
        reference.optimizer.step()
        self.compare(trainer.model.state_dict(), reference.model.state_dict())
        self.compare(trainer.optimizer.state_dict(), reference.optimizer.state_dict())

    def test_exact_resume_weights_optimizer_samplers_and_variable_turn_counts(self):
        trainer = self.trainer()
        names = tuple(sorted(self.banks))
        trainer.step(names)
        saved = trainer.snapshot()
        stream = io.BytesIO()
        torch.save(saved, stream)
        stream.seek(0)
        resumed = self.trainer(payload=torch.load(stream, weights_only=True))
        for order in ((names[-1], names[-1], names[0]), names):
            self.compare(trainer.step(order), resumed.step(order))
            self.compare(trainer.snapshot(), resumed.snapshot())
        saved["recipe"]["sampler_seed"] = 0
        saved["exposures"][names[0]]["turns"] = 0
        saved["samplers"][names[0]].zero_()
        self.compare(trainer.snapshot(), resumed.snapshot())
        fresh = self.trainer()
        restarted = self.trainer(payload=fresh.snapshot())
        self.compare(fresh.step(names), restarted.step(names))

    def test_caller_mutation_and_old_gradients_cannot_change_admitted_update(self):
        banks = copy.deepcopy(self.banks)
        trainer, reference = self.trainer(banks), self.trainer()
        names = tuple(sorted(banks))
        banks[names[0]][8][0]["turns"][0]["text"] = "Changed caller-owned text."
        banks[names[-1]].clear()
        for parameter in trainer.model.parameters():
            parameter.grad = torch.full_like(parameter, 100.)
        self.compare(trainer.step(names), reference.step(names))
        self.compare(trainer.snapshot(), reference.snapshot())

    def test_invalid_admission_recipe_optimizer_and_counter_rejected(self):
        names = tuple(sorted(self.banks))
        for change in ("split", "structure_partition", "label", "duplicate", "turn_bucket"):
            bad = copy.deepcopy(self.banks)
            rows = bad[names[0]][8]
            if change in ("split", "structure_partition"):
                rows[0][change] = "audit"
            elif change == "label":
                rows[0]["turns"][-1]["target"] = 3
            elif change == "duplicate":
                rows[1] = copy.deepcopy(rows[0])
            else:
                bad[names[0]][10] = bad[names[0]].pop(8)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.trainer(bad)
        trainer = self.trainer()
        for order in (names[:2], "color", ("unknown",) * 3):
            with self.assertRaises(ValueError):
                trainer.step(order)
        trainer.step(names)
        payload = trainer.snapshot()
        for options in ({"sampler_seed": 3802}, {"micro_batch_size": 4}, {"learning_rate": .002}):
            with self.assertRaises(ValueError):
                self.trainer(payload=payload, **options)
        for change in ("microbatches", "bucket", "bytes", "sampler", "optimizer_nan", "optimizer_step"):
            bad = copy.deepcopy(payload)
            if change == "microbatches":
                bad["family_microbatches"][names[0]] += 1
            elif change == "bucket":
                bad["bucket_microbatches"][names[0]][8] += 1
            elif change == "bytes":
                bad["exposures"][names[0]]["observation_bytes"] += 1
            elif change == "sampler":
                bad["samplers"][names[0]] = torch.zeros(2)
            elif change == "optimizer_nan":
                next(iter(bad["optimizer"]["state"].values()))["exp_avg"].flatten()[0] = float("nan")
            else:
                next(iter(bad["optimizer"]["state"].values()))["step"].add_(1)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.trainer(payload=bad)
        for options in ({"micro_batch_size": 3}, {"sampler_seed": True}, {"learning_rate": float("nan")}):
            with self.assertRaises(ValueError):
                self.trainer(**options)


if __name__ == "__main__":
    unittest.main()
