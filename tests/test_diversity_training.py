"""CPU-only training admission, fixed-objective and exact-resume checks."""
import copy
import io
import unittest
from unittest.mock import patch

import torch

from experiments.diverse_curriculum import generate_diverse, validate_diverse, validate_pair
from experiments.diversity_training import DiversityTrainer
from experiments.sequence_data import pack_cognitive_episodes
from experiments.sequence_student import SequenceConfig
from experiments.sequence_training import SequenceTrainer, sequence_objective


class DiversityTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def config(self, **overrides):
        options = dict(width=16, layers=1, heads=2, feedforward=32)
        options.update(overrides)
        return SequenceConfig(**options)

    def banks(self, seed=34_000_000, split="train", count=4):
        return {family: generate_diverse(seed + index * 10_000, count, split=split,
                                        family=family, level=2)
                for index, family in enumerate(("variable_binding", "graph_reachability"))}

    def trainer(self, banks=None, **options):
        args = dict(seed=77, config=self.config(), batch_size=2)
        args.update(options)
        return DiversityTrainer(self.banks() if banks is None else banks, **args)

    def compare(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.compare(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.compare(a, b)
        else:
            self.assertEqual(left, right)

    def test_authentication_precedes_structural_packing_and_rejects_heldout_or_corruption(self):
        with patch("experiments.diversity_training.pack_cognitive_episodes", wraps=pack_cognitive_episodes) as pack:
            trainer = self.trainer()
            self.assertEqual(pack.call_count, 2)
            self.assertTrue(all(call.kwargs["training"] is False for call in pack.call_args_list))
            self.assertEqual(set(trainer.encoded), {"variable_binding", "graph_reachability"})
        for kind in ("heldout", "label", "text", "numeric", "duplicate_variant", "cross_pair", "wrong_family"):
            banks = self.banks(split="dev" if kind == "heldout" else "train")
            rows = banks["variable_binding"]
            if kind == "label":
                rows[0]["turns"][-1]["target"] = (rows[0]["turns"][-1]["target"] + 1) % 4
            elif kind == "text":
                rows[0]["turns"][0]["text"] += " Answer: yes."
            elif kind == "numeric":
                rows[0]["turns"][0]["observations"]["visual"][0][0] = 1.
            elif kind == "duplicate_variant":
                rows[1] = copy.deepcopy(rows[0])
            elif kind == "cross_pair":
                rows[1], rows[2] = rows[2], rows[1]
            elif kind == "wrong_family":
                banks["variable_binding"] = copy.deepcopy(banks["graph_reachability"])
            with self.subTest(kind=kind), patch("experiments.diversity_training.pack_cognitive_episodes") as pack:
                with self.assertRaises(ValueError):
                    self.trainer(banks)
                pack.assert_not_called()

    def test_complete_pair_draws_same_sampler_objective_and_encoder_input_boundary(self):
        trainer = self.trainer(batch_size=6)
        family = "variable_binding"
        expected_generator = torch.Generator()
        expected_generator.set_state(trainer.generators[family].get_state())
        pairs = torch.randint(2, (3,), generator=expected_generator)
        indices = (pairs[:, None] * 2 + torch.arange(2)[None]).flatten()
        expected = trainer.encoded[family]
        width = int(expected["inputs"]["lengths"].index_select(0, indices).max())
        untouched = trainer.generators["graph_reachability"].get_state().clone()
        with patch.object(trainer.model, "forward", wraps=trainer.model.forward) as forward, \
                patch("experiments.sequence_training.sequence_objective", wraps=sequence_objective) as objective:
            metrics = trainer.step(family)
        self.assertIs(DiversityTrainer.step, SequenceTrainer.step)
        self.assertEqual(objective.call_count, 1)
        arguments = forward.call_args.kwargs
        self.assertEqual(set(arguments), {"token_ids", "valid_mask", "lengths", "eos_positions", "decoder_input_ids"})
        for name, value in expected["inputs"].items():
            selected = value.index_select(0, indices)
            if name in ("token_ids", "valid_mask"):
                selected = selected[:, :width]
            self.compare(arguments[name], selected)
        self.compare(arguments["decoder_input_ids"],
                     expected["supervision"]["reply_decoder_input_ids"].index_select(0, indices))
        self.assertTrue(torch.equal(untouched, trainer.generators["graph_reachability"].get_state()))
        self.assertTrue(torch.equal(expected_generator.get_state(), trainer.generators[family].get_state()))
        self.assertEqual(trainer.updates, 1)
        self.assertEqual(trainer.family_updates[family], 1)
        self.assertGreater(metrics["observation_tokens"], 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(metrics[key])) for key in
                            ("loss", "action_loss", "reply_loss", "observation_language_loss")))
        # Reply teacher forcing is a separate argument and cannot modify action
        # or observation predictions, even when a different reply is supplied.
        with torch.no_grad():
            first = trainer.model(**arguments)
            altered = dict(arguments)
            altered["decoder_input_ids"] = arguments["decoder_input_ids"][..., :1]
            second = trainer.model(**altered)
        for name in ("logits", "observation_language_logits", "context_states"):
            self.assertTrue(torch.equal(first[name], second[name]), name)

    def test_tensor_only_checkpoint_exact_next_update_and_independent_snapshots(self):
        trainer = self.trainer()
        trainer.step("variable_binding")
        saved = trainer.snapshot()
        stream = io.BytesIO()
        torch.save(saved, stream)
        stream.seek(0)
        loaded = torch.load(stream, weights_only=True)
        resumed = self.trainer(payload=loaded)
        metrics = trainer.step("graph_reachability")
        actual = resumed.step("graph_reachability")
        self.compare(metrics, actual)
        self.compare(trainer.snapshot(), resumed.snapshot())
        trainer.step("variable_binding")
        resumed.step("variable_binding")
        self.compare(trainer.snapshot(), resumed.snapshot())
        saved["recipe"]["batch_size"] = 100
        first_weight = next(iter(saved["weights"]))
        saved["weights"][first_weight].zero_()
        saved["samplers"]["variable_binding"].zero_()
        self.assertEqual(trainer.recipe["batch_size"], 2)
        self.compare(trainer.snapshot(), resumed.snapshot())

    def test_caller_bank_mutation_cannot_change_admitted_training_or_fingerprint(self):
        banks = self.banks()
        trainer = self.trainer(banks)
        reference = self.trainer(copy.deepcopy(banks))
        banks["variable_binding"][0]["turns"][0]["text"] = "not an admitted lesson"
        banks["variable_binding"].clear()
        self.compare(trainer.step("variable_binding"), reference.step("variable_binding"))
        self.compare(trainer.snapshot(), reference.snapshot())

    def test_repeated_logical_pair_slots_preserve_multiplicity_and_authenticate_unique_content(self):
        original = self.banks(count=2)
        repeated = {family: rows * 3 + copy.deepcopy(rows) for family, rows in original.items()}
        with patch("experiments.diverse_curriculum.validate_diverse", wraps=validate_diverse) as row_validator, \
                patch("experiments.diversity_training.validate_pair", wraps=validate_pair) as pair_validator:
            trainer = self.trainer(repeated)
        self.assertEqual(row_validator.call_count, 4)
        self.assertEqual(pair_validator.call_count, 2)
        self.assertTrue(all(len(rows) == 8 for rows in trainer.banks.values()))
        self.assertTrue(all(batch["inputs"]["token_ids"].shape[0] == 8 for batch in trainer.encoded.values()))
        self.assertIs(trainer.banks["variable_binding"][0], trainer.banks["variable_binding"][2])
        self.assertIsNot(trainer.banks["variable_binding"][0], repeated["variable_binding"][0])
        smaller = self.trainer(original)
        self.assertNotEqual(trainer.recipe["banks"], smaller.recipe["banks"])
        with self.assertRaisesRegex(ValueError, "recipe"):
            self.trainer(original, payload=trainer.snapshot())

    def test_resume_rejects_changed_recipe_banks_counters_sampler_and_optimizer(self):
        trainer = self.trainer()
        trainer.step("variable_binding")
        payload = trainer.snapshot()
        cases = ({"seed": 78}, {"learning_rate": .003}, {"batch_size": 4},
                 {"config": self.config(feedforward=64)}, {"banks": self.banks(seed=35_000_000)})
        for options in cases:
            with self.subTest(options=options.keys()), self.assertRaisesRegex(ValueError, "recipe"):
                self.trainer(payload=payload, **options)
        for kind in ("count", "negative_count", "sampler_dtype", "optimizer_nan", "weight_nan",
                     "optimizer_lr", "moment_shape", "optimizer_step"):
            bad = copy.deepcopy(payload)
            if kind == "count":
                bad["updates"] += 1
            elif kind == "negative_count":
                bad["family_updates"]["graph_reachability"] = -1
            elif kind == "sampler_dtype":
                bad["samplers"]["variable_binding"] = bad["samplers"]["variable_binding"].float()
            elif kind == "weight_nan":
                next(iter(bad["weights"].values())).flatten()[0] = float("nan")
            elif kind == "optimizer_lr":
                bad["optimizer"]["param_groups"][0]["lr"] = .003
            else:
                state = next(iter(bad["optimizer"]["state"].values()))
                if kind == "optimizer_nan":
                    state["exp_avg"].flatten()[0] = float("nan")
                elif kind == "moment_shape":
                    state["exp_avg"] = torch.zeros(1)
                else:
                    state["step"].add_(1)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.trainer(payload=bad)

    def test_invalid_configuration_unequal_banks_and_unknown_family_rejected(self):
        for options in ({"batch_size": 3}, {"batch_size": True}, {"learning_rate": float("inf")},
                        {"learning_rate": True}, {"seed": -1}, {"config": self.config(max_turns=5)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.trainer(**options)
        banks = self.banks()
        banks["variable_binding"] = banks["variable_binding"][:2]
        with self.assertRaisesRegex(ValueError, "equal sizes"):
            self.trainer(banks)
        trainer = self.trainer()
        with self.assertRaisesRegex(ValueError, "admitted"):
            trainer.step("unknown")
        restored = self.trainer(payload=trainer.snapshot())
        self.compare(trainer.step("variable_binding"), restored.step("variable_binding"))


if __name__ == "__main__":
    unittest.main()
