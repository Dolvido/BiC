"""One fresh tiny CPU fixture; no checkpoint/history reads or optimizer work."""
import copy
import json
import math
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments.composition_data import pack_composition_episodes
from experiments.foundation_curriculum import FAMILIES, generate_pair, validate_pair
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.sequence_training import sequence_objective
from experiments import foundation_gradient_diagnostic as diagnostic


def setUpModule():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


class GradientDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started = time.monotonic()
        cls.reports = []
        cls.direct_forwards = 0
        cls.direct_episodes = 0
        cls.direct_gradients = 0
        cls.direct_decoder_calls = 0
        cls.direct_decoder_row_steps = 0
        cls.model = build_sequence_student(831907799, device="cpu", config=SequenceConfig(
            width=8, layers=1, heads=2, feedforward=16, max_turns=8, max_positions=1024))
        for parameter in cls.model.parameters():
            parameter.requires_grad_(False)
        first = next(cls.model.parameters())
        first.grad = torch.ones_like(first)
        cls.model.train()
        cls.model.blocks[0].eval()
        cls.modes = [module.training for module in cls.model.modules()]
        cls.flags = [parameter.requires_grad for parameter in cls.model.parameters()]
        cls.grad = first.grad.clone()
        cls.weights = checkpoint_digest(cls.model)
        cls.inventory = tuple(sorted(family + "/d0/direct/t8" for family in FAMILIES))
        cls.banks = {}
        cls.seeds = []
        for index, family in enumerate(FAMILIES):
            rows = []
            for offset in (0, 10):
                seed = 831907701 + index + offset
                cls.seeds.append(seed)
                rows.extend(generate_pair(family, seed, depth=0, turns=8, split="train"))
            cls.banks[family + "/d0/direct/t8"] = rows
        cls.original_banks = copy.deepcopy(cls.banks)

    @classmethod
    def tearDownClass(cls):
        forwards = cls.direct_forwards + sum(r["family_forwards_attempted"] for r in cls.reports)
        episodes = cls.direct_episodes + sum(r["encoder_episodes_attempted"] for r in cls.reports)
        gradients = cls.direct_gradients + sum(r["component_gradient_evaluations_attempted"] for r in cls.reports)
        value = dict(schema="bic-gradient-diagnostic-cpu-proof-v1", model_constructions=1,
            model_seed=831907799, fixture_seeds=cls.seeds, configuration=vars(cls.model.config),
            reports=cls.reports, direct_family_forwards=cls.direct_forwards, direct_episode_forwards=cls.direct_episodes,
            direct_gradient_evaluations=cls.direct_gradients, direct_decoder_calls=cls.direct_decoder_calls,
            direct_decoder_row_steps=cls.direct_decoder_row_steps,
            total_family_forward_attempts=forwards, total_episode_forward_attempts=episodes,
            total_gradient_attempts=gradients, family_forward_bound=8, episode_forward_bound=16, gradient_bound=11,
            optimizer_calls=0, optimizer_updates=0, checkpoint_loads=0, historical_or_formal_inputs=False,
            wall_seconds=time.monotonic()-cls.started, within_approved_bounds=forwards <= 8 and episodes <= 16 and gradients <= 11)
        print("GRADIENT_DIAGNOSTIC_CPU_WORK=" + json.dumps(value, sort_keys=True))
        path = os.environ.get("BIC_GRADIENT_DIAGNOSTIC_ACCOUNTING")
        if path:
            with Path(path).open("x", encoding="utf8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True)
        if not value["within_approved_bounds"]:
            raise AssertionError("gradient fixture exceeded its authorized bounds")

    def tearDown(self):
        self.assertEqual(checkpoint_digest(self.model), self.weights)
        self.assertEqual([module.training for module in self.model.modules()], self.modes)
        self.assertEqual([parameter.requires_grad for parameter in self.model.parameters()], self.flags)
        torch.testing.assert_close(next(self.model.parameters()).grad, self.grad, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in list(self.model.parameters())[1:]))
        self.assertEqual(self.banks, self.original_banks)
        self.assertFalse(self.model.inferior_frontal.recurrent._forward_hooks)
        self.assertFalse(self.model.inferior_frontal.recurrent._forward_pre_hooks)

    def inspect(self, banks=None):
        try:
            report = diagnostic.inspect_gradients(self.model, self.banks if banks is None else banks,
                endpoint="curriculum", cell_inventory=self.inventory, pairs_per_cell=1)
            self.reports.append(report)
            return report
        except BaseException as error:
            if hasattr(error, "gradient_report"):
                self.reports.append(error.gradient_report)
            raise

    def test_components_match_original_loss_and_total_gradient(self):
        parameters, partitions, identities = diagnostic._partitions(self.model)
        self.assertEqual(identities["tied_token_embedding"]["parameter_count"], 1)
        self.assertEqual(identities["tied_token_embedding"]["scalar_count"], self.model.tokens.weight.numel())
        selected_ids = set(map(id, parameters))
        batch = pack_composition_episodes(self.banks["color/d0/direct/t8"][:2], device="cpu", training=True,
            pair_validator=validate_pair, max_turns=8, max_context_tokens=1024, max_reply_bytes=32)
        def count_decoder(module, arguments):
            type(self).direct_decoder_calls += 1
            type(self).direct_decoder_row_steps += arguments[0].shape[0] * arguments[0].shape[1]
        handle = self.model.inferior_frontal.recurrent.register_forward_pre_hook(count_decoder)
        try:
            for parameter in self.model.parameters():
                parameter.requires_grad_(id(parameter) in selected_ids)
            self.model.train()
            type(self).direct_forwards += 1
            type(self).direct_episodes += 2
            output = self.model(**batch["inputs"], decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
            parts, denominators = diagnostic._components(output, batch)
            original = sequence_objective(output, batch)
            torch.testing.assert_close(parts["query"] + .25*parts["acknowledgement"], original["action_loss"], rtol=0, atol=0)
            torch.testing.assert_close(parts["reply"], original["reply_loss"], rtol=0, atol=0)
            torch.testing.assert_close(parts["observation"], original["observation_language_loss"], rtol=0, atol=0)
            torch.testing.assert_close(sum(diagnostic.WEIGHTS[name]*parts[name] for name in parts), original["loss"])
            self.assertEqual(len(denominators["reply_tokens_per_turn"]), 8)
            self.assertEqual(len(denominators["observation_tokens_per_turn"]), 8)
            gradients = {}
            for name in diagnostic.COMPONENTS:
                type(self).direct_gradients += 1
                gradients[name] = torch.autograd.grad(parts[name], parameters, retain_graph=True)
            type(self).direct_gradients += 1
            total = torch.autograd.grad(original["loss"], parameters)
            for index, expected in enumerate(total):
                combined = sum(diagnostic.WEIGHTS[name]*gradients[name][index] for name in diagnostic.COMPONENTS)
                torch.testing.assert_close(combined, expected, rtol=2e-4, atol=2e-6)
        finally:
            handle.remove()
            for module, mode in zip(self.model.modules(), self.modes):
                module.training = mode
            for parameter, flag in zip(self.model.parameters(), self.flags):
                parameter.requires_grad_(flag)

    def test_interrupted_gradient_preserves_type_and_all_state(self):
        with patch.object(torch.autograd, "grad", side_effect=KeyboardInterrupt("fixture interrupt")):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.inspect()
        report = caught.exception.gradient_report
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual((report["family_forwards_completed"], report["encoder_episodes_completed"]), (3, 6))
        self.assertEqual((report["component_gradient_evaluations_attempted"],
                          report["component_gradient_evaluations_completed"]), (1, 0))
        self.assertTrue(report["model_state_unchanged"])
        self.assertTrue(report["requires_grad_flags_restored"])

    def test_malformed_pair_and_inference_mode_refuse_before_forward(self):
        malformed = copy.deepcopy(self.banks)
        malformed[self.inventory[0]][0]["turns"][-1]["target"] = 3
        with self.assertRaises(ValueError):
            self.inspect(malformed)
        self.assertEqual(self.reports[-1]["family_forwards_attempted"], 0)
        with torch.inference_mode(), self.assertRaises(ValueError):
            self.inspect()
        self.assertEqual(self.reports[-1]["component_gradient_evaluations_attempted"], 0)

    def test_mutated_forward_inputs_fail_and_unexpected_weights_restore(self):
        def mutate(module, arguments, keywords, output):
            keywords["token_ids"][0, 0] = 4
            with torch.no_grad():
                next(module.parameters()).add_(.25)
        handle = self.model.register_forward_hook(mutate, with_kwargs=True)
        try:
            with self.assertRaisesRegex(RuntimeError, "forward inputs") as caught:
                self.inspect()
        finally:
            handle.remove()
        report = caught.exception.gradient_report
        self.assertEqual((report["family_forwards_attempted"], report["encoder_episodes_attempted"]), (1, 2))
        self.assertEqual(report["component_gradient_evaluations_attempted"], 0)
        self.assertTrue(report["model_state_restored"])
        self.assertFalse(report["model_state_unchanged"])
        self.assertTrue(report["input_banks_unchanged"])

    def test_normal_inspection_uses_first_pairs_and_four_family_mean_gradients(self):
        with torch.no_grad():
            report = self.inspect()
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["production_contract"])
        self.assertEqual((report["groups_completed"], report["family_forwards_completed"],
            report["encoder_episodes_completed"], report["component_gradient_evaluations_completed"]), (1, 3, 6, 4))
        self.assertEqual(report["selected_banks_sha256"], diagnostic._sha({cell: rows[:2] for cell, rows in self.banks.items()}))
        self.assertEqual(report["component_weights"], dict(query=1., acknowledgement=.25, reply=.1, observation=.1))
        for group in report["groups"]:
            for partition in group["geometry"].values():
                for name in diagnostic.COMPONENTS:
                    self.assertEqual(partition["weighted_norms"][name],
                        diagnostic.WEIGHTS[name]*partition["unweighted_norms"][name])
                self.assertEqual(len(partition["component_cosines"]), 6)
                self.assertTrue(all(value is None or -1 <= value <= 1 for value in partition["component_cosines"].values()))
        self.assertTrue(report["packed_batches_unchanged"])
        self.assertTrue(report["model_state_unchanged"])
        self.assertEqual(report["weights_sha256_before"], report["weights_sha256_after"])

    def test_zero_norm_and_weighted_auxiliary_geometry(self):
        vectors = dict(query=(torch.tensor([1., 0.]),), acknowledgement=(torch.tensor([-1., 0.]),),
            reply=(torch.zeros(2),), observation=(torch.tensor([0., 1.]),))
        geometry = diagnostic._geometry(vectors, {"fixture": [0]})["fixture"]
        self.assertEqual(geometry["component_cosines"]["query/acknowledgement"], -1.)
        self.assertIsNone(geometry["component_cosines"]["query/reply"])
        self.assertAlmostEqual(geometry["weighted_auxiliary_norm"], math.sqrt(.25**2 + .1**2))
        self.assertAlmostEqual(geometry["query_weighted_auxiliary_cosine"], -.25/math.sqrt(.25**2 + .1**2))
        vectors["query"] = (torch.tensor([float("nan"), 0.]),)
        with self.assertRaises(FloatingPointError):
            diagnostic._geometry(vectors, {"fixture": [0]})


if __name__ == "__main__":
    unittest.main()
