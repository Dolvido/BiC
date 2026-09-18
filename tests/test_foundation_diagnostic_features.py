"""Fresh tiny CPU fixtures only; no historical/formal data or weight loads."""
import copy
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.dialogue_student import _exact_replies, _generate_reply_tokens, checkpoint_digest
from brain_in_computer.language import ByteCodec
from experiments.composition_data import pack_composition_episodes
from experiments.foundation_curriculum import generate_pair
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments import foundation_diagnostic_features as features


def setUpModule():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


class DiagnosticFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started = time.monotonic()
        cls.reports = []
        cls.direct_encoder_episodes = 0
        cls.direct_bos_row_steps = 0
        cls.direct_generation_row_steps = 0
        cls.config = SequenceConfig(width=8, layers=1, heads=2, feedforward=16,
                                    max_turns=12, max_positions=1024, max_output_bytes=32)
        cls.model = build_sequence_student(831904799, config=cls.config)
        cls.model.train()
        cls.model.blocks[0].eval()  # Preserve mixed module modes, not just the root flag.
        next(cls.model.parameters()).grad = torch.ones_like(next(cls.model.parameters()))
        cls.modes = [module.training for module in cls.model.modules()]
        cls.weights = checkpoint_digest(cls.model)
        cls.gradient = next(cls.model.parameters()).grad.clone()
        cls.inventory = ("color/d0/direct/t8", "count/d0/direct/t10")
        # Select only on visible byte length, before constructing any cache, to
        # guarantee genuine padding coverage without altering a canonical row.
        for cls.color_value_seed in range(831905701, 831905717):
            color = generate_pair("color", 831904701, depth=0, turns=8, split="train", value_seed=cls.color_value_seed)
            lengths = [sum(len(turn["text"].encode()) + 2 for turn in row["turns"]) for row in color]
            if lengths[0] != lengths[1]:
                break
        else:
            raise AssertionError("bounded fresh fixture failed to provide byte padding")
        cls.banks = {cls.inventory[0]: color,
                     cls.inventory[1]: generate_pair("count", 831904702, depth=0, turns=10, split="train")}
        cls.original_banks = copy.deepcopy(cls.banks)
        cls.cache = features.extract_features(cls.model, cls.banks, role="fit",
                                             cell_inventory=cls.inventory, batch_size=2)
        cls.reports.append(cls.cache.report)

    @classmethod
    def tearDownClass(cls):
        counters = sorted({name for report in cls.reports for name in report
                           if name.endswith(("_attempted", "_completed"))})
        totals = {name: sum(report.get(name, 0) for report in cls.reports) for name in counters}
        total_encoder = totals.get("encoder_episodes_attempted", 0) + cls.direct_encoder_episodes
        total_generation = totals.get("generated_decoder_row_steps_attempted", 0) + cls.direct_generation_row_steps
        value = dict(schema="bic-diagnostic-features-cpu-proof-v1", model_constructions=1,
            config=vars(cls.config), model_seed=831904799, fixture_seeds=[831904701, 831904702],
            color_value_seed=cls.color_value_seed,
            checkpoint_loads=0, optimizer_updates=0, backward_evaluations=0,
            historical_or_formal_inputs=False, reports=cls.reports, totals=totals,
            direct_reference_encoder_episodes=cls.direct_encoder_episodes,
            direct_reference_bos_row_steps=cls.direct_bos_row_steps,
            direct_reference_generation_row_steps=cls.direct_generation_row_steps,
            total_attempted_encoder_episodes=total_encoder,
            total_attempted_generation_row_steps=total_generation,
            encoder_bound=24, generation_row_step_bound=2600,
            within_approved_bounds=total_encoder <= 24 and total_generation <= 2600,
            wall_seconds=time.monotonic() - cls.started)
        print("DIAGNOSTIC_FEATURES_CPU_WORK=" + json.dumps(value, sort_keys=True))
        path = os.environ.get("BIC_DIAGNOSTIC_FEATURES_ACCOUNTING")
        if path:
            with Path(path).open("x", encoding="utf8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True)
        if not value["within_approved_bounds"]:
            raise AssertionError("CPU proof exceeded approved work bounds")

    def assert_preserved(self):
        self.assertEqual(checkpoint_digest(self.model), self.weights)
        self.assertEqual([module.training for module in self.model.modules()], self.modes)
        torch.testing.assert_close(next(self.model.parameters()).grad, self.gradient)
        self.assertEqual(self.banks, self.original_banks)
        self.assertFalse(self.model.inferior_frontal.recurrent._forward_pre_hooks)
        self.assertFalse(self.model.inferior_frontal.recurrent._forward_hooks)
        self.assertFalse(self.model.inferior_frontal.readout._forward_hooks)

    def test_alignment_variable_turns_padding_and_original_state(self):
        cached = self.cache.tensors
        expected = {name: [] for name in ("eos", "production", "action_logits", "bos_logits", "labels")}
        actual_positions, padded_positions, turn_count = 0, 0, 0
        try:
            self.model.eval()
            with torch.inference_mode():
                for cell in self.inventory:
                    batch = pack_composition_episodes(self.banks[cell], training=False,
                        max_turns=12, max_context_tokens=1024, max_reply_bytes=32)
                    original_inputs = {key: value.clone() for key, value in batch["inputs"].items()}
                    inputs = batch["inputs"]
                    shape = inputs["eos_positions"].shape
                    type(self).direct_encoder_episodes += shape[0]
                    type(self).direct_bos_row_steps += shape[0] * shape[1]
                    output = self.model(**inputs, decoder_input_ids=torch.full((*shape, 1), ByteCodec.BOS, dtype=torch.long))
                    for key, value in original_inputs.items():
                        torch.testing.assert_close(inputs[key], value)
                    expected["eos"].append(output["context_states"].gather(1,
                        inputs["eos_positions"][:, :, None].expand(-1, -1, 8)).flatten(0, 1))
                    expected["production"].append(output["production_context"].flatten(0, 1))
                    expected["action_logits"].append(output["logits"].flatten(0, 1))
                    expected["bos_logits"].append(output["language_logits"][:, :, 0].flatten(0, 1))
                    expected["labels"].append(batch["supervision"]["action_targets"].flatten())
                    actual_positions += int(inputs["lengths"].sum())
                    padded_positions += inputs["token_ids"].numel()
                    turn_count += shape[0] * shape[1]
        finally:
            for module, mode in zip(self.model.modules(), self.modes):
                module.training = mode
        for key, values in expected.items():
            torch.testing.assert_close(cached[key], torch.cat(values), atol=0, rtol=0)
        self.assertEqual((turn_count, cached["eos"].shape, cached["production"].shape, cached["bos_logits"].shape),
                         (36, (36, 8), (36, 128), (36, 259)))
        self.assertEqual(self.cache.report["encoder_actual_positions_completed"], actual_positions)
        self.assertEqual(self.cache.report["encoder_padded_positions_completed"], padded_positions)
        self.assertGreater(padded_positions, actual_positions)
        self.assertEqual(self.cache.report["bos_decoder_row_steps_completed"], 36)
        self.assert_preserved()

    def test_cached_decoding_uses_no_encoder_and_preserves_first_token_and_exactness(self):
        with patch.object(self.model, "forward", side_effect=AssertionError("cached decoding called encoder")):
            decoded = features.decode_cached_replies(self.model, self.cache, batch_size=11)
        self.reports.append(decoded["report"])
        cached = self.cache.tensors
        torch.testing.assert_close(decoded["first_tokens"], cached["bos_logits"].argmax(-1))
        torch.testing.assert_close(decoded["reply_correct"], _exact_replies(decoded["tokens"], cached["reply_targets"]))
        self.assertEqual(decoded["report"]["encoder_episodes_attempted"], 0)
        self.assertEqual(decoded["report"]["generated_decoder_calls_completed"], 4)
        # One independent direct generation, counted at each actual GRU call.
        def count(module, inputs):
            type(self).direct_generation_row_steps += inputs[0].shape[0] * inputs[0].shape[1]
        handle = self.model.inferior_frontal.recurrent.register_forward_pre_hook(count)
        try:
            with torch.inference_mode():
                direct = _generate_reply_tokens(self.model, cached["production"][:1])
        finally:
            handle.remove()
        torch.testing.assert_close(decoded["tokens"][:1, :direct.shape[1]], direct)
        self.assert_preserved()

    def test_detached_cache_and_stable_probe_coordinates(self):
        supplied = self.cache.probe_inputs("eos")
        expected_coordinates, expected_kinds, expected_labels = [], [], []
        for cell in self.inventory:
            for member, row in enumerate(self.banks[cell]):
                for turn_index, turn in enumerate(row["turns"]):
                    expected_coordinates.append([0, member, turn_index])
                    expected_kinds.append(turn["kind"])
                    expected_labels.append(turn["target"])
        self.assertEqual(supplied["coordinates"], expected_coordinates)
        self.assertEqual(supplied["kinds"], expected_kinds)
        self.assertEqual(supplied["labels"].tolist(), expected_labels)
        self.assertFalse(supplied["features"].requires_grad)
        supplied["features"].zero_()
        supplied["labels"].fill_(3)
        supplied["cells"].clear()
        self.cache.metadata["coordinates"].clear()
        self.assertEqual(self.cache.probe_inputs("eos")["labels"].tolist(), expected_labels)
        self.assertEqual(len(self.cache.probe_inputs("production")["cells"]), 36)
        self.cache._check()
        self.assert_preserved()

    def test_malformed_pairs_roles_and_capacity_fail_before_forward(self):
        bad = copy.deepcopy(self.banks)
        bad[self.inventory[1]][0]["turns"][-1]["target"] = 3
        with patch.object(self.model, "forward", side_effect=AssertionError("preflight failed late")):
            for banks, role in ((bad, "fit"), (self.banks, "evaluation")):
                with self.subTest(role=role), self.assertRaises(ValueError) as caught:
                    features.extract_features(self.model, banks, role=role, cell_inventory=self.inventory)
                self.reports.append(caught.exception.diagnostic_report)
                self.assertEqual(caught.exception.diagnostic_report["encoder_episodes_attempted"], 0)
            self.model.config = SequenceConfig(width=8, layers=1, heads=2, feedforward=16, max_turns=8)
            try:
                with self.assertRaises(ValueError) as caught:
                    features.extract_features(self.model, self.banks, role="fit", cell_inventory=self.inventory)
                self.reports.append(caught.exception.diagnostic_report)
                self.assertEqual(caught.exception.diagnostic_report["encoder_episodes_attempted"], 0)
            finally:
                self.model.config = self.config
        self.assert_preserved()

    def test_encoder_interruption_keeps_partial_receipt_original_exception_and_cleanup(self):
        original = self.model.forward
        calls = 0
        interruption = KeyboardInterrupt("fresh fixture interrupted after one batch")
        def stop_after_first(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise interruption
            return original(*args, **kwargs)
        with patch.object(self.model, "forward", stop_after_first):
            with self.assertRaises(KeyboardInterrupt) as caught:
                features.extract_features(self.model, self.banks, role="fit", cell_inventory=self.inventory, batch_size=2)
        self.assertIs(caught.exception, interruption)
        report = caught.exception.diagnostic_report
        self.reports.append(report)
        self.assertEqual((report["encoder_episodes_attempted"], report["encoder_episodes_completed"]), (4, 2))
        self.assertEqual(report["bos_decoder_row_steps_completed"], 16)
        self.assertTrue(report["model_state_unchanged"])
        self.assertTrue(report["modes_restored"])
        self.assert_preserved()

    def test_decoder_interruption_counts_entered_steps_and_cleans_up(self):
        original = self.model.inferior_frontal.recurrent.forward
        original_readout = self.model.inferior_frontal.readout.forward
        calls = 0
        interruption = SystemExit("fresh decoder fixture")
        def stop_after_first(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise interruption
            return original(*args, **kwargs)
        def continue_reply(*args, **kwargs):
            logits = torch.zeros_like(original_readout(*args, **kwargs))
            logits[..., ord("N") + ByteCodec.BYTE_OFFSET] = 1.
            return logits
        with patch.object(self.model.inferior_frontal.recurrent, "forward", stop_after_first), \
                patch.object(self.model.inferior_frontal.readout, "forward", continue_reply):
            with self.assertRaises(SystemExit) as caught:
                features.decode_cached_replies(self.model, self.cache, batch_size=4)
        self.assertIs(caught.exception, interruption)
        report = caught.exception.diagnostic_report
        self.reports.append(report)
        self.assertEqual((report["generated_decoder_calls_attempted"], report["generated_decoder_calls_completed"]), (1, 0))
        self.assertEqual((report["generated_decoder_row_steps_attempted"], report["generated_decoder_row_steps_completed"]), (8, 4))
        self.assertEqual(report["encoder_episodes_attempted"], 0)
        self.assert_preserved()

    def test_cached_decode_rejects_nonfinite_recurrent_and_readout_before_token_scoring(self):
        recurrent = self.model.inferior_frontal.recurrent.forward
        readout = self.model.inferior_frontal.readout.forward
        def invalid_hidden(*args, **kwargs):
            activity, hidden = recurrent(*args, **kwargs)
            return activity, hidden * float("nan")
        def invalid_logits(*args, **kwargs):
            return readout(*args, **kwargs) * float("nan")
        for module, replacement in ((self.model.inferior_frontal.recurrent, invalid_hidden),
                                    (self.model.inferior_frontal.readout, invalid_logits)):
            with self.subTest(module=type(module).__name__), patch.object(module, "forward", replacement):
                with self.assertRaisesRegex(FloatingPointError, "nonfinite diagnostic") as caught:
                    features.decode_cached_replies(self.model, self.cache, batch_size=4)
            report = caught.exception.diagnostic_report
            self.reports.append(report)
            self.assertEqual(report["generated_decoder_calls_completed"], 0)
            self.assertEqual(report["generated_decoder_row_steps_completed"], 4)
            self.assertTrue(report["model_state_unchanged"])
            self.assert_preserved()

    def test_dtype_mutation_is_rejected_without_false_restoration_claim(self):
        recurrent = self.model.inferior_frontal.recurrent.forward
        def wrong_dtype(*args, **kwargs):
            self.model.double()
            return recurrent(*args, **kwargs)
        try:
            with patch.object(self.model.inferior_frontal.recurrent, "forward", wrong_dtype):
                with self.assertRaises((RuntimeError, ValueError)) as caught:
                    features.decode_cached_replies(self.model, self.cache, batch_size=4)
            report = caught.exception.diagnostic_report
            self.reports.append(report)
            self.assertFalse(report["model_state_unchanged"])
            self.assertFalse(report["model_state_restored"])
            self.assertIn("dtype", report["cleanup_error"])
            self.assertTrue(report["modes_restored"])
        finally:
            self.model.float()
        self.assert_preserved()


if __name__ == "__main__":
    unittest.main(verbosity=2)
