"""Arithmetic and input-contract checks for the analytical language budget."""

import json
import unittest

from brain_in_computer.resources import estimate_language_budget


class LanguageBudgetTests(unittest.TestCase):
    def test_parameter_storage_scales_without_claiming_total_memory(self):
        for count in (1, 3, 162_810, 1_000_000, 10_000_000, 100_000_000, 3_000_000_000):
            with self.subTest(parameters=count):
                budget = estimate_language_budget(count)
                self.assertEqual(budget["parameter_count"], count)
                states = budget["parameter_state_bytes"]
                self.assertEqual(states["inference_fp32"], 4 * count)
                self.assertEqual(states["inference_fp16_bf16"], 2 * count)
                self.assertEqual(states["training_fp32_adam"], 16 * count)
                self.assertGreaterEqual(states["inference_4bit_ideal"] * 2, count)
                self.assertLess(states["inference_4bit_ideal"] * 2, count + 2)
                self.assertTrue(all(isinstance(value, int) for value in states.values()))
                self.assertFalse(budget["language_training_performed"])
                self.assertFalse(budget["model_downloaded"])
                self.assertIsNone(budget["estimated_training_seconds"])
                assumptions = " ".join(budget["assumptions"])
                self.assertIn("not total VRAM", assumptions)
                self.assertIn("Activations", assumptions)
                self.assertIn("quantization metadata", assumptions)
                self.assertIn("general intelligence", assumptions)
                self.assertEqual(json.loads(json.dumps(budget, allow_nan=False)), budget)

    def test_four_bit_packing_rounds_up_odd_parameter_counts(self):
        self.assertEqual(estimate_language_budget(1)["parameter_state_bytes"]["inference_4bit_ideal"], 1)
        self.assertEqual(estimate_language_budget(2)["parameter_state_bytes"]["inference_4bit_ideal"], 1)
        self.assertEqual(estimate_language_budget(3)["parameter_state_bytes"]["inference_4bit_ideal"], 2)

    def test_time_requires_external_measurement_and_counts_every_epoch(self):
        no_measurement = estimate_language_budget(100, corpus_tokens=6000, epochs=3)
        self.assertIsNone(no_measurement["estimated_training_seconds"])
        self.assertIsNone(no_measurement["training_workload"]["measured_tokens_per_second"])
        budget = estimate_language_budget(100, corpus_tokens=6000, epochs=3,
                                          measured_tokens_per_second=500)
        self.assertEqual(budget["training_workload"]["processed_tokens"], 18_000)
        self.assertEqual(budget["estimated_training_seconds"], 36.0)
        doubled_epochs = estimate_language_budget(100, corpus_tokens=6000, epochs=6,
                                                  measured_tokens_per_second=500)
        doubled_speed = estimate_language_budget(100, corpus_tokens=6000, epochs=3,
                                                 measured_tokens_per_second=1000)
        self.assertEqual(doubled_epochs["estimated_training_seconds"], 2 * budget["estimated_training_seconds"])
        self.assertEqual(doubled_speed["estimated_training_seconds"], budget["estimated_training_seconds"] / 2)
        self.assertEqual(json.loads(json.dumps(budget, allow_nan=False)), budget)

    def test_counts_require_positive_integers_and_reject_booleans(self):
        for field in ("parameter_count", "corpus_tokens", "epochs"):
            for invalid in (0, -1):
                with self.subTest(field=field, value=invalid):
                    kwargs = {"parameter_count": 100, field: invalid}
                    with self.assertRaises(ValueError):
                        estimate_language_budget(**kwargs)
            for invalid in (True, False, 1.5, "2", [], {}):
                with self.subTest(field=field, value=invalid):
                    kwargs = {"parameter_count": 100, field: invalid}
                    with self.assertRaises(TypeError):
                        estimate_language_budget(**kwargs)

    def test_rate_requires_corpus_and_a_positive_finite_number(self):
        with self.assertRaises(ValueError):
            estimate_language_budget(100, measured_tokens_per_second=500)
        for invalid in (0, -1, float("nan"), float("inf"), -float("inf"), 10 ** 1000):
            with self.subTest(rate=invalid), self.assertRaises(ValueError):
                estimate_language_budget(100, corpus_tokens=500, measured_tokens_per_second=invalid)
        for invalid in (True, False, "500", [], {}):
            with self.subTest(rate=invalid), self.assertRaises(TypeError):
                estimate_language_budget(100, corpus_tokens=500, measured_tokens_per_second=invalid)

    def test_unrepresentable_duration_is_rejected_instead_of_returning_infinity(self):
        with self.assertRaises(ValueError):
            estimate_language_budget(100, corpus_tokens=10**1000, measured_tokens_per_second=1)
        with self.assertRaises(ValueError):
            estimate_language_budget(100, corpus_tokens=10**20, measured_tokens_per_second=1e-300)


if __name__ == "__main__":
    unittest.main()
