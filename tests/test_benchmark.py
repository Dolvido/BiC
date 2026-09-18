"""Benchmark reporting, isolation, bounds, and one real small update."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from brain_in_computer.benchmark import benchmark_suite, _peak_rss


class BenchmarkTests(unittest.TestCase):
    def test_each_profile_uses_a_fresh_worker_and_report_is_saved(self):
        responses = [subprocess.CompletedProcess([], 0, json.dumps({"profile": name, "status": "ok"}), "")
                     for name in ("small", "medium")]
        with tempfile.TemporaryDirectory() as directory, patch(
            "brain_in_computer.benchmark.subprocess.run", side_effect=responses
        ) as run, patch("brain_in_computer.benchmark.platform.processor", return_value="test CPU"):
            output = Path(directory) / "costs.json"
            result = benchmark_suite(output, profiles=["small", "medium"], max_seconds=7)
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                self.assertEqual(call.args[0][-3:], ["-m", "brain_in_computer.benchmark", "--worker"])
                self.assertEqual(call.kwargs["timeout"], 7.0)
                self.assertEqual(call.kwargs["env"]["OMP_NUM_THREADS"], "1")
            payloads = [json.loads(call.kwargs["input"]) for call in run.call_args_list]
            self.assertNotEqual(payloads[0]["config"], payloads[1]["config"])
            self.assertIn("not detected hardware", result["home_reference_budget"]["scope"])

    def test_timeout_and_failure_are_recorded_without_inventing_measurements(self):
        with patch("brain_in_computer.benchmark.subprocess.run", side_effect=[
            subprocess.TimeoutExpired("worker", 1),
            subprocess.CompletedProcess([], 1, "", "CUDA unavailable"),
        ]), patch("brain_in_computer.benchmark.platform.processor", return_value="test CPU"):
            result = benchmark_suite(None, profiles=["small", "medium"], device="cuda", max_seconds=1)
        self.assertEqual([row["status"] for row in result["profiles"]], ["timeout", "error"])
        self.assertEqual(result["requested_device"], "cuda")
        for row in result["profiles"]:
            self.assertNotIn("mean_step_seconds", row)
            self.assertNotIn("host_peak_rss", row)

    def test_invalid_resource_bounds_fail_before_starting_a_worker(self):
        cases = [{"profiles": []}, {"profiles": "small"}, {"profiles": ["missing"]},
                 {"profiles": ["small", "small"]}, {"threads": True}, {"batch_size": 0},
                 {"warmup_steps": -1}, {"measured_steps": 0}, {"max_parameters": 0},
                 {"max_seconds": float("inf")}, {"max_seconds": True}, {"device": "mps"}]
        with patch("brain_in_computer.benchmark.subprocess.run") as run:
            for kwargs in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    benchmark_suite(None, **kwargs)
            run.assert_not_called()

    def test_peak_memory_is_explicitly_a_process_measurement(self):
        result = _peak_rss()
        self.assertIn("source", result)
        if result["bytes"] is not None:
            self.assertGreater(result["bytes"], 0)
            self.assertIn("process lifetime peak", result["scope"])

    def test_real_worker_updates_weights_and_reports_measured_memory(self):
        report = benchmark_suite(None, profiles=["small"], batch_size=1,
                                 warmup_steps=0, measured_steps=1, max_seconds=120)
        row = report["profiles"][0]
        self.assertEqual(row["status"], "ok", row.get("error"))
        self.assertTrue(row["first_parameter_changed"])
        self.assertGreater(row["optimizer_state_tensor_bytes"], 0)
        self.assertGreater(row["mean_step_seconds"], 0)
        self.assertEqual(row["optimizer_updates"], 1)
        self.assertEqual(row["precision"], "float32")
        self.assertFalse(row["workload"]["measures_task_competence"])
        self.assertIsNone(row["cuda_peak_allocated_bytes"])
        self.assertEqual(row["analytical_parameter_storage_bytes"]["fp32_weights"], row["parameter_count"] * 4)

    def test_parameter_cap_skips_updates(self):
        report = benchmark_suite(None, profiles=["small"], max_parameters=1)
        row = report["profiles"][0]
        self.assertEqual(row["status"], "skipped_parameter_cap", row.get("error"))
        self.assertGreater(row["parameter_count"], row["max_parameters"])
        self.assertNotIn("optimizer_updates", row)


if __name__ == "__main__":
    unittest.main()
