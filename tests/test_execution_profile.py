"""CPU-only guards for the opt-in profile; GPU repeatability is a separate probe."""
import importlib
import os
import unittest
from unittest.mock import patch

import torch

from experiments import execution_profile as profile


class ExecutionProfileTests(unittest.TestCase):
    def setUp(self):
        self.settings = profile.profile_settings()
        self.configured = profile._configured
        self.environment = patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        for key in ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE"):
            os.environ.pop(key, None)
        profile._configured = False

    def tearDown(self):
        value = self.settings
        torch.use_deterministic_algorithms(value["deterministic_algorithms"], warn_only=value["deterministic_warn_only"])
        torch.backends.cudnn.deterministic = value["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = value["cudnn_benchmark"]
        torch.set_float32_matmul_precision(value["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = value["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = value["cudnn_allow_tf32"]
        torch.utils.deterministic.fill_uninitialized_memory = value["fill_uninitialized_memory"]
        profile._configured = self.configured
        self.environment.stop()

    def configure(self):
        with patch.object(torch.cuda, "is_initialized", return_value=False):
            return profile.configure_strict_profile()

    def test_import_does_not_configure_or_consume_rng(self):
        before = profile.profile_settings();rng = torch.get_rng_state().clone()
        importlib.reload(profile)
        self.assertEqual(before, profile.profile_settings())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(profile._configured)

    def test_opt_in_sets_strict_contract_without_cuda_or_rng(self):
        rng = torch.get_rng_state().clone()
        with patch.object(torch.cuda, "init", side_effect=AssertionError("unexpected CUDA initialization")):
            result = self.configure()
        self.assertTrue(result["configured_before_cuda_initialization"])
        self.assertEqual(result["settings"], profile.assert_strict_profile())
        self.assertEqual(result["settings"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertFalse(result["settings"]["deterministic_warn_only"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_late_call_rejected_without_changing_settings(self):
        before = profile.profile_settings()
        with patch.object(torch.cuda, "is_initialized", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "before CUDA"):
                profile.configure_strict_profile()
        self.assertEqual(before, profile.profile_settings())

    def test_undeclared_and_changed_settings_fail_without_repair(self):
        with self.assertRaisesRegex(RuntimeError, "configure_strict_profile"):
            profile.assert_strict_profile()
        self.configure()
        torch.use_deterministic_algorithms(True, warn_only=True)
        with self.assertRaisesRegex(RuntimeError, "deterministic_warn_only"):
            profile.assert_strict_profile()
        self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        with self.assertRaisesRegex(RuntimeError, "CUBLAS_WORKSPACE_CONFIG"):
            profile.assert_strict_profile()

    def test_conflicting_tf32_overrides_and_runtime_version_rejected(self):
        for name in ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE"):
            with patch.dict(os.environ, {name: "1"}):
                with self.assertRaisesRegex(RuntimeError, "TF32-off"):
                    self.configure()
        with patch.object(torch, "__version__", "2.12.0"):
            with self.assertRaisesRegex(RuntimeError, "Torch 2.11"):
                self.configure()

    def test_runtime_profile_never_falls_back_to_cpu(self):
        self.configure()
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            profile.runtime_profile("cpu")
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                profile.runtime_profile()

    def test_tf32_and_uninitialized_memory_drift_are_detected(self):
        self.configure()
        torch.backends.cudnn.allow_tf32 = True
        torch.utils.deterministic.fill_uninitialized_memory = False
        with self.assertRaisesRegex(RuntimeError, "cudnn_allow_tf32.*fill_uninitialized_memory"):
            profile.assert_strict_profile()


if __name__ == "__main__":
    unittest.main()
