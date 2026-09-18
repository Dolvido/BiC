"""Explicit local Torch 2.11 execution profile; importing changes no settings.

Configure once before CUDA initialization. Unsupported deterministic operations
raise through Torch rather than falling back. This is a reproducibility request,
not evidence of repeatability across machines, releases or unseen workloads.

References: https://docs.pytorch.org/docs/2.11/notes/randomness.html and
https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform

import torch


SCHEMA = "bic-strict-local-execution-v1"
CUBLAS_WORKSPACE = ":4096:8"
_configured = False


def profile_settings():
    """Read the effective contract without initializing CUDA or changing RNGs."""
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }


def _check_version_and_overrides():
    if str(torch.__version__).split("+")[0].split(".")[:2] != ["2", "11"]:
        raise RuntimeError("strict profile requires the declared Torch 2.11 runtime")
    for name in ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE"):
        if os.environ.get(name) not in (None, "0"):
            raise RuntimeError(f"{name} conflicts with the explicit TF32-off profile")


def assert_strict_profile():
    """Reject drift or an undeclared profile; never repair settings during a run."""
    if not _configured:
        raise RuntimeError("configure_strict_profile must run before CUDA initialization")
    _check_version_and_overrides()
    current = profile_settings()
    expected = {
        "deterministic_algorithms": True, "deterministic_warn_only": False,
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest", "fill_uninitialized_memory": True,
        "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE,
    }
    differences = [name for name, value in expected.items() if current[name] != value]
    if differences:
        raise RuntimeError("strict execution settings changed: " + ", ".join(differences))
    return current


def configure_strict_profile():
    """Opt in before any CUDA work, without initializing CUDA or consuming RNGs.

    The process environment is changed only by this explicit call. A late call
    fails even if the current settings already look correct: cuBLAS may have
    consumed its environment when CUDA was initialized. Model and sampler seeds
    remain the caller's responsibility; this function never reseeds them.
    """
    global _configured
    if torch.cuda.is_initialized():
        raise RuntimeError("strict profile must be configured before CUDA initialization")
    _check_version_and_overrides()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.utils.deterministic.fill_uninitialized_memory = True
    _configured = True
    return {"schema": SCHEMA, "settings": assert_strict_profile(),
            "configured_before_cuda_initialization": True,
            "torch_version": str(torch.__version__), "source_sha256": source_hash()}


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def runtime_profile(device="cuda:0"):
    """Verify the declared profile, initialize the requested CUDA device, record it."""
    settings = assert_strict_profile()
    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError("strict local profile requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the strict profile has no CPU fallback")
    index = torch.cuda.current_device() if selected.index is None else selected.index
    properties = torch.cuda.get_device_properties(index)
    return {
        "schema": SCHEMA, "source_sha256": source_hash(), "settings": settings,
        "configured_before_cuda_initialization": True,
        "torch_version": str(torch.__version__), "torch_git_version": torch.version.git_version,
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
        "device": {"index": index, "name": properties.name,
                   "compute_capability": [properties.major, properties.minor],
                   "total_memory_bytes": properties.total_memory,
                   "multi_processor_count": properties.multi_processor_count,
                   "uuid": str(getattr(properties, "uuid", "unavailable"))},
        "scope": "Requested deterministic local execution; workload-specific tests still required. No cross-machine or cross-release guarantee.",
    }
