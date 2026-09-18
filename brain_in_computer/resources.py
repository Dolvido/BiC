"""Analytical parameter-state budgets, with no model loading or training.

These estimates describe the storage occupied by specified parameter tensors.
They do not estimate total device memory or establish that a model will fit.
"""

from __future__ import annotations

import math


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not a boolean or other type")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def estimate_language_budget(
    parameter_count: int,
    *,
    corpus_tokens: int | None = None,
    epochs: int = 1,
    measured_tokens_per_second: float | None = None,
) -> dict:
    """Estimate parameter storage and optionally extrapolate measured training time.

    Throughput must be supplied by the caller from a relevant training benchmark;
    this function neither measures throughput nor checks the measurement's origin.
    A measurement from another model, sequence length, batch size, precision, or
    device cannot establish performance for the proposed configuration.

    ``corpus_tokens`` counts tokens traversed once per epoch. Optional duration is
    simply ``corpus_tokens * epochs / measured_tokens_per_second``; it excludes
    evaluation, checkpointing, preprocessing, and other unmeasured overhead.
    """
    _positive_integer("parameter_count", parameter_count)
    _positive_integer("epochs", epochs)
    if corpus_tokens is not None:
        _positive_integer("corpus_tokens", corpus_tokens)

    measured_rate = None
    if measured_tokens_per_second is not None:
        if isinstance(measured_tokens_per_second, bool) or not isinstance(
            measured_tokens_per_second, (int, float)
        ):
            raise TypeError("measured_tokens_per_second must be a real number, not a boolean")
        try:
            measured_rate = float(measured_tokens_per_second)
        except OverflowError as error:
            raise ValueError("measured_tokens_per_second must be finite and positive") from error
        if not math.isfinite(measured_rate) or measured_rate <= 0:
            raise ValueError("measured_tokens_per_second must be finite and positive")
        if corpus_tokens is None:
            raise ValueError("corpus_tokens is required with measured_tokens_per_second")

    processed_tokens = None if corpus_tokens is None else corpus_tokens * epochs
    estimated_seconds = None
    if measured_rate is not None:
        try:
            estimated_seconds = processed_tokens / measured_rate
        except OverflowError as error:
            raise ValueError("estimated training duration exceeds the representable range") from error
        if not math.isfinite(estimated_seconds):
            raise ValueError("estimated training duration exceeds the representable range")

    return {
        "parameter_count": parameter_count,
        "parameter_state_bytes": {
            "inference_fp32": 4 * parameter_count,
            "inference_fp16_bf16": 2 * parameter_count,
            "inference_4bit_ideal": (parameter_count + 1) // 2,
            "training_fp32_adam": 16 * parameter_count,
        },
        "training_workload": {
            "corpus_tokens": corpus_tokens,
            "epochs": epochs,
            "processed_tokens": processed_tokens,
            "measured_tokens_per_second": measured_rate,
        },
        "estimated_training_seconds": estimated_seconds,
        "assumptions": [
            "Parameter-state storage only; these values are not total VRAM estimates or fit guarantees.",
            "FP32 inference weights use 4 bytes per parameter; FP16/BF16 inference weights use 2.",
            "Ideal packed 4-bit inference weights use half a byte per parameter, rounded up to whole bytes.",
            "4-bit storage assumes a compatible quantized representation; no model is quantized here.",
            "FP32 Adam training state uses 4-byte weights, 4-byte gradients, and two 4-byte moment tensors: 16 bytes per parameter.",
            "Activations, caches, workspaces, runtime allocations, quantization metadata, and temporary buffers are excluded.",
            "Mixed-precision training, optimizer variants, offloading, and distributed replicas are not estimated.",
            "Duration is unavailable unless corpus size and externally measured training throughput are both supplied.",
            "Supplied throughput is not independently verified; it must represent the intended model, hardware, precision, batch size, and sequence length.",
            "Duration assumes constant measured throughput for every corpus token in every epoch; unmeasured preprocessing, evaluation, and checkpoint overhead are excluded.",
            "Neither parameter count nor this budget establishes English competence or general intelligence.",
        ],
        "language_training_performed": False,
        "model_downloaded": False,
    }
