"""Isolated, bounded measurements of BiC training cost on this computer.

These synthetic updates measure computation and memory, not task learning or
the largest model a different computer can train. Each profile starts a fresh
Python process so process peak memory does not inherit an earlier profile.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence


PROFILES = {
    "small": {"hidden_size": 48, "language_hidden_size": 96, "embedding_size": 48, "visual_features": 32},
    "medium": {"hidden_size": 96, "language_hidden_size": 192, "embedding_size": 96, "visual_features": 96},
    "large": {"hidden_size": 160, "language_hidden_size": 320, "embedding_size": 160, "visual_features": 160},
    "home": {"hidden_size": 320, "language_hidden_size": 640, "embedding_size": 320, "visual_features": 320},
    "stretch": {"hidden_size": 576, "language_hidden_size": 1152, "embedding_size": 576, "visual_features": 576},
    "extended": {"hidden_size": 1152, "language_hidden_size": 2304, "embedding_size": 1152, "visual_features": 1152},
}
DEFAULT_PROFILES = ("small", "medium", "large", "home")


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _host_info() -> dict:
    result = {"system": platform.system(), "release": platform.release(),
              "machine": platform.machine(), "processor": platform.processor(),
              "logical_cpu_count": os.cpu_count(), "python": platform.python_version(),
              "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
              "contention": "Other processes are not suspended; load averages describe the shared host and may include work outside this container."}
    if sys.platform == "linux":
        for key, filename in (("cgroup_cpu_max", "/sys/fs/cgroup/cpu.max"),
                              ("cgroup_memory_max", "/sys/fs/cgroup/memory.max")):
            try:
                result[key] = Path(filename).read_text().strip()
            except OSError:
                result[key] = None
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            result["cpu_model"] = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                                         if line.startswith("model name")), None)
        except OSError:
            result["cpu_model"] = None
    return result


def _peak_rss() -> dict:
    """Return the process lifetime peak resident memory where supported."""
    if sys.platform in ("linux", "darwin"):
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        multiplier = 1 if sys.platform == "darwin" else 1024
        return {"bytes": int(raw * multiplier), "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
                "scope": "process lifetime peak resident set, including Python and framework"}
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in (
                            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                            "PagefileUsage", "PeakPagefileUsage")]]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        succeeded = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        if not succeeded:
            return {"bytes": None, "source": "GetProcessMemoryInfo failed", "scope": "unavailable"}
        return {"bytes": int(counters.PeakWorkingSetSize), "source": "Windows GetProcessMemoryInfo.PeakWorkingSetSize",
                "scope": "process lifetime peak working set, including Python and framework"}
    return {"bytes": None, "source": f"unsupported platform: {sys.platform}", "scope": "unavailable"}


def _run_profile(payload: dict) -> dict:
    """Worker implementation; normal callers should use benchmark_suite."""
    import torch
    from torch.nn import functional as F
    from .computer_use.model import ComputerBrain, ComputerConfig
    from .language import ByteCodec

    torch.set_num_threads(payload["threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(17041)
    device = torch.device(payload["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only cpu and cuda devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback was performed")
    config = ComputerConfig(**payload["config"])
    # Meta tensors provide exact shapes/counts before allocating weight storage.
    with torch.device("meta"):
        probe = ComputerBrain(config)
    count = sum(parameter.numel() for parameter in probe.parameters())
    del probe
    result = {
        "profile": payload["profile"], "config": dict(payload["config"], max_history=config.max_history),
        "parameter_count": count, "device": str(device), "precision": "float32",
        "mixed_precision": False, "torch_version": str(torch.__version__),
        "process_id": os.getpid(), "cpu_threads": torch.get_num_threads(),
        "max_parameters": payload["max_parameters"],
        "analytical_parameter_storage_bytes": {
            "fp32_weights": count * 4,
            "fp32_weights_gradients_and_two_adam_moments": count * 16,
        },
        "analytical_storage_scope": "Tensor arithmetic only; excludes activations, runtime, workspaces, and temporary buffers. Not measured training peak.",
    }
    if count > payload["max_parameters"]:
        return {**result, "status": "skipped_parameter_cap", "host_peak_rss": _peak_rss()}

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["cuda_device"] = {"name": properties.name, "total_memory_bytes": properties.total_memory,
                                 "compute_capability": [properties.major, properties.minor]}
        torch.cuda.reset_peak_memory_stats(device)
    model = ComputerBrain(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    batch = payload["batch_size"]
    pixels = torch.rand(batch, config.max_history, 3, 32, 32, device=device)
    body = torch.rand(batch, config.max_history, 4, device=device)
    prompts = ["click the red button."] * batch
    decoder, targets = model.prepare_reply(["clicked red."] * batch)
    action_targets = torch.randint(model.NUM_ACTIONS, (batch,), device=device)
    scene_targets = torch.stack([torch.randint(classes, (batch,), device=device)
                                 for classes in (4, 4, 4, 4, 5, 21)], dim=1)
    reference = next(model.parameters()).detach().clone()
    durations, losses = [], []

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for step in range(payload["warmup_steps"] + payload["measured_steps"]):
        synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(pixels, body, prompts, decoder)
        action_loss = F.cross_entropy(outputs["logits"][:, -1], action_targets)
        reply_loss = F.cross_entropy(outputs["language_logits"].reshape(-1, ByteCodec.VOCAB_SIZE),
                                     targets.reshape(-1), ignore_index=ByteCodec.PAD)
        scene_loss = torch.stack([
            F.cross_entropy(part, scene_targets[:, index])
            for index, part in enumerate(outputs["scene_logits"].split((4, 4, 4, 4, 5, 21), dim=-1))
        ]).mean()
        # This is a synthetic auxiliary target, not an environment prediction score.
        prediction_loss = outputs["prediction"].square().mean()
        loss = action_loss + reply_loss + 0.5 * scene_loss + 0.03 * prediction_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Benchmark produced a nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        synchronize()
        elapsed = time.perf_counter() - started
        if step >= payload["warmup_steps"]:
            durations.append(elapsed)
            losses.append(float(loss.detach()))

    optimizer_bytes = sum(value.numel() * value.element_size()
                          for state in optimizer.state.values() for value in state.values()
                          if isinstance(value, torch.Tensor))
    updated = not torch.equal(reference, next(model.parameters()).detach())
    mean = statistics.mean(durations)
    result.update({
        "status": "ok", "batch_size": batch,
        "workload": {"history_frames": config.max_history, "image_shape": [3, 32, 32],
                     "prompt_utf8_bytes": len(prompts[0].encode()), "decoder_steps": decoder.shape[1],
                     "loss": "action CE + reply CE + 0.5 scene CE + 0.03 synthetic prediction MSE",
                     "synthetic_targets": True, "measures_task_competence": False,
                     "timing_includes": "Forward (including input validation and prompt byte encoding), losses, backward, gradient clipping, AdamW update, and CUDA synchronization when applicable.",
                     "timing_excludes": "Model construction, synthetic batch allocation, reply target encoding, environment rendering, dataset creation, evaluation, and checkpoint IO."},
        "warmup_steps": payload["warmup_steps"], "measured_steps": payload["measured_steps"],
        "optimizer_updates": payload["warmup_steps"] + payload["measured_steps"],
        "first_parameter_changed": updated, "measured_step_seconds": durations,
        "mean_step_seconds": mean, "median_step_seconds": statistics.median(durations),
        "examples_per_second": batch / mean, "measured_losses": losses,
        "optimizer_state_tensor_bytes": optimizer_bytes,
        "optimizer_state_scope": "Actual optimizer tensors after updates; excludes weights, gradients, activations and allocator overhead.",
        "host_peak_rss": _peak_rss(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "cuda_peak_scope": "PyTorch allocator peak since before model construction; includes warmup and measurements; excludes non-PyTorch CUDA allocations.",
    })
    return result


def benchmark_suite(
    output: str | Path | None, *, device: str = "cpu", threads: int = 1,
    profiles: Sequence[str] | None = None, batch_size: int = 8,
    warmup_steps: int = 2, measured_steps: int = 3,
    max_parameters: int = 20_000_000, max_seconds: float = 120.0,
) -> dict:
    """Measure profiles sequentially, with a per-process wall time and size cap.

    ``max_seconds`` includes imports and initialization, not just updates.
    Requested CUDA is never silently replaced by CPU. The suite records errors
    and continues to later profiles, then saves all successes and failures.
    """
    for name, value in (("threads", threads), ("batch_size", batch_size),
                        ("measured_steps", measured_steps), ("max_parameters", max_parameters)):
        _positive_int(name, value)
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    if not isinstance(device, str) or not (device == "cpu" or device == "cuda" or
                                            (device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    names = list(DEFAULT_PROFILES) if profiles is None else profiles
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence) or not names:
        raise ValueError("profiles must be a nonempty sequence of profile names")
    if any(not isinstance(name, str) or name not in PROFILES for name in names):
        raise ValueError(f"profiles must be selected from {tuple(PROFILES)}")
    if len(set(names)) != len(names):
        raise ValueError("profiles must not contain duplicates")

    host = _host_info()
    results = []
    started = time.perf_counter()
    for name in names:
        payload = {"profile": name, "config": PROFILES[name], "device": device, "threads": threads,
                   "batch_size": batch_size, "warmup_steps": warmup_steps,
                   "measured_steps": measured_steps, "max_parameters": max_parameters}
        profile_started = time.perf_counter()
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                   OPENBLAS_NUM_THREADS=str(threads))
        # Keep the package importable when run from a different working directory.
        package_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (package_root, env.get("PYTHONPATH"))))
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "brain_in_computer.benchmark", "--worker"],
                input=json.dumps(payload), capture_output=True, text=True, env=env,
                timeout=float(max_seconds), check=False,
            )
            if completed.returncode:
                result = {"profile": name, "status": "error", "returncode": completed.returncode,
                          "error": (completed.stderr or completed.stdout)[-4000:]}
            else:
                result = json.loads(completed.stdout)
        except subprocess.TimeoutExpired:
            result = {"profile": name, "status": "timeout", "max_seconds": float(max_seconds),
                      "error": "Worker terminated at the wall-time cap; no completed measurement is claimed."}
        result["worker_wall_seconds"] = time.perf_counter() - profile_started
        results.append(result)

    report = {
        "schema": "bic-compute-benchmark-v1", "host": host,
        "requested_device": device, "per_profile_timeout_seconds": float(max_seconds),
        "total_wall_seconds": time.perf_counter() - started, "profiles": results,
        "home_reference_budget": {"ram_bytes": 32 * 1024 ** 3, "vram_bytes": 16 * 1024 ** 3,
            "scope": "User-specified reference capacity, not detected hardware or an available-memory measurement."},
        "limitations": [
            "Synthetic short batches measure execution cost, not language competence or learning quality.",
            "These measurements apply only to the recorded host, device, workload and precision.",
            "Each profile uses a fresh process; host RSS includes Python, framework and native allocations.",
            "CUDA allocator peaks and host RSS describe different memory pools and are reported separately.",
            "The parameter cap limits allocated weights; it is not an operating-system memory limit or a fit guarantee.",
            "This bounded suite does not attempt to exhaust RAM/VRAM or establish a maximum trainable model.",
            "No results for the user's RTX 5080 are implied unless that GPU is identified by a measured worker.",
            "No AMP or quantized training is performed. All parameters and floating observations are float32.",
        ],
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", default="runs/compute/benchmark.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=3)
    parser.add_argument("--max-parameters", type=int, default=20_000_000)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    args = parser.parse_args(argv)
    if args.worker:
        try:
            print(json.dumps(_run_profile(json.load(sys.stdin)), allow_nan=False))
        except Exception as error:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        return
    report = benchmark_suite(args.output, device=args.device, threads=args.threads, profiles=args.profiles,
                             batch_size=args.batch_size, warmup_steps=args.warmup_steps,
                             measured_steps=args.measured_steps, max_parameters=args.max_parameters,
                             max_seconds=args.max_seconds)
    for row in report["profiles"]:
        detail = (f"{row['parameter_count']:,} parameters, {row['mean_step_seconds']:.3f} s/update, "
                  f"{row['host_peak_rss']['bytes']} peak resident bytes") if row["status"] == "ok" else row.get("error", row["status"])
        print(f"{row['profile']}: {row['status']} — {detail}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
