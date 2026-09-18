"""One guarded CPU-only data preparation, with a durable enclosing receipt."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PHASE_SECONDS = dict(history=120, admission=600, canonical_evidence=600,
                     layouts=1800, evaluation=1800, publish=180)


def native(path):
    value = str(Path(path).absolute())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\"+value[2:] if value.startswith("\\\\") else "\\\\?\\"+value
    return Path(value)


def write(path, value):
    with native(path).open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False)+"\n").encode())
        stream.flush(); os.fsync(stream.fileno())


def run(output, expected_source_sha256):
    start, cpu = time.monotonic(), time.process_time()
    output = native(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="bic-foundation-layout-preparation-process-v1", status="running",
        process_id=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
        data_source_sha256=expected_source_sha256, guard_attempts={}, phase_seconds=PHASE_SECONDS,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(output/"invocation.json", report)
    error = None
    try:
        source = native(ROOT/"experiments/foundation_layout_data.py").read_bytes()
        if hashlib.sha256(source).hexdigest() != expected_source_sha256:
            raise ValueError("explicit reviewed data source digest differs")
        import torch
        report["torch_version"] = str(torch.__version__)
        report["cuda_initialized_before"] = torch.cuda.is_initialized()
        if report["cuda_initialized_before"]:
            raise RuntimeError("fresh CPU preparation process required")

        def forbid(name):
            report["guard_attempts"][name] = 0
            def blocked(*args, **kwargs):
                report["guard_attempts"][name] += 1
                raise AssertionError("forbidden data-preparation operation: "+name)
            return blocked

        with ExitStack() as stack:
            for obj, key, label in (
                (torch.nn.Module, "__init__", "module_construction"),
                (torch.nn.Module, "_call_impl", "module_forward"),
                (torch.Tensor, "backward", "tensor_backward"),
                (torch.autograd, "backward", "autograd_backward"),
                (torch.optim.Optimizer, "__init__", "optimizer_construction"),
                (torch.optim.AdamW, "step", "adamw_step"),
                (torch.cuda, "_lazy_init", "cuda_initialization")):
                stack.enter_context(patch.object(obj, key, side_effect=forbid(label)))
            from experiments import foundation_layout_data as data
            if set(data.PHASES) != set(PHASE_SECONDS):
                raise ValueError("reviewed preparation phase names differ")
            def progress(event):
                print(json.dumps(event, sort_keys=True, allow_nan=False), flush=True)
            manifest = data.prepare(output/"data", phase_seconds=PHASE_SECONDS, progress=progress)
            report["manifest_sha256"] = data.file_hash(output/"data/manifest.json")
            report["preparation_sha256"] = data.file_hash(output/"data/preparation.json")
            if data.source_hashes() != manifest["source_sha256"]:
                raise ValueError("data sources changed after publication")
        report["cuda_initialized_after"] = torch.cuda.is_initialized()
        if report["cuda_initialized_after"] or any(report["guard_attempts"].values()):
            raise RuntimeError("CPU-only preparation guard failed")
        report["status"] = "completed"
    except BaseException as caught:
        error = caught
        report.update(status="interrupted" if isinstance(caught, (KeyboardInterrupt, SystemExit)) else "failed",
                      error=repr(caught), traceback=traceback.format_exc())
    finally:
        report.update(ended_utc=datetime.now(timezone.utc).isoformat(),
            wall_seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu,
            timing_scope="Enclosing call including Torch/import and data preparation; final receipt write/process startup excluded.")
        write(output/"receipt.json", report)
    print(json.dumps({k: report[k] for k in ("status", "wall_seconds", "cpu_seconds", "guard_attempts")}), flush=True)
    if error is not None:
        raise error
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    run(args.output, args.source_sha256)
