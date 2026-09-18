"""Explicit bounded execution of the predetermined two-endpoint gradient check."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

from experiments.foundation_shared_diagnostic import Journal, sha, source_guard, write_json


SCHEMA = "bic-foundation-gradient-run-v1"


def run(repository, output, *, source_manifest, expected_source_sha256, fit_data,
        expected_fit_sha256, max_seconds, device):
    raw = source_manifest.read_bytes()
    if sha(raw) != expected_source_sha256:
        raise ValueError("gradient source manifest digest differs")
    manifest = json.loads(raw)
    guard = lambda: source_guard(repository, manifest)
    guard()
    if device not in ("cpu", "cuda:0"):
        raise ValueError("explicit local device required")
    journal = Journal(output, max_seconds=max_seconds, guard=guard,
        context=dict(diagnostic=SCHEMA, device=device, source_manifest_sha256=expected_source_sha256,
            fit_data_sha256=expected_fit_sha256, states=["curriculum", "mixed"],
            selected_pair_slots=[0, 1, 2, 3], family_forward_allowance=126,
            encoder_episode_allowance=1008, gradient_evaluation_allowance=168))
    journal.value["schema"] = SCHEMA
    journal.flush()
    auth = None
    artifacts = {}
    try:
        def save(name, value):
            artifacts[name] = write_json(output / name, value)
        save("source-manifest.json", manifest)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        from experiments.foundation_diagnostic_inputs import authenticate_inputs, load_state
        from experiments.foundation_gradient_diagnostic import inspect_gradients
        from brain_in_computer.dialogue_student import checkpoint_digest

        runtime = dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__),
            device=device, deterministic=True, threads=1, interop_threads=1, tf32=False)
        if device == "cuda:0":
            if not torch.cuda.is_available():
                raise RuntimeError("declared local GPU is unavailable")
            runtime["gpu"] = torch.cuda.get_device_name(0)
            torch.cuda.reset_peak_memory_stats(0)
        save("runtime.json", runtime)
        auth = journal.perform("authenticate-inputs", lambda: authenticate_inputs(repository),
                               describe=lambda value: value.metadata)
        save("input-authentication.json", auth.metadata)

        def read_fit():
            image = fit_data.read_bytes()
            if sha(image) != expected_fit_sha256:
                raise ValueError("sealed fitting dataset bytes differ")
            banks = json.loads(image)
            save("fit-data-binding.json", dict(path=str(fit_data), sha256=expected_fit_sha256,
                selected_pair_slots=[0, 1, 2, 3], cells=len(banks)))
            return banks
        banks = journal.perform("authenticate-sealed-fit-data", read_fit,
                                describe=lambda value: dict(sha256=expected_fit_sha256, cells=len(value)))
        results = {}
        for endpoint in ("curriculum", "mixed"):
            model = journal.perform(endpoint + "/load-state", lambda: load_state(repository, endpoint, auth).to(device),
                                    describe=lambda value: auth.load_reports[-1])
            weight_pin = checkpoint_digest(model)
            report = journal.perform(endpoint + "/gradient-inspection",
                lambda: inspect_gradients(model, banks, endpoint=endpoint), describe=lambda value: value)
            # Record actual completed work before any subsequent write/check fails.
            results[endpoint] = report
            save(endpoint + ".json", report)
            expected_work = dict(groups=21, family_forwards=63, encoder_episodes=504,
                                 component_gradient_evaluations=84)
            if report.get("status") != "completed" or any(report.get(key + "_" + suffix) != expected
                    for key, expected in expected_work.items() for suffix in ("attempted", "completed")):
                raise RuntimeError("gradient inspection work differs from the fixed endpoint contract")
            if checkpoint_digest(model) != weight_pin or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("gradient inspection changed source learner state")
            del model
            if device == "cuda:0":
                torch.cuda.synchronize(0)
                torch.cuda.empty_cache()
        guard()
        result = dict(schema=SCHEMA, status="completed", states=results, source_manifest_sha256=expected_source_sha256,
            fit_data_sha256=expected_fit_sha256, runtime=runtime, load_reports=auth.load_reports,
            learner_optimizer_updates=0, automatic_promotion=False,
            work={key + "_" + suffix: sum(report[key + "_" + suffix] for report in results.values())
                  for key in ("groups", "family_forwards", "encoder_episodes", "component_gradient_evaluations")
                  for suffix in ("attempted", "completed")},
            scope="Fixed local unpreconditioned gradients before clipping. No AdamW update, candidate adoption or causal benefit claim.")
        if device == "cuda:0":
            result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        save("result.json", result)
        save("artifacts.json", dict(artifacts))
        journal.finish("completed", result_sha256=artifacts["result.json"],
                       artifacts_manifest_sha256=artifacts["artifacts.json"])
        return result
    except BaseException as error:
        # The shared journal knows feature/probe reports; retain gradient work too.
        if hasattr(error, "gradient_report"):
            journal.value["partial_gradient_report"] = error.gradient_report
        journal.finish("interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                       error=repr(error), load_reports=[] if auth is None else auth.load_reports)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--expected-fit-sha256", required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parent.parent, args.output.resolve(),
        source_manifest=args.source_manifest.resolve(), expected_source_sha256=args.expected_source_sha256,
        fit_data=args.fit_data.resolve(), expected_fit_sha256=args.expected_fit_sha256,
        max_seconds=args.max_seconds, device=args.device)


if __name__ == "__main__":
    main()
