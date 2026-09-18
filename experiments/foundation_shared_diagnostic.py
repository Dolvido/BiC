"""One bounded, prospective, local diagnostic; never updates source learners.

The caller supplies a source manifest and its independent SHA256. Every phase
has a durable intent before work. An interrupted phase is never retried here.
Wall limits are checked at phase boundaries, not a preemption guarantee.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time


SCHEMA = "bic-foundation-shared-diagnostic-v1"
STATES = ("initial", "curriculum", "mixed")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value, *, replace=False):
    raw = encoded(value)
    target = path.with_suffix(path.suffix + ".pending") if replace else path
    with target.open("wb" if replace else "xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(target, path)
    return sha(raw)


def source_guard(repository, manifest):
    """Fresh bytes and complete Python membership; no persistent hash cache."""
    files = manifest.get("files")
    if (manifest.get("schema") != SCHEMA or type(files) is not dict or not files
            or manifest.get("python_directories") != ["brain_in_computer", "experiments"]):
        raise ValueError("explicit diagnostic source manifest required")
    actual = {p.relative_to(repository).as_posix()
              for folder in manifest["python_directories"]
              for p in (repository / folder).rglob("*.py")}
    expected = {name for name in files if name.endswith(".py")}
    if actual != expected:
        raise ValueError("repository Python membership changed")
    for name, digest in files.items():
        path = repository / name
        if (Path(name).is_absolute() or ".." in Path(name).parts
                or path.resolve() != path.absolute() or sha(path.read_bytes()) != digest):
            raise ValueError("source identity changed: " + name)


class Journal:
    def __init__(self, output, *, max_seconds, guard, context):
        if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("finite positive wall allowance required")
        output.mkdir(parents=True, exist_ok=False)
        self.output, self.guard = output, guard
        self.started, self.cpu_started = time.monotonic(), time.process_time()
        self.deadline = self.started + max_seconds
        self.value = dict(schema=SCHEMA, status="running", started_utc=now(), pid=os.getpid(),
            max_seconds=max_seconds, wall_limit_scope="Nonpreemptive phase-boundary limit; no automatic retry.",
            context=context, operations=[], learner_optimizer_updates=0, automatic_promotion=False)
        self.flush()

    def flush(self):
        self.value.update(wall_seconds=time.monotonic()-self.started,
                          cpu_seconds=time.process_time()-self.cpu_started)
        write_json(self.output / "execution.json", self.value, replace=True)

    def perform(self, name, function, *, describe=lambda value: None):
        self.guard()
        if time.monotonic() >= self.deadline:
            raise TimeoutError("diagnostic wall allowance reached before " + name)
        entry = dict(name=name, status="running", started_utc=now(),
            interruption_scope="A missing terminal record means physical work in this phase is unknown; do not retry automatically.")
        self.value["operations"].append(entry)
        self.flush()
        started, cpu_started = time.monotonic(), time.process_time()
        try:
            result = function()
            entry["evidence"] = describe(result)
            entry["status"] = "completed"
            return result
        except BaseException as error:
            entry.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                         error=repr(error))
            for attribute in ("diagnostic_report", "probe_report", "completed_probe_reports", "input_report", "receipt"):
                if hasattr(error, attribute):
                    entry[attribute] = getattr(error, attribute)
            raise
        finally:
            entry.update(ended_utc=now(), wall_seconds=time.monotonic()-started,
                         cpu_seconds=time.process_time()-cpu_started)
            self.flush()

    def finish(self, status, **extra):
        self.value.update(status=status, ended_utc=now(), **extra)
        self.flush()


def run(repository, output, *, source_manifest, expected_source_sha256, max_seconds, device):
    raw = source_manifest.read_bytes()
    if sha(raw) != expected_source_sha256:
        raise ValueError("independently supplied source manifest digest differs")
    manifest = json.loads(raw)
    guard = lambda: source_guard(repository, manifest)
    guard()
    if device not in ("cpu", "cuda:0"):
        raise ValueError("explicit cpu or local cuda:0 required")
    journal = Journal(output, max_seconds=max_seconds, guard=guard,
        context=dict(source_manifest_sha256=expected_source_sha256, device=device,
                     states=list(STATES), dataset_roots=dict(fit=916170001, evaluation=916170002),
                     encoder_episode_allowance=6048, probe_fits=12, probe_objective_allowance=12000,
                     generated_reply_turn_allowance=30240))
    artifacts = {}
    auth = None
    try:
        artifacts["source-manifest.json"] = write_json(output / "source-manifest.json", manifest)
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
        from experiments.foundation_diagnostic_data import CELLS, build_pair_of_splits
        from experiments.foundation_diagnostic_features import extract_features, decode_cached_replies
        from experiments.foundation_diagnostic_scoring import score_predictions
        from experiments.foundation_representation_probe import fit_true_and_shuffled
        from brain_in_computer.dialogue_student import checkpoint_digest

        runtime = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
            torch=str(torch.__version__), cuda_runtime=torch.version.cuda,
            cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads(),
            deterministic_algorithms=True, tf32=False, device=device,
            solver_sha256=sha((Path(torch.__file__).resolve().parent / "optim/lbfgs.py").read_bytes()))
        if (runtime["torch"] != "2.11.0+cu128" or runtime["solver_sha256"] !=
                "bce32f11cc29073610d5422a29d55864cbb15cd2a7c847aa5fabbf18ffa6eead"):
            raise RuntimeError("prospectively pinned Torch/solver runtime differs")
        if device == "cuda:0":
            if not torch.cuda.is_available():
                raise RuntimeError("the declared local GPU is unavailable")
            runtime.update(gpu=torch.cuda.get_device_name(0),
                           gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
            torch.cuda.reset_peak_memory_stats(0)
        artifacts["runtime.json"] = write_json(output / "runtime.json", runtime)

        def save_json(name, value):
            artifacts[name] = write_json(output / name, value)

        def save_tensor(name, value):
            with (output / name).open("xb") as stream:
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            artifacts[name] = sha((output / name).read_bytes())

        auth = journal.perform("authenticate-inputs", lambda: authenticate_inputs(repository),
                               describe=lambda value: value.metadata)
        save_json("input-authentication.json", auth.metadata)

        def prepare():
            banks, receipt = build_pair_of_splits(excluded_transcripts=auth.exclusions)
            if (receipt["production_contract"] is not True
                    or any(r["counts"]["accepted_episodes"] != 1008 for r in receipt["roles"].values())):
                raise ValueError("production diagnostic dataset inventory differs")
            for role in ("fit", "evaluation"):
                save_json("dataset-" + role + ".json", banks[role])
            save_json("dataset-admission.json", receipt)
            return banks, receipt

        banks, admission = journal.perform("prepare-and-seal-datasets", prepare, describe=lambda value: value[1])
        results, feature_reports, decoder_reports, probe_reports = {}, [], [], []
        for state in STATES:
            state_dir = output / state
            state_dir.mkdir()
            def restore():
                model = load_state(repository, state, auth)
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                return model.to(device).eval()
            model = journal.perform(state + "/load-frozen-state", restore,
                                    describe=lambda value: auth.load_reports[-1])
            original_digest = checkpoint_digest(model)
            caches, detached = {}, {}
            state_result = dict(weights_sha256=original_digest, roles={}, probes={})
            results[state] = state_result
            for role in ("fit", "evaluation"):
                cache = journal.perform(state + "/features-" + role,
                    lambda: extract_features(model, banks[role], role=role, cell_inventory=CELLS, batch_size=32),
                    describe=lambda value: value.report)
                caches[role] = cache
                feature_reports.append(cache.report)
                save_tensor(state + "/features-" + role + ".pt",
                    dict(metadata=cache.metadata, tensors=cache.tensors, report=cache.report, sha256=cache.sha256))
                values = cache.tensors
                detached[role] = values
                state_result["roles"][role] = dict(cache_sha256=cache.sha256)
                predictions = dict(action=values["action_logits"].argmax(dim=1).tolist(),
                                   first_byte=values["bos_logits"].argmax(dim=1).tolist())
                state_result["roles"][role]["original"] = journal.perform(state + "/score-original-" + role,
                    lambda: {kind: score_predictions(cache.metadata, values["labels"].tolist(), prediction, modality=kind)
                             for kind, prediction in predictions.items()},
                    describe=lambda value: dict(cached_decisions=2*len(values["labels"]), neural_work=0))

            decoded = journal.perform(state + "/generate-cached-evaluation-replies",
                lambda: decode_cached_replies(model, caches["evaluation"], batch_size=64),
                describe=lambda value: value["report"])
            decoder_reports.append(decoded["report"])
            save_tensor(state + "/decoded-replies.pt", decoded)
            expected_first = detached["evaluation"]["bos_logits"].argmax(dim=1)
            differences = expected_first.ne(decoded["first_tokens"])
            state_result["bos_generation_agreement"] = dict(equal=int((~differences).sum()),
                total=len(differences), differing_rows=differences.nonzero().flatten().tolist(),
                scope="Independent cached decoder batch shape; numerical disagreement is reported, never replaced.")
            state_result["roles"]["evaluation"]["original"]["exact_reply"] = journal.perform(state + "/score-replies",
                lambda: score_predictions(caches["evaluation"].metadata, detached["evaluation"]["labels"].tolist(),
                                          decoded["tokens"].tolist(), modality="exact_reply"),
                describe=lambda value: dict(replies=len(detached["evaluation"]["labels"]), neural_work=0))
            if state_result["roles"]["evaluation"]["original"]["exact_reply"]["overall"]["all_turns"]["correct"] != int(decoded["reply_correct"].sum()):
                raise RuntimeError("pure token scoring and original exact-reply semantics disagree")
            if checkpoint_digest(model) != original_digest or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("source learner changed during diagnostic work")
            del model, decoded
            if device == "cuda:0":
                torch.cuda.synchronize(0)
                torch.cuda.empty_cache()

            for representation in ("eos", "production"):
                fitted = journal.perform(state + "/fit-probes-" + representation,
                    lambda: fit_true_and_shuffled(**caches["fit"].probe_inputs(representation)),
                    describe=lambda value: {name: r.report for name, r in value.items()})
                save_tensor(state + "/probe-" + representation + ".pt",
                    {name: dict(report=value.report, normalization=value.normalization, coefficients=value.coefficients)
                     for name, value in fitted.items()})
                state_result["probes"][representation] = {}
                for kind, probe in fitted.items():
                    probe_reports.append(probe.report)
                    record = dict(fit=probe.report, scores={})
                    state_result["probes"][representation][kind] = record
                    if probe.coefficients is not None:
                        for role in ("fit", "evaluation"):
                            predictions = journal.perform(state + "/predict-" + representation + "-" + kind + "-" + role,
                                lambda: probe.predict(detached[role][representation]).tolist(),
                                describe=lambda value: dict(detached_probe_forward_rows=len(value), source_learner_work=0))
                            record["scores"][role] = score_predictions(caches[role].metadata,
                                detached[role]["labels"].tolist(), predictions, modality="action")
                            save_json(state + "/prediction-" + representation + "-" + kind + "-" + role + ".json", predictions)
            save_json(state + "/result.json", state_result)
            del caches, detached, fitted

        guard()
        work = dict(encoder_episodes_attempted=sum(r["encoder_episodes_attempted"] for r in feature_reports),
            encoder_episodes_completed=sum(r["encoder_episodes_completed"] for r in feature_reports),
            bos_decoder_row_steps=sum(r["bos_decoder_row_steps_completed"] for r in feature_reports),
            generated_reply_turns=sum(r["replies"] for r in decoder_reports),
            generated_decoder_row_steps=sum(r["generated_decoder_row_steps_completed"] for r in decoder_reports),
            probe_fits=len(probe_reports), probe_objectives_started=sum(r["objective_evaluations_started"] for r in probe_reports),
            probe_objectives_completed=sum(r["objective_evaluations_completed"] for r in probe_reports),
            probe_backward_evaluations=sum(r["backward_evaluations_completed"] for r in probe_reports),
            probe_objective_supervised_turn_exposures=sum(r["fitting_turns"]*r["objective_evaluations_completed"] for r in probe_reports),
            detached_probe_prediction_rows=sum(op.get("evidence", {}).get("detached_probe_forward_rows", 0)
                for op in journal.value["operations"] if isinstance(op.get("evidence"), dict)),
            probe_solver_iterations=sum(r["probe_solver_iterations"] for r in probe_reports),
            source_learner_backward_evaluations=0, source_learner_optimizer_updates=0)
        if (work["encoder_episodes_attempted"] != 6048 or work["encoder_episodes_completed"] != 6048
                or work["generated_reply_turns"] != 30240 or work["probe_fits"] != 12
                or work["probe_objectives_started"] > 12000):
            raise RuntimeError("diagnostic work differs from the prospective allowance")
        completed_status = "completed" if all(r["status"] in ("converged", "normal_return_unconverged")
                                               for r in probe_reports) else "partial"
        result = dict(schema=SCHEMA, status=completed_status, states=results, work=work,
            source_manifest_sha256=expected_source_sha256, runtime=runtime,
            feature_reports=feature_reports, decoder_reports=decoder_reports, load_reports=auth.load_reports,
            automatic_promotion=False,
            timing_scope="Execution journal records wall/CPU seconds per phase, including validation and copies. GPU work is counted by episode/recurrent operations and peak memory; no active-kernel GPU-hours estimate is claimed.",
            scope="Diagnostic readouts on new samples of a known finite grammar. No source learner change or general-intelligence claim.")
        if device == "cuda:0":
            result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        save_json("result.json", result)
        save_json("artifacts.json", dict(artifacts))
        journal.finish(completed_status, result_sha256=artifacts["result.json"], work=work,
                       artifacts_manifest_sha256=artifacts["artifacts.json"])
        return result
    except BaseException as error:
        journal.finish("interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                       error=repr(error), load_reports=[] if auth is None else auth.load_reports)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parent.parent
    run(repository, args.output.resolve(), source_manifest=args.source_manifest.resolve(),
        expected_source_sha256=args.expected_source_sha256, max_seconds=args.max_seconds, device=args.device)


if __name__ == "__main__":
    main()
