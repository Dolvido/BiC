"""One-shot separate-process foundation execution probe, never a learning study.

CLI: prepare --output NEW --width W [--admission]; then worker --output NEW --name NAME for
uninterrupted, split-first, resumed, repeat; finally verify --output NEW.
Default CUDA work is 16 + 8 + 8 + 16 = 48 updates / 4,608 episodes. No retries,
evaluation, architecture choice, or automatic promotion. A hard-killed worker
leaves its claim and started-step journal; unreported physical work stays unknown.

Programmatic prepare(device='cpu', config=..., steps=...) exists only for tiny
engineering tests. Its receipt can NEVER serve as a successful CUDA load_proof.
Timing is the only checkpoint field excluded from trajectory comparisons; exact
midpoint reload includes timing too. Worker walls include setup/restoration and
durable writes; step times include materialization and must not be added to it.
Admission mode protects original color pair zero in every executed prefix bundle
and proves the repaired naming-only path. Older proof directories stay intact
and describe their original source revision, not this revised workload.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import uuid

import torch

from brain_in_computer.learning_student import _check_finite_tree
from experiments.execution_profile import configure_strict_profile, runtime_profile
from experiments.foundation_plan import build_plan, materialize_pair, validate_plan
from experiments.foundation_training import FoundationTrainer, replay_evidence, source_hashes as trainer_sources
from experiments.sequence_student import SequenceConfig


SCHEMA = "bic-foundation-runtime-probe-v2"
WORKERS = ("uninterrupted", "split-first", "resumed", "repeat")
ROOT = Path(__file__).resolve().parents[1]
PROCESS_INSTANCE = uuid.uuid4().hex  # Once per imported process, not once per call.


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_hashes():
    return {**trainer_sources(), **{name: _file_hash(ROOT / name) for name in
        ("experiments/execution_profile.py", "experiments/foundation_runtime_probe.py",
         "experiments/foundation_admission.py", "experiments/foundation_evidence.py")}}


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_json(value).encode()); stream.flush(); os.fsync(stream.fileno())


def _save(path, value):
    with Path(path).open("xb") as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _append(path, value):
    with Path(path).open("ab") as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush(); os.fsync(stream.fileno())


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _error(error):
    return {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}


def _runtime(device):
    if torch.get_num_threads() != 1:
        raise ValueError("probe requires explicitly configured one CPU thread")
    if device == "cpu":
        return {"schema": "bic-cpu-probe-engineering-only", "device": "cpu",
                "torch_version": str(torch.__version__), "cpu_threads": 1}
    if device != "cuda:0":
        raise ValueError("probe supports explicit cuda:0 or test-only cpu")
    if torch.get_num_interop_threads() != 1:
        raise ValueError("strict CUDA probe requires explicitly configured one inter-op thread")
    configure_strict_profile()  # Fails if this process already initialized CUDA.
    return runtime_profile(device)


def _runtime_after(device):
    return _runtime(device) if device == "cpu" else runtime_profile(device)


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(_same(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _normalized(payload):
    value = dict(payload)
    if set(value.get("timing", {})) != {"retained_step_seconds", "materialization_seconds_included_in_step"}:
        raise ValueError("unexpected timing fields cannot be ignored")
    for name, number in value["timing"].items():
        _nonnegative(number, name)
    if value["timing"]["materialization_seconds_included_in_step"] > value["timing"]["retained_step_seconds"]:
        raise ValueError("checkpoint materialization is not a subset of step time")
    value.pop("timing")
    return value


def _difference(a, b):
    """Exact component comparisons plus tensor differences, without inference."""
    _check_finite_tree(a, "reference checkpoint")
    _check_finite_tree(b, "compared checkpoint")
    left, right = _normalized(a), _normalized(b)
    summary = {"exact": _same(left, right), "components": {
        key: key in right and _same(value, right[key]) for key, value in left.items()}}
    for component in ("weights", "optimizer"):
        totals = {"elements": 0, "different_elements": 0, "max_absolute": 0., "squared": 0.}
        def visit(x, y):
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.shape == y.shape:
                delta = x.double() - y.double()
                totals["elements"] += x.numel(); totals["different_elements"] += int(torch.count_nonzero(delta))
                if delta.numel():
                    totals["max_absolute"] = max(totals["max_absolute"], float(delta.abs().max()))
                    totals["squared"] += float(delta.square().sum())
            elif isinstance(x, dict) and isinstance(y, dict):
                for key in set(x) & set(y):
                    visit(x[key], y[key])
            elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
                for v, w in zip(x, y):
                    visit(v, w)
        visit(left[component], right.get(component))
        totals["rms"] = math.sqrt(totals.pop("squared") / totals["elements"]) if totals["elements"] else 0.
        summary[component] = totals
    return summary


def _synthetic_protected(base_plan, steps):
    from experiments.realization_banks import transcript_digest
    return sorted({transcript_digest(row)
        for bundle_id in base_plan["schedules"]["mixed"][:steps]
        for row in materialize_pair(base_plan, bundle_id, "color", 0)})


def _admission_coverage(plan, steps, protected, receipt_sha256):
    prefix = plan["schedules"]["mixed"][:steps]
    attempts = plan.get("admission", {}).get("realization_attempts", {})
    refs = [key.split("/") for key in attempts if int(key.split("/")[0]) in prefix]
    enabled = receipt_sha256 is not None
    if enabled and any(f"{bundle}/color/0" not in attempts for bundle in prefix):
        raise ValueError("admission probe must override color pair zero in every executed bundle")
    return {"mode": "naming_overrides" if enabled else "none",
        "executed_prefix_overridden_pairs": len(refs),
        "executed_prefix_overridden_bundles": len({int(row[0]) for row in refs}),
        "executed_prefix_by_family": {family: sum(row[1] == family for row in refs)
            for family in ("color", "count", "switch")},
        "before_midpoint_overridden_pairs": sum(int(row[0]) in prefix[:steps//2] for row in refs),
        "after_midpoint_overridden_pairs": sum(int(row[0]) in prefix[steps//2:] for row in refs),
        "protected_transcripts_count": len(protected), "receipt_sha256": receipt_sha256}


def prepare(directory, *, width=None, device="cuda:0", config=None, steps=16, micro_batch_size=32,
            admission=False):
    """Freeze inputs before worker work; an existing directory is never reused."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory / "prepare-started.json", {"schema": SCHEMA, "started_utc": _now(), "pid": os.getpid()})
    started = time.monotonic()
    try:
        if type(admission) is not bool:
            raise ValueError("admission must be an explicit boolean")
        if type(steps) is not int or steps < 2 or steps % 2:
            raise ValueError("even steps >=2 required")
        if device == "cuda:0":
            if width not in (96, 192, 256) or config is not None or steps != 16 or micro_batch_size != 32:
                raise ValueError("CUDA declaration requires width96/192/256 and the fixed16-update micro32 workload")
            config = SequenceConfig(width=width, layers=4, heads=4, feedforward=width * 4, max_turns=12)
        elif device == "cpu":
            if type(config) is not SequenceConfig:
                raise ValueError("CPU engineering requires an explicit small configuration")
        else:
            raise ValueError("unsupported probe device")
        profile = _runtime(device)
        sources = source_hashes()
        from experiments.capacity_study import source_hashes as preserved_sources
        preserved = preserved_sources()  # Source files only; no study data/results.
        plan = build_plan(seed=501000001, stage_updates=10, final_updates=6,
                          micro_batch_size=micro_batch_size, rehearsal_every=2, ordering_seed=0)
        if steps > len(plan["schedules"]["mixed"]):
            raise ValueError("probe exceeds the immutable plan")
        selected = [plan["bundles"][str(i)] for i in plan["schedules"]["mixed"][:steps]]
        coverage = {"depths": sorted({row["depth"] for row in selected}), "turns": sorted({row["turns"] for row in selected})}
        if device == "cuda:0" and coverage != {"depths": list(range(6)), "turns": [8, 10, 12]}:
            raise ValueError("declared CUDA prefix lacks required depth/length coverage")
        protected, base_sha256, admission_sha256 = [], None, None
        if admission:
            from experiments.foundation_admission import repair_plan
            _write(directory / "base-plan.json", plan)
            base_sha256 = _file_hash(directory / "base-plan.json")
            protected = _synthetic_protected(plan, steps)
            plan, admission_receipt = repair_plan(plan, protected_transcripts=protected)
            _write(directory / "admission.json", admission_receipt)
            admission_sha256 = _file_hash(directory / "admission.json")
        _write(directory / "protected.json", protected)
        admitted = _admission_coverage(plan, steps, protected, admission_sha256)
        _write(directory / "plan.json", plan)
        trainer = FoundationTrainer(plan, "mixed", seed=5101, config=config, device="cpu",
                                    protected_transcripts=protected)
        _save(directory / "initial.pt", trainer.snapshot())
        for name, digest in sources.items():
            source = ROOT / name
            destination = directory / "sources" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as target:
                target.write(source.read_bytes())
            if _file_hash(destination) != digest:
                raise RuntimeError("source changed during preparation")
        protocol = {"schema": SCHEMA, "device": device, "config": asdict(config), "model_seed": 5101,
            "learning_rate": .001, "order": "mixed", "steps": steps, "midpoint": steps // 2,
            "micro_batch_size": micro_batch_size, "planned_physical_updates": 3 * steps,
            "planned_episode_exposures": 3 * steps * 3 * micro_batch_size,
            "execution_profile": profile, "source_sha256": sources, "preserved_source_sha256": preserved,
            "plan_sha256": _file_hash(directory / "plan.json"), "initial_sha256": _file_hash(directory / "initial.pt"),
            "workers": list(WORKERS), "coverage": coverage, "protected_transcripts": protected,
            "protected_sha256": _file_hash(directory / "protected.json"),
            "base_plan_sha256": base_sha256, "admission": admitted,
            "comparison_ignored_fields": ["timing"], "evaluation_performed": False, "automatic_promotion": False,
            "scope": "Separate-process local execution repeatability, not capability or cross-machine guarantee."}
        if source_hashes() != sources or preserved_sources() != preserved:
            raise RuntimeError("sources changed before freeze")
        _write(directory / "protocol.json", protocol)
        _write(directory / "prepared.json", {"protocol_sha256": _file_hash(directory / "protocol.json"),
            "preparation_seconds": time.monotonic() - started, "completed_utc": _now()})
        return protocol
    except BaseException as error:
        _write(directory / "prepare-failed.json", {"error": _error(error), "wall_seconds": time.monotonic() - started})
        raise


def _inputs(directory):
    protocol = _read(directory / "protocol.json")
    if protocol.get("schema") != SCHEMA or _read(directory / "prepared.json")["protocol_sha256"] != _file_hash(directory / "protocol.json"):
        raise ValueError("frozen probe protocol identity differs")
    if source_hashes() != protocol["source_sha256"]:
        raise ValueError("probe source closure differs")
    for name, digest in protocol["source_sha256"].items():
        if _file_hash(directory / "sources" / name) != digest:
            raise ValueError("frozen source snapshot differs")
    for name, digest in protocol["preserved_source_sha256"].items():
        if _file_hash(ROOT / name) != digest:
            raise ValueError("preserved capacity source differs")
    for filename, key in (("plan.json", "plan_sha256"), ("initial.pt", "initial_sha256")):
        if _file_hash(directory / filename) != protocol[key]:
            raise ValueError("frozen plan/initial state differs")
    plan = _read(directory / "plan.json"); validate_plan(plan)
    if _file_hash(directory / "protected.json") != protocol["protected_sha256"]:
        raise ValueError("frozen protected transcript artifact differs")
    protected = _read(directory / "protected.json")
    from experiments.foundation_evidence import transcript_set
    if protected != sorted(transcript_set(protected)) or protected != protocol["protected_transcripts"]:
        raise ValueError("canonical protected transcript binding differs")
    admitted = protocol["admission"]
    if admitted["mode"] == "naming_overrides":
        from experiments.foundation_admission import authenticate_admission
        if (_file_hash(directory / "base-plan.json") != protocol["base_plan_sha256"]
                or _file_hash(directory / "admission.json") != admitted["receipt_sha256"]):
            raise ValueError("frozen admission artifacts differ")
        base = _read(directory / "base-plan.json")
        validate_plan(base)
        if protected != _synthetic_protected(base, protocol["steps"]):
            raise ValueError("protected set does not reserve the declared original prefix pairs")
        authenticate_admission(base, plan, protected_transcripts=protected,
                               receipt=_read(directory / "admission.json"))
    elif (admitted["mode"] != "none" or protected or "admission" in plan
          or protocol["base_plan_sha256"] is not None or admitted["receipt_sha256"] is not None):
        raise ValueError("default probe unexpectedly contains admission state")
    if admitted != _admission_coverage(plan, protocol["steps"], protected, admitted["receipt_sha256"]):
        raise ValueError("admitted prefix coverage differs")
    return protocol, plan


def _artifact_hashes(directory, exclude=()):
    return {path.relative_to(directory).as_posix(): _file_hash(path) for path in sorted(directory.rglob("*"))
            if path.is_file() and path.relative_to(directory).as_posix() not in exclude}


def _worker_receipt(directory, name):
    destination = directory / name
    record = _read(destination / "result.json")
    if record["status"] != "completed":
        raise ValueError(f"worker {name} did not complete; no later work allowed")
    if record["artifact_sha256"] != _artifact_hashes(destination, ("result.json",)):
        raise ValueError("worker artifact identity differs")
    return record


def worker(directory, name):
    directory = Path(directory).resolve()
    if name not in WORKERS:
        raise ValueError("unknown worker")
    destination = directory / name
    destination.mkdir(exist_ok=False)  # An interrupted or failed claim is permanent.
    started = time.monotonic()
    record = {"schema": SCHEMA, "status": "running", "name": name, "started_utc": _now(),
              "process": {"pid": os.getpid(), "instance": PROCESS_INSTANCE}, "parents": {}}
    _write(destination / "started.json", record)
    trainer, protocol = None, None
    try:
        protocol, plan = _inputs(directory)
        record["protocol_sha256"] = _file_hash(directory / "protocol.json")
        for previous in WORKERS[:WORKERS.index(name)]:
            prior = _worker_receipt(directory, previous)
            if prior["process"]["instance"] == PROCESS_INSTANCE:
                raise ValueError("each worker must run in a separate process")
            record["parents"][f"{previous}/result.json"] = _file_hash(directory / previous / "result.json")
        profile = _runtime(protocol["device"])
        if profile != protocol["execution_profile"]:
            raise ValueError("worker execution profile differs from frozen profile")
        record["execution_profile"] = profile
        if protocol["device"] == "cuda:0":
            torch.cuda.reset_peak_memory_stats()
        options = dict(seed=protocol["model_seed"], config=SequenceConfig(**protocol["config"]),
                       learning_rate=protocol["learning_rate"], device=protocol["device"],
                       protected_transcripts=protocol["protected_transcripts"])
        if name == "resumed":
            parent = directory / "split-first" / "final.pt"
            record["parents"]["split-first/final.pt"] = _file_hash(parent)
            payload = _load(parent)
            trainer = FoundationTrainer(plan, "mixed", payload=payload, **options)
            loaded = trainer.snapshot(); _save(destination / "loaded-midpoint.pt", loaded)
            record["load_at_midpoint_exact"] = _same(payload, loaded)
            if not record["load_at_midpoint_exact"]:
                raise ValueError("complete midpoint reload differs")
        else:
            trainer = FoundationTrainer(plan, "mixed", **options)
            initial = trainer.snapshot(); _save(destination / "initial.pt", initial)
            if not _same(initial, _load(directory / "initial.pt")):
                raise ValueError("worker differs from prospectively frozen initialization")
        record["start_cursor"] = trainer.cursor
        target = protocol["midpoint"] if name == "split-first" else protocol["steps"]
        while trainer.cursor < target:
            _append(destination / "steps.jsonl", {"event": "started", "cursor": trainer.cursor,
                "bundle_id": plan["schedules"]["mixed"][trainer.cursor], "utc": _now()})
            try:
                report = trainer.step()
            except BaseException:
                _append(destination / "steps.jsonl", {"event": "reported", "report": copy.deepcopy(trainer.last_report)})
                raise
            _append(destination / "steps.jsonl", {"event": "reported", "report": report})
            if trainer.cursor == protocol["midpoint"]:
                _save(destination / "midpoint.pt", trainer.snapshot())
        _save(destination / "final.pt", trainer.snapshot())
        record.update(status="completed", final_cursor=trainer.cursor, setup_seconds=trainer.setup_seconds,
                      restore_seconds=trainer.last_restore_seconds, parameters=sum(p.numel() for p in trainer.model.parameters()))
        record["execution_profile_after"] = _runtime_after(protocol["device"])
        if record["execution_profile_after"] != profile:
            raise ValueError("profile drifted during worker")
        _inputs(directory)
    except BaseException as error:
        record.update(status="failed", error=_error(error))
        if trainer is not None:
            record["last_trainer_report"] = copy.deepcopy(trainer.last_report)
    finally:
        if protocol is not None and protocol["device"] == "cuda:0":
            try:
                record["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                record["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
            except BaseException as error:
                record["memory_measurement_error"] = _error(error)
        record["wall_seconds"] = time.monotonic() - started
        record["completed_utc"] = _now()
        record["artifact_sha256"] = _artifact_hashes(destination)
        _write(destination / "result.json", record)
    return record


def _journal(directory, name):
    path = directory / name / "steps.jsonl"
    return [_read_line(line) for line in path.read_text(encoding="utf8").splitlines()] if path.exists() else []


def _read_line(line):
    return json.loads(line)


def _physical(directory):
    reports, dangling = [], []
    for name in WORKERS:
        events = _journal(directory, name)
        reports.extend(row["report"] for row in events if row.get("event") == "reported" and row.get("report") is not None)
        if len(events) % 2:
            dangling.append(name)
    return {"known_completed_optimizer_updates": sum(r.get("physical_optimizer_updates") or 0 for r in reports),
        "optimizer_completion_unknown_attempts": sum(r.get("physical_optimizer_updates") is None for r in reports),
        "unreported_started_attempts": dangling,
        "returned_candidate_episodes": sum(r.get("drawn_episode_exposures") or 0 for r in reports),
        "partial_materialization_unknown_attempts": sum(r.get("drawn_episode_exposures") is None for r in reports),
        "neural_attempted_episodes": sum(r.get("neural_attempted_episode_exposures", 0) for r in reports),
        "completed_microbatch_episodes": sum(r.get("completed_microbatch_episode_exposures", 0) for r in reports),
        "reported_step_seconds": sum(r.get("step_seconds", 0.) for r in reports),
        "reported_materialization_seconds_included": sum(r.get("materialization_seconds", 0.) for r in reports),
        "scope": "Known reported work only; dangling starts and interrupted materialization/optimizer attempts remain uncertain."}


def _verified(directory):
    protocol, plan = _inputs(directory)
    workers = {name: _worker_receipt(directory, name) for name in WORKERS}
    if len({r["process"]["instance"] for r in workers.values()}) != len(WORKERS):
        raise ValueError("workers did not use distinct processes")
    replay = replay_evidence(plan, "mixed", protocol["steps"],
                             protected_transcripts=protocol["protected_transcripts"], include_bundles=True)
    for name, record in workers.items():
        if (record["protocol_sha256"] != _file_hash(directory / "protocol.json")
                or record["execution_profile"] != protocol["execution_profile"]
                or record["execution_profile_after"] != protocol["execution_profile"]):
            raise ValueError("worker source/profile/recipe parent differs")
        _nonnegative(record["wall_seconds"], "worker wall time")
        for path, digest in record["parents"].items():
            if _file_hash(directory / path) != digest:
                raise ValueError("worker parent changed")
        start = protocol["midpoint"] if name == "resumed" else 0
        end = protocol["midpoint"] if name == "split-first" else protocol["steps"]
        if record["start_cursor"] != start or record["final_cursor"] != end:
            raise ValueError("worker cursor differs")
        events = _journal(directory, name)
        if len(events) != 2 * (end - start):
            raise ValueError("worker has incomplete or extra step journal")
        for cursor in range(start, end):
            begun, reported = events[2 * (cursor - start):2 * (cursor - start) + 2]
            expected = replay["bundles"][cursor]
            row = reported.get("report", {})
            for item, keys in ((begun, ("cursor", "bundle_id")),
                               (row, ("cursor", "bundle_id", "physical_optimizer_updates", "retained_optimizer_updates"))):
                if any(type(item.get(key)) is not int for key in keys):
                    raise ValueError("journal integer identity/count types differ")
            if (begun.get("event") != "started" or begun.get("cursor") != cursor
                    or begun.get("bundle_id") != expected["bundle_id"] or reported.get("event") != "reported"
                    or row.get("cursor") != cursor + 1 or row.get("bundle_id") != expected["bundle_id"]
                    or row.get("physical_optimizer_updates") != 1 or row.get("retained_optimizer_updates") != 1):
                raise ValueError("journal cursor/update identity differs")
            for key in ("wall_seconds", "step_seconds", "materialization_seconds", "loss", "action_loss", "reply_loss", "observation_language_loss"):
                _nonnegative(row[key], key)
            if not row["materialization_seconds"] <= row["step_seconds"] <= row["wall_seconds"]:
                raise ValueError("journal timing scopes disagree")
            for key in ("drawn_microbatches", "neural_attempted_microbatches", "completed_microbatches"):
                if type(row.get(key)) is not int or row[key] != 3:
                    raise ValueError("journal microbatch counts differ")
            for key in ("drawn_episode_exposures", "neural_attempted_episode_exposures", "completed_microbatch_episode_exposures"):
                if type(row.get(key)) is not int or row[key] != 3 * protocol["micro_batch_size"]:
                    raise ValueError("journal episode counts differ")
            micro = row.get("microbatches", [])
            if [m.get("family") for m in micro] != ["color", "count", "switch"]:
                raise ValueError("journal family order differs")
            for m in micro:
                item = expected["families"][m["family"]]
                if (m["rows_sha256"] != item["rows_sha256"] or m["recipes_sha256"] != item["recipes_sha256"]
                        or m["exposures"] != item["exposures"] or m["depth"] != expected["depth"] or m["turns"] != expected["turns"]):
                    raise ValueError("journal canonical lessons/exposures differ")
    initial = _load(directory / "initial.pt")
    midpoint = _load(directory / "uninterrupted/midpoint.pt")
    endpoint = _load(directory / "uninterrupted/final.pt")
    checker = FoundationTrainer(plan, "mixed", seed=protocol["model_seed"], config=SequenceConfig(**protocol["config"]),
                                learning_rate=protocol["learning_rate"], device="cpu",
                                protected_transcripts=protocol["protected_transcripts"])
    for payload in (initial, midpoint, endpoint):
        checker.restore(payload)  # Full weights/AdamW/input replay validation, no updates.
    if midpoint["cursor"] != protocol["midpoint"] or endpoint["cursor"] != protocol["steps"]:
        raise ValueError("saved checkpoint cursor differs")
    comparisons = {"midpoint": _difference(midpoint, _load(directory / "split-first/final.pt")),
                   "resumed": _difference(endpoint, _load(directory / "resumed/final.pt")),
                   "repeat": _difference(endpoint, _load(directory / "repeat/final.pt"))}
    flags = {"initial_state_equal": all(_same(initial, _load(directory / name / "initial.pt")) for name in ("uninterrupted", "split-first", "repeat")),
        "midpoint_equal": comparisons["midpoint"]["exact"],
        "load_at_midpoint_exact": _same(_load(directory / "split-first/final.pt"), _load(directory / "resumed/loaded-midpoint.pt")),
        "resumed_endpoint_equal": comparisons["resumed"]["exact"], "uninterrupted_repeat_equal": comparisons["repeat"]["exact"],
        "canonical_stream_exact": _same(endpoint["evidence"], replay["evidence"])}
    physical = _physical(directory)
    if (physical["known_completed_optimizer_updates"] != protocol["planned_physical_updates"]
            or physical["completed_microbatch_episodes"] != protocol["planned_episode_exposures"]
            or physical["optimizer_completion_unknown_attempts"] or physical["unreported_started_attempts"]):
        raise ValueError("physical work differs from declared bounded workload")
    return {"schema": SCHEMA, "status": ("passed" if protocol["device"] == "cuda:0" else "passed_cpu_engineering_only")
            if all(flags.values()) else "failed_equality", "flags": flags, "comparisons": comparisons,
        "execution_profile": protocol["execution_profile"], "config": protocol["config"],
        "admission": protocol["admission"],
        "learning_rate": protocol["learning_rate"],
        "source_sha256": protocol["source_sha256"], "preserved_source_sha256": protocol["preserved_source_sha256"],
        "sources_unchanged": True, "coverage": protocol["coverage"], "physical_updates": physical["known_completed_optimizer_updates"],
        "physical_episode_exposures": physical["completed_microbatch_episodes"], "physical": physical,
        "worker_wall_seconds": sum(row["wall_seconds"] for row in workers.values()),
        "peak_cuda_allocated_bytes": max((row.get("peak_cuda_allocated_bytes", 0) for row in workers.values()), default=0),
        "workers": {name: {key: value for key, value in row.items() if key != "artifact_sha256"} for name, row in workers.items()},
        "protocol_sha256": _file_hash(directory / "protocol.json"), "evaluation_performed": False, "automatic_promotion": False}


def verify(directory):
    """Write a one-shot receipt, preserving failed or incomplete evidence as-is."""
    directory = Path(directory).resolve()
    _write(directory / "verification-started.json", {"started_utc": _now(), "pid": os.getpid()})
    started = time.monotonic()
    try:
        record = _verified(directory)
    except BaseException as error:
        record = {"schema": SCHEMA, "status": "failed", "error": _error(error), "evaluation_performed": False,
                  "automatic_promotion": False}
        try:
            record["physical"] = _physical(directory)
        except BaseException as accounting_error:
            record["accounting_error"] = _error(accounting_error)
    record["verification_seconds"] = time.monotonic() - started
    record["completed_utc"] = _now()
    record["artifact_sha256"] = _artifact_hashes(directory, ("probe.json",))
    _write(directory / "probe.json", record)
    return record


def load_proof(directory, config):
    """Authenticate a strict CUDA receipt for an exact prospective model config."""
    directory = Path(directory).resolve()
    record = _read(directory / "probe.json")
    expected_config = asdict(config) if type(config) is SequenceConfig else config
    if (record.get("status") != "passed" or record.get("config") != expected_config
            or record.get("learning_rate") != .001
            or record.get("physical_updates") != 48 or record.get("physical_episode_exposures") != 4608
            or record.get("coverage") != {"depths": list(range(6)), "turns": [8, 10, 12]}):
        raise ValueError("matching complete strict CUDA proof required; CPU tests are not proof")
    if record["artifact_sha256"] != _artifact_hashes(directory, ("probe.json",)):
        raise ValueError("runtime proof artifact identity differs")
    verified = _verified(directory)
    for key in ("status", "flags", "comparisons", "source_sha256", "preserved_source_sha256", "execution_profile",
                "config", "learning_rate", "protocol_sha256", "physical_updates", "physical_episode_exposures", "coverage", "admission"):
        if not _same(record[key], verified[key]):
            raise ValueError("runtime proof differs from authenticated evidence")
    return copy.deepcopy(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "worker", "verify"):
        child = sub.add_parser(name); child.add_argument("--output", required=True)
        if name == "prepare":
            child.add_argument("--width", type=int, required=True, choices=(96, 192, 256))
            child.add_argument("--admission", action="store_true", help="exercise admitted naming overrides")
        elif name == "worker":
            child.add_argument("--name", required=True, choices=WORKERS)
    args = parser.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.command == "prepare":
        record = prepare(args.output, width=args.width, admission=args.admission)
        print(json.dumps({"prepared": True, "config": record["config"], "physical_updates": record["planned_physical_updates"]}))
        return 0
    record = worker(args.output, args.name) if args.command == "worker" else verify(args.output)
    print(json.dumps({key: record.get(key) for key in ("status", "physical_updates", "flags", "error")}, allow_nan=False))
    return 0 if record["status"] in ("completed", "passed", "passed_cpu_engineering_only") else 1


if __name__ == "__main__":
    raise SystemExit(main())
