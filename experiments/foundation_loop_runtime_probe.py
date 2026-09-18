"""Prospective, one-shot whole-loop execution proof; never a learning study.

prepare, then five separate worker processes in WORKERS order, then verify.
The strict-CUDA workload is 8 + 0 + 3 + 5 + 8 = 24 optimizer updates.
No command retries an occupied directory. CPU verification performs no forward
pass; real CUDA envelope loads are exercised and recorded by split workers.
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

from experiments import foundation_plan as planning
from experiments.execution_profile import configure_strict_profile, runtime_profile
from experiments.foundation_admission import repair_plan, authenticate_admission
from experiments.foundation_curriculum import FAMILIES, generate_pair
from experiments.foundation_evidence import json_digest
from experiments.foundation_loop import FoundationLoop, source_hashes as loop_sources, _same_tree
from experiments.foundation_metrics import _canonical, _validate_row
from experiments.foundation_provider import FoundationPracticeProvider, EVALUATION_SCHEMA, _progress
from experiments.foundation_training import FoundationTrainer, replay_evidence
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig
from experiments.train_cognitive import _check_finite_tree

SCHEMA = "bic-foundation-whole-loop-runtime-probe-v1"
WORKERS = ("uninterrupted", "partial-evaluation", "split-practice", "resumed", "repeat")
STEPS, MIDPOINT = 8, 3
MODEL_SEED = 8301
PLAN = dict(seed=831000001, stage_updates=10, final_updates=6,
            micro_batch_size=32, rehearsal_every=2, ordering_seed=0)
ROOT = Path(__file__).resolve().parents[1]
PROCESS_INSTANCE = uuid.uuid4().hex
TIMING_KEYS = {"retained_step_seconds", "materialization_seconds_included_in_step"}


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def source_hashes():
    return {**loop_sources(), "experiments/foundation_loop_runtime_probe.py": _hash(__file__)}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode())
        stream.flush(); os.fsync(stream.fileno())


def _save(path, value):
    with Path(path).open("xb") as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _append(path, value):
    with Path(path).open("ab") as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False)+"\n").encode())
        stream.flush(); os.fsync(stream.fileno())


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(name+" must be finite and nonnegative")


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name+" must be a nonnegative integer")


def _json_same(left, right):
    """JSON-bound metadata may turn optimizer tuples into lists; tensors may not."""
    def canonical(value):
        return json_digest(json.loads(json.dumps(value, allow_nan=False)))
    return canonical(left) == canonical(right)


def _error(error):
    return dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())


def _runtime(device, *, configure=False):
    if torch.get_num_threads() != 1:
        raise ValueError("one CPU thread must be configured explicitly")
    if device == "cpu":
        return dict(device="cpu", engineering_only=True, torch_version=str(torch.__version__),
                    threads=1, interop_threads=torch.get_num_interop_threads())
    if device != "cuda:0" or torch.get_num_interop_threads() != 1:
        raise ValueError("strict proof requires cuda:0 and one inter-op thread")
    if configure:
        configure_strict_profile()
    return runtime_profile(device)


def operations(name):
    """Fixed call boundaries; baseline slicing adds no optimizer work."""
    if name in ("uninterrupted", "repeat"):
        return [("partial-baseline", dict(max_updates=0, max_evaluations=1)),
                ("midpoint", dict(max_updates=MIDPOINT)),
                ("endpoint", dict(max_updates=STEPS-MIDPOINT))]
    if name == "partial-evaluation":
        return [("partial-baseline", dict(max_updates=0, max_evaluations=1))]
    if name == "split-practice":
        return [("midpoint", dict(max_updates=MIDPOINT))]
    if name == "resumed":
        return [("endpoint", dict(max_updates=STEPS-MIDPOINT))]
    raise ValueError("unknown prescribed worker")


def normalize(snapshot):
    """Remove only explicitly declared, finite timing fields; keep everything else."""
    value = copy.deepcopy(snapshot)
    timing = value["provider"]["learner"]["timing"]
    if set(timing) != TIMING_KEYS:
        raise ValueError("unexpected learner timing fields")
    for key, number in timing.items(): _finite(number, key)
    if timing["materialization_seconds_included_in_step"] > timing["retained_step_seconds"]:
        raise ValueError("materialization time is not a subset of step time")
    del value["provider"]["learner"]["timing"]
    for key in ("training_seconds", "evaluation_seconds"):
        _finite(value["state"]["work"][key], key)
        del value["state"]["work"][key]
    def visit(item):
        if isinstance(item, dict):
            if item.get("schema") == EVALUATION_SCHEMA:
                _finite(item["wall_seconds"], "response wall_seconds")
                del item["wall_seconds"]
                for row in item["per_bank"].values():
                    _finite(row["seconds"], "schema-qualified bank scorer seconds")
                    del row["seconds"]
            for child in item.values(): visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item: visit(child)
    # deepcopy preserves aliases: visit each shared response only once.
    seen = set()
    original_visit = visit
    def once(item):
        if isinstance(item, (dict, list, tuple)):
            if id(item) in seen: return
            seen.add(id(item))
        original_visit(item)
    visit = once
    visit(value["state"])
    _check_finite_tree(value, "normalized loop evidence")
    return value


def fixture(micro_batch_size=32):
    """Pure canonical preparation, independent of any project study files."""
    base = planning.build_plan(**{**PLAN, "micro_batch_size": micro_batch_size})
    prefix = base["schedules"]["mixed"][:STEPS]
    history = sorted({transcript_digest(row) for bundle in prefix
        for row in planning.materialize_pair(base, bundle, "color", 0)})
    plan, admission = repair_plan(base, history)
    banks = {}
    for role_index, role in enumerate(("development", "retention")):
        banks[role] = {}
        for family_index, family in enumerate(FAMILIES):
            turns = ((8, 10, 12), (12, 8, 10))[role_index][family_index]
            rows = []
            for index in range(micro_batch_size//2):
                rows += generate_pair(family, 832000001+role_index*100000+family_index*1000+index,
                    depth=0, turns=turns, split="dev")
            banks[role][f"engineering/{family}/d0/direct/t{turns}"] = rows
    evaluation_transcripts = [transcript_digest(row) for role in banks.values()
                             for rows in role.values() for row in rows]
    if len(evaluation_transcripts) != len(set(evaluation_transcripts)):
        raise ValueError("engineering evaluation transcripts collide")
    protection = sorted(set(history) | set(evaluation_transcripts))
    coverage = {"turns": sorted({plan["bundles"][str(i)]["turns"] for i in prefix}),
        "depths": sorted({plan["bundles"][str(i)]["depth"] for i in prefix}),
        "overridden_bundles": [i for i in prefix if f"{i}/color/0" in plan["admission"]["realization_attempts"]],
        "evaluation_turns": sorted({len(rows[0]["turns"]) for role in banks.values() for rows in role.values()})}
    if coverage["turns"] != [8, 10, 12] or coverage["evaluation_turns"] != [8, 10, 12] or coverage["overridden_bundles"] != prefix:
        raise ValueError("prescribed length/admission coverage differs")
    return dict(base_plan=base, plan=plan, admission=admission, admission_protected=history,
                protected=protection, evaluation_banks=banks, coverage=coverage)


def _provider(inputs, protocol):
    return FoundationPracticeProvider(inputs["plan"], "mixed", seed=MODEL_SEED,
        evaluation_banks=inputs["evaluation_banks"], admission_protected_transcripts=inputs["admission_protected"],
        protected_transcripts=inputs["protected"], admission_receipt=inputs["admission"],
        config=SequenceConfig(**protocol["config"]), learning_rate=.001, device=protocol["device"])


def prepare(directory, *, device="cuda:0", config=None, micro_batch_size=32):
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _write(directory/"prepare-started.json", dict(schema=SCHEMA, started_utc=_now(), pid=os.getpid()))
    try:
        if device == "cuda:0":
            if config is not None or micro_batch_size != 32:
                raise ValueError("declared CUDA workload is width192 and micro32")
            config = SequenceConfig(width=192, layers=4, heads=4, feedforward=768, max_turns=12)
        elif device != "cpu" or type(config) is not SequenceConfig:
            raise ValueError("CPU engineering requires an explicit small configuration")
        runtime = _runtime(device, configure=True)
        sources = source_hashes()
        inputs = fixture(micro_batch_size)
        _save(directory/"inputs.pt", inputs)
        protocol = dict(schema=SCHEMA, device=device, config=asdict(config), model_seed=MODEL_SEED,
            learning_rate=.001, order="mixed", steps=STEPS, midpoint=MIDPOINT, micro_batch_size=micro_batch_size,
            loop_options=dict(window_updates=STEPS, chunk_updates=2, history_limit=2), workers=list(WORKERS),
            expected_operations={name: operations(name) for name in WORKERS},
            execution_profile=runtime, source_sha256=sources, inputs_sha256=_hash(directory/"inputs.pt"),
            coverage=inputs["coverage"], planned_updates=3*STEPS,
            planned_training_episode_exposures=3*STEPS*3*micro_batch_size,
            planned_evaluation_banks=36, planned_evaluation_episode_exposures=36*micro_batch_size,
            full_plan_updates=len(inputs["plan"]["bundles"]), automatic_promotion=False,
            scope="Whole-loop local execution proof at the first 8-update window; not whole-plan exhaustion, learning, or cross-machine repeatability.",
            ignored_timing_fields=["provider.learner.timing", "state.work.training_seconds",
                "state.work.evaluation_seconds", "schema-qualified evaluation response wall_seconds",
                "schema-qualified evaluation response per_bank[*].seconds"])
        provider = _provider(inputs, protocol)
        loop = FoundationLoop(provider, **protocol["loop_options"])
        initial = loop.snapshot(); _save(directory/"initial.pt", initial)
        protocol["initial_sha256"] = _hash(directory/"initial.pt")
        protocol["provider_identity"] = initial["provider_identity"]
        for name, digest in sources.items():
            destination = directory/"sources"/name; destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream: stream.write((ROOT/name).read_bytes())
            if _hash(destination) != digest: raise RuntimeError("source changed during preparation")
        if source_hashes() != sources or _runtime(device) != runtime:
            raise RuntimeError("source/runtime changed before preparation completed")
        _write(directory/"protocol.json", protocol)
        _write(directory/"prepared.json", dict(protocol_sha256=_hash(directory/"protocol.json"),
            wall_seconds=time.monotonic()-started, completed_utc=_now(), optimizer_updates=0,
            neural_evaluation_episodes=0))
        return protocol
    except BaseException as error:
        _write(directory/"prepare-failed.json", dict(error=_error(error), wall_seconds=time.monotonic()-started))
        raise


def _inputs(directory):
    protocol = _read(directory/"protocol.json")
    if protocol["schema"] != SCHEMA or _read(directory/"prepared.json")["protocol_sha256"] != _hash(directory/"protocol.json"):
        raise ValueError("probe protocol binding differs")
    if source_hashes() != protocol["source_sha256"]: raise ValueError("probe sources changed")
    for name, digest in protocol["source_sha256"].items():
        if _hash(directory/"sources"/name) != digest: raise ValueError("source snapshot differs")
    for name in ("inputs", "initial"):
        if _hash(directory/(name+".pt")) != protocol[name+"_sha256"]: raise ValueError("frozen data differ")
    if (protocol["steps"] != STEPS or protocol["midpoint"] != MIDPOINT or protocol["workers"] != list(WORKERS)
            or json_digest(protocol["expected_operations"]) != json_digest({name: operations(name) for name in WORKERS})
            or protocol["model_seed"] != MODEL_SEED or protocol["learning_rate"] != .001
            or protocol["automatic_promotion"] is not False):
        raise ValueError("prescribed operation contract differs")
    inputs = _load(directory/"inputs.pt")
    planning.validate_plan(inputs["base_plan"])
    authenticate_admission(inputs["base_plan"], inputs["plan"], inputs["admission_protected"], receipt=inputs["admission"])
    initial = _load(directory/"initial.pt")
    if not _json_same(initial["provider_identity"], protocol["provider_identity"]):
        raise ValueError("prepared provider identity differs")
    identity = protocol["provider_identity"]
    if (identity["runtime"] != initial["provider"]["identity"]["runtime"]
            or identity["plan_sha256"] != json_digest(inputs["plan"])
            or identity["admission_receipt_sha256"] != json_digest(inputs["admission"])
            or identity["admission_protection_sha256"] != json_digest(inputs["admission_protected"])
            or identity["admission_protection_count"] != len(inputs["admission_protected"])
            or identity["trainer_protection_sha256"] != json_digest(inputs["protected"])
            or identity["trainer_protection_count"] != len(inputs["protected"])
            or (protocol["device"] == "cuda:0" and identity["runtime"] != protocol["execution_profile"])):
        raise ValueError("prepared runtime/admission/protection identity differs")
    return protocol, inputs, initial


def _artifacts(directory):
    return {path.relative_to(directory).as_posix(): _hash(path) for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "result.json"}


def _receipt(directory, name):
    result = _read(directory/name/"result.json")
    if result["schema"] != SCHEMA or result["name"] != name or result["status"] != "completed":
        raise ValueError("prior worker did not complete")
    if result["artifact_sha256"] != _artifacts(directory/name): raise ValueError("worker artifacts differ")
    return result


def worker(directory, name):
    directory = Path(directory).resolve()
    if name not in WORKERS: raise ValueError("unknown worker")
    destination = directory/name; destination.mkdir(exist_ok=False)
    started = time.monotonic()
    result = dict(schema=SCHEMA, name=name, status="running", started_utc=_now(),
        process=dict(pid=os.getpid(), instance=PROCESS_INSTANCE), parents={}, operations=[])
    _write(destination/"started.json", result)
    loop, active = None, None
    try:
        protocol, inputs, initial = _inputs(directory)
        result["protocol_sha256"] = _hash(directory/"protocol.json")
        for previous in WORKERS[:WORKERS.index(name)]:
            record = _receipt(directory, previous)
            if record["process"]["instance"] == PROCESS_INSTANCE:
                raise ValueError("workers must be distinct processes")
            result["parents"][previous] = _hash(directory/previous/"result.json")
        runtime = _runtime(protocol["device"], configure=True)
        if runtime != protocol["execution_profile"]: raise ValueError("worker runtime differs")
        result["execution_profile"] = runtime
        setup = time.monotonic()
        provider = _provider(inputs, protocol)
        fresh = FoundationLoop(provider, **protocol["loop_options"])
        if not _same_tree(fresh.snapshot(), initial): raise ValueError("fresh complete initial state differs")
        _save(destination/"initial.pt", fresh.snapshot())
        parent = {"split-practice": ("partial-evaluation", "partial-baseline"),
                  "resumed": ("split-practice", "midpoint")}.get(name)
        if parent is None:
            loop = fresh; loop.save(destination/"loop.pt")
        else:
            # The owner's registry is weak. Release the zero-update wrapper;
            # reuse its authenticated provider instead of preparing a second model.
            del fresh
            parent_path = directory/parent[0]/(parent[1]+".pt")
            with (destination/"loop.pt").open("xb") as stream:
                stream.write(parent_path.read_bytes()); stream.flush(); os.fsync(stream.fileno())
            loop = FoundationLoop.load(destination/"loop.pt", provider)
            loaded = loop.snapshot()
            if not _same_tree(loaded, _load(parent_path)):
                raise ValueError("immediate full-envelope reload differs, including timing")
            _save(destination/"loaded.pt", loaded)
            result["loaded_from"] = dict(worker=parent[0], checkpoint=parent[1], sha256=_hash(parent_path), exact=True)
        result["setup_and_restore_seconds"] = time.monotonic()-setup
        for index, (checkpoint, kwargs) in enumerate(operations(name)):
            active = dict(index=index, checkpoint=checkpoint, kwargs=kwargs, started_utc=_now(),
                          starting_updates=loop.status["updates"])
            _append(destination/"journal.jsonl", dict(event="started", **active))
            report = loop.run(**kwargs)
            payload = loop.snapshot(); _save(destination/(checkpoint+".pt"), payload)
            completed = dict(event="completed", **active, report=report,
                             checkpoint_sha256=_hash(destination/(checkpoint+".pt")))
            _append(destination/"journal.jsonl", completed)
            result["operations"].append(completed)
            active = None
        if source_hashes() != protocol["source_sha256"] or _runtime(protocol["device"]) != runtime:
            raise RuntimeError("sources/runtime changed during worker")
        result["accounting"] = accounting([row["report"] for row in result["operations"]], require_success=True)
        result.update(status="completed", endpoint_status=loop.status, completed_utc=_now())
    except BaseException as error:
        result.update(status="failed", error=_error(error), unfinished_operation=active,
                      last_report=None if loop is None else copy.deepcopy(loop.last_report))
        if active is not None:
            _append(destination/"journal.jsonl", dict(event="failed", **active, report=result["last_report"], error=result["error"]))
        raise
    finally:
        result["wall_seconds"] = time.monotonic()-started
        result["artifact_sha256"] = _artifacts(destination)
        _write(destination/"result.json", result)
    return result


def accounting(reports, *, require_success=False):
    """Count recorded physical work; never infer it from endpoint cursors."""
    keys = ("physical_optimizer_updates", "drawn_episode_exposures", "neural_attempted_episode_exposures",
            "completed_microbatch_episode_exposures", "retained_updates", "discarded_completed_updates",
            "discarded_optimizer_updates", "evaluation_banks", "evaluation_episode_exposures",
            "failed_step_attempts", "unknown_optimizer_attempts", "unknown_practice_attempts",
            "incomplete_evaluation_attempts")
    result = dict.fromkeys(keys, 0)
    result["unknown_evaluation_episode_exposures"] = False
    result["wall_seconds"] = 0.
    for report in reports:
        for key in keys:
            value = report[key]
            if value is None: result[key] = None
            else:
                _integer(value, key)
                if result[key] is not None: result[key] += value
        _finite(report["wall_seconds"], "operation wall")
        result["wall_seconds"] += report["wall_seconds"]
        if type(report["unknown_evaluation_episode_exposures"]) is not bool:
            raise ValueError("unknown evaluation flag must be boolean")
        result["unknown_evaluation_episode_exposures"] |= report["unknown_evaluation_episode_exposures"]
        if require_success and (report["status"] not in ("update_bound", "evaluation_bound")
                or report["publication_uncertain"] or report["unknown_evaluation_episode_exposures"]
                or any(report[k] != 0 for k in ("discarded_optimizer_updates", "failed_step_attempts",
                    "unknown_optimizer_attempts", "unknown_practice_attempts", "incomplete_evaluation_attempts"))
                or any(report[k] is None for k in keys)):
            raise ValueError("execution proof includes failed, discarded or unknown work")
    return result


def validate_operation(report, starting_cursor, ending_cursor, bundles):
    """Reconcile physical reports with canonical admitted lessons, not just totals."""
    accounting([report], require_success=True)
    if (report["starting_updates"] != starting_cursor or report["ending_updates"] != ending_cursor
            or report["retained_updates"] != ending_cursor-starting_cursor):
        raise ValueError("operation endpoints disagree with recorded snapshots")
    cursor, episodes = starting_cursor, 0
    for chunk in report["practice_reports"]:
        steps = chunk["step_reports"]
        if (chunk["starting_updates"] != cursor or chunk["completed_updates"] != len(steps)
                or chunk["ending_updates"] != cursor+len(steps) or chunk["physical_optimizer_updates"] != len(steps)
                or chunk["failed_step_attempts"] or chunk["unknown_optimizer_attempts"]
                or chunk["accounting_uncertain"] or chunk["restore_required"]):
            raise ValueError("practice chunk endpoints/accounting differ")
        chunk_episodes = 0
        for step in steps:
            if cursor >= len(bundles): raise ValueError("journal exceeds prospective stream")
            expected = bundles[cursor]
            for key, number in (("cursor", cursor+1), ("bundle_id", expected["bundle_id"]),
                    ("physical_optimizer_updates", 1), ("retained_optimizer_updates", 1),
                    ("drawn_microbatches", 3), ("neural_attempted_microbatches", 3), ("completed_microbatches", 3)):
                if type(step.get(key)) is not int or step[key] != number:
                    raise ValueError("step integer identity/count differs")
            for key in ("wall_seconds", "step_seconds", "materialization_seconds", "loss", "action_loss", "reply_loss", "observation_language_loss"):
                _finite(step[key], key)
            if not step["materialization_seconds"] <= step["step_seconds"] <= step["wall_seconds"]:
                raise ValueError("step timing scopes differ")
            if [row["family"] for row in step["microbatches"]] != list(FAMILIES):
                raise ValueError("step family order differs")
            observed = 0
            for row in step["microbatches"]:
                canonical = expected["families"][row["family"]]
                if (any(row[key] != canonical[key] for key in ("rows_sha256", "recipes_sha256", "exposures"))
                        or row["depth"] != expected["depth"] or row["turns"] != expected["turns"]):
                    raise ValueError("step lessons, bytes or labels differ from canonical recipe")
                observed += canonical["exposures"]["episodes"]
                for key in ("loss", "action_loss", "reply_loss", "observation_language_loss"): _finite(row[key], key)
            for key in ("drawn_episode_exposures", "neural_attempted_episode_exposures", "completed_microbatch_episode_exposures"):
                if type(step[key]) is not int or step[key] != observed: raise ValueError("step episode count differs")
            cursor += 1; chunk_episodes += observed
        for key in ("drawn_episode_exposures", "neural_attempted_episode_exposures", "completed_microbatch_episode_exposures"):
            if chunk[key] != chunk_episodes: raise ValueError("chunk episode count differs")
        episodes += chunk_episodes
    if cursor != ending_cursor or report["completed_updates"] != ending_cursor-starting_cursor:
        raise ValueError("operation lacks canonical step reports")
    for key, number in (("physical_optimizer_updates", ending_cursor-starting_cursor),
            ("drawn_episode_exposures", episodes), ("neural_attempted_episode_exposures", episodes),
            ("completed_microbatch_episode_exposures", episodes)):
        if report[key] != number: raise ValueError("operation physical totals differ from its steps")
    return True


class _EvidenceCheck:
    """CPU-only canonical controller validation, not a fabricated CPU provider.

    Reuse the frozen controller's validation algorithms against recorded CUDA
    identity. This performs no provider construction, load, scoring or inference.
    """
    _validate = FoundationLoop._validate
    _validate_references = FoundationLoop._validate_references
    _validate_phase = FoundationLoop._validate_phase
    _retention = FoundationLoop._retention

    def __init__(self, protocol, inputs):
        self.options = protocol["loop_options"]
        self._total = len(inputs["plan"]["bundles"])
        self._specs = protocol["provider_identity"]["evaluation_specs"]
        self._queue = [(r, n) for r in sorted(self._specs) for n in sorted(self._specs[r])]
        config = SequenceConfig(**protocol["config"])
        self.canonical = {(r, n): _canonical(rows, n, config, "dev")
            for r, banks in inputs["evaluation_banks"].items() for n, rows in banks.items()}
        if set(self.canonical) != set(self._queue) or any(
                identity[0] != self._specs[r][n]["identity"] for (r,n),identity in self.canonical.items()):
            raise ValueError("prepared bank specifications differ from canonical data")
        self.identity = json_digest(protocol["provider_identity"])

    def _response(self, response, role, name, producer):
        fields = {"schema", "provider_sha256", "role", "weights_sha256", "updates", "control", "per_bank",
                  "progress", "wall_seconds", "automatic_promotion"}
        if (type(response) is not dict or set(response) != fields or response["schema"] != EVALUATION_SCHEMA
                or response["provider_sha256"] != self.identity or response["role"] != role
                or response["updates"] != producer["updates"] or response["weights_sha256"] != producer["weights_sha256"]
                or response["control"] != "normal" or response["automatic_promotion"] is not False
                or set(response["per_bank"]) != {name} or set(response["progress"]) != {name}):
            raise ValueError("recorded evaluation identity/producer differs")
        _finite(response["wall_seconds"], "response wall")
        row = response["per_bank"][name]
        _validate_row(row, *self.canonical[role, name], "normal")
        if json_digest(response["progress"][name]) != json_digest(_progress(row)):
            raise ValueError("recorded progress differs from canonical counts")


def _checkpoint(directory, name, point, protocol, inputs, initial, validator, trainer):
    value = _load(directory/name/(point+".pt"))
    if (set(value) != set(initial) or set(value["provider"]) != set(initial["provider"])
            or value["provider"]["schema"] != initial["provider"]["schema"]
            or value["schema"] != initial["schema"] or value["source_sha256"] != initial["source_sha256"]
            or value["options"] != protocol["loop_options"] or not _json_same(value["provider_identity"], protocol["provider_identity"])
            or value["provider"]["identity"] != initial["provider"]["identity"]):
        raise ValueError("loop/provider envelope identity differs from prepared CUDA identity")
    trainer.restore(value["provider"]["learner"])
    validator._validate(value["state"], value["provider"])
    # Provider snapshot's pending hash and bounds are checked independently.
    pending = value["provider"]["pending"]
    if pending is not None:
        request = pending["request"]
        expected = dict(kind="prescribed_prefix", provider_sha256=json_digest(protocol["provider_identity"]),
            plan_sha256=json_digest(inputs["plan"]), order="mixed", start_cursor=0, stop_cursor=STEPS)
        if (set(pending) != {"request", "request_sha256", "consumed_updates"}
                or request != expected or pending["request_sha256"] != json_digest(expected)
                or pending["consumed_updates"] != value["provider"]["learner"]["cursor"]):
            raise ValueError("recorded pending provider prefix differs")
    normalize(value)  # Validate every ignored timing value before any comparison.
    return value


def verify(directory):
    """No neural forward/optimizer work; preserve any failure and never retry."""
    directory = Path(directory).resolve()
    claim = directory/"verification-started.json"
    _write(claim, dict(schema=SCHEMA, started_utc=_now(), pid=os.getpid()))
    started = time.monotonic()
    try:
        protocol, inputs, initial = _inputs(directory)
        results = {name: _receipt(directory, name) for name in WORKERS}
        input_hashes = {name: _hash(directory/name) for name in
            ("protocol.json", "prepared.json", "inputs.pt", "initial.pt",
             *(worker_name+"/result.json" for worker_name in WORKERS))}
        if len({row["process"]["instance"] for row in results.values()}) != len(WORKERS):
            raise ValueError("workers were not separate processes")
        validator = _EvidenceCheck(protocol, inputs)
        trainer = FoundationTrainer(inputs["plan"], "mixed", seed=MODEL_SEED,
            config=SequenceConfig(**protocol["config"]), learning_rate=.001, device="cpu",
            protected_transcripts=inputs["protected"])
        if not _same_tree(trainer.snapshot()["weights"], initial["provider"]["learner"]["weights"]):
            raise ValueError("prepared initialization differs from canonical seeded initialization")
        if not _json_same(trainer.recipe, protocol["provider_identity"]["trainer_recipe"]):
            raise ValueError("prepared trainer identity differs from canonical recipe")
        bundles = replay_evidence(inputs["plan"], "mixed", STEPS, inputs["protected"], include_bundles=True)["bundles"]
        checkpoints, accounting_rows = {}, []
        for name, receipt in results.items():
            if receipt["protocol_sha256"] != _hash(directory/"protocol.json") or receipt["execution_profile"] != protocol["execution_profile"]:
                raise ValueError("worker protocol/runtime differs")
            expected_parents = {p: _hash(directory/p/"result.json") for p in WORKERS[:WORKERS.index(name)]}
            if receipt["parents"] != expected_parents: raise ValueError("worker predecessor bindings differ")
            checkpoints[name, "initial"] = _checkpoint(directory, name, "initial", protocol, inputs, initial, validator, trainer)
            if not _same_tree(checkpoints[name, "initial"], initial): raise ValueError("initial full envelope differs")
            previous = initial
            if name in ("split-practice", "resumed"):
                previous = _checkpoint(directory, name, "loaded", protocol, inputs, initial, validator, trainer)
            journal = [json.loads(line) for line in (directory/name/"journal.jsonl").read_text().splitlines()]
            if len(journal) != 2*len(operations(name)) or len(receipt["operations"]) != len(operations(name)):
                raise ValueError("journal has missing/extra/unfinished operations")
            for index, (point, kwargs) in enumerate(operations(name)):
                begin, end = journal[2*index:2*index+2]
                if (begin["event"] != "started" or end["event"] != "completed" or begin["index"] != index
                        or begin["kwargs"] != kwargs or begin["checkpoint"] != point
                        or {k: end[k] for k in begin if k != "event"} != {k:v for k,v in begin.items() if k != "event"}
                        or end != receipt["operations"][index]
                        or end["checkpoint_sha256"] != _hash(directory/name/(point+".pt"))):
                    raise ValueError("operation journal/checkpoint identity differs")
                value = _checkpoint(directory, name, point, protocol, inputs, initial, validator, trainer)
                checkpoints[name, point] = value
                report = end["report"]
                expected_cursor = {"partial-baseline": 0, "midpoint": MIDPOINT, "endpoint": STEPS}[point]
                if (report["starting_updates"] != begin["starting_updates"]
                        or report["ending_updates"] != expected_cursor
                        or report["retained_updates"] != expected_cursor-begin["starting_updates"]
                        or value["provider"]["learner"]["cursor"] != expected_cursor):
                    raise ValueError("journal operation cursor differs")
                validate_operation(report, previous["provider"]["learner"]["cursor"], expected_cursor, bundles)
                for field in ("evaluation_banks", "evaluation_episode_exposures"):
                    if value["state"]["work"][field]-previous["state"]["work"][field] != report[field]:
                        raise ValueError("evaluation work differs from committed controller counters")
                if point == "partial-baseline" and (value["state"]["phase"] != "evaluate" or value["state"]["evaluation"]["cursor"] != 1):
                    raise ValueError("partial baseline was not actually persisted")
                if point == "midpoint" and value["provider"]["pending"] is None:
                    raise ValueError("practice midpoint has no pending request")
                if point == "endpoint" and (value["state"]["evaluations_completed"] != 2 or value["state"]["phase"] != "practice"):
                    raise ValueError("endpoint development/retention evaluation missing")
                previous = value
            reports = [row["report"] for row in receipt["operations"]]
            actual = accounting(reports, require_success=True)
            if actual != receipt["accounting"]: raise ValueError("worker physical accounting differs")
            accounting_rows += reports
            if not _same_tree(_load(directory/name/"loop.pt"), checkpoints[name, operations(name)[-1][0]]):
                raise ValueError("durable endpoint differs from recorded checkpoint")
        comparisons = {}
        for left, right, label in ((('uninterrupted','partial-baseline'),('partial-evaluation','partial-baseline'),'partial_baseline'),
            (('uninterrupted','midpoint'),('split-practice','midpoint'),'pending_midpoint'),
            (('uninterrupted','endpoint'),('resumed','endpoint'),'resumed_endpoint'),
            (('uninterrupted','endpoint'),('repeat','endpoint'),'independent_endpoint'),
            (('uninterrupted','partial-baseline'),('repeat','partial-baseline'),'repeat_partial_baseline'),
            (('uninterrupted','midpoint'),('repeat','midpoint'),'repeat_midpoint')):
            comparisons[label] = _same_tree(normalize(checkpoints[left]), normalize(checkpoints[right]))
        for name, parent, point in (("split-practice", "partial-evaluation", "partial-baseline"), ("resumed", "split-practice", "midpoint")):
            loaded = _checkpoint(directory, name, "loaded", protocol, inputs, initial, validator, trainer)
            comparisons[name+"_immediate_load_including_timing"] = _same_tree(loaded, checkpoints[parent, point])
            if results[name]["loaded_from"] != dict(worker=parent, checkpoint=point,
                    sha256=_hash(directory/parent/(point+".pt")), exact=True):
                raise ValueError("reload parent receipt differs")
        if not all(comparisons.values()): raise ValueError("whole-loop trajectories are not exact")
        totals = accounting(accounting_rows, require_success=True)
        if (totals["physical_optimizer_updates"] != protocol["planned_updates"]
                or totals["drawn_episode_exposures"] != protocol["planned_training_episode_exposures"]
                or totals["evaluation_banks"] != protocol["planned_evaluation_banks"]
                or totals["evaluation_episode_exposures"] != protocol["planned_evaluation_episode_exposures"]):
            raise ValueError("observed journals differ from the prospective physical workload")
        if source_hashes() != protocol["source_sha256"]: raise ValueError("sources changed during verification")
        if any(_hash(directory/name) != digest for name,digest in input_hashes.items()):
            raise ValueError("probe input receipts changed during verification")
        for name in WORKERS:
            if _receipt(directory, name) != results[name]:
                raise ValueError("worker evidence changed during verification")
        result = dict(schema=SCHEMA, status="passed", strict_cuda_proven=protocol["device"] == "cuda:0",
            comparisons=comparisons, observed_work=totals, automatic_promotion=False,
            worker_wall_seconds={name: row["wall_seconds"] for name,row in results.items()},
            worker_setup_and_restore_seconds={name: row["setup_and_restore_seconds"] for name,row in results.items()},
            input_sha256=input_hashes,
            source_sha256=protocol["source_sha256"], cpu_verification_seconds=time.monotonic()-started,
            scope="Exact observed local whole-loop execution under declared profile. CUDA loads happened in workers; CPU verifier authenticated canonical evidence, full optimizer restores and artifact comparisons. Timing alone excluded from independent trajectories. Not learning evidence or universal CUDA repeatability.")
        _write(directory/"proof.json", result)
        return result
    except BaseException as error:
        _write(directory/"verification-failed.json", dict(schema=SCHEMA, error=_error(error), wall_seconds=time.monotonic()-started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "worker", "verify")); parser.add_argument("--output", required=True)
    parser.add_argument("--name", choices=WORKERS)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.command == "prepare": result = prepare(args.output)
    elif args.command == "worker": result = worker(args.output, args.name)
    else: result = verify(args.output)
    print(json.dumps({"command": args.command, "status": result.get("status", "prepared"),
                      "automatic_promotion": False}, sort_keys=True))


if __name__ == "__main__": main()
