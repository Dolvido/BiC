"""Fixed, one-shot CUDA execution proof for the two prospective objectives.

One process constructs one fresh admitted fixture index. Each of six cases runs
reference6, split3/CPU-save/strict-restore/continued3, and repeat6: 108 physical
updates, with no scoring or optimizer tuning. This tests same-process execution,
not a process restart or all future workloads. Importing configures no runtime.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import uuid

import torch

from experiments.execution_profile import (
    assert_strict_profile, configure_strict_profile, runtime_profile)
from experiments.foundation_admission import repair_plan
from experiments.foundation_objective_training import (
    ObjectiveFoundationTrainer, source_hashes as trainer_sources)
from experiments.foundation_plan import build_plan
from experiments.foundation_plan_index import AuthenticatedPlanIndex
from experiments.sequence_student import SequenceConfig


SCHEMA = "bic-foundation-objective-runtime-proof-v1"
ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE_IDS = ("baseline", "balanced_reply")
RATES = (.0003, .001, .003)
CONFIG = SequenceConfig(width=192, layers=4, heads=4, feedforward=768, max_turns=12)
MODEL_SEED = 847001
PLAN_OPTIONS = dict(seed=847000001, stage_updates=20, final_updates=6,
    micro_batch_size=32, rehearsal_every=4, ordering_seed=8470)
STEPS, MIDPOINT, ORDER = 6, 3, "curriculum"
CASES = tuple(dict(id=f"{objective}-{rate:g}", objective_id=objective, learning_rate=rate)
              for objective in OBJECTIVE_IDS for rate in RATES)
CHECK_FIELDS = ("schema", "recipe", "weights", "optimizer", "cursor", "evidence")
CASE_CHECKS = ("common_seeded_initial_weights", "cpu_midpoint_reload_exact_including_timing",
    "split_prefix_every_step_exact", "resumed_every_step_exact", "repeat_every_step_exact",
    "full_optimizer_and_evidence_exact", "final_indexed_evidence_exact")
COMPARISONS = ([("split", step) for step in range(4)] + [("resumed_fresh", 0)]
    + [("resumed", step) for step in range(3, 7)] + [("repeat", step) for step in range(7)])
EXPECTED_WORK = dict(optimizer_step_calls_attempted=108, optimizer_step_calls_completed=108,
    known_physical_optimizer_updates=108, retained_optimizer_updates=108,
    optimizer_completion_unknown_calls=0, neural_attempted_microbatches=324,
    completed_microbatches=324, neural_attempted_episode_exposures=10368,
    completed_microbatch_episode_exposures=10368)
EXPECTED_OPERATIONS = dict(runtime_setup=1, plan_build=1, plan_admission=1, index_construction=1,
    trainer_construction=24, snapshot=138, training_step=108, checkpoint_save=24,
    checkpoint_load=6, strict_restore=6)


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(image):
    return hashlib.sha256(image).hexdigest()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _publish(path, image):
    """Publish one complete immutable image, never replacing existing evidence."""
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(image)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha(image)


def _json(path, value):
    return _publish(path, _bytes(value) + b"\n")


def _same(left, right):
    """Exact typed tree equality, including tensor bytes and signed zeros."""
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return (left.dtype == right.dtype and left.shape == right.shape
            and left.device == right.device and left.layout == right.layout
            and left.requires_grad == right.requires_grad
            and left.detach().cpu().contiguous().numpy().tobytes()
                == right.detach().cpu().contiguous().numpy().tobytes())
    if isinstance(left, dict):
        return set(left) == set(right) and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return _bytes(left) == _bytes(right)


def _digest(value):
    """Typed tensor-tree byte digest, independent of torch.save storage IDs."""
    def node(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous()
            return ["tensor", str(data.dtype), list(data.shape), str(item.device),
                    item.requires_grad, _sha(data.numpy().tobytes())]
        if isinstance(item, dict):
            entries = [[node(key), node(val)] for key, val in item.items()]
            return ["dict", sorted(entries, key=lambda row: _bytes(row[0]))]
        if isinstance(item, (tuple, list)):
            return [type(item).__name__, [node(val) for val in item]]
        return [type(item).__name__, item]
    return _sha(_bytes(node(value)))


def _comparison(expected, actual):
    checks = {name: _same(expected[name], actual[name]) for name in CHECK_FIELDS}
    return dict(checks=checks, exact=all(checks.values()), ignored_fields=["timing"],
        expected_sha256={name: _digest(expected[name]) for name in CHECK_FIELDS},
        actual_sha256={name: _digest(actual[name]) for name in CHECK_FIELDS})


def source_hashes():
    names = ("experiments/foundation_objective_runtime_probe.py",
             "experiments/execution_profile.py", "docs/FOUNDATION_OBJECTIVE_STUDY_PROTOCOL.md")
    return {**trainer_sources(), **{name: _sha((ROOT / name).read_bytes()) for name in names}}


class _Journal:
    def __init__(self, directory, allowance):
        self.path = directory / "operations.jsonl"
        with self.path.open("xb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        self.started, self.cpu_started = time.monotonic(), time.process_time()
        self.deadline = self.started + allowance
        self.sequence, self.previous = 0, None
        self.counts, self.active = {}, None
        self.work = dict(optimizer_step_calls_attempted=0, optimizer_step_calls_completed=0,
            known_physical_optimizer_updates=0, retained_optimizer_updates=0,
            optimizer_completion_unknown_calls=0, neural_attempted_microbatches=0,
            completed_microbatches=0, neural_attempted_episode_exposures=0,
            completed_microbatch_episode_exposures=0)

    def event(self, event_kind, **fields):
        record = dict(sequence=self.sequence, previous_event_sha256=self.previous,
                      utc=_utc(), event=event_kind, **fields)
        image = _bytes(record) + b"\n"
        with self.path.open("ab") as stream:
            stream.write(image)
            stream.flush()
            os.fsync(stream.fileno())
        self.previous, self.sequence = _sha(image), self.sequence + 1

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("fixed proof wall allowance elapsed; no retry or extension")

    def perform(self, kind, function, *, context=None, summarize=None, failure=None):
        self.check_time()
        counts = self.counts.setdefault(kind, dict(attempted=0, completed=0))
        intent = dict(kind=kind, context=context or {}, operation=self.sequence)
        self.event("intent", **intent)
        self.active = intent
        counts["attempted"] += 1
        started = time.monotonic()
        try:
            result = function()
            counts["completed"] += 1
            details = summarize(result) if summarize is not None else {}
            self.event("completed", **intent, wall_seconds=time.monotonic()-started, details=details)
            self.active = None
            return result
        except BaseException as error:
            details = failure() if failure is not None else {}
            self.event("failed", **intent, wall_seconds=time.monotonic()-started,
                error=type(error).__name__ + ": " + str(error), details=details)
            self.active = None
            raise

    def step_work(self, report, *, completed):
        self.work["optimizer_step_calls_completed"] += int(completed)
        physical = report.get("physical_optimizer_updates")
        if physical is None:
            self.work["optimizer_completion_unknown_calls"] += 1
        else:
            self.work["known_physical_optimizer_updates"] += physical
        for name in ("retained_optimizer_updates", "neural_attempted_microbatches",
                     "completed_microbatches", "neural_attempted_episode_exposures",
                     "completed_microbatch_episode_exposures"):
            value = report.get(name)
            if type(value) is int:
                self.work[name] += value


def run(directory, *, device="cuda:0", max_seconds=14400):
    """Run only the fixed CUDA proof in a fresh, never-reused output directory.

    Callers configure the strict profile and CPU intra/inter-op threads once
    before calling. Every operation is nonpreemptive; admission can overrun the
    deadline. A missing terminal receipt or unresolved intent means unknown work,
    never authorization to retry. This function implements no restart path.
    """
    if device != "cuda:0":
        raise ValueError("this selected-width proof requires explicit cuda:0; no CPU fallback")
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 14400:
        raise ValueError("finite positive proof allowance at most14400 seconds required")
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError("caller must configure one CPU intra/inter-op thread")
    assert_strict_profile()
    directory = Path(directory).absolute()
    directory.mkdir(parents=True, exist_ok=False)
    journal = _Journal(directory, max_seconds)
    record = dict(schema=SCHEMA, status="running", started_utc=_utc(), objective_ids=list(OBJECTIVE_IDS),
        config=asdict(CONFIG), model_seed=MODEL_SEED, plan_options=PLAN_OPTIONS, order=ORDER,
        rates=list(RATES), steps=STEPS, midpoint=MIDPOINT, results=[], max_seconds=max_seconds,
        planned_physical_updates=108, planned_episode_exposures=10368, planned_model_constructions=30,
        evaluation_performed=False, fitness_scores_read=False, automatic_promotion=False,
        historical_checkpoint_reads=0, scope="Fixed selected-width same-process CUDA continuation and repeatability only; not cross-process, cross-machine or general strict-execution proof.")
    _json(directory / "started.json", record)
    try:
        sources, producers = trainer_sources(), source_hashes()
        record.update(source_sha256=sources, producer_source_sha256=producers)
        for name, expected in producers.items():
            journal.check_time()
            image = (ROOT / name).read_bytes()
            if _sha(image) != expected:
                raise ValueError("source changed while preserving proof source: " + name)
            target = directory / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            _publish(target, image)

        def guard():
            assert_strict_profile()
            if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
                raise ValueError("CPU thread profile changed")
            if trainer_sources() != sources or source_hashes() != producers:
                raise ValueError("runtime proof source closure changed")

        runtime = journal.perform("runtime_setup", lambda: runtime_profile(device))
        if runtime["device"]["name"] != "NVIDIA GeForce RTX 5080":
            raise ValueError("proof requires the declared local RTX5080")
        record["execution_profile"] = runtime
        _json(directory / "runtime.json", runtime)
        torch.cuda.reset_peak_memory_stats(device)
        base = journal.perform("plan_build", lambda: build_plan(**PLAN_OPTIONS))
        plan, admission = journal.perform("plan_admission", lambda: repair_plan(base, []))
        if len(plan["bundles"]) != 126:
            raise ValueError("fixed fixture must contain126 admitted bundles")
        prefix = plan["schedules"][ORDER][:STEPS]
        turns = [plan["bundles"][str(bundle)]["turns"] for bundle in prefix]
        if sorted(turns) != [8, 8, 10, 10, 12, 12]:
            raise ValueError("six-update fixture must visit each utterance count twice")
        coverage = dict(bundle_ids=prefix, turns_in_order=turns,
            depths_in_order=[plan["bundles"][str(bundle)]["depth"] for bundle in prefix],
            complete_plan_bundles=126, executed_prefix_bundles=6, before_split=3, after_split=3,
            scope="This curriculum-prefix fixture tests all three lengths, not all curriculum depths.")
        index = journal.perform("index_construction", lambda: AuthenticatedPlanIndex(plan,
            admission_protected_transcripts=[], protected_transcripts=[], admission_receipt=admission))
        data = {"base-plan.json": base, "plan.json": plan, "admission.json": admission,
            "protected.json": [], "index-identity.json": index.identity,
            "index-construction.json": index.construction, "coverage.json": coverage}
        inputs = {name: _json(directory / name, value) for name, value in data.items()}
        record.update(coverage=coverage, input_sha256=inputs, index_identity=index.identity,
            index_construction=index.construction, fixture_protection="Empty fresh-fixture exclusion set; no fitness evaluation or historical adoption.")
        _json(directory / "protocol.json", {key: value for key, value in record.items() if key != "results"})
        initial_weights = None

        def snapshot(trainer, context):
            return journal.perform("snapshot", trainer.snapshot, context=context,
                summarize=lambda value: dict(cursor=value["cursor"], state_sha256=_digest(
                    {key: value[key] for key in CHECK_FIELDS})))

        def save_checkpoint(path, payload, context):
            def save():
                stream = io.BytesIO()
                torch.save(payload, stream)
                image = stream.getvalue()
                return dict(path=path.relative_to(directory).as_posix(), sha256=_publish(path, image), bytes=len(image))
            return journal.perform("checkpoint_save", save, context=context, summarize=lambda value: value)

        for case in CASES:
            journal.check_time()
            guard()
            case_path = directory / case["id"]
            case_path.mkdir(exist_ok=False)
            case_record = dict(case, status="running", checks={}, comparisons=[], artifacts={})
            record["results"].append(case_record)
            case_started = time.monotonic()
            context = dict(case=case["id"])

            def create(phase):
                guard()
                trainer = journal.perform("trainer_construction", lambda: ObjectiveFoundationTrainer(plan, ORDER,
                    objective_id=case["objective_id"], seed=MODEL_SEED, admission_protected_transcripts=[],
                    protected_transcripts=[], admission_receipt=admission, plan_index=index,
                    config=CONFIG, learning_rate=case["learning_rate"], device=device),
                    context=dict(context, phase=phase), summarize=lambda value: value.setup_report)
                if sum(parameter.numel() for parameter in trainer.model.parameters()) != 2221738:
                    raise ValueError("proof model parameter count differs")
                return trainer

            def advance(trainer, phase):
                guard()
                accounted = False
                preceding_report = trainer.last_report
                def invoke():
                    journal.work["optimizer_step_calls_attempted"] += 1
                    return trainer.step()
                def success(value):
                    nonlocal accounted
                    journal.step_work(value, completed=True)
                    accounted = True
                    return copy.deepcopy(value)
                def failure():
                    nonlocal accounted
                    value = (None if trainer.last_report is preceding_report
                             else copy.deepcopy(trainer.last_report))
                    if not accounted:
                        journal.step_work(value or {}, completed=False)
                        accounted = True
                    return dict(last_report=value)
                journal.perform("training_step", invoke,
                    context=dict(context, phase=phase, cursor_before=trainer.cursor),
                    summarize=success, failure=failure)
                guard()
                return snapshot(trainer, dict(context, phase=phase))

            def compare(expected, actual, phase, cursor):
                value = dict(phase=phase, cursor=cursor, **_comparison(expected, actual))
                case_record["comparisons"].append(value)
                if not value["exact"]:
                    raise ValueError("exact state mismatch at " + case["id"] + "/" + phase + "/" + str(cursor))

            reference = create("reference")
            refs = {0: snapshot(reference, dict(context, phase="reference_initial"))}
            if initial_weights is None:
                initial_weights = copy.deepcopy(refs[0]["weights"])
                record["common_initial_weights_sha256"] = _digest(initial_weights)
                record["common_initial_weights_digest_scope"] = "Typed snapshot tensor-tree digest; not the historical checkpoint_digest algorithm."
            case_record["checks"]["common_seeded_initial_weights"] = _same(initial_weights, refs[0]["weights"])
            if not case_record["checks"]["common_seeded_initial_weights"]:
                raise ValueError("objective/rate changed seeded initial model tensors")
            for cursor in range(1, STEPS + 1):
                refs[cursor] = advance(reference, "reference")
            case_record["artifacts"]["reference"] = save_checkpoint(case_path / "reference.pt", refs[STEPS], context)
            del reference

            split = create("split")
            compare(refs[0], snapshot(split, dict(context, phase="split_initial")), "split", 0)
            for cursor in range(1, MIDPOINT + 1):
                middle = advance(split, "split")
                compare(refs[cursor], middle, "split", cursor)
            midpoint = save_checkpoint(case_path / "midpoint.pt", middle, context)
            case_record["artifacts"]["midpoint"] = midpoint
            del split, middle

            def load_midpoint():
                image = (directory / midpoint["path"]).read_bytes()
                if len(image) != midpoint["bytes"] or _sha(image) != midpoint["sha256"]:
                    raise ValueError("saved midpoint bytes changed before safe decode")
                return torch.load(io.BytesIO(image), map_location="cpu", weights_only=True)
            payload = journal.perform("checkpoint_load", load_midpoint, context=context,
                summarize=lambda value: dict(cursor=value["cursor"], path=midpoint["path"], sha256=midpoint["sha256"]))
            resumed = create("resumed")
            compare(refs[0], snapshot(resumed, dict(context, phase="resumed_fresh")), "resumed_fresh", 0)
            journal.perform("strict_restore", lambda: resumed.restore(payload), context=context,
                summarize=lambda value: dict(cursor=value.cursor, restore_seconds=value.last_restore_seconds))
            restored = snapshot(resumed, dict(context, phase="restored_midpoint"))
            case_record["checks"]["cpu_midpoint_reload_exact_including_timing"] = _same(payload, restored)
            if not case_record["checks"]["cpu_midpoint_reload_exact_including_timing"]:
                raise ValueError("strict midpoint reload changed saved payload")
            compare(refs[MIDPOINT], restored, "resumed", MIDPOINT)
            for cursor in range(MIDPOINT + 1, STEPS + 1):
                actual = advance(resumed, "resumed")
                compare(refs[cursor], actual, "resumed", cursor)
            case_record["artifacts"]["resumed"] = save_checkpoint(case_path / "resumed.pt", actual, context)
            del resumed, payload, restored, actual

            repeat = create("repeat")
            compare(refs[0], snapshot(repeat, dict(context, phase="repeat_initial")), "repeat", 0)
            for cursor in range(1, STEPS + 1):
                actual = advance(repeat, "repeat")
                compare(refs[cursor], actual, "repeat", cursor)
            case_record["artifacts"]["repeat"] = save_checkpoint(case_path / "repeat.pt", actual, context)
            case_record["checks"].update(split_prefix_every_step_exact=True, resumed_every_step_exact=True,
                repeat_every_step_exact=True, full_optimizer_and_evidence_exact=True,
                final_indexed_evidence_exact=_same(actual["evidence"], index.replay(ORDER, STEPS)))
            if not all(case_record["checks"].values()):
                raise ValueError("case failed exact execution checks")
            case_record.update(status="passed", physical_updates=18, episode_exposures=1728,
                wall_seconds=time.monotonic()-case_started)
            _json(case_path / "result.json", case_record)
            del repeat, actual, refs

        guard()
        if runtime_profile(device) != runtime:
            raise ValueError("runtime identity changed during proof")
        expected_operations = {name: dict(attempted=count, completed=count)
                               for name, count in EXPECTED_OPERATIONS.items()}
        if journal.work != EXPECTED_WORK or journal.counts != expected_operations:
            raise ValueError("actual work differs from fixed runtime-proof allowance")
        record.update(status="passed", physical_updates=108, physical_episode_exposures=10368,
            sources_unchanged=True, physical_work_unknown=False)
        journal.event("proof_completed", status="passed", work=journal.work)
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            error=type(error).__name__ + ": " + str(error), active_operation=copy.deepcopy(journal.active),
            physical_work_unknown=bool(journal.active or journal.work["optimizer_completion_unknown_calls"]))
        try:
            journal.event("proof_failed", status=record["status"], error=record["error"], work=journal.work)
        except BaseException as accounting_error:
            record.update(accounting_error=repr(accounting_error), physical_work_unknown=True)
        raise
    finally:
        try:
            if torch.cuda.is_initialized():
                torch.cuda.synchronize(device)
                record.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
        except BaseException as sync_error:
            record.update(status="failed", synchronization_error=repr(sync_error), physical_work_unknown=True)
        record.update(completed_utc=_utc(), wall_seconds=time.monotonic()-journal.started,
            cpu_seconds=time.process_time()-journal.cpu_started, work=copy.deepcopy(journal.work),
            operations=copy.deepcopy(journal.counts),
            completed_model_template_constructions=journal.counts.get("trainer_construction", {}).get("completed", 0)
                + journal.counts.get("strict_restore", {}).get("completed", 0),
            model_construction_scope="One template per completed fresh constructor and one per completed strict restore; failed construction attempts may have partial work.",
            incomplete_attempt_scope="Missing terminal receipt, malformed journal or unmatched operation intent means unknown work and prohibits automatic replay.")
        record["artifact_sha256"] = {path.relative_to(directory).as_posix(): _sha(path.read_bytes())
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "probe.json"}
        _json(directory / "probe.json", record)
    return copy.deepcopy(record)


def load_proof(directory, *, expected_sha256):
    """Read a caller-pinned finished proof without constructing or scoring models."""
    directory = Path(directory).absolute()
    image = (directory / "probe.json").read_bytes()
    if type(expected_sha256) is not str or _sha(image) != expected_sha256:
        raise ValueError("caller-pinned proof image differs")
    record = json.loads(image)
    contract = dict(schema=SCHEMA, status="passed", objective_ids=list(OBJECTIVE_IDS),
        config=asdict(CONFIG), model_seed=MODEL_SEED, plan_options=PLAN_OPTIONS, order=ORDER,
        rates=list(RATES), steps=STEPS, midpoint=MIDPOINT, planned_physical_updates=108,
        planned_episode_exposures=10368, planned_model_constructions=30, physical_updates=108,
        physical_episode_exposures=10368, completed_model_template_constructions=30,
        physical_work_unknown=False, automatic_promotion=False, evaluation_performed=False,
        fitness_scores_read=False, historical_checkpoint_reads=0, sources_unchanged=True,
        work=EXPECTED_WORK, operations={name: dict(attempted=count, completed=count)
                                       for name, count in EXPECTED_OPERATIONS.items()})
    if (_bytes({key: record.get(key) for key in contract}) != _bytes(contract)
            or record.get("source_sha256") != trainer_sources() or record.get("producer_source_sha256") != source_hashes()
            or record.get("execution_profile", {}).get("schema") != "bic-strict-local-execution-v1"
            or record.get("execution_profile", {}).get("cpu_threads") != 1
            or record.get("execution_profile", {}).get("cpu_interop_threads") != 1
            or record.get("execution_profile", {}).get("device", {}).get("name") != "NVIDIA GeForce RTX 5080"
            or record.get("execution_profile", {}).get("device", {}).get("index") != 0):
        raise ValueError("complete matching fixed CUDA proof required")
    cases = record.get("results")
    if type(cases) is not list or len(cases) != 6:
        raise ValueError("six exact objective/rate proof cases required")
    for expected, actual in zip(CASES, cases):
        if (type(actual) is not dict or _bytes({key: actual.get(key) for key in expected}) != _bytes(expected)
                or actual.get("status") != "passed" or actual.get("checks") != dict.fromkeys(CASE_CHECKS, True)
                or any(value is not True for value in actual["checks"].values())
                or type(actual.get("physical_updates")) is not int or actual["physical_updates"] != 18
                or type(actual.get("episode_exposures")) is not int or actual["episode_exposures"] != 1728
                or type(actual.get("comparisons")) is not list or len(actual["comparisons"]) != 16):
            raise ValueError("proof case or every-step comparisons differ")
        for (phase, cursor), comparison in zip(COMPARISONS, actual["comparisons"]):
            if (type(comparison) is not dict or set(comparison) != {"phase", "cursor", "checks", "exact",
                    "ignored_fields", "expected_sha256", "actual_sha256"}
                    or comparison["phase"] != phase or type(comparison["cursor"]) is not int
                    or comparison["cursor"] != cursor or comparison["exact"] is not True
                    or comparison["checks"] != dict.fromkeys(CHECK_FIELDS, True)
                    or any(value is not True for value in comparison["checks"].values())
                    or comparison["ignored_fields"] != ["timing"]
                    or type(comparison["expected_sha256"]) is not dict
                    or set(comparison["expected_sha256"]) != set(CHECK_FIELDS)
                    or comparison["actual_sha256"] != comparison["expected_sha256"]
                    or any(type(digest) is not str or len(digest) != 64
                        or any(char not in "0123456789abcdef" for char in digest)
                        for digest in comparison["expected_sha256"].values())):
                raise ValueError("exact phase/cursor/component comparison contract differs")
    observed = {path.relative_to(directory).as_posix(): _sha(path.read_bytes())
        for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "probe.json"}
    if observed != record.get("artifact_sha256"):
        raise ValueError("runtime proof artifact inventory changed")
    return copy.deepcopy(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--max-seconds", type=float, default=14400)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configure_strict_profile()
    record = run(args.output, device=args.device, max_seconds=args.max_seconds)
    path = Path(args.output) / "probe.json"
    print(json.dumps(dict(status=record["status"], proof_path=str(path.absolute()),
        proof_sha256=_sha(path.read_bytes()), physical_updates=record.get("physical_updates"),
        episode_exposures=record.get("physical_episode_exposures")), allow_nan=False))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
