"""One-shot, caller-gated CUDA arithmetic proof for bounded CPU preparation.

Importing does not configure Torch or start work. The external launcher must
first authenticate the completed objective study and confirm its processes have
exited. This module has no waiting, retry, continuation, scoring or promotion
path. Its only historical input is the explicitly pinned CPU validation receipt.
Per-step comparisons retain typed byte digests, never a history of tensors or
snapshots that would drain a live preparation queue.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
import struct
import time

import torch

from experiments import foundation_training as canonical
from experiments import foundation_objective_training as synchronous
from experiments import foundation_prefetched_training as prefetched
from experiments.execution_profile import assert_strict_profile, runtime_profile
from experiments.foundation_admission import repair_plan
from experiments.foundation_curriculum import DEPTHS, FAMILIES, TURN_BUCKETS
from experiments.foundation_objective_runtime_probe import _bytes, _publish as _publish_image, _sha, _utc
from experiments.foundation_plan import build_plan, materialize_pair
from experiments.foundation_plan_index import AuthenticatedPlanIndex
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-foundation-prefetched-runtime-proof-v1"
CPU_RECEIPT = "runs/foundation-prefetched-training-validation-local/attempt-002/report.json"
CPU_RECEIPT_SHA256 = "2edb9d74f01dd69dd72709e2a3622ffe94b20ecb1ec6aaa06464ef9e60981343"
CONFIG = SequenceConfig(width=192, layers=4, heads=4, feedforward=768, max_turns=12)
PLAN_OPTIONS = dict(seed=831910001, ordering_seed=831910002, stage_updates=10,
                    final_updates=6, micro_batch_size=32, rehearsal_every=2)
MODEL_SEED, STEPS, MIDPOINT, ORDER = 831910799, 66, 33, "curriculum"
CASES = (dict(id="baseline", objective_id="baseline", learning_rate=.003),
         dict(id="balanced_reply", objective_id="balanced_reply", learning_rate=.001))
COMPONENTS = ("weights", "optimizer", "cursor", "evidence")
LOSSES = ("loss", "action_loss", "reply_loss", "observation_language_loss")
SNAPSHOT_FIELDS = {"schema", "recipe", "weights", "optimizer", "cursor", "evidence", "timing"}
EXPECTED_WORK = dict(optimizer_step_calls_attempted=396, optimizer_step_calls_completed=396,
    known_physical_optimizer_updates=396, retained_optimizer_updates=396,
    optimizer_completion_unknown_calls=0, neural_attempted_microbatches=1188,
    completed_microbatches=1188, neural_attempted_episode_exposures=38016,
    completed_microbatch_episode_exposures=38016)
COMPARISONS = ([('split', step) for step in range(34)] + [('resumed_fresh', 0), ('resumed', 33)]
              + [('resumed', step) for step in range(34, 67)]
              + [('repeat', step) for step in range(67)])


def _native_path(path):
    """Use the Windows extended namespace for all output I/O, not identities."""
    absolute = os.path.abspath(os.fspath(path))
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def _publish(path, image):
    # Keep the frozen exclusive temporary-file/fsync/hard-link publication.
    # Its UUID suffix can exceed MAX_PATH even when the destination does not.
    return _publish_image(_native_path(path), image)


def _json(path, value):
    # The imported helper's _json would call its own unwrapped _publish.
    return _publish(path, _bytes(value) + b"\n")


def source_hashes():
    """Current declared producer closure plus the fixed prerequisite image."""
    names = ("experiments/foundation_prefetched_runtime_probe.py",
             "experiments/foundation_objective_runtime_probe.py",
             "experiments/execution_profile.py", "docs/FOUNDATION_PREFETCH_CUDA_PROTOCOL.md", CPU_RECEIPT)
    return {**prefetched.source_hashes(), **{name: _sha((ROOT/name).read_bytes()) for name in names}}


def _tensor_metadata(value):
    if value.device.type != "cpu" or value.layout != torch.strided:
        raise ValueError("comparison requires canonical CPU strided tensors")
    return dict(dtype=str(value.dtype), shape=list(value.shape), layout=str(value.layout),
                stride=list(value.stride()), storage_offset=value.storage_offset(),
                requires_grad=value.requires_grad, device="cpu")


def _tensor_bytes(value):
    return value.detach().contiguous().numpy().tobytes()


def _node(value):
    """Typed canonical tree; scalar float bytes preserve signed zero exactly."""
    if isinstance(value, torch.Tensor):
        return ["tensor", _tensor_metadata(value), _sha(_tensor_bytes(value))]
    if type(value) is dict:
        pairs = [[_node(key), _node(item)] for key, item in value.items()]
        return ["dict", sorted(pairs, key=lambda item: _bytes(item[0]))]
    if type(value) in (list, tuple):
        return [type(value).__name__, [_node(item) for item in value]]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite comparison scalar")
        return ["float64", struct.pack(">d", value).hex()]
    if value is None or type(value) in (str, int, bool):
        return [type(value).__name__, value]
    raise TypeError("unsupported comparison value: " + type(value).__name__)


def _digest(value):
    return _sha(_bytes(_node(value)))


def _same(left, right):
    """Direct typed tree equality, used for the entire saved restart envelope."""
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return _tensor_metadata(left) == _tensor_metadata(right) and _tensor_bytes(left) == _tensor_bytes(right)
    if type(left) is dict:
        left_keys = sorted(left, key=lambda key: _bytes(_node(key)))
        right_keys = sorted(right, key=lambda key: _bytes(_node(key)))
        return len(left_keys) == len(right_keys) and all(
            _same(a, b) and _same(left[a], right[b]) for a, b in zip(left_keys, right_keys))
    if type(left) in (tuple, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return _node(left) == _node(right)


def _authenticate_cpu(path, expected):
    path = Path(path).resolve()
    if path != (ROOT/CPU_RECEIPT).resolve() or expected != CPU_RECEIPT_SHA256:
        raise ValueError("the declared caller-pinned passed CPU receipt is required")
    image = path.read_bytes()
    if _sha(image) != expected:
        raise ValueError("CPU prerequisite bytes differ from the caller pin")
    record = json.loads(image)
    contract = dict(schema="bic-prefetched-training-validation-v1", status="passed", tests_run=6,
        failures=0, errors=0, cuda_initialized_before=False, cuda_initialized_after=False,
        source_and_evidence_unchanged=True, setup_or_neural_work_may_be_unknown=False)
    if _bytes({key: record.get(key) for key in contract}) != _bytes(contract):
        raise ValueError("CPU prerequisite did not pass the declared six checks")
    launch_image = (path.parent/"launch.json").read_bytes()
    log_image = (path.parent/"tests.log").read_bytes()
    if _sha(launch_image) != record["launch_sha256"] or _sha(log_image) != record["log_sha256"]:
        raise ValueError("CPU prerequisite launch/transcript binding differs")
    launch = json.loads(launch_image)
    if any(launch["file_sha256"].get(name) != digest for name, digest in prefetched.source_hashes().items()):
        raise ValueError("current trainer/preparation closure differs from passed CPU proof")
    return dict(path=CPU_RECEIPT, sha256=expected, launch_sha256=record["launch_sha256"],
                log_sha256=record["log_sha256"], tests_run=6, status="passed")


class _Journal:
    """Durable operation boundaries; nonpreemptive work can overrun its deadline."""
    def __init__(self, directory, started, cpu_started, deadline):
        self.path = _native_path(directory)/"operations.jsonl"
        _publish(self.path, b"")
        self.started, self.cpu_started, self.deadline = started, cpu_started, deadline
        self.sequence, self.previous, self.active = 0, None, None
        self.counts, self.costs = {}, {}
        self.work = dict.fromkeys(EXPECTED_WORK, 0)
        self.unknown = False

    def event(self, event_kind, **fields):
        record = dict(sequence=self.sequence, previous_event_sha256=self.previous,
                      utc=_utc(), event=event_kind, **fields)
        image = _bytes(record)+b"\n"
        try:
            with self.path.open("ab") as stream:
                stream.write(image)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # Even a later successful failure record cannot repair a partial
            # journal line or prove that the earlier fsync reached storage.
            self.unknown = True
            raise
        self.previous, self.sequence = _sha(image), self.sequence+1

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("global proof allowance elapsed; no retry or extension")

    def perform(self, kind, function, *, context=None, summarize=None, failure=None, cleanup=False):
        if not cleanup:
            self.check_time()
        count = self.counts.setdefault(kind, dict(attempted=0, completed=0, failed=0))
        intent = dict(kind=kind, context=context or {}, operation=self.sequence)
        self.event("intent", **intent)
        count["attempted"] += 1
        self.active = intent
        started, cpu = time.monotonic(), time.process_time()
        try:
            value = function()
            count["completed"] += 1
            details = summarize(value) if summarize else {}
            self.event("completed", **intent, wall_seconds=time.monotonic()-started,
                       cpu_seconds=time.process_time()-cpu, details=details)
            self.active = None
            return value
        except BaseException as error:
            count["failed"] += 1
            try:
                details = failure() if failure else {}
                self.event("failed", **intent, wall_seconds=time.monotonic()-started,
                    cpu_seconds=time.process_time()-cpu, error=repr(error), details=details)
                self.active = None
            except BaseException:
                self.unknown = True
            raise
        finally:
            cost = self.costs.setdefault(kind, dict(attempted_wall_seconds=0., attempted_process_cpu_seconds=0.))
            cost["attempted_wall_seconds"] += time.monotonic()-started
            cost["attempted_process_cpu_seconds"] += time.process_time()-cpu

    def step_work(self, report, completed):
        self.work["optimizer_step_calls_completed"] += int(completed)
        physical = report.get("physical_optimizer_updates")
        if physical is None:
            self.work["optimizer_completion_unknown_calls"] += 1
        else:
            self.work["known_physical_optimizer_updates"] += physical
        for name in ("retained_optimizer_updates", "neural_attempted_microbatches", "completed_microbatches",
                     "neural_attempted_episode_exposures", "completed_microbatch_episode_exposures"):
            if type(report.get(name)) is int:
                self.work[name] += report[name]


def _peak_ram():
    """Best-effort process high-water mark; no polling or extra dependencies."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
            value = Counters()
            value.cb = ctypes.sizeof(value)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
                raise OSError(ctypes.get_last_error())
            return dict(bytes=int(value.PeakWorkingSetSize), scope="process lifetime peak working set")
        import resource
        return dict(bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                    scope="process lifetime ru_maxrss, KiB on this platform")
    except Exception as error:
        return dict(bytes=None, unavailable=repr(error))


def run(directory, *, cpu_receipt_path, expected_cpu_receipt_sha256, expected_source_sha256,
        device="cuda:0", max_seconds=900, absolute_deadline=None):
    """Execute exactly the declared proof after the external completed-study gate.

    The caller configures strict execution and one CPU intra/inter-op thread.
    absolute_deadline carries the launcher's allowance across authentication and
    imports. Snapshot/restore never renews it. Setup/operations are nonpreemptive;
    cleanup is bounded separately and any overrun or incomplete work is reported.
    """
    started, cpu_started = time.monotonic(), time.process_time()
    if device != "cuda:0":
        raise ValueError("fixed local CUDA proof requires cuda:0; no fallback")
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 900:
        raise ValueError("positive finite global allowance at most900 seconds required")
    deadline = started+max_seconds
    if absolute_deadline is not None:
        if (type(absolute_deadline) not in (int, float) or not math.isfinite(absolute_deadline)
                or absolute_deadline <= started):
            raise ValueError("absolute deadline must be finite and still in the future")
        deadline = min(deadline, absolute_deadline)
    expected = copy.deepcopy(expected_source_sha256)
    prerequisite = _authenticate_cpu(cpu_receipt_path, expected_cpu_receipt_sha256)
    if type(expected) is not dict or source_hashes() != expected:
        raise ValueError("exact externally pinned proof source/artifact closure required")
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError("caller must configure one CPU intra/inter-op thread")
    assert_strict_profile()
    if time.monotonic() >= deadline:
        raise TimeoutError("global allowance elapsed during probe authentication")
    directory = _native_path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    journal = _Journal(directory, started, cpu_started, deadline)
    record = dict(schema=SCHEMA, status="running", pid=os.getpid(), started_utc=_utc(),
        source_sha256=expected, cpu_prerequisite=prerequisite, config=asdict(CONFIG),
        plan_options=PLAN_OPTIONS, model_seed=MODEL_SEED, cases=list(CASES), order=ORDER,
        steps=STEPS, midpoint=MIDPOINT, max_seconds=max_seconds, absolute_deadline=absolute_deadline,
        deadline_monotonic=deadline, remaining_seconds_at_entry=deadline-started,
        inferred_pre_probe_allowance_used_seconds=max(0., max_seconds-(deadline-started)),
        authentication_wall_seconds=time.monotonic()-started, results=[], preparation_invocations=[],
        planned_work=EXPECTED_WORK, planned_fresh_models=8, planned_restore_templates=2,
        evaluation_performed=False, fitness_scores_read=False, automatic_promotion=False,
        automatic_retry=False, historical_checkpoint_reads=0,
        comparison="typed SHA256 byte-digest equality of canonical detached CPU states; direct full-tree midpoint equality",
        scope="Fixed same-process local CUDA trajectories only. External launcher owns the completed-study/process-exit gate. No capability, cross-process restart or throughput claim.")
    _json(directory/"started.json", record)
    active_owners, final_preparation = {}, {}
    failure_error = None
    forward_work = dict(native_forward_attempts=0, native_forward_completions=0,
                        attempted_episode_forwards=0, completed_episode_forwards=0,
                        first_forward_deadline_refusals=0)
    update_boundary = dict(active=False, first=True)

    def install_forward_boundary(trainer):
        # Per-instance observation hooks: no frozen method/global replacement,
        # no tensor changes. A restored model needs new hooks after its swap.
        def before(module, args, kwargs):
            if not update_boundary["active"]:
                raise RuntimeError("unexpected forward outside a journaled training step")
            if update_boundary["first"]:
                try:
                    journal.check_time()
                except TimeoutError:
                    forward_work["first_forward_deadline_refusals"] += 1
                    raise
                update_boundary["first"] = False
            forward_work["native_forward_attempts"] += 1
            forward_work["attempted_episode_forwards"] += int(kwargs["token_ids"].shape[0])
        def after(module, args, kwargs, output):
            forward_work["native_forward_completions"] += 1
            forward_work["completed_episode_forwards"] += int(kwargs["token_ids"].shape[0])
        trainer.model.register_forward_pre_hook(before, with_kwargs=True)
        trainer.model.register_forward_hook(after, with_kwargs=True)

    def guard():
        assert_strict_profile()
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1 or source_hashes() != expected:
            raise ValueError("proof runtime threads or source/artifact identity changed")

    def remember_owner(key, trainer):
        report = trainer.preparation_report
        final_preparation[key] = dict(key=key, report=report)
        if trainer._preparation is None:
            active_owners.pop(key, None)
        return report

    def finish_owner(key, trainer, *, cleanup=False):
        def finish():
            trainer.close_preparation(join_seconds=1)
            return remember_owner(key, trainer)
        return journal.perform("preparation_close", finish, context=dict(owner=key), cleanup=cleanup,
            summarize=lambda value: value or {}, failure=lambda: dict(report=remember_owner(key, trainer)))

    try:
        def preserve_sources():
            for name, digest in expected.items():
                journal.check_time()
                image = (ROOT/name).read_bytes()
                if _sha(image) != digest:
                    raise ValueError("source changed while preserving: " + name)
                target = directory/"sources"/name
                target.parent.mkdir(parents=True, exist_ok=True)
                _publish(target, image)
            return dict(files=len(expected))
        journal.perform("preserve_sources", preserve_sources, summarize=lambda value: value)
        runtime = journal.perform("runtime_setup", lambda: runtime_profile(device), summarize=lambda value: value)
        if runtime["device"]["name"] != "NVIDIA GeForce RTX 5080":
            raise ValueError("declared local RTX5080 required")
        record["execution_profile"] = runtime
        torch.cuda.reset_peak_memory_stats(device)
        _json(directory/"runtime.json", runtime)
        base = journal.perform("plan_build", lambda: build_plan(**PLAN_OPTIONS))
        history = journal.perform("protection_pair_generation", lambda: sorted(
            transcript_digest(row) for row in materialize_pair(base, 0, "color", 0)),
            summarize=lambda value: dict(generated_pairs=1, generated_episodes=2, protected_transcripts=value))
        if len(history) != 2 or len(set(history)) != 2:
            raise ValueError("fixed protection pair must have two distinct transcripts")
        plan, admission = journal.perform("plan_admission", lambda: repair_plan(base, history),
            summarize=lambda value: dict(counts=value[1]["counts"]))
        if not any(row["bundle_id"] == 0 and row["family"] == "color" and row["pair_index"] == 0
                   for row in admission["overrides"]):
            raise ValueError("the deliberately protected pair was not repaired")
        schedule = plan["schedules"][ORDER]
        cells = {(plan["bundles"][str(bundle)]["depth"], plan["bundles"][str(bundle)]["turns"])
                 for bundle in schedule}
        if len(schedule) != STEPS or cells != {(depth, turns) for depth in DEPTHS for turns in TURN_BUCKETS}:
            raise ValueError("complete66-bundle all-depth/all-length fixture required")
        index = journal.perform("index_construction", lambda: AuthenticatedPlanIndex(plan,
            admission_protected_transcripts=history, protected_transcripts=history, admission_receipt=admission),
            summarize=lambda value: value.construction)
        replay = journal.perform("index_metadata_replay", lambda: index.replay(ORDER, STEPS, include_bundles=True))
        record.update(index_identity=index.identity, setup_work=dict(protection_pairs_generated=1,
            protection_episodes_generated=2, initial_repair_counts=copy.deepcopy(admission["counts"]),
            index_construction=index.construction,
            scope="Original protection generation, initial repair, and index admission reconstruction/admitted scan are distinct passes; nested canonical validation is not an extra learner exposure."),
            coverage=dict(bundles=STEPS, cells=[list(cell) for cell in sorted(cells)]))
        record["input_sha256"] = {}
        for name, value in (("base-plan.json", base), ("plan.json", plan), ("admission.json", admission),
                            ("protected.json", history), ("index-identity.json", index.identity),
                            ("index-construction.json", index.construction)):
            record["input_sha256"][name] = _json(directory/name, value)
        guard()

        def snapshot(trainer, context):
            def take():
                guard()
                result = trainer.snapshot()
                if set(result) != SNAPSHOT_FIELDS:
                    raise ValueError("exact seven-field snapshot required")
                guard()
                return result
            return journal.perform("snapshot", take, context=context,
                summarize=lambda value: dict(cursor=value["cursor"], payload_sha256=_digest(value)))

        def save_checkpoint(path, payload, context):
            def save():
                stream = io.BytesIO()
                torch.save(payload, stream)
                image = stream.getvalue()
                return dict(path=path.relative_to(directory).as_posix(), sha256=_publish(path, image), bytes=len(image))
            return journal.perform("checkpoint_save", save, context=context, summarize=lambda value: value)

        def capture(trainer, context, step_report=None):
            def collect():
                guard()
                trainer._assert_sources()
                trainer._sync()
                state = dict(weights=canonical._cpu_copy(trainer.model.state_dict()),
                    optimizer=canonical._cpu_copy(trainer.optimizer.state_dict()), cursor=trainer.cursor,
                    evidence=copy.deepcopy(trainer._evidence))
                value = dict(cursor=trainer.cursor, sha256={name: _digest(state[name]) for name in COMPONENTS},
                             losses=None, microbatches=None)
                if step_report is not None:
                    value.update(losses={name: step_report[name] for name in LOSSES},
                                 microbatches=copy.deepcopy(step_report["microbatches"]))
                trainer._assert_sources()
                guard()
                return value
            return journal.perform("state_capture", collect, context=context, summarize=lambda value: value)

        initial_weights_digest = None
        for case in CASES:
            journal.check_time()
            guard()
            case_started = time.monotonic()
            case_path = directory/case["id"]
            case_path.mkdir(exist_ok=False)
            case_record = dict(case, status="running", comparisons=[], checkpoints={}, checks={}, recipes={})
            record["results"].append(case_record)
            common_recipe = None

            def context(phase, **extra):
                return dict(case=case["id"], phase=phase, **extra)

            def create(phase, prepared):
                nonlocal common_recipe
                trainer_type = prefetched.PrefetchedObjectiveFoundationTrainer if prepared else synchronous.ObjectiveFoundationTrainer
                guard()
                value = journal.perform("trainer_construction", lambda: trainer_type(plan, ORDER,
                    objective_id=case["objective_id"], seed=MODEL_SEED, config=CONFIG, learning_rate=case["learning_rate"],
                    device=device, admission_protected_transcripts=history, protected_transcripts=history,
                    admission_receipt=admission, plan_index=index), context=context(phase),
                    summarize=lambda trainer: trainer.setup_report)
                recipe = value.recipe
                current_common = {key: item for key, item in recipe.items()
                                  if key not in ("schema", "source_sha256", "preparation", "timing_contract")}
                if common_recipe is None:
                    common_recipe = current_common
                if not _same(common_recipe, current_common):
                    raise ValueError("common objective/config/index/optimizer recipe changed")
                wanted_schema = prefetched.SCHEMA if prepared else synchronous.SCHEMA
                wanted_sources = prefetched.source_hashes() if prepared else synchronous.source_hashes()
                if recipe["schema"] != wanted_schema or recipe["source_sha256"] != wanted_sources:
                    raise ValueError("exact implementation recipe/source required")
                if sum(parameter.numel() for parameter in value.model.parameters()) != 2221738:
                    raise ValueError("model parameter count differs")
                install_forward_boundary(value)
                case_record["recipes"][phase] = recipe
                guard()
                return value

            def begin(trainer, phase, stop):
                key = case["id"]+"/"+phase
                active_owners[key] = trainer
                def start():
                    guard()
                    remaining = journal.deadline-time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("no global allowance remains for preparation")
                    trainer.start_preparation(stop_cursor=stop, max_seconds=remaining)
                    original = trainer._preparation_deadline
                    trainer._preparation_deadline = min(original, journal.deadline)
                    trainer._preparation_invocation["deadline_monotonic"] = trainer._preparation_deadline
                    owner = trainer._preparation
                    with owner._condition:
                        original_producer = owner._deadline
                        owner._deadline = min(original_producer, journal.deadline)
                        clamped_producer = owner._deadline
                        owner._condition.notify_all()
                    return dict(owner=key, original_caller_deadline=original,
                        clamped_caller_deadline=trainer._preparation_deadline, global_deadline=journal.deadline,
                        original_producer_deadline=original_producer, clamped_producer_deadline=clamped_producer,
                        report=trainer.preparation_report,
                        scope="Proof-only caller and condition-locked producer deadline clamps after constructor return; already active construction or CPU preparation may finish nonpreemptively.")
                journal.perform("preparation_start", start, context=context(phase, stop_cursor=stop),
                    summarize=lambda value: value, failure=lambda: dict(report=remember_owner(key, trainer)))
                return key

            def advance(trainer, phase):
                guard()
                previous = trainer.last_report
                accounted = False
                def invoke():
                    journal.work["optimizer_step_calls_attempted"] += 1
                    update_boundary.update(active=True, first=True)
                    try:
                        return trainer.step()
                    finally:
                        update_boundary["active"] = False
                def summarize(value):
                    nonlocal accounted
                    journal.step_work(value, True)
                    accounted = True
                    # The full cumulative owner report is retained once at close.
                    return {key: copy.deepcopy(item) for key, item in value.items() if key != "preparation"}
                def failed():
                    nonlocal accounted
                    value = None if trainer.last_report is previous else copy.deepcopy(trainer.last_report)
                    if not accounted:
                        journal.step_work(value or {}, False)
                        accounted = True
                    return dict(last_report=value)
                value = journal.perform("training_step", invoke, context=context(phase, cursor_before=trainer.cursor),
                                        summarize=summarize, failure=failed)
                bundle = replay["bundles"][trainer.cursor-1]
                expected_rows = [dict(family=family, depth=bundle["depth"], turns=bundle["turns"],
                                     **bundle["families"][family]) for family in FAMILIES]
                actual_rows = [{key: item for key, item in row.items() if key not in LOSSES}
                               for row in value["microbatches"]]
                if (value["bundle_id"] != bundle["bundle_id"] or not _same(expected_rows, actual_rows)
                        or value["physical_optimizer_updates"] != 1 or value["retained_optimizer_updates"] != 1
                        or value["neural_attempted_microbatches"] != 3 or value["completed_microbatches"] != 3
                        or value["neural_attempted_episode_exposures"] != 96
                        or value["completed_microbatch_episode_exposures"] != 96):
                    raise ValueError("step work or ordered canonical family evidence differs")
                guard()
                return capture(trainer, context(phase), value)

            def compare(expected_state, actual, phase, cursor, *, losses=True):
                checks = {name: expected_state["sha256"][name] == actual["sha256"][name] for name in COMPONENTS}
                checks.update(losses=not losses or _same(expected_state["losses"], actual["losses"]),
                              ordered_family_evidence=not losses or _same(expected_state["microbatches"], actual["microbatches"]))
                value = dict(phase=phase, cursor=cursor, checks=checks, exact=all(checks.values()),
                    expected_sha256=expected_state["sha256"], actual_sha256=actual["sha256"],
                    loss_comparison="exact four aggregate scalars and every family loss/evidence" if losses else "no new update; state-only boundary",
                    expected_losses=expected_state["losses"] if losses else None,
                    actual_losses=actual["losses"] if losses else None)
                case_record["comparisons"].append(value)
                journal.event("comparison", case=case["id"], **value)
                if not value["exact"]:
                    raise ValueError("matched byte-digest/loss/family evidence differs: "+case["id"]+"/"+phase+"/"+str(cursor))

            def endpoint(trainer, phase):
                payload = snapshot(trainer, context(phase))
                if not _same(payload["evidence"], replay["evidence"]) or payload["cursor"] != STEPS:
                    raise ValueError("final actual index prefix differs")
                artifact = save_checkpoint(case_path/(phase+".pt"), payload, context(phase))
                case_record["checkpoints"][phase] = artifact

            reference = create("reference", False)
            refs = {0: capture(reference, context("reference"))}
            if initial_weights_digest is None:
                initial_weights_digest = refs[0]["sha256"]["weights"]
                record["common_initial_weights_sha256"] = initial_weights_digest
            if initial_weights_digest != refs[0]["sha256"]["weights"]:
                raise ValueError("objectives do not share seeded initial weights")
            for cursor in range(1, STEPS+1):
                refs[cursor] = advance(reference, "reference")
            endpoint(reference, "reference")
            del reference

            split = create("split", True)
            compare(refs[0], capture(split, context("split")), "split", 0, losses=False)
            key = begin(split, "split", MIDPOINT)
            for cursor in range(1, MIDPOINT+1):
                compare(refs[cursor], advance(split, "split"), "split", cursor)
            finish_owner(key, split)
            middle = snapshot(split, context("midpoint"))
            artifact = save_checkpoint(case_path/"midpoint.pt", middle, context("midpoint"))
            case_record["checkpoints"]["midpoint"] = artifact
            del split

            def load():
                image = (directory/artifact["path"]).read_bytes()
                if len(image) != artifact["bytes"] or _sha(image) != artifact["sha256"]:
                    raise ValueError("saved midpoint changed before weights-only decoding")
                return torch.load(io.BytesIO(image), map_location="cpu", weights_only=True)
            loaded = journal.perform("checkpoint_load", load, context=context("midpoint"),
                summarize=lambda value: dict(path=artifact["path"], sha256=artifact["sha256"], cursor=value["cursor"]))
            if not _same(middle, loaded):
                raise ValueError("CPU saved/reloaded seven-field payload differs directly")
            resumed = create("resumed", True)
            compare(refs[0], capture(resumed, context("resumed_fresh")), "resumed_fresh", 0, losses=False)
            journal.perform("strict_restore", lambda: resumed.restore(loaded), context=context("resumed"),
                summarize=lambda value: dict(cursor=value.cursor, restore_seconds=value.last_restore_seconds))
            install_forward_boundary(resumed)
            restored = snapshot(resumed, context("restored_midpoint"))
            if not _same(middle, restored) or not _same(loaded, restored):
                raise ValueError("strict restore changed the full seven-field midpoint")
            case_record["checks"]["direct_seven_field_saved_loaded_restored_equality"] = True
            compare(refs[MIDPOINT], capture(resumed, context("resumed")), "resumed", MIDPOINT, losses=False)
            del middle, loaded, restored
            key = begin(resumed, "resumed", STEPS)
            for cursor in range(MIDPOINT+1, STEPS+1):
                compare(refs[cursor], advance(resumed, "resumed"), "resumed", cursor)
            finish_owner(key, resumed)
            endpoint(resumed, "resumed")
            del resumed

            repeat = create("repeat", True)
            compare(refs[0], capture(repeat, context("repeat")), "repeat", 0, losses=False)
            key = begin(repeat, "repeat", STEPS)
            for cursor in range(1, STEPS+1):
                compare(refs[cursor], advance(repeat, "repeat"), "repeat", cursor)
            finish_owner(key, repeat)
            endpoint(repeat, "repeat")
            del repeat, refs
            if [(row["phase"], row["cursor"]) for row in case_record["comparisons"]] != COMPARISONS:
                raise ValueError("exact136 comparisons per objective required")
            case_record["checks"].update(common_seeded_initial_weights=True, common_learning_recipe=True,
                every_matched_state_byte_digest_equal=True, every_matched_loss_and_family_equal=True,
                final_actual_index_evidence_equal=True)
            case_record.update(status="passed", physical_updates=198, wall_seconds=time.monotonic()-case_started)
            _json(case_path/"result.json", case_record)

        if journal.work != EXPECTED_WORK:
            raise ValueError("actual learner work differs from fixed396-update allowance")
        for kind, count in (("trainer_construction", 8), ("strict_restore", 2), ("preparation_start", 6),
                            ("checkpoint_load", 2), ("training_step", 396)):
            if journal.counts.get(kind) != dict(attempted=count, completed=count, failed=0):
                raise ValueError("fixed operation count differs: "+kind)
        for key, value in final_preparation.items():
            wrapper = value["report"]
            owner = wrapper["owner"]
            window = wrapper["invocation"]["stop_cursor"]-wrapper["invocation"]["start_cursor"]
            counts = owner["counts"]
            if (not owner["accounting_final"] or not owner["accounting_complete"]
                    or owner["worker_alive"] or owner["active_work_incomplete"]
                    or owner["unreturned_operation_work_may_be_unknown"]
                    or any(counts[name] != window for name in ("submitted_bundles", "prepared_bundles", "delivered_leases", "released_leases", "materialized_bundles", "evidence_validated_bundles"))
                    or counts["materialized_episodes"] != 96*window
                    or counts["packed_family_batches"] != 3*window or counts["packed_episodes"] != 96*window
                    or any(counts[name] for name in ("discarded_prepared_bundles", "abandoned_leases", "interrupted_preparations", "failed_preparations", "consumer_rejections"))):
                raise ValueError("preparation window accounting differs: "+key)
        if len(final_preparation) != 6 or active_owners:
            raise ValueError("exact six joined preparation invocations required")
        if forward_work != dict(native_forward_attempts=1188, native_forward_completions=1188,
                attempted_episode_forwards=38016, completed_episode_forwards=38016,
                first_forward_deadline_refusals=0):
            raise ValueError("physical forward instrumentation differs from fixed workload")
        record["training_input_work"] = dict(returned_canonical_bundles=396, family_packing_calls=1188,
            packed_episode_passes=38016, synchronous_bundles=132, prepared_bundles=264,
            scope="Synchronous counts follow completed frozen three-family steps; prepared counts are checked against all six final owner reports. Setup and nested canonical validation are separately recorded; no monkeypatch instrumentation.")
        def final_guard():
            guard()
            if runtime_profile(device) != runtime:
                raise ValueError("runtime identity changed")
            return dict(runtime_unchanged=True, sources_unchanged=True)
        journal.perform("final_guard", final_guard, summarize=lambda value: value)
        record.update(status="passed", sources_unchanged=True, physical_work_unknown=False)
        journal.event("proof_completed", status="passed", work=journal.work)
    except BaseException as error:
        failure_error = error
        record.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                      error=repr(error), active_operation=copy.deepcopy(journal.active))
        try:
            journal.event("proof_failed", status=record["status"], error=record["error"], work=journal.work)
        except BaseException as error:
            record["accounting_error"] = repr(error)
            journal.unknown = True
    finally:
        cleanup_errors = []
        for key, trainer in list(active_owners.items()):
            try:
                finish_owner(key, trainer, cleanup=True)
            except BaseException as error:
                problem = dict(owner=key, error=repr(error))
                cleanup_errors.append(problem)
                # A broken journal must never prevent a direct bounded stop.
                # Keep this fallback explicitly unjournaled and the proof failed.
                tick, cpu = time.monotonic(), time.process_time()
                try:
                    trainer.close_preparation(join_seconds=1)
                    problem["unjournaled_bounded_close_returned"] = True
                except BaseException as close_error:
                    problem["unjournaled_bounded_close_error"] = repr(close_error)
                finally:
                    problem["unjournaled_close_wall_seconds"] = time.monotonic()-tick
                    problem["unjournaled_close_process_cpu_seconds"] = time.process_time()-cpu
                try:
                    remember_owner(key, trainer)
                except BaseException as reporting_error:
                    cleanup_errors.append(dict(owner=key, reporting_error=repr(reporting_error)))
        try:
            if torch.cuda.is_initialized():
                torch.cuda.synchronize(device)
                record.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                              peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
        except BaseException as error:
            cleanup_errors.append(dict(synchronization_error=repr(error)))
        if cleanup_errors or active_owners:
            record.update(status="failed", cleanup_errors=cleanup_errors,
                          unjoined_preparation_owners=sorted(active_owners))
        record["preparation_invocations"] = list(final_preparation.values())
        preparation_unknown = any(value["report"] is None or value["report"].get("owner") is None
            or not value["report"]["owner"].get("accounting_complete", False) for value in final_preparation.values())
        if record["status"] == "passed" and time.monotonic() >= deadline:
            failure_error = TimeoutError("global allowance elapsed before terminal evidence inventory")
            record.update(status="failed", error=repr(failure_error))
        # Terminal evidence publication remains necessary after a failed budget
        # check. It cannot authorize more model, data or preparation work.
        artifact_map = {}
        def inventory():
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.name != "probe.json" and path != journal.path:
                    artifact_map[path.relative_to(directory).as_posix()] = _sha(path.read_bytes())
            return dict(files=len(artifact_map))
        try:
            journal.perform("artifact_inventory", inventory, cleanup=True, summarize=lambda value: value)
            # Hash the journal only after its final inventory event is durable.
            artifact_map["operations.jsonl"] = _sha(journal.path.read_bytes())
        except BaseException as error:
            record.update(status="failed", artifact_inventory_error=repr(error))
            journal.unknown = True
            if failure_error is None:
                failure_error = error
        record.update(ended_utc=_utc(), wall_seconds=time.monotonic()-started,
            process_cpu_seconds=time.process_time()-cpu_started, deadline_overrun_seconds=max(0., time.monotonic()-deadline),
            work=copy.deepcopy(journal.work), operations=copy.deepcopy(journal.counts),
            physical_forward_work=copy.deepcopy(forward_work),
            backward_count_scope="Completed microbatch reports are appended only after backward returns; attempted backward internals are not separately instrumented.",
            operation_costs=copy.deepcopy(journal.costs), peak_ram=_peak_ram(),
            physical_work_unknown=bool(journal.unknown or journal.active or cleanup_errors
                or journal.work["optimizer_completion_unknown_calls"]), preparation_work_unknown=preparation_unknown,
            setup_work_may_be_unknown=any(journal.counts.get(kind, {}).get("failed", 0)
                for kind in ("plan_build", "protection_pair_generation", "plan_admission", "index_construction")),
            model_construction_scope="Each constructor attempts one fresh model/optimizer; each strict restore attempts one validation template. Failed attempts can contain partial construction work.",
            cost_scope="Whole-probe wall through terminal receipt preparation includes authentication, setup, hashing/copying, checkpoint I/O, final inventory and shutdown; the receipt's own serialization/write is outside its self-reported interval. Per-operation process CPU includes concurrent producer CPU; owner thread CPU is separate and overlapping wall/process intervals must not be added. Launcher reports gate/import cost and the enclosing invocation interval.",
            incomplete_attempt_scope="Missing terminal receipt or unmatched journal intent means unknown work; no automatic retry.")
        record["model_template_operations"] = dict(
            fresh=journal.counts.get("trainer_construction", dict(attempted=0, completed=0, failed=0)),
            restore=journal.counts.get("strict_restore", dict(attempted=0, completed=0, failed=0)))
        record["artifact_sha256"] = artifact_map
        _json(directory/"probe.json", record)
    if failure_error is not None:
        raise failure_error
    if record["status"] != "passed":
        raise RuntimeError("proof cleanup failed; inspect immutable probe.json")
    return copy.deepcopy(record)
