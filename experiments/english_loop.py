"""Budgeted English practice controller with one atomic learning envelope.

This engineering prototype selects practice through an explicit heuristic. It
does not promote a policy, call a tutor, or establish useful autonomous learning.
The backend is transferred to this controller and must not be used separately.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from contextlib import nullcontext

import torch

from brain_in_computer.learning_loop import run_lock
from brain_in_computer.learning_student import _check_finite_tree
from experiments import english_allocation as allocation
from experiments.capacity_evidence import aggregate_rows, validate_metrics
from experiments.realization_backend import RealizationBackend, progress_vector, source_hashes as backend_sources
from experiments.realization_banks import _stats
from experiments.sequence_student import SequenceConfig


SCHEMA = "bic-english-loop-v1"
MAX_CHUNK_UPDATES = 16


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("experiments/english_loop.py", "experiments/english_allocation.py",
             "experiments/execution_profile.py", "experiments/capacity_evidence.py",
             "experiments/summarize_composition_study.py")
    return {**backend_sources(), **{name: _digest(root/name) for name in names}}


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def runtime_identity(backend):
    device = next(backend.model.parameters()).device
    if device.type == "cuda":
        from experiments.execution_profile import runtime_profile
        return runtime_profile(str(device))
    if device.type != "cpu":
        raise ValueError("only local CPU or declared strict CUDA execution is supported")
    return {"device": "cpu", "torch_version": str(torch.__version__),
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


class EnglishLoop:
    """Own an unbound backend; decisions and weights commit in the same file.

    Both evaluation roles must cover every admitted family. Their immutable
    canonical rows are development data, never sealed audits. Fresh revisiting
    is rehearsal on fresh realizations, not an exact-example replay sampler.
    """

    def __init__(self, backend, config, *, max_seen_transcripts, history_limit=32, payload=None):
        if not isinstance(backend, RealizationBackend) or backend._disk_path is not None:
            raise ValueError("controller requires an unbound RealizationBackend")
        if not isinstance(config, allocation.AllocationConfig):
            raise ValueError("explicit allocation configuration required")
        _integer(max_seen_transcripts, "max_seen_transcripts", 1)
        _integer(history_limit, "history_limit", 1)
        self.backend, self.config = backend, config
        self.options = {"max_seen_transcripts": max_seen_transcripts, "history_limit": history_limit}
        self._lock = threading.RLock()
        self._sources = source_hashes()
        self._runtime = runtime_identity(backend)
        self._disk_path, self._disk_digest = None, None
        self._pending_publication, self._uncertain = None, False
        self.last_report = None
        self._validated_responses = OrderedDict()
        self._rows, self._specs = {}, {}
        saved = backend.snapshot()
        if set(backend._evaluations) != {"development", "retention"}:
            raise ValueError("both development and retention banks are required")
        families = set(saved["learner"]["family_microbatches"])
        if set(config.families) != families:
            raise ValueError("allocation families differ from the admitted backend")
        for role, banks in backend._evaluations.items():
            self._rows[role], self._specs[role] = {}, {}
            for name, prepared in banks.items():
                rows = json.loads(prepared._canonical)
                row_families = {row["family"] for row in rows}
                if len(row_families) != 1:
                    raise ValueError("one family per measured cell is required")
                family = next(iter(row_families))
                self._rows[role][name] = rows
                self._specs[role][name] = {"family": family, "identity": prepared.identity}
            if {item["family"] for item in self._specs[role].values()} != families:
                raise ValueError("each evaluation role must cover every family")
        self._queue = [(role, name) for role in sorted(self._rows) for name in sorted(self._rows[role])]
        self._model_config = SequenceConfig(**saved["learner"]["recipe"]["config"])
        self._state = {"phase": "evaluate", "evaluation_kind": "baseline", "evaluation_cursor": 0,
            "evaluation_results": {}, "allocation": allocation.initialize(config, self._specs),
            "base_updates": backend.updates,
            "base_family_microbatches": copy.deepcopy(saved["learner"]["family_microbatches"]),
            "training_work_seconds": 0., "evaluation_work_seconds": 0.,
            "history": [], "windows_completed": 0, "history_dropped": 0}
        if payload is not None:
            self._apply(payload)
        elif saved["curriculum"]["cursor"] != len(saved["curriculum"]["schedule"]):
            raise ValueError("new controller cannot adopt another pending schedule")
        self._validate(self._state, self.backend.snapshot())
        self._committed = self._snapshot()

    def _assert_sources(self):
        if self._uncertain:
            raise RuntimeError("envelope publication is uncertain; reload before continuing")
        if source_hashes() != self._sources or runtime_identity(self.backend) != self._runtime:
            raise ValueError("controller source or execution profile changed")

    def _validate_response(self, role, response, names):
        # Immutable bank/source identities are fixed by this instance. Cache only
        # exact response bytes after canonical validation; changed evidence or a
        # newly loaded envelope must pass the full check. Bound this cache.
        key = (role, tuple(names), hashlib.sha256(json.dumps(response, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest())
        if key in self._validated_responses:
            self._validated_responses.move_to_end(key)
            return
        if (set(response) != {"role", "per_bank", "progress", "wall_seconds", "automatic_promotion"}
                or response["role"] != role or response["automatic_promotion"] is not False
                or set(response["per_bank"]) != set(names) or set(response["progress"]) != set(names)):
            raise ValueError("development response identity or bank set differs")
        _finite(response["wall_seconds"], "evaluation wall")
        rows = {name: self._rows[role][name] for name in names}
        validate_metrics(aggregate_rows(response["per_bank"]), rows,
                         {name: _stats(bank) for name, bank in rows.items()}, self._model_config, "dev")
        for name in names:
            if not _same(response["progress"][name], progress_vector(response["per_bank"][name])):
                raise ValueError("progress vector differs from scored counts")
        self._validated_responses[key] = True
        if len(self._validated_responses) > 128:
            self._validated_responses.popitem(last=False)

    def _validate(self, state, backend):
        fields = {"phase", "evaluation_kind", "evaluation_cursor", "evaluation_results", "allocation",
            "base_updates", "base_family_microbatches", "training_work_seconds", "evaluation_work_seconds",
            "history", "windows_completed", "history_dropped"}
        if not isinstance(state, dict) or set(state) != fields:
            raise ValueError("controller state fields differ")
        policy = allocation.restore(state["allocation"], self.config, self._specs)
        for key in ("latest", "previous"):
            observation = policy[key]
            if observation is not None:
                for role, response in observation["evidence"].items():
                    self._validate_response(role, response, list(self._rows[role]))
        status = allocation.status(policy)
        for key in ("base_updates", "evaluation_cursor", "windows_completed", "history_dropped"):
            _integer(state[key], key)
        for key in ("training_work_seconds", "evaluation_work_seconds"):
            _finite(state[key], key)
        learner, cursor = backend["learner"], backend["curriculum"]
        base = state["base_family_microbatches"]
        if (not isinstance(base, dict) or set(base) != set(self.config.families)
                or any(type(value) is not int or value < 0 for value in base.values())
                or sum(base.values()) != 3*state["base_updates"]
                or learner["updates"] != state["base_updates"]+status["total_updates"]
                or any(learner["family_microbatches"][family] != base[family]+status["family_microbatches"][family]
                       for family in self.config.families)):
            raise ValueError("allocator and learner exposure counters differ")
        if len(learner["realization"]["seen_transcripts"]) > self.options["max_seen_transcripts"]:
            raise ValueError("seen-transcript resource bound exceeded")
        if state["windows_completed"] != status["window_index"]:
            raise ValueError("completed window count differs")
        history = state["history"]
        if (not isinstance(history, list) or len(history) != min(status["window_index"], self.options["history_limit"])
                or state["history_dropped"] != max(0, status["window_index"]-self.options["history_limit"])
                or [row.get("window_index") for row in history] != list(range(state["history_dropped"], status["window_index"]))):
            raise ValueError("bounded window history differs")
        if (state["phase"] not in ("evaluate", "train")
                or state["evaluation_kind"] not in ("baseline", "window")):
            raise ValueError("unknown controller phase")
        if status["has_baseline"]:
            plan = allocation.plan(policy)
            base_update = state["base_updates"]+status["total_updates"]-status["cursor"]
            if status["cursor"] > 0 or cursor["cursor"] < len(cursor["schedule"]):
                if (cursor["base_updates"] != base_update or cursor["cursor"] != status["cursor"]
                        or cursor["schedule"] != plan["full_schedule"]):
                    raise ValueError("allocator pending plan differs from backend cursor")
        elif cursor["cursor"] != len(cursor["schedule"]):
            raise ValueError("baseline cannot contain pending training")
        if state["phase"] == "train":
            if (not status["has_baseline"] or status["window_complete"]
                    or state["evaluation_cursor"] != 0 or state["evaluation_results"]):
                raise ValueError("training phase has incomplete evaluation or exhausted plan")
        else:
            expected_kind = "window" if status["has_baseline"] else "baseline"
            if (state["evaluation_kind"] != expected_kind
                    or status["has_baseline"] and not status["window_complete"]
                    or state["evaluation_cursor"] > len(self._queue)):
                raise ValueError("evaluation phase is not a completed training boundary")
            wanted = self._queue[:state["evaluation_cursor"]]
            if set(state["evaluation_results"]) != {role for role, _ in wanted}:
                raise ValueError("partial evaluation roles differ from cursor")
            for role, value in state["evaluation_results"].items():
                self._validate_response(role, value, [name for r, name in wanted if r == role])
            if not math.isclose(state["evaluation_work_seconds"],
                    sum(value["wall_seconds"] for value in state["evaluation_results"].values()),
                    rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError("partial evaluation timing differs from scored responses")
        _check_finite_tree(state, "controller state")

    def _snapshot(self):
        backend = self.backend.snapshot()
        self._validate(self._state, backend)
        return {"schema": SCHEMA, "source_sha256": copy.deepcopy(self._sources),
            "runtime": copy.deepcopy(self._runtime), "allocation_config": asdict(self.config),
            "options": copy.deepcopy(self.options), "bank_specs": copy.deepcopy(self._specs),
            "backend": backend, "state": copy.deepcopy(self._state)}

    def snapshot(self):
        with self._lock:
            self._assert_sources()
            return self._snapshot()

    @property
    def status(self):
        with self._lock:
            value = allocation.status(self._state["allocation"])
            return {**value, "phase": self._state["phase"], "evaluation_kind": self._state["evaluation_kind"],
                "evaluation_cursor": self._state["evaluation_cursor"],
                "updates": None if self._uncertain else self.backend.updates,
                "in_memory_updates": self.backend.updates, "publication_uncertain": self._uncertain,
                "window_work_seconds": self._state["training_work_seconds"]+self._state["evaluation_work_seconds"],
                "seen_transcripts": len(self.backend._trainer.stream.seen), "automatic_promotion": False}

    def _apply(self, payload):
        if (set(payload) != {"schema", "source_sha256", "runtime", "allocation_config", "options", "bank_specs", "backend", "state"}
                or payload["schema"] != SCHEMA or payload["source_sha256"] != self._sources
                or payload["runtime"] != self._runtime or not _same(payload["allocation_config"], asdict(self.config))
                or payload["options"] != self.options or payload["bank_specs"] != self._specs):
            raise ValueError("controller envelope contract differs")
        _check_finite_tree(payload, "controller envelope")
        self._validate(payload["state"], payload["backend"])
        self.backend.restore(payload["backend"])
        self._state = copy.deepcopy(payload["state"])

    def _assert_disk_current(self):
        if self._disk_path is not None:
            actual = _digest(self._disk_path) if self._disk_path.exists() else None
            if actual != self._disk_digest:
                raise RuntimeError("envelope advanced or changed; reload before writing")

    def _write(self, payload):
        self._assert_disk_current()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", prefix=self._disk_path.name+".", suffix=".tmp",
                                             dir=self._disk_path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(payload, stream)
                stream.flush(); os.fsync(stream.fileno())
            digest = _digest(temporary)
            self._pending_publication = {"payload": payload, "digest": digest, "prior_digest": self._disk_digest}
            os.replace(temporary, self._disk_path)
            self._disk_digest = digest
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _commit(self):
        self._assert_sources()
        payload = self._snapshot()
        self._pending_publication = None
        try:
            if self._disk_path is not None:
                self._write(payload)
            self._committed = payload
        except BaseException:
            pending = self._pending_publication
            if pending is not None:
                try:
                    actual = _digest(self._disk_path) if self._disk_path.exists() else None
                    if actual == pending["digest"]:
                        self._disk_digest, self._committed = actual, pending["payload"]
                    elif actual != pending["prior_digest"]:
                        self._uncertain = True
                except BaseException:
                    self._uncertain = True
            raise
        finally:
            self._pending_publication = None

    def save(self, path):
        with self._lock:
            path = Path(path).resolve()
            if self._disk_path is None:
                if path.exists():
                    raise FileExistsError("existing envelope requires explicit load")
                self._disk_path = path
            elif path != self._disk_path:
                raise ValueError("controller already owns a different envelope")
            path.parent.mkdir(parents=True, exist_ok=True)
            with run_lock(path.parent):
                self._commit()
            return {"path": str(path), "sha256": self._disk_digest, "updates": self.backend.updates}

    @classmethod
    def load(cls, path, train_banks, *, protected_transcripts, evaluation_banks, device="cpu"):
        path = Path(path).resolve()
        digest = _digest(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if _digest(path) != digest:
            raise RuntimeError("envelope changed while loading")
        recipe = payload["backend"]["learner"]["recipe"]
        backend = RealizationBackend(train_banks, mode=recipe["realization"]["mode"],
            protected_transcripts=protected_transcripts, evaluation_banks=evaluation_banks,
            seed=recipe["seed"], sampler_seed=recipe["sampler_seed"], device=device,
            micro_batch_size=recipe["micro_batch_size"], learning_rate=recipe["learning_rate"],
            config=SequenceConfig(**recipe["config"]), payload=payload["backend"])
        result = cls(backend, allocation.AllocationConfig(**payload["allocation_config"]),
                     **payload["options"], payload=payload)
        result._disk_path, result._disk_digest = path, digest
        return result

    def _evaluate_one(self):
        role, name = self._queue[self._state["evaluation_cursor"]]
        response = self.backend.evaluate(role, names=[name])
        self._validate_response(role, response, [name])
        target = self._state["evaluation_results"].setdefault(role,
            {"role": role, "per_bank": {}, "progress": {}, "wall_seconds": 0., "automatic_promotion": False})
        target["per_bank"].update(copy.deepcopy(response["per_bank"]))
        target["progress"].update(copy.deepcopy(response["progress"]))
        target["wall_seconds"] += response["wall_seconds"]
        self._state["evaluation_work_seconds"] += response["wall_seconds"]
        self._state["evaluation_cursor"] += 1
        return response["wall_seconds"]

    def _finish_evaluation(self):
        state = self._state
        status = allocation.status(state["allocation"])
        cost = state["training_work_seconds"]+state["evaluation_work_seconds"]
        if status["has_baseline"]:
            plan = allocation.plan(state["allocation"])
            record = {"window_index": status["window_index"], "updates": status["window_updates"],
                "decision": copy.deepcopy(plan["decision"]), "measured_window_work_seconds": cost,
                "cost_scope": "Backend training-call and per-bank evaluation intervals; outer envelope writes and orchestration are excluded.",
                "progress": {role: copy.deepcopy(value["progress"]) for role, value in state["evaluation_results"].items()}}
            state["history"].append(record)
            state["history"] = state["history"][-self.options["history_limit"]:]
            state["windows_completed"] += 1
            state["history_dropped"] = max(0, state["windows_completed"]-self.options["history_limit"])
        state["allocation"] = allocation.observe(state["allocation"], state["evaluation_results"],
                                                  window_seconds=cost if status["has_baseline"] else 0.)
        state.update(phase="train", evaluation_cursor=0, evaluation_results={},
                     training_work_seconds=0., evaluation_work_seconds=0.)

    def run(self, *, max_updates=None, deadline=None):
        """Report failures even when preflight rejects work before a lock/step."""
        started = time.monotonic()
        with self._lock:
            self.last_report = None
            try:
                result = self._run(max_updates=max_updates, deadline=deadline)
            except BaseException:
                if self.last_report is None:
                    self.last_report = {"status": "failed", "failure_stage": "preflight",
                        "starting_updates": self.backend.updates,
                        "ending_updates": None if self._uncertain else self.backend.updates,
                        "retained_updates": None if self._uncertain else 0,
                        "discarded_completed_updates": None if self._uncertain else 0,
                        "failed_step_attempts": 0, "training_work_seconds": 0.,
                        "evaluation_seconds": 0., "failed_training_work_seconds": 0.,
                        "failed_evaluation_seconds": 0., "checkpoint_seconds": 0.,
                        "publication_uncertain": self._uncertain,
                        "wall_seconds": time.monotonic()-started, "automatic_promotion": False}
                raise
            finally:
                if self.last_report is not None:
                    self.last_report["wall_seconds"] = time.monotonic()-started
                    self.last_report["wall_scope"] = "From public run entry, including lock wait, validation, scoring, training, commits and recovery; constructor preparation and caller overhead excluded."
            return copy.deepcopy(self.last_report)

    def _run(self, *, max_updates=None, deadline=None):
        """Bound an invocation; stop between a chunk, a bank score, or a commit.

        One in-flight operation and its atomic envelope write can overrun the
        wall deadline. A zero update budget performs no scoring or training.
        A finite seen-transcript bound is a resource stop, not a learning result.
        """
        started = time.monotonic()
        if max_updates is None and deadline is None:
            raise ValueError("an update or wall-clock budget is required")
        if max_updates is not None:
            _integer(max_updates, "max_updates")
        if deadline is not None:
            _finite(deadline, "deadline")
        with self._lock:
            self._assert_sources()
            guard = run_lock(self._disk_path.parent) if self._disk_path is not None else nullcontext()
            with guard:
                self._assert_disk_current()
                report = {"status": "running", "starting_updates": self.backend.updates,
                    "retained_updates": 0, "evaluation_banks": 0, "completed_window_evaluations": 0,
                    "training_work_seconds": 0., "evaluation_seconds": 0., "checkpoint_seconds": 0.,
                    "failed_training_work_seconds": 0., "failed_evaluation_seconds": 0.,
                    "discarded_completed_updates": 0, "failed_step_attempts": 0,
                    "automatic_promotion": False}
                self.last_report = report
                try:
                    while True:
                        self._assert_sources()
                        self._assert_disk_current()
                        if deadline is not None and time.monotonic() >= deadline:
                            report["status"] = "deadline"; break
                        consumed = self.backend.updates-report["starting_updates"]
                        if max_updates is not None and consumed >= max_updates:
                            report["status"] = "update_bound"; break
                        operation_start = self.backend.updates
                        self.backend.last_report = None
                        kind, training_recorded = "policy", False
                        operation_tick = time.monotonic()
                        try:
                            if self._state["phase"] == "evaluate":
                                if self._state["evaluation_cursor"] < len(self._queue):
                                    kind = "evaluation"
                                    report["evaluation_seconds"] += self._evaluate_one()
                                    report["evaluation_banks"] += 1
                                else:
                                    baseline = not allocation.status(self._state["allocation"])["has_baseline"]
                                    self._finish_evaluation()
                                    report["completed_window_evaluations"] += int(not baseline)
                            else:
                                status = allocation.status(self._state["allocation"])
                                left = status["window_updates"]-status["cursor"]
                                room = (self.options["max_seen_transcripts"]-len(self.backend._trainer.stream.seen)) // (3*self.backend._trainer.micro_batch_size)
                                bound = min(MAX_CHUNK_UPDATES, left, room,
                                            max_updates-consumed if max_updates is not None else MAX_CHUNK_UPDATES)
                                if bound <= 0:
                                    report["status"] = "transcript_bound"; break
                                plan = allocation.plan(self._state["allocation"])
                                cursor = self.backend.curriculum
                                pending = cursor["cursor"] < len(cursor["schedule"])
                                kind = "training"
                                chunk = self.backend.train_chunk(None if pending else plan["full_schedule"],
                                    max_updates=bound, deadline=deadline, commit_interval=bound)
                                report["training_work_seconds"] += chunk["wall_seconds"]
                                training_recorded = True
                                advanced = self.backend.updates-operation_start
                                self._state["allocation"] = allocation.advance(self._state["allocation"], advanced)
                                self._state["training_work_seconds"] += chunk["wall_seconds"]
                                if allocation.status(self._state["allocation"])["window_complete"]:
                                    self._state.update(phase="evaluate", evaluation_kind="window", evaluation_cursor=0, evaluation_results={})
                            tick = time.monotonic()
                            kind = "checkpoint"
                            try:
                                self._commit()
                            finally:
                                report["checkpoint_seconds"] += time.monotonic()-tick
                        except BaseException:
                            backend_report = self.backend.last_report or {}
                            report["failure_stage"] = kind
                            if kind == "training" and not training_recorded:
                                failed_seconds = backend_report.get("wall_seconds", time.monotonic()-operation_tick)
                                report["training_work_seconds"] += failed_seconds
                                report["failed_training_work_seconds"] += failed_seconds
                            elif kind == "evaluation":
                                # Failed scoring/validation may not return an
                                # evaluator timer. Record the enclosing attempt.
                                failed_seconds = time.monotonic()-operation_tick
                                report["evaluation_seconds"] += failed_seconds
                                report["failed_evaluation_seconds"] += failed_seconds
                            successful = max(self.backend.updates-operation_start, backend_report.get("completed_updates", 0))
                            report["failed_step_attempts"] += backend_report.get("failed_step_attempts", 0)
                            if self._uncertain:
                                report["publication_uncertain"] = True
                                report["discarded_completed_updates"] = None
                            else:
                                retained = self._committed["backend"]["learner"]["updates"]-operation_start
                                report["discarded_completed_updates"] += max(0, successful-retained)
                                try:
                                    self._apply(self._committed)
                                except BaseException:
                                    self._uncertain = True
                                    report["recovery_failed"] = True
                                    report["publication_uncertain"] = True
                                    report["discarded_completed_updates"] = None
                                    raise
                            raise
                except BaseException:
                    report["status"] = "failed"
                    raise
                finally:
                    report.update(ending_updates=None if self._uncertain else self.backend.updates,
                        retained_updates=None if self._uncertain else self.backend.updates-report["starting_updates"],
                        wall_seconds=time.monotonic()-started,
                        deadline_overrun_seconds=0. if deadline is None else max(0., time.monotonic()-deadline),
                        wall_scope="From run entry through validation, scoring, training, commits and recovery; constructor preparation and caller overhead excluded.")
                    report["component_scope"] = "Training/evaluation include returned failed-attempt timers or enclosing failed-call wall intervals. Failed-time fields are subsets; do not add them again. Full wall also includes validation, commits and recovery. Retained policy cost excludes failed/rolled-back operations."
                    self.last_report = copy.deepcopy(report)
                return copy.deepcopy(report)

    def run_for_hours(self, hours):
        _finite(hours, "hours")
        return self.run(deadline=time.monotonic()+hours*3600.)
