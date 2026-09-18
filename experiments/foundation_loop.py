"""Finite prescribed practice with resumable evaluation and one owned envelope.

No adaptive allocation, tutor, audit, promotion or automatic extension. Physical
work reports are in memory: a hard process kill can leave unreported work after
the last envelope. A retained checkpoint alone is not a complete compute ledger.
Window/chunk defaults are engineering choices, not measured efficiency settings.
Every bank commit still serializes one full learner/history envelope; the caller
must use reported scoring/checkpoint costs to choose a useful observation cadence.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import weakref

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_loop import run_lock
from experiments.foundation_evidence import json_digest
from experiments.foundation_provider import FoundationPracticeProvider, source_hashes as provider_sources
from experiments.train_cognitive import _check_finite_tree

SCHEMA = "bic-prescribed-foundation-loop-v1"
_OWNERS = weakref.WeakKeyDictionary()
_OWNER_LOCK = threading.Lock()
FIELDS = ("paired_action", "paired_reply", "known", "unsupported_ask")
WORK_FIELDS = ("physical_optimizer_updates", "drawn_episode_exposures",
               "neural_attempted_episode_exposures", "completed_microbatch_episode_exposures")


def source_hashes():
    return {**provider_sources(), "experiments/foundation_loop.py": _digest(__file__)}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _integer(value, name, minimum=0, maximum=4096):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(name+" must be an integer within bounds")


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(name+" must be finite and nonnegative")


def _same(a, b):
    return json_digest(a) == json_digest(b)


def _producer(provider_payload):
    learner = provider_payload["learner"]
    return {"updates": learner["cursor"], "weights_sha256": checkpoint_digest(
        SimpleNamespace(state_dict=lambda: learner["weights"]))}


def _same_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if type(a) is not type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(_same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(_same_tree(x, y) for x, y in zip(a, b))
    return a == b


class FoundationLoop:
    def __init__(self, provider, *, window_updates=16, chunk_updates=4, history_limit=32, payload=None):
        if not isinstance(provider, FoundationPracticeProvider):
            raise ValueError("FoundationPracticeProvider required")
        for name, value in (("window_updates", window_updates), ("chunk_updates", chunk_updates),
                            ("history_limit", history_limit)):
            _integer(value, name, 1)
        if chunk_updates > window_updates:
            raise ValueError("practice chunk exceeds prescribed window")
        self.provider = provider
        self.options = dict(window_updates=window_updates, chunk_updates=chunk_updates, history_limit=history_limit)
        self._lock = threading.RLock()
        self._sources = source_hashes()
        self._identity = provider.identity
        self._identity_sha256 = json_digest(self._identity)
        self._specs = provider.specs()
        self._queue = [(role, name) for role in sorted(self._specs) for name in sorted(self._specs[role])]
        self._cache_limit = min(65536, max(128, (history_limit+10)*len(self._queue)))
        self._disk_path = self._disk_digest = self._pending_publication = None
        self._uncertain = False
        self._validated_responses = OrderedDict()
        self.last_report = None
        saved = provider.snapshot()
        if saved["learner"]["cursor"] != 0 or saved["pending"] is not None:
            raise ValueError("new loop requires a zero-update provider; use explicit loop-envelope load")
        self._total = provider.status()["total_updates"]
        self._state = {"phase": "evaluate", "window_start": 0, "window_stop": 0,
            "evaluation": {"producer": _producer(saved), "cursor": 0, "responses": {}},
            "evaluations_completed": 0, "history": [], "history_dropped": 0, "references": {},
            "reference_anchor": {}, "work": {**dict.fromkeys(WORK_FIELDS, 0), "evaluation_banks": 0,
                "evaluation_episode_exposures": 0, "training_seconds": 0., "evaluation_seconds": 0.}}
        with _OWNER_LOCK:
            owner = _OWNERS.get(provider)
            if owner is not None and owner() is not None:
                raise ValueError("provider already has a loop owner")
            _OWNERS[provider] = weakref.ref(self)
        self._committed = self._snapshot()
        if payload is not None:
            self._apply(payload)
            self._committed = self._snapshot()

    def _check(self):
        if self._uncertain:
            raise RuntimeError("envelope publication/state is uncertain; explicit reload required")
        if (_digest(__file__) != self._sources["experiments/foundation_loop.py"]
                or not _same(self.provider.identity, self._identity)):
            raise RuntimeError("loop source/provider identity changed")
        # The provider guards its complete dependency closure and runtime once;
        # avoid hashing that same closure again in the outer controller.
        self.provider._check()

    def _entry(self):
        self._check(); self._disk_current()
        try:
            current = self.provider.snapshot()
            if not _same_tree(current, self._committed["provider"]):
                raise RuntimeError("provider changed outside its loop owner")
        except BaseException:
            self._uncertain = True
            raise
        return current

    def _response(self, response, role, name, producer):
        if response.get("role") != role or set(response.get("per_bank", {})) != {name} or response.get("control") != "normal":
            raise ValueError("partial evaluation role, bank or control differs")
        key = json_digest({"identity": self._identity_sha256, "producer": producer,
                           "role": role, "name": name, "response": response})
        if key in self._validated_responses:
            self._validated_responses.move_to_end(key)
            return
        self.provider.validate_evaluation(response, expected_weights_sha256=producer["weights_sha256"],
                                           expected_updates=producer["updates"])
        self._validated_responses[key] = True
        if len(self._validated_responses) > self._cache_limit:
            self._validated_responses.popitem(last=False)

    def _validate(self, state, provider):
        fields = {"phase", "window_start", "window_stop", "evaluation", "evaluations_completed",
                  "history", "history_dropped", "references", "reference_anchor", "work"}
        if type(state) is not dict or set(state) != fields or state["phase"] not in ("evaluate", "practice", "complete"):
            raise ValueError("loop state fields or phase differ")
        cursor = provider["learner"]["cursor"]
        for key in ("window_start", "window_stop", "evaluations_completed", "history_dropped"):
            _integer(state[key], key, maximum=self._total+1)
        start, stop = state["window_start"], state["window_stop"]
        if not 0 <= start <= cursor <= stop <= self._total or stop-start > self.options["window_updates"]:
            raise ValueError("prescribed window disagrees with learner cursor")
        evaluations = state["evaluations_completed"]
        history = state["history"]
        if (type(history) is not list or len(history) != min(evaluations, self.options["history_limit"])
                or state["history_dropped"] != max(0, evaluations-self.options["history_limit"])
                or [record.get("index") for record in history] != list(range(state["history_dropped"], evaluations))):
            raise ValueError("bounded history indices differ")
        for record in history:
            if set(record) != {"index", "producer", "responses", "alarms"}:
                raise ValueError("history record fields differ")
            expected_update = min(record["index"]*self.options["window_updates"], self._total)
            if record["producer"]["updates"] != expected_update:
                raise ValueError("evaluation history is not at a prescribed boundary")
            if set(record["responses"]) != {f"{role}:{name}" for role, name in self._queue}:
                raise ValueError("history requires every evaluation bank")
            for role, name in self._queue:
                self._response(record["responses"][f"{role}:{name}"], role, name, record["producer"])
        references = state["reference_anchor"]
        self._validate_references(references, state["history_dropped"], cursor)
        references = copy.deepcopy(references)
        for record in history:
            alarms = self._retention(references, record["responses"], record["producer"])
            if not _same(alarms, record["alarms"]):
                raise ValueError("retention alarms disagree with authenticated references")
        if not _same(references, state["references"]):
            raise ValueError("persistent references disagree with retained history")
        self._validate_references(state["references"], evaluations, cursor)
        # Every surviving record for a producing boundary must name the same
        # weights. Older reference predictions are authenticated evidence, not
        # independently recomputed historical model outputs.
        producers = {}
        records = [record["producer"] for record in history]
        records += [ref["producer"] for cell in references.values() for ref in cell.values()]
        records += [ref["producer"] for cell in state["reference_anchor"].values() for ref in cell.values()]
        for producer in records:
            prior = producers.setdefault(producer["updates"], producer["weights_sha256"])
            if prior != producer["weights_sha256"]: raise ValueError("historical producing weight digests differ")
        work = state["work"]
        if type(work) is not dict or set(work) != set(WORK_FIELDS) | {"evaluation_banks", "evaluation_episode_exposures", "training_seconds", "evaluation_seconds"}:
            raise ValueError("committed work fields differ")
        episodes = sum(item["episodes"] for item in provider["learner"]["evidence"]["exposures"].values())
        for key in WORK_FIELDS:
            _integer(work[key], key, maximum=2**63-1)
            if work[key] != (cursor if key == "physical_optimizer_updates" else episodes):
                raise ValueError("committed practice accounting differs from learner exposure")
        partial = state["evaluation"]["cursor"] if state["evaluation"] is not None else 0
        expected_banks = evaluations*len(self._queue)+partial
        expected_episodes = evaluations*sum(self._specs[r][n]["identity"]["episodes"] for r, n in self._queue)
        expected_episodes += sum(self._specs[r][n]["identity"]["episodes"] for r, n in self._queue[:partial])
        for key, expected in (("evaluation_banks", expected_banks), ("evaluation_episode_exposures", expected_episodes)):
            _integer(work[key], key, maximum=2**63-1)
            if work[key] != expected: raise ValueError("committed scoring accounting differs")
        for key in ("training_seconds", "evaluation_seconds"): _finite(work[key], key)
        self._validate_phase(state, provider)
        _check_finite_tree(state, "foundation loop state")

    def _validate_references(self, references, evaluations, cursor):
        expected_keys = {f"{role}:{name}" for role, name in self._queue} if evaluations else set()
        if type(references) is not dict or set(references) != expected_keys:
            raise ValueError("persistent reference bank set differs")
        for role, name in self._queue:
            key = f"{role}:{name}"
            if key not in references:
                continue
            if set(references[key]) != set(FIELDS):
                raise ValueError("persistent reference metric set differs")
            for field, ref in references[key].items():
                if set(ref) != {"producer", "response"}:
                    raise ValueError("reference producer differs")
                producer = ref["producer"]
                boundaries = {min(i*self.options["window_updates"], self._total) for i in range(evaluations)}
                if producer["updates"] not in boundaries or producer["updates"] > cursor:
                    raise ValueError("reference is not a completed evaluation boundary")
                self._response(ref["response"], role, name, ref["producer"])

    def _validate_phase(self, state, provider):
        cursor = provider["learner"]["cursor"]
        start, stop, evaluations = state["window_start"], state["window_stop"], state["evaluations_completed"]
        evaluation = state["evaluation"]
        if state["phase"] == "practice":
            if evaluation is not None or not evaluations or not start <= cursor < stop:
                raise ValueError("practice requires completed baseline and an unconsumed prefix")
            pending = provider["pending"]
            if pending is not None:
                request = pending["request"]
                if request["start_cursor"] != start or request["stop_cursor"] != stop:
                    raise ValueError("provider pending prefix differs from loop window")
            elif cursor != start:
                raise ValueError("partly consumed window lost its provider request")
            if start != min((evaluations-1)*self.options["window_updates"], self._total):
                raise ValueError("practice window index differs")
            if stop != min(self._total, start+self.options["window_updates"]):
                raise ValueError("practice window length differs")
        elif state["phase"] == "complete":
            if cursor != self._total or evaluation is not None or provider["pending"] is not None:
                raise ValueError("complete requires exhausted provider and final evaluation")
            if evaluations != math.ceil(self._total/self.options["window_updates"])+1:
                raise ValueError("final evaluation is missing")
            if start != self._total or stop != self._total: raise ValueError("complete window differs")
        else:
            if cursor != stop or provider["pending"] is not None:
                raise ValueError("evaluation must occur at a completed prefix boundary")
            if type(evaluation) is not dict or set(evaluation) != {"producer", "cursor", "responses"}:
                raise ValueError("partial evaluation fields differ")
            if not _same(evaluation["producer"], _producer(provider)):
                raise ValueError("partial evaluation belongs to another producing state")
            _integer(evaluation["cursor"], "evaluation cursor", maximum=len(self._queue))
            expected = self._queue[:evaluation["cursor"]]
            if set(evaluation["responses"]) != {f"{r}:{n}" for r, n in expected}:
                raise ValueError("partial evaluation queue differs")
            if cursor != min(evaluations*self.options["window_updates"], self._total):
                raise ValueError("evaluation boundary index differs")
            if start != max(0, min((evaluations-1)*self.options["window_updates"], self._total)):
                raise ValueError("evaluation window start differs")
            for role, name in expected:
                self._response(evaluation["responses"][f"{role}:{name}"], role, name, evaluation["producer"])

    def _snapshot(self, *, trusted=False, saved=None):
        saved = self.provider.snapshot() if saved is None else saved
        if not trusted: self._validate(self._state, saved)
        return {"schema": SCHEMA, "source_sha256": copy.deepcopy(self._sources),
            "provider_identity": copy.deepcopy(self._identity), "options": copy.deepcopy(self.options),
            "provider": saved, "state": self._owned_state() if trusted else copy.deepcopy(self._state)}

    def _owned_state(self):
        # History records and admitted responses are immutable within this
        # owner. Share those across private commit snapshots; copy every mutable
        # mapping/list. Public snapshots remain completely detached deep copies.
        state = self._state
        evaluation = state["evaluation"]
        return {**state, "history": list(state["history"]), "work": dict(state["work"]),
            "references": {key: dict(cell) for key, cell in state["references"].items()},
            "reference_anchor": {key: dict(cell) for key, cell in state["reference_anchor"].items()},
            "evaluation": None if evaluation is None else {**evaluation, "responses": dict(evaluation["responses"])}}

    def snapshot(self):
        with self._lock, self.provider._lock:
            return self._snapshot(saved=self._entry())

    def _apply(self, payload):
        if (type(payload) is not dict or set(payload) != {"schema", "source_sha256", "provider_identity", "options", "provider", "state"}
                or payload["schema"] != SCHEMA or payload["source_sha256"] != self._sources
                or not _same(payload["provider_identity"], self._identity) or payload["options"] != self.options):
            raise ValueError("loop envelope identity/options differ")
        _check_finite_tree(payload, "foundation loop envelope")
        # Restore the provider first so its public historical validator can also
        # validate after a poisoned practice attempt. Roll back on bad loop state.
        prior = self._committed
        try:
            self.provider.restore(payload["provider"])
            self._validate(payload["state"], payload["provider"])
        except BaseException:
            try: self.provider.restore(prior["provider"])
            except BaseException: self._uncertain = True
            raise
        self._state = copy.deepcopy(payload["state"])

    def _disk_current(self):
        if self._disk_path is not None:
            actual = _digest(self._disk_path) if self._disk_path.exists() else None
            if actual != self._disk_digest:
                self._uncertain = True
                raise RuntimeError("envelope changed externally; explicit reload required")

    def _write(self, payload):
        self._disk_current()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", prefix=self._disk_path.name+".", suffix=".tmp",
                                             dir=self._disk_path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                torch.save(payload, handle); handle.flush(); os.fsync(handle.fileno())
            digest = _digest(temporary)
            self._pending_publication = {"payload": payload, "digest": digest, "prior": self._disk_digest}
            os.replace(temporary, self._disk_path)
            self._disk_digest = digest
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)

    def _commit(self):
        self._check()
        # Owned transitions already validate each newly produced response and
        # provider request. Full historical validation belongs at trust boundaries,
        # not at every one-bank commit (O(history*banks) work per bank).
        payload = self._snapshot(trusted=True)
        self._pending_publication = None
        try:
            if self._disk_path is not None: self._write(payload)
            self._committed = payload
        except BaseException:
            pending = self._pending_publication
            if pending is not None:
                try:
                    actual = _digest(self._disk_path) if self._disk_path.exists() else None
                    if actual == pending["digest"]:
                        self._disk_digest, self._committed = actual, pending["payload"]
                    elif actual != pending["prior"]:
                        self._uncertain = True
                except BaseException: self._uncertain = True
            raise
        finally: self._pending_publication = None

    def save(self, path):
        with self._lock, self.provider._lock:
            self._entry()
            path = Path(path).resolve()
            if self._disk_path is None:
                if path.exists(): raise FileExistsError("existing loop envelope requires explicit load")
                self._disk_path = path
            elif path != self._disk_path:
                raise ValueError("loop already owns another envelope path")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with run_lock(path.parent): self._commit()
            except BaseException:
                if not self._uncertain: self._apply(self._committed)
                raise
            return {"path": str(path), "sha256": self._disk_digest,
                    "updates": self._committed["provider"]["learner"]["cursor"]}

    @classmethod
    def load(cls, path, provider):
        path = Path(path).resolve()
        before = _digest(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if _digest(path) != before: raise RuntimeError("loop envelope changed during load")
        result = cls(provider, **payload["options"], payload=payload)
        result._disk_path, result._disk_digest = path, before
        return result

    @property
    def status(self):
        with self._lock, self.provider._lock:
            if self._uncertain:
                return {"phase": "reload_required", "updates": None, "publication_uncertain": True}
            self._entry()
            return {"phase": self._state["phase"], "updates": self.provider.status()["updates"],
                "evaluation_cursor": None if self._state["evaluation"] is None else self._state["evaluation"]["cursor"],
                "evaluations_completed": self._state["evaluations_completed"],
                "window_start": self._state["window_start"], "window_stop": self._state["window_stop"],
                "publication_uncertain": False, "automatic_promotion": False}

    def _retention(self, all_references, responses, producer):
        alarms = []
        for role, name in self._queue:
            key = f"{role}:{name}"
            response = responses[key]
            references = all_references.setdefault(key, {})
            shared = {"producer": producer, "response": response}
            for field in FIELDS:
                current = response["progress"][name][field]["rate"]
                prior = references.get(field)
                best = None if prior is None else prior["response"]["progress"][name][field]["rate"]
                improved = prior is None or best is None or (current is not None and
                    (current < best if field == "unsupported_ask" else current > best))
                regressed = current is not None and best is not None and (
                    current > best if field == "unsupported_ask" else current < best)
                if regressed:
                    alarms.append({"role": role, "bank": name, "metric": field,
                                   "observed": current, "reference": best,
                                   "reference_producer": prior["producer"]})
                if improved:
                    references[field] = shared
        return alarms

    def _finish_evaluation(self):
        state, evaluation = self._state, self._state["evaluation"]
        alarms = self._retention(state["references"], evaluation["responses"], evaluation["producer"])
        state["history"].append({"index": state["evaluations_completed"],
            "producer": evaluation["producer"], "responses": evaluation["responses"], "alarms": alarms})
        state["evaluations_completed"] += 1
        for dropped in state["history"][:-self.options["history_limit"]]:
            self._retention(state["reference_anchor"], dropped["responses"], dropped["producer"])
        state["history"] = state["history"][-self.options["history_limit"]:]
        state["history_dropped"] = max(0, state["evaluations_completed"]-self.options["history_limit"])
        cursor = evaluation["producer"]["updates"]
        state.update(evaluation=None, window_start=cursor,
            window_stop=min(self._total, cursor+self.options["window_updates"]),
            phase="complete" if cursor == self._total else "practice")

    def run(self, *, max_updates=None, max_evaluations=None, deadline=None):
        """Run an explicitly bounded invocation; deadline is monotonic absolute.

        Time limits are checked between banks/chunks (and by the provider between
        updates), not by interrupting a neural operation. A zero update allowance
        still permits baseline/final evaluation. An evaluation-only allowance
        implies zero practice; a deadline alone may consume the finite plan.
        """
        started = time.monotonic()
        with self._lock, self.provider._lock:
            initial = self._committed["provider"]["learner"]["cursor"]
            initial_work = self._committed["state"]["work"]
            report = {"status": "preflight", "starting_updates": initial, "completed_updates": 0,
                "evaluation_banks": 0, "evaluation_episode_exposures": 0, "failed_step_attempts": 0,
                "evaluation_attempts": 0, "incomplete_evaluation_attempts": 0,
                "unknown_evaluation_episode_exposures": False, "unknown_practice_attempts": 0,
                "practice_reports": [], "training_seconds": 0., "evaluation_seconds": 0.,
                "checkpoint_seconds": 0., "rollback_seconds": 0., "discarded_completed_updates": 0,
                "automatic_promotion": False,
                "scope": "Invocation includes validation, scoring, training, commits and rollback; setup is excluded. Component intervals are included subsets. Evaluation exposures count returned banks; failed scoring may have unknown additional work. Committed work covers retained lineage only. A hard process kill can leave unreported physical work; no durable started-work marker exists."}
            self.last_report = report
            kind, tick = "preflight", started
            mutated = False
            in_flight = None
            try:
                if max_updates is None and max_evaluations is None and deadline is None:
                    raise ValueError("an explicit update, evaluation or deadline bound is required")
                if max_updates is not None: _integer(max_updates, "update allowance")
                if max_evaluations is not None: _integer(max_evaluations, "evaluation allowance", maximum=1000000)
                if deadline is not None: _finite(deadline, "monotonic deadline")
                update_limit = 0 if max_updates is None and deadline is None else max_updates
                self._snapshot(saved=self._entry())
                guard = run_lock(self._disk_path.parent) if self._disk_path is not None else nullcontext()
                with guard:
                    self._disk_current()
                    while True:
                        self._check(); self._disk_current()
                        state = self._state
                        if state["phase"] == "complete": report["status"] = "complete"; break
                        if deadline is not None and time.monotonic() >= deadline: report["status"] = "deadline"; break
                        kind, tick = "transition", time.monotonic()
                        if state["phase"] == "evaluate":
                            evaluation = state["evaluation"]
                            if evaluation["cursor"] == len(self._queue):
                                mutated = True
                                self._finish_evaluation()
                            else:
                                if max_evaluations is not None and report["evaluation_banks"] >= max_evaluations:
                                    report["status"] = "evaluation_bound"; break
                                kind = "evaluation"
                                role, name = self._queue[evaluation["cursor"]]
                                report["evaluation_attempts"] += 1
                                mutated = True
                                try:
                                    response = self.provider.evaluate(role, names=[name])
                                except BaseException:
                                    report["incomplete_evaluation_attempts"] += 1
                                    report["unknown_evaluation_episode_exposures"] = True
                                    raise
                                finally:
                                    elapsed = time.monotonic()-tick
                                    report["evaluation_seconds"] += elapsed
                                report["evaluation_banks"] += 1
                                episodes = self._specs[role][name]["identity"]["episodes"]
                                report["evaluation_episode_exposures"] += episodes
                                self._response(response, role, name, evaluation["producer"])
                                evaluation["responses"][f"{role}:{name}"] = response
                                evaluation["cursor"] += 1
                                state["work"]["evaluation_banks"] += 1
                                state["work"]["evaluation_episode_exposures"] += episodes
                                state["work"]["evaluation_seconds"] += elapsed
                        else:
                            completed = sum(chunk["completed_updates"] for chunk in report["practice_reports"])
                            if update_limit is not None and completed >= update_limit:
                                report["status"] = "update_bound"; break
                            kind = "practice"
                            before = self.provider.status()
                            request = None if before["pending"] is not None else {
                                "kind": "prescribed_prefix", "provider_sha256": self.provider.identity_sha256,
                                "plan_sha256": self._identity["plan_sha256"], "order": self._identity["order"],
                                "start_cursor": state["window_start"], "stop_cursor": state["window_stop"]}
                            limit = min(self.options["chunk_updates"], state["window_stop"]-before["updates"],
                                update_limit-completed if update_limit is not None else self.options["chunk_updates"])
                            self.provider.last_report = None
                            mutated = True
                            in_flight = len(report["practice_reports"])
                            try:
                                chunk = self.provider.practice(request, max_updates=limit, deadline=deadline)
                            finally:
                                elapsed = time.monotonic()-tick
                                report["training_seconds"] += elapsed
                            report["practice_reports"].append(copy.deepcopy(chunk))
                            in_flight = None
                            self._account_practice(state, chunk, elapsed)
                            if self.provider.status()["updates"] == state["window_stop"]:
                                state.update(phase="evaluate", evaluation={"producer": _producer(self.provider.snapshot()),
                                    "cursor": 0, "responses": {}})
                        kind, tick = "checkpoint", time.monotonic()
                        try: self._commit()
                        finally: report["checkpoint_seconds"] += time.monotonic()-tick
            except BaseException as error:
                report.update(status="failed", failure_stage=kind, error=repr(error))
                # Capture authoritative work before restore clears last_report.
                # If publication/accounting throws after practice returned, an
                # already appended chunk must not be counted a second time.
                if in_flight is not None and len(report["practice_reports"]) == in_flight:
                    if self.provider.last_report is None: report["unknown_practice_attempts"] += 1
                    else: report["practice_reports"].append(copy.deepcopy(self.provider.last_report))
                if mutated and not self._uncertain:
                    recovery = time.monotonic()
                    try: self._apply(self._committed)
                    except BaseException:
                        self._uncertain = True
                        report["recovery_failed"] = True
                    finally: report["rollback_seconds"] += time.monotonic()-recovery
                raise
            finally:
                chunks = report["practice_reports"]
                report["completed_updates"] = sum(chunk["completed_updates"] for chunk in chunks)
                report["failed_step_attempts"] = sum(chunk["failed_step_attempts"] for chunk in chunks)
                report["unknown_optimizer_attempts"] = sum(chunk["unknown_optimizer_attempts"] for chunk in chunks)
                for field in WORK_FIELDS:
                    values = [chunk[field] for chunk in chunks]
                    report[field] = None if report["unknown_practice_attempts"] or any(v is None for v in values) else sum(values)
                retained = None if self._uncertain else self._committed["provider"]["learner"]["cursor"]-initial
                report.update(retained_updates=retained,
                    durable_retained_updates=retained if self._disk_path is not None else None,
                    ending_updates=None if self._uncertain else self._committed["provider"]["learner"]["cursor"],
                    publication_uncertain=self._uncertain, wall_seconds=time.monotonic()-started)
                report["discarded_completed_updates"] = None if retained is None else max(0, report["completed_updates"]-retained)
                physical = report["physical_optimizer_updates"]
                report["discarded_optimizer_updates"] = None if retained is None or physical is None else max(0, physical-retained)
                for field in ("evaluation_banks", "evaluation_episode_exposures"):
                    kept = None if self._uncertain else self._committed["state"]["work"][field]-initial_work[field]
                    report["retained_"+field] = kept
                    report["durable_retained_"+field] = kept if self._disk_path is not None else None
                    report["discarded_"+field] = None if kept is None else report[field]-kept
                self.last_report = copy.deepcopy(report)
            return copy.deepcopy(report)

    @staticmethod
    def _account_practice(state, chunk, elapsed):
        if chunk["physical_optimizer_updates"] != chunk["completed_updates"] or chunk["accounting_uncertain"]:
            raise RuntimeError("successful practice returned uncertain accounting")
        addition = {field: state["work"][field]+chunk[field] for field in WORK_FIELDS}
        state["work"] = {**state["work"], **addition,
                         "training_seconds": state["work"]["training_seconds"]+elapsed}
