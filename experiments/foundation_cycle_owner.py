"""One durable owner for bounded, prescribed, fresh foundation cycles.

The inner loop is never independently disk-bound. Immutable parent capsules are
published before the one atomic owner envelope that references them. Every run
has an exclusive intent and receipt; an unfinished intent blocks automatic replay.
This is curriculum execution, not a tutor, self-directed policy or promotion.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import uuid

import torch

from brain_in_computer.learning_loop import run_lock
from experiments import foundation_loop as loops
from experiments import foundation_plan as planning
from experiments import foundation_provider as providers
from experiments.foundation_admission import repair_plan
from experiments.foundation_cycle_provider import CycleFoundationPracticeProvider
from experiments.foundation_cycle_training import CompletedParent
from experiments.foundation_evidence import json_digest, prepare_training, transcript_set
from experiments.foundation_lifetime_retention import LifetimeRetention, source_hashes as retention_sources
from experiments.foundation_parent_capsule import export_parent_capsule, load_parent_capsule
from experiments.foundation_plan_index import AuthenticatedPlanIndex
from experiments.train_cognitive import _check_finite_tree


SCHEMA = "bic-foundation-cycle-owner-v1"
INTENT_SCHEMA = "bic-foundation-cycle-owner-invocation-v1"
MAX_BYTES = 2 * 1024**3
WORK_FIELDS = (*loops.WORK_FIELDS, "evaluation_banks", "evaluation_episode_exposures",
               "training_seconds", "evaluation_seconds")
REPORT_COUNTS = ("completed_updates", "evaluation_banks", "evaluation_episode_exposures",
    "failed_step_attempts", "evaluation_attempts", "incomplete_evaluation_attempts",
    "unknown_practice_attempts", "unknown_optimizer_attempts")
SCOPE = ("Prescribed fresh balanced cycles; no adaptive allocation, external tutor or promotion. "
         "Retained work is measured since the supplied origin. Invocation receipts include discarded "
         "physical work; a missing receipt means unknown work and blocks automatic replay. "
         "Deadlines are checked between operations; admission, scoring and publication are nonpreemptive.")


def source_hashes():
    return {**retention_sources(), "experiments/foundation_cycle_owner.py":
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**63-1:
        raise ValueError(name + " must be a bounded integer")


def _keys(value, fields, name):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(name + " fields differ")


def _zero_work():
    return {key: 0. if key.endswith("seconds") else 0 for key in WORK_FIELDS}


def _add_work(first, second):
    return {key: first[key] + second[key] for key in WORK_FIELDS}


def _seed(root, cycle, purpose):
    return int(json_digest([SCHEMA, root, cycle, purpose]), 16) % 2**63


def _policy(root_seed, plan_config, loop_options):
    _integer(root_seed, "root_seed")
    defaults = dict(stage_updates=24, final_updates=48, micro_batch_size=32, rehearsal_every=4)
    if type(plan_config) is not dict or not set(plan_config) <= set(defaults):
        raise ValueError("plan_config accepts only stage/final updates, microbatch size and rehearsal cadence")
    defaults.update(plan_config)
    planning.build_plan(seed=0, ordering_seed=0, **defaults)
    options = dict(window_updates=16, chunk_updates=4, history_limit=32)
    if loop_options is not None:
        if type(loop_options) is not dict or not set(loop_options) <= set(options):
            raise ValueError("canonical loop options required")
        options.update(loop_options)
    for key, value in options.items():
        loops._integer(value, key, 1)
    if options["chunk_updates"] > options["window_updates"]:
        raise ValueError("chunk cannot exceed window")
    return dict(root_seed=root_seed, plan_config=defaults, loop_options=options,
                order="curriculum", seed_rule="sha256(owner-schema, root-seed, cycle-ordinal, purpose) mod 2**63",
                automatic_promotion=False)


def _base_plan(policy, cycle):
    return planning.build_plan(seed=_seed(policy["root_seed"], cycle, "lessons"),
        ordering_seed=_seed(policy["root_seed"], cycle, "order"), **policy["plan_config"])


def _exclusive_json(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    path, temporary = Path(path), None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=path.name+".", suffix=".tmp",
                dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        # Linking a complete file is atomic and fails if the final name exists.
        # A crash before publication leaves only an unreferenced temporary file.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class UnknownInvocationWork(RuntimeError):
    """A started invocation has no durable receipt; no automatic retry is safe."""


class CycleOwner:
    """Own one active finite loop and automatically continue within caller bounds."""

    @classmethod
    def create(cls, directory, *, parent, retention, evaluation_banks, plan_config,
               root_seed, loop_options=None, device="cpu"):
        if cls is not CycleOwner or type(parent) is not CompletedParent or type(retention) is not LifetimeRetention:
            raise ValueError("exact owner, completed parent and lifetime retention objects required")
        policy = _policy(root_seed, plan_config, loop_options)
        sources = source_hashes()
        runtime = providers.runtime_identity(torch.device(device))
        if runtime != parent.provider_identity["runtime"]:
            raise ValueError("owner runtime must match parent")
        prior = retention.snapshot()
        if prior["last_parent_identity_sha256"] != json_digest(parent.identity):
            raise ValueError("retention must describe this exact completed parent")
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "capsules").mkdir()
        (directory / "invocations").mkdir()
        owner = cls._empty(directory / "owner.pt", device)
        owner._parent, owner._retention = parent, retention
        started = time.monotonic()
        with run_lock(directory):
            capsule = owner._export(parent)
            owner._state = dict(schema=SCHEMA, source_sha256=sources, runtime=runtime,
                policy=policy, origin=dict(cycle=parent.identity["cycle"],
                    lifetime_updates=parent.identity["base_updates"],
                    parent_identity_sha256=json_digest(parent.identity)),
                phase="boundary", parent_capsule=capsule,
                retention=prior, retention_sha256=json_digest(prior),
                evaluation_banks=copy.deepcopy(evaluation_banks), completed_cycles=0,
                completed_work=_zero_work(), active=None, unknown_invocations=[])
            owner._prepare()
            owner._commit()
        owner.setup_report = dict(wall_seconds=time.monotonic()-started,
            scope="Capsule export, admitted fresh plan/index and child setup; no training or scoring.")
        return owner

    @classmethod
    def _empty(cls, path, device):
        result = object.__new__(cls)
        result._path, result._device = Path(path).resolve(), torch.device(device)
        result._lock = threading.RLock()
        result._head = result._pending_publication = result._committed = None
        result._uncertain = False
        result._loop = result._parent = result._retention = result._state = None
        result.last_report = None
        result.setup_report = None
        return result

    def _relative(self, value):
        if type(value) is not str or "\\" in value or Path(value).is_absolute():
            raise ValueError("canonical relative artifact path required")
        result = (self._path.parent / value).resolve()
        if not result.is_relative_to(self._path.parent) or result == self._path.parent:
            raise ValueError("artifact path escapes owner directory")
        if result.relative_to(self._path.parent).as_posix() != value:
            raise ValueError("noncanonical artifact path")
        return result

    def _export(self, parent):
        name = f"capsules/cycle-{parent.identity['cycle']:06d}-{uuid.uuid4().hex}.zip"
        receipt = export_parent_capsule(parent, self._relative(name))
        return dict(path=name, sha256=receipt["sha256"], parent_identity_sha256=receipt["parent_identity_sha256"])

    def _prepare(self):
        """Deterministic admission, before any new training; never a retry loop."""
        parent, policy = self._parent, self._state["policy"]
        protection = parent.protected_transcripts
        plan, receipt = repair_plan(_base_plan(policy, parent.identity["cycle"]+1), protection)
        _, transcripts = prepare_training(plan, protected_transcripts=protection, anchor_limit=1)
        index = AuthenticatedPlanIndex(plan, admission_receipt=receipt,
            admission_protected_transcripts=protection, protected_transcripts=protection)
        if (index.identity["transcript_sha256"] != json_digest(transcripts)
                or index.identity["unique_transcripts"] != len(transcripts)):
            raise ValueError("prepared transcript inventory differs from authenticated index")
        provider = CycleFoundationPracticeProvider(plan, parent=parent,
            evaluation_banks=self._state["evaluation_banks"], device=self._device,
            admission_protected_transcripts=protection, protected_transcripts=protection,
            admission_receipt=receipt, plan_index=index)
        loop = loops.FoundationLoop(provider, **policy["loop_options"])
        self._state = {**self._state, "phase": "active", "active": dict(plan=plan,
            admission_receipt=receipt, protected_transcripts=protection,
            training_transcripts=transcripts, index_identity=index.identity, loop=loop.snapshot())}
        self._loop = loop

    def _completed(self):
        active = self._state["active"]
        parent = CompletedParent.from_loop(self._loop, training_transcripts=active["training_transcripts"])
        retention = self._retention.advance(parent)
        capsule = self._export(parent)
        saved = self._loop.snapshot()
        value = retention.snapshot()
        self._state = {**self._state, "phase": "boundary", "active": None,
            "parent_capsule": capsule, "retention": value, "retention_sha256": json_digest(value),
            "completed_cycles": self._state["completed_cycles"]+1,
            "completed_work": _add_work(self._state["completed_work"], saved["state"]["work"])}
        self._parent, self._retention, self._loop = parent, retention, None

    def _work(self):
        result = self._state["completed_work"]
        return copy.deepcopy(result) if self._loop is None else _add_work(result, self._loop._state["work"])

    def _lifetime(self):
        return self._parent.identity["base_updates"] if self._loop is None else self._loop.provider.status()["lifetime_updates"]

    def _check(self):
        if self._uncertain:
            raise RuntimeError("owner publication or accounting is uncertain; explicit pinned reload required")
        if self._state["source_sha256"] != source_hashes():
            raise RuntimeError("owner sources changed; no implicit migration")
        if self._state["runtime"] != providers.runtime_identity(self._device):
            raise RuntimeError("owner runtime changed")
        self._parent.identity
        self._retention.snapshot()

    def _disk_current(self):
        actual = _digest(self._path) if self._path.exists() else None
        if actual != self._head:
            self._uncertain = True
            raise RuntimeError("owner envelope changed externally; pinned reload required")

    def _unresolved(self, acknowledged=None):
        directory = self._path.parent / "invocations"
        if acknowledged is None:
            acknowledged = [] if self._state is None else self._state["unknown_invocations"]
        unresolved = []
        for path in sorted(directory.glob("*.intent.json")):
            digest = _digest(path)
            def read(suffix):
                try:
                    value = json.loads(path.with_name(path.name.replace(".intent.json", suffix)).read_text("utf8"))
                    if (type(value) is not dict or value.get("schema") != INTENT_SCHEMA
                            or value.get("intent_sha256") != digest or not _sha(value.get("owner_sha256"))):
                        return None
                    return value
                except (OSError, ValueError, TypeError):
                    return None
            recovery = read(".recovery.json")
            if (recovery is not None and recovery.get("status") == "unknown_work_acknowledged"
                    and recovery.get("physical_work_unknown") is True
                    and dict(intent=path.relative_to(self._path.parent).as_posix(), sha256=digest) in acknowledged):
                continue
            receipt = read(".receipt.json")
            required = {*REPORT_COUNTS, *loops.WORK_FIELDS, "retained_updates", "completed_cycles",
                "starting_lifetime_updates", "ending_lifetime_updates", "physical_work_unknown",
                "unknown_evaluation_episode_exposures", "publication_uncertain", "wall_seconds"}
            if (receipt is None or not required <= set(receipt)
                    or receipt.get("status") not in ("update_bound", "cycle_bound", "deadline", "failed")
                    or receipt.get("physical_work_unknown") is not False
                    or receipt.get("publication_uncertain") is not False
                    or receipt.get("unknown_evaluation_episode_exposures") is not False
                    or any(receipt.get(name) != 0 for name in ("unknown_practice_attempts",
                        "unknown_optimizer_attempts", "incomplete_evaluation_attempts"))):
                unresolved.append(path)
                continue
            try:
                for name in (*REPORT_COUNTS, *loops.WORK_FIELDS, "retained_updates", "completed_cycles",
                             "starting_lifetime_updates", "ending_lifetime_updates"):
                    _integer(receipt[name], "receipt " + name)
                providers._finite(receipt["wall_seconds"], "receipt wall")
            except (ValueError, TypeError):
                unresolved.append(path)
        return unresolved

    def _entry(self):
        self._check(); self._disk_current()
        if self._unresolved():
            raise UnknownInvocationWork("unfinished invocation has unknown physical work; explicit recovery_policy required")
        if self._loop is not None:
            if self._loop._disk_path is not None or not loops._same_tree(self._loop.snapshot(), self._committed["active"]["loop"]):
                self._uncertain = True
                raise RuntimeError("inner loop changed outside its owner; explicit pinned reload required")

    def _snapshot_current(self):
        value = copy.deepcopy(self._state)
        if self._loop is not None:
            value["active"]["loop"] = self._loop.snapshot()
        _check_finite_tree(value, "cycle owner envelope")
        return value

    def snapshot(self):
        with self._lock:
            self._entry()
            return copy.deepcopy(self._committed)

    @property
    def status(self):
        with self._lock:
            self._entry()
            return dict(phase=self._state["phase"], completed_cycles=self._state["completed_cycles"],
                parent_cycle=self._parent.identity["cycle"],
                active_cycle=None if self._loop is None else self._loop.provider._trainer.cycle,
                cycle_updates=None if self._loop is None else self._loop.provider.status()["updates"],
                lifetime_updates=self._lifetime(), retained_work=self._work(),
                owner_sha256=self._head, path=str(self._path),
                acknowledged_unknown_invocations=len(self._state["unknown_invocations"]),
                automatic_promotion=False)

    def _write(self, payload):
        self._disk_current()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", prefix="owner.", suffix=".tmp",
                    dir=self._path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
            digest = _digest(temporary)
            self._pending_publication = dict(payload=payload, digest=digest, prior=self._head)
            os.replace(temporary, self._path)
            self._head = digest
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _commit(self):
        self._check()
        payload = self._snapshot_current()
        self._pending_publication = None
        try:
            self._write(payload)
            self._state = payload
            self._committed = copy.deepcopy(payload)
        except BaseException:
            pending = self._pending_publication
            if pending is not None:
                try:
                    actual = _digest(self._path) if self._path.exists() else None
                    if actual == pending["digest"]:
                        self._head, self._state = actual, pending["payload"]
                        self._committed = copy.deepcopy(pending["payload"])
                    elif actual != pending["prior"]:
                        self._uncertain = True
                except BaseException:
                    self._uncertain = True
            raise
        finally:
            self._pending_publication = None

    def _restore(self, payload):
        """Validate one pinned owner image; reconstruct only its active index."""
        value = copy.deepcopy(payload)
        _keys(value, ("schema", "source_sha256", "runtime", "policy", "origin", "phase", "parent_capsule",
            "retention", "retention_sha256", "evaluation_banks", "completed_cycles", "completed_work",
            "active", "unknown_invocations"), "owner")
        _check_finite_tree(value, "owner envelope")
        if value["schema"] != SCHEMA or value["source_sha256"] != source_hashes():
            raise ValueError("owner schema/source identity differs")
        policy = value["policy"]
        _keys(policy, ("root_seed", "plan_config", "loop_options", "order", "seed_rule", "automatic_promotion"), "policy")
        if policy != _policy(policy["root_seed"], policy["plan_config"], policy["loop_options"]):
            raise ValueError("noncanonical owner policy")
        ref = value["parent_capsule"]
        _keys(ref, ("path", "sha256", "parent_identity_sha256"), "parent capsule reference")
        if not _sha(ref["sha256"]) or not _sha(ref["parent_identity_sha256"]):
            raise ValueError("parent capsule digest required")
        parent = load_parent_capsule(self._relative(ref["path"]), expected_sha256=ref["sha256"],
            device=self._device, evaluation_banks=value["evaluation_banks"])
        identity = parent.identity
        if json_digest(identity) != ref["parent_identity_sha256"] or value["runtime"] != parent.provider_identity["runtime"]:
            raise ValueError("owner parent/runtime differs")
        retention = LifetimeRetention.restore(value["retention"], expected_sha256=value["retention_sha256"], parent=parent)
        origin = value["origin"]
        _keys(origin, ("cycle", "lifetime_updates", "parent_identity_sha256"), "origin")
        for key in ("cycle", "lifetime_updates"):
            _integer(origin[key], "origin " + key, 1)
        if not _sha(origin["parent_identity_sha256"]):
            raise ValueError("origin identity digest required")
        _integer(value["completed_cycles"], "completed cycles")
        if identity["cycle"] != origin["cycle"] + value["completed_cycles"] or identity["base_updates"] < origin["lifetime_updates"]:
            raise ValueError("owner generation/lifetime differs from origin")
        if value["completed_cycles"] == 0 and ref["parent_identity_sha256"] != origin["parent_identity_sha256"]:
            raise ValueError("initial parent differs from origin")
        _keys(value["completed_work"], WORK_FIELDS, "completed work")
        for key, count in value["completed_work"].items():
            if key.endswith("seconds"):
                providers._finite(count, key)
            else:
                _integer(count, key)
        if value["completed_work"]["physical_optimizer_updates"] != identity["base_updates"]-origin["lifetime_updates"]:
            raise ValueError("completed retained work differs from lifetime")
        unknown = value["unknown_invocations"]
        if (type(unknown) is not list or any(type(item) is not dict or set(item) != {"intent", "sha256"}
                or not _sha(item["sha256"]) for item in unknown)
                or len({item["intent"] for item in unknown}) != len(unknown)):
            raise ValueError("unknown invocation inventory differs")
        for item in unknown:
            self._relative(item["intent"])
        loop = None
        if value["phase"] == "active":
            active = value["active"]
            _keys(active, ("plan", "admission_receipt", "protected_transcripts", "training_transcripts",
                          "index_identity", "loop"), "active cycle")
            plan = active["plan"]
            planning.validate_plan(plan)
            expected = _base_plan(policy, identity["cycle"]+1)
            if plan["config"] != expected["config"] or active["protected_transcripts"] != parent.protected_transcripts:
                raise ValueError("active plan policy or cumulative protection differs")
            transcripts = active["training_transcripts"]
            if type(transcripts) is not list or transcripts != sorted(transcript_set(transcripts)):
                raise ValueError("canonical active transcript inventory required")
            index = AuthenticatedPlanIndex(plan, admission_receipt=active["admission_receipt"],
                admission_protected_transcripts=active["protected_transcripts"],
                protected_transcripts=active["protected_transcripts"])
            if (index.identity != active["index_identity"] or index.identity["transcript_sha256"] != json_digest(transcripts)
                    or index.identity["unique_transcripts"] != len(transcripts)):
                raise ValueError("active index or transcript identity differs")
            provider = CycleFoundationPracticeProvider(plan, parent=parent,
                evaluation_banks=value["evaluation_banks"], device=self._device,
                admission_protected_transcripts=active["protected_transcripts"],
                protected_transcripts=active["protected_transcripts"], admission_receipt=active["admission_receipt"],
                plan_index=index)
            loop = loops.FoundationLoop(provider, **policy["loop_options"], payload=active["loop"])
            if loop.status["phase"] == "complete":
                raise ValueError("completed cycles must be atomically folded to a boundary")
        elif value["phase"] != "boundary" or value["active"] is not None:
            raise ValueError("owner phase or boundary payload differs")
        if value["source_sha256"] != source_hashes():
            raise RuntimeError("owner sources changed during restore")
        self._state, self._parent, self._retention, self._loop = value, parent, retention, loop

    @classmethod
    def load(cls, path, *, expected_sha256, min_generation=None, device="cpu", recovery_policy="stop"):
        """A current caller pin authenticates the owner and its capsule/data links.

        ``acknowledge_unknown`` explicitly accepts unmeasured work after a stale
        invocation; it records that uncertainty and never labels it zero work.
        An extant run lock must independently be resolved by the caller.
        """
        if cls is not CycleOwner or not _sha(expected_sha256) or recovery_policy not in ("stop", "acknowledge_unknown"):
            raise ValueError("exact caller pin and explicit supported recovery policy required")
        if min_generation is not None:
            _integer(min_generation, "minimum generation", 1)
        result = cls._empty(path, device)
        with result._path.open("rb") as stream:
            raw = stream.read(MAX_BYTES+1)
        if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("owner differs from caller pin or supported byte limit")
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        result._head = expected_sha256
        unresolved = result._unresolved(acknowledged=payload.get("unknown_invocations", []) if type(payload) is dict else [])
        if unresolved and recovery_policy == "stop":
            raise UnknownInvocationWork("unfinished invocation may contain unreported physical work; no automatic replay")
        result._restore(payload)
        if min_generation is not None and result._parent.identity["cycle"] < min_generation:
            raise ValueError("owner generation precedes caller floor")
        result._committed = copy.deepcopy(result._state)
        result._disk_current()
        if unresolved:
            with run_lock(result._path.parent):
                result._disk_current()
                for path in unresolved:
                    relative = path.relative_to(result._path.parent).as_posix()
                    entry = dict(intent=relative, sha256=_digest(path))
                    if entry not in result._state["unknown_invocations"]:
                        result._state["unknown_invocations"].append(entry)
                result._commit()
                for path in unresolved:
                    _exclusive_json(path.with_name(path.name.replace(".intent.json", ".recovery.json")),
                        dict(schema=INTENT_SCHEMA, status="unknown_work_acknowledged", physical_work_unknown=True,
                            intent_sha256=_digest(path), owner_sha256=result._head,
                            scope="Caller explicitly acknowledged unknown physical work; no lost work is inferred to be zero."))
        return result

    @staticmethod
    def _account(report, value):
        for name in REPORT_COUNTS:
            report[name] += value[name]
        for name in loops.WORK_FIELDS:
            report[name] = None if report[name] is None or value[name] is None else report[name]+value[name]
        for name in ("training_seconds", "evaluation_seconds"):
            report[name] += value[name]
        report["unknown_evaluation_episode_exposures"] |= value["unknown_evaluation_episode_exposures"]
        report["inner_calls"] += 1

    def run(self, *, max_updates=None, max_cycles=None, deadline=None):
        """Use a finite update/cycle allowance and/or an absolute monotonic deadline.

        Once updates are exhausted, an active evaluation panel may finish, but
        no further plan is admitted or update attempted. Zero cycles permits no
        work. Completed boundaries are folded exactly once, including on resume.
        """
        started = time.monotonic()
        report = dict(schema=INTENT_SCHEMA, status="preflight", **dict.fromkeys(REPORT_COUNTS, 0),
            **dict.fromkeys(loops.WORK_FIELDS, 0), training_seconds=0., evaluation_seconds=0.,
            admission_seconds=0., completion_seconds=0., checkpoint_seconds=0., rollback_seconds=0.,
            inner_calls=0, unknown_evaluation_episode_exposures=False,
            automatic_promotion=False, scope=SCOPE)
        with self._lock:
            self.last_report = report
            if max_updates is None and max_cycles is None and deadline is None:
                raise ValueError("an explicit update, completed-cycle or deadline bound is required")
            for name, value in (("max_updates", max_updates), ("max_cycles", max_cycles)):
                if value is not None:
                    _integer(value, name)
            if deadline is not None:
                providers._finite(deadline, "monotonic deadline")
            self._entry()
            with run_lock(self._path.parent):
                self._entry()
                initial_lifetime, initial_work = self._lifetime(), self._work()
                initial_cycles = self._state["completed_cycles"]
                report.update(starting_lifetime_updates=initial_lifetime, starting_owner_sha256=self._head,
                    allowances=dict(max_updates=max_updates, max_cycles=max_cycles, deadline=deadline))
                name = uuid.uuid4().hex
                intent_path = self._path.parent / "invocations" / (name + ".intent.json")
                _exclusive_json(intent_path, dict(schema=INTENT_SCHEMA, status="started", invocation=name,
                    source_sha256=self._state["source_sha256"], owner_sha256=self._head,
                    starting_lifetime_updates=initial_lifetime, allowances=report["allowances"],
                    pid=os.getpid(), scope="No receipt means unknown physical work; do not automatically replay."))
                report["invocation"] = name
                failure = None
                try:
                    while True:
                        self._check(); self._disk_current()
                        completed = self._state["completed_cycles"]-initial_cycles
                        if max_cycles is not None and completed >= max_cycles:
                            report["status"] = "cycle_bound"; break
                        if deadline is not None and time.monotonic() >= deadline:
                            report["status"] = "deadline"; break
                        remaining = None if max_updates is None else max_updates-report["completed_updates"]
                        if self._loop is None:
                            if remaining == 0:
                                report["status"] = "update_bound"; break
                            tick = time.monotonic()
                            try: self._prepare()
                            finally: report["admission_seconds"] += time.monotonic()-tick
                            tick = time.monotonic()
                            try: self._commit()
                            finally: report["checkpoint_seconds"] += time.monotonic()-tick
                            continue
                        if remaining == 0 and self._loop._state["phase"] == "practice":
                            report["status"] = "update_bound"; break
                        limit = self._state["policy"]["loop_options"]["chunk_updates"]
                        if remaining is not None:
                            limit = min(limit, remaining)
                        self._loop.last_report = None
                        try:
                            inner = self._loop.run(max_updates=limit, max_evaluations=1, deadline=deadline)
                        finally:
                            if self._loop.last_report is not None:
                                self._account(report, self._loop.last_report)
                        if self._loop._state["phase"] == "complete":
                            tick = time.monotonic()
                            try: self._completed()
                            finally: report["completion_seconds"] += time.monotonic()-tick
                        tick = time.monotonic()
                        try: self._commit()
                        finally: report["checkpoint_seconds"] += time.monotonic()-tick
                        if inner["status"] == "deadline":
                            report["status"] = "deadline"; break
                except BaseException as error:
                    failure = error
                    report.update(status="failed", error=repr(error))
                    if not self._uncertain:
                        tick = time.monotonic()
                        try: self._restore(self._committed)
                        except BaseException as recovery:
                            self._uncertain = True
                            report["recovery_error"] = repr(recovery)
                        finally: report["rollback_seconds"] += time.monotonic()-tick
                finally:
                    retained = None if self._uncertain else self._lifetime()-initial_lifetime
                    current_work = None if self._uncertain else self._work()
                    report.update(retained_updates=retained, owner_sha256=self._head,
                        ending_lifetime_updates=None if self._uncertain else self._lifetime(),
                        completed_cycles=None if self._uncertain else self._state["completed_cycles"]-initial_cycles,
                        publication_uncertain=self._uncertain, wall_seconds=time.monotonic()-started,
                        acknowledged_unknown_invocations=len(self._state["unknown_invocations"]))
                    physical = report["physical_optimizer_updates"]
                    report["discarded_optimizer_updates"] = None if retained is None or physical is None else max(0, physical-retained)
                    report["discarded_completed_updates"] = None if retained is None else max(0, report["completed_updates"]-retained)
                    for key in ("evaluation_banks", "evaluation_episode_exposures"):
                        kept = None if current_work is None else current_work[key]-initial_work[key]
                        report["retained_"+key] = kept
                        report["discarded_"+key] = None if kept is None else report[key]-kept
                    try:
                        _exclusive_json(intent_path.with_name(name + ".receipt.json"),
                            dict(**report, intent_sha256=_digest(intent_path), physical_work_unknown=self._uncertain
                                or report["physical_optimizer_updates"] is None
                                or report["unknown_evaluation_episode_exposures"]))
                    except BaseException as receipt_error:
                        self._uncertain = True
                        report.update(receipt_error=repr(receipt_error), publication_uncertain=True,
                                      physical_work_unknown=True)
                        if failure is None: failure = receipt_error
                    self.last_report = copy.deepcopy(report)
                if failure is not None:
                    raise failure
                return copy.deepcopy(report)
