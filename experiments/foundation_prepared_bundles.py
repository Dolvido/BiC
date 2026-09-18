"""Transient, bounded CPU preparation; never a learner or persistent cache.

One producer owns at most a current bundle and one next bundle, counting an
active preparation and an outstanding consumer lease. Public index replay is
captured once at setup; subsequent checks do not clone an increasing prefix.
Take/release counts describe storage handoff, never neural exposure or commits.
Callers must validate their own update allowance before using a lease in a model.

Close requests stopping and joins for a bounded time. Python cannot interrupt an
active generator/packer; an unjoined daemon remains explicitly incomplete and
may finish that operation. No claim of atomic snapshots against concurrent
hostile mutation, sandboxing of Python internals, or throughput benefit is made.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import threading
import time
import weakref

import torch

from experiments import foundation_training as canonical
from experiments.foundation_curriculum import FAMILIES, validate_pair
from experiments.foundation_plan_index import AuthenticatedPlanIndex, source_hashes as index_sources
from experiments.sequence_student import SequenceConfig


SCHEMA = "bic-foundation-prepared-bundles-v1"
_MINT = object()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def source_hashes():
    name = "experiments/foundation_prepared_bundles.py"
    root = Path(__file__).resolve().parents[1]
    return {**index_sources(), name: hashlib.sha256((root/name).read_bytes()).hexdigest()}


def _runtime():
    # No CUDA API, model, RNG, optimizer or thread-setting access.
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads())


def _seconds(value, name, *, zero=False, maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value < 0 if zero else value <= 0) or maximum is not None and value > maximum):
        raise ValueError(name + " must be a bounded finite allowance")
    return float(value)


def _tensor_identity(batches):
    """Exact CPU integer/Boolean value identity, including shape and layout."""
    if type(batches) is not dict or tuple(batches) != tuple(FAMILIES):
        raise ValueError("exact canonical family order required")
    records, payload_bytes = {}, 0
    for family, batch in batches.items():
        if type(batch) is not dict or set(batch) != {"inputs", "supervision"}:
            raise ValueError("canonical packed input/supervision fields required")
        records[family] = {}
        for section, values in batch.items():
            if type(values) is not dict or not values:
                raise ValueError("nonempty named packed tensors required")
            records[family][section] = {}
            for name, tensor in values.items():
                if (type(name) is not str or type(tensor) is not torch.Tensor
                        or tensor.device.type != "cpu" or tensor.layout != torch.strided
                        or tensor.dtype not in (torch.int64, torch.bool) or tensor.requires_grad):
                    raise ValueError("packed inputs must remain ordinary CPU integer/Boolean tensors")
                image = tensor.detach().contiguous().numpy().tobytes()
                records[family][section][name] = dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
                    stride=list(tensor.stride()), storage_offset=tensor.storage_offset(),
                    sha256=hashlib.sha256(image).hexdigest())
                payload_bytes += tensor.numel()*tensor.element_size()
    return _digest(records), payload_bytes


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PreparedBundle:
    """Process-owned lease; metadata is detached, tensor views require validation."""
    _owner: object
    _identity: bytes
    _evidence: bytes
    _batches: dict
    _payload_bytes: int

    def __init__(self, mint, owner, identity, evidence, batches, payload_bytes):
        if mint is not _MINT or type(owner) is not PreparedBundleOwner:
            raise TypeError("prepared bundles are minted only by their live owner")
        for name, value in (("_owner", weakref.ref(owner)), ("_identity", _encoded(identity)),
                            ("_evidence", _encoded(evidence)), ("_batches", batches),
                            ("_payload_bytes", payload_bytes)):
            object.__setattr__(self, name, value)

    @property
    def identity(self):
        return json.loads(self._identity)

    @property
    def evidence(self):
        return json.loads(self._evidence)

    @property
    def batches(self):
        """New containers referencing lease tensors, suitable for explicit transfer.

        Tensors are not cloned. Mutating them invalidates the lease; a later
        versioned trainer must validate immediately before its consumer boundary.
        Retaining caller references after release lies outside owner memory bounds.
        """
        owner = self._owner()
        if owner is None:
            raise RuntimeError("prepared owner no longer exists")
        owner.validate(self)
        return {family: {section: dict(values) for section, values in batch.items()}
                for family, batch in self._batches.items()}

    def __reduce_ex__(self, protocol):
        raise TypeError("prepared leases are transient and cannot be serialized")


class _Stopped(Exception):
    pass


class PreparedBundleOwner:
    """Single-consumer lease owner with exactly one CPU producer thread.

    Construction starts preparation, not learning. Inputs are detached, the
    exact supplied index remains process-owned, and only its public full replay
    is captured once. Bounds include active work. No retry/reset/restore path is
    supplied. New owners after an explicit learner restore must charge setup and
    preparation again; this class cannot authorize that restart.
    """
    def __init__(self, plan, order="curriculum", *, plan_index, config,
                 admission_protected_transcripts, protected_transcripts, admission_receipt,
                 start_cursor=0, stop_cursor=None, max_seconds):
        if type(self) is not PreparedBundleOwner:
            raise TypeError("exact PreparedBundleOwner required")
        started, cpu_started = time.monotonic(), time.thread_time()
        allowance = _seconds(max_seconds, "max_seconds")
        if type(plan_index) is not AuthenticatedPlanIndex or type(config) is not SequenceConfig:
            raise ValueError("actual authenticated process index and exact SequenceConfig required")
        self._consumer = threading.get_ident()
        self._condition = threading.Condition()
        self._index, self._config = plan_index, SequenceConfig(**asdict(config))
        self._plan = canonical._admit_plan(copy.deepcopy(plan), order)
        self._order = order
        history = canonical._protected(admission_protected_transcripts)
        self._protection = canonical._protected(protected_transcripts)
        self._protection_set = frozenset(self._protection)
        receipt = copy.deepcopy(admission_receipt)
        if not set(history) <= set(self._protection) or type(receipt) is not dict:
            raise ValueError("complete original and expanded protection/admission required")
        self._sources, self._runtime = source_hashes(), _runtime()
        self._index_identity = plan_index.identity
        expected = dict(plan_sha256=_digest(self._plan), admission_receipt_sha256=_digest(receipt),
            admission_protection_sha256=_digest(list(history)), admission_protection_count=len(history),
            expanded_protection_sha256=_digest(list(self._protection)), expanded_protection_count=len(self._protection),
            bundle_count=len(self._plan["bundles"]), micro_batch_size=self._plan["config"]["micro_batch_size"])
        if any(self._index_identity.get(key) != value for key, value in expected.items()):
            raise ValueError("supplied index belongs to another admitted plan/protection contract")
        schedule = self._plan["schedules"][order]
        stop_cursor = len(schedule) if stop_cursor is None else stop_cursor
        if (type(start_cursor) is not int or type(stop_cursor) is not int
                or not 0 <= start_cursor <= stop_cursor <= len(schedule)):
            raise ValueError("bounded authorized preparation cursor window required")
        replay = plan_index.replay(order, len(schedule), include_bundles=True)
        self._records = tuple(_encoded(row) for row in replay["bundles"])
        if len(self._records) != len(schedule) or any(
                row["bundle_id"] != bundle_id for row, bundle_id in zip(replay["bundles"], schedule)):
            raise ValueError("authenticated replay schedule differs")
        self._declaration = dict(schema=SCHEMA, index_identity_sha256=_digest(self._index_identity),
            plan_sha256=expected["plan_sha256"], protection_sha256=expected["expanded_protection_sha256"],
            config=asdict(self._config), order=order, source_sha256=self._sources, cpu_runtime=self._runtime,
            start_cursor=start_cursor, stop_cursor=stop_cursor, max_seconds=allowance)
        self._guard()
        self._started, self._deadline = started, started+allowance
        self._cursor = self._next = start_cursor
        self._stop_cursor = stop_cursor
        self._ready, self._lease, self._active = deque(), None, None
        self._stopping, self._closed, self._error = False, False, None
        self._active_tensor_bytes = 0
        self._counts = dict(submitted_bundles=0, prepared_bundles=0, delivered_leases=0, released_leases=0,
            discarded_prepared_bundles=0, abandoned_leases=0, interrupted_preparations=0, failed_preparations=0,
            materialized_bundles=0, materialized_episodes=0, evidence_validated_bundles=0,
            packed_family_batches=0, packed_episodes=0, consumer_rejections=0)
        self._phases, self._stop_reason = {}, None
        self._stop_requested_at = self._worker_finished_at = None
        self._producer_runtime = None
        self._peak_resident, self._peak_tensor_bytes = 0, 0
        self._wait_seconds = self._consumer_check_seconds = self._join_seconds = 0.
        self._take_cpu = self._validation_seconds = self._validation_cpu = 0.
        self._preparation_seconds = self._preparation_cpu = 0.
        self._setup_seconds, self._setup_cpu = time.monotonic()-started, time.thread_time()-cpu_started
        self._worker = threading.Thread(target=self._produce, name="foundation-cpu-preparation", daemon=True)
        self._worker.start()

    @property
    def declaration(self):
        return copy.deepcopy(self._declaration)

    def _consumer_only(self):
        if threading.get_ident() != self._consumer:
            raise RuntimeError("lease operations belong to the original consumer thread")

    def _guard(self):
        if source_hashes() != self._sources or _runtime() != self._runtime:
            raise ValueError("preparation source or CPU runtime identity changed")
        if self._index.identity != self._index_identity:
            raise ValueError("authenticated process index identity changed")

    def _discard_ready(self):
        self._counts["discarded_prepared_bundles"] += len(self._ready)
        self._ready.clear()

    def _stop(self, reason):
        # Caller holds the condition. An active operation is never claimed canceled.
        self._stopping = True
        if self._stop_requested_at is None:
            self._stop_requested_at = time.monotonic()
        self._stop_reason = self._stop_reason or reason
        self._discard_ready()
        self._condition.notify_all()

    def _peak(self):
        resident = len(self._ready) + int(self._lease is not None) + int(self._active is not None)
        payload = sum(item._payload_bytes for item in self._ready) + self._active_tensor_bytes
        if self._lease is not None:
            payload += self._lease._payload_bytes
        self._peak_resident = max(self._peak_resident, resident)
        self._peak_tensor_bytes = max(self._peak_tensor_bytes, payload)
        if resident > 2:
            raise RuntimeError("preparation resident bundle bound violated")

    def _boundary(self):
        with self._condition:
            if time.monotonic() >= self._deadline:
                self._stop("deadline")
            if self._stopping:
                raise _Stopped()

    def _phase(self, name, function):
        self._boundary()
        started, cpu = time.monotonic(), time.thread_time()
        with self._condition:
            self._active["phase"] = name
            value = self._phases.setdefault(name, dict(attempted=0, completed=0, failed=0,
                                                        wall_seconds=0., thread_cpu_seconds=0.))
            value["attempted"] += 1
        completed = False
        try:
            result = function()
            completed = True
            return result
        finally:
            with self._condition:
                value["completed" if completed else "failed"] += 1
                value["wall_seconds"] += time.monotonic()-started
                value["thread_cpu_seconds"] += time.thread_time()-cpu

    def _prepare(self, cursor):
        bundle_id = self._plan["schedules"][self._order][cursor]
        self._phase("source_checks", self._guard)
        rows = self._phase("materialization", lambda: canonical._materialize_validated_bundle(self._plan, bundle_id))
        with self._condition:
            self._counts["materialized_bundles"] += 1
            self._counts["materialized_episodes"] += sum(map(len, rows.values()))
        evidence = self._phase("evidence", lambda: canonical._rows_evidence(self._plan, bundle_id, rows, self._protection_set))
        if _encoded(evidence) != self._records[cursor]:
            raise ValueError("prepared row/recipe/exposure evidence differs from authenticated index")
        with self._condition:
            self._counts["evidence_validated_bundles"] += 1
        batches = {}
        for family in FAMILIES:
            batches[family] = self._phase("packing", lambda: canonical.pack_composition_episodes(
                rows[family], device="cpu", training=True, pair_validator=validate_pair,
                max_turns=self._config.max_turns, max_input_bytes=self._config.max_input_bytes,
                max_context_tokens=self._config.max_positions, max_reply_bytes=self._config.max_output_bytes))
            with self._condition:
                self._counts["packed_family_batches"] += 1
                self._counts["packed_episodes"] += len(rows[family])
                self._active_tensor_bytes = sum(t.numel()*t.element_size() for batch in batches.values()
                    for section in batch.values() for t in section.values())
                self._peak()
        tensor_digest, tensor_bytes = self._phase("tensor_identity", lambda: _tensor_identity(batches))
        self._phase("source_checks", self._guard)
        identity = dict(schema=SCHEMA, declaration_sha256=_digest(self._declaration), cursor=cursor,
            bundle_id=bundle_id, evidence_sha256=_digest(evidence), tensors_sha256=tensor_digest)
        return PreparedBundle(_MINT, self, identity, evidence, batches, tensor_bytes)

    def _produce(self):
        try:
            with self._condition:
                self._producer_runtime = _runtime()
            self._producer_loop()
        except BaseException as error:
            # Also preserve startup/scheduling failures outside a bundle phase.
            with self._condition:
                self._error = dict(error=type(error).__name__ + ": " + str(error),
                    active=copy.deepcopy(self._active), unreturned_operation_work_may_be_unknown=self._active is not None)
                if self._active is not None:
                    self._counts["failed_preparations"] += 1
                self._active, self._active_tensor_bytes = None, 0
                self._stop("producer startup or scheduling failure")
        finally:
            with self._condition:
                self._worker_finished_at = time.monotonic()
                self._condition.notify_all()

    def _producer_loop(self):
        while True:
            with self._condition:
                while not self._stopping and self._next < self._stop_cursor and (
                        len(self._ready) + int(self._lease is not None) >= 2):
                    remaining = self._deadline-time.monotonic()
                    if remaining <= 0:
                        self._stop("deadline")
                        break
                    self._condition.wait(min(remaining, 60.))
                if self._stopping or self._next >= self._stop_cursor:
                    return
                if time.monotonic() >= self._deadline:
                    self._stop("deadline")
                    return
                cursor = self._next
                self._next += 1
                self._counts["submitted_bundles"] += 1
                self._active = dict(cursor=cursor, bundle_id=self._plan["schedules"][self._order][cursor], phase="starting")
                self._peak()
            item = None
            started, cpu_started = time.monotonic(), time.thread_time()
            try:
                item = self._prepare(cursor)
            except _Stopped:
                with self._condition:
                    self._counts["interrupted_preparations"] += 1
            except BaseException as error:
                with self._condition:
                    self._counts["failed_preparations"] += 1
                    self._error = dict(error=type(error).__name__ + ": " + str(error), active=copy.deepcopy(self._active),
                        unreturned_operation_work_may_be_unknown=True)
                    self._stop("producer failure")
            finally:
                with self._condition:
                    self._preparation_seconds += time.monotonic()-started
                    self._preparation_cpu += time.thread_time()-cpu_started
                    self._active, self._active_tensor_bytes = None, 0
                    if item is not None:
                        self._counts["prepared_bundles"] += 1
                        if self._stopping:
                            self._counts["discarded_prepared_bundles"] += 1
                        else:
                            self._ready.append(item)
                    self._peak()
                    self._condition.notify_all()

    def _check_token(self, item):
        if type(item) is not PreparedBundle or item._owner() is not self:
            raise ValueError("lease belongs to another live preparation owner")
        identity = item.identity
        if (identity["declaration_sha256"] != _digest(self._declaration)
                or identity["cursor"] != self._cursor
                or identity["bundle_id"] != self._plan["schedules"][self._order][self._cursor]
                or item._evidence != self._records[self._cursor]
                or identity["evidence_sha256"] != _digest(item.evidence)
                or _tensor_identity(item._batches) != (identity["tensors_sha256"], item._payload_bytes)):
            raise ValueError("prepared cursor, evidence or CPU tensors were modified")

    def _rejected(self, error):
        with self._condition:
            self._counts["consumer_rejections"] += 1
            self._error = self._error or dict(error=type(error).__name__ + ": " + str(error),
                                             unreturned_operation_work_may_be_unknown=False)
            self._stop("consumer rejection")

    def take(self, cursor, *, order, plan_index, config, protected_transcripts, timeout_seconds):
        """Validate exact consumer identity, then deliver the only allowed lease."""
        self._consumer_only()
        timeout = _seconds(timeout_seconds, "timeout_seconds")
        started, cpu_started = time.monotonic(), time.thread_time()
        try:
            if (type(cursor) is not int or cursor != self._cursor or order != self._order
                    or plan_index is not self._index or type(config) is not SequenceConfig
                    or asdict(config) != asdict(self._config)
                    or canonical._protected(protected_transcripts) != self._protection):
                raise ValueError("consumer cursor/order/index/config/protection differs")
            self._guard()
            deadline = min(self._deadline, started+timeout)
            with self._condition:
                if self._lease is not None:
                    raise RuntimeError("a second outstanding lease is forbidden")
                if self._cursor == self._stop_cursor and not self._stopping:
                    raise StopIteration("authorized preparation window complete")
                wait_started = time.monotonic()
                try:
                    while not self._ready and not self._stopping:
                        remaining = deadline-time.monotonic()
                        if remaining <= 0:
                            self._stop("consumer timeout")
                            raise TimeoutError("prepared bundle wait allowance reached; no retry")
                        self._condition.wait(min(remaining, 60.))
                finally:
                    self._wait_seconds += time.monotonic()-wait_started
                if self._stopping:
                    raise RuntimeError("preparation stopped: " + str(self._error or self._stop_reason))
                if time.monotonic() >= deadline:
                    self._stop("consumer timeout")
                    raise TimeoutError("prepared delivery allowance reached")
                item = self._ready[0]
            self._check_token(item)
            self._guard()
            with self._condition:
                if self._stopping or time.monotonic() >= deadline:
                    raise TimeoutError("preparation stopped or allowance elapsed before delivery")
                self._ready.popleft()
                self._lease = item
                self._counts["delivered_leases"] += 1
                self._condition.notify_all()
            return item
        except StopIteration:
            raise
        except BaseException as error:
            self._rejected(error)
            raise
        finally:
            with self._condition:
                self._consumer_check_seconds += time.monotonic()-started
                self._take_cpu += time.thread_time()-cpu_started

    def validate(self, item):
        """Fresh guard for the outstanding lease; no transfer or neural work."""
        self._consumer_only()
        started, cpu_started = time.monotonic(), time.thread_time()
        try:
            with self._condition:
                if self._closed or item is not self._lease:
                    raise ValueError("lease is absent, already released or closed")
                if time.monotonic() >= self._deadline:
                    self._stop("deadline")
                if self._stopping:
                    raise RuntimeError("no fresh lease use after stop, deadline or producer failure")
            self._guard()
            self._check_token(item)
            self._guard()
            with self._condition:
                if time.monotonic() >= self._deadline:
                    self._stop("deadline")
                if self._stopping:
                    raise RuntimeError("preparation stopped before consumer validation completed")
        except BaseException as error:
            self._rejected(error)
            raise
        finally:
            with self._condition:
                self._validation_seconds += time.monotonic()-started
                self._validation_cpu += time.thread_time()-cpu_started

    def release(self, item):
        """Storage cleanup, permitted after stopping; never approves fresh use.

        Release neither revalidates inputs nor certifies a learner commit. The
        caller must use validate() before consumption, and may always surrender
        its exact current lease after a producer failure or deadline.
        """
        self._consumer_only()
        with self._condition:
            if type(item) is not PreparedBundle or item._owner() is not self or item is not self._lease:
                raise ValueError("only the outstanding lease can release storage")
            self._lease = None
            self._cursor += 1
            self._counts["released_leases"] += 1
            self._condition.notify_all()

    def stop(self):
        self._consumer_only()
        with self._condition:
            self._stop("caller stop")
        return self.report()

    def close(self, *, join_seconds=1.):
        """Request stop; a timed-out join explicitly leaves accounting incomplete."""
        self._consumer_only()
        allowance = _seconds(join_seconds, "join_seconds", zero=True, maximum=60.)
        started = time.monotonic()
        with self._condition:
            self._closed = True
            self._stop("caller close")
            if self._lease is not None:
                self._counts["abandoned_leases"] += 1
                self._lease = None
        self._worker.join(allowance)
        with self._condition:
            self._join_seconds += time.monotonic()-started
        return self.report()

    def report(self):
        with self._condition:
            alive = self._worker.is_alive()
            unknown = bool(self._error and self._error.get("unreturned_operation_work_may_be_unknown"))
            return dict(schema=SCHEMA, counts=dict(self._counts), phases=copy.deepcopy(self._phases),
                declaration_sha256=_digest(self._declaration), next_release_cursor=self._cursor,
                prepared_cursor_limit=self._stop_cursor, pending=len(self._ready), leased=self._lease is not None,
                active=copy.deepcopy(self._active), error=copy.deepcopy(self._error), stopping=self._stopping,
                closed=self._closed, stop_reason=self._stop_reason, worker_alive=alive,
                accounting_final=self._closed and not alive, accounting_complete=self._closed and not alive and not unknown,
                unreturned_operation_work_may_be_unknown=unknown, active_work_incomplete=alive and self._active is not None,
                setup_wall_seconds=self._setup_seconds, setup_thread_cpu_seconds=self._setup_cpu,
                producer_runtime=copy.deepcopy(self._producer_runtime),
                index_record_payload_bytes=sum(map(len, self._records)), index_record_count=len(self._records),
                elapsed_wall_seconds=time.monotonic()-self._started, consumer_wait_seconds=self._wait_seconds,
                take_wall_seconds_including_wait=self._consumer_check_seconds, close_join_seconds=self._join_seconds,
                take_thread_cpu_seconds=self._take_cpu, validation_wall_seconds=self._validation_seconds,
                validation_thread_cpu_seconds=self._validation_cpu,
                attempted_preparation_wall_seconds=self._preparation_seconds,
                attempted_preparation_thread_cpu_seconds=self._preparation_cpu,
                shutdown_overrun_seconds=(0. if self._stop_requested_at is None else
                    max(0., (self._worker_finished_at or time.monotonic())-self._stop_requested_at)),
                peak_resident_bundles=self._peak_resident, peak_logical_tensor_payload_bytes=self._peak_tensor_bytes,
                neural_work_performed=False,
                scope="CPU preparation/storage accounting only. Materialized/packed episodes are not learner exposures. Phase times may overlap caller work. Tensor bytes exclude temporary pack/hash allocations, row objects, metadata, Python overhead and caller-retained views. Missing process completion means unknown work; an unjoined active operation can still finish.")

    def __reduce_ex__(self, protocol):
        raise TypeError("CPU preparation owners are transient and cannot be serialized")
