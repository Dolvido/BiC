"""Exact restart of SharedStateKernel, with a separate authenticated envelope.

The original recipe, evidence and cumulative accounting are retained verbatim.
This adapter authenticates compatibility; it does not replay historical learning,
admit lessons, choose curricula, contact a tutor or promote a checkpoint.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_student import _cpu_copy
from experiments import execution_profile, foundation_layout_training as training
from experiments import shared_state_training as kernel, shared_state_student as student
from experiments.foundation_layout_continuation import _keys, _pin, _count, _same, _decode, _tensor_state
from experiments.sequence_student import SequenceConfig

SCHEMA = "bic-shared-state-continuation-v1"
PRODUCER_SCHEMA = "bic-shared-state-rate-pilot-v1"
RATES = dict(fast=.003, slow=.0003)
ROOT = Path(__file__).resolve().parents[1]
SCOPE = "Exact SharedStateKernel continuation; original recipe and lifetime work preserved. Caller owns authenticated origin, curriculum admission, runtime configuration and publication. No automatic retry or promotion."
COST_KEYS = tuple(f"{kind}_{suffix}" for kind in ("restore", "step", "snapshot")
                  for suffix in ("invocations", "wall_seconds", "cpu_seconds"))
IDENTITY_KEYS = ("producing_schema", "launch_sha256", "arm", "step", "architecture", "config",
                 "learning_rate", "auxiliary_weight", "runtime", "weights_sha256", "recipe", "evidence")


def source_hashes():
    result = kernel.source_hashes()
    for name in ("experiments/shared_state_continuation.py", "experiments/foundation_layout_continuation.py",
                 "experiments/execution_profile.py"):
        result[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


def current_runtime(device="cpu"):
    """Read/validate runtime without changing settings; CUDA must be configured."""
    selected = torch.device(device)
    if selected.type == "cuda":
        return execution_profile.runtime_profile(device)
    if selected.type != "cpu":
        raise ValueError("local CPU or explicitly configured CUDA required")
    return dict(schema="bic-shared-state-cpu-validation-runtime-v1", torch_version=str(torch.__version__),
                python_version=platform.python_version(), cpu_threads=torch.get_num_threads(),
                cpu_interop_threads=torch.get_num_interop_threads(), default_dtype=str(torch.get_default_dtype()),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def _json_metadata(value):
    # Only external metadata crosses JSON's tuple/list boundary. Tensor and
    # optimizer state comparisons below remain byte/type/layout exact.
    return json.loads(training._json(value))


def _identity(value):
    _keys(value, IDENTITY_KEYS, "caller identity")
    result = _json_metadata(value)
    _count(result["step"], "origin step")
    if (result["producing_schema"] != PRODUCER_SCHEMA or result["arm"] not in RATES
            or not _pin(result["launch_sha256"]) or not _pin(result["weights_sha256"])
            or result["step"] < 1 or result["architecture"] != student.ARCHITECTURE
            or type(result["learning_rate"]) is not float or result["learning_rate"] != RATES[result["arm"]]
            or type(result["auxiliary_weight"]) is not float or result["auxiliary_weight"] != .3
            or type(result["runtime"]) is not dict):
        raise ValueError("declared rate-pilot architecture/rate/weights/runtime identity required")
    recipe = result["recipe"]
    if (type(recipe) is not dict or recipe.get("schema") != kernel.SCHEMA
            or recipe.get("config") != result["config"] or recipe.get("layout") != "original"
            or recipe.get("architecture") != student.ARCHITECTURE or recipe.get("auxiliary_weight") != .3
            or recipe.get("objective_id") != training.OBJECTIVE_ID
            or type(recipe.get("micro_batch_size")) is not int or recipe["micro_batch_size"] < 2
            or recipe["micro_batch_size"] % 2 or recipe.get("source_sha256") != kernel.source_hashes()):
        raise ValueError("unchanged original SharedStateKernel recipe/source required")
    SequenceConfig(**result["config"])
    groups = recipe.get("optimizer")
    if (type(groups) is not list or len(groups) != 1 or type(groups[0]) is not dict
            or type(groups[0].get("lr")) is not float or groups[0]["lr"] != result["learning_rate"]):
        raise ValueError("one unchanged AdamW group with the declared rate required")
    return result


def checkpoint_identity(payload):
    """Extract JSON-stable metadata after the caller authenticates checkpoint bytes."""
    _keys(payload, ("schema", "launch_sha256", "arm", "step", "architecture", "learning_rate",
                    "auxiliary_weight", "learner", "weights_sha256", "runtime"), "rate-pilot checkpoint")
    result = {name: payload[name] for name in ("launch_sha256", "arm", "step", "architecture",
              "learning_rate", "auxiliary_weight", "runtime", "weights_sha256")}
    result.update(producing_schema=payload["schema"], config=payload["learner"]["recipe"]["config"],
                  recipe=payload["learner"]["recipe"], evidence=payload["learner"]["evidence"])
    return _identity(result)


def _metadata(saved):
    _keys(saved, ("schema", "recipe", "weights", "optimizer", "cursor", "evidence", "accounting"), "kernel snapshot")
    if saved["schema"] != kernel.SCHEMA:
        raise ValueError("original SharedStateKernel snapshot schema required")
    cursor, recipe, evidence = saved["cursor"], saved["recipe"], saved["evidence"]
    _count(cursor, "cursor")
    _keys(evidence, training._initial_evidence(), "evidence")
    if type(evidence["cursor"]) is not int or evidence["cursor"] != cursor:
        raise ValueError("committed cursor/evidence differ")
    for key in ("consumed_bundle_identity_sha256", "consumed_common_parent_identity_sha256"):
        if not _pin(evidence[key]):
            raise ValueError("consumed evidence digest required")
    _keys(evidence["exposures"], training.FAMILIES, "exposure families")
    for counts in evidence["exposures"].values():
        _keys(counts, training.COUNTS, "exposure counts")
        for name, count in counts.items():
            _count(count, name)
        if (counts["episodes"] != cursor*recipe["micro_batch_size"]
                or not 8*counts["episodes"] <= counts["turns"] <= 12*counts["episodes"]
                or counts["observation_tokens"] != counts["observation_bytes"]+2*counts["turns"]
                or counts["reply_target_tokens"] != counts["reply_target_bytes"]+counts["turns"]):
            raise ValueError("lifetime exposure arithmetic differs")
    if len({v["turns"] for v in evidence["exposures"].values()}) != 1:
        raise ValueError("family turn counts differ")
    accounting = saved["accounting"]
    _keys(accounting, ("work", "state_work", "cost"), "original accounting")
    expected = dict.fromkeys(training.WORK_COUNTS, 0)
    for key in ("step_invocations", "optimizer_attempts", "optimizer_returns", "synchronized_optimizer_updates", "retained_updates"):
        expected[key] = cursor
    for key in ("attempted_forwards", "completed_forwards", "attempted_objectives", "completed_objectives", "attempted_backwards", "completed_backwards"):
        expected[key] = 3*cursor
    for key in ("attempted_forward_episodes", "completed_forward_episodes", "attempted_backward_episodes", "completed_backward_episodes", "retained_episodes"):
        expected[key] = 3*cursor*recipe["micro_batch_size"]
    _keys(accounting["work"], expected, "work counts")
    _keys(accounting["state_work"], kernel.EXTRA_COUNTS, "state work counts")
    for group in (accounting["work"], accounting["state_work"]):
        for key, value in group.items():
            _count(value, key)
    if accounting["work"] != expected or accounting["state_work"] != dict.fromkeys(kernel.EXTRA_COUNTS, 3*cursor):
        raise ValueError("only complete successful lifetime accounting is resumable")
    _keys(accounting["cost"], ("step_wall_seconds", "step_cpu_seconds"), "kernel cost")
    for value in accounting["cost"].values():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative original costs required")


def _cost(value):
    _keys(value, COST_KEYS, "bridge costs")
    for key, number in value.items():
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ValueError("finite nonnegative bridge cost required")
        if key.endswith("invocations"):
            _count(number, key)


class SharedStateContinuation:
    def __init__(self):
        raise TypeError("use caller-pinned from_checkpoint or from_snapshot")

    @classmethod
    def from_checkpoint(cls, checkpoint_bytes, *, expected_sha256, expected_identity, device="cpu"):
        return cls._load(checkpoint_bytes, expected_sha256, expected_identity, device, original=True)

    @classmethod
    def from_snapshot(cls, snapshot_bytes, *, expected_sha256, expected_identity, device="cpu"):
        return cls._load(snapshot_bytes, expected_sha256, expected_identity, device, original=False)

    @classmethod
    def _load(cls, image, pin, expected_identity, device, *, original):
        started, cpu = time.monotonic(), time.process_time()
        report = dict(schema=SCHEMA, archive_sha256=pin, archive_kind=PRODUCER_SCHEMA if original else SCHEMA,
            failed=True, archive_load_attempts=0, archive_load_completions=0,
            model_construction_attempts=0, model_constructions=0, optimizer_construction_attempts=0,
            optimizer_constructions=0, optimizer_updates=0, forwards=0, backwards=0,
            queued_device_work_synchronized=False)
        try:
            sources, origin = source_hashes(), _identity(expected_identity)
            payload = _decode(image, pin, report)
            if original:
                if checkpoint_identity(payload) != origin:
                    raise ValueError("checkpoint differs from caller-authenticated identity")
                identity = dict(schema=SCHEMA, origin_archive_sha256=pin, origin=origin, source_sha256=sources, scope=SCOPE)
                costs = {key: 0 if key.endswith("invocations") else 0. for key in COST_KEYS}
            else:
                _keys(payload, ("schema", "identity", "learner", "weights_sha256", "lifetime_updates", "continuation_updates", "bridge_cost"), "continuation envelope")
                identity = payload["identity"]
                _keys(identity, ("schema", "origin_archive_sha256", "origin", "source_sha256", "scope"), "bridge identity")
                if (payload["schema"] != SCHEMA or identity["schema"] != SCHEMA or identity["scope"] != SCOPE
                        or not _pin(identity["origin_archive_sha256"]) or identity["origin"] != origin
                        or identity["source_sha256"] != sources or not _pin(payload["weights_sha256"])):
                    raise ValueError("continuation origin/source identity differs")
                for name in ("lifetime_updates", "continuation_updates"):
                    _count(payload[name], name)
                if (payload["lifetime_updates"] != payload["learner"]["cursor"]
                        or payload["continuation_updates"] != payload["lifetime_updates"]-origin["step"]):
                    raise ValueError("continuation lifetime differs")
                costs = deepcopy(payload["bridge_cost"]); _cost(costs)
                if costs["step_invocations"] != payload["continuation_updates"]:
                    raise ValueError("bridge steps differ from committed continuation")
            saved = payload["learner"]; _metadata(saved)
            if (_json_metadata(saved["recipe"]) != origin["recipe"] or saved["cursor"] < origin["step"]
                    or (original and (saved["cursor"] != origin["step"] or saved["evidence"] != origin["evidence"]))):
                raise ValueError("original recipe or committed evidence differs")
            if current_runtime(device) != origin["runtime"] or torch.get_default_dtype() != torch.float32:
                raise ValueError("unchanged producing runtime and default FP32 required")
            recipe = saved["recipe"]
            report["model_construction_attempts"] += 1
            # Seed is immaterial after strict full restoration, but construction
            # must preserve caller RNG and use the original model factory.
            model = student.build_shared_state_student(0, device=device, config=SequenceConfig(**recipe["config"]))
            report["model_constructions"] += 1
            report["optimizer_construction_attempts"] += 1
            optimizer = torch.optim.AdamW(model.parameters(), lr=origin["learning_rate"])
            report["optimizer_constructions"] += 1
            trainer = kernel.SharedStateKernel(model, optimizer, config=model.config, layout=recipe["layout"],
                micro_batch_size=recipe["micro_batch_size"], objective_id=training.OBJECTIVE_ID, auxiliary_weight=.3)
            if not _same(trainer.recipe, recipe):
                raise ValueError("fresh original kernel recipe differs")
            _tensor_state(saved, trainer)
            model.load_state_dict(saved["weights"], strict=True)
            optimizer.load_state_dict(_cpu_copy(saved["optimizer"]))
            trainer._evidence = deepcopy(saved["evidence"])
            trainer._work = deepcopy(saved["accounting"]["work"])
            trainer._state_work = deepcopy(saved["accounting"]["state_work"])
            trainer._cost = deepcopy(saved["accounting"]["cost"])
            trainer._sync(); report["queued_device_work_synchronized"] = True
            if not _same(trainer.snapshot(), saved) or checkpoint_digest(model) != payload["weights_sha256"]:
                raise ValueError("exact restored snapshot or full weight digest differs")
            if source_hashes() != sources or current_runtime(device) != origin["runtime"]:
                raise ValueError("sources or runtime changed during restore")
            result = object.__new__(cls)
            result._kernel, result._identity, result._sources = trainer, deepcopy(identity), sources
            result._bridge_cost, result._failed = costs, False
            result.last_step_report = result.last_snapshot_report = None
            report.update(failed=False, exact_state_restored=True, cursor=trainer.cursor, scope=SCOPE)
        except BaseException as error:
            report.update(error_type=type(error).__name__, error=str(error))
            try:
                if report["model_construction_attempts"] and torch.device(device).type == "cuda" and torch.cuda.is_initialized():
                    torch.cuda.synchronize(torch.device(device))
                report["queued_device_work_synchronized"] = True
            except BaseException as sync_error:
                report["failure_sync_error"] = repr(sync_error)
            report.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu)
            error.continuation_report = deepcopy(report)
            raise
        report.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu)
        result._bridge_cost["restore_invocations"] += 1
        result._bridge_cost["restore_wall_seconds"] += report["wall_seconds"]
        result._bridge_cost["restore_cpu_seconds"] += report["cpu_seconds"]
        result.last_restore_report = deepcopy(report)
        return result

    @property
    def model(self): return self._kernel.model
    @property
    def optimizer(self): return self._kernel.optimizer
    @property
    def cursor(self): return self._kernel.cursor
    @property
    def evidence(self): return self._kernel.evidence
    @property
    def recipe(self): return self._kernel.recipe
    @property
    def last_report(self): return deepcopy(self._kernel.last_report)
    @property
    def failed(self): return self._failed or self._kernel.failed
    @property
    def bridge_identity(self): return deepcopy(self._identity)
    @property
    def accounting(self):
        return dict(lifetime_kernel=self._kernel.accounting, bridge_cost=deepcopy(self._bridge_cost),
                    scope="Bridge timings include nested kernel calls; do not sum enclosing and nested time twice.")

    def _guard(self):
        if self.failed:
            raise RuntimeError("poisoned continuation cannot retry or publish")
        if source_hashes() != self._sources or current_runtime(self.model.tokens.weight.device) != self._identity["origin"]["runtime"]:
            raise ValueError("continuation sources or runtime changed")

    def step(self, prepared, *, state_targets, deadline=None):
        started, cpu = time.monotonic(), time.process_time()
        report = dict(schema=SCHEMA, failed=True, cursor_before=self.cursor, cursor=self.cursor,
                      kernel_called=False, physical_optimizer_updates=0)
        self.last_step_report = report
        try:
            self._guard(); report["kernel_called"] = True
            result = self._kernel.step(prepared, state_targets=state_targets, deadline=deadline)
            report.update(cursor=self.cursor, physical_optimizer_updates=result["physical_optimizer_updates"])
            self._guard(); report["failed"] = False
            return result
        except BaseException as error:
            self._failed = self._kernel.failed = True
            report.update(error_type=type(error).__name__, cursor=self.cursor)
            if report["kernel_called"]:
                report["physical_optimizer_updates"] = self._kernel.last_report["physical_optimizer_updates"]
            error.continuation_report = report
            raise
        finally:
            report.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu)
            self._bridge_cost["step_invocations"] += 1
            self._bridge_cost["step_wall_seconds"] += report["wall_seconds"]
            self._bridge_cost["step_cpu_seconds"] += report["cpu_seconds"]

    def snapshot(self):
        started, cpu = time.monotonic(), time.process_time()
        report = dict(schema=SCHEMA, failed=True, cursor=self.cursor)
        self.last_snapshot_report = report
        try:
            self._guard(); saved = self._kernel.snapshot(); weight_pin = checkpoint_digest(self.model)
            self._guard(); report["failed"] = False
        except BaseException as error:
            self._failed = self._kernel.failed = True
            report["error_type"] = type(error).__name__; error.continuation_report = report
            raise
        finally:
            report.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu)
            self._bridge_cost["snapshot_invocations"] += 1
            self._bridge_cost["snapshot_wall_seconds"] += report["wall_seconds"]
            self._bridge_cost["snapshot_cpu_seconds"] += report["cpu_seconds"]
        return dict(schema=SCHEMA, identity=self.bridge_identity, learner=saved, weights_sha256=weight_pin,
                    lifetime_updates=self.cursor, continuation_updates=self.cursor-self._identity["origin"]["step"],
                    bridge_cost=deepcopy(self._bridge_cost))
