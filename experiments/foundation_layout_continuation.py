"""Explicit optimizer continuation of a caller-authenticated layout snapshot.

The old recipe, including resume_supported=False, is preserved verbatim. This
new bridge supplies restoration and records its own source/provenance/cost. It
is not a CycleOwner token, a provider migration, historical replay or promotion.
The caller owns archive/launch authenticity, fresh-plan admission, exclusions,
prescribed order, deadlines and publication. Only global bundle IDs advance;
canonical parent-pair recipes are passed unchanged to the original trainer.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_student import _check_finite_tree, _cpu_copy
from experiments import foundation_layout_training as training
from experiments.sequence_student import SequenceConfig


SCHEMA = "bic-foundation-layout-continuation-v1"
STUDY_SCHEMA = "bic-foundation-layout-study-v1"
PILOT_SCHEMA = "bic-foundation-tutor-loop-v1"
MAX_ARCHIVE_BYTES = 1 << 30
SCOPE = ("Explicit new optimizer-continuation bridge; original research recipe preserved. "
         "Caller-pinned archive authenticates prior execution; invariants do not replay it. "
         "Caller owns admitted fresh lessons, order, exclusions, execution policy and publication. No automatic promotion or retry.")


def source_hashes():
    result = training.source_hashes()
    name = "experiments/foundation_layout_continuation.py"
    result[name] = hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
    return result


def _keys(value, names, label):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(label+" fields differ")


def _pin(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _count(value, label):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(label+" must be a nonnegative integer")


def _same(left, right):
    """Exact detached state comparison, including signed-zero tensor bytes."""
    if isinstance(left, torch.Tensor):
        if (not isinstance(right, torch.Tensor) or left.dtype != right.dtype
                or left.shape != right.shape or left.layout != right.layout
                or left.stride() != right.stride() or left.storage_offset() != right.storage_offset()
                or left.requires_grad != right.requires_grad or left.device != right.device):
            return False
        return torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                           right.detach().contiguous().reshape(-1).view(torch.uint8))
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_same(a,b) for a,b in zip(left,right))
    return left == right


def _decode(image, expected_sha256, report):
    if type(image) is not bytes or not 0 < len(image) <= MAX_ARCHIVE_BYTES or not _pin(expected_sha256):
        raise ValueError("bounded immutable archive bytes and caller SHA256 required")
    if hashlib.sha256(image).hexdigest() != expected_sha256:
        raise ValueError("archive differs from caller pin before decoding")
    report["archive_load_attempts"] += 1
    payload = torch.load(io.BytesIO(image), map_location="cpu", weights_only=True)
    report["archive_load_completions"] += 1
    return payload


def _identity(value):
    _keys(value, ("producing_schema", "launch_sha256", "arm", "step", "weights_sha256", "recipe", "evidence", "execution_profile"), "research identity")
    # The external identity crosses durable JSON receipts/specs. JSON arrays
    # have no tuple distinction; canonicalize metadata only, never learner state.
    result = json.loads(training._json(value))
    if not _pin(result["launch_sha256"]) or not _pin(result["weights_sha256"]):
        raise ValueError("launch and producing weights pins required")
    _count(result["step"], "study step")
    if (result["producing_schema"] not in (STUDY_SCHEMA,PILOT_SCHEMA)
            or result["arm"] not in training.curriculum.LAYOUTS or type(result["execution_profile"]) is not dict):
        raise ValueError("explicit layout and execution profile required")
    recipe = result["recipe"]
    if (type(recipe) is not dict or recipe.get("schema") != training.SCHEMA
            or recipe.get("source_sha256") != training.source_hashes()
            or recipe.get("layout") != result["arm"]
            or recipe.get("objective", {}).get("id") != training.OBJECTIVE_ID
            or recipe.get("torch_version") != str(torch.__version__)
            or recipe.get("resume_supported") is not False):
        raise ValueError("original layout recipe/source/objective/runtime differs")
    training._json(result)  # detached finite JSON identity; never a tensor recipe
    return result


def _metadata(payload):
    _keys(payload, ("schema", "recipe", "weights", "optimizer", "cursor", "evidence", "accounting"), "layout snapshot")
    if payload["schema"] != training.SCHEMA:
        raise ValueError("original layout snapshot schema required")
    cursor, recipe, evidence = payload["cursor"], payload["recipe"], payload["evidence"]
    _count(cursor, "global cursor")
    _keys(evidence, ("cursor", "consumed_bundle_identity_sha256", "consumed_common_parent_identity_sha256", "exposures"), "consumed evidence")
    if type(evidence["cursor"]) is not int or evidence["cursor"] != cursor:
        raise ValueError("evidence cursor differs")
    for key in ("consumed_bundle_identity_sha256", "consumed_common_parent_identity_sha256"):
        if not _pin(evidence[key]):
            raise ValueError("consumed identity digest required")
    _keys(evidence["exposures"], training.FAMILIES, "exposure families")
    for family, counts in evidence["exposures"].items():
        _keys(counts, training.COUNTS, "exposure counts")
        for key, value in counts.items():
            _count(value, key)
        if (counts["episodes"] != cursor*recipe["micro_batch_size"]
                or not 8*counts["episodes"] <= counts["turns"] <= 12*counts["episodes"]
                or counts["observation_tokens"] != counts["observation_bytes"]+2*counts["turns"]
                or counts["reply_target_tokens"] != counts["reply_target_bytes"]+counts["turns"]):
            raise ValueError("lifetime episode/turn/token counts disagree")
    if len({row["turns"] for row in evidence["exposures"].values()}) != 1:
        raise ValueError("three-family turn counts differ")
    if cursor == 0 and not _same(evidence, training._initial_evidence()):
        raise ValueError("zero-update evidence differs from canonical origin")
    accounting = payload["accounting"]
    _keys(accounting, ("work", "validation_work", "cost"), "lifetime accounting")
    _keys(accounting["work"], training.WORK_COUNTS, "trainer work")
    for key, value in accounting["work"].items():
        _count(value, key)
    expected = dict.fromkeys(training.WORK_COUNTS, 0)
    for key in ("step_invocations", "optimizer_attempts", "optimizer_returns", "synchronized_optimizer_updates", "retained_updates"):
        expected[key] = cursor
    for key in ("packed_microbatches", "attempted_forwards", "completed_forwards", "attempted_objectives", "completed_objectives", "attempted_backwards", "completed_backwards"):
        expected[key] = cursor*3
    for key in ("packed_episodes", "attempted_forward_episodes", "completed_forward_episodes", "attempted_backward_episodes", "completed_backward_episodes", "retained_episodes"):
        expected[key] = cursor*3*recipe["micro_batch_size"]
    if accounting["work"] != expected:
        raise ValueError("retained snapshot work differs from complete successful lifetime")
    _keys(accounting["validation_work"], training.curriculum.WorkLedger().report(), "validation work")
    for key, value in accounting["validation_work"].items():
        _count(value, key)
    _keys(accounting["cost"], ("setup_wall_seconds", "setup_cpu_seconds", "step_wall_seconds", "step_cpu_seconds",
        "retained_step_wall_seconds", "retained_step_cpu_seconds", "snapshot_invocations", "snapshot_wall_seconds", "snapshot_cpu_seconds"), "cost")
    for key, value in accounting["cost"].items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative lifetime cost required")
    _count(accounting["cost"]["snapshot_invocations"], "snapshot invocations")
    for kind in ("wall", "cpu"):
        if accounting["cost"][f"retained_step_{kind}_seconds"] > accounting["cost"][f"step_{kind}_seconds"]:
            raise ValueError("retained time exceeds total step time")


def _tensor_state(payload, trainer):
    _check_finite_tree(payload, "layout continuation snapshot")
    weights, expected = payload["weights"], trainer.model.state_dict()
    _keys(weights, expected, "weights")
    for name, value in weights.items():
        ref = expected[name]
        if (type(value) is not torch.Tensor or value.shape != ref.shape or value.dtype != ref.dtype
                or value.device.type != "cpu" or value.requires_grad or value.layout != torch.strided
                or value.stride() != ref.stride() or value.storage_offset() != 0):
            raise ValueError("weight shape/dtype/layout/CPU boundary differs")
        if payload["cursor"] == 0 and not _same(value, _cpu_copy(ref)):
            raise ValueError("zero-update weights differ from seeded initialization")
    aliases = {}
    for name, parameter in trainer.model.named_parameters(remove_duplicate=False):
        first = aliases.setdefault(id(parameter), name)
        if not _same(weights[name], weights[first]):
            raise ValueError("tied parameter aliases differ")
    optimizer = payload["optimizer"]
    _keys(optimizer, ("state", "param_groups"), "AdamW")
    if not _same(optimizer["param_groups"], trainer.optimizer.state_dict()["param_groups"]):
        raise ValueError("AdamW parameter groups differ")
    parameters, moments = list(trainer.model.parameters()), optimizer["state"]
    if (type(moments) is not dict or any(type(key) is not int for key in moments)
            or set(moments) != (set(range(len(parameters))) if payload["cursor"] else set())):
        raise ValueError("full AdamW moment coverage differs")
    for index, values in moments.items():
        _keys(values, ("step", "exp_avg", "exp_avg_sq"), "AdamW moments")
        step = values["step"]
        if (type(step) is not torch.Tensor or step.shape != torch.Size([]) or step.dtype != torch.float32
                or step.device.type != "cpu" or step.requires_grad or step.layout != torch.strided
                or float(step) != payload["cursor"]):
            raise ValueError("AdamW step differs from global lifetime cursor")
        for name in ("exp_avg", "exp_avg_sq"):
            value, parameter = values[name], parameters[index]
            if (type(value) is not torch.Tensor or value.shape != parameter.shape or value.dtype != parameter.dtype
                    or value.device.type != "cpu" or value.requires_grad or value.layout != torch.strided
                    or value.stride() != parameter.stride() or value.storage_offset() != 0):
                raise ValueError("AdamW moment shape/dtype/layout/CPU boundary differs")
        if bool(values["exp_avg_sq"].lt(0).any()):
            raise ValueError("negative AdamW second moment")


class LayoutContinuation:
    """Same trainer and fixed layout, behind an explicit new restore contract."""

    def __init__(self):
        raise TypeError("use caller-pinned from_study, from_pilot or from_snapshot")

    @classmethod
    def from_study(cls, archive_bytes, *, expected_sha256, expected_identity, device="cpu"):
        return cls._load(archive_bytes, expected_sha256, expected_identity, device, producing_schema=STUDY_SCHEMA)

    @classmethod
    def from_pilot(cls, archive_bytes, *, expected_sha256, expected_identity, device="cpu"):
        return cls._load(archive_bytes, expected_sha256, expected_identity, device, producing_schema=PILOT_SCHEMA)

    @classmethod
    def from_snapshot(cls, archive_bytes, *, expected_sha256, expected_identity, device="cpu"):
        return cls._load(archive_bytes, expected_sha256, expected_identity, device, producing_schema=None)

    @classmethod
    def _load(cls, image, pin, expected_identity, device, *, producing_schema):
        started, cpu_started = time.monotonic(), time.process_time()
        original = producing_schema is not None
        report = dict(schema=SCHEMA, archive_sha256=pin, failed=True, archive_kind=producing_schema or SCHEMA,
                      model_construction_attempts=0, model_constructions=0, optimizer_updates=0, forwards=0, backwards=0,
                      archive_load_attempts=0, archive_load_completions=0, queued_device_work_synchronized=False)
        try:
            sources = source_hashes()
            origin = _identity(expected_identity)
            payload = _decode(image, pin, report)
            if original:
                _keys(payload, ("schema", "arm", "step", "launch_sha256", "learner", "weights_sha256", "execution_profile"), "study archive")
                actual = {key:payload[key] for key in ("arm", "step", "launch_sha256", "weights_sha256", "execution_profile")}
                actual.update(producing_schema=payload["schema"],recipe=payload["learner"]["recipe"], evidence=payload["learner"]["evidence"])
                if payload["schema"] != producing_schema or training._json(actual) != training._json(origin):
                    raise ValueError("study archive differs from caller-authenticated identity")
                identity = dict(schema=SCHEMA, origin_archive_sha256=pin, origin=origin, source_sha256=sources, scope=SCOPE)
                bridge_cost = dict(restore_invocations=0, restore_wall_seconds=0., restore_cpu_seconds=0.,
                                   step_invocations=0, step_wall_seconds=0., step_cpu_seconds=0.,
                                   snapshot_invocations=0, snapshot_wall_seconds=0., snapshot_cpu_seconds=0.)
            else:
                _keys(payload, ("schema", "identity", "learner", "weights_sha256", "lifetime_updates", "continuation_updates", "bridge_cost"), "continuation archive")
                identity = payload["identity"]
                _keys(identity, ("schema", "origin_archive_sha256", "origin", "source_sha256", "scope"), "bridge identity")
                if (payload["schema"] != SCHEMA or identity["schema"] != SCHEMA or identity["scope"] != SCOPE
                        or not _pin(identity["origin_archive_sha256"]) or not _same(identity["origin"], origin)
                        or identity["source_sha256"] != sources):
                    raise ValueError("continuation origin or source identity differs")
                if (type(payload["lifetime_updates"]) is not int or type(payload["continuation_updates"]) is not int
                        or payload["lifetime_updates"] != payload["learner"]["cursor"]
                        or payload["continuation_updates"] != payload["lifetime_updates"]-origin["step"]):
                    raise ValueError("continuation lifetime counts differ")
                bridge_cost = deepcopy(payload["bridge_cost"])
                _keys(bridge_cost, ("restore_invocations", "restore_wall_seconds", "restore_cpu_seconds", "step_invocations",
                    "step_wall_seconds", "step_cpu_seconds", "snapshot_invocations", "snapshot_wall_seconds", "snapshot_cpu_seconds"), "bridge cost")
                for key, value in bridge_cost.items():
                    if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                        raise ValueError("invalid inherited bridge cost")
                    if key.endswith("invocations"):
                        _count(value, key)
            saved = payload["learner"]
            if training._json(saved["recipe"]) != training._json(origin["recipe"]) or saved["cursor"] < origin["step"]:
                raise ValueError("original recipe/layout or monotonic lifetime differs")
            if original and saved["cursor"] != origin["step"]:
                raise ValueError("original study cursor differs")
            _metadata(saved)
            recipe = saved["recipe"]
            report["model_construction_attempts"] = 1
            trainer = training.FoundationLayoutTrainer(seed=recipe["seed"], config=SequenceConfig(**recipe["config"]),
                learning_rate=recipe["learning_rate"], micro_batch_size=recipe["micro_batch_size"],
                layout=recipe["layout"], objective_id=training.OBJECTIVE_ID, device=device)
            report["model_constructions"] = 1
            report["template_setup_accounting"] = trainer.accounting
            report["current_runtime"] = dict(torch_version=str(torch.__version__), device=str(trainer._device),
                cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
            if not _same(trainer.recipe, recipe):
                raise ValueError("fresh same-trainer recipe differs from original")
            _tensor_state(saved, trainer)
            trainer.model.load_state_dict(saved["weights"], strict=True)
            trainer.optimizer.load_state_dict(_cpu_copy(saved["optimizer"]))
            trainer._evidence = deepcopy(saved["evidence"])
            trainer._work = deepcopy(saved["accounting"]["work"])
            trainer._validation_work = deepcopy(saved["accounting"]["validation_work"])
            trainer._cost = deepcopy(saved["accounting"]["cost"])
            trainer._sync()
            report["queued_device_work_synchronized"] = True
            actual = dict(schema=training.SCHEMA, recipe=trainer.recipe, weights=_cpu_copy(trainer.model.state_dict()),
                          optimizer=_cpu_copy(trainer.optimizer.state_dict()), cursor=trainer.cursor,
                          evidence=deepcopy(trainer._evidence), accounting=trainer.accounting)
            if not _same(actual, saved) or checkpoint_digest(trainer.model) != payload["weights_sha256"]:
                raise ValueError("exact restored weights/full AdamW/metadata or producer digest differs")
            if source_hashes() != sources:
                raise ValueError("bridge sources changed during restoration")
            result = object.__new__(cls)
            result._trainer, result._identity, result._sources = trainer, deepcopy(identity), sources
            result._bridge_cost, result._failed = bridge_cost, False
            result.last_report = result.last_snapshot_report = None
            report.update(failed=False, exact_state_restored=True, cursor=trainer.cursor,
                          original_resume_supported=recipe["resume_supported"], scope=SCOPE)
        except BaseException as error:
            report["error_type"] = type(error).__name__
            sync_started = time.monotonic()
            try:
                if report["model_construction_attempts"] and torch.device(device).type == "cuda" and torch.cuda.is_initialized():
                    torch.cuda.synchronize(torch.device(device))
                report["queued_device_work_synchronized"] = True
            except BaseException as sync_error:
                report["queued_device_work_synchronized"] = False
                report["failure_sync_error"] = f"{type(sync_error).__name__}: {sync_error}"
            report["failure_sync_wall_seconds_included_in_restore"] = time.monotonic()-sync_started
            report.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started)
            error.continuation_report = deepcopy(report)
            raise
        report.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started)
        result._bridge_cost["restore_invocations"] += 1
        result._bridge_cost["restore_wall_seconds"] += report["wall_seconds"]
        result._bridge_cost["restore_cpu_seconds"] += report["cpu_seconds"]
        result.last_restore_report = deepcopy(report)
        return result

    @property
    def model(self): return self._trainer.model
    @property
    def optimizer(self): return self._trainer.optimizer
    @property
    def cursor(self): return self._trainer.cursor
    @property
    def lifetime_updates(self): return self.cursor
    @property
    def updates(self): return self.cursor
    @property
    def failed(self): return self._failed or self._trainer.failed
    @property
    def recipe(self): return self._trainer.recipe
    @property
    def bridge_identity(self): return deepcopy(self._identity)
    @property
    def accounting(self):
        return dict(lifetime_trainer=self._trainer.accounting, bridge_cost=deepcopy(self._bridge_cost),
                    scope="Bridge wall/CPU includes nested trainer calls; never sum those nested times twice.")

    def _guard(self):
        if self.failed:
            raise RuntimeError("poisoned continuation cannot retry or publish")
        if source_hashes() != self._sources:
            raise ValueError("continuation source identity changed")

    def step(self, admitted_bundle):
        started, cpu_started = time.monotonic(), time.process_time()
        report = dict(schema=SCHEMA, failed=True, cursor_before=self.cursor, cursor=self.cursor,
                      trainer_report=None, physical_optimizer_updates=0, retained_updates=0,
                      trainer_committed_updates=0)
        self.last_report = report
        called, failure = False, None
        try:
            self._guard()
            called = True
            trainer_report = self._trainer.step(admitted_bundle)
            report.update(trainer_report=trainer_report, cursor=self.cursor,
                          physical_optimizer_updates=trainer_report["physical_optimizer_updates"],
                          trainer_committed_updates=trainer_report["retained_updates"])
            self._guard()
            report.update(failed=False, retained_updates=trainer_report["retained_updates"])
        except BaseException as error:
            failure = error
            self._failed = self._trainer._failed = True
            if called:
                report.update(trainer_report=deepcopy(self._trainer.last_report), cursor=self.cursor,
                    physical_optimizer_updates=self._trainer.last_report["physical_optimizer_updates"],
                    trainer_committed_updates=self._trainer.last_report["retained_updates"])
            report["retained_updates"] = 0
            report["error_type"] = type(error).__name__
            raise
        finally:
            report.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started)
            self._bridge_cost["step_invocations"] += 1
            self._bridge_cost["step_wall_seconds"] += report["wall_seconds"]
            self._bridge_cost["step_cpu_seconds"] += report["cpu_seconds"]
            if failure is not None:
                failure.continuation_report = deepcopy(report)
        return deepcopy(report)

    def snapshot(self):
        started, cpu_started = time.monotonic(), time.process_time()
        report = dict(schema=SCHEMA, failed=True, cursor=self.cursor)
        self.last_snapshot_report = report
        failure = None
        try:
            self._guard()
            saved = self._trainer.snapshot()
            weight_pin = checkpoint_digest(self.model)
            self._guard()
            report["failed"] = False
        except BaseException as error:
            failure = error
            self._failed = self._trainer._failed = True
            report["error_type"] = type(error).__name__
            raise
        finally:
            report.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started)
            self._bridge_cost["snapshot_invocations"] += 1
            self._bridge_cost["snapshot_wall_seconds"] += report["wall_seconds"]
            self._bridge_cost["snapshot_cpu_seconds"] += report["cpu_seconds"]
            if failure is not None:
                failure.continuation_report = deepcopy(report)
        return dict(schema=SCHEMA, identity=self.bridge_identity, learner=saved, weights_sha256=weight_pin,
                    lifetime_updates=self.cursor, continuation_updates=self.cursor-self._identity["origin"]["step"],
                    bridge_cost=deepcopy(self._bridge_cost))
