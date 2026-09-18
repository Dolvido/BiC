"""Authenticated explicit v1-continuation to v2-packed execution transition.

The old archive/recipe/history remain recorded separately. New execution names
new sources. No curriculum, teacher or promotion decision is made by this bridge.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments import shared_state_continuation as old
from experiments import shared_state_training_v2 as kernel
from experiments import shared_state_student as student, foundation_layout_training as training
from experiments.foundation_layout_continuation import _decode, _keys, _pin, _same
from experiments.sequence_student import SequenceConfig

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="bic-shared-state-packed-continuation-v2"
TRANSITION_SCHEMA="bic-shared-state-packed-migration-v2"
COST_KEYS=tuple(f"{kind}_{suffix}" for kind in ("migration","restore","step","snapshot")
                for suffix in ("invocations","wall_seconds","cpu_seconds"))


def source_hashes():
    return {**kernel.source_hashes(),
        "experiments/shared_state_packed_migration.py":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _cost(value):
    _keys(value,COST_KEYS,"packed bridge costs")
    for key,number in value.items():
        if (type(number) not in (int,float) or not math.isfinite(number) or number<0
                or (key.endswith("invocations") and type(number) is not int)):
            raise ValueError("finite nonnegative physical costs required")


class PackedSharedStateContinuation:
    def __init__(self): raise TypeError("use from_continuation or from_snapshot")

    @classmethod
    def from_continuation(cls,raw,*,expected_sha256,expected_identity,device="cpu"):
        started,cpu=time.monotonic(),time.process_time()
        report=dict(schema=SCHEMA,operation="migration",failed=True,model_constructions=0,
                    archive_load_attempts=0,archive_load_completions=0,
                    optimizer_updates=0,forwards=0,backwards=0)
        legacy=None
        try:
            sources=source_hashes()
            # One explicit metadata decode retains exact prior costs. The old
            # public restorer performs its own separately reported decode.
            original=_decode(raw,expected_sha256,report)
            legacy=old.SharedStateContinuation.from_snapshot(raw,expected_sha256=expected_sha256,
                expected_identity=expected_identity,device=device)
            report["old_restore"]=deepcopy(legacy.last_restore_report)
            report["model_constructions"]=legacy.last_restore_report["model_constructions"]
            previous=legacy.snapshot()
            if not _same(original["identity"],previous["identity"]):
                raise ValueError("old snapshot identity differs after authenticated restoration")
            recipe=previous["learner"]["recipe"]
            trainer=kernel.SharedStateKernel(legacy.model,legacy.optimizer,config=legacy.model.config,
                layout=recipe["layout"],micro_batch_size=recipe["micro_batch_size"],
                objective_id=training.OBJECTIVE_ID,auxiliary_weight=recipe["auxiliary_weight"])
            trainer.restore_state(previous["learner"],migrate_v1=True)
            if checkpoint_digest(trainer.model)!=previous["weights_sha256"]:
                raise ValueError("migrated full weights differ")
            transition=dict(schema=TRANSITION_SCHEMA,old_archive_sha256=expected_sha256,
                old_bridge_identity=previous["identity"],old_recipe=deepcopy(recipe),
                old_bridge_cost=deepcopy(original["bridge_cost"]),migration_cursor=trainer.cursor,
                migration_evidence=trainer.evidence,migration_weights_sha256=previous["weights_sha256"],
                new_recipe=trainer.recipe,new_source_sha256=sources,
                scope="Old bridge cost is the exact prior archive value. Newly incurred old-restorer/snapshot work is inside migration cost; do not add nested restore time again. Old recipe is historical; new recipe owns future execution.")
            value=object.__new__(cls)
            value._kernel,value._transition,value._sources=trainer,transition,sources
            value._transition_pin=training._hash(transition)
            value._cost={key:0 if key.endswith("invocations") else 0. for key in COST_KEYS}
            value._failed=False;value.last_step_report=value.last_snapshot_report=None
            value._guard();report.update(failed=False,exact_state_preserved=True,cursor=value.cursor)
        except BaseException as error:
            if legacy is None and hasattr(error,"continuation_report"):
                report["old_restore"]=deepcopy(error.continuation_report)
                report["model_constructions"]=error.continuation_report.get("model_constructions",0)
            report.update(error=repr(error),wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
            error.migration_report=deepcopy(report);raise
        report.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
        value._record_cost("migration",report);value.last_restore_report=deepcopy(report)
        return value

    @classmethod
    def from_snapshot(cls,raw,*,expected_sha256,expected_transition_sha256,device="cpu"):
        started,cpu=time.monotonic(),time.process_time()
        report=dict(schema=SCHEMA,operation="restore",failed=True,archive_load_attempts=0,
            archive_load_completions=0,model_construction_attempts=0,model_constructions=0,
            optimizer_constructions=0,optimizer_updates=0,forwards=0,backwards=0)
        try:
            sources=source_hashes();payload=_decode(raw,expected_sha256,report)
            _keys(payload,("schema","transition","transition_sha256","learner","weights_sha256",
                           "lifetime_updates","packed_updates","cost"),"packed continuation snapshot")
            transition=payload["transition"]
            _keys(transition,("schema","old_archive_sha256","old_bridge_identity","old_recipe","old_bridge_cost",
                "migration_cursor","migration_evidence","migration_weights_sha256","new_recipe","new_source_sha256","scope"),"migration transition")
            if (payload["schema"]!=SCHEMA or not _pin(expected_transition_sha256)
                    or payload["transition_sha256"]!=expected_transition_sha256
                    or training._hash(transition)!=expected_transition_sha256
                    or transition["schema"]!=TRANSITION_SCHEMA or transition["new_source_sha256"]!=sources
                    or not _pin(transition["old_archive_sha256"])):
                raise ValueError("caller-pinned old/new source transition required")
            bridge=transition["old_bridge_identity"]
            _keys(bridge,("schema","origin_archive_sha256","origin","source_sha256","scope"),"old bridge identity")
            origin=old._identity(bridge["origin"])
            if (bridge["schema"]!=old.SCHEMA or bridge["source_sha256"]!=old.source_hashes()
                    or bridge["scope"]!=old.SCOPE or not _pin(bridge["origin_archive_sha256"])
                    or old._json_metadata(transition["old_recipe"])!=origin["recipe"]
                    or old.current_runtime(device)!=origin["runtime"] or torch.get_default_dtype()!=torch.float32):
                raise ValueError("original learning identity/runtime changed")
            old._cost(transition["old_bridge_cost"]);_cost(payload["cost"])
            saved=payload["learner"];cursor=saved["cursor"];start=transition["migration_cursor"]
            if (type(start) is not int or start<origin["step"] or transition["migration_evidence"]["cursor"]!=start
                    or type(cursor) is not int or cursor<start or payload["lifetime_updates"]!=cursor
                    or payload["packed_updates"]!=cursor-start or payload["cost"]["step_invocations"]!=cursor-start
                    or transition["old_bridge_cost"]["step_invocations"]!=start-origin["step"]
                    or not _same(saved["recipe"],transition["new_recipe"])):
                raise ValueError("explicit migration boundary or lifetime counters differ")
            report["model_construction_attempts"]+=1
            model=student.build_shared_state_student(0,device=device,config=SequenceConfig(**saved["recipe"]["config"]))
            report["model_constructions"]+=1
            optimizer=torch.optim.AdamW(model.parameters(),lr=origin["learning_rate"])
            report["optimizer_constructions"]+=1
            trainer=kernel.SharedStateKernel(model,optimizer,config=model.config,layout=saved["recipe"]["layout"],
                micro_batch_size=saved["recipe"]["micro_batch_size"],objective_id=training.OBJECTIVE_ID,
                auxiliary_weight=saved["recipe"]["auxiliary_weight"])
            trainer.restore_state(saved)
            if checkpoint_digest(model)!=payload["weights_sha256"]:raise ValueError("restored full weights differ")
            value=object.__new__(cls)
            value._kernel,value._transition,value._sources=trainer,deepcopy(transition),sources
            value._transition_pin=expected_transition_sha256;value._cost=deepcopy(payload["cost"])
            value._failed=False;value.last_step_report=value.last_snapshot_report=None
            value._guard();report.update(failed=False,exact_state_restored=True,cursor=cursor)
        except BaseException as error:
            report.update(error=repr(error),wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
            error.migration_report=deepcopy(report);raise
        report.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
        value._record_cost("restore",report);value.last_restore_report=deepcopy(report)
        return value

    def _record_cost(self,name,report):
        self._cost[name+"_invocations"]+=1
        for kind in ("wall","cpu"):
            self._cost[name+"_"+kind+"_seconds"]+=report[kind+"_seconds"]

    @property
    def model(self):return self._kernel.model
    @property
    def optimizer(self):return self._kernel.optimizer
    @property
    def cursor(self):return self._kernel.cursor
    @property
    def evidence(self):return self._kernel.evidence
    @property
    def recipe(self):return self._kernel.recipe
    @property
    def transition(self):return deepcopy(self._transition)
    @property
    def transition_sha256(self):return self._transition_pin
    @property
    def last_report(self):return deepcopy(self._kernel.last_report)
    @property
    def failed(self):return self._failed or self._kernel.failed
    @property
    def accounting(self):
        return dict(lifetime_kernel=self._kernel.accounting,packed_bridge_cost=deepcopy(self._cost),
            previous_bridge_cost=deepcopy(self._transition["old_bridge_cost"]),
            scope="Nested kernel and old restore/snapshot costs overlap outer bridge costs; preparation is separate.")

    def _guard(self):
        if self.failed:raise RuntimeError("poisoned packed continuation cannot continue")
        if (source_hashes()!=self._sources or training._hash(self._transition)!=self._transition_pin
                or old.current_runtime(self.model.tokens.weight.device)!=self._transition["old_bridge_identity"]["origin"]["runtime"]):
            raise ValueError("packed migration source/identity/runtime changed")

    def step(self,token,*,deadline=None):
        started,cpu=time.monotonic(),time.process_time()
        report=dict(schema=SCHEMA,failed=True,kernel_called=False,cursor_before=self.cursor,
                    cursor=self.cursor,physical_optimizer_updates=0)
        self.last_step_report=report
        try:
            self._guard();report["kernel_called"]=True
            result=self._kernel.step(token,deadline=deadline)
            report.update(cursor=self.cursor,physical_optimizer_updates=result["physical_optimizer_updates"])
            self._guard();report["failed"]=False
            return result
        except BaseException as error:
            self._failed=self._kernel.failed=True
            report.update(error=repr(error),cursor=self.cursor)
            if report["kernel_called"]:report["physical_optimizer_updates"]=self._kernel.last_report["physical_optimizer_updates"]
            error.migration_report=deepcopy(report);raise
        finally:
            report.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
            self._record_cost("step",report)

    def snapshot(self):
        started,cpu=time.monotonic(),time.process_time()
        report=dict(schema=SCHEMA,failed=True,cursor=self.cursor);self.last_snapshot_report=report
        try:
            self._guard();saved=self._kernel.snapshot();pin=checkpoint_digest(self.model)
            self._guard();report["failed"]=False
        except BaseException as error:
            self._failed=self._kernel.failed=True;report["error"]=repr(error)
            error.migration_report=deepcopy(report);raise
        finally:
            report.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
            self._record_cost("snapshot",report)
        return dict(schema=SCHEMA,transition=self.transition,transition_sha256=self._transition_pin,
            learner=saved,weights_sha256=pin,lifetime_updates=self.cursor,
            packed_updates=self.cursor-self._transition["migration_cursor"],cost=deepcopy(self._cost))
