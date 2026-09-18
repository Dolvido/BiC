"""One unchanged three-family optimizer update over a prepared CPU bundle.

Accepts an already-created flat or recurrent-read model and its AdamW. Model
construction, admission, scheduling and restoration remain caller-owned. The
kernel never invokes a curriculum oracle or packs lessons. Any failed step
poisons the kernel; an uncertain optimizer outcome is never retried implicitly.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import time

import torch

from brain_in_computer.learning_student import _cpu_copy, _check_finite_tree
from experiments import foundation_layout_training as training
from experiments import foundation_layout_prepared as preparation
from experiments.sequence_student import SequenceConfig
from experiments.sequence_training import sequence_objective

SCHEMA = "bic-foundation-layout-kernel-v1"
ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    result = preparation.source_hashes()
    name = "experiments/foundation_layout_kernel.py"
    result[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


class PreparedLayoutKernel:
    def __init__(self, model, optimizer, *, config, layout, micro_batch_size,
                 objective_id, source_sha256=None, evidence=None):
        if (not isinstance(model, torch.nn.Module) or type(optimizer) is not torch.optim.AdamW
                or type(config) is not SequenceConfig or asdict(model.config) != asdict(config)
                or layout not in training.curriculum.LAYOUTS or type(micro_batch_size) is not int
                or micro_batch_size < 2 or micro_batch_size % 2 or objective_id != training.OBJECTIVE_ID):
            raise ValueError("explicit compatible model, AdamW, config and unchanged objective required")
        params = list(model.parameters())
        if (not params or len({parameter.device for parameter in params}) != 1
                or any(parameter.dtype != torch.float32 for parameter in params)
                or [id(p) for group in optimizer.param_groups for p in group["params"]] != [id(p) for p in params]):
            raise ValueError("one-device FP32 model and its exact ordered optimizer parameters required")
        self.model, self.optimizer, self._config = model, optimizer, config
        self._parameter_ids = [id(parameter) for parameter in params]
        self._device, self._layout, self._micro = params[0].device, layout, micro_batch_size
        if self._device.type not in ("cpu", "cuda"):
            raise ValueError("local CPU or CUDA required")
        self._sources = source_hashes()
        for name, pin in (source_sha256 or {}).items():
            path = (ROOT/name).resolve()
            if not path.is_relative_to(ROOT) or path == ROOT or hashlib.sha256(path.read_bytes()).hexdigest() != pin:
                raise ValueError("caller model source pin differs")
            if name in self._sources and self._sources[name] != pin:
                raise ValueError("caller source conflicts with the unchanged kernel")
            self._sources[name] = pin
        self._evidence = deepcopy(training._initial_evidence() if evidence is None else evidence)
        if (type(self._evidence.get("cursor")) is not int or self._evidence["cursor"] < 0
                or set(self._evidence) != set(training._initial_evidence())):
            raise ValueError("caller-restored consumed bundle evidence required")
        self._recipe = dict(schema=SCHEMA, config=asdict(config), layout=layout, micro_batch_size=micro_batch_size,
            objective_id=objective_id, family_order=list(training.FAMILIES), gradient_clip=1.,
            family_outer_weight=1/3, source_sha256=deepcopy(self._sources),
            optimizer=deepcopy(optimizer.state_dict()["param_groups"]),
            scope="Execution kernel only; caller owns original model and full optimizer restoration.")
        self.failed, self.last_report = False, None
        self._work = dict.fromkeys(training.WORK_COUNTS, 0)
        self._cost = dict(step_wall_seconds=0., step_cpu_seconds=0.)
        self._guard()

    @property
    def cursor(self): return self._evidence["cursor"]

    @property
    def evidence(self): return deepcopy(self._evidence)

    @property
    def recipe(self): return deepcopy(self._recipe)

    @property
    def accounting(self): return dict(work=deepcopy(self._work), cost=deepcopy(self._cost))

    def _sync(self):
        if self._device.type == "cuda": torch.cuda.synchronize(self._device)

    def _guard(self):
        if self.failed: raise RuntimeError("poisoned prepared kernel cannot retry")
        for name, pin in self._sources.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != pin:
                raise ValueError("kernel or model source changed")
        if (asdict(self.model.config) != self._recipe["config"]
                or [id(p) for p in self.model.parameters()] != self._parameter_ids
                or [id(p) for group in self.optimizer.param_groups for p in group["params"]] != self._parameter_ids
                or training._json(self.optimizer.state_dict()["param_groups"]) != training._json(self._recipe["optimizer"])):
            raise ValueError("model config or optimizer recipe changed")

    def step(self, prepared, *, deadline=None):
        started, cpu = time.monotonic(), time.process_time()
        batches, owner = {}, None
        report = dict(cursor_before=self.cursor, cursor=self.cursor, failed=True, failure_stage="preflight",
            physical_optimizer_updates=0, queued_device_work_synchronized=False,
            microbatches=[], canonical_regenerations=0, **dict.fromkeys(training.WORK_COUNTS, 0))
        report["step_invocations"] = 1; self.last_report = report
        try:
            self._guard(); self._sync()
            if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
                raise ValueError("finite absolute monotonic deadline required")
            if type(prepared) is not preparation.PreparedLayoutBundle or type(prepared._owner) is not preparation.PreparedLayoutOwner:
                raise ValueError("a live exact prepared owner is required")
            owner = prepared._owner
            report["failure_stage"] = "prepared_consumption"
            cpu_batches, evidence = owner._consume(prepared, cursor=self.cursor, config=self._config,
                layout=self._layout, micro_batch_size=self._micro)
            report["bundle_evidence"] = evidence
            report["failure_stage"] = "transfer"
            for family in training.FAMILIES:
                batches[family] = {section: {name: tensor.to(self._device) for name, tensor in fields.items()}
                    for section, fields in cpu_batches[family].items()}
            del cpu_batches
            self._guard()
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("caller allowance ended before first forward")
            self.model.train(); self.optimizer.zero_grad(set_to_none=True)
            for family in training.FAMILIES:
                batch = batches.pop(family)
                report["failure_stage"] = "forward:"+family
                report["attempted_forwards"] += 1; report["attempted_forward_episodes"] += self._micro
                output = self.model(**batch["inputs"], decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
                report["completed_forwards"] += 1; report["completed_forward_episodes"] += self._micro
                report["failure_stage"] = "objective:"+family; report["attempted_objectives"] += 1
                losses = sequence_objective(output, batch)
                if any(not bool(torch.isfinite(value)) for value in losses.values()):
                    raise ValueError("nonfinite unchanged sequence objective")
                report["completed_objectives"] += 1
                report["failure_stage"] = "backward:"+family
                report["attempted_backwards"] += 1; report["attempted_backward_episodes"] += self._micro
                (losses["loss"]/len(training.FAMILIES)).backward()
                report["completed_backwards"] += 1; report["completed_backward_episodes"] += self._micro
                report["microbatches"].append(dict(family=family, **{name: float(value.detach()) for name, value in losses.items()}))
                del output, losses, batch
            report["failure_stage"] = "gradient_clip"
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            report["failure_stage"] = "optimizer"; report["optimizer_attempts"] = 1
            report["physical_optimizer_updates"] = None
            self.optimizer.step(); report["optimizer_returns"] = 1
            self._sync(); report["queued_device_work_synchronized"] = True
            report["physical_optimizer_updates"] = report["synchronized_optimizer_updates"] = 1
            report["failure_stage"] = "commit"; self._guard()
            next_evidence = training._accumulate(self._evidence, evidence)
            for name in ("loss", "action_loss", "reply_loss", "observation_language_loss"):
                report[name] = sum(row[name] for row in report["microbatches"])/len(training.FAMILIES)
            report.update(failed=False, failure_stage=None, retained_updates=1,
                retained_episodes=3*self._micro, cursor=next_evidence["cursor"])
            self._evidence = next_evidence
        except BaseException as error:
            self.failed = True; report["error"] = repr(error)
            if report["physical_optimizer_updates"] is None: report["unknown_optimizer_outcomes"] = 1
            try:
                self._sync(); report["queued_device_work_synchronized"] = True
            except BaseException as sync_error:
                report["failure_sync_error"] = repr(sync_error)
            raise
        finally:
            batches.clear()
            report.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
                preparation=owner.report() if owner is not None else None)
            for name in training.WORK_COUNTS: self._work[name] += report[name]
            self._cost["step_wall_seconds"] += report["wall_seconds"]
            self._cost["step_cpu_seconds"] += report["cpu_seconds"]
        return deepcopy(report)

    def snapshot(self):
        self._guard(); self._sync()
        saved = dict(schema=SCHEMA, recipe=self.recipe, weights=_cpu_copy(self.model.state_dict()),
            optimizer=_cpu_copy(self.optimizer.state_dict()), cursor=self.cursor,
            evidence=self.evidence, accounting=self.accounting)
        _check_finite_tree(saved, "prepared kernel snapshot")
        self._guard()
        return saved
