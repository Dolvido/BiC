"""Original three-family update with optional training-only shared-state loss.

The caller computes canonical prefix targets once from authenticated bundle
text. Targets are supervision only and never model.forward arguments. At weight
zero the auxiliary graph is skipped exactly, preserving the original update.
No model construction, lesson generation, evaluation or checkpoint restoration
occurs here. A failed step permanently poisons this kernel.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time

import torch

from experiments import foundation_layout_training as training
from experiments import foundation_layout_prepared as preparation
from experiments.foundation_layout_kernel import PreparedLayoutKernel
from experiments.sequence_training import sequence_objective
from experiments import shared_state_student as student, shared_state_targets as targets

SCHEMA = "bic-shared-state-training-v1"
ROOT = Path(__file__).resolve().parents[1]
EXTRA_COUNTS = ("attempted_state_readouts", "completed_state_readouts", "attempted_state_objectives", "completed_state_objectives")


def source_hashes():
    from experiments.foundation_layout_kernel import source_hashes as base_sources
    result = base_sources()
    result.update(student.source_hashes())
    for name in ("experiments/shared_state_training.py", "experiments/shared_state_targets.py"):
        result[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


class SharedStateKernel(PreparedLayoutKernel):
    def __init__(self, model, optimizer, *, auxiliary_weight, **kwargs):
        if type(model) is not student.SharedStateStudent or type(auxiliary_weight) not in (int, float) or auxiliary_weight not in (0., .3):
            raise ValueError("exact shared-state model and fixed zero/.3 weight required")
        super().__init__(model, optimizer, **kwargs)
        self._auxiliary_weight = float(auxiliary_weight)
        self._sources.update(source_hashes())
        self._recipe.update(schema=SCHEMA, source_sha256=deepcopy(self._sources),
            architecture=student.ARCHITECTURE, auxiliary_weight=self._auxiliary_weight,
            auxiliary_objective="shared_state_targets.state_loss; family mean after original plus weighted state loss",
            zero_weight="Auxiliary graph skipped; unused head gradients remain None",
            state_supervision="Caller-authenticated causal prefix labels; never passed to model.forward")
        self._state_work = dict.fromkeys(EXTRA_COUNTS, 0)
        self._guard()

    def _guard(self):
        super()._guard()
        if hasattr(self, "_auxiliary_weight") and (type(self.model) is not student.SharedStateStudent
                or self._auxiliary_weight != self._recipe["auxiliary_weight"]):
            raise ValueError("shared-state architecture or objective changed")

    @property
    def accounting(self):
        result = super().accounting
        result["state_work"] = deepcopy(self._state_work)
        return result

    def snapshot(self):
        result = super().snapshot()
        result["schema"] = SCHEMA
        return result

    def step(self, prepared, *, state_targets, deadline=None):
        started, cpu = time.monotonic(), time.process_time()
        batches, labels, owner = {}, {}, None
        report = dict(cursor_before=self.cursor, cursor=self.cursor, failed=True, failure_stage="preflight",
            physical_optimizer_updates=0, queued_device_work_synchronized=False, microbatches=[],
            canonical_regenerations=0, auxiliary_weight=self._auxiliary_weight,
            **dict.fromkeys(training.WORK_COUNTS+EXTRA_COUNTS, 0))
        report["step_invocations"] = 1; self.last_report = report
        try:
            self._guard(); self._sync()
            if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
                raise ValueError("finite absolute monotonic deadline required")
            if (type(prepared) is not preparation.PreparedLayoutBundle
                    or type(prepared._owner) is not preparation.PreparedLayoutOwner
                    or type(state_targets) is not dict or set(state_targets) != set(training.FAMILIES)):
                raise ValueError("exact prepared bundle and all-family state supervision required")
            turns = prepared.evidence["turns"]
            for family in training.FAMILIES:
                value = state_targets[family]
                if (type(value) is not torch.Tensor or value.device.type != "cpu" or value.dtype != torch.long
                        or value.shape != (self._micro, turns, 12) or value.requires_grad
                        or not bool(((value >= 0) & (value < 107)).all())):
                    raise ValueError("detached CPU [episodes,turns,12] typed state targets required")
                labels[family] = value.detach().clone()
            report["state_target_sha256"] = {family: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
                                             for family, value in labels.items()}
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
                    raise ValueError("nonfinite original sequence objective")
                report["completed_objectives"] += 1
                state_loss = None
                total = losses["loss"]
                if self._auxiliary_weight:
                    report["failure_stage"] = "state_readout:"+family; report["attempted_state_readouts"] += 1
                    state_logits = self.model.state_logits(output["context_states"], batch["inputs"]["eos_positions"])
                    report["completed_state_readouts"] += 1
                    report["failure_stage"] = "state_objective:"+family; report["attempted_state_objectives"] += 1
                    state_loss = targets.state_loss(state_logits, labels[family])
                    if not bool(torch.isfinite(state_loss)):
                        raise ValueError("nonfinite causal-state objective")
                    report["completed_state_objectives"] += 1
                    total = total+self._auxiliary_weight*state_loss
                    del state_logits
                if not bool(torch.isfinite(total)):
                    raise ValueError("nonfinite total objective")
                report["failure_stage"] = "backward:"+family
                report["attempted_backwards"] += 1; report["attempted_backward_episodes"] += self._micro
                (total/len(training.FAMILIES)).backward()
                report["completed_backwards"] += 1; report["completed_backward_episodes"] += self._micro
                report["microbatches"].append(dict(family=family,
                    **{name: float(value.detach()) for name, value in losses.items()},
                    state_loss=None if state_loss is None else float(state_loss.detach()),
                    total_loss=float(total.detach())))
                del output, losses, total, state_loss, batch
            report["failure_stage"] = "gradient_clip"
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            report["failure_stage"] = "optimizer"; report["optimizer_attempts"] = 1
            report["physical_optimizer_updates"] = None
            self.optimizer.step(); report["optimizer_returns"] = 1
            self._sync(); report["queued_device_work_synchronized"] = True
            report["physical_optimizer_updates"] = report["synchronized_optimizer_updates"] = 1
            report["failure_stage"] = "commit"; self._guard()
            next_evidence = training._accumulate(self._evidence, evidence)
            for name in ("loss", "action_loss", "reply_loss", "observation_language_loss", "total_loss"):
                report[name] = sum(row[name] for row in report["microbatches"])/len(training.FAMILIES)
            report["state_loss"] = None if not self._auxiliary_weight else sum(row["state_loss"] for row in report["microbatches"])/3
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
            batches.clear(); labels.clear()
            report.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
                preparation=owner.report() if owner is not None else None)
            for name in training.WORK_COUNTS: self._work[name] += report[name]
            for name in EXTRA_COUNTS: self._state_work[name] += report[name]
            self._cost["step_wall_seconds"] += report["wall_seconds"]
            self._cost["step_cpu_seconds"] += report["cpu_seconds"]
        return deepcopy(report)
