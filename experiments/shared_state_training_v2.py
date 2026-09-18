"""Explicit v2 packed-image boundary; unchanged shared-state update arithmetic."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time
import torch

from brain_in_computer.learning_student import _cpu_copy
from experiments import foundation_layout_training as training
from experiments import foundation_layout_prepared_v2 as preparation
from experiments import shared_state_training as v1, shared_state_continuation as old_bridge
from experiments import shared_state_targets as targets
from experiments.sequence_training import sequence_objective
from experiments.foundation_layout_continuation import _same, _tensor_state

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="bic-shared-state-training-v2"
EXTRA_COUNTS=v1.EXTRA_COUNTS


def source_hashes():
    return {**old_bridge.source_hashes(), **preparation.source_hashes(),
        "experiments/shared_state_training_v2.py":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class SharedStateKernel(v1.SharedStateKernel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._sources.update(source_hashes())
        self._recipe.update(schema=SCHEMA,source_sha256=deepcopy(self._sources),
            preparation=preparation.SCHEMA)
        self._guard()

    def snapshot(self):
        value=super().snapshot();value["schema"]=SCHEMA
        return value

    def restore_state(self,saved,*,migrate_v1=False):
        """Explicit complete-state load; caller already authenticated archive bytes.

        V1 metadata arithmetic is reused only as a validation projection. The
        published execution recipe remains v2 and never impersonates old sources.
        """
        try:
            expected_schema=v1.SCHEMA if migrate_v1 else SCHEMA
            if saved["schema"]!=expected_schema or saved["recipe"]["schema"]!=expected_schema:
                raise ValueError("explicit saved-state version required")
            projected=dict(saved,schema=v1.SCHEMA)
            old_bridge._metadata(projected)
            recipe=deepcopy(saved["recipe"])
            if migrate_v1:
                if recipe["source_sha256"]!=v1.source_hashes():
                    raise ValueError("original unchanged v1 recipe source required")
                recipe.update(schema=SCHEMA,source_sha256=deepcopy(self._sources),preparation=preparation.SCHEMA)
            if not _same(recipe,self.recipe):
                raise ValueError("optimizer/objective/configuration or explicit source transition differs")
            _tensor_state(saved,self)
            self.model.load_state_dict(saved["weights"],strict=True)
            self.optimizer.load_state_dict(_cpu_copy(saved["optimizer"]))
            self._evidence=deepcopy(saved["evidence"])
            self._work=deepcopy(saved["accounting"]["work"])
            self._state_work=deepcopy(saved["accounting"]["state_work"])
            self._cost=deepcopy(saved["accounting"]["cost"])
            self._sync();actual=self.snapshot()
            for key in ("weights","optimizer","cursor","evidence","accounting"):
                if not _same(actual[key],saved[key]):raise ValueError("complete migrated state differs: "+key)
            self._guard()
        except BaseException:
            self.failed=True;raise

    def step(self, prepared, *, deadline=None):
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
                    or type(prepared.owner) is not preparation.PreparedLayoutOwner):
                raise ValueError("exact v2 public owner and token required")
            owner = prepared.owner
            report["failure_stage"] = "prepared_consumption"
            cpu_batches, evidence, labels = owner.consume(prepared, cursor=self.cursor, config=self._config,
                layout=self._layout, micro_batch_size=self._micro)
            report["state_target_sha256"] = {family: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
                                             for family, value in labels.items()}
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
