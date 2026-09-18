"""Versioned objective training with bounded, transient CPU preparation.

The learner arithmetic remains three family forwards/backwards, each loss/3,
then clip1 and one AdamW step. CPU integer/Boolean tensors transfer with ordinary
.to(device), in family order. A fresh finite preparation allowance is checked
after acquisition/validation/first-family transfer immediately before the first
forward. Once started, that complete optimizer update is nonpreemptive.

Snapshots drain preparation and never serialize a queue. Explicit restores
accept only this schema/recipe, drain old preparation, then reuse the existing
strict tensor/AdamW validator through a private envelope-only adapter. This is
not an old-checkpoint migration interface. A restored learner has no live owner;
its caller must supply another explicit finite preparation invocation.

Consumer timing includes waits/transfers; preparation CPU/wall lives in separate
invocation reports. Overlapping wall intervals must never be added as elapsed
time. No execution-equivalence, speedup or adoption claim follows from this file.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import time

import torch

from brain_in_computer.learning_student import _check_finite_tree
from experiments import foundation_objective_training as synchronous
from experiments import foundation_prepared_bundles as preparation
from experiments import foundation_training as canonical
from experiments.foundation_curriculum import FAMILIES


SCHEMA = "bic-foundation-prefetched-objective-trainer-v1"
TIMING_SCHEMA = "bic-foundation-prefetched-consumer-timing-v1"
FIELDS = {"schema", "recipe", "weights", "optimizer", "cursor", "evidence", "timing"}


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    name = "experiments/foundation_prefetched_training.py"
    return {**synchronous.source_hashes(), **preparation.source_hashes(),
            name: hashlib.sha256((root/name).read_bytes()).hexdigest()}


def _allowance(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("explicit positive finite preparation allowance required")
    return float(value)


class PrefetchedObjectiveFoundationTrainer(synchronous.ObjectiveFoundationTrainer):
    """One learner plus at most one bounded preparation owner; no automatic retry."""
    def __init__(self, plan, order="curriculum", *, objective_id, seed,
                 admission_protected_transcripts, protected_transcripts, admission_receipt,
                 plan_index, config=None, learning_rate=.001, device="cpu", payload=None):
        started = time.monotonic()
        self._prefetch_sources = source_hashes()
        self._admission_history = canonical._protected(admission_protected_transcripts)
        self._admission_receipt = copy.deepcopy(admission_receipt)
        self._preparation = None
        self._preparation_deadline = None
        self._preparation_stop = None
        self._preparation_invocations = 0
        self._preparation_invocation = None
        self.last_preparation_report = None
        super().__init__(plan, order, objective_id=objective_id, seed=seed,
            admission_protected_transcripts=self._admission_history,
            protected_transcripts=protected_transcripts, admission_receipt=self._admission_receipt,
            plan_index=plan_index, config=config, learning_rate=learning_rate, device=device, payload=None)
        # Leave parent's _sources untouched: its strict guard still authenticates
        # the original objective closure during construction and every operation.
        self._recipe.update(schema=SCHEMA, source_sha256=copy.deepcopy(self._prefetch_sources),
            preparation=dict(schema=preparation.SCHEMA, policy="one CPU producer; current lease plus one next bundle",
                family_order=list(FAMILIES), transfer="ordinary tensor.to(device), one family at a time",
                update_boundary="finite caller allowance before first forward; complete three-family update is nonpreemptive",
                checkpoint_boundary="close and join transient preparation; no queue or invocation allowance in checkpoint"),
            timing_contract=dict(schema=TIMING_SCHEMA,
                retained_step_seconds="consumer acquisition, validation, transfers, all neural/optimizer work and post-update guards",
                materialization_seconds_included_in_step="always zero; CPU preparation is separate invocation work",
                preparation="cumulative owner reports, never attributed to a consumed step or added to overlapping wall time"))
        self._timing.update(consumer_acquisition_seconds_included_in_step=0.,
                            consumer_transfer_seconds_included_in_step=0.)
        self._assert_sources()
        if payload is not None:
            self.restore(payload)
        self.setup_seconds = time.monotonic()-started

    def _assert_sources(self):
        super()._assert_sources()
        if source_hashes() != self._prefetch_sources:
            raise ValueError("prefetched training source identity changed")

    @property
    def preparation_report(self):
        """Current or last invocation only; caller owns durable receipt retention."""
        if self._preparation is None:
            return copy.deepcopy(self.last_preparation_report)
        return dict(invocation=copy.deepcopy(self._preparation_invocation), owner=self._preparation.report(),
                    scope="Preparation/storage work, not consumed lessons or optimizer updates.")

    def start_preparation(self, *, stop_cursor, max_seconds):
        """Explicit finite CPU preparation window beginning at committed cursor."""
        if self._failed:
            raise RuntimeError("failed prefetched step requires explicit restoration")
        self._assert_sources()
        if self._preparation is not None:
            raise RuntimeError("close and join the existing preparation owner before starting another")
        allowance = _allowance(max_seconds)
        if (type(stop_cursor) is not int
                or not self.cursor <= stop_cursor <= len(self._plan["schedules"][self._order])):
            raise ValueError("explicit bounded preparation stop cursor required")
        started = time.monotonic()
        self._preparation_deadline = started+allowance
        self._preparation_stop = stop_cursor
        self._preparation_invocations += 1
        self._preparation_invocation = dict(number=self._preparation_invocations, start_cursor=self.cursor,
            stop_cursor=stop_cursor, max_seconds=allowance, started_monotonic=started,
            deadline_monotonic=self._preparation_deadline)
        try:
            self._preparation = preparation.PreparedBundleOwner(self._plan, self._order,
                plan_index=self._index, config=self._config,
                admission_protected_transcripts=self._admission_history,
                protected_transcripts=self._protected, admission_receipt=self._admission_receipt,
                start_cursor=self.cursor, stop_cursor=stop_cursor, max_seconds=allowance)
            self._assert_sources()
            return self.preparation_report
        except BaseException as error:
            if self._preparation is not None:
                try:
                    self.close_preparation()
                except BaseException as cleanup_error:
                    self.last_preparation_report = dict(invocation=copy.deepcopy(self._preparation_invocation),
                        owner=self._preparation.report(), error=type(error).__name__ + ": " + str(error),
                        cleanup_error=repr(cleanup_error), constructor_wall_seconds=time.monotonic()-started,
                        scope="Owner startup failed and cleanup also failed; owner remains attached and work is not assumed stopped.")
            else:
                self.last_preparation_report = dict(invocation=copy.deepcopy(self._preparation_invocation),
                    owner=None, error=type(error).__name__ + ": " + str(error),
                    constructor_wall_seconds=time.monotonic()-started,
                    scope="Failed owner construction; no learner update. Partial CPU setup work may be unknown.")
            raise

    def close_preparation(self, *, join_seconds=1.):
        """Stop and bound the join; retain an unjoined owner for explicit cleanup."""
        if self._preparation is None:
            return copy.deepcopy(self.last_preparation_report)
        report = self._preparation.close(join_seconds=join_seconds)
        self.last_preparation_report = dict(invocation=copy.deepcopy(self._preparation_invocation), owner=report,
            scope="Preparation/storage work only; release is not a learner commit. Unjoined work remains incomplete.")
        if not report["worker_alive"]:
            self._preparation = None
        return copy.deepcopy(self.last_preparation_report)

    def _drain(self):
        self.close_preparation()
        if self._preparation is not None:
            raise RuntimeError("previous CPU producer is unjoined; checkpoint/restore cannot proceed")

    def snapshot(self):
        if self._failed:
            raise RuntimeError("failed prefetched step requires explicit restoration before snapshot")
        self._assert_sources()
        self._drain()
        value = super().snapshot()
        value["schema"] = SCHEMA
        return value

    def restore(self, payload):
        """Validate only a new-schema image; queue drain cannot change learner state."""
        called = time.monotonic()
        self._assert_sources()
        if type(payload) is not dict or set(payload) != FIELDS or payload["schema"] != SCHEMA:
            raise ValueError("exact prefetched checkpoint schema required; historical images are not migrated")
        saved = copy.deepcopy(payload)
        if canonical._json(saved["recipe"]) != canonical._json(self._recipe):
            raise ValueError("prefetched objective/source/index/config/protection recipe differs")
        _check_finite_tree(saved, "prefetched checkpoint")
        timing = saved["timing"]
        if (type(timing) is not dict or set(timing) != set(self._timing)
                or type(timing["materialization_seconds_included_in_step"]) not in (int, float)
                or timing["materialization_seconds_included_in_step"] != 0):
            raise ValueError("prefetched timing contract requires separate CPU preparation accounting")
        acquisition = timing["consumer_acquisition_seconds_included_in_step"]
        transfer = timing["consumer_transfer_seconds_included_in_step"]
        retained = timing["retained_step_seconds"]
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (acquisition, transfer, retained))
                or acquisition+transfer > retained+1e-9):
            raise ValueError("consumer acquisition/transfer timing exceeds retained consumer step time")
        self._drain()
        # Private validator adapter, not migration: caller schema and full recipe
        # were checked above; recipe retains THIS schema/source/preparation contract.
        saved["schema"] = synchronous.SCHEMA
        super().restore(saved)
        self.last_restore_seconds = time.monotonic()-called
        return self

    def _before_first_forward(self):
        if self._preparation is None or self._preparation_deadline is None:
            raise RuntimeError("explicit live bounded preparation owner required")
        if time.monotonic() >= self._preparation_deadline:
            raise TimeoutError("caller preparation/update allowance elapsed before first forward")
        status = self._preparation.report()
        if status["closed"] or status["stopping"] or status["error"] is not None:
            raise RuntimeError("preparation stopped before the next optimizer update")
        if self.cursor >= self._preparation_stop:
            raise StopIteration("authorized preparation window complete")

    def step(self):
        called, cursor_before = time.monotonic(), self.cursor
        self.last_report = dict(cursor=self.cursor, failed=True, failure_stage="preflight",
            physical_optimizer_updates=0, retained_optimizer_updates=0,
            drawn_microbatches=0, drawn_episode_exposures=0,
            neural_attempted_microbatches=0, neural_attempted_episode_exposures=0,
            completed_microbatches=0, completed_microbatch_episode_exposures=0,
            step_seconds=0., wall_seconds=0., materialization_seconds=0.,
            preparation_is_separate=True, timing_schema=TIMING_SCHEMA)
        if self._failed:
            self.last_report.update(failure_stage="preflight:restore_required", wall_seconds=time.monotonic()-called)
            raise RuntimeError("failed prefetched step requires explicit restoration")
        if self.cursor == len(self._plan["schedules"][self._order]):
            self.last_report.update(failed=False, complete=True, failure_stage=None)
            raise StopIteration("foundation plan is complete")
        # No learner mutation before this preflight. All attempted work has its
        # own fresh report, including absence/exhaustion of a preparation window.
        try:
            self._assert_sources()
            self._before_first_forward()
            self._sync()
        except BaseException:
            self.last_report["wall_seconds"] = time.monotonic()-called
            if self._preparation is not None:
                try:
                    self.close_preparation()
                except BaseException as error:
                    self.last_report["cleanup_error"] = repr(error)
            self.last_report["preparation"] = self.preparation_report
            self.last_report["wall_seconds"] = time.monotonic()-called
            raise
        started = time.monotonic()
        item, batches, evidence = None, None, None
        reports, stage, physical = [], "acquisition", 0
        neural_microbatches = neural_episodes = 0
        acquisition_seconds = transfer_seconds = 0.
        update_started = False
        try:
            tick = time.monotonic()
            item = self._preparation.take(self.cursor, order=self._order, plan_index=self._index,
                config=self._config, protected_transcripts=self._protected,
                timeout_seconds=_allowance(self._preparation_deadline-time.monotonic()))
            batches, evidence = item.batches, item.evidence
            prepared_identity = item.identity
            acquisition_seconds = time.monotonic()-tick
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            for family in FAMILIES:
                stage = "transfer:"+family
                tick = time.monotonic()
                batch = {section: {name: tensor.to(self._device) for name, tensor in values.items()}
                         for section, values in batches[family].items()}
                transfer_seconds += time.monotonic()-tick
                if not update_started:
                    # Normal family transfer order is preserved. CPU lease guards
                    # happen once here; a later deadline cannot split this update.
                    self._preparation.validate(item)
                    self._assert_sources()
                    self._before_first_forward()
                    update_started = True
                stage = "microbatch:"+family
                neural_microbatches += 1
                neural_episodes += evidence["families"][family]["exposures"]["episodes"]
                output = self.model(**batch["inputs"], decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
                losses = synchronous.objective(self._objective_id, output, batch)
                if any(not bool(torch.isfinite(value)) for value in losses.values()):
                    raise ValueError("nonfinite prefetched objective")
                (losses["loss"]/len(FAMILIES)).backward()
                reports.append(dict(family=family, depth=evidence["depth"], turns=evidence["turns"],
                    **copy.deepcopy(evidence["families"][family]),
                    **{name: float(value.detach()) for name, value in losses.items()}))
                del output, losses, batch
            stage = "optimizer"
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            physical = None
            self.optimizer.step()
            self._sync()
            physical = 1
            stage = "post_optimizer_source_identity"
            self._assert_sources()
            new_evidence = canonical._accumulate(self._evidence, evidence)
            stage = "release_prepared_storage"
            self._preparation.release(item)
            item, batches = None, None
            self._assert_sources()
            overrun = max(0., time.monotonic()-self._preparation_deadline)
            close_seconds = 0.
            status = self._preparation.report()
            if overrun > 0 or new_evidence["cursor"] >= self._preparation_stop or status["stopping"]:
                stage = "close_preparation_after_update"
                close_started = time.monotonic()
                self.close_preparation()
                close_seconds = time.monotonic()-close_started
                self._assert_sources()
            seconds = time.monotonic()-started-close_seconds
            timing = copy.deepcopy(self._timing)
            timing["retained_step_seconds"] += seconds
            timing["consumer_acquisition_seconds_included_in_step"] += acquisition_seconds
            timing["consumer_transfer_seconds_included_in_step"] += transfer_seconds
            result = dict(bundle_id=evidence["bundle_id"], cursor=new_evidence["cursor"], microbatches=reports,
                **{name: sum(row[name] for row in reports)/len(FAMILIES) for name in synchronous.LOSS_FIELDS},
                step_seconds=seconds, wall_seconds=time.monotonic()-called, materialization_seconds=0.,
                acquisition_seconds=acquisition_seconds, transfer_seconds=transfer_seconds,
                preparation_close_seconds=close_seconds, preparation=self.preparation_report,
                prepared_identity=prepared_identity, preparation_is_separate=True, timing_schema=TIMING_SCHEMA,
                deadline_overrun_seconds=overrun, update_started=True,
                physical_optimizer_updates=1, retained_optimizer_updates=1,
                drawn_microbatches=len(FAMILIES), drawn_episode_exposures=sum(
                    row["exposures"]["episodes"] for row in evidence["families"].values()),
                neural_attempted_microbatches=neural_microbatches, neural_attempted_episode_exposures=neural_episodes,
                completed_microbatches=len(reports), completed_microbatch_episode_exposures=sum(
                    row["exposures"]["episodes"] for row in reports))
            # Finish fallible cleanup/report construction before metadata commit.
            # An interruption during final assignment still poisons the trainer;
            # failure accounting below reads actual cursor rather than assuming0.
            self._failed = True
            self._evidence, self._timing, self.last_report = new_evidence, timing, result
            self._failed = False
            return copy.deepcopy(result)
        except BaseException:
            self._failed = True
            failed_at = time.monotonic()
            sync_ok, sync_error = True, None
            try:
                self._sync()
            except BaseException as error:
                sync_ok, sync_error = False, type(error).__name__ + ": " + str(error)
            synchronized = time.monotonic()
            cleanup_errors = []
            if item is not None and self._preparation is not None:
                try:
                    self._preparation.release(item)
                except BaseException as error:
                    cleanup_errors.append("release: " + repr(error))
            cleanup_started = time.monotonic()
            try:
                self.close_preparation()
            except BaseException as error:
                cleanup_errors.append("close: " + repr(error))
            self.last_report = dict(bundle_id=self._plan["schedules"][self._order][cursor_before],
                cursor=self.cursor, failed=True, failure_stage=stage, update_started=update_started,
                physical_optimizer_updates=physical, retained_optimizer_updates=self.cursor-cursor_before,
                drawn_microbatches=0 if evidence is None else len(FAMILIES),
                drawn_episode_exposures=0 if evidence is None else sum(
                    row["exposures"]["episodes"] for row in evidence["families"].values()),
                neural_attempted_microbatches=neural_microbatches, neural_attempted_episode_exposures=neural_episodes,
                completed_microbatches=len(reports), completed_microbatch_episode_exposures=sum(
                    row["exposures"]["episodes"] for row in reports),
                step_seconds=synchronized-started, wall_seconds=time.monotonic()-called,
                materialization_seconds=0., acquisition_seconds=acquisition_seconds, transfer_seconds=transfer_seconds,
                failure_sync_seconds=synchronized-failed_at, failure_sync_succeeded=sync_ok, failure_sync_error=sync_error,
                preparation_close_seconds=time.monotonic()-cleanup_started, cleanup_errors=cleanup_errors,
                preparation=self.preparation_report, preparation_is_separate=True, timing_schema=TIMING_SCHEMA,
                deadline_overrun_seconds=max(0., synchronized-self._preparation_deadline),
                scope="Failed neural work poisons learner until strict restore. CPU preparation is separate; unjoined producer and throwing optimizer completion remain explicitly incomplete/unknown.")
            raise
