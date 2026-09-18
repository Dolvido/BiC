"""Prospective, admitted-bundle layout research; no accepted learner interface.

Each invocation validates every complete pair in all three families before any
forward, then packs all three observation-only microbatches with the public
training admission checks. The packer repeats pair validation; both passes and
their canonical regeneration work are counted. The unchanged sequence objective
is divided by three per family, followed by one clip and one AdamW update.

This is not a CycleOwner provider. It generates no input schedule, reads no
evaluation bank, writes no checkpoint, restores no state, and promotes nothing.
Callers own protection-inventory admission, study order, deadlines, publication,
and total process cost. Any step or snapshot error permanently poisons this
instance; there is no implicit retry or recovery. Snapshot records state, not a
claim that restoration or resumed arithmetic has been verified.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import torch

from brain_in_computer.learning_student import _check_finite_tree, _cpu_copy, _finite_number, _integer
from experiments.composition_data import pack_composition_episodes
from experiments.foundation_curriculum import FAMILIES
from experiments import foundation_layout_curriculum as curriculum
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.sequence_training import sequence_objective


SCHEMA = "bic-foundation-layout-trainer-v1"
BUNDLE_SCHEMA = "bic-foundation-layout-bundle-v1"
OBJECTIVE_ID = "unchanged-sequence-objective-three-family-mean-v1"
COUNTS = ("episodes", "turns", "observation_tokens", "observation_bytes",
          "reply_target_tokens", "reply_target_bytes")
WORK_COUNTS = ("step_invocations", "packed_microbatches", "packed_episodes",
    "attempted_forwards", "completed_forwards", "attempted_forward_episodes", "completed_forward_episodes",
    "attempted_objectives", "completed_objectives", "attempted_backwards", "completed_backwards",
    "attempted_backward_episodes", "completed_backward_episodes", "optimizer_attempts", "optimizer_returns",
    "synchronized_optimizer_updates", "unknown_optimizer_outcomes", "retained_updates", "retained_episodes")


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("finite canonical metadata required") from error


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def source_hashes():
    """Existing training closure plus the additive layout adapter and trainer."""
    from experiments.foundation_training import source_hashes as baseline_sources
    result = baseline_sources()
    result.update(curriculum.source_hashes())
    name = "experiments/foundation_layout_training.py"
    result[name] = hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
    return result


def _bundle_evidence(bundle, *, cursor, layout, micro_batch_size, work):
    fields = {"schema", "bundle_id", "layout", "families"}
    if type(bundle) is not dict or set(bundle) != fields or bundle["schema"] != BUNDLE_SCHEMA:
        raise ValueError("exact admitted layout-bundle envelope required")
    _integer("bundle_id", bundle["bundle_id"])
    if bundle["bundle_id"] != cursor or bundle["layout"] != layout:
        raise ValueError("bundle ID must equal cursor and layout must match the recipe")
    rows = bundle["families"]
    if type(rows) is not dict or set(rows) != set(FAMILIES):
        raise ValueError("one complete microbatch from every canonical family required")
    evidence = {"bundle_id": cursor, "layout": layout, "families": {}}
    common_parents, seen_parents, seen_ids, dimensions = [], set(), set(), set()
    for family in FAMILIES:
        examples = rows[family]
        if type(examples) is not list or len(examples) != micro_batch_size:
            raise ValueError("exact configured complete-pair microbatch size required")
        counts = dict.fromkeys(COUNTS, 0)
        parent_records = []
        for offset in range(0, len(examples), 2):
            pair = examples[offset:offset + 2]
            curriculum.validate_pair(pair, work=work)
            recipe = pair[0]["recipe"]
            if recipe["layout"] != layout:
                raise ValueError("mixed lesson layouts are not admitted")
            parent = recipe["base_pair_sha256"]
            base_ids = recipe["base_ids"]
            if parent in seen_parents or any(identity in seen_ids for identity in base_ids):
                raise ValueError("repeated or crossed canonical parent pair within bundle")
            seen_parents.add(parent)
            seen_ids.update(base_ids)
            record = [family, parent, copy.deepcopy(base_ids)]
            common_parents.append(record)
            parent_records.append(record)
            for row in pair:
                if (row["family"] != family or row["split"] != "train"
                        or row["structure_partition"] != "train"):
                    raise ValueError("bundle family or canonical training admission differs")
                dimensions.add((row["depth"], len(row["turns"])))
                observed = sum(len(turn["text"].encode("utf-8")) for turn in row["turns"])
                replied = sum(len(turn["reply"].encode("utf-8")) for turn in row["turns"])
                turns = len(row["turns"])
                for key, count in zip(COUNTS, (1, turns, observed + 2 * turns, observed, replied + turns, replied)):
                    counts[key] += count
        evidence["families"][family] = {"rows_sha256": _hash(examples),
            "recipes_sha256": _hash([row["recipe"] for row in examples[::2]]),
            "common_parents_sha256": _hash(parent_records), "exposures": counts}
    if len(dimensions) != 1:
        raise ValueError("all families must share one depth and actual turn count")
    evidence["depth"], evidence["turns"] = next(iter(dimensions))
    evidence["common_parents_sha256"] = _hash(common_parents)
    evidence["bundle_rows_sha256"] = _hash([[family, rows[family]] for family in FAMILIES])
    return evidence


def _initial_evidence():
    return {"cursor": 0, "consumed_bundle_identity_sha256": _hash([SCHEMA, "bundles"]),
        "consumed_common_parent_identity_sha256": _hash([SCHEMA, "common-parents"]),
        "exposures": {family: dict.fromkeys(COUNTS, 0) for family in FAMILIES}}


def _accumulate(previous, bundle):
    result = copy.deepcopy(previous)
    result["cursor"] += 1
    key = "consumed_bundle_identity_sha256"
    result[key] = _hash([result[key], bundle])
    key = "consumed_common_parent_identity_sha256"
    result[key] = _hash([result[key], bundle["bundle_id"], bundle["common_parents_sha256"]])
    for family in FAMILIES:
        for name in COUNTS:
            result["exposures"][family][name] += bundle["families"][family]["exposures"][name]
    return result


class FoundationLayoutTrainer:
    """Fresh initialization only; explicit recipe and complete external bundles."""

    def __init__(self, *, seed, config, learning_rate, micro_batch_size, layout, objective_id, device="cpu"):
        started, cpu_started = time.monotonic(), time.process_time()
        _integer("seed", seed)
        _integer("micro_batch_size", micro_batch_size, 2)
        _finite_number("learning_rate", learning_rate)
        if seed >= 2**63 or micro_batch_size % 2 or learning_rate <= 0:
            raise ValueError("bounded seed, complete pairs, and positive learning rate required")
        if type(config) is not SequenceConfig or not 2 <= config.max_turns <= 12:
            raise ValueError("explicit SequenceConfig with two to twelve turn capacity required")
        if type(layout) is not str or layout not in curriculum.LAYOUTS:
            raise ValueError("explicit original or varied layout required")
        if type(objective_id) is not str or objective_id != OBJECTIVE_ID:
            raise ValueError("explicit unchanged three-family sequence objective identity required")
        self._device = torch.device(device)
        if self._device.type not in ("cpu", "cuda"):
            raise ValueError("local CPU or CUDA device required")
        self._config, self._layout, self._micro_batch_size = config, layout, micro_batch_size
        self._sources = source_hashes()
        self.model = build_sequence_student(seed, device=self._device, config=config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self._recipe = {"schema": SCHEMA, "curriculum_version": curriculum.VERSION,
            "source_sha256": self._sources, "seed": seed, "config": asdict(config),
            "layout": layout, "micro_batch_size": micro_batch_size, "family_order": list(FAMILIES),
            "learning_rate": learning_rate,
            "optimizer": {"name": "AdamW", "param_groups": copy.deepcopy(self.optimizer.state_dict()["param_groups"])},
            "objective": {"id": objective_id, "function": "experiments.sequence_training.sequence_objective",
                "family_outer_weight": 1. / len(FAMILIES), "ack_coefficient": .25,
                "reply_coefficient": .1, "observation_coefficient": .1},
            "gradient_clip": 1., "torch_version": str(torch.__version__),
            "protection_admission": "caller-owned before step; this trainer validates canonical train admission",
            "resume_supported": False, "automatic_promotion": False}
        self._failed = False
        self._evidence = _initial_evidence()
        self._work = dict.fromkeys(WORK_COUNTS, 0)
        self._validation_work = curriculum.WorkLedger().report()
        self._cost = {"setup_wall_seconds": 0., "setup_cpu_seconds": 0.,
            "step_wall_seconds": 0., "step_cpu_seconds": 0.,
            "retained_step_wall_seconds": 0., "retained_step_cpu_seconds": 0.,
            "snapshot_invocations": 0, "snapshot_wall_seconds": 0., "snapshot_cpu_seconds": 0.}
        self.last_report = self.last_snapshot_report = None
        self._sync()
        self._assert_recipe_and_sources()
        self._cost.update(setup_wall_seconds=time.monotonic() - started,
                          setup_cpu_seconds=time.process_time() - cpu_started)

    @property
    def cursor(self):
        return self._evidence["cursor"]

    @property
    def updates(self):
        return self.cursor

    @property
    def failed(self):
        return self._failed

    @property
    def recipe(self):
        return copy.deepcopy(self._recipe)

    @property
    def accounting(self):
        return {"work": copy.deepcopy(self._work), "validation_work": copy.deepcopy(self._validation_work),
                "cost": copy.deepcopy(self._cost)}

    def _sync(self):
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _assert_recipe_and_sources(self):
        if source_hashes() != self._sources:
            raise ValueError("layout training source identity changed")
        if (asdict(self.model.config) != self._recipe["config"]
                or _json(self.optimizer.state_dict()["param_groups"]) != _json(self._recipe["optimizer"]["param_groups"])):
            raise ValueError("model configuration or optimizer recipe changed")

    def step(self, admitted_bundle):
        started, cpu_started = time.monotonic(), time.process_time()
        work, batches = curriculum.WorkLedger(), {}
        report = {"cursor_before": self.cursor, "cursor": self.cursor, "failed": True,
            "failure_stage": "preflight", "physical_optimizer_updates": 0,
            "queued_device_work_synchronized": False, "microbatches": [],
            **dict.fromkeys(WORK_COUNTS, 0)}
        report["step_invocations"] = 1
        self.last_report = report
        try:
            if self._failed:
                raise RuntimeError("poisoned layout trainer cannot retry; a separate fresh invocation is required")
            self._assert_recipe_and_sources()
            self._sync()
            report["failure_stage"] = "admission"
            bundle = copy.deepcopy(admitted_bundle)
            evidence = _bundle_evidence(bundle, cursor=self.cursor, layout=self._layout,
                micro_batch_size=self._micro_batch_size, work=work)
            report["bundle_evidence"] = evidence
            # No packer fast path exists: keep its public training checks and
            # charge the repeated canonical pair regeneration explicitly.
            validator = partial(curriculum.validate_pair, work=work)
            for family in FAMILIES:
                report["failure_stage"] = f"packing:{family}"
                batches[family] = pack_composition_episodes(bundle["families"][family], device=self._device,
                    training=True, pair_validator=validator, max_turns=self._config.max_turns,
                    max_input_bytes=self._config.max_input_bytes, max_context_tokens=self._config.max_positions,
                    max_reply_bytes=self._config.max_output_bytes)
                report["packed_microbatches"] += 1
                report["packed_episodes"] += self._micro_batch_size
            report["admission_and_packing_wall_seconds_included_in_step"] = time.monotonic() - started
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            for family in FAMILIES:
                batch = batches.pop(family)
                report["failure_stage"] = f"forward:{family}"
                report["attempted_forwards"] += 1
                report["attempted_forward_episodes"] += self._micro_batch_size
                output = self.model(**batch["inputs"], decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
                report["completed_forwards"] += 1
                report["completed_forward_episodes"] += self._micro_batch_size
                report["failure_stage"] = f"objective:{family}"
                report["attempted_objectives"] += 1
                losses = sequence_objective(output, batch)
                if any(not bool(torch.isfinite(value)) for value in losses.values()):
                    raise ValueError("nonfinite unchanged sequence objective")
                report["completed_objectives"] += 1
                report["failure_stage"] = f"backward:{family}"
                report["attempted_backwards"] += 1
                report["attempted_backward_episodes"] += self._micro_batch_size
                (losses["loss"] / len(FAMILIES)).backward()
                report["completed_backwards"] += 1
                report["completed_backward_episodes"] += self._micro_batch_size
                report["microbatches"].append({"family": family,
                    **{name: float(value.detach()) for name, value in losses.items()}})
                del output, losses, batch
            report["failure_stage"] = "gradient_clip"
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            report["failure_stage"] = "optimizer"
            report["optimizer_attempts"] = 1
            report["physical_optimizer_updates"] = None
            self.optimizer.step()
            report["optimizer_returns"] = 1
            self._sync()
            report["queued_device_work_synchronized"] = True
            report["physical_optimizer_updates"] = report["synchronized_optimizer_updates"] = 1
            report["failure_stage"] = "commit"
            self._assert_recipe_and_sources()
            next_evidence = _accumulate(self._evidence, evidence)
            for name in ("loss", "action_loss", "reply_loss", "observation_language_loss"):
                report[name] = sum(row[name] for row in report["microbatches"]) / len(FAMILIES)
            report.update(cursor=next_evidence["cursor"], failed=False, failure_stage=None,
                          retained_updates=1, retained_episodes=self._micro_batch_size * len(FAMILIES))
            self._evidence = next_evidence
        except BaseException as error:
            self._failed = True
            report["error"] = f"{type(error).__name__}: {error}"
            if report["physical_optimizer_updates"] is None:
                report["unknown_optimizer_outcomes"] = 1
            sync_started = time.monotonic()
            try:
                self._sync()
                report["queued_device_work_synchronized"] = True
            except BaseException as sync_error:
                report["queued_device_work_synchronized"] = False
                report["failure_sync_error"] = f"{type(sync_error).__name__}: {sync_error}"
            report["failure_sync_wall_seconds_included_in_step"] = time.monotonic() - sync_started
            raise
        finally:
            batches.clear()
            report["wall_seconds"] = time.monotonic() - started
            report["cpu_seconds"] = time.process_time() - cpu_started
            report["validation_work"] = work.report()
            report["completion_scope"] = ("completed forwards/backwards count returned calls; device completion is only known "
                "when queued_device_work_synchronized is true; a throwing optimizer outcome remains unknown")
            for name in WORK_COUNTS:
                self._work[name] += report[name]
            for name, count in report["validation_work"].items():
                self._validation_work[name] += count
            for kind in ("wall", "cpu"):
                self._cost[f"step_{kind}_seconds"] += report[f"{kind}_seconds"]
                if report["retained_updates"]:
                    self._cost[f"retained_step_{kind}_seconds"] += report[f"{kind}_seconds"]
        return copy.deepcopy(report)

    def snapshot(self):
        """Full CPU weights/AdamW/evidence; caller charges serialization and I/O."""
        started, cpu_started = time.monotonic(), time.process_time()
        report = {"failed": True, "cursor": self.cursor}
        self.last_snapshot_report = report
        try:
            if self._failed:
                raise RuntimeError("poisoned layout trainer cannot publish a retained snapshot")
            self._assert_recipe_and_sources()
            self._sync()
            payload = {"schema": SCHEMA, "recipe": self.recipe,
                "weights": _cpu_copy(self.model.state_dict()), "optimizer": _cpu_copy(self.optimizer.state_dict()),
                "cursor": self.cursor, "evidence": copy.deepcopy(self._evidence)}
            _check_finite_tree(payload, "layout research snapshot")
            self._assert_recipe_and_sources()
            report["failed"] = False
        except BaseException as error:
            self._failed = True
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            report["wall_seconds"] = time.monotonic() - started
            report["cpu_seconds"] = time.process_time() - cpu_started
            self._cost["snapshot_invocations"] += 1
            self._cost["snapshot_wall_seconds"] += report["wall_seconds"]
            self._cost["snapshot_cpu_seconds"] += report["cpu_seconds"]
        payload["accounting"] = self.accounting
        return payload
