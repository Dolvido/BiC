"""Prescribed immutable foundation-bundle training; no study or allocation policy.

One step consumes one plan ID: three independent family microbatches, each with
one third of the unchanged sequence objective, then one clip and AdamW update.
Order changes neither lesson IDs nor stateless name/value realizations. Failed
steps poison the trainer until explicit restoration; the caller owns atomic
durability and any ledger of discarded physical work. This module does not save
files, select models, invoke teachers, or claim a learning benefit.

Restore regenerates the consumed input prefix to authenticate exact exposure and
recipe evidence. It validates/restores optimizer tensors, but does not reproduce
the past gradient arithmetic. Replay/setup time is separate from retained step
time; materialization time is already INCLUDED in step time. No hidden RNG stream
or prefetch state exists. Runtime CUDA reproducibility policy belongs to callers.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from brain_in_computer.learning_student import _check_finite_tree, _cpu_copy, _finite_number, _integer
from experiments.composition_data import pack_composition_episodes
from experiments.foundation_curriculum import DEPTHS, FAMILIES, TURN_BUCKETS, VERSION, validate_pair
from experiments.foundation_plan import _materialize_validated_bundle, validate_plan
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.sequence_training import sequence_objective


SCHEMA = "bic-foundation-trainer-v1"
COUNTS = ("episodes", "turns", "observation_tokens", "observation_bytes", "reply_target_tokens", "reply_target_bytes")


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("finite canonical metadata required") from error


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def source_hashes():
    from experiments.realization_study import source_hashes as base_sources
    root = Path(__file__).resolve().parents[1]
    names = ("experiments/foundation_curriculum.py", "experiments/foundation_plan.py", "experiments/foundation_training.py")
    return {**base_sources(), **{name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}


def _protected(values):
    if isinstance(values, (str, bytes)):
        raise ValueError("protected transcripts require an iterable of SHA256 strings")
    try:
        values = list(values)
    except TypeError as error:
        raise ValueError("protected transcript iterable required") from error
    if any(type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in values):
        raise ValueError("protected transcripts must be lowercase SHA256 strings")
    return tuple(sorted(set(values)))


def _admit_plan(plan, order):
    validate_plan(plan)
    if type(order) is not str or order not in ("curriculum", "mixed"):
        raise ValueError("order must be curriculum or mixed")
    return copy.deepcopy(plan)


def _empty_evidence():
    return {"cursor": 0, "consumed_ids_sha256": _hash([SCHEMA, "ids"]),
        "consumed_rows_sha256": _hash([SCHEMA, "rows"]), "consumed_recipes_sha256": _hash([SCHEMA, "recipes"]),
        "family_microbatches": dict.fromkeys(FAMILIES, 0),
        "bucket_microbatches": {family: {str(turns): 0 for turns in TURN_BUCKETS} for family in FAMILIES},
        "depth_microbatches": {family: {str(depth): 0 for depth in DEPTHS} for family in FAMILIES},
        "exposures": {family: dict.fromkeys(COUNTS, 0) for family in FAMILIES}}


def _rows_evidence(plan, bundle_id, rows, protected):
    bundle = plan["bundles"][str(bundle_id)]
    if type(rows) is not dict or set(rows) != set(FAMILIES):
        raise ValueError("one canonical microbatch from every family required")
    result = {"bundle_id": bundle_id, "depth": bundle["depth"], "turns": bundle["turns"], "families": {}}
    for family in FAMILIES:
        examples = rows[family]
        if type(examples) is not list or len(examples) != plan["config"]["micro_batch_size"]:
            raise ValueError("bundle microbatch size differs")
        for index in range(0, len(examples), 2):
            validate_pair(examples[index:index + 2])
        counts = dict.fromkeys(COUNTS, 0)
        for row in examples:
            if (row["family"] != family or row["split"] != "train" or row["structure_partition"] != "train"
                    or row["depth"] != bundle["depth"] or len(row["turns"]) != bundle["turns"]):
                raise ValueError("canonical bundle family/depth/length/admission differs")
            if transcript_digest(row) in protected:
                raise ValueError("bundle collides with a protected exact transcript")
            turns = len(row["turns"])
            observed = sum(len(turn["text"].encode("utf-8")) for turn in row["turns"])
            replied = sum(len(turn["reply"].encode("utf-8")) for turn in row["turns"])
            for name, count in zip(COUNTS, (1, turns, observed + 2 * turns, observed, replied + turns, replied)):
                counts[name] += count
        result["families"][family] = {"rows_sha256": _hash(examples),
            "recipes_sha256": _hash([row["recipe"] for row in examples[::2]]), "exposures": counts}
    return result


def _accumulate(evidence, bundle):
    evidence = copy.deepcopy(evidence)
    evidence["cursor"] += 1
    evidence["consumed_ids_sha256"] = _hash([evidence["consumed_ids_sha256"], bundle["bundle_id"]])
    for kind in ("rows", "recipes"):
        key = f"consumed_{kind}_sha256"
        evidence[key] = _hash([evidence[key], bundle["bundle_id"],
            {family: bundle["families"][family][f"{kind}_sha256"] for family in FAMILIES}])
    for family in FAMILIES:
        evidence["family_microbatches"][family] += 1
        evidence["bucket_microbatches"][family][str(bundle["turns"])] += 1
        evidence["depth_microbatches"][family][str(bundle["depth"])] += 1
        for name in COUNTS:
            evidence["exposures"][family][name] += bundle["families"][family]["exposures"][name]
    return evidence


def replay_evidence(plan, order, cursor, protected_transcripts=(), *, include_bundles=False):
    """Pure canonical input replay; no model creation, gradients or file writes."""
    plan = _admit_plan(plan, order)
    _integer("cursor", cursor)
    if cursor > len(plan["schedules"][order]) or type(include_bundles) is not bool:
        raise ValueError("replay cursor or include_bundles differs")
    protected = set(_protected(protected_transcripts))
    evidence, records = _empty_evidence(), []
    for bundle_id in plan["schedules"][order][:cursor]:
        rows = _materialize_validated_bundle(plan, bundle_id)
        bundle = _rows_evidence(plan, bundle_id, rows, protected)
        evidence = _accumulate(evidence, bundle)
        if include_bundles:
            records.append(bundle)
    return {"evidence": evidence, "bundles": records} if include_bundles else evidence


class FoundationTrainer:
    def __init__(self, plan, order="curriculum", *, seed, config=None, learning_rate=.001,
                 device="cpu", protected_transcripts=(), payload=None):
        started = time.monotonic()
        self._plan = _admit_plan(plan, order)
        self._protected = _protected(protected_transcripts)
        self._protected_set = set(self._protected)
        self._order, self._seed, self._device = order, seed, torch.device(device)
        _integer("seed", seed)
        if seed >= 2**63:
            raise ValueError("seed must be below 2**63")
        _finite_number("learning_rate", learning_rate)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if config is not None and type(config) is not SequenceConfig:
            raise ValueError("SequenceConfig required")
        self._config = SequenceConfig(max_turns=12) if config is None else config
        if self._config.max_turns < max(bundle["turns"] for bundle in self._plan["bundles"].values()):
            raise ValueError("model turn capacity cannot fit the declared plan")
        self._learning_rate = learning_rate
        self.model, self.optimizer = self._build()
        self._sources = source_hashes()
        self._recipe = {"schema": SCHEMA, "curriculum_version": VERSION, "source_sha256": self._sources,
            "plan_sha256": _hash(self._plan), "order": order, "seed": seed, "config": asdict(self._config),
            "protected_transcripts_sha256": _hash(self._protected), "protected_transcripts_count": len(self._protected),
            "micro_batch_size": self._plan["config"]["micro_batch_size"], "family_order": list(FAMILIES),
            "learning_rate": learning_rate, "optimizer": {"name": "AdamW", "param_groups": copy.deepcopy(self.optimizer.state_dict()["param_groups"])},
            "objective": "mean of three unchanged sequence_objective microbatches", "gradient_clip": 1.}
        self._evidence = _empty_evidence()
        self._timing = {"retained_step_seconds": 0., "materialization_seconds_included_in_step": 0.}
        self._failed = False
        self.last_report = None
        self.last_restore_seconds = 0.
        if payload is not None:
            self.restore(payload)
        self.setup_seconds = time.monotonic() - started

    @property
    def cursor(self):
        return self._evidence["cursor"]

    @property
    def updates(self):
        return self.cursor

    @property
    def recipe(self):
        return copy.deepcopy(self._recipe)

    def _build(self):
        model = build_sequence_student(self._seed, device=self._device, config=self._config)
        return model, torch.optim.AdamW(model.parameters(), lr=self._learning_rate)

    def _assert_sources(self):
        if source_hashes() != self._sources:
            raise ValueError("foundation training source identity changed")

    def _sync(self):
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def step(self):
        called = time.monotonic()
        # Every invocation gets its OWN report, including no-work preflight
        # failures. Never leave an earlier update available for double counting.
        self.last_report = {"cursor": self.cursor, "failed": True, "failure_stage": "preflight",
            "physical_optimizer_updates": 0, "retained_optimizer_updates": 0,
            "drawn_microbatches": 0, "drawn_episode_exposures": 0,
            "neural_attempted_microbatches": 0, "neural_attempted_episode_exposures": 0,
            "completed_microbatches": 0, "completed_microbatch_episode_exposures": 0,
            "wall_seconds": 0., "step_seconds": 0., "materialization_seconds": 0.}
        if self._failed:
            self.last_report["failure_stage"] = "preflight:restore_required"
            self.last_report["wall_seconds"] = time.monotonic() - called
            raise RuntimeError("failed foundation step requires explicit restoration")
        try:
            self._assert_sources()
        except BaseException:
            self.last_report["failure_stage"] = "preflight:source_identity"
            self.last_report["wall_seconds"] = time.monotonic() - called
            raise
        schedule = self._plan["schedules"][self._order]
        if self.cursor == len(schedule):
            self.last_report.update(failed=False, failure_stage=None, complete=True,
                                    wall_seconds=time.monotonic() - called)
            raise StopIteration("foundation plan is complete")
        bundle_id = schedule[self.cursor]
        try:
            self._sync()
        except BaseException:
            self.last_report["failure_stage"] = "preflight:device_sync"
            self.last_report["wall_seconds"] = time.monotonic() - called
            raise
        started = time.monotonic()
        reports, stage, physical_updates = [], "materialization", 0
        rows, materialization_seconds = None, None
        neural_microbatches, neural_episodes = 0, 0
        try:
            rows = _materialize_validated_bundle(self._plan, bundle_id)
            bundle = _rows_evidence(self._plan, bundle_id, rows, self._protected_set)
            materialization_seconds = time.monotonic() - started
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            for family in FAMILIES:
                stage = f"microbatch:{family}"
                batch = pack_composition_episodes(rows[family], device=self._device, training=True,
                    pair_validator=validate_pair, max_turns=self._config.max_turns,
                    max_input_bytes=self._config.max_input_bytes, max_context_tokens=self._config.max_positions,
                    max_reply_bytes=self._config.max_output_bytes)
                neural_microbatches += 1
                neural_episodes += len(rows[family])
                output = self.model(**batch["inputs"], decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
                losses = sequence_objective(output, batch)
                if any(not bool(torch.isfinite(value)) for value in losses.values()):
                    raise ValueError("nonfinite foundation objective")
                (losses["loss"] / len(FAMILIES)).backward()
                reports.append({"family": family, "depth": bundle["depth"], "turns": bundle["turns"],
                    **copy.deepcopy(bundle["families"][family]),
                    **{name: float(value.detach()) for name, value in losses.items()}})
                del output, losses, batch
            stage = "optimizer"
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            physical_updates = None  # A throwing optimizer may have applied some/all arithmetic.
            self.optimizer.step()
            self._sync()
            physical_updates = 1  # CUDA completion is known only after synchronization.
            seconds = time.monotonic() - started
            evidence = _accumulate(self._evidence, bundle)
            self._timing["retained_step_seconds"] += seconds
            self._timing["materialization_seconds_included_in_step"] += materialization_seconds
            self._evidence = evidence
            self.last_report = {"bundle_id": bundle_id, "cursor": self.cursor, "microbatches": reports,
                **{name: sum(row[name] for row in reports) / len(FAMILIES)
                   for name in ("loss", "action_loss", "reply_loss", "observation_language_loss")},
                "step_seconds": seconds, "materialization_seconds": materialization_seconds,
                "wall_seconds": time.monotonic() - called,
                "physical_optimizer_updates": 1, "retained_optimizer_updates": 1,
                "drawn_microbatches": len(rows), "drawn_episode_exposures": sum(map(len, rows.values())),
                "neural_attempted_microbatches": neural_microbatches, "neural_attempted_episode_exposures": neural_episodes,
                "completed_microbatches": len(reports),
                "completed_microbatch_episode_exposures": sum(row["exposures"]["episodes"] for row in reports)}
            return copy.deepcopy(self.last_report)
        except BaseException:
            self._failed = True
            failure_occurred = time.monotonic()
            sync_succeeded, sync_error = True, None
            try:
                # Include already queued device work when synchronization is
                # possible. Never replace the original failure with this error.
                self._sync()
            except BaseException as error:
                sync_succeeded, sync_error = False, f"{type(error).__name__}: {error}"
            synchronized = time.monotonic()
            elapsed = synchronized - started
            returned_rows = type(rows) is dict and all(type(value) is list for value in rows.values())
            self.last_report = {"bundle_id": bundle_id, "cursor": self.cursor, "failed": True,
                "failure_stage": stage, "completed_microbatches": len(reports),
                "drawn_microbatches": len(rows) if returned_rows else None,
                "drawn_episode_exposures": sum(map(len, rows.values())) if returned_rows else None,
                "neural_attempted_microbatches": neural_microbatches, "neural_attempted_episode_exposures": neural_episodes,
                "completed_microbatch_episode_exposures": sum(row["exposures"]["episodes"] for row in reports),
                "physical_optimizer_updates": physical_updates, "retained_optimizer_updates": 0,
                "step_seconds": elapsed, "wall_seconds": time.monotonic() - called,
                "materialization_seconds": failure_occurred - started if materialization_seconds is None else materialization_seconds,
                "materialization_completed": materialization_seconds is not None,
                "failure_sync_seconds": synchronized - failure_occurred,
                "failure_sync_succeeded": sync_succeeded, "failure_sync_error": sync_error,
                "step_time_includes_queued_device_work": sync_succeeded,
                "scope": "failed work is not committed; materialization and failure synchronization are included in step time; incomplete synchronization makes queued-device timing uncertain; partial generation is unknown if materializer did not return; throwing optimizer completion remains unknown"}
            raise

    def snapshot(self):
        if self._failed:
            raise RuntimeError("failed foundation step requires explicit restoration before snapshot")
        self._assert_sources()
        payload = {"schema": SCHEMA, "recipe": self.recipe, "weights": _cpu_copy(self.model.state_dict()),
            "optimizer": _cpu_copy(self.optimizer.state_dict()), "cursor": self.cursor,
            "evidence": copy.deepcopy(self._evidence), "timing": copy.deepcopy(self._timing)}
        _check_finite_tree(payload, "foundation checkpoint")
        return payload

    def restore(self, payload):
        """Preflight exact input evidence and all tensors before replacing state."""
        started = time.monotonic()
        self._assert_sources()
        fields = {"schema", "recipe", "weights", "optimizer", "cursor", "evidence", "timing"}
        if type(payload) is not dict or set(payload) != fields or payload["schema"] != SCHEMA:
            raise ValueError("foundation checkpoint fields differ")
        _check_finite_tree(payload, "foundation checkpoint")
        if _json(payload["recipe"]) != _json(self._recipe):
            raise ValueError("foundation resume recipe/source/plan/order/config differs")
        cursor = payload["cursor"]
        _integer("checkpoint cursor", cursor)
        expected_evidence = replay_evidence(self._plan, self._order, cursor, self._protected)
        if _json(payload["evidence"]) != _json(expected_evidence):
            raise ValueError("checkpoint consumed IDs/recipes/counts differ from canonical replay")
        timing = payload["timing"]
        if type(timing) is not dict or set(timing) != set(self._timing):
            raise ValueError("checkpoint timing fields differ")
        for name, value in timing.items():
            _finite_number(name, value)
            if value < 0 or cursor == 0 and value != 0:
                raise ValueError("checkpoint timing cannot be negative or precede work")
        if timing["materialization_seconds_included_in_step"] > timing["retained_step_seconds"]:
            raise ValueError("materialization is a subset of step time")
        model, optimizer = self._build()
        expected_weights = model.state_dict()
        weights = payload["weights"]
        if type(weights) is not dict or set(weights) != set(expected_weights):
            raise ValueError("checkpoint model parameter set differs")
        for name, expected in expected_weights.items():
            value = weights[name]
            if (not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype
                    or value.device.type != "cpu" or value.requires_grad):
                raise ValueError("checkpoint model tensor shape/dtype/CPU boundary differs")
            if cursor == 0 and not torch.equal(value, expected.detach().cpu()):
                raise ValueError("zero-update checkpoint differs from seeded initialization")
        aliases = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            first = aliases.setdefault(id(parameter), name)
            if not torch.equal(weights[name], weights[first]):
                raise ValueError("checkpoint tied parameter aliases differ")
        state = payload["optimizer"]
        if (type(state) is not dict or set(state) != {"state", "param_groups"}
                or _json(state["param_groups"]) != _json(optimizer.state_dict()["param_groups"])
                or type(state["state"]) is not dict):
            raise ValueError("checkpoint AdamW configuration differs")
        parameters = list(model.parameters())
        moments = state["state"]
        if any(type(index) is not int for index in moments) or set(moments) != (set(range(len(parameters))) if cursor else set()):
            raise ValueError("checkpoint optimizer state/update coverage differs")
        for index, values in moments.items():
            if type(values) is not dict or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("checkpoint AdamW moment fields differ")
            step = values["step"]
            if (not isinstance(step, torch.Tensor) or step.numel() != 1 or not step.is_floating_point()
                    or step.device.type != "cpu" or step.requires_grad or float(step) != cursor):
                raise ValueError("checkpoint optimizer step differs from consumed cursor")
            for name in ("exp_avg", "exp_avg_sq"):
                value = values[name]
                if (not isinstance(value, torch.Tensor) or value.shape != parameters[index].shape
                        or value.dtype != parameters[index].dtype or value.device.type != "cpu" or value.requires_grad):
                    raise ValueError("checkpoint optimizer moment shape/dtype/CPU boundary differs")
            if bool(values["exp_avg_sq"].lt(0).any()):
                raise ValueError("checkpoint optimizer second moment must be nonnegative")
        model.load_state_dict(weights, strict=True)
        optimizer.load_state_dict(_cpu_copy(state))
        restored_evidence, restored_timing = copy.deepcopy(expected_evidence), copy.deepcopy(timing)
        # Preflight errors above leave the old object intact. An asynchronous
        # interruption during these final assignments must instead poison it,
        # never expose a mixture of old counters and restored learner tensors.
        self._failed = True
        self.model, self.optimizer = model, optimizer
        self._evidence, self._timing = restored_evidence, restored_timing
        self.last_report = None
        self.last_restore_seconds = time.monotonic() - started
        self._failed = False
        return self
