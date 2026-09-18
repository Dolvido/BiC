"""Two fixed shared objectives on a fresh flat authenticated foundation learner.

The baseline calls the unchanged sequence objective. The sole candidate replaces
its reply reduction with a mean of episode-turn token means, balanced over the
present semantic response classes. Architecture, all positive coefficients,
outer family averaging, clipping and AdamW are unchanged. No callable objective,
old-checkpoint migration, source mutation or persistent-index trust is provided.

Step/restore are versioned copies of the frozen foundation/variant boundaries:
failed physical work poisons the trainer until explicit strict indexed restore.
Canonical input evidence is checked; historical gradient arithmetic is not replayed.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from brain_in_computer.language import ByteCodec
from brain_in_computer.learning_student import _check_finite_tree, _cpu_copy, _finite_number, _integer
from experiments import foundation_training as legacy
from experiments.foundation_curriculum import FAMILIES, validate_pair
from experiments.foundation_evidence import json_digest
from experiments.foundation_plan_index import AuthenticatedPlanIndex, source_hashes as index_sources
from experiments.sequence_student import build_sequence_student
from experiments.sequence_training import sequence_objective


SCHEMA = "bic-foundation-objective-trainer-v1"
OBJECTIVES = {
    "baseline": "bic-foundation-original-sequence-objective-v1",
    "balanced_reply": "bic-foundation-balanced-reply-objective-v1",
}
LOSS_FIELDS = ("loss", "action_loss", "reply_loss", "observation_language_loss")


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    name = "experiments/foundation_objective_training.py"
    return {**index_sources(), name: hashlib.sha256((root / name).read_bytes()).hexdigest()}


def objective_recipe(objective_id):
    if type(objective_id) is not str or objective_id not in OBJECTIVES:
        raise ValueError("explicit baseline or balanced_reply objective ID required")
    return dict(id=objective_id, schema=OBJECTIVES[objective_id],
        weights=dict(query=1., acknowledgement=.25, reply=.1, observation=.1),
        query_reduction="Original equal mean over present query classes 0/1/2 within each family",
        acknowledgement_reduction="Original class3 cross entropy",
        observation_reduction="Original mean turn-index pooled nonPAD token means",
        reply_reduction=("Original mean turn-index pooled nonPAD token means" if objective_id == "baseline" else
            "Each episode-turn mean over nonPAD targets including EOS; mean utterances within each present semantic class0/1/2/3; mean present classes"),
        outer_reduction="Mean of the three independent family objectives",
        vocabulary_size=ByteCodec.VOCAB_SIZE, action_classes=[0, 1, 2, 3],
        model_inputs="Unchanged observations and teacher-forced reply prefixes; semantic classes affect loss only")


def balanced_reply_loss(output, batch):
    """Shared full-vocabulary reply CE; no class enters the model's inputs."""
    labels = batch["supervision"]["action_targets"]
    targets = batch["supervision"]["reply_targets"]
    logits = output["language_logits"]
    if (not isinstance(labels, torch.Tensor) or labels.ndim != 2 or min(labels.shape) < 1
            or labels.dtype != torch.long or not bool(((labels >= 0) & (labels <= 3)).all())
            or not isinstance(targets, torch.Tensor) or targets.ndim != 3 or targets.shape[:2] != labels.shape
            or targets.shape[2] < 1 or targets.dtype != torch.long or targets.device != labels.device
            or not isinstance(logits, torch.Tensor) or logits.shape != (*targets.shape, ByteCodec.VOCAB_SIZE)
            or logits.device != targets.device):
        raise ValueError("nonempty canonical episode-turn labels, targets and full259 logits required")
    counts = targets.ne(ByteCodec.PAD).sum(dim=2)
    if not bool(counts.gt(0).all()):
        raise ValueError("every actual reply turn must have at least one nonPAD target")
    tokens = F.cross_entropy(logits.flatten(0, 2), targets.flatten(), ignore_index=ByteCodec.PAD,
        reduction="none").reshape_as(targets)
    utterances = tokens.sum(dim=2) / counts
    present = [utterances[labels.eq(target)].mean() for target in range(4) if bool(labels.eq(target).any())]
    return torch.stack(present).mean()


def objective(objective_id, output, batch):
    """Exactly two code-defined objectives; no injected callable or coefficient."""
    if type(objective_id) is not str or objective_id not in OBJECTIVES:
        raise ValueError("explicit baseline or balanced_reply objective ID required")
    original = sequence_objective(output, batch)
    if objective_id == "baseline":
        return original
    reply = balanced_reply_loss(output, batch)
    return dict(loss=original["action_loss"] + .1 * reply + .1 * original["observation_language_loss"],
        action_loss=original["action_loss"], reply_loss=reply,
        observation_language_loss=original["observation_language_loss"])


class ObjectiveFoundationTrainer(legacy.FoundationTrainer):
    def __init__(self, plan, order="curriculum", *, objective_id, seed,
                 admission_protected_transcripts, protected_transcripts, admission_receipt,
                 plan_index, config=None, learning_rate=.001, device="cpu", payload=None):
        started = time.monotonic()
        descriptor = objective_recipe(objective_id)
        if type(plan_index) is not AuthenticatedPlanIndex:
            raise ValueError("a real in-process AuthenticatedPlanIndex must be supplied")
        admitted = legacy._admit_plan(plan, order)
        history = legacy._protected(admission_protected_transcripts)
        protection = legacy._protected(protected_transcripts)
        if not set(history) <= set(protection):
            raise ValueError("original admission protection must be included in expanded protection")
        if type(admission_receipt) is not dict:
            raise ValueError("explicit canonical admission receipt required")
        preparation_started = time.monotonic()
        identity = plan_index.identity
        expected = {"plan_sha256": json_digest(admitted),
            "admission_receipt_sha256": json_digest(admission_receipt),
            "admission_protection_sha256": json_digest(list(history)), "admission_protection_count": len(history),
            "expanded_protection_sha256": json_digest(list(protection)), "expanded_protection_count": len(protection),
            "bundle_count": len(admitted["bundles"]), "micro_batch_size": admitted["config"]["micro_batch_size"]}
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError("plan index belongs to another plan/admission/protection contract")
        self._index, self._index_identity = plan_index, identity
        self._objective_id = objective_id
        self.index_preparation_seconds = time.monotonic() - preparation_started
        # Full index/identity admission precedes every fresh learner construction.
        super().__init__(admitted, order, seed=seed, config=config, learning_rate=learning_rate,
                         device=device, protected_transcripts=protection, payload=None)
        self._sources = source_hashes()
        self._recipe.update(schema=SCHEMA, source_sha256=copy.deepcopy(self._sources), objective=descriptor,
            architecture={"name": "flat", "schema": "bic-flat-sequence-v1"},
            plan_index_identity=copy.deepcopy(identity))
        if self._index.identity != self._index_identity:
            raise ValueError("plan index identity changed during trainer construction")
        self._assert_sources()
        if payload is not None:
            self.restore(payload)
        self.setup_seconds = time.monotonic() - started

    @property
    def index_identity(self):
        self._assert_sources()
        return copy.deepcopy(self._index_identity)

    @property
    def setup_report(self):
        self._assert_sources()
        return dict(setup_seconds=self.setup_seconds, index_preparation_seconds_included=self.index_preparation_seconds,
            index_reused=True, index_construction=self._index.construction,
            scope="Current setup checks the already authenticated process index and constructs the model, plus optional restore. Historical index construction is descriptive, not additional work in this setup.")

    def _build(self):
        model = build_sequence_student(self._seed, device=self._device, config=self._config)
        return model, torch.optim.AdamW(model.parameters(), lr=self._learning_rate)

    def _assert_sources(self):
        if source_hashes() != self._sources:
            raise ValueError("objective foundation training source identity changed")

    def snapshot(self):
        result = super().snapshot()
        result["schema"] = SCHEMA
        return result

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
            rows = legacy._materialize_validated_bundle(self._plan, bundle_id)
            bundle = legacy._rows_evidence(self._plan, bundle_id, rows, self._protected_set)
            materialization_seconds = time.monotonic() - started
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            for family in FAMILIES:
                stage = f"microbatch:{family}"
                batch = legacy.pack_composition_episodes(rows[family], device=self._device, training=True,
                    pair_validator=validate_pair, max_turns=self._config.max_turns,
                    max_input_bytes=self._config.max_input_bytes, max_context_tokens=self._config.max_positions,
                    max_reply_bytes=self._config.max_output_bytes)
                neural_microbatches += 1
                neural_episodes += len(rows[family])
                output = self.model(**batch["inputs"], decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
                losses = objective(self._objective_id, output, batch)
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
            stage = "post_optimizer_source_identity"
            self._assert_sources()
            seconds = time.monotonic() - started
            evidence = legacy._accumulate(self._evidence, bundle)
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

    def restore(self, payload):
        """Validate a new-schema snapshot and indexed prefix before atomic swap.

        The tensor/AdamW checks intentionally match frozen FoundationTrainer.
        Canonical input evidence comes from the authenticated process index;
        optimizer arithmetic is validated structurally, not recomputed.
        """
        started = time.monotonic()
        self._assert_sources()
        fields = {"schema", "recipe", "weights", "optimizer", "cursor", "evidence", "timing"}
        if type(payload) is not dict or set(payload) != fields or payload["schema"] != SCHEMA:
            raise ValueError("objective checkpoint fields/schema differ; old or variant checkpoints are not migrated")
        _check_finite_tree(payload, "objective foundation checkpoint")
        if legacy._json(payload["recipe"]) != legacy._json(self._recipe):
            raise ValueError("objective resume objective/source/index/plan/order/config differs")
        if self._index.identity != self._index_identity:
            raise ValueError("authenticated in-process index identity changed")
        cursor = payload["cursor"]
        _integer("checkpoint cursor", cursor)
        expected_evidence = self._index.replay(self._order, cursor)
        if legacy._json(payload["evidence"]) != legacy._json(expected_evidence):
            raise ValueError("checkpoint consumed IDs/recipes/counts differ from authenticated index")
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
                or legacy._json(state["param_groups"]) != legacy._json(optimizer.state_dict()["param_groups"])
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
        self._assert_sources()
        # The source/tensor/index preflight above leaves the old trainer intact.
        # Interrupted final assignments poison it rather than exposing a mixture.
        self._failed = True
        self.model, self.optimizer = model, optimizer
        self._evidence, self._timing = restored_evidence, restored_timing
        self.last_report = None
        self.last_restore_seconds = time.monotonic()-started
        self._failed = False
        return self
