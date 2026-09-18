"""Explicit architecture variants with process-authenticated foundation replay.

The unchanged FoundationTrainer.step supplies the objective, materialization,
clipping, AdamW update and failed-step accounting. Only model construction,
checkpoint identity and canonical restore evidence differ. A complete admitted
plan index must authenticate before any model exists. Reuse requires the exact
plan/admission/protection identity; this module loads no persistent index.

Restore below is a deliberately versioned copy of the frozen trainer preflight:
its global checkpoint schema and replay function have no overridable hook.
No global monkeypatch or implicit old-checkpoint migration is used. New flat
arithmetic is compatible with old flat training, but old checkpoints are not.
Index preparation, model setup and restore cost remain separate from step time.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time

import torch

from brain_in_computer.learning_student import _check_finite_tree, _cpu_copy, _finite_number, _integer
from experiments import foundation_training as legacy
from experiments.foundation_evidence import json_digest
from experiments.foundation_plan_index import AuthenticatedPlanIndex, source_hashes as index_sources
from experiments.sequence_student import build_sequence_student

SCHEMA = "bic-foundation-variant-trainer-v1"
ARCHITECTURES = {
    "flat": "bic-flat-sequence-v1",
    "hierarchical": "bic-hierarchical-sequence-v1",
}


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("experiments/foundation_variant_training.py", "experiments/hierarchical_sequence_student.py")
    return {**index_sources(), **{name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}}


class VariantFoundationTrainer(legacy.FoundationTrainer):
    def __init__(self, plan, order="curriculum", *, architecture, seed,
                 admission_protected_transcripts, protected_transcripts, admission_receipt,
                 plan_index=None, config=None, learning_rate=.001, device="cpu", payload=None):
        started = time.monotonic()
        if type(architecture) is not str or architecture not in ARCHITECTURES:
            raise ValueError("explicit flat or hierarchical architecture required")
        admitted = legacy._admit_plan(plan, order)
        history = legacy._protected(admission_protected_transcripts)
        protection = legacy._protected(protected_transcripts)
        if not set(history) <= set(protection):
            raise ValueError("original admission protection must be included in expanded protection")
        if type(admission_receipt) is not dict:
            raise ValueError("explicit canonical admission receipt required")
        self._architecture = architecture
        self._index_reused = plan_index is not None
        preparation_started = time.monotonic()
        if plan_index is None:
            plan_index = AuthenticatedPlanIndex(admitted,
                admission_protected_transcripts=history, protected_transcripts=protection,
                admission_receipt=admission_receipt)
        elif type(plan_index) is not AuthenticatedPlanIndex:
            raise ValueError("a real in-process AuthenticatedPlanIndex is required")
        identity = plan_index.identity
        expected = {"plan_sha256": json_digest(admitted),
            "admission_receipt_sha256": json_digest(admission_receipt),
            "admission_protection_sha256": json_digest(list(history)), "admission_protection_count": len(history),
            "expanded_protection_sha256": json_digest(list(protection)), "expanded_protection_count": len(protection),
            "bundle_count": len(admitted["bundles"]), "micro_batch_size": admitted["config"]["micro_batch_size"]}
        if any(identity.get(key) != value for key,value in expected.items()):
            raise ValueError("plan index belongs to another plan/admission/protection contract")
        self._index, self._index_identity = plan_index, identity
        self.index_preparation_seconds = time.monotonic()-preparation_started
        # No model construction occurs until the full index has been admitted.
        super().__init__(admitted, order, seed=seed, config=config, learning_rate=learning_rate,
                         device=device, protected_transcripts=protection, payload=None)
        self._sources = source_hashes()
        self._recipe.update(schema=SCHEMA, source_sha256=copy.deepcopy(self._sources),
            architecture={"name": architecture, "schema": ARCHITECTURES[architecture]},
            plan_index_identity=copy.deepcopy(identity))
        if self._index.identity != self._index_identity:
            raise ValueError("plan index identity changed during trainer construction")
        self._assert_sources()
        if payload is not None:
            self.restore(payload)
        self.setup_seconds = time.monotonic()-started

    @property
    def index_identity(self):
        self._assert_sources()
        return copy.deepcopy(self._index_identity)

    @property
    def setup_report(self):
        self._assert_sources()
        return {"setup_seconds": self.setup_seconds,
            "index_preparation_seconds_included": self.index_preparation_seconds,
            "index_reused": self._index_reused,
            "index_construction": self._index.construction,
            "scope": "Setup includes new index admission or existing index checks plus model creation and optional restore. Historical index construction is descriptive and must not be added when reused. No speedup claim."}

    def _build(self):
        if self._architecture == "flat":
            model = build_sequence_student(self._seed, device=self._device, config=self._config)
        else:
            from experiments.hierarchical_sequence_student import ARCHITECTURE, build_hierarchical_sequence_student
            if ARCHITECTURE != ARCHITECTURES[self._architecture]:
                raise ValueError("hierarchical model architecture version differs")
            model = build_hierarchical_sequence_student(self._seed, device=self._device, config=self._config)
        return model, torch.optim.AdamW(model.parameters(), lr=self._learning_rate)

    def _assert_sources(self):
        # This closure includes the index's own closure. Lookup also checks its
        # immutable source identity; no admission or lesson scan is repeated.
        if source_hashes() != self._sources:
            raise ValueError("variant foundation training source identity changed")

    def snapshot(self):
        result = super().snapshot()
        result["schema"] = SCHEMA
        return result

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
            raise ValueError("variant checkpoint fields/schema differ; old flat checkpoints are not migrated")
        _check_finite_tree(payload, "variant foundation checkpoint")
        if legacy._json(payload["recipe"]) != legacy._json(self._recipe):
            raise ValueError("variant resume architecture/source/index/plan/order/config differs")
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
