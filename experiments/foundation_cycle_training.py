"""Explicit repeated handoffs from completed foundation loops to fresh practice.

The inherited step keeps the original objective and AdamW arithmetic. Only the
finite-cycle coordinate is reset; complete weights and moments are carried.
This module supplies no curriculum policy, source migration, persisted parent
loader, promotion or claim of improved learning. Parent validation
authenticates inputs and structural state, not recomputed historical gradients.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import time

import torch

from brain_in_computer.learning_student import _check_finite_tree, _cpu_copy, _finite_number, _integer
from experiments import foundation_loop as loops
from experiments import foundation_provider as providers
from experiments import foundation_training as legacy
from experiments import foundation_variant_provider as variant_provider
from experiments import foundation_variant_training as variant
from experiments.foundation_evidence import json_digest, transcript_set
from experiments.sequence_student import SequenceConfig

SCHEMA = "bic-foundation-cycle-trainer-v2"
PARENT_SCHEMA = "bic-completed-foundation-parent-v2"


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = set(variant_provider.source_hashes()) | set(loops.source_hashes()) | {
        "experiments/foundation_cycle_training.py", "experiments/foundation_cycle_provider.py",
        "experiments/foundation_parent_capsule.py"}
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(names)}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def _tensor_tree_digest(value):
    """Deterministic identity independent of pickle container metadata."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return json_digest(["tensor", str(tensor.dtype), list(tensor.shape),
            hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()])
    if type(value) is dict:
        return json_digest(["dict", [[type(key).__name__, key, _tensor_tree_digest(item)]
            for key, item in sorted(value.items(), key=lambda pair: (type(pair[0]).__name__, str(pair[0]))) ]])
    if type(value) in (tuple, list):
        return json_digest([type(value).__name__, [_tensor_tree_digest(item) for item in value]])
    return json_digest([type(value).__name__, value])


@dataclass(frozen=True, slots=True, init=False, eq=False)
class CompletedParent:
    """Process-owned completion evidence; no public checkpoint/sidecar loader.

    All retained payloads are immutable bytes/tuples and public metadata is
    detached. One parent learner image is retained, never recursive tensor
    ancestry. Python-internals mutation is outside this evidence boundary.
    Exact Variant and current Cycle providers are accepted. No v1 token or
    checkpoint is migrated. Without the separate caller-pinned capsule API,
    rebuilding a saved chain requires earlier completed loop envelopes and its
    validation cost grows with cycle count.
    """

    _sources: bytes
    _identity: bytes
    _learner_image: bytes
    _metadata: bytes
    _training_transcripts: tuple[str, ...]
    _protected_transcripts: tuple[str, ...]

    def __init__(self, loop, *, training_transcripts):
        sources = _encoded(source_hashes())
        # Imported only at the capture boundary to avoid a module import cycle.
        from experiments.foundation_cycle_provider import CycleFoundationPracticeProvider
        if type(loop) is not loops.FoundationLoop or type(loop.provider) not in (
                variant_provider.VariantFoundationPracticeProvider, CycleFoundationPracticeProvider):
            raise ValueError("completed exact FoundationLoop with exact Variant or Cycle provider required")
        if type(training_transcripts) is not list or training_transcripts != sorted(transcript_set(training_transcripts)):
            raise ValueError("parent training transcripts require a sorted unique list")
        training_transcripts = tuple(training_transcripts)
        provider = loop.provider
        with loop._lock, provider._lock:
            saved = loop.snapshot()
            state, supplied = saved["state"], saved["provider"]
            if (state["phase"] != "complete" or supplied["pending"] is not None
                    or state["evaluation"] is not None or not state["history"]
                    or supplied["learner"]["cursor"] != len(provider._plan["bundles"])):
                raise ValueError("parent requires exhausted training and complete final evaluation")
            final = state["history"][-1]
            if not loops._same(final["producer"], loops._producer(supplied)):
                raise ValueError("final evaluation does not describe the current parent learner")
            continued = type(provider) is CycleFoundationPracticeProvider
            expected_type = CycleFoundationTrainer if continued else variant.VariantFoundationTrainer
            if type(provider._trainer) is not expected_type:
                raise ValueError("exact matching Variant or Cycle trainer required")
            index = provider._trainer.index_identity
            if (len(training_transcripts) != index["unique_transcripts"]
                    or json_digest(list(training_transcripts)) != index["transcript_sha256"]):
                raise ValueError("parent training transcript inventory differs from authenticated index")
            # Original restore checks all tensors, moments, counters and index
            # evidence on a candidate, without modifying the actual loop.
            candidate = provider._new_trainer(copy.deepcopy(supplied["learner"]))
            learner = candidate.snapshot()
            if not loops._same_tree(learner, supplied["learner"]):
                raise ValueError("validated parent learner differs from committed loop state")
            lifetime = candidate.lifetime_updates if continued else candidate.updates
            ordinal = candidate.cycle if continued else 1
            predecessor = provider._parent.identity if continued else None
            exclusions = tuple(sorted(set(provider._protected) | set(training_transcripts)))
            metadata = dict(provider_identity=saved["provider_identity"],
                evaluation_specs=provider.specs(), final_evaluation=final,
                retention_references=state["references"], reference_anchor=state["reference_anchor"],
                trainer_recipe=learner["recipe"], parent_work=state["work"],
                parent_timing=learner["timing"], device=str(provider._device))
            identity = dict(schema=PARENT_SCHEMA, source_sha256=json.loads(sources),
                provider_sha256=json_digest(saved["provider_identity"]),
                loop_source_sha256=saved["source_sha256"], loop_options=saved["options"],
                loop_envelope_sha256=_tensor_tree_digest(saved),
                parent_envelope=dict(kind="durable" if loop._disk_path is not None else "memory",
                    file_sha256=loop._disk_digest),
                plan_sha256=index["plan_sha256"], index_identity=index,
                base_updates=lifetime, cycle=ordinal,
                cycle_updates=learner["cursor"], inherited_updates=lifetime-learner["cursor"],
                predecessor_identity_sha256=None if predecessor is None else json_digest(predecessor),
                predecessor_loop_envelope_sha256=None if predecessor is None else predecessor["loop_envelope_sha256"],
                weights_sha256=_tensor_tree_digest(learner["weights"]),
                optimizer_sha256=_tensor_tree_digest(learner["optimizer"]),
                learner_metadata_sha256=json_digest({k: v for k, v in learner.items() if k not in ("weights", "optimizer")}),
                metadata_sha256=json_digest(metadata),
                excluded_transcripts_count=len(exclusions), excluded_transcripts_sha256=json_digest(list(exclusions)))
            buffer = io.BytesIO()
            torch.save(learner, buffer)
            # Recheck ownership, sources and the completed envelope after all
            # candidate work; same-process mutations cannot be silently carried.
            if not loops._same_tree(loop.snapshot(), saved) or _encoded(source_hashes()) != sources:
                raise RuntimeError("parent state or sources changed during capture")
        for name, value in (("_sources", sources), ("_identity", _encoded(identity)),
                ("_learner_image", buffer.getvalue()), ("_metadata", _encoded(metadata)),
                ("_training_transcripts", training_transcripts), ("_protected_transcripts", exclusions)):
            object.__setattr__(self, name, value)

    @classmethod
    def from_loop(cls, loop, *, training_transcripts):
        if cls is not CompletedParent:
            raise ValueError("exact CompletedParent type required")
        return cls(loop, training_transcripts=training_transcripts)

    def _check(self):
        if _encoded(source_hashes()) != self._sources:
            raise RuntimeError("completed parent source identity changed")

    def _field(self, name):
        self._check()
        return json.loads(self._metadata)[name]

    @property
    def identity(self):
        self._check()
        return json.loads(self._identity)

    @property
    def training_transcripts(self):
        self._check()
        return list(self._training_transcripts)

    @property
    def protected_transcripts(self):
        self._check()
        return list(self._protected_transcripts)

    @property
    def evaluation_specs(self): return self._field("evaluation_specs")

    @property
    def final_evaluation(self): return self._field("final_evaluation")

    @property
    def retention_references(self): return self._field("retention_references")

    @property
    def reference_anchor(self): return self._field("reference_anchor")

    @property
    def provider_identity(self): return self._field("provider_identity")

    @property
    def trainer_recipe(self): return self._field("trainer_recipe")

    def _learner(self):
        self._check()
        result = torch.load(io.BytesIO(self._learner_image), map_location="cpu", weights_only=True)
        self._check()
        return result

    def __reduce_ex__(self, protocol):
        raise TypeError("completed parents are process-owned and cannot be serialized")


def capture_completed_parent(loop, *, training_transcripts):
    return CompletedParent.from_loop(loop, training_transcripts=training_transcripts)


class CycleFoundationTrainer(variant.VariantFoundationTrainer):
    """Fresh finite plan, full inherited learner state, explicit lifetime steps.

    ``cursor`` and ``updates`` remain cycle-local for finite-provider accounting.
    ``base_updates`` and ``lifetime_updates`` expose retained optimizer history.
    Optional contract arguments must equal the completed parent's values.
    """

    def __init__(self, plan, order="curriculum", *, parent,
                 admission_protected_transcripts, protected_transcripts, admission_receipt,
                 plan_index=None, architecture=None, seed=None, config=None,
                 learning_rate=None, device=None, payload=None):
        started = time.monotonic()
        if type(parent) is not CompletedParent:
            raise ValueError("real in-process CompletedParent required")
        self._cycle_sources = source_hashes()
        self._parent, self._parent_identity = parent, parent.identity
        inherited = parent.trainer_recipe
        architecture = inherited["architecture"]["name"] if architecture is None else architecture
        seed = inherited["seed"] if seed is None else seed
        config = SequenceConfig(**inherited["config"]) if config is None else config
        learning_rate = inherited["learning_rate"] if learning_rate is None else learning_rate
        if (type(architecture) is not str or architecture != inherited["architecture"]["name"]
                or type(seed) is not int or seed != inherited["seed"]
                or type(config) is not SequenceConfig or config != SequenceConfig(**inherited["config"])
                or type(learning_rate) not in (float, int) or learning_rate != inherited["learning_rate"]):
            raise ValueError("cycle architecture/seed/config/learning rate must match parent")
        device = parent._field("device") if device is None else device
        self._cycle_runtime = providers.runtime_identity(torch.device(device))
        if not loops._same(self._cycle_runtime, parent.provider_identity["runtime"]):
            raise ValueError("cycle runtime must match parent; migration is not implemented")
        history, protection = transcript_set(admission_protected_transcripts), transcript_set(protected_transcripts)
        if not set(parent.protected_transcripts) <= history or not history <= protection:
            raise ValueError("cycle admission must protect all parent training and protected transcripts")
        admitted = legacy._admit_plan(plan, order)
        if json_digest(admitted) == self._parent_identity["plan_sha256"]:
            raise ValueError("cycle requires a fresh admitted plan")
        self._base_updates = self._parent_identity["base_updates"]
        self._cycle = self._parent_identity["cycle"]+1
        super().__init__(admitted, order, architecture=architecture, seed=seed,
            admission_protected_transcripts=sorted(history), protected_transcripts=sorted(protection),
            admission_receipt=admission_receipt, plan_index=plan_index, config=config,
            learning_rate=learning_rate, device=device, payload=None)
        for name in ("architecture", "seed", "config", "learning_rate", "optimizer", "objective", "gradient_clip", "family_order"):
            if not loops._same(self._recipe[name], inherited[name]):
                raise ValueError("cycle learner contract differs from parent: "+name)
        self._recipe.update(schema=SCHEMA, source_sha256=copy.deepcopy(self._cycle_sources),
            parent_identity=copy.deepcopy(self._parent_identity), base_updates=self._base_updates,
            cycle=self._cycle,
            runtime=copy.deepcopy(self._cycle_runtime), parent_evaluation_specs=parent.evaluation_specs,
            continuity="same weights, complete AdamW state and objective; fresh cycle-local evidence")
        parent_state = parent._learner()
        self.model.load_state_dict(parent_state["weights"], strict=True)
        self.optimizer.load_state_dict(_cpu_copy(parent_state["optimizer"]))
        self._assert_sources()
        if payload is not None:
            self.restore(payload)
        self.setup_seconds = time.monotonic()-started

    @property
    def base_updates(self): return self._base_updates

    @property
    def lifetime_updates(self): return self._base_updates+self.cursor

    @property
    def cycle(self): return self._cycle

    @property
    def parent_identity(self):
        self._assert_sources()
        return copy.deepcopy(self._parent_identity)

    def _assert_sources(self):
        super()._assert_sources()
        if source_hashes() != self._cycle_sources:
            raise ValueError("cycle training source identity changed")
        if not loops._same(providers.runtime_identity(self._device), self._cycle_runtime):
            raise ValueError("cycle runtime identity changed")

    def snapshot(self):
        result = legacy.FoundationTrainer.snapshot(self)
        result.update(schema=SCHEMA, base_updates=self.base_updates, lifetime_updates=self.lifetime_updates, cycle=self.cycle)
        return result

    def restore(self, payload):
        """Detach, fully preflight, then atomically replace the retained state."""
        started = time.monotonic()
        self._assert_sources()
        payload = copy.deepcopy(payload)
        fields = {"schema", "recipe", "weights", "optimizer", "cursor", "evidence", "timing", "base_updates", "lifetime_updates", "cycle"}
        if type(payload) is not dict or set(payload) != fields or payload["schema"] != SCHEMA:
            raise ValueError("cycle checkpoint fields/schema differ")
        _check_finite_tree(payload, "cycle checkpoint")
        if legacy._json(payload["recipe"]) != legacy._json(self._recipe):
            raise ValueError("cycle resume parent/source/index/plan/contract differs")
        for name in ("cursor", "base_updates", "lifetime_updates", "cycle"):
            _integer(name, payload[name])
        cursor = payload["cursor"]
        lifetime = self.base_updates+cursor
        if payload["base_updates"] != self.base_updates or payload["lifetime_updates"] != lifetime or payload["cycle"] != self.cycle:
            raise ValueError("cycle ordinal/base/lifetime counters differ")
        if self._index.identity != self._index_identity:
            raise ValueError("cycle authenticated index identity changed")
        evidence = self._index.replay(self._order, cursor)
        if legacy._json(payload["evidence"]) != legacy._json(evidence):
            raise ValueError("cycle consumed evidence differs from authenticated index")
        timing = payload["timing"]
        if type(timing) is not dict or set(timing) != set(self._timing):
            raise ValueError("cycle timing fields differ")
        for name, value in timing.items():
            _finite_number(name, value)
            if value < 0 or cursor == 0 and value != 0:
                raise ValueError("cycle timing cannot be negative or precede local work")
        if timing["materialization_seconds_included_in_step"] > timing["retained_step_seconds"]:
            raise ValueError("materialization is a subset of step time")
        model, optimizer = self._build()
        expected_weights, weights = model.state_dict(), payload["weights"]
        if type(weights) is not dict or set(weights) != set(expected_weights):
            raise ValueError("cycle model parameter set differs")
        for name, expected in expected_weights.items():
            value = weights[name]
            if (not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype
                    or value.device.type != "cpu" or value.requires_grad):
                raise ValueError("cycle model tensor shape/dtype/CPU boundary differs")
        aliases = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            first = aliases.setdefault(id(parameter), name)
            if not torch.equal(weights[name], weights[first]):
                raise ValueError("cycle tied parameter aliases differ")
        state = payload["optimizer"]
        if (type(state) is not dict or set(state) != {"state", "param_groups"}
                or legacy._json(state["param_groups"]) != legacy._json(optimizer.state_dict()["param_groups"])
                or type(state["state"]) is not dict):
            raise ValueError("cycle AdamW configuration differs")
        parameters, moments = list(model.parameters()), state["state"]
        if any(type(index) is not int for index in moments) or set(moments) != set(range(len(parameters))):
            raise ValueError("cycle optimizer state/update coverage differs")
        for index, values in moments.items():
            if type(values) is not dict or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("cycle AdamW moment fields differ")
            step = values["step"]
            if (not isinstance(step, torch.Tensor) or step.numel() != 1 or not step.is_floating_point()
                    or step.device.type != "cpu" or step.requires_grad or float(step) != lifetime):
                raise ValueError("cycle optimizer step differs from lifetime updates")
            for name in ("exp_avg", "exp_avg_sq"):
                value = values[name]
                if (not isinstance(value, torch.Tensor) or value.shape != parameters[index].shape
                        or value.dtype != parameters[index].dtype or value.device.type != "cpu" or value.requires_grad):
                    raise ValueError("cycle optimizer moment shape/dtype/CPU boundary differs")
            if bool(values["exp_avg_sq"].lt(0).any()):
                raise ValueError("cycle second moment must be nonnegative")
        if cursor == 0:
            parent = self._parent._learner()
            if not loops._same_tree(weights, parent["weights"]) or not loops._same_tree(state, parent["optimizer"]):
                raise ValueError("zero-cycle checkpoint must equal authenticated parent weights and moments")
        model.load_state_dict(weights, strict=True)
        optimizer.load_state_dict(_cpu_copy(state))
        self._assert_sources()
        self._failed = True
        self.model, self.optimizer = model, optimizer
        self._evidence, self._timing = copy.deepcopy(evidence), copy.deepcopy(timing)
        self.last_report = None
        self.last_restore_seconds = time.monotonic()-started
        self._failed = False
        return self
