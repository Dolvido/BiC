"""Opt-in architecture-aware provider for the unchanged FoundationLoop.

Only construction, source identity and provider checkpoint admission differ
from FoundationPracticeProvider. Its prescribed requests, teacher-free scoring,
historical evidence checks and physical-work accounting are inherited unchanged.
The existing loop owns the sole durable envelope. No old provider or learner
checkpoint is migrated, and no learned study weights are adopted implicitly.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import threading
import time

import torch

from experiments import foundation_provider as base
from experiments import foundation_plan as planning
from experiments.foundation_curriculum import FAMILIES, VERSION
from experiments.foundation_evaluation import FoundationBank
from experiments.foundation_evidence import json_digest, transcript_set
from experiments.foundation_metrics import _canonical, _cell
from experiments.foundation_plan_index import AuthenticatedPlanIndex
from experiments.foundation_variant_training import ARCHITECTURES, VariantFoundationTrainer, source_hashes as variant_sources
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig
from experiments.train_cognitive import _check_finite_tree


SCHEMA = "bic-foundation-variant-practice-provider-v1"
# Response semantics are unchanged; provider_sha256 binds the new architecture.
EVALUATION_SCHEMA = base.EVALUATION_SCHEMA


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = set(base.source_hashes()) | set(variant_sources()) | {"experiments/foundation_variant_provider.py"}
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(names)}


class VariantFoundationPracticeProvider(base.FoundationPracticeProvider):
    """A distinct provider identity; same finite loop-facing behavior.

    An optional index must be the actual immutable AuthenticatedPlanIndex built
    in this process. Serialized metadata never substitutes for that admission.
    Candidate restores share that index; setup and model/optimizer restore costs
    remain measured and are not claimed to vanish.
    """

    def __init__(self, plan, order="curriculum", *, architecture, seed, evaluation_banks,
                 admission_protected_transcripts, protected_transcripts, admission_receipt,
                 plan_index=None, config=None, learning_rate=.001, device="cpu", payload=None):
        started = time.monotonic()
        if type(architecture) is not str or architecture not in ARCHITECTURES:
            raise ValueError("explicit flat or hierarchical provider architecture required")
        planning.validate_plan(plan)
        if type(order) is not str or order not in plan["schedules"]:
            raise ValueError("declared foundation order required")
        if config is not None and type(config) is not SequenceConfig:
            raise ValueError("explicit SequenceConfig required")
        if type(admission_receipt) is not dict:
            raise ValueError("explicit canonical admission receipt required")
        self._lock = threading.RLock()
        self._sources = source_hashes()
        self._device = torch.device(device)
        self._runtime = base.runtime_identity(self._device)
        self._plan, self._receipt = copy.deepcopy(plan), copy.deepcopy(admission_receipt)
        history, protection = transcript_set(admission_protected_transcripts), transcript_set(protected_transcripts)
        if not history <= protection:
            raise ValueError("original admission protection must be a subset of trainer protection")
        self._history, self._protected = tuple(sorted(history)), tuple(sorted(protection))
        self._config = config or SequenceConfig(max_turns=12)
        self._order, self._seed, self._learning_rate = order, seed, learning_rate
        self._architecture = architecture

        tick = time.monotonic()
        self._index_reused = plan_index is not None
        if plan_index is None:
            plan_index = AuthenticatedPlanIndex(self._plan, admission_receipt=self._receipt,
                admission_protected_transcripts=self._history, protected_transcripts=self._protected)
        if type(plan_index) is not AuthenticatedPlanIndex:
            raise ValueError("a real in-process AuthenticatedPlanIndex is required")
        identity = plan_index.identity
        expected = dict(plan_sha256=json_digest(self._plan), admission_receipt_sha256=json_digest(self._receipt),
            admission_protection_sha256=json_digest(list(self._history)), admission_protection_count=len(history),
            expanded_protection_sha256=json_digest(list(self._protected)), expanded_protection_count=len(protection),
            bundle_count=len(self._plan["bundles"]), micro_batch_size=self._plan["config"]["micro_batch_size"])
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError("provider index belongs to another plan/admission/protection boundary")
        self._index, self._index_identity = plan_index, identity
        self._index_preparation_seconds = time.monotonic()-tick

        tick = time.monotonic()
        if type(evaluation_banks) is not dict or set(evaluation_banks) != set(base.ROLES):
            raise ValueError("exact development and retention bank roles required; no audits")
        self._banks, self._canonical_metrics, self._specs = {}, {}, {}
        for role in base.ROLES:
            named = evaluation_banks[role]
            if type(named) is not dict or not named:
                raise ValueError("each evaluation role requires nonempty named banks")
            self._banks[role], self._canonical_metrics[role], self._specs[role] = {}, {}, {}
            for name in sorted(named):
                rows = copy.deepcopy(named[name])
                bank_identity, counts, opposite = _canonical(rows, name, self._config, "dev")
                if any(transcript_digest(row) not in protection for row in rows):
                    raise ValueError("all evaluation transcripts must be in trainer protection")
                bank = FoundationBank(rows, role="dev", config=self._config)
                if not base._same(bank_identity, bank.identity):
                    raise ValueError("foundation preparation and metric identities differ")
                _, family, depth, mechanism, turns = _cell(name)
                self._banks[role][name] = bank
                self._canonical_metrics[role][name] = (bank_identity, counts, opposite)
                self._specs[role][name] = dict(identity=bank_identity,
                    cell=dict(family=family, depth=int(depth[1:]), mechanism=mechanism, turns=int(turns)))
            if {row["cell"]["family"] for row in self._specs[role].values()} != set(FAMILIES):
                raise ValueError("each evaluation role must cover all three declared families")
        self._bank_preparation_seconds = time.monotonic()-tick

        # Every plan, protection and evaluation check above precedes this model.
        self._trainer = self._new_trainer()
        self._initial_trainer_setup = self._trainer.setup_report
        self._identity = dict(schema=SCHEMA, curriculum_version=VERSION,
            architecture=dict(name=architecture, schema=ARCHITECTURES[architecture]),
            source_sha256=self._sources, runtime=self._runtime, plan_sha256=json_digest(self._plan), order=order,
            admission_receipt_sha256=json_digest(self._receipt),
            admission_protection_sha256=json_digest(list(self._history)), admission_protection_count=len(history),
            trainer_protection_sha256=json_digest(list(self._protected)), trainer_protection_count=len(protection),
            evaluation_specs=self._specs, training_identity=dict(authenticated_plan_index=self._index_identity),
            trainer_recipe=self._trainer.recipe, default_chunk_updates=base.DEFAULT_CHUNK_UPDATES,
            maximum_chunk_updates=base.MAX_CHUNK_UPDATES)
        self._identity_hash = json_digest(self._identity)
        self._pending, self._failed, self.last_report = None, False, None
        self._check()
        if payload is not None:
            self.restore(payload)
        self.setup_seconds = time.monotonic()-started

    @property
    def setup_report(self):
        self._check()
        return dict(setup_seconds=self.setup_seconds, index_reused=self._index_reused,
            index_preparation_seconds_included=self._index_preparation_seconds,
            bank_preparation_seconds_included=self._bank_preparation_seconds,
            index_construction=self._index.construction,
            trainer_setup_at_initialization=copy.deepcopy(self._initial_trainer_setup),
            scope="Setup includes index admission/checks, canonical evaluation preparation, model setup and optional restore. Component times are nested. Reused index construction is historical and must not be added again. No persistent cached-index trust or speedup claim.")

    def _new_trainer(self, payload=None):
        return VariantFoundationTrainer(self._plan, self._order, architecture=self._architecture,
            seed=self._seed, config=self._config, learning_rate=self._learning_rate, device=self._device,
            admission_protected_transcripts=self._history, protected_transcripts=self._protected,
            admission_receipt=self._receipt, plan_index=self._index, payload=payload)

    def _check(self, *, allow_failed=False):
        if self._failed and not allow_failed:
            raise RuntimeError("failed variant provider requires explicit transactional restoration")
        if source_hashes() != self._sources or not base._same(base.runtime_identity(self._device), self._runtime):
            raise RuntimeError("variant foundation provider source or runtime identity changed")

    def snapshot(self):
        with self._lock:
            self._check()
            learner = self._trainer.snapshot()
            self._validate_pending(self._pending, learner["cursor"])
            return dict(schema=SCHEMA, identity=self.identity, learner=learner, pending=copy.deepcopy(self._pending))

    def restore(self, payload):
        with self._lock:
            self._check(allow_failed=True)
            payload = copy.deepcopy(payload)
            if (type(payload) is not dict or set(payload) != {"schema", "identity", "learner", "pending"}
                    or payload["schema"] != SCHEMA or not base._same(payload["identity"], self._identity)
                    or type(payload["learner"]) is not dict or "cursor" not in payload["learner"]):
                raise ValueError("variant provider checkpoint identity or fields differ")
            _check_finite_tree(payload, "variant provider checkpoint")
            self._validate_pending(payload["pending"], payload["learner"]["cursor"])
            candidate = self._new_trainer(copy.deepcopy(payload["learner"]))
            pending = copy.deepcopy(payload["pending"])
            self._validate_pending(pending, candidate.cursor)
            self._check(allow_failed=True)
            # Candidate construction/validation never mutates the live learner.
            # An interrupted final swap stays poisoned until explicit recovery.
            self._failed = True
            self._trainer, self._pending = candidate, pending
            self.last_report = None
            self._failed = False
            return self
