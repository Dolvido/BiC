"""Explicit repeated curriculum handoffs through the existing durable loop.

The parent must come from a completed exact variant or cycle loop. Training arithmetic,
prescribed requests, teacher-free scoring and publication ownership are unchanged.
The same evaluation banks are required for a comparable child baseline. Parent
evidence is bound in the identity, but old retention alarms are not reinterpreted
as child-loop alarms. Persistent parent capsules and source migration are separate APIs.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time

from experiments import foundation_provider as base
from experiments import foundation_variant_provider as variant
from experiments.foundation_cycle_training import CompletedParent, CycleFoundationTrainer
from experiments.foundation_cycle_training import source_hashes as cycle_sources
from experiments.foundation_evidence import json_digest
from experiments.sequence_student import SequenceConfig
from experiments.train_cognitive import _check_finite_tree


SCHEMA = "bic-foundation-cycle-practice-provider-v2"


def source_hashes():
    return {**variant.source_hashes(), **cycle_sources(),
            "experiments/foundation_cycle_provider.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class CycleFoundationPracticeProvider(variant.VariantFoundationPracticeProvider):
    """One explicitly admitted fresh cycle, with complete optimizer continuity."""

    def __init__(self, plan, order="curriculum", *, parent, evaluation_banks,
                 admission_protected_transcripts, protected_transcripts, admission_receipt,
                 plan_index=None, device="cpu", payload=None):
        started = time.monotonic()
        if type(parent) is not CompletedParent:
            raise ValueError("an authenticated process-owned completed parent is required")
        self._parent = parent
        self._cycle_sources = source_hashes()
        recipe = parent.trainer_recipe
        # The superclass admits all banks before calling our trainer factory.
        # Its original source map is retained for its own source guard.
        super().__init__(plan, order, architecture=recipe["architecture"]["name"],
            seed=recipe["seed"], config=SequenceConfig(**recipe["config"]),
            learning_rate=recipe["learning_rate"], device=device,
            evaluation_banks=evaluation_banks,
            admission_protected_transcripts=admission_protected_transcripts,
            protected_transcripts=protected_transcripts, admission_receipt=admission_receipt,
            plan_index=plan_index, payload=None)
        self._identity.update(schema=SCHEMA, source_sha256=copy.deepcopy(self._cycle_sources),
            parent_identity=parent.identity,
            counter_scope="Provider updates and evaluation producers are cycle-local; lifetime updates are separate.")
        self._identity_hash = json_digest(self._identity)
        self._check()
        if payload is not None:
            self.restore(payload)
        self.setup_seconds = time.monotonic()-started

    def _new_trainer(self, payload=None):
        if not base._same(self._specs, self._parent.evaluation_specs):
            raise ValueError("handoff requires the exact parent evaluation roles, names and bank identities")
        return CycleFoundationTrainer(self._plan, order=self._order, parent=self._parent,
            device=self._device, admission_protected_transcripts=self._history,
            protected_transcripts=self._protected, admission_receipt=self._receipt,
            plan_index=self._index, payload=payload)

    def _check(self, *, allow_failed=False):
        super()._check(allow_failed=allow_failed)
        if source_hashes() != self._cycle_sources:
            raise RuntimeError("cycle provider source identity changed")
        # Access checks the parent token's captured source identity.
        self._parent.identity

    @property
    def parent_identity(self):
        with self._lock:
            self._check()
            return self._parent.identity

    @property
    def parent_evidence(self):
        """Detached prior evidence; never relabeled as child-cycle alarms."""
        with self._lock:
            self._check()
            return dict(identity=self._parent.identity,
                        final_evaluation=self._parent.final_evaluation,
                        retention_references=self._parent.retention_references,
                        reference_anchor=self._parent.reference_anchor)

    def status(self):
        with self._lock:
            result = super().status()
            result.update(base_updates=self._trainer.base_updates,
                          lifetime_updates=self._trainer.lifetime_updates, cycle=self._trainer.cycle)
            return result

    def snapshot(self):
        with self._lock:
            result = super().snapshot()
            result["schema"] = SCHEMA
            return result

    def restore(self, payload):
        with self._lock:
            self._check(allow_failed=True)
            payload = copy.deepcopy(payload)
            if (type(payload) is not dict or set(payload) != {"schema", "identity", "learner", "pending"}
                    or payload["schema"] != SCHEMA or not base._same(payload["identity"], self._identity)
                    or type(payload["learner"]) is not dict or "cursor" not in payload["learner"]):
                raise ValueError("cycle provider checkpoint identity or fields differ")
            _check_finite_tree(payload, "cycle provider checkpoint")
            self._validate_pending(payload["pending"], payload["learner"]["cursor"])
            candidate = self._new_trainer(payload["learner"])
            pending = copy.deepcopy(payload["pending"])
            self._validate_pending(pending, candidate.cursor)
            self._check(allow_failed=True)
            self._failed = True
            self._trainer, self._pending = candidate, pending
            self.last_report = None
            self._failed = False
            return self
