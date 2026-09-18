"""Opt-in, process-owned canonical evidence for a finite foundation plan.

Construction independently authenticates naming admission, then scans the full
admitted plan against expanded protection and global transcript uniqueness.
Those are separate passes. Only compact immutable evidence remains in memory;
later lookups never regenerate lessons. No file loader, model, optimizer,
training integration or persistent-cache trust is provided.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

from experiments import foundation_admission as admission
from experiments import foundation_plan as planning
from experiments import foundation_training as training
from experiments.foundation_curriculum import FAMILIES, VERSION
from experiments.foundation_evidence import json_digest, transcript_set
from experiments.foundation_plan import _materialize_validated_bundle
from experiments.realization_banks import transcript_digest


SCHEMA = "bic-authenticated-foundation-plan-index-v1"
ORDERS = ("curriculum", "mixed")


def source_hashes():
    """Bind the existing replay closure plus every added admission/index source."""
    root = Path(__file__).resolve().parents[1]
    names = set(training.source_hashes()) | {
        "experiments/foundation_plan_index.py", "experiments/foundation_admission.py",
        "experiments/foundation_evidence.py", "experiments/composition_training.py",
        "experiments/train_cognitive.py"}
    for package in ("experiments", "brain_in_computer"):
        name = f"{package}/__init__.py"
        if (root / name).exists():
            names.add(name)
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sorted(names)}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf8")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class AuthenticatedPlanIndex:
    """Immutable evidence authenticated in this process, never from a sidecar.

    Caller inputs and returned dictionaries have no aliases into retained state.
    Source-file identity is checked at construction and each lookup boundary.
    This is not a sandbox against arbitrary code modifying Python internals.
    Its finite memory use scales with the declared plan, not training lifetime.
    """

    _sources: bytes
    _identity: bytes
    _construction: bytes
    _records: tuple[bytes, ...]
    _schedules: tuple[tuple[int, ...], ...]
    _prefixes: tuple[tuple[bytes, ...], ...]

    def __init__(self, plan, *, admission_protected_transcripts,
                 protected_transcripts, admission_receipt):
        started = time.monotonic()
        sources = _encoded(source_hashes())
        plan, receipt = copy.deepcopy(plan), copy.deepcopy(admission_receipt)
        planning.validate_plan(plan)
        if type(receipt) is not dict:
            raise ValueError("an explicit canonical admission receipt is required")
        history = transcript_set(admission_protected_transcripts)
        protection = transcript_set(protected_transcripts)
        if not history <= protection:
            raise ValueError("historical admission protection must be included in expanded protection")
        base = planning.build_plan(**plan["config"])
        admission_started = time.monotonic()
        authenticated = admission.authenticate_admission(base, plan,
            protected_transcripts=sorted(history), receipt=receipt)
        admission_seconds = time.monotonic() - admission_started

        scan_started = time.monotonic()
        records, seen = [], set()
        targets = {family: dict.fromkeys(map(str, range(4)), 0) for family in FAMILIES}
        for bundle_id in range(len(plan["bundles"])):
            rows = _materialize_validated_bundle(plan, bundle_id)
            # The unchanged helper authenticates complete pairs, provenance,
            # every target/reply, declared cell, exact bytes and protection.
            record = training._rows_evidence(plan, bundle_id, rows, protection)
            for family in FAMILIES:
                for row in rows[family]:
                    digest = transcript_digest(row)
                    if digest in seen:
                        raise ValueError("admitted plan contains a duplicate exact transcript")
                    seen.add(digest)
                    for turn in row["turns"]:
                        targets[family][str(turn["target"])] += 1
            records.append(_encoded(record))
        if (len(seen) != authenticated["counts"]["unique_accepted_transcripts"]
                or json_digest(sorted(seen)) != authenticated["accepted_transcripts_sha256"]):
            raise ValueError("expanded-protection scan differs from authenticated admitted transcripts")
        scan_seconds = time.monotonic() - scan_started

        prefix_started = time.monotonic()
        schedules = tuple(tuple(plan["schedules"][order]) for order in ORDERS)
        prefixes = []
        for schedule in schedules:
            evidence = training._empty_evidence()
            values = [_encoded(evidence)]
            for bundle_id in schedule:
                evidence = training._accumulate(evidence, json.loads(records[bundle_id]))
                values.append(_encoded(evidence))
            prefixes.append(tuple(values))
        prefix_seconds = time.monotonic() - prefix_started
        identity = _encoded({"schema": SCHEMA, "curriculum_version": VERSION,
            "source_sha256": json.loads(sources), "plan_sha256": json_digest(plan),
            "admission_receipt_sha256": json_digest(authenticated),
            "admission_protection_sha256": json_digest(sorted(history)),
            "admission_protection_count": len(history),
            "expanded_protection_sha256": json_digest(sorted(protection)),
            "expanded_protection_count": len(protection),
            "bundle_count": len(records), "micro_batch_size": plan["config"]["micro_batch_size"],
            "unique_transcripts": len(seen), "transcript_sha256": json_digest(sorted(seen)),
            "bundle_records_sha256": json_digest([json.loads(value) for value in records]),
            "order_prefix_sha256": {order: json_digest([json.loads(value) for value in values])
                                    for order, values in zip(ORDERS, prefixes)}})
        if _encoded(source_hashes()) != sources:
            raise RuntimeError("foundation plan index sources changed during construction")
        report = _encoded({"admission_seconds": admission_seconds,
            "admitted_plan_scan_seconds": scan_seconds, "prefix_fold_seconds": prefix_seconds,
            "construction_wall_seconds": time.monotonic() - started,
            "admission_counts": authenticated["counts"],
            "admitted_scan_bundles": len(records), "admitted_scan_episodes": len(seen),
            "target_counts": targets,
            "record_payload_bytes": sum(map(len, records)),
            "prefix_payload_bytes": sum(len(value) for values in prefixes for value in values),
            "scope": "Admission reconstruction and the admitted-plan scan are separate passes. Timings are nested in construction wall time and exclude imports. Payload byte counts exclude Python/container overhead. No model, inference, optimizer, training or persistent loader."})
        for name, value in (("_sources", sources), ("_identity", identity),
                            ("_construction", report), ("_records", tuple(records)),
                            ("_schedules", schedules), ("_prefixes", tuple(prefixes))):
            object.__setattr__(self, name, value)

    def _check_sources(self):
        if _encoded(source_hashes()) != self._sources:
            raise RuntimeError("foundation plan index source identity changed")

    def _isolated(self, value):
        self._check_sources()
        result = json.loads(value)
        self._check_sources()
        return result

    @property
    def identity(self):
        return self._isolated(self._identity)

    @property
    def construction(self):
        return self._isolated(self._construction)

    def replay(self, order, cursor, *, include_bundles=False):
        """Return exact original replay evidence without lesson materialization."""
        self._check_sources()
        if type(order) is not str or order not in ORDERS:
            raise ValueError("order must be curriculum or mixed")
        if type(cursor) is not int or not 0 <= cursor <= len(self._records):
            raise ValueError("cursor must be a bounded integer in the declared plan")
        if type(include_bundles) is not bool:
            raise ValueError("include_bundles must be Boolean")
        index = ORDERS.index(order)
        evidence = json.loads(self._prefixes[index][cursor])
        if include_bundles:
            evidence = {"evidence": evidence, "bundles": [json.loads(self._records[bundle_id])
                for bundle_id in self._schedules[index][:cursor]]}
        self._check_sources()
        return evidence

    def __reduce_ex__(self, protocol):
        raise TypeError("authenticated plan indexes are process-owned and cannot be serialized")
