"""Bounded retention evidence across authenticated completed curriculum cycles.

Best references survive cycle boundaries, while local evaluation producers gain
explicit lifetime coordinates. Counts are observations, not mastery/promotion
rules. Restore requires an external pin to the exact saved ledger; it does not
recompute past neural scores or authenticate a self-declared history.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

from experiments.foundation_cycle_training import CompletedParent, source_hashes as parent_sources
from experiments.foundation_evidence import json_digest


SCHEMA = "bic-foundation-lifetime-retention-v1"
FIELDS = ("paired_action", "paired_reply", "known", "unsupported_ask")


def source_hashes():
    return {**parent_sources(), "experiments/foundation_lifetime_retention.py":
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name+" must be a bounded nonnegative integer")


def _metric(value, field):
    numerator = "count" if field == "unsupported_ask" else "correct"
    if type(value) is not dict or set(value) != {numerator, "total", "rate"}:
        raise ValueError("retention metric fields differ")
    for key in (numerator, "total"):
        _integer(value[key], key)
    if value[numerator] > value["total"]:
        raise ValueError("retention numerator exceeds denominator")
    if value["total"] == 0:
        if value["rate"] is not None:
            raise ValueError("unavailable retention denominator requires null rate")
    elif (type(value["rate"]) not in (int, float) or not math.isfinite(value["rate"])
          or not math.isclose(value["rate"], value[numerator]/value["total"], rel_tol=0., abs_tol=1e-7)):
        raise ValueError("retention rate disagrees with counts")
    return value[numerator], value["total"]


def _better(left, right, field):
    a, total = _metric(left["value"], field)
    b, other = _metric(right["value"], field)
    if total != other:
        raise ValueError("the same bank must retain the same metric denominator")
    return total > 0 and (a < b if field == "unsupported_ask" else a > b)


def _frame(parent):
    if type(parent) is not CompletedParent:
        raise ValueError("an authenticated completed parent is required")
    identity = parent.identity
    parent_hash = json_digest(identity)
    final, specs, references = parent.final_evaluation, parent.evaluation_specs, parent.retention_references
    cycle, lifetime = identity["cycle"], identity["base_updates"]
    local = final["producer"]["updates"]
    _integer(cycle, "cycle", 1); _integer(lifetime, "lifetime", 1); _integer(local, "local", 1)
    inherited = lifetime-local
    if (inherited < 0 or identity["cycle_updates"] != local
            or identity["inherited_updates"] != inherited):
        raise ValueError("parent lifetime precedes its local work")
    banks = {f"{role}:{name}": spec for role, named in specs.items() for name, spec in named.items()}
    if not banks or set(final["responses"]) != set(banks) or set(references) != set(banks):
        raise ValueError("complete final and reference bank inventories required")

    def record(key, field, producer, response):
        role, name = key.split(":", 1)
        step, weights = producer["updates"], producer["weights_sha256"]
        _integer(step, "reference local update")
        if (step > local or not _sha(weights) or response["updates"] != step
                or response["weights_sha256"] != weights or response["role"] != role
                or response["provider_sha256"] != identity["provider_sha256"]
                or response["control"] != "normal" or set(response["per_bank"]) != {name}
                or set(response["progress"]) != {name}):
            raise ValueError("reference producer, role or provider differs")
        value = copy.deepcopy(response["progress"][name][field])
        _metric(value, field)
        return dict(value=value, producer=dict(cycle=cycle, local_updates=step,
            lifetime_updates=inherited+step, weights_sha256=weights,
            provider_sha256=identity["provider_sha256"], parent_identity_sha256=parent_hash),
            response_sha256=json_digest(response))

    endpoint, best = {}, {}
    for key in sorted(banks):
        if set(references[key]) != set(FIELDS):
            raise ValueError("all reference fields required")
        endpoint[key], best[key] = {}, {}
        for field in FIELDS:
            ref = references[key][field]
            endpoint[key][field] = record(key, field, final["producer"], final["responses"][key])
            best[key][field] = record(key, field, ref["producer"], ref["response"])
            if _better(endpoint[key][field], best[key][field], field):
                raise ValueError("parent reference omits a better final observation")
    if parent.identity != identity:
        raise RuntimeError("parent identity changed during retention capture")
    return dict(cycle=cycle, lifetime_updates=lifetime, inherited_updates=inherited,
        parent_identity_sha256=parent_hash, predecessor_identity_sha256=identity["predecessor_identity_sha256"],
        bank_specs_sha256=json_digest(specs), banks=banks, endpoint=endpoint, cycle_best=best)


def _alarms(endpoint, best):
    result = []
    for key in sorted(endpoint):
        for field in FIELDS:
            current, reference = endpoint[key][field], best[key][field]
            if _better(reference, current, field):
                result.append(dict(bank=key, field=field, current=copy.deepcopy(current),
                    best=copy.deepcopy(reference), direction="lower" if field == "unsupported_ask" else "higher"))
    return result


def _fold(prior, frame, sources):
    """Pure count fold. Only the public parent adapter supplies authenticated frames."""
    if prior is None:
        if frame["cycle"] != 1 or frame["predecessor_identity_sha256"] is not None or frame["inherited_updates"] != 0:
            raise ValueError("lifetime retention must begin at the original completed cycle")
        best = copy.deepcopy(frame["cycle_best"])
        first = frame["parent_identity_sha256"]
        predecessor = json_digest([SCHEMA, "origin"])
    else:
        if (frame["cycle"] != prior["cycle"]+1
                or frame["predecessor_identity_sha256"] != prior["last_parent_identity_sha256"]
                or frame["inherited_updates"] != prior["lifetime_updates"]
                or frame["lifetime_updates"] <= prior["lifetime_updates"]):
            raise ValueError("retention cycle lineage or lifetime boundary differs")
        if frame["bank_specs_sha256"] != prior["bank_specs_sha256"] or frame["banks"] != prior["banks"]:
            raise ValueError("retention roles, names and bank identities must remain identical")
        best = copy.deepcopy(prior["best"])
        for key in best:
            for field in FIELDS:
                candidate = frame["cycle_best"][key][field]
                if _better(candidate, best[key][field], field):
                    best[key][field] = copy.deepcopy(candidate)
        first, predecessor = prior["first_parent_identity_sha256"], prior["chain_sha256"]
    return dict(schema=SCHEMA, source_sha256=copy.deepcopy(sources), cycle=frame["cycle"],
        lifetime_updates=frame["lifetime_updates"], first_parent_identity_sha256=first,
        last_parent_identity_sha256=frame["parent_identity_sha256"], bank_specs_sha256=frame["bank_specs_sha256"],
        banks=copy.deepcopy(frame["banks"]), best=best, endpoint=copy.deepcopy(frame["endpoint"]),
        alarms=_alarms(frame["endpoint"], best),
        chain_sha256=json_digest([predecessor, frame["parent_identity_sha256"], json_digest(frame["cycle_best"])]))


def _validate_state(state, frame, sources):
    fields = {"schema", "source_sha256", "cycle", "lifetime_updates", "first_parent_identity_sha256",
        "last_parent_identity_sha256", "bank_specs_sha256", "banks", "best", "endpoint", "alarms", "chain_sha256"}
    if (type(state) is not dict or set(state) != fields or state["schema"] != SCHEMA
            or state["source_sha256"] != sources or state["cycle"] != frame["cycle"]
            or state["lifetime_updates"] != frame["lifetime_updates"]
            or state["last_parent_identity_sha256"] != frame["parent_identity_sha256"]
            or state["banks"] != frame["banks"] or state["bank_specs_sha256"] != frame["bank_specs_sha256"]
            or state["endpoint"] != frame["endpoint"]):
        raise ValueError("saved lifetime retention identity or endpoint differs")
    _integer(state["cycle"], "saved cycle", 1)
    _integer(state["lifetime_updates"], "saved lifetime", 1)
    for key in ("first_parent_identity_sha256", "last_parent_identity_sha256", "bank_specs_sha256", "chain_sha256"):
        if not _sha(state[key]): raise ValueError("retention SHA-256 identity required")
    if type(state["best"]) is not dict or set(state["best"]) != set(frame["banks"]):
        raise ValueError("saved best bank inventory differs")
    for key, values in state["best"].items():
        if type(values) is not dict or set(values) != set(FIELDS):
            raise ValueError("saved best field inventory differs")
        for field, ref in values.items():
            if type(ref) is not dict or set(ref) != {"value", "producer", "response_sha256"} or not _sha(ref["response_sha256"]):
                raise ValueError("saved reference fields differ")
            producer = ref["producer"]
            if type(producer) is not dict or set(producer) != {"cycle", "local_updates", "lifetime_updates", "weights_sha256", "provider_sha256", "parent_identity_sha256"}:
                raise ValueError("saved producer fields differ")
            for name in ("cycle", "local_updates", "lifetime_updates"):
                _integer(producer[name], name, 1 if name == "cycle" else 0)
            if (producer["cycle"] > state["cycle"] or not producer["local_updates"] <= producer["lifetime_updates"] <= state["lifetime_updates"]
                    or any(not _sha(producer[name]) for name in ("weights_sha256", "provider_sha256", "parent_identity_sha256"))):
                raise ValueError("saved producer lies outside completed lineage")
            if _better(frame["cycle_best"][key][field], ref, field):
                raise ValueError("saved lifetime best omits a better current-cycle observation")
            if producer["cycle"] == frame["cycle"]:
                if ref != frame["cycle_best"][key][field]:
                    raise ValueError("current-cycle best must match authenticated current references")
            elif producer["lifetime_updates"] > frame["inherited_updates"]:
                raise ValueError("earlier-cycle reference exceeds its lifetime boundary")
    if state["alarms"] != _alarms(state["endpoint"], state["best"]):
        raise ValueError("saved alarms disagree with best references")
    if state["cycle"] == 1 and state != _fold(None, frame, sources):
        raise ValueError("first-cycle retention must exactly match its original observations")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class LifetimeRetention:
    _state: bytes

    def __init__(self, parent):
        sources = source_hashes()
        state = _fold(None, _frame(parent), sources)
        if source_hashes() != sources: raise RuntimeError("retention sources changed")
        object.__setattr__(self, "_state", _encoded(state))

    @classmethod
    def start(cls, parent):
        if cls is not LifetimeRetention: raise ValueError("exact LifetimeRetention type required")
        return cls(parent)

    def snapshot(self):
        state = json.loads(self._state)
        if source_hashes() != state["source_sha256"]:
            raise RuntimeError("retention source identity changed")
        return state

    def advance(self, parent):
        prior = self.snapshot()
        state = _fold(prior, _frame(parent), prior["source_sha256"])
        self.snapshot()
        result = object.__new__(LifetimeRetention)
        object.__setattr__(result, "_state", _encoded(state))
        return result

    @classmethod
    def restore(cls, payload, *, expected_sha256, parent):
        state = copy.deepcopy(payload)
        if cls is not LifetimeRetention or not _sha(expected_sha256) or json_digest(state) != expected_sha256:
            raise ValueError("external pin to exact lifetime retention payload required")
        sources = source_hashes()
        _validate_state(state, _frame(parent), sources)
        if source_hashes() != sources: raise RuntimeError("retention sources changed")
        result = object.__new__(cls)
        object.__setattr__(result, "_state", _encoded(state))
        return result

    def __reduce_ex__(self, protocol):
        raise TypeError("use detached snapshot plus explicit pinned restore for lifetime retention")
