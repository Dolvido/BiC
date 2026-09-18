"""Conservative, offline import boundary for the optional local tutor pilot.

This parser labels saved training data only. It is never an inference policy.
"""
from __future__ import annotations

import hashlib
import json
import re

RELATIONS = ("find", "left", "right", "above", "below")
MAX_INPUT_BYTES = 128
MAX_REPLY_BYTES = 64
MAX_CANDIDATE_BYTES = 2048
CELL_NAMES = ("top left", "top right", "bottom left", "bottom right")
QUOTED_NAME = re.compile(r'"(?:[^"\\]|\\.)*"')

# Explicit whole sentences: adding a wording family is a reviewed code change.
# No keyword matching, fuzzy matching, LLM judge, or silent meaning truncation.
GRAMMAR = {
    "choose_item": {
        "find": "choose the item called {quoted_name}.",
        "left": "choose the item immediately to the left of {quoted_name}.",
        "right": "choose the item immediately to the right of {quoted_name}.",
        "above": "choose the item immediately above {quoted_name}.",
        "below": "choose the item immediately below {quoted_name}.",
    },
    "pick_object": {
        "find": "pick the object named {quoted_name}.",
        "left": "pick the object directly to the left of {quoted_name}.",
        "right": "pick the object directly to the right of {quoted_name}.",
        "above": "pick the object directly above {quoted_name}.",
        "below": "pick the object directly below {quoted_name}.",
    },
    "please_select_item": {
        "find": "please select the item named {quoted_name}.",
        "left": "please select the item directly to the left of {quoted_name}.",
        "right": "please select the item directly to the right of {quoted_name}.",
        "above": "please select the item directly above {quoted_name}.",
        "below": "please select the item directly below {quoted_name}.",
    },
}

CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "lesson_id": {"type": "string"},
        "instruction": {"type": "string"},
        "relation": {"type": "string", "enum": list(RELATIONS)},
    },
    "required": ["lesson_id", "instruction", "relation"],
    "additionalProperties": False,
}


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def strict_json(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON constant")
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    # Reject overflow-to-infinity and lone UTF-16 surrogates anywhere, including
    # unrecognized envelope fields, before hashes or output serialization.
    canonical_json(value).encode("utf-8", errors="strict")
    return value


def _visible_name(name):
    return (isinstance(name, str) and bool(name) and name == name.strip()
            and all(c.isprintable() for c in name) and len(name.encode("utf-8")) <= 32)


def verified_label(spec: dict) -> dict:
    """Recompute the task outcome from scene facts, not proposed model labels."""
    from experiments.train_regional_memory import relation_target
    if not isinstance(spec, dict) or spec.get("relation") not in RELATIONS:
        raise ValueError("invalid simulator relation")
    if not _visible_name(spec.get("name")):
        raise ValueError("invalid simulator name")
    if type(spec.get("known")) is not bool:
        raise ValueError("invalid simulator known flag")
    objects, query = spec.get("objects"), spec.get("query")
    if (not isinstance(objects, list) or len(objects) != 4
            or any(type(v) is not int or not 0 <= v < 65536 for v in objects)
            or type(query) is not int or not 0 <= query < 65536):
        raise ValueError("invalid simulator identities")
    matches = [i for i, value in enumerate(objects) if value == query]
    if not spec["known"]:
        category, target = "unknown_label", 10
    elif not matches:
        category, target = "absent", 10
    elif len(matches) > 1:
        category, target = "ambiguous", 10
    else:
        target = relation_target(matches[0], spec["relation"])
        category = "boundary" if target == 10 else "valid"
    reply = "cannot select." if target == 10 else f"selected {CELL_NAMES[target]}."
    if len(reply.encode("utf-8")) > MAX_REPLY_BYTES:
        raise ValueError("canonical reply exceeds byte limit")
    return {"target": target, "category": category, "reply": reply}


def validate_candidate(candidate, spec: dict) -> dict:
    """Return a deterministic decision; no model-supplied target is accepted."""
    result = {"accepted": False, "lesson_id": spec["lesson_id"]}
    def reject(reason):
        return dict(result, reason=reason)
    try:
        raw = candidate if isinstance(candidate, str) else canonical_json(candidate)
        if len(raw.encode("utf-8")) > MAX_CANDIDATE_BYTES:
            return reject("candidate_byte_limit")
        value = strict_json(raw)
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return reject("invalid_json")
    if not isinstance(value, dict) or set(value) != set(CANDIDATE_SCHEMA["required"]):
        return reject("schema_fields")
    if any(not isinstance(value[k], str) for k in CANDIDATE_SCHEMA["required"]):
        return reject("schema_types")
    if value["lesson_id"] != spec["lesson_id"]:
        return reject("lesson_id_mismatch")
    if value["relation"] not in RELATIONS:
        return reject("unsupported_relation")
    instruction = value["instruction"]
    try:
        if len(instruction.encode("utf-8")) > MAX_INPUT_BYTES:
            return reject("instruction_byte_limit")
    except UnicodeError:
        return reject("invalid_utf8")
    quoted = list(QUOTED_NAME.finditer(instruction))
    if len(quoted) != 1:
        return reject("quoted_name_count")
    match = quoted[0]
    if '"' in instruction[:match.start()] + instruction[match.end():]:
        return reject("unbalanced_quotes")
    try:
        if strict_json(match.group()) != spec["name"]:
            return reject("quoted_name_mismatch")
    except ValueError:
        return reject("invalid_quoted_name")
    name_literal = json.dumps(spec["name"], ensure_ascii=False)
    meanings = [(family, relation) for family, rules in GRAMMAR.items()
                for relation, rule in rules.items()
                if instruction == rule.format(quoted_name=name_literal)]
    if len(meanings) != 1:
        return reject("unapproved_wording")
    family, meaning = meanings[0]
    if meaning != value["relation"] or meaning != spec["relation"]:
        return reject("relation_mismatch")
    label = verified_label(spec)
    return {"accepted": True, "lesson_id": spec["lesson_id"], "instruction": instruction,
            "relation": meaning, "phrase_family": family, **label,
            "candidate_sha256": digest(value), "spec_sha256": digest(spec)}
