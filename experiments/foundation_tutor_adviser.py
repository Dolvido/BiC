"""Persist one bounded local curriculum choice, outside learner execution.

The caller authenticates completed-parent and development evidence, freezes the
request hash, and retains the returned decision hash in its durable owner. This
module neither establishes that provenance independently nor teaches a learner.
Choices are complete canonical curriculum profiles, never individual weak cells.
No learned scheduling policy, benefit, promotion or independence is claimed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
import time

from brain_in_computer import curriculum_tutor as transport
from experiments import foundation_plan as planning


SCHEMA = "bic-foundation-tutor-decision-v1"
REQUEST_SCHEMA = "bic-foundation-tutor-request-v1"
CATALOGUE_SCHEMA = "bic-foundation-tutor-catalogue-v1"
EVIDENCE_SCHEMA = "bic-foundation-tutor-development-v1"
MAX_BYTES = 131072
FAMILIES = ("color", "count", "switch")
METRICS = ("anchor_pair_action", "anchor_pair_reply", "anchor_pair_both",
    "known_action", "known_reply", "other_known_action", "other_known_reply",
    "unknown_action", "unknown_reply", "known_unsupported_ask", "known_reply_unsupported_ask")
SCOPE = ("External engineering adviser chooses one caller-frozen whole curriculum. "
         "Caller owns parent/development authenticity and learning execution. "
         "No learned self-direction, learning benefit, automatic promotion or independence claim.")


class UnknownTutorRequest(RuntimeError):
    """An intent may include physical teacher work; never automatically replay."""


class DecisionPinRequired(RuntimeError):
    """A completed orphan is not automatically adopted from its own contents."""


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def request_sha256(request):
    return hashlib.sha256(_json(request).encode("utf-8")).hexdigest()


def _pin(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(name+" fields differ")


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**63-1:
        raise ValueError(name+" must be a bounded integer")


def _native(path):
    absolute = os.path.abspath(path)
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        absolute = "\\\\?\\UNC\\"+absolute[2:] if absolute.startswith("\\\\") else "\\\\?\\"+absolute
    return Path(absolute)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("brain_in_computer/__init__.py", "brain_in_computer/curriculum.py",
        "brain_in_computer/curriculum_tutor.py", "experiments/foundation_plan.py",
        "experiments/foundation_curriculum.py", "experiments/composition_curriculum.py",
        "experiments/cognitive_curriculum.py", "experiments/foundation_layout_curriculum.py",
        "experiments/foundation_tutor_adviser.py")
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def _catalogue(value):
    _keys(value, ("schema", "profiles"), "catalogue")
    profiles = value["profiles"]
    if value["schema"] != CATALOGUE_SCHEMA or type(profiles) is not list or not 2 <= len(profiles) <= 4:
        raise ValueError("two to four fixed curriculum profiles required")
    identifiers, variants, configurations, coverage = set(), set(), [], {}
    for profile in profiles:
        _keys(profile, ("id", "order", "layout", "plan_config"), "profile")
        identifier = profile["id"]
        if (type(identifier) is not str or re.fullmatch(r"[a-z][a-z0-9_]{0,47}", identifier) is None
                or identifier in identifiers or profile["order"] not in ("curriculum", "mixed")
                or profile["layout"] not in ("original", "varied")
                or (profile["order"], profile["layout"]) in variants):
            raise ValueError("unique fixed profile identifiers/order/layout choices required")
        identifiers.add(identifier); variants.add((profile["order"], profile["layout"]))
        config = profile["plan_config"]
        _keys(config, ("stage_updates", "final_updates", "micro_batch_size", "rehearsal_every"), "profile plan")
        # This is the actual canonical plan recipe, not a caller's coverage claim.
        plan = planning.build_plan(seed=0, ordering_seed=0, **config)
        schedule = plan["schedules"][profile["order"]]
        bundles = [plan["bundles"][str(identifier)] for identifier in schedule]
        wanted = {(depth, turns) for depth in planning.DEPTHS for turns in planning.TURN_BUCKETS}
        cells = Counter((row["depth"], row["turns"]) for row in bundles)
        tail = Counter((row["depth"], row["turns"]) for row in bundles if row["phase"] == "mixed")
        if set(cells) != wanted or set(tail) != wanted or len(set(tail.values())) != 1:
            raise ValueError("every depth/length and equal complete mixed-tail coverage required")
        rehearsals = {}
        for stage in planning.DEPTHS[1:]:
            selected = [row for row in bundles if row["stage"] == stage and row["depth"] < stage]
            counts = Counter(row["depth"] for row in selected)
            if set(counts) != set(range(stage)) or min(counts.values()) < 1:
                raise ValueError("each original stage must reserve every earlier depth")
            rehearsals[str(stage)] = {str(depth): counts[depth] for depth in sorted(counts)}
        configurations.append(config)
        coverage[profile["id"]] = dict(families=list(FAMILIES), updates=len(bundles),
            episodes=len(bundles)*3*config["micro_batch_size"],
            depth_turn_bundles={f"d{d}/t{t}": cells[(d,t)] for d,t in sorted(wanted)},
            mixed_tail_depth_turn_bundles={f"d{d}/t{t}": tail[(d,t)] for d,t in sorted(wanted)},
            earlier_depth_bundles_by_original_stage=rehearsals,
            rehearsal_scope="Canonical stage reserves are exact. Mixed order permutes those bundles and does not preserve chronological stage cadence.")
    if any(config != configurations[0] for config in configurations[1:]):
        raise ValueError("catalogue profiles must have identical plan configuration and exposure budget")
    return coverage


def _development(value, parent):
    _keys(value, ("schema", "role", "current", "reference"), "development evidence")
    if value["schema"] != EVIDENCE_SCHEMA or value["role"] != "development":
        raise ValueError("development-only evidence required; no audit role")
    for name in ("current", "reference"):
        panel = value[name]
        _keys(panel, ("weights_sha256", "lifetime_updates", "by_family"), name+" panel")
        if not _pin(panel["weights_sha256"]):
            raise ValueError("explicit producing weights required")
        _integer(panel["lifetime_updates"], "evidence updates")
        if panel["lifetime_updates"] > parent["lifetime_updates"]:
            raise ValueError("development evidence cannot follow parent")
        if name == "current" and any(panel[k] != parent[k] for k in ("weights_sha256", "lifetime_updates")):
            raise ValueError("current development producer differs from completed parent")
        _keys(panel["by_family"], FAMILIES, "development families")
        for family, metrics in panel["by_family"].items():
            _keys(metrics, METRICS, "development metrics")
            for metric, counts in metrics.items():
                _keys(counts, ("count", "total"), "metric counts")
                _integer(counts["count"], metric+" count"); _integer(counts["total"], metric+" total")
                if counts["count"] > counts["total"]:
                    raise ValueError("metric count exceeds denominator")
            for names in (("anchor_pair_action", "anchor_pair_reply", "anchor_pair_both"),
                          ("known_action", "known_reply", "known_unsupported_ask", "known_reply_unsupported_ask"),
                          ("other_known_action", "other_known_reply"), ("unknown_action", "unknown_reply")):
                if len({metrics[key]["total"] for key in names}) != 1:
                    raise ValueError("corresponding action/reply denominators differ")
            if metrics["anchor_pair_action"]["total"] < 1 or metrics["unknown_action"]["total"] < 1:
                raise ValueError("each family needs anchor and unknown development evidence")
            if metrics["anchor_pair_both"]["count"] > min(metrics[k]["count"] for k in ("anchor_pair_action", "anchor_pair_reply")):
                raise ValueError("joint paired count exceeds either modality")


def _chat_payload(value, coverage):
    menu = [profile["id"] for profile in value["catalogue"]["profiles"]]
    payload = dict(model=value["teacher"]["model"], stream=False, keep_alive=0,
        format=dict(type="object", additionalProperties=False, required=["profile_id"],
            properties=dict(profile_id=dict(type="string", enum=menu))),
        messages=[dict(role="system", content="Select one exact profile_id from the frozen whole-curriculum catalogue using development evidence. Return only that JSON field. Do not add lessons, labels, commands, code, new facts or withdrawal choices."),
                  dict(role="user", content=_json(dict(catalogue=value["catalogue"], coverage=coverage,
                      development=value["development"], parent=value["parent"],
                      procedural_profile_id=value["procedural_profile_id"])))],
        options=dict(temperature=0, seed=17092026, num_ctx=4096, num_predict=64))
    # Reject locally before intent or any request, so oversize input is never
    # misreported as uncertain physical generation.
    if len(transport._canonical_json(payload).encode("utf-8")) > transport.MAX_REQUEST_BYTES:
        raise ValueError("local tutor payload exceeds transport byte limit")
    return payload


def validate_request(request, *, expected_sha256):
    """Validate a caller-pinned detached recipe, not past learner execution."""
    if not _pin(expected_sha256):
        raise ValueError("caller-pinned request SHA256 required")
    raw = _json(request).encode("utf-8")
    if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("request bytes differ from caller pin or exceed limit")
    value = json.loads(raw)
    _keys(value, ("schema", "parent", "catalogue", "development", "procedural_profile_id",
                  "mode", "teacher", "max_seconds", "source_sha256"), "request")
    if value["schema"] != REQUEST_SCHEMA or value["source_sha256"] != source_hashes():
        raise ValueError("request schema or current source contract differs")
    parent = value["parent"]
    _keys(parent, ("identity_sha256", "weights_sha256", "cycle", "lifetime_updates"), "parent")
    if not _pin(parent["identity_sha256"]) or not _pin(parent["weights_sha256"]):
        raise ValueError("explicit parent and weight digests required")
    _integer(parent["cycle"], "parent cycle", 1); _integer(parent["lifetime_updates"], "parent updates", 1)
    coverage = _catalogue(value["catalogue"])
    _development(value["development"], parent)
    if value["procedural_profile_id"] not in coverage or value["mode"] not in ("local", "procedural", "withdrawn"):
        raise ValueError("frozen procedural profile and explicit teacher mode required")
    seconds = value["max_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 300:
        raise ValueError("one positive at-most-300-second allowance required")
    if value["mode"] == "local":
        teacher = value["teacher"]
        _keys(teacher, ("model", "sha256"), "local teacher")
        transport.LocalTutor(model=teacher["model"], timeout=seconds, cache_size=0)  # validation only, no I/O
        if not _pin(teacher["sha256"]):
            raise ValueError("caller-pinned installed teacher digest required")
        _chat_payload(value, coverage)
    elif value["teacher"] is not None:
        raise ValueError("procedural and withdrawal modes must not configure a teacher")
    return value, coverage


def _publish(path, value):
    """Complete fsynced immutable JSON, atomically linked without replacement."""
    path = _native(path)
    raw = (_json(value)+"\n").encode("utf-8")
    if len(raw) > 2*MAX_BYTES:
        raise ValueError("persisted tutor record exceeds bound")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=path.name+".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


def _read(path, expected_sha256):
    if not _pin(expected_sha256):
        raise ValueError("external expected artifact hash required")
    with _native(path).open("rb") as stream:
        raw = stream.read(2*MAX_BYTES+1)
    if len(raw) > 2*MAX_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("persisted tutor artifact differs from caller pin")
    return transport._strict_json(raw.decode("utf-8"))


def load_decision(path, *, expected_sha256, expected_request_sha256):
    """Reuse an externally pinned published choice; performs no service calls."""
    path = _native(path)
    decision = _read(path, expected_sha256)
    _keys(decision, ("schema", "request_sha256", "intent_sha256", "parent", "catalogue_sha256",
        "development_sha256", "profile_id", "profile", "coverage", "mode", "outcome", "rejection_reason",
        "teacher_cost", "source_sha256", "scope"), "decision")
    if decision["schema"] != SCHEMA or decision["request_sha256"] != expected_request_sha256:
        raise ValueError("decision request binding differs")
    intent = _read(path.parent/"intent.json", decision["intent_sha256"])
    _keys(intent, ("schema", "status", "request_sha256", "request", "coverage", "pid", "created_unix_seconds", "scope"), "intent")
    if intent["schema"] != SCHEMA or intent["status"] != "started" or intent["request_sha256"] != expected_request_sha256:
        raise ValueError("decision intent binding differs")
    request, coverage = validate_request(intent["request"], expected_sha256=expected_request_sha256)
    profiles = {p["id"]:p for p in request["catalogue"]["profiles"]}
    if (decision["profile_id"] not in profiles or decision["profile"] != profiles[decision["profile_id"]]
            or decision["coverage"] != coverage[decision["profile_id"]] or intent["coverage"] != coverage
            or decision["parent"] != request["parent"] or decision["source_sha256"] != request["source_sha256"]
            or decision["catalogue_sha256"] != request_sha256(request["catalogue"])
            or decision["development_sha256"] != request_sha256(request["development"])
            or decision["mode"] != request["mode"] or decision["scope"] != SCOPE):
        raise ValueError("persisted decision profile, coverage or evidence differs")
    outcomes = {"local": ("local_accepted", "procedural_fallback"),
                "procedural": ("procedural",), "withdrawn": ("teacher_withdrawn",)}
    if decision["outcome"] not in outcomes[request["mode"]]:
        raise ValueError("decision outcome differs from frozen mode")
    if decision["outcome"] != "local_accepted" and decision["profile_id"] != request["procedural_profile_id"]:
        raise ValueError("fallback or withdrawal differs from frozen procedural choice")
    cost = decision["teacher_cost"]
    response = None
    if cost["response_file_sha256"] is not None:
        response = _read(path.parent/"response.json", cost["response_file_sha256"])
        if type(response) is not dict or request_sha256(response) != cost["response_sha256"]:
            raise ValueError("saved untrusted response envelope differs")
    if cost["chat_completions"] == 1 and (response is None or response.get("done") is not True):
        raise ValueError("completed chat requires the exact saved response envelope")
    if decision["outcome"] == "local_accepted":
        if (cost["chat_attempts"] != 1 or cost["chat_completions"] != 1
                or cost["model_digest_before"] != request["teacher"]["sha256"]
                or cost["model_digest_after"] != request["teacher"]["sha256"]
                or cost["physical_teacher_work_unknown"] is not False or decision["rejection_reason"] is not None):
            raise ValueError("accepted teacher evidence is inconsistent")
        if _selection(response, request) != decision["profile_id"]:
            raise ValueError("accepted profile differs from saved untrusted response")
    return deepcopy(decision)


def _selection(response, value):
    if response.get("model") != value["teacher"]["model"] or response.get("done_reason", "stop") != "stop":
        raise ValueError("invalid generation envelope")
    message = response.get("message")
    if (type(message) is not dict or message.get("role") != "assistant" or message.get("tool_calls")
            or type(message.get("content")) is not str or len(message["content"].encode("utf-8")) > 512):
        raise ValueError("invalid generation message")
    selection = transport._strict_json(message["content"])
    _keys(selection, ("profile_id",), "selection")
    if type(selection["profile_id"]) is not str or selection["profile_id"] not in [p["id"] for p in value["catalogue"]["profiles"]]:
        raise ValueError("profile outside fixed catalogue")
    return selection["profile_id"]


def decide(directory, *, request, expected_request_sha256, expected_decision_sha256=None, deadline=None):
    """Attempt at most one local chat for this persisted decision slot.

    A completed choice requires its external hash on reuse. A published choice
    lost before owner commit is an orphan for explicit adoption, never a reason
    to request again. Withdrawal is caller-controlled and makes zero API calls.
    """
    started, cpu_started = time.monotonic(), time.process_time()
    value, coverage = validate_request(request, expected_sha256=expected_request_sha256)
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("finite absolute monotonic deadline required")
    directory = _native(directory)
    decision_path, intent_path = directory/"decision.json", directory/"intent.json"
    if expected_decision_sha256 is not None:
        decision = load_decision(decision_path, expected_sha256=expected_decision_sha256,
                                 expected_request_sha256=expected_request_sha256)
        return dict(decision=decision, decision_sha256=expected_decision_sha256, reused=True,
            invocation_cost=dict(api_attempts=0, chat_attempts=0, wall_seconds=time.monotonic()-started,
                                 cpu_seconds=time.process_time()-cpu_started))
    if decision_path.exists():
        raise DecisionPinRequired("existing decision requires its externally retained SHA256; no new request")
    if intent_path.exists():
        raise UnknownTutorRequest("existing request intent without an adopted decision; no automatic replay")
    expires = min(started+value["max_seconds"], deadline) if deadline is not None else started+value["max_seconds"]
    transport._remaining(expires)
    directory.mkdir(parents=True, exist_ok=True)
    intent = dict(schema=SCHEMA, status="started", request_sha256=expected_request_sha256,
        request=value, coverage=coverage, pid=os.getpid(), created_unix_seconds=time.time(), scope=SCOPE)
    # Atomic exclusivity is the cross-process winner check before any API call.
    intent_sha = _publish(intent_path, intent)
    cost = dict(api_attempts=0, api_completions=0, chat_attempts=0, chat_completions=0,
        api_events=[], model_digest_before=None, model_digest_after=None, response_sha256=None,
        response_file_sha256=None,
        usage=None, physical_teacher_work_unknown=False, keep_alive=0,
        runtime=dict(python=platform.python_version(), platform=platform.platform()),
        timing_scope="Decision interval includes request validation and source hashing; API times are nested. Final publication is in returned invocation_cost, not stored teacher_cost.")
    profile_id, rejection = value["procedural_profile_id"], None
    outcome = "teacher_withdrawn" if value["mode"] == "withdrawn" else "procedural"

    def call(route, payload=None):
        transport._remaining(expires)
        tick = time.monotonic()
        entry = dict(route=route, status="attempted")
        cost["api_events"].append(entry); cost["api_attempts"] += 1
        if route == "/api/chat":
            cost["chat_attempts"] += 1
            cost["physical_teacher_work_unknown"] = True
        try:
            response = transport._request_json(route, payload, expires)
            cost["api_completions"] += 1; entry["status"] = "returned"
            if route == "/api/chat" and response.get("done") is True:
                cost["chat_completions"] += 1
                cost["physical_teacher_work_unknown"] = False
            return response
        finally:
            entry["wall_seconds"] = time.monotonic()-tick

    def identity(*, show):
        teacher = value["teacher"]
        inventory = call("/api/tags")
        models = inventory.get("models")
        if type(models) is not list:
            raise ValueError("invalid local inventory")
        matches = [item for item in models if type(item) is dict and teacher["model"] in (item.get("name"), item.get("model"))]
        if (len(matches) != 1 or matches[0].get("digest") != teacher["sha256"]
                or transport._remote_metadata(matches[0])):
            raise ValueError("installed local model identity differs")
        if show:
            details = call("/api/show", {"model": teacher["model"]})
            if type(details.get("details")) is not dict or transport._remote_metadata(details):
                raise ValueError("local model details invalid or remote")
        return teacher["sha256"]

    def finish_cost():
        cost.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started)

    try:
        if value["mode"] == "local":
            outcome = "procedural_fallback"
            try:
                cost["model_digest_before"] = identity(show=True)
            except Exception:
                rejection = "pre_identity_unavailable_or_invalid"
            if rejection is None:
                payload = _chat_payload(value, coverage)
                try:
                    response = call("/api/chat", payload)
                except Exception as failure:
                    raise UnknownTutorRequest("local generation outcome is uncertain; intent retained and no retry") from failure
                # Raw model text remains untrusted data. Preserve the exact
                # returned envelope before interpreting or accepting a choice.
                cost["response_file_sha256"] = _publish(directory/"response.json", response)
                cost["response_sha256"] = request_sha256(response)
                if cost["physical_teacher_work_unknown"]:
                    raise UnknownTutorRequest("local generation did not confirm completion; no retry")
                names = ("prompt_eval_count", "eval_count", "total_duration", "load_duration", "prompt_eval_duration", "eval_duration")
                usage = {key:response.get(key) for key in names}
                valid_usage = all(v is None or type(v) is int and v >= 0 for v in usage.values())
                cost["usage"] = usage if valid_usage else None
                choice = None
                try:
                    if not valid_usage:
                        raise ValueError("invalid generation usage")
                    choice = _selection(response, value)
                except Exception:
                    rejection = "selection_unavailable_or_invalid"
                try:
                    cost["model_digest_after"] = identity(show=False)
                except Exception:
                    rejection = "post_identity_unavailable_or_invalid"
                if rejection is None:
                    profile_id, outcome = choice, "local_accepted"
        if source_hashes() != value["source_sha256"]:
            raise RuntimeError("adviser source changed during decision")
        finish_cost()
        profiles = {p["id"]:p for p in value["catalogue"]["profiles"]}
        decision = dict(schema=SCHEMA, request_sha256=expected_request_sha256, intent_sha256=intent_sha,
            parent=value["parent"], catalogue_sha256=request_sha256(value["catalogue"]),
            development_sha256=request_sha256(value["development"]), profile_id=profile_id,
            profile=profiles[profile_id], coverage=coverage[profile_id], mode=value["mode"], outcome=outcome,
            rejection_reason=rejection, teacher_cost=deepcopy(cost), source_sha256=value["source_sha256"], scope=SCOPE)
        pin = _publish(decision_path, decision)
        return dict(decision=decision, decision_sha256=pin, reused=False,
            invocation_cost=dict(api_attempts=cost["api_attempts"], chat_attempts=cost["chat_attempts"],
                wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started))
    except BaseException as failure:
        finish_cost()
        report = dict(schema=SCHEMA, status="unknown_or_uncommitted", request_sha256=expected_request_sha256,
            intent_sha256=intent_sha, teacher_cost=deepcopy(cost), error_type=type(failure).__name__,
            automatic_retry=False, scope=SCOPE)
        try:
            report["outcome_sha256"] = _publish(directory/"uncommitted.json", report)
        except BaseException as publication:
            report["receipt_error_type"] = type(publication).__name__
        failure.tutor_report = deepcopy(report)
        raise
