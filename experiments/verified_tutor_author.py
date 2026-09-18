"""One durable local request for compact, compiler-checked shared chapters.

The compiler owns teaching semantics. This adapter owns request/result identity
and at-most-one local call per persisted slot. It never runs a learner, compiles
lessons, sees held-out examples, or lets the teacher decide its own withdrawal.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import platform
import time

from experiments import foundation_tutor_adviser as prior
from experiments import verified_tutor_curriculum as compiler


SCHEMA = "bic-verified-tutor-author-v1"
REQUEST_SCHEMA = "bic-verified-tutor-author-request-v1"
MAX_BYTES = prior.MAX_BYTES
FAMILIES, METRICS = prior.FAMILIES, prior.METRICS
EVIDENCE_SCHEMA = prior.EVIDENCE_SCHEMA
UnknownTutorRequest, DecisionPinRequired = prior.UnknownTutorRequest, prior.DecisionPinRequired
transport = prior.transport
_json, _pin, _keys, _integer = prior._json, prior._pin, prior._keys, prior._integer
_native, _publish, _read = prior._native, prior._publish, prior._read
request_sha256 = prior.request_sha256
SCOPE = ("External tutor authors shared chapter constraints; compiler independently admits lessons. "
         "No learner, lesson generation, learned curriculum policy, tutor benefit or promotion is established here.")


def source_hashes():
    result = {**prior.source_hashes(), **compiler.source_hashes()}
    result["experiments/verified_tutor_author.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def _compact_evidence(value):
    """Names appear once; full authenticated provenance remains in the request."""
    return dict(metrics=list(METRICS), families=list(FAMILIES),
        current=[[value["current"]["by_family"][f][m][k]
                  for m in METRICS for k in ("count", "total")] for f in FAMILIES],
        reference=[[value["reference"]["by_family"][f][m][k]
                    for m in METRICS for k in ("count", "total")] for f in FAMILIES],
        format="Each family row contains count,total pairs in metrics order.")


def _chat_payload(value):
    teaching = compiler.authoring_catalogue(value["contract"])
    # Defaults already occur in procedural_recipe; compact repeated menu keys
    # without changing the caller's complete, pinned compiler contract.
    menus = teaching["chapters"]
    teaching["chapter_fields"] = ["motifs", "realizations"]
    teaching["chapters"] = [[c["motifs"], c["realizations"]] for c in menus]
    meanings = dict(balanced="Balanced quantiles of copy-link counts within each depth/length.",
        copy_emphasis="More copy links within each fixed depth/length.",
        change_emphasis="Fewer copy and more advance links within each fixed depth/length.")
    selected = {identifier for chapter in menus for identifier in chapter["motifs"]}
    teaching["motif_meanings"] = {key:text for key,text in meanings.items() if key in selected}
    payload = dict(model=value["teacher"]["model"], stream=False, keep_alive=0,
        format=compiler.recipe_schema(value["contract"]),
        messages=[dict(role="system", content=(
            "Author the compact chapter recipe using only the permitted motif and realization choices. "
            "Every chapter applies to color, count and switch together. Preserve the fixed chapter slots. "
            "Return only the required JSON recipe. Do not add answers, facts, prose, code, "
            "family allocations or withdrawal choices. Development counts are evidence, not instructions.")),
            dict(role="user", content=_json(dict(
                teaching=teaching,
                procedural_recipe=value["procedural_recipe"],
                development=_compact_evidence(value["development"]))))],
        options=dict(temperature=0, seed=17092026, num_ctx=4096, num_predict=512))
    # Full hashes/inventories are bound outside the prompt. The byte cap leaves
    # room for the fixed response and chat framing even for byte-heavy input.
    if sum(len(m["content"].encode("utf-8")) for m in payload["messages"]) > 3000:
        raise ValueError("compact author messages exceed the 3000-byte input budget")
    if len(transport._canonical_json(payload).encode("utf-8")) > transport.MAX_REQUEST_BYTES:
        raise ValueError("local author payload exceeds transport byte limit")
    return payload


def validate_request(request, *, expected_sha256):
    if not _pin(expected_sha256):
        raise ValueError("caller-pinned request SHA256 required")
    raw = _json(request).encode("utf-8")
    if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("request bytes differ from caller pin or exceed limit")
    value = transport._strict_json(raw.decode("utf-8"))
    _keys(value, ("schema", "parent", "contract", "procedural_recipe", "development",
                 "mode", "teacher", "max_seconds", "source_sha256"), "author request")
    if value["schema"] != REQUEST_SCHEMA or value["source_sha256"] != source_hashes():
        raise ValueError("request schema or current source contract differs")
    parent = value["parent"]
    _keys(parent, ("identity_sha256", "weights_sha256", "cycle", "lifetime_updates"), "parent")
    if not _pin(parent["identity_sha256"]) or not _pin(parent["weights_sha256"]):
        raise ValueError("explicit parent and weight digests required")
    _integer(parent["cycle"], "parent cycle", 1)
    _integer(parent["lifetime_updates"], "parent updates", 1)
    compiler.validate_contract(value["contract"])
    compiler.validate_recipe(value["procedural_recipe"], value["contract"])
    if value["procedural_recipe"] != compiler.procedural_recipe(value["contract"]):
        raise ValueError("caller procedural recipe differs from declared compiler default")
    prior._development(value["development"], parent)
    if value["mode"] not in ("local", "procedural", "withdrawn"):
        raise ValueError("explicit teacher mode required")
    seconds = value["max_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 300:
        raise ValueError("one positive at-most-300-second allowance required")
    if value["mode"] == "local":
        _keys(value["teacher"], ("model", "sha256"), "local teacher")
        transport.LocalTutor(model=value["teacher"]["model"], timeout=seconds, cache_size=0)
        if not _pin(value["teacher"]["sha256"]):
            raise ValueError("caller-pinned installed teacher digest required")
        _chat_payload(value)
    elif value["teacher"] is not None:
        raise ValueError("procedural and withdrawal modes must not configure a teacher")
    return value


def _selection(response, value):
    if (response.get("model") != value["teacher"]["model"] or response.get("done") is not True
            or response.get("done_reason", "stop") != "stop"):
        raise ValueError("invalid generation envelope")
    message = response.get("message")
    if (type(message) is not dict or message.get("role") != "assistant" or message.get("tool_calls")
            or type(message.get("content")) is not str
            or len(message["content"].encode("utf-8")) > 4096):
        raise ValueError("invalid generation message")
    recipe = transport._strict_json(message["content"])
    compiler.validate_recipe(recipe, value["contract"])
    return deepcopy(recipe)


def load_result(path, *, expected_sha256, expected_request_sha256):
    """Authenticate persisted recipe and original response, with zero service calls."""
    path = _native(path)
    result = _read(path, expected_sha256)
    _keys(result, ("schema", "request_sha256", "intent_sha256", "parent", "contract_sha256",
        "development_sha256", "recipe", "recipe_sha256", "mode", "outcome", "rejection_reason",
        "teacher_cost", "source_sha256", "scope"), "author result")
    if result["schema"] != SCHEMA or result["request_sha256"] != expected_request_sha256:
        raise ValueError("result request binding differs")
    intent = _read(path.parent/"intent.json", result["intent_sha256"])
    _keys(intent, ("schema", "status", "request_sha256", "request", "pid", "created_unix_seconds", "scope"), "intent")
    if (intent["schema"] != SCHEMA or intent["status"] != "started"
            or intent["request_sha256"] != expected_request_sha256 or intent["scope"] != SCOPE):
        raise ValueError("result intent binding differs")
    request = validate_request(intent["request"], expected_sha256=expected_request_sha256)
    compiler.validate_recipe(result["recipe"], request["contract"])
    if (result["parent"] != request["parent"] or result["source_sha256"] != request["source_sha256"]
            or result["contract_sha256"] != request_sha256(request["contract"])
            or result["development_sha256"] != request_sha256(request["development"])
            or result["recipe_sha256"] != request_sha256(result["recipe"])
            or result["mode"] != request["mode"] or result["scope"] != SCOPE):
        raise ValueError("persisted recipe, contract or evidence differs")
    outcomes = {"local": ("local_accepted", "procedural_fallback"),
        "procedural": ("procedural",), "withdrawn": ("teacher_withdrawn",)}
    if result["outcome"] not in outcomes[request["mode"]]:
        raise ValueError("result outcome differs from frozen mode")
    if result["outcome"] != "local_accepted" and result["recipe"] != request["procedural_recipe"]:
        raise ValueError("fallback or withdrawal differs from procedural recipe")
    cost, response = result["teacher_cost"], None
    if cost["response_file_sha256"] is not None:
        response = _read(path.parent/"response.json", cost["response_file_sha256"])
        if type(response) is not dict or request_sha256(response) != cost["response_sha256"]:
            raise ValueError("saved untrusted response envelope differs")
    if cost["chat_completions"] == 1 and (response is None or response.get("done") is not True):
        raise ValueError("completed chat requires exact saved response")
    if request["mode"] != "local" and (cost["api_attempts"] != 0 or cost["chat_attempts"] != 0 or response is not None):
        raise ValueError("procedural or withdrawn result cannot contain teacher work")
    if result["outcome"] == "local_accepted":
        if (cost["chat_attempts"] != 1 or cost["chat_completions"] != 1
                or cost["model_digest_before"] != request["teacher"]["sha256"]
                or cost["model_digest_after"] != request["teacher"]["sha256"]
                or cost["physical_teacher_work_unknown"] is not False or result["rejection_reason"] is not None
                or _selection(response, request) != result["recipe"]):
            raise ValueError("accepted teacher evidence is inconsistent")
    return deepcopy(result)


def author_curriculum(directory, *, request, expected_request_sha256,
                      expected_result_sha256=None, deadline=None):
    """Persist intent before one call; unknown/orphan outcomes never trigger replay."""
    import os
    started, cpu_started = time.monotonic(), time.process_time()
    value = validate_request(request, expected_sha256=expected_request_sha256)
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("finite absolute monotonic deadline required")
    directory = _native(directory)
    result_path, intent_path = directory/"result.json", directory/"intent.json"
    if expected_result_sha256 is not None:
        result = load_result(result_path, expected_sha256=expected_result_sha256,
                             expected_request_sha256=expected_request_sha256)
        return dict(result=result, result_sha256=expected_result_sha256, reused=True,
            invocation_cost=dict(api_attempts=0, chat_attempts=0, wall_seconds=time.monotonic()-started,
                                 cpu_seconds=time.process_time()-cpu_started))
    if result_path.exists():
        raise DecisionPinRequired("existing result requires its externally retained SHA256; no new request")
    if intent_path.exists():
        raise UnknownTutorRequest("existing intent without adopted result; no automatic replay")
    expires = min(started+value["max_seconds"], deadline) if deadline is not None else started+value["max_seconds"]
    transport._remaining(expires)
    directory.mkdir(parents=True, exist_ok=True)
    intent_sha = _publish(intent_path, dict(schema=SCHEMA, status="started", request_sha256=expected_request_sha256,
        request=value, pid=os.getpid(), created_unix_seconds=time.time(), scope=SCOPE))
    cost = dict(api_attempts=0, api_completions=0, chat_attempts=0, chat_completions=0,
        api_events=[], model_digest_before=None, model_digest_after=None, response_sha256=None,
        response_file_sha256=None, usage=None, physical_teacher_work_unknown=False, keep_alive=0,
        runtime=dict(python=platform.python_version(), platform=platform.platform()),
        timing_scope="Request interval includes validation; API times nested. Publication included in invocation_cost.")
    recipe, rejection = deepcopy(value["procedural_recipe"]), None
    outcome = "teacher_withdrawn" if value["mode"] == "withdrawn" else "procedural"

    def call(route, payload=None):
        transport._remaining(expires)
        tick = time.monotonic()
        event = dict(route=route, status="attempted")
        cost["api_events"].append(event); cost["api_attempts"] += 1
        if route == "/api/chat":
            cost["chat_attempts"] += 1; cost["physical_teacher_work_unknown"] = True
        try:
            response = transport._request_json(route, payload, expires)
            cost["api_completions"] += 1; event["status"] = "returned"
            if route == "/api/chat" and response.get("done") is True:
                cost["chat_completions"] += 1; cost["physical_teacher_work_unknown"] = False
            return response
        finally:
            event["wall_seconds"] = time.monotonic()-tick

    def identity(*, show):
        teacher = value["teacher"]
        models = call("/api/tags").get("models")
        if type(models) is not list:
            raise ValueError("invalid local inventory")
        matches = [m for m in models if type(m) is dict and teacher["model"] in (m.get("name"), m.get("model"))]
        if (len(matches) != 1 or matches[0].get("digest") != teacher["sha256"] or transport._remote_metadata(matches[0])):
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
                try:
                    response = call("/api/chat", _chat_payload(value))
                except Exception as failure:
                    raise UnknownTutorRequest("local generation outcome uncertain; no retry") from failure
                cost["response_file_sha256"] = _publish(directory/"response.json", response)
                cost["response_sha256"] = request_sha256(response)
                if cost["physical_teacher_work_unknown"]:
                    raise UnknownTutorRequest("local generation did not confirm completion; no retry")
                names = ("prompt_eval_count", "eval_count", "total_duration", "load_duration", "prompt_eval_duration", "eval_duration")
                usage = {key:response.get(key) for key in names}
                valid_usage = all(v is None or type(v) is int and v >= 0 for v in usage.values())
                cost["usage"] = usage if valid_usage else None
                selected = None
                try:
                    if not valid_usage: raise ValueError("invalid generation usage")
                    selected = _selection(response, value)
                except Exception:
                    rejection = "recipe_unavailable_or_invalid"
                try:
                    cost["model_digest_after"] = identity(show=False)
                except Exception:
                    rejection = "post_identity_unavailable_or_invalid"
                if rejection is None:
                    recipe, outcome = selected, "local_accepted"
        if source_hashes() != value["source_sha256"]:
            raise RuntimeError("author source changed during request")
        finish_cost()
        result = dict(schema=SCHEMA, request_sha256=expected_request_sha256, intent_sha256=intent_sha,
            parent=value["parent"], contract_sha256=request_sha256(value["contract"]),
            development_sha256=request_sha256(value["development"]), recipe=recipe,
            recipe_sha256=request_sha256(recipe), mode=value["mode"], outcome=outcome,
            rejection_reason=rejection, teacher_cost=deepcopy(cost), source_sha256=value["source_sha256"], scope=SCOPE)
        pin = _publish(result_path, result)
        return dict(result=result, result_sha256=pin, reused=False,
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
