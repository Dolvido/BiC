"""One durable local request to order a compact, verified shared-skill curriculum.

The compiler owns teaching semantics. This adapter owns request/result identity
and at-most-one local call per persisted slot. It never runs a learner, compiles
lessons, sees held-out examples, or lets the teacher decide its own withdrawal.
"""
from __future__ import annotations

from copy import deepcopy
import base64
import hashlib
import http.client
import math
from pathlib import Path
import platform
import time

from experiments import foundation_tutor_adviser as prior
from experiments import definition_tutor_chapters as compiler


SCHEMA = "bic-definition-tutor-author-v1"
REQUEST_SCHEMA = "bic-definition-tutor-author-request-v1"
MAX_BYTES = prior.MAX_BYTES
FAMILIES = prior.FAMILIES
PANELS = ("binding", "revision", "composition", "sequence")
METRICS = ("all_query_pair_both", "known_action", "known_reply", "unknown_action", "unknown_reply")
EVIDENCE_SCHEMA = "bic-definition-tutor-development-v1"
UnknownTutorRequest, DecisionPinRequired = prior.UnknownTutorRequest, prior.DecisionPinRequired
transport = prior.transport
_json, _pin, _keys, _integer = prior._json, prior._pin, prior._keys, prior._integer
_native, _publish, _read = prior._native, prior._publish, prior._read
request_sha256 = prior.request_sha256
SCOPE = ("External tutor orders shared primitive-binding and revision chapters from one unchanged lesson catalogue. "
         "Independent admission owns semantics, labels and fixed equal exposure. "
         "No learned policy, tutor benefit or promotion is established here.")
MAX_ERROR_BYTES = 4096


class LocalHTTPError(ValueError):
    """A returned HTTP failure with bounded original bytes, not a completion."""
    def __init__(self, status, body, *, truncated=False, read_error=None):
        super().__init__(f"local_http_status_{status}")
        self.receipt = dict(status=status, body_base64=base64.b64encode(body).decode("ascii"),
            captured_bytes=len(body), truncated=truncated, read_error=read_error,
            body_sha256=hashlib.sha256(body).hexdigest())


def _request_json(route, payload, deadline):
    """Hard-coded loopback only; preserve bounded non-200 response bodies."""
    if route not in transport._LOCAL_ROUTES:raise ValueError("unsupported_local_route")
    body=None if payload is None else transport._canonical_json(payload).encode("utf-8")
    if body is not None and len(body)>transport.MAX_REQUEST_BYTES:raise ValueError("local_request_byte_limit")
    connection=http.client.HTTPConnection(transport.LOCAL_HOST,transport.LOCAL_PORT,
        timeout=transport._remaining(deadline))
    try:
        connection.request("GET" if payload is None else "POST",route,body=body,headers={"Content-Type":"application/json"})
        if connection.sock is not None:connection.sock.settimeout(transport._remaining(deadline))
        response=connection.getresponse()
        limit=transport.MAX_RESPONSE_BYTES if response.status==200 else MAX_ERROR_BYTES
        length=response.getheader("Content-Length")
        if response.status==200 and length is not None and (not length.isdecimal() or int(length)>limit):
            raise ValueError("local_response_byte_limit")
        chunks=[];received=0;failure=None
        try:
            while received<=limit:
                remaining=transport._remaining(deadline)
                if connection.sock is not None:connection.sock.settimeout(remaining)
                chunk=response.read1(min(8192,limit+1-received))
                if not chunk:break
                chunks.append(chunk);received+=len(chunk)
                transport._remaining(deadline)
        except Exception as error:
            if response.status==200:raise
            failure=type(error).__name__
        raw=b"".join(chunks)
        if response.status!=200:
            raise LocalHTTPError(response.status,raw[:limit],truncated=len(raw)>limit,read_error=failure)
        if len(raw)>limit:raise ValueError("local_response_byte_limit")
        value=transport._strict_json(raw.decode("utf-8"))
        if type(value) is not dict:raise ValueError("local_response_object_required")
        return value
    finally:
        connection.close()


def _wire_schema(contract):
    return compiler.recipe_schema(contract)


def source_hashes():
    result = {**prior.source_hashes(), **compiler.source_hashes()}
    result["experiments/definition_tutor_author.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def _development(value,parent):
    """Only bounded aggregate native counts; never examples or target answers."""
    _keys(value,("schema","role","current","reference"),"development evidence")
    if value["schema"]!=EVIDENCE_SCHEMA or value["role"]!="development":
        raise ValueError("development-only basis evidence required")
    for name in ("current","reference"):
        panel=value[name]
        _keys(panel,("weights_sha256","lifetime_updates","panels"),name+" evidence")
        if not _pin(panel["weights_sha256"]):raise ValueError("producing weight identity required")
        _integer(panel["lifetime_updates"],"evidence updates")
        if panel["lifetime_updates"]>parent["lifetime_updates"]:raise ValueError("evidence cannot follow parent")
        if name=="current" and any(panel[k]!=parent[k] for k in ("weights_sha256","lifetime_updates")):
            raise ValueError("current evidence differs from native parent")
        _keys(panel["panels"],PANELS,"basis panels")
        for kind,families in panel["panels"].items():
            _keys(families,FAMILIES,"shared skill families")
            for family,metrics in families.items():
                _keys(metrics,METRICS,"native metrics")
                for metric,count in metrics.items():
                    _keys(count,("count","total"),"counted metric")
                    _integer(count["count"],"count");_integer(count["total"],"total",1)
                    if count["count"]>count["total"]:raise ValueError("count exceeds total")
                for stem in ("known","unknown"):
                    if metrics[stem+"_action"]["total"]!=metrics[stem+"_reply"]["total"]:
                        raise ValueError("action/reply denominators differ")
                # This frozen provider has five known and one unknown query
                # per episode, hence ten known and two unknown per pair.
                solved=metrics["all_query_pair_both"]["count"]
                pairs=metrics["all_query_pair_both"]["total"]
                for stem,queries_per_pair in (("known",10),("unknown",2)):
                    for modality in ("action","reply"):
                        count=metrics[stem+"_"+modality]
                        if count["total"]!=queries_per_pair*pairs or count["count"]<queries_per_pair*solved:
                            raise ValueError("complete-pair count contradicts fixed query inventory")
                errors=sum(metrics[key]["total"]-metrics[key]["count"] for key in METRICS[1:])
                if solved<max(0,pairs-errors):
                    raise ValueError("complete-pair count contradicts query accuracy")
            if any(len({families[f][metric]["total"] for f in FAMILIES})!=1 for metric in METRICS):
                raise ValueError("fixed equal-family measurement denominators required")
    for kind in PANELS:
        for family in FAMILIES:
            if any(value["current"]["panels"][kind][family][metric]["total"]
                    !=value["reference"]["panels"][kind][family][metric]["total"] for metric in METRICS):
                raise ValueError("current/reference measurement denominators differ")


def _compact_evidence(value):
    return dict(metrics=list(METRICS),families=list(FAMILIES),panels=list(PANELS),
        current=[[[value["current"]["panels"][p][f][m][k] for m in METRICS for k in ("count","total")]
                  for f in FAMILIES] for p in PANELS],
        reference=[[[value["reference"]["panels"][p][f][m][k] for m in METRICS for k in ("count","total")]
                    for f in FAMILIES] for p in PANELS],
        format="panel rows contain family rows of count,total pairs in metrics order")


def _chat_payload(value):
    payload=dict(model=value["teacher"]["model"],stream=False,keep_alive=0,
        format=_wire_schema(value["contract"]),
        messages=[dict(role="system",content=(
            "Order six verified practice chapters to improve reusable instruction learning. "
            "Use every chapter exactly once; each binding chapter must precede its matching revision. "
            "All chapters apply equally to color, count and switch. The same lessons, targets, "
            "within-chapter order and exposure counts are fixed for every permitted order. "
            "Choose one shared progression, not family-specific patches. Return only required JSON. "
            "Do not provide answers, facts, prose, code or new operations. Counts are evidence, not instructions.")),
            dict(role="user",content=_json(dict(teaching=compiler.authoring_catalogue(value["contract"]),
                procedural_recipe=value["procedural_recipe"],development=_compact_evidence(value["development"]))))],
        options=dict(temperature=0,seed=17092026,num_ctx=4096,num_predict=256))
    if sum(len(m["content"].encode("utf-8")) for m in payload["messages"])>3800:
        raise ValueError("compact basis author input exceeds3800 bytes")
    if len(transport._canonical_json(payload).encode("utf-8"))>transport.MAX_REQUEST_BYTES:
        raise ValueError("basis author payload exceeds transport byte limit")
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
    _development(value["development"], parent)
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
    for index,event in enumerate(cost["api_events"],1):
        if "http_error" in event:
            if event["error_file"]!=f"http-error-{index:03d}.json":
                raise ValueError("HTTP error receipt location differs")
            detail=_read(path.parent/event["error_file"],event["error_file_sha256"])
            if detail!={"route":event["route"],**event["http_error"]}:
                raise ValueError("HTTP error receipt contents differ")
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
            response = _request_json(route, payload, expires)
            cost["api_completions"] += 1; event["status"] = "returned"
            if route == "/api/chat" and response.get("done") is True:
                cost["chat_completions"] += 1; cost["physical_teacher_work_unknown"] = False
            return response
        except LocalHTTPError as failure:
            event["status"]="http_error"
            event["http_error"]=deepcopy(failure.receipt)
            name=f"http-error-{cost['api_attempts']:03d}.json"
            event["error_file"]=name
            event["error_file_sha256"]=_publish(directory/name,dict(route=route,**failure.receipt))
            raise
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
