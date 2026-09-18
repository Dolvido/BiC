"""Optional, bounded offline tutor data pipeline; never imported by BiC inference."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.error
import urllib.request

from experiments.local_tutor_validation import (
    CANDIDATE_SCHEMA, GRAMMAR, RELATIONS, canonical_json, digest,
    strict_json, validate_candidate, verified_label,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:11434"
SCHEMA = "bic-local-tutor-smoke-v1"
SETTINGS = {"temperature": 0, "seed": 16092026, "num_ctx": 4096, "num_predict": 192}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_hashes():
    return {name: sha_file(ROOT / name) for name in (
        "experiments/local_tutor_pilot.py", "experiments/local_tutor_validation.py",
        "experiments/train_regional_memory.py")}


def smoke_specs(seed):
    """Deterministic trusted simulator fixtures, recomputed when importing."""
    from experiments.train_regional_memory import relation_target
    rng = random.Random(seed)
    specs = []
    for slot in range(10):
        for relation_index, relation in enumerate(RELATIONS):
            index = len(specs)
            if slot < 5:
                category = "valid"
            elif slot < 8:
                category = ("unknown_label", "absent", "ambiguous")[slot - 5]
            elif slot == 8:
                category = "unknown_label" if relation == "find" else "boundary"
            else:
                category = ("unknown_label", "absent", "ambiguous", "boundary", "boundary")[relation_index]
            query, *objects = rng.sample(range(1, 65536), 5)
            eligible = [cell for cell in range(4)
                        if (relation_target(cell, relation) == 10) == (category == "boundary")]
            reference = rng.choice(eligible) if category in ("valid", "boundary") else rng.randrange(4)
            if category != "absent":
                objects[reference] = query
            if category == "ambiguous":
                objects[(reference + 1) % 4] = query
            spec = {"lesson_id": f"pilot-{index:03d}", "name": f"pilot{index:03d}",
                    "relation": relation, "query": query, "objects": objects,
                    "known": category != "unknown_label", "support_seed": rng.randrange(2**31),
                    "object_seeds": [rng.randrange(2**31) for _ in range(4)]}
            assert verified_label(spec)["category"] == category
            specs.append(spec)
    return specs


def prepare(path, seed=16092026):
    """Freeze 50 pipeline cases, not benchmark splits or novel test identities."""
    specs = smoke_specs(seed)
    protocol = {"schema": SCHEMA, "purpose": "pipeline_smoke_only", "seed": seed,
                "count": 50, "specs": specs, "grammar": GRAMMAR,
                "grammar_sha256": digest(GRAMMAR), "candidate_schema": CANDIDATE_SCHEMA,
                "source_sha256": source_hashes(),
                "split_policy": "No training or benchmark split; these fixtures must not enter a final benchmark."}
    output = new_directory(path)
    write_json(output / "protocol.json", protocol)
    return {"protocol": str(output / "protocol.json"), "sha256": sha_file(output / "protocol.json")}


def read_protocol(path):
    if Path(path).stat().st_size > 131072:
        raise ValueError("protocol exceeds 128KiB limit")
    protocol = strict_json(Path(path).read_text(encoding="utf-8"))
    if (protocol.get("schema") != SCHEMA or protocol.get("count") != 50
            or len(protocol.get("specs", [])) != 50
            or protocol.get("grammar_sha256") != digest(GRAMMAR)
            or protocol.get("grammar") != GRAMMAR
            or protocol.get("candidate_schema") != CANDIDATE_SCHEMA
            or protocol.get("source_sha256") != source_hashes()):
        raise ValueError("protocol/schema/grammar/source mismatch; prepare a fresh output")
    ids = [s["lesson_id"] for s in protocol["specs"]]
    if len(set(ids)) != 50:
        raise ValueError("duplicate protocol lesson IDs")
    if type(protocol.get("seed")) is not int or protocol["specs"] != smoke_specs(protocol["seed"]):
        raise ValueError("simulator fixtures differ from frozen deterministic seed")
    for spec in protocol["specs"]:
        verified_label(spec)
    return protocol


def import_candidates(protocol_path, candidates_path, output_path, provenance=None):
    """Revalidate all data without contacting Ollama; preserve every rejection."""
    protocol = read_protocol(protocol_path)
    if Path(candidates_path).stat().st_size > 50 * 32768:
        raise ValueError("candidate file exceeds bounded smoke import size")
    output = new_directory(output_path)
    specs = {s["lesson_id"]: s for s in protocol["specs"]}
    rows = []
    with Path(candidates_path).open(encoding="utf-8") as source:
        for index, line in enumerate(source, 1):
            if index > 50:
                raise ValueError("candidate file exceeds the 50-candidate smoke budget")
            if len(line.encode("utf-8")) > 32768:
                rows.append({"line": index, "error": "envelope_byte_limit",
                             "raw_line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest()})
                continue
            try:
                row = strict_json(line)
                if not isinstance(row, dict):
                    raise ValueError("envelope must be an object")
                rows.append(row)
            except (ValueError, RecursionError):
                rows.append({"line": index, "error": "invalid_envelope_json", "raw_line": line})
    counts = Counter(row.get("lesson_id") for row in rows if isinstance(row.get("lesson_id"), str))
    accepted, rejected = [], []
    for row in rows:
        identifier = row.get("lesson_id")
        if not isinstance(identifier, str) or identifier not in specs:
            decision = {"accepted": False, "lesson_id": identifier, "reason": row.get("error", "unknown_lesson_id")}
        elif counts[identifier] != 1:
            decision = {"accepted": False, "lesson_id": identifier, "reason": "duplicate_lesson_id"}
        elif row.get("error"):
            decision = {"accepted": False, "lesson_id": identifier, "reason": "generation_error", "detail": row["error"]}
        else:
            decision = validate_candidate(row.get("candidate"), specs[identifier])
        decision["envelope_sha256"] = digest(row)
        if decision["accepted"]:
            # Hidden scene facts and targets are teacher/training metadata only.
            decision["spec"] = dict(specs[identifier], prompt=decision["instruction"],
                                    **{key: decision[key] for key in ("target", "reply", "category")})
            accepted.append(decision)
        else:
            decision["raw_envelope"] = row
            rejected.append(decision)
    for identifier in sorted(specs.keys() - counts.keys()):
        rejected.append({"accepted": False, "lesson_id": identifier, "reason": "missing_candidate"})
    for filename, records in (("accepted.jsonl", accepted), ("rejected.jsonl", rejected)):
        (output / filename).write_text("".join(canonical_json(row) + "\n" for row in records), encoding="utf-8")
    report = {"schema": SCHEMA, "purpose": "pipeline_smoke_only", "candidates_received": len(rows),
              "accepted": len(accepted), "rejected_records": len(rejected),
              "acceptance_rate": len(accepted) / 50,
              "rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
              "accepted_relations": dict(Counter(r["relation"] for r in accepted)),
              "accepted_categories": dict(Counter(r["category"] for r in accepted)),
              "accepted_families": dict(Counter(r["phrase_family"] for r in accepted)),
              "protocol_sha256": sha_file(protocol_path), "candidates_sha256": sha_file(candidates_path),
              "accepted_sha256": sha_file(output / "accepted.jsonl"),
              "rejected_sha256": sha_file(output / "rejected.jsonl"),
              "source_sha256": source_hashes(), "provenance": provenance or {"origin": "external_file_unverified"}}
    write_json(output / "report.json", report)
    return report


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirects are disabled for local-only generation", headers, fp)


def api(route, payload=None, timeout=60):
    if route not in ("/api/tags", "/api/version", "/api/ps", "/api/show", "/api/chat", "/api/generate"):
        raise ValueError("unsupported local API route")
    body = None if payload is None else canonical_json(payload).encode("utf-8")
    request = urllib.request.Request(BASE + route, data=body, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(65537)
    if len(raw) > 65536:
        raise ValueError("API response exceeds 64KiB limit")
    return strict_json(raw.decode("utf-8"))


def remote_metadata(value):
    if isinstance(value, dict):
        return any((key in ("remote_model", "remote_host") and bool(item)) or remote_metadata(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(remote_metadata(item) for item in value)
    return False


def make_request(spec, model, index):
    family = tuple(GRAMMAR)[index % len(GRAMMAR)]
    approved = GRAMMAR[family][spec["relation"]].format(quoted_name=json.dumps(spec["name"], ensure_ascii=False))
    # The teacher gets the requested meaning and exact name, not benchmark data.
    example = {"lesson_id": spec["lesson_id"], "instruction": approved, "relation": spec["relation"]}
    prompt = ("Reproduce this exact JSON object. Preserve every character in each string, including "
              "the final period in instruction and its double-quoted object name. The escaped double "
              "quotes below encode actual double-quote characters inside the instruction; do not "
              "replace them with apostrophes, curly quotes, or markdown. Do not remove the period.\n"
              "BEGIN EXACT JSON\n" + json.dumps(example, ensure_ascii=False, indent=2) + "\nEND EXACT JSON")
    return {"model": model, "messages": [
                {"role": "system", "content": "You copy JSON records exactly. Output only the JSON object. Never change punctuation or wording."},
                {"role": "user", "content": prompt}],
            "stream": False, "format": CANDIDATE_SCHEMA,
            "options": dict(SETTINGS, seed=SETTINGS["seed"] + index), "keep_alive": "2m"}


def generate(protocol_path, output_path, model="ministral-3:3b", max_seconds=600):
    protocol = read_protocol(protocol_path)
    if not 1 <= max_seconds <= 1800:
        raise ValueError("max-seconds must be from 1 through 1800")
    tags = api("/api/tags")
    matches = [m for m in tags["models"] if model in (m.get("name"), m.get("model"))]
    if len(matches) != 1 or not matches[0].get("digest"):
        raise ValueError("model must already exist locally with an identifiable digest; no automatic download")
    if "cloud" in model.lower() or remote_metadata(matches[0]):
        raise ValueError("cloud/remote model aliases are not permitted by the offline tutor pilot")
    details = api("/api/show", {"model": model})
    if remote_metadata(details):
        raise ValueError("cloud/remote model aliases are not permitted by the offline tutor pilot")
    before = api("/api/ps")
    # Generation must not evict someone else's model or alter their keepalive.
    if before.get("models"):
        raise ValueError("generation requires an empty Ollama model list; finish other local inference first")
    output = new_directory(output_path)
    raw_dir = output / "raw"
    raw_dir.mkdir()
    manifest = {"schema": SCHEMA, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "model": matches[0], "model_details_sha256": digest(details), "ollama_version": api("/api/version"),
                "settings": SETTINGS, "max_seconds": max_seconds,
                "protocol_sha256": sha_file(protocol_path), "source_sha256": source_hashes(),
                "ps_before": before, "candidate_limit": 50, "endpoint": BASE,
                "documents": ["https://docs.ollama.com/api/chat", "https://docs.ollama.com/api/ps",
                              "https://docs.ollama.com/capabilities/structured-outputs"]}
    write_json(output / "manifest.json", manifest)
    start = time.perf_counter()
    attempted = False
    try:
        with (output / "candidates.jsonl").open("w", encoding="utf-8") as candidates:
            for index, spec in enumerate(protocol["specs"]):
                row = {"lesson_id": spec["lesson_id"]}
                remaining = max_seconds - (time.perf_counter() - start)
                if remaining <= 0:
                    row["error"] = "generation_budget_exhausted"
                else:
                    request = make_request(spec, model, index)
                    write_json(raw_dir / f"{index:03d}-request.json", request)
                    row["request_sha256"] = digest(request)
                    tick = time.perf_counter()
                    try:
                        attempted = True
                        response = api("/api/chat", request, timeout=min(60, remaining))
                        write_json(raw_dir / f"{index:03d}-response.json", response)
                        row["response_sha256"] = digest(response)
                        running = [m for m in api("/api/ps")["models"]
                                   if model in (m.get("name"), m.get("model"))]
                        if len(running) != 1 or running[0].get("digest") != matches[0]["digest"]:
                            raise ValueError("running model digest differs from frozen model revision")
                        row["running_model_digest"] = running[0]["digest"]
                        if response.get("done") is not True or response.get("done_reason") == "length":
                            raise ValueError("incomplete or token-limited generation")
                        row["candidate"] = response["message"]["content"]
                    except Exception as error:
                        row["error"] = f"{type(error).__name__}: {error}"
                    row["wall_seconds"] = time.perf_counter() - tick
                candidates.write(canonical_json(row) + "\n")
                candidates.flush()
                print(f"candidate {index + 1}/50: {'error' if row.get('error') else 'saved'}", flush=True)
                if row.get("error") and row["error"] != "generation_budget_exhausted":
                    # A failed HTTP call can leave server work running; do not pile up requests.
                    break
    finally:
        manifest["wall_seconds"] = time.perf_counter() - start
        if attempted:
            try:
                manifest["unload"] = api("/api/generate", {"model": model, "stream": False, "keep_alive": 0})
            except Exception as error:
                manifest["unload_error"] = f"{type(error).__name__}: {error}"
        manifest["ps_after"] = api("/api/ps")
        after = [m for m in api("/api/tags")["models"] if model in (m.get("name"), m.get("model"))]
        manifest["model_revision_unchanged"] = len(after) == 1 and after[0].get("digest") == matches[0]["digest"]
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
    if not manifest["model_revision_unchanged"]:
        raise ValueError("model revision changed; raw data retained but import refused")
    report = import_candidates(protocol_path, output / "candidates.jsonl", output / "validated",
                               {"origin": "local_ollama", "manifest_sha256": sha_file(output / "manifest.json"),
                                "model_digest": matches[0]["digest"], "runtime": manifest["ollama_version"],
                                "settings": SETTINGS})
    report["generation_wall_seconds"] = manifest["wall_seconds"]
    report["accepted_per_minute"] = report["accepted"] * 60 / max(manifest["wall_seconds"], .001)
    write_json(output / "validated" / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--seed", type=int, default=16092026)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--protocol", required=True)
    generate_parser.add_argument("--output", required=True)
    generate_parser.add_argument("--model", default="ministral-3:3b")
    generate_parser.add_argument("--max-seconds", type=int, default=600)
    import_parser = commands.add_parser("import")
    import_parser.add_argument("--protocol", required=True)
    import_parser.add_argument("--candidates", required=True)
    import_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output, args.seed)
    elif args.command == "generate":
        result = generate(args.protocol, args.output, args.model, args.max_seconds)
    else:
        result = import_candidates(args.protocol, args.candidates, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
