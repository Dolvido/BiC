"""Authenticate completed historical evidence through its original source archive.

Standard-library byte verification only: no model loading, execution of archived
code, lesson replay, score recomputation, or integration with the active study.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import time


SCHEMA = "bic-foundation-historical-archive-v1"
PILOT_SUMMARY_SHA256 = "188b0d026c4debce596f3cca5a1b64f6f5f0e0e3992dd4174904a33b6c3e87c2"


@dataclass(frozen=True)
class ExpectedEvidence:
    """Caller trust anchor and explicit expected identity, never inferred counts."""

    summary_sha256: str
    input_count: int
    source_count: int
    summary_schema: str = "bic-foundation-pilot-summary-v1"
    protocol_schema: str = "bic-foundation-order-pilot-v2"
    results_schema: str = "bic-foundation-results-v1"
    version: str = "bic-shared-foundation-v1"
    arms: tuple[str, ...] = ("curriculum", "mixed")
    checkpoints: tuple[int, ...] = (0, 192, 384, 576, 768, 960, 1152, 1536)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _json_bytes(raw):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate JSON member: " + name)
            result[name] = value
        return result

    def finite(value):
        raise ValueError("nonfinite JSON constant: " + value)

    result = json.loads(raw, object_pairs_hook=unique, parse_constant=finite)
    if type(result) is not dict:
        raise ValueError("JSON object required")
    return result


def _canonical_component(part):
    return (part not in ("", ".", "..") and not part.endswith((" ", "."))
            and not any(ord(char) < 32 or char in '<>:"\\|?*' for char in part)
            and not PureWindowsPath(part).is_reserved())


def _relative_name(name):
    if (type(name) is not str or not name or name.startswith("/")
            or any(not _canonical_component(part) for part in name.split("/"))):
        raise ValueError("canonical relative path required: " + repr(name))
    return name


def _absolute_name(name):
    if type(name) is not str:
        raise ValueError("canonical absolute input path required")
    path = Path(name)
    if (not path.is_absolute() or str(path) != name
            or any(not _canonical_component(part) for part in path.parts[1:])):
        raise ValueError("canonical absolute input path required: " + name)
    return path


def _physical(path, *, directory=False):
    """Reject symlinks, junctions, case/short-name aliases, and nonregular files."""
    path = _absolute_name(str(path))
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("symlink or junction in evidence path: " + str(part))
    if str(path.resolve(strict=True)) != str(path):
        raise ValueError("aliased evidence path: " + str(path))
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError("regular evidence file/directory required: " + str(path))
    return path


def _map(value, *, count, relative):
    if type(value) is not dict or len(value) != count:
        raise ValueError("declared input/source count differs")
    seen = set()
    for name, digest in value.items():
        (_relative_name if relative else _absolute_name)(name)
        if not _sha(digest):
            raise ValueError("lowercase SHA-256 required")
        if name.casefold() in seen:
            raise ValueError("ambiguous case alias in declared paths")
        seen.add(name.casefold())
    return value


def _fingerprint(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def verify_archive(directory, *, repository_root, expected: ExpectedEvidence):
    """Return an explicit receipt; read bytes and JSON, never write evidence.

    Historical absolute input locations remain binding. Only a protocol-declared
    source at repository_root/name is resolved to directory/source/name, and that
    archive copy must itself already be bound by the pinned summary.
    """
    started, cpu_started = time.monotonic(), time.process_time()
    verifier_path = _physical(Path(__file__).absolute())
    verifier_sha256 = file_hash(verifier_path)
    if (not isinstance(expected, ExpectedEvidence) or not _sha(expected.summary_sha256)
            or type(expected.input_count) is not int or expected.input_count < 1
            or type(expected.source_count) is not int or expected.source_count < 1
            or type(expected.arms) is not tuple or not expected.arms
            or len(set(expected.arms)) != len(expected.arms)
            or type(expected.checkpoints) is not tuple or len(expected.checkpoints) < 2
            or any(type(step) is not int or step < 0 for step in expected.checkpoints)
            or expected.checkpoints[0] != 0
            or tuple(sorted(set(expected.checkpoints))) != expected.checkpoints):
        raise ValueError("explicit valid caller expectations required")
    for arm in expected.arms:
        _relative_name(arm)
        if "/" in arm:
            raise ValueError("arm must be one canonical path component")
    directory = _physical(Path(directory).absolute(), directory=True)
    root = _physical(Path(repository_root).absolute(), directory=True)
    summary_path = _physical(directory / "evaluation/summary.json")
    raw = summary_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected.summary_sha256:
        raise ValueError("historical summary differs from caller-pinned SHA-256")
    summary = _json_bytes(raw)
    integrity = summary.get("integrity")
    if (summary.get("schema") != expected.summary_schema
            or summary.get("status") != "completed_descriptive"
            or summary.get("automatic_promotion") is not False or type(integrity) is not dict
            or integrity.get("canonical_replay_repeated") is not False
            or integrity.get("neural_training_or_inference") is not False):
        raise ValueError("completed descriptive non-neural historical summary required")
    inputs = _map(integrity.get("input_file_sha256"), count=expected.input_count, relative=False)
    sources = _map(integrity.get("source_sha256"), count=expected.source_count, relative=True)
    fingerprints, verified, json_cache, physical_ids = {}, {}, {}, {}
    bytes_hashed = 0

    def bind(path, digest, *, read_json=False):
        nonlocal bytes_hashed
        path = _physical(path)
        name = str(path)
        if name not in verified:
            before = _fingerprint(path)
            physical_id = before[:2]
            if physical_id in physical_ids and physical_ids[physical_id] != name:
                raise ValueError("hardlinked evidence aliases are unsupported")
            physical_ids[physical_id] = name
            if read_json:
                content = path.read_bytes()
                actual = hashlib.sha256(content).hexdigest()
            else:
                actual = file_hash(path)
            if actual != digest or before != _fingerprint(_physical(path)):
                raise ValueError("bound evidence changed: " + name)
            verified[name], fingerprints[name] = digest, before
            bytes_hashed += before[2]
            if read_json:
                json_cache[name] = _json_bytes(content)
        elif verified[name] != digest:
            raise ValueError("conflicting digest for bound evidence: " + name)
        if read_json and name not in json_cache:
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("bound JSON changed while reading")
            json_cache[name] = _json_bytes(content)
        return json_cache.get(name)

    def bound_json(relative):
        path = directory / _relative_name(relative)
        if str(path) not in inputs:
            raise ValueError("required completion/protocol record is not summary-bound: " + relative)
        return bind(path, inputs[str(path)], read_json=True)

    protocol = bound_json("protocol.json")
    contract = protocol.get("contract")
    if (type(contract) is not dict or contract.get("schema") != expected.protocol_schema
            or contract.get("version") != expected.version
            or contract.get("arms") != list(expected.arms)
            or not _same(contract.get("checkpoints"), list(expected.checkpoints))
            or type(contract.get("updates")) is not int or contract["updates"] != expected.checkpoints[-1]
            or contract.get("automatic_promotion") is not False
            or contract.get("evaluation_during_training") is not False
            or not _same(summary.get("contract"), contract)):
        raise ValueError("caller-declared historical protocol identity differs")
    if not _same(protocol.get("source_sha256"), sources):
        raise ValueError("protocol source map differs from pinned summary integrity map")
    remappings = {}
    for name, digest in sources.items():
        original, archived = root / name, directory / "source" / name
        if original == archived or str(original) not in inputs or str(archived) not in inputs:
            raise ValueError("source and original archive copy must both be summary-bound: " + name)
        if inputs[str(original)] != digest or inputs[str(archived)] != digest:
            raise ValueError("source/archive digest differs from summary source map: " + name)
        remappings[str(original)] = {"source_relative_path": name,
                                    "archived_path": str(archived), "sha256": digest}
    if any(row["archived_path"] in remappings for row in remappings.values()):
        raise ValueError("source remapping chains are forbidden")

    protocol_hash = inputs[str(directory / "protocol.json")]
    preparation = bound_json("preparation.json")
    verification = bound_json("verification.json")
    report = bound_json("evaluation/report.json")
    if (preparation.get("status") != "prepared"
            or preparation.get("protocol_sha256") != protocol_hash
            or preparation.get("neural_training_or_inference") is not False):
        raise ValueError("completed protocol-bound preparation required")
    if (verification.get("schema") != expected.results_schema or verification.get("status") != "completed"
            or verification.get("protocol_sha256") != protocol_hash
            or verification.get("automatic_promotion") is not False
            or verification.get("neural_training_or_inference") is not False
            or type(verification.get("arms")) is not dict or set(verification["arms"]) != set(expected.arms)):
        raise ValueError("completed protocol-bound verification required")
    if (report.get("schema") != expected.results_schema or report.get("status") != "completed"
            or report.get("protocol_sha256") != protocol_hash
            or report.get("verification_sha256") != inputs[str(directory / "verification.json")]
            or report.get("automatic_promotion") is not False or report.get("base_checkpoints_unchanged") is not True
            or type(report.get("results")) is not dict or set(report["results"]) != set(expected.arms)
            or type(summary.get("arms")) is not dict or set(summary["arms"]) != set(expected.arms)):
        raise ValueError("completed protocol-bound evaluation required")
    for arm in expected.arms:
        receipt = bound_json(arm + "/receipt.json")
        arm_verification = verification["arms"][arm]
        if (receipt.get("schema") != expected.protocol_schema or receipt.get("status") != "completed"
                or receipt.get("arm") != arm or receipt.get("protocol_sha256") != protocol_hash
                or type(receipt.get("retained_updates")) is not int
                or receipt["retained_updates"] != expected.checkpoints[-1]
                or receipt.get("physical_work_unknown") is not False or receipt.get("automatic_promotion") is not False
                or type(arm_verification) is not dict
                or arm_verification.get("exact_official_checkpoint_restores") is not True
                or not _same(arm_verification.get("receipt"), receipt)):
            raise ValueError("completed arm and declared checkpoint verification required: " + arm)
    for name, digest in sorted(inputs.items()):
        target = remappings[name]["archived_path"] if name in remappings else name
        bind(Path(target), digest)
    for name, before in fingerprints.items():
        if _fingerprint(_physical(Path(name))) != before:
            raise ValueError("evidence changed during verification: " + name)
    if file_hash(_physical(summary_path)) != expected.summary_sha256:
        raise ValueError("summary changed during verification")
    if file_hash(_physical(verifier_path)) != verifier_sha256:
        raise ValueError("archive verifier changed during verification")
    return dict(schema=SCHEMA, status="verified", created_utc=datetime.now(timezone.utc).isoformat(),
                directory=str(directory), repository_root=str(root), expected=asdict(expected),
                verifier_path=str(verifier_path), verifier_sha256=verifier_sha256,
                summary_sha256=expected.summary_sha256, protocol_sha256=protocol_hash,
                input_file_sha256=inputs, source_sha256=sources, source_input_remapping=remappings,
                logical_input_count=len(inputs), remapped_source_input_count=len(remappings),
                unchanged_input_count=len(inputs) - len(remappings), physical_file_count=len(verified),
                physical_file_sha256=verified, input_bytes_hashed=bytes_hashed,
                wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started,
                neural_training_or_inference=False, checkpoint_deserialization=False,
                canonical_replay_repeated=False, artifact_migration=False, automatic_promotion=False,
                scope="Point-in-time byte authentication of caller-pinned historical evidence; original absolute non-source paths remain binding. No reexecution, regeneration, migration, score validation, or claim about current source equivalence.")


def _receipt_destination(destination, directory):
    destination = _absolute_name(str(Path(destination).absolute()))
    directory = _physical(Path(directory).absolute(), directory=True)
    if destination.is_relative_to(directory):
        raise ValueError("receipt must be outside the historical evidence directory")
    if destination.exists():
        raise ValueError("receipt destination already exists")
    existing_parent = destination.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    _physical(existing_parent, directory=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--input-count", required=True, type=int)
    parser.add_argument("--source-count", required=True, type=int)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    destination = _receipt_destination(args.receipt, args.directory)
    expected = ExpectedEvidence(args.summary_sha256, args.input_count, args.source_count)
    result = verify_archive(args.directory, repository_root=args.repository_root, expected=expected)
    _receipt_destination(destination, args.directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "logical_input_count",
          "remapped_source_input_count", "unchanged_input_count", "wall_seconds", "cpu_seconds")}))


if __name__ == "__main__":
    main()
