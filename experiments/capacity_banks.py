"""New main/calibration banks for shared-learner width comparisons.

The immutable composition grammar and independent truth checks are reused.
Only exact historical observation transcripts are excluded; old syntactic motifs
and familiar alias vocabulary may recur. No model evaluation or training occurs.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from experiments import composition_curriculum as curriculum
from experiments.composition_banks import _check_pair, known_composed_queries
from experiments.realization_banks import (
    _anonymous_realization, _check_familiar, _map_id, _recipe, _stats,
    transcript_digest,
)

SCHEMA = "bic-capacity-banks-v1"
FAMILIES = curriculum.FAMILIES
LENGTHS = (8, 10, 12)
PANELS = ("name_only", "value_only", "both", "composed")
STAGES = ("main", "calibration")
ROLES = {"main": ("train", "development", "audit"),
         "calibration": ("train", "development")}
SEEDS = {"main": {"train": 210_000_000, "development": 220_000_000, "audit": 230_000_000},
         "calibration": {"train": 240_000_000, "development": 250_000_000}}
NAMING_OFFSET, VALUE_OFFSET, COMPOSED_OFFSET = 100_000_000, 200_000_000, 300_000
MAX_ATTEMPTS_PER_PAIR = 1_000
HISTORICAL_DIRECTORY = Path(__file__).resolve().parents[1] / "runs" / "realization-study-local"


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf8")).hexdigest()


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha_values(values, name):
    if not isinstance(values, (list, tuple, set, frozenset)) or any(
            type(v) is not str or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in values):
        raise ValueError(f"{name} requires canonical lowercase SHA256 strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate SHA256 strings")
    return set(values)


def _stage_banks(stage):
    yield from (rows for buckets in stage["train"].values() for rows in buckets.values())
    for role in ("development", "audit"):
        if role in stage:
            yield from stage[role].values()


def _all_banks(banks):
    for stage in STAGES:
        yield from _stage_banks(banks[stage])


def _historical_spec():
    """Bind earlier files to their completed verified summary, before loading."""
    root = HISTORICAL_DIRECTORY
    decision = json.loads((root / "decision.json").read_text(encoding="utf8"))
    summary_path = root / "audit" / "summary.json"
    summary_hash = _file_digest(summary_path)
    if summary_hash != decision["evidence_sha256"]["audit/summary.json"]:
        raise ValueError("historical verified summary digest differs from decision")
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    identities = summary["verification"]["input_file_sha256"]
    protocol_path = root / "protocol.json"
    if _file_digest(protocol_path) != identities["protocol.json"]:
        raise ValueError("historical protocol digest differs from verified summary")
    protocol = json.loads(protocol_path.read_text(encoding="utf8"))
    checkpoints = protocol["checkpoints"]
    if (not isinstance(checkpoints, list) or not checkpoints
            or any(type(n) is not int or n < 0 for n in checkpoints)
            or checkpoints != sorted(set(checkpoints))
            or checkpoints[-1] != protocol["updates"]):
        raise ValueError("historical checkpoint budget is malformed")
    relative = ["protocol.json", "banks.pt"]
    for job in ("fixed", "fresh"):
        relative += [f"verification/{job}.json", f"main/{job}/latest.pt"]
        relative += [f"main/{job}/checkpoint-{step:06d}.pt" for step in checkpoints]
    files = [(str(root / name), identities[name]) for name in relative]
    old = summary["verification"]["historical_evidence"]
    old_root = root.parent / "composition-study-local"
    files += [(str(old_root / "protocol.json"), old["protocol_sha256"]),
              (str(old_root / "banks.pt"), old["banks_sha256"])]
    for path, expected in files:
        if _file_digest(path) != expected:
            raise ValueError(f"historical file digest differs from verified summary: {Path(path).name}")
    if protocol["banks_file_sha256"] != identities["banks.pt"]:
        raise ValueError("historical bank protocol binding differs")
    return root, summary_hash, tuple(files)


@lru_cache(maxsize=2)
def _read_historical(directory, summary_hash, files):
    """Read only admitted text and cumulative transcript indices, never infer."""
    import torch
    from experiments.realization_training import stream_evidence
    root = Path(directory)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf8"))
    union, evidence = set(), {"summary_sha256": summary_hash, "files_sha256": dict(files),
                             "bank_rows": {}, "checkpoint_streams": []}
    for label, path in (("composition", root.parent / "composition-study-local" / "banks.pt"),
                        ("realization", root / "banks.pt")):
        banks = torch.load(path, map_location="cpu", weights_only=True)
        rows = [row for values in _stage_banks(banks) for row in values]
        observed = {transcript_digest(row) for row in rows}
        if len(observed) != len(rows):
            raise ValueError("historical bank contains duplicate exact transcripts")
        union.update(observed)
        evidence["bank_rows"][label] = {"episodes": len(rows), "unique_transcripts": len(observed)}
    for job in ("fixed", "fresh"):
        verified = json.loads((root / "verification" / f"{job}.json").read_text(encoding="utf8"))
        if verified["protocol_sha256"] != dict(files)[str(root / "protocol.json")]:
            raise ValueError("historical stream verifier belongs to another protocol")
        previous = set()
        names = [f"checkpoint-{step:06d}.pt" for step in protocol["checkpoints"]] + ["latest.pt"]
        for name in names:
            payload = torch.load(root / "main" / job / name, map_location="cpu", weights_only=True)
            if payload["job"] != job or payload["protocol"] != protocol:
                raise ValueError("historical checkpoint provenance differs")
            step = payload["training"]["updates"]
            if step not in protocol["checkpoints"] or (name == "latest.pt" and step != protocol["updates"]):
                raise ValueError("historical checkpoint step differs")
            stream = payload["training"]["realization"]
            # Sealed JSON stringifies integer length-bucket keys from torch payloads.
            if json.loads(_json(stream_evidence(stream))) != verified["replayed_streams"][str(step)]:
                raise ValueError("historical consumed stream differs from sealed canonical replay")
            seen = _sha_values(stream["seen_transcripts"], "historical consumed stream")
            if not previous <= seen:
                raise ValueError("historical consumed stream lost earlier observations")
            previous = seen
            union.update(seen)
            evidence["checkpoint_streams"].append({"job": job, "file": name, "updates": step,
                "consumed_unique_transcripts": len(seen), "consumed_sha256": _digest(sorted(seen))})
    evidence.update({"unique_transcripts": len(union), "transcripts_sha256": _digest(sorted(union)),
        "scope": "All earlier composition and realization banks plus cumulative consumed streams from every saved fixed/fresh realization checkpoint, including latest. No historical motif novelty is asserted."})
    return frozenset(union), _json(evidence)


def historical_evidence():
    root, summary_hash, files = _historical_spec()
    _, details = _read_historical(str(root), summary_hash, files)
    return json.loads(details)


def historical_transcript_digests():
    root, summary_hash, files = _historical_spec()
    values, _ = _read_historical(str(root), summary_hash, files)
    return sorted(values)


def protected_transcripts(banks, *, stage="main", consumed_calibration=()):
    """Canonical exclusions; main also binds planned/verified calibration draws.

    The caller must authenticate or prospectively reconstruct the supplied
    calibration stream. This function checks its digest values, not its origin.
    Both stages protect every evaluation row and the other stage's originals.
    """
    if stage not in STAGES:
        raise ValueError("protection stage must be main or calibration")
    consumed = _sha_values(consumed_calibration, "consumed calibration")
    if stage == "calibration" and consumed:
        raise ValueError("calibration protection cannot include consumed_calibration")
    values = set(historical_transcript_digests()) | consumed
    for source in STAGES:
        for role in ROLES[source]:
            if role == "train":
                if source == stage:
                    continue
                rows = (row for bs in banks[source][role].values() for rs in bs.values() for row in rs)
            else:
                rows = (row for rs in banks[source][role].values() for row in rs)
            values.update(transcript_digest(row) for row in rows)
    initial = {transcript_digest(row) for bs in banks[stage]["train"].values() for rs in bs.values() for row in rs}
    if initial & values:
        raise ValueError("protected transcripts overlap the selected stage's initial training bank")
    return sorted(values)


protected_transcript_digests = protected_transcripts


def bank_manifest(banks):
    return {stage: {"train": {f: {str(t): _stats(rows) for t, rows in bs.items()}
                                for f, bs in banks[stage]["train"].items()},
                   **{role: {name: _stats(rows) for name, rows in banks[stage][role].items()}
                      for role in ROLES[stage] if role != "train"}} for stage in STAGES}


def verify_boundaries(banks):
    """Authenticate coupled controls and exclusions across both study stages."""
    if not isinstance(banks, dict) or set(banks) != set(STAGES):
        raise ValueError("expected main/calibration banks")
    expected = {f"{p}/{f}/t{t}" for p in PANELS for f in FAMILIES for t in LENGTHS}
    for stage in STAGES:
        group = banks[stage]
        if not isinstance(group, dict) or set(group) != set(ROLES[stage]):
            raise ValueError("capacity stage roles differ from declaration")
        if set(group["train"]) != set(FAMILIES) or any(set(bs) != set(LENGTHS) for bs in group["train"].values()):
            raise ValueError("training requires every family/length bucket")
        if any(set(group[role]) != expected for role in ROLES[stage] if role != "train"):
            raise ValueError("evaluation panel set differs from declaration")
    if any(not isinstance(rows, (list, tuple)) or not rows or len(rows) % 2 for rows in _all_banks(banks)):
        raise ValueError("every bank requires nonempty adjacent complete pairs")
    historical = set(historical_transcript_digests())
    exact, train_queries, dev_queries = set(), set(), set()
    composed = {"development": set(), "audit": set()}
    counts = Counter()

    def admit(rows, stage, role, family, turns, partition):
        admission = "dev" if role == "development" else role
        for i in range(0, len(rows), 2):
            _check_pair(rows[i:i+2], family=family, turns=turns, admission=admission, structure_split=partition)
        for row in rows:
            digest = transcript_digest(row)
            if digest in exact or digest in historical:
                raise ValueError("exact transcript reused across current stages or historical observations")
            exact.add(digest)
            queries = known_composed_queries(row)
            if role == "train":
                train_queries.update(q["structure_id"] for q in queries)
            elif role == "development":
                if any(q["structure_partition"] == "audit" for q in queries):
                    raise ValueError("development exposes supervised audit ancestry")
                dev_queries.update(q["structure_id"] for q in queries)
            counts[f"{stage}/{role}"] += 1

    for stage in STAGES:
        for family, buckets in banks[stage]["train"].items():
            for turns, rows in buckets.items():
                admit(rows, stage, "train", family, turns, "train")
        for role in ROLES[stage][1:]:
            partition = "dev" if role == "development" else "audit"
            for family in FAMILIES:
                for turns in LENGTHS:
                    source = banks[stage]["train"][family][turns]
                    familiar = {p: banks[stage][role][f"{p}/{family}/t{turns}"] for p in PANELS[:3]}
                    lengths = {len(rows) for rows in familiar.values()}
                    if len(lengths) != 1 or next(iter(lengths)) > len(source):
                        raise ValueError("familiar panels must use matching first training recipes")
                    for i in range(0, next(iter(lengths)), 2):
                        _check_familiar(source[i:i+2], {p: rows[i:i+2] for p, rows in familiar.items()})
                    for panel in PANELS:
                        rows = banks[stage][role][f"{panel}/{family}/t{turns}"]
                        admit(rows, stage, role, family, turns, partition if panel == "composed" else "train")
                        if panel == "composed":
                            composed[role].update(row["structure_id"] for row in rows)
    if composed["development"] & train_queries or composed["audit"] & (train_queries | dev_queries):
        raise ValueError("held final ancestry overlaps an earlier supervised composition")
    return {"episodes": dict(counts), "unique_exact_transcripts": len(exact),
        "historical_excluded_transcripts": len(historical), "training_known_composed_query_ids": sorted(train_queries),
        "development_known_composed_query_ids": sorted(dev_queries),
        "development_final_composed_ids": sorted(composed["development"]),
        "audit_final_composed_ids": sorted(composed["audit"]),
        "global_exact_transcript_disjoint": True, "historical_exact_transcript_disjoint": True,
        "supervised_composed_query_boundary_verified": True, "familiar_factor_controls_verified": True}


def prepare_banks(*, with_diagnostics=False, train_pairs=256, panel_pairs=64,
                  calibration_train_pairs=64, calibration_panel_pairs=32):
    """Build new CPU-only canonical banks with bounded deterministic rejection."""
    if type(with_diagnostics) is not bool:
        raise ValueError("with_diagnostics must be boolean")
    sizes = {"main": (train_pairs, panel_pairs), "calibration": (calibration_train_pairs, calibration_panel_pairs)}
    if any(type(t) is not int or type(p) is not int or not 1 <= p <= t for t, p in sizes.values()):
        raise ValueError("positive panel pair count must not exceed its stage training count")
    historical = set(historical_transcript_digests())
    exact = set(historical)
    banks = {stage: {"train": {f: {} for f in FAMILIES},
                    **{role: {} for role in ROLES[stage][1:]}} for stage in STAGES}
    selections = []

    def build(stage, role, family, turns, count, familiar=False):
        admission = "dev" if role == "development" else role
        partition = "train" if role == "train" or familiar else admission
        panels = PANELS[:3] if familiar else ("train" if role == "train" else "composed",)
        base = SEEDS[stage][role] + FAMILIES.index(family)*1_000_000 + turns*10_000
        if panels == ("composed",):
            base += COMPOSED_OFFSET
        output, recipes = {p: [] for p in panels}, {p: [] for p in panels}
        attempts, rejected = 0, Counter()
        while len(next(iter(output.values()))) < count*2:
            if attempts >= max(1_000, count*MAX_ATTEMPTS_PER_PAIR):
                raise ValueError("insufficient disjoint capacity realizations within bounded search")
            index = len(next(iter(output.values())))//2
            candidate = base + attempts
            attempts += 1
            if familiar:
                source = banks[stage]["train"][family][turns][index*2:index*2+2]
                original = source[0]["recipe"]
                names, values = candidate+NAMING_OFFSET, candidate+VALUE_OFFSET
                coordinates = {"name_only": (names, original["value_seed"]),
                               "value_only": (original["naming_seed"], values), "both": (names, values)}
                pairs = {p: curriculum.generate_pair(family, original["seed"], split=admission, turns=turns,
                         naming_seed=n, value_seed=v, structure_split="train") for p,(n,v) in coordinates.items()}
                try:
                    _check_familiar(source, pairs)
                except ValueError:
                    rejected["unchanged_realization"] += 1
                    continue
            else:
                pairs = {panels[0]: curriculum.generate_pair(family, candidate, split=admission, turns=turns,
                         naming_seed=candidate+NAMING_OFFSET, value_seed=candidate+VALUE_OFFSET,
                         structure_split=partition)}
            for pair in pairs.values():
                _check_pair(pair, family=family, turns=turns, admission=admission, structure_split=partition)
            if role == "development" and any(q["structure_partition"] == "audit"
                    for pair in pairs.values() for row in pair for q in known_composed_queries(row)):
                rejected["development_audit_query"] += 1
                continue
            digests = [transcript_digest(row) for pair in pairs.values() for row in pair]
            if len(set(digests)) != len(digests) or any(d in exact for d in digests):
                rejected["historical_transcript" if any(d in historical for d in digests) else "current_transcript"] += 1
                continue
            exact.update(digests)
            for panel,pair in pairs.items():
                output[panel].extend(deepcopy(pair)); recipes[panel].append(_recipe(pair))
        selections.append({"stage": stage, "role": role, "family": family, "turns": turns,
            "panels": list(panels), "base_seed": base, "requested_pairs_per_panel": count,
            "candidate_attempts": attempts, "rejected_pairs_or_factor_triplets": dict(rejected), "recipes": recipes})
        return output

    for stage in STAGES:
        for family in FAMILIES:
            for turns in LENGTHS:
                banks[stage]["train"][family][turns] = build(stage,"train",family,turns,sizes[stage][0])["train"]
        for role in ROLES[stage][1:]:
            for family in FAMILIES:
                for turns in LENGTHS:
                    for familiar in (True, False):
                        values = build(stage,role,family,turns,sizes[stage][1],familiar)
                        banks[stage][role].update({f"{p}/{family}/t{turns}": rows for p,rows in values.items()})
    details = {"schema": SCHEMA, "seed_namespaces": deepcopy(SEEDS),
        "family_seed_stride": 1_000_000, "length_seed_stride": 10_000,
        "naming_seed_offset": NAMING_OFFSET, "value_seed_offset": VALUE_OFFSET,
        "composed_panel_offset": COMPOSED_OFFSET, "pairs": {k:{"train":v[0],"evaluation":v[1]} for k,v in sizes.items()},
        "selection": selections, "historical_evidence": historical_evidence(), "boundaries": verify_boundaries(banks),
        "factor_note": "Each stage has its own first training procedures. Familiar triplets preserve procedure/ancestry and unchanged seeds; changed name/value coordinates are coupled.",
        "protection_note": "All current evaluation and other-stage originals are protected. Main protection must additionally include the authenticated or prospectively reconstructed consumed calibration stream before main optimization.",
        "scope_note": "Historical exact observation transcripts are excluded, including cumulative consumed fixed/fresh indices. Historical ancestry IDs may recur; aliases and grammar remain familiar.",
        "semantic_note": "Name maps, values and syntactic dependency motifs do not prove distinct semantic algorithms. Fresh values change increments, truth and byte lengths."}
    return (banks, details) if with_diagnostics else banks
