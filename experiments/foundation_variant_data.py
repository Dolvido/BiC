"""Prospective, model-free data boundary for the flat/hierarchical comparison.

The completed historical summary authenticates old evidence; no old lesson
replay or checkpoint deserialization is needed. Independent verification scans
the newly declared plans and regenerates their evaluation banks. Ordinary loads
authenticate immutable files, not canonical generation. In-process plan indexes
are constructed once per stage and can then be shared across variant trainers.
Nothing in this module trains, scores, selects or promotes a learner.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch

from experiments import foundation_admission as admission
from experiments import foundation_banks as evaluation
from experiments import foundation_plan as planning
from experiments.foundation_curriculum import VERSION
from experiments.foundation_evidence import json_digest, prepare_training, reconstruct_anchor, transcript_set
from experiments.foundation_plan_index import AuthenticatedPlanIndex, source_hashes as index_sources
from experiments.realization_banks import transcript_digest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-foundation-variant-data-v1"
STAGES = ("calibration", "main")
OLD_SUMMARY_SHA256 = "188b0d026c4debce596f3cca5a1b64f6f5f0e0e3992dd4174904a33b6c3e87c2"
OLD_INPUT_COUNT = 424
OLD_SOURCE_COUNT = 79
PLAN_OPTIONS = {
    "calibration": dict(seed=841000001, stage_updates=64, final_updates=126,
                        micro_batch_size=32, rehearsal_every=4, ordering_seed=8410),
    "main": dict(seed=842000001, stage_updates=384, final_updates=768,
                 micro_batch_size=32, rehearsal_every=4, ordering_seed=8420),
}
ANCHOR_LIMITS = {"calibration": 16, "main": 32}
EVALUATIONS = (("calibration", "dev", 843000001, 16),
               ("main", "dev", 844000001, 16), ("main", "audit", 845000001, 32))
FIT_PAIRS = 16
STAGE_FILES = {
    "plan": "plan.json", "admission": "admission.json", "manifest": "training-manifest.json",
    "training_transcripts": "training-transcripts.pt",
    "admission_protected_transcripts": "admission-protected.pt",
    "protected_transcripts": "protected.pt", "banks": "banks.pt", "bank_reports": "bank-admission.json",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    names = set(index_sources()) | {"experiments/foundation_variant_data.py", "experiments/foundation_banks.py"}
    return {name: file_hash(ROOT / name) for name in sorted(names)}


def contract():
    return dict(schema=SCHEMA, curriculum_version=VERSION, order="curriculum",
        plans=copy.deepcopy(PLAN_OPTIONS), anchor_limits=dict(ANCHOR_LIMITS),
        evaluations=[dict(stage=stage, role=role, seed=seed, pairs_per_cell=pairs)
                     for stage, role, seed, pairs in EVALUATIONS], fit_pairs_per_cell=FIT_PAIRS,
        old_summary_sha256=OLD_SUMMARY_SHA256, old_input_count=OLD_INPUT_COUNT,
        old_source_count=OLD_SOURCE_COUNT, automatic_promotion=False,
        scope="Exact complete-observation transcript exclusions, not new vocabulary, semantic algorithms, or historically unseen ancestry partitions. Both architectures consume identical admitted lesson recipes.")


def _same(left, right, message):
    if json_digest(left) != json_digest(right):
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _relative(directory, name):
    if type(name) is not str or "\\" in name or Path(name).is_absolute():
        raise ValueError("canonical relative artifact path required")
    result = (Path(directory) / name).resolve()
    if not result.is_relative_to(Path(directory).resolve()) or result == Path(directory).resolve():
        raise ValueError("artifact path escapes its data directory")
    if result.relative_to(Path(directory).resolve()).as_posix() != name:
        raise ValueError("noncanonical artifact path")
    return result


def _digests(path):
    values = torch.load(path, map_location="cpu", weights_only=True)
    actual = transcript_set(values)
    if type(values) is not list or values != sorted(actual):
        raise ValueError("sorted unique transcript list required")
    return values


def _historical(directory, *, full):
    """Only the full gate hashes old checkpoint bytes; neither mode loads them."""
    directory = Path(directory).resolve()
    summary_path = directory / "evaluation/summary.json"
    if file_hash(summary_path) != OLD_SUMMARY_SHA256:
        raise ValueError("historical foundation summary differs from the declared completed evidence")
    summary = _read(summary_path)
    integrity = summary.get("integrity", {})
    inputs, sources = integrity.get("input_file_sha256"), integrity.get("source_sha256")
    if (summary.get("schema") != "bic-foundation-pilot-summary-v1"
            or summary.get("status") != "completed_descriptive"
            or summary.get("automatic_promotion") is not False
            or integrity.get("canonical_replay_repeated") is not False
            or integrity.get("neural_training_or_inference") is not False
            or type(inputs) is not dict or len(inputs) != OLD_INPUT_COUNT
            or type(sources) is not dict or len(sources) != OLD_SOURCE_COUNT
            or any(type(name) is not str or not Path(name).is_absolute() or not _sha(digest)
                   for name, digest in inputs.items())
            or any(not _sha(digest) for digest in sources.values())):
        raise ValueError("historical summary completion or input/source contract differs")
    chosen = {}
    for name in ("protected.pt", "training-transcripts.pt"):
        path = directory / name
        matching = [digest for location, digest in inputs.items() if Path(location).resolve() == path]
        if len(matching) != 1 or file_hash(path) != matching[0]:
            raise ValueError("historical transcript artifact is not uniquely summary-bound")
        chosen[str(path)] = matching[0]
    if full:
        for name, digest in inputs.items():
            if file_hash(name) != digest:
                raise ValueError("historical input changed: " + name)
        for name, digest in sources.items():
            if file_hash(_relative(ROOT, name)) != digest:
                raise ValueError("historical frozen source changed: " + name)
    history = sorted(set(_digests(directory / "protected.pt")) | set(_digests(directory / "training-transcripts.pt")))
    if file_hash(summary_path) != OLD_SUMMARY_SHA256 or any(file_hash(name) != digest for name, digest in chosen.items()):
        raise ValueError("historical boundary changed while reading")
    receipt = dict(schema=SCHEMA, directory=str(directory), summary_sha256=OLD_SUMMARY_SHA256,
        input_file_sha256=inputs, source_sha256=sources, transcript_file_sha256=chosen,
        union_count=len(history), union_sha256=json_digest(history))
    return history, receipt


def _evaluation_banks(stages, history, specification):
    all_training = set().union(*(set(stages[stage]["training_transcripts"]) for stage in STAGES))
    excluded = set(history) | all_training
    for stage in STAGES:
        stages[stage]["banks"], stages[stage]["bank_reports"] = {}, {}
    for recipe in specification["evaluations"]:
        stage, role = recipe["stage"], recipe["role"]
        data = stages[stage]
        banks, report = evaluation.build_evaluation(data["plan"], data["manifest"],
            training_transcripts=data["training_transcripts"], role=role, seed=recipe["seed"],
            pairs_per_cell=recipe["pairs_per_cell"], excluded_transcripts=sorted(excluded))
        data["banks"][role], data["bank_reports"][role] = banks, report
        admitted = transcript_set(report["transcript_sha256"])
        if admitted & excluded:
            raise ValueError("evaluation overlaps a prior role or training/history")
        excluded.update(admitted)
    main = stages["main"]
    fit, references = {}, {}
    for cell, anchors in sorted(main["manifest"]["anchors"].items()):
        count = specification["fit_pairs_per_cell"]
        if len(anchors) < count:
            raise ValueError("insufficient canonical fit anchors")
        refs = copy.deepcopy(anchors[:count])
        name = "fit/" + cell
        references[name] = refs
        fit[name] = [row for ref in refs for row in reconstruct_anchor(main["plan"], ref)]
    main["banks"]["train_fit"] = fit
    main["bank_reports"]["train_fit"] = dict(schema=SCHEMA, role="train_fit",
        pairs_per_cell=specification["fit_pairs_per_cell"], anchors=references,
        banks={name: evaluation._stats(rows) for name, rows in fit.items()},
        scope="First canonical training-manifest anchors per cell; observed training fit, not generalization.")
    evaluation_hashes = excluded - set(history) - all_training
    for stage in STAGES:
        other = STAGES[1 - STAGES.index(stage)]
        protection = set(history) | set(stages[other]["training_transcripts"]) | evaluation_hashes
        if (protection & set(stages[stage]["training_transcripts"])
                or not set(stages[stage]["admission_protected_transcripts"]) <= protection):
            raise ValueError("expanded protection differs from stage admission boundary")
        stages[stage]["protected_transcripts"] = sorted(protection)


def _construct(history, specification):
    stages, blocked = {}, set(history)
    for stage in STAGES:
        base = planning.build_plan(**specification["plans"][stage])
        original = sorted(blocked)
        plan, receipt = admission.repair_plan(base, protected_transcripts=original)
        manifest, transcripts = prepare_training(plan, protected_transcripts=original,
            anchor_limit=specification["anchor_limits"][stage])
        stages[stage] = dict(plan=plan, admission=receipt, manifest=manifest,
            training_transcripts=transcripts, admission_protected_transcripts=original)
        blocked.update(transcripts)
    _evaluation_banks(stages, history, specification)
    return stages


def _stage_identity(data):
    return {name: json_digest(data[name]) for name in STAGE_FILES}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pt":
        with path.open("xb") as handle:
            torch.save(value, handle)
    else:
        with path.open("x", encoding="utf8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")


def prepare(directory, old_foundation_dir):
    """Reserve a new directory; preserve every partial artifact on failure."""
    started = time.monotonic()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources, specification = source_hashes(), contract()
    try:
        history, historical = _historical(old_foundation_dir, full=True)
        stages = _construct(history, specification)
        _write(directory / "history.pt", history)
        _write(directory / "history-receipt.json", historical)
        names = {"history.pt", "history-receipt.json"}
        for stage, data in stages.items():
            for key, name in STAGE_FILES.items():
                relative = f"{stage}/{name}"
                _write(directory / relative, data[key]); names.add(relative)
        for name, digest in sources.items():
            content = (ROOT / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("source changed during preparation")
            snapshot = directory / "source" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            with snapshot.open("xb") as handle:
                handle.write(content)
            names.add("source/" + name)
        _same(source_hashes(), sources, "source changed during data preparation")
        _same(_historical(old_foundation_dir, full=False), (history, historical), "history changed during preparation")
        manifest = dict(schema=SCHEMA, contract=specification, source_sha256=sources,
            historical_receipt_sha256=json_digest(historical),
            artifacts_sha256={name: file_hash(directory / name) for name in sorted(names)},
            stages={stage: _stage_identity(data) for stage, data in stages.items()},
            history=dict(count=len(history), sha256=json_digest(history)),
            neural_training_or_inference=False, automatic_promotion=False)
        _write(directory / "manifest.json", manifest)
        _write(directory / "preparation.json", dict(schema=SCHEMA, status="completed",
            created_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-started,
            manifest_sha256=file_hash(directory / "manifest.json"),
            source_sha256=sources, neural_training_or_inference=False))
        return manifest
    except BaseException as error:
        _write(directory / "failure.json", dict(schema=SCHEMA, status="failed",
            error=repr(error), wall_seconds=time.monotonic()-started, neural_training_or_inference=False))
        raise


def _validate_loaded(data):
    manifest, stages, history = data["manifest"], data["stages"], data["history"]
    specification = manifest["contract"]
    _same(manifest["history"], dict(count=len(history), sha256=json_digest(history)), "history identity differs")
    expected_admission = set(history)
    all_training, all_evaluation = set(), set()
    for stage in STAGES:
        record = stages[stage]
        planning.validate_plan(record["plan"])
        _same(record["plan"]["config"], specification["plans"][stage], "stage plan configuration differs")
        _same(_stage_identity(record), manifest["stages"][stage], "stage artifact semantic identity differs")
        _same(record["admission_protected_transcripts"], sorted(expected_admission), "stage admission history differs")
        own = transcript_set(record["training_transcripts"])
        if own & expected_admission:
            raise ValueError("training stage overlaps historical or earlier training transcripts")
        evidence = record["manifest"]
        if (evidence["plan_sha256"] != json_digest(record["plan"])
                or evidence["anchor_limit"] != specification["anchor_limits"][stage]
                or evidence["unique_transcripts"] != len(own)
                or evidence["transcript_sha256"] != json_digest(sorted(own))
                or evidence["protected_sha256"] != json_digest(sorted(expected_admission))
                or evidence["protected_count"] != len(expected_admission)):
            raise ValueError("training evidence boundary differs")
        expected_admission.update(own); all_training.update(own)
    excluded = set(history) | all_training
    expected_roles = {"calibration": {"dev"}, "main": {"dev", "audit", "train_fit"}}
    for stage in STAGES:
        if set(stages[stage]["banks"]) != expected_roles[stage] or set(stages[stage]["bank_reports"]) != expected_roles[stage]:
            raise ValueError("stage evaluation roles differ")
    for recipe in specification["evaluations"]:
        stage, role = recipe["stage"], recipe["role"]
        bank, report = stages[stage]["banks"][role], stages[stage]["bank_reports"][role]
        values = [transcript_digest(row) for rows in bank.values() for row in rows]
        unique = set(values)
        if len(unique) != len(values) or unique & excluded:
            raise ValueError("evaluation observations overlap another data role")
        _same(sorted(unique), report["transcript_sha256"], "evaluation transcript identities differ")
        _same(report["external_exclusions"], dict(count=len(excluded), sha256=json_digest(sorted(excluded))), "evaluation cumulative exclusions differ")
        if report["role"] != role or report["seed"] != recipe["seed"] or report["pairs_per_cell"] != recipe["pairs_per_cell"]:
            raise ValueError("evaluation recipe differs")
        excluded.update(unique); all_evaluation.update(unique)
    for stage in STAGES:
        other = STAGES[1 - STAGES.index(stage)]
        expected = set(history) | set(stages[other]["training_transcripts"]) | all_evaluation
        _same(stages[stage]["protected_transcripts"], sorted(expected), "expanded training protection differs")
    fit = stages["main"]["banks"]["train_fit"]
    main_training = set(stages["main"]["training_transcripts"])
    if any(transcript_digest(row) not in main_training for rows in fit.values() for row in rows):
        raise ValueError("fit rows are not admitted main training observations")


def load(directory):
    """Authenticate new immutable files and the small pinned historical boundary.

This is not independent canonical regeneration. The runner must bind the
separate verify() receipt before training. Its manifest hash seals this loader's
data root; standalone unsigned directory contents are not a trust certificate.
"""
    directory = Path(directory).resolve()
    manifest = _read(directory / "manifest.json")
    preparation = _read(directory / "preparation.json")
    if (manifest.get("schema") != SCHEMA or manifest.get("automatic_promotion") is not False
            or manifest.get("neural_training_or_inference") is not False
            or preparation.get("status") != "completed"
            or preparation.get("manifest_sha256") != file_hash(directory / "manifest.json")):
        raise ValueError("completed immutable data preparation required")
    _same(manifest["contract"], contract(), "prospective data contract differs")
    _same(manifest["source_sha256"], source_hashes(), "data source identity changed")
    _same(preparation["source_sha256"], manifest["source_sha256"], "preparation source identity differs")
    expected_files = {"history.pt", "history-receipt.json"}
    expected_files.update(f"{stage}/{name}" for stage in STAGES for name in STAGE_FILES.values())
    expected_files.update("source/"+name for name in manifest["source_sha256"])
    if set(manifest["artifacts_sha256"]) != expected_files or set(manifest["stages"]) != set(STAGES):
        raise ValueError("data artifact or stage inventory differs")
    for name, digest in manifest["artifacts_sha256"].items():
        if not _sha(digest) or file_hash(_relative(directory, name)) != digest:
            raise ValueError("data artifact changed: " + name)
    for name, digest in manifest["source_sha256"].items():
        if manifest["artifacts_sha256"]["source/"+name] != digest:
            raise ValueError("preserved source differs from its declared current identity")
    historical = _read(directory / "history-receipt.json")
    history, current = _historical(historical["directory"], full=False)
    _same(current, historical, "historical receipt differs")
    if json_digest(historical) != manifest["historical_receipt_sha256"]:
        raise ValueError("historical receipt is not manifest-bound")
    _same(history, _digests(directory / "history.pt"), "history union differs")
    stages = {}
    for stage in STAGES:
        stages[stage] = {key: (torch.load(directory/stage/name, map_location="cpu", weights_only=True)
                              if name.endswith(".pt") else _read(directory/stage/name))
                         for key, name in STAGE_FILES.items()}
        for key in ("training_transcripts", "admission_protected_transcripts", "protected_transcripts"):
            values = stages[stage][key]
            if type(values) is not list or values != sorted(transcript_set(values)):
                raise ValueError("canonical sorted stage transcript evidence required")
    data = dict(manifest=manifest, stages=stages, history=history, history_receipt=historical,
                paths=dict(directory=str(directory), manifest=str(directory / "manifest.json")))
    _validate_loaded(data)
    if file_hash(directory / "manifest.json") != preparation["manifest_sha256"]:
        raise ValueError("data manifest changed while loading")
    _same(source_hashes(), manifest["source_sha256"], "data source changed while loading")
    return data


def build_index(directory, stage, *, data=None):
    """One process-authenticated canonical plan; no evaluation-bank regeneration."""
    if type(stage) is not str or stage not in STAGES:
        raise ValueError("explicit calibration or main stage required")
    data = load(directory) if data is None else data
    if Path(data["paths"]["directory"]).resolve() != Path(directory).resolve():
        raise ValueError("loaded data belongs to another directory")
    manifest_path = Path(directory) / "manifest.json"
    prepared = _read(Path(directory) / "preparation.json")
    if prepared.get("status") != "completed" or prepared.get("manifest_sha256") != file_hash(manifest_path):
        raise ValueError("index requires the completed on-disk data manifest")
    _same(data["manifest"], _read(manifest_path), "index caller manifest differs from the prepared artifact")
    _same(data["manifest"]["contract"], contract(), "index data contract differs")
    _same(data["manifest"]["source_sha256"], source_hashes(), "index source identity changed")
    record = data["stages"][stage]
    _same(_stage_identity(record), data["manifest"]["stages"][stage], "index inputs changed after data load")
    return AuthenticatedPlanIndex(record["plan"], admission_receipt=record["admission"],
        admission_protected_transcripts=record["admission_protected_transcripts"],
        protected_transcripts=record["protected_transcripts"])


def verify(directory):
    """Regenerate only new declared evidence, returning reusable live indexes.

The index independently authenticates admission and expanded protection. One
additional training-manifest pass checks complete anchors/counts, then each
evaluation bank is regenerated once. Old models are never deserialized.
"""
    started = time.monotonic()
    sources = source_hashes()
    data = load(directory)
    historical, receipt = _historical(data["history_receipt"]["directory"], full=True)
    _same((historical, receipt), (data["history"], data["history_receipt"]), "full historical gate differs")
    directory = Path(directory).resolve()
    inputs = {name: file_hash(directory / name) for name in
              ("manifest.json", "preparation.json", *data["manifest"]["artifacts_sha256"])}
    indexes, regenerated = {}, {}
    for stage in STAGES:
        record = data["stages"][stage]
        indexes[stage] = build_index(directory, stage, data=data)
        manifest, training = prepare_training(record["plan"],
            protected_transcripts=record["admission_protected_transcripts"],
            anchor_limit=data["manifest"]["contract"]["anchor_limits"][stage])
        _same(manifest, record["manifest"], "canonical training manifest differs")
        _same(training, record["training_transcripts"], "canonical training transcripts differ")
        identity = indexes[stage].identity
        if identity["unique_transcripts"] != len(training) or identity["transcript_sha256"] != json_digest(training):
            raise ValueError("canonical index and training manifest disagree")
        regenerated[stage] = {key: record[key] for key in STAGE_FILES if key not in ("banks", "bank_reports", "protected_transcripts")}
    _evaluation_banks(regenerated, data["history"], data["manifest"]["contract"])
    for stage in STAGES:
        for key in ("banks", "bank_reports", "protected_transcripts"):
            _same(regenerated[stage][key], data["stages"][stage][key], "canonical " + key + " differ")
    _same(source_hashes(), sources, "source changed during independent verification")
    for name, digest in inputs.items():
        if file_hash(directory / name) != digest:
            raise ValueError("data changed during independent verification: " + name)
    _same(_historical(receipt["directory"], full=False), (historical, receipt), "historical boundary changed during verification")
    for name, digest in receipt["source_sha256"].items():
        if file_hash(_relative(ROOT, name)) != digest:
            raise ValueError("historical source changed during verification: " + name)
    report = dict(schema=SCHEMA, status="verified", directory=str(directory),
        source_sha256_before=sources, source_sha256_after=source_hashes(),
        input_file_sha256=inputs, manifest_sha256=inputs["manifest.json"],
        historical_receipt_sha256=json_digest(receipt), historical_full_input_count=len(receipt["input_file_sha256"]),
        historical_full_source_count=len(receipt["source_sha256"]),
        canonical_index={stage: index.identity for stage, index in indexes.items()},
        index_construction={stage: index.construction for stage, index in indexes.items()},
        wall_seconds=time.monotonic()-started, old_canonical_replay=False,
        new_canonical_admission_and_manifests_and_banks=True,
        neural_training_or_inference=False, automatic_promotion=False)
    return dict(data=data, indexes=indexes, report=report)
