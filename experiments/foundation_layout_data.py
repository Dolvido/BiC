"""Prospective source-pinned layout data; no learner or trusted serialized index.

Preparation owns full transcript admission. Runtime recreates a compact plan's
bundle and exposes its complete canonical evidence for comparison to the sealed
manifest. The runner owns that comparison and never substitutes this module for
a source/data boundary. All original/varied examples share canonical parents.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from experiments import foundation_admission as admission
from experiments import foundation_banks as evaluation
from experiments import foundation_curriculum as foundation
from experiments import foundation_evidence as evidence
from experiments import foundation_layout_curriculum as curriculum
from experiments import foundation_layout_training as training
from experiments import foundation_plan as planning
from experiments.realization_banks import transcript_digest
from experiments.train_cognitive import fingerprint_rows


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-foundation-layout-data-v1"
EXPECTED_SCHEMA = "bic-foundation-layout-expected-bundles-v1"
HISTORY_ROOT = "runs/variant-study-local/data"
HISTORY_MANIFEST_SHA256 = "d0170189ffa373c68492972df83ed7e39aaa1c40f2cc5c9520b89454f0672be6"
HISTORY_FILES = {
    "main/protected.pt": "b77641a1a203a10cfdd44fc66bd1c72bdc77f60709f608b8fec14755ea8f72b8",
    "main/training-transcripts.pt": "f28bdfb445b7ef8e73845be83713b0a3bc4055972ff14d7540ead59c1926b213",
}
CONTEXT_BANK = "runs/foundation-context-diagnostic-local/attempt-001/bank.json"
CONTEXT_BANK_SHA256 = "8d74f61c808837d5232c28a64c50966c61c68a8e5464c371accf51f2b5294910"
PLAN_OPTIONS = dict(seed=852020001, stage_updates=96, final_updates=288,
                    micro_batch_size=32, rehearsal_every=4, ordering_seed=852020002)
LAYOUT_NAMESPACE = 852020003
EVALUATIONS = {"dev": 852021001, "audit": 852022001}
PAIRS_PER_CELL = 8
ANCHOR_LIMIT = 8
PAIRED_ADMISSION_SCHEMA = "bic-foundation-layout-paired-admission-v1"
PRIOR_PREPARATION_RECEIPTS = [{"path": "runs/foundation-layout-preparation-local/attempt-001/receipt.json",
    "sha256": "465404d4c9295fddfdb33b546e5c0c3570d41346ceda176ae3690c03abd53e60"}]
PHASE_SECONDS = dict(history=120, admission=600, canonical_evidence=600, layouts=1800, evaluation=1800, publish=180)
PHASES = tuple(PHASE_SECONDS)
_ACTIVE_WORK = None


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def native(path):
    value = str(Path(path).absolute())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def file_hash(path):
    digest = hashlib.sha256()
    with native(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(native(path).read_bytes())


def _write(path, value):
    target = native(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        if target.suffix == ".pt":
            torch.save(value, handle)
        else:
            handle.write((_json(value)+"\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def source_hashes():
    names = set(training.source_hashes()) | {
        "experiments/foundation_layout_data.py", "experiments/foundation_admission.py",
        "experiments/foundation_evidence.py", "experiments/foundation_banks.py",
        "experiments/composition_training.py", "experiments/train_cognitive.py",
        "docs/FOUNDATION_LAYOUT_STUDY_PROTOCOL.md",
    }
    if native(ROOT / "experiments/__init__.py").exists():
        raise ValueError("experiment namespace initializer must remain absent")
    return {name: file_hash(ROOT / name) for name in sorted(names)}


def contract():
    return dict(schema=SCHEMA, curriculum_version=curriculum.VERSION, order="curriculum",
        plan=deepcopy(PLAN_OPTIONS), layouts=list(curriculum.LAYOUTS), layout_namespace=LAYOUT_NAMESPACE,
        training_layout_seed="sha256(canonical JSON [852020003,'train',bundle_id,pair_index]) modulo2**63",
        evaluation_layout_seed="sha256(canonical JSON [852020003,role,bank-name-with-family-token-removed,pair_index]) modulo2**63",
        fit_layout_seed="Exactly the original training bundle_id/pair_index seed",
        evaluations=deepcopy(EVALUATIONS), pairs_per_cell=PAIRS_PER_CELL, anchor_limit=ANCHOR_LIMIT,
        preparation_phase_seconds=deepcopy(PHASE_SECONDS), maximum_preparation_phase_seconds=sum(PHASE_SECONDS.values()),
        expected_training_bundles=864, expected_training_episodes_per_layout=82944,
        history_manifest=dict(path=HISTORY_ROOT+"/manifest.json", sha256=HISTORY_MANIFEST_SHA256),
        history_transcript_files={HISTORY_ROOT+"/"+name: digest for name, digest in HISTORY_FILES.items()},
        context_bank=dict(path=CONTEXT_BANK, sha256=CONTEXT_BANK_SHA256, episodes=2016),
        collision_policy="Same original pair member across its two views may coincide. Different pairs, history and prior roles may not. Only exact original training counterparts in train_fit may overlap training.",
        paired_view_admission=dict(schema=PAIRED_ADMISSION_SCHEMA, maximum_naming_attempt=1000,
            training="Initial canonical-admitted names, then larger existing admission-names-N coordinates through1000; both views commit together.",
            evaluation="Initial accepted canonical problem, then names-only seed coordinates [foundation evaluation schema, role seed, role, cell, pair index, 'layout-admission-names', attempt] through1000.",
            preserves="Procedure, values, typed read-version ancestry, supervision, layout seed and permutation; only aliases change."),
        prior_preparation_receipts=deepcopy(PRIOR_PREPARATION_RECEIPTS),
        no_full_training_rows_serialized=True, no_trusted_index_serialized=True,
        automatic_retry=False, automatic_promotion=False,
        scope="Exact complete-transcript exclusion against the explicit pinned inventory only; no semantic novelty or exhaustive historical exclusion claim.")


@dataclass
class WorkLedger:
    counts: dict = field(default_factory=lambda: dict.fromkeys(("canonical_generate_attempts",
        "canonical_generate_completions", "canonical_rows_returned", "canonical_bundle_attempts",
        "canonical_bundle_completions", "layout_bundles_materialized", "bundle_evidence_calls",
        "weights_only_data_loads", "transcript_rows_checked", "accepted_view_rows",
        "same_member_cross_view_coincidences", "fit_rows_verified", "collision_rejections",
        "paired_view_candidate_pairs", "paired_view_rejected_candidates", "paired_view_name_retries",
        "paired_view_rebound_pairs"), 0))
    layout: curriculum.WorkLedger = field(default_factory=curriculum.WorkLedger)
    generation_errors: dict = field(default_factory=dict)
    phases: dict = field(default_factory=dict)
    paired_admissions: dict = field(default_factory=dict)
    last_bundle_evidence: dict | None = None
    phase: str | None = None
    deadline: float | None = None
    progress: object = None

    def boundary(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("layout data phase allowance expired: "+str(self.phase))

    def event(self, value):
        if self.progress is not None:
            self.progress(dict(phase=self.phase, **value))

    def report(self):
        return dict(counts=deepcopy(self.counts), layout=self.layout.report(),
            generation_errors=deepcopy(self.generation_errors), phases=deepcopy(self.phases),
            paired_admissions=deepcopy(self.paired_admissions),
            scope="Canonical rows returned include copied/reconstructed parents. Layout counts include explicit outputs and validation reconstructions; neither equals unique teaching lessons.")


def _work(value):
    if value is None:
        return WorkLedger()
    if type(value) is not WorkLedger:
        raise ValueError("layout data WorkLedger required")
    return value


@contextmanager
def _tracking(work):
    """Single-process call accounting; never replaces canonical algorithms."""
    global _ACTIVE_WORK
    if _ACTIVE_WORK is work:
        yield
        return
    if _ACTIVE_WORK is not None:
        raise RuntimeError("concurrent layout data accounting owners are not supported")
    original_generate = foundation.generate_pair
    original_bundle = planning._materialize_validated_bundle
    def generated(*args, **kwargs):
        work.boundary()
        work.counts["canonical_generate_attempts"] += 1
        try:
            result = original_generate(*args, **kwargs)
        except BaseException as error:
            key = type(error).__name__+": "+str(error)
            work.generation_errors[key] = work.generation_errors.get(key, 0)+1
            raise
        work.counts["canonical_generate_completions"] += 1
        work.counts["canonical_rows_returned"] += len(result)
        work.boundary()
        return result
    def bundled(plan, bundle_id):
        work.boundary()
        work.counts["canonical_bundle_attempts"] += 1
        result = original_bundle(plan, bundle_id)
        work.counts["canonical_bundle_completions"] += 1
        count = work.counts["canonical_bundle_completions"]
        if count % 48 == 0:
            work.event(dict(event="canonical_bundle_progress", completed_bundle_calls=count, bundle_id=bundle_id,
                            canonical_rows_returned=work.counts["canonical_rows_returned"]))
        work.boundary()
        return result
    _ACTIVE_WORK = work
    foundation.generate_pair, planning._materialize_validated_bundle = generated, bundled
    try:
        yield
    finally:
        foundation.generate_pair, planning._materialize_validated_bundle = original_generate, original_bundle
        _ACTIVE_WORK = None


@contextmanager
def _phase(work, name, seconds):
    started, cpu_started = time.monotonic(), time.process_time()
    before = work.report()
    work.phase, work.deadline = name, started+seconds
    result = dict(status="running", allowance_seconds=seconds)
    work.phases[name] = result
    try:
        work.event(dict(event="phase_start", allowance_seconds=seconds))
        yield
        work.boundary()
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started,
            count_delta={key: value-before["counts"][key] for key, value in work.counts.items()},
            layout_delta={key: value-before["layout"][key] for key, value in work.layout.report().items()})
        work.event(dict(event="phase_end", **deepcopy(result)))
        work.deadline = None


def _verify_files(pins, work=None):
    for name, expected in pins.items():
        if work is not None:
            work.boundary()
        if file_hash(ROOT / name) != expected:
            raise ValueError("pinned source/input changed: "+name)


def _prior_preparation_cost(work):
    pins, records = {}, []
    for reference in PRIOR_PREPARATION_RECEIPTS:
        _verify_files({reference["path"]: reference["sha256"]}, work)
        receipt = _read(ROOT/reference["path"])
        if (receipt.get("schema") != "bic-foundation-layout-preparation-process-v1"
                or receipt.get("status") != "failed" or receipt.get("cuda_initialized_before") is not False
                or not receipt.get("guard_attempts") or any(receipt["guard_attempts"].values())
                or any(type(receipt.get(key)) not in (int, float) or not math.isfinite(receipt[key]) or receipt[key] < 0
                       for key in ("wall_seconds", "cpu_seconds"))):
            raise ValueError("pinned prior failed CPU-only preparation receipt differs")
        pins[reference["path"]] = reference["sha256"]
        records.append(dict(reference, wall_seconds=receipt["wall_seconds"], cpu_seconds=receipt["cpu_seconds"]))
    return pins, dict(receipts=records, wall_seconds=sum(row["wall_seconds"] for row in records),
        cpu_seconds=sum(row["cpu_seconds"] for row in records),
        scope="Prior enclosing preparation process costs; current call and its enclosing process are excluded.")


def _history(work):
    manifest_name = HISTORY_ROOT+"/manifest.json"
    pins = {manifest_name: HISTORY_MANIFEST_SHA256, CONTEXT_BANK: CONTEXT_BANK_SHA256,
            **{HISTORY_ROOT+"/"+name: digest for name, digest in HISTORY_FILES.items()}}
    _verify_files(pins, work)
    manifest = _read(ROOT / manifest_name)
    if (manifest.get("schema") != "bic-foundation-variant-data-v1"
            or manifest.get("neural_training_or_inference") is not False
            or manifest.get("automatic_promotion") is not False):
        raise ValueError("pinned inherited data contract differs")
    union, files = set(), {}
    for name, expected in HISTORY_FILES.items():
        work.boundary()
        if manifest["artifacts_sha256"].get(name) != expected:
            raise ValueError("inherited manifest does not bind selected transcript file")
        with native(ROOT / HISTORY_ROOT / name).open("rb") as handle:
            values = torch.load(handle, map_location="cpu", weights_only=True)
        work.counts["weights_only_data_loads"] += 1
        unique = evidence.transcript_set(values)
        if type(values) is not list or values != sorted(unique):
            raise ValueError("inherited transcript data must be sorted unique strings")
        files[name] = dict(file_sha256=expected, count=len(values), transcript_sha256=evidence.json_digest(values))
        union.update(unique)
    bank = _read(ROOT / CONTEXT_BANK)
    if (bank.get("schema") != "bic-context-sensitivity-bank-v1"
            or len(bank.get("rows", [])) != 2016 or _hash(bank["rows"]) != bank.get("rows_sha256")):
        raise ValueError("pinned context bank schema/row identity differs")
    context = {transcript_digest(row) for row in bank["rows"]}
    if len(context) != len(bank["rows"]):
        raise ValueError("pinned context bank has duplicate transcripts")
    union.update(context)
    _verify_files(pins, work)
    values = sorted(union)
    return values, dict(schema=SCHEMA, input_file_sha256=pins, transcript_files=files,
        context_transcripts=dict(count=len(context), sha256=evidence.json_digest(sorted(context))),
        union_count=len(values), union_sha256=evidence.json_digest(values),
        scope=contract()["scope"])


def layout_seed(role, *, bundle_id=None, pair_index, bank_name=None):
    if type(pair_index) is not int or pair_index < 0:
        raise ValueError("nonnegative pair index required")
    if role == "train":
        if type(bundle_id) is not int or not 0 <= bundle_id < 864 or bank_name is not None:
            raise ValueError("canonical training bundle coordinate required")
        parts = [LAYOUT_NAMESPACE, "train", bundle_id, pair_index]
    elif role in EVALUATIONS:
        tokens = bank_name.split("/") if type(bank_name) is str else []
        if len(tokens) != 5 or tokens[1] not in foundation.FAMILIES or bundle_id is not None:
            raise ValueError("evaluation bank name must contain one canonical family token")
        parts = [LAYOUT_NAMESPACE, role, "/".join([tokens[0], *tokens[2:]]), pair_index]
    else:
        raise ValueError("train/dev/audit layout seed role required")
    return int(_hash(parts), 16) % (2**63)


def _adapt_bundle(families, bundle_id, mode, work):
    work.last_bundle_evidence = None
    adapted = {}
    for family in foundation.FAMILIES:
        rows = families[family]
        adapted[family] = []
        for offset in range(0, len(rows), 2):
            work.boundary()
            adapted[family].extend(curriculum.materialize_pair(rows[offset:offset+2], layout=mode,
                seed=layout_seed("train", bundle_id=bundle_id, pair_index=offset//2), work=work.layout))
    return _bundle(adapted, bundle_id, mode, work)


def _bundle(adapted, bundle_id, mode, work):
    work.last_bundle_evidence = None
    bundle = dict(schema=training.BUNDLE_SCHEMA, bundle_id=bundle_id, layout=mode, families=adapted)
    work.counts["layout_bundles_materialized"] += 1
    work.counts["bundle_evidence_calls"] += 1
    work.last_bundle_evidence = training._bundle_evidence(bundle, cursor=bundle_id, layout=mode,
        micro_batch_size=PLAN_OPTIONS["micro_batch_size"], work=work.layout)
    return bundle


def materialize_bundle(plan, bundle_id, layout, work=None):
    """Return the exact trainer envelope; runner compares last_bundle_evidence.

    This authenticates the compact plan and every lesson, but does not repeat
    historical set scans or certify an unsigned plan as preadmitted. The runner
    must bind the plan/data/source manifest and compare the complete evidence.
    """
    work = _work(work)
    if type(layout) is not str or layout not in curriculum.LAYOUTS:
        raise ValueError("original or varied layout required")
    planning.validate_plan(plan)
    if plan["config"] != PLAN_OPTIONS or "admission" not in plan:
        raise ValueError("fixed admitted study plan required")
    with _tracking(work):
        families = planning._materialize_validated_bundle(plan, bundle_id)
        return _adapt_bundle(families, bundle_id, layout, work)


def _empty_distribution():
    return dict(pairs=0, episodes=0, changed_pairs=0, terminal_kinds={}, feasible_terminal_support={}, anchor_turns={}, by_turn={})


def _add_distribution(destination, pair):
    first = pair[0]
    destination["pairs"] += 1
    destination["episodes"] += 2
    destination["changed_pairs"] += int(first["layout_summary"]["changed"])
    for category, key in (("terminal_kinds", first["recipe"]["selected_terminal_kind"]),
                          ("feasible_terminal_support", "/".join(first["recipe"]["feasible_terminal_kinds"])),
                          ("anchor_turns", str(first["anchor"]["turn_index"]))):
        destination[category][key] = destination[category].get(key, 0)+1
    for row in pair:
        for index, turn in enumerate(row["turns"]):
            item = destination["by_turn"].setdefault(str(index), dict(episodes=0,
                target_counts=dict.fromkeys(map(str, range(4)), 0), observation_utf8_bytes=0, reply_utf8_bytes=0))
            item["episodes"] += 1
            item["target_counts"][str(turn["target"])] += 1
            item["observation_utf8_bytes"] += len(turn["text"].encode("utf-8"))
            item["reply_utf8_bytes"] += len(turn["reply"].encode("utf-8"))


def _admission_record(work, role):
    return work.paired_admissions.setdefault(role, dict(schema=PAIRED_ADMISSION_SCHEMA, role=role,
        status="preparing", accepted_pairs=0, rejections=[], overrides=[],
        accepted_stream_sha256=_hash([PAIRED_ADMISSION_SCHEMA, role])))


def _stage_views(views, location, history, owners, work):
    """Read-only transaction: no rejected member enters global ownership."""
    staged, hashes, conflicts, coincidences = {}, {}, [], 0
    for mode in curriculum.LAYOUTS:
        hashes[mode] = []
        for row in views[mode]:
            work.counts["transcript_rows_checked"] += 1
            digest = transcript_digest(row)
            owner = (*location, row["recipe"]["base_pair_sha256"], row["variant"])
            prior = staged.get(digest, owners.get(digest))
            reasons = []
            if digest in history:
                reasons.append(dict(inventory="explicit_pinned_history_union"))
            if prior is not None and prior != owner:
                reasons.append(dict(inventory="candidate_transaction" if digest in staged else "accepted_pair_owners",
                                    prior_owner=list(prior)))
            if reasons:
                conflicts.append(dict(location=list(location), layout=mode, variant=row["variant"],
                    episode_id=row["id"], transcript_sha256=digest, candidate_owner=list(owner), reasons=reasons))
            elif prior == owner:
                coincidences += 1
            if digest not in staged:
                staged[digest] = owner
            hashes[mode].append(digest)
    return staged, hashes, conflicts, coincidences


def _placement(views):
    return {mode: dict(permutation=pair[0]["recipe"]["permutation"],
        feasible_terminal_kinds=pair[0]["recipe"]["feasible_terminal_kinds"],
        selected_terminal_kind=pair[0]["recipe"]["selected_terminal_kind"],
        query_slots=[[(query["original_turn_index"], query["turn_index"], query["valid_statement_slots"])
                      for query in row["queries"]] for row in pair]) for mode, pair in views.items()}


def _admit_views(initial, location, seed, history, owners, work, rebind, *, start_attempt=0, fresh_relation=None):
    """Preserve one canonical problem; rebind names only after a conflict."""
    record = _admission_record(work, location[0])
    initial_identity = admission._identity(initial)
    initial_meaning, placement = None, None
    for attempt in range(start_attempt, admission.MAX_ATTEMPTS+1):
        work.boundary()
        pair = initial if attempt == start_attempt else rebind(attempt)
        if attempt != start_attempt:
            work.counts["paired_view_name_retries"] += 1
            if admission._meaning(pair) != initial_meaning:
                record.update(status="failed", failed_location=list(location), failure="names-only retry changed typed lesson")
                raise ValueError("layout naming retry changed procedure, values, ancestry or supervision")
        views = {mode: curriculum.materialize_pair(pair, layout=mode, seed=seed, work=work.layout)
                 for mode in curriculum.LAYOUTS}
        work.counts["paired_view_candidate_pairs"] += 1
        current_placement = _placement(views)
        if placement is None:
            placement = current_placement
        elif current_placement != placement:
            record.update(status="failed", failed_location=list(location), failure="names-only retry changed query placement")
            raise ValueError("names-only admission changed layout policy, permutation or causal query slots")
        staged, hashes, conflicts, coincidences = _stage_views(views, location, history, owners, work)
        if fresh_relation is not None:
            anonymous = evaluation._anonymous(pair[0])
            if (anonymous[1] == fresh_relation["original_naming_map_sha256"]
                    or anonymous[2] == fresh_relation["original_visible_names_sha256"]):
                conflicts.append(dict(location=list(location), inventory="fresh_anchor_relation",
                                      reason="names must remain different from the exact training anchor"))
        if conflicts:
            work.counts["collision_rejections"] += len(conflicts)
            work.counts["paired_view_rejected_candidates"] += 1
            rejection = dict(location=list(location), naming_attempt=attempt,
                canonical_identity=admission._identity(pair), conflicts=conflicts)
            record["rejections"].append(rejection)
            work.event(dict(event="paired_view_candidate_rejected", **rejection))
            if initial_meaning is None:
                initial_meaning = admission._meaning(initial)
            continue
        # All four members have passed. Only this boundary mutates ownership.
        owners.update(staged)
        work.counts["accepted_view_rows"] += 4
        work.counts["same_member_cross_view_coincidences"] += coincidences
        record["accepted_pairs"] += 1
        accepted = admission._identity(pair)
        record["accepted_stream_sha256"] = _hash([record["accepted_stream_sha256"], location, accepted, hashes])
        if attempt != start_attempt:
            work.counts["paired_view_rebound_pairs"] += 1
            record["overrides"].append(dict(location=list(location), previous_naming_attempt=start_attempt,
                naming_attempt=attempt, initial=initial_identity, accepted=accepted,
                typed_lesson_sha256=initial_meaning, unchanged_placement_sha256=_hash(placement),
                typed_program_values_ancestry_targets_replies_unchanged=True))
        return pair, views, hashes, attempt
    record.update(status="failed", failed_location=list(location), failure="bounded names-only admission exhausted")
    raise ValueError("paired layout admission exhausted the fixed naming namespace")


def _training_layouts(plan, history, work):
    plan = deepcopy(plan)
    original_plan_sha256 = evidence.json_digest(plan)
    planning.validate_plan(plan)
    owners, transcripts = {}, {mode: set() for mode in curriculum.LAYOUTS}
    expected = {mode: {} for mode in curriculum.LAYOUTS}
    accumulators = {mode: training._initial_evidence() for mode in curriculum.LAYOUTS}
    distributions = {mode: {} for mode in curriculum.LAYOUTS}
    for bundle_id in plan["schedules"]["curriculum"]:
        work.boundary()
        families = planning._materialize_validated_bundle(plan, bundle_id)
        adapted = {mode: {family: [] for family in foundation.FAMILIES} for mode in curriculum.LAYOUTS}
        for family in foundation.FAMILIES:
            rows = families[family]
            for offset in range(0, len(rows), 2):
                index = offset//2
                key = f"{bundle_id}/{family}/{index}"
                initial_attempt = plan["admission"]["realization_attempts"].get(key, 0)
                _, views, hashes, accepted_attempt = _admit_views(rows[offset:offset+2],
                    ("train", bundle_id, family, index), layout_seed("train", bundle_id=bundle_id, pair_index=index),
                    history, owners, work,
                    lambda attempt: planning._materialize_validated_pair(plan, bundle_id, family, index, attempt),
                    start_attempt=initial_attempt)
                if accepted_attempt:
                    plan["admission"]["realization_attempts"][key] = accepted_attempt
                for mode in curriculum.LAYOUTS:
                    adapted[mode][family].extend(views[mode])
                    transcripts[mode].update(hashes[mode])
        for mode in curriculum.LAYOUTS:
            bundle = _bundle(adapted[mode], bundle_id, mode, work)
            record = deepcopy(work.last_bundle_evidence)
            expected[mode][str(bundle_id)] = record
            accumulators[mode] = training._accumulate(accumulators[mode], record)
            for family, rows in bundle["families"].items():
                cell = f"{family}/d{record['depth']}/t{record['turns']}"
                stats = distributions[mode].setdefault(cell, _empty_distribution())
                for offset in range(0, len(rows), 2):
                    pair = rows[offset:offset+2]
                    _add_distribution(stats, pair)
        left, right = (expected[mode][str(bundle_id)] for mode in curriculum.LAYOUTS)
        if (left["common_parents_sha256"] != right["common_parents_sha256"]
                or any(left["families"][family]["exposures"] != right["families"][family]["exposures"] for family in foundation.FAMILIES)):
            raise ValueError("paired layouts changed canonical parents or exposure totals")
        if (bundle_id+1) % 48 == 0:
            work.event(dict(event="both_layouts_admitted", bundles=bundle_id+1))
    original, varied = (accumulators[mode] for mode in curriculum.LAYOUTS)
    if (original["consumed_common_parent_identity_sha256"] != varied["consumed_common_parent_identity_sha256"]
            or original["exposures"] != varied["exposures"]):
        raise ValueError("matched common-parent chain or exposure totals differ")
    if any(len(values) != 82944 for values in transcripts.values()):
        raise ValueError("training views require exactly82944 unique episodes each")
    planning.validate_plan(plan)
    _admission_record(work, "train").update(status="admitted", initial_plan_sha256=original_plan_sha256,
        admitted_plan_sha256=evidence.json_digest(plan))
    return plan, dict(schema=EXPECTED_SCHEMA, plan_sha256=evidence.json_digest(plan), order="curriculum",
        bundles=expected, final_arm_evidence=accumulators, distributions=distributions), transcripts, owners


def _evaluation_rebind(pair, role_seed, role, name, index, attempt):
    recipe = pair[0]["recipe"]
    return foundation.generate_pair(pair[0]["family"], recipe["seed"], depth=recipe["depth"],
        turns=recipe["turns"], split=role, structure_split=recipe["structure_split"], value_seed=recipe["value_seed"],
        naming_seed=evaluation._seed(role_seed, role, name.split("/", 1)[1], index, "layout-admission-names", attempt))


def _fresh_relation_after_names(relation, pair):
    result = deepcopy(relation)
    values = [evaluation._anonymous(row) for row in pair]
    if (values[0][1] == relation["original_naming_map_sha256"]
            or values[0][2] == relation["original_visible_names_sha256"]
            or [value[0] for value in values] != relation["anonymous_values_sha256"]):
        raise ValueError("paired-view name repair changed fresh realization relation")
    result.update(pair_sha256=evidence.json_digest(pair), naming_map_sha256=values[0][1],
        visible_names_sha256=values[0][2], recipe=deepcopy(pair[0]["recipe"]))
    return result


def _adapt_banks(base_banks, role, history, owners, work, *, fit_references=None, canonical_manifest=None):
    banks = {mode: {} for mode in curriculum.LAYOUTS}
    reports = {mode: {} for mode in curriculum.LAYOUTS}
    role_hashes, seen_parents, accepted_canonical, fresh_relations = set(), set(), {}, {}
    if role != "train_fit" and canonical_manifest is None:
        raise ValueError("original canonical evaluation admission required")
    for name, rows in sorted(base_banks.items()):
        accepted_canonical[name] = []
        if name.startswith("fresh/"):
            fresh_relations[name] = []
        for mode in curriculum.LAYOUTS:
            banks[mode][name] = []
            reports[mode][name] = _empty_distribution()
        for offset in range(0, len(rows), 2):
            work.boundary()
            pair = rows[offset:offset+2]
            parent_hash = _hash(pair)
            if parent_hash in seen_parents:
                raise ValueError("canonical evaluation parent repeated in a different pair/cell")
            seen_parents.add(parent_hash)
            if role == "train_fit":
                reference = fit_references[name][offset//2]
                seed = layout_seed("train", bundle_id=reference["bundle_id"], pair_index=reference["pair_index"])
                location = ("train", reference["bundle_id"], reference["family"], reference["pair_index"])
            else:
                seed = layout_seed(role, pair_index=offset//2, bank_name=name)
                location = (role, name, offset//2)
                relation = (canonical_manifest["fresh_anchor_relations"][name][offset//2]
                            if name.startswith("fresh/") else None)
                pair, views, view_hashes, _ = _admit_views(pair, location, seed, history, owners, work,
                    lambda attempt: _evaluation_rebind(rows[offset:offset+2], EVALUATIONS[role], role, name, offset//2, attempt),
                    fresh_relation=relation)
                if relation is not None:
                    fresh_relations[name].append(_fresh_relation_after_names(relation, pair))
            accepted_canonical[name].extend(pair)
            for mode in curriculum.LAYOUTS:
                adapted = (curriculum.materialize_pair(pair, layout=mode, seed=seed, work=work.layout)
                           if role == "train_fit" else views[mode])
                curriculum.validate_pair(adapted, work=work.layout)
                if role == "train_fit":
                    hashes = []
                    for row in adapted:
                        digest = transcript_digest(row)
                        owner = (*location, row["recipe"]["base_pair_sha256"], row["variant"])
                        if digest in history or owners.get(digest) != owner:
                            raise ValueError("fit counterpart was not this exact admitted training lesson")
                        work.counts["fit_rows_verified"] += 1
                        hashes.append(digest)
                else:
                    hashes = view_hashes[mode]
                role_hashes.update(hashes)
                banks[mode][name].extend(adapted)
                _add_distribution(reports[mode][name], adapted)
    for mode in curriculum.LAYOUTS:
        for name, rows in banks[mode].items():
            reports[mode][name].update(rows_sha256=_hash(rows), query_turns=sum(len(row["queries"]) for row in rows),
                observation_utf8_bytes=sum(len(turn["text"].encode("utf-8")) for row in rows for turn in row["turns"]),
                reply_utf8_bytes=sum(len(turn["reply"].encode("utf-8")) for row in rows for turn in row["turns"]))
    if role != "train_fit":
        _admission_record(work, role).update(status="admitted")
    canonical_report = None if role == "train_fit" else dict(schema=PAIRED_ADMISSION_SCHEMA, role=role,
        initial_canonical_banks_sha256=_hash(base_banks), initial_canonical_manifest_sha256=_hash(canonical_manifest),
        banks={name: evaluation._stats(rows) for name, rows in accepted_canonical.items()},
        fresh_anchor_relations=fresh_relations,
        transcript_sha256=sorted({transcript_digest(row) for rows in accepted_canonical.values() for row in rows}),
        paired_view_admission=deepcopy(work.paired_admissions[role]),
        scope="Final canonical parents after transactional names-only admission of both layout views; initial frozen-builder report is preserved separately.")
    return banks, dict(schema=SCHEMA, role=role, views=reports,
        unique_transcripts=len(role_hashes), transcript_sha256=sorted(role_hashes)), role_hashes, accepted_canonical, canonical_report


def _fit(plan, manifest, work):
    banks, references = {}, {}
    for cell, candidates in sorted(manifest["anchors"].items()):
        if len(candidates) < PAIRS_PER_CELL:
            raise ValueError("eight canonical fit references required per cell")
        name = "fit/"+cell
        references[name] = deepcopy(candidates[:PAIRS_PER_CELL])
        banks[name] = []
        for reference in references[name]:
            work.boundary()
            bundle_id, family, index = (reference[key] for key in ("bundle_id", "family", "pair_index"))
            attempt = plan["admission"]["realization_attempts"].get(f"{bundle_id}/{family}/{index}", 0)
            pair = planning._materialize_validated_pair(plan, bundle_id, family, index, attempt)
            if fingerprint_rows(pair) != reference["pair_sha256"] or evidence._cell(pair[0]) != cell:
                raise ValueError("fit pair differs from its canonical admitted anchor")
            banks[name].extend(pair)
    return banks, references


def _failure_detail(error):
    detail = {}
    if isinstance(error, admission.AdmissionError):
        detail["admission_receipt"] = deepcopy(error.receipt)
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        if frame.f_code.co_name == "build_evaluation" and Path(frame.f_code.co_filename).resolve() == Path(evaluation.__file__).resolve():
            local = frame.f_locals
            detail["partial_evaluation"] = dict(role=local.get("role"),
                rejections=dict(local.get("rejected", {})), attempts_by_bank=deepcopy(local.get("attempts", {})),
                accepted_unique_transcripts=len(local.get("used", ())))
        trace = trace.tb_next
    return detail


def prepare(directory, *, phase_seconds, progress=None):
    """Exclusive preparation with declared nonpreemptive phase deadlines.

    Bounds apply before/after canonical calls and at phase/bundle/pair boundaries;
    a started file or canonical operation may finish after its deadline. No
    failed attempt is resumed or retried by this API. The caller guards neural
    entry points and records the enclosing process/import cost independently.
    """
    if (type(phase_seconds) is not dict or set(phase_seconds) != set(PHASES)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in phase_seconds.values())
            or phase_seconds != PHASE_SECONDS
            or progress is not None and not callable(progress)):
        raise ValueError("exact frozen positive phase allowances and optional callback required")
    started, cpu_started = time.monotonic(), time.process_time()
    output = native(directory)
    output.mkdir(parents=True, exist_ok=False)
    work, artifacts = WorkLedger(), set()
    journal = (output / "events.jsonl").open("xb")
    def event(value):
        journal.write((_json(dict(utc=datetime.now(timezone.utc).isoformat(), **value))+"\n").encode())
        journal.flush(); os.fsync(journal.fileno())
        if progress is not None:
            progress(value)
    work.progress = event
    sources, manifest = {}, None
    report = dict(schema=SCHEMA, status="running", phase_seconds=deepcopy(phase_seconds),
        started_utc=datetime.now(timezone.utc).isoformat(), neural_forwards=0, backward_calls=0, optimizer_updates=0,
        cuda_calls=0, automatic_retry=False, automatic_promotion=False)
    def save(name, value):
        work.boundary()
        _write(output / name, value)
        artifacts.add(name)
    try:
        sources = source_hashes()
        save("contract.json", contract())
        save("invocation.json", dict(schema=SCHEMA, source_sha256=sources, phase_seconds=phase_seconds,
            started_utc=report["started_utc"], no_full_training_rows=True, no_trusted_serialized_index=True))
        cache_before = dict(pair=foundation._generate.cache_info()._asdict(), procedure=foundation._procedure.cache_info()._asdict())
        with _tracking(work):
            with _phase(work, "history", phase_seconds["history"]):
                history_values, historical = _history(work)
                prior_pins, report["prior_preparation_cost"] = _prior_preparation_cost(work)
                historical["input_file_sha256"].update(prior_pins)
                history = set(history_values)
                save("history.pt", history_values); save("history-receipt.json", historical)
            with _phase(work, "admission", phase_seconds["admission"]):
                plan, admitted = admission.repair_plan(planning.build_plan(**PLAN_OPTIONS), history_values)
                save("initial-plan.json", plan); save("admission.json", admitted)
            with _phase(work, "layouts", phase_seconds["layouts"]):
                plan, expected, training_transcripts, owners = _training_layouts(plan, history, work)
                expected["source_sha256"] = sources
                save("plan.json", plan)
                save("paired-training-admission.json", work.paired_admissions["train"])
                save("expected-bundles.json", expected)
                save("training-transcripts.pt", {mode: sorted(values) for mode, values in training_transcripts.items()})
            with _phase(work, "canonical_evidence", phase_seconds["canonical_evidence"]):
                training_manifest, canonical_transcripts = evidence.prepare_training(plan,
                    protected_transcripts=history_values, anchor_limit=ANCHOR_LIMIT)
                if set(canonical_transcripts) != training_transcripts["original"]:
                    raise ValueError("reconstructed final canonical transcripts differ from jointly admitted original view")
                if set(training_manifest["anchors"]) != set(evaluation.expected_fresh_cells()):
                    raise ValueError("missing canonical training/fit cells")
                save("training-manifest.json", training_manifest)
                save("canonical-training-transcripts.pt", canonical_transcripts)
            all_training = set().union(*training_transcripts.values())
            excluded = history | all_training
            banks, base_banks, bank_reports, canonical_reports = {}, {}, {}, {}
            with _phase(work, "evaluation", phase_seconds["evaluation"]):
                for role, seed in EVALUATIONS.items():
                    work.event(dict(event="evaluation_role_start", role=role))
                    initial_banks, initial_report = evaluation.build_evaluation(plan, training_manifest,
                        training_transcripts=canonical_transcripts, role=role, seed=seed,
                        pairs_per_cell=PAIRS_PER_CELL, excluded_transcripts=sorted(excluded))
                    save(role+"-initial-canonical.pt", initial_banks)
                    save(role+"-initial-canonical-admission.json", initial_report)
                    banks[role], bank_reports[role], role_hashes, base_banks[role], canonical_reports[role] = _adapt_banks(
                        initial_banks, role, history, owners, work, canonical_manifest=initial_report)
                    if role_hashes & excluded:
                        raise ValueError("evaluation views overlap history/training/prior role")
                    excluded.update(role_hashes)
                    save(role+"-canonical.pt", base_banks[role]); save(role+"-banks.pt", banks[role])
                    save(role+"-canonical-admission.json", canonical_reports[role]); save(role+"-layout-admission.json", bank_reports[role])
                work.event(dict(event="evaluation_role_start", role="train_fit"))
                base_banks["train_fit"], references = _fit(plan, training_manifest, work)
                banks["train_fit"], bank_reports["train_fit"], fit_hashes, final_fit, _ = _adapt_banks(base_banks["train_fit"],
                    "train_fit", history, owners, work, fit_references=references)
                if final_fit != base_banks["train_fit"]:
                    raise ValueError("fit admission changed an exact final training counterpart")
                if not fit_hashes <= all_training:
                    raise ValueError("fit views are not exact observed training counterparts")
                save("train_fit-canonical.pt", base_banks["train_fit"]); save("train_fit-banks.pt", banks["train_fit"])
                save("train_fit-layout-admission.json", dict(bank_reports["train_fit"], references=references))
            with _phase(work, "publish", phase_seconds["publish"]):
                evaluation_hashes = excluded-history-all_training
                protection = sorted(history | evaluation_hashes)
                if set(protection) & all_training:
                    raise ValueError("final protection overlaps either training view")
                save("protected.pt", protection)
                snapshots = {}
                for index, (name, digest) in enumerate(sources.items()):
                    work.boundary()
                    content = native(ROOT / name).read_bytes()
                    if hashlib.sha256(content).hexdigest() != digest:
                        raise ValueError("source changed before snapshot: "+name)
                    short = f"source/{index:03d}.bin"
                    destination = output / short
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as handle:
                        handle.write(content)
                    if file_hash(destination) != digest:
                        raise ValueError("source snapshot bytes differ: "+name)
                    snapshots[name] = short
                    artifacts.add(short)
                _verify_files(historical["input_file_sha256"], work)
                if source_hashes() != sources:
                    raise ValueError("sources changed during data preparation")
                save("admission-summary.json", dict(schema=SCHEMA,
                    history_count=len(history), training_unique_by_layout={mode: len(v) for mode, v in training_transcripts.items()},
                    training_union_count=len(all_training), evaluation_union_count=len(evaluation_hashes),
                    protected_count=len(protection), protected_sha256=evidence.json_digest(protection),
                    canonical_admission_rejections=admitted["counts"]["rejected_candidate_pairs"],
                    paired_view_admissions=deepcopy(work.paired_admissions),
                    evaluation_rejections={role: len(record["paired_view_admission"]["rejections"])
                                           for role, record in canonical_reports.items()},
                    banks={role: {mode: {name: len(rows) for name, rows in view.items()} for mode, view in data.items()}
                           for role, data in banks.items()}, source_snapshots=snapshots))
                artifact_pins = {}
                for name in sorted(artifacts):
                    work.boundary()
                    artifact_pins[name] = file_hash(output / name)
                manifest = dict(schema=SCHEMA, contract=contract(), source_sha256=sources, source_snapshots=snapshots,
                    input_file_sha256=historical["input_file_sha256"], artifacts_sha256=artifact_pins,
                    plan_sha256=evidence.json_digest(plan), expected_bundles_sha256=_hash(expected),
                    canonical_training_manifest_sha256=evidence.json_digest(training_manifest),
                    no_neural_training_or_inference=True, automatic_promotion=False, automatic_retry=False)
                work.boundary()
                _write(output / "manifest.json", manifest)
                report["manifest_sha256"] = file_hash(output / "manifest.json")
        report.update(canonical_cache_before=cache_before,
            canonical_cache_after=dict(pair=foundation._generate.cache_info()._asdict(), procedure=foundation._procedure.cache_info()._asdict()))
        journal.close()
        report.update(status="completed")
    except BaseException as error:
        report.update(status="failed", error=repr(error), failure_detail=_failure_detail(error))
        raise
    finally:
        if not journal.closed:
            journal.close()
        report.update(ended_utc=datetime.now(timezone.utc).isoformat(), work=work.report(),
            wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started,
            source_sha256=sources, partial_artifacts=sorted(artifacts), journal_sha256=file_hash(output / "events.jsonl"),
            cost_scope="Entire prepare call through receipt assembly, including source/input authentication, all admission/reconstruction and artifact I/O; caller additionally records imports and final receipt serialization.")
        _write(output / "preparation.json", report)
    return manifest


def _relative(directory, name):
    if type(name) is not str or "\\" in name or Path(name).is_absolute():
        raise ValueError("canonical relative artifact path required")
    result = (Path(directory) / name).resolve()
    if not result.is_relative_to(Path(directory).resolve()) or result == Path(directory).resolve():
        raise ValueError("artifact escapes its data root")
    if result.relative_to(Path(directory).resolve()).as_posix() != name:
        raise ValueError("noncanonical artifact path")
    return native(result)


def _sealed_header(directory, expected_manifest_sha256):
    output = native(directory)
    if file_hash(output / "manifest.json") != expected_manifest_sha256:
        raise ValueError("explicit layout data manifest identity differs")
    manifest, receipt = _read(output / "manifest.json"), _read(output / "preparation.json")
    if (manifest.get("schema") != SCHEMA or manifest.get("contract") != contract()
            or manifest.get("no_neural_training_or_inference") is not True
            or receipt.get("status") != "completed" or receipt.get("manifest_sha256") != expected_manifest_sha256
            or receipt.get("source_sha256") != manifest.get("source_sha256")
            or source_hashes() != manifest["source_sha256"]
            or file_hash(output / "events.jsonl") != receipt.get("journal_sha256")):
        raise ValueError("completed source-pinned layout preparation required")
    return output, manifest, receipt


def _bank_files(output, role, manifest):
    if type(role) is not str or role not in ("dev", "audit", "train_fit"):
        raise ValueError("explicit dev/audit/train_fit bank role required")
    names = [role+suffix for suffix in ("-banks.pt", "-canonical.pt", "-layout-admission.json")]
    for name in names:
        if file_hash(output / name) != manifest["artifacts_sha256"][name]:
            raise ValueError("sealed role artifact changed: "+name)
    with (output / names[0]).open("rb") as handle:
        banks = torch.load(handle, map_location="cpu", weights_only=True)
    with (output / names[1]).open("rb") as handle:
        canonical = torch.load(handle, map_location="cpu", weights_only=True)
    return dict(banks=banks, canonical_banks=canonical, bank_report=_read(output / names[2]))


def load_banks(directory, role, *, expected_manifest_sha256):
    """Explicit role load, allowing audit to remain unopened until endpoints."""
    output, manifest, _ = _sealed_header(directory, expected_manifest_sha256)
    return _bank_files(output, role, manifest)


def load(directory, *, expected_manifest_sha256, roles=("dev", "train_fit")):
    """Authenticate sealed bytes only; no regeneration or serialized trust index.

    The explicit expected digest comes from the independent prospective launch.
    Preparation's full admission receipt must also be bound by that launch.
    """
    if type(roles) not in (list, tuple) or len(set(roles)) != len(roles) or any(role not in ("dev", "audit", "train_fit") for role in roles):
        raise ValueError("distinct explicit bank roles required")
    output, manifest, receipt = _sealed_header(directory, expected_manifest_sha256)
    _verify_files(manifest["input_file_sha256"])
    for name, digest in manifest["artifacts_sha256"].items():
        if file_hash(_relative(output, name)) != digest:
            raise ValueError("sealed data artifact changed: "+name)
    plan, expected = _read(output / "plan.json"), _read(output / "expected-bundles.json")
    planning.validate_plan(plan)
    if (plan["config"] != PLAN_OPTIONS or "admission" not in plan
            or evidence.json_digest(plan) != manifest["plan_sha256"]
            or _hash(expected) != manifest["expected_bundles_sha256"]
            or expected["source_sha256"] != manifest["source_sha256"]):
        raise ValueError("compact plan or expected bundle evidence differs")
    banks, base_banks, reports = {}, {}, {}
    for role in roles:
        record = _bank_files(output, role, manifest)
        banks[role], base_banks[role], reports[role] = record["banks"], record["canonical_banks"], record["bank_report"]
    return dict(manifest=manifest, preparation=receipt, plan=plan, expected_bundles=expected,
        banks=banks, canonical_banks=base_banks, training_manifest=_read(output / "training-manifest.json"),
        bank_reports=reports)
