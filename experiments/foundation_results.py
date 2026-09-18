"""All-endpoint verification and one-shot scoring of the foundation-order pilot.

Verification regenerates canonical lessons and restores every official checkpoint
on CPU; it does not reproduce gradient arithmetic or score models. Evaluation
requires that sealed verification and never changes a learner or selects a model.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments import foundation_study as study
from experiments.composition_evaluation import METRICS, evaluate_banks
from experiments.foundation_evidence import json_digest, prepare_training, reconstruct_anchor, transcript_set
from experiments.foundation_training import COUNTS, FAMILIES, replay_evidence
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.train_cognitive import atomic_json


SCHEMA = "bic-foundation-results-v1"
PHYSICAL = ("physical_optimizer_updates", "drawn_episode_exposures",
            "neural_attempted_episode_exposures", "completed_microbatch_episode_exposures")
LOSSES = ("loss", "action_loss", "reply_loss", "observation_language_loss")


def _same(left, right, message="foundation evidence differs"):
    if json_digest(left) != json_digest(right):
        raise ValueError(message)


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be finite and nonnegative")
    return value


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def _proof(protocol):
    from experiments.foundation_runtime_probe import load_proof
    proof = load_proof(protocol["proof_directory"], SequenceConfig(**protocol["config"]))
    study.require_admitted_proof(proof)
    if json_digest(proof) != protocol["proof_sha256"]:
        raise ValueError("foundation runtime proof changed")
    _same(proof["execution_profile"], protocol["execution_profile"], "proof runtime differs")
    if proof["learning_rate"] != protocol["learning_rate"]:
        raise ValueError("proof learning rate differs from pilot")
    return proof


def _input_hashes(directory, protocol):
    """Require both complete arm receipts before any model is constructed/scored."""
    directory = Path(directory)
    names = {"protocol.json", "preparation.json", *protocol["files_sha256"]}
    receipts = {}
    expected = {f"checkpoint-{step:06d}.pt" for step in study.STEPS}
    for arm in study.ARMS:
        path = directory / arm / "receipt.json"
        if not path.is_file():
            raise ValueError("both completed foundation arm receipts are required")
        receipt = study.read(path)
        if (receipt.get("schema") != study.SCHEMA or receipt.get("arm") != arm
                or receipt.get("status") != "completed" or receipt.get("automatic_promotion") is not False
                or receipt.get("physical_work_unknown") is not False
                or type(receipt.get("retained_updates")) is not int or receipt["retained_updates"] != study.TOTAL
                or set(receipt.get("checkpoints", {})) != expected):
            raise ValueError("both exact complete endpoints and checkpoint sets are required")
        receipts[arm] = receipt
        names.update({f"{arm}/receipt.json", f"{arm}/steps.jsonl", *(f"{arm}/{name}" for name in expected)})
    if any(not (directory / name).is_file() for name in names):
        raise ValueError("both complete checkpoint and journal collections are required")
    hashes = {name: study.file_hash(directory / name) for name in sorted(names)}
    for arm, receipt in receipts.items():
        if (receipt["protocol_sha256"] != hashes["protocol.json"]
                or receipt["journal_sha256"] != hashes[f"{arm}/steps.jsonl"]):
            raise ValueError("completed receipt belongs to different protocol or journal")
        _same(receipt["execution_profile"], protocol["execution_profile"], "receipt runtime differs")
        for name, digest in receipt["checkpoints"].items():
            if hashes[f"{arm}/{name}"] != digest:
                raise ValueError("official checkpoint file changed")
    preparation = study.read(directory / "preparation.json")
    if (preparation.get("status") != "prepared" or preparation.get("protocol_sha256") != hashes["protocol.json"]
            or preparation.get("neural_training_or_inference") is not False):
        raise ValueError("preparation receipt differs")
    _number(preparation["wall_seconds"], "preparation wall")
    return hashes, receipts


def _unchanged(directory, inputs):
    if any(study.file_hash(Path(directory) / name) != digest for name, digest in inputs.items()):
        raise ValueError("foundation inputs changed during verification or scoring")
    protocol = study.load_protocol(directory)
    _proof(protocol)
    return protocol


def _load_rows(directory, protocol):
    """Reconstruct the frozen training/admission evidence without neural work."""
    from experiments.foundation_admission import authenticate_admission
    from experiments.foundation_banks import build_evaluation, transcript_digest
    directory = Path(directory)
    plan = study.read(directory / "plan.json")
    manifest = study.read(directory / "training-manifest.json")
    admissions = study.read(directory / "bank-admission.json")
    banks = torch.load(directory / "banks.pt", map_location="cpu", weights_only=True)
    protected = transcript_set(torch.load(directory / "protected.pt", map_location="cpu", weights_only=True))
    training = torch.load(directory / "training-transcripts.pt", map_location="cpu", weights_only=True)
    if set(banks) != {"dev", "audit", "train_fit"} or set(admissions) != {"dev", "audit"}:
        raise ValueError("foundation bank roles differ")
    evaluated = {transcript_digest(row) for role in ("dev", "audit")
                 for rows in banks[role].values() for row in rows}
    if not evaluated <= protected:
        raise ValueError("evaluation observations absent from protected index")
    history = sorted(protected - evaluated)
    _same(history, study.historical_transcripts(protocol["capacity_choice"]),
          "historical exclusions differ from the authenticated capacity union")
    authenticate_admission(study.planning.build_plan(**study.PLAN_OPTIONS), plan,
        protected_transcripts=history, receipt=study.read(directory / "lesson-admission.json"))
    if (len(history) != protocol["history_count"] or len(protected) != protocol["protected_count"]
            or len(training) != protocol["training_count"]):
        raise ValueError("historical/training/protected counts differ")
    regenerated, transcripts = prepare_training(plan, protected_transcripts=history,
                                                anchor_limit=manifest["anchor_limit"])
    _same(manifest, regenerated, "training manifest differs from canonical materialization")
    _same(training, transcripts, "training transcript index differs")
    exclusions = set(history)
    for role in ("dev", "audit"):
        admission = admissions[role]
        expected_rows, expected_admission = build_evaluation(plan, manifest, training_transcripts=training,
            role=role, seed=admission["seed"], pairs_per_cell=study.PAIRS, excluded_transcripts=sorted(exclusions))
        _same(banks[role], expected_rows, "evaluation bank differs from prospective generation")
        _same(admission, expected_admission, "evaluation admission receipt differs")
        exclusions.update(admission["transcript_sha256"])
    _same(sorted(exclusions), sorted(protected), "protected observations differ")
    expected_fit = {"fit/" + cell: [row for ref in refs for row in reconstruct_anchor(plan, ref)]
                    for cell, refs in manifest["anchors"].items()}
    _same(banks["train_fit"], expected_fit, "fit anchors differ from admitted training pairs")
    return plan, manifest, banks, sorted(protected)


def validate_journal(events, plan, arm, manifest, replayed):
    """Bind each successful update to its actual canonical bundle and exposure."""
    schedule = plan["schedules"][arm]
    if type(events) is not list or len(events) != 2 * len(schedule) or len(replayed) != len(schedule):
        raise ValueError("journal must contain exactly one start/completion pair per prescribed update")
    micro = plan["config"]["micro_batch_size"]
    physical = dict.fromkeys(PHYSICAL, 0)
    step_seconds = materialization_seconds = wall_seconds = 0.
    for cursor, bundle_id in enumerate(schedule):
        started, completed = events[2 * cursor:2 * cursor + 2]
        if (set(started) != {"event", "cursor", "utc"} or started["event"] != "started"
                or type(started["cursor"]) is not int or started["cursor"] != cursor
                or type(started["utc"]) is not str or not started["utc"]
                or set(completed) != {"event", "report"} or completed["event"] != "completed"):
            raise ValueError("journal sequence, identity or completion differs")
        report, expected = completed["report"], replayed[cursor]
        if (report.get("failed", False) is not False or report.get("cursor") != cursor + 1
                or type(report.get("cursor")) is not int or report.get("bundle_id") != bundle_id
                or type(report.get("bundle_id")) is not int or expected["bundle_id"] != bundle_id
                or len(report.get("microbatches", [])) != len(FAMILIES)):
            raise ValueError("journal report cursor or bundle differs")
        wanted = {"physical_optimizer_updates": 1, "retained_optimizer_updates": 1,
            "drawn_microbatches": 3, "neural_attempted_microbatches": 3, "completed_microbatches": 3,
            "drawn_episode_exposures": 3 * micro, "neural_attempted_episode_exposures": 3 * micro,
            "completed_microbatch_episode_exposures": 3 * micro}
        for name, value in wanted.items():
            if _integer(report.get(name), name) != value:
                raise ValueError("journal physical completion or exposure differs")
        for family, measured in zip(FAMILIES, report["microbatches"]):
            if (measured["family"] != family or measured["depth"] != expected["depth"]
                    or measured["turns"] != expected["turns"]):
                raise ValueError("journal microbatch order/depth/length differs")
            wanted_row = expected["families"][family]
            for name in ("rows_sha256", "recipes_sha256", "exposures"):
                _same(measured[name], wanted_row[name], "journal rows/recipe/exposure differs from canonical replay")
            declared = manifest["bundles"][str(bundle_id)]
            if measured["rows_sha256"] != declared["family_sha256"][family]:
                raise ValueError("journal bundle differs from admitted manifest")
            _same(measured["exposures"], {key: declared["family_counts"][family][key] for key in COUNTS})
            for name in LOSSES:
                _number(measured[name], name)
        for name in LOSSES:
            value = _number(report[name], name)
            mean = sum(row[name] for row in report["microbatches"]) / len(FAMILIES)
            if not math.isclose(value, mean, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("journal objective mean differs")
        step = _number(report["step_seconds"], "step seconds")
        materialized = _number(report["materialization_seconds"], "materialization seconds")
        wall = _number(report["wall_seconds"], "step invocation seconds")
        if materialized > step or step > wall:
            raise ValueError("nested step timing differs")
        step_seconds += step; materialization_seconds += materialized; wall_seconds += wall
        for key in PHYSICAL:
            physical[key] += report[key]
    return {**physical, "retained_step_seconds": step_seconds,
            "materialization_seconds_included_in_step": materialization_seconds,
            "step_invocation_seconds": wall_seconds}


def _validate_receipt(receipt, journal, updates, micro):
    for key in PHYSICAL:
        expected = updates if key == "physical_optimizer_updates" else updates * 3 * micro
        if _integer(receipt[key], key) != expected or journal[key] != expected:
            raise ValueError("receipt physical counts differ from complete journal")
    for name in ("wall_seconds", "setup_seconds", "peak_cuda_allocated_mib", "peak_cuda_reserved_mib"):
        _number(receipt[name], name)
    if receipt["setup_seconds"] > receipt["wall_seconds"]:
        raise ValueError("setup must be included in invocation wall")
    if journal["step_invocation_seconds"] > receipt["wall_seconds"] + 1e-6:
        raise ValueError("step calls cannot exceed enclosing worker wall")


def _checkpoint(path, arm, protocol_hash, runtime):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (type(saved) is not dict or set(saved) != {"schema", "arm", "protocol_sha256", "learner", "weights_sha256", "execution_profile"}
            or saved["schema"] != study.SCHEMA or saved["arm"] != arm
            or saved["protocol_sha256"] != protocol_hash):
        raise ValueError("official checkpoint envelope differs")
    _same(saved["execution_profile"], runtime, "checkpoint runtime differs")
    return saved


def verify(directory):
    directory = Path(directory).resolve()
    started = time.monotonic()
    protocol = study.load_protocol(directory)
    inputs, receipts = _input_hashes(directory, protocol)
    _proof(protocol)
    if (directory / "verification.json").exists():
        raise FileExistsError("foundation verification already exists")
    with (directory / "verification-started.json").open("x", encoding="utf8") as stream:
        json.dump({"schema": SCHEMA, "started_utc": study.utc(), "input_file_sha256": inputs}, stream)
    plan, manifest, _, protected = _load_rows(directory, protocol)
    result = {"schema": SCHEMA, "status": "completed", "protocol_sha256": inputs["protocol.json"],
        "input_file_sha256": inputs, "source_sha256": protocol["source_sha256"],
        "proof_sha256": protocol["proof_sha256"], "neural_training_or_inference": False,
        "automatic_promotion": False, "arms": {}}
    for arm in study.ARMS:
        with (directory / arm / "steps.jsonl").open(encoding="utf8") as stream:
            events = [json.loads(line) for line in stream if line.strip()]
        replay = replay_evidence(plan, arm, study.TOTAL, protected, include_bundles=True)
        journal = validate_journal(events, plan, arm, manifest, replay["bundles"])
        _validate_receipt(receipts[arm], journal, study.TOTAL, plan["config"]["micro_batch_size"])
        learner = study.trainer(directory, arm, device="cpu")
        weights = {}
        for step in study.STEPS:
            saved = _checkpoint(directory / arm / f"checkpoint-{step:06d}.pt", arm,
                                inputs["protocol.json"], protocol["execution_profile"])
            learner.restore(saved["learner"])
            digest = checkpoint_digest(learner.model)
            if learner.cursor != step or saved["weights_sha256"] != digest:
                raise ValueError("checkpoint step or weight hash differs")
            if step == 0 and digest != protocol["initial_weights_sha256"]:
                raise ValueError("seeded initialization differs")
            timing = saved["learner"]["timing"]
            reports = [events[2 * index + 1]["report"] for index in range(step)]
            for name, field in (("retained_step_seconds", "step_seconds"),
                                ("materialization_seconds_included_in_step", "materialization_seconds")):
                if not math.isclose(timing[name], sum(row[field] for row in reports), rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError("checkpoint retained timing differs from journal prefix")
            weights[str(step)] = digest
        _same(learner.snapshot()["evidence"], replay["evidence"], "endpoint prefix differs")
        expected_exposures = {family: {key: manifest["totals"][family][key] for key in COUNTS} for family in FAMILIES}
        _same(replay["evidence"]["exposures"], expected_exposures, "endpoint lesson exposure differs from plan manifest")
        result["arms"][arm] = {"weights_sha256": weights, "endpoint_evidence": replay["evidence"],
            "journal": journal, "receipt": receipts[arm], "exact_official_checkpoint_restores": True}
        del learner
    a, b = (result["arms"][arm]["endpoint_evidence"] for arm in study.ARMS)
    for key in ("cursor", "exposures", "family_microbatches", "bucket_microbatches", "depth_microbatches"):
        _same(a[key], b[key], "the two teaching orders differ in final exposure")
    _unchanged(directory, inputs)
    result["wall_seconds"] = time.monotonic() - started
    result["scope"] = "Every official optimizer state restored on CPU and consumed input prefix regenerated; this verifies input/state consistency, not reproduced gradient arithmetic. No model scoring performed."
    atomic_json(directory / "verification.json", result)
    return result


def _verified(directory):
    protocol = study.load_protocol(directory)
    inputs, _ = _input_hashes(directory, protocol)
    value = study.read(Path(directory) / "verification.json")
    if (value.get("schema") != SCHEMA or value.get("status") != "completed"
            or value.get("neural_training_or_inference") is not False
            or value.get("automatic_promotion") is not False
            or value.get("protocol_sha256") != inputs["protocol.json"]
            or value.get("proof_sha256") != protocol["proof_sha256"] or set(value.get("arms", {})) != set(study.ARMS)):
        raise ValueError("complete sealed foundation verification required")
    _same(value["input_file_sha256"], inputs, "verified files changed")
    _same(value["source_sha256"], protocol["source_sha256"], "verified sources changed")
    for arm in study.ARMS:
        row = value["arms"][arm]
        if row.get("exact_official_checkpoint_restores") is not True or set(row["weights_sha256"]) != {str(step) for step in study.STEPS}:
            raise ValueError("verification is missing official checkpoints")
    _proof(protocol)
    return protocol, inputs, value


def _score(model, prepared, rows, config, role, *, control="normal"):
    from experiments.foundation_metrics import compact, validate_metrics
    started = time.monotonic()
    metrics = evaluate_banks(model, prepared, batch_size=32, score_replies=True, control=control)
    validate_metrics(metrics, rows, config, role, control=control)
    episodes = sum(len(bank) for bank in rows.values())
    turns = sum(len(row["turns"]) for bank in rows.values() for row in bank)
    return {"metrics": metrics, "summary": compact(metrics),
        "cost": {"wall_seconds": time.monotonic() - started,
            "bank_score_seconds_included_in_wall": sum(row["seconds"] for row in metrics["per_bank"].values()),
            "scored_episodes": episodes, "scored_turns": turns,
            "encoder_episode_instances": turns if control == "reset" else episodes,
            "free_reply_turns": turns, "control": control}}


def evaluate(directory):
    from experiments.execution_profile import assert_strict_profile, runtime_profile
    from experiments.foundation_evaluation import FoundationBank
    directory = Path(directory).resolve()
    started = time.monotonic()
    protocol, inputs, verified = _verified(directory)  # Must precede any scoring or CUDA work.
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError("foundation evaluation requires the declared one-thread profile")
    assert_strict_profile()
    runtime = runtime_profile("cuda:0")
    _same(runtime, protocol["execution_profile"], "evaluation execution profile differs")
    output = directory / "evaluation"
    output.mkdir(exist_ok=False)  # An interrupted scoring invocation remains visible; no retry.
    verification_hash = study.file_hash(directory / "verification.json")
    binding = {"schema": SCHEMA, "protocol_sha256": inputs["protocol.json"],
               "verification_sha256": verification_hash, "input_file_sha256": inputs,
               "execution_profile": runtime, "automatic_promotion": False}
    atomic_json(output / "started.json", {**binding, "started_utc": study.utc()})
    banks = torch.load(directory / "banks.pt", map_location="cpu", weights_only=True)
    config = SequenceConfig(**protocol["config"])
    tick = time.monotonic()
    prepared = {role: {name: FoundationBank(rows, role=role, config=config) for name, rows in named.items()}
                for role, named in banks.items()}
    preparation_seconds = time.monotonic() - tick
    model = build_sequence_student(study.MODEL_SEED, config=config, device="cuda:0")
    results = {}
    for arm in study.ARMS:
        tick = time.monotonic()
        receipt = {**binding, "arm": arm, "status": "running", "started_utc": study.utc(),
                   "development_curve": []}
        atomic_json(output / f"{arm}-started.json", receipt)
        try:
            for step in study.STEPS:
                assert_strict_profile()
                saved = _checkpoint(directory / arm / f"checkpoint-{step:06d}.pt", arm,
                                    inputs["protocol.json"], protocol["execution_profile"])
                model.load_state_dict(saved["learner"]["weights"], strict=True)
                digest = checkpoint_digest(model)
                if digest != verified["arms"][arm]["weights_sha256"][str(step)]:
                    raise ValueError("scored model differs from verified checkpoint")
                scored = _score(model, prepared["dev"], banks["dev"], config, "dev")
                receipt["development_curve"].append({"updates": step, "weights_sha256": digest, **scored})
            receipt["train_fit"] = _score(model, prepared["train_fit"], banks["train_fit"], config, "train_fit")
            receipt["audit"] = _score(model, prepared["audit"], banks["audit"], config, "audit")
            receipt["controls"] = {control: _score(model, prepared["audit"], banks["audit"], config, "audit", control=control)
                                   for control in ("blank", "reset")}
            pieces = [*receipt["development_curve"], receipt["train_fit"], receipt["audit"], *receipt["controls"].values()]
            receipt["cost"] = {key: sum(piece["cost"][key] for piece in pieces)
                               for key in ("scored_episodes", "scored_turns", "encoder_episode_instances", "free_reply_turns")}
            receipt["cost"]["score_wall_seconds_included_in_arm_wall"] = sum(piece["cost"]["wall_seconds"] for piece in pieces)
            receipt["status"] = "completed"
        except BaseException as error:
            receipt.update(status="failed", error=repr(error), failed_scoring_work_may_be_unrecorded=True)
            raise
        finally:
            receipt.update(wall_seconds=time.monotonic() - tick, finished_utc=study.utc())
            atomic_json(output / f"{arm}.json", receipt)
        results[arm] = receipt
        print(json.dumps({"evaluation_arm": arm, "status": "completed"}), flush=True)
    _unchanged(directory, inputs)
    if study.file_hash(directory / "verification.json") != verification_hash:
        raise ValueError("verification changed during scoring")
    assert_strict_profile()
    report = {**binding, "status": "completed", "results": results,
        "base_checkpoints_unchanged": True, "preparation_seconds": preparation_seconds,
        "wall_seconds": time.monotonic() - started,
        "arm_files_sha256": {arm: study.file_hash(output / f"{arm}.json") for arm in study.ARMS},
        "scope": "One initialization and two matched lesson orders; descriptive outcomes only. No automatic promotion. Development curves are scored only after both endpoints complete; audit and fitting scores are endpoint-only. Score/arm/preparation intervals are nested within total wall and must not be added twice; reset controls encode each original utterance separately."}
    atomic_json(output / "report.json", report)
    return report
