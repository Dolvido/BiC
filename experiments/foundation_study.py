"""One prospective matched foundation-order pilot, with no automatic promotion.

Both fresh learners consume identical immutable lessons and a common mixed tail.
Training performs no evaluation. Separate verification and scoring must wait for
both endpoints. Workers are deliberately one-shot: a failed invocation is kept,
never silently retried or counted as a successful prescribed run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments import foundation_plan as planning
from experiments.foundation_curriculum import FAMILIES, VERSION
from experiments.foundation_evidence import json_digest, prepare_training, reconstruct_anchor
from experiments.foundation_training import FoundationTrainer, source_hashes as training_sources
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.train_cognitive import atomic_checkpoint, atomic_json

SCHEMA = "bic-foundation-order-pilot-v2"
ARMS = ("curriculum", "mixed")
MODEL_SEED = 6101
PLAN_OPTIONS = dict(seed=601000001, stage_updates=192, final_updates=384,
                    micro_batch_size=32, rehearsal_every=4, ordering_seed=6100)
STEPS = (0, 192, 384, 576, 768, 960, 1152, 1536)
TOTAL = STEPS[-1]
PAIRS = 16


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def require_admitted_proof(proof):
    admission = proof.get("admission", {})
    count = admission.get("executed_prefix_overridden_pairs")
    if (admission.get("mode") != "naming_overrides" or type(count) is not int or count < 1):
        raise ValueError("foundation execution proof must exercise admitted naming overrides")


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("experiments/foundation_admission.py", "experiments/foundation_evidence.py", "experiments/foundation_banks.py",
        "experiments/foundation_evaluation.py", "experiments/foundation_metrics.py",
        "experiments/foundation_runtime_probe.py", "experiments/foundation_study.py",
        "experiments/foundation_results.py", "experiments/execution_profile.py",
        "experiments/capacity_banks.py", "experiments/capacity_evidence.py", "experiments/capacity_study.py")
    return {**training_sources(), **{name: file_hash(root/name) for name in names}}


def contract():
    return dict(schema=SCHEMA, version=VERSION, plan_options=PLAN_OPTIONS,
        arms=list(ARMS), model_seed=MODEL_SEED, checkpoints=list(STEPS), updates=TOTAL,
        pairs_per_evaluation_cell=PAIRS, evaluation_during_training=False,
        automatic_promotion=False, automatic_retry=False,
        lesson_admission="Canonical complete-pair order; first noncolliding deterministic naming attempt, bounded at1000. Preserve all procedures, values, targets, schedules and budgets. No historical exclusions removed; no score-dependent repair.",
        model_selection="Capacity endpoint development ranking only: worst nine-cell mean paired action/reply, overall paired mean, known accuracy, negative unsupported ASK; round12, smaller width on exact tie. Use that width's independently calibrated rate. No foundation feedback.",
        scope="One initialization. Matched lesson/byte/target exposure; not matched FLOPs or wall time. Order within foundation practice, not a comparison against composition-only practice. No general-intelligence or autonomy certificate.")


def capacity_choice(directory):
    """Read verified development endpoints only after the old study is closed."""
    from experiments import capacity_study as capacity
    from experiments.capacity_evidence import calibration_rank
    from experiments import summarize_capacity_study as summary_helper
    directory = Path(directory).resolve()
    # The completed summary is a gate, not a source of architecture scores.
    if not (directory/"audit/summary.json").is_file():
        raise ValueError("completed verified capacity summary required before foundation selection")
    old_protocol = capacity.load_protocol(directory)
    summary = read(directory/"audit/summary.json")
    verified = summary.get("verification", {})
    required = {path.relative_to(directory).as_posix() for path in summary_helper._complete_gate(directory)}
    hashes = verified.get("input_file_sha256", {})
    if (summary.get("schema") != summary_helper.SCHEMA or not required <= set(hashes)
            or verified.get("source_sha256") != old_protocol["source_sha256"]
            or verified.get("no_neural_inference_or_training_performed") is not True
            or verified.get("helper_sha256", {}).get("summary") != file_hash(summary_helper.__file__)):
        raise ValueError("capacity completed-summary provenance differs")
    for name, digest in hashes.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or file_hash(path) != digest:
            raise ValueError("capacity completed-summary input changed")
    for stage in ("calibration", "main"):
        capacity.load_verification(directory, stage)
    selection = capacity.load_selection(directory)
    candidates = {}
    for width in capacity.WIDTHS:
        saved = torch.load(directory/f"main/w{width}/checkpoint-007200.pt", map_location="cpu", weights_only=True)
        candidates[str(width)] = {"rank": list(calibration_rank(saved["development"])),
            "rate": selection["selected"][str(width)]["rate"]}
    width = max(capacity.WIDTHS, key=lambda value: (tuple(candidates[str(value)]["rank"]), -value))
    paths = [directory/name for name in hashes] + [directory/"audit/summary.json"]
    return {"directory": str(directory), "candidates": candidates, "width": width,
        "learning_rate": candidates[str(width)]["rate"], "config": asdict(capacity.config(width)),
        "files_sha256": {str(path): file_hash(path) for path in paths},
        "score_source": "verified final development checkpoints only; no audit scores used"}


def historical_transcripts(choice):
    from experiments import capacity_study as capacity
    from experiments.capacity_banks import _all_banks
    from experiments.realization_banks import transcript_digest
    from experiments.realization_training import stream_evidence
    directory = Path(choice["directory"])
    banks, protected = capacity.data(directory)
    values = set(protected["calibration"]) | set(protected["main"])
    for rows in _all_banks(banks):
        values.update(transcript_digest(row) for row in rows)
    for stage in ("calibration", "main"):
        # The completed summary verifies every job against one common replay.
        job = next(iter(capacity.jobs(stage)))
        saved = torch.load(directory/stage/job/"learner/backend.pt", map_location="cpu", weights_only=True)
        stream = saved["learner"]["realization"]
        verified = read(directory/"verification"/f"{stage}.json")
        normalized = json.loads(json.dumps(stream_evidence(stream), allow_nan=False))
        if normalized != verified["replayed_streams"][str(capacity.TOTALS[stage])]:
            raise ValueError("capacity cumulative exclusion stream differs from verified replay")
        values.update(stream["seen_transcripts"])
    for path, digest in choice["files_sha256"].items():
        if file_hash(path) != digest:
            raise ValueError("capacity inputs changed while collecting exclusions")
    return sorted(values)


def prepare(directory, capacity_directory, proof_directory):
    from experiments.foundation_admission import repair_plan
    from experiments.foundation_banks import build_evaluation
    from experiments.foundation_evaluation import FoundationBank
    from experiments.foundation_runtime_probe import load_proof
    started = time.monotonic()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    atomic_json(directory/"preparation-started.json", {"started_utc": utc(), "contract": contract(), "sources": sources})
    choice = capacity_choice(capacity_directory)
    config = SequenceConfig(**choice["config"])
    proof = load_proof(proof_directory, config)
    require_admitted_proof(proof)
    if proof["learning_rate"] != choice["learning_rate"]:
        raise ValueError("selected learning rate needs matching foundation execution proof")
    history = historical_transcripts(choice)
    plan, lesson_admission = repair_plan(planning.build_plan(**PLAN_OPTIONS), protected_transcripts=history)
    manifest, training = prepare_training(plan, protected_transcripts=history, anchor_limit=PAIRS)
    banks, admissions, exclusions = {}, {}, set(history)
    for role, seed in (("dev", 602000001), ("audit", 603000001)):
        banks[role], admissions[role] = build_evaluation(plan, manifest, training_transcripts=training,
            role=role, seed=seed, pairs_per_cell=PAIRS, excluded_transcripts=sorted(exclusions))
        exclusions.update(admissions[role]["transcript_sha256"])
    # Exact admitted training anchors are fit evidence only at the completed endpoint.
    banks["train_fit"] = {"fit/"+cell: [row for ref in refs for row in reconstruct_anchor(plan, ref)]
                           for cell, refs in manifest["anchors"].items()}
    for role, named in banks.items():
        for rows in named.values():
            FoundationBank(rows, role=role, config=config)  # codec/context admission, no inference
    if set(training) & exclusions:
        raise ValueError("foundation training overlaps reserved evaluation/history")
    atomic_json(directory/"plan.json", plan)
    atomic_json(directory/"lesson-admission.json", lesson_admission)
    atomic_json(directory/"training-manifest.json", manifest)
    atomic_json(directory/"bank-admission.json", admissions)
    atomic_json(directory/"capacity-choice.json", choice)
    atomic_checkpoint(directory/"banks.pt", banks)
    atomic_checkpoint(directory/"protected.pt", sorted(exclusions))
    atomic_checkpoint(directory/"training-transcripts.pt", training)
    if source_hashes() != sources:
        raise ValueError("sources changed during foundation preparation")
    root = Path(__file__).resolve().parents[1]
    for name, digest in sources.items():
        target = directory/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (root/name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("source changed while preserving its frozen copy")
        target.write_bytes(payload)
    files = {name: file_hash(directory/name) for name in ("plan.json", "lesson-admission.json", "training-manifest.json",
        "bank-admission.json", "capacity-choice.json", "banks.pt", "protected.pt", "training-transcripts.pt")}
    protocol = {"contract": contract(), "source_sha256": sources, "files_sha256": files,
        "capacity_choice": choice, "proof_directory": str(Path(proof_directory).resolve()),
        "proof_sha256": json_digest(proof), "execution_profile": proof["execution_profile"],
        "config": asdict(config), "learning_rate": choice["learning_rate"],
        "initial_weights_sha256": checkpoint_digest(build_sequence_student(MODEL_SEED, config=config)),
        "history_count": len(history), "protected_count": len(exclusions),
        "training_count": len(training), "created_utc": utc()}
    atomic_json(directory/"protocol.json", protocol)
    atomic_json(directory/"preparation.json", {"status": "prepared", "protocol_sha256": file_hash(directory/"protocol.json"),
        "wall_seconds": time.monotonic()-started, "neural_training_or_inference": False})
    return protocol


def load_protocol(directory):
    directory = Path(directory)
    protocol = read(directory/"protocol.json")
    if protocol["contract"] != contract() or protocol["source_sha256"] != source_hashes():
        raise ValueError("foundation contract or source identity changed")
    for name, digest in protocol["source_sha256"].items():
        if file_hash(directory/"source"/name) != digest:
            raise ValueError("foundation preserved source changed")
    for name, digest in protocol["files_sha256"].items():
        if file_hash(directory/name) != digest:
            raise ValueError("foundation frozen input changed: " + name)
    for path, digest in protocol["capacity_choice"]["files_sha256"].items():
        if file_hash(path) != digest:
            raise ValueError("capacity provenance changed")
    return protocol


def trainer(directory, arm, device="cpu", payload=None):
    if arm not in ARMS:
        raise ValueError("unknown foundation arm")
    directory = Path(directory)
    protocol = load_protocol(directory)
    return FoundationTrainer(read(directory/"plan.json"), arm, seed=MODEL_SEED,
        config=SequenceConfig(**protocol["config"]), learning_rate=protocol["learning_rate"], device=device,
        protected_transcripts=torch.load(directory/"protected.pt", map_location="cpu", weights_only=True), payload=payload)


def append_journal(path, value):
    with Path(path).open("a", encoding="utf8") as handle:
        handle.write(json.dumps(value, allow_nan=False, separators=(",", ":"))+"\n")
        handle.flush()
        os.fsync(handle.fileno())


def account_report(receipt, report):
    if report["physical_optimizer_updates"] is None:
        receipt["physical_work_unknown"] = True
    for key in ("physical_optimizer_updates", "neural_attempted_episode_exposures",
                "completed_microbatch_episode_exposures", "drawn_episode_exposures"):
        if report[key] is not None:
            receipt[key] += report[key]


def train(directory, arm):
    from experiments.execution_profile import runtime_profile, assert_strict_profile
    from experiments.foundation_runtime_probe import load_proof
    if arm not in ARMS:
        raise ValueError("unknown foundation arm")
    started = time.monotonic()
    directory = Path(directory)
    protocol = load_protocol(directory)
    proof = load_proof(protocol["proof_directory"], SequenceConfig(**protocol["config"]))
    require_admitted_proof(proof)
    if json_digest(proof) != protocol["proof_sha256"]:
        raise ValueError("foundation runtime proof changed")
    if proof["learning_rate"] != protocol["learning_rate"]:
        raise ValueError("foundation learning rate differs from execution proof")
    runtime = runtime_profile("cuda:0")
    if runtime != protocol["execution_profile"]:
        raise ValueError("foundation execution runtime differs from proof")
    folder = directory/arm
    folder.mkdir(exist_ok=False)  # Claims the arm; failure is preserved, never retried.
    journal = folder/"steps.jsonl"
    receipt = {"schema": SCHEMA, "arm": arm, "status": "running", "started_utc": utc(),
        "protocol_sha256": file_hash(directory/"protocol.json"), "execution_profile": runtime,
        "checkpoints": {}, "physical_optimizer_updates": 0, "neural_attempted_episode_exposures": 0,
        "completed_microbatch_episode_exposures": 0, "drawn_episode_exposures": 0,
        "physical_work_unknown": False, "automatic_promotion": False}
    atomic_json(folder/"receipt.json", receipt)
    learner = None
    torch.cuda.reset_peak_memory_stats()
    try:
        learner = trainer(directory, arm, device="cuda:0")
        receipt["setup_seconds"] = time.monotonic()-started
        for step in range(TOTAL+1):
            if step in STEPS:
                load_protocol(directory)
                payload = learner.snapshot()
                digest = checkpoint_digest(learner.model)
                if step == 0 and digest != protocol["initial_weights_sha256"]:
                    raise ValueError("foundation initialization differs")
                name = f"checkpoint-{step:06d}.pt"
                atomic_checkpoint(folder/name, {"schema": SCHEMA, "arm": arm,
                    "protocol_sha256": receipt["protocol_sha256"], "learner": payload,
                    "weights_sha256": digest, "execution_profile": runtime})
                receipt["checkpoints"][name] = file_hash(folder/name)
                atomic_json(folder/"receipt.json", receipt)
                print(json.dumps({"arm": arm, "checkpoint": step}), flush=True)
            if step == TOTAL:
                break
            assert_strict_profile()
            append_journal(journal, {"event": "started", "cursor": step, "utc": utc()})
            try:
                report = learner.step()
            except BaseException:
                report = learner.last_report
                account_report(receipt, report)
                try:
                    append_journal(journal, {"event": "failed", "report": report})
                except BaseException as journal_error:
                    receipt["failure_journal_error"] = repr(journal_error)
                raise
            # Count completed arithmetic before journal I/O can fail.
            account_report(receipt, report)
            append_journal(journal, {"event": "completed", "report": report})
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt.update(ended_utc=utc(), wall_seconds=time.monotonic()-started,
            retained_updates=None if learner is None else learner.cursor,
            journal_sha256=file_hash(journal) if journal.exists() else None,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            timing_scope="train function entry through finalization; includes setup/materialization/checkpoint I/O, excludes imports/profile setup/final receipt write; overlap is not dedicated GPU time",
            hard_stop_note="An unfinished running receipt or journal start has unknown work; no automatic retry.")
        atomic_json(folder/"receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "verify", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=Path)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.phase in ("train", "evaluate"):
        from experiments.execution_profile import configure_strict_profile
        configure_strict_profile()
    if args.phase == "prepare":
        if args.capacity is None or args.proof is None:
            parser.error("prepare requires --capacity and --proof")
        prepare(args.output, args.capacity, args.proof)
    elif args.phase == "train":
        if args.arm is None:
            parser.error("train requires --arm")
        train(args.output, args.arm)
    else:
        from experiments import foundation_results
        getattr(foundation_results, args.phase)(args.output)


if __name__ == "__main__":
    main()
