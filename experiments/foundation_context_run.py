"""Frozen, local inference-only contextual diagnostic; never resumes or retries.

Preparation creates one declared bank and binds previously verified checkpoint
bytes. Inference restores weights, not an optimizer or a curriculum index.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import time

SCHEMA = "bic-context-diagnostic-run-v1"
REPOSITORY = Path(__file__).resolve().parents[1]
STUDY = "runs/foundation-objective-study-local/study"
PARENT_STAGE_SHA = "6a2936f07bbc93565261c55852f860db8b02896df79eb26a88ea2ffc7cd20d79"
PARENT_PROTOCOL_SHA = "6bcb678ae8f0f15f34ece60df7d7077d8ab9d479d2f34a5e6f2bb29e103ef7f3"
PARENT_VERIFICATION_SHA = "2dbf7a9262b27fd08391e66ae0cf4428de952b9595e092f061ed391389b57f19"
SEEDS = (8472, 8473, 8474)
BANK_SEED = 851910001
CONFIG = dict(width=192, layers=4, heads=4, feedforward=768,
              max_positions=1024, max_turns=12, max_input_bytes=128, max_output_bytes=32)
SOURCES = tuple("brain_in_computer/"+n+".py" for n in
    ("__init__", "model", "regions", "language", "learning_student", "dialogue_student")) + tuple(
    "experiments/"+n+".py" for n in ("cognitive_curriculum", "composition_curriculum", "sequence_data",
    "composition_data", "sequence_student", "execution_profile", "foundation_context_diagnostic",
    "foundation_context_evaluation", "foundation_context_run"))
PROTOCOL = "docs/FOUNDATION_CONTEXT_DIAGNOSTIC_PROTOCOL.md"
VALIDATION = "runs/foundation-context-evaluator-validation-local/attempt-001/report.json"
GENERATOR_VALIDATION = "runs/foundation-context-validation-local/attempt-001/report.json"
GENERATOR_VALIDATION_SHA = "dab652cd33a5783dcfd70beda07bdd19f63b9cfcba0bfbeef519a117f6c7f9fe"


def utc():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def native(path):
    value = str(Path(path).absolute())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\"+value[2:] if value.startswith("\\\\") else "\\\\?\\"+value
    return Path(value)


def digest(path):
    result = hashlib.sha256()
    with native(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(native(path).read_bytes())


def publish(path, value):
    """Exclusive new artifact; output directories are never reused."""
    path = native(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())


def source_pins():
    if (REPOSITORY/"experiments/__init__.py").exists():
        raise ValueError("experiment namespace initializer must remain absent")
    return {name: digest(REPOSITORY/name) for name in (*SOURCES, PROTOCOL)}


def verify_pins(pins):
    for name, expected in pins.items():
        if digest(REPOSITORY/name) != expected:
            raise ValueError("pinned input changed: "+name)


def parents():
    """Read only authenticated terminal metadata; no historic tensors/models."""
    pins = {STUDY+"/main/stage.json": PARENT_STAGE_SHA,
            STUDY+"/protocol.json": PARENT_PROTOCOL_SHA,
            STUDY+"/main/verification.json": PARENT_VERIFICATION_SHA}
    verify_pins(pins)
    stage, protocol, verified = (read(REPOSITORY/n) for n in pins)
    if stage["status"] != "completed" or verified["status"] != "completed":
        raise ValueError("completed parent study required")
    if (stage["protocol_sha256"] != PARENT_PROTOCOL_SHA or
        verified["protocol_sha256"] != PARENT_PROTOCOL_SHA or
        stage["artifacts_sha256"]["main/verification.json"] != PARENT_VERIFICATION_SHA or
        protocol["contract"]["config"] != CONFIG):
        raise ValueError("parent evidence bindings differ")
    # The model implementation must still be the checkpoint producer's version.
    for name in SOURCES:
        if name in protocol["source_sha256"]:
            if digest(REPOSITORY/name) != protocol["source_sha256"][name]:
                raise ValueError("checkpoint producer source changed: "+name)
    models = []
    for seed in SEEDS:
        initial = [verified["results"][f"{arm}-seed{seed}"]["weights_sha256"]["0"]
                   for arm in ("baseline", "balanced_reply")]
        if initial[0] != initial[1]:
            raise ValueError("objectives do not share the initial state")
        for label, objective, step in (("initial", "baseline", 0),
                                      ("baseline", "baseline", 3072),
                                      ("balanced_reply", "balanced_reply", 3072)):
            job_id = f"{objective}-seed{seed}"
            record = verified["results"][job_id]
            if record["exact_official_checkpoint_restores"] is not True:
                raise ValueError("unverified parent restoration")
            name = f"main/{job_id}/checkpoint-{step:06d}.pt"
            archive = verified["input_file_sha256"][name]
            if stage["artifacts_sha256"][name] != archive:
                raise ValueError("parent archive manifests disagree")
            pins[STUDY+"/"+name] = archive
            models.append(dict(id=f"{label}-seed{seed}", seed=seed, condition=label,
                step=step, job=record["job"], checkpoint=STUDY+"/"+name,
                archive_sha256=archive, weights_sha256=record["weights_sha256"][str(step)]))
    verify_pins(pins)
    return pins, models, protocol["execution_profile"]


def prepare(output):
    start, cpu = time.monotonic(), time.process_time()
    output = native(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, status="running", started_utc=utc(), neural_work=0)
    work = None
    try:
        source = source_pins()
        pins, models, runtime = parents()
        verify_pins({GENERATOR_VALIDATION: GENERATOR_VALIDATION_SHA})
        generator_validation = read(REPOSITORY/GENERATOR_VALIDATION)
        verify_pins(generator_validation["source_sha256"])
        pins[GENERATOR_VALIDATION] = GENERATOR_VALIDATION_SHA
        pins.update(generator_validation["source_sha256"])
        validation = read(REPOSITORY/VALIDATION)
        if validation["status"] != "passed" or validation["tests"] != 8:
            raise ValueError("passed fixed evaluator validation required")
        verify_pins(validation["source_sha256"])
        for name in ("experiments/foundation_context_evaluation.py", "tests/test_foundation_context_evaluation.py"):
            if name not in validation["source_sha256"]:
                raise ValueError("tested evaluator source binding missing")
        pins[VALIDATION] = digest(REPOSITORY/VALIDATION)
        pins.update(validation["source_sha256"])
        from experiments.foundation_context_diagnostic import build_bank, WorkLedger
        work = WorkLedger()
        bank = build_bank(seed=BANK_SEED, pairs_per_group=8, work=work)
        publish(output/"bank.json", bank)
        for name in source:
            destination = native(output/"source"/name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(native(REPOSITORY/name).read_bytes())
        if source_pins() != source:
            raise ValueError("sources changed while preparing")
        verify_pins(pins)
        launch = dict(schema=SCHEMA, source_sha256=source, input_sha256=pins,
            bank_sha256=digest(output/"bank.json"), rows_sha256=bank["rows_sha256"],
            models=models, expected_runtime=runtime, config=CONFIG, batch_size=32,
            bank_seed=BANK_SEED, pairs_per_group=8, max_seconds=1800,
            work_bound=dict(model_constructions=9, checkpoint_loads=9,
                episode_evaluations=18144, scored_query_turns=36288,
                sequence_forward_calls=567, free_decoder_calls_max=18711,
                optimizer_updates=0, backwards=0, tutor_calls=0),
            absent_paths=["experiments/__init__.py"], automatic_retry=False,
            automatic_promotion=False, model_order="seed; initial, baseline, balanced_reply",
            scope="Post hoc development diagnostic; new two-query grammar; no pristine-test claim or causal training inference.")
        publish(output/"launch.json", launch)
        report.update(status="completed", launch_sha256=digest(output/"launch.json"),
            bank_sha256=launch["bank_sha256"], rows_sha256=bank["rows_sha256"])
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report.update(ended_utc=utc(), wall_seconds=time.monotonic()-start,
            cpu_seconds=time.process_time()-cpu, work=work.report() if work else None)
        publish(output/"preparation.json", report)
    return report


def run(output, expected_launch_sha256):
    start, cpu = time.monotonic(), time.process_time()
    output = native(output)
    if digest(output/"launch.json") != expected_launch_sha256:
        raise ValueError("explicit prospective launch digest differs")
    launch = read(output/"launch.json")
    attempt = output/"inference"
    attempt.mkdir(exist_ok=False)
    receipt = dict(schema=SCHEMA, status="running", started_utc=utc(),
        launch_sha256=expected_launch_sha256, completed_models=[], model_constructions=0,
        checkpoint_loads=0, optimizer_updates=0, backwards=0, tutor_calls=0,
        process_id=os.getpid(), partial_work_unknown=False)
    deadline = start+launch["max_seconds"]
    ledger, validation_work, ledgers = None, None, []
    journal = native(attempt/"events.jsonl").open("xb")

    def boundary():
        if time.monotonic() >= deadline:
            raise TimeoutError("fixed inference time budget expired; no automatic continuation")

    def event(value):
        journal.write(encoded(dict(utc=utc(), **value))); journal.flush()
        os.fsync(journal.fileno())

    def check():
        if source_pins() != launch["source_sha256"]:
            raise ValueError("diagnostic sources changed")
        if digest(output/"bank.json") != launch["bank_sha256"]:
            raise ValueError("diagnostic bank changed")
        verify_pins(launch["input_sha256"])

    def combined_work():
        if not ledgers:
            return None
        totals = {key:sum(item.counts[key] for item in ledgers) for key in ledgers[0].counts}
        totals.update(unmatched_sequence_attempts=totals["sequence_forward_attempts"]-totals["sequence_forward_completions"],
            unmatched_decoder_attempts=totals["decoder_recurrent_attempts"]-totals["decoder_recurrent_completions"])
        return totals

    try:
        check()
        boundary()
        import torch
        from experiments.execution_profile import configure_strict_profile, runtime_profile, assert_strict_profile
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        configure_strict_profile()
        runtime = runtime_profile("cuda:0")
        if runtime != launch["expected_runtime"]:
            raise ValueError("local inference runtime differs from fixed parent runtime")
        receipt["execution_profile"] = runtime
        from experiments.sequence_student import SequenceConfig, build_sequence_student
        from brain_in_computer.dialogue_student import checkpoint_digest
        from experiments.foundation_context_diagnostic import WorkLedger
        from experiments.foundation_context_evaluation import PreparedContextBank, EvaluationLedger
        validation_work = WorkLedger()
        boundary()
        bank = read(output/"bank.json")
        if bank["rows_sha256"] != launch["rows_sha256"]:
            raise ValueError("frozen row identity differs")
        prepared = PreparedContextBank(bank, validation_work=validation_work)
        del bank
        torch.cuda.reset_peak_memory_stats()
        for spec in launch["models"]:
            boundary()
            assert_strict_profile()
            event(dict(kind="model_intent", model=spec["id"]))
            image = native(REPOSITORY/spec["checkpoint"]).read_bytes()
            if hashlib.sha256(image).hexdigest() != spec["archive_sha256"]:
                raise ValueError("checkpoint byte image differs")
            boundary()
            saved = torch.load(io.BytesIO(image), map_location="cpu", weights_only=True)
            receipt["checkpoint_loads"] += 1
            if (type(saved) is not dict or set(saved) != {"schema", "stage", "job", "protocol_sha256",
                    "learner", "weights_sha256", "execution_profile"}
                or saved["schema"] != "bic-foundation-objective-study-v1" or saved["stage"] != "main"
                or saved["job"] != spec["job"] or saved["protocol_sha256"] != PARENT_PROTOCOL_SHA
                or saved["learner"]["schema"] != "bic-foundation-objective-trainer-v1"
                or type(saved["learner"]["cursor"]) is not int or saved["learner"]["cursor"] != spec["step"]
                or saved["execution_profile"] != runtime
                or saved["learner"]["recipe"]["config"] != CONFIG
                or saved["weights_sha256"] != spec["weights_sha256"]):
                raise ValueError("checkpoint inference envelope differs")
            boundary()
            model = build_sequence_student(spec["seed"], config=SequenceConfig(**CONFIG), device="cuda:0")
            receipt["model_constructions"] += 1
            model.load_state_dict(saved["learner"]["weights"], strict=True)
            del saved, image
            if checkpoint_digest(model) != spec["weights_sha256"]:
                raise ValueError("loaded parameter bytes differ")
            boundary()
            def progress(value):
                event(dict(model=spec["id"], **value))
            ledger = EvaluationLedger()
            ledgers.append(ledger)
            result = prepared.score(model, batch_size=32, progress=progress, deadline=deadline, work=ledger)
            torch.cuda.synchronize()
            assert_strict_profile()
            if checkpoint_digest(model) != spec["weights_sha256"]:
                raise ValueError("inference changed parameter bytes")
            publish(attempt/(spec["id"]+".json"), dict(model=spec, result=result))
            receipt["completed_models"].append(spec["id"])
            event(dict(kind="model_complete", model=spec["id"], work=ledger.report()))
            print(json.dumps(dict(status="model_complete", model=spec["id"], work=ledger.report())), flush=True)
            del model
            gc.collect(); torch.cuda.empty_cache()
        check()
        if (receipt["completed_models"] != [m["id"] for m in launch["models"]]
            or receipt["model_constructions"] != 9 or receipt["checkpoint_loads"] != 9):
            raise ValueError("completed model coverage differs from declared work")
        total = combined_work()
        exact = dict(batch_intents=567, completed_batches=567, sequence_forward_attempts=567,
            sequence_forward_completions=567, encoder_episode_attempts=18144,
            encoder_episode_completions=18144, bos_reply_context_attempts=217728,
            free_query_context_attempts=36288, free_query_context_completions=36288,
            scored_episode_records=18144, unmatched_sequence_attempts=0, unmatched_decoder_attempts=0)
        if any(total[k] != v for k,v in exact.items()):
            raise ValueError("completed inference work differs from declared counts")
        if (not 567 <= total["decoder_recurrent_completions"] <= 18711
            or total["decoder_row_step_attempts"] != total["decoder_row_step_completions"]
            or not 36288 <= total["decoder_row_step_completions"] <= 1197504):
            raise ValueError("free reply work exceeds the declared bounds")
        receipt.update(status="completed", peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        receipt.update(status="failed", error=repr(error), partial_work_unknown=True)
        try:
            if "torch" in locals() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
        except BaseException as sync_error:
            receipt["failure_synchronization_error"] = repr(sync_error)
        raise
    finally:
        try:
            journal.flush(); os.fsync(journal.fileno()); journal.close()
        except BaseException as error:
            receipt.update(status="failed", journal_close_error=repr(error), partial_work_unknown=True)
        receipt.update(ended_utc=utc(), wall_seconds=time.monotonic()-start,
            cpu_seconds=time.process_time()-cpu, evaluation_work=combined_work(),
            deadline_overrun_seconds=max(0., time.monotonic()-deadline),
            validation_work=validation_work.report() if validation_work else None,
            artifacts_sha256={p.name:digest(p) for p in attempt.iterdir() if p.is_file()})
        publish(attempt/"receipt.json", receipt)
    if receipt["status"] != "completed":
        raise RuntimeError("inference evidence publication failed; see preserved receipt")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-launch-sha256")
    args = parser.parse_args()
    if args.command == "run" and not args.expected_launch_sha256:
        parser.error("run requires the frozen launch SHA256")
    result = prepare(args.output) if args.command == "prepare" else run(args.output, args.expected_launch_sha256)
    print(json.dumps({k:result[k] for k in ("status", "wall_seconds", "cpu_seconds")}), flush=True)


if __name__ == "__main__":
    main()
