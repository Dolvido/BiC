"""Exact paired continuation over authenticated replay; no automatic retry or promotion.

The caller freezes admitted file-backed data and reviewed source before run.
Both learners receive every bundle, with first-arm order alternated. Full
checkpoints and native raw evaluation records are retained at fixed endpoints.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import platform
import time
import traceback

from experiments.foundation_layout_study import (ROOT, CONFIG, native, digest,
    read, publish, relative_root, verify_pins, encoded, _hash, utc, denominators,
    Journal as BaseJournal)

SCHEMA = "bic-shared-state-continuation-pilot-v1"
PROTOCOL = "docs/SHARED_STATE_CONTINUATION_PROTOCOL.md"
ARMS = ("fast", "slow")
STEPS = (648, 1296, 1944)
SEED, ORIGIN_STEP, UPDATES, SOURCE_UPDATES, MICRO, BATCH, SECONDS = 852604001, 648, 1944, 648, 32, 32, 3600
NEW_UPDATES = UPDATES - ORIGIN_STEP
BANKS = ("dev", "previous_dev", "train_fit", "retention")
PRIOR = "runs/shared-state-rate-pilot-local/attempt-001"
PRIOR_LAUNCH = "bcfdfb8f5b369d4591988e7b7db26321256cd6a9db263003979b3d10619fd3f0"
PRIOR_SUMMARY = "e268e983015ef56eb10b14d864e82566e2a102c0de0e156b3d5d03f314548931"
SOURCE_DATA = "runs/entity-retrieval-data-local/attempt-001"
RATES = dict(fast=.003, slow=.0003)
DATA_MANIFEST_SHA256 = "7c0d1f208f59a183cf5c052c991c842c2358107a3c40c84f4fc36ba42fa63612"
OBJECTIVE = "unchanged-sequence-objective-three-family-mean-v1"


def contract():
    return dict(schema=SCHEMA, arms=list(ARMS), original_seed=SEED, config=CONFIG,
        auxiliary_weights=dict(fast=.3, slow=.3), origin_step=ORIGIN_STEP,
        lifetime_updates_per_arm=UPDATES, new_updates_per_arm=NEW_UPDATES,
        source_bundles=SOURCE_UPDATES, micro_batch_size=MICRO, evaluation_batch_size=BATCH,
        learning_rates=dict(RATES), optimizer="Original full AdamW continuation",
        gradient_clip=1., objective_id=OBJECTIVE, layout="original",
        checkpoints=list(STEPS[1:]), evaluation_steps=list(STEPS), evaluation_banks=list(BANKS),
        device="cuda:0", max_seconds=SECONDS, cpu_threads=1, cpu_interop_threads=1,
        first_arm="fast at even global consumed IDs; slow at odd IDs",
        origin=dict(directory=PRIOR,launch_sha256=PRIOR_LAUNCH,summary_sha256=PRIOR_SUMMARY),
        source_data_directory=SOURCE_DATA, source_data_manifest_sha256=DATA_MANIFEST_SHA256,
        restoration="Exact complete weights, full AdamW, original recipe/evidence/accounting; no reset",
        replay="Global IDs648..1943; source ID=global%648, replay pass=global//648; nested rows unchanged",
        state_targets="Computed once per shared consumed bundle; labels only enter training loss",
        evaluation_provenance="dev720 fresh recorded-study-excluded; previous_dev720/fit108/retention720 reused",
        primary_screens="Existing compare fast-to-slow final, and each own648-to1944; previous-dev factual regressions separately",
        automatic_retry=False, automatic_promotion=False, teacher_calls=0,
        scope="Single-seed continued acquisition and rate histories, not broad capability or tutor benefit.")


def source_pins():
    from experiments import shared_state_continuation_data as data
    from experiments import shared_state_continuation as kernel, shared_state_replay as replay, foundation_layout_evaluation as evaluation
    names = set(data.source_hashes()) | set(kernel.source_hashes()) | set(replay.source_hashes()) | set(evaluation.source_hashes())
    names |= {"experiments/shared_state_continuation_pilot.py", "experiments/shared_state_student.py", "experiments/shared_state_targets.py", "experiments/shared_state_screen.py", "experiments/shared_state_continuation_report.py",
              "experiments/foundation_layout_study.py", "experiments/execution_profile.py",
              "docs/SHARED_STATE_CONTINUATION_DATA.md", PROTOCOL}
    if native(ROOT/"experiments/__init__.py").exists():
        raise ValueError("experiment namespace initializer must remain absent")
    return {name: digest(ROOT/name) for name in sorted(names)}


def installed_runtime():
    import torch
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                torch_git=torch.version.git_version, cuda=torch.version.cuda)


def _manifest(directory, pin):
    from experiments import shared_state_continuation_data as data
    manifest = data.load_manifest(directory, expected_manifest_sha256=pin)
    if (manifest["updates"] != SOURCE_UPDATES or manifest["layout"] != "original"
            or manifest["micro_batch_size"] != MICRO or len(manifest["training"]) != SOURCE_UPDATES
            or manifest["training_episodes"] != SOURCE_UPDATES*3*MICRO
            or set(manifest["bank_inventory"]) != set(BANKS)
            or manifest["dev_seed"] != 852702001 or manifest["exclusion_count"] != 1703160):
        raise ValueError("fixed data schedule differs")
    for cursor, record in enumerate(manifest["training"]):
        if record["global_cursor"] != cursor or record["path"] != f"training/{cursor:04d}.pt":
            raise ValueError("canonical ordered global bundle IDs required")
    if {name: row["episodes"] for name, row in manifest["bank_inventory"].items()} != dict(dev=720, previous_dev=720, train_fit=108, retention=720):
        raise ValueError("fixed fresh/reused bank sizes differ")
    return manifest


def freeze(output, *, data_directory, expected_manifest_sha256, input_pins=None):
    """Bind exact origin metadata with two CPU checkpoint loads and no models."""
    started, cpu = time.monotonic(), time.process_time()
    output, directory = Path(output).resolve(), Path(data_directory).resolve()
    sources = source_pins(); manifest = _manifest(directory, expected_manifest_sha256)
    pins = dict(input_pins or {})
    pins.update({PRIOR+"/launch.json":PRIOR_LAUNCH, PRIOR+"/execution/summary.json":PRIOR_SUMMARY,
        relative_root(directory/"manifest.json"):expected_manifest_sha256,
        relative_root(directory/"preparation.json"):digest(directory/"preparation.json")})
    prior_launch, prior_summary = read(ROOT/PRIOR/"launch.json"), read(ROOT/PRIOR/"execution/summary.json")
    if (prior_summary["status"]!="completed" or prior_summary["launch_sha256"]!=PRIOR_LAUNCH
            or prior_launch["contract"]["learning_rates"]!=RATES
            or any(prior_summary["arms"][arm]["cursor"]!=ORIGIN_STEP for arm in ARMS)):
        raise ValueError("completed original paired rate study required")
    verify_pins(prior_launch["source_sha256"])
    parent = ROOT/manifest["parent_data_directory"]
    pins[relative_root(parent/"manifest.json")] = manifest["parent_manifest_sha256"]
    for record in manifest["training"]: pins[relative_root(parent/record["path"])] = record["sha256"]
    for name,pin in manifest["artifacts_sha256"].items(): pins[relative_root(directory/name)] = pin
    origin_records = {arm:prior_summary["checkpoints"][arm][str(ORIGIN_STEP)] for arm in ARMS}
    for arm,record in origin_records.items():
        if record["path"]!=PRIOR+f"/execution/checkpoints/{arm}-0648.pt": raise ValueError("exact final checkpoint paths required")
        pins[record["path"]]=record["sha256"]
    verify_pins(pins)
    native(output).mkdir(parents=True,exist_ok=False)
    receipt=dict(status="running",checkpoint_metadata_load_attempts=0,checkpoint_metadata_loads=0,
        model_constructions=0,curriculum_archive_loads=0,automatic_retry=False)
    try:
        import torch
        from experiments.shared_state_continuation import checkpoint_identity
        identities={}
        for arm,record in origin_records.items():
            receipt["checkpoint_metadata_load_attempts"]+=1
            image=native(ROOT/record["path"]).read_bytes()
            if hashlib.sha256(image).hexdigest()!=record["sha256"]: raise ValueError("origin image changed")
            payload=torch.load(io.BytesIO(image),map_location="cpu",weights_only=True)
            receipt["checkpoint_metadata_loads"]+=1
            identities[arm]=checkpoint_identity(payload)
            if (identities[arm]["arm"]!=arm or identities[arm]["step"]!=ORIGIN_STEP
                    or identities[arm]["launch_sha256"]!=PRIOR_LAUNCH or identities[arm]["learning_rate"]!=RATES[arm]
                    or identities[arm]["config"]!=CONFIG or identities[arm]["runtime"]!=prior_summary["runtime"]):
                raise ValueError("origin checkpoint metadata differs")
            del image,payload
        snapshots={}
        for index,(name,pin) in enumerate(sorted(sources.items())):
            raw=native(ROOT/name).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=pin: raise ValueError("source changed while freezing")
            local=f"sources/{index:03d}.bin";target=native(output/local);target.parent.mkdir(exist_ok=True)
            with target.open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
            snapshots[name]=local
        if source_pins()!=sources: raise ValueError("source closure changed while freezing")
        receipt["status"]="completed"
        launch=dict(schema=SCHEMA,contract=contract(),data_directory=relative_root(directory),
            data_manifest_sha256=expected_manifest_sha256,source_sha256=sources,source_snapshots=snapshots,
            input_sha256=pins,installed_runtime=installed_runtime(),bank_inventory=manifest["bank_inventory"],
            origin_checkpoints=origin_records,origin_identities=identities,expected_runtime=prior_summary["runtime"],
            origin_accounting={arm:prior_summary["arms"][arm]["accounting"] for arm in ARMS},
            created_utc=utc(),preparation=read(directory/"preparation.json"),
            freeze_cost=dict(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu),
            freeze_work=deepcopy(receipt))
        return publish(output/"launch.json",launch)
    except BaseException as error:
        receipt.update(status="failed",error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
        publish(output/"freeze-receipt.json",receipt)


class Journal(BaseJournal):
    def __init__(self, path):
        super().__init__(path)
        self.chain = _hash([SCHEMA, "events"])


def run(output, *, expected_launch_sha256):
    started, cpu = time.monotonic(), time.process_time()
    output = Path(output).resolve()
    if digest(output/"launch.json") != expected_launch_sha256:
        raise ValueError("caller-pinned launch differs")
    launch = read(output/"launch.json")
    if launch["schema"] != SCHEMA or launch["contract"] != contract():
        raise ValueError("fixed run contract differs")
    attempt = output/"execution"; native(attempt).mkdir(exist_ok=False)
    deadline = started+SECONDS
    receipt = dict(schema=SCHEMA, status="running", launch_sha256=expected_launch_sha256,
        started_utc=utc(), pid=os.getpid(), automatic_retry=False, automatic_promotion=False,
        teacher_calls=0, artifact_sha256={}, evaluations={}, checkpoints={}, arms={},
        model_construction_attempts=0, model_constructions=0, snapshot_attempts=0, snapshots=0,
        data_load_attempts=0, data_load_completions=0, partial_work_unknown=False)
    journal, active_bank, active_ledger = None, None, None
    models, kernels, owners, prepared_banks, operations = {}, {}, {}, {}, {}
    replay_reader = None
    validation_work, profile = None, None

    def boundary():
        if time.monotonic() >= deadline:
            raise TimeoutError("fixed 3600-second allowance expired; no automatic continuation")

    def event(value):
        try: journal.event(value)
        except BaseException:
            receipt["partial_work_unknown"] = True
            raise

    def operation(kind, identity, call):
        boundary()
        stats = operations.setdefault(kind, dict(attempts=0, completions=0, failures=0, wall_seconds=0., cpu_seconds=0.))
        stats["attempts"] += 1
        receipt["active_operation"] = dict(kind=kind, identity=identity)
        t, c = time.monotonic(), time.process_time()
        try:
            event(dict(event="intent", kind=kind, identity=identity))
            result = call(); stats["completions"] += 1
            event(dict(event="complete", kind=kind, identity=identity,
                wall_seconds=time.monotonic()-t, cpu_seconds=time.process_time()-c))
            receipt["active_operation"] = None
            return result
        except BaseException:
            stats["failures"] += 1
            raise
        finally:
            stats["wall_seconds"] += time.monotonic()-t
            stats["cpu_seconds"] += time.process_time()-c

    def artifact(name, value, *, checkpoint=False):
        pin = operation("publication", name, lambda: publish(attempt/name, value, checkpoint=checkpoint))
        receipt["artifact_sha256"][name] = pin
        return dict(path=relative_root(attempt/name), sha256=pin)

    def authenticate():
        if source_pins() != launch["source_sha256"] or installed_runtime() != launch["installed_runtime"]:
            raise ValueError("source or installed runtime changed")
        verify_pins(launch["input_sha256"])
        for name, local in launch["source_snapshots"].items():
            if digest(output/local) != launch["source_sha256"][name]:
                raise ValueError("frozen source copy changed")

    def load_image(record, base_directory=None):
        receipt["data_load_attempts"] += 1
        base_directory = directory if base_directory is None else base_directory
        path = (base_directory/record["path"]).resolve()
        if not path.is_relative_to(base_directory) or path == base_directory:
            raise ValueError("data artifact must stay inside admitted directory")
        raw = native(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("immutable data image pin differs")
        result = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        receipt["data_load_completions"] += 1
        return result

    def model_guard(arm):
        model = models[arm]
        if (type(model) is not SharedStateStudent or model.config != config
                or any(parameter.device != torch.device("cuda:0") or parameter.dtype != torch.float32
                       or not parameter.requires_grad for parameter in model.parameters())):
            raise ValueError("fixed shared-state model architecture/placement changed")
        execution.assert_strict_profile()

    def evaluate(arm, step):
        nonlocal active_bank, active_ledger
        model_guard(arm)
        metrics = {}
        for name, groups in prepared_banks.items():
            records, details = [], {}
            for turns, bank in groups.items():
                active_bank, active_ledger = bank, evaluation.EvaluationLedger()
                identity = f"{arm}/{step:04d}/{name}/t{turns}"
                result = operation("evaluation", identity, lambda: bank.score(models[arm], batch_size=BATCH,
                    control="normal", deadline=deadline, work=active_ledger,
                    progress=lambda value: event(dict(kind="evaluation_batch", identity=identity, **value))))
                records.extend(result["raw_records"])
                details[str(turns)] = artifact("scores/"+identity+".json", result)
                active_bank = active_ledger = None
            pooled = evaluation.score_records(records)
            if {key: value["total"] for key, value in pooled["overall"]["counts"].items()} != bank_denominators[name]:
                raise ValueError("pooled evaluation denominators differ from sealed bank")
            metrics[name] = dict(metrics=pooled, groups=details)
        model_guard(arm)
        receipt["evaluations"].setdefault(arm, {})[str(step)] = metrics
        artifact(f"scores/{arm}/{step:04d}/summary.json", metrics)

    def checkpoint(arm, step):
        model_guard(arm); receipt["snapshot_attempts"] += 1
        image = operation("snapshot", f"{arm}/{step}", kernels[arm].snapshot)
        receipt["snapshots"] += 1
        payload = dict(schema=SCHEMA, launch_sha256=expected_launch_sha256, arm=arm, step=step,
            architecture=student.ARCHITECTURE, learning_rate=RATES[arm],
            auxiliary_weight=launch["contract"]["auxiliary_weights"][arm], learner=image,
            weights_sha256=checkpoint_digest(models[arm]), runtime=receipt["runtime"])
        receipt["checkpoints"].setdefault(arm, {})[str(step)] = artifact(f"checkpoints/{arm}-{step:04d}.pt", payload, checkpoint=True)

    try:
        journal = Journal(attempt/"events.jsonl")
        publish(attempt/"started.json", receipt)
        operation("authentication", "launch", authenticate)
        import torch
        from experiments import execution_profile as execution, shared_state_student as student
        from experiments.shared_state_targets import pack_state_targets
        from experiments import foundation_layout_evaluation as evaluation
        from experiments.foundation_layout_curriculum import WorkLedger
        from experiments.foundation_layout_prepared import PreparedLayoutOwner, evidence_sha256
        from experiments.foundation_evidence import json_digest
        from experiments.shared_state_continuation import SharedStateContinuation
        from experiments.shared_state_replay import ReplayReader
        from experiments.sequence_student import SequenceConfig
        from experiments.shared_state_student import SharedStateStudent
        from brain_in_computer.dialogue_student import checkpoint_digest
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        execution.configure_strict_profile()
        receipt["runtime"] = operation("runtime", "cuda:0", execution.runtime_profile)
        if receipt["runtime"] != launch["expected_runtime"] or "5080" not in receipt["runtime"]["device"]["name"]:
            raise ValueError("declared local RTX 5080 required")
        artifact("runtime.json", receipt["runtime"])
        directory = ROOT/launch["data_directory"]
        manifest = operation("manifest", "data", lambda: _manifest(directory, launch["data_manifest_sha256"]))
        parent_directory = ROOT/manifest["parent_data_directory"]
        banks = operation("data_load", "banks", lambda: load_image(manifest["banks"]))
        if set(banks) != set(BANKS):
            raise ValueError("four fixed banks required")
        config, validation_work, bank_denominators = SequenceConfig(**CONFIG), WorkLedger(), {}
        for name in BANKS:
            bank = banks[name]
            if set(bank) != {"role", "rows"} or bank["role"] != ("train_fit" if name == "train_fit" else "dev"):
                raise ValueError("fixed evaluation roles required")
            rows = bank["rows"]
            if len(rows) != manifest["bank_inventory"][name]["episodes"]:
                raise ValueError("sealed bank size differs")
            bank_denominators[name] = denominators(rows)
            grouped = defaultdict(list)
            for index in range(0, len(rows), 2):
                pair = rows[index:index+2]
                if len(pair) != 2 or len(pair[0]["turns"]) != len(pair[1]["turns"]):
                    raise ValueError("complete uniform-turn pairs required")
                grouped[len(pair[0]["turns"])].extend(pair)
            prepared_banks[name] = {turns: operation("bank_preparation", f"{name}/t{turns}",
                lambda rows=group: evaluation.PreparedLayoutBank(rows, role=bank["role"], config=config,
                    validation_work=validation_work)) for turns, group in sorted(grouped.items())}
        del banks
        artifact("bank-identities.json", {name: {str(turns): bank.identity for turns, bank in groups.items()}
                                          for name, groups in prepared_banks.items()})
        torch.cuda.reset_peak_memory_stats()
        receipt["restoration"]={}
        for arm in ARMS:
            record=launch["origin_checkpoints"][arm]
            raw=operation("checkpoint_bytes",arm,lambda record=record:native(ROOT/record["path"]).read_bytes())
            try:
                kernels[arm]=operation("restoration",arm,lambda:SharedStateContinuation.from_checkpoint(raw,
                    expected_sha256=record["sha256"],expected_identity=launch["origin_identities"][arm],device="cuda:0"))
            except BaseException as error:
                receipt["restoration"][arm]=getattr(error,"continuation_report",{"failed":True,"details_unavailable":True})
                raise
            receipt["restoration"][arm]=deepcopy(kernels[arm].last_restore_report)
            models[arm]=kernels[arm].model;model_guard(arm)
            if (kernels[arm].cursor!=ORIGIN_STEP or kernels[arm].accounting["lifetime_kernel"]!=launch["origin_accounting"][arm]
                    or checkpoint_digest(models[arm])!=launch["origin_identities"][arm]["weights_sha256"]):
                raise ValueError("exact original lifetime state/accounting not restored")
            owners[arm]=PreparedLayoutOwner(config=config,layout="original",micro_batch_size=MICRO)
            del raw
        if kernels["fast"].evidence!=kernels["slow"].evidence: raise ValueError("paired origins consumed different lessons")
        receipt["initial_weights_sha256"]={arm:checkpoint_digest(models[arm]) for arm in ARMS}
        receipt["parameter_counts"]={arm:models[arm].parameter_counts() for arm in ARMS}
        prior_summary=read(ROOT/PRIOR/"execution/summary.json")
        for arm in ARMS:
            evaluate(arm,ORIGIN_STEP)
            for name,old_name in (("previous_dev","dev"),("train_fit","train_fit"),("retention","retention")):
                if receipt["evaluations"][arm][str(ORIGIN_STEP)][name]["metrics"]!=prior_summary["evaluations"][arm][str(ORIGIN_STEP)][old_name]["metrics"]:
                    raise ValueError("restored reused-bank native metrics differ from original endpoint")
        receipt["restored_native_metrics_equal_to_origin"]=True
        replay_reader=operation("replay_reader",SOURCE_DATA,lambda:ReplayReader(ROOT/SOURCE_DATA,expected_manifest_sha256=DATA_MANIFEST_SHA256))
        for cursor in range(ORIGIN_STEP,UPDATES):
            source=operation("replay_load",str(cursor),lambda:replay_reader.load(cursor%SOURCE_UPDATES))
            image,provenance=operation("replay_remap",str(cursor),lambda:source.remap(cursor))
            event(dict(event="replay_provenance",global_cursor=cursor,provenance=provenance))
            del source
            order = ARMS if cursor % 2 == 0 else ARMS[::-1]
            state_targets = operation("state_targets", str(cursor), lambda: {
                family: pack_state_targets(rows) for family, rows in image["bundle"]["families"].items()})
            for arm in order:
                model_guard(arm)
                token = operation("training_preparation", f"{arm}/{cursor}", lambda: owners[arm].prepare(image["bundle"],
                    expected_evidence=image["expected_evidence"], expected_evidence_sha256=provenance["prepared_evidence_sha256"]))
                report = operation("training", f"{arm}/{cursor}", lambda: kernels[arm].step(token, state_targets=state_targets, deadline=deadline))
                event(dict(event="step_report", arm=arm, report=report))
                model_guard(arm)
            del image, state_targets
            if kernels["fast"].evidence != kernels["slow"].evidence:
                raise ValueError("paired arms consumed different evidence")
            step = cursor+1
            if step % 24 == 0:
                print(encoded(dict(event="progress", completed_updates_per_arm=step,
                    wall_seconds=time.monotonic()-started)).decode().strip(), flush=True)
            if step in STEPS:
                operation("authentication", str(step), authenticate)
                for arm in ARMS: checkpoint(arm, step)
                for arm in ARMS: evaluate(arm, step)
        for arm in ARMS:
            work = kernels[arm].accounting["lifetime_kernel"]["work"]
            if (kernels[arm].cursor != UPDATES or work["retained_updates"] != UPDATES
                    or work["synchronized_optimizer_updates"] != UPDATES
                    or work["completed_forwards"] != UPDATES*3 or work["completed_backwards"] != UPDATES*3
                    or work["retained_episodes"] != UPDATES*3*MICRO or work["unknown_optimizer_outcomes"]):
                raise ValueError("completed physical training accounting differs")
            if any(value != UPDATES*3
                   for value in kernels[arm].accounting["lifetime_kernel"]["state_work"].values()):
                raise ValueError("completed auxiliary readout/objective accounting differs")
            if kernels[arm].accounting["bridge_cost"]["step_invocations"] != NEW_UPDATES:
                raise ValueError("actual new continuation update count differs")
        if operations["state_targets"]["completions"] != NEW_UPDATES:
            raise ValueError("state supervision was not computed exactly once per bundle")
        if any(operations[name]["completions"]!=NEW_UPDATES for name in ("replay_load","replay_remap")):
            raise ValueError("complete declared replay workload required")
        replay_work=replay_reader.report()
        if (replay_work["archive_loads"]!=NEW_UPDATES or replay_work["remaps"]!=NEW_UPDATES
                or replay_work["load_failures"] or replay_work["remap_failures"] or replay_work["failed"]):
            raise ValueError("actual replay work differs from complete declared passes")
        operation("authentication", "terminal", authenticate)
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        for key in ("model_construction_attempts", "model_constructions"):
            receipt[key]=sum(report.get(key,0) for report in receipt.get("restoration",{}).values())
        for arm, owner in owners.items():
            owner.close()
            receipt["arms"][arm] = dict(cursor=kernels[arm].cursor, accounting=kernels[arm].accounting,
                preparation=owner.report(), last_report=deepcopy(kernels[arm].last_report))
        if replay_reader is not None:
            receipt["replay_work"] = replay_reader.report()
        if active_bank is not None:
            receipt["active_evaluation"] = dict(report=deepcopy(active_bank.last_report),
                work=active_ledger.report() if active_ledger is not None else None)
        try:
            if "torch" in locals() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
                receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            if journal is not None:
                journal.close(); receipt["journal_sha256"] = digest(attempt/"events.jsonl")
        except BaseException as error:
            receipt.update(cleanup_error=repr(error), partial_work_unknown=True)
            if receipt["status"] == "completed": receipt["status"] = "failed"
        receipt.update(operations=operations, bank_validation_work=validation_work.report() if validation_work else None,
            ended_utc=utc(), wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
            timing_scope="Entire execution, including data reads, source guards, training, evaluation, publications and cleanup; preparation/freeze are separate.")
        publish(attempt/"summary.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "run")); parser.add_argument("output")
    parser.add_argument("--data-directory"); parser.add_argument("--manifest-sha256")
    parser.add_argument("--launch-sha256"); parser.add_argument("--input-pins")
    args = parser.parse_args()
    if args.mode == "freeze":
        result = freeze(args.output, data_directory=args.data_directory,
            expected_manifest_sha256=args.manifest_sha256,
            input_pins=read(args.input_pins) if args.input_pins else None)
        print(encoded(dict(launch_sha256=result)).decode().strip(), flush=True)
    else:
        result = run(args.output, expected_launch_sha256=args.launch_sha256)
        print(encoded(dict(status=result["status"], wall_seconds=result["wall_seconds"])).decode().strip(), flush=True)
