"""Budgeted whole-curriculum acquisition, durable state and native transfer tests.

The only learning path is the unchanged public continuation/owner interface.
No teacher transport, score-based stopping, sample allocation or model promotion.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import gc
import hashlib
import io
from pathlib import Path
import time
import traceback

from experiments.foundation_layout_study import (ROOT, CONFIG, Journal, native,
    digest, read, publish, relative_root, verify_pins, encoded, utc, denominators)

SCHEMA = "bic-sustained-shared-acquisition-v1"
PROTOCOL = "docs/SHARED_ACQUISITION_PROTOCOL.md"
ORIGIN_CURSOR, INTERVAL, MICRO, BATCH = 2160, 648, 32, 32
CACHE_BYTES, RESIDENT_LIMIT, FINAL_RESERVE = 512*1024**2, 8*1024**3, 60
PRIOR = "runs/verified-tutor-run-local/attempt-001"
PRIOR_LAUNCH = "e5879d9e433240cd82ffbf131941a452fb6897ad75c3ec925f8dce2c10840d9b"
PRIOR_SUMMARY = "23e6286ee25b8f52c974e18d692670619cf9c200c7e3d53117099f854f7695ea"
PARENT_SHA = "365c4f5018f5a74c3ada25ed0f9168578bf3088afeab22866727a6823258b536"
TRAINING = "runs/recurrent-read-data-local/attempt-001"
TRAINING_SHA = "c420a43a3a45f63347288dda7db48c71b3e7fa183b4a1681c2ec89eabeb08654"
BANK_COUNTS = dict(dev=648, transfer_original=432, transfer_varied=432,
                   train_fit=108, retention=720)


def source_hashes():
    from experiments import shared_state_continuation as bridge
    from experiments import foundation_layout_evaluation as evaluation
    from experiments import sustained_replay as replay
    from experiments import shared_acquisition_transfer_data as data
    result = {**bridge.source_hashes(), **evaluation.source_hashes(),
              **replay.source_hashes(), **data.source_hashes()}
    for name in ("experiments/sustained_acquisition.py", "experiments/sustained_acquisition_report.py",
                 "experiments/shared_state_screen.py", "experiments/foundation_layout_study.py", PROTOCOL):
        result[name] = digest(ROOT/name)
    return result


def curriculum_state(cursor):
    if type(cursor) is not int or cursor < ORIGIN_CURSOR:
        raise ValueError("valid continuing learner cursor required")
    updates = cursor-ORIGIN_CURSOR
    return dict(origin_cursor=ORIGIN_CURSOR, cursor=cursor,
        curriculum_updates=updates, completed_passes=updates//INTERVAL,
        next_source_index=updates%INTERVAL, source_bundles=INTERVAL)


def contract(budget_seconds):
    if type(budget_seconds) is not int or not 120 <= budget_seconds <= 86400:
        raise ValueError("explicit local wall budget from120 seconds to24 hours required")
    return dict(schema=SCHEMA, budget_seconds=budget_seconds,
        final_reserve_seconds=FINAL_RESERVE, cache_archive_bytes=CACHE_BYTES,
        resident_peak_limit_bytes=RESIDENT_LIMIT, evaluation_interval=INTERVAL,
        origin_cursor=ORIGIN_CURSOR, micro_batch_size=MICRO, evaluation_batch_size=BATCH,
        config=CONFIG, bank_counts=BANK_COUNTS, restart_after_complete_pass=True,
        backend="unchanged SharedStateContinuation and PreparedLayoutOwner",
        teacher_calls=0, automatic_retry=False, automatic_promotion=False,
        stopping="wall budget or reported resource guard; never learned score",
        scope="Sustained single-parent acquisition and repeatedly observed finite structural/layout transfer.")


def validate_resume(previous_launch, previous_summary, commit, *, launch_sha256,
                    data_manifest_sha256, current_sources):
    """Pure cross-binding of completed work, committed state and curriculum position."""
    if (previous_launch["schema"]!=SCHEMA or previous_summary["status"]!="completed"
            or previous_summary["partial_work_unknown"]
            or previous_summary["launch_sha256"]!=launch_sha256
            or previous_launch["data_manifest_sha256"]!=data_manifest_sha256
            or previous_launch["source_sha256"]!=current_sources
            or commit["schema"]!=SCHEMA or commit["launch_sha256"]!=launch_sha256
            or commit["data_manifest_sha256"]!=data_manifest_sha256
            or commit["training_manifest_sha256"]!=TRAINING_SHA):
        raise ValueError("completed compatible learner and controller identities required")
    cursor=commit["curriculum"]["cursor"]
    if (commit["curriculum"]!=curriculum_state(cursor)
            or commit["curriculum"]!=previous_summary["curriculum"]
            or cursor<previous_launch["parent_cursor"]
            or commit["scores"]!=previous_summary["evaluations"][str(cursor)]):
        raise ValueError("committed final curriculum position and scores differ")
    return cursor


def freeze(output, *, data_directory, data_manifest_sha256, budget_seconds=1800,
           input_pins=None, resume_directory=None, resume_summary_sha256=None):
    """Bind a complete learner snapshot and cycle state; one CPU load, no models."""
    from experiments import shared_acquisition_transfer_data as data
    from experiments import recurrent_read_data as cached
    started, cpu = time.monotonic(), time.process_time()
    output, data_directory = Path(output).resolve(), Path(data_directory).resolve()
    spec = contract(budget_seconds)
    manifest = data.load_manifest(data_directory, expected_manifest_sha256=data_manifest_sha256)
    if {name:entry["episodes"] for name,entry in manifest["bank_inventory"].items()} != BANK_COUNTS:
        raise ValueError("fixed complete native bank inventory required")
    training = cached.load_manifest(ROOT/TRAINING, expected_manifest_sha256=TRAINING_SHA)
    if len(training["training"]) != INTERVAL:
        raise ValueError("full648-bundle learning stream required")
    pins = dict(input_pins or {})
    pins.update({relative_root(data_directory/"manifest.json"):data_manifest_sha256,
        relative_root(data_directory/"preparation.json"):digest(data_directory/"preparation.json"),
        TRAINING+"/manifest.json":TRAINING_SHA})
    resumed = None
    if resume_directory is None:
        pins.update({PRIOR+"/launch.json":PRIOR_LAUNCH, PRIOR+"/execution/summary.json":PRIOR_SUMMARY})
        verify_pins(pins)
        previous_launch, previous_summary = read(ROOT/PRIOR/"launch.json"), read(ROOT/PRIOR/"execution/summary.json")
        if previous_summary["status"] != "completed" or previous_summary["launch_sha256"] != PRIOR_LAUNCH:
            raise ValueError("completed producing tutor comparison required")
        parent = previous_summary["phase_commits"]["withdrawal"]["checkpoints"]["procedural"]
        if parent["sha256"] != PARENT_SHA:
            raise ValueError("declared procedural research parent required")
        expected_cursor = ORIGIN_CURSOR
    else:
        previous_directory = Path(resume_directory).resolve()
        if digest(previous_directory/"execution/summary.json") != resume_summary_sha256:
            raise ValueError("caller-pinned previous completed budget receipt required")
        previous_launch = read(previous_directory/"launch.json")
        previous_summary = read(previous_directory/"execution/summary.json")
        saved = previous_summary["last_commit"]
        if digest(ROOT/saved["path"]) != saved["sha256"]:
            raise ValueError("previous committed controller state changed")
        commit = read(ROOT/saved["path"])
        expected_cursor=validate_resume(previous_launch,previous_summary,commit,
            launch_sha256=digest(previous_directory/"launch.json"),
            data_manifest_sha256=data_manifest_sha256,current_sources=source_hashes())
        parent = commit["checkpoint"]
        pins.update({relative_root(previous_directory/"launch.json"):previous_summary["launch_sha256"],
            relative_root(previous_directory/"execution/summary.json"):resume_summary_sha256,
            saved["path"]:saved["sha256"]})
        resumed = dict(directory=relative_root(previous_directory), summary_sha256=resume_summary_sha256,
            commit=deepcopy(saved),prior_curriculum=deepcopy(commit["curriculum"]),
            expected_baseline_metrics={name:entry["metrics"] for name,entry in commit["scores"].items()})
    pins[parent["path"]] = parent["sha256"]
    verify_pins(pins); verify_pins(previous_launch["source_sha256"])
    raw = native(ROOT/parent["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != parent["sha256"]:
        raise ValueError("parent changed before metadata deserialization")
    import torch
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (payload["schema"] != "bic-shared-state-continuation-v1"
            or payload["lifetime_updates"] != expected_cursor
            or payload["learner"]["cursor"] != expected_cursor
            or payload["identity"]["origin"] != previous_launch["origin_identity"]):
        raise ValueError("raw bridge checkpoint and original identity must agree")
    sources = source_hashes()
    native(output).mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for index, (name,pin) in enumerate(sorted(sources.items())):
        image = native(ROOT/name).read_bytes()
        if hashlib.sha256(image).hexdigest() != pin:
            raise ValueError("source changed during freeze")
        local=f"sources/{index:03d}.bin"
        path=native(output/local);path.parent.mkdir(exist_ok=True)
        with path.open("xb") as stream: stream.write(image)
        snapshots[name]=local
    if source_hashes()!=sources: raise ValueError("source closure changed while freezing")
    launch=dict(schema=SCHEMA,created_utc=utc(),contract=spec,parent=deepcopy(parent),
        parent_cursor=expected_cursor,parent_weights_sha256=payload["weights_sha256"],
        parent_accounting=payload["learner"]["accounting"],
        origin_identity=deepcopy(previous_launch["origin_identity"]),
        expected_runtime=deepcopy(previous_launch.get("expected_runtime",previous_summary.get("runtime"))),
        curriculum=curriculum_state(expected_cursor),resumed=resumed,
        training_directory=TRAINING,training_manifest_sha256=TRAINING_SHA,
        data_directory=relative_root(data_directory),data_manifest_sha256=data_manifest_sha256,
        data_preparation_cost=read(data_directory/"preparation.json"),
        source_sha256=sources,source_snapshots=snapshots,input_sha256=pins,
        freeze_work=dict(checkpoint_metadata_loads=1,models=0,teacher_calls=0),
        freeze_cost=dict(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu))
    return publish(output/"launch.json",launch)


def run(directory, *, launch_sha256):
    started,cpu=time.monotonic(),time.process_time()
    directory=Path(directory).resolve()
    if digest(directory/"launch.json")!=launch_sha256: raise ValueError("caller-pinned launch required")
    launch=read(directory/"launch.json");spec=launch["contract"]
    if launch["schema"]!=SCHEMA or spec!=contract(spec["budget_seconds"]):
        raise ValueError("frozen budget contract differs")
    deadline=started+spec["budget_seconds"];training_stop=deadline-FINAL_RESERVE
    out=directory/"execution";native(out).mkdir(exist_ok=False)
    receipt=dict(schema=SCHEMA,status="running",launch_sha256=launch_sha256,started_utc=utc(),
        teacher_calls=0,automatic_retry=False,partial_work_unknown=False,new_updates_returned=0,
        artifact_sha256={},evaluations={},commits=[],restorations=[],operations={},archive_attempts=0,
        archive_completions=0,snapshot_attempts=0,snapshots=0)
    learner=owner=reader=journal=active_bank=active_ledger=None
    prepared_banks={};validation_work=None;last_evaluated=None;last_checkpoint_cursor=None

    def op(kind,identity,callback):
        if time.monotonic()>=deadline: raise TimeoutError("inclusive local worker budget expired")
        stats=receipt["operations"].setdefault(kind,dict(attempts=0,completed=0,failed=0,wall_seconds=0.,cpu_seconds=0.))
        tick,proc=time.monotonic(),time.process_time();stats["attempts"]+=1
        receipt["active_operation"]=dict(kind=kind,identity=identity)
        journal.event(dict(event="intent",kind=kind,identity=identity))
        try:
            value=callback();stats["completed"]+=1
            journal.event(dict(event="complete",kind=kind,identity=identity,wall_seconds=time.monotonic()-tick))
            receipt["active_operation"]=None;return value
        except BaseException:
            stats["failed"]+=1;raise
        finally:
            stats["wall_seconds"]+=time.monotonic()-tick;stats["cpu_seconds"]+=time.process_time()-proc

    def artifact(name,value,*,checkpoint=False):
        pin=op("publication",name,lambda:publish(out/name,value,checkpoint=checkpoint))
        receipt["artifact_sha256"][name]=pin
        return dict(path=relative_root(out/name),sha256=pin)

    def authenticate():
        verify_pins(launch["input_sha256"])
        if source_hashes()!=launch["source_sha256"]: raise ValueError("frozen execution source changed")
        for name,local in launch["source_snapshots"].items():
            if digest(directory/local)!=launch["source_sha256"][name]: raise ValueError("source image changed")
        if reader is not None: reader.authenticate_sources()

    def checkpoint_bytes(record):
        raw=native(ROOT/record["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record["sha256"]: raise ValueError("checkpoint image changed")
        return raw

    def restore(record,expected_cursor):
        nonlocal learner,owner
        raw=checkpoint_bytes(record)
        try:
            learner=bridge.SharedStateContinuation.from_snapshot(raw,expected_sha256=record["sha256"],
                expected_identity=launch["origin_identity"],device="cuda:0")
        except BaseException as exc:
            receipt["restorations"].append(dict(cursor=expected_cursor,report=getattr(exc,"continuation_report",None)))
            raise
        receipt["restorations"].append(dict(cursor=expected_cursor,report=learner.last_restore_report))
        if learner.cursor!=expected_cursor: raise ValueError("restored curriculum position differs")
        owner=PreparedLayoutOwner(config=config,layout="original",micro_batch_size=MICRO)

    def evaluate():
        nonlocal active_bank,active_ledger,last_evaluated
        cursor=learner.cursor;result={}
        for name,groups in prepared_banks.items():
            records=[];saved={}
            for turns,bank in groups.items():
                active_bank,active_ledger=bank,evaluation.EvaluationLedger()
                label=f"{cursor:08d}/{name}/t{turns}"
                scored=op("evaluation",label,lambda:bank.score(learner.model,batch_size=BATCH,
                    control="normal",deadline=deadline,work=active_ledger))
                records.extend(scored["raw_records"])
                saved[str(turns)]=artifact("scores/"+label+".json",scored)
                active_bank=active_ledger=None
            metrics=evaluation.score_records(records)
            if {k:v["total"] for k,v in metrics["overall"]["counts"].items()}!=bank_totals[name]:
                raise ValueError("native score denominator changed")
            result[name]=dict(metrics=metrics,groups=saved)
        receipt["evaluations"][str(cursor)]=result;last_evaluated=cursor

    def commit(reason,*,initial=False):
        nonlocal last_checkpoint_cursor
        if last_evaluated!=learner.cursor: evaluate()
        if initial: record=deepcopy(launch["parent"])
        else:
            receipt["snapshot_attempts"]+=1
            snapshot=op("snapshot",str(learner.cursor),learner.snapshot);receipt["snapshots"]+=1
            record=artifact(f"checkpoints/{learner.cursor:08d}.pt",snapshot,checkpoint=True)
            del snapshot
        value=dict(schema=SCHEMA,launch_sha256=launch_sha256,reason=reason,
            curriculum=curriculum_state(learner.cursor),checkpoint=record,
            data_manifest_sha256=launch["data_manifest_sha256"],training_manifest_sha256=TRAINING_SHA,
            evidence=learner.evidence,accounting=learner.accounting,
            scores=deepcopy(receipt["evaluations"][str(learner.cursor)]))
        saved=artifact(f"commits/{learner.cursor:08d}.json",value)
        receipt["commits"].append(saved);receipt["last_commit"]=saved
        last_checkpoint_cursor=learner.cursor
        return record

    try:
        journal=Journal(out/"events.jsonl");publish(out/"started.json",receipt)
        op("authentication","start",authenticate)
        import torch
        from brain_in_computer.benchmark import _peak_rss
        from brain_in_computer.dialogue_student import checkpoint_digest
        from experiments import execution_profile as execution
        from experiments import shared_state_continuation as bridge
        from experiments import foundation_layout_evaluation as evaluation
        from experiments.foundation_layout_curriculum import WorkLedger
        from experiments.foundation_layout_prepared import PreparedLayoutOwner
        from experiments.sequence_student import SequenceConfig
        from experiments.sustained_replay import ResidentReplay
        from experiments import shared_acquisition_transfer_data as data
        torch.set_num_threads(1);torch.set_num_interop_threads(1);execution.configure_strict_profile()
        receipt["runtime"]=execution.runtime_profile()
        if receipt["runtime"]!=launch["expected_runtime"]: raise ValueError("strict parent runtime changed")
        config=SequenceConfig(**CONFIG);validation_work=WorkLedger()
        data_directory=ROOT/launch["data_directory"]
        manifest=op("manifest","evaluation",lambda:data.load_manifest(data_directory,
            expected_manifest_sha256=launch["data_manifest_sha256"]))
        raw=native(data_directory/manifest["banks"]["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest["banks"]["sha256"]: raise ValueError("bank archive changed")
        receipt["archive_attempts"]+=1
        banks=op("archive","evaluation-banks",lambda:torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True))
        receipt["archive_completions"]+=1;del raw
        if set(banks)!=set(BANK_COUNTS): raise ValueError("all five native bank roles required")
        bank_totals={}
        for name,bank in banks.items():
            if len(bank["rows"])!=BANK_COUNTS[name]: raise ValueError("bank episode count differs")
            bank_totals[name]=denominators(bank["rows"]);groups=defaultdict(list)
            for row in bank["rows"]: groups[len(row["turns"])].append(row)
            prepared_banks[name]={turns:op("bank_preparation",f"{name}/{turns}",
                lambda rows=rows,role=bank["role"]:evaluation.PreparedLayoutBank(rows,role=role,
                    config=config,validation_work=validation_work)) for turns,rows in sorted(groups.items())}
        del banks
        reader=op("reader_setup","whole-curriculum",lambda:ResidentReplay(ROOT/TRAINING,
            expected_manifest_sha256=TRAINING_SHA,start_cursor=ORIGIN_CURSOR,max_archive_bytes=CACHE_BYTES))
        torch.cuda.reset_peak_memory_stats()
        op("restoration","baseline",lambda:restore(launch["parent"],launch["parent_cursor"]))
        if learner.accounting["lifetime_kernel"]!=launch["parent_accounting"]:
            raise ValueError("complete inherited work changed on restart")
        if checkpoint_digest(learner.model)!=launch["parent_weights_sha256"]:
            raise ValueError("restored parent weights differ")
        commit("restored-baseline",initial=True)
        if launch["resumed"] is not None:
            baseline={name:entry["metrics"] for name,entry in receipt["evaluations"][str(learner.cursor)].items()}
            if baseline!=launch["resumed"]["expected_baseline_metrics"]:
                raise ValueError("explicit resume baseline differs from committed native scores")
        while True:
            memory=_peak_rss();receipt["host_peak_rss"]=memory
            if memory["bytes"] is not None and memory["bytes"]>RESIDENT_LIMIT:
                receipt["stop_reason"]="resident_memory_guard";break
            if time.monotonic()>=training_stop:
                receipt["stop_reason"]="wall_budget_reserve";break
            cursor=learner.cursor
            token,labels,provenance=op("lesson_preparation",str(cursor),lambda:reader.prepare(cursor,owner))
            if time.monotonic()>=training_stop:
                receipt["unused_preparation_at_stop"]=dict(cursor=cursor,provenance=provenance)
                owner.close();del token,labels
                receipt["stop_reason"]="wall_budget_reserve";break
            report=op("training",str(cursor),lambda:learner.step(token,state_targets=labels,deadline=deadline))
            receipt["new_updates_returned"]+=report["physical_optimizer_updates"]
            journal.event(dict(event="step",curriculum=curriculum_state(learner.cursor),
                provenance=provenance,report=report))
            del token,labels,report
            if (learner.cursor-ORIGIN_CURSOR)%INTERVAL==0:
                record=commit("complete-curriculum-pass")
                print(encoded(dict(event="cycle",**curriculum_state(learner.cursor),
                    new_updates=receipt["new_updates_returned"],wall_seconds=time.monotonic()-started)).decode().strip(),flush=True)
                op("authentication",str(learner.cursor),authenticate)
                if time.monotonic()>=training_stop:
                    receipt["stop_reason"]="wall_budget_reserve";break
                expected=learner.cursor
                receipt["last_closed_state"]=dict(curriculum=curriculum_state(expected),
                    accounting=learner.accounting,last_training_report=learner.last_report)
                receipt.setdefault("closed_preparation",[]).append(owner.report())
                owner.close();owner=None;learner=None;gc.collect();torch.cuda.empty_cache()
                op("restoration",str(expected),lambda:restore(record,expected))
            elif receipt["new_updates_returned"]%108==0:
                print(encoded(dict(event="progress",**curriculum_state(learner.cursor),
                    new_updates=receipt["new_updates_returned"],wall_seconds=time.monotonic()-started)).decode().strip(),flush=True)
        if last_checkpoint_cursor!=learner.cursor: commit(receipt["stop_reason"])
        op("authentication","end",authenticate)
        receipt["status"]="completed"
    except BaseException as exc:
        receipt.update(status="failed",error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        latest=(dict(curriculum=curriculum_state(learner.cursor),accounting=learner.accounting,
                     last_training_report=learner.last_report) if learner is not None
                else receipt.get("last_closed_state"))
        if latest is not None:
            receipt.update(latest)
            actual=latest["accounting"]["lifetime_kernel"];origin=launch["parent_accounting"]
            receipt["physical_training"]={group:{key:value-origin[group][key] for key,value in actual[group].items()}
                for group in ("work","state_work")}
            if receipt["physical_training"]["work"]["unknown_optimizer_outcomes"]:
                receipt["partial_work_unknown"]=True
        if owner is not None:
            receipt["final_preparation"]=owner.report();owner.close()
        if reader is not None: receipt["replay_work"]=reader.report()
        if active_bank is not None:
            receipt["active_evaluation"]=dict(report=deepcopy(active_bank.last_report),work=active_ledger.report())
        if validation_work is not None: receipt["bank_validation_work"]=validation_work.report()
        if "_peak_rss" in locals(): receipt["host_peak_rss"]=_peak_rss()
        try:
            if "torch" in locals() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
                receipt.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
            if journal is not None: journal.close();receipt["journal_sha256"]=digest(out/"events.jsonl")
        except BaseException as exc:
            receipt.update(status="failed",cleanup_error=repr(exc),partial_work_unknown=True)
        receipt.update(ended_utc=utc(),wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
        publish(out/"summary.json",receipt)
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("freeze","run"));p.add_argument("directory")
    p.add_argument("--data-directory");p.add_argument("--data-manifest-sha256");p.add_argument("--input-pins")
    p.add_argument("--budget-seconds",type=int,default=1800);p.add_argument("--launch-sha256")
    p.add_argument("--resume-directory");p.add_argument("--resume-summary-sha256")
    args=p.parse_args()
    if args.mode=="freeze":
        pin=freeze(args.directory,data_directory=args.data_directory,data_manifest_sha256=args.data_manifest_sha256,
            budget_seconds=args.budget_seconds,input_pins=read(args.input_pins) if args.input_pins else None,
            resume_directory=args.resume_directory,resume_summary_sha256=args.resume_summary_sha256)
        print(encoded(dict(launch_sha256=pin)).decode().strip())
    else:
        result=run(args.directory,launch_sha256=args.launch_sha256)
        print(encoded(dict(status=result["status"],wall_seconds=result["wall_seconds"],
            stop_reason=result.get("stop_reason"))).decode().strip())
