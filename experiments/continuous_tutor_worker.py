"""One paired, source-pinned teaching/withdrawal cycle over a completed learner.

The CPU freeze authenticates metadata only. The exclusive worker restores the
public continuation bridge, trains matched curricula, saves raw native results,
and restarts both full AdamW branches before their common tutor-free withdrawal.
No author request, branch selection, or automatic promotion occurs here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import time
import traceback

from experiments.foundation_layout_study import (
    ROOT, CONFIG, Journal, native, digest, read, publish, relative_root,
    verify_pins, encoded, utc, denominators)
from experiments import continuous_tutor_policy as policy

SCHEMA = "bic-continuous-tutor-worker-v1"
ARMS, PHASES = ("procedural", "tutor"), ("teaching", "withdrawal")
UPDATES, MICRO, BATCH = 108, 32, 32
BANK_COUNTS = dict(dev=648, transfer_original=432, transfer_varied=432,
                   train_fit=108, retention=720)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def source_hashes():
    from experiments import shared_state_continuation as bridge
    from experiments import foundation_layout_evaluation as evaluation
    from experiments import shared_acquisition_transfer_data as banks
    from experiments import verified_tutor_cycle_data as data
    names = set(bridge.source_hashes()) | set(evaluation.source_hashes())
    names |= set(banks.source_hashes()) | set(data.source_hashes())
    names |= {"experiments/continuous_tutor_worker.py", "experiments/continuous_tutor_policy.py",
              "experiments/shared_state_targets.py", "experiments/foundation_layout_study.py",
              "experiments/execution_profile.py"}
    return {name: digest(ROOT/name) for name in sorted(names)}


def contract(max_seconds):
    if (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 600):
        raise ValueError("finite positive worker allowance at most600 seconds required")
    return dict(schema=SCHEMA, arms=list(ARMS), phases=list(PHASES),
        updates_per_phase_per_arm=UPDATES, new_updates=4*UPDATES,
        new_episode_exposures=4*UPDATES*MICRO*3, micro_batch_size=MICRO,
        evaluation_batch_size=BATCH, evaluation_steps=[0,UPDATES,2*UPDATES],
        bank_counts=BANK_COUNTS, config=CONFIG, max_seconds=max_seconds,
        identical_full_optimizer_parent=True, withdrawal_full_restarts=2,
        worker_teacher_calls=0, automatic_retry=False, automatic_promotion=False,
        scope="Paired compact curriculum and explicit withdrawal; repeated limited-grammar native diagnostics.")


def validate_inputs(manifest, evaluation_manifest, *, parent_checkpoint,
                    parent_metadata, origin_identity, parent_runtime, cycle_spec):
    """Pure crossbinding after caller authentication; no model or tensor work."""
    if _json(cycle_spec) != _json(policy.cycle_spec(cycle_spec["campaign_seed"], cycle_spec["cycle_index"])):
        raise ValueError("unchanged fixed campaign cycle specification required")
    if (set(parent_checkpoint) != {"path", "sha256"}
            or type(parent_checkpoint["sha256"]) is not str
            or len(parent_checkpoint["sha256"]) != 64):
        raise ValueError("caller-pinned parent checkpoint required")
    parent = manifest["parent"]
    expected_parent = dict(identity_sha256=parent_checkpoint["sha256"],
        weights_sha256=parent_metadata["weights_sha256"], cycle=cycle_spec["cycle_index"],
        lifetime_updates=parent_metadata["cursor"])
    if (manifest["schema"] != "bic-verified-tutor-cycle-data-v1" or parent != expected_parent
            or manifest["start_cursor"] != parent["lifetime_updates"]
            or manifest["micro_batch_size"] != MICRO or manifest["updates_per_phase"] != UPDATES
            or manifest["phase_seeds"] != {p:cycle_spec["seeds"][p] for p in PHASES}
            or manifest["seed"] != cycle_spec["seeds"]["teaching"]):
        raise ValueError("compiled lessons belong to another parent, cycle or curriculum")
    if (parent_metadata["origin"] != origin_identity or origin_identity["runtime"] != parent_runtime
            or _json(parent_metadata["recipe"]) != _json(origin_identity["recipe"])
            or origin_identity["config"] != CONFIG
            or origin_identity["learning_rate"] != cycle_spec["learning_rate"]
            or origin_identity["auxiliary_weight"] != cycle_spec["auxiliary_weight"]
            or origin_identity["architecture"] != cycle_spec["architecture"]):
        raise ValueError("unchanged inherited recipe, configuration and runtime required")
    recipe = parent_metadata["recipe"]
    if (recipe["layout"] != "original" or recipe["micro_batch_size"] != MICRO
            or recipe["objective_id"] != cycle_spec["objective_id"]):
        raise ValueError("original objective and complete-pair layout required")
    if set(manifest["phases"]) != set(PHASES):
        raise ValueError("complete matched phases required")
    start = parent["lifetime_updates"]
    for phase_index, phase in enumerate(PHASES):
        by_arm = manifest["phases"][phase]
        if set(by_arm) != set(ARMS):
            raise ValueError("both matched branches required")
        for records in by_arm.values():
            if len(records) != UPDATES or [r["cursor"] for r in records] != list(
                    range(start+phase_index*UPDATES, start+(phase_index+1)*UPDATES)):
                raise ValueError("exact contiguous phase cursors required")
    if manifest["phases"]["withdrawal"]["procedural"] != manifest["phases"]["withdrawal"]["tutor"]:
        raise ValueError("common withdrawal bytes and order required")
    if ({k:v["episodes"] for k,v in evaluation_manifest["bank_inventory"].items()} != BANK_COUNTS
            or set(evaluation_manifest["bank_inventory"]) != set(policy.ROLES)):
        raise ValueError("all five fixed native banks required")
    return deepcopy(expected_parent)


def parent_metadata(payload):
    """Extract only immutable metadata from authenticated own-envelope bytes."""
    saved = payload["learner"]
    if (payload["schema"] != "bic-shared-state-continuation-v1"
            or payload["identity"]["schema"] != payload["schema"]
            or saved["schema"] != "bic-shared-state-training-v1"
            or type(saved["cursor"]) is not int or saved["cursor"] < 1
            or saved["cursor"] != saved["evidence"]["cursor"]
            or saved["cursor"] != payload["lifetime_updates"]
            or payload["continuation_updates"] != saved["cursor"]-payload["identity"]["origin"]["step"]
            or saved["accounting"]["work"]["unknown_optimizer_outcomes"] != 0
            or saved["accounting"]["work"]["retained_updates"] != saved["cursor"]):
        raise ValueError("complete committed public continuation envelope required")
    # The bridge performs complete finite tensor/moment/layout and postload
    # equality checks later, before the first forward in the worker.
    return json.loads(_json(dict(origin=payload["identity"]["origin"],
        bridge_source_sha256=payload["identity"]["source_sha256"],
        cursor=saved["cursor"], weights_sha256=payload["weights_sha256"],
        recipe=saved["recipe"], evidence=saved["evidence"], accounting=saved["accounting"])))


def phase_work(before_accounting, after_accounting, before_evidence, after_evidence):
    """Exact policy projection from measured lifetime deltas, not planned counts."""
    before, after = before_accounting["work"], after_accounting["work"]
    result = {key:after[key]-before[key] for key in
              ("synchronized_optimizer_updates", "retained_episodes", "unknown_optimizer_outcomes")}
    result["family_episode_exposures"] = {f:after_evidence["exposures"][f]["episodes"]-
        before_evidence["exposures"][f]["episodes"] for f in policy.FAMILIES}
    if any(type(v) is not int or v < 0 for v in list(result.values())[:3]+
           list(result["family_episode_exposures"].values())):
        raise ValueError("nonnegative exact phase work deltas required")
    return result


def effective_deadline(started, max_seconds, deadline=None):
    contract(max_seconds)
    if deadline is not None and (type(deadline) not in (int,float) or not math.isfinite(deadline)):
        raise ValueError("finite absolute monotonic deadline required")
    return min(started+max_seconds,deadline if deadline is not None else math.inf)


def freeze(output, *, parent_checkpoint, origin_identity, parent_runtime,
           data_directory, data_manifest_sha256, evaluation_directory,
           evaluation_manifest_sha256, cycle_spec, max_seconds=600, input_pins=None):
    """Authenticate one metadata archive on CPU, then publish exclusive launch."""
    started, cpu = time.monotonic(), time.process_time()
    output, data_directory, evaluation_directory = map(lambda p:Path(p).resolve(),
        (output,data_directory,evaluation_directory))
    native(output).mkdir(parents=True,exist_ok=False)
    receipt = dict(schema=SCHEMA,status="running",checkpoint_load_attempts=0,checkpoint_loads=0,
                   models=0,optimizer_updates=0,teacher_calls=0)
    publish(output/"freeze-started.json",receipt)
    try:
        declared = contract(max_seconds)
        from experiments import verified_tutor_cycle_data as data
        from experiments import shared_acquisition_transfer_data as bank_data
        from experiments import shared_state_continuation as bridge
        manifest=data.load_manifest(data_directory,expected_manifest_sha256=data_manifest_sha256)
        bank_manifest=bank_data.load_manifest(evaluation_directory,expected_manifest_sha256=evaluation_manifest_sha256)
        pins=deepcopy(input_pins or {})
        pins.update({relative_root(data_directory/"manifest.json"):data_manifest_sha256,
            relative_root(data_directory/"preparation.json"):digest(data_directory/"preparation.json"),
            relative_root(evaluation_directory/"manifest.json"):evaluation_manifest_sha256,
            relative_root(evaluation_directory/"preparation.json"):digest(evaluation_directory/"preparation.json"),
            parent_checkpoint["path"]:parent_checkpoint["sha256"]})
        verify_pins(pins)
        raw=native(ROOT/parent_checkpoint["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=parent_checkpoint["sha256"]:
            raise ValueError("parent changed before authenticated deserialization")
        import torch
        receipt["checkpoint_load_attempts"]+=1
        payload=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
        receipt["checkpoint_loads"]+=1
        metadata=parent_metadata(payload);del payload,raw
        if metadata["bridge_source_sha256"] != bridge.source_hashes():
            raise ValueError("original continuation source closure changed")
        validate_inputs(manifest,bank_manifest,parent_checkpoint=parent_checkpoint,parent_metadata=metadata,
            origin_identity=origin_identity,parent_runtime=parent_runtime,cycle_spec=cycle_spec)
        verify_pins(metadata["recipe"]["source_sha256"])
        sources=source_hashes();snapshots={}
        for index,(name,pin) in enumerate(sorted(sources.items())):
            raw=native(ROOT/name).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError("source changed during freeze")
            local=f"sources/{index:03d}.bin";target=native(output/local);target.parent.mkdir(exist_ok=True)
            with target.open("xb") as stream:stream.write(raw)
            snapshots[name]=local
        if source_hashes()!=sources:raise ValueError("source closure changed during freeze")
        verify_pins(pins)
        launch=dict(schema=SCHEMA,contract=declared,created_utc=utc(),
            parent=deepcopy(parent_checkpoint),parent_cursor=metadata["cursor"],
            origin_identity=deepcopy(origin_identity),expected_runtime=deepcopy(parent_runtime),
            parent_weights_sha256=metadata["weights_sha256"],parent_accounting=metadata["accounting"],
            parent_evidence=metadata["evidence"],recipe_sha256=_hash(metadata["recipe"]),
            cycle_spec=deepcopy(cycle_spec),data_directory=relative_root(data_directory),
            data_manifest_sha256=data_manifest_sha256,evaluation_directory=relative_root(evaluation_directory),
            evaluation_manifest_sha256=evaluation_manifest_sha256,
            evaluation_inputs={k:v["rows_sha256"] for k,v in bank_manifest["bank_inventory"].items()},
            common_withdrawal_sha256=_hash(manifest["phases"]["withdrawal"]["procedural"]),
            teacher_decision=manifest["teacher_decision"],tutor_compilation=manifest["tutor_compilation"],
            treatment=manifest["treatment"],source_sha256=sources,source_snapshots=snapshots,input_sha256=pins)
        pin=publish(output/"launch.json",launch)
        receipt.update(status="completed",launch_sha256=pin)
        return pin
    except BaseException as error:
        receipt.update(status="failed",error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
        publish(output/"freeze-receipt.json",receipt)


def run(output, *, launch_sha256, deadline=None):
    started, cpu = time.monotonic(), time.process_time()
    output = Path(output).resolve()
    if digest(output/"launch.json") != launch_sha256:
        raise ValueError("caller-pinned launch required")
    launch = read(output/"launch.json")
    if launch["schema"] != SCHEMA or launch["contract"] != contract(launch["contract"]["max_seconds"]):
        raise ValueError("declared experiment contract differs")
    deadline=effective_deadline(started,launch["contract"]["max_seconds"],deadline)
    target = output/"execution"; native(target).mkdir(exist_ok=False)
    receipt = dict(schema=SCHEMA, status="running", launch_sha256=launch_sha256,
        started_utc=utc(), teacher_calls=0, automatic_retry=False,
        artifact_sha256={}, evaluations={}, phase_commits={}, restorations=[],
        archive_attempts=0, archive_loads=0, archive_bytes=0, snapshot_attempts=0, snapshots=0,
        phase_work={}, branches={}, continuations={},
        effective_deadline=deadline, evaluation_work={},
        target_work=dict(batch_attempts=0,batch_completions=0,row_attempts=0,row_completions=0,
            completed_english_checks=0,completed_typed_checks=0,completed_state_labels=0,
            partial_batch_internal_work_unknown=False),
        new_updates={arm:0 for arm in ARMS}, arms={}, operations={}, partial_work_unknown=False)
    learners, owners, prepared_banks = {}, {}, {}
    journal = active_bank = active_ledger = None
    validation_work = None
    phase_starts = {}

    def op(kind, identity, callback):
        if time.monotonic() >= deadline:
            raise TimeoutError("declared run allowance reached")
        stats = receipt["operations"].setdefault(kind, dict(attempts=0, completed=0,
            failed=0, wall_seconds=0., cpu_seconds=0.))
        tick, proc = time.monotonic(), time.process_time()
        stats["attempts"] += 1
        receipt["active_operation"] = dict(kind=kind, identity=identity)
        journal.event(dict(event="intent", kind=kind, identity=identity))
        try:
            result = callback(); stats["completed"] += 1
            journal.event(dict(event="complete",kind=kind,identity=identity,
                wall_seconds=time.monotonic()-tick))
            receipt["active_operation"] = None
            return result
        except BaseException:
            stats["failed"] += 1
            raise
        finally:
            stats["wall_seconds"] += time.monotonic()-tick
            stats["cpu_seconds"] += time.process_time()-proc

    def artifact(name, value, *, checkpoint=False):
        pin = op("publication", name, lambda: publish(target/name, value, checkpoint=checkpoint))
        receipt["artifact_sha256"][name] = pin
        return dict(path=relative_root(target/name), sha256=pin)

    def authenticate():
        verify_pins(launch["input_sha256"])
        if source_hashes() != launch["source_sha256"]:
            raise ValueError("frozen source closure changed")
        for name, local in launch["source_snapshots"].items():
            if digest(output/local) != launch["source_sha256"][name]:
                raise ValueError("frozen source image changed")

    def image(record, base):
        path = (base/record["path"]).resolve()
        if not path.is_relative_to(base) or path == base:
            raise ValueError("archive outside declared directory")
        raw = native(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("immutable archive changed")
        return raw

    def decode(record, base):
        raw = image(record, base)
        receipt["archive_attempts"] += 1
        receipt["archive_bytes"] += len(raw)
        value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        receipt["archive_loads"] += 1
        return value

    def restore(arm, raw, phase):
        try:
            learner = SharedStateContinuation.from_snapshot(raw,
                expected_sha256=hashlib.sha256(raw).hexdigest(),
                expected_identity=launch["origin_identity"], device="cuda:0")
        except BaseException as error:
            receipt["restorations"].append(dict(arm=arm, phase=phase,
                report=getattr(error,"continuation_report",dict(failed=True))))
            raise
        receipt["restorations"].append(dict(arm=arm,phase=phase,report=learner.last_restore_report))
        return learner

    def targets_for(lesson):
        packed={}
        work=receipt["target_work"]
        for family in policy.FAMILIES:
            rows=lesson["bundle"]["families"][family]
            work["batch_attempts"]+=1;work["row_attempts"]+=len(rows)
            try:
                packed[family]=pack_state_targets(rows)
            except BaseException:
                work["partial_batch_internal_work_unknown"]=True
                raise
            work["batch_completions"]+=1;work["row_completions"]+=len(rows)
            work["completed_english_checks"]+=len(rows);work["completed_typed_checks"]+=len(rows)
            work["completed_state_labels"]+=packed[family].numel()
        return packed

    def evaluate(arm, relative_step):
        nonlocal active_bank, active_ledger
        result = {}
        for bank_name, groups in prepared_banks.items():
            records, artifacts = [], {}
            for turns, bank in groups.items():
                active_bank, active_ledger = bank, evaluation.EvaluationLedger()
                name = f"{arm}/{relative_step:04d}/{bank_name}/t{turns}"
                scored = op("evaluation", name, lambda: bank.score(learners[arm].model,
                    batch_size=BATCH,control="normal",deadline=deadline,work=active_ledger))
                records.extend(scored["raw_records"])
                for key,value in active_ledger.report().items():
                    if type(value) is int:
                        receipt["evaluation_work"][key]=receipt["evaluation_work"].get(key,0)+value
                artifacts[str(turns)] = artifact("scores/"+name+".json",scored)
                active_bank = active_ledger = None
            metrics = evaluation.score_records(records)
            if {k:v["total"] for k,v in metrics["overall"]["counts"].items()} != bank_totals[bank_name]:
                raise ValueError("evaluation denominator changed")
            result[bank_name] = dict(metrics=metrics,groups=artifacts,raw_records_sha256=_hash(records))
        receipt["evaluations"].setdefault(arm,{})[str(relative_step)] = result

    def close_learners(phase):
        for arm, owner in owners.items():
            owner.close()
            receipt["arms"].setdefault(phase,{})[arm] = dict(cursor=learners[arm].cursor,
                accounting=learners[arm].accounting,preparation=owner.report())
        owners.clear(); learners.clear(); gc.collect()
        torch.cuda.empty_cache()

    try:
        journal = Journal(target/"events.jsonl")
        publish(target/"started.json",receipt)
        op("authentication","start",authenticate)
        import torch
        from experiments import execution_profile as execution
        from experiments import foundation_layout_evaluation as evaluation
        from experiments.foundation_layout_curriculum import WorkLedger
        from experiments.foundation_layout_prepared import PreparedLayoutOwner
        from experiments.sequence_student import SequenceConfig
        from experiments.shared_state_targets import pack_state_targets
        from experiments.shared_state_continuation import SharedStateContinuation
        from experiments import verified_tutor_cycle_data as data
        from experiments import shared_acquisition_transfer_data as bank_data
        from brain_in_computer.dialogue_student import checkpoint_digest
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        execution.configure_strict_profile()
        receipt["runtime"] = execution.runtime_profile()
        if receipt["runtime"] != launch["expected_runtime"]:
            raise ValueError("exact local strict runtime required")
        data_directory = ROOT/launch["data_directory"]
        manifest = op("manifest","compiled",lambda:data.load_manifest(data_directory,expected_manifest_sha256=launch["data_manifest_sha256"]))
        config,validation_work = SequenceConfig(**CONFIG),WorkLedger()
        evaluation_directory=ROOT/launch["evaluation_directory"]
        bank_manifest=op("manifest","evaluation",lambda:bank_data.load_manifest(evaluation_directory,
            expected_manifest_sha256=launch["evaluation_manifest_sha256"]))
        banks = op("archive","banks",lambda:decode(bank_manifest["banks"],evaluation_directory))
        if set(banks) != set(BANK_COUNTS):
            raise ValueError("fixed five bank roles required")
        bank_totals = {}
        for name, bank in banks.items():
            expected_role = "audit" if name.startswith("transfer_") else ("train_fit" if name=="train_fit" else "dev")
            if (len(bank["rows"]) != BANK_COUNTS[name] or bank["role"] != expected_role
                    or hashlib.sha256(encoded(bank["rows"]).rstrip(b"\n")).hexdigest()!=launch["evaluation_inputs"][name]):
                raise ValueError("bank role/count differs")
            bank_totals[name] = denominators(bank["rows"])
            groups = defaultdict(list)
            for row in bank["rows"]: groups[len(row["turns"])].append(row)
            prepared_banks[name] = {turns:op("bank_preparation",f"{name}/{turns}",
                lambda rows=rows,role=bank["role"]:evaluation.PreparedLayoutBank(rows,role=role,
                    config=config,validation_work=validation_work)) for turns,rows in sorted(groups.items())}
        del banks
        parent_bytes=op("archive_bytes","parent",lambda:image(launch["parent"],ROOT))
        parent_weights=launch["parent_weights_sha256"]
        torch.cuda.reset_peak_memory_stats()
        for phase_index, phase in enumerate(PHASES):
            for arm in ARMS:
                raw = parent_bytes if phase_index==0 else image(receipt["phase_commits"]["teaching"]["checkpoints"][arm],ROOT)
                learners[arm] = op("restoration",f"{phase}/{arm}",lambda:restore(arm,raw,phase))
                phase_starts[arm]=dict(accounting=deepcopy(learners[arm].accounting["lifetime_kernel"]),
                    evidence=learners[arm].evidence)
                owners[arm] = PreparedLayoutOwner(config=config,layout="original",micro_batch_size=MICRO)
                if learners[arm].cursor != manifest["start_cursor"]+phase_index*UPDATES:
                    raise ValueError("restored phase cursor differs")
                if phase_index==0 and checkpoint_digest(learners[arm].model)!=parent_weights:
                    raise ValueError("identical initial weights required")
                if _hash(learners[arm].recipe)!=launch["recipe_sha256"]:
                    raise ValueError("inherited optimizer/objective recipe changed")
                if phase_index==0 and learners[arm].evidence!=launch["parent_evidence"]:
                    raise ValueError("complete inherited evidence required")
                if phase_index==0 and learners[arm].accounting["lifetime_kernel"]!=launch["parent_accounting"]:
                    raise ValueError("complete inherited work and optimizer state required")
            del raw
            if phase_index==0:
                if learners["procedural"].evidence != learners["tutor"].evidence:
                    raise ValueError("identical original evidence required")
                for arm in ARMS: evaluate(arm,0)
                for name in prepared_banks:
                    left=receipt["evaluations"]["procedural"]["0"][name]
                    right=receipt["evaluations"]["tutor"]["0"][name]
                    if left["metrics"] != right["metrics"] or left["raw_records_sha256"] != right["raw_records_sha256"]:
                        raise ValueError("equal parent produced different baseline scores")
            for index in range(UPDATES):
                common = None
                if phase=="withdrawal":
                    common=op("archive",f"{phase}/{index}",lambda:decode(manifest["phases"][phase]["procedural"][index],data_directory))
                    targets=op("targets",f"{phase}/{index}",lambda:targets_for(common))
                for arm in ARMS if index%2==0 else ARMS[::-1]:
                    record=manifest["phases"][phase][arm][index]
                    lesson=common if common is not None else op("archive",f"{phase}/{arm}/{index}",lambda:decode(record,data_directory))
                    if common is None:
                        targets=op("targets",f"{phase}/{arm}/{index}",lambda:targets_for(lesson))
                    if (lesson["bundle"]["bundle_id"] != record["cursor"]
                            or lesson["expected_evidence"]["bundle_id"] != record["cursor"]
                            or learners[arm].cursor != record["cursor"]):
                        raise ValueError("authenticated lesson and learner cursors differ")
                    token=op("training_preparation",f"{phase}/{arm}/{index}",lambda:owners[arm].prepare(lesson["bundle"],
                        expected_evidence=lesson["expected_evidence"],expected_evidence_sha256=lesson["expected_evidence_sha256"]))
                    result=op("training",f"{phase}/{arm}/{index}",lambda:learners[arm].step(token,state_targets=targets,deadline=deadline))
                    receipt["new_updates"][arm]+=result["physical_optimizer_updates"]
                    journal.event(dict(event="step",phase=phase,arm=arm,report=result))
                    del lesson
                del targets,common
                if (index+1)%18==0: print(encoded(dict(event="progress",phase=phase,
                    updates_per_arm=phase_index*UPDATES+index+1,wall_seconds=time.monotonic()-started)).decode().strip(),flush=True)
            checkpoints={}
            for arm in ARMS:
                evaluate(arm,(phase_index+1)*UPDATES)
                receipt["snapshot_attempts"]+=1
                snapshot=op("snapshot",f"{phase}/{arm}",learners[arm].snapshot);receipt["snapshots"]+=1
                checkpoints[arm]=artifact(f"checkpoints/{phase}-{arm}.pt",snapshot,checkpoint=True)
                receipt["phase_work"].setdefault(phase,{})[arm]=phase_work(
                    phase_starts[arm]["accounting"],learners[arm].accounting["lifetime_kernel"],
                    phase_starts[arm]["evidence"],learners[arm].evidence)
                if phase=="withdrawal":
                    receipt["continuations"][arm]=dict(checkpoint=checkpoints[arm],
                        weights_sha256=snapshot["weights_sha256"],lifetime_updates=snapshot["lifetime_updates"],
                        metrics={k:v["metrics"] for k,v in receipt["evaluations"][arm][str(2*UPDATES)].items()},
                        scores=deepcopy(receipt["evaluations"][arm][str(2*UPDATES)]))
                del snapshot
            commit=dict(phase=phase,launch_sha256=launch_sha256,
                teacher_decision=launch["teacher_decision"],data_manifest_sha256=launch["data_manifest_sha256"],
                parent_checkpoint_sha256=launch["parent"]["sha256"],
                cycle_spec=launch["cycle_spec"],evaluation_manifest_sha256=launch["evaluation_manifest_sha256"],
                actual_work=receipt["phase_work"][phase],checkpoints=checkpoints,evidence={a:learners[a].evidence for a in ARMS},
                scores={a:deepcopy(receipt["evaluations"][a][str((phase_index+1)*UPDATES)]) for a in ARMS})
            artifact(f"{phase}-commit.json",commit);receipt["phase_commits"][phase]=commit
            close_learners(phase)
            op("authentication",phase,authenticate)
        if receipt["new_updates"] != {arm:2*UPDATES for arm in ARMS} or receipt["snapshots"]!=4 or len(receipt["restorations"])!=4:
            raise ValueError("complete matched physical workload required")
        for arm in ARMS:
            receipt["branches"][arm]=dict(status="completed",parent_checkpoint_sha256=launch["parent"]["sha256"],
                recipe_sha256=launch["recipe_sha256"],cycle_spec=deepcopy(launch["cycle_spec"]),
                evaluation_inputs=deepcopy(launch["evaluation_inputs"]),
                common_withdrawal_sha256=launch["common_withdrawal_sha256"],
                actual_work={phase:receipt["phase_work"][phase][arm] for phase in PHASES},
                teaching_metrics={k:receipt["evaluations"][arm][str(UPDATES)][k]["metrics"] for k in policy.CORE_ROLES},
                final_metrics=deepcopy(receipt["continuations"][arm]["metrics"]))
            # The public selector validates denominators and actual phase counts;
            # branch choice itself remains the outer campaign's responsibility.
        policy.select_continuation(receipt["branches"]["procedural"],receipt["branches"]["tutor"],
            expected_parent_identity=launch["parent"]["sha256"])
        receipt["status"]="completed"
    except BaseException as error:
        receipt.update(status="failed",error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if learners:
            for arm,learner in learners.items():
                receipt.setdefault("partial_arms",{})[arm]=dict(cursor=learner.cursor,
                    accounting=learner.accounting,last_report=learner.last_report,
                    preparation_before_close=owners[arm].report() if arm in owners else None)
        receipt["returned_step_updates"] = deepcopy(receipt["new_updates"])
        receipt["physical_training"] = {}
        for arm in ARMS:
            latest = receipt.get("partial_arms",{}).get(arm)
            if latest is None:
                for phase in reversed(PHASES):
                    if arm in receipt["arms"].get(phase,{}):
                        latest = receipt["arms"][phase][arm]
                        break
            if latest is not None:
                actual = latest["accounting"]["lifetime_kernel"]
                origin = launch["parent_accounting"]
                delta = {group:{key:value-origin[group][key] for key,value in actual[group].items()}
                         for group in ("work","state_work")}
                receipt["physical_training"][arm] = delta
                if delta["work"]["unknown_optimizer_outcomes"]:
                    receipt["partial_work_unknown"] = True
        # Lifetimes already contain earlier phases; latest-minus-parent counts
        # each actual attempt once, including updates followed by an error.
        for arm,owner in owners.items():
            owner.close()
            receipt.setdefault("partial_arms",{}).setdefault(arm,{})["preparation_after_close"]=owner.report()
        if active_bank is not None:
            receipt["active_evaluation"]=dict(report=deepcopy(active_bank.last_report),work=active_ledger.report())
        try:
            if "torch" in locals() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
                receipt["peak_gpu_allocated_bytes"]=torch.cuda.max_memory_allocated()
                receipt["peak_gpu_reserved_bytes"]=torch.cuda.max_memory_reserved()
            if journal is not None:journal.close();receipt["journal_sha256"]=digest(target/"events.jsonl")
        except BaseException as error:receipt.update(status="failed",cleanup_error=repr(error),partial_work_unknown=True)
        receipt["physical_totals"]={group:{key:sum(value[group][key] for value in receipt["physical_training"].values())
            for key in launch["parent_accounting"][group]} for group in ("work","state_work")}
        if receipt["status"]!="completed":
            receipt["branches"]={}
            receipt["continuations"]={}
        try:
            import psutil
            receipt["peak_process_rss_bytes"]=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)
        except Exception as error:
            receipt["peak_process_rss_bytes"]=None
            receipt["process_memory_error"]=repr(error)
        receipt.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
            bank_validation_work=validation_work.report() if validation_work else None,ended_utc=utc())
        publish(target/"summary.json",receipt)
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("freeze","run"));parser.add_argument("output")
    for name in ("parent-checkpoint","origin-identity","parent-runtime","data-directory",
                 "data-manifest-sha256","evaluation-directory","evaluation-manifest-sha256",
                 "cycle-spec","input-pins","launch-sha256"):
        parser.add_argument("--"+name)
    parser.add_argument("--max-seconds",type=float,default=600)
    parser.add_argument("--deadline",type=float)
    args=parser.parse_args()
    if args.mode=="freeze":
        pin=freeze(args.output,parent_checkpoint=read(args.parent_checkpoint),
            origin_identity=read(args.origin_identity),parent_runtime=read(args.parent_runtime),
            data_directory=args.data_directory,data_manifest_sha256=args.data_manifest_sha256,
            evaluation_directory=args.evaluation_directory,evaluation_manifest_sha256=args.evaluation_manifest_sha256,
            cycle_spec=read(args.cycle_spec),max_seconds=args.max_seconds,
            input_pins=read(args.input_pins) if args.input_pins else None)
        print(encoded(dict(launch_sha256=pin)).decode().strip())
    else:
        result=run(args.output,launch_sha256=args.launch_sha256,deadline=args.deadline)
        print(encoded(dict(status=result["status"],wall_seconds=result["wall_seconds"])).decode().strip())
