"""Matched verified teaching followed by a common, tutor-free continuation.

Curriculum authorship and compilation finish before this GPU worker begins.
Both branches start with the same complete learner and optimizer. The worker
has no transport API; teaching choices survive restart as pinned input data.
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

from experiments.foundation_layout_study import (
    ROOT, CONFIG, Journal, native, digest, read, publish, relative_root,
    verify_pins, encoded, utc, denominators)

SCHEMA = "bic-verified-tutor-run-v1"
DATA_SCHEMA = "bic-verified-tutor-data-v1"
ARMS = ("procedural", "tutor")
PHASES = ("teaching", "withdrawal")
UPDATES, MICRO, BATCH, SECONDS = 108, 32, 32, 1800
PROTOCOL = "docs/VERIFIED_TUTOR_PROTOCOL.md"


def source_hashes():
    from experiments import shared_state_continuation as bridge
    from experiments import foundation_layout_evaluation as evaluation
    from experiments import verified_tutor_curriculum_v2 as compiler
    from experiments import verified_tutor_author_v2 as author
    names = set(bridge.source_hashes()) | set(evaluation.source_hashes())
    names |= set(compiler.source_hashes()) | set(author.source_hashes())
    names |= {"experiments/verified_tutor_run.py", "experiments/verified_tutor_data.py",
              "experiments/verified_tutor_data_v2.py", "experiments/verified_tutor_data_v3.py",
              "docs/VERIFIED_TUTOR_RECOVERY.md", "docs/VERIFIED_TUTOR_COMPILATION_RECOVERY.md",
              "experiments/verified_tutor_report.py", "experiments/shared_state_screen.py",
              "experiments/shared_state_targets.py", "experiments/foundation_layout_study.py",
              "experiments/execution_profile.py", PROTOCOL}
    return {name: digest(ROOT/name) for name in sorted(names)}


def contract():
    return dict(schema=SCHEMA, arms=list(ARMS), phases=list(PHASES),
        updates_per_phase_per_arm=UPDATES, new_updates=UPDATES*4,
        new_episode_exposures=UPDATES*4*MICRO*3, micro_batch_size=MICRO,
        evaluation_batch_size=BATCH, config=CONFIG, max_seconds=SECONDS,
        identical_full_optimizer_parent=True, withdrawal_full_restarts=2,
        evaluation_steps=[0, UPDATES, 2*UPDATES], worker_teacher_calls=0,
        automatic_retry=False, automatic_promotion=False,
        scope="Single-parent authored-curriculum comparison and explicit withdrawal; limited English grammar.")


def data_header(directory, pin):
    directory = Path(directory).resolve()
    if digest(directory/"manifest.json") != pin:
        raise ValueError("caller-pinned compiled manifest required")
    manifest = read(directory/"manifest.json")
    if manifest["schema"] != DATA_SCHEMA or set(manifest["phases"]) != set(PHASES):
        raise ValueError("complete verified teaching/withdrawal manifest required")
    if manifest["micro_batch_size"] != MICRO:
        raise ValueError("fixed complete-pair microbatch size required")
    start = manifest["start_cursor"]
    if type(start) is not int or start < 1:
        raise ValueError("positive continuing learner cursor required")
    for phase_index, phase in enumerate(PHASES):
        by_arm = manifest["phases"][phase]
        if set(by_arm) != set(ARMS):
            raise ValueError("both matched curricula required")
        for arm, records in by_arm.items():
            if len(records) != UPDATES or [r["cursor"] for r in records] != list(
                    range(start+phase_index*UPDATES, start+(phase_index+1)*UPDATES)):
                raise ValueError("exact ordered continuation IDs required")
        if phase == "withdrawal" and by_arm["procedural"] != by_arm["tutor"]:
            raise ValueError("withdrawal must use the same compiled inventory and order")
    if manifest["bank_counts"] != dict(dev=720, train_fit=108, retention=720):
        raise ValueError("fixed reused native evaluation banks required")
    verify_pins(manifest["input_sha256"])
    for name, pin in manifest["artifact_sha256"].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or path == directory or digest(path) != pin:
            raise ValueError("compiled artifact identity differs")
    verify_pins(manifest["source_sha256"])
    return manifest


def freeze(output, *, data_directory, manifest_sha256, parent_directory,
           parent_launch_sha256, parent_summary_sha256, parent_arm, input_pins):
    """Freeze authenticated data and one completed parent; no neural work."""
    started, cpu = time.monotonic(), time.process_time()
    output, data_directory, parent_directory = map(lambda p: Path(p).resolve(),
        (output, data_directory, parent_directory))
    manifest = data_header(data_directory, manifest_sha256)
    preparation = read(data_directory/"preparation.json")
    if (preparation["status"] != "completed" or preparation["manifest_sha256"] != manifest_sha256
            or preparation["source_sha256"] != manifest["source_sha256"]):
        raise ValueError("completed compilation receipt required")
    pins = dict(input_pins)
    pins.update({relative_root(data_directory/"manifest.json"): manifest_sha256,
        relative_root(data_directory/"preparation.json"): digest(data_directory/"preparation.json"),
        relative_root(parent_directory/"launch.json"): parent_launch_sha256,
        relative_root(parent_directory/"execution/summary.json"): parent_summary_sha256})
    verify_pins(pins)
    parent_launch = read(parent_directory/"launch.json")
    parent_summary = read(parent_directory/"execution/summary.json")
    if (parent_launch["schema"] != "bic-shared-state-continuation-pilot-v1"
            or parent_summary["status"] != "completed"
            or parent_summary["launch_sha256"] != parent_launch_sha256
            or parent_arm not in ("fast", "slow") or manifest["start_cursor"] != 1944):
        raise ValueError("completed declared continuation parent required")
    verify_pins(parent_launch["source_sha256"])
    parent = parent_summary["checkpoints"][parent_arm]["1944"]
    pins[parent["path"]] = parent["sha256"]
    pins.update(manifest["input_sha256"])
    verify_pins(pins)
    from experiments import verified_tutor_author_v2 as author
    selected = manifest["teacher_decision"]
    decision = author.load_result(ROOT/selected["path"], expected_sha256=selected["sha256"],
        expected_request_sha256=selected["request_sha256"])
    if (decision["parent"]["identity_sha256"] != parent["sha256"]
            or decision["parent"]["lifetime_updates"] != 1944
            or any(decision["parent"][key] != manifest["parent"][key]
                   for key in ("identity_sha256", "weights_sha256", "lifetime_updates"))):
        raise ValueError("teacher request and compiled curriculum must bind this exact parent")
    if manifest["tutor_compilation"] != {key:decision[key] for key in ("contract_sha256","recipe_sha256")}:
        raise ValueError("compiled tutor lessons must implement the accepted recipe and contract")
    pins[selected["path"]] = selected["sha256"]
    sources = source_hashes()
    native(output).mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for index, (name, pin) in enumerate(sorted(sources.items())):
        raw = native(ROOT/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError("source changed while freezing")
        local = f"sources/{index:03d}.bin"
        target = native(output/local); target.parent.mkdir(exist_ok=True)
        with target.open("xb") as stream: stream.write(raw)
        snapshots[name] = local
    if source_hashes() != sources:
        raise ValueError("source closure changed during freeze")
    launch = dict(schema=SCHEMA, contract=contract(), created_utc=utc(),
        data_directory=relative_root(data_directory), data_manifest_sha256=manifest_sha256,
        parent=parent, parent_arm=parent_arm, parent_step=1944,
        parent_launch_sha256=parent_launch_sha256,
        origin_identity=parent_launch["origin_identities"][parent_arm],
        expected_runtime=parent_summary["runtime"],
        source_sha256=sources, source_snapshots=snapshots, input_sha256=pins,
        teacher_decision=manifest["teacher_decision"], treatment=manifest["treatment"],
        prior_author_attempt=manifest.get("prior_author_attempt"),
        prior_compilation=manifest.get("prior_compilation"),
        compilation_cost={key:deepcopy(preparation[key]) for key in
            ("wall_seconds","cpu_seconds","archive_attempts","archive_completions",
             "bytes_deserialized","compiler_attempts","compiler_completions","compiler_receipts")},
        inventory_preparation=manifest["inventory_preparation"],
        parent_weights_sha256=decision["parent"]["weights_sha256"],
        parent_accounting=parent_summary["arms"][parent_arm]["accounting"]["lifetime_kernel"],
        freeze_cost=dict(wall_seconds=time.monotonic()-started,
                         cpu_seconds=time.process_time()-cpu),
        freeze_work=dict(models=0, optimizer_updates=0, checkpoint_loads=0))
    return publish(output/"launch.json", launch)


def run(output, *, launch_sha256):
    started, cpu = time.monotonic(), time.process_time()
    output = Path(output).resolve()
    if digest(output/"launch.json") != launch_sha256:
        raise ValueError("caller-pinned launch required")
    launch = read(output/"launch.json")
    if launch["schema"] != SCHEMA or launch["contract"] != contract():
        raise ValueError("declared experiment contract differs")
    target = output/"execution"; native(target).mkdir(exist_ok=False)
    deadline = started+SECONDS
    receipt = dict(schema=SCHEMA, status="running", launch_sha256=launch_sha256,
        started_utc=utc(), teacher_calls=0, automatic_retry=False,
        artifact_sha256={}, evaluations={}, phase_commits={}, restorations=[],
        archive_attempts=0, archive_loads=0, snapshot_attempts=0, snapshots=0,
        new_updates={arm:0 for arm in ARMS}, arms={}, operations={}, partial_work_unknown=False)
    learners, owners, prepared_banks = {}, {}, {}
    journal = active_bank = active_ledger = None
    validation_work = None

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
                artifacts[str(turns)] = artifact("scores/"+name+".json",scored)
                active_bank = active_ledger = None
            metrics = evaluation.score_records(records)
            if {k:v["total"] for k,v in metrics["overall"]["counts"].items()} != bank_totals[bank_name]:
                raise ValueError("evaluation denominator changed")
            result[bank_name] = dict(metrics=metrics,groups=artifacts)
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
        from brain_in_computer.dialogue_student import checkpoint_digest
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        execution.configure_strict_profile()
        receipt["runtime"] = execution.runtime_profile()
        if receipt["runtime"] != launch["expected_runtime"]:
            raise ValueError("exact local strict runtime required")
        data_directory = ROOT/launch["data_directory"]
        manifest = op("manifest","compiled",lambda:data_header(data_directory,launch["data_manifest_sha256"]))
        config,validation_work = SequenceConfig(**CONFIG),WorkLedger()
        banks = op("archive","banks",lambda:decode(manifest["banks"],data_directory))
        if set(banks) != set(manifest["bank_counts"]):
            raise ValueError("fixed three bank roles required")
        bank_totals = {}
        for name, bank in banks.items():
            if len(bank["rows"]) != manifest["bank_counts"][name] or bank["role"] != ("train_fit" if name=="train_fit" else "dev"):
                raise ValueError("bank role/count differs")
            bank_totals[name] = denominators(bank["rows"])
            groups = defaultdict(list)
            for row in bank["rows"]: groups[len(row["turns"])].append(row)
            prepared_banks[name] = {turns:op("bank_preparation",f"{name}/{turns}",
                lambda rows=rows,role=bank["role"]:evaluation.PreparedLayoutBank(rows,role=role,
                    config=config,validation_work=validation_work)) for turns,rows in sorted(groups.items())}
        del banks
        parent = op("archive","parent",lambda:decode(launch["parent"],ROOT))
        if (parent["schema"] != "bic-shared-state-continuation-pilot-v1"
                or parent["launch_sha256"] != launch["parent_launch_sha256"]
                or parent["arm"] != launch["parent_arm"] or parent["step"] != launch["parent_step"]
                or parent["runtime"] != receipt["runtime"]
                or parent["learner"]["lifetime_updates"] != manifest["start_cursor"]
                or parent["weights_sha256"] != parent["learner"]["weights_sha256"]
                or parent["weights_sha256"] != launch["parent_weights_sha256"]):
            raise ValueError("authenticated parent envelope disagrees")
        buffer = io.BytesIO();torch.save(parent["learner"],buffer);parent_bytes=buffer.getvalue()
        parent_weights = parent["weights_sha256"]; del parent,buffer
        torch.cuda.reset_peak_memory_stats()
        for phase_index, phase in enumerate(PHASES):
            for arm in ARMS:
                raw = parent_bytes if phase_index==0 else image(receipt["phase_commits"]["teaching"]["checkpoints"][arm],ROOT)
                learners[arm] = op("restoration",f"{phase}/{arm}",lambda:restore(arm,raw,phase))
                owners[arm] = PreparedLayoutOwner(config=config,layout="original",micro_batch_size=MICRO)
                if learners[arm].cursor != manifest["start_cursor"]+phase_index*UPDATES:
                    raise ValueError("restored phase cursor differs")
                if phase_index==0 and checkpoint_digest(learners[arm].model)!=parent_weights:
                    raise ValueError("identical initial weights required")
                if phase_index==0 and learners[arm].accounting["lifetime_kernel"]!=launch["parent_accounting"]:
                    raise ValueError("complete inherited work and optimizer state required")
            del raw
            if phase_index==0:
                if learners["procedural"].evidence != learners["tutor"].evidence:
                    raise ValueError("identical original evidence required")
                for arm in ARMS: evaluate(arm,0)
                for name in prepared_banks:
                    if receipt["evaluations"]["procedural"]["0"][name]["metrics"] != receipt["evaluations"]["tutor"]["0"][name]["metrics"]:
                        raise ValueError("equal parent produced different baseline scores")
            for index in range(UPDATES):
                common = None
                if phase=="withdrawal":
                    common=op("archive",f"{phase}/{index}",lambda:decode(manifest["phases"][phase]["procedural"][index],data_directory))
                    targets=op("targets",f"{phase}/{index}",lambda:{f:pack_state_targets(rows) for f,rows in common["bundle"]["families"].items()})
                for arm in ARMS if index%2==0 else ARMS[::-1]:
                    record=manifest["phases"][phase][arm][index]
                    lesson=common if common is not None else op("archive",f"{phase}/{arm}/{index}",lambda:decode(record,data_directory))
                    if common is None:
                        targets=op("targets",f"{phase}/{arm}/{index}",lambda:{f:pack_state_targets(rows) for f,rows in lesson["bundle"]["families"].items()})
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
                del snapshot
            commit=dict(phase=phase,launch_sha256=launch_sha256,
                teacher_decision=launch["teacher_decision"],data_manifest_sha256=launch["data_manifest_sha256"],
                checkpoints=checkpoints,evidence={a:learners[a].evidence for a in ARMS},
                scores={a:deepcopy(receipt["evaluations"][a][str((phase_index+1)*UPDATES)]) for a in ARMS})
            artifact(f"{phase}-commit.json",commit);receipt["phase_commits"][phase]=commit
            close_learners(phase)
            op("authentication",phase,authenticate)
        if receipt["new_updates"] != {arm:2*UPDATES for arm in ARMS} or receipt["snapshots"]!=4 or len(receipt["restorations"])!=4:
            raise ValueError("complete matched physical workload required")
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
        for owner in owners.values(): owner.close()
        if active_bank is not None:
            receipt["active_evaluation"]=dict(report=deepcopy(active_bank.last_report),work=active_ledger.report())
        try:
            if "torch" in locals() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
                receipt["peak_gpu_allocated_bytes"]=torch.cuda.max_memory_allocated()
                receipt["peak_gpu_reserved_bytes"]=torch.cuda.max_memory_reserved()
            if journal is not None:journal.close();receipt["journal_sha256"]=digest(target/"events.jsonl")
        except BaseException as error:receipt.update(status="failed",cleanup_error=repr(error),partial_work_unknown=True)
        receipt.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
            bank_validation_work=validation_work.report() if validation_work else None,ended_utc=utc())
        publish(target/"summary.json",receipt)
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("freeze","run"));parser.add_argument("output")
    for name in ("data-directory","manifest-sha256","parent-directory","parent-launch-sha256",
                 "parent-summary-sha256","parent-arm","input-pins","launch-sha256"):
        parser.add_argument("--"+name)
    args=parser.parse_args()
    if args.mode=="freeze":
        pin=freeze(args.output,data_directory=args.data_directory,manifest_sha256=args.manifest_sha256,
            parent_directory=args.parent_directory,parent_launch_sha256=args.parent_launch_sha256,
            parent_summary_sha256=args.parent_summary_sha256,parent_arm=args.parent_arm,input_pins=read(args.input_pins))
        print(encoded(dict(launch_sha256=pin)).decode().strip())
    else:
        result=run(args.output,launch_sha256=args.launch_sha256)
        print(encoded(dict(status=result["status"],wall_seconds=result["wall_seconds"])).decode().strip())
