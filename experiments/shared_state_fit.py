"""Fixed-buffer trainability diagnostic; four fresh arms, no promotion/retry."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import io
import math
import os
from pathlib import Path
import time
import traceback

from experiments.foundation_layout_study import (ROOT, CONFIG, native, digest, read, publish,
    relative_root, verify_pins, encoded, _hash, utc, Journal as BaseJournal)
from experiments.shared_state_pilot import installed_runtime

SCHEMA = "bic-shared-state-fit-v1"
PROTOCOL = "docs/SHARED_STATE_FIT_PROTOCOL.md"
PRIOR = "runs/shared-state-pilot-local/attempt-001"
PRIOR_LAUNCH = "100f899ec4d42748fe2c4a7bb89a79415af695759f11c2c3f6fe117fba37d0e5"
PRIOR_SUMMARY = "4dd146676f035e409c1befc6b963767e926a4657e83ad7263a57040a961dbb04"
DATA = "runs/shared-state-data-local/attempt-001"
MANIFEST_SHA = "266dfe643074b8d66b65b1e506f6fc726fa6cb75f2a532b04232ec87bf30eb75"
BANK_SHA = "c42c117d04070cae026e6372615bcb044b493ff41f80603a268dc8aff93e4efb"
ARMS = {"joint_fast": (.003, "joint"), "joint_slow": (.0003, "joint"),
        "state_fast": (.003, "state"), "state_slow": (.0003, "state")}
FAMILIES, LENGTHS, ENDPOINTS = ("color", "count", "switch"), (8, 10, 12), (0, 216, 432)
SEED, UPDATES, SECONDS = 852404001, 432, 900


def contract():
    return dict(schema=SCHEMA, seed=SEED, config=CONFIG, arms={name:dict(learning_rate=lr,objective=objective)
        for name,(lr,objective) in ARMS.items()}, updates_per_arm=UPDATES, family_batch_size=12,
        family_order=list(FAMILIES), length_order=list(LENGTHS), endpoints=list(ENDPOINTS),
        auxiliary_weight=.3, gradient_clip=1., optimizer="AdamW defaults except learning rate",
        arm_order="Rotate first arm by zero-based update modulo four; every arm receives the same length group",
        objective_schema="joint=(sequence_objective+.3state_loss)/3; state=(.3state_loss)/3 per family",
        policy_forward="Identical teacher-forced native forward in all training arms; state labels never inputs",
        device="cuda:0", max_seconds=SECONDS, cpu_threads=1, cpu_interop_threads=1,
        expected_work=dict(models=4, training_forwards=5184, training_backwards=5184,
            training_episode_exposures=62208, optimizer_updates=1728, snapshots=8,
            auxiliary_evaluation_forwards=108, auxiliary_evaluation_episodes=1296,
            native_evaluation_forwards=108, native_evaluation_episodes=1296, target_packs=9,target_rows=108),
        fitting_thresholds=dict(known_sensitivity=.95,unknown_specificity=.95,conditional_known_value=.9,flat_known_exact=.8),
        readouts="Unmasked106-class conditional argmax; true-value rank1+strictlygreater; argmax tieslowestindex; knownness P>.5",
        automatic_retry=False, automatic_promotion=False, teacher_calls=0,
        scope="Trainability on108 actual training rows; possible memorization, no transfer or comprehension claim")


def source_pins():
    from experiments.shared_state_training import source_hashes
    from experiments.foundation_layout_evaluation import source_hashes as evaluation_sources
    names = set(source_hashes()) | set(evaluation_sources()) | {"experiments/shared_state_fit.py",
        "experiments/shared_state_pilot.py", "experiments/foundation_layout_study.py", "experiments/execution_profile.py", PROTOCOL}
    if native(ROOT/"experiments/__init__.py").exists(): raise ValueError("namespace initializer must remain absent")
    return {name:digest(ROOT/name) for name in sorted(names)}


def freeze(output, *, input_pins=None):
    started, cpu = time.monotonic(), time.process_time()
    sources = source_pins()
    pins = {**(input_pins or {}), PRIOR+"/launch.json":PRIOR_LAUNCH,
            PRIOR+"/execution/summary.json":PRIOR_SUMMARY, DATA+"/manifest.json":MANIFEST_SHA, DATA+"/banks.pt":BANK_SHA}
    verify_pins(pins)
    previous, summary, manifest = read(ROOT/PRIOR/"launch.json"), read(ROOT/PRIOR/"execution/summary.json"), read(ROOT/DATA/"manifest.json")
    if (summary["status"] != "completed" or summary["launch_sha256"] != PRIOR_LAUNCH
            or previous["data_manifest_sha256"] != MANIFEST_SHA or previous["contract"]["config"] != CONFIG
            or manifest["banks"] != dict(path="banks.pt",sha256=BANK_SHA)
            or manifest["bank_inventory"]["train_fit"]["episodes"] != 108):
        raise ValueError("completed shared-state source and exact108-row training inventory required")
    verify_pins(previous["source_sha256"])
    output = Path(output).resolve(); native(output).mkdir(parents=True,exist_ok=False)
    snapshots = {}
    for index,(name,pin) in enumerate(sorted(sources.items())):
        image = native(ROOT/name).read_bytes()
        if hashlib.sha256(image).hexdigest() != pin: raise ValueError("source changed during freeze")
        local = f"sources/{index:03d}.bin"; target = native(output/local); target.parent.mkdir(exist_ok=True)
        with target.open("xb") as stream: stream.write(image); stream.flush(); os.fsync(stream.fileno())
        snapshots[name] = local
    if source_pins() != sources: raise ValueError("source closure changed during freeze")
    launch = dict(schema=SCHEMA,contract=contract(),source_sha256=sources,source_snapshots=snapshots,
        input_sha256=pins,installed_runtime=installed_runtime(),expected_runtime=summary["runtime"],
        fit_inventory=manifest["bank_inventory"]["train_fit"],created_utc=utc(),
        freeze_cost=dict(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu))
    return publish(output/"launch.json",launch)


def state_metrics(groups):
    """Pure recount of saved state predictions/probabilities; no neural imports."""
    pooled = defaultdict(list)
    for group in groups.values():
        keys = ("targets","predictions","conditional_predictions","true_value_ranks","known_probability")
        arrays = [group[key] for key in keys]
        if len({len(value) for value in arrays}) != 1: raise ValueError("raw state row counts differ")
        for rows in zip(*arrays):
            if len({len(value) for value in rows}) != 1: raise ValueError("raw state turn counts differ")
            for turns in zip(*rows):
                if any(len(value) != 12 for value in turns): raise ValueError("all twelve aliases required")
                for target,prediction,conditional,rank,probability in zip(*turns):
                    if (any(type(v) is not int for v in (target,prediction,conditional,rank))
                            or not 0 <= target <107 or not 0 <= prediction <107 or not 1 <= conditional <107
                            or (not 1 <= rank <=106 if target else rank !=0)
                            or type(probability) not in (int,float) or not math.isfinite(probability) or not 0 <= probability <=1):
                        raise ValueError("raw typed state values differ")
                    pooled[group["family"]].append((target,prediction,conditional,rank,probability))
    if set(pooled) != set(FAMILIES): raise ValueError("complete family state readout required")
    def calculate(rows):
        known = [row for row in rows if row[0]]; unknown = [row for row in rows if not row[0]]
        ratio = lambda count,total: dict(count=count,total=total,rate=count/total if total else None)
        tp,fp = sum(row[4]>.5 for row in known),sum(row[4]>.5 for row in unknown)
        return dict(states=len(rows),known_exact=ratio(sum(t==p for t,p,_,_,_ in known),len(known)),
            unknown_exact=ratio(sum(p==0 for _,p,_,_,_ in unknown),len(unknown)),
            conditional_known_value=ratio(sum(t==c for t,_,c,_,_ in known),len(known)),
            known_sensitivity=ratio(tp,len(known)),unknown_specificity=ratio(len(unknown)-fp,len(unknown)),
            knownness_confusion=dict(true_positive=tp,false_negative=len(known)-tp,false_positive=fp,true_negative=len(unknown)-fp),
            mean_true_value_rank=sum(row[3] for row in known)/len(known) if known else None,
            mean_known_probability=sum(row[4] for row in known)/len(known) if known else None,
            mean_unknown_probability_of_known=sum(row[4] for row in unknown)/len(unknown) if unknown else None,
            brier=sum((p-int(bool(t)))**2 for t,_,_,_,p in rows)/len(rows))
    return dict(overall=calculate([row for family in FAMILIES for row in pooled[family]]),
                by_family={family:calculate(pooled[family]) for family in FAMILIES})


def adequate_fitting(metrics):
    thresholds = dict(known_sensitivity=(95,100),unknown_specificity=(95,100),conditional_known_value=(9,10),known_exact=(4,5))
    checks = {family+"/"+key: row[key]["total"]>0 and row[key]["count"]*den>=num*row[key]["total"]
              for family,row in metrics["by_family"].items() for key,(num,den) in thresholds.items()}
    return dict(passed=all(checks.values()),checks=checks,scope="Fitting diagnostic only; no generalization or promotion")


class Journal(BaseJournal):
    def __init__(self,path):
        super().__init__(path); self.chain = _hash([SCHEMA,"events"])


def run(output, *, expected_launch_sha256):
    started,cpu = time.monotonic(),time.process_time(); deadline=started+SECONDS
    output=Path(output).resolve()
    if digest(output/"launch.json") != expected_launch_sha256: raise ValueError("explicit launch pin differs")
    launch=read(output/"launch.json")
    if launch["schema"] != SCHEMA or launch["contract"] != contract(): raise ValueError("fixed fit contract differs")
    attempt=output/"execution"; native(attempt).mkdir(exist_ok=False)
    receipt=dict(schema=SCHEMA,status="running",launch_sha256=expected_launch_sha256,started_utc=utc(),pid=os.getpid(),
        evaluations={},checkpoints={},artifact_sha256={},arms={},native_evaluation_work={},teacher_calls=0,automatic_retry=False,automatic_promotion=False,
        partial_work_unknown=False,work=dict(model_attempts=0,models=0,target_packs=0,target_rows=0,training_forward_attempts=0,
        training_forwards=0,training_episode_exposures=0,training_state_readout_attempts=0,training_state_readouts=0,
        backward_attempts=0,backwards=0,optimizer_attempts=0,
        optimizer_returns=0,synchronized_updates=0,unknown_optimizer_outcomes=0,aux_evaluation_attempts=0,
        aux_evaluation_forwards=0,aux_evaluation_episodes=0,aux_evaluation_readout_attempts=0,aux_evaluation_readouts=0,snapshot_attempts=0,snapshots=0))
    models,optimizers,cursors,prepared,cpu_batches,gpu_batches,labels={}, {}, {}, {}, {}, {}, {}
    operations,journal,validation,active_bank,active_ledger={},None,None,None,None
    active_step, tensor_pins = None,None

    def boundary():
        if time.monotonic() >=deadline: raise TimeoutError("fixed900-second allowance expired; no automatic retry")
    def event(value):
        try: journal.event(value)
        except BaseException: receipt["partial_work_unknown"]=True; raise
    def op(kind,identity,call):
        boundary(); stats=operations.setdefault(kind,dict(attempts=0,completions=0,failures=0,wall_seconds=0.,cpu_seconds=0.))
        stats["attempts"]+=1; t,c=time.monotonic(),time.process_time(); receipt["active_operation"]=dict(kind=kind,identity=identity)
        try:
            event(dict(event="intent",kind=kind,identity=identity)); value=call(); stats["completions"]+=1
            event(dict(event="complete",kind=kind,identity=identity,wall_seconds=time.monotonic()-t,cpu_seconds=time.process_time()-c))
            receipt["active_operation"]=None; return value
        except BaseException: stats["failures"]+=1; raise
        finally: stats["wall_seconds"]+=time.monotonic()-t; stats["cpu_seconds"]+=time.process_time()-c
    def artifact(name,value,checkpoint=False):
        pin=op("publication",name,lambda:publish(attempt/name,value,checkpoint=checkpoint)); receipt["artifact_sha256"][name]=pin
        return dict(path=relative_root(attempt/name),sha256=pin)
    def authenticate():
        if source_pins()!=launch["source_sha256"] or installed_runtime()!=launch["installed_runtime"]: raise ValueError("source/runtime drift")
        verify_pins(launch["input_sha256"])
        for name,local in launch["source_snapshots"].items():
            if digest(output/local)!=launch["source_sha256"][name]: raise ValueError("source snapshot differs")
    def tensor_digest(tree):
        if isinstance(tree,torch.Tensor):
            return [str(tree.dtype),str(tree.device),list(tree.shape),list(tree.stride()),tree.storage_offset(),
                    hashlib.sha256(tree.detach().cpu().contiguous().numpy().tobytes()).hexdigest()]
        return {str(key):tensor_digest(value) for key,value in tree.items()}
    def data_guard():
        current=_hash(dict(cpu=tensor_digest(cpu_batches),gpu=tensor_digest(gpu_batches),labels=tensor_digest(labels)))
        if current!=tensor_pins: raise ValueError("cached training tensors changed")
    def model_guard(arm):
        model=models[arm]
        if (type(model) is not SharedStateStudent or model.config!=config or any(p.device!=torch.device("cuda:0")
                or p.dtype!=torch.float32 or not p.requires_grad for p in model.parameters())): raise ValueError("fixed model differs")
        if optimizers[arm].param_groups[0]["lr"]!=ARMS[arm][0]: raise ValueError("fixed optimizer rate changed")
        execution.assert_strict_profile()

    def evaluate(arm,step):
        nonlocal active_bank,active_ledger
        model_guard(arm); data_guard(); model=models[arm]; modes=[(module,module.training) for module in model.modules()]
        before=evaluation._state(model); raw={}
        try:
            model.eval()
            with torch.inference_mode():
                for family in FAMILIES:
                    for turns in LENGTHS:
                        key=f"{family}/{turns}"; batch=gpu_batches[key]; target=labels[key]
                        def auxiliary():
                            work=receipt["work"]; work["aux_evaluation_attempts"]+=1
                            output=model(**batch["inputs"],decoder_input_ids=torch.full((12,turns,1),ByteCodec.BOS,dtype=torch.long,device="cuda:0"))
                            work["aux_evaluation_forwards"]+=1; work["aux_evaluation_episodes"]+=12
                            work["aux_evaluation_readout_attempts"]+=1
                            logits=model.state_logits(output["context_states"],batch["inputs"]["eos_positions"])
                            work["aux_evaluation_readouts"]+=1
                            if not bool(torch.isfinite(logits).all()): raise ValueError("nonfinite state logits")
                            values=logits[...,1:]; wanted=(target-1).clamp_min(0); true=values.gather(-1,wanted[...,None])
                            ranks=1+(values>true).sum(-1)
                            ranks=ranks.masked_fill(target.eq(0),0)
                            result=dict(family=family,turns=turns,episode_ids=group_ids[key],targets=target.cpu().tolist(),
                                predictions=logits.argmax(-1).cpu().tolist(),conditional_predictions=(values.argmax(-1)+1).cpu().tolist(),
                                true_value_ranks=ranks.cpu().tolist(),known_probability=(1-logits.softmax(-1)[...,0]).cpu().tolist(),
                                balanced_state_loss=float(state_loss(logits,target)))
                            torch.cuda.synchronize(); return result
                        raw[key]=op("auxiliary_evaluation",f"{arm}/{step}/{key}",auxiliary)
        finally:
            for module,mode in modes: module.training=mode
            if evaluation._state(model)!=before: raise RuntimeError("auxiliary evaluation changed learner state")
        state=state_metrics(raw); state_artifact=artifact(f"scores/{arm}/{step:04d}/state.json",dict(raw=raw,metrics=state))
        records,groups=[],{}
        for key,bank in prepared.items():
            active_bank,active_ledger=bank,evaluation.EvaluationLedger()
            result=op("native_evaluation",f"{arm}/{step}/{key}",lambda:bank.score(model,batch_size=12,deadline=deadline,work=active_ledger))
            for name,value in active_ledger.counts.items():receipt["native_evaluation_work"][name]=receipt["native_evaluation_work"].get(name,0)+value
            records.extend(result["raw_records"]); groups[key]=artifact(f"scores/{arm}/{step:04d}/native/{key}.json",result)
            active_bank=active_ledger=None
        native_metrics=evaluation.score_records(records)
        data_guard(); model_guard(arm)
        receipt["evaluations"].setdefault(arm,{})[str(step)]=dict(state=state,native=native_metrics,state_artifact=state_artifact,native_artifacts=groups,
            adequate_fitting=adequate_fitting(state),native_scope="State-only native heads receive no direct training; descriptive scores only")
        artifact(f"scores/{arm}/{step:04d}/summary.json",receipt["evaluations"][arm][str(step)])

    def step(arm,turns):
        nonlocal active_step
        model_guard(arm); model,optimizer=models[arm],optimizers[arm]; model.train(); optimizer.zero_grad(set_to_none=True)
        report=dict(arm=arm,cursor_before=cursors[arm],turns=turns,objective=ARMS[arm][1],failed=True,physical_optimizer_updates=0,microbatches=[])
        active_step=report; work=receipt["work"]
        try:
            for family in FAMILIES:
                batch=gpu_batches[f"{family}/{turns}"]; target=labels[f"{family}/{turns}"]
                work["training_forward_attempts"]+=1
                output=model(**batch["inputs"],decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
                work["training_forwards"]+=1; work["training_episode_exposures"]+=12
                original=sequence_objective(output,batch)
                work["training_state_readout_attempts"]+=1
                state_logits=model.state_logits(output["context_states"],batch["inputs"]["eos_positions"])
                work["training_state_readouts"]+=1
                state=state_loss(state_logits,target)
                total=original["loss"]+.3*state if ARMS[arm][1]=="joint" else .3*state
                if not all(bool(torch.isfinite(v)) for v in [*original.values(),state,total]): raise ValueError("nonfinite training objective")
                work["backward_attempts"]+=1; (total/3).backward(); work["backwards"]+=1
                report["microbatches"].append(dict(family=family,**{name:float(value.detach()) for name,value in original.items()},
                    state_loss=float(state.detach()),total_loss=float(total.detach())))
                del output,original,state,total,state_logits
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            work["optimizer_attempts"]+=1; report["physical_optimizer_updates"]=None
            optimizer.step(); work["optimizer_returns"]+=1; torch.cuda.synchronize()
            work["synchronized_updates"]+=1; report["physical_optimizer_updates"]=1
            cursors[arm]+=1; report.update(failed=False,cursor=cursors[arm]); return report
        finally:
            if report["physical_optimizer_updates"] is None: work["unknown_optimizer_outcomes"]+=1
            receipt["arms"][arm]=dict(cursor=cursors[arm],last_report=deepcopy(report))

    def checkpoint(arm,step):
        model_guard(arm); receipt["work"]["snapshot_attempts"]+=1; torch.cuda.synchronize()
        payload=dict(schema=SCHEMA,launch_sha256=expected_launch_sha256,arm=arm,step=step,objective=ARMS[arm][1],
            learning_rate=ARMS[arm][0],auxiliary_weight=.3,architecture=ARCHITECTURE,config=CONFIG,
            weights=_cpu_copy(models[arm].state_dict()),optimizer=_cpu_copy(optimizers[arm].state_dict()),
            weights_sha256=checkpoint_digest(models[arm]),runtime=receipt["runtime"],data_identity=tensor_pins,
            lifetime_updates=cursors[arm],recipe=launch["contract"])
        _check_finite_tree(payload,"shared-state fit checkpoint"); receipt["work"]["snapshots"]+=1
        receipt["checkpoints"].setdefault(arm,{})[str(step)]=artifact(f"checkpoints/{arm}-{step:04d}.pt",payload,True)

    try:
        journal=Journal(attempt/"events.jsonl"); publish(attempt/"started.json",receipt)
        op("authentication","initial",authenticate)
        import torch
        from experiments import execution_profile as execution,foundation_layout_evaluation as evaluation
        from experiments.shared_state_student import SharedStateStudent,build_shared_state_student,ARCHITECTURE
        from experiments.shared_state_targets import pack_state_targets,state_loss
        from experiments.sequence_student import SequenceConfig
        from experiments.sequence_training import sequence_objective
        from experiments.composition_data import pack_composition_episodes
        from experiments.foundation_layout_curriculum import WorkLedger
        from experiments.foundation_evidence import json_digest
        from brain_in_computer.language import ByteCodec
        from brain_in_computer.learning_student import _cpu_copy,_check_finite_tree
        from brain_in_computer.dialogue_student import checkpoint_digest
        torch.set_num_threads(1);torch.set_num_interop_threads(1);execution.configure_strict_profile()
        receipt["runtime"]=op("runtime","cuda:0",execution.runtime_profile)
        if receipt["runtime"]!=launch["expected_runtime"]: raise ValueError("original strict local runtime differs")
        def load():
            raw=native(ROOT/DATA/"banks.pt").read_bytes()
            if hashlib.sha256(raw).hexdigest()!=BANK_SHA: raise ValueError("bank image differs")
            decoded=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
            selected=decoded["train_fit"]; del decoded; return selected
        bank=op("bank_load","training-fit only",load); rows=bank["rows"]
        if bank["role"]!="train_fit" or len(rows)!=108 or json_digest(rows)!=launch["fit_inventory"]["rows_sha256"]: raise ValueError("actual fitting rows differ")
        config=SequenceConfig(**CONFIG); validation=WorkLedger(); grouped=defaultdict(list);group_ids={}
        for row in rows: grouped[f"{row['family']}/{len(row['turns'])}"].append(row)
        if set(grouped)!={f"{family}/{turns}" for family in FAMILIES for turns in LENGTHS} or any(len(v)!=12 for v in grouped.values()): raise ValueError("nine fixed twelve-row groups required")
        for key,group in sorted(grouped.items()):
            prepared[key]=op("bank_preparation",key,lambda:evaluation.PreparedLayoutBank(group,role="train_fit",config=config,validation_work=validation))
            cpu_batches[key]=op("packing",key,lambda:pack_composition_episodes(group,training=False,max_turns=12,
                max_input_bytes=128,max_context_tokens=1024,max_reply_bytes=32))
            target=op("target_preparation",key,lambda:pack_state_targets(group));receipt["work"]["target_packs"]+=1;receipt["work"]["target_rows"]+=12
            gpu_batches[key]={section:{name:value.to("cuda:0") for name,value in values.items()} for section,values in cpu_batches[key].items()}
            labels[key]=target.to("cuda:0");group_ids[key]=[row["id"] for row in group]
        del bank,rows,grouped,group,target
        tensor_pins=_hash(dict(cpu=tensor_digest(cpu_batches),gpu=tensor_digest(gpu_batches),labels=tensor_digest(labels)))
        artifact("data-identity.json",dict(tensors_sha256=tensor_pins,groups=group_ids,
            banks={key:bank.identity for key,bank in prepared.items()},source_bank_sha256=BANK_SHA))
        torch.cuda.reset_peak_memory_stats()
        for arm,(lr,objective) in ARMS.items():
            receipt["work"]["model_attempts"]+=1
            models[arm]=op("model_construction",arm,lambda:build_shared_state_student(SEED,device="cuda:0",config=config));receipt["work"]["models"]+=1
            optimizers[arm]=torch.optim.AdamW(models[arm].parameters(),lr=lr);cursors[arm]=0;receipt["arms"][arm]=dict(cursor=0)
            model_guard(arm)
        base=models[next(iter(ARMS))].state_dict()
        for model in models.values():
            other=model.state_dict()
            if set(base)!=set(other) or any(base[key].dtype!=other[key].dtype or base[key].shape!=other[key].shape or not torch.equal(base[key],other[key]) for key in base): raise ValueError("four initial full states differ")
        receipt["initial_weights_sha256"]=checkpoint_digest(models[next(iter(ARMS))]);del base,other
        receipt["setup_wall_seconds"]=time.monotonic()-started
        for arm in ARMS: evaluate(arm,0)
        names=list(ARMS)
        for cursor in range(UPDATES):
            turns=LENGTHS[cursor%3];offset=cursor%4
            for arm in names[offset:]+names[:offset]:
                report=op("training",f"{arm}/{cursor}",lambda:step(arm,turns))
                event(dict(event="step_report",**report));active_step=None
            if (cursor+1)%36==0:
                data_guard();print(encoded(dict(event="progress",updates_per_arm=cursor+1,wall_seconds=time.monotonic()-started)).decode().strip(),flush=True)
            if cursor+1 in ENDPOINTS:
                op("authentication",str(cursor+1),authenticate)
                for arm in ARMS: op("checkpoint",f"{arm}/{cursor+1}",lambda:checkpoint(arm,cursor+1))
                for arm in ARMS: evaluate(arm,cursor+1)
        expected=dict(models=4,target_packs=9,target_rows=108,training_forward_attempts=5184,training_forwards=5184,
            training_episode_exposures=62208,training_state_readout_attempts=5184,training_state_readouts=5184,
            backward_attempts=5184,backwards=5184,optimizer_attempts=1728,optimizer_returns=1728,
            synchronized_updates=1728,unknown_optimizer_outcomes=0,aux_evaluation_attempts=108,aux_evaluation_forwards=108,
            aux_evaluation_episodes=1296,aux_evaluation_readout_attempts=108,aux_evaluation_readouts=108,snapshot_attempts=8,snapshots=8,model_attempts=4)
        if receipt["work"]!=expected or any(value!=432 for value in cursors.values()): raise ValueError("completed fixed work differs")
        if operations["native_evaluation"]["completions"]!=108: raise ValueError("native evaluation work differs")
        if (receipt["native_evaluation_work"]["sequence_forward_completions"]!=108
                or receipt["native_evaluation_work"]["encoder_episode_completions"]!=1296): raise ValueError("native physical evaluation counts differ")
        data_guard();op("authentication","terminal",authenticate);receipt["status"]="completed"
    except BaseException as error:
        receipt.update(status="interrupted" if isinstance(error,(KeyboardInterrupt,SystemExit)) else "failed",error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if active_step is not None:receipt["active_step"]=deepcopy(active_step)
        if active_bank is not None:receipt["active_evaluation"]=dict(report=deepcopy(active_bank.last_report),work=active_ledger.report())
        try:
            if "torch" in locals() and torch.cuda.is_initialized():
                torch.cuda.synchronize();receipt["peak_gpu_allocated_bytes"]=torch.cuda.max_memory_allocated()
            if journal is not None:journal.close();receipt["journal_sha256"]=digest(attempt/"events.jsonl")
        except BaseException as error:
            receipt.update(cleanup_error=repr(error),partial_work_unknown=True)
            if receipt["status"]=="completed":receipt["status"]="failed"
        receipt.update(operations=operations,bank_validation_work=validation.report() if validation else None,
            ended_utc=utc(),wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
        publish(attempt/"summary.json",receipt)
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("freeze","run"));parser.add_argument("output")
    parser.add_argument("--launch-sha256");parser.add_argument("--input-pins");args=parser.parse_args()
    if args.mode=="freeze":print(encoded(dict(launch_sha256=freeze(args.output,input_pins=read(args.input_pins) if args.input_pins else None))).decode().strip(),flush=True)
    else:
        result=run(args.output,expected_launch_sha256=args.launch_sha256)
        print(encoded(dict(status=result["status"],wall_seconds=result["wall_seconds"])).decode().strip(),flush=True)
