"""Pinned final joint checkpoints on reused development: inference only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

from experiments.foundation_layout_study import (ROOT,CONFIG,native,digest,read,publish,
    relative_root,verify_pins,encoded,utc,_hash,Journal as BaseJournal)

SCHEMA="bic-shared-state-fit-transfer-v1"
PROTOCOL="docs/SHARED_STATE_FIT_TRANSFER.md"
FIT="runs/shared-state-fit-local/attempt-001"
FIT_LAUNCH="1b8a920caf81d3c0dbc02a447adc0480505577192632003a913df147003642e1"
FIT_SUMMARY="9e4de8297cf917d54d58306fb1337ea407089cfef5ea19eee1f274286b383f5c"
DATA="runs/shared-state-data-local/attempt-001"
MANIFEST_SHA="266dfe643074b8d66b65b1e506f6fc726fa6cb75f2a532b04232ec87bf30eb75"
BANK_SHA="c42c117d04070cae026e6372615bcb044b493ff41f80603a268dc8aff93e4efb"
ARMS,FAMILIES,LENGTHS=("joint_fast","joint_slow"),("color","count","switch"),(8,10,12)
SECONDS=180


class Journal(BaseJournal):
    def __init__(self,path):
        super().__init__(path);self.chain=_hash([SCHEMA,"events"])


def run(directory, *, source_sha256, protocol_sha256):
    started,cpu=time.monotonic(),time.process_time();deadline=started+SECONDS
    own=relative_root(Path(__file__))
    initial={own:source_sha256,PROTOCOL:protocol_sha256,FIT+"/launch.json":FIT_LAUNCH,
             FIT+"/execution/summary.json":FIT_SUMMARY,DATA+"/manifest.json":MANIFEST_SHA,DATA+"/banks.pt":BANK_SHA}
    verify_pins(initial)
    launch,summary,manifest=read(ROOT/FIT/"launch.json"),read(ROOT/FIT/"execution/summary.json"),read(ROOT/DATA/"manifest.json")
    if (launch["schema"]!="bic-shared-state-fit-v1" or summary["status"]!="completed"
            or summary["launch_sha256"]!=FIT_LAUNCH or launch["contract"]["config"]!=CONFIG
            or set(summary["arms"])!=set(launch["contract"]["arms"])
            or any(row["cursor"]!=432 for row in summary["arms"].values())
            or manifest["banks"]!=dict(path="banks.pt",sha256=BANK_SHA)):
        raise ValueError("completed final fit identities and fixed source bank required")
    pins={**launch["source_sha256"],**initial}
    # The authenticated data producer binds the canonical inventory digest helper.
    for name in ("experiments/foundation_evidence.py",):
        pin=manifest["source_sha256"][name]
        if name in pins and pins[name]!=pin:raise ValueError("data/training source identity differs")
        pins[name]=pin
    for arm in ARMS:
        record=summary["checkpoints"][arm]["432"]
        expected=f"{FIT}/execution/checkpoints/{arm}-0432.pt"
        if (record["path"]!=expected or summary["artifact_sha256"][f"checkpoints/{arm}-0432.pt"]!=record["sha256"]
                or launch["contract"]["arms"][arm]["objective"]!="joint"):
            raise ValueError("only both predetermined final joint checkpoints allowed")
        pins[record["path"]]=record["sha256"]
    verify_pins(pins)
    directory=Path(directory).resolve();native(directory).mkdir(parents=True,exist_ok=False)
    receipt=dict(schema=SCHEMA,status="running",started_utc=utc(),pid=os.getpid(),max_seconds=SECONDS,
        source_sha256=source_sha256,protocol_sha256=protocol_sha256,input_sha256=pins,results={},artifact_sha256={},
        expected_work=dict(models=2,checkpoint_loads=2,bank_loads=1,native_forwards=54,episodes=1440),
        work=dict(model_attempts=0,models=0,checkpoint_load_attempts=0,checkpoint_loads=0,bank_load_attempts=0,bank_loads=0,
                  optimizer_calls=0,backwards=0,auxiliary_calls=0,teacher_calls=0),native_work={},guard_attempts={},
        automatic_retry=False,automatic_promotion=False,partial_work_unknown=False,
        scope="Reused development, disjoint observed transcripts from fixed fitting buffer; shared finite grammar remains")
    journal,model,validation,active_bank,active_ledger=None,None,None,None,None
    operations,prepared={},{}

    def boundary():
        if time.monotonic()>=deadline:raise TimeoutError("180-second transfer allowance expired; no retry")
    def event(value):
        try:journal.event(value)
        except BaseException:receipt["partial_work_unknown"]=True;raise
    def op(kind,identity,call):
        boundary();stats=operations.setdefault(kind,dict(attempts=0,completions=0,failures=0,wall_seconds=0.,cpu_seconds=0.))
        stats["attempts"]+=1;t,c=time.monotonic(),time.process_time();receipt["active_operation"]=dict(kind=kind,identity=identity)
        try:
            event(dict(event="intent",kind=kind,identity=identity));value=call();stats["completions"]+=1
            event(dict(event="complete",kind=kind,identity=identity,wall_seconds=time.monotonic()-t,cpu_seconds=time.process_time()-c))
            receipt["active_operation"]=None;return value
        except BaseException:stats["failures"]+=1;raise
        finally:stats["wall_seconds"]+=time.monotonic()-t;stats["cpu_seconds"]+=time.process_time()-c
    def artifact(name,value):
        pin=op("publication",name,lambda:publish(directory/name,value));receipt["artifact_sha256"][name]=pin
        return dict(path=relative_root(directory/name),sha256=pin)
    def forbidden(name):
        def call(*args,**kwargs):
            receipt["guard_attempts"][name]=receipt["guard_attempts"].get(name,0)+1
            raise AssertionError("inference-only contract forbids "+name)
        return call
    def load(path,pin,kind):
        receipt["work"][kind+"_load_attempts"]+=1
        image=native(path).read_bytes()
        if hashlib.sha256(image).hexdigest()!=pin:raise ValueError("immutable input image changed")
        value=torch.load(io.BytesIO(image),map_location="cpu",weights_only=True)
        receipt["work"][kind+"_loads"]+=1;return value

    try:
        journal=Journal(directory/"events.jsonl");publish(directory/"started.json",receipt)
        import torch
        from experiments import execution_profile as execution,foundation_layout_evaluation as evaluation
        from experiments.shared_state_student import SharedStateStudent,build_shared_state_student,ARCHITECTURE
        from experiments.sequence_student import SequenceConfig
        from experiments.foundation_layout_curriculum import WorkLedger
        from experiments.foundation_evidence import json_digest
        from brain_in_computer.learning_student import _check_finite_tree
        from brain_in_computer.dialogue_student import checkpoint_digest
        torch.set_num_threads(1);torch.set_num_interop_threads(1);execution.configure_strict_profile()
        receipt["runtime"]=op("runtime","cuda:0",execution.runtime_profile)
        if receipt["runtime"]!=summary["runtime"]:raise ValueError("exact fit runtime required")
        torch.cuda.reset_peak_memory_stats()
        with ExitStack() as stack:
            for owner,name,label in ((torch.optim.Optimizer,"__init__","optimizer"),(torch.Tensor,"backward","backward"),
                (torch.autograd,"backward","autograd.backward"),(torch.autograd,"grad","autograd.grad"),
                (SharedStateStudent,"state_logits","auxiliary_decoder")):
                stack.enter_context(patch.object(owner,name,forbidden(label)))
            banks=op("bank_load","selected dev and fit",lambda:load(ROOT/DATA/"banks.pt",BANK_SHA,"bank"))
            dev,fit=banks["dev"],banks["train_fit"];del banks
            if (dev["role"]!="dev" or fit["role"]!="train_fit" or len(dev["rows"])!=720 or len(fit["rows"])!=108
                    or json_digest(dev["rows"])!=manifest["bank_inventory"]["dev"]["rows_sha256"]
                    or json_digest(fit["rows"])!=launch["fit_inventory"]["rows_sha256"]):
                raise ValueError("selected dev/actual-fit inventories differ")
            dev_hashes={tuple(turn["text"] for turn in row["turns"]) for row in dev["rows"]}
            fit_hashes={tuple(turn["text"] for turn in row["turns"]) for row in fit["rows"]}
            overlap=dev_hashes&fit_hashes
            receipt["transcript_overlap"]=dict(dev_rows=720,fit_rows=108,dev_unique=len(dev_hashes),fit_unique=len(fit_hashes),
                intersection=len(overlap),dev_sha256=json_digest(sorted(dev_hashes)),fit_sha256=json_digest(sorted(fit_hashes)))
            if overlap:raise ValueError("development overlaps actual fitting transcripts")
            config=SequenceConfig(**CONFIG);validation=WorkLedger();groups=defaultdict(list)
            for row in dev["rows"]:groups[f"{row['family']}/{len(row['turns'])}"].append(row)
            if set(groups)!={f"{family}/{turns}" for family in FAMILIES for turns in LENGTHS} or any(len(rows)!=80 for rows in groups.values()):
                raise ValueError("nine complete eighty-row development groups required")
            for key,rows in sorted(groups.items()):
                prepared[key]=op("canonical_bank_validation",key,lambda:evaluation.PreparedLayoutBank(rows,role="dev",config=config,validation_work=validation))
            del dev,fit,groups,rows
            artifact("bank-identities.json",{key:bank.identity for key,bank in prepared.items()})
            for arm in ARMS:
                op("source_guard",arm,lambda:verify_pins(pins));record=summary["checkpoints"][arm]["432"]
                saved=op("checkpoint_load",arm,lambda:load(ROOT/record["path"],record["sha256"],"checkpoint"))
                if (saved["schema"]!=launch["schema"] or saved["launch_sha256"]!=FIT_LAUNCH or saved["arm"]!=arm
                        or saved["step"]!=432 or saved["lifetime_updates"]!=432 or saved["objective"]!="joint"
                        or saved["learning_rate"]!=launch["contract"]["arms"][arm]["learning_rate"]
                        or saved["auxiliary_weight"]!=.3 or saved["architecture"]!=ARCHITECTURE
                        or saved["config"]!=CONFIG or saved["recipe"]!=launch["contract"] or saved["runtime"]!=receipt["runtime"]):
                    raise ValueError("exact final checkpoint producer identity differs")
                _check_finite_tree(saved["weights"],"saved transfer weights")
                receipt["work"]["model_attempts"]+=1
                model=op("model_construction",arm,lambda:build_shared_state_student(launch["contract"]["seed"],device="cuda:0",config=config))
                receipt["work"]["models"]+=1
                expected=model.state_dict()
                if set(saved["weights"])!=set(expected) or any(type(value) is not torch.Tensor or value.device.type!="cpu"
                    or value.dtype!=expected[name].dtype or value.shape!=expected[name].shape for name,value in saved["weights"].items()):
                    raise ValueError("complete finite FP32 weight layout differs")
                if not torch.equal(saved["weights"]["tokens.weight"],saved["weights"]["observation_head.weight"]):raise ValueError("tied source weights differ")
                model.load_state_dict(saved["weights"],strict=True);model.eval();torch.cuda.synchronize()
                before=checkpoint_digest(model)
                if before!=saved["weights_sha256"] or model.tokens.weight is not model.observation_head.weight:raise ValueError("exact restored weight digest/alias differs")
                del saved,expected
                records,artifacts=[],{}
                for key,bank in prepared.items():
                    active_bank,active_ledger=bank,evaluation.EvaluationLedger()
                    result=op("native_evaluation",arm+"/"+key,lambda:bank.score(model,batch_size=32,deadline=deadline,work=active_ledger,
                        progress=lambda value:event(dict(kind="evaluation_batch",arm=arm,group=key,**value))))
                    for name,value in active_ledger.counts.items():receipt["native_work"][name]=receipt["native_work"].get(name,0)+value
                    records.extend(result["raw_records"]);artifacts[key]=artifact("scores/"+arm+"/"+key+".json",result)
                    active_bank=active_ledger=None
                if checkpoint_digest(model)!=before or any(p.grad is not None for p in model.parameters()):raise ValueError("inference changed full learner state")
                metrics=evaluation.score_records(records)
                if metrics["overall"]["episodes"]!=720 or metrics["overall"]["pairs"]!=360:raise ValueError("complete development denominator differs")
                receipt["results"][arm]=dict(checkpoint=record,weights_sha256=before,model_unchanged=True,metrics=metrics,groups=artifacts)
                artifact("scores/"+arm+"/summary.json",receipt["results"][arm]);del model;model=None
            if (receipt["work"]!=dict(model_attempts=2,models=2,checkpoint_load_attempts=2,checkpoint_loads=2,
                    bank_load_attempts=1,bank_loads=1,optimizer_calls=0,backwards=0,auxiliary_calls=0,teacher_calls=0)
                    or receipt["native_work"]["sequence_forward_completions"]!=54
                    or receipt["native_work"]["encoder_episode_completions"]!=1440 or receipt["guard_attempts"]):
                raise ValueError("fixed inference-only work differs")
            op("source_guard","terminal",lambda:verify_pins(pins));receipt["status"]="completed"
    except BaseException as error:
        receipt.update(status="interrupted" if isinstance(error,(KeyboardInterrupt,SystemExit)) else "failed",error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if active_bank is not None:receipt["active_evaluation"]=dict(report=deepcopy(active_bank.last_report),work=active_ledger.report())
        try:
            if "torch" in locals() and torch.cuda.is_initialized():
                torch.cuda.synchronize();receipt["peak_gpu_allocated_bytes"]=torch.cuda.max_memory_allocated()
            if journal is not None:journal.close();receipt["journal_sha256"]=digest(directory/"events.jsonl")
        except BaseException as error:
            receipt.update(cleanup_error=repr(error),partial_work_unknown=True)
            if receipt["status"]=="completed":receipt["status"]="failed"
        receipt.update(operations=operations,validation_work=validation.report() if validation else None,
            ended_utc=utc(),wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
        publish(directory/"summary.json",receipt)
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("directory");parser.add_argument("--source-sha256",required=True)
    parser.add_argument("--protocol-sha256",required=True);args=parser.parse_args()
    result=run(args.directory,source_sha256=args.source_sha256,protocol_sha256=args.protocol_sha256)
    print(encoded(dict(status=result["status"],wall_seconds=result["wall_seconds"])).decode().strip(),flush=True)
