"""Final training-fit state readout; descriptive, inference-only and exclusive."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import io
from pathlib import Path
import time
import traceback
from unittest.mock import patch

from experiments.foundation_layout_study import ROOT,digest,native,read,publish,verify_pins,relative_root,utc

SCHEMA = "bic-shared-state-rate-diagnostic-v1"
PROTOCOL = "docs/SHARED_STATE_RATE_DIAGNOSTIC.md"
METRIC_ORIGIN = "runs/shared-state-fit-local/attempt-001/launch.json"
METRIC_ORIGIN_SHA = "1b8a920caf81d3c0dbc02a447adc0480505577192632003a913df147003642e1"
METRIC_SOURCES = {
    "experiments/shared_state_fit.py":"0de6121556fb64149dd695673cd26536aa8bdcc485289eb6ddacb24869011348",
    "experiments/shared_state_pilot.py":"aa323cf0097a01d5b6156dac90d1754fb202c9f96451ebe4a21ef746e2164b88"}
ARMS,FAMILIES,LENGTHS = ("fast","slow"),("color","count","switch"),(8,10,12)
RATES = dict(fast=.003, slow=.0003)
EXPECTED = dict(models=2,checkpoint_loads=2,bank_loads=1,forwards=18,auxiliary_calls=18,
    episodes=216,state_predictions=25920,target_packs=9,target_rows=108,target_turns=1080,
    optimizer_calls=0,backwards=0,free_replies=0,teacher_calls=0)


def run(directory, *, main_directory,summary_sha256,launch_sha256,source_sha256,protocol_sha256):
    started,cpu = time.monotonic(),time.process_time()
    directory,main = Path(directory).resolve(),Path(main_directory).resolve()
    summary_path,launch_path = main/"execution/summary.json",main/"launch.json"
    pins = {relative_root(Path(__file__)):source_sha256,PROTOCOL:protocol_sha256,
        relative_root(summary_path):summary_sha256,relative_root(launch_path):launch_sha256,
        METRIC_ORIGIN:METRIC_ORIGIN_SHA,**METRIC_SOURCES}
    verify_pins(pins)
    launch,summary,origin = read(launch_path),read(summary_path),read(ROOT/METRIC_ORIGIN)
    if (summary["status"]!="completed" or summary["launch_sha256"]!=launch_sha256
            or launch["schema"]!="bic-shared-state-rate-pilot-v1" or summary["schema"]!=launch["schema"]
            or launch["contract"]["learning_rates"]!=RATES
            or launch["contract"]["auxiliary_weights"]!=dict(fast=.3,slow=.3)
            or launch["contract"]["updates_per_arm"]!=648 or launch["contract"]["seed"]!=852604001 or set(summary["arms"])!=set(ARMS)
            or any(summary["arms"][arm]["cursor"]!=648 for arm in ARMS)
            or any(origin["source_sha256"].get(name)!=pin for name,pin in METRIC_SOURCES.items())):
        raise ValueError("completed final shared-state rate run and verified metric origin required")
    for name,pin in launch["source_sha256"].items():
        if name in pins and pins[name]!=pin: raise ValueError("main/diagnostic source identities conflict")
        pins[name]=pin
    verify_pins(pins)
    native(directory).mkdir(parents=True,exist_ok=False)
    receipt = dict(schema=SCHEMA,status="running",started_utc=utc(),input_sha256=pins,
        expected_work=EXPECTED,work=dict.fromkeys(EXPECTED,0),guard_attempts={},max_seconds=180,
        attempts=dict(models=0,checkpoint_loads=0,bank_loads=0,forwards=0,auxiliary_calls=0,target_packs=0),
        automatic_retry=False,automatic_promotion=False,active_operation=None)
    publish(directory/"started.json",receipt)
    model=None
    def boundary(operation):
        receipt["active_operation"]=operation
        if time.monotonic()-started>=180: raise TimeoutError("fixed diagnostic allowance expired")
    def forbidden(name):
        def call(*args,**kwargs):
            receipt["guard_attempts"][name]=receipt["guard_attempts"].get(name,0)+1
            raise AssertionError("inference-only diagnostic forbids "+name)
        return call
    try:
        import torch
        from experiments import execution_profile as execution,shared_state_student as student
        from experiments.shared_state_targets import pack_state_targets,state_loss
        from experiments.shared_state_fit import state_metrics
        from experiments.sequence_student import SequenceConfig
        from experiments.composition_data import pack_observations
        from experiments.foundation_evidence import json_digest
        from brain_in_computer.dialogue_student import checkpoint_digest
        from brain_in_computer.language import ByteCodec
        torch.set_num_threads(1);torch.set_num_interop_threads(1);execution.configure_strict_profile()
        boundary("runtime");receipt["runtime"]=execution.runtime_profile()
        if receipt["runtime"]!=summary["runtime"]: raise ValueError("exact main strict FP32 runtime required")
        torch.cuda.reset_peak_memory_stats()
        def load(path,pin,kind):
            boundary(kind+"_load");receipt["attempts"][kind+"_loads"]+=1
            image=native(path).read_bytes()
            if hashlib.sha256(image).hexdigest()!=pin: raise ValueError("immutable artifact pin differs")
            pins[relative_root(path)]=pin
            value=torch.load(io.BytesIO(image),map_location="cpu",weights_only=True)
            receipt["work"][kind+"_loads"]+=1;return value
        with ExitStack() as stack:
            for owner,name,label in ((torch.optim.Optimizer,"__init__","optimizer"),
                (torch.Tensor,"backward","backward"),(torch.autograd,"backward","autograd.backward"),
                (torch.autograd,"grad","autograd.grad")):
                stack.enter_context(patch.object(owner,name,forbidden(label)))
            data=ROOT/launch["data_directory"];manifest_path=data/"manifest.json"
            if digest(manifest_path)!=launch["data_manifest_sha256"]: raise ValueError("main data manifest changed")
            pins[relative_root(manifest_path)]=launch["data_manifest_sha256"];manifest=read(manifest_path)
            banks=load(data/manifest["banks"]["path"],manifest["banks"]["sha256"],"bank")
            bank=banks["train_fit"];del banks
            inventory=manifest["bank_inventory"]["train_fit"]
            if (bank["role"]!="train_fit" or inventory["provenance"]!="reused-actual-training" or len(bank["rows"])!=108
                    or json_digest(bank["rows"])!=inventory["rows_sha256"]): raise ValueError("only exact actual training-fit rows allowed")
            grouped=defaultdict(list)
            for row in bank["rows"]: grouped[f"{row['family']}/{len(row['turns'])}"].append(row)
            if set(grouped)!={f"{f}/{t}" for f in FAMILIES for t in LENGTHS} or any(len(rows)!=12 for rows in grouped.values()):
                raise ValueError("nine twelve-row family/length groups required")
            config=SequenceConfig(**launch["contract"]["config"]);prepared={}
            for key,rows in sorted(grouped.items()):
                boundary("prepare/"+key);receipt["attempts"]["target_packs"]+=1
                labels=pack_state_targets(rows);family,turns=rows[0]["family"],len(rows[0]["turns"])
                inputs=pack_observations([[turn["text"] for turn in row["turns"]] for row in rows],
                    max_turns=config.max_turns,max_input_bytes=config.max_input_bytes,max_context_tokens=config.max_positions)
                prepared[key]=(family,turns,inputs,labels,[row["id"] for row in rows])
                receipt["work"]["target_packs"]+=1;receipt["work"]["target_rows"]+=12;receipt["work"]["target_turns"]+=12*turns
            del bank,grouped,rows
            result=dict(schema=SCHEMA,launch_sha256=launch_sha256,summary_sha256=summary_sha256,arms={},
                scope="Training-fit representation only; labels never native inputs; no held-out state targets or primary-screen changes.")
            for arm in ARMS:
                arm_started=time.monotonic();record=summary["checkpoints"][arm]["648"];path=ROOT/record["path"]
                if record["path"]!=relative_root(main/f"execution/checkpoints/{arm}-0648.pt") or summary["artifact_sha256"][f"checkpoints/{arm}-0648.pt"]!=record["sha256"]:
                    raise ValueError("only predetermined final checkpoint images allowed")
                saved=load(path,record["sha256"],"checkpoint")
                module=student
                if (saved["schema"]!=launch["schema"] or saved["launch_sha256"]!=launch_sha256 or saved["arm"]!=arm
                        or saved["step"]!=648 or saved["learner"]["cursor"]!=648 or saved["architecture"]!=module.ARCHITECTURE
                        or saved["auxiliary_weight"]!=.3 or saved["learning_rate"]!=RATES[arm]
                        or saved["runtime"]!=receipt["runtime"]
                        or saved["learner"]["recipe"]["config"]!=launch["contract"]["config"]): raise ValueError("final checkpoint identity differs")
                recipe=saved["learner"]["recipe"]
                groups=recipe["optimizer"];stored_groups=saved["learner"]["optimizer"]["param_groups"]
                if (type(groups) is not list or len(groups)!=1 or type(groups[0]) is not dict
                        or type(groups[0].get("lr")) is not float or groups[0]["lr"]!=launch["contract"]["learning_rates"][arm]
                        or type(stored_groups) is not list or stored_groups!=groups
                        or recipe["architecture"]!=student.ARCHITECTURE or recipe["auxiliary_weight"]!=.3):
                    raise ValueError("saved AdamW recipe/rate or state objective differs from the arm contract")
                boundary("model/"+arm);receipt["attempts"]["models"]+=1
                builder=student.build_shared_state_student
                model=builder(launch["contract"]["seed"],device="cuda:0",config=config);receipt["work"]["models"]+=1
                expected=model.state_dict();weights=saved["learner"]["weights"]
                if set(weights)!=set(expected) or any(type(v) is not torch.Tensor or v.device.type!="cpu"
                    or v.dtype!=expected[name].dtype or v.shape!=expected[name].shape for name,v in weights.items()): raise ValueError("full FP32 weight layout differs")
                if not torch.equal(weights["tokens.weight"],weights["observation_head.weight"]): raise ValueError("tied saved weights differ")
                model.load_state_dict(weights,strict=True);model.eval();before=checkpoint_digest(model)
                if before!=saved["weights_sha256"] or model.tokens.weight is not model.observation_head.weight: raise ValueError("exact restored weights differ")
                del saved,weights,expected
                raw={};loss_sums=defaultdict(float);turn_counts=defaultdict(int)
                with torch.inference_mode():
                    for key,(family,turns,inputs,labels,ids) in prepared.items():
                        boundary("forward/"+arm+"/"+key);execution.assert_strict_profile()
                        batch={name:value.to("cuda:0") for name,value in inputs.items()}
                        bos=torch.full((12,turns,1),ByteCodec.BOS,dtype=torch.long,device="cuda:0")
                        receipt["attempts"]["forwards"]+=1;output=model(**batch,decoder_input_ids=bos)
                        receipt["work"]["forwards"]+=1;receipt["work"]["episodes"]+=12
                        boundary("auxiliary/"+arm+"/"+key);receipt["attempts"]["auxiliary_calls"]+=1
                        logits=model.state_logits(output["context_states"],batch["eos_positions"])
                        receipt["work"]["auxiliary_calls"]+=1;target=labels.to("cuda:0")
                        balanced=float(state_loss(logits,target));values=logits[...,1:]
                        truth=values.gather(-1,(target-1).clamp_min(0)[...,None])
                        ranks=(1+(values>truth).sum(-1)).masked_fill(target.eq(0),0)
                        raw[key]=dict(family=family,turns=turns,episode_ids=ids,targets=labels.tolist(),
                            predictions=logits.argmax(-1).cpu().tolist(),conditional_predictions=(values.argmax(-1)+1).cpu().tolist(),
                            true_value_ranks=ranks.cpu().tolist(),known_probability=(1-logits.softmax(-1)[...,0]).cpu().tolist(),balanced_state_loss=balanced)
                        loss_sums[family]+=balanced*12*turns;turn_counts[family]+=12*turns
                        receipt["work"]["state_predictions"]+=labels.numel()
                        del output,logits,values,truth,ranks,target,batch,bos
                torch.cuda.synchronize();execution.assert_strict_profile()
                if checkpoint_digest(model)!=before or any(m.training for m in model.modules()) or any(p.grad is not None for p in model.parameters()):
                    raise ValueError("inference changed weights, mode or gradients")
                raw=dict(sorted(raw.items()));metrics=state_metrics(raw)
                losses=dict(overall=sum(loss_sums.values())/sum(turn_counts.values()),
                    by_family={family:loss_sums[family]/turn_counts[family] for family in FAMILIES})
                result["arms"][arm]=dict(checkpoint=record,learning_rate=RATES[arm],weights_sha256=before,model_unchanged=True,
                    raw=raw,metrics=metrics,balanced_state_loss=losses,wall_seconds=time.monotonic()-arm_started)
                del model;model=None
            if receipt["work"]!=EXPECTED or receipt["guard_attempts"]: raise ValueError("fixed inference-only work differs")
            boundary("terminal_authentication");verify_pins(pins)
            boundary("publication");receipt["diagnostic_sha256"]=publish(directory/"diagnostic.json",result)
            receipt.update(status="completed",active_operation=None)
    except BaseException as error:
        receipt.update(status="failed",error=repr(error),traceback=traceback.format_exc());raise
    finally:
        try:
            if "torch" in locals() and torch.cuda.is_initialized():
                torch.cuda.synchronize();receipt["peak_gpu_allocated_bytes"]=torch.cuda.max_memory_allocated()
                receipt["peak_gpu_reserved_bytes"]=torch.cuda.max_memory_reserved()
        except BaseException as error: receipt.update(status="failed",cleanup_error=repr(error))
        receipt.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,input_sha256=pins)
        publish(directory/"receipt.json",receipt)
    return dict(status=receipt["status"],diagnostic_sha256=receipt.get("diagnostic_sha256"),work=receipt["work"])


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("directory");parser.add_argument("--main-directory",required=True)
    for name in ("summary-sha256","launch-sha256","source-sha256","protocol-sha256"):parser.add_argument("--"+name,required=True)
    args=parser.parse_args();print(run(args.directory,main_directory=args.main_directory,summary_sha256=args.summary_sha256,
        launch_sha256=args.launch_sha256,source_sha256=args.source_sha256,protocol_sha256=args.protocol_sha256))
