"""One bounded production-shape exact restart proof; no automatic retry."""
import argparse
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SHA = "a5bcb0171cc038451a97f799130fae831b252a8b78f7016aa4e25b36ec1950b7"
CPU_RECEIPT = "runs/shared-state-continuation-validation-local/attempt-001/report.json"
CPU_SHA = "645d3c36e469c14baa81cb15e7c2d30a311e2df5e9a4f7d39231546e78a822f7"
DATA = "runs/recurrent-read-data-local/attempt-001"
MANIFEST_SHA = "c420a43a3a45f63347288dda7db48c71b3e7fa183b4a1681c2ec89eabeb08654"
LIMITS = dict(models=2, optimizers=2, updates=3, forwards=9, backwards=9,
              archive_loads=3, target_packs=6, english_oracles=192, typed_oracles=192,
              canonical_generations=0)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path, value, *, raw=False):
    image = value if raw else (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with path.open("xb") as stream:
        stream.write(image); stream.flush(); os.fsync(stream.fileno())
    return hashlib.sha256(image).hexdigest()


def run(directory, *, expected_source_sha256):
    started, cpu = time.monotonic(), time.process_time()
    here = Path(directory).resolve(); here.mkdir(parents=True, exist_ok=False)
    calls = {name:dict(attempts=0,completions=0) for name in LIMITS}
    report = dict(schema="bic-shared-state-continuation-gpu-proof-v1", status="failed", limits=LIMITS,
        max_seconds=180, automatic_retry=False, automatic_promotion=False, forward_episodes=0,
        target_rows=0, target_turns=0, new_distinct_lessons=0, teacher_calls=0,
        source_sha256={}, source_snapshots={}, steps=[], guards={}, active_operation="authentication",
        scope="One production-shape .0003 restart case; not arbitrary-training equivalence or learned capability.")
    owners, caught = [], None
    def boundary(operation):
        report["active_operation"] = operation
        if time.monotonic()-started >= 180: raise TimeoutError("GPU proof fixed allowance expired")
    def limited(name, original):
        def call(*args, **kwargs):
            boundary(name); calls[name]["attempts"] += 1
            if calls[name]["attempts"] > LIMITS[name]: raise AssertionError("work limit: "+name)
            result = original(*args, **kwargs); calls[name]["completions"] += 1
            return result
        return call
    publish(here/"harness.py", Path(__file__).read_bytes(), raw=True)
    publish(here/"command.json", dict(argv=sys.argv, executable=sys.executable, cwd=str(Path.cwd())))
    publish(here/"started.json", report)
    with (here/"stdout.txt").open("x",encoding="utf-8") as stdout, (here/"stderr.txt").open("x",encoding="utf-8") as stderr:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                if (sha(Path(__file__)) != expected_source_sha256
                        or sha(ROOT/"experiments/shared_state_continuation.py") != BRIDGE_SHA
                        or sha(ROOT/CPU_RECEIPT) != CPU_SHA
                        or sha(ROOT/DATA/"manifest.json") != MANIFEST_SHA):
                    raise ValueError("proof, adapter, CPU evidence or admitted manifest pin differs")
                report["input_sha256"] = {CPU_RECEIPT:CPU_SHA, DATA+"/manifest.json":MANIFEST_SHA}
                import torch
                from experiments import shared_state_continuation as bridge, shared_state_student as student
                from experiments import shared_state_targets as targets, execution_profile as execution
                from experiments import foundation_layout_training as training
                from experiments.foundation_layout_prepared import PreparedLayoutOwner, evidence_sha256
                from experiments.shared_state_training import SharedStateKernel
                from experiments.sequence_student import SequenceConfig
                from brain_in_computer.dialogue_student import checkpoint_digest
                torch.set_num_threads(1); torch.set_num_interop_threads(1)
                execution.configure_strict_profile(); boundary("runtime")
                report["runtime"] = execution.runtime_profile()
                torch.cuda.reset_peak_memory_stats()
                sources = bridge.source_hashes()
                sources["experiments/shared_state_continuation_gpu_proof.py"] = expected_source_sha256
                for index,(name,pin) in enumerate(sorted(sources.items())):
                    copy = f"source-{index:03d}.py"
                    if publish(here/copy,(ROOT/name).read_bytes(),raw=True) != pin: raise ValueError("source copy differs")
                    report["source_sha256"][name] = pin; report["source_snapshots"][name] = copy
                config = SequenceConfig(width=192,layers=4,heads=4,feedforward=768,max_turns=12)
                def before(model,args,kwargs):
                    boundary("forward"); calls["forwards"]["attempts"] += 1
                    if calls["forwards"]["attempts"] > 9: raise AssertionError("forward ceiling")
                    if args or set(kwargs) != {"token_ids","eos_positions","valid_mask","lengths","decoder_input_ids"}:
                        raise AssertionError("unexpected model inputs")
                    if kwargs["token_ids"].shape[0] != 32: raise AssertionError("production batch differs")
                def after(model,args,kwargs,output):
                    calls["forwards"]["completions"] += 1; report["forward_episodes"] += 32
                original_builder = student.build_shared_state_student
                def construct(*args,**kwargs):
                    model = original_builder(*args,**kwargs)
                    model.register_forward_pre_hook(before,with_kwargs=True)
                    model.register_forward_hook(after,with_kwargs=True)
                    return model
                with ExitStack() as stack:
                    for owner,name,key in ((student,"build_shared_state_student","models"),
                            (torch.optim.AdamW,"__init__","optimizers"),(torch.optim.AdamW,"step","updates"),
                            (torch.autograd,"backward","backwards"),(torch,"load","archive_loads"),
                            (targets,"pack_state_targets","target_packs"),(targets,"english_oracle","english_oracles"),
                            (targets,"abstract_oracle","typed_oracles"),
                            (training.curriculum.foundation,"generate_pair","canonical_generations")):
                        original = construct if key == "models" else getattr(owner,name)
                        stack.enter_context(patch.object(owner,name,limited(key,original)))
                    manifest=json.loads((ROOT/DATA/"manifest.json").read_bytes());images=[];labels=[]
                    report["data_records"] = []
                    for index in (0,1):
                        record=manifest["training"][index];path=ROOT/DATA/record["path"]
                        raw=path.read_bytes()
                        if record["global_cursor"] != index or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                            raise ValueError("cached image identity differs before load")
                        report["input_sha256"][str(path.relative_to(ROOT)).replace('\\','/')] = record["sha256"]
                        image=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
                        if image["bundle"]["bundle_id"] != index or image["expected_evidence"]["bundle_id"] != index:
                            raise ValueError("cached bundle index differs")
                        images.append(image);report["data_records"].append(record)
                        labels.append({family:targets.pack_state_targets(rows) for family,rows in image["bundle"]["families"].items()})
                        for rows in image["bundle"]["families"].values():
                            report["target_rows"] += len(rows);report["target_turns"] += sum(len(row["turns"]) for row in rows)
                    def owner():
                        value=PreparedLayoutOwner(config=config,layout="original",micro_batch_size=32);owners.append(value);return value
                    def token(owner,index):
                        image=images[index]
                        return owner.prepare(image["bundle"],expected_evidence=image["expected_evidence"],
                            expected_evidence_sha256=evidence_sha256(image["expected_evidence"]))
                    model=student.build_shared_state_student(852610099,device="cuda:0",config=config)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
                    reference=SharedStateKernel(model,optimizer,config=config,layout="original",micro_batch_size=32,
                                               objective_id=training.OBJECTIVE_ID,auxiliary_weight=.3)
                    reference_owner=owner()
                    report["steps"].append(reference.step(token(reference_owner,0),state_targets=labels[0],deadline=started+180))
                    boundary("synthetic_checkpoint")
                    saved=reference.snapshot()
                    payload=dict(schema=bridge.PRODUCER_SCHEMA,launch_sha256="b"*64,arm="slow",step=1,
                        architecture=student.ARCHITECTURE,learning_rate=.0003,auxiliary_weight=.3,runtime=report["runtime"],
                        learner=saved,weights_sha256=checkpoint_digest(model))
                    identity=json.loads(json.dumps(bridge.checkpoint_identity(payload)))
                    stream=io.BytesIO();torch.save(payload,stream);archive=stream.getvalue()
                    archive_pin=publish(here/"synthetic-checkpoint.pt",archive,raw=True)
                    boundary("restore")
                    restored=bridge.SharedStateContinuation.from_checkpoint(archive,expected_sha256=archive_pin,
                        expected_identity=identity,device="cuda:0")
                    report["restore"] = restored.last_restore_report
                    if not bridge._same(saved,restored._kernel.snapshot()): raise AssertionError("restored source differs")
                    report["steps"].append(reference.step(token(reference_owner,1),state_targets=labels[1],deadline=started+180))
                    restored_owner=owner()
                    report["steps"].append(restored.step(token(restored_owner,1),state_targets=labels[1],deadline=started+180))
                    boundary("exact_comparison");torch.cuda.synchronize();execution.assert_strict_profile()
                    left,right=reference.snapshot(),restored._kernel.snapshot()
                    for key in ("weights","optimizer","recipe","cursor","evidence"):
                        if not bridge._same(left[key],right[key]): raise AssertionError("exact next state differs: "+key)
                    for key in ("work","state_work"):
                        if not bridge._same(left["accounting"][key],right["accounting"][key]): raise AssertionError("exact accounting differs: "+key)
                    for key in ("loss","action_loss","reply_loss","observation_language_loss","state_loss","total_loss","microbatches","bundle_evidence"):
                        if not bridge._same(reference.last_report[key],restored.last_report[key]): raise AssertionError("next loss/evidence differs: "+key)
                    report["equivalence"] = dict(weights=True,full_adamw=True,recipe=True,evidence=True,work=True,state_work=True,losses=True)
                    report["reference_accounting"]=reference.accounting;report["restored_accounting"]=restored.accounting
                    report["final_weights_sha256"]=checkpoint_digest(model)
                if any(calls[k] != dict(attempts=v,completions=v) for k,v in LIMITS.items()): raise AssertionError("actual work differs from fixed scope")
                if report["forward_episodes"] != 288: raise AssertionError("physical episode work differs")
                boundary("final_authentication")
                if not all(sha(ROOT/name)==pin and sha(here/report["source_snapshots"][name])==pin for name,pin in sources.items()):
                    raise ValueError("proof source changed")
                report.update(status="passed",active_operation=None,source_and_copies_unchanged=True)
            except BaseException as error:
                caught=traceback.format_exc();stderr.write(caught)
                report.update(error=repr(error),traceback=caught)
                if hasattr(error,"continuation_report"):report["failed_continuation_report"]=error.continuation_report
            finally:
                for item in owners:item.close()
                report["owners"]=[item.report() for item in owners]
                if "torch" in locals() and torch.cuda.is_initialized():
                    torch.cuda.synchronize();report["peak_gpu_allocated_bytes"]=torch.cuda.max_memory_allocated()
                    report["peak_gpu_reserved_bytes"]=torch.cuda.max_memory_reserved()
                report.update(actual_calls=calls,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
                stdout.flush();stderr.flush()
    report["artifact_sha256"]={p.name:sha(p) for p in here.iterdir() if p.is_file()}
    pin=publish(here/"report.json",report)
    print(json.dumps(dict(status=report["status"],report_sha256=pin,wall_seconds=report["wall_seconds"],actual_calls=calls)))
    return 0 if report["status"]=="passed" else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("directory");parser.add_argument("--expected-source-sha256",required=True)
    args=parser.parse_args();raise SystemExit(run(args.directory,expected_source_sha256=args.expected_source_sha256))
