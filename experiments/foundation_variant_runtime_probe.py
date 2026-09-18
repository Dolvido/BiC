"""One-shot, separate-process execution proof for six foundation variants.

The CUDA contract is eight updates, split after three, for both architectures
and three learning rates. Four workers perform 144 updates / 13,824 episodes.
An authenticated finite plan index is constructed once per process and reused.
Evaluation checks exact teacher-free outputs, not capability or model selection.
CPU fixtures are engineering evidence only and cannot satisfy load_proof.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
import uuid

import torch

from brain_in_computer.dialogue_student import ByteCodec, _exact_replies, _generate_reply_tokens, checkpoint_digest
from experiments import foundation_runtime_probe as core
from experiments.composition_evaluation import prediction_metrics
from experiments.foundation_admission import repair_plan
from experiments.foundation_curriculum import FAMILIES, REPLIES, generate_pair
from experiments.foundation_evaluation import FoundationBank
from experiments.foundation_evidence import json_digest, transcript_set
from experiments.foundation_plan import build_plan, materialize_pair
from experiments.foundation_plan_index import AuthenticatedPlanIndex
from experiments.foundation_variant_training import VariantFoundationTrainer, ARCHITECTURES, source_hashes as trainer_sources
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig


SCHEMA = "bic-foundation-variant-runtime-probe-v1"
ROOT = Path(__file__).resolve().parents[1]
PROCESS_INSTANCE = uuid.uuid4().hex
WORKERS = ("uninterrupted", "split-first", "resumed", "repeat")
RATES = (.0003, .001, .003)
CASES = tuple({"id": f"{architecture}-{label}", "architecture": architecture, "learning_rate": rate}
              for architecture in ("flat", "hierarchical")
              for label, rate in zip(("r0003", "r001", "r003"), RATES))
CUDA_CONFIG = SequenceConfig(width=192, layers=4, heads=4, feedforward=768, max_turns=12)
MODEL_SEED = 9911


def source_hashes():
    names = ("experiments/foundation_variant_runtime_probe.py", "experiments/foundation_runtime_probe.py",
             "experiments/foundation_evaluation.py", "experiments/composition_evaluation.py",
             "experiments/execution_profile.py", "docs/FOUNDATION_VARIANT_RUNTIME_PROTOCOL.md")
    return {**trainer_sources(), **{name: core._file_hash(ROOT/name) for name in names}}


def _contract(device, config, steps, midpoint, micro):
    if (device not in ("cpu", "cuda:0") or type(config) is not SequenceConfig or config.layers != 4
            or config.max_turns < 12 or type(steps) is not int or type(midpoint) is not int
            or not 1 <= midpoint < steps <= 8 or type(micro) is not int or micro < 2 or micro % 2):
        raise ValueError("explicit four-block config, complete microbatches and bounded split required")
    if device == "cuda:0" and (config != CUDA_CONFIG or (steps, midpoint, micro) != (8, 3, 32)):
        raise ValueError("strict CUDA proof requires width192, eight updates, split3 and micro32")
    return {"device": device, "config": asdict(config), "steps": steps, "midpoint": midpoint,
            "micro_batch_size": micro, "cases": list(CASES), "architectures": dict(ARCHITECTURES),
            "rates": list(RATES), "workers": list(WORKERS), "model_seed": MODEL_SEED,
            "order": "mixed", "planned_physical_updates": len(CASES)*3*steps,
            "planned_episode_exposures": len(CASES)*3*steps*3*micro,
            "checkpoint_comparison_ignored_fields": ["timing"],
            "output_comparison_ignored_fields": [], "automatic_promotion": False}


def _base(micro):
    return build_plan(seed=991000001, stage_updates=10, final_updates=6,
                      micro_batch_size=micro, rehearsal_every=2, ordering_seed=0)


def _history(base, steps):
    return sorted({transcript_digest(row) for bundle in base["schedules"]["mixed"][:steps]
                   for row in materialize_pair(base, bundle, "color", 0)})


def _evaluation_rows():
    return {f"{family}/d{depth}/t{turns}": generate_pair(family, 991100001+depth,
        depth=depth, turns=turns, split="dev", structure_split="shared" if depth < 2 else "train")
        for family, depth, turns in zip(FAMILIES, (0, 1, 2), (8, 10, 12))}


def _coverage(plan, steps, midpoint):
    prefix = plan["schedules"]["mixed"][:steps]
    if any(f"{bundle}/color/0" not in plan["admission"]["realization_attempts"] for bundle in prefix):
        raise ValueError("every executed bundle must exercise the naming override")
    return {"depths": sorted({plan["bundles"][str(i)]["depth"] for i in prefix}),
            "turns": sorted({plan["bundles"][str(i)]["turns"] for i in prefix}),
            "overridden_prefix_bundles": len(prefix), "before_split": midpoint,
            "after_split": steps-midpoint}


def _trainer(context, case, device, payload=None):
    p = context["protocol"]
    return VariantFoundationTrainer(context["plan"], "mixed", architecture=case["architecture"],
        seed=MODEL_SEED, config=SequenceConfig(**p["config"]), learning_rate=case["learning_rate"],
        admission_protected_transcripts=context["history"], protected_transcripts=context["protected"],
        admission_receipt=context["admission"], plan_index=context["index"], device=device, payload=payload)


def prepare(directory, *, device="cuda:0", config=None, steps=8, midpoint=3, micro_batch_size=32):
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=False)
    core._write(directory/"prepare-started.json", {"schema": SCHEMA, "utc": core._now(),
        "process": {"pid": os.getpid(), "instance": PROCESS_INSTANCE}})
    started = time.monotonic()
    try:
        config = CUDA_CONFIG if config is None and device == "cuda:0" else config
        contract = _contract(device, config, steps, midpoint, micro_batch_size)
        runtime = core._runtime(device); sources = source_hashes()
        base = _base(micro_batch_size); history = _history(base, steps)
        plan, admission = repair_plan(base, history)
        rows = _evaluation_rows()
        protected = sorted(set(history) | {transcript_digest(row) for bank in rows.values() for row in bank})
        banks = {name: FoundationBank(bank, role="dev", config=config) for name, bank in rows.items()}
        index = AuthenticatedPlanIndex(plan, admission_protected_transcripts=history,
            protected_transcripts=protected, admission_receipt=admission)
        coverage = _coverage(plan, steps, midpoint)
        if device == "cuda:0" and coverage["turns"] != [8, 10, 12]:
            raise ValueError("CUDA prefix must exercise all three utterance counts")
        values = {"base-plan.json": base, "plan.json": plan, "admission.json": admission,
            "history.json": history, "protected.json": protected, "evaluation-banks.json": rows,
            "index-identity.json": index.identity, "index-construction.json": index.construction}
        for name, value in values.items(): core._write(directory/name, value)
        context = {"protocol": contract, "plan": plan, "history": history, "protected": protected,
                   "admission": admission, "index": index}
        (directory/"initial").mkdir()
        for case in CASES:
            trainer = _trainer(context, case, "cpu")
            core._save(directory/"initial"/f"{case['id']}.pt", trainer.snapshot())
            del trainer
        inputs = {name: core._file_hash(directory/name) for name in values}
        inputs.update({f"initial/{c['id']}.pt": core._file_hash(directory/"initial"/f"{c['id']}.pt") for c in CASES})
        for name, digest in sources.items():
            destination = directory/"sources"/name; destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream: stream.write((ROOT/name).read_bytes())
            if core._file_hash(destination) != digest: raise ValueError("source changed while freezing")
        protocol = {"schema": SCHEMA, **contract, "execution_profile": runtime, "source_sha256": sources,
            "input_sha256": inputs, "coverage": coverage, "evaluation_identities": {n:b.identity for n,b in banks.items()},
            "prepare_process": {"pid": os.getpid(), "instance": PROCESS_INSTANCE},
            "scope": "Same-source local execution/teacher-free output repeatability; no capability, selection, learning-benefit or cross-machine claim."}
        if source_hashes() != sources: raise ValueError("source drift during preparation")
        core._write(directory/"protocol.json", protocol)
        core._write(directory/"prepared.json", {"schema": SCHEMA, "protocol_sha256": core._file_hash(directory/"protocol.json"),
            "preparation_seconds": time.monotonic()-started, "index_construction": index.construction})
        return protocol
    except BaseException as error:
        core._write(directory/"prepare-failed.json", {"error": core._error(error), "wall_seconds": time.monotonic()-started})
        raise


def _sealed(directory, protocol, protocol_hash):
    if core._file_hash(directory/"protocol.json") != protocol_hash or source_hashes() != protocol["source_sha256"]:
        raise ValueError("probe protocol or source identity changed")
    for name, digest in protocol["source_sha256"].items():
        if core._file_hash(directory/"sources"/name) != digest: raise ValueError("source snapshot changed")
    for name, digest in protocol["input_sha256"].items():
        if core._file_hash(directory/name) != digest: raise ValueError("frozen input artifact changed")


def _inputs(directory):
    p = core._read(directory/"protocol.json"); digest = core._file_hash(directory/"protocol.json")
    if p.get("schema") != SCHEMA or core._read(directory/"prepared.json").get("protocol_sha256") != digest:
        raise ValueError("completed frozen preparation required")
    expected = _contract(p["device"], SequenceConfig(**p["config"]), p["steps"], p["midpoint"], p["micro_batch_size"])
    if any(not core._same(p.get(k), v) for k,v in expected.items()): raise ValueError("declared six-case workload differs")
    names = {"base-plan.json", "plan.json", "admission.json", "history.json", "protected.json",
             "evaluation-banks.json", "index-identity.json", "index-construction.json"} | {f"initial/{c['id']}.pt" for c in CASES}
    if set(p["input_sha256"]) != names: raise ValueError("frozen input set differs")
    _sealed(directory, p, digest)
    base, plan = core._read(directory/"base-plan.json"), core._read(directory/"plan.json")
    history, protected = core._read(directory/"history.json"), core._read(directory/"protected.json")
    rows, admission = core._read(directory/"evaluation-banks.json"), core._read(directory/"admission.json")
    if (base != _base(p["micro_batch_size"]) or history != _history(base, p["steps"])
            or rows != _evaluation_rows() or protected != sorted(transcript_set(protected))
            or set(protected) != set(history) | {transcript_digest(r) for bank in rows.values() for r in bank}):
        raise ValueError("canonical probe recipes or protected bank differ")
    index = AuthenticatedPlanIndex(plan, admission_protected_transcripts=history,
        protected_transcripts=protected, admission_receipt=admission)
    if index.identity != core._read(directory/"index-identity.json") or _coverage(plan,p["steps"],p["midpoint"]) != p["coverage"]:
        raise ValueError("admitted index or override coverage differs")
    banks = {name: FoundationBank(bank, role="dev", config=SequenceConfig(**p["config"])) for name,bank in rows.items()}
    if {name:bank.identity for name,bank in banks.items()} != p["evaluation_identities"]:
        raise ValueError("prepared evaluation bank identity differs")
    return {"protocol": p, "protocol_hash": digest, "plan": plan, "history": history, "protected": protected,
            "admission": admission, "index": index, "banks": banks}


def _observe(trainer, banks, journal):
    """One BOS-only forward and actual greedy reply generation per whole bank."""
    model, started = trainer.model, time.monotonic()
    before, cursor = checkpoint_digest(model), trainer.cursor
    modes = [(module,module.training) for module in model.modules()]
    output, episodes, turns = {}, 0, 0
    cost={"wall_seconds":0.,"encoder_episodes":0,"free_reply_turns":0}
    try:
        model.eval()
        with torch.inference_mode():
            for name, bank in banks.items():
                count, length = bank._targets.shape
                core._append(journal, {"event":"eval_started", "cursor":cursor, "bank":name,
                                      "episodes":count, "turns":count*length})
                encoded, generated_ok = False, False
                try:
                    device = next(model.parameters()).device
                    inputs = {key:value.clone().to(device) for key,value in bank._inputs["normal"].items()}
                    predicted = model(**inputs, decoder_input_ids=torch.full((count,length,1),ByteCodec.BOS,dtype=torch.long,device=device))
                    encoded = True
                    generated = _generate_reply_tokens(model,predicted["production_context"].flatten(0,1)); generated_ok = True
                    reply_correct = _exact_replies(generated,bank._reply_targets.flatten(0,1).to(device)).reshape(count,length).cpu()
                    reply_actions=[]
                    for tokens in generated.cpu().tolist():
                        try: text=model.codec.decode(tokens)
                        except (ValueError,UnicodeError): text=None
                        reply_actions.append(REPLIES.index(text) if text in REPLIES else -1)
                    logits=predicted["logits"].cpu()
                    output[name]={"bank":bank.identity,"action_logits":logits,
                        "bos_reply_logits":predicted["language_logits"].cpu(), "generated_reply_tokens":generated.cpu(),
                        "metrics":prediction_metrics(logits,bank._targets,reply_correct=reply_correct,
                            reply_actions=torch.tensor(reply_actions).reshape(count,length))}
                    core._check_finite_tree(output[name],"probe evaluation output")
                    core._append(journal,{"event":"eval_completed","cursor":cursor,"bank":name,
                        "episodes":count,"turns":count*length,"encoder_completed":True,"generation_completed":True})
                    episodes+=count; turns+=count*length
                except BaseException as error:
                    core._append(journal,{"event":"eval_failed","cursor":cursor,"bank":name,
                        "episodes":count,"turns":count*length,"encoder_completed":encoded,
                        "generation_completed":generated_ok,"error":core._error(error)})
                    raise
        cost.update(encoder_episodes=episodes,free_reply_turns=turns)
        return {"cursor":cursor,"weights_sha256":before,"decoder_prefix":"BOS only",
                "teacher_used_for_policy":False,"outputs":output}, cost
    finally:
        for module,mode in modes: module.training=mode
        if checkpoint_digest(model)!=before or trainer.cursor!=cursor:
            raise RuntimeError("probe evaluation changed learner state")
        cost["wall_seconds"]=time.monotonic()-started


def _worker_receipt(directory,name):
    path=directory/name; record=core._read(path/"result.json")
    if record.get("schema")!=SCHEMA or record.get("status")!="completed": raise ValueError("complete preceding worker required")
    if record.get("artifact_sha256")!=core._artifact_hashes(path,("result.json",)):
        raise ValueError("worker artifacts changed")
    return record


def worker(directory,name):
    directory=Path(directory).resolve()
    if name not in WORKERS: raise ValueError("unknown worker")
    destination=directory/name; destination.mkdir(exist_ok=False)
    started=time.monotonic(); context=trainer=None
    record={"schema":SCHEMA,"status":"running","name":name,"started_utc":core._now(),
            "process":{"pid":os.getpid(),"instance":PROCESS_INSTANCE},"parents":{},"cases":{}}
    core._write(destination/"started.json",record)
    try:
        context=_inputs(directory); p=context["protocol"]
        if p["prepare_process"]["instance"]==PROCESS_INSTANCE: raise ValueError("preparation and workers need separate processes")
        record["protocol_sha256"]=context["protocol_hash"]
        for prior_name in WORKERS[:WORKERS.index(name)]:
            prior=_worker_receipt(directory,prior_name)
            if prior["process"]["instance"]==PROCESS_INSTANCE: raise ValueError("workers need separate processes")
            record["parents"][f"{prior_name}/result.json"]=core._file_hash(directory/prior_name/"result.json")
        profile=core._runtime(p["device"])
        if profile!=p["execution_profile"]: raise ValueError("worker execution profile differs")
        record["execution_profile"]=profile; record["index_construction"]=context["index"].construction
        if p["device"]=="cuda:0": torch.cuda.reset_peak_memory_stats()
        for case in CASES:
            case_path=destination/case["id"]; case_path.mkdir(); case_start=time.monotonic()
            record["active_case"]=case["id"]
            core._write(case_path/"started.json",{"case":case,"utc":core._now()})
            payload_path=directory/"split-first"/case["id"]/"final.pt" if name=="resumed" else directory/"initial"/f"{case['id']}.pt"
            record["parents"][payload_path.relative_to(directory).as_posix()]=core._file_hash(payload_path)
            payload=core._load(payload_path)
            trainer=_trainer(context,case,p["device"],payload=payload)
            loaded=trainer.snapshot(); core._save(case_path/("loaded-midpoint.pt" if name=="resumed" else "initial.pt"),loaded)
            if not core._same(payload,loaded): raise ValueError("exact complete initial/midpoint reload failed")
            begin=trainer.cursor; target=p["midpoint"] if name=="split-first" else p["steps"]
            evaluations={}
            def capture():
                observed,cost=_observe(trainer,context["banks"],case_path/"evaluation.jsonl")
                core._save(case_path/f"evaluation-{trainer.cursor:03d}.pt",observed)
                evaluations[str(trainer.cursor)]=cost
            if name=="resumed": capture()
            while trainer.cursor<target:
                core._append(case_path/"steps.jsonl",{"event":"started","cursor":trainer.cursor,
                    "bundle_id":context["plan"]["schedules"]["mixed"][trainer.cursor],"utc":core._now()})
                try: report=trainer.step()
                except BaseException:
                    core._append(case_path/"steps.jsonl",{"event":"reported","report":copy.deepcopy(trainer.last_report)})
                    raise
                core._append(case_path/"steps.jsonl",{"event":"reported","report":report})
                if trainer.cursor==p["midpoint"]:
                    core._save(case_path/"midpoint.pt",trainer.snapshot()); capture()
            core._save(case_path/"final.pt",trainer.snapshot())
            if str(target) not in evaluations: capture()
            record["cases"][case["id"]]={"start_cursor":begin,"final_cursor":trainer.cursor,
                "setup":trainer.setup_report,"restore_seconds":trainer.last_restore_seconds,
                "evaluation":evaluations,"wall_seconds":time.monotonic()-case_start}
            trainer=None
        if core._runtime_after(p["device"])!=profile: raise ValueError("execution profile drifted")
        record["execution_profile_after"]=profile
        _sealed(directory,p,context["protocol_hash"])
        record["status"]="completed"
        record.pop("active_case",None)
    except BaseException as error:
        record.update(status="failed",error=core._error(error))
        if trainer is not None: record["last_trainer_report"]=copy.deepcopy(trainer.last_report)
    finally:
        if context is not None and context["protocol"]["device"]=="cuda:0":
            try:
                record["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated()
                record["peak_cuda_reserved_bytes"]=torch.cuda.max_memory_reserved()
            except BaseException as error: record["memory_measurement_error"]=core._error(error)
        record.update(wall_seconds=time.monotonic()-started,completed_utc=core._now())
        record["artifact_sha256"]=core._artifact_hashes(destination)
        core._write(destination/"result.json",record)
    return record


def _events(path):
    return [core._read_line(line) for line in path.read_text(encoding="utf8").splitlines()] if path.exists() else []


def _physical(directory):
    result={"known_completed_optimizer_updates":0,"completed_microbatch_episodes":0,
        "returned_candidate_episodes":0,"neural_attempted_episodes":0,
        "optimizer_completion_unknown_attempts":0,"partial_materialization_unknown_attempts":0,
        "unreported_started_attempts":[],"reported_step_seconds":0.,"reported_materialization_seconds_included":0.,
        "known_encoder_episodes":0,"known_free_reply_turns":0,"evaluation_incomplete_or_unknown":[]}
    for name in WORKERS:
        for case in CASES:
            path=directory/name/case["id"]; events=_events(path/"steps.jsonl")
            if len(events)%2: result["unreported_started_attempts"].append(f"{name}/{case['id']}")
            for event in events:
                if event.get("event")!="reported": continue
                r=event.get("report")
                if r is None:
                    result["optimizer_completion_unknown_attempts"]+=1;result["partial_materialization_unknown_attempts"]+=1;continue
                result["known_completed_optimizer_updates"]+=r.get("physical_optimizer_updates") or 0
                result["optimizer_completion_unknown_attempts"]+=int(r.get("physical_optimizer_updates") is None)
                result["partial_materialization_unknown_attempts"]+=int(r.get("drawn_episode_exposures") is None)
                for dst,src in (("completed_microbatch_episodes","completed_microbatch_episode_exposures"),
                    ("returned_candidate_episodes","drawn_episode_exposures"),("neural_attempted_episodes","neural_attempted_episode_exposures"),
                    ("reported_step_seconds","step_seconds"),("reported_materialization_seconds_included","materialization_seconds")):
                    result[dst]+=r.get(src) or 0
            evaluations=_events(path/"evaluation.jsonl")
            if len(evaluations)%2: result["evaluation_incomplete_or_unknown"].append(f"{name}/{case['id']}/started")
            for event in evaluations:
                if event.get("event") not in ("eval_completed","eval_failed"): continue
                if event.get("encoder_completed"): result["known_encoder_episodes"]+=event["episodes"]
                if event.get("generation_completed"): result["known_free_reply_turns"]+=event["turns"]
                if event["event"]=="eval_failed": result["evaluation_incomplete_or_unknown"].append(f"{name}/{case['id']}/failed")
    return result


def _validate_journal(events, records, start, end, micro):
    if len(events)!=2*(end-start): raise ValueError("incomplete or extra step journal")
    reports=[]
    for cursor in range(start,end):
        begun,done=events[2*(cursor-start):2*(cursor-start)+2]; row=done.get("report",{}); expected=records[cursor]
        if any(type(item.get(key)) is not int for item in (begun,row) for key in ("cursor","bundle_id")):
            raise ValueError("journal integer identity types differ")
        if (begun.get("event")!="started" or begun.get("cursor")!=cursor or begun.get("bundle_id")!=expected["bundle_id"]
                or done.get("event")!="reported" or row.get("cursor")!=cursor+1 or row.get("bundle_id")!=expected["bundle_id"]):
            raise ValueError("journal cursor or bundle differs")
        for key,value in (("physical_optimizer_updates",1),("retained_optimizer_updates",1),
            ("drawn_microbatches",3),("neural_attempted_microbatches",3),("completed_microbatches",3),
            ("drawn_episode_exposures",3*micro),("neural_attempted_episode_exposures",3*micro),("completed_microbatch_episode_exposures",3*micro)):
            if type(row.get(key)) is not int or row[key]!=value: raise ValueError("journal physical counts differ")
        for key in ("wall_seconds","step_seconds","materialization_seconds","loss","action_loss","reply_loss","observation_language_loss"):
            core._nonnegative(row[key],key)
        if not row["materialization_seconds"]<=row["step_seconds"]<=row["wall_seconds"]: raise ValueError("timing scopes differ")
        if [m.get("family") for m in row["microbatches"]]!=list(FAMILIES): raise ValueError("microbatch family order differs")
        for m in row["microbatches"]:
            e=expected["families"][m["family"]]
            if any(m[k]!=e[k] for k in ("rows_sha256","recipes_sha256","exposures")) or m["depth"]!=expected["depth"] or m["turns"]!=expected["turns"]:
                raise ValueError("canonical sampled evidence differs")
        reports.append(row)
    return reports


def _verified(directory):
    context=_inputs(directory); p=context["protocol"]; workers={n:_worker_receipt(directory,n) for n in WORKERS}
    instances={r["process"]["instance"] for r in workers.values()}
    if (len(instances)!=4 or p["prepare_process"]["instance"] in instances or PROCESS_INSTANCE in instances
            or PROCESS_INSTANCE==p["prepare_process"]["instance"]):
        raise ValueError("prepare, training workers and verifier must be separate processes")
    replay=context["index"].replay("mixed",p["steps"],include_bundles=True)
    for name,r in workers.items():
        if (r["protocol_sha256"]!=context["protocol_hash"] or r["execution_profile"]!=p["execution_profile"]
                or r["execution_profile_after"]!=p["execution_profile"] or set(r["cases"])!={c["id"] for c in CASES}):
            raise ValueError("worker protocol/runtime/case set differs")
        for path,digest in r["parents"].items():
            if core._file_hash(directory/path)!=digest: raise ValueError("worker parent changed")
    comparisons={}
    for case in CASES:
        cid=case["id"]; paths={n:directory/n/cid for n in WORKERS}; reports={}
        checker=_trainer(context,case,"cpu")
        initial=core._load(directory/"initial"/f"{cid}.pt"); checker.restore(initial)
        for name in WORKERS:
            begin=p["midpoint"] if name=="resumed" else 0; end=p["midpoint"] if name=="split-first" else p["steps"]
            cost=workers[name]["cases"][cid]
            if cost["start_cursor"]!=begin or cost["final_cursor"]!=end: raise ValueError("case bounds differ")
            reports[name]=_validate_journal(_events(paths[name]/"steps.jsonl"),replay["bundles"],begin,end,p["micro_batch_size"])
            for filename in ("initial.pt","loaded-midpoint.pt","midpoint.pt","final.pt"):
                path=paths[name]/filename
                if path.exists(): checker.restore(core._load(path))
        mid=core._load(paths["uninterrupted"]/"midpoint.pt"); end=core._load(paths["uninterrupted"]/"final.pt")
        flags={"initial_exact":all(core._same(initial,core._load(paths[n]/"initial.pt")) for n in ("uninterrupted","split-first","repeat")),
            "midpoint_exact":core._difference(mid,core._load(paths["split-first"]/"final.pt"))["exact"],
            "repeat_midpoint_exact":core._difference(mid,core._load(paths["repeat"]/"midpoint.pt"))["exact"],
            "immediate_reload_including_timing_exact":core._same(core._load(paths["split-first"]/"final.pt"),core._load(paths["resumed"]/"loaded-midpoint.pt")),
            "resumed_endpoint_exact":core._difference(end,core._load(paths["resumed"]/"final.pt"))["exact"],
            "repeat_endpoint_exact":core._difference(end,core._load(paths["repeat"]/"final.pt"))["exact"],
            "canonical_endpoint_exact":core._same(end["evidence"],replay["evidence"])}
        for name in WORKERS:
            prior=reports["split-first"] if name=="resumed" else []
            for filename,cursor in (("midpoint.pt",p["midpoint"]),("final.pt",p["midpoint"] if name=="split-first" else p["steps"])):
                path=paths[name]/filename
                if not path.exists(): continue
                payload=core._load(path)
                if payload["cursor"]!=cursor: raise ValueError("checkpoint cursor differs")
                used=(prior+reports[name])[:cursor]
                for key,field in (("retained_step_seconds","step_seconds"),("materialization_seconds_included_in_step","materialization_seconds")):
                    if not math.isclose(payload["timing"][key],sum(r[field] for r in used),rel_tol=1e-10,abs_tol=1e-8):
                        raise ValueError("checkpoint timing differs from physical journal")
        for cursor,names in ((p["midpoint"],WORKERS),(p["steps"],("uninterrupted","resumed","repeat"))):
            reference=core._load(paths["uninterrupted"]/f"evaluation-{cursor:03d}.pt")
            state=mid if cursor==p["midpoint"] else end
            checker.restore(state)
            if (reference["cursor"]!=cursor or reference["weights_sha256"]!=checkpoint_digest(checker.model)
                    or reference["decoder_prefix"]!="BOS only" or reference["teacher_used_for_policy"] is not False
                    or {n:o["bank"] for n,o in reference["outputs"].items()}!=p["evaluation_identities"]):
                raise ValueError("evaluation producer/bank boundary differs")
            flags[f"outputs_{cursor}_exact"]=all(core._same(reference,core._load(paths[n]/f"evaluation-{cursor:03d}.pt")) for n in names)
        comparisons[cid]=flags
        del checker
    physical=_physical(directory)
    if (physical["known_completed_optimizer_updates"]!=p["planned_physical_updates"]
            or physical["completed_microbatch_episodes"]!=p["planned_episode_exposures"]
            or physical["optimizer_completion_unknown_attempts"] or physical["unreported_started_attempts"]
            or physical["partial_materialization_unknown_attempts"] or physical["evaluation_incomplete_or_unknown"]):
        raise ValueError("incomplete or different physical work")
    if physical["known_encoder_episodes"]!=252 or physical["known_free_reply_turns"]!=2520:
        raise ValueError("teacher-free evaluation exposure differs")
    _sealed(directory,p,context["protocol_hash"])
    passed=all(all(flags.values()) for flags in comparisons.values())
    return {"schema":SCHEMA,"status":("passed" if p["device"]=="cuda:0" else "passed_cpu_engineering_only") if passed else "failed_equality",
        "config":p["config"],"rates":p["rates"],"architectures":p["architectures"],"cases":comparisons,
        "source_sha256":p["source_sha256"],"input_sha256":p["input_sha256"],"protocol_sha256":context["protocol_hash"],
        "execution_profile":p["execution_profile"],"coverage":p["coverage"],"physical":physical,
        "physical_updates":physical["known_completed_optimizer_updates"],"physical_episode_exposures":physical["completed_microbatch_episodes"],
        "workers":workers,"verification_index_construction":context["index"].construction,
        "worker_wall_seconds_sum":sum(r["wall_seconds"] for r in workers.values()),
        "automatic_promotion":False,"neural_work_in_verification":False,
        "scope":"Exact same-source local tensor/optimizer/input and BOS-generated-output proof only. Not cross-machine repeatability or learned capability. Evaluation cost is separate; materialization is included in step time, setup/restore/evaluation/save in worker wall."}


def verify(directory):
    directory=Path(directory).resolve(); started=time.monotonic()
    core._write(directory/"verification-started.json",{"utc":core._now(),"process":{"pid":os.getpid(),"instance":PROCESS_INSTANCE}})
    try: record=_verified(directory)
    except BaseException as error:
        record={"schema":SCHEMA,"status":"failed","error":core._error(error),"automatic_promotion":False}
        try: record["physical"]=_physical(directory)
        except BaseException as failure: record["accounting_error"]=core._error(failure)
    record.update(verification_seconds=time.monotonic()-started,completed_utc=core._now())
    record["artifact_sha256"]=core._artifact_hashes(directory,("probe.json",))
    core._write(directory/"probe.json",record); return record


_PROOF_CACHE = None  # One authenticated JSON result; no tensors or persistent cache.


def _proof_state(directory,expected):
    path=directory/"probe.json"; digest=core._file_hash(path); value=core._read(path)
    sources=source_hashes(); artifacts=core._artifact_hashes(directory,("probe.json",))
    if (digest!=core._file_hash(path) or value.get("source_sha256")!=sources
            or value.get("artifact_sha256")!=artifacts):
        raise ValueError("runtime proof artifacts or sources changed")
    return value, core._json({"directory":str(directory),"config":expected,
        "proof_sha256":digest,"source_sha256":sources,"artifact_sha256":artifacts})


def load_proof(directory,config):
    global _PROOF_CACHE
    directory=Path(directory).resolve(); expected=asdict(config) if type(config) is SequenceConfig else config
    value,identity=_proof_state(directory,expected)
    if (value.get("status")!="passed" or value.get("config")!=expected or expected!=asdict(CUDA_CONFIG)
            or value.get("rates")!=list(RATES) or value.get("architectures")!=ARCHITECTURES
            or value.get("physical_updates")!=144 or value.get("physical_episode_exposures")!=13824):
        raise ValueError("complete matching six-case strict CUDA proof required; CPU fixtures are not proof")
    if _PROOF_CACHE is None or _PROOF_CACHE[0]!=identity:
        _PROOF_CACHE=None
        verified=_verified(directory)
        for key in ("status","config","rates","architectures","cases","source_sha256","input_sha256","protocol_sha256",
                    "execution_profile","coverage","physical","physical_updates","physical_episode_exposures","automatic_promotion"):
            if not core._same(value[key],verified[key]): raise ValueError("proof differs from authenticated complete evidence")
    _,after=_proof_state(directory,expected)
    if after!=identity:
        _PROOF_CACHE=None
        raise ValueError("runtime proof changed during authentication")
    _PROOF_CACHE=(identity,core._json(value))
    return json.loads(_PROOF_CACHE[1])


def main():
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest="command",required=True)
    for name in ("prepare","worker","verify"):
        command=commands.add_parser(name); command.add_argument("--output",required=True)
        if name=="worker": command.add_argument("--name",choices=WORKERS,required=True)
    args=parser.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.command=="prepare":
        record=prepare(args.output); print(core._json({"prepared":True,"updates":record["planned_physical_updates"]})); return 0
    record=worker(args.output,args.name) if args.command=="worker" else verify(args.output)
    print(core._json({key:record.get(key) for key in ("status","physical_updates","error")}))
    return 0 if record["status"] in ("completed","passed","passed_cpu_engineering_only") else 1


if __name__=="__main__":
    raise SystemExit(main())
