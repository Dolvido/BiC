"""One prospective architecture comparison; no automatic promotion or retries.

Independent data verification precedes gradients. Architecture workers reuse one
authenticated process index across three fresh learners. Canonical checkpoint
verification precedes any phase scoring. The audit is read for scoring only
after all six main endpoints are sealed. Original foundation sources stay frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments import foundation_study as old
from experiments import foundation_results as old_results
from experiments import foundation_variant_data as data_api
from experiments.foundation_evidence import json_digest
from experiments.foundation_variant_training import VariantFoundationTrainer, source_hashes as trainer_sources
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.hierarchical_sequence_student import build_hierarchical_sequence_student
from experiments.train_cognitive import atomic_checkpoint, atomic_json

SCHEMA = "bic-foundation-variant-study-v1"
ARCHITECTURES = ("flat", "hierarchical")
RATES = (.0003, .001, .003)
SEEDS = (8462, 8463, 8464)
CALIBRATION_SEED = 8461
TOTALS = {"calibration": 510, "main": 3072}
STEPS = {"calibration": (0, 510), "main": (0,384,768,1152,1536,1920,2304,2688,3072)}
CONFIG = SequenceConfig(width=192, layers=4, heads=4, feedforward=768, max_turns=12)


def contract():
    return dict(schema=SCHEMA, architectures=list(ARCHITECTURES), rates=list(RATES),
        main_seeds=list(SEEDS), calibration_seed=CALIBRATION_SEED, order="curriculum",
        totals=TOTALS, checkpoints={k:list(v) for k,v in STEPS.items()}, config=asdict(CONFIG),
        evaluation_during_training=False, automatic_promotion=False, automatic_retry=False,
        formal_updates=21492, formal_episode_exposures=2063232,
        calibration="Full90 dev cells; equal domains/cells; mean paired action, paired reply, target-balanced known; round12 then lower rate",
        primary_checkpoint=3072, replication="Initialization only; shared lessons and banks",
        matching="Same parameters, initial tensors, lessons, order, objective and search budget; not matched computation",
        interpretation="docs/FOUNDATION_VARIANT_STUDY_PROTOCOL.md")


def source_hashes():
    from experiments.foundation_variant_runtime_probe import source_hashes as runtime_sources
    root = Path(__file__).resolve().parents[1]
    names = ("experiments/foundation_variant_study.py", "experiments/foundation_variant_data.py",
        "experiments/foundation_variant_analysis.py", "experiments/foundation_variant_runtime_probe.py",
        "docs/FOUNDATION_VARIANT_STUDY_PROTOCOL.md")
    return {**old.source_hashes(), **trainer_sources(), **runtime_sources(),
            **{name:old.file_hash(root/name) for name in names}}


def _same(a,b,message):
    if json_digest(a) != json_digest(b):
        raise ValueError(message)


def _files(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix():old.file_hash(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def _sources(protocol):
    _same(source_hashes(), protocol["source_sha256"], "variant study sources changed")


def _proof(protocol):
    from experiments.foundation_variant_runtime_probe import load_proof
    proof = load_proof(protocol["proof_directory"], CONFIG)
    _same(json_digest(proof), protocol["proof_sha256"], "variant execution proof changed")
    _same(proof["execution_profile"], protocol["execution_profile"], "variant proof runtime changed")
    return proof


def prepare(directory, foundation_directory, proof_directory):
    from experiments.foundation_variant_runtime_probe import load_proof
    started = time.monotonic()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    atomic_json(directory/"preparation-started.json",
                dict(started_utc=old.utc(), contract=contract(), source_sha256=sources))
    proof = load_proof(proof_directory, CONFIG)
    if set(proof["architectures"]) != set(ARCHITECTURES) or set(proof["rates"]) != set(RATES):
        raise ValueError("all declared architecture/rate execution proofs required")
    data_api.prepare(directory/"data", foundation_directory)
    initial = {}
    for seed in (CALIBRATION_SEED, *SEEDS):
        flat = build_sequence_student(seed, config=CONFIG)
        hierarchy = build_hierarchical_sequence_student(seed, config=CONFIG)
        if sum(p.numel() for p in flat.parameters()) != 2221738:
            raise ValueError("prospective parameter inventory differs")
        if set(flat.state_dict()) != set(hierarchy.state_dict()) or any(
                not torch.equal(value, hierarchy.state_dict()[key]) for key,value in flat.state_dict().items()):
            raise ValueError("paired architecture initial tensors differ")
        initial[str(seed)] = checkpoint_digest(flat)
        del flat, hierarchy
    _same(sources, source_hashes(), "sources changed during preparation")
    root = Path(__file__).resolve().parents[1]
    for name,digest in sources.items():
        payload = (root/name).read_bytes()
        target = directory/"source"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(payload)
        if old.file_hash(target) != digest:
            raise ValueError("frozen source copy differs")
    protocol = dict(contract=contract(), source_sha256=sources, data_files_sha256=_files(directory/"data"),
        proof_directory=str(Path(proof_directory).resolve()), proof_sha256=json_digest(proof),
        execution_profile=proof["execution_profile"], initial_weights_sha256=initial,
        created_utc=old.utc())
    atomic_json(directory/"protocol.json",protocol)
    atomic_json(directory/"preparation.json",dict(status="completed",wall_seconds=time.monotonic()-started,
        protocol_sha256=old.file_hash(directory/"protocol.json"),neural_training_or_inference=False))
    return protocol


def load_protocol(directory):
    directory = Path(directory)
    protocol = old.read(directory/"protocol.json")
    _same(protocol["contract"],contract(),"variant study contract differs")
    _sources(protocol)
    for name,digest in protocol["source_sha256"].items():
        if old.file_hash(directory/"source"/name) != digest:
            raise ValueError("preserved variant source differs")
    _same(_files(directory/"data"),protocol["data_files_sha256"],"frozen variant data differs")
    _proof(protocol)
    prep = old.read(directory/"preparation.json")
    if prep.get("status") != "completed" or prep.get("protocol_sha256") != old.file_hash(directory/"protocol.json"):
        raise ValueError("complete preparation required")
    return protocol


def verify_data(directory):
    directory = Path(directory)
    started=time.monotonic()
    protocol=load_protocol(directory)
    folder=directory/"data-verification"
    folder.mkdir(exist_ok=False)
    atomic_json(folder/"started.json",dict(started_utc=old.utc(),protocol_sha256=old.file_hash(directory/"protocol.json")))
    verified=data_api.verify(directory/"data")
    from experiments.foundation_evaluation import FoundationBank
    tick=time.monotonic()
    admitted_banks=0
    for stage in TOTALS:
        for role,named in verified["data"]["stages"][stage]["banks"].items():
            for rows in named.values():
                FoundationBank(rows,role=role,config=CONFIG)
                admitted_banks+=1
    # The independently rebuilt indexes are intentionally process-owned, not saved.
    receipt=dict(schema=SCHEMA,status="completed",protocol_sha256=old.file_hash(directory/"protocol.json"),
        data_files_sha256=protocol["data_files_sha256"],source_sha256=protocol["source_sha256"],
        report=verified["report"],index_identity={k:v.identity for k,v in verified["indexes"].items()},
        index_construction={k:v.construction for k,v in verified["indexes"].items()},
        codec_bank_admission=dict(banks=admitted_banks,wall_seconds=time.monotonic()-tick),
        wall_seconds=time.monotonic()-started,neural_training_or_inference=False)
    load_protocol(directory)
    atomic_json(folder/"receipt.json",receipt)
    return receipt


def _data_gate(directory,protocol):
    receipt=old.read(Path(directory)/"data-verification/receipt.json")
    if (receipt.get("schema") != SCHEMA or receipt.get("status") != "completed"
            or receipt.get("neural_training_or_inference") is not False
            or receipt.get("protocol_sha256") != old.file_hash(Path(directory)/"protocol.json")):
        raise ValueError("independent complete data verification required")
    _same(receipt["data_files_sha256"],protocol["data_files_sha256"],"verified data changed")
    _same(receipt["source_sha256"],protocol["source_sha256"],"verified data source changed")
    if set(receipt["index_identity"]) != set(TOTALS):
        raise ValueError("both independently authenticated plans required")
    return receipt


def jobs(stage, selection=None):
    if stage not in TOTALS:
        raise ValueError("unknown study phase")
    if stage=="calibration":
        return [dict(id=f"{arch}-lr{rate:g}",architecture=arch,seed=CALIBRATION_SEED,rate=rate)
                for arch in ARCHITECTURES for rate in RATES]
    if selection is None:
        raise ValueError("sealed calibration selection required")
    return [dict(id=f"{arch}-seed{seed}",architecture=arch,seed=seed,rate=selection["selected"][arch]["rate"])
            for arch in ARCHITECTURES for seed in SEEDS]


def _phase_choice(directory,stage,protocol):
    selection=load_selection(directory,protocol=protocol) if stage=="main" else None
    return jobs(stage,selection),selection


def _selection_inputs(directory,report):
    names={*report["input_file_sha256"],"evaluation/calibration/report.json",
           "evaluation/calibration/score-intents.jsonl","verification/calibration/receipt.json"}
    names.update("evaluation/calibration/"+job+".json" for job in report["job_files_sha256"])
    return {name:old.file_hash(Path(directory)/name) for name in sorted(names)}


def _selection_unchanged(directory,selection,digest):
    if selection is None: return
    if old.file_hash(Path(directory)/"selection.json")!=digest or any(
            old.file_hash(Path(directory)/name)!=value for name,value in selection["input_file_sha256"].items()):
        raise ValueError("calibration selection or its authenticated inputs changed")


def _trainer(data,stage,job,index,device="cpu",payload=None):
    values=data["stages"][stage]
    return VariantFoundationTrainer(values["plan"], "curriculum", architecture=job["architecture"],
        seed=job["seed"],config=CONFIG,learning_rate=job["rate"],device=device,payload=payload,
        admission_receipt=values["admission"],
        admission_protected_transcripts=values["admission_protected_transcripts"],
        protected_transcripts=values["protected_transcripts"],plan_index=index)


def _index(directory,stage,data,gate):
    plan=data["stages"][stage]["plan"]
    if len(plan["schedules"]["curriculum"])!=TOTALS[stage] or plan["config"]["micro_batch_size"]!=32:
        raise ValueError("canonical plan differs from prescribed update/exposure budget")
    result=data_api.build_index(Path(directory)/"data",stage,data=data)
    _same(result.identity,gate["index_identity"][stage],"new process canonical index differs")
    return result


def _runtime(protocol):
    from experiments.execution_profile import runtime_profile
    if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:
        raise ValueError("one-thread declared runtime required")
    runtime=runtime_profile("cuda:0")
    _same(runtime,protocol["execution_profile"],"formal runtime differs from proof")
    return runtime


def train(directory,stage,architecture):
    from experiments.execution_profile import assert_strict_profile
    directory=Path(directory)
    started=time.monotonic()
    protocol=load_protocol(directory)
    gate=_data_gate(directory,protocol)
    all_jobs,selection=_phase_choice(directory,stage,protocol)
    selection_digest=old.file_hash(directory/"selection.json") if selection is not None else None
    declared=[j for j in all_jobs if j["architecture"]==architecture]
    if len(declared)!=3:
        raise ValueError("one architecture worker with three prescribed jobs required")
    runtime=_runtime(protocol)
    folder=directory/stage/architecture
    folder.mkdir(parents=True,exist_ok=False)
    worker=dict(schema=SCHEMA,status="running",stage=stage,architecture=architecture,started_utc=old.utc(),
        protocol_sha256=old.file_hash(directory/"protocol.json"),execution_profile=runtime,
        completed_jobs=[],automatic_promotion=False,hard_stop_note="Incomplete intent/receipt means unknown work; no automatic retry.")
    atomic_json(folder/"worker.json",worker)
    try:
        data=data_api.load(directory/"data")
        index=_index(directory,stage,data,gate)
        worker["index_construction"]=index.construction
        worker["setup_before_jobs_seconds"]=time.monotonic()-started
        for job in declared:
            _train_job(directory,stage,job,data,index,protocol,runtime)
            worker["completed_jobs"].append(job["id"])
            atomic_json(folder/"worker.json",worker)
            gc.collect(); torch.cuda.empty_cache()
        load_protocol(directory)
        _selection_unchanged(directory,selection,selection_digest)
        worker["status"]="completed"
    except BaseException as error:
        worker.update(status="failed",error=repr(error))
        raise
    finally:
        worker.update(ended_utc=old.utc(),wall_seconds=time.monotonic()-started)
        atomic_json(folder/"worker.json",worker)
    return worker


def _train_job(directory,stage,job,data,index,protocol,runtime):
    from experiments.execution_profile import assert_strict_profile
    started=time.monotonic()
    folder=Path(directory)/stage/job["architecture"]/job["id"]
    folder.mkdir(exist_ok=False)
    journal=folder/"steps.jsonl"
    receipt=dict(schema=SCHEMA,status="running",stage=stage,job=job,started_utc=old.utc(),
        protocol_sha256=old.file_hash(Path(directory)/"protocol.json"),execution_profile=runtime,
        checkpoints={},physical_optimizer_updates=0,neural_attempted_episode_exposures=0,
        completed_microbatch_episode_exposures=0,drawn_episode_exposures=0,
        physical_work_unknown=False,automatic_promotion=False)
    atomic_json(folder/"receipt.json",receipt)
    learner=None
    torch.cuda.reset_peak_memory_stats()
    try:
        learner=_trainer(data,stage,job,index,device="cuda:0")
        receipt["setup_seconds"]=time.monotonic()-started
        receipt["trainer_setup"]=learner.setup_report
        for step in range(TOTALS[stage]+1):
            if step in STEPS[stage]:
                _sources(protocol); assert_strict_profile()
                payload=learner.snapshot()
                digest=checkpoint_digest(learner.model)
                if step==0 and digest!=protocol["initial_weights_sha256"][str(job["seed"])]:
                    raise ValueError("formal paired initialization differs")
                name=f"checkpoint-{step:06d}.pt"
                atomic_checkpoint(folder/name,dict(schema=SCHEMA,stage=stage,job=job,
                    protocol_sha256=receipt["protocol_sha256"],learner=payload,
                    weights_sha256=digest,execution_profile=runtime))
                receipt["checkpoints"][name]=old.file_hash(folder/name)
                atomic_json(folder/"receipt.json",receipt)
                print(json.dumps(dict(stage=stage,job=job["id"],checkpoint=step)),flush=True)
            if step==TOTALS[stage]:
                break
            assert_strict_profile()
            old.append_journal(journal,dict(event="started",cursor=step,utc=old.utc()))
            try:
                report=learner.step()
            except BaseException:
                report=learner.last_report
                old.account_report(receipt,report)
                old.append_journal(journal,dict(event="failed",report=report))
                raise
            old.account_report(receipt,report)
            old.append_journal(journal,dict(event="completed",report=report))
        receipt["status"]="completed"
    except BaseException as error:
        receipt.update(status="failed",error=repr(error))
        raise
    finally:
        acknowledged=max((int(name.split("-")[1].split(".")[0]) for name in receipt["checkpoints"]),default=0)
        receipt.update(ended_utc=old.utc(),wall_seconds=time.monotonic()-started,
            retained_updates=None if learner is None else learner.cursor,
            acknowledged_checkpoint_updates=acknowledged,
            completed_updates_without_acknowledged_checkpoint=None if learner is None else learner.cursor-acknowledged,
            journal_sha256=old.file_hash(journal) if journal.exists() else None,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            timing_scope="Includes model setup, steps and checkpoint I/O; excludes shared index setup, imports and final receipt write. Step/materialization intervals are nested.")
        atomic_json(folder/"receipt.json",receipt)
    return receipt


def _inputs(directory,stage,protocol,declared=None,selection=None):
    directory=Path(directory)
    named={"protocol.json","preparation.json","data-verification/receipt.json"}
    if stage=="main":
        named.add("selection.json")
    receipts={}
    if declared is None:
        declared,selection=_phase_choice(directory,stage,protocol)
    if stage=="main":
        named.update(selection["input_file_sha256"])
    for arch in ARCHITECTURES:
        path=f"{stage}/{arch}/worker.json"
        worker=old.read(directory/path)
        if (worker.get("status")!="completed" or worker.get("schema")!=SCHEMA
                or worker.get("stage")!=stage or worker.get("architecture")!=arch
                or worker.get("automatic_promotion") is not False
                or worker.get("protocol_sha256")!=old.file_hash(directory/"protocol.json")
                or worker.get("completed_jobs")!=[j["id"] for j in declared if j["architecture"]==arch]):
            raise ValueError("both complete architecture workers required")
        _same(worker["execution_profile"],protocol["execution_profile"],"worker runtime differs")
        named.add(path)
    expected={f"checkpoint-{s:06d}.pt" for s in STEPS[stage]}
    for job in declared:
        prefix=f"{stage}/{job['architecture']}/{job['id']}"
        receipt=old.read(directory/prefix/"receipt.json")
        if (receipt.get("status")!="completed" or receipt.get("schema")!=SCHEMA
                or receipt.get("stage")!=stage or receipt.get("job")!=job
                or receipt.get("physical_work_unknown") is not False
                or receipt.get("retained_updates")!=TOTALS[stage]
                or type(receipt.get("retained_updates")) is not int
                or receipt.get("automatic_promotion") is not False
                or set(receipt.get("checkpoints",{}))!=expected):
            raise ValueError("all six exact completed jobs and checkpoints required")
        _same(receipt["execution_profile"],protocol["execution_profile"],"job runtime differs")
        if receipt["protocol_sha256"]!=old.file_hash(directory/"protocol.json"):
            raise ValueError("job protocol differs")
        if receipt["journal_sha256"]!=old.file_hash(directory/prefix/"steps.jsonl"):
            raise ValueError("job journal changed")
        for name,digest in receipt["checkpoints"].items():
            if old.file_hash(directory/prefix/name)!=digest:
                raise ValueError("official checkpoint changed")
        named.update({f"{prefix}/receipt.json",f"{prefix}/steps.jsonl",*(f"{prefix}/{n}" for n in expected)})
        receipts[job["id"]]=receipt
    return {name:old.file_hash(directory/name) for name in sorted(named)},receipts


def _checkpoint(directory,stage,job,step,protocol):
    path=Path(directory)/stage/job["architecture"]/job["id"]/f"checkpoint-{step:06d}.pt"
    saved=torch.load(path,map_location="cpu",weights_only=True)
    if (type(saved) is not dict or set(saved)!={"schema","stage","job","protocol_sha256","learner","weights_sha256","execution_profile"}
            or saved["schema"]!=SCHEMA or saved["stage"]!=stage or saved["job"]!=job
            or saved["protocol_sha256"]!=old.file_hash(Path(directory)/"protocol.json")):
        raise ValueError("variant checkpoint envelope differs")
    _same(saved["execution_profile"],protocol["execution_profile"],"checkpoint runtime differs")
    if saved["learner"]["cursor"]!=step or type(saved["learner"]["cursor"]) is not int:
        raise ValueError("official checkpoint cursor differs")
    return saved


def verify(directory,stage):
    directory=Path(directory)
    started=time.monotonic()
    protocol=load_protocol(directory)
    gate=_data_gate(directory,protocol)
    declared,selection=_phase_choice(directory,stage,protocol)
    inputs,receipts=_inputs(directory,stage,protocol,declared,selection)
    folder=directory/"verification"/stage
    folder.mkdir(parents=True,exist_ok=False)
    atomic_json(folder/"started.json",dict(started_utc=old.utc(),input_file_sha256=inputs))
    data=data_api.load(directory/"data")
    index=_index(directory,stage,data,gate)
    values=data["stages"][stage]
    canonical=index.replay("curriculum",TOTALS[stage],include_bundles=True)
    result={}
    for job in declared:
        prefix=directory/stage/job["architecture"]/job["id"]
        events=[json.loads(line) for line in (prefix/"steps.jsonl").read_text(encoding="utf8").splitlines()]
        journal=old_results.validate_journal(events,values["plan"],"curriculum",values["manifest"],canonical["bundles"])
        old_results._validate_receipt(receipts[job["id"]],journal,TOTALS[stage],values["plan"]["config"]["micro_batch_size"])
        learner=_trainer(data,stage,job,index)
        hashes={}
        restore_seconds=0.
        for step in STEPS[stage]:
            saved=_checkpoint(directory,stage,job,step,protocol)
            learner.restore(saved["learner"])
            restore_seconds+=learner.last_restore_seconds
            digest=checkpoint_digest(learner.model)
            if digest!=saved["weights_sha256"]:
                raise ValueError("verified weights differ from checkpoint digest")
            if step==0 and digest!=protocol["initial_weights_sha256"][str(job["seed"])]:
                raise ValueError("verified seeded initialization differs")
            reports=[events[2*i+1]["report"] for i in range(step)]
            for name,field in (("retained_step_seconds","step_seconds"),
                               ("materialization_seconds_included_in_step","materialization_seconds")):
                if not math.isclose(saved["learner"]["timing"][name],sum(row[field] for row in reports),rel_tol=1e-10,abs_tol=1e-8):
                    raise ValueError("official checkpoint timing differs from its journal prefix")
            hashes[str(step)]=digest
        _same(learner.snapshot()["evidence"],canonical["evidence"],"final canonical prefix differs")
        expected_exposures={family:{key:values["manifest"]["totals"][family][key] for key in old_results.COUNTS}
                            for family in old_results.FAMILIES}
        _same(canonical["evidence"]["exposures"],expected_exposures,"full plan exposure differs from canonical manifest")
        for key in ("retained_step_seconds","materialization_seconds_included_in_step"):
            if abs(learner.snapshot()["timing"][key]-journal[key])>1e-6:
                raise ValueError("checkpoint timing disagrees with complete journal")
        result[job["id"]]=dict(job=job,weights_sha256=hashes,exact_official_checkpoint_restores=True,
            journal=journal,restore_seconds=restore_seconds)
        del learner
    _same(_inputs(directory,stage,protocol,declared,selection)[0],inputs,"verification inputs changed")
    load_protocol(directory)
    receipt=dict(schema=SCHEMA,status="completed",stage=stage,protocol_sha256=inputs["protocol.json"],
        input_file_sha256=inputs,source_sha256=protocol["source_sha256"],results=result,
        index_construction=index.construction,index_identity=index.identity,
        wall_seconds=time.monotonic()-started,neural_training_or_inference=False,automatic_promotion=False,
        scope="Canonical index rebuilt once; all official optimizer states restored. Past gradients are not recomputed.")
    atomic_json(folder/"receipt.json",receipt)
    return receipt


def _verified(directory,stage,protocol,declared=None,selection=None):
    if declared is None:
        declared,selection=_phase_choice(directory,stage,protocol)
    inputs,_=_inputs(directory,stage,protocol,declared,selection)
    verified=old.read(Path(directory)/"verification"/stage/"receipt.json")
    if (verified.get("schema")!=SCHEMA or verified.get("status")!="completed" or verified.get("stage")!=stage
            or verified.get("neural_training_or_inference") is not False
            or verified.get("automatic_promotion") is not False or verified.get("protocol_sha256")!=inputs["protocol.json"]
            or set(verified.get("results",{}))!={j["id"] for j in declared}):
        raise ValueError("all completed phase checkpoint verification required")
    _same(verified["input_file_sha256"],inputs,"verified phase inputs changed")
    _same(verified["source_sha256"],protocol["source_sha256"],"verified phase sources changed")
    for row in verified["results"].values():
        if row.get("exact_official_checkpoint_restores") is not True or set(row.get("weights_sha256",{}))!={str(s) for s in STEPS[stage]}:
            raise ValueError("missing official checkpoint verification")
    return inputs,verified


def evaluate(directory,stage):
    from experiments.foundation_evaluation import FoundationBank
    from experiments.execution_profile import assert_strict_profile
    directory=Path(directory)
    started=time.monotonic()
    protocol=load_protocol(directory)
    _data_gate(directory,protocol)
    declared,selection=_phase_choice(directory,stage,protocol)
    inputs,verified=_verified(directory,stage,protocol,declared,selection)
    runtime=_runtime(protocol)
    folder=directory/"evaluation"/stage
    folder.mkdir(parents=True,exist_ok=False)
    binding=dict(schema=SCHEMA,stage=stage,protocol_sha256=inputs["protocol.json"],
        verification_sha256=old.file_hash(directory/"verification"/stage/"receipt.json"),
        input_file_sha256=inputs,execution_profile=runtime,automatic_promotion=False)
    atomic_json(folder/"started.json",dict(**binding,started_utc=old.utc()))
    data=data_api.load(directory/"data")
    banks=data["stages"][stage]["banks"]
    tick=time.monotonic()
    prepared={role:{name:FoundationBank(rows,role=role,config=CONFIG) for name,rows in named.items()}
              for role,named in banks.items()}
    bank_preparation_seconds=time.monotonic()-tick
    results={}
    for job in declared:
        tick=time.monotonic()
        model=(build_sequence_student if job["architecture"]=="flat" else build_hierarchical_sequence_student)(
            job["seed"],device="cuda:0",config=CONFIG)
        receipt=dict(**binding,job=job,status="running",development_curve=[])
        atomic_json(folder/(job["id"]+"-started.json"),dict(**binding,job=job,started_utc=old.utc()))
        try:
            for step in (STEPS[stage] if stage=="main" else (TOTALS[stage],)):
                assert_strict_profile()
                saved=_checkpoint(directory,stage,job,step,protocol)
                model.load_state_dict(saved["learner"]["weights"],strict=True)
                digest=checkpoint_digest(model)
                if digest!=verified["results"][job["id"]]["weights_sha256"][str(step)]:
                    raise ValueError("scored architecture/weights differ from verified checkpoint")
                old.append_journal(folder/"score-intents.jsonl",dict(event="started",job=job["id"],updates=step,role="dev",utc=old.utc()))
                scored=old_results._score(model,prepared["dev"],banks["dev"],CONFIG,"dev")
                receipt["development_curve"].append(dict(updates=step,weights_sha256=digest,**scored))
                old.append_journal(folder/"score-intents.jsonl",dict(event="completed",job=job["id"],updates=step,role="dev",cost=scored["cost"]))
            if stage=="main":
                for role,control in (("train_fit","normal"),("audit","normal"),("audit","blank"),("audit","reset")):
                    old.append_journal(folder/"score-intents.jsonl",dict(event="started",job=job["id"],updates=TOTALS[stage],role=role,control=control,utc=old.utc()))
                    scored=old_results._score(model,prepared[role],banks[role],CONFIG,role,control=control)
                    scored.update(updates=TOTALS[stage],weights_sha256=digest,architecture=job["architecture"])
                    if control=="normal":
                        receipt[role]=scored
                    else:
                        receipt.setdefault("controls",{})[control]=scored
                    old.append_journal(folder/"score-intents.jsonl",dict(event="completed",job=job["id"],updates=TOTALS[stage],role=role,control=control,cost=scored["cost"]))
            receipt["status"]="completed"
        except BaseException as error:
            receipt.update(status="failed",error=repr(error),failed_scoring_work_may_be_unrecorded=True)
            raise
        finally:
            receipt.update(wall_seconds=time.monotonic()-tick,ended_utc=old.utc())
            atomic_json(folder/(job["id"]+".json"),receipt)
        results[job["id"]]=receipt
        del model
        gc.collect(); torch.cuda.empty_cache()
    load_protocol(directory)
    _same(_verified(directory,stage,protocol,declared,selection)[0],inputs,"scoring inputs changed")
    if old.file_hash(directory/"verification"/stage/"receipt.json")!=binding["verification_sha256"]:
        raise ValueError("verification changed during scoring")
    assert_strict_profile()
    report=dict(**binding,status="completed",results=results,bank_preparation_seconds=bank_preparation_seconds,
        wall_seconds=time.monotonic()-started,score_journal_sha256=old.file_hash(folder/"score-intents.jsonl"),
        job_files_sha256={j:old.file_hash(folder/(j+".json")) for j in results},
        scope="BOS-only generated replies; exact verified weights. Scoring and setup intervals nested in evaluation wall.")
    atomic_json(folder/"report.json",report)
    return report


def _evaluated(directory,stage,protocol):
    from experiments.foundation_metrics import validate_metrics
    declared,selection=_phase_choice(directory,stage,protocol)
    inputs,verified=_verified(directory,stage,protocol,declared,selection)
    folder=Path(directory)/"evaluation"/stage
    report=old.read(folder/"report.json")
    if report.get("schema")!=SCHEMA or report.get("stage")!=stage or report.get("status")!="completed" or report.get("automatic_promotion") is not False:
        raise ValueError("complete evaluation required")
    _same(report["input_file_sha256"],inputs,"evaluation input hashes changed")
    _same(report["execution_profile"],protocol["execution_profile"],"evaluation runtime differs")
    if (report["protocol_sha256"]!=inputs["protocol.json"] or report["verification_sha256"]!=old.file_hash(Path(directory)/"verification"/stage/"receipt.json")
            or report["score_journal_sha256"]!=old.file_hash(folder/"score-intents.jsonl")):
        raise ValueError("evaluation bindings differ")
    if set(report["results"])!={j["id"] for j in declared} or set(report["job_files_sha256"])!=set(report["results"]):
        raise ValueError("all exact evaluation jobs required")
    data=data_api.load(Path(directory)/"data")
    banks=data["stages"][stage]["banks"]
    for job in declared:
        value=report["results"][job["id"]]
        if old.file_hash(folder/(job["id"]+".json"))!=report["job_files_sha256"][job["id"]]:
            raise ValueError("evaluation job changed")
        _same(value,old.read(folder/(job["id"]+".json")),"report/job result differs")
        if value["status"]!="completed" or value["job"]!=job:
            raise ValueError("evaluation job incomplete")
        for key in ("schema","stage","protocol_sha256","verification_sha256","execution_profile","input_file_sha256","automatic_promotion"):
            _same(value[key],report[key],"evaluation job binding differs: "+key)
        curve=value["development_curve"]
        if [r["updates"] for r in curve]!=list(STEPS[stage] if stage=="main" else (TOTALS[stage],)):
            raise ValueError("evaluation curve checkpoint set differs")
        for row in curve:
            if row["weights_sha256"]!=verified["results"][job["id"]]["weights_sha256"][str(row["updates"])]:
                raise ValueError("score producer weight binding differs")
            validate_metrics(row["metrics"],banks["dev"],CONFIG,"dev")
        if stage=="main":
            for role in ("train_fit","audit"):
                scored=value[role]
                if (scored.get("updates")!=TOTALS[stage] or scored.get("architecture")!=job["architecture"]
                        or scored.get("weights_sha256")!=verified["results"][job["id"]]["weights_sha256"][str(TOTALS[stage])]):
                    raise ValueError("endpoint score producer differs")
                validate_metrics(value[role]["metrics"],banks[role],CONFIG,role)
            if set(value["controls"])!={"blank","reset"}:
                raise ValueError("both controls required")
            for control,row in value["controls"].items():
                if (row.get("updates")!=TOTALS[stage] or row.get("architecture")!=job["architecture"]
                        or row.get("weights_sha256")!=verified["results"][job["id"]]["weights_sha256"][str(TOTALS[stage])]):
                    raise ValueError("control score producer differs")
                validate_metrics(row["metrics"],banks["audit"],CONFIG,"audit",control=control)
    return report


def select(directory):
    from experiments.foundation_variant_analysis import choose_rate
    directory=Path(directory)
    protocol=load_protocol(directory)
    report=_evaluated(directory,"calibration",protocol)
    if (directory/"selection.json").exists():
        raise FileExistsError("calibration selection already sealed")
    selected={}
    for arch in ARCHITECTURES:
        candidates={str(job["rate"]):report["results"][job["id"]]["development_curve"][0]["metrics"]
                    for job in jobs("calibration") if job["architecture"]==arch}
        selected[arch]=choose_rate(candidates)
    selection=dict(schema=SCHEMA,status="completed",protocol_sha256=old.file_hash(directory/"protocol.json"),
        evaluation_sha256=old.file_hash(directory/"evaluation/calibration/report.json"),
        input_file_sha256=_selection_inputs(directory,report),
        selected=selected,automatic_promotion=False,created_utc=old.utc())
    atomic_json(directory/"selection.json",selection)
    return selection


def load_selection(directory,*,protocol=None):
    from experiments.foundation_variant_analysis import choose_rate
    directory=Path(directory)
    # Calibration loading never invokes this function; recursion stays bounded.
    protocol=load_protocol(directory) if protocol is None else protocol
    report=_evaluated(directory,"calibration",protocol)
    selection=old.read(directory/"selection.json")
    if (selection.get("schema")!=SCHEMA or selection.get("status")!="completed"
            or selection.get("automatic_promotion") is not False
            or selection.get("protocol_sha256")!=old.file_hash(directory/"protocol.json")
            or selection.get("evaluation_sha256")!=old.file_hash(directory/"evaluation/calibration/report.json")
            or set(selection.get("selected",{}))!=set(ARCHITECTURES)):
        raise ValueError("complete sealed calibration choice required")
    _same(selection["input_file_sha256"],_selection_inputs(directory,report),"selection authenticated inputs changed")
    for arch in ARCHITECTURES:
        candidates={str(job["rate"]):report["results"][job["id"]]["development_curve"][0]["metrics"]
                    for job in jobs("calibration") if job["architecture"]==arch}
        _same(selection["selected"][arch],choose_rate(candidates),"calibration ranking changed")
    return selection


def summarize(directory):
    from experiments.foundation_variant_analysis import comparison_screen
    directory=Path(directory)
    protocol=load_protocol(directory)
    report=_evaluated(directory,"main",protocol)
    path=directory/"summary.json"
    if path.exists():
        raise FileExistsError("summary already sealed")
    results={arch:{str(seed):report["results"][f"{arch}-seed{seed}"]["audit"]["metrics"] for seed in SEEDS}
             for arch in ARCHITECTURES}
    value=dict(schema=SCHEMA,status="completed",protocol_sha256=old.file_hash(directory/"protocol.json"),
        evaluation_sha256=old.file_hash(directory/"evaluation/main/report.json"),screen=comparison_screen(results),
        automatic_promotion=False,formal_updates=21492,formal_episode_exposures=2063232,
        scope="Descriptive three-initialization comparison on one common data stream; inspect complete fitting/retention/ASK/domain/operator evidence before choosing any next change.")
    atomic_json(path,value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","verify-data","train","verify","evaluate","select","summarize"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=tuple(TOTALS))
    parser.add_argument("--architecture",choices=ARCHITECTURES)
    parser.add_argument("--foundation",type=Path)
    parser.add_argument("--proof",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.command in ("train","evaluate"):
        from experiments.execution_profile import configure_strict_profile
        configure_strict_profile()
    if args.command=="prepare":
        if args.foundation is None or args.proof is None: parser.error("prepare needs --foundation and --proof")
        prepare(args.output,args.foundation,args.proof)
    elif args.command=="verify-data": verify_data(args.output)
    elif args.command=="select": select(args.output)
    elif args.command=="summarize": summarize(args.output)
    else:
        if args.stage is None: parser.error("phase command needs --stage")
        if args.command=="train":
            if args.architecture is None: parser.error("train needs --architecture")
            train(args.output,args.stage,args.architecture)
        else: globals()[args.command](args.output,args.stage)


if __name__=="__main__":
    main()
