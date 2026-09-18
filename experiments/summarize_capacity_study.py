"""Descriptive width summaries after all calibration, main and audit work.

The complete-file gate precedes all score reads. This helper restores CPU
checkpoint state and reconstructs canonical fitting rows but never invokes a
neural forward pass, optimizer, development selector or promotion operation.
"""
from __future__ import annotations

import argparse
import copy
from itertools import combinations
from datetime import datetime
import json
import math
from pathlib import Path

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_loop import run_lock
from brain_in_computer.learning_student import _check_finite_tree
from experiments import capacity_study as study
from experiments import capacity_evidence as evidence
from experiments import summarize_composition_study as metric_helper
from experiments.capacity_banks import bank_manifest, historical_evidence, protected_transcripts, verify_boundaries
from experiments.composition_evaluation import METRICS
from experiments.realization_banks import _stats
from experiments.realization_training import COUNTS, stream_evidence
from experiments.sequence_student import build_sequence_student
from experiments.train_cognitive import atomic_json

SCHEMA = "bic-shared-capacity-summary-v1"
EXTRA_METRICS = ("ask_precision", "ask_recall", "action_reply_agreement", "query_loss", "brier_score", "unsupported_ask_rate")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _same(left, right):
    if not evidence._json_equal(left, right):
        raise ValueError("capacity summary evidence differs")


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def compact(metrics):
    """Preserve every bank denominator and expose unsupported ASK explicitly."""
    result = metric_helper.compact(metrics)
    for families in result["per_panel_family"].values():
        for group in families.values():
            rows = list(group["per_length"].values())
            for row in rows:
                confusion = row["confusion_matrix"]
                count = confusion[0][2] + confusion[1][2]
                row["unsupported_ask_count"] = count
                row["unsupported_ask_rate"] = count / row["known_total"] if row["known_total"] else None
            for key in EXTRA_METRICS:
                group["macro"][key] = metric_helper._mean([row[key] for row in rows])
            group["pooled_counts"] = {
                "final_pairs": {k: sum(row["final_pairs"][k] for row in rows) for k in ("correct", "total")},
                "final_reply_pairs": {"correct": sum(row["final_reply_pair_correct"] for row in rows),
                                      "total": sum(row["final_pairs"]["total"] for row in rows)},
                "known_queries": {"correct": sum(row["known_correct"] for row in rows),
                                  "total": sum(row["known_total"] for row in rows)},
                "queries": {"correct": sum(row["query_correct"] for row in rows),
                            "total": sum(row["query_total"] for row in rows)},
                "unsupported_ask_count": sum(row["unsupported_ask_count"] for row in rows),
            }
    return result


def metric_values(metrics):
    return {panel: {family: {"macro": group["macro"], "per_length": {
        length: {key: row[key] for key in (*METRICS, *EXTRA_METRICS)}
        for length, row in group["per_length"].items()}} for family, group in families.items()}
        for panel, families in compact(metrics)["per_panel_family"].items()}


def curve_areas(curve):
    adapted = [{**point, "held_episode_exposure": point["episodes_per_family"]} for point in curve]
    return {family: metric_helper.curve_areas(adapted, family) for family in study.FAMILIES}


def _required(directory):
    root = Path(directory)
    files = [root/name for name in ("protocol.json", "banks.pt", "protected.pt", "planned-calibration.pt", "selection.json", "preparation.json")]
    for stage in study.TOTALS:
        files += study.required(root, stage)
        files.append(root/"verification"/f"{stage}.json")
    files += [root/"audit"/name for name in ("evaluation-started.json", "report.json", *(f"{job}.json" for job in study.jobs("main")))]
    return files


def _complete_gate(directory):
    paths = _required(directory)
    if any(not path.is_file() for path in paths):
        raise ValueError("summary requires all9 calibration jobs, all3 main jobs, sealed stage replays, selection and all3 completed audits")
    return paths


def invocation_costs(receipts, *, total_updates):
    """Do not turn missing hard-stop accounting into zero physical work."""
    if not receipts or not isinstance(receipts, dict):
        raise ValueError("nonempty invocation ledger required")
    ordered = [receipts[key] for key in sorted(receipts)]
    if ordered[-1].get("status") != "completed" or ordered[-1].get("ending_updates") != total_updates:
        raise ValueError("final invocation did not complete the prescribed endpoint")
    wall, setup, post_setup, steps, commits, rollback, discard_time = [], [], [], [], [], [], []
    completed, discarded, failed_steps = 0, 0, 0
    unknown = False
    peaks, reserved_peaks, setup_peaks, setup_reserved_peaks = [], [], [], []
    previous_end = 0
    for receipt in ordered:
        if receipt.get("status") not in ("completed", "failed", "running"):
            raise ValueError("invalid invocation status")
        for key in ("starting_updates", "ending_updates"):
            if key in receipt and (type(receipt[key]) is not int or not 0 <= receipt[key] <= total_updates):
                raise ValueError("invocation update boundary differs")
        start = receipt.get("starting_updates")
        if start is None:
            unknown = True
        elif previous_end is not None:
            if start < previous_end:
                raise ValueError("settled invocation update ranges overlap or reverse")
            if start > previous_end:
                unknown = True
        setup.append(_finite(receipt["setup_seconds"], "invocation setup"))
        if "wall_seconds" in receipt:
            wall.append(_finite(receipt["wall_seconds"], "invocation wall"))
            post_setup.append(_finite(receipt["post_setup_wall_seconds"], "post-setup wall"))
            if setup[-1] > wall[-1] or post_setup[-1] > wall[-1] + .001:
                raise ValueError("setup/post-setup time exceeds invocation wall")
        else:
            unknown = True
        if receipt.get("status") == "running":
            unknown = True
        for key, output in (("peak_cuda_allocated_mib", peaks), ("peak_cuda_reserved_mib", reserved_peaks),
                            ("setup_peak_cuda_allocated_mib", setup_peaks), ("setup_peak_cuda_reserved_mib", setup_reserved_peaks)):
            if key in receipt:
                output.append(_finite(receipt[key], key))
        chunks = receipt.get("chunks", [])
        if not isinstance(chunks, list):
            raise ValueError("invocation chunks must be a list")
        ledger = list(chunks)
        if receipt.get("failure_report") is not None and receipt["failure_report"] not in ledger:
            # A failure during later scoring can repeat the already recorded
            # successful chunk as last_report; do not charge it twice.
            ledger.append(receipt["failure_report"])
        retained_delta, retained_known = 0, True
        for chunk in ledger:
            committed = chunk.get("committed_updates")
            if committed is None or chunk.get("publication_uncertain"):
                retained_known = False
                unknown = True
            elif type(committed) is not int or committed < 0:
                raise ValueError("committed update ledger must contain nonnegative integers")
            else:
                retained_delta += committed
                done, lost = chunk.get("completed_updates"), chunk.get("discarded_completed_updates")
                if type(done) is int and type(lost) is int and committed + lost != done:
                    raise ValueError("committed/discarded updates differ from completed work")
            for key, output in (("step_seconds", steps), ("commit_seconds", commits),
                                ("rollback_seconds", rollback), ("discarded_step_seconds", discard_time)):
                if key in chunk:
                    output.append(_finite(chunk[key], key))
            for key in ("completed_updates", "discarded_completed_updates", "failed_step_attempts"):
                value = chunk.get(key)
                if value is None:
                    unknown = True
                    continue
                if type(value) is not int or value < 0:
                    raise ValueError("invocation work count must be a nonnegative integer or null")
                if key == "completed_updates": completed += value
                elif key == "discarded_completed_updates": discarded += value
                else: failed_steps += value
            if chunk.get("publication_uncertain"):
                unknown = True
        if receipt.get("status") != "running" and retained_known and "ending_updates" in receipt:
            if start is None or retained_delta != receipt["ending_updates"] - start:
                raise ValueError("committed update ledger differs from invocation boundaries")
            previous_end = receipt["ending_updates"]
        else:
            # Hard-stop or uncertain publication receipts cannot establish a
            # trustworthy next boundary or completeness of physical work.
            previous_end = None
            unknown = True
    return {"invocations": copy.deepcopy(receipts), "recorded_worker_wall_seconds": sum(wall),
        "recorded_step_seconds": sum(steps), "recorded_commit_seconds": sum(commits),
        "recorded_rollback_seconds": sum(rollback), "recorded_discarded_step_seconds": sum(discard_time),
        "recorded_completed_update_calls": completed, "known_discarded_completed_updates": discarded,
        "known_failed_step_attempts": failed_steps, "unrecorded_physical_work_possible": unknown,
        "peak_cuda_allocated_mib": max(peaks+setup_peaks) if peaks+setup_peaks else None,
        "peak_cuda_reserved_mib": max(reserved_peaks+setup_reserved_peaks) if reserved_peaks+setup_reserved_peaks else None,
        "post_setup_peak_cuda_allocated_mib": max(peaks) if peaks else None,
        "post_setup_peak_cuda_reserved_mib": max(reserved_peaks) if reserved_peaks else None,
        "setup_peak_cuda_allocated_mib": max(setup_peaks) if setup_peaks else None,
        "setup_peak_cuda_reserved_mib": max(setup_reserved_peaks) if setup_reserved_peaks else None,
        "recorded_setup_seconds": sum(setup), "recorded_post_setup_wall_seconds": sum(post_setup),
        "scope": "Invocation wall begins at train-function entry and includes source/data checks, backend construction/load, development, chunks and checkpoint work; it excludes Python imports, prior CLI profile setup and final report construction. Setup/post-setup are subsets, not extra wall. Running receipts can conceal hard-stop work; these are recorded intervals, not a lifetime compute guarantee."}


def _load_invocations(directory, stage, job, report, protocol, hashes):
    folder = Path(directory)/stage/job/"invocations"
    declared = report["invocation_files_sha256"]
    if (not isinstance(declared, dict) or not declared
            or set(declared) != {p.name for p in folder.glob("*.json")}
            or any(Path(name).name != name or not name.endswith(".json") or not name[:-5].isdigit() for name in declared)):
        raise ValueError("invocation receipt set differs")
    receipts = {}
    for name, sha in declared.items():
        path = folder/name
        if study.file_hash(path) != sha:
            raise ValueError("invocation receipt hash differs")
        value = _read(path)
        if value.get("schema") != study.SCHEMA:
            raise ValueError("invocation schema differs")
        _same(value["execution_profile"], protocol["profile_proof"]["execution_profile"])
        hashes[path.relative_to(directory).as_posix()] = sha
        receipts[name] = value
    return invocation_costs(receipts, total_updates=study.TOTALS[stage])


def _validate_endpoint_binding(final, checkpoint, replay):
    _same(stream_evidence(final["learner"]["realization"]), replay)
    study.tensor_equal(final, checkpoint)


def _cost_summary(job_rows):
    rows = list(job_rows.values())
    return {"jobs": len(rows), "retained_optimizer_updates": sum(r["updates"] for r in rows),
        "sampled_exposures": {key: sum(fs[key] for r in rows for fs in r["exposures"].values()) for key in COUNTS},
        "retained_worker_step_seconds": sum(r["retained_step_seconds"] for r in rows),
        "generation_validation_seconds_already_in_steps": sum(r["generation_timing_subset"]["generation_validation_seconds"] for r in rows),
        "rejected_candidate_seconds_already_in_generation": sum(r["generation_timing_subset"]["rejected_candidate_seconds"] for r in rows),
        "recorded_invocation_wall_seconds": sum(r["invocation_costs"]["recorded_worker_wall_seconds"] for r in rows),
        "recorded_development_seconds": sum(p["seconds"] for r in rows for p in r["development_curve"]),
        "recorded_commit_seconds": sum(r["invocation_costs"]["recorded_commit_seconds"] for r in rows),
        "recorded_discarded_step_seconds": sum(r["invocation_costs"]["recorded_discarded_step_seconds"] for r in rows),
        "known_discarded_completed_updates": sum(r["invocation_costs"]["known_discarded_completed_updates"] for r in rows),
        "unrecorded_physical_work_possible": any(r["invocation_costs"]["unrecorded_physical_work_possible"] for r in rows),
        "recorded_setup_seconds": sum(r["invocation_costs"]["recorded_setup_seconds"] for r in rows),
        "recorded_post_setup_wall_seconds": sum(r["invocation_costs"]["recorded_post_setup_wall_seconds"] for r in rows)}


def assemble_summary(protocol, selection, stages, audit):
    """Arithmetic only; the caller first authenticates every input and metric."""
    if set(stages) != set(study.TOTALS) or set(audit["results"]) != set(study.jobs("main")):
        raise ValueError("all study stages and audited widths required")
    for stage, rows in stages.items():
        if set(rows) != set(study.jobs(stage)):
            raise ValueError("every declared calibration/main job is required")
        reference = next(iter(rows.values()))["stream"]
        for row in rows.values():
            _same(row["stream"], reference)
            if row["updates"] != study.TOTALS[stage]:
                raise ValueError("job update budget differs")
            _same(row["exposures"], reference["exposures"])
            timing = row["generation_timing_subset"]
            if not (0 <= _finite(timing["rejected_candidate_seconds"],"rejected time")
                    <= _finite(timing["generation_validation_seconds"],"generation time")
                    <= _finite(row["retained_step_seconds"],"retained steps")):
                raise ValueError("generation timing must be a subset of retained step time")
    result = {"schema": SCHEMA, "study_schema": protocol["schema"], "automatic_promotion": False,
        "selected_checkpoint": None, "selection": copy.deepcopy(selection),
        "comparison_note": "One main initialization per width; all widths share the accepted fresh transcripts, update/exposure budgets and objective, after equal development-only rate search. Width changes capacity and optimization geometry. Differences are descriptive absolute values, without significance or a general-intelligence claim.",
        "metric_note": "Per-domain/panel/length counts are retained. Final pairs require both opposite known answers; earlier unknown and known questions remain separate. Final uncertainty and wholly unseen semantic algorithms are not tested.",
        "fit_note": "Initial fit covers admitted originals; latest fit is the latest actually observed realization per recipe. Coverage/absences are explicit. These selected training examples are not an unbiased fresh-distribution sample.",
        "area_note": "Sparse normalized trapezoids over per-family episode exposure combine initial score and later change; they do not isolate learning speed.",
        "time_note": "Generation/authentication and rejected-candidate time are nested subsets of retained steps. Development, commits and steps overlap invocation wall intervals and must not be summed again. Concurrent worker sums are not elapsed wall time or dedicated GPU-hours. Measured setup is part of invocation wall, while Python imports/CLI setup and incomplete hard-stop records can omit physical work. Probe/preparation costs are separate from these retained training totals.",
        "stages": copy.deepcopy(stages), "audit": {}, "width_differences": {}}
    for stage, rows in result["stages"].items():
        for row in rows.values():
            row["development_curve"] = [{**point,"metrics":compact(point["metrics"])} for point in row["development_curve"]]
            row["stream"] = {k:v for k,v in row["stream"].items() if k not in ("recipe","occurrences","latest")}
    for job, row in audit["results"].items():
        if [p["updates"] for p in row["curve"]] != list(study.STEPS["main"]):
            raise ValueError("audit curve checkpoints differ")
        result["audit"][job] = {
            "curve": [{**{k:copy.deepcopy(p[k]) for k in ("updates","episodes_per_family","weights_sha256")},
                       "metrics":compact(p["metrics"])} for p in row["curve"]],
            "areas_by_family":curve_areas(row["curve"]), "final":compact(row["final"]),
            "initial_fit":compact(row["initial_fit"]), "latest_observed_fit":compact(row["latest_observed_fit"]),
            "latest_observed_metadata":copy.deepcopy(row["latest_observed_metadata"]),
            "fit_coverage":{"initial_episodes":sum(2*r["final_pairs"]["total"] for r in row["initial_fit"]["per_bank"].values()),
                "latest_episodes":sum(2*r["final_pairs"]["total"] for r in row["latest_observed_fit"]["per_bank"].values()),
                "latest_observed_recipes":len(row["latest_observed_metadata"]["per_pair"]),
                "absent_recipes":len(row["latest_observed_metadata"]["absent_recipes"]),
                "all_initial_realizations_encountered":row["latest_observed_metadata"]["all_initial_realizations_encountered"]},
            "controls":{k:compact(v) for k,v in row["controls"].items()},
            "cpu_restart":copy.deepcopy(row["cpu_restart"]), "seconds":row["seconds"]}
    for lower, higher in combinations(study.WIDTHS,2):
        lo,hi=f"w{lower}",f"w{higher}"
        left,right=audit["results"][hi],audit["results"][lo]
        result["width_differences"][f"{hi}_minus_{lo}"]={
            "endpoint":metric_helper._difference(metric_values(left["final"]),metric_values(right["final"])),
            "areas_by_family":metric_helper._difference(result["audit"][hi]["areas_by_family"],result["audit"][lo]["areas_by_family"]),
            "curve":[{"updates":a["updates"],"episodes_per_family":a["episodes_per_family"],
                      "metrics":metric_helper._difference(metric_values(a["metrics"]),metric_values(b["metrics"]))}
                     for a,b in zip(left["curve"],right["curve"])],
            "initial_fit":metric_helper._difference(metric_values(left["initial_fit"]),metric_values(right["initial_fit"])),
            "latest_observed_fit":metric_helper._difference(metric_values(left["latest_observed_fit"]),metric_values(right["latest_observed_fit"])),
            "retained_step_seconds":stages["main"][hi]["retained_step_seconds"]-stages["main"][lo]["retained_step_seconds"]}
    costs={stage:_cost_summary(rows) for stage,rows in stages.items()}
    costs["total"]=_cost_summary({f"{stage}/{job}":row for stage,rows in stages.items() for job,row in rows.items()})
    costs["audit_worker_seconds"]=sum(row["seconds"] for row in audit["results"].values())
    result["compute"]=costs
    return result


def _validated_audit(value, cached, **arguments):
    _same(value, cached)
    evidence.validate_audit_result(value, **arguments)


def _execution(directory, hashes):
    """Optional dispatcher envelope, gated on terminal success before score reads."""
    path = directory/"execution.json"
    if not path.exists():
        return {"available": False, "scope": "No dispatcher ledger supplied; worker intervals do not establish total elapsed wall time."}
    value = _read(path)
    if (value.get("schema") != "bic-capacity-dispatch-v1" or value.get("status") != "completed"
            or value.get("protocol_sha256") != hashes["protocol.json"]):
        raise ValueError("dispatcher must be terminal and bound to this protocol before summary")
    expected = {f"{stage}-{job}" for stage in study.TOTALS for job in study.jobs(stage)} | {"verify-calibration","select","verify-main","audit"}
    processes=value.get("processes",[])
    if len(processes)!=len(expected) or {p.get("name") for p in processes}!=expected:
        raise ValueError("dispatcher process coverage differs")
    for process in processes:
        if process.get("status")!="completed" or process.get("exit_code")!=0:
            raise ValueError("dispatcher process did not complete successfully")
        _finite(process["observed_wall_seconds"],"observed process time")
        for output in ("stdout","stderr"):
            log=directory/"process-logs"/f"{process['name']}.{output}.log"
            sha=process[f"{output}_sha256"]
            if study.file_hash(log)!=sha:raise ValueError("dispatcher log hash differs")
            hashes[log.relative_to(directory).as_posix()]=sha
    seconds=(datetime.fromisoformat(value["completed_utc"])-datetime.fromisoformat(value["started_utc"])).total_seconds()
    _finite(seconds,"observed dispatcher envelope")
    dispatcher=Path(__file__).resolve().with_name("run_capacity_study.py")
    if study.file_hash(dispatcher)!=value["dispatcher_sha256"]:
        raise ValueError("dispatcher source differs from recorded execution")
    hashes["execution.json"]=study.file_hash(path)
    return {"available":True,"observed_envelope_seconds":seconds,"ledger":value,
            "scope":"External process observations include startup, orchestration and polling; overlapping process intervals are not dedicated GPU-hours."}


def _verify_job(directory, stage, job, spec, protocol, banks, protected, verification, hashes):
    folder=directory/stage/job
    final=torch.load(folder/"learner/backend.pt",map_location="cpu",weights_only=True)
    study.validate_backend(final,stage,spec)
    learner=study.backend(banks,protected,stage,spec,payload=final)
    report=_read(folder/"report.json")
    marker={"schema":study.SCHEMA,"stage":stage,"job":job,"recipe":spec,
        "protocol_sha256":protocol["_sha256"],"execution_profile":protocol["profile_proof"]["execution_profile"],
        "selection_sha256":hashes["selection.json"] if stage=="main" else None}
    _same(_read(folder/"job.json"),marker)
    if (report.get("schema")!=study.SCHEMA or report.get("stage")!=stage or report.get("job")!=job
            or report["recipe"]!=spec or report["protocol_sha256"]!=protocol["_sha256"]
            or report["updates"]!=study.TOTALS[stage] or learner.updates!=study.TOTALS[stage]
            or report["heldout_evaluation_performed"] is not False or report["automatic_promotion"] is not False
            or report["backend_file_sha256"]!=hashes[f"{stage}/{job}/learner/backend.pt"]
            or report["job_marker_sha256"]!=hashes[f"{stage}/{job}/job.json"]
            or report["retained_step_seconds"]!=final["accounting"]["retained_step_seconds"]):
        raise ValueError("completed job report binding differs")
    _same(report["execution_profile"],protocol["profile_proof"]["execution_profile"])
    _same(report["stream"],stream_evidence(final["learner"]["realization"]))
    _same(report["exposures"],final["learner"]["exposures"])
    calls=_load_invocations(directory,stage,job,report,protocol,hashes)
    manifest={name:_stats(rows) for name,rows in banks[stage]["development"].items()}
    development=[];digests={}
    for step in study.STEPS[stage]:
        saved=torch.load(folder/f"checkpoint-{step:06d}.pt",map_location="cpu",weights_only=True)
        study.validate_snapshot(saved,protocol,stage,job,banks,spec)
        learner.restore(saved["backend"])
        if learner.updates!=step:
            raise ValueError("checkpoint step differs from its filename")
        _same(stream_evidence(saved["backend"]["learner"]["realization"]),verification["replayed_streams"][str(step)])
        digest=checkpoint_digest(learner.model)
        if digest!=saved["weights_sha256"] or digest!=verification["jobs"][job]["weights_sha256"][str(step)]:
            raise ValueError("checkpoint weight digest differs")
        if step==0 and digest!=protocol["initial_weights_sha256"][stage][str(spec["width"])]:
            raise ValueError("initialization differs")
        evidence.validate_metrics(saved["development"],banks[stage]["development"],manifest,study.config(spec["width"]),"dev")
        seconds=_finite(saved["development_seconds"],"development time")
        development.append({"updates":step,"episodes_per_family":step*study.MICRO,"seconds":seconds,
                            "weights_sha256":digest,"metrics":saved["development"]})
        digests[str(step)]=digest
        if step==study.TOTALS[stage]:
            _validate_endpoint_binding(final,saved["backend"],verification["replayed_streams"][str(step)])
            if digest!=report["weights_sha256"] or report["endpoint_file_sha256"]!=hashes[f"{stage}/{job}/checkpoint-{step:06d}.pt"]:
                raise ValueError("endpoint identity differs")
    output={k:copy.deepcopy(report[k]) for k in ("recipe","updates","exposures","stream","retained_step_seconds","weights_sha256","invocation_files_sha256")}
    output.update(parameters=sum(p.numel() for p in learner.model.parameters()),generation_timing_subset=copy.deepcopy(final["learner"]["realization"]["timing"]),
                  development_curve=development,invocation_costs=calls)
    return output,learner,digests


def verify_completed(directory):
    directory=Path(directory)
    paths=_complete_gate(directory)
    hashes={p.relative_to(directory).as_posix():study.file_hash(p) for p in paths}
    helper_hashes={"summary":study.file_hash(__file__),"metric_helper":study.file_hash(metric_helper.__file__)}
    protocol=study.load_protocol(directory)
    external_execution=_execution(directory,hashes)
    selection=study.load_selection(directory)
    verifications={stage:study.load_verification(directory,stage) for stage in study.TOTALS}
    for stage,receipt in verifications.items():
        if (receipt.get("neural_training_or_audit_performed") is not False
                or set(receipt["replayed_streams"])!={str(n) for n in study.STEPS[stage]}
                or set(receipt["jobs"])!=set(study.jobs(stage))
                or any(row.get("exact_final_backend_binding") is not True for row in receipt["jobs"].values())):
            raise ValueError("sealed stage verification fields differ")
    banks,protected=study.data(directory)
    _same(bank_manifest(banks),protocol["bank_manifest"])
    _same(verify_boundaries(banks),protocol["bank_diagnostics"]["boundaries"])
    _same(historical_evidence(),protocol["bank_diagnostics"]["historical_evidence"])
    planned=torch.load(directory/"planned-calibration.pt",map_location="cpu",weights_only=True)
    _same(stream_evidence(planned),verifications["calibration"]["replayed_streams"][str(study.TOTALS["calibration"])])
    _same(protected["calibration"],protected_transcripts(banks,stage="calibration"))
    _same(protected["main"],protected_transcripts(banks,stage="main",consumed_calibration=planned["seen_transcripts"]))
    _same({stage:len(values) for stage,values in protected.items()},protocol["protected_counts"])
    for stage in study.TOTALS:
        _same(study.structural_evidence(banks[stage]["train"],study.TOTALS[stage],study.SAMPLERS[stage]),protocol["expected_final_structural_streams"][stage])
        for width in study.WIDTHS:
            model=build_sequence_student(study.SEEDS[stage],config=study.config(width))
            if checkpoint_digest(model)!=protocol["initial_weights_sha256"][stage][str(width)]:
                raise ValueError("prescribed seed/config initialization differs")
            del model
    marker={"schema":study.SCHEMA,"protocol_sha256":hashes["protocol.json"],"selection_sha256":hashes["selection.json"],
            "input_files_sha256":study.completed_inputs(directory,"main"),"verification_sha256":hashes["verification/main.json"]}
    audit=_read(directory/"audit/report.json")
    if (audit.get("schema")!=study.SCHEMA or audit.get("base_checkpoints_unchanged") is not True
            or audit.get("automatic_promotion") is not False or set(audit.get("results",{}))!=set(study.jobs("main"))):
        raise ValueError("completed audit fields differ")
    _same(audit["protocol"],protocol);_same(audit["inputs"],marker)
    _same(_read(directory/"audit/evaluation-started.json"),marker)
    validated_protocol={**protocol,"_sha256":hashes["protocol.json"]}
    stages,weight_digests={},{ }
    for stage in study.TOTALS:
        stages[stage]={};weight_digests[stage]={}
        for job,declared in study.jobs(stage).items():
            spec=dict(declared)
            if stage=="main":spec["rate"]=selection["selected"][str(spec["width"])]["rate"]
            row,learner,digests=_verify_job(directory,stage,job,spec,validated_protocol,banks,protected,verifications[stage],hashes)
            stages[stage][job]=row;weight_digests[stage][job]=digests
            if stage=="main":
                value=audit["results"][job]
                latest,metadata=learner._trainer.latest_observed_banks()
                initial_rows={f"initial/{f}/t{t}":rs for f,bs in banks[stage]["train"].items() for t,rs in bs.items()}
                latest_rows={f"latest/{f}/t{t}":rs for f,bs in latest.items() for t,rs in bs.items()}
                _validated_audit(value,_read(directory/"audit"/f"{job}.json"),job=job,inputs=marker,steps=study.STEPS["main"],config=study.config(spec["width"]),
                    expected_weights=digests,audit_rows=banks["main"]["audit"],initial_rows=initial_rows,latest_rows=latest_rows,latest_metadata=metadata)
            del learner
    result=assemble_summary(protocol,selection,stages,audit)
    result["external_execution"]=external_execution
    coverages=[Path(name) for name in protocol["profile_proof"]["files_sha256"] if Path(name).name=="coverage.json"]
    preparation=_read(directory/"preparation.json")
    if (preparation.get("schema")!=study.SCHEMA or preparation.get("protocol_sha256")!=hashes["protocol.json"]
            or preparation.get("neural_training_or_evaluation_performed") is not False
            or preparation.get("planned_calibration_updates")!=study.TOTALS["calibration"]
            or preparation.get("planned_calibration_seen_transcripts")!=len(planned["seen_transcripts"])):
        raise ValueError("preparation receipt differs from the frozen plan")
    _finite(preparation["wall_seconds"],"preparation wall")
    result["separate_setup_and_probes"]={"profile_probe_coverage":_read(coverages[0]) if len(coverages)==1 else None,
        "prepared_utc":protocol.get("prepared_utc"),"preparation":preparation,
        "scope":"Profile probes precede study training and are outside retained study totals. Preparation includes bank/proof authentication, planned calibration replay, initialization digests and frozen file writes, excluding Python import/startup. Invocation setup is separately recorded within invocation wall."}
    expected_updates=sum(study.TOTALS[stage]*len(study.jobs(stage)) for stage in study.TOTALS)
    if result["compute"]["total"]["retained_optimizer_updates"]!=expected_updates or result["compute"]["total"]["sampled_exposures"]["episodes"]!=expected_updates*study.MICRO*len(study.FAMILIES):
        raise ValueError("complete study budget differs")
    result["verification"]={"input_file_sha256":hashes,"source_sha256":protocol["source_sha256"],
        "verified_source_files":len(protocol["source_sha256"]),"helper_sha256":helper_hashes,
        "checkpoint_weights_sha256":weight_digests,"bank_boundaries":protocol["bank_diagnostics"]["boundaries"],
        "protected_counts":protocol["protected_counts"],"historical_evidence":protocol["bank_diagnostics"]["historical_evidence"],
        "profile_proof":protocol["profile_proof"],"no_neural_inference_or_training_performed":True,
        "stream_note":"Every optimizer checkpoint restored on CPU and bound to sealed canonical replay; the verifier does not reproduce AdamW arithmetic or claim cross-device bitwise optimization."}
    study.load_protocol(directory);study.load_selection(directory)
    for stage in study.TOTALS:study.load_verification(directory,stage)
    if any(study.file_hash(directory/name)!=sha for name,sha in hashes.items()) or study.file_hash(__file__)!=helper_hashes["summary"] or study.file_hash(metric_helper.__file__)!=helper_hashes["metric_helper"]:
        raise RuntimeError("evidence changed during summary verification")
    _same(historical_evidence(),protocol["bank_diagnostics"]["historical_evidence"])
    _check_finite_tree(result,"capacity summary")
    return result


def summarize(directory):
    directory=Path(directory)
    _complete_gate(directory)
    with run_lock(directory/"audit"):
        result=verify_completed(directory)
        atomic_json(directory/"audit/summary.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True,type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    result=summarize(args.study)
    print(json.dumps({"compute":result["compute"],"verified_source_files":result["verification"]["verified_source_files"]},indent=2))


if __name__=="__main__":main()
