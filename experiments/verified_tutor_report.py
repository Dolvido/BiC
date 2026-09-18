"""Pure recount of native teaching/withdrawal outcomes; no model execution."""
import argparse
from fractions import Fraction

from experiments.foundation_layout_study import ROOT, digest, read, publish, verify_pins, encoded
from experiments.foundation_layout_evaluation import score_records
from experiments.shared_state_screen import compare


def analyze(directory, *, launch_sha256, summary_sha256):
    from pathlib import Path
    directory=Path(directory).resolve()
    if digest(directory/"launch.json")!=launch_sha256 or digest(directory/"execution/summary.json")!=summary_sha256:
        raise ValueError("caller-pinned completed launch and summary required")
    launch,summary=read(directory/"launch.json"),read(directory/"execution/summary.json")
    if (launch["schema"]!="bic-verified-tutor-run-v1" or summary["status"]!="completed"
            or summary["launch_sha256"]!=launch_sha256):
        raise ValueError("completed matching experiment required")
    verify_pins(launch["source_sha256"]);verify_pins(launch["input_sha256"])
    for name,pin in summary["artifact_sha256"].items():
        if digest(directory/"execution"/name)!=pin:raise ValueError("result artifact changed")
    if digest(directory/"execution/events.jsonl")!=summary["journal_sha256"]:
        raise ValueError("execution journal changed")
    metrics={};episodes=endpoints=0
    if set(summary["evaluations"])!={"procedural","tutor"}:raise ValueError("two fixed arms required")
    for arm,steps in summary["evaluations"].items():
        if set(steps)!={"0","108","216"}:raise ValueError("complete fixed endpoint schedule required")
        metrics[arm]={}
        for step,banks in steps.items():
            if set(banks)!={"dev","train_fit","retention"}:raise ValueError("three fixed banks required")
            metrics[arm][step]={}
            for bank,saved in banks.items():
                records=[]
                for group in saved["groups"].values():
                    if digest(ROOT/group["path"])!=group["sha256"]:raise ValueError("raw score image changed")
                    image=read(ROOT/group["path"])
                    if score_records(image["raw_records"],expected_bank=image["bank"])!=image["metrics"]:
                        raise ValueError("raw group arithmetic differs")
                    records.extend(image["raw_records"])
                recounted=score_records(records)
                if recounted!=saved["metrics"]:raise ValueError("pooled arithmetic differs")
                metrics[arm][step][bank]=recounted;episodes+=len(records);endpoints+=1
    if (episodes,endpoints)!=(9288,18):raise ValueError("declared full evaluation census differs")
    if metrics["procedural"]["0"]!=metrics["tutor"]["0"]:raise ValueError("matched baseline differs")
    comparisons={step:compare(metrics["procedural"][step],metrics["tutor"][step]) for step in ("108","216")}
    acquisitions={arm:compare(values["0"],values["216"]) for arm,values in metrics.items()}
    withdrawal={}
    for arm,values in metrics.items():
        changes={}
        for bank in ("dev","train_fit","retention"):
            for family in ("color","count","switch"):
                first=values["108"][bank]["by_family"][family]["counts"]
                final=values["216"][bank]["by_family"][family]["counts"]
                for name in ("known_action","known_reply","unknown_action","unknown_reply"):
                    old,new=first[name],final[name]
                    if old["total"]!=new["total"] or old["total"]<1:raise ValueError("fixed withdrawal slice denominator required")
                    change=Fraction(new["count"]-old["count"],old["total"])
                    changes[f"{bank}/{family}/{name}"]=dict(before=old["count"],after=new["count"],
                        total=old["total"],percentage_points=float(100*change),passed=change>=-Fraction(1,20))
        withdrawal[arm]=dict(passed=all(x["passed"] for x in changes.values()),changes=changes)
    for arm,work in summary["physical_training"].items():
        if (work["work"]["synchronized_optimizer_updates"]!=216
                or work["work"]["retained_episodes"]!=20736
                or work["work"]["unknown_optimizer_outcomes"]):
            raise ValueError("complete known physical training workload required")
    selected=launch["teacher_decision"]
    if digest(ROOT/selected["path"])!=selected["sha256"]:raise ValueError("teacher result changed")
    teacher=read(ROOT/selected["path"])
    treatment=launch["treatment"]
    nonzero=treatment["different_teaching_bundles"]>0
    if nonzero == treatment["zero_treatment_contrast"]:
        raise ValueError("inconsistent materialized treatment contrast")
    benefit=bool(nonzero and comparisons["216"]["passed"] and withdrawal["tutor"]["passed"])
    report=dict(status="verified",launch_sha256=launch_sha256,summary_sha256=summary_sha256,
        comparisons=comparisons,acquisition_screens=acquisitions,withdrawal_nonregression=withdrawal,
        tutor_benefit_screen_passed=benefit,treatment=treatment,teacher_outcome=teacher["outcome"],
        teacher_cost=teacher["teacher_cost"],worker_teacher_calls=summary["teacher_calls"],
        prior_author_attempt=launch.get("prior_author_attempt"),
        prior_compilation=launch.get("prior_compilation"),
        compilation_cost=launch["compilation_cost"],
        inventory_preparation=launch["inventory_preparation"],
        total_chat_attempts=teacher["teacher_cost"]["chat_attempts"] +
            (launch["prior_author_attempt"]["teacher_cost"]["chat_attempts"] if launch.get("prior_author_attempt") else 0),
        native_metrics=metrics,evaluation_endpoints=endpoints,evaluation_episode_passes=episodes,
        physical_training=summary["physical_training"],new_updates=432,new_episode_exposures=41472,
        operations=summary["operations"],worker_seconds=summary["wall_seconds"],
        peak_gpu_allocated_bytes=summary["peak_gpu_allocated_bytes"],
        peak_gpu_reserved_bytes=summary["peak_gpu_reserved_bytes"],
        development_previously_observed=True,new_model_work=0,automatic_promotion=False,
        scope="Single-parent comparison in finite English grammar. Reused banks; pure raw recount, not independent neural evaluation. A recipe or working loop alone is not learning benefit or learned self-direction.")
    pin=publish(directory/"analysis.json",report)
    print(encoded(dict(status="verified",analysis_sha256=pin,tutor_benefit_screen_passed=benefit,
        nonzero_treatment=nonzero,teacher_outcome=teacher["outcome"])).decode().strip())
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("directory")
    parser.add_argument("--launch-sha256",required=True);parser.add_argument("--summary-sha256",required=True)
    args=parser.parse_args();analyze(args.directory,launch_sha256=args.launch_sha256,summary_sha256=args.summary_sha256)
