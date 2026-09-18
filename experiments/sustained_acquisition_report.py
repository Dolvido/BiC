"""Pure final and trajectory recount for budgeted shared acquisition."""
import argparse
from fractions import Fraction
from pathlib import Path

from experiments.foundation_layout_study import ROOT, digest, read, publish, verify_pins, encoded
from experiments.foundation_layout_evaluation import score_records
from experiments.shared_state_screen import compare


def transfer_change(first, final):
    changes={}
    def change(name, family=None):
        left=first["overall"] if family is None else first["by_family"][family]
        right=final["overall"] if family is None else final["by_family"][family]
        a,b=left["counts"][name],right["counts"][name]
        if a["total"]!=b["total"] or a["total"]<=0:
            raise ValueError("matched positive transfer denominators required")
        delta=Fraction(b["count"]-a["count"],a["total"])
        changes[(family or "overall")+"/"+name]=dict(before=a["count"],after=b["count"],
            total=a["total"],percentage_points=float(100*delta))
        return delta
    checks={"joint_gain_at_least_5_points":change("anchor_pair_both")>=Fraction(1,20)}
    family_progress={}
    for family in ("color","count","switch"):
        family_progress[family]=change("anchor_pair_both",family)>0
        for metric in ("known_action","known_reply","unknown_action","unknown_reply"):
            checks[family+"/"+metric]=change(metric,family)>=-Fraction(1,20)
    return dict(passed=all(checks.values()),checks=checks,differences=changes,
        every_family_joint_improves=all(family_progress.values()),family_joint_progress=family_progress)


def analyze(directory, *, launch_sha256, summary_sha256):
    directory=Path(directory).resolve()
    if digest(directory/"launch.json")!=launch_sha256 or digest(directory/"execution/summary.json")!=summary_sha256:
        raise ValueError("caller-pinned completed launch and receipt required")
    launch,summary=read(directory/"launch.json"),read(directory/"execution/summary.json")
    if (launch["schema"]!="bic-sustained-shared-acquisition-v1" or summary["status"]!="completed"
            or summary["launch_sha256"]!=launch_sha256 or summary["partial_work_unknown"]):
        raise ValueError("completed known-work study required")
    verify_pins(launch["source_sha256"]);verify_pins(launch["input_sha256"])
    for name,pin in summary["artifact_sha256"].items():
        if digest(directory/"execution"/name)!=pin: raise ValueError("recorded output changed")
    if digest(directory/"execution/events.jsonl")!=summary["journal_sha256"]:
        raise ValueError("physical work journal changed")
    steps=sorted(map(int,summary["evaluations"]))
    if steps[0]!=launch["parent_cursor"] or steps[-1]!=summary["curriculum"]["cursor"]:
        raise ValueError("baseline and actual compute endpoint required")
    expected_middle=list(range(((steps[0]-2160)//648+1)*648+2160,steps[-1]+1,648))
    expected=sorted(set([steps[0],*expected_middle,steps[-1]]))
    if steps!=expected: raise ValueError("fixed complete-pass and final evaluation schedule required")
    metrics={};episodes=queries=groups=0
    for step in steps:
        banks=summary["evaluations"][str(step)]
        if set(banks)!=set(launch["contract"]["bank_counts"]): raise ValueError("all native bank roles required")
        metrics[str(step)]={}
        for bank,saved in banks.items():
            records=[]
            for group in saved["groups"].values():
                if digest(ROOT/group["path"])!=group["sha256"]: raise ValueError("raw native group changed")
                image=read(ROOT/group["path"])
                if score_records(image["raw_records"],expected_bank=image["bank"])!=image["metrics"]:
                    raise ValueError("raw group arithmetic differs")
                records.extend(image["raw_records"]);groups+=1
            if len(records)!=launch["contract"]["bank_counts"][bank]: raise ValueError("bank episode census differs")
            recounted=score_records(records)
            if recounted!=saved["metrics"]: raise ValueError("pooled metric arithmetic differs")
            metrics[str(step)][bank]=recounted;episodes+=len(records)
            queries+=sum(len(row["queries"]) for row in records)
    physical=summary["physical_training"]["work"]
    updates=steps[-1]-steps[0]
    if (physical["synchronized_optimizer_updates"]!=updates or physical["retained_updates"]!=updates
            or physical["retained_episodes"]!=updates*96 or physical["unknown_optimizer_outcomes"]
            or summary["new_updates_returned"]!=updates or summary["teacher_calls"]!=0):
        raise ValueError("complete physical learning/work accounting required")
    for saved in summary["commits"]:
        if digest(ROOT/saved["path"])!=saved["sha256"]: raise ValueError("controller commit changed")
        commit=read(ROOT/saved["path"])
        if (commit["launch_sha256"]!=launch_sha256
                or commit["data_manifest_sha256"]!=launch["data_manifest_sha256"]
                or digest(ROOT/commit["checkpoint"]["path"])!=commit["checkpoint"]["sha256"]):
            raise ValueError("committed learner/controller identities differ")
    first,last=metrics[str(steps[0])],metrics[str(steps[-1])]
    acquisition=compare(first,last)
    transfer={name:transfer_change(first[name],last[name]) for name in ("transfer_original","transfer_varied")}
    gaps={}
    for step,banks in metrics.items():
        a=banks["transfer_original"]["overall"]["counts"]["anchor_pair_both"]
        b=banks["transfer_varied"]["overall"]["counts"]["anchor_pair_both"]
        if a["total"]!=b["total"]: raise ValueError("paired transfer views require equal denominator")
        gaps[step]=dict(original=a["count"],varied=b["count"],total=a["total"],
            varied_minus_original_points=float(100*Fraction(b["count"]-a["count"],a["total"])))
    report=dict(schema="bic-sustained-shared-acquisition-report-v1",status="verified",
        launch_sha256=launch_sha256,summary_sha256=summary_sha256,stop_reason=summary["stop_reason"],
        evaluation_cursors=steps,evaluation_endpoints=len(steps)*5,raw_groups=groups,
        evaluation_episode_passes=episodes,evaluation_query_passes=queries,
        native_metrics=metrics,acquisition_screen=acquisition,transfer_screens=transfer,
        transfer_layout_gaps=gaps,new_updates=updates,new_episode_exposures=updates*96,
        original_unique_training_rows=62208,new_unique_training_rows=0,
        curriculum=summary["curriculum"],physical_training=summary["physical_training"],
        worker_seconds=summary["wall_seconds"],worker_cpu_seconds=summary["cpu_seconds"],
        operations=summary["operations"],replay_work=summary["replay_work"],
        data_preparation_seconds=launch["data_preparation_cost"]["wall_seconds"],
        host_peak_rss=summary["host_peak_rss"],peak_gpu_allocated_bytes=summary["peak_gpu_allocated_bytes"],
        peak_gpu_reserved_bytes=summary["peak_gpu_reserved_bytes"],teacher_calls=0,new_model_work=0,
        automatic_promotion=False,scope="One continuing research learner; repeated finite structural/layout transfer diagnostics, no pristine audit or general-English claim.")
    pin=publish(directory/"analysis.json",report)
    print(encoded(dict(status="verified",analysis_sha256=pin,new_updates=updates,
        acquisition_passed=acquisition["passed"],transfer_passed={k:v["passed"] for k,v in transfer.items()})).decode().strip())
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("directory");p.add_argument("--launch-sha256",required=True)
    p.add_argument("--summary-sha256",required=True);args=p.parse_args()
    analyze(args.directory,launch_sha256=args.launch_sha256,summary_sha256=args.summary_sha256)
