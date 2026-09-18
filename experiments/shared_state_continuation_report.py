"""Recount saved evaluation records and apply the prospective pilot screen."""
import argparse
import json
from pathlib import Path

from experiments.foundation_tutor_loop_data import digest, native, read, write
from experiments.foundation_layout_evaluation import score_records
from experiments.shared_state_screen import compare

ROOT = Path(__file__).resolve().parents[1]


def analyze(directory):
    directory = Path(directory).resolve()
    summary = read(directory / "execution/summary.json")
    launch = read(directory / "launch.json")
    if (summary["status"] != "completed" or digest(directory / "launch.json") != summary["launch_sha256"]
            or launch["schema"] != "bic-shared-state-continuation-pilot-v1"):
        raise ValueError("completed, matching launch and execution required")
    for name, pin in launch["source_sha256"].items():
        if digest(ROOT / name) != pin:
            raise ValueError("frozen screen or supporting source changed: " + name)
    if set(summary["evaluations"]) != {"fast", "slow"}:
        raise ValueError("both fixed arms required")
    for name, pin in summary["artifact_sha256"].items():
        if digest(directory / "execution" / name) != pin:
            raise ValueError("saved result artifact changed: " + name)
    if digest(directory / "execution/events.jsonl") != summary["journal_sha256"]:
        raise ValueError("execution journal changed")
    recounted, evaluation_episodes = 0, 0
    for arm, endpoints in summary["evaluations"].items():
        if set(endpoints) != {"648", "1296", "1944"}:
            raise ValueError("complete fixed evaluation endpoints required")
        for step, banks in endpoints.items():
            if set(banks) != {"dev", "previous_dev", "train_fit", "retention"}:
                raise ValueError("all four fixed banks required")
            for name, saved in banks.items():
                records = []
                for group in saved["groups"].values():
                    path = ROOT / group["path"]
                    if digest(path) != group["sha256"]:
                        raise ValueError("raw evaluation group changed")
                    scored = read(path)
                    if score_records(scored["raw_records"], expected_bank=scored["bank"]) != scored["metrics"]:
                        raise ValueError("group arithmetic differs")
                    records.extend(scored["raw_records"])
                if score_records(records) != saved["metrics"]:
                    raise ValueError("pooled arithmetic differs")
                recounted += 1
                evaluation_episodes += len(records)
    if recounted != 24 or evaluation_episodes != 13608:
        raise ValueError("complete prospective evaluation census required")
    endpoints = {arm: {name: record["metrics"] for name, record in steps["1944"].items()}
                 for arm, steps in summary["evaluations"].items()}
    screen = compare(endpoints["fast"], endpoints["slow"])
    baselines = {arm:{name:record["metrics"] for name,record in steps["648"].items()}
                 for arm,steps in summary["evaluations"].items()}
    acquisition = {arm:compare(baselines[arm], endpoints[arm]) for arm in endpoints}
    prior_checks, prior_differences = {}, {}
    for arm in endpoints:
        for family in ("color","count","switch"):
            first=baselines[arm]["previous_dev"]["by_family"][family]["counts"]
            final=endpoints[arm]["previous_dev"]["by_family"][family]["counts"]
            for metric in ("known_action","known_reply","unknown_action","unknown_reply"):
                old,new=first[metric],final[metric]
                if old["total"]!=new["total"] or old["total"]<=0:raise ValueError("fixed previous-dev denominators required")
                key=f"{arm}/{family}/{metric}";delta=new["count"]-old["count"]
                prior_checks[key]=20*delta>=-old["total"]
                prior_differences[key]=dict(baseline=old["count"],final=new["count"],total=old["total"],percentage_points=100*delta/old["total"])
    previous_dev_regression=dict(checks=prior_checks,differences=prior_differences,passed=all(prior_checks.values()),
        scope="Each arm against its own restored648 baseline; every factual/unknown family slice loss<=5points")
    timing = {arm: dict(training=0., training_preparation=0., evaluation=0.) for arm in endpoints}
    for line in native(directory / "execution/events.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("event") == "complete" and entry.get("kind") in ("training", "training_preparation", "evaluation"):
            arm = entry["identity"].split("/")[0]
            timing[arm][entry["kind"]] += entry["wall_seconds"]
    for costs in timing.values():
        costs["training_including_preparation"] = costs["training"] + costs["training_preparation"]
    extra_seconds = timing["slow"]["training_including_preparation"] - timing["fast"]["training_including_preparation"]
    joint = screen["differences"]["dev/overall/anchor_pair_both"]
    extra_correct = joint["candidate"] - joint["baseline"]
    target_seconds = summary["operations"]["state_targets"]["wall_seconds"]
    result = dict(status="verified", summary_sha256=digest(directory / "execution/summary.json"),
        launch_sha256=summary["launch_sha256"], recounted_bank_endpoints=recounted,
        evaluation_episode_exposures=evaluation_episodes, screen=screen, endpoint_metrics=endpoints,
        timing=timing, execution_seconds=summary["wall_seconds"],
        new_data_preparation_seconds=launch["preparation"]["wall_seconds"],
        development_bank_previously_observed=False, previous_development_bank_previously_observed=True,
        acquisition_screens=acquisition, previous_dev_nonregression=previous_dev_regression,
        baseline_metrics=baselines,
        new_training_updates=2592, new_training_episode_exposures=248832,
        lifetime_updates_per_arm=1944, lifetime_episode_exposures_per_arm=186624,
        unique_cached_training_rows=62208,
        replay_work=summary["replay_work"], restoration=summary["restoration"], freeze_work=launch["freeze_work"],
        per_arm_acquisition={arm:dict(joint_pair_gain=endpoints[arm]["dev"]["overall"]["counts"]["anchor_pair_both"]["count"]-baselines[arm]["dev"]["overall"]["counts"]["anchor_pair_both"]["count"],
            training_seconds=timing[arm]["training_including_preparation"]) for arm in endpoints},
        freeze_seconds=launch["freeze_cost"]["wall_seconds"],
        peak_gpu_allocated_bytes=summary["cuda_peak_allocated_bytes"],
        peak_gpu_reserved_bytes=summary["cuda_peak_reserved_bytes"],
        parameter_counts=summary["parameter_counts"],
        incremental_correct_pairs=extra_correct, incremental_training_seconds=extra_seconds,
        shared_target_preparation_seconds=target_seconds,
        shared_target_cost_is_common_to_both_arms=True,
        incremental_correct_pairs_per_extra_training_second=(extra_correct / extra_seconds if extra_seconds > 0 else None),
        independent_neural_evaluation=False, new_model_work=0, teacher_calls=0,
        scope="Pure recount of recorded predictions, not another model evaluation; one exploratory initialization. Both original rate histories continued with full AdamW on repeated broad lessons; new development bank plus three reused banks. Between-arm and within-arm screens are distinct. No promotion or broad capability claim.")
    write(directory / "analysis.json", result)
    print(json.dumps({"status":result["status"],"analysis_sha256":digest(directory / "analysis.json"),"screen_passed":screen["passed"],"acquisition_passed":{a:v["passed"] for a,v in acquisition.items()},"previous_dev_nonregression":previous_dev_regression["passed"],"recounted_bank_endpoints":recounted,"evaluation_episodes":evaluation_episodes}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    analyze(parser.parse_args().directory)
