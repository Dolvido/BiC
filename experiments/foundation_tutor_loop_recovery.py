"""Explicit adoption of one known completed pilot cycle and accepted tutor choice.

This is deliberately specific to the preserved first invocation. It permits
only a reviewed metadata repair; it never retries unknown neural or chat work.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import time

from experiments import foundation_tutor_loop_run as runner

ORIGINAL = "runs/foundation-tutor-loop-local/attempt-001"
PINS = {
    "launch.json": "e1b78d1b529cf840e546f6d7f9d4a747900b35e9b4ace1276815f70064029b50",
    "orchestration/receipt.json": "2fe394d3a205a5c314d68753e7a5708a00bea8ec10dffe882489c7390c504b63",
    "orchestration/worker-0/receipt.json": "10385b5478bf6c7d0c360406c39b3aabb935817a20617ef9959ede06c94ac153",
    "orchestration/worker-1/receipt.json": "f91d87fac6e9ffb812786d1e386ec48d2876bccb87a00493dd607974e64c279e",
    "orchestration/worker-1-spec.json": "128916158d1abe5ff282d2c9c9abd158f2541d46824f8ea54d46244d660e688f",
    "orchestration/decision-1/decision.json": "afb7391c58d6bcb58433f59874857037b229728b0f99d4dcacfb5c7086e799f9",
    "orchestration/request-1.json": "524ca0d7a23595936c5da7f5a46580b2bec258e09d5864a33a0b82afeced37c7",
}
REPAIRED = {"experiments/foundation_layout_continuation.py", "experiments/foundation_tutor_loop_run.py"}
ADDED = {"experiments/foundation_tutor_loop_recovery.py"}
BRIDGE_BEFORE = "79a99012587ae572d17ee9edc8b4d0ae840acfb7193ddf71dcc13bfeef45b96f"
BRIDGE_AFTER = "fcd4af775ca9c0529d2666bebee8fb349be2474fdf99906a760788ad77e9efe6"
REPAIR_PROOF = "runs/foundation-layout-continuation-json-validation-local/attempt-001/report.json"
REPAIR_PROOF_SHA256 = "3240df9484f4b5c9420513e2f8a0d0a9e0182cbef08f6aed8cf3430aa3d0c9c8"


def evidence():
    directory = runner.ROOT/ORIGINAL
    runner.verify_pins({ORIGINAL+"/"+name: pin for name, pin in PINS.items()})
    old = runner.read(directory/"launch.json")
    for name, local in old["source_snapshots"].items():
        pin = old["source_sha256"].get(name, old["input_sha256"].get(name))
        if runner.digest(directory/local) != pin:
            raise ValueError("original frozen source/input changed")
    complete = runner.read(directory/"orchestration/worker-0/receipt.json")
    failed = runner.read(directory/"orchestration/worker-1/receipt.json")
    spec = runner.read(directory/"orchestration/worker-1-spec.json")
    parent = runner.read(directory/"orchestration/receipt.json")
    if (complete["status"] != "completed" or complete["retained_cycle_updates"] != 216
            or complete["lifetime_updates"] != 216 or complete["snapshots"] != 1
            or complete["launch_sha256"] != PINS["launch.json"] or failed["status"] != "failed"
            or failed["retained_cycle_updates"] or failed["model_constructions"] or failed["model_construction_attempts"] not in (0, 1)
            or failed["snapshots"] or failed["scores"] or failed["evaluation_work"]):
        raise ValueError("only completed cycle0 and proven zero-work failed restart may be adopted")
    proof = failed["continuation_report"]
    if (any(proof[k] != 0 for k in ("model_construction_attempts", "model_constructions", "forwards", "backwards", "optimizer_updates"))
            or proof["archive_load_completions"] != 1 or proof["queued_device_work_synchronized"] is not True
            or spec["deadline_monotonic"] != spec["parent_started_monotonic"]+runner.SECONDS):
        raise ValueError("failed restart's zero-work evidence or original deadline differs")
    for name, pin in complete["artifact_sha256"].items():
        if runner.digest(directory/"orchestration/worker-0"/name) != pin:
            raise ValueError("adopted completed artifact changed")
    return old, complete, failed, spec, parent


def authenticate(launch):
    old, complete, failed, spec, parent = evidence()
    recovery = launch["recovery"]
    if (recovery["original_directory"] != ORIGINAL or recovery["adoption_sha256"] != PINS
            or recovery["parent_launch_sha256"] != PINS["launch.json"]
            or recovery["original_started_monotonic"] != spec["parent_started_monotonic"]
            or recovery["original_deadline_monotonic"] != spec["deadline_monotonic"]):
        raise ValueError("recovery adoption or original deadline differs")
    for key in ("schema", "contract", "data_directory", "data_manifest_sha256", "bank_inventory",
                "expected_work", "expected_runtime", "cycle_plan_sha256"):
        if launch[key] != old[key]:
            raise ValueError("recovery changed learning/runtime/data contract: "+key)
    before, after = old["source_sha256"], launch["source_sha256"]
    if (set(before)-set(after) or set(after)-set(before) != ADDED
            or {name for name in before if before[name] != after[name]} != REPAIRED
            or before["experiments/foundation_layout_continuation.py"] != BRIDGE_BEFORE
            or after["experiments/foundation_layout_continuation.py"] != BRIDGE_AFTER
            or recovery["source_repairs"] != {name: dict(before=before[name], after=after[name]) for name in sorted(REPAIRED)}):
        raise ValueError("only the explicit bridge/runner repair is permitted")
    if any(launch["input_sha256"].get(name) != pin for name, pin in old["input_sha256"].items()):
        raise ValueError("original input pins must remain bound")
    return complete, failed, parent


def freeze(output, *, validation_pins):
    """No models, lessons, inference, chat, or deadline extension."""
    started, cpu = time.monotonic(), time.process_time()
    old, complete, failed, spec, parent = evidence()
    runner._boundary(spec["deadline_monotonic"])
    sources = runner.source_pins()
    pins = dict(old["input_sha256"])
    pins.update({ORIGINAL+"/"+name: pin for name, pin in PINS.items()})
    pins.update(validation_pins)
    if ("docs/FOUNDATION_TUTOR_LOOP_RECOVERY.md" not in pins
            or validation_pins.get(REPAIR_PROOF) != REPAIR_PROOF_SHA256):
        raise ValueError("explicit recovery protocol and completed targeted bridge proof pins required")
    runner.verify_pins(pins)
    launch = deepcopy(old)
    launch.update(source_sha256=sources, input_sha256=pins, created_utc=runner.utc(),
        recovery=dict(original_directory=ORIGINAL, adoption_sha256=PINS, parent_launch_sha256=PINS["launch.json"],
            original_started_monotonic=spec["parent_started_monotonic"], original_deadline_monotonic=spec["deadline_monotonic"],
            source_repairs={name: dict(before=old["source_sha256"][name], after=sources[name]) for name in sorted(REPAIRED)},
            reused_cycles=[0], reused_live_decisions=[1], resumed_cycles=[1, 2], maximum_new_chat_attempts=0,
            prior_failed_invocation_wall_seconds=parent["wall_seconds"], no_prior_artifacts_relabelled=True))
    authenticate(launch)
    output = Path(output).resolve(); runner.native(output).mkdir(parents=True, exist_ok=False)
    copies = {}
    for index, (name, pin) in enumerate(sorted({**sources, **pins}.items())):
        local = f"sources/{index:03d}-{Path(name).name}"
        target = runner.native(output/local); target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(runner.native(runner.ROOT/name).read_bytes()); stream.flush(); os.fsync(stream.fileno())
        if runner.digest(target) != pin: raise ValueError("frozen recovery input changed")
        copies[name] = local
    launch.update(source_snapshots=copies, freeze_cost=dict(wall_seconds=time.monotonic()-started,
        cpu_seconds=time.process_time()-cpu, no_neural_or_chat_work=True))
    return runner.publish(output/"launch.json", launch)


def adopt(output, launch):
    """Copy the exact completed decision bytes into the new exclusive namespace."""
    from experiments import foundation_tutor_adviser as adviser
    complete, failed, parent = authenticate(launch)
    original = runner.ROOT/ORIGINAL/"orchestration"
    request = runner.read(original/"request-1.json")
    decision = adviser.load_decision(original/"decision-1/decision.json",
        expected_sha256=PINS["orchestration/decision-1/decision.json"],
        expected_request_sha256=adviser.request_sha256(request))
    if decision["outcome"] != "local_accepted" or decision["teacher_cost"]["chat_completions"] != 1:
        raise ValueError("one known accepted local choice must be reused")
    if (len(parent["decisions"]) != 1 or parent["decisions"][0]["decision"] != decision
            or parent["decisions"][0]["decision_sha256"] != PINS["orchestration/decision-1/decision.json"]):
        raise ValueError("parent receipt differs from the authenticated accepted decision")
    for local in ("request-1.json", "decision-1/intent.json", "decision-1/response.json", "decision-1/decision.json"):
        source = runner.native(original/local); target = runner.native(Path(output)/"orchestration"/local)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes()); stream.flush(); os.fsync(stream.fileno())
        if runner.digest(target) != runner.digest(source): raise ValueError("adopted decision copy differs")
    return complete, failed, parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-pins", required=True)
    args = parser.parse_args()
    print(freeze(args.output, validation_pins=runner.read(args.validation_pins)))


if __name__ == "__main__":
    main()
