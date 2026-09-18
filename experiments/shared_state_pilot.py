"""Fixed paired local learning comparison; no resume, retry or promotion.

The caller freezes admitted file-backed data and reviewed source before run.
Both learners receive every bundle, with first-arm order alternated. Full
checkpoints and native raw evaluation records are retained at fixed endpoints.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import platform
import time
import traceback

from experiments.foundation_layout_study import (ROOT, CONFIG, native, digest,
    read, publish, relative_root, verify_pins, encoded, _hash, utc, denominators,
    Journal as BaseJournal)

SCHEMA = "bic-shared-state-pilot-v1"
PROTOCOL = "docs/SHARED_STATE_PILOT.md"
ARMS = ("control", "state")
STEPS = (0, 216, 432, 648)
SEED, UPDATES, MICRO, BATCH, RATE, SECONDS = 852304001, 648, 32, 32, .003, 3600
OBJECTIVE = "unchanged-sequence-objective-three-family-mean-v1"


def contract():
    return dict(schema=SCHEMA, arms=list(ARMS), seed=SEED, config=CONFIG,
        auxiliary_weights=dict(control=0., state=.3), updates_per_arm=UPDATES,
        micro_batch_size=MICRO, evaluation_batch_size=BATCH, learning_rate=RATE,
        optimizer="AdamW", gradient_clip=1., objective_id=OBJECTIVE, layout="original",
        checkpoints=list(STEPS), evaluation_banks=["dev", "train_fit", "retention"],
        device="cuda:0", max_seconds=SECONDS, cpu_threads=1, cpu_interop_threads=1,
        first_arm="control at even zero-based bundle IDs; state at odd IDs",
        initial_weights="Exact equal whole-model tensors including auxiliary heads; both policies evaluated separately at zero",
        state_targets="Computed once per authenticated bundle from causal English prefixes; labels only enter training loss",
        zero_weight="Auxiliary graph skipped exactly; unused head gradients remain None",
        automatic_retry=False, automatic_promotion=False, teacher_calls=0,
        scope="Single-seed shared learning screen; no architecture promotion or general capability claim.")


def source_pins():
    from experiments import shared_state_data as data
    from experiments import shared_state_training as kernel, foundation_layout_evaluation as evaluation
    names = set(data.source_hashes()) | set(kernel.source_hashes()) | set(evaluation.source_hashes())
    names |= {"experiments/shared_state_pilot.py", "experiments/shared_state_student.py", "experiments/shared_state_targets.py", "experiments/shared_state_screen.py", "experiments/shared_state_report.py",
              "experiments/foundation_layout_study.py", "experiments/execution_profile.py", PROTOCOL}
    if native(ROOT/"experiments/__init__.py").exists():
        raise ValueError("experiment namespace initializer must remain absent")
    return {name: digest(ROOT/name) for name in sorted(names)}


def installed_runtime():
    import torch
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                torch_git=torch.version.git_version, cuda=torch.version.cuda)


def _manifest(directory, pin):
    from experiments import shared_state_data as data
    manifest = data.load_manifest(directory, expected_manifest_sha256=pin)
    if (manifest["updates"] != UPDATES or manifest["layout"] != "original"
            or manifest["micro_batch_size"] != MICRO or len(manifest["training"]) != UPDATES
            or manifest["training_episodes"] != UPDATES*3*MICRO
            or set(manifest["bank_inventory"]) != {"dev", "train_fit", "retention"}):
        raise ValueError("fixed data schedule differs")
    for cursor, record in enumerate(manifest["training"]):
        if record["global_cursor"] != cursor or record["path"] != f"training/{cursor:04d}.pt":
            raise ValueError("canonical ordered global bundle IDs required")
    if {name: row["episodes"] for name, row in manifest["bank_inventory"].items()} != dict(dev=720, train_fit=108, retention=720):
        raise ValueError("prospective bank sizes differ")
    return manifest


def freeze(output, *, data_directory, expected_manifest_sha256, input_pins=None):
    """No model or dataset decoding; bind completed data, source and protocol."""
    started, cpu = time.monotonic(), time.process_time()
    output, directory = Path(output).resolve(), Path(data_directory).resolve()
    sources = source_pins()
    manifest = _manifest(directory, expected_manifest_sha256)
    pins = dict(input_pins or {})
    pins[relative_root(directory/"manifest.json")] = expected_manifest_sha256
    pins[relative_root(directory/"preparation.json")] = digest(directory/"preparation.json")
    parent = ROOT/manifest["parent_data_directory"]
    pins[relative_root(parent/"manifest.json")] = manifest["parent_manifest_sha256"]
    for record in manifest["training"]:
        pins[relative_root(parent/record["path"])] = record["sha256"]
    for name, pin in manifest["artifacts_sha256"].items():
        pins[relative_root(directory/name)] = pin
    verify_pins(pins)
    native(output).mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for index, (name, pin) in enumerate(sorted(sources.items())):
        raw = native(ROOT/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError("source changed while freezing")
        local = f"sources/{index:03d}.bin"
        target = native(output/local); target.parent.mkdir(exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        snapshots[name] = local
    if source_pins() != sources:
        raise ValueError("source closure changed while freezing")
    launch = dict(schema=SCHEMA, contract=contract(), data_directory=relative_root(directory),
        data_manifest_sha256=expected_manifest_sha256, source_sha256=sources,
        source_snapshots=snapshots, input_sha256=pins, installed_runtime=installed_runtime(),
        bank_inventory=manifest["bank_inventory"], created_utc=utc(),
        preparation=read(directory/"preparation.json"),
        freeze_cost=dict(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu))
    return publish(output/"launch.json", launch)


class Journal(BaseJournal):
    def __init__(self, path):
        super().__init__(path)
        self.chain = _hash([SCHEMA, "events"])


def run(output, *, expected_launch_sha256):
    started, cpu = time.monotonic(), time.process_time()
    output = Path(output).resolve()
    if digest(output/"launch.json") != expected_launch_sha256:
        raise ValueError("caller-pinned launch differs")
    launch = read(output/"launch.json")
    if launch["schema"] != SCHEMA or launch["contract"] != contract():
        raise ValueError("fixed run contract differs")
    attempt = output/"execution"; native(attempt).mkdir(exist_ok=False)
    deadline = started+SECONDS
    receipt = dict(schema=SCHEMA, status="running", launch_sha256=expected_launch_sha256,
        started_utc=utc(), pid=os.getpid(), automatic_retry=False, automatic_promotion=False,
        teacher_calls=0, artifact_sha256={}, evaluations={}, checkpoints={}, arms={},
        model_construction_attempts=0, model_constructions=0, snapshot_attempts=0, snapshots=0,
        data_load_attempts=0, data_load_completions=0, partial_work_unknown=False)
    journal, active_bank, active_ledger = None, None, None
    models, kernels, owners, prepared_banks, operations = {}, {}, {}, {}, {}
    validation_work, profile = None, None

    def boundary():
        if time.monotonic() >= deadline:
            raise TimeoutError("fixed 3600-second allowance expired; no automatic continuation")

    def event(value):
        try: journal.event(value)
        except BaseException:
            receipt["partial_work_unknown"] = True
            raise

    def operation(kind, identity, call):
        boundary()
        stats = operations.setdefault(kind, dict(attempts=0, completions=0, failures=0, wall_seconds=0., cpu_seconds=0.))
        stats["attempts"] += 1
        receipt["active_operation"] = dict(kind=kind, identity=identity)
        t, c = time.monotonic(), time.process_time()
        try:
            event(dict(event="intent", kind=kind, identity=identity))
            result = call(); stats["completions"] += 1
            event(dict(event="complete", kind=kind, identity=identity,
                wall_seconds=time.monotonic()-t, cpu_seconds=time.process_time()-c))
            receipt["active_operation"] = None
            return result
        except BaseException:
            stats["failures"] += 1
            raise
        finally:
            stats["wall_seconds"] += time.monotonic()-t
            stats["cpu_seconds"] += time.process_time()-c

    def artifact(name, value, *, checkpoint=False):
        pin = operation("publication", name, lambda: publish(attempt/name, value, checkpoint=checkpoint))
        receipt["artifact_sha256"][name] = pin
        return dict(path=relative_root(attempt/name), sha256=pin)

    def authenticate():
        if source_pins() != launch["source_sha256"] or installed_runtime() != launch["installed_runtime"]:
            raise ValueError("source or installed runtime changed")
        verify_pins(launch["input_sha256"])
        for name, local in launch["source_snapshots"].items():
            if digest(output/local) != launch["source_sha256"][name]:
                raise ValueError("frozen source copy changed")

    def load_image(record, base_directory=None):
        receipt["data_load_attempts"] += 1
        base_directory = directory if base_directory is None else base_directory
        path = (base_directory/record["path"]).resolve()
        if not path.is_relative_to(base_directory) or path == base_directory:
            raise ValueError("data artifact must stay inside admitted directory")
        raw = native(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("immutable data image pin differs")
        result = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        receipt["data_load_completions"] += 1
        return result

    def model_guard(arm):
        model = models[arm]
        if (type(model) is not SharedStateStudent or model.config != config
                or any(parameter.device != torch.device("cuda:0") or parameter.dtype != torch.float32
                       or not parameter.requires_grad for parameter in model.parameters())):
            raise ValueError("fixed shared-state model architecture/placement changed")
        execution.assert_strict_profile()

    def evaluate(arm, step):
        nonlocal active_bank, active_ledger
        model_guard(arm)
        metrics = {}
        for name, groups in prepared_banks.items():
            records, details = [], {}
            for turns, bank in groups.items():
                active_bank, active_ledger = bank, evaluation.EvaluationLedger()
                identity = f"{arm}/{step:04d}/{name}/t{turns}"
                result = operation("evaluation", identity, lambda: bank.score(models[arm], batch_size=BATCH,
                    control="normal", deadline=deadline, work=active_ledger,
                    progress=lambda value: event(dict(kind="evaluation_batch", identity=identity, **value))))
                records.extend(result["raw_records"])
                details[str(turns)] = artifact("scores/"+identity+".json", result)
                active_bank = active_ledger = None
            pooled = evaluation.score_records(records)
            if {key: value["total"] for key, value in pooled["overall"]["counts"].items()} != bank_denominators[name]:
                raise ValueError("pooled evaluation denominators differ from sealed bank")
            metrics[name] = dict(metrics=pooled, groups=details)
        model_guard(arm)
        receipt["evaluations"].setdefault(arm, {})[str(step)] = metrics
        artifact(f"scores/{arm}/{step:04d}/summary.json", metrics)

    def checkpoint(arm, step):
        model_guard(arm); receipt["snapshot_attempts"] += 1
        image = operation("snapshot", f"{arm}/{step}", kernels[arm].snapshot)
        receipt["snapshots"] += 1
        payload = dict(schema=SCHEMA, launch_sha256=expected_launch_sha256, arm=arm, step=step,
            architecture=student.ARCHITECTURE,
            auxiliary_weight=launch["contract"]["auxiliary_weights"][arm], learner=image,
            weights_sha256=checkpoint_digest(models[arm]), runtime=receipt["runtime"])
        receipt["checkpoints"].setdefault(arm, {})[str(step)] = artifact(f"checkpoints/{arm}-{step:04d}.pt", payload, checkpoint=True)

    try:
        journal = Journal(attempt/"events.jsonl")
        publish(attempt/"started.json", receipt)
        operation("authentication", "launch", authenticate)
        import torch
        from experiments import execution_profile as execution, shared_state_student as student
        from experiments.shared_state_targets import pack_state_targets
        from experiments import foundation_layout_evaluation as evaluation
        from experiments.foundation_layout_curriculum import WorkLedger
        from experiments.foundation_layout_prepared import PreparedLayoutOwner, evidence_sha256
        from experiments.foundation_evidence import json_digest
        from experiments.shared_state_training import SharedStateKernel
        from experiments.sequence_student import SequenceConfig
        from experiments.shared_state_student import SharedStateStudent, build_shared_state_student
        from brain_in_computer.dialogue_student import checkpoint_digest
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        execution.configure_strict_profile()
        receipt["runtime"] = operation("runtime", "cuda:0", execution.runtime_profile)
        if "5080" not in receipt["runtime"]["device"]["name"]:
            raise ValueError("declared local RTX 5080 required")
        artifact("runtime.json", receipt["runtime"])
        directory = ROOT/launch["data_directory"]
        manifest = operation("manifest", "data", lambda: _manifest(directory, launch["data_manifest_sha256"]))
        parent_directory = ROOT/manifest["parent_data_directory"]
        banks = operation("data_load", "banks", lambda: load_image(manifest["banks"]))
        if set(banks) != {"dev", "train_fit", "retention"}:
            raise ValueError("three fixed banks required")
        config, validation_work, bank_denominators = SequenceConfig(**CONFIG), WorkLedger(), {}
        for name in ("dev", "train_fit", "retention"):
            bank = banks[name]
            if set(bank) != {"role", "rows"} or bank["role"] != ("train_fit" if name == "train_fit" else "dev"):
                raise ValueError("fixed evaluation roles required")
            rows = bank["rows"]
            if len(rows) != manifest["bank_inventory"][name]["episodes"]:
                raise ValueError("sealed bank size differs")
            bank_denominators[name] = denominators(rows)
            grouped = defaultdict(list)
            for index in range(0, len(rows), 2):
                pair = rows[index:index+2]
                if len(pair) != 2 or len(pair[0]["turns"]) != len(pair[1]["turns"]):
                    raise ValueError("complete uniform-turn pairs required")
                grouped[len(pair[0]["turns"])].extend(pair)
            prepared_banks[name] = {turns: operation("bank_preparation", f"{name}/t{turns}",
                lambda rows=group: evaluation.PreparedLayoutBank(rows, role=bank["role"], config=config,
                    validation_work=validation_work)) for turns, group in sorted(grouped.items())}
        del banks
        artifact("bank-identities.json", {name: {str(turns): bank.identity for turns, bank in groups.items()}
                                          for name, groups in prepared_banks.items()})
        torch.cuda.reset_peak_memory_stats()
        for arm in ARMS:
            receipt["model_construction_attempts"] += 1
            models[arm] = operation("model_construction", arm, lambda: build_shared_state_student(SEED, device="cuda:0", config=config))
            receipt["model_constructions"] += 1
            model_guard(arm)
        a, b = (models[arm].state_dict() for arm in ARMS)
        if set(a) != set(b) or any(a[key].dtype != b[key].dtype or a[key].shape != b[key].shape
                                 or not torch.equal(a[key], b[key]) for key in a):
            raise ValueError("matched seeded models have different initial weights")
        receipt["initial_weights_sha256"] = checkpoint_digest(models["control"])
        del a, b
        for arm in ARMS:
            optimizer = torch.optim.AdamW(models[arm].parameters(), lr=RATE)
            kernels[arm] = SharedStateKernel(models[arm], optimizer, config=config, layout="original",
                micro_batch_size=MICRO, objective_id=OBJECTIVE,
                auxiliary_weight=launch["contract"]["auxiliary_weights"][arm])
            owners[arm] = PreparedLayoutOwner(config=config, layout="original", micro_batch_size=MICRO)
        for arm in ARMS: evaluate(arm, 0)
        for cursor, record in enumerate(manifest["training"]):
            image = operation("data_load", str(cursor), lambda: load_image(record, parent_directory))
            if (set(image) != {"bundle", "expected_evidence", "expected_evidence_sha256"}
                    or image["bundle"]["bundle_id"] != cursor or image["expected_evidence"]["bundle_id"] != cursor
                    or json_digest(image["expected_evidence"]) != image["expected_evidence_sha256"]):
                raise ValueError("immutable bundle and consumed evidence disagree")
            order = ARMS if cursor % 2 == 0 else ARMS[::-1]
            state_targets = operation("state_targets", str(cursor), lambda: {
                family: pack_state_targets(rows) for family, rows in image["bundle"]["families"].items()})
            for arm in order:
                model_guard(arm)
                token = operation("training_preparation", f"{arm}/{cursor}", lambda: owners[arm].prepare(image["bundle"],
                    expected_evidence=image["expected_evidence"], expected_evidence_sha256=evidence_sha256(image["expected_evidence"])))
                report = operation("training", f"{arm}/{cursor}", lambda: kernels[arm].step(token, state_targets=state_targets, deadline=deadline))
                event(dict(event="step_report", arm=arm, report=report))
                model_guard(arm)
            del image, state_targets
            if kernels["control"].evidence != kernels["state"].evidence:
                raise ValueError("paired arms consumed different evidence")
            step = cursor+1
            if step % 24 == 0:
                print(encoded(dict(event="progress", completed_updates_per_arm=step,
                    wall_seconds=time.monotonic()-started)).decode().strip(), flush=True)
            if step in STEPS:
                operation("authentication", str(step), authenticate)
                for arm in ARMS: checkpoint(arm, step)
                for arm in ARMS: evaluate(arm, step)
        for arm in ARMS:
            work = kernels[arm].accounting["work"]
            if (kernels[arm].cursor != UPDATES or work["retained_updates"] != UPDATES
                    or work["synchronized_optimizer_updates"] != UPDATES
                    or work["completed_forwards"] != UPDATES*3 or work["completed_backwards"] != UPDATES*3
                    or work["retained_episodes"] != UPDATES*3*MICRO or work["unknown_optimizer_outcomes"]):
                raise ValueError("completed physical training accounting differs")
            if any(value != (0 if arm == "control" else UPDATES*3)
                   for value in kernels[arm].accounting["state_work"].values()):
                raise ValueError("completed auxiliary readout/objective accounting differs")
        if operations["state_targets"]["completions"] != UPDATES:
            raise ValueError("state supervision was not computed exactly once per bundle")
        operation("authentication", "terminal", authenticate)
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        for arm, owner in owners.items():
            owner.close()
            receipt["arms"][arm] = dict(cursor=kernels[arm].cursor, accounting=kernels[arm].accounting,
                preparation=owner.report(), last_report=deepcopy(kernels[arm].last_report))
        if active_bank is not None:
            receipt["active_evaluation"] = dict(report=deepcopy(active_bank.last_report),
                work=active_ledger.report() if active_ledger is not None else None)
        try:
            if "torch" in locals() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
                receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            if journal is not None:
                journal.close(); receipt["journal_sha256"] = digest(attempt/"events.jsonl")
        except BaseException as error:
            receipt.update(cleanup_error=repr(error), partial_work_unknown=True)
            if receipt["status"] == "completed": receipt["status"] = "failed"
        receipt.update(operations=operations, bank_validation_work=validation_work.report() if validation_work else None,
            ended_utc=utc(), wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
            timing_scope="Entire execution, including data reads, source guards, training, evaluation, publications and cleanup; preparation/freeze are separate.")
        publish(attempt/"summary.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "run")); parser.add_argument("output")
    parser.add_argument("--data-directory"); parser.add_argument("--manifest-sha256")
    parser.add_argument("--launch-sha256"); parser.add_argument("--input-pins")
    args = parser.parse_args()
    if args.mode == "freeze":
        result = freeze(args.output, data_directory=args.data_directory,
            expected_manifest_sha256=args.manifest_sha256,
            input_pins=read(args.input_pins) if args.input_pins else None)
        print(encoded(dict(launch_sha256=result)).decode().strip(), flush=True)
    else:
        result = run(args.output, expected_launch_sha256=args.launch_sha256)
        print(encoded(dict(status=result["status"], wall_seconds=result["wall_seconds"])).decode().strip(), flush=True)
