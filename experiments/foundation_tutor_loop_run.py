"""One local tutor-loop pilot: three sequential workers, no automatic retry.

Freeze already admitted data before execution. A CPU parent makes one bounded
local adviser decision between workers; only workers construct the learner.
Restoration carries full AdamW state. This is a pipeline demonstration, not an
estimate of tutor benefit, autonomous learned direction, or model promotion.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.foundation_layout_study import (ROOT, CONFIG, CONTEXT_LAUNCH,
    CONTEXT_LAUNCH_SHA256, native, digest, read, publish, relative_root,
    verify_pins, encoded, _hash, utc, denominators, _encoder_positions,
    counted_data_load, Journal as BaseJournal)

SCHEMA = "bic-foundation-tutor-loop-v1"
SECONDS, UPDATES, MICRO, BATCH, RATE = 3600, 216, 32, 32, .003
PLAN_CONFIG = dict(stage_updates=24, final_updates=72, micro_batch_size=32, rehearsal_every=4)
TEACHER_SHA256 = "f04aa1c738f64e13c625b82ae92504fc0260fa6723b509ed1ece0fa188179b1d"
SCOPE = "Local pipeline proof only; no causal tutor-benefit, learned self-direction, general intelligence, or automatic promotion claim."


class Journal(BaseJournal):
    def __init__(self, path):
        super().__init__(path)
        self.chain = _hash([SCHEMA, "events"])


def source_pins():
    from experiments import foundation_tutor_loop_data as data, foundation_tutor_adviser as adviser
    from experiments import foundation_layout_continuation as bridge, foundation_layout_evaluation as evaluation
    names = set(data.source_hashes()) | set(adviser.source_hashes()) | set(bridge.source_hashes()) | set(evaluation.source_hashes())
    names |= {"experiments/foundation_tutor_loop_run.py", "experiments/foundation_tutor_loop_recovery.py",
              "experiments/foundation_layout_study.py", "experiments/execution_profile.py"}
    if native(ROOT/"experiments/__init__.py").exists():
        raise ValueError("experiment namespace initializer must remain absent")
    return {name: digest(ROOT/name) for name in sorted(names)}


def contract(seed, teacher_model):
    if type(seed) is not int or seed != 852104001 or teacher_model != "ministral-3:3b":
        raise ValueError("fixed seed 852104001 and installed ministral-3:3b required")
    return dict(schema=SCHEMA, seed=seed, config=CONFIG, learning_rate=RATE, plan_config=PLAN_CONFIG,
        cycles=3, updates_per_cycle=UPDATES, lifetime_updates=3*UPDATES, layout="original",
        orders=["curriculum", "one persisted local choice: curriculum or mixed", "curriculum"],
        max_seconds=SECONDS, device="cuda:0", evaluation_batch_size=BATCH,
        teacher=dict(model=teacher_model, sha256=TEACHER_SHA256), teacher_max_seconds=180,
        model_constructions=3, fresh_initializations=1, full_optimizer_restorations=2,
        full_checkpoint_publications=3, maximum_chat_attempts=1, teacher_keep_alive=0,
        automatic_retry=False, automatic_promotion=False, scope=SCOPE)


def schedule(cycle):
    before = ["retention"] if cycle == 0 else [f"acquisition-{cycle}"]
    after = ["retention"] + ([] if cycle == 0 else [f"acquisition-{cycle}"])
    if cycle == 2:
        after.append("audit")
    return dict(before=before, after=after)


def inventory(banks):
    result = {}
    for bank_id, bank in sorted(banks.items()):
        grouped = defaultdict(list)
        for name, rows in sorted(bank["rows"].items()):
            turns = {len(row["turns"]) for row in rows}
            if len(turns) != 1 or not rows or len(rows) % 2:
                raise ValueError("complete uniform-turn cells required")
            grouped[next(iter(turns))].append(name)
        groups = {}
        for turns, cells in sorted(grouped.items()):
            rows = [row for cell in cells for row in bank["rows"][cell]]
            mapping = {row["id"]: cell for cell in cells for row in bank["rows"][cell]}
            if len(mapping) != len(rows):
                raise ValueError("duplicate evaluation episode identity")
            groups[str(turns)] = dict(cells=cells, cell_by_episode=mapping, rows_sha256=_hash(rows),
                episodes=len(rows), turns=turns, denominators=denominators(rows),
                cell_denominators={cell: denominators(bank["rows"][cell]) for cell in cells},
                encoder_positions=_encoder_positions(rows)["normal"], batches=(len(rows)+BATCH-1)//BATCH)
        result[bank_id] = dict(role=bank["role"], groups=groups,
            episodes=sum(group["episodes"] for group in groups.values()))
    return result


def expected_work(census):
    work = dict(model_constructions=3, fresh_initializations=1, full_optimizer_restorations=2,
        optimizer_updates=648, training_forwards=1944, training_backwards=1944,
        training_episodes=62208, full_checkpoints=3, evaluation_sequence_calls=0,
        evaluation_episodes=0, bos_contexts=0, free_contexts=0,
        encoder_actual_positions=0, encoder_padded_positions=0)
    for cycle in range(3):
        for bank_id in schedule(cycle)["before"]+schedule(cycle)["after"]:
            for group in census[bank_id]["groups"].values():
                work["evaluation_sequence_calls"] += group["batches"]
                work["evaluation_episodes"] += group["episodes"]
                work["bos_contexts"] += group["episodes"]*group["turns"]
                work["free_contexts"] += group["episodes"]*group["turns"]
                work["encoder_actual_positions"] += group["encoder_positions"]["actual"]
                work["encoder_padded_positions"] += group["encoder_positions"]["padded"]
    work["free_decoder_recurrent_calls_max"] = work["evaluation_sequence_calls"]*33
    work["free_decoder_row_steps_max"] = work["free_contexts"]*33
    return work


def freeze(output, *, data_directory, expected_manifest_sha256, seed, teacher_model, input_pins):
    """CPU-only inventory and immutable source/input copies; never generates data."""
    from experiments import foundation_tutor_loop_data as data
    started, cpu = time.monotonic(), time.process_time()
    output, directory = Path(output).resolve(), Path(data_directory).resolve()
    pins = dict(input_pins)
    pins[CONTEXT_LAUNCH] = CONTEXT_LAUNCH_SHA256
    pins[relative_root(directory/"manifest.json")] = expected_manifest_sha256
    verify_pins(pins)
    sources = source_pins()
    counters = dict(weights_only_bank_loads_attempted=0, weights_only_bank_loads_completed=0)
    sealed = counted_data_load(lambda: data.load(directory, expected_manifest_sha256=expected_manifest_sha256, include_audit=True), counters)
    if [cycle["cycle_id"] for cycle in sealed["cycles"]] != [0, 1, 2]:
        raise ValueError("three fixed pilot cycles required")
    for cycle in sealed["cycles"]:
        plan = cycle["plan"]
        if ({key: plan["config"][key] for key in PLAN_CONFIG} != PLAN_CONFIG
                or plan["schedules"]["curriculum"] != list(range(UPDATES))
                or sorted(plan["schedules"]["mixed"]) != list(range(UPDATES))):
            raise ValueError("fixed matched pilot plan required")
    census = inventory(sealed["banks"])
    if set(census) != {"retention", "acquisition-1", "acquisition-2", "audit"} or any(
            value["episodes"] != (792 if name == "audit" else 720) for name, value in census.items()):
        raise ValueError("frozen pilot evaluation coverage differs")
    native(output).mkdir(parents=True, exist_ok=False)
    copies = {}
    for index, (name, pin) in enumerate(sorted({**sources, **pins}.items())):
        local = f"sources/{index:03d}-{Path(name).name}"
        target = native(output/local); target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(native(ROOT/name).read_bytes()); stream.flush(); os.fsync(stream.fileno())
        if digest(target) != pin:
            raise ValueError("source/input changed while freezing")
        copies[name] = local
    if source_pins() != sources:
        raise ValueError("source changed during freeze")
    launch = dict(schema=SCHEMA, contract=contract(seed, teacher_model), source_sha256=sources,
        input_sha256=pins, source_snapshots=copies, data_directory=relative_root(directory),
        data_manifest_sha256=expected_manifest_sha256, bank_inventory=census,
        expected_work=expected_work(census), expected_runtime=read(ROOT/CONTEXT_LAUNCH)["expected_runtime"],
        cycle_plan_sha256=[_hash(cycle["plan"]) for cycle in sealed["cycles"]],
        freeze_cost=dict(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu, **counters),
        created_utc=utc(), scope=SCOPE)
    return publish(output/"launch.json", launch)


def authenticate(output, pin):
    if digest(output/"launch.json") != pin:
        raise ValueError("launch digest differs")
    launch = read(output/"launch.json")
    c = launch["contract"]
    if (launch["schema"] != SCHEMA or c != contract(c["seed"], c["teacher"]["model"])
            or source_pins() != launch["source_sha256"]
            or launch["expected_work"] != expected_work(launch["bank_inventory"])):
        raise ValueError("executable pilot contract/source differs")
    verify_pins(launch["input_sha256"])
    for name, local in launch["source_snapshots"].items():
        expected = launch["source_sha256"].get(name, launch["input_sha256"].get(name))
        if digest(output/local) != expected:
            raise ValueError("frozen source/input copy differs")
    if "recovery" in launch:
        from experiments.foundation_tutor_loop_recovery import authenticate as authenticate_recovery
        authenticate_recovery(launch)
    return launch


def _boundary(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("fixed pilot orchestration allowance expired; no automatic retry")


def _inner(learner):
    return getattr(learner, "_trainer", learner)


def worker(output, *, spec_path, expected_spec_sha256):
    """One fresh OS process, one model construction, one complete cycle."""
    started, cpu = time.monotonic(), time.process_time()
    output, spec_path = Path(output).resolve(), Path(spec_path).resolve()
    if digest(spec_path) != expected_spec_sha256:
        raise ValueError("worker spec pin differs")
    spec = read(spec_path); cycle = spec["cycle"]
    if type(cycle) is not int or cycle not in range(3) or spec["order"] not in ("curriculum", "mixed"):
        raise ValueError("bounded cycle/order required")
    launch = authenticate(output, spec["launch_sha256"])
    deadline = spec["deadline_monotonic"]
    if not started < deadline <= spec["parent_started_monotonic"]+SECONDS:
        raise ValueError("expired or extended worker allowance")
    attempt = output/"orchestration"/f"worker-{cycle}"
    native(attempt).mkdir(exist_ok=False)
    receipt = dict(schema=SCHEMA, status="running", cycle=cycle, order=spec["order"], pid=os.getpid(),
        spec_sha256=expected_spec_sha256, launch_sha256=spec["launch_sha256"], started_utc=utc(),
        model_construction_attempts=0, model_constructions=0, snapshot_attempts=0, snapshots=0,
        weights_only_bank_loads_attempted=0, weights_only_bank_loads_completed=0,
        retained_cycle_updates=0, scores={}, artifact_sha256={}, teacher_calls=0, automatic_retry=False)
    journal = Journal(attempt/"events.jsonl")
    learner, data_work, validation_work, active_evaluator = None, None, None, None
    prepared, ledgers, operations = {}, [], {}

    def op(kind, identity, call):
        _boundary(deadline)
        stats = operations.setdefault(kind, dict(attempts=0, completions=0, wall_seconds=0., cpu_seconds=0.))
        stats["attempts"] += 1
        receipt["active_operation"] = dict(kind=kind, identity=identity)
        journal.event(dict(event="intent", kind=kind, identity=identity))
        t, c = time.monotonic(), time.process_time()
        try:
            value = call(); stats["completions"] += 1
            journal.event(dict(event="complete", kind=kind, identity=identity))
            receipt["active_operation"] = None
            return value
        finally:
            stats["wall_seconds"] += time.monotonic()-t; stats["cpu_seconds"] += time.process_time()-c

    def artifact(name, value, checkpoint=False):
        pin = op("publication", name, lambda: publish(attempt/name, value, checkpoint=checkpoint))
        receipt["artifact_sha256"][name] = pin
        return dict(path=relative_root(attempt/name), sha256=pin)

    def evaluate(bank_id, when):
        nonlocal active_evaluator
        if bank_id == "audit" and (cycle != 2 or when != "after" or learner.cursor != 648):
            raise ValueError("audit is endpoint-only")
        if bank_id not in banks:
            banks[bank_id] = op("bank_load", bank_id, lambda: counted_data_load(lambda: data.load_bank(
                ROOT/launch["data_directory"], bank_id, expected_manifest_sha256=launch["data_manifest_sha256"]), receipt))
        frozen = launch["bank_inventory"][bank_id]
        if inventory({bank_id: banks[bank_id]}) != {bank_id: frozen}:
            raise ValueError("bank census differs from frozen launch")
        records, group_files = [], {}
        for group_id, group in frozen["groups"].items():
            key = bank_id+"/"+group_id
            if key not in prepared:
                rows = [row for cell in group["cells"] for row in banks[bank_id]["rows"][cell]]
                prepared[key] = op("bank_compile", key, lambda: evaluation.PreparedLayoutBank(
                    rows, role=frozen["role"], config=SequenceConfig(**CONFIG), validation_work=validation_work))
                if prepared[key].identity["sha256"] != group["rows_sha256"]:
                    raise ValueError("compiled bank identity differs")
            active_evaluator = prepared[key]
            ledger = evaluation.EvaluationLedger(); ledgers.append(ledger)
            result = op("evaluation", when+"/"+key, lambda: active_evaluator.score(learner.model,
                batch_size=BATCH, control="normal", deadline=deadline, work=ledger,
                progress=lambda value: journal.event(dict(bank=key, when=when, **value))))
            if (result["status"] != "completed" or not result["model_state_unchanged"]
                    or not result["training_modes_restored"]
                    or result["metrics"] != evaluation.score_records(result["raw_records"], expected_bank=result["bank"])
                    or {k: v["total"] for k, v in result["metrics"]["overall"]["counts"].items()} != group["denominators"]):
                raise ValueError("evaluation integrity or denominator mismatch")
            cells = defaultdict(list)
            for row in result["raw_records"]:
                cells[group["cell_by_episode"][row["episode_id"]]].append(row)
            cell_metrics = {name: evaluation.score_records(rows) for name, rows in cells.items()}
            if {name: {k: v["total"] for k, v in metric["overall"]["counts"].items()} for name, metric in cell_metrics.items()} != group["cell_denominators"]:
                raise ValueError("cell denominators differ")
            group_files[group_id] = artifact(f"scores/{when}-{bank_id}-{group_id}.json",
                dict(result=result, cell_by_episode=group["cell_by_episode"], cell_metrics=cell_metrics))
            records.extend(result["raw_records"]); active_evaluator = None
        summary = dict(bank_id=bank_id, when=when, lifetime_updates=learner.cursor,
            weights_sha256=checkpoint_digest(learner.model), metrics=evaluation.score_records(records), groups=group_files)
        item = artifact(f"scores/{when}-{bank_id}.json", summary)
        receipt["scores"][when+"/"+bank_id] = dict(**item, **summary)

    try:
        _boundary(deadline)
        if cycle:
            from experiments import foundation_tutor_adviser as adviser
            request = read(output/"orchestration"/f"request-{cycle}.json")
            decision = op("decision_authentication", str(cycle), lambda: adviser.load_decision(
                output/"orchestration"/f"decision-{cycle}"/"decision.json",
                expected_sha256=spec["decision_sha256"], expected_request_sha256=adviser.request_sha256(request)))
            if (decision["profile"]["order"] != spec["order"] or decision["profile"]["layout"] != "original"
                    or decision["mode"] != ("local" if cycle == 1 else "withdrawn")
                    or decision["parent"]["lifetime_updates"] != cycle*UPDATES):
                raise ValueError("worker schedule differs from its persisted decision")
        import torch
        from experiments import execution_profile as execution, foundation_tutor_loop_data as data
        from experiments import foundation_layout_training as training, foundation_layout_evaluation as evaluation
        from experiments.foundation_layout_continuation import LayoutContinuation
        from experiments.foundation_layout_curriculum import WorkLedger
        from experiments.sequence_student import SequenceConfig
        from brain_in_computer.dialogue_student import checkpoint_digest
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("strict FP32 default required")
        execution.configure_strict_profile()
        runtime = op("runtime", "cuda:0", lambda: execution.runtime_profile("cuda:0"))
        if runtime != launch["expected_runtime"]:
            raise ValueError("local runtime differs from pinned strict runtime")
        receipt["execution_profile"] = runtime; torch.cuda.reset_peak_memory_stats()
        data_work, validation_work = data.WorkLedger(), WorkLedger()
        data_work.deadline, data_work.phase = deadline, "runtime_materialization"
        sealed = op("data_load", "plans/dev", lambda: counted_data_load(lambda: data.load(
            ROOT/launch["data_directory"], expected_manifest_sha256=launch["data_manifest_sha256"], include_audit=False), receipt))
        record, banks = sealed["cycles"][cycle], sealed["banks"]
        if _hash(record["plan"]) != launch["cycle_plan_sha256"][cycle] or "audit" in banks:
            raise ValueError("cycle plan differs or audit loaded early")
        receipt["model_construction_attempts"] += 1
        if cycle == 0:
            if spec["order"] != "curriculum" or spec.get("parent") is not None:
                raise ValueError("fixed procedural bootstrap required")
            learner = op("fresh_initialization", "cycle-0", lambda: training.FoundationLayoutTrainer(
                seed=launch["contract"]["seed"], config=SequenceConfig(**CONFIG), learning_rate=RATE,
                micro_batch_size=MICRO, layout="original", objective_id=training.OBJECTIVE_ID, device="cuda:0"))
        else:
            parent = spec["parent"]
            if digest(ROOT/parent["receipt_path"]) != parent["receipt_sha256"]:
                raise ValueError("parent receipt changed")
            parent_receipt = read(ROOT/parent["receipt_path"])
            producing_launch = (launch["recovery"]["parent_launch_sha256"]
                if cycle == 1 and "recovery" in launch else spec["launch_sha256"])
            if (parent_receipt["status"] != "completed" or parent_receipt["cycle"] != cycle-1
                    or parent_receipt["launch_sha256"] != producing_launch
                    or decision["parent"]["weights_sha256"] != parent_receipt["weights_sha256"]):
                raise ValueError("completed producing parent differs from the decision")
            archive = parent_receipt["checkpoint"]
            image = native(ROOT/archive["path"]).read_bytes()
            loader = LayoutContinuation.from_pilot if cycle == 1 else LayoutContinuation.from_snapshot
            learner = op("full_optimizer_restore", f"cycle-{cycle}", lambda: loader(image,
                expected_sha256=archive["sha256"], expected_identity=parent["origin_identity"], device="cuda:0"))
            receipt["restore_report"] = learner.last_restore_report
            if (learner.cursor != cycle*UPDATES or _inner(learner)._evidence != parent_receipt["evidence"]
                    or checkpoint_digest(learner.model) != parent_receipt["weights_sha256"]):
                raise ValueError("restored parent weights/cursor/evidence differs")
            previous = parent_receipt["scores"]["after/retention"]
            receipt["reused_before_retention"] = dict(source=previous["path"], sha256=previous["sha256"],
                weights_sha256=previous["weights_sha256"], zero_new_forwards=True)
        receipt["model_constructions"] += 1
        if any(t.dtype != torch.float32 or t.device.type != "cuda" for t in learner.model.state_dict().values()):
            raise ValueError("only FP32 CUDA learner state accepted")
        receipt["initial_weights_sha256"] = checkpoint_digest(learner.model)
        receipt["initial_cursor"] = learner.cursor
        accumulator = deepcopy(_inner(learner)._evidence)
        for bank_id in schedule(cycle)["before"]:
            evaluate(bank_id, "before")
        for canonical_id in record["plan"]["schedules"][spec["order"]]:
            execution.assert_strict_profile()
            cursor = learner.cursor
            expected = data.expected_bundle(record, canonical_id, cursor)
            bundle = op("materialize", str(canonical_id), lambda: data.materialize_bundle(
                record["plan"], canonical_id, global_cursor=cursor, work=data_work))
            if data_work.last_bundle_evidence != expected:
                raise ValueError("materialized lesson differs before any forward")
            report = op("learner_step", str(cursor), lambda: learner.step(bundle))
            committed = report.get("trainer_report", report)
            accumulator = training._accumulate(accumulator, expected)
            if (committed["bundle_evidence"] != expected or _inner(learner)._evidence != accumulator
                    or learner.cursor != cursor+1 or report["retained_updates"] != 1):
                raise ValueError("committed learner evidence differs")
            receipt["retained_cycle_updates"] += 1
            journal.event(dict(event="retained_step", canonical_bundle_id=canonical_id, report=report,
                consumed_bundle_identity_sha256=accumulator["consumed_bundle_identity_sha256"]))
            if learner.cursor % 24 == 0:
                print(encoded(dict(event="progress", cycle=cycle, lifetime_updates=learner.cursor)).decode().strip(), flush=True)
        for bank_id in schedule(cycle)["after"]:
            evaluate(bank_id, "after")
        receipt["weights_sha256"] = checkpoint_digest(learner.model)
        receipt["evidence"] = deepcopy(_inner(learner)._evidence)
        receipt["snapshot_attempts"] += 1
        snapshot = op("snapshot", "full-weights-and-AdamW", learner.snapshot)
        receipt["snapshots"] += 1
        if cycle == 0:
            identity = dict(producing_schema=SCHEMA, launch_sha256=spec["launch_sha256"], arm="original",
                step=learner.cursor, weights_sha256=receipt["weights_sha256"], recipe=learner.recipe,
                evidence=receipt["evidence"], execution_profile=runtime)
            snapshot = dict(schema=SCHEMA, arm="original", step=learner.cursor,
                launch_sha256=spec["launch_sha256"], learner=snapshot,
                weights_sha256=receipt["weights_sha256"], execution_profile=runtime)
        else:
            identity = spec["parent"]["origin_identity"]
        receipt["origin_identity"] = identity
        receipt["checkpoint"] = artifact("checkpoint.pt", snapshot, checkpoint=True)
        del snapshot
        op("authentication", "final", lambda: authenticate(output, spec["launch_sha256"]))
        execution.assert_strict_profile()
        if runtime != execution.runtime_profile("cuda:0") or learner.cursor != (cycle+1)*UPDATES:
            raise ValueError("runtime or final cursor differs")
        receipt.update(status="completed", lifetime_updates=learner.cursor)
    except BaseException as error:
        receipt.update(status="incomplete" if isinstance(error, TimeoutError) else "failed",
            error=repr(error), traceback=traceback.format_exc())
        for attribute in ("continuation_report", "layout_report"):
            if hasattr(error, attribute): receipt[attribute] = getattr(error, attribute)
        if hasattr(error, "continuation_report") and "model_constructions" in error.continuation_report:
            receipt["model_construction_attempts"] = error.continuation_report["model_construction_attempts"]
            receipt["model_constructions"] = error.continuation_report["model_constructions"]
        raise
    finally:
        if learner is not None:
            receipt.update(accounting=learner.accounting, evidence=deepcopy(_inner(learner)._evidence),
                last_step_report=deepcopy(learner.last_report), last_snapshot_report=deepcopy(learner.last_snapshot_report))
        if active_evaluator is not None: receipt["last_evaluation_report"] = active_evaluator.last_report
        receipt["evaluation_work"] = {key: sum(ledger.counts[key] for ledger in ledgers) for key in ledgers[0].counts} if ledgers else {}
        receipt["materialization_work"] = data_work.report() if data_work is not None else None
        receipt["evaluation_validation_work"] = validation_work.report() if validation_work is not None else None
        t, c = time.monotonic(), time.process_time()
        try:
            learner = None; prepared.clear(); gc.collect()
            if "torch" in locals() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
                receipt.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), shutdown_device_synchronized=True)
                torch.cuda.empty_cache()
        except BaseException as error:
            receipt.update(status="failed", shutdown_error=repr(error), physical_work_unknown=True)
        receipt["shutdown_cost"] = dict(wall_seconds=time.monotonic()-t, cpu_seconds=time.process_time()-c)
        journal.event(dict(event="worker_finished", status=receipt["status"])); journal.close()
        receipt.update(operations=operations, journal_sha256=digest(attempt/"events.jsonl"),
            wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
            timing_scope="Worker interval including cleanup, excluding final receipt publication; parent includes process startup, exit and receipt I/O.")
        publish(attempt/"receipt.json", receipt)
    return receipt


def request_for(launch, parent, reference, cycle, mode):
    from experiments import foundation_tutor_adviser as adviser
    def evidence(score):
        return dict(weights_sha256=score["weights_sha256"], lifetime_updates=score["lifetime_updates"],
            by_family={family: {name: {k: score["metrics"]["by_family"][family]["counts"][name][k]
                for k in ("count", "total")} for name in adviser.METRICS} for family in adviser.FAMILIES})
    return dict(schema=adviser.REQUEST_SCHEMA,
        parent=dict(identity_sha256=_hash(dict(checkpoint=parent["checkpoint"], evidence=parent["evidence"])),
            weights_sha256=parent["weights_sha256"], cycle=cycle, lifetime_updates=cycle*UPDATES),
        catalogue=dict(schema=adviser.CATALOGUE_SCHEMA, profiles=[dict(id=order, order=order,
            layout="original", plan_config=PLAN_CONFIG) for order in ("curriculum", "mixed")]),
        development=dict(schema=adviser.EVIDENCE_SCHEMA, role="development",
            current=evidence(parent["scores"]["after/retention"]), reference=evidence(reference)),
        procedural_profile_id="curriculum", mode=mode,
        teacher=launch["contract"]["teacher"] if mode == "local" else None,
        max_seconds=launch["contract"]["teacher_max_seconds"], source_sha256=adviser.source_hashes())


def run(output, *, expected_launch_sha256):
    """CPU parent; waits for each worker to exit before any adviser request."""
    started, cpu = time.monotonic(), time.process_time()
    output = Path(output).resolve(); launch = authenticate(output, expected_launch_sha256)
    attempt = output/"orchestration"; native(attempt).mkdir(exist_ok=False)
    recovery = launch.get("recovery")
    original_started = recovery["original_started_monotonic"] if recovery else started
    deadline = recovery["original_deadline_monotonic"] if recovery else started+SECONDS
    receipt = dict(schema=SCHEMA, status="running", launch_sha256=expected_launch_sha256,
        pid=os.getpid(), started_utc=utc(), workers=[], decisions=[], automatic_retry=False,
        automatic_promotion=False, scope=SCOPE)
    journal = Journal(attempt/"events.jsonl")
    workers, terminal_workers, process = [], [], None
    try:
        from experiments import foundation_tutor_adviser as adviser
        if recovery:
            from experiments.foundation_tutor_loop_recovery import adopt, ORIGINAL, PINS
            complete, failed, prior = adopt(output, launch)
            workers.append(complete); terminal_workers.extend([complete, failed])
            receipt["workers"].append(dict(path=ORIGINAL+"/orchestration/worker-0/receipt.json",
                sha256=PINS["orchestration/worker-0/receipt.json"], returncode=0, adopted=True))
            receipt["decisions"] = deepcopy(prior["decisions"])
            receipt["adoption"] = dict(recovery=recovery, failed_restart_receipt_sha256=PINS["orchestration/worker-1/receipt.json"],
                prior_parent_cpu_seconds=prior["parent_cpu_seconds"], prior_parent_wall_seconds=prior["wall_seconds"],
                new_chat_attempts=0, repeated_completed_training_updates=0, repeated_completed_evaluations=0)
            journal.event(dict(event="explicit_adoption", completed_cycle=0, accepted_local_decision=1,
                failed_restart_zero_model_and_update_work=True, original_deadline_monotonic=deadline))
        for cycle in range(1 if recovery else 0, 3):
            _boundary(deadline); order = "curriculum"
            if recovery and cycle == 1:
                order = receipt["decisions"][0]["decision"]["profile"]["order"]
                journal.event(dict(event="accepted_decision_reused", cycle=1,
                    decision_sha256=receipt["decisions"][0]["decision_sha256"], new_api_calls=0))
            elif cycle:
                parent = workers[-1]
                reference = workers[0]["scores"]["before/retention"] if cycle == 1 else workers[0]["scores"]["after/retention"]
                request = request_for(launch, parent, reference, cycle, "local" if cycle == 1 else "withdrawn")
                slot = attempt/f"decision-{cycle}"
                request_pin = publish(attempt/f"request-{cycle}.json", request)
                journal.event(dict(event="adviser_intent", cycle=cycle, request_file_sha256=request_pin,
                    prior_worker_exited=True, mode=request["mode"]))
                decision = adviser.decide(slot, request=request,
                    expected_request_sha256=adviser.request_sha256(request), deadline=deadline)
                receipt["decisions"].append(decision)
                journal.event(dict(event="adviser_returned", cycle=cycle, decision_sha256=decision["decision_sha256"]))
                order = decision["decision"]["profile"]["order"]
                if cycle == 2 and (order != "curriculum" or decision["decision"]["teacher_cost"]["api_attempts"] != 0):
                    raise ValueError("teacher withdrawal must use procedural order and zero API calls")
            parent_spec = None if cycle == 0 else dict(receipt_path=receipt["workers"][-1]["path"],
                receipt_sha256=receipt["workers"][-1]["sha256"], origin_identity=workers[0]["origin_identity"])
            spec = dict(schema=SCHEMA, cycle=cycle, order=order, parent=parent_spec,
                launch_sha256=expected_launch_sha256, parent_started_monotonic=original_started,
                deadline_monotonic=deadline, decision_sha256=None if cycle == 0 else receipt["decisions"][-1]["decision_sha256"])
            spec_path = attempt/f"worker-{cycle}-spec.json"; spec_pin = publish(spec_path, spec)
            command = [sys.executable, "-m", "experiments.foundation_tutor_loop_run", "worker", "--output", str(output),
                "--spec", str(spec_path), "--spec-sha256", spec_pin]
            command_pin = publish(attempt/f"worker-{cycle}-command.json", dict(argv=command, cwd=str(ROOT)))
            journal.event(dict(event="worker_intent", cycle=cycle, spec_sha256=spec_pin, command_sha256=command_pin))
            with native(attempt/f"worker-{cycle}-stdout.txt").open("xb") as stdout, native(attempt/f"worker-{cycle}-stderr.txt").open("xb") as stderr:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                try:
                    returncode = process.wait(timeout=max(.001, deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                    receipt["terminated_worker"] = dict(cycle=cycle, pid=process.pid, physical_work_unknown=True)
                    raise TimeoutError("worker exceeded whole-loop deadline; process terminated without retry")
            journal.event(dict(event="worker_exited", cycle=cycle, pid=process.pid, returncode=returncode))
            process = None
            path = attempt/f"worker-{cycle}"/"receipt.json"
            if not native(path).exists(): raise RuntimeError("worker exited without a terminal receipt")
            child = read(path); child_pin = digest(path)
            terminal_workers.append(child)
            receipt["workers"].append(dict(path=relative_root(path), sha256=child_pin, returncode=returncode))
            if returncode != 0 or child["status"] != "completed" or child["spec_sha256"] != spec_pin:
                raise RuntimeError("worker failed; receipt retained and no automatic retry")
            for name, pin in child["artifact_sha256"].items():
                if digest(path.parent/name) != pin: raise ValueError("worker artifact digest differs")
            if child["retained_cycle_updates"] != UPDATES or child["model_constructions"] != 1 or child["snapshots"] != 1:
                raise ValueError("worker did not complete its exact construction/update/checkpoint allowance")
            workers.append(child)
        authenticate(output, expected_launch_sha256); _boundary(deadline)
        final = workers[-1]["accounting"]["lifetime_trainer"]["work"]
        expected = dict(retained_updates=648, retained_episodes=62208, completed_forwards=1944,
            completed_backwards=1944, synchronized_optimizer_updates=648, unknown_optimizer_outcomes=0)
        if any(final[k] != v for k, v in expected.items()): raise ValueError("lifetime training work differs")
        evaluation = {key: sum(w["evaluation_work"][key] for w in workers) for key in workers[0]["evaluation_work"]}
        budget = launch["expected_work"]
        for key, amount in dict(sequence_forward_completions=budget["evaluation_sequence_calls"],
                scored_episode_records=budget["evaluation_episodes"], bos_reply_context_completions=budget["bos_contexts"],
                free_reply_context_completions=budget["free_contexts"],
                encoder_actual_position_completions=budget["encoder_actual_positions"],
                encoder_padded_position_completions=budget["encoder_padded_positions"]).items():
            if evaluation[key] != amount: raise ValueError("evaluation work differs: "+key)
        if (evaluation["sequence_forward_attempts"] != evaluation["sequence_forward_completions"]
                or evaluation["decoder_recurrent_attempts"] != evaluation["decoder_recurrent_completions"]
                or evaluation["decoder_recurrent_completions"] > budget["free_decoder_recurrent_calls_max"]
                or evaluation["decoder_row_step_completions"] > budget["free_decoder_row_steps_max"]):
            raise ValueError("unmatched or excessive inference work")
        receipt.update(status="completed", lifetime_training_work=final, evaluation_work=evaluation,
            live_tutor_choice_accepted=receipt["decisions"][0]["decision"]["outcome"] == "local_accepted",
            actual_learning_process_restarts=2, final_checkpoint=workers[-1]["checkpoint"],
            retention_trajectory=[workers[0]["scores"]["before/retention"]]+[w["scores"]["after/retention"] for w in workers],
            acquisition=[dict(cycle=i, before=w["scores"]["before/"+("retention" if i == 0 else f"acquisition-{i}")],
                after=w["scores"]["after/"+("retention" if i == 0 else f"acquisition-{i}")]) for i, w in enumerate(workers)],
            final_audit=workers[-1]["scores"]["after/audit"])
    except BaseException as error:
        receipt.update(status="incomplete" if isinstance(error, TimeoutError) else "failed",
            error=repr(error), traceback=traceback.format_exc())
        if hasattr(error, "tutor_report"): receipt["uncommitted_tutor_report"] = error.tutor_report
        raise
    finally:
        if process is not None and process.poll() is None:
            process.kill(); process.wait(); receipt["physical_work_unknown"] = True
        journal.event(dict(event="orchestration_finished", status=receipt["status"])); journal.close()
        receipt.update(journal_sha256=digest(attempt/"events.jsonl"),
            wall_seconds=time.monotonic()-started, parent_cpu_seconds=time.process_time()-cpu,
            cumulative_elapsed_seconds_from_original_start=time.monotonic()-original_started,
            cumulative_parent_cpu_seconds=time.process_time()-cpu+receipt.get("adoption", {}).get("prior_parent_cpu_seconds", 0.),
            all_terminal_worker_cpu_seconds=sum(w["cpu_seconds"] for w in terminal_workers),
            all_terminal_worker_wall_seconds=sum(w["wall_seconds"] for w in terminal_workers),
            completed_worker_cpu_seconds=sum(w["cpu_seconds"] for w in workers),
            completed_worker_wall_seconds=sum(w["wall_seconds"] for w in workers),
            timing_scope="Parent wall encloses workers, tutor, publication and process exits. Worker and adviser durations are nested, not added to parent wall. CPU includes parent plus separately reported workers; tutor service CPU is not available.")
        publish(attempt/"receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freezing = sub.add_parser("freeze"); freezing.add_argument("--output", required=True)
    freezing.add_argument("--data-directory", required=True); freezing.add_argument("--manifest-sha256", required=True)
    freezing.add_argument("--seed", type=int, required=True); freezing.add_argument("--teacher-model", required=True)
    freezing.add_argument("--input-pins", required=True, help="Reviewed JSON mapping of protocol/validation/preparation receipt paths to SHA256")
    running = sub.add_parser("run"); running.add_argument("--output", required=True); running.add_argument("--launch-sha256", required=True)
    child = sub.add_parser("worker"); child.add_argument("--output", required=True)
    child.add_argument("--spec", required=True); child.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.output, data_directory=args.data_directory, expected_manifest_sha256=args.manifest_sha256,
            seed=args.seed, teacher_model=args.teacher_model, input_pins=read(args.input_pins))
        print(result)
    elif args.command == "run":
        result = run(args.output, expected_launch_sha256=args.launch_sha256)
        print(encoded(dict(status=result["status"], live_tutor_choice_accepted=result["live_tutor_choice_accepted"])).decode().strip())
    else:
        result = worker(args.output, spec_path=args.spec, expected_spec_sha256=args.spec_sha256)
        if result["status"] != "completed": raise RuntimeError("worker cleanup did not complete")


if __name__ == "__main__":
    main()
