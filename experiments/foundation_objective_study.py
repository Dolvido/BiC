"""One fixed objective comparison on explicitly reused foundation benchmarks.

Prepare performs no model construction. Each exclusive run-stage keeps one real
canonical index alive through six fresh jobs, all checkpoint restores, and then
scoring. No scoring precedes completion and verification of every stage job.
An interrupted stage is retained, never resumed or retried by this runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time


SCHEMA = "bic-foundation-objective-study-v1"
OBJECTIVES = ("baseline", "balanced_reply")
RATES = (.0003, .001, .003)
SEEDS = (8472, 8473, 8474)
CALIBRATION_SEED = 8471
TOTALS = {"calibration": 510, "main": 3072}
STEPS = {"calibration": (0, 510), "main": (0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072)}
CONFIG_VALUES = dict(width=192, layers=4, heads=4, feedforward=768, max_positions=1024,
                     max_turns=12, max_input_bytes=128, max_output_bytes=32)
PROTOCOL_SOURCE = "docs/FOUNDATION_OBJECTIVE_STUDY_PROTOCOL.md"
DATA_ROOT = "runs/variant-study-local/data"
DATA_GATE = "runs/variant-study-local/data-verification/receipt.json"
DATA_SHA256 = "d0170189ffa373c68492972df83ed7e39aaa1c40f2cc5c9520b89454f0672be6"
GATE_SHA256 = "585bf512466d55b759d2e0bd96595783b2b5b1892952c96be9d774625f56cd33"
PRIOR_PROTOCOL_SHA256 = "5aa70402ec5c2c3edd1f8f9a94709b8c08cd94dbacb1bbcc88c48a0b3728b5e3"
BENCHMARK_SCOPE = ("All reused development/audit banks were previously viewed. "
    "Three seeds replicate initialization on the same lessons/banks; no untouched confirmation or automatic promotion.")


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_bytes())


def _same(a, b, message):
    if _encoded(a) != _encoded(b):
        raise ValueError(message)


def _write(path, value, *, replace=False):
    path = Path(path)
    raw = _encoded(value)
    target = path.with_suffix(path.suffix + ".pending") if replace else path
    with target.open("wb" if replace else "xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    if replace:
        os.replace(target, path)
    return hashlib.sha256(raw).hexdigest()


def _config():
    from experiments.sequence_student import SequenceConfig
    return SequenceConfig(**CONFIG_VALUES)


def contract():
    return dict(schema=SCHEMA, objectives=list(OBJECTIVES), rates=list(RATES),
        seeds=list(SEEDS), calibration_seed=CALIBRATION_SEED, architecture="flat",
        config=dict(CONFIG_VALUES), order="curriculum", totals=dict(TOTALS),
        checkpoints={k: list(v) for k, v in STEPS.items()}, max_phase_seconds=14400,
        formal_updates=21492, formal_episode_exposures=2063232,
        data_manifest_sha256=DATA_SHA256, prior_data_verification_sha256=GATE_SHA256,
        benchmark_scope=BENCHMARK_SCOPE, automatic_promotion=False, automatic_retry=False,
        interpretation=PROTOCOL_SOURCE)


def source_hashes():
    from experiments.foundation_variant_study import source_hashes as prior_sources
    from experiments.foundation_objective_training import source_hashes as trainer_sources
    root = Path(__file__).resolve().parents[1]
    names = set(prior_sources()) | set(trainer_sources()) | {
        "experiments/foundation_objective_study.py",
        "experiments/foundation_objective_runtime_probe.py", PROTOCOL_SOURCE}
    return {name: _file(root / name) for name in sorted(names)}


def _sources(protocol):
    _same(source_hashes(), protocol["source_sha256"], "objective study sources changed")


def _data_gate(repository):
    root = Path(repository)
    if _file(root / DATA_GATE) != GATE_SHA256 or _file(root / DATA_ROOT / "manifest.json") != DATA_SHA256:
        raise ValueError("independently pinned reused data/verification changed")
    gate = _read(root / DATA_GATE)
    if (gate.get("schema") != "bic-foundation-variant-study-v1" or gate.get("status") != "completed"
            or gate.get("protocol_sha256") != PRIOR_PROTOCOL_SHA256
            or gate.get("neural_training_or_inference") is not False
            or set(gate.get("index_identity", {})) != set(TOTALS)):
        raise ValueError("complete prior canonical data verification required")
    expected = gate["data_files_sha256"]
    actual = {p.relative_to(root / DATA_ROOT).as_posix(): _file(p)
              for p in sorted((root / DATA_ROOT).rglob("*")) if p.is_file()}
    _same(actual, expected, "reused data files differ from completed canonical verification")
    return gate


def _proof(directory, expected_sha256):
    from experiments.foundation_objective_runtime_probe import load_proof
    value = load_proof(directory, expected_sha256=expected_sha256)
    _same(value["config"], CONFIG_VALUES, "runtime proof and study config differ")
    _same(value["objective_ids"], list(OBJECTIVES), "runtime proof objectives differ")
    _same(value["rates"], list(RATES), "runtime proof rates differ")
    return value


def prepare(repository, directory, *, proof_path, expected_proof_sha256):
    """Seal the caller-pinned proof, reused-data boundary and exact sources."""
    repository, directory = Path(repository), Path(directory)
    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory / "preparation-started.json", dict(schema=SCHEMA, status="running",
        pid=os.getpid(), started_utc=_utc()))
    proof_path = Path(proof_path).resolve()
    if proof_path.name != "probe.json":
        raise ValueError("proof path must identify the completed probe.json envelope")
    image = proof_path.read_bytes()
    if hashlib.sha256(image).hexdigest() != expected_proof_sha256:
        raise ValueError("caller-pinned runtime proof image differs")
    proof, gate, sources = _proof(proof_path.parent, expected_proof_sha256), _data_gate(repository), source_hashes()
    with (directory / "runtime-proof.json").open("xb") as stream:
        stream.write(image); stream.flush(); os.fsync(stream.fileno())
    for name, digest in sources.items():
        raw = (repository / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("source changed while preserving protocol")
        path = directory / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    protocol = dict(schema=SCHEMA, contract=contract(), source_sha256=sources,
        proof_sha256=expected_proof_sha256, execution_profile=proof["execution_profile"],
        proof_directory=str(proof_path.parent), proof_artifact_sha256=proof["artifact_sha256"],
        proof_producer_source_sha256=proof["producer_source_sha256"],
        prior_data_gate_sha256=GATE_SHA256, data_manifest_sha256=DATA_SHA256,
        index_identity=gate["index_identity"], benchmark_scope=BENCHMARK_SCOPE)
    _sources(protocol)
    pin = _write(directory / "protocol.json", protocol)
    _write(directory / "preparation.json", dict(schema=SCHEMA, status="completed",
        protocol_sha256=pin, wall_seconds=time.monotonic()-started, model_constructions=0,
        learner_updates=0, automatic_promotion=False, ended_utc=_utc(), pid=os.getpid()))
    return protocol


def load_protocol(repository, directory):
    directory = Path(directory)
    protocol, receipt = _read(directory / "protocol.json"), _read(directory / "preparation.json")
    if (protocol.get("schema") != SCHEMA or receipt.get("schema") != SCHEMA
            or receipt.get("status") != "completed"
            or receipt.get("protocol_sha256") != _file(directory / "protocol.json")):
        raise ValueError("complete objective preparation required")
    _same(protocol["contract"], contract(), "objective study contract differs")
    _sources(protocol)
    for name, digest in protocol["source_sha256"].items():
        if _file(directory / "source" / name) != digest:
            raise ValueError("preserved objective source differs")
    image = (directory / "runtime-proof.json").read_bytes()
    if hashlib.sha256(image).hexdigest() != protocol["proof_sha256"]:
        raise ValueError("preserved runtime proof changed")
    proof = _proof(protocol["proof_directory"], protocol["proof_sha256"])
    _same(proof["artifact_sha256"], protocol["proof_artifact_sha256"], "proof artifact inventory differs")
    _same(proof["producer_source_sha256"], protocol["proof_producer_source_sha256"], "proof producer sources differ")
    _same(proof["execution_profile"], protocol["execution_profile"], "proof runtime differs")
    gate = _data_gate(repository)
    _same(gate["index_identity"], protocol["index_identity"], "canonical index pins differ")
    return protocol


def jobs(stage, selection=None):
    if stage == "calibration":
        return [dict(id=f"{objective}-lr{rate:g}", objective_id=objective,
                     architecture="flat", seed=CALIBRATION_SEED, rate=rate)
                for rate in RATES for objective in OBJECTIVES]
    if stage != "main" or selection is None:
        raise ValueError("main requires sealed calibration selection")
    return [dict(id=f"{objective}-seed{seed}", objective_id=objective,
                 architecture="flat", seed=seed, rate=selection["selected"][objective]["rate"])
            for seed in SEEDS for objective in OBJECTIVES]


class _Stage:
    """Small durable phase ledger; each training update also has its own journal."""
    def __init__(self, directory, stage, max_seconds):
        if (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
                or not 0 < max_seconds <= 14400):
            raise ValueError("finite positive phase allowance at most 14400 seconds required")
        self.started, self.cpu_started = time.monotonic(), time.process_time()
        self.deadline = self.started + max_seconds
        self.folder = Path(directory) / stage
        self.folder.mkdir(exist_ok=False)
        self.guard = lambda: None
        self.receipt = dict(schema=SCHEMA, stage=stage, status="running", max_seconds=max_seconds,
            pid=os.getpid(), started_utc=_utc(), ended_utc=None, active_phase=None,
            completed_jobs=[], automatic_retry=False, automatic_promotion=False,
            limit_scope="Nonpreemptive operation boundaries; interrupted work is never retried here.")
        self.flush()

    def flush(self):
        self.receipt.update(wall_seconds=time.monotonic()-self.started,
                            cpu_seconds=time.process_time()-self.cpu_started)
        _write(self.folder / "stage.json", self.receipt, replace=True)

    def boundary(self, name):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("phase allowance reached before " + name)

    def perform(self, name, function, describe=lambda value: None):
        from experiments.foundation_study import append_journal, utc
        self.guard(); self.boundary(name)
        journal = self.folder / "phase-intents.jsonl"
        append_journal(journal, dict(event="started", name=name, utc=utc(),
            scope="Missing completion means physical work may be unknown; no automatic retry."))
        self.receipt["active_phase"] = name
        self.flush()
        started, cpu_started = time.monotonic(), time.process_time()
        evidence, error = None, None
        try:
            result = function()
            evidence = describe(result)
            return result
        except BaseException as caught:
            error = repr(caught)
            raise
        finally:
            append_journal(journal, dict(event="completed" if error is None else "failed", name=name,
                evidence=evidence, error=error, wall_seconds=time.monotonic()-started,
                cpu_seconds=time.process_time()-cpu_started, utc=utc()))
            if error is not None:
                self.receipt["failed_phase"] = name
            self.receipt["active_phase"] = None
            self.flush()


def _trainer(data, stage, job, index, device="cpu"):
    from experiments.foundation_objective_training import ObjectiveFoundationTrainer
    value = data["stages"][stage]
    return ObjectiveFoundationTrainer(value["plan"], "curriculum", objective_id=job["objective_id"],
        seed=job["seed"], config=_config(), learning_rate=job["rate"], device=device,
        admission_receipt=value["admission"], plan_index=index,
        admission_protected_transcripts=value["admission_protected_transcripts"],
        protected_transcripts=value["protected_transcripts"])


def _train_job(stage, data, job, index, protocol_pin, runtime, initial, ledger):
    import torch
    from brain_in_computer.dialogue_student import checkpoint_digest
    from experiments.execution_profile import assert_strict_profile
    from experiments.foundation_study import append_journal, account_report, utc
    from experiments.train_cognitive import atomic_checkpoint
    folder = ledger.folder / job["id"]
    folder.mkdir()
    started, cpu_started = time.monotonic(), time.process_time()
    receipt = dict(schema=SCHEMA, status="running", stage=stage, job=job, protocol_sha256=protocol_pin,
        pid=os.getpid(), started_utc=_utc(),
        execution_profile=runtime, checkpoints={}, physical_optimizer_updates=0,
        neural_attempted_episode_exposures=0, completed_microbatch_episode_exposures=0,
        drawn_episode_exposures=0, physical_work_unknown=False, automatic_promotion=False)
    _write(folder / "receipt.json", receipt)
    learner = None
    torch.cuda.reset_peak_memory_stats(0)
    try:
        ledger.boundary("fresh learner construction")
        learner = _trainer(data, stage, job, index, "cuda:0")
        receipt.update(setup_seconds=time.monotonic()-started, trainer_setup=learner.setup_report)
        for step in range(TOTALS[stage]+1):
            ledger.boundary("training update/checkpoint")
            assert_strict_profile()
            if step in STEPS[stage]:
                ledger.guard()
                payload, digest = learner.snapshot(), checkpoint_digest(learner.model)
                if step == 0:
                    expected = initial.setdefault(str(job["seed"]), digest)
                    if digest != expected:
                        raise ValueError("paired seeded initial tensors differ")
                name = f"checkpoint-{step:06d}.pt"
                append_journal(folder / "checkpoint-intents.jsonl", dict(event="started", updates=step, utc=utc()))
                atomic_checkpoint(folder / name, dict(schema=SCHEMA, stage=stage, job=job,
                    protocol_sha256=protocol_pin, learner=payload, weights_sha256=digest, execution_profile=runtime))
                receipt["checkpoints"][name] = _file(folder / name)
                append_journal(folder / "checkpoint-intents.jsonl", dict(event="completed", updates=step,
                    sha256=receipt["checkpoints"][name]))
                _write(folder / "receipt.json", receipt, replace=True)
            if step == TOTALS[stage]:
                break
            append_journal(folder / "steps.jsonl", dict(event="started", cursor=step, utc=utc()))
            try:
                report = learner.step()
            except BaseException:
                account_report(receipt, learner.last_report)
                append_journal(folder / "steps.jsonl", dict(event="failed", report=learner.last_report))
                raise
            account_report(receipt, report)
            append_journal(folder / "steps.jsonl", dict(event="completed", report=report))
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        acknowledged = max((int(name[11:17]) for name in receipt["checkpoints"]), default=0)
        receipt.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started,
            ended_utc=_utc(),
            retained_updates=None if learner is None else learner.cursor, acknowledged_checkpoint_updates=acknowledged,
            completed_updates_without_acknowledged_checkpoint=None if learner is None else learner.cursor-acknowledged,
            journal_sha256=_file(folder / "steps.jsonl") if (folder / "steps.jsonl").exists() else None,
            checkpoint_journal_sha256=_file(folder / "checkpoint-intents.jsonl") if (folder / "checkpoint-intents.jsonl").exists() else None,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(0)/2**20,
            peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20,
            timing_scope="Model setup, all steps and checkpoint I/O; excludes shared index. Materialization is inside step time.")
        _write(folder / "receipt.json", receipt, replace=True)
    return receipt


def _inputs(directory, stage, declared, protocol_pin):
    folder = Path(directory) / stage
    names, receipts = {}, {}
    expected = {f"checkpoint-{step:06d}.pt" for step in STEPS[stage]}
    for job in declared:
        prefix = folder / job["id"]
        receipt = _read(prefix / "receipt.json")
        if (receipt.get("status") != "completed" or receipt.get("schema") != SCHEMA
                or receipt.get("stage") != stage or receipt.get("job") != job
                or receipt.get("automatic_promotion") is not False
                or receipt.get("protocol_sha256") != protocol_pin or receipt.get("physical_work_unknown") is not False
                or type(receipt.get("retained_updates")) is not int or receipt["retained_updates"] != TOTALS[stage]
                or set(receipt.get("checkpoints", {})) != expected):
            raise ValueError("all six exact completed endpoints and checkpoint sets are required")
        for filename, digest in {**receipt["checkpoints"], "steps.jsonl": receipt["journal_sha256"],
                                 "checkpoint-intents.jsonl": receipt["checkpoint_journal_sha256"]}.items():
            if _file(prefix / filename) != digest:
                raise ValueError("completed training artifact changed")
            names[f"{stage}/{job['id']}/{filename}"] = digest
        _checkpoint_journal(prefix / "checkpoint-intents.jsonl", stage, receipt["checkpoints"])
        names[f"{stage}/{job['id']}/receipt.json"] = _file(prefix / "receipt.json")
        receipts[job["id"]] = receipt
    return names, receipts


def _checkpoint_journal(path, stage, checkpoints):
    events = [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines()]
    if len(events) != 2 * len(STEPS[stage]):
        raise ValueError("one durable start/completion pair per official checkpoint required")
    for step, started, completed in zip(STEPS[stage], events[::2], events[1::2]):
        if (set(started) != {"event", "updates", "utc"} or started["event"] != "started"
                or type(started["updates"]) is not int or started["updates"] != step
                or type(started["utc"]) is not str or not started["utc"]
                or set(completed) != {"event", "updates", "sha256"} or completed["event"] != "completed"
                or type(completed["updates"]) is not int or completed["updates"] != step
                or completed["sha256"] != checkpoints[f"checkpoint-{step:06d}.pt"]):
            raise ValueError("official checkpoint journal differs from sealed receipt")


def _checkpoint(directory, stage, job, step, pin, runtime, expected_sha256):
    import torch
    path = Path(directory) / stage / job["id"] / f"checkpoint-{step:06d}.pt"
    image = path.read_bytes()
    if hashlib.sha256(image).hexdigest() != expected_sha256:
        raise ValueError("checkpoint immutable byte image differs")
    saved = torch.load(io.BytesIO(image), map_location="cpu", weights_only=True)
    if (type(saved) is not dict or set(saved) != {"schema", "stage", "job", "protocol_sha256", "learner", "weights_sha256", "execution_profile"}
            or saved["schema"] != SCHEMA or saved["stage"] != stage or saved["job"] != job
            or saved["protocol_sha256"] != pin or type(saved["learner"]["cursor"]) is not int
            or saved["learner"]["cursor"] != step):
        raise ValueError("objective checkpoint envelope differs")
    _same(saved["execution_profile"], runtime, "checkpoint runtime differs")
    return saved


def _verify_job(directory, stage, job, data, index, canonical, receipt, initial, pin, runtime, ledger):
    from brain_in_computer.dialogue_student import checkpoint_digest
    from experiments import foundation_results as results
    folder = Path(directory) / stage / job["id"]
    events = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
    value = data["stages"][stage]
    journal = results.validate_journal(events, value["plan"], "curriculum", value["manifest"], canonical["bundles"])
    results._validate_receipt(receipt, journal, TOTALS[stage], 32)
    _same(receipt["execution_profile"], runtime, "training receipt runtime differs")
    learner = _trainer(data, stage, job, index)
    hashes, seconds = {}, 0.
    for step in STEPS[stage]:
        ledger.boundary("official checkpoint restore")
        saved = _checkpoint(directory, stage, job, step, pin, runtime, receipt["checkpoints"][f"checkpoint-{step:06d}.pt"])
        learner.restore(saved["learner"])
        seconds += learner.last_restore_seconds
        digest = checkpoint_digest(learner.model)
        if digest != saved["weights_sha256"] or step == 0 and digest != initial[str(job["seed"])]:
            raise ValueError("restored checkpoint weight/initialization identity differs")
        for name, field in (("retained_step_seconds", "step_seconds"),
                            ("materialization_seconds_included_in_step", "materialization_seconds")):
            if not math.isclose(saved["learner"]["timing"][name],
                    sum(events[2*i+1]["report"][field] for i in range(step)), rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError("checkpoint timing differs from complete step-journal prefix")
        hashes[str(step)] = digest
    _same(learner.snapshot()["evidence"], canonical["evidence"], "endpoint canonical prefix differs")
    expected_exposures = {family: {key: value["manifest"]["totals"][family][key] for key in results.COUNTS}
                          for family in results.FAMILIES}
    _same(canonical["evidence"]["exposures"], expected_exposures, "full plan exposure differs from canonical manifest")
    return dict(job=job, journal=journal, weights_sha256=hashes, exact_official_checkpoint_restores=True,
                restore_seconds=seconds, model_constructions_completed=1+len(hashes))


def comparison_screen(results):
    """Same conservative count arithmetic, with explicit objective/seed labels."""
    from experiments import foundation_variant_analysis as analysis
    if set(results) != set(OBJECTIVES) or any(set(rows) != set(map(str, SEEDS)) for rows in results.values()):
        raise ValueError("exact paired objective/seed results required")
    digests = {analysis._validate(rows[str(seed)], "audit") for rows in results.values() for seed in SEEDS}
    if len(digests) != 1:
        raise ValueError("comparison used different reused benchmark banks")
    panels = {}
    for panel in ("fresh_primitive", "held_composed"):
        names = [name for name in analysis.expected_cells("audit") if
            (name.startswith("fresh/") and name.split("/")[2] in ("d0", "d1")
             if panel == "fresh_primitive" else name.startswith("composed/"))]
        per_seed = {}
        for seed in map(str, SEEDS):
            per_seed[seed] = {}
            for objective in OBJECTIVES:
                rows = results[objective][seed]["per_bank"]
                per_seed[seed][objective] = dict(overall=analysis._aggregate(rows[n] for n in names),
                    by_domain={f: analysis._aggregate(rows[n] for n in names if n.split("/")[1] == f)
                               for f in analysis.FAMILIES})
            baseline, candidate = (per_seed[seed][o] for o in OBJECTIVES)
            delta = sum((analysis._counts_fraction(candidate["overall"], m)-analysis._counts_fraction(baseline["overall"], m)
                         for m in ("action", "reply")), Fraction())/2
            per_seed[seed].update(paired_composite_delta=float(delta), positive_paired_composite=delta > 0,
                by_domain_paired_delta={f: {m: float(analysis._counts_fraction(candidate["by_domain"][f], m)
                    - analysis._counts_fraction(baseline["by_domain"][f], m)) for m in ("action", "reply")}
                    for f in analysis.FAMILIES},
                candidate_nonzero={f: {m: candidate["by_domain"][f]["counts"][m+"_pairs"]["correct"] > 0
                    for m in ("action", "reply")} for f in analysis.FAMILIES})
        pooled = {}
        for family in analysis.FAMILIES:
            pooled[family] = {}
            for modality in ("action", "reply"):
                counts = {}
                for objective in OBJECTIVES:
                    rows = [per_seed[str(seed)][objective]["by_domain"][family]["counts"][modality+"_pairs"] for seed in SEEDS]
                    counts[objective] = analysis._count(sum(r["correct"] for r in rows), sum(r["total"] for r in rows))
                a, b = (counts[o] for o in reversed(OBJECTIVES))
                delta = Fraction(a["correct"], a["total"])-Fraction(b["correct"], b["total"])
                pooled[family][modality] = dict(**counts, delta=float(delta), nonnegative=delta >= 0)
        criteria = dict(every_seed_improves=all(r["positive_paired_composite"] for r in per_seed.values()),
            every_pooled_domain_modality_nonnegative=all(r["nonnegative"] for f in pooled.values() for r in f.values()),
            every_candidate_domain_seed_modality_nonzero=all(v for r in per_seed.values()
                for f in r["candidate_nonzero"].values() for v in f.values()))
        panels[panel] = dict(cell_names=names, cells=len(names), pairs_per_cell=32, per_seed=per_seed,
            pooled_by_domain_modality=pooled, criteria=criteria, passes=all(criteria.values()))
    return dict(schema=SCHEMA, panels=panels, promising_comparison=all(p["passes"] for p in panels.values()),
        bank_identity_sha256=next(iter(digests)), seed_order=list(map(str, SEEDS)), automatic_promotion=False,
        omitted_from_primary_screen="fresh composed cells remain available in the complete audit metrics",
        scope=BENCHMARK_SCOPE)


def _selection(directory, protocol):
    from experiments.foundation_variant_analysis import choose_rate
    directory = Path(directory)
    folder = directory / "calibration"
    stage, selection, evaluation, verification = (_read(folder / name)
        for name in ("stage.json", "selection.json", "evaluation.json", "verification.json"))
    pin, declared = _file(directory / "protocol.json"), jobs("calibration")
    for value in (stage, selection, evaluation, verification):
        if (value.get("schema") != SCHEMA or value.get("status") != "completed"
                or value.get("protocol_sha256") != pin or value.get("automatic_promotion") is not False):
            raise ValueError("complete matching calibration receipts required")
    for value in (stage, evaluation, verification):
        if value.get("stage") != "calibration":
            raise ValueError("calibration receipt stage differs")
        _same(value["execution_profile"], protocol["execution_profile"], "calibration runtime differs")
        _same(value["source_sha256"], protocol["source_sha256"], "calibration sources differ")
    _same(stage["jobs"], declared, "calibration jobs differ")
    _same(stage["completed_jobs"], [job["id"] for job in declared], "calibration completion order differs")
    actual = {p.relative_to(directory).as_posix(): _file(p)
              for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "stage.json"}
    _same(stage["artifacts_sha256"], actual, "complete calibration artifact inventory differs")
    inputs, _ = _inputs(directory, "calibration", declared, pin)
    for value in (evaluation, verification):
        _same(value["input_file_sha256"], inputs, "calibration training input pins differ")
        if set(value["results"]) != {job["id"] for job in declared}:
            raise ValueError("six exact calibration results required")
    if (selection["evaluation_sha256"] != _file(folder / "evaluation.json")
            or evaluation["verification_sha256"] != _file(folder / "verification.json")
            or verification.get("neural_training_or_inference") is not False):
        raise ValueError("selection bindings differ")
    for job in declared:
        verified, scored = verification["results"][job["id"]], evaluation["results"][job["id"]]
        if (verified.get("job") != job or verified.get("exact_official_checkpoint_restores") is not True
                or set(verified.get("weights_sha256", {})) != {str(n) for n in STEPS["calibration"]}
                or scored.get("job") != job or scored.get("status") != "completed"
                or len(scored.get("development_curve", [])) != 1):
            raise ValueError("calibration verification or score producer differs")
        row = scored["development_curve"][0]
        if (type(row.get("updates")) is not int or row["updates"] != TOTALS["calibration"]
                or row.get("objective_id") != job["objective_id"]
                or row.get("weights_sha256") != verified["weights_sha256"][str(TOTALS["calibration"])]):
            raise ValueError("calibration score did not use the verified endpoint")
    for objective in OBJECTIVES:
        candidates = {str(j["rate"]): evaluation["results"][j["id"]]["development_curve"][0]["metrics"]
                      for j in jobs("calibration") if j["objective_id"] == objective}
        _same(selection["selected"][objective], choose_rate(candidates), "calibration choice differs")
    return selection


def run_stage(repository, directory, stage, *, max_seconds=14400):
    """One explicit, exclusive stage; no restore/retry path for an old stage."""
    if stage not in TOTALS:
        raise ValueError("calibration or main stage required")
    repository, directory = Path(repository), Path(directory)
    ledger = _Stage(directory, stage, max_seconds)
    try:
        protocol = ledger.perform("authenticate-protocol-data-proof", lambda: load_protocol(repository, directory))
        ledger.guard = lambda: _sources(protocol)
        pin = _file(directory / "protocol.json")
        import torch
        from brain_in_computer.dialogue_student import checkpoint_digest
        from experiments.execution_profile import assert_strict_profile, runtime_profile
        from experiments import foundation_variant_data as data_api
        from experiments.foundation_evaluation import FoundationBank
        from experiments.foundation_results import _score
        from experiments.foundation_variant_analysis import choose_rate
        from experiments.sequence_student import build_sequence_student
        runtime = runtime_profile("cuda:0")
        _same(runtime, protocol["execution_profile"], "formal runtime differs from objective proof")
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise ValueError("one-thread formal runtime required")
        selection = ledger.perform("authenticate-calibration-selection", lambda: _selection(directory, protocol)) if stage == "main" else None
        selection_pin = _file(directory / "calibration/selection.json") if selection is not None else None
        declared = jobs(stage, selection)
        ledger.receipt.update(protocol_sha256=pin, source_sha256=protocol["source_sha256"], execution_profile=runtime,
            jobs=declared, benchmark_scope=BENCHMARK_SCOPE, selection_sha256=selection_pin)
        ledger.flush()
        data = ledger.perform("load-authenticated-reused-data", lambda: data_api.load(repository / DATA_ROOT))
        index = ledger.perform("construct-one-canonical-index", lambda: data_api.build_index(repository / DATA_ROOT, stage, data=data),
            lambda value: dict(identity=value.identity, construction=value.construction))
        _same(index.identity, protocol["index_identity"][stage], "actual process index differs from pinned canonical verification")
        if len(data["stages"][stage]["plan"]["bundles"]) != TOTALS[stage] or index.identity["micro_batch_size"] != 32:
            raise ValueError("actual admitted plan differs from formal budget")
        ledger.receipt.update(index_identity=index.identity, index_construction=index.construction)
        initial = {}
        for job in declared:
            ledger.perform("train/"+job["id"], lambda: _train_job(stage, data, job, index, pin, runtime, initial, ledger),
                lambda value: {k: value[k] for k in ("physical_optimizer_updates", "completed_microbatch_episode_exposures", "wall_seconds", "cpu_seconds")})
            ledger.receipt["completed_jobs"].append(job["id"])
            ledger.receipt["initial_weights_sha256"] = dict(initial)
            ledger.flush()
            gc.collect(); torch.cuda.empty_cache()
        inputs, receipts = ledger.perform("seal-complete-training-inputs", lambda: _inputs(directory, stage, declared, pin))
        canonical = ledger.perform("read-canonical-index-prefix", lambda: index.replay("curriculum", TOTALS[stage], include_bundles=True))
        verified = {}
        for job in declared:
            verified[job["id"]] = ledger.perform("verify/"+job["id"],
                lambda: _verify_job(directory, stage, job, data, index, canonical, receipts[job["id"]], initial, pin, runtime, ledger),
                lambda value: dict(checkpoints=len(value["weights_sha256"]), restore_seconds=value["restore_seconds"],
                                   model_constructions_completed=value["model_constructions_completed"]))
        _same(_inputs(directory, stage, declared, pin)[0], inputs, "training inputs changed during verification")
        verification = dict(schema=SCHEMA, status="completed", stage=stage, protocol_sha256=pin,
            source_sha256=protocol["source_sha256"], execution_profile=runtime,
            input_file_sha256=inputs, results=verified, index_identity=index.identity,
            neural_training_or_inference=False, automatic_promotion=False)
        _write(ledger.folder / "verification.json", verification)
        banks = data["stages"][stage]["banks"]
        prepared = ledger.perform("prepare-evaluation-banks", lambda: {
            role: {name: FoundationBank(rows, role=role, config=_config()) for name, rows in named.items()}
            for role, named in banks.items()}, lambda value: dict(banks=sum(map(len, value.values()))))
        evaluated = {}
        for job in declared:
            model = ledger.perform("evaluation-model/"+job["id"], lambda: build_sequence_student(job["seed"], config=_config(), device="cuda:0"),
                                   lambda value: dict(model_constructions_completed=1, optimizer_updates=0))
            record = dict(job=job, status="running", development_curve=[])
            try:
                for step in (STEPS[stage] if stage == "main" else (TOTALS[stage],)):
                    def load_weights():
                        saved = _checkpoint(directory, stage, job, step, pin, runtime, receipts[job["id"]]["checkpoints"][f"checkpoint-{step:06d}.pt"])
                        model.load_state_dict(saved["learner"]["weights"], strict=True)
                        digest = checkpoint_digest(model)
                        if digest != verified[job["id"]]["weights_sha256"][str(step)]:
                            raise ValueError("evaluation weight producer differs from verified checkpoint")
                        return digest
                    digest = ledger.perform(f"score-load/{job['id']}/{step}", load_weights, lambda value: dict(weights_sha256=value))
                    def score(role, control="normal"):
                        assert_strict_profile()
                        value = _score(model, prepared[role], banks[role], _config(), role, control=control)
                        if checkpoint_digest(model) != digest:
                            raise RuntimeError("scoring changed learner weights")
                        return dict(updates=step, weights_sha256=digest, objective_id=job["objective_id"], **value)
                    record["development_curve"].append(ledger.perform(f"score/{job['id']}/{step}/dev", lambda: score("dev"), lambda value: value["cost"]))
                if stage == "main":
                    for role, control in (("train_fit", "normal"), ("audit", "normal"), ("audit", "blank"), ("audit", "reset")):
                        value = ledger.perform(f"score/{job['id']}/{step}/{role}/{control}", lambda: score(role, control), lambda value: value["cost"])
                        if control == "normal":
                            record[role] = value
                        else:
                            record.setdefault("controls", {})[control] = value
                record["status"] = "completed"
            except BaseException as error:
                record.update(status="failed", error=repr(error), failed_scoring_work_may_be_unknown=True)
                raise
            finally:
                _write(ledger.folder / (job["id"]+"-scores.json"), record)
            evaluated[job["id"]] = record
            del model
            gc.collect(); torch.cuda.empty_cache()
        _same(_inputs(directory, stage, declared, pin)[0], inputs, "training inputs changed during scoring")
        evaluation = dict(schema=SCHEMA, status="completed", stage=stage, protocol_sha256=pin,
            source_sha256=protocol["source_sha256"], execution_profile=runtime,
            verification_sha256=_file(ledger.folder / "verification.json"), input_file_sha256=inputs,
            results=evaluated, automatic_promotion=False, benchmark_scope=BENCHMARK_SCOPE)
        _write(ledger.folder / "evaluation.json", evaluation)
        if stage == "calibration":
            selected = {objective: choose_rate({str(j["rate"]): evaluated[j["id"]]["development_curve"][0]["metrics"]
                for j in declared if j["objective_id"] == objective}) for objective in OBJECTIVES}
            _write(ledger.folder / "selection.json", dict(schema=SCHEMA, status="completed", selected=selected,
                protocol_sha256=pin, evaluation_sha256=_file(ledger.folder / "evaluation.json"), automatic_promotion=False))
        else:
            screen = comparison_screen({o: {str(seed): evaluated[f"{o}-seed{seed}"]["audit"]["metrics"] for seed in SEEDS} for o in OBJECTIVES})
            _write(ledger.folder / "summary.json", dict(schema=SCHEMA, status="completed", protocol_sha256=pin,
                evaluation_sha256=_file(ledger.folder / "evaluation.json"), screen=screen,
                formal_updates=21492, formal_episode_exposures=2063232, automatic_promotion=False, scope=BENCHMARK_SCOPE))
        ledger.perform("final-protocol-data-proof-check", lambda: load_protocol(repository, directory))
        if selection_pin is not None and _file(directory / "calibration/selection.json") != selection_pin:
            raise ValueError("calibration selection changed during main")
        ledger.guard()
        ledger.receipt.update(status="completed", ended_utc=_utc(), physical_optimizer_updates=sum(r["physical_optimizer_updates"] for r in receipts.values()),
            completed_microbatch_episode_exposures=sum(r["completed_microbatch_episode_exposures"] for r in receipts.values()),
            artifacts_sha256={p.relative_to(directory).as_posix(): _file(p)
                for p in sorted(ledger.folder.rglob("*")) if p.is_file() and p.name != "stage.json"})
        ledger.flush()
        return ledger.receipt
    except BaseException as error:
        ledger.receipt.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            ended_utc=_utc(), error=repr(error))
        ledger.flush()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run-stage"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--expected-proof-sha256")
    parser.add_argument("--stage", choices=tuple(TOTALS))
    parser.add_argument("--max-seconds", type=float, default=14400)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        if args.proof is None or args.expected_proof_sha256 is None:
            parser.error("prepare requires --proof and --expected-proof-sha256")
        result = prepare(repository, args.output.resolve(), proof_path=args.proof.resolve(),
                         expected_proof_sha256=args.expected_proof_sha256)
    else:
        if args.stage is None:
            parser.error("run-stage requires --stage")
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        from experiments.execution_profile import configure_strict_profile
        configure_strict_profile()
        result = run_stage(repository, args.output.resolve(), args.stage, max_seconds=args.max_seconds)
    print(json.dumps({k: result[k] for k in ("schema", "status", "wall_seconds") if k in result}), flush=True)


if __name__ == "__main__":
    main()
