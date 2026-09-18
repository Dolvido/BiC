"""Prospective width comparison using the transactional raw-English backend."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_loop import run_lock
from brain_in_computer.learning_student import _check_finite_tree
from experiments import realization_study as previous
from experiments.composition_curriculum import FAMILIES
from experiments.composition_evaluation import PreparedBank, evaluate_banks, verify_cpu_restart
from experiments.realization_backend import RealizationBackend
from experiments.realization_training import RealizationStream, replay_evidence, stream_evidence
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.train_cognitive import atomic_checkpoint, atomic_json


SCHEMA = "bic-shared-capacity-study-v1"
WIDTHS = (96, 192, 256)
RATES = {"r0003": .0003, "r001": .001, "r003": .003}
SEEDS = {"calibration": 3100, "main": 3101}
SAMPLERS = {"calibration": 4100, "main": 4101}
TOTALS = {"calibration": 600, "main": 7200}
STEPS = {"calibration": (0, 600), "main": (0, 600, 1800, 3600, 7200)}
MICRO, BLOCK = 32, 600
file_hash = previous.file_hash
_same = previous._same
_canonical = previous._canonical


def config(width):
    if width not in WIDTHS:
        raise ValueError("undeclared width")
    return SequenceConfig(width=width, layers=4, heads=4, feedforward=width*4, max_turns=12)


def jobs(stage):
    if stage == "calibration":
        return {f"w{width}-{name}": {"width": width, "rate": rate}
                for width in WIDTHS for name, rate in RATES.items()}
    if stage == "main":
        return {f"w{width}": {"width": width} for width in WIDTHS}
    raise ValueError("unknown study stage")


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = [*previous.source_hashes(), "experiments/realization_backend.py",
             "experiments/execution_profile.py", "experiments/capacity_banks.py",
             "experiments/capacity_evidence.py", "experiments/capacity_study.py",
             "experiments/summarize_composition_study.py", "docs/CAPACITY_STUDY_PROTOCOL.md"]
    return {name: file_hash(root/name) for name in names}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def tensor_equal(left, right):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError("saved tensor state differs")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError("saved fields differ")
        for key in left:
            tensor_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        if not isinstance(right, (tuple, list)) or len(left) != len(right):
            raise ValueError("saved sequence differs")
        for first, second in zip(left, right):
            tensor_equal(first, second)
    elif left != right:
        raise ValueError("saved value differs")


def validate_profile_receipt(receipt, width, expected_sources):
    count = 96 if width == 96 else 48
    endpoint = count//3
    required = ("initial_state_equal", "load_at_midpoint_exact", "midpoint_equal",
                "resumed_endpoint_equal", "uninterrupted_repeat_equal",
                "all_three_lengths_consumed_per_family", "sources_unchanged", "preserved_inputs_unchanged")
    if (receipt.get("status") != "passed" or receipt.get("width") != width
            or receipt.get("config") != asdict(config(width))
            or receipt.get("physical_updates") != count or receipt.get("physical_episode_exposures") != count*MICRO*3
            or receipt.get("micro_batch_size") != MICRO
            or any(receipt.get(field) is not True for field in required)
            or receipt.get("source_sha256") != expected_sources
            or receipt.get("source_sha256_after") != expected_sources):
        raise ValueError("runtime receipt does not certify this configuration/source")
    buckets = receipt.get("bucket_microbatches", {})
    if set(buckets) != set(FAMILIES):
        raise ValueError("runtime probe family coverage differs")
    for counts in buckets.values():
        if (set(counts) != {"8", "10", "12"} or sum(counts.values()) != endpoint
                or any(type(value) is not int or value <= 0 for value in counts.values())):
            raise ValueError("runtime probe did not exercise every declared length")


def profile_proof(path):
    """Authenticate all declared width probes without doing neural work."""
    path = Path(path).resolve()
    coverage = read(path)
    if coverage.get("status") != "passed" or set(coverage.get("width_receipts", {})) != {str(x) for x in WIDTHS}:
        raise ValueError("every width needs completed strict-profile evidence")
    files = {str(path): file_hash(path)}
    root = Path(__file__).resolve().parents[1]
    expected_sources = {**previous.source_hashes(),
        **{name: file_hash(root/name) for name in ("experiments/realization_backend.py", "experiments/execution_profile.py")}}
    for width, entry in coverage["width_receipts"].items():
        receipt_path = (path.parent/entry["path"]).resolve()
        if not receipt_path.is_relative_to(path.parent) or file_hash(receipt_path) != entry["sha256"]:
            raise ValueError("runtime receipt identity differs")
        receipt = read(receipt_path)
        validate_profile_receipt(receipt, int(width), expected_sources)
        _same(receipt["execution_profile"], coverage["execution_profile"])
        files[str(receipt_path)] = file_hash(receipt_path)
        for name, sha in receipt["artifact_sha256"].items():
            artifact = (receipt_path.parent/name).resolve()
            if not artifact.is_relative_to(receipt_path.parent) or file_hash(artifact) != sha:
                raise ValueError("runtime probe artifact differs")
            files[str(artifact)] = sha
    if coverage.get("physical_updates") != 192:
        raise ValueError("runtime probe accounting differs")
    from experiments.execution_profile import source_hash
    if coverage["execution_profile"]["source_sha256"] != source_hash():
        raise ValueError("runtime implementation differs from verified profile")
    return {"execution_profile": coverage["execution_profile"], "files_sha256": files}


def contract():
    return {"schema": SCHEMA, "widths": list(WIDTHS),
            "configs": {str(width): asdict(config(width)) for width in WIDTHS},
            "jobs": {stage: jobs(stage) for stage in TOTALS}, "seeds": SEEDS,
            "sampler_seeds": SAMPLERS, "updates": TOTALS,
            "checkpoints": {stage: list(values) for stage, values in STEPS.items()},
            "micro_batch_size": MICRO, "microbatches_per_update": 3,
            "block_updates": BLOCK, "commit_interval": 16,
            "calibration_ranking": "Rounded12 tuple: minimum nine cell mean(action/reply pairs), mean cells, macro known accuracy, negative unsupported ASK. Equal four panels per cell. Tie .001,.0003,.003.",
            "automatic_promotion": False}


def structural_evidence(banks, updates, sampler_seed):
    """Independent bucket/pair draw reconstruction; no lesson generation."""
    if type(updates) is not int or not 0 <= updates <= max(TOTALS.values()):
        raise ValueError("update count outside declared study")
    result = {}
    for family in FAMILIES:
        generator = torch.Generator().manual_seed(sampler_seed + sorted(FAMILIES).index(family)*7919)
        buckets = sorted(banks[family])
        counts = dict.fromkeys(buckets, 0)
        occurrences = {turns: torch.zeros(len(banks[family][turns])//2, dtype=torch.long) for turns in buckets}
        for _ in range(updates):
            turns = buckets[int(torch.randint(len(buckets), (1,), generator=generator))]
            indices = torch.randint(len(occurrences[turns]), (MICRO//2,), generator=generator)
            occurrences[turns] += torch.bincount(indices, minlength=len(occurrences[turns]))
            counts[turns] += 1
        result[family] = {"sampler_sha256": hashlib.sha256(bytes(generator.get_state().tolist())).hexdigest(),
            "bucket_microbatches": counts, "occurrences": {key: value.tolist() for key, value in occurrences.items()},
            "episodes": updates*MICRO, "turns": sum(key*value*MICRO for key, value in counts.items())}
    return _canonical(result)


def prepare(directory, proof_path):
    from experiments.capacity_banks import prepare_banks, bank_manifest, protected_transcripts
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        if (directory/"protocol.json").exists():
            return load_protocol(directory)
        started = time.monotonic()
        proof = profile_proof(proof_path)
        previous.load_protocol("runs/realization-study-local")
        banks, diagnostics = prepare_banks(with_diagnostics=True)
        expected_counts = {"main": {"train": 4608, "development": 4608, "audit": 4608},
                           "calibration": {"train": 1152, "development": 2304}}
        for stage, roles in expected_counts.items():
            for role, count in roles.items():
                actual = (sum(len(rows) for buckets in banks[stage][role].values() for rows in buckets.values())
                          if role == "train" else sum(map(len, banks[stage][role].values())))
                if actual != count:
                    raise ValueError("default study bank count differs")
        protected = {"calibration": sorted(protected_transcripts(banks, stage="calibration"))}
        stream = RealizationStream(banks["calibration"]["train"], mode="fresh",
            protected_transcripts=protected["calibration"], sampler_seed=SAMPLERS["calibration"], micro_batch_size=MICRO)
        for _ in range(TOTALS["calibration"]):
            for family in FAMILIES:
                stream.draw(family)
        planned = stream.snapshot()
        protected["main"] = sorted(protected_transcripts(banks, stage="main",
            consumed_calibration=planned["seen_transcripts"]))
        for name, value in (("banks.pt", banks), ("protected.pt", protected), ("planned-calibration.pt", planned)):
            atomic_checkpoint(directory/name, value)
        initial = {stage: {str(width): checkpoint_digest(build_sequence_student(seed, config=config(width)))
                          for width in WIDTHS} for stage, seed in SEEDS.items()}
        protocol = _canonical({**contract(), "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_hashes(), "profile_proof": proof,
            "data_files_sha256": {name: file_hash(directory/name) for name in ("banks.pt", "protected.pt", "planned-calibration.pt")},
            "bank_manifest": bank_manifest(banks), "bank_diagnostics": diagnostics,
            "initial_weights_sha256": initial,
            "expected_final_structural_streams": {stage: structural_evidence(banks[stage]["train"], TOTALS[stage], SAMPLERS[stage]) for stage in TOTALS},
            "protected_counts": {stage: len(values) for stage, values in protected.items()},
            "comparison": "Width-only architecture configurations, same accepted fresh lessons and updates; learning rate selected under equal declared calibration budgets. Runtime/FLOPs/memory are measured, not matched.",
            "interpretation": "One main initialization per configuration. Width changes capacity and optimization geometry. Finite synthetic motifs, inspected historical semantics, no general-intelligence or automatic promotion claim."})
        root = Path(__file__).resolve().parents[1]
        for name in protocol["source_sha256"]:
            target = directory/"source"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root/name).read_bytes())
        atomic_json(directory/"protocol.json", protocol)
        atomic_json(directory/"preparation.json", {"schema": SCHEMA, "wall_seconds": time.monotonic()-started,
            "scope": "Profile/source authentication, canonical bank preparation, planned calibration replay, initial model digests and data/source/protocol writes. Python import/startup excluded.",
            "neural_training_or_evaluation_performed": False,
            "planned_calibration_updates": TOTALS["calibration"],
            "planned_calibration_seen_transcripts": len(planned["seen_transcripts"]),
            "protocol_sha256": file_hash(directory/"protocol.json")})
        print(json.dumps({"prepared": True, "source_files": len(protocol["source_sha256"])}), flush=True)
        return protocol


def load_protocol(directory):
    directory = Path(directory)
    value = read(directory/"protocol.json")
    if (any(value.get(key) != item for key, item in contract().items()) or value["source_sha256"] != source_hashes()
            or set(value["data_files_sha256"]) != {"banks.pt", "protected.pt", "planned-calibration.pt"}):
        raise ValueError("study contract or sources changed")
    for name, sha in value["source_sha256"].items():
        if file_hash(directory/"source"/name) != sha:
            raise ValueError("frozen source copy changed")
    for name, sha in value["data_files_sha256"].items():
        if file_hash(directory/name) != sha:
            raise ValueError("frozen curriculum changed")
    for name, sha in value["profile_proof"]["files_sha256"].items():
        if file_hash(name) != sha:
            raise ValueError("runtime proof changed")
    return value


def data(directory):
    return (torch.load(Path(directory)/"banks.pt", map_location="cpu", weights_only=True),
            torch.load(Path(directory)/"protected.pt", map_location="cpu", weights_only=True))


def required(directory, stage):
    return [Path(directory)/stage/job/name for job in jobs(stage)
            for name in ("job.json", "learner/backend.pt", "report.json", *(f"checkpoint-{step:06d}.pt" for step in STEPS[stage]))]


def completed_inputs(directory, stage):
    directory = Path(directory)
    paths = required(directory, stage)
    if any(not path.is_file() for path in paths):
        raise ValueError("all stage endpoints and curve checkpoints are required")
    return {path.relative_to(directory).as_posix(): file_hash(path) for path in paths}


def load_verification(directory, stage):
    directory = Path(directory)
    value = read(directory/"verification"/f"{stage}.json")
    if (value.get("schema") != SCHEMA or value.get("stage") != stage
            or value["protocol_sha256"] != file_hash(directory/"protocol.json")
            or value["input_files_sha256"] != completed_inputs(directory, stage)):
        raise ValueError("stage replay provenance differs")
    return value


def load_selection(directory):
    directory = Path(directory)
    load_verification(directory, "calibration")
    selected = read(directory/"selection.json")
    if (selected.get("schema") != SCHEMA or selected.get("automatic_promotion") is not False
            or selected["protocol_sha256"] != file_hash(directory/"protocol.json")
            or selected["calibration_verification_sha256"] != file_hash(directory/"verification/calibration.json")
            or selected["input_files_sha256"] != completed_inputs(directory, "calibration")):
        raise ValueError("calibration selection provenance differs")
    derived = selection_data(directory)
    _same({key: selected.get(key) for key in derived}, derived)
    return selected


def selection_data(directory):
    from experiments.capacity_evidence import select_rates
    inputs = {}
    for job, spec in jobs("calibration").items():
        saved = torch.load(Path(directory)/"calibration"/job/"checkpoint-000600.pt", map_location="cpu", weights_only=True)
        inputs[job] = {**spec, "metrics": saved["development"]}
    return select_rates(inputs)


def recipe(directory, stage, job):
    options = jobs(stage)
    if job not in options:
        raise ValueError("unknown job")
    value = dict(options[job])
    if stage == "main":
        value["rate"] = load_selection(directory)["selected"][str(value["width"])]["rate"]
    return value


def backend(banks, protected, stage, spec, device="cpu", payload=None):
    return RealizationBackend(banks[stage]["train"], mode="fresh", protected_transcripts=protected[stage],
        evaluation_banks={"development": banks[stage]["development"]}, seed=SEEDS[stage], sampler_seed=SAMPLERS[stage],
        device=device, micro_batch_size=MICRO, learning_rate=spec["rate"], config=config(spec["width"]), payload=payload)


def validate_backend(payload, stage, spec):
    learner, cursor = payload["learner"], payload["curriculum"]
    recipe = learner["recipe"]
    if (type(learner["updates"]) is not int or not 0 <= learner["updates"] <= TOTALS[stage]
            or recipe["seed"] != SEEDS[stage] or recipe["sampler_seed"] != SAMPLERS[stage]
            or recipe["learning_rate"] != spec["rate"] or recipe["config"] != asdict(config(spec["width"]))
            or recipe["micro_batch_size"] != MICRO or recipe["realization"]["mode"] != "fresh"
            or learner["family_microbatches"] != dict.fromkeys(FAMILIES, learner["updates"])
            or cursor["base_updates"] % BLOCK != 0 or len(cursor["schedule"]) > BLOCK
            or any(row != list(FAMILIES) for row in cursor["schedule"])):
        raise ValueError("durable backend does not belong to the declared job")


def validate_snapshot(saved, protocol, stage, job, banks, spec):
    if (saved.get("schema") != SCHEMA or saved.get("stage") != stage or saved.get("job") != job
            or saved.get("protocol_sha256") != protocol["_sha256"] or saved.get("recipe") != spec
            or saved.get("execution_profile") != protocol["profile_proof"]["execution_profile"]):
        raise ValueError("checkpoint identity differs")
    payload = saved["backend"]
    validate_backend(payload, stage, spec)
    learner, cursor = payload["learner"], payload["curriculum"]
    updates = learner["updates"]
    if type(updates) is not int or updates not in STEPS[stage]:
        raise ValueError("unplanned checkpoint")
    if (learner["recipe"]["seed"] != SEEDS[stage] or learner["recipe"]["sampler_seed"] != SAMPLERS[stage]
            or learner["recipe"]["learning_rate"] != spec["rate"]
            or learner["recipe"]["config"] != asdict(config(spec["width"]))
            or learner["family_microbatches"] != dict.fromkeys(FAMILIES, updates)
            or any(row != list(FAMILIES) for row in cursor["schedule"])):
        raise ValueError("learner recipe or joint schedule differs")
    expected = structural_evidence(banks[stage]["train"], updates, SAMPLERS[stage])
    for family, item in expected.items():
        if hashlib.sha256(bytes(learner["samplers"][family].tolist())).hexdigest() != item["sampler_sha256"]:
            raise ValueError("sampler differs from independent reconstruction")
        _same(learner["bucket_microbatches"][family], item["bucket_microbatches"])
        _same(learner["realization"]["occurrences"][family], item["occurrences"])
        if (learner["exposures"][family]["episodes"] != item["episodes"]
                or learner["exposures"][family]["turns"] != item["turns"]):
            raise ValueError("episode/turn exposures differ")
    _check_finite_tree(saved, "capacity checkpoint")


def train(directory, stage, job):
    from experiments.capacity_evidence import aggregate_rows
    from experiments.execution_profile import assert_strict_profile, runtime_profile
    invocation_started = time.monotonic()
    directory = Path(directory)
    protocol = load_protocol(directory)
    protocol["_sha256"] = file_hash(directory/"protocol.json")
    spec = recipe(directory, stage, job)
    runtime = runtime_profile("cuda:0")
    _same(runtime, protocol["profile_proof"]["execution_profile"])
    torch.cuda.reset_peak_memory_stats()
    banks, protected = data(directory)
    folder = directory/stage/job
    (folder/"worker").mkdir(parents=True, exist_ok=True)
    with run_lock(folder/"worker"):
        path = folder/"learner/backend.pt"
        marker = {"schema": SCHEMA, "stage": stage, "job": job, "recipe": spec,
            "protocol_sha256": protocol["_sha256"], "execution_profile": runtime,
            "selection_sha256": file_hash(directory/"selection.json") if stage == "main" else None}
        if (folder/"job.json").exists():
            _same(read(folder/"job.json"), marker)
        elif path.exists():
            raise ValueError("durable learner has no authenticated job marker")
        else:
            atomic_json(folder/"job.json", marker)
        if path.exists():
            learner = RealizationBackend.load(path, banks[stage]["train"], protected_transcripts=protected[stage],
                evaluation_banks={"development": banks[stage]["development"]}, device="cuda")
        else:
            learner = backend(banks, protected, stage, spec, device="cuda")
            learner.save(path)
        validate_backend(learner.snapshot(), stage, spec)
        initial_updates = learner.updates
        for step in STEPS[stage]:
            if step < learner.updates and not (folder/f"checkpoint-{step:06d}.pt").exists():
                raise ValueError("cannot continue beyond a missing curve checkpoint")
        invocation_folder = folder/"invocations"
        invocation_folder.mkdir(exist_ok=True)
        invocation_path = invocation_folder/f"{len(list(invocation_folder.glob('*.json')))+1:04d}.json"
        invocation = {"schema": SCHEMA, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
            "starting_updates": initial_updates, "chunks": [], "execution_profile": runtime,
            "setup_seconds": time.monotonic()-invocation_started,
            "setup_peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
            "setup_peak_cuda_reserved_mib": torch.cuda.max_memory_reserved()/2**20}
        atomic_json(invocation_path, invocation)
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()

        def save_point():
            checkpoint = folder/f"checkpoint-{learner.updates:06d}.pt"
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                tensor_equal(saved["backend"], learner.snapshot())
                validate_snapshot(saved, protocol, stage, job, banks, spec)
                return
            evidence = learner.evaluate("development")
            saved = {"schema": SCHEMA, "stage": stage, "job": job, "recipe": spec,
                "protocol_sha256": protocol["_sha256"], "execution_profile": runtime,
                "backend": learner.snapshot(), "development": aggregate_rows(evidence["per_bank"]),
                "progress": evidence["progress"], "development_seconds": evidence["wall_seconds"],
                "weights_sha256": checkpoint_digest(learner.model)}
            validate_snapshot(saved, protocol, stage, job, banks, spec)
            atomic_checkpoint(checkpoint, saved)
            print(json.dumps({"stage": stage, "job": job, "checkpoint": learner.updates}), flush=True)

        try:
            if learner.updates in STEPS[stage]:
                save_point()
            while learner.updates < TOTALS[stage]:
                assert_strict_profile()
                load_protocol(directory)
                next_boundary = min(TOTALS[stage], ((learner.updates//BLOCK)+1)*BLOCK)
                pending = learner.curriculum["cursor"] < len(learner.curriculum["schedule"])
                schedule = None if pending else [tuple(FAMILIES)]*(next_boundary-learner.updates)
                result = learner.train_chunk(schedule, max_updates=next_boundary-learner.updates, commit_interval=16)
                invocation["chunks"].append(result)
                atomic_json(invocation_path, invocation)
                if learner.updates in STEPS[stage]:
                    save_point()
                else:
                    print(json.dumps({"stage": stage, "job": job, "updates": learner.updates}), flush=True)
            invocation["status"] = "completed"
        except BaseException:
            invocation["status"] = "failed"
            invocation["failure_report"] = learner.last_report
            raise
        finally:
            invocation.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                ending_updates=learner.updates, wall_seconds=time.monotonic()-invocation_started,
                post_setup_wall_seconds=time.monotonic()-started,
                wall_scope="From train function entry, including source/data checks and backend construction/load, excluding Python import and prior CLI profile configuration.",
                peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            atomic_json(invocation_path, invocation)
        final = learner.snapshot()
        endpoint = torch.load(folder/f"checkpoint-{TOTALS[stage]:06d}.pt", map_location="cpu", weights_only=True)
        tensor_equal(endpoint["backend"], final)
        report = {"schema": SCHEMA, "stage": stage, "job": job, "recipe": spec,
            "protocol_sha256": protocol["_sha256"], "execution_profile": runtime,
            "updates": learner.updates, "exposures": final["learner"]["exposures"],
            "stream": stream_evidence(final["learner"]["realization"]),
            "retained_step_seconds": final["accounting"]["retained_step_seconds"],
            "backend_file_sha256": file_hash(path), "endpoint_file_sha256": file_hash(folder/f"checkpoint-{TOTALS[stage]:06d}.pt"),
            "weights_sha256": checkpoint_digest(learner.model), "heldout_evaluation_performed": False,
            "job_marker_sha256": file_hash(folder/"job.json"),
            "invocation_files_sha256": {value.name: file_hash(value) for value in invocation_folder.glob("*.json")},
            "physical_work_note": "Invocation receipts preserve completed calls and known failed work. A hard stop may leave a running receipt and unknown uncommitted work; retained counters are not a total physical-work guarantee.",
            "automatic_promotion": False}
        load_protocol(directory)
        atomic_json(folder/"report.json", report)


def verify(directory, stage):
    from experiments.capacity_evidence import validate_metrics
    from experiments.realization_banks import _stats
    directory = Path(directory)
    protocol = load_protocol(directory)
    protocol["_sha256"] = file_hash(directory/"protocol.json")
    files = completed_inputs(directory, stage)
    banks, protected = data(directory)
    expected = replay_evidence(banks[stage]["train"], "fresh", protected[stage],
        STEPS[stage], sampler_seed=SAMPLERS[stage], micro_batch_size=MICRO)
    if stage == "calibration":
        planned = torch.load(directory/"planned-calibration.pt", map_location="cpu", weights_only=True)
        _same(expected[TOTALS[stage]], stream_evidence(planned))
    results = {}
    for job in jobs(stage):
        spec = recipe(directory, stage, job)
        folder = directory/stage/job
        final = torch.load(folder/"learner/backend.pt", map_location="cpu", weights_only=True)
        validate_backend(final, stage, spec)
        learner = backend(banks, protected, stage, spec, payload=final)
        report = read(folder/"report.json")
        if (report.get("schema") != SCHEMA or report.get("stage") != stage or report.get("job") != job
                or report["recipe"] != spec or report["protocol_sha256"] != protocol["_sha256"]
                or report["updates"] != TOTALS[stage] or report["heldout_evaluation_performed"] is not False
                or report["backend_file_sha256"] != files[f"{stage}/{job}/learner/backend.pt"]
                or report["job_marker_sha256"] != files[f"{stage}/{job}/job.json"]
                or report["retained_step_seconds"] != final["accounting"]["retained_step_seconds"]):
            raise ValueError("endpoint report differs")
        _same(read(folder/"job.json"), {"schema": SCHEMA, "stage": stage, "job": job, "recipe": spec,
            "protocol_sha256": protocol["_sha256"], "execution_profile": protocol["profile_proof"]["execution_profile"],
            "selection_sha256": file_hash(directory/"selection.json") if stage == "main" else None})
        _same(report["execution_profile"], protocol["profile_proof"]["execution_profile"])
        _same(report["stream"], expected[TOTALS[stage]])
        _same(report["exposures"], final["learner"]["exposures"])
        for name, sha in report["invocation_files_sha256"].items():
            if file_hash(folder/"invocations"/name) != sha:
                raise ValueError("invocation ledger differs")
        weights = {}
        manifests = {name: _stats(rows) for name, rows in banks[stage]["development"].items()}
        for step in STEPS[stage]:
            saved = torch.load(folder/f"checkpoint-{step:06d}.pt", map_location="cpu", weights_only=True)
            validate_snapshot(saved, protocol, stage, job, banks, spec)
            learner.restore(saved["backend"])
            _same(stream_evidence(saved["backend"]["learner"]["realization"]), expected[step])
            digest = checkpoint_digest(learner.model)
            if digest != saved["weights_sha256"]:
                raise ValueError("checkpoint tensor digest differs")
            if step == 0 and digest != protocol["initial_weights_sha256"][stage][str(spec["width"])]:
                raise ValueError("prescribed initialization differs")
            validate_metrics(saved["development"], banks[stage]["development"], manifests, config(spec["width"]), role="dev")
            weights[str(step)] = digest
            if step == TOTALS[stage]:
                tensor_equal(final, saved["backend"])
                if digest != report["weights_sha256"] or report["endpoint_file_sha256"] != file_hash(folder/f"checkpoint-{step:06d}.pt"):
                    raise ValueError("final endpoint identity differs")
        results[job] = {"weights_sha256": weights, "exact_final_backend_binding": True}
        del learner
    load_protocol(directory)
    if completed_inputs(directory, stage) != files:
        raise RuntimeError("inputs changed during verification")
    output = directory/"verification"
    output.mkdir(exist_ok=True)
    result = {"schema": SCHEMA, "stage": stage, "protocol_sha256": protocol["_sha256"],
        "input_files_sha256": files, "replayed_streams": expected, "jobs": results,
        "neural_training_or_audit_performed": False}
    atomic_json(output/f"{stage}.json", result)
    print(json.dumps({"verified": stage, "jobs": len(results)}), flush=True)
    return result


def select(directory):
    directory = Path(directory)
    load_protocol(directory)
    verification = load_verification(directory, "calibration")
    result = {"schema": SCHEMA, "protocol_sha256": file_hash(directory/"protocol.json"),
        "calibration_verification_sha256": file_hash(directory/"verification/calibration.json"),
        "input_files_sha256": verification["input_files_sha256"], **selection_data(directory), "automatic_promotion": False}
    atomic_json(directory/"selection.json", result)
    print(json.dumps({"selected": result["selected"]}), flush=True)
    return result


def audit(directory):
    from experiments.execution_profile import runtime_profile
    from experiments.capacity_evidence import validate_audit_result
    directory = Path(directory)
    protocol = load_protocol(directory)
    selection = load_selection(directory)
    verification = load_verification(directory, "main")
    files = completed_inputs(directory, "main")
    _same(runtime_profile("cuda:0"), protocol["profile_proof"]["execution_profile"])
    banks, protected = data(directory)
    marker = {"schema": SCHEMA, "protocol_sha256": file_hash(directory/"protocol.json"),
        "selection_sha256": file_hash(directory/"selection.json"), "input_files_sha256": files,
        "verification_sha256": file_hash(directory/"verification/main.json")}
    output = directory/"audit"
    output.mkdir(exist_ok=True)
    with run_lock(output):
        if (output/"evaluation-started.json").exists() and read(output/"evaluation-started.json") != marker:
            raise ValueError("sealed audit inputs differ")
        atomic_json(output/"evaluation-started.json", marker)
        results = {}
        for job, values in jobs("main").items():
            path = output/f"{job}.json"
            started = time.monotonic()
            width = values["width"]
            payload = torch.load(directory/"main"/job/"learner/backend.pt", map_location="cpu", weights_only=True)
            learner = backend(banks, protected, "main", recipe(directory, "main", job), payload=payload)
            latest, metadata = learner._trainer.latest_observed_banks()
            initial_rows = {f"initial/{family}/t{turns}": rows
                for family, buckets in banks["main"]["train"].items() for turns, rows in buckets.items()}
            latest_rows = {f"latest/{family}/t{turns}": rows
                for family, buckets in latest.items() for turns, rows in buckets.items()}
            validation = {"job": job, "inputs": marker, "steps": STEPS["main"], "config": config(width),
                "expected_weights": verification["jobs"][job]["weights_sha256"],
                "audit_rows": banks["main"]["audit"], "initial_rows": initial_rows,
                "latest_rows": latest_rows, "latest_metadata": metadata}
            if path.exists():
                value = read(path)
                validate_audit_result(value, **validation)
                results[job] = value
                del learner
                continue
            model = build_sequence_student(SEEDS["main"], device="cuda", config=config(width))
            prepared = {name: PreparedBank(rows, role="audit", config=config(width)) for name, rows in banks["main"]["audit"].items()}
            curve = []
            for step in STEPS["main"]:
                saved = torch.load(directory/"main"/job/f"checkpoint-{step:06d}.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(saved["backend"]["learner"]["weights"], strict=True)
                digest = checkpoint_digest(model)
                if digest != verification["jobs"][job]["weights_sha256"][str(step)]:
                    raise ValueError("verified model identity differs")
                curve.append({"updates": step, "episodes_per_family": step*MICRO, "weights_sha256": digest,
                              "metrics": evaluate_banks(model, prepared)})
            initial_prepared = {name: PreparedBank(rows, role="train_fit", config=config(width))
                for name, rows in initial_rows.items()}
            latest_prepared = {name: PreparedBank(rows, role="train_fit", config=config(width))
                for name, rows in latest_rows.items()}
            value = {"schema": SCHEMA, "job": job, "inputs": marker, "curve": curve, "final": curve[-1]["metrics"],
                "initial_fit": evaluate_banks(model, initial_prepared),
                "latest_observed_fit": evaluate_banks(model, latest_prepared), "latest_observed_metadata": metadata,
                "controls": {control: evaluate_banks(model, prepared, control=control) for control in ("blank", "reset")},
                "cpu_restart": verify_cpu_restart(model, next(iter(banks["main"]["audit"].values()))[0]),
                "seconds": time.monotonic()-started, "automatic_promotion": False}
            validate_audit_result(value, **validation)
            atomic_json(path, value)
            results[job] = value
            del learner, model
            print(json.dumps({"audited": job}), flush=True)
        load_protocol(directory)
        load_verification(directory, "main")
        if completed_inputs(directory, "main") != files or load_selection(directory) != selection:
            raise RuntimeError("study evidence changed during audit")
        atomic_json(output/"report.json", {"schema": SCHEMA, "inputs": marker, "protocol": protocol,
            "results": results, "base_checkpoints_unchanged": True, "automatic_promotion": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", required=True, choices=("prepare", "train", "verify", "select", "audit"))
    parser.add_argument("--stage", choices=("calibration", "main"), default="main")
    parser.add_argument("--job")
    parser.add_argument("--profile-proof", default="runs/capacity-validation-local/coverage.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase in ("train", "audit"):
        from experiments.execution_profile import configure_strict_profile
        configure_strict_profile()
    if args.phase == "prepare":
        prepare(args.output, args.profile_proof)
    elif args.phase == "train":
        train(args.output, args.stage, args.job)
    elif args.phase == "verify":
        verify(args.output, args.stage)
    elif args.phase == "select":
        select(args.output)
    else:
        audit(args.output)


if __name__ == "__main__":
    main()
