"""Compact original-layout data for one bounded three-cycle local tutor pilot."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-foundation-tutor-loop-data-v1"
CYCLES, UPDATES, PAIRS = 3, 216, 4
PLAN_SEEDS = (852100001, 852100002, 852100003)
ORDER_SEEDS = (852101001, 852101002, 852101003)
DEV_SEEDS = (852102001, 852102002, 852102003)
AUDIT_SEED = 852103001
MAX_SECONDS = 1800
HISTORY_ROOT = "runs/foundation-layout-preparation-local/attempt-001"
HISTORY_PINS = {
    HISTORY_ROOT+"/receipt.json": "465404d4c9295fddfdb33b546e5c0c3570d41346ceda176ae3690c03abd53e60",
    HISTORY_ROOT+"/data/preparation.json": "6cc3b82885dfd826e060b9e620f40e69409a87fb49a819641bb97c41b15be1a6",
    HISTORY_ROOT+"/data/history.pt": "03f9d41578ce1a5a2854eb7fc0bff0af7dab9474eea97773390c4702981ff4a8",
}
HISTORY_COUNT = 1635840
HISTORY_UNION_SHA256 = "d849b952a012aead59160fc7f677eb58bd5778c44e63d130a4b085024a15c33c"
_ACTIVE = None


def native(path):
    text = str(Path(path).absolute())
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        text = "\\\\?\\UNC\\"+text[2:] if text.startswith("\\\\") else "\\\\?\\"+text
    return Path(text)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8")


def digest(path):
    result = hashlib.sha256()
    with native(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(native(path).read_bytes())


def write(path, value):
    target = native(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        if target.suffix == ".pt":
            import torch
            torch.save(value, handle)
        else:
            handle.write(encoded(value))
        handle.flush(); os.fsync(handle.fileno())
    return digest(target)


def _modules():
    from experiments import foundation_curriculum as foundation, foundation_plan as planning
    from experiments import foundation_layout_curriculum as layout, foundation_layout_training as training
    return foundation, planning, layout, training


def source_hashes():
    _, _, _, training = _modules()
    names = set(training.source_hashes()) | {"experiments/foundation_tutor_loop_data.py",
        "experiments/foundation_admission.py", "experiments/foundation_evidence.py", "experiments/foundation_banks.py",
        "experiments/composition_training.py", "experiments/train_cognitive.py"}
    if (ROOT/"experiments/__init__.py").exists():
        raise ValueError("experiment namespace initializer must remain absent")
    return {name: digest(ROOT/name) for name in sorted(names)}


def contract():
    return dict(schema=SCHEMA, cycles=CYCLES, updates_per_cycle=UPDATES, plan_seeds=list(PLAN_SEEDS),
        ordering_seeds=list(ORDER_SEEDS), stage_updates=24, final_updates=72, micro_batch_size=32,
        rehearsal_every=4, layout="original", pairs_per_cell=PAIRS,
        development_seeds=list(DEV_SEEDS), audit_seed=AUDIT_SEED,
        banks=dict(retention=dict(role="dev", source_cycle=0, episodes=720),
            **{"acquisition-1": dict(role="dev", source_cycle=1, episodes=720),
               "acquisition-2": dict(role="dev", source_cycle=2, episodes=720),
               "audit": dict(role="audit", source_cycle=0, episodes=792)}),
        cycle_zero_acquisition="retention", training_episodes=62208, maximum_preparation_seconds=MAX_SECONDS,
        history_file_sha256=HISTORY_PINS, history_union_count=HISTORY_COUNT, history_union_sha256=HISTORY_UNION_SHA256,
        history_scope="Exact union authenticated from the failed layout preparation's completed history phase; no broader historical exclusion claim.",
        original_layout_seed="sha256(canonical JSON [schema,plan seed,canonical bundle ID,pair index]) modulo2**63",
        runtime_cursor="Only the trainer envelope/evidence bundle_id becomes the lifetime cursor; canonical rows and seeds never depend on chosen order.",
        no_full_training_rows_serialized=True, no_trusted_index_serialized=True,
        automatic_retry=False, automatic_promotion=False)


@dataclass
class WorkLedger:
    counts: dict = field(default_factory=lambda: dict.fromkeys(("canonical_generate_attempts",
        "canonical_generate_completions", "canonical_rows_returned", "canonical_bundle_attempts",
        "canonical_bundle_completions", "adapted_training_bundles", "bundle_evidence_calls",
        "adapted_evaluation_pairs", "safe_archive_loads"), 0))
    layout: object = None
    deadline: float | None = None
    progress: object = None
    observe_bundle: object = None
    last_bundle_evidence: dict | None = None
    guard_attempts: dict = field(default_factory=dict)

    def boundary(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("fixed tutor pilot preparation allowance expired")

    def event(self, value):
        if self.progress is not None:
            self.progress(value)

    def report(self):
        return dict(counts=deepcopy(self.counts), layout=self.layout.report() if self.layout is not None else {},
            guard_attempts=deepcopy(self.guard_attempts),
            scope="All generate calls and copied/reconstructed rows are counted; they are not unique training lessons.")


def _work(work):
    if work is None:
        work = WorkLedger()
    if type(work) is not WorkLedger:
        raise ValueError("tutor-loop WorkLedger required")
    if work.layout is None:
        work.layout = _modules()[2].WorkLedger()
    return work


@contextmanager
def _tracking(work):
    global _ACTIVE
    if _ACTIVE is work:
        yield
        return
    if _ACTIVE is not None:
        raise RuntimeError("one synchronous data accounting owner required")
    foundation, planning, _, _ = _modules()
    original_generate, original_bundle = foundation.generate_pair, planning._materialize_validated_bundle
    def generated(*args, **kwargs):
        work.boundary(); work.counts["canonical_generate_attempts"] += 1
        result = original_generate(*args, **kwargs)
        work.counts["canonical_generate_completions"] += 1
        work.counts["canonical_rows_returned"] += len(result)
        work.boundary()
        return result
    def bundled(plan, bundle_id):
        work.boundary(); work.counts["canonical_bundle_attempts"] += 1
        result = original_bundle(plan, bundle_id)
        work.counts["canonical_bundle_completions"] += 1
        if work.observe_bundle is not None:
            work.observe_bundle(plan, bundle_id, result)
        if work.counts["canonical_bundle_completions"] % 48 == 0:
            work.event(dict(event="canonical_bundle_progress", bundle_id=bundle_id, work=work.report()))
        work.boundary()
        return result
    _ACTIVE = work
    foundation.generate_pair, planning._materialize_validated_bundle = generated, bundled
    try:
        yield
    finally:
        foundation.generate_pair, planning._materialize_validated_bundle = original_generate, original_bundle
        _ACTIVE = None


@contextmanager
def _cpu_guard(work):
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("fresh CPU-only preparation process required")
    def forbidden(name):
        work.guard_attempts[name] = 0
        def fail(*args, **kwargs):
            work.guard_attempts[name] += 1
            raise AssertionError("forbidden tutor data preparation operation: "+name)
        return fail
    with ExitStack() as stack:
        for obj, key, name in ((torch.nn.Module, "__init__", "module_construction"),
            (torch.nn.Module, "_call_impl", "module_forward"), (torch.Tensor, "backward", "tensor_backward"),
            (torch.autograd, "backward", "autograd_backward"), (torch.optim.Optimizer, "__init__", "optimizer_construction"),
            (torch.optim.AdamW, "step", "optimizer_step"), (torch.cuda, "_lazy_init", "cuda_initialization")):
            stack.enter_context(patch.object(obj, key, side_effect=forbidden(name)))
        yield
    if torch.cuda.is_initialized() or any(work.guard_attempts.values()):
        raise AssertionError("CPU-only data guard breached")


def _load_pt(path, work=None):
    import torch
    if work is not None:
        work.boundary(); work.counts["safe_archive_loads"] += 1
    with native(path).open("rb") as handle:
        return torch.load(handle, weights_only=True, map_location="cpu")


def _layout_seed(plan, bundle_id, index):
    payload = [SCHEMA, plan["config"]["seed"], bundle_id, index]
    return int(hashlib.sha256(encoded(payload).rstrip(b"\n")).hexdigest(), 16) % 2**63


def _adapt(plan, bundle_id, families, cursor, work):
    foundation, _, layout, training = _modules()
    adapted = {}
    for family in foundation.FAMILIES:
        adapted[family] = []
        for offset in range(0, len(families[family]), 2):
            work.boundary()
            adapted[family].extend(layout.materialize_pair(families[family][offset:offset+2], layout="original",
                seed=_layout_seed(plan, bundle_id, offset//2), work=work.layout))
    bundle = dict(schema=training.BUNDLE_SCHEMA, bundle_id=cursor, layout="original", families=adapted)
    work.counts["adapted_training_bundles"] += 1
    work.counts["bundle_evidence_calls"] += 1
    work.last_bundle_evidence = training._bundle_evidence(bundle, cursor=cursor, layout="original", micro_batch_size=32, work=work.layout)
    return bundle


def materialize_bundle(plan, bundle_id, *, global_cursor, work=None):
    work = _work(work)
    _, planning, _, _ = _modules()
    planning.validate_plan(plan)
    if (plan["config"]["seed"] not in PLAN_SEEDS or len(plan["bundles"]) != UPDATES
            or "admission" not in plan or type(global_cursor) is not int or not 0 <= global_cursor < CYCLES*UPDATES):
        raise ValueError("admitted pilot plan and bounded lifetime cursor required")
    with _tracking(work):
        families = planning._materialize_validated_bundle(plan, bundle_id)
        return _adapt(plan, bundle_id, families, global_cursor, work)


def expected_bundle(cycle_record, bundle_id, global_cursor):
    if type(global_cursor) is not int or not 0 <= global_cursor < CYCLES*UPDATES:
        raise ValueError("bounded lifetime cursor required")
    result = deepcopy(cycle_record["expected_bundles"][str(bundle_id)])
    if result["bundle_id"] != bundle_id:
        raise ValueError("canonical bundle evidence identity differs")
    result["bundle_id"] = global_cursor
    return result


def _history(work):
    from experiments import foundation_evidence as evidence
    for name, expected in HISTORY_PINS.items():
        work.boundary()
        if digest(ROOT/name) != expected:
            raise ValueError("inherited history artifact changed: "+name)
    outer = read(ROOT/HISTORY_ROOT/"receipt.json")
    inner = read(ROOT/HISTORY_ROOT/"data/preparation.json")
    if (outer.get("status") != "failed" or any(outer["guard_attempts"].values())
            or inner["work"]["phases"]["history"]["status"] != "completed"
            or "history.pt" not in inner["partial_artifacts"]):
        raise ValueError("completed authenticated history phase required")
    values = _load_pt(ROOT/HISTORY_ROOT/"data/history.pt", work)
    protected = evidence.transcript_set(values)
    if values != sorted(protected) or len(values) != HISTORY_COUNT or evidence.json_digest(values) != HISTORY_UNION_SHA256:
        raise ValueError("history union content differs")
    return protected, dict(file_sha256=deepcopy(HISTORY_PINS), count=len(values), transcript_sha256=HISTORY_UNION_SHA256,
        prior_failed_preparation_cost=dict(wall_seconds=outer["wall_seconds"], cpu_seconds=outer["cpu_seconds"]))


def _bank(plan, manifest, transcripts, name, role, seed, blocked, work):
    from experiments import foundation_banks as banks, foundation_evidence as evidence
    _, _, layout, _ = _modules()
    canonical, admission = banks.build_evaluation(plan, manifest, training_transcripts=transcripts,
        role=role, seed=seed, pairs_per_cell=PAIRS, excluded_transcripts=sorted(blocked))
    adapted, hashes = {}, set()
    for cell, rows in sorted(canonical.items()):
        adapted[cell] = []
        for offset in range(0, len(rows), 2):
            work.boundary()
            pair = layout.materialize_pair(rows[offset:offset+2], layout="original", seed=0, work=work.layout)
            layout.validate_pair(pair, work=work.layout)
            current = {banks.transcript_digest(row) for row in pair}
            if len(current) != 2 or current & blocked or current & hashes:
                raise ValueError("original-view bank transcript admission differs")
            hashes.update(current); adapted[cell].extend(pair)
            work.counts["adapted_evaluation_pairs"] += 1
    if hashes != set(admission["transcript_sha256"]):
        raise ValueError("original layout changed canonical evaluation transcripts")
    bank = dict(role=role, rows=adapted)
    report = dict(name=name, role=role, canonical_admission=admission, episodes=sum(map(len, adapted.values())),
        adapted_sha256=evidence.json_digest(bank), transcript_sha256=evidence.json_digest(sorted(hashes)))
    if report["episodes"] != contract()["banks"][name]["episodes"]:
        raise ValueError("fixed pilot bank size differs")
    return bank, canonical, report, hashes


def prepare(directory, *, max_seconds=MAX_SECONDS, progress=None):
    """One exclusive guarded preparation; a failure is terminal, never retried."""
    if max_seconds != MAX_SECONDS or progress is not None and not callable(progress):
        raise ValueError("fixed1800-second budget and optional callback required")
    started, cpu = time.monotonic(), time.process_time()
    output = native(directory); output.mkdir(parents=True, exist_ok=False)
    work = WorkLedger(deadline=started+max_seconds)
    artifacts, sources, cycles = {}, {}, []
    report = dict(schema=SCHEMA, status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        max_seconds=max_seconds, neural_forwards=0, optimizer_updates=0, teacher_calls=0, automatic_retry=False)
    journal = (output/"events.jsonl").open("xb")
    def event(value):
        journal.write(encoded(dict(elapsed_seconds=time.monotonic()-started, **value)))
        journal.flush()
        if progress is not None:
            progress(value)
    work.progress = event
    def save(name, value):
        work.boundary(); artifacts[name] = write(output/name, value)
    try:
        with _cpu_guard(work):
            from experiments import foundation_admission as admission, foundation_evidence as evidence
            _, planning, _, _ = _modules()
            work = _work(work)
            sources = source_hashes()
            save("invocation.json", dict(contract=contract(), source_sha256=sources))
            history, historical = _history(work)
            save("history-receipt.json", historical)
            report["prior_failed_preparation_cost"] = historical["prior_failed_preparation_cost"]
            blocked, trained, reserved = set(history), set(), set()
            with _tracking(work):
                for cycle in range(CYCLES):
                    event(dict(event="cycle_preparation_start", cycle=cycle))
                    t0, c0 = time.monotonic(), time.process_time()
                    plan, admission_report = admission.repair_plan(planning.build_plan(seed=PLAN_SEEDS[cycle],
                        stage_updates=24, final_updates=72, micro_batch_size=32, rehearsal_every=4,
                        ordering_seed=ORDER_SEEDS[cycle]), sorted(blocked))
                    expected = {}
                    def capture(current_plan, bundle_id, families):
                        if str(bundle_id) in expected:
                            raise ValueError("canonical evidence requested the same bundle twice")
                        _adapt(current_plan, bundle_id, families, bundle_id, work)
                        expected[str(bundle_id)] = deepcopy(work.last_bundle_evidence)
                    work.observe_bundle = capture
                    try:
                        manifest, transcripts = evidence.prepare_training(plan, protected_transcripts=sorted(blocked), anchor_limit=PAIRS)
                    finally:
                        work.observe_bundle = None
                    if set(expected) != set(map(str, range(UPDATES))) or set(transcripts) & blocked:
                        raise ValueError("complete unique admitted cycle required")
                    trained.update(transcripts); blocked.update(transcripts)
                    record = dict(cycle_id=cycle, plan=plan, expected_bundles=expected, training_manifest=manifest)
                    cycles.append(record)
                    save(f"cycle-{cycle}.json", record)
                    save(f"cycle-{cycle}-admission.json", admission_report)
                    bank_name = "retention" if cycle == 0 else f"acquisition-{cycle}"
                    requests = [(bank_name, "dev", DEV_SEEDS[cycle])]
                    if cycle == 0:
                        requests.append(("audit", "audit", AUDIT_SEED))
                    for name, role, seed in requests:
                        bank, canonical, bank_report, hashes = _bank(plan, manifest, transcripts, name, role, seed, blocked, work)
                        blocked.update(hashes); reserved.update(hashes)
                        save(name+".pt", bank); save(name+"-canonical.pt", canonical); save(name+"-admission.json", bank_report)
                    event(dict(event="cycle_preparation_complete", cycle=cycle,
                        wall_seconds=time.monotonic()-t0, cpu_seconds=time.process_time()-c0, training_episodes=len(transcripts)))
            if len(trained) != 62208 or trained & (history | reserved):
                raise ValueError("pilot training/held transcript exclusion failed")
            save("training-transcripts.pt", sorted(trained)); save("reserved-transcripts.pt", sorted(reserved))
            save("admission-summary.json", dict(history_count=len(history), training_count=len(trained), reserved_count=len(reserved),
                training_sha256=evidence.json_digest(sorted(trained)), reserved_sha256=evidence.json_digest(sorted(reserved))))
            snapshots = {}
            for index, (name, expected) in enumerate(sources.items()):
                work.boundary(); content = native(ROOT/name).read_bytes()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError("source changed during preparation")
                local = f"source/{index:03d}.bin"; target = output/local; target.parent.mkdir(exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(content)
                artifacts[local] = digest(target); snapshots[name] = local
            if source_hashes() != sources:
                raise ValueError("source closure changed during preparation")
            work.boundary()
            manifest = dict(schema=SCHEMA, contract=contract(), source_sha256=sources, source_snapshots=snapshots,
                inherited_history_sha256=HISTORY_PINS, artifacts_sha256=artifacts,
                no_neural_training_or_inference=True, automatic_retry=False)
            report["manifest_sha256"] = write(output/"manifest.json", manifest)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        if hasattr(error, "receipt"):
            report["admission_failure"] = deepcopy(error.receipt)
        raise
    finally:
        journal.close()
        report.update(work=work.report(), source_sha256=sources, artifacts_sha256=artifacts,
            journal_sha256=digest(output/"events.jsonl"), wall_seconds=time.monotonic()-started,
            cpu_seconds=time.process_time()-cpu, ended_utc=datetime.now(timezone.utc).isoformat(),
            timing_scope="Preparation entry including imports, guards, all canonical/adapted copies and file I/O; final receipt write excluded.")
        write(output/"preparation.json", report)
    return manifest


def _header(directory, expected_manifest_sha256):
    output = native(directory)
    if digest(output/"manifest.json") != expected_manifest_sha256:
        raise ValueError("explicit completed pilot manifest digest required")
    manifest, receipt = read(output/"manifest.json"), read(output/"preparation.json")
    if (manifest["schema"] != SCHEMA or manifest["contract"] != contract() or receipt["status"] != "completed"
            or receipt["manifest_sha256"] != expected_manifest_sha256 or manifest["source_sha256"] != source_hashes()
            or receipt["source_sha256"] != manifest["source_sha256"]
            or receipt["journal_sha256"] != digest(output/"events.jsonl")):
        raise ValueError("source-pinned successful pilot data preparation required")
    return output, manifest, receipt


def load_bank(directory, name, *, expected_manifest_sha256):
    output, manifest, _ = _header(directory, expected_manifest_sha256)
    if name not in contract()["banks"] or digest(output/(name+".pt")) != manifest["artifacts_sha256"][name+".pt"]:
        raise ValueError("explicit authenticated pilot bank required")
    return _load_pt(output/(name+".pt"))


def load(directory, *, expected_manifest_sha256, include_audit=False):
    output, manifest, receipt = _header(directory, expected_manifest_sha256)
    for name, expected in manifest["artifacts_sha256"].items():
        if digest(output/name) != expected:
            raise ValueError("sealed pilot artifact changed: "+name)
    cycles = [read(output/f"cycle-{cycle}.json") for cycle in range(CYCLES)]
    names = ["retention", "acquisition-1", "acquisition-2"]+(["audit"] if include_audit else [])
    return dict(manifest=manifest, preparation=receipt, cycles=cycles,
        banks={name: _load_pt(output/(name+".pt")) for name in names})
