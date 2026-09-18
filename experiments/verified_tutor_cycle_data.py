"""Generic CPU-only cycle packaging over the frozen verified tutor compiler.

The owner supplies the parent, seed, additional protected fingerprints and saved
author decision. This adapter neither calls a teacher nor chooses a learner.
Old inventory admission is authenticated transitively, without rescanning its
historical archives. All newly read bytes and newly compiled lessons are pinned.
"""
from copy import deepcopy
import hashlib
import io as streams
import json
import math
from pathlib import Path
import time
import traceback

from experiments import foundation_tutor_loop_data as io
from experiments import verified_tutor_data as original
from experiments import verified_tutor_curriculum_v2 as compiler

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-verified-tutor-cycle-data-v1"
CONTEXT_SCHEMA = "bic-verified-tutor-cycle-context-v1"
MICRO, UPDATES = 32, 108
ARMS, PHASES = ("procedural", "tutor"), ("teaching", "withdrawal")


def source_hashes():
    from experiments import verified_tutor_author_v2 as author
    # Pin the helpers actually executed here. Calling the old inventory builder's
    # closure function imports its training module; no builder is run in a cycle.
    names = ("experiments/verified_tutor_cycle_data.py", "experiments/verified_tutor_data.py",
             "experiments/foundation_tutor_loop_data.py")
    return {**compiler.source_hashes(), **author.source_hashes(),
            **{name:io.digest(ROOT/name) for name in names}}


def _path(path):
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT) or result == ROOT:
        raise ValueError("artifact must remain inside the repository")
    return result


def _relative(path):
    return _path(path).relative_to(ROOT).as_posix()


def _parent(value):
    compiler._keys(value, ("identity_sha256", "weights_sha256", "cycle", "lifetime_updates"), "parent")
    if any(not compiler._pin(value[k]) for k in ("identity_sha256", "weights_sha256")):
        raise ValueError("caller-pinned parent and native weights required")
    for key in ("cycle", "lifetime_updates"):
        compiler._integer(value[key], 1, 2**63-1, key)
    return deepcopy(value)


def _fingerprints(value):
    if (type(value) is not list or any(not compiler._pin(x) for x in value)
            or value != sorted(set(value))):
        raise ValueError("sorted unique transcript SHA256 list required")
    return set(value)


def make_contract(inventory_sha256, protected_sha256, protected_count, *, parent,
                  seed, withdrawal=False, compiler_seconds=180):
    """Pure contract construction; every chapter retains all eighteen cells."""
    parent = _parent(parent)
    fixed = dict(motifs=["balanced"], realizations=["independent"],
                 default=dict(motif="balanced", realization="independent"))
    menu = dict(motifs=list(original.POOLS), realizations=list(compiler.REALIZATIONS),
                default=deepcopy(fixed["default"]))
    return compiler.validate_contract(dict(schema=compiler.CONTRACT_SCHEMA,
        inventory_sha256=inventory_sha256, protected_sha256=protected_sha256,
        protected_count=protected_count, seed=seed,
        start_cursor=parent["lifetime_updates"]+(UPDATES if withdrawal else 0),
        micro_batch_size=MICRO, layout="original",
        chapters=[deepcopy(fixed if withdrawal or c < 2 else menu) for c in range(6)],
        slots=[dict(chapter=c, depth=d, turns=t, replay=withdrawal or c < 2)
               for c in range(6) for d,t in compiler.CELLS],
        rehearsal_floor_by_depth={str(d): 18 for d in range(5)},
        replay_floor_per_cell=6 if withdrawal else 2,
        limits=dict(max_episodes=UPDATES*MICRO*3, max_observation_bytes=UPDATES*MICRO*3*1024,
            max_reply_target_bytes=UPDATES*MICRO*3*12*32, max_generator_calls=250000,
            max_seconds=compiler_seconds)))


class _Attempt:
    def __init__(self, directory, stage, max_seconds, deadline, progress):
        if (type(max_seconds) not in (int,float) or not math.isfinite(max_seconds)
                or not 0 < max_seconds <= 1800):
            raise ValueError("finite at-most1800-second stage budget required")
        if deadline is not None and (type(deadline) not in (int,float) or not math.isfinite(deadline)):
            raise ValueError("finite absolute monotonic deadline required")
        self.started, self.cpu = time.monotonic(), time.process_time()
        self.deadline = min(self.started+max_seconds, deadline if deadline is not None else math.inf)
        if self.deadline <= self.started:
            raise TimeoutError("campaign deadline already expired")
        self.output = _path(directory)
        io.native(self.output).mkdir(parents=True, exist_ok=False)
        self.work = io.WorkLedger(deadline=self.deadline, progress=progress)
        self.sources, self.inputs, self.artifacts = source_hashes(), {}, {}
        self.receipt = dict(schema=SCHEMA, stage=stage, status="running", source_sha256=self.sources,
            archive_attempts=0, archive_completions=0, bytes_read=0, compiler_attempts=0,
            compiler_completions=0, compiler_receipts=[], neural_work=0, teacher_calls=0)
        io.write(self.output/"started.json", self.receipt)

    def read(self, path, expected=None, *, archive=False):
        self.work.boundary()
        path = _path(path)
        raw = io.native(path).read_bytes()
        pin = hashlib.sha256(raw).hexdigest()
        if expected is not None and pin != expected:
            raise ValueError("input bytes differ: "+_relative(path))
        self.inputs[_relative(path)] = pin
        self.receipt["bytes_read"] += len(raw)
        if archive:
            import torch
            self.receipt["archive_attempts"] += 1
            self.work.counts["safe_archive_loads"] += 1
            value = torch.load(streams.BytesIO(raw), map_location="cpu", weights_only=True)
            self.receipt["archive_completions"] += 1
        else:
            value = json.loads(raw)
        self.work.boundary()
        return value

    def save(self, name, value):
        self.work.boundary()
        path = (self.output/name).resolve()
        if not path.is_relative_to(self.output) or path == self.output:
            raise ValueError("output escapes exclusive stage")
        self.artifacts[name] = io.write(path, value)
        return dict(path=_relative(path), sha256=self.artifacts[name])

    def publish(self, name, value):
        self.work.boundary()
        original._verify(self.inputs)
        if source_hashes() != self.sources:
            raise ValueError("cycle packaging source changed")
        value.update(source_sha256=self.sources, input_sha256=deepcopy(self.inputs),
                     artifact_sha256=deepcopy(self.artifacts))
        self.receipt["manifest_sha256"] = io.write(self.output/name, value)
        return value

    def finish(self, error):
        self.receipt.update(status="failed" if error is not None else "completed",
            input_sha256=self.inputs, artifact_sha256=self.artifacts, work=self.work.report(),
            wall_seconds=time.monotonic()-self.started, cpu_seconds=time.process_time()-self.cpu,
            accounting="New stage cost only. Compiler costs are nested; inherited inventory and old failures are not charged again.")
        if error is not None:
            self.receipt.update(error=repr(error), traceback=traceback.format_exc())
        io.write(self.output/"preparation.json", self.receipt)


def _inventory(run, directory, pin, extras):
    directory = _path(directory)
    manifest = run.read(directory/"manifest.json", pin)
    prior = run.read(directory/"preparation.json")
    if (manifest["schema"] != original.INVENTORY_SCHEMA or prior["status"] != "completed"
            or prior["manifest_sha256"] != pin
            or manifest["source_sha256"].get("experiments/verified_tutor_data.py") != io.digest(original.__file__)):
        raise ValueError("original completed admitted inventory required")
    original._verify(manifest["source_sha256"])
    inventory = run.read(directory/"inventory.json", manifest["artifact_sha256"]["inventory.json"])
    if compiler.inventory_sha256(inventory) != manifest["inventory_sha256"]:
        raise ValueError("admitted inventory identity differs")
    protected = _fingerprints(run.read(directory/"protected.pt",
        manifest["artifact_sha256"]["protected.pt"], archive=True))
    if (len(protected) != manifest["protected_count"]
            or compiler._hash(sorted(protected)) != manifest["protected_sha256"]):
        raise ValueError("original protected inventory differs")
    records, seen = [], set()
    for record in extras:
        compiler._keys(record, ("path", "sha256"), "additional protection record")
        path = _path(record["path"])
        if path.suffix != ".json" or not compiler._pin(record["sha256"]) or path in seen:
            raise ValueError("distinct caller-pinned JSON protection lists required")
        seen.add(path)
        protected.update(_fingerprints(run.read(path, record["sha256"])))
        records.append(dict(path=_relative(path), sha256=record["sha256"]))
    return inventory, protected, records


def _fresh_only(run, records):
    protected, normalized, seen = set(), [], set()
    for record in records:
        compiler._keys(record, ("path", "sha256"), "fresh-only protection record")
        path = _path(record["path"])
        if path.suffix != ".json" or not compiler._pin(record["sha256"]) or path in seen:
            raise ValueError("distinct caller-pinned JSON fresh-only protection lists required")
        seen.add(path)
        protected.update(_fingerprints(run.read(path, record["sha256"])))
        normalized.append(dict(path=_relative(path), sha256=record["sha256"]))
    return protected, dict(sha256=compiler._hash(sorted(protected)), count=len(protected), records=normalized)


def prepare_cycle(output, *, inventory_directory, inventory_manifest_sha256, parent,
                  seed, withdrawal_seed=None, extra_protected=(), compiler_seconds=180, max_seconds=120,
                  deadline=None, progress=None, fresh_only_protected=()):
    """Freeze the exact contract before a single author request is constructed."""
    run = _Attempt(output, "prepare-cycle", max_seconds, deadline, progress)
    error = None
    try:
        with io._cpu_guard(run.work):
            parent = _parent(parent)
            inventory, protected, extras = _inventory(run, inventory_directory,
                inventory_manifest_sha256, extra_protected)
            protection = dict(sha256=compiler._hash(sorted(protected)), count=len(protected), extra=extras)
            _, fresh_protection = _fresh_only(run, fresh_only_protected)
            seeds = dict(teaching=seed, withdrawal=seed if withdrawal_seed is None else withdrawal_seed)
            contracts = {}
            for name in PHASES:
                value = make_contract(compiler.inventory_sha256(inventory), protection["sha256"],
                    protection["count"], parent=parent, seed=seeds[name], withdrawal=name=="withdrawal",
                    compiler_seconds=compiler_seconds)
                contracts[name+"_contract"] = {**run.save(name+"-contract.json", value),
                    "identity_sha256": compiler.contract_sha256(value)}
            return run.publish("cycle.json", dict(schema=CONTEXT_SCHEMA, parent=parent, seed=seed, phase_seeds=seeds,
                start_cursor=parent["lifetime_updates"], micro_batch_size=MICRO, updates_per_phase=UPDATES,
                inventory=dict(directory=_relative(inventory_directory), manifest_sha256=inventory_manifest_sha256),
                protection=protection, fresh_only_protection=fresh_protection, **contracts))
    except BaseException as exc:
        error = exc
        raise
    finally:
        run.finish(error)


def _contracts(run, context):
    result = {}
    for phase in PHASES:
        record = context[phase+"_contract"]
        value = run.read(record["path"], record["sha256"])
        if (compiler.contract_sha256(value) != record["identity_sha256"]
                or value != make_contract(value["inventory_sha256"], context["protection"]["sha256"],
                    context["protection"]["count"], parent=context["parent"], seed=context["phase_seeds"][phase],
                    withdrawal=phase=="withdrawal", compiler_seconds=value["limits"]["max_seconds"])):
            raise ValueError("frozen cycle contract changed")
        result[phase] = value
    return result


def compile_data(output, *, cycle_sha256, teacher_decision, max_seconds=1800,
                 deadline=None, progress=None):
    """Compile three streams once from an authenticated saved author result."""
    directory = _path(output)
    run = _Attempt(directory/"compiled", "compile-cycle", max_seconds, deadline, progress)
    error = None
    try:
        with io._cpu_guard(run.work):
            from experiments import verified_tutor_author_v2 as author
            context = run.read(directory/"cycle.json", cycle_sha256)
            preparation = run.read(directory/"preparation.json")
            if (context["schema"] != CONTEXT_SCHEMA or preparation["status"] != "completed"
                    or preparation["manifest_sha256"] != cycle_sha256 or context["source_sha256"] != run.sources):
                raise ValueError("completed current-source cycle context required")
            original._verify(context["input_sha256"])
            inventory, protected, extras = _inventory(run, context["inventory"]["directory"],
                context["inventory"]["manifest_sha256"], context["protection"]["extra"])
            if dict(sha256=compiler._hash(sorted(protected)), count=len(protected), extra=extras) != context["protection"]:
                raise ValueError("cycle protection union differs")
            fresh_protected, fresh_protection = _fresh_only(run, context["fresh_only_protection"]["records"])
            if fresh_protection != context["fresh_only_protection"]:
                raise ValueError("fresh-only protection union differs")
            contracts = _contracts(run, context)
            compiler._keys(teacher_decision, ("path", "sha256", "request_sha256"), "teacher decision")
            path = _path(teacher_decision["path"])
            run.read(path, teacher_decision["sha256"])
            decision = author.load_result(path, expected_sha256=teacher_decision["sha256"],
                expected_request_sha256=teacher_decision["request_sha256"])
            if (decision["parent"] != context["parent"]
                    or decision["contract_sha256"] != compiler.contract_sha256(contracts["teaching"])):
                raise ValueError("saved author decision belongs to another cycle")
            run.read(path.parent/"intent.json", decision["intent_sha256"])
            if decision["teacher_cost"]["response_file_sha256"] is not None:
                run.read(path.parent/"response.json", decision["teacher_cost"]["response_file_sha256"])
            for event in decision["teacher_cost"]["api_events"]:
                if "http_error" in event:
                    run.read(path.parent/event["error_file"], event["error_file_sha256"])
            phases, manifests, learner_pins, row_pins, fresh = {p:{} for p in PHASES}, {}, {}, {}, set()
            for label, recipe, contract in (("procedural", compiler.procedural_recipe(contracts["teaching"]), contracts["teaching"]),
                    ("tutor", decision["recipe"], contracts["teaching"]),
                    ("withdrawal", compiler.procedural_recipe(contracts["withdrawal"]), contracts["withdrawal"])):
                run.work.boundary()
                # Do not silently change a request-bound compiler allowance or reset the global budget.
                if time.monotonic()+contract["limits"]["max_seconds"]+2 > run.deadline:
                    raise TimeoutError("insufficient remaining budget for the frozen compiler allowance")
                run.receipt["compiler_attempts"] += 1
                try:
                    result = compiler.compile_curriculum(recipe, admitted_parent_inventory=inventory,
                        protected_transcripts=protected, coverage_contract=contract)
                except BaseException as exc:
                    if hasattr(exc, "compilation_report"):
                        run.receipt["compiler_receipts"].append(exc.compilation_report)
                    raise
                run.receipt["compiler_completions"] += 1
                run.receipt["compiler_receipts"].append(result["receipt"])
                if len(result["images"]) != UPDATES:
                    raise ValueError("one complete108-bundle curriculum required")
                actual_fresh = {compiler.transcript_sha256(row)
                    for index,image in enumerate(result["images"]) if not contract["slots"][index]["replay"]
                    for rows in image["bundle"]["families"].values() for row in rows}
                overlap = actual_fresh & fresh_protected
                if overlap:
                    run.receipt["fresh_only_rejection"] = dict(condition=label,
                        checked_unique_transcripts=len(actual_fresh), overlap=sorted(overlap),
                        overlap_count=len(overlap), protection_sha256=fresh_protection["sha256"])
                    raise ValueError("fresh curriculum repeats an admitted prior training transcript")
                records, learner_pins[label], row_pins[label] = [], [], []
                for index, image in enumerate(result["images"]):
                    cursor = contract["start_cursor"]+index
                    if image["bundle"]["bundle_id"] != cursor or image["expected_evidence"]["bundle_id"] != cursor:
                        raise ValueError("materialized lesson cursor differs")
                    name = f"{label}/{index:03d}.pt"
                    pin = run.save(name, image)["sha256"]
                    records.append(dict(path=name, sha256=pin, cursor=cursor))
                    learner_pins[label].append(original._learner_digest(image["bundle"]))
                    row_pins[label].append(image["expected_evidence"]["bundle_rows_sha256"])
                    if not contract["slots"][index]["replay"]:
                        fresh.update(compiler.transcript_sha256(row)
                            for rows in image["bundle"]["families"].values() for row in rows)
                run.save(label+"-manifest.json", result["manifest"])
                run.save(label+"-provenance.json", result["provenance"])
                manifests[label] = result["manifest"]
                if label == "withdrawal":
                    phases["withdrawal"] = dict(procedural=records, tutor=deepcopy(records))
                else:
                    phases["teaching"][label] = records
                run.work.event(dict(event="compiled", condition=label, updates=UPDATES))
                del result
            tutor_pins = {k:manifests["tutor"][k] for k in ("contract_sha256", "recipe_sha256")}
            if tutor_pins != {k:decision[k] for k in tutor_pins}:
                raise ValueError("compiled treatment is not the accepted recipe")
            different = sum(a != b for a,b in zip(learner_pins["procedural"], learner_pins["tutor"]))
            run.save("teaching-contrast.json", dict(learner_sha256=learner_pins, row_identity_sha256=row_pins))
            fresh_record = {**run.save("fresh-transcripts.json", sorted(fresh)), "count":len(fresh)}
            return run.publish("manifest.json", dict(schema=SCHEMA, parent=deepcopy(context["parent"]),
                cycle_sha256=cycle_sha256, seed=context["seed"], phase_seeds=deepcopy(context["phase_seeds"]),
                start_cursor=context["start_cursor"], micro_batch_size=MICRO,
                updates_per_phase=UPDATES, phases=phases, teacher_decision=deepcopy(teacher_decision),
                tutor_compilation=tutor_pins, fresh_transcripts=fresh_record,
                fresh_only_protection=fresh_protection,
                treatment=dict(different_teaching_bundles=different, zero_treatment_contrast=different==0,
                    different_row_identity_bundles=sum(a != b for a,b in zip(row_pins["procedural"],row_pins["tutor"])),
                    actual_exposures={k:v["exposures"] for k,v in manifests.items()},
                    accepted_local_teacher=decision["outcome"]=="local_accepted", equal_caps_not_equal_actual_tokens=True),
                scope="Fixed broad coverage; persisted external authorship, independently verified lessons, no learned self-direction or promotion claim."))
    except BaseException as exc:
        error = exc
        raise
    finally:
        run.finish(error)


def load_manifest(directory, *, expected_manifest_sha256):
    """Read-only worker boundary: no author imports, tensor loads or generation."""
    directory = _path(directory)
    if io.digest(directory/"manifest.json") != expected_manifest_sha256:
        raise ValueError("caller-pinned completed cycle manifest required")
    manifest, receipt = io.read(directory/"manifest.json"), io.read(directory/"preparation.json")
    if (manifest["schema"] != SCHEMA or receipt["status"] != "completed"
            or receipt["manifest_sha256"] != expected_manifest_sha256
            or manifest["micro_batch_size"] != MICRO or manifest["updates_per_phase"] != UPDATES
            or set(manifest["phases"]) != set(PHASES)):
        raise ValueError("completed exact cycle packaging required")
    parent = _parent(manifest["parent"])
    if manifest["start_cursor"] != parent["lifetime_updates"]:
        raise ValueError("cycle parent cursor differs")
    original._verify(manifest["source_sha256"])
    original._verify(manifest["input_sha256"])
    for name, pin in manifest["artifact_sha256"].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or path == directory or io.digest(path) != pin:
            raise ValueError("compiled artifact differs")
    for phase_index, phase in enumerate(PHASES):
        if set(manifest["phases"][phase]) != set(ARMS):
            raise ValueError("two complete matched branches required")
        for records in manifest["phases"][phase].values():
            if len(records) != UPDATES or [r["cursor"] for r in records] != list(range(
                    manifest["start_cursor"]+phase_index*UPDATES, manifest["start_cursor"]+(phase_index+1)*UPDATES)):
                raise ValueError("ordered complete phase cursor range differs")
            for record in records:
                if manifest["artifact_sha256"].get(record["path"]) != record["sha256"]:
                    raise ValueError("phase image absent from authenticated artifacts")
    if manifest["phases"]["withdrawal"]["procedural"] != manifest["phases"]["withdrawal"]["tutor"]:
        raise ValueError("withdrawal must be the same lessons and order")
    return manifest
