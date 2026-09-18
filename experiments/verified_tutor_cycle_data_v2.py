"""Versioned cycle adapter: disclose repeated training text; keep heldout exclusion.

The owner supplies the parent, seed, additional protected fingerprints and saved
author decision. This adapter neither calls a teacher nor chooses a learner.
Old inventory admission is authenticated transitively, without rescanning its
historical archives. An explicitly pinned failed v1 attempt may contribute its
completed procedural archives, but never an uncommitted teacher call or lesson.
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
from experiments import verified_tutor_cycle_data as legacy

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-verified-tutor-cycle-data-v1"
CONTEXT_SCHEMA = "bic-verified-tutor-cycle-context-v1"
ADAPTER_SCHEMA = "bic-verified-tutor-cycle-data-adapter-v2"
MICRO, UPDATES = 32, 108
ARMS, PHASES = ("procedural", "tutor"), ("teaching", "withdrawal")


def source_hashes():
    from experiments import verified_tutor_author_v2 as author
    # Pin the helpers actually executed here. Calling the old inventory builder's
    # closure function imports its training module; no builder is run in a cycle.
    names = ("experiments/verified_tutor_cycle_data.py", "experiments/verified_tutor_cycle_data_v2.py", "experiments/verified_tutor_data.py",
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
            compiler_completions=0, compiler_receipts=[], neural_work=0, teacher_calls=0,
            reused_archive_copies=0, reused_archive_bytes=0, adapter_schema=ADAPTER_SCHEMA)
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

    def copy_archive(self, name, path, expected):
        """Authenticate, decode once and preserve the exact prior archive bytes."""
        self.work.boundary()
        path = _path(path)
        raw = io.native(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("reused procedural archive changed")
        self.inputs[_relative(path)] = expected
        self.receipt["bytes_read"] += len(raw)
        import torch
        self.receipt["archive_attempts"] += 1
        self.work.counts["safe_archive_loads"] += 1
        value = torch.load(streams.BytesIO(raw), map_location="cpu", weights_only=True)
        self.receipt["archive_completions"] += 1
        target = (self.output/name).resolve()
        if not target.is_relative_to(self.output) or target == self.output:
            raise ValueError("reused archive output escapes stage")
        io.native(target).parent.mkdir(parents=True, exist_ok=True)
        with io.native(target).open("xb") as stream:
            stream.write(raw)
        self.artifacts[name] = expected
        self.receipt["reused_archive_copies"] += 1
        self.receipt["reused_archive_bytes"] += len(raw)
        self.work.boundary()
        return value

    def copy_json(self, name, path, expected):
        self.work.boundary()
        path = _path(path)
        raw = io.native(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("reused procedural metadata changed")
        self.inputs[_relative(path)] = expected
        self.receipt["bytes_read"] += len(raw)
        target = (self.output/name).resolve()
        if not target.is_relative_to(self.output) or target == self.output:
            raise ValueError("reused metadata output escapes stage")
        with io.native(target).open("xb") as stream:
            stream.write(raw)
        self.artifacts[name] = expected

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
            return run.publish("cycle.json", dict(schema=CONTEXT_SCHEMA, adapter_schema=ADAPTER_SCHEMA,
                parent=parent, seed=seed, phase_seeds=seeds,
                start_cursor=parent["lifetime_updates"], micro_batch_size=MICRO, updates_per_phase=UPDATES,
                inventory=dict(directory=_relative(inventory_directory), manifest_sha256=inventory_manifest_sha256),
                protection=protection, fresh_only_protection=fresh_protection,
                training_reuse_policy="Original admitted training overlap is counted as repeated practice, not rejected.",
                **contracts))
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


def _reuse_procedural(run, record, *, context, context_path, cycle_sha256, teacher_decision, contract):
    compiler._keys(record, ("directory", "preparation_sha256"), "procedural reuse")
    directory = _path(record["directory"])
    if directory != context_path.parent/"compiled":
        raise ValueError("reuse must be the failed compilation of the adopted context")
    failed = run.read(directory/"preparation.json", record["preparation_sha256"])
    if (failed["schema"] != SCHEMA or failed["status"] != "failed"
            or failed["source_sha256"] != context["source_sha256"]
            or failed["source_sha256"] != legacy.source_hashes()
            or failed["compiler_attempts"] != 2 or failed["compiler_completions"] != 2
            or len(failed["compiler_receipts"]) != 2
            or any(r["status"] != "completed" for r in failed["compiler_receipts"])
            or failed.get("fresh_only_rejection", {}).get("condition") != "tutor"
            or failed["error"] != "ValueError('fresh curriculum repeats an admitted prior training transcript')"
            or failed["neural_work"] != 0 or failed["teacher_calls"] != 0
            or failed["input_sha256"].get(_relative(context_path)) != cycle_sha256
            or failed["input_sha256"].get(_relative(teacher_decision["path"])) != teacher_decision["sha256"]):
        raise ValueError("exact known completed procedural / failed novelty-admission attempt required")
    original._verify(failed["input_sha256"])
    pins = failed["artifact_sha256"]
    names = {f"procedural/{i:03d}.pt" for i in range(UPDATES)} | {"procedural-manifest.json", "procedural-provenance.json"}
    if set(pins) != names:
        raise ValueError("only the complete108 procedural archives and metadata may be reused")
    manifest = run.read(directory/"procedural-manifest.json", pins["procedural-manifest.json"])
    provenance = run.read(directory/"procedural-provenance.json", pins["procedural-provenance.json"])
    if (manifest["schema"] != compiler.SCHEMA or manifest["source_sha256"] != compiler.source_hashes()
            or manifest["contract_sha256"] != compiler.contract_sha256(contract)
            or manifest["recipe_sha256"] != compiler.recipe_sha256(compiler.procedural_recipe(contract))
            or manifest["provenance_sha256"] != compiler._hash(provenance)
            or len(provenance) != UPDATES):
        raise ValueError("saved procedural compiler identity differs")
    images = [run.copy_archive(f"procedural/{i:03d}.pt", directory/f"procedural/{i:03d}.pt",
                              pins[f"procedural/{i:03d}.pt"]) for i in range(UPDATES)]
    if compiler._hash(images) != manifest["images_sha256"]:
        raise ValueError("reused decoded images differ from the completed compiler result")
    for name in ("procedural-manifest.json", "procedural-provenance.json"):
        run.copy_json(name, directory/name, pins[name])
    recovery = dict(legacy_context=dict(path=_relative(context_path), sha256=cycle_sha256),
        failed_preparation=dict(path=_relative(directory/"preparation.json"), sha256=record["preparation_sha256"]),
        reused_compiler_manifest=dict(path=_relative(directory/"procedural-manifest.json"),sha256=pins["procedural-manifest.json"]),
        reused_images=UPDATES, regenerated_procedural_images=0,
        prior_failure_cost=dict(wall_seconds=failed["wall_seconds"],cpu_seconds=failed["cpu_seconds"],
            scope="Previously incurred failed preparation; recorded separately, never added to this invocation cost."))
    run.receipt["recovery"] = deepcopy(recovery)
    return dict(images=images, manifest=manifest, provenance=provenance), recovery


def _reuse_counts(images, contract, reference):
    """Actual rows and unique text identities, keeping explicit replay separate."""
    by_family, repeated, novel = {}, set(), set()
    for family in compiler.FAMILIES:
        selected = [compiler.transcript_sha256(row) for index,image in enumerate(images)
            if not contract["slots"][index]["replay"] for row in image["bundle"]["families"][family]]
        hits = [pin for pin in selected if pin in reference]
        unique = set(selected)
        repeated.update(hits); novel.update(unique-reference)
        by_family[family] = dict(nonreplay_rows=len(selected), repeated_original_training_rows=len(hits),
            novel_to_original_training_rows=len(selected)-len(hits), unique_nonreplay_transcripts=len(unique),
            unique_repeated_original_training_transcripts=len(set(hits)),
            unique_novel_to_original_training_transcripts=len(unique-reference))
    return dict(by_family=by_family, overall={k:sum(v[k] for v in by_family.values())
        for k in next(iter(by_family.values()))}, repeated_fingerprints=sorted(repeated),
        novel_fingerprints=sorted(novel)), novel


def compile_data(output, *, cycle_sha256, teacher_decision, max_seconds=1800,
                 deadline=None, progress=None, legacy_context=None, reuse_procedural=None):
    """Compile three streams once from an authenticated saved author result."""
    directory = _path(output)
    run = _Attempt(directory/"compiled", "compile-cycle", max_seconds, deadline, progress)
    error = None
    try:
        with io._cpu_guard(run.work):
            from experiments import verified_tutor_author_v2 as author
            if (legacy_context is None) != (reuse_procedural is None):
                raise ValueError("legacy context adoption and procedural reuse must be declared together")
            if legacy_context is not None:
                compiler._keys(legacy_context, ("path", "sha256"), "legacy context")
                if legacy_context["sha256"] != cycle_sha256:
                    raise ValueError("explicit old context pin differs")
                context_path = _path(legacy_context["path"])
                wanted_sources = legacy.source_hashes()
            else:
                context_path, wanted_sources = directory/"cycle.json", run.sources
            context = run.read(context_path, cycle_sha256)
            preparation = run.read(context_path.parent/"preparation.json")
            if (context["schema"] != CONTEXT_SCHEMA or preparation["status"] != "completed"
                    or preparation["manifest_sha256"] != cycle_sha256 or context["source_sha256"] != wanted_sources
                    or legacy_context is None and context.get("adapter_schema") != ADAPTER_SCHEMA):
                raise ValueError("completed current or explicitly pinned legacy cycle context required")
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
            reuse_counts, recovery = {}, None
            for label, recipe, contract in (("procedural", compiler.procedural_recipe(contracts["teaching"]), contracts["teaching"]),
                    ("tutor", decision["recipe"], contracts["teaching"]),
                    ("withdrawal", compiler.procedural_recipe(contracts["withdrawal"]), contracts["withdrawal"])):
                run.work.boundary()
                reused = label == "procedural" and reuse_procedural is not None
                if reused:
                    result, recovery = _reuse_procedural(run, reuse_procedural, context=context,
                        context_path=context_path, cycle_sha256=cycle_sha256,
                        teacher_decision=teacher_decision, contract=contract)
                else:
                    # Keep the request-bound allowance and the outer deadline unchanged.
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
                reuse_counts[label], novel = _reuse_counts(result["images"], contract, fresh_protected)
                fresh.update(novel)
                records, learner_pins[label], row_pins[label] = [], [], []
                for index, image in enumerate(result["images"]):
                    cursor = contract["start_cursor"]+index
                    if image["bundle"]["bundle_id"] != cursor or image["expected_evidence"]["bundle_id"] != cursor:
                        raise ValueError("materialized lesson cursor differs")
                    name = f"{label}/{index:03d}.pt"
                    pin = run.artifacts[name] if reused else run.save(name, image)["sha256"]
                    records.append(dict(path=name, sha256=pin, cursor=cursor))
                    learner_pins[label].append(original._learner_digest(image["bundle"]))
                    row_pins[label].append(image["expected_evidence"]["bundle_rows_sha256"])
                if not reused:
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
            accounting_record = run.save("nonreplay-accounting.json", reuse_counts)
            return run.publish("manifest.json", dict(schema=SCHEMA, adapter_schema=ADAPTER_SCHEMA,
                recovery=recovery, parent=deepcopy(context["parent"]),
                cycle_sha256=cycle_sha256, seed=context["seed"], phase_seeds=deepcopy(context["phase_seeds"]),
                start_cursor=context["start_cursor"], micro_batch_size=MICRO,
                updates_per_phase=UPDATES, phases=phases, teacher_decision=deepcopy(teacher_decision),
                tutor_compilation=tutor_pins, fresh_transcripts=fresh_record,
                training_reuse=dict(reference=fresh_protection, accounting=accounting_record,
                    by_stream={k:{field:v[field] for field in ("by_family","overall")} for k,v in reuse_counts.items()},
                    policy="Original admitted training repetition is explicit practice; historical/evaluation/prior-fresh exclusions remain strict.",
                    future_exclusion="fresh-transcripts contains only nonreplay texts outside the original training reference."),
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
    if (manifest["schema"] != SCHEMA or manifest.get("adapter_schema") != ADAPTER_SCHEMA
            or receipt["status"] != "completed"
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
