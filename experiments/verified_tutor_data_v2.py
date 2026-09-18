"""Explicit compile-only recovery using the completed v1 inventory unchanged.

A separate v2 author request replaces no prior artifact. The failed v1 request's
original unknown physical-work record remains a pinned input and separate cost.
No inventory preparation, teacher transport or learner work occurs in this module.
"""
from copy import deepcopy
from pathlib import Path
import time
import traceback

from experiments import verified_tutor_data as original_data
from experiments import foundation_tutor_loop_data as io
from experiments import verified_tutor_curriculum as compiler

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = original_data.SCHEMA
INVENTORY_SCHEMA = original_data.INVENTORY_SCHEMA
RECOVERY_SCHEMA = "bic-verified-tutor-data-recovery-v2"
RECOVERY_DECLARATION = "docs/VERIFIED_TUTOR_RECOVERY.md"
START, MICRO, UPDATES = original_data.START, original_data.MICRO, original_data.UPDATES
BANK_COUNTS = original_data.BANK_COUNTS
PRIOR_AUTHOR = "runs/verified-tutor-author-local/attempt-001/uncommitted.json"
PRIOR_AUTHOR_SHA256 = "6284cc0d41cf1dbab7c8eae77fb8e90e286ee645f0bdc484a3bb1896e3a3d150"
PRIOR_SERVER = "runs/verified-tutor-author-local/attempt-001/server-context.json"
PRIOR_SERVER_SHA256 = "5bd55a337313442cdf6b5fc753bf85600dd543f366fa1863ca0b2aa333c7bd25"
AUTHOR_RESULT = "runs/verified-tutor-author-local/attempt-002/result.json"
_relative, _verify = original_data._relative, original_data._verify
_contract, _learner_digest = original_data._contract, original_data._learner_digest


def source_hashes(*, include_author=True):
    result = original_data.source_hashes(include_author=False)
    for name in ("experiments/verified_tutor_data_v2.py", "experiments/verified_tutor_author.py", RECOVERY_DECLARATION):
        result[name] = io.digest(ROOT/name)
    if include_author:
        from experiments import verified_tutor_author_v2 as author
        result.update(author.source_hashes())
    return result


def _prior_author_attempt(run):
    path = ROOT/PRIOR_AUTHOR
    run.pin(path, PRIOR_AUTHOR_SHA256)
    record = io.read(path)
    cost = record["teacher_cost"]
    if (record["schema"] != "bic-verified-tutor-author-v1" or record["status"] != "unknown_or_uncommitted"
            or record["automatic_retry"] is not False or cost["chat_attempts"] != 1
            or cost["chat_completions"] != 0 or cost["physical_teacher_work_unknown"] is not True):
        raise ValueError("exact unresolved original author attempt required")
    run.pin(path.parent/"intent.json", record["intent_sha256"])
    run.pin(ROOT/PRIOR_SERVER, PRIOR_SERVER_SHA256)
    return dict(path=PRIOR_AUTHOR, sha256=PRIOR_AUTHOR_SHA256, status=record["status"],
        error_type=record["error_type"], request_sha256=record["request_sha256"],
        intent_sha256=record["intent_sha256"], teacher_cost=deepcopy(cost),
        server_context=dict(path=PRIOR_SERVER, sha256=PRIOR_SERVER_SHA256),
        original_scope=record["scope"],
        recovery_scope="Explicit separate attempt002 after an HTTP compatibility failure. Prior server warmup was observed; original generation/usage remains unknown, never zeroed or counted as a completion. Original slot is not retried or adopted.")


class _Attempt:
    def __init__(self, directory, stage, max_seconds, progress, *, include_author=False):
        if type(max_seconds) not in (int,float) or not 0 < max_seconds <= (1800 if include_author else 600):
            raise ValueError("explicit bounded preparation allowance required")
        self.started, self.cpu = time.monotonic(), time.process_time()
        self.output = Path(directory).resolve()
        io.native(self.output).mkdir(parents=True, exist_ok=False)
        self.work = io.WorkLedger(deadline=self.started+max_seconds, progress=progress)
        self.include_author, self.sources = include_author, source_hashes(include_author=include_author)
        self.artifacts, self.inputs = {}, {}
        self.receipt = dict(schema=SCHEMA, stage=stage, status="running", source_sha256=self.sources,
            archive_attempts=0, archive_completions=0, bytes_deserialized=0, compiler_attempts=0,
            compiler_completions=0, compiler_receipts=[], neural_work=0, teacher_calls=0)
        io.write(self.output/"started.json", self.receipt)

    def save(self, name, value):
        self.work.boundary()
        self.artifacts[name] = io.write(self.output/name, value)
        return self.artifacts[name]

    def pin(self, path, expected=None):
        actual = io.digest(path)
        if expected is not None and actual != expected:
            raise ValueError("input bytes differ: " + str(path))
        self.inputs[_relative(path)] = actual
        return actual

    def archive(self, path, expected):
        self.work.boundary(); self.pin(path, expected)
        self.receipt["archive_attempts"] += 1
        value = io._load_pt(path, self.work)
        self.receipt["archive_completions"] += 1
        self.receipt["bytes_deserialized"] += io.native(path).stat().st_size
        self.work.boundary()
        return value

    def finish(self, error=None):
        self.receipt.update(status="failed" if error is not None else "completed",
            input_sha256=self.inputs, artifact_sha256=self.artifacts, work=self.work.report(),
            wall_seconds=time.monotonic()-self.started, cpu_seconds=time.process_time()-self.cpu,
            accounting="Outer wall/CPU include all nested archive/compiler costs; do not add them again.")
        if error is not None:
            self.receipt.update(error=repr(error), traceback=traceback.format_exc())
        io.write(self.output/"preparation.json", self.receipt)

    def complete_manifest(self, manifest):
        self.work.boundary(); _verify(self.inputs)
        if source_hashes(include_author=self.include_author) != self.sources:
            raise ValueError("packaging source changed")
        manifest.update(source_sha256=self.sources, input_sha256=self.inputs, artifact_sha256=deepcopy(self.artifacts))
        self.receipt["manifest_sha256"] = io.write(self.output/"manifest.json", manifest)
        return manifest


def compile_data(output, *, inventory_directory, inventory_manifest_sha256,
                 teacher_decision, parent, max_seconds=1800, progress=None):
    """Compile only the authenticated persisted decision; never invoke a teacher."""
    run = _Attempt(output, "compilation-author-v2-recovery", max_seconds, progress, include_author=True)
    error = None
    try:
        with io._cpu_guard(run.work):
            from experiments import verified_tutor_author_v2 as author
            prior_author_attempt = _prior_author_attempt(run)
            run.receipt["prior_author_attempt"] = deepcopy(prior_author_attempt)
            directory = Path(inventory_directory).resolve()
            run.pin(directory/"manifest.json", inventory_manifest_sha256)
            inventory_manifest, prior = io.read(directory/"manifest.json"), io.read(directory/"preparation.json")
            run.pin(directory/"preparation.json")
            if (inventory_manifest["schema"] != INVENTORY_SCHEMA or prior["status"] != "completed"
                    or prior["manifest_sha256"] != inventory_manifest_sha256
                    or inventory_manifest["source_sha256"] != original_data.source_hashes(include_author=False)):
                raise ValueError("completed source-pinned inventory preparation required")
            _verify(inventory_manifest["input_sha256"])
            for name, pin in inventory_manifest["artifact_sha256"].items():
                path = (directory/name).resolve()
                if not path.is_relative_to(directory): raise ValueError("inventory artifact escapes directory")
                run.pin(path, pin)
            inventory = io.read(directory/"inventory.json")
            protected = set(run.archive(directory/"protected.pt", inventory_manifest["artifact_sha256"]["protected.pt"]))
            teaching, withdrawal = io.read(directory/"teaching-contract.json"), io.read(directory/"withdrawal-contract.json")
            if teaching != _contract(inventory, compiler._hash(sorted(protected))) or withdrawal != _contract(inventory, compiler._hash(sorted(protected)), withdrawal=True):
                raise ValueError("fixed curriculum contract changed")
            if (set(teacher_decision) != {"path","sha256","request_sha256"}
                    or set(parent) != {"identity_sha256","weights_sha256","lifetime_updates"}
                    or parent["lifetime_updates"] != START):
                raise ValueError("pinned actual1944 parent and persisted decision required")
            if teacher_decision["path"] != AUTHOR_RESULT:
                raise ValueError("the explicitly separate author attempt002 result is required")
            decision_path = (ROOT/teacher_decision["path"]).resolve()
            run.pin(decision_path, teacher_decision["sha256"])
            decision = author.load_result(decision_path, expected_sha256=teacher_decision["sha256"],
                                          expected_request_sha256=teacher_decision["request_sha256"])
            if (any(decision["parent"][k] != v for k,v in parent.items())
                    or decision["contract_sha256"] != compiler.contract_sha256(teaching)):
                raise ValueError("teacher decision belongs to another parent or contract")
            run.pin(decision_path.parent/"intent.json", decision["intent_sha256"])
            if decision["teacher_cost"]["response_file_sha256"] is not None:
                run.pin(decision_path.parent/"response.json", decision["teacher_cost"]["response_file_sha256"])
            phases = dict(teaching={}, withdrawal={})
            summaries, rows_pins, learner_pins = {}, {}, {}
            for label, recipe, contract in (("procedural", compiler.procedural_recipe(teaching), teaching),
                    ("tutor", decision["recipe"], teaching), ("withdrawal", compiler.procedural_recipe(withdrawal), withdrawal)):
                run.work.boundary(); run.receipt["compiler_attempts"] += 1
                try:
                    result = compiler.compile_curriculum(recipe, admitted_parent_inventory=inventory,
                        protected_transcripts=protected, coverage_contract=contract)
                except BaseException as exc:
                    if hasattr(exc, "compilation_report"): run.receipt["compiler_receipts"].append(exc.compilation_report)
                    raise
                run.receipt["compiler_completions"] += 1
                run.receipt["compiler_receipts"].append(result["receipt"])
                records, current_pins, actual_pins = [], [], []
                if len(result["images"]) != UPDATES: raise ValueError("exact108 compiled updates required")
                for index, image in enumerate(result["images"]):
                    cursor = contract["start_cursor"]+index
                    if image["bundle"]["bundle_id"] != cursor or image["expected_evidence"]["bundle_id"] != cursor:
                        raise ValueError("compiler cursor differs")
                    name = f"{label}/{index:03d}.pt"
                    records.append(dict(path=name, sha256=run.save(name,image), cursor=cursor))
                    current_pins.append(image["expected_evidence"]["bundle_rows_sha256"])
                    actual_pins.append(_learner_digest(image["bundle"]))
                run.save(label+"-manifest.json", result["manifest"])
                run.save(label+"-provenance.json", result["provenance"])
                summaries[label], rows_pins[label] = result["manifest"], current_pins
                learner_pins[label] = actual_pins
                if label == "withdrawal": phases["withdrawal"] = dict(procedural=records, tutor=deepcopy(records))
                else: phases["teaching"][label] = records
                run.work.event(dict(event="compiled", condition=label, updates=UPDATES))
                del result
            if (summaries["tutor"]["contract_sha256"] != decision["contract_sha256"]
                    or summaries["tutor"]["recipe_sha256"] != decision["recipe_sha256"]):
                raise ValueError("materialized tutor lessons do not match the authenticated accepted recipe")
            different = sum(a != b for a,b in zip(learner_pins["procedural"],learner_pins["tutor"]))
            row_differences = sum(a != b for a,b in zip(rows_pins["procedural"],rows_pins["tutor"]))
            run.save("teaching-contrast.json",dict(learner_sha256=learner_pins, row_identity_sha256=rows_pins))
            raw = io.native(directory/"banks.pt").read_bytes()
            with io.native(run.output/"banks.pt").open("xb") as stream: stream.write(raw)
            run.artifacts["banks.pt"] = io.digest(run.output/"banks.pt")
            if run.artifacts["banks.pt"] != inventory_manifest["banks"]["sha256"]:
                raise ValueError("reused bank copy differs")
            manifest = run.complete_manifest(dict(schema=SCHEMA, recovery_schema=RECOVERY_SCHEMA,
                prior_author_attempt=deepcopy(prior_author_attempt),
                start_cursor=START, micro_batch_size=MICRO,
                phases=phases, banks=dict(path="banks.pt",sha256=run.artifacts["banks.pt"]), bank_counts=BANK_COUNTS,
                teacher_decision=deepcopy(teacher_decision), parent=deepcopy(parent),
                tutor_compilation={k: summaries["tutor"][k] for k in ("contract_sha256","recipe_sha256")},
                treatment=dict(different_teaching_bundles=different, zero_treatment_contrast=different==0,
                    different_row_identity_bundles=row_differences,
                    actual_exposures={k: v["exposures"] for k,v in summaries.items()},
                    accepted_local_teacher=decision["outcome"]=="local_accepted",
                    equal_caps_not_equal_actual_tokens=True),
                inventory_preparation=dict(manifest_sha256=inventory_manifest_sha256,
                    wall_seconds=prior["wall_seconds"],cpu_seconds=prior["cpu_seconds"]),
                scope="Shared admitted parents and bounds; reused observed evaluation banks; no automatic promotion."))
        return manifest
    except BaseException as exc:
        error = exc; raise
    finally:
        run.finish(error)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile",))
    parser.add_argument("output")
    parser.add_argument("--inventory-directory", required=True)
    parser.add_argument("--inventory-manifest-sha256", required=True)
    parser.add_argument("--teacher-decision-json", required=True)
    parser.add_argument("--parent-json", required=True)
    args = parser.parse_args()
    compile_data(args.output, inventory_directory=args.inventory_directory,
        inventory_manifest_sha256=args.inventory_manifest_sha256,
        teacher_decision=io.read(args.teacher_decision_json), parent=io.read(args.parent_json),
        progress=lambda value: print(io.encoded(value).decode().strip(), flush=True))
    print(io.encoded(dict(manifest_sha256=io.digest(Path(args.output)/"manifest.json"))).decode().strip(), flush=True)
