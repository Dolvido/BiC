"""CPU packaging for authenticated external curriculum authorship and withdrawal.

Inventory reads admitted cache images without regeneration. Compilation alone
constructs lessons through the independently checked, bounded compiler. Neither
stage constructs a learner or contacts a teacher; failures are terminal receipts.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import time
import traceback

from experiments import foundation_tutor_loop_data as io
from experiments import recurrent_read_data as cached
from experiments import shared_state_continuation_data as current
from experiments import verified_tutor_curriculum as compiler

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-verified-tutor-data-v1"
INVENTORY_SCHEMA = "bic-verified-tutor-data-inventory-v1"
DECLARATION = "docs/VERIFIED_TUTOR_DATA.md"
CURRENT = "runs/shared-state-continuation-data-local/attempt-001"
CURRENT_SHA = "34ac898adce7446ce9edcd67e821fa42c2d5217883ccb3f51f0272225cab3795"
START, MICRO, UPDATES, SEED = 1944, 32, 108, 852802001
PROTECTED_COUNT = 1641672
BANK_COUNTS = dict(dev=720, train_fit=108, retention=720)
POOLS = ("balanced", "copy_emphasis", "change_emphasis")


def source_hashes(*, include_author=False):
    result = {**current.source_hashes(), **compiler.source_hashes(),
        "experiments/verified_tutor_data.py": io.digest(__file__),
        DECLARATION: io.digest(ROOT/DECLARATION)}
    if include_author:
        from experiments import verified_tutor_author as author
        result.update(author.source_hashes())
    return result


def _relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def _verify(pins):
    for name, pin in pins.items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or io.digest(path) != pin:
            raise ValueError("authenticated input changed: " + name)


def _pools(candidates):
    """Deterministic relative-copy pools, with identical choices for tied cells."""
    pools = {name: [] for name in POOLS}
    report = {}
    for depth, turns in compiler.CELLS:
        values = candidates[(depth, turns)]
        ranked = sorted(values, key=lambda x: (x["copies"], x["descriptor"]["bundle_id"],
                                              x["descriptor"]["pair_index"]))
        if len(ranked) < 16:
            raise ValueError("each cell needs sixteen distinct admitted parents")
        balanced = [ranked[i*(len(ranked)-1)//15] for i in range(16)]
        tied = ranked[0]["copies"] == ranked[-1]["copies"]
        chosen = dict(balanced=balanced,
            copy_emphasis=balanced if tied else sorted(values, key=lambda x:
                (-x["copies"], x["descriptor"]["bundle_id"], x["descriptor"]["pair_index"]))[:16],
            change_emphasis=balanced if tied else ranked[:16])
        ids = {name: {compiler._hash(v["descriptor"]) for v in group} for name, group in chosen.items()}
        for name, group in chosen.items():
            pools[name].extend(deepcopy(v["descriptor"]) for v in group)
        report[f"d{depth}/t{turns}"] = dict(available=len(values), copies_range=[ranked[0]["copies"], ranked[-1]["copies"]],
            pools={name: dict(count=len(group), copy_counts=dict(sorted(Counter(str(v["copies"]) for v in group).items())))
                   for name, group in chosen.items()},
            overlap={a+"/"+b: len(ids[a]&ids[b]) for i,a in enumerate(POOLS) for b in POOLS[i+1:]},
            all_pools_identical=tied)
    return pools, report


def _learner_digest(bundle):
    # Names, sentence order and all supervision count; generator metadata does not.
    return compiler._hash([[family, [[{key: turn.get(key) for key in
        ("text", "observations", "target", "reply")} for turn in row["turns"]]
        for row in bundle["families"][family]]] for family in compiler.FAMILIES])


def _contract(inventory, protected_sha256, *, withdrawal=False):
    fixed = dict(motifs=["balanced"], realizations=["independent"],
                 default=dict(motif="balanced", realization="independent"))
    menu = dict(motifs=list(POOLS), realizations=list(compiler.REALIZATIONS), default=deepcopy(fixed["default"]))
    value = dict(schema=compiler.CONTRACT_SCHEMA, inventory_sha256=compiler.inventory_sha256(inventory),
        protected_sha256=protected_sha256, protected_count=PROTECTED_COUNT, seed=SEED,
        start_cursor=START+(UPDATES if withdrawal else 0), micro_batch_size=MICRO, layout="original",
        chapters=[deepcopy(fixed if withdrawal or c < 2 else menu) for c in range(6)],
        slots=[dict(chapter=c, depth=d, turns=t, replay=withdrawal or c < 2)
               for c in range(6) for d,t in compiler.CELLS],
        rehearsal_floor_by_depth={str(d): 18 for d in range(5)},
        replay_floor_per_cell=6 if withdrawal else 2,
        limits=dict(max_episodes=UPDATES*MICRO*3, max_observation_bytes=UPDATES*MICRO*3*1024,
            max_reply_target_bytes=UPDATES*MICRO*3*12*32, max_generator_calls=250000, max_seconds=600))
    return compiler.validate_contract(value)


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


def prepare_inventory(output, *, max_seconds=600, progress=None):
    """Read only admitted cycle-zero images plus exact protected inventories."""
    run = _Attempt(output, "inventory", max_seconds, progress)
    error = None
    try:
        with io._cpu_guard(run.work):
            from experiments import foundation_evidence as evidence, foundation_plan as planning
            from experiments import shared_state_data as earlier, entity_retrieval_data as previous
            manifests = {}
            for name, directory, pin, loader in (
                ("cache", current.PARENT, current.PARENT_SHA, cached.load_manifest),
                ("earlier", current.EARLIER, current.EARLIER_SHA, earlier.load_manifest),
                ("previous", current.PREVIOUS, current.PREVIOUS_SHA, previous.load_manifest),
                ("current", CURRENT, CURRENT_SHA, current.load_manifest)):
                run.pin(ROOT/directory/"manifest.json", pin)
                manifests[name] = loader(ROOT/directory, expected_manifest_sha256=pin)
                run.pin(ROOT/directory/"preparation.json")
                run.work.boundary()
            tutor, tutor_manifest, _ = io._header(ROOT/cached.PARENT, cached.PARENT_SHA256)
            run.pin(ROOT/cached.PARENT/"manifest.json", cached.PARENT_SHA256)
            run.pin(ROOT/cached.PARENT/"preparation.json")
            cycle_path = ROOT/cached.PARENT/"cycle-0.json"
            run.pin(cycle_path, tutor_manifest["artifacts_sha256"]["cycle-0.json"])
            cycle = io.read(cycle_path); plan = cycle["plan"]
            planning.validate_plan(plan)
            if cycle["cycle_id"] != 0 or len(plan["bundles"]) != 216 or plan["config"]["micro_batch_size"] != MICRO:
                raise ValueError("original admitted cycle-zero plan required")
            plan_pin = compiler._hash(plan)
            trained = evidence.transcript_set(run.archive(ROOT/cached.PARENT/"training-transcripts.pt",
                tutor_manifest["artifacts_sha256"]["training-transcripts.pt"]))
            reserved = evidence.transcript_set(run.archive(ROOT/cached.PARENT/"reserved-transcripts.pt",
                tutor_manifest["artifacts_sha256"]["reserved-transcripts.pt"]))
            run.inputs.update(io.HISTORY_PINS)
            run.receipt["archive_attempts"] += 1
            history, history_receipt = io._history(run.work)
            run.receipt["archive_completions"] += 1
            run.receipt["bytes_deserialized"] += io.native(ROOT/io.HISTORY_ROOT/"data/history.pt").stat().st_size
            run.receipt["inherited_history"] = history_receipt
            if len(trained) != 62208 or len(reserved) != 2952 or history & trained or reserved & trained or history & reserved:
                raise ValueError("exact disjoint recorded training/reserved/history required")
            protected, banks = history | reserved, None
            for name, directory in (("cache", current.PARENT), ("earlier", current.EARLIER),
                                    ("previous", current.PREVIOUS), ("current", CURRENT)):
                record = manifests[name]["banks"]
                image = run.archive(ROOT/directory/record["path"], record["sha256"])
                dev = {compiler.transcript_sha256(r) for r in image["dev"]["rows"]}
                if len(dev) != 720 or dev & (protected | trained):
                    raise ValueError("four disjoint prior/current development banks required")
                protected.update(dev)
                if banks is None:
                    banks = {k: deepcopy(image[k]) for k in BANK_COUNTS}
                else:
                    if any(image[k] != banks[k] for k in ("train_fit", "retention")):
                        raise ValueError("actual training-fit or retention bank changed")
                if name == "current": banks["dev"] = deepcopy(image["dev"])
            if len(protected) != PROTECTED_COUNT:
                raise ValueError("exact historical/reserved/four-development union required")
            if (not {compiler.transcript_sha256(r) for r in banks["train_fit"]["rows"]} <= trained
                    or not {compiler.transcript_sha256(r) for r in banks["retention"]["rows"]} <= reserved):
                raise ValueError("fitting and retained evaluation provenance differs")
            if {k: len(v["rows"]) for k,v in banks.items()} != BANK_COUNTS:
                raise ValueError("fixed reused bank sizes required")
            candidates, replay, cycle_transcripts = defaultdict(list), [], set()
            records = manifests["cache"]["training"]
            if len(records) != 648 or any(manifests[k]["training"] != records for k in ("earlier","previous","current")):
                raise ValueError("all overlays must retain the original admitted cache")
            for record in records[:216]:
                run.work.boundary()
                image = run.archive(ROOT/current.PARENT/record["path"], record["sha256"])
                bundle, expected = image["bundle"], image["expected_evidence"]
                canonical = record["canonical_bundle_id"]
                if (set(image) != {"bundle","expected_evidence","expected_evidence_sha256"}
                        or record["cycle_id"] != 0 or image["expected_evidence_sha256"] != evidence.json_digest(expected)
                        or expected != io.expected_bundle(cycle, canonical, record["global_cursor"])
                        or compiler._evidence(bundle) != expected):
                    raise ValueError("cached rows differ from original admitted evidence")
                slot = plan["bundles"][str(canonical)]
                for index in range(MICRO//2):
                    descriptor = dict(plan_sha256=plan_pin, bundle_id=canonical, pair_index=index, pair_sha256={})
                    signatures = []
                    for family in compiler.FAMILIES:
                        pair = bundle["families"][family][2*index:2*index+2]
                        if len(pair) != 2 or any(r["depth"] != slot["depth"] or len(r["turns"]) != slot["turns"] for r in pair):
                            raise ValueError("complete declared family cell required")
                        descriptor["pair_sha256"][family] = pair[0]["recipe"]["base_pair_sha256"]
                        if pair[0]["recipe"] != pair[1]["recipe"]:
                            raise ValueError("cached counterfactual parent identity differs")
                        anchor = next(q for q in pair[0]["queries"] if q["query_id"] == pair[0]["anchor"]["query_id"])
                        ops = [v["event"]["op"] for v in anchor["version_signature"]["versions"]]
                        signatures.append(ops)
                        cycle_transcripts.update(compiler.transcript_sha256(r) for r in pair)
                    if signatures[0] != signatures[1] or signatures[0] != signatures[2]:
                        raise ValueError("one shared all-family parent procedure required")
                    candidates[slot["depth"],slot["turns"]].append(dict(descriptor=descriptor, copies=signatures[0].count("copy")))
                    replay.append(descriptor)
                if len(replay) % (48*16) == 0:
                    run.work.event(dict(event="inventory_progress", cached_bundles=len(replay)//16))
            if (len(cycle_transcripts) != 20736 or not cycle_transcripts <= trained or cycle_transcripts & protected
                    or evidence.json_digest(sorted(cycle_transcripts)) != cycle["training_manifest"]["transcript_sha256"]):
                raise ValueError("exact original first-cycle admitted transcripts required")
            pools, pool_report = _pools(candidates)
            inventory = dict(schema=compiler.INVENTORY_SCHEMA, plans={plan_pin: plan}, motifs=pools, replay=replay)
            protected_pin = compiler._hash(sorted(protected))
            teaching, withdrawal = _contract(inventory, protected_pin), _contract(inventory, protected_pin, withdrawal=True)
            run.save("inventory.json", inventory); run.save("protected.pt", sorted(protected))
            run.save("teaching-contract.json", teaching); run.save("withdrawal-contract.json", withdrawal)
            run.save("pool-report.json", pool_report)
            bank_pin = run.save("banks.pt", banks)
            manifest = run.complete_manifest(dict(schema=INVENTORY_SCHEMA,
                inventory_sha256=compiler.inventory_sha256(inventory), protected_sha256=protected_pin,
                protected_count=len(protected), teaching_contract_sha256=compiler.contract_sha256(teaching),
                withdrawal_contract_sha256=compiler.contract_sha256(withdrawal), banks=dict(path="banks.pt",sha256=bank_pin),
                teaching_contract=dict(path="teaching-contract.json",sha256=run.artifacts["teaching-contract.json"],
                    identity_sha256=compiler.contract_sha256(teaching)),
                withdrawal_contract=dict(path="withdrawal-contract.json",sha256=run.artifacts["withdrawal-contract.json"],
                    identity_sha256=compiler.contract_sha256(withdrawal)),
                bank_counts={k: len(v["rows"]) for k,v in banks.items()}, admitted_cycle_zero_descriptors=len(replay),
                canonical_regenerations=0, newly_generated_rows=0,
                protection_scope="Recorded historical/reserved/four-development union; admitted training is allowed replay."))
        return manifest
    except BaseException as exc:
        error = exc; raise
    finally:
        run.finish(error)


def compile_data(output, *, inventory_directory, inventory_manifest_sha256,
                 teacher_decision, parent, max_seconds=1800, progress=None):
    """Compile only the authenticated persisted decision; never invoke a teacher."""
    run = _Attempt(output, "compilation", max_seconds, progress, include_author=True)
    error = None
    try:
        with io._cpu_guard(run.work):
            from experiments import verified_tutor_author as author
            directory = Path(inventory_directory).resolve()
            run.pin(directory/"manifest.json", inventory_manifest_sha256)
            inventory_manifest, prior = io.read(directory/"manifest.json"), io.read(directory/"preparation.json")
            run.pin(directory/"preparation.json")
            if (inventory_manifest["schema"] != INVENTORY_SCHEMA or prior["status"] != "completed"
                    or prior["manifest_sha256"] != inventory_manifest_sha256
                    or inventory_manifest["source_sha256"] != source_hashes()):
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
            manifest = run.complete_manifest(dict(schema=SCHEMA, start_cursor=START, micro_batch_size=MICRO,
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
    parser.add_argument("mode", choices=("inventory","compile")); parser.add_argument("output")
    parser.add_argument("--inventory-directory"); parser.add_argument("--inventory-manifest-sha256")
    parser.add_argument("--teacher-decision-json"); parser.add_argument("--parent-json")
    args = parser.parse_args()
    progress = lambda value: print(io.encoded(value).decode().strip(),flush=True)
    if args.mode == "inventory": prepare_inventory(args.output, progress=progress)
    else: compile_data(args.output, inventory_directory=args.inventory_directory,
        inventory_manifest_sha256=args.inventory_manifest_sha256,
        teacher_decision=io.read(args.teacher_decision_json), parent=io.read(args.parent_json), progress=progress)
    print(io.encoded(dict(manifest_sha256=io.digest(Path(args.output)/"manifest.json"))).decode().strip(),flush=True)
