"""Fresh finite-grammar transfer diagnostics, with paired causal layout views.

The audit partition means withheld from the current train-partition curriculum,
not never previously examined by a developer. Exact text exclusion is limited to
the authenticated recorded inventory. Reused fitting/retention rows are labeled.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import time
import traceback

from experiments import foundation_curriculum as foundation
from experiments import foundation_layout_curriculum as layout
from experiments import foundation_tutor_loop_data as io

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-shared-acquisition-transfer-data-v1"
SEED, PAIRS, MAX_SECONDS, MAX_GENERATOR_CALLS, MAX_ATTEMPTS = 852902001, 6, 900, 12000, 64
INVENTORY = "runs/verified-tutor-inventory-local/attempt-001"
INVENTORY_SHA = "6777954dd97141a09fd2a2fa5e664dbdbbf092efe3d3e587fbf91b109be6b088"
COMPILED = "runs/verified-tutor-data-local/attempt-002"
COMPILED_SHA = "a35f6b3ab0c2a104c9038e28e4cd58178c59e14343e341565415f24054170419"
COMPILED_PREPARATION_SHA = "41e498b71730bace8a475df819dff550a6b734193f9e94854ce58855d0ccf6fe"
TRAINED = "runs/foundation-tutor-loop-data-local/attempt-001/training-transcripts.pt"
TRAINED_SHA = "f28c4ac79999404ebe19493b3be364f8929adda95d23fcbb01c39b345ef899ee"
DECLARATION = "docs/SHARED_ACQUISITION_TRANSFER_DATA.md"
BANK_COUNTS = dict(dev=648, transfer_original=432, transfer_varied=432, train_fit=108, retention=720)


def source_hashes():
    names = set(layout.source_hashes()) | {"experiments/foundation_tutor_loop_data.py",
        "experiments/shared_acquisition_transfer_data.py", DECLARATION}
    return {name: io.digest(ROOT / name) for name in sorted(names)}


def transcript(row):
    return layout._hash([turn["text"] for turn in row["turns"]])


def _seed(seed, *parts):
    return int(layout._hash([SCHEMA, seed, *parts]), 16) % (2**63)


@contextmanager
def generation_accounting(work, maximum=MAX_GENERATOR_CALLS):
    original = foundation.generate_pair
    def generate(*args, **kwargs):
        work.boundary()
        if work.counts["canonical_generate_attempts"] >= maximum:
            raise ValueError("bounded canonical generation allowance exhausted")
        work.counts["canonical_generate_attempts"] += 1
        pair = original(*args, **kwargs)
        work.counts["canonical_generate_completions"] += 1
        work.counts["canonical_rows_returned"] += len(pair)
        work.boundary()
        return pair
    foundation.generate_pair = generate
    try:
        yield
    finally:
        foundation.generate_pair = original


def paired_views(pair, *, seed, transfer, work):
    """Keep original and varied parents/queries coupled; no inference inputs added."""
    expected = "audit" if transfer else "dev"
    if (any(row["split"] != expected for row in pair)
            or transfer and (pair[0]["depth"] < 2 or pair[0]["structure_partition"] != "audit")
            or not transfer and pair[0]["structure_partition"] != "train"):
        raise ValueError("declared panel partition differs")
    original = layout.materialize_pair(pair, layout="original", seed=seed, work=work.layout)
    layout.validate_pair(original, work=work.layout)
    views = {"transfer_original" if transfer else "dev": original}
    if transfer:
        varied = layout.materialize_pair(pair, layout="varied", seed=seed, work=work.layout)
        layout.validate_pair(varied, work=work.layout)
        for a, b in zip(original, varied):
            if (a["recipe"]["base_ids"] != b["recipe"]["base_ids"] or a["anchor"]["query_id"] != b["anchor"]["query_id"]
                    or a["anchor"]["version_signature_sha256"] != b["anchor"]["version_signature_sha256"]):
                raise ValueError("paired layout changed the known causal anchor")
        views["transfer_varied"] = varied
    return views


def admit_views(views, *, blocked, owners):
    """Atomic transcript admission; only the same paired-view member may overlap."""
    proposed = {}
    for rows in views.values():
        if len(rows) != 2 or [r["variant"] for r in rows] != [0, 1]:
            raise ValueError("two complete ordered members required")
        for row in rows:
            digest = transcript(row)
            owner = (row["recipe"]["base_pair_sha256"], row["variant"])
            if digest in blocked:
                return False, "excluded_transcript"
            if digest in owners or digest in proposed and proposed[digest] != owner:
                return False, "different_pair_transcript"
            proposed[digest] = owner
    owners.update(proposed)
    return True, None


def build_cell(family, depth, turns, *, transfer, seed, pairs, blocked, owners, work, rejections):
    """Small independently callable construction boundary, also used by tests."""
    panel = "transfer" if transfer else "dev"
    result = {"transfer_original": [], "transfer_varied": []} if transfer else {"dev": []}
    provenance = []
    for slot in range(pairs):
        for attempt in range(MAX_ATTEMPTS):
            work.boundary()
            key = (panel, depth, turns, slot, attempt)
            try:
                pair = foundation.generate_pair(family, _seed(seed, *key, "procedure"), depth=depth, turns=turns,
                    split="audit" if transfer else "dev", structure_split="audit" if transfer else "shared" if depth < 2 else "train",
                    naming_seed=_seed(seed, *key, "names"), value_seed=_seed(seed, *key, "values"))
            except ValueError as error:
                if str(error).startswith("no admitted foundation procedure within"):
                    rejections["unavailable_procedure"] += 1
                    continue
                raise
            # Family and labels are deliberately absent from the layout seed.
            placement_seed = _seed(seed, panel, depth, turns, slot, "layout")
            views = paired_views(pair, seed=placement_seed, transfer=transfer, work=work)
            accepted, reason = admit_views(views, blocked=blocked, owners=owners)
            if not accepted:
                rejections[reason] += 1
                continue
            for name, rows in views.items(): result[name].extend(rows)
            provenance.append(dict(family=family, depth=depth, turns=turns, slot=slot, attempt=attempt,
                canonical_pair_sha256=layout._hash(pair), recipe=deepcopy(pair[0]["recipe"]), layout_seed=placement_seed,
                structure_id=pair[0]["structure_id"], structure_partition=pair[0]["structure_partition"]))
            break
        else:
            raise ValueError(f"bounded transfer-bank cell exhausted: {panel}/{family}/d{depth}/t{turns}/pair{slot}")
    return result, provenance


def _inventory(bank, provenance):
    rows = bank["rows"]
    return dict(role=bank["role"], episodes=len(rows), pairs=len(rows)//2, rows_sha256=layout._hash(rows),
        provenance=provenance, cells=dict(sorted(Counter(f"{r['family']}/d{r['depth']}/t{len(r['turns'])}" for r in rows).items())),
        anchor_partitions=dict(sorted(Counter(r["structure_partition"] for r in rows).items())))


def prepare(output, *, seed=SEED, pairs_per_cell=PAIRS, max_seconds=MAX_SECONDS, progress=None):
    """One declared CPU-only invocation. No teacher, learner, optimizer or replay."""
    if (seed, pairs_per_cell, max_seconds) != (SEED, PAIRS, MAX_SECONDS):
        raise ValueError("fixed prospective preparation seed, counts and allowance required")
    started, cpu = time.monotonic(), time.process_time()
    output = Path(output).resolve(); io.native(output).mkdir(parents=True, exist_ok=False)
    work = io.WorkLedger(deadline=started+max_seconds, progress=progress, layout=layout.WorkLedger())
    receipt = dict(schema=SCHEMA, status="running", neural_work=0, teacher_calls=0, optimizer_updates=0,
        max_seconds=max_seconds, max_generator_calls=MAX_GENERATOR_CALLS, max_attempts_per_pair=MAX_ATTEMPTS)
    pins, artifacts, sources = {}, {}, source_hashes()
    receipt["source_sha256"] = sources

    def pin(path, expected):
        actual = io.digest(path)
        if actual != expected: raise ValueError("authenticated input changed: " + str(path))
        pins[Path(path).relative_to(ROOT).as_posix()] = actual

    def archive(path, expected):
        work.boundary(); pin(path, expected)
        value = io._load_pt(path, work); work.boundary()
        return value

    def save(name, value):
        work.boundary(); artifacts[name] = io.write(output/name, value)
        return artifacts[name]

    try:
        io.write(output/"started.json", receipt)
        with io._cpu_guard(work):
            invdir, compiled = ROOT/INVENTORY, ROOT/COMPILED
            pin(invdir/"manifest.json", INVENTORY_SHA); inv = io.read(invdir/"manifest.json")
            pin(compiled/"manifest.json", COMPILED_SHA); current = io.read(compiled/"manifest.json")
            pin(compiled/"preparation.json", COMPILED_PREPARATION_SHA); preparation = io.read(compiled/"preparation.json")
            if preparation["status"] != "completed" or preparation["manifest_sha256"] != COMPILED_SHA:
                raise ValueError("completed compiled curriculum required")
            for name, expected in {**inv["source_sha256"], **current["source_sha256"]}.items(): pin(ROOT/name, expected)
            protected_list = archive(invdir/"protected.pt", inv["artifact_sha256"]["protected.pt"])
            protected = set(protected_list)
            if (protected_list != sorted(protected) or len(protected) != 1641672
                    or layout._hash(protected_list) != inv["protected_sha256"]):
                raise ValueError("exact recorded historical/reserved/development protection required")
            trained_list = archive(ROOT/TRAINED, TRAINED_SHA); trained = set(trained_list)
            if trained_list != sorted(trained) or len(trained) != 62208 or trained & protected:
                raise ValueError("exact distinct original admitted training list required")
            blocked = protected | trained
            old = archive(compiled/current["banks"]["path"], current["banks"]["sha256"])
            if not {transcript(r) for r in old["train_fit"]["rows"]} <= trained:
                raise ValueError("reused actual fitting rows differ")
            if not {transcript(r) for r in old["retention"]["rows"]} <= protected:
                raise ValueError("reused retention provenance differs")
            archived, compiled_transcripts = set(), set()
            for phase in ("teaching", "withdrawal"):
                for arm in ("procedural", "tutor"):
                    for record in current["phases"][phase][arm]:
                        if record["path"] in archived: continue
                        archived.add(record["path"])
                        lesson = archive(compiled/record["path"], record["sha256"])
                        if layout._hash(lesson["expected_evidence"]) != lesson["expected_evidence_sha256"]:
                            raise ValueError("compiled evidence digest differs")
                        for family, rows in lesson["bundle"]["families"].items():
                            if any(r["family"] != family or r["split"] != "train" or r["structure_partition"] != "train" for r in rows):
                                raise ValueError("all consumed training structures must be train-admitted")
                            compiled_transcripts.update(transcript(row) for row in rows)
                        if len(archived) % 48 == 0: work.event(dict(event="exclusion_archives", completed=len(archived)))
            if len(archived) != 324 or compiled_transcripts & protected:
                raise ValueError("exact completed training streams required")
            blocked.update(compiled_transcripts)
            receipt["exclusion_inventory"] = dict(protected=len(protected), original_training=len(trained),
                compiled_distinct_transcripts=len(compiled_transcripts), compiled_new_transcripts=len(compiled_transcripts-trained),
                compiled_archives=len(archived), total=len(blocked), sha256=layout._hash(sorted(blocked)))
            del protected_list, trained_list, protected, trained, compiled_transcripts, lesson
            banks = {name: dict(role="audit" if name.startswith("transfer") else "dev", rows=[])
                     for name in ("dev", "transfer_original", "transfer_varied")}
            banks.update(train_fit=deepcopy(old["train_fit"]), retention=deepcopy(old["retention"]))
            del old
            owners, provenance, rejections = {}, [], Counter()
            with generation_accounting(work):
                for transfer in (False, True):
                    for family in foundation.FAMILIES:
                        for depth in range(2 if transfer else 0, 6):
                            for turns in foundation.TURN_BUCKETS:
                                views, evidence = build_cell(family, depth, turns, transfer=transfer, seed=seed,
                                    pairs=pairs_per_cell, blocked=blocked, owners=owners, work=work, rejections=rejections)
                                for name, rows in views.items(): banks[name]["rows"].extend(rows)
                                provenance.extend(evidence)
                        work.event(dict(event="bank_family", panel="transfer" if transfer else "dev", family=family,
                                        canonical_calls=work.counts["canonical_generate_completions"]))
            if {k: len(v["rows"]) for k,v in banks.items()} != BANK_COUNTS:
                raise ValueError("declared2340 episode bank census differs")
            varied = banks["transfer_varied"]["rows"][::2]
            placement = dict(parent_pairs=len(varied), changed_pairs=sum(r["layout_summary"]["changed"] for r in varied),
                anchor_nonterminal_pairs=sum(r["layout_summary"]["anchor_nonterminal"] for r in varied),
                terminal_kinds=dict(sorted(Counter(r["layout_summary"]["terminal_kind"] for r in varied).items())),
                feasible_terminal_sets=dict(sorted(Counter("/".join(r["recipe"]["feasible_terminal_kinds"]) for r in varied).items())))
            bank_pin = save("banks.pt", banks)
            save("provenance.json", provenance); save("fresh-transcripts.json", sorted(owners))
            metadata = {"dev": "Fresh exact text, train-partition structures; shared depth0/1 primitives.",
                "transfer_original": "Fresh exact text, audit-partition depth2..5 structures, original layout; repeatedly observed diagnostic.",
                "transfer_varied": "Paired audit parents with independently verified causal query relocation; repeatedly observed diagnostic.",
                "train_fit": "Reused actual original training, deliberate training overlap.",
                "retention": "Reused previously observed development, deliberate protected-bank overlap."}
            if source_hashes() != sources: raise ValueError("preparation source changed")
            snapshots = {}
            for index, (name, expected) in enumerate(sources.items()):
                raw = io.native(ROOT/name).read_bytes()
                if io.digest(ROOT/name) != expected: raise ValueError("source changed before publication")
                local = f"sources/{index:02d}.bin"; path = io.native(output/local); path.parent.mkdir(exist_ok=True)
                with path.open("xb") as stream: stream.write(raw)
                snapshots[name] = local
            manifest = dict(schema=SCHEMA, seed=seed, pairs_per_cell=pairs_per_cell,
                banks=dict(path="banks.pt", sha256=bank_pin), bank_inventory={k:_inventory(v,metadata[k]) for k,v in banks.items()},
                exclusion_inventory=receipt["exclusion_inventory"], fresh_unique_transcripts=len(owners),
                paired_transfer_placement=placement, rejections=dict(rejections),
                source_sha256=sources, source_snapshots=snapshots, input_sha256=pins, artifacts_sha256=artifacts,
                scope="Finite syntactic query-ancestry and causal query-layout transfer, no new grammar/algorithms. Fresh bank exact transcript exclusions cover the explicit authenticated inventory. Paired views may share identical text only for the same parent/member. Repeated diagnostic transfer is not pristine final audit.")
            receipt["manifest_sha256"] = io.write(output/"manifest.json", manifest)
            receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(error=repr(error), traceback=traceback.format_exc()); raise
    finally:
        receipt.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
                       work=work.report(), source_sha256=sources, input_sha256=pins, artifacts_sha256=artifacts)
        io.write(output/"preparation.json", receipt)
    return receipt


def load_manifest(directory, *, expected_manifest_sha256):
    directory = Path(directory).resolve()
    if io.digest(directory/"manifest.json") != expected_manifest_sha256: raise ValueError("caller manifest pin differs")
    manifest, receipt = io.read(directory/"manifest.json"), io.read(directory/"preparation.json")
    if (manifest["schema"] != SCHEMA or receipt["status"] != "completed" or receipt["manifest_sha256"] != expected_manifest_sha256
            or manifest["source_sha256"] != source_hashes()
            or {k:v["episodes"] for k,v in manifest["bank_inventory"].items()} != BANK_COUNTS):
        raise ValueError("completed source-pinned transfer data required")
    for name, expected in manifest["artifacts_sha256"].items():
        if io.digest(directory/name) != expected: raise ValueError("prepared artifact differs")
    return manifest


def load_banks(directory, manifest):
    path = Path(directory)/manifest["banks"]["path"]
    if io.digest(path) != manifest["banks"]["sha256"]: raise ValueError("bank archive differs")
    return io._load_pt(path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("output")
    args = parser.parse_args()
    print(io.encoded(prepare(args.output, progress=lambda event: print(io.encoded(event).decode().strip(), flush=True))).decode(), flush=True)
