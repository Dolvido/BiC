"""One CPU packaging pass over admitted lessons for the tied-reader comparison."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
import traceback

from experiments import foundation_tutor_loop_data as pilot


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-recurrent-read-data-v1"
PARENT = "runs/foundation-tutor-loop-data-local/attempt-001"
PARENT_SHA256 = "883321129b481ff621c0538d5b8fe8f857d5349da3b91b1a358358b2f3382cba"
DEV_SEED, PAIRS_PER_CELL, UPDATES, MAX_SECONDS = 852202001, 4, 648, 1800


def source_hashes():
    return {**pilot.source_hashes(), "experiments/recurrent_read_data.py": pilot.digest(ROOT / "experiments/recurrent_read_data.py")}


def _authenticated(parent, manifest, name, work):
    expected = manifest["artifacts_sha256"][name]
    if pilot.digest(parent / name) != expected:
        raise ValueError("parent artifact changed: " + name)
    return pilot._load_pt(parent / name, work) if name.endswith(".pt") else pilot.read(parent / name)


def prepare(directory, *, max_seconds=MAX_SECONDS, progress=None):
    """Stream each admitted curriculum bundle once; no model or optimizer work."""
    if max_seconds != MAX_SECONDS or progress is not None and not callable(progress):
        raise ValueError("fixed 1800-second packaging bound and optional callback required")
    started, cpu = time.monotonic(), time.process_time()
    output = pilot.native(directory)
    output.mkdir(parents=True, exist_ok=False)
    work = pilot.WorkLedger(deadline=started + max_seconds)
    report = dict(schema=SCHEMA, status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                  maximum_seconds=max_seconds, neural_forwards=0, optimizer_updates=0, teacher_calls=0)
    training, artifacts, sources, selected = [], {}, {}, {}
    journal = (output / "events.jsonl").open("xb")

    def event(value):
        journal.write(pilot.encoded(dict(elapsed_seconds=time.monotonic() - started, **value)))
        journal.flush()
        if progress is not None:
            progress(value)

    def save(name, value):
        work.boundary()
        artifacts[name] = pilot.write(output / name, value)
        return artifacts[name]

    try:
        with pilot._cpu_guard(work):
            from experiments import foundation_banks as banks, foundation_evidence as evidence
            from experiments import foundation_layout_curriculum as layout
            work = pilot._work(work)
            sources = source_hashes()
            save("invocation.json", dict(schema=SCHEMA, parent_manifest=PARENT_SHA256,
                                        dev_seed=DEV_SEED, updates=UPDATES, source_sha256=sources))
            parent, inherited, parent_receipt = pilot._header(ROOT / PARENT, PARENT_SHA256)
            cycles = [_authenticated(parent, inherited, f"cycle-{cycle}.json", work) for cycle in range(3)]
            trained_values = _authenticated(parent, inherited, "training-transcripts.pt", work)
            reserved_values = _authenticated(parent, inherited, "reserved-transcripts.pt", work)
            trained, reserved = evidence.transcript_set(trained_values), evidence.transcript_set(reserved_values)
            history, historical = pilot._history(work)
            if (len(trained) != 62208 or len(reserved) != 2952 or len(history) != 1635840
                    or trained & reserved or trained & history or reserved & history):
                raise ValueError("authenticated transcript inventories have unexpected overlap or size")
            blocked = history | trained | reserved
            exclusion = dict(history_count=len(history), training_count=len(trained), reserved_count=len(reserved),
                             union_count=len(blocked), union_sha256=evidence.json_digest(sorted(blocked)),
                             training_sha256=evidence.json_digest(sorted(trained)),
                             reserved_sha256=evidence.json_digest(sorted(reserved)),
                             historical_receipt=historical,
                             scope="Exact transcript exclusion from this authenticated inventory only; semantic equivalences and vocabulary may recur.")
            save("exclusions.json", exclusion)
            event(dict(event="authenticated", exclusion_count=len(blocked)))
            cycle_transcripts, all_seen = [], set()
            for cycle, record in enumerate(cycles):
                plan = record["plan"]
                schedule = plan["schedules"]["curriculum"]
                if record["cycle_id"] != cycle or sorted(schedule) != list(range(216)):
                    raise ValueError("complete admitted curriculum schedule required")
                seen = set()
                for bundle_id in schedule:
                    cursor = len(training)
                    bundle = pilot.materialize_bundle(plan, bundle_id, global_cursor=cursor, work=work)
                    expected = pilot.expected_bundle(record, bundle_id, cursor)
                    if work.last_bundle_evidence != expected:
                        raise ValueError(f"materialized evidence changed at cursor {cursor}")
                    for family, rows in bundle["families"].items():
                        for row in rows:
                            digest = banks.transcript_digest(row)
                            if digest not in trained or digest in all_seen:
                                raise ValueError(f"training transcript identity or uniqueness changed at {cursor}")
                            all_seen.add(digest); seen.add(digest)
                        first = rows[0]
                        key = f"{family}/d{first['depth']}/t{len(first['turns'])}"
                        if key not in selected:
                            selected[key] = dict(rows=deepcopy(rows[:2]), cycle_id=cycle,
                                                 canonical_bundle_id=bundle_id, global_cursor=cursor, pair_index=0)
                    path = f"training/{cursor:04d}.pt"
                    archive = dict(bundle=bundle, expected_evidence=expected,
                                   expected_evidence_sha256=evidence.json_digest(expected))
                    training.append(dict(path=path, sha256=save(path, archive), cycle_id=cycle,
                                         canonical_bundle_id=bundle_id, global_cursor=cursor))
                    if len(training) % 48 == 0:
                        event(dict(event="packaged", bundles=len(training), episodes=len(all_seen)))
                manifest = record["training_manifest"]
                if (len(seen) != manifest["unique_transcripts"]
                        or evidence.json_digest(sorted(seen)) != manifest["transcript_sha256"]):
                    raise ValueError("materialized cycle transcript inventory changed")
                cycle_transcripts.append(sorted(seen))
            expected_cells = {f"{family}/d{depth}/t{turns}" for family in ("color", "count", "switch")
                              for depth in range(6) for turns in (8, 10, 12)}
            if len(training) != UPDATES or all_seen != trained or set(selected) != expected_cells:
                raise ValueError("complete shared training inventory and balanced fit selection required")
            event(dict(event="fresh_development_start", seed=DEV_SEED))
            with pilot._tracking(work):
                canonical, admission = banks.build_evaluation(cycles[0]["plan"], cycles[0]["training_manifest"],
                    training_transcripts=cycle_transcripts[0], role="dev", seed=DEV_SEED,
                    pairs_per_cell=PAIRS_PER_CELL, excluded_transcripts=sorted(blocked))
                development, fresh = [], set()
                for cell, rows in sorted(canonical.items()):
                    for offset in range(0, len(rows), 2):
                        work.boundary()
                        pair = layout.materialize_pair(rows[offset:offset + 2], layout="original", seed=0, work=work.layout)
                        layout.validate_pair(pair, work=work.layout)
                        identities = {banks.transcript_digest(row) for row in pair}
                        if len(identities) != 2 or identities & blocked or identities & fresh:
                            raise ValueError("fresh development transcript admission changed")
                        fresh.update(identities); development.extend(pair)
                        work.counts["adapted_evaluation_pairs"] += 1
            if len(development) != 720 or fresh != set(admission["transcript_sha256"]):
                raise ValueError("fixed fresh development coverage changed")
            retention = _authenticated(parent, inherited, "retention.pt", work)
            retained = [row for cell, rows in sorted(retention["rows"].items()) for row in rows]
            retained_hashes = {banks.transcript_digest(row) for row in retained}
            if retention["role"] != "dev" or len(retained) != 720 or not retained_hashes <= reserved:
                raise ValueError("reused development inventory changed")
            fit = [row for cell in sorted(selected) for row in selected[cell]["rows"]]
            if len(fit) != 108 or not {banks.transcript_digest(row) for row in fit} <= trained:
                raise ValueError("actual-training fit sample changed")
            bank_image = dict(dev=dict(role="dev", rows=development),
                              train_fit=dict(role="train_fit", rows=fit),
                              retention=dict(role="dev", rows=deepcopy(retained)))
            bank_sha = save("banks.pt", bank_image)
            save("development-canonical.pt", canonical)
            save("development-admission.json", admission)
            inventory = {name: dict(role=bank["role"], episodes=len(bank["rows"]),
                                   pairs=len(bank["rows"]) // 2, rows_sha256=evidence.json_digest(bank["rows"]),
                                   provenance={"dev": "fresh-development", "train_fit": "actual-training",
                                               "retention": "reused-development"}[name])
                         for name, bank in bank_image.items()}
            inventory["train_fit"]["selections"] = {cell: {key: value for key, value in record.items() if key != "rows"}
                                                      for cell, record in sorted(selected.items())}
            snapshots = {}
            for index, (name, expected) in enumerate(sorted(sources.items())):
                work.boundary()
                content = pilot.native(ROOT / name).read_bytes()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError("source changed during data preparation: " + name)
                local = f"sources/{index:03d}.bin"
                target = output / local
                target.parent.mkdir(exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(content)
                artifacts[local] = pilot.digest(target); snapshots[name] = local
            if source_hashes() != sources:
                raise ValueError("data source closure changed")
            manifest = dict(schema=SCHEMA, updates=UPDATES, layout="original", micro_batch_size=32,
                schedule="three admitted curriculum cycles in canonical curriculum order", training=training,
                banks=dict(path="banks.pt", sha256=bank_sha), bank_inventory=inventory,
                bank_admission=dict(path="development-admission.json", sha256=artifacts["development-admission.json"]),
                exclusion_inventory=exclusion, parent_manifest=dict(path=PARENT + "/manifest.json", sha256=PARENT_SHA256),
                parent_preparation=dict(path=PARENT + "/preparation.json", sha256=pilot.digest(parent / "preparation.json"),
                    wall_seconds=parent_receipt["wall_seconds"], cpu_seconds=parent_receipt["cpu_seconds"]),
                source_sha256=sources, source_snapshots=snapshots, artifacts_sha256=artifacts,
                training_episodes=62208, no_neural_training_or_inference=True,
                expected_evidence_digest="foundation_evidence.json_digest: sorted compact JSON, default ensure_ascii=True, no newline")
            report["manifest_sha256"] = pilot.write(output / "manifest.json", manifest)
            report["status"] = "completed"
            event(dict(event="completed", bundles=len(training), bank_episodes={name: value["episodes"] for name, value in inventory.items()}))
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        journal.close()
        report.update(work=work.report(), packaged_bundles=len(training), selected_fit_pairs=len(selected),
                      source_sha256=sources, artifacts_sha256=artifacts, journal_sha256=pilot.digest(output / "events.jsonl"),
                      wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        pilot.write(output / "preparation.json", report)
    return manifest


def load_manifest(directory, *, expected_manifest_sha256):
    output = pilot.native(directory)
    if pilot.digest(output / "manifest.json") != expected_manifest_sha256:
        raise ValueError("explicit data manifest hash differs")
    manifest, receipt = pilot.read(output / "manifest.json"), pilot.read(output / "preparation.json")
    if (manifest["schema"] != SCHEMA or receipt["status"] != "completed"
            or receipt["manifest_sha256"] != expected_manifest_sha256
            or manifest["source_sha256"] != source_hashes()):
        raise ValueError("completed source-pinned recurrent-read data required")
    return manifest


def load_bundle(directory, record):
    from experiments.foundation_evidence import json_digest
    path = pilot.native(directory) / record["path"]
    if pilot.digest(path) != record["sha256"]:
        raise ValueError("frozen bundle bytes changed")
    image = pilot._load_pt(path)
    if (set(image) != {"bundle", "expected_evidence", "expected_evidence_sha256"}
            or image["expected_evidence_sha256"] != json_digest(image["expected_evidence"])
            or image["bundle"]["bundle_id"] != record["global_cursor"]
            or image["expected_evidence"]["bundle_id"] != record["global_cursor"]):
        raise ValueError("bundle image evidence identity differs")
    return image


def load_banks(directory, manifest):
    path = pilot.native(directory) / manifest["banks"]["path"]
    if pilot.digest(path) != manifest["banks"]["sha256"]:
        raise ValueError("frozen bank bytes changed")
    return pilot._load_pt(path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    result = prepare(args.directory, progress=lambda value: print(pilot.encoded(value).decode().strip(), flush=True))
    print(pilot.encoded(dict(manifest_sha256=pilot.digest(pilot.native(args.directory) / "manifest.json"),
                             bundles=len(result["training"]))).decode().strip(), flush=True)
