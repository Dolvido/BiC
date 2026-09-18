"""Fresh evaluation overlay for exact shared-state continuation; no new training."""
from copy import deepcopy
from pathlib import Path
import time
import traceback

from experiments import foundation_tutor_loop_data as io
from experiments import recurrent_read_data as parent_data
from experiments import entity_retrieval_data as previous_data
from experiments import shared_state_data as earlier_data

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-shared-state-continuation-data-v1"
PARENT = "runs/recurrent-read-data-local/attempt-001"
PARENT_SHA = "c420a43a3a45f63347288dda7db48c71b3e7fa183b4a1681c2ec89eabeb08654"
PREVIOUS = "runs/entity-retrieval-data-local/attempt-001"
PREVIOUS_SHA = "7c0d1f208f59a183cf5c052c991c842c2358107a3c40c84f4fc36ba42fa63612"
EARLIER = "runs/shared-state-data-local/attempt-001"
EARLIER_SHA = "266dfe643074b8d66b65b1e506f6fc726fa6cb75f2a532b04232ec87bf30eb75"
DECLARATION = "docs/SHARED_STATE_CONTINUATION_DATA.md"
DEV_SEED, MAX_SECONDS = 852702001, 600
BANK_COUNTS = {"dev": 720, "previous_dev": 720, "train_fit": 108, "retention": 720}


def source_hashes():
    return {**previous_data.source_hashes(),
            "experiments/shared_state_continuation_data.py": io.digest(Path(__file__))}


def prepare(directory):
    """One exclusive CPU-only preparation; failures remain terminal artifacts."""
    started, cpu = time.monotonic(), time.process_time()
    output = io.native(directory)
    output.mkdir(parents=True, exist_ok=False)
    work = io.WorkLedger(deadline=started + MAX_SECONDS)
    receipt = dict(schema=SCHEMA, status="running", neural_work=0, teacher_calls=0,
                   archive_attempts={"direct": 0, "delegated": 0},
                   archive_completions={"direct": 0, "delegated": 0})
    artifacts = {}

    def archive(kind, call):
        work.boundary()
        receipt["archive_attempts"][kind] += 1
        result = call()
        receipt["archive_completions"][kind] += 1
        return result

    def save(name, value):
        artifacts[name] = io.write(output / name, value)
        return artifacts[name]

    def existing_banks(directory, manifest):
        path = ROOT / directory / manifest["banks"]["path"]
        if io.digest(path) != manifest["banks"]["sha256"]:
            raise ValueError("inherited bank archive changed")
        return archive("direct", lambda: io._load_pt(path, work))

    try:
        with io._cpu_guard(work):
            from experiments import foundation_banks as banks, foundation_evidence as evidence
            sources, declaration_sha = source_hashes(), io.digest(ROOT / DECLARATION)
            receipt.update(source_sha256=sources, declaration_sha256=declaration_sha)
            io.write(output / "started.json", receipt)
            parent = parent_data.load_manifest(ROOT / PARENT, expected_manifest_sha256=PARENT_SHA)
            previous = previous_data.load_manifest(ROOT / PREVIOUS, expected_manifest_sha256=PREVIOUS_SHA)
            earlier = earlier_data.load_manifest(ROOT / EARLIER, expected_manifest_sha256=EARLIER_SHA)
            old_banks = archive("delegated", lambda: parent_data.load_banks(ROOT / PARENT, parent))
            previous_banks = existing_banks(PREVIOUS, previous)
            earlier_banks = existing_banks(EARLIER, earlier)
            tutor, tutor_manifest, _ = io._header(ROOT / parent_data.PARENT, parent_data.PARENT_SHA256)

            def inherited(name):
                if io.digest(tutor / name) != tutor_manifest["artifacts_sha256"][name]:
                    raise ValueError("inherited transcript evidence changed")
                return (archive("direct", lambda: io._load_pt(tutor / name, work))
                        if name.endswith(".pt") else io.read(tutor / name))

            trained = evidence.transcript_set(inherited("training-transcripts.pt"))
            reserved = evidence.transcript_set(inherited("reserved-transcripts.pt"))
            history, _ = archive("direct", lambda: io._history(work))
            prior_dev = {banks.transcript_digest(row) for row in old_banks["dev"]["rows"]}
            earlier_dev = {banks.transcript_digest(row) for row in earlier_banks["dev"]["rows"]}
            previous_dev = {banks.transcript_digest(row) for row in previous_banks["dev"]["rows"]}
            prior_blocked = history | trained | reserved | prior_dev | earlier_dev
            if (len(prior_dev) != 720 or len(earlier_dev) != 720
                    or len(prior_blocked) != 1702440
                    or evidence.json_digest(sorted(prior_blocked)) != previous["exclusion_sha256"]
                    or len(previous_dev) != 720 or previous_dev & prior_blocked):
                raise ValueError("exact prior exclusion union plus720 new transcripts required")
            blocked = prior_blocked | previous_dev
            if len(blocked) != 1703160:
                raise ValueError("complete recorded-study exclusion inventory required")
            for name in ("train_fit", "retention"):
                if previous_banks[name] != old_banks[name]:
                    raise ValueError("reused fitting or retention rows changed")
            del earlier_banks
            cycle = inherited("cycle-0.json")
            cycle_transcripts = set()
            for record in parent["training"][:216]:
                image = archive("delegated", lambda: parent_data.load_bundle(ROOT / PARENT, record))
                for rows in image["bundle"]["families"].values():
                    cycle_transcripts.update(banks.transcript_digest(row) for row in rows)
            if (len(parent["training"]) != 648 or previous["training"] != parent["training"]
                    or len(cycle_transcripts) != 20736 or not cycle_transcripts <= trained
                    or evidence.json_digest(sorted(cycle_transcripts)) != cycle["training_manifest"]["transcript_sha256"]):
                raise ValueError("cached schedule or admitted first-cycle transcript identity differs")
            print("Authenticated original cache and prior banks; preparing fresh development bank.", flush=True)
            work = io._work(work)
            with io._tracking(work):
                fresh, canonical, admission, hashes = io._bank(cycle["plan"], cycle["training_manifest"],
                    sorted(cycle_transcripts), "retention", "dev", DEV_SEED, blocked, work)
            admission["name"] = "dev"
            rows = [row for _, group in sorted(fresh["rows"].items()) for row in group]
            image = dict(dev=dict(role="dev", rows=rows), previous_dev=deepcopy(previous_banks["dev"]),
                         train_fit=deepcopy(previous_banks["train_fit"]),
                         retention=deepcopy(previous_banks["retention"]))
            provenance = {"dev": "fresh-development", "previous_dev": "reused-entity-development",
                          "train_fit": "reused-actual-training", "retention": "reused-development"}
            inventory = {name: dict(role=bank["role"], episodes=len(bank["rows"]), pairs=len(bank["rows"]) // 2,
                rows_sha256=evidence.json_digest(bank["rows"]), provenance=provenance[name])
                for name, bank in image.items()}
            if ({name: value["episodes"] for name, value in inventory.items()} != BANK_COUNTS
                    or len(hashes) != 720 or hashes & blocked):
                raise ValueError("fixed four-bank inventory or fresh transcript admission differs")
            banks_sha = save("banks.pt", image)
            save("development-admission.json", admission)
            save("development-canonical.pt", canonical)
            work.boundary()
            if source_hashes() != sources or io.digest(ROOT / DECLARATION) != declaration_sha:
                raise ValueError("source or preparation declaration changed")
            manifest = dict(schema=SCHEMA, updates=648, layout="original", micro_batch_size=32,
                training_episodes=62208, parent_data_directory=PARENT, parent_manifest_sha256=PARENT_SHA,
                training=deepcopy(parent["training"]), banks=dict(path="banks.pt", sha256=banks_sha),
                bank_inventory=inventory, dev_seed=DEV_SEED, source_sha256=sources,
                declaration=dict(path=DECLARATION, sha256=declaration_sha),
                previous_data_directory=PREVIOUS, previous_manifest_sha256=PREVIOUS_SHA,
                earlier_data_directory=EARLIER, earlier_manifest_sha256=EARLIER_SHA,
                newly_excluded_dev_transcripts=720, prior_exclusion_count=len(prior_blocked),
                artifacts_sha256=artifacts, exclusion_count=len(blocked),
                exclusion_sha256=evidence.json_digest(sorted(blocked)),
                fresh_transcripts_sha256=evidence.json_digest(sorted(hashes)),
                training_archive_reads=216, new_training_bundles=0,
                scope="Original648 records; exact recorded-study transcript exclusions within the existing grammar. Replay IDs and training remain caller-owned.")
            receipt["manifest_sha256"] = io.write(output / "manifest.json", manifest)
            receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu,
            work=work.report(), artifacts_sha256=artifacts,
            total_archive_attempts=sum(receipt["archive_attempts"].values()),
            total_archive_completions=sum(receipt["archive_completions"].values()),
            archive_accounting="Direct5 includes two overlays, training/reserved/history inventories; delegated217 includes one parent bank and216 cached bundles. WorkLedger safe_archive_loads overlaps direct attempts; do not add it again.")
        io.write(output / "preparation.json", receipt)
    return receipt


def load_manifest(directory, *, expected_manifest_sha256):
    output = io.native(directory)
    if io.digest(output / "manifest.json") != expected_manifest_sha256:
        raise ValueError("explicit data manifest hash differs")
    manifest, receipt = io.read(output / "manifest.json"), io.read(output / "preparation.json")
    if (manifest["schema"] != SCHEMA or receipt["status"] != "completed"
            or receipt["manifest_sha256"] != expected_manifest_sha256
            or manifest["source_sha256"] != source_hashes()
            or manifest["declaration"] != dict(path=DECLARATION, sha256=io.digest(ROOT / DECLARATION))
            or manifest["parent_data_directory"] != PARENT or manifest["parent_manifest_sha256"] != PARENT_SHA
            or manifest["previous_data_directory"] != PREVIOUS or manifest["previous_manifest_sha256"] != PREVIOUS_SHA
            or manifest["dev_seed"] != DEV_SEED or manifest["exclusion_count"] != 1703160
            or {key: value["episodes"] for key, value in manifest["bank_inventory"].items()} != BANK_COUNTS):
        raise ValueError("completed source-pinned continuation overlay required")
    parent = parent_data.load_manifest(ROOT / PARENT, expected_manifest_sha256=PARENT_SHA)
    if manifest["training"] != parent["training"]:
        raise ValueError("shared original training schedule changed")
    for name, expected in manifest["artifacts_sha256"].items():
        if io.digest(output / name) != expected:
            raise ValueError("prepared evaluation artifact changed: " + name)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    print(io.encoded(prepare(parser.parse_args().directory)).decode(), flush=True)
