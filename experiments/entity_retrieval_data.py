"""Fresh evaluation overlay reusing the already admitted training cache."""
from copy import deepcopy
from pathlib import Path
import time
import traceback

from experiments import foundation_tutor_loop_data as io
from experiments import recurrent_read_data as parent_data
from experiments import shared_state_data as previous_data

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-entity-retrieval-data-v1"
PARENT = "runs/recurrent-read-data-local/attempt-001"
PARENT_SHA = "c420a43a3a45f63347288dda7db48c71b3e7fa183b4a1681c2ec89eabeb08654"
DEV_SEED = 852502001
PREVIOUS = "runs/shared-state-data-local/attempt-001"
PREVIOUS_SHA = "266dfe643074b8d66b65b1e506f6fc726fa6cb75f2a532b04232ec87bf30eb75"


def source_hashes():
    return {**previous_data.source_hashes(), "experiments/entity_retrieval_data.py": io.digest(ROOT / "experiments/entity_retrieval_data.py")}


def prepare(directory):
    start, cpu = time.monotonic(), time.process_time()
    output = io.native(directory)
    output.mkdir(parents=True, exist_ok=False)
    work = io.WorkLedger(deadline=start + 600)
    receipt = dict(schema=SCHEMA, status="running", neural_work=0, teacher_calls=0, inherited_archive_attempts=0, inherited_archive_loads=0)
    artifacts = {}

    def save(name, value):
        artifacts[name] = io.write(output / name, value)
        return artifacts[name]

    def delegated_archive(call):
        receipt["inherited_archive_attempts"] += 1
        result = call()
        receipt["inherited_archive_loads"] += 1
        return result

    try:
        with io._cpu_guard(work):
            from experiments import foundation_banks as banks, foundation_evidence as evidence
            sources = source_hashes()
            parent = parent_data.load_manifest(ROOT / PARENT, expected_manifest_sha256=PARENT_SHA)
            old_banks = delegated_archive(lambda: parent_data.load_banks(ROOT / PARENT, parent))
            previous = previous_data.load_manifest(ROOT / PREVIOUS, expected_manifest_sha256=PREVIOUS_SHA)
            previous_bank_path = ROOT / PREVIOUS / previous["banks"]["path"]
            if io.digest(previous_bank_path) != previous["banks"]["sha256"]:
                raise ValueError("previous development bank changed")
            previous_banks = io._load_pt(previous_bank_path, work)
            tutor, tutor_manifest, _ = io._header(ROOT / parent_data.PARENT, parent_data.PARENT_SHA256)
            def inherited(name):
                if io.digest(tutor / name) != tutor_manifest["artifacts_sha256"][name]:
                    raise ValueError("inherited transcript evidence changed")
                return io._load_pt(tutor / name, work) if name.endswith(".pt") else io.read(tutor / name)
            trained = evidence.transcript_set(inherited("training-transcripts.pt"))
            reserved = evidence.transcript_set(inherited("reserved-transcripts.pt"))
            history, _ = io._history(work)
            prior_dev = {banks.transcript_digest(row) for row in old_banks["dev"]["rows"]}
            newly_excluded = {banks.transcript_digest(row) for row in previous_banks["dev"]["rows"]}
            old_blocked = history | trained | reserved | prior_dev
            if len(newly_excluded) != 720 or len(old_blocked) != 1701720 or old_blocked & newly_excluded:
                raise ValueError("previous development exclusion must add720 distinct transcripts")
            blocked = old_blocked | newly_excluded
            del previous_banks
            if len(blocked) != 1702440:
                raise ValueError("complete prior exclusion inventory required")
            cycle = inherited("cycle-0.json")
            cycle_transcripts = set()
            for record in parent["training"][:216]:
                work.boundary()
                image = delegated_archive(lambda: parent_data.load_bundle(ROOT / PARENT, record))
                for rows in image["bundle"]["families"].values():
                    cycle_transcripts.update(banks.transcript_digest(row) for row in rows)
            if (len(cycle_transcripts) != 20736 or not cycle_transcripts <= trained
                    or evidence.json_digest(sorted(cycle_transcripts)) != cycle["training_manifest"]["transcript_sha256"]):
                raise ValueError("cached cycle differs from admitted training manifest")
            print("Authenticated cached lessons; generating fresh evaluation questions.", flush=True)
            work = io._work(work)
            with io._tracking(work):
                fresh, canonical, admission, hashes = io._bank(cycle["plan"], cycle["training_manifest"],
                    sorted(cycle_transcripts), "retention", "dev", DEV_SEED, blocked, work)
            admission["name"] = "dev"
            rows = [row for _, group in sorted(fresh["rows"].items()) for row in group]
            image = dict(dev=dict(role="dev", rows=rows), train_fit=deepcopy(old_banks["train_fit"]),
                         retention=deepcopy(old_banks["retention"]))
            banks_sha = save("banks.pt", image)
            save("development-admission.json", admission)
            save("development-canonical.pt", canonical)
            inventory = {name: dict(role=bank["role"], episodes=len(bank["rows"]), pairs=len(bank["rows"]) // 2,
                rows_sha256=evidence.json_digest(bank["rows"]),
                provenance={"dev": "fresh-development", "train_fit": "reused-actual-training", "retention": "reused-development"}[name])
                for name, bank in image.items()}
            if source_hashes() != sources:
                raise ValueError("source changed during evaluation preparation")
            manifest = dict(schema=SCHEMA, updates=648, layout="original", micro_batch_size=32,
                training_episodes=62208, parent_data_directory=PARENT, parent_manifest_sha256=PARENT_SHA,
                training=deepcopy(parent["training"]), banks=dict(path="banks.pt", sha256=banks_sha),
                bank_inventory=inventory, dev_seed=DEV_SEED, source_sha256=sources,
                previous_data_directory=PREVIOUS, previous_manifest_sha256=PREVIOUS_SHA, newly_excluded_dev_transcripts=720,
                artifacts_sha256=artifacts, exclusion_count=len(blocked),
                exclusion_sha256=evidence.json_digest(sorted(blocked)), fresh_transcripts_sha256=evidence.json_digest(sorted(hashes)),
                target_generation="Derived once per bundle from observed English, shared across both arms; training-only.",
                scope="Exact transcript exclusions within the existing finite grammar; reused immutable training cache.")
            receipt["manifest_sha256"] = io.write(output / "manifest.json", manifest)
            receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu,
                       work=work.report(), artifacts_sha256=artifacts,
                       archive_accounting="WorkLedger counts direct archive reads; inherited_archive_loads counts the217 delegated parent-bank/bundle reads separately. Sum them for total archive reads; wall includes both.")
        io.write(output / "preparation.json", receipt)
    return receipt


def load_manifest(directory, *, expected_manifest_sha256):
    output = io.native(directory)
    if io.digest(output / "manifest.json") != expected_manifest_sha256:
        raise ValueError("explicit data manifest hash differs")
    manifest, receipt = io.read(output / "manifest.json"), io.read(output / "preparation.json")
    if (manifest["schema"] != SCHEMA or receipt["status"] != "completed"
            or receipt["manifest_sha256"] != expected_manifest_sha256 or manifest["source_sha256"] != source_hashes()):
        raise ValueError("completed source-pinned data overlay required")
    parent = parent_data.load_manifest(ROOT / manifest["parent_data_directory"], expected_manifest_sha256=manifest["parent_manifest_sha256"])
    if manifest["training"] != parent["training"]:
        raise ValueError("shared training schedule changed")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    result = prepare(parser.parse_args().directory)
    print(io.encoded(result).decode(), flush=True)
