"""Matched world/naming diversity study, with a separately opened transfer audit.

The student is the fixed diagnostic sequence learner, not a regional promotion.
Curriculum changes are tested independently from architecture/objective changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_loop import run_lock
from brain_in_computer.learning_student import _cpu_copy
from experiments.train_cognitive import atomic_checkpoint, atomic_json, fingerprint_rows

SCHEMA = "bic-world-naming-diversity-study-v1"
FAMILIES = ("variable_binding", "graph_reachability", "conditional_logic")
HELDOUT = "arithmetic_updates"
ARMS = ("w32-n1", "w32-n8", "w256-n1", "w256-n8")
SEED, STEPS, BATCH = 2601, 3600, 64
WORLD_COUNT, NAME_COUNT = 256, 8
RATE = .001
BUDGETS = (0, 1, 4, 16, 64)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    from experiments.sequence_study import source_hashes as previous
    root = Path(__file__).resolve().parents[1]
    names = [*previous(), *("experiments/" + name + ".py" for name in (
        "diverse_curriculum", "diversity_training", "diversity_evaluation", "diversity_study"))]
    names.append("docs/DIVERSITY_STUDY_PROTOCOL.md")
    return {name: file_hash(root / name) for name in names}


def normalized_transcript(row):
    import re
    from experiments.cognitive_curriculum import ALIASES
    names = {}
    def normalize(match):
        name = match.group()
        return names.setdefault(name, f"ENTITY{len(names)}")
    return re.sub(r"\b(" + "|".join(ALIASES) + r")\b", normalize,
                  "\n".join(turn["text"] for turn in row["turns"]))


def world_variants(world):
    from experiments.diverse_curriculum import render_world_pair
    return {normalized_transcript(row) for row in render_world_pair(
        world, 42_000_000, split="audit", name_split="train")}


def all_variants(worlds):
    return set().union(*(world_variants(world) for world in worlds))


def world_pool(family, seed, count, split, *, levels=(1, 2), excluded=(), excluded_variants=(), diagnostics=None):
    from experiments.diverse_curriculum import generate_world_pair, world_fingerprint
    seen, rows, attempts = set(excluded), [], 0
    banned, duplicate_count, overlap_count = set(excluded_variants), 0, 0
    while len(rows) < count:
        level = levels[len(rows) % len(levels)]
        candidate = generate_world_pair(family, seed + attempts, level=level, split=split)
        identity = world_fingerprint(candidate)
        attempts += 1
        if identity in seen:
            duplicate_count += 1
        elif banned and banned & world_variants(candidate):
            overlap_count += 1
        else:
            seen.add(identity)
            rows.append(candidate)
        if attempts >= count * 200:
            raise ValueError(f"insufficient distinct syntactic worlds for {family}, level {level}")
    if diagnostics is not None:
        diagnostics.append({"family": family, "base_seed": seed, "split": split, "levels": list(levels),
            "accepted_world_pairs": count, "candidate_seeds": attempts,
            "duplicate_pair_rejections": duplicate_count, "single_variant_overlap_pair_rejections": overlap_count})
    return rows


def render(worlds, naming_seeds, *, split, name_split):
    from experiments.diverse_curriculum import render_world_pair
    return [row for index, world in enumerate(worlds)
            for row in render_world_pair(world, naming_seeds[index % len(naming_seeds)],
                                         split=split, name_split=name_split)]


def arm_indices(arm, slots=WORLD_COUNT * NAME_COUNT):
    if arm not in ARMS:
        raise ValueError("unknown diversity arm")
    worlds = 32 if arm.startswith("w32-") else WORLD_COUNT
    names = 1 if arm.endswith("-n1") else NAME_COUNT
    return [(index % WORLD_COUNT % worlds, index // WORLD_COUNT % names)
            for index in range(slots)]


def expected_samplers(updates):
    counts = {family: updates // 3 + int(index < updates % 3) for index, family in enumerate(FAMILIES)}
    result = {}
    for index, family in enumerate(sorted(FAMILIES)):
        generator = torch.Generator().manual_seed(SEED + index * 7919)
        for _ in range(counts[family]):
            torch.randint(WORLD_COUNT * NAME_COUNT, (BATCH // 2,), generator=generator)
        result[family] = generator.get_state()
    return result


def prepare_banks(*, with_diagnostics=False):
    """Freeze all rows before optimization; evaluation worlds are never trainable."""
    from experiments.diverse_curriculum import render_world_pair, world_fingerprint
    train_worlds, dev_worlds, audit_worlds, advanced = {}, {}, {}, {}
    selections = []
    for index, family in enumerate(FAMILIES):
        base = 41_000_000 + index * 100_000
        train_worlds[family] = world_pool(family, base, WORLD_COUNT, "train", diagnostics=selections)
        prior_variants = all_variants(train_worlds[family])
        dev_worlds[family] = world_pool(family, base + 20_000, 64, "dev",
            excluded_variants=prior_variants, diagnostics=selections)
        prior_variants |= all_variants(dev_worlds[family])
        audit_worlds[family] = world_pool(family, base + 40_000, 64, "audit",
            excluded_variants=prior_variants, diagnostics=selections)
        prior_variants |= all_variants(audit_worlds[family])
        advanced[family] = world_pool(family, base + 60_000, 64, "audit", levels=(3,),
            excluded=map(world_fingerprint, audit_worlds[family]),
            excluded_variants=prior_variants, diagnostics=selections)
    banks = {"train": {}, "development": {}, "retained": {}, "advanced": {}}
    rendered = {family: {} for family in FAMILIES}
    for arm in ARMS:
        banks["train"][arm] = {}
        for family in FAMILIES:
            # The logical slot sampler is identical in every arm. World and
            # naming indices are separate factors, with intentional duplicates.
            rows = []
            for wi, ni in arm_indices(arm):
                if (wi, ni) not in rendered[family]:
                    rendered[family][wi, ni] = render_world_pair(train_worlds[family][wi],
                        42_000_000 + ni, split="train", name_split="train")
                rows.extend(rendered[family][wi, ni])
            banks["train"][arm][family] = rows
    for destination, new_worlds, admission, name_partition, name_seed in (
            ("development", dev_worlds, "dev", "dev", 43_000_000),
            ("retained", audit_worlds, "audit", "audit", 44_000_000)):
        for family in FAMILIES:
            # All arms saw these first 32 worlds, although at different frequencies.
            # Each gets two novel maps; exact new maps, not just new RNG seeds,
            # are confirmed below by transcript separation.
            familiar = train_worlds[family][:32] * 2
            banks[destination][f"names/{family}"] = render(familiar,
                list(range(name_seed, name_seed + 64)), split=admission, name_split=name_partition)
            banks[destination][f"worlds/{family}"] = render(new_worlds[family],
                [42_000_000], split=admission, name_split="train")
            banks[destination][f"both/{family}"] = render(new_worlds[family],
                list(range(name_seed, name_seed + 64)), split=admission, name_split=name_partition)
            banks["advanced"][family] = render(advanced[family],
                list(range(45_000_000, 45_000_064)), split="audit", name_split="audit")
    support = world_pool(HELDOUT, 46_000_000, 64, "train", diagnostics=selections)
    banks["support"] = {HELDOUT: render(support, [42_000_000], split="train", name_split="train")}
    banks["query"] = {}
    seen = set()
    prior_variants = all_variants(support)
    for level in (2, 3):
        worlds = world_pool(HELDOUT, 47_000_000 + level * 10_000, 128, "audit", levels=(level,), excluded=seen,
            excluded_variants=prior_variants, diagnostics=selections)
        seen.update(map(world_fingerprint, worlds))
        prior_variants |= all_variants(worlds)
        banks["query"][f"level_{level}"] = render(worlds,
            list(range(48_000_000, 48_000_128)), split="audit", name_split="audit")
    # Exact transcript disjointness is independently checked even though the
    # semantic partition concerns only paired anonymous program identities.
    def transcripts(rows):
        return {tuple(turn["text"] for turn in row["turns"]) for row in rows}
    trained = set().union(*(transcripts(rows) for arm in banks["train"].values() for rows in arm.values()))
    for group in ("development", "retained", "advanced"):
        for key, rows in banks[group].items():
            if trained & transcripts(rows):
                raise ValueError(f"exact transcript overlap in {group}/{key}")
    if transcripts(banks["support"][HELDOUT]) & set().union(*map(transcripts, banks["query"].values())):
        raise ValueError("support/query exact transcript overlap")
    return (banks, selections) if with_diagnostics else banks


def bank_manifest(banks):
    return {group: ({arm: {family: fingerprint_rows(rows) for family, rows in values.items()}
                    for arm, values in groups.items()} if group == "train" else
                   {name: fingerprint_rows(rows) for name, rows in groups.items()})
            for group, groups in banks.items()}


def diversity_counts(rows):
    from experiments.cognitive_curriculum import ALIASES
    from experiments.diverse_curriculum import naming_map
    import re
    pattern = re.compile(r"\b(" + "|".join(ALIASES) + r")\b")
    texts, normalized, maps, assignments = set(), set(), set(), set()
    for row in rows:
        joined = "\n".join(turn["text"] for turn in row["turns"])
        names = {}
        def normalize(match):
            name = match.group()
            return names.setdefault(name, f"ENTITY{len(names)}")
        normalized.add(pattern.sub(normalize, joined))
        texts.add(joined)
        maps.add(tuple(names))
        assignments.add(tuple(sorted(naming_map(row["naming_seed"], row["name_split"]).items())))
    return {"episodes": len(rows), "unique_transcripts": len(texts),
            "anonymous_pair_worlds": len({row["world_fingerprint"] for row in rows}),
            "complete_naming_maps": [dict(items) for items in sorted(assignments)],
            "unique_rendered_pair_groups": len({row["counterfactual_group"] for row in rows}),
            "first_occurrence_normalized_transcripts": len(normalized),
            "distinct_observed_name_orders": len(maps),
            "query_target_counts": {str(target): sum(turn["target"] == target for row in rows for turn in row["turns"])
                                    for target in range(3)},
            "target_counts_by_turn": [{str(target): sum(row["turns"][turn]["target"] == target for row in rows)
                                       for target in range(4)} for turn in range(6)],
            "note": "Normalized strings and observed name orders are syntactic counts, not independent semantic samples."}


def prepare(directory):
    from experiments.diversity_evaluation import validate_diverse_evaluation_rows
    from experiments.sequence_student import build_sequence_student
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        if (directory / "protocol.json").exists():
            return load_protocol(directory)
        banks, selections = prepare_banks(with_diagnostics=True)
        counts = {arm: {family: diversity_counts(rows) for family, rows in values.items()}
                  for arm, values in banks["train"].items()}
        for arm, families in counts.items():
            worlds = 32 if arm.startswith("w32-") else 256
            names = 1 if arm.endswith("-n1") else 8
            for row in families.values():
                if (row["anonymous_pair_worlds"] != worlds or len(row["complete_naming_maps"]) != names
                        or row["unique_rendered_pair_groups"] != worlds * names):
                    raise ValueError("actual world/naming diversity differs from intended factors")
        for group in ("development", "retained", "advanced", "support", "query"):
            for rows in banks[group].values():
                validate_diverse_evaluation_rows(rows, BATCH)
        protocol = {"schema": SCHEMA, "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_hashes(), "seed": SEED, "steps": STEPS, "batch_size": BATCH,
            "world_selection": selections,
            "learning_rate": RATE, "arms": list(ARMS), "training_families": list(FAMILIES),
            "heldout_family": HELDOUT, "world_counts": [32, 256], "naming_seed_counts": [1, 8],
            "logical_pairs_per_family": WORLD_COUNT * NAME_COUNT, "levels": [1, 2],
            "model": "unchanged default independent SequenceStudent (753610 parameters)",
            "initial_weights_sha256": checkpoint_digest(build_sequence_student(SEED)),
            "final_sampler_state_sha256": {family: hashlib.sha256(state.numpy().tobytes()).hexdigest()
                                           for family, state in expected_samplers(STEPS).items()},
            "optimizer": "AdamW, fresh initialization, fixed rate from prior study, no new rate search",
            "objective": "unchanged balanced queries + .25 ACK + .1 per-turn replies + .1 per-turn observation bytes",
            "selection": "No winner selected for this audit; all four fixed endpoints plus one common fresh control.",
            "automatic_promotion": False,
            "audit_budgets": list(BUDGETS), "audit_seed": 3601, "support_episodes": 128,
            "interpretation": "Exploratory one-seed finite-bank diversity factorial. Worlds are anonymous program identities, not semantic equivalence classes. Name seeds induce bijections; counts are verified separately. Different world pools may differ in class/composition frequencies. This is not online curriculum refresh, a general intelligence claim or regional BiC promotion.",
            "banks": bank_manifest(banks)}
        # torch serialization preserves repeated-object references, avoiding a
        # large expanded JSON copy of the intentionally repeated slot banks.
        atomic_checkpoint(directory / "banks.pt", banks)
        protocol["banks_file_sha256"] = file_hash(directory / "banks.pt")
        atomic_json(directory / "diversity.json", counts)
        root = Path(__file__).resolve().parents[1]
        for name in protocol["source_sha256"]:
            target = directory / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
        atomic_json(directory / "protocol.json", protocol)
        return protocol


def load_protocol(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf8"))
    if protocol.get("schema") != SCHEMA or protocol["source_sha256"] != source_hashes():
        raise ValueError("study source changed")
    if any(file_hash(directory / "source" / name) != sha for name, sha in protocol["source_sha256"].items()):
        raise ValueError("frozen source changed")
    if file_hash(directory / "banks.pt") != protocol["banks_file_sha256"]:
        raise ValueError("frozen banks changed")
    return protocol


def read_banks(directory):
    protocol = load_protocol(directory)
    banks = torch.load(Path(directory) / "banks.pt", map_location="cpu", weights_only=True)
    if bank_manifest(banks) != protocol["banks"]:
        raise ValueError("bank identities differ")
    return protocol, banks


def evaluate(model, banks, **kwargs):
    from experiments.diversity_evaluation import evaluate_diverse_banks
    return evaluate_diverse_banks(model, banks, **kwargs)


def validate_resume(saved, protocol, arm):
    if saved.get("schema") != SCHEMA or saved.get("protocol") != protocol or saved.get("arm") != arm:
        raise ValueError("resume contract differs")
    updates = saved["training"]["updates"]
    if type(updates) is not int or not 0 <= updates <= STEPS:
        raise ValueError("invalid resume updates")
    if saved["training"]["family_updates"] != {family: updates // 3 + int(index < updates % 3)
                                               for index, family in enumerate(FAMILIES)}:
        raise ValueError("resume schedule differs")
    expected = expected_samplers(updates)
    if set(saved["training"]["samplers"]) != set(expected) or any(
            not torch.equal(saved["training"]["samplers"][family], state) for family, state in expected.items()):
        raise ValueError("resume sampler stream differs")
    if saved.get("initial_weights_sha256") != protocol["initial_weights_sha256"]:
        raise ValueError("initial weight provenance differs")


def train(directory, arm, device="cuda"):
    from experiments.diversity_training import DiversityTrainer
    if arm not in ARMS:
        raise ValueError("unknown arm")
    directory = Path(directory)
    protocol, banks = read_banks(directory)
    output = directory / "main" / arm
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        latest = output / "latest.pt"
        saved = torch.load(latest, map_location="cpu", weights_only=True) if latest.exists() else None
        if saved:
            validate_resume(saved, protocol, arm)
        trainer = DiversityTrainer(banks["train"][arm], seed=SEED, device=device,
            batch_size=BATCH, learning_rate=RATE, payload=saved["training"] if saved else None)
        if saved is None:
            if checkpoint_digest(trainer.model) != protocol["initial_weights_sha256"]:
                raise ValueError("fresh initialization differs")
            atomic_checkpoint(output / "initial.pt", {"weights": _cpu_copy(trainer.model.state_dict())})
        history = saved["history"] if saved else []
        elapsed = saved["training_seconds"] if saved else 0.
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        def snapshot():
            return {"schema": SCHEMA, "protocol": protocol, "arm": arm,
                "initial_weights_sha256": protocol["initial_weights_sha256"],
                "training": trainer.snapshot(), "history": history, "training_seconds": elapsed}
        while trainer.updates < STEPS:
            tick = time.monotonic()
            last = trainer.step(FAMILIES[trainer.updates % 3])
            elapsed += time.monotonic() - tick
            if trainer.updates % 600 == 0:
                metrics = evaluate(trainer.model, banks["development"], score_replies=False)
                history.append({"updates": trainer.updates, "training_seconds": elapsed,
                                "development": metrics, "loss": last["loss"]})
                load_protocol(directory)
                atomic_checkpoint(latest, snapshot())
                atomic_json(output / "progress.json", {"arm": arm, "history": history})
                print(json.dumps({"arm": arm, "updates": trainer.updates,
                    "pairs": metrics["macro_pair_accuracy"], "later": metrics["macro_later_known_accuracy"]}), flush=True)
        load_protocol(directory)
        atomic_checkpoint(latest, snapshot())
        final = evaluate(trainer.model, banks["development"], score_replies=True)
        # Same 32 familiar worlds and first naming map, unique pairs in all arms.
        fit = evaluate(trainer.model, {family: rows[:64] for family, rows in banks["train"][arm].items()},
                       score_replies=False)
        atomic_json(output / "report.json", {"schema": SCHEMA, "arm": arm, "protocol": protocol,
            "updates": trainer.updates, "episodes": trainer.updates * BATCH,
            "training_seconds": elapsed, "current_invocation_seconds_after_setup": time.monotonic() - start,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
            "peak_scope": "After encoding, through training and fixed-size evaluation batches; cached banks included.",
            "development": final, "fit_subset": fit, "history": history,
            "weights_sha256": checkpoint_digest(trainer.model), "checkpoint_file_sha256": file_hash(latest),
            "heldout_evaluation_performed": False})


def audit(directory, device="cuda"):
    from experiments.diversity_training import DiversityTrainer
    from experiments.sequence_student import build_sequence_student
    from experiments.audit_sequence_study import sequence_restart
    directory = Path(directory)
    protocol, banks = read_banks(directory)
    output = directory / "audit"
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        candidates, hashes = {}, {}
        for arm in ARMS:
            path = directory / "main" / arm
            report = json.loads((path / "report.json").read_text(encoding="utf8"))
            saved = torch.load(path / "latest.pt", map_location="cpu", weights_only=True)
            validate_resume(saved, protocol, arm)
            if (saved["training"]["updates"] != STEPS or report["updates"] != STEPS
                    or report["protocol"] != protocol or report["arm"] != arm
                    or report["heldout_evaluation_performed"]
                    or report["checkpoint_file_sha256"] != file_hash(path / "latest.pt")):
                raise ValueError("all four intact completed endpoints required")
            verifier = build_sequence_student(SEED)
            verifier.load_state_dict(saved["training"]["weights"], strict=True)
            if checkpoint_digest(verifier) != report["weights_sha256"]:
                raise ValueError("reported endpoint tensor digest differs")
            verifier.load_state_dict(torch.load(path / "initial.pt", map_location="cpu", weights_only=True)["weights"], strict=True)
            if checkpoint_digest(verifier) != protocol["initial_weights_sha256"]:
                raise ValueError("saved initial weights differ")
            candidates[arm] = saved["training"]["weights"]
            hashes[arm] = file_hash(path / "latest.pt")
        candidates["fresh"] = _cpu_copy(build_sequence_student(SEED).state_dict())
        marker = {"schema": SCHEMA, "parent_sha256": hashes, "protocol_sha256": file_hash(directory / "protocol.json")}
        marker_path = output / "evaluation-started.json"
        if marker_path.exists() and json.loads(marker_path.read_text(encoding="utf8")) != marker:
            raise ValueError("audit inputs changed")
        atomic_json(marker_path, marker)
        results = {}
        for name, weights in candidates.items():
            result_path = output / f"{name}.json"
            if result_path.exists():
                prior = json.loads(result_path.read_text(encoding="utf8"))
                if prior["inputs"] != marker or prior["name"] != name:
                    raise ValueError("cached audit identity differs")
                results[name] = prior
                continue
            trainer = DiversityTrainer(banks["support"], seed=3601, device=device, batch_size=BATCH, learning_rate=RATE)
            trainer.model.load_state_dict(weights, strict=True)
            initial = checkpoint_digest(trainer.model)
            retained = evaluate(trainer.model, banks["retained"])
            advanced = evaluate(trainer.model, banks["advanced"])
            curve, seconds = [], 0.
            for budget in BUDGETS:
                while trainer.updates < budget:
                    tick = time.monotonic()
                    trainer.step(HELDOUT)
                    seconds += time.monotonic() - tick
                curve.append({"updates": budget, "exposures": budget * BATCH,
                              **evaluate(trainer.model, banks["query"])})
            after = evaluate(trainer.model, banks["retained"])
            advanced_after = evaluate(trainer.model, banks["advanced"])
            controls = {control: evaluate(trainer.model, banks["query"], **options) for control, options in
                        (("blank_text", {"blank_text": True}), ("reset_history", {"reset_history": True}))}
            areas = {metric: sum((b["updates"] - a["updates"]) * (a[metric] + b[metric]) / 2
                                 for a, b in zip(curve, curve[1:])) / BUDGETS[-1]
                     for metric in ("macro_query_accuracy", "macro_pair_accuracy", "macro_later_known_accuracy")}
            result = {"name": name, "inputs": marker, "initial_weights_sha256": initial,
                "final_weights_sha256": checkpoint_digest(trainer.model), "curve": curve, "areas": areas,
                "retention_before": retained, "retention_after": after, "advanced_before": advanced,
                "advanced_after": advanced_after, "controls": controls, "training_seconds": seconds,
                "cpu_restart": sequence_restart(trainer.model, banks["query"]["level_2"][0])}
            atomic_checkpoint(output / f"{name}-adapted.pt", trainer.snapshot())
            atomic_json(result_path, result)
            results[name] = result
            print(json.dumps({"candidate": name, "query_auc": areas["macro_query_accuracy"],
                "pair_auc": areas["macro_pair_accuracy"], "retention": [retained["macro_query_accuracy"], after["macro_query_accuracy"]]}), flush=True)
        if hashes != {arm: file_hash(directory / "main" / arm / "latest.pt") for arm in ARMS}:
            raise RuntimeError("parent checkpoint changed during audit")
        load_protocol(directory)
        atomic_json(output / "report.json", {"schema": SCHEMA, "inputs": marker, "results": results,
            "base_checkpoints_unchanged": True, "automatic_promotion": False,
            "prior_learning_area_gains": {name: {key: row["areas"][key] - results["fresh"]["areas"][key]
                for key in row["areas"]} for name, row in results.items() if name != "fresh"}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("prepare", "main", "audit"), required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "prepare":
        result = prepare(args.output)
        print(json.dumps({"prepared": True, "source_files": len(result["source_sha256"])}))
    elif args.phase == "main":
        train(args.output, args.arm, args.device)
    else:
        audit(args.output, args.device)


if __name__ == "__main__":
    main()
