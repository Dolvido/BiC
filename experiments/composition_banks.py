"""Prospectively separated shared-composition banks; no model access or scoring.

Canonical truth is authenticated before admission. Exact observed transcripts are
globally distinct. Familiar panels deliberately reuse training structure seeds;
held panels protect every supervised composed-query signature across domains and
lengths, rather than relying on full-program or name hashes alone.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

from experiments.train_cognitive import fingerprint_rows


FAMILIES = ("color", "count", "switch")
LENGTHS = (8, 10, 12)
TRAIN_PAIRS, PANEL_PAIRS = 256, 64
SEEDS = {"train": 70_000_000, "development": 80_000_000, "audit": 90_000_000}
NAMING_OFFSET, VALUE_OFFSET = 100_000_000, 200_000_000
COMPOSED_OFFSET = 300_000
MAX_INPUT_BYTES, MAX_CONTEXT_TOKENS = 128, 1024


def curriculum():
    from experiments import composition_curriculum
    return composition_curriculum


def transcript(row):
    return tuple(turn["text"] for turn in row["turns"])


def known_composed_queries(row):
    return [item for item in curriculum().query_ancestries(row)
            if item["known"] and item["composed"]]


def _seed(group, family, turns):
    return SEEDS[group] + FAMILIES.index(family) * 1_000_000 + turns * 10_000


def _check_pair(pair, *, family, turns, admission, structure_split):
    api = curriculum()
    if api.validate_pair(pair) is False:
        raise ValueError("canonical composition pair rejected")
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError("complete canonical pair required")
    for row in pair:
        if (row["family"] != family or row["split"] != admission
                or row["structure_partition"] != structure_split
                or len(row["turns"]) != turns):
            raise ValueError("canonical pair escaped the requested bank role")
        sizes = [len(turn["text"].encode("utf8")) for turn in row["turns"]]
        if max(sizes) > MAX_INPUT_BYTES or sum(sizes) + 2 * turns > MAX_CONTEXT_TOKENS:
            raise ValueError("canonical observation exceeds model capacity; no truncation")
        queries = known_composed_queries(row)
        if not queries or queries[-1]["turn_index"] != turns - 1:
            raise ValueError("final known query must have composed causal ancestry")
        if queries[-1]["structure_id"] != row["structure_id"]:
            raise ValueError("final ancestry metadata disagrees with row structure")
        if admission == "train" and any(q["structure_partition"] != "train" for q in queries):
            raise ValueError("training contains a supervised held-out composition")
    if {row["turns"][-1]["target"] for row in pair} != {0, 1}:
        raise ValueError("final counterfactual answers must be known and opposite")


def _bank_stats(rows):
    lengths = {len(row["turns"]) for row in rows}
    if len(lengths) != 1:
        raise ValueError("bank must have one actual episode length")
    turns = next(iter(lengths))
    signatures = set()
    by_partition = {name: set() for name in ("train", "dev", "audit")}
    for row in rows:
        for query in known_composed_queries(row):
            signatures.add(query["structure_id"])
            by_partition[query["structure_partition"]].add(query["structure_id"])
    bytes_per_row = [sum(len(turn["text"].encode("utf8")) for turn in row["turns"])
                     for row in rows]
    return {"sha256": fingerprint_rows(rows), "episodes": len(rows),
        "complete_pairs": len(rows) // 2, "turns": turns,
        "unique_exact_transcripts": len({transcript(row) for row in rows}),
        "distinct_full_procedures": len({row["program_id"] for row in rows}),
        "distinct_naming_maps": len({tuple(sorted(curriculum().naming_map(row["recipe"]["naming_seed"]).items())) for row in rows}),
        "distinct_value_seeds": len({row["recipe"]["value_seed"] for row in rows}),
        "final_structure_ids": sorted({row["structure_id"] for row in rows}),
        "distinct_final_structures": len({row["structure_id"] for row in rows}),
        "known_composed_query_ids": sorted(signatures),
        "distinct_known_composed_queries": len(signatures),
        "known_composed_query_ids_by_partition": {name: sorted(values) for name, values in by_partition.items()},
        "target_counts_by_turn": [{str(target): sum(row["turns"][index]["target"] == target for row in rows)
                                  for target in range(4)} for index in range(turns)],
        "observation_utf8_bytes": sum(bytes_per_row),
        "observation_tokens": sum(bytes_per_row) + len(rows) * turns * 2,
        "max_context_tokens": max(bytes_per_row) + 2 * turns,
        "max_utterance_bytes": max(len(turn["text"].encode("utf8")) for row in rows for turn in row["turns"])}


def bank_manifest(banks):
    if not isinstance(banks, dict) or set(banks) != {"train", "development", "audit"}:
        raise ValueError("expected train/development/audit composition banks")
    return {"train": {family: {str(turns): _bank_stats(rows) for turns, rows in buckets.items()}
                      for family, buckets in banks["train"].items()},
            **{group: {name: _bank_stats(rows) for name, rows in banks[group].items()}
               for group in ("development", "audit")}}


def verify_boundaries(banks):
    """Audit exact observations and all supervised composition identities."""
    if not isinstance(banks, dict) or set(banks) != {"train", "development", "audit"}:
        raise ValueError("expected train/development/audit composition banks")
    if set(banks["train"]) != set(FAMILIES) or any(set(values) != set(LENGTHS)
            for values in banks["train"].values()):
        raise ValueError("training requires every declared family and length bucket")
    expected_panels = {f"{panel}/{family}/t{turns}" for panel in ("seen", "composed")
                       for family in FAMILIES for turns in LENGTHS}
    if any(set(banks[group]) != expected_panels for group in ("development", "audit")):
        raise ValueError("evaluation panel set differs from the declared study")
    groups = [rows for values in banks["train"].values() for rows in values.values()]
    groups += list(banks["development"].values()) + list(banks["audit"].values())
    if any(not isinstance(rows, (list, tuple)) or not rows or len(rows) % 2 for rows in groups):
        raise ValueError("every bank requires nonempty complete adjacent pairs")
    exact, train_queries, development_queries = set(), set(), set()
    final_composed = {"development": set(), "audit": set()}
    counts = Counter()
    for family, buckets in banks["train"].items():
        for turns, rows in buckets.items():
            for index in range(0, len(rows), 2):
                _check_pair(rows[index:index + 2], family=family, turns=turns,
                            admission="train", structure_split="train")
            for row in rows:
                value = transcript(row)
                if value in exact:
                    raise ValueError("exact observation transcript reused across banks")
                exact.add(value)
                train_queries.update(q["structure_id"] for q in known_composed_queries(row))
                counts["train_episodes"] += 1
    for group, admission in (("development", "dev"), ("audit", "audit")):
        for name, rows in banks[group].items():
            panel, family, length = name.split("/")
            if panel not in ("seen", "composed") or not length.startswith("t"):
                raise ValueError("unknown evaluation panel name")
            structure_split = "train" if panel == "seen" else admission
            for index in range(0, len(rows), 2):
                _check_pair(rows[index:index + 2], family=family, turns=int(length[1:]),
                            admission=admission, structure_split=structure_split)
            for row in rows:
                value = transcript(row)
                if value in exact:
                    raise ValueError("exact observation transcript reused across banks")
                exact.add(value)
                queries = known_composed_queries(row)
                if group == "development":
                    if any(q["structure_partition"] == "audit" for q in queries):
                        raise ValueError("development exposes supervised audit ancestry")
                    development_queries.update(q["structure_id"] for q in queries)
                if panel == "composed":
                    final_composed[group].add(row["structure_id"])
                counts[f"{group}_episodes"] += 1
    if final_composed["development"] & train_queries:
        raise ValueError("development final motifs overlap supervised training queries")
    if final_composed["audit"] & (train_queries | development_queries):
        raise ValueError("audit final motifs overlap earlier supervised query motifs")
    return {**dict(counts), "unique_exact_transcripts": len(exact),
        "training_known_composed_query_ids": sorted(train_queries),
        "development_known_composed_query_ids": sorted(development_queries),
        "development_final_composed_ids": sorted(final_composed["development"]),
        "audit_final_composed_ids": sorted(final_composed["audit"]),
        "global_exact_transcript_disjoint": True,
        "supervised_composed_query_boundary_verified": True}


def prepare_banks(*, with_diagnostics=False, train_pairs=TRAIN_PAIRS, panel_pairs=PANEL_PAIRS):
    """Generate authenticated banks without model access, scoring or file writes.

    Smaller counts are for bounded CPU tests; the study manifest freezes the
    default256 training/64 evaluation pairs. Seen panels reuse the first training
    structure recipes with independently seeded fresh values and full naming.
    Anonymous-world repetition is allowed; exact observed transcript reuse is not.
    """
    if type(with_diagnostics) is not bool:
        raise ValueError("with_diagnostics must be boolean")
    if (type(train_pairs) is not int or type(panel_pairs) is not int
            or not 1 <= panel_pairs <= train_pairs):
        raise ValueError("positive panel_pairs must not exceed train_pairs")
    api = curriculum()
    if tuple(api.FAMILIES) != FAMILIES:
        raise ValueError("composition family order differs from the frozen bank contract")
    banks = {"train": {family: {} for family in FAMILIES}, "development": {}, "audit": {}}
    recipes, diagnostics, exact = {}, [], set()

    def build(group, panel, family, turns, count, source=None):
        admission = "dev" if group == "development" else group
        partition = "train" if group == "train" or panel == "seen" else admission
        base = _seed(group, family, turns) + (COMPOSED_OFFSET if panel == "composed" else 0)
        result, accepted = [], []
        attempts, rejected_exact, rejected_audit = 0, 0, 0
        while len(accepted) < count:
            if attempts >= max(10_000, count * 1_000):
                raise ValueError("insufficient disjoint canonical compositions within bounded search")
            candidate = base + attempts
            original = source[len(accepted)] if source is not None else None
            seed = original["seed"] if original else candidate
            recipe = {"family": family, "seed": seed, "split": admission, "turns": turns,
                "naming_seed": candidate + NAMING_OFFSET, "value_seed": candidate + VALUE_OFFSET,
                "structure_split": partition}
            attempts += 1
            pair = api.generate_pair(**recipe)
            _check_pair(pair, family=family, turns=turns, admission=admission, structure_split=partition)
            if original and pair[0]["structure_id"] != original["structure_id"]:
                raise ValueError("fresh values or names changed a familiar structural recipe")
            if group == "development" and any(q["structure_partition"] == "audit"
                    for row in pair for q in known_composed_queries(row)):
                rejected_audit += 1
                continue
            observed = [transcript(row) for row in pair]
            if len(set(observed)) != 2 or any(value in exact for value in observed):
                rejected_exact += 1
                continue
            exact.update(observed)
            result.extend(deepcopy(pair))
            accepted.append({**recipe, "structure_id": pair[0]["structure_id"],
                             "canonical_recipe": deepcopy(pair[0]["recipe"]),
                             "program_id": pair[0]["program_id"],
                             "pair_sha256": fingerprint_rows(pair)})
        key = f"{panel}/{family}/t{turns}"
        diagnostics.append({"group": group, "bank": key, "base_seed": base,
            "requested_pairs": count, "candidate_attempts": attempts,
            "rejected_exact_transcript_pairs": rejected_exact,
            "rejected_development_audit_query_pairs": rejected_audit,
            "recipes": accepted})
        return result, accepted

    for family in FAMILIES:
        for turns in LENGTHS:
            rows, selected = build("train", "train", family, turns, train_pairs)
            banks["train"][family][turns] = rows
            recipes[family, turns] = selected
    for group in ("development", "audit"):
        for family in FAMILIES:
            for turns in LENGTHS:
                for panel in ("seen", "composed"):
                    rows, _ = build(group, panel, family, turns, panel_pairs,
                        source=recipes[family, turns][:panel_pairs] if panel == "seen" else None)
                    banks[group][f"{panel}/{family}/t{turns}"] = rows
    boundaries = verify_boundaries(banks)
    details = {"seed_namespaces": dict(SEEDS), "family_seed_stride": 1_000_000,
        "length_seed_stride": 10_000, "composed_panel_offset": COMPOSED_OFFSET,
        "naming_seed_offset": NAMING_OFFSET, "value_seed_offset": VALUE_OFFSET,
        "training_pairs_per_bucket": train_pairs, "evaluation_pairs_per_panel": panel_pairs,
        "selection": diagnostics, "boundaries": boundaries,
        "seen_note": "First training structure recipes with fresh independently seeded names and values; anonymous programs may repeat but exact observed transcripts cannot.",
        "composition_note": "Final supervised copy/advance ancestry signatures are withheld across domains and lengths; all known composed training queries are train-partition and development contains no audit-partition composed query. Unqueried subexpressions and semantically equivalent chains are not claimed novel."}
    return (banks, details) if with_diagnostics else banks
