"""Fresh panels and bounded original-data replay for consolidation experiments.

Old weights and banks are read-only. Novel world pools reject every normalized
single-member transcript in the prior study and earlier new pools. Naming-only
panels intentionally reuse familiar worlds but require new rendered transcripts.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache

from experiments.diverse_curriculum import (
    generate_world_pair, naming_map, render_world_pair, validate_pair,
)
from experiments.diversity_evaluation import validate_diverse_evaluation_rows
from experiments.diversity_study import normalized_transcript, world_pool, all_variants
from experiments.train_cognitive import fingerprint_rows


FAMILIES = ("variable_binding", "graph_reachability", "conditional_logic")
HELDOUT = "arithmetic_updates"
TRAIN_NAME_SEEDS = tuple(range(42_000_000, 42_000_008))
ADVANCED_ARITHMETIC_WORLD_SPLIT = "dev"


def _transcript(row):
    return tuple(turn["text"] for turn in row["turns"])


@lru_cache(maxsize=8192)
def _naming_key(seed, split):
    return tuple(sorted(naming_map(seed, split).items()))


def _map_key(row):
    return _naming_key(row["naming_seed"], row["name_split"])


def _unique_pairs(rows, *, train=False, family=None):
    if not isinstance(rows, (list, tuple)) or not rows or len(rows) % 2:
        raise ValueError("nonempty complete adjacent pairs required")
    seen_contents, groups, pairs = set(), {}, []
    for index in range(0, len(rows), 2):
        pair = rows[index:index + 2]
        content = fingerprint_rows(pair)
        if content in seen_contents:
            continue
        validate_pair(pair)
        for row in pair:
            if family is not None and row["family"] != family:
                raise ValueError("named family disagrees with canonical pair")
            if train and any(row[key] != "train" for key in ("split", "world_partition", "name_split")):
                raise ValueError("replay requires original training admission and partitions")
        group = pair[0]["counterfactual_group"]
        if group in groups and groups[group] != content:
            raise ValueError("one pair identity has conflicting provenance")
        seen_contents.add(content)
        groups[group] = content
        pairs.append(pair)
    return pairs


def build_replay_banks(parent_train_arm, max_pairs=256):
    """Choose original pairs with balanced world, level and naming coverage.

    Selection first favors the least represented world, then level and naming
    map. It never invents a rendering or admits held-out data. Each family has
    the same number of selected pairs, bounded by its available unique pairs.
    """
    if type(max_pairs) is not int or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer")
    if not isinstance(parent_train_arm, dict) or set(parent_train_arm) != set(FAMILIES):
        raise ValueError("replay requires exactly the three original training families")
    candidates = {family: _unique_pairs(rows, train=True, family=family)
                  for family, rows in parent_train_arm.items()}
    count = min(max_pairs, *(len(pairs) for pairs in candidates.values()))
    banks = {}
    for family in FAMILIES:
        available = [(index, pair, pair[0]["world_fingerprint"], pair[0]["level"], _map_key(pair[0]))
                     for index, pair in enumerate(candidates[family])]
        worlds, levels, names, level_names = Counter(), Counter(), Counter(), Counter()
        selected = []
        for _ in range(count):
            chosen = min(available, key=lambda row: (
                worlds[row[2]], levels[row[3]], names[row[4]], level_names[row[3], row[4]], row[0]))
            available.remove(chosen)
            _, pair, world, level, name = chosen
            selected.extend(deepcopy(pair))
            worlds[world] += 1
            levels[level] += 1
            names[name] += 1
            level_names[level, name] += 1
        banks[family] = selected
    return banks


def _parent_rows(parent_banks):
    required = {"train", "development", "retained", "advanced", "support", "query"}
    if not isinstance(parent_banks, dict) or set(parent_banks) != required:
        raise ValueError("expected complete frozen diversity-study banks")
    for arms in parent_banks["train"].values():
        for rows in arms.values():
            yield from rows
    for group in required - {"train"}:
        for rows in parent_banks[group].values():
            yield from rows


def _worlds_from_pairs(pairs):
    worlds, seen = [], set()
    for pair in pairs:
        row = pair[0]
        if row["world_fingerprint"] in seen:
            continue
        worlds.append(generate_world_pair(row["family"], row["world_seed"], row["level"], row["world_requested_split"]))
        seen.add(row["world_fingerprint"])
    return worlds


def prepare_banks(parent_banks, *, with_diagnostics=False):
    """Build banks in memory, without reading models or writing study files."""
    if type(with_diagnostics) is not bool:
        raise ValueError("with_diagnostics must be boolean")
    original = list(_parent_rows(parent_banks))
    prior_variants = {normalized_transcript(row) for row in original}
    exact_prior = {_transcript(row) for row in original}
    forbidden_names = {_map_key(row) for row in original}
    selections, naming_rejections = [], []
    banks = {key: {} for key in ("development", "retained", "advanced", "support", "query")}

    common_worlds = {}
    for family in FAMILIES:
        low = _worlds_from_pairs(_unique_pairs(parent_banks["train"]["w32-n8"][family], train=True, family=family))
        high = _worlds_from_pairs(_unique_pairs(parent_banks["train"]["w256-n8"][family], train=True, family=family))
        if len(low) != 32 or not {w["world_fingerprint"] for w in low} <= {w["world_fingerprint"] for w in high}:
            raise ValueError("parents do not share the expected 32 familiar worlds")
        common_worlds[family] = low

    def new_worlds(family, seed, count, split, levels=(1, 2)):
        rows = world_pool(family, seed, count, split, levels=levels,
                          excluded_variants=prior_variants, diagnostics=selections)
        prior_variants.update(all_variants(rows))
        return rows

    def new_names(worlds, seed, admission, partition):
        """Avoid full-map reuse and any exact prior rendered transcript."""
        rendered, rejected, candidate_seed = [], 0, seed
        for world in worlds:
            for _ in range(10_000):
                mapping = _naming_key(candidate_seed, partition)
                chosen_seed = candidate_seed
                candidate_seed += 1
                if mapping in forbidden_names:
                    rejected += 1
                    continue
                pair = render_world_pair(world, chosen_seed, split=admission, name_split=partition)
                if any(_transcript(row) in exact_prior for row in pair):
                    rejected += 1
                    continue
                forbidden_names.add(mapping)
                exact_prior.update(_transcript(row) for row in pair)
                rendered.extend(pair)
                break
            else:
                raise ValueError("could not render a disjoint fresh naming pair")
        naming_rejections.append({"base_seed": seed, "split": admission, "name_split": partition,
                                  "pairs": len(worlds), "rejected_candidates": rejected})
        return rendered

    def familiar_names(worlds, admission):
        rows = [row for world in worlds for row in render_world_pair(
            world, TRAIN_NAME_SEEDS[0], split=admission, name_split="train")]
        if any(_transcript(row) in exact_prior for row in rows):
            raise ValueError("fresh world rendering overlaps an earlier exact transcript")
        exact_prior.update(_transcript(row) for row in rows)
        return rows

    for index, family in enumerate(FAMILIES):
        base = 51_000_000 + index * 100_000
        for group, split, offset in (("development", "dev", 0), ("retained", "audit", 20_000)):
            fresh = new_worlds(family, base + offset, 64, split)
            banks[group][f"names/{family}"] = new_names(common_worlds[family] * 2,
                54_000_000 + index * 100_000 + offset, split, split)
            banks[group][f"worlds/{family}"] = familiar_names(fresh, split)
            banks[group][f"both/{family}"] = new_names(fresh,
                55_000_000 + index * 100_000 + offset, split, split)
        fresh = new_worlds(family, base + 40_000, 64, "audit", levels=(3,))
        banks["advanced"][family] = new_names(fresh, 56_000_000 + index * 100_000, "audit", "audit")

    support = new_worlds(HELDOUT, 57_000_000, 128, "train")
    maps = {tuple(sorted(naming_map(seed, "train").items())) for seed in TRAIN_NAME_SEEDS}
    if len(maps) != 8:
        raise ValueError("arithmetic support requires eight distinct original training maps")
    banks["support"][HELDOUT] = [row for world in support for seed in TRAIN_NAME_SEEDS
        for row in render_world_pair(world, seed, split="train", name_split="train")]
    support_text = {_transcript(row) for row in banks["support"][HELDOUT]}
    if exact_prior & support_text:
        raise ValueError("arithmetic support overlaps earlier exact transcripts")
    exact_prior.update(support_text)
    familiar = [world for world in support if world["level"] == 2][:32]
    if len(familiar) != 32:
        raise ValueError("arithmetic support needs at least 32 level-two worlds")
    banks["query"]["names"] = new_names(familiar * 2, 58_000_000, "audit", "audit")
    fresh = new_worlds(HELDOUT, 59_000_000, 128, "audit", levels=(2,))
    banks["query"]["worlds"] = familiar_names(fresh, "audit")
    banks["query"]["both"] = new_names(fresh, 60_000_000, "audit", "audit")
    # Reserve a previously unused world partition, still sealed as final audit.
    advanced = new_worlds(HELDOUT, 61_000_000, 128, ADVANCED_ARITHMETIC_WORLD_SPLIT, levels=(3,))
    banks["query"]["advanced"] = new_names(advanced, 62_000_000, "audit", "audit")
    for group in banks.values():
        for rows in group.values():
            validate_diverse_evaluation_rows(rows, 64)
    diagnostics = {"world_selection": selections, "naming_selection": naming_rejections,
        "parent_unique_normalized_transcripts": len({normalized_transcript(row) for row in original}),
        "parent_unique_exact_transcripts": len({_transcript(row) for row in original}),
        "support_world_pairs": 128, "support_maps": 8, "support_episodes": 2048,
        "advanced_arithmetic_partition": {"world_partition": ADVANCED_ARITHMETIC_WORLD_SPLIT,
            "admission_split": "audit", "name_split": "audit",
            "reason": "Before optimization, the audit world partition failed to supply 128 new level-three pairs after 25,600 candidates with prior transcript exclusions. The previously unused dev world partition supplies the reserved final panel without changing the grammar or exclusions."},
        "note": "Naming panels deliberately share familiar worlds. Other new pools reject normalized single-member overlap with all prior-study rows and earlier new pools. Anonymous transcript identities are syntactic, not semantic equivalence classes."}
    return (banks, diagnostics) if with_diagnostics else banks


def bank_manifest(banks):
    """Compact immutable identity and denominator diagnostics for every bank."""
    return {group: {name: {"sha256": fingerprint_rows(rows), "episodes": len(rows),
        "counterfactual_pairs": len({row["counterfactual_group"] for row in rows}),
        "world_pairs": len({row["world_fingerprint"] for row in rows}),
        "naming_maps": len({_map_key(row) for row in rows}),
        "unique_transcripts": len({_transcript(row) for row in rows}),
        "unique_normalized_transcripts": len({normalized_transcript(row) for row in rows}),
        "levels": dict(sorted(Counter(str(row["level"]) for row in rows).items())),
        "query_target_counts": {str(target): sum(turn["target"] == target for row in rows for turn in row["turns"])
                                for target in range(3)}} for name, rows in values.items()}
        for group, values in banks.items()}
