"""Balanced fresh samples for the prospective shared-representation diagnostic.

This module owns no learner, historical-file loader, evaluation or promotion.
Callers authenticate the exclusion inventory and pin sources before preparation.
The two roles use independent samples of the same train-admissible grammar, not
matched procedures or a new-algorithm test. No diagnostic files are read here.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

from experiments import foundation_curriculum as curriculum
from experiments.foundation_banks import expected_fresh_cells, transcript_digest, _anonymous
from experiments.foundation_evidence import json_digest, training_cell, transcript_set


SCHEMA = "bic-shared-diagnostic-data-v1"
ROLE_SEEDS = {"fit": 916170001, "evaluation": 916170002}
PAIRS_PER_CELL = 8
MAX_ATTEMPTS_PER_PAIR = 1000
CELLS = tuple(sorted(expected_fresh_cells()))


class DiagnosticAdmissionError(ValueError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = deepcopy(receipt)


def candidate_seeds(role, root_seed, cell, slot, attempt):
    """Independent, stable procedure/name/value coordinates; no RNG state."""
    if role not in ROLE_SEEDS or type(root_seed) is not int or not 0 <= root_seed < 2**63:
        raise ValueError("diagnostic role and bounded integer root seed required")
    if cell not in CELLS or any(type(value) is not int or value < 0 for value in (slot, attempt)):
        raise ValueError("canonical cell and nonnegative pair/attempt coordinates required")
    return {purpose: int(json_digest([SCHEMA, role, root_seed, cell, slot, attempt, purpose]), 16) % 2**63
            for purpose in ("procedure", "naming", "value")}


def _inventory(banks):
    seen = set()
    rows_by_cell, concepts = {}, {name: set() for name in ("program", "naming_map", "anonymous_values")}
    for cell, rows in sorted(banks.items()):
        targets = Counter(str(turn["target"]) for row in rows for turn in row["turns"])
        hashes = [transcript_digest(row) for row in rows]
        if seen.intersection(hashes) or len(set(hashes)) != len(hashes):
            raise ValueError("admitted diagnostic transcripts are not unique")
        seen.update(hashes)
        for row in rows:
            anonymous, naming, _ = _anonymous(row)
            concepts["program"].add(row["program_id"])
            concepts["naming_map"].add(naming)
            concepts["anonymous_values"].add(anonymous)
        rows_by_cell[cell] = dict(sha256=json_digest(rows), pairs=len(rows)//2, episodes=len(rows),
            turns=sum(len(row["turns"]) for row in rows),
            target_counts={str(target): targets[str(target)] for target in range(4)},
            observation_utf8_bytes=sum(len(turn["text"].encode("utf8")) for row in rows for turn in row["turns"]),
            reply_utf8_bytes=sum(len(turn["reply"].encode("utf8")) for row in rows for turn in row["turns"]))
    return rows_by_cell, sorted(seen), concepts


def build_split(role, *, excluded_transcripts, root_seed=None, pairs_per_cell=PAIRS_PER_CELL, cells=CELLS):
    """Build deterministic canonical samples; explicit smaller fixtures allowed.

    The production contract uses the defaults: all 63 cells and eight pairs each.
    Custom fixture settings are recorded, never relabelled as that contract.
    Exclusion collisions resample the entire pair within its declared cell.
    """
    if role not in ROLE_SEEDS:
        raise ValueError("fit or evaluation role required")
    root_seed = ROLE_SEEDS[role] if root_seed is None else root_seed
    if type(root_seed) is not int or not 0 <= root_seed < 2**63:
        raise ValueError("bounded integer root seed required")
    if type(pairs_per_cell) is not int or pairs_per_cell < 1:
        raise ValueError("positive integer pair count required")
    if type(cells) not in (tuple, list) or not cells or len(set(cells)) != len(cells) or any(cell not in CELLS for cell in cells):
        raise ValueError("explicit distinct canonical cells required")
    cells = tuple(sorted(cells))
    excluded = transcript_set(excluded_transcripts)
    seen, banks = set(), {}
    split = "train" if role == "fit" else "dev"
    counts = dict(candidate_pairs=0, generated_episode_candidates=0, canonical_pair_checks=0,
                  accepted_pairs=0, accepted_episodes=0, rejected_wrong_cell=0, rejected_collision=0)
    receipt = dict(schema=SCHEMA, status="preparing", role=role, split=split, root_seed=root_seed,
        cells=list(cells), pairs_per_cell=pairs_per_cell, max_attempts_per_pair=MAX_ATTEMPTS_PER_PAIR,
        production_contract=(cells == CELLS and pairs_per_cell == PAIRS_PER_CELL and root_seed == ROLE_SEEDS[role]),
        excluded_count=len(excluded), excluded_sha256=json_digest(sorted(excluded)), counts=counts,
        accepted_coordinates=[],
        scope="Fresh complete pairs from the known train-admissible grammar. Transcript exclusion is not semantic novelty. No model or score inputs.")
    try:
        for cell in cells:
            family, depth, _, turns = cell.split("/")
            depth, turns = int(depth[1:]), int(turns[1:])
            banks[cell] = []
            for slot in range(pairs_per_cell):
                accepted = False
                for attempt in range(MAX_ATTEMPTS_PER_PAIR):
                    receipt["current_coordinate"] = dict(cell=cell, pair_slot=slot, attempt=attempt)
                    seeds = candidate_seeds(role, root_seed, cell, slot, attempt)
                    counts["candidate_pairs"] += 1
                    pair = curriculum.generate_pair(family, seeds["procedure"], depth=depth, turns=turns,
                        split=split, naming_seed=seeds["naming"], value_seed=seeds["value"],
                        structure_split="shared" if depth < 2 else "train")
                    counts["generated_episode_candidates"] += len(pair)
                    curriculum.validate_pair(pair)
                    counts["canonical_pair_checks"] += 1
                    if any(training_cell(row) != cell for row in pair):
                        counts["rejected_wrong_cell"] += 1
                        continue
                    hashes = [transcript_digest(row) for row in pair]
                    if len(set(hashes)) != 2 or any(value in excluded or value in seen for value in hashes):
                        counts["rejected_collision"] += 1
                        continue
                    banks[cell].extend(pair)
                    seen.update(hashes)
                    counts["accepted_pairs"] += 1
                    counts["accepted_episodes"] += 2
                    receipt["accepted_coordinates"].append(dict(cell=cell, pair_slot=slot, attempt=attempt,
                        seeds=seeds, pair_sha256=json_digest(pair), transcript_sha256=hashes))
                    accepted = True
                    break
                if not accepted:
                    raise ValueError("diagnostic pair admission exhausted its fixed attempt allowance")
    except BaseException as error:
        receipt.update(status="failed", failure=repr(error))
        if isinstance(error, Exception):
            raise DiagnosticAdmissionError(str(error), receipt) from error
        # Interruptions retain no successful result; the owning runner records its intent.
        raise
    rows, hashes, _ = _inventory(banks)
    receipt.pop("current_coordinate", None)
    receipt.update(status="prepared", banks=rows, transcript_sha256=hashes,
        transcripts_sha256=json_digest(hashes), banks_sha256=json_digest(banks))
    return banks, receipt


def build_pair_of_splits(*, excluded_transcripts, roots=None, pairs_per_cell=PAIRS_PER_CELL, cells=CELLS):
    """Prepare fit then evaluation, protecting the complete fitting transcripts."""
    excluded = transcript_set(excluded_transcripts)
    roots = dict(ROLE_SEEDS) if roots is None else deepcopy(roots)
    if type(roots) is not dict or set(roots) != set(ROLE_SEEDS):
        raise ValueError("explicit fit and evaluation root seeds required")
    if any(type(value) is not int or not 0 <= value < 2**63 for value in roots.values()):
        raise ValueError("both role seeds must be bounded integers before preparation")
    if type(cells) not in (tuple, list):
        raise ValueError("explicit list or tuple of cells required")
    cells = tuple(cells)
    fit, fit_receipt = build_split("fit", excluded_transcripts=excluded, root_seed=roots["fit"],
        pairs_per_cell=pairs_per_cell, cells=cells)
    try:
        evaluation, evaluation_receipt = build_split("evaluation",
            excluded_transcripts=excluded | set(fit_receipt["transcript_sha256"]), root_seed=roots["evaluation"],
            pairs_per_cell=pairs_per_cell, cells=cells)
    except DiagnosticAdmissionError as error:
        raise DiagnosticAdmissionError(str(error), dict(schema=SCHEMA, status="failed", roles={
            "fit": fit_receipt, "evaluation": error.receipt})) from error
    _, _, fit_concepts = _inventory(fit)
    _, _, evaluation_concepts = _inventory(evaluation)
    overlap = {kind: dict(fit_distinct=len(fit_concepts[kind]), evaluation_distinct=len(evaluation_concepts[kind]),
        shared_distinct=len(fit_concepts[kind] & evaluation_concepts[kind])) for kind in fit_concepts}
    return dict(fit=fit, evaluation=evaluation), dict(schema=SCHEMA, status="prepared", roles={
        "fit": fit_receipt, "evaluation": evaluation_receipt}, overlap=overlap,
        production_contract=fit_receipt["production_contract"] and evaluation_receipt["production_contract"],
        source_and_exclusion_authentication="Owning runner requirement; supplied hashes alone do not prove origin.")
