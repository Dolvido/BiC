"""Prospective foundation evaluation cells; no models or historical-file reads.

Fresh panels preserve actual admitted procedures while changing both realization
coordinates. Held panels retain legacy query-ancestry partitions. Callers must
authenticate the supplied training manifest and source closure before a study.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path

from experiments import composition_curriculum as legacy
from experiments import foundation_curriculum as curriculum
from experiments.foundation_evidence import (
    SCHEMA as TRAINING_EVIDENCE_SCHEMA, json_digest, reconstruct_anchor, training_cell, transcript_set,
)
from experiments.foundation_plan import validate_plan


SCHEMA = "bic-foundation-evaluation-banks-v1"
MAX_ATTEMPTS_PER_PAIR = 1000
LENGTHS = (8, 10, 12)


def transcript_digest(row):
    """Exact observed text sequence only; no labels or provenance metadata."""
    return json_digest([turn["text"] for turn in row["turns"]])


def _hash_list(values, *, canonical=False):
    result = sorted(transcript_set(values))
    if canonical and (type(values) not in (list, tuple) or list(values) != result):
        raise ValueError("training transcripts must be sorted and unique")
    return result


def _seed(seed, *parts):
    return int(json_digest([SCHEMA, seed, *parts]), 16) % (2**63)


def expected_fresh_cells():
    return tuple(f"{family}/d{depth}/{operator}/t{turns}"
        for family in curriculum.FAMILIES for depth in curriculum.DEPTHS
        for operator in (("direct",) if depth == 0 else ("copy", "advance") if depth == 1 else ("composed",))
        for turns in LENGTHS)


def _anonymous(row):
    """Constants and ordered events with original roles; not semantic identity."""
    mapping = legacy.naming_map(row["recipe"]["naming_seed"])
    inverse = {name: role for role, name in mapping.items()}
    events, visible = [], set()
    for turn in row["turns"]:
        family, event = legacy.parse_sentence(turn["text"])
        event = dict(event)
        for key in ("name", "source"):
            if key in event:
                event[key] = inverse[event[key]]
                visible.add(event[key])
        events.append([family, event])
    return json_digest(events), json_digest(mapping), json_digest({role: mapping[role] for role in sorted(visible)})


def _fresh_relation(anchor, pair, reference):
    for original, row in zip(anchor, pair):
        if any(row[key] != original[key] for key in ("program_id", "structure_id", "query_ancestries", "depth", "primitive_shared")):
            raise ValueError("fresh realization changed the admitted procedure or query ancestry")
        if any(row["recipe"][key] != original["recipe"][key]
               for key in ("seed", "depth", "turns", "structure_split", "procedure_attempt")):
            raise ValueError("fresh realization changed its source recipe")
    old = [_anonymous(row) for row in anchor]
    new = [_anonymous(row) for row in pair]
    if old[0][1] == new[0][1] or old[0][2] == new[0][2]:
        return None, "unchanged_names"
    if {item[0] for item in old} & {item[0] for item in new}:
        return None, "unchanged_anonymous_values"
    return {"anchor": deepcopy(reference), "pair_sha256": json_digest(pair),
        "program_id": pair[0]["program_id"], "structure_id": pair[0]["structure_id"],
        "original_naming_map_sha256": old[0][1], "naming_map_sha256": new[0][1],
        "original_visible_names_sha256": old[0][2], "visible_names_sha256": new[0][2],
        "original_anonymous_values_sha256": [item[0] for item in old],
        "anonymous_values_sha256": [item[0] for item in new],
        "recipe": deepcopy(pair[0]["recipe"])}, None


def _stats(rows):
    turns = len(rows[0]["turns"])
    targets = [{str(target): sum(row["turns"][index]["target"] == target for row in rows)
                for target in range(4)} for index in range(turns)]
    opposite = [sum({rows[index]["turns"][turn]["target"], rows[index+1]["turns"][turn]["target"]} == {0, 1}
                    for index in range(0, len(rows), 2)) for turn in range(turns)]
    operators = Counter(legacy.parse_sentence(turn["text"])[1]["op"] for row in rows for turn in row["turns"])
    values = [_anonymous(row) for row in rows]
    return {"sha256": json_digest(rows), "version": curriculum.VERSION, "split": rows[0]["split"],
        "cell": training_cell(rows[0]), "episodes": len(rows), "complete_pairs": len(rows)//2,
        "turns": turns, "depth": rows[0]["depth"], "primitive_shared": rows[0]["primitive_shared"],
        "target_counts_by_turn": targets,
        "target_counts": {str(target): sum(t[str(target)] for t in targets) for target in range(4)},
        "query_target_counts": {str(target): sum(t[str(target)] for t in targets) for target in range(3)},
        "final_opposite_pair_total": opposite[-1], "opposite_pair_total": sum(opposite),
        "opposite_pair_total_by_turn": opposite,
        "observation_utf8_bytes": sum(len(turn["text"].encode("utf8")) for row in rows for turn in row["turns"]),
        "reply_utf8_bytes": sum(len(turn["reply"].encode("utf8")) for row in rows for turn in row["turns"]),
        "max_input_utf8_bytes": max(len(turn["text"].encode("utf8")) for row in rows for turn in row["turns"]),
        "observed_operator_counts": dict(sorted(operators.items())),
        "distinct_programs": len({row["program_id"] for row in rows}),
        "distinct_final_ancestries": len({row["structure_id"] for row in rows}),
        "distinct_naming_maps": len({value[1] for value in values}),
        "distinct_anonymous_value_transcripts": len({value[0] for value in values}),
        "unique_transcripts": len({transcript_digest(row) for row in rows})}


def build_evaluation(plan, training_manifest, *, training_transcripts, role="dev", seed,
                     pairs_per_cell, excluded_transcripts):
    """Return (canonical named banks, descriptive admission manifest).

    Requires enough distinct admitted anchors for every fresh cell. Each anchor
    is reconstructed and authenticated before use. Exclusions are whole observed
    transcripts, globally across this call, not only pair identities. Exhausted
    attempts fail without changing split, depth, operator or source procedure.
    """
    plan, training_manifest = deepcopy(plan), deepcopy(training_manifest)
    validate_plan(plan)
    if role not in ("dev", "audit"):
        raise ValueError("prospective dev or audit admission required")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2**63)")
    if type(pairs_per_cell) is not int or pairs_per_cell < 1:
        raise ValueError("positive complete-pair count required")
    training = _hash_list(training_transcripts, canonical=True)
    external = _hash_list(excluded_transcripts)
    if (not isinstance(training_manifest, dict) or training_manifest.get("schema") != TRAINING_EVIDENCE_SCHEMA
            or training_manifest.get("curriculum_version") != curriculum.VERSION
            or training_manifest.get("plan_sha256") != json_digest(plan)
            or training_manifest.get("unique_transcripts") != len(training)
            or training_manifest.get("transcript_sha256") != json_digest(training)):
        raise ValueError("training transcript list or plan differs from supplied manifest")
    anchors = training_manifest.get("anchors")
    expected = set(expected_fresh_cells())
    if not isinstance(anchors, dict) or set(anchors) != expected:
        raise ValueError("all family/depth/operator/length fresh anchor cells are required")
    admitted = {}
    training_set = set(training)
    for cell in sorted(expected):
        references = anchors[cell]
        if type(references) is not list or len(references) < pairs_per_cell:
            raise ValueError("insufficient independent anchors for fresh cell " + cell)
        seen = set()
        admitted[cell] = []
        for reference in references:
            pair = reconstruct_anchor(plan, reference)
            if training_cell(pair[0]) != cell or training_cell(pair[1]) != cell:
                raise ValueError("anchor is assigned to a different cell")
            digest = json_digest(pair)
            if digest in seen:
                raise ValueError("duplicate anchor pair in fresh cell")
            if not {transcript_digest(row) for row in pair} <= training_set:
                raise ValueError("reconstructed anchor is absent from authenticated training transcripts")
            seen.add(digest)
            admitted[cell].append((reference, pair))
    blocked = training_set | set(external)
    used, banks, relations, attempts, rejected = set(), {}, {}, {}, Counter()

    def accept(pair, cell):
        curriculum.validate_pair(pair)
        if any(training_cell(row) != cell or row["split"] != role for row in pair):
            raise ValueError("generated evaluation cell or admission differs")
        digests = {transcript_digest(row) for row in pair}
        if len(digests) != 2:
            raise ValueError("counterfactual observations are not distinct")
        reason = "excluded_transcript" if digests & blocked else "current_transcript" if digests & used else None
        if reason is not None:
            rejected[reason] += 1
            return False
        used.update(digests)
        return True

    for cell in sorted(expected):
        name = "fresh/" + cell
        banks[name], relations[name], attempts[name] = [], [], 0
        chosen = sorted(admitted[cell], key=lambda item: json_digest([seed, cell, item[0]]))[:pairs_per_cell]
        for slot, (reference, original) in enumerate(chosen):
            recipe = original[0]["recipe"]
            for attempt in range(MAX_ATTEMPTS_PER_PAIR):
                attempts[name] += 1
                pair = curriculum.generate_pair(original[0]["family"], recipe["seed"], depth=recipe["depth"],
                    turns=recipe["turns"], split=role, structure_split=recipe["structure_split"],
                    naming_seed=_seed(seed, role, cell, slot, attempt, "names"),
                    value_seed=_seed(seed, role, cell, slot, attempt, "values"))
                relation, reason = _fresh_relation(original, pair, reference)
                if reason is not None:
                    rejected[reason] += 1
                    continue
                if accept(pair, cell):
                    banks[name].extend(pair); relations[name].append(relation)
                    break
            else:
                raise ValueError("fresh foundation cell exhausted bounded attempts: " + cell)

    for family in curriculum.FAMILIES:
        for depth in range(3 if role == "dev" else 2, 6):
            for turns in LENGTHS:
                cell = f"{family}/d{depth}/composed/t{turns}"
                name = "composed/" + cell
                banks[name], attempts[name] = [], 0
                for slot in range(pairs_per_cell):
                    for attempt in range(MAX_ATTEMPTS_PER_PAIR):
                        attempts[name] += 1
                        try:
                            pair = curriculum.generate_pair(family, _seed(seed, role, cell, slot, attempt, "procedure"),
                                depth=depth, turns=turns, split=role, structure_split=role,
                                naming_seed=_seed(seed, role, cell, slot, attempt, "names"),
                                value_seed=_seed(seed, role, cell, slot, attempt, "values"))
                        except ValueError as error:
                            if not str(error).startswith("no admitted foundation procedure within"):
                                raise
                            rejected["unavailable_candidate_procedure"] += 1
                            continue
                        if accept(pair, cell):
                            banks[name].extend(pair)
                            break
                    else:
                        raise ValueError("held foundation cell exhausted bounded attempts: " + cell)
    manifest = {"schema": SCHEMA, "version": curriculum.VERSION, "role": role, "seed": seed,
        "pairs_per_cell": pairs_per_cell, "max_attempts_per_pair": MAX_ATTEMPTS_PER_PAIR,
        "plan_sha256": json_digest(plan), "training_manifest_sha256": json_digest(training_manifest),
        "training_transcripts": {"count": len(training), "sha256": json_digest(training)},
        "external_exclusions": {"count": len(external), "sha256": json_digest(external)},
        "effective_exclusions": {"count": len(blocked), "sha256": json_digest(sorted(blocked))},
        "transcript_sha256": sorted(used), "unique_transcripts": len(used),
        "banks": {name: _stats(rows) for name, rows in banks.items()},
        "fresh_anchor_relations": relations, "attempts_by_bank": attempts,
        "rejections": dict(sorted(rejected.items())),
        "source_sha256": {name: hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
            for name in ("experiments/foundation_banks.py", "experiments/foundation_evidence.py",
                         "experiments/foundation_plan.py", "experiments/foundation_curriculum.py",
                         "experiments/composition_curriculum.py")},
        "scope": "Exact individual transcript exclusions and legacy syntactic query-ancestry partitions; shared vocabulary and semantic equivalences remain. Fresh rows preserve admitted procedures with changed visible names and anonymous value-event transcripts. Caller authenticates training manifest and full frozen source closure; no historical files, models or scores were read."}
    return banks, manifest
