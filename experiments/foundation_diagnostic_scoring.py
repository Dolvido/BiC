"""Pure counts for supplied diagnostic predictions; no model/tensor/file access.

The caller authenticates cache and prediction provenance. This module validates
the complete coordinate inventory and never infers pair membership from order.
Exact-reply predictions are generated token lists, including BOS and padding.
"""
from __future__ import annotations

import hashlib
import json
import re


SCHEMA = "bic-foundation-diagnostic-scoring-v1"
FEATURE_SCHEMA = "bic-foundation-diagnostic-features-v1"
FOUNDATION_CELLS = tuple(sorted(f"{family}/d{depth}/{operator}/t{turns}"
    for family in ("color", "count", "switch") for depth in range(6)
    for operator in (("direct",) if depth == 0 else ("copy", "advance") if depth == 1 else ("composed",))
    for turns in (8, 10, 12)))
REPLIES = ("No.", "Yes.", "I need more information.", "Understood.")
FIRST_TOKENS = tuple(ord(reply[0])+3 for reply in REPLIES)
_CONFIG_KEYS = {"width", "layers", "heads", "feedforward", "max_positions", "max_turns",
                "max_input_bytes", "max_output_bytes"}
_METADATA_KEYS = {"schema", "role", "cell_inventory", "cells", "kinds", "coordinates", "config",
                  "model_weights_sha256", "dataset_sha256", "bank_identities", "coordinate_order"}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _cell(name):
    match = re.fullmatch(r"(color|count|switch)/d([0-5])/(direct|copy|advance|composed)/t(8|10|12)", name) if type(name) is str else None
    if match is None:
        raise ValueError("canonical family/depth/operator/length cell required")
    family, depth, operator, turns = match.groups()
    depth, turns = int(depth), int(turns)
    if operator not in (("direct",) if depth == 0 else ("copy", "advance") if depth == 1 else ("composed",)):
        raise ValueError("cell operator disagrees with depth")
    return dict(family=family, depth=depth, operator=operator, turns=turns,
                panel="primitive" if depth < 2 else "composed")


def _validate(metadata, labels, inventory, pairs_per_cell):
    if (type(metadata) is not dict or set(metadata) != _METADATA_KEYS
            or metadata["schema"] != FEATURE_SCHEMA or metadata["role"] not in ("fit", "evaluation")):
        raise ValueError("exact feature-cache metadata schema and role required")
    if (type(inventory) not in (list, tuple) or not inventory or len(set(inventory)) != len(inventory)
            or any(type(cell) is not str for cell in inventory)):
        raise ValueError("explicit distinct cell inventory required")
    inventory = tuple(sorted(inventory))
    dimensions = {cell: _cell(cell) for cell in inventory}
    if type(pairs_per_cell) is not int or pairs_per_cell < 1:
        raise ValueError("positive exact pairs_per_cell required")
    if (type(metadata["cell_inventory"]) is not list
            or sorted(metadata["cell_inventory"]) != list(inventory)
            or type(metadata["coordinate_order"]) is not str):
        raise ValueError("cache inventory differs from declared scoring inventory")
    if not _sha(metadata["model_weights_sha256"]) or not _sha(metadata["dataset_sha256"]):
        raise ValueError("cache model and dataset byte identities required")
    config = metadata["config"]
    if (type(config) is not dict or set(config) != _CONFIG_KEYS
            or any(type(value) is not int or value < 1 for value in config.values())
            or config["max_turns"] < max(row["turns"] for row in dimensions.values())):
        raise ValueError("cache model configuration differs")
    banks = metadata["bank_identities"]
    if type(banks) is not dict or set(banks) != set(inventory):
        raise ValueError("exact per-cell bank identities required")
    for cell, identity in banks.items():
        if (type(identity) is not dict or set(identity) != {"sha256", "version", "episodes", "turns", "role", "config"}
                or not _sha(identity["sha256"]) or identity["version"] != "bic-shared-foundation-v1"
                or type(identity["episodes"]) is not int or identity["episodes"] != 2*pairs_per_cell
                or type(identity["turns"]) is not int or identity["turns"] != dimensions[cell]["turns"]
                or identity["role"] != ("train_fit" if metadata["role"] == "fit" else "dev")
                or _encoded(identity["config"]) != _encoded(config)):
            raise ValueError("bank coordinate/configuration/role identity differs")
    size = sum(2*pairs_per_cell*value["turns"] for value in dimensions.values())
    arrays = [metadata[key] for key in ("cells", "kinds", "coordinates")] + [labels]
    if any(type(value) is not list or len(value) != size for value in arrays):
        raise ValueError("one prediction/label record for every actual cell/pair/member/turn required")
    records = {}
    for index, (cell, kind, coordinate, label) in enumerate(zip(*arrays)):
        if type(cell) is not str or cell not in dimensions:
            raise ValueError("record cell is outside the admitted inventory")
        if (type(coordinate) is not list or len(coordinate) != 3
                or any(type(value) is not int for value in coordinate)):
            raise ValueError("canonical integer pair/member/turn coordinate required")
        pair, member, turn = coordinate
        if not (0 <= pair < pairs_per_cell and member in (0, 1) and 0 <= turn < dimensions[cell]["turns"]):
            raise ValueError("record coordinate is outside its complete cell")
        key = (cell, pair, member, turn)
        if key in records:
            raise ValueError("duplicate diagnostic turn coordinate")
        if (type(label) is not int or label not in range(4) or kind not in ("query", "statement")
                or type(kind) is not str or (label == 3) != (kind == "statement")):
            raise ValueError("true labels and canonical turn kinds disagree")
        records[key] = dict(index=index, label=label, **dimensions[cell])
    # Exact length, unique bounded coordinates and every cell's declared size
    # imply completeness, but explicitly check the semantic final-pair contract.
    for cell in inventory:
        for pair in range(pairs_per_cell):
            final = dimensions[cell]["turns"]-1
            if {records[(cell, pair, member, final)]["label"] for member in (0, 1)} != {0, 1}:
                raise ValueError("every final turn must be a complete opposite known-answer pair")
    return records, inventory


def _reply_prediction(tokens, label, max_bytes):
    if (type(tokens) is not list or not 1 <= len(tokens) <= max_bytes+2
            or any(type(value) is not int or not 0 <= value < 259 for value in tokens)):
        raise ValueError("generated reply must be a bounded list of 259-way token IDs")
    expected = [1, *(value+3 for value in REPLIES[label].encode("utf8")), 2]
    width = max(len(tokens), len(expected))
    correct = tokens+[0]*(width-len(tokens)) == expected+[0]*(width-len(expected))
    # Match ByteCodec's strict UTF-8 display interpretation for class confusion;
    # exactness above remains stricter about all special and post-EOS tokens.
    raw = []
    for token in tokens:
        if token == 2:
            break
        if token >= 3:
            raw.append(token-3)
    try:
        text = bytes(raw).decode("utf8", errors="strict")
    except UnicodeError:
        text = None
    action = REPLIES.index(text) if text in REPLIES else -1
    return action, correct


def _ratio(correct, total):
    return dict(correct=correct, total=total, accuracy=correct/total if total else None)


def _summarize(keys, records):
    keys = set(keys)
    rows = [records[key] for key in sorted(keys)]
    def accuracy(predicate):
        selected = [row for row in rows if predicate(row)]
        return _ratio(sum(row["correct"] for row in selected), len(selected))
    confusion, query_confusion = [[0]*5 for _ in range(4)], [[0]*5 for _ in range(4)]
    for row in rows:
        prediction = row["prediction"] if row["prediction"] >= 0 else 4
        confusion[row["label"]][prediction] += 1
        if row["label"] != 3:
            query_confusion[row["label"]][prediction] += 1
    opposite_total = opposite_correct = final_total = final_correct = 0
    for cell, pair, member, turn in sorted(keys):
        other = (cell, pair, 1, turn)
        if member != 0 or other not in keys:
            continue
        first, second = records[(cell, pair, 0, turn)], records[other]
        if {first["label"], second["label"]} == {0, 1}:
            opposite_total += 1
            opposite_correct += int(first["correct"] and second["correct"])
        if turn == first["turns"]-1:
            final_total += 1
            final_correct += int(first["correct"] and second["correct"])
    known_total = sum(row["label"] < 2 for row in rows)
    unsupported = sum(row["label"] < 2 and row["prediction"] == 2 for row in rows)
    ask_true = sum(row["label"] == 2 for row in rows)
    ask_predicted = sum(row["label"] != 3 and row["prediction"] == 2 for row in rows)
    ask_correct = sum(row["label"] == 2 and row["prediction"] == 2 for row in rows)
    return dict(turns=len(rows), episodes=len({key[:3] for key in keys}),
        all_turns=accuracy(lambda row: True), queries=accuracy(lambda row: row["label"] != 3),
        known=accuracy(lambda row: row["label"] < 2), unknown=accuracy(lambda row: row["label"] == 2),
        acknowledgements=accuracy(lambda row: row["label"] == 3),
        final_known=accuracy(lambda row: row["final"]), final_pairs=_ratio(final_correct, final_total),
        opposite_pairs=_ratio(opposite_correct, opposite_total),
        unsupported_ask=dict(count=unsupported, known_total=known_total,
                             rate=unsupported/known_total if known_total else None),
        ask=dict(true=ask_true, predicted_on_queries=ask_predicted, correct=ask_correct,
                 precision=ask_correct/ask_predicted if ask_predicted else None,
                 recall=ask_correct/ask_true if ask_true else None),
        confusion=dict(target_rows=[0, 1, 2, 3], prediction_columns=[0, 1, 2, 3, "invalid"],
                       all_turns=confusion, queries=query_confusion),
        invalid_predictions=sum(row["prediction"] == -1 for row in rows),
        per_target={str(label): accuracy(lambda row, label=label: row["label"] == label) for label in range(4)})


def _macro(groups):
    result = {}
    for metric in ("all_turns", "queries", "known", "unknown", "acknowledgements", "final_known", "final_pairs", "opposite_pairs", "unsupported_ask"):
        field = "rate" if metric == "unsupported_ask" else "accuracy"
        values = [row[metric][field] for row in groups.values() if row[metric][field] is not None]
        result[metric] = {field: sum(values)/len(values) if values else None,
                          "eligible_groups": len(values), "total_groups": len(groups)}
    return result


def score_predictions(metadata, labels, predictions, *, modality="action",
                      cell_inventory=FOUNDATION_CELLS, pairs_per_cell=8):
    """Score aligned Python lists; exact_reply takes full generated token rows.

    Explicit smaller canonical inventories/pair counts are fixtures, not the
    production 504-pair contract. Fit/evaluation are reported separately by role.
    """
    if modality not in ("action", "first_byte", "exact_reply"):
        raise ValueError("modality must be action, first_byte or exact_reply")
    # Freeze one plain JSON image before validation; never modify caller arrays.
    metadata, labels, predictions = json.loads(_encoded([metadata, labels, predictions]))
    records, inventory = _validate(metadata, labels, cell_inventory, pairs_per_cell)
    if type(predictions) is not list or len(predictions) != len(labels):
        raise ValueError("one aligned prediction per actual turn required")
    if modality == "exact_reply" and metadata["role"] != "evaluation":
        raise ValueError("full-reply diagnostic scoring is evaluation-only")
    for key, row in records.items():
        prediction = predictions[row["index"]]
        if modality == "exact_reply":
            action, correct = _reply_prediction(prediction, row["label"], metadata["config"]["max_output_bytes"])
        else:
            limit = 4 if modality == "action" else 259
            if type(prediction) is not int or not 0 <= prediction < limit:
                raise ValueError("prediction must be an unrestricted argmax ID in its vocabulary")
            action = prediction if modality == "action" else FIRST_TOKENS.index(prediction) if prediction in FIRST_TOKENS else -1
            correct = prediction == (row["label"] if modality == "action" else FIRST_TOKENS[row["label"]])
        row.update(prediction=action, correct=correct, final=key[3] == row["turns"]-1)
    def group(selector):
        groups = {}
        for key, row in records.items():
            name = selector(key, row)
            if name is not None:
                groups.setdefault(name, []).append(key)
        return {name: _summarize(keys, records) for name, keys in sorted(groups.items())}
    by_cell = group(lambda key, row: key[0])
    by_family = group(lambda key, row: row["family"])
    return dict(schema=SCHEMA, modality=modality, role=metadata["role"],
        model_weights_sha256=metadata["model_weights_sha256"], dataset_sha256=metadata["dataset_sha256"],
        metadata_sha256=_digest(metadata), labels_sha256=_digest(labels), predictions_sha256=_digest(predictions),
        cell_inventory=list(inventory), pairs_per_cell=pairs_per_cell,
        production_contract=inventory == FOUNDATION_CELLS and pairs_per_cell == 8,
        overall=_summarize(records, records), by_cell=by_cell, by_family=by_family,
        by_panel=group(lambda key, row: row["panel"]),
        by_primitive_operator=group(lambda key, row: row["operator"] if row["depth"] < 2 else None),
        by_family_primitive_operator=group(lambda key, row: row["family"]+"/"+row["operator"] if row["depth"] < 2 else None),
        by_depth=group(lambda key, row: "d"+str(row["depth"])),
        by_length=group(lambda key, row: "t"+str(row["turns"])),
        by_turn=group(lambda key, row: str(key[3])),
        macro_family=_macro(by_family), macro_cell=_macro(by_cell),
        scope="Pure metrics of supplied predictions; no model work or provenance attestation. Exact-reply correctness uses full tokens; reply confusion/ASK describe separately decoded semantic classes.")
