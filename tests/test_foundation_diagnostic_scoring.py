"""Synthetic Python-list metric checks: no tensor/model/probe execution."""
import copy
import json
import os
from pathlib import Path
import sys
import time
import unittest

from experiments import foundation_diagnostic_scoring as scoring


SMALL = ("color/d0/direct/t8", "count/d1/copy/t10", "switch/d1/advance/t12", "switch/d2/composed/t12")


def fixture(cells=SMALL, pairs=2, role="evaluation"):
    config = dict(width=4, layers=1, heads=1, feedforward=8, max_positions=4096,
                  max_turns=12, max_input_bytes=128, max_output_bytes=32)
    metadata = dict(schema=scoring.FEATURE_SCHEMA, role=role, cell_inventory=list(cells),
        cells=[], kinds=[], coordinates=[], config=config, model_weights_sha256="a"*64,
        dataset_sha256="b"*64, bank_identities={},
        coordinate_order="cell inventory, pair slot, member zero/one, actual turn")
    labels = []
    for cell in cells:
        turns = int(cell.rsplit("t", 1)[1])
        metadata["bank_identities"][cell] = dict(sha256="c"*64, version="bic-shared-foundation-v1",
            episodes=pairs*2, turns=turns, role="train_fit" if role == "fit" else "dev", config=copy.deepcopy(config))
        for pair in range(pairs):
            for member in (0, 1):
                for turn in range(turns):
                    label = 2 if turn == 1 else member if turn == 2 else member ^ (pair % 2) if turn == turns-1 else 3
                    metadata["cells"].append(cell)
                    metadata["kinds"].append("statement" if label == 3 else "query")
                    metadata["coordinates"].append([pair, member, turn])
                    labels.append(label)
    return metadata, labels


def tokens(label):
    value = [1, *(byte+3 for byte in scoring.REPLIES[label].encode("utf8")), 2]
    return value+[0]*(34-len(value))


class DiagnosticScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started = time.monotonic()
        cls.calls = cls.completed = cls.rows = 0

    @classmethod
    def tearDownClass(cls):
        value = dict(schema="bic-diagnostic-scoring-pure-work-v1", scoring_attempts=cls.calls,
            completed_scoring_calls=cls.completed, completed_prediction_rows=cls.rows,
            wall_seconds=time.monotonic()-cls.started, model_calls=0, probe_fits=0,
            optimizer_updates=0, tensor_work=0, historical_or_formal_evidence_loaded=False,
            scope="Synthetic JSON/list metric verification only; no learned predictions.")
        print("DIAGNOSTIC_SCORING_PURE_WORK="+json.dumps(value, sort_keys=True))
        path = os.environ.get("BIC_DIAGNOSTIC_SCORING_ACCOUNTING")
        if path:
            with Path(path).open("x", encoding="utf8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True)

    def score(self, metadata, labels, predictions, **kwargs):
        type(self).calls += 1
        result = scoring.score_predictions(metadata, labels, predictions, **kwargs)
        type(self).completed += 1
        type(self).rows += len(labels)
        return result

    def small_score(self, metadata, labels, predictions, **kwargs):
        return self.score(metadata, labels, predictions, cell_inventory=SMALL, pairs_per_cell=2, **kwargs)

    def test_production_denominators_and_complete_reporting(self):
        metadata, labels = fixture(scoring.FOUNDATION_CELLS, 8)
        result = self.score(metadata, labels, labels)
        self.assertTrue(result["production_contract"])
        self.assertEqual(result["overall"]["final_pairs"], dict(correct=504, total=504, accuracy=1.))
        self.assertEqual(result["overall"]["final_known"]["total"], 1008)
        self.assertEqual(result["overall"]["turns"], 10080)
        self.assertEqual(len(result["by_cell"]), 63)
        self.assertEqual(result["by_panel"]["primitive"]["final_pairs"]["total"], 216)
        self.assertEqual(result["by_panel"]["composed"]["final_pairs"]["total"], 288)
        self.assertEqual({row["final_pairs"]["total"] for row in result["by_family"].values()}, {168})
        self.assertEqual({row["final_pairs"]["total"] for row in result["by_family_primitive_operator"].values()}, {24})
        self.assertEqual(set(result["by_depth"]), {"d"+str(i) for i in range(6)})
        self.assertEqual(set(result["by_length"]), {"t8", "t10", "t12"})
        self.assertEqual(result["macro_family"]["final_pairs"]["accuracy"], 1.)

    def test_pair_failure_actual_turn_counts_and_order_independence(self):
        metadata, labels = fixture()
        predictions = list(labels)
        # Fail only member zero of the first pair's final known query.
        predictions[7] = 2
        result = self.small_score(metadata, labels, predictions)
        overall = result["overall"]
        self.assertEqual(overall["final_pairs"]["correct"], 7)
        self.assertEqual(overall["final_known"]["correct"], 15)
        self.assertEqual(overall["known"]["total"], 32)
        self.assertEqual(overall["unknown"]["total"], 16)
        self.assertEqual(overall["queries"]["total"], 48)
        self.assertEqual(overall["acknowledgements"]["total"], len(labels)-48)
        self.assertEqual(overall["unsupported_ask"], dict(count=1, known_total=32, rate=1/32))
        self.assertEqual(overall["confusion"]["queries"][0][2], 1)
        self.assertEqual(overall["ask"]["precision"], 16/17)
        reordered = copy.deepcopy(metadata)
        for name in ("cells", "kinds", "coordinates"):
            reordered[name].reverse()
        other = self.small_score(reordered, list(reversed(labels)), list(reversed(predictions)))
        for key in ("overall", "by_cell", "by_family", "by_panel", "by_turn", "macro_family"):
            self.assertEqual(result[key], other[key])

    def test_equal_family_macro_differs_from_pooled_unequal_inventory(self):
        metadata, labels = fixture()
        predictions = [label if cell.startswith("switch/") else 3 for cell, label in zip(metadata["cells"], labels)]
        result = self.small_score(metadata, labels, predictions)
        self.assertEqual(result["overall"]["final_pairs"]["accuracy"], .5)
        self.assertEqual(result["macro_family"]["final_pairs"], dict(accuracy=1/3, eligible_groups=3, total_groups=3))

    def test_first_byte_scores_unrestricted_token_ids_and_keeps_invalid_column(self):
        metadata, labels = fixture()
        predictions = [scoring.FIRST_TOKENS[label] for label in labels]
        for index, token in zip((0, 1, 2, 7), (0, 1, 2, ord("N"))):
            predictions[index] = token
        result = self.small_score(metadata, labels, predictions, modality="first_byte")
        self.assertEqual(result["overall"]["all_turns"]["correct"], len(labels)-4)
        self.assertEqual(result["overall"]["invalid_predictions"], 4)
        self.assertEqual(result["overall"]["final_pairs"]["correct"], 7)
        self.assertEqual(sum(row[4] for row in result["overall"]["confusion"]["all_turns"]), 4)

    def test_exact_reply_requires_eos_and_exact_tokens_but_ask_uses_decoded_class(self):
        metadata, labels = fixture()
        predictions = [tokens(label) for label in labels]
        # A parseable but inexact ASK on a known query still counts as unsupported ASK.
        predictions[7] = [1, 0]+tokens(2)[1:-1]
        # Missing EOS leaves a parseable display string but cannot be an exact reply.
        predictions[2][predictions[2].index(2)] = 0
        # Trailing non-PAD after EOS also breaks exactness without changing display.
        predictions[3][-1] = ord("x")+3
        result = self.small_score(metadata, labels, predictions, modality="exact_reply")
        self.assertEqual(result["overall"]["all_turns"]["correct"], len(labels)-3)
        self.assertEqual(result["overall"]["final_pairs"]["correct"], 7)
        self.assertEqual(result["overall"]["invalid_predictions"], 0)
        self.assertEqual(result["overall"]["unsupported_ask"]["count"], 1)
        self.assertEqual(result["overall"]["confusion"]["queries"][0][2], 1)
        metadata, labels = fixture(role="fit")
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            self.small_score(metadata, labels, [tokens(label) for label in labels], modality="exact_reply")

    def test_fit_scores_are_explicit_and_results_do_not_mutate_inputs(self):
        metadata, labels = fixture(role="fit")
        original = copy.deepcopy([metadata, labels])
        result = self.small_score(metadata, labels, labels)
        self.assertEqual(result["role"], "fit")
        result["cell_inventory"].clear()
        result["by_cell"].clear()
        self.assertEqual([metadata, labels], original)
        self.assertNotIn("torch", sys.modules)

    def test_incomplete_duplicate_or_nonopposite_coordinates_fail(self):
        metadata, labels = fixture()
        changed = copy.deepcopy(metadata)
        changed["coordinates"][1] = changed["coordinates"][0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.small_score(changed, labels, labels)
        changed = copy.deepcopy(metadata)
        changed["coordinates"][0][0] = 2
        with self.assertRaisesRegex(ValueError, "outside"):
            self.small_score(changed, labels, labels)
        changed_labels = list(labels)
        changed_labels[15] = 0
        with self.assertRaisesRegex(ValueError, "opposite"):
            self.small_score(metadata, changed_labels, labels)
        with self.assertRaises(ValueError): self.small_score(metadata, labels[:-1], labels[:-1])
        changed = copy.deepcopy(metadata)
        changed["bank_identities"][SMALL[0]]["episodes"] = 2
        with self.assertRaisesRegex(ValueError, "bank"):
            self.small_score(changed, labels, labels)

    def test_invalid_boolean_labels_predictions_and_vocabulary_are_rejected(self):
        metadata, labels = fixture()
        for invalid in (True, 4, -1, .5):
            predictions = list(labels)
            predictions[0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.small_score(metadata, labels, predictions)
        changed = list(labels)
        changed[0] = True
        with self.assertRaises(ValueError): self.small_score(metadata, changed, labels)
        with self.assertRaises(ValueError): self.small_score(metadata, labels, [259]*len(labels), modality="first_byte")
        with self.assertRaises(ValueError): self.small_score(metadata, labels, [True]*len(labels), modality="exact_reply")
        changed = copy.deepcopy(metadata)
        changed["kinds"][0] = "query"
        with self.assertRaises(ValueError): self.small_score(changed, labels, labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
