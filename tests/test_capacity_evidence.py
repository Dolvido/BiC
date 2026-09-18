"""Synthetic metric/selection checks; no training, GPU or live study results."""
import copy
from dataclasses import asdict
import unittest

import torch

from experiments.capacity_evidence import (
    EXPECTED_BANKS, RATES, WIDTHS, aggregate_rows, calibration_rank, select_rates,
    validate_audit_result, validate_metrics,
)
from experiments.composition_curriculum import VERSION, generate_pair
from experiments.composition_evaluation import METRICS, prediction_metrics
from experiments.realization_banks import _stats
from experiments.sequence_student import SequenceConfig
from experiments.summarize_composition_study import study as previous_study


def calibration(width=96, paired=16, reply=None, known=192, unsupported=0):
    reply = paired if reply is None else reply
    config = asdict(SequenceConfig(width=width, feedforward=width * 4, max_turns=12))
    half_correct, half_ask = known // 2, unsupported // 2
    confusion = [[half_correct, 128-half_correct-half_ask, half_ask, 0],
                 [128-half_correct-half_ask, half_correct, half_ask, 0], [0, 0, 64, 0], [0, 0, 0, 0]]
    row = {"bank": {"config": config}, "final_pairs": {"correct": paired, "total": 64},
        "final_reply_pair_correct": reply, "final_pair_accuracy": paired / 64,
        "final_reply_pair_accuracy": reply / 64, "known_correct": known, "known_total": 256,
        "known_accuracy": known / 256, "confusion_matrix": confusion, "free_running_replies": True,
        "teacher_used_for_policy": False, "decoder_prefix": "BOS only", "control": "normal"}
    return {"per_bank": {name: copy.deepcopy(row) for name in EXPECTED_BANKS}}


def jobs():
    return {f"w{width}-{label}": {"width": width, "rate": rate, "metrics": calibration(width)}
            for width in WIDTHS for label, rate in RATES.items()}


def scored_bank(width=192, replies=True, role="dev", name="composed/color/t8"):
    config = SequenceConfig(width=width, feedforward=width * 4, max_turns=12)
    rows = generate_pair("color", 1801001, split="train" if role == "train_fit" else role, turns=8)
    targets = torch.tensor([[turn["target"] for turn in row["turns"]] for row in rows])
    logits = torch.full((*targets.shape, 4), -2.)
    logits.scatter_(-1, targets.unsqueeze(-1), 2.)
    row = prediction_metrics(logits, targets,
        reply_correct=torch.ones_like(targets, dtype=torch.bool) if replies else None,
        reply_actions=targets.clone() if replies else None)
    manifest = _stats(rows)
    row.update(bank={"sha256": manifest["sha256"], "version": VERSION, "episodes": 2,
        "turns": 8, "role": role, "config": asdict(config)}, control="normal",
        teacher_used_for_policy=False, decoder_prefix="BOS only", free_running_replies=replies, seconds=.01)
    return aggregate_rows({name: row}), {name: rows}, {name: manifest}, config


def audit_fixture():
    audit, audit_rows, _, config = scored_bank(role="audit")
    initial, initial_rows, _, _ = scored_bank(role="train_fit", name="initial/color/t8")
    latest, latest_rows, _, _ = scored_bank(role="train_fit", name="latest/color/t8")
    inputs, metadata = {"protocol_sha256": "p" * 64}, {"occurrences": {8: [1]}}
    weights = {"0": "0" * 64, "600": "6" * 64}
    controls = {control: copy.deepcopy(audit) for control in ("blank", "reset")}
    for control, metrics in controls.items():
        for row in metrics["per_bank"].values():
            row["control"] = control
    result = {"schema": "bic-shared-capacity-study-v1", "job": "w192", "inputs": inputs,
        "curve": [{"updates": step, "episodes_per_family": step * 32,
                   "weights_sha256": weights[str(step)], "metrics": copy.deepcopy(audit)} for step in (0, 600)],
        "final": copy.deepcopy(audit), "initial_fit": initial, "latest_observed_fit": latest,
        "latest_observed_metadata": metadata, "controls": controls,
        "cpu_restart": {"exact": True, "device": "cpu", "weights_sha256": weights["600"], "utterances": 8},
        "seconds": 1., "automatic_promotion": False}
    arguments = {"job": "w192", "inputs": inputs, "steps": (0, 600), "config": config,
        "expected_weights": weights, "audit_rows": audit_rows, "initial_rows": initial_rows,
        "latest_rows": latest_rows, "latest_metadata": metadata}
    return result, arguments


class CapacityEvidenceTests(unittest.TestCase):
    def test_actual_width_validated_before_private_normalization(self):
        evidence, rows, manifests, config = scored_bank()
        before, old_config = copy.deepcopy(evidence), asdict(previous_study.CONFIG)
        self.assertTrue(validate_metrics(evidence, rows, manifests, config, "dev"))
        self.assertEqual(evidence, before)
        self.assertEqual(asdict(previous_study.CONFIG), old_config)
        with self.assertRaisesRegex(ValueError, "actual configuration"):
            validate_metrics(evidence, rows, manifests, SequenceConfig(max_turns=12), "dev")

    def test_canonical_manifest_and_count_corruption_rejected(self):
        evidence, rows, manifests, config = scored_bank()
        bad_manifest = copy.deepcopy(manifests)
        next(iter(bad_manifest.values()))["target_counts_by_turn"][-1]["0"] += 1
        with self.assertRaisesRegex(ValueError, "canonical rows"):
            validate_metrics(evidence, rows, bad_manifest, config, "dev")
        changed = copy.deepcopy(evidence)
        next(iter(changed["per_bank"].values()))["ask_predicted"] += 1
        with self.assertRaisesRegex(ValueError, "ASK"):
            validate_metrics(changed, rows, manifests, config, "dev")
        changed = copy.deepcopy(evidence)
        next(iter(changed["per_bank"].values()))["by_turn"][-1]["opposite_pair_total"] = 0
        with self.assertRaisesRegex(ValueError, "pair denominator"):
            validate_metrics(changed, rows, manifests, config, "dev")

    def test_probability_and_reply_plausibility_and_unmeasured_nulls(self):
        evidence, rows, manifests, config = scored_bank()
        for key, value in (("query_loss", float("nan")), ("brier_score", 2.1), ("reply_parseable_queries", 0)):
            changed = copy.deepcopy(evidence)
            next(iter(changed["per_bank"].values()))[key] = value
            with self.assertRaises(ValueError):
                validate_metrics(changed, rows, manifests, config, "dev")
        evidence, rows, manifests, config = scored_bank(replies=False)
        self.assertIsNone(evidence["macro_final_reply_pair_accuracy"])
        self.assertTrue(validate_metrics(evidence, rows, manifests, config, "dev", replies=False))
        next(iter(evidence["per_bank"].values()))["query_reply_correct"] = 0
        with self.assertRaisesRegex(ValueError, "unmeasured reply counts"):
            validate_metrics(evidence, rows, manifests, config, "dev", replies=False)

    def test_aggregate_equal_weight_and_missing_replies(self):
        first, second = dict.fromkeys(METRICS, .25), dict.fromkeys(METRICS, .75)
        second["final_reply_pair_accuracy"] = None
        result = aggregate_rows({"a": first, "b": second})
        self.assertEqual(result["macro_query_accuracy"], .5)
        self.assertIsNone(result["macro_final_reply_pair_accuracy"])
        result["per_bank"]["a"]["query_accuracy"] = 0
        self.assertEqual(first["query_accuracy"], .25)

    def test_rank_preserves_worst_cell_and_reply_evidence_not_unknowns(self):
        balanced = calibration(paired=16)
        high_average = calibration(paired=48)
        for panel in ("name_only", "value_only", "both", "composed"):
            row = high_average["per_bank"][f"{panel}/count/t12"]
            row.update(final_pairs={"correct": 0, "total": 64}, final_pair_accuracy=0.,
                       final_reply_pair_correct=0, final_reply_pair_accuracy=0.)
        self.assertGreater(calibration_rank(balanced), calibration_rank(high_average))
        self.assertEqual(calibration_rank(balanced), (.25, .25, .75, -0.))
        self.assertLess(calibration_rank(calibration(paired=16, reply=0)), calibration_rank(balanced))
        only_unknown = copy.deepcopy(balanced)
        for row in only_unknown["per_bank"].values():
            row["confusion_matrix"][2] = [64, 0, 0, 0]
        self.assertEqual(calibration_rank(only_unknown), calibration_rank(balanced))

    def test_known_and_unsupported_ask_are_later_tiebreakers(self):
        self.assertGreater(calibration_rank(calibration(known=192)), calibration_rank(calibration(known=128)))
        self.assertGreater(calibration_rank(calibration(known=128, unsupported=0)),
                           calibration_rank(calibration(known=128, unsupported=64)))

    def test_selection_requires_all9_and_applies_declared_tie_preference(self):
        inputs = jobs()
        selected = select_rates(inputs)
        self.assertEqual(selected["tie_preference"], [.001, .0003, .003])
        self.assertEqual({choice["rate"] for choice in selected["selected"].values()}, {.001})
        self.assertEqual(len(selected["candidates"]["w96-r001"]["cells"]), 9)
        self.assertFalse(selected["automatic_promotion"])
        del inputs["w96-r001"]
        with self.assertRaisesRegex(ValueError, "all9"):
            select_rates(inputs)

    def test_selection_rejects_incomplete_cells_config_and_rate_mismatch(self):
        for kind in ("cell", "config", "rate"):
            inputs = jobs()
            result = inputs["w192-r001"]
            if kind == "cell":
                result["metrics"]["per_bank"].pop("both/color/t8")
            elif kind == "config":
                result["metrics"]["per_bank"]["both/color/t8"]["bank"]["config"]["width"] = 96
            else:
                result["rate"] = .003
            with self.assertRaises(ValueError):
                select_rates(inputs)

    def test_complete_audit_is_read_only_and_normalizes_only_metadata_keys(self):
        result, arguments = audit_fixture()
        before = copy.deepcopy(result)
        result["latest_observed_metadata"] = {"occurrences": {"8": [1]}}
        self.assertTrue(validate_audit_result(result, **arguments))
        self.assertEqual(result["curve"], before["curve"])
        self.assertEqual(arguments["latest_metadata"], {"occurrences": {8: [1]}})

    def test_audit_cache_rejects_truncation_config_curve_and_final_corruption(self):
        original, arguments = audit_fixture()
        for mutation in ("missing_fit", "truncated_curve", "step", "exposure", "weight", "config", "final"):
            value = copy.deepcopy(original)
            if mutation == "missing_fit":
                del value["initial_fit"]
            elif mutation == "truncated_curve":
                value["curve"].pop()
            elif mutation == "step":
                value["curve"][1]["updates"] = 599
            elif mutation == "exposure":
                value["curve"][1]["episodes_per_family"] += 1
            elif mutation == "weight":
                value["curve"][1]["weights_sha256"] = "0" * 64
            elif mutation == "config":
                next(iter(value["curve"][1]["metrics"]["per_bank"].values()))["bank"]["config"]["width"] = 96
            else:
                value["final"]["macro_final_pair_accuracy"] = .5
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_audit_result(value, **arguments)

    def test_audit_cache_rejects_control_restart_latest_metadata_and_runtime_corruption(self):
        original, arguments = audit_fixture()
        for mutation in ("controls", "restart", "metadata", "elapsed", "identity", "promotion"):
            value = copy.deepcopy(original)
            if mutation == "controls":
                del value["controls"]["reset"]
            elif mutation == "restart":
                value["cpu_restart"]["weights_sha256"] = "0" * 64
            elif mutation == "metadata":
                value["latest_observed_metadata"]["occurrences"][8] = [2]
            elif mutation == "elapsed":
                value["seconds"] = 0
            elif mutation == "identity":
                value["inputs"]["protocol_sha256"] = "x" * 64
            else:
                value["automatic_promotion"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_audit_result(value, **arguments)


if __name__ == "__main__":
    unittest.main()
