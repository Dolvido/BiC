"""Synthetic summary arithmetic and closed-audit gate; no study data reads."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import diversity_study as study
from experiments.summarize_diversity_study import (
    AREA_KEYS, _verify_recipe, assemble_summary, compact, curve_areas, summarize, verify_completed,
)
from experiments.sequence_student import SequenceConfig


def metrics(names, accuracy, *, replies=True):
    last = {"total": 40, "correct": int(40 * accuracy), "accuracy": accuracy,
        "predicted_action_counts": [10, 10, 10, 10], "reply_exact_correct": 10 if replies else None,
        "reply_exact_accuracy": accuracy * .8 if replies else None,
        "pairs": {"total": 20, "correct": int(20 * accuracy), "accuracy": accuracy}}
    row = {"query_accuracy": accuracy, "query_correct": int(100 * accuracy), "query_total": 100,
        "counterfactual_accuracy": accuracy, "counterfactual_correct": int(40 * accuracy),
        "counterfactual_query_pairs": 40, "positions": {"last_known_later_query": last},
        "query_reply_exact_accuracy": accuracy * .8 if replies else None,
        "action_reply_agreement": accuracy * .9 if replies else None,
        "reply_parseable_query_count": 80 if replies else None,
        "ask_precision": .5, "ask_recall": .5, "ask_true": 20, "ask_predicted": 20, "brier_score": .2}
    return {**{key: accuracy for key in AREA_KEYS}, "per_bank": {name: copy.deepcopy(row) for name in names}}


def reports():
    families = study.FAMILIES
    panels = [f"{panel}/{family}" for panel in ("names", "worlds", "both") for family in families]
    protocol = {"seed": 2601, "interpretation": "Synthetic single-seed fixture."}
    main, audit_rows = {}, {}
    for index, arm in enumerate(study.ARMS):
        accuracy = (.1, .2, .3, .5)[index]
        main[arm] = {"updates": 3600, "episodes": 3600 * 64, "training_seconds": 12.5,
            "current_invocation_seconds_after_setup": 15., "peak_cuda_allocated_mib": None,
            "peak_scope": "synthetic CPU", "development": metrics(panels, accuracy),
            "fit_subset": metrics(families, .8, replies=False), "weights_sha256": arm}
    for index, name in enumerate((*study.ARMS, "fresh")):
        accuracy = .1 if name == "fresh" else .2 + index * .1
        curve = [{"updates": budget, "exposures": budget * 64,
                  **metrics(("level_2", "level_3"), accuracy + budget / 1000)}
                 for budget in (0, 1, 4, 16, 64)]
        audit_rows[name] = {"name": name, "curve": curve, "areas": curve_areas(curve),
            "retention_before": metrics(panels, accuracy), "retention_after": metrics(panels, 0.),
            "advanced_before": metrics(families, accuracy), "advanced_after": metrics(families, 0.),
            "controls": {kind: metrics(("level_2", "level_3"), .1) for kind in ("blank_text", "reset_history")},
            "training_seconds": 3., "cpu_restart": {"exact": True},
            "initial_weights_sha256": name + "initial", "final_weights_sha256": name + "final"}
    audit = {"results": audit_rows, "base_checkpoints_unchanged": True}
    return protocol, main, audit


class DiversitySummaryTests(unittest.TestCase):
    def test_nonuniform_update_auc_is_normalized_linear_trapezoid(self):
        curve = [{"updates": step, **{key: value for key in AREA_KEYS}}
                 for step, value in zip((0, 1, 4, 16, 64), (0., .2, .4, .6, .8))]
        expected = (1 * .1 + 3 * .3 + 12 * .5 + 48 * .7) / 64
        for area in curve_areas(curve).values():
            self.assertAlmostEqual(area, expected)
        for bad in (curve[1:], [curve[0], curve[0]], [{**row, "macro_pair_accuracy": None} for row in curve]):
            with self.assertRaises(ValueError):
                curve_areas(bad)

    def test_factorial_effect_direction_fixed_factors_and_fit_gaps(self):
        result = assemble_summary(*reports())
        expected = {"naming_at_32_worlds": .1, "naming_at_256_worlds": .2,
                    "worlds_at_1_naming_map": .2, "worlds_at_8_naming_maps": .3}
        for name, difference in expected.items():
            panel = result["development_factorial_effects"][name]["panels"]["both"]
            self.assertAlmostEqual(panel["macro_difference"]["query_accuracy"], difference)
            self.assertAlmostEqual(panel["per_family_difference"]["variable_binding"]["paired_accuracy"], difference)
            self.assertAlmostEqual(panel["macro_difference"]["query_reply_exact_accuracy"], difference * .8)
        gap = result["main"]["w32-n1"]["development_minus_fit"]["names"]["variable_binding"]
        self.assertAlmostEqual(gap["query_accuracy"], -.7)
        self.assertIsNone(gap["query_reply_exact_accuracy"])
        self.assertFalse(result["automatic_promotion"])

    def test_exposures_count_final_budget_once_and_reply_degradation_is_explicit(self):
        result = assemble_summary(*reports())
        compute = result["compute"]
        self.assertEqual(compute["main_updates"], 4 * 3600)
        self.assertEqual(compute["main_episode_exposures"], 4 * 3600 * 64)
        self.assertEqual(compute["adaptation_updates"], 5 * 64)
        self.assertEqual(compute["adaptation_episode_exposures"], 5 * 64 * 64)
        self.assertEqual(compute["main_worker_training_seconds"], 50.)
        self.assertEqual(compute["adaptation_training_seconds"], 15.)
        arm = result["adaptation"]["w32-n1"]
        self.assertAlmostEqual(result["prior_learning_area_gains_vs_fresh"]["w32-n1"]["macro_query_accuracy"], .1)
        later = arm["final_later_known_by_level"]["level_3"]
        self.assertEqual(later["pairs"]["total"], 20)
        self.assertAlmostEqual(later["pairs"]["accuracy"], .264)
        reply = arm["reply_diagnostics"]["both/variable_binding"]
        self.assertEqual(reply["after_exact_accuracy"], 0.)
        self.assertAlmostEqual(reply["exact_accuracy_change"], -.16)
        self.assertEqual(reply["unparseable_query_replies_after"], 20)
        self.assertIn("cannot be established", result["reply_note"])

    def test_missing_denominators_remain_none(self):
        source = metrics(("level_2",), .5, replies=False)
        source["per_bank"]["level_2"]["positions"]["last_known_later_query"]["pairs"] = {
            "total": 0, "correct": 0, "accuracy": None}
        compacted = compact(source)["per_bank"]["level_2"]
        self.assertIsNone(compacted["later_known_pair_accuracy"])
        self.assertIsNone(compacted["reply_parseable_query_rate"])
        self.assertIsNone(compacted["query_reply_exact_accuracy"])

    def test_recipe_verification_rejects_changed_model_budget_and_actual_optimizer(self):
        config = SequenceConfig()
        protocol = {"batch_size": 64, "learning_rate": .001}
        banks, options = {"family": "fingerprint"}, {"lr": .001, "betas": (.9, .999)}
        training = {"recipe": {"model": "bic-diversity-sequence-v1", "banks": banks, "seed": 2601,
            "config": asdict(config), "batch_size": 64, "learning_rate": .001, "gradient_clip": 1.,
            "optimizer": {"name": "AdamW", "options": options}},
            "optimizer": {"param_groups": [{"params": [0], **options}]}}
        _verify_recipe(training, protocol, banks, 2601, config)
        for kind in ("seed", "batch", "config", "optimizer"):
            bad = copy.deepcopy(training)
            if kind == "seed":
                bad["recipe"]["seed"] += 1
            elif kind == "batch":
                bad["recipe"]["batch_size"] = 32
            elif kind == "config":
                bad["recipe"]["config"]["max_input_bytes"] = 64
            else:
                bad["optimizer"]["param_groups"][0]["lr"] = .003
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                _verify_recipe(bad, protocol, banks, 2601, config)

    def test_incomplete_audit_gate_reads_no_results_or_checkpoints_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with patch("experiments.summarize_diversity_study.study.load_protocol") as load, \
                    patch("experiments.summarize_diversity_study.torch.load") as checkpoint:
                with self.assertRaisesRegex(ValueError, "completed audit"):
                    summarize(path)
                with self.assertRaisesRegex(ValueError, "all four"):
                    verify_completed(path)
                load.assert_not_called()
                checkpoint.assert_not_called()
            self.assertFalse((path / "audit" / "summary.json").exists())

    def test_parent_digest_mismatch_rejected_before_checkpoint_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "audit").mkdir()
            (path / "protocol.json").write_text("{}", encoding="utf8")
            for arm in study.ARMS:
                main = path / "main" / arm
                main.mkdir(parents=True)
                for name in ("report.json", "latest.pt", "initial.pt"):
                    (main / name).write_text("{}", encoding="utf8")
            candidates = (*study.ARMS, "fresh")
            for name in candidates:
                for filename in (f"{name}.json", f"{name}-adapted.pt"):
                    (path / "audit" / filename).write_text("{}", encoding="utf8")
            marker = {"schema": study.SCHEMA, "parent_sha256": {arm: "changed" for arm in study.ARMS},
                      "protocol_sha256": study.file_hash(path / "protocol.json")}
            (path / "audit" / "evaluation-started.json").write_text(json.dumps(marker), encoding="utf8")
            audit = {"schema": study.SCHEMA, "inputs": marker, "results": {name: {} for name in candidates},
                     "base_checkpoints_unchanged": True, "automatic_promotion": False}
            (path / "audit" / "report.json").write_text(json.dumps(audit), encoding="utf8")
            with patch("experiments.summarize_diversity_study.study.load_protocol", return_value={}), \
                    patch("experiments.summarize_diversity_study.torch.load") as checkpoint:
                with self.assertRaisesRegex(ValueError, "provenance"):
                    summarize(path)
                checkpoint.assert_not_called()
            self.assertFalse((path / "audit" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
