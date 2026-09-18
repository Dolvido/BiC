"""Synthetic calculations and strict completion gates; no live study reads."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import realization_study as study
from experiments.composition_curriculum import VERSION
from experiments.composition_evaluation import METRICS, prediction_metrics
from experiments.realization_training import COUNTS, stream_evidence
from experiments.summarize_realization_study import (
    _required, _same, _same_tensors, _validate_endpoint_binding, _validate_scores,
    assemble_summary, summarize, verify_completed,
)


def metrics(value, *, panels=("name_only", "value_only", "both", "composed"), replies=True):
    rows = {}
    for panel in panels:
        for family in study.FAMILIES:
            for turns in (8, 10, 12):
                row = {**dict.fromkeys(METRICS, value), "query_total": 320,
                    "known_total": 256, "final_pairs": {"correct": 16, "total": 64},
                    "ask_precision": None, "ask_recall": None, "action_reply_agreement": .9 if replies else None,
                    "query_loss": .4, "brier_score": .3, "per_target": {"2": {"accuracy": None, "total": 0, "correct": 0}}}
                if not replies:
                    row.update(query_reply_accuracy=None, final_reply_pair_accuracy=None)
                rows[f"{panel}/{family}/t{turns}"] = row
    return {"per_bank": rows, **{f"macro_{key}": next(iter(rows.values()))[key] for key in METRICS}}


def reports():
    protocol = {"schema": study.SCHEMA, "interpretation": "Synthetic fixture"}
    main, results, timings = {}, {}, {}
    for job, value in (("fixed", .2), ("fresh", .3)):
        fresh = job == "fresh"
        exposures = {family: dict(zip(COUNTS, (115200, 1152000, 20_000_000 + int(fresh),
            17_696_000 + int(fresh), 10_000_000, 8_848_000))) for family in study.FAMILIES}
        stream = {"sampler_sha256": dict.fromkeys(study.FAMILIES, "same"),
            "occurrences": {family: {8: [1800, 1800]} for family in study.FAMILIES},
            "family_microbatches": dict.fromkeys(study.FAMILIES, 3600),
            "bucket_microbatches": {family: {8: 3600} for family in study.FAMILIES},
            "structural_stream_sha256": "same", "canonical_stream_sha256": job,
            "exposures": exposures, "collisions": {"rejected_candidates": 2 if fresh else 0}}
        main[job] = {"updates": 3600, "exposures": exposures, "stream": stream,
            "training_step_seconds": 70. if fresh else 40., "step_time_scope": "inclusive steps",
            "invocation_seconds_after_preparation": 80. if fresh else 50., "peak_cuda_allocated_mib": None,
            "weights_sha256": job, "history": [{"updates": step, "training_step_seconds": step / 100.,
                "development": metrics(value, replies=False)} for step in study.STEPS]}
        curve = [{"updates": step, "episodes_per_family": step * study.MICRO,
            "weights_sha256": f"{job}-{step}", "metrics": metrics(value + .1 * step / study.TOTAL)} for step in study.STEPS]
        results[job] = {"curve": curve, "final": curve[-1]["metrics"],
            "initial_realization_fit": metrics(.9 if not fresh else .7, panels=("initial",)),
            "latest_observed_fit": metrics(.9 if not fresh else .8, panels=("latest",)),
            "latest_observed_metadata": {"all_initial_realizations_encountered": True,
                "absent_recipes": [], "per_pair": [{"family": "color", "turns": 8, "recipe_index": 0,
                    "occurrence": 9, "attempt": 1, "pair_sha256": "fixture"}]},
            "controls": {name: metrics(.05) for name in ("blank", "reset")},
            "cpu_restart": {"exact": True}, "seconds": 5.}
        timings[job] = {"generation_validation_seconds": 20. if fresh else 3.,
                        "rejected_candidate_seconds": 1. if fresh else 0.}
    return protocol, main, {"results": results}, timings


class RealizationSummaryTests(unittest.TestCase):
    def test_exact_budget_and_generation_cost_is_not_added_twice(self):
        summary = assemble_summary(*reports())
        self.assertEqual(summary["compute"]["retained_optimizer_updates"], 7200)
        self.assertEqual(summary["compute"]["sampled_exposures"]["episodes"], 691200)
        self.assertEqual(summary["compute"]["worker_training_step_seconds"], 110.)
        self.assertEqual(summary["compute"]["generation_validation_seconds_already_in_steps"], 23.)
        self.assertEqual(summary["fresh_minus_fixed"]["step_seconds_difference"], 30.)
        self.assertIn("nested subsets", summary["time_note"])
        self.assertFalse(summary["automatic_promotion"])

    def test_panel_family_length_curve_differences_preserve_realized_byte_changes(self):
        summary = assemble_summary(*reports())
        comparison = summary["fresh_minus_fixed"]
        for panel in ("name_only", "value_only", "both", "composed"):
            self.assertAlmostEqual(comparison["endpoint_difference"][panel]["color"]["per_length"]["t12"]["final_pair_accuracy"], .1)
            self.assertAlmostEqual(comparison["area_difference_by_family"]["count"][panel]["query_accuracy"], .1)
        self.assertEqual(comparison["actual_exposure_difference"]["color"]["episodes"], 0)
        self.assertEqual(comparison["actual_exposure_difference"]["color"]["observation_bytes"], 1)
        self.assertIn("not establish an additive", summary["comparison_note"])

    def test_fitting_coverage_denominators_and_missing_unknown_metrics_remain_explicit(self):
        protocol, main, audit, timings = reports()
        audit["results"]["fresh"]["latest_observed_metadata"].update(all_initial_realizations_encountered=False,
            absent_recipes=[{"family": "switch", "turns": 12, "recipe_index": 2}])
        summary = assemble_summary(protocol, main, audit, timings)
        fresh = summary["audit"]["fresh"]
        self.assertFalse(fresh["latest_observed_metadata"]["all_initial_realizations_encountered"])
        self.assertEqual(len(fresh["latest_observed_metadata"]["absent_recipes"]), 1)
        row = fresh["final"]["per_panel_family"]["both"]["color"]["per_length"]["t8"]
        self.assertEqual(row["final_pairs"]["total"], 64)
        self.assertEqual(row["known_total"], 256)
        self.assertIsNone(row["ask_recall"])
        self.assertIsNone(fresh["final"]["per_panel_family"]["both"]["color"]["macro"]["ask_precision"])
        self.assertIn("not the complete fresh stream", summary["fit_note"])

    def test_matching_rejects_structural_changes_but_json_bucket_roundtrip_is_valid(self):
        protocol, main, audit, timings = reports()
        main["fresh"]["stream"] = json.loads(json.dumps(main["fresh"]["stream"]))
        assemble_summary(protocol, main, audit, timings)
        main["fresh"]["stream"]["occurrences"]["color"]["8"][0] += 1
        with self.assertRaisesRegex(ValueError, "evidence differs"):
            assemble_summary(protocol, main, audit, timings)
        _same({8: 1, 10: 2, 12: 3}, {"8": 1, "10": 2, "12": 3})
        first = {"moment": torch.tensor([1., 2.]), "sampler": torch.tensor([1, 2], dtype=torch.uint8)}
        _same_tensors(first, copy.deepcopy(first))
        changed = copy.deepcopy(first)
        changed["moment"][1] += 1
        with self.assertRaisesRegex(ValueError, "tensor state differs"):
            _same_tensors(first, changed)

    def test_canonical_pairs_validate_fit_and_audit_denominators(self):
        truth = torch.tensor([[3, 2, 0, 0], [3, 2, 0, 1]])
        rows = [{"turns": [{"target": value} for value in values]} for values in truth.tolist()]
        name = "latest/color/t4"
        row = prediction_metrics(torch.nn.functional.one_hot(truth, 4).float() * 4, truth,
            reply_correct=torch.ones_like(truth, dtype=torch.bool), reply_actions=truth)
        manifest = {"sha256": "fixture", "episodes": 2, "complete_pairs": 1, "turns": 4,
            "target_counts_by_turn": [{str(target): int(truth[:, turn].eq(target).sum()) for target in range(4)} for turn in range(4)]}
        row.update(bank={"sha256": "fixture", "version": VERSION, "episodes": 2, "turns": 4,
            "role": "train_fit", "config": asdict(study.CONFIG)}, teacher_used_for_policy=False,
            decoder_prefix="BOS only", free_running_replies=True, control="normal")
        scores = {"per_bank": {name: row}, **{f"macro_{key}": row[key] for key in METRICS}}
        _validate_scores(scores, {name: rows}, {name: manifest}, role="train_fit")
        scores["per_bank"][name]["by_turn"][2]["opposite_pair_total"] = 1
        with self.assertRaisesRegex(ValueError, "opposite-pair"):
            _validate_scores(scores, {name: rows}, {name: manifest}, role="train_fit")

    def test_endpoint_canonical_stream_and_adamw_tampering_cannot_hide_behind_same_weights(self):
        latest = {"weights": {"value": torch.tensor([1.])},
            "optimizer": {"state": {0: {"exp_avg": torch.tensor([.1]), "exp_avg_sq": torch.tensor([.2])}}},
            "samplers": {"color": torch.tensor([1, 2], dtype=torch.uint8)},
            "realization": {"canonical_stream_sha256": "0" * 64, "structural_stream_sha256": "1" * 64,
                "statistics": {"color": {8: {"targets": [2, 2, 2, 2]}}},
                "samplers": {"color": torch.tensor([1, 2], dtype=torch.uint8)},
                "seen_transcripts": [], "timing": {"generation_validation_seconds": 1.}}}
        checkpoint = copy.deepcopy(latest)
        replay_total = stream_evidence(latest["realization"])
        _validate_endpoint_binding(latest, checkpoint, replay_total)
        changed = copy.deepcopy(latest)
        changed["realization"]["canonical_stream_sha256"] = "f" * 64
        # Even mirroring the changed stream into another endpoint record does
        # not let it replace the authenticated replay's original stream.
        with self.assertRaisesRegex(ValueError, "evidence differs"):
            _validate_endpoint_binding(changed, copy.deepcopy(changed), replay_total)
        changed = copy.deepcopy(latest)
        changed["optimizer"]["state"][0]["exp_avg"][0] += 1
        self.assertTrue(torch.equal(changed["weights"]["value"], checkpoint["weights"]["value"]))
        self.assertEqual(stream_evidence(changed["realization"]), replay_total)
        with self.assertRaisesRegex(ValueError, "tensor state differs"):
            _validate_endpoint_binding(changed, checkpoint, replay_total)

    def test_every_completion_file_is_required_before_any_numerical_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = _required(directory)
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf8")
            for missing in ("verification/fresh.json", "audit/fresh.json", "audit/report.json"):
                path = directory / missing
                path.unlink()
                with patch("experiments.summarize_realization_study.study.load_protocol") as protocol, \
                        patch("experiments.summarize_realization_study._read") as read, \
                        patch("experiments.summarize_realization_study.torch.load") as load:
                    for operation in (summarize, verify_completed):
                        with self.assertRaisesRegex(ValueError, "both complete replays"):
                            operation(directory)
                    protocol.assert_not_called()
                    read.assert_not_called()
                    load.assert_not_called()
                path.write_text("{}", encoding="utf8")
            self.assertFalse((directory / "audit" / "summary.json").exists())

    def test_source_hash_rejection_precedes_checkpoint_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for path in _required(directory):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf8")
            with patch("experiments.summarize_realization_study.study.load_protocol", side_effect=ValueError("source changed")), \
                    patch("experiments.summarize_realization_study.torch.load") as load:
                with self.assertRaisesRegex(ValueError, "source changed"):
                    summarize(directory)
                load.assert_not_called()
            self.assertFalse((directory / "audit" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
