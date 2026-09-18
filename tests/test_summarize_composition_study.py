"""Synthetic summary arithmetic and completion gates; no live result access."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import composition_study as study
from experiments.composition_curriculum import VERSION
from experiments.composition_evaluation import METRICS, prediction_metrics
from experiments.summarize_composition_study import (
    _json_equal, _opposite_pair_denominators, _required, _validate_metrics, assemble_summary, compact,
    curve_areas, summarize, verify_completed,
)


def metrics(families, value, *, replies=True):
    rows = {f"{panel}/{family}/t{turns}": {**dict.fromkeys(METRICS, value),
        "query_total": 320, "known_total": 256, "final_pairs": {"correct": 16, "total": 64},
        "per_target": {"2": {"correct": 0, "total": 0, "accuracy": None}},
        "ask_precision": None, "ask_recall": None, "action_reply_agreement": .75 if replies else None,
        "reply_parseable_queries": 320 if replies else None}
        for panel in ("seen", "composed") for family in families for turns in (8, 10, 12)}
    if not replies:
        for row in rows.values():
            row.update(query_reply_accuracy=None, final_reply_pair_accuracy=None)
    return {"per_bank": rows, **{f"macro_{key}": next(iter(rows.values()))[key] for key in METRICS}}


def reports():
    protocol = {"schema": study.SCHEMA, "interpretation": "Synthetic fixture only"}
    main, results = {}, {}
    base = {"joint": .4, "seq-color": .3, "seq-count": .6, "fresh-color": .2, "fresh-count": .2}
    for job in study.JOBS:
        families = study.FAMILIES if not job.startswith("fresh-") else (job.split("-", 1)[1],)
        held = study.FAMILIES if job == "joint" else (job.split("-", 1)[1],)
        counts = study.expected_counts(job, study.total_updates(job))
        exposures = {family: {"episodes": count * 32, "turns": count * 320,
            "observation_bytes": count * 6400, "observation_tokens": count * 7040,
            "reply_target_bytes": count * 2560, "reply_target_tokens": count * 2880} for family, count in counts.items()}
        main[job] = {"updates": study.total_updates(job), "family_microbatches": counts,
            "exposures": exposures, "training_seconds": 10., "invocation_seconds_after_preparation": 12.,
            "peak_cuda_allocated_mib": None, "weights_sha256": job,
            "history": [{"updates": 0, "family_microbatches": dict.fromkeys(families, 0),
                         "training_seconds": 0., "development": metrics(study.FAMILIES, .1, replies=False)}]}
        curve = [{"updates": step, "held_episode_exposure": exposure, "weights_sha256": f"{job}-{step}",
                  "metrics": metrics(held, base[job] + .1 * exposure / study.EXPOSURES[-1])}
                 for step, exposure in zip(study.curve_steps(job), study.EXPOSURES)]
        results[job] = {"curve": curve, "final": metrics(study.FAMILIES, base[job] + .1),
            "retained_before_new_phase": metrics(set(study.FAMILIES) - set(held), .8) if job.startswith("seq-") else None,
            "controls": {name: metrics(held, .05) for name in ("blank", "reset")},
            "cpu_restart": {"exact": True}, "seconds": 2.}
    return protocol, main, {"results": results}


class CompositionSummaryTests(unittest.TestCase):
    def test_checkpoint_bucket_keys_match_json_round_trip_without_hiding_changed_counts(self):
        streams = {"color": {"sampler_sha256": "fixture",
            "bucket_microbatches": {8: 1100, 10: 1200, 12: 1300},
            "exposures": {"episodes": 115200, "observation_bytes": 123456}}}
        serialized = json.loads(json.dumps(streams))
        # Integer sorting produces 8,10,12; string sorting produces10,12,8.
        self.assertNotEqual(json.dumps(streams, sort_keys=True), json.dumps(serialized, sort_keys=True))
        self.assertTrue(_json_equal(streams, serialized))
        changed = copy.deepcopy(serialized)
        changed["color"]["bucket_microbatches"]["10"] += 1
        self.assertFalse(_json_equal(streams, changed))
        changed = copy.deepcopy(serialized)
        changed["color"]["exposures"]["observation_bytes"] += 1
        self.assertFalse(_json_equal(streams, changed))

    def test_sparse_area_uses_exposure_spacing_and_one_family_not_joint_macro(self):
        _, _, audit = reports()
        curve = audit["results"]["joint"]["curve"]
        for point in curve:
            for name, row in point["metrics"]["per_bank"].items():
                if "/count/" in name:
                    row.update(dict.fromkeys(METRICS, .95))
        area = curve_areas(curve, "color")
        self.assertAlmostEqual(area["composed"]["query_accuracy"], .45)
        self.assertAlmostEqual(curve_areas(curve, "count")["seen"]["final_pair_accuracy"], .95)
        self.assertNotAlmostEqual(.45, sum(point["metrics"]["per_bank"]["seen/color/t8"]["query_accuracy"]
                                          for point in curve) / len(curve))

    def test_two_rotations_reuse_one_joint_without_double_counting_compute(self):
        result = assemble_summary(*reports())
        self.assertEqual(result["compute"]["optimizer_updates"], 13200)
        self.assertEqual(result["compute"]["microbatches"], 39600)
        self.assertEqual(result["compute"]["sampled_exposures"]["episodes"], 1267200)
        self.assertEqual(result["compute"]["worker_training_seconds"], 50.)
        color = result["rotations"]["color"]
        count = result["rotations"]["count"]
        self.assertEqual(color["common_joint_reference"], count["common_joint_reference"])
        self.assertAlmostEqual(color["sequential_minus_joint"]["area_difference"]["seen"]["query_accuracy"], -.1)
        self.assertAlmostEqual(count["sequential_minus_joint"]["area_difference"]["seen"]["query_accuracy"], .2)
        self.assertAlmostEqual(color["sequential_minus_fresh"]["area_difference"]["composed"]["final_pair_accuracy"], .1)
        self.assertIn("not independent", result["comparison_note"])
        self.assertIn("do not isolate learning speed", result["area_note"])
        self.assertFalse(result["automatic_promotion"])

    def test_denominators_missing_values_and_retention_stay_visible(self):
        result = assemble_summary(*reports())
        retention = result["audit"]["seq-color"]["retention"]
        self.assertAlmostEqual(retention["change"]["seen"]["count"]["macro"]["query_accuracy"], -.4)
        row = retention["after"]["per_panel_family"]["seen"]["count"]["per_length"]["t8"]
        self.assertEqual(row["final_pairs"]["total"], 64)
        self.assertEqual(row["query_total"], 320)
        self.assertIsNone(row["ask_recall"])
        self.assertIsNone(row["per_target"]["2"]["accuracy"])
        development = result["main"]["joint"]["development_curve"][0]["metrics"]
        self.assertIsNone(development["per_panel_family"]["seen"]["color"]["macro"]["query_reply_accuracy"])

    def test_matched_exposure_and_curve_mixture_rejections(self):
        protocol, main, audit = reports()
        main["fresh-color"]["exposures"]["color"]["observation_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "matched held"):
            assemble_summary(protocol, main, audit)
        curve = audit["results"]["joint"]["curve"]
        wrong = copy.deepcopy(curve)
        wrong[1]["held_episode_exposure"] = 0
        with self.assertRaisesRegex(ValueError, "increasing"):
            curve_areas(wrong, "color")
        wrong = copy.deepcopy(curve)
        del wrong[1]["metrics"]["per_bank"]["seen/color/t12"]
        with self.assertRaisesRegex(ValueError, "length mixture"):
            curve_areas(wrong, "color")

    def test_recorded_count_and_teacher_boundary_validation(self):
        truth = torch.tensor([[3, 2, 0, 1, 0, 3, 2, 0], [3, 2, 1, 0, 1, 3, 2, 1]])
        row = prediction_metrics(torch.nn.functional.one_hot(truth, 4).float() * 4, truth,
                                 reply_correct=torch.ones_like(truth, dtype=torch.bool), reply_actions=truth)
        manifest = {"sha256": "fixture", "episodes": 2, "complete_pairs": 1, "turns": 8,
            "target_counts_by_turn": [{str(target): int(truth[:, turn].eq(target).sum()) for target in range(4)}
                                      for turn in range(8)]}
        row.update(bank={"sha256": "fixture", "version": VERSION, "episodes": 2, "turns": 8,
            "role": "audit", "config": asdict(study.CONFIG)}, teacher_used_for_policy=False,
            decoder_prefix="BOS only", free_running_replies=True, control="normal")
        value = {"per_bank": {"seen/color/t8": row}, **{f"macro_{key}": row[key] for key in METRICS}}
        manifests = {"seen/color/t8": manifest}
        _validate_metrics(value, manifests, manifests, role="audit")
        for key, changed in (("query_correct", 0), ("query_total", 1), ("teacher_used_for_policy", True),
                             ("free_running_replies", False), ("decoder_prefix", "teacher reply"),
                             ("ask_true", 999), ("reply_parseable_queries", 999)):
            bad = copy.deepcopy(value)
            bad["per_bank"]["seen/color/t8"][key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                _validate_metrics(bad, manifests, manifests, role="audit")
        for field in ("per_target", "by_turn", "confusion_matrix"):
            bad = copy.deepcopy(value)
            scores = bad["per_bank"]["seen/color/t8"]
            if field == "per_target":
                scores[field]["0"]["total"] += 1
            elif field == "by_turn":
                scores[field][-1]["total"] += 1
            else:
                scores[field][0][0] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                _validate_metrics(bad, manifests, manifests, role="audit")

    def test_all_completed_gate_precedes_any_result_or_checkpoint_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch("experiments.summarize_composition_study.study.load_protocol") as protocol, \
                    patch("experiments.summarize_composition_study.torch.load") as checkpoint, \
                    patch("experiments.summarize_composition_study._read") as read:
                for operation in (summarize, verify_completed):
                    with self.assertRaisesRegex(ValueError, "five completed"):
                        operation(directory)
                protocol.assert_not_called()
                checkpoint.assert_not_called()
                read.assert_not_called()
            self.assertFalse((directory / "audit" / "summary.json").exists())

    def test_per_turn_accuracy_pairs_and_global_sums_must_match_canonical_targets(self):
        truth = torch.tensor([[3, 2, 0, 0], [3, 2, 0, 1]])
        name = "seen/color/t4"
        rows = [{"turns": [{"target": value} for value in values]} for values in truth.tolist()]
        denominators = _opposite_pair_denominators({name: rows})
        self.assertEqual(denominators, {name: [0, 0, 0, 1]})
        row = prediction_metrics(torch.nn.functional.one_hot(truth, 4).float() * 4, truth,
            reply_correct=torch.ones_like(truth, dtype=torch.bool), reply_actions=truth)
        manifest = {"sha256": "fixture", "episodes": 2, "complete_pairs": 1, "turns": 4,
            "target_counts_by_turn": [{str(target): int(truth[:, turn].eq(target).sum()) for target in range(4)}
                                      for turn in range(4)]}
        row.update(bank={"sha256": "fixture", "version": VERSION, "episodes": 2, "turns": 4,
            "role": "audit", "config": asdict(study.CONFIG)}, teacher_used_for_policy=False,
            decoder_prefix="BOS only", free_running_replies=True, control="normal")
        value = {"per_bank": {name: row}, **{f"macro_{key}": row[key] for key in METRICS}}
        manifests = {name: manifest}
        _validate_metrics(value, manifests, manifests, role="audit", pair_denominators=denominators)
        for change in ("accuracy", "huge_pairs", "pair_correct", "query_sum", "pair_sum", "canonical_eligibility"):
            bad = copy.deepcopy(value)
            score = bad["per_bank"][name]
            turn = score["by_turn"][2]
            if change == "accuracy":
                turn["accuracy"] = 0.
            elif change == "huge_pairs":
                turn["opposite_pair_total"] = 999
            elif change == "pair_correct":
                turn["opposite_pair_correct"] = 1
            elif change == "query_sum":
                turn.update(correct=1, accuracy=.5)
            elif change == "pair_sum":
                score["opposite_pair_total"] = 2
                score["opposite_pair_accuracy"] = .5
            else:
                # Internally coherent and bounded counts still cannot invent
                # an opposite pair where canonical targets are both DENY.
                turn.update(opposite_pair_total=1, opposite_pair_correct=1)
                score.update(opposite_pair_total=2, opposite_pair_correct=2)
            with self.subTest(change=change), self.assertRaises(ValueError):
                _validate_metrics(bad, manifests, manifests, role="audit", pair_denominators=denominators)

    def test_frozen_source_rejection_precedes_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for path in _required(directory):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf8")
            with patch("experiments.summarize_composition_study.study.load_protocol", side_effect=ValueError("source changed")), \
                    patch("experiments.summarize_composition_study.torch.load") as checkpoint:
                with self.assertRaisesRegex(ValueError, "source changed"):
                    summarize(directory)
                checkpoint.assert_not_called()
            self.assertFalse((directory / "audit" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
