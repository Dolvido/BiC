"""Synthetic arithmetic/provenance gates; never opens live study evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import consolidation_study as study
from experiments.summarize_consolidation_study import (
    COUNTS, JOBS, QUERY_PANELS, _json_equal, assemble_summary, compact, sampled_counts, summarize, verify_completed,
)
from tests.test_summarize_diversity_study import metrics as old_metrics


def metrics(names, accuracy, *, replies=True):
    result = old_metrics(names, accuracy, replies=replies)
    for row in result["per_bank"].values():
        row.update(ask_correct=10,
            per_target={str(label): {"total": 20, "correct": int(20 * accuracy), "accuracy": accuracy} for label in range(3)},
            confusion_matrix_true_rows_predicted_columns=[[10, 5, 5, 0]] * 4)
    return result


def exposures(updates, replay=False):
    result = {}
    for source, count in (("support", updates * 64), ("replay", updates * 32 if replay else 0)):
        result.update({f"{source}_episodes": count, f"{source}_observation_bytes": count * 30,
            f"{source}_observation_tokens": count * 42, f"{source}_reply_target_bytes": count * 12,
            f"{source}_reply_target_tokens": count * 18})
    return result


def reports():
    protocol = {"interpretation": "Synthetic single-seed fixture", "parent_updates": 3600,
                "banks": {"query": {"names": {"naming_maps": 64}, "both": {"naming_maps": 128}}}}
    families = study.prior.FAMILIES
    old_panels = [f"{panel}/{family}" for panel in ("names", "worlds", "both") for family in families]
    extensions = {}
    for arm in study.ARMS:
        history = [{"updates": step, "training_seconds": index * 10.,
            "development": metrics(old_panels, .2 + index * .1, replies=False),
            "common_fit": metrics(families, .5 + index * .1, replies=False)}
            for index, step in enumerate((3600, 7200, 10800, 14400))]
        extensions[arm] = {"updates": 14400, "additional_updates": 10800,
            "additional_episode_exposures": 10800 * 64, "training_seconds": 30.,
            "invocation_seconds_after_setup": 35., "peak_cuda_allocated_mib": None,
            "peak_scope": "synthetic CPU", "history": history,
            "development": metrics(old_panels, .5), "common_fit": metrics(families, .8, replies=False)}
    results = {}
    for name in JOBS:
        parent, mode = name.rsplit("-", 1)
        is_replay = mode == "replay"
        accuracy = .1 if parent == "fresh" else .2 + (.1 if parent.endswith("long") else 0)
        accuracy += .05 if is_replay else 0
        curve = [{"updates": step, "optimizer_updates": step, "exposures": exposures(step, is_replay),
                  **metrics(QUERY_PANELS, accuracy + step / 10000)} for step in (0, 4, 16, 64, 256)]
        results[name] = {"parent": parent, "mode": mode, "optimizer_updates": 256, "curve": curve,
            "query_macro_weighting": "equal names/worlds/both/advanced panels, not pooled query counts",
            "development_before": metrics(old_panels, .5), "development_after": metrics(old_panels, .3),
            "retention_before": metrics(old_panels, .5), "retention_after": metrics(old_panels, .4 if is_replay else .1),
            "advanced_before": metrics(families, .2), "advanced_after": metrics(families, .1),
            "controls_at_256": {name: metrics(QUERY_PANELS, .1) for name in ("blank_text", "reset_history")},
            "cpu_restart": {"exact": True}, "exposures": exposures(256, is_replay),
            "support_storage": {"episodes": 2048}, "replay_storage": {"episodes": 1536 if is_replay else 0},
            "training_seconds": 6. if is_replay else 4., "wall_seconds": 10., "time_scope": "synthetic",
            "initial_weights_sha256": parent, "final_weights_sha256": name}
    return protocol, extensions, {"results": results, "base_checkpoints_unchanged": True}


class ConsolidationSummaryTests(unittest.TestCase):
    def test_checkpoint_recipe_matches_its_json_copy_without_weakening_value_checks(self):
        recipe = {"optimizer": {"name": "AdamW", "options": {"betas": (.9, .999), "lr": .001}}}
        serialized = json.loads(json.dumps(recipe))
        self.assertNotEqual(recipe, serialized)
        self.assertTrue(_json_equal(recipe, serialized))
        serialized["optimizer"]["options"]["betas"][0] = .8
        self.assertFalse(_json_equal(recipe, serialized))

    def test_compute_counts_extensions_once_and_nine_final_adaptations_once(self):
        result = assemble_summary(*reports())
        compute = result["compute"]
        self.assertEqual(compute["extension_optimizer_updates"], 21600)
        self.assertEqual(compute["extension_episode_exposures"], 21600 * 64)
        self.assertEqual(compute["adaptation_optimizer_updates"], 2304)
        self.assertEqual(compute["adaptation_exposures"]["support_episodes"], 2304 * 64)
        self.assertEqual(compute["adaptation_exposures"]["replay_episodes"], 4 * 256 * 32)
        self.assertEqual(compute["extension_worker_training_seconds"], 60.)
        self.assertEqual(compute["adaptation_training_seconds"], 4 * 6. + 5 * 4.)
        self.assertIn("not intrinsic transfer", result["comparison_note"])
        self.assertEqual(result["bank_denominators"]["query"]["both"]["naming_maps"], 128)
        self.assertIn("not zero", result["missing_value_note"])
        self.assertIn("do not isolate learning speed", result["area_note"])
        self.assertFalse(result["automatic_promotion"])

    def test_replay_longer_training_and_fresh_effects_keep_correct_comparators(self):
        result = assemble_summary(*reports())
        replay = result["replay_minus_ordinary"]["w32-n8-short"]
        self.assertAlmostEqual(replay["query_area_difference"]["macro_query_accuracy"], .05)
        self.assertAlmostEqual(replay["old_retention_endpoint_difference"]["macro"]["macro_pair_accuracy"], .3)
        self.assertEqual(replay["extra_exposure"]["support_episodes"], 0)
        self.assertEqual(replay["extra_exposure"]["replay_episodes"], 8192)
        longer = result["long_minus_short_at_fixed_replay_condition"]["w32-n8"]["replay"]
        self.assertAlmostEqual(longer["query_panel_area_difference"]["advanced"]["macro_pair_accuracy"], .1)
        ordinary = result["ordinary_pretrained_minus_common_fresh"]["w32-n8-short"]
        self.assertAlmostEqual(ordinary["query_area_difference"]["macro_query_accuracy"], .1)
        row = result["adaptation"]["w32-n8-short-replay"]
        self.assertEqual(row["endpoint_later_known_pairs"]["advanced"]["total"], 20)
        self.assertAlmostEqual(row["endpoint_later_known_pairs"]["advanced"]["accuracy"], .2756)
        self.assertEqual(row["retention_after"]["per_bank"]["names/variable_binding"]["ask_correct"], 10)

    def test_comparison_rejects_mismatched_new_bytes_despite_matching_update_budget(self):
        protocol, extensions, audit = reports()
        audit["results"]["w32-n8-short-replay"]["exposures"]["support_observation_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "matched new"):
            assemble_summary(protocol, extensions, audit)

    def test_missing_reply_and_pair_denominators_stay_absent(self):
        value = metrics(("worlds",), .5, replies=False)
        value["per_bank"]["worlds"]["positions"]["last_known_later_query"]["pairs"] = {
            "total": 0, "correct": 0, "accuracy": None}
        row = compact(value)["per_bank"]["worlds"]
        self.assertIsNone(row["later_known_pair_accuracy"])
        self.assertIsNone(row["query_reply_exact_accuracy"])
        self.assertEqual(row["query_reply_denominator"], 100)

    def test_utf8_exposure_reconstruction_matches_draws_and_excludes_parent_phase(self):
        rows = [{"turns": [{"text": text, "reply": reply} for _ in range(6)]}
                for text, reply in (("é", "Y"), ("éé", "No"), ("abc", "?"), ("z", "é"))]
        actual, states = sampled_counts({"family": rows}, 77, 4, {"family": 5}, skip_updates={"family": 2})
        generator = torch.Generator().manual_seed(77)
        observed = []
        for update in range(5):
            pairs = torch.randint(2, (2,), generator=generator)
            if update >= 2:
                for pair in pairs.tolist():
                    observed.extend(rows[pair * 2:pair * 2 + 2])
        text_bytes = sum(len(turn["text"].encode("utf8")) for row in observed for turn in row["turns"])
        reply_bytes = sum(len(turn["reply"].encode("utf8")) for row in observed for turn in row["turns"])
        self.assertEqual(actual, {"episodes": 12, "observation_bytes": text_bytes,
            "observation_tokens": text_bytes + 144, "reply_target_bytes": reply_bytes,
            "reply_target_tokens": reply_bytes + 72})
        self.assertTrue(torch.equal(states["family"], generator.get_state()))
        empty, empty_states = sampled_counts({}, 77, 4, {})
        self.assertEqual(empty, dict.fromkeys(COUNTS, 0))
        self.assertEqual(empty_states, {})

    def test_completion_gate_opens_no_protocol_or_checkpoint_when_any_audit_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with patch("experiments.summarize_consolidation_study.study.load_protocol") as protocol, \
                    patch("experiments.summarize_consolidation_study.torch.load") as checkpoint:
                with self.assertRaisesRegex(ValueError, "completed audit"):
                    summarize(path)
                with self.assertRaisesRegex(ValueError, "all nine"):
                    verify_completed(path)
                protocol.assert_not_called()
                checkpoint.assert_not_called()
            self.assertFalse((path / "audit" / "summary.json").exists())

    def test_source_rejection_precedes_deserialization_and_summary_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            files = [path / name for name in ("protocol.json", "banks.pt", "replay.pt")]
            files += [path / "audit" / name for name in ("report.json", "evaluation-started.json")]
            files += [path / "extension" / arm / name for arm in study.ARMS for name in
                      ("latest.pt", "report.json", "checkpoint-007200.pt", "checkpoint-010800.pt", "checkpoint-014400.pt")]
            files += [path / "audit" / filename for name in JOBS for filename in (f"{name}.json", f"{name}-adapted.pt")]
            for file in files:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("{}", encoding="utf8")
            with patch("experiments.summarize_consolidation_study.study.load_protocol", side_effect=ValueError("source changed")), \
                    patch("experiments.summarize_consolidation_study.torch.load") as checkpoint:
                with self.assertRaisesRegex(ValueError, "source changed"):
                    summarize(path)
                checkpoint.assert_not_called()
            self.assertFalse((path / "audit" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
