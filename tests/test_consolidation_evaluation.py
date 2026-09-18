"""CPU-only audit accounting, preservation and strict evaluation boundaries."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments import consolidation_evaluation as audit
from experiments.consolidation_training import ConsolidationTrainer
from experiments.diverse_curriculum import generate_diverse
from experiments.sequence_student import SequenceConfig, build_sequence_student


class ConsolidationEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def config(self):
        return SequenceConfig(width=16, layers=1, heads=2, feedforward=32)

    def banks(self):
        groups = {}
        for group_index, group in enumerate(("development", "retained", "advanced")):
            groups[group] = {family: generate_diverse(9200 + group_index * 100 + index * 10, 2,
                split="dev" if group == "development" else "audit", family=family,
                level=3 if group == "advanced" else 2) for index, family in enumerate(audit.OLD_FAMILIES)}
        groups["support"] = {audit.HELDOUT: generate_diverse(9400, 4, family=audit.HELDOUT)}
        groups["query"] = {panel: generate_diverse(9500 + index * 10, 2,
            split="audit", family=audit.HELDOUT, level=3 if panel == "advanced" else 2)
            for index, panel in enumerate(audit.QUERY_PANELS)}
        return groups

    def replay(self):
        return {family: generate_diverse(9600 + index * 10, 2, family=family)
                for index, family in enumerate(audit.OLD_FAMILIES)}

    def small_trainer(self, *args, **kwargs):
        return ConsolidationTrainer(*args, config=self.config(), **kwargs)

    def test_sparse_areas_keep_budget_spacing_and_undefined_metrics_fail(self):
        values = [(0, .1), (4, .7), (16, .3), (64, .9), (256, .2)]
        curve = [{"updates": budget, **{key: value for key in audit.METRICS}} for budget, value in values]
        expected = (4 * .4 + 12 * .5 + 48 * .6 + 192 * .55) / 256
        self.assertEqual(set(audit.normalized_areas(curve)), set(audit.METRICS))
        self.assertAlmostEqual(audit.normalized_areas(curve)[audit.METRICS[0]], expected)
        for alteration in (None, float("nan"), True, -0.1, 1.1):
            bad = deepcopy(curve)
            bad[1][audit.METRICS[0]] = alteration
            with self.assertRaises(ValueError):
                audit.normalized_areas(bad)
        for bad in (curve[:1], list(reversed(curve)), [curve[0], curve[0]], [{**curve[0], "updates": 1}, curve[1]]):
            with self.assertRaises(ValueError):
                audit.normalized_areas(bad)

    def test_invalid_eval_provenance_and_subjects_fail_before_trainer_creation(self):
        for mode in ("label", "train_admission", "old_subject", "missing_panel", "missing_group"):
            banks = self.banks()
            if mode == "label":
                banks["query"]["worlds"][0]["turns"][0]["target"] = -1
            elif mode == "train_admission":
                banks["query"]["worlds"] = generate_diverse(9700, 2, family=audit.HELDOUT)
            elif mode == "old_subject":
                banks["query"]["worlds"] = generate_diverse(9700, 2, split="audit", family="variable_binding")
            elif mode == "missing_panel":
                del banks["query"]["names"]
            else:
                del banks["retained"]
            with patch("experiments.consolidation_training.ConsolidationTrainer") as trainer:
                with self.assertRaises(ValueError):
                    audit.adapt_candidate({}, banks)
                trainer.assert_not_called()

    def test_replay_rejects_wrong_family_sets_and_audit_rows(self):
        model = build_sequence_student(11, config=self.config())
        replay = self.replay()
        with self.assertRaises(ValueError):
            audit.adapt_candidate(model.state_dict(), self.banks(), {"variable_binding": replay["variable_binding"]})
        replay["variable_binding"] = generate_diverse(9700, 2, split="audit", family="variable_binding")
        with patch("experiments.consolidation_training.ConsolidationTrainer", side_effect=self.small_trainer):
            with self.assertRaises(ValueError):
                audit.adapt_candidate(model.state_dict(), self.banks(), replay)

    def test_one_step_smoke_records_both_conditions_without_mutating_inputs(self):
        banks, replay = self.banks(), self.replay()
        bank_copy, replay_copy = deepcopy(banks), deepcopy(replay)
        model = build_sequence_student(31, config=self.config())
        before = checkpoint_digest(model)
        results, states = [], []
        with patch.object(audit, "BUDGETS", (0, 1)), \
             patch("experiments.consolidation_training.ConsolidationTrainer", side_effect=self.small_trainer):
            for old in (None, replay):
                result, state = audit.adapt_candidate(model.state_dict(), banks, old, seed=3701)
                results.append(result)
                states.append(state)
                self.assertEqual(result["optimizer_updates"], 1)
                self.assertEqual(result["controls_at_updates"], 1)
                self.assertEqual([point["updates"] for point in result["curve"]], [0, 1])
                self.assertEqual(set(result["curve"][0]["per_bank"]), set(audit.QUERY_PANELS))
                self.assertEqual(set(result["controls_at_256"]), {"blank_text", "reset_history"})
                self.assertEqual(result["exposures"]["support_episodes"], 64)
                self.assertEqual(result["exposures"]["replay_episodes"], 0 if old is None else 32)
                self.assertEqual(result["exposures"], state["exposures"])
                self.assertEqual(result["recipe"], state["recipe"])
                self.assertTrue(result["cpu_restart"]["exact"])
                self.assertTrue(result["parent_weights_unchanged"])
                self.assertEqual(result["initial_weights_sha256"], before)
                self.assertNotEqual(result["final_weights_sha256"], before)
                self.assertGreater(result["training_seconds"], 0)
                self.assertGreater(result["wall_seconds"], result["training_seconds"])
                self.assertGreater(result["support_storage"]["canonical_json_bytes"], result["support_storage"]["observation_utf8_bytes"])
                for group in ("development_before", "development_after", "retention_before", "retention_after", "advanced_before", "advanced_after"):
                    self.assertTrue(all(x["free_running_replies"] and not x["teacher_used_for_policy"] for x in result[group]["per_bank"].values()))
        self.assertEqual(checkpoint_digest(model), before)
        self.assertEqual(banks, bank_copy)
        self.assertEqual(replay, replay_copy)
        self.assertTrue(torch.equal(states[0]["samplers"][audit.HELDOUT], states[1]["samplers"][audit.HELDOUT]))
        self.assertEqual(results[0]["exposures"]["support_observation_bytes"], results[1]["exposures"]["support_observation_bytes"])
        self.assertEqual(results[0]["replay_storage"]["canonical_json_bytes"], 0)
        self.assertGreater(results[1]["replay_storage"]["canonical_json_bytes"], 0)
        self.assertEqual(results[0]["condition"], "ordinary")
        self.assertEqual(results[1]["condition"], "replay")


if __name__ == "__main__":
    unittest.main()
