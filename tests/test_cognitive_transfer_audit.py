"""Audit boundaries and adaptation mechanics; at most one tiny training update."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.dialogue_student import checkpoint_digest, evaluate_dialogues
from experiments import audit_cognitive_transfer as audit
from experiments.cognitive_curriculum import generate_cognitive
from experiments.cognitive_student import build_cognitive_student
from experiments.train_cognitive import CognitiveTrainer


class CognitiveTransferAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_overlap_removes_whole_pair_and_rejects_unpaired_rows(self):
        rows = generate_cognitive(90, 4, family="conditional_logic")
        original = copy.deepcopy(rows)
        kept, removed = audit.exclude_overlapping_pairs(rows, {audit.transcript(rows[0])})
        self.assertEqual(kept, rows[2:])
        self.assertEqual(removed, [row["id"] for row in rows[:2]])
        self.assertEqual(original, rows)
        with self.assertRaisesRegex(ValueError, "complete pairs"):
            audit.exclude_overlapping_pairs(rows[:3], set())

    def test_prepare_is_idempotent_and_source_or_bank_change_fails(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit, "audit_source_hashes", return_value={"fixture": "one"}):
            first = audit.prepare(temp)
            self.assertEqual(first, audit.prepare(temp))
            manifest, banks = audit.load_prepared(temp)
            self.assertEqual(manifest["query_split"], "audit")
            self.assertEqual(len(banks["support"]["conditional_logic"]), 64)
            self.assertTrue(all(row["level"] == 3 and row["split"] == "audit"
                for rows in banks["trained_family_transfer"].values() for row in rows))
            self.assertTrue(all(row["split"] == "audit" for rows in banks["query"].values() for row in rows))
            banks["query"]["level_2"][0]["turns"][0]["text"] = "tampered"
            (Path(temp) / "banks.json").write_text(json.dumps(banks), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "bank mismatch"):
                audit.load_prepared(temp)
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(audit, "audit_source_hashes", return_value={"fixture": "one"}):
                audit.prepare(temp)
            with patch.object(audit, "audit_source_hashes", return_value={"fixture": "two"}):
                with self.assertRaisesRegex(ValueError, "source mismatch"):
                    audit.load_prepared(temp)

    def test_adaptation_preserves_weights_but_resets_moments_and_sampling(self):
        support = {"conditional_logic": generate_cognitive(200, 2, family="conditional_logic")}
        trained = CognitiveTrainer(support, seed=19, batch_size=2)
        trained.step("conditional_logic")
        self.assertTrue(trained.optimizer.state)
        original = trained.snapshot()
        left = audit.fresh_adaptation_trainer(original["weights"], support, "episodic")
        right = audit.fresh_adaptation_trainer(original["weights"], support, "episodic")
        self.assertEqual(checkpoint_digest(left.model), checkpoint_digest(trained.model))
        self.assertEqual(left.updates, 0)
        self.assertFalse(left.optimizer.state)
        self.assertTrue(torch.equal(left.generators["conditional_logic"].get_state(),
                                    right.generators["conditional_logic"].get_state()))
        self.assertEqual(left.batch_size, 64)
        for key, tensor in original["weights"].items():
            self.assertTrue(torch.equal(tensor, trained.model.state_dict()[key]))

    def test_metrics_match_existing_evaluator_and_null_missing_ask(self):
        model = build_cognitive_student(29)
        candidates = generate_cognitive(100, 16, split="audit", family="conditional_logic", level=2)
        rows = next(candidates[index:index + 2] for index in range(0, len(candidates), 2)
                    if all(turn["target"] != 2 for row in candidates[index:index + 2] for turn in row["turns"]))
        before = checkpoint_digest(model)
        expected = evaluate_dialogues(model, rows, score_replies=False)
        actual = audit.evaluate_bank(model, rows, replies=True)
        self.assertEqual(actual["query_accuracy"], expected["query_accuracy"])
        self.assertEqual(actual["counterfactual_accuracy"], expected["counterfactual_accuracy"])
        self.assertEqual(actual["counterfactual_query_pairs"], expected["counterfactual_query_pairs"])
        self.assertEqual(actual["ask_true"], 0)
        self.assertIsNone(actual["ask_recall"])
        self.assertEqual(sum(map(sum, actual["confusion_matrix_true_rows_predicted_columns"])), actual["query_total"])
        self.assertEqual(before, checkpoint_digest(model))

    def test_cpu_restart_in_both_memory_modes(self):
        episode = generate_cognitive(50, 2, family="conditional_logic", level=2)[0]
        for mode in ("episodic", "recurrent"):
            model = build_cognitive_student(23, memory_mode=mode)
            result = audit.verify_cpu_restart(model.state_dict(), mode, episode)
            self.assertTrue(result["exact"])
            self.assertEqual(result["checked_remaining_turns"], 3)

    def test_incomplete_runs_rejected_before_scoring(self):
        with self.assertRaisesRegex(ValueError, "exactly four"):
            audit.verify_runs([], {})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            protocol = {"seed": 2201, "steps": 1800, "memory_mode": "episodic", "schedule": "interleaved"}
            (path / "protocol.json").write_text(json.dumps(protocol))
            (path / "report.json").write_text(json.dumps({"updates": 3}))
            torch.save({"training": {"updates": 3}}, path / "latest.pt")
            manifest = {"prepared_utc": "2026-01-01T00:00:00+00:00", "initial_seed": 2201,
                        "planned_main_updates": 1800, "required_arms": [["episodic", "interleaved"]]}
            with self.assertRaisesRegex(ValueError, "incomplete"):
                audit.verify_runs([path] * 4, manifest)


if __name__ == "__main__":
    unittest.main()
