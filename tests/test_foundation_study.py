"""Study gates and failure accounting, without neural inference or training."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import foundation_study as study


class FakeTrainer:
    def __init__(self, failure=False):
        self.cursor = 0
        self.model = object()
        self.last_report = None
        self.failure = failure

    def snapshot(self):
        return {"cursor": self.cursor}

    def step(self):
        failed = self.failure and self.cursor == 1
        self.last_report = {"physical_optimizer_updates": None if failed else 1,
            "neural_attempted_episode_exposures": 6, "drawn_episode_exposures": 6,
            "completed_microbatch_episode_exposures": 4 if failed else 6,
            "cursor": self.cursor if failed else self.cursor+1}
        if failed:
            raise RuntimeError("injected unknown optimizer completion")
        self.cursor += 1
        return self.last_report


class FoundationStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        study.atomic_json(self.directory/"protocol.json", {"test_fixture": True})
        self.fake = FakeTrainer()
        self.proof = {"execution_profile": {"fixture": True}, "learning_rate": .001,
                      "admission": {"mode": "naming_overrides", "executed_prefix_overridden_pairs": 1}}
        self.protocol = {"proof_directory": "fixture", "config": {"width": 8},
            "execution_profile": self.proof["execution_profile"],
            "proof_sha256": study.json_digest(self.proof), "initial_weights_sha256": "weights", "learning_rate": .001}
        stack = self.enterContext(ExitStack())
        patches = (
            patch.object(study, "TOTAL", 2), patch.object(study, "STEPS", (0, 2)),
            patch.object(study, "load_protocol", return_value=self.protocol),
            patch.object(study, "trainer", side_effect=lambda *a, **k: self.fake),
            patch.object(study, "checkpoint_digest", return_value="weights"),
            patch("experiments.foundation_runtime_probe.load_proof", return_value=self.proof),
            patch("experiments.execution_profile.runtime_profile", return_value=self.proof["execution_profile"]),
            patch("experiments.execution_profile.assert_strict_profile"),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.max_memory_allocated", return_value=0),
            patch("torch.cuda.max_memory_reserved", return_value=0),
        )
        for context in patches:
            stack.enter_context(context)

    def receipt(self):
        return study.read(self.directory/"curriculum/receipt.json")

    def test_one_shot_endpoint_keeps_steps_and_rejects_overwrite(self):
        report = study.train(self.directory, "curriculum")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["physical_optimizer_updates"], 2)
        self.assertEqual(report["neural_attempted_episode_exposures"], 12)
        self.assertFalse(report["physical_work_unknown"])
        self.assertEqual(set(report["checkpoints"]), {"checkpoint-000000.pt", "checkpoint-000002.pt"})
        path = self.directory/"curriculum/receipt.json"
        digest = study.file_hash(path)
        with self.assertRaises(FileExistsError):
            study.train(self.directory, "curriculum")
        self.assertEqual(study.file_hash(path), digest)
        events = [study.json.loads(line) for line in (path.parent/"steps.jsonl").read_text().splitlines()]
        self.assertEqual([item["event"] for item in events], ["started", "completed"]*2)

    def test_proof_without_actual_override_coverage_cannot_claim_a_training_arm(self):
        for admission in ({}, {"mode": "none", "executed_prefix_overridden_pairs": 16},
                          {"mode": "naming_overrides", "executed_prefix_overridden_pairs": 0},
                          {"mode": "naming_overrides", "executed_prefix_overridden_pairs": True}):
            self.proof["admission"] = admission
            self.protocol["proof_sha256"] = study.json_digest(self.proof)
            with self.subTest(admission=admission), self.assertRaisesRegex(ValueError, "naming overrides"):
                study.train(self.directory, "curriculum")
            self.assertFalse((self.directory/"curriculum").exists())
            self.assertEqual(self.fake.cursor, 0)

    def test_unknown_failed_optimizer_is_never_counted_as_zero_or_endpoint(self):
        self.fake.failure = True
        with self.assertRaisesRegex(RuntimeError, "unknown optimizer"):
            study.train(self.directory, "curriculum")
        report = self.receipt()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["retained_updates"], 1)
        self.assertEqual(report["physical_optimizer_updates"], 1)
        self.assertTrue(report["physical_work_unknown"])
        self.assertEqual(report["neural_attempted_episode_exposures"], 12)
        self.assertEqual(report["completed_microbatch_episode_exposures"], 10)
        self.assertNotIn("checkpoint-000002.pt", report["checkpoints"])

    def test_journal_failure_after_optimizer_preserves_known_physical_work(self):
        append = study.append_journal
        def fail_completed(path, value):
            if value["event"] == "completed":
                raise OSError("injected journal disk failure")
            append(path, value)
        with patch.object(study, "append_journal", side_effect=fail_completed):
            with self.assertRaisesRegex(OSError, "journal disk"):
                study.train(self.directory, "curriculum")
        report = self.receipt()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["physical_optimizer_updates"], 1)
        self.assertEqual(report["neural_attempted_episode_exposures"], 6)
        self.assertEqual(report["retained_updates"], 1)

    def test_failed_step_journal_cannot_mask_original_failure(self):
        self.fake.failure = True
        append = study.append_journal
        def fail_failure(path, value):
            if value["event"] == "failed":
                raise OSError("secondary journal failure")
            append(path, value)
        with patch.object(study, "append_journal", side_effect=fail_failure):
            with self.assertRaisesRegex(RuntimeError, "unknown optimizer"):
                study.train(self.directory, "curriculum")
        report = self.receipt()
        self.assertTrue(report["physical_work_unknown"])
        self.assertIn("secondary journal", report["failure_journal_error"])

    def test_proof_or_runtime_mismatch_precedes_claim_and_training(self):
        self.protocol["proof_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "proof changed"):
            study.train(self.directory, "curriculum")
        self.assertFalse((self.directory/"curriculum").exists())
        self.protocol["proof_sha256"] = study.json_digest(self.proof)
        self.protocol["execution_profile"] = {"different": True}
        with self.assertRaisesRegex(ValueError, "runtime differs"):
            study.train(self.directory, "curriculum")
        self.assertFalse((self.directory/"curriculum").exists())
        self.assertEqual(self.fake.cursor, 0)
        self.protocol["execution_profile"] = self.proof["execution_profile"]
        self.protocol["learning_rate"] = .003
        with self.assertRaisesRegex(ValueError, "learning rate differs"):
            study.train(self.directory, "curriculum")
        self.assertFalse((self.directory/"curriculum").exists())

    def test_missing_completed_capacity_summary_precedes_score_reads(self):
        with patch("torch.load", side_effect=AssertionError("must not read scores")):
            with self.assertRaisesRegex(ValueError, "completed verified"):
                study.capacity_choice(self.directory)

    def test_stub_capacity_summary_is_rejected_before_score_reads(self):
        (self.directory/"audit").mkdir()
        study.atomic_json(self.directory/"audit/summary.json", {})
        with patch("experiments.capacity_study.load_protocol", return_value={"source_sha256": {}}), \
                patch("experiments.summarize_capacity_study._complete_gate", return_value=[]), \
                patch("torch.load", side_effect=AssertionError("must not read scores")):
            with self.assertRaisesRegex(ValueError, "completed-summary"):
                study.capacity_choice(self.directory)

    def test_historical_stream_authentication_normalizes_json_integer_bucket_keys(self):
        from experiments import capacity_study as capacity
        evidence = {"bucket_microbatches": {"color": {8: 4, 10: 5, 12: 9}}}
        (self.directory/"verification").mkdir()
        for stage in ("calibration", "main"):
            study.atomic_json(self.directory/"verification"/f"{stage}.json",
                {"replayed_streams": {str(capacity.TOTALS[stage]): evidence}})
        protected = {"calibration": ["a"*64], "main": ["b"*64]}
        payload = {"learner": {"realization": {"seen_transcripts": ["c"*64]}}}
        with patch.object(capacity, "data", return_value=({}, protected)), \
                patch("experiments.capacity_banks._all_banks", return_value=[]), \
                patch("torch.load", return_value=payload), \
                patch("experiments.realization_training.stream_evidence", return_value=evidence):
            actual = study.historical_transcripts({"directory": str(self.directory), "files_sha256": {}})
            self.assertEqual(actual, ["a"*64, "b"*64, "c"*64])
            evidence["bucket_microbatches"]["color"][8] += 1
            with self.assertRaisesRegex(ValueError, "verified replay"):
                study.historical_transcripts({"directory": str(self.directory), "files_sha256": {}})


if __name__ == "__main__":
    unittest.main()
