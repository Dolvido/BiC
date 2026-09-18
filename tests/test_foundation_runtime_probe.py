"""Tiny separate CPU processes validate probe integrity; zero CUDA work."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_runtime_probe as probe
from experiments.sequence_student import SequenceConfig


class ProbePureHelpersTests(unittest.TestCase):
    def test_nonfinite_comparison_rejected_before_receipt_numeric_summary(self):
        reference = {"weights": {"weight": torch.zeros(2)}, "optimizer": {},
                     "timing": {"retained_step_seconds": 1., "materialization_seconds_included_in_step": .1}}
        for value in (float("nan"), float("inf")):
            changed = copy.deepcopy(reference)
            changed["weights"]["weight"][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe._difference(reference, changed)


class FoundationRuntimeProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.directory = cls.root / "reference"
        cls.config = SequenceConfig(width=8, layers=1, heads=2, feedforward=16, max_turns=12)
        cls.protocol = probe.prepare(cls.directory, device="cpu", config=cls.config, steps=2, micro_batch_size=2)
        for name in probe.WORKERS:
            result = subprocess.run([sys.executable, "-m", "experiments.foundation_runtime_probe", "worker",
                                     "--output", str(cls.directory), "--name", name], cwd=probe.ROOT,
                                    capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
        cls.receipt = probe.verify(cls.directory)
        if cls.receipt["status"] != "passed_cpu_engineering_only":
            raise AssertionError(cls.receipt)
        cls.admitted_directory = cls.root / "admitted-reference"
        cls.admitted_protocol = probe.prepare(cls.admitted_directory, device="cpu", config=cls.config,
                                              steps=2, micro_batch_size=2, admission=True)
        for name in probe.WORKERS:
            result = subprocess.run([sys.executable, "-m", "experiments.foundation_runtime_probe", "worker",
                                     "--output", str(cls.admitted_directory), "--name", name], cwd=probe.ROOT,
                                    capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
        cls.admitted_receipt = probe.verify(cls.admitted_directory)
        if cls.admitted_receipt["status"] != "passed_cpu_engineering_only":
            raise AssertionError(cls.admitted_receipt)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup(); torch.set_num_threads(cls.threads)
        print("Foundation runtime probe CPU fixtures: 12 completed optimizer updates / 72 episode exposures across eight separate processes, half exercising admitted overrides; no evaluation or CUDA.")

    def copy_fixture(self, name, *, prepared_only=False):
        directory = self.root / name
        ignored = ["probe.json", "verification-started.json"]
        if prepared_only:
            ignored.extend(probe.WORKERS)
        shutil.copytree(self.directory, directory, ignore=shutil.ignore_patterns(*ignored))
        return directory

    def rewrite(self, path, value):
        path.write_text(probe._json(value), encoding="utf8")

    def copy_admitted(self, name):
        directory = self.root / name
        shutil.copytree(self.admitted_directory, directory,
                        ignore=shutil.ignore_patterns("probe.json", "verification-started.json"))
        return directory

    def rebind_protocol(self, directory, protocol):
        self.rewrite(directory / "protocol.json", protocol)
        prepared = probe._read(directory / "prepared.json")
        prepared["protocol_sha256"] = probe._file_hash(directory / "protocol.json")
        self.rewrite(directory / "prepared.json", prepared)

    def rebind_worker_artifacts(self, directory, name):
        path = directory / name / "result.json"
        record = probe._read(path)
        record["artifact_sha256"] = probe._artifact_hashes(directory / name, ("result.json",))
        self.rewrite(path, record)
        # Later receipts bind prior results. Repair those parents only to make
        # tests exercise semantic verification beyond ordinary digest mismatch.
        for later in probe.WORKERS[probe.WORKERS.index(name) + 1:]:
            other_path = directory / later / "result.json"
            other = probe._read(other_path)
            for parent in other["parents"]:
                other["parents"][parent] = probe._file_hash(directory / parent)
            self.rewrite(other_path, other)

    def test_separate_process_exact_resume_repeat_and_physical_counts(self):
        record = self.receipt
        self.assertEqual(record["status"], "passed_cpu_engineering_only")
        self.assertTrue(all(record["flags"].values()))
        self.assertEqual(record["physical_updates"], 6)
        self.assertEqual(record["physical_episode_exposures"], 36)
        self.assertEqual(record["config"], asdict(self.config))
        self.assertEqual(record["learning_rate"], .001)
        self.assertEqual(len({r["process"]["instance"] for r in record["workers"].values()}), 4)
        self.assertTrue(probe._same(probe._load(self.directory / "split-first/final.pt"),
                                   probe._load(self.directory / "resumed/loaded-midpoint.pt")))
        self.assertEqual(record["physical"]["optimizer_completion_unknown_attempts"], 0)
        self.assertFalse(record["physical"]["unreported_started_attempts"])
        self.assertFalse(record["evaluation_performed"])
        self.assertFalse(record["automatic_promotion"])

    def test_cpu_fixture_never_authenticates_as_cuda_proof(self):
        with self.assertRaisesRegex(ValueError, "strict CUDA proof"):
            probe.load_proof(self.directory, self.config)

    def test_admitted_overrides_execute_on_both_sides_of_exact_cpu_resume(self):
        record = self.admitted_receipt
        self.assertTrue(all(record["flags"].values()))
        self.assertEqual(record["physical_updates"], 6)
        self.assertEqual(record["physical_episode_exposures"], 36)
        admission = record["admission"]
        self.assertEqual(admission["mode"], "naming_overrides")
        self.assertEqual(admission["executed_prefix_by_family"]["color"], 2)
        self.assertGreaterEqual(admission["before_midpoint_overridden_pairs"], 1)
        self.assertGreaterEqual(admission["after_midpoint_overridden_pairs"], 1)
        self.assertEqual(admission["executed_prefix_overridden_bundles"], 2)
        protocol, plan = probe._inputs(self.admitted_directory)
        original = probe._read(self.admitted_directory / "base-plan.json")
        self.assertEqual(protocol["protected_transcripts"], probe._synthetic_protected(original, 2))
        self.assertEqual(plan["config"], original["config"])
        self.assertEqual(plan["bundles"], original["bundles"])
        self.assertEqual(plan["schedules"], original["schedules"])
        endpoint = probe._load(self.admitted_directory / "resumed/final.pt")
        initial = probe._load(self.admitted_directory / "initial.pt")
        self.assertEqual(endpoint["recipe"]["protected_transcripts_count"], len(protocol["protected_transcripts"]))
        self.assertFalse(probe._same(initial["weights"], endpoint["weights"]))
        checker = probe.FoundationTrainer(plan, "mixed", seed=5101, config=self.config,
                                           protected_transcripts=protocol["protected_transcripts"])
        checker.restore(endpoint)
        self.assertTrue(probe._same(endpoint, checker.snapshot()))
        wrong = probe.FoundationTrainer(plan, "mixed", seed=5101, config=self.config)
        with self.assertRaises(ValueError):
            wrong.restore(endpoint)
        with self.assertRaisesRegex(ValueError, "strict CUDA proof"):
            probe.load_proof(self.admitted_directory, self.config)

    def test_admission_artifacts_and_forged_protected_prefix_are_rejected(self):
        for index, relative in enumerate(("base-plan.json", "admission.json", "protected.json")):
            directory = self.copy_admitted(f"admitted-artifact-{index}")
            with (directory / relative).open("ab") as stream:
                stream.write(b"tamper")
            with self.subTest(path=relative), self.assertRaises((ValueError, json.JSONDecodeError)):
                probe._inputs(directory)
        directory = self.copy_admitted("admitted-protected-rebound")
        protocol = probe._read(directory / "protocol.json")
        self.rewrite(directory / "protected.json", ["0" * 64])
        protocol["protected_transcripts"] = ["0" * 64]
        protocol["protected_sha256"] = probe._file_hash(directory / "protected.json")
        self.rebind_protocol(directory, protocol)
        with self.assertRaisesRegex(ValueError, "declared original prefix"):
            probe._inputs(directory)

    def test_admitted_prefix_coverage_cannot_be_relabelled_after_freeze(self):
        directory = self.copy_admitted("admitted-coverage-rebound")
        protocol = probe._read(directory / "protocol.json")
        protocol["admission"]["executed_prefix_overridden_pairs"] += 1
        self.rebind_protocol(directory, protocol)
        with self.assertRaisesRegex(ValueError, "prefix coverage"):
            probe._inputs(directory)

    def test_no_overwrite_prepare_worker_or_verify(self):
        for operation in (lambda: probe.prepare(self.directory, device="cpu", config=self.config),
                          lambda: probe.worker(self.directory, "uninterrupted"),
                          lambda: probe.verify(self.directory)):
            with self.assertRaises(FileExistsError):
                operation()

    def test_frozen_plan_source_snapshot_and_checkpoint_hash_tampering_fail(self):
        for index, relative in enumerate(("plan.json", "initial.pt", "sources/experiments/foundation_training.py",
                                          "repeat/final.pt")):
            directory = self.copy_fixture(f"tamper-{index}")
            with (directory / relative).open("ab") as stream:
                stream.write(b"tamper")
            with self.subTest(path=relative), self.assertRaises((ValueError, json.JSONDecodeError)):
                probe._verified(directory)

    def test_full_model_optimizer_cursor_and_stream_are_compared(self):
        directory = self.copy_fixture("tensor-tamper")
        path = directory / "repeat/final.pt"
        payload = probe._load(path)
        payload["optimizer"]["state"][0]["exp_avg"].flatten()[0] += .01
        torch.save(payload, path)
        self.rebind_worker_artifacts(directory, "repeat")
        record = probe._verified(directory)
        self.assertEqual(record["status"], "failed_equality")
        self.assertFalse(record["flags"]["uninterrupted_repeat_equal"])
        self.assertFalse(record["comparisons"]["repeat"]["components"]["optimizer"])
        self.assertGreater(record["comparisons"]["repeat"]["optimizer"]["max_absolute"], 0.)
        self.assertTrue(record["comparisons"]["repeat"]["components"]["weights"])

    def test_only_declared_valid_timing_is_ignored_and_midpoint_reload_is_exact(self):
        payload = probe._load(self.directory / "uninterrupted/final.pt")
        changed = copy.deepcopy(payload)
        changed["timing"]["retained_step_seconds"] += 100.
        self.assertTrue(probe._difference(payload, changed)["exact"])
        changed["evidence"]["consumed_ids_sha256"] = "0" * 64
        self.assertFalse(probe._difference(payload, changed)["exact"])
        changed = copy.deepcopy(payload); changed["timing"]["extra"] = 0
        with self.assertRaises(ValueError):
            probe._difference(payload, changed)
        changed = copy.deepcopy(payload); changed["timing"]["retained_step_seconds"] = -1
        with self.assertRaises(ValueError):
            probe._difference(payload, changed)
        directory = self.copy_fixture("load-timing-tamper")
        path = directory / "resumed/loaded-midpoint.pt"
        changed = probe._load(path); changed["timing"]["retained_step_seconds"] += 1.
        torch.save(changed, path); self.rebind_worker_artifacts(directory, "resumed")
        self.assertFalse(probe._verified(directory)["flags"]["load_at_midpoint_exact"])

    def test_journal_semantics_reject_boolean_counts_after_hashes_rebound(self):
        directory = self.copy_fixture("journal-type")
        path = directory / "repeat/steps.jsonl"
        events = probe._journal(directory, "repeat")
        events[1]["report"]["physical_optimizer_updates"] = True
        path.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf8")
        self.rebind_worker_artifacts(directory, "repeat")
        with self.assertRaisesRegex(ValueError, "integer"):
            probe._verified(directory)

    def test_same_process_identity_is_not_independent_repetition(self):
        directory = self.copy_fixture("same-process")
        path = directory / "repeat/result.json"
        result = probe._read(path)
        result["process"]["instance"] = probe._read(directory / "uninterrupted/result.json")["process"]["instance"]
        self.rewrite(path, result)
        with self.assertRaisesRegex(ValueError, "distinct processes"):
            probe._verified(directory)

    def test_worker_failure_keeps_claim_receipt_and_blocks_later_work_without_training(self):
        directory = self.copy_fixture("failed-worker", prepared_only=True)
        with patch.object(probe, "FoundationTrainer", side_effect=RuntimeError("injected setup failure")):
            first = probe.worker(directory, "uninterrupted")
            later = probe.worker(directory, "split-first")
        self.assertEqual(first["status"], "failed")
        self.assertEqual(later["status"], "failed")
        self.assertIn("did not complete", later["error"]["message"])
        self.assertTrue((directory / "uninterrupted/started.json").is_file())
        self.assertTrue((directory / "uninterrupted/result.json").is_file())
        with self.assertRaises(FileExistsError):
            probe.worker(directory, "uninterrupted")
        record = probe.verify(directory)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["physical"]["known_completed_optimizer_updates"], 0)

    def test_dangling_started_work_and_unknown_completion_are_not_reported_zero(self):
        directory = self.copy_fixture("uncertain", prepared_only=True)
        worker = directory / "uninterrupted"; worker.mkdir()
        probe._append(worker / "steps.jsonl", {"event": "started", "cursor": 0, "bundle_id": 12})
        known = probe._physical(directory)
        self.assertEqual(known["known_completed_optimizer_updates"], 0)
        self.assertEqual(known["unreported_started_attempts"], ["uninterrupted"])
        probe._append(worker / "steps.jsonl", {"event": "reported", "report": {
            "physical_optimizer_updates": None, "drawn_episode_exposures": None,
            "neural_attempted_episode_exposures": 2, "completed_microbatch_episode_exposures": 0,
            "step_seconds": 1., "materialization_seconds": .2}})
        known = probe._physical(directory)
        self.assertEqual(known["optimizer_completion_unknown_attempts"], 1)
        self.assertEqual(known["partial_materialization_unknown_attempts"], 1)
        self.assertEqual(known["neural_attempted_episodes"], 2)
        self.assertEqual(known["unreported_started_attempts"], [])

    def test_preflight_profile_mismatch_is_preserved_without_optimizer_work(self):
        directory = self.copy_fixture("profile-mismatch", prepared_only=True)
        with patch.object(probe, "_runtime", return_value={"schema": "changed"}), \
             patch.object(probe, "FoundationTrainer", side_effect=AssertionError("must not build learner")):
            record = probe.worker(directory, "uninterrupted")
        self.assertEqual(record["status"], "failed")
        self.assertIn("execution profile differs", record["error"]["message"])
        self.assertEqual(probe._physical(directory)["known_completed_optimizer_updates"], 0)


if __name__ == "__main__":
    unittest.main()
