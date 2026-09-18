"""Pure orchestration checks: no torch import or learner work."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.foundation_shared_diagnostic import Journal, SCHEMA, sha, source_guard


class DiagnosticJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "attempt"

    def journal(self, guard=lambda: None):
        return Journal(self.output, max_seconds=30, guard=guard, context={})

    def test_intent_durable_before_work_and_completed_cost(self):
        journal = self.journal()
        def operation():
            saved = json.loads((self.output / "execution.json").read_text())
            self.assertEqual(saved["operations"][0]["status"], "running")
            return {"completed": 3}
        self.assertEqual(journal.perform("one", operation, describe=lambda x: x), {"completed": 3})
        journal.finish("completed")
        saved = json.loads((self.output / "execution.json").read_text())
        self.assertEqual(saved["operations"][0]["evidence"], {"completed": 3})
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["learner_optimizer_updates"], 0)

    def test_interrupt_keeps_completed_and_partial_work_no_retry(self):
        journal = self.journal()
        journal.perform("first", lambda: 2, describe=lambda x: {"count": x})
        error = KeyboardInterrupt("fixture")
        error.probe_report = {"objective_evaluations_completed": 7}
        error.completed_probe_reports = [{"objective_evaluations_completed": 11}]
        with self.assertRaises(KeyboardInterrupt):
            journal.perform("interrupted", lambda: (_ for _ in ()).throw(error))
        saved = json.loads((self.output / "execution.json").read_text())
        self.assertEqual(len(saved["operations"]), 2)
        self.assertEqual(saved["operations"][0]["status"], "completed")
        self.assertEqual(saved["operations"][1]["status"], "interrupted")
        self.assertEqual(saved["operations"][1]["probe_report"]["objective_evaluations_completed"], 7)
        self.assertEqual(saved["operations"][1]["completed_probe_reports"][0]["objective_evaluations_completed"], 11)

    def test_expired_allowance_or_failed_guard_does_not_start_work(self):
        journal = self.journal()
        journal.deadline = 0
        called = []
        with self.assertRaises(TimeoutError):
            journal.perform("expired", lambda: called.append(True))
        self.assertEqual(called, [])
        self.assertEqual(journal.value["operations"], [])
        journal.deadline = float("inf")
        journal.guard = lambda: (_ for _ in ()).throw(ValueError("source changed"))
        with self.assertRaises(ValueError):
            journal.perform("source changed", lambda: called.append(True))
        self.assertEqual(called, [])

    def test_existing_output_and_nonfinite_allowance_rejected(self):
        journal = self.journal()
        original = (self.output / "execution.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.journal()
        self.assertEqual((self.output / "execution.json").read_bytes(), original)
        for bad in (0, -1, float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                Journal(Path(self.temp.name) / "bad", max_seconds=bad, guard=lambda: None, context={})
        self.assertFalse((Path(self.temp.name) / "bad").exists())

    def test_sources_check_fresh_bytes_and_membership(self):
        root = Path(self.temp.name)
        for name in ("brain_in_computer", "experiments"):
            (root / name).mkdir()
        source = root / "experiments/one.py"
        source.write_bytes(b"old")
        manifest = dict(schema=SCHEMA, python_directories=["brain_in_computer", "experiments"],
                        files={"experiments/one.py": sha(b"old")})
        source_guard(root, manifest)
        source.write_bytes(b"new")
        with self.assertRaises(ValueError):
            source_guard(root, manifest)
        source.write_bytes(b"old")
        (root / "experiments/two.py").write_bytes(b"new member")
        with self.assertRaises(ValueError):
            source_guard(root, manifest)


if __name__ == "__main__":
    unittest.main()
