"""Frozen selection and resume contracts for the architecture comparison."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import sequence_study as study


class SequenceStudyTests(unittest.TestCase):
    def candidates(self):
        return [{"learning_rate": rate, "development": {
            "macro_pair_accuracy": .25, "macro_later_known_accuracy": .5}}
            for rate in study.LEARNING_RATES]

    def test_selection_uses_both_scores_and_lower_rate_for_exact_ties(self):
        candidates = self.candidates()
        self.assertEqual(study.choose_rate(list(reversed(candidates)))["learning_rate"], .0003)
        candidates[1]["development"].update(macro_pair_accuracy=.75, macro_later_known_accuracy=.25)
        candidates[2]["development"].update(macro_pair_accuracy=.25, macro_later_known_accuracy=.75)
        self.assertEqual(study.choose_rate(candidates)["learning_rate"], .001)
        candidates[2]["development"]["macro_later_known_accuracy"] = 1.
        self.assertEqual(study.choose_rate(candidates)["learning_rate"], .003)

    def test_selection_rejects_missing_duplicate_rates_and_invalid_criteria(self):
        for candidates in (self.candidates()[:-1], [self.candidates()[0]] * 3):
            with self.assertRaises(ValueError):
                study.choose_rate(candidates)
        for key in ("macro_pair_accuracy", "macro_later_known_accuracy"):
            for value in (None, float("nan"), float("inf"), -.01, 1.01):
                candidates = self.candidates()
                candidates[0]["development"][key] = value
                with self.assertRaises(ValueError):
                    study.choose_rate(candidates)

    def test_resume_requires_identical_protocol_and_interleaved_family_counts(self):
        protocol = {"steps": 12, "architecture": "sequence", "learning_rate": .001}
        saved = {"schema": study.SCHEMA, "protocol": protocol, "training": {
            "updates": 5, "family_updates": dict(zip(study.TRAIN_FAMILIES, (2, 2, 1)))}}
        study.validate_resume(saved, protocol)
        changes = (
            lambda row: row.update(schema="wrong"),
            lambda row: row["protocol"].update(learning_rate=.003),
            lambda row: row["training"].update(updates=True),
            lambda row: row["training"].update(updates=-1),
            lambda row: row["training"].update(updates=13),
            lambda row: row["training"]["family_updates"].update({study.TRAIN_FAMILIES[0]: 1}),
        )
        for change in changes:
            bad = copy.deepcopy(saved)
            change(bad)
            with self.assertRaises(ValueError):
                study.validate_resume(bad, protocol)

    def test_all_nine_completed_calibrations_are_required_and_selection_is_immutable(self):
        protocol = {"selection_rule": "test frozen rule", "source_sha256": {"test.py": "abc"}}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for architecture in study.ARCHITECTURES:
                for rate in study.LEARNING_RATES:
                    path = study.calibration_path(directory, architecture, rate)
                    path.mkdir(parents=True)
                    report = {"updates": study.CALIBRATION_STEPS, "stop_reason": "planned_updates",
                        "protocol": {"phase": "calibration", "architecture": architecture,
                                     "learning_rate": rate, "study": protocol},
                        "heldout_evaluation_performed": False,
                        "development": {"macro_pair_accuracy": .25, "macro_later_known_accuracy": .5}}
                    (path / "report.json").write_text(json.dumps(report), encoding="utf8")
            with patch.object(study, "load_protocol", return_value=protocol):
                with self.assertRaisesRegex(ValueError, "before launching"):
                    study.select_rates(directory, persist=False)
                selected = study.select_rates(directory)
                self.assertEqual(selected["choices"], dict.fromkeys(study.ARCHITECTURES, .0003))
                self.assertEqual(study.select_rates(directory), selected)
                with patch.object(study, "atomic_json", side_effect=AssertionError("read-only selection wrote a file")), \
                     patch.object(study, "run_lock", side_effect=AssertionError("read-only selection acquired root lock")):
                    self.assertEqual(study.select_rates(directory, persist=False), selected)
                report_path = study.calibration_path(directory, "sequence", .001) / "report.json"
                report = json.loads(report_path.read_text())
                report["updates"] -= 1
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    study.select_rates(directory)
                report["updates"] += 1
                report["development"]["macro_pair_accuracy"] = 1.
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "immutable"):
                    study.select_rates(directory)


if __name__ == "__main__":
    unittest.main()
