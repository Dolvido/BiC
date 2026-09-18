"""Continuation provenance and phase boundaries without long optimization."""
import copy
from dataclasses import asdict
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments import consolidation_study as study


class ConsolidationStudyTests(unittest.TestCase):
    def fixture(self, count=4800):
        arm = study.ARMS[0]
        banks = {family: "test" for family in study.prior.FAMILIES}
        protocol = {"parent": {"records": {arm: {"training_bank_sha256": banks}}}}
        saved = {"schema": study.SCHEMA, "protocol": protocol, "arm": arm,
            "training": {"updates": count,
                "family_updates": {family: count // 3 + int(index < count % 3)
                                   for index, family in enumerate(study.prior.FAMILIES)},
                "samplers": study.prior.expected_samplers(count),
                "recipe": {"banks": banks, "model": "bic-diversity-sequence-v1", "seed": 2601,
                    "batch_size": 64, "learning_rate": .001, "config": asdict(study.SequenceConfig())}}}
        return arm, protocol, saved

    def test_continuation_keeps_original_sampler_and_recipe(self):
        arm, protocol, saved = self.fixture()
        study.validate_extension(saved, protocol, arm)
        for change in (
            lambda row: row["training"].update(updates=True),
            lambda row: row["training"].update(updates=3599),
            lambda row: row["training"].update(updates=14401),
            lambda row: row.update(arm="w256-n8"),
            lambda row: row["training"].update(samplers=study.prior.expected_samplers(1200)),
            lambda row: row["training"]["family_updates"].update(variable_binding=3),
            lambda row: row["training"]["recipe"].update(seed=1),
            lambda row: row["training"]["recipe"].update(learning_rate=.003),
            lambda row: row["training"]["recipe"]["config"].update(heads=2),
        ):
            bad = copy.deepcopy(saved)
            change(bad)
            with self.assertRaises(ValueError):
                study.validate_extension(bad, protocol, arm)

    def test_bank_fingerprint_binds_the_entire_original_bank(self):
        arm, protocol, saved = self.fixture()
        bad = copy.deepcopy(saved)
        bad["training"]["recipe"]["banks"] = {"variable_binding": "changed"}
        with self.assertRaisesRegex(ValueError, "bank identity"):
            study.validate_extension(bad, protocol, arm)

    def test_audit_requires_all_four_short_and_long_endpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            protocol = {"parent": {"directory": str(directory / "parent")}}
            with self.assertRaises(FileNotFoundError):
                study.verify_endpoints(directory, protocol)
            self.assertFalse((directory / "audit" / "evaluation-started.json").exists())

    def test_named_arm_validation_precedes_any_file_reads(self):
        with patch.object(study, "load_protocol", side_effect=AssertionError("read before validation")):
            with self.assertRaises(ValueError):
                study.extend("missing", "unsupported", "cpu")


if __name__ == "__main__":
    unittest.main()
