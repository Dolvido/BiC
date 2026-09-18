"""Pure canonical data boundary checks; no models, scoring or optimizer work."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_variant_data as data_module
from experiments.foundation_evidence import json_digest, reconstruct_anchor
from experiments.foundation_plan import build_plan, materialize_bundle
from experiments.realization_banks import transcript_digest


class VariantDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.old, cls.directory = cls.root / "old", cls.root / "new"
        (cls.old / "evaluation").mkdir(parents=True)
        plans = {
            "calibration": dict(seed=941000001, stage_updates=64, final_updates=48,
                                micro_batch_size=2, rehearsal_every=4, ordering_seed=9410),
            "main": dict(seed=942000001, stage_updates=64, final_updates=48,
                         micro_batch_size=2, rehearsal_every=4, ordering_seed=9420),
        }
        pair = materialize_bundle(build_plan(**plans["calibration"]), 0)["color"]
        cls.collision = transcript_digest(pair[0])
        torch.save([cls.collision], cls.old / "protected.pt")
        torch.save(["a"*64], cls.old / "training-transcripts.pt")
        cls.opaque = cls.old / "old-model.pt"
        cls.opaque.write_bytes(b"Opaque historical checkpoint: never deserialize this.")
        inputs = {str(path.resolve()): data_module.file_hash(path)
                  for path in (cls.old / "protected.pt", cls.old / "training-transcripts.pt", cls.opaque)}
        relative = "experiments/foundation_plan.py"
        summary = dict(schema="bic-foundation-pilot-summary-v1", status="completed_descriptive",
            automatic_promotion=False, integrity=dict(canonical_replay_repeated=False,
                neural_training_or_inference=False, input_file_sha256=inputs,
                source_sha256={relative: data_module.file_hash(data_module.ROOT / relative)}))
        cls.summary_path = cls.old / "evaluation/summary.json"
        cls.summary_path.write_text(json.dumps(summary), encoding="utf8")
        cls.stack = ExitStack()
        cls.addClassCleanup(cls.stack.close)
        for name, value in {
            "PLAN_OPTIONS": plans, "ANCHOR_LIMITS": {"calibration": 1, "main": 2},
            "EVALUATIONS": (("calibration", "dev", 943000001, 1),
                            ("main", "dev", 944000001, 1), ("main", "audit", 945000001, 1)),
            "FIT_PAIRS": 1, "OLD_SUMMARY_SHA256": data_module.file_hash(cls.summary_path),
            "OLD_INPUT_COUNT": 3, "OLD_SOURCE_COUNT": 1,
        }.items():
            cls.stack.enter_context(patch.object(data_module, name, value))
        cls.manifest = data_module.prepare(cls.directory, cls.old)
        cls.verified = data_module.verify(cls.directory)
        cls.data = cls.verified["data"]

    def test_full_new_preparation_and_independent_verification_are_model_free(self):
        report = self.verified["report"]
        self.assertEqual(report["status"], "verified")
        self.assertFalse(report["neural_training_or_inference"])
        self.assertFalse(report["old_canonical_replay"])
        self.assertFalse(report["automatic_promotion"])
        self.assertEqual(report["historical_full_input_count"], 3)
        self.assertEqual(report["historical_full_source_count"], 1)
        self.assertEqual(report["source_sha256_before"], report["source_sha256_after"])
        self.assertEqual(set(self.verified["indexes"]), {"calibration", "main"})
        self.assertEqual(self.data["history"], sorted([self.collision, "a"*64]))
        self.assertGreater(len(self.data["stages"]["calibration"]["plan"]["admission"]["realization_attempts"]), 0)
        for stage in data_module.STAGES:
            record = self.data["stages"][stage]
            identity = self.verified["indexes"][stage].identity
            self.assertEqual(identity["transcript_sha256"], record["manifest"]["transcript_sha256"])
            self.assertEqual(len(record["training_transcripts"]), 432*3*2)
            self.assertEqual(sum(value["episodes"] for value in record["manifest"]["totals"].values()), 432*3*2)
            self.assertEqual(set(record["manifest"]["anchors"]), set(data_module.evaluation.expected_fresh_cells()))

    def test_exact_cross_stage_history_and_expanded_evaluation_protection(self):
        stages, history = self.data["stages"], set(self.data["history"])
        calibration, main = (set(stages[key]["training_transcripts"]) for key in data_module.STAGES)
        self.assertFalse(calibration & main or calibration & history or main & history)
        self.assertEqual(set(stages["calibration"]["admission_protected_transcripts"]), history)
        self.assertEqual(set(stages["main"]["admission_protected_transcripts"]), history | calibration)
        blocked = history | calibration | main
        evaluated = set()
        for stage, role, _, _ in data_module.EVALUATIONS:
            record = stages[stage]
            hashes = [transcript_digest(row) for rows in record["banks"][role].values() for row in rows]
            self.assertEqual(len(hashes), len(set(hashes)))
            self.assertFalse(set(hashes) & blocked)
            self.assertEqual(record["bank_reports"][role]["external_exclusions"],
                             dict(count=len(blocked), sha256=json_digest(sorted(blocked))))
            blocked.update(hashes); evaluated.update(hashes)
        self.assertEqual(set(stages["calibration"]["protected_transcripts"]), history | main | evaluated)
        self.assertEqual(set(stages["main"]["protected_transcripts"]), history | calibration | evaluated)

    def test_first_canonical_main_anchors_and_expected_available_depths(self):
        record = self.data["stages"]["main"]
        fit = record["banks"]["train_fit"]
        self.assertEqual(len(fit), 63)
        for cell, anchors in record["manifest"]["anchors"].items():
            self.assertEqual(fit["fit/"+cell], reconstruct_anchor(record["plan"], anchors[0]))
            self.assertEqual(record["bank_reports"]["train_fit"]["anchors"]["fit/"+cell], anchors[:1])
        self.assertEqual(len(record["banks"]["dev"]), 90)
        self.assertEqual(len(record["banks"]["audit"]), 99)
        self.assertFalse(any(name.startswith("composed/") and "/d2/" in name for name in record["banks"]["dev"]))
        self.assertEqual(sum(name.startswith("composed/") and "/d2/" in name for name in record["banks"]["audit"]), 9)

    def test_ordinary_load_does_not_replay_old_or_new_lessons_or_deserialize_old_models(self):
        original_load = torch.load
        loaded = []
        def checked(path, *args, **kwargs):
            loaded.append(Path(path).resolve())
            self.assertNotEqual(Path(path).resolve(), self.opaque.resolve())
            self.assertIs(kwargs.get("weights_only"), True)
            return original_load(path, *args, **kwargs)
        with patch.object(data_module.admission, "repair_plan", side_effect=AssertionError("replay")), \
             patch.object(data_module, "prepare_training", side_effect=AssertionError("generation")), \
             patch.object(data_module.evaluation, "build_evaluation", side_effect=AssertionError("generation")), \
             patch.object(data_module.torch, "load", side_effect=checked):
            again = data_module.load(self.directory)
        self.assertEqual(again["manifest"], self.manifest)
        self.assertTrue(loaded)

    def test_full_gate_catches_old_checkpoint_change_while_ordinary_boundary_is_explicit(self):
        original = self.opaque.read_bytes()
        try:
            self.opaque.write_bytes(original+b"changed")
            data_module.load(self.directory)
            with self.assertRaisesRegex(ValueError, "historical input changed"):
                data_module.verify(self.directory)
        finally:
            self.opaque.write_bytes(original)

    def test_pinned_history_and_new_file_changes_rejected(self):
        for path in (self.summary_path, self.old / "protected.pt", self.directory / "main/plan.json"):
            original = path.read_bytes()
            try:
                path.write_bytes(original+b" ")
                with self.assertRaises(ValueError):
                    data_module.load(self.directory)
            finally:
                path.write_bytes(original)

    def test_mutated_supplied_data_or_stage_never_reaches_index_construction(self):
        changed = copy.deepcopy(self.data)
        changed["stages"]["main"]["protected_transcripts"] = []
        with patch.object(data_module, "AuthenticatedPlanIndex", side_effect=AssertionError("must fail before index")):
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                data_module.build_index(self.directory, "main", data=changed)
            changed["manifest"]["stages"]["main"] = data_module._stage_identity(changed["stages"]["main"])
            with self.assertRaisesRegex(ValueError, "caller manifest"):
                data_module.build_index(self.directory, "main", data=changed)
            with self.assertRaises(ValueError):
                data_module.build_index(self.directory, "other", data=self.data)
            with self.assertRaisesRegex(ValueError, "another directory"):
                data_module.build_index(self.root / "elsewhere", "main", data=self.data)

    def test_manifest_and_index_source_drift_rejected(self):
        changed = {**data_module.source_hashes(), "experiments/foundation_variant_data.py": "0"*64}
        with patch.object(data_module, "source_hashes", return_value=changed):
            with self.assertRaisesRegex(ValueError, "source identity"):
                data_module.load(self.directory)
            with self.assertRaisesRegex(ValueError, "source identity"):
                data_module.build_index(self.directory, "main", data=self.data)

    def test_failed_preparation_preserved_and_existing_directories_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            data_module.prepare(self.directory, self.old)
        target = self.root / "failed"
        with patch.object(data_module, "_construct", side_effect=ValueError("deliberate construction failure")):
            with self.assertRaisesRegex(ValueError, "deliberate"):
                data_module.prepare(target, self.old)
        failure = json.loads((target / "failure.json").read_text())
        self.assertEqual(failure["status"], "failed")
        self.assertFalse(failure["neural_training_or_inference"])
        self.assertFalse((target / "manifest.json").exists())

    def test_stage_boundary_and_exact_artifact_inventory_tampering_rejected(self):
        changed = copy.deepcopy(self.data)
        changed["stages"]["main"]["admission_protected_transcripts"] = changed["history"]
        changed["manifest"]["stages"]["main"] = data_module._stage_identity(changed["stages"]["main"])
        with self.assertRaisesRegex(ValueError, "admission history"):
            data_module._validate_loaded(changed)
        for name in ("../outside", "main/../history.pt", "/absolute", "main\\plan.json"):
            with self.assertRaises(ValueError):
                data_module._relative(self.directory, name)


if __name__ == "__main__":
    unittest.main()
