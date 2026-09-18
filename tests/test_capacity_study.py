import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import capacity_study as study
from experiments.composition_curriculum import FAMILIES, generate_pair
from experiments.realization_backend import RealizationBackend
from experiments.realization_training import RealizationStream


class CapacityStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.banks = {family: {turns: sum((generate_pair(family, 263_000_000+index*1000+turns*10+offset,
                    turns=turns) for offset in range(2)), [])
                    for turns in (8, 10, 12)} for index, family in enumerate(FAMILIES)}
        learner = RealizationBackend(cls.banks, mode="fresh", protected_transcripts=[],
            seed=study.SEEDS["calibration"], sampler_seed=study.SAMPLERS["calibration"],
            config=study.config(96))
        cls.payload = learner.snapshot()

    def test_independent_sampler_reconstruction_matches_both_stage_streams(self):
        for stage in ("calibration", "main"):
            stream = RealizationStream(self.banks, mode="fresh", protected_transcripts=[],
                sampler_seed=study.SAMPLERS[stage])
            for _ in range(5):
                for family in FAMILIES:
                    stream.draw(family)
            expected = study.structural_evidence(self.banks, 5, study.SAMPLERS[stage])
            state = stream.snapshot()
            for family in FAMILIES:
                study._same(state["occurrences"][family], expected[family]["occurrences"])
                study._same(state["bucket_microbatches"][family], expected[family]["bucket_microbatches"])
                self.assertEqual(state["exposures"][family]["turns"], expected[family]["turns"])
        for value in (True, -1, 7201):
            with self.assertRaises(ValueError):
                study.structural_evidence(self.banks, value, 4100)

    def test_resume_recipe_must_match_before_any_more_optimization(self):
        spec = {"width": 96, "rate": .001}
        study.validate_backend(self.payload, "calibration", spec)
        for section, key, value in (("recipe", "learning_rate", .003), ("recipe", "seed", 3101),
                                    ("recipe", "sampler_seed", 4101), ("learner", "updates", 601)):
            changed = copy.deepcopy(self.payload)
            target = changed["learner"]["recipe"] if section == "recipe" else changed["learner"]
            target[key] = value
            with self.assertRaises(ValueError):
                study.validate_backend(changed, "calibration", spec)
        changed = copy.deepcopy(self.payload)
        changed["curriculum"]["schedule"] = [["color", "color", "count"]]
        with self.assertRaises(ValueError):
            study.validate_backend(changed, "calibration", spec)
        with self.assertRaises(ValueError):
            study.validate_backend(self.payload, "calibration", {"width": 192, "rate": .001})

    @staticmethod
    def receipt(width=96):
        updates = 96 if width == 96 else 48
        lengths = {"8": 10, "10": 11, "12": 11} if width == 96 else {"8": 5, "10": 5, "12": 6}
        return {"status": "passed", "width": width, "config": asdict(study.config(width)),
            "physical_updates": updates, "physical_episode_exposures": updates*96, "micro_batch_size": 32,
            "initial_state_equal": True, "load_at_midpoint_exact": True, "midpoint_equal": True,
            "resumed_endpoint_equal": True, "uninterrupted_repeat_equal": True,
            "all_three_lengths_consumed_per_family": True, "sources_unchanged": True,
            "preserved_inputs_unchanged": True, "source_sha256": {"model": "a"},
            "source_sha256_after": {"model": "a"},
            "bucket_microbatches": {family: dict(lengths) for family in FAMILIES}}

    def test_runtime_success_requires_current_source_and_actual_width(self):
        for width in study.WIDTHS:
            row = self.receipt(width)
            study.validate_profile_receipt(row, width, {"model": "a"})
            with self.assertRaises(ValueError):
                study.validate_profile_receipt(row, width, {"model": "changed"})
            row["config"]["layers"] = 8
            with self.assertRaises(ValueError):
                study.validate_profile_receipt(row, width, {"model": "a"})

    def test_runtime_coverage_cannot_be_claimed_by_boolean_alone(self):
        changed = self.receipt()
        changed["bucket_microbatches"]["color"] = {"8": 16, "10": 16, "12": 0}
        with self.assertRaises(ValueError):
            study.validate_profile_receipt(changed, 96, {"model": "a"})
        changed = self.receipt()
        changed["physical_updates"] -= 1
        with self.assertRaises(ValueError):
            study.validate_profile_receipt(changed, 96, {"model": "a"})
        changed = self.receipt()
        changed["resumed_endpoint_equal"] = False
        with self.assertRaises(ValueError):
            study.validate_profile_receipt(changed, 96, {"model": "a"})

    def test_all_stage_inputs_required_before_replay_reads_banks(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(study, "load_protocol", return_value={}), patch.object(study, "file_hash", return_value="p"), \
                    patch.object(study, "data") as read:
                with self.assertRaisesRegex(ValueError, "all stage endpoints"):
                    study.verify(directory, "calibration")
                read.assert_not_called()

    def test_audit_gates_precede_cuda_and_bank_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(study, "load_protocol", return_value={}), \
                    patch.object(study, "load_selection", return_value={}), \
                    patch("experiments.execution_profile.runtime_profile") as runtime, patch.object(study, "data") as read:
                with self.assertRaises(FileNotFoundError):
                    study.audit(directory)
                read.assert_not_called()
                runtime.assert_not_called()

    def test_selection_payload_is_rederived_from_verified_calibration(self):
        selected = {"schema": study.SCHEMA, "automatic_promotion": False, "protocol_sha256": "h",
            "calibration_verification_sha256": "h", "input_files_sha256": {},
            "selected": {"96": {"rate": .003}}}
        with patch.object(study, "load_verification", return_value={}), \
                patch.object(study, "read", return_value=selected), patch.object(study, "file_hash", return_value="h"), \
                patch.object(study, "completed_inputs", return_value={}), \
                patch.object(study, "selection_data", return_value={"selected": {"96": {"rate": .001}}}):
            with self.assertRaises(ValueError):
                study.load_selection("unused")

    def test_required_identity_hashes_include_job_marker_and_all_curves(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = study.required(directory, "main")
            self.assertEqual(len(paths), 3*(3+len(study.STEPS["main"])))
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(str(index).encode())
            before = study.completed_inputs(directory, "main")
            (Path(directory)/"main/w192/job.json").write_bytes(b"changed")
            self.assertNotEqual(before, study.completed_inputs(directory, "main"))

    def test_tensor_comparison_catches_optimizer_and_stream_drift(self):
        value = {"optimizer": {"moment": torch.tensor([1., 2.])}, "stream": [1, 2, 3]}
        study.tensor_equal(value, copy.deepcopy(value))
        for changed in ({"optimizer": {"moment": torch.tensor([1., 2.1])}, "stream": [1, 2, 3]},
                        {"optimizer": {"moment": torch.tensor([1., 2.])}, "stream": [1, 3, 2]}):
            with self.assertRaises(ValueError):
                study.tensor_equal(value, changed)


if __name__ == "__main__":
    unittest.main()
