"""Architecture audit provenance and fresh/restart integration on CPU only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments import audit_sequence_study as audit
from experiments import sequence_study as study
from experiments.cognitive_curriculum import generate_cognitive
from experiments.sequence_student import SequenceConfig, build_sequence_student


class SequenceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_prepare_preserves_support_size_split_and_bank_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = audit.prepare(temporary)
            self.assertEqual(audit.prepare(temporary), manifest)
            _, banks = audit.load_prepared(temporary)
            self.assertEqual(len(banks["support"]["arithmetic_updates"]), 128)
            self.assertEqual(set(banks["retained"]), set(study.TRAIN_FAMILIES))
            self.assertTrue(all(len(rows) == 256 for rows in banks["retained"].values()))
            self.assertTrue(all(row["split"] == "audit" and row["family"] == "arithmetic_updates"
                for rows in banks["query"].values() for row in rows))
            support = {audit.transcript(row) for row in banks["support"]["arithmetic_updates"]}
            self.assertFalse(any(audit.transcript(row) in support
                for rows in banks["query"].values() for row in rows))
            banks["query"]["level_2"][0]["turns"][0]["text"] = "changed"
            (Path(temporary) / "frozen_banks.json").write_text(json.dumps(banks), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "bank mismatch"):
                audit.load_prepared(temporary)

    def test_rate_selection_needs_measured_denominators_and_fixed_grid(self):
        self.assertEqual(audit.choose_rate({.003: .5, .001: .5, .0003: .5}), .0003)
        self.assertEqual(audit.calibration_score({"macro_pair_accuracy": .2,
                                                "macro_later_known_accuracy": .8}), .5)
        for value in (None, float("nan"), float("inf"), -1.):
            with self.assertRaises(ValueError):
                audit.calibration_score({"macro_pair_accuracy": value, "macro_later_known_accuracy": .5})
        with self.assertRaises(ValueError):
            audit.choose_rate({.001: .5})

    def test_regional_fresh_adaptation_loads_full_weights_and_resets_moments(self):
        rows = generate_cognitive(82, 2, family="arithmetic_updates")
        banks = {"arithmetic_updates": rows}
        for architecture in ("recurrent", "episodic"):
            parent = study.make_trainer(architecture, banks, seed=2501, batch_size=2, learning_rate=.001)
            parent.step("arithmetic_updates")
            self.assertTrue(parent.optimizer.state)
            adapted = audit.fresh_adaptation(architecture, parent.model.state_dict(), banks, rate=.0003)
            self.assertEqual(checkpoint_digest(parent.model), checkpoint_digest(adapted.model))
            self.assertFalse(adapted.optimizer.state)
            self.assertEqual(adapted.updates, 0)
            self.assertEqual(adapted.aux_weight, 0.)
            self.assertEqual(adapted.optimizer.param_groups[0]["lr"], .0003)
            self.assertTrue(torch.equal(adapted.generators["arithmetic_updates"].get_state(),
                                       torch.Generator().manual_seed(3501).get_state()))

    def test_full_regional_cpu_restart_both_modes(self):
        episode = generate_cognitive(42, 2, family="arithmetic_updates", level=2)[0]
        for architecture in ("recurrent", "episodic"):
            model = audit.build_model(architecture, 2501)
            result = audit.regional_restart(model.state_dict(), architecture, episode)
            self.assertTrue(result["exact"])
            self.assertTrue(result["includes_auxiliary_weights"])
            self.assertEqual(result["checkpoint_sha256"], checkpoint_digest(model))

    def test_sequence_cpu_restart_preserves_prefix_and_config(self):
        model = build_sequence_student(2501, config=SequenceConfig(width=16, layers=1, heads=2, feedforward=32))
        episode = generate_cognitive(40, 2, family="arithmetic_updates", level=2)[0]
        result = audit.sequence_restart(model, episode)
        self.assertTrue(result["exact"])
        self.assertEqual(result["checked_remaining_turns"], 3)
        self.assertIn("observed prefix", result["state_kind"])

    def write_completed_fixture(self, root, frozen, architecture, phase="calibration"):
        """Counter fixtures validate admission only; they do not claim training occurred."""
        seed, steps = (2401, 600) if phase == "calibration" else (2501, 3600)
        path = (study.calibration_path(root, architecture, .001) if phase == "calibration"
                else root / "main" / architecture)
        path.mkdir(parents=True)
        protocol = {"schema": study.SCHEMA, "study": frozen, "architecture": architecture,
            "phase": phase, "seed": seed, "steps": steps, "batch_size": 64,
            "checkpoint_every": 600, "learning_rate": .001,
            "source_sha256": frozen["source_sha256"],
            "within_family_draw_sha256": frozen[f"{phase}_draw_sha256"]}
        model = audit.build_model(architecture, seed)
        weights = model.state_dict()
        counts = dict.fromkeys(study.TRAIN_FAMILIES, steps // 3)
        datasets = json.loads((root / "datasets.json").read_text())
        report = {"protocol": protocol, "updates": steps, "family_updates": counts,
            "examples": steps * 64, "stop_reason": "planned_updates", "datasets": datasets,
            "heldout_evaluation_performed": False, "checkpoint_sha256": checkpoint_digest(model)}
        saved = {"schema": study.SCHEMA, "protocol": protocol,
            "training": {"updates": steps, "family_updates": counts, "weights": weights}}
        for name, value in (("protocol.json", protocol), ("report.json", report), ("datasets.json", datasets)):
            (path / name).write_text(json.dumps(value), encoding="utf8")
        torch.save({"weights": weights}, path / "initial.pt")
        torch.save(saved, path / "latest.pt")
        return path, report, saved

    def test_verifier_uses_actual_nested_protocol_root_sources_and_exact_model_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = study.prepare(root)
            manifest, _ = audit.load_prepared(root / "audit")
            for architecture in study.ARCHITECTURES:
                path, _, _ = self.write_completed_fixture(root, frozen, architecture)
                checked = audit.verify_run(path, manifest, phase="calibration", architecture=architecture, rate=.001)
                self.assertEqual(checked["architecture"], architecture)
                self.assertFalse((path / "source").exists())
            path, report, saved = self.write_completed_fixture(root, frozen, "recurrent", phase="main")
            audit.verify_run(path, manifest, phase="main", architecture="recurrent", rate=.001)
            report["updates"] = 3599
            (path / "report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                audit.verify_run(path, manifest, phase="main", architecture="recurrent", rate=.001)
            report["updates"] = 3600
            changed = copy.deepcopy(report["protocol"])
            changed["within_family_draw_sha256"] = {family: "wrong" for family in study.TRAIN_FAMILIES}
            report["protocol"] = saved["protocol"] = changed
            (path / "protocol.json").write_text(json.dumps(changed))
            (path / "report.json").write_text(json.dumps(report))
            torch.save(saved, path / "latest.pt")
            with self.assertRaisesRegex(ValueError, "sample streams"):
                audit.verify_run(path, manifest, phase="main", architecture="recurrent", rate=.001)

    def test_incomplete_study_cannot_reach_any_heldout_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            study.prepare(temporary)
            with patch.object(study, "evaluate", side_effect=AssertionError("premature heldout scoring")):
                with self.assertRaises(FileNotFoundError):
                    audit.audit(temporary)


if __name__ == "__main__":
    unittest.main()
