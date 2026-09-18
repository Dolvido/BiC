"""Phase integrity fixtures only: no model construction, inference or updates.

Canonical bank/gradient verification has its own tests. These fixtures stand in
for already-verified receipts and mutate actual on-disk phase artifacts to test
the orchestration boundary, identity routing and calibration-context reuse.
"""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_variant_study as study
from experiments import foundation_variant_analysis as analysis
from experiments import foundation_metrics
from experiments.train_cognitive import atomic_json as _atomic_json


def atomic_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, value)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def metric_check(metrics, banks, config, role, control="normal"):
    """Stand-in for previously authenticated banks; assert correct role routing."""
    if (metrics["role"] != role or metrics["control"] != control
            or banks != {"fixture_role": role} or config != study.CONFIG):
        raise ValueError("fixture canonical role/control/config mismatch")
    return True


def choose_fixture_rate(candidates):
    if set(map(float, candidates)) != set(study.RATES):
        raise ValueError("incomplete fixture calibration rates")
    chosen = max(candidates, key=lambda rate: (candidates[rate]["score"], -float(rate)))
    return dict(rate=float(chosen), rule="fixture count score then lower rate")


class VariantStudyGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = dict(source_sha256={"fixture.py": digest("source")},
                             execution_profile={"strict_fixture": True})
        for name in ("protocol.json", "preparation.json", "data-verification/receipt.json"):
            atomic_json(self.root/name, {"fixture": name})
        self.data = {"stages": {
            "calibration": {"banks": {"dev": {"fixture_role": "dev"}}},
            "main": {"banks": {role: {"fixture_role": role} for role in ("dev", "audit", "train_fit")}},
        }}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(study, "load_protocol", return_value=self.protocol))
        self.stack.enter_context(patch.object(study, "_data_gate", return_value={}))
        self.stack.enter_context(patch.object(study.data_api, "load", return_value=self.data))
        self.metrics = self.stack.enter_context(patch.object(foundation_metrics, "validate_metrics", side_effect=metric_check))
        self.rank = self.stack.enter_context(patch.object(analysis, "choose_rate", side_effect=choose_fixture_rate))
        self.stack.enter_context(patch.object(study, "build_sequence_student", side_effect=AssertionError("no model work allowed")))
        self.stack.enter_context(patch.object(study, "build_hierarchical_sequence_student", side_effect=AssertionError("no model work allowed")))
        self.stack.enter_context(patch.object(study, "VariantFoundationTrainer", side_effect=AssertionError("no optimizer work allowed")))

    def read(self, relative):
        return json.loads((self.root/relative).read_text(encoding="utf8"))

    def _score(self, role, job, control="normal"):
        desired = .0003 if job["architecture"] == "flat" else .003
        return dict(metrics=dict(role=role, control=control, score=10 if job["rate"] == desired else 1),
                    cost={"fixture_scoring": 0})

    def phase(self, stage, selection=None):
        jobs = study.jobs(stage, selection)
        protocol_hash = study.old.file_hash(self.root/"protocol.json")
        for arch in study.ARCHITECTURES:
            atomic_json(self.root/stage/arch/"worker.json", dict(schema=study.SCHEMA, status="completed",
                stage=stage, architecture=arch, automatic_promotion=False, protocol_sha256=protocol_hash,
                execution_profile=self.protocol["execution_profile"],
                completed_jobs=[job["id"] for job in jobs if job["architecture"] == arch]))
        for job in jobs:
            folder = self.root/stage/job["architecture"]/job["id"]
            folder.mkdir(parents=True)
            (folder/"steps.jsonl").write_text("Canonical journal fixture, not optimizer execution.\n")
            hashes = {}
            for step in study.STEPS[stage]:
                name = f"checkpoint-{step:06d}.pt"
                (folder/name).write_bytes(f"opaque {job['id']} {step}".encode())
                hashes[name] = study.old.file_hash(folder/name)
            atomic_json(folder/"receipt.json", dict(schema=study.SCHEMA, status="completed", stage=stage,
                job=job, physical_work_unknown=False, retained_updates=study.TOTALS[stage],
                automatic_promotion=False, protocol_sha256=protocol_hash,
                execution_profile=self.protocol["execution_profile"], checkpoints=hashes,
                journal_sha256=study.old.file_hash(folder/"steps.jsonl")))
        inputs, _ = study._inputs(self.root, stage, self.protocol, jobs, selection)
        verified = dict(schema=study.SCHEMA, status="completed", stage=stage,
            protocol_sha256=protocol_hash, neural_training_or_inference=False, automatic_promotion=False,
            input_file_sha256=inputs, source_sha256=self.protocol["source_sha256"],
            results={job["id"]: dict(job=job, exact_official_checkpoint_restores=True,
                weights_sha256={str(step): digest(job["id"]+str(step)) for step in study.STEPS[stage]}) for job in jobs})
        atomic_json(self.root/"verification"/stage/"receipt.json", verified)
        folder = self.root/"evaluation"/stage
        folder.mkdir(parents=True)
        (folder/"score-intents.jsonl").write_text("Completed score fixture, not inference.\n")
        binding = dict(schema=study.SCHEMA, stage=stage, protocol_sha256=protocol_hash,
            verification_sha256=study.old.file_hash(self.root/"verification"/stage/"receipt.json"),
            execution_profile=self.protocol["execution_profile"], input_file_sha256=inputs, automatic_promotion=False)
        results = {}
        for job in jobs:
            curve = [dict(updates=step, weights_sha256=verified["results"][job["id"]]["weights_sha256"][str(step)],
                          **self._score("dev", job))
                     for step in (study.STEPS[stage] if stage == "main" else (study.TOTALS[stage],))]
            value = dict(**binding, job=job, status="completed", development_curve=curve)
            if stage == "main":
                producer = dict(updates=study.TOTALS[stage], architecture=job["architecture"],
                    weights_sha256=verified["results"][job["id"]]["weights_sha256"][str(study.TOTALS[stage])])
                for role in ("audit", "train_fit"):
                    value[role] = dict(**producer, **self._score(role, job))
                value["controls"] = {control: dict(**producer, **self._score("audit", job, control))
                                     for control in ("blank", "reset")}
            results[job["id"]] = value
            atomic_json(folder/(job["id"]+".json"), value)
        report = dict(**binding, status="completed", results=results,
            job_files_sha256={name: study.old.file_hash(folder/(name+".json")) for name in results},
            score_journal_sha256=study.old.file_hash(folder/"score-intents.jsonl"))
        atomic_json(folder/"report.json", report)
        return report

    def selected(self):
        self.phase("calibration")
        return study.select(self.root)

    def rewrite_result(self, stage, job, value):
        """Reseal outer JSON hashes so tests reach semantic producer validation."""
        folder = self.root/"evaluation"/stage
        atomic_json(folder/(job+".json"), value)
        report = self.read(f"evaluation/{stage}/report.json")
        report["results"][job] = value
        report["job_files_sha256"][job] = study.old.file_hash(folder/(job+".json"))
        atomic_json(folder/"report.json", report)

    def test_incomplete_phase_cannot_enter_scoring_or_model_setup(self):
        self.phase("calibration")
        worker_path = "calibration/hierarchical/worker.json"
        original = self.read(worker_path)
        cases = [dict(original, status="running"), dict(original, completed_jobs=original["completed_jobs"][:-1])]
        for changed in cases:
            atomic_json(self.root/worker_path, changed)
            with patch.object(study, "_runtime", side_effect=AssertionError("runtime entered before completion")), \
                 self.assertRaisesRegex(ValueError, "complete architecture workers"):
                study.evaluate(self.root, "calibration")
        atomic_json(self.root/worker_path, original)
        job = study.jobs("calibration")[0]
        path = f"calibration/{job['architecture']}/{job['id']}/receipt.json"
        receipt = self.read(path)
        receipt["physical_work_unknown"] = True
        atomic_json(self.root/path, receipt)
        with patch.object(study, "_runtime", side_effect=AssertionError("runtime entered before completion")), \
             self.assertRaisesRegex(ValueError, "six exact completed jobs"):
            study.evaluate(self.root, "calibration")

    def test_missing_phase_checkpoint_or_verification_is_not_usable(self):
        self.phase("calibration")
        verified = self.read("verification/calibration/receipt.json")
        original = copy.deepcopy(verified)
        verified["results"].pop(next(iter(verified["results"])))
        atomic_json(self.root/"verification/calibration/receipt.json", verified)
        with self.assertRaisesRegex(ValueError, "all completed phase"):
            study._verified(self.root, "calibration", self.protocol)
        atomic_json(self.root/"verification/calibration/receipt.json", original)
        job = study.jobs("calibration")[0]
        path = self.root/"calibration"/job["architecture"]/job["id"]/"checkpoint-000510.pt"
        path.write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            study._verified(self.root, "calibration", self.protocol)

    def test_main_declared_jobs_use_sealed_architecture_specific_rates(self):
        selection = self.selected()
        declared, authenticated = study._phase_choice(self.root, "main", self.protocol)
        self.assertEqual(authenticated, selection)
        self.assertEqual({row["rate"] for row in declared if row["architecture"] == "flat"}, {.0003})
        self.assertEqual({row["rate"] for row in declared if row["architecture"] == "hierarchical"}, {.003})
        selection["selected"]["flat"]["rate"] = .001
        atomic_json(self.root/"selection.json", selection)
        with self.assertRaisesRegex(ValueError, "ranking changed"):
            study._phase_choice(self.root, "main", self.protocol)

    def test_main_train_resolves_selection_once_for_all_three_jobs_without_neural_work(self):
        self.selected()
        original = study.load_selection
        with patch.object(study, "load_selection", wraps=original) as choice, \
             patch.object(study, "_runtime", return_value=self.protocol["execution_profile"]), \
             patch.object(study, "_index", return_value=SimpleNamespace(construction={"fixture": True})), \
             patch.object(study, "_train_job", return_value={}) as jobs, \
             patch.object(torch.cuda, "empty_cache", return_value=None):
            receipt = study.train(self.root, "main", "flat")
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(choice.call_count, 1)
        self.assertEqual(jobs.call_count, 3)
        self.assertEqual({call.args[2]["rate"] for call in jobs.call_args_list}, {.0003})
        self.assertEqual({call.args[2]["seed"] for call in jobs.call_args_list}, set(study.SEEDS))

    def test_main_result_validation_reuses_one_calibration_context_not_one_per_job(self):
        selection = self.selected()
        self.phase("main", selection)
        original = study._evaluated
        phases = []
        def tracked(directory, stage, protocol):
            phases.append(stage)
            return original(directory, stage, protocol)
        self.rank.reset_mock()
        with patch.object(study, "_evaluated", side_effect=tracked):
            report = original(self.root, "main", self.protocol)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(phases, ["calibration"])
        self.assertEqual(self.rank.call_count, 2)

    def test_final_metrics_reject_wrong_weights_cursor_architecture_role_and_control(self):
        selection = self.selected()
        report = self.phase("main", selection)
        job = next(iter(report["results"]))
        original = report["results"][job]
        cases = []
        for field, value in (("weights_sha256", "f"*64), ("updates", 3071), ("architecture", "hierarchical")):
            changed = copy.deepcopy(original); changed["audit"][field] = value; cases.append(changed)
        changed = copy.deepcopy(original); changed["train_fit"]["metrics"]["role"] = "audit"; cases.append(changed)
        changed = copy.deepcopy(original); changed["controls"]["reset"]["weights_sha256"] = "e"*64; cases.append(changed)
        changed = copy.deepcopy(original); changed["controls"]["blank"]["metrics"]["control"] = "normal"; cases.append(changed)
        changed = copy.deepcopy(original); changed["development_curve"][0]["weights_sha256"] = "d"*64; cases.append(changed)
        for changed in cases:
            self.rewrite_result("main", job, changed)
            with self.assertRaises(ValueError):
                study._evaluated(self.root, "main", self.protocol)
        self.rewrite_result("main", job, original)
        self.assertEqual(study._evaluated(self.root, "main", self.protocol)["status"], "completed")

    def test_worker_checkpoint_and_score_runtime_cannot_change_silently(self):
        self.phase("calibration")
        worker_path = "calibration/flat/worker.json"
        value = self.read(worker_path); original = copy.deepcopy(value)
        value["execution_profile"] = {"strict_fixture": False}; atomic_json(self.root/worker_path, value)
        with self.assertRaisesRegex(ValueError, "worker runtime"):
            study._inputs(self.root, "calibration", self.protocol)
        atomic_json(self.root/worker_path, original)
        report = self.read("evaluation/calibration/report.json")
        report["execution_profile"] = {"strict_fixture": False}
        atomic_json(self.root/"evaluation/calibration/report.json", report)
        with self.assertRaisesRegex(ValueError, "evaluation runtime"):
            study._evaluated(self.root, "calibration", self.protocol)
        job = study.jobs("calibration")[0]
        saved = dict(schema=study.SCHEMA, stage="calibration", job=job,
            protocol_sha256=study.old.file_hash(self.root/"protocol.json"), learner={"cursor":0},
            weights_sha256="a"*64, execution_profile={"strict_fixture": False})
        with patch.object(torch, "load", return_value=saved), self.assertRaisesRegex(ValueError, "checkpoint runtime"):
            study._checkpoint(self.root, "calibration", job, 0, self.protocol)
        with patch.object(torch, "get_num_threads", return_value=1), \
             patch.object(torch, "get_num_interop_threads", return_value=1), \
             patch("experiments.execution_profile.runtime_profile", return_value={"strict_fixture": False}), \
             self.assertRaisesRegex(ValueError, "runtime differs"):
            study._runtime(self.protocol)

    def test_selection_binds_all_calibration_inputs_and_detects_later_change(self):
        selection = self.selected()
        job = study.jobs("calibration")[0]
        checkpoint = f"calibration/{job['architecture']}/{job['id']}/checkpoint-000510.pt"
        required = {checkpoint, "evaluation/calibration/report.json", "evaluation/calibration/score-intents.jsonl",
                    "verification/calibration/receipt.json", "evaluation/calibration/"+job["id"]+".json"}
        self.assertTrue(required <= set(selection["input_file_sha256"]))
        saved = study.old.file_hash(self.root/"selection.json")
        study._selection_unchanged(self.root, selection, saved)
        (self.root/checkpoint).write_bytes(b"late changed checkpoint")
        with self.assertRaisesRegex(ValueError, "authenticated inputs changed"):
            study._selection_unchanged(self.root, selection, saved)


if __name__ == "__main__":
    unittest.main()
