"""Model-free orchestration contracts; synthetic envelopes are never adopted."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import foundation_objective_study as study


class ObjectiveStudyContracts(unittest.TestCase):
    def test_exact_configuration_and_prescribed_order(self):
        from experiments.foundation_objective_runtime_probe import CONFIG
        from experiments.foundation_variant_study import CONFIG as previous
        self.assertEqual(study.CONFIG_VALUES, asdict(CONFIG))
        self.assertEqual(study.CONFIG_VALUES, asdict(previous))
        self.assertEqual(asdict(study._config()), asdict(CONFIG))
        calibration = study.jobs("calibration")
        selection = {"selected": {o: {"rate": .001} for o in study.OBJECTIVES}}
        main = study.jobs("main", selection)
        self.assertEqual([(j["rate"], j["objective_id"]) for j in calibration],
                         [(r, o) for r in study.RATES for o in study.OBJECTIVES])
        self.assertEqual([(j["seed"], j["objective_id"]) for j in main],
                         [(s, o) for s in study.SEEDS for o in study.OBJECTIVES])
        updates = len(calibration)*study.TOTALS["calibration"] + len(main)*study.TOTALS["main"]
        self.assertEqual(updates, study.contract()["formal_updates"])
        self.assertEqual(updates*96, study.contract()["formal_episode_exposures"])

    def test_source_closure_covers_producer_and_current_trainer(self):
        from experiments.foundation_objective_training import source_hashes as trainer_sources
        sources = study.source_hashes()
        root = Path(study.__file__).resolve().parents[1]
        for name, digest in trainer_sources().items():
            self.assertEqual(sources[name], digest)
        for name in ("experiments/foundation_objective_study.py",
                     "experiments/foundation_objective_runtime_probe.py", study.PROTOCOL_SOURCE):
            self.assertEqual(sources[name], hashlib.sha256((root/name).read_bytes()).hexdigest())
        with patch.object(study, "source_hashes", return_value={**sources, "changed.py": "0"*64}):
            with self.assertRaisesRegex(ValueError, "sources changed"):
                study._sources({"source_sha256": sources})

    def test_exclusive_stage_and_finite_allowance(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = study._Stage(temp, "calibration", 1)
            before = (stage.folder/"stage.json").read_bytes()
            with self.assertRaises(FileExistsError):
                study._Stage(temp, "calibration", 1)
            self.assertEqual((stage.folder/"stage.json").read_bytes(), before)
            for allowance in (True, 0, -1, float("nan"), float("inf"), 14401):
                with self.subTest(allowance=allowance), self.assertRaises(ValueError):
                    study._Stage(temp, "invalid", allowance)
            stage.deadline = 0
            called = []
            with self.assertRaises(TimeoutError):
                stage.perform("forbidden", lambda: called.append(True))
            self.assertEqual(called, [])
            self.assertFalse((stage.folder/"phase-intents.jsonl").exists())

    def test_failed_protocol_has_durable_terminal_and_phase_evidence(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(
                study, "load_protocol", side_effect=ValueError("fixture proof refused")):
            with self.assertRaisesRegex(ValueError, "fixture proof refused"):
                study.run_stage(temp, temp, "calibration", max_seconds=1)
            receipt = study._read(Path(temp)/"calibration/stage.json")
            events = [json.loads(line) for line in
                      (Path(temp)/"calibration/phase-intents.jsonl").read_text().splitlines()]
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["pid"], os.getpid())
            self.assertTrue(receipt["started_utc"] and receipt["ended_utc"])
            self.assertEqual(receipt["failed_phase"], "authenticate-protocol-data-proof")
            self.assertIsNone(receipt["active_phase"])
            self.assertEqual(receipt["completed_jobs"], [])
            self.assertEqual([r["event"] for r in events], ["started", "failed"])
            self.assertIn("fixture proof refused", events[-1]["error"])

    def test_corrupt_or_incomplete_runtime_proof_refused_without_models(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"probe.json"
            path.write_bytes(b"not JSON")
            with self.assertRaisesRegex(ValueError, "caller-pinned"):
                study._proof(temp, "0"*64)
            with self.assertRaises(json.JSONDecodeError):
                study._proof(temp, study._file(path))
            path.write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "complete matching"):
                study._proof(temp, study._file(path))

    def test_proof_delegates_full_validation_and_checks_study_compatibility(self):
        value = dict(config=dict(study.CONFIG_VALUES), objective_ids=list(study.OBJECTIVES), rates=list(study.RATES))
        with patch("experiments.foundation_objective_runtime_probe.load_proof", return_value=value) as loader:
            self.assertEqual(study._proof("fixture", "a"*64), value)
            loader.assert_called_once_with("fixture", expected_sha256="a"*64)
        bad = deepcopy(value)
        bad["config"]["max_output_bytes"] = 48
        with patch("experiments.foundation_objective_runtime_probe.load_proof", return_value=bad):
            with self.assertRaisesRegex(ValueError, "config differ"):
                study._proof("fixture", "a"*64)

    def test_checkpoint_intents_must_match_every_official_image(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"journal.jsonl"
            checkpoints = {f"checkpoint-{n:06d}.pt": str(n).zfill(64) for n in study.STEPS["calibration"]}
            events = []
            for step in study.STEPS["calibration"]:
                events.extend([dict(event="started", updates=step, utc="fixture"),
                    dict(event="completed", updates=step, sha256=checkpoints[f"checkpoint-{step:06d}.pt"])])
            def write(rows):
                path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf8")
            write(events)
            study._checkpoint_journal(path, "calibration", checkpoints)
            corruptions = [events[:-1], deepcopy(events), deepcopy(events)]
            corruptions[1][-1]["sha256"] = "f"*64
            corruptions[2][0]["updates"] = False
            for rows in corruptions:
                write(rows)
                with self.assertRaises(ValueError):
                    study._checkpoint_journal(path, "calibration", checkpoints)

    def test_calibration_gate_binds_producers_and_complete_inventory(self):
        # Only gate plumbing: no metrics, checkpoint decode, learner or index.
        with tempfile.TemporaryDirectory() as temp, patch(
                "experiments.foundation_variant_analysis.choose_rate", return_value={"rate": .001}):
            root = Path(temp)
            protocol = dict(source_sha256={"fixture.py": "a"*64}, execution_profile={"fixture": True})
            pin = study._write(root/"protocol.json", protocol)
            folder = root/"calibration"
            folder.mkdir()
            declared, verified, scored = study.jobs("calibration"), {}, {}
            for job in declared:
                prefix = folder/job["id"]
                prefix.mkdir()
                checkpoints, events = {}, []
                for step in study.STEPS["calibration"]:
                    name = f"checkpoint-{step:06d}.pt"
                    (prefix/name).write_bytes(b"synthetic never decoded")
                    checkpoints[name] = study._file(prefix/name)
                    events.extend([dict(event="started", updates=step, utc="fixture"),
                                   dict(event="completed", updates=step, sha256=checkpoints[name])])
                (prefix/"steps.jsonl").write_text("synthetic never replayed", encoding="utf8")
                (prefix/"checkpoint-intents.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf8")
                study._write(prefix/"receipt.json", dict(schema=study.SCHEMA, status="completed", stage="calibration",
                    job=job, protocol_sha256=pin, automatic_promotion=False, physical_work_unknown=False,
                    retained_updates=510, checkpoints=checkpoints, journal_sha256=study._file(prefix/"steps.jsonl"),
                    checkpoint_journal_sha256=study._file(prefix/"checkpoint-intents.jsonl")))
                verified[job["id"]] = dict(job=job, exact_official_checkpoint_restores=True,
                                           weights_sha256={"0": "b"*64, "510": "c"*64})
                scored[job["id"]] = dict(job=job, status="completed", development_curve=[dict(updates=510,
                    objective_id=job["objective_id"], weights_sha256="c"*64, metrics={"fixture": True})])
            inputs, _ = study._inputs(root, "calibration", declared, pin)
            common = dict(schema=study.SCHEMA, status="completed", stage="calibration", protocol_sha256=pin,
                          automatic_promotion=False, **protocol)
            verification = dict(**common, input_file_sha256=inputs, results=verified, neural_training_or_inference=False)
            study._write(folder/"verification.json", verification)
            evaluation = dict(**common, input_file_sha256=inputs, results=scored,
                              verification_sha256=study._file(folder/"verification.json"))
            selection = dict(schema=study.SCHEMA, status="completed", protocol_sha256=pin,
                             selected={o: {"rate": .001} for o in study.OBJECTIVES}, automatic_promotion=False)
            stage = dict(**common, jobs=declared, completed_jobs=[j["id"] for j in declared])
            def seal():
                study._write(folder/"evaluation.json", evaluation, replace=True)
                selection["evaluation_sha256"] = study._file(folder/"evaluation.json")
                study._write(folder/"selection.json", selection, replace=True)
                stage["artifacts_sha256"] = {p.relative_to(root).as_posix(): study._file(p)
                    for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "stage.json"}
                study._write(folder/"stage.json", stage, replace=True)
            seal()
            self.assertEqual(study._selection(root, protocol), selection)
            row = scored[declared[0]["id"]]["development_curve"][0]
            row["weights_sha256"] = "d"*64
            seal()
            with self.assertRaisesRegex(ValueError, "verified endpoint"):
                study._selection(root, protocol)
            row["weights_sha256"] = "c"*64
            seal()
            stage["artifacts_sha256"].pop(next(iter(stage["artifacts_sha256"])))
            study._write(folder/"stage.json", stage, replace=True)
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                study._selection(root, protocol)


if __name__ == "__main__":
    unittest.main()
