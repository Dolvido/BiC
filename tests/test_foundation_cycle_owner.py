"""Owner publication checks plus explicitly pinned tiny CPU-cycle integration.

Set BIC_CYCLE_OWNER_FIXTURE to a JSON file naming a caller-pinned cycle-one
capsule and pinned evaluation-banks file. No formal data or weights are used.
"""
import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_cycle_owner as module
from experiments import foundation_loop as loops
from experiments import foundation_training as training
from experiments.foundation_evaluation import FoundationBank
from experiments.foundation_lifetime_retention import LifetimeRetention
from experiments.foundation_parent_capsule import load_parent_capsule


class OwnerPublicationTests(unittest.TestCase):
    def test_policy_is_prospective_and_cycle_seeds_are_deterministic(self):
        policy = module._policy(9101, dict(stage_updates=10, final_updates=6,
            micro_batch_size=2, rehearsal_every=2), dict(window_updates=66, chunk_updates=4))
        self.assertEqual(module._base_plan(policy, 2), module._base_plan(policy, 2))
        self.assertNotEqual(module._base_plan(policy, 2)["config"], module._base_plan(policy, 3)["config"])
        for args in ((True, {}, None), (1, {"seed": 2}, None), (1, {}, {"chunk_updates": 20})):
            with self.assertRaises(ValueError): module._policy(*args)

    def test_relative_artifacts_cannot_escape_or_use_noncanonical_names(self):
        with tempfile.TemporaryDirectory() as folder:
            owner = module.CycleOwner._empty(Path(folder)/"owner.pt", "cpu")
            self.assertTrue(owner._relative("capsules/a.zip").is_relative_to(Path(folder)))
            for path in ("../escape.zip", "capsules/../a.zip", "capsules\\a.zip", str(Path(folder)/"a.zip")):
                with self.subTest(path=path), self.assertRaises(ValueError): owner._relative(path)

    def fake_owner(self, folder):
        owner = module.CycleOwner._empty(Path(folder)/"owner.pt", "cpu")
        torch.save({"value": "old"}, owner._path)
        owner._head = module._digest(owner._path)
        owner._committed = {"value": "old"}
        owner._state = {"value": "new"}
        owner._check = lambda: None
        owner._snapshot_current = lambda: copy.deepcopy(owner._state)
        return owner

    def test_failed_replacement_keeps_old_commit_and_late_failure_recognizes_new_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            owner = self.fake_owner(folder)
            original = owner._head
            with patch.object(module.os, "replace", side_effect=OSError("before replace")):
                with self.assertRaises(OSError): owner._commit()
            self.assertEqual((owner._head, owner._committed), (original, {"value": "old"}))
            self.assertEqual(module._digest(owner._path), original)
            real = os.replace
            def late(source, destination):
                real(source, destination)
                raise OSError("after replace")
            with patch.object(module.os, "replace", side_effect=late):
                with self.assertRaises(OSError): owner._commit()
            self.assertEqual(owner._committed, {"value": "new"})
            self.assertEqual(owner._head, module._digest(owner._path))
            self.assertFalse(owner._uncertain)

    def test_foreign_publication_poisons_owner_and_stale_writer_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            owner = self.fake_owner(folder)
            def foreign(source, destination):
                Path(destination).write_bytes(b"foreign")
                raise OSError("uncertain replace")
            with patch.object(module.os, "replace", side_effect=foreign):
                with self.assertRaises(OSError): owner._commit()
            self.assertTrue(owner._uncertain)
            self.assertEqual(owner._path.read_bytes(), b"foreign")
        with tempfile.TemporaryDirectory() as folder:
            owner = self.fake_owner(folder)
            owner._path.write_bytes(b"foreign")
            with self.assertRaisesRegex(RuntimeError, "externally"): owner._disk_current()
            self.assertEqual(owner._path.read_bytes(), b"foreign")

    def test_json_receipt_is_exclusive_and_not_visible_before_complete_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"receipt.json"
            with patch.object(module.os, "link", side_effect=OSError("crash before name publication")):
                with self.assertRaises(OSError): module._exclusive_json(path, {"done": True})
            self.assertFalse(path.exists())
            module._exclusive_json(path, {"done": True})
            with self.assertRaises(FileExistsError): module._exclusive_json(path, {"done": False})
            self.assertEqual(json.loads(path.read_text()), {"done": True})

    def test_missing_truncated_or_incomplete_accounting_receipt_blocks_unpinned_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            owner = module.CycleOwner._empty(Path(folder)/"owner.pt", "cpu")
            (Path(folder)/"invocations").mkdir()
            intent = Path(folder)/"invocations"/"attempt.intent.json"
            intent.write_text('{"status":"started"}', encoding="utf8")
            receipt = intent.with_name("attempt.receipt.json")
            for raw in (None, '{"status":', json.dumps(dict(schema=module.INTENT_SCHEMA,
                        intent_sha256=module._digest(intent), owner_sha256="a"*64, status="update_bound"))):
                if raw is not None: receipt.write_text(raw, encoding="utf8")
                self.assertEqual(owner._unresolved(), [intent])
            torch.save({}, owner._path)
            with self.assertRaises(module.UnknownInvocationWork):
                module.CycleOwner.load(owner._path, expected_sha256=module._digest(owner._path))
            recovery = intent.with_name("attempt.recovery.json")
            module._exclusive_json(recovery, dict(schema=module.INTENT_SCHEMA,
                intent_sha256=module._digest(intent), owner_sha256="a"*64,
                status="unknown_work_acknowledged", physical_work_unknown=True))
            self.assertEqual(owner._unresolved(), [intent])
            owner._state = dict(unknown_invocations=[dict(intent="invocations/attempt.intent.json", sha256=module._digest(intent))])
            self.assertEqual(owner._unresolved(), [])


def arithmetic(owner):
    saved = owner.snapshot()["active"]["loop"]["provider"]["learner"]
    return {key: saved[key] for key in ("weights", "optimizer", "cursor", "evidence",
                                      "base_updates", "lifetime_updates", "cycle")}


class OwnerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filename = os.environ.get("BIC_CYCLE_OWNER_FIXTURE")
        if not filename:
            raise unittest.SkipTest("explicit pinned tiny cycle-one fixture required")
        fixture = json.loads(Path(filename).read_text("utf8"))
        banks_path = Path(fixture["banks_path"])
        if module._digest(banks_path) != fixture["banks_sha256"]:
            raise AssertionError("caller-pinned bank artifact differs")
        cls.stack, cls.work = ExitStack(), dict(trainer_attempts=0, completed_optimizer_calls=0,
            drawn_episode_exposures=0, neural_attempted_episode_exposures=0,
            completed_microbatch_episode_exposures=0, unknown_optimizer_reports=0,
            evaluation_bank_attempts=0, evaluation_banks_completed=0,
            evaluation_episode_attempts=0, evaluation_episodes_completed=0)
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.reports, cls.artifacts = [], None
        cls.addClassCleanup(cls.finish)
        actual_step, actual_optimizer, actual_score = training.FoundationTrainer.step, torch.optim.AdamW.step, FoundationBank.score
        def step(trainer):
            cls.work["trainer_attempts"] += 1
            try: return actual_step(trainer)
            finally:
                value = trainer.last_report
                if value is not None:
                    for name in ("drawn_episode_exposures", "neural_attempted_episode_exposures", "completed_microbatch_episode_exposures"):
                        cls.work[name] += value[name] or 0
                    cls.work["unknown_optimizer_reports"] += int(value["physical_optimizer_updates"] is None)
        def optimizer(instance, *args, **kwargs):
            value = actual_optimizer(instance, *args, **kwargs)
            cls.work["completed_optimizer_calls"] += 1
            return value
        def score(bank, *args, **kwargs):
            cls.work["evaluation_bank_attempts"] += 1
            cls.work["evaluation_episode_attempts"] += bank.identity["episodes"]
            value = actual_score(bank, *args, **kwargs)
            cls.work["evaluation_banks_completed"] += 1
            cls.work["evaluation_episodes_completed"] += bank.identity["episodes"]
            return value
        cls.stack.enter_context(patch.object(training.FoundationTrainer, "step", step))
        cls.stack.enter_context(patch.object(torch.optim.AdamW, "step", optimizer))
        cls.stack.enter_context(patch.object(FoundationBank, "score", score))
        cls.banks = json.loads(banks_path.read_text("utf8")) if banks_path.suffix == ".json" else torch.load(banks_path, map_location="cpu", weights_only=True)
        cls.parent = load_parent_capsule(fixture["capsule_path"], expected_sha256=fixture["capsule_sha256"],
            device="cpu", evaluation_banks=cls.banks)
        if cls.parent.identity["cycle"] != 1:
            raise AssertionError("this fixture must be an original completed cycle")
        cls.retention = LifetimeRetention.start(cls.parent)
        cls.plan_config = dict(stage_updates=10, final_updates=6, micro_batch_size=2, rehearsal_every=2)

    @classmethod
    def finish(cls):
        cls.stack.close()
        torch.set_num_threads(cls.threads)
        value = dict(**cls.work, reports=cls.reports, artifacts=cls.artifacts, device="cpu", threads=1,
            formal_data_or_weights_used=False,
            scope="Actual attempts across owner tests including discarded publication work; source fixture learning is accounted in its separate repeated-cycle receipt. No capability claim.")
        print("CYCLE_OWNER_PHYSICAL_WORK="+json.dumps(value, sort_keys=True))
        destination = os.environ.get("BIC_CYCLE_OWNER_ACCOUNTING")
        if destination: Path(destination).write_text(json.dumps(value, indent=2), encoding="utf8")

    def create(self, path):
        return module.CycleOwner.create(path, parent=self.parent, retention=self.retention,
            evaluation_banks=self.banks, plan_config=self.plan_config, root_seed=920100001,
            loop_options=dict(window_updates=66, chunk_updates=4, history_limit=1))

    def remember(self, report):
        self.reports.append({key: report[key] for key in ("status", "physical_optimizer_updates",
            "retained_updates", "discarded_optimizer_updates", "evaluation_banks", "evaluation_episode_exposures",
            "completed_cycles", "owner_sha256", "wall_seconds")})

    def test_two_cycles_stop_at_boundary_and_restart_builds_only_current_parent_and_index(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.create(Path(root)/"primary")
            report = owner.run(max_updates=132, max_cycles=2)
            self.remember(report)
            self.assertEqual((report["physical_optimizer_updates"], report["retained_updates"], report["completed_cycles"]), (132, 132, 2))
            self.assertEqual(report["evaluation_banks"], 24)
            self.assertEqual((owner.status["phase"], owner.status["parent_cycle"], owner.status["lifetime_updates"]), ("boundary", 3, 198))
            self.assertIsNone(owner.snapshot()["active"])
            self.assertEqual(len(list((Path(root)/"primary"/"capsules").glob("*.zip"))), 3)
            with (patch.object(module, "load_parent_capsule", wraps=module.load_parent_capsule) as capsule,
                  patch.object(module, "AuthenticatedPlanIndex", wraps=module.AuthenticatedPlanIndex) as index):
                restored = module.CycleOwner.load(owner._path, expected_sha256=report["owner_sha256"], min_generation=3)
                self.assertEqual((capsule.call_count, index.call_count), (1, 0))
            following = restored.run(max_updates=2, max_cycles=1)
            self.remember(following)
            self.assertEqual((following["retained_updates"], following["completed_cycles"]), (2, 0))
            self.assertEqual(restored.status["active_cycle"], 4)
            clone = Path(root)/"clone"
            shutil.copytree(Path(root)/"primary", clone)
            with (patch.object(module, "load_parent_capsule", wraps=module.load_parent_capsule) as capsule,
                  patch.object(module, "AuthenticatedPlanIndex", wraps=module.AuthenticatedPlanIndex) as index):
                split = module.CycleOwner.load(restored._path, expected_sha256=following["owner_sha256"])
                self.assertEqual((capsule.call_count, index.call_count), (1, 1))
            direct = module.CycleOwner.load(clone/"owner.pt", expected_sha256=following["owner_sha256"])
            self.assertTrue(loops._same_tree(arithmetic(split), arithmetic(direct)))
            self.remember(split.run(max_updates=2)); self.remember(direct.run(max_updates=2))
            self.assertTrue(loops._same_tree(arithmetic(split), arithmetic(direct)))
            self.assertEqual(split._retention.snapshot(), direct._retention.snapshot())
            with self.assertRaisesRegex(RuntimeError, "externally"): restored.run(max_updates=1)
            destination = os.environ.get("BIC_CYCLE_OWNER_ARTIFACTS")
            if destination:
                destination = Path(destination)
                destination.mkdir(parents=True, exist_ok=True)
                preserved = destination/"owner"
                shutil.copytree(Path(root)/"primary", preserved)
                type(self).artifacts = dict(owner_path=str((preserved/"owner.pt").resolve()),
                    owner_sha256=split.status["owner_sha256"], lifetime_updates=split.status["lifetime_updates"],
                    active_cycle=split.status["active_cycle"], cycle_updates=split.status["cycle_updates"],
                    source_sha256=module.source_hashes())
                module._exclusive_json(destination/"final-owner.json", type(self).artifacts)

    def test_zero_cycles_deadline_wrong_pin_and_external_inner_work(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.create(Path(root)/"owner")
            head = owner.status["owner_sha256"]
            for options in (dict(max_cycles=0), dict(deadline=time.monotonic()-1)):
                report = owner.run(**options); self.remember(report)
                self.assertEqual((report["physical_optimizer_updates"], report["evaluation_banks"], report["owner_sha256"]), (0, 0, head))
            with self.assertRaises(ValueError): owner.run()
            with self.assertRaises(ValueError): module.CycleOwner.load(owner._path, expected_sha256="0"*64)
            with self.assertRaises(ValueError): module.CycleOwner.load(owner._path, expected_sha256=head, min_generation=2)
            # Foreign scoring is real work and is captured by the test ledger.
            owner._loop.run(max_updates=0, max_evaluations=1)
            with self.assertRaisesRegex(RuntimeError, "outside its owner"): owner.run(max_updates=1)

    def test_failed_owner_commit_restores_complete_prior_state_and_counts_discarded_work(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.create(Path(root)/"owner")
            baseline = owner.run(max_updates=0); self.remember(baseline)
            before = arithmetic(owner)
            actual = owner._write
            def fail(payload):
                if payload["active"] is not None and payload["active"]["loop"]["provider"]["learner"]["cursor"]:
                    raise OSError("owner commit before publish")
                return actual(payload)
            with patch.object(owner, "_write", side_effect=fail):
                with self.assertRaisesRegex(OSError, "before publish"): owner.run(max_updates=2)
            self.remember(owner.last_report)
            self.assertEqual((owner.last_report["physical_optimizer_updates"], owner.last_report["retained_updates"],
                              owner.last_report["discarded_optimizer_updates"]), (2, 0, 2))
            self.assertTrue(loops._same_tree(arithmetic(owner), before))
            self.assertFalse(owner._unresolved())

    def test_stale_intent_requires_explicit_unknown_recovery_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.create(Path(root)/"owner")
            head = owner.status["owner_sha256"]
            intent = owner._path.parent/"invocations"/"hard-kill.intent.json"
            module._exclusive_json(intent, dict(status="started", scope="deliberate crash fixture"))
            broken = intent.with_name("hard-kill.receipt.json")
            broken.write_text('{"partial":', encoding="utf8")
            with self.assertRaises(module.UnknownInvocationWork):
                module.CycleOwner.load(owner._path, expected_sha256=head)
            recovered = module.CycleOwner.load(owner._path, expected_sha256=head, recovery_policy="acknowledge_unknown")
            self.assertEqual(recovered.status["acknowledged_unknown_invocations"], 1)
            self.assertEqual(broken.read_text("utf8"), '{"partial":')
            self.assertNotEqual(recovered.status["owner_sha256"], head)
            self.assertTrue(intent.with_name("hard-kill.recovery.json").exists())
            self.assertFalse(recovered._unresolved())


if __name__ == "__main__":
    unittest.main(verbosity=2)
