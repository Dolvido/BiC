"""Actual tiny completed-loop handoffs; CPU only, no formal study data."""
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_cycle_training as module
from experiments import foundation_curriculum as curriculum
from experiments import foundation_loop as loops
from experiments import foundation_training as legacy
from experiments.foundation_admission import repair_plan
from experiments.foundation_evaluation import FoundationBank
from experiments.foundation_loop import FoundationLoop
from experiments.foundation_plan import build_plan, materialize_bundle
from experiments.foundation_plan_index import AuthenticatedPlanIndex
from experiments.foundation_variant_provider import VariantFoundationPracticeProvider
from experiments.foundation_variant_training import VariantFoundationTrainer
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig


def learning(payload):
    result = copy.deepcopy(payload)
    result.pop("timing")
    return result


class CycleTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.stack = ExitStack()
        cls.work = dict.fromkeys(("trainer_step_attempts", "completed_optimizer_calls", "unknown_optimizer_reports",
            "drawn_episode_exposures", "neural_attempted_episode_exposures", "completed_microbatch_episode_exposures",
            "evaluation_bank_attempts", "evaluation_banks_completed", "evaluation_episode_attempts", "evaluation_episodes_completed"), 0)
        actual_step, actual_optimizer, actual_score = legacy.FoundationTrainer.step, torch.optim.AdamW.step, FoundationBank.score
        def step(trainer):
            cls.work["trainer_step_attempts"] += 1
            try: return actual_step(trainer)
            finally:
                report = trainer.last_report
                if report is not None:
                    for name in ("drawn_episode_exposures", "neural_attempted_episode_exposures", "completed_microbatch_episode_exposures"):
                        cls.work[name] += report[name] or 0
                    cls.work["unknown_optimizer_reports"] += int(report["physical_optimizer_updates"] is None)
        def optimizer(instance, *args, **kwargs):
            result = actual_optimizer(instance, *args, **kwargs)
            cls.work["completed_optimizer_calls"] += 1
            return result
        def score(bank, *args, **kwargs):
            cls.work["evaluation_bank_attempts"] += 1
            cls.work["evaluation_episode_attempts"] += bank.identity["episodes"]
            result = actual_score(bank, *args, **kwargs)
            cls.work["evaluation_banks_completed"] += 1
            cls.work["evaluation_episodes_completed"] += bank.identity["episodes"]
            return result
        cls.stack.enter_context(patch.object(legacy.FoundationTrainer, "step", step))
        cls.stack.enter_context(patch.object(torch.optim.AdamW, "step", optimizer))
        cls.stack.enter_context(patch.object(FoundationBank, "score", score))
        cls.addClassCleanup(cls.finish)
        cls.config = SequenceConfig(width=8, layers=4, heads=2, feedforward=16, max_turns=12)
        cls.banks = {role: {f"probe/{family}/d0/direct/t{turns}": curriculum.generate_pair(
            family, 854200001+100*role_index+index, depth=0, turns=turns, split="dev")
            for index, (family, turns) in enumerate(zip(curriculum.FAMILIES, (8, 10, 12)))}
            for role_index, role in enumerate(("development", "retention"))}
        cls.protection = sorted({transcript_digest(row) for named in cls.banks.values() for rows in named.values() for row in rows})
        cls.plan, cls.receipt = repair_plan(build_plan(seed=854100001, stage_updates=10,
            final_updates=6, micro_batch_size=2, rehearsal_every=2, ordering_seed=854100002), [])
        cls.index = AuthenticatedPlanIndex(cls.plan, admission_receipt=cls.receipt,
            admission_protected_transcripts=[], protected_transcripts=cls.protection)
        cls.transcripts = sorted({transcript_digest(row) for bundle_id in range(len(cls.plan["bundles"]))
            for rows in materialize_bundle(cls.plan, bundle_id).values() for row in rows})
        cls.completed, cls.parents, cls.parent_reports = {}, {}, {}
        for architecture in ("flat", "hierarchical"):
            loop = FoundationLoop(cls.provider(architecture), window_updates=66, chunk_updates=4, history_limit=2)
            report = loop.run(max_updates=66)
            if loop.status["phase"] != "complete":
                raise AssertionError("actual parent fixture did not complete")
            cls.completed[architecture] = loop
            cls.parents[architecture] = module.capture_completed_parent(loop, training_transcripts=cls.transcripts)
            cls.parent_reports[architecture] = {key: report[key] for key in (
                "completed_updates", "physical_optimizer_updates", "drawn_episode_exposures", "evaluation_banks",
                "evaluation_episode_exposures", "wall_seconds", "training_seconds", "evaluation_seconds")}
        cls.exclusions = cls.parents["flat"].protected_transcripts
        cls.next_plan, cls.next_receipt = repair_plan(build_plan(seed=854300001, stage_updates=10,
            final_updates=6, micro_batch_size=2, rehearsal_every=2, ordering_seed=854300002), cls.exclusions)
        cls.next_index = AuthenticatedPlanIndex(cls.next_plan, admission_receipt=cls.next_receipt,
            admission_protected_transcripts=cls.exclusions, protected_transcripts=cls.exclusions)

    @classmethod
    def finish(cls):
        cls.stack.close()
        torch.set_num_threads(cls.threads)
        report = dict(**cls.work, completed_parent_loops=getattr(cls, "parent_reports", {}),
            device="cpu", threads=1, formal_data_or_weights_used=False,
            scope="Actual optimizer calls, all attempted training exposure and bank scoring, including deliberately discarded work. Repeated construction/restore is no training. Tiny engineering fixtures do not demonstrate capability gain.")
        print("CYCLE_TRAINING_PHYSICAL_WORK="+json.dumps(report, sort_keys=True))
        destination = os.environ.get("BIC_CYCLE_TRAINING_ACCOUNTING")
        if destination: Path(destination).write_text(json.dumps(report, indent=2), encoding="utf8")

    @classmethod
    def provider(cls, architecture):
        return VariantFoundationPracticeProvider(cls.plan, architecture=architecture, seed=8541,
            evaluation_banks=cls.banks, admission_protected_transcripts=[], protected_transcripts=cls.protection,
            admission_receipt=cls.receipt, plan_index=cls.index, config=cls.config)

    def trainer(self, parent_architecture="flat", **options):
        args = dict(parent=self.parents[parent_architecture], admission_protected_transcripts=self.exclusions,
            protected_transcripts=self.exclusions, admission_receipt=self.next_receipt, plan_index=self.next_index)
        args.update(options)
        return module.CycleFoundationTrainer(self.next_plan, **args)

    def same(self, left, right):
        self.assertTrue(loops._same_tree(left, right))

    def test_completed_live_and_restored_loops_issue_immutable_detached_tokens(self):
        for architecture in self.parents:
            parent, loop = self.parents[architecture], self.completed[architecture]
            self.assertEqual(parent.identity["base_updates"], 66)
            self.assertEqual(parent.training_transcripts, self.transcripts)
            self.assertEqual(set(parent.protected_transcripts), set(self.transcripts) | set(self.protection))
            self.assertEqual(parent.evaluation_specs, loop.provider.specs())
            changed = parent.identity; changed["base_updates"] = 0
            changed = parent.protected_transcripts; changed.clear()
            changed = parent.evaluation_specs; changed.clear()
            self.assertEqual(parent.identity["base_updates"], 66)
            self.assertTrue(parent.protected_transcripts)
            self.assertTrue(parent.evaluation_specs)
            with self.assertRaises((AttributeError, TypeError)): parent._identity = b"{}"
            with self.assertRaisesRegex(TypeError, "process-owned"): pickle.dumps(parent)
            restored = FoundationLoop(self.provider(architecture), **loop.options, payload=loop.snapshot())
            again = module.CompletedParent.from_loop(restored, training_transcripts=self.transcripts)
            self.assertEqual(parent.identity, again.identity)
            self.same(parent._learner(), again._learner())

    def test_parent_requires_real_complete_loop_and_exact_authenticated_transcripts(self):
        with self.assertRaises(ValueError): module.capture_completed_parent({}, training_transcripts=self.transcripts)
        incomplete = FoundationLoop(self.provider("flat"))
        with self.assertRaisesRegex(ValueError, "complete final evaluation"):
            module.capture_completed_parent(incomplete, training_transcripts=self.transcripts)
        for values in (self.transcripts[:-1], list(reversed(self.transcripts)), self.transcripts+[self.transcripts[-1]],
                       sorted(self.transcripts[:-1]+["0"*64]), tuple(self.transcripts)):
            with self.assertRaises(ValueError):
                module.capture_completed_parent(self.completed["flat"], training_transcripts=values)
        loop = self.completed["flat"]
        before = copy.deepcopy(loop._state)
        try:
            # Loop's general historical validator does not bind a complete
            # history's last producing hash to the current tensor image.
            loop._state["history"][-1]["producer"]["weights_sha256"] = "0"*64
            with self.assertRaises(ValueError):
                module.capture_completed_parent(loop, training_transcripts=self.transcripts)
        finally: loop._state = before

    def test_durable_parent_reconstruction_preserves_identity_and_child_restart_contract(self):
        for architecture in self.parents:
            # Save a separately restored real completed loop so other fixtures
            # retain their intentionally in-memory provenance.
            original = self.completed[architecture]
            parent_loop = FoundationLoop(self.provider(architecture), **original.options, payload=original.snapshot())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"completed-parent.pt"
                parent_loop.save(path)
                token = module.capture_completed_parent(parent_loop, training_transcripts=self.transcripts)
                self.assertEqual(token.identity["parent_envelope"]["kind"], "durable")
                self.assertIsNotNone(token.identity["parent_envelope"]["file_sha256"])
                reloaded = FoundationLoop.load(path, self.provider(architecture))
                reconstructed = module.capture_completed_parent(reloaded, training_transcripts=self.transcripts)
                self.assertEqual(token.identity, reconstructed.identity)
                self.assertEqual(token.retention_references, reconstructed.retention_references)
                self.assertEqual(token.reference_anchor, reconstructed.reference_anchor)
                child = self.trainer(architecture, parent=token)
                restored = self.trainer(architecture, parent=reconstructed, payload=child.snapshot())
                self.same(child.snapshot(), restored.snapshot())

    def test_handoff_preserves_every_tensor_and_moment_while_resetting_only_local_work(self):
        for architecture, parent in self.parents.items():
            trainer = self.trainer(architecture)
            old, new = parent._learner(), trainer.snapshot()
            self.same(old["weights"], new["weights"])
            self.same(old["optimizer"], new["optimizer"])
            self.assertEqual((trainer.cursor, trainer.updates, trainer.base_updates, trainer.lifetime_updates), (0, 0, 66, 66))
            self.assertEqual(new["evidence"], self.next_index.replay("curriculum", 0))
            self.assertTrue(all(value == 0 for value in new["timing"].values()))
            self.assertNotEqual(new["recipe"]["plan_sha256"], old["recipe"]["plan_sha256"])
            self.assertEqual(new["recipe"]["parent_evaluation_specs"], parent.evaluation_specs)

    def test_first_update_matches_unchanged_step_with_explicit_full_state_reference(self):
        for architecture, parent in self.parents.items():
            child = self.trainer(architecture)
            direct = VariantFoundationTrainer(self.next_plan, architecture=architecture, seed=8541,
                config=self.config, admission_protected_transcripts=self.exclusions, protected_transcripts=self.exclusions,
                admission_receipt=self.next_receipt, plan_index=self.next_index)
            state = parent._learner()
            # Test-only arithmetic control deliberately installs the same full
            # learner state; it is never presented as an admissible checkpoint.
            direct.model.load_state_dict(state["weights"])
            direct.optimizer.load_state_dict(state["optimizer"])
            expected, actual = direct.step(), child.step()
            for key in ("bundle_id", "microbatches", "loss", "action_loss", "reply_loss", "observation_language_loss"):
                self.same(expected[key], actual[key])
            self.same(direct.snapshot()["weights"], child.snapshot()["weights"])
            self.same(direct.snapshot()["optimizer"], child.snapshot()["optimizer"])
            self.assertEqual(child.lifetime_updates, 67)

    def test_disk_restart_full_adamw_state_and_next_update_are_exact_for_both_architectures(self):
        for architecture in self.parents:
            first = self.trainer(architecture); first.step()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"cycle.pt"
                torch.save(first.snapshot(), path)
                second = self.trainer(architecture, payload=torch.load(path, map_location="cpu", weights_only=True))
            self.same(first.snapshot(), second.snapshot())
            first.step(); second.step()
            self.same(learning(first.snapshot()), learning(second.snapshot()))
            self.assertEqual(second.lifetime_updates, 68)

    def test_wrong_contract_or_missing_parent_exclusions_reject_before_model_construction(self):
        with patch.object(module.CycleFoundationTrainer, "_build", side_effect=AssertionError("early model")) as creation:
            cases = (dict(parent=self.parents["flat"].identity), dict(architecture="hierarchical"), dict(seed=3),
                dict(seed=True), dict(config=SequenceConfig()), dict(learning_rate=.002),
                dict(admission_protected_transcripts=self.protection), dict(protected_transcripts=self.protection),
                dict(admission_receipt={}), dict(plan_index=self.index))
            for options in cases:
                with self.subTest(options=list(options)), self.assertRaises(ValueError): self.trainer(**options)
            with self.assertRaisesRegex(ValueError, "fresh"):
                module.CycleFoundationTrainer(self.plan, parent=self.parents["flat"],
                    admission_protected_transcripts=self.exclusions, protected_transcripts=self.exclusions,
                    admission_receipt=self.receipt)
            creation.assert_not_called()

    def test_hostile_checkpoint_counters_moments_schema_parent_and_protection_are_transactional(self):
        trainer = self.trainer(); trainer.step()
        before = trainer.snapshot(); model, optimizer = trainer.model, trainer.optimizer
        for kind in ("schema", "v1_schema", "cycle", "base", "lifetime", "bool", "cursor", "parent", "protection", "source", "evidence",
                     "step", "moment", "dtype", "coverage", "alias", "timing", "group", "nan"):
            changed = copy.deepcopy(before)
            if kind == "schema": changed["schema"] = "bic-foundation-variant-trainer-v1"
            elif kind == "v1_schema": changed["schema"] = "bic-foundation-cycle-trainer-v1"
            elif kind == "cycle": changed["cycle"] += 1
            elif kind == "base": changed["base_updates"] += 1
            elif kind == "lifetime": changed["lifetime_updates"] += 1
            elif kind == "bool": changed["cursor"] = True
            elif kind == "cursor": changed["cursor"] += 1
            elif kind == "parent": changed["recipe"]["parent_identity"]["weights_sha256"] = "0"*64
            elif kind == "protection": changed["recipe"]["protected_transcripts_count"] -= 1
            elif kind == "source": changed["recipe"]["source_sha256"] = {}
            elif kind == "evidence": changed["evidence"]["exposures"]["color"]["episodes"] += 1
            elif kind == "step": changed["optimizer"]["state"][0]["step"] -= 1
            elif kind == "moment": changed["optimizer"]["state"][0]["exp_avg_sq"].flatten()[0] = -1
            elif kind == "dtype": changed["optimizer"]["state"][0]["exp_avg"] = changed["optimizer"]["state"][0]["exp_avg"].double()
            elif kind == "coverage": changed["optimizer"]["state"] = {}
            elif kind == "alias": changed["weights"]["observation_head.weight"][0, 0] += 1
            elif kind == "timing": changed["timing"]["retained_step_seconds"] = -1
            elif kind == "group": changed["optimizer"]["param_groups"][0]["lr"] = .01
            elif kind == "nan": changed["weights"]["action_head.bias"][0] = float("nan")
            with self.subTest(kind=kind), self.assertRaises(ValueError): trainer.restore(changed)
            self.assertIs(trainer.model, model); self.assertIs(trainer.optimizer, optimizer)
            self.same(before, trainer.snapshot())
        with self.assertRaises(ValueError): trainer.restore(self.trainer("hierarchical").snapshot())
        self.same(before, trainer.snapshot())

    def test_zero_local_cursor_requires_exact_parent_weights_and_moments(self):
        trainer = self.trainer(); before = trainer.snapshot()
        for kind in ("weight", "first_moment", "second_moment"):
            changed = copy.deepcopy(before)
            if kind == "weight": changed["weights"]["action_head.bias"][0] += .5
            else: changed["optimizer"]["state"][0]["exp_avg" if kind == "first_moment" else "exp_avg_sq"].flatten()[0] += .5
            with self.assertRaisesRegex(ValueError, "authenticated parent"): trainer.restore(changed)
            self.same(before, trainer.snapshot())

    def test_partial_microbatch_failure_poison_requires_explicit_full_state_restore(self):
        trainer, control = self.trainer("hierarchical"), self.trainer("hierarchical")
        before = trainer.snapshot(); forward = trainer.model.forward; calls = 0
        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2: raise RuntimeError("second microbatch failure")
            return forward(*args, **kwargs)
        with patch.object(trainer.model, "forward", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "second microbatch"): trainer.step()
        self.assertEqual(trainer.last_report["physical_optimizer_updates"], 0)
        self.assertEqual(trainer.last_report["completed_microbatch_episode_exposures"], 2)
        with self.assertRaises(RuntimeError): trainer.snapshot()
        trainer.restore(before); self.same(before, trainer.snapshot())
        trainer.step(); control.step()
        self.same(learning(trainer.snapshot()), learning(control.snapshot()))

    def test_source_runtime_drift_and_caller_mutation_cannot_commit_partial_restore(self):
        trainer = self.trainer(); before = trainer.snapshot()
        with patch.object(module, "source_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "source identity"): trainer.restore(before)
            with self.assertRaisesRegex(RuntimeError, "source identity"): self.parents["flat"].identity
        with patch.object(module.providers, "runtime_identity", return_value={}):
            with self.assertRaisesRegex(ValueError, "runtime identity"): trainer.restore(before)
        self.same(before, trainer.snapshot())
        caller = copy.deepcopy(before); original = trainer._build
        def mutate_caller():
            caller["optimizer"]["state"][0]["exp_avg"].flatten()[0] += 1
            return original()
        with patch.object(trainer, "_build", side_effect=mutate_caller): trainer.restore(caller)
        self.same(before, trainer.snapshot())
        original_check = trainer._assert_sources; calls = 0
        def drift():
            nonlocal calls
            calls += 1
            if calls > 1: raise ValueError("late drift")
            original_check()
        model, optimizer = trainer.model, trainer.optimizer
        with patch.object(trainer, "_assert_sources", side_effect=drift):
            with self.assertRaisesRegex(ValueError, "late drift"): trainer.restore(before)
        self.assertIs(trainer.model, model); self.assertIs(trainer.optimizer, optimizer)
        self.same(before, trainer.snapshot())


if __name__ == "__main__": unittest.main(verbosity=2)
