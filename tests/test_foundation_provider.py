"""Tiny CPU provider integration; no pilot inputs, GPU or capability claims."""
import copy
from contextlib import ExitStack
import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch

from experiments import foundation_curriculum as curriculum
from experiments import foundation_plan as planning
from experiments import foundation_provider as provider_module
from experiments.foundation_admission import repair_plan
from experiments.foundation_provider import FoundationPracticeProvider
from experiments.foundation_training import FoundationTrainer
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig


def same_tree(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertIsInstance(right, torch.Tensor)
        test.assertEqual(left.dtype, right.dtype)
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left: same_tree(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(type(left), type(right))
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right): same_tree(test, a, b)
    else:
        test.assertEqual(left, right)


def learning_only(payload):
    result = copy.deepcopy(payload)
    result.pop("timing")
    return result


class FoundationProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config = SequenceConfig(width=8, layers=1, heads=2, feedforward=16,
                                    max_turns=12, max_positions=1560)
        base = planning.build_plan(seed=801000001, stage_updates=10, final_updates=6,
            micro_batch_size=2, rehearsal_every=2, ordering_seed=801000002)
        cls.history = sorted(map(transcript_digest, planning.materialize_pair(base, 0, "color", 0)))
        cls.plan, cls.receipt = repair_plan(base, cls.history)
        cls.banks = {}
        for role_index, role in enumerate(("development", "retention")):
            cls.banks[role] = {f"probe/{family}/d0/direct/t8": curriculum.generate_pair(
                family, 802000001+role_index*100+index, depth=0, turns=8, split="dev")
                for index, family in enumerate(curriculum.FAMILIES)}
        cls.protection = sorted(set(cls.history) | {transcript_digest(row)
            for banks in cls.banks.values() for rows in banks.values() for row in rows})
        cls.physical_updates = 0
        cls.completed_microbatch_episodes = 0
        cls.neural_attempted_episodes = 0
        cls.drawn_episodes = 0
        cls.unknown_optimizer_reports = 0
        actual_optimizer = torch.optim.AdamW.step
        actual_trainer = FoundationTrainer.step
        def optimizer_step(optimizer, *args, **kwargs):
            result = actual_optimizer(optimizer, *args, **kwargs)
            cls.physical_updates += 1
            return result
        def trainer_step(trainer):
            try:
                return actual_trainer(trainer)
            finally:
                report = trainer.last_report
                if report is not None:
                    cls.completed_microbatch_episodes += report["completed_microbatch_episode_exposures"] or 0
                    cls.neural_attempted_episodes += report["neural_attempted_episode_exposures"] or 0
                    cls.drawn_episodes += report["drawn_episode_exposures"] or 0
                    cls.unknown_optimizer_reports += int(report["physical_optimizer_updates"] is None)
        cls.patches = ExitStack()
        cls.patches.enter_context(patch.object(torch.optim.AdamW, "step", optimizer_step))
        cls.patches.enter_context(patch.object(FoundationTrainer, "step", trainer_step))

    @classmethod
    def tearDownClass(cls):
        cls.patches.close()
        torch.set_num_threads(cls.threads)
        print(f"FoundationProvider CPU test accounting: {cls.physical_updates} completed optimizer calls; "
              f"{cls.drawn_episodes} returned candidate episodes; {cls.neural_attempted_episodes} neural-attempted; "
              f"{cls.completed_microbatch_episodes} completed-microbatch episodes; "
              f"{cls.unknown_optimizer_reports} deliberately uncertain trainer reports. No GPU.")

    def make(self, **options):
        defaults = dict(seed=8101, evaluation_banks=self.banks,
            admission_protected_transcripts=self.history, protected_transcripts=self.protection,
            admission_receipt=self.receipt, config=self.config)
        defaults.update(options)
        return FoundationPracticeProvider(self.plan, **defaults)

    def request(self, provider, start, stop):
        return {"kind": "prescribed_prefix", "provider_sha256": provider.identity_sha256,
            "plan_sha256": provider.identity["plan_sha256"], "order": "curriculum",
            "start_cursor": start, "stop_cursor": stop}

    def test_direct_trainer_equivalence_includes_full_optimizer_and_consumption(self):
        provider = self.make()
        direct = FoundationTrainer(self.plan, seed=8101, config=self.config,
                                   protected_transcripts=self.protection)
        expected = [direct.step() for _ in range(3)]
        actual = provider.practice(self.request(provider, 0, 3))
        self.assertEqual(actual["status"], "prefix_complete")
        self.assertEqual(actual["completed_updates"], 3)
        self.assertEqual(actual["physical_optimizer_updates"], 3)
        self.assertEqual(actual["in_memory_retained_updates"], 3)
        self.assertIsNone(actual["durable_retained_updates"])
        self.assertEqual(actual["neural_attempted_episode_exposures"], 18)
        self.assertGreater(actual["wall_seconds"], 0)
        self.assertGreaterEqual(actual["wall_seconds"], actual["step_seconds"])
        for first, second in zip(expected, actual["step_reports"]):
            for key in ("bundle_id", "microbatches", "loss", "action_loss", "reply_loss"):
                self.assertEqual(first[key], second[key])
        same_tree(self, learning_only(direct.snapshot()), learning_only(provider.snapshot()["learner"]))
        self.assertEqual(provider.status()["family_microbatches"], dict.fromkeys(curriculum.FAMILIES, 3))

    def test_pending_prefix_resume_has_identical_next_updates(self):
        uninterrupted, split = self.make(), self.make()
        uninterrupted.practice(self.request(uninterrupted, 0, 3))
        partial = split.practice(self.request(split, 0, 3), max_updates=1)
        self.assertEqual(partial["status"], "update_bound")
        saved = split.snapshot()
        self.assertEqual(saved["pending"]["consumed_updates"], 1)
        resumed = self.make(payload=saved)
        same_tree(self, resumed.snapshot(), saved)
        resumed.practice(max_updates=2)
        self.assertIsNone(resumed.status()["pending"])
        same_tree(self, learning_only(uninterrupted.snapshot()["learner"]),
                  learning_only(resumed.snapshot()["learner"]))

    def test_evaluation_teacher_free_immutable_and_real_curriculum_version(self):
        provider = self.make()
        before = provider.snapshot()
        name = next(iter(self.banks["development"]))
        with ExitStack() as stack:
            for attr in ("generate_pair", "validate_pair", "validate_row", "query_ancestries"):
                stack.enter_context(patch.object(curriculum, attr, side_effect=AssertionError("teacher called")))
            for attr in ("parse_sentence", "english_oracle", "abstract_oracle"):
                stack.enter_context(patch.object(curriculum.legacy, attr, side_effect=AssertionError("oracle called")))
            for control in ("normal", "blank", "reset"):
                result = provider.evaluate("development", names=[name], control=control, batch_size=2)
                self.assertTrue(provider.validate_evaluation(result))
                row = result["per_bank"][name]
                self.assertEqual(row["bank"]["version"], curriculum.VERSION)
                self.assertEqual(row["decoder_prefix"], "BOS only")
                self.assertIs(row["teacher_used_for_policy"], False)
        same_tree(self, before, provider.snapshot())
        malformed = copy.deepcopy(result)
        malformed["per_bank"][name]["bank"]["version"] = curriculum.legacy.VERSION
        with self.assertRaises(ValueError): provider.validate_evaluation(malformed)
        malformed = copy.deepcopy(result)
        malformed["per_bank"][name]["final_pairs"]["total"] += 1
        with self.assertRaises(ValueError): provider.validate_evaluation(malformed)

    def test_admission_history_and_full_trainer_protection_are_distinct(self):
        provider = self.make()
        self.assertEqual(provider.identity["admission_protection_count"], 2)
        self.assertEqual(provider.identity["trainer_protection_count"], 14)
        with self.assertRaises(ValueError): self.make(admission_protected_transcripts=self.protection)
        with self.assertRaises(ValueError): self.make(protected_transcripts=self.history)
        with self.assertRaises(ValueError): self.make(protected_transcripts=self.protection[2:])
        # Additional sealed protections need not belong to visible provider banks.
        extra = self.make(protected_transcripts=sorted([*self.protection, "f"*64]))
        self.assertNotEqual(extra.identity_sha256, provider.identity_sha256)
        with self.assertRaises(ValueError): extra.restore(provider.snapshot())

    def test_historical_evaluation_requires_explicit_authenticated_producing_state(self):
        provider = self.make()
        name = next(iter(self.banks["development"]))
        response = provider.evaluate("development", names=[name], batch_size=2)
        producing = provider.snapshot()
        provider.practice(self.request(provider, 0, 1))
        with self.assertRaisesRegex(ValueError, "another learner state"):
            provider.validate_evaluation(response)
        from brain_in_computer.dialogue_student import checkpoint_digest
        expected = checkpoint_digest(SimpleNamespace(state_dict=lambda: producing["learner"]["weights"]))
        self.assertTrue(provider.validate_evaluation(response, expected_weights_sha256=expected,
                                                     expected_updates=producing["learner"]["cursor"]))
        with self.assertRaises(ValueError): provider.validate_evaluation(response, expected_updates=0)
        with self.assertRaises(ValueError):
            provider.validate_evaluation(response, expected_weights_sha256="0"*64, expected_updates=0)

    def test_scorer_exception_checks_purity_before_allowing_further_practice(self):
        provider = self.make()
        saved = provider.snapshot()
        name = next(iter(self.banks["development"]))
        bank = provider._banks["development"][name]
        with patch.object(bank, "score", side_effect=RuntimeError("ordinary scorer failure")):
            with self.assertRaisesRegex(RuntimeError, "ordinary scorer"):
                provider.evaluate("development", names=[name])
        same_tree(self, saved, provider.snapshot())
        def mutate_then_fail(model, **kwargs):
            with torch.no_grad(): next(model.parameters()).add_(1.)
            raise RuntimeError("scorer failed after mutation")
        with patch.object(bank, "score", side_effect=mutate_then_fail):
            with self.assertRaisesRegex(RuntimeError, "after mutation"):
                provider.evaluate("development", names=[name])
        with self.assertRaisesRegex(RuntimeError, "restoration"): provider.snapshot()
        with self.assertRaisesRegex(RuntimeError, "restoration"): provider.practice()
        provider.restore(saved)
        same_tree(self, saved, provider.snapshot())

    def test_wrong_roles_audit_rows_versions_and_bank_names_fail_admission(self):
        for kind in ("role", "audit", "version", "name", "family_missing"):
            banks = copy.deepcopy(self.banks)
            name = next(iter(banks["development"]))
            if kind == "role": banks["audit"] = banks.pop("retention")
            elif kind == "audit":
                banks["development"][name] = curriculum.generate_pair("color", 803000001, depth=0, split="audit")
            elif kind == "version": banks["development"][name][0]["version"] = curriculum.legacy.VERSION
            elif kind == "name": banks["development"]["pretend/family"] = banks["development"].pop(name)
            else: banks["development"].pop(name)
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.make(evaluation_banks=banks)

    def test_later_reserved_canonical_dev_transcript_rejects_before_learner_creation(self):
        planned = planning.materialize_bundle(self.plan, 0)["color"][0]
        recipe = planned["recipe"]
        dev = curriculum.generate_pair("color", recipe["seed"], depth=recipe["depth"],
            turns=recipe["turns"], split="dev", naming_seed=recipe["naming_seed"],
            value_seed=recipe["value_seed"], structure_split=recipe["structure_split"])
        self.assertEqual(transcript_digest(planned), transcript_digest(dev[0]))
        banks = copy.deepcopy(self.banks)
        banks["development"]["probe/color/d0/direct/t8"] = dev
        full = sorted(set(self.protection) | set(map(transcript_digest, dev)))
        with patch.object(FoundationPracticeProvider, "_new_trainer", side_effect=AssertionError("learner created")):
            with self.assertRaisesRegex(ValueError, "protected"):
                self.make(evaluation_banks=banks, protected_transcripts=full)

    def test_request_rejection_and_expired_deadline_do_not_advance_learning(self):
        provider = self.make()
        before = provider.snapshot()
        for field, value in (("provider_sha256", "0"*64), ("plan_sha256", "0"*64),
                ("order", "mixed"), ("start_cursor", 1), ("start_cursor", True), ("stop_cursor", 999)):
            request = self.request(provider, 0, 2); request[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): provider.practice(request)
            same_tree(self, provider.snapshot(), before)
            self.assertEqual(provider.last_report["completed_updates"], 0)
        for limit in (None, True, 0, 4097):
            with self.assertRaises(ValueError): provider.practice(self.request(provider, 0, 2), max_updates=limit)
        expired = provider.practice(self.request(provider, 0, 2), deadline=0.)
        self.assertEqual(expired["status"], "deadline")
        self.assertEqual(expired["completed_updates"], 0)
        same_tree(self, before["learner"], provider.snapshot()["learner"])
        self.assertEqual(provider.status()["pending"]["consumed_updates"], 0)
        with self.assertRaises(ValueError): provider.practice(self.request(provider, 0, 1))

    def test_restore_rejects_tampered_requests_counters_and_optimizer_transactionally(self):
        provider = self.make()
        provider.practice(self.request(provider, 0, 3), max_updates=1)
        saved = provider.snapshot()
        for kind in ("pending_count", "pending_stop", "cursor", "exposures", "moment", "version"):
            changed = copy.deepcopy(saved)
            if kind == "pending_count": changed["pending"]["consumed_updates"] = True
            elif kind == "pending_stop": changed["pending"]["request"]["stop_cursor"] = 4
            elif kind == "cursor": changed["learner"]["cursor"] = 2
            elif kind == "exposures": changed["learner"]["evidence"]["exposures"]["color"]["episodes"] += 2
            elif kind == "moment": changed["learner"]["optimizer"]["state"][0]["exp_avg"].flatten()[0] = float("nan")
            else: changed["identity"]["curriculum_version"] = curriculum.legacy.VERSION
            with self.subTest(kind=kind), self.assertRaises(ValueError): provider.restore(changed)
            same_tree(self, provider.snapshot(), saved)

    def test_source_runtime_and_public_metadata_are_bound_and_isolated(self):
        provider = self.make()
        identity, specs = provider.identity, provider.specs()
        identity["curriculum_version"] = "caller mutation"
        specs["development"].clear()
        self.assertEqual(provider.identity["curriculum_version"], curriculum.VERSION)
        self.assertEqual(len(provider.specs()["development"]), 3)
        for name in ("source_hashes", "runtime_identity"):
            with patch.object(provider_module, name, return_value={}):
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    provider.practice(self.request(provider, 0, 1))
            self.assertEqual(provider.status()["updates"], 0)
            self.assertEqual(provider.last_report["completed_updates"], 0)

    def test_partial_microbatch_failure_poison_and_explicit_restore(self):
        provider = self.make()
        before = provider.snapshot()
        actual = provider._trainer.model.forward
        calls = 0
        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2: raise RuntimeError("second microbatch")
            return actual(*args, **kwargs)
        with patch.object(provider._trainer.model, "forward", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "second microbatch"):
                provider.practice(self.request(provider, 0, 2))
        report = provider.last_report
        self.assertEqual(report["physical_optimizer_updates"], 0)
        self.assertEqual(report["drawn_episode_exposures"], 6)
        self.assertEqual(report["neural_attempted_episode_exposures"], 4)
        self.assertEqual(report["completed_microbatch_episode_exposures"], 2)
        self.assertIsNone(report["in_memory_retained_updates"])
        for call in (provider.snapshot, provider.status, lambda: provider.practice()):
            with self.assertRaisesRegex(RuntimeError, "restoration"): call()
        provider.restore(before)
        same_tree(self, before, provider.snapshot())
        provider.practice(self.request(provider, 0, 1))

    def test_optimizer_throw_after_arithmetic_remains_unknown_until_restore(self):
        provider = self.make()
        before = provider.snapshot()
        actual = provider._trainer.optimizer.step
        def fail_after_step(*args, **kwargs):
            actual(*args, **kwargs)
            raise RuntimeError("post-optimizer failure")
        with patch.object(provider._trainer.optimizer, "step", side_effect=fail_after_step):
            with self.assertRaisesRegex(RuntimeError, "post-optimizer"):
                provider.practice(self.request(provider, 0, 1))
        self.assertIsNone(provider.last_report["physical_optimizer_updates"])
        self.assertEqual(provider.last_report["unknown_optimizer_attempts"], 1)
        self.assertEqual(provider.last_report["completed_updates"], 0)
        self.assertIsNone(provider.last_report["ending_updates"])
        with self.assertRaises(RuntimeError): provider.snapshot()
        provider.restore(before)
        same_tree(self, before, provider.snapshot())

    def test_report_failure_after_successful_optimizer_marks_accounting_unknown(self):
        provider = self.make()
        before = provider.snapshot()
        with patch.object(provider, "_account", side_effect=RuntimeError("account interrupted")):
            with self.assertRaisesRegex(RuntimeError, "account interrupted"):
                provider.practice(self.request(provider, 0, 1))
        self.assertTrue(provider.last_report["accounting_uncertain"])
        self.assertIsNone(provider.last_report["physical_optimizer_updates"])
        self.assertIsNone(provider.last_report["neural_attempted_episode_exposures"])
        with self.assertRaises(RuntimeError): provider.snapshot()
        provider.restore(before)
        same_tree(self, before, provider.snapshot())

    def test_exhaustion_control_flow_returns_normally_and_does_not_recycle(self):
        provider = self.make()
        # Control-flow fixture only: no claim of training the full 66-step plan.
        with patch.object(FoundationTrainer, "cursor", new_callable=PropertyMock,
                          return_value=len(self.plan["bundles"])):
            result = provider.practice()
            self.assertEqual(result["status"], "exhausted")
            self.assertEqual(result["completed_updates"], 0)
            self.assertEqual(result["in_memory_retained_updates"], 0)
            self.assertFalse(result["restore_required"])
        self.assertEqual(provider.status()["updates"], 0)


if __name__ == "__main__":
    unittest.main()
