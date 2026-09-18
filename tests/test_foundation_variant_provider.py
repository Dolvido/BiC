"""Tiny CPU adapter/unchanged-loop checks, never formal data or learned weights."""
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.language import InferiorFrontal
from experiments import foundation_curriculum as curriculum
from experiments import foundation_loop as loop_module
from experiments import foundation_plan as planning
from experiments import foundation_provider as old_provider
from experiments import foundation_variant_provider as module
from experiments.foundation_admission import repair_plan
from experiments.foundation_evaluation import FoundationBank
from experiments.foundation_loop import FoundationLoop
from experiments.foundation_plan_index import AuthenticatedPlanIndex
from experiments.foundation_variant_training import VariantFoundationTrainer
from experiments.hierarchical_sequence_student import HierarchicalSequenceStudent
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig, SequenceStudent


def learning(payload):
    result = copy.deepcopy(payload)
    result.pop("timing")
    return result


class VariantProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.config = SequenceConfig(width=8, layers=4, heads=2, feedforward=16, max_turns=12)
        base = planning.build_plan(seed=851100001, stage_updates=10, final_updates=6,
            micro_batch_size=2, rehearsal_every=2, ordering_seed=851100002)
        cls.history = sorted(map(transcript_digest, planning.materialize_pair(base, 0, "color", 0)))
        cls.plan, cls.admission = repair_plan(base, cls.history)
        cls.banks = {role: {f"probe/{family}/d0/direct/t{turns}": curriculum.generate_pair(
            family, 851200001+role_index*100+index, depth=0, turns=turns, split="dev")
            for index, (family, turns) in enumerate(zip(curriculum.FAMILIES, (8, 10, 12)))}
            for role_index, role in enumerate(("development", "retention"))}
        cls.protection = sorted(set(cls.history) | {transcript_digest(row)
            for named in cls.banks.values() for rows in named.values() for row in rows})
        cls.index = AuthenticatedPlanIndex(cls.plan, admission_receipt=cls.admission,
            admission_protected_transcripts=cls.history, protected_transcripts=cls.protection)
        cls.work = dict.fromkeys(("completed_optimizer_calls", "trainer_step_attempts", "unknown_optimizer_reports",
            "drawn_episode_exposures", "neural_attempted_episode_exposures", "completed_microbatch_episode_exposures",
            "training_encoder_forward_attempts", "training_encoder_forward_completed", "training_encoder_episode_instances",
            "evaluation_encoder_forward_attempts", "evaluation_encoder_forward_completed", "evaluation_encoder_episode_instances",
            "decoder_forward_attempts", "decoder_forward_completed", "decoder_sequence_instances",
            "evaluation_bank_attempts", "evaluation_banks_completed", "evaluation_episode_attempts", "evaluation_episodes_completed"), 0)
        cls.stack = ExitStack()
        actual_step = VariantFoundationTrainer.step
        def step(trainer):
            cls.work["trainer_step_attempts"] += 1
            try:
                return actual_step(trainer)
            finally:
                report = trainer.last_report
                if report is not None:
                    for key in ("drawn_episode_exposures", "neural_attempted_episode_exposures", "completed_microbatch_episode_exposures"):
                        cls.work[key] += report[key] or 0
                    cls.work["unknown_optimizer_reports"] += int(report["physical_optimizer_updates"] is None)
        cls.stack.enter_context(patch.object(VariantFoundationTrainer, "step", step))
        actual_optimizer = torch.optim.AdamW.step
        def optimizer(optimizer, *args, **kwargs):
            result = actual_optimizer(optimizer, *args, **kwargs)
            cls.work["completed_optimizer_calls"] += 1
            return result
        cls.stack.enter_context(patch.object(torch.optim.AdamW, "step", optimizer))
        for model_type in (SequenceStudent, HierarchicalSequenceStudent):
            actual_forward = model_type.forward
            def forward(model, token_ids, *args, _actual=actual_forward, **kwargs):
                role = "training" if torch.is_grad_enabled() else "evaluation"
                cls.work[role+"_encoder_forward_attempts"] += 1
                cls.work[role+"_encoder_episode_instances"] += len(token_ids)
                result = _actual(model, token_ids, *args, **kwargs)
                cls.work[role+"_encoder_forward_completed"] += 1
                return result
            cls.stack.enter_context(patch.object(model_type, "forward", forward))
        actual_decoder = InferiorFrontal.forward
        def decoder(model, ids, *args, **kwargs):
            cls.work["decoder_forward_attempts"] += 1
            cls.work["decoder_sequence_instances"] += len(ids)
            result = actual_decoder(model, ids, *args, **kwargs)
            cls.work["decoder_forward_completed"] += 1
            return result
        cls.stack.enter_context(patch.object(InferiorFrontal, "forward", decoder))
        actual_score = FoundationBank.score
        def score(bank, *args, **kwargs):
            cls.work["evaluation_bank_attempts"] += 1
            cls.work["evaluation_episode_attempts"] += bank.identity["episodes"]
            result = actual_score(bank, *args, **kwargs)
            cls.work["evaluation_banks_completed"] += 1
            cls.work["evaluation_episodes_completed"] += bank.identity["episodes"]
            return result
        cls.stack.enter_context(patch.object(FoundationBank, "score", score))

    @classmethod
    def tearDownClass(cls):
        cls.stack.close(); torch.set_num_threads(cls.threads)
        result = dict(**cls.work, device="cpu", formal_data_or_weights_used=False,
            scope="Actual completed optimizer calls and attempted/completed encoder, decoder and bank calls, including deliberately discarded work. Decoder sequences are turn instances, not extra training episodes. No GPU.")
        print("VARIANT_PROVIDER_PHYSICAL_WORK="+json.dumps(result, sort_keys=True))
        destination = os.environ.get("BIC_VARIANT_PROVIDER_ACCOUNTING")
        if destination:
            Path(destination).write_text(json.dumps(result, indent=2), encoding="utf8")

    def make(self, architecture="flat", **changes):
        options = dict(architecture=architecture, seed=8513, evaluation_banks=self.banks,
            admission_protected_transcripts=self.history, protected_transcripts=self.protection,
            admission_receipt=self.admission, plan_index=self.index, config=self.config)
        options.update(changes)
        return module.VariantFoundationPracticeProvider(self.plan, **options)

    def request(self, provider, start, stop):
        return dict(kind="prescribed_prefix", provider_sha256=provider.identity_sha256,
            plan_sha256=provider.identity["plan_sha256"], order="curriculum", start_cursor=start, stop_cursor=stop)

    def same(self, left, right):
        self.assertTrue(loop_module._same_tree(left, right))

    def test_both_architectures_match_direct_variant_optimizer_and_lesson_consumption(self):
        for architecture in ("flat", "hierarchical"):
            provider = self.make(architecture)
            direct = VariantFoundationTrainer(self.plan, architecture=architecture, seed=8513, config=self.config,
                admission_protected_transcripts=self.history, protected_transcripts=self.protection,
                admission_receipt=self.admission, plan_index=self.index)
            expected = [direct.step() for _ in range(2)]
            report = provider.practice(self.request(provider, 0, 2))
            self.assertEqual(report["completed_updates"], 2)
            for actual, reference in zip(report["step_reports"], expected):
                for key in ("loss", "action_loss", "reply_loss", "microbatches", "bundle_id"):
                    self.assertEqual(actual[key], reference[key])
            self.same(learning(provider.snapshot()["learner"]), learning(direct.snapshot()))
            self.assertIs(provider._trainer._index, self.index)
            self.assertTrue(provider.setup_report["index_reused"])
            self.assertEqual(provider.identity["architecture"]["name"], architecture)

    def test_both_architectures_restore_pending_prefix_without_rebuilding_index(self):
        for architecture in ("flat", "hierarchical"):
            full, part = self.make(architecture), self.make(architecture)
            full.practice(self.request(full, 0, 3))
            part.practice(self.request(part, 0, 3), max_updates=1)
            saved = part.snapshot()
            with patch.object(module, "AuthenticatedPlanIndex", wraps=AuthenticatedPlanIndex) as construction:
                # Construction is not called during provider.restore; it shares
                # the actual immutable object, not a persisted metadata cache.
                part.restore(saved)
                construction.assert_not_called()
            self.same(saved, part.snapshot())
            self.assertIs(part._trainer._index, self.index)
            self.assertEqual(part.snapshot()["pending"]["consumed_updates"], 1)
            part.practice(max_updates=2)
            self.same(learning(full.snapshot()["learner"]), learning(part.snapshot()["learner"]))
            self.assertIsNone(part.snapshot()["pending"])

    def test_teacher_free_evidence_bound_to_architecture_and_producing_weights(self):
        for architecture in ("flat", "hierarchical"):
            provider = self.make(architecture)
            before = provider.snapshot()
            name = next(iter(self.banks["development"]))
            original = provider._trainer.model.forward
            def bos_only(*args, **kwargs):
                decoder = args[2] if len(args) > 2 else kwargs["decoder_input_ids"]
                self.assertEqual(decoder.shape[-1], 1)
                self.assertTrue(torch.all(decoder == 1))
                return original(*args, **kwargs)
            with patch.object(provider._trainer.model, "forward", side_effect=bos_only):
                response = provider.evaluate("development", names=[name])
            self.same(before, provider.snapshot())
            self.assertEqual(response["schema"], old_provider.EVALUATION_SCHEMA)
            self.assertTrue(response["per_bank"][name]["free_running_replies"])
            self.assertFalse(response["per_bank"][name]["teacher_used_for_policy"])
            provider.validate_evaluation(response)
            with self.assertRaisesRegex(ValueError, "another learner state"):
                provider.validate_evaluation(response, expected_weights_sha256="f"*64, expected_updates=0)
            changed = copy.deepcopy(response); changed["provider_sha256"] = self.make(
                "hierarchical" if architecture == "flat" else "flat").identity_sha256
            with self.assertRaises(ValueError): provider.validate_evaluation(changed)

    def test_unchanged_loop_partial_evaluation_disk_load_and_pending_optimizer_resume(self):
        for architecture in ("flat", "hierarchical"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"loop.pt"
                first = FoundationLoop(self.make(architecture), window_updates=2, chunk_updates=1, history_limit=1)
                first.run(max_updates=0, max_evaluations=1)
                first.save(path)
                saved = first.snapshot()
                self.assertEqual(saved["state"]["evaluation"]["cursor"], 1)
                second = FoundationLoop.load(path, self.make(architecture))
                self.same(saved, second.snapshot())
                second.run(max_updates=1)
                saved = second.snapshot()
                self.assertEqual(saved["provider"]["pending"]["consumed_updates"], 1)
                third = FoundationLoop.load(path, self.make(architecture))
                self.same(saved, third.snapshot())
                third.run(max_updates=1)
                final = third.snapshot()
                self.assertEqual(final["provider"]["learner"]["cursor"], 2)
                self.assertEqual(final["state"]["evaluations_completed"], 2)
                self.assertEqual(final["state"]["work"]["physical_optimizer_updates"], 2)
                self.assertIsNone(final["provider"]["pending"])
                self.assertEqual(final["provider"]["schema"], module.SCHEMA)
                self.assertEqual(final["schema"], loop_module.SCHEMA)

    def test_unchanged_loop_failed_and_published_commits_keep_atomic_state_and_true_work(self):
        for architecture in ("flat", "hierarchical"):
            loop = FoundationLoop(self.make(architecture), window_updates=2, chunk_updates=1)
            loop.run(max_updates=0)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"loop.pt"; loop.save(path)
                before = loop.snapshot()
                with patch.object(loop, "_write", side_effect=OSError("before publication")):
                    with self.assertRaisesRegex(OSError, "before publication"):
                        loop.run(max_updates=1)
                self.same(before, loop.snapshot())
                self.assertEqual(loop.last_report["physical_optimizer_updates"], 1)
                self.assertEqual(loop.last_report["discarded_optimizer_updates"], 1)
                self.assertEqual(loop.last_report["durable_retained_updates"], 0)
                replace = loop_module.os.replace
                def after_replace(source, target):
                    replace(source, target)
                    raise OSError("after publication")
                with patch.object(loop_module.os, "replace", side_effect=after_replace):
                    with self.assertRaisesRegex(OSError, "after publication"):
                        loop.run(max_updates=1)
                self.assertEqual(loop.last_report["retained_updates"], 1)
                self.assertEqual(loop.last_report["discarded_optimizer_updates"], 0)
                loaded = FoundationLoop.load(path, self.make(architecture))
                self.same(loop.snapshot(), loaded.snapshot())

    def test_failed_partial_microbatch_poison_and_explicit_restore_preserve_pending(self):
        for architecture in ("flat", "hierarchical"):
            provider = self.make(architecture)
            before = provider.snapshot(); original = provider._trainer.model.forward
            calls = 0
            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                value = original(*args, **kwargs)
                if calls == 2: raise RuntimeError("after second encoder forward")
                return value
            with patch.object(provider._trainer.model, "forward", side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, "second encoder"):
                    provider.practice(self.request(provider, 0, 2))
            self.assertEqual(provider.last_report["physical_optimizer_updates"], 0)
            self.assertEqual(provider.last_report["neural_attempted_episode_exposures"], 4)
            with self.assertRaisesRegex(RuntimeError, "explicit transactional"):
                provider.snapshot()
            provider.restore(before)
            self.same(before, provider.snapshot())

    def test_wrong_index_protection_and_roles_reject_before_model_creation(self):
        cases = [dict(architecture="other"), dict(plan_index=self.index.identity),
                 dict(protected_transcripts=self.protection[:-1]),
                 dict(admission_protected_transcripts=[]), dict(admission_receipt={}),
                 dict(evaluation_banks={"audit": self.banks["development"], "retention": self.banks["retention"]})]
        missing = copy.deepcopy(self.banks); missing["development"].pop(next(iter(missing["development"])))
        cases.append(dict(evaluation_banks=missing))
        corrupted = copy.deepcopy(self.banks)
        next(iter(corrupted["retention"].values()))[0]["turns"][-1]["target"] = 3
        cases.append(dict(evaluation_banks=corrupted))
        with patch.object(module.VariantFoundationPracticeProvider, "_new_trainer", side_effect=AssertionError("model constructed too early")) as creation:
            for options in cases:
                with self.subTest(options=list(options)), self.assertRaises((ValueError, TypeError)):
                    self.make(**options)
            creation.assert_not_called()

    def test_old_cross_architecture_and_corrupt_checkpoints_reject_without_mutation(self):
        flat, hierarchy = self.make("flat"), self.make("hierarchical")
        before = flat.snapshot()
        cases = [hierarchy.snapshot()]
        changed = copy.deepcopy(before); changed["schema"] = old_provider.SCHEMA; cases.append(changed)
        changed = copy.deepcopy(before); changed["learner"]["schema"] = "bic-foundation-trainer-v1"; cases.append(changed)
        changed = copy.deepcopy(before); changed["learner"]["cursor"] = 1; cases.append(changed)
        changed = copy.deepcopy(before); changed["learner"]["optimizer"]["param_groups"][0]["lr"] = .3; cases.append(changed)
        for changed in cases:
            with self.assertRaises(ValueError): flat.restore(changed)
            self.same(before, flat.snapshot())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"loop.pt"
            loop = FoundationLoop(flat); loop.save(path)
            with self.assertRaisesRegex(ValueError, "identity/options"):
                FoundationLoop.load(path, self.make("hierarchical"))

    def test_source_runtime_and_returned_identity_are_not_mutable_aliases(self):
        provider = self.make()
        before = provider.snapshot()
        public = provider.identity
        public["architecture"]["name"] = "hierarchical"
        self.assertEqual(provider.identity["architecture"]["name"], "flat")
        changed = {**module.source_hashes(), "experiments/foundation_variant_provider.py": "0"*64}
        with patch.object(module, "source_hashes", return_value=changed):
            with self.assertRaisesRegex(RuntimeError, "source or runtime"):
                provider.snapshot()
        with patch.object(module.base, "runtime_identity", return_value={"device": "other"}):
            with self.assertRaisesRegex(RuntimeError, "source or runtime"):
                provider.restore(before)
        self.same(before, provider.snapshot())

    def test_restore_detaches_caller_and_rechecks_drift_before_atomic_swap(self):
        provider = self.make()
        before = provider.snapshot()
        caller = copy.deepcopy(before)
        original = provider._new_trainer
        def mutate_caller(payload):
            caller["pending"] = {"invalid": "changed while candidate was constructed"}
            return original(payload)
        with patch.object(provider, "_new_trainer", side_effect=mutate_caller):
            provider.restore(caller)
        self.same(before, provider.snapshot())
        for kind in ("source", "runtime"):
            original_check = provider._check
            count = 0
            def drift_check(*args, **kwargs):
                nonlocal count
                count += 1
                if count > 1:
                    raise RuntimeError(kind+" drift during candidate construction")
                return original_check(*args, **kwargs)
            with patch.object(provider, "_check", side_effect=drift_check), \
                 self.assertRaisesRegex(RuntimeError, "drift during candidate"):
                provider.restore(before)
            self.same(before, provider.snapshot())


if __name__ == "__main__":
    unittest.main()
