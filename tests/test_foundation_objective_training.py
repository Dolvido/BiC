"""One bounded fresh CPU engineering fixture; no historical/formal inputs."""
import copy
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.language import ByteCodec
from experiments import foundation_training as legacy
from experiments import foundation_objective_training as objective
from experiments import foundation_plan_index as indexing
from experiments.foundation_admission import repair_plan
from experiments.foundation_plan import build_plan, materialize_pair
from experiments.foundation_plan_index import AuthenticatedPlanIndex
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig, SequenceStudent
from experiments.sequence_training import sequence_objective


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def arithmetic(payload):
    return {name: payload[name] for name in ("weights", "optimizer", "cursor", "evidence")}


class ObjectiveTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.started = time.monotonic()
        cls.config = SequenceConfig(width=8, layers=1, heads=2, feedforward=16, max_turns=12)
        base = build_plan(seed=831908001, stage_updates=10, final_updates=6, micro_batch_size=2,
                          rehearsal_every=2, ordering_seed=831908002)
        cls.history = sorted(transcript_digest(row) for row in materialize_pair(base, 0, "color", 0))
        cls.plan, cls.admission = repair_plan(base, cls.history)
        cls.protection = list(cls.history)
        cls.index = AuthenticatedPlanIndex(cls.plan, admission_protected_transcripts=cls.history,
            protected_transcripts=cls.protection, admission_receipt=cls.admission)
        cls.work = dict(model_constructions_attempted=0, model_constructions_completed=0,
            optimizer_attempts=0, optimizer_updates=0, physical_forward_calls=0,
            physical_forward_episodes=0, backward_attempts=0, backward_completed=0, step_reports=[])
        cls.stack = ExitStack()
        def tracked_build(function):
            def build(instance):
                cls.work["model_constructions_attempted"] += 1
                result = function(instance)
                cls.work["model_constructions_completed"] += 1
                return result
            return build
        for owner in (legacy.FoundationTrainer, objective.ObjectiveFoundationTrainer):
            cls.stack.enter_context(patch.object(owner, "_build", tracked_build(owner._build)))
        optimize = torch.optim.AdamW.step
        def tracked_optimizer(instance, *args, **kwargs):
            cls.work["optimizer_attempts"] += 1
            result = optimize(instance, *args, **kwargs)
            cls.work["optimizer_updates"] += 1
            return result
        cls.stack.enter_context(patch.object(torch.optim.AdamW, "step", tracked_optimizer))
        forward = SequenceStudent.forward
        def tracked_forward(instance, *args, **kwargs):
            cls.work["physical_forward_calls"] += 1
            cls.work["physical_forward_episodes"] += kwargs["token_ids"].shape[0]
            return forward(instance, *args, **kwargs)
        cls.stack.enter_context(patch.object(SequenceStudent, "forward", tracked_forward))
        backward = torch.Tensor.backward
        def tracked_backward(tensor, *args, **kwargs):
            cls.work["backward_attempts"] += 1
            result = backward(tensor, *args, **kwargs)
            cls.work["backward_completed"] += 1
            return result
        cls.stack.enter_context(patch.object(torch.Tensor, "backward", tracked_backward))
        def tracked_step(function):
            def step(instance):
                try:
                    return function(instance)
                finally:
                    if instance.last_report is not None:
                        cls.work["step_reports"].append(copy.deepcopy(instance.last_report))
            return step
        for owner in (legacy.FoundationTrainer, objective.ObjectiveFoundationTrainer):
            cls.stack.enter_context(patch.object(owner, "step", tracked_step(owner.step)))

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()
        forwards = sum(row["neural_attempted_microbatches"] for row in cls.work["step_reports"])
        episodes = sum(row["neural_attempted_episode_exposures"] for row in cls.work["step_reports"])
        cls.work.update(schema="bic-objective-training-cpu-proof-v1", configuration=vars(cls.config),
            fixture_plan_seeds=[831908001, 831908002], learner_seed=831908799,
            index_construction=cls.index.construction, plan_bundles=len(cls.plan["bundles"]),
            canonical_episode_inventory=len(cls.plan["bundles"])*3*2,
            attempted_family_forwards=forwards, attempted_episode_forwards=episodes,
            optimizer_update_bound=8, family_forward_bound=30, episode_forward_bound=60,
            backward_bound=30, model_construction_bound=12,
            historical_or_formal_inputs=False, historical_checkpoint_loads=0,
            analytical_reply_fixture_batches=1, analytical_fixture_model_passes=0,
            wall_seconds=time.monotonic()-cls.started,
            within_approved_bounds=(cls.work["optimizer_updates"] <= 8 and forwards <= 30 and episodes <= 60
                and cls.work["backward_attempts"] <= 30 and cls.work["model_constructions_attempted"] <= 12))
        print("OBJECTIVE_TRAINING_CPU_WORK=" + json.dumps(cls.work, sort_keys=True))
        path = os.environ.get("BIC_OBJECTIVE_TRAINING_ACCOUNTING")
        if path:
            with Path(path).open("x", encoding="utf8") as stream:
                json.dump(cls.work, stream, indent=2, sort_keys=True)
        if not cls.work["within_approved_bounds"]:
            raise AssertionError("objective fixture exceeded its approved work bounds")

    def trainer(self, objective_id="baseline", **kwargs):
        options = dict(objective_id=objective_id, seed=831908799, order="mixed", config=self.config,
            admission_protected_transcripts=self.history, protected_transcripts=self.protection,
            admission_receipt=self.admission, plan_index=self.index)
        options.update(kwargs)
        return objective.ObjectiveFoundationTrainer(self.plan, **options)

    def test_01_both_objectives_exact_resume_and_baseline_original_arithmetic(self):
        old = legacy.FoundationTrainer(self.plan, "mixed", seed=831908799, config=self.config,
            protected_transcripts=self.protection)
        baseline, candidate = self.trainer(), self.trainer("balanced_reply")
        type(self).baseline, type(self).candidate = baseline, candidate
        type(self).zero_baseline = baseline.snapshot()
        self.assertTrue(same(arithmetic(old.snapshot()), arithmetic(baseline.snapshot())))
        self.assertTrue(same(arithmetic(baseline.snapshot()), arithmetic(candidate.snapshot())))
        a, b = old.step(), baseline.step()
        self.assertTrue(same(arithmetic(old.snapshot()), arithmetic(baseline.snapshot())))
        for name in objective.LOSS_FIELDS:
            self.assertEqual(a[name], b[name])
        resumed_baseline = self.trainer(payload=baseline.snapshot())
        self.assertTrue(same(baseline.snapshot(), resumed_baseline.snapshot()))
        a, b = old.step(), baseline.step()
        resumed_baseline.step()
        self.assertTrue(same(arithmetic(old.snapshot()), arithmetic(baseline.snapshot())))
        self.assertTrue(same(arithmetic(baseline.snapshot()), arithmetic(resumed_baseline.snapshot())))
        for name in objective.LOSS_FIELDS:
            self.assertEqual(a[name], b[name])
        candidate.step()
        resumed_candidate = self.trainer("balanced_reply", payload=candidate.snapshot())
        self.assertTrue(same(candidate.snapshot(), resumed_candidate.snapshot()))
        candidate.step()
        resumed_candidate.step()
        self.assertTrue(same(arithmetic(candidate.snapshot()), arithmetic(resumed_candidate.snapshot())))
        self.assertFalse(same(candidate.snapshot()["weights"], baseline.snapshot()["weights"]))
        self.assertEqual(len(candidate.snapshot()["optimizer"]["state"]), len(list(candidate.model.parameters())))
        self.assertTrue(candidate.setup_report["index_reused"])
        self.assertEqual(candidate.snapshot()["recipe"]["objective"]["id"], "balanced_reply")
        self.assertEqual(baseline.snapshot()["schema"], objective.SCHEMA)
        self.assertEqual(self.work["optimizer_updates"], 8)

    def test_02_strict_restore_rejects_cross_objective_and_tensor_corruption(self):
        trainer = self.candidate
        saved = trainer.snapshot()
        model, optimizer = trainer.model, trainer.optimizer
        for payload in (self.baseline.snapshot(), dict(saved, schema=legacy.SCHEMA)):
            before = self.work["model_constructions_attempted"]
            with self.assertRaises(ValueError): trainer.restore(payload)
            self.assertEqual(self.work["model_constructions_attempted"], before)
        for kind in ("alias", "moment", "dtype"):
            malformed = copy.deepcopy(saved)
            if kind == "alias": malformed["weights"]["observation_head.weight"][0, 0] += 1
            elif kind == "moment": malformed["optimizer"]["state"][0]["exp_avg_sq"].flatten()[0] = -1
            else: malformed["optimizer"]["state"][0]["exp_avg"] = malformed["optimizer"]["state"][0]["exp_avg"].double()
            with self.subTest(kind=kind), self.assertRaises(ValueError): trainer.restore(malformed)
            self.assertIs(trainer.model, model)
            self.assertIs(trainer.optimizer, optimizer)
            self.assertTrue(same(saved, trainer.snapshot()))
        for kind in ("source", "objective", "index", "evidence", "nan"):
            malformed = copy.deepcopy(saved)
            if kind == "source": malformed["recipe"]["source_sha256"] = {}
            elif kind == "objective": malformed["recipe"]["objective"]["weights"]["reply"] = 0
            elif kind == "index": malformed["recipe"]["plan_index_identity"]["bundle_count"] += 1
            elif kind == "evidence": malformed["evidence"]["cursor"] += 1
            else: malformed["weights"]["action_head.bias"][0] = float("nan")
            before = self.work["model_constructions_attempted"]
            with self.subTest(kind=kind), self.assertRaises(ValueError): trainer.restore(malformed)
            self.assertEqual(self.work["model_constructions_attempted"], before)
            self.assertTrue(same(saved, trainer.snapshot()))
        malformed = copy.deepcopy(self.zero_baseline)
        malformed["weights"]["action_head.bias"][0] += 1
        with self.assertRaisesRegex(ValueError, "seeded initialization"):
            self.baseline.restore(malformed)

    def test_03_failed_second_family_poison_and_index_only_restore(self):
        trainer = self.candidate
        saved = trainer.snapshot()
        forward = trainer.model.forward
        calls = 0
        def failed(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("synthetic second family")
            return forward(*args, **kwargs)
        with patch.object(trainer.model, "forward", side_effect=failed), self.assertRaises(KeyboardInterrupt):
            trainer.step()
        self.assertEqual(trainer.last_report["physical_optimizer_updates"], 0)
        self.assertEqual(trainer.last_report["neural_attempted_episode_exposures"], 4)
        self.assertEqual(trainer.last_report["completed_microbatch_episode_exposures"], 2)
        with self.assertRaises(RuntimeError): trainer.snapshot()
        with patch.object(legacy, "_materialize_validated_bundle", side_effect=AssertionError("unexpected replay")), \
                patch.object(indexing, "_materialize_validated_bundle", side_effect=AssertionError("unexpected index scan")):
            trainer.restore(saved)
        self.assertTrue(same(saved, trainer.snapshot()))

    def test_04_invalid_objective_or_index_refuses_before_model(self):
        for options in (dict(objective_id="arbitrary"), dict(objective_id=lambda *args: None),
                dict(plan_index=None), dict(plan_index={}), dict(admission_receipt={}),
                dict(admission_protected_transcripts=[])):
            before = self.work["model_constructions_attempted"]
            with self.subTest(options=options), self.assertRaises(ValueError): self.trainer(**options)
            self.assertEqual(self.work["model_constructions_attempted"], before)
        saved = self.baseline.snapshot()
        with patch.object(objective, "source_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "source identity"):
                self.baseline.restore(saved)
        self.assertTrue(same(saved, self.baseline.snapshot()))

    def test_05_one_analytical_full_vocabulary_reply_fixture(self):
        labels = torch.tensor([[0, 3, 3], [1, 3, 2]], dtype=torch.long)
        lengths = [[1, 2, 4], [3, 1, 2]]
        targets = torch.zeros((2, 3, 4), dtype=torch.long)
        logits = torch.zeros((2, 3, 4, 259), dtype=torch.float64)
        utterances = {}
        for episode in range(2):
            for turn in range(3):
                losses = []
                for token in range(lengths[episode][turn]):
                    target = ByteCodec.EOS if token == lengths[episode][turn]-1 else 20+token
                    targets[episode, turn, token] = target
                    score = .4*(episode*3+turn) + .2*token
                    logits[episode, turn, token, target] = score
                    losses.append(math.log(math.exp(score)+258)-score)
                utterances[episode, turn] = sum(losses)/len(losses)
        class_means = [sum(value for coordinate, value in utterances.items() if int(labels[coordinate]) == target)
            / int(labels.eq(target).sum()) for target in range(4)]
        expected = sum(class_means)/4
        batch = dict(inputs=dict(eos_positions=torch.tensor([[1, 3, 5], [1, 3, 5]])), supervision=dict(
            action_targets=labels, reply_targets=targets, observation_next_byte_targets=torch.full((2, 6), 3)))
        output = dict(logits=torch.zeros((2, 3, 4), dtype=torch.float64), language_logits=logits,
            observation_language_logits=torch.zeros((2, 6, 259), dtype=torch.float64))
        before = copy.deepcopy((output, batch))
        baseline = sequence_objective(output, batch)
        new_baseline = objective.objective("baseline", output, batch)
        candidate = objective.objective("balanced_reply", output, batch)
        self.assertTrue(same(baseline, new_baseline))
        self.assertAlmostEqual(float(candidate["reply_loss"]), expected, places=12)
        self.assertNotAlmostEqual(float(baseline["reply_loss"]), expected, places=7)
        self.assertTrue(torch.equal(candidate["action_loss"], baseline["action_loss"]))
        self.assertTrue(torch.equal(candidate["observation_language_loss"], baseline["observation_language_loss"]))
        self.assertTrue(same((output, batch), before))
        for kind in ("label", "empty_turn", "vocabulary"):
            bad_output, bad_batch = copy.deepcopy((output, batch))
            if kind == "label": bad_batch["supervision"]["action_targets"][0, 0] = 4
            elif kind == "empty_turn": bad_batch["supervision"]["reply_targets"][0, 0].zero_()
            else: bad_output["language_logits"] = bad_output["language_logits"][..., :258]
            with self.subTest(kind=kind), self.assertRaises(ValueError): objective.balanced_reply_loss(bad_output, bad_batch)


if __name__ == "__main__":
    unittest.main()
