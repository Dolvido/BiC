"""End-to-end loop boundaries, retention gates, and parallel/restart fidelity."""

import copy
from dataclasses import asdict, FrozenInstanceError
import hashlib
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.curriculum import CURRICULUM, SKILLS, curriculum_digest, generate, oracle
from brain_in_computer.learning_loop import (
    LearningLoop, LoopConfig, POLICY, promotion_decision, run_lock,
)


def metrics(accuracy=0.5, loss=1.0, brier=0.5):
    return {"accuracy": accuracy, "loss": loss, "brier": brier,
            "per_skill": {skill: {"accuracy": accuracy, "loss": loss, "brier": brier,
                                   "correct": round(accuracy * 100), "total": 100}
                          for skill in SKILLS}}


def tiny_config(**overrides):
    options = dict(hidden_size=8, candidates=1, workers=1, steps=2, batch_size=4,
                   lessons_per_skill=4, replay_per_skill=3, dev_per_skill=4)
    options.update(overrides)
    return LoopConfig(**options)


class LearningLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_retention_anchors_prevent_cumulative_small_regressions(self):
        config = tiny_config()
        before, after = metrics(), metrics(.52, .9, .45)
        anchor = {skill: .70 for skill in SKILLS}
        retained = metrics(.64)
        accepted, reason, _ = promotion_decision(before, after, retained, anchor, config)
        self.assertFalse(accepted)
        self.assertTrue(reason.startswith("anchor_retention:"))
        # The most recent comparison alone would accept this candidate.
        accepted, _, _ = promotion_decision(before, after, metrics(.70), anchor, config)
        self.assertTrue(accepted)

    def test_fresh_skill_regression_and_probability_error_reject_promotion(self):
        config = tiny_config()
        before, after = metrics(), metrics(.6, .9, .45)
        anchor = {skill: .5 for skill in SKILLS}
        after["per_skill"][SKILLS[0]]["accuracy"] = .3
        accepted, reason, _ = promotion_decision(before, after, metrics(), anchor, config)
        self.assertFalse(accepted)
        self.assertTrue(reason.startswith("fresh_retention:"))
        accepted, reason, _ = promotion_decision(before, metrics(.6, .9, .6), metrics(), anchor, config)
        self.assertFalse(accepted)
        self.assertEqual(reason, "calibration_regression")

    def test_nonfinite_metrics_at_every_gate_level_are_rejected(self):
        config = tiny_config()
        anchor = {skill: .5 for skill in SKILLS}
        for location in ("aggregate", "skill", "retention"):
            after, retention = metrics(.6, .9, .4), metrics(.6)
            if location == "aggregate":
                after["loss"] = math.nan
            elif location == "skill":
                after["per_skill"][SKILLS[0]]["accuracy"] = math.nan
            else:
                retention["per_skill"][SKILLS[0]]["accuracy"] = math.nan
            accepted, _, _ = promotion_decision(metrics(), after, retention, anchor, config)
            with self.subTest(location=location):
                self.assertFalse(accepted)

    def test_replay_is_bounded_regenerable_training_descriptors(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = LearningLoop(directory, tiny_config())
            loop.run(.01, max_cycles=3, emit=False)
            self.assertEqual(loop.state["cycle"], 3)
            self.assertLessEqual(len(loop.state["history"]), 128)
            for skill, descriptors in loop.state["replay"].items():
                self.assertLessEqual(len(descriptors), loop.config.replay_per_skill)
                self.assertEqual(len(descriptors), len(set(descriptors)))
                for descriptor in descriptors:
                    self.assertIs(type(descriptor), int)
                    lesson = generate(skill, descriptor, 1, split="train")[0]
                    self.assertEqual(lesson["split"], "train")
                    self.assertEqual(lesson["seed"], descriptor)
            self.assertGreater(sum(map(len, loop.state["replay"].values())), 0)

    def test_audit_does_not_change_checkpoint_weights_or_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = LearningLoop(directory, tiny_config())
            loop.run(.01, max_cycles=1, emit=False)
            state, weights = copy.deepcopy(loop.state), copy.deepcopy(loop.weights)
            before = hashlib.sha256(loop.path.read_bytes()).hexdigest()
            report = loop.audit(count=4)
            self.assertTrue(report["checkpoint_unchanged"])
            self.assertEqual(report["checkpoint_sha256"], before)
            self.assertEqual(hashlib.sha256(loop.path.read_bytes()).hexdigest(), before)
            self.assert_tree_equal(loop.state, state)
            self.assert_tree_equal(loop.weights, weights)
            self.assertEqual(len(list(Path(directory).glob("audit-*.json"))), 1)
            with self.assertRaises(FileExistsError):
                loop.audit(count=4)

    def test_resume_matches_uninterrupted_learning(self):
        with tempfile.TemporaryDirectory() as root:
            config = tiny_config()
            full = LearningLoop(Path(root) / "full", config)
            split = LearningLoop(Path(root) / "split", config)
            # Force actual candidates through selection so unchanged initial
            # weights cannot make a broken continuation pass accidentally.
            with patch("brain_in_computer.learning_loop.promotion_decision",
                       return_value=(True, "test_verified_benefit", .1)):
                full.run(.01, max_cycles=2, emit=False)
                split.run(.01, max_cycles=1, emit=False)
                resumed = LearningLoop(split.directory, resume=True)
                resumed.run(.01, max_cycles=1, emit=False)
            self.assertEqual(full.state["promotions"], 2)
            self.assert_tree_equal(full.weights, resumed.weights)
            self.assert_tree_equal(full.optimizer, resumed.optimizer)
            for key in ("cycle", "promotions", "updates", "examples_seen", "strategy",
                        "replay", "attempts", "last_practiced", "progress", "best_retention"):
                with self.subTest(key=key):
                    self.assert_tree_equal(full.state[key], resumed.state[key])

    def test_parallel_candidates_match_serial_candidates(self):
        with tempfile.TemporaryDirectory() as root:
            serial = LearningLoop(Path(root) / "serial", tiny_config(candidates=2, workers=1))
            parallel = LearningLoop(Path(root) / "parallel", tiny_config(candidates=2, workers=2))
            before = torch.random.get_rng_state().clone()
            with patch("brain_in_computer.learning_loop.promotion_decision",
                       return_value=(True, "test_verified_benefit", .1)):
                serial.run(.01, max_cycles=1, emit=False)
                parallel.run(.01, max_cycles=1, emit=False)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            self.assertEqual(serial.state["history"][-1]["winner"],
                             parallel.state["history"][-1]["winner"])
            self.assert_tree_equal(serial.weights, parallel.weights)
            self.assert_tree_equal(serial.optimizer, parallel.optimizer)

    def test_rejected_exploration_keeps_learning_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            config = tiny_config(candidates=2, workers=2)
            full = LearningLoop(Path(root) / "full", config)
            split = LearningLoop(Path(root) / "split", config)
            champion = copy.deepcopy(full.weights)
            with patch("brain_in_computer.learning_loop.promotion_decision",
                       return_value=(False, "test_insufficient_benefit", 0.0)):
                full.run(.01, max_cycles=2, emit=False)
                split.run(.01, max_cycles=1, emit=False)
                first_branch = copy.deepcopy(split.exploration[1])
                resumed = LearningLoop(split.directory, resume=True)
                self.assert_tree_equal(first_branch, resumed.exploration[1])
                resumed.run(.01, max_cycles=1, emit=False)
            self.assertEqual(full.state["promotions"], 0)
            self.assertEqual(set(resumed.exploration), {1})
            self.assertEqual(first_branch["age"], 1)
            self.assertEqual(resumed.exploration[1]["age"], 2)
            self.assert_tree_equal(champion, full.weights)
            self.assert_tree_equal(champion, resumed.weights)
            self.assertIsNone(resumed.optimizer)
            self.assert_tree_equal(full.exploration, resumed.exploration)
            self.assertTrue(any(not torch.equal(value, resumed.exploration[1]["weights"][name])
                                for name, value in first_branch["weights"].items()))
            self.assertTrue(any(not torch.equal(value, first_branch["weights"][name])
                                for name, value in champion.items()))

    def test_champion_promotion_resets_unpromoted_exploration(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = LearningLoop(directory, tiny_config(candidates=2))
            with patch("brain_in_computer.learning_loop.promotion_decision",
                       return_value=(False, "test_insufficient_benefit", 0.0)):
                loop.run(.01, max_cycles=1, emit=False)
            self.assertIn(1, loop.exploration)
            with patch("brain_in_computer.learning_loop.promotion_decision",
                       return_value=(True, "test_verified_benefit", .1)):
                loop.run(.01, max_cycles=1, emit=False)
            self.assertEqual(loop.state["history"][-1]["winner"], 0)
            self.assertEqual(loop.exploration, {})

    def test_zero_update_deadline_preserves_previously_learned_exploration(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = LearningLoop(directory, tiny_config(candidates=2))
            with patch("brain_in_computer.learning_loop.promotion_decision",
                       return_value=(False, "test_insufficient_benefit", 0.0)):
                loop.run(.01, max_cycles=1, emit=False)
            learned_branch = copy.deepcopy(loop.exploration)

            def no_updates(weights, _lessons, **options):
                return {"state_dict": copy.deepcopy(weights),
                        "optimizer_state": copy.deepcopy(options["optimizer_state"]),
                        "updates": 0, "examples_seen": 0, "training_seconds": 0.0,
                        "replay_examples_seen": 0, "loss": None, "deadline_reached": True}

            with patch("brain_in_computer.learning_loop.train_candidate", side_effect=no_updates):
                loop.run(.01, max_cycles=1, emit=False)
            self.assert_tree_equal(loop.exploration, learned_branch)

    def test_evolution_changes_only_learning_rate_and_replay_not_policy_or_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = LearningLoop(directory, tiny_config(candidates=4, workers=2))
            policy, registry = copy.deepcopy(POLICY), copy.deepcopy(CURRICULUM)
            frozen_config, digest = asdict(loop.config), curriculum_digest()
            reference = generate("color", 337, 1, split="train")[0]
            answer = oracle(reference)
            with self.assertRaises(FrozenInstanceError):
                loop.config.retention_tolerance = 1.0
            with patch("brain_in_computer.learning_loop.promotion_decision",
                       return_value=(True, "test_verified_benefit", .1)):
                loop.run(.01, max_cycles=2, emit=False)
            for event in loop.state["history"]:
                for candidate in event["candidates"]:
                    strategy = candidate["strategy"]
                    self.assertEqual(set(strategy), {"learning_rate", "replay_fraction"})
                    self.assertGreater(strategy["learning_rate"], 0)
                    self.assertLessEqual(strategy["learning_rate"], .02)
                    self.assertGreaterEqual(strategy["replay_fraction"], .15)
                    self.assertLessEqual(strategy["replay_fraction"], .8)
            self.assertEqual(POLICY, policy)
            self.assertEqual(CURRICULUM, registry)
            self.assertEqual(asdict(loop.config), frozen_config)
            self.assertEqual(curriculum_digest(), digest)
            self.assertEqual(oracle(reference), answer)

    def test_injected_gate_mutation_and_oracle_change_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = LearningLoop(directory, tiny_config())
            champion = copy.deepcopy(loop.weights)
            strategy = dict(loop._strategies()[0], retention_tolerance=1.0)
            with patch.object(loop, "_strategies", return_value=[strategy]):
                with self.assertRaises(TypeError):
                    loop.run(.01, max_cycles=1, emit=False)
            self.assertEqual(loop.state["cycle"], 0)
            self.assert_tree_equal(loop.weights, champion)
            checkpoint_digest = hashlib.sha256(loop.path.read_bytes()).hexdigest()
            with patch.dict(CURRICULUM["color"], {"oracle": "Invent every answer"}):
                with self.assertRaisesRegex(ValueError, "changed"):
                    loop.run(.01, max_cycles=1, emit=False)
            self.assertEqual(hashlib.sha256(loop.path.read_bytes()).hexdigest(), checkpoint_digest)

    def test_expired_cycle_and_candidate_interrupt_preserve_champion(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = LearningLoop(directory, tiny_config())
            before = copy.deepcopy(loop.snapshot())
            self.assertIsNone(loop.cycle(time.monotonic() - 1))
            self.assert_tree_equal(before, loop.snapshot())
            with patch("brain_in_computer.learning_loop.train_candidate", side_effect=KeyboardInterrupt):
                loop.run(.01, max_cycles=1, emit=False)
            restored = LearningLoop(directory, resume=True)
            self.assertEqual(restored.state["cycle"], 0)
            self.assertEqual(restored.state["status"], "interrupted")
            self.assert_tree_equal(before["weights"], restored.weights)
            self.assert_tree_equal(before["optimizer"], restored.optimizer)

    def test_stale_resumed_writer_cannot_overwrite_newer_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            first = LearningLoop(directory, tiny_config())
            stale = LearningLoop(directory, resume=True)
            first.run(.01, max_cycles=1, emit=False)
            before = hashlib.sha256(first.path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "(?i)stale|changed|advanced"):
                stale.run(.01, max_cycles=1, emit=False)
            self.assertEqual(hashlib.sha256(first.path.read_bytes()).hexdigest(), before)

    def test_interrupt_during_final_candidate_install_rolls_back_all_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = LearningLoop(directory, tiny_config())
            before = copy.deepcopy(loop.snapshot())

            def interrupted_commit(_deadline):
                # Emulate an interrupt after the weight pointer swap and before
                # the controller pointer swap at the end of a staged cycle.
                loop.weights = {name: value + 1 for name, value in loop.weights.items()}
                loop.optimizer = {"partial_candidate": True}
                raise KeyboardInterrupt

            with patch.object(loop, "cycle", side_effect=interrupted_commit):
                loop.run(.01, max_cycles=1, emit=False)
            restored = LearningLoop(directory, resume=True)
            self.assertEqual(restored.state["cycle"], 0)
            self.assert_tree_equal(before["weights"], restored.weights)
            self.assert_tree_equal(before["optimizer"], restored.optimizer)

    def test_source_change_is_detected_without_rewriting_frozen_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = LearningLoop(directory, tiny_config())
            before = hashlib.sha256(loop.path.read_bytes()).hexdigest()
            with patch("brain_in_computer.learning_loop.source_digest", return_value="changed"):
                with self.assertRaisesRegex(ValueError, "changed"):
                    loop.run(.01, max_cycles=1, emit=False)
            self.assertEqual(hashlib.sha256(loop.path.read_bytes()).hexdigest(), before)

    def test_second_writer_is_locked_out_and_lock_is_removed_after_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "sentinel"):
                with run_lock(directory):
                    with self.assertRaisesRegex(RuntimeError, "locked"):
                        with run_lock(directory):
                            pass
                    raise ValueError("sentinel")
            self.assertFalse((Path(directory) / ".learning.lock").exists())


if __name__ == "__main__":
    unittest.main()
