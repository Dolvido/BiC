"""Tiny CPU transaction tests, not curriculum or capability measurements.

All canonical fixtures use separate synthetic recipe namespaces. Physical CPU
step attempts/completions are reported once per test-class run for engineering
accounting; no formal study bank, checkpoint or score is opened.
"""
import copy
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from experiments.composition_curriculum import FAMILIES, generate_pair, query_ancestries
from experiments.realization_backend import RealizationBackend, progress_vector
from experiments.realization_training import RealizationTrainer, stream_evidence
from experiments.sequence_student import SequenceConfig


def evaluation_rows(seed, prefix):
    result = {}
    for index, family in enumerate(FAMILIES):
        for attempt in range(1000):
            pair = generate_pair(family, seed + index * 10000 + attempt,
                                 split="dev", turns=8, structure_split="train")
            if not any(query["known"] and query["composed"] and query["structure_partition"] == "audit"
                       for row in pair for query in query_ancestries(row)):
                result[f"{prefix}/{family}/t8"] = pair
                break
        else:
            raise AssertionError("could not construct development-only fixture")
    return result


class EnglishLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.banks = {family: {8: generate_pair(family, 301000000 + index * 10000, turns=8)}
                     for index, family in enumerate(FAMILIES)}
        cls.evaluation = {"development": evaluation_rows(311000000, "practice"),
                          "retention": evaluation_rows(321000000, "anchor")}
        cls.model_config = SequenceConfig(width=16, layers=1, heads=2, feedforward=32, max_turns=12)
        cls.step_attempts = cls.completed_steps = cls.drawn_episode_exposures = 0
        real_step = RealizationTrainer.step
        real_draw = RealizationTrainer._draw

        def counted_step(trainer, *args, **kwargs):
            cls.step_attempts += 1
            result = real_step(trainer, *args, **kwargs)
            cls.completed_steps += 1
            return result

        def counted_draw(trainer, *args, **kwargs):
            result = real_draw(trainer, *args, **kwargs)
            cls.drawn_episode_exposures += trainer.micro_batch_size
            return result

        cls.step_counter = patch.object(RealizationTrainer, "step", new=counted_step)
        cls.draw_counter = patch.object(RealizationTrainer, "_draw", new=counted_draw)
        cls.step_counter.start()
        cls.draw_counter.start()

    @classmethod
    def tearDownClass(cls):
        cls.step_counter.stop()
        cls.draw_counter.stop()
        torch.set_num_threads(cls.threads)
        print(f"English-loop CPU engineering only: {cls.step_attempts} step attempts, "
              f"{cls.completed_steps} completed updates, {cls.drawn_episode_exposures} drawn episode exposures "
              "including failed/discarded attempts; 6 episodes per completed update.")

    def backend(self, *, evaluation=None):
        return RealizationBackend(self.banks, mode="fresh", protected_transcripts=[],
            evaluation_banks=self.evaluation if evaluation is None else evaluation,
            micro_batch_size=2, config=self.model_config, seed=3201, sampler_seed=4201)

    def loop(self, *, backend=None, **options):
        from experiments.english_allocation import AllocationConfig
        from experiments.english_loop import EnglishLoop
        # Short windows exercise a full observe/plan/consume/evaluate boundary.
        allocation = AllocationConfig(mode=options.pop("mode", "joint"), window_updates=2,
                                      min_family_microbatches=1)
        return EnglishLoop(self.backend() if backend is None else backend, allocation,
            max_seen_transcripts=options.pop("max_seen_transcripts", 1000),
            history_limit=options.pop("history_limit", 2), **options)

    def load(self, path):
        from experiments.english_loop import EnglishLoop
        return EnglishLoop.load(path, self.banks, protected_transcripts=[],
                                evaluation_banks=self.evaluation, device="cpu")

    def compare_tree(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.compare_tree(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.compare_tree(a, b)
        else:
            self.assertEqual(left, right)

    def compare_learning(self, left, right):
        from experiments.english_allocation import status
        first, second = left.backend.snapshot(), right.backend.snapshot()
        self.compare_tree(first["curriculum"], second["curriculum"])
        for field in ("weights", "optimizer", "samplers", "exposures", "family_microbatches", "bucket_microbatches"):
            self.compare_tree(first["learner"][field], second["learner"][field])
        self.compare_tree(stream_evidence(first["learner"]["realization"]),
                          stream_evidence(second["learner"]["realization"]))
        a, b = left.snapshot()["state"], right.snapshot()["state"]
        for key in ("phase", "evaluation_kind", "evaluation_cursor", "base_updates",
                    "base_family_microbatches", "windows_completed", "history_dropped"):
            self.compare_tree(a[key], b[key])
        self.compare_tree(status(a["allocation"]), status(b["allocation"]))
        for key in ("schedule", "reference"):
            self.compare_tree(a["allocation"][key], b["allocation"][key])
        first_decision = copy.deepcopy(a["allocation"]["decision"])
        second_decision = copy.deepcopy(b["allocation"]["decision"])
        for decision in (first_decision, second_decision):
            # The shared measured time denominator can vary across CPU calls;
            # schedules, alarms, count-backed deltas and chosen families cannot.
            decision.pop("progress_per_shared_window_second")
        self.compare_tree(first_decision, second_decision)

    def test_exact_joint_and_progress_learning_state_resume_from_one_atomic_envelope(self):
        for mode, updates in (("joint", 3), ("progress", 4)):
            with self.subTest(mode=mode):
                reference = self.loop(mode=mode)
                reference.run(max_updates=updates)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "loop.pt"
                    first = self.loop(mode=mode)
                    first.save(path)
                    first.run(max_updates=1)
                    self.assertEqual(first.backend.updates, 1)
                    restored = self.load(path)
                    self.compare_tree(first.snapshot(), restored.snapshot())
                    restored.run(max_updates=updates - 1)
                    self.compare_learning(reference, restored)
                    self.compare_tree(restored.snapshot(), self.load(path).snapshot())
                    self.assertFalse(restored.status["automatic_promotion"])
                    if mode == "progress":
                        decision = restored.snapshot()["state"]["allocation"]["decision"]
                        self.assertNotEqual(decision["reason"], "joint_cold_start")
                        self.assertEqual(decision["policy"], "ENGINEERED")

    def test_missing_or_mislabeled_evaluation_roles_and_bound_backend_rejected(self):
        missing_role = {"development": self.evaluation["development"]}
        with self.assertRaises(ValueError):
            self.loop(backend=self.backend(evaluation=missing_role))
        missing_family = copy.deepcopy(self.evaluation)
        missing_family["retention"].pop("anchor/count/t8")
        with self.assertRaises(ValueError):
            self.loop(backend=self.backend(evaluation=missing_family))
        mislabeled = copy.deepcopy(self.evaluation)
        mislabeled["retention"]["anchor/color/t8"] = mislabeled["retention"]["anchor/count/t8"]
        with self.assertRaises(ValueError):
            self.loop(backend=self.backend(evaluation=mislabeled))
        with tempfile.TemporaryDirectory() as directory:
            bound = self.backend()
            bound.save(Path(directory) / "backend.pt")
            with self.assertRaises(ValueError):
                self.loop(backend=bound)

    def test_pending_backend_rejected_but_completed_start_uses_relative_policy_counts(self):
        pending = self.backend()
        pending.train_chunk([FAMILIES] * 2, max_updates=1)
        with self.assertRaises(ValueError):
            self.loop(backend=pending)
        complete = self.backend()
        complete.train_chunk([FAMILIES], max_updates=1)
        loop = self.loop(backend=complete)
        self.assertEqual(loop.snapshot()["state"]["base_updates"], 1)
        loop.run(max_updates=1)
        self.assertEqual(loop.backend.updates, 2)
        self.assertEqual(loop.status["total_updates"], 1)

    def test_history_has_an_explicit_bound_across_completed_windows(self):
        loop = self.loop(history_limit=1)
        loop.run(max_updates=5)
        state = loop.snapshot()["state"]
        self.assertLessEqual(len(state["history"]), 1)
        self.assertGreaterEqual(state["windows_completed"], 2)
        self.assertGreaterEqual(state["history_dropped"], 1)

    def test_failed_wrapper_commit_rolls_controller_and_learning_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loop.pt"
            loop = self.loop()
            loop.save(path)
            loop.run(max_updates=1)
            before, disk = loop.snapshot(), path.read_bytes()
            with patch("experiments.english_loop.os.replace", side_effect=OSError("wrapper save failure")):
                with self.assertRaisesRegex(OSError, "wrapper save failure"):
                    loop.run(max_updates=1)
            self.compare_tree(loop.snapshot(), before)
            self.assertEqual(path.read_bytes(), disk)
            self.compare_tree(self.load(path).snapshot(), before)
            self.assertEqual(loop.last_report["discarded_completed_updates"], 1)
            self.assertEqual(loop.last_report["retained_updates"], 0)

    def test_preflight_failure_replaces_prior_report_without_changing_learning(self):
        loop = self.loop()
        prior = loop.run(max_updates=0)
        before = loop.snapshot()
        self.assertEqual(prior["status"], "update_bound")
        with patch("experiments.english_loop.source_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "source or execution profile changed"):
                loop.run(max_updates=1)
        self.assertEqual(loop.last_report["status"], "failed")
        self.assertEqual(loop.last_report["failure_stage"], "preflight")
        self.assertEqual(loop.last_report["retained_updates"], 0)
        self.assertEqual(loop.last_report["failed_step_attempts"], 0)
        self.assertGreaterEqual(loop.last_report["wall_seconds"], 0.)
        self.assertFalse(loop.last_report["publication_uncertain"])
        self.assertEqual(prior["status"], "update_bound")
        self.compare_tree(loop.snapshot(), before)

    def test_failed_evaluation_wall_is_accounted_without_advancing_cursor(self):
        loop = self.loop()
        before, clock = loop.snapshot(), [0.]

        def failed_evaluation(*args, **kwargs):
            clock[0] += 4.
            raise RuntimeError("failed scoring attempt")

        with patch("experiments.english_loop.time.monotonic", side_effect=lambda: clock[0]), \
             patch.object(loop.backend, "evaluate", side_effect=failed_evaluation):
            with self.assertRaisesRegex(RuntimeError, "failed scoring attempt"):
                loop.run(max_updates=1)
        report = loop.last_report
        self.assertEqual(report["failure_stage"], "evaluation")
        self.assertEqual(report["evaluation_seconds"], 4.)
        self.assertEqual(report["failed_evaluation_seconds"], 4.)
        self.assertEqual(report["failed_training_work_seconds"], 0.)
        self.assertEqual(report["wall_seconds"], 4.)
        self.assertEqual(report["retained_updates"], 0)
        self.compare_tree(loop.snapshot(), before)

    def test_partial_failed_training_wall_and_exposures_are_not_hidden(self):
        loop = self.loop()
        loop.run(max_updates=1)
        before, clock = loop.snapshot(), [0.]
        draw = loop.backend._trainer._draw
        prior_exposures = type(self).drawn_episode_exposures

        def consumed_then_failed(*args, **kwargs):
            draw(*args, **kwargs)
            clock[0] += 3.
            raise RuntimeError("failed after one consumed microbatch")

        with patch("experiments.english_loop.time.monotonic", side_effect=lambda: clock[0]), \
             patch.object(loop.backend._trainer, "_draw", side_effect=consumed_then_failed):
            with self.assertRaisesRegex(RuntimeError, "one consumed microbatch"):
                loop.run(max_updates=1)
        report = loop.last_report
        self.assertEqual(report["failure_stage"], "training")
        self.assertEqual(report["training_work_seconds"], 3.)
        self.assertEqual(report["failed_training_work_seconds"], 3.)
        self.assertEqual(report["failed_evaluation_seconds"], 0.)
        self.assertEqual(report["failed_step_attempts"], 1)
        self.assertEqual(report["retained_updates"], 0)
        self.assertEqual(report["wall_seconds"], 3.)
        self.assertEqual(type(self).drawn_episode_exposures - prior_exposures, 2)
        self.compare_tree(loop.snapshot(), before)

    def test_latest_and_previous_saved_evidence_need_canonical_counts_even_when_internally_consistent(self):
        from experiments import english_allocation as allocation
        loop = self.loop()
        loop.run(max_updates=3)
        before = loop.snapshot()
        self.assertIsNotNone(before["state"]["allocation"]["previous"])
        with tempfile.TemporaryDirectory() as directory:
            for corruption in ("latest", "previous", "both_turns"):
                with self.subTest(corruption=corruption):
                    payload = copy.deepcopy(before)
                    policy = payload["state"]["allocation"]
                    keys = ("latest", "previous") if corruption == "both_turns" else (corruption,)
                    for key in keys:
                        observation = policy[key]
                        response = observation["evidence"]["development"]
                        name = "practice/count/t8"
                        row = response["per_bank"][name]
                        if corruption == "both_turns":
                            query = next(i for i, turn in enumerate(row["by_turn"][:-1]) if turn["total"])
                            statement = next(i for i, turn in enumerate(row["by_turn"][:-1]) if not turn["total"])
                            row["by_turn"][query], row["by_turn"][statement] = row["by_turn"][statement], row["by_turn"][query]
                            row["by_turn"][query]["turn"] = query
                            row["by_turn"][statement]["turn"] = statement
                        else:
                            # Exchange known label identities in both matrix axes:
                            # every aggregate and internal consistency check still holds.
                            self.assertNotEqual(row["per_target"]["0"]["total"], row["per_target"]["1"]["total"])
                            order = (1, 0, 2, 3)
                            matrix = row["confusion_matrix"]
                            row["confusion_matrix"] = [[matrix[i][j] for j in order] for i in order]
                            row["per_target"]["0"], row["per_target"]["1"] = row["per_target"]["1"], row["per_target"]["0"]
                        vector = progress_vector(row)
                        response["progress"][name] = copy.deepcopy(vector)
                        observation["vectors"]["development"][name] = copy.deepcopy(vector)
                    # These are coherent policy observations, but disagree with
                    # the immutable English examples at the scoring boundary.
                    allocation.restore(policy, loop.config, payload["bank_specs"])
                    with self.assertRaises(ValueError):
                        loop._apply(payload)  # Previously cached responses cannot bless changed content.
                    self.compare_tree(loop.snapshot(), before)
                    path = Path(directory) / f"{corruption}.pt"
                    torch.save(payload, path)
                    with self.assertRaises(ValueError):
                        self.load(path)

    def test_failed_rollback_blocks_use_and_reports_retention_as_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loop.pt"
            loop = self.loop()
            loop.save(path)
            loop.run(max_updates=1)
            before, disk = loop.snapshot(), path.read_bytes()
            with patch("experiments.english_loop.os.replace", side_effect=OSError("save failed")), \
                 patch.object(loop, "_apply", side_effect=RuntimeError("rollback unavailable")):
                with self.assertRaisesRegex(RuntimeError, "rollback unavailable"):
                    loop.run(max_updates=1)
            report = copy.deepcopy(loop.last_report)
            self.assertTrue(report["recovery_failed"])
            self.assertTrue(report["publication_uncertain"])
            for name in ("retained_updates", "ending_updates", "discarded_completed_updates"):
                self.assertIsNone(report[name])
            self.assertTrue(loop.status["publication_uncertain"])
            self.assertIsNone(loop.status["updates"])
            self.assertEqual(loop.status["in_memory_updates"], 2)
            self.assertEqual(path.read_bytes(), disk)
            self.compare_tree(self.load(path).snapshot(), before)
            with patch.object(loop.backend._trainer, "step", side_effect=AssertionError("uncertain learner used")):
                with self.assertRaisesRegex(RuntimeError, "publication is uncertain"):
                    loop.run(max_updates=1)
            self.assertEqual(loop.last_report["failure_stage"], "preflight")
            self.assertIsNone(loop.last_report["retained_updates"])

    def test_public_wall_includes_lock_wait_and_returned_report_is_detached(self):
        loop, clock = self.loop(), [0.]

        class DelayedLock:
            first = True

            def __enter__(self):
                if self.first:
                    self.first = False
                    clock[0] += 5.
                return self

            def __exit__(self, *args):
                return False

        with patch.object(loop, "_lock", DelayedLock()), \
             patch("experiments.english_loop.time.monotonic", side_effect=lambda: clock[0]):
            returned = loop.run(max_updates=0)
        self.assertEqual(returned["wall_seconds"], 5.)
        self.assertEqual(loop.last_report["wall_seconds"], 5.)
        returned["wall_seconds"] = -1.
        self.assertEqual(loop.last_report["wall_seconds"], 5.)

    def test_exception_after_replace_retains_the_actually_published_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loop.pt"
            loop = self.loop()
            loop.save(path)
            loop.run(max_updates=1)
            replace = os.replace

            def published_then_interrupted(*args):
                replace(*args)
                raise KeyboardInterrupt("published before interruption")

            with patch("experiments.english_loop.os.replace", side_effect=published_then_interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    loop.run(max_updates=1)
            self.assertEqual(loop.backend.updates, 2)
            self.compare_tree(loop.snapshot(), self.load(path).snapshot())
            self.assertEqual(loop.snapshot()["state"]["phase"], "evaluate")
            self.assertEqual(loop.last_report["discarded_completed_updates"], 0)
            self.assertEqual(loop.last_report["retained_updates"], 1)

    def test_stale_writer_cannot_overwrite_another_committed_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loop.pt"
            first = self.loop()
            first.save(path)
            stale = self.load(path)
            first.run(max_updates=1)
            published = path.read_bytes()
            with self.assertRaises((ValueError, RuntimeError)):
                stale.run(max_updates=1)
            self.assertEqual(path.read_bytes(), published)
            self.assertEqual(stale.backend.updates, 0)

    def test_zero_and_expired_budgets_start_neither_evaluation_nor_training(self):
        for budget in ({"max_updates": 0}, {"deadline": time.monotonic() - 1}):
            loop = self.loop()
            before = loop.snapshot()
            with patch.object(loop.backend, "evaluate", side_effect=AssertionError("unexpected evaluation")), \
                 patch.object(loop.backend._trainer, "step", side_effect=AssertionError("unexpected step")):
                loop.run(**budget)
            self.compare_tree(before, loop.snapshot())

    def test_seen_transcript_bound_stops_before_another_optimizer_step(self):
        loop = self.loop(max_seen_transcripts=6)
        loop.run(max_updates=1)
        self.assertEqual(loop.backend.updates, 1)
        self.assertEqual(loop.status["seen_transcripts"], 6)
        with patch.object(loop.backend._trainer, "step", side_effect=AssertionError("seen bound exceeded")):
            loop.run(max_updates=1)
        self.assertEqual(loop.backend.updates, 1)
        self.assertLessEqual(loop.status["seen_transcripts"], 6)

    def test_bad_partial_evaluation_metric_or_identity_does_not_advance(self):
        for corrupt in ("identity", "metric"):
            loop = self.loop()
            before = loop.snapshot()
            evaluate = loop.backend.evaluate

            def invalid(*args, **kwargs):
                result = evaluate(*args, **kwargs)
                row = next(iter(result["per_bank"].values()))
                if corrupt == "identity":
                    row["bank"]["sha256"] = "0" * 64
                else:
                    row["final_pairs"]["correct"] = row["final_pairs"]["total"] + 1
                return result

            with patch.object(loop.backend, "evaluate", side_effect=invalid):
                with self.assertRaises(ValueError):
                    loop.run(max_updates=1)
            self.compare_tree(loop.snapshot(), before)
            self.assertEqual(loop.snapshot()["state"]["evaluation_cursor"], 0)

    def test_one_bank_evaluation_commits_then_resumes_without_repeating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loop.pt"
            loop = self.loop()
            loop.save(path)
            clock, observed = [0.], []
            evaluate = loop.backend.evaluate

            def stop_after_bank(*args, **kwargs):
                observed.append((args, copy.deepcopy(kwargs)))
                result = evaluate(*args, **kwargs)
                clock[0] = 2.
                return result

            with patch("experiments.english_loop.time.monotonic", side_effect=lambda: clock[0]), \
                 patch.object(loop.backend, "evaluate", side_effect=stop_after_bank):
                loop.run(deadline=1.)
            self.assertEqual(len(observed), 1)
            self.assertEqual(loop.backend.updates, 0)
            self.assertEqual(loop.snapshot()["state"]["evaluation_cursor"], 1)
            restored = self.load(path)
            self.compare_tree(loop.snapshot(), restored.snapshot())
            seen = []
            evaluate_restored = restored.backend.evaluate

            def capture(*args, **kwargs):
                seen.append((args, copy.deepcopy(kwargs)))
                return evaluate_restored(*args, **kwargs)

            with patch.object(restored.backend, "evaluate", side_effect=capture):
                restored.run(max_updates=1)
            self.assertEqual(len(seen), 5)
            self.assertNotIn(observed[0], seen)
            self.assertEqual(restored.backend.updates, 1)


if __name__ == "__main__":
    unittest.main()
