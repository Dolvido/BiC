"""Tiny CPU engineering updates only; no study bank or capability evaluation."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.foundation_plan import build_plan, materialize_bundle
from experiments.foundation_training import FoundationTrainer, replay_evidence
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig


class FoundationTrainingTests(unittest.TestCase):
    physical_updates = 0
    completed_microbatch_exposures = 0
    drawn_episode_exposures = 0
    neural_attempted_episode_exposures = 0

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.plan = build_plan(seed=412000001, stage_updates=10, final_updates=6,
                              rehearsal_every=2, micro_batch_size=2, ordering_seed=412000002)
        cls.config = SequenceConfig(width=8, layers=1, heads=2, feedforward=16, max_turns=12)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        print(f"FoundationTrainer test-only CPU accounting: {cls.physical_updates} completed optimizer updates; "
              f"{cls.drawn_episode_exposures} returned candidate episodes; {cls.neural_attempted_episode_exposures} neural-attempted episodes; "
              f"{cls.completed_microbatch_exposures} completed-microbatch episode exposures (includes failed work).")

    def trainer(self, **options):
        return FoundationTrainer(self.plan, seed=412000003, config=self.config, **options)

    def step(self, trainer):
        try:
            report = trainer.step()
        except BaseException:
            report = trainer.last_report
            if report is not None and report.get("failed"):
                type(self).drawn_episode_exposures += report["drawn_episode_exposures"] or 0
                type(self).neural_attempted_episode_exposures += report["neural_attempted_episode_exposures"]
                type(self).completed_microbatch_exposures += report["completed_microbatch_episode_exposures"]
                if report["physical_optimizer_updates"] is not None:
                    type(self).physical_updates += report["physical_optimizer_updates"]
            raise
        type(self).physical_updates += report["physical_optimizer_updates"]
        type(self).drawn_episode_exposures += report["drawn_episode_exposures"]
        type(self).neural_attempted_episode_exposures += report["neural_attempted_episode_exposures"]
        type(self).completed_microbatch_exposures += report["completed_microbatch_episode_exposures"]
        return report

    def equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.equal(left, right)
        else:
            self.assertEqual(a, b)

    def learning(self, trainer):
        payload = trainer.snapshot(); payload.pop("timing")
        return payload

    def test_exact_cpu_disk_resume_restores_next_bundle_optimizer_and_counts(self):
        for order in ("curriculum", "mixed"):
            with self.subTest(order=order):
                reference, first = self.trainer(order=order), self.trainer(order=order)
                for _ in range(3):
                    self.step(reference)
                self.step(first)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "state.pt"
                    torch.save(first.snapshot(), path)
                    restored = self.trainer(order=order, payload=torch.load(path, weights_only=True))
                self.equal(first.snapshot(), restored.snapshot())
                self.assertGreaterEqual(restored.last_restore_seconds, 0.)
                self.step(restored); self.step(restored)
                self.equal(self.learning(reference), self.learning(restored))
                self.assertEqual(restored.cursor, 3)

    def test_same_ids_and_lessons_across_orders_have_exact_final_exposure_match(self):
        total = len(self.plan["bundles"])
        a = replay_evidence(self.plan, "curriculum", total, include_bundles=True)
        b = replay_evidence(self.plan, "mixed", total, include_bundles=True)
        self.assertEqual(sorted(a["bundles"], key=lambda row: row["bundle_id"]),
                         sorted(b["bundles"], key=lambda row: row["bundle_id"]))
        for name in ("family_microbatches", "bucket_microbatches", "depth_microbatches", "exposures", "cursor"):
            self.assertEqual(a["evidence"][name], b["evidence"][name])
        self.assertNotEqual(a["evidence"]["consumed_ids_sha256"], b["evidence"]["consumed_ids_sha256"])
        self.assertEqual(sum(v["episodes"] for v in a["evidence"]["exposures"].values()), total * 6)

    def test_actual_bytes_tokens_recipe_digests_and_gradient_updates(self):
        trainer = self.trainer()
        before = trainer.snapshot()["weights"]
        report = self.step(trainer)
        expected = replay_evidence(self.plan, "curriculum", 1, include_bundles=True)
        self.assertEqual(trainer.snapshot()["evidence"], expected["evidence"])
        bundle_id = self.plan["schedules"]["curriculum"][0]
        rows = materialize_bundle(self.plan, bundle_id)
        for micro in report["microbatches"]:
            family = micro["family"]
            byte_count = sum(len(turn["text"].encode("utf-8")) for row in rows[family] for turn in row["turns"])
            self.assertEqual(micro["exposures"]["observation_bytes"], byte_count)
            self.assertEqual(micro["exposures"]["observation_tokens"], byte_count + 2 * micro["exposures"]["turns"])
            self.assertEqual(micro["rows_sha256"], expected["bundles"][0]["families"][family]["rows_sha256"])
        self.assertEqual(report["completed_microbatch_episode_exposures"], 6)
        self.assertGreaterEqual(report["step_seconds"], report["materialization_seconds"])
        after = trainer.snapshot()["weights"]
        for key in ("tokens.weight", "action_head.weight", "blocks.0.self_attn.in_proj_weight"):
            self.assertFalse(torch.equal(before[key], after[key]))

    def test_partial_backward_failure_poison_requires_explicit_restoration(self):
        trainer = self.trainer(); original = trainer.snapshot()
        forward, calls = trainer.model.forward, 0
        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected second microbatch failure")
            return forward(*args, **kwargs)
        with patch.object(trainer.model, "forward", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "second microbatch"):
                self.step(trainer)
        self.assertEqual(trainer.last_report["completed_microbatches"], 1)
        self.assertEqual(trainer.last_report["drawn_episode_exposures"], 6)
        self.assertEqual(trainer.last_report["neural_attempted_episode_exposures"], 4)
        self.assertEqual(trainer.last_report["neural_attempted_microbatches"], 2)
        self.assertEqual(trainer.last_report["physical_optimizer_updates"], 0)
        self.assertGreaterEqual(trainer.last_report["wall_seconds"], trainer.last_report["step_seconds"])
        self.assertGreaterEqual(trainer.last_report["step_seconds"], trainer.last_report["materialization_seconds"])
        self.assertEqual(trainer.cursor, 0)
        for operation in (trainer.step, trainer.snapshot):
            with self.assertRaisesRegex(RuntimeError, "restoration"):
                operation()
        self.assertEqual(trainer.last_report["completed_microbatch_episode_exposures"], 0)
        self.assertEqual(trainer.last_report["failure_stage"], "preflight:restore_required")
        trainer.restore(original)
        self.equal(trainer.snapshot(), original)
        reference = self.trainer()
        self.step(trainer); self.step(reference)
        self.equal(self.learning(trainer), self.learning(reference))

    def test_throwing_optimizer_marks_physical_completion_unknown_and_is_not_reusable(self):
        trainer = self.trainer(); original = trainer.snapshot()
        optimize = trainer.optimizer.step
        def update_then_throw(*args, **kwargs):
            value = optimize(*args, **kwargs)
            type(self).physical_updates += 1  # Test harness knows this injected optimizer completed.
            raise RuntimeError("after optimizer mutation")
        with patch.object(trainer.optimizer, "step", side_effect=update_then_throw):
            with self.assertRaisesRegex(RuntimeError, "after optimizer"):
                self.step(trainer)
        self.assertIsNone(trainer.last_report["physical_optimizer_updates"])
        self.assertEqual(trainer.last_report["retained_optimizer_updates"], 0)
        self.assertEqual(trainer.last_report["completed_microbatch_episode_exposures"], 6)
        with self.assertRaises(RuntimeError):
            trainer.snapshot()
        trainer.restore(original)
        self.equal(trainer.snapshot(), original)

    def test_protected_transcript_and_noncanonical_rows_fail_before_neural_work(self):
        bundle_id = self.plan["schedules"]["curriculum"][0]
        rows = materialize_bundle(self.plan, bundle_id)
        protected = [transcript_digest(rows["count"][0])]
        trainer = self.trainer(protected_transcripts=protected)
        with patch.object(trainer.model, "forward", side_effect=AssertionError("model must not run")):
            with self.assertRaisesRegex(ValueError, "protected"):
                self.step(trainer)
        self.assertEqual(trainer.cursor, 0)
        self.assertEqual(trainer.last_report["completed_microbatch_episode_exposures"], 0)
        self.assertEqual(trainer.last_report["drawn_episode_exposures"], 6)
        self.assertEqual(trainer.last_report["neural_attempted_episode_exposures"], 0)
        self.assertFalse(trainer.last_report["materialization_completed"])
        self.assertGreaterEqual(trainer.last_report["step_seconds"], trainer.last_report["materialization_seconds"])
        with self.assertRaises(ValueError):
            replay_evidence(self.plan, "curriculum", 1, protected)
        trainer = self.trainer()
        malformed = copy.deepcopy(rows)
        malformed["color"][0]["turns"][-1]["target"] = 2
        with patch("experiments.foundation_training._materialize_validated_bundle", return_value=malformed):
            with self.assertRaises(ValueError):
                self.step(trainer)
        with self.assertRaises(RuntimeError):
            trainer.snapshot()

    def test_restore_rejects_changed_plan_order_config_protection_and_sources(self):
        trainer = self.trainer(); self.step(trainer)
        payload = trainer.snapshot()
        changed_plan = build_plan(seed=412000010, stage_updates=10, final_updates=6,
                                  rehearsal_every=2, micro_batch_size=2, ordering_seed=412000002)
        with self.assertRaisesRegex(ValueError, "recipe"):
            FoundationTrainer(changed_plan, seed=412000003, config=self.config, payload=payload)
        for options in ({"order": "mixed"}, {"learning_rate": .0003}, {"protected_transcripts": ["0" * 64]}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "recipe"):
                self.trainer(payload=payload, **options)
        with self.assertRaisesRegex(ValueError, "recipe"):
            FoundationTrainer(self.plan, seed=412000003, config=SequenceConfig(width=16, layers=1, heads=2, feedforward=32, max_turns=12), payload=payload)
        malformed = copy.deepcopy(payload)
        malformed["recipe"]["source_sha256"]["experiments/foundation_training.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "recipe"):
            trainer.restore(malformed)

    def test_counter_digest_and_optimizer_corruption_rejected_transactionally(self):
        trainer = self.trainer(); self.step(trainer)
        before = trainer.snapshot()
        mutations = [lambda p: p.update(cursor=True), lambda p: p.update(cursor=2),
                     lambda p: p["evidence"]["exposures"]["color"].update(observation_bytes=1),
                     lambda p: p["evidence"].update(consumed_recipes_sha256="0" * 64),
                     lambda p: p["evidence"]["family_microbatches"].update(color=True),
                     lambda p: p["optimizer"]["param_groups"][0].update(lr=.5),
                     lambda p: p["optimizer"]["state"][0]["step"].fill_(0),
                     lambda p: p["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1),
                     lambda p: p["optimizer"]["state"][0]["exp_avg"].fill_(float("nan")),
                     lambda p: p["optimizer"]["state"][0].update(exp_avg=torch.zeros(1)),
                     lambda p: p["optimizer"]["state"].pop(0),
                     lambda p: p["weights"].update(**{"tokens.weight": p["weights"]["tokens.weight"].double()}),
                     lambda p: p["weights"]["observation_head.weight"].add_(1),
                     lambda p: p["timing"].update(retained_step_seconds=-1)]
        for mutate in mutations:
            payload = copy.deepcopy(before); mutate(payload)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                trainer.restore(payload)
            self.equal(before, trainer.snapshot())

    def test_seeded_initial_weights_are_exact_and_plan_snapshot_aliases_are_isolated(self):
        caller_plan = copy.deepcopy(self.plan)
        trainer = FoundationTrainer(caller_plan, seed=412000003, config=self.config)
        original = trainer.snapshot()
        caller_plan["bundles"]["0"]["depth"] = 5
        recipe = trainer.recipe; recipe["order"] = "mutated"
        payload = trainer.snapshot(); payload["weights"]["tokens.weight"].add_(1)
        payload["evidence"]["exposures"]["color"]["episodes"] = 99
        self.equal(trainer.snapshot(), original)
        with self.assertRaises(ValueError):
            trainer.restore(payload)
        self.equal(trainer.snapshot(), original)
        damaged = copy.deepcopy(original)
        damaged["weights"]["tokens.weight"].add_(1)
        damaged["weights"]["observation_head.weight"].copy_(damaged["weights"]["tokens.weight"])
        with self.assertRaisesRegex(ValueError, "seeded initialization"):
            trainer.restore(damaged)

    def test_model_receives_no_family_depth_recipe_target_or_other_session_inputs(self):
        trainer = self.trainer()
        forward, captured = trainer.model.forward, []
        def inspect(*args, **kwargs):
            self.assertFalse(args)
            self.assertEqual(set(kwargs), {"token_ids", "valid_mask", "lengths", "eos_positions", "decoder_input_ids"})
            captured.append({key: value.detach().clone() for key, value in kwargs.items()})
            return forward(**kwargs)
        with patch.object(trainer.model, "forward", side_effect=inspect):
            self.step(trainer)
        self.assertEqual(len(captured), 3)
        self.assertTrue(all(row["token_ids"].shape[0] == 2 for row in captured))
        self.assertTrue(all(row["eos_positions"].shape[1] == 8 for row in captured))
        # Reply prefixes are a separate decoder argument; model causal isolation
        # is already covered by the unchanged SequenceStudent tests.

    def test_constructor_input_validation_and_source_drift_fail_before_work(self):
        for options in ({"order": "adaptive"}, {"learning_rate": True}, {"learning_rate": float("inf")},
                        {"learning_rate": 0}, {"protected_transcripts": "a" * 64}, {"protected_transcripts": ["BAD"]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.trainer(**options)
        with self.assertRaises(ValueError):
            FoundationTrainer(self.plan, seed=True, config=self.config)
        with self.assertRaises(ValueError):
            FoundationTrainer(self.plan, seed=412000003, config=SequenceConfig(max_turns=8))
        trainer = self.trainer()
        with patch("experiments.foundation_training.source_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "source identity"):
                trainer.step()
        self.assertEqual(trainer.cursor, 0)
        self.assertEqual(trainer.last_report["failure_stage"], "preflight:source_identity")
        self.assertEqual(trainer.last_report["physical_optimizer_updates"], 0)

    def test_materializer_failure_reports_unknown_partial_generation_and_fresh_preflight(self):
        trainer = self.trainer()
        with patch("experiments.foundation_training._materialize_validated_bundle", side_effect=RuntimeError("partial materializer")):
            with self.assertRaisesRegex(RuntimeError, "partial materializer"):
                self.step(trainer)
        report = trainer.last_report
        self.assertIsNone(report["drawn_episode_exposures"])
        self.assertIsNone(report["drawn_microbatches"])
        self.assertEqual(report["neural_attempted_episode_exposures"], 0)
        self.assertGreaterEqual(report["step_seconds"], report["materialization_seconds"])
        self.assertGreaterEqual(report["wall_seconds"], report["step_seconds"])
        with self.assertRaises(RuntimeError):
            trainer.step()
        self.assertEqual(trainer.last_report["drawn_episode_exposures"], 0)
        self.assertEqual(trainer.last_report["materialization_seconds"], 0.)

    def test_preflight_source_and_endpoint_reports_never_reuse_a_prior_success(self):
        trainer = self.trainer(); self.step(trainer)
        self.assertEqual(trainer.last_report["physical_optimizer_updates"], 1)
        with patch("experiments.foundation_training.source_hashes", return_value={}):
            with self.assertRaises(ValueError):
                trainer.step()
        self.assertEqual(trainer.last_report["physical_optimizer_updates"], 0)
        self.assertEqual(trainer.last_report["completed_microbatch_episode_exposures"], 0)
        # Source-only endpoint sentinel: no completed learning state is claimed.
        trainer._evidence["cursor"] = len(self.plan["schedules"]["curriculum"])
        with self.assertRaises(StopIteration):
            trainer.step()
        self.assertTrue(trainer.last_report["complete"])
        self.assertEqual(trainer.last_report["physical_optimizer_updates"], 0)
        self.assertEqual(trainer.last_report["drawn_episode_exposures"], 0)

    def test_interrupt_during_restore_swap_poison_requires_another_restore(self):
        trainer = self.trainer(); payload = trainer.snapshot()
        assign = FoundationTrainer.__setattr__
        def interrupt(instance, name, value):
            if instance is trainer and name == "_evidence":
                raise KeyboardInterrupt("during final restore swap")
            return assign(instance, name, value)
        with patch.object(FoundationTrainer, "__setattr__", interrupt):
            with self.assertRaisesRegex(KeyboardInterrupt, "restore swap"):
                trainer.restore(payload)
        with self.assertRaisesRegex(RuntimeError, "restoration"):
            trainer.snapshot()
        with self.assertRaisesRegex(RuntimeError, "restoration"):
            trainer.step()
        trainer.restore(payload)
        self.equal(trainer.snapshot(), payload)

    def test_post_optimizer_sync_failure_keeps_completion_unknown(self):
        trainer = self.trainer()
        optimize, calls = trainer.optimizer.step, 0
        def completed_cpu_optimizer(*args, **kwargs):
            result = optimize(*args, **kwargs)
            type(self).physical_updates += 1  # Known by this CPU injection only.
            return result
        def synchronize():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected asynchronous optimizer error")
        with patch.object(trainer.optimizer, "step", side_effect=completed_cpu_optimizer), \
             patch.object(trainer, "_sync", side_effect=synchronize):
            with self.assertRaisesRegex(RuntimeError, "asynchronous optimizer"):
                self.step(trainer)
        self.assertEqual(calls, 3)  # preflight, failed completion, guarded cleanup
        self.assertIsNone(trainer.last_report["physical_optimizer_updates"])
        self.assertTrue(trainer.last_report["failure_sync_succeeded"])
        self.assertEqual(trainer.last_report["retained_optimizer_updates"], 0)
        with self.assertRaises(RuntimeError):
            trainer.snapshot()

    def test_failure_sync_time_is_inclusive_without_replacing_original_exception(self):
        trainer = self.trainer()
        clock, calls = [0.], 0
        def materialize(*args):
            clock[0] += 2.
            raise ValueError("original materialization failure")
        def synchronize():
            nonlocal calls
            calls += 1
            if calls == 2:
                clock[0] += 3.
                raise RuntimeError("secondary synchronization failure")
        with patch("experiments.foundation_training.time.monotonic", side_effect=lambda: clock[0]), \
             patch("experiments.foundation_training._materialize_validated_bundle", side_effect=materialize), \
             patch.object(trainer, "_sync", side_effect=synchronize):
            with self.assertRaisesRegex(ValueError, "original materialization"):
                self.step(trainer)
        report = trainer.last_report
        self.assertEqual(report["materialization_seconds"], 2.)
        self.assertEqual(report["failure_sync_seconds"], 3.)
        self.assertEqual(report["step_seconds"], 5.)
        self.assertEqual(report["wall_seconds"], 5.)
        self.assertFalse(report["failure_sync_succeeded"])
        self.assertFalse(report["step_time_includes_queued_device_work"])
        self.assertIn("secondary synchronization", report["failure_sync_error"])
        self.assertEqual(report["physical_optimizer_updates"], 0)


if __name__ == "__main__":
    unittest.main()
