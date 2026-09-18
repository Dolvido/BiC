"""Tiny CPU integration plus explicitly non-neural finite controller fixtures."""
import copy
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_curriculum as curriculum
from experiments import foundation_plan as planning
from experiments import foundation_loop as module
from experiments.foundation_admission import repair_plan
from experiments.foundation_evaluation import FoundationBank
from experiments.foundation_evidence import json_digest
from experiments.foundation_loop import FoundationLoop
from experiments.foundation_provider import FoundationPracticeProvider
from experiments.foundation_training import FoundationTrainer
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig


def without_time(value):
    if isinstance(value, dict):
        return {key: without_time(item) for key, item in value.items()
                if key != "timing" and (not isinstance(key, str) or "seconds" not in key)}
    if isinstance(value, list): return [without_time(item) for item in value]
    return value


class FiniteProvider(FoundationPracticeProvider):
    """Three-update control-flow double. No model forward, optimizer or data."""
    def __init__(self):
        self._lock = threading.RLock()
        self._identity = {"plan_sha256": "a"*64, "order": "curriculum", "fixture": "non-neural",
                          "curriculum_version": curriculum.VERSION}
        self._identity_hash = json_digest(self._identity)
        self._specs = {role: {"fixture": {"identity": {"episodes": 2}}}
                       for role in ("development", "retention")}
        self.cursor, self.pending, self.last_report = 0, None, None
        self.weight = torch.zeros(1)
        self.evaluations = 0
        self._failed = False

    def _check(self):
        if self._failed: raise RuntimeError("fixture poisoned")

    def snapshot(self):
        self._check()
        return {"identity": self.identity, "learner": {"cursor": self.cursor,
            "weights": {"weight": self.weight.clone()}, "optimizer": {"fixture_step": self.cursor},
            "evidence": {"exposures": {family: {"episodes": 2*self.cursor} for family in curriculum.FAMILIES}}},
            "pending": copy.deepcopy(self.pending)}

    def restore(self, payload):
        saved = copy.deepcopy(payload)
        if saved["identity"] != self.identity: raise ValueError("fixture identity")
        cursor = saved["learner"]["cursor"]
        if type(cursor) is not int or not 0 <= cursor <= 3: raise ValueError("fixture cursor")
        if not torch.equal(saved["learner"]["weights"]["weight"], torch.tensor([float(cursor)])):
            raise ValueError("fixture weight")
        pending = saved["pending"]
        if pending is not None and pending["consumed_updates"] != cursor-pending["request"]["start_cursor"]:
            raise ValueError("fixture pending")
        self.cursor, self.pending = cursor, pending
        self.weight = saved["learner"]["weights"]["weight"]
        self._failed = False
        self.last_report = None
        return self

    def status(self):
        self._check()
        return {"updates": self.cursor, "total_updates": 3, "pending": copy.deepcopy(self.pending)}

    def evaluate(self, role, *, names=None, **kwargs):
        self._check(); self.evaluations += 1
        producer = module._producer(self.snapshot())
        value = 1 if self.cursor % 2 == 0 else 0
        progress = {field: {"rate": (1-value if field == "unsupported_ask" else value),
                            "total": 1, "correct": value} for field in module.FIELDS}
        return {"role": role, "control": "normal", **producer,
                "per_bank": {"fixture": {"episodes": 2}}, "progress": {"fixture": progress}}

    def validate_evaluation(self, response, *, expected_weights_sha256=None, expected_updates=None):
        if (response["weights_sha256"] != expected_weights_sha256 or response["updates"] != expected_updates
                or response["role"] not in self._specs or set(response["per_bank"]) != {"fixture"}):
            raise ValueError("fixture response identity")
        value = 1 if expected_updates % 2 == 0 else 0
        for field in module.FIELDS:
            if response["progress"]["fixture"][field]["rate"] != (1-value if field == "unsupported_ask" else value):
                raise ValueError("fixture response rate")
        return True

    def practice(self, request=None, *, max_updates=16, deadline=None):
        self._check()
        if self.pending is None:
            if request["start_cursor"] != self.cursor: raise ValueError("stale fixture request")
            self.pending = {"request": copy.deepcopy(request), "request_sha256": json_digest(request), "consumed_updates": 0}
        elif request is not None: raise ValueError("replacing fixture request")
        steps = min(max_updates, self.pending["request"]["stop_cursor"]-self.cursor)
        if deadline is not None and deadline <= time.monotonic(): steps = 0
        self.cursor += steps; self.weight.add_(steps)
        self.pending["consumed_updates"] += steps
        if self.cursor == self.pending["request"]["stop_cursor"]: self.pending = None
        self.last_report = {"completed_updates": steps, "physical_optimizer_updates": steps,
            "failed_step_attempts": 0, "unknown_optimizer_attempts": 0, "accounting_uncertain": False,
            "drawn_episode_exposures": steps*6, "neural_attempted_episode_exposures": steps*6,
            "completed_microbatch_episode_exposures": steps*6, "fixture": "simulated counts, no neural work"}
        return copy.deepcopy(self.last_report)


class FoundationLoopControlTests(unittest.TestCase):
    def make(self, **options):
        return FoundationLoop(FiniteProvider(), window_updates=2, chunk_updates=1, **options)

    def test_finite_final_evaluation_with_zero_update_allowance_and_no_recycle(self):
        loop = self.make(history_limit=1)
        first = loop.run(max_updates=3, max_evaluations=4)
        self.assertEqual(first["completed_updates"], 3)
        self.assertEqual(loop.status["phase"], "evaluate")
        self.assertEqual(loop.status["evaluation_cursor"], 0)
        partial = loop.run(max_updates=0, max_evaluations=1)
        self.assertEqual(partial["completed_updates"], 0)
        resumed = FoundationLoop(FiniteProvider(), window_updates=2, chunk_updates=1,
                                 history_limit=1, payload=loop.snapshot())
        final = resumed.run(max_updates=0, max_evaluations=1)
        self.assertEqual(final["status"], "complete")
        saved = resumed.snapshot()
        self.assertEqual(saved["state"]["history_dropped"], 2)
        self.assertEqual(saved["state"]["work"]["evaluation_banks"], 6)
        self.assertEqual(saved["state"]["work"]["physical_optimizer_updates"], 3)
        self.assertEqual(len(saved["state"]["history"][-1]["alarms"]), 8)
        ref = saved["state"]["references"]["retention:fixture"]["paired_action"]
        self.assertEqual(ref["producer"]["updates"], 0)
        self.assertEqual(resumed.run(max_updates=3)["completed_updates"], 0)

    def test_loaded_state_rejects_tampered_prefix_queue_work_and_retention(self):
        loop = self.make(history_limit=1)
        loop.run(max_updates=3)
        saved = loop.snapshot()
        for kind in ("alarm", "reference_boundary", "reference_rate", "work", "final_evaluation", "source", "identity"):
            changed = copy.deepcopy(saved)
            if kind == "alarm": changed["state"]["history"][0]["alarms"].clear()
            elif kind == "reference_boundary":
                changed["state"]["reference_anchor"]["retention:fixture"]["paired_action"]["producer"]["updates"] = 1
            elif kind == "reference_rate":
                changed["state"]["references"]["retention:fixture"]["paired_action"]["response"]["progress"]["fixture"]["paired_action"]["rate"] = .5
            elif kind == "work": changed["state"]["work"]["physical_optimizer_updates"] = True
            elif kind == "final_evaluation": changed["state"]["evaluations_completed"] = 2
            elif kind == "source": changed["source_sha256"] = {}
            else: changed["provider_identity"]["curriculum_version"] = "legacy"
            fresh = FiniteProvider()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                FoundationLoop(fresh, window_updates=2, chunk_updates=1, history_limit=1, payload=changed)
            self.assertEqual(fresh.cursor, 0)
        partial = self.make(); partial.run(max_updates=1)
        changed = partial.snapshot(); changed["state"]["window_stop"] = 3
        with self.assertRaises(ValueError): FoundationLoop(FiniteProvider(), window_updates=2, chunk_updates=1, payload=changed)
        initial = self.make(); initial.run(max_updates=0, max_evaluations=1)
        changed = initial.snapshot(); changed["state"]["evaluation"]["producer"]["weights_sha256"] = "0"*64
        with self.assertRaises(ValueError): FoundationLoop(FiniteProvider(), window_updates=2, chunk_updates=1, payload=changed)

    def test_explicit_limits_and_fresh_preflight_report(self):
        loop = self.make(); loop.run(max_updates=1)
        for kwargs in ({}, {"max_updates": True}, {"max_updates": -1}, {"max_evaluations": True}, {"deadline": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): loop.run(**kwargs)
            self.assertEqual(loop.last_report["physical_optimizer_updates"], 0)
            self.assertEqual(loop.last_report["evaluation_banks"], 0)
        self.assertEqual(loop.run(deadline=0)["status"], "deadline")
        self.assertEqual(loop.status["updates"], 1)

    def test_single_owner_and_external_provider_or_envelope_drift(self):
        loop = self.make()
        with self.assertRaises(ValueError): FoundationLoop(loop.provider)
        loop.provider.practice({"start_cursor": 0, "stop_cursor": 2}, max_updates=1)
        with self.assertRaisesRegex(RuntimeError, "outside its loop"): loop.run(max_updates=1)
        self.assertEqual(loop.status["phase"], "reload_required")
        pending = self.make(); pending.run(max_updates=1)
        pending.provider.practice(max_updates=1)
        with self.assertRaisesRegex(RuntimeError, "outside its loop"): pending.save(Path("unused.pt"))
        loop = self.make()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"loop.pt"; loop.save(path)
            path.write_bytes(b"foreign writer")
            with self.assertRaisesRegex(RuntimeError, "externally"): loop.snapshot()
            self.assertEqual(loop.status["phase"], "reload_required")

    def test_source_guard_cache_tampering_and_incremental_commit_cost(self):
        loop = self.make()
        with patch.object(loop, "_validate", wraps=loop._validate) as validate:
            loop.run(max_updates=0)
            # Full state check once at invocation entry, not once per bank.
            self.assertEqual(validate.call_count, 1)
        response = copy.deepcopy(loop._state["history"][0]["responses"]["development:fixture"])
        producer = loop._state["history"][0]["producer"]
        with patch.object(loop.provider, "validate_evaluation", wraps=loop.provider.validate_evaluation) as verify:
            loop._response(response, "development", "fixture", producer)
            self.assertEqual(verify.call_count, 0)
            response["progress"]["fixture"]["known"]["rate"] = .5
            with self.assertRaises(ValueError): loop._response(response, "development", "fixture", producer)
            self.assertEqual(verify.call_count, 1)
        with patch.object(loop.provider, "_check", side_effect=RuntimeError("source drift")):
            with self.assertRaisesRegex(RuntimeError, "source drift"): loop.run(max_updates=0)
        self.assertEqual(loop.last_report["evaluation_banks"], 0)

    def test_deadline_during_prefix_installation_is_committed(self):
        loop = self.make(); loop.run(max_updates=0)
        actual = loop.provider.practice
        now = [0.]
        def expires(request=None, **kwargs):
            result = actual(request, max_updates=1, deadline=0.)
            now[0] = 100.
            return result
        with patch.object(loop.provider, "practice", side_effect=expires), patch.object(
                module.time, "monotonic", side_effect=lambda: now[0]):
            # A zero-step deadline response is committed before next outer check.
            loop.run(max_updates=1, deadline=1.)
        saved = loop.snapshot()
        self.assertIsNotNone(saved["provider"]["pending"])
        self.assertEqual(saved["provider"]["pending"]["consumed_updates"], 0)

    def test_publication_failure_after_replace_recognizes_retained_state(self):
        loop = self.make(); loop.run(max_updates=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"loop.pt"; loop.save(path)
            replace = os.replace
            def published_then_fail(a, b): replace(a, b); raise OSError("after replace")
            with patch.object(module.os, "replace", side_effect=published_then_fail):
                with self.assertRaisesRegex(OSError, "after replace"): loop.run(max_updates=1)
            self.assertEqual(loop.last_report["retained_updates"], 1)
            self.assertEqual(loop.last_report["discarded_completed_updates"], 0)
            loaded = FoundationLoop.load(path, FiniteProvider())
            self.assertTrue(module._same_tree(loop.snapshot(), loaded.snapshot()))

    def test_ambiguous_publication_requires_reload_and_preserves_unknown_counts(self):
        loop = self.make(); loop.run(max_updates=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"loop.pt"; loop.save(path)
            def ambiguous(a, b): Path(b).write_bytes(b"unknown publication"); raise OSError("ambiguous")
            with patch.object(module.os, "replace", side_effect=ambiguous):
                with self.assertRaises(OSError): loop.run(max_updates=1)
            self.assertEqual(loop.last_report["physical_optimizer_updates"], 1)
            self.assertIsNone(loop.last_report["retained_updates"])
            self.assertIsNone(loop.last_report["discarded_optimizer_updates"])
            with self.assertRaisesRegex(RuntimeError, "reload"): loop.snapshot()

    def test_failed_scoring_keeps_completed_exposures_separate_from_unknown_work(self):
        loop = self.make()
        with patch.object(loop.provider, "evaluate", side_effect=RuntimeError("scoring failed")):
            with self.assertRaises(RuntimeError): loop.run(max_updates=0)
        self.assertEqual(loop.last_report["evaluation_attempts"], 1)
        self.assertEqual(loop.last_report["incomplete_evaluation_attempts"], 1)
        self.assertTrue(loop.last_report["unknown_evaluation_episode_exposures"])
        self.assertEqual(loop.last_report["evaluation_episode_exposures"], 0)
        self.assertLessEqual(loop.last_report["evaluation_seconds"], loop.last_report["wall_seconds"])
        self.assertEqual(loop.snapshot()["state"]["evaluation"]["cursor"], 0)
        with tempfile.TemporaryDirectory() as directory:
            loop.save(Path(directory)/"loop.pt")
            with patch.object(module.torch, "save", side_effect=OSError("evaluation publication failed")):
                with self.assertRaises(OSError): loop.run(max_updates=0, max_evaluations=1)
            report = loop.last_report
            self.assertEqual(report["evaluation_banks"], 1)
            self.assertEqual(report["retained_evaluation_banks"], 0)
            self.assertEqual(report["discarded_evaluation_banks"], 1)
            self.assertEqual(report["evaluation_episode_exposures"], 2)
            self.assertEqual(report["durable_retained_evaluation_episode_exposures"], 0)
            self.assertEqual(report["discarded_evaluation_episode_exposures"], 2)


class FoundationLoopNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.config = SequenceConfig(width=8, layers=1, heads=2, feedforward=16, max_turns=12, max_positions=1560)
        base = planning.build_plan(seed=811000001, stage_updates=10, final_updates=6,
            micro_batch_size=2, rehearsal_every=2, ordering_seed=811000002)
        cls.history = sorted(map(transcript_digest, planning.materialize_pair(base, 0, "color", 0)))
        cls.plan, cls.receipt = repair_plan(base, cls.history)
        cls.banks = {role: {f"loop/{family}/d0/direct/t8": curriculum.generate_pair(family,
            812000001+100*role_index+index, depth=0, turns=8, split="dev")
            for index, family in enumerate(curriculum.FAMILIES)}
            for role_index, role in enumerate(("development", "retention"))}
        cls.protection = sorted(set(cls.history) | {transcript_digest(row)
            for banks in cls.banks.values() for rows in banks.values() for row in rows})
        cls.counts = dict(optimizer=0, drawn=0, neural=0, completed=0, uncertain=0, score_banks=0, score_episodes=0)
        optimizer, step, score = torch.optim.AdamW.step, FoundationTrainer.step, FoundationBank.score
        def tracked_optimizer(instance, *args, **kwargs):
            value = optimizer(instance, *args, **kwargs); cls.counts["optimizer"] += 1; return value
        def tracked_step(instance):
            try: return step(instance)
            finally:
                report = instance.last_report
                if report:
                    for key, field in (("drawn", "drawn_episode_exposures"), ("neural", "neural_attempted_episode_exposures"),
                                       ("completed", "completed_microbatch_episode_exposures")):
                        cls.counts[key] += report[field] or 0
                    cls.counts["uncertain"] += int(report["physical_optimizer_updates"] is None)
        def tracked_score(instance, *args, **kwargs):
            value = score(instance, *args, **kwargs)
            cls.counts["score_banks"] += 1; cls.counts["score_episodes"] += value["episodes"]
            return value
        cls.patches = ExitStack()
        cls.patches.enter_context(patch.object(torch.optim.AdamW, "step", tracked_optimizer))
        cls.patches.enter_context(patch.object(FoundationTrainer, "step", tracked_step))
        cls.patches.enter_context(patch.object(FoundationBank, "score", tracked_score))

    @classmethod
    def tearDownClass(cls):
        cls.patches.close(); torch.set_num_threads(cls.threads)
        print("FoundationLoop actual CPU work (control doubles excluded):", cls.counts, "No GPU or pilot inputs.")

    def provider(self):
        return FoundationPracticeProvider(self.plan, seed=8201, config=self.config, evaluation_banks=self.banks,
            admission_protected_transcripts=self.history, protected_transcripts=self.protection, admission_receipt=self.receipt)

    def loop(self): return FoundationLoop(self.provider(), window_updates=2, chunk_updates=1)

    def test_direct_provider_equivalence_and_split_one_envelope_resume(self):
        direct = self.provider()
        request = {"kind": "prescribed_prefix", "provider_sha256": direct.identity_sha256,
            "plan_sha256": direct.identity["plan_sha256"], "order": "curriculum", "start_cursor": 0, "stop_cursor": 2}
        direct.practice(request, max_updates=2)
        uninterrupted = self.loop(); uninterrupted.run(max_updates=2)
        self.assertTrue(module._same_tree(without_time(direct.snapshot()), without_time(uninterrupted.snapshot()["provider"])))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"envelope.pt"
            split = self.loop(); split.save(path)
            first = split.run(max_updates=0, max_evaluations=1)
            self.assertEqual(first["evaluation_banks"], 1)
            partial = FoundationLoop.load(path, self.provider())
            partial.run(max_updates=1)
            saved = partial.snapshot()
            self.assertEqual(saved["provider"]["pending"]["consumed_updates"], 1)
            resumed = FoundationLoop.load(path, self.provider())
            resumed.run(max_updates=1)
            actual, expected = resumed.snapshot(), uninterrupted.snapshot()
            self.assertTrue(module._same_tree(without_time(actual), without_time(expected)))
            self.assertEqual(actual["state"]["work"]["evaluation_episode_exposures"], 24)
            self.assertEqual(actual["state"]["work"]["physical_optimizer_updates"], 2)

    def test_definite_failed_publication_discards_chunk_but_accounts_real_work(self):
        loop = self.loop(); loop.run(max_updates=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"envelope.pt"; loop.save(path)
            before = loop.snapshot()
            with patch.object(module.torch, "save", side_effect=OSError("write failure")):
                with self.assertRaisesRegex(OSError, "write failure"): loop.run(max_updates=1)
            self.assertTrue(module._same_tree(before, loop.snapshot()))
            report = loop.last_report
            self.assertEqual(report["physical_optimizer_updates"], 1)
            self.assertEqual(report["neural_attempted_episode_exposures"], 6)
            self.assertEqual(report["discarded_optimizer_updates"], 1)
            self.assertEqual(report["durable_retained_updates"], 0)
            self.assertEqual(loop.run(max_updates=1)["retained_updates"], 1)

    def test_failed_partial_step_restores_owned_state_and_reports_attempted_work(self):
        loop = self.loop(); loop.run(max_updates=0)
        before = loop.snapshot(); actual = loop.provider._trainer.model.forward
        calls = 0
        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2: raise RuntimeError("second microbatch")
            return actual(*args, **kwargs)
        with patch.object(loop.provider._trainer.model, "forward", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "second microbatch"): loop.run(max_updates=1)
        self.assertTrue(module._same_tree(before, loop.snapshot()))
        report = loop.last_report
        self.assertEqual(report["failed_step_attempts"], 1)
        self.assertEqual(report["physical_optimizer_updates"], 0)
        self.assertEqual(report["drawn_episode_exposures"], 6)
        self.assertEqual(report["neural_attempted_episode_exposures"], 4)
        self.assertEqual(report["completed_microbatch_episode_exposures"], 2)
        self.assertEqual(loop.run(max_updates=1)["retained_updates"], 1)

    def test_throw_after_optimizer_keeps_physical_completion_unknown(self):
        loop = self.loop(); loop.run(max_updates=0)
        before = loop.snapshot(); actual = loop.provider._trainer.optimizer.step
        def fail(*args, **kwargs): actual(*args, **kwargs); raise RuntimeError("optimizer completion uncertain")
        with patch.object(loop.provider._trainer.optimizer, "step", side_effect=fail):
            with self.assertRaises(RuntimeError): loop.run(max_updates=1)
        self.assertTrue(module._same_tree(before, loop.snapshot()))
        self.assertIsNone(loop.last_report["physical_optimizer_updates"])
        self.assertIsNone(loop.last_report["discarded_optimizer_updates"])
        self.assertEqual(loop.last_report["unknown_optimizer_attempts"], 1)

    def test_post_practice_accounting_failure_does_not_erase_completed_work(self):
        loop = self.loop(); loop.run(max_updates=0)
        with patch.object(loop, "_account_practice", side_effect=RuntimeError("account boundary")):
            with self.assertRaises(RuntimeError): loop.run(max_updates=1)
        self.assertEqual(loop.last_report["physical_optimizer_updates"], 1)
        self.assertEqual(loop.last_report["discarded_optimizer_updates"], 1)
        self.assertEqual(loop.status["updates"], 0)


if __name__ == "__main__": unittest.main()
