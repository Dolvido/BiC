"""Prospective bounded CPU equivalence proof; no historical/formal inputs.

One 66-bundle micro2 admitted plan and one actual index are shared. Both fixed
objectives run synchronous66 versus prepared33/strict-restore/prepared33:264
updates. A single prescribed restored-prefix step adds one update after a
partial-second-family interruption. Two queued snapshot-discard bundles and one
after-transfer deadline-refusal bundle add CPU preparation only. No other test
may learn. This source is not an execution receipt; the external caller
authenticates sources and work limits.
"""
from contextlib import ExitStack
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_training as canonical
from experiments import foundation_objective_training as synchronous
from experiments import foundation_prefetched_training as prefetched
from experiments import foundation_plan_index as indexing
from experiments.foundation_admission import repair_plan
from experiments.foundation_curriculum import FAMILIES
from experiments.foundation_plan import build_plan, materialize_pair
from experiments.foundation_plan_index import AuthenticatedPlanIndex
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig, SequenceStudent


PLAN_SEED, ORDER_SEED, MODEL_SEED = 831909001, 831909002, 831909799
OBJECTIVES = ("baseline", "balanced_reply")
ARITHMETIC_FIELDS = ("weights", "optimizer", "cursor", "evidence")
SNAPSHOT_FIELDS = {"schema", "recipe", "weights", "optimizer", "cursor", "evidence", "timing"}
BOUNDS = dict(model_constructions_attempted=11, optimizer_attempts=265, optimizer_updates=265,
    trainer_attempted_family_forwards=797, trainer_attempted_episode_forwards=1594,
    physical_forward_calls=796, physical_forward_episodes=1592,
    backward_attempts=796, backward_completed=796,
    returned_canonical_bundles=269, returned_canonical_episodes=1614,
    packing_calls=807, completed_packing_calls=807, packed_episode_passes=1614)


def same(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return (left.dtype == right.dtype and left.shape == right.shape and left.layout == right.layout
                and left.stride() == right.stride() and left.storage_offset() == right.storage_offset()
                and left.device == right.device and left.requires_grad == right.requires_grad and torch.equal(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def arithmetic(payload):
    """Only recipe/version/source and timing differ by design; never tensors."""
    return {name: payload[name] for name in ARITHMETIC_FIELDS}


def learner_state(trainer):
    """Use snapshot's canonical CPU copies without its deliberate queue drain."""
    return dict(weights=canonical._cpu_copy(trainer.model.state_dict()),
                optimizer=canonical._cpu_copy(trainer.optimizer.state_dict()),
                cursor=copy.deepcopy(trainer.cursor), evidence=copy.deepcopy(trainer._evidence))


class PrefetchedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.started, cls.cpu_started = time.monotonic(), time.process_time()
        cls.config = SequenceConfig(width=8, layers=1, heads=2, feedforward=16, max_turns=12)
        setup = time.monotonic()
        base = build_plan(seed=PLAN_SEED, stage_updates=10, final_updates=6, micro_batch_size=2,
                          rehearsal_every=2, ordering_seed=ORDER_SEED)
        cls.history = sorted(transcript_digest(row) for row in materialize_pair(base, 0, "color", 0))
        cls.plan, cls.admission = repair_plan(base, cls.history)
        cls.index = AuthenticatedPlanIndex(cls.plan, admission_protected_transcripts=cls.history,
            protected_transcripts=cls.history, admission_receipt=cls.admission)
        if len(cls.plan["bundles"]) != 66 or cls.index.identity["micro_batch_size"] != 2:
            raise AssertionError("one exact66-bundle micro2 fixture required")
        cls.work = dict(schema="bic-prefetched-training-cpu-proof-v1", config=asdict(cls.config),
            plan_seed=PLAN_SEED, ordering_seed=ORDER_SEED, model_seed=MODEL_SEED,
            setup_protection_pairs_generated=1, setup_protection_episodes_generated=2,
            initial_repair_counts=copy.deepcopy(cls.admission["counts"]), index_constructions=1,
            index_construction=cls.index.construction, setup_wall_seconds=time.monotonic()-setup,
            setup_scope="Protection pair and initial repair precede the index's distinct admission reconstruction and admitted-plan scan. Returned helper counts exclude nested canonical regeneration/checks.",
            model_constructions_attempted=0, model_constructions_completed=0,
            optimizer_attempts=0, optimizer_updates=0, physical_forward_calls=0, physical_forward_episodes=0,
            backward_attempts=0, backward_completed=0, canonical_materialize_calls=0,
            returned_canonical_bundles=0, returned_canonical_episodes=0,
            packing_calls=0, completed_packing_calls=0, packed_episode_passes=0,
            step_reports=[], preparation_reports=[], exact_comparisons=[], bounds=dict(BOUNDS),
            historical_or_formal_inputs=False, historical_checkpoint_loads=0, gpu_tensor_work=0)
        cls.lock, cls.stack = threading.Lock(), ExitStack()
        construct, forward = SequenceStudent.__init__, SequenceStudent.forward
        backward, optimize = torch.Tensor.backward, torch.optim.AdamW.step
        materialize, pack = canonical._materialize_validated_bundle, canonical.pack_composition_episodes

        def built(instance, *args, **kwargs):
            cls.work["model_constructions_attempted"] += 1
            result = construct(instance, *args, **kwargs)
            cls.work["model_constructions_completed"] += 1
            return result

        def forwarded(instance, *args, **kwargs):
            cls.work["physical_forward_calls"] += 1
            cls.work["physical_forward_episodes"] += int(kwargs["token_ids"].shape[0])
            return forward(instance, *args, **kwargs)

        def backpropagated(tensor, *args, **kwargs):
            cls.work["backward_attempts"] += 1
            result = backward(tensor, *args, **kwargs)
            cls.work["backward_completed"] += 1
            return result

        def optimized(instance, *args, **kwargs):
            cls.work["optimizer_attempts"] += 1
            result = optimize(instance, *args, **kwargs)
            cls.work["optimizer_updates"] += 1
            return result

        def materialized(*args, **kwargs):
            with cls.lock:
                cls.work["canonical_materialize_calls"] += 1
            rows = materialize(*args, **kwargs)
            with cls.lock:
                cls.work["returned_canonical_bundles"] += 1
                cls.work["returned_canonical_episodes"] += sum(map(len, rows.values()))
            return rows

        def packed(*args, **kwargs):
            with cls.lock:
                cls.work["packing_calls"] += 1
            batch = pack(*args, **kwargs)
            with cls.lock:
                cls.work["completed_packing_calls"] += 1
                cls.work["packed_episode_passes"] += len(args[0])
            return batch

        for owner, name, function in ((SequenceStudent, "__init__", built),
                (SequenceStudent, "forward", forwarded), (torch.Tensor, "backward", backpropagated),
                (torch.optim.AdamW, "step", optimized), (canonical, "_materialize_validated_bundle", materialized),
                (canonical, "pack_composition_episodes", packed)):
            cls.stack.enter_context(patch.object(owner, name, function))
        cls.final_trainers, cls.prefixes, cls.references, cls.reference_reports = {}, {}, {}, {}

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()
        reports = cls.work["step_reports"]
        cls.work["trainer_attempted_family_forwards"] = sum(row["report"]["neural_attempted_microbatches"] for row in reports)
        cls.work["trainer_attempted_episode_forwards"] = sum(row["report"]["neural_attempted_episode_exposures"] for row in reports)
        cls.work["wall_seconds"] = time.monotonic()-cls.started
        cls.work["cpu_seconds"] = time.process_time()-cls.cpu_started
        cls.work["within_declared_bounds"] = all(cls.work[name] <= bound for name, bound in BOUNDS.items())
        cls.work["comparison_scope"] = "All weights, full AdamW state, cursor, canonical evidence and four per-step losses compare exactly. Trainer recipe/schema/source and timing are intentionally excluded from arithmetic equivalence and checked separately. Queue/storage counters never count as learner exposure."
        print("PREFETCHED_TRAINING_CPU_WORK=" + json.dumps({
            **{name: cls.work[name] for name in BOUNDS},
            "within_declared_bounds": cls.work["within_declared_bounds"],
            "wall_seconds": cls.work["wall_seconds"], "cpu_seconds": cls.work["cpu_seconds"]}, sort_keys=True))
        path = os.environ.get("BIC_PREFETCHED_TRAINING_ACCOUNTING")
        if path:
            with Path(path).open("x", encoding="utf8") as stream:
                json.dump(cls.work, stream, sort_keys=True, indent=2)
        if not cls.work["within_declared_bounds"]:
            raise AssertionError("prefetched fixture exceeded declared physical work bound")

    def trainer(self, objective_id="baseline", *, prepared=True, **overrides):
        options = dict(objective_id=objective_id, seed=MODEL_SEED, config=self.config, order="curriculum",
            admission_protected_transcripts=self.history, protected_transcripts=self.history,
            admission_receipt=self.admission, plan_index=self.index, device="cpu")
        options.update(overrides)
        owner = prefetched.PrefetchedObjectiveFoundationTrainer if prepared else synchronous.ObjectiveFoundationTrainer
        return owner(self.plan, **options)

    def step(self, trainer, label):
        try:
            return trainer.step()
        finally:
            self.work["step_reports"].append(dict(label=label, report=copy.deepcopy(trainer.last_report)))

    def close(self, trainer, label):
        invocation = trainer.close_preparation(join_seconds=10.)
        if invocation is None:
            return None
        report = invocation["owner"]
        self.assertFalse(report["worker_alive"])
        self.assertTrue(report["accounting_final"])
        self.assertLessEqual(report["peak_resident_bundles"], 2)
        self.work["preparation_reports"].append(dict(label=label, report=copy.deepcopy(invocation)))
        return report

    def compare(self, actual, reference, report, reference_report, *, objective_id, cursor, label):
        self.assertTrue(same(arithmetic(actual), arithmetic(reference)),
                        f"Canonical learner arithmetic differs: {objective_id}, cursor {cursor}, {label}")
        self.assertEqual(actual["cursor"], cursor)
        for field in synchronous.LOSS_FIELDS:
            self.assertEqual(report[field], reference_report[field])
        evidence_fields = ("family", "depth", "turns", "rows_sha256", "recipes_sha256", "exposures", *synchronous.LOSS_FIELDS)
        self.assertEqual([{key: row[key] for key in evidence_fields} for row in report["microbatches"]],
                         [{key: row[key] for key in evidence_fields} for row in reference_report["microbatches"]])
        self.work["exact_comparisons"].append(dict(objective_id=objective_id, cursor=cursor, label=label,
            weights=True, full_adamw=True, canonical_evidence=True, losses=True))

    def test_01_both_objectives_all_cells_exact_split_resume(self):
        for objective_id in OBJECTIVES:
            reference = self.trainer(objective_id, prepared=False)
            states, reports = [learner_state(reference)], [None]
            for cursor in range(1, 67):
                reports.append(self.step(reference, f"{objective_id}/synchronous/{cursor}"))
                states.append(learner_state(reference))
            trainer = self.trainer(objective_id)
            initial = trainer.snapshot()
            self.assertEqual(initial["schema"], prefetched.SCHEMA)
            self.assertEqual(set(initial), SNAPSHOT_FIELDS)
            self.assertEqual(initial["recipe"]["objective"], reference.recipe["objective"])
            self.assertTrue(same(arithmetic(initial), arithmetic(states[0])),
                            "Seeded initial arithmetic differs after canonical CPU capture")
            seen = set()
            trainer.start_preparation(stop_cursor=33, max_seconds=300.)
            try:
                for cursor in range(1, 34):
                    report = self.step(trainer, f"{objective_id}/prefetched/{cursor}")
                    saved = learner_state(trainer)
                    self.compare(saved, states[cursor], report, reports[cursor],
                        objective_id=objective_id, cursor=cursor, label="first_half")
                    for row in report["microbatches"]:
                        seen.add((row["depth"], row["turns"]))
                prefix = trainer.snapshot()
                self.assertEqual(set(prefix), SNAPSHOT_FIELDS)
                self.assertIsNone(trainer._preparation)
            finally:
                first_owner = self.close(trainer, objective_id+"/first_half")
            self.assertEqual(first_owner["counts"]["released_leases"], 33)
            self.assertEqual(first_owner["counts"]["discarded_prepared_bundles"], 0)
            resumed = self.trainer(objective_id, payload=copy.deepcopy(prefix))
            self.assertTrue(same(prefix, resumed.snapshot()))
            self.assertIsNone(resumed.preparation_report)
            resumed.start_preparation(stop_cursor=66, max_seconds=300.)
            try:
                for cursor in range(34, 67):
                    report = self.step(resumed, f"{objective_id}/resumed/{cursor}")
                    self.compare(learner_state(resumed), states[cursor], report, reports[cursor],
                        objective_id=objective_id, cursor=cursor, label="second_half")
                    for row in report["microbatches"]:
                        seen.add((row["depth"], row["turns"]))
            finally:
                second_owner = self.close(resumed, objective_id+"/second_half")
            self.assertEqual(second_owner["counts"]["released_leases"], 33)
            self.assertEqual(second_owner["counts"]["discarded_prepared_bundles"], 0)
            self.assertEqual(seen, {(depth, turns) for depth in range(6) for turns in (8, 10, 12)})
            self.final_trainers[objective_id] = resumed
            self.prefixes[objective_id] = prefix
            self.references[objective_id], self.reference_reports[objective_id] = states[34], reports[34]
        self.assertEqual(self.work["optimizer_updates"], 264)

    def test_02_wrong_snapshot_and_admission_rejected_without_state_swap(self):
        trainer = self.final_trainers["baseline"]
        saved = trainer.snapshot()
        model, optimizer = trainer.model, trainer.optimizer
        bad = []
        wrong = copy.deepcopy(saved); wrong["schema"] = synchronous.SCHEMA; bad.append(wrong)
        bad.append(self.final_trainers["balanced_reply"].snapshot())
        wrong = copy.deepcopy(saved); wrong["recipe"]["plan_index_identity"]["bundle_count"] += 1; bad.append(wrong)
        wrong = copy.deepcopy(saved); wrong["recipe"]["plan_index_identity"]["expanded_protection_sha256"] = "f"*64; bad.append(wrong)
        for payload in bad:
            constructions = self.work["model_constructions_attempted"]
            with self.assertRaises(ValueError):
                trainer.restore(payload)
            self.assertEqual(self.work["model_constructions_attempted"], constructions)
            self.assertIs(trainer.model, model)
            self.assertIs(trainer.optimizer, optimizer)
            self.assertTrue(same(saved, trainer.snapshot()))
        for overrides in (dict(plan_index=object()), dict(admission_receipt={}),
                          dict(protected_transcripts=[*self.history, "f"*64])):
            constructions = self.work["model_constructions_attempted"]
            with self.assertRaises(ValueError):
                self.trainer(**overrides)
            self.assertEqual(self.work["model_constructions_attempted"], constructions)
        with patch.object(prefetched, "source_hashes", return_value={}):
            constructions = self.work["model_constructions_attempted"]
            with self.assertRaises(ValueError):
                trainer.restore(copy.deepcopy(saved))
            self.assertEqual(self.work["model_constructions_attempted"], constructions)
        self.assertIs(trainer.model, model)
        self.assertIs(trainer.optimizer, optimizer)
        self.assertTrue(same(saved, trainer.snapshot()))

    def test_03_partial_failure_poison_explicit_restore_then_one_update(self):
        saved = self.prefixes["baseline"]
        trainer = self.trainer(payload=copy.deepcopy(saved))
        type(self).failure_trainer = trainer
        trainer.start_preparation(stop_cursor=34, max_seconds=300.)
        forward, calls = trainer.model.forward, 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("prescribed second-family fixture interruption")
            return forward(*args, **kwargs)
        try:
            with patch.object(trainer.model, "forward", side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
                self.step(trainer, "failure/second_family")
            self.assertEqual(trainer.last_report["physical_optimizer_updates"], 0)
            self.assertEqual(trainer.last_report["retained_optimizer_updates"], 0)
            self.assertEqual(trainer.last_report["neural_attempted_microbatches"], 2)
            self.assertEqual(trainer.last_report["completed_microbatches"], 1)
            with self.assertRaises(RuntimeError):
                trainer.snapshot()
            before = self.work["physical_forward_calls"]
            with self.assertRaises(RuntimeError):
                self.step(trainer, "failure/no_hidden_continuation")
            self.assertEqual(self.work["physical_forward_calls"], before)
        finally:
            self.close(trainer, "failure/interrupted_owner")
        with patch.object(canonical, "_materialize_validated_bundle", side_effect=AssertionError("unexpected restore generation")), \
                patch.object(indexing, "_materialize_validated_bundle", side_effect=AssertionError("unexpected index rescan")):
            trainer.restore(copy.deepcopy(saved))
        self.assertTrue(same(saved, trainer.snapshot()))
        self.assertIsNone(trainer._preparation)
        trainer.start_preparation(stop_cursor=34, max_seconds=300.)
        try:
            report = self.step(trainer, "failure/explicit_restored_update34")
            self.compare(learner_state(trainer), self.references["baseline"], report, self.reference_reports["baseline"],
                objective_id="baseline", cursor=34, label="explicit_failure_restore")
        finally:
            self.close(trainer, "failure/restored_owner")
        self.assertEqual(self.work["optimizer_updates"], 265)

    def test_04_missing_preparation_source_drift_and_expired_allowance_do_no_learning(self):
        trainer = self.failure_trainer
        before = learner_state(trainer)
        forwards, updates = self.work["physical_forward_calls"], self.work["optimizer_updates"]
        with self.assertRaises(ValueError):
            trainer.start_preparation(stop_cursor=trainer.cursor-1, max_seconds=300.)
        self.assertTrue(same(before, learner_state(trainer)))
        with self.assertRaises(RuntimeError):
            self.step(trainer, "refusal/missing_live_preparation")
        self.assertTrue(same(before, learner_state(trainer)))
        self.assertEqual(self.work["physical_forward_calls"], forwards)
        with patch.object(prefetched, "source_hashes", return_value={}):
            with self.assertRaises(ValueError):
                self.step(trainer, "refusal/source_identity")
        self.assertTrue(same(before, learner_state(trainer)))
        # Empty authorized window starts no generation. Only the deadline is
        # moved to the past; no live source or global clock is modified.
        trainer.start_preparation(stop_cursor=trainer.cursor, max_seconds=300.)
        trainer._preparation_deadline = time.monotonic()-1.
        try:
            with self.assertRaises(TimeoutError):
                self.step(trainer, "refusal/expired_before_first_forward")
            self.assertTrue(same(before, learner_state(trainer)))
            self.assertEqual(self.work["physical_forward_calls"], forwards)
            self.assertEqual(self.work["optimizer_updates"], updates)
            self.assertEqual(trainer.last_report["neural_attempted_microbatches"], 0)
            self.assertEqual(trainer.last_report["physical_optimizer_updates"], 0)
        finally:
            report = self.close(trainer, "refusal/empty_expired_window")
        self.assertEqual(report["counts"]["submitted_bundles"], 0)

    def test_05_live_queue_snapshot_discards_two_items_without_changing_checkpoint(self):
        trainer = self.failure_trainer
        saved = trainer.snapshot()
        self.assertEqual(saved["cursor"], 34)
        forwards, updates = self.work["physical_forward_calls"], self.work["optimizer_updates"]
        trainer.start_preparation(stop_cursor=36, max_seconds=300.)
        owner = trainer._preparation
        try:
            # The exact two-item window fills without a consumer; the producer
            # exits naturally at its finite stop, leaving both queued items live.
            owner._worker.join(10.)
            self.assertFalse(owner._worker.is_alive())
            queued = owner.report()
            self.assertEqual(queued["pending"], 2)
            self.assertEqual(queued["counts"]["prepared_bundles"], 2)
            self.assertFalse(queued["closed"])
            self.assertTrue(same(saved, trainer.snapshot()))
            self.assertIsNone(trainer._preparation)
            self.assertEqual(set(saved), SNAPSHOT_FIELDS)
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                owner.take(34, order="curriculum", plan_index=self.index, config=self.config,
                           protected_transcripts=self.history, timeout_seconds=10.)
            self.work["stale_owner_after_snapshot"] = copy.deepcopy(owner.report())
        finally:
            report = self.close(trainer, "snapshot/discard_two_ready_bundles")
        self.assertEqual(report["counts"]["discarded_prepared_bundles"], 2)
        self.assertEqual(report["counts"]["delivered_leases"], 0)
        self.assertEqual(self.work["physical_forward_calls"], forwards)
        self.assertEqual(self.work["optimizer_updates"], updates)
        self.assertTrue(same(saved, trainer.snapshot()))

    def test_06_deadline_after_first_transfer_poisoned_without_first_forward(self):
        trainer = self.failure_trainer
        before = learner_state(trainer)
        forwards, updates = self.work["physical_forward_calls"], self.work["optimizer_updates"]
        trainer.start_preparation(stop_cursor=35, max_seconds=300.)
        original, calls = trainer._before_first_forward, 0
        def expire_after_transfer():
            nonlocal calls
            calls += 1
            if calls == 2:
                trainer._preparation_deadline = time.monotonic()-1.
            return original()
        try:
            with patch.object(trainer, "_before_first_forward", side_effect=expire_after_transfer), \
                    self.assertRaises(TimeoutError):
                self.step(trainer, "refusal/deadline_after_first_transfer")
            self.assertEqual(calls, 2)
            self.assertEqual(trainer.last_report["failure_stage"], "transfer:"+FAMILIES[0])
            self.assertFalse(trainer.last_report["update_started"])
            self.assertEqual(trainer.last_report["neural_attempted_microbatches"], 0)
            self.assertEqual(trainer.last_report["completed_microbatches"], 0)
            self.assertEqual(trainer.last_report["physical_optimizer_updates"], 0)
            self.assertEqual(trainer.last_report["retained_optimizer_updates"], 0)
            self.assertTrue(same(before, learner_state(trainer)))
            self.assertTrue(trainer._failed)
            self.assertIsNone(trainer._preparation)
            with self.assertRaises(RuntimeError):
                trainer.snapshot()
            with self.assertRaises(RuntimeError):
                self.step(trainer, "refusal/after_transfer_no_hidden_continuation")
            self.assertEqual(self.work["physical_forward_calls"], forwards)
            self.assertEqual(self.work["optimizer_updates"], updates)
        finally:
            report = self.close(trainer, "refusal/after_transfer_deadline_owner")
        self.assertEqual(report["counts"]["prepared_bundles"], 1)
        self.assertEqual(report["counts"]["packed_family_batches"], 3)
        self.assertEqual(report["counts"]["released_leases"], 1)


if __name__ == "__main__":
    unittest.main()
