"""Bounded CPU preparation fixtures; never constructs or executes a learner.

One reused tiny fixture definition: 66 admitted bundles, micro2, repaired naming
exclusions, one real index. Successful paths cover all six depths/three lengths.
Excluding index/admission setup, the fixed ceiling is 139 returned canonical
bundles (834 episodes), 409 pack calls with one deliberate failure, and zero
models/forwards/backwards/optimizer updates. Blocked/failing producer operations
are explicitly recorded separately. This source is not a test-run receipt.
"""
from contextlib import ExitStack
import copy
from dataclasses import replace
import pickle
import threading
import time
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_prepared_bundles as preparation
from experiments import foundation_training as canonical
from experiments.foundation_admission import repair_plan
from experiments.foundation_curriculum import FAMILIES, validate_pair
from experiments.foundation_plan import build_plan, materialize_pair
from experiments.foundation_plan_index import AuthenticatedPlanIndex
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig


def equal_tree(test, left, right):
    test.assertEqual(type(left), type(right))
    if isinstance(left, torch.Tensor):
        test.assertEqual(left.dtype, right.dtype)
        test.assertEqual(left.shape, right.shape)
        test.assertEqual(left.layout, right.layout)
        test.assertEqual(left.stride(), right.stride())
        test.assertEqual(left.storage_offset(), right.storage_offset())
        test.assertEqual(left.requires_grad, right.requires_grad)
        test.assertEqual(left.device.type, "cpu")
        test.assertEqual(right.device.type, "cpu")
        test.assertTrue(torch.equal(left, right))
    else:
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            equal_tree(test, left[key], right[key])


class PreparedBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        setup_started, setup_cpu_started = time.monotonic(), time.thread_time()
        cls.config = SequenceConfig(width=8, layers=1, heads=2, feedforward=16, max_turns=12)
        base = build_plan(seed=831908001, stage_updates=10, final_updates=6, micro_batch_size=2,
                          rehearsal_every=2, ordering_seed=831908002)
        cls.history = sorted(transcript_digest(row) for row in materialize_pair(base, 0, "color", 0))
        cls.plan, cls.admission = repair_plan(base, cls.history)
        cls.index = AuthenticatedPlanIndex(cls.plan, admission_protected_transcripts=cls.history,
            protected_transcripts=cls.history, admission_receipt=cls.admission)
        cls.work = dict(index_constructions=1, index_construction=cls.index.construction,
            setup_protection_pairs_generated=1, setup_protection_episodes_generated=2,
            initial_repair_counts=copy.deepcopy(cls.admission["counts"]),
            setup_wall_seconds=time.monotonic()-setup_started, setup_thread_cpu_seconds=time.thread_time()-setup_cpu_started,
            setup_scope="Initial protection pair and first admission repair are separate from the index's admission reconstruction and admitted-plan scan. Canonical validation may internally regenerate/check pairs; helper-return counts below are not a count of every internal generator/cache call.",
            canonical_materialize_calls=0, returned_canonical_bundles=0, returned_canonical_episodes=0,
            packing_calls=0, completed_packing_calls=0, packed_episode_passes=0, owner_reports=[],
            model_attempts=0, forward_attempts=0, backward_attempts=0, optimizer_attempts=0)
        cls.lock, cls.stack = threading.Lock(), ExitStack()
        materialize, pack = canonical._materialize_validated_bundle, canonical.pack_composition_episodes
        def tracked_materialize(*args, **kwargs):
            with cls.lock:
                cls.work["canonical_materialize_calls"] += 1
            result = materialize(*args, **kwargs)
            with cls.lock:
                cls.work["returned_canonical_bundles"] += 1
                cls.work["returned_canonical_episodes"] += sum(map(len, result.values()))
            return result
        def tracked_pack(*args, **kwargs):
            with cls.lock:
                cls.work["packing_calls"] += 1
            result = pack(*args, **kwargs)
            with cls.lock:
                cls.work["completed_packing_calls"] += 1
                cls.work["packed_episode_passes"] += len(args[0])
            return result
        def forbidden(counter):
            def fail(*args, **kwargs):
                cls.work[counter] += 1
                raise AssertionError("no learner work authorized in preparation fixture")
            return fail
        cls.stack.enter_context(patch.object(canonical, "_materialize_validated_bundle", tracked_materialize))
        cls.stack.enter_context(patch.object(canonical, "pack_composition_episodes", tracked_pack))
        cls.stack.enter_context(patch.object(canonical, "build_sequence_student", forbidden("model_attempts")))
        cls.stack.enter_context(patch.object(torch.nn.Module, "_call_impl", forbidden("forward_attempts")))
        cls.stack.enter_context(patch.object(torch.Tensor, "backward", forbidden("backward_attempts")))
        cls.stack.enter_context(patch.object(torch.optim.AdamW, "step", forbidden("optimizer_attempts")))

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()
        if cls.work["returned_canonical_bundles"] > 139 or cls.work["packing_calls"] > 409:
            raise AssertionError("declared CPU fixture preparation ceiling exceeded")
        if any(cls.work[key] for key in ("model_attempts", "forward_attempts", "backward_attempts", "optimizer_attempts")):
            raise AssertionError("unexpected learner work")

    def owner(self, *, stop_cursor=1, **overrides):
        options = dict(plan_index=self.index, config=self.config,
            admission_protected_transcripts=self.history, protected_transcripts=self.history,
            admission_receipt=self.admission, stop_cursor=stop_cursor, max_seconds=300.)
        options.update(overrides)
        return preparation.PreparedBundleOwner(self.plan, **options)

    def take(self, owner, cursor=0, **overrides):
        options = dict(order="curriculum", plan_index=self.index, config=self.config,
                       protected_transcripts=self.history, timeout_seconds=30.)
        options.update(overrides)
        return owner.take(cursor, **options)

    def finish(self, owner):
        report = owner.close(join_seconds=10.)
        self.assertFalse(report["worker_alive"])
        self.assertTrue(report["accounting_final"])
        self.assertLessEqual(report["peak_resident_bundles"], 2)
        self.work["owner_reports"].append(report)
        return report

    def test_all_cells_exact_cpu_tensors_and_evidence_in_declared_order(self):
        schedule = self.plan["schedules"]["curriculum"]
        owner = self.owner(stop_cursor=len(schedule))
        seen = set()
        try:
            for cursor, bundle_id in enumerate(schedule):
                item = self.take(owner, cursor)
                rows = canonical._materialize_validated_bundle(self.plan, bundle_id)
                evidence = canonical._rows_evidence(self.plan, bundle_id, rows, set(self.history))
                self.assertEqual(item.evidence, evidence)
                self.assertEqual(item.identity["bundle_id"], bundle_id)
                self.assertEqual(item.identity["cursor"], cursor)
                batches = item.batches
                self.assertEqual(tuple(batches), tuple(FAMILIES))
                for family in FAMILIES:
                    reference = canonical.pack_composition_episodes(rows[family], device="cpu", training=True,
                        pair_validator=validate_pair, max_turns=self.config.max_turns,
                        max_input_bytes=self.config.max_input_bytes, max_context_tokens=self.config.max_positions,
                        max_reply_bytes=self.config.max_output_bytes)
                    equal_tree(self, batches[family], reference)
                seen.add((evidence["depth"], evidence["turns"]))
                owner.validate(item)
                owner.release(item)
            with self.assertRaises(StopIteration):
                self.take(owner, len(schedule))
        finally:
            report = self.finish(owner)
        self.assertEqual(seen, {(depth, turns) for depth in range(6) for turns in (8, 10, 12)})
        self.assertEqual(report["counts"]["prepared_bundles"], 66)
        self.assertEqual(report["counts"]["released_leases"], 66)
        self.assertEqual(report["counts"]["packed_episodes"], 396)
        self.assertEqual(report["counts"]["discarded_prepared_bundles"], 0)
        self.assertEqual(report["producer_runtime"], owner.declaration["cpu_runtime"])
        self.assertEqual(report["producer_runtime"]["cpu_threads"], 1)
        self.assertGreater(report["attempted_preparation_thread_cpu_seconds"], 0.)

    def test_detached_inputs_metadata_and_modified_tensor_refusal(self):
        caller_plan, caller_history = copy.deepcopy(self.plan), list(self.history)
        owner = preparation.PreparedBundleOwner(caller_plan, plan_index=self.index, config=self.config,
            admission_protected_transcripts=caller_history, protected_transcripts=caller_history,
            admission_receipt=self.admission, stop_cursor=1, max_seconds=300.)
        caller_plan["schedules"]["curriculum"].reverse()
        caller_history.clear()
        try:
            item = self.take(owner)
            value = item.identity
            value["cursor"] = 999
            self.assertEqual(item.identity["cursor"], 0)
            with self.assertRaises(TypeError):
                pickle.dumps(item)
            with self.assertRaises(TypeError):
                pickle.dumps(owner)
            batches = item.batches
            batches[FAMILIES[0]]["inputs"]["token_ids"][0, 0] += 1
            with self.assertRaisesRegex(ValueError, "modified"):
                owner.validate(item)
            owner.release(item)  # Storage cleanup still works after refusal.
        finally:
            report = self.finish(owner)
        self.assertEqual(report["counts"]["consumer_rejections"], 1)
        self.assertEqual(report["counts"]["released_leases"], 1)

    def test_consumer_bindings_and_empty_authorized_window(self):
        owner = self.owner(stop_cursor=0)
        try:
            with self.assertRaises(StopIteration):
                self.take(owner)
        finally:
            self.assertEqual(self.finish(owner)["counts"]["submitted_bundles"], 0)
        cases = [dict(cursor=1), dict(order="mixed"), dict(plan_index=object()),
                 dict(config=replace(self.config, max_output_bytes=33)), dict(protected_transcripts=[])]
        for changes in cases:
            owner = self.owner(stop_cursor=0)
            try:
                with self.assertRaisesRegex(ValueError, "consumer cursor/order/index/config/protection"):
                    self.take(owner, **changes)
            finally:
                self.assertEqual(self.finish(owner)["counts"]["submitted_bundles"], 0)
        with self.assertRaisesRegex(ValueError, "another admitted"):
            self.owner(stop_cursor=0, protected_transcripts=[*self.history, "f"*64])

    def test_source_drift_refuses_delivery(self):
        owner = self.owner(stop_cursor=0)
        changed = {**owner.declaration["source_sha256"], "new_source.py": "f"*64}
        try:
            with patch.object(preparation, "source_hashes", return_value=changed):
                with self.assertRaisesRegex(ValueError, "source or CPU runtime"):
                    self.take(owner)
        finally:
            report = self.finish(owner)
        self.assertEqual(report["counts"]["submitted_bundles"], 0)
        self.assertEqual(report["counts"]["consumer_rejections"], 1)

    def test_second_outstanding_and_released_lease_refused(self):
        owner = self.owner()
        try:
            item = self.take(owner)
            with self.assertRaisesRegex(RuntimeError, "second outstanding"):
                self.take(owner)
            owner.release(item)
            with self.assertRaisesRegex(ValueError, "released"):
                owner.validate(item)
        finally:
            self.finish(owner)

    def test_held_lease_refuses_fresh_use_after_deadline(self):
        owner = self.owner()
        try:
            item = self.take(owner)
            owner._worker.join(10.)
            self.assertFalse(owner._worker.is_alive())
            with patch.object(preparation.time, "monotonic", return_value=owner._deadline+1):
                with self.assertRaisesRegex(RuntimeError, "fresh lease use"):
                    owner.validate(item)
            owner.release(item)
        finally:
            self.finish(owner)

    def test_bounded_close_retains_active_incomplete_work(self):
        entered, leave = threading.Event(), threading.Event()
        materialize = canonical._materialize_validated_bundle
        def blocked(*args):
            entered.set()
            if not leave.wait(10.):
                raise TimeoutError("fixture gate was not released")
            return materialize(*args)
        with patch.object(canonical, "_materialize_validated_bundle", blocked):
            owner = self.owner()
            try:
                self.assertTrue(entered.wait(10.))
                report = owner.close(join_seconds=0.)
                self.assertTrue(report["worker_alive"])
                self.assertTrue(report["active_work_incomplete"])
                self.assertFalse(report["accounting_final"])
                self.assertEqual(report["counts"]["materialized_bundles"], 0)
            finally:
                leave.set()
                report = self.finish(owner)
        self.assertEqual(report["counts"]["submitted_bundles"], 1)
        self.assertEqual(report["counts"]["materialized_bundles"], 1)
        self.assertEqual(report["counts"]["interrupted_preparations"], 1)
        self.assertEqual(report["counts"]["prepared_bundles"], 0)

    def test_wait_timeout_stops_submission_without_retry(self):
        entered, leave = threading.Event(), threading.Event()
        materialize = canonical._materialize_validated_bundle
        def blocked(*args):
            entered.set()
            if not leave.wait(10.):
                raise TimeoutError("fixture gate was not released")
            return materialize(*args)
        with patch.object(canonical, "_materialize_validated_bundle", blocked):
            owner = self.owner()
            try:
                self.assertTrue(entered.wait(10.))
                with self.assertRaises(TimeoutError):
                    self.take(owner, timeout_seconds=.001)
                self.assertTrue(owner.report()["stopping"])
            finally:
                leave.set()
                report = self.finish(owner)
        self.assertEqual(report["counts"]["submitted_bundles"], 1)
        self.assertEqual(report["counts"]["delivered_leases"], 0)
        self.assertEqual(report["counts"]["materialized_bundles"], 1)

    def test_pack_exception_preserves_partial_preparation_counts(self):
        def broken(*args, **kwargs):
            with self.lock:
                self.work["packing_calls"] += 1
            raise RuntimeError("deliberate CPU pack failure")
        with patch.object(canonical, "pack_composition_episodes", broken):
            owner = self.owner()
            try:
                with self.assertRaisesRegex(RuntimeError, "pack failure"):
                    self.take(owner)
            finally:
                report = self.finish(owner)
        self.assertEqual(report["counts"]["failed_preparations"], 1)
        self.assertEqual(report["counts"]["materialized_episodes"], 6)
        self.assertEqual(report["counts"]["evidence_validated_bundles"], 1)
        self.assertEqual(report["counts"]["packed_family_batches"], 0)
        self.assertEqual(report["phases"]["packing"]["attempted"], 1)
        self.assertEqual(report["phases"]["packing"]["failed"], 1)
        self.assertTrue(report["unreturned_operation_work_may_be_unknown"])
        self.assertFalse(report["accounting_complete"])

    def test_held_lease_refuses_fresh_use_after_next_producer_failure(self):
        entered, leave = threading.Event(), threading.Event()
        materialize = canonical._materialize_validated_bundle
        second_id = self.plan["schedules"]["curriculum"][1]
        def fail_next(plan, bundle_id):
            if bundle_id == second_id:
                entered.set()
                if not leave.wait(10.):
                    raise TimeoutError("fixture gate was not released")
                raise RuntimeError("deliberate next bundle failure")
            return materialize(plan, bundle_id)
        with patch.object(canonical, "_materialize_validated_bundle", fail_next):
            owner = self.owner(stop_cursor=2)
            try:
                item = self.take(owner)
                self.assertTrue(entered.wait(10.))
                leave.set()
                owner._worker.join(10.)
                self.assertFalse(owner._worker.is_alive())
                with self.assertRaisesRegex(RuntimeError, "fresh lease use"):
                    item.batches
                owner.release(item)
            finally:
                leave.set()
                report = self.finish(owner)
        self.assertEqual(report["counts"]["submitted_bundles"], 2)
        self.assertEqual(report["counts"]["prepared_bundles"], 1)
        self.assertEqual(report["counts"]["released_leases"], 1)
        self.assertEqual(report["counts"]["failed_preparations"], 1)


if __name__ == "__main__":
    unittest.main()
