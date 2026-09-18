"""Two tiny CPU trajectories plus non-neural boundary failures; receipt required.

Declared real work: 2 model constructions, 4 updates, 12 forwards/backwards,
24 episode exposures. Existing fixture parents are reconstructed 18 times
(6 expected-evidence checks + 12 original-trainer checks); no distinct draws.
One injected throwing forward is an additional attempted call, not real neural
execution. No GPU, teacher, new data preparation, or production models.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from brain_in_computer.learning_student import _cpu_copy

from experiments import foundation_layout_training as baseline
from experiments import foundation_layout_prepared as prepared
from experiments.foundation_layout_kernel import PreparedLayoutKernel
from experiments.sequence_student import SequenceConfig, build_sequence_student

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/"runs/foundation-layout-curriculum-validation-local/attempt-001/fixtures.json"
FIXTURE_SHA256 = "9cb18c2032ca5492347d482432daab77877450f11ec6f667fcfe14b214fe5c0c"
CONFIG = SequenceConfig(width=16, layers=1, heads=2, feedforward=32, max_positions=1024,
    max_turns=12, max_input_bytes=128, max_output_bytes=32)
TEST_WORK = dict(model_construction_attempts=0, model_constructions=0, optimizer_attempts=0,
    optimizer_returns=0, attempted_forwards=0, completed_forwards=0, attempted_backwards=0,
    completed_backwards=0, completed_forward_episodes=0, canonical_generate_calls=0,
    canonical_generate_returns=0, canonical_rows_returned=0, distinct_parent_draws=0,
    expected_evidence_checks=0, malformed_preparations=0, rejected_tokens=0,
    mocked_forward_failures=0, exact_tensor_batch_comparisons=0)
PREPARATION_WORK = []
PREPARATION_OWNERS = []
STEP_REPORTS = []
REFERENCE_VALIDATION_WORK = []


def _model(*args, **kwargs):
    TEST_WORK["model_construction_attempts"] += 1
    result = build_sequence_student(*args, **kwargs)
    TEST_WORK["model_constructions"] += 1
    return result


def _record(report):
    STEP_REPORTS.append(deepcopy(report))
    for key in ("attempted_forwards", "completed_forwards", "attempted_backwards", "completed_backwards", "completed_forward_episodes"):
        TEST_WORK[key] += report[key]
    TEST_WORK["optimizer_attempts"] += report["optimizer_attempts"]
    TEST_WORK["optimizer_returns"] += report["optimizer_returns"]


class PreparedLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = FIXTURE.read_bytes()
        if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256: raise ValueError("fixture pin differs")
        rows = {tuple(item["key"]): item["pair"] for item in json.loads(raw)["layout_pairs"]}
        original = baseline.curriculum.foundation.generate_pair
        def generated(*args, **kwargs):
            TEST_WORK["canonical_generate_calls"] += 1
            result = original(*args, **kwargs)
            TEST_WORK["canonical_generate_returns"] += 1
            TEST_WORK["canonical_rows_returned"] += len(result)
            return result
        guard = patch.object(baseline.curriculum.foundation, "generate_pair", side_effect=generated)
        guard.start(); cls.addClassCleanup(guard.stop)
        cls.bundles, cls.expected = [], []
        for cursor, (depth, turns) in enumerate(((0, 8), (5, 12))):
            bundle = dict(schema=baseline.BUNDLE_SCHEMA, bundle_id=cursor, layout="original",
                families={family: deepcopy(rows[(family, depth, turns, "original", 0)]) for family in baseline.FAMILIES})
            work = baseline.curriculum.WorkLedger()
            cls.expected.append(baseline._bundle_evidence(bundle, cursor=cursor, layout="original", micro_batch_size=2, work=work))
            TEST_WORK["expected_evidence_checks"] += 1
            REFERENCE_VALIDATION_WORK.append(work.report()); cls.bundles.append(bundle)

    def same(self, a, b):
        self.assertIs(type(a), type(b))
        if isinstance(a, torch.Tensor):
            self.assertEqual((a.dtype, a.shape, a.stride(), a.storage_offset()), (b.dtype, b.shape, b.stride(), b.storage_offset()))
            self.assertTrue(torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                       b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a: self.same(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b): self.same(x, y)
        else: self.assertEqual(a, b)

    def owner(self):
        owner = prepared.PreparedLayoutOwner(config=CONFIG, layout="original", micro_batch_size=2)
        PREPARATION_OWNERS.append(owner)
        return owner

    def kernel(self, model, optimizer, evidence=None):
        return PreparedLayoutKernel(model, optimizer, config=CONFIG, layout="original", micro_batch_size=2,
            objective_id=baseline.OBJECTIVE_ID, evidence=evidence)

    def token(self, owner, index=0, cursor=None):
        bundle, evidence = deepcopy(self.bundles[index]), deepcopy(self.expected[index])
        if cursor is not None: bundle["bundle_id"] = evidence["bundle_id"] = cursor
        return owner.prepare(bundle, expected_evidence=evidence, expected_evidence_sha256=prepared.evidence_sha256(evidence))

    def test_exact_two_step_state_tensors_and_failed_boundaries(self):
        with patch.object(baseline, "build_sequence_student", side_effect=_model):
            reference = baseline.FoundationLayoutTrainer(seed=852105001, config=CONFIG, learning_rate=.003,
                micro_batch_size=2, layout="original", objective_id=baseline.OBJECTIVE_ID, device="cpu")
        model = _model(852105001, config=CONFIG, device="cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        kernel, owner = self.kernel(model, optimizer), self.owner()
        original_pack = baseline.pack_composition_episodes
        for index in range(2):
            token = self.token(owner, index)
            captured = []
            def packing(*args, **kwargs):
                value = original_pack(*args, **kwargs); captured.append(deepcopy(value)); return value
            with patch.object(baseline, "pack_composition_episodes", side_effect=packing):
                try: expected = reference.step(self.bundles[index])
                finally:
                    _record(reference.last_report)
                    REFERENCE_VALIDATION_WORK.append(reference.last_report["validation_work"])
            for family, packed in zip(baseline.FAMILIES, captured):
                self.same(token._batches[family], packed); TEST_WORK["exact_tensor_batch_comparisons"] += 1
            before_generations = TEST_WORK["canonical_generate_calls"]
            try: actual = kernel.step(token)
            finally: _record(kernel.last_report)
            self.assertEqual(TEST_WORK["canonical_generate_calls"], before_generations)
            self.same(reference.model.state_dict(), model.state_dict())
            self.same(reference.optimizer.state_dict(), optimizer.state_dict())
            self.same(reference._evidence, kernel.evidence)
            self.same(expected["microbatches"], actual["microbatches"])
            for name in ("loss", "action_loss", "reply_loss", "observation_language_loss"):
                self.assertEqual(expected[name], actual[name])
        PREPARATION_WORK.append(owner.report())
        saved = kernel.snapshot()
        self.same(saved["weights"], _cpu_copy(reference.model.state_dict()))
        self.same(saved["optimizer"], _cpu_copy(reference.optimizer.state_dict()))
        saved["weights"]["action_head.bias"].add_(1)
        next(iter(saved["optimizer"]["state"].values()))["exp_avg"].add_(1)
        self.same(reference.model.state_dict(), model.state_dict())
        self.same(reference.optimizer.state_dict(), optimizer.state_dict())

        # Every rejection reuses the two existing models, without a new update.
        for kind in ("tensor", "metadata", "wrong_cursor", "foreign_owner", "reused", "source"):
            bad_owner = self.owner(); bad = self.token(bad_owner, cursor=2)
            current = self.kernel(model, optimizer, kernel.evidence)
            if kind == "tensor": bad._batches["color"]["inputs"]["token_ids"][0, 0] += 1
            if kind == "metadata": bad._evidence = bad._evidence.replace('"turns":8', '"turns":10')
            if kind == "wrong_cursor": current._evidence["cursor"] = 3
            if kind == "foreign_owner": bad._owner = self.owner()
            if kind == "reused": bad = token
            before = deepcopy(optimizer.state_dict())
            with patch.object(model, "forward", side_effect=AssertionError("no forward permitted")) as forward:
                with patch.object(prepared, "source_hashes", return_value={}) if kind == "source" else patch.object(current, "_sync", wraps=current._sync):
                    with self.assertRaises((ValueError, RuntimeError)): current.step(bad)
                _record(current.last_report); self.assertEqual(forward.call_count, 0)
            self.assertTrue(current.failed); self.assertEqual(current.last_report["retained_updates"], 0)
            with self.assertRaises(RuntimeError): current.step(bad)
            _record(current.last_report)
            self.same(before, optimizer.state_dict()); TEST_WORK["rejected_tokens"] += 1
            PREPARATION_WORK.append(bad_owner.report()); bad_owner.close()

        failing_owner = self.owner(); bad = self.token(failing_owner, cursor=2)
        failing = self.kernel(model, optimizer, kernel.evidence)
        with patch.object(model, "forward", side_effect=RuntimeError("injected before neural execution")):
            with self.assertRaises(RuntimeError): failing.step(bad)
        _record(failing.last_report); TEST_WORK["mocked_forward_failures"] += 1
        self.assertTrue(failing.failed)
        self.assertEqual(failing.last_report["attempted_forwards"], 1)
        self.assertEqual(failing.last_report["completed_forwards"], 0)
        self.assertEqual(failing.last_report["physical_optimizer_updates"], 0)
        self.same(reference.model.state_dict(), model.state_dict())
        self.same(reference.optimizer.state_dict(), optimizer.state_dict())
        PREPARATION_WORK.append(failing_owner.report())

    def test_wrong_rows_or_expected_pin_fail_before_packing(self):
        for kind in ("row_text", "pin"):
            owner = self.owner(); bundle, expected = deepcopy(self.bundles[0]), deepcopy(self.expected[0])
            pin = prepared.evidence_sha256(expected)
            if kind == "row_text": bundle["families"]["color"][0]["turns"][0]["text"] += " changed"
            else: pin = "0"*64
            with patch.object(prepared, "pack_composition_episodes", side_effect=AssertionError("packing forbidden")) as packer:
                with self.assertRaises(ValueError): owner.prepare(bundle, expected_evidence=expected, expected_evidence_sha256=pin)
                self.assertEqual(packer.call_count, 0)
            TEST_WORK["malformed_preparations"] += 1
            PREPARATION_WORK.append(owner.report())
