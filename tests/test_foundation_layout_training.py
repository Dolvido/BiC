"""Bounded tiny CPU comparison; execute only under a separate work receipt.

Fixtures come from the completed layout validation, with no new distinct parent
draws. Planned real work is two model constructions, two optimizer updates,
six forwards/backwards and twelve episode exposures. Seven malformed cases use
mock model/optimizer constructors. Canonical validation regenerations, including
the packer's repeated admission, are independently counted below.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import torch

from experiments import foundation_layout_training as training
from experiments.composition_data import pack_composition_episodes
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.sequence_training import sequence_objective


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "runs/foundation-layout-curriculum-validation-local/attempt-001/fixtures.json"
FIXTURE_SHA256 = "9cb18c2032ca5492347d482432daab77877450f11ec6f667fcfe14b214fe5c0c"
TEST_WORK = dict(distinct_canonical_pair_draws=0, canonical_generate_pair_calls=0,
    canonical_generate_pair_returns=0, canonical_rows_returned=0, fixture_pairs_loaded=0,
    actual_model_constructions_attempted=0, actual_model_constructions_completed=0,
    mocked_model_constructions=0, mocked_optimizer_constructions=0,
    attempted_forwards=0, completed_forwards=0, attempted_backwards=0, completed_backwards=0,
    attempted_forward_episodes=0, completed_forward_episodes=0,
    actual_optimizer_attempts=0, actual_optimizer_returns=0, malformed_cases=0)
VALIDATION_WORK = training.curriculum.WorkLedger().report()
CONFIG = SequenceConfig(width=16, layers=1, heads=2, feedforward=32,
                        max_positions=1024, max_turns=12, max_input_bytes=128, max_output_bytes=32)
SEED, RATE = 918242000, .003


def _digest(value):
    image = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(image.encode("utf-8")).hexdigest()


def _count_model(*args, **kwargs):
    TEST_WORK["actual_model_constructions_attempted"] += 1
    model = build_sequence_student(*args, **kwargs)
    TEST_WORK["actual_model_constructions_completed"] += 1
    return model


def _record_validation(work):
    for name, value in work.items():
        VALIDATION_WORK[name] += value


def _record_step(report):
    for name in ("attempted_forwards", "completed_forwards", "attempted_backwards", "completed_backwards",
                 "attempted_forward_episodes", "completed_forward_episodes"):
        TEST_WORK[name] += report[name]
    TEST_WORK["actual_optimizer_attempts"] += report["optimizer_attempts"]
    TEST_WORK["actual_optimizer_returns"] += report["optimizer_returns"]
    _record_validation(report["validation_work"])


class FoundationLayoutTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        image = FIXTURE.read_bytes()
        if hashlib.sha256(image).hexdigest() != FIXTURE_SHA256:
            raise ValueError("preserved layout fixture digest differs")
        fixture = json.loads(image)
        cls.pairs = {tuple(record["key"]): record["pair"] for record in fixture["layout_pairs"]}
        TEST_WORK["fixture_pairs_loaded"] = len(cls.pairs)
        cls.bundle = {"schema": training.BUNDLE_SCHEMA, "bundle_id": 0, "layout": "varied",
            "families": {family: deepcopy(cls.pairs[(family, 0, 8, "varied", 1)]) for family in training.FAMILIES}}
        original_generate = training.curriculum.foundation.generate_pair

        def counted(*args, **kwargs):
            TEST_WORK["canonical_generate_pair_calls"] += 1
            pair = original_generate(*args, **kwargs)
            TEST_WORK["canonical_generate_pair_returns"] += 1
            TEST_WORK["canonical_rows_returned"] += len(pair)
            return pair

        cls.generation_patch = patch.object(training.curriculum.foundation, "generate_pair", side_effect=counted)
        cls.generation_patch.start()
        cls.addClassCleanup(cls.generation_patch.stop)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)

    def equal_tree(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.equal_tree(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal_tree(a, b)
        else:
            self.assertEqual(left, right)

    def trainer(self, *, micro_batch_size=2):
        return training.FoundationLayoutTrainer(seed=SEED, config=CONFIG, learning_rate=RATE,
            micro_batch_size=micro_batch_size, layout="varied", objective_id=training.OBJECTIVE_ID, device="cpu")

    def test_one_step_matches_manual_objective_clip_and_adamw_and_snapshot_is_independent(self):
        original_bundle = deepcopy(self.bundle)
        with patch.object(training, "build_sequence_student", side_effect=_count_model):
            trainer = self.trainer()
        reference = _count_model(SEED, config=CONFIG, device="cpu")
        optimizer = torch.optim.AdamW(reference.parameters(), lr=RATE)
        self.equal_tree(trainer.model.state_dict(), reference.state_dict())
        self.equal_tree(trainer.optimizer.state_dict(), optimizer.state_dict())
        try:
            report = trainer.step(self.bundle)
        finally:
            _record_step(trainer.last_report)

        work, losses_by_family = training.curriculum.WorkLedger(), []
        reference.train()
        optimizer.zero_grad(set_to_none=True)
        try:
            for family in training.FAMILIES:
                batch = pack_composition_episodes(self.bundle["families"][family], device="cpu", training=True,
                    pair_validator=lambda pair: training.curriculum.validate_pair(pair, work=work),
                    max_turns=CONFIG.max_turns, max_input_bytes=CONFIG.max_input_bytes,
                    max_context_tokens=CONFIG.max_positions, max_reply_bytes=CONFIG.max_output_bytes)
                TEST_WORK["attempted_forwards"] += 1
                TEST_WORK["attempted_forward_episodes"] += 2
                output = reference(**batch["inputs"], decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
                TEST_WORK["completed_forwards"] += 1
                TEST_WORK["completed_forward_episodes"] += 2
                losses = sequence_objective(output, batch)
                TEST_WORK["attempted_backwards"] += 1
                (losses["loss"] / 3).backward()
                TEST_WORK["completed_backwards"] += 1
                losses_by_family.append({name: float(value.detach()) for name, value in losses.items()})
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
            TEST_WORK["actual_optimizer_attempts"] += 1
            optimizer.step()
            TEST_WORK["actual_optimizer_returns"] += 1
        finally:
            _record_validation(work.report())
        self.equal_tree(trainer.model.state_dict(), reference.state_dict())
        self.equal_tree(trainer.optimizer.state_dict(), optimizer.state_dict())
        for name in losses_by_family[0]:
            self.assertEqual(report[name], sum(row[name] for row in losses_by_family) / 3)
        self.assertEqual(self.bundle, original_bundle)
        self.assertEqual(trainer.cursor, 1)
        self.assertFalse(trainer.failed)
        self.assertEqual(report["retained_updates"], 1)
        self.assertEqual(report["retained_episodes"], 6)
        self.assertEqual(report["physical_optimizer_updates"], 1)
        self.assertTrue(report["queued_device_work_synchronized"])
        for name in ("attempted_forwards", "completed_forwards", "attempted_backwards", "completed_backwards"):
            self.assertEqual(report[name], 3)
        self.assertEqual(report["validation_work"]["validate_pair_calls"], 6)
        self.assertEqual(report["validation_work"]["canonical_parent_regeneration_calls"], 6)
        self.assertEqual(report["packed_microbatches"], 3)
        self.assertEqual(report["packed_episodes"], 6)
        self.assertEqual(trainer.accounting["work"]["retained_updates"], 1)
        self.assertEqual(trainer.recipe["config"], asdict(CONFIG))
        self.assertEqual(trainer.recipe["objective"]["id"], training.OBJECTIVE_ID)
        parents = [[family, self.bundle["families"][family][0]["recipe"]["base_pair_sha256"],
                    self.bundle["families"][family][0]["recipe"]["base_ids"]] for family in training.FAMILIES]
        self.assertEqual(report["bundle_evidence"]["common_parents_sha256"], _digest(parents))

        saved = trainer.snapshot()
        self.equal_tree(saved["weights"], reference.state_dict())
        self.equal_tree(saved["optimizer"], optimizer.state_dict())
        self.assertEqual(saved["cursor"], saved["evidence"]["cursor"])
        self.assertEqual(saved["accounting"]["cost"]["snapshot_invocations"], 1)
        self.assertFalse(saved["recipe"]["resume_supported"])
        self.assertEqual(saved["evidence"]["consumed_common_parent_identity_sha256"],
            _digest([_digest([training.SCHEMA, "common-parents"]), 0, _digest(parents)]))
        for family in training.FAMILIES:
            self.assertEqual(saved["evidence"]["exposures"][family]["episodes"], 2)
            self.assertEqual(saved["evidence"]["exposures"][family]["turns"], 16)
        for value in saved["weights"].values():
            self.assertEqual(value.device.type, "cpu")
            self.assertFalse(value.requires_grad)
        saved["weights"]["action_head.bias"].add_(10)
        next(iter(saved["optimizer"]["state"].values()))["exp_avg"].add_(10)
        saved["recipe"]["config"]["width"] = 999
        saved["evidence"]["cursor"] = 999
        saved["accounting"]["work"]["retained_updates"] = 999
        self.equal_tree(trainer.model.state_dict(), reference.state_dict())
        self.equal_tree(trainer.optimizer.state_dict(), optimizer.state_dict())
        self.assertEqual(trainer.recipe["config"], asdict(CONFIG))
        self.assertEqual(trainer.cursor, 1)
        self.assertEqual(trainer.accounting["work"]["retained_updates"], 1)

    def test_malformed_bundles_fail_before_packing_or_forward_and_permanently_poison(self):
        cases = []
        bad = deepcopy(self.bundle)
        bad["bundle_id"] = 1
        cases.append(("wrong_cursor", bad, 2, 0))
        bad = deepcopy(self.bundle)
        del bad["families"][training.FAMILIES[-1]]
        cases.append(("missing_family", bad, 2, 0))
        bad = deepcopy(self.bundle)
        bad["families"][training.FAMILIES[-1]].reverse()
        cases.append(("crossed_final_pair", bad, 2, 2))
        bad = deepcopy(self.bundle)
        bad["families"] = {family: rows + deepcopy(rows) for family, rows in bad["families"].items()}
        cases.append(("duplicate_parent", bad, 4, 2))
        bad = deepcopy(self.bundle)
        bad["families"][training.FAMILIES[0]] = deepcopy(bad["families"][training.FAMILIES[1]])
        cases.append(("mixed_family", bad, 2, 1))
        bad = deepcopy(self.bundle)
        bad["families"][training.FAMILIES[1]] = deepcopy(self.pairs[(training.FAMILIES[1], 1, 8, "varied", 1)])
        cases.append(("mixed_depth", bad, 2, 3))
        bad = deepcopy(self.bundle)
        bad["families"][training.FAMILIES[0]] = deepcopy(self.pairs[(training.FAMILIES[0], 0, 8, "original", 0)])
        cases.append(("mixed_layout", bad, 2, 1))
        for name, bundle, micro, expected_regenerations in cases:
            with self.subTest(name=name):
                model, optimizer = MagicMock(), MagicMock()
                model.config = CONFIG
                model.parameters.return_value = []
                optimizer.state_dict.return_value = {"state": {}, "param_groups": [{"params": [], "lr": RATE}]}
                with patch.object(training, "build_sequence_student", return_value=model), \
                        patch.object(training.torch.optim, "AdamW", return_value=optimizer), \
                        patch.object(training, "pack_composition_episodes", side_effect=AssertionError("packing must not run")) as packer:
                    trainer = self.trainer(micro_batch_size=micro)
                    TEST_WORK["mocked_model_constructions"] += 1
                    TEST_WORK["mocked_optimizer_constructions"] += 1
                    TEST_WORK["malformed_cases"] += 1
                    calls_before = TEST_WORK["canonical_generate_pair_calls"]
                    with self.assertRaises(ValueError):
                        trainer.step(bundle)
                    _record_step(trainer.last_report)
                    self.assertEqual(TEST_WORK["canonical_generate_pair_calls"] - calls_before, expected_regenerations)
                    self.assertTrue(trainer.failed)
                    self.assertEqual(trainer.cursor, 0)
                    self.assertEqual(trainer.last_report["attempted_forwards"], 0)
                    self.assertEqual(trainer.last_report["attempted_backwards"], 0)
                    self.assertEqual(trainer.last_report["physical_optimizer_updates"], 0)
                    self.assertEqual(trainer.last_report["retained_updates"], 0)
                    with self.assertRaisesRegex(RuntimeError, "poisoned"):
                        trainer.step(self.bundle)
                    _record_step(trainer.last_report)
                    with self.assertRaisesRegex(RuntimeError, "poisoned"):
                        trainer.snapshot()
                    packer.assert_not_called()
                    model.assert_not_called()
                    model.train.assert_not_called()
                    optimizer.step.assert_not_called()
                    self.assertEqual(trainer.accounting["work"]["step_invocations"], 2)
                    self.assertEqual(trainer.accounting["work"]["retained_updates"], 0)


if __name__ == "__main__":
    unittest.main()
