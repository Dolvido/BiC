"""Two tiny CPU learners, three updates, nine forwards/backwards, 18 episodes.

Reuses one existing complete-pair fixture per family. Nine canonical parent
reconstructions authenticate these fixtures; no distinct parent draw is added.
No CUDA, teacher, historical learner loading or evaluation inference.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.learning_student import _cpu_copy
from experiments import foundation_layout_training as old
from experiments.foundation_layout_prepared import PreparedLayoutOwner, evidence_sha256
from experiments.shared_state_student import build_shared_state_student
from experiments.shared_state_targets import pack_state_targets
from experiments.shared_state_training import SharedStateKernel
from experiments.sequence_student import SequenceConfig

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/"runs/foundation-layout-curriculum-validation-local/attempt-001/fixtures.json"
FIXTURE_SHA = "9cb18c2032ca5492347d482432daab77877450f11ec6f667fcfe14b214fe5c0c"
CONFIG = SequenceConfig(width=16, layers=1, heads=2, feedforward=32, max_turns=12)
WORK = dict(model_constructions=0, forward_attempts=0, forwards=0, forward_episodes=0,
    backwards=0, optimizer_attempts=0, optimizer_returns=0, canonical_generations=0,
    state_target_calls=0, state_target_episodes=0, distinct_parent_draws=0)
REPORTS = []
OWNERS = []


def same(test, left, right):
    test.assertIs(type(left), type(right))
    if isinstance(left, torch.Tensor):
        test.assertEqual((left.dtype, left.shape, left.stride(), left.storage_offset()),
                         (right.dtype, right.shape, right.stride(), right.storage_offset()))
        test.assertTrue(torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                    right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left: same(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right): same(test, a, b)
    else:
        test.assertEqual(left, right)


class SharedStateTrainingTests(unittest.TestCase):
    def test_zero_weight_exact_update_then_positive_state_and_refusal(self):
        threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        raw = FIXTURE.read_bytes(); self.assertEqual(hashlib.sha256(raw).hexdigest(), FIXTURE_SHA)
        rows = {tuple(item["key"]): item["pair"] for item in json.loads(raw)["layout_pairs"]}
        bundle = dict(schema=old.BUNDLE_SCHEMA, bundle_id=0, layout="original",
            families={family: deepcopy(rows[(family, 0, 8, "original", 0)]) for family in old.FAMILIES})
        generate, backward, optimizer_step = old.curriculum.foundation.generate_pair, torch.Tensor.backward, torch.optim.AdamW.step
        def generation(*args, **kwargs):
            WORK["canonical_generations"] += 1
            return generate(*args, **kwargs)
        def back(tensor, *args, **kwargs):
            WORK["backwards"] += 1
            return backward(tensor, *args, **kwargs)
        def optimize(optimizer, *args, **kwargs):
            WORK["optimizer_attempts"] += 1
            result = optimizer_step(optimizer, *args, **kwargs)
            WORK["optimizer_returns"] += 1
            return result
        def forward_before(module, args, kwargs):
            WORK["forward_attempts"] += 1
            self.assertFalse(args)
            self.assertEqual(set(kwargs), {"token_ids", "valid_mask", "lengths", "eos_positions", "decoder_input_ids"})
        def forward_after(module, args, kwargs, output):
            WORK["forwards"] += 1; WORK["forward_episodes"] += len(kwargs["token_ids"])
        with patch.object(old.curriculum.foundation, "generate_pair", side_effect=generation), \
             patch.object(torch.Tensor, "backward", back), patch.object(torch.optim.AdamW, "step", optimize):
            work = old.curriculum.WorkLedger()
            expected = old._bundle_evidence(bundle, cursor=0, layout="original", micro_batch_size=2, work=work)
            labels = {}
            for family in old.FAMILIES:
                labels[family] = pack_state_targets(bundle["families"][family])
                WORK["state_target_calls"] += 1; WORK["state_target_episodes"] += 2
            original_labels = _cpu_copy(labels)
            reference = old.FoundationLayoutTrainer(seed=852305001, config=CONFIG, learning_rate=.003,
                micro_batch_size=2, layout="original", objective_id=old.OBJECTIVE_ID, device="cpu")
            WORK["model_constructions"] += 1
            model = build_shared_state_student(852305001, device="cpu", config=CONFIG)
            WORK["model_constructions"] += 1
            for key, value in reference.model.state_dict().items(): same(self, value, model.state_dict()[key])
            initial_aux = {name: value.clone() for name, value in model.state_dict().items()
                           if name not in reference.model.state_dict()}
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
            handles = []
            for current in (reference.model, model):
                handles.append(current.register_forward_pre_hook(forward_before, with_kwargs=True))
                handles.append(current.register_forward_hook(forward_after, with_kwargs=True))
            self.addCleanup(lambda: [handle.remove() for handle in handles])
            def kernel(weight, evidence=None):
                return SharedStateKernel(model, optimizer, auxiliary_weight=weight, evidence=evidence,
                    config=CONFIG, layout="original", micro_batch_size=2, objective_id=old.OBJECTIVE_ID)
            def token(cursor):
                owner = PreparedLayoutOwner(config=CONFIG, layout="original", micro_batch_size=2); OWNERS.append(owner)
                actual, identity = deepcopy(bundle), deepcopy(expected)
                actual["bundle_id"] = identity["bundle_id"] = cursor
                return owner.prepare(actual, expected_evidence=identity, expected_evidence_sha256=evidence_sha256(identity))
            control = kernel(0.)
            try: reference_report = reference.step(bundle)
            finally: REPORTS.append(deepcopy(reference.last_report))
            with patch.object(model, "state_logits", side_effect=AssertionError("zero weight must skip auxiliary graph")):
                try: control_report = control.step(token(0), state_targets=labels)
                finally: REPORTS.append(deepcopy(control.last_report))
            for key, value in reference.model.state_dict().items(): same(self, value, model.state_dict()[key])
            for key, value in initial_aux.items(): same(self, value, model.state_dict()[key])
            for (name, left), (_, right) in zip(reference.model.named_parameters(), model.named_parameters()):
                same(self, reference.optimizer.state[left], optimizer.state[right])
            for key, value in reference.optimizer.param_groups[0].items():
                if key != "params": same(self, value, optimizer.param_groups[0][key])
            auxiliary_parameters = [parameter for name, parameter in model.named_parameters()
                                    if name not in dict(reference.model.named_parameters())]
            self.assertTrue(auxiliary_parameters)
            self.assertTrue(all(parameter.grad is None and parameter not in optimizer.state for parameter in auxiliary_parameters))
            same(self, reference._evidence, control.evidence)
            for name in ("loss", "action_loss", "reply_loss", "observation_language_loss"):
                self.assertEqual(reference_report[name], control_report[name])
            self.assertEqual(control_report["completed_state_readouts"], 0)
            positive = kernel(.3, control.evidence)
            try: report = positive.step(token(1), state_targets=labels)
            finally: REPORTS.append(deepcopy(positive.last_report))
            self.assertEqual(report["completed_state_readouts"], 3)
            self.assertEqual(report["completed_state_objectives"], 3)
            self.assertGreater(report["state_loss"], 0.)
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                and optimizer.state[parameter]["step"].item() == 1 for parameter in auxiliary_parameters))
            self.assertTrue(any(not torch.equal(value, model.state_dict()[key]) for key, value in initial_aux.items()))
            same(self, labels, original_labels)
            saved = positive.snapshot(); same(self, saved["weights"], _cpu_copy(model.state_dict()))
            same(self, saved["optimizer"], _cpu_copy(optimizer.state_dict()))
            # Malformed supervision fails before any forward, consumes no update,
            # and leaves the existing kernel permanently poisoned.
            bad = kernel(.3, positive.evidence); unused = token(2)
            wrong = _cpu_copy(labels); wrong["color"] = wrong["color"][:, :, :-1]
            before = _cpu_copy(model.state_dict()), _cpu_copy(optimizer.state_dict())
            with self.assertRaises(ValueError): bad.step(unused, state_targets=wrong)
            REPORTS.append(deepcopy(bad.last_report))
            with self.assertRaises(RuntimeError): bad.step(unused, state_targets=labels)
            REPORTS.append(deepcopy(bad.last_report))
            same(self, before, (_cpu_copy(model.state_dict()), _cpu_copy(optimizer.state_dict())))
            for owner in OWNERS: owner.close()
        self.assertEqual(WORK["model_constructions"], 2)
        self.assertEqual(WORK["forwards"], 9); self.assertEqual(WORK["forward_attempts"], 9)
        self.assertEqual(WORK["backwards"], 9); self.assertEqual(WORK["forward_episodes"], 18)
        self.assertEqual(WORK["optimizer_returns"], 3); self.assertEqual(WORK["optimizer_attempts"], 3)
        self.assertEqual(WORK["canonical_generations"], 9)


if __name__ == "__main__":
    unittest.main()
