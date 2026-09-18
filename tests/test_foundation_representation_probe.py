"""Synthetic cached-feature tests only; no BiC source learner or evidence loads."""
import copy
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments import foundation_representation_probe as probe


def setUpModule():
    # This test module runs in its own process; configure once before tensor work.
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def signal():
    inventory = ("synthetic/a", "synthetic/b")
    labels = torch.tensor(list(range(4)) * 48 * len(inventory), dtype=torch.long)
    cells = [cell for cell in inventory for _ in range(192)]
    kinds = ["statement" if target == 3 else "query" for target in labels.tolist()]
    coordinates = [(i // 2, i % 2, 0) for _ in inventory for i in range(192)]
    features = torch.cat((F.one_hot(labels, 4).double(), torch.full((len(labels), 1), 7.)), dim=1)
    return features, labels, dict(cells=cells, kinds=kinds, coordinates=coordinates, cell_inventory=inventory)


class RepresentationProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started = time.monotonic()
        cls.reports = []
        cls.manual_objective_evaluations = 0

    @classmethod
    def tearDownClass(cls):
        fields = ("probe_optimizer_step_attempts", "completed_probe_optimizer_steps",
                  "probe_solver_iterations", "closure_attempts", "completed_closures",
                  "objective_evaluations_started", "objective_evaluations_completed",
                  "backward_evaluations_started", "backward_evaluations_completed",
                  "final_evaluation_attempts", "final_evaluations_completed")
        value = dict(schema="bic-representation-probe-synthetic-work-v1", device="cpu", dtype="float64",
            threads=1, interop_threads=1, learner_optimizer_updates=0, source_learner_calls=0,
            formal_or_historical_evidence_loaded=False, reports=cls.reports,
            totals={field: sum(row[field] for row in cls.reports) for field in fields},
            manual_objective_forward_evaluations=cls.manual_objective_evaluations,
            wall_seconds=time.monotonic()-cls.started,
            scope="Synthetic probes and forced failure paths only; no original learner or BiC feature cache.")
        print("REPRESENTATION_PROBE_SYNTHETIC_WORK="+json.dumps(value, sort_keys=True))
        path = os.environ.get("BIC_REPRESENTATION_PROBE_ACCOUNTING")
        if path:
            with Path(path).open("x", encoding="utf8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True)

    def fit(self, *args, **kwargs):
        try:
            result = probe.fit_probe(*args, **kwargs)
        except BaseException as error:
            if hasattr(error, "probe_report"):
                self.reports.append(copy.deepcopy(error.probe_report))
            raise
        self.reports.append(result.report)
        return result

    def test_exact_weighting_and_ridge_are_not_global_turn_averages(self):
        labels = torch.tensor([0, 0, 1, 2, 3, 3, 0, 2, 2, 3], dtype=torch.long)
        cells = ("a",)*6 + ("b",)*4
        weights = probe._weights(labels, cells, ("a", "b"))
        expected = torch.tensor([1/12, 1/12, 1/6, 1/6, 1/16, 1/16,
                                 1/4, 1/8, 1/8, 1/8], dtype=torch.float64)
        torch.testing.assert_close(weights, expected, atol=1e-15, rtol=0)
        self.assertAlmostEqual(float(weights.sum()), 1.25)
        features = torch.arange(20, dtype=torch.float64).reshape(10, 2)/20
        weight = torch.arange(8, dtype=torch.float64).reshape(4, 2)/8
        bias = torch.tensor([0., .1, .2, .3], dtype=torch.float64)
        per = F.cross_entropy(features @ weight.T+bias, labels, reduction="none")
        manual = .5 * ((per[:2].mean()+per[2]+per[3])/3 + .25*per[4:6].mean()
                       +(per[6]+per[7:9].mean())/2 + .25*per[9])
        manual += .5*1e-3*(weight.square().sum()+bias.square().sum())
        self.manual_objective_evaluations += 1
        type(self).manual_objective_evaluations = self.manual_objective_evaluations
        torch.testing.assert_close(probe._objective(features, labels, weights, weight, bias), manual)

    def test_shuffle_preserves_cell_kind_counts_and_is_coordinate_bound(self):
        _, labels, metadata = signal()
        before = labels.clone()
        result = probe.shuffled_labels(labels, **metadata)
        self.assertGreater(result["changed_labels"], 0)
        for cell in metadata["cell_inventory"]:
            for kind in ("statement", "query"):
                positions = [i for i, item in enumerate(zip(metadata["cells"], metadata["kinds"])) if item == (cell, kind)]
                self.assertEqual(sorted(labels[positions].tolist()), sorted(result["labels"][positions].tolist()))
        reversed_metadata = {key: list(reversed(value)) if key != "cell_inventory" else value
                             for key, value in metadata.items()}
        reversed_result = probe.shuffled_labels(labels.flip(0), **reversed_metadata)
        self.assertEqual(result["sha256"], reversed_result["sha256"])
        torch.testing.assert_close(result["labels"], reversed_result["labels"].flip(0))
        result["labels"].fill_(3)
        torch.testing.assert_close(labels, before)

    def test_true_signal_and_shuffle_share_fit_only_statistics_and_do_not_mutate_inputs(self):
        features, labels, metadata = signal()
        features.requires_grad_()
        before, true_labels, metadata_before = features.detach().clone(), labels.clone(), copy.deepcopy(metadata)
        rng = torch.random.get_rng_state().clone()
        results = probe.fit_true_and_shuffled(features, labels, **metadata)
        self.reports.extend(result.report for result in results.values())
        first, shuffled = results["true"], results["shuffled"]
        self.assertIn(first.report["status"], ("converged", "normal_return_unconverged"))
        self.assertIn(shuffled.report["status"], ("converged", "normal_return_unconverged"))
        self.assertLess(first.report["final_objective"], .1)
        self.assertGreater(shuffled.report["final_objective"], .5)
        self.assertTrue(bool(first.predict(features).eq(labels).all()))
        for key in first.normalization:
            torch.testing.assert_close(first.normalization[key], shuffled.normalization[key])
        self.assertFalse(bool(first.normalization["active"][-1]))
        norm_before = first.normalization
        torch.testing.assert_close(norm_before["mean"], before.mean(dim=0))
        torch.testing.assert_close(norm_before["scale"][:-1], before.std(dim=0, correction=0)[:-1])
        evaluation = before.clone()
        evaluation[:, -1] = 1e6  # Constant-at-fit column is ignored, including on evaluation.
        torch.testing.assert_close(first.predict(evaluation), labels)
        for key in norm_before:
            torch.testing.assert_close(norm_before[key], first.normalization[key])
        exported = first.coefficients
        exported["weight"].fill_(float("nan"))
        torch.testing.assert_close(first.predict(evaluation), labels)
        torch.testing.assert_close(features.detach(), before)
        torch.testing.assert_close(labels, true_labels)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        self.assertIsNone(features.grad)
        self.assertEqual(metadata, metadata_before)
        self.assertEqual(first.report["learner_optimizer_updates"], 0)

    def test_hard_closure_cap_discards_displaced_trial_and_counts_actual_work(self):
        features, labels, metadata = signal()
        # One cell, four turns, keeps the deliberately forced 999 closures small.
        features, labels = features[:4].clone(), labels[:4].clone()
        metadata = {key: value[:4] if key != "cell_inventory" else (value[0],) for key, value in metadata.items()}
        original = features.clone()
        def runaway(optimizer, closure):
            for _ in range(probe.MAX_CLOSURES):
                closure()
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].add_(999.)
            closure()  # Must refuse before objective/gradient work.
        with patch.object(torch.optim.LBFGS, "step", runaway):
            result = self.fit(features, labels, **metadata)
        row = result.report
        self.assertEqual(row["status"], "budget_interrupted")
        self.assertEqual((row["closure_attempts"], row["completed_closures"],
                          row["objective_evaluations_started"], row["backward_evaluations_completed"]), (1000, 999, 999, 999))
        self.assertEqual(row["final_evaluation_attempts"], 0)
        self.assertEqual(row["completed_probe_optimizer_steps"], 0)
        self.assertIsNone(result.coefficients)
        with self.assertRaises(RuntimeError): result.predict(features)
        torch.testing.assert_close(features, original)

    def test_normal_return_is_not_convergence_and_errors_never_get_predictions(self):
        features, labels, metadata = signal()
        def unchanged(optimizer, closure):
            return closure()
        with patch.object(torch.optim.LBFGS, "step", unchanged):
            result = self.fit(features, labels, **metadata)
        self.assertEqual(result.report["status"], "normal_return_unconverged")
        self.assertFalse(result.report["converged"])
        self.assertEqual(result.report["objective_evaluations_completed"], 2)
        self.assertEqual(result.predict(features).shape, labels.shape)
        def failure(optimizer, closure):
            closure()
            raise RuntimeError("synthetic line-search error")
        with patch.object(torch.optim.LBFGS, "step", failure):
            result = self.fit(features, labels, **metadata)
        self.assertEqual(result.report["status"], "solver_error")
        self.assertIsNone(result.coefficients)
        with self.assertRaises(RuntimeError): result.logits(features)
        with patch.object(probe, "_objective", side_effect=FloatingPointError("synthetic nonfinite objective")):
            result = self.fit(features, labels, **metadata)
        self.assertEqual(result.report["status"], "nonfinite")
        self.assertEqual(result.report["objective_evaluations_started"], 1)
        self.assertEqual(result.report["objective_evaluations_completed"], 0)
        self.assertIsNone(result.coefficients)

    def test_ambient_no_grad_works_and_inference_mode_refuses_before_preparation(self):
        features, labels, metadata = signal()
        with torch.no_grad():
            result = self.fit(features, labels, **metadata)
        self.assertIn(result.report["status"], ("converged", "normal_return_unconverged"))
        self.assertEqual(result.report["final_evaluations_completed"], 1)
        with torch.inference_mode():
            torch.testing.assert_close(result.predict(features), labels)
            with patch.object(probe, "_metadata") as metadata_call:
                with self.assertRaisesRegex(ValueError, "inference_mode"):
                    probe.fit_probe(features, labels, **metadata)
                metadata_call.assert_not_called()

    def test_interruptions_preserve_original_type_and_all_completed_fit_work(self):
        features, labels, metadata = signal()
        def interrupt(optimizer, closure):
            closure()
            raise KeyboardInterrupt("synthetic cancellation")
        with patch.object(torch.optim.LBFGS, "step", interrupt):
            with self.assertRaises(KeyboardInterrupt) as raised:
                self.fit(features, labels, **metadata)
        self.assertEqual(raised.exception.probe_report["completed_closures"], 1)
        self.assertEqual(raised.exception.probe_report["status"], "interrupted")
        self.assertNotIn("coefficients", raised.exception.probe_report)
        actual_step, calls = torch.optim.LBFGS.step, []
        def second_interrupt(optimizer, closure):
            calls.append(1)
            if len(calls) == 1:
                return actual_step(optimizer, closure)
            closure()
            raise SystemExit(23)
        with patch.object(torch.optim.LBFGS, "step", second_interrupt):
            try:
                probe.fit_true_and_shuffled(features, labels, **metadata)
            except SystemExit as error:
                self.reports.extend(copy.deepcopy(error.completed_probe_reports))
                self.reports.append(copy.deepcopy(error.probe_report))
                self.assertEqual(error.code, 23)
                self.assertEqual(len(error.completed_probe_reports), 1)
                self.assertEqual(error.probe_report["completed_closures"], 1)
                self.assertEqual(error.probe_report["final_evaluation_attempts"], 0)
            else:
                self.fail("second-fit SystemExit must propagate")
        self.assertEqual(len(calls), 2)

    def test_invalid_metadata_and_runtime_fail_before_fitting(self):
        features, labels, metadata = signal()
        self.assertEqual(len(probe.FOUNDATION_CELLS), 63)
        self.assertEqual(probe.FOUNDATION_CELLS, tuple(sorted(probe.FOUNDATION_CELLS)))
        with self.assertRaises(ValueError): probe.fit_probe(features, labels, **{k:v for k,v in metadata.items() if k != "cell_inventory"})
        for change in ({"coordinates": [metadata["coordinates"][0]]*len(labels)},
                       {"kinds": ["query"]*len(labels)},
                       {"cell_inventory": ("synthetic/a", "synthetic/a")},
                       {"coordinates": [(True, 0, 0)]*len(labels)}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                probe.fit_probe(features, labels, **{**metadata, **change})
        with self.assertRaises(ValueError): probe.fit_probe(features[:-1], labels, **metadata)
        with self.assertRaises(ValueError): probe.fit_probe(features, labels.float(), **metadata)
        with patch.object(torch, "get_num_threads", return_value=2):
            with self.assertRaisesRegex(ValueError, "caller must configure"):
                probe.fit_probe(features, labels, **metadata)
        self.assertEqual(torch.get_num_threads(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
