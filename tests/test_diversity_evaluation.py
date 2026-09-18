"""V3 admission, inference isolation and deterministic checkpoint scoring."""
import copy
import io
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.dialogue_student import ByteCodec, checkpoint_digest
from experiments.cognitive_curriculum import REPLIES, generate_cognitive
from experiments.diverse_curriculum import VERSION, generate_diverse
from experiments.diversity_evaluation import evaluate_diverse_banks, evaluate_diverse_sequence
from experiments.sequence_student import SequenceConfig, build_sequence_student


class DiversityEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self):
        return build_sequence_student(81, config=SequenceConfig(width=16, layers=1, heads=2, feedforward=32))

    def rows(self, count=2):
        return generate_diverse(120, count, split="dev", family="graph_reachability", level=2)

    def test_new_provenance_authenticates_without_v2_admission(self):
        model, rows = self.model(), self.rows(4)
        before, original = checkpoint_digest(model), copy.deepcopy(rows)
        model.train()
        model.inferior_frontal.eval()
        modes = [module.training for module in model.modules()]
        with patch("experiments.sequence_data.validate_cognitive", side_effect=AssertionError("v2 admission called")), \
             patch.object(model, "forward", wraps=model.forward) as forward:
            metrics = evaluate_diverse_sequence(model, rows, batch_size=2)
        self.assertEqual(forward.call_count, 2)
        for call in forward.call_args_list:
            self.assertEqual(set(call.kwargs), {"token_ids", "valid_mask", "lengths", "eos_positions", "decoder_input_ids"})
            prefix = call.kwargs["decoder_input_ids"]
            self.assertEqual(tuple(prefix.shape), (2, 6, 1))
            self.assertTrue(prefix.eq(ByteCodec.BOS).all())
        self.assertEqual(metrics["curriculum_version"], VERSION)
        self.assertFalse(metrics["teacher_used_for_policy"])
        self.assertIsNone(metrics["teacher_forced_reply_loss"])
        self.assertIsNotNone(metrics["query_reply_exact_accuracy"])
        self.assertEqual(checkpoint_digest(model), before)
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertEqual(rows, original)

    def test_invalid_truth_or_provenance_rejected_before_blank_reset_inference(self):
        rows, model = self.rows(), self.model()
        variants = []
        wrong_target = copy.deepcopy(rows)
        turn = wrong_target[-1]["turns"][-1]
        turn["target"] = (turn["target"] + 1) % 4
        turn["reply"] = REPLIES[turn["target"]]
        variants.append(wrong_target)
        wrong_version = copy.deepcopy(rows)
        wrong_version[-1]["version"] = "untrusted-version"
        variants.append(wrong_version)
        wrong_world = copy.deepcopy(rows)
        wrong_world[-1]["world_fingerprint"] = "not-the-canonical-world"
        variants.append(wrong_world)
        variants.append(generate_cognitive(120, 2, split="dev", family="graph_reachability", level=2))
        for invalid in variants:
            with patch.object(model, "forward", wraps=model.forward) as forward:
                with self.assertRaises(ValueError):
                    evaluate_diverse_sequence(model, invalid, blank_text=True, reset_history=True)
                forward.assert_not_called()

    def test_pairs_require_both_variants_once_and_whole_pair_batches(self):
        model, rows = self.model(), self.rows(4)
        for malformed in ([rows[0], rows[0]], rows[:2] * 2, [rows[0], rows[2], rows[1], rows[3]], rows[:1]):
            with patch.object(model, "forward", wraps=model.forward) as forward:
                with self.assertRaises(ValueError):
                    evaluate_diverse_sequence(model, malformed)
                forward.assert_not_called()
        with self.assertRaisesRegex(ValueError, "whole pairs"):
            evaluate_diverse_sequence(model, rows, batch_size=3)

    def test_reset_control_keeps_original_utterances_and_original_truth(self):
        model, rows = self.model(), self.rows()
        texts = [turn["text"] for row in rows for turn in row["turns"]]
        with patch.object(model, "forward", wraps=model.forward) as forward:
            metrics = evaluate_diverse_sequence(model, rows, reset_history=True, score_replies=False)
        inputs = forward.call_args.kwargs
        self.assertEqual(tuple(inputs["eos_positions"].shape), (12, 1))
        self.assertEqual(tuple(inputs["decoder_input_ids"].shape), (12, 1, 1))
        self.assertEqual([model.codec.decode(ids.tolist()) for ids in inputs["token_ids"]], texts)
        self.assertEqual(metrics["query_total"], sum(turn["target"] < 3 for row in rows for turn in row["turns"]))
        self.assertIsNone(metrics["query_reply_exact_accuracy"])

    def test_blank_control_keeps_boundaries_and_unknown_denominator(self):
        model, rows = self.model(), self.rows()
        with patch.object(model, "forward", wraps=model.forward) as forward:
            metrics = evaluate_diverse_sequence(model, rows, blank_text=True, score_replies=False)
        inputs = forward.call_args.kwargs
        self.assertTrue(inputs["token_ids"].le(ByteCodec.EOS).all())
        self.assertEqual(inputs["lengths"].tolist(), [12, 12])
        self.assertEqual(metrics["ask_true"], sum(turn["target"] == 2 for row in rows for turn in row["turns"]))
        self.assertTrue(metrics["canonical_validation_before_controls"])

    def test_cpu_checkpoint_restart_preserves_all_scored_metrics(self):
        model, rows = self.model(), self.rows()
        expected = evaluate_diverse_sequence(model, rows)
        stream = io.BytesIO()
        torch.save({"weights": model.state_dict()}, stream)
        stream.seek(0)
        restored = self.model()
        restored.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True)["weights"])
        self.assertEqual(expected, evaluate_diverse_sequence(restored, rows))
        # An unrelated previous call cannot become hidden observation history.
        evaluate_diverse_sequence(restored, generate_diverse(240, 2, split="dev", family="variable_binding"))
        self.assertEqual(expected, evaluate_diverse_sequence(restored, rows))

    def test_nonfinite_actions_raise_and_restore_mixed_modes(self):
        model = self.model()
        model.train()
        model.inferior_frontal.eval()
        before = checkpoint_digest(model)
        modes = [module.training for module in model.modules()]
        original = model.forward
        def invalid(*args, **kwargs):
            result = original(*args, **kwargs)
            result["logits"] = torch.full_like(result["logits"], float("nan"))
            return result
        with patch.object(model, "forward", side_effect=invalid):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                evaluate_diverse_sequence(model, self.rows())
        self.assertEqual(checkpoint_digest(model), before)
        self.assertEqual([module.training for module in model.modules()], modes)

    def test_named_bank_summary_keeps_official_metric_denominators(self):
        result = evaluate_diverse_banks(self.model(), {"graph": self.rows()}, score_replies=False)
        row = result["per_bank"]["graph"]
        self.assertEqual(result["macro_query_accuracy"], row["query_correct"] / row["query_total"])
        self.assertEqual(result["macro_pair_accuracy"], row["counterfactual_accuracy"])
        self.assertEqual(result["macro_later_known_accuracy"], row["positions"]["last_known_later_query"]["accuracy"])
        self.assertEqual(sum(map(sum, row["confusion_matrix_true_rows_predicted_columns"])), row["query_total"])
        self.assertGreaterEqual(row["brier_score"], 0)


if __name__ == "__main__":
    unittest.main()
