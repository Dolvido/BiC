"""Small CPU-only checks; no training or saved audit banks are accessed."""
import copy
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.dialogue_student import ByteCodec, checkpoint_digest
from experiments.cognitive_curriculum import generate_cognitive
from experiments.sequence_evaluation import evaluate_banks, evaluate_sequence, prediction_metrics
from experiments.sequence_student import SequenceConfig, build_sequence_student


class SequenceEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self):
        return build_sequence_student(71, config=SequenceConfig(width=16, layers=1, heads=2, feedforward=32))

    def rows(self, count=2):
        return generate_cognitive(120, count, split="dev", family="graph_reachability", level=2)

    def test_hand_predictions_keep_unknown_pair_and_later_denominators_separate(self):
        targets = torch.tensor([[3, 0, 3, 2, 3, 1], [3, 1, 3, 2, 3, 0]])
        predicted = targets.clone()
        predicted[1, 5] = 2
        logits = torch.nn.functional.one_hot(predicted, 4).float() * 8
        reply_correct = torch.ones_like(targets, dtype=torch.bool)
        reply_correct[:, 5] = False
        reply_actions = targets.clone()
        reply_actions[0, 5], reply_actions[1, 5] = -1, 1
        metrics = prediction_metrics(logits, targets, [{"counterfactual_group": "pair"}] * 2,
            reply_correct=reply_correct, reply_actions=reply_actions)
        self.assertEqual((metrics["query_correct"], metrics["query_total"]), (5, 6))
        self.assertEqual((metrics["ask_true"], metrics["ask_predicted"], metrics["ask_correct"]), (2, 3, 2))
        self.assertEqual(metrics["ask_precision"], 2 / 3)
        self.assertEqual(metrics["counterfactual_query_pairs"], 2)
        self.assertEqual(metrics["counterfactual_accuracy"], .5)
        later = metrics["positions"]["last_known_later_query"]
        self.assertEqual((later["correct"], later["total"]), (1, 2))
        self.assertEqual(later["pairs"], {"total": 1, "correct": 0, "accuracy": 0.})
        self.assertEqual(later["reply_exact_correct"], 0)
        self.assertEqual(metrics["reply_parseable_query_count"], 5)
        self.assertAlmostEqual(metrics["query_reply_exact_accuracy"], 2 / 3, places=6)
        self.assertGreater(metrics["brier_score"], 0)

    def test_forward_gets_bos_only_and_no_supervision_weights_and_modes_restore(self):
        model = self.model()
        model.train()
        model.inferior_frontal.eval()
        modes = [module.training for module in model.modules()]
        before = checkpoint_digest(model)
        with patch.object(model, "forward", wraps=model.forward) as forward:
            metrics = evaluate_sequence(model, self.rows(4), batch_size=2, score_replies=True)
        self.assertEqual(forward.call_count, 2)
        for call in forward.call_args_list:
            self.assertEqual(set(call.kwargs), {"token_ids", "valid_mask", "lengths", "eos_positions", "decoder_input_ids"})
            decoder = call.kwargs["decoder_input_ids"]
            self.assertEqual(tuple(decoder.shape), (2, 6, 1))
            self.assertTrue(decoder.eq(ByteCodec.BOS).all())
        self.assertEqual(before, checkpoint_digest(model))
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertFalse(metrics["teacher_used_for_policy"])
        self.assertIsNone(metrics["teacher_forced_reply_loss"])
        self.assertIsNotNone(metrics["query_reply_exact_accuracy"])
        self.assertIn("not regional BiC", metrics["student_scope"])

    def test_reset_history_presents_the_unchanged_utterances_individually(self):
        model, rows = self.model(), self.rows()
        expected_texts = [turn["text"] for row in rows for turn in row["turns"]]
        with patch.object(model, "forward", wraps=model.forward) as forward:
            metrics = evaluate_sequence(model, rows, batch_size=2, score_replies=False, reset_history=True)
        inputs = forward.call_args.kwargs
        self.assertEqual(tuple(inputs["eos_positions"].shape), (12, 1))
        self.assertEqual(tuple(inputs["decoder_input_ids"].shape), (12, 1, 1))
        actual_texts = [model.codec.decode(row.tolist()) for row in inputs["token_ids"]]
        self.assertEqual(actual_texts, expected_texts)
        self.assertEqual(metrics["query_total"], sum(turn["target"] < 3 for row in rows for turn in row["turns"]))
        self.assertTrue(metrics["reset_history"])
        self.assertIsNone(metrics["query_reply_exact_accuracy"])

    def test_blank_control_preserves_boundaries_and_invalid_original_rejected_first(self):
        model, rows = self.model(), self.rows()
        with patch.object(model, "forward", wraps=model.forward) as forward:
            metrics = evaluate_sequence(model, rows, score_replies=False, blank_text=True)
        inputs = forward.call_args.kwargs
        self.assertTrue(inputs["token_ids"].le(ByteCodec.EOS).all())
        self.assertEqual(inputs["lengths"].tolist(), [12, 12])
        self.assertEqual(metrics["ask_true"], sum(turn["target"] == 2 for row in rows for turn in row["turns"]))
        malformed = copy.deepcopy(rows)
        malformed[0]["turns"][0]["text"] = "Unsupported altered fact."
        with patch.object(model, "forward", wraps=model.forward) as forward:
            with self.assertRaises(ValueError):
                evaluate_sequence(model, malformed, blank_text=True, reset_history=True)
            forward.assert_not_called()

    def test_pair_batch_contract_and_named_bank_metrics(self):
        model, rows = self.model(), self.rows(4)
        with self.assertRaisesRegex(ValueError, "whole pairs"):
            evaluate_sequence(model, rows, batch_size=3)
        malformed = [rows[0], rows[2], rows[1], rows[3]]
        with self.assertRaisesRegex(ValueError, "adjacent pair"):
            evaluate_sequence(model, malformed)
        with self.assertRaisesRegex(ValueError, "counterfactual group"):
            evaluate_sequence(model, rows[:2] * 2)
        result = evaluate_banks(model, {"one": rows[:2]}, score_replies=False, batch_size=2)
        self.assertEqual(result["macro_query_accuracy"], result["per_bank"]["one"]["query_accuracy"])
        self.assertEqual(result["macro_pair_accuracy"], result["per_bank"]["one"]["counterfactual_accuracy"])

    def test_nonfinite_policy_output_raises_without_changing_modes_or_weights(self):
        model = self.model()
        before, training = checkpoint_digest(model), model.training
        original = model.forward
        def invalid(*args, **kwargs):
            result = original(*args, **kwargs)
            result["logits"] = torch.full_like(result["logits"], float("nan"))
            return result
        with patch.object(model, "forward", side_effect=invalid):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                evaluate_sequence(model, self.rows(), score_replies=False)
        self.assertEqual(before, checkpoint_digest(model))
        self.assertEqual(training, model.training)


if __name__ == "__main__":
    unittest.main()
