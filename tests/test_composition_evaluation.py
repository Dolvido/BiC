import copy
import unittest
from unittest.mock import patch

import torch

from experiments.composition_evaluation import PreparedBank, prediction_metrics, verify_cpu_restart
from experiments.sequence_student import SequenceConfig, build_sequence_student


class CompositionMetricsTests(unittest.TestCase):
    def test_aggregate_accuracy_cannot_hide_failed_opposite_pairs(self):
        truth = torch.tensor([[3, 2, 0, 3, 1, 0, 3, 0], [3, 2, 1, 3, 0, 1, 3, 1]])
        chosen = truth.clone()
        chosen[chosen < 2] = 0
        logits = torch.nn.functional.one_hot(chosen, 4).float() * 5
        row = prediction_metrics(logits, truth, reply_correct=chosen.eq(truth), reply_actions=chosen)
        self.assertEqual(row['query_accuracy'], .6)
        self.assertEqual(row['known_accuracy'], .5)
        self.assertEqual(row['opposite_pair_total'], 4)
        self.assertEqual(row['final_pairs'], {'total': 1, 'correct': 0})
        self.assertEqual(row['final_pair_accuracy'], 0.)
        self.assertEqual(row['final_reply_pair_accuracy'], 0.)
        self.assertEqual(row['ask_recall'], 1.)
        self.assertEqual(row['action_reply_agreement'], 1.)

    def test_absent_ask_denominators_and_unmeasured_replies_stay_null(self):
        truth = torch.tensor([[3, 0], [3, 1]])
        row = prediction_metrics(torch.nn.functional.one_hot(truth, 4).float() * 5, truth)
        self.assertEqual(row['final_pair_accuracy'], 1.)
        self.assertIsNone(row['ask_precision'])
        self.assertIsNone(row['ask_recall'])
        self.assertIsNone(row['per_target']['2']['accuracy'])
        self.assertIsNone(row['query_reply_accuracy'])

    def test_malformed_final_counterfactual_is_rejected(self):
        truth = torch.tensor([[3, 0], [3, 0]])
        with self.assertRaisesRegex(ValueError, 'opposite'):
            prediction_metrics(torch.zeros(2, 2, 4), truth)


class PreparedCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.composition_curriculum import generate_pair
        torch.set_num_threads(1)
        cls.rows = generate_pair('color', 3001, split='dev', turns=8)
        cls.config = SequenceConfig(width=8, layers=1, heads=2, feedforward=16, max_turns=12)

    def test_original_mutation_cannot_change_prepared_inference_and_no_revalidation(self):
        rows = copy.deepcopy(self.rows)
        bank = PreparedBank(rows, role='dev', config=self.config)
        model = build_sequence_student(19, config=self.config)
        model.train()
        first = bank.score(model, score_replies=False)
        rows[0]['turns'][-1]['text'] = 'Tampered caller input'
        with patch('experiments.composition_curriculum.validate_pair', side_effect=AssertionError('revalidated')):
            second = bank.score(model, score_replies=False)
        for key in ('query_correct', 'final_pairs', 'confusion_matrix', 'bank'):
            self.assertEqual(first[key], second[key])
        self.assertTrue(model.training)
        self.assertFalse(second['teacher_used_for_policy'])

    def test_tampered_truth_or_role_rejected_before_compilation(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['turns'][-1]['target'] = 2
        with self.assertRaises(ValueError):
            PreparedBank(rows, role='dev', config=self.config)
        with self.assertRaises(ValueError):
            PreparedBank(self.rows, role='audit', config=self.config)

    def test_controls_keep_truth_and_cpu_restart_is_exact(self):
        model = build_sequence_student(19, config=self.config)
        bank = PreparedBank(self.rows, role='dev', config=self.config)
        results = [bank.score(model, control=control, score_replies=True) for control in ('normal', 'blank', 'reset')]
        self.assertTrue(all(row['final_pairs']['total'] == 1 for row in results))
        self.assertTrue(all(row['query_total'] == results[0]['query_total'] for row in results))
        self.assertTrue(verify_cpu_restart(model, self.rows[0])['exact'])


if __name__ == '__main__':
    unittest.main()
