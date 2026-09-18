import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from experiments import composition_study as study
from experiments.composition_curriculum import generate_pair, FAMILIES
from experiments.composition_training import CompositionTrainer, COUNTS
from experiments.sequence_student import SequenceConfig


class CompositionStudyTests(unittest.TestCase):
    def test_schedules_match_all_domain_quotas_without_duplicate_joint_run(self):
        self.assertEqual(len(study.JOBS), 5)
        total_episodes = total_updates = 0
        for job in study.JOBS:
            counts = dict.fromkeys(study.expected_counts(job, 0), 0)
            for step in range(study.total_updates(job)):
                for family in study.families_at(job, step):
                    counts[family] += 1
                if step in (0, 1799, 1800, study.total_updates(job) - 1):
                    self.assertEqual(counts, study.expected_counts(job, step + 1))
            self.assertTrue(all(value == 3600 for value in counts.values()))
            total_episodes += sum(counts.values()) * 32
            total_updates += study.total_updates(job)
        self.assertEqual(total_episodes, 1267200)
        self.assertEqual(total_updates, 13200)

    def test_checkpoints_match_held_exposure_and_preserve_pretraining_endpoint(self):
        for job in study.JOBS:
            late = 'color' if job == 'joint' else job.split('-')[1]
            for updates, exposures in zip(study.curve_steps(job), study.EXPOSURES):
                self.assertEqual(study.expected_counts(job, updates)[late] * 32, exposures)
        self.assertEqual(study.expected_counts('seq-count', 1800), {'count': 0, 'color': 2700, 'switch': 2700})
        for invalid in (True, -1, 3601):
            with self.assertRaises(ValueError):
                study.expected_counts('joint', invalid)

    def test_sampler_reconstruction_matches_actual_variable_turn_draws(self):
        torch.set_num_threads(1)
        family = 'count'
        banks = {family: {turns: generate_pair(family, 8101 + turns, turns=turns) for turns in (8, 10, 12)}}
        trainer = CompositionTrainer(banks, config=SequenceConfig(width=8, layers=1, heads=2, feedforward=16, max_turns=12))
        expected = dict.fromkeys(COUNTS, 0)
        buckets = dict.fromkeys((8, 10, 12), 0)
        for _ in range(7):
            _, turns, _, counts = trainer._draw(family)
            buckets[turns] += 1
            for name, count in counts.items():
                expected[name] += count
        state, actual_buckets, actual = study.SampleEvidence(banks).at(family, 7)
        self.assertEqual(expected, actual)
        self.assertEqual(buckets, actual_buckets)
        self.assertTrue(torch.equal(state, trainer.generators[family].get_state()))

    def test_audit_missing_endpoint_gate_precedes_bank_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(study, 'load_protocol', return_value={}), patch.object(study.torch, 'load') as read:
                with self.assertRaisesRegex(ValueError, 'all five'):
                    study.audit(directory, 'cpu')
                read.assert_not_called()
            self.assertFalse((Path(directory) / 'audit').exists())


if __name__ == '__main__':
    unittest.main()
