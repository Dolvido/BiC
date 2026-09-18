import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import realization_study as study
from experiments.composition_curriculum import FAMILIES, generate_pair
from experiments.realization_training import RealizationStream, RealizationTrainer
from experiments.sequence_student import SequenceConfig


class RealizationStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.banks = {family: {turns: sum((generate_pair(family, 263_000_000 + index * 1000 + turns * 10 + offset,
                    turns=turns) for offset in range(2)), [])
                    for turns in (8, 10, 12)} for index, family in enumerate(FAMILIES)}

    def test_independent_structural_reconstruction_matches_actual_recipe_occurrences(self):
        stream = RealizationStream(self.banks, mode="fixed", protected_transcripts=[])
        for _ in range(7):
            for family in FAMILIES:
                stream.draw(family)
        expected = study.structural_evidence(self.banks, 7)
        state = stream.snapshot()
        for family in FAMILIES:
            self.assertEqual(study._canonical(state["occurrences"][family]), expected[family]["occurrences"])
            self.assertEqual(study._canonical(state["bucket_microbatches"][family]),
                             expected[family]["bucket_microbatches"])
            self.assertEqual(state["exposures"][family]["episodes"], expected[family]["episodes"])
            self.assertEqual(state["exposures"][family]["turns"], expected[family]["turns"])
        for invalid in (True, -1, study.TOTAL + 1):
            with self.assertRaises(ValueError):
                study.structural_evidence(self.banks, invalid)

    def test_checkpoint_gate_rejects_counter_and_sampler_corruption(self):
        config = SequenceConfig(width=8, heads=2, layers=1, feedforward=16, max_turns=12)
        trainer = RealizationTrainer(self.banks, mode="fresh", protected_transcripts=[], config=config)
        trainer.step()
        protocol = {"test": "fixed structural schedule"}
        saved = {"schema": study.SCHEMA, "protocol": protocol, "job": "fresh", "training": trainer.snapshot()}
        study.validate_snapshot(saved, protocol, "fresh", {"train": self.banks})
        variants = []
        changed = copy.deepcopy(saved)
        changed["training"]["exposures"]["color"]["turns"] += 1
        variants.append(changed)
        changed = copy.deepcopy(saved)
        changed["training"]["samplers"]["color"] = torch.Generator().manual_seed(1).get_state()
        variants.append(changed)
        changed = copy.deepcopy(saved)
        changed["training"]["realization"]["occurrences"]["color"][8][0] += 1
        variants.append(changed)
        for changed in variants:
            with self.assertRaises(ValueError):
                study.validate_snapshot(changed, protocol, "fresh", {"train": self.banks})

    def test_audit_missing_endpoint_gate_precedes_bank_read(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(study, "load_protocol", return_value={}), patch.object(study, "_read_banks") as read:
                with self.assertRaisesRegex(ValueError, "both completed endpoints"):
                    study.audit(directory, "cpu")
                read.assert_not_called()

    def test_audit_requires_both_stream_verifications_before_bank_read(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in study._required(directory):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"endpoint file presence only")
            with patch.object(study, "load_protocol", return_value={}), patch.object(study, "_read_banks") as read:
                with self.assertRaisesRegex(ValueError, "both complete stream replays"):
                    study.audit(directory, "cpu")
                read.assert_not_called()

    def test_json_round_trip_does_not_change_integer_bucket_evidence(self):
        value = {"bucket": {8: 7, 10: 3, 12: 4}, "episodes": 448}
        study._same(value, json.loads(json.dumps(value)))
        wrong = copy.deepcopy(value)
        wrong["bucket"][10] += 1
        with self.assertRaises(ValueError):
            study._same(value, wrong)

    def test_completed_inputs_hash_every_curve_and_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            required = study._required(directory)
            self.assertEqual(len(required), 2 * (2 + len(study.STEPS)))
            for index, path in enumerate(required):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(str(index).encode())
            before = study._completed_inputs(directory)
            required[-1].write_bytes(b"changed")
            self.assertNotEqual(before, study._completed_inputs(directory))


if __name__ == "__main__":
    unittest.main()
