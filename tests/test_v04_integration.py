"""Independent checks of memory measurement and process-boundary behavior."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from brain_in_computer.associative import (
    AssociativeMemory, ENCODER_SCHEMA, EncoderConfig, GlyphEncoder,
    encoder_hash, render_glyph,
)
from experiments.train_associative import decisions


class MemoryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(812)
        self.encoder = GlyphEncoder(EncoderConfig(16)).eval()
        self.patterns = (31823, 51191, 28665, 47421)
        self.crops = [render_glyph(pattern, index + 71)
                      for index, pattern in enumerate(self.patterns)]
        self.memory = AssociativeMemory(self.encoder, threshold=.999, margin=0)
        for index, crop in enumerate(self.crops):
            self.memory.teach(f"object_{index}", crop)

    def test_public_single_exemplar_retrieval_matches_experiment_scoring(self):
        candidates = torch.stack([
            self.crops[2], self.crops[0], self.crops[1],
            render_glyph(self.patterns[3], 882),
        ])
        with torch.no_grad():
            support = self.encoder(torch.stack(self.crops))
            encoded = self.encoder(candidates)
        expected = decisions(support @ encoded.T, self.memory.threshold,
                             self.memory.margin).tolist()
        actual = [self.memory.find(label, candidates)["index"]
                  for label in self.memory.labels]
        self.assertEqual(actual, [index if index >= 0 else None for index in expected])
        self.assertEqual(actual[0], 1)
        self.assertEqual(actual[1], 2)
        self.assertEqual(actual[2], 0)

    def test_public_single_exemplar_naming_matches_experiment_scoring(self):
        queries = torch.stack(self.crops + [render_glyph(40955, 13)])
        with torch.no_grad():
            scores = self.encoder(queries) @ self.encoder(torch.stack(self.crops)).T
        expected = decisions(scores, self.memory.threshold, self.memory.margin).tolist()
        actual = [self.memory.name(crop)["label"] for crop in queries]
        self.assertEqual(actual, [f"object_{index}" if index >= 0 else None
                                  for index in expected])
        self.assertEqual(actual[:4], self.memory.labels)

    def test_checkpoint_and_memory_restore_in_fresh_process_without_teaching(self):
        """A child loads only serialized state; it inherits no Python objects."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "encoder.pt"
            memory_path = root / "memory.json"
            torch.save({"schema": ENCODER_SCHEMA,
                        "config": asdict(self.encoder.config),
                        "model": self.encoder.state_dict(),
                        "encoder_hash": encoder_hash(self.encoder),
                        "calibration": {"threshold": .999, "margin": 0}}, checkpoint)
            self.memory.save(memory_path)
            # The probe is already a tensor: no glyph identity enters the child.
            torch.save(torch.stack(self.crops), root / "queries.pt")
            expected = [self.memory.name(crop) for crop in self.crops]
            script = """
import json, sys, torch
from pathlib import Path
from unittest.mock import patch
from brain_in_computer.associative import AssociativeMemory, load_encoder_checkpoint
torch.set_num_threads(1)
root = Path(sys.argv[1])
encoder = load_encoder_checkpoint(root / 'encoder.pt')
with patch.object(AssociativeMemory, 'teach', side_effect=AssertionError('retraining during restore')):
    memory = AssociativeMemory.load(root / 'memory.json', encoder)
    queries = torch.load(root / 'queries.pt', weights_only=True)
    print(json.dumps([memory.name(crop) for crop in queries]))
"""
            result = subprocess.run([sys.executable, "-c", script, str(root)],
                                    cwd=Path(__file__).resolve().parents[1],
                                    capture_output=True, text=True, timeout=30, check=True)
            self.assertEqual(json.loads(result.stdout), expected)


if __name__ == "__main__":
    unittest.main()
