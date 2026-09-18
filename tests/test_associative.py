import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.associative import (
    AssociativeMemory, ENCODER_SCHEMA, EncoderConfig, GlyphEncoder,
    distinct_patterns, encoder_hash, load_calibration,
    load_encoder_checkpoint, render_glyph,
)


class AssociativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(3)
        self.encoder = GlyphEncoder(EncoderConfig(16))
        self.memory = AssociativeMemory(self.encoder, threshold=.9, margin=.03, capacity=2, examples_per_label=2)
        self.crop = render_glyph(31823, seed=19)

    def test_renderer_variation_changes_pixels_without_identity_input_to_memory(self):
        other = render_glyph(31823, seed=21)
        self.assertEqual(tuple(self.crop.shape), (3, 32, 32))
        self.assertFalse(torch.equal(self.crop, other))
        self.assertTrue(torch.equal(self.crop, render_glyph(31823, seed=19)))
        self.assertGreaterEqual(float(self.crop.min()), 0)
        self.assertLessEqual(float(self.crop.max()), 1)

    def test_identity_partitions_disjoint(self):
        first = distinct_patterns(80, 3)
        second = distinct_patterns(80, 4, first)
        self.assertEqual(len(set(first)), 80)
        self.assertFalse(set(first) & set(second))

    def test_encoder_outputs_normalized_vectors(self):
        result = self.encoder(self.crop.unsqueeze(0))
        self.assertEqual(tuple(result.shape), (1, 16))
        torch.testing.assert_close(result.norm(dim=-1), torch.ones(1))

    def test_teach_find_name_roundtrip(self):
        self.memory.teach("dax", self.crop)
        self.assertEqual(self.memory.find("dax", [self.crop])["index"], 0)
        self.assertEqual(self.memory.name(self.crop)["label"], "dax")
        self.assertEqual(self.memory.labels, ["dax"])
        self.assertFalse(any(p.grad is not None for p in self.encoder.parameters()))

    def test_duplicate_candidates_abstain_even_zero_margin(self):
        self.memory.margin = 0
        self.memory.teach("dax", self.crop)
        result = self.memory.find("dax", [self.crop, self.crop])
        self.assertIsNone(result["index"])
        self.assertEqual(result["status"], "ambiguous")

    def test_two_names_for_identical_object_abstain(self):
        self.memory.teach("dax", self.crop)
        self.memory.teach("blicket", self.crop)
        result = self.memory.name(self.crop)
        self.assertIsNone(result["label"])
        self.assertEqual(result["status"], "ambiguous")

    def test_unknown_label_and_empty_memory(self):
        self.assertEqual(self.memory.name(self.crop)["status"], "empty_memory")
        self.assertEqual(self.memory.find("not_taught", [self.crop])["status"], "unknown_label")

    def test_threshold_and_margin_both_gate_decisions(self):
        self.assertEqual(self.memory._decision(torch.tensor([.95, .8]))["index"], 0)
        self.assertEqual(self.memory._decision(torch.tensor([.88, .4]))["status"], "unknown")
        self.assertEqual(self.memory._decision(torch.tensor([.95, .94]))["status"], "ambiguous")
        self.assertEqual(self.memory._decision(torch.tensor([.99, .94]))["status"], "ambiguous")
        self.assertEqual(self.memory._decision(torch.tensor([.8, .95]))["index"], 1)

    def test_capacity_rejects_without_silent_forgetting(self):
        self.memory.teach("dax", self.crop)
        self.memory.teach("blicket", self.crop)
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.memory.teach("new", self.crop)
        self.assertEqual(self.memory.labels, ["dax", "blicket"])
        self.assertTrue(self.memory.forget("blicket"))
        self.assertFalse(self.memory.forget("missing"))
        self.memory.teach("new", self.crop)
        self.assertEqual(self.memory.labels, ["dax", "new"])

    def test_exemplar_count_is_bounded(self):
        for seed in range(4):
            result = self.memory.teach("dax", render_glyph(31823, seed=seed))
        self.assertEqual(result["examples"], 2)
        self.assertEqual(len(self.memory._entries["dax"]), 2)

    def test_json_roundtrip_and_original_memory_unchanged_by_new_items(self):
        self.memory.teach("dax", self.crop)
        original = self.memory._entries["dax"][0].clone()
        self.memory.teach("blicket", render_glyph(51191))
        torch.testing.assert_close(self.memory._entries["dax"][0], original, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            self.memory.save(path)
            loaded = AssociativeMemory.load(path, self.encoder)
            self.assertEqual(loaded.labels, self.memory.labels)
            self.assertEqual(loaded.find("dax", [self.crop]), self.memory.find("dax", [self.crop]))
            self.assertEqual(loaded.name(self.crop), self.memory.name(self.crop))

    def test_encoder_change_rejects_stale_embeddings(self):
        self.memory.teach("dax", self.crop)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            self.memory.save(path)
            with torch.no_grad():
                next(self.encoder.parameters()).add_(.01)
            with self.assertRaisesRegex(ValueError, "encoder changed"):
                self.memory.name(self.crop)
            with self.assertRaisesRegex(ValueError, "encoder changed"):
                self.memory.save(path)
            with self.assertRaisesRegex(ValueError, "different encoder"):
                AssociativeMemory.load(path, self.encoder)

    def test_invalid_embeddings_and_schema_rejected(self):
        self.memory.teach("dax", self.crop)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            self.memory.save(path)
            original = json.loads(path.read_text())
            bad = dict(original, schema="unknown")
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "schema"):
                AssociativeMemory.load(path, self.encoder)
            bad = dict(original, entries={"dax": [[0] * 16]})
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "normalized"):
                AssociativeMemory.load(path, self.encoder)
            bad = dict(original, entries={"dax": [[float("nan")] * 16]})
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "embedding"):
                AssociativeMemory.load(path, self.encoder)

    def test_invalid_pixels_and_label_rejected(self):
        for pixels in (torch.zeros(3, 31, 32), torch.zeros(3, 32, 32, dtype=torch.uint8), self.crop * float("nan"), self.crop + 1):
            with self.assertRaises(ValueError):
                self.memory.teach("dax", pixels)
        for label in ("", " ", "x\n", "x" * 81, 3):
            with self.assertRaises(ValueError):
                self.memory.teach(label, self.crop)
        self.assertEqual(self.memory.labels, [])

    def test_encoder_checkpoint_hash_validates_and_loads_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoder.pt"
            checkpoint = {"schema": ENCODER_SCHEMA, "config": {"embedding_size": 16},
                          "model": self.encoder.state_dict(), "encoder_hash": encoder_hash(self.encoder),
                          "calibration": {"threshold": .9, "margin": .03}}
            torch.save(checkpoint, path)
            loaded = load_encoder_checkpoint(path)
            self.assertEqual(encoder_hash(loaded), encoder_hash(self.encoder))
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            self.assertEqual(load_calibration(path)["threshold"], .9)
            checkpoint["encoder_hash"] = "bad"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "hash"):
                load_encoder_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
