"""Portability, inference isolation, and explicit input limits for the demo."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from examples.english_demo import DEFAULT_WEIGHTS, EnglishSession, load_model, run_examples


class PublicDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model, cls.manifest = load_model()

    def setUp(self):
        self.session = EnglishSession(self.model)

    def test_exact_archived_weights_and_cpu_only(self):
        self.assertEqual(checkpoint_digest(self.model),
            "002a5cdd39c1ad1e4625f2a7e804eb1d1a13ba62598db90ff0bc3354467d758b")
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 2264725)
        self.assertTrue(all(p.device.type == "cpu" and not p.requires_grad for p in self.model.parameters()))
        self.assertFalse(self.model.training)

    def test_export_has_no_training_state(self):
        artifact = torch.load(DEFAULT_WEIGHTS, map_location="cpu", weights_only=True)
        self.assertEqual(set(artifact), {"schema", "architecture", "config", "weights", "weights_sha256", "provenance"})
        self.assertTrue(all(isinstance(t, torch.Tensor) for t in artifact["weights"].values()))

    def test_all_examples_run_without_oracles_auxiliary_head_or_parameter_changes(self):
        before = checkpoint_digest(self.model)
        with patch("experiments.cognitive_curriculum.cognitive_oracle", side_effect=AssertionError("oracle called")), \
             patch.object(self.model, "state_logits", side_effect=AssertionError("auxiliary head called")):
            results = run_examples(self.model)
        self.assertEqual([len(result["turns"]) for result in results], [6, 6, 7, 8])
        self.assertTrue(all(turn["reply_tokens"][0] == 1 for result in results for turn in result["turns"]))
        self.assertEqual(checkpoint_digest(self.model), before)

    def test_reset_repeats_the_same_native_prediction(self):
        first = self.session.observe("dax is red.")
        self.session.observe("wug is blue.")
        self.session.reset()
        self.assertEqual(first, self.session.observe("dax is red."))

    def test_turn_overflow_rejected_without_forgetting_history(self):
        for _ in range(12):
            self.session.observe("dax is red.")
        before = list(self.session.history)
        with self.assertRaisesRegex(ValueError, "session limit"):
            self.session.observe("Is dax red?")
        self.assertEqual(self.session.history, before)

    def test_utf8_byte_overflow_rejected_before_forward(self):
        self.session.observe("dax is red.")
        with patch.object(self.model, "forward", side_effect=AssertionError("overflow reached model")):
            with self.assertRaisesRegex(ValueError, "UTF-8 bytes"):
                self.session.observe("\u00e9" * 65)
        self.assertEqual(self.session.history, ["dax is red."])

    def test_aggregate_context_overflow_rejected_without_truncation(self):
        self.session.history = ["x" * 128] * 7
        with patch.object(self.model, "forward", side_effect=AssertionError("overflow reached model")):
            with self.assertRaisesRegex(ValueError, "no truncation"):
                self.session.observe("x" * 128)
        self.assertEqual(len(self.session.history), 7)

    def test_empty_input_rejected_without_mutation(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.session.observe("  ")
        self.assertEqual(self.session.history, [])

    def test_tampered_asset_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / "tampered.pt"
            asset.write_bytes(b"not an inference checkpoint")
            asset.with_suffix(".json").write_text(json.dumps(self.manifest), encoding="utf-8")
            with patch("examples.english_demo.torch.load", side_effect=AssertionError("bad asset deserialized")):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    load_model(asset)


if __name__ == "__main__":
    unittest.main()
