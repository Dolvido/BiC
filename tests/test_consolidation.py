"""Structural checks for replay-trained shared recall; capability is separate."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.associative import AssociativeMemory, EncoderConfig, GlyphEncoder, render_glyph
from brain_in_computer.computer_use.model import ComputerBrain, ComputerConfig
from brain_in_computer.consolidation import ConsolidatedMemory, NeuralRecall, RecallConfig, replay_examples
from brain_in_computer.regional_memory import RegionalMemoryAgent


class ConsolidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(61)
        self.encoder = GlyphEncoder(EncoderConfig(16)).eval().requires_grad_(False)
        self.recall = NeuralRecall(RecallConfig(8, 16, 16))
        self.memory = ConsolidatedMemory(self.recall, self.encoder, ["dax", "wug"])
        self.crops = torch.stack([render_glyph(13579 + i * 19, i) for i in range(4)])

    def test_optimizer_changes_shared_byte_and_recurrent_weights(self):
        bank = AssociativeMemory(self.encoder)
        bank.teach("dax", self.crops[0])
        bank.teach("wug", self.crops[1])
        names, targets = replay_examples(bank)
        before = {key: value.clone() for key, value in self.recall.state_dict().items()}
        optimizer = torch.optim.AdamW(self.recall.parameters(), lr=.01)
        loss = (1 - (self.recall(names) * targets).sum(-1)).mean()
        loss.backward()
        optimizer.step()
        for key in ("bytes.weight", "sequence.weight_ih_l0", "sequence.weight_hh_l0", "projection.2.weight"):
            self.assertFalse(torch.equal(before[key], self.recall.state_dict()[key]), key)
        self.assertTrue(all(p.grad is None for p in self.encoder.parameters()))

    def test_inference_after_bank_deletion_never_constructs_or_reads_bank(self):
        bank = AssociativeMemory(self.encoder)
        bank.teach("dax", self.crops[0])
        del bank
        with patch.object(AssociativeMemory, "__init__", side_effect=AssertionError("bank")), \
             patch.object(AssociativeMemory, "evidence", side_effect=AssertionError("lookup")):
            result = self.memory.evidence("dax", self.crops)
        self.assertTrue(result["known"])
        self.assertEqual(result["scores"].shape, (4,))

    def test_unknown_gate_has_no_visual_vectors_and_skips_neural_prediction(self):
        with patch.object(self.recall, "forward", side_effect=AssertionError("unknown recall")):
            result = self.memory.evidence("untaught", self.crops)
        self.assertFalse(result["known"])
        self.assertEqual(int(torch.count_nonzero(result["scores"])), 0)
        self.assertEqual(self.memory._known, frozenset(("dax", "wug")))

    def test_erasing_shared_weights_removes_known_visual_evidence(self):
        before = self.memory.evidence("dax", self.crops)["scores"]
        with torch.no_grad():
            for parameter in self.recall.parameters():
                parameter.zero_()
        after = self.memory.evidence("dax", self.crops)
        self.assertTrue(after["known"])
        self.assertEqual(int(torch.count_nonzero(after["scores"])), 0)
        self.assertFalse(torch.equal(before, after["scores"]))

    def test_serialized_state_contains_shared_weights_and_names_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recall.pt"
            self.memory.save(path)
            payload = torch.load(path, weights_only=True)
            self.assertEqual(set(payload), {"schema", "config", "encoder_hash", "known_names", "model", "metadata"})
            self.assertEqual(payload["known_names"], ["dax", "wug"])
            self.assertEqual(set(payload["model"]), set(self.recall.state_dict()))
            loaded = ConsolidatedMemory.load(path, self.encoder)
            self.assertTrue(torch.equal(self.memory.evidence("dax", self.crops)["scores"],
                                        loaded.evidence("dax", self.crops)["scores"]))

    def test_loading_rejects_different_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recall.pt"
            self.memory.save(path)
            other = GlyphEncoder(EncoderConfig(16))
            with self.assertRaisesRegex(ValueError, "different visual encoder"):
                ConsolidatedMemory.load(path, other)

    def test_act_routes_all_scores_into_existing_regional_network(self):
        computer = ComputerBrain(ComputerConfig(hidden_size=12, language_hidden_size=16,
                                                embedding_size=8, visual_features=8))
        agent = RegionalMemoryAgent(computer, self.encoder).eval()
        with patch.object(agent, "forward_evidence", wraps=agent.forward_evidence) as forward:
            result = self.memory.act(agent, 'click the object left of "dax"', self.crops)
        self.assertEqual(forward.call_count, 1)
        args = forward.call_args.args
        self.assertEqual(args[0].shape, (1, 4))
        self.assertEqual(args[2], ["click the object left of <name>"])
        self.assertEqual(result["action_source"], "regional_motor_cortex")
        self.assertEqual(result["response_source"], "neural_byte_decoder")
        self.assertEqual(result["memory_source"], "shared_neural_weights")

    def test_utf8_name_and_padding_do_not_change_address(self):
        alone = self.recall(["猫"])[0]
        together = self.recall(["a much longer name", "猫"])[1]
        self.assertTrue(torch.allclose(alone, together, atol=1e-6, rtol=1e-6))

    def test_malformed_names_pixels_and_dimension_mismatch_rejected(self):
        for names in ("dax", [], ["\n"], [None]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.recall(names)
        for pixels in (torch.zeros(3, 32, 32), self.crops.long(), torch.full_like(self.crops, float("nan")), self.crops + 2):
            with self.assertRaises(ValueError):
                self.memory.evidence("dax", pixels)
        with self.assertRaises(ValueError):
            ConsolidatedMemory(self.recall, GlyphEncoder(), ["dax"])
        with self.assertRaises(ValueError):
            ConsolidatedMemory(self.recall, self.encoder, ["dax", " dax "])

    def test_encoder_mutation_invalidates_inference(self):
        with torch.no_grad():
            next(self.encoder.parameters()).add_(.01)
        with self.assertRaisesRegex(ValueError, "original frozen visual encoder"):
            self.memory.evidence("dax", self.crops)

    def test_cached_evaluator_derives_known_flag_from_memory_not_teacher_spec(self):
        from experiments.consolidate_memory import score_data
        with torch.no_grad():
            embeddings = self.encoder(self.crops).unsqueeze(0)
        spec = {"name": "untaught", "known": True, "target": 1}
        data = score_data(self.memory, [spec], embeddings)
        self.assertEqual(float(data["known"].item()), 0)
        self.assertEqual(int(torch.count_nonzero(data["scores"])), 0)
        spec = {"name": "dax", "known": False, "target": 1}
        data = score_data(self.memory, [spec], embeddings)
        self.assertEqual(float(data["known"].item()), 1)
        self.assertTrue(torch.allclose(data["scores"][0], self.memory.evidence("dax", self.crops)["scores"], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
