"""Tiny CPU admission/inference checks; no optimizer or formal study evidence."""
import copy
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.language import ByteCodec
from experiments import composition_curriculum as legacy
from experiments import foundation_curriculum as curriculum
from experiments.foundation_evaluation import FoundationBank
from experiments.sequence_student import SequenceConfig, build_sequence_student


class FoundationEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config = SequenceConfig(width=8, layers=1, heads=2, feedforward=16,
                                    max_positions=1560, max_turns=12)
        cls.rows = cls.pair()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    @staticmethod
    def pair(family="color", seed=401000001, *, depth=2, turns=8, split="dev"):
        # Private test namespace only. Familiar composed dev probes deliberately
        # request train motifs; depth-two has no legacy dev motif.
        partition = None if depth < 2 else "train" if split == "dev" else split
        return curriculum.generate_pair(family, seed, depth=depth, turns=turns,
                                        split=split, structure_split=partition)

    def model(self):
        return build_sequence_student(4101, config=self.config)

    def test_canonical_roles_and_uniform_cell_metadata(self):
        for index, (family, depth, turns) in enumerate((
                ("color", 0, 8), ("count", 1, 10), ("switch", 5, 12))):
            for role, split in (("train_fit", "train"), ("dev", "dev"), ("audit", "audit")):
                with self.subTest(family=family, role=role):
                    rows = self.pair(family, 402000000 + index, depth=depth, turns=turns, split=split)
                    bank = FoundationBank(rows, role=role, config=self.config)
                    self.assertEqual(bank.cell, {"family": family, "depth": depth,
                                                "turns": turns, "primitive_shared": depth <= 1})
                    self.assertEqual(bank.identity["version"], curriculum.VERSION)
                    self.assertEqual(bank.identity["role"], role)
                    self.assertEqual(bank.identity["episodes"], 2)
                    self.assertEqual(bank.identity["turns"], turns)

    def test_wrong_roles_duplicates_incomplete_pairs_and_mixed_cells_rejected(self):
        for role in ("train_fit", "audit", "train", ""):
            with self.subTest(role=role), self.assertRaises(ValueError):
                FoundationBank(self.rows, role=role, config=self.config)
        other_cells = [self.pair("count", 403000001),
                       self.pair(seed=403000002, depth=1),
                       self.pair(seed=403000003, turns=10)]
        for extra in other_cells:
            with self.assertRaisesRegex(ValueError, "one family, depth and turn count"):
                FoundationBank(self.rows + extra, role="dev", config=self.config)
        for malformed in ([], self.rows[:1], self.rows + copy.deepcopy(self.rows),
                          [self.rows[0], self.rows[0]], list(reversed(self.rows))):
            with self.assertRaises(ValueError):
                FoundationBank(malformed, role="dev", config=self.config)

    def test_canonical_tampering_and_relabelled_audit_ancestry_rejected(self):
        for field in ("text", "target", "reply", "depth", "family", "version"):
            changed = copy.deepcopy(self.rows)
            if field == "text": changed[0]["turns"][0]["text"] += " Extra."
            elif field == "target": changed[0]["turns"][-1]["target"] = 2
            elif field == "reply": changed[0]["turns"][-1]["reply"] = "tampered"
            elif field == "depth": changed[0]["depth"] = 5
            elif field == "family": changed[0]["family"] = "count"
            else: changed[0]["version"] = legacy.VERSION
            with self.subTest(field=field), self.assertRaises(ValueError):
                FoundationBank(changed, role="dev", config=self.config)
        audit = self.pair(seed=404000001, split="audit")
        self.assertTrue(any(q["composed"] and q["legacy_structure_partition"] == "audit"
                            for row in audit for q in curriculum.query_ancestries(row)))
        for row in audit:
            row["split"] = "dev"
        with self.assertRaises(ValueError):
            FoundationBank(audit, role="dev", config=self.config)
        with self.assertRaisesRegex(ValueError, "admission partition"):
            curriculum.generate_pair("color", 404000002, depth=2, split="dev", structure_split="audit")

    def test_snapshot_and_public_metadata_are_isolated_from_callers(self):
        rows = copy.deepcopy(self.rows)
        bank = FoundationBank(rows, role="dev", config=self.config)
        model = self.model()
        before = bank.score(model, score_replies=False)
        rows[0]["turns"][-1]["text"] = "Caller changed the observations."
        rows[1]["turns"][-1]["target"] = 2
        identity, cell = bank.identity, bank.cell
        identity["sha256"] = "0" * 64
        identity["config"]["width"] = 999
        cell["depth"] = 999
        after = bank.score(model, score_replies=False)
        for key in ("bank", "query_correct", "final_pairs", "confusion_matrix", "by_turn"):
            self.assertEqual(before[key], after[key])
        self.assertEqual(bank.cell["depth"], 2)
        self.assertEqual(bank.identity["config"]["width"], self.config.width)

    def test_inference_calls_no_curriculum_regeneration_or_oracle(self):
        bank = FoundationBank(self.rows, role="dev", config=self.config)
        model = self.model()
        with ExitStack() as stack:
            for name in ("generate_pair", "validate_pair", "validate_row", "query_ancestries", "final_depth"):
                stack.enter_context(patch.object(curriculum, name, side_effect=AssertionError("curriculum at inference")))
            for name in ("generate_pair", "validate_pair", "abstract_oracle", "english_oracle", "parse_sentence"):
                stack.enter_context(patch.object(legacy, name, side_effect=AssertionError("oracle at inference")))
            for control in ("normal", "blank", "reset"):
                result = bank.score(model, batch_size=2, score_replies=True, control=control)
                self.assertFalse(result["teacher_used_for_policy"])
                self.assertTrue(result["free_running_replies"])
                self.assertEqual(result["decoder_prefix"], "BOS only")

    def test_encoder_receives_only_observed_text_and_decoder_only_bos(self):
        bank = FoundationBank(self.rows, role="dev", config=self.config)
        model = self.model()
        actual_forward = model.forward
        for control in ("normal", "blank", "reset"):
            captured = []
            def inspect(*args, **kwargs):
                self.assertFalse(args)
                self.assertEqual(set(kwargs), {"token_ids", "valid_mask", "lengths", "eos_positions", "decoder_input_ids"})
                self.assertFalse(torch.is_grad_enabled())
                self.assertTrue(all(not module.training for module in model.modules()))
                self.assertTrue(torch.all(kwargs["decoder_input_ids"] == ByteCodec.BOS))
                self.assertEqual(kwargs["decoder_input_ids"].shape[-1], 1)
                captured.append({key: value.clone() for key, value in kwargs.items()})
                return actual_forward(**kwargs)
            with patch.object(model, "forward", side_effect=inspect):
                bank.score(model, batch_size=2, score_replies=True, control=control)
            self.assertEqual(len(captured), 1)
            inputs = captured[0]
            texts = [[turn["text"] for turn in row["turns"]] for row in self.rows]
            if control == "reset": texts = [[text] for row in texts for text in row]
            for index, row in enumerate(texts):
                expected = sum(([ByteCodec.BOS, ByteCodec.EOS] if control == "blank"
                                else model.codec.encode(text) for text in row), [])
                length = int(inputs["lengths"][index])
                self.assertEqual(inputs["token_ids"][index, :length].tolist(), expected)
                self.assertEqual(inputs["eos_positions"].shape[1], len(row))
                self.assertTrue(inputs["valid_mask"][index, :length].all())
                self.assertFalse(inputs["valid_mask"][index, length:].any())
                self.assertTrue(torch.all(inputs["token_ids"][index, length:] == ByteCodec.PAD))

    def test_controls_preserve_weights_mixed_modes_and_truth_denominators(self):
        bank = FoundationBank(self.rows, role="dev", config=self.config)
        model = self.model().train()
        model.tokens.eval()
        modes = [module.training for module in model.modules()]
        digest = checkpoint_digest(model)
        targets = torch.tensor([[turn["target"] for turn in row["turns"]] for row in self.rows])
        eligible = (targets[::2] < 2) & (targets[1::2] < 2) & targets[::2].ne(targets[1::2])
        for control in ("normal", "blank", "reset"):
            result = bank.score(model, control=control, score_replies=True)
            self.assertEqual(checkpoint_digest(model), digest)
            self.assertEqual([module.training for module in model.modules()], modes)
            self.assertEqual(result["query_total"], int(targets.ne(3).sum()))
            self.assertEqual(result["known_total"], int(targets.lt(2).sum()))
            self.assertEqual(result["ask_true"], int(targets.eq(2).sum()))
            self.assertEqual(result["final_pairs"]["total"], 1)
            self.assertEqual(result["opposite_pair_total"], int(eligible.sum()))
            self.assertEqual([row["total"] for row in result["by_turn"]], targets.ne(3).sum(0).tolist())
            self.assertEqual([row["opposite_pair_total"] for row in result["by_turn"]], eligible.sum(0).tolist())
            self.assertIsNotNone(result["query_reply_correct"])
            self.assertIsNotNone(result["final_reply_pair_correct"])

    def test_invalid_score_options_and_inference_failure_preserve_modes(self):
        bank = FoundationBank(self.rows, role="dev", config=self.config)
        model = self.model().train()
        model.tokens.eval()
        modes = [module.training for module in model.modules()]
        digest = checkpoint_digest(model)
        for options in ({"batch_size": 1}, {"batch_size": 3}, {"control": "oracle"}, {"score_replies": 1}):
            with self.assertRaises(ValueError): bank.score(model, **options)
        with patch.object(model, "forward", side_effect=RuntimeError("inference failure")):
            with self.assertRaisesRegex(RuntimeError, "inference failure"):
                bank.score(model)
        self.assertEqual(checkpoint_digest(model), digest)
        self.assertEqual([module.training for module in model.modules()], modes)


if __name__ == "__main__":
    unittest.main()
