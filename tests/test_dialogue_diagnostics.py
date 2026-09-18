"""Semantic and leakage boundaries for the known-development diagnostic matrix."""
import copy
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.dialogue_curriculum import (
    ACK, FOCI, _PHRASES, allowed_bindings, dialogue_oracle, generate_dialogues,
    parse_sentence, validate_dialogue,
)
from brain_in_computer.dialogue_diagnostics import (
    DIAGNOSTIC_SPLIT, SCHEMA, SLICE_SUPPORTS, diagnostic_bank, transcript_fingerprint,
)
from brain_in_computer.dialogue_student import (
    build_dialogue_student, evaluate_dialogues, train_dialogue_candidate,
)


class DialogueDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = diagnostic_bank(1200, 8)

    def test_every_slice_preserves_oracle_truth_support_and_complete_pairs(self):
        self.assertEqual(set(self.bank), set(SLICE_SUPPORTS))
        for name, episodes in self.bank.items():
            binding_split, phrase_split = SLICE_SUPPORTS[name]
            self.assertEqual(len(episodes), 3 * 8)
            self.assertEqual({focus: sum(ep["focus"] == focus for ep in episodes)
                              for focus in FOCI}, dict.fromkeys(FOCI, 8))
            query_total = paired_total = 0
            for left, right in zip(episodes[::2], episodes[1::2]):
                self.assertEqual(left["counterfactual_group"], right["counterfactual_group"])
                for episode in (left, right):
                    with self.subTest(slice=name, seed=episode["seed"]):
                        self.assertEqual(dialogue_oracle(episode),
                                         [turn["target"] for turn in episode["turns"]])
                        self.assertEqual(episode["split"], DIAGNOSTIC_SPLIT)
                        metadata = episode["diagnostic"]
                        self.assertEqual(metadata["schema"], SCHEMA)
                        self.assertFalse(metadata["training_allowed"])
                        self.assertIn("not_pristine_audit", metadata["evidence"])
                        original = generate_dialogues(episode["seed"], 1, binding_split,
                                                      episode["focus"])[0]
                        self.assertEqual(metadata["canonical_episode_id"], original["id"])
                        self.assertEqual(episode["counterfactual_group"], original["counterfactual_group"])
                        for turn, source in zip(episode["turns"], original["turns"]):
                            self.assertEqual({k: v for k, v in turn.items() if k != "text"},
                                             {k: v for k, v in source.items() if k != "text"})
                            kind, facts = parse_sentence(turn["text"])
                            self.assertIn(turn["text"], [t.format(**facts) for t in _PHRASES[phrase_split][kind]])
                            if kind == "definition":
                                self.assertIn((facts["alias"], facts["color"]), allowed_bindings(binding_split))
                            query_total += turn["target"] != ACK
                for a, b in zip(left["turns"], right["turns"]):
                    if a["target"] != ACK:
                        self.assertEqual(a["text"], b["text"])
                    paired_total += a["target"] != ACK and a["target"] != b["target"]
            self.assertEqual(query_total, 80)
            self.assertEqual(paired_total, 24)

    def test_phrase_contrasts_preserve_facts_order_and_variant_index(self):
        for prefix in ("familiar_bindings", "unseen_bindings"):
            familiar = self.bank[prefix + "_familiar_phrases"]
            unseen = self.bank[prefix + "_unseen_phrases"]
            for first, second in zip(familiar, unseen):
                self.assertEqual(first["counterfactual_group"], second["counterfactual_group"])
                self.assertEqual(first["seed"], second["seed"])
                self.assertNotEqual(first["id"], second["id"])
                for a, b in zip(first["turns"], second["turns"]):
                    kind, facts = parse_sentence(a["text"])
                    self.assertEqual((kind, facts), parse_sentence(b["text"]))
                    self.assertNotEqual(a["text"], b["text"])
                    train_options = [t.format(**facts) for t in _PHRASES["train"][kind]]
                    dev_options = [t.format(**facts) for t in _PHRASES["dev"][kind]]
                    self.assertEqual(train_options.index(a["text"]), dev_options.index(b["text"]))

    def test_diagnostic_metadata_and_fingerprints_never_enter_student_policy(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            model = build_dialogue_student(42)
            episodes = self.bank["unseen_bindings_familiar_phrases"][:2]
            # Construction has already verified truth; evaluation must not parse.
            with patch("brain_in_computer.dialogue_curriculum.dialogue_oracle",
                       side_effect=AssertionError("oracle entered student evaluation")):
                metrics = evaluate_dialogues(model, episodes, score_replies=False)
            self.assertEqual(metrics["query_total"], 6)
            self.assertEqual(metrics["counterfactual_query_pairs"], 3)
            self.assertFalse(metrics["teacher_used_for_policy"])
            for name, slice_episodes in self.bank.items():
                with self.subTest(slice=name), self.assertRaisesRegex(ValueError, "held-out"):
                    train_dialogue_candidate(model.state_dict(), slice_episodes[:2],
                                             seed=42, steps=1, batch_size=2)
                relabeled = copy.deepcopy(slice_episodes[0])
                relabeled["split"] = "train"
                with self.assertRaises(ValueError):
                    validate_dialogue(relabeled)
        finally:
            torch.set_num_threads(previous_threads)

    def test_fingerprint_is_transcript_only_deterministic_and_order_sensitive(self):
        episodes = self.bank["familiar_bindings_familiar_phrases"][:2]
        digest = transcript_fingerprint(episodes)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, transcript_fingerprint(copy.deepcopy(episodes)))
        self.assertEqual(transcript_fingerprint(episodes[0]), transcript_fingerprint(episodes[:1]))
        altered = copy.deepcopy(episodes)
        altered[0]["diagnostic"]["slice"] = "changed_metadata"
        altered[0]["turns"][0]["target"] = 0
        self.assertEqual(digest, transcript_fingerprint(altered))
        altered[0]["turns"][0]["text"] += " "
        self.assertNotEqual(digest, transcript_fingerprint(altered))
        self.assertNotEqual(digest, transcript_fingerprint(list(reversed(episodes))))

    def test_generation_is_reproducible_and_returns_independent_mutable_episodes(self):
        self.assertEqual(self.bank, diagnostic_bank(1200, 8))
        bank = diagnostic_bank(1200, 2)
        first = bank["familiar_bindings_familiar_phrases"][0]
        other = copy.deepcopy(bank["familiar_bindings_unseen_phrases"][0])
        first["turns"][0]["observations"]["visual"][0][0] = 1
        self.assertEqual(other, bank["familiar_bindings_unseen_phrases"][0])

    def test_invalid_requests_cannot_silently_create_incomplete_pairs(self):
        for seed in (True, -2, 1, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                diagnostic_bank(seed)
        for count in (True, 0, -2, 1, 3, 2.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                diagnostic_bank(1200, count)
        for episodes in (None, "text", [{}], [{"turns": [None]}], [{"turns": [{"text": 3}]}]):
            with self.subTest(episodes=episodes), self.assertRaises(ValueError):
                transcript_fingerprint(episodes)


if __name__ == "__main__":
    unittest.main()
