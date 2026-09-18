"""Exact fixed/fresh streams, admission boundaries and CPU optimizer restart."""
import copy
import io
import unittest
from unittest.mock import patch

import torch

from experiments.composition_curriculum import FAMILIES, generate_pair
from experiments.composition_training import CompositionTrainer
from experiments.realization_banks import transcript_digest
from experiments.realization_training import RealizationStream, RealizationTrainer, replay_evidence, stream_evidence
from experiments.sequence_student import SequenceConfig
from experiments.train_cognitive import fingerprint_rows


class RealizationTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.banks = {family: {turns: generate_pair(family, 173000000 + index * 1000 + turns,
            turns=turns) for turns in (8, 10, 12)} for index, family in enumerate(FAMILIES)}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def trainer(self, mode="fresh", **kwargs):
        return RealizationTrainer(self.banks, mode=mode, protected_transcripts=[], micro_batch_size=2,
            config=SequenceConfig(width=16, layers=1, heads=2, feedforward=32, max_turns=12), **kwargs)

    def compare(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.compare(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.compare(left, right)
        else:
            self.assertEqual(first, second)

    def test_fixed_optimizer_and_draws_match_unchanged_composition_trainer(self):
        fixed = self.trainer("fixed")
        self.assertFalse(fixed.encoded)
        self.assertFalse(fixed.row_counts)
        original = CompositionTrainer(self.banks, seed=2901, sampler_seed=3901, micro_batch_size=2,
                                      config=fixed.model.config)
        for _ in range(2):
            self.compare(fixed.step(), original.step(FAMILIES))
            for field in ("weights", "optimizer", "samplers", "exposures", "family_microbatches", "bucket_microbatches"):
                self.compare(fixed.snapshot()[field], original.snapshot()[field])

    def test_first_occurrence_same_then_fresh_changes_only_realization_stream(self):
        banks = {family: {8: self.banks[family][8]} for family in FAMILIES}
        streams = [RealizationStream(banks, mode=mode, protected_transcripts=[], micro_batch_size=2)
                   for mode in ("fixed", "fresh")]
        for family in FAMILIES:
            fixed, fresh = [stream.draw(family) for stream in streams]
            self.assertEqual(fixed, fresh)
        for family in FAMILIES:
            fixed, fresh = [stream.draw(family) for stream in streams]
            self.assertEqual(fixed[1:3], fresh[1:3])
            self.assertNotEqual(fingerprint_rows(fixed[0]), fingerprint_rows(fresh[0]))
            for original, changed in zip(fixed[0], fresh[0]):
                for key in ("program_id", "structure_id", "query_ancestries"):
                    self.assertEqual(original[key], changed[key])
        evidence = [stream_evidence(stream.snapshot()) for stream in streams]
        for field in ("sampler_sha256", "occurrences", "family_microbatches", "bucket_microbatches", "structural_stream_sha256"):
            self.assertEqual(evidence[0][field], evidence[1][field])
        self.assertNotEqual(evidence[0]["canonical_stream_sha256"], evidence[1]["canonical_stream_sha256"])
        self.assertEqual(evidence[0]["collisions"]["accepted_duplicate_transcripts"], 6)
        self.assertEqual(evidence[1]["collisions"]["accepted_duplicate_transcripts"], 0)

    def test_protected_seen_and_reserved_collisions_retry_deterministically(self):
        banks = {"color": {8: self.banks["color"][8]}}
        seed_stream = RealizationStream(banks, mode="fresh", protected_transcripts=[], micro_batch_size=2)
        forbidden = seed_stream._candidate("color", 8, 0, 1, 0)
        protected = [transcript_digest(row) for row in forbidden]
        stream = RealizationStream(banks, mode="fresh", protected_transcripts=protected, micro_batch_size=2)
        stream.draw("color")
        observed, _, _, _ = stream.draw("color")
        self.assertEqual(stream.state["latest"]["color"][8][0]["attempt"], 1)
        self.assertEqual(stream.state["collisions"]["protected_pairs"], 1)
        self.assertEqual(stream.state["collisions"]["rejected_candidates"], 1)
        self.assertFalse(set(map(transcript_digest, observed)) & set(protected))
        original_candidate = stream._candidate
        def forced_collision(family, turns, index, occurrence, attempt):
            return copy.deepcopy(banks[family][turns]) if attempt == 0 else original_candidate(family, turns, index, occurrence, attempt)
        with patch.object(stream, "_candidate", side_effect=forced_collision):
            stream.draw("color")
        self.assertEqual(stream.state["collisions"]["seen_pairs"], 1)
        self.assertEqual(stream.state["collisions"]["reserved_pairs"], 1)
        self.assertEqual(stream.state["collisions"]["rejected_candidates"], 2)

    def test_exact_tensor_resume_and_one_pass_replay_match_all_checkpoints(self):
        trainer = self.trainer()
        evidence = {0: trainer.stream_evidence()}
        trainer.step()
        evidence[1] = trainer.stream_evidence()
        saved = trainer.snapshot()
        buffer = io.BytesIO()
        torch.save(saved, buffer)
        buffer.seek(0)
        restored = self.trainer(payload=torch.load(buffer, weights_only=True))
        for step in (2, 3):
            self.compare(trainer.step(), restored.step())
            for field in ("weights", "optimizer", "samplers", "exposures"):
                self.compare(trainer.snapshot()[field], restored.snapshot()[field])
            self.assertEqual(trainer.stream_evidence(), restored.stream_evidence())
            evidence[step] = trainer.stream_evidence()
        self.assertEqual(evidence, replay_evidence(self.banks, "fresh", [], [0, 1, 2, 3], micro_batch_size=2))
        saved["realization"]["seen_transcripts"].clear()
        saved["realization"]["occurrences"]["color"][8][0] += 99
        self.assertEqual(trainer.stream_evidence(), restored.stream_evidence())

    def test_latest_observed_only_contains_consumed_recipes_and_regenerates_attempts(self):
        stream = RealizationStream(self.banks, mode="fresh", protected_transcripts=[], micro_batch_size=2)
        empty, before = stream.latest_observed_banks()
        self.assertFalse(empty)
        self.assertEqual(len(before["absent_recipes"]), 9)
        actual = {}
        for _ in range(4):
            rows, turns, _, _ = stream.draw("color")
            actual[turns] = rows
        latest, metadata = stream.latest_observed_banks()
        self.assertEqual(latest, {"color": actual})
        self.assertFalse(metadata["all_initial_realizations_encountered"])
        self.assertEqual(len(metadata["per_pair"]) + len(metadata["absent_recipes"]), 9)
        for row in metadata["per_pair"]:
            self.assertEqual(row["pair_sha256"], fingerprint_rows(latest[row["family"]][row["turns"]]))
        self.assertIn("not an unbiased", metadata["selection"])

    def test_actual_byte_target_histograms_and_caller_isolation(self):
        banks = copy.deepcopy(self.banks)
        stream = RealizationStream(banks, mode="fresh", protected_transcripts=[], micro_batch_size=2)
        banks["color"][8][0]["turns"][0]["text"] = "changed caller text"
        rows, turns, _, counts = stream.draw("color")
        stats = stream.state["statistics"]["color"][turns]
        self.assertEqual(counts["observation_bytes"], sum(len(turn["text"].encode("utf8")) for row in rows for turn in row["turns"]))
        self.assertEqual(counts["observation_tokens"] - counts["observation_bytes"], 4 * turns)
        self.assertEqual(stats["target_counts_by_turn"], [[sum(row["turns"][turn]["target"] == target for row in rows)
            for target in range(4)] for turn in range(turns)])
        self.assertEqual(stats["opposite_pair_counts_by_turn"][-1], 1)
        self.assertTrue(stats["increments"])
        self.assertTrue(stats["set_values"])
        self.assertTrue(stats["query_values"])

    def test_failed_update_cannot_continue_or_be_snapshotted(self):
        trainer = self.trainer()
        saved = trainer.snapshot()
        with patch.object(trainer.model, "forward", side_effect=RuntimeError("deliberate failure")):
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                trainer.step()
        for operation in (trainer.snapshot, trainer.stream_evidence, trainer.step):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                operation()
        trainer._restore(saved)
        reference = self.trainer(payload=saved)
        self.compare(trainer.step(), reference.step())
        self.assertEqual(trainer.stream_evidence(), reference.stream_evidence())

    def test_invalid_provenance_mode_occurrences_and_optimizer_are_rejected(self):
        original_hash = transcript_digest(self.banks["color"][8][0])
        with self.assertRaisesRegex(ValueError, "protected"):
            RealizationStream(self.banks, mode="fresh", protected_transcripts=[original_hash])
        with self.assertRaises(ValueError):
            self.trainer("unknown")
        trainer = self.trainer()
        trainer.step()
        payload = trainer.snapshot()
        with self.assertRaises(ValueError):
            self.trainer("fixed", payload=payload)
        for field in ("occurrences", "sampler", "histogram", "nan"):
            bad = copy.deepcopy(payload)
            if field == "occurrences":
                bad["realization"]["occurrences"]["color"][8][0] += 1
            elif field == "sampler":
                bad["realization"]["samplers"]["color"] = torch.zeros(2)
            elif field == "histogram":
                bad["realization"]["statistics"]["color"][8]["target_counts_by_turn"][0][0] += 1
            else:
                next(iter(bad["optimizer"]["state"].values()))["exp_avg"].flatten()[0] = float("nan")
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.trainer(payload=bad)


if __name__ == "__main__":
    unittest.main()
