"""Check evaluator import integrity, sample accounting and output isolation."""
import copy
import json
from pathlib import Path
import socket
import tempfile
import unittest

from experiments.evaluate_tutor_sample import (
    metrics, network_disabled, output_location, summarize, validated_lessons,
)
from experiments.local_tutor_pilot import prepare, read_protocol
from experiments.local_tutor_validation import GRAMMAR, canonical_json, validate_candidate


class TutorSampleEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        prepare(self.root / "frozen")
        self.protocol = self.root / "frozen" / "protocol.json"
        self.spec = read_protocol(self.protocol)["specs"][0]
        candidate = {
            "lesson_id": self.spec["lesson_id"], "relation": self.spec["relation"],
            "instruction": GRAMMAR["choose_item"][self.spec["relation"]].format(
                quoted_name=json.dumps(self.spec["name"])),
        }
        self.record = validate_candidate(candidate, self.spec)
        self.record["envelope_sha256"] = "a" * 64
        self.record["spec"] = dict(self.spec, prompt=self.record["instruction"],
                                  **{key: self.record[key] for key in ("target", "reply", "category")})
        self.accepted = self.root / "accepted.jsonl"

    def write_records(self, records):
        self.accepted.write_text("".join(canonical_json(row) + "\n" for row in records), encoding="utf-8")

    def test_revalidation_retains_valid_record_and_allows_empty_sample(self):
        self.write_records([self.record])
        protocol, records = validated_lessons(self.protocol, self.accepted)
        self.assertEqual(protocol["count"], 50)
        self.assertEqual(records, [self.record])
        self.write_records([])
        self.assertEqual(validated_lessons(self.protocol, self.accepted)[1], [])

    def test_scene_and_label_tampering_and_duplicate_id_fail(self):
        for field, value in (("target", 10), ("query", 999999), ("known", 1)):
            record = copy.deepcopy(self.record)
            record["spec"][field] = value
            self.write_records([record])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "changed"):
                validated_lessons(self.protocol, self.accepted)
        self.write_records([self.record, self.record])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validated_lessons(self.protocol, self.accepted)

    def test_decision_hash_tampering_fails(self):
        record = copy.deepcopy(self.record)
        record["candidate_sha256"] = "b" * 64
        self.write_records([record])
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validated_lessons(self.protocol, self.accepted)

    def test_empty_groups_and_stop_baseline_are_explicit(self):
        self.assertIsNone(metrics([])["action_accuracy"])
        rows = [{"category": "unknown_label", "relation": "find", "phrase_family": "choose_item",
                 "target": 10, "action": 10, "action_correct": True,
                 "reply_exact": False, "joint_correct": False}]
        result = summarize(rows)
        self.assertEqual(result["overall"]["action_accuracy"], 1)
        self.assertEqual(result["overall"]["all_stop_accuracy"], 1)
        self.assertEqual(result["overall"]["joint_accuracy"], 0)
        self.assertEqual(result["valid_selections"]["episodes"], 0)
        self.assertIsNone(result["relations"]["left"]["reply_exact_accuracy"])

    def test_network_guard_blocks_connections_and_restores_socket(self):
        original = socket.create_connection
        with network_disabled():
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                socket.create_connection(("127.0.0.1", 11434))
        self.assertIs(socket.create_connection, original)

    def test_output_cannot_overwrite_or_touch_release(self):
        existing = self.root / "old.json"
        existing.write_text("{}")
        with self.assertRaises(FileExistsError):
            output_location(existing, [])
        from experiments.evaluate_tutor_sample import ROOT
        with self.assertRaisesRegex(ValueError, "checkpoint directory"):
            output_location(ROOT / "runs/retention-v06/new-evaluation.json", [])
        self.assertEqual(output_location(self.root / "fresh.json", []), self.root / "fresh.json")


if __name__ == "__main__":
    unittest.main()

