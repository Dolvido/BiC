"""Prospective canonical bank preparation only; no model or study-file reads."""
from collections import Counter
import copy
import unittest
from unittest.mock import patch

from experiments import foundation_banks as banks_module
from experiments import foundation_curriculum as curriculum
from experiments.foundation_evidence import json_digest, prepare_training, reconstruct_anchor, training_cell
from experiments.foundation_plan import build_plan


class FoundationBankBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_plan(seed=501000001, ordering_seed=501000002,
                              stage_updates=64, micro_batch_size=2)
        cls.training, cls.transcripts = prepare_training(cls.plan, anchor_limit=2)
        cls.dev, cls.dev_manifest = cls.build()
        cls.audit, cls.audit_manifest = cls.build(role="audit", seed=503000001,
                                                 excluded=cls.dev_manifest["transcript_sha256"])

    @classmethod
    def build(cls, *, role="dev", seed=502000001, excluded=(), manifest=None, transcripts=None, pairs=1):
        return banks_module.build_evaluation(cls.plan, cls.training if manifest is None else manifest,
            training_transcripts=cls.transcripts if transcripts is None else transcripts,
            role=role, seed=seed, pairs_per_cell=pairs, excluded_transcripts=excluded)

    def test_expected_primitive_cells_and_no_phantom_depth_two_development(self):
        fresh = {name.removeprefix("fresh/") for name in self.dev if name.startswith("fresh/")}
        self.assertEqual(fresh, set(banks_module.expected_fresh_cells()))
        self.assertEqual(len(fresh), 63)
        self.assertEqual(len(self.dev), 90)
        self.assertEqual(len(self.audit), 99)
        for family in curriculum.FAMILIES:
            for turns in (8, 10, 12):
                for operator in ("copy", "advance"):
                    self.assertIn(f"fresh/{family}/d1/{operator}/t{turns}", self.dev)
                self.assertNotIn(f"composed/{family}/d2/composed/t{turns}", self.dev)
                self.assertIn(f"composed/{family}/d2/composed/t{turns}", self.audit)

    def test_actual_anchor_procedures_preserved_and_both_realizations_changed(self):
        for name, relations in self.dev_manifest["fresh_anchor_relations"].items():
            rows = self.dev[name]
            self.assertEqual(len(relations), 1)
            relation = relations[0]
            original = reconstruct_anchor(self.plan, relation["anchor"])
            self.assertEqual(json_digest(rows), relation["pair_sha256"])
            for before, after in zip(original, rows):
                for key in ("program_id", "structure_id", "query_ancestries", "depth", "primitive_shared"):
                    self.assertEqual(before[key], after[key])
                for key in ("seed", "depth", "turns", "structure_split", "procedure_attempt"):
                    self.assertEqual(before["recipe"][key], after["recipe"][key])
                self.assertNotEqual(before["recipe"]["naming_seed"], after["recipe"]["naming_seed"])
                self.assertNotEqual(before["recipe"]["value_seed"], after["recipe"]["value_seed"])
            self.assertNotEqual(relation["original_naming_map_sha256"], relation["naming_map_sha256"])
            self.assertNotEqual(relation["original_visible_names_sha256"], relation["visible_names_sha256"])
            self.assertFalse(set(relation["original_anonymous_values_sha256"]) & set(relation["anonymous_values_sha256"]))

    def test_every_pair_truth_and_supervised_partition_matches_role(self):
        for role, banks in (("dev", self.dev), ("audit", self.audit)):
            for name, rows in banks.items():
                self.assertTrue(curriculum.validate_pair(rows))
                self.assertTrue(all(row["split"] == role for row in rows))
                self.assertEqual(name.split("/", 1)[1], training_cell(rows[0]))
                self.assertEqual({row["turns"][-1]["target"] for row in rows}, {0, 1})
                for row in rows:
                    queries = curriculum.query_ancestries(row)
                    if role == "dev":
                        self.assertFalse(any(q["composed"] and q["legacy_structure_partition"] == "audit" for q in queries))
                    if name.startswith("composed/"):
                        self.assertEqual(queries[-1]["legacy_structure_partition"], role)

    def test_training_and_external_exclusions_are_global_and_bound_to_manifest(self):
        train, dev, audit = map(set, (self.transcripts, self.dev_manifest["transcript_sha256"],
                                     self.audit_manifest["transcript_sha256"]))
        self.assertFalse(train & dev or train & audit or dev & audit)
        self.assertEqual(len(dev), 180)
        self.assertEqual(len(audit), 198)
        self.assertEqual(self.audit_manifest["training_transcripts"],
                         {"count": len(train), "sha256": json_digest(sorted(train))})
        self.assertEqual(self.audit_manifest["external_exclusions"],
                         {"count": len(dev), "sha256": json_digest(sorted(dev))})
        self.assertEqual(self.audit_manifest["effective_exclusions"],
                         {"count": len(train | dev), "sha256": json_digest(sorted(train | dev))})
        blocked = self.dev_manifest["transcript_sha256"][0]
        rebuilt, manifest = self.build(excluded=[blocked])
        self.assertNotIn(blocked, manifest["transcript_sha256"])
        self.assertGreater(manifest["rejections"]["excluded_transcript"], 0)
        self.assertEqual(set(rebuilt), set(self.dev))

    def test_counts_and_byte_mixtures_derive_from_actual_rows(self):
        for name, rows in self.dev.items():
            record = self.dev_manifest["banks"][name]
            counts = Counter(str(turn["target"]) for row in rows for turn in row["turns"])
            self.assertEqual(record["target_counts"], {str(target): counts[str(target)] for target in range(4)})
            self.assertEqual(record["query_target_counts"], {str(target): counts[str(target)] for target in range(3)})
            self.assertEqual(record["sha256"], json_digest(rows))
            self.assertEqual(record["observation_utf8_bytes"], sum(len(t["text"].encode()) for row in rows for t in row["turns"]))
            self.assertEqual(record["reply_utf8_bytes"], sum(len(t["reply"].encode()) for row in rows for t in row["turns"]))
            self.assertEqual(sum(record["observed_operator_counts"].values()), 2*record["turns"])
            self.assertEqual(record["opposite_pair_total"], sum(record["opposite_pair_total_by_turn"]))
            self.assertEqual(record["final_opposite_pair_total"], 1)
            self.assertEqual(record["unique_transcripts"], 2)
            self.assertLessEqual(record["max_input_utf8_bytes"], 128)

    def test_determinism_and_callers_are_not_mutated(self):
        before = copy.deepcopy((self.plan, self.training, self.transcripts))
        again, manifest = self.build()
        self.assertEqual(again, self.dev)
        self.assertEqual(manifest, self.dev_manifest)
        self.assertEqual(before, (self.plan, self.training, self.transcripts))
        again[next(iter(again))][0]["turns"][0]["text"] = "Caller changed output"
        self.assertNotEqual(again, self.dev)

    def test_manifest_anchor_and_transcript_corruption_rejected(self):
        cases = []
        changed = copy.deepcopy(self.training); changed["schema"] = "unknown"; cases.append(changed)
        changed = copy.deepcopy(self.training); changed["curriculum_version"] = "unknown"; cases.append(changed)
        changed = copy.deepcopy(self.training); changed["plan_sha256"] = "0"*64; cases.append(changed)
        changed = copy.deepcopy(self.training); changed["unique_transcripts"] += 1; cases.append(changed)
        changed = copy.deepcopy(self.training); changed["anchors"].pop(next(iter(changed["anchors"]))); cases.append(changed)
        changed = copy.deepcopy(self.training); next(iter(changed["anchors"].values()))[0]["pair_sha256"] = "0"*64; cases.append(changed)
        changed = copy.deepcopy(self.training); refs = next(iter(changed["anchors"].values())); refs[1] = copy.deepcopy(refs[0]); cases.append(changed)
        for changed in cases:
            with self.assertRaises(ValueError): self.build(manifest=changed)
        for transcripts in (self.transcripts[:-1], list(reversed(self.transcripts)), self.transcripts+[self.transcripts[0]]):
            with self.assertRaises(ValueError): self.build(transcripts=transcripts)
        with self.assertRaisesRegex(ValueError, "insufficient independent anchors"):
            self.build(pairs=3)
        for excluded in (["invalid"], ["0"*64, "0"*64]):
            with self.assertRaises(ValueError): self.build(excluded=excluded)

    def test_bounded_exhaustion_fails_without_fallback(self):
        with patch.object(banks_module, "MAX_ATTEMPTS_PER_PAIR", 1):
            with self.assertRaisesRegex(ValueError, "exhausted bounded attempts"):
                self.build(excluded=self.dev_manifest["transcript_sha256"])


if __name__ == "__main__":
    unittest.main()
