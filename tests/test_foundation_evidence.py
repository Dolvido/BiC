"""Canonical lesson evidence tests; no model inference or optimizer work."""
from collections import Counter
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from experiments.foundation_curriculum import FAMILIES, validate_pair
from experiments.foundation_plan import build_plan, materialize_bundle


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def observation_sha(row):
    return canonical_sha([turn["text"] for turn in row["turns"]])


def observed_counts(rows):
    """Count actual raw UTF-8 targets independently of manifest aggregation."""
    turns = [turn for row in rows for turn in row["turns"]]
    observed = sum(len(turn["text"].encode()) for turn in turns)
    replied = sum(len(turn["reply"].encode()) for turn in turns)
    targets = Counter(str(turn["target"]) for turn in turns)
    return {"episodes": len(rows), "turns": len(turns),
            "observation_bytes": observed, "observation_tokens": observed + 2 * len(turns),
            "reply_target_bytes": replied, "reply_target_tokens": replied + len(turns),
            "target_counts": {str(label): targets[str(label)] for label in range(4)},
            "query_target_counts": {str(label): targets[str(label)] for label in range(3)},
            "final_opposite_pairs": sum({rows[index]["turns"][-1]["target"],
                                          rows[index + 1]["turns"][-1]["target"]} == {0, 1}
                                         for index in range(0, len(rows), 2))}


class FoundationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.foundation_evidence import prepare_training
        cls.plan = build_plan(seed=402000001, ordering_seed=402000002, stage_updates=10,
                              rehearsal_every=2, final_updates=6, micro_batch_size=2)
        cls.actual = {str(index): materialize_bundle(cls.plan, index)
                      for index in cls.plan["schedules"]["curriculum"]}
        cls.manifest, cls.transcripts = prepare_training(cls.plan, anchor_limit=2)

    def test_full_tiny_plan_manifest_matches_actual_rows_bytes_targets_and_unique_texts(self):
        from experiments.foundation_curriculum import VERSION
        from experiments.foundation_evidence import SCHEMA
        manifest = self.manifest
        self.assertEqual(manifest["schema"], SCHEMA)
        self.assertEqual(manifest["curriculum_version"], VERSION)
        self.assertEqual(manifest["plan_sha256"], canonical_sha(self.plan))
        self.assertEqual(manifest["protected_count"], 0)
        self.assertEqual(manifest["protected_sha256"], canonical_sha([]))
        self.assertEqual(set(manifest["bundles"]), set(self.actual))
        actual_transcripts = []
        for bundle_id, lesson in self.actual.items():
            for family, rows in lesson.items():
                actual_transcripts.extend(map(observation_sha, rows))
                entry = manifest["bundles"][bundle_id]
                self.assertEqual(entry["family_sha256"][family], canonical_sha(rows))
                self.assertEqual(entry["family_counts"][family], observed_counts(rows))
        self.assertEqual(len(actual_transcripts), 66 * 3 * 2)
        self.assertEqual(len(actual_transcripts), len(set(actual_transcripts)))
        self.assertEqual(self.transcripts, sorted(actual_transcripts))
        self.assertEqual(manifest["unique_transcripts"], len(actual_transcripts))
        self.assertEqual(manifest["transcript_sha256"], canonical_sha(self.transcripts))
        for family in FAMILIES:
            rows = [row for lesson in self.actual.values() for row in lesson[family]]
            self.assertEqual(manifest["totals"][family], observed_counts(rows))
            self.assertEqual(manifest["totals"][family]["episodes"], 132)
            self.assertEqual(manifest["totals"][family]["final_opposite_pairs"], 66)

    def test_cells_separate_direct_copy_advance_and_composed_with_actual_totals(self):
        from experiments import composition_curriculum as legacy
        from experiments import foundation_curriculum as curriculum
        from experiments.foundation_evidence import training_cell
        actual_cells = {}
        primitive_kinds = {family: set() for family in FAMILIES}
        for lesson in self.actual.values():
            for family, rows in lesson.items():
                for row in rows:
                    if row["depth"] == 0:
                        kind = "direct"
                    elif row["depth"] == 1:
                        signature = legacy._ancestry_records(curriculum._program(row))[-1]["signature"]
                        transformations = [step[0] for step in signature if step[0] in ("copy", "advance")]
                        self.assertEqual(len(transformations), 1)
                        kind = transformations[0]
                        primitive_kinds[family].add(kind)
                    else:
                        kind = "composed"
                    expected = f"{family}/d{row['depth']}/{kind}/t{len(row['turns'])}"
                    self.assertEqual(training_cell(row), expected)
                    actual_cells.setdefault(expected, []).append(row)
        self.assertEqual(set(actual_cells), set(self.manifest["cells"]))
        for family in FAMILIES:
            self.assertEqual(primitive_kinds[family], {"copy", "advance"})
        for cell, rows in actual_cells.items():
            self.assertEqual(self.manifest["cells"][cell], observed_counts(rows))

    def test_anchor_references_are_capped_admitted_pairs_and_reconstruct_exactly(self):
        from experiments.foundation_evidence import reconstruct_anchor, training_cell
        self.assertEqual(self.manifest["anchor_limit"], 2)
        self.assertEqual(set(self.manifest["anchors"]), set(self.manifest["cells"]))
        capped = False
        for cell, references in self.manifest["anchors"].items():
            expected_count = min(2, self.manifest["cells"][cell]["final_opposite_pairs"])
            self.assertEqual(len(references), expected_count)
            capped |= self.manifest["cells"][cell]["final_opposite_pairs"] > len(references)
            for reference in references:
                self.assertEqual(set(reference), {"bundle_id", "pair_index", "family", "pair_sha256"})
                rows = self.actual[str(reference["bundle_id"])][reference["family"]]
                start = 2 * reference["pair_index"]
                expected = rows[start:start + 2]
                pair = reconstruct_anchor(self.plan, reference)
                self.assertEqual(pair, expected)
                self.assertTrue(validate_pair(pair))
                self.assertEqual(reference["pair_sha256"], canonical_sha(pair))
                self.assertEqual(training_cell(pair[0]), cell)
        self.assertTrue(capped)

    def test_changing_anchor_cap_preserves_complete_exposure_evidence(self):
        from experiments.foundation_evidence import prepare_training, reconstruct_anchor
        small, transcripts = prepare_training(self.plan, anchor_limit=1)
        self.assertEqual(transcripts, self.transcripts)
        for field in ("bundles", "cells", "totals", "schedule_totals", "unique_transcripts", "transcript_sha256"):
            self.assertEqual(small[field], self.manifest[field])
        for cell, references in small["anchors"].items():
            self.assertEqual(references, self.manifest["anchors"][cell][:1])
        reference = next(iter(small["anchors"].values()))[0]
        pair = reconstruct_anchor(self.plan, reference)
        pair[0]["turns"][0]["text"] = "caller mutation"
        self.assertNotEqual(pair, reconstruct_anchor(self.plan, reference))

    def test_nonzero_anchor_pair_index_selects_the_complete_later_pair(self):
        from experiments.foundation_evidence import reconstruct_anchor
        plan = build_plan(**{**self.plan["config"], "micro_batch_size": 4})
        lesson = materialize_bundle(plan, 0)
        for family, rows in lesson.items():
            pair = rows[2:4]
            reference = {"bundle_id": 0, "pair_index": 1, "family": family,
                         "pair_sha256": canonical_sha(pair)}
            self.assertEqual(reconstruct_anchor(plan, reference), pair)
            self.assertNotEqual(pair, rows[:2])
            with self.assertRaises(ValueError):
                reconstruct_anchor(plan, {**reference, "pair_index": 0})

    def test_both_orders_match_prefix_final_and_total_counts_despite_different_stage_order(self):
        orders = self.manifest["schedule_totals"]
        self.assertEqual(orders["curriculum"], orders["mixed"])
        for arm, ids in self.plan["schedules"].items():
            for phase in ("foundation", "mixed"):
                phase_ids = [index for index in ids if self.plan["bundles"][str(index)]["phase"] == phase]
                self.assertEqual(len(phase_ids), 60 if phase == "foundation" else 6)
                for family in FAMILIES:
                    rows = [row for index in phase_ids for row in self.actual[str(index)][family]]
                    self.assertEqual(orders[arm][phase][family], observed_counts(rows))
        def initial_stages(arm):
            return [self.plan["bundles"][str(index)]["stage"]
                    for index in self.plan["schedules"][arm][:10]]
        self.assertNotEqual(initial_stages("curriculum"), initial_stages("mixed"))

    def test_ordering_changes_plan_binding_but_not_admitted_lessons_or_anchors(self):
        from experiments.foundation_evidence import prepare_training
        changed = build_plan(**{**self.plan["config"], "ordering_seed": 402000099})
        manifest, transcripts = prepare_training(changed, anchor_limit=2)
        self.assertNotEqual(manifest["plan_sha256"], self.manifest["plan_sha256"])
        self.assertEqual(transcripts, self.transcripts)
        for field in set(manifest) - {"plan_sha256"}:
            self.assertEqual(manifest[field], self.manifest[field])

    def test_protected_exclusions_are_exact_and_canonical(self):
        from experiments.foundation_evidence import prepare_training
        unused = ["f" * 64, "0" * 64]
        self.assertTrue(set(unused).isdisjoint(self.transcripts))
        manifest, transcripts = prepare_training(self.plan, protected_transcripts=unused, anchor_limit=2)
        self.assertEqual(manifest["protected_count"], 2)
        self.assertEqual(manifest["protected_sha256"], canonical_sha(sorted(unused)))
        self.assertEqual(transcripts, self.transcripts)
        collision = observation_sha(self.actual["0"][FAMILIES[0]][0])
        with self.assertRaisesRegex(ValueError, "collides"):
            prepare_training(self.plan, protected_transcripts=[collision])

    def test_bad_protection_types_anchor_limits_and_plans_fail_before_generation(self):
        from experiments.foundation_evidence import prepare_training
        invalid = (None, "0" * 64, {"digest": "0" * 64}, [123], ["F" * 64], ["0" * 63],
                   ["g" * 64], ["0" * 64, "0" * 64], iter(["0" * 64]))
        with patch("experiments.foundation_plan._materialize_validated_bundle", side_effect=AssertionError("unexpected generation")):
            for value in invalid:
                with self.subTest(protected=value), self.assertRaises(ValueError):
                    prepare_training(self.plan, protected_transcripts=value)
            for limit in (0, -1, True, 1.0, None, 4097):
                with self.subTest(anchor_limit=limit), self.assertRaises(ValueError):
                    prepare_training(self.plan, anchor_limit=limit)
            changed = copy.deepcopy(self.plan)
            changed["bundles"]["0"]["depth"] = 5
            with self.assertRaises(ValueError):
                prepare_training(changed)

    def test_duplicate_canonical_observations_across_bundles_are_rejected(self):
        from experiments import foundation_plan as planning
        from experiments.foundation_evidence import prepare_training
        materialize = planning._materialize_validated_bundle
        first = self.actual["0"][FAMILIES[0]]
        collision_id = next(index for index in self.plan["schedules"]["curriculum"] if index > 0
                            and self.plan["bundles"][str(index)]["depth"] == 0
                            and self.plan["bundles"][str(index)]["turns"] == 8)

        def repeated_pair(plan, index):
            result = materialize(plan, index)
            if index == collision_id:
                result[FAMILIES[0]] = copy.deepcopy(first)
            return result

        with patch.object(planning, "_materialize_validated_bundle", side_effect=repeated_pair):
            with self.assertRaisesRegex(ValueError, "collides"):
                prepare_training(self.plan)

    def test_anchor_and_row_tampering_cannot_be_reconstructed_as_admitted_evidence(self):
        from experiments.foundation_evidence import reconstruct_anchor, training_cell
        original = next(iter(self.manifest["anchors"].values()))[0]
        changes = ({"family": "unknown"}, {"family": None}, {"pair_sha256": "0" * 64},
                   {"pair_index": -1}, {"pair_index": 1}, {"pair_index": True},
                   {"bundle_id": -1}, {"bundle_id": 66}, {"bundle_id": "0"},
                   {"extra": "not canonical"})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                reconstruct_anchor(self.plan, {**original, **change})
        missing = copy.deepcopy(original)
        missing.pop("pair_sha256")
        with self.assertRaises(ValueError):
            reconstruct_anchor(self.plan, missing)
        changed_seed = build_plan(**{**self.plan["config"], "seed": self.plan["config"]["seed"] + 1})
        with self.assertRaisesRegex(ValueError, "admitted training recipe"):
            reconstruct_anchor(changed_seed, original)
        row = copy.deepcopy(self.actual["0"][FAMILIES[0]][0])
        row["turns"][-1]["target"] = 2
        with self.assertRaises(ValueError):
            training_cell(row)


if __name__ == "__main__":
    unittest.main()
