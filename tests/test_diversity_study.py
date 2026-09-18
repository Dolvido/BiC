"""Study boundaries and independent world/naming factor contracts."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import diversity_study as study


class DiversityStudyTests(unittest.TestCase):
    def test_factor_slots_hold_other_factor_and_sample_count_fixed(self):
        indices = {arm: study.arm_indices(arm) for arm in study.ARMS}
        self.assertEqual({len(rows) for rows in indices.values()}, {2048})
        for index in range(2048):
            self.assertEqual(indices["w32-n1"][index][0], indices["w32-n8"][index][0])
            self.assertEqual(indices["w256-n1"][index][0], indices["w256-n8"][index][0])
            self.assertEqual(indices["w32-n8"][index][1], indices["w256-n8"][index][1])
        self.assertEqual([len(set(indices[arm])) for arm in study.ARMS], [32, 256, 256, 2048])
        for arm, slots in indices.items():
            counts = {pair: slots.count(pair) for pair in set(slots)}
            self.assertEqual(len(set(counts.values())), 1)
        with self.assertRaises(ValueError):
            study.arm_indices("invalid")

    def test_resume_requires_exact_schedule_endpoint_and_provenance(self):
        protocol = {"seed": 2601, "initial_weights_sha256": "test"}
        saved = {"schema": study.SCHEMA, "arm": "w32-n1", "protocol": protocol,
                 "initial_weights_sha256": "test",
                 "training": {"updates": 5, "family_updates": dict(zip(study.FAMILIES, (2, 2, 1))),
                              "samplers": study.expected_samplers(5)}}
        study.validate_resume(saved, protocol, "w32-n1")
        for mutation in (
            lambda row: row.update(arm="w32-n8"),
            lambda row: row.update(schema="old"),
            lambda row: row["protocol"].update(seed=1),
            lambda row: row["training"].update(updates=True),
            lambda row: row["training"].update(updates=3601),
            lambda row: row["training"]["family_updates"].update({study.FAMILIES[0]: 3}),
            lambda row: row["training"].update(samplers=study.expected_samplers(4)),
            lambda row: row.update(initial_weights_sha256="wrong"),
        ):
            candidate = copy.deepcopy(saved)
            mutation(candidate)
            with self.assertRaises(ValueError):
                study.validate_resume(candidate, protocol, "w32-n1")

    def test_frozen_bank_and_source_identity_checked_before_reading_rows(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "source").mkdir()
            (directory / "source" / "test.py").write_text("source")
            (directory / "banks.pt").write_bytes(b"not deserialized in identity check")
            hashes = {"test.py": study.file_hash(directory / "source" / "test.py")}
            protocol = {"schema": study.SCHEMA, "source_sha256": hashes,
                        "banks_file_sha256": study.file_hash(directory / "banks.pt")}
            (directory / "protocol.json").write_text(json.dumps(protocol))
            with patch.object(study, "source_hashes", return_value=hashes):
                self.assertEqual(study.load_protocol(directory), protocol)
                (directory / "banks.pt").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "banks changed"):
                    study.load_protocol(directory)
                (directory / "source" / "test.py").write_text("changed")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    study.load_protocol(directory)

    def test_a_shared_single_variant_rejects_the_entire_world_pair(self):
        from experiments import diverse_curriculum
        selections = []
        with patch.object(diverse_curriculum, "generate_world_pair", side_effect=lambda family, seed, **kwargs: {"seed": seed}), \
             patch.object(diverse_curriculum, "world_fingerprint", side_effect=lambda row: str(row["seed"])), \
             patch.object(study, "world_variants", side_effect=lambda row: {"shared", "other"} if row["seed"] == 0 else {str(row["seed"])}):
            rows = study.world_pool("conditional_logic", 0, 2, "dev", excluded_variants={"shared"}, diagnostics=selections)
        self.assertEqual(rows, [{"seed": 1}, {"seed": 2}])
        self.assertEqual(selections[0]["single_variant_overlap_pair_rejections"], 1)

    def test_alias_normalization_retains_structure_but_removes_consistent_renaming(self):
        left = {"turns": [{"text": "dax is red."}, {"text": "Is dax red?"}]}
        renamed = {"turns": [{"text": "wug is red."}, {"text": "Is wug red?"}]}
        changed_role = {"turns": [{"text": "wug is red."}, {"text": "Is dax red?"}]}
        self.assertEqual(study.normalized_transcript(left), study.normalized_transcript(renamed))
        self.assertNotEqual(study.normalized_transcript(left), study.normalized_transcript(changed_role))


if __name__ == "__main__":
    unittest.main()
