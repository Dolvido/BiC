"""Canonical, non-neural equivalence fixtures; never read formal study data."""
import copy
import json
import pickle
import random
import unittest
from unittest.mock import patch

from experiments import foundation_canonical_validation as cached
from experiments import foundation_curriculum as original
from experiments import foundation_plan as planning


class IntSubclass(int):
    pass


class StrSubclass(str):
    pass


class DictSubclass(dict):
    pass


class CanonicalValidationTests(unittest.TestCase):
    validators = []
    fixture_generator_calls = 0
    reference_validator_calls = 0
    plan_pair_materializations = 0

    @classmethod
    def pair(cls, family="color", seed=992100001, **options):
        cls.fixture_generator_calls += 1
        return original.generate_pair(family, seed, **{"depth": 0, **options})

    @classmethod
    def setUpClass(cls):
        cls.grid = [cls.pair(family, 992110000 + depth, depth=depth, turns=turns)
            for family in original.FAMILIES for depth in original.DEPTHS for turns in original.TURN_BUCKETS]

    @classmethod
    def tearDownClass(cls):
        fields = ("canonical_generation_attempts", "canonical_pairs_regenerated",
                  "candidate_rows_attempted", "candidate_rows_validated", "pairs_validated", "cache_evictions")
        print("CANONICAL_VALIDATION_ACCOUNTING=" + json.dumps({
            "fixture_public_generator_calls": cls.fixture_generator_calls,
            "reference_validator_calls": cls.reference_validator_calls,
            "plan_pair_materialization_calls": cls.plan_pair_materializations,
            "prototype_totals": {key: sum(v.stats[key] for v in cls.validators) for key in fields},
            "max_observed_cache_entries": max(v.stats["peak_cache_entries"] for v in cls.validators),
            "max_observed_cache_payload_bytes": max(v.stats["peak_cache_bytes"] for v in cls.validators),
            "scope": "Generator/validation calls only. Public generation may use existing generator caches; original reference validation internally regenerates expected rows. No neural inference, optimizer, GPU or formal data."}, sort_keys=True))

    def validator(self, **options):
        value = cached.CanonicalValidator(**options); self.validators.append(value); return value

    def old_accepts(self, pair):
        type(self).reference_validator_calls += 1
        try:
            return original.validate_pair(pair)
        except (ValueError, TypeError, KeyError):
            return False

    def assert_equivalent(self, validator, pair, expected):
        self.assertEqual(self.old_accepts(pair), expected)
        try:
            accepted = validator.validate_pair(pair)
        except (ValueError, TypeError, KeyError):
            accepted = False
        self.assertEqual(accepted, expected)

    def test_all_domains_depths_lengths_cold_warm_and_detached_inputs(self):
        validator = self.validator(); before = json.dumps(self.grid, sort_keys=True)
        for pair in self.grid:
            self.assertTrue(self.old_accepts(pair))
        self.assertTrue(validator.validate_pairs(self.grid))
        first = validator.stats
        self.assertEqual(first["canonical_pairs_regenerated"], 54)
        self.assertEqual(first["candidate_rows_validated"], 108)
        self.assertEqual(first["cache_hits"], 54)
        self.assertTrue(validator.validate_pairs(tuple(tuple(pair) for pair in self.grid)))
        second = validator.stats
        self.assertEqual(second["canonical_pairs_regenerated"], 54)
        self.assertEqual(second["candidate_rows_validated"], 216)
        self.assertEqual(second["cache_hits"], 162)
        self.assertEqual(json.dumps(self.grid, sort_keys=True), before)

    def test_all_admission_roles_and_familiar_depth_two_development(self):
        validator = self.validator()
        for family in original.FAMILIES:
            for split in ("dev", "audit"):
                for depth in (0, 1, 2, 3, 5):
                    options = {"structure_split": "train"} if depth == 2 and split == "dev" else {}
                    pair = self.pair(family, 992120000+depth, depth=depth, turns=10, split=split, **options)
                    self.assert_equivalent(validator, pair, True)

    def test_semantic_provenance_and_pair_tampering_match_original_rejection(self):
        validator = self.validator(); pair = self.pair("count", 992130001, depth=3)
        self.assertTrue(validator.validate_pair(pair))
        changes = (
            lambda r:r.update(version="other"), lambda r:r.update(id="0"*64),
            lambda r:r.update(structure_id="0"*64), lambda r:r.update(extra="unbound"),
            lambda r:r["recipe"].update(value_seed=r["recipe"]["value_seed"]+1),
            lambda r:r["recipe"].update(procedure_attempt=r["recipe"]["procedure_attempt"]+1),
            lambda r:r["turns"][-1].update(target=2,reply=original.REPLIES[2]),
            lambda r:r["turns"][0].update(text=r["turns"][0]["text"]+" "),
            lambda r:r["turns"][0]["observations"].update(tokens=[1]),
            lambda r:r["query_ancestries"][-1].update(legacy_structure_partition="audit"),
            lambda r:r.update(depth=True), lambda r:r.update(variant=False),
            lambda r:r.update(primitive_shared=1),
        )
        for change in changes:
            rows = copy.deepcopy(pair); change(rows[0])
            with self.subTest(change=change): self.assert_equivalent(validator, rows, False)
        for rows in ([pair[0]], pair[::-1], [pair[0],pair[0]],
                     [pair[0],self.pair("count",992130002,depth=3)[1]]):
            self.assert_equivalent(validator, rows, False)

    def test_invalid_recipe_equality_aliases_reject_before_cache_lookup(self):
        validator = self.validator(); pair = self.pair("color",1,depth=1,naming_seed=1,value_seed=1)
        validator.validate_pair(pair)
        changes = []
        for name in ("seed", "naming_seed", "value_seed", "depth", "turns", "procedure_attempt"):
            for value in (float(pair[0]["recipe"][name]), IntSubclass(pair[0]["recipe"][name])):
                changes.append(lambda r,n=name,v=value:r["recipe"].update({n:v}))
        for name in ("seed", "naming_seed", "value_seed", "depth"):
            changes.append(lambda r,n=name:r["recipe"].update({n:True}))
        changes.extend((lambda r:r.update(family=StrSubclass("color")),
                        lambda r:r.update(split=StrSubclass("train")),
                        lambda r:r["recipe"].update(structure_split=StrSubclass("shared")),
                        lambda r:r.update(recipe=DictSubclass(r["recipe"]))))
        for change in changes:
            rows=copy.deepcopy(pair); change(rows[0]); before=validator.stats
            self.assert_equivalent(validator,rows,False)
            self.assertEqual(validator.stats["cache_hits"],before["cache_hits"])
            self.assertEqual(validator.stats["cache_misses"],before["cache_misses"])

    def test_original_json_normalization_is_preserved_outside_strict_recipe(self):
        pair=copy.deepcopy(self.grid[0]); pair[0]["turns"]=tuple(pair[0]["turns"])
        pair[0]["turns"][0]["observations"]["tokens"]=tuple(pair[0]["turns"][0]["observations"]["tokens"])
        pair[0]["id"]=StrSubclass(pair[0]["id"])
        self.assert_equivalent(self.validator(),pair,True)

    def test_caller_alias_mutation_cannot_poison_cached_truth_or_returned_metadata(self):
        pair=self.pair("switch",992140001,depth=5,turns=12); original_pair=copy.deepcopy(pair)
        validator=self.validator(); validator.validate_pair(pair)
        stats=validator.stats; stats["cache_entries"]=999
        identity=validator.identity; identity["source_sha256"].clear()
        pair[0]["turns"][-1]["reply"]="Yes." if pair[0]["turns"][-1]["reply"]!="Yes." else "No."
        self.assert_equivalent(validator,pair,False)
        self.assertTrue(validator.validate_pair(original_pair))
        self.assertEqual(validator.stats["canonical_pairs_regenerated"],1)
        self.assertTrue(validator.identity["source_sha256"])
        self.assertEqual(validator.stats["cache_entries"],1)
        self.assertEqual(self.pair("switch",992140001,depth=5,turns=12),original_pair)

    def test_names_only_override_candidates_preserve_exact_validation(self):
        plan=planning.build_plan(seed=992150001,stage_updates=10,final_updates=6,
                                 micro_batch_size=2,rehearsal_every=2)
        before=random.getstate(); validator=self.validator()
        for family in original.FAMILIES:
            left=planning.materialize_pair(plan,0,family,0,attempt=0)
            right=planning.materialize_pair(plan,0,family,0,attempt=1)
            type(self).plan_pair_materializations += 2
            self.assertNotEqual(left[0]["recipe"]["naming_seed"],right[0]["recipe"]["naming_seed"])
            self.assertEqual(left[0]["recipe"]["value_seed"],right[0]["recipe"]["value_seed"])
            self.assertEqual([r["program_id"] for r in left],[r["program_id"] for r in right])
            for pair in (left,right): self.assert_equivalent(validator,pair,True)
        self.assertEqual(random.getstate(),before)
        self.assertEqual(validator.stats["canonical_pairs_regenerated"],6)

    def test_entry_lru_eviction_and_regeneration(self):
        a,b,c=self.grid[:3]; validator=self.validator(max_entries=2)
        validator.validate_pairs([a,b,a,c])
        self.assertEqual(validator.stats["canonical_pairs_regenerated"],3)
        self.assertEqual(validator.stats["cache_evictions"],1)
        validator.validate_pair(a)
        self.assertEqual(validator.stats["canonical_pairs_regenerated"],3)
        validator.validate_pair(b)
        self.assertEqual(validator.stats["canonical_pairs_regenerated"],4)
        self.assertEqual(validator.stats["peak_cache_entries"],2)

    def test_payload_byte_cap_evictions_and_oversize_bypass(self):
        a,b=self.grid[:2]
        one,two=self.validator(),self.validator(); one.validate_pair(a);two.validate_pair(b)
        cap=max(one.stats["cache_bytes"],two.stats["cache_bytes"])
        validator=self.validator(max_cache_bytes=cap);validator.validate_pairs([a,b,a])
        self.assertEqual(validator.stats["cache_evictions"],2)
        self.assertLessEqual(validator.stats["peak_cache_bytes"],cap)
        tiny=self.validator(max_cache_bytes=1); tiny.validate_pairs([a,a])
        self.assertEqual(tiny.stats["cache_entries"],0)
        self.assertEqual(tiny.stats["oversize_entries_not_cached"],4)
        self.assertEqual(tiny.stats["canonical_pairs_regenerated"],4)

    def test_batch_source_checks_are_constant_not_per_row(self):
        validator=self.validator()
        with patch.object(cached,"source_hashes",wraps=cached.source_hashes) as hashes:
            self.assertTrue(validator.validate_pairs(self.grid))
            self.assertEqual(hashes.call_count,2)

    def test_source_drift_before_or_after_batch_fails_closed_and_clears_cache(self):
        for boundary in ("before","after"):
            validator=self.validator();validator.validate_pair(self.grid[0]); hashes=cached.source_hashes()
            changed={**hashes,"experiments/foundation_curriculum.py":"0"*64}
            sequence=[changed] if boundary=="before" else [hashes,changed]
            with patch.object(cached,"source_hashes",side_effect=sequence):
                with self.assertRaisesRegex(RuntimeError,"identity changed"):
                    validator.validate_pairs(self.grid[:2])
            self.assertTrue(validator.stats["poisoned"])
            self.assertEqual(validator.stats["cache_entries"],0)
            self.assertEqual(validator.stats["batches_failed"],1)
            with self.assertRaises(RuntimeError):validator.validate_pairs([])
        with patch.object(cached,"source_hashes",return_value=changed):
            with self.assertRaisesRegex(RuntimeError,"after module import"):cached.CanonicalValidator()

    def test_finite_batch_limits_failed_work_and_no_persistent_loader(self):
        validator=self.validator(max_batch_pairs=2)
        self.assertTrue(validator.validate_pairs([]))
        for bad in (iter([self.grid[0]]),{},None,[*self.grid[:3]]):
            with self.assertRaises(ValueError):validator.validate_pairs(bad)
        corrupted=copy.deepcopy(self.grid[1]);corrupted[0]["id"]="0"*64
        with self.assertRaises(ValueError):validator.validate_pairs([self.grid[0],corrupted])
        self.assertEqual(validator.stats["pairs_validated"],1)
        self.assertEqual(validator.stats["candidate_rows_validated"],2)
        self.assertEqual(validator.stats["candidate_rows_attempted"],3)
        self.assertEqual(validator.stats["batches_failed"],5)
        with self.assertRaises(TypeError):pickle.dumps(validator)
        for options in ({"max_entries":True},{"max_entries":0},{"max_cache_bytes":0},
                        {"max_cache_bytes":1.0},{"max_batch_pairs":0}):
            with self.assertRaises(ValueError):cached.CanonicalValidator(**options)

    def test_unreadable_source_poisoning_survives_later_file_recovery(self):
        for boundary in ("before", "after"):
            validator=self.validator(); validator.validate_pair(self.grid[0])
            values=[OSError("source temporarily unreadable")]
            if boundary=="after":values.insert(0,cached.source_hashes())
            with patch.object(cached,"source_hashes",side_effect=values):
                with self.assertRaisesRegex(OSError,"temporarily unreadable"):
                    validator.validate_pairs(self.grid[:2])
            self.assertTrue(validator.stats["poisoned"])
            self.assertEqual(validator.stats["cache_bytes"],0)
            self.assertEqual(validator.stats["batches_failed"],1)
            with self.assertRaises(RuntimeError):validator.validate_pair(self.grid[0])


if __name__ == "__main__":
    unittest.main()
