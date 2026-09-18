"""Pure CPU metadata/recipe checks: no model construction, forward or update."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_loop_runtime_probe as probe
from experiments.foundation_provider import EVALUATION_SCHEMA


def envelope():
    response = dict(schema=EVALUATION_SCHEMA, wall_seconds=.25, weights_sha256="a"*64,
                    per_bank={"a": {"correct": 1, "seconds": .1}}, progress={"a": {"rate": .5}})
    return {"provider": {"learner": {"timing": {"retained_step_seconds": 2.,
                "materialization_seconds_included_in_step": .2},
                "weights": {"x": torch.tensor([1.])}, "optimizer": {"m": torch.tensor([2.])}, "cursor": 3},
                "pending": {"consumed_updates": 3}},
            "state": {"work": {"training_seconds": 2., "evaluation_seconds": 1., "physical_optimizer_updates": 3},
                "history": [{"response": response}], "references": {"same": response},
                "evaluation": {"cursor": 1, "response": response}, "other_seconds": 42.}}


def report():
    return dict(status="update_bound", publication_uncertain=False, starting_updates=0, ending_updates=0,
        retained_updates=0, durable_retained_updates=0, completed_updates=0, discarded_completed_updates=0,
        discarded_optimizer_updates=0, physical_optimizer_updates=0, drawn_episode_exposures=0,
        neural_attempted_episode_exposures=0, completed_microbatch_episode_exposures=0,
        evaluation_banks=0, evaluation_episode_exposures=0, failed_step_attempts=0,
        unknown_optimizer_attempts=0, unknown_practice_attempts=0, incomplete_evaluation_attempts=0,
        unknown_evaluation_episode_exposures=False, practice_reports=[], wall_seconds=.5)


def successful_step_fixture():
    families = {family: dict(rows_sha256=str(index)*64, recipes_sha256=str(index+3)*64,
        exposures={"episodes": 2, "observation_bytes": 100}) for index,family in enumerate(probe.FAMILIES)}
    bundle = dict(bundle_id=7, depth=1, turns=10, families=families)
    micro = [dict(family=f, depth=1, turns=10, **copy.deepcopy(families[f]), loss=1., action_loss=.3,
                  reply_loss=.4, observation_language_loss=.3) for f in probe.FAMILIES]
    step = dict(cursor=1, bundle_id=7, physical_optimizer_updates=1, retained_optimizer_updates=1,
        drawn_microbatches=3, neural_attempted_microbatches=3, completed_microbatches=3,
        wall_seconds=.3, step_seconds=.2, materialization_seconds=.01,
        loss=1., action_loss=.3, reply_loss=.4, observation_language_loss=.3,
        microbatches=micro, drawn_episode_exposures=6, neural_attempted_episode_exposures=6,
        completed_microbatch_episode_exposures=6)
    chunk = dict(starting_updates=0, ending_updates=1, completed_updates=1, physical_optimizer_updates=1,
        failed_step_attempts=0, unknown_optimizer_attempts=0, accounting_uncertain=False, restore_required=False,
        step_reports=[step], drawn_episode_exposures=6, neural_attempted_episode_exposures=6,
        completed_microbatch_episode_exposures=6)
    value = report()
    value.update(ending_updates=1, retained_updates=1, durable_retained_updates=1, completed_updates=1,
        physical_optimizer_updates=1, drawn_episode_exposures=6, neural_attempted_episode_exposures=6,
        completed_microbatch_episode_exposures=6, practice_reports=[chunk])
    return value, bundle


class FoundationLoopProbeTests(unittest.TestCase):
    def test_json_identity_roundtrip_preserves_betas_but_rejects_changed_recipe(self):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=.001)
        identity = {"optimizer": optimizer.state_dict(), "length_counts": {8:2,10:3,12:4}}
        self.assertIsInstance(identity["optimizer"]["param_groups"][0]["betas"], tuple)
        serialized = json.loads(json.dumps(identity))
        self.assertTrue(probe._json_same(identity, serialized))
        serialized["optimizer"]["param_groups"][0]["betas"][1] = .99
        self.assertFalse(probe._json_same(identity, serialized))
        # Actual reload evidence uses strict tree equality, not metadata normalization.
        self.assertFalse(probe._same_tree((.9,.999), [.9,.999]))

    def test_prescribed_decomposition_is_twenty_four_updates_and_same_partial_points(self):
        calls = {name: probe.operations(name) for name in probe.WORKERS}
        self.assertEqual(sum(kwargs.get("max_updates", 0) for items in calls.values() for _,kwargs in items), 24)
        self.assertEqual(calls["uninterrupted"], calls["repeat"])
        self.assertEqual(calls["partial-evaluation"], calls["uninterrupted"][:1])
        self.assertEqual(calls["split-practice"], calls["uninterrupted"][1:2])
        self.assertEqual(calls["resumed"], calls["uninterrupted"][2:])
        with self.assertRaises(ValueError): probe.operations("retry")

    def test_timing_normalization_is_alias_safe_precise_and_nonmutating(self):
        before = envelope(); other = copy.deepcopy(before)
        other["provider"]["learner"]["timing"]["retained_step_seconds"] = 8.
        other["state"]["work"]["training_seconds"] = 8.
        other["state"]["history"][0]["response"]["wall_seconds"] = 4.
        other["state"]["history"][0]["response"]["per_bank"]["a"]["seconds"] = 2.
        self.assertTrue(probe._same_tree(probe.normalize(before), probe.normalize(other)))
        self.assertEqual(before["state"]["history"][0]["response"]["wall_seconds"], .25)
        self.assertEqual(before["state"]["history"][0]["response"]["per_bank"]["a"]["seconds"], .1)
        for path in ("optimizer", "pending", "queue", "unrecognized_time"):
            changed = copy.deepcopy(before)
            if path == "optimizer": changed["provider"]["learner"]["optimizer"]["m"] += 1
            elif path == "pending": changed["provider"]["pending"]["consumed_updates"] = 2
            elif path == "queue": changed["state"]["evaluation"]["cursor"] = 0
            else: changed["state"]["other_seconds"] = 43.
            self.assertFalse(probe._same_tree(probe.normalize(before), probe.normalize(changed)))

    def test_invalid_or_undeclared_timing_cannot_be_ignored(self):
        for kind in ("nan", "inf", "negative", "unknown", "scope", "bank_timer"):
            value = envelope()
            if kind == "unknown": value["provider"]["learner"]["timing"]["another_seconds"] = 1
            elif kind == "scope": value["provider"]["learner"]["timing"]["materialization_seconds_included_in_step"] = 3
            elif kind == "bank_timer": value["state"]["history"][0]["response"]["per_bank"]["a"]["seconds"] = float("nan")
            else: value["state"]["history"][0]["response"]["wall_seconds"] = {"nan":float("nan"),"inf":float("inf"),"negative":-1}[kind]
            with self.subTest(kind=kind), self.assertRaises(ValueError): probe.normalize(value)

    def test_unknown_physical_work_is_not_replaced_with_zero(self):
        failed = report(); failed.update(status="failed", physical_optimizer_updates=None,
            unknown_optimizer_attempts=1, retained_updates=None, discarded_optimizer_updates=None,
            discarded_completed_updates=None, incomplete_evaluation_attempts=1,
            unknown_evaluation_episode_exposures=True)
        result = probe.accounting([report(), failed])
        self.assertIsNone(result["physical_optimizer_updates"])
        self.assertIsNone(result["retained_updates"])
        self.assertTrue(result["unknown_evaluation_episode_exposures"])
        with self.assertRaises(ValueError): probe.accounting([failed], require_success=True)
        changed = report(); changed["physical_optimizer_updates"] = True
        with self.assertRaises(ValueError): probe.accounting([changed])

    def test_canonical_step_reports_bind_recipes_counts_and_full_prefix(self):
        value, bundle = successful_step_fixture()
        self.assertTrue(probe.validate_operation(value, 0, 1, [bundle]))
        for kind in ("recipe", "bytes", "cursor", "family", "missing_step", "physical", "timer"):
            changed = copy.deepcopy(value); step = changed["practice_reports"][0]["step_reports"][0]
            if kind == "recipe": step["microbatches"][0]["recipes_sha256"] = "f"*64
            elif kind == "bytes": step["microbatches"][0]["exposures"]["observation_bytes"] += 1
            elif kind == "cursor": step["cursor"] = 2
            elif kind == "family": step["microbatches"].reverse()
            elif kind == "missing_step": changed["practice_reports"] = []
            elif kind == "physical": changed["physical_optimizer_updates"] = 2
            else: step["step_seconds"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError): probe.validate_operation(changed, 0, 1, [bundle])

    def test_claim_and_artifact_receipts_preserve_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/"proof.json"
            probe._write(path, {"original": True})
            with self.assertRaises(FileExistsError): probe._write(path, {"overwritten": True})
            self.assertEqual(probe._read(path), {"original": True})
            work = root/"uninterrupted"; work.mkdir()
            probe._write(work/"data.json", {"a": 1})
            probe._write(work/"result.json", dict(schema=probe.SCHEMA, name="uninterrupted", status="completed",
                artifact_sha256=probe._artifacts(work)))
            self.assertEqual(probe._receipt(root, "uninterrupted")["status"], "completed")
            (work/"data.json").write_text("changed")
            with self.assertRaises(ValueError): probe._receipt(root, "uninterrupted")
            with patch.object(probe, "_inputs", side_effect=AssertionError("must not get past occupied claim")):
                with self.assertRaises(FileExistsError): probe.worker(root, "uninterrupted")

    def test_prepare_existing_directory_cannot_launch_runtime(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, "_runtime", side_effect=AssertionError("runtime called")):
            with self.assertRaises(FileExistsError): probe.prepare(directory)

    def test_fixture_has_real_admitted_pairs_and_both_role_length_coverage(self):
        # CPU recipe generation only. Guard constructors to keep this suite non-neural.
        with patch.object(probe, "FoundationTrainer", side_effect=AssertionError("model constructed")), patch.object(
                probe, "FoundationPracticeProvider", side_effect=AssertionError("provider constructed")):
            value = probe.fixture(micro_batch_size=2)
        self.assertEqual(value["coverage"]["turns"], [8, 10, 12])
        self.assertEqual(value["coverage"]["evaluation_turns"], [8, 10, 12])
        self.assertEqual(len(value["coverage"]["overridden_bundles"]), 8)
        self.assertEqual(set(value["evaluation_banks"]), {"development", "retention"})
        for banks in value["evaluation_banks"].values():
            self.assertEqual(len(banks), 3)
            self.assertTrue(all(len(rows) == 2 for rows in banks.values()))
        self.assertTrue(set(value["admission_protected"]) <= set(value["protected"]))


if __name__ == "__main__": unittest.main(verbosity=2)
