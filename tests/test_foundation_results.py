"""Pure canonical verification fixtures plus one tiny untrained CPU score."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_results as results
from experiments import foundation_study as study
from experiments.foundation_evidence import prepare_training
from experiments.foundation_plan import build_plan
from experiments.foundation_training import replay_evidence


class FoundationResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_plan(seed=404000001, ordering_seed=404000002, stage_updates=10,
                              rehearsal_every=2, final_updates=6, micro_batch_size=2)
        cls.manifest, _ = prepare_training(cls.plan, anchor_limit=1)
        cls.replays = {arm: replay_evidence(cls.plan, arm, 66, include_bundles=True) for arm in study.ARMS}
        cls.events = {arm: cls.events_for(value["bundles"]) for arm, value in cls.replays.items()}

    @staticmethod
    def events_for(bundles):
        events = []
        for index, bundle in enumerate(bundles):
            microbatches = [{"family": family, "depth": bundle["depth"], "turns": bundle["turns"],
                **copy.deepcopy(record), **dict.fromkeys(results.LOSSES, .25)}
                for family, record in bundle["families"].items()]
            report = {"bundle_id": bundle["bundle_id"], "cursor": index + 1, "microbatches": microbatches,
                **dict.fromkeys(results.LOSSES, .25), "step_seconds": .5, "materialization_seconds": .1,
                "wall_seconds": .6, "physical_optimizer_updates": 1, "retained_optimizer_updates": 1,
                "drawn_microbatches": 3, "neural_attempted_microbatches": 3, "completed_microbatches": 3,
                "drawn_episode_exposures": 6, "neural_attempted_episode_exposures": 6,
                "completed_microbatch_episode_exposures": 6}
            events.extend([{"event": "started", "cursor": index, "utc": "2026-09-17T00:00:00+00:00"},
                           {"event": "completed", "report": report}])
        return events

    def validate(self, events, arm="curriculum", manifest=None):
        return results.validate_journal(events, self.plan, arm, manifest or self.manifest,
                                        self.replays[arm]["bundles"])

    def test_both_canonical_orders_have_exact_complete_physical_totals(self):
        summaries = []
        for arm in study.ARMS:
            value = self.validate(self.events[arm], arm)
            self.assertEqual(value["physical_optimizer_updates"], 66)
            for key in results.PHYSICAL[1:]:
                self.assertEqual(value[key], 396)
            self.assertAlmostEqual(value["retained_step_seconds"], 33.)
            self.assertAlmostEqual(value["materialization_seconds_included_in_step"], 6.6)
            summaries.append(value)
        self.assertEqual(*summaries)
        self.assertNotEqual(self.events["curriculum"][1]["report"]["bundle_id"],
                            self.events["mixed"][1]["report"]["bundle_id"])

    def test_journal_rejects_wrong_order_digest_counts_unknown_or_duplicate_completion(self):
        mutations = [lambda e: e.pop(), lambda e: e.extend(copy.deepcopy(e[:2])),
            lambda e: e[0].update(cursor=True), lambda e: e[2].update(cursor=0),
            lambda e: e[1].update(event="failed"),
            lambda e: e[1]["report"].update(cursor=2),
            lambda e: e[1]["report"].update(bundle_id=1),
            lambda e: e[1]["report"].update(physical_optimizer_updates=None),
            lambda e: e[1]["report"].update(drawn_episode_exposures=8),
            lambda e: e[1]["report"].update(failed=True),
            lambda e: e[1]["report"]["microbatches"].reverse(),
            lambda e: e[1]["report"]["microbatches"][0].update(depth=5),
            lambda e: e[1]["report"]["microbatches"][0].update(rows_sha256="0" * 64),
            lambda e: e[1]["report"]["microbatches"][0].update(recipes_sha256="0" * 64),
            lambda e: e[1]["report"]["microbatches"][0]["exposures"].update(observation_bytes=0)]
        for mutate in mutations:
            events = copy.deepcopy(self.events["curriculum"])
            mutate(events)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.validate(events)

    def test_journal_rejects_false_loss_and_nested_or_nonfinite_timing(self):
        for change in ({"loss": .5}, {"loss": float("nan")}, {"step_seconds": -.1},
                       {"materialization_seconds": .8}, {"wall_seconds": .1}):
            events = copy.deepcopy(self.events["curriculum"])
            events[1]["report"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(events)
        manifest = copy.deepcopy(self.manifest)
        manifest["bundles"]["0"]["family_sha256"]["color"] = "0" * 64
        with self.assertRaises(ValueError):
            self.validate(self.events["curriculum"], manifest=manifest)

    def test_receipt_totals_and_worker_timing_must_reconcile_with_journal(self):
        journal = self.validate(self.events["curriculum"])
        receipt = {key: journal[key] for key in results.PHYSICAL}
        receipt.update(wall_seconds=100., setup_seconds=1., peak_cuda_allocated_mib=20., peak_cuda_reserved_mib=30.)
        results._validate_receipt(receipt, journal, 66, 2)
        for change in ({"physical_optimizer_updates": 65}, {"drawn_episode_exposures": 395},
                       {"neural_attempted_episode_exposures": True}, {"wall_seconds": 2.},
                       {"setup_seconds": 200.}, {"peak_cuda_reserved_mib": float("inf")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                results._validate_receipt({**receipt, **change}, journal, 66, 2)

    def input_fixture(self, directory):
        def write(name, value):
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf8")
        protocol = {"files_sha256": {"plan.json": "unused here"}, "execution_profile": {"profile": "strict"}}
        write("protocol.json", protocol); write("plan.json", {})
        protocol_hash = study.file_hash(directory / "protocol.json")
        write("preparation.json", {"status": "prepared", "protocol_sha256": protocol_hash,
                                   "wall_seconds": 1., "neural_training_or_inference": False})
        for arm in study.ARMS:
            write(f"{arm}/steps.jsonl", {})
            for step in (0, 2):
                write(f"{arm}/checkpoint-{step:06d}.pt", {"synthetic": True})
            receipt = {"schema": study.SCHEMA, "arm": arm, "status": "completed",
                "automatic_promotion": False, "physical_work_unknown": False, "retained_updates": 2,
                "protocol_sha256": protocol_hash, "execution_profile": protocol["execution_profile"],
                "journal_sha256": study.file_hash(directory / arm / "steps.jsonl"),
                "checkpoints": {f"checkpoint-{step:06d}.pt": study.file_hash(directory / arm / f"checkpoint-{step:06d}.pt")
                                for step in (0, 2)}}
            write(f"{arm}/receipt.json", receipt)
        return protocol

    def test_all_endpoint_gate_binds_exact_files_and_refuses_unfinished_arm(self):
        with tempfile.TemporaryDirectory() as name, patch.object(study, "STEPS", (0, 2)), patch.object(study, "TOTAL", 2):
            directory = Path(name)
            protocol = self.input_fixture(directory)
            hashes, receipts = results._input_hashes(directory, protocol)
            self.assertEqual(set(receipts), set(study.ARMS))
            self.assertIn("mixed/checkpoint-000002.pt", hashes)
            path = directory / "mixed/receipt.json"
            value = study.read(path)
            value["status"] = "running"
            path.write_text(json.dumps(value), encoding="utf8")
            with patch.object(study, "load_protocol", return_value=protocol), \
                 patch.object(study, "trainer", side_effect=AssertionError("model construction before both endpoints")), \
                 patch.object(results, "_proof", side_effect=AssertionError("proof before complete gate")):
                with self.assertRaisesRegex(ValueError, "complete endpoints"):
                    results.verify(directory)
            self.assertFalse((directory / "verification-started.json").exists())

    def test_checkpoint_or_journal_replacement_fails_input_binding(self):
        for relative in ("curriculum/checkpoint-000002.pt", "mixed/steps.jsonl"):
            with tempfile.TemporaryDirectory() as name, patch.object(study, "STEPS", (0, 2)), patch.object(study, "TOTAL", 2):
                directory = Path(name)
                protocol = self.input_fixture(directory)
                (directory / relative).write_text("changed", encoding="utf8")
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    results._input_hashes(directory, protocol)

    def test_evaluation_cannot_initialize_cuda_or_score_before_verification(self):
        with patch.object(results, "_verified", side_effect=ValueError("not verified")), \
             patch("experiments.execution_profile.runtime_profile", side_effect=AssertionError("CUDA must not initialize")), \
             patch.object(results, "_score", side_effect=AssertionError("model must not score")):
            with self.assertRaisesRegex(ValueError, "not verified"):
                results.evaluate("unused")

    def test_one_tiny_cpu_score_matches_actual_foundation_validator_and_cost_counts(self):
        from experiments.foundation_curriculum import generate_pair
        from experiments.foundation_evaluation import FoundationBank
        from experiments.foundation_evidence import training_cell
        from experiments.sequence_student import SequenceConfig, build_sequence_student
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            rows = generate_pair("color", 404100001, depth=0, turns=8, split="dev")
            name = "fresh/" + training_cell(rows[0])
            config = SequenceConfig(width=8, layers=1, heads=2, feedforward=16, max_turns=12)
            model = build_sequence_student(404100002, config=config)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            prepared = {name: FoundationBank(rows, role="dev", config=config)}
            result = results._score(model, prepared, {name: rows}, config, "dev")
            self.assertEqual(result["cost"]["scored_episodes"], 2)
            self.assertEqual(result["cost"]["scored_turns"], 16)
            self.assertEqual(result["cost"]["free_reply_turns"], 16)
            self.assertEqual(result["cost"]["encoder_episode_instances"], 2)
            self.assertEqual(result["metrics"]["per_bank"][name]["decoder_prefix"], "BOS only")
            self.assertIn("color", result["summary"]["per_panel_family_depth_operator"]["fresh"])
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        finally:
            torch.set_num_threads(threads)
