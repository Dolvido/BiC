"""Actual tiny CPU handoff/owned-loop checks; no formal study data or GPU."""
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_cycle_provider as module
from experiments import foundation_loop as loops
from experiments import foundation_training as training
from experiments.foundation_admission import repair_plan
from experiments.foundation_curriculum import FAMILIES, generate_pair
from experiments.foundation_cycle_training import CompletedParent, CycleFoundationTrainer
from experiments.foundation_evaluation import FoundationBank
from experiments.foundation_evidence import json_digest
from experiments.foundation_plan import build_plan, materialize_pair
from experiments.foundation_plan_index import AuthenticatedPlanIndex
from experiments.foundation_variant_provider import VariantFoundationPracticeProvider
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig


def arithmetic(value):
    return {key: value[key] for key in ("weights", "optimizer", "cursor", "evidence", "base_updates", "lifetime_updates", "cycle")}


class CycleProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.work = dict(completed_optimizer_calls=0, trainer_attempts=0,
            drawn_episode_exposures=0, neural_attempted_episode_exposures=0,
            completed_microbatch_episode_exposures=0, unknown_optimizer_reports=0,
            evaluation_bank_attempts=0, evaluation_banks_completed=0,
            evaluation_episode_attempts=0, evaluation_episodes_completed=0)
        cls.stack = ExitStack()
        cls.addClassCleanup(cls.stack.close)
        cls.addClassCleanup(torch.set_num_threads, cls.threads)
        actual_step, actual_optimizer, actual_score = training.FoundationTrainer.step, torch.optim.AdamW.step, FoundationBank.score
        def step(trainer):
            cls.work["trainer_attempts"] += 1
            try:
                return actual_step(trainer)
            finally:
                report = trainer.last_report
                if report:
                    for key in ("drawn_episode_exposures", "neural_attempted_episode_exposures", "completed_microbatch_episode_exposures"):
                        cls.work[key] += report[key] or 0
                    cls.work["unknown_optimizer_reports"] += int(report["physical_optimizer_updates"] is None)
        def optimizer(instance, *args, **kwargs):
            result = actual_optimizer(instance, *args, **kwargs)
            cls.work["completed_optimizer_calls"] += 1
            return result
        def score(bank, *args, **kwargs):
            cls.work["evaluation_bank_attempts"] += 1
            cls.work["evaluation_episode_attempts"] += bank.identity["episodes"]
            result = actual_score(bank, *args, **kwargs)
            cls.work["evaluation_banks_completed"] += 1
            cls.work["evaluation_episodes_completed"] += bank.identity["episodes"]
            return result
        cls.stack.enter_context(patch.object(training.FoundationTrainer, "step", step))
        cls.stack.enter_context(patch.object(torch.optim.AdamW, "step", optimizer))
        cls.stack.enter_context(patch.object(FoundationBank, "score", score))
        cls.addClassCleanup(cls.record_work)
        config = SequenceConfig(width=8, layers=4, heads=2, feedforward=16, max_turns=12)
        cls.banks = {role: {f"cycle/{family}/d0/direct/t{turns}": generate_pair(
            family, 871200001+ri*100+fi, depth=0, turns=turns, split="dev")
            for fi, (family, turns) in enumerate(zip(FAMILIES, (8, 10, 12)))}
            for ri, role in enumerate(("development", "retention"))}
        protection = sorted({transcript_digest(row) for banks in cls.banks.values() for rows in banks.values() for row in rows})
        plan, receipt = repair_plan(build_plan(seed=871100001, stage_updates=10, final_updates=6,
            micro_batch_size=2, rehearsal_every=2), protection)
        index = AuthenticatedPlanIndex(plan, admission_protected_transcripts=protection,
            protected_transcripts=protection, admission_receipt=receipt)
        transcripts = sorted({transcript_digest(row) for bundle in range(len(plan["bundles"]))
            for family in FAMILIES for row in materialize_pair(plan, bundle, family, 0)})
        cls.parents, cls.parent_states, cls.parent_loops = {}, {}, {}
        cls.original_inputs = dict(plan=plan, config=config, history=protection, receipt=receipt, index=index, transcripts=transcripts)
        cls.repeated_evidence = {}
        for architecture in ("flat", "hierarchical"):
            provider = VariantFoundationPracticeProvider(plan, architecture=architecture, seed=8713,
                config=config, evaluation_banks=cls.banks, admission_protected_transcripts=protection,
                protected_transcripts=protection, admission_receipt=receipt, plan_index=index)
            parent_loop = loops.FoundationLoop(provider, window_updates=66, chunk_updates=66, history_limit=2)
            parent_loop.run(max_updates=66)
            assert parent_loop.status["phase"] == "complete"
            cls.parent_states[architecture] = parent_loop.snapshot()["provider"]["learner"]
            cls.parent_loops[architecture] = parent_loop
            cls.parents[architecture] = CompletedParent.from_loop(parent_loop, training_transcripts=transcripts)
        cls.history = cls.parents["flat"].protected_transcripts
        cls.plan, cls.receipt = repair_plan(build_plan(seed=872100001, stage_updates=10, final_updates=6,
            micro_batch_size=2, rehearsal_every=2), cls.history)
        cls.index = AuthenticatedPlanIndex(cls.plan, admission_protected_transcripts=cls.history,
            protected_transcripts=cls.history, admission_receipt=cls.receipt)

    @classmethod
    def record_work(cls):
        record = dict(**cls.work, repeated_cycle_evidence=cls.repeated_evidence, device="cpu", formal_data_or_weights_used=False,
            scope="Includes completed 66-update parent fixtures, child and repeated-cycle checks, and intentionally discarded optimizer work; counts are physical calls, not capability results.")
        print("CYCLE_PROVIDER_PHYSICAL_WORK="+json.dumps(record, sort_keys=True))
        path = os.environ.get("BIC_CYCLE_PROVIDER_ACCOUNTING")
        if path:
            Path(path).write_text(json.dumps(record, indent=2)+"\n", encoding="utf8")

    def make(self, architecture="flat", **changes):
        options = dict(parent=self.parents[architecture], evaluation_banks=self.banks,
            admission_protected_transcripts=self.history, protected_transcripts=self.history,
            admission_receipt=self.receipt, plan_index=self.index)
        options.update(changes)
        return module.CycleFoundationPracticeProvider(self.plan, **options)

    def request(self, provider, start, stop):
        return dict(kind="prescribed_prefix", provider_sha256=provider.identity_sha256,
            plan_sha256=provider.identity["plan_sha256"], order="curriculum", start_cursor=start, stop_cursor=stop)

    def test_parent_weights_and_moments_are_owned_by_child_loop_at_zero_cursor(self):
        for architecture in self.parents:
            provider = self.make(architecture)
            saved = provider.snapshot()["learner"]
            for name in ("weights", "optimizer"):
                self.assertTrue(loops._same_tree(saved[name], self.parent_states[architecture][name]))
            self.assertEqual((provider.status()["updates"], provider.status()["lifetime_updates"]), (0, 66))
            loop = loops.FoundationLoop(provider)
            self.assertEqual(loop.snapshot()["state"]["work"]["physical_optimizer_updates"], 0)
            self.assertEqual(provider.identity["schema"], module.SCHEMA)

    def test_direct_and_provider_optimizer_continuation_match_both_architectures(self):
        for architecture in self.parents:
            provider = self.make(architecture)
            direct = CycleFoundationTrainer(self.plan, parent=self.parents[architecture],
                admission_protected_transcripts=self.history, protected_transcripts=self.history,
                admission_receipt=self.receipt, plan_index=self.index)
            provider.practice(self.request(provider, 0, 2), max_updates=2)
            direct.step(); direct.step()
            self.assertTrue(loops._same_tree(arithmetic(provider.snapshot()["learner"]), arithmetic(direct.snapshot())))
            self.assertEqual(provider.status()["lifetime_updates"], 68)

    def test_partial_evaluation_and_pending_practice_survive_disk_load(self):
        for architecture in self.parents:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"loop.pt"
                first = loops.FoundationLoop(self.make(architecture), window_updates=2, chunk_updates=1)
                first.run(max_updates=0, max_evaluations=1)
                first.save(path)
                saved = first.snapshot()
                second = loops.FoundationLoop.load(path, self.make(architecture))
                self.assertTrue(loops._same_tree(saved, second.snapshot()))
                second.run(max_updates=1)
                saved = second.snapshot()
                self.assertEqual(saved["provider"]["pending"]["consumed_updates"], 1)
                third = loops.FoundationLoop.load(path, self.make(architecture))
                self.assertTrue(loops._same_tree(saved, third.snapshot()))
                third.run(max_updates=1)
                self.assertEqual(third.provider.status()["lifetime_updates"], 68)
                reference = self.make(architecture)
                reference.practice(self.request(reference, 0, 2), max_updates=2)
                self.assertTrue(loops._same_tree(arithmetic(third.provider.snapshot()["learner"]), arithmetic(reference.snapshot()["learner"])))

    def test_failed_publication_rolls_back_full_inherited_optimizer(self):
        loop = loops.FoundationLoop(self.make("hierarchical"), window_updates=2, chunk_updates=1)
        loop.run(max_updates=0)
        with tempfile.TemporaryDirectory() as directory:
            loop.save(Path(directory)/"loop.pt")
            before = loop.snapshot()
            with patch.object(loop, "_write", side_effect=OSError("cycle publication failure")):
                with self.assertRaisesRegex(OSError, "cycle publication failure"):
                    loop.run(max_updates=1)
            self.assertTrue(loops._same_tree(before, loop.snapshot()))
            self.assertEqual(loop.last_report["discarded_optimizer_updates"], 1)
            self.assertEqual(loop.provider.status()["lifetime_updates"], 66)

    def test_changed_bank_roles_or_names_reject_before_child_model(self):
        changed = copy.deepcopy(self.banks)
        name = next(iter(changed["development"]))
        changed["development"]["renamed/"+name.split("/", 1)[1]] = changed["development"].pop(name)
        with patch.object(module, "CycleFoundationTrainer", side_effect=AssertionError("unexpected child model")):
            with self.assertRaisesRegex(ValueError, "exact parent evaluation"):
                self.make(evaluation_banks=changed)

    def test_child_baseline_matches_parent_final_teacher_free_evidence(self):
        for architecture in self.parents:
            provider = self.make(architecture)
            evidence = provider.parent_evidence
            final = evidence["final_evaluation"]["responses"]
            for role in self.banks:
                response = provider.evaluate(role)
                self.assertEqual(response["updates"], 0)
                for name, actual in response["per_bank"].items():
                    expected = copy.deepcopy(final[f"{role}:{name}"]["per_bank"][name])
                    actual = copy.deepcopy(actual)
                    expected.pop("seconds"); actual.pop("seconds")
                    self.assertEqual(json.dumps(actual, sort_keys=True), json.dumps(expected, sort_keys=True))
                    self.assertTrue(actual["free_running_replies"])
                    self.assertFalse(actual["teacher_used_for_policy"])
            evidence["identity"]["base_updates"] = 0
            evidence["retention_references"].clear()
            self.assertEqual(provider.parent_evidence["identity"]["base_updates"], 66)
            self.assertTrue(provider.parent_evidence["retention_references"])

    def test_foreign_parent_schema_and_corrupt_counters_reject_transactionally(self):
        provider = self.make(); before = provider.snapshot()
        cases = [self.make("hierarchical").snapshot()]
        changed = copy.deepcopy(before); changed["schema"] = "bic-foundation-variant-practice-provider-v1"; cases.append(changed)
        changed = copy.deepcopy(before); changed["schema"] = "bic-foundation-cycle-practice-provider-v1"; cases.append(changed)
        changed = copy.deepcopy(before); changed["learner"]["cycle"] += 1; cases.append(changed)
        changed = copy.deepcopy(before); changed["learner"]["lifetime_updates"] += 1; cases.append(changed)
        changed = copy.deepcopy(before); changed["learner"]["optimizer"]["state"][0]["step"] -= 1; cases.append(changed)
        for saved in cases:
            with self.assertRaises(ValueError):
                provider.restore(saved)
            self.assertTrue(loops._same_tree(before, provider.snapshot()))

    def test_source_drift_and_detached_identity_are_guarded(self):
        provider = self.make(); before = provider.snapshot()
        identity = provider.parent_identity; identity["base_updates"] = 0
        self.assertEqual(provider.parent_identity["base_updates"], 66)
        with patch.object(module, "source_hashes", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "cycle provider source"):
                provider.restore(before)
        self.assertTrue(loops._same_tree(before, provider.snapshot()))

    def test_three_completed_cycles_preserve_full_state_compact_lineage_and_saved_chain_restart(self):
        """Six actual finite cycles including the two class parent fixtures."""
        from experiments.foundation_lifetime_retention import LifetimeRetention
        from experiments.foundation_parent_capsule import export_parent_capsule, load_parent_capsule
        second_transcripts = sorted({transcript_digest(row) for bundle in range(len(self.plan["bundles"]))
            for family in FAMILIES for row in materialize_pair(self.plan, bundle, family, 0)})
        third_history = sorted(set(self.history) | set(second_transcripts))
        third_plan, third_receipt = repair_plan(build_plan(seed=873100001, stage_updates=10, final_updates=6,
            micro_batch_size=2, rehearsal_every=2), third_history)
        third_index = AuthenticatedPlanIndex(third_plan, admission_protected_transcripts=third_history,
            protected_transcripts=third_history, admission_receipt=third_receipt)
        third_transcripts = sorted({transcript_digest(row) for bundle in range(len(third_plan["bundles"]))
            for family in FAMILIES for row in materialize_pair(third_plan, bundle, family, 0)})
        def original_provider(architecture):
            args = self.original_inputs
            return VariantFoundationPracticeProvider(args["plan"], architecture=architecture, seed=8713,
                config=args["config"], evaluation_banks=self.banks, admission_protected_transcripts=args["history"],
                protected_transcripts=args["history"], admission_receipt=args["receipt"], plan_index=args["index"])
        def continued_provider(plan, receipt, index, parent):
            return module.CycleFoundationPracticeProvider(plan, parent=parent, evaluation_banks=self.banks,
                admission_protected_transcripts=parent.protected_transcripts,
                protected_transcripts=parent.protected_transcripts, admission_receipt=receipt, plan_index=index)
        for architecture in self.parents:
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as directory:
                paths = [Path(directory)/f"cycle-{number}.pt" for number in (1, 2, 3)]
                original = self.parent_loops[architecture]
                first = loops.FoundationLoop(original_provider(architecture), **original.options, payload=original.snapshot())
                first.save(paths[0])
                first_token = CompletedParent.from_loop(first, training_transcripts=self.original_inputs["transcripts"])
                capsule_records = []
                def capsule_restart(token):
                    capsule_path = Path(directory)/f"parent-{token.identity['cycle']}.zip"
                    receipt = export_parent_capsule(token, capsule_path)
                    with self.assertRaises(FileExistsError): export_parent_capsule(token, capsule_path)
                    with patch.object(torch, "load", side_effect=AssertionError("tensor load before trust pin")):
                        with self.assertRaises(ValueError):
                            load_parent_capsule(capsule_path, expected_sha256="0"*64, device="cpu", evaluation_banks=self.banks)
                    updates_before = self.work["completed_optimizer_calls"]
                    with patch.object(AuthenticatedPlanIndex, "__init__", side_effect=AssertionError("historical index replay")), \
                            patch.object(loops.FoundationLoop, "load", side_effect=AssertionError("historical loop replay")):
                        loaded = load_parent_capsule(capsule_path, expected_sha256=receipt["sha256"],
                            device="cpu", evaluation_banks=self.banks)
                    self.assertEqual(self.work["completed_optimizer_calls"], updates_before)
                    self.assertEqual(loaded.identity, token.identity)
                    self.assertEqual(loaded.retention_references, token.retention_references)
                    self.assertEqual(loaded.protected_transcripts, token.protected_transcripts)
                    self.assertTrue(loops._same_tree(loaded._learner(), token._learner()))
                    capsule_records.append(dict(cycle=token.identity["cycle"], sha256=receipt["sha256"],
                        bytes=capsule_path.stat().st_size, exact_identity=True, historical_index_or_loop_replay=False,
                        export_seconds=receipt["wall_seconds"], load_cost=copy.deepcopy(load_parent_capsule.last_report)))
                    return loaded
                first_token = capsule_restart(first_token)
                retention = LifetimeRetention.start(first_token)
                retention_states = [retention.snapshot()]
                tokens, completed = [first_token], [first]
                for ordinal, plan, receipt, index, transcripts in (
                        (2, self.plan, self.receipt, self.index, second_transcripts),
                        (3, third_plan, third_receipt, third_index, third_transcripts)):
                    parent = tokens[-1]
                    provider = continued_provider(plan, receipt, index, parent)
                    before = provider.snapshot()["learner"]
                    predecessor = completed[-1].snapshot()["provider"]["learner"]
                    for key in ("weights", "optimizer"):
                        self.assertTrue(loops._same_tree(before[key], predecessor[key]))
                    self.assertEqual((before["cursor"], before["base_updates"], before["lifetime_updates"], before["cycle"]),
                                     (0, 66*(ordinal-1), 66*(ordinal-1), ordinal))
                    loop = loops.FoundationLoop(provider, window_updates=66, chunk_updates=66, history_limit=2)
                    loop.run(max_updates=66)
                    self.assertEqual(loop.status["phase"], "complete")
                    self.assertEqual(provider.status()["lifetime_updates"], 66*ordinal)
                    after = provider.snapshot()["learner"]
                    self.assertTrue(all(float(item["step"]) == 66*ordinal for item in after["optimizer"]["state"].values()))
                    loop.save(paths[ordinal-1])
                    token = CompletedParent.from_loop(loop, training_transcripts=transcripts)
                    self.assertEqual(token.identity["cycle"], ordinal)
                    self.assertEqual(token.identity["base_updates"], 66*ordinal)
                    self.assertEqual(token.identity["inherited_updates"], 66*(ordinal-1))
                    self.assertEqual(token.identity["cycle_updates"], 66)
                    self.assertEqual(token.identity["predecessor_identity_sha256"], json_digest(parent.identity))
                    self.assertEqual(token.identity["predecessor_loop_envelope_sha256"], parent.identity["loop_envelope_sha256"])
                    self.assertEqual(set(token.protected_transcripts), set(parent.protected_transcripts) | set(transcripts))
                    self.assertFalse(set(transcripts) & set(parent.protected_transcripts))
                    token = capsule_restart(token)
                    retention = retention.advance(token)
                    ledger = retention.snapshot()
                    self.assertEqual(ledger["cycle"], ordinal)
                    self.assertEqual(ledger["lifetime_updates"], 66*ordinal)
                    reconstructed_ledger = LifetimeRetention.restore(ledger, expected_sha256=json_digest(ledger), parent=token)
                    self.assertEqual(reconstructed_ledger.snapshot(), ledger)
                    retention_states.append(ledger)
                    # Parent identity contains predecessor digests, never the
                    # prior full identity, tensors or envelope payload.
                    def no_recursive_parent(value):
                        if isinstance(value, dict):
                            self.assertNotIn("parent_identity", value)
                            for item in value.values(): no_recursive_parent(item)
                        elif isinstance(value, list):
                            for item in value: no_recursive_parent(item)
                    no_recursive_parent(token.identity)
                    for field in ("cycle", "base_updates", "lifetime_updates"):
                        changed = copy.deepcopy(after); changed[field] += 1
                        with self.assertRaises(ValueError): provider.restore({**provider.snapshot(), "learner": changed})
                        self.assertTrue(loops._same_tree(after, provider.snapshot()["learner"]))
                    tokens.append(token); completed.append(loop)
                self.assertLessEqual(len(tokens[2]._identity), len(tokens[1]._identity)+512)
                self.assertLessEqual(len(tokens[2]._learner_image), len(tokens[1]._learner_image)+4096)
                # Metrics contain variable-length generated replies; allow the
                # fixed tiny bank's bounded content variation. Ancestry itself
                # is checked structurally as digest-only, independently of size.
                self.assertLessEqual(len(tokens[2]._metadata), len(tokens[1]._metadata)+262144)
                # Explicit legacy reconstruction route: load each saved finite
                # loop, validating its current plan and complete learner. This
                # performs no optimizer work but scales with historical cycles.
                restored_first = loops.FoundationLoop.load(paths[0], original_provider(architecture))
                previous = CompletedParent.from_loop(restored_first, training_transcripts=self.original_inputs["transcripts"])
                self.assertEqual(previous.identity, tokens[0].identity)
                for ordinal, plan, receipt, index, transcripts in (
                        (2, self.plan, self.receipt, self.index, second_transcripts),
                        (3, third_plan, third_receipt, third_index, third_transcripts)):
                    restored = loops.FoundationLoop.load(paths[ordinal-1], continued_provider(plan, receipt, index, previous))
                    previous = CompletedParent.from_loop(restored, training_transcripts=transcripts)
                    self.assertEqual(previous.identity, tokens[ordinal-1].identity)
                    self.assertTrue(loops._same_tree(previous._learner(), tokens[ordinal-1]._learner()))
                self.repeated_evidence[architecture] = dict(completed_cycles=3, lifetime_updates=198,
                    token_identity_bytes=[len(token._identity) for token in tokens],
                    token_learner_image_bytes=[len(token._learner_image) for token in tokens],
                    token_metadata_bytes=[len(token._metadata) for token in tokens],
                    protected_transcript_counts=[len(token.protected_transcripts) for token in tokens],
                    saved_chain_reconstruction_exact=True, capsule_roundtrips=capsule_records,
                    lifetime_retention_cycle=retention.snapshot()["cycle"],
                    lifetime_retention_ledger_sha256=json_digest(retention.snapshot()),
                    lifetime_retention_alarm_count=len(retention.snapshot()["alarms"]))
                artifact_root = os.environ.get("BIC_CYCLE_REPEATED_ARTIFACTS")
                if artifact_root:
                    destination = Path(artifact_root)/architecture
                    destination.mkdir(parents=True, exist_ok=False)
                    for record, ledger in zip(capsule_records, retention_states):
                        ordinal = record["cycle"]
                        shutil.copyfile(Path(directory)/f"parent-{ordinal}.zip", destination/f"parent-{ordinal}.zip")
                        (destination/f"retention-{ordinal}.json").write_text(json.dumps(ledger, indent=2), encoding="utf8")
                    (destination/"banks.json").write_text(json.dumps(self.banks, indent=2), encoding="utf8")
                    (destination/"origin.json").write_text(json.dumps(dict(
                        schema="bic-actual-three-cycle-test-artifacts-v1", architecture=architecture,
                        capsule_exports=capsule_records,
                        retention_sha256=[json_digest(state) for state in retention_states],
                        bank_rows_sha256=json_digest(self.banks),
                        scope="Actual engineering-test completed loops; tiny CPU weights and authenticated artifact pins for further continuity checks, not capability evidence."), indent=2), encoding="utf8")


if __name__ == "__main__":
    unittest.main(verbosity=2)
