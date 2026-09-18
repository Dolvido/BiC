"""Leakage, official-policy scoring, later-query slices, and screen mechanics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments import audit_cognitive_credit as audit
from experiments.cognitive_credit import CreditTrainer, build_credit_student
from experiments.cognitive_curriculum import generate_cognitive


class CreditAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_later_mask_counts_unknown_ordinals_and_excludes_only_first_known(self):
        labels = torch.tensor([[3, 3], [0, 0], [2, 3], [1, 2], [3, 3], [2, 2]])
        expected = torch.zeros_like(labels, dtype=torch.bool)
        expected[3, 0] = True
        self.assertTrue(torch.equal(audit.last_known_later_mask(labels), expected))

    def test_position_and_reply_metrics_preserve_pair_and_known_denominators(self):
        labels = torch.tensor([[3, 3], [0, 1], [3, 3], [2, 2], [3, 3], [1, 0]])
        replies = torch.ones_like(labels, dtype=torch.bool)
        replies[5, 1] = False
        rows = [{"counterfactual_group": "same"}] * 2
        scored = audit.position_metrics(labels.clone(), labels, rows, replies)
        later = scored["last_known_later_query"]
        self.assertEqual((later["correct"], later["total"]), (2, 2))
        self.assertEqual(later["pairs"], {"total": 1, "correct": 1, "accuracy": 1.})
        self.assertEqual(later["reply_exact_accuracy"], .5)
        self.assertEqual(scored["by_query_ordinal"][1]["slices"]["unknown"]["total"], 2)
        self.assertIsNone(scored["by_query_ordinal"][1]["slices"]["known"]["accuracy"])

    def test_prepare_freezes_seed_banks_sources_and_auxiliary_free_adaptation(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit, "audit_sources", return_value={"fixture": "one"}):
            manifest = audit.prepare(temp, seed=2303)
            self.assertEqual(manifest, audit.prepare(temp, seed=2303))
            _, banks = audit.load_prepared(temp)
            self.assertEqual(manifest["adaptation_aux_weight"], 0.)
            self.assertEqual(manifest["heldout_family"], "graph_reachability")
            self.assertEqual(len(banks["support"]["graph_reachability"]), 64)
            self.assertTrue(all(row["split"] == "audit" and row["family"] == "graph_reachability"
                for rows in banks["query"].values() for row in rows))
            self.assertNotIn("graph_reachability", banks["main_development"])
            with self.assertRaisesRegex(ValueError, "contract changed"):
                audit.prepare(temp, seed=2301)
            banks["query"]["level_2"][0]["turns"][0]["text"] = "tampered"
            (Path(temp) / "banks.json").write_text(json.dumps(banks), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "bank mismatch"):
                audit.load_prepared(temp)

    def test_adaptation_copies_full_weights_but_discards_optimizer_and_auxiliary_loss(self):
        support = {"graph_reachability": generate_cognitive(80, 2, family="graph_reachability")}
        parent = CreditTrainer(support, seed=7, aux_weight=.3, batch_size=2)
        parent.step("graph_reachability")
        self.assertTrue(parent.optimizer.state)
        adapted = audit.fresh_adaptation(parent.model.state_dict(), support)
        self.assertEqual(checkpoint_digest(parent.model), checkpoint_digest(adapted.model))
        self.assertFalse(adapted.optimizer.state)
        self.assertEqual(adapted.aux_weight, 0.)
        self.assertEqual(adapted.updates, 0)
        other = audit.fresh_adaptation(parent.model.state_dict(), support)
        self.assertTrue(torch.equal(adapted.generators["graph_reachability"].get_state(),
                                    other.generators["graph_reachability"].get_state()))

    def test_scoring_ignores_auxiliary_head_and_full_cpu_restart_works(self):
        model = build_credit_student(11)
        rows = generate_cognitive(42, 2, family="graph_reachability", level=2)
        before = audit.score_banks(model, {"graph": rows}, replies=True)
        with torch.no_grad():
            model.credit_head.weight.fill_(1000)
            model.credit_head.bias.copy_(torch.tensor([0., 1000., 0., 0.]))
        after = audit.score_banks(model, {"graph": rows}, replies=True)
        self.assertEqual(before, after)
        self.assertTrue(audit.cpu_restart(model.state_dict(), rows[0], seed=11)["exact"])

    def test_incomplete_or_wrong_number_of_arms_rejected_before_scoring(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            audit.verify_runs([], {})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            protocol = {"schema": "bic-credit-training-v1", "seed": 2301, "steps": 1800, "aux_weight": 0.}
            (path / "protocol.json").write_text(json.dumps(protocol))
            (path / "report.json").write_text(json.dumps({"updates": 3}))
            torch.save({"training": {"updates": 3}}, path / "latest.pt")
            manifest = {"prepared_utc": "2026-01-01T00:00:00+00:00", "seed": 2301,
                        "aux_weights": [0., .3], "exposure_sha256": {}}
            with self.assertRaisesRegex(ValueError, "incomplete"):
                audit.verify_runs([path, path], manifest)

    def test_screen_requires_later_gains_and_retention_in_both_phases(self):
        thresholds = {"main_macro_pair_gain": .10, "family_pair_gain": .05,
            "max_family_query_loss": .05, "later_known_macro_gain": .05,
            "later_known_family_gain": .05, "max_later_known_family_loss": .05,
            "max_adaptation_auc_loss": .02, "max_graph_later_query_loss": .02,
            "max_graph_later_pair_loss": .02, "max_retention_macro_loss": .02,
            "max_retention_family_query_loss": .05}
        def bank(score):
            return {"query_accuracy": score, "counterfactual_accuracy": score,
                "positions": {"last_known_later_query": {"accuracy": score, "pairs": {"accuracy": score}}}}
        def result(score):
            family = {"macro_query_accuracy": score, "macro_pair_accuracy": score,
                "macro_later_known_accuracy": score,
                "per_bank": {name: bank(score) for name in audit.TRAIN_FAMILIES}}
            return {"main_development": copy.deepcopy(family), "retention_before": copy.deepcopy(family),
                "retention_after": copy.deepcopy(family), "normalized_adaptation_auc": score,
                "curve": [{"per_bank": {name: bank(score) for name in ("level_2", "level_3")}}]}
        control, candidate = result(.4), result(.55)
        self.assertTrue(audit.benefit_screen(candidate, control, thresholds)["passed"])
        candidate["main_development"]["macro_later_known_accuracy"] = .4
        self.assertFalse(audit.benefit_screen(candidate, control, thresholds)["passed"])
        candidate = result(.55)
        candidate["retention_before"]["macro_query_accuracy"] = .37
        self.assertFalse(audit.benefit_screen(candidate, control, thresholds)["passed"])
        candidate = result(.55)
        candidate["curve"][-1]["per_bank"]["level_2"]["positions"]["last_known_later_query"]["pairs"]["accuracy"] = None
        checked = audit.benefit_screen(candidate, control, thresholds)
        self.assertIsNone(checked["requirements"]["graph_level_2_later_pairs"])
        self.assertFalse(checked["passed"])


if __name__ == "__main__":
    unittest.main()
