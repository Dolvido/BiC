"""Controlled metric fixtures and one tiny CPU scoring check; no optimization."""
from copy import deepcopy
from dataclasses import asdict, replace
import unittest

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments.composition_evaluation import METRICS, evaluate_banks, prediction_metrics
from experiments.foundation_curriculum import VERSION, generate_pair
from experiments.foundation_evidence import json_digest, training_cell
from experiments.foundation_evaluation import FoundationBank
from experiments.foundation_metrics import compact, validate_metrics
from experiments.sequence_student import SequenceConfig, build_sequence_student


CONFIG = SequenceConfig(width=8, layers=1, heads=2, feedforward=16, max_positions=1560, max_turns=12)


def aggregate(per_bank):
    return {"per_bank": per_bank, **{"macro_"+key: sum(row[key] for row in per_bank.values())/len(per_bank) for key in METRICS}}


def fixture(*, role="dev", control="normal", constant=None, lengths=(8, 10)):
    banks, scores = {}, {}
    for turns in lengths:
        split = "train" if role == "train_fit" else role
        rows = generate_pair("color", 511000000+turns, depth=2, turns=turns,
                             split=split, structure_split="train" if split != "audit" else "audit")
        name = "probe/"+training_cell(rows[0])
        targets = torch.tensor([[turn["target"] for turn in row["turns"]] for row in rows])
        chosen = targets.clone()
        if constant is not None: chosen[targets.ne(3)] = constant
        score = prediction_metrics(torch.nn.functional.one_hot(chosen, 4).float()*5, targets,
                                   reply_correct=chosen.eq(targets), reply_actions=chosen)
        score.update(bank={"sha256": json_digest(rows), "version": VERSION,
                           "episodes": len(rows), "turns": turns, "role": role, "config": asdict(CONFIG)},
                     control=control, free_running_replies=True, teacher_used_for_policy=False,
                     decoder_prefix="BOS only", seconds=.125)
        banks[name], scores[name] = rows, score
    return banks, aggregate(scores)


class FoundationMetricTests(unittest.TestCase):
    def test_actual_cpu_scorer_normal_blank_reset_matches_metric_wire_contract(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            # Two private fixture pairs, not study data. The reset control uses
            # independent utterance inputs but retains original episode truth.
            rows, _ = fixture()
            banks = {name: FoundationBank(value, role="dev", config=CONFIG)
                     for name, value in rows.items()}
            model = build_sequence_student(511000101, config=CONFIG).train()
            model.tokens.eval()
            modes = [module.training for module in model.modules()]
            before = checkpoint_digest(model)
            results = {}
            for control in ("normal", "blank", "reset"):
                scores = evaluate_banks(model, banks, batch_size=2, score_replies=True, control=control)
                self.assertTrue(validate_metrics(scores, rows, CONFIG, "dev", control))
                self.assertEqual(checkpoint_digest(model), before)
                self.assertEqual([module.training for module in model.modules()], modes)
                results[control] = {"validated": True,
                    "episodes": sum(value["episodes"] for value in scores["per_bank"].values()),
                    "final_opposite_pairs": sum(value["final_pairs"]["total"] for value in scores["per_bank"].values())}
                self.assertEqual(results[control]["final_opposite_pairs"], 2)
            episodes = sum(len(value) for value in rows.values())
            utterances = sum(len(row["turns"]) for value in rows.values() for row in value)
            self.__class__.scoring_integration_report = {
                "schema": "bic-foundation-metrics-integration-v1", "device": "cpu",
                "config": asdict(CONFIG), "curriculum_version": VERSION,
                "controls": results, "optimizer_updates": 0, "logical_episode_exposures": 3*episodes,
                "utterance_exposures": 3*utterances, "encoder_input_sequences": 2*episodes+utterances,
                "weights_sha256_before": before, "weights_sha256_after": checkpoint_digest(model),
                "weights_and_modes_preserved": True,
                "scope": "Private untrained CPU fixture; reset expands episodes to independent utterance inputs. No capability evidence."}
            self.assertEqual(self.scoring_integration_report["logical_episode_exposures"], 12)
        finally:
            torch.set_num_threads(threads)

    def test_all_roles_controls_real_version_and_no_input_mutation(self):
        for role in ("train_fit", "dev", "audit"):
            for control in ("normal", "blank", "reset"):
                rows, metrics = fixture(role=role, control=control)
                before = deepcopy((rows, metrics))
                self.assertTrue(validate_metrics(metrics, rows, CONFIG, role, control))
                self.assertEqual(before, (rows, metrics))
                self.assertTrue(all(row["bank"]["version"] == VERSION for row in metrics["per_bank"].values()))

    def test_constant_answers_can_pass_query_items_but_not_opposite_final_pairs(self):
        rows, metrics = fixture(constant=0)
        self.assertTrue(validate_metrics(metrics, rows, CONFIG, "dev"))
        summary = compact(metrics)
        group = summary["per_panel_family_depth_operator"]["probe"]["color"]["d2"]["composed"]
        self.assertEqual(group["pooled_counts"]["final_pairs"], {"correct": 0, "total": 2})
        self.assertEqual(group["pooled_counts"]["final_reply_pairs"], {"correct": 0, "total": 2})
        self.assertGreater(group["pooled_counts"]["query"]["correct"], 0)
        self.assertEqual(group["macro"]["final_pair_accuracy"], 0.)
        self.assertIsNone(group["macro"]["ask_precision"])
        self.assertEqual(group["macro"]["ask_recall"], 0.)
        self.assertEqual(set(group["per_length"]), {"8", "10"})
        self.assertEqual(metrics["macro_query_accuracy"], summary["macro"]["query_accuracy"])

    def test_unsupported_ask_and_all_original_denominators_are_retained(self):
        rows, metrics = fixture(constant=2)
        self.assertTrue(validate_metrics(metrics, rows, CONFIG, "dev"))
        before = deepcopy(metrics)
        summary = compact(metrics)
        group = summary["per_panel_family_depth_operator"]["probe"]["color"]["d2"]["composed"]
        self.assertEqual(group["macro"]["unsupported_ask_rate"], 1.)
        self.assertEqual(group["pooled_counts"]["unsupported_ask_count"], group["pooled_counts"]["known"]["total"])
        for key, original in metrics["per_bank"].items():
            result = group["per_length"][str(original["turns_per_episode"])]
            for field, value in original.items(): self.assertEqual(result[field], value)
        self.assertEqual(metrics, before)
        group["per_length"]["8"]["bank"]["version"] = "changed copy"
        self.assertEqual(metrics, before)

    def test_identity_role_config_policy_and_canonical_row_tampering_rejected(self):
        rows, metrics = fixture(lengths=(8,))
        name = next(iter(rows))
        changes = (("bank", "version", "bic-shared-composition-v1"), ("bank", "sha256", "0"*64),
                   ("bank", "role", "audit"), (None, "control", "blank"),
                   (None, "free_running_replies", False), (None, "teacher_used_for_policy", True),
                   (None, "decoder_prefix", "teacher labels"))
        for group, key, value in changes:
            bad = deepcopy(metrics)
            target = bad["per_bank"][name] if group is None else bad["per_bank"][name][group]
            target[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError): validate_metrics(bad, rows, CONFIG, "dev")
        bad = deepcopy(metrics); bad["per_bank"][name]["bank"]["config"]["layers"] = True
        with self.assertRaises(ValueError): validate_metrics(bad, rows, CONFIG, "dev")
        bad = deepcopy(metrics); bad["per_bank"][name]["bank"]["version"] = "bic-shared-composition-v1"
        with self.assertRaisesRegex(ValueError, "real version"): compact(bad)
        corrupted = deepcopy(rows); corrupted[name][0]["turns"][-1]["target"] = 2
        with self.assertRaises(ValueError): validate_metrics(metrics, corrupted, CONFIG, "dev")
        other_config = replace(CONFIG, max_positions=2)
        bad = deepcopy(metrics); bad["per_bank"][name]["bank"]["config"] = asdict(other_config)
        with self.assertRaisesRegex(ValueError, "model capacity"): validate_metrics(bad, rows, other_config, "dev")

    def test_counts_ratios_confusion_and_macro_corruption_rejected(self):
        rows, metrics = fixture(lengths=(8,))
        name = next(iter(rows))
        changes = {"query_total": 999, "known_total": 999, "opposite_pair_total": 999,
                   "ask_true": 999, "ask_predicted": 999, "ask_correct": 999,
                   "query_correct": True, "known_accuracy": .3, "final_accuracy": .3,
                   "query_reply_correct": 999, "reply_parseable_queries": 0,
                   "action_reply_agreement": .12345, "query_loss": float("nan"),
                   "brier_score": 3., "seconds": -1.}
        for key, value in changes.items():
            bad = deepcopy(metrics); bad["per_bank"][name][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError): validate_metrics(bad, rows, CONFIG, "dev")
        bad = deepcopy(metrics); bad["per_bank"][name]["confusion_matrix"][0][0] += 1
        with self.assertRaises(ValueError): validate_metrics(bad, rows, CONFIG, "dev")
        bad = deepcopy(metrics); bad["per_bank"][name]["per_target"]["0"]["accuracy"] = .123
        with self.assertRaises(ValueError): validate_metrics(bad, rows, CONFIG, "dev")
        bad = deepcopy(metrics); bad["macro_query_accuracy"] = .123
        with self.assertRaises(ValueError): validate_metrics(bad, rows, CONFIG, "dev")

    def test_canonically_impossible_pair_counts_cannot_hide_in_consistent_macros(self):
        rows, metrics = fixture(lengths=(8,))
        name = next(iter(rows))
        bad = deepcopy(metrics); score = bad["per_bank"][name]
        score["by_turn"][-1]["opposite_pair_correct"] = 0
        score["opposite_pair_correct"] -= 1
        score["opposite_pair_accuracy"] = score["opposite_pair_correct"]/score["opposite_pair_total"]
        score["final_pairs"]["correct"] = 0; score["final_pair_accuracy"] = 0.
        bad = aggregate(bad["per_bank"])
        with self.assertRaisesRegex(ValueError, "cannot yield"): validate_metrics(bad, rows, CONFIG, "dev")
        bad = deepcopy(metrics); score = bad["per_bank"][name]
        score["final_reply_pair_correct"] = 0; score["final_reply_pair_accuracy"] = 0.
        with self.assertRaisesRegex(ValueError, "complete reply pairs"):
            validate_metrics(aggregate(bad["per_bank"]), rows, CONFIG, "dev")

    def test_absent_precision_is_null_and_bank_name_and_pair_identity_are_checked(self):
        rows, metrics = fixture(constant=0, lengths=(8,))
        name = next(iter(rows))
        bad = deepcopy(metrics); bad["per_bank"][name]["ask_precision"] = 0.
        with self.assertRaisesRegex(ValueError, "absent denominator"): validate_metrics(bad, rows, CONFIG, "dev")
        wrong_name = name.replace("/color/", "/count/")
        with self.assertRaises(ValueError):
            validate_metrics(aggregate({wrong_name: metrics["per_bank"][name]}), {wrong_name: rows[name]}, CONFIG, "dev")
        duplicate = {name: rows[name]+deepcopy(rows[name])}
        with self.assertRaisesRegex(ValueError, "duplicate foundation pair"):
            validate_metrics(metrics, duplicate, CONFIG, "dev")
        bad = deepcopy(metrics); bad["per_bank"][name]["by_turn"][-1]["opposite_pair_total"] = 0
        with self.assertRaises(ValueError): validate_metrics(bad, rows, CONFIG, "dev")


if __name__ == "__main__": unittest.main()
