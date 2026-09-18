"""Count-only fixtures: no tensors, model calls, training or scored study files."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import unittest

from experiments import foundation_variant_analysis as analysis


def row_fixture(name, role, action, reply, *, extra_known=False, unsupported=False):
    pairs = 16 if role == "dev" else 32
    n, turns = 2*pairs, int(name.rsplit("/t", 1)[1])
    targets = [[2]+[3]*(turns-2)+[e % 2] for e in range(n)]
    if extra_known:
        for values in targets: values[1] = 0
    predictions, replies = deepcopy(targets), deepcopy(targets)
    for e in range(n):
        if e//2 >= action: predictions[e][-1] = 2 if unsupported else 1-targets[e][-1]
        if e//2 >= reply: replies[e][-1] = 1-targets[e][-1]
    confusion = [[0]*4 for _ in range(4)]
    by_turn = []
    for turn in range(turns):
        total = sum(targets[e][turn] != 3 for e in range(n))
        correct = sum(targets[e][turn] != 3 and targets[e][turn] == predictions[e][turn] for e in range(n))
        eligible = [e for e in range(0, n, 2) if {targets[e][turn], targets[e+1][turn]} == {0, 1}]
        paired = sum(all(predictions[i][turn] == targets[i][turn] for i in (e, e+1)) for e in eligible)
        by_turn.append(dict(turn=turn, correct=correct, total=total, accuracy=correct/total if total else None,
                            opposite_pair_correct=paired, opposite_pair_total=len(eligible)))
        for e in range(n):
            if targets[e][turn] != 3: confusion[targets[e][turn]][predictions[e][turn]] += 1
    per_target = {str(t):dict(correct=confusion[t][t], total=sum(confusion[t]),
        accuracy=confusion[t][t]/sum(confusion[t]) if sum(confusion[t]) else None) for t in range(3)}
    q, k = sum(map(sum, confusion)), sum(map(sum, confusion[:2]))
    qc, kc = sum(confusion[t][t] for t in range(3)), confusion[0][0]+confusion[1][1]
    qr = sum(replies[e][t] == targets[e][t] for e in range(n) for t in range(turns) if targets[e][t] != 3)
    agreement = sum(replies[e][t] == predictions[e][t] for e in range(n) for t in range(turns) if targets[e][t] != 3)
    ask_true, ask_pred, ask_correct = sum(confusion[2]), sum(v[2] for v in confusion), confusion[2][2]
    return dict(bank=dict(sha256=hashlib.sha256((role+name).encode()).hexdigest(), version=analysis.VERSION,
            episodes=n, turns=turns, role=role, config=deepcopy(analysis.CONFIG)),
        control="normal", free_running_replies=True, teacher_used_for_policy=False, decoder_prefix="BOS only",
        episodes=n, turns_per_episode=turns, query_correct=qc, query_total=q, query_accuracy=qc/q,
        known_correct=kc, known_total=k, known_accuracy=kc/k,
        final_correct=2*action, final_total=n, final_accuracy=action/pairs,
        final_pairs=dict(correct=action, total=pairs), final_pair_accuracy=action/pairs,
        opposite_pair_correct=action, opposite_pair_total=pairs, opposite_pair_accuracy=action/pairs,
        confusion_matrix=confusion, per_target=per_target, ask_true=ask_true, ask_predicted=ask_pred,
        ask_correct=ask_correct, ask_precision=ask_correct/ask_pred, ask_recall=ask_correct/ask_true,
        by_turn=by_turn, query_reply_correct=qr, query_reply_accuracy=qr/q,
        final_reply_pair_correct=reply, final_reply_pair_accuracy=reply/pairs,
        reply_parseable_queries=q, action_reply_agreement=agreement/q,
        query_loss=1., brier_score=.5, seconds=.01)


def remacro(metrics):
    for key in analysis.METRICS:
        metrics["macro_"+key] = sum(row[key] for row in metrics["per_bank"].values())/len(metrics["per_bank"])
    return metrics


def metrics_fixture(role="dev", action=8, reply=4, *, choices=None, **options):
    rows = {}
    for name in analysis.expected_cells(role):
        a, r = choices(name) if choices else (action, reply)
        rows[name] = row_fixture(name, role, a, r, **options)
    return remacro(dict(per_bank=rows))


def result_fixture(flat=(4, 4), hierarchical=(8, 8), *, choices=None):
    return {architecture:{seed:metrics_fixture("audit", *counts,
        choices=(lambda name, a=architecture, s=seed: choices(a, s, name)) if choices else None)
        for seed in analysis.SEEDS} for architecture, counts in (("flat", flat), ("hierarchical", hierarchical))}


class CalibrationTests(unittest.TestCase):
    def test_exact_protocol_cell_sets(self):
        dev, audit = analysis.expected_cells("dev"), analysis.expected_cells("audit")
        self.assertEqual((len(dev), len(audit)), (90, 99))
        self.assertEqual(len([n for n in dev if n.startswith("fresh/")]), 63)
        self.assertEqual(len([n for n in audit if n.startswith("composed/")]), 36)
        self.assertNotIn("composed/color/d2/composed/t8", dev)
        self.assertIn("composed/color/d2/composed/t8", audit)
        with self.assertRaises(ValueError): analysis.expected_cells("train_fit")

    def test_rank_exact_components_and_balanced_known_not_pooled(self):
        metrics = metrics_fixture(extra_known=True)
        before = json.dumps(metrics, sort_keys=True)
        result = analysis.calibration_rank(metrics)
        expected = (Fraction(1,2)+Fraction(1,4)+Fraction(2,3))/3
        self.assertEqual(result["rank"], round(float(expected), 12))
        self.assertEqual(result["score_unrounded"], float(expected))
        self.assertEqual(len(result["by_cell"]), 90)
        for family, group in result["by_domain"].items():
            self.assertEqual(group["cells"], 30)
            self.assertEqual(group["components"]["target_balanced_known"], 2/3)
            self.assertEqual(group["counts"]["known"]["rate"], .75)
            self.assertEqual(group["counts"]["action_pairs"], dict(correct=240, total=480, rate=.5))
            self.assertEqual(group["counts"]["unsupported_ask"], dict(count=0, total=1920, rate=0.))
        self.assertEqual(json.dumps(metrics, sort_keys=True), before)
        self.assertFalse(result["automatic_promotion"])

    def test_target_balanced_rank_preserves_ask_and_zero_domain(self):
        def choices(name): return (0, 0) if "/count/" in name else (8, 4)
        result = analysis.calibration_rank(metrics_fixture(choices=choices, unsupported=True))
        self.assertEqual(result["by_domain"]["count"]["components"]["composite"], 0.)
        self.assertGreater(result["rank"], 0.)
        self.assertEqual(result["by_domain"]["count"]["counts"]["unsupported_ask"]["rate"], 1.)
        self.assertEqual(result["by_domain"]["count"]["counts"]["unknown_ask"]["rate"], 1.)

    def test_choose_rate_same_banks_and_lower_numeric_tie(self):
        row = metrics_fixture()
        result = analysis.choose_rate({".003":row, ".001":deepcopy(row), "0.0003":deepcopy(row)})
        self.assertEqual(result["selected_rate"], "0.0003")
        self.assertEqual(result["rate"], .0003)
        self.assertEqual(result["selected_learning_rate"], result["rate"])
        winner = analysis.choose_rate({".003":metrics_fixture(action=12), ".001":row, ".0003":row})
        self.assertEqual(winner["selected_rate"], ".003")

    def test_choose_rate_rejects_incomplete_alias_and_changed_bank(self):
        row = metrics_fixture()
        for rates in ({".0003":row,".001":row}, {".001":row,".0003":row,"0.0003":row},
                      {"NaN":row,".001":row,".003":row}, {".0004":row,".001":row,".003":row}):
            with self.subTest(rates=list(rates)), self.assertRaises(ValueError): analysis.choose_rate(rates)
        changed = deepcopy(row)
        next(iter(changed["per_bank"].values()))["bank"]["sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "different development banks"):
            analysis.choose_rate({".0003":row,".001":row,".003":changed})

    def test_count_and_provenance_tampering_rejected(self):
        original = metrics_fixture()
        def mutate(fn):
            value = deepcopy(original); fn(value, next(iter(value["per_bank"].values()))); return value
        changes = [
            lambda m,r:m["per_bank"].pop(next(iter(m["per_bank"]))),
            lambda m,r:m["per_bank"].__setitem__("extra/color/d0/direct/t8",deepcopy(r)),
            lambda m,r:r.__setitem__("episodes",64),
            lambda m,r:r["final_pairs"].__setitem__("total",True),
            lambda m,r:r.__setitem__("query_loss",float("nan")),
            lambda m,r:r.__setitem__("query_accuracy",.01),
            lambda m,r:r["per_target"]["0"].__setitem__("correct",1),
            lambda m,r:r["by_turn"][-1].__setitem__("opposite_pair_total",999),
            lambda m,r:r["by_turn"][-1].__setitem__("accuracy",0.),
            lambda m,r:r["bank"].__setitem__("role","audit"),
            lambda m,r:r["bank"].__setitem__("version","legacy"),
            lambda m,r:r["bank"]["config"].__setitem__("width",96),
            lambda m,r:r.__setitem__("control","blank"),
            lambda m,r:r.__setitem__("teacher_used_for_policy",True),
            lambda m,r:r.__setitem__("decoder_prefix","answers"),
            lambda m,r:r.__setitem__("action_reply_agreement",.123456789),
            lambda m,r:m.__setitem__("macro_final_pair_accuracy",.1),
            lambda m,r:r.__setitem__("unexpected",float("inf")),
        ]
        for index, fn in enumerate(changes):
            with self.subTest(change=index), self.assertRaises(ValueError): analysis.calibration_rank(mutate(fn))


class ComparisonTests(unittest.TestCase):
    def test_positive_comparison_counts_and_exact_two_panel_scope(self):
        metrics = result_fixture()
        before = json.dumps(metrics, sort_keys=True)
        result = analysis.comparison_screen(metrics)
        self.assertTrue(result["promising_comparison"])
        self.assertFalse(result["automatic_promotion"])
        self.assertEqual(len(result["omitted_from_primary_screen"]), 36)
        for panel, cells in (("fresh_primitive",27),("held_composed",36)):
            row = result["panels"][panel]
            self.assertEqual(row["cells"], cells)
            self.assertTrue(all(row["criteria"].values()))
            for seed in analysis.SEEDS:
                state = row["per_seed"][seed]
                self.assertEqual(state["paired_composite_delta"], .125)
                self.assertEqual(state["flat"]["overall"]["counts"]["action_pairs"]["total"], cells*32)
                self.assertEqual(state["hierarchical"]["by_domain"]["color"]["cells"], cells//3)
            pooled = row["pooled_by_domain_modality"]["count"]["reply"]
            self.assertEqual(pooled["hierarchical"]["total"], cells*32)
            self.assertEqual(pooled["hierarchical"]["correct"], cells*8)
            self.assertEqual(pooled["delta"], .125)
        self.assertEqual(json.dumps(metrics, sort_keys=True), before)

    def test_zero_domain_cannot_hide_behind_positive_aggregate(self):
        def choices(architecture, seed, name):
            if "/count/" in name: return 0,0
            return (4,4) if architecture=="flat" else (12,12)
        result = analysis.comparison_screen(result_fixture(choices=choices))
        self.assertFalse(result["promising_comparison"])
        for row in result["panels"].values():
            self.assertTrue(row["criteria"]["every_seed_improves"])
            self.assertTrue(row["criteria"]["every_pooled_domain_modality_nonnegative"])
            self.assertFalse(row["criteria"]["every_candidate_domain_seed_modality_nonzero"])

    def test_reply_regression_cannot_hide_behind_action_and_other_domains(self):
        def choices(architecture, seed, name):
            if architecture=="flat": return 8,8
            return (10,7) if "/count/" in name else (12,12)
        result = analysis.comparison_screen(result_fixture(choices=choices))
        for row in result["panels"].values():
            self.assertTrue(row["criteria"]["every_seed_improves"])
            self.assertTrue(row["criteria"]["every_candidate_domain_seed_modality_nonzero"])
            self.assertFalse(row["criteria"]["every_pooled_domain_modality_nonnegative"])
            self.assertEqual(row["pooled_by_domain_modality"]["count"]["reply"]["delta"], -1/32)
        self.assertFalse(result["promising_comparison"])

    def test_each_seed_must_improve_not_only_pooled(self):
        def choices(architecture, seed, name):
            return (8,8) if architecture=="flat" else (4,4) if seed=="8464" else (12,12)
        result = analysis.comparison_screen(result_fixture(choices=choices))
        for row in result["panels"].values():
            self.assertFalse(row["criteria"]["every_seed_improves"])
            self.assertTrue(row["criteria"]["every_pooled_domain_modality_nonnegative"])
            self.assertTrue(row["criteria"]["every_candidate_domain_seed_modality_nonzero"])
        self.assertFalse(result["promising_comparison"])

    def test_primary_panels_independent_and_nonprimary_improvement_not_substitute(self):
        def choices(architecture, seed, name):
            if architecture=="flat" or name.startswith("composed/"): return 4,4
            return 12,12
        result = analysis.comparison_screen(result_fixture(choices=choices))
        self.assertTrue(result["panels"]["fresh_primitive"]["passes"])
        self.assertFalse(result["panels"]["held_composed"]["passes"])
        self.assertFalse(result["promising_comparison"])
        def nonprimary(architecture, seed, name):
            return (32,32) if architecture=="hierarchical" and name.startswith("fresh/") and name.split("/")[2] not in ("d0","d1") else (4,4)
        result = analysis.comparison_screen(result_fixture(choices=nonprimary))
        self.assertFalse(any(panel["passes"] for panel in result["panels"].values()))

    def test_all_architectures_seeds_cells_and_matching_banks_required(self):
        original = result_fixture()
        changes = [lambda r:r.pop("flat"),lambda r:r["flat"].pop("8464"),
            lambda r:r["flat"].__setitem__("8465",r["flat"]["8462"]),
            lambda r:r["hierarchical"]["8464"]["per_bank"].pop("fresh/count/d1/copy/t8"),
            lambda r:r["flat"]["8462"]["per_bank"]["fresh/count/d1/copy/t8"]["bank"].__setitem__("sha256","f"*64),
            lambda r:r["flat"]["8462"]["per_bank"]["fresh/count/d1/copy/t8"]["final_pairs"].__setitem__("total",16)]
        for index, fn in enumerate(changes):
            value = deepcopy(original); fn(value)
            with self.subTest(change=index), self.assertRaises(ValueError): analysis.comparison_screen(value)


if __name__ == "__main__": unittest.main()
