"""One bounded 252-episode CPU fixture; no Torch/model imports or historical data."""
from collections import Counter
from copy import deepcopy
import random
import sys
import unittest
from unittest.mock import patch

from experiments import foundation_context_diagnostic as diagnostic


WORK = diagnostic.WorkLedger()
MALFORMED_VARIANTS = 0


class FoundationContextDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "torch" in sys.modules:
            raise RuntimeError("this bounded validation must run without Torch")
        before = random.getstate()
        cls.bank = diagnostic.build_bank(seed=851900001, pairs_per_group=1, work=WORK)
        cls.global_rng_unchanged = random.getstate() == before

    def test_complete_coverage_and_fixed_queries(self):
        rows = self.bank["rows"]
        self.assertEqual(len(rows), 252)
        self.assertEqual(WORK.report()["generated_rows"], 252)
        self.assertTrue(self.global_rng_unchanged)
        self.assertEqual(Counter(r["family"] for r in rows), dict.fromkeys(diagnostic.FAMILIES, 84))
        self.assertEqual(Counter(r["operator_group"] for r in rows), dict.fromkeys(diagnostic.GROUPS, 36))
        self.assertEqual(len({r["comparison_group"] for r in rows}), 21)
        for row in rows:
            self.assertEqual(row["split"], "diagnostic")
            self.assertEqual([t["target"] for t in row["turns"][:10]], [3] * 10)
            self.assertIn(row["turns"][10]["target"], (0, 1))
            self.assertEqual(row["turns"][11]["target"], 2)
            self.assertEqual(row["known_query_turn"], 10)
            self.assertEqual(row["unknown_query_turn"], 11)
            self.assertLessEqual(sum(len(t["text"].encode()) + 2 for t in row["turns"]), 1024)

    def test_deepest_chain_gap_and_disjoint_roles_without_aliasing(self):
        for row in self.bank["rows"]:
            if row["depth"] != 5:
                continue
            parsed = [diagnostic.oracle.parse_sentence(t["text"])[1] for t in row["turns"]]
            causal = [parsed[i] for i in row["causal_statement_indices"]]
            self.assertEqual(len(causal), 6)
            self.assertEqual(causal[0]["op"], "set")
            self.assertEqual({t["op"] for t in causal[1:]}, {"copy", "advance"})
            self.assertEqual(9 - row["causal_statement_indices"][-1], row["gap"])
            causal_roles = {t["name"] for t in causal} | {t["source"] for t in causal if t["op"] == "copy"}
            other = [parsed[i] for i in range(10) if i not in row["causal_statement_indices"]]
            self.assertEqual(len(other), 4)
            self.assertTrue(all(t["op"] == "set" for t in other))
            self.assertFalse(causal_roles & {t["name"] for t in other})
            self.assertNotIn(parsed[11]["name"], causal_roles | {t["name"] for t in other})
            destinations = [t["name"] for t in causal if t["op"] == "copy"]
            self.assertEqual(len(destinations), len(set(destinations)))
            self.assertNotIn(causal[0]["name"], destinations)

    def test_renaming_is_injective_equal_bytes_and_deranged(self):
        for group in diagnostic.GROUPS:
            a = diagnostic._names(851900001, group, 0, 0)
            b = diagnostic._names(851900001, group, 0, 1)
            self.assertEqual(len(set(a.values())), 10)
            self.assertEqual(len(set(b.values())), 10)
            self.assertEqual(set(a.values()), set(b.values()))
            for role in a:
                self.assertNotEqual(a[role], b[role])
                self.assertEqual(len(a[role].encode()), 3)
                self.assertEqual(len(b[role].encode()), 3)
        self.assertNotIn("torch", sys.modules)

    def test_sixteen_fail_closed_malformed_cases(self):
        global MALFORMED_VARIANTS
        reference = self.bank["rows"][0]

        def reject_row(mutate, *, reseal=True):
            global MALFORMED_VARIANTS
            row = deepcopy(reference)
            mutate(row)
            if reseal:
                row["id"] = diagnostic._hash({k: v for k, v in row.items() if k != "id"})
            MALFORMED_VARIANTS += 1
            with self.assertRaises(ValueError):
                diagnostic.validate_row(row, work=WORK)

        reject_row(lambda r: r.pop("gap"))
        reject_row(lambda r: r.update(hidden_state={"answer": 1}))
        reject_row(lambda r: r["turns"].pop())
        def swap_queries(row):
            row["turns"][10], row["turns"][11] = row["turns"][11], row["turns"][10]
        reject_row(swap_queries)
        reject_row(lambda r: r["turns"][0].update(target=True))
        reject_row(lambda r: r["turns"][0]["observations"]["visual"][0].__setitem__(0, 1.0))
        reject_row(lambda r: r["turns"][11].update(text=r["turns"][10]["text"]))
        reject_row(lambda r: r["recipe"].update(gap=1))
        reject_row(lambda r: r["recipe"].update(seed=True))
        reject_row(lambda r: r.update(variant=False))
        reject_row(lambda r: r.update(id="0" * 64), reseal=False)
        reject_row(lambda r: r.update(split="train"))
        MALFORMED_VARIANTS += 1
        with self.assertRaises(ValueError):
            diagnostic.validate_pair([reference, reference], work=WORK)
        MALFORMED_VARIANTS += 1
        comparison = deepcopy(self.bank["rows"][:12])
        comparison[0:2], comparison[4:6] = comparison[4:6], comparison[0:2]
        with self.assertRaises(ValueError):
            diagnostic.validate_comparison(comparison, work=WORK)
        MALFORMED_VARIANTS += 1
        malformed_bank = {**self.bank, "rows": self.bank["rows"][:-1]}
        with self.assertRaises(ValueError):
            diagnostic.validate_bank(malformed_bank, work=WORK)
        MALFORMED_VARIANTS += 1
        with patch.object(diagnostic.oracle, "english_oracle", return_value=[3] * 12):
            with self.assertRaisesRegex(ValueError, "independent English"):
                diagnostic.validate_row(reference, work=WORK)
        self.assertEqual(MALFORMED_VARIANTS, 16)
        self.assertEqual(WORK.report()["generated_rows"], 252)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
