"""Pure JSON and mocked admission tests; no tensors or neural calls.

These fixtures do not claim canonical curriculum correctness. The real provider
owns that proof. Mocked observation packing captures exactly the public input.
"""
from copy import deepcopy
import unittest
from unittest.mock import patch

from experiments import definition_evaluation as evaluation


def rows():
    result = []
    for pair, family in enumerate(("color", "count")):
        for variant in (0, 1):
            turns = [dict(text=f"Observed sentence {turn}.", target=3, reply="ACK") for turn in range(12)]
            # Explicit anchors differ by pair; the evaluator must never infer last.
            anchor = 8 if pair == 0 else 11
            queries = [dict(query_id="source", turn_index=2, kind="source", rule_version=0),
                       dict(query_id="old", turn_index=4, kind="prior_value", rule_version=1),
                       dict(query_id="unknown", turn_index=6, kind="unknown", rule_version=None),
                       dict(query_id="anchor", turn_index=anchor, kind="destination", rule_version=1)]
            for query, target in zip(queries, (1, 0, 2, variant)):
                turns[query["turn_index"]].update(target=target, reply=evaluation.REPLIES[target])
            result.append(dict(id=f"row-{pair}-{variant}", counterfactual_group=f"pair-{pair}",
                family=family, panel="revision", split="dev", variant=variant,
                recipe=dict(layout="original", base_pair_sha256=f"base-{pair}", base_ids=[f"b{pair}-0",f"b{pair}-1"]),
                anchor=dict(query_id="anchor", turn_index=anchor), queries=queries, turns=turns))
    return result


def records(values=None):
    result = []
    for row in values or rows():
        queries = []
        for query in row["queries"]:
            target = row["turns"][query["turn_index"]]["target"]
            logits = [0.] * 4; logits[target] = 1.
            queries.append(dict(query_id=query["query_id"], turn=query["turn_index"], kind=query["kind"],
                rule_version=query["rule_version"], target=target, action=target, logits=logits,
                **evaluation._reply(evaluation.ByteCodec(32).encode(evaluation.REPLIES[target]), target)))
        result.append(dict(**evaluation._row_identity(row), queries=queries))
    return result


def admission(pair, **kwargs):
    if [row["variant"] for row in pair] != [0, 1]:
        raise ValueError("mock pair mismatch")
    return True


class DefinitionEvaluationTests(unittest.TestCase):
    def test_explicit_anchor_and_kind_joint_metrics(self):
        raw = records()
        destination = raw[1]["queries"][-1]
        destination.update(evaluation._reply(evaluation.ByteCodec(32).encode(evaluation.REPLIES[2]), destination["target"]))
        old = raw[2]["queries"][1]
        old.update(action=2, logits=[0.,0.,1.,0.])
        result = evaluation.score_records(raw)
        count = result["overall"]["counts"]
        self.assertEqual(count["anchor_pair_action"]["count"], 2)
        self.assertEqual(count["anchor_pair_reply"]["count"], 1)
        self.assertEqual(count["anchor_pair_both"]["count"], 1)
        self.assertEqual(count["all_query_pair_both"]["count"], 0)
        self.assertEqual(count["source_pair_both"]["count"], 2)
        self.assertEqual(count["prior_value_pair_action"]["count"], 1)
        self.assertEqual(count["unknown_pair_both"]["count"], 2)
        self.assertEqual(count["revision_action"], dict(count=7,total=8,rate=7/8))
        self.assertEqual(count["known_unsupported_ask"]["count"], 1)
        self.assertEqual(count["known_reply_unsupported_ask"]["count"], 1)
        self.assertEqual(set(result["by_family"]), {"color","count"})
        self.assertEqual(set(result["by_panel"]), {"revision"})

    def test_native_tie_exact_reply_and_empty_revision_denominators(self):
        raw = records()
        for row in raw:
            for query in row["queries"]:
                if query["rule_version"] is not None:
                    query["rule_version"] = 0
        q = raw[0]["queries"][1]
        q.update(logits=[1.,1.,0.,0.], action=0)
        q.update(evaluation._reply(evaluation.ByteCodec(32).encode("No "), q["target"]))
        count = evaluation.score_records(raw)["overall"]["counts"]
        self.assertEqual(count["revision_action"], dict(count=0,total=0,rate=None))
        self.assertEqual(count["revision_pair_both"], dict(count=0,total=0,rate=None))
        self.assertEqual(count["reply_parseable"]["count"], 15)
        self.assertEqual(count["query_reply"]["count"], 15)

    def test_recount_rejects_corruption_and_incomplete_pairs(self):
        mutations = [lambda r:r.pop(), lambda r:r[1].update(variant=0),
            lambda r:r[0].update(anchor_turn_index=11),
            lambda r:r[1].update(panel="binding"),
            lambda r:r[0]["queries"][0].update(logits=[float("nan"),1.,0.,0.]),
            lambda r:r[0]["queries"][0].update(reply_exact=False),
            lambda r:r[0]["queries"][2].update(kind="source"),
            lambda r:r[0]["queries"][1].update(rule_version=True),
            lambda r:r[1]["queries"][-1].update(target=0, action=0, logits=[1.,0.,0.,0.],
                **evaluation._reply(evaluation.ByteCodec(32).encode(evaluation.REPLIES[0]),0))]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                raw=records(); mutate(raw)
                with self.assertRaises(ValueError): evaluation.score_records(raw)

    def test_expected_inventory_binds_semantic_slices_and_every_query(self):
        raw=records(); expected=dict(query_inventory=evaluation._inventory(raw))
        self.assertEqual(evaluation.score_records(raw,expected_bank=expected),evaluation.score_records(raw))
        for change in ("omit", "family", "kind"):
            modified=deepcopy(raw)
            for row in modified[:2]:
                if change=="omit": row["queries"].pop(0)
                elif change=="family": row["family"]="switch"
                else: row["queries"][0]["kind"]="prior_value"
            with self.assertRaisesRegex(ValueError,"complete query inventory"):
                evaluation.score_records(modified,expected_bank=expected)

    def test_admit_all_pairs_then_pack_text_only_with_immutable_identity(self):
        original=rows(); inputs=deepcopy(original); events=[]
        def admit(pair,**kwargs):
            events.append("admit"); return admission(pair,**kwargs)
        def pack(text,**kwargs):
            events.append("pack")
            self.assertEqual(text,[[t["text"] for t in row["turns"]] for row in original])
            self.assertTrue(all(type(t) is str for row in text for t in row))
            self.assertEqual(set(kwargs),{"max_turns","max_input_bytes","max_context_tokens"})
            return {}  # Deliberately no tensor construction in this pure proof.
        with patch.object(evaluation.curriculum,"validate_pair",side_effect=admit), patch.object(evaluation,"pack_observations",side_effect=pack):
            bank=evaluation.PreparedDefinitionBank(inputs,role="dev")
        self.assertEqual(events,["admit","admit","pack"])
        identity=bank.identity
        inputs[0]["turns"][0]["text"]="outside mutation"
        identity["anchors"][0]["turn_index"]=0
        self.assertEqual(bank.identity["anchors"][0]["turn_index"],8)
        evaluation.score_records(records(original),expected_bank=bank.identity)

    def test_bad_role_duplicate_or_pair_fails_before_packing(self):
        cases=[("train_fit",rows()),("dev",rows()+rows()),("dev",rows()[:-1]),("dev",rows())]
        cases[-1][1][1]["variant"]=0
        for role, data in cases:
            with self.subTest(role=role,length=len(data)), patch.object(evaluation.curriculum,"validate_pair",side_effect=admission), patch.object(evaluation,"pack_observations") as pack:
                with self.assertRaises(ValueError): evaluation.PreparedDefinitionBank(data,role=role)
                pack.assert_not_called()


if __name__ == "__main__":
    unittest.main()
