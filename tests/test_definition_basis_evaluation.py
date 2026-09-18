"""Synthetic records and mocked admission/packing only; no canonical generation.

The curriculum's separate proof establishes oracle truth. Here mocked admission
isolates the evaluator's trust boundary and labels-to-observation separation.
"""
from copy import deepcopy
import unittest
from unittest.mock import patch

from experiments import definition_basis_evaluation as evaluation


WORK = dict(synthetic_pairs=4, synthetic_episodes=8, synthetic_queries=48,
            score_calls=0, mock_pair_admissions=0, mock_text_packings=0)


def _fixtures():
    rows=[]
    for index,panel in enumerate(evaluation.PANELS):
        for variant in (0,1):
            turns=[dict(text=f"Observed sentence {i}.", target=3, reply="ACK") for i in range(12)]
            if panel == "revision":
                specs=((4,"prior_value",0,"jaskel"),(6,"prior_value",0,"jaskel"),
                       (8,"source",1,"jaskel"),(9,"unknown",None,None),
                       (10,"destination",1,"jaskel"),(11,"source",1,"jaskel"))
                anchor=10
            else:
                specs=((5,"source",0,"jaskel"),(6,"destination",0,"jaskel"),
                       (8,"source",0,"klunvo"),(9,"destination",0,"klunvo"),
                       (10,"unknown",None,None),(11,"source",0,"klunvo"))
                anchor=6
            queries=[]
            for position,kind,version,word in specs:
                queries.append(dict(query_id=f"q{position}",turn_index=position,kind=kind,
                                    rule_version=version,tested_rule_word=word))
                target=2 if kind=="unknown" else variant if position==anchor else position%2
                turns[position].update(target=target,reply=evaluation.REPLIES[target])
            rows.append(dict(id=f"row{index}-{variant}",counterfactual_group=f"pair{index}",
                family=evaluation.FAMILIES[index%3],panel=panel,split="dev",variant=variant,
                recipe=dict(base_pair_sha256=f"base{index}",base_ids=[f"base{index}-0",f"base{index}-1"]),
                anchor=dict(query_id=f"q{anchor}",turn_index=anchor),queries=queries,turns=turns))
    return rows


ROWS=_fixtures()


def records(rows=ROWS):
    result=[]
    for row in rows:
        queries=[]
        for q in row["queries"]:
            target=row["turns"][q["turn_index"]]["target"]
            queries.append(dict(query_id=q["query_id"],turn=q["turn_index"],kind=q["kind"],
                rule_version=q["rule_version"],tested_rule_word=q["tested_rule_word"],target=target,
                action=target,logits=[float(i==target) for i in range(4)],
                **evaluation._reply(evaluation.ByteCodec(32).encode(evaluation.REPLIES[target]),target)))
        result.append(dict(**evaluation._row_identity(row),queries=queries))
    return result


def score(rows,**kwargs):
    WORK["score_calls"]+=1
    return evaluation.score_records(rows,**kwargs)


def admit(pair):
    WORK["mock_pair_admissions"]+=1
    if [row["variant"] for row in pair]!=[0,1]:raise ValueError("mock pair admission rejected")
    return True


class DefinitionBasisEvaluationTests(unittest.TestCase):
    def test_four_panels_and_entire_paired_query_inventory(self):
        result=score(records());c=result["overall"]["counts"]
        self.assertEqual(result["overall"]["query_turns"],48)
        self.assertEqual(c[evaluation.PRIMARY_METRIC],dict(count=4,total=4,rate=1.0))
        self.assertEqual(set(result["by_panel"]),set(evaluation.PANELS))
        self.assertEqual(c["known_both"]["count"],40)
        self.assertEqual(c["unknown_both"]["count"],8)

    def test_second_word_free_reply_failure_defeats_pair_despite_anchor(self):
        raw=records();q=raw[0]["queries"][2]
        self.assertEqual(q["tested_rule_word"],"klunvo")
        q.update(evaluation._reply(evaluation.ByteCodec(32).encode(evaluation.REPLIES[2]),q["target"]))
        c=score(raw)["overall"]["counts"]
        self.assertEqual(c["anchor_pair_both"]["count"],4)
        self.assertEqual(c["all_query_pair_action"]["count"],4)
        self.assertEqual(c["all_query_pair_reply"]["count"],3)
        self.assertEqual(c[evaluation.PRIMARY_METRIC]["count"],3)

    def test_application_version_includes_unchanged_roles(self):
        raw=records();expected=dict(query_inventory=evaluation._inventory(raw))
        c=score(raw,expected_bank=expected)["by_panel"]["revision"]["counts"]
        self.assertEqual(c["revision_both"],dict(count=6,total=6,rate=1.0))
        self.assertEqual(c["prior_value_both"]["count"],4)
        for row in raw[2:4]:row["queries"][2]["rule_version"]=0
        with self.assertRaisesRegex(ValueError,"complete query inventory"):score(raw,expected_bank=expected)

    def test_word_and_query_metadata_cannot_be_relabelled(self):
        original=records();expected=dict(query_inventory=evaluation._inventory(original))
        for field,value in (("tested_rule_word","marnix"),("rule_version",1),("kind","prior_value")):
            raw=deepcopy(original)
            for row in raw[:2]:row["queries"][2][field]=value
            with self.assertRaisesRegex(ValueError,"complete query inventory"):score(raw,expected_bank=expected)
        raw=deepcopy(original);raw[1]["queries"][2]["tested_rule_word"]="marnix"
        with self.assertRaisesRegex(ValueError,"pair query coordinates"):score(raw)

    def test_reply_bytes_after_eos_are_not_exact_and_action_ties(self):
        raw=records();q=raw[0]["queries"][1]
        q.update(logits=[1.,1.,0.,0.],action=0)
        q.update(evaluation._reply(evaluation.ByteCodec(32).encode(evaluation.REPLIES[0])+[99],0))
        self.assertEqual(q["reply_action"],0);self.assertFalse(q["reply_exact"])
        c=score(raw)["overall"]["counts"]
        self.assertEqual(c["anchor_pair_action"]["count"],4)
        self.assertEqual(c["anchor_pair_reply"]["count"],3)

    def test_malformed_native_or_metadata_records_rejected(self):
        changes=(lambda r:r.pop(),lambda r:r[1].update(variant=0),
            lambda r:r[0]["queries"][0].update(logits=[float("nan"),0.,0.,0.]),
            lambda r:r[0]["queries"][0].update(tested_rule_word=None),
            lambda r:r[0]["queries"][0].update(rule_version=None),
            lambda r:r[0]["queries"][4].update(tested_rule_word="jaskel"),
            lambda r:r[0]["queries"][0].update(reply_exact=False))
        for change in changes:
            raw=records();change(raw)
            with self.assertRaises(ValueError):score(raw)

    def test_mocked_admission_then_text_only_packing_and_immutable_inventory(self):
        original=deepcopy(ROWS);events=[];packed=[]
        def admission(pair):events.append("admit");return admit(pair)
        def pack(texts,**kwargs):
            WORK["mock_text_packings"]+=1;events.append("pack");packed.append(deepcopy(texts))
            self.assertTrue(all(type(t) is str for row in texts for t in row))
            self.assertEqual(set(kwargs),{"max_turns","max_input_bytes","max_context_tokens"})
            return {}
        with patch.object(evaluation.curriculum,"validate_pair",side_effect=admission),patch.object(evaluation,"pack_observations",side_effect=pack):
            bank=evaluation.PreparedDefinitionBank(original,role="dev")
            modified=deepcopy(original)
            # Deliberately impossible oracle metadata is accepted only by this
            # mocked provider; it must still never become an observation input.
            for row in modified:
                row["turns"][0].update(target=1,reply="hidden target")
                row["queries"][0]["tested_rule_word"]="othermetadata"
            evaluation.PreparedDefinitionBank(modified,role="dev")
        self.assertEqual(events,["admit"]*4+["pack"]+["admit"]*4+["pack"])
        self.assertEqual(packed[0],packed[1])
        identity=bank.identity
        self.assertEqual(identity["primary_metric"],"all_query_pair_both")
        self.assertEqual(identity["query_inventory"][0]["queries"][2]["tested_rule_word"],"klunvo")
        original[0]["queries"][0]["tested_rule_word"]="externalchange"
        identity["query_inventory"][0]["queries"][0]["tested_rule_word"]="externalchange"
        self.assertEqual(bank.identity["query_inventory"][0]["queries"][0]["tested_rule_word"],"jaskel")
        score(records(),expected_bank=bank.identity)

    def test_oracle_rejection_and_bad_role_stop_before_packing(self):
        for rejected in (False,ValueError("oracle conflict")):
            with patch.object(evaluation.curriculum,"validate_pair",side_effect=rejected if isinstance(rejected,Exception) else None,
                              return_value=rejected),patch.object(evaluation,"pack_observations") as pack:
                with self.assertRaises(ValueError):evaluation.PreparedDefinitionBank(ROWS,role="dev")
                pack.assert_not_called()
        with patch.object(evaluation.curriculum,"validate_pair",side_effect=admit),patch.object(evaluation,"pack_observations") as pack:
            with self.assertRaises(ValueError):evaluation.PreparedDefinitionBank(ROWS,role="train_fit")
            pack.assert_not_called()


if __name__=="__main__":unittest.main()
