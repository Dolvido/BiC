"""Small, no-neural proof of role semantics, binding, order and admission."""
from copy import deepcopy
import hashlib
import json
import unittest

from experiments import definition_basis_curriculum as c


def vector(**values):return [values.get(alias,0) for alias in c.ALIASES]
def row(family,texts):return dict(family=family,turns=[dict(text=text) for text in texts])
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
CASES=(("color","red","yellow","green","blue",1,4,2,3),
       ("count","40","47","41","42",45,52,46,47),
       ("switch","off","on","on","off",105,106,106,105))


class BasisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs={(family,panel,k):c.generate_pair(family,91002+k,panel=panel)
            for family in c.FAMILIES for panel in ("binding","revision") for k in range(3)}
        cls.held={(family,panel):c.generate_pair(family,91008,panel=panel,split="dev")
            for family in c.FAMILIES for panel in ("composition","sequence")}

    def test_hand_computed_three_word_primitive_basis(self):
        for family,a,b,n,nn,la,lb,ln,lnn in CASES:
            texts=[f"Define bralvek for {family}: copy source to destination.",
                f"Define crimlo for {family}: advance source once.",
                f"Define dunvek for {family}: advance destination once.",
                f"Set the {family} of dax to {a}.",f"Set the {family} of wug to {b}.",
                f"Apply crimlo to {family} from dax to wug.",
                f"Apply bralvek to {family} from dax to wug.",
                f"Apply dunvek to {family} from dax to wug.",f"Is the {family} of zot {a}?",
                f"Revise bralvek for {family}: advance source once.",
                f"Is the {family} of wug {nn}?",f"Is the {family} of dax {a}?"]
            expected=[vector()]*3+[vector(dax=la),vector(dax=la,wug=lb),
                vector(dax=ln,wug=lb),vector(dax=ln,wug=ln)]+[vector(dax=ln,wug=lnn)]*5
            with self.subTest(family=family):
                self.assertEqual(c.prefix_labels(row(family,texts)),expected)
                domain,actions,labels,_=c._english_run(texts)
                self.assertEqual(domain,family);self.assertEqual(labels,expected)
                self.assertEqual(actions,[3]*8+[2,3,1,0])
                cast=lambda value:(value=="on") if family=="switch" else int(value) if family=="count" else value
                program=[dict(op="define",word=w,meaning=m) for w,m in zip(("bralvek","crimlo","dunvek"),c.TRAIN_MEANINGS)]
                program += [dict(op="set",name="dax",value=cast(a)),dict(op="set",name="wug",value=cast(b))]
                program += [dict(op="apply",word=w,source="dax",destination="wug") for w in ("crimlo","bralvek","dunvek")]
                program += [dict(op="query",name="zot",value=cast(a)),dict(op="revise",word="bralvek",meaning="advance_source"),
                    dict(op="query",name="wug",value=cast(nn)),dict(op="query",name="dax",value=cast(a))]
                answers,states,_=c._typed_run(program,family)
                self.assertEqual(answers,actions);self.assertEqual(states,expected)

    def test_hand_computed_order_and_switch_double_advance(self):
        specs=(("advance_source_copy","advance source once; copy source to destination",False,True),
            ("copy_advance_source","copy source to destination; advance source once",False,False),
            ("advance_source_copy_advance_source","advance source once; copy source to destination; advance source once",True,True),
            ("copy_advance_source_twice","copy source to destination; advance source once; advance source once",True,False))
        for family,a,b,n,nn,la,lb,ln,lnn in CASES:
            for name,body,twice,destination_advanced in specs:
                with self.subTest(family=family,meaning=name):
                    texts=[f"Define jaskel for {family}: {body}.",f"Set the {family} of dax to {a}.",
                        f"Set the {family} of wug to {b}.",f"Apply jaskel to {family} from dax to wug."]
                    expected=vector(dax=lnn if twice else ln,wug=ln if destination_advanced else la)
                    self.assertEqual(c.prefix_labels(row(family,texts))[-1],expected)
                    cast=lambda value:(value=="on") if family=="switch" else int(value) if family=="count" else value
                    program=[dict(op="define",word="jaskel",meaning=name),dict(op="set",name="dax",value=cast(a)),
                        dict(op="set",name="wug",value=cast(b)),dict(op="apply",word="jaskel",source="dax",destination="wug")]
                    self.assertEqual(c._typed_run(program,family)[1][-1],expected)

    def test_unknown_dependencies_differ_by_role(self):
        for family,a,b,n,nn,la,lb,ln,lnn in CASES:
            texts=[f"Define bralvek for {family}: advance source once.",
                f"Define crimlo for {family}: advance destination once.",
                f"Define dunvek for {family}: copy source to destination.",
                f"Set the {family} of wug to {a}.",
                f"Apply bralvek to {family} from dax to wug.",
                f"Apply crimlo to {family} from dax to wug.",f"Is the {family} of dax {b}?",
                f"Apply dunvek to {family} from dax to wug."]
            expected=[vector()]*3+[vector(wug=la)]*2+[vector(wug=ln)]*2+[vector()]
            self.assertEqual(c.prefix_labels(row(family,texts)),expected)

    def test_all_canonical_panels_pair_truth_and_bounds(self):
        for pair in list(self.pairs.values())+list(self.held.values()):
            self.assertTrue(c.validate_pair(pair))
            at=pair[0]["anchor"]["turn_index"]
            self.assertEqual({r["turns"][at]["target"] for r in pair},{0,1})
            changed=[i for i in range(12) if pair[0]["turns"][i]["text"]!=pair[1]["turns"][i]["text"]]
            self.assertEqual(len(changed),1)
            self.assertEqual(pair[0]["program"][changed[0]]["op"],"revise" if pair[0]["panel"]=="revision" else "define")
            for sample in pair:
                self.assertEqual(len(sample["turns"]),12)
                self.assertLessEqual(sum(len(t["text"].encode())+2 for t in sample["turns"]),1024)
                self.assertTrue(all(len(t["text"].encode())<=128 for t in sample["turns"]))
                if sample["family"]=="count":self.assertTrue(all(30<=e["value"]<=61 for e in sample["program"] if "value" in e))
                unknown=[q for q in sample["queries"] if q["kind"]=="unknown"]
                self.assertEqual(len(unknown),1)
                self.assertEqual(sample["turns"][unknown[0]["turn_index"]]["target"],2)

    def test_two_rules_and_explicit_tested_word_not_latest_definition(self):
        for (family,panel,k),pair in self.pairs.items():
            if panel!="binding":continue
            self.assertEqual(set(pair[0]["recipe"]["meaning_pair"]),set(c.BASIS_PAIRS[k]))
            for sample in pair:
                program=sample["program"];words={e["word"] for e in program[:2]}
                self.assertEqual(len(words),2)
                self.assertEqual(program[4]["word"],sample["recipe"]["word"])
                self.assertEqual(program[7]["word"],sample["recipe"]["other_word"])
                self.assertNotEqual(program[4]["word"],program[7]["word"])
                self.assertEqual(program[4]["source"],program[7]["destination"])
                self.assertEqual(program[4]["destination"],program[7]["source"])
                for query in sample["queries"]:
                    if query["kind"]=="unknown":continue
                    self.assertEqual(query["tested_rule_word"],program[4 if query["turn_index"]<7 else 7]["word"])
                    self.assertEqual(query["rule_version"],0)
        # Both positional cases exist in this fixed, declared bounded fixture.
        self.assertEqual({p[0]["recipe"]["definition_order"] for (f,panel,k),p in self.pairs.items() if panel=="binding"},{0,1})

    def test_revision_old_state_and_unchanged_argument_version(self):
        for (family,panel,k),pair in self.pairs.items():
            if panel!="revision":continue
            for sample in pair:
                texts=[t["text"] for t in sample["turns"]]
                labels=c.prefix_labels(sample)
                self.assertEqual(labels[4],labels[5]);self.assertEqual(labels[5],labels[6])
                self.assertEqual(sample["turns"][4]["target"],1)
                self.assertEqual(sample["turns"][6]["target"],1)
                for query in sample["queries"]:
                    if query["kind"]=="prior_value":self.assertEqual(query["rule_version"],0)
                    elif query["kind"]!="unknown":self.assertEqual(query["rule_version"],1)
                self.assertEqual(sample["program"][3]["source"],sample["program"][7]["destination"])
                # Mutation version can remain old for an unchanged argument;
                # tested application version is nevertheless one for BOTH rows.
                mutations=c._english_run(texts)[3]
                if sample["program"][5]["meaning"]=="advance_destination":
                    self.assertNotEqual(mutations[8],sample["queries"][2]["rule_version"])

    def test_switch_counterfactual_equivalence_is_not_mislabeled(self):
        def result(body,initial):
            texts=[f"Define bralvek for switch: {body}.","Set the switch of dax to off.",
                f"Set the switch of wug to {initial}.","Apply bralvek to switch from dax to wug."]
            return c.prefix_labels(row("switch",texts))[-1]
        self.assertEqual(result("copy source to destination","on"),result("advance destination once","on"))
        self.assertNotEqual(result("copy source to destination","off"),result("advance destination once","off"))
        pair=self.pairs["switch","binding",2]
        self.assertEqual(pair[0]["program"][2]["value"],pair[0]["program"][3]["value"])

    def test_prefix_causality_and_target_metadata_not_inputs(self):
        sample=deepcopy(self.pairs["count","binding",0][0]);before=c.prefix_labels(sample)
        sample["program"]=[];sample["queries"]=[];sample["recipe"]={}
        for turn in sample["turns"]:turn.update(target=2,reply="I need more information.")
        self.assertEqual(c.prefix_labels(sample),before)
        sample["turns"].append(dict(text="Define qelvix for count: advance source once."))
        self.assertEqual(c.prefix_labels(sample)[:-1],before)

    def test_nonce_partition_and_rejections(self):
        old={"plin","trave","glorp","smeb","yorn","quib","snarp","blen","froop","zindle","mave","tulp",
             "vindle","cren","julp","spave","drel","noof"}
        words=[w for values in c.NONCES.values() for w in values]
        self.assertEqual(len(words),len(set(words)));self.assertFalse(set(words)&(old|set(c.ALIASES)))
        for panel in ("composition","sequence"):
            with self.assertRaises(ValueError):c.generate_pair("color",0,panel=panel)
        for bad in (8,10,True):
            with self.assertRaises(ValueError):c.generate_pair("count",0,turns=bad)
        for mutation in ("rule_version","tested_rule_word","target","definition"):
            pair=deepcopy(self.pairs["count","revision",1])
            if mutation=="rule_version":pair[0]["queries"][2]["rule_version"]=0
            elif mutation=="tested_rule_word":pair[0]["queries"][2]["tested_rule_word"]="nottherule"
            elif mutation=="target":pair[0]["turns"][10]["target"]=2
            else:pair[0]["turns"][5]["text"]="Revise bralvek for count: copy source to destination."
            with self.assertRaises(ValueError):c.validate_pair(pair)
        with self.assertRaises(ValueError):c.prefix_labels(row("count",["Set the count of dax to 99.",
            "Define bralvek for count: advance source once.","Apply bralvek to count from dax to wug."]))

    def test_existing_evidence_carrier_exact_and_not_heldout(self):
        bundle=dict(schema=c.BUNDLE_SCHEMA,bundle_id=8032,layout="original",
            families={f:deepcopy(self.pairs[f,"binding",0]) for f in c.FAMILIES})
        evidence=c.bundle_evidence(bundle)
        self.assertEqual(set(evidence),{"bundle_id","layout","families","depth","turns","common_parents_sha256","bundle_rows_sha256"})
        self.assertEqual(evidence["depth"],1);self.assertEqual(evidence["turns"],12)
        self.assertEqual(evidence["bundle_rows_sha256"],digest([[f,bundle["families"][f]] for f in c.FAMILIES]))
        for family,rows in bundle["families"].items():
            detail=evidence["families"][family]
            self.assertEqual(detail["rows_sha256"],digest(rows))
            observations=sum(len(t["text"].encode()) for r in rows for t in r["turns"])
            replies=sum(len(t["reply"].encode()) for r in rows for t in r["turns"])
            self.assertEqual(detail["exposures"],dict(episodes=2,turns=24,observation_bytes=observations,
                observation_tokens=observations+48,reply_target_bytes=replies,reply_target_tokens=replies+24))
        held=deepcopy(bundle);held["families"]["color"]=self.held["color","composition"]
        with self.assertRaises(ValueError):c.bundle_evidence(held)
        mixed=deepcopy(bundle);mixed["families"]["color"]=self.pairs["color","revision",0]
        with self.assertRaises(ValueError):c.bundle_evidence(mixed)


if __name__=="__main__":unittest.main()
