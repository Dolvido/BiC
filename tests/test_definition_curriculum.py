"""Bounded stdlib-only admission and independently hand-computed semantics."""
import copy
import hashlib
import json
import unittest

from experiments import definition_curriculum as c


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def row(family,texts):
    return {"family":family,"turns":[{"text":text} for text in texts]}


def vector(**known):
    return [known.get(alias,0) for alias in c.ALIASES]


class DefinitionCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ten requested pairs total; all subsequent tests reuse these in memory.
        cls.pairs={(f,p):c.generate_pair(f,91001,split="dev" if p=="composition" else "train",panel=p)
                   for f in c.FAMILIES for p in c.PANELS}
        cls.dev_binding=c.generate_pair("color",91001,split="dev",panel="binding")

    def test_hand_computed_copy_revision_and_unknown_all_families(self):
        for family,a,b,a_label,next_label,b_label in (
                ("color","red","blue",1,2,3),
                ("count","12","20",17,18,25),
                ("switch","off","on",105,106,106)):
            with self.subTest(family=family):
                texts=[f"Define plin for {family}: copy source to destination.",
                    f"Set the {family} of dax to {a}.",
                    f"Apply plin to {family} from dax to wug.",
                    f"Is the {family} of zot {b}?",
                    f"Revise plin for {family}: copy source to destination; advance destination once.",
                    f"Is the {family} of wug {a}?",
                    f"Apply plin to {family} from dax to fep.",
                    f"Set the {family} of dax to {b}.",
                    f"Is the {family} of wug {a}?",
                    f"Apply plin to {family} from zot to wug.",
                    f"Is the {family} of wug {a}?",
                    f"Is the {family} of dax {b}?"]
                expected=[vector(),vector(dax=a_label),vector(dax=a_label,wug=a_label)]
                expected += [expected[-1]]*3
                expected += [vector(dax=a_label,wug=a_label,fep=next_label),
                    vector(dax=b_label,wug=a_label,fep=next_label),
                    vector(dax=b_label,wug=a_label,fep=next_label),
                    vector(dax=b_label,fep=next_label),vector(dax=b_label,fep=next_label),
                    vector(dax=b_label,fep=next_label)]
                self.assertEqual(c.prefix_labels(row(family,texts)),expected)
                domain,actions,labels,causes=c._english_run(texts)
                self.assertEqual(domain,family)
                self.assertEqual(actions,[3,3,3,2,3,1,3,3,1,3,2,1])
                self.assertEqual(labels,expected)
                self.assertEqual(causes[5],0)
                self.assertEqual(causes[8],0)
                self.assertEqual(causes[10],1)
                typed_a=(a=="on") if family=="switch" else int(a) if family=="count" else a
                typed_b=(b=="on") if family=="switch" else int(b) if family=="count" else b
                program=[dict(op="define",word="plin",meaning="copy"),dict(op="set",name="dax",value=typed_a),
                    dict(op="apply",word="plin",source="dax",destination="wug"),dict(op="query",name="zot",value=typed_b),
                    dict(op="revise",word="plin",meaning="copy_advance_destination"),dict(op="query",name="wug",value=typed_a),
                    dict(op="apply",word="plin",source="dax",destination="fep"),dict(op="set",name="dax",value=typed_b),
                    dict(op="query",name="wug",value=typed_a),dict(op="apply",word="plin",source="zot",destination="wug"),
                    dict(op="query",name="wug",value=typed_a),dict(op="query",name="dax",value=typed_b)]
                self.assertEqual(c._typed_run(program,family),(actions,expected,causes))

    def test_hand_computed_ordered_compositions_all_families(self):
        for family,a,advanced,a_label,next_label in (("color","yellow","red",4,1),
                ("count","9","10",14,15),("switch","on","off",106,105)):
            for meaning,phrase,destination,answer in (
                    ("advance_source_copy","advance source once; copy source to destination",next_label,0),
                    ("copy_advance_source","copy source to destination; advance source once",a_label,1)):
                with self.subTest(family=family,meaning=meaning):
                    texts=[f"Define snarp for {family}: {phrase}.",f"Set the {family} of dax to {a}.",
                        f"Apply snarp to {family} from dax to wug.",
                        f"Is the {family} of dax {advanced}?",f"Is the {family} of wug {a}?"]
                    expected=[vector(),vector(dax=a_label)]+[vector(dax=next_label,wug=destination)]*3
                    _,actions,labels,_=c._english_run(texts)
                    self.assertEqual(actions,[3,3,3,1,answer]);self.assertEqual(labels,expected)
                    typed_a=(a=="on") if family=="switch" else int(a) if family=="count" else a
                    typed_next=(advanced=="on") if family=="switch" else int(advanced) if family=="count" else advanced
                    program=[dict(op="define",word="snarp",meaning=meaning),dict(op="set",name="dax",value=typed_a),
                        dict(op="apply",word="snarp",source="dax",destination="wug"),
                        dict(op="query",name="dax",value=typed_next),dict(op="query",name="wug",value=typed_a)]
                    actual_actions,actual_labels,_=c._typed_run(program,family)
                    self.assertEqual(actual_actions,actions);self.assertEqual(actual_labels,expected)

    def test_prefix_is_causal_and_ignores_hidden_metadata(self):
        source=copy.deepcopy(self.pairs["color","revision"][0])
        before=c.prefix_labels(source)
        source["program"]=[];source["recipe"]={"fabricated":"do not use"}
        for turn in source["turns"]:turn.update(target=0,reply="No.")
        self.assertEqual(c.prefix_labels(source),before)
        word=self.pairs["color","revision"][0]["recipe"]["word"]
        source["turns"].append({"text":f"Revise {word} for color: advance source once; copy source to destination."})
        after=c.prefix_labels(source)
        self.assertEqual(after[:-1],before);self.assertEqual(after[-1],before[-1])

    def test_generated_pairs_are_complete_admitted_and_bounded(self):
        for (family,panel),pair in self.pairs.items():
            with self.subTest(family=family,panel=panel):
                self.assertTrue(c.validate_pair(pair))
                self.assertEqual({r["turns"][11]["target"] for r in pair},{0,1})
                changed=[i for i in range(12) if pair[0]["turns"][i]["text"]!=pair[1]["turns"][i]["text"]]
                self.assertEqual(changed,[5 if panel=="revision" else 0])
                for sample in pair:
                    self.assertEqual(len(sample["turns"]),12)
                    self.assertLessEqual(sum(len(t["text"].encode())+2 for t in sample["turns"]),1024)
                    self.assertTrue(all(len(t["text"].encode())<=128 for t in sample["turns"]))
                    self.assertEqual(sample["anchor"],{"query_id":"query-11","turn_index":11})
                    queries=sample["queries"]
                    self.assertTrue({"source","destination","unknown","prior_value"}<=set(q["kind"] for q in queries))
                    for query in queries:
                        self.assertTrue(sample["turns"][query["turn_index"]]["text"].startswith("Is "))
                        if query["kind"]=="unknown":self.assertEqual(sample["turns"][query["turn_index"]]["target"],2)
                    self.assertEqual(queries[-1]["rule_version"],1 if panel=="revision" else 0)
                    self.assertEqual(sample["semantic_partition"],"heldout_composition" if panel=="composition" else "binding_revision")
                    if family=="count":
                        self.assertTrue(all(30<=event["value"]<=61 for event in sample["program"] if "value" in event))

    def test_determinism_and_nonce_meaning_binding(self):
        self.assertEqual(c.generate_pair("count",91001,panel="binding"),self.pairs["count","binding"])
        for pair in self.pairs.values():
            self.assertEqual(pair[0]["recipe"]["word"],pair[1]["recipe"]["word"])
            point=5 if pair[0]["panel"]=="revision" else 0
            self.assertNotEqual(pair[0]["program"][point]["meaning"],pair[1]["program"][point]["meaning"])
        all_words=[word for values in c.NONCES.values() for word in values]
        self.assertEqual(len(all_words),len(set(all_words)))
        self.assertFalse(set(all_words)&set(c.ALIASES))
        self.assertIn(self.dev_binding[0]["recipe"]["word"],c.NONCES["dev"])
        self.assertNotIn(self.dev_binding[0]["recipe"]["word"],c.NONCES["train"])

    def test_pair_rejects_mutations_and_crossing(self):
        original=self.pairs["count","revision"]
        for field in ("target","text","program","recipe","observations"):
            pair=copy.deepcopy(original)
            if field=="target":pair[0]["turns"][11]["target"]=2
            elif field=="text":pair[0]["turns"][1]["text"]="Set the count of dax to 22."
            elif field=="program":pair[0]["program"][1]["value"]+=1
            elif field=="recipe":pair[1]["recipe"]["seed"]+=1
            else:pair[0]["turns"][0]["observations"]["tokens"]=[1]
            with self.subTest(field=field),self.assertRaises(ValueError):c.validate_pair(pair)
        with self.assertRaises(ValueError):c.validate_pair(original[::-1])
        with self.assertRaises(ValueError):c.validate_pair([original[0],self.pairs["count","binding"][1]])

    def test_invalid_generation_contract_rejected(self):
        for options in ({"split":"train","panel":"composition"},{"turns":8},{"turns":10},
                {"turns":True},{"split":"unknown"},{"panel":"unknown"}):
            with self.subTest(options=options),self.assertRaises(ValueError):c.generate_pair("color",1,**options)
        with self.assertRaises(ValueError):c.generate_pair("count",-1)
        with self.assertRaises(ValueError):c.generate_pair("count",True)

    def test_malformed_visible_english_and_count_overflow_rejected(self):
        cases=[("count",["Set the count of dax to 100."]),
            ("count",["Is the count of dax 100?"]),
            ("count",["Define plin for count: copy source to destination; advance destination once.",
                "Set the count of dax to 99.","Apply plin to count from dax to wug."]),
            ("color",["Set the color of dax to red.","Is the count of dax 1?"]),
            ("color",["Revise plin for color: copy source to destination."]),
            ("color",["Define dax for color: copy source to destination."]),
            ("switch",["Set the switch of dax to true."]),
            ("color",["Define plin for color: copy source to destination.","Apply plin to color from dax to dax."])]
        for family,texts in cases:
            with self.subTest(texts=texts),self.assertRaises(ValueError):c.prefix_labels(row(family,texts))
        with self.assertRaises(ValueError):c._typed_run([dict(op="set",name="dax",value=100)],"count")
        with self.assertRaises(ValueError):c._typed_run([dict(op="set",name="dax",value=True)],"count")

    def test_bundle_evidence_matches_independent_carrier_counts(self):
        bundle={"schema":c.BUNDLE_SCHEMA,"bundle_id":9000,"layout":"original",
            "families":{family:copy.deepcopy(self.pairs[family,"binding"]) for family in c.FAMILIES}}
        actual=c.bundle_evidence(bundle)
        expected={"bundle_id":9000,"layout":"original","families":{},"depth":2,"turns":12}
        all_parents=[]
        for family,rows in bundle["families"].items():
            parent=[[family,rows[0]["recipe"]["base_pair_sha256"],rows[0]["recipe"]["base_ids"]]]
            all_parents.extend(parent)
            texts=[t["text"].encode() for sample in rows for t in sample["turns"]]
            replies=[t["reply"].encode() for sample in rows for t in sample["turns"]]
            expected["families"][family]={"rows_sha256":digest(rows),"recipes_sha256":digest([rows[0]["recipe"]]),
                "common_parents_sha256":digest(parent),"exposures":{"episodes":2,"turns":24,
                "observation_tokens":sum(map(len,texts))+48,"observation_bytes":sum(map(len,texts)),
                "reply_target_tokens":sum(map(len,replies))+24,"reply_target_bytes":sum(map(len,replies))}}
        expected["common_parents_sha256"]=digest(all_parents)
        expected["bundle_rows_sha256"]=digest([[family,bundle["families"][family]] for family in c.FAMILIES])
        self.assertEqual(actual,expected)

    def test_bundle_rejects_mixed_panel_repeated_pair_and_nontraining(self):
        original={"schema":c.BUNDLE_SCHEMA,"bundle_id":9000,"layout":"original",
            "families":{family:copy.deepcopy(self.pairs[family,"binding"]) for family in c.FAMILIES}}
        mixed=copy.deepcopy(original);mixed["families"]["count"]=self.pairs["count","revision"]
        repeated=copy.deepcopy(original)
        for family in c.FAMILIES:repeated["families"][family]*=2
        heldout=copy.deepcopy(original);heldout["families"]["color"]=self.dev_binding
        for bundle in (mixed,repeated,heldout):
            with self.assertRaises(ValueError):c.bundle_evidence(bundle)


if __name__=="__main__":unittest.main()
