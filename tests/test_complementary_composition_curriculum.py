"""Bounded stdlib semantics/admission checks; no learner or dataset build."""
from copy import deepcopy
import hashlib
import json
import unittest

from experiments import complementary_composition_curriculum as c


def vector(**values): return [values.get(alias, 0) for alias in c.ALIASES]
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
def row(family, texts): return dict(family=family, turns=[dict(text=text) for text in texts])


class ComplementaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs={(family,panel,k):c.generate_pair(family,91000+k,panel=panel)
                   for family in c.FAMILIES for panel in c.PANELS for k in range(5)}
        cls.foreign={(family,split):c.generate_pair(family,91100,split=split)
                     for family in c.FAMILIES for split in ("dev","audit")}

    def test_hand_computed_all_meanings_and_three_worlds(self):
        # Explicit algebraic answers, independent of the provider's executor.
        cases=(("color","red","yellow",1,4,2,3,1,2),
               ("count","40","47",45,52,46,47,53,54),
               ("switch","off","on",105,106,106,105,105,106))
        for family,a,b,la,lb,a1,a2,b1,b2 in cases:
            expected={"CD":(la,a1),"DC":(la,la),"SD":(a1,b1),"DD":(la,b2),
                      "SS":(a2,lb),"DS":(a1,b1),"DCD":(la,a1),"CDD":(la,a2),
                      "SSD":(a2,b1),"SDD":(a1,b2)}
            phrases={"C":"copy source to destination","S":"advance source once","D":"advance destination once"}
            cast=lambda x:(x=="on") if family=="switch" else int(x) if family=="count" else x
            for code,(source,destination) in expected.items():
                with self.subTest(family=family,code=code):
                    texts=[f"Define bralvek for {family}: "+"; ".join(phrases[x] for x in code)+".",
                           f"Set the {family} of dax to {a}.",f"Set the {family} of wug to {b}.",
                           f"Apply bralvek to {family} from dax to wug."]
                    program=[dict(op="define",word="bralvek",meaning=code),
                             dict(op="set",name="dax",value=cast(a)),dict(op="set",name="wug",value=cast(b)),
                             dict(op="apply",word="bralvek",source="dax",destination="wug")]
                    self.assertEqual(c._english_run(texts)[2][-1],vector(dax=source,wug=destination))
                    self.assertEqual(c._typed_run(program,family)[1][-1],vector(dax=source,wug=destination))

    def test_all_thirty_training_cells_and_literal_limits(self):
        for pair in list(self.pairs.values())+list(self.foreign.values()):
            self.assertTrue(c.validate_pair(pair))
            first=pair[0];at=first["anchor"]["turn_index"]
            self.assertEqual({sample["turns"][at]["target"] for sample in pair},{0,1})
            self.assertEqual(first["recipe"]["contrast_codes"],list(c.CONTRASTS[first["recipe"]["seed"]%5]))
            self.assertEqual(set(first["recipe"]["meaning_pair"]),set(first["recipe"]["contrast_codes"]))
            changed=[i for i in range(12) if pair[0]["turns"][i]["text"]!=pair[1]["turns"][i]["text"]]
            self.assertEqual(len(changed),1)
            self.assertEqual(first["program"][changed[0]]["op"],"revise" if first["panel"]=="revision" else "define")
            for sample in pair:
                self.assertEqual(len(sample["turns"]),12);self.assertEqual(len(sample["queries"]),6)
                self.assertLessEqual(sum(len(t["text"].encode())+2 for t in sample["turns"]),1024)
                self.assertTrue(all(len(t["text"].encode())<=128 and len(t["reply"].encode())+1<=32 for t in sample["turns"]))
                self.assertEqual(sample["depth"],2 if sample["recipe"]["contrast_index"]<3 else 3)
                if sample["family"]=="count":
                    self.assertTrue(all(30<=e["value"]<=61 for e in sample["program"] if "value" in e))
                query=next(q for q in sample["queries"] if q["kind"]=="unknown")
                self.assertEqual(sample["turns"][query["turn_index"]]["target"],2)
                self.assertFalse({e["meaning"] for e in sample["program"] if "meaning" in e}&set(c.FORBIDDEN_CODES))

    def test_binding_rule_selection_and_swapped_roles(self):
        for (family,panel,k),pair in self.pairs.items():
            if panel!="binding": continue
            for sample in pair:
                program=sample["program"];recipe=sample["recipe"]
                defined={e["word"]:e["meaning"] for e in program[:2]}
                self.assertEqual(len(defined),2)
                self.assertIn(defined[recipe["other_word"]],c.ATOMS)
                self.assertEqual(program[4]["word"],recipe["word"])
                self.assertEqual(program[7]["word"],recipe["other_word"])
                self.assertEqual(program[4]["source"],program[7]["destination"])
                self.assertEqual(program[4]["destination"],program[7]["source"])
                for query in sample["queries"]:
                    if query["kind"]=="unknown":continue
                    self.assertEqual(query["tested_rule_word"],program[4 if query["turn_index"]<7 else 7]["word"])
                    self.assertEqual(query["rule_version"],0)

    def test_revision_changes_rule_not_past_state(self):
        for (family,panel,k),pair in self.pairs.items():
            if panel!="revision":continue
            sample=pair[0];program=sample["program"];labels=c.prefix_labels(sample)
            self.assertIn(program[0]["meaning"],c.ATOMS)
            self.assertEqual(labels[4],labels[5]);self.assertEqual(labels[5],labels[6])
            self.assertEqual(sample["turns"][4]["target"],1);self.assertEqual(sample["turns"][6]["target"],1)
            self.assertEqual(program[3]["source"],program[7]["destination"])
            self.assertEqual(program[3]["destination"],program[7]["source"])
            for query in sample["queries"]:
                if query["kind"]!="unknown":
                    self.assertEqual(query["rule_version"],0 if query["kind"]=="prior_value" else 1)

    def test_prefix_causality_unknown_and_supervision_separation(self):
        sample=deepcopy(self.pairs["count","binding",4][0]);before=c.prefix_labels(sample)
        sample.update(program=[],queries=[],recipe={})
        for turn in sample["turns"]:turn.update(target=2,reply="fabricated label")
        self.assertEqual(c.prefix_labels(sample),before)
        sample["turns"].append(dict(text="Define qelvix for count: advance source once."))
        self.assertEqual(c.prefix_labels(sample)[:-1],before)
        texts=["Define bralvek for count: advance source once; advance destination once.",
               "Set the count of wug to 40.","Apply bralvek to count from dax to wug.",
               "Define crimlo for count: copy source to destination.","Apply crimlo to count from dax to wug."]
        labels=c.prefix_labels(row("count",texts))
        self.assertEqual(labels[2],vector(wug=46));self.assertEqual(labels[-1],vector())

    def test_partition_and_deterministic_fresh_copy(self):
        words=[w for values in c.NONCES.values() for w in values]
        self.assertEqual(len(words),len(set(words)));self.assertFalse(set(words)&set(c.ALIASES))
        again=c.generate_pair("color",91000)
        self.assertEqual(again,self.pairs["color","binding",0])
        again[0]["turns"][0]["text"]="changed"
        self.assertNotEqual(again,self.pairs["color","binding",0])
        for (family,split),pair in self.foreign.items():
            self.assertTrue({e["word"] for e in pair[0]["program"] if "word" in e}<=set(c.NONCES[split]))

    def test_reserved_programs_and_unsupported_options_fail_closed(self):
        phrases={"C":"copy source to destination","S":"advance source once","D":"advance destination once"}
        for code in c.FORBIDDEN_CODES:
            with self.assertRaises(ValueError):c._typed_run([dict(op="define",word="bralvek",meaning=code)],"color")
            with self.assertRaises(ValueError):c._english_run(["Define bralvek for color: "+"; ".join(phrases[x] for x in code)+"."])
        for panel in ("composition","sequence"):
            with self.assertRaises(ValueError):c.generate_pair("color",0,panel=panel)
        for turns in (8,10,True):
            with self.assertRaises(ValueError):c.generate_pair("count",0,turns=turns)
        for seed in (-1,True):
            with self.assertRaises(ValueError):c.generate_pair("count",seed)

    def test_labels_recipe_query_and_program_tampering_rejected(self):
        for mutation in ("target","reply","text","program","query","contrast","pair_order"):
            pair=deepcopy(self.pairs["count","revision",3])
            if mutation=="target":pair[0]["turns"][10]["target"]=2
            elif mutation=="reply":pair[0]["turns"][10]["reply"]="invented"
            elif mutation=="text":pair[0]["turns"][5]["text"]="Revise bralvek for count: copy source to destination."
            elif mutation=="program":pair[0]["program"][5]["meaning"]="SCS"
            elif mutation=="query":pair[0]["queries"][2]["rule_version"]=0
            elif mutation=="contrast":pair[0]["recipe"]["contrast_index"]=0
            else:pair.reverse()
            with self.assertRaises(ValueError):c.validate_pair(pair)

    def test_bundle_carrier_counts_and_homogeneous_contrast(self):
        def bundle(k=0):return dict(schema=c.BUNDLE_SCHEMA,bundle_id=9760,layout="original",
                    families={family:deepcopy(self.pairs[family,"binding",k]) for family in c.FAMILIES})
        for k in (0,3):
            image=bundle(k);evidence=c.bundle_evidence(image)
            self.assertEqual(set(evidence),{"bundle_id","layout","families","depth","turns","common_parents_sha256","bundle_rows_sha256"})
            self.assertEqual(evidence["depth"],2 if k==0 else 3)
            self.assertEqual(evidence["bundle_rows_sha256"],digest([[f,image["families"][f]] for f in c.FAMILIES]))
            for family,rows in image["families"].items():
                obs=sum(len(t["text"].encode()) for r in rows for t in r["turns"])
                replies=sum(len(t["reply"].encode()) for r in rows for t in r["turns"])
                self.assertEqual(evidence["families"][family]["exposures"],dict(episodes=2,turns=24,
                    observation_tokens=obs+48,observation_bytes=obs,reply_target_tokens=replies+24,reply_target_bytes=replies))
        for mutation in ("contrast","panel","split","duplicate"):
            image=bundle()
            if mutation=="contrast":image["families"]["color"]=deepcopy(self.pairs["color","binding",1])
            elif mutation=="panel":image["families"]["color"]=deepcopy(self.pairs["color","revision",0])
            elif mutation=="split":image["families"]["color"]=deepcopy(self.foreign["color","dev"])
            else:
                for family in c.FAMILIES:image["families"][family]*=2
            with self.assertRaises(ValueError):c.bundle_evidence(image)


if __name__=="__main__":unittest.main()
