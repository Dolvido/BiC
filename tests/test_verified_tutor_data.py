"""Pure packaging arithmetic; no archive, lesson generation or Torch import."""
from copy import deepcopy
import unittest
from experiments import verified_tutor_data as data


class DataBoundaryTests(unittest.TestCase):
    def candidates(self):
        return {(d,t): [dict(copies=0 if d in (0,2) else i%4,
            descriptor=dict(plan_sha256="a"*64,bundle_id=cell,pair_index=i,
                pair_sha256={f:"b"*64 for f in data.compiler.FAMILIES})) for i in range(32)]
            for cell,(d,t) in enumerate(data.compiler.CELLS)}

    def test_pools(self):
        values=self.candidates(); before=deepcopy(values)
        pools, report=data._pools(values)
        self.assertEqual(values,before)
        self.assertEqual((pools,report),data._pools({k:list(reversed(v)) for k,v in values.items()}))
        self.assertEqual({k:len(v) for k,v in pools.items()},dict.fromkeys(data.POOLS,288))
        for d,t in data.compiler.CELLS:
            cell=report[f"d{d}/t{t}"]
            if d in (0,2): self.assertTrue(cell["all_pools_identical"]); self.assertEqual(set(cell["overlap"].values()),{16})
            else:
                self.assertEqual(cell["pools"]["copy_emphasis"]["copy_counts"],{"2":8,"3":8})
                self.assertEqual(cell["pools"]["change_emphasis"]["copy_counts"],{"0":8,"1":8})

    def test_underfilled(self):
        values=self.candidates(); values[(0,8)]=values[(0,8)][:15]
        with self.assertRaises(ValueError): data._pools(values)

    def test_fixed_contracts(self):
        inventory=dict(schema=data.compiler.INVENTORY_SCHEMA,plans={},motifs={},replay=[])
        teaching=data._contract(inventory,"c"*64)
        withdrawal=data._contract(inventory,"c"*64,withdrawal=True)
        self.assertEqual((len(teaching["slots"]),len(withdrawal["slots"])),(108,108))
        self.assertEqual((teaching["start_cursor"],withdrawal["start_cursor"]),(1944,2052))
        self.assertEqual(sum(s["replay"] for s in teaching["slots"]),36)
        self.assertTrue(all(s["replay"] for s in withdrawal["slots"]))
        self.assertEqual([s["depth"] for s in teaching["slots"]],[s["depth"] for s in withdrawal["slots"]])
        self.assertEqual(teaching["seed"],852802001)
        self.assertEqual(teaching["limits"],withdrawal["limits"])
        for i,c in enumerate(teaching["chapters"]):
            self.assertEqual(len(c["motifs"]),1 if i<2 else 3)
            self.assertEqual(len(c["realizations"]),1 if i<2 else 3)
        data.compiler.validate_recipe(data.compiler.procedural_recipe(teaching),teaching)

    def test_actual_examples_not_metadata(self):
        original=dict(bundle_id=1944,families={f:[dict(recipe={"seed":1},id="a",
            turns=[dict(text="synthetic",observations={},target=0,reply="synthetic")])]
            for f in data.compiler.FAMILIES})
        same=deepcopy(original); same["bundle_id"]=1945
        same["families"]["color"][0].update(recipe={"seed":2},id="b")
        self.assertEqual(data._learner_digest(original),data._learner_digest(same))
        for key,value in (("text","other"),("target",1),("reply","other"),("observations",{"v":1})):
            changed=deepcopy(same); changed["families"]["color"][0]["turns"][0][key]=value
            self.assertNotEqual(data._learner_digest(original),data._learner_digest(changed))


if __name__ == "__main__": unittest.main()
