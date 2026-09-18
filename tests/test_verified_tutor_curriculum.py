"""Small canonical CPU compiler proof, never a learner or production preparation.

One synthetic66-slot admitted-plan envelope; only18 selected original parent
descriptors are generated (108 rows). Five successful18-bundle compilations at
micro2 cover every family/depth/turn cell and all realization/replay modes.
At most3000 canonical generator calls/6000 returned rows, six CPU pack calls,
zero models/forwards/optimizers/CUDA/teacher. Failure tests stop at first cause.
"""
from copy import deepcopy
import unittest
from unittest.mock import patch

from experiments import verified_tutor_curriculum as compiler
from experiments import foundation_plan as planning, foundation_curriculum as foundation

WORK = dict(canonical_generate_attempts=0, canonical_generate_completions=0, canonical_rows_returned=0,
    successful_compilations=0, failed_compilations=0, pack_attempts=0, packed_microbatches=0, packed_episodes=0,
    exact_evidence_comparisons=0, model_work=0, teacher_calls=0)
REPORTS = []


class VerifiedTutorCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_generate = foundation.generate_pair
        def observed(*args, **kwargs):
            WORK["canonical_generate_attempts"] += 1
            if WORK["canonical_generate_attempts"] > 3000: raise AssertionError("declared generation ceiling exceeded")
            pair = cls.original_generate(*args, **kwargs)
            WORK["canonical_generate_completions"] += 1
            WORK["canonical_rows_returned"] += len(pair)
            return pair
        patcher=patch.object(foundation,"generate_pair",side_effect=observed)
        patcher.start();cls.addClassCleanup(patcher.stop)
        empty = compiler._hash([])
        cls.plan=planning.build_plan(seed=852890001,stage_updates=10,final_updates=6,micro_batch_size=2,
            rehearsal_every=2,ordering_seed=852890002,
            admission=dict(schema=planning.ADMISSION_SCHEMA,protected_sha256=empty,protected_count=0,realization_attempts={}))
        cls.plan_pin=compiler._hash(cls.plan)
        selected={}
        for slot in cls.plan["bundles"].values(): selected.setdefault((slot["depth"],slot["turns"]),slot["id"])
        if set(selected)!=set(compiler.CELLS): raise AssertionError("fixture must cover every18cell")
        cls.parents, descriptors={},[]
        for cell in compiler.CELLS:
            bundle=selected[cell]
            parents={family:planning.materialize_pair(cls.plan,bundle,family,0) for family in compiler.FAMILIES}
            cls.parents[cell]=parents
            descriptors.append(dict(plan_sha256=cls.plan_pin,bundle_id=bundle,pair_index=0,
                pair_sha256={family:compiler._hash(rows) for family,rows in parents.items()}))
        cls.inventory=dict(schema=compiler.INVENTORY_SCHEMA,plans={cls.plan_pin:cls.plan},
            motifs=dict(balanced=descriptors,second_pool=deepcopy(descriptors)),replay=deepcopy(descriptors))

    def contract(self, *, replay=False, protected=(), realization="independent", inventory=None):
        inventory=self.inventory if inventory is None else inventory
        return dict(schema=compiler.CONTRACT_SCHEMA,inventory_sha256=compiler.inventory_sha256(inventory),
            protected_sha256=compiler._hash(sorted(set(protected))),protected_count=len(set(protected)),
            seed=852890003,start_cursor=5000,micro_batch_size=2,layout="original",
            chapters=[dict(motifs=["balanced","second_pool"],realizations=list(compiler.REALIZATIONS),
                default=dict(motif="balanced",realization=realization))],
            slots=[dict(chapter=0,depth=d,turns=t,replay=replay) for d,t in compiler.CELLS],
            rehearsal_floor_by_depth={str(d):3 for d in range(5)},replay_floor_per_cell=int(replay),
            limits=dict(max_episodes=108,max_observation_bytes=200000,max_reply_target_bytes=40000,
                max_generator_calls=1000,max_seconds=60))

    def compile(self, contract, *, inventory=None, protected=(), recipe=None):
        try:
            result=compiler.compile_curriculum(recipe or compiler.procedural_recipe(contract),
                admitted_parent_inventory=self.inventory if inventory is None else inventory,
                protected_transcripts=protected,coverage_contract=contract)
            WORK["successful_compilations"]+=1;REPORTS.append(deepcopy(result["receipt"]))
            return result
        except BaseException as error:
            WORK["failed_compilations"]+=1;REPORTS.append(deepcopy(getattr(error,"compilation_report",{})))
            raise

    def test_01_complete_deterministic_lessons_and_existing_evidence_compatibility(self):
        from experiments import foundation_layout_training as training
        from experiments.foundation_layout_prepared import PreparedLayoutOwner
        from experiments import foundation_layout_prepared as preparation
        from experiments.sequence_student import SequenceConfig
        contract=self.contract(); before=deepcopy((contract,self.inventory))
        first=self.compile(contract);second=self.compile(contract)
        self.assertEqual(first["images"],second["images"])
        self.assertEqual(first["manifest"],second["manifest"])
        self.assertEqual((contract,self.inventory),before)
        self.assertEqual(first["manifest"]["fresh_bundles"],18)
        self.assertEqual(set(first["manifest"]["coverage"].values()),{1})
        self.assertEqual(len(first["manifest"]["coverage"]),54)
        self.assertEqual(first["manifest"]["unique_transcripts"],108)
        for offset,image in enumerate(first["images"]):
            self.assertEqual(image["bundle"]["bundle_id"],5000+offset)
            exact=training._bundle_evidence(image["bundle"],cursor=5000+offset,layout="original",
                micro_batch_size=2,work=training.curriculum.WorkLedger())
            self.assertEqual(exact,image["expected_evidence"])
            self.assertEqual(training._hash(exact),image["expected_evidence_sha256"])
            WORK["exact_evidence_comparisons"]+=1
        # Packing remains observation-only, despite richer provenance outside it.
        config=SequenceConfig(max_turns=12)
        owner=PreparedLayoutOwner(config=config,layout="original",micro_batch_size=2)
        original_pack=preparation.pack_composition_episodes
        def packed(rows,**kwargs):
            WORK["pack_attempts"]+=1
            result=original_pack(rows,**kwargs)
            WORK["packed_microbatches"]+=1;WORK["packed_episodes"]+=len(rows)
            return result
        pack_patch=patch.object(preparation,"pack_composition_episodes",side_effect=packed)
        pack_patch.start()
        try:
            for image in (first["images"][0],first["images"][-1]):
                token=owner.prepare(image["bundle"],expected_evidence=image["expected_evidence"],
                    expected_evidence_sha256=image["expected_evidence_sha256"])
                batches,_=owner._consume(token,cursor=image["bundle"]["bundle_id"],config=config,layout="original",micro_batch_size=2)
                for batch in batches.values():
                    self.assertNotIn("state_targets",batch["inputs"])
                    self.assertNotIn("recipe",batch["inputs"])
                    self.assertNotIn("target",batch["inputs"])
        finally:
            owner.close();pack_patch.stop()

    def test_02_progressions_keep_original_procedure_and_vary_only_declared_seeds(self):
        for realization in ("rename","revalue"):
            result=self.compile(self.contract(realization=realization))
            for cell,image in zip(compiler.CELLS,result["images"]):
                for family in compiler.FAMILIES:
                    parent=self.parents[cell][family][0]["recipe"]
                    actual=image["bundle"]["families"][family][0]["recipe"]["base_recipe"]
                    self.assertEqual(actual["seed"],parent["seed"])
                    self.assertEqual(actual["structure_split"],parent["structure_split"])
                    fixed="value_seed" if realization=="rename" else "naming_seed"
                    changed="naming_seed" if realization=="rename" else "value_seed"
                    self.assertEqual(actual[fixed],parent[fixed]);self.assertNotEqual(actual[changed],parent[changed])

    def test_03_replay_ignores_teacher_choice_and_retains_exact_parent_rows(self):
        contract=self.contract(replay=True)
        recipe=compiler.procedural_recipe(contract)
        recipe["chapters"][0]=dict(motif="second_pool",realization="revalue")
        result=self.compile(contract,recipe=recipe)
        self.assertEqual(result["manifest"]["replay_bundles"],18)
        self.assertEqual(result["manifest"]["fresh_bundles"],0)
        for cell,image,origin in zip(compiler.CELLS,result["images"],result["provenance"]):
            self.assertTrue(all(x["choice"] is None for x in origin["parents"]))
            for family,rows in image["bundle"]["families"].items():
                self.assertEqual([row["turns"] for row in rows],[row["turns"] for row in self.parents[cell][family]])

    def test_04_invalid_recipe_skeleton_and_missing_pool_fail_before_generation(self):
        contract=self.contract(); recipe=compiler.procedural_recipe(contract)
        cases=[]
        bad=deepcopy(recipe);bad["chapters"][0]["family"]="count";cases.append((bad,contract))
        bad=deepcopy(recipe);bad["explanation"]="Unchecked text";cases.append((bad,contract))
        bad=deepcopy(recipe);bad["chapters"][0]["motif"]="unsupported";cases.append((bad,contract))
        bad=deepcopy(contract);bad["slots"].pop();cases.append((recipe,bad))
        before=WORK["canonical_generate_attempts"]
        for chosen,coverage in cases:
            with self.assertRaises(ValueError):self.compile(coverage,recipe=chosen)
        inventory=deepcopy(self.inventory);inventory["motifs"]["second_pool"].pop()
        with self.assertRaisesRegex(ValueError,"unsupported motif"):
            self.compile(self.contract(inventory=inventory),inventory=inventory)
        self.assertEqual(WORK["canonical_generate_attempts"],before)

    def test_05_protected_replay_and_altered_admitted_parent_are_rejected(self):
        protected={compiler.transcript_sha256(self.parents[compiler.CELLS[0]]["color"][0])}
        with self.assertRaisesRegex(ValueError,"protected") as caught:
            self.compile(self.contract(replay=True,protected=protected),protected=protected)
        self.assertEqual(caught.exception.compilation_report["work"]["materialized_bundles"],0)
        inventory=deepcopy(self.inventory)
        inventory["motifs"]["balanced"][0]["pair_sha256"]["color"]="0"*64
        with self.assertRaisesRegex(ValueError,"parent bytes"):
            self.compile(self.contract(inventory=inventory),inventory=inventory)

    def test_06_actual_token_limit_failure_and_interrupt_preserve_counts_and_generator(self):
        contract=self.contract();contract["limits"]["max_observation_bytes"]=1
        original=foundation.generate_pair
        with self.assertRaisesRegex(ValueError,"actual exposure") as caught:self.compile(contract)
        self.assertGreater(caught.exception.compilation_report["work"]["canonical_generate_completions"],0)
        self.assertIs(foundation.generate_pair,original)
        with patch.object(planning,"_materialize_validated_pair",side_effect=KeyboardInterrupt("fixture")):
            with self.assertRaises(KeyboardInterrupt) as caught:self.compile(self.contract())
        self.assertEqual(caught.exception.compilation_report["error_type"],"KeyboardInterrupt")
        self.assertIs(foundation.generate_pair,original)
