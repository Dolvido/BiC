"""Synthetic metadata/packaging proof, without Torch or canonical generation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments import verified_tutor_cycle_data_v3 as data

# Reuse fixture methods only; no old source edit or old test invocation.
spec=importlib.util.spec_from_file_location("cycle_v3_fixture",Path(__file__).with_name("test_verified_tutor_cycle_data.py"))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
fixture.data=data
WORK=fixture.WORK
WORK.update(mock_archive_copies=0)


def exhaustion():
    message="fresh realization exhausted 64 candidates: slot=105, offset=14, family=switch, realization=revalue"
    report=dict(status="failed",error_type="ValueError",error=message,max_candidates_per_fresh_pair=64,
        neural_work=0,teacher_calls=0,wall_seconds=2.,cpu_seconds=1.,
        work=dict(canonical_generate_attempts=100,canonical_generate_completions=100,
            fresh_candidate_attempts=64,fresh_candidate_completions=64),
        collision_rejections=[dict(slot=105,pair_offset=14,family="switch",realization="revalue",
            replay=False,candidate_attempt=i,rejected_members=[dict(member=0,reasons=["protected"])]) for i in range(64)])
    error=ValueError(message);error.compilation_report=report
    return error


class CycleDataV3Tests(unittest.TestCase):
    setUp=fixture.CycleDataTests.setUp
    mock_inventory=fixture.CycleDataTests.mock_inventory
    write=fixture.CycleDataTests.write
    load_author=fixture.CycleDataTests.load_author
    prepare=fixture.CycleDataTests.prepare
    execute=fixture.CycleDataTests.execute

    def compile(self,recipe,**kwargs):
        if getattr(self,"fail_tutor",False) and recipe==self.decision["recipe"]:
            WORK["mocked_compiler_calls"]+=1
            raise self.fail_tutor
        result=fixture.CycleDataTests.compile(self,recipe,**kwargs)
        result["provenance"]=[dict(mock_slot=i) for i in range(108)]
        result["receipt"].update(status="completed",teacher_calls=0)
        result["manifest"].update(schema=data.compiler.SCHEMA,source_sha256=data.compiler.source_hashes(),
            provenance_sha256=data.compiler._hash(result["provenance"]),images_sha256=data.compiler._hash(result["images"]))
        return result

    def make_recovery(self):
        context=self.prepare()
        context.update(source_sha256=data.previous.source_hashes(),adapter_schema=data.previous.ADAPTER_SCHEMA)
        (self.output/"cycle.json").write_bytes(data.io.encoded(context))
        self.context_pin=data.io.digest(self.output/"cycle.json")
        prep=data.io.read(self.output/"preparation.json");prep["manifest_sha256"]=self.context_pin
        (self.output/"preparation.json").write_bytes(data.io.encoded(prep))
        contract=data.io.read(data.ROOT/context["teaching_contract"]["path"])
        result=self.compile(data.compiler.procedural_recipe(contract),admitted_parent_inventory=self.inventory,
            protected_transcripts=self.protected,coverage_contract=contract)
        folder=self.output/"compiled";folder.mkdir();artifacts={}
        for i,image in enumerate(result["images"]):
            name=f"procedural/{i:03d}.pt";artifacts[name]=self.write(folder/name,image)
        for name,key in (("procedural-manifest.json","manifest"),("procedural-provenance.json","provenance")):
            artifacts[name]=self.original_write(folder/name,result[key])
        error=exhaustion()
        failed=dict(schema=data.SCHEMA,adapter_schema=data.previous.ADAPTER_SCHEMA,status="failed",
            source_sha256=data.previous.source_hashes(),compiler_attempts=2,compiler_completions=1,
            compiler_receipts=[result["receipt"],error.compilation_report],error=repr(error),
            neural_work=0,teacher_calls=0,work=dict(guard_attempts=dict(forward=0)),
            input_sha256={data._relative(self.output/"cycle.json"):self.context_pin,
                self.decision_record["path"]:self.decision_record["sha256"]},
            artifact_sha256=artifacts,wall_seconds=7.,cpu_seconds=6.)
        pin=self.original_write(folder/"preparation.json",failed)
        return context,dict(path=data._relative(self.output/"cycle.json"),sha256=self.context_pin),dict(directory=data._relative(folder),preparation_sha256=pin)

    def copy_archive(self,run,name,path,expected):
        # JSON stands in for the archive decoder; real bytes and hashes persist.
        WORK["mock_archive_copies"]+=1
        value=run.read(path,expected)
        target=run.output/name;target.parent.mkdir(parents=True,exist_ok=True)
        raw=Path(path).read_bytes();target.write_bytes(raw)
        run.artifacts[name]=expected
        run.receipt["reused_archive_copies"]+=1
        return value

    def test_exact_finite_failure_only(self):
        error=exhaustion();self.assertTrue(data.known_exhaustion(error,error.compilation_report))
        for mutate in (lambda r:r.update(status="completed"),lambda r:r.update(partial_work_unknown=True),
            lambda r:r.update(neural_work=1),lambda r:r["work"].update(canonical_generate_completions=99),
            lambda r:r["collision_rejections"].pop(),lambda r:r["collision_rejections"][-1].update(candidate_attempt=62)):
            report=deepcopy(error.compilation_report);mutate(report)
            self.assertFalse(data.known_exhaustion(error,report))
        for other in (TimeoutError(str(error)),KeyboardInterrupt(str(error)),ValueError("unrelated")):
            self.assertFalse(data.known_exhaustion(other,error.compilation_report))

    def test_future_exhaustion_uses_same_procedural_bytes(self):
        self.prepare();self.fail_tutor=exhaustion();before=WORK["mocked_compiler_calls"]
        manifest=self.execute()
        self.assertEqual(WORK["mocked_compiler_calls"]-before,3)
        self.assertEqual(manifest["phases"]["teaching"]["tutor"],manifest["phases"]["teaching"]["procedural"])
        self.assertFalse(manifest["treatment"]["teacher_recipe_applied"])
        self.assertFalse(manifest["treatment"]["accepted_local_teacher"])
        self.assertTrue(manifest["treatment"]["zero_treatment_contrast"])
        self.assertNotEqual(manifest["tutor_compilation"]["recipe_sha256"],self.decision["recipe_sha256"])
        self.assertEqual(manifest["teacher_proposal"]["recipe"],self.decision["recipe"])
        receipt=data.io.read(self.output/"compiled"/"preparation.json")
        self.assertEqual((receipt["compiler_attempts"],receipt["compiler_completions"]),(3,2))
        self.assertEqual([r["status"] for r in receipt["compiler_receipts"]],["completed","failed","completed"])
        data.load_manifest(self.output/"compiled",expected_manifest_sha256=data.io.digest(self.output/"compiled"/"manifest.json"))

    def test_unrelated_failure_never_falls_back(self):
        self.prepare();self.fail_tutor=ValueError("wrong truth or admission")
        before=WORK["mocked_compiler_calls"]
        with self.assertRaisesRegex(ValueError,"wrong truth"):self.execute()
        self.assertEqual(WORK["mocked_compiler_calls"]-before,2)
        receipt=data.io.read(self.output/"compiled"/"preparation.json")
        self.assertNotIn("proposal_fallback",receipt)
        self.assertEqual(receipt["status"],"failed")
        self.assertFalse((self.output/"compiled"/"manifest.json").exists())

    def test_pinned_recovery_compiles_withdrawal_only(self):
        context,legacy,reuse=self.make_recovery();destination=self.root/"recovery"
        before=WORK["mocked_compiler_calls"]
        with patch.object(data._Attempt,"copy_archive",autospec=True,side_effect=self.copy_archive):
            manifest=data.compile_data(destination,cycle_sha256=self.context_pin,teacher_decision=self.decision_record,
                legacy_context=legacy,reuse_procedural=reuse,max_seconds=60)
        self.assertEqual(WORK["mocked_compiler_calls"]-before,1)
        self.assertEqual(manifest["phases"]["teaching"]["procedural"],manifest["phases"]["teaching"]["tutor"])
        self.assertEqual(manifest["recovery"]["regenerated_procedural_images"],0)
        self.assertTrue(manifest["recovery"]["skipped_tutor_compilation"])
        self.assertEqual(manifest["recovery"]["prior_failure_cost"]["wall_seconds"],7.)
        receipt=data.io.read(destination/"compiled"/"preparation.json")
        self.assertEqual((receipt["compiler_attempts"],receipt["compiler_completions"]),(1,1))
        self.assertEqual(receipt["reused_archive_copies"],108)
        data.load_manifest(destination/"compiled",expected_manifest_sha256=data.io.digest(destination/"compiled"/"manifest.json"))

    def test_wrong_preparation_pin_prevents_reuse_or_generation(self):
        _,legacy,reuse=self.make_recovery();reuse["preparation_sha256"]="0"*64
        before=(WORK["mocked_compiler_calls"],WORK["mock_archive_copies"])
        with patch.object(data._Attempt,"copy_archive",autospec=True,side_effect=self.copy_archive):
            with self.assertRaisesRegex(ValueError,"input bytes differ"):
                data.compile_data(self.root/"recovery",cycle_sha256=self.context_pin,teacher_decision=self.decision_record,
                    legacy_context=legacy,reuse_procedural=reuse,max_seconds=60)
        self.assertEqual(before,(WORK["mocked_compiler_calls"],WORK["mock_archive_copies"]))

    def test_foreign_teacher_binding_prevents_adoption(self):
        _,legacy,reuse=self.make_recovery()
        self.decision["parent"]["lifetime_updates"]+=1
        before=(WORK["mocked_compiler_calls"],WORK["mock_archive_copies"])
        with self.assertRaisesRegex(ValueError,"another cycle"):
            data.compile_data(self.root/"recovery",cycle_sha256=self.context_pin,teacher_decision=self.decision_record,
                legacy_context=legacy,reuse_procedural=reuse,max_seconds=60)
        self.assertEqual(before,(WORK["mocked_compiler_calls"],WORK["mock_archive_copies"]))


if __name__=="__main__":unittest.main()
