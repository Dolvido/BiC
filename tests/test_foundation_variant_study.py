"""CPU integration of study durability/gates, separate from formal CUDA proof."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments import foundation_variant_study as study
from experiments.foundation_admission import repair_plan
from experiments.foundation_evidence import prepare_training
from experiments.foundation_plan import build_plan
from experiments.foundation_plan_index import AuthenticatedPlanIndex
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.train_cognitive import atomic_json, atomic_checkpoint

WORK = dict(physical_optimizer_updates=0,drawn_episode_exposures=0,
            neural_attempted_episode_exposures=0,completed_microbatch_episode_exposures=0)
ORIGINAL_TRAINER=study._trainer


class VariantStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config=SequenceConfig(width=8,layers=4,heads=2,feedforward=16,max_turns=12,max_positions=512)
        cls.plan,cls.admission=repair_plan(build_plan(seed=847100101,stage_updates=10,final_updates=6,
            micro_batch_size=2,rehearsal_every=2,ordering_seed=8471),protected_transcripts=[])
        cls.manifest,cls.transcripts=prepare_training(cls.plan,protected_transcripts=[],anchor_limit=1)
        cls.index=AuthenticatedPlanIndex(cls.plan,admission_protected_transcripts=[],
            protected_transcripts=[],admission_receipt=cls.admission)
        cls.data={"stages":{"calibration":dict(plan=cls.plan,admission=cls.admission,manifest=cls.manifest,
            training_transcripts=cls.transcripts,admission_protected_transcripts=[],protected_transcripts=[])}}

    def fixture(self,root):
        (root/"data-verification").mkdir()
        atomic_json(root/"protocol.json",dict(test_fixture=True))
        atomic_json(root/"preparation.json",dict(test_fixture=True))
        atomic_json(root/"data-verification/receipt.json",dict(test_fixture=True))
        initial=checkpoint_digest(build_sequence_student(study.CALIBRATION_SEED,config=self.config))
        return dict(initial_weights_sha256={str(study.CALIBRATION_SEED):initial},
            source_sha256={},execution_profile={"cpu_fixture":True})

    def cpu_trainer(self,*args,**kwargs):
        kwargs["device"]="cpu"
        learner=ORIGINAL_TRAINER(*args,**kwargs)
        original=learner.step
        def counted():
            try:
                return original()
            finally:
                report=learner.last_report
                for key in WORK:
                    WORK[key]+=report[key] or 0
        learner.step=counted
        return learner

    def patches(self):
        from contextlib import ExitStack
        stack=ExitStack()
        stack.enter_context(patch.object(study,"CONFIG",self.config))
        stack.enter_context(patch.object(study,"TOTALS",{"calibration":66,"main":3072}))
        stack.enter_context(patch.object(study,"STEPS",{"calibration":(0,3,66),"main":study.STEPS["main"]}))
        stack.enter_context(patch.object(study,"_sources",return_value=None))
        stack.enter_context(patch.object(study,"_trainer",side_effect=self.cpu_trainer))
        stack.enter_context(patch("experiments.execution_profile.assert_strict_profile",return_value=None))
        stack.enter_context(patch.object(torch.cuda,"reset_peak_memory_stats",return_value=None))
        stack.enter_context(patch.object(torch.cuda,"max_memory_allocated",return_value=0))
        stack.enter_context(patch.object(torch.cuda,"max_memory_reserved",return_value=0))
        return stack

    def test_declared_budget_and_fixed_initialization_pairing(self):
        self.assertEqual(study.TOTALS,{"calibration":510,"main":3072})
        self.assertEqual(study.contract()["formal_updates"],6*510+6*3072)
        self.assertEqual(study.contract()["formal_episode_exposures"],(6*510+6*3072)*96)
        jobs=study.jobs("calibration")
        self.assertEqual(len(jobs),6)
        self.assertEqual({j["seed"] for j in jobs},{8461})
        self.assertEqual({j["rate"] for j in jobs},{.0003,.001,.003})
        self.assertEqual(study.contract()["order"],"curriculum")
        with self.assertRaises(ValueError):
            study.jobs("main")
        selected={"selected":{a:{"rate":.001} for a in study.ARCHITECTURES}}
        main=study.jobs("main",selected)
        self.assertEqual({j["seed"] for j in main},{8462,8463,8464})
        self.assertEqual(len(main),6)

    def test_complete_cpu_jobs_journals_and_official_optimizer_verification(self):
        with tempfile.TemporaryDirectory() as tmp, self.patches():
            root=Path(tmp); protocol=self.fixture(root); runtime=protocol["execution_profile"]
            for arch in study.ARCHITECTURES:
                (root/"calibration"/arch).mkdir(parents=True)
                declared=[j for j in study.jobs("calibration") if j["architecture"]==arch]
                for job in declared:
                    receipt=study._train_job(root,"calibration",job,self.data,self.index,protocol,runtime)
                    self.assertEqual(receipt["physical_optimizer_updates"],66)
                    self.assertEqual(receipt["completed_microbatch_episode_exposures"],396)
                atomic_json(root/"calibration"/arch/"worker.json",dict(schema=study.SCHEMA,status="completed",
                    stage="calibration",architecture=arch,automatic_promotion=False,
                    protocol_sha256=study.old.file_hash(root/"protocol.json"),execution_profile=runtime,
                    completed_jobs=[j["id"] for j in declared]))
            with patch.object(study,"load_protocol",return_value=protocol), patch.object(study,"_data_gate",return_value={}), \
                    patch.object(study.data_api,"load",return_value=self.data), patch.object(study,"_index",return_value=self.index):
                result=study.verify(root,"calibration")
            self.assertEqual(len(result["results"]),6)
            self.assertTrue(all(r["exact_official_checkpoint_restores"] for r in result["results"].values()))
            self.assertTrue(result["neural_training_or_inference"] is False)
            # A receipt cannot relabel an incomplete phase as ready for scoring.
            worker=root/"calibration/hierarchical/worker.json"
            value=study.old.read(worker); value["completed_jobs"].pop(); atomic_json(worker,value)
            with self.assertRaises(ValueError):
                study._inputs(root,"calibration",protocol)

    def test_checkpoint_cannot_relabel_architecture_or_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); protocol=self.fixture(root)
            job=study.jobs("calibration")[0]
            path=root/"calibration"/job["architecture"]/job["id"]/"checkpoint-000000.pt"
            path.parent.mkdir(parents=True)
            payload=dict(schema=study.SCHEMA,stage="calibration",job=copy.deepcopy(job),
                protocol_sha256=study.old.file_hash(root/"protocol.json"),learner={"cursor":0},
                weights_sha256="fixture",execution_profile=protocol["execution_profile"])
            atomic_checkpoint(path,payload)
            self.assertEqual(study._checkpoint(root,"calibration",job,0,protocol)["job"],job)
            payload["job"]["architecture"]="hierarchical"; atomic_checkpoint(path,payload)
            with self.assertRaises(ValueError): study._checkpoint(root,"calibration",job,0,protocol)
            payload["job"]=job; payload["learner"]["cursor"]=True; atomic_checkpoint(path,payload)
            with self.assertRaises(ValueError): study._checkpoint(root,"calibration",job,0,protocol)

    def test_completed_arithmetic_survives_journal_publication_failure(self):
        with tempfile.TemporaryDirectory() as tmp, self.patches():
            root=Path(tmp); protocol=self.fixture(root)
            job=study.jobs("calibration")[0]
            (root/"calibration"/job["architecture"]).mkdir(parents=True)
            original=study.old.append_journal
            def fail_completion(path,event):
                if event["event"]=="completed":
                    raise OSError("intentional durable journal failure after optimizer update")
                original(path,event)
            with patch.object(study.old,"append_journal",side_effect=fail_completion):
                with self.assertRaises(OSError):
                    study._train_job(root,"calibration",job,self.data,self.index,protocol,protocol["execution_profile"])
            folder=root/"calibration"/job["architecture"]/job["id"]
            receipt=study.old.read(folder/"receipt.json")
            self.assertEqual(receipt["status"],"failed")
            self.assertEqual(receipt["physical_optimizer_updates"],1)
            self.assertEqual(receipt["completed_microbatch_episode_exposures"],6)
            self.assertEqual(receipt["retained_updates"],1)
            self.assertEqual(set(receipt["checkpoints"]),{"checkpoint-000000.pt"})
            events=[json.loads(s) for s in (folder/"steps.jsonl").read_text().splitlines()]
            self.assertEqual([e["event"] for e in events],["started"])


if __name__=="__main__":
    unittest.main()
