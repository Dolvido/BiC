"""Tiny CPU engineering arithmetic only; never formal study inputs or scores."""
import copy
from contextlib import ExitStack
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_training as legacy
from experiments import foundation_variant_training as variant
from experiments import foundation_plan_index as indexing
from experiments.foundation_admission import repair_plan
from experiments.foundation_curriculum import generate_pair
from experiments.foundation_plan import build_plan, materialize_pair
from experiments.foundation_plan_index import AuthenticatedPlanIndex
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a,b)
    if type(a) is not type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a, (list,tuple)): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def arithmetic(payload):
    return {key:payload[key] for key in ("weights","optimizer","cursor","evidence")}


class FoundationVariantTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.config=SequenceConfig(width=8,layers=4,heads=2,feedforward=16,max_turns=12)
        base=build_plan(seed=841000001,stage_updates=10,final_updates=6,micro_batch_size=2,
                        rehearsal_every=2,ordering_seed=841000002)
        cls.history=sorted(transcript_digest(row) for row in materialize_pair(base,0,"color",0))
        cls.plan,cls.admission=repair_plan(base,cls.history)
        external=generate_pair("count",842000001,depth=0,turns=8,split="dev")
        cls.protection=sorted(set(cls.history)|{transcript_digest(row) for row in external})
        cls.index=AuthenticatedPlanIndex(cls.plan,admission_protected_transcripts=cls.history,
            protected_transcripts=cls.protection,admission_receipt=cls.admission)
        cls.work=dict(actual_optimizer_updates=0,drawn_episodes=0,neural_attempted_episodes=0,
                      completed_microbatch_episodes=0,unknown_optimizer_attempts=0)
        optimize=torch.optim.AdamW.step;step=legacy.FoundationTrainer.step
        def tracked_optimizer(instance,*args,**kwargs):
            result=optimize(instance,*args,**kwargs);cls.work["actual_optimizer_updates"]+=1;return result
        def tracked_step(instance):
            try:return step(instance)
            finally:
                report=instance.last_report
                if report:
                    for target,source in (("drawn_episodes","drawn_episode_exposures"),
                        ("neural_attempted_episodes","neural_attempted_episode_exposures"),
                        ("completed_microbatch_episodes","completed_microbatch_episode_exposures")):
                        cls.work[target]+=report[source] or 0
                    cls.work["unknown_optimizer_attempts"]+=int(report["physical_optimizer_updates"] is None)
        cls.patches=ExitStack()
        cls.patches.enter_context(patch.object(torch.optim.AdamW,"step",tracked_optimizer))
        cls.patches.enter_context(patch.object(legacy.FoundationTrainer,"step",tracked_step))

    @classmethod
    def tearDownClass(cls):
        cls.patches.close();torch.set_num_threads(cls.threads)
        print("Variant foundation test-only CPU work:",cls.work,"No GPU, formal data or capability scoring.")

    def trainer(self,architecture="flat",**kwargs):
        options=dict(architecture=architecture,seed=8401,config=self.config,
            admission_protected_transcripts=self.history,protected_transcripts=self.protection,
            admission_receipt=self.admission,plan_index=self.index)
        options.update(kwargs)
        return variant.VariantFoundationTrainer(self.plan,**options)

    def test_new_flat_arithmetic_exactly_matches_frozen_trainer_in_both_orders(self):
        for order in ("curriculum","mixed"):
            with self.subTest(order=order):
                old=legacy.FoundationTrainer(self.plan,order,seed=8401,config=self.config,protected_transcripts=self.protection)
                new=self.trainer(order=order)
                self.assertTrue(same(arithmetic(old.snapshot()),arithmetic(new.snapshot())))
                a,b=old.step(),new.step()
                self.assertTrue(same(arithmetic(old.snapshot()),arithmetic(new.snapshot())))
                for key in ("bundle_id","cursor","microbatches","loss","action_loss","reply_loss","observation_language_loss"):
                    self.assertTrue(same(a[key],b[key]),key)
                self.assertNotEqual(old.snapshot()["schema"],new.snapshot()["schema"])

    def test_both_variant_full_optimizer_disk_resume_and_next_update_match(self):
        for architecture in variant.ARCHITECTURES:
            with self.subTest(architecture=architecture):
                reference=self.trainer(architecture,order="mixed");reference.step()
                with tempfile.TemporaryDirectory() as directory:
                    path=Path(directory)/"snapshot.pt";torch.save(reference.snapshot(),path)
                    restored=self.trainer(architecture,order="mixed",payload=torch.load(path,weights_only=True))
                self.assertTrue(same(reference.snapshot(),restored.snapshot()))
                reference.step();restored.step()
                self.assertTrue(same(arithmetic(reference.snapshot()),arithmetic(restored.snapshot())))

    def test_index_evidence_matches_old_replay_and_restore_does_not_materialize(self):
        for order in ("curriculum","mixed"):
            for cursor in (0,3,len(self.plan["bundles"])):
                self.assertEqual(self.index.replay(order,cursor),legacy.replay_evidence(self.plan,order,cursor,self.protection))
        trainer=self.trainer();trainer.step();saved=trainer.snapshot()
        # These local test sentinels never modify any implementation or replay hook.
        with patch.object(legacy,"_materialize_validated_bundle",side_effect=AssertionError("unexpected lesson generation")), patch.object(
                indexing,"_materialize_validated_bundle",side_effect=AssertionError("unexpected index rescan")):
            trainer.restore(saved);trainer.restore(saved)
        self.assertTrue(same(saved,trainer.snapshot()))

    def test_reused_index_must_match_full_plan_original_and_expanded_protection(self):
        with patch.object(variant.VariantFoundationTrainer,"_build",side_effect=AssertionError("model constructed before admission")):
            for options in (dict(admission_protected_transcripts=[]),dict(protected_transcripts=self.history),
                    dict(admission_receipt={}),dict(plan_index={}),dict(architecture="unknown")):
                with self.subTest(options=options),self.assertRaises(ValueError):self.trainer(**options)
            changed=copy.deepcopy(self.plan);changed["config"]["ordering_seed"]+=1
            with self.assertRaises(ValueError):
                variant.VariantFoundationTrainer(changed,architecture="flat",seed=8401,config=self.config,
                    admission_protected_transcripts=self.history,protected_transcripts=self.protection,
                    admission_receipt=self.admission,plan_index=self.index)

    def test_architecture_source_index_and_counter_corruption_cannot_mutate_live_state(self):
        trainer=self.trainer("hierarchical");trainer.step();saved=trainer.snapshot()
        model,optimizer=trainer.model,trainer.optimizer
        for kind in ("architecture","source","index","cursor","count","alias","moment","step","dtype","timing"):
            changed=copy.deepcopy(saved)
            if kind=="architecture":changed["recipe"]["architecture"]["name"]="flat"
            elif kind=="source":changed["recipe"]["source_sha256"]={}
            elif kind=="index":changed["recipe"]["plan_index_identity"]["expanded_protection_count"]+=1
            elif kind=="cursor":changed["cursor"]+=1
            elif kind=="count":changed["evidence"]["exposures"]["color"]["episodes"]+=1
            elif kind=="alias":changed["weights"]["observation_head.weight"][0,0]+=1
            elif kind=="moment":changed["optimizer"]["state"][0]["exp_avg_sq"].flatten()[0]=-1
            elif kind=="step":changed["optimizer"]["state"][0]["step"]+=1
            elif kind=="dtype":changed["optimizer"]["state"][0]["exp_avg"]=changed["optimizer"]["state"][0]["exp_avg"].double()
            else:changed["timing"]["materialization_seconds_included_in_step"]=changed["timing"]["retained_step_seconds"]+1
            with self.subTest(kind=kind),self.assertRaises(ValueError):trainer.restore(changed)
            self.assertIs(trainer.model,model);self.assertIs(trainer.optimizer,optimizer)
            self.assertTrue(same(saved,trainer.snapshot()))

    def test_same_inventory_does_not_allow_cross_architecture_or_old_flat_checkpoint(self):
        flat,hierarchical=self.trainer(),self.trainer("hierarchical")
        self.assertTrue(same(flat.snapshot()["weights"],hierarchical.snapshot()["weights"]))
        with self.assertRaises(ValueError):hierarchical.restore(flat.snapshot())
        with self.assertRaises(ValueError):flat.restore(hierarchical.snapshot())
        old=legacy.FoundationTrainer(self.plan,seed=8401,config=self.config,protected_transcripts=self.protection)
        with self.assertRaises(ValueError):flat.restore(old.snapshot())
        changed=flat.snapshot();changed["weights"]["action_head.bias"][0]+=1
        with self.assertRaisesRegex(ValueError,"seeded initialization"):flat.restore(changed)

    def test_partial_step_poison_and_indexed_restoration_preserve_original_semantics(self):
        trainer=self.trainer("hierarchical");saved=trainer.snapshot()
        forward=trainer.model.forward;calls=0
        def fail_second(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError("injected second microbatch failure")
            return forward(*args,**kwargs)
        with patch.object(trainer.model,"forward",side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError,"second microbatch"):trainer.step()
        self.assertEqual(trainer.last_report["physical_optimizer_updates"],0)
        self.assertEqual(trainer.last_report["completed_microbatch_episode_exposures"],2)
        with self.assertRaises(RuntimeError):trainer.snapshot()
        trainer.restore(saved);self.assertTrue(same(saved,trainer.snapshot()));trainer.step()

    def test_source_drift_rejects_cached_replay_and_public_identity_is_isolated(self):
        trainer=self.trainer();saved=trainer.snapshot()
        identity=trainer.index_identity;identity["plan_sha256"]="0"*64
        self.assertNotEqual(identity,trainer.index_identity)
        with patch.object(variant,"source_hashes",return_value={}):
            with self.assertRaisesRegex(ValueError,"source identity"):trainer.restore(saved)
        with patch.object(indexing,"source_hashes",return_value={}):
            with self.assertRaisesRegex(RuntimeError,"index source identity"):trainer.restore(saved)
        self.assertTrue(same(saved,trainer.snapshot()))


if __name__=="__main__":unittest.main(verbosity=2)
