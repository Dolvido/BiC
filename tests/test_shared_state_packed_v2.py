"""Four tiny models, five updates, fifteen family forward/backwards; CPU only.

One preserved three-family fixture is admitted (three reconstructions). Packed
image creation occurs once. No distinct lesson draws, GPU or tutor calls.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.learning_student import _cpu_copy
from experiments import foundation_layout_training as training
from experiments import foundation_layout_prepared as v1
from experiments import foundation_layout_prepared_v2 as v2
from experiments import shared_state_training as old_kernel, shared_state_continuation as old
from experiments import shared_state_packed_migration as migration
from experiments import shared_state_student as student, shared_state_targets as targets
from experiments.sequence_student import SequenceConfig

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/"runs/foundation-layout-curriculum-validation-local/attempt-001/fixtures.json"
FIXTURE_SHA="9cb18c2032ca5492347d482432daab77877450f11ec6f667fcfe14b214fe5c0c"
WORK=dict(model_constructions=0,optimizer_constructions=0,optimizer_updates=0,forwards=0,backwards=0,
    forward_episodes=0,archive_loads=0,canonical_regenerations=0,state_target_calls=0,state_target_episodes=0,
    cuda_attempts=0,distinct_parent_draws=0,teacher_calls=0)
OWNERS=[];REPORTS=[];IMAGE_WORK=[]


def serialize(value):
    stream=io.BytesIO();torch.save(value,stream);raw=stream.getvalue()
    return raw,hashlib.sha256(raw).hexdigest()


class PackedV2Tests(unittest.TestCase):
    def same(self,left,right):self.assertTrue(old._same(left,right))

    def test_tensor_ownership_migration_exact_updates_restart_and_rejection(self):
        torch.set_num_threads(1)
        config=SequenceConfig(width=16,layers=1,heads=2,feedforward=32,max_turns=12)
        raw=FIXTURE.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),FIXTURE_SHA)
        pairs={tuple(v["key"]):v["pair"] for v in json.loads(raw)["layout_pairs"]}
        bundle=dict(schema=training.BUNDLE_SCHEMA,bundle_id=0,layout="original",
            families={f:deepcopy(pairs[(f,0,8,"original",0)]) for f in training.FAMILIES})
        builder,init,step=student.build_shared_state_student,torch.optim.AdamW.__init__,torch.optim.AdamW.step
        backward,load,generate,packtargets=torch.Tensor.backward,torch.load,training.curriculum.foundation.generate_pair,targets.pack_state_targets
        def hooked_builder(*a,**k):
            model=builder(*a,**k);WORK["model_constructions"]+=1
            def hook(module,args,kwargs,result):
                WORK["forwards"]+=1;WORK["forward_episodes"]+=len(kwargs["token_ids"])
            model.register_forward_hook(hook,with_kwargs=True);return model
        def hooked_init(self,*a,**k):init(self,*a,**k);WORK["optimizer_constructions"]+=1
        def hooked_step(self,*a,**k):
            value=step(self,*a,**k);WORK["optimizer_updates"]+=1;return value
        def hooked_backward(self,*a,**k):
            value=backward(self,*a,**k);WORK["backwards"]+=1;return value
        def hooked_load(*a,**k):
            self.assertTrue(k.get("weights_only"));self.assertEqual(k.get("map_location"),"cpu")
            WORK["archive_loads"]+=1;return load(*a,**k)
        def hooked_generate(*a,**k):WORK["canonical_regenerations"]+=1;return generate(*a,**k)
        def hooked_targets(rows):
            WORK["state_target_calls"]+=1;WORK["state_target_episodes"]+=len(rows);return packtargets(rows)
        def no_cuda(*a,**k):WORK["cuda_attempts"]+=1;raise AssertionError("CPU proof only")
        with patch.object(student,"build_shared_state_student",hooked_builder),patch.object(torch.optim.AdamW,"__init__",hooked_init), \
             patch.object(torch.optim.AdamW,"step",hooked_step),patch.object(torch.Tensor,"backward",hooked_backward), \
             patch.object(torch,"load",hooked_load),patch.object(training.curriculum.foundation,"generate_pair",hooked_generate), \
             patch.object(targets,"pack_state_targets",hooked_targets),patch.object(torch.cuda,"_lazy_init",no_cuda):
            evidence=training._bundle_evidence(bundle,cursor=0,layout="original",micro_batch_size=2,
                                              work=training.curriculum.WorkLedger())
            labels={f:targets.pack_state_targets(rows) for f,rows in bundle["families"].items()}
            image=v2.VerifiedPackedImage.from_bundle(bundle,expected_evidence=evidence,
                expected_evidence_sha256=v1.evidence_sha256(evidence),config=config,layout="original",micro_batch_size=2)
            IMAGE_WORK.append(image.report())
            raw_image,pin_image=serialize(image.snapshot())
            packed=v2.VerifiedPackedImage.from_bytes(raw_image,expected_sha256=pin_image,
                                                    expected_identity_sha256=image.identity_sha256)
            IMAGE_WORK.append(packed.report());self.same(image.snapshot(),packed.snapshot())
            def expected(cursor):
                value=deepcopy(evidence);value["bundle_id"]=cursor;return value
            def owner(new=True,other_config=config):
                value=(v2 if new else v1).PreparedLayoutOwner(config=other_config,layout="original",micro_batch_size=2)
                OWNERS.append(value);return value
            def new_token(cursor,who=None):
                who=owner() if who is None else who;value=expected(cursor)
                return who.prepare_from_verified_image(packed,expected_evidence=value,
                    expected_evidence_sha256=v2.evidence_sha256(value))
            def old_token(cursor):
                who=owner(False);rows=deepcopy(bundle);rows["bundle_id"]=cursor;value=expected(cursor)
                return who.prepare(rows,expected_evidence=value,expected_evidence_sha256=v1.evidence_sha256(value))
            def consume(who,token,cursor):
                return who.consume(token,cursor=cursor,config=config,layout="original",micro_batch_size=2)
            initial=old_token(0)
            self.same(_cpu_copy(initial._batches),packed.snapshot()["batches"])
            self.same(labels,packed.snapshot()["state_targets"])
            a,b=owner(),owner();ta,tb=new_token(0,a),new_token(7,b)
            ba,ea,la=consume(a,ta,0);bb,eb,lb=consume(b,tb,7)
            self.same(ba,bb);self.same(la,lb);self.same(la,labels)
            self.assertEqual(ea,expected(0));self.assertEqual(eb,expected(7))
            ba["color"]["inputs"]["token_ids"].fill_(0)
            self.same(packed.snapshot()["batches"],image.snapshot()["batches"])
            with self.assertRaises(ValueError):consume(a,ta,0)
            wrong=owner();fresh=new_token(0,wrong)
            with self.assertRaises(ValueError):consume(wrong,fresh,1)
            cross_a,cross_b=owner(),owner();one,two=new_token(0,cross_a),new_token(0,cross_b)
            with self.assertRaises(ValueError):consume(cross_b,one,0)
            consume(cross_a,one,0)
            bad=owner()
            with self.assertRaises(ValueError):bad.prepare_from_verified_image(packed,expected_evidence=evidence,expected_evidence_sha256="0"*64)
            bad=owner(other_config=SequenceConfig(width=32,layers=1,heads=2,feedforward=32,max_turns=12))
            with self.assertRaises(ValueError):new_token(0,bad)
            corrupt=packed.snapshot();corrupt["batches"]["color"]["inputs"]["token_ids"][0,0]+=1
            corrupt_raw,corrupt_pin=serialize(corrupt)
            with self.assertRaises(ValueError):v2.VerifiedPackedImage.from_bytes(corrupt_raw,expected_sha256=corrupt_pin,
                expected_identity_sha256=packed.identity_sha256)
            before=WORK["archive_loads"]
            with self.assertRaises(ValueError):v2.VerifiedPackedImage.from_bytes(raw_image+b"bad",expected_sha256=pin_image,
                expected_identity_sha256=packed.identity_sha256)
            self.assertEqual(WORK["archive_loads"],before)
            model=student.build_shared_state_student(852905001,device="cpu",config=config)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
            reference=old_kernel.SharedStateKernel(model,optimizer,config=config,layout="original",micro_batch_size=2,
                objective_id=training.OBJECTIVE_ID,auxiliary_weight=.3)
            REPORTS.append(reference.step(initial,state_targets=labels))
            parent=dict(schema=old.PRODUCER_SCHEMA,launch_sha256="a"*64,arm="fast",step=1,
                architecture=student.ARCHITECTURE,learning_rate=.003,auxiliary_weight=.3,
                runtime=old.current_runtime(),learner=reference.snapshot(),weights_sha256=checkpoint_digest(model))
            identity=json.loads(json.dumps(old.checkpoint_identity(parent)))
            data,pin=serialize(parent)
            original=old.SharedStateContinuation.from_checkpoint(data,expected_sha256=pin,expected_identity=identity)
            prior=original.snapshot();old_data,old_pin=serialize(prior)
            candidate=migration.PackedSharedStateContinuation.from_continuation(old_data,expected_sha256=old_pin,
                expected_identity=identity)
            self.same(candidate.transition["old_bridge_cost"],prior["bridge_cost"])
            self.same(candidate.transition["old_recipe"],prior["learner"]["recipe"])
            self.assertNotEqual(candidate.recipe["schema"],prior["learner"]["recipe"]["schema"])
            self.assertNotEqual(candidate.recipe["source_sha256"],prior["learner"]["recipe"]["source_sha256"])
            for cursor in (1,2):
                before=reference.step(old_token(cursor),state_targets=labels)
                after=candidate.step(new_token(cursor));REPORTS.extend([before,after])
                for key in ("loss","action_loss","reply_loss","observation_language_loss","state_loss","total_loss",
                            "bundle_evidence","cursor","state_target_sha256"):
                    self.same(before[key],after[key])
                saved=candidate.snapshot();baseline=reference.snapshot()
                for key in ("weights","optimizer","cursor","evidence"):
                    self.same(saved["learner"][key],baseline[key])
                for key in ("work","state_work"):
                    self.same(saved["learner"]["accounting"][key],baseline["accounting"][key])
                if cursor==1:
                    data,pin=serialize(saved)
                    candidate=migration.PackedSharedStateContinuation.from_snapshot(data,expected_sha256=pin,
                        expected_transition_sha256=saved["transition_sha256"])
                    self.same(candidate.snapshot()["learner"],saved["learner"])
            corrupt=deepcopy(saved);corrupt["transition"]["migration_cursor"]+=1
            data,pin=serialize(corrupt);before_models=WORK["model_constructions"]
            with self.assertRaises(ValueError):migration.PackedSharedStateContinuation.from_snapshot(data,expected_sha256=pin,
                expected_transition_sha256=saved["transition_sha256"])
            self.assertEqual(WORK["model_constructions"],before_models)
            # A live token at the wrong cursor poisons the new learner before any forward.
            prior_weights=_cpu_copy(candidate.model.state_dict());before_fwd=WORK["forwards"]
            with self.assertRaises(ValueError):candidate.step(new_token(4))
            self.assertEqual(WORK["forwards"],before_fwd);self.assertTrue(candidate.failed)
            self.same(prior_weights,_cpu_copy(candidate.model.state_dict()))
            with self.assertRaises(RuntimeError):candidate.snapshot()
            for who in OWNERS:who.close()
        self.assertEqual(WORK["model_constructions"],4)
        self.assertEqual(WORK["optimizer_constructions"],4)
        self.assertEqual(WORK["optimizer_updates"],5)
        self.assertEqual((WORK["forwards"],WORK["backwards"],WORK["forward_episodes"]),(15,15,30))
        self.assertEqual(WORK["archive_loads"],7)
        self.assertEqual(WORK["canonical_regenerations"],3)
        self.assertEqual((WORK["state_target_calls"],WORK["state_target_episodes"]),(6,12))
        self.assertEqual(WORK["cuda_attempts"],0)


if __name__=="__main__":unittest.main()
