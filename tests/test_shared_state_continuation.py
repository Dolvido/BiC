"""Tiny exact next-update proof using one preserved training fixture triplet."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments import foundation_layout_training as training
from experiments import shared_state_continuation as bridge, shared_state_student as student
from experiments.foundation_layout_prepared import PreparedLayoutOwner, evidence_sha256
from experiments.sequence_student import SequenceConfig
from experiments.shared_state_targets import pack_state_targets
from experiments.shared_state_training import SharedStateKernel

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/"runs/foundation-layout-curriculum-validation-local/attempt-001/fixtures.json"
FIXTURE_SHA = "9cb18c2032ca5492347d482432daab77877450f11ec6f667fcfe14b214fe5c0c"
WORK = dict(model_constructions=0, optimizer_constructions=0, forwards=0, forward_episodes=0,
            backwards=0, optimizer_attempts=0, optimizer_returns=0, archive_loads=0,
            canonical_regenerations=0, distinct_parent_draws=0, state_target_calls=0,
            state_target_rows=0, successful_restores=0, rejected_restores=0, snapshots=0)
RESTORES, STEPS, OWNERS = [], [], []


def serialize(payload):
    stream = io.BytesIO(); torch.save(payload, stream)
    image = stream.getvalue()
    return image, hashlib.sha256(image).hexdigest()


class SharedStateContinuationTests(unittest.TestCase):
    def test_exact_continuation_json_identity_and_corruption_rejection(self):
        prior_threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, prior_threads)
        config = SequenceConfig(width=16, layers=1, heads=2, feedforward=32, max_turns=12)
        raw = FIXTURE.read_bytes(); self.assertEqual(hashlib.sha256(raw).hexdigest(), FIXTURE_SHA)
        pairs = {tuple(item["key"]): item["pair"] for item in json.loads(raw)["layout_pairs"]}
        bundle = dict(schema=training.BUNDLE_SCHEMA, bundle_id=0, layout="original",
                      families={f:deepcopy(pairs[(f,0,8,"original",0)]) for f in training.FAMILIES})
        original_builder, original_init = student.build_shared_state_student, torch.optim.AdamW.__init__
        original_step, original_backward = torch.optim.AdamW.step, torch.Tensor.backward
        original_load, original_generate = torch.load, training.curriculum.foundation.generate_pair
        def forward_hook(module,args,kwargs,out):
            WORK["forwards"] += 1; WORK["forward_episodes"] += len(kwargs["token_ids"])
        def builder(*args,**kwargs):
            model=original_builder(*args,**kwargs);WORK["model_constructions"]+=1
            model.register_forward_hook(forward_hook,with_kwargs=True)
            return model
        def optimizer_init(optimizer,*args,**kwargs):
            original_init(optimizer,*args,**kwargs);WORK["optimizer_constructions"]+=1
        def optimizer_step(optimizer,*args,**kwargs):
            WORK["optimizer_attempts"]+=1;result=original_step(optimizer,*args,**kwargs)
            WORK["optimizer_returns"]+=1;return result
        def backward(tensor,*args,**kwargs):
            WORK["backwards"]+=1;return original_backward(tensor,*args,**kwargs)
        def load(*args,**kwargs):
            self.assertTrue(kwargs.get("weights_only"));self.assertEqual(kwargs.get("map_location"),"cpu")
            WORK["archive_loads"]+=1;return original_load(*args,**kwargs)
        def generate(*args,**kwargs):
            WORK["canonical_regenerations"]+=1;return original_generate(*args,**kwargs)
        with patch.object(student,"build_shared_state_student",builder), \
                patch.object(torch.optim.AdamW,"__init__",optimizer_init), \
                patch.object(torch.optim.AdamW,"step",optimizer_step), \
                patch.object(torch.Tensor,"backward",backward), patch.object(torch,"load",load), \
                patch.object(training.curriculum.foundation,"generate_pair",generate):
            evidence=training._bundle_evidence(bundle,cursor=0,layout="original",micro_batch_size=2,work=training.curriculum.WorkLedger())
            targets={}
            for family,rows in bundle["families"].items():
                targets[family]=pack_state_targets(rows);WORK["state_target_calls"]+=1;WORK["state_target_rows"]+=len(rows)
            def token(cursor):
                owner=PreparedLayoutOwner(config=config,layout="original",micro_batch_size=2);OWNERS.append(owner)
                row,expected=deepcopy(bundle),deepcopy(evidence)
                row["bundle_id"]=expected["bundle_id"]=cursor
                return owner.prepare(row,expected_evidence=expected,expected_evidence_sha256=evidence_sha256(expected))
            model=student.build_shared_state_student(852610001,config=config)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
            reference=SharedStateKernel(model,optimizer,config=config,layout="original",micro_batch_size=2,
                                        objective_id=training.OBJECTIVE_ID,auxiliary_weight=.3)
            STEPS.append(reference.step(token(0),state_targets=targets))
            saved=reference.snapshot()
            payload=dict(schema=bridge.PRODUCER_SCHEMA,launch_sha256="a"*64,arm="fast",step=1,
                architecture=student.ARCHITECTURE,learning_rate=.003,auxiliary_weight=.3,
                runtime=bridge.current_runtime(),learner=saved,weights_sha256=checkpoint_digest(model))
            identity=json.loads(json.dumps(bridge.checkpoint_identity(payload)))
            self.assertIsInstance(identity["recipe"]["optimizer"][0]["betas"],list)
            image,pin=serialize(payload)
            def restore(data,digest,expected=identity,*,own=False,reject=False):
                method=bridge.SharedStateContinuation.from_snapshot if own else bridge.SharedStateContinuation.from_checkpoint
                try:
                    obj=method(data,expected_sha256=digest,expected_identity=expected,device="cpu")
                except (ValueError,RuntimeError) as error:
                    RESTORES.append(deepcopy(error.continuation_report));WORK["rejected_restores"]+=1
                    if not reject:raise
                    return
                if reject:self.fail("corrupt restore was accepted")
                RESTORES.append(deepcopy(obj.last_restore_report));WORK["successful_restores"]+=1
                return obj
            restored=restore(image,pin)
            self.assertTrue(bridge._same(saved,restored._kernel.snapshot()))
            self.assertTrue(restored.model.tokens.weight is restored.model.observation_head.weight)
            self.assertIsNone(restored.last_report)
            expected=reference.step(token(1),state_targets=targets);STEPS.append(expected)
            actual=restored.step(token(1),state_targets=targets);STEPS.append(actual)
            for field in ("weights","optimizer","recipe","evidence","cursor"):
                self.assertTrue(bridge._same(reference.snapshot()[field],restored._kernel.snapshot()[field]),field)
            for field in ("loss","action_loss","reply_loss","observation_language_loss","state_loss","total_loss",
                          "bundle_evidence","cursor","completed_state_readouts","completed_state_objectives"):
                self.assertTrue(bridge._same(expected[field],actual[field]),field)
            self.assertTrue(bridge._same(reference.accounting["work"],restored.accounting["lifetime_kernel"]["work"]))
            self.assertTrue(bridge._same(reference.accounting["state_work"],restored.accounting["lifetime_kernel"]["state_work"]))
            self.assertTrue(bridge._same(actual,restored.last_report))
            outer=restored.snapshot();WORK["snapshots"]+=1
            snap,snap_pin=serialize(outer);reloaded=restore(snap,snap_pin,own=True)
            self.assertTrue(bridge._same(outer["learner"],reloaded._kernel.snapshot()))
            self.assertEqual(reloaded.accounting["bridge_cost"]["restore_invocations"],2)
            self.assertEqual(reloaded.cursor,2)
            before_loads=WORK["archive_loads"]
            restore(image+b"tamper",pin,reject=True)
            wrong=deepcopy(identity);wrong["learning_rate"]=.0003
            restore(image,pin,wrong,reject=True)
            wrong=deepcopy(identity);wrong["recipe"]["source_sha256"][next(iter(wrong["recipe"]["source_sha256"]))]="0"*64
            restore(image,pin,wrong,reject=True)
            self.assertEqual(WORK["archive_loads"],before_loads)
            # Validly re-pinned synthetic corruptions exercise semantic checks.
            bad=deepcopy(payload);bad["learner"]["optimizer"]["state"][0]["exp_avg_sq"].reshape(-1)[0]=-1.
            restore(*serialize(bad),reject=True)
            bad=deepcopy(payload);bad["learner"]["optimizer"]["state"][0]["step"].fill_(0)
            restore(*serialize(bad),reject=True)
            bad=deepcopy(payload);bad["learner"]["weights"]["observation_head.weight"]=bad["learner"]["weights"]["observation_head.weight"].clone()
            bad["learner"]["weights"]["observation_head.weight"][0,0]+=1.
            restore(*serialize(bad),reject=True)
            bad=deepcopy(outer);bad["identity"]["source_sha256"]["experiments/shared_state_continuation.py"]="0"*64
            restore(*serialize(bad),own=True,reject=True)
            # A delegated failure leaves both adapter and underlying kernel poisoned.
            unchanged=deepcopy(reloaded._kernel.snapshot())
            invalid=deepcopy(targets);invalid["color"]=invalid["color"][:,:,:-1]
            with self.assertRaises(ValueError):reloaded.step(token(2),state_targets=invalid)
            self.assertTrue(reloaded.failed)
            with self.assertRaises(RuntimeError):reloaded.snapshot()
            self.assertTrue(bridge._same(unchanged["weights"],bridge._cpu_copy(reloaded.model.state_dict())))
            self.assertTrue(bridge._same(unchanged["optimizer"],bridge._cpu_copy(reloaded.optimizer.state_dict())))
            self.assertEqual(reloaded.cursor,2)
            for owner in OWNERS:owner.close()
        self.assertEqual(WORK["successful_restores"],2)
        self.assertEqual(WORK["rejected_restores"],7)
        self.assertEqual(WORK["model_constructions"],6)
        self.assertEqual(WORK["optimizer_constructions"],6)
        self.assertEqual(WORK["optimizer_returns"],3)
        self.assertEqual(WORK["forwards"],9)
        self.assertEqual(WORK["backwards"],9)
        self.assertEqual(WORK["forward_episodes"],18)
        self.assertEqual(WORK["canonical_regenerations"],3)
        print("SHARED_STATE_CONTINUATION_CPU_ACCOUNTING="+json.dumps(WORK,sort_keys=True))


if __name__=="__main__":
    unittest.main()
