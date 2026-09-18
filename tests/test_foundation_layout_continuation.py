"""Five fresh tiny CPU updates; existing fixture parents only; explicit receipt required."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_layout_continuation as bridge
from experiments import foundation_layout_training as training
from experiments.sequence_student import SequenceConfig

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'runs/foundation-layout-curriculum-validation-local/attempt-001/fixtures.json'
FIXTURE_SHA256 = '9cb18c2032ca5492347d482432daab77877450f11ec6f667fcfe14b214fe5c0c'
WORK = dict(model_attempts=0, models=0, forward_attempts=0, forwards=0, forward_episodes=0,
    backward_attempts=0, backwards=0, optimizer_attempts=0, optimizer_returns=0,
    canonical_regeneration_attempts=0, canonical_regenerations=0, canonical_rows=0,
    distinct_new_parent_draws=0, archive_load_attempts=0, archive_load_completions=0)
REPORTS = []


def image(payload):
    stream = io.BytesIO(); torch.save(payload, stream); raw = stream.getvalue()
    return raw, hashlib.sha256(raw).hexdigest()


def identity(payload):
    value = {key:deepcopy(payload[key]) for key in ('launch_sha256','arm','step','weights_sha256','execution_profile')}
    value.update(producing_schema=payload['schema'],recipe=deepcopy(payload['learner']['recipe']), evidence=deepcopy(payload['learner']['evidence']))
    return value


def arithmetic(payload):
    return {key:payload[key] for key in ('schema','recipe','weights','optimizer','cursor','evidence')}


class LayoutContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = FIXTURE.read_bytes()
        if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
            raise ValueError('preserved fixture differs')
        pairs = {tuple(row['key']):row['pair'] for row in json.loads(raw)['layout_pairs']}
        cls.bundle = dict(schema=training.BUNDLE_SCHEMA, bundle_id=0, layout='varied',
            families={family:deepcopy(pairs[(family,0,8,'varied',1)]) for family in training.FAMILIES})
        cls.config = SequenceConfig(width=8,layers=1,heads=2,feedforward=16,
            max_positions=1024,max_turns=12,max_input_bytes=128,max_output_bytes=32)
        originals = (training.build_sequence_student, torch.Tensor.backward, torch.optim.AdamW.step,
                     training.curriculum.foundation.generate_pair)
        def model(*args, **kwargs):
            WORK['model_attempts'] += 1
            value = originals[0](*args, **kwargs); WORK['models'] += 1
            def before(module, args, kwargs):
                WORK['forward_attempts'] += 1
                WORK['forward_episodes'] += kwargs['token_ids'].shape[0]
            def after(module, args, output):
                WORK['forwards'] += 1
            value.register_forward_pre_hook(before, with_kwargs=True)
            value.register_forward_hook(after)
            return value
        def backward(value, *args, **kwargs):
            WORK['backward_attempts'] += 1
            result = originals[1](value,*args,**kwargs); WORK['backwards'] += 1
            return result
        def optimizer(value, *args, **kwargs):
            WORK['optimizer_attempts'] += 1
            result = originals[2](value,*args,**kwargs); WORK['optimizer_returns'] += 1
            return result
        def generate(*args, **kwargs):
            WORK['canonical_regeneration_attempts'] += 1
            value = originals[3](*args,**kwargs)
            WORK['canonical_regenerations'] += 1; WORK['canonical_rows'] += len(value)
            return value
        for target, name, replacement in ((training,'build_sequence_student',model),
            (torch.Tensor,'backward',backward),(torch.optim.AdamW,'step',optimizer),
            (training.curriculum.foundation,'generate_pair',generate)):
            context = patch.object(target,name,replacement); context.start(); cls.addClassCleanup(context.stop)

    def equal(self,left,right):
        self.assertTrue(bridge._same(left,right),'exact typed tensor/state tree differs')

    def bundle_at(self, cursor):
        value = deepcopy(self.bundle); value['bundle_id'] = cursor
        return value

    def load(self, payload, expected, *, continuation=False, pin=None):
        raw, actual_pin = image(payload)
        method = bridge.LayoutContinuation.from_snapshot if continuation else bridge.LayoutContinuation.from_pilot
        try:
            result = method(raw, expected_sha256=actual_pin if pin is None else pin,
                            expected_identity=expected, device='cpu')
            report = result.last_restore_report
        except BaseException as error:
            report = error.continuation_report
            raise
        finally:
            REPORTS.append(deepcopy(report))
            WORK['archive_load_attempts'] += report['archive_load_attempts']
            WORK['archive_load_completions'] += report['archive_load_completions']
        return result

    def test_01_exact_uninterrupted_pilot_bridge_and_second_restart(self):
        original = training.FoundationLayoutTrainer(seed=919260001,config=self.config,learning_rate=.003,
            micro_batch_size=2,layout='varied',objective_id=training.OBJECTIVE_ID,device='cpu')
        references, reports = {}, {}
        for cursor in range(3):
            reports[cursor+1] = original.step(self.bundle_at(cursor))
            references[cursor+1] = original.snapshot()
            if cursor == 0:
                archive = dict(schema=bridge.PILOT_SCHEMA,arm='varied',step=1,launch_sha256='1'*64,
                    learner=deepcopy(references[1]),weights_sha256=bridge.checkpoint_digest(original.model),
                    execution_profile=dict(fixture='fresh_tiny_cpu_no_historical_checkpoint',cpu_threads=1))
        expected = identity(archive)
        restored = self.load(archive,expected)
        self.equal(restored._trainer.accounting,references[1]['accounting'])
        self.assertFalse(restored.recipe['resume_supported'])
        self.assertEqual(restored.last_restore_report['model_constructions'],1)
        first = restored.step(self.bundle_at(1))
        resumed_snapshot = restored.snapshot()
        self.equal(arithmetic(resumed_snapshot['learner']),arithmetic(references[2]))
        self.equal(resumed_snapshot['learner']['accounting']['work'],references[2]['accounting']['work'])
        for name in ('loss','action_loss','reply_loss','observation_language_loss'):
            self.assertEqual(first['trainer_report'][name],reports[2][name])
        again = self.load(resumed_snapshot,expected,continuation=True)
        self.equal(again._trainer.accounting,resumed_snapshot['learner']['accounting'])
        second = again.step(self.bundle_at(2))
        final = again.snapshot()
        self.equal(arithmetic(final['learner']),arithmetic(references[3]))
        self.equal(final['learner']['accounting']['work'],references[3]['accounting']['work'])
        self.equal(final['learner']['accounting']['validation_work'],references[3]['accounting']['validation_work'])
        for name in ('loss','action_loss','reply_loss','observation_language_loss'):
            self.assertEqual(second['trainer_report'][name],reports[3][name])
        self.assertEqual(final['lifetime_updates'],3)
        self.assertEqual(final['continuation_updates'],2)
        self.assertEqual(final['bridge_cost']['restore_invocations'],2)
        self.assertEqual(final['bridge_cost']['step_invocations'],2)
        self.assertEqual(final['bridge_cost']['snapshot_invocations'],2)
        self.assertEqual(again.cursor,3)
        self.assertFalse(again.failed)
        self.assertEqual(self.bundle['bundle_id'],0)
        self.__class__.archive, self.__class__.expected, self.__class__.continued = archive, expected, again

    def test_02_pin_identity_tied_weights_and_moments_fail_closed(self):
        before = WORK['models']
        with self.assertRaisesRegex(ValueError,'before decoding'):
            self.load(self.archive,self.expected,pin='0'*64)
        self.assertEqual(REPORTS[-1]['archive_load_attempts'],0)
        bad_identity = deepcopy(self.expected); bad_identity['launch_sha256'] = '2'*64
        with self.assertRaisesRegex(ValueError,'caller-authenticated identity'):
            self.load(self.archive,bad_identity)
        bad_identity = deepcopy(self.expected); bad_identity['recipe']['layout'] = 'original'
        with self.assertRaises(ValueError):
            self.load(self.archive,bad_identity)
        self.assertEqual(WORK['models'],before)
        changed = deepcopy(self.archive)
        changed['learner']['weights']['observation_head.weight'][0,0] += 1
        with self.assertRaisesRegex(ValueError,'tied parameter aliases'):
            self.load(changed,self.expected)
        changed = deepcopy(self.archive)
        next(iter(changed['learner']['optimizer']['state'].values()))['step'].add_(1)
        with self.assertRaisesRegex(ValueError,'global lifetime cursor'):
            self.load(changed,self.expected)
        self.assertEqual(WORK['models']-before,2)

    def test_03_malformed_global_id_poison_refuses_retry_without_learning(self):
        value = self.continued
        before = deepcopy(WORK)
        original_state = bridge._cpu_copy(dict(weights=value.model.state_dict(),optimizer=value.optimizer.state_dict()))
        with self.assertRaises(ValueError) as raised:
            value.step(self.bundle_at(4))
        self.assertEqual(raised.exception.continuation_report,value.last_report)
        self.assertTrue(value.failed)
        self.assertEqual(value.last_report['physical_optimizer_updates'],0)
        self.assertEqual(value.last_report['retained_updates'],0)
        self.assertEqual(value.cursor,3)
        with self.assertRaisesRegex(RuntimeError,'poisoned'):
            value.step(self.bundle_at(3))
        with self.assertRaisesRegex(RuntimeError,'poisoned'):
            value.snapshot()
        self.equal(original_state,bridge._cpu_copy(dict(weights=value.model.state_dict(),optimizer=value.optimizer.state_dict())))
        self.assertEqual(WORK,before)


if __name__ == '__main__':
    unittest.main()
