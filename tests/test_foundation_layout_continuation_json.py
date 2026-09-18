"""Specific durable-JSON regression: three CPU updates, three templates, no new parents."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

from experiments import foundation_layout_continuation as bridge
from experiments import foundation_layout_training as training

_spec = importlib.util.spec_from_file_location('preserved_continuation_fixture',
    Path(__file__).with_name('test_foundation_layout_continuation.py'))
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)
WORK, REPORTS = fixture.WORK, fixture.REPORTS


class LayoutContinuationJSONTests(unittest.TestCase):
    # Reuse the already-preserved fixture, instrumentation and helpers only.
    # The earlier five-update test methods are not inherited or executed.
    setUpClass = classmethod(fixture.LayoutContinuationTests.setUpClass.__func__)
    equal = fixture.LayoutContinuationTests.equal
    bundle_at = fixture.LayoutContinuationTests.bundle_at
    load = fixture.LayoutContinuationTests.load

    def test_01_pilot_and_restart_accept_durable_json_identity_without_state_conversion(self):
        original = training.FoundationLayoutTrainer(seed=919260001,config=self.config,learning_rate=.003,
            micro_batch_size=2,layout='varied',objective_id=training.OBJECTIVE_ID,device='cpu')
        original.step(self.bundle_at(0))
        first = original.snapshot()
        archive = dict(schema=bridge.PILOT_SCHEMA,arm='varied',step=1,launch_sha256='1'*64,learner=first,
            weights_sha256=bridge.checkpoint_digest(original.model),
            execution_profile=dict(fixture='fresh_tiny_cpu_json_regression',cpu_threads=1))
        expected = json.loads(json.dumps(fixture.identity(archive)))
        self.assertIs(type(first['recipe']['optimizer']['param_groups'][0]['betas']),tuple)
        self.assertIs(type(expected['recipe']['optimizer']['param_groups'][0]['betas']),list)
        reference_loss = original.step(self.bundle_at(1))
        reference = original.snapshot()
        restored = self.load(archive,expected)
        self.equal(restored._trainer.accounting,first['accounting'])
        actual_loss = restored.step(self.bundle_at(1))
        saved = restored.snapshot()
        self.equal(fixture.arithmetic(saved['learner']),fixture.arithmetic(reference))
        self.assertIs(type(saved['learner']['recipe']['optimizer']['param_groups'][0]['betas']),tuple)
        self.assertIs(type(saved['learner']['optimizer']['param_groups'][0]['betas']),tuple)
        for name in ('loss','action_loss','reply_loss','observation_language_loss'):
            self.assertEqual(actual_loss['trainer_report'][name],reference_loss[name])
        durable_origin = json.loads(json.dumps(restored.bridge_identity['origin']))
        again = self.load(saved,durable_origin,continuation=True)
        self.equal(bridge._cpu_copy(again.model.state_dict()),saved['learner']['weights'])
        self.equal(bridge._cpu_copy(again.optimizer.state_dict()),saved['learner']['optimizer'])
        self.equal(again._trainer.accounting,saved['learner']['accounting'])
        self.assertEqual(again.cursor,2)
        self.assertFalse(again.recipe['resume_supported'])
        self.__class__.archive,self.__class__.expected = archive,expected

    def test_02_json_array_normalization_does_not_accept_changed_recipe_or_relax_tensor_state(self):
        before = dict(WORK)
        changed = deepcopy(self.expected)
        changed['recipe']['optimizer']['param_groups'][0]['betas'][0] = .8
        with self.assertRaisesRegex(ValueError,'caller-authenticated identity'):
            self.load(self.archive,changed)
        changed = deepcopy(self.expected)
        changed['recipe']['source_sha256']['experiments/foundation_layout_training.py'] = '0'*64
        with self.assertRaisesRegex(ValueError,'recipe/source/objective/runtime'):
            self.load(self.archive,changed)
        self.assertEqual(WORK['models'],before['models'])
        self.assertEqual(WORK['optimizer_attempts'],before['optimizer_attempts'])
        # Only external JSON identity comparisons accept arrays. The exact
        # optimizer/state comparator still distinguishes tuple from list.
        self.assertFalse(bridge._same((.9,.999),[.9,.999]))
        self.assertFalse(bridge._same(True,1))


if __name__ == '__main__':
    unittest.main()
