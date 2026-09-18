"""Pure runner admission tests; stop before any data or checkpoint load."""
from copy import deepcopy
from pathlib import Path
from types import ModuleType
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments import definition_basis_learning as runner

WORK=dict(tests=0,freeze_calls=0,run_calls=0,mocked_data_admissions=0,
    checkpoint_loads=0,torch_imports=0,models=0,generation=0,updates=0,teacher_calls=0)


class StopBeforeData(Exception):
    pass


class DefinitionBasisLearningTests(unittest.TestCase):
    def setUp(self):
        WORK['tests']+=1
        folder=runner.ROOT/'runs/definition-basis-learning-validation-local'
        folder.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=folder);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_freeze_rejects_nonprotocol_allowance_before_files_or_data(self):
        for seconds in (120,899,901,1800,900.,True):
            with self.subTest(seconds=seconds):
                WORK['freeze_calls']+=1
                with self.assertRaisesRegex(ValueError,'900'):
                    runner.freeze(self.root/'out',campaign_directory=self.root/'campaign',
                        campaign_summary_sha256='a'*64,data_directory=self.root/'data',
                        data_manifest_sha256='b'*64,max_seconds=seconds)
                self.assertFalse((self.root/'out').exists())

    def test_run_rejects_wrong_pin_or_changed_contract_before_execution(self):
        launch=dict(schema=runner.SCHEMA,updates=648,micro_batch_size=32,max_seconds=900,
            endpoints=[0,216,432,648],teacher_calls=0,automatic_retry=False,automatic_promotion=False)
        with patch.object(runner,'digest',return_value='a'*64),patch.object(runner,'read') as read:
            WORK['run_calls']+=1
            with self.assertRaisesRegex(ValueError,'pinned'):runner.run(self.root,launch_sha256='b'*64)
            read.assert_not_called()
        bad=dict(schema='foreign',updates=216,micro_batch_size=16,max_seconds=1800,
            endpoints=[0,648],teacher_calls=1,automatic_retry=True,automatic_promotion=True)
        for key,value in bad.items():
            candidate=deepcopy(launch);candidate[key]=value
            with self.subTest(key=key),patch.object(runner,'digest',return_value='a'*64),patch.object(runner,'read',return_value=candidate):
                WORK['run_calls']+=1
                with self.assertRaisesRegex(ValueError,'fixed complete'):runner.run(self.root,launch_sha256='a'*64)
                self.assertFalse((self.root/'execution').exists())

    def test_terminal_campaign_gate_before_checkpoint_or_dataset_work(self):
        base=dict(status='completed',partial_work_unknown=False,launch_sha256='a'*64,
            worker_processes=[dict(exit_code=0)],last_state=dict(path='synthetic-state.json',sha256='a'*64),
            parent=dict(synthetic='parent'),physical_totals=dict(updates=432))
        previous=dict(source_sha256={},input_sha256={})
        state=dict(parent=base['parent'],physical_totals=base['physical_totals'])
        def stop(*args,**kwargs):
            WORK['mocked_data_admissions']+=1
            raise StopBeforeData('valid terminal boundary reached')
        data=ModuleType('experiments.definition_basis_data');data.load_manifest=Mock(side_effect=stop)
        worker=ModuleType('experiments.continuous_tutor_worker')
        banks=ModuleType('experiments.shared_acquisition_transfer_data')
        modules={m.__name__:m for m in (data,worker,banks)}
        cases=[dict(status='running'),dict(status='failed'),dict(partial_work_unknown=True),
            dict(worker_processes=[dict(exit_code=None)]),dict(worker_processes=[dict(exit_code=1)]),{}]
        for index,change in enumerate(cases):
            summary={**deepcopy(base),**change}
            def read(path):
                return deepcopy(summary if Path(path).name=='summary.json' else previous if Path(path).name=='launch.json' else state)
            with self.subTest(index=index),patch.dict(sys.modules,modules),patch.object(runner,'digest',return_value='a'*64),\
                    patch.object(runner,'read',side_effect=read),patch.object(runner,'verify_pins'):
                WORK['freeze_calls']+=1
                expected=StopBeforeData if not change else ValueError
                with self.assertRaises(expected):
                    runner.freeze(self.root/f'out-{index}',campaign_directory=self.root/'campaign',
                        campaign_summary_sha256='a'*64,data_directory=self.root/'data',
                        data_manifest_sha256='b'*64,max_seconds=900)
        self.assertEqual(data.load_manifest.call_count,1)


if __name__=='__main__':unittest.main()
