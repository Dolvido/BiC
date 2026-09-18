"""Temporary JSON/opaque bytes and fake owner only; no learner or service."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments import home_learning as h


class HomeTests(unittest.TestCase):
    calls = dict(mock_freezes=0, mock_runs=0, mock_inventories=0)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        original = patch.object(h, 'ROOT', self.root); original.start(); self.addCleanup(original.stop)
        self.profile = self.root/'home/session.json'
        self.source = self.root/'frozen.py'; self.source.write_text('immutable fixture\n', encoding='utf-8')
        self.old_checkpoint = self.root/'parent.bin'; self.old_checkpoint.write_bytes(b'opaque full state 7816')
        self.checkpoint = self.root/'retained.bin'; self.checkpoint.write_bytes(b'opaque full state 9112')
        self.evaluation = h._publish(self.root/'banks/manifest.json', {'fixture':'no real lessons'})
        self.teacher = dict(model='local-fixture', sha256='a'*64)
        self.parent = dict(checkpoint=h._record(self.checkpoint), weights_sha256='b'*64, lifetime_updates=9112, metrics={})
        self.state = dict(schema=h.OWNER_SCHEMA, stage='compiled', cycle=8, completed_cycles=7,
            parent=deepcopy(self.parent), original_parent_checkpoint=h._record(self.old_checkpoint),
            physical_totals=dict(worker_updates=3024, teacher_chat_attempts=8, teacher_chat_completions=8),
            teacher_decision=dict(path='existing-response.json', sha256='c'*64), compiled={'already':'complete'})
        initial = deepcopy(self.state); initial['parent'].update(checkpoint=h._record(self.old_checkpoint), lifetime_updates=7816)
        self.launch = dict(schema=h.OWNER_SCHEMA, initial_state=initial, seed=123,
            teacher=self.teacher, evaluation=dict(directory='banks',manifest_sha256=self.evaluation['sha256']),
            source_sha256={'frozen.py':h._sha(self.source)}, budget_seconds=3008, resumed=False)
        self.campaign = self.root/'completed'; self.summary = self.publish_campaign(self.campaign,self.launch,self.state)

    def publish_campaign(self, directory, launch, state, *, status='completed', unknown=False):
        pin = h._publish(directory/'launch.json',launch)
        saved = h._publish(directory/'execution/states/0000.json',state)
        summary = dict(schema=h.OWNER_SCHEMA,status=status,partial_work_unknown=unknown,
            launch_sha256=pin['sha256'],last_state=saved,parent=state['parent'],completed_cycles=state['completed_cycles'],
            physical_totals=state['physical_totals'],final_stage=state['stage'])
        return h._publish(directory/'execution/summary.json',summary)

    def initialize(self):
        return h.init_profile(self.profile,campaign_directory=self.campaign,summary_sha256=self.summary['sha256'])

    def inventory(self):
        type(self).calls['mock_inventories']+=1
        return [dict(ProcessId=os.getpid(),ParentProcessId=987654,Name='python.exe',Created='real-fixture-process',
                     ExecutablePath=str(self.root/'python.exe'),CommandLine='home_learning resume'),
                dict(ProcessId=987654,ParentProcessId=0,Name='python.exe',Created='venv-parent',
                     ExecutablePath=str(self.root/'.venv/python.exe'),CommandLine='venv bootstrap')]

    def controller(self, *, fail=False, unknown=False, foreign=False):
        def freeze(output, **kwargs):
            type(self).calls['mock_freezes']+=1
            self.assertIs(kwargs['resume'],True)
            self.assertEqual(kwargs['parent_summary_sha256'],self.summary['sha256'])
            launch=deepcopy(self.launch); launch.update(resumed=True,initial_state=deepcopy(self.state),budget_seconds=kwargs['budget_seconds'])
            if foreign:launch['initial_state']['parent']['weights_sha256']='f'*64
            return h._publish(output/'launch.json',launch)['sha256']
        def run(output, *, launch_sha256):
            type(self).calls['mock_runs']+=1
            if fail:raise RuntimeError('injected unknown worker failure')
            state=deepcopy(self.state)
            if not unknown:
                state.update(stage='ready',cycle=9,completed_cycles=8)
                new_checkpoint=self.root/'next.bin';new_checkpoint.write_bytes(b'opaque next state')
                state['parent'].update(checkpoint=h._record(new_checkpoint),lifetime_updates=9328,weights_sha256='d'*64)
                state['physical_totals']['worker_updates']+=432
            saved=h._publish(output/'execution/states/0000.json',state)
            summary=dict(schema=h.OWNER_SCHEMA,status='failed' if unknown else 'completed',partial_work_unknown=unknown,
                launch_sha256=launch_sha256,last_state=saved,parent=state['parent'],completed_cycles=state['completed_cycles'],
                physical_totals=state['physical_totals'],final_stage=state['stage'])
            h._publish(output/'execution/summary.json',summary)
            return summary
        return SimpleNamespace(freeze=Mock(side_effect=freeze),run=Mock(side_effect=run))

    def test_init_status_are_read_only_and_keep_retained_identity(self):
        result=self.initialize();before=self.profile.read_bytes()
        (self.root/'unadopted-newest.bin').write_bytes(b'not a parent')
        with (patch.object(h.importlib,'import_module',side_effect=AssertionError('no owner import')),
              patch.object(h.subprocess,'run',side_effect=AssertionError('no process/service calls'))):
            self.assertEqual(h.status(self.profile),result)
            stream=io.StringIO()
            with patch.object(h.sys,'stdout',stream):self.assertEqual(h.main(['status','--profile',str(self.profile),'--json']),0)
        self.assertEqual(self.profile.read_bytes(),before)
        self.assertEqual(result['retained_research']['lifetime_updates'],9112)
        self.assertEqual(result['retained_lineage_updates'],1296)
        self.assertEqual(result['committed_campaign_work']['worker_updates'],3024)
        self.assertEqual(result['retained_research']['pending_stage'],'compiled')
        self.assertEqual(result['retained_research']['checkpoint'],self.parent['checkpoint'])
        with self.assertRaises(FileExistsError):self.initialize()

    def test_changed_pins_failed_parent_and_mismatched_work_refuse(self):
        with self.assertRaises(ValueError):h.init_profile(self.profile,campaign_directory=self.campaign,summary_sha256='0'*64)
        self.initialize();self.source.write_text('changed',encoding='utf-8')
        with self.assertRaises(ValueError):h.status(self.profile)
        self.source.write_text('immutable fixture\n',encoding='utf-8')
        self.checkpoint.write_bytes(b'changed checkpoint')
        with self.assertRaises(ValueError):h.status(self.profile)
        self.checkpoint.write_bytes(b'opaque full state 9112')
        original=h._read(self.summary)
        for label in ('failed','unknown','work','parent'):
            summary=deepcopy(original)
            if label=='failed':summary['status']='failed'
            elif label=='unknown':summary['partial_work_unknown']=True
            elif label=='work':summary['physical_totals']['worker_updates']+=1
            else:summary['parent']['weights_sha256']='e'*64
            changed=h._publish(self.campaign/'execution/summary.json',summary,replace=True)
            with self.assertRaises(ValueError):h._campaign(self.campaign,changed['sha256'])
            h._publish(self.campaign/'execution/summary.json',original,replace=True)

    def test_resume_reuses_compiled_stage_and_advances_only_completed_owner(self):
        self.initialize();controller=self.controller()
        result=h.resume(self.profile,training_hours=1,process_inventory=self.inventory,controller=controller)
        self.assertEqual(controller.freeze.call_count,1);self.assertEqual(controller.run.call_count,1)
        frozen=controller.freeze.call_args.kwargs
        self.assertEqual(frozen['budget_seconds'],3600)
        self.assertEqual(result['retained_research']['lifetime_updates'],9328)
        self.assertIsNone(result['pending_invocation']);self.assertIsNone(result['session_lock'])
        receipt=h._read(result['last_invocation']);self.assertEqual(receipt['status'],'completed')
        launch=h._read(receipt['launch'])
        self.assertEqual(launch['initial_state']['teacher_decision'],self.state['teacher_decision'])
        self.assertEqual(launch['initial_state']['compiled'],self.state['compiled'])
        self.assertGreaterEqual(receipt['setup_wall_seconds'],receipt['freeze_wall_seconds'])

    def test_failed_or_unknown_run_stays_pending_and_is_never_retried(self):
        for kind in ('raised','unknown'):
            self.profile=self.root/f'{kind}/session.json';self.initialize()
            controller=self.controller(fail=kind=='raised',unknown=kind=='unknown')
            with self.assertRaises((RuntimeError,ValueError)):
                h.resume(self.profile,training_hours=1,process_inventory=self.inventory,controller=controller)
            result=h.status(self.profile)
            self.assertEqual(result['status'],'attention_required');self.assertEqual(result['retained_research']['lifetime_updates'],9112)
            receipt=h._read(result['last_invocation']);self.assertEqual(receipt['status'],'failed')
            with self.assertRaises(RuntimeError):h.resume(self.profile,training_hours=1,process_inventory=self.inventory,controller=controller)
            self.assertEqual(controller.freeze.call_count,1);self.assertEqual(controller.run.call_count,1)

    def test_foreign_freeze_never_launches_worker(self):
        self.initialize();controller=self.controller(foreign=True)
        with self.assertRaises(ValueError):h.resume(self.profile,training_hours=1,process_inventory=self.inventory,controller=controller)
        self.assertEqual(controller.run.call_count,0)
        self.assertEqual(h.status(self.profile)['retained_research']['lifetime_updates'],9112)

    def test_process_inventory_allows_own_bootstrap_but_refuses_other_or_unknown(self):
        self.assertIn(987654,h._idle(self.inventory)['observed_ancestor_pids'])
        records=self.inventory()
        worker=dict(ProcessId=987655,ParentProcessId=987654,Name='python.exe',Created='other',
                    ExecutablePath=str(self.root/'.venv/python.exe'),CommandLine='-m experiments.worker')
        with self.assertRaises(RuntimeError):h._idle(lambda:records+[worker])
        worker.update(ExecutablePath=None,CommandLine=None)
        with self.assertRaises(RuntimeError):h._idle(lambda:records+[worker])
        for invalid in ([],None,[{}],records+[records[0]]):
            with self.assertRaises(RuntimeError):h._idle(lambda:invalid)
        with self.assertRaises(OSError):h._idle(Mock(side_effect=OSError('inventory unavailable')))
        self.initialize();controller=self.controller()
        with self.assertRaises(RuntimeError):h.resume(self.profile,training_hours=1,process_inventory=lambda:records+[worker],controller=controller)
        self.assertEqual(controller.freeze.call_count,0);self.assertIsNone(h.status(self.profile)['pending_invocation'])

    def test_existing_or_parallel_lock_is_preserved(self):
        self.initialize();lock=self.profile.with_suffix('.lock')
        h._publish(lock,dict(pid=123,process_created='unknown',lock_id='stale'))
        before=lock.read_bytes()
        with self.assertRaises(FileExistsError):h.resume(self.profile,training_hours=1,process_inventory=self.inventory,controller=self.controller())
        self.assertEqual(lock.read_bytes(),before)
        self.assertEqual(h.status(self.profile)['status'],'attention_required')
        lock.unlink()
        with h._locks(self.profile,h._idle(self.inventory)):
            project=self.root/'runs/home-learning-local/project.lock';before=project.read_bytes()
            with self.assertRaises(FileExistsError):h.resume(self.profile,training_hours=1,process_inventory=self.inventory,controller=self.controller())
            self.assertEqual(project.read_bytes(),before)

    def test_finite_training_budget_boundaries(self):
        self.assertEqual(h.training_seconds(120/3600),120)
        self.assertEqual(h.training_seconds(1),3600);self.assertEqual(h.training_seconds(24),86400)
        for value in (None,True,0,-1,float('nan'),float('inf'),1e308,24.01,119/3600):
            with self.assertRaises(ValueError):h.training_seconds(value)


if __name__=='__main__':unittest.main()
