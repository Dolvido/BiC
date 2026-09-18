"""Small UI checks with fake metadata and child processes only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments import offline_home as ui


class OfflineHomeTests(unittest.TestCase):
    calls = dict(mock_metadata_requests=0, mock_children=0, mock_train_menu_calls=0)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        change = patch.object(ui.h, 'ROOT', self.root)
        change.start(); self.addCleanup(change.stop)
        self.profile = self.root / 'session.json'
        self.value = dict(status='ready', teacher=dict(model='local-fixture', sha256='a'*64),
            retained_research=dict(lifetime_updates=9112, completed_cycles=7, pending_stage='compiled'))
        change = patch.object(ui.h, 'status', return_value=self.value)
        self.status = change.start(); self.addCleanup(change.stop)
        self.output = []

    def transport(self, *, digest=None, remote=False, failure=None):
        def request(route, payload, deadline):
            type(self).calls['mock_metadata_requests'] += 1
            if failure:
                raise failure
            if route == '/api/tags':
                self.assertIsNone(payload)
                return dict(models=[dict(name='local-fixture', digest=digest or 'a'*64)])
            self.assertEqual(route, '/api/show')
            self.assertEqual(payload, {'model':'local-fixture'})
            return dict(details={}, remote_host='cloud' if remote else '')
        return SimpleNamespace(LocalTutor=Mock(), _request_json=request,
            _remote_metadata=lambda v: bool(v.get('remote_host')))

    def test_status_has_no_service_or_process_work(self):
        with (patch.object(ui, '_transport', side_effect=AssertionError('no service')),
              patch.object(ui.subprocess, 'Popen', side_effect=AssertionError('no child'))):
            self.assertEqual(ui.status(self.profile, emit=self.output.append), self.value)
        self.assertIn('9,112 updates; 7 completed cycles', self.output[0])
        self.assertIn('Restricted English', self.output[-1])

    def test_check_exact_digest_and_remote_rejection(self):
        for digest, remote, valid in [('a'*64,False,True), ('b'*64,False,False), ('a'*64,True,False)]:
            with self.subTest(digest=digest, remote=remote), patch.object(ui, '_transport', return_value=self.transport(digest=digest, remote=remote)):
                if valid:
                    self.assertEqual(ui.check(self.profile, emit=self.output.append), self.value)
                else:
                    with self.assertRaises(ValueError):
                        ui.check(self.profile, emit=self.output.append)
        self.assertNotIn('remote_host', '\n'.join(self.output))

    def test_check_pending_and_unavailable_service(self):
        self.value['status'] = 'attention_required'
        with patch.object(ui, '_transport', side_effect=AssertionError('no service while pending')):
            with self.assertRaises(RuntimeError):
                ui.check(self.profile)
        self.value['status'] = 'ready'
        with patch.object(ui, '_transport', return_value=self.transport(failure=ConnectionRefusedError())):
            with self.assertRaisesRegex(RuntimeError, 'Open the installed Ollama app'):
                ui.check(self.profile)

    def test_transport_source_pin_is_checked_before_import(self):
        source = self.root/'brain_in_computer/curriculum_tutor.py'
        source.parent.mkdir(); source.write_bytes(b'fake stdlib transport')
        self.profile.write_text(json.dumps(dict(current=dict(source_sha256={
            'brain_in_computer/curriculum_tutor.py':hashlib.sha256(source.read_bytes()).hexdigest()}))))
        with patch.object(ui.importlib, 'import_module', return_value='mock module') as imported:
            self.assertEqual(ui._transport(self.profile), 'mock module')
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                ui._transport(self.profile)
            imported.assert_called_once_with('brain_in_computer.curriculum_tutor')

    def test_train_delegates_once_and_reports_stages_and_failure(self):
        python = self.root/'.venv/Scripts/python.exe'
        python.parent.mkdir(parents=True); python.write_bytes(b'not executable')
        self.profile.write_text(json.dumps(dict(pending_invocation={'directory':'new-invocation'})))
        state = self.root/'new-invocation/campaign/execution/states/0000.json'
        state.parent.mkdir(parents=True)
        state.write_text(json.dumps(dict(schema=ui.h.OWNER_SCHEMA, cycle=8, stage='compiled')))
        for code in (0, 7):
            self.output.clear()
            child = SimpleNamespace(returncode=code, poll=Mock(side_effect=[None,None,code]), pid=1234)
            def launch(command, **kwargs):
                type(self).calls['mock_children'] += 1
                self.assertEqual(command[:5], [str(python),'-B','-m','experiments.home_learning','resume'])
                self.assertEqual(command[-2:], ['--training-hours','0.35'])
                self.assertEqual(kwargs['cwd'],self.root)
                kwargs['stdout'].write(b'injected child detail\n'); kwargs['stdout'].flush()
                return child
            with (patch.object(ui, 'check', return_value=self.value),
                  patch.object(ui.subprocess, 'Popen', side_effect=launch) as spawn,
                  patch.object(ui.time, 'sleep')):
                self.assertEqual(ui.train(self.profile,hours=.35,emit=self.output.append),code)
                spawn.assert_called_once()
            self.assertEqual(sum(line.startswith('Cycle 8:') for line in self.output),1)
            if code:
                self.assertIn('injected child detail','\n'.join(self.output))
        logs = list((self.root/'runs/offline-home-local').glob('*.log'))
        self.assertEqual(len(logs),2)
        self.assertEqual([p.read_bytes() for p in logs],[b'injected child detail\n']*2)

    def test_menu_rejects_invalid_hours_then_uses_default(self):
        inputs = iter(['1','bad','1','nan','1','','q'])
        def train(*args, **kwargs):
            type(self).calls['mock_train_menu_calls'] += 1
            self.assertEqual(kwargs['hours'],1.0)
            return 0
        with patch.object(ui,'train',side_effect=train) as mocked:
            self.assertEqual(ui.menu(self.profile,emit=self.output.append,read=lambda _:next(inputs)),0)
            mocked.assert_called_once()
        self.assertEqual(sum(line.startswith('Unable to continue:') for line in self.output),2)


if __name__ == '__main__':
    unittest.main()
