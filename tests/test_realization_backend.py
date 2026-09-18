"""CPU tests for budgeted transactional raw-English backend, not capability."""
import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from experiments.composition_curriculum import FAMILIES,generate_pair
from experiments.realization_backend import RealizationBackend,progress_vector
from experiments.composition_evaluation import prediction_metrics
from experiments.realization_training import stream_evidence
from experiments.sequence_student import SequenceConfig


class RealizationBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.banks={family:{8:generate_pair(family,271000000+i*100,turns=8)} for i,family in enumerate(FAMILIES)}
        cls.evaluation={'development':{'composed/color/t8':generate_pair('color',281000001,split='dev',turns=8)},
                        'retention':{'anchor/count/t8':generate_pair('count',281000100,split='dev',turns=8)}}
        cls.config=SequenceConfig(width=16,layers=1,heads=2,feedforward=32,max_turns=12)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def backend(self,**options):
        return RealizationBackend(self.banks,mode='fresh',protected_transcripts=[],evaluation_banks=self.evaluation,
                                  micro_batch_size=2,config=self.config,**options)
    def compare(self,a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for key in a:self.compare(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.compare(x,y)
        else:self.assertEqual(a,b)
    def compare_learning(self,a,b):
        left,right=a.snapshot(),b.snapshot()
        self.compare(left['curriculum'],right['curriculum'])
        for field in ('weights','optimizer','samplers','exposures','family_microbatches'):
            self.compare(left['learner'][field],right['learner'][field])
        self.compare(stream_evidence(left['learner']['realization']),stream_evidence(right['learner']['realization']))

    def test_exact_committed_resume_restores_schedule_optimizer_and_stream(self):
        schedule=[FAMILIES,('count','count','color'),FAMILIES]
        reference=self.backend();reference.train_chunk(schedule,max_updates=3)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'backend.pt';first=self.backend()
            report=first.train_chunk(schedule,max_updates=1,checkpoint_path=path)
            self.assertEqual(report['committed_updates'],1)
            self.assertEqual(first.curriculum['cursor'],1)
            restored=RealizationBackend.load(path,self.banks,protected_transcripts=[],evaluation_banks=self.evaluation)
            restored.train_chunk(max_updates=2)
            self.compare_learning(reference,restored)
            durable=RealizationBackend.load(path,self.banks,protected_transcripts=[],evaluation_banks=self.evaluation)
            self.compare_learning(restored,durable)
            self.assertEqual(restored.curriculum['cursor'],3)

    def test_partial_step_rolls_back_successful_uncommitted_updates_and_rng(self):
        backend=self.backend();draw=backend._trainer._draw;calls=0
        def fail_after_consumption(family):
            nonlocal calls
            result=draw(family);calls+=1
            if calls==4:raise RuntimeError('partial second step')
            return result
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.pt'
            with patch.object(backend._trainer,'_draw',side_effect=fail_after_consumption):
                with self.assertRaisesRegex(RuntimeError,'partial second'):
                    backend.train_chunk([FAMILIES]*3,max_updates=3,commit_interval=3,checkpoint_path=path)
            self.assertEqual(backend.updates,0);self.assertEqual(backend.curriculum['cursor'],0)
            self.assertEqual(backend.last_report['discarded_completed_updates'],1)
            self.assertEqual(backend.last_report['failed_step_attempts'],1)
            self.assertGreater(backend.last_report['discarded_step_seconds'],0)
            self.assertGreaterEqual(backend.last_report['wall_seconds'],backend.last_report['step_seconds'])
            restored=RealizationBackend.load(path,self.banks,protected_transcripts=[],evaluation_banks=self.evaluation)
            self.compare_learning(backend,restored)
            backend.train_chunk(max_updates=1);restored._disk_path=None;restored._disk_digest=None
            restored.train_chunk(max_updates=1);self.compare_learning(backend,restored)

    def test_failure_keeps_earlier_interval_commit(self):
        backend=self.backend();draw=backend._trainer._draw;calls=0
        def fail(family):
            nonlocal calls
            result=draw(family);calls+=1
            if calls==4:raise KeyboardInterrupt()
            return result
        with patch.object(backend._trainer,'_draw',side_effect=fail):
            with self.assertRaises(KeyboardInterrupt):backend.train_chunk([FAMILIES]*2,max_updates=2,commit_interval=1)
        self.assertEqual(backend.updates,1);self.assertEqual(backend.curriculum['cursor'],1)
        self.assertEqual(backend.last_report['discarded_completed_updates'],0)
        self.assertEqual(backend.last_report['committed_updates'],1)
        backend.train_chunk(max_updates=1)
        self.assertEqual(backend.updates,2)

    def test_failed_atomic_replace_preserves_disk_and_rolls_back(self):
        backend=self.backend()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.pt'
            backend.train_chunk([FAMILIES]*2,max_updates=1,checkpoint_path=path)
            before=path.read_bytes();original=backend.snapshot()
            with patch('experiments.realization_backend.os.replace',side_effect=OSError('save failure')):
                with self.assertRaisesRegex(OSError,'save failure'):backend.train_chunk(max_updates=1)
            self.assertEqual(path.read_bytes(),before)
            self.compare(backend.snapshot(),original)
            self.assertEqual(backend.last_report['failure_stage'],'commit')
            self.assertEqual(backend.last_report['discarded_completed_updates'],1)
            self.assertFalse(list(Path(directory).glob('*.tmp')))

    def test_error_after_publication_keeps_new_durable_state_and_correct_accounting(self):
        import os
        for boundary in ('replace','write_return'):
            with self.subTest(boundary=boundary),tempfile.TemporaryDirectory() as directory:
                backend=self.backend();path=Path(directory)/'state.pt'
                backend.train_chunk([FAMILIES]*2,max_updates=1,checkpoint_path=path)
                replace=os.replace;write=backend._write
                def after_replace(*args):
                    replace(*args)
                    raise KeyboardInterrupt('after atomic publication')
                def after_write(payload):
                    write(payload)
                    raise KeyboardInterrupt('after atomic publication')
                target=patch('experiments.realization_backend.os.replace',side_effect=after_replace) if boundary=='replace' else patch.object(backend,'_write',side_effect=after_write)
                with target,self.assertRaises(KeyboardInterrupt):backend.train_chunk(max_updates=1)
                self.assertEqual(backend.updates,2)
                self.assertEqual(backend.last_report['committed_updates'],1)
                self.assertEqual(backend.last_report['discarded_completed_updates'],0)
                self.assertEqual(backend.last_report['discarded_step_seconds'],0.)
                loaded=RealizationBackend.load(path,self.banks,protected_transcripts=[],evaluation_banks=self.evaluation)
                self.compare_learning(backend,loaded)
                backend.train_chunk([FAMILIES],max_updates=1)
                self.assertEqual(backend.updates,3)

    def test_wall_deadline_includes_save_and_stops_after_one_inflight_step(self):
        backend=self.backend();clock=[0.];step=backend._trainer.step;write=backend._write
        def timed_step(*a,**kw):
            result=step(*a,**kw);clock[0]+=4.;return result
        def timed_write(*a,**kw):
            result=write(*a,**kw);clock[0]+=2.;return result
        with tempfile.TemporaryDirectory() as directory,patch('experiments.realization_backend.time.monotonic',side_effect=lambda:clock[0]),patch.object(backend._trainer,'step',side_effect=timed_step),patch.object(backend,'_write',side_effect=timed_write):
            report=backend.train_chunk([FAMILIES]*3,deadline=3.,checkpoint_path=Path(directory)/'state.pt')
        self.assertEqual(report['status'],'deadline')
        self.assertEqual(report['completed_updates'],1);self.assertEqual(report['committed_updates'],1)
        self.assertEqual(report['step_seconds'],4.);self.assertEqual(report['commit_seconds'],4.)
        self.assertEqual(report['wall_seconds'],8.);self.assertEqual(report['deadline_overrun_seconds'],5.)
        self.assertEqual(backend.curriculum['cursor'],1)

    def test_uncertain_publication_blocks_writes_without_claiming_retention(self):
        import os
        backend=self.backend();replace=os.replace
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.pt'
            backend.train_chunk([FAMILIES]*2,max_updates=1,checkpoint_path=path)
            def externally_changed(*args):
                replace(*args)
                Path(args[1]).write_bytes(b'unknown publication state')
                raise OSError('ambiguous publication')
            with patch('experiments.realization_backend.os.replace',side_effect=externally_changed):
                with self.assertRaises(OSError):backend.train_chunk(max_updates=1)
            self.assertTrue(backend.last_report['publication_uncertain'])
            self.assertIsNone(backend.last_report['retained_total_updates'])
            self.assertIsNone(backend.last_report['committed_updates'])
            self.assertEqual(backend.last_report['in_memory_updates'],2)
            damaged=path.read_bytes()
            with self.assertRaisesRegex(RuntimeError,'publication is uncertain'):backend.train_chunk(max_updates=1)
            self.assertIsNone(backend.last_report['retained_total_updates'])
            self.assertEqual(path.read_bytes(),damaged)

    def test_synchronization_failure_records_step_time_and_rolls_back(self):
        backend=self.backend();calls=0
        def failure():
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('synchronization failed')
        with patch.object(backend,'_synchronize',side_effect=failure):
            with self.assertRaisesRegex(RuntimeError,'synchronization failed'):
                backend.train_chunk([FAMILIES],max_updates=1)
        self.assertEqual(backend.updates,0)
        self.assertGreater(backend.last_report['step_seconds'],0)
        self.assertEqual(backend.last_report['failed_step_attempts'],1)

    def test_expired_or_zero_budget_does_not_consume_or_install_schedule(self):
        backend=self.backend();before=backend.snapshot()
        for bounds in ({'max_updates':0},{'deadline':time.monotonic()-1}):
            report=backend.train_chunk([FAMILIES],**bounds)
            self.assertEqual(report['completed_updates'],0)
            self.compare(before,backend.snapshot())

    def test_pre_step_synchronization_cannot_start_work_after_deadline(self):
        backend=self.backend();clock=[0.]
        def synchronize():clock[0]+=2.
        with patch('experiments.realization_backend.time.monotonic',side_effect=lambda:clock[0]),patch.object(backend,'_synchronize',side_effect=synchronize):
            report=backend.train_chunk([FAMILIES],deadline=1.)
        self.assertEqual(report['status'],'deadline')
        self.assertEqual(report['completed_updates'],0)
        self.assertEqual(report['step_seconds'],0.)
        self.assertEqual(report['wall_seconds'],2.)
        self.assertEqual(backend.updates,0)

    def test_pending_schedule_cannot_be_replaced_and_exhausted_can_advance(self):
        backend=self.backend();backend.train_chunk([FAMILIES]*2,max_updates=1)
        with self.assertRaisesRegex(ValueError,'pending schedule'):backend.train_chunk([FAMILIES],max_updates=1)
        backend.train_chunk(max_updates=1)
        backend.train_chunk([('count','color','count')],max_updates=1)
        self.assertEqual(backend.updates,3);self.assertEqual(backend.curriculum['generation'],2)
        for bounds in ({},{'max_updates':True},{'deadline':float('nan')},{'max_updates':1,'commit_interval':0}):
            with self.assertRaises(ValueError):backend.train_chunk(**bounds)
        with self.assertRaises(ValueError):backend.train_chunk([('unknown',)*3],max_updates=1)

    def test_restore_rejects_cursor_source_and_stream_tampering_transactionally(self):
        backend=self.backend();backend.train_chunk([FAMILIES],max_updates=1)
        before=backend.snapshot()
        for mutate in (
            lambda p:p['curriculum'].update(cursor=0),
            lambda p:p['contract']['source_sha256'].update(fake='0'*64),
            lambda p:p['learner']['realization']['occurrences']['color'][8].__setitem__(0,999),
            lambda p:p['accounting'].update(retained_step_seconds=float('nan'))):
            bad=copy.deepcopy(before);mutate(bad)
            with self.assertRaises(ValueError):backend.restore(bad)
            self.compare(before,backend.snapshot())
        changed=copy.deepcopy(backend._sources);changed['fake']='0'*64
        with patch('experiments.realization_backend.source_hashes',return_value=changed):
            with self.assertRaisesRegex(ValueError,'source changed'):backend.train_chunk(max_updates=1)
        self.assertEqual(backend.last_report['status'],'failed')
        self.assertEqual(backend.last_report['failure_stage'],'preflight')

    def test_named_evaluation_preserves_state_and_returns_teacher_free_progress(self):
        backend=self.backend();before=backend.snapshot()
        result=backend.evaluate('development',batch_size=2)
        self.compare(before,backend.snapshot())
        self.assertFalse(result['automatic_promotion']);self.assertNotIn('macro_accuracy',result)
        vector=result['progress']['composed/color/t8']
        self.assertEqual(vector['final_pairs']['total'],1)
        self.assertEqual(vector['final_reply_pairs']['total'],1)
        self.assertGreater(vector['known_queries']['total'],0)
        self.assertGreater(vector['unknown_queries']['total'],0)
        metrics=result['per_bank']['composed/color/t8']
        self.assertFalse(metrics['teacher_used_for_policy']);self.assertEqual(metrics['decoder_prefix'],'BOS only')
        backend.evaluate('retention',batch_size=2,control='reset')
        with self.assertRaises(ValueError):backend.evaluate('audit')
        with self.assertRaises(ValueError):backend.evaluate('development',names=['missing'])

    def test_progress_vector_reports_unsupported_ask_instead_of_hiding_it(self):
        targets=torch.tensor([[3,2,0],[3,2,1]],dtype=torch.long)
        logits=torch.zeros(2,3,4);logits[...,2]=10
        metrics=prediction_metrics(logits,targets,reply_correct=torch.zeros_like(targets,dtype=torch.bool),
                                   reply_actions=torch.full_like(targets,2))
        vector=progress_vector(metrics)
        self.assertEqual(vector['unsupported_ask'],{'count':2,'known_total':2,'rate':1.})
        self.assertEqual(vector['unknown_queries']['recall'],1.)
        self.assertEqual(vector['final_pairs'],{'correct':0,'total':1})
        self.assertEqual(vector['known_queries']['accuracy'],0.)

    def test_evaluation_admission_rejects_audit_and_training_overlap(self):
        audit={'development':{'bad':generate_pair('color',281000001,split='audit',turns=8)}}
        with self.assertRaises(ValueError):
            RealizationBackend(self.banks,mode='fresh',protected_transcripts=[],evaluation_banks=audit,config=self.config)
        original=self.banks['color'][8][0]['recipe']
        duplicate=generate_pair('color',original['seed'],split='dev',turns=8,
                    naming_seed=original['naming_seed'],value_seed=original['value_seed'],structure_split='train')
        with self.assertRaisesRegex(ValueError,'protected'):
            RealizationBackend(self.banks,mode='fresh',protected_transcripts=[],
                               evaluation_banks={'development':{'duplicate':duplicate}},config=self.config)

    def test_dev_admission_does_not_expose_final_or_earlier_audit_ancestry(self):
        from experiments.composition_curriculum import query_ancestries
        final=generate_pair('color',282000001,split='dev',turns=8,structure_split='audit')
        earlier=None
        for seed in range(282000010,282000510):
            pair=generate_pair('color',seed,split='dev',turns=12,structure_split='dev')
            if any(q['known'] and q['composed'] and q['structure_partition']=='audit'
                   for q in query_ancestries(pair[0])[:-1]):
                earlier=pair;break
        self.assertIsNotNone(earlier)
        for rows in (final,earlier):
            for role in ('development','retention'):
                with self.assertRaisesRegex(ValueError,'supervised audit ancestry'):
                    RealizationBackend(self.banks,mode='fresh',protected_transcripts=[],
                                       evaluation_banks={role:{'bad':rows}},config=self.config)

    def test_stale_writer_and_wrong_resume_bank_are_rejected(self):
        backend=self.backend()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.pt';backend.save(path)
            stale=RealizationBackend.load(path,self.banks,protected_transcripts=[],evaluation_banks=self.evaluation)
            backend.train_chunk([FAMILIES],max_updates=1)
            with self.assertRaisesRegex(RuntimeError,'advanced or changed'):stale.train_chunk([FAMILIES],max_updates=1)
            self.assertEqual(stale.updates,0)
            with self.assertRaises(FileExistsError):self.backend().save(path)
            changed=copy.deepcopy(self.evaluation)
            changed['development']['composed/color/t8']=generate_pair('color',281000003,split='dev',turns=8)
            with self.assertRaises(ValueError):RealizationBackend.load(path,self.banks,protected_transcripts=[],evaluation_banks=changed)


if __name__=='__main__':unittest.main()
