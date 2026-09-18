"""Synthetic recovery admission only; zero real author, archive, learner or child."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("campaign_fixture_helpers", ROOT/"tests/test_continuous_tutor_campaign.py")
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        fixture.SOURCE = ROOT/"experiments/continuous_tutor_campaign_v2.py"
        self.h = fixture.CampaignTests("runTest"); self.h.setUp(); self.addCleanup(self.h.doCleanups)
        self.owner = self.h.owner
        self.state = self.h.state('authored'); self.state['schema'] = 'bic-continuous-tutor-campaign-v1'
        self.parent = deepcopy(self.state['parent'])
        self.context = dict(parent=dict(identity_sha256=self.parent['checkpoint']['sha256'],
            weights_sha256=self.parent['weights_sha256'], cycle=1,lifetime_updates=self.parent['lifetime_updates']),
            teaching_contract=dict(identity_sha256='c'*64))
        self.state['cycle_context'] = self.h.record('context.json', self.context)
        self.preparation = dict(status='failed',error=self.owner.REPEATED_PRACTICE_ERROR,
            compiler_attempts=2,compiler_completions=2,compiler_receipts=[dict(status='completed') for _ in range(2)],
            fresh_only_rejection=dict(condition='tutor',overlap_count=48),neural_work=0,teacher_calls=0,
            wall_seconds=118.422,cpu_seconds=116.59375)
        self.preparation_record = self.h.record('seed-cycle/compiled/preparation.json',self.preparation)
        self.accepted = dict(outcome='local_accepted',parent=self.context['parent'],contract_sha256='c'*64,
            teacher_cost=dict(physical_teacher_work_unknown=False,chat_attempts=1,chat_completions=1))
        self.state['teacher_decision'] = dict(**self.h.record('accepted.json',self.accepted),request_sha256='d'*64)
        self.previous = dict(schema='bic-continuous-tutor-campaign-v1',budget_seconds=3600,
            initial_state=dict(parent=deepcopy(self.parent)),seed=853002001,teacher=self.h.teacher,
            source_sha256=self.owner.source_hashes(),input_sha256={},origin_identity={'origin':'fixed'},
            parent_runtime={'runtime':'fixed'},evaluation=dict(directory='banks',manifest_sha256=self.h.evaluation['sha256']),
            inventory=dict(directory='inventory',manifest_sha256=self.owner.INVENTORY_SHA),
            freeze_cost=dict(wall_seconds=34.75,cpu_seconds=34),scope='mock research only')
        self.summary = dict(status='failed',error=self.owner.REPEATED_PRACTICE_ERROR,partial_work_unknown=False,
            worker_processes=[],physical_totals=deepcopy(self.state['physical_totals']),
            active_stage=dict(stage='authored',cycle=1),final_stage='authored',wall_seconds=130.156,cpu_seconds=128.)
        self.h.author.load_result = Mock(side_effect=self.load_result)
        # Any attempt to issue transport or execute packaging is a test failure.
        self.h.author.author_curriculum = Mock(side_effect=AssertionError('no new author request'))
        self.h.data.compile_data = Mock(side_effect=AssertionError('no data compilation'))
        self.h.worker.freeze = Mock(side_effect=AssertionError('no worker freeze'))

    def admit(self, **changes):
        inputs = {name:deepcopy(getattr(self,name)) for name in ('previous','summary','state','context','preparation')}
        inputs.update(changes)
        return self.owner.recovery_state(**inputs)

    def load_result(self, path, *, expected_sha256, expected_request_sha256):
        if self.h.digest(path) != expected_sha256 or expected_request_sha256 != 'd'*64:
            raise ValueError('mock external teacher pin differs')
        return self.h.read(path)

    def persist(self):
        prior=self.h.root/'prior';prior.mkdir()
        pin=self.h.publish(prior/'launch.json',self.previous)
        self.summary['launch_sha256']=pin
        self.summary['last_state']=self.h.record('prior/state.json',self.state)
        summary_pin=self.h.publish(prior/'execution/summary.json',self.summary)
        return prior,summary_pin

    def test_exact_known_failure_admission_is_detached(self):
        original=deepcopy(self.state)
        restored,budget=self.admit()
        self.assertEqual(budget,3469)
        self.assertEqual(restored['schema'],self.owner.SCHEMA)
        self.assertEqual(restored['parent'],self.parent)
        self.assertEqual(restored['teacher_decision'],self.state['teacher_decision'])
        self.assertEqual(restored['physical_totals']['teacher_chat_completions'],1)
        self.assertEqual(restored['physical_totals']['worker_updates'],0)
        restored['parent']['lifetime_updates']+=1
        self.assertEqual(self.state,original)

    def test_unknown_worker_started_or_foreign_failure_refused(self):
        cases=(('partial_work_unknown',True),('worker_processes',[{'pid':1}]),
            ('status','completed'),('error',"ValueError('different failure')"),
            ('active_stage',dict(stage='compiled',cycle=1)),('active_stage',dict(stage='authored',cycle=2)))
        for key,value in cases:
            with self.subTest(key=key,value=value):
                changed=deepcopy(self.summary);changed[key]=value
                with self.assertRaises(ValueError):self.admit(summary=changed)
        for field,value in (('worker_updates',1),('teacher_chat_attempts',2),('teacher_chat_completions',0)):
            changed=deepcopy(self.summary);changed['physical_totals'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.admit(summary=changed)

    def test_parent_and_complete_failed_compiler_bindings(self):
        for field in ('weights_sha256','lifetime_updates','checkpoint'):
            changed=deepcopy(self.state)
            changed['parent'][field]=('f'*64 if field=='weights_sha256' else 999 if field=='lifetime_updates' else dict(path='foreign',sha256='f'*64))
            with self.subTest(field=field),self.assertRaises(ValueError):self.admit(state=changed)
        changed=deepcopy(self.context);changed['parent']['identity_sha256']='f'*64
        with self.assertRaises(ValueError):self.admit(context=changed)
        for field,value in (('status','completed'),('error','different'),('compiler_completions',1),
                ('compiler_receipts',[dict(status='completed'),dict(status='failed')]),('neural_work',1),
                ('teacher_calls',1),('fresh_only_rejection',dict(condition='procedural',overlap_count=48)),
                ('fresh_only_rejection',dict(condition='tutor',overlap_count=0))):
            changed=deepcopy(self.preparation);changed[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.admit(preparation=changed)

    def test_remaining_budget_debits_failed_work_without_rounding_up(self):
        for spent,expected in ((130.156,3469),(130.,3470),(3480.,120)):
            changed=deepcopy(self.summary);changed['wall_seconds']=spent
            self.assertEqual(self.admit(summary=changed)[1],expected)
        for spent in (0,-1,float('nan'),float('inf'),True,3480.001,3600,3601):
            changed=deepcopy(self.summary);changed['wall_seconds']=spent
            with self.subTest(spent=spent),self.assertRaises(ValueError):self.admit(summary=changed)

    def test_freeze_reuses_exact_teacher_and_pins_all_prior_work(self):
        prior,pin=self.persist()
        out=self.h.root/'recovery'
        result_pin=self.owner.freeze_recovery(out,parent_directory=prior,parent_summary_sha256=pin,expected_preparation_sha256=self.preparation_record["sha256"])
        self.assertEqual(result_pin,self.h.digest(out/'launch.json'))
        launch=self.h.read(out/'launch.json');state=launch['initial_state']
        self.assertEqual(launch['budget_seconds'],3469)
        self.assertEqual(state['parent'],self.parent)
        self.assertEqual(state['teacher_decision'],self.state['teacher_decision'])
        self.assertEqual(state['legacy_context'],self.state['cycle_context'])
        self.assertEqual(state['cycle_directory'],'recovery/execution/cycles/000001')
        self.assertEqual(state['reuse_procedural']['preparation_sha256'],self.preparation_record['sha256'])
        self.assertEqual(launch['recovery']['wall_seconds'],130.156)
        self.assertEqual(launch['recovery']['compilation_cost']['compiler_receipts'],self.preparation['compiler_receipts'])
        for record in (self.state['cycle_context'],self.state['teacher_decision'],self.preparation_record):
            self.assertEqual(launch['input_sha256'][record['path']],record['sha256'])
        self.h.author.load_result.assert_called_once_with(self.h.root/'accepted.json',
            expected_sha256=self.state['teacher_decision']['sha256'],expected_request_sha256='d'*64)
        self.h.author.author_curriculum.assert_not_called();self.h.data.compile_data.assert_not_called();self.h.worker.freeze.assert_not_called()

    def test_freeze_refuses_foreign_teacher_pin_or_unaccepted_result(self):
        prior,pin=self.persist()
        with self.assertRaises(ValueError):
            self.owner.freeze_recovery(self.h.root/'wrong-preparation',parent_directory=prior,
                parent_summary_sha256=pin,expected_preparation_sha256='f'*64)
        self.h.author.load_result.assert_not_called()
        accepted_path=self.h.root/'accepted.json';accepted_path.write_text('{}')
        with self.assertRaises(ValueError):self.owner.freeze_recovery(self.h.root/'wrong-pin',parent_directory=prior,parent_summary_sha256=pin,expected_preparation_sha256=self.preparation_record["sha256"])
        accepted_path.write_bytes(self.h.encoded(self.accepted))
        for field,value in (('outcome','procedural_fallback'),('parent',{}),('contract_sha256','f'*64)):
            rejected=deepcopy(self.accepted);rejected[field]=value
            self.h.author.load_result=Mock(return_value=rejected)
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.owner.freeze_recovery(self.h.root/('rejected-'+field),parent_directory=prior,parent_summary_sha256=pin,expected_preparation_sha256=self.preparation_record["sha256"])
        self.h.author.author_curriculum.assert_not_called()


if __name__=='__main__':unittest.main()
