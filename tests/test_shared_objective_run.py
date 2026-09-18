"""Pure shared-parent objective comparison boundaries; no neural fixtures."""
from copy import deepcopy
import unittest

from experiments import shared_objective_run as r

TEST_WORK=dict(test_cases=0,synthetic_manifests=0,parent_fixtures=0,
               schedule_lookups=0,terminal_cases=0,accounting_cases=0)


def manifest():
    TEST_WORK['synthetic_manifests']+=1
    records={arm:[] for arm in ('control','curriculum')}
    for group in range(108):
        for position in range(6):
            kind='replay' if position in (1,3,5) else 'old_definition' if position==2 else 'basis'
            for arm in records:
                actual=kind
                if arm=='curriculum' and position in (0,4) and (2*group+(position==4))%6:actual='complementary'
                records[arm].append(dict(kind=actual,index=len(records[arm]),sha256=str(len(records[arm]))))
    return dict(schema='bic-complementary-composition-data-v1',updates_per_arm=648,micro_batch_size=32,
        parent={'lifetime_updates':9760},phases={'training':records})


def parents():
    TEST_WORK['parent_fixtures']+=1
    previous=dict(shared_parent={'lifetime_updates':9760},reference_parent={'lifetime_updates':9112},
        parents={'curriculum':dict(lifetime_updates=10408,checkpoint={'sha256':'incoming-curriculum'})},
        data_manifest_sha256=r.DATA_MANIFEST_SHA256,origin_identity={'recipe':'unchanged-origin'})
    checkpoint=dict(path='synthetic-unused.pt',sha256=r.PARENT_CHECKPOINT_SHA256)
    scores={name:dict(metrics={'source':'same-parent'}) for name in r.BANK_NAMES}
    summary=dict(schema='bic-sustained-composition-run-v1',status='completed',partial_work_unknown=False,
        new_updates=dict(control=2592,curriculum=2592),launch_sha256=r.PARENT_LAUNCH_SHA256,
        continuations={'curriculum':dict(checkpoint=deepcopy(checkpoint),weights_sha256='a'*64,lifetime_updates=13000)},
        evaluations={'curriculum':{'2592':deepcopy(scores)}},
        physical_training={arm:dict(work={'unknown_optimizer_outcomes':0}) for arm in ('control','curriculum')})
    commit=dict(schema='bic-sustained-composition-run-v1',arm='curriculum',relative_step=2592,lifetime_updates=13000,
        launch_sha256=r.PARENT_LAUNCH_SHA256,checkpoint=checkpoint,weights_sha256='a'*64,
        parent_checkpoint=deepcopy(previous['parents']['curriculum']['checkpoint']),
        data_manifest_sha256=r.DATA_MANIFEST_SHA256,scores=deepcopy(scores))
    completion=dict(schema='bic-sustained-composition-completion-v1',status='completed',summary_status='completed',
        summary={'sha256':r.PARENT_SUMMARY_SHA256},launch_sha256=r.PARENT_LAUNCH_SHA256,
        completed_within_deadline=True,continuation_eligible=True,wall_seconds=1000.,max_seconds=3600)
    return summary,previous,commit,completion


class ObjectiveRunTests(unittest.TestCase):
    def setUp(self):TEST_WORK['test_cases']+=1

    def terminal_fixture(self,start,summary_seconds=0.,marker_seconds=0.,status='completed'):
        TEST_WORK['terminal_cases']+=1
        tick=[start];saved={};pins={}
        def publisher(path,value):
            saved[path.name]=deepcopy(value);pins[path.name]=r.identity(value)
            tick[0]+=summary_seconds if path.name=='summary.json' else marker_seconds
            return pins[path.name]
        receipt=dict(status=status,launch_sha256='launch',continuations={'zero':{'cursor':15592}})
        result=r.publish_terminal(receipt,r.ROOT/'runs/synthetic-unused/execution',started=0.,cpu_started=0.,
            deadline=3600.,clock=lambda:tick[0],cpu_clock=lambda:0.,publisher=publisher)
        self.assertEqual(saved['completion.json']['summary']['sha256'],pins['summary.json'])
        return result,saved

    def test_cleanup_and_publication_overrun_do_not_admit_continuation(self):
        result,saved=self.terminal_fixture(3600.)
        self.assertEqual(result['status'],'failed');self.assertEqual(saved['summary.json']['continuations'],{})
        self.assertFalse(saved['completion.json']['continuation_eligible'])
        result,saved=self.terminal_fixture(3599.5,summary_seconds=.5)
        self.assertEqual(saved['summary.json']['status'],'completed');self.assertEqual(result['status'],'failed')
        self.assertEqual(saved['completion.json']['wall_seconds'],3600.)
        self.assertFalse(saved['completion.json']['continuation_eligible'])

    def test_terminal_success_separates_marker_and_cannot_hide_failed_worker(self):
        result,saved=self.terminal_fixture(3599.,summary_seconds=.5,marker_seconds=1.)
        self.assertEqual(result['status'],'completed');self.assertEqual(result['wall_seconds'],3599.5)
        self.assertEqual(result['completion_publication_wall_seconds'],1.)
        result,saved=self.terminal_fixture(10.,status='failed')
        self.assertEqual(result['status'],'failed');self.assertFalse(saved['completion.json']['continuation_eligible'])

    def test_four_replays_preserve_original_coordinates(self):
        source=manifest()['phases']['training']['curriculum'];before=deepcopy(source)
        for _ in ('control','zero'):
            for index in range(2592):
                TEST_WORK['schedule_lookups']+=1
                self.assertIs(r.replay_record(source,index),source[index%648])
        self.assertEqual(source,before)
        for bad in (-1,2592,True):
            with self.assertRaises(ValueError):r.replay_record(source,bad)
        with self.assertRaises(ValueError):r.replay_record(source[:-1],0)

    def test_delta_preserves_history_and_reports_skipped_auxiliary_work(self):
        before=dict(work={'retained_updates':13000,'unknown_optimizer_outcomes':0},
            state_work={'completed_state_objectives':39000},cost={'seconds':100.})
        for state_delta in (7776,0):
            TEST_WORK['accounting_cases']+=1
            after=dict(work={'retained_updates':15592,'unknown_optimizer_outcomes':0},
                state_work={'completed_state_objectives':39000+state_delta},cost={'seconds':150.})
            delta=r.accounting_delta(after,before)
            self.assertEqual(delta['work']['retained_updates'],2592)
            self.assertEqual(delta['state_work']['completed_state_objectives'],state_delta)
            self.assertEqual(delta['cost']['seconds'],50.)
        after['work']['retained_updates']=12999
        with self.assertRaises(ValueError):r.accounting_delta(after,before)

    def test_fixed_endpoints_budget_schedule_and_only_declared_recipe_change(self):
        value=r.contract()
        self.assertEqual(value['endpoints'],[0,648,1296,2592]);self.assertEqual(value['max_seconds'],3600)
        self.assertEqual(value['auxiliary_weights'],{'control':.3,'zero':0.})
        self.assertEqual(value['source_schedule'],'curriculum')
        self.assertEqual(r.PARENT_CURSOR+r.UPDATES,15592)
        for seconds in (1200,3599,3601,3600.,True):
            with self.assertRaises(ValueError):r.contract(seconds)
        parent={'auxiliary_weight':.3,'optimizer':[{'lr':.0003,'betas':(.9,.999)}],
            'source_sha256':{'same.py':'b'*64},'objective_id':'unchanged','micro_batch_size':32}
        original=deepcopy(parent)
        for arm,weight in (('control',.3),('zero',0.)):
            changed=r.expected_recipe(parent,arm);self.assertEqual(changed,{**parent,'auxiliary_weight':weight})
            changed['optimizer'][0]['lr']=99
            self.assertEqual(parent,original)
        with self.assertRaises(ValueError):r.expected_recipe(parent,'other')
        with self.assertRaises(ValueError):r.expected_recipe({**parent,'auxiliary_weight':0.},'zero')

    def test_both_arms_receive_independent_copies_of_same_parent(self):
        values=parents();admitted=r.parent_admission(*values)
        self.assertEqual(admitted['control'],admitted['zero'])
        self.assertEqual(admitted['control']['checkpoint']['sha256'],r.PARENT_CHECKPOINT_SHA256)
        self.assertEqual(admitted['control']['identity_sha256'],r.identity(values[1]['origin_identity']))
        self.assertEqual(set(admitted['control']['metrics']),set(r.BANK_NAMES))
        admitted['control']['metrics']['basis_binding']['source']='mutation'
        self.assertEqual(admitted['zero']['metrics']['basis_binding']['source'],'same-parent')
        self.assertEqual(values[2]['scores']['basis_binding']['metrics']['source'],'same-parent')

    def test_foreign_partial_or_ineligible_parent_is_rejected(self):
        for case in ('checkpoint','branch','weights','scores','unknown','incomplete','data','reference',
                     'completion_missing','completion_summary','completion_launch','ineligible','deadline','nan'):
            summary,previous,commit,completion=parents()
            if case=='checkpoint':commit['checkpoint']['sha256']='c'*64
            elif case=='branch':commit['arm']='control'
            elif case=='weights':summary['continuations']['curriculum']['weights_sha256']='d'*64
            elif case=='scores':commit['scores'].pop('basis_binding')
            elif case=='unknown':summary['physical_training']['control']['work']['unknown_optimizer_outcomes']=1
            elif case=='incomplete':summary['new_updates']['curriculum']=2591
            elif case=='data':previous['data_manifest_sha256']='bad'
            elif case=='reference':previous['reference_parent']['lifetime_updates']=9760
            elif case=='completion_missing':completion={}
            elif case=='completion_summary':completion['summary']['sha256']='e'*64
            elif case=='completion_launch':completion['launch_sha256']='f'*64
            elif case=='ineligible':completion['continuation_eligible']=False
            elif case=='deadline':completion['wall_seconds']=3600.
            else:completion['wall_seconds']=float('nan')
            with self.subTest(case=case),self.assertRaises(ValueError):r.parent_admission(summary,previous,commit,completion)

    def test_launch_requires_identical_full_incoming_metadata(self):
        values=parents();admitted=r.parent_admission(*values);previous=values[1]
        launch=dict(**r.contract(),parents=admitted,
            parent_metadata={arm:dict(cursor=13000,recipe={'auxiliary_weight':.3},optimizer_identity='same') for arm in r.ARMS},
            parent_summary_sha256=r.PARENT_SUMMARY_SHA256,parent_completion_sha256=r.PARENT_COMPLETION_SHA256,
            data_manifest_sha256=r.DATA_MANIFEST_SHA256,shared_parent=previous['shared_parent'],
            previous_curriculum_parent=previous['parents']['curriculum'],reference_parent=previous['reference_parent'])
        r.validate_launch(launch)
        for case in ('parent','optimizer','weight','schedule','completion','endpoint'):
            bad=deepcopy(launch)
            if case=='parent':bad['parents']['zero']['weights_sha256']='b'*64
            elif case=='optimizer':bad['parent_metadata']['zero']['optimizer_identity']='different'
            elif case=='weight':bad['auxiliary_weights']['zero']=.3
            elif case=='schedule':bad['source_schedule']='control'
            elif case=='completion':bad['parent_completion_sha256']='a'*64
            else:bad['endpoints']=[0,648,1296,1944,2592]
            with self.subTest(case=case),self.assertRaises(ValueError):r.validate_launch(bad)

    def test_complete_counts_include_exact_zero_state_work(self):
        value=manifest();plan=r.physical_plan(value)
        self.assertEqual((plan['updates'],plan['episode_exposures'],plan['restores'],plan['snapshots']),
            (5184,497664,8,6))
        self.assertEqual(plan['new_state_readouts'],{'control':7776,'zero':0})
        work=dict(retained_updates=2592,synchronized_optimizer_updates=2592,optimizer_attempts=2592,optimizer_returns=2592,
            unknown_optimizer_outcomes=0,retained_episodes=248832,attempted_forwards=7776,completed_forwards=7776,
            attempted_backwards=7776,completed_backwards=7776,attempted_forward_episodes=248832,completed_forward_episodes=248832,
            attempted_backward_episodes=248832,completed_backward_episodes=248832)
        state_keys=('attempted_state_readouts','completed_state_readouts','attempted_state_objectives','completed_state_objectives')
        physical={arm:dict(work=deepcopy(work),state_work=dict.fromkeys(state_keys,7776 if arm=='control' else 0)) for arm in r.ARMS}
        receipt=dict(new_updates=dict.fromkeys(r.ARMS,2592),restorations=[{}]*8,snapshots=[{}]*6)
        r.validate_completed_work(plan,receipt,physical)
        for case in ('zero_state','control_state','episodes','unknown','restores','snapshots','updates'):
            bad,summary=deepcopy(physical),deepcopy(receipt)
            if case=='zero_state':bad['zero']['state_work'][state_keys[0]]=3
            elif case=='control_state':bad['control']['state_work'][state_keys[0]]=7773
            elif case=='episodes':bad['zero']['work']['retained_episodes']-=1
            elif case=='unknown':bad['zero']['work']['unknown_optimizer_outcomes']=1
            elif case=='restores':summary['restorations'].pop()
            elif case=='snapshots':summary['snapshots'].pop()
            else:summary['new_updates']['zero']-=1
            with self.subTest(case=case),self.assertRaises(ValueError):r.validate_completed_work(plan,summary,bad)
        value['phases']['training']['curriculum'][1]['kind']='complementary'
        with self.assertRaises(ValueError):r.physical_plan(value)

    def test_zero_auxiliary_fingerprint_must_remain_exact(self):
        for arm in r.ARMS:r.validate_auxiliary('a'*64,'a'*64,arm)
        r.validate_auxiliary('b'*64,'a'*64,'control')
        with self.assertRaises(ValueError):r.validate_auxiliary('b'*64,'a'*64,'zero')
        for pin in ('a'*63,'A'*64,'g'*64,None,True):
            with self.assertRaises(ValueError):r.validate_auxiliary(pin,'a'*64,'zero')
        with self.assertRaises(ValueError):r.validate_auxiliary('a'*64,'a'*64,'foreign')


if __name__=='__main__':unittest.main()
