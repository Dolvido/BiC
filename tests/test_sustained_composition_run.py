"""Pure continuation admission/accounting; no checkpoints, model or dataset."""
from copy import deepcopy
import unittest

from experiments import sustained_composition_run as r


def manifest():
    records={arm:[] for arm in r.ARMS}
    for group in range(108):
        for position in range(6):
            kind='replay' if position in (1,3,5) else 'old_definition' if position==2 else 'basis'
            for arm in r.ARMS:
                actual=kind
                if arm=='curriculum' and position in (0,4) and (2*group+(position==4))%6:
                    actual='complementary'
                records[arm].append(dict(kind=actual,index=len(records[arm]),sha256=str(len(records[arm]))))
    return dict(schema='bic-complementary-composition-data-v1',updates_per_arm=648,micro_batch_size=32,
        parent={'lifetime_updates':9760},phases={'training':records})


def parents():
    previous=dict(parent=dict(lifetime_updates=9760,checkpoint={'sha256':'shared'}),
        reference_parent={'lifetime_updates':9112},data_manifest_sha256=r.DATA_MANIFEST_SHA256,
        origin_identity={'recipe':'unchanged'})
    summary=dict(schema='bic-complementary-composition-run-v1',status='completed',partial_work_unknown=False,
        new_updates={a:648 for a in r.ARMS},launch_sha256=r.PARENT_LAUNCH_SHA256,
        continuations={},evaluations={},physical_training={})
    commits={}
    for arm in r.ARMS:
        checkpoint=dict(path=f'{arm}.pt',sha256=r.PARENT_CHECKPOINTS[arm])
        scores={name:{'metrics':{'branch':arm}} for name in r.BANK_NAMES}
        commits[arm]=dict(schema='bic-complementary-composition-run-v1',arm=arm,relative_step=648,lifetime_updates=10408,
            launch_sha256=r.PARENT_LAUNCH_SHA256,checkpoint=checkpoint,weights_sha256=arm,
            parent_checkpoint=previous['parent']['checkpoint'],data_manifest_sha256=r.DATA_MANIFEST_SHA256,scores=scores)
        summary['continuations'][arm]=dict(checkpoint=deepcopy(checkpoint),weights_sha256=arm,lifetime_updates=10408)
        summary['evaluations'][arm]={'648':deepcopy(scores)}
        summary['physical_training'][arm]={'work':{'unknown_optimizer_outcomes':0}}
    return summary,previous,commits


class SustainedTests(unittest.TestCase):
    def terminal_fixture(self,start,summary_seconds=0.,marker_seconds=0.,status='completed'):
        tick=[start];saved={};pins={}
        def publisher(path,value):
            saved[path.name]=deepcopy(value);pins[path.name]=r.identity(value)
            tick[0]+=summary_seconds if path.name=='summary.json' else marker_seconds
            return pins[path.name]
        receipt=dict(status=status,launch_sha256='launch',continuations={'control':{'cursor':13000}})
        result=r.publish_terminal(receipt,r.ROOT/'runs/synthetic-unused/execution',started=0.,cpu_started=0.,
            deadline=3600.,clock=lambda:tick[0],cpu_clock=lambda:0.,publisher=publisher)
        self.assertEqual(saved['completion.json']['summary']['sha256'],pins['summary.json'])
        return result,saved

    def test_cleanup_overrun_refuses_success_before_summary(self):
        result,saved=self.terminal_fixture(3600.)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(saved['summary.json']['status'],'failed')
        self.assertEqual(saved['summary.json']['continuations'],{})
        self.assertEqual(saved['summary.json']['deadline_failure'],'cleanup_exceeded_worker_deadline')
        self.assertFalse(saved['completion.json']['continuation_eligible'])

    def test_summary_publication_overrun_preserves_summary_but_fails_marker(self):
        result,saved=self.terminal_fixture(3599.5,summary_seconds=.5)
        self.assertEqual(saved['summary.json']['status'],'completed')
        self.assertEqual(result['status'],'failed')
        self.assertEqual(saved['completion.json']['wall_seconds'],3600.)
        self.assertEqual(saved['completion.json']['reason'],'summary_publication_exceeded_worker_deadline')
        self.assertFalse(saved['completion.json']['continuation_eligible'])

    def test_success_excludes_separately_measured_marker_but_cannot_hide_worker_failure(self):
        result,saved=self.terminal_fixture(3599.,summary_seconds=.5,marker_seconds=1.)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['wall_seconds'],3599.5)
        self.assertEqual(result['completion_publication_wall_seconds'],1.)
        self.assertTrue(saved['completion.json']['continuation_eligible'])
        result,saved=self.terminal_fixture(10.,status='failed')
        self.assertEqual(result['status'],'failed')
        self.assertEqual(saved['completion.json']['summary_status'],'failed')

    def test_single_fixed_budget_and_endpoints(self):
        self.assertEqual(r.contract()['endpoints'],[0,648,1296,2592])
        self.assertEqual(r.contract()['max_seconds'],3600)
        self.assertEqual(r.PARENT_CURSOR+r.UPDATES,13000)
        for value in (1200,3599,3601,3600.,True):
            with self.assertRaises(ValueError):r.contract(value)

    def test_four_exact_passes_keep_original_records_and_order(self):
        data=manifest();before=deepcopy(data);plan=r.physical_plan(data)
        self.assertEqual((plan['updates'],plan['episode_exposures'],plan['restores'],plan['snapshots']),
                         (5184,497664,8,6))
        for arm in r.ARMS:
            source=data['phases']['training'][arm]
            for i in range(2592):self.assertIs(r.replay_record(source,i),source[i%648])
            for bad in (-1,2592,True):
                with self.assertRaises(ValueError):r.replay_record(source,bad)
        self.assertEqual(data,before)
        data['phases']['training']['curriculum'][1]['kind']='complementary'
        with self.assertRaises(ValueError):r.physical_plan(data)

    def test_distinct_own_parents_and_sixteen_baselines(self):
        summary,previous,commits=parents();actual=r.parent_admission(summary,previous,commits)
        self.assertNotEqual(actual['control']['metrics'],actual['curriculum']['metrics'])
        for arm in r.ARMS:
            self.assertEqual(actual[arm]['checkpoint']['sha256'],r.PARENT_CHECKPOINTS[arm])
            self.assertEqual(set(actual[arm]['metrics']),set(r.BANK_NAMES))
            self.assertEqual(actual[arm]['identity_sha256'],r.identity(previous['origin_identity']))
        actual['control']['metrics'].clear()
        self.assertEqual(len(commits['control']['scores']),16)

    def test_foreign_incomplete_or_unknown_parent_refused(self):
        for kind in ('swapped','scores','unknown','incomplete','reference','data'):
            summary,previous,commits=parents()
            if kind=='swapped':commits['control']['checkpoint']=deepcopy(commits['curriculum']['checkpoint'])
            elif kind=='scores':commits['curriculum']['scores']=deepcopy(commits['control']['scores'])
            elif kind=='unknown':summary['physical_training']['curriculum']['work']['unknown_optimizer_outcomes']=1
            elif kind=='incomplete':summary['new_updates']['control']=647
            elif kind=='reference':previous['reference_parent']['lifetime_updates']=9760
            else:previous['data_manifest_sha256']='bad'
            with self.subTest(kind=kind),self.assertRaises(ValueError):r.parent_admission(summary,previous,commits)

    def test_launch_keeps_branch_shared_and_retained_identities_separate(self):
        summary,previous,commits=parents();admitted=r.parent_admission(summary,previous,commits)
        launch=dict(**r.contract(),parents=admitted,parent_metadata={a:{'cursor':10408} for a in r.ARMS},
            parent_summary_sha256=r.PARENT_SUMMARY_SHA256,data_manifest_sha256=r.DATA_MANIFEST_SHA256,
            shared_parent=previous['parent'],reference_parent=previous['reference_parent'])
        r.validate_launch(launch)
        for kind in ('parent','shared','data','endpoint'):
            bad=deepcopy(launch)
            if kind=='parent':bad['parents']['curriculum']['checkpoint']=deepcopy(bad['parents']['control']['checkpoint'])
            elif kind=='shared':bad['shared_parent']['lifetime_updates']=10408
            elif kind=='data':bad['data_manifest_sha256']='bad'
            else:bad['endpoints']=[0,648,1296,1944,2592]
            with self.assertRaises(ValueError):r.validate_launch(bad)

    def test_accounting_uses_each_own_parent_including_partial_failure(self):
        def account(updates,cost,unknown=0):return dict(work={'retained_updates':updates,'unknown_optimizer_outcomes':unknown},
            state_work={'completed_state_objectives':3*updates},cost={'wall_seconds':cost})
        inherited={'control':{'accounting':account(10408,100.)},'curriculum':{'accounting':account(10408,200.)}}
        latest={'control':{'accounting':{'lifetime_kernel':account(13000,400.)}},
                'curriculum':{'accounting':{'lifetime_kernel':account(10409,207.,1)}}}
        delta=r.physical_deltas(latest,inherited)
        self.assertEqual(delta['control']['work']['retained_updates'],2592)
        self.assertEqual(delta['curriculum']['work'],{'retained_updates':1,'unknown_optimizer_outcomes':1})
        self.assertEqual(delta['curriculum']['cost']['wall_seconds'],7.)
        self.assertEqual(r.physical_deltas({},inherited),{})
        latest['curriculum']['accounting']['lifetime_kernel']['work']['retained_updates']=10407
        with self.assertRaises(ValueError):r.physical_deltas(latest,inherited)

    def test_complete_physical_counts_reject_missing_or_unknown_work(self):
        work=dict(retained_updates=2592,synchronized_optimizer_updates=2592,optimizer_attempts=2592,optimizer_returns=2592,
            unknown_optimizer_outcomes=0,retained_episodes=248832,attempted_forwards=7776,completed_forwards=7776,
            attempted_backwards=7776,completed_backwards=7776,attempted_forward_episodes=248832,completed_forward_episodes=248832,
            attempted_backward_episodes=248832,completed_backward_episodes=248832)
        state={k:7776 for k in ('attempted_state_readouts','completed_state_readouts','attempted_state_objectives','completed_state_objectives')}
        physical={a:dict(work=deepcopy(work),state_work=deepcopy(state)) for a in r.ARMS}
        receipt=dict(new_updates={a:2592 for a in r.ARMS},restorations=[{}]*8,snapshots=[{}]*6)
        plan=r.physical_plan(manifest());r.validate_completed_work(plan,receipt,physical)
        for key,value in (('retained_episodes',248831),('unknown_optimizer_outcomes',1),('optimizer_returns',2591)):
            bad=deepcopy(physical);bad['curriculum']['work'][key]=value
            with self.assertRaises(ValueError):r.validate_completed_work(plan,receipt,bad)


if __name__=='__main__':unittest.main()
