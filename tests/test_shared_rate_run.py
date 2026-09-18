"""Synthetic rate-run boundaries; no tensors, archives or lesson contents."""
from copy import deepcopy
import unittest

from experiments import shared_rate_run as r

TEST_WORK=dict(test_cases=0,synthetic_manifests=0,parent_fixtures=0,
    schedule_lookups=0,terminal_cases=0,accounting_cases=0,proof_cases=0,uncertainty_cases=0)


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
        parents={'control':dict(lifetime_updates=13000,checkpoint={'sha256':'incoming-control'})},
        previous_curriculum_parent={'lifetime_updates':10408},
        data_manifest_sha256=r.DATA_MANIFEST_SHA256,origin_identity={'recipe':'unchanged-origin'})
    checkpoint=dict(path='synthetic-unused.pt',sha256=r.PARENT_CHECKPOINT_SHA256)
    scores={name:dict(metrics={'source':'same-parent'}) for name in r.BANK_NAMES}
    transition={'identity':{'auxiliary_weight':.3,'parent':'unchanged-objective-control'}}
    transition['sha256']=r.identity(transition['identity'])
    summary=dict(schema='bic-shared-objective-run-v1',status='completed',partial_work_unknown=False,
        new_updates=dict(control=2592,zero=2592),launch_sha256=r.PARENT_LAUNCH_SHA256,
        continuations={'control':dict(checkpoint=deepcopy(checkpoint),weights_sha256='a'*64,
            lifetime_updates=15592,transition=deepcopy(transition))},
        evaluations={'control':{'2592':deepcopy(scores)}},transitions={'control':deepcopy(transition)},
        physical_training={arm:dict(work={'unknown_optimizer_outcomes':0}) for arm in ('control','zero')})
    commit=dict(schema='bic-shared-objective-run-v1',arm='control',relative_step=2592,lifetime_updates=15592,
        launch_sha256=r.PARENT_LAUNCH_SHA256,checkpoint=checkpoint,weights_sha256='a'*64,
        parent_checkpoint=deepcopy(previous['parents']['control']['checkpoint']),transition=deepcopy(transition),
        data_manifest_sha256=r.DATA_MANIFEST_SHA256,scores=deepcopy(scores))
    completion=dict(schema='bic-shared-objective-completion-v1',status='completed',summary_status='completed',
        summary={'sha256':r.PARENT_SUMMARY_SHA256},launch_sha256=r.PARENT_LAUNCH_SHA256,
        completed_within_deadline=True,continuation_eligible=True,wall_seconds=1000.,max_seconds=3600)
    return summary,previous,commit,completion


class RateRunTests(unittest.TestCase):
    def setUp(self):TEST_WORK['test_cases']+=1

    def terminal_fixture(self,start,summary_seconds=0.,marker_seconds=0.,status='completed'):
        TEST_WORK['terminal_cases']+=1
        tick=[start];saved={};pins={}
        def publisher(path,value):
            saved[path.name]=deepcopy(value);pins[path.name]=r.identity(value)
            tick[0]+=summary_seconds if path.name=='summary.json' else marker_seconds
            return pins[path.name]
        receipt=dict(status=status,launch_sha256='launch',continuations={'lower':{'cursor':18184}})
        result=r.publish_terminal(receipt,r.ROOT/'runs/synthetic-unused/execution',started=0.,cpu_started=0.,
            deadline=3600.,clock=lambda:tick[0],cpu_clock=lambda:0.,publisher=publisher)
        self.assertEqual(saved['completion.json']['summary']['sha256'],pins['summary.json'])
        return result,saved

    def test_terminal_includes_cleanup_publication_and_preserves_failure(self):
        result,saved=self.terminal_fixture(3600.)
        self.assertEqual(result['status'],'failed');self.assertEqual(saved['summary.json']['continuations'],{})
        self.assertFalse(saved['completion.json']['continuation_eligible'])
        result,saved=self.terminal_fixture(3599.5,summary_seconds=.5)
        self.assertEqual(saved['summary.json']['status'],'completed');self.assertEqual(result['status'],'failed')
        self.assertEqual(saved['completion.json']['wall_seconds'],3600.)
        self.assertFalse(saved['completion.json']['continuation_eligible'])
        result,saved=self.terminal_fixture(3599.,summary_seconds=.5,marker_seconds=1.)
        self.assertEqual(result['status'],'completed');self.assertEqual(result['wall_seconds'],3599.5)
        self.assertEqual(result['completion_publication_wall_seconds'],1.)
        result,saved=self.terminal_fixture(10.,status='failed')
        self.assertEqual(result['status'],'failed');self.assertFalse(saved['completion.json']['continuation_eligible'])

    def test_four_replays_preserve_identical_source_coordinates(self):
        source=manifest()['phases']['training']['curriculum'];before=deepcopy(source)
        counts={}
        for arm in r.ARMS:
            counts[arm]={}
            for index in range(2592):
                TEST_WORK['schedule_lookups']+=1;record=r.replay_record(source,index)
                self.assertIs(record,source[index%648])
                counts[arm][record['kind']]=counts[arm].get(record['kind'],0)+1
        expected=dict(basis=144,complementary=720,old_definition=432,replay=1296)
        self.assertEqual(counts,dict(control=expected,lower=expected));self.assertEqual(source,before)
        for bad in (-1,2592,True):
            with self.assertRaises(ValueError):r.replay_record(source,bad)
        with self.assertRaises(ValueError):r.replay_record(source[:-1],0)

    def test_deltas_preserve_history_with_full_auxiliary_work_in_both_arms(self):
        before=dict(work={'retained_updates':15592,'unknown_optimizer_outcomes':0},
            state_work={'completed_state_objectives':46776},cost={'seconds':100.})
        for arm in r.ARMS:
            TEST_WORK['accounting_cases']+=1
            after=dict(work={'retained_updates':18184,'unknown_optimizer_outcomes':0},
                state_work={'completed_state_objectives':54552},cost={'seconds':150.})
            delta=r.physical_deltas({arm:{'accounting':{'lifetime_kernel':after}}},{arm:{'accounting':before}})[arm]
            self.assertEqual(delta['work']['retained_updates'],2592)
            self.assertEqual(delta['state_work']['completed_state_objectives'],7776)
            self.assertEqual(delta['cost']['seconds'],50.)
        after['work']['retained_updates']=15591
        with self.assertRaises(ValueError):r.accounting_delta(after,before)

    def test_fixed_budget_endpoints_and_rate_only_recipe_difference(self):
        value=r.contract()
        self.assertEqual(value['endpoints'],[0,648,1296,2592]);self.assertEqual(value['max_seconds'],3600)
        self.assertEqual(value['auxiliary_weights'],dict(control=.3,lower=.3))
        self.assertEqual(value['learning_rates'],dict(control=.0003,lower=.0001))
        self.assertEqual(value['source_schedule'],'curriculum');self.assertEqual(r.PARENT_CURSOR+r.UPDATES,18184)
        for seconds in (1200,3599,3601,3600.,True):
            with self.assertRaises(ValueError):r.contract(seconds)
        parent={'auxiliary_weight':.3,'optimizer':[{'lr':.0003,'betas':(.9,.999),'eps':1e-8}],
            'source_sha256':{'same.py':'b'*64},'objective_id':'unchanged','micro_batch_size':32}
        original=deepcopy(parent)
        for arm,lr in (('control',.0003),('lower',.0001)):
            changed=r.expected_recipe(parent,arm);self.assertEqual(changed['optimizer'][0]['lr'],lr)
            changed['optimizer'][0]['lr']=.0003;self.assertEqual(changed,parent)
            changed['optimizer'][0]['betas']=(0.,0.);self.assertEqual(parent,original)
        for bad,arm in ((parent,'other'),({**parent,'auxiliary_weight':0.},'lower'),
                ({**parent,'optimizer':[{'lr':.0001}]},'lower'),({**parent,'optimizer':[]},'control')):
            with self.assertRaises(ValueError):r.expected_recipe(bad,arm)

    def test_parent_is_one_detached_full_control_for_both_arms(self):
        values=parents();admitted=r.parent_admission(*values)
        self.assertEqual(admitted['control'],admitted['lower'])
        self.assertEqual(admitted['control']['checkpoint']['sha256'],r.PARENT_CHECKPOINT_SHA256)
        self.assertEqual(admitted['control']['identity_sha256'],r.identity(values[1]['origin_identity']))
        self.assertEqual(set(admitted['control']['metrics']),set(r.BANK_NAMES))
        admitted['control']['metrics']['basis_binding']['source']='mutation'
        self.assertEqual(admitted['lower']['metrics']['basis_binding']['source'],'same-parent')
        self.assertEqual(values[2]['scores']['basis_binding']['metrics']['source'],'same-parent')

    def test_foreign_partial_ineligible_or_relabelled_parent_rejected(self):
        for case in ('checkpoint','branch','weights','scores','unknown','partial','incomplete','data',
                     'reference','shared_reference','previous_reference','prior_common','transition',
                     'completion_missing','completion_summary','completion_launch','ineligible','deadline','nan'):
            summary,previous,commit,completion=parents()
            if case=='checkpoint':commit['checkpoint']['sha256']='c'*64
            elif case=='branch':commit['arm']='zero'
            elif case=='weights':summary['continuations']['control']['weights_sha256']='d'*64
            elif case=='scores':commit['scores'].pop('basis_binding')
            elif case=='unknown':summary['physical_training']['zero']['work']['unknown_optimizer_outcomes']=1
            elif case=='partial':summary['partial_work_unknown']=True
            elif case=='incomplete':summary['new_updates']['control']=2591
            elif case=='data':previous['data_manifest_sha256']='bad'
            elif case=='reference':previous['reference_parent']['lifetime_updates']=9760
            elif case=='shared_reference':previous['shared_parent']['lifetime_updates']=9112
            elif case=='previous_reference':previous['previous_curriculum_parent']['lifetime_updates']=13000
            elif case=='prior_common':previous['parents']['control']['lifetime_updates']=10408
            elif case=='transition':commit['transition']['identity']['auxiliary_weight']=0.
            elif case=='completion_missing':completion={}
            elif case=='completion_summary':completion['summary']['sha256']='e'*64
            elif case=='completion_launch':completion['launch_sha256']='f'*64
            elif case=='ineligible':completion['continuation_eligible']=False
            elif case=='deadline':completion['wall_seconds']=3600.
            else:completion['wall_seconds']=float('nan')
            with self.subTest(case=case),self.assertRaises(ValueError):r.parent_admission(summary,previous,commit,completion)

    def test_launch_requires_shared_metadata_rate_contract_and_five_references(self):
        values=parents();admitted=r.parent_admission(*values);previous=values[1]
        launch=dict(**r.contract(),parents=admitted,
            parent_metadata={arm:dict(cursor=15592,recipe={'auxiliary_weight':.3,'optimizer':[{'lr':.0003}]},
                optimizer_identity='same') for arm in r.ARMS},parent_transition=values[2]['transition'],
            parent_summary_sha256=r.PARENT_SUMMARY_SHA256,parent_completion_sha256=r.PARENT_COMPLETION_SHA256,
            data_manifest_sha256=r.DATA_MANIFEST_SHA256,shared_parent=previous['shared_parent'],
            prior_common_parent=previous['parents']['control'],
            previous_curriculum_parent=previous['previous_curriculum_parent'],reference_parent=previous['reference_parent'])
        r.validate_launch(launch)
        for case in ('parent','optimizer','weight','rate','schedule','completion','endpoint','reference','transition'):
            bad=deepcopy(launch)
            if case=='parent':bad['parents']['lower']['weights_sha256']='b'*64
            elif case=='optimizer':bad['parent_metadata']['lower']['optimizer_identity']='different'
            elif case=='weight':bad['auxiliary_weights']['lower']=0.
            elif case=='rate':bad['learning_rates']['lower']=.0003
            elif case=='schedule':bad['source_schedule']='control'
            elif case=='completion':bad['parent_completion_sha256']='a'*64
            elif case=='endpoint':bad['endpoints']=[0,648,1296,1944,2592]
            elif case=='reference':bad['prior_common_parent']['lifetime_updates']=10408
            else:bad['parent_transition']['sha256']='b'*64
            with self.subTest(case=case),self.assertRaises(ValueError):r.validate_launch(bad)

    def test_complete_work_requires_full_native_and_state_counts(self):
        value=manifest();plan=r.physical_plan(value)
        self.assertEqual((plan['updates'],plan['episode_exposures'],plan['restores'],plan['snapshots']),
            (5184,497664,8,6))
        self.assertEqual(plan['new_state_readouts'],dict(control=7776,lower=7776))
        work=dict(retained_updates=2592,synchronized_optimizer_updates=2592,optimizer_attempts=2592,optimizer_returns=2592,
            unknown_optimizer_outcomes=0,retained_episodes=248832,attempted_forwards=7776,completed_forwards=7776,
            attempted_backwards=7776,completed_backwards=7776,attempted_forward_episodes=248832,completed_forward_episodes=248832,
            attempted_backward_episodes=248832,completed_backward_episodes=248832)
        state_keys=('attempted_state_readouts','completed_state_readouts','attempted_state_objectives','completed_state_objectives')
        physical={arm:dict(work=deepcopy(work),state_work=dict.fromkeys(state_keys,7776)) for arm in r.ARMS}
        receipt=dict(new_updates=dict.fromkeys(r.ARMS,2592),restorations=[{}]*8,snapshots=[{}]*6)
        r.validate_completed_work(plan,receipt,physical)
        for case in ('lower_state','control_state','episodes','unknown','restores','snapshots','updates'):
            bad,summary=deepcopy(physical),deepcopy(receipt)
            if case=='lower_state':bad['lower']['state_work'][state_keys[0]]=0
            elif case=='control_state':bad['control']['state_work'][state_keys[0]]=7773
            elif case=='episodes':bad['lower']['work']['retained_episodes']-=1
            elif case=='unknown':bad['lower']['work']['unknown_optimizer_outcomes']=1
            elif case=='restores':summary['restorations'].pop()
            elif case=='snapshots':summary['snapshots'].pop()
            else:summary['new_updates']['lower']-=1
            with self.subTest(case=case),self.assertRaises(ValueError):r.validate_completed_work(plan,summary,bad)
        value['phases']['training']['curriculum'][1]['kind']='complementary'
        with self.assertRaises(ValueError):r.physical_plan(value)

    def test_auxiliary_fingerprints_require_valid_pins_but_both_heads_may_learn(self):
        for arm in r.ARMS:
            r.validate_auxiliary('a'*64,'a'*64,arm);r.validate_auxiliary('b'*64,'a'*64,arm)
        for pin in ('a'*63,'A'*64,'g'*64,None,True):
            with self.assertRaises(ValueError):r.validate_auxiliary(pin,'a'*64,'lower')
        with self.assertRaises(ValueError):r.validate_auxiliary('a'*64,'a'*64,'foreign')

    def test_partial_work_flags_are_monotonic_and_include_nested_restore_failures(self):
        clean=dict(partial_work_unknown=False,physical_totals={'work':{'unknown_optimizer_outcomes':0}},
            replay_target_work={'partial_batch_internal_work_unknown':False},
            restorations=[{'report':{'failed':False,'partial_work_unknown':False}}],latest_arms={})
        TEST_WORK['uncertainty_cases']+=1;self.assertFalse(r.partial_work_unknown(clean))
        for case in ('prior','optimizer','oracle','restore','restore_sync','objective','legacy','snapshot','latest'):
            TEST_WORK['uncertainty_cases']+=1;value=deepcopy(clean)
            if case=='prior':value['partial_work_unknown']=True
            elif case=='optimizer':value['physical_totals']['work']['unknown_optimizer_outcomes']=1
            elif case=='oracle':value['replay_target_work']['partial_batch_internal_work_unknown']=True
            elif case=='restore':value['restorations'][0]['report']['partial_work_unknown']=True
            elif case=='restore_sync':value['restorations'][0]['report']['failure_sync_error']='synthetic sync failure'
            elif case=='objective':value['restorations'][0]['report']['objective_restore']={'partial_work_unknown':True}
            elif case=='legacy':value['restorations'][0]['report']['objective_restore']={'legacy_restore':{'failure_sync_error':'sync'}}
            elif case=='snapshot':value['restorations'][0]['report']['parent_snapshot']={'kernel_report':{'unknown_optimizer_outcomes':1}}
            else:value['latest_arms']['lower']={'last_report':{'failure_sync_error':'sync'}}
            before=deepcopy(value);self.assertTrue(r.partial_work_unknown(value));self.assertEqual(value,before)
        # A known metadata rejection is a failure, but not unknown physical work.
        TEST_WORK['uncertainty_cases']+=1
        clean['restorations'][0]['report'].update(failed=True,error='known bad pin')
        self.assertFalse(r.partial_work_unknown(clean))

    def test_numerical_proof_requires_matching_inclusive_completion_marker(self):
        proof=dict(schema='bic-shared-rate-runtime-validation-v1',status='passed',device='cpu',max_seconds=180,
            source_sha256={'experiments/shared_rate_continuation.py':'a'*64})
        marker=dict(schema=proof['schema'],status='passed',report_status='passed',device='cpu',report_sha256='b'*64,
            max_seconds=180,wall_seconds=179.9,completed_within_deadline=True)
        def check(p,m):
            TEST_WORK['proof_cases']+=1
            return r.validate_numerical_proof(p,m,report_sha256='b'*64,bridge_sha256='a'*64)
        for device in ('cpu','cuda:0'):
            p,m=deepcopy(proof),deepcopy(marker);p['device']=m['device']=device
            self.assertEqual(check(p,m),device)
        for case in ('missing','schema','failed','report_status','report_pin','device','deadline','late','nan','bool_time',
                'wrong_budget','proof_failed','proof_schema','source','proof_budget','foreign_device'):
            p,m=deepcopy(proof),deepcopy(marker)
            if case=='missing':m={}
            elif case=='schema':m['schema']='wrong'
            elif case=='failed':m['status']='failed'
            elif case=='report_status':m['report_status']='failed'
            elif case=='report_pin':m['report_sha256']='c'*64
            elif case=='device':m['device']='cuda:0'
            elif case=='deadline':m['completed_within_deadline']=False
            elif case=='late':m['wall_seconds']=180.
            elif case=='nan':m['wall_seconds']=float('nan')
            elif case=='bool_time':m['wall_seconds']=True
            elif case=='wrong_budget':m['max_seconds']=181
            elif case=='proof_failed':p['status']='failed'
            elif case=='proof_schema':p['schema']='wrong'
            elif case=='source':p['source_sha256']['experiments/shared_rate_continuation.py']='c'*64
            elif case=='proof_budget':p['max_seconds']=181
            else:p['device']=m['device']='cuda:1'
            with self.subTest(case=case),self.assertRaises(ValueError):check(p,m)


if __name__=='__main__':unittest.main()
