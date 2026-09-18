"""Pure admission, evidence reuse and physical accounting; no learner work."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import definition_tutor_pair as worker

WORK=dict(tests=0,contract_checks=0,parent_checks=0,binding_checks=0,plan_checks=0,
    sharing_checks=0,accounting_checks=0,run_preflight_calls=0,models=0,archives=0,generation=0,teacher_calls=0)


def manifest(same=False):
    return dict(schema='bic-definition-tutor-pair-data-v1',updates_per_arm=216,teaching_updates=108,
        withdrawal_updates=108,micro_batch_size=32,parent=dict(identity_sha256='a'*64,weights_sha256='b'*64,lifetime_updates=9760),
        teacher_decision=dict(path='synthetic-author/result.json',sha256='c'*64,request_sha256='d'*64),
        phases=dict(teaching={a:[dict(index=i) for i in range(108)] for a in worker.ARMS},withdrawal=[dict(index=i) for i in range(108)]),
        teaching_learning_stream_sha256=dict(procedural='e'*64,tutor=('e' if same else 'f')*64),
        zero_treatment_contrast=same,different_teaching_positions=0 if same else 72)


def parent_fixture():
    record=dict(path='synthetic/checkpoint.pt',sha256=worker.PARENT_CHECKPOINT_SHA256)
    scores={name:dict(metrics={'count':1}) for name in worker.BANK_NAMES}
    previous=dict(parent=dict(lifetime_updates=9112),origin_identity=dict(schema='original',rate=.0003,auxiliary_weight=.3))
    summary=dict(schema='bic-definition-basis-learning-v1',status='completed',partial_work_unknown=False,updates=648,
        lifetime_updates=9760,launch_sha256=worker.PARENT_LAUNCH_SHA256,endpoint=record,
        physical_training=dict(work=dict(unknown_optimizer_outcomes=0)),evaluations={'648':scores})
    commit=dict(checkpoint=deepcopy(record),relative_step=648,lifetime_updates=9760,launch_sha256=worker.PARENT_LAUNCH_SHA256,
        weights_sha256='b'*64,scores=deepcopy(scores))
    return summary,previous,commit


def physical_work():
    return dict(work=dict(retained_updates=216,synchronized_optimizer_updates=216,optimizer_attempts=216,optimizer_returns=216,
        unknown_optimizer_outcomes=0,retained_episodes=20736,attempted_forwards=648,completed_forwards=648,
        attempted_backwards=648,completed_backwards=648,attempted_forward_episodes=20736,completed_forward_episodes=20736,
        attempted_backward_episodes=20736,completed_backward_episodes=20736),
        state_work=dict.fromkeys(('attempted_state_readouts','completed_state_readouts','attempted_state_objectives','completed_state_objectives'),648),cost=dict(step_seconds=2.))


class DefinitionTutorPairTests(unittest.TestCase):
    def setUp(self):WORK['tests']+=1

    def test_total_allowance_and_run_preflight_before_execution(self):
        for value in (899,901,1800,900.,True,float('inf')):
            WORK['contract_checks']+=1
            with self.assertRaises(ValueError):worker.contract(value)
        with tempfile.TemporaryDirectory() as folder:
            launch=dict(worker.contract(),parent=dict(lifetime_updates=9760,checkpoint=dict(sha256=worker.PARENT_CHECKPOINT_SHA256)),
                parent_summary_sha256=worker.PARENT_SUMMARY_SHA256,reference_parent=dict(lifetime_updates=9112))
            for change in (dict(max_seconds=1800),dict(endpoints=[0,216]),dict(teacher_calls=1),dict(automatic_retry=True)):
                with patch.object(worker,'digest',return_value='a'*64),patch.object(worker,'read',return_value={**launch,**change}):
                    WORK['run_preflight_calls']+=1
                    with self.assertRaises(ValueError):worker.run(folder,launch_sha256='a'*64)
                self.assertFalse((Path(folder)/'execution').exists())

    def test_exact_unadopted_parent_and_all_bank_scores_are_mandatory(self):
        summary,previous,commit=parent_fixture();WORK['parent_checks']+=1
        actual=worker.parent_admission(summary,previous,commit)
        self.assertEqual(actual['identity_sha256'],worker.identity(previous['origin_identity']))
        self.assertEqual(actual['checkpoint']['sha256'],worker.PARENT_CHECKPOINT_SHA256)
        for group,key,value in (('summary','status','failed'),('summary','partial_work_unknown',True),
                ('summary','lifetime_updates',9112),('summary','updates',647),('commit','relative_step',432),
                ('commit','scores',dict(list(commit['scores'].items())[:-1]))):
            left,right=deepcopy(summary),deepcopy(commit);(left if group=='summary' else right)[key]=value
            WORK['parent_checks']+=1
            with self.subTest(key=key),self.assertRaises(ValueError):worker.parent_admission(left,previous,right)

    def test_teacher_origin_identity_is_distinct_from_current_checkpoint(self):
        parent=worker.parent_admission(*parent_fixture());data=manifest()
        data['parent']['identity_sha256']=parent['identity_sha256']
        teacher=dict(request_sha256='d'*64,outcome='local_accepted',parent={**data['parent'],'cycle':1},
            teacher_cost=dict(physical_teacher_work_unknown=False))
        WORK['binding_checks']+=1
        self.assertEqual(worker.validate_bindings(data,parent,teacher)['updates'],432)
        self.assertNotEqual(parent['identity_sha256'],parent['checkpoint']['sha256'])
        for key,value in (('identity_sha256',parent['checkpoint']['sha256']),('weights_sha256','0'*64),('lifetime_updates',9112)):
            bad=deepcopy(teacher);bad['parent'][key]=value;WORK['binding_checks']+=1
            with self.subTest(key=key),self.assertRaises(ValueError):worker.validate_bindings(data,parent,bad)
        bad=deepcopy(teacher);bad['teacher_cost']['physical_teacher_work_unknown']=True;WORK['binding_checks']+=1
        with self.assertRaises(ValueError):worker.validate_bindings(data,parent,bad)

    def test_stream_difference_and_common_withdrawal_control_physical_plan(self):
        for same,arms,updates in ((False,['procedural','tutor'],432),(True,['procedural'],216)):
            WORK['plan_checks']+=1;plan=worker.physical_plan(manifest(same))
            self.assertEqual(plan['physical_arms'],arms);self.assertEqual(plan['updates'],updates)
            self.assertEqual(plan['restores'],3*len(arms));self.assertEqual(plan['snapshots'],2*len(arms))
        for change in ('wrong_zero','missing_withdrawal','partial_tutor','wrong_micro','wrong_changed'):
            value=manifest()
            if change=='wrong_zero':value['zero_treatment_contrast']=True
            elif change=='missing_withdrawal':value['phases']['withdrawal'].pop()
            elif change=='partial_tutor':value['phases']['teaching']['tutor'].pop()
            elif change=='wrong_micro':value['micro_batch_size']=16
            else:value['different_teaching_positions']=0
            WORK['plan_checks']+=1
            with self.subTest(change=change),self.assertRaises(ValueError):worker.physical_plan(value)

    def test_identical_evidence_reuse_does_not_invent_updates_or_checkpoint(self):
        plan=worker.physical_plan(manifest(True))
        receipt=dict(new_updates=dict(procedural=216,tutor=0),evaluations={'procedural':{str(n):{'raw':'actual'} for n in (0,108,216)}},
            endpoint_commits={'procedural':{'216':dict(path='procedural-commit.json',sha256='a'*64)}},
            continuations={'procedural':dict(checkpoint=dict(path='procedural.pt',sha256='b'*64),shared_result_of=None)})
        WORK['sharing_checks']+=1;worker.share_identical_result(receipt,plan)
        self.assertEqual(receipt['new_updates'],dict(procedural=216,tutor=0))
        self.assertEqual(receipt['continuations']['tutor']['checkpoint'],receipt['continuations']['procedural']['checkpoint'])
        self.assertEqual(receipt['continuations']['tutor']['shared_result_of'],'procedural')
        receipt['evaluations']['tutor']['0']['raw']='changed-copy'
        self.assertEqual(receipt['evaluations']['procedural']['0']['raw'],'actual')
        receipt['new_updates']['tutor']=216;WORK['sharing_checks']+=1
        with self.assertRaises(ValueError):worker.share_identical_result(receipt,plan)

    def test_lifetime_deltas_and_completion_require_all_actual_work(self):
        parent=physical_work();current=deepcopy(parent)
        for group in current:
            for key in current[group]:current[group][key]+=parent[group][key]
        WORK['accounting_checks']+=1
        self.assertEqual(worker.accounting_delta(current,parent),parent)
        current['work']['retained_updates']=0;WORK['accounting_checks']+=1
        with self.assertRaises(ValueError):worker.accounting_delta(current,parent)
        plan=worker.physical_plan(manifest(False));receipt=dict(new_updates=dict.fromkeys(worker.ARMS,216),restorations=[{}]*6,snapshots=[{}]*4)
        actual={a:physical_work() for a in worker.ARMS};WORK['accounting_checks']+=1;worker.validate_completed_work(plan,receipt,actual)
        for group,key,value in (('work','unknown_optimizer_outcomes',1),('work','completed_forwards',647),
                ('work','retained_episodes',20735),('state_work','completed_state_readouts',647)):
            altered=deepcopy(actual);altered['tutor'][group][key]=value;WORK['accounting_checks']+=1
            with self.subTest(key=key),self.assertRaises(ValueError):worker.validate_completed_work(plan,receipt,altered)


if __name__=='__main__':unittest.main()
