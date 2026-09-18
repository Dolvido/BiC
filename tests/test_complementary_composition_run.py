"""Pure boundary proofs; no model construction or synthetic training."""
from copy import deepcopy
import unittest

from experiments import complementary_composition_run as run


def manifest():
    control=[];candidate=[]
    for group in range(108):
        control.extend({'kind':k} for k in ('basis','replay','old_definition','replay','basis','replay'))
        first='basis' if (2*group)%6==0 else 'complementary'
        second='basis' if (2*group+1)%6==0 else 'complementary'
        candidate.extend({'kind':k} for k in (first,'replay','old_definition','replay',second,'replay'))
    return dict(schema='bic-complementary-composition-data-v1',updates_per_arm=648,micro_batch_size=32,
        parent=dict(identity_sha256='a'*64,weights_sha256='b'*64,lifetime_updates=9760),
        parent_checkpoint=dict(path='checkpoint.pt',sha256=run.PARENT_CHECKPOINT_SHA256),
        phases=dict(training=dict(control=control,curriculum=candidate)))


class BoundaryTests(unittest.TestCase):
    def test_one_total_allowance(self):
        self.assertEqual(run.contract()['max_seconds'],1200)
        self.assertEqual(run.contract()['endpoints'],[0,216,432,648])
        for value in (900,1201,2400,1200.0,True):
            with self.assertRaises(ValueError):run.contract(value)

    def test_schedule_rejects_displaced_rehearsal(self):
        value=manifest();plan=run.physical_plan(value)
        self.assertEqual((plan['updates'],plan['episode_exposures'],plan['restores'],plan['snapshots']),
            (1296,124416,8,6))
        value['phases']['training']['curriculum'][1]['kind']='complementary'
        with self.assertRaises(ValueError):run.physical_plan(value)

    def test_incomplete_branch_and_unplanned_atomic_change(self):
        value=manifest();value['phases']['training']['control'].pop()
        with self.assertRaises(ValueError):run.physical_plan(value)
        value=manifest();value['phases']['training']['curriculum'][0]['kind']='complementary'
        with self.assertRaises(ValueError):run.physical_plan(value)

    def test_origin_weights_and_full_checkpoint_are_distinct_bindings(self):
        value=manifest();parent={**value['parent'],'checkpoint':value['parent_checkpoint']}
        run.validate_bindings(value,parent)
        wrong=deepcopy(value);wrong['parent']['identity_sha256']=run.PARENT_CHECKPOINT_SHA256
        with self.assertRaises(ValueError):run.validate_bindings(wrong,parent)
        wrong=deepcopy(value);wrong['parent_checkpoint']['sha256']='c'*64
        with self.assertRaises(ValueError):run.validate_bindings(wrong,parent)
        wrong=deepcopy(value);wrong['parent']['weights_sha256']='d'*64
        with self.assertRaises(ValueError):run.validate_bindings(wrong,parent)

    def test_parent_admission_keeps_twelve_existing_banks(self):
        checkpoint=dict(path='checkpoint.pt',sha256=run.PARENT_CHECKPOINT_SHA256)
        scores={name:dict(metrics={'source':'existing'}) for name in run.BASE_BANK_NAMES}
        origin={'recipe':'fixture'}
        previous=dict(parent=dict(lifetime_updates=9112),origin_identity=origin)
        summary=dict(schema='bic-definition-basis-learning-v1',status='completed',partial_work_unknown=False,
            updates=648,lifetime_updates=9760,launch_sha256=run.PARENT_LAUNCH_SHA256,
            physical_training={'work':{'unknown_optimizer_outcomes':0}},endpoint=checkpoint,evaluations={'648':scores})
        commit=dict(checkpoint=checkpoint,relative_step=648,lifetime_updates=9760,
            launch_sha256=run.PARENT_LAUNCH_SHA256,scores=scores,weights_sha256='b'*64)
        actual=run.parent_admission(summary,previous,commit)
        self.assertEqual(actual['identity_sha256'],run.identity(origin))
        self.assertEqual(len(actual['metrics']),12)
        commit['scores']=dict(scores,complementary_dev_binding={'metrics':{}})
        with self.assertRaises(ValueError):run.parent_admission(summary,previous,commit)

    def test_lifetime_work_does_not_turn_history_into_new_updates(self):
        before={'work':{'retained_updates':9760,'unknown_optimizer_outcomes':0},
            'state_work':{'completed_state_objectives':29280},'cost':{'seconds':1000.}}
        after={'work':{'retained_updates':10408,'unknown_optimizer_outcomes':0},
            'state_work':{'completed_state_objectives':31224},'cost':{'seconds':1234.}}
        delta=run.accounting_delta(after,before)
        self.assertEqual(delta['work']['retained_updates'],648)
        self.assertEqual(delta['state_work']['completed_state_objectives'],1944)
        after['work']['retained_updates']=9759
        with self.assertRaises(ValueError):run.accounting_delta(after,before)


if __name__=='__main__':unittest.main()
