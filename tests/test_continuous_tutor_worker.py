"""Pure authenticated-boundary checks; no learner, archives, or lesson generation."""
from collections import Counter
from copy import deepcopy
import unittest
from experiments import continuous_tutor_worker as worker

WORK=Counter()


def call(name,*args,**kwargs):
    WORK['direct_'+name+'_calls']+=1
    return getattr(worker,name)(*args,**kwargs)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        WORK['test_methods']+=1

    def fixture(self,cursor=9876,cycle=3):
        WORK['synthetic_parent_data_envelopes']+=1
        spec=worker.policy.cycle_spec(853000001,cycle)
        recipe=dict(config=worker.CONFIG,architecture=spec['architecture'],layout='original',
            micro_batch_size=32,objective_id=spec['objective_id'],optimizer=[{'lr':.0003}])
        origin=dict(step=648,runtime={'strict':'synthetic'},recipe=recipe,config=worker.CONFIG,
            architecture=spec['architecture'],learning_rate=.0003,auxiliary_weight=.3)
        checkpoint=dict(path='synthetic/parent.pt',sha256='a'*64)
        metadata=dict(origin=origin,cursor=cursor,weights_sha256='b'*64,recipe=recipe)
        parent=dict(identity_sha256='a'*64,weights_sha256='b'*64,cycle=cycle,lifetime_updates=cursor)
        phases={}
        for index,phase in enumerate(worker.PHASES):
            records=[dict(path=f'{phase}/{i}.pt',sha256='c'*64,cursor=cursor+108*index+i) for i in range(108)]
            phases[phase]={arm:deepcopy(records) for arm in worker.ARMS}
        manifest=dict(schema='bic-verified-tutor-cycle-data-v1',parent=parent,start_cursor=cursor,
            micro_batch_size=32,updates_per_phase=108,phase_seeds={p:spec['seeds'][p] for p in worker.PHASES},
            seed=spec['seeds']['teaching'],phases=phases)
        evaluation=dict(bank_inventory={k:dict(episodes=v) for k,v in worker.BANK_COUNTS.items()})
        kwargs=dict(parent_checkpoint=checkpoint,parent_metadata=metadata,origin_identity=origin,
                    parent_runtime=origin['runtime'],cycle_spec=spec)
        return manifest,evaluation,kwargs

    def test_01_arbitrary_completed_cursor_and_detached_binding(self):
        for cursor in (649,2160,9876):
            manifest,evaluation,kwargs=self.fixture(cursor)
            parent=call('validate_inputs',manifest,evaluation,**kwargs)
            self.assertEqual(parent['lifetime_updates'],cursor)
            parent['weights_sha256']='d'*64
            self.assertEqual(manifest['parent']['weights_sha256'],'b'*64)

    def test_02_foreign_parent_cycle_or_seed_rejected(self):
        for key,value in (('identity_sha256','d'*64),('weights_sha256','e'*64),('cycle',4),('lifetime_updates',9877)):
            manifest,evaluation,kwargs=self.fixture()
            manifest['parent'][key]=value;WORK['malformed_cases']+=1
            with self.assertRaises(ValueError):call('validate_inputs',manifest,evaluation,**kwargs)
        manifest,evaluation,kwargs=self.fixture()
        manifest['phase_seeds']['withdrawal']+=1;WORK['malformed_cases']+=1
        with self.assertRaises(ValueError):call('validate_inputs',manifest,evaluation,**kwargs)

    def test_03_order_and_common_withdrawal_cannot_be_relabelled(self):
        for mutation in ('order','withdrawal','missing'):
            manifest,evaluation,kwargs=self.fixture()
            if mutation=='order':manifest['phases']['teaching']['tutor'][0]['cursor']+=1
            elif mutation=='withdrawal':manifest['phases']['withdrawal']['tutor'][0]['sha256']='d'*64
            else:manifest['phases']['teaching']['tutor'].pop()
            WORK['malformed_cases']+=1
            with self.assertRaises(ValueError):call('validate_inputs',manifest,evaluation,**kwargs)

    def test_04_native_bank_and_original_recipe_boundaries(self):
        for mutation in ('bank','rate','runtime','spec'):
            manifest,evaluation,kwargs=self.fixture()
            if mutation=='bank':evaluation['bank_inventory']['transfer_varied']['episodes']-=2
            elif mutation=='rate':kwargs['origin_identity']['learning_rate']=.003
            elif mutation=='runtime':kwargs['parent_runtime']={'strict':'other'}
            else:kwargs['cycle_spec']['teaching_updates']=107
            WORK['malformed_cases']+=1
            with self.assertRaises(ValueError):call('validate_inputs',manifest,evaluation,**kwargs)

    def test_05_own_envelope_metadata_preserves_origin(self):
        manifest,_,kwargs=self.fixture()
        payload=dict(schema='bic-shared-state-continuation-v1',identity=dict(
            schema='bic-shared-state-continuation-v1',origin=kwargs['origin_identity'],source_sha256={}),
            learner=dict(schema='bic-shared-state-training-v1',cursor=9876,evidence={'cursor':9876},
                recipe=kwargs['origin_identity']['recipe'],accounting={'work':{'unknown_optimizer_outcomes':0,'retained_updates':9876}}),
            lifetime_updates=9876,continuation_updates=9876-648,weights_sha256='b'*64)
        WORK['synthetic_checkpoint_metadata_envelopes']+=1
        result=call('parent_metadata',payload)
        self.assertEqual(result['origin'],kwargs['origin_identity'])
        for mutation in ('outer','cursor','unknown'):
            damaged=deepcopy(payload);WORK['malformed_cases']+=1
            if mutation=='outer':damaged['schema']='bic-shared-state-rate-pilot-v1'
            elif mutation=='cursor':damaged['learner']['evidence']['cursor']-=1
            else:damaged['learner']['accounting']['work']['unknown_optimizer_outcomes']=1
            with self.assertRaises(ValueError):call('parent_metadata',damaged)

    def test_06_real_phase_deltas_not_planned_totals(self):
        before={'work':dict(synchronized_optimizer_updates=9000,retained_episodes=864000,unknown_optimizer_outcomes=0)}
        evidence=lambda n:{'exposures':{f:{'episodes':n} for f in worker.policy.FAMILIES}}
        for updates,unknown in ((108,0),(7,0),(7,1)):
            after={'work':dict(synchronized_optimizer_updates=9000+updates,
                retained_episodes=864000+96*updates,unknown_optimizer_outcomes=unknown)}
            value=call('phase_work',before,after,evidence(288000),evidence(288000+32*updates))
            self.assertEqual(value,dict(synchronized_optimizer_updates=updates,retained_episodes=96*updates,
                unknown_optimizer_outcomes=unknown,family_episode_exposures={f:32*updates for f in worker.policy.FAMILIES}))
        bad=deepcopy(before);bad['work']['retained_episodes']-=1
        with self.assertRaises(ValueError):call('phase_work',before,bad,evidence(0),evidence(0))

    def test_07_absolute_deadline_counts_freeze_and_cannot_extend(self):
        self.assertEqual(call('effective_deadline',200,600,650),650)
        self.assertEqual(call('effective_deadline',200,600,1000),800)
        self.assertEqual(call('effective_deadline',200,600,100),100)
        self.assertEqual(call('effective_deadline',200,600),800)
        for invalid in (float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):call('effective_deadline',200,600,invalid)
        for invalid in (0,601,float('inf'),True):
            with self.assertRaises(ValueError):call('contract',invalid)

    def test_08_fixed_complete_work_and_native_roles(self):
        declared=call('contract',600)
        self.assertEqual(declared['new_updates'],432)
        self.assertEqual(declared['new_episode_exposures'],41472)
        self.assertEqual(sum(declared['bank_counts'].values()),2340)
        self.assertEqual(declared['evaluation_steps'],[0,108,216])
        self.assertEqual(declared['worker_teacher_calls'],0)
        self.assertFalse(declared['automatic_retry']);self.assertFalse(declared['automatic_promotion'])


if __name__=='__main__':unittest.main()
