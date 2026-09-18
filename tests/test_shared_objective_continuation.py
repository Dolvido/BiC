"""Synthetic metadata only; importing this module must not import Torch."""
from copy import deepcopy
import unittest
from experiments import shared_objective_continuation as b


def saved(cursor,weight=.3,start=10):
    work=dict.fromkeys(b.WORK_COUNTS,0)
    for key in ('step_invocations','optimizer_attempts','optimizer_returns','synchronized_optimizer_updates','retained_updates'):work[key]=cursor
    for key in ('attempted_forwards','completed_forwards','attempted_objectives','completed_objectives','attempted_backwards','completed_backwards'):work[key]=3*cursor
    for key in ('attempted_forward_episodes','completed_forward_episodes','attempted_backward_episodes','completed_backward_episodes','retained_episodes'):work[key]=6*cursor
    evidence=dict(cursor=cursor,consumed_bundle_identity_sha256='a'*64,consumed_common_parent_identity_sha256='b'*64,
        exposures={f:dict(episodes=2*cursor,turns=16*cursor,observation_tokens=132*cursor,observation_bytes=100*cursor,
            reply_target_tokens=66*cursor,reply_target_bytes=50*cursor) for f in b.FAMILIES})
    recipe=dict(schema=b.KERNEL_SCHEMA,auxiliary_weight=weight,micro_batch_size=2,optimizer=[dict(lr=.0003,betas=[.9,.999])])
    return dict(schema=b.KERNEL_SCHEMA,recipe=recipe,weights=None,optimizer=None,cursor=cursor,evidence=evidence,
        accounting=dict(work=work,state_work=b.expected_state_work(start,cursor,weight),cost=dict(step_wall_seconds=float(cursor),step_cpu_seconds=float(cursor))))


def transition(weight):
    start=saved(10);old=start['recipe'];origin=dict(step=4,recipe=deepcopy(old),auxiliary_weight=.3,learning_rate=.0003)
    legacy=dict(schema=b.LEGACY_SCHEMA,origin_archive_sha256='d'*64,origin=origin,source_sha256={'legacy.py':'e'*64},scope='synthetic legacy')
    costs={f'{kind}_{suffix}':0 if suffix=='invocations' else 0. for kind in ('restore','step','snapshot') for suffix in ('invocations','wall_seconds','cpu_seconds')}
    costs['step_invocations']=6
    return dict(schema=b.TRANSITION_SCHEMA,parent_archive_sha256='c'*64,legacy_identity=legacy,legacy_recipe=deepcopy(old),
        legacy_bridge_cost=costs,start_cursor=10,start_weights_sha256='f'*64,start_evidence=deepcopy(start['evidence']),
        start_accounting=deepcopy(start['accounting']),auxiliary_weight=weight,recipe={**deepcopy(old),'auxiliary_weight':weight},
        source_sha256={'new.py':'1'*64},scope=b.SCOPE)


class ObjectiveMetadataTests(unittest.TestCase):
    def test_failure_exception_includes_final_cost_and_poisoned_state(self):
        class FailingKernel:
            cursor=10;failed=False;last_report={'physical_optimizer_updates':0}
            def step(self,*args,**kwargs):raise ValueError('injected metadata-only step failure')
            def snapshot(self):raise ValueError('injected metadata-only snapshot failure')
        for operation in ('step','snapshot'):
            obj=object.__new__(b.SharedObjectiveContinuation);obj._kernel=FailingKernel();obj._failed=False
            obj._cost=dict.fromkeys(b.COST_KEYS,0);obj._guard=lambda:None
            with self.assertRaises(ValueError) as caught:
                if operation=='step':obj.step(None,state_targets={})
                else:obj.snapshot()
            report=caught.exception.objective_report
            self.assertEqual(report,obj.last_step_report if operation=='step' else obj.last_snapshot_report)
            self.assertEqual(report['cursor'],10);self.assertTrue(report['failed']);self.assertTrue(obj.failed)
            self.assertEqual(obj._cost[operation+'_invocations'],1)
            self.assertGreaterEqual(report['wall_seconds'],0);self.assertGreaterEqual(report['cpu_seconds'],0)

    def test_only_explicit_zero_or_point_three_weights(self):
        for value in (0,0.,.3):self.assertEqual(b.auxiliary_weight(value),float(value))
        for value in (True,False,None,'0',1.,float('nan'),float('inf'),-.3):
            with self.subTest(value=value),self.assertRaises(ValueError):b.auxiliary_weight(value)

    def test_declared_recipe_change_preserves_every_other_field(self):
        for weight in (0.,.3):
            t=transition(weight);self.assertEqual(b.validate_transition(t,b.identity(t)),t)
            for field,value in (('micro_batch_size',4),('optimizer',[dict(lr=.003,betas=[.9,.999])]),('extra','bad')):
                bad=deepcopy(t);bad['recipe'][field]=value
                with self.assertRaises(ValueError):b.validate_transition(bad,b.identity(bad))

    def test_complete_post_transition_state_counts_are_actual_not_lifetime_inferred(self):
        for weight,expected in ((0.,30),(.3,45)):
            t=transition(weight);value=saved(15,weight);b.validate_metadata(value,t)
            self.assertEqual(set(value['accounting']['state_work'].values()),{expected})
            for key in b.STATE_COUNTS:
                bad=deepcopy(value);bad['accounting']['state_work'][key]+=3
                with self.assertRaises(ValueError):b.validate_metadata(bad,t)
        with self.assertRaises(ValueError):b.expected_state_work(10,9,0.)

    def test_adamw_active_base_steps_and_frozen_auxiliary_steps(self):
        for name in ('state_alias_embedding.weight','state_decoder.0.weight','state_decoder.2.bias'):
            self.assertEqual(b.expected_parameter_step(name,10,15,0.),10)
            self.assertEqual(b.expected_parameter_step(name,10,15,.3),15)
        for name in ('tokens.weight','blocks.0.attention.in_proj_weight','action_head.bias','decoder.weight'):
            self.assertEqual(b.expected_parameter_step(name,10,15,0.),15)

    def test_boundary_history_cost_and_exposure_are_retained(self):
        t=transition(0.);value=saved(10,0.);b.validate_metadata(value,t)
        for change in ('evidence','cost','exposure'):
            bad=saved(10 if change=='evidence' else 15,0.)
            if change=='evidence':bad['evidence']['consumed_bundle_identity_sha256']='2'*64
            elif change=='cost':bad['accounting']['cost']['step_wall_seconds']=9.
            else:
                for counts in bad['evidence']['exposures'].values():
                    counts['observation_bytes']=1;counts['observation_tokens']=1+2*counts['turns']
            with self.assertRaises(ValueError):b.validate_metadata(bad,t)

    def test_foreign_pin_schema_and_legacy_history_refused(self):
        t=transition(0.)
        with self.assertRaises(ValueError):b.validate_transition(t,'0'*64)
        for change in ('schema','legacy_weight','bridge_steps','source','start'):
            bad=deepcopy(t)
            if change=='schema':bad['schema']=b.LEGACY_SCHEMA
            elif change=='legacy_weight':bad['legacy_recipe']['auxiliary_weight']=0.
            elif change=='bridge_steps':bad['legacy_bridge_cost']['step_invocations']=5
            elif change=='source':bad['source_sha256']['new.py']='bad'
            else:bad['start_cursor']=9
            with self.assertRaises(ValueError):b.validate_transition(bad,b.identity(bad))

    def test_unknown_incomplete_and_malformed_lifetimes_refused(self):
        t=transition(.3)
        for change in ('unknown','optimizer','episodes','family','boolean','state_fields'):
            bad=saved(15)
            if change=='unknown':bad['accounting']['work']['unknown_optimizer_outcomes']=1
            elif change=='optimizer':bad['accounting']['work']['optimizer_returns']=14
            elif change=='episodes':bad['evidence']['exposures']['count']['episodes']=1
            elif change=='family':del bad['evidence']['exposures']['switch']
            elif change=='boolean':bad['accounting']['work']['packed_episodes']=False
            else:bad['accounting']['state_work']['fabricated']=0
            with self.assertRaises(ValueError):b.validate_metadata(bad,t)

    def test_json_roundtrip_and_detached_transition(self):
        t=transition(.3);t['recipe']['optimizer'][0]['betas']=(.9,.999)
        roundtrip=b._normal(t);self.assertEqual(b.identity(t),b.identity(roundtrip))
        result=b.validate_transition(roundtrip,b.identity(t));result['start_evidence']['exposures']['color']['episodes']=0
        self.assertEqual(roundtrip['start_evidence']['exposures']['color']['episodes'],20)


if __name__=='__main__':unittest.main()
