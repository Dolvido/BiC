"""Pure synthetic metadata admission; no tensors, archives or learner imports."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from experiments import shared_rate_continuation as rate
from experiments import shared_objective_continuation as objective


def costs(steps, *, transition=True):
    result = {f'{op}_{unit}': 0 if unit == 'invocations' else 0.
        for op in ('transition', 'restore', 'step', 'snapshot') for unit in ('invocations', 'wall_seconds', 'cpu_seconds')}
    result.update(step_invocations=steps, transition_invocations=int(transition), snapshot_invocations=1)
    return result


def saved(cursor, lr=.0003):
    work = dict(step_invocations=cursor, packed_microbatches=0, packed_episodes=0,
        attempted_forwards=3*cursor, completed_forwards=3*cursor, attempted_forward_episodes=6*cursor,
        completed_forward_episodes=6*cursor, attempted_objectives=3*cursor, completed_objectives=3*cursor,
        attempted_backwards=3*cursor, completed_backwards=3*cursor, attempted_backward_episodes=6*cursor,
        completed_backward_episodes=6*cursor, optimizer_attempts=cursor, optimizer_returns=cursor,
        synchronized_optimizer_updates=cursor, unknown_optimizer_outcomes=0, retained_updates=cursor, retained_episodes=6*cursor)
    state = dict(attempted_state_readouts=3*cursor, completed_state_readouts=3*cursor,
        attempted_state_objectives=3*cursor, completed_state_objectives=3*cursor)
    evidence = dict(cursor=cursor, consumed_bundle_identity_sha256='a'*64,
        consumed_common_parent_identity_sha256='b'*64, exposures={family:dict(episodes=2*cursor, turns=16*cursor,
        observation_bytes=100*cursor, observation_tokens=132*cursor, reply_target_bytes=50*cursor,
        reply_target_tokens=66*cursor) for family in ('color', 'count', 'switch')})
    recipe = dict(schema=objective.KERNEL_SCHEMA, architecture='sequence-shared-entity-state-v1',
        auxiliary_weight=.3, micro_batch_size=2, layout='original', gradient_clip=1.,
        optimizer=[dict(lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=.01, params=[0, 1], amsgrad=False)])
    return dict(schema=objective.KERNEL_SCHEMA, recipe=recipe, weights=None, optimizer=None, cursor=cursor,
        evidence=evidence, accounting=dict(work=work, state_work=state,
        cost=dict(step_wall_seconds=float(cursor), step_cpu_seconds=float(cursor))))


def parent():
    start = saved(10)
    recipe = deepcopy(start['recipe'])
    legacy_cost = {f'{op}_{unit}': 0 if unit == 'invocations' else 0.
        for op in ('restore', 'step', 'snapshot') for unit in ('invocations', 'wall_seconds', 'cpu_seconds')}
    legacy_cost['step_invocations'] = 6
    transition = dict(schema=objective.TRANSITION_SCHEMA, parent_archive_sha256='c'*64,
        legacy_identity=dict(schema=objective.LEGACY_SCHEMA, origin_archive_sha256='d'*64,
            origin=dict(step=4, recipe=deepcopy(recipe), auxiliary_weight=.3, learning_rate=.0003),
            source_sha256={'old.py':'e'*64}, scope='synthetic parent'), legacy_recipe=deepcopy(recipe),
        legacy_bridge_cost=legacy_cost, start_cursor=10, start_weights_sha256='f'*64,
        start_evidence=deepcopy(start['evidence']), start_accounting=deepcopy(start['accounting']),
        auxiliary_weight=.3, recipe=recipe, source_sha256={'objective.py':'1'*64}, scope=objective.SCOPE)
    return dict(schema=objective.SCHEMA, transition=transition, transition_sha256=objective.identity(transition),
        learner=saved(15), weights_sha256='2'*64, lifetime_updates=15, objective_updates=5, cost=costs(5))


def transition(lr=.0001):
    p = parent(); s = p['learner']
    changed = deepcopy(s['recipe']); changed['optimizer'][0]['lr'] = lr
    return dict(schema=rate.TRANSITION_SCHEMA, parent_archive_sha256='3'*64,
        parent_transition=p['transition'], parent_transition_sha256=p['transition_sha256'],
        parent_recipe=s['recipe'], parent_bridge_cost=p['cost'], start_cursor=15,
        start_weights_sha256=p['weights_sha256'], start_evidence=s['evidence'], start_accounting=s['accounting'],
        source_learning_rate=.0003, learning_rate=lr, recipe=changed,
        source_sha256={'rate.py':'4'*64}, scope=rate.SCOPE)


def envelope(cursor=17, lr=.0001):
    t = transition(lr)
    return dict(schema=rate.SCHEMA, transition=t, transition_sha256=rate.identity(t), learner=saved(cursor, lr),
        weights_sha256='2'*64 if cursor == 15 else '5'*64, lifetime_updates=cursor,
        rate_updates=cursor-15, cost=costs(cursor-15))


class RateMetadataTests(unittest.TestCase):
    def test_supported_rate_and_control_boundary(self):
        for lr in (.0003, .0001):
            t = transition(lr)
            self.assertEqual(rate.validate_transition(t, rate.identity(t)), t)
            rate.validate_snapshot(envelope(15, lr), rate.identity(t))
        for value in (None, True, 0, .003, 0., float('nan'), float('inf'), '.0001'):
            with self.subTest(value=value), self.assertRaises(ValueError): rate.learning_rate(value)

    def test_rehashed_optimizer_and_objective_drift_is_not_authorized(self):
        for field, value in (('betas', (.8, .999)), ('weight_decay', 0.), ('eps', 1e-5),
                ('params', [1, 0]), ('amsgrad', True), ('lr', .003)):
            bad = transition(); bad['recipe']['optimizer'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): rate.validate_transition(bad, rate.identity(bad))
        for field, value in (('auxiliary_weight', 0.), ('micro_batch_size', 4), ('layout', 'varied'),
                ('gradient_clip', .5), ('architecture', 'another-model')):
            bad = transition(); bad['recipe'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): rate.validate_transition(bad, rate.identity(bad))
        bad = transition(); bad['recipe']['optimizer'].append(deepcopy(bad['recipe']['optimizer'][0]))
        with self.assertRaises(ValueError): rate.validate_transition(bad, rate.identity(bad))

    def test_parent_must_be_completed_pinned_objective_control(self):
        p = parent()
        rate.validate_parent(p, p['transition_sha256'])
        with self.assertRaises(ValueError): rate.validate_parent(p, '0'*64)
        for mutation in ('legacy_schema', 'zero_objective', 'no_updates', 'wrong_lifetime', 'wrong_cost', 'no_snapshot'):
            bad = deepcopy(p)
            if mutation == 'legacy_schema': bad['schema'] = objective.LEGACY_SCHEMA
            elif mutation == 'zero_objective':
                bad['transition']['auxiliary_weight'] = 0.; bad['transition']['recipe']['auxiliary_weight'] = 0.
                bad['transition_sha256'] = objective.identity(bad['transition'])
            elif mutation == 'no_updates': bad['objective_updates'] = 0
            elif mutation == 'wrong_lifetime': bad['lifetime_updates'] = 16
            elif mutation == 'wrong_cost': bad['cost']['step_invocations'] = 4
            else: bad['cost']['snapshot_invocations'] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): rate.validate_parent(bad, bad['transition_sha256'])

    def test_rate_transition_cannot_relabel_origin_or_parent_recipe(self):
        for mutation in ('original_lr', 'parent_pin', 'parent_recipe', 'history', 'weight_pin', 'sources'):
            bad = transition()
            if mutation == 'original_lr': bad['source_learning_rate'] = .0001
            elif mutation == 'parent_pin': bad['parent_transition_sha256'] = '0'*64
            elif mutation == 'parent_recipe': bad['parent_recipe']['optimizer'][0]['weight_decay'] = .02
            elif mutation == 'history': bad['parent_bridge_cost']['step_invocations'] = 4
            elif mutation == 'weight_pin': bad['start_weights_sha256'] = 'bad'
            else: bad['source_sha256']['rate.py'] = 'bad'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): rate.validate_transition(bad, rate.identity(bad))

    def test_every_parameter_and_state_objective_remains_active(self):
        t = transition(); s = saved(17, .0001)
        rate.validate_metadata(s, t)
        self.assertEqual(set(s['accounting']['state_work'].values()), {51})
        for key in s['accounting']['state_work']:
            bad = deepcopy(s); bad['accounting']['state_work'][key] = 45
            with self.assertRaises(ValueError): rate.validate_metadata(bad, t)
        for key in ('unknown_optimizer_outcomes', 'optimizer_returns', 'retained_episodes'):
            bad = deepcopy(s); bad['accounting']['work'][key] += 1
            with self.assertRaises(ValueError): rate.validate_metadata(bad, t)
        for name in ('state_alias_embedding.weight', 'state_decoder.0.weight', 'tokens.weight'):
            self.assertEqual(objective.expected_parameter_step(name, 15, 17, .3), 17)

    def test_history_and_zero_update_weights_cannot_reset(self):
        for mutation in ('cursor', 'evidence', 'historical_cost', 'zero_weights', 'bool_count'):
            value = envelope(15)
            if mutation == 'cursor': value['learner']['cursor'] = 14
            elif mutation == 'evidence': value['learner']['evidence']['consumed_bundle_identity_sha256'] = '6'*64
            elif mutation == 'historical_cost': value['learner']['accounting']['cost']['step_wall_seconds'] = 14.
            elif mutation == 'zero_weights': value['weights_sha256'] = '6'*64
            else: value['rate_updates'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): rate.validate_snapshot(value, value['transition_sha256'])

    def test_descendant_cannot_retarget_rate_or_reset_transition_cost(self):
        for mutation in ('rate', 'steps', 'transition_count', 'schema', 'snapshot_count', 'unknown_field'):
            value = envelope()
            if mutation == 'rate': value['learner']['recipe']['optimizer'][0]['lr'] = .0003
            elif mutation == 'steps': value['cost']['step_invocations'] = 1
            elif mutation == 'transition_count': value['cost']['transition_invocations'] = 2
            elif mutation == 'schema': value['schema'] = objective.SCHEMA
            elif mutation == 'snapshot_count': value['cost']['snapshot_invocations'] = 0
            else: value['promotion'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): rate.validate_snapshot(value, value['transition_sha256'])

    def test_json_roundtrip_is_detached_and_only_lr_differs(self):
        original = transition()
        wire = json.loads(json.dumps(original))
        self.assertEqual(rate.identity(wire), rate.identity(original))
        accepted = rate.validate_transition(wire, rate.identity(original))
        reverted = deepcopy(accepted['recipe']); reverted['optimizer'][0]['lr'] = .0003
        self.assertEqual(reverted, accepted['parent_recipe'])
        accepted['start_accounting']['work']['retained_updates'] = 0
        self.assertEqual(wire['start_accounting']['work']['retained_updates'], 15)

    def test_failure_reports_include_final_cost_and_poison(self):
        class FailingKernel:
            cursor = 15; failed = False; last_report = {'physical_optimizer_updates': None, 'unknown_optimizer_outcomes': 1}
            def step(self, *args, **kwargs): raise ValueError('synthetic unknown step')
            def snapshot(self): raise ValueError('synthetic snapshot failure')
        for operation in ('step', 'snapshot'):
            obj = object.__new__(rate.SharedRateContinuation)
            obj._kernel = FailingKernel(); obj._failed = False; obj._cost = dict.fromkeys(rate.COST_KEYS, 0)
            with patch.object(obj, '_guard', return_value=None), self.assertRaises(ValueError) as caught:
                if operation == 'step': obj.step(None, state_targets={})
                else: obj.snapshot()
            report = caught.exception.rate_report
            self.assertTrue(obj.failed); self.assertEqual(obj._cost[operation+'_invocations'], 1)
            self.assertGreaterEqual(report['wall_seconds'], 0); self.assertGreaterEqual(report['cpu_seconds'], 0)
            if operation == 'step': self.assertEqual(report['kernel_report']['unknown_optimizer_outcomes'], 1)
            else:
                self.assertEqual(report['kernel_snapshot_attempts'], 1)
                self.assertEqual(report['kernel_snapshot_completions'], 0)
            with self.assertRaises(RuntimeError): obj._guard()


if __name__ == '__main__': unittest.main()
