"""Pure controller boundaries and completed-resume cross-binding; no I/O/model."""
from collections import Counter
from copy import deepcopy
import unittest

from experiments import sustained_acquisition as controller

WORK = Counter()


def call(name, *args, **kwargs):
    WORK['direct_' + name + '_calls'] += 1
    return getattr(controller, name)(*args, **kwargs)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        WORK['test_methods'] += 1

    def fixture(self, cursor=2811, parent=2160):
        WORK['synthetic_resume_envelopes'] += 1
        state = dict(origin_cursor=2160, cursor=cursor, curriculum_updates=cursor-2160,
            completed_passes=(cursor-2160)//648, next_source_index=(cursor-2160)%648, source_bundles=648)
        sources = {'synthetic.py': 'a'*64}
        scores = {name: {'metrics': {'count': 1}, 'groups': {}} for name in controller.BANK_COUNTS}
        launch = dict(schema=controller.SCHEMA, data_manifest_sha256='b'*64,
                      source_sha256=deepcopy(sources), parent_cursor=parent)
        summary = dict(status='completed', partial_work_unknown=False, launch_sha256='c'*64,
                       curriculum=deepcopy(state), evaluations={str(cursor): deepcopy(scores)})
        commit = dict(schema=controller.SCHEMA, launch_sha256='c'*64, data_manifest_sha256='b'*64,
                      training_manifest_sha256=controller.TRAINING_SHA,
                      curriculum=deepcopy(state), scores=deepcopy(scores))
        kwargs = dict(launch_sha256='c'*64, data_manifest_sha256='b'*64, current_sources=sources)
        return launch, summary, commit, kwargs

    def test_01_cycle_boundary_and_partial_cycle_coordinates(self):
        for cursor, passes, index in ((2160,0,0),(2807,0,647),(2808,1,0),(2811,1,3),
                                      (3455,1,647),(3456,2,0),(2160+5184,8,0)):
            state = call('curriculum_state', cursor)
            self.assertEqual(state['completed_passes'], passes)
            self.assertEqual(state['next_source_index'], index)
            self.assertEqual(passes*648+index, state['curriculum_updates'])
            self.assertEqual(state['origin_cursor']+state['curriculum_updates'], cursor)
        for invalid in (True, None, 2159, -1, 2160.0):
            with self.assertRaises(ValueError): call('curriculum_state', invalid)

    def test_02_explicit_budget_bounds_and_no_cycle_cap(self):
        for budget in (120,1800,86400):
            spec = call('contract', budget)
            self.assertEqual(spec['budget_seconds'], budget)
            self.assertEqual(spec['evaluation_interval'],648)
            self.assertEqual(spec['final_reserve_seconds'],60)
            self.assertEqual(sum(spec['bank_counts'].values()),2340)
            self.assertFalse(spec['automatic_retry']); self.assertFalse(spec['automatic_promotion'])
            self.assertNotIn('max_cycles', spec)
        for invalid in (True,None,119,86401,1800.0):
            with self.assertRaises(ValueError): call('contract', invalid)

    def test_03_valid_completed_partial_resume_is_detached_read_only(self):
        for cursor,parent,index in ((2811,2160,3),(3456,2811,0),(2160,2160,0)):
            launch,summary,commit,kwargs=self.fixture(cursor,parent)
            original=deepcopy((launch,summary,commit,kwargs))
            self.assertEqual(call('validate_resume',launch,summary,commit,**kwargs),cursor)
            self.assertEqual(call('curriculum_state',cursor)['next_source_index'],index)
            self.assertEqual((launch,summary,commit,kwargs),original)

    def test_04_foreign_launch_data_training_and_sources_rejected(self):
        mutations=(('launch','schema','foreign'),('commit','schema','foreign'),
            ('summary','launch_sha256','d'*64),('commit','launch_sha256','d'*64),
            ('launch','data_manifest_sha256','d'*64),('commit','data_manifest_sha256','d'*64),
            ('commit','training_manifest_sha256','d'*64),('launch','source_sha256',{'synthetic.py':'d'*64}))
        for destination,key,value in mutations:
            launch,summary,commit,kwargs=self.fixture()
            {'launch':launch,'summary':summary,'commit':commit}[destination][key]=value
            with self.assertRaises(ValueError): call('validate_resume',launch,summary,commit,**kwargs)
            WORK['rejected_mutated_envelopes'] += 1

    def test_05_incomplete_unknown_stale_and_backwards_state_rejected(self):
        for mutation in ('failed','unknown','stale_final','wrong_index','backwards'):
            launch,summary,commit,kwargs=self.fixture()
            if mutation=='failed':summary['status']='failed'
            elif mutation=='unknown':summary['partial_work_unknown']=True
            elif mutation=='stale_final':summary['curriculum']['cursor']+=1
            elif mutation=='wrong_index':commit['curriculum']['next_source_index']=4
            else:launch['parent_cursor']=2812
            with self.assertRaises(ValueError):call('validate_resume',launch,summary,commit,**kwargs)
            WORK['rejected_mutated_envelopes'] += 1

    def test_06_committed_scores_must_be_completed_final_scores(self):
        launch,summary,commit,kwargs=self.fixture()
        commit['scores']['transfer_varied']['metrics']['count']=2
        with self.assertRaises(ValueError):call('validate_resume',launch,summary,commit,**kwargs)
        WORK['rejected_mutated_envelopes'] += 1


if __name__=='__main__':unittest.main()
