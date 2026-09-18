"""Focused CPU collision repair proof: at most2000 canonical calls/4000 rows.

Two18-bundle micro2 compilations compare unchanged successful v1/v2 images.
The actual protected switch example is reproduced separately. Other tests use
small fake pair generators for atomicity, repeated text and finite exhaustion.
No learner, tensor packing, teacher, GPU or production curriculum compilation.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments import verified_tutor_curriculum as old, verified_tutor_curriculum_v2 as new
from experiments import foundation_plan as planning, foundation_curriculum as foundation

WORK=dict(canonical_generate_attempts=0,canonical_generate_completions=0,canonical_rows_returned=0,
    successful_compilations=0,failed_compilations=0,mock_generate_calls=0,model_work=0,teacher_calls=0)
REPORTS=[]

class CollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original=foundation.generate_pair
        def counted(*args,**kwargs):
            WORK['canonical_generate_attempts']+=1
            if WORK['canonical_generate_attempts']>2000:raise AssertionError('declared generator bound exceeded')
            value=original(*args,**kwargs)
            WORK['canonical_generate_completions']+=1;WORK['canonical_rows_returned']+=len(value)
            return value
        p=patch.object(foundation,'generate_pair',side_effect=counted);p.start();cls.addClassCleanup(p.stop)
        cls.plan=planning.build_plan(seed=852890001,stage_updates=10,final_updates=6,micro_batch_size=2,
            rehearsal_every=2,ordering_seed=852890002,
            admission=dict(schema=planning.ADMISSION_SCHEMA,protected_sha256=old._hash([]),protected_count=0,realization_attempts={}))
        pin=old._hash(cls.plan); selected={};descriptors=[];cls.parents={}
        for slot in cls.plan['bundles'].values():selected.setdefault((slot['depth'],slot['turns']),slot['id'])
        for cell in old.CELLS:
            bundle=selected[cell];pairs={f:planning.materialize_pair(cls.plan,bundle,f,0) for f in old.FAMILIES}
            cls.parents[cell]=pairs
            descriptors.append(dict(plan_sha256=pin,bundle_id=bundle,pair_index=0,pair_sha256={f:old._hash(p) for f,p in pairs.items()}))
        cls.inventory=dict(schema=old.INVENTORY_SCHEMA,plans={pin:cls.plan},motifs=dict(balanced=descriptors),replay=deepcopy(descriptors))

    def contract(self,protected=(),replay=False):
        return dict(schema=old.CONTRACT_SCHEMA,inventory_sha256=old.inventory_sha256(self.inventory),
            protected_sha256=old._hash(sorted(protected)),protected_count=len(protected),seed=852890003,
            start_cursor=5000,micro_batch_size=2,layout='original',
            chapters=[dict(motifs=['balanced'],realizations=list(old.REALIZATIONS),default=dict(motif='balanced',realization='independent'))],
            slots=[dict(chapter=0,depth=d,turns=t,replay=replay) for d,t in old.CELLS],
            rehearsal_floor_by_depth={str(d):3 for d in range(5)},replay_floor_per_cell=int(replay),
            limits=dict(max_episodes=108,max_observation_bytes=200000,max_reply_target_bytes=40000,max_generator_calls=1000,max_seconds=60))

    def compile(self,module,contract,protected=()):
        try:
            result=module.compile_curriculum(old.procedural_recipe(contract),admitted_parent_inventory=self.inventory,
                protected_transcripts=protected,coverage_contract=contract)
            WORK['successful_compilations']+=1;REPORTS.append(deepcopy(result['receipt']));return result
        except BaseException as error:
            WORK['failed_compilations']+=1;REPORTS.append(deepcopy(error.compilation_report));raise

    def helper(self, generator, *, mode='rename', protected=(), seen=None, boundary=lambda:None):
        work=dict(fresh_candidate_attempts=0,fresh_candidate_completions=0,rejected_fresh_candidates=0);rejections=[]
        base=dict(seed=4046328733178575489,depth=4,turns=8,structure_split='train',naming_seed=123,value_seed=7453660802251341196)
        args=dict(family='switch',choice=dict(motif='change_emphasis',realization=mode),contract=dict(seed=852802001),
            index=84,offset=2,protected=set(protected),fresh_seen=set() if seen is None else seen,work=work,rejections=rejections,boundary=boundary)
        return base,args,work,rejections

    def test_01_valid_candidate_zero_images_and_evidence_are_exact_v1(self):
        contract=self.contract();before=deepcopy((contract,self.inventory))
        original=self.compile(old,contract);current=self.compile(new,contract)
        self.assertEqual(original['images'],current['images'])
        self.assertEqual(current['manifest']['schema'],new.SCHEMA)
        for key in ('recipe_sha256','contract_sha256','inventory_sha256','protected_sha256','images_sha256','exposures','coverage','operator_counts','target_counts','unique_transcripts'):
            self.assertEqual(original['manifest'][key],current['manifest'][key])
        self.assertEqual(current['receipt']['work']['rejected_fresh_candidates'],0)
        for p in current['provenance']:
            for parent in p['parents']:
                self.assertEqual(set(parent['realizations']),set(new.FAMILIES))
                self.assertTrue(all(v['candidate_attempt']==0 for v in parent['realizations'].values()))
        self.assertEqual((contract,self.inventory),before)

    def test_02_actual_protected_rename_pair_is_repaired_without_value_change(self):
        protected={'2ecf199a8f4e93cb8dc4790dd7e18671f15fd7c3e6dafd94a6ab25770171552f','5654984b6d2c9f3c6e42efef34ee0e50e11d58739758cc1ab2ea630377031d11'}
        base,args,work,rejections=self.helper(foundation,protected=protected)
        pair,actual=new._fresh_pair(foundation,base,**args)
        self.assertEqual(rejections[0]['naming_seed'],5417659525411527266)
        self.assertEqual({x['transcript_sha256'] for x in rejections[0]['rejected_members']},protected)
        self.assertTrue(all(x['reasons']==['protected'] for x in rejections[0]['rejected_members']))
        self.assertGreater(actual['candidate_attempt'],0)
        self.assertEqual(actual['value_seed'],base['value_seed'])
        self.assertTrue(all(new.transcript_sha256(r) not in protected for r in pair))
        self.assertEqual(args['fresh_seen'],set())
        foundation.validate_pair(pair)
        self.assertEqual({r['recipe']['seed'] for r in pair},{base['seed']})
        REPORTS.append(dict(scope='actual failed pair retry',work=work,rejections=rejections,accepted=actual))

    def test_03_second_member_collision_does_not_commit_first_member(self):
        first=[dict(turns=[dict(text='first')]),dict(turns=[dict(text='seen')])]
        second=[dict(turns=[dict(text='second')]),dict(turns=[dict(text='third')])]
        seen={new.transcript_sha256(first[1])};before=set(seen);calls=[]
        def generate(*a,**kw):
            WORK['mock_generate_calls']+=1;calls.append(kw);return deepcopy(first if len(calls)==1 else second)
        base,args,work,rejections=self.helper(None,seen=seen)
        pair,actual=new._fresh_pair(SimpleNamespace(generate_pair=generate),base,**args)
        self.assertEqual(pair,second);self.assertEqual(seen,before)
        self.assertEqual(rejections[0]['rejected_members'][0]['reasons'],['repeated_fresh'])
        self.assertEqual(actual['candidate_attempt'],1)
        self.assertNotEqual(calls[0]['naming_seed'],calls[1]['naming_seed'])
        self.assertEqual(calls[0]['value_seed'],calls[1]['value_seed'])
        for mode in ('rename','revalue','independent'):
            values=[new._candidate_seeds(base,dict(realization=mode),dict(seed=852802001),84,2,a) for a in (0,1)]
            if mode=='revalue':self.assertEqual(values[0][0],values[1][0]);self.assertNotEqual(values[0][1],values[1][1])
            else:self.assertNotEqual(values[0][0],values[1][0]);self.assertEqual(values[0][1],values[1][1])

    def test_04_finite_revalue_exhaustion_never_renames_or_changes_parent(self):
        pair=[dict(turns=[dict(text='blocked')]),dict(turns=[dict(text='other')])];digest=new.transcript_sha256(pair[0]);calls=[]
        def generate(*a,**kw):WORK['mock_generate_calls']+=1;calls.append((a,kw));return deepcopy(pair)
        base,args,work,rejections=self.helper(None,mode='revalue',protected={digest},seen={digest})
        with self.assertRaisesRegex(ValueError,'exhausted 64 candidates'):
            new._fresh_pair(SimpleNamespace(generate_pair=generate),base,**args)
        self.assertEqual(work,dict(fresh_candidate_attempts=64,fresh_candidate_completions=64,rejected_fresh_candidates=64))
        self.assertEqual(len(rejections),64)
        self.assertEqual({kw['naming_seed'] for _,kw in calls},{base['naming_seed']})
        self.assertEqual({a for a,_ in calls},{('switch',base['seed'])})
        self.assertEqual(rejections[0]['rejected_members'][0]['reasons'],['protected','repeated_fresh'])

    def test_05_deadline_stops_retry_preserving_partial_counts(self):
        pair=[dict(turns=[dict(text='same')]),dict(turns=[dict(text='same')])];checks=[]
        def boundary():
            checks.append(1)
            if len(checks)==2:raise TimeoutError('fixture allowance')
        def generate(*a,**kw):WORK['mock_generate_calls']+=1;return deepcopy(pair)
        base,args,work,rejections=self.helper(None,boundary=boundary)
        with self.assertRaises(TimeoutError):new._fresh_pair(SimpleNamespace(generate_pair=generate),base,**args)
        self.assertEqual(work['fresh_candidate_attempts'],1)
        self.assertEqual(rejections[0]['rejected_members'][0]['reasons'],['repeated_within_pair'])

    def test_06_protected_replay_is_never_resampled(self):
        protected={new.transcript_sha256(self.parents[new.CELLS[0]]['color'][0])}
        with self.assertRaisesRegex(ValueError,'protected replay') as caught:self.compile(new,self.contract(protected,replay=True),protected)
        work=caught.exception.compilation_report['work']
        self.assertEqual(work['fresh_candidate_attempts'],0);self.assertEqual(work['materialized_pairs'],0)
