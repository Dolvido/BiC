"""Pure author constraints and bounded rebatching of one pinned existing image."""
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import permutations
import unittest

from experiments import definition_tutor_chapters as c

SOURCE_DIRECTORY=c.ROOT/'runs/definition-basis-data-local/attempt-001'
SOURCE_IMAGE_SHA256='591909b215eeeda42fc2fd3ffa16320dcf5b94d680950623a01be97de2c1c5bc'
TEST_WORK=dict(permutations_checked=0,valid_permutations=0,invalid_permutations=0,
    valid_order_schedules=0,compiled_small_images=0,compiled_small_rows=0,negative_packet_checks=0)


class ChapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract=c.make_contract('a'*64);cls.io_work=c._work()
        manifest=c._read_json(SOURCE_DIRECTORY/'manifest.json',c.SOURCE_DATA_SHA256,cls.io_work)
        cls.record=c._source_records(manifest)[0]
        if cls.record['sha256']!=SOURCE_IMAGE_SHA256:raise AssertionError('exact declared source image required')
        cls.image=c._read_json(SOURCE_DIRECTORY/cls.record['path'],SOURCE_IMAGE_SHA256,cls.io_work)
        cls.packets=c.extract_pairs(cls.image,cls.record,cls.io_work)

    def test_contract_exact_and_detached(self):
        self.assertEqual(c.validate_contract(self.contract),self.contract)
        copied=c.validate_contract(self.contract);copied['skills'].reverse()
        self.assertEqual(self.contract['skills'],list(c.SKILLS))
        for key,value in (('updates',109),('chapter_updates',True),('micro_batch_size',16),
                ('families',['color']),('source_data_manifest_sha256','b'*64),('catalogue_manifest_sha256','notasha')):
            bad=deepcopy(self.contract);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_contract(bad)
        bad=deepcopy(self.contract);bad['extra']='not admitted'
        with self.assertRaises(ValueError):c.validate_contract(bad)

    def test_all_permutations_exactly_ninety_respect_prerequisites(self):
        for order in permutations(c.SKILLS):
            TEST_WORK['permutations_checked']+=1
            allowed=all(order.index(c.SKILLS[i])<order.index(c.SKILLS[i+3]) for i in range(3))
            recipe=dict(schema=c.RECIPE_SCHEMA,chapters=list(order))
            if allowed:
                self.assertEqual(c.validate_recipe(recipe,self.contract),recipe)
                TEST_WORK['valid_permutations']+=1
            else:
                with self.assertRaises(ValueError):c.validate_recipe(recipe,self.contract)
                TEST_WORK['invalid_permutations']+=1
        self.assertEqual(TEST_WORK['valid_permutations'],90);self.assertEqual(TEST_WORK['invalid_permutations'],630)

    def test_recipe_shape_and_default(self):
        self.assertEqual(c.procedural_recipe(self.contract),dict(schema=c.RECIPE_SCHEMA,chapters=list(c.SKILLS)))
        for chapters in ([c.SKILLS[0]]*6,list(c.SKILLS[:-1]),list(c.SKILLS[:-1])+['composition'],tuple(c.SKILLS)):
            with self.assertRaises(ValueError):c.validate_recipe(dict(schema=c.RECIPE_SCHEMA,chapters=chapters),self.contract)
        with self.assertRaises(ValueError):c.validate_recipe(dict(schema=c.RECIPE_SCHEMA,order=list(c.SKILLS)),self.contract)

    def test_author_catalogue_and_schema_expose_only_order(self):
        schema=c.recipe_schema(self.contract);catalogue=c.authoring_catalogue(self.contract)
        self.assertFalse(schema['additionalProperties']);self.assertEqual(set(schema['properties']),{'schema','chapters'})
        self.assertEqual(schema['properties']['chapters']['enum'] if 'enum' in schema['properties']['chapters'] else
            schema['properties']['chapters']['items']['enum'],list(c.SKILLS))
        self.assertEqual([s['id'] for s in catalogue['skills']],list(c.SKILLS))
        self.assertEqual([s['updates'] for s in catalogue['skills']],[18]*6)
        self.assertTrue(all(s['families']==list(c.FAMILIES) for s in catalogue['skills']))
        self.assertEqual([s['prerequisites'] for s in catalogue['skills']],[[],[],[],[c.SKILLS[0]],[c.SKILLS[1]],[c.SKILLS[2]]])

    def test_compiled_order_changes_only_immutable_lesson_order(self):
        manifest=dict(schema=c.SCHEMA,source_data=dict(sha256=c.SOURCE_DATA_SHA256),
            chapters={skill:list(range(18*i,18*(i+1))) for i,skill in enumerate(c.SKILLS)})
        contract=c.make_contract(hashlib.sha256(c.encoded(manifest)).hexdigest())
        for order in (list(c.SKILLS),[c.SKILLS[i] for i in (0,3,1,4,2,5)]):
            indices=c.ordered_indices(dict(schema=c.RECIPE_SCHEMA,chapters=order),contract,manifest)
            self.assertEqual(Counter(indices),Counter(range(108)))
            self.assertEqual(indices[:18],list(range(18)))
            TEST_WORK['valid_order_schedules']+=1
        bad=deepcopy(manifest);bad['chapters'][c.SKILLS[5]][-1]=0
        with self.assertRaises(ValueError):c.ordered_indices(c.procedural_recipe(contract),contract,bad)
        with self.assertRaises(ValueError):c.ordered_indices(c.procedural_recipe(self.contract),self.contract,manifest)
        shifted=deepcopy(manifest);shifted['chapters'][c.SKILLS[0]],shifted['chapters'][c.SKILLS[1]]=shifted['chapters'][c.SKILLS[1]],shifted['chapters'][c.SKILLS[0]]
        shifted_contract=c.make_contract(hashlib.sha256(c.encoded(shifted)).hexdigest())
        with self.assertRaises(ValueError):c.ordered_indices(c.procedural_recipe(shifted_contract),shifted_contract,shifted)

    def test_source_packets_preserve_cross_family_coordinates(self):
        self.assertEqual(len(self.packets),16)
        self.assertEqual(self.io_work['json_reads'],2);self.assertEqual(self.io_work['prefix_label_checks'],96)
        for pair_index,packet in enumerate(self.packets):
            self.assertEqual(packet['source'],dict(index=0,pair_index=pair_index,image_sha256=SOURCE_IMAGE_SHA256))
            seeds={rows[0]['recipe']['seed'] for rows in packet['families'].values()}
            self.assertEqual(len(seeds),1);self.assertEqual(packet['skill'],c.SKILLS[next(iter(seeds))%3])
            for family in c.FAMILIES:
                self.assertEqual(packet['families'][family],self.image['bundle']['families'][family][2*pair_index:2*pair_index+2])
                self.assertEqual(packet['state_targets'][family],self.image['state_targets'][family][2*pair_index:2*pair_index+2])

    def test_three_small_regrouped_images_exact_multisets(self):
        work=c._work();before=Counter();after=Counter()
        for index,packet in enumerate(self.packets[:3]):
            image=c.compile_image([packet],index,work)
            before.update(c.row_target_multiset(packet['families'],packet['state_targets']))
            after.update(c.row_target_multiset(image['bundle']['families'],image['state_targets']))
            self.assertEqual(image['expected_evidence_sha256'],c.identity(image['expected_evidence']))
            self.assertEqual(image['bundle']['bundle_id'],index)
            for family in c.FAMILIES:
                self.assertEqual(image['state_targets'][family],packet['state_targets'][family])
                self.assertEqual(image['bundle']['families'][family],packet['families'][family])
            TEST_WORK['compiled_small_images']+=1;TEST_WORK['compiled_small_rows']+=6
        self.assertEqual(before,after);self.assertEqual(sum(after.values()),18)
        self.assertEqual(work['regrouped_pairs'],9);self.assertEqual(work['new_requested_pairs'],0)

    def test_packet_label_crossing_and_row_mutation_rejected(self):
        for kind in ('targets','row'):
            packet=deepcopy(self.packets[0])
            if kind=='targets':packet['state_targets']['color'][0][0][0]=106
            else:packet['families']['color'][0]['program'][0]['meaning']='not-an-admitted-meaning'
            TEST_WORK['negative_packet_checks']+=1
            with self.assertRaises(ValueError):c.compile_image([packet],0,c._work())

    def test_source_label_not_silently_attached_to_wrong_prefix(self):
        image=deepcopy(self.image);image['state_targets']['color'][0][0][0]=1
        with self.assertRaises(ValueError):c.extract_pairs(image,self.record,c._work())
        correct=c.row_target_multiset(self.image['bundle']['families'],self.image['state_targets'])
        wrong=c.row_target_multiset(image['bundle']['families'],image['state_targets'])
        self.assertNotEqual(correct,wrong)

    def test_mixed_skill_and_duplicate_pair_rejected(self):
        with self.assertRaises(ValueError):c.compile_image(self.packets[:2],0,c._work())
        with self.assertRaises(ValueError):c.compile_image([self.packets[0],self.packets[0]],0,c._work())


if __name__=='__main__':unittest.main()
