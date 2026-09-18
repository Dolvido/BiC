"""Two pinned chapter reconstructions plus bounded pure integrity checks."""
from copy import deepcopy
import hashlib
import unittest

from experiments import referenced_definition_lessons as d

DIRECTORY=d.ROOT/'runs/definition-tutor-chapters-local/attempt-001'
EXPECTED={0:'f0fe591454e48107584e23c900196198f4436a84581438954838f11f66b877e4',
          107:'794e929e47264caabb9430fcad8b558863bb5e0978ee0d8a0af0a38c80a1c5cb'}
TEST_WORK=dict(exact_reconstruction_checks=0,ownership_checks=0,reference_checks=0,
    image_negative_checks=0,compact_metadata_checks=0,cache_checks=0,boundary_checks=0,poison_checks=0)


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store=d.ChapterLessonStore.from_chapters(DIRECTORY,manifest_sha256=d.CHAPTER_SHA,
            indices=(0,107),max_source_images=3)
        cls.images={index:cls.store.resolve(index) for index in (0,107)}
        cls.snapshots=deepcopy(cls.images)
        cls.success_work=cls.store.report()

    def test_a_exact_original_bytes_labels_and_cursor(self):
        for index,image in self.images.items():
            TEST_WORK['exact_reconstruction_checks']+=1
            self.assertEqual(hashlib.sha256(d.encoded(image)).hexdigest(),EXPECTED[index])
            self.assertEqual(image['bundle']['bundle_id'],index)
            self.assertEqual(image['expected_evidence']['bundle_id'],index)
            self.assertEqual(image['admission_schema'],d.basis.curriculum.VERSION)
            self.assertEqual({len(v) for v in image['bundle']['families'].values()},{32})
            self.assertEqual({len(v) for v in image['state_targets'].values()},{32})

    def test_b_returned_rows_labels_and_evidence_are_detached(self):
        # Observe private cache bytes only to check isolation, never mint objects.
        before={k:d.identity(v[0]) for k,v in self.store._cache.items()}
        manifest_before=d.identity(self.store._manifest)
        returned=self.images[107];returned['bundle']['families']['color'][0]['turns'][0]['text']='caller mutation'
        returned['state_targets']['count'][0][0][0]=106
        returned['expected_evidence']['bundle_id']=999
        self.assertEqual({k:d.identity(v[0]) for k,v in self.store._cache.items()},before)
        self.assertEqual(d.identity(self.store._manifest),manifest_before)
        self.assertEqual(hashlib.sha256(d.encoded(self.images[0])).hexdigest(),EXPECTED[0])
        TEST_WORK['ownership_checks']+=1

    def test_c_pair_references_reject_crossing_and_wrong_source(self):
        sources=[dict(index=0,sha256='a'*64)]
        refs=[dict(index=0,pair_index=i,image_sha256='a'*64) for i in range(16)]
        TEST_WORK['reference_checks']+=1;self.assertTrue(d.validate_references(refs,sources))
        for mutation in ('duplicate','pair','source','hash','extra','count'):
            value=deepcopy(refs)
            if mutation=='duplicate':value[1]=deepcopy(value[0])
            elif mutation=='pair':value[0]['pair_index']=16
            elif mutation=='source':value[0]['index']=1
            elif mutation=='hash':value[0]['image_sha256']='b'*64
            elif mutation=='extra':value[0]['family']='color'
            else:value.pop()
            TEST_WORK['reference_checks']+=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):d.validate_references(value,sources)

    def test_d_labels_partition_pair_and_evidence_mutations_rejected(self):
        record=self.store._manifest['selected_training']['0']['original']
        for mutation in ('labels','partition','pair','bool','evidence','rows'):
            image=deepcopy(self.snapshots[0])
            if mutation=='labels':image['state_targets']['color'][0][0][0]=(image['state_targets']['color'][0][0][0]+1)%107
            elif mutation=='partition':image['bundle']['families']['color'][0]['split']='dev'
            elif mutation=='pair':image['bundle']['families']['color'][1]=deepcopy(image['bundle']['families']['color'][3])
            elif mutation=='bool':image['state_targets']['count'][0][0][0]=True
            elif mutation=='evidence':image['expected_evidence_sha256']='e'*64
            else:image['bundle']['families']['switch'][0]['turns'][0]['text']='changed English'
            TEST_WORK['image_negative_checks']+=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):d.authenticate_image(image,record)

    def test_e_compact_expected_metadata_is_bound_to_frozen_image(self):
        entry=self.store._manifest['selected_training']['0'];original=entry['original']
        for name in ('bundle_identity_sha256','canonical_rows_sha256','canonical_targets_sha256','row_target_multiset_sha256'):
            bad=deepcopy(original);bad[name]='e'*64;TEST_WORK['compact_metadata_checks']+=1
            with self.subTest(name=name),self.assertRaises(ValueError):d.compact_record(bad,entry['references'],self.snapshots[0])

    def test_f_actual_cache_read_and_reconstruction_bound(self):
        work=self.success_work;TEST_WORK['cache_checks']+=1
        for key,value in dict(json_read_attempts=13,json_reads=13,source_image_loads=6,source_image_checks=6,
                expected_image_checks=2,resolve_attempts=2,resolves=2,reconstructed_pairs=96,reconstructed_rows=192,
                cache_hits=26,cache_misses=6,evictions=3,peak_cached_images=3,cached_images=3,failures=0).items():
            self.assertEqual(work[key],value,key)
        self.assertEqual(len(work['input_sha256']),13)
        self.assertEqual([k for k in self.store._cache],[103,105,107])
        for key in ('new_requested_pairs','canonical_regenerations','prefix_interpretations','archive_loads','models','teacher_calls'):
            self.assertEqual(work[key],0)

    def test_g_bad_selection_and_capacity_rejected_before_io(self):
        for indices,capacity in (((0,0),3),((-1,),3),((108,),3),((True,),3),((0,),0),((0,),109)):
            TEST_WORK['boundary_checks']+=1
            with self.assertRaises(ValueError):d.ChapterLessonStore.from_chapters(DIRECTORY,manifest_sha256=d.CHAPTER_SHA,
                indices=indices,max_source_images=capacity)

    def test_z_outside_selection_poisons_without_new_reads_or_reconstruction(self):
        before=self.store.report()
        for index in (1,0):
            TEST_WORK['poison_checks']+=1
            with self.assertRaises(ValueError):self.store.resolve(index)
        after=self.store.report();self.assertTrue(after['failed'])
        for key in ('json_reads','source_image_loads','reconstructed_pairs','reconstructed_rows'):
            self.assertEqual(before[key],after[key])
        self.assertEqual(after['resolve_attempts'],4);self.assertEqual(after['resolves'],2);self.assertEqual(after['failures'],2)


if __name__=='__main__':unittest.main()
