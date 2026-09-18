"""Synthetic metadata/content checks: no canonical lesson or archive loads."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import unittest

from experiments import definition_tutor_pair_data as d

TEST_WORK=dict(synthetic_images=0,synthetic_rows=0,schedules=0,contrast_checks=0,
    content_identity_checks=0,image_envelope_checks=0,tiny_json_read_attempts=0,tiny_json_reads=0)


def image_fixture():
    rows={family:[dict(id=f'{family}-{i}',program={'metadata':'ignored'},turns=[
        dict(text=f'Synthetic {family} {i} first.',observations=[],target=i,reply=('Ack.','Yes.')[i]),
        dict(text=f'Synthetic {family} {i} second.',observations=[],target=2,reply='No.')]) for i in range(2)] for family in d.FAMILIES}
    targets={family:[[[i]*12,[i+1]*12] for i in range(2)] for family in d.FAMILIES}
    evidence=dict(bundle_id=7,bundle_rows_sha256=d.identity([[family,rows[family]] for family in d.FAMILIES]))
    TEST_WORK['synthetic_images']+=1;TEST_WORK['synthetic_rows']+=6
    return dict(bundle=dict(bundle_id=7,families=rows),expected_evidence=evidence,
        expected_evidence_sha256=d.identity(evidence),state_targets=targets,admission_schema='synthetic-provider')


def records():
    return [dict(index=i,path=f'runs/synthetic-pair-proof/{i:03d}.json',sha256=d.identity(['image',i]),
        provider='synthetic-provider',format='json',learning_identity_sha256=d.identity(['learning',i])) for i in range(108)]


class PairedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.image=image_fixture()

    def content(self,image):
        TEST_WORK['content_identity_checks']+=1;return d.learning_identity(image)

    def contrast(self,arms):
        TEST_WORK['contrast_checks']+=1;return d.teaching_contrast(arms)

    def envelope(self,image,record):
        TEST_WORK['image_envelope_checks']+=1;return d._check_image(image,record)

    def test_withdrawal_fixed_whole_curriculum_schedule(self):
        chapter=dict(training=records());old=records();replay=records()
        for record in replay:record.update(format='torch_weights_only',original_cursor=500+record['index'])
        result=d.withdrawal_records(chapter,dict(old_definition=old,replay=replay),d.ROOT)
        TEST_WORK['schedules']+=1
        self.assertEqual(len(result),108)
        self.assertEqual([r['kind'] for r in result],['basis','replay','old_definition','replay','basis','replay']*18)
        chosen={kind:[r for r in result if r['kind']==kind] for kind in ('basis','old_definition','replay')}
        self.assertEqual([r['index'] for r in chosen['basis']],[(j%6)*18+j//6 for j in range(36)])
        self.assertEqual(Counter(r['index']//18 for r in chosen['basis']),Counter({i:6 for i in range(6)}))
        self.assertEqual([r['index'] for r in chosen['old_definition']],list(range(18)))
        self.assertEqual([r['index'] for r in chosen['replay']],list(range(54)))
        self.assertEqual([r['original_cursor'] for r in chosen['replay']],list(range(500,554)))
        self.assertTrue(all(not Path(r['path']).is_absolute() for r in result))

    def test_accepted_order_exact_multiset_and_seventy_two_different_positions(self):
        base=records();order=[i for chapter in (0,3,1,4,2,5) for i in range(chapter*18,(chapter+1)*18)]
        actual=self.contrast(dict(procedural=base,tutor=[deepcopy(base[i]) for i in order]))
        self.assertFalse(actual['zero_treatment_contrast']);self.assertEqual(actual['different_teaching_positions'],72)
        self.assertNotEqual(actual['teaching_learning_stream_sha256']['procedural'],actual['teaching_learning_stream_sha256']['tutor'])

    def test_metadata_order_alone_does_not_create_treatment(self):
        base=records()
        for record in base:record['learning_identity_sha256']='a'*64
        actual=self.contrast(dict(procedural=base,tutor=list(reversed(deepcopy(base)))))
        self.assertTrue(actual['zero_treatment_contrast']);self.assertEqual(actual['different_teaching_positions'],0)

    def test_missing_crossed_content_and_changed_image_membership_rejected(self):
        base=records()
        for mutation in ('missing','duplicate','file','content'):
            arms=dict(procedural=deepcopy(base),tutor=deepcopy(base))
            if mutation=='missing':arms['tutor'].pop()
            elif mutation=='duplicate':arms['tutor'][1]=deepcopy(arms['tutor'][0])
            elif mutation=='file':arms['tutor'][0]['sha256']='f'*64
            else:arms['tutor'][0]['learning_identity_sha256']='f'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.contrast(arms)

    def test_every_learning_field_and_label_alignment_changes_identity(self):
        expected=self.content(self.image)
        for field in ('text','observations','target','reply','state','row_order','labels_crossed'):
            image=deepcopy(self.image);turn=image['bundle']['families']['color'][0]['turns'][0]
            if field=='text':turn['text']='Changed visible text.'
            elif field=='observations':turn['observations']=[1]
            elif field=='target':turn['target']=3
            elif field=='reply':turn['reply']='Changed reply.'
            elif field=='state':image['state_targets']['color'][0][0][0]=106
            elif field=='row_order':image['bundle']['families']['color'].reverse()
            else:image['state_targets']['color'].reverse()
            self.assertNotEqual(self.content(image),expected,field)
        image=deepcopy(self.image);image['bundle']['bundle_id']=999
        image['bundle']['families']['color'][0].update(id='new provenance',program={'ignored':True},queries=[{'metadata':3}])
        self.assertEqual(self.content(image),expected)

    def test_invalid_label_alignment_and_native_fields_rejected(self):
        for mutation in ('missing_family','missing_row','missing_turn','width','bool','range','native_bool'):
            image=deepcopy(self.image)
            if mutation=='missing_family':del image['state_targets']['switch']
            elif mutation=='missing_row':image['state_targets']['count'].pop()
            elif mutation=='missing_turn':image['state_targets']['count'][0].pop()
            elif mutation=='width':image['state_targets']['count'][0][0].pop()
            elif mutation=='bool':image['state_targets']['count'][0][0][0]=True
            elif mutation=='range':image['state_targets']['count'][0][0][0]=107
            else:image['bundle']['families']['count'][0]['turns'][0]['target']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.content(image)

    def test_image_evidence_provider_cursor_and_row_binding(self):
        record=dict(index=7,provider='synthetic-provider')
        self.assertEqual(self.envelope(self.image,record),d.learning_identity(self.image))
        for mutation in ('provider','cursor','evidence','rows','extra'):
            image=deepcopy(self.image);actual=deepcopy(record)
            if mutation=='provider':actual['provider']='wrong'
            elif mutation=='cursor':actual['index']=8
            elif mutation=='evidence':image['expected_evidence_sha256']='a'*64
            elif mutation=='rows':image['bundle']['families']['color'][0]['id']='altered full row'
            else:image['unexpected']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.envelope(image,actual)

    def test_authenticated_tiny_json_content_and_hash_before_parse(self):
        directory=Path(os.environ['BIC_PAIR_TEST_DIR']);path=directory/'synthetic-image.json'
        raw=(json.dumps(self.image,sort_keys=True,separators=(',',':'))+'\n').encode()
        with path.open('xb') as file:file.write(raw)
        record=dict(path=path.relative_to(d.ROOT).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),
            index=7,provider='synthetic-provider',format='json',kind='basis',learning_identity_sha256=d.learning_identity(self.image))
        TEST_WORK['tiny_json_read_attempts']+=1
        self.assertEqual(d.load_json_image(record),self.image);TEST_WORK['tiny_json_reads']+=1
        wrong=deepcopy(record);wrong['learning_identity_sha256']='f'*64
        TEST_WORK['tiny_json_read_attempts']+=1
        with self.assertRaisesRegex(ValueError,'full learning content'):d.load_json_image(wrong)
        TEST_WORK['tiny_json_reads']+=1
        bad=directory/'invalid-json.json'
        with bad.open('xb') as file:file.write(b'not even JSON')
        wrong=dict(record,path=bad.relative_to(d.ROOT).as_posix())
        TEST_WORK['tiny_json_read_attempts']+=1
        with self.assertRaisesRegex(ValueError,'input bytes changed'):d.load_json_image(wrong)


if __name__=='__main__':unittest.main()
