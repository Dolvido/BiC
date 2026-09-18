"""Bounded schedule, provenance, immutable I/O and six-row admission checks."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import unittest

from experiments import definition_basis_data as d

TEST_WORK=dict(synthetic_catalogues=0,scheduled_updates_inspected=0,training_seed_coordinates_inspected=0,
               small_image_rows=0,authenticated_fixture_reads=0,publication_attempts=0,invalid_build_calls=0)

def pin(text):return hashlib.sha256(text.encode()).hexdigest()


def original_catalogue():
    TEST_WORK['synthetic_catalogues']+=1
    training=[dict(index=i,panel=('binding','revision')[i%2],path=f'training/{i:03d}.json',sha256=pin(f'definition{i}')) for i in range(108)]
    replay=[dict(path=f'runs/fake-replay/{i:03d}.pt',sha256=pin(f'replay{i}'),original_cursor=7924+i) for i in range(108)]
    banks=dict(path='banks.json',sha256=pin('banks'))
    return dict(schema=d.OLD_SCHEMA,definition_schema=d.OLD_VERSION,micro_batch_size=32,definition_updates=108,replay_updates=108,
        total_updates=216,training=training,replay=replay,banks=banks,
        artifact_sha256={r['path']:r['sha256'] for r in training+[banks]},input_sha256={r['path']:r['sha256'] for r in replay},
        bank_inventory={panel:dict(episodes=192,pairs=96,family_episodes={f:64 for f in d.curriculum.FAMILIES},rows_sha256=pin(panel))
            for panel in d.PANELS[:3]})


class DataTests(unittest.TestCase):
    def test_schedule_exact_reuse_and_216_update_coverage(self):
        plan=d.schedule();self.assertEqual(len(plan),648)
        TEST_WORK['scheduled_updates_inspected']+=len(plan)
        self.assertEqual(plan[:6],[dict(kind=kind,source_index=i) for kind,i in
            (('basis',0),('replay',0),('old_definition',0),('replay',1),('basis',1),('replay',2))])
        self.assertEqual(plan[-6:],[dict(kind=kind,source_index=i) for kind,i in
            (('basis',106),('replay',105),('old_definition',107),('replay',106),('basis',107),('replay',107))])
        for kind,uses,updates,first_distinct in (('basis',2,216,72),('old_definition',1,108,36),('replay',3,324,108)):
            counts=Counter(record['source_index'] for record in plan if record['kind']==kind)
            self.assertEqual(counts,{i:uses for i in range(108)})
            self.assertEqual(sum(counts.values()),updates)
            self.assertEqual(len({r['source_index'] for r in plan[:216] if r['kind']==kind}),first_distinct)
        self.assertTrue(all(set(record)=={'kind','source_index'} for record in plan))

    def test_seed_plan_balances_each_panel_and_contrast_without_generation(self):
        counts=Counter();seeds=set()
        for index in range(108):
            for offset in range(16):
                number=d.training_seed(d.SEED,index,offset)
                TEST_WORK['training_seed_coordinates_inspected']+=1
                self.assertNotIn(number,seeds);seeds.add(number)
                counts[(index%2,number%3)]+=1
        self.assertEqual(counts,{(panel,contrast):288 for panel in range(2) for contrast in range(3)})
        self.assertLess(max(seeds),d.SEED+1000000)
        for coordinates in ((True,0),(108,0),(0,16),(0,-1)):
            with self.assertRaises(ValueError):d.training_seed(d.SEED,*coordinates)

    def test_inherited_catalogue_membership_without_lesson_decode(self):
        original=original_catalogue();snapshot=deepcopy(original)
        definitions,replay,banks=d._inherit(original,d.OLD_DIRECTORY)
        self.assertEqual(original,snapshot)
        self.assertEqual(len(definitions),108);self.assertEqual(len(replay),108)
        self.assertEqual(definitions[0]['path'],'runs/definition-data-local/attempt-001/training/000.json')
        self.assertEqual(definitions[-1]['panel'],'revision')
        self.assertEqual(replay[0]['original_cursor'],7924);self.assertEqual(replay[-1]['original_cursor'],8031)
        self.assertEqual(banks['path'],'runs/definition-data-local/attempt-001/banks.json')
        self.assertEqual(definitions[0]['provider'],d.OLD_VERSION)
        self.assertEqual(replay[0]['format'],'torch_weights_only')

    def test_inherited_mutation_and_path_escape_rejected(self):
        for defect in ('order','pin','escape','cursor','inventory','count'):
            original=original_catalogue()
            if defect=='order':original['training'][1]['index']=2
            elif defect=='pin':original['training'][0]['sha256']='0'*64
            elif defect=='escape':
                original['training'][0]['path']='../../escaped.json';original['artifact_sha256']['../../escaped.json']=original['training'][0]['sha256']
            elif defect=='cursor':original['replay'][0]['original_cursor']+=1
            elif defect=='inventory':original['bank_inventory']['binding']['pairs']=95
            else:original['replay'].pop()
            with self.subTest(defect=defect),self.assertRaises(ValueError):d._inherit(original,d.OLD_DIRECTORY)

    def test_six_row_basis_image_evidence_and_causal_targets(self):
        bundle=dict(schema=d.curriculum.BUNDLE_SCHEMA,bundle_id=0,layout='original',families={
            family:d.curriculum.generate_pair(family,91002,panel='binding') for family in d.curriculum.FAMILIES})
        work=d._work();image=d._image(bundle,work)
        TEST_WORK['small_image_rows']+=sum(len(rows) for rows in bundle['families'].values())
        self.assertEqual(set(image),{'bundle','expected_evidence','expected_evidence_sha256','state_targets','admission_schema'})
        self.assertEqual(image['admission_schema'],d.curriculum.VERSION)
        self.assertEqual(image['expected_evidence_sha256'],d.identity(image['expected_evidence']))
        self.assertEqual(work['admitted_pairs'],3);self.assertEqual(work['prefix_label_rows'],6)
        self.assertEqual(work['prefix_label_elements'],864)
        for family,labels in image['state_targets'].items():
            self.assertEqual(len(labels),2)
            for sample in labels:
                self.assertEqual(len(sample),12);self.assertTrue(all(len(turn)==12 for turn in sample))
                self.assertEqual(sample[0],[0]*12);self.assertEqual(sample[1],[0]*12)
            self.assertEqual(image['expected_evidence']['families'][family]['exposures']['episodes'],2)

    def test_hash_checked_before_json_decode(self):
        path=Path(os.environ['BIC_DATA_TEST_SCRATCH'])/'pin.json'
        raw=b'{"accepted":true}';path.write_bytes(raw);expected=hashlib.sha256(raw).hexdigest();work=d._work()
        TEST_WORK['authenticated_fixture_reads']+=1
        self.assertEqual(d._checked(path,expected,work,decode=True),{'accepted':True})
        path.write_bytes(b'{not-json')
        TEST_WORK['authenticated_fixture_reads']+=1
        with self.assertRaises(ValueError):d._checked(path,expected,work,decode=True)
        self.assertEqual(work['input_hash_attempts'],2);self.assertEqual(work['input_hashes'],1)
        self.assertEqual(work['json_decode_attempts'],1);self.assertEqual(work['json_decodes'],1)
        self.assertEqual(work['archive_loads'],0)

    def test_publication_refuses_overwrite(self):
        path=Path(os.environ['BIC_DATA_TEST_SCRATCH'])/'exclusive.json'
        TEST_WORK['publication_attempts']+=1
        expected=d.publish(path,{'first':1});before=path.read_bytes()
        TEST_WORK['publication_attempts']+=1
        with self.assertRaises(FileExistsError):d.publish(path,{'second':2})
        self.assertEqual(path.read_bytes(),before);self.assertEqual(hashlib.sha256(before).hexdigest(),expected)

    def test_invalid_build_contract_fails_before_any_dataset_work(self):
        for seed,limit in ((True,120),(d.SEED,121),(d.SEED,0),(d.SEED,float('nan'))):
            with self.assertRaises(ValueError):d._contract(seed,limit)
        output=Path(os.environ['BIC_DATA_TEST_SCRATCH'])/'must-not-exist'
        TEST_WORK['invalid_build_calls']+=1
        with self.assertRaises(ValueError):d.build(output,old_manifest_sha256='0'*64)
        self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
