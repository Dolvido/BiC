"""Pure fixed-schedule, pin, label-alignment and exclusion checks."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import unittest

from experiments import complementary_composition_data as d

TEST_WORK=dict(schedule_checks=0,training_seed_checks=0,development_seed_checks=0,catalogue_checks=0,
    learning_identity_checks=0,envelope_checks=0,exclusion_checks=0,protection_checks=0,
    tiny_json_read_attempts=0,tiny_json_reads=0,synthetic_images=0,synthetic_rows=0)


def catalogues():
    result={}
    for name in ('basis','atomic','old_definition','replay','complementary'):
        kind='basis' if name=='atomic' else name
        result[name]=[]
        for i in range(180 if name=='complementary' else 108):
            record=dict(index=i,path=f'runs/synthetic-complementary/{name}/{i:03d}.json',sha256=d.identity([name,i]),kind=kind,
                provider=d.curriculum.VERSION if kind=='complementary' else d.basis.PROVIDERS[kind],
                format='torch_weights_only' if kind=='replay' else 'json')
            if kind=='replay':record['original_cursor']=400+i
            else:record['learning_identity_sha256']=d.identity(['learning',name,i])
            if kind=='complementary':record['panel'],record['contrast_index']=d._coordinate(i)
            result[name].append(record)
    return result


def image():
    rows={f:[dict(id=f'{f}/{i}',turns=[dict(text=f'Synthetic {f} {i} {t}.',observations=[],target=i,reply=('No.','Yes.')[i])
        for t in range(2)]) for i in range(2)] for f in d.FAMILIES}
    targets={f:[[[i]*12,[i+1]*12] for i in range(2)] for f in d.FAMILIES}
    evidence=dict(bundle_id=0,bundle_rows_sha256=d.identity([[f,rows[f]] for f in d.FAMILIES]))
    TEST_WORK['synthetic_images']+=1;TEST_WORK['synthetic_rows']+=6
    return dict(bundle=dict(bundle_id=0,families=rows),expected_evidence=evidence,expected_evidence_sha256=d.identity(evidence),
        state_targets=targets,admission_schema=d.curriculum.VERSION)


class ComplementaryDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.image=image()

    def test_schedule_exact_shared_rehearsal_and_declared_exposures(self):
        schedule=d.schedule();TEST_WORK['schedule_checks']+=1
        for arm in schedule:self.assertEqual(len(schedule[arm]),648)
        self.assertEqual(Counter(r['kind'] for r in schedule['control']),dict(basis=216,replay=324,old_definition=108))
        self.assertEqual(Counter(r['kind'] for r in schedule['curriculum']),dict(basis=36,complementary=180,replay=324,old_definition=108))
        for a,b in zip(schedule['control'],schedule['curriculum']):
            if a['kind']!='basis':self.assertEqual(a,b)
        for arm in schedule:
            for kind,times in (('replay',3),('old_definition',1)):
                self.assertEqual(Counter(r['source_index'] for r in schedule[arm] if r['kind']==kind),Counter({i:times for i in range(108)}))

    def test_atomic_and_complementary_catalogues_distributed_as_declared(self):
        schedule=d.schedule();TEST_WORK['schedule_checks']+=1
        control=[r for r in schedule['control'] if r['kind']=='basis']
        self.assertEqual([r['source_index'] for r in control],[k%108 for k in range(216)])
        learning=[r for r in schedule['curriculum'] if r['kind'] in ('basis','complementary')]
        atomic=[r['source_index'] for r in learning if r['kind']=='basis']
        self.assertEqual(atomic,[(j%6)*18+j//6 for j in range(36)])
        self.assertEqual(Counter(i//18 for i in atomic),Counter({i:6 for i in range(6)}))
        self.assertEqual([i for i,r in enumerate(learning) if r['kind']=='basis'],list(range(0,216,6)))
        complementary=[r['source_index'] for r in learning if r['kind']=='complementary']
        self.assertEqual(complementary,list(range(180)))
        self.assertEqual(Counter(i%10 for i in complementary),Counter({i:18 for i in range(10)}))

    def test_seed_coordinates_are_unique_disjoint_and_obey_contrast(self):
        train=[];dev=[]
        for i in range(180):
            for pair in range(16):
                value=d.training_seed(d.SEED,i,pair);TEST_WORK['training_seed_checks']+=1;train.append(value)
                self.assertEqual(value%5,d._coordinate(i)[1])
        for panel in d.PANELS:
            for contrast in range(5):
                for pair in range(8):
                    value=d.development_seed(d.SEED,panel,contrast,pair);TEST_WORK['development_seed_checks']+=1;dev.append(value)
                    self.assertEqual(value%5,contrast)
        self.assertEqual(len(set(train)),2880);self.assertEqual(len(set(dev)),80);self.assertFalse(set(train)&set(dev))
        for args in ((d.SEED,180,0),(d.SEED,0,16),(True,0,0)):
            with self.assertRaises(ValueError):d.training_seed(*args)

    def test_catalogue_shape_provider_and_cursor_admission(self):
        value=catalogues();TEST_WORK['catalogue_checks']+=1;self.assertTrue(d.validate_catalogues(value))
        expanded=d.expand_schedule(value)
        self.assertEqual(expanded['control'][0],value['basis'][0]);self.assertEqual(expanded['curriculum'][0],value['atomic'][0])
        self.assertEqual(expanded['curriculum'][4],value['complementary'][0])
        for mutation in ('size','provider','cursor','contrast','path'):
            bad=deepcopy(value)
            if mutation=='size':bad['complementary'].pop()
            elif mutation=='provider':bad['complementary'][0]['provider']=d.basis.curriculum.VERSION
            elif mutation=='cursor':bad['replay'][0]['original_cursor']=-1
            elif mutation=='contrast':bad['complementary'][0]['contrast_index']=1
            else:bad['basis'][0]['path']='../outside.json'
            TEST_WORK['catalogue_checks']+=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):d.validate_catalogues(bad)

    def test_ordered_learning_identity_includes_labels_not_provenance(self):
        expected=d.learning_identity(self.image);TEST_WORK['learning_identity_checks']+=1
        for mutation in ('text','target','reply','observations','labels','row_order'):
            value=deepcopy(self.image);turn=value['bundle']['families']['color'][0]['turns'][0]
            if mutation=='text':turn['text']='changed'
            elif mutation=='target':turn['target']=2
            elif mutation=='reply':turn['reply']='changed'
            elif mutation=='observations':turn['observations']=[1]
            elif mutation=='labels':value['state_targets']['color'].reverse()
            else:value['bundle']['families']['color'].reverse()
            TEST_WORK['learning_identity_checks']+=1;self.assertNotEqual(d.learning_identity(value),expected)
        value=deepcopy(self.image);value['bundle']['families']['color'][0]['id']='differentmetadata'
        TEST_WORK['learning_identity_checks']+=1;self.assertEqual(d.learning_identity(value),expected)
        value['state_targets']['color'][0][0][0]=True
        TEST_WORK['learning_identity_checks']+=1
        with self.assertRaises(ValueError):d.learning_identity(value)

    def test_evidence_envelope_and_hashed_json_reject_target_crossing(self):
        value=deepcopy(self.image);record=dict(index=0,provider=d.curriculum.VERSION,kind='complementary',format='json',
            learning_identity_sha256=d.learning_identity(value))
        TEST_WORK['envelope_checks']+=1;self.assertEqual(d.authenticate_image(value,record),record['learning_identity_sha256'])
        for mutation in ('target','evidence','provider'):
            wrong=deepcopy(value);r=deepcopy(record)
            if mutation=='target':wrong['state_targets']['switch'][0][0][0]=106
            elif mutation=='evidence':wrong['expected_evidence']['bundle_id']=1
            else:r['provider']='wrong'
            TEST_WORK['envelope_checks']+=1
            with self.assertRaises(ValueError):d.authenticate_image(wrong,r)
        path=Path(os.environ['BIC_COMPLEMENTARY_DATA_TEST_DIR'])/'synthetic.json';raw=json.dumps(value).encode()
        with path.open('xb') as out:out.write(raw)
        record.update(path=path.relative_to(d.ROOT).as_posix(),sha256=hashlib.sha256(raw).hexdigest())
        TEST_WORK['tiny_json_read_attempts']+=1;self.assertEqual(d.load_json_image(record),value);TEST_WORK['tiny_json_reads']+=1
        record['sha256']='f'*64;TEST_WORK['tiny_json_read_attempts']+=1
        with self.assertRaisesRegex(ValueError,'pinned input changed'):d.load_json_image(record)

    def test_exact_protected_and_train_dev_overlap_is_rejected(self):
        rows=self.image['bundle']['families']['color'];hashes=[d.transcript(r) for r in rows]
        TEST_WORK['exclusion_checks']+=1;self.assertEqual(d._fingerprints(rows,set()),hashes)
        for protected,other in (({hashes[0]},set()),(set(),{hashes[1]})):
            TEST_WORK['exclusion_checks']+=1
            with self.assertRaises(ValueError):d._fingerprints(rows,protected,other=other)

    def test_protection_export_exact_bank_and_source_binding(self):
        evaluation=dict(banks=dict(path='banks.pt',sha256='a'*64),bank_inventory=dict(dev=dict(episodes=2),retention=dict(episodes=1)))
        export=dict(schema='bic-complementary-protected-transcripts-v1',manifest_sha256=d.EVALUATION['manifest_sha256'],
            source=dict(path=d.EVALUATION['directory']+'/banks.pt',sha256='a'*64),by_bank=dict(dev=['b'*64,'c'*64],retention=['d'*64]))
        TEST_WORK['protection_checks']+=1;self.assertEqual(d._current_protection(export,evaluation),{'b'*64,'c'*64,'d'*64})
        for mutation in ('source','missing','count','pin'):
            bad=deepcopy(export)
            if mutation=='source':bad['source']['sha256']='f'*64
            elif mutation=='missing':del bad['by_bank']['retention']
            elif mutation=='count':bad['by_bank']['dev'].pop()
            else:bad['by_bank']['dev'][0]='bad'
            TEST_WORK['protection_checks']+=1
            with self.assertRaises(ValueError):d._current_protection(bad,evaluation)


if __name__=='__main__':unittest.main()
