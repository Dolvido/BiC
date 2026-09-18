"""Six CPU-only tests on two synthetic pinned tiny archives, not real study data.

Ceilings:30 canonical calls/60 returned rows,40 fixture archive writes,
16 owner prepare calls/96 packed episodes,16 archive decodes,48 target calls.
No learner, optimizer, inference, CUDA, teacher or full648-cache probe.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from experiments import sustained_replay as replay
from experiments import foundation_curriculum as canonical, foundation_layout_curriculum as layout
from experiments import verified_tutor_curriculum as evidence
from experiments.sequence_student import SequenceConfig

WORK=dict(canonical_calls=0,canonical_rows=0,fixture_archive_writes=0,pack_calls=0,packed_episodes=0,
          target_calls=0,target_rows=0,archive_decodes=0,prepare_calls=0)
REPORTS=[]

class ResidentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original=canonical.generate_pair
        def generate(*a,**kw):
            WORK['canonical_calls']+=1
            assert WORK['canonical_calls']<=30
            rows=original(*a,**kw);WORK['canonical_rows']+=len(rows);return rows
        p=patch.object(canonical,'generate_pair',side_effect=generate);p.start();cls.addClassCleanup(p.stop)
        cls.images=[]
        for index,turns in enumerate((8,10)):
            families={family:layout.materialize_pair(canonical.generate_pair(family,852990001+index,depth=0,
                turns=turns,split='train'),layout='original',seed=0) for family in replay.targets.FAMILIES}
            bundle=dict(schema='bic-foundation-layout-bundle-v1',bundle_id=index,layout='original',families=families)
            ev=evidence._evidence(bundle)
            cls.images.append(dict(bundle=bundle,expected_evidence=ev,expected_evidence_sha256=replay._hash(ev)))
        old_pack=replay.preparation.pack_composition_episodes
        def pack(rows,**kw):
            WORK['pack_calls']+=1;assert WORK['pack_calls']<=48
            result=old_pack(rows,**kw);WORK['packed_episodes']+=len(rows);return result
        old_targets=replay.targets.pack_state_targets
        def targets(rows):
            WORK['target_calls']+=1;assert WORK['target_calls']<=48
            result=old_targets(rows);WORK['target_rows']+=len(rows);return result
        old_load=torch.load
        def load(*a,**kw):
            WORK['archive_decodes']+=1;assert WORK['archive_decodes']<=16
            assert kw.get('weights_only') is True and kw.get('map_location')=='cpu'
            return old_load(*a,**kw)
        for obj,name,fn in ((replay.preparation,'pack_composition_episodes',pack),(replay.targets,'pack_state_targets',targets),(torch,'load',load)):
            p=patch.object(obj,name,side_effect=fn);p.start();cls.addClassCleanup(p.stop)

    def fixture(self, *, bad_evidence=False, cap=None):
        tmp=tempfile.TemporaryDirectory(prefix='resident-fixture-',dir=replay.ROOT/'runs');self.addCleanup(tmp.cleanup)
        directory=Path(tmp.name);(directory/'training').mkdir();source=directory/'fixture-source.txt';source.write_text('immutable fixture')
        pins={source.relative_to(replay.ROOT).as_posix():hashlib.sha256(source.read_bytes()).hexdigest()}
        records=[];sizes=[]
        for index,image in enumerate(self.images):
            image=deepcopy(image)
            if bad_evidence and index==0:image['expected_evidence_sha256']='0'*64
            buffer=io.BytesIO();torch.save(image,buffer);raw=buffer.getvalue();name=f'training/{index:04d}.pt'
            (directory/name).write_bytes(raw);WORK['fixture_archive_writes']+=1;assert WORK['fixture_archive_writes']<=40
            sizes.append(len(raw));records.append(dict(path=name,sha256=hashlib.sha256(raw).hexdigest(),global_cursor=index,cycle_id=0,canonical_bundle_id=index))
        manifest=dict(schema=replay.original_data.SCHEMA,updates=2,micro_batch_size=2,layout='original',training=records,source_sha256=pins)
        raw=json.dumps(manifest,sort_keys=True).encode();pin=hashlib.sha256(raw).hexdigest();(directory/'manifest.json').write_bytes(raw)
        (directory/'preparation.json').write_text(json.dumps(dict(status='completed',manifest_sha256=pin)))
        for obj,key,value in ((replay,'UPDATES',2),(replay,'MICRO',2),(replay,'MANIFEST_SHA256',pin)):
            p=patch.object(obj,key,value);p.start();self.addCleanup(p.stop)
        p=patch.object(replay.original_data,'source_hashes',return_value=pins);p.start();self.addCleanup(p.stop)
        reader=replay.ResidentReplay(directory,pin,max_archive_bytes=max(sizes) if cap=='one' else cap or 536870912)
        self.addCleanup(lambda:REPORTS.append(reader.report()))
        return reader,directory,source

    def owner(self):
        owner=replay.preparation.PreparedLayoutOwner(config=SequenceConfig(max_turns=12),layout='original',micro_batch_size=2)
        self.addCleanup(owner.close);return owner

    def prepare(self,r,cursor,owner):
        WORK['prepare_calls']+=1;assert WORK['prepare_calls']<=16
        return r.prepare(cursor,owner)

    def consume(self,owner,token,cursor):
        return owner._consume(token,cursor=cursor,config=owner.config,layout='original',micro_batch_size=2)

    def test_01_repeated_cycle_uses_one_decode_and_targets_and_exact_packing(self):
        reader,_,_=self.fixture();owner=self.owner()
        first,labels,p=self.prepare(reader,2160,owner);batch,ev=self.consume(owner,first,2160)
        second,again,q=self.prepare(reader,2162,owner);other,ev2=self.consume(owner,second,2162)
        self.assertEqual((p['source_index'],q['source_index'],q['replay_cycle']),(0,0,1))
        self.assertEqual((ev['bundle_id'],ev2['bundle_id']),(2160,2162))
        for family in replay.targets.FAMILIES:
            self.assertTrue(torch.equal(labels[family],again[family]))
            for section in batch[family]:
                for name,t in batch[family][section].items():self.assertTrue(torch.equal(t,other[family][section][name]))
        ev2['bundle_id']=2160;self.assertEqual(ev,ev2)
        self.assertEqual(reader.work['archive_loads'],1);self.assertEqual(reader.work['target_completions'],3)
        self.assertEqual(reader.work['cache_hits'],1)

    def test_02_returned_token_targets_and_provenance_cannot_mutate_cache(self):
        reader,_,_=self.fixture();owner=self.owner()
        token,labels,provenance=self.prepare(reader,2160,owner);expected={k:v.clone() for k,v in labels.items()}
        labels['color'].fill_(106);provenance['source_index']=999
        token._batches['color']['inputs']['token_ids'].fill_(0)
        with self.assertRaisesRegex(ValueError,'mutated'):self.consume(owner,token,2160)
        clean=self.owner();token,labels,p=self.prepare(reader,2162,clean);b,_=self.consume(clean,token,2162)
        self.assertEqual(p['source_index'],0)
        for k in labels:self.assertTrue(torch.equal(labels[k],expected[k]))
        self.assertGreater(int(b['color']['inputs']['token_ids'].count_nonzero()),0)
        self.assertEqual(reader.work['archive_loads'],1)

    def test_03_serialized_byte_lru_evicts_then_reauthenticates(self):
        reader,_,_=self.fixture(cap='one');owner=self.owner()
        for cursor in (2160,2161,2162):
            token,_,_=self.prepare(reader,cursor,owner);self.consume(owner,token,cursor)
        r=reader.report();self.assertEqual(r['archive_loads'],3);self.assertEqual(r['evictions'],2)
        self.assertEqual(r['target_completions'],9);self.assertLessEqual(r['cached_archive_bytes'],r['max_archive_bytes'])

    def test_04_archive_and_evidence_tampering_fail_closed(self):
        reader,directory,_=self.fixture();p=directory/'training/0000.pt';p.write_bytes(p.read_bytes()+b'altered')
        count=WORK['archive_decodes']
        with self.assertRaisesRegex(ValueError,'archive bytes'):self.prepare(reader,2160,self.owner())
        self.assertEqual(WORK['archive_decodes'],count);self.assertTrue(reader.failed)
        reader,_,_=self.fixture(bad_evidence=True)
        with self.assertRaisesRegex(ValueError,'evidence'):self.prepare(reader,2160,self.owner())
        self.assertEqual(reader.work['target_attempts'],0)

    def test_05_cursor_owner_pin_and_oversize_refuse_before_decode(self):
        for cursor in (2159,True,2160.0):
            reader,_,_=self.fixture();count=WORK['archive_decodes']
            with self.assertRaises(ValueError):self.prepare(reader,cursor,self.owner())
            self.assertEqual(WORK['archive_decodes'],count)
        reader,directory,_=self.fixture();count=WORK['archive_decodes']
        with self.assertRaises(ValueError):reader.prepare(2160,object())
        with self.assertRaises(ValueError):replay.ResidentReplay(directory,'0'*64)
        reader,_,_=self.fixture(cap=1)
        with self.assertRaisesRegex(ValueError,'exceeds'):self.prepare(reader,2160,self.owner())
        self.assertEqual(WORK['archive_decodes'],count)

    def test_06_explicit_source_boundary_detects_drift_and_poison(self):
        reader,_,source=self.fixture();source.write_text('changed fixture source')
        with self.assertRaisesRegex(ValueError,'source bytes'):reader.authenticate_sources()
        self.assertTrue(reader.failed)
        with self.assertRaisesRegex(RuntimeError,'poisoned'):reader.prepare(2160,self.owner())
        self.assertEqual(reader.work['archive_loads'],0)
