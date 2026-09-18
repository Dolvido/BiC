"""Pure synthetic JSON checks; no dataset generation or provider imports."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import immutable_lesson_bytes_cache as cache

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'experiments/complementary_composition_data.py'
# Execute only the two existing pure identity checks, not the provider module.
nodes = [n for n in ast.parse(DATA.read_text()).body
         if isinstance(n, ast.FunctionDef) and n.name in ('learning_identity', 'authenticate_image')]
assert len(nodes) == 2
namespace = dict(identity=cache.identity, FAMILIES=('color','count','switch'))
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(DATA), 'exec'), namespace)
WORK = dict(synthetic_images=0, synthetic_json_bytes=0, callback_attempts=0,
            callback_completions=0, large_synthetic_images=0)
CACHES, RETAINED = [], []


def admission(image, record):
    WORK['callback_attempts'] += 1
    result = namespace['authenticate_image'](image, record)
    RETAINED.append(image)
    WORK['callback_completions'] += 1
    return result


def mutating_admission(image, record):
    WORK['callback_attempts'] += 1
    image['bundle']['bundle_id'] += 1
    WORK['callback_completions'] += 1


def failing_admission(image, record):
    WORK['callback_attempts'] += 1
    raise ValueError('deliberate callback failure')


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='fixtures-', dir=ROOT/'runs/immutable-lesson-bytes-cache-validation-local')
        self.directory = Path(self.tmp.name)
        self.pins = {str(p.relative_to(ROOT)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (Path(__file__), DATA, Path(cache.__file__))}
        self.records, self.images = [], []
        for index in range(3):
            families = {family:[dict(turns=[dict(text=f'{family} fixture {index}',observations=[0],target=0,reply='fixture')])]
                        for family in namespace['FAMILIES']}
            labels = {family:[[[0]*12]] for family in families}
            evidence = dict(bundle_id=index,bundle_rows_sha256=cache.identity([[f,families[f]] for f in namespace['FAMILIES']]))
            image = dict(bundle=dict(bundle_id=index,families=families),expected_evidence=evidence,
                expected_evidence_sha256=cache.identity(evidence),state_targets=labels,admission_schema='synthetic-provider')
            raw = cache._encoded(image)+b'\n';path=self.directory/f'{index}.json';path.write_bytes(raw)
            self.records.append(dict(index=index,format='json',kind='basis',provider='synthetic-provider',
                path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),
                learning_identity_sha256=namespace['learning_identity'](image)))
            self.images.append(image);WORK['synthetic_images']+=1;WORK['synthetic_json_bytes']+=len(raw)

    def tearDown(self): self.tmp.cleanup()

    def make(self, *, records=None, callback=admission, **kwargs):
        records = self.records if records is None else records
        value = cache.ImmutableLessonBytesCache(ROOT,records=records,records_sha256=cache.identity(records),
            admission=callback,admission_source_sha256=self.pins,**kwargs)
        CACHES.append(value);return value

    def test_detached_exact_snapshot_and_admission_once(self):
        value=self.make();first=value.get(self.records[0]);self.assertEqual(first,self.images[0])
        first['state_targets']['color'][0][0][0]=106
        first['bundle']['bundle_id']=991
        RETAINED[-1]['expected_evidence']['bundle_id']=998
        before=value.report();again=value.get(self.records[0]);after=value.report()
        self.assertEqual(again,self.images[0]);self.assertIsNot(first,again)
        self.assertEqual(after['admissions'],1);self.assertEqual(after['decodes'],3)
        self.assertEqual(after['admission_decodes'],1);self.assertEqual(after['return_decodes'],2)
        self.assertEqual(first['expected_evidence']['bundle_id'],0)
        self.assertTrue(all(type(entry[0]) is bytes for entry in value._cache.values()))
        self.assertEqual(after['cache_hits'],1);self.assertEqual(after['source_file_hashes'],before['source_file_hashes'])

    def test_lru_entry_bound_and_reread_after_eviction(self):
        value=self.make(max_images=2)
        for index in (0,1,0,2,1):self.assertEqual(value.get(self.records[index]),self.images[index])
        work=value.report();self.assertEqual((work['cache_hits'],work['cache_misses'],work['evictions']),(1,4,2))
        self.assertEqual(work['peak_images'],2);self.assertEqual(work['admissions'],4)
        self.assertEqual(work['decodes'],9);self.assertEqual(work['return_decodes'],5)

    def test_retained_byte_bound_and_oversize(self):
        size=(ROOT/self.records[0]['path']).stat().st_size
        value=self.make(max_images=3,max_serialized_bytes=size*2-1)
        value.get(self.records[0]);value.get(self.records[1])
        self.assertEqual(value.report()['cached_images'],1);self.assertLessEqual(value.report()['peak_serialized_bytes'],size*2-1)
        small=self.make(max_serialized_bytes=16)
        with self.assertRaises(ValueError):small.get(self.records[0])
        self.assertEqual(small.report()['json_bytes'],17);self.assertEqual(small.report()['decodes'],0)

    def test_disk_change_keeps_snapshot_until_explicit_boundary(self):
        value=self.make();value.get(self.records[0]);value.verify_sources()
        (ROOT/self.records[0]['path']).write_text('{}')
        self.assertEqual(value.get(self.records[0]),self.images[0])
        with self.assertRaises(ValueError):value.verify_sources()
        with self.assertRaises(ValueError):value.get(self.records[0])
        self.assertEqual(value.report()['verifications'],1);self.assertEqual(value.report()['admissions'],1)

    def test_mutated_record_and_file_pin_fail_before_decode(self):
        value=self.make();bad=deepcopy(self.records[0]);bad['index']=8
        with self.assertRaises(ValueError):value.get(bad)
        self.assertEqual(value.report()['json_read_attempts'],0)
        records=deepcopy(self.records);records[0]['sha256']='0'*64;value=self.make(records=records)
        with self.assertRaises(ValueError):value.get(records[0])
        self.assertEqual(value.report()['decodes'],0);self.assertGreater(value.report()['json_bytes'],0)

    def test_callback_failure_mutation_and_target_rejection(self):
        for callback,unknown in ((failing_admission,True),(mutating_admission,False)):
            value=self.make(callback=callback)
            with self.assertRaises(ValueError):value.get(self.records[0])
            self.assertEqual(value.report()['callback_partial_work_unknown'],unknown)
            self.assertEqual(value.report()['gets'],0)
        image=deepcopy(self.images[0]);image['state_targets']['color'][0][0][0]=106
        raw=cache._encoded(image);(ROOT/self.records[0]['path']).write_bytes(raw)
        records=deepcopy(self.records);records[0]['sha256']=hashlib.sha256(raw).hexdigest();value=self.make(records=records)
        with self.assertRaises(ValueError):value.get(records[0])
        self.assertEqual(value.report()['admission_attempts'],1);self.assertEqual(value.report()['admissions'],0)

    def test_source_guard_is_boundary_owned_and_record_list_detached(self):
        guard=self.directory/'guard.py';guard.write_text('version=1\n')
        self.pins[guard.relative_to(ROOT).as_posix()]=hashlib.sha256(guard.read_bytes()).hexdigest()
        records=deepcopy(self.records);value=self.make(records=records);value.get(self.records[0])
        records[0]['index']=999;guard.write_text('version=2\n')
        self.assertEqual(value.get(self.records[0]),self.images[0])
        with self.assertRaises(ValueError):value.verify_sources()
        self.assertTrue(value.report()['failed'])

    def test_strict_json_paths_and_configuration(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}'):
            (ROOT/self.records[0]['path']).write_bytes(raw);records=deepcopy(self.records)
            records[0]['sha256']=hashlib.sha256(raw).hexdigest();value=self.make(records=records)
            with self.assertRaises(ValueError):value.get(records[0])
            self.assertEqual(value.report()['admission_attempts'],0)
        for key,bad in (('path','../outside.json'),('format','torch_weights_only'),('index',True)):
            records=deepcopy(self.records);records[0][key]=bad
            with self.assertRaises(ValueError):self.make(records=records)
        for kwargs in (dict(max_images=0),dict(max_images=True),dict(max_serialized_bytes=0)):
            with self.assertRaises(ValueError):self.make(**kwargs)

    def test_production_chunk_bound_on_public_load_and_boundary(self):
        # One explicit synthetic >1MiB JSON image, not a production lesson.
        image=deepcopy(self.images[0]);image['bundle']['families']['color'][0]['turns'][0]['text']='x'*(cache.READ_CHUNK_BYTES+23)
        image['expected_evidence']['bundle_rows_sha256']=cache.identity([[f,image['bundle']['families'][f]] for f in namespace['FAMILIES']])
        image['expected_evidence_sha256']=cache.identity(image['expected_evidence'])
        raw=cache._encoded(image)+b'\n';target=ROOT/self.records[0]['path'];target.write_bytes(raw)
        records=deepcopy(self.records);records[0]['sha256']=hashlib.sha256(raw).hexdigest()
        records[0]['learning_identity_sha256']=namespace['learning_identity'](image)
        WORK['large_synthetic_images']+=1;WORK['synthetic_json_bytes']+=len(raw)
        value=self.make(records=records,max_serialized_bytes=4*cache.READ_CHUNK_BYTES)
        requests=[];original=Path.open
        class Reader:
            def __init__(self,handle):self.handle=handle
            def __enter__(self):return self
            def __exit__(self,*args):self.handle.close()
            def read(self,n):requests.append(n);return self.handle.read(n)
        def opening(path,*args,**kwargs):
            handle=original(path,*args,**kwargs)
            return Reader(handle) if path.name==target.name else handle
        with patch.object(Path,'open',opening):
            self.assertEqual(value.get(records[0]),image);value.verify_sources()
        self.assertEqual(len(requests),6)
        self.assertTrue(all(0<n<=cache.READ_CHUNK_BYTES for n in requests))
        self.assertEqual(value.report()['chunk_reads'],6)
        self.assertEqual(value.report()['resident_check_bytes'],len(raw))
        self.assertEqual(value.report()['json_bytes'],len(raw))
        self.assertEqual(value.report()['max_chunk_request_bytes'],cache.READ_CHUNK_BYTES)


if __name__=='__main__': unittest.main()
