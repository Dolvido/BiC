"""Reference-only chapter rebatching from immutable admitted basis images.

No semantic generation, oracle calls, tensors or teacher transport are used.
Source-file hashes authenticate stored prefix labels; complete reconstructed
canonical JSON must match the original chapter-file SHA256. Cache limits count
source images and serialized bytes, not Python resident memory. Returned images
are independent copies; consumption-cursor remapping remains caller-owned.
"""
from __future__ import annotations

import argparse
from collections import Counter,OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

from experiments import definition_tutor_chapters as chapters
from experiments import definition_basis_data as basis
from experiments.foundation_layout_study import ROOT,native,digest,publish,relative_root,encoded

SCHEMA='bic-referenced-definition-lessons-v1'
CHAPTER_SHA='0084f79dcab6dd5dce6fbace1eba0679d3c8b415a42eee47e7e4ef98e32b78be'
FAMILIES=chapters.FAMILIES
identity=basis.identity


def source_hashes():
    return {**chapters.source_hashes(),'experiments/referenced_definition_lessons.py':digest(Path(__file__))}


def _work():
    return dict(json_read_attempts=0,json_reads=0,json_bytes=0,source_image_loads=0,source_image_checks=0,
        expected_image_checks=0,resolve_attempts=0,resolves=0,reconstructed_pairs=0,reconstructed_rows=0,
        cache_hits=0,cache_misses=0,evictions=0,peak_cached_images=0,peak_cached_serialized_bytes=0,
        failures=0,wall_seconds=0.,cpu_seconds=0.,new_requested_pairs=0,canonical_regenerations=0,
        prefix_interpretations=0,archive_loads=0,models=0,teacher_calls=0)


def _read_json(path,pin,work,inputs):
    basis._pin(pin)
    if path.suffix.lower()!='.json':raise ValueError('only authenticated JSON inputs allowed')
    work['json_read_attempts']+=1;raw=native(path).read_bytes();work['json_bytes']+=len(raw)
    if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('reference source bytes changed')
    inputs[relative_root(path)]=pin
    value=json.loads(raw);work['json_reads']+=1
    return value,len(raw)


def _headers(directory,pin,work,inputs):
    """Metadata only: five JSON reads, no lesson images or tensor archives."""
    if pin!=CHAPTER_SHA:raise ValueError('the declared original chapter catalogue is required')
    manifest,_=_read_json(directory/'manifest.json',pin,work,inputs)
    p=directory/'preparation.json';prep,_=_read_json(p,digest(p),work,inputs)
    if (manifest.get('schema')!=chapters.SCHEMA or manifest.get('source_sha256')!=chapters.source_hashes()
            or manifest.get('source_data',{}).get('sha256')!=chapters.SOURCE_DATA_SHA256
            or manifest.get('updates')!=108 or manifest.get('micro_batch_size')!=32
            or manifest.get('families')!=list(FAMILIES) or manifest.get('skills')!=list(chapters.SKILLS)
            or manifest.get('chapters')!={skill:list(range(18*i,18*(i+1))) for i,skill in enumerate(chapters.SKILLS)}
            or len(manifest.get('training',[]))!=108
            or manifest.get('source_row_target_multiset_sha256')!=manifest.get('compiled_row_target_multiset_sha256')
            or prep.get('status')!='completed' or prep.get('manifest_sha256')!=pin
            or prep.get('source_sha256')!=manifest['source_sha256']):
        raise ValueError('complete admitted source-identical chapters required')
    record=manifest['provenance'];provenance,_=_read_json(basis._path(directory,record['path']),record['sha256'],work,inputs)
    source=manifest['source_data'];source_path=basis._path(ROOT,source['path'])
    original,_=_read_json(source_path,source['sha256'],work,inputs)
    original_prep=source_path.parent/'preparation.json'
    oldprep,_=_read_json(original_prep,manifest['input_sha256'][relative_root(original_prep)],work,inputs)
    if oldprep.get('status')!='completed' or oldprep.get('manifest_sha256')!=source['sha256'] or oldprep.get('source_sha256')!=original.get('source_sha256'):
        raise ValueError('original completed basis admission required')
    records=chapters._source_records(original)
    sources=[dict(record,path=relative_root(basis._path(source_path.parent,record['path']))) for record in records]
    if set(provenance)!={str(i) for i in range(108)}:raise ValueError('exact108 provenance entries required')
    used=Counter()
    for i,record in enumerate(manifest['training']):
        if (record.get('index')!=i or record.get('skill')!=chapters.SKILLS[i//18]
                or record.get('provider')!=basis.curriculum.VERSION or record.get('format')!='json'
                or manifest['artifact_sha256'].get(record['path'])!=basis._pin(record['sha256'])):
            raise ValueError('original chapter record membership differs')
        validate_references(provenance[str(i)],sources)
        used.update((ref['index'],ref['pair_index']) for ref in provenance[str(i)])
    if used!=Counter({(i,p):1 for i in range(108) for p in range(16)}):
        raise ValueError('chapter reference multiset differs from original complete pairs')
    return dict(manifest=manifest,provenance=provenance,source_records=sources)


def validate_references(references,sources):
    if type(references) is not list or len(references)!=16:raise ValueError('sixteen ordered shared-family pair references required')
    seen=set()
    for ref in references:
        if (type(ref) is not dict or set(ref)!={'index','pair_index','image_sha256'}
                or type(ref['index']) is not int or not 0<=ref['index']<len(sources)
                or type(ref['pair_index']) is not int or not 0<=ref['pair_index']<16
                or sources[ref['index']]['index']!=ref['index']
                or sources[ref['index']]['sha256']!=ref['image_sha256']):raise ValueError('pair reference differs from admitted source')
        key=(ref['index'],ref['pair_index'])
        if key in seen:raise ValueError('a complete pair is repeated inside one lesson')
        seen.add(key)
    return True


def authenticate_image(image,record):
    """Identity checks only; semantic truth remains the pinned prior admission."""
    if (type(image) is not dict or set(image)!={'bundle','expected_evidence','expected_evidence_sha256','state_targets','admission_schema'}
            or image['admission_schema']!=basis.curriculum.VERSION):raise ValueError('unchanged basis image required')
    bundle,evidence,labels=image['bundle'],image['expected_evidence'],image['state_targets']
    if (set(bundle)!={'schema','bundle_id','layout','families'} or bundle['schema']!=basis.curriculum.BUNDLE_SCHEMA
            or bundle['layout']!='original' or type(bundle['bundle_id']) is not int or bundle['bundle_id']!=record['index']
            or evidence['bundle_id']!=record['index'] or evidence['layout']!='original'
            or identity(evidence)!=image['expected_evidence_sha256'] or set(bundle['families'])!=set(FAMILIES)
            or set(labels)!=set(FAMILIES) or identity([[f,bundle['families'][f]] for f in FAMILIES])!=evidence['bundle_rows_sha256']):
        raise ValueError('source bundle/row/evidence identity differs')
    for family in FAMILIES:
        rows=bundle['families'][family]
        if len(rows)!=32 or len(labels[family])!=32 or identity(rows)!=evidence['families'][family]['rows_sha256']:
            raise ValueError('complete family row/label identity differs')
        for i in range(0,32,2):
            pair=rows[i:i+2]
            if ([row['variant'] for row in pair]!=[0,1] or pair[0]['recipe']!=pair[1]['recipe']
                    or [row['id'] for row in pair]!=pair[0]['recipe']['base_ids']):raise ValueError('crossed original counterfactual pair')
        for row,state in zip(rows,labels[family]):
            if (row['family']!=family or row['split']!='train' or row['structure_partition']!='train'
                    or len(row['turns'])!=12 or type(state) is not list or len(state)!=12
                    or any(type(turn) is not list or len(turn)!=12 or any(type(v) is not int or not 0<=v<107 for v in turn) for turn in state)):
                raise ValueError('original training partition or aligned prefix labels differ')
    # Original files use this exact canonical serializer, including its newline.
    if hashlib.sha256(encoded(image)).hexdigest()!=record['sha256']:raise ValueError('canonical image bytes differ')
    return True


def compact_record(original_record,references,image):
    authenticate_image(image,original_record)
    if (identity(image['bundle'])!=original_record['bundle_identity_sha256']
            or image['expected_evidence']['bundle_rows_sha256']!=original_record['canonical_rows_sha256']
            or identity(image['state_targets'])!=original_record['canonical_targets_sha256']
            or chapters.multiset_identity(chapters.row_target_multiset(image['bundle']['families'],image['state_targets']))!=original_record['row_target_multiset_sha256']):
        raise ValueError('chapter full row/target declaration differs')
    return dict(original=deepcopy(original_record),references=deepcopy(references),
        bundle_header={k:deepcopy(image['bundle'][k]) for k in ('schema','bundle_id','layout')},
        expected_evidence=deepcopy(image['expected_evidence']),expected_evidence_sha256=image['expected_evidence_sha256'],
        admission_schema=image['admission_schema'])


def _assemble(record,load_source,work):
    rows={f:[] for f in FAMILIES};labels={f:[] for f in FAMILIES}
    for reference in record['references']:
        source=load_source(reference['index']);start=reference['pair_index']*2
        for family in FAMILIES:
            rows[family].extend(deepcopy(source['bundle']['families'][family][start:start+2]))
            labels[family].extend(deepcopy(source['state_targets'][family][start:start+2]))
            work['reconstructed_pairs']+=1;work['reconstructed_rows']+=2
    image=dict(bundle=dict(**deepcopy(record['bundle_header']),families=rows),
        expected_evidence=deepcopy(record['expected_evidence']),expected_evidence_sha256=record['expected_evidence_sha256'],
        state_targets=labels,admission_schema=record['admission_schema'])
    compact_record(record['original'],record['references'],image)
    return image


def build_compact(output,*,chapters_directory,manifest_sha256,max_seconds=120):
    if type(max_seconds) not in (int,float) or not math.isfinite(max_seconds) or not 0<max_seconds<=120:
        raise ValueError('at-most120-second local compilation required')
    output,chapters_directory=Path(output).resolve(),Path(chapters_directory).resolve()
    if any(not p.is_relative_to(ROOT) for p in (output,chapters_directory)):raise ValueError('repository directories required')
    wall,cpu=time.monotonic(),time.process_time();native(output).mkdir(parents=True,exist_ok=False)
    sources,inputs,work=source_hashes(),{},_work();receipt=dict(schema=SCHEMA,status='running',work=work)
    publish(output/'started.json',dict(schema=SCHEMA,source_sha256=sources,max_seconds=max_seconds,chapter_manifest_sha256=manifest_sha256))
    def boundary():
        if time.monotonic()-wall>=max_seconds:raise TimeoutError('compact catalogue allowance ended')
    try:
        header=_headers(chapters_directory,manifest_sha256,work,inputs);training=[]
        for original in header['manifest']['training']:
            boundary();image,_=_read_json(basis._path(chapters_directory,original['path']),original['sha256'],work,inputs)
            training.append(compact_record(original,header['provenance'][str(original['index'])],image));work['expected_image_checks']+=1
        boundary()
        if source_hashes()!=sources:raise ValueError('compact source closure changed')
        manifest=dict(schema=SCHEMA,source_sha256=sources,chapter_catalogue=dict(directory=relative_root(chapters_directory),sha256=manifest_sha256),
            source_data=deepcopy(header['manifest']['source_data']),source_records=header['source_records'],training=training,
            chapters=deepcopy(header['manifest']['chapters']),inventory=deepcopy(header['manifest']['inventory']),input_sha256=inputs,
            expected_image_bytes=sum(native(basis._path(chapters_directory,r['path'])).stat().st_size for r in header['manifest']['training']),
            scope='Reference-only exact rebatching; no new examples, semantic regeneration, policy integration or measured speedup. Original image SHA is checked on every resolve; runtime cursor is unchanged.')
        pin=publish(output/'manifest.json',manifest);receipt.update(status='completed',manifest_sha256=pin,
            compact_manifest_bytes=native(output/'manifest.json').stat().st_size,original_image_bytes=manifest['expected_image_bytes']);return pin
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        work['wall_seconds']+=time.monotonic()-wall;work['cpu_seconds']+=time.process_time()-cpu
        receipt.update(source_sha256=sources,input_sha256=inputs);publish(output/'preparation.json',receipt)


class ChapterLessonStore:
    """A bounded LRU of authenticated source images, never copied chapter files."""
    def __init__(self,directory,*,expected_manifest_sha256,max_source_images=8):
        started,cpu=time.monotonic(),time.process_time();self.work=_work();self.inputs={};self.failed=True
        self._cache=OrderedDict();self._cached_bytes=0
        try:
            if type(max_source_images) is not int or not 1<=max_source_images<=108:raise ValueError('cache capacity1..108 required')
            self.max_source_images=max_source_images;self.directory=Path(directory).resolve()
            if not self.directory.is_relative_to(ROOT):raise ValueError('repository compact manifest required')
            manifest,_=_read_json(self.directory/'manifest.json',expected_manifest_sha256,self.work,self.inputs)
            p=self.directory/'preparation.json';prep,_=_read_json(p,digest(p),self.work,self.inputs)
            self._sources=source_hashes()
            if (manifest.get('schema')!=SCHEMA or manifest.get('source_sha256')!=self._sources
                    or prep.get('status')!='completed' or prep.get('manifest_sha256')!=expected_manifest_sha256
                    or prep.get('source_sha256')!=self._sources):raise ValueError('completed source-identical compact catalogue required')
            origin=manifest['chapter_catalogue'];header=_headers(basis._path(ROOT,origin['directory']),origin['sha256'],self.work,self.inputs)
            if (manifest['source_data']!=header['manifest']['source_data'] or manifest['source_records']!=header['source_records']
                    or manifest['chapters']!=header['manifest']['chapters'] or manifest['inventory']!=header['manifest']['inventory']
                    or len(manifest['training'])!=108):raise ValueError('compact source/order inventory differs')
            for index,record in enumerate(manifest['training']):
                if (record['original']!=header['manifest']['training'][index] or record['references']!=header['provenance'][str(index)]
                        or record['bundle_header']!=dict(schema=basis.curriculum.BUNDLE_SCHEMA,bundle_id=index,layout='original')
                        or record['admission_schema']!=basis.curriculum.VERSION
                        or record['expected_evidence']['bundle_id']!=index
                        or identity(record['expected_evidence'])!=record['expected_evidence_sha256']
                        or record['expected_evidence']['bundle_rows_sha256']!=record['original']['canonical_rows_sha256']):
                    raise ValueError('compact chapter reference/evidence differs')
            self._manifest=manifest;self._manifest_pin=identity(manifest);self.failed=False
        except BaseException:
            self.work['failures']+=1;raise
        finally:
            self.work['wall_seconds']+=time.monotonic()-started;self.work['cpu_seconds']+=time.process_time()-cpu

    @classmethod
    def from_chapters(cls,chapters_directory,*,manifest_sha256,indices,max_source_images=8):
        """Read-only bounded verification before publishing a full compact index.

        Only selected original chapter images are read for expected metadata.
        Resolution still loads exclusively original basis sources. This factory
        neither builds a production compact manifest nor expands its selection.
        """
        if (type(indices) not in (list,tuple) or not indices or len(set(indices))!=len(indices)
                or any(type(i) is not int or not 0<=i<108 for i in indices)
                or type(max_source_images) is not int or not 1<=max_source_images<=108):
            raise ValueError('distinct bounded chapter indices and cache capacity required')
        directory=Path(chapters_directory).resolve()
        if not directory.is_relative_to(ROOT):raise ValueError('repository chapter directory required')
        started,cpu=time.monotonic(),time.process_time();value=object.__new__(cls)
        value.work=_work();value.inputs={};value.failed=True;value._cache=OrderedDict();value._cached_bytes=0
        value.max_source_images=max_source_images;value.directory=directory
        try:
            value._sources=source_hashes();header=_headers(directory,manifest_sha256,value.work,value.inputs);records={}
            for index in indices:
                original=header['manifest']['training'][index]
                image,_=_read_json(basis._path(directory,original['path']),original['sha256'],value.work,value.inputs)
                records[str(index)]=compact_record(original,header['provenance'][str(index)],image)
                value.work['expected_image_checks']+=1
            value._manifest=dict(schema=SCHEMA,source_records=header['source_records'],selected_training=records,
                chapter_catalogue_sha256=manifest_sha256,source_sha256=value._sources)
            value._manifest_pin=identity(value._manifest);value.failed=False;value._guard();return value
        except BaseException:
            value.work['failures']+=1;raise
        finally:
            value.work['wall_seconds']+=time.monotonic()-started;value.work['cpu_seconds']+=time.process_time()-cpu

    def _guard(self):
        if self.failed or source_hashes()!=self._sources or identity(self._manifest)!=self._manifest_pin:
            raise ValueError('poisoned/mutated compact lesson store')

    def _load(self,index):
        if index in self._cache:
            self.work['cache_hits']+=1;self._cache.move_to_end(index);return self._cache[index][0]
        self.work['cache_misses']+=1;record=self._manifest['source_records'][index]
        image,size=_read_json(basis._path(ROOT,record['path']),record['sha256'],self.work,self.inputs)
        self.work['source_image_loads']+=1;authenticate_image(image,record);self.work['source_image_checks']+=1
        while len(self._cache)>=self.max_source_images:
            _,(_,oldsize)=self._cache.popitem(last=False);self._cached_bytes-=oldsize;self.work['evictions']+=1
        self._cache[index]=(image,size);self._cached_bytes+=size
        self.work['peak_cached_images']=max(self.work['peak_cached_images'],len(self._cache))
        self.work['peak_cached_serialized_bytes']=max(self.work['peak_cached_serialized_bytes'],self._cached_bytes)
        return image

    def resolve(self,index):
        started,cpu=time.monotonic(),time.process_time();self.work['resolve_attempts']+=1
        try:
            self._guard()
            if type(index) is not int or not 0<=index<108:raise ValueError('original chapter index0..107 required')
            selected=self._manifest.get('selected_training')
            if selected is not None and str(index) not in selected:raise ValueError('chapter outside authenticated read-only selection')
            record=selected[str(index)] if selected is not None else self._manifest['training'][index]
            value=_assemble(record,self._load,self.work)
            self._guard();self.work['resolves']+=1;return value
        except BaseException:
            self.failed=True;self.work['failures']+=1;raise
        finally:
            self.work['wall_seconds']+=time.monotonic()-started;self.work['cpu_seconds']+=time.process_time()-cpu

    def report(self):
        return dict(schema=SCHEMA,**deepcopy(self.work),failed=self.failed,cached_images=len(self._cache),
            cached_serialized_bytes=self._cached_bytes,max_source_images=self.max_source_images,input_sha256=deepcopy(self.inputs),
            scope='Loaded sources are immutable authenticated byte snapshots. Cache capacity counts images; byte statistics are serialized sizes, not RSS. No global-cursor remapping or learned work.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--chapters-directory',required=True)
    parser.add_argument('--manifest-sha256',default=CHAPTER_SHA);args=parser.parse_args()
    print(json.dumps(dict(manifest_sha256=build_compact(args.output,chapters_directory=args.chapters_directory,manifest_sha256=args.manifest_sha256))))
