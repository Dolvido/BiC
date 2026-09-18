"""Verified chapter ordering over unchanged, already admitted basis lessons.

An external author chooses only a prerequisite-respecting permutation. The
catalogue has one common row/target multiset for every order. Rebatching produces
new bundle identities; original pair, row, recipe and target identities persist.
No held-out bank is a lesson source. No tensor archive or model is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

from experiments import definition_basis_curriculum as curriculum
from experiments import definition_basis_data as basis
from experiments.foundation_layout_study import ROOT,native,digest,publish,relative_root,encoded

SCHEMA='bic-definition-tutor-chapter-data-v1'
CONTRACT_SCHEMA='bic-definition-tutor-chapter-contract-v1'
RECIPE_SCHEMA='bic-definition-tutor-chapters-v1'
SOURCE_DATA_SHA256='3ae5d0d93685675c247c68aaa7a87254abc9ad32e59782b1e34f353c6b5e285b'
SKILLS=('bind_c_s','bind_s_d','bind_c_d','revise_c_s','revise_s_d','revise_c_d')
FAMILIES=curriculum.FAMILIES
UPDATES,CHAPTER_UPDATES,MICRO=108,18,32
identity=basis.identity


def source_hashes():
    return {**basis.source_hashes(),'experiments/definition_tutor_chapters.py':digest(Path(__file__))}


def make_contract(catalogue_manifest_sha256):
    return validate_contract(dict(schema=CONTRACT_SCHEMA,catalogue_manifest_sha256=catalogue_manifest_sha256,
        source_data_manifest_sha256=SOURCE_DATA_SHA256,updates=108,chapter_updates=18,micro_batch_size=32,
        families=list(FAMILIES),skills=list(SKILLS)))


def validate_contract(contract):
    fields={'schema','catalogue_manifest_sha256','source_data_manifest_sha256','updates','chapter_updates',
            'micro_batch_size','families','skills'}
    if (type(contract) is not dict or set(contract)!=fields or contract['schema']!=CONTRACT_SCHEMA
            or contract['source_data_manifest_sha256']!=SOURCE_DATA_SHA256
            or any(type(contract[name]) is not int or contract[name]!=value for name,value in
                (('updates',108),('chapter_updates',18),('micro_batch_size',32)))
            or contract['families']!=list(FAMILIES) or contract['skills']!=list(SKILLS)):
        raise ValueError('exact fixed six-chapter contract required')
    basis._pin(contract['catalogue_manifest_sha256'])
    return deepcopy(contract)


def validate_recipe(recipe,contract):
    validate_contract(contract)
    if (type(recipe) is not dict or set(recipe)!={'schema','chapters'} or recipe['schema']!=RECIPE_SCHEMA
            or type(recipe['chapters']) is not list or len(recipe['chapters'])!=6
            or any(type(skill) is not str for skill in recipe['chapters']) or set(recipe['chapters'])!=set(SKILLS)):
        raise ValueError('exact six-skill permutation required')
    order=recipe['chapters']
    if any(order.index(SKILLS[i])>order.index(SKILLS[i+3]) for i in range(3)):
        raise ValueError('each binding chapter must precede its matching revision')
    return deepcopy(recipe)


def procedural_recipe(contract):
    return validate_recipe(dict(schema=RECIPE_SCHEMA,chapters=list(SKILLS)),contract)


def recipe_schema(contract):
    validate_contract(contract)
    return dict(type='object',additionalProperties=False,required=['schema','chapters'],properties={
        'schema':dict(type='string',const=RECIPE_SCHEMA),
        'chapters':dict(type='array',minItems=6,maxItems=6,uniqueItems=True,items=dict(type='string',enum=list(SKILLS)))})


def authoring_catalogue(contract):
    validate_contract(contract)
    contrasts=('copy versus advance-source','advance-source versus advance-destination','copy versus advance-destination')
    return dict(skills=[dict(id=skill,description=('Named-rule binding: ' if i<3 else 'Nonretroactive rule revision: ')+contrasts[i%3],
            prerequisites=[] if i<3 else [SKILLS[i-3]],updates=18,families=list(FAMILIES)) for i,skill in enumerate(SKILLS)],
        constraints='Choose each chapter once; binding precedes its matching revision. All orders use identical admitted lessons and targets.',
        default_chapters=list(SKILLS),scope='Order shared practice only; no lesson writing, answer access, held-out compositions or sequence training.')


def ordered_indices(recipe,contract,manifest):
    recipe=validate_recipe(recipe,contract)
    if (manifest.get('schema')!=SCHEMA or manifest.get('source_data',{}).get('sha256')!=SOURCE_DATA_SHA256
            or hashlib.sha256(encoded(manifest)).hexdigest()!=contract['catalogue_manifest_sha256']
            or manifest.get('chapters')!={skill:list(range(i*18,(i+1)*18)) for i,skill in enumerate(SKILLS)}):
        raise ValueError('author-bound fixed chapter catalogue required')
    indices=[index for skill in recipe['chapters'] for index in manifest['chapters'][skill]]
    if len(indices)!=108 or sorted(indices)!=list(range(108)):raise ValueError('every immutable lesson must occur exactly once')
    return indices


def row_target_multiset(families,targets):
    """Binds each target to its original full row, preserving multiplicity."""
    if set(families)!=set(FAMILIES) or set(targets)!=set(FAMILIES):raise ValueError('all three matched families required')
    counts=Counter()
    for family in FAMILIES:
        if len(families[family])!=len(targets[family]):raise ValueError('row/target lengths differ')
        for row,label in zip(families[family],targets[family]):
            counts[identity([family,row,label])]+=1
    return counts


def multiset_identity(counter):return identity(sorted(counter.items()))


def _work():
    return dict(json_read_attempts=0,json_reads=0,input_bytes=0,source_images=0,source_pairs=0,
        prefix_label_checks=0,regrouped_images=0,regrouped_pairs=0,new_requested_pairs=0,
        archive_loads=0,models=0,training_updates=0,teacher_calls=0)


def _read_json(path,pin,work):
    basis._pin(pin);work['json_read_attempts']+=1
    if path.suffix.lower()!='.json':raise ValueError('chapter input must be pinned JSON')
    raw=native(path).read_bytes();work['input_bytes']+=len(raw)
    if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('chapter input bytes changed')
    value=json.loads(raw);work['json_reads']+=1;return value


def _source_records(manifest):
    if (manifest.get('schema')!=basis.SCHEMA or manifest.get('definition_schema')!=curriculum.VERSION
            or manifest.get('micro_batch_size')!=32 or len(manifest.get('training',[]))!=108
            or manifest.get('source_sha256')!=basis.source_hashes()
            or set(manifest['training_inventory']['pairs_by_family_panel_contrast'].values())!={288}
            or len(manifest['training_inventory']['pairs_by_family_panel_contrast'])!=18):
        raise ValueError('the fixed admitted basis catalogue is required')
    records=[]
    for index,record in enumerate(manifest['training']):
        if (record.get('index')!=index or record.get('panel')!=('binding','revision')[index%2]
                or record.get('provider')!=curriculum.VERSION or record.get('type')!='basis' or record.get('format')!='json'
                or manifest['artifact_sha256'].get(record['path'])!=basis._pin(record['sha256'])):
            raise ValueError('basis training source record differs')
        records.append(deepcopy(record))
    return records


def extract_pairs(image,record,work):
    """Source image must already have passed its externally pinned byte hash."""
    if (type(image) is not dict or set(image)!={'bundle','expected_evidence','expected_evidence_sha256','state_targets','admission_schema'}
            or image['admission_schema']!=curriculum.VERSION):raise ValueError('exact original basis image required')
    bundle,evidence,targets=image['bundle'],image['expected_evidence'],image['state_targets']
    if (set(bundle)!={'schema','bundle_id','layout','families'} or bundle['schema']!=curriculum.BUNDLE_SCHEMA
            or bundle['layout']!='original' or bundle['bundle_id']!=record['index'] or evidence['bundle_id']!=record['index']
            or identity(evidence)!=image['expected_evidence_sha256'] or evidence['layout']!='original'
            or set(bundle['families'])!=set(FAMILIES) or set(targets)!=set(FAMILIES)
            or identity([[f,bundle['families'][f]] for f in FAMILIES])!=evidence['bundle_rows_sha256']):
        raise ValueError('source bundle identity differs')
    for family in FAMILIES:
        rows=bundle['families'][family]
        if len(rows)!=32 or len(targets[family])!=32 or identity(rows)!=evidence['families'][family]['rows_sha256']:
            raise ValueError('source family/target size or hash differs')
        for row,labels in zip(rows,targets[family]):
            work['prefix_label_checks']+=1
            if (type(labels) is not list or len(labels)!=12 or any(type(turn) is not list or len(turn)!=12
                    or any(type(value) is not int or not 0<=value<=106 for value in turn) for turn in labels)
                    or labels!=curriculum.prefix_labels(row)):
                raise ValueError('source target is not its exact integer causal English prefix state')
    packets=[]
    for pair_index in range(16):
        families={};labels={};seeds=set();panels=set()
        for family in FAMILIES:
            pair=bundle['families'][family][2*pair_index:2*pair_index+2]
            if (any(row['family']!=family or row['split']!='train' or row['structure_partition']!='train' for row in pair)
                    or pair[0]['recipe']!=pair[1]['recipe'] or [row['variant'] for row in pair]!=[0,1]):
                raise ValueError('complete aligned admitted training pair required')
            recipe=pair[0]['recipe'];seed=recipe['seed'];panel=recipe['panel']
            if (panel not in ('binding','revision') or panel!=record['panel'] or recipe['basis_pair_class']!=seed%3):
                raise ValueError('training panel/contrast differs')
            seeds.add(seed);panels.add(panel);families[family]=deepcopy(pair)
            labels[family]=deepcopy(targets[family][2*pair_index:2*pair_index+2])
        if len(seeds)!=1 or len(panels)!=1:raise ValueError('shared skill and source coordinates across all families required')
        contrast=next(iter(seeds))%3;skill=SKILLS[contrast+(3 if record['panel']=='revision' else 0)]
        packets.append(dict(skill=skill,families=families,state_targets=labels,
            row_target_multiset_sha256=multiset_identity(row_target_multiset(families,labels)),
            source=dict(index=record['index'],pair_index=pair_index,image_sha256=record['sha256'])))
        work['source_pairs']+=3
    work['source_images']+=1
    return packets


def compile_image(packets,index,work):
    if (type(index) is not int or index<0 or not 1<=len(packets)<=16
            or len({packet['skill'] for packet in packets})!=1):raise ValueError('one skill and bounded complete pairs required')
    for packet in packets:
        if multiset_identity(row_target_multiset(packet['families'],packet['state_targets']))!=packet['row_target_multiset_sha256']:
            raise ValueError('authenticated source packet row/target association changed')
    families={f:[deepcopy(row) for packet in packets for row in packet['families'][f]] for f in FAMILIES}
    labels={f:[deepcopy(value) for packet in packets for value in packet['state_targets'][f]] for f in FAMILIES}
    bundle=dict(schema=curriculum.BUNDLE_SCHEMA,bundle_id=index,layout='original',families=families)
    evidence=curriculum.bundle_evidence(bundle)
    before=Counter()
    for packet in packets:before.update(row_target_multiset(packet['families'],packet['state_targets']))
    if before!=row_target_multiset(families,labels):raise ValueError('rebatching changed a row/target association')
    work['regrouped_images']+=1;work['regrouped_pairs']+=3*len(packets)
    return dict(bundle=bundle,expected_evidence=evidence,expected_evidence_sha256=identity(evidence),
        state_targets=labels,admission_schema=curriculum.VERSION)


def build(output,*,data_directory,data_manifest_sha256,max_seconds=120):
    if (data_manifest_sha256!=SOURCE_DATA_SHA256 or type(max_seconds) not in (int,float)
            or not math.isfinite(max_seconds) or not 0<max_seconds<=120):raise ValueError('fixed source and at-most120-second preparation required')
    output,data_directory=Path(output).resolve(),Path(data_directory).resolve()
    if not output.is_relative_to(ROOT) or not data_directory.is_relative_to(ROOT):raise ValueError('repository directories required')
    wall,cpu=time.monotonic(),time.process_time();native(output).mkdir(parents=True,exist_ok=False)
    sources,inputs,artifacts=source_hashes(),{},{};before=curriculum.work_report();work=_work()
    receipt=dict(schema=SCHEMA,status='running',work=work)
    publish(output/'started.json',dict(schema=SCHEMA,source_sha256=sources,source_data_manifest_sha256=data_manifest_sha256,
        max_seconds=max_seconds,scope='Rebatch exact admitted rows and targets; zero newly sampled pairs.'))
    def boundary():
        if time.monotonic()-wall>=max_seconds:raise TimeoutError('chapter compilation allowance ended')
    def read(path,pin):
        boundary();value=_read_json(path,pin,work);inputs[relative_root(path)]=pin;return value
    def save(name,value):
        boundary();pin=publish(output/name,value);artifacts[name]=pin;return dict(path=name,sha256=pin)
    try:
        data=read(data_directory/'manifest.json',data_manifest_sha256)
        preparation_path=data_directory/'preparation.json';preparation=read(preparation_path,digest(preparation_path))
        if (preparation.get('status')!='completed' or preparation.get('manifest_sha256')!=data_manifest_sha256
                or preparation.get('source_sha256')!=data.get('source_sha256')):raise ValueError('completed source preparation required')
        pools={skill:[] for skill in SKILLS};source_counts=Counter()
        for record in _source_records(data):
            image=read(basis._path(data_directory,record['path']),record['sha256'])
            source_counts.update(row_target_multiset(image['bundle']['families'],image['state_targets']))
            for packet in extract_pairs(image,record,work):pools[packet['skill']].append(packet)
        if any(len(pool)!=288 for pool in pools.values()):raise ValueError('288 shared pair coordinates per skill required')
        training=[];chapters={};provenance={};compiled_counts=Counter()
        for skill in SKILLS:
            chapters[skill]=[]
            for offset in range(18):
                boundary();index=len(training);packets=pools[skill][offset*16:(offset+1)*16]
                image=compile_image(packets,index,work);counts=row_target_multiset(image['bundle']['families'],image['state_targets'])
                compiled_counts.update(counts);chapters[skill].append(index)
                training.append(dict(**save(f'training/{index:03d}.json',image),index=index,skill=skill,
                    provider=curriculum.VERSION,format='json',bundle_identity_sha256=identity(image['bundle']),
                    canonical_rows_sha256=image['expected_evidence']['bundle_rows_sha256'],
                    canonical_targets_sha256=identity(image['state_targets']),row_target_multiset_sha256=multiset_identity(counts)))
                provenance[str(index)]=[deepcopy(packet['source']) for packet in packets]
        if source_counts!=compiled_counts or sum(source_counts.values())!=10368:
            raise ValueError('complete source and compiled row/target multisets must agree exactly')
        provenance_record=save('provenance.json',provenance)
        multiset_record=save('row-target-multiset.json',sorted(source_counts.items()))
        for name,pin in inputs.items():
            boundary()
            if digest(basis._path(ROOT,name))!=pin:raise ValueError('chapter source bytes changed during compilation')
        boundary()
        if source_hashes()!=sources:raise ValueError('chapter compiler source changed')
        manifest=dict(schema=SCHEMA,source_data=dict(path=relative_root(data_directory/'manifest.json'),sha256=data_manifest_sha256),
            updates=108,chapter_updates=18,micro_batch_size=32,families=list(FAMILIES),skills=list(SKILLS),
            training=training,chapters=chapters,provenance=provenance_record,row_target_multiset=multiset_record,
            source_row_target_multiset_sha256=multiset_identity(source_counts),compiled_row_target_multiset_sha256=multiset_identity(compiled_counts),
            inventory=dict(episodes=10368,pairs=5184,unique_row_target_records=len(source_counts),
                by_skill={skill:dict(updates=18,episodes=1728,pairs=864,family_episodes={f:576 for f in FAMILIES}) for skill in SKILLS}),
            source_sha256=sources,input_sha256=inputs,artifact_sha256=artifacts,
            replay_scope='Explicit reuse of all original training rows and targets, preserving multiplicity; no novelty requirement.',
            identity_scope='New grouped bundle identities; unchanged source row, parent recipe and target identities. Consumption cursor remapping is external.')
        pin=publish(output/'manifest.json',manifest);receipt.update(status='completed',manifest_sha256=pin);return pin
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        after=curriculum.work_report()
        receipt.update(wall_seconds=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,
            source_sha256=sources,input_sha256=inputs,artifact_sha256=artifacts,
            curriculum_work={key:after[key]-before[key] for key in after})
        publish(output/'preparation.json',receipt)


def load_manifest(directory,expected_sha256):
    directory=Path(directory).resolve();work=_work()
    if not directory.is_relative_to(ROOT):raise ValueError('repository chapter directory required')
    manifest=_read_json(directory/'manifest.json',expected_sha256,work)
    prep_path=directory/'preparation.json';prep=_read_json(prep_path,digest(prep_path),work)
    if (manifest.get('schema')!=SCHEMA or manifest.get('source_data',{}).get('sha256')!=SOURCE_DATA_SHA256
            or manifest.get('updates')!=108 or manifest.get('chapter_updates')!=18 or manifest.get('micro_batch_size')!=32
            or manifest.get('families')!=list(FAMILIES) or manifest.get('skills')!=list(SKILLS)
            or len(manifest.get('training',[]))!=108 or manifest.get('source_sha256')!=source_hashes()
            or manifest.get('source_row_target_multiset_sha256')!=manifest.get('compiled_row_target_multiset_sha256')
            or prep.get('status')!='completed' or prep.get('manifest_sha256')!=expected_sha256
            or prep.get('source_sha256')!=manifest['source_sha256']):raise ValueError('complete verified chapter catalogue required')
    if manifest['chapters']!={skill:list(range(i*18,(i+1)*18)) for i,skill in enumerate(SKILLS)}:
        raise ValueError('fixed18-image chapter membership required')
    for index,record in enumerate(manifest['training']):
        if (record.get('index')!=index or record.get('skill')!=SKILLS[index//18]
                or record.get('provider')!=curriculum.VERSION or record.get('format')!='json'
                or manifest['artifact_sha256'].get(record['path'])!=basis._pin(record['sha256'])):
            raise ValueError('chapter image membership differs')
    for name,pin in manifest['input_sha256'].items():
        if digest(basis._path(ROOT,name))!=pin:raise ValueError('pinned chapter input differs')
    for name,pin in manifest['artifact_sha256'].items():
        if digest(basis._path(directory,name))!=pin:raise ValueError('pinned chapter artifact differs')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--data-directory',required=True)
    parser.add_argument('--data-manifest-sha256',default=SOURCE_DATA_SHA256);parser.add_argument('--max-seconds',type=float,default=120)
    args=parser.parse_args()
    print(json.dumps(dict(manifest_sha256=build(args.output,data_directory=args.data_directory,
        data_manifest_sha256=args.data_manifest_sha256,max_seconds=args.max_seconds))))
