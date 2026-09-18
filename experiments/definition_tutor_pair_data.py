"""Exact shared teaching multiset, accepted chapter order, common withdrawal.

This module authenticates existing data and a completed author result. It never
samples lessons, decodes tensor archives, or contacts a teacher. Teaching-stream
identity includes every ordered learner input and target; provenance IDs do not
create a treatment difference. Withdrawal uses one identical pinned record list.
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

from experiments import definition_tutor_chapters as chapters
from experiments import definition_basis_data as basis
from experiments import definition_tutor_author as author
from experiments.foundation_layout_study import ROOT,native,digest,publish,relative_root

SCHEMA='bic-definition-tutor-pair-data-v1'
CHAPTER_MANIFEST_SHA256='0084f79dcab6dd5dce6fbace1eba0679d3c8b415a42eee47e7e4ef98e32b78be'
BASIS_MANIFEST_SHA256=chapters.SOURCE_DATA_SHA256
AUTHOR_RESULT_SHA256='e9d7cf4ad140df1a4e225a4cbbf99a607a2baa82a2105a15e9a9c94423626192'
AUTHOR_REQUEST_SHA256='9bfad51b75c95318f02fb15faa21f128972a4e779976d903e9919f9708c6ca81'
TARGET_SOURCE='experiments/shared_state_targets.py'
FAMILIES=chapters.FAMILIES
identity=basis.identity


def source_hashes():
    return {**author.source_hashes(),TARGET_SOURCE:digest(ROOT/TARGET_SOURCE),
        'experiments/definition_tutor_pair_data.py':digest(Path(__file__))}


def learning_identity(image):
    """Ordered full input/target declarations, excluding nonlearning metadata."""
    families=image['bundle']['families'];targets=image['state_targets']
    if set(families)!=set(FAMILIES) or set(targets)!=set(FAMILIES):raise ValueError('all three aligned learning families required')
    payload=[]
    for family in FAMILIES:
        rows=families[family];labels=targets[family]
        if len(rows)!=len(labels) or not rows:raise ValueError('nonempty aligned row/target arrays required')
        episodes=[]
        for row,state in zip(rows,labels):
            turns=row['turns']
            if len(turns)!=len(state):raise ValueError('state-target turn alignment differs')
            if any(type(values) is not list or len(values)!=12 or any(type(v) is not int or not 0<=v<=106 for v in values) for values in state):
                raise ValueError('exact107-class integer prefix targets required')
            if any(type(turn['text']) is not str or type(turn['reply']) is not str
                    or type(turn['target']) is not int or not 0<=turn['target']<=3 for turn in turns):
                raise ValueError('exact native text/action/reply supervision required')
            episodes.append(dict(turns=[{key:deepcopy(turn[key]) for key in ('text','observations','target','reply')} for turn in turns],
                state_targets=deepcopy(state)))
        payload.append([family,episodes])
    return identity(payload)


def _record(record,kind,directory=None):
    path=record['path'] if directory is None else relative_root(basis._path(directory,record['path']))
    basis._path(ROOT,path);basis._pin(record['sha256'])
    result=dict(path=path,sha256=record['sha256'],index=record['index'],kind=kind,
        provider=record['provider'],format=record['format'])
    if kind=='replay':result['original_cursor']=record['original_cursor']
    return result


def withdrawal_records(chapter_manifest,basis_manifest,chapters_directory):
    result=[]
    for group in range(18):
        def selected(j):return (j%6)*18+j//6
        first,second=selected(2*group),selected(2*group+1)
        result.extend((_record(chapter_manifest['training'][first],'basis',chapters_directory),
            _record(basis_manifest['replay'][3*group],'replay'),
            _record(basis_manifest['old_definition'][group],'old_definition'),
            _record(basis_manifest['replay'][3*group+1],'replay'),
            _record(chapter_manifest['training'][second],'basis',chapters_directory),
            _record(basis_manifest['replay'][3*group+2],'replay')))
    return result


def teaching_contrast(teaching):
    if type(teaching) is not dict or set(teaching)!={'procedural','tutor'}:raise ValueError('both teaching arms required')
    streams={};record_sets={};content_sets={}
    for arm,records in teaching.items():
        if len(records)!=108 or sorted(record['index'] for record in records)!=list(range(108)):
            raise ValueError('each teaching image must occur exactly once')
        values=[basis._pin(record['learning_identity_sha256']) for record in records]
        streams[arm]=identity(values);content_sets[arm]=Counter(values)
        record_sets[arm]=Counter((record['index'],record['path'],record['sha256']) for record in records)
    if record_sets['procedural']!=record_sets['tutor'] or content_sets['procedural']!=content_sets['tutor']:
        raise ValueError('both teaching arms must share the exact image and learning-content multiset')
    differing=sum(a['learning_identity_sha256']!=b['learning_identity_sha256']
        for a,b in zip(teaching['procedural'],teaching['tutor']))
    return dict(teaching_learning_stream_sha256=streams,
        teaching_learning_multiset_sha256=identity(sorted(content_sets['procedural'].items())),
        teaching_record_multiset_sha256=identity(sorted(record_sets['procedural'].items())),
        zero_treatment_contrast=streams['procedural']==streams['tutor'],different_teaching_positions=differing)


def _work():
    return dict(json_read_attempts=0,json_reads=0,authenticated_bytes=0,archive_hash_attempts=0,archive_hashes=0,
        author_authentication_attempts=0,author_authentications=0,delegated_author_json_reads=0,
        learning_identity_checks=0,new_requested_pairs=0,canonical_regenerations=0,archive_loads=0,
        models=0,training_updates=0,teacher_calls=0)


def _read(path,pin,work,inputs,*,decode=True):
    basis._pin(pin);archive=path.suffix.lower() in ('.pt','.pth')
    if archive:
        if decode:raise ValueError('tensor archive decoding is forbidden in paired-data preparation')
        work['archive_hash_attempts']+=1
    elif decode:work['json_read_attempts']+=1
    raw=native(path).read_bytes();work['authenticated_bytes']+=len(raw)
    if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('paired-data input bytes changed')
    inputs[relative_root(path)]=pin
    if archive:work['archive_hashes']+=1
    if decode:
        if path.suffix.lower()!='.json':raise ValueError('JSON data input required')
        value=json.loads(raw);work['json_reads']+=1;return value


def _check_image(image,record):
    if (type(image) is not dict or set(image)!={'bundle','expected_evidence','expected_evidence_sha256','state_targets','admission_schema'}
            or image['admission_schema']!=record['provider'] or image['bundle']['bundle_id']!=record['index']
            or image['expected_evidence']['bundle_id']!=record['index']
            or identity(image['expected_evidence'])!=image['expected_evidence_sha256']
            or identity([[f,image['bundle']['families'][f]] for f in FAMILIES])!=image['expected_evidence']['bundle_rows_sha256']):
        raise ValueError('authenticated source image provider, cursor or evidence differs')
    return learning_identity(image)


def load_json_image(record):
    if record['format']!='json' or record['kind'] not in ('basis','old_definition'):
        raise ValueError('plain JSON lesson record required')
    image=_read(basis._path(ROOT,record['path']),record['sha256'],_work(),{})
    actual=_check_image(image,record)
    if actual!=record['learning_identity_sha256']:raise ValueError('full learning content differs from paired catalogue')
    return image


def _compose(chapters_directory,chapters_pin,basis_directory,basis_pin,decision,work,inputs,boundary):
    if chapters_pin!=CHAPTER_MANIFEST_SHA256 or basis_pin!=BASIS_MANIFEST_SHA256:
        raise ValueError('the fixed chapter and original basis catalogues are required')
    if (type(decision) is not dict or set(decision)!={'path','sha256','request_sha256'}
            or decision['sha256']!=AUTHOR_RESULT_SHA256 or decision['request_sha256']!=AUTHOR_REQUEST_SHA256):
        raise ValueError('the exact accepted persisted author decision is required')
    def read(path,pin,decode=True):boundary();return _read(path,pin,work,inputs,decode=decode)
    catalogue=read(chapters_directory/'manifest.json',chapters_pin)
    chapter_prep_path=chapters_directory/'preparation.json';chapter_prep=read(chapter_prep_path,digest(chapter_prep_path))
    original=read(basis_directory/'manifest.json',basis_pin)
    basis_prep_path=basis_directory/'preparation.json';basis_prep=read(basis_prep_path,digest(basis_prep_path))
    if (catalogue.get('schema')!=chapters.SCHEMA or catalogue.get('source_sha256')!=chapters.source_hashes()
            or catalogue.get('source_data',{}).get('sha256')!=basis_pin
            or original.get('schema')!=basis.SCHEMA or original.get('source_sha256')!=basis.source_hashes()
            or any(prep.get('status')!='completed' or prep.get('manifest_sha256')!=pin
                or prep.get('source_sha256')!=manifest['source_sha256'] for prep,pin,manifest in
                ((chapter_prep,chapters_pin,catalogue),(basis_prep,basis_pin,original)))):
        raise ValueError('completed source-identical chapter/basis catalogues required')
    boundary();result_path=basis._path(ROOT,decision['path']);work['author_authentication_attempts']+=1
    result=author.load_result(result_path,expected_sha256=decision['sha256'],expected_request_sha256=decision['request_sha256'])
    work['author_authentications']+=1
    contract=chapters.make_contract(chapters_pin)
    if (result['outcome']!='local_accepted' or result['mode']!='local' or result['contract_sha256']!=author.request_sha256(contract)):
        raise ValueError('accepted author recipe must bind this exact teaching catalogue')
    intent=dict(path=relative_root(result_path.parent/'intent.json'),sha256=result['intent_sha256'])
    response=dict(path=relative_root(result_path.parent/'response.json'),sha256=result['teacher_cost']['response_file_sha256'])
    for record in (decision,intent,response):
        inputs[record['path']]=record['sha256'];work['delegated_author_json_reads']+=1
        work['authenticated_bytes']+=native(basis._path(ROOT,record['path'])).stat().st_size
    recipes=dict(procedural=chapters.procedural_recipe(contract),tutor=result['recipe'])
    orders={arm:chapters.ordered_indices(recipe,contract,catalogue) for arm,recipe in recipes.items()}
    records=[];image_identity_cache={}
    def json_record(record):
        key=(record['path'],record['sha256'])
        if key not in image_identity_cache:
            image=read(basis._path(ROOT,record['path']),record['sha256'])
            image_identity_cache[key]=_check_image(image,record);work['learning_identity_checks']+=1
        return dict(**record,learning_identity_sha256=image_identity_cache[key])
    if len(catalogue['training'])!=108:raise ValueError('exact108 chapter images required')
    for index,record in enumerate(catalogue['training']):
        if (record['index']!=index or record['skill']!=chapters.SKILLS[index//18]
                or catalogue['artifact_sha256'].get(record['path'])!=record['sha256']):raise ValueError('chapter image membership differs')
        records.append(json_record(_record(record,'basis',chapters_directory)))
    teaching={arm:[deepcopy(records[index]) for index in order] for arm,order in orders.items()}
    withdrawal=[]
    for record in withdrawal_records(catalogue,original,chapters_directory):
        if record['kind']=='replay':
            read(basis._path(ROOT,record['path']),record['sha256'],False);withdrawal.append(record)
        else:withdrawal.append(json_record(record))
    if len(withdrawal)!=108:raise ValueError('fixed108 common withdrawal updates required')
    banks=dict(basis={**original['banks'],'path':relative_root(basis._path(basis_directory,original['banks']['path']))},
        old_definition=deepcopy(original['old_banks']))
    for record in banks.values():read(basis._path(ROOT,record['path']),record['sha256'],False)
    return dict(schema=SCHEMA,updates_per_arm=216,teaching_updates=108,withdrawal_updates=108,micro_batch_size=32,
        phases=dict(teaching=teaching,withdrawal=withdrawal),**teaching_contrast(teaching),
        parent={key:result['parent'][key] for key in ('identity_sha256','weights_sha256','lifetime_updates')},
        author_parent=deepcopy(result['parent']),teacher_decision=deepcopy(decision),recipes=recipes,
        author_inputs=dict(intent=intent,response=response,contract=contract,contract_sha256=result['contract_sha256'],
            recipe_sha256=result['recipe_sha256'],development_sha256=result['development_sha256']),
        prior_teacher_cost=deepcopy(result['teacher_cost']),new_teacher_calls=0,
        chapters=dict(directory=relative_root(chapters_directory),manifest_sha256=chapters_pin),
        basis_data=dict(directory=relative_root(basis_directory),manifest_sha256=basis_pin),
        banks=banks,bank_inventory=dict(basis=deepcopy(original['bank_inventory']),old_definition=deepcopy(original['old_bank_inventory'])),
        withdrawal_target_source=dict(path=TARGET_SOURCE,sha256=digest(ROOT/TARGET_SOURCE)),
        planned_exposures_per_arm=dict(teaching=10368,withdrawal=10368,total=20736),
        comparison_scope='Teaching digests cover full ordered learner inputs/action/reply/prefix targets; withdrawal is one identical pinned record list with shared target code. Different streams do not assert different final weights.',
        teacher_cost_scope='Previously completed author work is provenance, not new preparation work; no service calls occur here.')


def build(output,*,chapters_directory,chapters_manifest_sha256,basis_directory,basis_manifest_sha256,teacher_decision,max_seconds=120):
    if type(max_seconds) not in (int,float) or not math.isfinite(max_seconds) or not 0<max_seconds<=120:
        raise ValueError('at-most120-second local preparation required')
    output=Path(output).resolve();chapters_directory=Path(chapters_directory).resolve();basis_directory=Path(basis_directory).resolve()
    if any(not path.is_relative_to(ROOT) for path in (output,chapters_directory,basis_directory)):raise ValueError('repository directories required')
    wall,cpu=time.monotonic(),time.process_time();native(output).mkdir(parents=True,exist_ok=False)
    sources,inputs,work=source_hashes(),{},_work();receipt=dict(schema=SCHEMA,status='running',work=work)
    publish(output/'started.json',dict(schema=SCHEMA,source_sha256=sources,max_seconds=max_seconds,teacher_decision=teacher_decision))
    def boundary():
        if time.monotonic()-wall>=max_seconds:raise TimeoutError('paired data preparation allowance ended')
    try:
        manifest=_compose(chapters_directory,chapters_manifest_sha256,basis_directory,basis_manifest_sha256,teacher_decision,work,inputs,boundary)
        boundary()
        if source_hashes()!=sources:raise ValueError('paired data sources changed')
        manifest.update(source_sha256=sources,input_sha256=inputs)
        pin=publish(output/'manifest.json',manifest);receipt.update(status='completed',manifest_sha256=pin);return pin
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(wall_seconds=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,source_sha256=sources,input_sha256=inputs)
        publish(output/'preparation.json',receipt)


def load_manifest(directory,expected_sha256):
    directory=Path(directory).resolve();work=_work();inputs={}
    if not directory.is_relative_to(ROOT):raise ValueError('repository paired data required')
    manifest=_read(directory/'manifest.json',expected_sha256,work,{})
    prep_path=directory/'preparation.json';prep=_read(prep_path,digest(prep_path),work,{})
    if (manifest.get('schema')!=SCHEMA or manifest.get('source_sha256')!=source_hashes() or prep.get('status')!='completed'
            or prep.get('manifest_sha256')!=expected_sha256 or prep.get('source_sha256')!=manifest['source_sha256']):
        raise ValueError('completed source-identical paired data required')
    expected=_compose(basis._path(ROOT,manifest['chapters']['directory']),manifest['chapters']['manifest_sha256'],
        basis._path(ROOT,manifest['basis_data']['directory']),manifest['basis_data']['manifest_sha256'],
        manifest['teacher_decision'],work,inputs,lambda:None)
    expected.update(source_sha256=source_hashes(),input_sha256=inputs)
    if manifest!=expected:raise ValueError('paired phases, full learning identities or author binding differ')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--chapters-directory',required=True)
    parser.add_argument('--basis-directory',required=True);parser.add_argument('--teacher-result',required=True)
    parser.add_argument('--max-seconds',type=float,default=120);args=parser.parse_args()
    print(json.dumps(dict(manifest_sha256=build(args.output,chapters_directory=args.chapters_directory,
        chapters_manifest_sha256=CHAPTER_MANIFEST_SHA256,basis_directory=args.basis_directory,basis_manifest_sha256=BASIS_MANIFEST_SHA256,
        teacher_decision=dict(path=args.teacher_result,sha256=AUTHOR_RESULT_SHA256,request_sha256=AUTHOR_REQUEST_SHA256),max_seconds=args.max_seconds))))
