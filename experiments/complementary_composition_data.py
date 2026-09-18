"""Paired complementary-program practice with unchanged broad/old rehearsal.

Only the new provider generates or validates lessons. Existing lessons are
authenticated bytes, never regenerated, and tensor archives are never decoded.
Training-fit rows are exact training rows; development transcripts stay separate.
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

from experiments import complementary_composition_curriculum as curriculum
from experiments import definition_basis_data as basis
from experiments import definition_tutor_chapters as chapters
from experiments.foundation_layout_study import ROOT,native,digest,publish,relative_root

SCHEMA='bic-complementary-composition-data-v1'
SEED=853101001
FAMILIES=curriculum.FAMILIES
PANELS=('binding','revision')
BASIS_DIRECTORY=ROOT/'runs/definition-basis-data-local/attempt-001'
BASIS_SHA='3ae5d0d93685675c247c68aaa7a87254abc9ad32e59782b1e34f353c6b5e285b'
CHAPTER_DIRECTORY=ROOT/'runs/definition-tutor-chapters-local/attempt-001'
CHAPTER_SHA='0084f79dcab6dd5dce6fbace1eba0679d3c8b415a42eee47e7e4ef98e32b78be'
PARENT_DIRECTORY=ROOT/'runs/definition-basis-learning-local/attempt-001'
PARENT_SUMMARY_SHA='5f247097e43e8a4f31d209a9a0601731111da3b69de67e13941c5ebbded071cb'
PARENT_LAUNCH_SHA='4fd471d4b5efc9357a1eccd6ac86c1867c4f863d6ddd86a443bca22c04d82415'
PARENT_CHECKPOINT_SHA='5548a2f46897f33f32afd5320b348756fc2c591a81cd27ba009e8d1ddd8a39f8'
PROTECTION=dict(path='runs/continuous-tutor-campaign-local/attempt-001/initial-protection.json',
    sha256='9fabf77ce915da82b0448718ff57739d9fabe2c5159b84b19758ecf529dbe63f')
CURRENT_PROTECTION=dict(path='runs/complementary-protection-local/attempt-001/protected.json',
    sha256='6f13ab920538ab19b4acc9738f52c8a53ee65acf9609aa333f1e439f777dcefc')
CURRENT_PROTECTION_RECEIPT=dict(path='runs/complementary-protection-local/attempt-001/report.json',
    sha256='115a36f104da2c5abd493f3a76c690a3f717206a4ea43e23da1fec7281dfc071')
EVALUATION=dict(directory='runs/shared-acquisition-transfer-data-local/attempt-001',
    manifest_sha256='3ce2402df0edfb54fca9f5dd01aa42e16a37b4b3f27b0adf8884bc24449d5c2a')
BANK_NAMES=tuple(f'complementary_{role}_{panel}' for role in ('dev','fit') for panel in PANELS)
identity=basis.identity
transcript=basis.transcript


def source_hashes():
    return {**chapters.source_hashes(),**curriculum.source_hashes(),
        'experiments/complementary_composition_data.py':digest(Path(__file__))}


def _coordinate(index):
    if type(index) is not int or not 0<=index<180:raise ValueError('complementary image index0..179 required')
    cell=index%10
    return PANELS[cell//5],cell%5


def training_seed(seed,index,pair_index):
    _,contrast=_coordinate(index)
    if type(seed) is not int or not 0<=seed<2**63-2000000 or type(pair_index) is not int or not 0<=pair_index<16:
        raise ValueError('declared seed and pair index required')
    return seed-seed%5+100000+5*(index*16+pair_index)+contrast


def development_seed(seed,panel,contrast,pair_index):
    if (type(seed) is not int or not 0<=seed<2**63-2000000 or panel not in PANELS
            or type(contrast) is not int or not 0<=contrast<5 or type(pair_index) is not int or not 0<=pair_index<8):
        raise ValueError('declared development cell required')
    return seed-seed%5+1000000+5*(PANELS.index(panel)*40+contrast*8+pair_index)+contrast


def schedule():
    """Pure catalogue references; both arms share all432 rehearsal positions."""
    result=dict(control=[],curriculum=[]);new_index=0
    for group in range(108):
        for kind,index in (('learning',2*group),('replay',3*group%108),('old_definition',group),
                ('replay',(3*group+1)%108),('learning',2*group+1),('replay',(3*group+2)%108)):
            if kind!='learning':
                for arm in result:result[arm].append(dict(kind=kind,catalogue=kind,source_index=index))
            else:
                result['control'].append(dict(kind='basis',catalogue='basis',source_index=index%108))
                if index%6==0:
                    j=index//6;record=dict(kind='basis',catalogue='atomic',source_index=(j%6)*18+j//6)
                else:
                    record=dict(kind='complementary',catalogue='complementary',source_index=new_index);new_index+=1
                result['curriculum'].append(record)
    if new_index!=180:raise AssertionError('fixed complementary schedule arithmetic differs')
    return result


def _record(record,kind,directory=None):
    path=record['path'] if directory is None else relative_root(basis._path(directory,record['path']))
    basis._path(ROOT,path);basis._pin(record['sha256'])
    result=dict(path=path,sha256=record['sha256'],index=record['index'],kind=kind,
        provider=record['provider'],format=record['format'])
    if kind=='replay':result['original_cursor']=record['original_cursor']
    for name in ('panel','contrast_index','learning_identity_sha256'):
        if name in record:result[name]=record[name]
    return result


def expand_schedule(catalogues):
    return {arm:[deepcopy(catalogues[item['catalogue']][item['source_index']]) for item in order]
        for arm,order in schedule().items()}


def validate_catalogues(catalogues):
    if set(catalogues)!={'basis','atomic','old_definition','replay','complementary'}:raise ValueError('all five fixed catalogues required')
    for name,records in catalogues.items():
        if len(records)!=(180 if name=='complementary' else 108):raise ValueError('fixed catalogue size differs')
        kind='basis' if name=='atomic' else name
        provider=curriculum.VERSION if kind=='complementary' else basis.PROVIDERS[kind]
        for index,record in enumerate(records):
            if (record.get('index')!=index or record.get('kind')!=kind or record.get('provider')!=provider
                    or record.get('format')!=('torch_weights_only' if kind=='replay' else 'json')):
                raise ValueError('canonical record kind/provider/index differs')
            basis._path(ROOT,record['path']);basis._pin(record['sha256'])
            if kind=='replay':
                if type(record.get('original_cursor')) is not int or record['original_cursor']<0:raise ValueError('original replay cursor required')
            else:basis._pin(record['learning_identity_sha256'])
            if name=='complementary' and (record.get('panel'),record.get('contrast_index'))!=_coordinate(index):
                raise ValueError('complementary cell assignment differs')
    return True


def learning_identity(image):
    families,targets=image['bundle']['families'],image['state_targets']
    if set(families)!=set(FAMILIES) or set(targets)!=set(FAMILIES):raise ValueError('three aligned learning families required')
    result=[]
    for family in FAMILIES:
        if len(families[family])!=len(targets[family]) or not families[family]:raise ValueError('row/label alignment differs')
        rows=[]
        for row,labels in zip(families[family],targets[family]):
            if (len(labels)!=len(row['turns']) or any(type(v) is not list or len(v)!=12
                    or any(type(x) is not int or not 0<=x<107 for x in v) for v in labels)):
                raise ValueError('causal107-class prefix target shape differs')
            rows.append(dict(turns=[{k:t[k] for k in ('text','observations','target','reply')} for t in row['turns']],state_targets=labels))
        result.append([family,rows])
    return identity(result)


def authenticate_image(image,record):
    if (type(image) is not dict or set(image)!={'bundle','expected_evidence','expected_evidence_sha256','state_targets','admission_schema'}
            or image['admission_schema']!=record['provider'] or image['bundle']['bundle_id']!=record['index']
            or image['expected_evidence']['bundle_id']!=record['index']
            or identity(image['expected_evidence'])!=image['expected_evidence_sha256']
            or identity([[f,image['bundle']['families'][f]] for f in FAMILIES])!=image['expected_evidence']['bundle_rows_sha256']):
        raise ValueError('source image evidence/provider/cursor differs')
    actual=learning_identity(image)
    if 'learning_identity_sha256' in record and actual!=record['learning_identity_sha256']:
        raise ValueError('ordered learning inputs/prefix labels differ')
    return actual


def _work():
    return dict(input_hash_attempts=0,input_hashes=0,input_hash_bytes=0,archive_hash_attempts=0,archive_hashes=0,
        json_decode_attempts=0,json_decodes=0,generated_training_pairs=0,generated_development_pairs=0,
        admitted_pairs=0,prefix_label_rows=0,prefix_label_elements=0,published_training_images=0,
        existing_image_identity_checks=0,archive_loads=0,models=0,training_updates=0,teacher_calls=0)


def _checked(path,pin,work,*,decode=False):
    return basis._checked(path,pin,work,decode=decode)


def load_json_image(record):
    if record['format']!='json' or record['kind'] not in ('basis','old_definition','complementary'):
        raise ValueError('admitted JSON image required')
    image=_checked(basis._path(ROOT,record['path']),record['sha256'],_work(),decode=True)
    authenticate_image(image,record)
    return image


def _image(bundle,work):
    evidence=curriculum.bundle_evidence(bundle)
    work['admitted_pairs']+=sum(len(rows)//2 for rows in bundle['families'].values())
    targets={f:[curriculum.prefix_labels(row) for row in bundle['families'][f]] for f in FAMILIES}
    work['prefix_label_rows']+=sum(len(rows) for rows in targets.values())
    work['prefix_label_elements']+=sum(len(turn) for rows in targets.values() for row in rows for turn in row)
    return dict(bundle=bundle,expected_evidence=evidence,expected_evidence_sha256=identity(evidence),
        state_targets=targets,admission_schema=curriculum.VERSION)


def _inventory(rows):
    return dict(episodes=len(rows),pairs=len(rows)//2,family_episodes=dict(Counter(r['family'] for r in rows)),
        family_contrast_pairs=dict(Counter(f"{rows[i]['family']}/{rows[i]['recipe']['contrast_index']}" for i in range(0,len(rows),2))),
        rows_sha256=identity(rows))


def _fingerprints(rows,protected,*,other=()):
    hashes=[transcript(row) for row in rows]
    if set(hashes)&set(protected) or set(hashes)&set(other):raise ValueError('new lesson overlaps protected/evaluation text')
    return hashes


def _current_protection(value,evaluation):
    if (value.get('schema')!='bic-complementary-protected-transcripts-v1'
            or value.get('manifest_sha256')!=EVALUATION['manifest_sha256']
            or value.get('source')!=dict(path=EVALUATION['directory']+'/'+evaluation['banks']['path'],sha256=evaluation['banks']['sha256'])
            or set(value.get('by_bank',{}))!=set(evaluation['bank_inventory'])):
        raise ValueError('exact current legacy-bank fingerprint export required')
    result=set()
    for name,hashes in value['by_bank'].items():
        if len(hashes)!=evaluation['bank_inventory'][name]['episodes'] or any(basis._pin(h)!=h for h in hashes):
            raise ValueError('current legacy-bank fingerprint census differs')
        result.update(hashes)
    return result


def _parent(summary,launch,commit):
    if (summary.get('schema')!='bic-definition-basis-learning-v1' or summary.get('status')!='completed'
            or summary.get('partial_work_unknown') or summary.get('updates')!=648 or summary.get('lifetime_updates')!=9760
            or summary.get('launch_sha256')!=PARENT_LAUNCH_SHA or summary.get('endpoint',{}).get('sha256')!=PARENT_CHECKPOINT_SHA
            or commit.get('checkpoint')!=summary['endpoint'] or commit.get('relative_step')!=648
            or commit.get('lifetime_updates')!=9760 or commit.get('launch_sha256')!=PARENT_LAUNCH_SHA
            or commit.get('scores')!=summary.get('evaluations',{}).get('648')
            or launch.get('data_manifest_sha256')!=BASIS_SHA or launch.get('evaluation')!=EVALUATION):
        raise ValueError('exact completed9760 full-state parent required')
    return dict(identity_sha256=identity(launch['origin_identity']),weights_sha256=commit['weights_sha256'],lifetime_updates=9760)


def build(output,*,basis_directory=BASIS_DIRECTORY,basis_manifest_sha256=BASIS_SHA,
          chapters_directory=CHAPTER_DIRECTORY,chapters_manifest_sha256=CHAPTER_SHA,
          parent_directory=PARENT_DIRECTORY,parent_summary_sha256=PARENT_SUMMARY_SHA,seed=SEED,max_seconds=120):
    if (seed!=SEED or type(seed) is not int or type(max_seconds) not in (int,float)
            or not math.isfinite(max_seconds) or not 0<max_seconds<=120
            or (basis_manifest_sha256,chapters_manifest_sha256,parent_summary_sha256)!=(BASIS_SHA,CHAPTER_SHA,PARENT_SUMMARY_SHA)):
        raise ValueError('fixed seed/catalogues/parent and at-most120-second preparation required')
    output,basis_directory,chapters_directory,parent_directory=map(lambda p:Path(p).resolve(),
        (output,basis_directory,chapters_directory,parent_directory))
    if any(not p.is_relative_to(ROOT) for p in (output,basis_directory,chapters_directory,parent_directory)):
        raise ValueError('repository paths required')
    wall,cpu=time.monotonic(),time.process_time();native(output).mkdir(parents=True,exist_ok=False)
    sources,pins,artifacts=source_hashes(),{},{};before=curriculum.work_report();work=_work()
    receipt=dict(schema=SCHEMA,status='running',work=work)
    publish(output/'started.json',dict(schema=SCHEMA,seed=seed,max_seconds=max_seconds,source_sha256=sources))
    def boundary():
        if time.monotonic()-wall>=max_seconds:raise TimeoutError('complementary preparation allowance ended')
    def checked(path,pin,decode=False):
        boundary();value=_checked(path,pin,work,decode=decode);pins[relative_root(path)]=pin;return value
    def save(name,value):
        boundary();pin=publish(output/name,value);artifacts[name]=pin;return dict(path=relative_root(output/name),sha256=pin)
    def pair(family,number,split,panel):
        boundary();value=curriculum.generate_pair(family,number,split=split,panel=panel)
        work['generated_training_pairs' if split=='train' else 'generated_development_pairs']+=1;return value
    try:
        original=checked(basis_directory/'manifest.json',BASIS_SHA,True)
        catalogue=checked(chapters_directory/'manifest.json',CHAPTER_SHA,True)
        if original.get('source_sha256')!=basis.source_hashes() or catalogue.get('source_sha256')!=chapters.source_hashes():
            raise ValueError('original catalogue source closure changed')
        for directory,pin,manifest in ((basis_directory,BASIS_SHA,original),(chapters_directory,CHAPTER_SHA,catalogue)):
            p=directory/'preparation.json';prep=checked(p,digest(p),True)
            if prep.get('status')!='completed' or prep.get('manifest_sha256')!=pin or prep.get('source_sha256')!=manifest['source_sha256']:
                raise ValueError('completed source-identical catalogue preparation required')
        summary=checked(parent_directory/'execution/summary.json',PARENT_SUMMARY_SHA,True)
        launch=checked(parent_directory/'launch.json',PARENT_LAUNCH_SHA,True);record=summary['endpoint_commits']['648']
        commit=checked(basis._path(ROOT,record['path']),record['sha256'],True);parent=_parent(summary,launch,commit)
        parent_inputs={relative_root(parent_directory/'execution/summary.json'):PARENT_SUMMARY_SHA,
            relative_root(parent_directory/'launch.json'):PARENT_LAUNCH_SHA,record['path']:record['sha256']}
        # Checkpoint pin is inherited here; root's freeze authenticates bytes and full state.
        catalogues=dict(basis=[_record(r,'basis',basis_directory) for r in original['training']],
            atomic=[_record(r,'basis',chapters_directory) for r in catalogue['training']],
            old_definition=[_record(r,'old_definition') for r in original['old_definition']],
            replay=[_record(r,'replay') for r in original['replay']],complementary=[])
        if any(len(catalogues[k])!=108 for k in ('basis','atomic','old_definition','replay')):raise ValueError('complete108-record inherited catalogues required')
        for name,records in catalogues.items():
            for index,record in enumerate(records):
                if record['index']!=index:raise ValueError('canonical source index differs')
                image=checked(basis._path(ROOT,record['path']),record['sha256'],name!='replay')
                if name!='replay':
                    record['learning_identity_sha256']=authenticate_image(image,record);work['existing_image_identity_checks']+=1
        protection=checked(ROOT/PROTECTION['path'],PROTECTION['sha256'],True)
        if type(protection) is not list or protection!=sorted(set(protection)) or any(basis._pin(v)!=v for v in protection):
            raise ValueError('pinned transcript census required')
        protected=set(protection)
        evaluation_directory=ROOT/EVALUATION['directory'];evaluation=checked(evaluation_directory/'manifest.json',EVALUATION['manifest_sha256'],True)
        current=checked(ROOT/CURRENT_PROTECTION['path'],CURRENT_PROTECTION['sha256'],True)
        protected.update(_current_protection(current,evaluation))
        checked(ROOT/CURRENT_PROTECTION_RECEIPT['path'],CURRENT_PROTECTION_RECEIPT['sha256'])
        fresh=checked(evaluation_directory/'fresh-transcripts.json',evaluation['artifacts_sha256']['fresh-transcripts.json'],True)
        if fresh!=sorted(set(fresh)) or any(basis._pin(v)!=v for v in fresh):raise ValueError('legacy fresh fingerprint census differs')
        protected.update(fresh)
        inherited_banks=dict(basis={**original['banks'],'path':relative_root(basis._path(basis_directory,original['banks']['path']))},
            old_definition=deepcopy(original['old_banks']),legacy=dict(directory=EVALUATION['directory'],manifest_sha256=EVALUATION['manifest_sha256'],
                archive=dict(path=relative_root(evaluation_directory/evaluation['banks']['path']),sha256=evaluation['banks']['sha256'])))
        inherited_inventory={**{f'basis_{k}':deepcopy(v) for k,v in original['bank_inventory'].items()},
            **{f'definition_{k}':deepcopy(v) for k,v in original['old_bank_inventory'].items()},**deepcopy(evaluation['bank_inventory'])}
        for group,inventory in (('basis',original['bank_inventory']),('old_definition',original['old_bank_inventory'])):
            record=inherited_banks[group];banks=checked(ROOT/record['path'],record['sha256'],True)
            if set(banks)!=set(inventory) or any(identity(rows)!=inventory[k]['rows_sha256'] for k,rows in banks.items()):
                raise ValueError('original definition bank membership differs')
            for rows in banks.values():protected.update(transcript(row) for row in rows)
        banks={name:[] for name in BANK_NAMES};development=set();fit_provenance={name:[] for name in BANK_NAMES if '_fit_' in name}
        for panel in PANELS:
            rows=banks['complementary_dev_'+panel]
            for family in FAMILIES:
                for contrast in range(5):
                    for offset in range(8):
                        value=pair(family,development_seed(seed,panel,contrast,offset),'dev',panel)
                        curriculum.validate_pair(value);work['admitted_pairs']+=1;rows.extend(value)
            fingerprints=_fingerprints(rows,protected,other=development)
            if len(set(fingerprints))!=len(rows):raise ValueError('development transcript repeats')
            development.update(fingerprints)
        train_hashes=[];dimensions=Counter();costs=Counter();atom_counts=Counter()
        for index in range(180):
            panel,contrast=_coordinate(index);families={}
            for family in FAMILIES:
                rows=[]
                for offset in range(16):rows.extend(pair(family,training_seed(seed,index,offset),'train',panel))
                train_hashes.extend(_fingerprints(rows,protected,other=development));families[family]=rows
                dimensions[f'{family}/{panel}/{contrast}']+=len(rows)//2
                for row in rows[::2]:
                    atom=(row['program'][0]['meaning'] if panel=='revision' else
                        next(event['meaning'] for event in row['program'] if event['op']=='define' and event['word']==row['recipe']['other_word']))
                    atom_counts[f'{family}/{panel}/{contrast}/{atom}']+=1
                if index<10:
                    banks['complementary_fit_'+panel].extend(deepcopy(rows[:8]))
                    fit_provenance['complementary_fit_'+panel].append(dict(image_index=index,family=family,pair_indices=list(range(4))))
                for row in rows:
                    costs['observation_bytes']+=sum(len(t['text'].encode()) for t in row['turns'])
                    costs['reply_target_bytes']+=sum(len(t['reply'].encode()) for t in row['turns'])
                    costs['turns']+=len(row['turns'])
            bundle=dict(schema=curriculum.BUNDLE_SCHEMA,bundle_id=index,layout='original',families=families)
            image=_image(bundle,work);record=save(f'training/{index:03d}.json',image);work['published_training_images']+=1
            catalogues['complementary'].append(dict(**record,index=index,kind='complementary',panel=panel,contrast_index=contrast,
                provider=curriculum.VERSION,format='json',learning_identity_sha256=learning_identity(image)))
        if set(dimensions.values())!={288} or len(dimensions)!=30:raise ValueError('all30 family/panel/contrast cells need288 pairs')
        validate_catalogues(catalogues)
        inventory={name:_inventory(rows) for name,rows in banks.items()}
        if any(v['episodes']!=(120 if '_fit_' in k else 240) for k,v in inventory.items()):raise ValueError('new bank census differs')
        bank_record=save('banks.json',banks);train_record=save('training-transcripts.json',sorted(set(train_hashes)))
        dev_record=save('development-transcripts.json',sorted(development));boundary()
        if source_hashes()!=sources:raise ValueError('complementary sources changed')
        manifest=dict(schema=SCHEMA,seed=seed,micro_batch_size=32,updates_per_arm=648,parent=parent,
            parent_checkpoint=deepcopy(summary['endpoint']),parent_inputs=parent_inputs,
            basis_data=dict(directory=relative_root(basis_directory),manifest_sha256=BASIS_SHA),
            chapters=dict(directory=relative_root(chapters_directory),manifest_sha256=CHAPTER_SHA),evaluation=deepcopy(EVALUATION),
            catalogues=catalogues,phases=dict(training=expand_schedule(catalogues)),banks=bank_record,bank_inventory=inventory,
            fit_provenance=fit_provenance,inherited_banks=inherited_banks,inherited_bank_inventory=inherited_inventory,
            training_inventory=dict(episodes=17280,unique_transcripts=len(set(train_hashes)),repeated_transcript_rows=17280-len(set(train_hashes)),
                pairs_by_family_panel_contrast=dict(sorted(dimensions.items())),other_or_prior_atom_pair_counts=dict(sorted(atom_counts.items())),
                new_complementary_text_cost=dict(costs)),
            exposure_counts=dict(control=dict(basis=20736,old_definition=10368,replay=31104,total=62208),
                curriculum=dict(basis=3456,complementary=17280,old_definition=10368,replay=31104,total=62208)),
            protection=dict(existing_unique=len(protected),existing_sha256=identity(sorted(protected)),
                legacy_reference=deepcopy(PROTECTION),current_bank_reference=deepcopy(CURRENT_PROTECTION),
                current_bank_export_receipt=deepcopy(CURRENT_PROTECTION_RECEIPT),new_train=train_record,new_development=dev_record,
                train_development_overlap=0,new_training_existing_evaluation_overlap=0),
            source_sha256=sources,input_sha256=pins,artifact_sha256=artifacts,
            scope='One paired change in practiced programs; exact old rehearsal and12 observed banks retained. New fit is actual training, newdev is finite-grammar acquisition; no teacher or automatic promotion.')
        pin=publish(output/'manifest.json',manifest);receipt.update(status='completed',manifest_sha256=pin);return pin
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        after=curriculum.work_report();receipt.update(wall_seconds=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,
            source_sha256=sources,input_sha256=pins,artifact_sha256=artifacts,
            curriculum_work={k:after[k]-before[k] for k in after},
            cost_scope='Includes inherited JSON authentication, archive byte hashes and all new generation/admission; no archive decode or neural work. Prior preparation costs are not new work.')
        publish(output/'preparation.json',receipt)


def load_manifest(directory,expected_sha256):
    directory=Path(directory).resolve();work=_work()
    if not directory.is_relative_to(ROOT):raise ValueError('repository data directory required')
    manifest=_checked(directory/'manifest.json',expected_sha256,work,decode=True)
    prep_path=directory/'preparation.json';prep=_checked(prep_path,digest(prep_path),work,decode=True)
    if (manifest.get('schema')!=SCHEMA or manifest.get('source_sha256')!=source_hashes() or prep.get('status')!='completed'
            or prep.get('manifest_sha256')!=expected_sha256 or prep.get('source_sha256')!=manifest['source_sha256']
            or manifest.get('seed')!=SEED or manifest.get('updates_per_arm')!=648 or manifest.get('micro_batch_size')!=32
            or manifest.get('parent',{}).get('lifetime_updates')!=9760):raise ValueError('completed fixed source-identical complementary data required')
    for path,pin in manifest['input_sha256'].items():_checked(basis._path(ROOT,path),pin,work)
    for path,pin in manifest['artifact_sha256'].items():_checked(basis._path(directory,path),pin,work)
    validate_catalogues(manifest['catalogues'])
    for name,records in manifest['catalogues'].items():
        for record in records:
            path=basis._path(ROOT,record['path'])
            declared=(manifest['artifact_sha256'].get(path.relative_to(directory).as_posix())
                if name=='complementary' and path.is_relative_to(directory) else manifest['input_sha256'].get(record['path']))
            if declared!=record['sha256']:raise ValueError('scheduled record outside authenticated inventory')
    if manifest['phases']!=dict(training=expand_schedule(manifest['catalogues'])):raise ValueError('fixed paired rehearsal/learning schedule differs')
    if {k:v['episodes'] for k,v in manifest['bank_inventory'].items()}!={name:120 if '_fit_' in name else 240 for name in BANK_NAMES}:
        raise ValueError('all four bank sizes required')
    if set(manifest['inherited_bank_inventory'])!={*(f'basis_{p}' for p in basis.PANELS),*(f'definition_{p}' for p in basis.PANELS[:3]),
            'dev','retention','train_fit','transfer_original','transfer_varied'}:raise ValueError('all12 previous banks required')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--max-seconds',type=float,default=120)
    args=parser.parse_args();print(json.dumps(dict(manifest_sha256=build(args.output,max_seconds=args.max_seconds))))
