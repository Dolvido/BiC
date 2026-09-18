"""CPU-only basis lessons plus an explicit, immutable 648-update practice plan.

Old JSON lessons, replay archives and retention banks are authenticated bytes;
they are neither decoded nor regenerated here. New lessons are independently
admitted by the basis provider. No tensor/neural dependency is imported.
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
from experiments.foundation_layout_study import ROOT, native, digest, publish, relative_root

SCHEMA='bic-definition-basis-data-v1'
OLD_SCHEMA='bic-definition-learning-data-v1'
OLD_VERSION='bic-english-definition-curriculum-v1'
OLD_DIRECTORY=ROOT/'runs/definition-data-local/attempt-001'
OLD_MANIFEST_SHA256='eb75fb8487fe83b6cac32d8f746fb37f4e0d699140ffa1cb2e2619e4fd39d696'
SEED=855001001
MICRO,UPDATES,TOTAL_UPDATES=32,108,648
PANELS=('binding','revision','composition','sequence')
PROVIDERS={'basis':curriculum.VERSION,'old_definition':OLD_VERSION,'replay':'immutable-admitted-layout-image'}


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {**curriculum.source_hashes(),
        'experiments/definition_basis_data.py':digest(Path(__file__)),
        'experiments/foundation_layout_study.py':digest(ROOT/'experiments/foundation_layout_study.py')}


def transcript(row):return identity([turn['text'] for turn in row['turns']])


def schedule():
    result=[]
    for group in range(UPDATES):
        result.extend(dict(kind=kind,source_index=index) for kind,index in (
            ('basis',2*group%108),('replay',3*group%108),('old_definition',group),
            ('replay',(3*group+1)%108),('basis',(2*group+1)%108),('replay',(3*group+2)%108)))
    return result


def training_seed(seed,index,pair_index):
    if type(index) is not int or not 0<=index<108 or type(pair_index) is not int or not 0<=pair_index<16:
        raise ValueError('declared training catalogue coordinates required')
    return seed+index*1000+pair_index


def _path(base,name):
    if type(name) is not str or not name or Path(name).is_absolute():raise ValueError('relative artifact path required')
    path=(base/name).resolve()
    if not path.is_relative_to(base.resolve()):raise ValueError('artifact path leaves its declared directory')
    return path


def _pin(value):
    if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('lowercase SHA256 required')
    return value


def _inventory_valid(inventory,panels):
    return (type(inventory) is dict and set(inventory)==set(panels) and all(
        value.get('episodes')==192 and value.get('pairs')==96
        and value.get('family_episodes')=={family:64 for family in curriculum.FAMILIES}
        and type(value.get('rows_sha256')) is str and len(value['rows_sha256'])==64
        for value in inventory.values()))


def _inherit(old,directory):
    """Pure structural admission of the pinned old catalogue, no lesson decode."""
    if (old.get('schema')!=OLD_SCHEMA or old.get('definition_schema')!=OLD_VERSION
            or old.get('micro_batch_size')!=32 or old.get('definition_updates')!=108
            or old.get('replay_updates')!=108 or old.get('total_updates')!=216
            or len(old.get('training',[]))!=108 or len(old.get('replay',[]))!=108
            or not _inventory_valid(old.get('bank_inventory'),PANELS[:3])):
        raise ValueError('the completed original 108-definition/108-replay catalogue is required')
    definitions=[];replay=[]
    for index,record in enumerate(old['training']):
        if (record.get('index')!=index or record.get('panel')!=('binding','revision')[index%2]
                or old['artifact_sha256'].get(record['path'])!=_pin(record['sha256'])):
            raise ValueError('old definition membership/order differs')
        definitions.append(dict(index=index,panel=record['panel'],path=relative_root(_path(directory,record['path'])),
            sha256=record['sha256'],type='old_definition',provider=PROVIDERS['old_definition'],format='json'))
    cursors=[record.get('original_cursor') for record in old['replay']]
    if any(type(value) is not int or value<0 for value in cursors) or cursors!=list(range(cursors[0],cursors[0]+108)):
        raise ValueError('ordered admitted broad source cursors required')
    for index,record in enumerate(old['replay']):
        if old['input_sha256'].get(record['path'])!=_pin(record['sha256']):raise ValueError('broad source pin differs')
        _path(ROOT,record['path'])
        replay.append(dict(index=index,path=record['path'],sha256=record['sha256'],original_cursor=cursors[index],
            type='replay',provider=PROVIDERS['replay'],format='torch_weights_only'))
    record=old['banks']
    if old['artifact_sha256'].get(record['path'])!=_pin(record['sha256']):raise ValueError('old bank membership differs')
    banks=dict(path=relative_root(_path(directory,record['path'])),sha256=record['sha256'],
        type='old_definition',provider=OLD_VERSION,format='json')
    return definitions,replay,banks


def _image(bundle,work):
    evidence=curriculum.bundle_evidence(bundle)
    work['admitted_pairs']+=sum(len(rows)//2 for rows in bundle['families'].values())
    labels={family:[curriculum.prefix_labels(row) for row in rows] for family,rows in bundle['families'].items()}
    work['prefix_label_rows']+=sum(len(rows) for rows in labels.values())
    work['prefix_label_elements']+=sum(len(turn) for rows in labels.values() for sample in rows for turn in sample)
    return dict(bundle=bundle,expected_evidence=evidence,expected_evidence_sha256=identity(evidence),
        state_targets=labels,admission_schema=curriculum.VERSION)


def _contract(seed,max_seconds):
    if (type(seed) is not int or not 0<=seed<2**63-2000000 or type(max_seconds) not in (int,float)
            or not math.isfinite(max_seconds) or not 0<max_seconds<=120):
        raise ValueError('63-bit seed and at-most120-second CPU preparation required')


def _work():
    return dict(generated_training_pairs=0,generated_evaluation_pairs=0,admitted_pairs=0,
        prefix_label_rows=0,prefix_label_elements=0,input_hash_attempts=0,input_hashes=0,input_hash_bytes=0,
        archive_hash_attempts=0,archive_hashes=0,json_decode_attempts=0,json_decodes=0,
        published_training_images=0,archive_loads=0,models=0,training_updates=0,teacher_calls=0)


def _checked(path,pin,work,*,decode=False):
    _pin(pin);work['input_hash_attempts']+=1
    archive=path.suffix.lower() in ('.pt','.pth')
    if archive:work['archive_hash_attempts']+=1
    raw=native(path).read_bytes();work['input_hash_bytes']+=len(raw)
    if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError(f'pinned input changed: {relative_root(path)}')
    work['input_hashes']+=1
    if archive:work['archive_hashes']+=1
    if decode:
        if path.suffix.lower()!='.json':raise ValueError('only explicit JSON metadata may be decoded')
        work['json_decode_attempts']+=1;value=json.loads(raw);work['json_decodes']+=1
        return value


def build(output,*,old_directory=OLD_DIRECTORY,old_manifest_sha256=OLD_MANIFEST_SHA256,seed=SEED,max_seconds=120):
    _contract(seed,max_seconds)
    if old_manifest_sha256!=OLD_MANIFEST_SHA256:raise ValueError('the exact declared v1 data manifest is required')
    output,old_directory=Path(output).resolve(),Path(old_directory).resolve()
    if not output.is_relative_to(ROOT) or not old_directory.is_relative_to(ROOT):raise ValueError('repository directories required')
    started,cpu=time.monotonic(),time.process_time();native(output).mkdir(parents=True,exist_ok=False)
    sources,pins,artifacts=source_hashes(),{},{}
    before=curriculum.work_report();work=_work();report=dict(schema=SCHEMA,status='running',work=work)
    publish(output/'started.json',dict(schema=SCHEMA,seed=seed,max_seconds=max_seconds,source_sha256=sources,
        original_manifest_sha256=old_manifest_sha256,work=deepcopy(work)))
    def boundary():
        if time.monotonic()-started>=max_seconds:raise TimeoutError('basis data preparation allowance ended')
    def checked(path,pin,decode=False):
        boundary();value=_checked(path,pin,work,decode=decode);pins[relative_root(path)]=pin;return value
    def save(name,value):
        boundary();pin=publish(output/name,value);artifacts[name]=pin;return dict(path=name,sha256=pin)
    def pair(family,number,split,panel):
        boundary();value=curriculum.generate_pair(family,number,split=split,panel=panel)
        work['generated_training_pairs' if split=='train' else 'generated_evaluation_pairs']+=1
        return value
    try:
        old=checked(old_directory/'manifest.json',old_manifest_sha256,True)
        prep_path=old_directory/'preparation.json';prep=checked(prep_path,digest(prep_path),True)
        if (prep.get('status')!='completed' or prep.get('manifest_sha256')!=old_manifest_sha256
                or prep.get('source_sha256')!=old.get('source_sha256')):raise ValueError('completed pinned original preparation required')
        definitions,replay,old_banks=_inherit(old,old_directory)
        for name,pin in old['source_sha256'].items():checked(_path(ROOT,name),pin)
        for record in definitions+replay+[old_banks]:checked(_path(ROOT,record['path']),record['sha256'])
        # Bind the old replay's producing catalogue without decoding its lessons.
        origin=old['replay_manifest'];checked(_path(ROOT,origin['path']),origin['sha256'])
        bank_payload,inventory,protected={}, {}, set()
        for panel_index,panel in enumerate(PANELS):
            rows=[]
            for family_index,family in enumerate(curriculum.FAMILIES):
                for index in range(32):
                    value=pair(family,seed+1000000+panel_index*10000+family_index*1000+index,'dev',panel)
                    curriculum.validate_pair(value);work['admitted_pairs']+=1;rows.extend(value)
            fingerprints=[transcript(row) for row in rows]
            if len(set(fingerprints))!=192 or protected.intersection(fingerprints):raise ValueError('new evaluation transcripts repeat or cross panels')
            protected.update(fingerprints);bank_payload[panel]=rows
            inventory[panel]=dict(episodes=192,pairs=96,family_episodes=dict(Counter(r['family'] for r in rows)),rows_sha256=identity(rows))
        banks=save('banks.json',bank_payload)
        training,seen,classes,dimensions=[],set(),Counter(),Counter()
        for index in range(108):
            panel=('binding','revision')[index%2];families={}
            for family in curriculum.FAMILIES:
                rows=[]
                for offset in range(16):
                    number=training_seed(seed,index,offset);rows.extend(pair(family,number,'train',panel))
                    classes[(family,panel,number%3)]+=1
                families[family]=rows
                for sample in rows:
                    fingerprint=transcript(sample)
                    if fingerprint in protected:raise ValueError('basis training overlaps reserved evaluation')
                    seen.add(fingerprint);dimensions[(family,panel)]+=1
            bundle=dict(schema=curriculum.BUNDLE_SCHEMA,bundle_id=index,layout='original',families=families)
            record=save(f'training/{index:03d}.json',_image(bundle,work));work['published_training_images']+=1
            training.append(dict(**record,index=index,panel=panel,type='basis',provider=curriculum.VERSION,format='json'))
        if set(classes.values())!={288} or len(classes)!=18:raise ValueError('every family/panel/contrast must have288 pairs')
        for name,pin in list(pins.items()):boundary();_checked(_path(ROOT,name),pin,work)
        boundary()
        if source_hashes()!=sources:raise ValueError('basis data source changed during preparation')
        manifest=dict(schema=SCHEMA,seed=seed,micro_batch_size=32,definition_schema=curriculum.VERSION,
            basis_updates=216,old_definition_updates=108,replay_updates=324,total_updates=648,
            training=training,old_definition=definitions,replay=replay,banks=banks,bank_inventory=inventory,
            old_banks=old_banks,old_bank_inventory=deepcopy(old['bank_inventory']),schedule=schedule(),
            training_inventory=dict(episodes=10368,unique_transcripts=len(seen),repeated_transcript_rows=10368-len(seen),
                by_family_panel={f'{f}/{p}':n for (f,p),n in sorted(dimensions.items())},
                pairs_by_family_panel_contrast={f'{f}/{p}/{k}':n for (f,p,k),n in sorted(classes.items())}),
            exposures=dict(basis=20736,old_definition=10368,replay=31104,total=62208),
            path_scopes=dict(training='data_directory',banks='data_directory',old_definition='repository',replay='repository',old_banks='repository'),
            train_eval_transcript_overlap=0,source_sha256=sources,input_sha256=pins,artifact_sha256=artifacts,
            original_data=dict(path=relative_root(old_directory/'manifest.json'),sha256=old_manifest_sha256),
            broad_origin=deepcopy(origin),
            retention_scope='Original three definition banks reused as observed retention, not fresh evaluation.',
            scope='Fixed shared semantic-basis practice plus exact old lessons; no architecture or policy changes.')
        pin=publish(output/'manifest.json',manifest);report.update(status='completed',manifest_sha256=pin)
        return pin
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        after=curriculum.work_report()
        report.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
            source_sha256=sources,input_sha256=pins,artifact_sha256=artifacts,
            curriculum_work={key:after[key]-before[key] for key in after},
            accounting_scope='Archive byte hashes are counted separately from zero archive decodes; requested pairs, validation reconstructions and repeated scheduled exposures are distinct.')
        publish(output/'preparation.json',report)


def load_manifest(directory,expected_sha256):
    directory=Path(directory).resolve();work=_work()
    if not directory.is_relative_to(ROOT):raise ValueError('repository data directory required')
    manifest=_checked(directory/'manifest.json',expected_sha256,work,decode=True)
    prep_path=directory/'preparation.json';prep=_checked(prep_path,digest(prep_path),work,decode=True)
    if (manifest.get('schema')!=SCHEMA or manifest.get('definition_schema')!=curriculum.VERSION
            or manifest.get('total_updates')!=648 or manifest.get('micro_batch_size')!=32
            or manifest.get('basis_updates')!=216 or manifest.get('old_definition_updates')!=108 or manifest.get('replay_updates')!=324
            or manifest.get('schedule')!=schedule() or manifest.get('source_sha256')!=source_hashes()
            or prep.get('status')!='completed' or prep.get('manifest_sha256')!=expected_sha256
            or prep.get('source_sha256')!=manifest['source_sha256']
            or not _inventory_valid(manifest.get('bank_inventory'),PANELS)
            or not _inventory_valid(manifest.get('old_bank_inventory'),PANELS[:3])):
        raise ValueError('complete fixed basis dataset, source and schedule required')
    original=manifest['original_data']
    if original['sha256']!=OLD_MANIFEST_SHA256:raise ValueError('original v1 dataset identity differs')
    old_path=_path(ROOT,original['path']);old=_checked(old_path,original['sha256'],work,decode=True)
    inherited=_inherit(old,old_path.parent)
    if (manifest['old_definition'],manifest['replay'],manifest['old_banks'])!=inherited:raise ValueError('old lesson catalogue differs')
    if manifest['old_bank_inventory']!=old['bank_inventory']:raise ValueError('old definition retention inventory differs')
    if len(manifest['training'])!=108:raise ValueError('exact108 new basis images required')
    for index,record in enumerate(manifest['training']):
        if (record.get('index')!=index or record.get('panel')!=('binding','revision')[index%2]
                or record.get('type')!='basis' or record.get('provider')!=curriculum.VERSION or record.get('format')!='json'
                or manifest['artifact_sha256'].get(record['path'])!=_pin(record['sha256'])):raise ValueError('new basis catalogue membership differs')
        _path(directory,record['path'])
    bank=manifest['banks']
    if manifest['artifact_sha256'].get(bank['path'])!=_pin(bank['sha256']):raise ValueError('new basis bank pin differs')
    for name,pin in manifest['input_sha256'].items():_checked(_path(ROOT,name),pin,work)
    for name,pin in manifest['artifact_sha256'].items():_checked(_path(directory,name),pin,work)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--old-directory',default=str(OLD_DIRECTORY));parser.add_argument('--old-manifest-sha256',default=OLD_MANIFEST_SHA256)
    parser.add_argument('--max-seconds',type=float,default=120);parser.add_argument('--load-manifest-sha256')
    args=parser.parse_args()
    if args.load_manifest_sha256:
        value=load_manifest(args.output,args.load_manifest_sha256)
        print(json.dumps(dict(schema=value['schema'],total_updates=value['total_updates'])))
    else:
        print(json.dumps(dict(manifest_sha256=build(args.output,old_directory=args.old_directory,
            old_manifest_sha256=args.old_manifest_sha256,seed=args.seed,max_seconds=args.max_seconds))))
