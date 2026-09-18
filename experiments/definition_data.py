"""Compact, independently admitted English-definition lessons and test panels.

No model, teacher, tensor archive decode, or training occurs in this producer.
The separately pinned replay catalogue remains in its original directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import traceback

from experiments import definition_curriculum as curriculum
from experiments.foundation_layout_study import ROOT, native, digest, read, publish, relative_root, verify_pins

SCHEMA = 'bic-definition-learning-data-v1'
MICRO, UPDATES = 32, 108
PANELS = ('binding', 'revision', 'composition')
SEED = 854001001


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def source_hashes():
    return {**curriculum.source_hashes(),
        'experiments/definition_data.py': digest(Path(__file__)),
        'experiments/foundation_layout_study.py': digest(ROOT/'experiments/foundation_layout_study.py')}


def transcript(row):
    return identity([turn['text'] for turn in row['turns']])


def _check_rows(rows, *, split, panel):
    if len(rows) % 2:
        raise ValueError('complete definition pairs required')
    for index in range(0, len(rows), 2):
        curriculum.validate_pair(rows[index:index+2])
    for row in rows:
        if row['split'] != split or row['recipe']['panel'] != panel:
            raise ValueError('honest declared partition and panel required')
        if len(row['turns']) != 12 or any(len(t['text'].encode('utf-8')) > 128 for t in row['turns']):
            raise ValueError('twelve bounded visible utterances required')
        if sum(len(t['text'].encode('utf-8'))+2 for t in row['turns']) > 1024:
            raise ValueError('definition episode exceeds native context')


def build(output, *, replay_directory, replay_manifest_sha256, seed=SEED, max_seconds=120):
    started, cpu = time.monotonic(), time.process_time()
    if (type(seed) is not int or not 0 <= seed < 2**63-1072015
            or type(max_seconds) not in (int, float) or not 0 < max_seconds <= 600):
        raise ValueError('explicit seed and bounded CPU preparation required')
    output, replay_directory = Path(output).resolve(), Path(replay_directory).resolve()
    if not output.is_relative_to(ROOT) or not replay_directory.is_relative_to(ROOT):
        raise ValueError('repository artifact paths required')
    native(output).mkdir(parents=True, exist_ok=False)
    sources, artifacts, pins = source_hashes(), {}, {}
    curriculum_before = curriculum.work_report()
    work = dict(schema=SCHEMA, status='running', generated_pairs=0, admitted_pairs=0,
        prefix_label_rows=0, prefix_label_elements=0, models=0, teacher_calls=0,
        archive_loads=0, training_updates=0)
    publish(output/'started.json', work)

    def boundary():
        if time.monotonic()-started >= max_seconds:
            raise TimeoutError('definition data preparation allowance ended')

    def checked(path, pin):
        boundary()
        if digest(path) != pin:
            raise ValueError('pinned input changed')
        pins[relative_root(path)] = pin
        return read(path)

    def save(name, value):
        boundary()
        pin = publish(output/name, value)
        artifacts[name] = pin
        return dict(path=name, sha256=pin)

    def pair(family, number, split, panel):
        boundary()
        value = curriculum.generate_pair(family, number, split=split, panel=panel, turns=12)
        work['generated_pairs'] += 1
        _check_rows(value, split=split, panel=panel)
        work['admitted_pairs'] += 1
        return value

    try:
        # This is an already completed, independently verified common replay
        # stream. No new canonical generation or archive decoding is performed.
        replay = checked(replay_directory/'manifest.json', replay_manifest_sha256)
        prep_path = replay_directory/'preparation.json'
        preparation = checked(prep_path, digest(prep_path))
        if (replay['schema'] != 'bic-verified-tutor-cycle-data-v1'
                or preparation['status'] != 'completed'
                or preparation['manifest_sha256'] != replay_manifest_sha256
                or replay['micro_batch_size'] != MICRO or replay['updates_per_phase'] != UPDATES
                or replay['phases']['withdrawal']['procedural'] != replay['phases']['withdrawal']['tutor']):
            raise ValueError('completed common broad replay catalogue required')
        verify_pins(replay['source_sha256']); verify_pins(replay['input_sha256'])
        original_replay = replay['phases']['withdrawal']['procedural']
        expected_start = replay['start_cursor']+UPDATES
        if (len(original_replay) != UPDATES or [r['cursor'] for r in original_replay] !=
                list(range(expected_start,expected_start+UPDATES))):
            raise ValueError('exact ordered108-bundle common replay required')
        replay_records = []
        for record in original_replay:
            path = (replay_directory/record['path']).resolve()
            if not path.is_relative_to(replay_directory) or replay['artifact_sha256'].get(record['path']) != record['sha256']:
                raise ValueError('replay catalogue membership differs')
            boundary()
            if digest(path) != record['sha256']:
                raise ValueError('replay archive bytes changed')
            pins[relative_root(path)] = record['sha256']
            replay_records.append(dict(path=relative_root(path), sha256=record['sha256'], original_cursor=record['cursor']))
        banks, protected, bank_inventory = {}, set(), {}
        for panel_index, panel in enumerate(PANELS):
            rows = []
            for family_index, family in enumerate(curriculum.FAMILIES):
                for index in range(32):
                    rows.extend(pair(family, seed+1000000+panel_index*10000+family_index*1000+index,
                        'dev', panel))
            fingerprints = [transcript(row) for row in rows]
            if len(set(fingerprints)) != len(fingerprints) or protected.intersection(fingerprints):
                raise ValueError('definition evaluation transcripts overlap')
            protected.update(fingerprints)
            banks[panel] = rows
            bank_inventory[panel] = dict(episodes=len(rows), pairs=len(rows)//2,
                family_episodes=dict(Counter(row['family'] for row in rows)), rows_sha256=identity(rows))
        bank_record = save('banks.json', banks)
        training, seen, dimensions = [], set(), Counter()
        for index in range(UPDATES):
            panel = ('binding', 'revision')[index % 2]
            families = {}
            for family_index, family in enumerate(curriculum.FAMILIES):
                rows = []
                for offset in range(MICRO//2):
                    rows.extend(pair(family, seed+index*10000+family_index*1000+offset, 'train', panel))
                families[family] = rows
                for row in rows:
                    pin = transcript(row)
                    if pin in protected:
                        raise ValueError('training overlaps heldout definition transcripts')
                    seen.add(pin)
                    dimensions[(family, panel)] += 1
            bundle = dict(schema='bic-foundation-layout-bundle-v1', bundle_id=index,
                layout='original', families=families)
            evidence = curriculum.bundle_evidence(bundle)
            labels = {family:[curriculum.prefix_labels(row) for row in rows] for family,rows in families.items()}
            for values in labels.values():
                work['prefix_label_rows'] += len(values)
                work['prefix_label_elements'] += sum(len(turn) for row in values for turn in row)
            image = dict(bundle=bundle, expected_evidence=evidence,
                expected_evidence_sha256=identity(evidence), state_targets=labels,
                admission_schema=curriculum.VERSION)
            training.append({**save(f'training/{index:03d}.json', image), 'index':index, 'panel':panel})
        if len(replay_records) != UPDATES:
            raise ValueError('exact matched replay count required')
        verify_pins(pins)
        boundary()
        if source_hashes() != sources:
            raise ValueError('definition producer source changed during preparation')
        manifest = dict(schema=SCHEMA, seed=seed, micro_batch_size=MICRO,
            definition_updates=UPDATES, replay_updates=UPDATES, total_updates=2*UPDATES,
            training=training, replay=replay_records, banks=bank_record, bank_inventory=bank_inventory,
            training_inventory=dict(episodes=UPDATES*MICRO*len(curriculum.FAMILIES), unique_transcripts=len(seen),
                repeated_transcript_rows=UPDATES*MICRO*len(curriculum.FAMILIES)-len(seen),
                by_family_panel={f'{f}/{p}':count for (f,p),count in sorted(dimensions.items())}),
            train_eval_transcript_overlap=0, source_sha256=sources, input_sha256=pins,
            artifact_sha256=artifacts, definition_schema=curriculum.VERSION,
            replay_manifest=dict(path=relative_root(replay_directory/'manifest.json'),sha256=replay_manifest_sha256),
            schedule='Alternate one new-definition bundle then one broad replay bundle, for108 pairs of updates.',
            scope='New definition admission; unchanged native learner; no claim of broad English or learned self-direction.')
        pin = publish(output/'manifest.json', manifest)
        work.update(status='completed', manifest_sha256=pin)
        return pin
    except BaseException as error:
        work.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        curriculum_after = curriculum.work_report()
        work.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
            source_sha256=sources, input_sha256=pins, artifact_sha256=artifacts,
            curriculum_work={key:value-curriculum_before[key] for key,value in curriculum_after.items()},
            accounting_scope='Requested generated pairs are distinct from canonical reconstruction and interpreter calls; all actual curriculum calls are retained separately.')
        publish(output/'preparation.json', work)


def load_manifest(directory, expected_sha256):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(ROOT) or digest(directory/'manifest.json') != expected_sha256:
        raise ValueError('caller-pinned definition data required')
    manifest, prep = read(directory/'manifest.json'), read(directory/'preparation.json')
    if (manifest['schema'] != SCHEMA or prep['status'] != 'completed'
            or prep['manifest_sha256'] != expected_sha256
            or manifest['total_updates'] != 216 or len(manifest['training']) != 108
            or len(manifest['replay']) != 108 or manifest['definition_schema'] != curriculum.VERSION
            or manifest['micro_batch_size'] != MICRO or manifest['definition_updates'] != UPDATES
            or manifest['replay_updates'] != UPDATES or prep['source_sha256'] != manifest['source_sha256']
            or manifest['source_sha256'] != source_hashes()):
        raise ValueError('complete definition and broad-practice dataset required')
    for index,record in enumerate(manifest['training']):
        if (record['index'] != index or record['panel'] != ('binding','revision')[index%2]
                or manifest['artifact_sha256'].get(record['path']) != record['sha256']):
            raise ValueError('definition order/panel or admitted artifact identity differs')
    banks = manifest['banks']
    if (manifest['artifact_sha256'].get(banks['path']) != banks['sha256']
            or set(manifest['bank_inventory']) != set(PANELS)
            or any(value['episodes'] != 192 or value['pairs'] != 96
                   or value['family_episodes'] != {family:64 for family in curriculum.FAMILIES}
                   for value in manifest['bank_inventory'].values())):
        raise ValueError('complete three-family definition panels required')
    cursors=[record['original_cursor'] for record in manifest['replay']]
    if (any(type(c) is not int for c in cursors) or cursors != list(range(cursors[0],cursors[0]+UPDATES))
            or any(manifest['input_sha256'].get(record['path']) != record['sha256'] for record in manifest['replay'])):
        raise ValueError('authenticated ordered original replay records required')
    verify_pins(manifest['source_sha256']); verify_pins(manifest['input_sha256'])
    for name,pin in manifest['artifact_sha256'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or digest(path) != pin:
            raise ValueError('definition artifact changed')
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('output')
    parser.add_argument('--replay-directory', required=True); parser.add_argument('--replay-manifest-sha256', required=True)
    parser.add_argument('--seed', type=int, default=SEED); parser.add_argument('--max-seconds', type=int, default=120)
    args=parser.parse_args()
    print(json.dumps(dict(manifest_sha256=build(args.output,replay_directory=args.replay_directory,
        replay_manifest_sha256=args.replay_manifest_sha256,seed=args.seed,max_seconds=args.max_seconds))))
