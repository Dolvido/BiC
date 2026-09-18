"""Matched complementary composition and atomic-basis practice with broad replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import gc
import hashlib
import io
import json
from pathlib import Path
import time
import traceback

from experiments.foundation_layout_study import ROOT, CONFIG, native, digest, read, publish, relative_root, verify_pins, utc

SCHEMA = 'bic-complementary-composition-run-v1'
PROTOCOL = 'docs/COMPLEMENTARY_COMPOSITION_PROTOCOL.md'
ARMS = ('control', 'curriculum')
ENDPOINTS = (216,432,648)
UPDATES, MICRO, PARENT_CURSOR = 648, 32, 9760
PARENT_LAUNCH_SHA256 = '4fd471d4b5efc9357a1eccd6ac86c1867c4f863d6ddd86a443bca22c04d82415'
PARENT_SUMMARY_SHA256 = '5f247097e43e8a4f31d209a9a0601731111da3b69de67e13941c5ebbded071cb'
PARENT_CHECKPOINT_SHA256 = '5548a2f46897f33f32afd5320b348756fc2c591a81cd27ba009e8d1ddd8a39f8'
BASE_BANK_NAMES = ('basis_binding','basis_revision','basis_composition','basis_sequence',
    'definition_binding','definition_revision','definition_composition',
    'dev','train_fit','retention','transfer_original','transfer_varied')
EXTRA_BANK_NAMES = ('complementary_dev_binding','complementary_dev_revision',
    'complementary_fit_binding','complementary_fit_revision')
BANK_NAMES = BASE_BANK_NAMES+EXTRA_BANK_NAMES


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def contract(max_seconds=1200):
    if type(max_seconds) is not int or max_seconds != 1200:
        raise ValueError('one protocol-fixed1200-second total allowance required')
    return dict(schema=SCHEMA,arms=list(ARMS),updates_per_arm=UPDATES,micro_batch_size=MICRO,
        endpoints=[0,*ENDPOINTS],max_seconds=1200,config=CONFIG,teacher_calls=0,
        automatic_retry=False,automatic_promotion=False,parent_is_unadopted_research_candidate=True)


def physical_plan(manifest):
    if (manifest.get('schema')!='bic-complementary-composition-data-v1'
            or manifest.get('parent',{}).get('lifetime_updates')!=PARENT_CURSOR
            or manifest.get('updates_per_arm')!=UPDATES or manifest.get('micro_batch_size')!=MICRO
            or set(manifest.get('phases',{}))!={'training'}
            or set(manifest['phases']['training'])!=set(ARMS)):
        raise ValueError('exact complementary paired data required')
    for arm in ARMS:
        records=manifest['phases']['training'][arm]
        if len(records)!=UPDATES:raise ValueError('exact648 ordered lessons required')
        learning=0
        for index,record in enumerate(records):
            position=index%6
            expected='replay' if position in (1,3,5) else 'old_definition' if position==2 else 'basis'
            if position in (0,4):
                if arm=='curriculum' and learning%6:expected='complementary'
                learning+=1
            if record['kind']!=expected:raise ValueError('fixed whole-curriculum schedule differs')
        if learning!=216:raise ValueError('216 learning slots required')
    return dict(physical_arms=list(ARMS),updates=1296,episode_exposures=124416,
        family_forwards=3888,backwards=3888,restores=8,snapshots=6)


def parent_admission(summary, previous, commit):
    if (summary.get('schema')!='bic-definition-basis-learning-v1' or summary.get('status')!='completed'
            or summary.get('partial_work_unknown') or summary.get('updates')!=648
            or summary.get('lifetime_updates')!=PARENT_CURSOR or summary.get('launch_sha256')!=PARENT_LAUNCH_SHA256
            or previous.get('parent',{}).get('lifetime_updates')!=9112
            or summary.get('physical_training',{}).get('work',{}).get('unknown_optimizer_outcomes')!=0
            or summary.get('endpoint',{}).get('sha256')!=PARENT_CHECKPOINT_SHA256
            or commit.get('checkpoint')!=summary['endpoint'] or commit.get('relative_step')!=648
            or commit.get('lifetime_updates')!=PARENT_CURSOR or commit.get('launch_sha256')!=PARENT_LAUNCH_SHA256
            or commit.get('scores')!=summary.get('evaluations',{}).get('648')
            or set(commit.get('scores',{}))!=set(BASE_BANK_NAMES)):
        raise ValueError('completed unadopted9760 basis candidate with exact full endpoint required')
    return dict(checkpoint=deepcopy(summary['endpoint']),identity_sha256=identity(previous['origin_identity']),lifetime_updates=PARENT_CURSOR,
        weights_sha256=commit['weights_sha256'],metrics={k:v['metrics'] for k,v in commit['scores'].items()})


def validate_bindings(manifest, parent):
    expected=dict(identity_sha256=parent['identity_sha256'],weights_sha256=parent['weights_sha256'],lifetime_updates=PARENT_CURSOR)
    if manifest.get('parent')!=expected or manifest.get('parent_checkpoint')!=parent['checkpoint']:
        raise ValueError('curriculum belongs to another complete parent')
    return physical_plan(manifest)


def accounting_delta(accounting, parent):
    result={group:{key:value-parent[group][key] for key,value in accounting[group].items()}
        for group in ('work','state_work','cost')}
    if any(value<0 for group in ('work','state_work') for value in result[group].values()):
        raise ValueError('lifetime work cannot regress on restart')
    return result


def validate_completed_work(plan, receipt, physical):
    expected={a:UPDATES for a in ARMS}
    if (receipt['new_updates']!=expected or set(physical)!=set(plan['physical_arms'])
            or len(receipt['restorations'])!=plan['restores'] or len(receipt['snapshots'])!=plan['snapshots']):
        raise ValueError('measured paired physical workload differs')
    for value in physical.values():
        work=value['work']
        expected_work=dict(retained_updates=648,synchronized_optimizer_updates=648,optimizer_attempts=648,
            optimizer_returns=648,unknown_optimizer_outcomes=0,retained_episodes=62208,
            attempted_forwards=1944,completed_forwards=1944,attempted_backwards=1944,completed_backwards=1944,
            attempted_forward_episodes=62208,completed_forward_episodes=62208,
            attempted_backward_episodes=62208,completed_backward_episodes=62208)
        if any(work.get(k)!=v for k,v in expected_work.items()):raise ValueError('actual complete learner accounting differs')
        if set(value['state_work'])!={'attempted_state_readouts','completed_state_readouts','attempted_state_objectives','completed_state_objectives'} or any(v!=1944 for v in value['state_work'].values()):
            raise ValueError('complete unchanged auxiliary learning work required')


def source_hashes():
    from experiments import definition_basis_learning as basis, complementary_composition_data as data
    from experiments import complementary_composition_evaluation as evaluation
    result={**basis.source_hashes(),**data.source_hashes(),**evaluation.source_hashes()}
    for name in ('experiments/complementary_composition_run.py',PROTOCOL):result[name]=digest(ROOT/name)
    return result


def freeze(output, *, parent_directory, parent_summary_sha256, data_directory, data_manifest_sha256, max_seconds=1200):
    started,cpu=time.monotonic(),time.process_time();fixed=contract(max_seconds)
    if parent_summary_sha256!=PARENT_SUMMARY_SHA256:raise ValueError('exact completed basis summary pin required')
    output,parent_directory,data_directory=map(lambda p:Path(p).resolve(),(output,parent_directory,data_directory))
    if any(not p.is_relative_to(ROOT) for p in (output,parent_directory,data_directory)):raise ValueError('repository paths required')
    native(output).mkdir(parents=True,exist_ok=False)
    receipt=dict(schema=SCHEMA,status='running',checkpoint_load_attempts=0,checkpoint_loads=0,models=0,updates=0,teacher_calls=0)
    publish(output/'freeze-started.json',receipt)
    try:
        summary_path=parent_directory/'execution/summary.json';previous_path=parent_directory/'launch.json'
        if digest(summary_path)!=parent_summary_sha256 or digest(previous_path)!=PARENT_LAUNCH_SHA256:raise ValueError('pinned basis parent changed')
        summary,previous=read(summary_path),read(previous_path)
        endpoint=summary['endpoint_commits']['648']
        if digest(ROOT/endpoint['path'])!=endpoint['sha256']:raise ValueError('parent endpoint commit changed')
        commit=read(ROOT/endpoint['path']);parent=parent_admission(summary,previous,commit)
        verify_pins(previous['source_sha256']);verify_pins(previous['input_sha256'])
        from experiments import complementary_composition_data as data, continuous_tutor_worker as worker
        manifest=data.load_manifest(data_directory,data_manifest_sha256)
        plan=validate_bindings(manifest,parent)
        raw=native(ROOT/parent['checkpoint']['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=parent['checkpoint']['sha256']:raise ValueError('parent archive changed')
        import torch
        receipt['checkpoint_load_attempts']+=1
        payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);receipt['checkpoint_loads']+=1
        metadata=worker.parent_metadata(payload);bridge_cost=deepcopy(payload['bridge_cost']);del payload,raw
        if (metadata['cursor']!=PARENT_CURSOR or metadata['weights_sha256']!=parent['weights_sha256']
                or metadata['origin']!=previous['origin_identity'] or metadata['recipe']['config']!=CONFIG
                or metadata['evidence']!=commit['evidence'] or metadata['accounting']!=commit['accounting']['lifetime_kernel']):
            raise ValueError('parent complete optimizer/evidence/origin metadata differs')
        pins={relative_root(summary_path):parent_summary_sha256,relative_root(previous_path):PARENT_LAUNCH_SHA256,
            endpoint['path']:endpoint['sha256'],parent['checkpoint']['path']:parent['checkpoint']['sha256'],
            relative_root(data_directory/'manifest.json'):data_manifest_sha256,
            relative_root(data_directory/'preparation.json'):digest(data_directory/'preparation.json')}
        sources=source_hashes();snapshots={}
        for index,(name,pin) in enumerate(sorted(sources.items())):
            raw=native(ROOT/name).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('source changed during freeze')
            local=f'sources/{index:03d}.bin';path=native(output/local);path.parent.mkdir(exist_ok=True)
            with path.open('xb') as handle:handle.write(raw)
            snapshots[name]=local
        verify_pins(pins)
        if source_hashes()!=sources:raise ValueError('source changed during freeze')
        launch=dict(**fixed,created_utc=utc(),source_sha256=sources,source_snapshots=snapshots,input_sha256=pins,
            parent=parent,parent_metadata=metadata,parent_bridge_cost=bridge_cost,origin_identity=previous['origin_identity'],
            expected_runtime=previous['expected_runtime'],reference_parent={**deepcopy(previous['parent']),
                'metrics':{k:v['metrics'] for k,v in summary['evaluations']['0'].items()}},
            parent_directory=relative_root(parent_directory),parent_summary_sha256=parent_summary_sha256,
            basis_data_directory=previous['data_directory'],basis_data_manifest_sha256=previous['data_manifest_sha256'],
            evaluation=previous['evaluation'],data_directory=relative_root(data_directory),data_manifest_sha256=data_manifest_sha256,
            physical_plan=plan,
            scope='Complementary composition versus atomic practice with identical broad rehearsal; no automatic adoption.')
        pin=publish(output/'launch.json',launch);receipt.update(status='completed',launch_sha256=pin);return pin
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu);publish(output/'freeze-receipt.json',receipt)


def validate_launch(launch):
    fixed=contract(launch.get('max_seconds'))
    if any(launch.get(k)!=v for k,v in fixed.items()):raise ValueError('unchanged paired execution contract required')
    if (launch['parent']['lifetime_updates']!=PARENT_CURSOR
            or launch['parent']['checkpoint']['sha256']!=PARENT_CHECKPOINT_SHA256
            or launch['parent_summary_sha256']!=PARENT_SUMMARY_SHA256
            or launch['reference_parent']['lifetime_updates']!=9112):
        raise ValueError('exact unadopted research parent and retained reference required')


def run(output, *, launch_sha256):
    started,cpu=time.monotonic(),time.process_time();output=Path(output).resolve()
    if digest(output/'launch.json')!=launch_sha256:raise ValueError('pinned paired launch required')
    launch=read(output/'launch.json');validate_launch(launch)
    deadline=started+1200;target=output/'execution';native(target).mkdir(exist_ok=False)
    receipt=dict(schema=SCHEMA,status='running',launch_sha256=launch_sha256,started_utc=utc(),
        physical_plan=deepcopy(launch['physical_plan']),
        new_updates={a:0 for a in ARMS},teacher_calls=0,archive_load_attempts=0,archive_loads=0,archive_bytes=0,
        definition_json_load_attempts=0,definition_json_loads=0,restorations=[],snapshot_attempts=0,snapshots=[],
        evaluations={},endpoint_commits={},continuations={},steps=[],artifacts={},evaluation_reports=[],
        closed_preparations=[],partial_work_unknown=False,publications=dict(attempts=0,completions=0),
        replay_target_work=dict(batch_attempts=0,batch_completions=0,row_completions=0,
            completed_english_checks=0,completed_typed_checks=0,partial_batch_internal_work_unknown=False))
    learner=owner=arm=active_bank=active_ledger=active_evaluation=None
    latest={};evaluation_accounted=False;definition_before=basis_before=complementary_before=old_validation_work=None

    def boundary():
        if time.monotonic()>=deadline:raise TimeoutError('shared1200-second paired allowance ended')

    def artifact(name,value,checkpoint=False):
        boundary();receipt['publications']['attempts']+=1
        pin=publish(target/name,value,checkpoint=checkpoint);receipt['publications']['completions']+=1
        record=dict(path=relative_root(target/name),sha256=pin);receipt['artifacts'][name]=record;return record

    def authenticated(path,pin):
        boundary();raw=native(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('immutable paired input changed')
        return raw

    def archive(record):
        raw=authenticated(ROOT/record['path'],record['sha256']);receipt['archive_bytes']+=len(raw)
        receipt['archive_load_attempts']+=1
        value=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);receipt['archive_loads']+=1;return value

    def authenticate():
        boundary();verify_pins(launch['input_sha256'])
        if source_hashes()!=launch['source_sha256']:raise ValueError('paired source closure changed')
        for name,local in launch['source_snapshots'].items():
            if digest(output/local)!=launch['source_sha256'][name]:raise ValueError('paired source snapshot changed')
        boundary()

    def state():
        return dict(cursor=learner.cursor,accounting=learner.accounting,evidence=learner.evidence,last_report=learner.last_report)

    def close_learner(step):
        nonlocal learner,owner
        if learner is not None:latest[arm]=state()
        if owner is not None:
            owner.close();receipt['closed_preparations'].append(dict(arm=arm,step=step,report=owner.report()))
        learner=owner=None;gc.collect();torch.cuda.empty_cache()

    def restore(record,step):
        boundary();raw=authenticated(ROOT/record['path'],record['sha256'])
        try:
            restored=SharedStateContinuation.from_snapshot(raw,expected_sha256=record['sha256'],
                expected_identity=launch['origin_identity'],device='cuda:0')
        except BaseException as error:
            receipt['restorations'].append(dict(arm=arm,step=step,report=getattr(error,'continuation_report',dict(failed=True))));raise
        receipt['restorations'].append(dict(arm=arm,step=step,report=restored.last_restore_report));return restored

    def replay_targets(lesson):
        targets={};counts=receipt['replay_target_work']
        for family in definition_curriculum.FAMILIES:
            rows=lesson['bundle']['families'][family];counts['batch_attempts']+=1
            try:targets[family]=shared_state_targets.pack_state_targets(rows)
            except BaseException:
                counts['partial_batch_internal_work_unknown']=True;raise
            counts['batch_completions']+=1;counts['row_completions']+=len(rows)
            counts['completed_english_checks']+=len(rows);counts['completed_typed_checks']+=len(rows)
        return targets

    def score_bank(name,bank,provider):
        nonlocal active_bank,active_ledger,active_evaluation,evaluation_accounted
        boundary();active_bank=bank;active_evaluation=name;evaluation_accounted=False
        active_ledger=provider.EvaluationLedger();kwargs=dict(batch_size=32,deadline=deadline,work=active_ledger)
        if provider is old_evaluation:kwargs['control']='normal'
        scored=bank.score(learner.model,**kwargs)
        receipt['evaluation_reports'].append(dict(arm=arm,name=name,report=deepcopy(bank.last_report),work=active_ledger.report()))
        evaluation_accounted=True;return scored

    def clear_evaluation():
        nonlocal active_bank,active_ledger,active_evaluation,evaluation_accounted
        active_bank=active_ledger=active_evaluation=None;evaluation_accounted=False

    def evaluate(step):
        results={}
        for prefix,banks,provider in (('basis',basis_banks,basis_evaluation),('definition',definition_banks,definition_evaluation)):
            for name,bank in banks.items():
                full=f'{prefix}_{name}';score=score_bank(f'{step}/{full}',bank,provider)
                record=artifact(f'scores/{arm}/{step:04d}/{full}.json',score)
                results[full]=dict(metrics=score['metrics'],record=record);clear_evaluation()
        for name,bank in complementary_banks.items():
            score=score_bank(f'{step}/{name}',bank,complementary_evaluation)
            results[name]=dict(metrics=score['metrics'],record=artifact(f'scores/{arm}/{step:04d}/{name}.json',score))
            clear_evaluation()
        for name,groups in old_banks.items():
            rows,records=[],{}
            for turns,bank in groups.items():
                score=score_bank(f'{step}/{name}/t{turns}',bank,old_evaluation)
                rows.extend(score['raw_records']);records[str(turns)]=artifact(f'scores/{arm}/{step:04d}/{name}-{turns}.json',score)
                clear_evaluation()
            results[name]=dict(metrics=old_evaluation.score_records(rows),groups=records)
        if set(results)!=set(BANK_NAMES):raise ValueError('all16 native banks required')
        receipt['evaluations'].setdefault(arm,{})[str(step)]=results

    try:
        publish(target/'started.json',receipt);authenticate()
        import torch
        from experiments import execution_profile, complementary_composition_data as data, definition_basis_data as basis_data
        from experiments import definition_basis_evaluation as basis_evaluation, definition_basis_curriculum as basis_curriculum
        from experiments import complementary_composition_evaluation as complementary_evaluation, complementary_composition_curriculum as complementary_curriculum
        from experiments import definition_evaluation, definition_curriculum, shared_state_targets
        from experiments import foundation_layout_evaluation as old_evaluation, shared_acquisition_transfer_data as bank_data
        from experiments.foundation_layout_prepared import PreparedLayoutOwner, evidence_sha256
        from experiments.foundation_layout_curriculum import WorkLedger
        from experiments.shared_state_continuation import SharedStateContinuation
        from experiments.sequence_student import SequenceConfig
        from brain_in_computer.dialogue_student import checkpoint_digest
        torch.set_num_threads(1);torch.set_num_interop_threads(1);execution_profile.configure_strict_profile()
        receipt['runtime']=execution_profile.runtime_profile()
        if receipt['runtime']!=launch['expected_runtime']:raise ValueError('original strict runtime required')
        config=SequenceConfig(**CONFIG);data_dir=ROOT/launch['data_directory']
        definition_before=definition_curriculum.work_report();basis_before=basis_curriculum.work_report()
        complementary_before=complementary_curriculum.work_report();old_validation_work=WorkLedger()
        manifest=data.load_manifest(data_dir,launch['data_manifest_sha256'])
        if validate_bindings(manifest,launch['parent'])!=launch['physical_plan']:
            raise ValueError('actual paired curriculum plan changed')
        record=manifest['banks'];extra=json.loads(authenticated(ROOT/record['path'],record['sha256']))
        if set(extra)!=set(EXTRA_BANK_NAMES) or any(identity(rows)!=manifest['bank_inventory'][name]['rows_sha256'] for name,rows in extra.items()):
            raise ValueError('supplementary exact fit/development rows differ')
        complementary_banks={name:complementary_evaluation.PreparedDefinitionBank(rows,
            role='train_fit' if '_fit_' in name else 'dev',config=config) for name,rows in extra.items()}
        del extra
        basis_dir=ROOT/launch['basis_data_directory']
        basis_manifest=basis_data.load_manifest(basis_dir,launch['basis_data_manifest_sha256'])
        record=basis_manifest['banks'];banks=json.loads(authenticated(basis_dir/record['path'],record['sha256']))
        if set(banks)!=set(basis_data.PANELS) or any(identity(rows)!=basis_manifest['bank_inventory'][name]['rows_sha256'] for name,rows in banks.items()):
            raise ValueError('basis bank rows changed')
        basis_banks={name:basis_evaluation.PreparedDefinitionBank(rows,role='dev',config=config) for name,rows in banks.items()}
        record=basis_manifest['old_banks'];banks=json.loads(authenticated(ROOT/record['path'],record['sha256']))
        if set(banks)!={'binding','revision','composition'} or any(identity(rows)!=basis_manifest['old_bank_inventory'][name]['rows_sha256'] for name,rows in banks.items()):
            raise ValueError('prior definition bank rows changed')
        definition_banks={name:definition_evaluation.PreparedDefinitionBank(rows,role='dev',config=config) for name,rows in banks.items()}
        evaluation=launch['evaluation'];old_manifest=bank_data.load_manifest(ROOT/evaluation['directory'],expected_manifest_sha256=evaluation['manifest_sha256'])
        record=old_manifest['banks'];payload=archive(dict(path=relative_root(ROOT/evaluation['directory']/record['path']),sha256=record['sha256']))
        if set(payload)!={'dev','train_fit','retention','transfer_original','transfer_varied'}:raise ValueError('all five broad banks required')
        old_banks={}
        for name,bank in payload.items():
            if identity(bank['rows'])!=old_manifest['bank_inventory'][name]['rows_sha256']:raise ValueError('broad evaluation row identity differs')
            groups=defaultdict(list)
            for row in bank['rows']:groups[len(row['turns'])].append(row)
            old_banks[name]={turns:old_evaluation.PreparedLayoutBank(rows,role=bank['role'],config=config,validation_work=old_validation_work)
                for turns,rows in sorted(groups.items())}
        del banks,payload
        for arm in launch['physical_plan']['physical_arms']:
            learner=restore(launch['parent']['checkpoint'],0)
            if (learner.cursor!=PARENT_CURSOR or checkpoint_digest(learner.model)!=launch['parent']['weights_sha256']
                    or learner.accounting['lifetime_kernel']!=launch['parent_metadata']['accounting']
                    or learner.evidence!=launch['parent_metadata']['evidence']
                    or identity(learner.recipe)!=identity(launch['parent_metadata']['recipe'])):
                raise ValueError('restored full paired parent differs')
            owner=PreparedLayoutOwner(config=config,layout='original',micro_batch_size=MICRO)
            evaluate(0)
            if {name:receipt['evaluations'][arm]['0'][name]['metrics'] for name in BASE_BANK_NAMES}!=launch['parent']['metrics']:
                raise ValueError('all12 baseline native scores must reproduce exact parent')
            if arm=='curriculum' and {k:v['metrics'] for k,v in receipt['evaluations'][arm]['0'].items()}!={k:v['metrics'] for k,v in receipt['evaluations']['control']['0'].items()}:
                raise ValueError('both arms must reproduce all16 baseline metrics')
            records=manifest['phases']['training'][arm]
            for index,record in enumerate(records):
                boundary();kind=record['kind'];receipt['active_operation']=dict(arm=arm,index=index,kind=kind,stage='lesson')
                if kind in ('basis','old_definition','complementary'):
                    receipt['definition_json_load_attempts']+=1;lesson=data.load_json_image(record);receipt['definition_json_loads']+=1
                    provider={'basis':basis_curriculum,'old_definition':definition_curriculum,'complementary':complementary_curriculum}[kind]
                    if lesson['admission_schema']!=provider.VERSION:raise ValueError('honest definition provider required')
                    labels={f:torch.tensor(value,dtype=torch.long) for f,value in lesson['state_targets'].items()}
                elif kind=='replay':lesson=archive(record);labels=replay_targets(lesson)
                else:raise ValueError('unsupported paired lesson provider')
                if identity(lesson['expected_evidence'])!=lesson['expected_evidence_sha256']:raise ValueError('canonical lesson evidence differs')
                original=record['original_cursor'] if kind=='replay' else record['index']
                if lesson['bundle']['bundle_id']!=original or lesson['expected_evidence']['bundle_id']!=original or learner.cursor!=PARENT_CURSOR+index:
                    raise ValueError('source/global consumption cursor differs')
                bundle=deepcopy(lesson['bundle']);evidence=deepcopy(lesson['expected_evidence'])
                bundle['bundle_id']=evidence['bundle_id']=learner.cursor
                token=owner.prepare(bundle,expected_evidence=evidence,expected_evidence_sha256=evidence_sha256(evidence))
                receipt['active_operation']['stage']='step'
                report=learner.step(token,state_targets=labels,deadline=deadline);receipt['new_updates'][arm]+=report['physical_optimizer_updates']
                receipt['steps'].append(artifact(f'steps/{arm}/{index:04d}.json',dict(arm=arm,relative_step=index+1,kind=kind,
                    source_record=record,consumed_cursor=bundle['bundle_id'],report=report)))
                receipt['active_operation']=None;del lesson,labels,bundle,evidence,token,report
                if (index+1)%54==0:print(json.dumps(dict(event='progress',arm=arm,updates=index+1,wall_seconds=time.monotonic()-started)),flush=True)
                if index+1 in ENDPOINTS:
                    boundary();receipt['snapshot_attempts']+=1;snapshot=learner.snapshot()
                    checkpoint=artifact(f'checkpoints/{arm}-{index+1:04d}.pt',snapshot,checkpoint=True)
                    receipt['snapshots'].append(dict(arm=arm,relative_step=index+1,checkpoint=checkpoint))
                    saved_weights=snapshot['weights_sha256'];del snapshot
                    close_learner(index+1);closed=deepcopy(latest[arm]);learner=restore(checkpoint,index+1)
                    if (learner.cursor!=PARENT_CURSOR+index+1 or checkpoint_digest(learner.model)!=saved_weights
                            or learner.accounting['lifetime_kernel']!=closed['accounting']['lifetime_kernel'] or learner.evidence!=closed['evidence']):
                        raise ValueError('exact endpoint full-state restart differs')
                    owner=PreparedLayoutOwner(config=config,layout='original',micro_batch_size=MICRO)
                    evaluate(index+1);authenticate()
                    commit=dict(schema=SCHEMA,launch_sha256=launch_sha256,arm=arm,relative_step=index+1,lifetime_updates=learner.cursor,
                        parent_checkpoint=launch['parent']['checkpoint'],data_manifest_sha256=launch['data_manifest_sha256'],
                        checkpoint=checkpoint,weights_sha256=saved_weights,evidence=learner.evidence,accounting=learner.accounting,
                        scores=deepcopy(receipt['evaluations'][arm][str(index+1)]))
                    receipt['endpoint_commits'].setdefault(arm,{})[str(index+1)]=artifact(f'endpoint-{arm}-{index+1:04d}.json',commit)
                    if index+1==UPDATES:
                        receipt['continuations'][arm]=dict(checkpoint=checkpoint,weights_sha256=saved_weights,lifetime_updates=learner.cursor,
                            metrics={k:v['metrics'] for k,v in commit['scores'].items()},scores=deepcopy(commit['scores']))
            close_learner(UPDATES)
        physical={a:accounting_delta(value['accounting']['lifetime_kernel'],launch['parent_metadata']['accounting']) for a,value in latest.items()}
        validate_completed_work(launch['physical_plan'],receipt,physical)
        authenticate();receipt['status']='completed'
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if learner is not None:latest[arm]=state()
        receipt['latest_arms']=deepcopy(latest);receipt['returned_step_updates']=deepcopy(receipt['new_updates'])
        receipt['physical_training']={a:accounting_delta(value['accounting']['lifetime_kernel'],launch['parent_metadata']['accounting']) for a,value in latest.items()}
        receipt['physical_totals']={group:{key:sum(v[group][key] for v in receipt['physical_training'].values())
            for key in launch['parent_metadata']['accounting'][group]} for group in ('work','state_work','cost')}
        receipt['partial_work_unknown']=bool(receipt['physical_totals']['work']['unknown_optimizer_outcomes'])
        if active_bank is not None:receipt['active_evaluation']=dict(arm=arm,name=active_evaluation,report=deepcopy(active_bank.last_report),
            work=active_ledger.report(),already_in_completed_evaluation_reports=evaluation_accounted)
        try:
            if owner is not None:
                owner.close();receipt['closed_preparations'].append(dict(arm=arm,step=receipt['new_updates'][arm],report=owner.report()))
            if 'torch' in locals() and torch.cuda.is_initialized():
                torch.cuda.synchronize();receipt['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated();receipt['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved()
        except BaseException as error:receipt.update(status='failed',cleanup_error=repr(error),partial_work_unknown=True)
        receipt['old_bank_validation_work']=old_validation_work.report() if old_validation_work is not None else None
        receipt['definition_admission_work']={k:v-definition_before[k] for k,v in definition_curriculum.work_report().items()} if definition_before is not None else None
        receipt['basis_admission_work']={k:v-basis_before[k] for k,v in basis_curriculum.work_report().items()} if basis_before is not None else None
        receipt['complementary_admission_work']={k:v-complementary_before[k] for k,v in complementary_curriculum.work_report().items()} if complementary_before is not None else None
        if receipt['status']!='completed':receipt['continuations']={}
        receipt.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,ended_utc=utc())
        publish(target/'summary.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','run'));parser.add_argument('directory')
    for name in ('parent-directory','parent-summary-sha256','data-directory','data-manifest-sha256','launch-sha256'):parser.add_argument('--'+name)
    parser.add_argument('--max-seconds',type=int,default=1200);args=parser.parse_args()
    if args.mode=='freeze':
        print(json.dumps(dict(launch_sha256=freeze(args.directory,parent_directory=args.parent_directory,parent_summary_sha256=args.parent_summary_sha256,
            data_directory=args.data_directory,data_manifest_sha256=args.data_manifest_sha256,max_seconds=args.max_seconds))))
    else:
        value=run(args.directory,launch_sha256=args.launch_sha256);print(json.dumps({k:value[k] for k in ('status','new_updates','wall_seconds')}))
