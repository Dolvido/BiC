"""Native acquisition of reusable English-defined operations across three worlds.

Six-step groups balance two basis lessons, one prior-definition lesson, and
three broad replay lessons. All 648 steps and four endpoints are fixed before
training. Both full optimizer and native observation-only evaluation persist.
"""
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

SCHEMA = 'bic-definition-basis-learning-v1'
UPDATES, MICRO = 648, 32
ENDPOINTS = (216, 432, 648)
PROTOCOL = 'docs/DEFINITION_BASIS_PROTOCOL.md'


def source_hashes():
    from experiments import definition_basis_data as data, definition_evaluation as evaluation
    from experiments import definition_basis_evaluation as basis_evaluation
    from experiments import shared_state_continuation as bridge
    from experiments import shared_acquisition_transfer_data as banks
    from experiments import foundation_layout_evaluation as old_evaluation
    from experiments import foundation_layout_prepared as prepared, shared_state_targets as targets
    result = {**data.source_hashes(), **evaluation.source_hashes(), **basis_evaluation.source_hashes(), **bridge.source_hashes(),
        **banks.source_hashes(), **old_evaluation.source_hashes(), **prepared.source_hashes(), **targets.source_hashes()}
    for name in ('experiments/definition_basis_learning.py','experiments/continuous_tutor_worker.py',PROTOCOL):
        result[name] = digest(ROOT/name)
    return result


def freeze(output, *, campaign_directory, campaign_summary_sha256,
           data_directory, data_manifest_sha256, max_seconds=900):
    started, cpu = time.monotonic(), time.process_time()
    if type(max_seconds) is not int or max_seconds != 900:
        raise ValueError('protocol-fixed900-second local allowance required')
    from experiments import definition_basis_data as data, continuous_tutor_worker as worker
    from experiments import shared_acquisition_transfer_data as bank_data
    output, campaign_directory, data_directory = map(lambda p:Path(p).resolve(),
        (output,campaign_directory,data_directory))
    if any(not p.is_relative_to(ROOT) for p in (output,campaign_directory,data_directory)):
        raise ValueError('repository paths required')
    native(output).mkdir(parents=True,exist_ok=False)
    receipt=dict(schema=SCHEMA,status='running',checkpoint_load_attempts=0,checkpoint_loads=0,models=0,teacher_calls=0,updates=0)
    publish(output/'freeze-started.json',receipt)
    try:
        summary_path=campaign_directory/'execution/summary.json'
        if digest(summary_path) != campaign_summary_sha256:
            raise ValueError('caller-pinned terminal campaign summary required')
        summary, previous = read(summary_path), read(campaign_directory/'launch.json')
        if (summary['status']!='completed' or summary['partial_work_unknown']
                or summary['launch_sha256']!=digest(campaign_directory/'launch.json')
                or any(child.get('exit_code')!=0 for child in summary.get('worker_processes',[]))):
            raise ValueError('completed campaign with known terminal workers required before new GPU run')
        verify_pins(previous['source_sha256']);verify_pins(previous['input_sha256'])
        record=summary['last_state'];state=read(ROOT/record['path'])
        if (digest(ROOT/record['path'])!=record['sha256'] or state['parent']!=summary['parent']
                or state['physical_totals']!=summary['physical_totals']):
            raise ValueError('selected durable learner and recorded work differ')
        manifest=data.load_manifest(data_directory,data_manifest_sha256)
        evaluation=previous['evaluation']
        bank_data.load_manifest(ROOT/evaluation['directory'],expected_manifest_sha256=evaluation['manifest_sha256'])
        parent=state['parent']
        raw=native(ROOT/parent['checkpoint']['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=parent['checkpoint']['sha256']:
            raise ValueError('selected parent checkpoint changed')
        import torch
        receipt['checkpoint_load_attempts']+=1
        payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);receipt['checkpoint_loads']+=1
        metadata=worker.parent_metadata(payload);parent_bridge_cost=deepcopy(payload['bridge_cost']);del payload,raw
        if (metadata['origin']!=previous['origin_identity'] or metadata['cursor']!=parent['lifetime_updates']
                or metadata['weights_sha256']!=parent['weights_sha256'] or metadata['recipe']['config']!=CONFIG):
            raise ValueError('unchanged original native learner required')
        pins={relative_root(summary_path):campaign_summary_sha256,
            relative_root(campaign_directory/'launch.json'):summary['launch_sha256'],
            record['path']:record['sha256'],parent['checkpoint']['path']:parent['checkpoint']['sha256'],
            relative_root(data_directory/'manifest.json'):data_manifest_sha256,
            relative_root(data_directory/'preparation.json'):digest(data_directory/'preparation.json'),
            relative_root(ROOT/evaluation['directory']/'manifest.json'):evaluation['manifest_sha256']}
        verify_pins(pins)
        sources=source_hashes();snapshots={}
        for index,(name,expected) in enumerate(sorted(sources.items())):
            raw=native(ROOT/name).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('source changed while freezing')
            local=f'sources/{index:03d}.bin';path=native(output/local);path.parent.mkdir(exist_ok=True)
            with path.open('xb') as stream:stream.write(raw)
            snapshots[name]=local
        if source_hashes()!=sources:raise ValueError('source changed while freezing')
        launch=dict(schema=SCHEMA,created_utc=utc(),source_sha256=sources,source_snapshots=snapshots,input_sha256=pins,
            parent=deepcopy(parent),parent_metadata=metadata,parent_bridge_cost=parent_bridge_cost,origin_identity=previous['origin_identity'],
            expected_runtime=previous['parent_runtime'],evaluation=evaluation,
            data_directory=relative_root(data_directory),data_manifest_sha256=data_manifest_sha256,
            max_seconds=max_seconds,updates=UPDATES,micro_batch_size=MICRO,config=CONFIG,
            endpoints=[0,*ENDPOINTS],training_basis_episodes=20736,training_prior_definition_episodes=10368,training_replay_episodes=31104,
            teacher_calls=0,automatic_retry=False,automatic_promotion=False,
            parent_campaign_outcome=summary['status'],primary_metric='all_query_pair_both',
            scope='Shared operation grounding and unseen composition; full native continuation with retained broad and prior-definition practice.')
        pin=publish(output/'launch.json',launch)
        receipt.update(status='completed',launch_sha256=pin)
        return pin
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
        publish(output/'freeze-receipt.json',receipt)


def run(output, *, launch_sha256):
    started,cpu=time.monotonic(),time.process_time()
    output=Path(output).resolve()
    if digest(output/'launch.json')!=launch_sha256:
        raise ValueError('pinned definition launch required')
    launch=read(output/'launch.json')
    if (launch['schema']!=SCHEMA or launch['updates']!=UPDATES or launch['micro_batch_size']!=MICRO
            or launch['max_seconds']!=900 or launch['endpoints']!=[0,*ENDPOINTS]
            or launch['teacher_calls']!=0 or launch['automatic_retry'] or launch['automatic_promotion']):
        raise ValueError('fixed complete acquisition required')
    deadline=started+launch['max_seconds'];target=output/'execution';native(target).mkdir(exist_ok=False)
    receipt=dict(schema=SCHEMA,status='running',launch_sha256=launch_sha256,started_utc=utc(),
        updates=0,teacher_calls=0,archive_load_attempts=0,archive_loads=0,archive_bytes=0,
        definition_json_load_attempts=0,definition_json_loads=0,snapshot_attempts=0,
        restorations=[],snapshots=[],evaluations={},artifacts={},steps=[],partial_work_unknown=False,
        closed_preparations=[],evaluation_reports=[],endpoint_commits={},
        publications=dict(attempts=0,completions=0),
        replay_target_work=dict(batch_attempts=0,batch_completions=0,row_completions=0,
            completed_english_checks=0,completed_typed_checks=0,partial_batch_internal_work_unknown=False))
    learner=owner=last_closed_state=active_bank=active_ledger=active_evaluation=None
    evaluation_accounted=False
    definition_work_before=basis_work_before=old_validation_work=None

    def boundary():
        if time.monotonic()>=deadline:raise TimeoutError('fixed definition run allowance ended')

    def artifact(name,value,checkpoint=False):
        boundary();receipt['publications']['attempts']+=1
        pin=publish(target/name,value,checkpoint=checkpoint);receipt['publications']['completions']+=1
        record=dict(path=relative_root(target/name),sha256=pin);receipt['artifacts'][name]=record
        return record

    def authenticated(path,pin):
        boundary();raw=native(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('authenticated input changed')
        return raw

    def archive(record):
        raw=authenticated(ROOT/record['path'],record['sha256'])
        receipt['archive_bytes']+=len(raw)
        receipt['archive_load_attempts']+=1
        result=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        receipt['archive_loads']+=1
        return result

    def restore(record):
        raw=authenticated(ROOT/record['path'],record['sha256'])
        try:
            restored=SharedStateContinuation.from_snapshot(raw,expected_sha256=record['sha256'],
                expected_identity=launch['origin_identity'],device='cuda:0')
        except BaseException as error:
            receipt['restorations'].append(getattr(error,'continuation_report',dict(failed=True)));raise
        receipt['restorations'].append(restored.last_restore_report)
        return restored

    def authenticate():
        verify_pins(launch['input_sha256'])
        if source_hashes()!=launch['source_sha256']:raise ValueError('frozen acquisition source changed')
        for name,local in launch['source_snapshots'].items():
            if digest(output/local)!=launch['source_sha256'][name]:raise ValueError('frozen source image changed')

    def learner_state():
        return dict(cursor=learner.cursor,accounting=learner.accounting,evidence=learner.evidence,
                    last_report=learner.last_report)

    def close_learner(step):
        nonlocal learner,owner,last_closed_state
        # Preserve actual committed/attempted work before releasing the learner;
        # a subsequent failed restore must not erase the completed first half.
        last_closed_state=learner_state()
        owner.close()
        receipt['closed_preparations'].append(dict(step=step,report=owner.report()))
        learner=owner=None;gc.collect();torch.cuda.empty_cache()

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

    def score_bank(name,bank,*,legacy=False,basis=False):
        nonlocal active_bank,active_ledger,active_evaluation,evaluation_accounted
        boundary();active_bank=bank;active_evaluation=name;evaluation_accounted=False
        active_ledger=(old_evaluation if legacy else basis_evaluation if basis else definition_evaluation).EvaluationLedger()
        kwargs=dict(batch_size=32,deadline=deadline,work=active_ledger)
        if legacy:kwargs['control']='normal'
        scored=bank.score(learner.model,**kwargs)
        receipt['evaluation_reports'].append(dict(name=name,report=deepcopy(bank.last_report)))
        evaluation_accounted=True
        return scored

    def clear_evaluation():
        nonlocal active_bank,active_ledger,active_evaluation,evaluation_accounted
        active_bank=active_ledger=active_evaluation=None;evaluation_accounted=False

    def evaluate(step):
        results={}
        for name,bank in basis_banks.items():
            score=score_bank(f'{step}/basis-{name}',bank,basis=True)
            record=artifact(f'scores/{step:04d}/basis-{name}.json',score)
            results['basis_'+name]=dict(metrics=score['metrics'],record=record)
            clear_evaluation()
        for name,bank in definition_banks.items():
            score=score_bank(f'{step}/definition-{name}',bank)
            record=artifact(f'scores/{step:04d}/definition-{name}.json',score)
            results['definition_'+name]=dict(metrics=score['metrics'],record=record)
            clear_evaluation()
        for name,groups in old_banks.items():
            rows,records=[],{}
            for turns,bank in groups.items():
                score=score_bank(f'{step}/{name}/t{turns}',bank,legacy=True)
                rows.extend(score['raw_records']);records[str(turns)]=artifact(f'scores/{step:04d}/{name}-{turns}.json',score)
                clear_evaluation()
            results[name]=dict(metrics=old_evaluation.score_records(rows),groups=records)
        receipt['evaluations'][str(step)]=results

    try:
        publish(target/'started.json',receipt);authenticate()
        import torch
        from experiments import execution_profile, definition_basis_data as data, definition_evaluation
        from experiments import definition_basis_evaluation as basis_evaluation, definition_basis_curriculum as basis_curriculum
        from experiments import foundation_layout_evaluation as old_evaluation, shared_acquisition_transfer_data as bank_data
        from experiments import definition_curriculum, shared_state_targets
        from experiments.foundation_layout_prepared import PreparedLayoutOwner, evidence_sha256
        from experiments.foundation_layout_curriculum import WorkLedger
        from experiments.shared_state_continuation import SharedStateContinuation
        from experiments.sequence_student import SequenceConfig
        from brain_in_computer.dialogue_student import checkpoint_digest
        torch.set_num_threads(1);torch.set_num_interop_threads(1);execution_profile.configure_strict_profile()
        if execution_profile.runtime_profile()!=launch['expected_runtime']:raise ValueError('original strict runtime required')
        config=SequenceConfig(**CONFIG);data_dir=ROOT/launch['data_directory']
        definition_work_before=definition_curriculum.work_report();basis_work_before=basis_curriculum.work_report();old_validation_work=WorkLedger()
        manifest=data.load_manifest(data_dir,launch['data_manifest_sha256'])
        record=manifest['banks'];banks=json.loads(authenticated(data_dir/record['path'],record['sha256']))
        if set(banks)!=set(data.PANELS) or any(data.identity(rows)!=manifest['bank_inventory'][name]['rows_sha256'] for name,rows in banks.items()):
            raise ValueError('definition evaluation catalogue differs')
        basis_banks={name:basis_evaluation.PreparedDefinitionBank(rows,role='dev',config=config)
            for name,rows in banks.items()}
        prior_record=manifest['old_banks']
        prior_banks=json.loads(authenticated(ROOT/prior_record['path'],prior_record['sha256']))
        if (set(prior_banks)!={'binding','revision','composition'}
                or any(data.identity(rows)!=manifest['old_bank_inventory'][name]['rows_sha256'] for name,rows in prior_banks.items())):
            raise ValueError('all original definition panels required unchanged')
        definition_banks={name:definition_evaluation.PreparedDefinitionBank(rows,role='dev',config=config)
            for name,rows in prior_banks.items()}
        del prior_banks
        evaluation=launch['evaluation'];old_manifest=bank_data.load_manifest(ROOT/evaluation['directory'],
            expected_manifest_sha256=evaluation['manifest_sha256'])
        old_record=old_manifest['banks']
        old_payload=archive(dict(path=relative_root(ROOT/evaluation['directory']/old_record['path']),sha256=old_record['sha256']))
        if set(old_payload)!=set(launch['parent']['metrics']):raise ValueError('all five original native banks required')
        old_banks={}
        for name,bank in old_payload.items():
            if data.identity(bank['rows'])!=old_manifest['bank_inventory'][name]['rows_sha256']:
                raise ValueError('original evaluation row identity differs')
            groups=defaultdict(list)
            for row in bank['rows']:groups[len(row['turns'])].append(row)
            old_banks[name]={turns:old_evaluation.PreparedLayoutBank(rows,role=bank['role'],config=config,validation_work=old_validation_work)
                for turns,rows in groups.items()}
        del banks,old_payload
        learner=restore(launch['parent']['checkpoint'])
        if (learner.cursor!=launch['parent']['lifetime_updates'] or checkpoint_digest(learner.model)!=launch['parent']['weights_sha256']
                or learner.accounting['lifetime_kernel']!=launch['parent_metadata']['accounting']
                or learner.evidence!=launch['parent_metadata']['evidence']
                or data.identity(learner.recipe)!=data.identity(launch['parent_metadata']['recipe'])):
            raise ValueError('restored native parent weights, evidence, recipe or accounting differs')
        owner=PreparedLayoutOwner(config=config,layout='original',micro_batch_size=MICRO)
        evaluate(0)
        if {name:receipt['evaluations']['0'][name]['metrics'] for name in old_banks}!=launch['parent']['metrics']:
            raise ValueError('reloaded parent broad native scores differ')
        for index,item in enumerate(manifest['schedule']):
            boundary();kind=item['kind'];source_index=item['source_index']
            record=manifest[{'basis':'training','old_definition':'old_definition','replay':'replay'}[kind]][source_index]
            if kind in ('basis','old_definition'):
                receipt['definition_json_load_attempts']+=1
                lesson=json.loads(authenticated((data_dir if kind=='basis' else ROOT)/record['path'],record['sha256']))
                receipt['definition_json_loads']+=1
                provider=basis_curriculum if kind=='basis' else definition_curriculum
                if lesson['admission_schema']!=provider.VERSION:raise ValueError('honest lesson provider admission required')
                labels={family:torch.tensor(value,dtype=torch.long) for family,value in lesson['state_targets'].items()}
            else:
                lesson=archive(record)
                labels=replay_targets(lesson)
            if data.identity(lesson['expected_evidence'])!=lesson['expected_evidence_sha256']:
                raise ValueError('lesson evidence differs')
            original_cursor=record['original_cursor'] if kind=='replay' else record['index']
            if (lesson['bundle']['bundle_id']!=original_cursor or lesson['expected_evidence']['bundle_id']!=original_cursor
                    or learner.cursor!=launch['parent']['lifetime_updates']+index):
                raise ValueError('declared source or continuing consumption cursor differs')
            bundle=deepcopy(lesson['bundle']);evidence=deepcopy(lesson['expected_evidence'])
            bundle['bundle_id']=evidence['bundle_id']=learner.cursor
            token=owner.prepare(bundle,expected_evidence=evidence,expected_evidence_sha256=evidence_sha256(evidence))
            report=learner.step(token,state_targets=labels,deadline=deadline)
            receipt['updates']+=report['physical_optimizer_updates']
            receipt['steps'].append(artifact(f'steps/{index:04d}.json',dict(kind=kind,source_index=source_index,
                source_record=record,consumed_cursor=bundle['bundle_id'],report=report)))
            del lesson,labels,bundle,evidence,token,report
            if (index+1)%18==0:
                print(json.dumps(dict(event='progress',updates=index+1,wall_seconds=time.monotonic()-started)),flush=True)
            if index+1 in ENDPOINTS:
                boundary();receipt['snapshot_attempts']+=1
                snapshot=learner.snapshot();record=artifact(f'checkpoints/{index+1:04d}.pt',snapshot,checkpoint=True)
                receipt['snapshots'].append(record)
                saved_weights=snapshot['weights_sha256'];del snapshot
                close_learner(index+1)
                learner=restore(record);owner=PreparedLayoutOwner(config=config,layout='original',micro_batch_size=MICRO)
                if (learner.cursor!=launch['parent']['lifetime_updates']+index+1
                        or learner.accounting['lifetime_kernel']!=last_closed_state['accounting']['lifetime_kernel']
                        or learner.evidence!=last_closed_state['evidence'] or checkpoint_digest(learner.model)!=saved_weights):
                    raise ValueError('exact completed phase restart differs')
                evaluate(index+1);authenticate()
                commit=dict(schema=SCHEMA,launch_sha256=launch_sha256,data_manifest_sha256=launch['data_manifest_sha256'],
                    relative_step=index+1,lifetime_updates=learner.cursor,checkpoint=record,weights_sha256=saved_weights,
                    evidence=learner.evidence,accounting=learner.accounting,scores=deepcopy(receipt['evaluations'][str(index+1)]))
                receipt['endpoint_commits'][str(index+1)]=artifact(f'endpoint-{index+1:04d}.json',commit)
        if receipt['updates']!=UPDATES:raise ValueError('complete measured update count required')
        receipt.update(status='completed',endpoint=receipt['snapshots'][-1],
            lifetime_updates=learner.cursor,final_accounting=learner.accounting)
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        latest=learner_state() if learner is not None else last_closed_state
        if latest is not None:
            receipt['latest_accounting']=latest['accounting']
            receipt['latest_evidence']=latest['evidence'];receipt['latest_cursor']=latest['cursor']
            receipt['last_step_report']=latest['last_report']
            actual=latest['accounting']['lifetime_kernel'];parent=launch['parent_metadata']['accounting']
            receipt['physical_training']={group:{key:value-parent[group][key] for key,value in actual[group].items()}
                for group in ('work','state_work','cost')}
            receipt['new_bridge_cost']={key:value-launch['parent_bridge_cost'][key]
                for key,value in latest['accounting']['bridge_cost'].items()}
            receipt['partial_work_unknown']=bool(receipt['physical_training']['work']['unknown_optimizer_outcomes'])
        else:
            receipt['physical_training']={group:dict.fromkeys(launch['parent_metadata']['accounting'][group],0)
                for group in ('work','state_work','cost')}
        receipt['returned_step_updates']=receipt['updates']
        if owner is not None:
            owner.close();receipt['preparation']=owner.report()
            receipt['closed_preparations'].append(dict(step=receipt['updates'],report=owner.report()))
        receipt['preparation_totals']={}
        for item in receipt['closed_preparations']:
            for key,value in item['report'].items():
                if type(value) in (int,float):
                    receipt['preparation_totals'][key]=receipt['preparation_totals'].get(key,0)+value
        if active_bank is not None:
            receipt['active_evaluation']=dict(name=active_evaluation,report=deepcopy(active_bank.last_report),
                work=active_ledger.report(),already_in_completed_evaluation_reports=evaluation_accounted)
        try:
            if 'torch' in locals() and torch.cuda.is_initialized():
                torch.cuda.synchronize();receipt['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
                receipt['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved()
        except BaseException as error:
            receipt.update(status='failed',cleanup_error=repr(error),partial_work_unknown=True)
        receipt['old_bank_validation_work']=old_validation_work.report() if old_validation_work is not None else None
        receipt['definition_admission_work']=({key:value-definition_work_before[key]
            for key,value in definition_curriculum.work_report().items()} if definition_work_before is not None else None)
        receipt['basis_admission_work']=({key:value-basis_work_before[key]
            for key,value in basis_curriculum.work_report().items()} if basis_work_before is not None else None)
        receipt.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,ended_utc=utc())
        publish(target/'summary.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('freeze','run'));p.add_argument('directory')
    p.add_argument('--campaign-directory');p.add_argument('--campaign-summary-sha256')
    p.add_argument('--data-directory');p.add_argument('--data-manifest-sha256');p.add_argument('--launch-sha256')
    p.add_argument('--max-seconds',type=int,default=900);args=p.parse_args()
    if args.mode=='freeze':
        print(json.dumps(dict(launch_sha256=freeze(args.directory,campaign_directory=args.campaign_directory,
            campaign_summary_sha256=args.campaign_summary_sha256,data_directory=args.data_directory,
            data_manifest_sha256=args.data_manifest_sha256,max_seconds=args.max_seconds))))
    else:
        result=run(args.directory,launch_sha256=args.launch_sha256)
        print(json.dumps({k:result[k] for k in ('status','updates','wall_seconds')}))
