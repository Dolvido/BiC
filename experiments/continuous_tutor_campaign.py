"""Local, resumable owner of verified tutor/practice/withdrawal comparisons.

Only completed stages advance the durable state. The GPU child never contacts a
teacher; accepted tutor choices are reused after a budget boundary. This owner
is an engineered research controller, not a learned policy or model release.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.foundation_layout_study import (
    ROOT, native, digest, read, publish, encoded, relative_root, verify_pins, utc)

SCHEMA = 'bic-continuous-tutor-campaign-v1'
STAGES = ('ready', 'prepared', 'authored', 'compiled', 'trained')
STAGE_SECONDS = dict(ready=120, prepared=60, authored=600, compiled=600, trained=30)
FINAL_RESERVE = 60
INVENTORY = 'runs/verified-tutor-inventory-local/attempt-001'
INVENTORY_SHA = '6777954dd97141a09fd2a2fa5e664dbdbbf092efe3d3e587fbf91b109be6b088'
PROTOCOL = 'docs/CONTINUOUS_TUTOR_CAMPAIGN.md'


def source_hashes():
    from experiments import verified_tutor_cycle_data as data
    from experiments import verified_tutor_author_v2 as author
    from experiments import continuous_tutor_worker as worker
    result = {**data.source_hashes(), **author.source_hashes(), **worker.source_hashes()}
    for path in ('experiments/continuous_tutor_campaign.py',
                 'experiments/continuous_tutor_policy.py', PROTOCOL):
        result[path] = digest(ROOT/path)
    return result


def _checked(path, sha):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or digest(path) != sha:
        raise ValueError('authenticated repository artifact required')
    return read(path)


def _record(path):
    return dict(path=relative_root(Path(path).resolve()), sha256=digest(path))


def _pin_record(record):
    return _checked(ROOT/record['path'], record['sha256'])


def _hash(value):
    from experiments.verified_tutor_curriculum import contract_sha256
    return contract_sha256(value)


def _metadata(checkpoint, origin):
    import torch
    raw = native(ROOT/checkpoint['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != checkpoint['sha256']:
        raise ValueError('checkpoint bytes differ')
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if (value['schema'] != 'bic-shared-state-continuation-v1'
            or value['identity']['origin'] != origin
            or value['learner']['cursor'] != value['lifetime_updates']):
        raise ValueError('original full-state continuation required')
    return dict(checkpoint=deepcopy(checkpoint), weights_sha256=value['weights_sha256'],
                lifetime_updates=value['lifetime_updates'])


def _initial_protection(output, evaluation_directory, evaluation_manifest):
    """One initial census; later cycles add compact fingerprint lists only."""
    import torch
    from experiments import shared_acquisition_transfer_data as transfer
    from experiments.verified_tutor_curriculum import transcript_sha256
    started, cpu = time.monotonic(), time.process_time()
    pins, loads, bytes_loaded = {}, 0, 0
    def archive(path, pin):
        nonlocal loads, bytes_loaded
        path = Path(path)
        raw = native(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError('protection archive changed')
        pins[relative_root(path)] = pin
        value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        loads += 1; bytes_loaded += len(raw)
        return value
    allowed_training = set(archive(ROOT/transfer.TRAINED, transfer.TRAINED_SHA))
    protected = set()
    bank_record = evaluation_manifest['banks']
    banks = archive(Path(evaluation_directory)/bank_record['path'], bank_record['sha256'])
    protected.update(transcript_sha256(row) for bank in banks.values() for row in bank['rows'])
    prior_dir = ROOT/transfer.COMPILED
    manifest = _checked(prior_dir/'manifest.json', transfer.COMPILED_SHA)
    pins[relative_root(prior_dir/'manifest.json')] = transfer.COMPILED_SHA
    seen = set()
    for arms in manifest['phases'].values():
        for records in arms.values():
            for record in records:
                if record['path'] in seen:
                    continue
                seen.add(record['path'])
                image = archive(prior_dir/record['path'], record['sha256'])
                protected.update(transcript_sha256(row)
                    for rows in image['bundle']['families'].values() for row in rows)
    removed_replay_overlap=len(protected & allowed_training)
    protected.difference_update(allowed_training)
    original_pin=publish(output/'original-training-transcripts.json',sorted(allowed_training))
    fresh_only=dict(path=relative_root(output/'original-training-transcripts.json'),sha256=original_pin)
    pin = publish(output/'initial-protection.json', sorted(protected))
    cost = dict(archive_loads=loads, bytes_deserialized=bytes_loaded,
        fingerprint_count=len(protected), original_training_count=len(allowed_training),
        allowed_replay_overlap_removed=removed_replay_overlap, wall_seconds=time.monotonic()-started,
        cpu_seconds=time.process_time()-cpu, models=0, teacher_calls=0,
        canonical_generation_calls=0)
    publish(output/'protection-cost.json', cost)
    return dict(path=relative_root(output/'initial-protection.json'), sha256=pin), fresh_only, pins, cost


def validate_state(state):
    if (state.get('schema') != SCHEMA or state.get('stage') not in STAGES
            or type(state.get('cycle')) is not int or state['cycle'] < 1
            or type(state.get('completed_cycles')) is not int
            or state['completed_cycles'] != state['cycle']-1):
        raise ValueError('exact completed-stage campaign state required')
    required = {'prepared':('cycle_directory','cycle_context','cycle_spec'),
        'authored':('cycle_directory','cycle_context','cycle_spec','teacher_decision'),
        'compiled':('cycle_directory','cycle_context','cycle_spec','teacher_decision','compiled'),
        'trained':('cycle_directory','cycle_context','cycle_spec','teacher_decision','compiled','worker')}
    if any(k not in state for k in required.get(state['stage'], ())):
        raise ValueError('pending stage has no durable predecessor')
    parent=state.get('parent', {})
    if (type(parent.get('lifetime_updates')) is not int or parent['lifetime_updates'] < 1
            or set(parent.get('metrics', {})) != {'dev','train_fit','retention','transfer_original','transfer_varied'}
            or set(state.get('retention_reference', {})) != set(parent.get('metrics', {}))):
        raise ValueError('complete native parent evidence required')
    return deepcopy(state)


def freeze(output, *, parent_directory, parent_summary_sha256,
           evaluation_directory, evaluation_manifest_sha256, teacher,
           seed=853002001, budget_seconds=3600, resume=False):
    """Freeze an invocation after a completed parent; CPU metadata work only."""
    if type(budget_seconds) is not int or not 120 <= budget_seconds <= 86400:
        raise ValueError('explicit positive local compute allowance required')
    from experiments import shared_acquisition_transfer_data as transfer
    from experiments import verified_tutor_author_v2 as author
    from experiments import continuous_tutor_policy as policy
    if (type(seed) is not int or not 0 <= seed < 2**63
            or set(teacher) != {'model','sha256'}):
        raise ValueError('fixed seed and installed local teacher identity required')
    # Validate the transport's local-only model name without making a request.
    from brain_in_computer.curriculum_tutor import LocalTutor
    LocalTutor(model=teacher['model'], timeout=STAGE_SECONDS['prepared'], cache_size=0)
    from experiments.verified_tutor_curriculum import _pin
    if not _pin(teacher['sha256']): raise ValueError('installed teacher digest required')
    started,cpu=time.monotonic(),time.process_time()
    output,prior=Path(output).resolve(),Path(parent_directory).resolve()
    summary=_checked(prior/'execution/summary.json',parent_summary_sha256)
    previous=read(prior/'launch.json'); prior_launch=digest(prior/'launch.json')
    if (summary.get('status') != 'completed' or summary.get('partial_work_unknown')
            or summary.get('launch_sha256') != prior_launch):
        raise ValueError('completed known-work parent required; unknown work is not retried')
    verify_pins(previous['source_sha256'])
    evaluation_directory=Path(evaluation_directory).resolve()
    evaluation=transfer.load_manifest(evaluation_directory,
        expected_manifest_sha256=evaluation_manifest_sha256)
    pins={relative_root(prior/'launch.json'):prior_launch,
          relative_root(prior/'execution/summary.json'):parent_summary_sha256,
          relative_root(evaluation_directory/'manifest.json'):evaluation_manifest_sha256,
          INVENTORY+'/manifest.json':INVENTORY_SHA}
    native(output).mkdir(parents=True,exist_ok=False)
    if resume:
        if (previous.get('schema')!=SCHEMA or previous['seed']!=seed
                or previous['teacher']!=teacher
                or previous['evaluation']['manifest_sha256']!=evaluation_manifest_sha256
                or previous['source_sha256']!=source_hashes()):
            raise ValueError('resume needs unchanged campaign, teacher, banks and sources')
        saved=summary['last_state'];state=validate_state(_pin_record(saved))
        pins[saved['path']]=saved['sha256']
        origin=previous['origin_identity'];runtime=previous['parent_runtime']
        metadata_loads=0;protection_cost=None
    else:
        if (previous.get('schema')!='bic-sustained-shared-acquisition-v1'
                or previous['data_manifest_sha256']!=evaluation_manifest_sha256):
            raise ValueError('initial campaign requires completed sustained acquisition')
        saved=summary['last_commit'];commit=_pin_record(saved)
        if (commit['launch_sha256']!=prior_launch or commit['curriculum']!=summary['curriculum']
                or commit['scores']!=summary['evaluations'][str(summary['curriculum']['cursor'])]):
            raise ValueError('final parent learner/controller commit differs')
        pins[saved['path']]=saved['sha256']
        origin=previous['origin_identity'];runtime=previous['expected_runtime']
        parent=_metadata(commit['checkpoint'],origin);metadata_loads=1
        parent['metrics']={name:value['metrics'] for name,value in commit['scores'].items()}
        protection,fresh_only,extra_pins,protection_cost=_initial_protection(output,evaluation_directory,evaluation)
        pins.update(extra_pins);pins[protection['path']]=protection['sha256']
        pins[fresh_only['path']]=fresh_only['sha256']
        state=dict(schema=SCHEMA,stage='ready',cycle=1,completed_cycles=0,
            parent=parent,reference=deepcopy(parent),retention_reference=deepcopy(parent['metrics']),
            extra_protected=[protection],fresh_only_protected=[fresh_only],
            decisions=[],physical_totals=dict(worker_updates=0,teacher_chat_attempts=0,
                teacher_chat_completions=0),original_parent_checkpoint=deepcopy(parent['checkpoint']))
    validate_state(state)
    pins[state['parent']['checkpoint']['path']]=state['parent']['checkpoint']['sha256']
    verify_pins(pins)
    sources=source_hashes()
    launch=dict(schema=SCHEMA,created_utc=utc(),seed=seed,budget_seconds=budget_seconds,
        final_reserve_seconds=FINAL_RESERVE,stage_seconds=STAGE_SECONDS,teacher=deepcopy(teacher),
        origin_identity=origin,parent_runtime=runtime,initial_state=state,resumed=resume,
        evaluation=dict(directory=relative_root(evaluation_directory),manifest_sha256=evaluation_manifest_sha256),
        inventory=dict(directory=INVENTORY,manifest_sha256=INVENTORY_SHA),
        source_sha256=sources,input_sha256=pins,
        freeze_work=dict(checkpoint_metadata_loads=metadata_loads,models=0,teacher_calls=0,
            protection=protection_cost),freeze_cost=dict(wall_seconds=time.monotonic()-started,
            cpu_seconds=time.process_time()-cpu),automatic_promotion=False,
        scope='Engineered local research loop; repeated development diagnostics, not broad English or learned self-direction.')
    return publish(output/'launch.json',launch)


def request_for(launch,state,context,contract):
    from experiments import verified_tutor_author_v2 as author
    from experiments import foundation_tutor_adviser as adviser
    from experiments import verified_tutor_curriculum as compiler
    def evidence(parent):
        return dict(weights_sha256=parent['weights_sha256'],lifetime_updates=parent['lifetime_updates'],
            by_family={f:{m:{k:parent['metrics']['dev']['by_family'][f]['counts'][m][k]
                for k in ('count','total')} for m in adviser.METRICS} for f in adviser.FAMILIES})
    return dict(schema=author.REQUEST_SCHEMA,parent=context['parent'],contract=contract,
        procedural_recipe=compiler.procedural_recipe(contract),
        development=dict(schema=adviser.EVIDENCE_SCHEMA,role='development',
            current=evidence(state['parent']),reference=evidence(state['reference'])),
        mode='local',teacher=launch['teacher'],max_seconds=STAGE_SECONDS['prepared'],
        source_sha256=author.source_hashes())


def run(directory, *, launch_sha256):
    started,cpu=time.monotonic(),time.process_time()
    from experiments import continuous_tutor_policy as policy
    from experiments import verified_tutor_cycle_data as data
    from experiments import verified_tutor_author_v2 as author
    from experiments import continuous_tutor_worker as worker
    directory=Path(directory).resolve();launch=_checked(directory/'launch.json',launch_sha256)
    if launch['schema']!=SCHEMA or launch['stage_seconds']!=STAGE_SECONDS:
        raise ValueError('frozen campaign schema/stages required')
    deadline=started+launch['budget_seconds']
    out=directory/'execution';native(out).mkdir(exist_ok=False)
    state=validate_state(launch['initial_state'])
    initial_physical_totals=deepcopy(state['physical_totals'])
    receipt=dict(schema=SCHEMA,status='running',launch_sha256=launch_sha256,started_utc=utc(),
        partial_work_unknown=False,stages=[],states=[],worker_processes=[],
        completed_cycles_this_invocation=0,automatic_retry=False,automatic_promotion=False)
    process=None;active=None
    def checkpoint():
        path=out/f'states/{len(receipt["states"]):04d}.json'
        pin=publish(path,state);record=dict(path=relative_root(path),sha256=pin)
        receipt['states'].append(record);receipt['last_state']=record
    def authenticate():
        verify_pins(launch['input_sha256'])
        if source_hashes()!=launch['source_sha256']: raise ValueError('campaign source changed')
    try:
        authenticate();checkpoint();publish(out/'started.json',receipt)
        while True:
            required=STAGE_SECONDS[state['stage']]
            decision=policy.stage_decision(budget_seconds=launch['budget_seconds'],
                elapsed_seconds=time.monotonic()-started,required_seconds=required,
                final_reserve_seconds=FINAL_RESERVE)
            # Policy exposes its Boolean; the exact clock bound is also enforced here.
            if not decision['start'] or time.monotonic()+required+FINAL_RESERVE>deadline:
                receipt['stop_reason']='wall_budget_before_stage';receipt['budget_decision']=decision;break
            authenticate()
            if time.monotonic()+required+FINAL_RESERVE>deadline:
                receipt['stop_reason']='wall_budget_after_authentication';break
            stage=state['stage'];tick,proc=time.monotonic(),time.process_time()
            stage_deadline=min(deadline-FINAL_RESERVE,tick+required)
            active=dict(stage=stage,cycle=state['cycle'],started_utc=utc(),max_seconds=required)
            publish(out/f'intents/{len(receipt["stages"]):04d}.json',active)
            if stage=='ready':
                spec=policy.cycle_spec(launch['seed'],state['cycle'])
                cycle_dir=out/f'cycles/{state["cycle"]:06d}'
                parent=state['parent']
                identity=parent['checkpoint']['sha256']
                tutor_parent=dict(identity_sha256=identity,weights_sha256=parent['weights_sha256'],
                    cycle=state['cycle'],lifetime_updates=parent['lifetime_updates'])
                context=data.prepare_cycle(cycle_dir,inventory_directory=ROOT/launch['inventory']['directory'],
                    inventory_manifest_sha256=launch['inventory']['manifest_sha256'],parent=tutor_parent,
                    seed=spec['seeds']['teaching'],withdrawal_seed=spec['seeds']['withdrawal'],
                    extra_protected=state['extra_protected'],fresh_only_protected=state['fresh_only_protected'],
                    max_seconds=required,compiler_seconds=180,deadline=min(deadline-FINAL_RESERVE,tick+required))
                state.update(stage='prepared',cycle_directory=relative_root(cycle_dir),
                    cycle_context=_record(cycle_dir/'cycle.json'),cycle_spec=spec)
            elif stage=='prepared':
                cycle_dir=ROOT/state['cycle_directory'];context=_pin_record(state['cycle_context'])
                contract=_pin_record(context['teaching_contract'])
                request=request_for(launch,state,context,contract)
                request_sha=hashlib.sha256(author._json(request).encode('utf-8')).hexdigest()
                publish(cycle_dir/'author-request.json',request)
                result=author.author_curriculum(cycle_dir/'author',request=request,expected_request_sha256=request_sha,
                    deadline=min(deadline-FINAL_RESERVE,tick+required))
                selected=_record(cycle_dir/'author/result.json');selected['request_sha256']=request_sha
                state.update(stage='authored',teacher_decision=selected)
                cost=result['result']['teacher_cost']
                state['physical_totals']['teacher_chat_attempts']+=cost['chat_attempts']
                state['physical_totals']['teacher_chat_completions']+=cost['chat_completions']
            elif stage=='authored':
                cycle_dir=ROOT/state['cycle_directory']
                manifest=data.compile_data(cycle_dir,cycle_sha256=state['cycle_context']['sha256'],
                    teacher_decision=state['teacher_decision'],max_seconds=required,
                    deadline=min(deadline-FINAL_RESERVE,tick+required))
                state.update(stage='compiled',compiled=_record(cycle_dir/'compiled/manifest.json'))
            elif stage=='compiled':
                cycle_dir=ROOT/state['cycle_directory'];worker_dir=cycle_dir/'worker'
                pin=worker.freeze(worker_dir,parent_checkpoint=state['parent']['checkpoint'],
                    origin_identity=launch['origin_identity'],parent_runtime=launch['parent_runtime'],
                    data_directory=cycle_dir/'compiled',data_manifest_sha256=state['compiled']['sha256'],
                    evaluation_directory=ROOT/launch['evaluation']['directory'],
                    evaluation_manifest_sha256=launch['evaluation']['manifest_sha256'],
                    cycle_spec=state['cycle_spec'],max_seconds=required,
                    input_pins={state['cycle_context']['path']:state['cycle_context']['sha256']})
                if time.monotonic()>=stage_deadline:
                    raise TimeoutError('worker preparation consumed admitted stage allowance')
                log=native(worker_dir/'worker.log').open('xb')
                try:
                    process=subprocess.Popen([sys.executable,'-B','-m','experiments.continuous_tutor_worker',
                        'run',relative_root(worker_dir),'--launch-sha256',pin,'--deadline',str(stage_deadline)],cwd=ROOT,
                        stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    receipt['worker_processes'].append(dict(pid=process.pid,directory=relative_root(worker_dir),launch_sha256=pin))
                    while process.poll() is None:
                        if time.monotonic()>min(stage_deadline+5,deadline):
                            raise TimeoutError('native worker exceeded its stage deadline and cleanup grace')
                        time.sleep(.25)
                    code=process.returncode;receipt['worker_processes'][-1]['exit_code']=code;process=None
                finally: log.close()
                if code!=0: raise RuntimeError('native worker failed; preserve its terminal receipt, no retry')
                result=_record(worker_dir/'execution/summary.json');summary=_pin_record(result)
                if summary['status']!='completed' or summary['partial_work_unknown']:
                    raise ValueError('completed known-work native worker required')
                state.update(stage='trained',worker=result)
                state['physical_totals']['worker_updates']+=432
            else:
                summary=_pin_record(state['worker'])
                choice=policy.select_continuation(summary['branches']['procedural'],summary['branches']['tutor'],
                    expected_parent_identity=state['parent']['checkpoint']['sha256'])
                choice=policy.protect_retention(choice,
                    candidates={arm:summary['continuations'][arm]['metrics'] for arm in ('procedural','tutor')},
                    current_parent=state['parent']['metrics'],reference_parent=state['retention_reference'])
                selected=(state['parent'] if choice['selected_arm']=='parent'
                    else summary['continuations'][choice['selected_arm']])
                expected_cursor=state['parent']['lifetime_updates']+(0 if choice['selected_arm']=='parent' else 216)
                if selected['lifetime_updates']!=expected_cursor:
                    raise ValueError('selected continuation cursor differs')
                data_manifest=_pin_record(state['compiled'])
                decision_path=out/f'decisions/{state["cycle"]:06d}.json'
                choice.update(worker=state['worker'],teacher_decision=state['teacher_decision'],
                    compiled=state['compiled'],treatment=data_manifest['treatment'])
                choice_pin=publish(decision_path,choice)
                state['decisions'].append(dict(path=relative_root(decision_path),sha256=choice_pin))
                previous=state['parent'];state['parent']={k:deepcopy(selected[k]) for k in
                    ('checkpoint','weights_sha256','lifetime_updates','metrics')}
                state['reference']=previous
                extra=data_manifest['fresh_transcripts']
                state['extra_protected'].append({k:extra[k] for k in ('path','sha256')})
                state['completed_cycles']+=1;state['cycle']+=1;state['stage']='ready'
                for key in ('cycle_directory','cycle_context','cycle_spec','teacher_decision','compiled','worker'):
                    state.pop(key,None)
                receipt['completed_cycles_this_invocation']+=1
            validate_state(state);checkpoint()
            receipt['stages'].append(dict(**active,status='completed',wall_seconds=time.monotonic()-tick,
                cpu_seconds=time.process_time()-proc));active=None
            print(encoded(dict(event='stage',cycle=state['cycle'],next_stage=state['stage'],
                completed_cycles=state['completed_cycles'],lifetime_updates=state['parent']['lifetime_updates'],
                wall_seconds=time.monotonic()-started)).decode().strip(),flush=True)
        authenticate();receipt['status']='completed'
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),active_stage=active)
        # A live child must be accounted for, never presumed to have stopped.
        if process is not None and process.poll() is None:
            watchdog=dict(terminate_requested=True,kill_requested=False,still_running=True)
            try:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    watchdog['kill_requested']=True;process.kill();process.wait(timeout=5)
            except (OSError,subprocess.TimeoutExpired) as cleanup_error:
                watchdog['error']=repr(cleanup_error)
            watchdog.update(exit_code=process.poll(),still_running=process.poll() is None)
            receipt['worker_processes'][-1]['watchdog']=watchdog
            receipt['worker_processes'][-1]['exit_code']=process.returncode
        if active and active['stage'] in ('prepared','compiled'):
            receipt['partial_work_unknown']=True
        raise
    finally:
        receipt.update(final_stage=state['stage'],completed_cycles=state['completed_cycles'],
            parent=state['parent'],physical_totals=state['physical_totals'],
            new_committed_physical_work={k:v-initial_physical_totals[k] for k,v in state['physical_totals'].items()},
            physical_accounting_scope='Committed stages only; any failed active-stage work remains in its child/author receipt and is not inferred to be zero.',ended_utc=utc(),
            wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
        publish(out/'summary.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('freeze','run'));p.add_argument('directory')
    p.add_argument('--launch-sha256');p.add_argument('--parent-directory');p.add_argument('--parent-summary-sha256')
    p.add_argument('--evaluation-directory');p.add_argument('--evaluation-manifest-sha256')
    p.add_argument('--teacher-json');p.add_argument('--seed',type=int,default=853002001)
    p.add_argument('--budget-seconds',type=int,default=3600);p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.mode=='freeze':
        pin=freeze(args.directory,parent_directory=args.parent_directory,parent_summary_sha256=args.parent_summary_sha256,
            evaluation_directory=args.evaluation_directory,evaluation_manifest_sha256=args.evaluation_manifest_sha256,
            teacher=read(args.teacher_json),seed=args.seed,budget_seconds=args.budget_seconds,resume=args.resume)
        print(encoded(dict(launch_sha256=pin)).decode().strip())
    else:
        value=run(args.directory,launch_sha256=args.launch_sha256)
        print(encoded(dict(status=value['status'],completed_cycles=value['completed_cycles'],
            final_stage=value['final_stage'],wall_seconds=value['wall_seconds'])).decode().strip())
