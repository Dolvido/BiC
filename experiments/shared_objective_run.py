"""Matched continuation from one parent with explicit auxiliary-objective transition."""
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

SCHEMA = 'bic-shared-objective-run-v1'
COMPLETION_SCHEMA = 'bic-shared-objective-completion-v1'
PROTOCOL = 'docs/SHARED_OBJECTIVE_TRANSITION_PROTOCOL.md'
ARMS = ('control', 'zero')
WEIGHTS = dict(control=.3, zero=0.)
ENDPOINTS = (648,1296,2592)
UPDATES, MICRO, PARENT_CURSOR = 2592, 32, 13000
CYCLE_UPDATES, REPLAY_PASSES = 648, 4
PARENT_LAUNCH_SHA256 = '4bd912311f64cf25939aa7eb437bfd09e265d23ed2b9ab7911e32ef237cbe40d'
PARENT_SUMMARY_SHA256 = 'fb46f0972dd37d5d273dfee51a017b63fc22c2e0d6856cd65ba2746421da7246'
PARENT_COMPLETION_SHA256 = 'db150cd061e6876e11495163a12fe7d2f2655aae000207faf8859a3fb45f1335'
PARENT_ANALYSIS = 'runs/sustained-composition-analysis-local/attempt-001/independent-analysis.json'
PARENT_ANALYSIS_SHA256 = '7ac87b22735316769293576f9e90996f773a1b2effbf4a6b5169c7f76b1c4acf'
DATA_MANIFEST_SHA256 = 'f5a71eafc49e0b99f9fa893b0c1f9385df26b618e12c79a99b9dd2f7747de896'
PARENT_CHECKPOINT_SHA256 = '4780cd241a17516003e2e162b45d2fb3072d9f533e68586edb4e562de4b412fa'
BASE_BANK_NAMES = ('basis_binding','basis_revision','basis_composition','basis_sequence',
    'definition_binding','definition_revision','definition_composition',
    'dev','train_fit','retention','transfer_original','transfer_varied')
EXTRA_BANK_NAMES = ('complementary_dev_binding','complementary_dev_revision',
    'complementary_fit_binding','complementary_fit_revision')
BANK_NAMES = BASE_BANK_NAMES+EXTRA_BANK_NAMES


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def contract(max_seconds=3600):
    if type(max_seconds) is not int or max_seconds != 3600:
        raise ValueError('one protocol-fixed3600-second total allowance required')
    return dict(schema=SCHEMA,arms=list(ARMS),updates_per_arm=UPDATES,micro_batch_size=MICRO,
        replay_passes=REPLAY_PASSES,source_updates_per_pass=CYCLE_UPDATES,
        endpoints=[0,*ENDPOINTS],max_seconds=3600,config=CONFIG,teacher_calls=0,
        automatic_retry=False,automatic_promotion=False,parent_is_unadopted_research_candidate=True,
        auxiliary_weights=deepcopy(WEIGHTS),source_schedule='curriculum')


def physical_plan(manifest):
    from experiments import complementary_composition_run as previous
    previous.physical_plan(manifest)
    return dict(physical_arms=list(ARMS),updates=5184,episode_exposures=497664,
        family_forwards=15552,backwards=15552,restores=8,snapshots=6,
        new_state_readouts=dict(control=7776,zero=0),new_state_objectives=dict(control=7776,zero=0))


def replay_record(records,index):
    """Only the consumer cursor advances; admitted source records stay exact."""
    if len(records)!=CYCLE_UPDATES or type(index) is not int or not 0<=index<UPDATES:
        raise ValueError('one exact648-record schedule repeated four times required')
    return records[index%CYCLE_UPDATES]


def parent_admission(summary, previous, commit, completion):
    if (summary.get('schema')!='bic-sustained-composition-run-v1' or summary.get('status')!='completed'
            or summary.get('partial_work_unknown') is not False
            or summary.get('new_updates')!=dict(control=2592,curriculum=2592)
            or summary.get('launch_sha256')!=PARENT_LAUNCH_SHA256
            or previous.get('shared_parent',{}).get('lifetime_updates')!=9760
            or previous.get('reference_parent',{}).get('lifetime_updates')!=9112
            or previous.get('parents',{}).get('curriculum',{}).get('lifetime_updates')!=10408
            or previous.get('data_manifest_sha256')!=DATA_MANIFEST_SHA256):
        raise ValueError('completed sustained curriculum and historical references required')
    if (completion.get('schema')!='bic-sustained-composition-completion-v1'
            or completion.get('status')!='completed' or completion.get('summary_status')!='completed'
            or completion.get('summary',{}).get('sha256')!=PARENT_SUMMARY_SHA256
            or completion.get('launch_sha256')!=PARENT_LAUNCH_SHA256
            or completion.get('completed_within_deadline') is not True
            or completion.get('continuation_eligible') is not True
            or type(completion.get('wall_seconds')) not in (int,float)
            or not 0<=completion['wall_seconds']<3600 or completion.get('max_seconds')!=3600):
        raise ValueError('inclusive completed parent terminal marker required')
    continuation=summary.get('continuations',{}).get('curriculum',{})
    if (commit.get('schema')!='bic-sustained-composition-run-v1' or commit.get('arm')!='curriculum'
            or commit.get('relative_step')!=2592 or commit.get('lifetime_updates')!=PARENT_CURSOR
            or commit.get('launch_sha256')!=PARENT_LAUNCH_SHA256
            or commit.get('checkpoint',{}).get('sha256')!=PARENT_CHECKPOINT_SHA256
            or continuation.get('checkpoint')!=commit['checkpoint']
            or continuation.get('weights_sha256')!=commit.get('weights_sha256')
            or continuation.get('lifetime_updates')!=PARENT_CURSOR
            or commit.get('parent_checkpoint')!=previous['parents']['curriculum']['checkpoint']
            or commit.get('data_manifest_sha256')!=DATA_MANIFEST_SHA256
            or commit.get('scores')!=summary.get('evaluations',{}).get('curriculum',{}).get('2592')
            or set(commit.get('scores',{}))!=set(BANK_NAMES)
            or any(summary.get('physical_training',{}).get(a,{}).get('work',{}).get('unknown_optimizer_outcomes')!=0
                   for a in ('control','curriculum'))):
        raise ValueError('one exact complete curriculum13000 checkpoint and all16 scores required')
    parent=dict(checkpoint=deepcopy(commit['checkpoint']),identity_sha256=identity(previous['origin_identity']),
        lifetime_updates=PARENT_CURSOR,weights_sha256=commit['weights_sha256'],
        metrics={name:deepcopy(value['metrics']) for name,value in commit['scores'].items()})
    return {arm:deepcopy(parent) for arm in ARMS}


def expected_recipe(recipe,arm):
    if arm not in ARMS or recipe.get('auxiliary_weight')!=.3:raise ValueError('declared .3 parent and arm required')
    result=deepcopy(recipe);result['auxiliary_weight']=WEIGHTS[arm];return result


def validate_auxiliary(pin,parent_pin,arm):
    if arm not in ARMS or any(type(v) is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in (pin,parent_pin)):
        raise ValueError('explicit auxiliary fingerprints and arm required')
    if arm=='zero' and pin!=parent_pin:raise ValueError('inactive auxiliary weights or AdamW state changed')


def auxiliary_fingerprint(learner):
    # Read-only host inspection, not an extra model snapshot or optimizer step.
    # Stable ordered tensor bytes bind weights plus all three AdamW fields.
    h=hashlib.sha256();count=byte_count=0
    for name,parameter in learner.model.named_parameters():
        if not name.startswith(('state_alias_embedding.','state_decoder.')):continue
        values={'weight':parameter,**learner.optimizer.state[parameter]}
        if set(values)!={'weight','step','exp_avg','exp_avg_sq'}:raise ValueError('complete auxiliary AdamW state required')
        for key,value in sorted(values.items()):
            tensor=value.detach().cpu().contiguous()
            h.update(identity(dict(name=name,field=key,dtype=str(tensor.dtype),shape=list(tensor.shape))).encode())
            raw=tensor.numpy().tobytes();h.update(raw);count+=1;byte_count+=len(raw)
    if not count:raise ValueError('auxiliary parameters absent')
    return dict(sha256=h.hexdigest(),tensors=count,bytes=byte_count)


def validate_bindings(manifest, shared_parent):
    from experiments import complementary_composition_run as previous
    previous.validate_bindings(manifest,shared_parent)
    return physical_plan(manifest)


def accounting_delta(accounting, parent):
    result={group:{key:value-parent[group][key] for key,value in accounting[group].items()}
        for group in ('work','state_work','cost')}
    if any(value<0 for group in ('work','state_work') for value in result[group].values()):
        raise ValueError('lifetime work cannot regress on restart')
    return result


def physical_deltas(latest, parent_metadata):
    """Charge each physical branch against its own exact inherited accounting."""
    return {arm:accounting_delta(value['accounting']['lifetime_kernel'],parent_metadata[arm]['accounting'])
        for arm,value in latest.items()}


def validate_completed_work(plan, receipt, physical):
    expected={a:UPDATES for a in ARMS}
    if (receipt['new_updates']!=expected or set(physical)!=set(plan['physical_arms'])
            or len(receipt['restorations'])!=plan['restores'] or len(receipt['snapshots'])!=plan['snapshots']):
        raise ValueError('measured paired physical workload differs')
    for arm,value in physical.items():
        work=value['work']
        expected_work=dict(retained_updates=2592,synchronized_optimizer_updates=2592,optimizer_attempts=2592,
            optimizer_returns=2592,unknown_optimizer_outcomes=0,retained_episodes=248832,
            attempted_forwards=7776,completed_forwards=7776,attempted_backwards=7776,completed_backwards=7776,
            attempted_forward_episodes=248832,completed_forward_episodes=248832,
            attempted_backward_episodes=248832,completed_backward_episodes=248832)
        if any(work.get(k)!=v for k,v in expected_work.items()):raise ValueError('actual complete learner accounting differs')
        if set(value['state_work'])!={'attempted_state_readouts','completed_state_readouts','attempted_state_objectives','completed_state_objectives'} or any(v!=(7776 if WEIGHTS[arm] else 0) for v in value['state_work'].values()):
            raise ValueError('actual declared auxiliary learning work required')


def source_hashes():
    from experiments import sustained_composition_run as previous, shared_objective_continuation as bridge
    return {**previous.source_hashes(),**bridge.source_hashes(),
        'experiments/shared_objective_run.py':digest(Path(__file__)),PROTOCOL:digest(ROOT/PROTOCOL)}


def freeze(output, *, parent_directory, parent_summary_sha256, data_directory,
           data_manifest_sha256, validation_records, max_seconds=3600):
    started,cpu=time.monotonic(),time.process_time();fixed=contract(max_seconds)
    if parent_summary_sha256!=PARENT_SUMMARY_SHA256 or data_manifest_sha256!=DATA_MANIFEST_SHA256:
        raise ValueError('exact completed comparison and unchanged data pins required')
    output,parent_directory,data_directory=map(lambda p:Path(p).resolve(),(output,parent_directory,data_directory))
    if any(not p.is_relative_to(ROOT) for p in (output,parent_directory,data_directory)):raise ValueError('repository paths required')
    native(output).mkdir(parents=True,exist_ok=False)
    receipt=dict(schema=SCHEMA,status='running',checkpoint_load_attempts=0,checkpoint_loads=0,models=0,updates=0,teacher_calls=0)
    publish(output/'freeze-started.json',receipt)
    try:
        pins={};devices=set()
        for record in validation_records:
            path=ROOT/record['path']
            if not path.resolve().is_relative_to(ROOT) or digest(path)!=record['sha256']:raise ValueError('pinned numerical validation required')
            proof=read(path)
            if (proof.get('schema')!='bic-shared-objective-runtime-validation-v1' or proof.get('status')!='passed'
                    or proof.get('source_sha256',{}).get('experiments/shared_objective_continuation.py')
                    !=digest(ROOT/'experiments/shared_objective_continuation.py')):raise ValueError('successful current bridge numerical proof required')
            devices.add(proof.get('device'));pins[record['path']]=record['sha256']
        if devices!={'cpu','cuda:0'}:raise ValueError('both exact CPU and CUDA continuation proofs required')
        summary_path=parent_directory/'execution/summary.json';previous_path=parent_directory/'launch.json'
        completion_path=parent_directory/'execution/completion.json'
        for path,pin in ((summary_path,parent_summary_sha256),(previous_path,PARENT_LAUNCH_SHA256),
                         (completion_path,PARENT_COMPLETION_SHA256),(ROOT/PARENT_ANALYSIS,PARENT_ANALYSIS_SHA256)):
            if digest(path)!=pin:raise ValueError('pinned completed parent evidence changed')
            pins[relative_root(path)]=pin
        summary,previous,completion=read(summary_path),read(previous_path),read(completion_path)
        record=summary['endpoint_commits']['curriculum']['2592']
        if digest(ROOT/record['path'])!=record['sha256']:raise ValueError('curriculum endpoint changed')
        pins[record['path']]=record['sha256'];commit=read(ROOT/record['path'])
        parents=parent_admission(summary,previous,commit,completion)
        verify_pins(previous['source_sha256']);verify_pins(previous['input_sha256'])
        from experiments import complementary_composition_data as data, continuous_tutor_worker as worker
        manifest=data.load_manifest(data_directory,data_manifest_sha256)
        plan=validate_bindings(manifest,previous['shared_parent'])
        import torch
        parent=parents['control'];raw=native(ROOT/parent['checkpoint']['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=parent['checkpoint']['sha256']:raise ValueError('common full archive changed')
        receipt['checkpoint_load_attempts']+=1
        payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);receipt['checkpoint_loads']+=1
        actual=worker.parent_metadata(payload);bridge_cost=deepcopy(payload['bridge_cost']);del payload,raw
        if (actual['cursor']!=PARENT_CURSOR or actual['weights_sha256']!=parent['weights_sha256']
                or actual['origin']!=previous['origin_identity'] or actual['recipe']['config']!=CONFIG
                or actual['evidence']!=commit['evidence'] or actual['accounting']!=commit['accounting']['lifetime_kernel']):
            raise ValueError('common optimizer/evidence/origin metadata differs')
        metadata={a:deepcopy(actual) for a in ARMS};pins[parent['checkpoint']['path']]=parent['checkpoint']['sha256']
        pins.update({relative_root(data_directory/'manifest.json'):data_manifest_sha256,
            relative_root(data_directory/'preparation.json'):digest(data_directory/'preparation.json')})
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
            validation_records=deepcopy(validation_records),parents=parents,parent_metadata=metadata,
            parent_bridge_cost={a:deepcopy(bridge_cost) for a in ARMS},origin_identity=previous['origin_identity'],
            previous_curriculum_parent=deepcopy(previous['parents']['curriculum']),
            shared_parent=deepcopy(previous['shared_parent']),reference_parent=deepcopy(previous['reference_parent']),
            shared_baseline_metrics=deepcopy(previous['shared_baseline_metrics']),
            expected_runtime=previous['expected_runtime'],parent_directory=relative_root(parent_directory),
            parent_summary_sha256=parent_summary_sha256,parent_completion_sha256=PARENT_COMPLETION_SHA256,
            parent_analysis=dict(path=PARENT_ANALYSIS,sha256=PARENT_ANALYSIS_SHA256),
            basis_data_directory=previous['basis_data_directory'],basis_data_manifest_sha256=previous['basis_data_manifest_sha256'],
            evaluation=previous['evaluation'],data_directory=relative_root(data_directory),
            data_manifest_sha256=data_manifest_sha256,physical_plan=plan,
            scope='Identical curriculum four times from one13000 full-state parent; explicit .3 versus zero objective; no new lessons, tutor calls or automatic selection.')
        validate_launch(launch)
        pin=publish(output/'launch.json',launch);receipt.update(status='completed',launch_sha256=pin);return pin
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu);publish(output/'freeze-receipt.json',receipt)


def validate_launch(launch):
    fixed=contract(launch.get('max_seconds'))
    if any(launch.get(k)!=v for k,v in fixed.items()):raise ValueError('unchanged objective comparison contract required')
    if (set(launch.get('parents',{}))!=set(ARMS) or set(launch.get('parent_metadata',{}))!=set(ARMS)
            or launch.get('parent_summary_sha256')!=PARENT_SUMMARY_SHA256
            or launch.get('parent_completion_sha256')!=PARENT_COMPLETION_SHA256
            or launch.get('data_manifest_sha256')!=DATA_MANIFEST_SHA256
            or launch.get('shared_parent',{}).get('lifetime_updates')!=9760
            or launch.get('previous_curriculum_parent',{}).get('lifetime_updates')!=10408
            or launch.get('reference_parent',{}).get('lifetime_updates')!=9112):
        raise ValueError('exact common parent and all historical references required')
    if launch['parents']['control']!=launch['parents']['zero'] or launch['parent_metadata']['control']!=launch['parent_metadata']['zero']:
        raise ValueError('both objectives require identical complete incoming parent')
    for arm in ARMS:
        parent=launch['parents'][arm]
        if (parent['lifetime_updates']!=PARENT_CURSOR or parent['checkpoint']['sha256']!=PARENT_CHECKPOINT_SHA256
                or set(parent['metrics'])!=set(BANK_NAMES) or launch['parent_metadata'][arm]['cursor']!=PARENT_CURSOR
                or launch['parent_metadata'][arm]['recipe']['auxiliary_weight']!=.3):
            raise ValueError('common full-state13000 parent required')


def publish_terminal(receipt, target, *, started, cpu_started, deadline,
                     clock=None, cpu_clock=None, publisher=None):
    """The mandatory marker decides success after durable summary publication.

    Marker publication is administrative bookkeeping, separately timed in the
    returned value. A missing/failed marker never admits a completed study.
    """
    clock = time.monotonic if clock is None else clock
    cpu_clock = time.process_time if cpu_clock is None else cpu_clock
    publisher = publish if publisher is None else publisher
    after_cleanup = clock()
    if receipt['status']=='completed' and after_cleanup>=deadline:
        receipt.update(status='failed',deadline_failure='cleanup_exceeded_worker_deadline')
    if receipt['status']!='completed':receipt['continuations']={}
    receipt.update(wall_seconds=after_cleanup-started,cpu_seconds=cpu_clock()-cpu_started,ended_utc=utc())
    summary_path=target/'summary.json'
    summary_pin=publisher(summary_path,receipt)
    after_summary,after_summary_cpu=clock(),cpu_clock()
    within_deadline=after_summary<deadline
    completed=receipt['status']=='completed' and within_deadline
    completion=dict(schema=COMPLETION_SCHEMA,status='completed' if completed else 'failed',
        launch_sha256=receipt['launch_sha256'],summary=dict(path=relative_root(summary_path),sha256=summary_pin),
        summary_status=receipt['status'],wall_seconds=after_summary-started,
        cpu_seconds=after_summary_cpu-cpu_started,max_seconds=deadline-started,
        summary_publication_wall_seconds=after_summary-after_cleanup,
        completed_within_deadline=within_deadline,continuation_eligible=completed,
        reason='completed_within_worker_deadline' if completed else
            'summary_publication_exceeded_worker_deadline' if receipt['status']=='completed' else 'worker_failed',
        timing_scope='Includes all worker work, cleanup and durable summary publication; excludes this completion marker publication.')
    completion_path=target/'completion.json'
    completion_pin=publisher(completion_path,completion)
    return dict(status=completion['status'],completion=dict(path=relative_root(completion_path),sha256=completion_pin),
        wall_seconds=completion['wall_seconds'],cpu_seconds=completion['cpu_seconds'],
        completion_publication_wall_seconds=clock()-after_summary,
        completion_publication_cpu_seconds=cpu_clock()-after_summary_cpu)


def run(output, *, launch_sha256):
    started,cpu=time.monotonic(),time.process_time();output=Path(output).resolve()
    if digest(output/'launch.json')!=launch_sha256:raise ValueError('pinned paired launch required')
    launch=read(output/'launch.json');validate_launch(launch)
    deadline=started+3600;target=output/'execution';native(target).mkdir(exist_ok=False)
    receipt=dict(schema=SCHEMA,status='running',launch_sha256=launch_sha256,started_utc=utc(),
        physical_plan=deepcopy(launch['physical_plan']),
        new_updates={a:0 for a in ARMS},teacher_calls=0,archive_load_attempts=0,archive_loads=0,archive_bytes=0,
        definition_json_load_attempts=0,definition_json_loads=0,restorations=[],snapshot_attempts=0,snapshots=[],
        evaluations={},endpoint_commits={},continuations={},steps=[],artifacts={},evaluation_reports=[],
        transitions={},auxiliary_inspections=[],closed_preparations=[],partial_work_unknown=False,publications=dict(attempts=0,completions=0),
        replay_target_work=dict(batch_attempts=0,batch_completions=0,row_completions=0,
            completed_english_checks=0,completed_typed_checks=0,partial_batch_internal_work_unknown=False))
    learner=owner=arm=active_bank=active_ledger=active_evaluation=None
    latest={};evaluation_accounted=False;definition_before=basis_before=complementary_before=old_validation_work=None

    def boundary():
        if time.monotonic()>=deadline:raise TimeoutError('shared3600-second sustained allowance ended')

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
            if step==0:
                restored=SharedObjectiveContinuation.from_continuation(raw,expected_sha256=record['sha256'],
                    expected_identity=launch['origin_identity'],auxiliary_weight=WEIGHTS[arm],device='cuda:0')
                receipt['transitions'][arm]=dict(identity=restored.transition,sha256=restored.transition_sha256)
            else:
                restored=SharedObjectiveContinuation.from_snapshot(raw,expected_sha256=record['sha256'],
                    expected_transition_sha256=receipt['transitions'][arm]['sha256'],device='cuda:0')
        except BaseException as error:
            receipt['restorations'].append(dict(arm=arm,step=step,source_checkpoint=deepcopy(record),report=getattr(error,'objective_report',dict(failed=True))));raise
        receipt['restorations'].append(dict(arm=arm,step=step,source_checkpoint=deepcopy(record),report=restored.last_restore_report));return restored

    def inspect_auxiliary(step,parent_pin=None):
        began,processor=time.monotonic(),time.process_time();report=dict(arm=arm,step=step,failed=True)
        receipt['auxiliary_inspections'].append(report)
        try:
            report.update(auxiliary_fingerprint(learner))
            if parent_pin is not None:validate_auxiliary(report['sha256'],parent_pin,arm)
            report['failed']=False;return report['sha256']
        finally:
            report.update(wall_seconds=time.monotonic()-began,cpu_seconds=time.process_time()-processor)

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
        from experiments.shared_objective_continuation import SharedObjectiveContinuation
        from experiments.sequence_student import SequenceConfig
        from brain_in_computer.dialogue_student import checkpoint_digest
        torch.set_num_threads(1);torch.set_num_interop_threads(1);execution_profile.configure_strict_profile()
        receipt['runtime']=execution_profile.runtime_profile()
        if receipt['runtime']!=launch['expected_runtime']:raise ValueError('original strict runtime required')
        config=SequenceConfig(**CONFIG);data_dir=ROOT/launch['data_directory']
        definition_before=definition_curriculum.work_report();basis_before=basis_curriculum.work_report()
        complementary_before=complementary_curriculum.work_report();old_validation_work=WorkLedger()
        manifest=data.load_manifest(data_dir,launch['data_manifest_sha256'])
        if validate_bindings(manifest,launch['shared_parent'])!=launch['physical_plan']:
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
            learner=restore(launch['parents'][arm]['checkpoint'],0)
            if (learner.cursor!=PARENT_CURSOR or checkpoint_digest(learner.model)!=launch['parents'][arm]['weights_sha256']
                    or learner.accounting['lifetime_kernel']!=launch['parent_metadata'][arm]['accounting']
                    or learner.evidence!=launch['parent_metadata'][arm]['evidence']
                    or identity(learner.recipe)!=identity(expected_recipe(launch['parent_metadata'][arm]['recipe'],arm))):
                raise ValueError('restored full paired parent differs')
            auxiliary_parent_pin=inspect_auxiliary(0)
            if arm=='zero' and auxiliary_parent_pin!=receipt['auxiliary_inspections'][0]['sha256']:
                raise ValueError('initial auxiliary full state differs across arms')
            owner=PreparedLayoutOwner(config=config,layout='original',micro_batch_size=MICRO)
            evaluate(0)
            if {name:value['metrics'] for name,value in receipt['evaluations'][arm]['0'].items()}!=launch['parents'][arm]['metrics']:
                raise ValueError('all16 baseline native scores must reproduce this own branch parent')
            records=manifest['phases']['training']['curriculum']
            for index in range(UPDATES):
                record=replay_record(records,index)
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
                    source_record=record,replay_pass=index//CYCLE_UPDATES+1,source_position=index%CYCLE_UPDATES,
                    consumed_cursor=bundle['bundle_id'],report=report)))
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
                    inspect_auxiliary(index+1,auxiliary_parent_pin)
                    owner=PreparedLayoutOwner(config=config,layout='original',micro_batch_size=MICRO)
                    evaluate(index+1);authenticate()
                    commit=dict(schema=SCHEMA,launch_sha256=launch_sha256,arm=arm,relative_step=index+1,lifetime_updates=learner.cursor,
                        parent_checkpoint=launch['parents'][arm]['checkpoint'],data_manifest_sha256=launch['data_manifest_sha256'],
                        checkpoint=checkpoint,weights_sha256=saved_weights,transition=deepcopy(receipt['transitions'][arm]),
                        auxiliary_inspection=deepcopy(receipt['auxiliary_inspections'][-1]),evidence=learner.evidence,accounting=learner.accounting,
                        scores=deepcopy(receipt['evaluations'][arm][str(index+1)]))
                    receipt['endpoint_commits'].setdefault(arm,{})[str(index+1)]=artifact(f'endpoint-{arm}-{index+1:04d}.json',commit)
                    if index+1==UPDATES:
                        receipt['continuations'][arm]=dict(checkpoint=checkpoint,weights_sha256=saved_weights,lifetime_updates=learner.cursor,
                            transition=deepcopy(receipt['transitions'][arm]),metrics={k:v['metrics'] for k,v in commit['scores'].items()},scores=deepcopy(commit['scores']))
            close_learner(UPDATES)
        physical=physical_deltas(latest,launch['parent_metadata'])
        validate_completed_work(launch['physical_plan'],receipt,physical)
        authenticate();receipt['status']='completed'
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if learner is not None:latest[arm]=state()
        receipt['latest_arms']=deepcopy(latest);receipt['returned_step_updates']=deepcopy(receipt['new_updates'])
        receipt['physical_training']=physical_deltas(latest,launch['parent_metadata'])
        receipt['physical_totals']={group:{key:sum(v[group][key] for v in receipt['physical_training'].values())
            for key in launch['parent_metadata']['control']['accounting'][group]} for group in ('work','state_work','cost')}
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
        terminal=publish_terminal(receipt,target,started=started,cpu_started=cpu,deadline=deadline)
    # The immutable summary may predate a publication overrun. Only the marker
    # and this returned terminal status determine successful completion.
    return {**receipt,**terminal,'continuations':receipt['continuations'] if terminal['status']=='completed' else {}}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','run'));parser.add_argument('directory')
    for name in ('parent-directory','parent-summary-sha256','data-directory','data-manifest-sha256','launch-sha256','validation-records'):parser.add_argument('--'+name)
    parser.add_argument('--max-seconds',type=int,default=3600);args=parser.parse_args()
    if args.mode=='freeze':
        print(json.dumps(dict(launch_sha256=freeze(args.directory,parent_directory=args.parent_directory,parent_summary_sha256=args.parent_summary_sha256,
            data_directory=args.data_directory,data_manifest_sha256=args.data_manifest_sha256,validation_records=json.loads(args.validation_records),max_seconds=args.max_seconds))))
    else:
        value=run(args.directory,launch_sha256=args.launch_sha256)
        print(json.dumps({k:value[k] for k in ('status','new_updates','wall_seconds','completion',
            'completion_publication_wall_seconds','completion_publication_cpu_seconds')}))
        if value['status']!='completed':raise SystemExit(1)
