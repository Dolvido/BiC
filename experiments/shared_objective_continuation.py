"""Explicit .3 -> .3/0 auxiliary-objective transition; no curriculum decision.

Metadata helpers import only stdlib. Runtime methods use the frozen native
SharedStateStudent/SharedStateKernel, retain full AdamW and never admit lessons.
Old continuation envelopes remain historical and cannot load these snapshots.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='bic-shared-objective-continuation-v1'
TRANSITION_SCHEMA='bic-shared-objective-transition-v1'
LEGACY_SCHEMA='bic-shared-state-continuation-v1'
KERNEL_SCHEMA='bic-shared-state-training-v1'
FAMILIES=('color','count','switch')
STATE_COUNTS=('attempted_state_readouts','completed_state_readouts','attempted_state_objectives','completed_state_objectives')
WORK_COUNTS=('step_invocations','packed_microbatches','packed_episodes','attempted_forwards','completed_forwards',
    'attempted_forward_episodes','completed_forward_episodes','attempted_objectives','completed_objectives',
    'attempted_backwards','completed_backwards','attempted_backward_episodes','completed_backward_episodes',
    'optimizer_attempts','optimizer_returns','synchronized_optimizer_updates','unknown_optimizer_outcomes','retained_updates','retained_episodes')
EXPOSURES=('episodes','turns','observation_tokens','observation_bytes','reply_target_tokens','reply_target_bytes')
COST_KEYS=tuple(f'{kind}_{suffix}' for kind in ('transition','restore','step','snapshot')
    for suffix in ('invocations','wall_seconds','cpu_seconds'))
AUX_PREFIXES=('state_alias_embedding.','state_decoder.')
SCOPE='Explicit auxiliary-weight transition only; full native weights, AdamW, evidence and historical work retained. No automatic retry, curriculum admission or promotion.'


def _keys(value,names,label):
    if type(value) is not dict or set(value)!=set(names):raise ValueError('exact '+label+' fields required')


def _pin(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def _count(value):
    if type(value) is not int or value<0:raise ValueError('nonnegative integer count required')
def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def identity(value):return hashlib.sha256(_json(value).encode()).hexdigest()
def _normal(value):return json.loads(_json(value))


def auxiliary_weight(value):
    if type(value) not in (int,float) or value not in (0.,.3):raise ValueError('declared auxiliary weight zero or .3 required')
    return float(value)


def _cost(value,keys=COST_KEYS):
    _keys(value,keys,'cost')
    for key,number in value.items():
        if type(number) not in (int,float) or not math.isfinite(number) or number<0:raise ValueError('finite nonnegative cost required')
        if key.endswith('invocations'):_count(number)


def expected_state_work(start,cursor,weight):
    _count(start);_count(cursor);weight=auxiliary_weight(weight)
    if cursor<start:raise ValueError('objective transition cursor cannot regress')
    return dict.fromkeys(STATE_COUNTS,3*start+(3*(cursor-start) if weight else 0))


def expected_parameter_step(name,start,cursor,weight):
    expected_state_work(start,cursor,weight)
    return start if not weight and name.startswith(AUX_PREFIXES) else cursor


def _metadata(saved,start,weight):
    """Validate actual history, without pretending skipped readouts occurred."""
    _keys(saved,('schema','recipe','weights','optimizer','cursor','evidence','accounting'),'kernel snapshot')
    cursor=saved['cursor'];_count(cursor);recipe=saved['recipe'];micro=recipe['micro_batch_size'];_count(micro)
    if saved['schema']!=KERNEL_SCHEMA or micro<2 or micro%2 or cursor<start:raise ValueError('complete native kernel snapshot required')
    evidence=saved['evidence']
    _keys(evidence,('cursor','consumed_bundle_identity_sha256','consumed_common_parent_identity_sha256','exposures'),'evidence')
    if type(evidence['cursor']) is not int or evidence['cursor']!=cursor or any(not _pin(evidence[k]) for k in
        ('consumed_bundle_identity_sha256','consumed_common_parent_identity_sha256')):raise ValueError('evidence cursor/pins differ')
    _keys(evidence['exposures'],FAMILIES,'families')
    for counts in evidence['exposures'].values():
        _keys(counts,EXPOSURES,'exposures')
        for number in counts.values():_count(number)
        if (counts['episodes']!=cursor*micro or not 8*counts['episodes']<=counts['turns']<=12*counts['episodes']
            or counts['observation_tokens']!=counts['observation_bytes']+2*counts['turns']
            or counts['reply_target_tokens']!=counts['reply_target_bytes']+counts['turns']):raise ValueError('lifetime exposure arithmetic differs')
    if len({v['turns'] for v in evidence['exposures'].values()})!=1:raise ValueError('family turns differ')
    accounting=saved['accounting'];_keys(accounting,('work','state_work','cost'),'accounting')
    expected=dict.fromkeys(WORK_COUNTS,0)
    for key in ('step_invocations','optimizer_attempts','optimizer_returns','synchronized_optimizer_updates','retained_updates'):expected[key]=cursor
    for key in ('attempted_forwards','completed_forwards','attempted_objectives','completed_objectives','attempted_backwards','completed_backwards'):expected[key]=3*cursor
    for key in ('attempted_forward_episodes','completed_forward_episodes','attempted_backward_episodes','completed_backward_episodes','retained_episodes'):expected[key]=3*cursor*micro
    for group in ('work','state_work'):
        for value in accounting[group].values():_count(value)
    if accounting['work']!=expected or accounting['state_work']!=expected_state_work(start,cursor,weight):raise ValueError('actual complete native/state work differs')
    _cost(accounting['cost'],('step_wall_seconds','step_cpu_seconds'))


def validate_transition(value,expected_sha256):
    _keys(value,('schema','parent_archive_sha256','legacy_identity','legacy_recipe','legacy_bridge_cost','start_cursor',
        'start_weights_sha256','start_evidence','start_accounting','auxiliary_weight','recipe','source_sha256','scope'),'transition')
    if not _pin(expected_sha256) or identity(value)!=expected_sha256 or value['schema']!=TRANSITION_SCHEMA or value['scope']!=SCOPE:
        raise ValueError('caller-pinned explicit objective transition required')
    for key in ('parent_archive_sha256','start_weights_sha256'):
        if not _pin(value[key]):raise ValueError('historical archive/weights pins required')
    start=value['start_cursor'];_count(start)
    if start<1:raise ValueError('completed nonempty legacy learner required')
    weight=auxiliary_weight(value['auxiliary_weight']);old=value['legacy_recipe'];expected=deepcopy(old);expected['auxiliary_weight']=weight
    if old.get('auxiliary_weight')!=.3 or _normal(value['recipe'])!=_normal(expected):raise ValueError('only declared auxiliary weight may change')
    bridge=value['legacy_identity'];_keys(bridge,('schema','origin_archive_sha256','origin','source_sha256','scope'),'legacy identity')
    origin=bridge['origin'];origin_step=origin['step'];_count(origin_step)
    if (bridge['schema']!=LEGACY_SCHEMA or not _pin(bridge['origin_archive_sha256']) or origin_step>start
        or origin['auxiliary_weight']!=.3 or _normal(origin['recipe'])!=_normal(old)
        or old['optimizer'][0]['lr']!=origin['learning_rate']):raise ValueError('unchanged historical learning identity required')
    legacy_keys=tuple(f'{kind}_{suffix}' for kind in ('restore','step','snapshot') for suffix in ('invocations','wall_seconds','cpu_seconds'))
    _cost(value['legacy_bridge_cost'],legacy_keys)
    if value['legacy_bridge_cost']['step_invocations']!=start-origin_step:raise ValueError('historical bridge steps differ')
    for pins in (bridge['source_sha256'],value['source_sha256']):
        if type(pins) is not dict or not pins or any(not _pin(pin) for pin in pins.values()):raise ValueError('explicit source pins required')
    _metadata(dict(schema=KERNEL_SCHEMA,recipe=old,weights=None,optimizer=None,cursor=start,
        evidence=value['start_evidence'],accounting=value['start_accounting']),start,.3)
    return deepcopy(value)


def validate_metadata(saved,transition):
    _metadata(saved,transition['start_cursor'],transition['auxiliary_weight'])
    if _normal(saved['recipe'])!=_normal(transition['recipe']):raise ValueError('post-transition recipe changed')
    for kind in ('step_wall_seconds','step_cpu_seconds'):
        if saved['accounting']['cost'][kind]<transition['start_accounting']['cost'][kind]:raise ValueError('historical cost regressed')
    for family in FAMILIES:
        if any(saved['evidence']['exposures'][family][key]<transition['start_evidence']['exposures'][family][key]
            for key in EXPOSURES):raise ValueError('historical exposure regressed')
    if saved['cursor']==transition['start_cursor'] and (saved['evidence']!=transition['start_evidence']
        or saved['accounting']!=transition['start_accounting']):raise ValueError('transition boundary history changed')


def source_hashes():
    from experiments import shared_state_continuation as old
    return {**old.source_hashes(),'experiments/shared_objective_continuation.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _runtime():
    import torch
    from experiments import shared_state_continuation as old
    from experiments import shared_state_training as kernel
    from experiments.foundation_layout_continuation import _same,_decode
    from brain_in_computer.learning_student import _cpu_copy,_check_finite_tree
    return torch,old,kernel,_same,_decode,_cpu_copy,_check_finite_tree


def _tensor_state(saved,trainer,transition):
    """Legacy strict tensor rules, with explicit inactive auxiliary AdamW steps."""
    torch,old,kernel,same,decode,cpu_copy,finite=_runtime();finite(saved,'objective transition snapshot')
    weights=saved['weights'];expected=trainer.model.state_dict();_keys(weights,expected,'weights')
    def tensor(value,reference):
        if (type(value) is not torch.Tensor or value.shape!=reference.shape or value.dtype!=reference.dtype
            or value.device.type!='cpu' or value.requires_grad or value.layout!=torch.strided
            or value.stride()!=reference.stride() or value.storage_offset()!=0):raise ValueError('tensor shape/dtype/layout/CPU boundary differs')
    for name,value in weights.items():tensor(value,expected[name])
    aliases={}
    for name,parameter in trainer.model.named_parameters(remove_duplicate=False):
        first=aliases.setdefault(id(parameter),name)
        if not same(weights[name],weights[first]):raise ValueError('tied parameter aliases differ')
    optimizer=saved['optimizer'];_keys(optimizer,('state','param_groups'),'AdamW')
    if not same(optimizer['param_groups'],trainer.optimizer.state_dict()['param_groups']):raise ValueError('AdamW parameter groups differ')
    parameters=list(trainer.model.named_parameters());moments=optimizer['state']
    if type(moments) is not dict or any(type(k) is not int for k in moments) or set(moments)!=set(range(len(parameters))):raise ValueError('full AdamW moments required')
    for index,values in moments.items():
        name,parameter=parameters[index];_keys(values,('step','exp_avg','exp_avg_sq'),'moments');step=values['step']
        wanted=expected_parameter_step(name,transition['start_cursor'],saved['cursor'],transition['auxiliary_weight'])
        if (type(step) is not torch.Tensor or step.shape!=torch.Size([]) or step.dtype!=torch.float32 or step.device.type!='cpu'
            or step.requires_grad or step.layout!=torch.strided or float(step)!=wanted):raise ValueError('active/inactive AdamW step differs')
        for key in ('exp_avg','exp_avg_sq'):tensor(values[key],parameter)
        if bool(values['exp_avg_sq'].lt(0).any()):raise ValueError('negative AdamW second moment')


def _restore_history(trainer,saved):
    # The frozen v1 kernel has no public restore method. This adapter restores
    # its accounting exactly as the reviewed legacy adapter does; no recipe,
    # source guard, optimizer behavior or method is replaced.
    trainer._evidence=deepcopy(saved['evidence']);trainer._work=deepcopy(saved['accounting']['work'])
    trainer._state_work=deepcopy(saved['accounting']['state_work']);trainer._cost=deepcopy(saved['accounting']['cost'])


class SharedObjectiveContinuation:
    def __init__(self):raise TypeError('use from_continuation or from_snapshot')

    @classmethod
    def from_continuation(cls,raw,*,expected_sha256,expected_identity,auxiliary_weight,device='cpu'):
        return cls._load(raw,expected_sha256,device,origin=expected_identity,weight=auxiliary_weight)

    @classmethod
    def from_snapshot(cls,raw,*,expected_sha256,expected_transition_sha256,device='cpu'):
        return cls._load(raw,expected_sha256,device,transition_pin=expected_transition_sha256)

    @classmethod
    def _load(cls,raw,pin,device,*,origin=None,weight=None,transition_pin=None):
        started,cpu=time.monotonic(),time.process_time();operation='transition' if origin is not None else 'restore'
        report=dict(schema=SCHEMA,operation=operation,failed=True,archive_load_attempts=0,archive_load_completions=0,
            model_construction_attempts=0,model_constructions=0,optimizer_construction_attempts=0,optimizer_constructions=0,
            forwards=0,backwards=0,optimizer_updates=0,legacy_snapshot_attempts=0,legacy_snapshot_completions=0,
            count_scope='Top-level counters are direct; legacy_restore counters are additional nested physical work. Legacy snapshot extraction is separately recorded and included in outer transition time.')
        torch,old,kernel,same,decode,cpu_copy,finite=_runtime()
        try:
            sources=source_hashes();payload=decode(raw,pin,report)
            if operation=='transition':
                weight=auxiliary_weight(weight)
                try:legacy=old.SharedStateContinuation.from_snapshot(raw,expected_sha256=pin,expected_identity=origin,device=device)
                except BaseException as error:
                    report['legacy_restore']=deepcopy(getattr(error,'continuation_report',{}));raise
                report['legacy_restore']=deepcopy(legacy.last_restore_report)
                report['legacy_snapshot_attempts']+=1
                try:previous=legacy.snapshot()
                finally:report['legacy_snapshot']=deepcopy(legacy.last_snapshot_report)
                report['legacy_snapshot_completions']+=1
                saved=previous['learner'];recipe=saved['recipe']
                trainer=kernel.SharedStateKernel(legacy.model,legacy.optimizer,config=legacy.model.config,layout=recipe['layout'],
                    micro_batch_size=recipe['micro_batch_size'],objective_id=old.training.OBJECTIVE_ID,auxiliary_weight=weight)
                _restore_history(trainer,saved)
                transition=dict(schema=TRANSITION_SCHEMA,parent_archive_sha256=pin,legacy_identity=deepcopy(payload['identity']),
                    legacy_recipe=deepcopy(recipe),legacy_bridge_cost=deepcopy(payload['bridge_cost']),start_cursor=saved['cursor'],
                    start_weights_sha256=payload['weights_sha256'],start_evidence=deepcopy(saved['evidence']),
                    start_accounting=deepcopy(saved['accounting']),auxiliary_weight=weight,recipe=trainer.recipe,source_sha256=sources,scope=SCOPE)
                transition_pin=identity(transition);validate_transition(transition,transition_pin)
                expected={**saved,'recipe':trainer.recipe};validate_metadata(expected,transition);_tensor_state(expected,trainer,transition)
                if not same(trainer.snapshot(),expected):raise ValueError('transition changed incoming weights/AdamW/history')
                costs={key:0 if key.endswith('invocations') else 0. for key in COST_KEYS}
            else:
                _keys(payload,('schema','transition','transition_sha256','learner','weights_sha256','lifetime_updates','objective_updates','cost'),'objective envelope')
                transition=validate_transition(payload['transition'],transition_pin)
                if payload['schema']!=SCHEMA or payload['transition_sha256']!=transition_pin:raise ValueError('objective envelope identity differs')
                saved=payload['learner'];validate_metadata(saved,transition);costs=deepcopy(payload['cost']);_cost(costs)
                if (payload['lifetime_updates']!=saved['cursor'] or payload['objective_updates']!=saved['cursor']-transition['start_cursor']
                    or costs['step_invocations']!=payload['objective_updates'] or costs['transition_invocations']!=1):raise ValueError('objective lifetime/cost boundary differs')
                origin=old._identity(transition['legacy_identity']['origin'])
                if (old.current_runtime(device)!=origin['runtime'] or torch.get_default_dtype()!=torch.float32
                    or transition['legacy_identity']['source_sha256']!=old.source_hashes()
                    or transition['legacy_identity']['scope']!=old.SCOPE):raise ValueError('legacy identity/runtime/source changed')
                from experiments.sequence_student import SequenceConfig
                report['model_construction_attempts']+=1
                model=old.student.build_shared_state_student(0,device=device,config=SequenceConfig(**saved['recipe']['config']));report['model_constructions']+=1
                report['optimizer_construction_attempts']+=1
                optimizer=torch.optim.AdamW(model.parameters(),lr=origin['learning_rate']);report['optimizer_constructions']+=1
                recipe=saved['recipe'];trainer=kernel.SharedStateKernel(model,optimizer,config=model.config,layout=recipe['layout'],
                    micro_batch_size=recipe['micro_batch_size'],objective_id=old.training.OBJECTIVE_ID,auxiliary_weight=transition['auxiliary_weight'])
                if not same(trainer.recipe,recipe):raise ValueError('declared native kernel recipe differs')
                _tensor_state(saved,trainer,transition);model.load_state_dict(saved['weights'],strict=True)
                optimizer.load_state_dict(cpu_copy(saved['optimizer']));_restore_history(trainer,saved);trainer._sync()
                if not same(trainer.snapshot(),saved) or old.checkpoint_digest(model)!=payload['weights_sha256']:raise ValueError('exact restored full state differs')
            if transition['source_sha256']!=sources or source_hashes()!=sources:raise ValueError('objective transition sources changed')
            value=object.__new__(cls);value._kernel,value._transition,value._sources=trainer,deepcopy(transition),sources
            value._transition_pin=transition_pin;value._cost=costs;value._failed=False
            value.last_step_report=value.last_snapshot_report=None;value._guard()
            report.update(failed=False,exact_state_preserved=True,exact_state_restored=True,cursor=value.cursor)
        except BaseException as error:
            report['error']=repr(error)
            try:
                if torch.device(device).type=='cuda' and torch.cuda.is_initialized():torch.cuda.synchronize(torch.device(device))
                report['queued_device_work_synchronized']=True
            except BaseException as sync_error:report['failure_sync_error']=repr(sync_error)
            report.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
            error.objective_report=deepcopy(report);raise
        report.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
        value._record(operation,report);value.last_restore_report=deepcopy(report);return value

    def _record(self,kind,report):
        self._cost[kind+'_invocations']+=1
        for clock in ('wall','cpu'):self._cost[kind+'_'+clock+'_seconds']+=report[clock+'_seconds']
    @property
    def model(self):return self._kernel.model
    @property
    def optimizer(self):return self._kernel.optimizer
    @property
    def cursor(self):return self._kernel.cursor
    @property
    def evidence(self):return self._kernel.evidence
    @property
    def recipe(self):return self._kernel.recipe
    @property
    def transition(self):return deepcopy(self._transition)
    @property
    def transition_sha256(self):return self._transition_pin
    @property
    def last_report(self):return deepcopy(self._kernel.last_report)
    @property
    def failed(self):return self._failed or self._kernel.failed
    @property
    def accounting(self):return dict(lifetime_kernel=self._kernel.accounting,objective_bridge_cost=deepcopy(self._cost),
        previous_bridge_cost=deepcopy(self._transition['legacy_bridge_cost']),scope='Outer bridge timings include nested kernel/legacy work; do not sum twice.')

    def _guard(self):
        if self.failed:raise RuntimeError('poisoned objective continuation cannot retry')
        torch,old,*_=_runtime()
        if (identity(self._transition)!=self._transition_pin or source_hashes()!=self._sources
            or old.current_runtime(self.model.tokens.weight.device)!=self._transition['legacy_identity']['origin']['runtime']):raise ValueError('objective identity/source/runtime changed')

    def step(self,prepared,*,state_targets,deadline=None):
        started,cpu=time.monotonic(),time.process_time();report=dict(failed=True,kernel_called=False,cursor_before=self.cursor)
        self.last_step_report=report;failure=None
        try:
            self._guard();report['kernel_called']=True
            result=self._kernel.step(prepared,state_targets=state_targets,deadline=deadline);self._guard();report['failed']=False;return result
        except BaseException as error:
            failure=error;self._failed=self._kernel.failed=True;report['error']=repr(error)
            if report['kernel_called']:report['kernel_report']=self.last_report
            raise
        finally:
            report.update(cursor=self.cursor,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu);self._record('step',report)
            if failure is not None:failure.objective_report=deepcopy(report)

    def snapshot(self):
        started,cpu=time.monotonic(),time.process_time();report=dict(failed=True,cursor=self.cursor);self.last_snapshot_report=report;failure=None
        try:
            self._guard();saved=self._kernel.snapshot();validate_metadata(saved,self._transition)
            torch,old,*_=_runtime();pin=old.checkpoint_digest(self.model);self._guard();report['failed']=False
        except BaseException as error:
            failure=error;self._failed=self._kernel.failed=True;report['error']=repr(error);raise
        finally:
            report.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu);self._record('snapshot',report)
            if failure is not None:failure.objective_report=deepcopy(report)
        return dict(schema=SCHEMA,transition=self.transition,transition_sha256=self._transition_pin,learner=saved,
            weights_sha256=pin,lifetime_updates=self.cursor,objective_updates=self.cursor-self._transition['start_cursor'],cost=deepcopy(self._cost))
