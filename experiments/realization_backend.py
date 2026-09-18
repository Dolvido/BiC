"""Transactional, budgeted raw-English training backend; no curriculum policy.

This separate adapter never changes the frozen study or selects/promotes models.
Callers supply finite schedules and development-only evaluation banks. Durable
commits contain weights, optimizer, consumed stream and schedule cursor together.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import tempfile
import threading
import time

import torch

from brain_in_computer.learning_loop import run_lock
from brain_in_computer.learning_student import _check_finite_tree
from experiments.composition_evaluation import PreparedBank
from experiments.composition_curriculum import query_ancestries
from experiments.realization_banks import transcript_digest
from experiments.realization_training import RealizationTrainer
from experiments.sequence_student import SequenceConfig


SCHEMA = 'bic-raw-english-backend-v1'
MAX_SCHEDULE_UPDATES = 4096


def source_hashes():
    # The existing closure includes all transitive learner/packing/oracle code;
    # reading source files does not open any live study results or checkpoint.
    from experiments.realization_study import source_hashes as frozen_sources
    root = Path(__file__).resolve().parents[1]
    name = 'experiments/realization_backend.py'
    return {**frozen_sources(), name: hashlib.sha256((root/name).read_bytes()).hexdigest()}


def _file_digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _finite(value, name):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')


def progress_vector(metrics):
    """Keep paired/reply/known/ASK denominators; never substitute an aggregate."""
    pairs = metrics['final_pairs']
    false_ask = metrics['ask_predicted'] - metrics['ask_correct']
    known = metrics['known_total']
    return {'final_pairs': copy.deepcopy(pairs),
            'final_pair_accuracy': metrics['final_pair_accuracy'],
            'opposite_pairs': {'correct':metrics['opposite_pair_correct'], 'total':metrics['opposite_pair_total']},
            'known_queries': {'correct':metrics['known_correct'], 'total':known, 'accuracy':metrics['known_accuracy']},
            'unknown_queries': {'correct':metrics['ask_correct'], 'total':metrics['ask_true'],
                                'recall':metrics['ask_recall'], 'precision':metrics['ask_precision']},
            'unsupported_ask': {'count':false_ask, 'known_total':known,
                                'rate':false_ask/known if known else None},
            'query_replies': {'correct':metrics['query_reply_correct'], 'total':metrics['query_total'],
                              'accuracy':metrics['query_reply_accuracy']},
            'final_reply_pairs': {'correct':metrics['final_reply_pair_correct'], 'total':pairs['total'],
                                  'accuracy':metrics['final_reply_pair_accuracy']},
            'action_reply_agreement':metrics['action_reply_agreement'],
            'parseable_query_replies':metrics['reply_parseable_queries'],
            'query_loss':metrics['query_loss'], 'brier_score':metrics['brier_score'],
            'by_turn':copy.deepcopy(metrics['by_turn'])}


class RealizationBackend:
    """Reusable backend around canonical streams, not an autonomous controller.

    A schedule is a finite list of three-family microbatch tuples, one tuple per
    update. It may repeat a family. A pending schedule cannot be replaced. Supply
    no schedule after resume to continue its stored cursor. `deadline` is absolute
    monotonic time, and budgets cover validation, steps and commit/save overhead.
    """

    def __init__(self, train_banks, *, mode, protected_transcripts, evaluation_banks=None,
                 seed=2901, sampler_seed=3901, device='cpu', micro_batch_size=32,
                 learning_rate=.001, config=None, payload=None):
        self._lock = threading.RLock()
        self._sources = source_hashes()
        self._evaluations = {}
        config = config or SequenceConfig(max_turns=12)
        evaluation_banks = {} if evaluation_banks is None else evaluation_banks
        if not isinstance(evaluation_banks, dict) or not set(evaluation_banks) <= {'development','retention'}:
            raise ValueError('only named development/retention banks belong in the backend')
        protected = set(protected_transcripts)
        for role, banks in evaluation_banks.items():
            if not isinstance(banks, dict) or not banks or any(not isinstance(name,str) or not name for name in banks):
                raise ValueError('evaluation role requires nonempty named banks')
            if any(query['known'] and query['composed'] and query['structure_partition']=='audit'
                    for rows in banks.values() for row in rows for query in query_ancestries(row)):
                raise ValueError('development/retention exposes supervised audit ancestry')
            self._evaluations[role] = {name: PreparedBank(rows,role='dev',config=config) for name,rows in banks.items()}
            protected.update(transcript_digest(row) for rows in banks.values() for row in rows)
        self._trainer = RealizationTrainer(train_banks,mode=mode,protected_transcripts=sorted(protected),
            seed=seed,sampler_seed=sampler_seed,device=device,micro_batch_size=micro_batch_size,
            learning_rate=learning_rate,config=config)
        self._contract = {'schema':SCHEMA, 'source_sha256':self._sources,
                          'evaluation_banks':{role:{name:bank.identity for name,bank in banks.items()}
                                              for role,banks in self._evaluations.items()}}
        self._curriculum = {'schedule':[], 'cursor':0, 'generation':0, 'base_updates':0,
                            'base_family_microbatches':dict.fromkeys(self._trainer.banks,0)}
        self._accounting = {'retained_step_seconds':0.0}
        self._disk_path, self._disk_digest = None, None
        self._pending_publication = None
        self._publication_uncertain = False
        self.last_report = None
        if payload is not None:
            self._apply(payload)
        self._committed = self._snapshot()

    @property
    def model(self):
        return self._trainer.model

    @property
    def updates(self):
        return self._trainer.updates

    @property
    def curriculum(self):
        return copy.deepcopy(self._curriculum)

    def _assert_sources(self):
        if self._publication_uncertain:
            raise RuntimeError('checkpoint publication is uncertain; explicitly reload before continuing')
        if source_hashes() != self._sources:
            raise ValueError('backend dependency source changed; start a new declared run')

    def _schedule(self, value):
        if not isinstance(value,(list,tuple)) or not 1 <= len(value) <= MAX_SCHEDULE_UPDATES:
            raise ValueError('finite nonempty schedule of at most 4096 updates required')
        result=[]
        for families in value:
            if (not isinstance(families,(list,tuple)) or len(families)!=3
                    or any(type(family) is not str or family not in self._trainer.banks for family in families)):
                raise ValueError('every scheduled update requires three admitted family names')
            result.append(list(families))
        return result

    def _validate_cursor(self, curriculum, learner):
        expected={'schedule','cursor','generation','base_updates','base_family_microbatches'}
        if not isinstance(curriculum,dict) or set(curriculum)!=expected:
            raise ValueError('invalid curriculum cursor fields')
        schedule=curriculum['schedule']
        if schedule:
            if self._schedule(schedule)!=schedule:
                raise ValueError('schedule must use canonical list representation')
        elif schedule!=[]:
            raise ValueError('empty schedule must be a list')
        for key in ('cursor','generation','base_updates'):
            _integer(curriculum[key],key)
        cursor=curriculum['cursor']
        base=curriculum['base_family_microbatches']
        if (cursor>len(schedule) or learner['updates']!=curriculum['base_updates']+cursor
                or not isinstance(base,dict) or set(base)!=set(self._trainer.banks)):
            raise ValueError('curriculum cursor disagrees with learner updates')
        for count in base.values():_integer(count,'base family count')
        if sum(base.values())!=curriculum['base_updates']*3:
            raise ValueError('curriculum base family counts disagree')
        expected_counts=dict(base)
        for families in schedule[:cursor]:
            for family in families:expected_counts[family]+=1
        if expected_counts!=learner['family_microbatches']:
            raise ValueError('curriculum schedule disagrees with consumed family stream')

    def _snapshot(self):
        learner=self._trainer.snapshot()
        self._validate_cursor(self._curriculum,learner)
        return {'schema':SCHEMA,'contract':copy.deepcopy(self._contract), 'learner':learner,
                'curriculum':copy.deepcopy(self._curriculum),'accounting':copy.deepcopy(self._accounting)}

    def snapshot(self):
        with self._lock:
            self._assert_sources()
            return self._snapshot()

    def _apply(self,payload):
        if (not isinstance(payload,dict) or set(payload)!={'schema','contract','learner','curriculum','accounting'}
                or payload['schema']!=SCHEMA or payload['contract']!=self._contract):
            raise ValueError('backend checkpoint contract/source/evaluation identity differs')
        _check_finite_tree(payload,'backend checkpoint')
        self._validate_cursor(payload['curriculum'],payload['learner'])
        accounting=payload['accounting']
        if (not isinstance(accounting,dict) or set(accounting)!={'retained_step_seconds'}
                or type(accounting['retained_step_seconds']) not in (int,float)
                or accounting['retained_step_seconds']<0):
            raise ValueError('invalid retained timing ledger')
        self._trainer._restore(payload['learner'])
        self._curriculum=copy.deepcopy(payload['curriculum'])
        self._accounting=copy.deepcopy(accounting)

    def restore(self,payload):
        """Transactional in-memory restore; cannot silently detach a disk writer."""
        with self._lock:
            self._assert_sources()
            if self._disk_path is not None:
                raise ValueError('load a separate backend to replace a disk-bound run')
            before=self._snapshot()
            try:self._apply(payload)
            except BaseException:
                self._apply(before)
                raise
            self._committed=self._snapshot()

    def _bind_path(self,path):
        path=Path(path).resolve()
        if self._disk_path is not None and path!=self._disk_path:
            raise ValueError('backend is already bound to a different checkpoint path')
        if self._disk_path is None:
            if path.exists():
                raise FileExistsError('existing checkpoint requires explicit RealizationBackend.load')
            self._disk_path=path
        path.parent.mkdir(parents=True,exist_ok=True)
        return path

    def _assert_disk_current(self):
        if self._disk_path is not None:
            exists=self._disk_path.exists()
            if self._disk_digest is None and exists or self._disk_digest is not None and (
                    not exists or _file_digest(self._disk_path)!=self._disk_digest):
                raise RuntimeError('checkpoint advanced or changed; reload before writing')

    def _write(self,payload):
        """Same-directory atomic replace; digest is known before publication."""
        self._assert_disk_current()
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='wb',prefix=self._disk_path.name+'.',suffix='.tmp',
                                              dir=self._disk_path.parent,delete=False) as stream:
                temporary=Path(stream.name)
                torch.save(payload,stream)
                stream.flush();os.fsync(stream.fileno())
            digest=_file_digest(temporary)
            self._pending_publication={'payload':payload,'digest':digest,'prior_digest':self._disk_digest}
            os.replace(temporary,self._disk_path)
            self._disk_digest=digest
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)

    def _commit(self):
        self._assert_sources()
        payload=self._snapshot()
        self._pending_publication=None
        try:
            if self._disk_path is not None:self._write(payload)
            self._committed=payload
        except BaseException:
            # Publication may have succeeded immediately before an interrupt or
            # cleanup error. Resolve the actual file before deciding rollback.
            pending=self._pending_publication
            if pending is not None:
                try:
                    actual=_file_digest(self._disk_path) if self._disk_path.exists() else None
                    if actual==pending['digest']:
                        self._disk_digest=actual
                        self._committed=pending['payload']
                    elif actual!=pending['prior_digest']:
                        self._publication_uncertain=True
                except BaseException:
                    self._publication_uncertain=True
            raise
        finally:
            self._pending_publication=None

    def save(self,path):
        with self._lock:
            self._bind_path(path)
            with run_lock(self._disk_path.parent):self._commit()
            return {'path':str(self._disk_path),'sha256':self._disk_digest,'updates':self.updates}

    @classmethod
    def load(cls,path,train_banks,*,protected_transcripts,evaluation_banks=None,device='cpu'):
        path=Path(path).resolve()
        before=_file_digest(path)
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if _file_digest(path)!=before:raise RuntimeError('checkpoint changed during load')
        recipe=payload['learner']['recipe']
        result=cls(train_banks,mode=recipe['realization']['mode'],protected_transcripts=protected_transcripts,
            evaluation_banks=evaluation_banks,seed=recipe['seed'],sampler_seed=recipe['sampler_seed'],
            device=device,micro_batch_size=recipe['micro_batch_size'],learning_rate=recipe['learning_rate'],
            config=SequenceConfig(**recipe['config']),payload=payload)
        result._disk_path,result._disk_digest=path,before
        return result

    def train_chunk(self,family_schedule=None,*,max_updates=None,deadline=None,commit_interval=16,checkpoint_path=None):
        """Run a bounded schedule chunk; all failures leave an explicit report."""
        with self._lock:
            started=time.monotonic()
            self.last_report=None
            try:
                return self._train_chunk(family_schedule,max_updates=max_updates,deadline=deadline,
                                         commit_interval=commit_interval,checkpoint_path=checkpoint_path)
            except BaseException:
                if self.last_report is None or self.last_report.get('status')=='running':
                    self.last_report={'status':'failed','failure_stage':'preflight',
                        'completed_updates':0,'committed_updates':0,'discarded_completed_updates':0,
                        'failed_step_attempts':0,'wall_seconds':time.monotonic()-started,
                        'retained_total_updates':None if self._publication_uncertain else self.updates,
                        'in_memory_updates':self.updates}
                raise

    def _synchronize(self):
        device=next(self.model.parameters()).device
        if device.type=='cuda':torch.cuda.synchronize(device)

    def _train_chunk(self,family_schedule=None,*,max_updates=None,deadline=None,commit_interval=16,checkpoint_path=None):
        """Bound work by caller deadline/update count and commit at safe boundaries.

        At most one in-flight step plus the current snapshot/save may finish after
        the deadline; no next step starts then. Failed steps or saves discard all
        successful work since the last commit. `last_report` records physical
        wall time and discarded completed updates even when an exception escapes.
        """
        started=time.monotonic()
        starting_updates=self.updates
        if max_updates is None and deadline is None:raise ValueError('an update bound or deadline is required')
        if max_updates is not None:_integer(max_updates,'max_updates')
        if deadline is not None:_finite(deadline,'deadline')
        _integer(commit_interval,'commit_interval',1)
        if commit_interval>MAX_SCHEDULE_UPDATES:raise ValueError('commit interval exceeds bounded schedule capacity')
        report={'status':'running','completed_updates':0,'committed_updates':0,
                'discarded_completed_updates':0,'failed_step_attempts':0,'step_seconds':0.,
                'discarded_step_seconds':0.,'commit_seconds':0.,'rollback_seconds':0.,
                'wall_seconds':0.,'deadline_overrun_seconds':0.,'commit_interval':commit_interval,
                'time_scope':'Invocation wall includes validation, generation, packing, optimization, commits and rollback; step_seconds includes generation and is not GPU-only.'}
        self.last_report=report
        with self._lock:
            self._assert_sources()
            if family_schedule is not None:family_schedule=self._schedule(family_schedule)
            if family_schedule is not None and self._curriculum['cursor']<len(self._curriculum['schedule']):
                raise ValueError('cannot replace a pending schedule; resume it without a new schedule')
            if checkpoint_path is not None:self._bind_path(checkpoint_path)
            from contextlib import nullcontext
            guard=run_lock(self._disk_path.parent) if self._disk_path is not None else nullcontext()
            with guard:
                self._assert_disk_current()
                pending=0;pending_seconds=0.;stage='setup'
                def commit():
                    tick=time.monotonic()
                    try:self._commit()
                    finally:report['commit_seconds']+=time.monotonic()-tick
                try:
                    expired=deadline is not None and time.monotonic()>=deadline
                    if expired or max_updates==0:
                        report['status']='deadline' if expired else 'update_bound'
                        return report
                    if family_schedule is not None:
                        self._curriculum={'schedule':family_schedule,'cursor':0,
                            'generation':self._curriculum['generation']+1,'base_updates':self.updates,
                            'base_family_microbatches':copy.deepcopy(self._trainer.family_microbatches)}
                        stage='commit';commit()
                    while self._curriculum['cursor']<len(self._curriculum['schedule']):
                        if deadline is not None and time.monotonic()>=deadline:
                            report['status']='deadline';break
                        if max_updates is not None and report['completed_updates']>=max_updates:
                            report['status']='update_bound';break
                        self._synchronize()
                        if deadline is not None and time.monotonic()>=deadline:
                            report['status']='deadline';break
                        stage='step';tick=time.monotonic()
                        try:self._trainer.step(self._curriculum['schedule'][self._curriculum['cursor']])
                        finally:
                            try:self._synchronize()
                            finally:
                                elapsed=time.monotonic()-tick
                                report['step_seconds']+=elapsed;pending_seconds+=elapsed
                        self._curriculum['cursor']+=1;pending+=1;report['completed_updates']+=1
                        self._accounting['retained_step_seconds']+=elapsed
                        if pending>=commit_interval:
                            stage='commit';commit()
                            report['committed_updates']+=pending;pending=0;pending_seconds=0.
                    if report['status']=='running':report['status']='schedule_exhausted'
                    if pending:
                        stage='commit';commit()
                        report['committed_updates']+=pending;pending=0;pending_seconds=0.
                    return report
                except BaseException:
                    report['status']='failed';report['failure_stage']=stage
                    report['failed_step_attempts']=int(stage=='step')
                    if self._publication_uncertain:
                        report['publication_uncertain']=True
                        report['discarded_completed_updates']=None
                        report['committed_updates']=None
                    else:
                        retained=self._committed['learner']['updates']
                        discarded=max(0,starting_updates+report['completed_updates']-retained)
                        report['discarded_completed_updates']=discarded
                        report['discarded_step_seconds']=pending_seconds if discarded or stage=='step' else 0.
                        tick=time.monotonic();self._apply(self._committed)
                        report['rollback_seconds']=time.monotonic()-tick
                    raise
                finally:
                    report['wall_seconds']=time.monotonic()-started
                    if deadline is not None:report['deadline_overrun_seconds']=max(0.,time.monotonic()-deadline)
                    report['retained_total_updates']=None if self._publication_uncertain else self.updates
                    report['in_memory_updates']=self.updates
                    if not self._publication_uncertain:
                        report['committed_updates']=self.updates-starting_updates
                    report['schedule_cursor']=self._curriculum['cursor']
                    report['remaining_schedule_updates']=len(self._curriculum['schedule'])-self._curriculum['cursor']
                    self.last_report=copy.deepcopy(report)

    def evaluate(self,role,*,names=None,batch_size=32,control='normal'):
        """Teacher-free development/retention scoring; no weights or stream changes."""
        started=time.monotonic()
        with self._lock:
            self._assert_sources()
            if role not in self._evaluations:raise ValueError('unknown prepared development/retention role')
            available=self._evaluations[role]
            chosen=list(available) if names is None else list(names)
            if not chosen or len(set(chosen))!=len(chosen) or any(name not in available for name in chosen):
                raise ValueError('evaluation requires distinct existing bank names')
            rows={name:available[name].score(self.model,batch_size=batch_size,score_replies=True,control=control)
                  for name in chosen}
            return {'role':role,'per_bank':rows,'progress':{name:progress_vector(row) for name,row in rows.items()},
                    'wall_seconds':time.monotonic()-started,'automatic_promotion':False}
