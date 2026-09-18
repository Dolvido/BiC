"""Explicit AdamW rate transition from a completed objective-control snapshot.

Pure metadata admission is stdlib-only. Runtime methods reuse the frozen
SharedStateStudent/SharedStateKernel; no lesson, objective or source guard changes.
The sole deliberate state difference at admission is optimizer-group learning rate.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import time

from experiments import shared_objective_continuation as objective

SCHEMA = 'bic-shared-rate-continuation-v1'
TRANSITION_SCHEMA = 'bic-shared-rate-transition-v1'
RATES = (.0003, .0001)
SOURCE_RATE = .0003
COST_KEYS = objective.COST_KEYS
SCOPE = ('Explicit .0003 -> .0003/.0001 AdamW rate transition; auxiliary weight .3, '
         'full weights, moments, individual steps, evidence and historical work retained. '
         'No automatic retry, lesson admission or promotion.')
identity = objective.identity


def learning_rate(value):
    if type(value) is not float or value not in RATES:
        raise ValueError('explicit finite float rate .0003 or .0001 required')
    return value


def changed_recipe(recipe, rate):
    """Detached recipe with exactly the declared single-group lr change."""
    rate = learning_rate(rate)
    if (type(recipe) is not dict or recipe.get('schema') != objective.KERNEL_SCHEMA
            or recipe.get('auxiliary_weight') != .3 or type(recipe.get('optimizer')) is not list
            or len(recipe['optimizer']) != 1 or type(recipe['optimizer'][0]) is not dict
            or type(recipe['optimizer'][0].get('lr')) is not float
            or recipe['optimizer'][0]['lr'] != SOURCE_RATE):
        raise ValueError('one unchanged native .3 objective and .0003 AdamW group required')
    result = deepcopy(recipe)
    result['optimizer'][0]['lr'] = rate
    return result


def validate_parent(payload, expected_transition_sha256):
    """Validate completed objective-control metadata; tensor checks are runtime-only."""
    objective._keys(payload, ('schema', 'transition', 'transition_sha256', 'learner',
        'weights_sha256', 'lifetime_updates', 'objective_updates', 'cost'), 'objective parent')
    transition = objective.validate_transition(payload['transition'], expected_transition_sha256)
    if (payload['schema'] != objective.SCHEMA or payload['transition_sha256'] != expected_transition_sha256
            or transition['auxiliary_weight'] != .3 or not objective._pin(payload['weights_sha256'])):
        raise ValueError('caller-pinned weight-.3 objective-control parent required')
    saved = payload['learner']
    objective.validate_metadata(saved, transition)
    changed_recipe(saved['recipe'], SOURCE_RATE)
    objective._cost(payload['cost'])
    for name in ('lifetime_updates', 'objective_updates'):
        objective._count(payload[name])
    if (payload['lifetime_updates'] != saved['cursor'] or payload['objective_updates'] < 1
            or payload['objective_updates'] != saved['cursor'] - transition['start_cursor']
            or payload['cost']['step_invocations'] != payload['objective_updates']
            or payload['cost']['transition_invocations'] != 1 or payload['cost']['snapshot_invocations'] < 1):
        raise ValueError('completed nonempty objective continuation history required')
    return transition


def validate_transition(value, expected_sha256):
    objective._keys(value, ('schema', 'parent_archive_sha256', 'parent_transition', 'parent_transition_sha256',
        'parent_recipe', 'parent_bridge_cost', 'start_cursor', 'start_weights_sha256', 'start_evidence',
        'start_accounting', 'source_learning_rate', 'learning_rate', 'recipe', 'source_sha256', 'scope'), 'rate transition')
    if (not objective._pin(expected_sha256) or identity(value) != expected_sha256
            or value['schema'] != TRANSITION_SCHEMA or value['scope'] != SCOPE):
        raise ValueError('caller-pinned versioned rate transition required')
    for key in ('parent_archive_sha256', 'start_weights_sha256'):
        if not objective._pin(value[key]):
            raise ValueError('explicit incoming archive and weight pins required')
    if type(value['source_learning_rate']) is not float or value['source_learning_rate'] != SOURCE_RATE:
        raise ValueError('historical source rate must remain .0003')
    expected = changed_recipe(value['parent_recipe'], value['learning_rate'])
    if objective._normal(value['recipe']) != objective._normal(expected):
        raise ValueError('only optimizer-group lr may change')
    start = value['start_cursor']
    objective._count(start)
    saved = dict(schema=objective.KERNEL_SCHEMA, recipe=value['parent_recipe'], weights=None, optimizer=None,
        cursor=start, evidence=value['start_evidence'], accounting=value['start_accounting'])
    parent = dict(schema=objective.SCHEMA, transition=value['parent_transition'],
        transition_sha256=value['parent_transition_sha256'], learner=saved, weights_sha256=value['start_weights_sha256'],
        lifetime_updates=start, objective_updates=start-value['parent_transition']['start_cursor'], cost=value['parent_bridge_cost'])
    validate_parent(parent, value['parent_transition_sha256'])
    pins = value['source_sha256']
    if type(pins) is not dict or not pins or any(type(k) is not str or not objective._pin(v) for k, v in pins.items()):
        raise ValueError('explicit rate-boundary source pins required')
    return deepcopy(value)


def validate_metadata(saved, transition):
    # Both rates keep every parameter active and every state readout: 3 * lifetime.
    objective.validate_metadata(saved, dict(start_cursor=transition['start_cursor'], auxiliary_weight=.3,
        recipe=transition['recipe'], start_accounting=transition['start_accounting'], start_evidence=transition['start_evidence']))


def validate_snapshot(payload, expected_transition_sha256):
    objective._keys(payload, ('schema', 'transition', 'transition_sha256', 'learner', 'weights_sha256',
        'lifetime_updates', 'rate_updates', 'cost'), 'rate envelope')
    transition = validate_transition(payload['transition'], expected_transition_sha256)
    if (payload['schema'] != SCHEMA or payload['transition_sha256'] != expected_transition_sha256
            or not objective._pin(payload['weights_sha256'])):
        raise ValueError('declared rate envelope identity differs')
    validate_metadata(payload['learner'], transition)
    objective._cost(payload['cost'])
    for name in ('lifetime_updates', 'rate_updates'):
        objective._count(payload[name])
    if (payload['lifetime_updates'] != payload['learner']['cursor']
            or payload['rate_updates'] != payload['learner']['cursor'] - transition['start_cursor']
            or payload['cost']['step_invocations'] != payload['rate_updates']
            or payload['cost']['transition_invocations'] != 1 or payload['cost']['snapshot_invocations'] < 1):
        raise ValueError('rate continuation lifetime/cost differs')
    if payload['rate_updates'] == 0 and payload['weights_sha256'] != transition['start_weights_sha256']:
        raise ValueError('zero-step weights differ from the parent')
    return transition


def source_hashes():
    return {**objective.source_hashes(), 'experiments/shared_rate_continuation.py':
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _origin(transition):
    return transition['parent_transition']['legacy_identity']['origin']


def _tensor_state(saved, trainer, transition):
    objective._tensor_state(saved, trainer, dict(start_cursor=transition['start_cursor'], auxiliary_weight=.3))


class SharedRateContinuation:
    def __init__(self):
        raise TypeError('use from_objective_snapshot or from_snapshot')

    @classmethod
    def from_objective_snapshot(cls, raw, *, expected_sha256, expected_transition_sha256, learning_rate, device='cpu'):
        return cls._load(raw, expected_sha256, expected_transition_sha256, device, rate=learning_rate)

    @classmethod
    def from_snapshot(cls, raw, *, expected_sha256, expected_transition_sha256, device='cpu'):
        return cls._load(raw, expected_sha256, expected_transition_sha256, device)

    @classmethod
    def _load(cls, raw, pin, transition_pin, device, *, rate=None):
        started, cpu = time.monotonic(), time.process_time()
        operation = 'restore' if rate is None else 'transition'
        report = dict(schema=SCHEMA, operation=operation, failed=True, archive_load_attempts=0, archive_load_completions=0,
            model_construction_attempts=0, model_constructions=0, optimizer_construction_attempts=0, optimizer_constructions=0,
            kernel_construction_attempts=0, kernel_constructions=0, forwards=0, backwards=0, optimizer_updates=0,
            parent_snapshot_attempts=0, parent_snapshot_completions=0, verification_snapshot_attempts=0,
            verification_snapshot_completions=0, partial_work_unknown=False,
            count_scope='Direct counters only; objective_restore is additional nested work. Parent/verification snapshot calls are explicit. Outer time includes all nested time.')
        torch = None
        try:
            torch, old, kernel, same, decode, cpu_copy, finite = objective._runtime()
            sources = source_hashes()
            payload = decode(raw, pin, report)
            if operation == 'transition':
                rate = learning_rate(rate)
                validate_parent(payload, transition_pin)
                try:
                    parent = objective.SharedObjectiveContinuation.from_snapshot(raw, expected_sha256=pin,
                        expected_transition_sha256=transition_pin, device=device)
                except BaseException as error:
                    report['objective_restore'] = deepcopy(getattr(error, 'objective_report', {}))
                    raise
                report['objective_restore'] = deepcopy(parent.last_restore_report)
                report['parent_snapshot_attempts'] += 1
                try:
                    previous = parent.snapshot()
                finally:
                    report['parent_snapshot'] = deepcopy(parent.last_snapshot_report)
                report['parent_snapshot_completions'] += 1
                saved = previous['learner']
                if not same(saved, payload['learner']):
                    raise ValueError('public parent restoration changed incoming full state')
                recipe = changed_recipe(saved['recipe'], rate)
                model, optimizer = parent.model, parent.optimizer
                optimizer.param_groups[0]['lr'] = rate
                report['kernel_construction_attempts'] += 1
                trainer = kernel.SharedStateKernel(model, optimizer, config=model.config, layout=recipe['layout'],
                    micro_batch_size=recipe['micro_batch_size'], objective_id=old.training.OBJECTIVE_ID, auxiliary_weight=.3)
                report['kernel_constructions'] += 1
                objective._restore_history(trainer, saved)
                transition = dict(schema=TRANSITION_SCHEMA, parent_archive_sha256=pin,
                    parent_transition=deepcopy(payload['transition']), parent_transition_sha256=transition_pin,
                    parent_recipe=deepcopy(saved['recipe']), parent_bridge_cost=deepcopy(payload['cost']),
                    start_cursor=saved['cursor'], start_weights_sha256=payload['weights_sha256'],
                    start_evidence=deepcopy(saved['evidence']), start_accounting=deepcopy(saved['accounting']),
                    source_learning_rate=SOURCE_RATE, learning_rate=rate, recipe=trainer.recipe, source_sha256=sources, scope=SCOPE)
                transition_pin = identity(transition)
                validate_transition(transition, transition_pin)
                expected = {**saved, 'recipe': recipe, 'optimizer': {**saved['optimizer'], 'param_groups': deepcopy(recipe['optimizer'])}}
                costs = {key: 0 if key.endswith('invocations') else 0. for key in COST_KEYS}
            else:
                transition = validate_snapshot(payload, transition_pin)
                saved = payload['learner']
                origin = old._identity(_origin(transition))
                if (old.current_runtime(device) != origin['runtime'] or torch.get_default_dtype() != torch.float32
                        or transition['parent_transition']['source_sha256'] != objective.source_hashes()):
                    raise ValueError('historical objective source/runtime changed')
                from experiments.sequence_student import SequenceConfig
                report['model_construction_attempts'] += 1
                model = old.student.build_shared_state_student(0, device=device, config=SequenceConfig(**saved['recipe']['config']))
                report['model_constructions'] += 1
                report['optimizer_construction_attempts'] += 1
                optimizer = torch.optim.AdamW(model.parameters(), lr=transition['learning_rate'])
                report['optimizer_constructions'] += 1
                recipe = saved['recipe']
                report['kernel_construction_attempts'] += 1
                trainer = kernel.SharedStateKernel(model, optimizer, config=model.config, layout=recipe['layout'],
                    micro_batch_size=recipe['micro_batch_size'], objective_id=old.training.OBJECTIVE_ID, auxiliary_weight=.3)
                report['kernel_constructions'] += 1
                if not same(trainer.recipe, recipe):
                    raise ValueError('native kernel rate recipe differs')
                _tensor_state(saved, trainer, transition)
                model.load_state_dict(saved['weights'], strict=True)
                optimizer.load_state_dict(cpu_copy(saved['optimizer']))
                objective._restore_history(trainer, saved)
                trainer._sync()
                expected, costs = saved, deepcopy(payload['cost'])
            validate_metadata(expected, transition)
            _tensor_state(expected, trainer, transition)
            report['verification_snapshot_attempts'] += 1
            actual = trainer.snapshot()
            report['verification_snapshot_completions'] += 1
            if not same(actual, expected) or old.checkpoint_digest(model) != payload['weights_sha256']:
                raise ValueError('weights/moments/individual steps/history or declared rate differs')
            if transition['source_sha256'] != sources or source_hashes() != sources:
                raise ValueError('rate source closure changed')
            value = object.__new__(cls)
            value._kernel, value._transition, value._sources = trainer, deepcopy(transition), sources
            value._transition_pin, value._cost, value._failed = transition_pin, costs, False
            value.last_step_report = value.last_snapshot_report = None
            value._guard()
            report.update(failed=False, exact_state_preserved_except_declared_lr=True, exact_state_restored=True, cursor=value.cursor)
        except BaseException as error:
            report['error'] = repr(error)
            try:
                if torch is not None and torch.device(device).type == 'cuda' and torch.cuda.is_initialized():
                    torch.cuda.synchronize(torch.device(device))
                report['queued_device_work_synchronized'] = True
            except BaseException as sync_error:
                report.update(failure_sync_error=repr(sync_error), partial_work_unknown=True)
            report.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu)
            error.rate_report = deepcopy(report)
            raise
        report.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu)
        value._record(operation, report)
        value.last_restore_report = deepcopy(report)
        return value

    def _record(self, kind, report):
        self._cost[kind+'_invocations'] += 1
        for clock in ('wall', 'cpu'):
            self._cost[kind+'_'+clock+'_seconds'] += report[clock+'_seconds']

    @property
    def model(self): return self._kernel.model
    @property
    def optimizer(self): return self._kernel.optimizer
    @property
    def cursor(self): return self._kernel.cursor
    @property
    def evidence(self): return self._kernel.evidence
    @property
    def recipe(self): return self._kernel.recipe
    @property
    def transition(self): return deepcopy(self._transition)
    @property
    def transition_sha256(self): return self._transition_pin
    @property
    def last_report(self): return deepcopy(self._kernel.last_report)
    @property
    def failed(self): return self._failed or self._kernel.failed
    @property
    def accounting(self):
        return dict(lifetime_kernel=self._kernel.accounting, rate_bridge_cost=deepcopy(self._cost),
            previous_objective_bridge_cost=deepcopy(self._transition['parent_bridge_cost']),
            previous_legacy_bridge_cost=deepcopy(self._transition['parent_transition']['legacy_bridge_cost']),
            scope='Historical costs retained; current outer bridge time includes nested operations. Do not sum twice.')

    def _guard(self):
        if self.failed: raise RuntimeError('poisoned rate continuation cannot retry')
        torch, old, *_ = objective._runtime()
        if (identity(self._transition) != self._transition_pin or source_hashes() != self._sources
                or old.current_runtime(self.model.tokens.weight.device) != _origin(self._transition)['runtime']):
            raise ValueError('rate identity/source/runtime changed')
        self._kernel._guard()

    def step(self, prepared, *, state_targets, deadline=None):
        started, cpu = time.monotonic(), time.process_time()
        report = dict(schema=SCHEMA, failed=True, kernel_called=False, cursor_before=self.cursor)
        self.last_step_report, failure = report, None
        try:
            self._guard(); report['kernel_called'] = True
            result = self._kernel.step(prepared, state_targets=state_targets, deadline=deadline)
            self._guard(); report['failed'] = False
            return result
        except BaseException as error:
            failure = error; self._failed = self._kernel.failed = True; report['error'] = repr(error)
            if report['kernel_called']: report['kernel_report'] = self.last_report
            raise
        finally:
            report.update(cursor=self.cursor, wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu)
            self._record('step', report)
            if failure is not None: failure.rate_report = deepcopy(report)

    def snapshot(self):
        started, cpu = time.monotonic(), time.process_time()
        report = dict(schema=SCHEMA, failed=True, cursor=self.cursor, kernel_snapshot_attempts=0, kernel_snapshot_completions=0)
        self.last_snapshot_report, failure = report, None
        try:
            self._guard(); report['kernel_snapshot_attempts'] += 1
            saved = self._kernel.snapshot(); report['kernel_snapshot_completions'] += 1
            validate_metadata(saved, self._transition)
            torch, old, *_ = objective._runtime()
            pin = old.checkpoint_digest(self.model)
            self._guard(); report['failed'] = False
        except BaseException as error:
            failure = error; self._failed = self._kernel.failed = True; report['error'] = repr(error)
            raise
        finally:
            report.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu)
            self._record('snapshot', report)
            if failure is not None: failure.rate_report = deepcopy(report)
        return dict(schema=SCHEMA, transition=self.transition, transition_sha256=self._transition_pin, learner=saved,
            weights_sha256=pin, lifetime_updates=self.cursor, rate_updates=self.cursor-self._transition['start_cursor'], cost=deepcopy(self._cost))
