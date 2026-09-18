"""Small home front end for the existing verified, finite-curriculum campaign.

Status never imports a learner. Resume preserves the v3 campaign controller and
its pending work; it does not adopt unrelated research or released checkpoints.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / 'runs/home-learning-local/session.json'
SCHEMA = 'bic-home-learning-session-v1'
OWNER_SCHEMA = 'bic-continuous-tutor-campaign-v3'
OWNER_MODULE = 'experiments.continuous_tutor_campaign_v3'
SCOPE = ('Retained research learner; restricted English and verified finite curricula. '
         'Selection is engineered; tutor advantage and general intelligence are unproven. '
         'Released desktop weights and unrelated research candidates are unchanged.')


def _native(path):
    value = str(Path(path).resolve())
    return Path('\\\\?\\' + value if os.name == 'nt' and not value.startswith('\\\\?\\') else value)


def _path(path):
    result = Path(path)
    if not result.is_absolute(): result = ROOT / result
    result = result.resolve()
    if not result.is_relative_to(ROOT): raise ValueError('home artifacts must stay within the project')
    return result


def _relative(path): return _path(path).relative_to(ROOT).as_posix()
def _utc(): return datetime.now(timezone.utc).isoformat()
def _encoded(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _sha(path):
    digest = hashlib.sha256()
    with _native(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def _pin(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('explicit SHA256 identity required')
    return value


def _record(path): return dict(path=_relative(path), sha256=_sha(path))


def _read(record):
    path = _path(record['path']); raw = _native(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != _pin(record['sha256']): raise ValueError('pinned home artifact changed')
    return json.loads(raw)


def _publish(path, value, *, replace=False):
    path = _path(path); _native(path.parent).mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp') if replace else path
    with _native(temporary).open('xb') as handle:
        handle.write(_encoded(value)); handle.flush(); os.fsync(handle.fileno())
    if replace: os.replace(_native(temporary), _native(path))
    return _record(path)


def _campaign(directory, summary_sha256):
    """Authenticate completed JSON/state and opaque checkpoint bytes only."""
    directory = _path(directory)
    summary_record = dict(path=_relative(directory / 'execution/summary.json'), sha256=_pin(summary_sha256))
    summary = _read(summary_record)
    launch_record = dict(path=_relative(directory / 'launch.json'), sha256=_pin(summary['launch_sha256']))
    launch = _read(launch_record); state_record = summary['last_state']; state = _read(state_record)
    if (summary.get('schema') != OWNER_SCHEMA or launch.get('schema') != OWNER_SCHEMA
            or state.get('schema') != OWNER_SCHEMA or summary.get('status') != 'completed'
            or summary.get('partial_work_unknown') is not False
            or state.get('parent') != summary.get('parent')
            or state.get('completed_cycles') != summary.get('completed_cycles')
            or state.get('physical_totals') != summary.get('physical_totals')
            or state.get('stage') != summary.get('final_stage')
            or state.get('stage') not in ('ready', 'prepared', 'authored', 'compiled', 'trained')
            or type(state.get('parent', {}).get('lifetime_updates')) is not int):
        raise ValueError('completed known-work v3 campaign with matching committed state required')
    parent = state['parent']; checkpoint = parent['checkpoint']
    work = state['physical_totals']
    if (type(work) is not dict or set(work) != {'worker_updates','teacher_chat_attempts','teacher_chat_completions'}
            or any(type(v) is not int or v < 0 for v in work.values())
            or work['teacher_chat_completions'] > work['teacher_chat_attempts']):
        raise ValueError('exact nonnegative committed campaign counters required')
    if _sha(_path(checkpoint['path'])) != _pin(checkpoint['sha256']): raise ValueError('retained checkpoint changed')
    _pin(parent['weights_sha256'])
    if (set(launch.get('teacher', {})) != {'model', 'sha256'} or not launch['teacher']['model']
            or type(launch.get('source_sha256')) is not dict or not launch['source_sha256']):
        raise ValueError('exact frozen sources and installed teacher identity required')
    _pin(launch['teacher']['sha256'])
    for name, pin in launch['source_sha256'].items():
        if _sha(_path(name)) != _pin(pin): raise ValueError('frozen campaign source changed')
    evaluation = launch['evaluation']
    _read(dict(path=_relative(_path(evaluation['directory']) / 'manifest.json'), sha256=evaluation['manifest_sha256']))
    return dict(directory=_relative(directory), summary=summary_record, launch=launch_record,
        state=deepcopy(state_record), checkpoint=deepcopy(checkpoint), weights_sha256=parent['weights_sha256'],
        lifetime_updates=parent['lifetime_updates'], completed_cycles=state['completed_cycles'],
        pending_stage=state['stage'], teacher=deepcopy(launch['teacher']), evaluation=deepcopy(evaluation),
        seed=launch['seed'], physical_totals=deepcopy(state['physical_totals']),
        source_sha256=deepcopy(launch['source_sha256']))


def init_profile(profile=DEFAULT_PROFILE, *, campaign_directory, summary_sha256):
    current = _campaign(campaign_directory, summary_sha256)
    initial = _read(current['launch'])['initial_state']['parent']
    state = _read(current['state'])
    baseline = (dict(checkpoint=deepcopy(initial['checkpoint']), lifetime_updates=initial['lifetime_updates'])
                if initial['checkpoint'] == state['original_parent_checkpoint'] else None)
    value = dict(schema=SCHEMA, created_utc=_utc(), launcher_sha256=_sha(Path(__file__)),
        current=current, campaign_start=baseline, pending_invocation=None, history=[], scope=SCOPE,
        budget_scope='training/controller wall time; CPU freeze/setup is separate and reported')
    _publish(profile, value)
    return status(profile)


def _profile(profile):
    value = json.loads(_native(_path(profile)).read_bytes())
    if (value.get('schema') != SCHEMA or value.get('launcher_sha256') != _sha(Path(__file__))
            or type(value.get('history')) is not list): raise ValueError('unchanged home profile/launcher required')
    current = value['current']
    if _campaign(current['directory'], current['summary']['sha256']) != current:
        raise ValueError('home profile differs from its authenticated retained campaign')
    return value


def status(profile=DEFAULT_PROFILE):
    """Read-only byte/JSON checks; no process, service, archive decode or ML calls."""
    value = _profile(profile); current = value['current']; lock = _path(profile).with_suffix('.lock')
    return dict(schema=SCHEMA, status='attention_required' if value['pending_invocation'] or _native(lock).exists() else 'ready',
        retained_research=dict(lifetime_updates=current['lifetime_updates'], checkpoint=current['checkpoint'],
            completed_cycles=current['completed_cycles'], pending_stage=current['pending_stage']),
        teacher=current['teacher'], evaluation=current['evaluation'], campaign=current['directory'],
        committed_campaign_work=current['physical_totals'], campaign_start=value['campaign_start'],
        retained_lineage_updates=(current['lifetime_updates']-value['campaign_start']['lifetime_updates']
                                  if value['campaign_start'] is not None else None),
        accounting_scope='Committed campaign totals include inherited attempts; no discarded-work estimate is inferred.',
        pending_invocation=value['pending_invocation'], last_invocation=value['history'][-1] if value['history'] else None,
        session_lock=json.loads(_native(lock).read_bytes()) if _native(lock).exists() else None,
        lock_scope='Recorded ownership only; status does not query whether a process is alive.',
        budget_scope=value['budget_scope'], scope=SCOPE)


def windows_process_inventory():
    """Read authoritative Windows process metadata; never terminate a process."""
    if os.name != 'nt': raise RuntimeError('authoritative Windows process inventory required')
    script = ('$ErrorActionPreference="Stop"; Get-CimInstance Win32_Process | '
              'Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine,'
              '@{Name="Created";Expression={if($null -ne $_.CreationDate){$_.CreationDate.ToUniversalTime().ToString("o")}else{$null}}} | '
              'ConvertTo-Json -Depth 3 -Compress')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
        capture_output=True, text=True, timeout=20, check=True,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    rows = json.loads(result.stdout)
    if type(rows) is not list or not rows: raise RuntimeError('complete process inventory unavailable')
    return rows


def _idle(inventory):
    rows = inventory()
    if type(rows) is not list or not rows: raise RuntimeError('process state is unknown')
    if any(type(r) is not dict or type(r.get('ProcessId')) is not int
           or type(r.get('ParentProcessId')) is not int or not r.get('Name') for r in rows):
        raise RuntimeError('incomplete process inventory')
    by_pid = {r['ProcessId']: r for r in rows}
    if len(by_pid) != len(rows): raise RuntimeError('ambiguous process inventory')
    own = [r for r in rows if r.get('ProcessId') == os.getpid()]
    if len(own) != 1 or not own[0].get('Created'): raise RuntimeError('current process identity unavailable')
    # Windows venv bootstrap may remain alive as this process's Python parent.
    # Exclude the exact observed ancestor chain, never unrelated Python workers.
    ancestors = {os.getpid()}; parent = own[0]['ParentProcessId']
    while parent in by_pid and parent != 0:
        if parent in ancestors: raise RuntimeError('cyclic process ancestry')
        ancestors.add(parent); parent = by_pid[parent]['ParentProcessId']
    project = str(ROOT).lower().replace('/', '\\')
    for record in rows:
        if record['ProcessId'] in ancestors: continue
        name = record['Name'].lower(); command = record.get('CommandLine'); executable = record.get('ExecutablePath')
        python = name.startswith(('python', 'pypy', 'torchrun'))
        if python and (not command or not executable): raise RuntimeError('uninspectable Python process; refuse concurrent work')
        text = ((command or '') + ' ' + (executable or '')).lower().replace('/', '\\')
        if python and (project in text or '-m experiments.' in text or 'brain_in_computer' in text):
            raise RuntimeError('another project Python/learner process is running: ' + str(record['ProcessId']))
    return dict(pid=os.getpid(), process_created=own[0]['Created'], executable=sys.executable,
        observed_ancestor_pids=sorted(ancestors-{os.getpid()}))


@contextmanager
def _locks(profile, owner):
    paths = [_path('runs/home-learning-local/project.lock'), _path(profile).with_suffix('.lock')]
    acquired = []
    try:
        for path in paths:
            record = _publish(path, dict(**owner, lock_id=uuid.uuid4().hex, created_utc=_utc()))
            acquired.append(record)
        yield
    finally:
        for record in reversed(acquired):
            path = _path(record['path'])
            if _native(path).exists() and _sha(path) == record['sha256']: _native(path).unlink()


def training_seconds(hours):
    if type(hours) not in (int, float) or not math.isfinite(hours) or not 0 < hours <= 24:
        raise ValueError('finite positive training hours required')
    seconds = math.floor(hours * 3600)
    if not 120 <= seconds <= 86400: raise ValueError('controller training budget must be120..86400seconds')
    return seconds


def resume(profile=DEFAULT_PROFILE, *, training_hours, process_inventory=windows_process_inventory, controller=None):
    """Run the existing owner once; persist failures and never retry implicitly."""
    seconds = training_seconds(training_hours); profile = _path(profile)
    process = _idle(process_inventory)
    with _locks(profile, process):
        value = _profile(profile)
        if value['pending_invocation'] is not None: raise RuntimeError('prior invocation unresolved; inspect receipts, no automatic retry')
        _idle(process_inventory)
        prior = deepcopy(value['current']); directory = profile.parent / 'invocations' / uuid.uuid4().hex
        _native(directory).mkdir(parents=True, exist_ok=False)
        started, cpu = time.monotonic(), time.process_time()
        intent = dict(schema=SCHEMA, status='started', created_utc=_utc(), process=process,
            previous_campaign=prior, training_budget_seconds=seconds, budget_scope=value['budget_scope'],
            controller=OWNER_MODULE, automatic_retry=False, automatic_promotion=False)
        intent_record = _publish(directory / 'intent.json', intent)
        value['pending_invocation'] = dict(directory=_relative(directory), intent=intent_record)
        _publish(profile, value, replace=True)
        receipt = dict(schema=SCHEMA, status='failed', intent=intent_record, controller_run_started=False,
            training_budget_seconds=seconds, setup_wall_seconds=None, run_wall_seconds=None)
        try:
            active = controller if controller is not None else importlib.import_module(OWNER_MODULE)
            output = directory / 'campaign'; tick = time.monotonic()
            launch_pin = active.freeze(output, parent_directory=_path(prior['directory']),
                parent_summary_sha256=prior['summary']['sha256'], evaluation_directory=_path(prior['evaluation']['directory']),
                evaluation_manifest_sha256=prior['evaluation']['manifest_sha256'], teacher=deepcopy(prior['teacher']),
                seed=prior['seed'], budget_seconds=seconds, resume=True)
            receipt['freeze_wall_seconds'] = time.monotonic() - tick
            launch_record = dict(path=_relative(output / 'launch.json'), sha256=_pin(launch_pin)); launch = _read(launch_record)
            if (launch.get('schema') != OWNER_SCHEMA or launch.get('resumed') is not True
                    or launch.get('initial_state') != _read(prior['state'])
                    or launch.get('teacher') != prior['teacher'] or launch.get('evaluation') != prior['evaluation']
                    or launch.get('seed') != prior['seed'] or launch.get('budget_seconds') != seconds
                    or launch.get('source_sha256') != prior['source_sha256']):
                raise ValueError('owner freeze changed the authenticated resume contract')
            receipt['launch'] = launch_record; _idle(process_inventory)
            receipt['setup_wall_seconds'] = time.monotonic() - started
            receipt['controller_run_started'] = True; tick = time.monotonic()
            try: active.run(output, launch_sha256=launch_pin)
            finally: receipt['run_wall_seconds'] = time.monotonic() - tick
            summary_record = _record(output / 'execution/summary.json'); receipt['summary'] = summary_record
            current = _campaign(output, summary_record['sha256'])
            if current['launch'] != launch_record: raise ValueError('completed owner belongs to a different launch')
            receipt.update(status='completed', retained_before=prior['lifetime_updates'], retained_after=current['lifetime_updates'])
            value['current'] = current; value['pending_invocation'] = None
        except BaseException as error:
            receipt.update(error_type=type(error).__name__, error=str(error), requires_explicit_review=True)
            output_summary = directory / 'campaign/execution/summary.json'
            if _native(output_summary).exists(): receipt['summary'] = _record(output_summary)
            raise
        finally:
            if receipt['setup_wall_seconds'] is None: receipt['setup_wall_seconds'] = time.monotonic()-started
            receipt.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu, ended_utc=_utc())
            record = _publish(directory / 'receipt.json', receipt)
            value['history'].append(record)
            _publish(profile, value, replace=True)
    return status(profile)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('init', 'status', 'resume'))
    parser.add_argument('--profile', default=str(DEFAULT_PROFILE)); parser.add_argument('--json', action='store_true')
    parser.add_argument('--campaign-directory'); parser.add_argument('--summary-sha256')
    parser.add_argument('--training-hours', type=float)
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            if not args.campaign_directory or not args.summary_sha256: raise ValueError('init requires campaign directory and summary SHA256')
            result = init_profile(args.profile, campaign_directory=args.campaign_directory, summary_sha256=args.summary_sha256)
        elif args.command == 'resume': result = resume(args.profile, training_hours=args.training_hours)
        else: result = status(args.profile)
        if args.json: print(_encoded(result).decode().strip())
        else:
            learner = result['retained_research']
            print(f"Home learning: {result['status']}. Retained research learner: update {learner['lifetime_updates']}; "
                  f"{learner['completed_cycles']} completed cycles; pending stage: {learner['pending_stage']}.")
            print(result['budget_scope'] + '.'); print(result['scope'])
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr); return 1


if __name__ == '__main__': raise SystemExit(main())
