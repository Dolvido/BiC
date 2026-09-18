"""A small offline menu for the existing retained learning session."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from experiments import home_learning as h

SCOPE = ('Restricted English training with verified lessons and local tutor choices. '
         'This is a research learner, not an open-ended English conversation system.')
STAGES = {
    'ready': 'Preparing the next curriculum',
    'prepared': 'Curriculum prepared; local tutor choice is next',
    'authored': 'Tutor choice recorded; checking lessons',
    'compiled': 'Lessons checked; training and evaluation are next',
    'trained': 'Training and evaluation recorded; reviewing retention',
}


def show_status(value, *, emit=print):
    learner = value['retained_research']
    emit(f"Retained learner: {learner['lifetime_updates']:,} updates; "
         f"{learner['completed_cycles']} completed cycles.")
    emit(f"Session: {value['status'].replace('_', ' ')}; saved stage: {learner['pending_stage']}.")
    if value.get('pending_invocation'):
        emit('An unfinished invocation is recorded; it may be running. Do not start a second session.')
    elif value['status'] != 'ready':
        emit('An invocation or ownership record needs attention. Do not start a second training session.')
    emit(SCOPE)


def status(profile=h.DEFAULT_PROFILE, *, emit=print):
    value = h.status(profile)
    show_status(value, emit=emit)
    return value


def _transport(profile):
    # h.status has already authenticated this profile and its entire source map.
    value = json.loads(h._native(h._path(profile)).read_bytes())
    name = 'brain_in_computer/curriculum_tutor.py'
    pin = value['current']['source_sha256'].get(name)
    if pin is None or h._sha(h.ROOT / name) != h._pin(pin):
        raise ValueError('The installed tutor transport differs from the saved session.')
    return importlib.import_module('brain_in_computer.curriculum_tutor')


def check(profile=h.DEFAULT_PROFILE, *, emit=print):
    value = h.status(profile)
    if value['status'] != 'ready':
        raise RuntimeError('The session has pending work or an ownership record. Use status; do not retry training automatically.')
    transport = _transport(profile)
    teacher = value['teacher']
    # Constructor validates a local-only model name; it performs no requests.
    transport.LocalTutor(model=teacher['model'], timeout=10, cache_size=0)
    deadline = time.monotonic() + 10
    try:
        models = transport._request_json('/api/tags', None, deadline).get('models')
        if type(models) is not list:
            raise ValueError('The installed Ollama model list is invalid.')
        matches = [m for m in models if type(m) is dict and
                   (m.get('name') == teacher['model'] or m.get('model') == teacher['model'])]
        if (len(matches) != 1 or matches[0].get('digest') != teacher['sha256']
                or transport._remote_metadata(matches[0])):
            raise ValueError('The exact saved local tutor is missing, changed, or remote. No model was downloaded.')
        details = transport._request_json('/api/show', {'model': teacher['model']}, deadline)
        if type(details.get('details')) is not dict or transport._remote_metadata(details):
            raise ValueError('The saved tutor has invalid or remote model metadata.')
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError('Cannot reach the local tutor. Open the installed Ollama app, then check again. No downloads are needed.') from error
    emit(f"Ready: saved learner and installed local tutor ({teacher['model']}) verified.")
    emit('Check read local model metadata only; it did not run the tutor or train BiC.')
    return value


def _progress(profile):
    """Best-effort display only; never used to authorize or retain learning."""
    try:
        path = h._native(h._path(profile))
        if path.stat().st_size > 2_000_000:
            return None
        pending = json.loads(path.read_bytes()).get('pending_invocation')
        if not pending:
            return None
        directory = h._native(h._path(pending['directory']) / 'campaign/execution/states')
        paths = sorted(directory.glob('[0-9][0-9][0-9][0-9].json'))
        if not paths or paths[-1].stat().st_size > 2_000_000:
            return None
        state = json.loads(paths[-1].read_bytes())
        if (state.get('schema') != h.OWNER_SCHEMA or state.get('stage') not in STAGES
                or type(state.get('cycle')) is not int):
            return None
        return pending['directory'], state['cycle'], state['stage']
    except (OSError, ValueError, KeyError, TypeError):
        return None  # Atomic publication may not yet be visible to the UI.


def _tail(path):
    with h._native(path).open('rb') as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 4096))
        return handle.read().decode('utf-8', errors='replace').strip()


def train(profile=h.DEFAULT_PROFILE, *, hours=1, emit=print):
    h.training_seconds(hours)
    check(profile, emit=emit)
    python = h.ROOT / '.venv/Scripts/python.exe'
    if not python.is_file():
        raise RuntimeError('The project Python environment is missing. Use the existing project setup instructions.')
    directory = h.ROOT / 'runs/offline-home-local'
    h._native(directory).mkdir(parents=True, exist_ok=True)
    log = directory / ('session-' + uuid.uuid4().hex + '.log')
    command = [str(python), '-B', '-m', 'experiments.home_learning', 'resume',
               '--profile', str(h._path(profile)), '--training-hours', str(hours)]
    emit(f'Starting up to {hours:g} hours of the existing local learning loop.')
    emit('Setup and final saving add time. Retention checks decide whether a candidate is kept.')
    emit(f'Log: {log}')
    with h._native(log).open('xb') as handle:
        child = subprocess.Popen(command, cwd=h.ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        last = None
        try:
            while child.poll() is None:
                current = _progress(profile)
                if current is not None and current != last:
                    emit(f'Cycle {current[1]}: {STAGES[current[2]]}.')
                    last = current
                time.sleep(2)
        except KeyboardInterrupt:
            emit(f'The menu was interrupted; learning process {child.pid} may still be running.')
            emit(f'Inspect status and this log before restarting: {log}')
            return 130
    if child.returncode != 0:
        emit('The learning session stopped with an error. Its saved work needs review before another start.')
        tail = _tail(log)
        if tail:
            emit(tail)
        emit(f'Full log: {log}')
        return child.returncode or 1
    emit('Learning session finished.')
    status(profile, emit=emit)
    emit(f'Full log: {log}')
    return 0


def menu(profile=h.DEFAULT_PROFILE, *, emit=print, read=input):
    emit('BiC offline learning')
    emit(SCOPE)
    while True:
        emit('\n1 Train   2 Status   3 Check setup   q Exit')
        try:
            choice = read('Choose: ').strip().lower()
            if choice == 'q':
                return 0
            if choice == '1':
                raw = read('Training hours [1]: ').strip()
                hours = float(raw) if raw else 1.0
                h.training_seconds(hours)
                result = train(profile, hours=hours, emit=emit)
                if result == 130:
                    return result
            elif choice == '2':
                status(profile, emit=emit)
            elif choice == '3':
                check(profile, emit=emit)
            else:
                emit('Choose 1, 2, 3, or q.')
        except (EOFError, KeyboardInterrupt):
            return 0
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            emit(f'Unable to continue: {error}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('menu', 'check', 'status', 'train'))
    parser.add_argument('--profile', default=str(h.DEFAULT_PROFILE))
    parser.add_argument('--hours', type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        if args.command == 'menu':
            return menu(args.profile)
        if args.command == 'train':
            return train(args.profile, hours=args.hours)
        {'check': check, 'status': status}[args.command](args.profile)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'Unable to continue: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
