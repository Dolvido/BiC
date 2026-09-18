"""Reproducible pilot: learn a feedback strategy, then adapt with frozen weights.

Run from the repository root. Test evaluation follows development selection;
all conditions and raw episode summaries are saved in a new output directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn import functional as F

from brain_in_computer.adaptation import RuleWorld, CandidateLearner, encode_observations, split_layouts
from brain_in_computer.adaptive_agent import (
    AdaptiveSession, build_model, export_state, import_state, load_agent,
    model_digest, parameter_count, save_agent,
)


SEEDS = (11, 23, 37, 51, 71)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def make_worlds(count, seed_start, layouts, reversal_range):
    return [RuleWorld(seed=seed_start+i, horizon=40,
                      reversal_step=random.Random(seed_start+i+9000000).randint(*reversal_range),
                      allowed_layouts=layouts) for i in range(count)]


def training_data(count, seed_start, layouts):
    """Causal scripted-strategy demonstrations, never hidden target labels."""
    worlds = make_worlds(count, seed_start, layouts, (12, 28))
    teachers = [CandidateLearner() for _ in worlds]
    choices = [random.Random(seed_start+i+8000000) for i in range(count)]
    channels, targets = defaultdict(list), []
    for _ in range(40):
        observations = [world.observe() for world in worlds]
        probabilities = [teacher.probabilities(obs) for teacher, obs in zip(teachers, observations)]
        for key, value in encode_observations(observations).items():
            channels[key].append(value)
        targets.append(torch.tensor(probabilities, dtype=torch.float32)[:, None])
        for world, obs, probs, rng in zip(worlds, observations, probabilities, choices):
            candidates = [i for i, probability in enumerate(probs) if probability > 0]
            category = rng.randrange(4) if rng.random() < .25 else rng.choice(candidates)
            world.step(obs['layout'].index(category))
    return {key: torch.cat(values, dim=1) for key, values in channels.items()}, torch.cat(targets, dim=1)


def episode_metrics(rewards, categories, reversal):
    phases = (rewards[:reversal], rewards[reversal:])
    delays, failures, repeated = [], [], []
    offset = 0
    for phase in phases:
        confirmation = next((i+1 for i in range(2, len(phase)) if sum(phase[i-2:i+1]) == 3), None)
        delays.append(confirmation if confirmation is not None else len(phase)+1)
        failures.append(confirmation is None)
        failed_categories, repeats = set(), 0
        for reward, category in zip(phase, categories[offset:offset+len(phase)]):
            if not reward:
                repeats += category in failed_categories
                failed_categories.add(category)
        repeated.append(repeats)
        offset += len(phase)
    return {'reward': statistics.mean(rewards),
            'late_reward': statistics.mean(phases[0][-8:] + phases[1][-8:]),
            'initial_late_reward': statistics.mean(phases[0][-8:]),
            'reversal_late_reward': statistics.mean(phases[1][-8:]),
            'post_reversal_reward': statistics.mean(phases[1]),
            'initial_first4_reward': statistics.mean(phases[0][:4]),
            'reversal_first4_reward': statistics.mean(phases[1][:4]),
            'initial_recovery_trials': delays[0], 'reversal_recovery_trials': delays[1],
            'initial_recovery_failed': int(failures[0]), 'reversal_recovery_failed': int(failures[1]),
            'initial_repeated_errors': repeated[0], 'reversal_repeated_errors': repeated[1],
            'initial_repeated_error_rate': repeated[0]/len(phases[0]),
            'reversal_repeated_error_rate': repeated[1]/len(phases[1])}


@torch.inference_mode()
def evaluate(model, *, count, seed_start, layouts, reversal_range=(8, 32), variant='persistent', traces=False):
    worlds = make_worlds(count, seed_start, layouts, reversal_range)
    if model is not None:
        model.eval()
        device = next(model.parameters()).device
    state = None
    teachers = [CandidateLearner() for _ in worlds]
    choices = [random.Random(seed_start+i+7000000) for i in range(count)]
    reward_rows, category_rows = [[] for _ in worlds], [[] for _ in worlds]
    trace_rows = [[] for _ in worlds]
    reversals = [random.Random(seed_start+i+9000000).randint(*reversal_range) for i in range(count)]
    for step in range(40):
        observations = [world.observe() for world in worlds]
        if variant == 'candidate':
            categories = [teacher.choose_category(obs) for teacher, obs in zip(teachers, observations)]
        elif variant == 'random':
            categories = [rng.randrange(4) for rng in choices]
        else:
            encoded = encode_observations(observations, device=device)
            if variant == 'no_feedback':
                encoded['feedback'][:, :, 0].zero_()
            outputs, state = model.forward_with_state(encoded, None if variant == 'reset_state' else state)
            categories = outputs['logits'][:, -1].argmax(-1).cpu().tolist()
        for i, (world, obs, category) in enumerate(zip(worlds, observations, categories)):
            cell = obs['layout'].index(category)
            _, reward = world.step(cell)
            reward_rows[i].append(reward)
            category_rows[i].append(category)
            if traces:
                trace_rows[i].append({'step': step, 'observation': obs, 'category': category,
                                      'cell': cell, 'reward': reward})
    rows = [episode_metrics(rewards, categories, reversal)
            for rewards, categories, reversal in zip(reward_rows, category_rows, reversals)]
    summary = {key: statistics.mean(row[key] for row in rows) for key in rows[0]}
    result = {'summary': summary, 'episodes': rows, 'count': count,
              'seed_start': seed_start, 'variant': variant,
              'trajectories': [{'seed': seed_start+i, 'reversal_step': reversal,
                                'rewards': rewards, 'categories': categories}
                               for i, (rewards, categories, reversal) in enumerate(zip(reward_rows, category_rows, reversals))]}
    if traces:
        result['traces'] = trace_rows
    return result


@torch.inference_mode()
def check_restart(model, layouts, seed=3000001):
    """Round-trip environment and neural state via the actual persistence formats."""
    world = RuleWorld(seed=seed, horizon=40, reversal_step=22, allowed_layouts=layouts)
    session = AdaptiveSession(model)
    for _ in range(13):
        world.step(session.act(world.observe()))
    world_copy = RuleWorld.from_snapshot(json.loads(json.dumps(world.snapshot())))
    buffer = io.BytesIO()
    torch.save(session.snapshot(), buffer)
    buffer.seek(0)
    resumed = AdaptiveSession(model)
    resumed.restore(torch.load(buffer, weights_only=True))
    actions_equal = True
    while not world.done:
        left = session.act(world.observe())
        right = resumed.act(world_copy.observe())
        first, reward1 = world.step(left)
        second, reward2 = world_copy.step(right)
        actions_equal &= left == right and reward1 == reward2 and first == second
    a, b = export_state(model, session.state), export_state(model, resumed.state)
    def equal(x, y):
        if isinstance(x, torch.Tensor):
            return torch.equal(x, y)
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
        return x == y
    return {'exact_actions_rewards_observations': actions_equal,
            'exact_final_neural_state': equal(a, b), 'restart_after_steps': 13}


def train_one(kind, seed, index, args, data, targets, train_layouts, folder):
    torch.manual_seed(seed)
    model = build_model(kind, args.hidden_size).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.0001)
    sampler = torch.Generator().manual_seed(seed+6000000)
    history, best, best_update = [], -1., 0
    checkpoint = folder / f'{kind}-{seed}.pt'
    started = time.perf_counter()
    for update in range(1, args.updates+1):
        model.train()
        selection = torch.randint(len(targets), (args.batch_size,), generator=sampler)
        observations = {key: value[selection].to(args.device) for key, value in data.items()}
        target = targets[selection].to(args.device)
        output = model(observations)['logits']
        loss = -(target * F.log_softmax(output, dim=-1)).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if update % args.check_every == 0 or update == args.updates:
            development = evaluate(model, count=args.dev_episodes, seed_start=1000000,
                                   layouts=train_layouts, reversal_range=(10,30))['summary']
            score = development['late_reward']
            record = {'update': update, 'loss': float(loss.detach()), 'development': development,
                      'elapsed_seconds': time.perf_counter()-started}
            history.append(record)
            if score > best:
                best, best_update = score, update
                save_agent(model, checkpoint, seed=seed, hidden_size=args.hidden_size,
                           metadata={'selected_update': update, 'development': development})
            print(json.dumps({'kind': kind, 'seed': seed, **record}), flush=True)
    write_json(folder/f'{kind}-{seed}-training.json', history)
    return {'kind': kind, 'seed': seed, 'parameters': parameter_count(model),
            'selected_update': best_update, 'development_late_reward': best,
            'training_seconds': time.perf_counter()-started, 'checkpoint': checkpoint.name}


def run(args):
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    train_layouts, test_layouts = split_layouts(7001)
    protocol = {key: value for key, value in vars(args).items() if key != 'func'}
    protocol.update({'schema': 'bic-adaptation-pilot-1', 'train_layouts': train_layouts,
                     'test_layouts': test_layouts, 'evaluation_weights_frozen': True,
                     'training_objective': 'causal scripted-candidate soft-label imitation',
                     'torch_version': str(torch.__version__), 'sealed_test_seed_start': 2000000,
                     'reversal_rng_seed_offset': 9000000, 'teacher_action_rng_seed_offset': 8000000,
                     'random_policy_rng_seed_offset': 7000000, 'batch_rng_seed_offset': 6000000,
                     'optimizer': {'name': 'AdamW', 'lr': .003, 'weight_decay': .0001, 'clip_norm': 1.},
                     'feedback_mask': 'reward component only; previous choice and presence unchanged'})
    write_json(folder/'protocol.json', protocol)
    # Capture exactly the sources used, independent of later workspace changes.
    sources = ['brain_in_computer/model.py', 'brain_in_computer/regions.py',
               'brain_in_computer/adaptation.py', 'brain_in_computer/adaptive_agent.py',
               'experiments/learn_adaptation.py', 'docs/ADAPTATION_PROTOCOL.md']
    hashes = {}
    for name in sources:
        payload = Path(name).read_bytes()
        destination = folder/'source-snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    write_json(folder/'source-hashes.json', hashes)
    training = []
    for index, seed in enumerate(args.seeds):
        data, targets = training_data(args.train_episodes, index*10000, train_layouts)
        digest = hashlib.sha256()
        for key, value in sorted({**data, 'targets': targets}.items()):
            digest.update(key.encode('ascii'))
            digest.update(value.contiguous().numpy().tobytes())
        write_json(folder/f'corpus-{seed}.json', {'sha256': digest.hexdigest(),
                   'environment_seed_start': index*10000, 'episodes': args.train_episodes})
        for kind in ('bic', 'gru'):
            training.append(train_one(kind, seed, index, args, data, targets, train_layouts, folder))
        write_json(folder/'training-summary.json', training)
    if args.development_only:
        print(json.dumps({'development_only_complete': str(folder), 'sealed_test_run': False}), flush=True)
        return
    # All checkpoint choices are complete before any test outcome is examined.
    results, restarts = {}, {}
    for record in training:
        key = f"{record['kind']}-{record['seed']}"
        model, _ = load_agent(folder/record['checkpoint'], args.device)
        before = model_digest(model)
        results[key] = {}
        for variant in ('persistent', 'reset_state', 'no_feedback'):
            result = evaluate(model, count=args.test_episodes, seed_start=2000000,
                              layouts=test_layouts, variant=variant)
            write_json(folder/f'{key}-{variant}-test.json', result)
            results[key][variant] = result['summary']
        restarts[key] = check_restart(model, test_layouts)
        restarts[key]['weights_unchanged'] = before == model_digest(model)
        print(json.dumps({'test_complete': key, 'results': results[key], 'restart': restarts[key]}), flush=True)
    baselines = {}
    for variant in ('candidate', 'random'):
        result = evaluate(None, count=args.test_episodes, seed_start=2000000,
                          layouts=test_layouts, variant=variant)
        baselines[variant] = result['summary']
        write_json(folder/f'{variant}-test.json', result)
    aggregate = {}
    for kind in ('bic', 'gru'):
        aggregate[kind] = {}
        for variant in ('persistent', 'reset_state', 'no_feedback'):
            rows = [value[variant] for key, value in results.items() if key.startswith(kind+'-')]
            aggregate[kind][variant] = {metric: {'mean': statistics.mean(row[metric] for row in rows),
                'sd': statistics.stdev(row[metric] for row in rows) if len(rows)>1 else None}
                for metric in rows[0]}
    bic = aggregate['bic']
    paired = {}
    for comparator in ('reset_state', 'no_feedback', 'gru'):
        differences = []
        for seed in args.seeds:
            left = results[f'bic-{seed}']['persistent']
            right = results[f'gru-{seed}']['persistent'] if comparator == 'gru' else results[f'bic-{seed}'][comparator]
            differences.append({key: left[key]-right[key] for key in left})
        paired[comparator] = {metric: {'mean': statistics.mean(row[metric] for row in differences),
                              'sd': statistics.stdev(row[metric] for row in differences) if len(differences)>1 else None,
                              'per_seed': [row[metric] for row in differences]} for metric in differences[0]}
    gates = {'late_reward_at_least_075': bic['persistent']['late_reward']['mean'] >= .75,
             'feedback_gain_at_least_015': bic['persistent']['late_reward']['mean'] - bic['no_feedback']['late_reward']['mean'] >= .15,
             'all_restart_and_weight_checks': all(all(v for k,v in r.items() if k != 'restart_after_steps') for r in restarts.values())}
    report = {'training': training, 'aggregate': aggregate, 'baselines': baselines,
              'per_seed': results, 'paired_bic_minus_comparator': paired, 'restart': restarts, 'gates': gates,
              'limitations': ['symbolic categories and explicit category-to-cell mapping',
                'strategy imitation, not reward-only reinforcement learning',
                'same four categories; withheld layouts only',
                'fast recurrent adaptation; no online weight updates or consolidation',
                'new experimental checkpoint; no combined old/new skill retention claim']}
    write_json(folder/'report.json', report)
    print(json.dumps({'complete': str(folder), 'gates': gates}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--updates', type=int, default=400)
    parser.add_argument('--train-episodes', type=int, default=2048)
    parser.add_argument('--dev-episodes', type=int, default=128)
    parser.add_argument('--test-episodes', type=int, default=256)
    parser.add_argument('--check-every', type=int, default=100)
    parser.add_argument('--hidden-size', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--development-only', action='store_true', help='Never run sealed tests in a smoke run')
    args = parser.parse_args()
    if any(getattr(args, key) < 1 for key in ('updates', 'train_episodes', 'dev_episodes',
            'test_episodes', 'check_every', 'hidden_size', 'batch_size', 'threads')):
        parser.error('counts and sizes must be positive')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be unique')
    run(args)


if __name__ == '__main__':
    main()
