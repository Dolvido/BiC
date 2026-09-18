"""Run once after development freezes; preserve all conditions and failures."""

import argparse
import hashlib
import math
from pathlib import Path
import platform
import time

import torch

from brain_in_computer.computer_use.training import load_computer_checkpoint, evaluate_computer
from brain_in_computer.training import atomic_json


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    center = (p + z*z/(2*count)) / (1+z*z/count)
    radius = z * math.sqrt(p*(1-p)/count+z*z/(4*count*count)) / (1+z*z/count)
    return [center-radius, center+radius]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/computer/checkpoint.pt')
    parser.add_argument('--output', default='runs/computer/final_evaluation.json')
    parser.add_argument('--episodes-per-task', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = load_computer_checkpoint(args.checkpoint)
    conditions = {
        'familiar_wording': {},
        'unseen_wording': {'split': 'test'},
        'shifted_layout': {'shift': True},
        'blank_pixels': {'blank_pixels': True},
        'blank_prompt': {'blank_prompt': True},
        'hippocampus_lesion': {'ablate': ('hippocampus',)},
    }
    result = {'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                          'platform': platform.platform(), 'device': 'cpu', 'threads': 1},
              'single_training_seed': True, 'independent_training_replications': 1,
              'interval_note': 'Wilson intervals describe sampled episodes, not training-seed variability; finite states may repeat.',
              'conditions': {}}
    for name, options in conditions.items():
        measured = evaluate_computer(model, seed=900031, episodes_per_task=args.episodes_per_task, **options)
        for task in measured['tasks'].values():
            task['joint_success_wilson_95'] = wilson(task['joint_success_count'], task['episodes'])
        total = sum(task['episodes'] for task in measured['tasks'].values())
        successes = sum(task['joint_success_count'] for task in measured['tasks'].values())
        measured['joint_success_wilson_95'] = wilson(successes, total)
        result['conditions'][name] = measured
        atomic_json(args.output, result)
        print(f"{name}: action={measured['macro_action_success']:.3f} reply={measured['macro_reply_exact']:.3f} joint={measured['macro_joint_success']:.3f}", flush=True)
    return result


if __name__ == '__main__':
    main()
