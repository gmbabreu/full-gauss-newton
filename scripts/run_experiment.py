#!/usr/bin/env python3
"""Dependency-free preset resolution and single-experiment execution."""
import json
import os
from pathlib import Path
import re
import shlex
import sys
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
PRESETS = ROOT / 'scripts' / 'presets'
BATCH = 'train_dataset.huggingface_dataset.batch_size'
IDENTITY_FLAGS = {'experiment_id', 'wandb_run_id', 'resume_from', 'program'}


def choices():
    return sorted(p.stem for p in PRESETS.glob('*.json') if not p.stem.startswith('_'))


def read(name):
    return json.loads((PRESETS / (name + '.json')).read_text())


def resolve(preset, overrides):
    if preset not in choices():
        raise ValueError('Choose a preset: ' + ', '.join(choices()))
    spec = read(preset)
    flags = dict(read('_common'))
    if spec['route'] == 'sweep':
        flags.update(read('_gn'))
    flags.update(read('_adamw'))
    if spec.get('inner_adam'):
        flags.update(read('_inner_adam'))
    flags.update(spec['flags'])
    explicit = {}
    for arg in overrides:
        if not re.fullmatch(r'--[A-Za-z_][A-Za-z_0-9.]*=.*', arg, re.DOTALL):
            raise ValueError('Overrides must be --flag=value: ' + repr(arg))
        key, value = arg[2:].split('=', 1)
        if key in IDENTITY_FLAGS:
            raise ValueError(key + ' is controller-owned; each run has a fresh identity. '
                             'Use explicit checkpoint-loading flags for recovery.')
        explicit[key] = value
    if 'train_dataset_batch_size' in explicit and BATCH in explicit:
        if explicit['train_dataset_batch_size'] != explicit[BATCH]:
            raise ValueError('Conflicting explicit training batch sizes')
    flags.update(explicit)
    batch = explicit.get('train_dataset_batch_size', explicit.get(BATCH,
                         flags['train_dataset_batch_size']))
    flags['train_dataset_batch_size'] = batch
    flags[BATCH] = batch
    return dict(preset=preset, program=spec['program'], route=spec['route'],
                explicit_overrides=list(overrides), flags=flags)


def make_plan(preset, overrides, run_id=None):
    plan = resolve(preset, overrides)
    run_id = run_id or (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                        + '-' + uuid.uuid4().hex)
    plan['run_id'] = run_id
    plan['experiment_id'] = run_id + ('_0' if plan['route'] == 'sweep' else '')
    plan['flags']['experiment_id'] = plan['experiment_id']
    plan['flags']['wandb_run_id'] = ''
    plan['output_location'] = (str(plan['flags']['output_dir']).rstrip('/')
                               + '/' + plan['experiment_id'])
    return plan


def arguments(plan):
    return [f'--{key}={value}' for key, value in plan['flags'].items()]


def preview(plan):
    if plan['route'] == 'sweep':
        command = ['python', 'sweep_launcher.py', '--program=' + plan['program']]
    else:
        command = ['python', '-m', plan['program']]
    return shlex.join(command + arguments(plan))


def execute(plan):
    """Run the existing launcher's main, using its single-config input API.

    Its CLI parser treats ampersands in arbitrary strings as sweeps. Resolve
    flags here, supply one static configuration, and let main retain checkpoint
    lookup, _0 identity, and exec behavior. It appends BATCH itself, so omit that
    static flag to pass each final trainer flag exactly once.
    """
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ['PYTHONUNBUFFERED'] = '1'
    if plan['route'] == 'sweep':
        import sweep_launcher
        os.environ['SLURM_ARRAY_TASK_ID'] = '1'
        os.environ['SLURM_ARRAY_JOB_ID'] = plan['run_id']
        static = [arg for arg in arguments(plan) if not arg.startswith('--' + BATCH + '=')]
        sweep_launcher.parse_sweep_arguments = lambda args: (plan['program'], None, {}, static)
        sweep_launcher.main()
    else:
        os.execv(sys.executable, [sys.executable, '-m', plan['program']] + arguments(plan))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--execute':
        execute(json.loads(Path(sys.argv[2]).read_text()))
    else:
        try:
            if len(sys.argv) < 2:
                raise ValueError('Choose a preset: ' + ', '.join(choices()))
            print(preview(make_plan(sys.argv[1], sys.argv[2:])))
        except ValueError as error:
            sys.exit(str(error))
