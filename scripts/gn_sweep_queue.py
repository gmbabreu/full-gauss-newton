#!/usr/bin/env python3
"""Sequential GN-Adam screening. All token budgets count SOLVE data only."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shlex
import sys
import time

from run_experiment import make_plan
from tpu_control import Controller

# id, batch, peak inner LR, microbatch count
TRIALS = [(i, b, lr, micro) for i, (b, lr, micro) in enumerate([
    (720, .001, 1), (720, .003, 1),
    (256, .001, 1), (256, .003, 1),
    (1024, .001, 2), (1024, .003, 2),
    (2048, .001, 4), (2048, .003, 4),
    (64, .001, 1), (64, .003, 1),
])]


def build_plan(trial, tokens, schedule_tokens):
    _, batch, lr, micro = TRIALS[trial]
    unit = batch * 1024
    steps = tokens // unit
    horizon = (schedule_tokens + unit - 1) // unit
    if steps < 1 or schedule_tokens < tokens:
        raise ValueError('Budget must fit one update; schedule budget must be >= run budget')
    cadence = max(1, math.ceil(steps / 30))
    flags = dict(train_dataset_batch_size=batch, n_micro=micro,
        total_steps=steps, schedule_steps=horizon,
        dtype='fp32', param_dtype='fp32', mesh_dim='16,1,1', seed=42,
        inner_loop_iter=50, inner_loop_lr=lr, lr_sched='global_cosine',
        global_warmup=0, inner_loop_warmup=0, end_lr=0,
        inner_b1=.9, inner_b2=.999, inner_clip_gradient=1,
        inner_loop_wd=.01, optimizer_wd=.001,
        gauss_newton=True, single_batch_inner=True, reset_start=False,
        adaptive_inner_loop=False, ls_eval_batches=1, linesearch=True,
        armijo_linesearch=True, armijo_beta=.5, armijo_init_step=1,
        normalize_step=False, weight_average=False,
        log_freq=cadence, eval_freq=cadence, eval_steps=8,
        log_inner_steps=False, condition_log=False, log_gn_quadratic=False,
        log_update_balance=False, save_model_freq=0, save_milestone_freq=0,
        train_batch_growth_interval=0, fixed_step_size=0,
        notes=f'GN_Adam_trial{trial}_B{batch}_lr{lr:g}_K50_wd0.001_T{steps*unit}_H{schedule_tokens}')
    flags['checkpointer.save_optimizer_state'] = False
    return make_plan('gn_adam', [f'--{k}={v}' for k, v in flags.items()])


def read_record(controller, worker, run_id):
    path = str(Path(controller.cfg['state']) / run_id / 'manifest.json')
    code = 'import json; print("QUEUE_RECORD="+json.dumps(json.load(open(' + repr(path) + '))))'
    result = controller.ssh(worker, shlex.join(['python3', '-c', code]))
    lines = [s[len('QUEUE_RECORD='):] for s in result.stdout.splitlines()
             if s.startswith('QUEUE_RECORD=')]
    if result.returncode or len(lines) != 1:
        raise RuntimeError(f'Worker {worker}: cannot read manifest: {result.stderr}')
    return json.loads(lines[0])


def verdict(records, run_id):
    if not records:
        raise RuntimeError('No worker records')
    for worker, record in records.items():
        if record.get('run_id') != run_id:
            raise RuntimeError(f'Worker {worker}: wrong run identity')
        state = record.get('state')
        if state not in ('started', 'prepared', 'exited', 'cancelled'):
            raise RuntimeError(f'Worker {worker}: unexpected state {state}')
        if state == 'cancelled' or (state == 'exited' and record.get('exit_status') != 0):
            raise RuntimeError(f'Worker {worker}: unsuccessful training: {record}')
    return all(r.get('state') == 'exited' and r.get('exit_status') == 0
               for r in records.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['preview', 'run'])
    p.add_argument('--target', choices=['1', '2'], default='1')
    p.add_argument('--trials', default=','.join(str(t[0]) for t in TRIALS))
    p.add_argument('--tokens', type=int, default=299000000, help='Solve-token cap per trial')
    p.add_argument('--schedule-tokens', type=int, default=1000000000, help='Solve-token cosine horizon')
    p.add_argument('--poll', type=int, default=60)
    p.add_argument('--max-hours', type=float, default=72, help='Timeout per trial; stops queue, leaves training for inspection')
    p.add_argument('--json', action='store_true', help='Print complete resolved plans in preview')
    a = p.parse_args()
    ids = [int(x) for x in a.trials.split(',')]
    if not ids or len(set(ids)) != len(ids) or any(i not in range(len(TRIALS)) for i in ids):
        p.error('Choose unique trial IDs 0 through 9')
    if a.poll < 5 or a.max_hours <= 0:
        p.error('Invalid polling interval or timeout')
    plans = [(i, build_plan(i, a.tokens, a.schedule_tokens)) for i in ids]
    if a.action == 'preview':
        if a.json:
            print(json.dumps(plans, indent=2))
        else:
            print('trial batch lr micro updates schedule_updates solve_tokens log/eval_every')
            for i, plan in plans:
                f = plan['flags']; b = int(f['train_dataset_batch_size'])
                print(i, b, f['inner_loop_lr'], f['n_micro'], f['total_steps'],
                      f['schedule_steps'], int(f['total_steps'])*b*1024, f['log_freq'])
            print(f'{len(plans)} trials; <= {a.tokens:,} solve tokens each; '
                  f'{a.schedule_tokens:,} solve-token schedule; target {a.target}')
        return
    state = Path.home() / 'adam-queues'
    state.mkdir(exist_ok=True)
    # Same queue lock convention as the existing Adam controllers.
    with (state / f'target{a.target}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = state / plans[0][1]['run_id']
        directory.mkdir()
        (directory / 'plans.json').write_text(json.dumps(plans, indent=2))
        print(f'QUEUE DIRECTORY: {directory}', flush=True)
        print('Create STOP here to finish the current trial and pause. '
              'Interrupting this controller leaves training running. '
              'Failed/cancelled trials halt the queue.', flush=True)
        controller = Controller(a.target)
        for trial, plan in plans:
            if (directory / 'STOP').exists():
                print('STOP requested; no further trials launched.', flush=True)
                return
            print(f'START trial={trial} run={plan["run_id"]}', flush=True)
            controller.operate('run', plan)
            deadline = time.monotonic() + a.max_hours * 3600
            read_failures = 0
            while True:
                try:
                    records = {w: read_record(controller, w, plan['run_id']) for w in controller.workers}
                except (RuntimeError, ValueError) as error:
                    read_failures += 1
                    if read_failures >= 3:
                        raise
                    print(f'Manifest read retry {read_failures}/3: {error}', flush=True)
                else:
                    read_failures = 0
                    if verdict(records, plan['run_id']):
                        break
                if time.monotonic() > deadline:
                    raise RuntimeError('Trial timeout; inspect workers before continuing')
                time.sleep(a.poll)
            with (directory / 'completed.jsonl').open('a') as out:
                out.write(json.dumps(dict(trial=trial, run_id=plan['run_id'], records=records))+'\n')
            print(f'COMPLETE trial={trial} run={plan["run_id"]}', flush=True)
            time.sleep(10)
        print('ALL TRIALS COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(f'QUEUE STOPPED: {error}. Existing training may still be active; '
              'inspect tpu.sh status/logs.', file=sys.stderr, flush=True)
        sys.exit(1)
