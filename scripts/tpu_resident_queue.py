#!/usr/bin/env python3
"""TPU-local synchronized ordered sweep: one process per managed TPU worker."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(plan):
    return hashlib.sha256(canonical({k: v for k, v in plan.items()
                                    if k != 'plan_sha256'}).encode()).hexdigest()


def positive_int(x, label):
    if type(x) is not int or x <= 0:
        raise ValueError(label + ' must be a positive integer')


def parse_adam_spec(spec):
    """Strict AdamW queue schema; GN queue schema remains unchanged."""
    required = {'preset', 'default_solve_tokens', 'schedule_tokens',
                'target_loss', 'trials'}
    optional = {'protocol'}
    if not required.issubset(spec) or set(spec) - required - optional:
        raise ValueError('Invalid Adam queue configuration fields')
    protocol = spec.get('protocol')
    if protocol not in (None, 'adam_followup_600m'):
        raise ValueError('Unsupported Adam evaluation/schedule protocol')
    positive_int(spec['default_solve_tokens'], 'default_solve_tokens')
    positive_int(spec['schedule_tokens'], 'schedule_tokens')
    target = spec['target_loss']
    if type(target) not in (int, float) or not math.isfinite(target) or target <= 0:
        raise ValueError('target_loss must be finite and positive')
    if not isinstance(spec['trials'], list) or not spec['trials']:
        raise ValueError('Missing ordered Adam trials')
    names, configurations = set(), set()
    for item in spec['trials']:
        required_trial = {'name', 'batch', 'lr', 'b2', 'weight_decay'}
        if not isinstance(item, dict) or set(item) not in (
                required_trial, required_trial | {'solve_tokens'}):
            raise ValueError('Invalid Adam trial fields')
        name = item['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', name):
            raise ValueError('Unsafe trial name')
        batch = item['batch']
        positive_int(batch, 'batch')
        if batch % 16:
            raise ValueError('Batch must be divisible by 16-way data parallel mesh')
        lr, b2, wd = item['lr'], item['b2'], item['weight_decay']
        if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
            raise ValueError('lr must be finite and positive')
        if type(b2) not in (int, float) or not math.isfinite(b2) or not 0 < b2 < 1:
            raise ValueError('b2 must be finite and in (0, 1)')
        if type(wd) not in (int, float) or not math.isfinite(wd) or wd < 0:
            raise ValueError('weight_decay must be finite and nonnegative')
        budget = item.get('solve_tokens', spec['default_solve_tokens'])
        positive_int(budget, 'solve_tokens')
        if budget < batch * 1024:
            raise ValueError('Invalid Adam token budget')
        if budget > spec['schedule_tokens'] and protocol != 'adam_followup_600m':
            raise ValueError('Adam solve budget exceeds schedule horizon without explicit legacy protocol')
        signature = (batch, float(lr), float(b2), float(wd), budget)
        if name in names or signature in configurations:
            raise ValueError('Duplicate Adam trial or name')
        names.add(name)
        configurations.add(signature)
    return spec


def build_adam_plan(trial_index, item, tokens, schedule_tokens, target_loss, *, protocol=None):
    """One AdamW run. The named legacy protocol preserves old follow-up settings."""
    from run_experiment import make_plan
    batch = item['batch']
    lr, b2, wd = item['lr'], item['b2'], item['weight_decay']
    unit = batch * 1024
    steps = tokens // unit
    if protocol == 'adam_followup_600m':
        # Original adam_followup trial (B128): 4577 decay steps, 46 warmup
        # steps, 8 validation batches, every 1144 steps, no initial eval.
        # A longer token cap is allowed; Optax holds end_lr after decay.
        horizon = schedule_tokens // unit
        warmup = max(1, round(.01 * horizon))
        cadence = max(1, min(steps, horizon // 4))
        eval_steps = 8
        log_initial_eval = False
    elif protocol is None:
        # Preserve the existing, non-comparable 2B-horizon queue semantics.
        horizon = (schedule_tokens + unit - 1) // unit
        warmup = max(1, math.ceil(.01 * horizon))
        cadence = max(1, min(steps, math.ceil(20_000_000 / unit)))
        eval_steps = 50
        log_initial_eval = True
    else:
        raise ValueError('Unsupported Adam evaluation/schedule protocol')
    flags = {
        'mesh_dim': '16,1,1', 'dtype': 'fp32', 'param_dtype': 'fp32',
        'seed': 42, 'train_dataset_batch_size': batch,
        'total_steps': steps, 'target_loss': target_loss,
        'log_freq': cadence, 'eval_freq': cadence, 'eval_steps': eval_steps,
        'log_initial_eval': log_initial_eval,
        'optimizer.type': 'adamw',
        'optimizer.accumulate_gradient_steps': 1,
        'optimizer.adamw_optimizer.lr': lr,
        'optimizer.adamw_optimizer.b1': .9,
        'optimizer.adamw_optimizer.b2': b2,
        'optimizer.adamw_optimizer.lr_sched': 'cosine',
        'optimizer.adamw_optimizer.init_lr': lr * .1,
        'optimizer.adamw_optimizer.end_lr': lr * .1,
        'optimizer.adamw_optimizer.lr_warmup_steps': warmup,
        'optimizer.adamw_optimizer.lr_decay_steps': horizon,
        'optimizer.adamw_optimizer.weight_decay': wd,
        'optimizer.adamw_optimizer.clip_gradient': 1,
        'optimizer.adamw_optimizer.bf16_momentum': False,
        'optimizer.adamw_optimizer.multiply_by_parameter_scale': False,
        'weight_average': False,
        'save_model_freq': 0, 'save_milestone_freq': 0,
        'checkpointer.save_optimizer_state': False,
        'notes': f'Adam_B{batch}_lr{lr:g}_b2{b2:g}_wd{wd:g}',
    }
    return make_plan('adam', [f'--{key}={value}' for key, value in flags.items()])


def parse_spec(spec):
    if isinstance(spec, dict) and spec.get('preset') == 'adam':
        return parse_adam_spec(spec)
    if not isinstance(spec, dict) or set(spec) != {
            'preset', 'default_solve_tokens', 'schedule_tokens', 'trials'}:
        raise ValueError('Invalid queue configuration fields')
    if spec['preset'] != 'gn_adam':
        raise ValueError('Only gn_adam queue preset supported')
    positive_int(spec['default_solve_tokens'], 'default_solve_tokens')
    positive_int(spec['schedule_tokens'], 'schedule_tokens')
    if not isinstance(spec['trials'], list) or not spec['trials']:
        raise ValueError('Missing ordered trials')
    names, configurations = set(), set()
    for item in spec['trials']:
        if not isinstance(item, dict) or set(item) not in (
                {'name', 'batch', 'inner_lr', 'n_micro'},
                {'name', 'batch', 'inner_lr', 'n_micro', 'solve_tokens'},
                {'name', 'batch', 'inner_lr', 'n_micro', 'inner_loop_iter'},
                {'name', 'batch', 'inner_lr', 'n_micro', 'solve_tokens', 'inner_loop_iter'}):
            raise ValueError('Invalid trial fields')
        name = item['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', name):
            raise ValueError('Unsafe trial name')
        batch, micro = item['batch'], item['n_micro']
        positive_int(batch, 'batch')
        positive_int(micro, 'n_micro')
        positive_int(item.get('inner_loop_iter', 50), 'inner_loop_iter')
        lr = item['inner_lr']
        if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
            raise ValueError('inner_lr must be finite and positive')
        budget = item.get('solve_tokens', spec['default_solve_tokens'])
        positive_int(budget, 'solve_tokens')
        if budget < batch * 1024 or budget > spec['schedule_tokens']:
            raise ValueError('Invalid solve/schedule token budget')
        if batch % 16 or micro > batch // 16:
            raise ValueError('Batch and microbatch must fit the 16-device data-parallel mesh')
        signature = (batch, float(lr), micro, budget, item.get('inner_loop_iter', 50))
        if name in names or signature in configurations:
            raise ValueError('Duplicate trial or trial name')
        names.add(name)
        configurations.add(signature)
    return spec


def make_queue_plan(spec, *, queue_id=None):
    """Controller-side only; resolve each trial before starting any TPU workers."""
    parse_spec(spec)
    queue_id = queue_id or (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                            + '-' + uuid4().hex)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', queue_id):
        raise ValueError('Invalid queue identity')
    trials = []
    for i, item in enumerate(spec['trials'], 1):
        budget = item.get('solve_tokens', spec['default_solve_tokens'])
        if spec['preset'] == 'gn_adam':
            from gn_sweep_queue import build_plan_for_config
            p = build_plan_for_config(i, item['batch'], item['inner_lr'],
                                      item['n_micro'], budget, spec['schedule_tokens'],
                                      inner_loop_iter=item.get('inner_loop_iter', 50))
        else:
            p = build_adam_plan(i, item, budget, spec['schedule_tokens'],
                                spec['target_loss'], protocol=spec.get('protocol'))
        trials.append(dict(index=i, name=item['name'], plan=p,
                           requested_solve_tokens=budget,
                           actual_solve_tokens=int(p['flags']['total_steps']) * item['batch'] * 1024))
    return dict(route='queue', preset=spec['preset'], run_id=queue_id, flags={},
                explicit_overrides=[], workers=[], plan_sha256='', trials=trials,
                queue_root='gs://gfn-soo/checkpoints/queue_runs/' + queue_id)


def finish_plan(plan, workers):
    if plan['route'] != 'queue' or not workers or len(workers) != len(set(workers)) or any(
            type(w) is not int or w < 0 for w in workers):
        raise ValueError('Invalid discovered worker IDs')
    plan['workers'] = sorted(workers)
    plan['plan_sha256'] = digest(plan)
    return plan


def verify_plan(plan, worker):
    if plan.get('route') != 'queue' or plan.get('plan_sha256') != digest(plan):
        raise ValueError('Queue plan hash mismatch')
    workers = plan.get('workers', [])
    if not isinstance(workers, list) or not workers or worker not in workers or any(
            type(w) is not int for w in workers):
        raise ValueError('Unexpected queue worker')
    for i, item in enumerate(plan['trials'], 1):
        if item['index'] != i or not re.fullmatch(r'[A-Za-z0-9_-]+', item['plan']['run_id']):
            raise ValueError('Invalid trial identity')
    return plan


def record(plan, worker, index, run_id, status=None):
    return dict(queue_id=plan['run_id'], plan_sha256=plan['plan_sha256'],
                worker_id=worker, trial_index=index, run_id=run_id, exit_status=status)


class GCSStore:
    def __init__(self, root):
        if not root.startswith('gs://') or '/' not in root[5:]:
            raise ValueError('Invalid queue GCS path')
        from google.cloud import storage  # Deliberately TPU-runtime only
        bucket, prefix = root[5:].split('/', 1)
        self.bucket = storage.Client().bucket(bucket)
        self.prefix = prefix.rstrip('/') + '/'

    def create(self, name, data):
        if not re.fullmatch(r'[A-Za-z0-9_./-]+', name):
            raise ValueError('Unsafe marker key')
        self.bucket.blob(self.prefix + name).upload_from_string(
            canonical(data), content_type='application/json', if_generation_match=0)

    def read(self, name):
        obj = self.bucket.get_blob(self.prefix + name)
        return None if obj is None else json.loads(obj.download_as_text())


def barrier(store, plan, stage, index, run_id, timeout_s, *, clock=time.monotonic,
            sleep=time.sleep, cancelled=lambda: False, poll_s=10):
    """Never advance until all expected workers show matching ready/results."""
    if stage not in ('ready', 'result'):
        raise ValueError('Invalid barrier')
    deadline = clock() + timeout_s
    expected = {w: record(plan, w, index, run_id,
                          None if stage == 'ready' else 0) for w in plan['workers']}
    while True:
        if cancelled():
            raise InterruptedError('Queue cancelled')
        seen = set()
        for w in plan['workers']:
            failed = store.read(f'fail/{w}.json')
            if failed is not None:
                baseline = record(plan, w, failed.get('trial_index'), failed.get('run_id'),
                                  failed.get('exit_status'))
                if failed != baseline or failed['trial_index'] > index:
                    raise RuntimeError('Invalid failure marker')
                raise RuntimeError(f'Queue worker {w} failed at trial {failed["trial_index"]}')
            key = f'ready/{w}.json' if stage == 'ready' else f'result/{index}/{w}.json'
            value = store.read(key)
            if value is None:
                continue
            if value != expected[w]:
                raise RuntimeError(f'Invalid {stage} record from worker {w}: {value}')
            seen.add(w)
        if len(seen) == len(expected):
            return
        if clock() >= deadline:
            raise TimeoutError(f'{stage} barrier {index} timeout; missing: '
                               f'{sorted(set(expected) - seen)}')
        sleep(min(poll_s, max(0, deadline - clock())))


class ResidentRunner:
    def __init__(self, plan, worker, store=None, *, clock=time.monotonic, sleep=time.sleep):
        self.plan = verify_plan(plan, worker)
        self.worker = worker
        self.store = store if store is not None else GCSStore(plan['queue_root'])
        self.clock, self.sleep = clock, sleep
        self.cancelled = False
        self.child = None

    def terminate(self, sig, frame):
        self.cancelled = True
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()

    def fail(self, index, run_id):
        try:
            self.store.create(f'fail/{self.worker}.json',
                              record(self.plan, self.worker, index, run_id, 1))
        except Exception as error:
            print('Cannot publish failure marker:', error, flush=True)

    def check_peer_failure(self, index):
        for w in self.plan['workers']:
            failed = self.store.read(f'fail/{w}.json')
            if failed is not None:
                if failed.get('queue_id') != self.plan['run_id'] or failed.get('plan_sha256') != self.plan['plan_sha256']:
                    raise RuntimeError('Mismatched worker failure marker')
                raise RuntimeError(f'Worker {w} failed during trial {index}')

    def train(self, item, directory, *, hours=72):
        idx = item['index']
        path = directory / f'trial_{idx:03}.json'
        path.write_text(canonical(item['plan']) + '\n')
        env = dict(os.environ)
        env.pop('SLURM_ARRAY_TASK_ID', None)
        env.pop('SLURM_ARRAY_JOB_ID', None)
        if self.worker == 0:
            env.pop('WANDB_MODE', None)
        else:
            env['WANDB_MODE'] = 'disabled'
        if self.cancelled:
            raise InterruptedError('Queue cancelled before trial launch')
        print(f'START trial={idx} {item["name"]} run={item["plan"]["run_id"]}', flush=True)
        self.child = subprocess.Popen([sys.executable, str(Path(__file__).with_name('run_experiment.py')),
                                       '--execute', str(path)], env=env)
        deadline = self.clock() + hours * 3600
        try:
            while self.child.poll() is None:
                if self.cancelled or self.clock() >= deadline:
                    self.child.terminate()
                    try:
                        self.child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.child.kill()
                        self.child.wait()
                    raise RuntimeError(f'Trial {idx} cancelled or timed out')
                self.check_peer_failure(idx)
                self.sleep(10)
            rc = self.child.wait()
            if rc:
                raise RuntimeError(f'Trial {idx} exited with status {rc}')
        except BaseException:
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait()
            raise
        finally:
            self.child = None

    def run(self, directory, *, startup_s=1800, barrier_s=1800, trial_hours=72,
            install_signals=True):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        old_term, old_int = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
        if install_signals:
            signal.signal(signal.SIGTERM, self.terminate)
            signal.signal(signal.SIGINT, self.terminate)
        idx, run_id = 0, self.plan['run_id']
        try:
            ready = record(self.plan, self.worker, 0, run_id)
            self.store.create(f'ready/{self.worker}.json', ready)
            if self.store.read(f'ready/{self.worker}.json') != ready:
                raise RuntimeError('GCS read-after-write mismatch')
            barrier(self.store, self.plan, 'ready', 0, run_id, startup_s,
                    clock=self.clock, sleep=self.sleep, cancelled=lambda: self.cancelled)
            for item in self.plan['trials']:
                idx, run_id = item['index'], item['plan']['run_id']
                if self.cancelled:
                    raise InterruptedError('Queue cancelled before trial')
                self.train(item, directory, hours=trial_hours)
                self.store.create(f'result/{idx}/{self.worker}.json',
                                  record(self.plan, self.worker, idx, run_id, 0))
                print(f'LOCAL COMPLETE trial={idx}', flush=True)
                barrier(self.store, self.plan, 'result', idx, run_id, barrier_s,
                        clock=self.clock, sleep=self.sleep, cancelled=lambda: self.cancelled)
                print(f'ALL WORKERS COMPLETE trial={idx}', flush=True)
            print('ALL TRIALS COMPLETE', flush=True)
        except BaseException as error:
            self.fail(idx, run_id)
            print(f'QUEUE FAILED worker={self.worker} trial={idx}: {error}', flush=True)
            raise
        finally:
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait()
            if install_signals:
                signal.signal(signal.SIGTERM, old_term)
                signal.signal(signal.SIGINT, old_int)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', required=True)
    p.add_argument('--worker', required=True, type=int)
    a = p.parse_args()
    path = Path(a.execute)
    ResidentRunner(json.loads(path.read_text()), a.worker).run(path.parent)


if __name__ == '__main__':
    main()
